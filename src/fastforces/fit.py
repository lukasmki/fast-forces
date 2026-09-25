"""Fitting the bonded parameters to a reference calculator.

The whole fit rests on one observation: every bonded term in `QForce` except
the bond is a force constant `k` multiplying a function of the geometry alone.
So with the equilibrium values fixed, the energy of a frame is

    E = sum_c  k_c * g_c(R)  +  E0

which is *linear* in the parameters being fit, and the forces are linear in them
too.  Recovering every force constant is therefore one bounded least-squares
solve rather than a nonlinear optimization.

The basis functions `g_c` are not re-derived here.  They are obtained by calling
DynamicTopology's own `QForce.compute_*` methods with `k = 1`, which returns
exactly `(g_c, -dg_c/dR)`.  That keeps the fit and the simulation the same
code: a term cannot be fit under one functional form and evaluated under
another.  The nonbonded baseline goes through `DynamicTopology.forcefield.
evaluate` for the same reason.

`E0` is not in that solve, and is not in any solve.  Every energy residual in
the fit is mean-centered, so a constant shift of the whole profile costs nothing
anywhere; `E0` is read off the leftover mean once the fit has converged.  That is
not a convenience.  The Morse form carries a constant `-D` per bond, and with
`a = sqrt(k / 2D)` its well is `k*dr**2/2` to second order, so near equilibrium
`D` reaches the energy through that constant and almost nothing else.  Fitting
`D` and `E0` together therefore puts a flat direction through the parameter
space, and the two blocks walk along it: `D` up, `E0` up to compensate, for
hundreds of cycles, with the residual barely moving and no fixed point in
reach.  Centering removes the direction instead of penalizing it, which leaves
`D` determined by the anharmonicity it actually describes and `E0` determined by
one mean at the end.

Bonds are the exception to the linearity, and are fit in a separate nonlinear
block.  Two reasons, and both are forced:

  * The Morse exponent is `sqrt(k / 2D)`, so a Morse bond is not linear in `k`.
    Its per-bond asymptote `h` is not *fitted* to the training frames, which
    cannot see where the stretched branch ends up, but it is *solved*: the
    training set carries the two fragments of each bond class
    (`sampling.fragment_frames`), and the stretched well is set so that
    pulling the bond apart ends `bond_asymptote` above them (`_asymptotes`).
    Left at `bond_asymptote` instead, `h` put the limit wherever the fitted `D`
    happened to fall -- 0.28 eV *below* two H atoms for H2 at B3LYP, whose
    bonded diabat then never crossed theirs.  A training set without fragment
    frames still reads `bond_asymptote`.  `refine.fit_force_constants` refits
    `h` against the reaction barriers on top of this.
  * `r0` cannot be read off the equilibrium geometry.  It is an effective
    parameter that balances whatever nonbonded baseline survives on the atoms
    the bond connects, not a measurement of the bond length -- so it is fit,
    inside bounds taken from the measured geometry, rather than fixed at it.
    How much balancing there is to do depends on the exclusions: within a
    molecule small enough that every pair is inside `exclusion_depth` there is
    none left and `r0` lands near the true length, while a pair that survives
    the exclusion still carries ~20 eV/A of ZBL against a reference force of
    zero and has to be pre-compressed against it.  The example force field is
    built the second way -- its O-H `r0` is 0.70 A against a true 0.97 A, while
    the `bondangle` cross term that references the same bond keeps the true
    value.

So the fit alternates: solve every other force constant linearly with the bond
contribution held fixed, then refine the handful of bond parameters nonlinearly
with everything else held fixed.  Both blocks decrease the same residual, and a
few cycles converge.

A fit can be started from an existing force field rather than from the element
table, by passing `initial=`.  What that buys differs by block, and the split
follows directly from the paragraphs above:

  * The bond parameters are the starting point of a *local* nonlinear solve, so
    supplying them changes where the fit ends up, not just how fast it gets
    there.
  * The equilibrium values and the nonbonded baseline are held fixed, not fit,
    so supplying them replaces what would otherwise be measured off the
    geometry or read from the element table.
  * The remaining force constants are recovered by a bounded *global* least
    squares that has no starting point at all.  Supplying them reaches the
    result only through the one bond refinement that runs before the first
    linear solve -- which is why that refinement exists.
"""

from dataclasses import asdict, dataclass, field
from os import PathLike

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from ase import Atoms
from DynamicTopology.forcefield.evaluate import evaluate_term_dict, term_dict
from DynamicTopology.forcefield.exclusions import exclusion_terms
from DynamicTopology.forcefield.params import active
from DynamicTopology.forcefield.qforce import QForce

from . import elements
from .charges import class_average
from .params import ELECTROSTATIC_TERMS, Parameters
from .topology import Topology

# Which kwargs of each term are read off the equilibrium geometry, and from
# which atom slots.  Everything not listed is either the fitted `k` or comes
# from `Topology.fixed` (`n`, `phi0`).
GEOMETRIC: dict[str, dict[str, tuple[str, tuple[int, ...]]]] = {
    "bond": {"r0": ("distance", (0, 1))},
    "angle": {"theta0": ("angle", (0, 1, 2))},
    "bondbond": {"r1_0": ("distance", (0, 1)), "r2_0": ("distance", (2, 3))},
    "bondangle": {"theta0": ("angle", (0, 1, 2)), "r0": ("distance", (3, 4))},
    "angleangle": {
        "theta1_0": ("angle", (0, 1, 2)),
        "theta2_0": ("angle", (3, 4, 5)),
    },
    "dihedralangle": {"theta0": ("angle", (4, 5, 6))},
    "dihedralbond": {"r0": ("distance", (4, 5))},
    "dihedralangleangle": {
        "theta0_1": ("angle", (0, 1, 2)),
        "theta0_2": ("angle", (1, 2, 3)),
    },
    "periodicdihedral": {},
}

# Force constants the linear block must keep positive: a negative angle `k`
# inverts the well.  The cross terms are genuinely signed and are left free.
# Bonds are not here because they are not in the linear block at all; their
# positivity is enforced by the bounds in `_refine_bonds`.
POSITIVE_K = ("angle",)

# The bond parameters the nonlinear block refines, in the order they are packed
# into its parameter vector.  The stretched well `Dw` (so `h = Dw - D`) is not
# refined: `_asymptotes` solves it from the fragments and it is held fixed here.
_BOND_NAMES: tuple[str, ...] = ("r0", "k", "D")


@dataclass
class FitConfig:
    """Everything that determines the fit, so it can be written to the file."""

    temperature: float = 500.0
    n_mode_frames: int = 40
    n_conformers: int = 20
    torsion_step_deg: float = 15.0
    hessian_delta: float = 0.01
    fmax: float = 1e-3
    energy_weight: float | None = None  # None: balance against the force block
    regularization: float = 1e-3
    seed: int = 0
    bond_form: str = "morse"
    # Which electrostatic term the fitted field carries.  `"acks2"` is the
    # `atom` block, with the charges re-solved at every geometry from element
    # defaults; `"fixed"` is the `charge` block, with one charge per atom taken
    # from the reference calculation's Mulliken populations, evaluated under
    # DynamicTopology's `global_params.electrostatics = "pointcharge"`.  Neither is fitted
    # -- both are part of the baseline the bonded terms are fit against -- so
    # this changes what that baseline is, and a field refit with the other
    # setting is a different force field rather than a reparametrized one.
    electrostatics: str = "acks2"
    # A cap, not a target: the alternation exits on `cycle_tol` well inside it
    # (H2O2 takes 21 to 52 cycles depending on the training set).  The budget is
    # generous because a fit that stops early is no longer idempotent, and
    # because `E0` is derived from a converged residual -- both properties are
    # promises this number has to keep.  It was 25 when `E0` was fitted, which
    # was not enough for either: convergence took 620 cycles then.
    n_cycles: int = 200
    cycle_tol: float = 1e-4

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def _distance(pos, idx):
    return np.linalg.norm(pos[idx[:, 0]] - pos[idx[:, 1]], axis=-1)


def _angle(pos, idx):
    a = pos[idx[:, 0]] - pos[idx[:, 1]]
    b = pos[idx[:, 2]] - pos[idx[:, 1]]
    cos = np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.arccos(np.clip(cos, -1.0, 1.0))


_MEASURE = {"distance": _distance, "angle": _angle}


def equilibrium_values(topology: Topology, positions: np.ndarray) -> dict[str, dict]:
    """Per-term equilibrium values, averaged within each parameter class.

    Averaging over the class rather than taking each term's own value is what
    makes the sharing real: two symmetry-equivalent bonds get one `r0`, so they
    stay equivalent even when the reference geometry is slightly asymmetric.
    """
    out: dict[str, dict] = {}
    for term, spec in GEOMETRIC.items():
        if term not in topology.atoms:
            continue
        atoms = topology.atoms[term]
        classes = topology.classes[term]
        values = {}
        for name, (kind, slots) in spec.items():
            measured = _MEASURE[kind](positions, atoms[:, list(slots)])
            averaged = np.zeros_like(measured)
            for c in range(topology.n_classes(term)):
                mask = classes == c
                if np.any(mask):
                    averaged[mask] = measured[mask].mean()
            values[name] = averaged
        out[term] = values
    return out


# ---------------------------------------------------------------------------
# initial parameters
# ---------------------------------------------------------------------------


def as_parameters(initial) -> Parameters | None:
    """Coerce a caller-supplied starting force field into `Parameters`.

    Accepts a `Parameters`, a bare `term_dict`, or a path to a DynamicTopology
    jsonl file -- the same file `Parameters.to_jsonl` writes, so one fit's
    output can be handed straight back in as the next fit's starting point
    without the caller doing any conversion.
    """
    if initial is None or isinstance(initial, Parameters):
        return initial
    if isinstance(initial, (str, PathLike)):
        return Parameters.from_jsonl(str(initial))
    if isinstance(initial, dict):
        return Parameters(numbers=np.zeros(0, dtype=int), terms=initial)
    raise TypeError(
        f"initial parameters must be Parameters, a term dict, or a jsonl path, "
        f"not {type(initial).__name__}"
    )


@dataclass
class _Seed:
    """An initial force field mapped onto the classes of a topology.

    Terms are matched by their atom slots, so an initial field only contributes
    where it describes the same term over the same atoms.  Anything it does not
    cover is left as `nan` here and falls back to the ordinary default -- the
    geometry for an equilibrium value, the element table for a bond.
    """

    values: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    bond: dict[str, np.ndarray] = field(default_factory=dict)
    k: dict[tuple[str, int], float] = field(default_factory=dict)
    nonbonded: dict[str, dict] = field(default_factory=dict)
    n_seeded: int = 0


def _row_key(atoms, n=None) -> tuple:
    """The identity of one term row: its atom slots, plus `n` for a series.

    Dihedral terms put several rows on the same four atoms, one per periodicity,
    so the atom slots alone do not name a row there.
    """
    key = tuple(int(a) for a in atoms)
    return key if n is None else key + (int(round(float(n))),)


def _rows_by_atoms(block: dict) -> dict[tuple, dict[str, float]]:
    """Index one term block of an existing force field by row identity."""
    atoms = np.asarray(block["atoms"], dtype=int)
    kwargs = block["kwargs"]
    series = kwargs.get("n")
    return {
        _row_key(atoms[i], None if series is None else series[i]): {
            name: float(value[i]) for name, value in kwargs.items()
        }
        for i in range(len(atoms))
    }


def _per_class(matched: list, classes: np.ndarray, n_classes: int) -> dict:
    """Average each matched parameter within its class; `nan` where unmatched.

    Averaging rather than taking the first row is the same rule
    `equilibrium_values` uses, and for the same reason: it keeps the sharing
    real even if the supplied field happens to break it.
    """
    names: set[str] = set()
    for row in matched:
        if row is not None:
            names |= set(row)
    names -= {"n", "phi0"}

    out = {}
    for name in sorted(names):
        values = np.full(n_classes, np.nan)
        for c in range(n_classes):
            found = [
                row[name]
                for row, cls in zip(matched, classes, strict=True)
                if row is not None and cls == c and name in row
            ]
            if found:
                values[c] = float(np.mean(found))
        out[name] = values
    return out


def _seed_from(topology: Topology, initial: Parameters, n_atoms: int) -> _Seed:
    """Read `initial` onto `topology`, matching terms by their atom slots."""
    seed = _Seed()
    for term, block in initial.terms.items():
        atoms = np.asarray(block["atoms"], dtype=int)
        if len(atoms) and int(atoms.max()) >= n_atoms:
            raise ValueError(
                f"initial `{term}` parameters index atom {int(atoms.max())}, but "
                f"this training set has {n_atoms} atoms -- different molecule?"
            )
        if term in ("atom", "charge", "lennardjones"):
            seed.nonbonded[term] = block
            seed.n_seeded += len(atoms)
            continue
        if term == "reference":
            # A supplied `E0` is deliberately dropped.  It is not optimized, so
            # it is not something a fit can be started from -- `fit` derives it
            # from the converged residual instead.
            continue
        if term not in topology.atoms:
            continue

        rows = _rows_by_atoms(block)
        series = topology.fixed.get(term, {}).get("n")
        matched = [
            rows.get(_row_key(row, None if series is None else series[i]))
            for i, row in enumerate(topology.atoms[term])
        ]
        if not any(row is not None for row in matched):
            raise ValueError(
                f"none of the {len(rows)} initial `{term}` terms match this "
                f"topology's {len(matched)}; check that the initial parameters "
                "use the same atom ordering"
            )

        classes = topology.classes[term]
        per_class = _per_class(matched, classes, topology.n_classes(term))
        # Every parameter of a class is matched or not together, so any one of
        # them counts the classes this term contributed.
        any_name = sorted(per_class)[:1]
        seed.n_seeded += sum(int(np.sum(~np.isnan(per_class[n]))) for n in any_name)

        if term == "bond":
            # Bond `r0` seeds the nonlinear block, never the fixed equilibrium
            # values: those stay measured, because they set the bounds the
            # refinement runs under.
            seed.bond = {n: v for n, v in per_class.items() if n in _BOND_NAMES}
            # A supplied asymptote arrives as the stretched well it fixes
            # (`_asymptotes` says why that and not `h`), so a refit from a
            # fitted field starts at its fixed point instead of re-solving it.
            if "h" in per_class and "D" in per_class:
                seed.bond["Dw"] = per_class["D"] + per_class["h"]
            continue
        supplied = {
            name: per_class[name][classes]
            for name in GEOMETRIC.get(term, {})
            if name in per_class
        }
        if supplied:
            seed.values[term] = supplied
        for c, value in enumerate(per_class.get("k", ())):
            if not np.isnan(value):
                seed.k[(term, c)] = float(value)
    return seed


def _merge(default: np.ndarray, supplied: np.ndarray | None) -> np.ndarray:
    """`supplied` wherever it is not `nan`, `default` elsewhere."""
    if supplied is None:
        return default
    supplied = np.asarray(supplied, dtype=float)
    return np.where(np.isnan(supplied), default, supplied)


def _apply_nonbonded(params: Parameters, seed: _Seed) -> None:
    """Replace element-table nonbonded defaults with the supplied values.

    An electrostatic block in `initial` *replaces* whichever electrostatic block
    this fit was configured with, rather than joining it: the two are
    alternatives, a field carrying both is rejected by
    `Parameters.electrostatics`, and "start from this field" most plainly means
    "use its electrostatics too".  So an `initial` fitted with fixed charges
    switches a default-configured fit over to them, and vice versa.
    """
    n = len(params.numbers)
    for term, block in seed.nonbonded.items():
        if term in ELECTROSTATIC_TERMS:
            for other in ELECTROSTATIC_TERMS:
                if other != term:
                    params.terms.pop(other, None)
        slots = np.asarray(block["atoms"], dtype=int)[:, 0]
        kwargs = dict(params.terms.get(term, {}).get("kwargs", {}))
        for name, value in block["kwargs"].items():
            existing = np.asarray(kwargs.get(name, np.zeros(n)), dtype=float)
            existing = existing.copy() if len(existing) == n else np.zeros(n)
            existing[slots] = np.asarray(value, dtype=float)
            kwargs[name] = existing
        params.terms[term] = {
            "atoms": np.arange(n)[:, None],
            "kwargs": kwargs,
        }


# ---------------------------------------------------------------------------
# the linear system
# ---------------------------------------------------------------------------


def _pair_vectors(frame):
    pos = frame.get_positions()
    vecs = pos[:, None, :] - pos[None, :, :]
    if np.any(frame.pbc):
        cell = np.array(frame.cell)
        f = vecs @ np.linalg.inv(cell)
        vecs = vecs - (frame.pbc * np.floor(f + 0.5)) @ cell
    return vecs


def _basis_columns(topology, values, qforce, vecs, skip=("bond",)):
    """`(energy, forces)` per class, evaluated at `k = 1`.

    `skip` names the term types handled outside the linear block.
    """
    energies, forces, labels = [], [], []
    for term in sorted(topology.atoms):
        if term in skip:
            continue
        compute = getattr(qforce, f"compute_{term}", None)
        if compute is None:
            continue
        atoms = topology.atoms[term]
        classes = topology.classes[term]
        for c in range(topology.n_classes(term)):
            mask = classes == c
            kwargs = {name: value[mask] for name, value in values.get(term, {}).items()}
            kwargs.update(
                {
                    name: value[mask]
                    for name, value in topology.fixed.get(term, {}).items()
                }
            )
            kwargs["k"] = np.ones(int(mask.sum()))
            # `QForce` is in eV and Angstrom throughout, which is what the fit
            # works in, so the `compute_*` methods can be called directly with
            # `vecs` straight off the frame.  The third return is the virial,
            # which is fixed-cell here and unused.
            energy, force, _ = compute(vecs, atoms[mask], **kwargs)
            energies.append(energy)
            forces.append(force.reshape(-1))
            labels.append((term, c))
    return np.array(energies), np.array(forces), labels


def _charge_block(topology: Topology, equilibrium) -> dict:
    """The fixed-charge electrostatic block, read off the reference frame.

    The charges are the `mulliken` array that
    `calculators.pyscf.PySCFCalculator` and `calculators.tblite.TBLiteCalculator`
    write onto every frame they evaluate and `io.write_training_set` carries
    into the training file.  They are read
    from the *equilibrium* frame, the same frame every other fixed value in this
    fit is measured on, and they are not fit afterwards -- see
    `FitConfig.electrostatics`.

    They are then averaged within each atom equivalence class.  Mulliken
    charges come out of a single geometry, so the three hydrogens of a methyl
    group get three slightly different values, and freezing that asymmetry in
    would put a spurious electrostatic torsion on a rotor that has none.  Every
    other parameter in this fit is per-class for the same reason.  A class-wise
    mean also leaves the total charge exactly where it was, so the template
    carries its formal charge, as DynamicTopology's point charges need.
    """
    if "mulliken" not in equilibrium.arrays:
        raise ValueError(
            "fitting with `electrostatics='fixed'` needs per-atom charges, and "
            "this training set carries none: the equilibrium frame has no "
            "`mulliken` array.  It is written by "
            "`calculators.pyscf.PySCFCalculator` and "
            "`calculators.tblite.TBLiteCalculator` (a manifest's `tblite`), so "
            "a set sampled with plain `tblite.ase.TBLite` or an older version "
            "of either has to be re-sampled, or the fit has to run with "
            "`electrostatics='acks2'`"
        )
    raw = np.asarray(equilibrium.get_array("mulliken"), dtype=float)
    q = class_average(raw, topology.atom_classes)
    return {"atoms": np.arange(len(raw))[:, None], "kwargs": {"q": q}}


def _nonbonded(params, frames, graph):
    """Energies and forces of the fixed nonbonded baseline, per frame.

    DynamicTopology's own single-topology sum over the nonbonded blocks alone:
    the whole-system `ZBL`, switched 12-6 and electrostatics, with the
    1-2/1-3/1-4 part taken back off by the exclusion terms it derives from
    `graph` -- the topology being fitted, whose bond *parameters* do not exist
    yet.  What survives the exclusion at a bond length is part of the baseline
    the bonded fit has to absorb, which is the bargain `ZBL` and the switched
    12-6 both make.

    **It has to be exactly the sum the simulation evaluates.**  This is the
    number the bonded residuals are taken against, so an exclusion applied here
    and not there -- or the other way round -- lands in every fitted force
    constant as a silent offset.  Going through `evaluate` is what rules that
    out rather than merely checking it.

    The virial is dropped: the fit works at fixed cell.
    """
    numbers = np.asarray(params.numbers)
    terms = params.to_terms()
    terms = terms + exclusion_terms(terms, numbers, graph=graph)
    td = term_dict(Atoms(numbers=numbers), terms)

    energies, forces = [], []
    for frame in frames:
        result = evaluate_term_dict(frame, td)
        energies.append(result.energy)
        forces.append(result.forces.reshape(-1))
    return np.array(energies), np.array(forces)


def _bond_kwargs(shape: dict, classes: np.ndarray) -> dict:
    """Per-bond Morse kwargs from per-class `shape`.

    `shape` carries `(r0, k, D)` and, once `_asymptotes` has run, `Dw`: the
    stretched well depth, pinned by the fragments.  `QForce` takes the
    asymptote as `h = Dw - D`, so `h` is derived here rather than stored, and a
    class with no fragments (`Dw` nan) keeps `bond_asymptote`.
    """
    kwargs = {name: shape[name][classes] for name in _BOND_NAMES}
    if "Dw" in shape:
        dw = shape["Dw"][classes]
        kwargs["h"] = np.where(
            np.isnan(dw), active().bond_asymptote, dw - kwargs["D"]
        )
    return kwargs


def _bond_block(topology, qforce, all_vecs, shape):
    """Energy and forces of every bond, from per-class `(r0, k, D)` and maybe `Dw`."""
    classes = topology.classes["bond"]
    atoms = topology.atoms["bond"]
    kwargs = _bond_kwargs(shape, classes)
    energies, forces = [], []
    for vecs in all_vecs:
        energy, force, _ = qforce.compute_bond(vecs, atoms, **kwargs)
        energies.append(energy)
        forces.append(force.reshape(-1))
    return np.array(energies), np.array(forces)


def _centered(values: np.ndarray) -> np.ndarray:
    """`values` with their mean over frames removed.

    Every energy residual in the fit passes through here.  `E0` is not a fitted
    parameter -- it is read off the leftover mean once the fit has converged --
    so an energy offset is not the fit's to chase, and both blocks are made
    blind to it.  That is what keeps `D` and `E0` from trading: see the note on
    the alternation in `fit`.
    """
    return values - values.mean(axis=0)


def _solve_linear(a_energy, a_force, b_energy, b_force, labels, config, weight):
    """Bounded, regularized least squares for every non-bond force constant."""
    n_classes = len(labels)
    if n_classes == 0:
        return np.zeros(0)
    a_energy = _centered(a_energy)
    b_energy = _centered(b_energy)

    a = np.vstack([a_force, weight * a_energy])
    b = np.concatenate([b_force, weight * b_energy])

    # Tikhonov rows, scaled per column so the penalty does not depend on a
    # term's units.
    norms = np.linalg.norm(a, axis=0)
    norms[norms == 0] = 1.0
    penalty = config.regularization * norms
    a = np.vstack([a, np.diag(penalty)])
    b = np.concatenate([b, np.zeros(len(penalty))])

    lower = np.full(n_classes, -np.inf)
    for i, (term, _) in enumerate(labels):
        if term in POSITIVE_K:
            lower[i] = 0.0
    solution = lsq_linear(a, b, bounds=(lower, np.full(n_classes, np.inf)))
    return solution.x


def _refine_bonds(
    topology, qforce, all_vecs, shape, b_energy, b_force, weight, geometric_r0
):
    """Nonlinear refine of `(r0, k, D)` per bond class, everything else fixed.

    `r0` is bounded to a window around the geometric bond length rather than
    fixed at it: whatever nonbonded baseline survives the exclusions on a bonded
    pair has to be balanced somewhere, and this is where.  The window is wide
    downwards because that balancing can be large -- a pair outside
    `exclusion_depth` carries ~20 eV/A of ZBL -- and a floor at half the
    measured length, because a bond that collapses to nothing is a fit artifact
    and not a force field.
    """
    n = topology.n_classes("bond")

    # `Dw` rides along fixed: it is solved outside this block (`_asymptotes`).
    fixed = {name: value for name, value in shape.items() if name not in _BOND_NAMES}

    def unpack(x):
        refined = {name: x[i * n : (i + 1) * n] for i, name in enumerate(_BOND_NAMES)}
        return {**refined, **fixed}

    def residual(x):
        energies, forces = _bond_block(topology, qforce, all_vecs, unpack(x))
        # Centered, so the constant `-D` per bond that the Morse form carries is
        # invisible here.  `D` is then fixed by the anharmonicity it describes
        # and by nothing else, which is the whole point of the exercise.
        return np.concatenate(
            [
                (forces - b_force).reshape(-1),
                weight * _centered(energies - b_energy),
            ]
        )

    x0 = np.concatenate([shape[name] for name in _BOND_NAMES])
    # `r0` is allowed down to half the true bond length: fitted without
    # exclusions, an O-H bond compressed from 0.97 A to 0.70 A against the ZBL
    # it then had to balance, so a wide range is needed -- but a bond shorter
    # than that has stopped describing a bond.  `D` is capped well above any
    # real dissociation energy because it can be absorbing part of the
    # surviving repulsion, not just a bond.
    lower = np.concatenate(
        [
            0.5 * geometric_r0,
            np.full(n, 1.0),
            np.full(n, 0.25),
        ]
    )
    upper = np.concatenate(
        [
            geometric_r0 + 0.5,
            np.full(n, 20000.0),
            np.full(n, 200.0),
        ]
    )
    x0 = np.clip(x0, lower + 1e-9, upper - 1e-9)
    result = least_squares(
        residual, x0, bounds=(lower, upper), xtol=1e-10, max_nfev=400
    )
    return unpack(result.x)


# How far apart the two fragments are put to read the nonbonded energy the
# stretched limit keeps, in Angstrom.  Far enough that what the fragments still
# see of each other is the Coulomb tail of their partial charges, q_A q_B / r:
# 2 meV at +/-0.4 e.  Nothing else reaches it -- ZBL and the 12-6 are cut off
# long before, and the Morse exponent has underflowed to zero.
SEPARATION: float = 1000.0

# Cap on the rounds of `_asymptotes` and a refit under the result.  They end
# when `Dw` moves by less than `cycle_tol` of itself; with `Dw` held, what moves
# between rounds is `E0` and the minimum, which the training frames pin, so it
# takes two to four.
ASYMPTOTE_ROUNDS: int = 20


def _bond_class(topology, bond) -> int:
    pairs = [frozenset(map(int, b)) for b in topology.atoms["bond"]]
    return int(topology.classes["bond"][pairs.index(frozenset(map(int, bond)))])


def _asymptote_targets(training_set, topology, baseline) -> dict:
    """Per bond class, `(bond, moving atoms, target)` for its stretched limit.

    The limit is the molecule with one bond of the class pulled out to
    infinity, its two fragments held at their equilibrium geometry.  The
    target puts it `bond_asymptote` above the two fragments' reference energies
    -- the place DynamicTopology's defaults put it for a bond whose `D` is its
    dissociation energy, where the limit is `-D + (D + h) = h` above free
    fragments and `h` defaults to `bond_asymptote`.  A fitted `D` here is not
    the dissociation energy (the module docstring says why), so the height is
    stated against the fragments instead of against `D`, and `h` is whatever
    reaches it.  Without that, H2 fitted at B3LYP levels off 0.28 eV *below*
    two H atoms and its bonded diabat never crosses theirs.

    The nonbonded part of the limit is read off `baseline` -- the fit's
    nonbonded block, before any bonded term is in it -- at `SEPARATION`, and
    folded into the target, since it does not depend on anything being fitted.
    """
    fragments = training_set.of_kind("fragment")
    if not fragments:
        return {}
    equilibrium = training_set.equilibrium
    pairs: dict = {}
    for frame in fragments:
        bond = tuple(int(i) for i in np.atleast_1d(frame.info["fragment_bond"]))
        pairs.setdefault(bond, []).append(frame)

    targets = {}
    for bond, pair in pairs.items():
        if len(pair) != 2:
            continue
        # The side holding the bond's second atom moves off along the bond.
        moving = next(
            np.atleast_1d(f.info["fragment_atoms"]).astype(int)
            for f in pair
            if bond[1] in np.atleast_1d(f.info["fragment_atoms"])
        )
        positions = equilibrium.get_positions()
        axis = positions[bond[1]] - positions[bond[0]]
        stretched = equilibrium.copy()
        positions[moving] += SEPARATION * axis / np.linalg.norm(axis)
        stretched.set_positions(positions)
        nonbonded, _ = _nonbonded(baseline, [stretched], topology.graph)
        fragments_energy = sum(f.get_potential_energy() for f in pair)
        target = fragments_energy + active().bond_asymptote - float(nonbonded[0])
        targets[_bond_class(topology, bond)] = (bond, moving, target)
    return targets


def _asymptotes(targets, topology, qforce, vecs, shape, valence: float) -> np.ndarray:
    """Per-class stretched well depth `Dw` that puts each limit on its target.

    At the limit the cut bond's Morse term is exactly `Dw - D` (the
    exponential has underflowed) and every other bonded term keeps its
    equilibrium value: the fragments are rigid, and pulling one straight along
    the bond moves no angle or dihedral.  The cross terms that carry the cut
    bond's own stretch are the exception -- they are linear in it and have no
    limit -- and are held at their equilibrium value too, which is what the
    fragments' own templates (with no such term) would say.  So

        E_limit = E_bonded(eq) - E_cut(eq) + (Dw - D) + E_nonbonded(apart)

    **It is `Dw` that is solved, not `h`,** because the training frames cannot
    tell the two apart.  They see the stretched branch only through its well
    `Dw`, and `D` only through the constant `-D` the fit's `E0` absorbs -- so a
    lower `h` is answered by a higher `D` with `Dw`, and the limit, where they
    were.  Solved for `h` after the fit, O2 walked `D` from 6.6 to 10.1 eV over
    six rounds with its limit 0.45 eV off and not converging.  Held fixed
    instead, `Dw` is what the fragments say and `D` is what the compressed
    branch says, and `h = Dw - D` is what reaches the file.

    `E_bonded(eq)` carries `Dw` wherever a bond of the class is stretched at
    equilibrium (a fitted `r0` shorter than the bond), weakly, so the solve is
    iterated to its fixed point.  Classes with no fragments come back nan and
    keep `bond_asymptote`.
    """
    n = topology.n_classes("bond")
    atoms, classes = topology.atoms["bond"], topology.classes["bond"]
    dw = np.array(shape["Dw"], dtype=float) if "Dw" in shape else np.full(n, np.nan)
    for c in targets:
        if np.isnan(dw[c]):
            dw[c] = float(shape["D"][c]) + active().bond_asymptote
    for _ in range(50):
        kwargs = _bond_kwargs({**shape, "Dw": dw}, classes)
        bonded, _, _ = qforce.compute_bond(vecs, atoms, **kwargs)
        updated = dw.copy()
        for c, (bond, _, target) in targets.items():
            row = next(
                r for r, b in enumerate(atoms) if frozenset(map(int, b)) == frozenset(bond)
            )
            cut, _, _ = qforce.compute_bond(
                vecs, atoms[[row]], **{name: value[[row]] for name, value in kwargs.items()}
            )
            limit = valence + bonded - cut + kwargs["h"][row]
            # A well has to stay a well.  A limit this far below the target
            # means fragments far below the molecule, which is a reference
            # problem and not something a Morse depth can express.
            updated[c] = max(dw[c] + target - limit, 0.1)
        if np.nanmax(np.abs(updated - dw)) < 1e-12:
            return updated
        dw = updated
    return dw


def fit(
    training_set,
    topology: Topology,
    config: FitConfig | None = None,
    initial=None,
) -> Parameters:
    """Fit every bonded force constant, plus the reference energy offset.

    `initial` optionally supplies a force field to start from -- a `Parameters`,
    a term dict, or a path to a jsonl file.  Terms are matched to this topology
    by their atom slots, and anything the initial field does not cover keeps its
    ordinary default.  See the module docstring for which blocks a starting
    point actually moves.
    """
    config = config or FitConfig()
    # The fragment frames are not this molecule, and are read only by
    # `_asymptote_targets`; everything the least squares sees is the molecule.
    frames = [f for f in training_set.frames if f.info.get("frame_kind") != "fragment"]
    equilibrium = training_set.equilibrium
    numbers = equilibrium.get_atomic_numbers()
    initial = as_parameters(initial)
    seed = None if initial is None else _seed_from(topology, initial, len(numbers))

    # --- nonbonded baseline, never fit -------------------------------------
    params = Parameters(numbers=numbers)
    index_column = np.arange(len(numbers))[:, None]
    defaults = elements.defaults_for(numbers, config.electrostatics)
    for term, kwargs in defaults.items():
        params.terms[term] = {"atoms": index_column.copy(), "kwargs": dict(kwargs)}
    if config.electrostatics == "fixed":
        # Not an element default, so it comes from the reference calculation
        # rather than from `elements`.
        params.terms["charge"] = _charge_block(topology, equilibrium)
    if seed is not None:
        _apply_nonbonded(params, seed)

    e_nb, f_nb = _nonbonded(params, frames, topology.graph)
    b_energy = np.array([f.get_potential_energy() for f in frames]) - e_nb
    b_force = np.array([f.get_forces().reshape(-1) for f in frames]) - f_nb

    # --- fixed equilibrium values for every term except the bond -----------
    values = equilibrium_values(topology, equilibrium.get_positions())
    all_vecs = [_pair_vectors(frame) for frame in frames]

    # The bond bounds are set from the *measured* geometry even when an initial
    # field supplies `r0`, so a supplied value is a starting point inside the
    # same box rather than a way to move the box.
    bond_classes = topology.classes["bond"]
    n_bond_classes = topology.n_classes("bond")
    geometric_r0 = np.array(
        [values["bond"]["r0"][bond_classes == c][0] for c in range(n_bond_classes)]
    )

    if seed is not None:
        for term, supplied in seed.values.items():
            for name, value in supplied.items():
                values[term][name] = _merge(values[term][name], value)

    weight = config.energy_weight
    if weight is None:
        # Balance the two blocks so neither dominates purely by having more rows
        # or larger units.
        scale_f = np.linalg.norm(b_force) or 1.0
        scale_e = np.linalg.norm(b_energy - b_energy.mean()) or 1.0
        weight = float(scale_f / scale_e / np.sqrt(len(frames)))

    qforce = QForce(bond_form=config.bond_form)

    # --- bond parameters: seeded from geometry and the element table -------
    pairs = topology.atoms["bond"][:, :2]
    well_depth = elements.morse_well_depth(numbers[pairs[:, 0]], numbers[pairs[:, 1]])
    shape = {
        "r0": geometric_r0.copy(),
        "k": np.full(n_bond_classes, 30.0),
        "D": np.array(
            [well_depth[bond_classes == c][0] for c in range(n_bond_classes)]
        ),
    }
    if seed is not None:
        shape = {
            name: _merge(value, seed.bond.get(name)) for name, value in shape.items()
        }
        if "Dw" in seed.bond:
            shape["Dw"] = seed.bond["Dw"]

    # --- the linear block's basis, which the parameters do not enter -------
    # `_basis_columns` evaluates every non-bond term at `k = 1`, so it depends
    # only on the geometry and the fixed equilibrium values.  Neither changes
    # during the fit, so this is built once rather than every cycle.
    energy_rows, force_rows, labels = [], [], None
    for vecs in all_vecs:
        energies, forces, labels = _basis_columns(topology, values, qforce, vecs)
        energy_rows.append(energies)
        force_rows.append(forces.T)
    # Shaped explicitly: a diatomic has nothing but its bond, and an empty
    # column list would otherwise collapse to `(0,)` rather than `(rows, 0)`.
    a_energy = np.array(energy_rows).reshape(len(frames), len(labels))
    a_force = np.concatenate(force_rows, axis=0).reshape(b_force.size, len(labels))

    def bond_target(k_values):
        """What the bonds have to reproduce once everything else is fixed."""
        target_e = b_energy - a_energy @ k_values
        target_f = (b_force.reshape(-1) - a_force @ k_values).reshape(len(frames), -1)
        return target_e, target_f

    if seed is not None and seed.k:
        # The one place a supplied force constant reaches the result: refine the
        # bonds against the residual the supplied field leaves, before the first
        # linear solve replaces those constants with its own.
        k_start = np.array([seed.k.get(label, 0.0) for label in labels])
        target_e, target_f = bond_target(k_start)
        shape = _refine_bonds(
            topology, qforce, all_vecs, shape, target_e, target_f, weight, geometric_r0
        )

    # --- alternate: linear for every other k, nonlinear for the bonds ------
    # Each block minimizes the same weighted residual, so it decreases
    # monotonically; the loop stops when it stops moving rather than after a
    # fixed count.
    #
    # Neither block fits `E0`.  Both see their energy residual mean-centered, so
    # a constant shift of the whole profile costs nothing anywhere in the loop.
    # That is deliberate, and it is what makes the alternation terminate.  With
    # `a = sqrt(k/2D)`, `D*(1 - exp(-a*dr))**2` is `k*dr**2/2` to second order,
    # so near equilibrium `D` reaches the energy only through its trailing
    # constant `-D` per bond -- and a constant per bond is exactly what `E0` is.
    # Fitting both put a flat direction straight through the parameter space:
    # the nonlinear block would raise `D`, the linear block would raise `E0` to
    # compensate, and the pair would walk for hundreds of cycles while the
    # residual barely moved.  Centering deletes that direction rather than
    # penalizing it, which leaves `D` fixed by the anharmonicity it actually
    # describes.
    def converge(shape):
        """The alternation to convergence at the `h` in `shape`, then `E0`."""
        k_values = np.zeros(len(labels))
        previous, cycles_used = None, 0
        for cycle in range(max(1, config.n_cycles)):
            cycles_used = cycle + 1
            e_bond, f_bond = _bond_block(topology, qforce, all_vecs, shape)

            k_values = _solve_linear(
                a_energy,
                a_force,
                b_energy - e_bond,
                (b_force - f_bond).reshape(-1),
                labels,
                config,
                weight,
            )

            # everything except the bonds, at the freshly solved force constants
            target_e, target_f = bond_target(k_values)
            shape = _refine_bonds(
                topology, qforce, all_vecs, shape, target_e, target_f, weight, geometric_r0
            )

            e_bond, f_bond = _bond_block(topology, qforce, all_vecs, shape)
            residual = float(
                np.linalg.norm(f_bond - target_f) ** 2
                + weight**2 * np.linalg.norm(_centered(e_bond - target_e)) ** 2
            )
            if previous is not None and previous - residual <= config.cycle_tol * previous:
                break
            previous = residual

        # The loop leaves the bonds one refinement ahead of the force constants that
        # refinement was run against.  Re-solving the linear block against the final
        # bonds costs one solve and makes the returned pair mutually consistent, so
        # that what comes back is a point the alternation actually visits rather
        # than a half-step past one.
        e_bond, f_bond = _bond_block(topology, qforce, all_vecs, shape)
        k_values = _solve_linear(
            a_energy,
            a_force,
            b_energy - e_bond,
            (b_force - f_bond).reshape(-1),
            labels,
            config,
            weight,
        )

        # --- and only now, the offset ------------------------------------------
        # Nothing above has any opinion about where the energy zero sits, so what is
        # left over is a constant: the mean error of the converged field against the
        # reference.  Setting `E0` to it is the least-squares optimum for a constant
        # by definition, and it is the last thing the fit does.
        e0 = float(np.mean(b_energy - e_bond - a_energy @ k_values))
        return shape, k_values, e0, cycles_used

    shape, k_values, e0, cycles_used = converge(shape)

    # --- the asymptotes, against the fragments ------------------------------
    # The stretched well `Dw` is solved rather than fitted -- one condition per
    # bond class, from the fragment frames (`_asymptotes` says why `Dw` and not
    # `h`) -- and it reaches every stretched training frame, so the
    # alternation is re-run under it until the limit stops moving.
    targets = _asymptote_targets(training_set, topology, params)
    asymptotes = {}
    if targets:
        at_eq = next(i for i, f in enumerate(frames) if f is equilibrium)
        for _ in range(ASYMPTOTE_ROUNDS):
            dw = _asymptotes(
                targets, topology, qforce, all_vecs[at_eq], shape,
                valence=float(a_energy[at_eq] @ k_values) + e0,
            )
            moved = np.nanmax(np.abs(dw - shape["Dw"])) if "Dw" in shape else np.inf
            shape = {**shape, "Dw": dw}
            # Relative, and at the fit's own tolerance: a fixed floor here
            # would outlast a tightened `cycle_tol` and leave a refit drifting
            # by it however tightly the alternation converged.
            if moved <= config.cycle_tol * np.nanmax(np.abs(dw)):
                break
            shape, k_values, e0, cycles_used = converge(shape)
        symbols = equilibrium.get_chemical_symbols()
        for c, (bond, _, _) in sorted(targets.items()):
            h = float(shape["Dw"][c] - shape["D"][c])
            asymptotes[f"{symbols[bond[0]]}-{symbols[bond[1]]} {c}"] = h

    params = _assemble(
        params, topology, values, labels, k_values, e0, shape, bond_classes
    )
    params.report = _report(params, frames)
    params.report["classes"] = len(labels) + n_bond_classes
    params.report["frames"] = len(frames)
    params.report["cycles"] = cycles_used
    if asymptotes:
        params.report["asymptote_h"] = asymptotes
    if seed is not None:
        params.report["seeded"] = seed.n_seeded
    return params


def _assemble(params, topology, values, labels, k_values, e0, shape, bond_classes):
    """Expand per-class parameters back onto every term."""
    by_class = {label: k for label, k in zip(labels, k_values, strict=True)}
    for term in sorted(topology.atoms):
        if term not in GEOMETRIC:
            continue
        kwargs = {name: value.copy() for name, value in values.get(term, {}).items()}
        kwargs.update({n: v.copy() for n, v in topology.fixed.get(term, {}).items()})
        if term == "bond":
            kwargs = _bond_kwargs(shape, bond_classes)
        else:
            kwargs["k"] = np.array(
                [by_class[(term, int(c))] for c in topology.classes[term]]
            )
        params.terms[term] = {"atoms": topology.atoms[term].copy(), "kwargs": kwargs}

    params.terms["reference"] = {
        "atoms": np.zeros((1, 1), dtype=int),
        "kwargs": {"E0": np.array([e0])},
    }
    return params


def _report(params, frames) -> dict:
    """Energy and force RMSE, overall and by frame kind."""
    from .calculator import term_dict_for

    td = term_dict_for(params)
    per_kind: dict[str, list] = {}
    energy_errors, force_errors = [], []
    for frame in frames:
        result = evaluate_term_dict(frame, td)
        de = result.energy - frame.get_potential_energy()
        df = (result.forces - frame.get_forces()).reshape(-1)
        energy_errors.append(de)
        force_errors.append(df)
        per_kind.setdefault(frame.info.get("frame_kind", "unknown"), []).append(
            (de, df)
        )

    report = {
        "energy_rmse_eV": float(np.sqrt(np.mean(np.square(energy_errors)))),
        "force_rmse_eV_A": float(
            np.sqrt(np.mean(np.square(np.concatenate(force_errors))))
        ),
    }
    for kind, entries in sorted(per_kind.items()):
        des = np.array([e for e, _ in entries])
        dfs = np.concatenate([f for _, f in entries])
        report[kind] = {
            "n": len(entries),
            "E_rmse": float(np.sqrt(np.mean(des**2))),
            "F_rmse": float(np.sqrt(np.mean(dfs**2))),
        }
    return report
