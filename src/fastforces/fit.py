"""Fitting the bonded parameters to a reference calculator.

The whole fit rests on one observation: with the equilibrium values fixed, every
bonded term in `QForce` is a force constant `k` multiplying a function of the
geometry alone. So the energy of a frame is

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

Bonds are fitted as harmonic springs, `k*dr**2/2` about the measured `r0`, and
written as the Morse bond the simulation evaluates:

  * A Morse bond is not linear in `k` -- its exponent is `sqrt(k / 2D)` -- but
    its curvature at `r0` is `k` whatever `D` is: `D*(1 - exp(-a*dr))**2 - D`
    is `k*dr**2/2 - D` to second order.  So the harmonic fit is the Morse `k`,
    and what the written bond adds on top is anharmonicity the fit never saw.
    `D` is not fitted to the frames, which see it only through anharmonicity:
    it starts from the element table (`elements.morse_well_depth`) or from
    `initial=`, and every depth is scaled by one factor so the bonds carry the
    molecule's atomization energy, against the free atoms the training set
    also carries (`sampling.atom_frames`, `_depth_scale`).  The per-bond
    asymptote `h` is not fitted either, but *solved*: the training set carries the two
    fragments of each bond class (`sampling.fragment_frames`), and the
    stretched well is set so that pulling the bond apart ends `bond_asymptote`
    above them (`_asymptotes`).  A training set without fragment frames reads
    `bond_asymptote`.  `refine.fit_force_constants` refits `h` against the
    reaction barriers on top of this.
  * `r0` is the equilibrium bond length, averaged within its class like every
    other equilibrium value.

`E0` is a fixed scalar added to the sum of the molecule's terms: a constant
shift of its whole potential energy surface, with no force.  It is what keeps
the relative energies of molecules right, so that a reactant state and a
product state -- each the sum of its molecules' energies -- carry the reference
offset between them at the two ends of a reaction coordinate.  That needs every
molecule on one absolute scale, the reference calculator's, and `E0` is what
puts it there: the offset between the terms and the molecule's reference
energies.

It is not in the solve.  Every energy residual is mean-centered, so the shift
costs nothing there, and `E0` is read off the leftover mean afterwards -- the
least-squares optimum for a constant -- plus the `-D` per bond that the Morse
form carries at its minimum and the harmonic one does not.  Scaling the depths
changes how much of the total the bonds hold and how much `E0` holds, and not
the total, so no molecule moves on the scale.  At the scale that puts the
atomization energy on the bonds, `E0` comes out as the summed free-atom
energies: the same for any set of molecules with the same atoms, so between a
reactant and a product state it cancels and the offset is the terms' alone.  A
training set with no atom frames keeps the unscaled depths, and an `E0` that
holds part of the atomization energy.

A fit can be started from an existing force field rather than from the element
table, by passing `initial=`.  Only what the fit holds fixed can be supplied:

  * The equilibrium values (bond `r0` excepted, which is always measured), the
    bond depths `D` and asymptotes, and the nonbonded baseline replace what
    would otherwise be measured off the geometry or read from the element
    table.  Supplied depths are still rescaled to the atomization energy, so
    what they set is the ratio between them.
  * The force constants are recovered by a bounded *global* least squares that
    has no starting point at all, so a supplied `k` does not reach the result.
"""

from dataclasses import asdict, dataclass, field
from os import PathLike

import numpy as np
from scipy.optimize import lsq_linear

from ase import Atoms
from DynamicTopology.forcefield.evaluate import evaluate_term_dict, nonbonded, term_dict
from DynamicTopology.forcefield.exclusions import exclusion_terms
from DynamicTopology.forcefield.params import active
from DynamicTopology.forcefield.qforce import QForce

from . import elements
from .refine import SCALE_BRACKET
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

# Force constants the solve must keep positive: a negative bond or angle `k`
# inverts the well.  The cross terms are genuinely signed and are left free.
POSITIVE_K = ("bond", "angle")

# The Morse parameters of a bond class, per class, as `_bond_kwargs` reads
# them.  `r0` is measured and `k` fitted; `D` is the only one a starting point
# supplies.  The stretched well `Dw` (so `h = Dw - D`) rides alongside once
# `_asymptotes` has solved it.
_BOND_NAMES: tuple[str, ...] = ("r0", "k", "D")

# Where a template's reference charges can come from; `FitConfig.electrostatics`.
# `mulliken` and `esp` name the per-atom array on the equilibrium frame they are
# read from.
CHARGE_SOURCES: tuple[str, ...] = ("mulliken", "esp", "neutral")
# What `FitConfig.electrostatics` used to hold, and the source each one meant.
_LEGACY_SOURCES: dict[str, str] = {"acks2": "neutral", "fixed": "mulliken"}
# `FitConfig` fields of the nonlinear bond refinement, which the linear fit
# retired: the bond form it evaluated and the cap and tolerance on alternating
# it with the linear block.  Older training files still record them.
_RETIRED_FIELDS: tuple[str, ...] = ("bond_form", "n_cycles", "cycle_tol")


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
    # Where each atom's reference charge comes from -- one of `CHARGE_SOURCES`.
    # `"mulliken"` and `"esp"` are read off the equilibrium frame
    # (`_reference_charges`); `"neutral"` is zero on every atom, which only a
    # neutral template can carry.  *How* those charges are used is not this
    # field's to say: that is DynamicTopology's `global_params.electrostatics`,
    # read through `active()` -- `"acks2"` makes them the `atom` block's `q0`,
    # the reference each state equilibrates around, and `"pointcharge"` makes
    # them the `charge` block itself.  Neither is fitted -- both are part of the
    # baseline the bonded terms are fit against -- so a field refit with another
    # source is a different force field rather than a reparametrized one.
    electrostatics: str = "neutral"

    def __post_init__(self):
        if self.electrostatics not in CHARGE_SOURCES:
            hint = ""
            if self.electrostatics in _LEGACY_SOURCES:
                hint = (
                    f"; {self.electrostatics!r} was the old name for the "
                    "inference method, which is now `global_params."
                    "electrostatics` ('acks2' or 'pointcharge') alone"
                )
            raise ValueError(
                f"electrostatics is the reference-charge source, one of "
                f"{list(CHARGE_SOURCES)}, not {self.electrostatics!r}{hint}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_stored(cls, stored: dict) -> "FitConfig":
        """The config a training set's metadata records, old names translated.

        A set written before the charge source and the inference method were
        split stores `"acks2"` (element defaults, no reference charge -- what
        `"neutral"` is now) or `"fixed"` (Mulliken charges).  A *manifest*
        stating either is refused instead, since there the old name is a
        setting somebody should restate.  A set written before the bond fit
        went linear also records `_RETIRED_FIELDS`, which are dropped.
        """
        stored = {k: v for k, v in stored.items() if k not in _RETIRED_FIELDS}
        if stored.get("electrostatics") in _LEGACY_SOURCES:
            stored["electrostatics"] = _LEGACY_SOURCES[stored["electrostatics"]]
        return cls(**stored)


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
    geometry for an equilibrium value, the element table for a bond depth.

    Only what the fit holds fixed is kept.  A supplied force constant has
    nowhere to go -- the linear solve has no starting point -- and neither has
    a bond `r0`, which is always measured; both still count towards
    `n_seeded`, which is the classes the supplied field matched.
    """

    values: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    bond: dict[str, np.ndarray] = field(default_factory=dict)
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
            # Only the depth: `r0` stays measured and `k` is fitted.
            if "D" in per_class:
                seed.bond["D"] = per_class["D"]
                # A supplied asymptote arrives as the stretched well it fixes,
                # the form `_asymptotes` solves for, so a class with no
                # fragments keeps the supplied limit rather than the default.
                if "h" in per_class:
                    seed.bond["Dw"] = per_class["D"] + per_class["h"]
            continue
        supplied = {
            name: per_class[name][classes]
            for name in GEOMETRIC.get(term, {})
            if name in per_class
        }
        if supplied:
            seed.values[term] = supplied
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


def _basis_columns(topology, values, qforce, vecs):
    """`(energy, forces)` per class, evaluated at `k = 1`.

    `qforce` has to be the harmonic one, which is what makes the bond a column
    like any other: `k*dr**2/2` is linear in `k`, where the Morse exponent is
    not.
    """
    energies, forces, labels = [], [], []
    for term in sorted(topology.atoms):
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
            if term == "bond":
                # The harmonic form ignores `D`, but the signature takes it.
                kwargs["D"] = np.zeros(int(mask.sum()))
            # `QForce` is in eV and Angstrom throughout, which is what the fit
            # works in, so the `compute_*` methods can be called directly with
            # `vecs` straight off the frame.  The third return is the virial,
            # which is fixed-cell here and unused.
            energy, force, _ = compute(vecs, atoms[mask], **kwargs)
            energies.append(energy)
            forces.append(force.reshape(-1))
            labels.append((term, c))
    return np.array(energies), np.array(forces), labels


def formal_charge(frame, meta: dict | None = None) -> int:
    """The total charge of the template `frame` stands for.

    The frame's own initial charges when it carries any (`build` sets them from
    the SMILES, and the training file keeps them), else the training set's
    `charge` metadata, else neutral.
    """
    charges = frame.get_initial_charges()
    if np.any(charges):
        return int(round(float(np.sum(charges))))
    return int((meta or {}).get("charge", 0))


def check_charge_source(source: str) -> None:
    """Refuse a charge source the active inference method cannot use.

    Neutral reference charges under `pointcharge` put no charge on any atom of
    any template -- a dataset with no electrostatics at all, which one that
    asked for point charges did not mean.  Under `acks2` they are the ordinary
    choice for a neutral dataset: the charges still equilibrate.
    """
    if source == "neutral" and active().electrostatics == "pointcharge":
        raise ValueError(
            "`global_params.electrostatics='pointcharge'` with "
            "`electrostatics='neutral'` reference charges puts no charge on any "
            "atom; fit the charges with 'mulliken' or 'esp'"
        )


def _reference_charges(topology: Topology, equilibrium, source: str, formal: int):
    """Each atom's reference charge, from `FitConfig.electrostatics`.

    `mulliken` and `esp` are the per-atom arrays of those names on the
    *equilibrium* frame -- the same frame every other fixed value in this fit is
    measured on.  `mulliken` is written by `calculators.pyscf.PySCFCalculator`
    and `calculators.tblite.TBLiteCalculator` onto every frame they evaluate;
    `esp` by `add_reference_charges`, which fits Merz-Kollman charges to the
    reference density once (`charges.esp_charges`).  They are not fit
    afterwards.

    They are then averaged within each atom equivalence class.  Either comes out
    of a single geometry, so the three hydrogens of a methyl group get three
    slightly different values, and freezing that asymmetry in would put a
    spurious electrostatic torsion on a rotor that has none.  Every other
    parameter in this fit is per-class for the same reason.  A class-wise mean
    also leaves the total where it was, which is the formal charge -- and that
    is checked, because it is the property both uses rely on: fragment ACKS2
    holds each molecule at the sum of its `q0`, and a point-charge template
    carries its ion's charge.

    `neutral` is zero everywhere, so it is refused for a charged template: that
    is the case the reference charges exist to carry.
    """
    n = len(equilibrium)
    check_charge_source(source)
    if source == "neutral":
        if formal != 0:
            raise ValueError(
                f"`electrostatics='neutral'` gives every atom a zero reference "
                f"charge, which cannot carry this template's formal charge of "
                f"{formal:+d}; fit it with 'mulliken' or 'esp'"
            )
        return np.zeros(n)
    if source not in equilibrium.arrays:
        raise ValueError(
            f"fitting with `electrostatics={source!r}` needs per-atom charges, "
            f"and this training set's equilibrium frame has no `{source}` array.  "
            "`add_reference_charges` computes it from the reference calculator "
            "(`parameterize` and a manifest run call it); `mulliken` is also "
            "written by `calculators.pyscf.PySCFCalculator` and "
            "`calculators.tblite.TBLiteCalculator`, but not by plain "
            "`tblite.ase.TBLite`"
        )
    raw = np.asarray(equilibrium.get_array(source), dtype=float)
    q = class_average(raw, topology.atom_classes)
    if abs(float(q.sum()) - formal) > 1e-4:
        raise ValueError(
            f"the `{source}` charges sum to {q.sum():+.4f}, not this template's "
            f"formal charge of {formal:+d}; they were computed for another "
            "charge state"
        )
    return q


def _electrostatic_block(numbers, q: np.ndarray) -> tuple[str, dict]:
    """`(term, block)`: the reference charges `q` as `active()` evaluates them.

    Under `acks2` they are the `atom` block's `q0`, alongside the element
    defaults; under `pointcharge` they are the `charge` block.
    """
    index_column = np.arange(len(numbers))[:, None]
    if active().electrostatics == "pointcharge":
        return "charge", {"atoms": index_column, "kwargs": {"q": q}}
    kwargs = dict(elements.acks2_defaults(numbers))
    kwargs["q0"] = np.asarray(q, dtype=float)
    return "atom", {"atoms": index_column, "kwargs": kwargs}


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
    out rather than merely checking it: its `nonbonded` half, plus `QForce` over
    the exclusions, is the sum `evaluate_term_dict` adds.

    **The electrostatics is handed `graph`'s bonds**, as atom pairs only.
    Fragment ACKS2 reads its molecules off the `bond` block, and without one
    every atom is a molecule of its own, pinned at its own `q0` -- so a
    hydroxide's two atoms became a -1.03 and a +0.03 point charge 0.97 A apart,
    0.39 eV below the zero a lone template scores in the simulation, and the
    bonded fit absorbed that as a 0.2 eV/A force error.  With every `q0` zero
    the same mistake scored exactly zero, which is why it went unseen.

    The virial is dropped: the fit works at fixed cell.
    """
    numbers = np.asarray(params.numbers)
    terms = params.to_terms()
    terms = terms + exclusion_terms(terms, numbers, graph=graph)
    td = term_dict(Atoms(numbers=numbers), terms)
    edges = np.array(sorted(tuple(sorted(e)) for e in graph.edges()), dtype=int)
    molecules = dict(td)
    molecules["bond"] = {"atoms": edges.reshape(-1, 2), "kwargs": {}}
    qforce = QForce()

    energies, forces = [], []
    for frame in frames:
        pos, pbc, cell = frame.positions, frame.pbc, frame.cell.array
        e_x, f_x, _ = qforce(pos, pbc, cell, td)
        e_n, f_n, _ = nonbonded(frame, molecules)
        energies.append(float(e_x + e_n))
        forces.append((f_x + f_n).reshape(-1))
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


def _centered(values: np.ndarray) -> np.ndarray:
    """`values` with their mean over frames removed.

    Every energy residual in the fit passes through here.  `E0` is not a fitted
    parameter -- it is read off the leftover mean once the force constants are
    solved -- so an energy offset is not the fit's to chase, and the solve is
    made blind to it.  That is the same as an unpenalized constant column, but
    it keeps the Tikhonov rows off the offset without special-casing it.
    """
    return values - values.mean(axis=0)


def _solve_linear(a_energy, a_force, b_energy, b_force, labels, config, weight):
    """Bounded, regularized least squares for every force constant."""
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


# Frame kinds a training set carries that are not the molecule, so are not fit
# to: `_asymptote_targets` reads the fragments and `_free_atoms` the atoms.
NOT_FITTED: tuple[str, ...] = ("fragment", "atom")


def _free_atoms(training_set, numbers) -> float | None:
    """Summed reference energy of the molecule's atoms, each free and neutral.

    `None` when the training set lacks the atom frame of some element -- one
    written before they existed, or whose atoms failed to label -- and the
    depths then keep their unscaled values.
    """
    energies = {
        int(frame.get_atomic_numbers()[0]): frame.get_potential_energy()
        for frame in training_set.of_kind("atom")
    }
    if not all(int(z) in energies for z in numbers):
        return None
    return float(sum(energies[int(z)] for z in numbers))


def _depth_scale(offset: float, depth: float, free_atoms: float) -> float:
    """The one factor on every `D` that puts the atomization energy on the bonds.

    The written field is `E0 + sum_b Morse_b + rest`, with `E0 = offset +
    sum_b D_b`, and `Morse_b + D_b` is the harmonic spring the bond was fitted
    as, to second order.  So near equilibrium a common scale `s` on the depths
    moves no energy the fit sees, and moves the molecule's total not at all: it
    only decides how much of the energy sits in the depths and how much in the
    shift `E0`.  The bonds carry the whole atomization energy when `E0` holds
    none of it -- when the shift is just the free atoms,

        offset + s * sum_b D_b  =  sum_i E(atom_i)

    which is the condition `refine.fit_dissociation_energies` root-finds
    against the whole field at a template's reference geometry, with its shift
    zeroed -- closed form here, because `E0` is already tied to the depths.

    "Carry" is in that whole-field sense: at equilibrium the bonds contribute
    `-sum_b D_b`, and the angles, cross terms and nonbonded baseline contribute
    what they do there, so the depths add up to the atomization energy less
    those.  Matching the depths alone to it left water 25% overbound by its own
    electrostatic energy (see `refine.nonbonded_energy`).

    `SCALE_BRACKET` is the tripwire `refine` uses: a factor outside it means the
    reference energies and the force field disagree about the molecule.
    """
    scale = (free_atoms - offset) / depth
    low, high = SCALE_BRACKET
    if not low <= scale <= high:
        raise ValueError(
            f"the bond depths would need scaling by {scale:.4g} to carry the "
            f"atomization energy, outside [{low}, {high}]: the fit's energy zero "
            f"sits {offset - free_atoms:+.4f} eV from the free atoms against "
            f"{depth:.4f} eV of element-table depth.  Check the atom frames' "
            "spins and the molecule's reference energy."
        )
    return float(scale)


# How far apart the two fragments are put to read the nonbonded energy the
# stretched limit keeps, in Angstrom.  Far enough that what the fragments still
# see of each other is the Coulomb tail of their partial charges, q_A q_B / r:
# 2 meV at +/-0.4 e.  Nothing else reaches it -- ZBL and the 12-6 are cut off
# long before, and the Morse exponent has underflowed to zero.
SEPARATION: float = 1000.0


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
    fragments and `h` defaults to `bond_asymptote`.  The `D` here is an
    element-table single-bond depth, not this bond's dissociation energy at this
    reference level, so the height is stated against the fragments instead of
    against `D`, and `h` is whatever reaches it.  Without that, H2 at B3LYP
    levelled off 0.28 eV *below* two H atoms and its bonded diabat never
    crossed theirs.

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

    With `D` held, solving `Dw` is solving `h = Dw - D`, which is what reaches
    the file; `Dw` is carried because it is the well the stretched branch
    climbs, and the form a supplied asymptote arrives in.

    `E_bonded(eq)` carries `Dw` wherever a bond of the class is stretched at
    equilibrium -- a bond longer than its class's averaged `r0` -- weakly, so
    the solve is iterated to its fixed point.  Classes with no fragments come
    back nan and keep `bond_asymptote`.
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
    # The fragment and atom frames are not this molecule, and are read only by
    # `_asymptote_targets` and `_free_atoms`; everything the least squares sees
    # is the molecule.
    frames = [
        f for f in training_set.frames if f.info.get("frame_kind") not in NOT_FITTED
    ]
    equilibrium = training_set.equilibrium
    numbers = equilibrium.get_atomic_numbers()
    initial = as_parameters(initial)
    seed = None if initial is None else _seed_from(topology, initial, len(numbers))

    # --- nonbonded baseline, never fit -------------------------------------
    params = Parameters(numbers=numbers)
    q = _reference_charges(
        topology,
        equilibrium,
        config.electrostatics,
        formal_charge(equilibrium, getattr(training_set, "meta", None)),
    )
    term, block = _electrostatic_block(numbers, q)
    params.terms[term] = block
    params.terms["lennardjones"] = {
        "atoms": np.arange(len(numbers))[:, None],
        "kwargs": elements.lj_defaults(numbers),
    }
    if seed is not None:
        _apply_nonbonded(params, seed)

    e_nb, f_nb = _nonbonded(params, frames, topology.graph)
    b_energy = np.array([f.get_potential_energy() for f in frames]) - e_nb
    b_force = np.array([f.get_forces().reshape(-1) for f in frames]) - f_nb

    # --- fixed equilibrium values, the bond `r0` included ------------------
    values = equilibrium_values(topology, equilibrium.get_positions())
    all_vecs = [_pair_vectors(frame) for frame in frames]
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

    # --- every force constant, in one linear solve -------------------------
    # `_basis_columns` evaluates every term at `k = 1` -- the bond as a harmonic
    # spring about its measured `r0` -- so the basis depends only on the
    # geometry and the fixed equilibrium values.
    harmonic = QForce(bond_form="harmonic")
    energy_rows, force_rows, labels = [], [], None
    for vecs in all_vecs:
        energies, forces, labels = _basis_columns(topology, values, harmonic, vecs)
        energy_rows.append(energies)
        force_rows.append(forces.T)
    a_energy = np.array(energy_rows).reshape(len(frames), len(labels))
    a_force = np.concatenate(force_rows, axis=0).reshape(b_force.size, len(labels))
    k_values = _solve_linear(
        a_energy, a_force, b_energy, b_force.reshape(-1), labels, config, weight
    )

    # --- the Morse bond the harmonic one becomes ----------------------------
    # Same `r0`, same `k` -- the Morse curvature at `r0` is `k` -- and a depth
    # from the element table unless the starting point supplies one, scaled
    # below to carry the atomization energy.
    by_class = dict(zip(labels, k_values, strict=True))
    bond_classes = topology.classes["bond"]
    n_bond_classes = topology.n_classes("bond")
    pairs = topology.atoms["bond"][:, :2]
    well_depth = elements.morse_well_depth(numbers[pairs[:, 0]], numbers[pairs[:, 1]])
    first = [int(np.flatnonzero(bond_classes == c)[0]) for c in range(n_bond_classes)]
    shape = {
        "r0": values["bond"]["r0"][first],
        "k": np.array([by_class[("bond", c)] for c in range(n_bond_classes)]),
        "D": well_depth[first],
    }
    if seed is not None:
        shape["D"] = _merge(shape["D"], seed.bond.get("D"))
        if "Dw" in seed.bond:
            shape["Dw"] = seed.bond["Dw"]

    # --- the offset, and the depths that carry the atomization energy -------
    # The solve has no opinion about where the energy zero sits, so what is left
    # over is a constant: the mean error of the harmonic field against the
    # reference, which is the least-squares optimum for a constant by
    # definition.  The Morse form sits `-D` below the harmonic one at each
    # bond's minimum, and `E0` puts that back -- so the molecule's total, and so
    # its offset from every other molecule, is fixed here, while how it splits
    # between `E0` and the depths is free and `_depth_scale` chooses it.
    offset = float(np.mean(b_energy - a_energy @ k_values))
    free_atoms = _free_atoms(training_set, numbers)
    depth_scale = None
    if free_atoms is not None:
        depth_scale = _depth_scale(
            offset, float(np.sum(shape["D"][bond_classes])), free_atoms
        )
        shape["D"] = shape["D"] * depth_scale
    e0 = offset + float(np.sum(shape["D"][bond_classes]))

    # --- the asymptotes, against the fragments ------------------------------
    # Solved once the force constants are known, and nothing is refitted under
    # them: the harmonic fit never sees the stretched branch they set.
    targets = _asymptote_targets(training_set, topology, params)
    asymptotes = {}
    if targets:
        at_eq = next(i for i, f in enumerate(frames) if f is equilibrium)
        # Everything at equilibrium but the bonds, which `_asymptotes` evaluates
        # as Morse itself.
        valence = np.array([term != "bond" for term, _ in labels])
        shape["Dw"] = _asymptotes(
            targets,
            topology,
            QForce(bond_form="morse"),
            all_vecs[at_eq],
            shape,
            valence=float(a_energy[at_eq, valence] @ k_values[valence]) + e0,
        )
        symbols = equilibrium.get_chemical_symbols()
        for c, (bond, _, _) in sorted(targets.items()):
            h = float(shape["Dw"][c] - shape["D"][c])
            asymptotes[f"{symbols[bond[0]]}-{symbols[bond[1]]} {c}"] = h

    params = _assemble(
        params, topology, values, labels, k_values, e0, shape, bond_classes
    )
    params.report = _report(params, frames)
    params.report["classes"] = len(labels)
    params.report["frames"] = len(frames)
    if asymptotes:
        params.report["asymptote_h"] = asymptotes
    if depth_scale is not None:
        params.report["depth_scale"] = depth_scale
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
