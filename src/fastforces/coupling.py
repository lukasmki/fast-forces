"""Fitting the EVB off-diagonal coupling from a reaction's stationary points.

**There are three forms, and which one a channel gets is decided by its own
connectivity change.**  `reaction.Reaction.channel` decides; `fit` acts on the
answer.

    fission      V(r)  = A * exp(-a * (r - r_cross)**2)     compute_twobody
    transfer     V(g)  = A * exp(-a * g),  g in A**2        compute_threebody
    anything     V(x)  = A * exp(-a * RMSD(x, x_TS)**2)     compute_rmsd

**Two independent things distinguish them, and it is worth keeping them apart:
where the amplitude comes from, and what the width is measured in.**

    form        amplitude from            width measured in
    twobody     the diabatic crossing     the breaking bond's length
    threebody   the reference barrier     the transferring atom's triangle
    rmsd        the reference barrier     the RMSD to the whole geometry

Only `fit_twobody` changes the amplitude, and only because it has to: a fission
has no saddle, so there is no reference barrier to invert.  A transfer *does*
have one, and a reference barrier is reference data worth fitting to, so
`fit_threebody` keeps `fit_amplitude` untouched and changes only the metric the
width lives in.  An RMSD is a tolerance on all `3N` coordinates at once, so any
spectator switches the coupling off: with a water autoionization's transfer
atoms held exactly at its transition state, displacing only the three spectators
by 0.1 A -- less than thermal motion at 300 K -- takes the RMSD coupling from
4.14 eV to 8.4e-3, while the triangle form holds at 4.14.

The fission form exists because the RMSD one cannot describe that kind of
channel even in principle, not because it is more convenient.  For a barrierless
fission the bound diabat *is* the ground state up to the crossing, so a perfect
reactant diabat puts the reference barrier exactly on it and `fit_amplitude`'s
discriminant `(Hbar - E)**2 - dH**2` vanishes identically: the best attainable
amplitude is zero, every imperfection makes it imaginary, and no refit of the
force field moves that.  `fit_twobody` asks a question that has an answer
instead -- where do the two diabats cross, and what coupling hands one over to
the other smoothly -- and needs neither a transition-state geometry nor a
reference energy.

A crossing-centred fit is *not* the right answer for a transfer: two diabats
whose products lie uphill never cross along the transfer coordinate at all, so
there is no degeneracy to centre anything on.  Such a channel has a real saddle
and a real barrier instead, which is exactly the data `fit_amplitude` wants.

**Units.** Like every other term, coupling parameters are written to jsonl in
the calculator's own units -- eV and Angstrom, unconverted (`A` in eV, `ra0` in
Angstrom) -- so the round trip is exact rather than merely consistent.
"""

from dataclasses import dataclass, field

import numpy as np
from ase import Atoms

from DynamicTopology.forcefield.coupling import EVBCoupling, kabsch
from DynamicTopology.io.json import read_jsonl, write_jsonl

from .params import block_to_terms, terms_to_blocks

# Coupling considered switched off at this magnitude (eV).
DEFAULT_EPS: float = 1e-3

# Stand-in amplitude used to give a decoupled channel a finite width.  A zero
# amplitude has no width -- the condition `|V| <= eps` at the endpoints is
# satisfied everywhere -- but writing a nonsense number would make the term
# unreadable if someone later fills the amplitude in by hand.  1 eV is a
# plausible coupling, so the stored width stays meaningful.
NOMINAL_AMPLITUDE: float = 1.0

# Largest separation (A) searched for the diabatic crossing of a fission.
DEFAULT_MAX_SEPARATION: float = 8.0

# Separation (A) past which a bimolecular channel is no longer enumerated, and
# so past which its coupling has no state to couple to.
DEFAULT_BIMOL_CUTOFF: float = 4.0

# Step (A) of the coarse scan that brackets the crossing.  Only has to be finer
# than the distance over which the gap changes sign once; the bisection that
# follows does the accuracy.
DEFAULT_SCAN_STEP: float = 0.1


class CouplingError(ValueError):
    """Raised when the reference data cannot define a real coupling."""


# ---------------------------------------------------------------------------
# the container
# ---------------------------------------------------------------------------


@dataclass
class Coupling:
    """One reaction's off-diagonal term, in the shape `EVBCoupling` reads.

    `terms` is a `term_dict` exactly like `Parameters.terms`, so the evaluator
    takes it unchanged.  `ensemble` is the `(E, n, 3)` stack of template frames
    the RMSD form superposes onto -- unused by the other two forms, and carried
    regardless because it is what makes the term reproducible.

    `provenance` records, per term type, whether the amplitude was `fitted`,
    handed in as a `placeholder`, is zero (`decoupled`), or was set by hand and
    is to be kept (`manual`) -- DynamicTopology's vocabulary, since the rows are
    its.  It is not a parameter and no evaluator reads it; it is how a surface
    built on an unfitted coupling stays readable instead of having to be
    inferred from a magic amplitude value.  `limited_by` says, for a fission's
    `twobody` term, which end of the path set its width: `"reactant"` or
    `"cutoff"`.
    """

    terms: dict = field(default_factory=dict)
    ensemble: np.ndarray | None = None
    provenance: dict = field(default_factory=dict)
    limited_by: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = []
        for term, block in self.terms.items():
            amplitude = float(block["kwargs"]["A"][0])
            parts.append(f"{term}(A={amplitude:+.3f} eV, {self.provenance.get(term)})")
        return f"Coupling({', '.join(parts) or 'empty'})"

    # -- jsonl ----------------------------------------------------------

    def to_terms(self) -> list[dict]:
        """The DynamicTopology term list, as a reaction's `.jsonl` loads to."""
        out = []
        for term, block in self.terms.items():
            for row in block_to_terms(term, block):
                row["provenance"] = self.provenance.get(term, "fitted")
                if term in self.limited_by:
                    row["limited_by"] = self.limited_by[term]
                out.append(row)
        return out

    def to_jsonl(self, path: str) -> None:
        write_jsonl(path, self.to_terms(), exist_ok=True)

    @classmethod
    def from_terms(cls, terms: list[dict], ensemble=None):
        """Invert `to_terms`."""
        provenance = {t["type"]: t.get("provenance", "fitted") for t in terms}
        limited_by = {t["type"]: t["limited_by"] for t in terms if "limited_by" in t}
        return cls(
            terms=terms_to_blocks(terms),
            ensemble=ensemble,
            provenance=provenance,
            limited_by=limited_by,
        )

    @classmethod
    def from_jsonl(cls, path: str, ensemble=None):
        return cls.from_terms(read_jsonl(path), ensemble)

    # -- evaluation -----------------------------------------------------

    def __call__(self, positions, pbc=None, cell=None):
        """`(V, dV/dx, virial)` at one geometry, in eV and Angstrom."""
        positions = np.asarray(positions, dtype=float)
        pbc = np.zeros(3, dtype=bool) if pbc is None else pbc
        cell = np.zeros((3, 3)) if cell is None else np.asarray(cell)
        ensemble = self.ensemble
        if ensemble is None:
            ensemble = positions[None]
        return EVBCoupling()(positions, pbc, cell, ensemble, self.terms)

    def value(self, positions, pbc=None, cell=None) -> float:
        return float(self(positions, pbc, cell)[0])


# ---------------------------------------------------------------------------
# the two questions every form answers
# ---------------------------------------------------------------------------


def fit_amplitude(
    diabatic_reactant: float, diabatic_product: float, barrier: float
) -> float:
    """Invert the 2x2 secular equation at the transition state.

    With `E = Hbar - sqrt(dH**2 + V**2)` and `V(x_TS) = A`,

        A = -sqrt( (Hbar - E)**2 - dH**2 )

    All three energies must share one zero.  They do when the diabats are
    evaluated with force fields fitted against the same reference calculator the
    barrier was computed with: `fit` sets `E0` per fragment from that
    calculator's own energies, so a `FastForces` total is on the reference
    method's absolute scale.
    """
    mean = 0.5 * (diabatic_reactant + diabatic_product)
    half_gap = 0.5 * (diabatic_reactant - diabatic_product)

    # The adiabatic ground state of a two-level block lies below both diabats,
    # strictly so for any nonzero coupling.  A barrier above either of them
    # still yields a real discriminant, so test the ordering directly rather
    # than relying on the square root to catch it.
    if barrier > min(diabatic_reactant, diabatic_product):
        raise CouplingError(
            f"reference barrier {barrier:+.4f} eV lies above a diabatic energy at "
            f"the transition state ({diabatic_reactant:+.4f} and "
            f"{diabatic_product:+.4f} eV). The EVB ground state is below every "
            "diabat by construction, so no coupling reproduces this; the "
            "diabatic energies are wrong, not the barrier."
        )

    discriminant = (mean - barrier) ** 2 - half_gap**2
    if discriminant < 0.0:
        raise CouplingError(
            f"reference barrier {barrier:+.4f} eV does not lie below both diabatic "
            f"energies at the transition state ({diabatic_reactant:+.4f} and "
            f"{diabatic_product:+.4f} eV), so no real coupling reproduces it."
        )
    # Negative by convention.  Only `A**2` enters the ground-state eigenvalue of
    # a two-level block, so the sign is a phase choice.
    return -float(np.sqrt(discriminant))


def rmsd(reference: np.ndarray, positions: np.ndarray) -> float:
    """Optimally superposed RMSD, matching `EVBCoupling.compute_rmsd`.

    Goes through the evaluator's own `kabsch` rather than a second
    superposition routine, so the metric the width is fitted in is the metric
    the coupling is later evaluated in.
    """
    reference = np.asarray(reference, dtype=float)
    mobile = np.asarray(positions, dtype=float)[None]
    rotation, translation = kabsch(reference, mobile)
    moved = np.einsum("mni,mji->mnj", mobile, rotation) + translation[:, None, :]
    return float(np.sqrt(np.mean(np.sum((reference - moved[0]) ** 2, axis=-1))))


def fit_width(frames: list[Atoms], amplitude: float, eps: float = DEFAULT_EPS) -> float:
    """Width that quenches the coupling to `eps` at both endpoint geometries.

    Uses the endpoint nearer the transition state, so the condition holds at
    both.  Geometry only -- no reference energies required.
    """
    reference = frames[len(frames) // 2].positions
    nearest = min(
        rmsd(reference, frames[0].positions), rmsd(reference, frames[-1].positions)
    )
    if nearest <= 0.0:
        raise CouplingError(
            "an endpoint coincides with the transition state, so no width can "
            "switch the coupling off there"
        )
    magnitude = abs(amplitude) if amplitude != 0.0 else NOMINAL_AMPLITUDE
    return float(np.log(magnitude / eps) / nearest**2)


# ---------------------------------------------------------------------------
# rmsd
# ---------------------------------------------------------------------------


def fit_rmsd(
    frames: list[Atoms],
    diabatic_energies: tuple[float, float] | None = None,
    amplitude: float | None = None,
    eps: float = DEFAULT_EPS,
) -> Coupling:
    """Fit the RMSD coupling: the form that works for any channel with a saddle.

    Both numbers come from the three frames and nothing else:

      A   At the transition state RMSD is zero, so `V(x_TS) = A` exactly.  That
          makes the 2x2 secular equation invertible at that one geometry: given
          the two diabatic energies there and the reference barrier height, `A`
          follows in closed form.

      a   One point cannot set a width, but the endpoints can.  The coupling is
          a correction to the diabatic picture and must vanish where that
          picture is already correct, at the reactant and product minima.
          Requiring `|V| <= eps` at whichever endpoint is *closer* pins `a`.
    """
    transition, amplitude, provenance = _amplitude_for(
        frames, diabatic_energies, amplitude
    )
    width = fit_width(frames, amplitude, eps)
    # The RMSD is taken over every atom of the reacting system, so `atoms` here
    # is positional bookkeeping for the term format rather than a subset.
    indices = np.arange(len(transition))[None, :]
    return Coupling(
        terms={
            "rmsd": {
                "atoms": indices,
                "kwargs": {
                    "A": np.array([float(amplitude)]),
                    "a": np.array([float(width)]),
                },
            }
        },
        ensemble=np.array([transition.get_positions()]),
        provenance={"rmsd": provenance},
    )


def _amplitude_for(frames, diabatic_energies, amplitude):
    """The transition-state frame, the amplitude to use, and where it came from.

    An amplitude handed in by the caller is a stand-in, not a fit.  Recording
    which is which lets a surface report the channels it was built on unfitted
    couplings for, instead of inferring it from a magic value.  Zero is the
    honest stand-in: it contributes no stabilization, so the state is never
    admitted, where a large placeholder would drive the Hamiltonian with a
    number nobody fitted.
    """
    if len(frames) < 3:
        raise CouplingError(
            f"need at least reactant, transition state and product, got {len(frames)}"
        )
    transition = frames[len(frames) // 2]

    if amplitude is None:
        provenance = "fitted"
    elif amplitude == 0.0:
        provenance = "decoupled"
    else:
        provenance = "placeholder"

    if amplitude is None:
        if diabatic_energies is None:
            raise CouplingError(
                "fitting an amplitude needs the diabatic energies at the "
                "transition-state geometry; pass `amplitude` to skip the fit"
            )
        if transition.calc is None:
            raise CouplingError(
                "the transition-state frame carries no reference energy"
            )
        amplitude = fit_amplitude(*diabatic_energies, transition.get_potential_energy())
    return transition, amplitude, provenance


# ---------------------------------------------------------------------------
# threebody
# ---------------------------------------------------------------------------


def _triangle(positions: np.ndarray, triple) -> tuple[float, float, float]:
    """`(ra, rb, t)` of the transfer triangle: two bond lengths and the angle.

    `triple` is `(donor, moving, acceptor)` and `t` is the angle at the moving
    atom, in radians, matching `QForce`'s own `theta0` convention.
    """
    donor, moving, acceptor = triple
    va = positions[moving] - positions[donor]
    vb = positions[moving] - positions[acceptor]
    ra = float(np.linalg.norm(va))
    rb = float(np.linalg.norm(vb))
    if ra <= 0.0 or rb <= 0.0:
        raise CouplingError("the transferring atom coincides with a heavy atom")
    cosine = float(np.dot(va, vb) / (ra * rb))
    return ra, rb, float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _deviation(positions: np.ndarray, triple, centre) -> float:
    """`g`, the squared deviation `compute_threebody` measures, in A**2."""
    ra0, rb0, t0 = centre
    d0 = np.sqrt(ra0**2 + rb0**2 - 2.0 * ra0 * rb0 * np.cos(t0))
    ra, rb, t = _triangle(positions, triple)
    d = np.sqrt(ra**2 + rb**2 - 2.0 * ra * rb * np.cos(t))
    return (ra - ra0) ** 2 + (rb - rb0) ** 2 + (d - d0) ** 2


def fit_threebody(
    frames: list[Atoms],
    triple,
    diabatic_energies: tuple[float, float] | None = None,
    amplitude: float | None = None,
    eps: float = DEFAULT_EPS,
) -> Coupling:
    """Fit the transfer coupling of one atom-transfer channel.

    Same two questions as `fit_rmsd`, asked in the transferring atom's own
    internal coordinates instead of in the RMSD to a whole geometry:

      A   unchanged, and deliberately so.  A transfer has a saddle and a
          reference barrier, and that barrier is reference data worth fitting to
          -- the difference between this and `fit_twobody`, where there is no
          barrier to invert.  `compute_threebody` is centred on the transition
          state's own triangle, so `g = 0` and `V = A` exactly there, which is
          the one property `fit_amplitude` needs to stay valid.

      a   the width that quenches to `eps` at whichever endpoint is nearer the
          transition state, measured in `g` rather than in RMSD.

    **Only the width changes, and it is the width that was wrong.**  An RMSD is
    a tolerance on all `3N` coordinates at once, so it is dominated by whichever
    spectator happens to have moved.  In `g` the same channel is a far wider
    function of the coordinate that actually matters, because the spectators are
    no longer in the metric at all.
    """
    transition, amplitude, provenance = _amplitude_for(
        frames, diabatic_energies, amplitude
    )
    centre = _triangle(transition.get_positions(), triple)

    # The nearer endpoint, so the condition holds at both -- `fit_width`'s
    # choice, in `g`.  A transfer's two endpoints are generally *not*
    # equidistant here even when they are in RMSD, because `g` sees only the
    # moving atom and its two heavy atoms.
    nearest = min(
        _deviation(frames[0].get_positions(), triple, centre),
        _deviation(frames[-1].get_positions(), triple, centre),
    )
    if nearest <= 0.0:
        raise CouplingError(
            "an endpoint has the same transfer triangle as the transition state, "
            "so no width can switch the coupling off there"
        )
    magnitude = abs(amplitude) if amplitude != 0.0 else NOMINAL_AMPLITUDE
    width = float(np.log(magnitude / eps) / nearest)

    donor, moving, acceptor = triple
    return Coupling(
        terms={
            "threebody": {
                "atoms": np.array([[int(donor), int(moving), int(acceptor)]]),
                "kwargs": {
                    "A": np.array([float(amplitude)]),
                    "a": np.array([width]),
                    "ra0": np.array([float(centre[0])]),
                    "rb0": np.array([float(centre[1])]),
                    "t0": np.array([float(centre[2])]),
                },
            }
        },
        ensemble=np.array([transition.get_positions()]),
        provenance={"threebody": provenance},
    )


# ---------------------------------------------------------------------------
# twobody
# ---------------------------------------------------------------------------


def _separate(positions, moving, axis, distance) -> np.ndarray:
    """`positions` with `moving` rigidly translated `distance` along `axis`."""
    shifted = np.array(positions, dtype=float)
    shifted[list(moving)] += axis * distance
    return shifted


def find_crossing(
    positions: np.ndarray,
    pair,
    moving,
    diabats,
    max_separation: float = DEFAULT_MAX_SEPARATION,
    step: float = DEFAULT_SCAN_STEP,
) -> tuple[float, float, float]:
    """Where a fission's two diabats cross, and where its reactant sits.

    Returns `(r_min, r_cross, slope)` in A and eV/A: the bond length at which
    the *bonded* diabat is lowest along the path, the length at which
    `H_reactant - H_product` changes sign, and `d(gap)/dr` there.

    The path is a **rigid separation**: the fragment `moving` is translated
    along the breaking bond's axis and nothing relaxes.  That is the right path
    for this purpose and not a shortcut.  The gap is a difference of two diabats
    that share every nonbonded term and, for a fission, differ in exactly one
    bond term, so it is a function of that bond's length and of nothing else --
    relaxing the fragments would move both diabats by the same amount and leave
    the crossing where it is.
    """
    i, j = pair
    separation = positions[j] - positions[i]
    r_stored = float(np.linalg.norm(separation))
    if r_stored <= 0.0:
        raise CouplingError("the breaking bond's two atoms coincide")
    axis = separation / r_stored

    def both(r: float):
        return diabats(_separate(positions, moving, axis, r - r_stored))

    def gap(r: float) -> float:
        bonded, free = both(r)
        return bonded - free

    at_stored = gap(r_stored)
    if at_stored >= 0.0:
        raise CouplingError(
            f"the bonded diabat is already the higher one at the stored bond "
            f"length ({r_stored:.3f} A, gap {at_stored:+.4f} eV), so this "
            "channel has no bound reactant to dissociate from"
        )

    low, high = r_stored, None
    r = r_stored
    while r < max_separation:
        r = min(r + step, max_separation)
        if gap(r) >= 0.0:
            high = r
            break
        low = r
    if high is None:
        raise CouplingError(
            f"the two diabats do not cross within {max_separation:.1f} A "
            f"(gap {gap(max_separation):+.4f} eV there), so the bonded state is "
            "the lower one at every separation and no coupling can hand over to "
            "the dissociated one"
        )

    for _ in range(60):
        middle = 0.5 * (low + high)
        if gap(middle) < 0.0:
            low = middle
        else:
            high = middle
    crossing = 0.5 * (low + high)

    # Central difference over a tenth of the scan step.  The gap is a smooth
    # function of one variable here, so this is accurate to `h**2`.
    h = 0.1 * step
    slope = (gap(crossing + h) - gap(crossing - h)) / (2.0 * h)
    if slope <= 0.0:
        raise CouplingError(
            f"the diabatic gap is not increasing at the crossing "
            f"({slope:+.4f} eV/A), so the two states cross back"
        )

    inner = max(0.3, 0.5 * r_stored)
    grid = np.arange(inner, crossing, step)
    if len(grid) < 3:
        return r_stored, crossing, slope
    bonded = np.array([both(r)[0] for r in grid])
    k = int(np.argmin(bonded))
    if k == 0 or k == len(grid) - 1:
        # The minimum is not bracketed on this path -- a reactant frame already
        # outside the well.  The stored length is then the honest quench point.
        return r_stored, crossing, slope
    low, high = grid[k - 1], grid[k + 1]
    for _ in range(40):
        middle = 0.5 * (low + high)
        if both(middle + h)[0] < both(middle - h)[0]:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high), crossing, slope


def fit_twobody(
    positions: np.ndarray,
    pair,
    moving,
    diabats,
    eps: float = DEFAULT_EPS,
    bimol_cutoff: float = DEFAULT_BIMOL_CUTOFF,
    max_separation: float = DEFAULT_MAX_SEPARATION,
    step: float = DEFAULT_SCAN_STEP,
) -> Coupling:
    """Fit the bond-length coupling of one fission or recombination channel.

    Three numbers, each pinned by its own condition:

      r0  The crossing, from `find_crossing`.  At a crossing the diabats are
          degenerate, so the block's stabilization is exactly `|V|`: it is the
          one geometry where the coupling alone decides the answer, and the only
          defensible place to centre a Gaussian on a monotone path.

      A   The largest amplitude that introduces **no spurious minimum** at the
          crossing.  With the gap linearized as `g ~ s*u` (`u = r - r0`), the
          lower root `E = g/2 - hypot(g/2, V)` has `E'' = 2*a*|A| - s**2/(4*|A|)`
          at `u = 0`, so the curve turns from concave to convex -- acquiring a
          bound radical pair and a barrier to dissociation that should not exist
          -- at exactly `8*a*A**2 = s**2`.  Sitting on that boundary is the
          smoothest handover available.

      a   The width that quenches to `eps` at whichever end of the path is
          **nearer** the crossing, with `L = min(r0 - r_min, bimol_cutoff - r0)`.
          Inwards, a coupling at the reactant's own minimum is spurious
          stabilization an intact molecule collects from its own dissociation
          channel.  Outwards, past `bimol_cutoff` the other state is not in the
          basis at all, so a coupling still worth something there is
          stabilization the surface loses discontinuously.

    The last two are one system in two unknowns with exactly one solution.
    Eliminating `a` and writing `u = 2*c/A**2` with `c = (L*s)**2/8` turns them
    into `u + log(u) = log(2*c/eps**2)`, whose left side is strictly increasing
    from `-inf` to `+inf`.  (It is `u = W(2*c/eps**2)`; bisection avoids a
    `scipy.special` import for a one-dimensional monotone solve.)
    """
    minimum, crossing, slope = find_crossing(
        positions, pair, moving, diabats, max_separation, step
    )
    if not minimum < crossing < bimol_cutoff:
        raise CouplingError(
            f"the crossing ({crossing:.3f} A) is not between the reactant's "
            f"minimum ({minimum:.3f} A) and the bimolecular cutoff "
            f"({bimol_cutoff:.3f} A), so there is no interval to quench over"
        )
    length = min(crossing - minimum, bimol_cutoff - crossing)

    target = np.log(2.0 * (0.125 * (length * slope) ** 2) / eps**2)
    low, high = np.finfo(float).tiny, 1.0
    while high + np.log(high) < target:
        high *= 2.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if middle + np.log(middle) < target:
            low = middle
        else:
            high = middle
    u = 0.5 * (low + high)

    width = u / (2.0 * length**2)
    amplitude = -slope / np.sqrt(8.0 * width)
    if abs(amplitude) <= eps:
        raise CouplingError(
            f"the crossing ({crossing:.3f} A) leaves only {length:.3f} A to "
            f"quench over, so the widest admissible Gaussian has amplitude "
            f"{amplitude:+.2e} eV -- at or below eps ({eps:.1e})"
        )
    i, j = pair
    return Coupling(
        terms={
            "twobody": {
                "atoms": np.array([[int(i), int(j)]]),
                "kwargs": {
                    "A": np.array([float(amplitude)]),
                    "a": np.array([float(width)]),
                    "r0": np.array([float(crossing)]),
                },
            }
        },
        ensemble=np.array([np.asarray(positions, dtype=float)]),
        provenance={"twobody": "fitted"},
        # Which end of the path set the width, so a squeezed amplitude is
        # readable from the term file rather than having to be re-derived.
        limited_by={
            "twobody": (
                "cutoff" if bimol_cutoff - crossing < crossing - minimum else "reactant"
            )
        },
    )


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------


def fit(
    reaction,
    frames: list[Atoms],
    states,
    amplitude: float | None = None,
    eps: float = DEFAULT_EPS,
    bimol_cutoff: float = DEFAULT_BIMOL_CUTOFF,
) -> Coupling:
    """Fit the coupling for one reaction, in whichever form its channel takes.

    `frames` is `[reactant, transition state, product]` as
    `reaction.stationary_points` returns them, and `states` the pair of
    `Parameters` `reaction.state_parameters` builds -- reactant first.

    The diabatic energies are evaluated here rather than taken as an argument,
    through `evb.diabatic_energies`, which is the same code path the running
    surface puts on its diagonal.  An amplitude fitted against a diagonal
    nobody will evaluate reproduces a barrier nobody will see.
    """
    from .calculator import term_dict_for
    from .evb import diabatic_energies

    kind, spec = reaction.channel()

    if kind == "fission":
        pair, moving = spec
        bonded = 0 if reaction.broken else 1
        ordered = [states[bonded], states[1 - bonded]]
        term_dicts = [term_dict_for(p) for p in ordered]
        work = frames[0].copy()

        def diabats(positions):
            work.set_positions(positions)
            return diabatic_energies(work, ordered, term_dicts=term_dicts)

        return fit_twobody(
            frames[0].get_positions(),
            pair,
            moving,
            diabats,
            eps=eps,
            bimol_cutoff=bimol_cutoff,
        )

    energies = diabatic_energies(frames[len(frames) // 2], states)
    if kind == "transfer":
        return fit_threebody(frames, spec, energies, amplitude=amplitude, eps=eps)
    return fit_rmsd(frames, energies, amplitude=amplitude, eps=eps)
