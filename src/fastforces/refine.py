"""Rescale Morse well depths so a template's bonds carry its atomization energy.

`QForce` writes a bond as

    E = Dw * (1 - exp(-a * dr))**2 - D,    Dw = D + h,
                                          a  = sqrt(k / 2Dw)

which is -D at the minimum, so the well depths a molecule's bonds carry *are*
its atomization energy -- if they add up to it.  It is the per-bond asymptote
`h` (`bond_asymptote` where a bond states none) rather than zero at dissociation, which is what makes a bonded diabat and its own
fragments' diabat cross instead of converging; see that constant.
As fitted by q-force they do not: each bond is parameterized locally, and for
the HCombustion set the sum is off by -0.44 to +2.59 eV per template.

`ReactionSet` currently absorbs that difference into a constant `reference`
term.  That keeps the energy right at the equilibrium geometry but is wrong
everywhere else, and wrong in a way that matters here: the constant belongs to
the *template*, so a reactant diabat still carries the whole parent molecule's
shift at a geometry where its bond is nearly broken, while the dissociated
state has moved to the fragments' own templates and carries theirs instead.
The reference energy is therefore discontinuous across exactly the region an
EVB coupling has to describe, which is why a coupling fitted at the transition
state cannot reproduce a reference barrier.

Scaling the well depths instead puts the atomization energy where the
functional form already expects it.  The reference term then vanishes
identically, the dissociation limit is exact rather than offset, and the
reactant and product descriptions agree about the energy of a broken bond.

The scale factor is fixed by one condition per template -- its bonded energy at
its own reference geometry must equal its reference atomization energy -- so
there is nothing to choose and nothing to over-fit.  The cost is
transferability: an O-H depth that was one number across HO, H2O, HO2 and H2O2
becomes four.  Templates are stored and looked up independently, so this costs
nothing mechanically, but the depths are no longer a per-element-pair table.


Making the barriers reachable: the per-bond asymptote
-----------------------------------------------------

Scaling `D` fixes the minimum and the dissociation limit and leaves the shape in
between untouched, which is where the remaining error lives.  With `D` pinned by
the atomization energy, `r0` by the geometry and `k` by the vibrational
frequency, two-parameter Morse has nothing left, and it comes out *too deep* at
stretched geometries.  A reference barrier then lands below a diabat, and
`coupling.fit_amplitude` -- which has a real root only below *both* --
cannot fit the channel at all.

**The route that does not work.**  Since `a = sqrt(k / 2D)`, raising `k` steepens
the exponential and lifts the curve mid-range, and `mode="k"` still does exactly
that.  It buys the depth with the frequency, and the exchange rate is terrible:
before the repulsion was tapered, reaching even 17 of 19 channels needed H2 at
12402 cm^-1 against an experimental 4401, and the count saturated there.

**The route that does.**  Each bond carries its own asymptote height `h`, so the
well the exponential climbs is `Dw = D + h` on the stretched branch.  At fixed
`k`, raising `h` lifts every `dr > 0` monotonically towards the harmonic
`k dr**2 / 2` while the curvature at `dr = 0` stays `k`, and the curve stays a
Morse, so it cannot turn over.  `h` is bounded below by `bond_asymptote`, which
is also what a term file without `h` reads as.

`fit_force_constants` fits one `h` per distinct bond type (`asymptote`), the
force constants alone (`k`), or both (`asymptote-k`), always with
`fit_template` re-solved underneath it, so neither the atomization energy nor
the geometry is something the objective can spend.  The objective is a hinge:
once a margin is positive the barrier is reproduced exactly by the amplitude,
which is free per reaction, so overshooting buys nothing.

**This replaced a Hulburt-Hirschfelder shape term** `c s**3 exp(-b s)`, fitted
as a `(c, b)` pair per bond under a monotonicity bound `c <= c_max(b)`.  From
the same q-force start on HCombustion (2026-09-22) both routes make 19 of 19
margins positive at the frequencies q-force fitted; `h` does it with one
parameter per bond, no bound to enforce, and in 16 s against the pair's 650.
The two lift different parts of the curve -- the shape term mid-range, `h` out
towards dissociation -- and the channels that needed lifting at all are the
fissions, whose couplings `coupling.fit_twobody` places at the diabatic
crossing rather than inverting a barrier.  Plain Morse (`h = bond_asymptote`
everywhere) fits all 19 couplings end to end; what `h` moves is where those
crossings sit, 2.1-2.4 A in to 1.8-2.0.


Where the minimum is: `r0` against a repulsion that is not zero there
---------------------------------------------------------------------

The paragraph above says `r0` is pinned by the geometry, and while the bonded
terms were the whole molecular potential that was true by construction: q-force
fitted `r0` to the geometry, so the bonded minimum sat on it and nothing had to
be solved.  `ZBL` ended that.  It is a real repulsion at bonding distances --
2.0 eV at the H2 bond length, 5.4 at O-H, 11.6 at O-O, with slopes to match --
and the *total* is what has a minimum, so the Morse has to lean into it.

Nothing made it.  `fit_dissociation_energies` matched each template's energy at
its stored QM geometry, exactly, to 1e-13 -- and nothing anywhere looked at the
gradient there.  The templates came out with the right energies at geometries
they were not at rest in, and relaxed away from them: H2 by 0.105 A, HO2's O-O
by 0.825, every stretching frequency 1.7 to 2.6x experiment.  H2's own minimum
landed *outside* the bond-perception radius, so a relaxed H2 re-perceived as two
free atoms.

`fit_bond_lengths` adds the missing condition -- one equation per bond type, the
total force along it vanishing at the reference geometry -- and `fit_template`
alternates it with the depth solve until both hold.  Both are determinate, so
neither is fitted and neither competes with the barriers.

It is also not always solvable, which is worth stating plainly: a Morse pulls at
most `D*a/2`, and `ZBL` pushes O-O in HO2 apart with 25.0 eV/A against a ceiling
of 8.3 at q-force's own force constants, measured against the untapered
repulsion.  The ceiling is `sqrt(k Dw / 8)`, so it rises with `D`, `k` and `h`.
Since `ZBL` was tapered no HCombustion bond is near it: the solved `r0` sits
within 0.008 A of the reference bond length.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms, units
from ase.data import atomic_masses, atomic_numbers
from scipy.optimize import brentq, minimize

from DynamicTopology.core.types import Term
from DynamicTopology.forcefield.evaluate import nonbonded
from DynamicTopology.forcefield.params import active
from DynamicTopology.forcefield.qforce import QForce

logger: logging.Logger = logging.getLogger(__name__)

# Bracket for the scale factor.  Wide enough for any sane reparameterization;
# a root outside it means the reference energy and the force field disagree
# about the molecule, not that the bracket is too tight.
SCALE_BRACKET: tuple[float, float] = (0.05, 20.0)

# How far below the reference barrier the lower diabat has to sit before the
# channel counts as fitted.  Small on purpose: once the margin is positive the
# barrier is reproduced *exactly* by the coupling amplitude, which is free per
# reaction, so overshooting buys nothing and costs frequency.  It also keeps
# |A| small, and a large amplitude is a coupling that reaches geometries it
# should not.
DEFAULT_MARGIN: float = 0.02

# Weight on log(k-scale)**2 in the objective, i.e. how hard the fit is pulled
# back towards q-force's force constants.  The soft half of the trade.  Light,
# because the hard bound below is doing most of the work: the hinge stops
# pushing on its own as soon as a channel is feasible.
DEFAULT_FREQUENCY_WEIGHT: float = 0.005

# Hard bound on each k-scale; the cap on frequency drift is its square root.
# Only read by `mode="k"` and `mode="asymptote-k"` -- the default `asymptote`
# mode leaves every force constant where q-force put it.  Set it to 1.0 to
# freeze the force constants, which in `mode="k"` is the vacuity check.
#
# Two -- 1.41x in wavenumbers -- was the knee of a sweep over the retired
# `(c, b)` shape term's `both` mode, where it took HCombustion from 13 to 18 of
# 19 channels.  Since the per-bond asymptote reaches 19 of 19 with the force
# constants frozen, nothing currently needs it: `asymptote-k` from the same start
# spends up to 1.19x in `k` (H2 at 4021 cm^-1) for margins the hinge does not
# value.
#
# **No nonbonded term moves the margins, and one of them provably cannot.**  A
# Lennard-Jones with per-state exclusions was added on the hypothesis that Pauli
# repulsion would lift the diabats at the transition states and make the channels
# fittable for free, and the numbers looked emphatic: 14 of 19 feasible with no
# refitting whatever.  All of it was artefact -- a diabat that had broken a bond
# called its two atoms different molecules while they sat at the bond length, and
# the whole-system sum charged them 727 to 1550 eV there.  Excluding those pairs
# properly returned the count to 1 of 19.
#
# The repulsion is now `ZBL`, which takes no topology at all, so it is the *same
# number* on every diabat of a block and cancels exactly out of every margin this
# module computes.  It cannot help here and it cannot hurt here, by construction.
# The overbinding this fit exists to repair is in the Morse form, and the bonded
# parameters are the only thing that can pay for it.
#
# Note that this bounds the scale relative to whatever `k` the templates handed
# in already carry, not relative to q-force's original fit, so **running the fit
# twice over its own output compounds the bound**.  One pass is the intended
# use; the shipped parameters are one pass, from q-force's own values.
DEFAULT_MAX_SCALE: float = 2.0


# Weight on the leftover force at each template's reference geometry, in
# 1/(eV/A)**2, i.e. how hard the objective insists that a molecule be at rest
# where its reference energy says it is.
#
# It is a penalty rather than a constraint, and that is a deliberate second
# attempt.  Treating it as a constraint -- `score` returning `inf` wherever
# `fit_bond_lengths` had no solution -- is exactly right on paper and
# catastrophic in practice: at plain Morse, q-force's own force constants and
# the then-untapered `ZBL`, most of these bonds could not cancel it at any length, so the infeasible set is
# most of the box, and Powell line-searching across a plateau of infinities
# turned a 40 second fit into one that ran for half an hour without converging.
# A quadratic penalty puts the same pressure on a surface the optimizer can
# actually descend.
#
# 10.0 makes a 0.3 eV/A leftover force -- the worst any HCombustion template
# shows -- cost about as much as a 1 eV margin shortfall, so the geometry is
# worth roughly one channel.  That is the intended exchange rate: a template
# that cannot sit still is a worse defect than a barrier that cannot be fitted,
# but not by so much that the fit will spend every force constant it has to buy
# the last milli-eV per Angstrom.
DEFAULT_GEOMETRY_WEIGHT: float = 10.0


# Wavenumber (cm^-1) above which a stretching mode starts costing the
# objective.  This is the timestep, expressed as a property of the force field:
# velocity Verlet wants ~15 steps per vibrational period, so `dt` femtoseconds
# needs every mode under `33356 / (15 dt)`, which is 4450 at the 0.5 fs the
# production sweep is trying to reach.  4400 is that, rounded down to H2's own
# experimental stretch so the cap is a real number rather than a derived one.
#
# Nothing priced this before, and the result was a surface with an 11735 cm^-1
# mode on it -- a 2.84 fs period, which is what pinned the sweep at 0.05 fs.
# The stiffness was coming not from the `k`-scale this fit reports, which was
# bounded at 1.41x, but from the retired shape term at an `r0` displaced inside
# the bond.  Neither survives: `h` adds no curvature at `dr = 0` and `r0` now
# sits on the bond, so the asymptote fit leaves every mode where q-force put it
# (fastest 3748 cm^-1, H2) and this cap does not bind.
DEFAULT_MAX_WAVENUMBER: float = 4400.0

# Weight on `max(0, nu - max_wavenumber)**2`, in 1/cm**-2.
#
# A hinge and not a bound, for the reason `DEFAULT_GEOMETRY_WEIGHT` records:
# a hard constraint here returned `inf` over most of the box at plain Morse,
# and Powell line-searching a plateau of infinities is what turned a 40 second
# fit into an hour-long one last time.
#
# 1e-2 was chosen by a sweep over the retired shape term's `both` mode, where
# it was the middle of a plateau three decades wide: every bond type under a
# 4400 cap, no channel lost against the uncapped fit, and the timestep up 2.7x
# (0.190 fs to 0.506).  Higher was not better -- at 1e-1 the penalty distorted
# the search before it bound any harder.  In the default `asymptote` mode the
# force constants are frozen and nothing is over the cap, so the weight only
# matters for `k` and `asymptote-k`.
DEFAULT_CURVATURE_WEIGHT: float = 1e-2

# Stateless, and constructed once: the outer fit calls `bonded_energy`
# thousands of times.
_QFORCE = QForce(bond_form="morse")


class DissociationFitError(ValueError):
    """Raised when no scaling reproduces the reference atomization energy."""


def _scaled(terms: list[Term], scale: float) -> list[Term]:
    """Copy of `terms` with every Morse depth scaled and the shift zeroed.

    The zero shift is stated explicitly even when the input had no `reference`
    term.  ReactionSet synthesizes one from `E_atomization + sum(D)` for any
    template that does not carry it, and after this fit that formula no longer
    evaluates to zero -- the depths were solved against the *full* bonded
    energy, so the leftover is the angle and cross-term contribution at the
    reference geometry.  Writing the zero down keeps the loader from adding
    that leftover a second time.
    """
    out: list[Term] = []
    seen_reference = False
    for term in terms:
        if term["type"] == "bond":
            kwargs = dict(term["kwargs"])
            kwargs["D"] = kwargs["D"] * scale
            out.append({**term, "kwargs": kwargs})
        elif term["type"] == "reference":
            seen_reference = True
            out.append({**term, "kwargs": {**term["kwargs"], "E0": 0.0}})
        else:
            out.append(term)
    if not seen_reference:
        out.append({"type": "reference", "atoms": {"a1": 0}, "kwargs": {"E0": 0.0}})
    return out


# Memo for the nonbonded half of a template's energy and forces, keyed by the
# geometry and the ACKS2 parameters it was computed from.
#
# **Why this is safe, and why it matters.**  All three nonbonded terms are
# functions of the geometry alone -- `ZBL` reads only atomic numbers, `ACKS2`
# reads the `atom` terms and `LennardJones` the `lennardjones` terms, and no
# part of this module fits either.  The fit moves `D`, `r0`, `k` and `h`, every
# one of them bonded.  So across an entire
# `fit_force_constants` run, at a fixed template geometry, this pair of numbers
# never changes.
#
# It was being recomputed for every one of them: `fit_dissociation_energies`
# runs `brentq` to 1e-12, which is ~40 evaluations of `bonded_energy`, each
# solving the ACKS2 charge equilibration from scratch, and that happens once per
# round per template per objective evaluation.  Caching it is most of the
# difference between a fit that takes a minute and one that takes an hour.
#
# The key includes the ACKS2 parameters rather than trusting the argument
# above, so a caller that *did* fit them would miss the cache rather than read
# a stale number from it.
_NONBONDED_CACHE: dict[tuple, tuple[float, np.ndarray]] = {}


def _nonbonded_key(atoms: Atoms, term_dict: dict) -> tuple:
    def _params(name: str) -> tuple:
        block = term_dict.get(name, {})
        return (
            tuple(
                (key, np.asarray(value).tobytes())
                for key, value in sorted(block.get("kwargs", {}).items())
            ),
            np.asarray(block.get("atoms", ())).tobytes(),
        )

    return (
        atoms.positions.tobytes(),
        atoms.numbers.tobytes(),
        atoms.cell.array.tobytes(),
        tuple(atoms.pbc),
        # The global parameters, because all three nonbonded terms read them:
        # two taper radii, the core fraction and the smearing width.  A fit run
        # under `params.use(...)` must miss the cache rather than read the
        # previous surface's answer out of it.
        active(),
        # Both parameter sets the memoized terms read, for the reason above:
        # neither is fitted here, but a caller that fitted one should miss the
        # cache rather than read a stale number out of it.
        _params("atom"),
        _params("charge"),
        _params("lennardjones"),
        # And the exclusions, which `nonbonded_curvatures` now reads.  Without
        # them a raw `.jsonl` and the same list `with_exclusions` has been
        # applied to hash identically and the second caller reads the first
        # one's answer -- which is the whole-system sum, uncorrected.
        _params("exclusion"),
        _params("zblexclusion"),
        # Which pairs the Coulomb sum drops.  Under `pointcharge` two diabats of
        # one reaction at one geometry differ in exactly this and in `charge`,
        # so without both the product diabat would read the reactant's energy.
        _params("coulombexclusion"),
    )


def _nonbonded(atoms: Atoms, term_dict: dict) -> tuple[float, np.ndarray]:
    """ACKS2 + ZBL + 12-6 energy and forces at `atoms`, memoized on the geometry.

    All three of the terms `System.calculate` adds outside the EVB, and for the
    same reason: this has to be the *same* sum, or a template stops reproducing
    its own reference energy through the calculator that it was fitted through.
    The 12-6 was absent here for as long as it was absent there.
    """
    key = _nonbonded_key(atoms, term_dict)
    hit = _NONBONDED_CACHE.get(key)
    if hit is None:
        energy, forces, _ = nonbonded(atoms, term_dict)
        hit = _NONBONDED_CACHE[key] = (energy, forces)
    return hit


def bonded_energy(atoms: Atoms, terms: list[Term]) -> float:
    """Total energy of `terms` at `atoms`' geometry, in eV.

    Bonded *and* nonbonded, because that is the sum `System.calculate` reports
    and therefore the sum a reference atomization energy has to be matched
    against.  Fitting the depths against the bonded part alone left every
    heteronuclear template overbound by exactly its own ACKS2 energy -- water at
    -12.3701 eV against a reference of -9.8735, 25% too deep -- with the error
    invisible on H2 and O2, which have no charge separation and so no nonbonded
    energy at all.

    The name is now a slight lie, kept because it is the vocabulary the rest of
    this module and `refit` are written in; DynamicTopology's `tests/test_reference_
    energies.py` is what pins the meaning.
    """
    from DynamicTopology.core.topology import Topology

    topology = Topology.from_terms(terms, atoms)
    energy = _QFORCE(atoms.positions, atoms.pbc, atoms.cell, topology.term_dict)[0]
    return energy + nonbonded_energy(atoms, topology.term_dict)


def nonbonded_energy(atoms: Atoms, term_dict: dict) -> float:
    """The electrostatics, whole-system ZBL and switched 12-6, in eV.

    DynamicTopology's `evaluate.nonbonded`: everything `evaluate` adds on top of
    `QForce`.  The ZBL and 12-6 sums read no topology, and their intramolecular
    exclusions are ordinary `QForce` terms; the electrostatics carries its own
    Coulomb exclusion screen and, under `pointcharge`, the template's charges --
    so this half is per diabat, which is why `_nonbonded_key` keys on both.

    This is the sum a reference atomization energy has to be matched against.
    Fitting the Morse depths against the bonded part alone left every
    heteronuclear template overbound by exactly its own ACKS2 energy -- water at
    -12.3701 eV against a reference of -9.8735, 25% too deep -- and the error was
    invisible on H2 and O2, which have no charge separation.  `ZBL` closes that
    hole for the homonuclear templates too: it is nonzero on every bonded pair
    (+2.0 eV at the H2 bond length, +11.6 at O2's), so leaving it out here would
    reintroduce the same class of error on exactly the two templates the old
    version of this bug hid behind.  The 12-6 is the smallest of the three at a
    bond length by design -- `lj.switch` holds it to 0.031 eV on an O-H and
    0.35 eV on O2 -- but it is not zero, and it is not zero at the *stretched*
    geometries either, which is where it reaches the couplings.
    """
    return _nonbonded(atoms, term_dict)[0]


def fit_dissociation_energies(atoms: Atoms, terms: list[Term]) -> list[Term]:
    """Scale one template's Morse depths to match its atomization energy.

    Args:
        atoms: the molecule template, carrying its reference atomization energy
            (eV, referenced to free atoms, so exactly zero for a lone atom).
        terms: the template's force field terms.

    Returns:
        The terms with every bond's `D` scaled by a single factor and the
        constant `reference` shift set to zero.  Templates with no bonds are
        returned unchanged.
    """
    bonds = [t for t in terms if t["type"] == "bond"]
    if not bonds:
        # A free atom: its atomization energy is zero by definition and there is
        # no well depth to carry it.
        return list(terms)

    if atoms.calc is None:
        raise DissociationFitError(
            "template carries no reference atomization energy; run "
            "`fast-forces label` over the molecule files first"
        )
    target = atoms.get_potential_energy()

    def residual(scale: float) -> float:
        return bonded_energy(atoms, _scaled(terms, scale)) - target

    low, high = SCALE_BRACKET
    f_low, f_high = residual(low), residual(high)
    if f_low * f_high > 0.0:
        raise DissociationFitError(
            f"no scale factor in [{low}, {high}] reproduces the reference "
            f"atomization energy {target:+.4f} eV: the bonded energy runs from "
            f"{f_low + target:+.4f} to {f_high + target:+.4f} eV over that range. "
            "The template's geometry, its parameters, or its reference energy "
            "disagree with each other."
        )

    scale = brentq(residual, low, high, xtol=1e-12, rtol=1e-14)
    return _scaled(terms, scale)


# --------------------------------------------------------------------------
# Bond lengths
# --------------------------------------------------------------------------

# Half-width, in Angstrom, of the window `fit_bond_lengths` searches around the
# `r0` a template arrived with.  It is a tripwire rather than a tuning parameter:
# the shifts this actually needs are 0.05 to 0.14 A, so a solve that wants
# more than 0.3 has gone somewhere it should not, and the caller should be
# told rather than handed a molecule with a 3 Angstrom bond in it.
MAX_LENGTH_SHIFT: float = 0.3

# How many times `fit_template` alternates the `r0` solve with the `D` solve.
# Each is exact given the other and they couple only through `a = sqrt(k/2D)`,
# so the alternation contracts by about an order of magnitude a round.  Worst
# residual force left on any HCombustion template, in eV/A:
#
#     rounds   1        2        3        4
#              1.5e-3   1.2e-4   1.9e-5   1.9e-6
#
# Iterating the `r0` solve *within* a round instead makes it worse -- 3.0e-3,
# 5.6e-4, 1.1e-4, 2.0e-5 for the same work -- because the extra passes refine
# towards a fixed point of a stale `D`.  So `fit_bond_lengths` is one pass and
# the alternation is here, which is both twice as fast and ten times as
# accurate as looping in both places.
LENGTH_DEPTH_ROUNDS: int = 4


def _morse_stretch_force(r, D: float, r0, k: float, h: float | None = None):
    """`-dE/dr` of one Morse bond, in eV/A.

    Positive is the force pulling the two atoms *apart*, i.e. the sign a
    compressed bond carries.  This mirrors `QForce._bond_morse` exactly,
    including the one-sided asymptote; it is written out a second time here
    because the fit needs the derivative of a single bond as a function of `r0`
    with everything else held still, and the force field only ever offers the
    assembled Cartesian forces of a whole system.

    `r` and `r0` broadcast, and the search below depends on it.  This is the
    innermost thing in `fit_force_constants` -- one call per grid point, per
    bond type, per round, per objective evaluation, and Powell's evaluation
    count runs to five figures -- so scanning a bond length has to be one array
    expression rather than four hundred scalar ones.  Left as a scalar loop it
    turned a fit that took seconds into one that took the better part of an
    hour.
    """
    dr = np.asarray(r, dtype=float) - np.asarray(r0, dtype=float)
    # `Dw`, not `D`: the well the exponential climbs is the depth plus the
    # asymptote, on the stretched branch only.  See `params.bond_asymptote`;
    # `h` is the per-bond override `QForce._bond_morse` accepts.
    if h is None:
        h = active().bond_asymptote
    Dw = np.where(dr > 0.0, D + h, D)
    al = np.sqrt(k / (2 * Dw))
    exp_term = np.exp(-al * dr)
    de_dr = 2 * Dw * (1 - exp_term) * al * exp_term
    return -de_dr


def bond_lengths(terms: list[Term], atoms: Atoms) -> list[list[float]]:
    """Actual bond distance in Angstrom of every bond, in `bond_types` order.

    Two bonds share a type when q-force gave them identical `(r0, k)`, which
    does not make them the same length -- H2O2's two O-H bonds are one type and
    happen to be symmetric, but nothing guarantees that in general.
    """
    types = bond_types(terms)
    lengths: list[list[float]] = [[] for _ in types]
    for term in terms:
        if term["type"] != "bond":
            continue
        i, j = list(term["atoms"].values())
        index = types.index((term["kwargs"]["r0"], term["kwargs"]["k"]))
        lengths[index].append(float(atoms.get_distance(i, j)))
    return lengths


def set_bond_lengths(terms: list[Term], values: list[float]) -> list[Term]:
    """Copy of `terms` with each bond type's `r0` set, in `bond_types` order."""
    order = {pair: index for index, pair in enumerate(bond_types(terms))}
    out: list[Term] = []
    for term in terms:
        if term["type"] != "bond":
            out.append(term)
            continue
        kwargs = dict(term["kwargs"])
        kwargs["r0"] = float(values[order[(kwargs["r0"], kwargs["k"])]])
        out.append({**term, "kwargs": kwargs})
    return out


def stretch_forces(atoms: Atoms, terms: list[Term]) -> list[float]:
    """Net force along the bonds of each type, in eV/A, in `bond_types` order.

    For a bond `(i, j)` this is `0.5 * (F_i - F_j) . u_ij` summed over the
    bonds of the type -- the part of the total force that a change in that
    type's `r0` can move, and nothing else.  `F` is the *whole* force:
    every bonded term, ACKS2 and ZBL, through the same pipeline
    `System.calculate` uses, so no contribution is assumed away.
    """
    from DynamicTopology.core.topology import Topology

    topology = Topology.from_terms(terms, atoms)
    term_dict = topology.term_dict
    forces = _QFORCE(atoms.positions, atoms.pbc, atoms.cell, term_dict)[1]
    forces = forces + _nonbonded(atoms, term_dict)[1]

    types = bond_types(terms)
    out = [0.0] * len(types)
    for term in terms:
        if term["type"] != "bond":
            continue
        i, j = list(term["atoms"].values())
        index = types.index((term["kwargs"]["r0"], term["kwargs"]["k"]))
        unit = atoms.positions[j] - atoms.positions[i]
        unit = unit / np.linalg.norm(unit)
        out[index] += 0.5 * float(np.dot(forces[j] - forces[i], unit))
    return out


def bond_curvatures(atoms: Atoms, terms: list[Term]) -> list[float]:
    """Second derivative along each bond type, in eV/A**2, in `bond_types` order.

    The *total*, not the Morse's own: measured by displacing the two atoms of
    each bond along their axis and differencing the assembled forces, so ZBL
    and ACKS2 are in it.  Averaged over the bonds of a type, which is what a
    per-type wavenumber can mean at all.

    This exists because `frequency` does not answer the question any more.  It
    converts a bonded force constant to a wavenumber, and while the bonded
    terms were the whole potential that was the frequency.  `ZBL`'s curvature
    at a bond length is not small next to a bond's -- 68.5 eV/A**2 at the O-H
    distance, which on its own is 4431 cm^-1 -- so a report built on the bonded
    `k` alone understates the real stiffness by about a factor of two, and the
    force-constant fit's headline "drift" number was understating it by that
    much.
    """
    from DynamicTopology.core.topology import Topology

    topology = Topology.from_terms(terms, atoms)
    term_dict = topology.term_dict
    qforce = _QFORCE

    def stretch(i: int, j: int, delta: float) -> float:
        """`-dE/dr` of the whole system with bond `(i, j)` stretched by `delta`."""
        moved = atoms.copy()
        unit = atoms.positions[j] - atoms.positions[i]
        unit = unit / np.linalg.norm(unit)
        moved.positions[j] = moved.positions[j] + delta * unit
        forces = qforce(moved.positions, moved.pbc, moved.cell, term_dict)[1]
        forces = forces + _nonbonded(moved, term_dict)[1]
        return float(np.dot(forces[j], unit))

    # 1e-4, not the 1e-3 a central difference would otherwise want.  The Morse
    # switches branch at `dr = 0` (`Dw = D + h` only when stretched), and its
    # third derivative jumps there, so a bond sitting exactly there has a
    # different cubic on each side of the stencil and the difference picks up
    # an O(step) contribution that the true second derivative does not have.
    #
    # A bond sits exactly there whenever `fit_bond_lengths` can solve its
    # geometry condition exactly, which after `forcefield/exclusions.py` is
    # every homonuclear diatomic: the intramolecular nonbonded is removed
    # entirely, so the Morse is the only force along the bond and `r0` lands on
    # the bond length.  Measured on O2 against the analytic 217.856:
    #
    #     step 1e-2   224.748   +6.892
    #     step 1e-3   218.594   +0.738
    #     step 1e-4   217.930   +0.074
    #     step 1e-5   217.863   +0.007
    #
    # (measured with the retired shape term, which was clamped at the same
    # point).  Linear in the step, as an artifact of straddling the join has to
    # be.
    # 1e-4 puts it under a tenth of an eV/A**2 -- 0.03% on O2, a few cm^-1 on
    # the wavenumber this feeds -- at no cost, since this function is called for
    # the report and the tests and never from the objective's inner loop, which
    # uses `stretch_curvatures`.
    step = 1e-4
    types = bond_types(terms)
    totals = [0.0] * len(types)
    counts = [0] * len(types)
    for term in terms:
        if term["type"] != "bond":
            continue
        i, j = list(term["atoms"].values())
        index = types.index((term["kwargs"]["r0"], term["kwargs"]["k"]))
        totals[index] += -(stretch(i, j, step) - stretch(i, j, -step)) / (2 * step)
        counts[index] += 1
    return [t / max(n, 1) for t, n in zip(totals, counts)]


def _bonded_curvature(kwargs: dict, r: float) -> float:
    """`d2E/dr2` of one Morse bond at separation `r`, in eV/A**2.

    Analytic, and written out here for the same reason `_morse_stretch_force`
    is: the objective needs the second derivative of a *single* bond as a
    function of that bond's parameters, thousands of times, and the force field
    only offers assembled Cartesian forces of a whole system.  Differentiating
    `QForce._bond_morse` twice,

        d2/dr2 [ Dw (1 - exp(-a dr))**2 ]  =  2 Dw a**2 exp(-a dr) (2 exp(-a dr) - 1)

    which is `k` at `dr = 0` whatever `Dw` is.  Away from it the asymptote does
    move the curvature, and `fit_bond_lengths` can put a bond there: `r0` is
    solved against the repulsion rather than set to the bond length.  With the
    tapered `ZBL` the displacement is under 0.008 A on every HCombustion bond,
    so this is `k` to within a few cm^-1; before the taper it was 0.04-0.22 A,
    and the retired shape term's curvature at that displacement was most of the
    stiffness (509 eV/A**2 of O2's 793).

    `r` in Angstrom; `kwargs` in eV and Angstrom, like every term in memory.
    """
    D = kwargs["D"]
    k = kwargs["k"]
    dr = r - kwargs["r0"]
    # `Dw`, not `D`, and on the stretched branch only.  The curvature *at the
    # minimum* is `k` either way, which is what makes the asymptote free of the
    # fitted frequencies -- but `fit_bond_lengths` displaces `r0` *inside* the
    # reference bond length, so the bond sits at `dr > 0` and this is exactly
    # where the two differ.  See `params.bond_asymptote`.
    h = kwargs.get("h", active().bond_asymptote)
    Dw = D + h if dr > 0.0 else D
    al = np.sqrt(k / (2 * Dw))

    exp_term = np.exp(-al * dr)
    curvature = 2 * Dw * al * al * exp_term * (2 * exp_term - 1)
    return float(curvature)


# Nonbonded curvature per bond type, memoized exactly like `_NONBONDED_CACHE`
# and for a stronger reason: it is a function of the geometry and the ACKS2
# parameters alone, and neither moves during a force-constant fit.  One
# measurement per template covers every objective evaluation.
_CURVATURE_CACHE: dict[tuple, list[float]] = {}


def nonbonded_curvatures(atoms: Atoms, terms: list[Term]) -> list[float]:
    """ACKS2 + `ZBL` second derivative along each bond type, in eV/A**2.

    The half of `bond_curvatures` the fit cannot change.  Split out because the
    other half is analytic and this one is not: measuring it costs two assembled
    force calls per bond, and the objective is evaluated four figures of times.

    Averaged over the bonds of a type, in `bond_types` order, which is what a
    per-type wavenumber can mean at all.
    """
    from DynamicTopology.core.topology import Topology

    topology = Topology.from_terms(terms, atoms)
    term_dict = topology.term_dict

    # The exclusions belong to this half of the split, not to the Morse's.
    # `_nonbonded` sums `ZBL`, the 12-6 and ACKS2 over every pair with no
    # reference to the bond graph -- that is the property that puts those three
    # outside the EVB at all -- and the intramolecular correction to them is
    # carried as ordinary `QForce` terms, which is what `bond_curvatures` picks
    # up by assembling the whole force field and this function did not.
    #
    # Leaving it out charged the objective an intramolecular `ZBL` the
    # calculator does not apply: 33.4 eV/A**2 on H2's own bond, against a
    # *total* excluded curvature of 26.2.  H2 scored 5674 cm^-1 where
    # `bond_curvatures` -- and `report_force_constants`, which reads it --
    # reported 3762, so the objective spent every force constant it was allowed
    # to spend pulling a wavenumber under a cap it was never over.  Every bond
    # type in both datasets ran to the `--max-k-scale` floor, and the report
    # said the cap was satisfied while it happened.
    #
    # `coulombexclusion` is deliberately not here: it is not a `QForce` term at
    # all (see `forcefield/exclusions.py` on why Coulomb cannot be subtracted
    # additively), so `bond_curvatures` does not see it either and this stays
    # the same quantity.
    #
    # Nothing the fit varies appears in any of them, so they are exactly as
    # frozen as the rest of this measurement and stay on the cached path.
    excluded = [term for term in terms if term["type"].endswith("exclusion")]
    exclusion_dict: dict = {}
    if excluded:
        exclusion_topology = Topology.from_terms(terms, atoms)
        exclusion_topology.set_terms(excluded)
        exclusion_dict = exclusion_topology.term_dict
    qforce = _QFORCE

    types = bond_types(terms)
    grouping: list[tuple[int, int]] = []
    for term in terms:
        if term["type"] != "bond":
            continue
        i, j = list(term["atoms"].values())
        grouping.append(
            (types.index((term["kwargs"]["r0"], term["kwargs"]["k"])), i, j)
        )

    # The `(r0, k)` a type is *named* by moves as the fit runs; which bonds are
    # grouped together does not.  Key on the grouping, so the cache cannot be
    # read across a genuinely different partition.
    key = (_nonbonded_key(atoms, term_dict), tuple(grouping))
    hit = _CURVATURE_CACHE.get(key)
    if hit is not None:
        return hit

    def stretch(i: int, j: int, delta: float) -> float:
        """`-dE_nonbonded/dr` with bond `(i, j)` stretched by `delta`."""
        moved = atoms.copy()
        unit = atoms.positions[j] - atoms.positions[i]
        unit = unit / np.linalg.norm(unit)
        moved.positions[j] = moved.positions[j] + delta * unit
        forces = _nonbonded(moved, term_dict)[1]
        if exclusion_dict:
            forces = (
                forces
                + qforce(moved.positions, moved.pbc, moved.cell, exclusion_dict)[1]
            )
        return float(np.dot(forces[j], unit))

    step = 1e-3
    totals = [0.0] * len(types)
    counts = [0] * len(types)
    for index, i, j in grouping:
        totals[index] += -(stretch(i, j, step) - stretch(i, j, -step)) / (2 * step)
        counts[index] += 1

    hit = [t / max(n, 1) for t, n in zip(totals, counts)]
    _CURVATURE_CACHE[key] = hit
    return hit


def stretch_curvatures(
    atoms: Atoms, terms: list[Term], nonbonded: list[float] | None = None
) -> list[float]:
    """Per-bond-type stretch stiffness in eV/A**2: own Morse plus nonbonded.

    The cheap stand-in for `bond_curvatures` that the objective can afford --
    one analytic expression plus a cached measurement, against two assembled
    force calls per bond -- and deliberately *not* the same quantity.

    `bond_curvatures` differences the whole potential along "displace atom `j`
    along the `ij` axis", so it also picks up every angle, dihedral and *other*
    bond term that touches `j`.  This one takes the bond's own Morse and the
    nonbonded terms and stops.  Measured against it on the shipped parameters:

        template  bond   bond_curvatures   this   difference
        H2        H-H             33.661   33.660     0.001
        O2        O-O            472.040  472.042    -0.003
        HO        O-H             60.366   60.365     0.000
        H2O       O-H             61.731   61.729     0.002
        HO2       O-O            417.618  417.609     0.010
        HO2       O-H             59.628   59.628     0.000
        H2O2      O-O            540.200  518.653    21.547
        H2O2      O-H             64.909   64.907     0.002

    One row differs and it is the one row that can: H2O2's O-O is the only bond
    here whose displaced atom carries both another bond and a dihedral.  Most of
    the 21.5 is the O-H Morse on the moved oxygen; the rest is the angle and
    dihedral terms.

    **The argument for tolerating it has changed, and it is weaker than it was.**
    It used to be that the cap binds only on X-H stretches -- the rows that agree
    to 1e-3 -- while this row was a 2967 cm^-1 heavy-atom mode nowhere near a cap
    of 4400.  Re-enabling `forcefield/lj.py` ended that: the 12-6 is 0.30 eV at
    H2O2's 1.45 A O-O and steeply varying there, the fit answered by stiffening
    the bond, and the mode is now **4285 cm^-1** -- the second fastest in the
    dataset, and the cap does bind on it.

    What is left is that the error is small in the units the cap is stated in:
    4198.9 cm^-1 here against 4285.2 measured, 86.3 cm^-1 or 2.0%.  So a
    `--max-wavenumber 4200` is enforced against a number 86 cm^-1 low and the
    timestep that follows is 0.529 fs where the truth is 0.515.  Both are covered
    by the 0.5 fs the production sweeps use, which is the margin this is trading
    on; `tests/test_fit.py` bounds the gap so that trade stays visible.  Anything
    that wants the real number should call `bond_curvatures`, which is what the
    report does.
    """
    if nonbonded is None:
        nonbonded = nonbonded_curvatures(atoms, terms)

    types = bond_types(terms)
    totals = [0.0] * len(types)
    counts = [0] * len(types)
    for term in terms:
        if term["type"] != "bond":
            continue
        i, j = list(term["atoms"].values())
        index = types.index((term["kwargs"]["r0"], term["kwargs"]["k"]))
        totals[index] += _bonded_curvature(term["kwargs"], atoms.get_distance(i, j))
        counts[index] += 1
    return [t / max(n, 1) + nb for t, n, nb in zip(totals, counts, nonbonded)]


def fit_bond_lengths(
    atoms: Atoms, terms: list[Term], strict: bool = True
) -> list[Term]:
    """Shift each bond type's `r0` so the *total* potential is flat along it.

    **Why this exists.**  `TestTemplateEnergies` pins each template's energy at
    its reference geometry and `fit_dissociation_energies` solves it to 1e-13.
    Nothing pinned the *gradient* there, and once `ZBL` was added it stopped
    being anywhere near zero: the repulsion is 2.0 eV at the H2 bond length and
    11.6 at O2's, with slopes to match, and `r0` was the only parameter that
    could have leaned against it -- so the minima simply moved outward.  H2 by
    0.105 A, HO2's O-O by 0.825, and every stretching frequency with them.

    The condition is one equation per bond type and it is determinate, not
    fitted: the net force along the type's bonds must vanish at the geometry the
    template's reference energy belongs to.  It is solved rather than optimized
    for the same reason the depth is -- a margin objective given a say in the
    geometry would trade the molecule against the barriers, and the geometry is
    data.

    **What absorbs what.**  Only the bond `r0` moves.  `bondbond` and
    `bondangle` carry reference lengths of their own and keep them: those terms
    are q-force's own cross-coupling fit, and shifting their reference changes
    what they mean rather than where they sit.  Their gradient at the reference
    geometry is part of what the bond `r0` is solved against, which is the
    consistent reading -- the condition is on the total, and the total is what
    the calculator computes.

    Solved by fixed point rather than by a root finder.  Everything except a
    type's own Morse is nearly constant in that type's `r0`, so subtracting the
    Morse's own contribution leaves an external force to cancel, and the `r0`
    that cancels it follows from a one-dimensional bracket on an analytic
    function.  The residual coupling -- through atoms two bonds share, and
    through the angle terms -- is what the iteration is for.
    """
    types = bond_types(terms)
    if not types:
        return list(terms)

    original = [r0 for r0, _ in types]
    current = list(original)

    # One pass.  The alternation with the depth solve lives in
    # `fit_template`; doing it in both places converges to the wrong place,
    # because the extra passes refine towards a fixed point of a stale `D`.
    # See `LENGTH_DEPTH_ROUNDS` for the numbers.
    working = set_bond_lengths(terms, current)
    params = {
        key: (
            t["kwargs"]["D"],
            t["kwargs"]["k"],
            t["kwargs"].get("h"),
        )
        for t in working
        if t["type"] == "bond"
        for key in [(t["kwargs"]["r0"], t["kwargs"]["k"])]
    }
    total = stretch_forces(atoms, working)
    lengths = bond_lengths(working, atoms)

    for index, (r0, k) in enumerate(bond_types(working)):
        D, k_value, height = params[(r0, k)]
        # The type's own Morse contribution to `total[index]`, so that what
        # is left is the part no choice of `r0` can change.
        bond_r = np.asarray(lengths[index], dtype=float)
        own = float(_morse_stretch_force(bond_r, D, r0, k_value, height).sum())
        external = total[index] - own

        def residual(trial, _r=bond_r, _D=D, _k=k_value, _h=height, _e=external):
            """Total force along this bond type if its `r0` were `trial`.

            Vectorized over `trial`, so the grid below is a single call.
            """
            mine = _morse_stretch_force(
                _r, _D, np.asarray(trial, dtype=float)[..., None], _k, _h
            ).sum(-1)
            return mine + _e

        # Shortening `r0` stretches the bond and so pulls harder -- but
        # only up to a point.  A Morse's pull peaks at `D*a/2` and falls
        # away again past the inflection, so an external push above that
        # ceiling cannot be cancelled at any bond length.  That is not a
        # bracketing failure to be widened around; it is the functional
        # form running out, and it is common enough here to be worth
        # naming: `ZBL` pushes O-O in HO2 apart with 25.0 eV/A and that
        # bond's Morse, as q-force parameterizes it, tops out at 8.3.
        #
        # The ceiling is `sqrt(k Dw / 8)`, so it moves with `D`, `k` and
        # `h`, all of which the outer fit and the depth solve are free to
        # raise: infeasible here means "not at these parameters" rather
        # than "not at all".  (The numbers above predate the `ZBL` taper;
        # no HCombustion bond is near the ceiling now.)
        # The search window is `MAX_LENGTH_SHIFT` either side of the
        # length the template arrived with, so both branches below are
        # bounded by the same tripwire rather than by whatever bracket
        # happened to be tried.
        low = original[index] - MAX_LENGTH_SHIFT
        high = original[index] + MAX_LENGTH_SHIFT
        grid = np.linspace(low, high, 400)
        values = residual(grid)

        # Not a sign test on the endpoints.  The pull peaks part-way down
        # and falls off again past the inflection, so when a solution
        # exists there are *two* roots and both ends of the window can sit
        # on the same side of zero.  Bracket from the strongest pull
        # outward to `high`, which picks the root nearer the bond length:
        # the far one shortens `r0` past the inflection, where the
        # curvature has the wrong sign and the "minimum" is a maximum.
        peak = int(np.argmin(values))
        if values[peak] <= 0.0 <= values[-1]:
            current[index] = brentq(
                lambda x: float(residual(x)),
                grid[peak],
                high,
                # 1e-13 A: the 1e-14 nm it was stated as when `r0` was in nm.
                xtol=1e-13,
                rtol=1e-15,
            )
        elif strict:
            raise DissociationFitError(
                f"no bond length within {MAX_LENGTH_SHIFT} A cancels the "
                f"{external:+.3f} eV/A pushing the {types[index]} bonds "
                f"apart: this Morse pulls at most "
                f"{-(values[peak] - external):.3f} eV/A there"
            )
        else:
            # Best effort, for a baseline or a report: the length that
            # leaves the least force behind.  The caller is told nothing
            # here, which is why `strict` is the default.
            current[index] = float(grid[np.argmin(np.abs(values))])

    return set_bond_lengths(terms, current)


def fit_template(
    atoms: Atoms,
    terms: list[Term],
    strict: bool = True,
    rounds: int = LENGTH_DEPTH_ROUNDS,
) -> list[Term]:
    """Solve one template's `r0` and `D` together at its reference geometry.

    Two determinate conditions, not an optimization: the total energy equals the
    reference atomization energy (`fit_dissociation_energies`), and the total
    force along every bond vanishes (`fit_bond_lengths`).  They couple only
    through `a = sqrt(k / 2D)` -- a deeper well is a narrower one, which moves
    the gradient at a fixed geometry -- so alternating them converges rather
    than needing a joint solve.

    This is the inner solve of `fit_force_constants`: whatever `h` and `k` the
    outer search is trying, the geometry and the atomization energy are restored
    underneath it, so neither is something the margin objective can spend.
    """
    fitted = list(terms)
    for _ in range(rounds):
        fitted = fit_dissociation_energies(
            atoms, fit_bond_lengths(atoms, fitted, strict)
        )
    return fitted


def scale_factor(atoms: Atoms, terms: list[Term], fitted: list[Term]) -> float:
    """Ratio between fitted and original depths, for reporting."""
    original = sum(t["kwargs"]["D"] for t in terms if t["type"] == "bond")
    updated = sum(t["kwargs"]["D"] for t in fitted if t["type"] == "bond")
    return updated / original if original else 1.0


# --------------------------------------------------------------------------
# Force constants
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BondVariable:
    """One fitted force constant.

    Identified by the template it belongs to and its `(r0, k)` there, so the
    two O-H bonds of water -- which q-force gave identical parameters -- are
    one variable rather than two.  Not shared *across* templates: the fit wants
    O=O in O2 six times stiffer and O-O in H2O2 left alone, and
    `fit_dissociation_energies` already gave up transferability for `D`.
    """

    template: str
    r0: float
    k: float
    elements: tuple[str, str]


@dataclass
class ForceConstantFit:
    """Result of `fit_force_constants`."""

    # Template name -> its refitted term list, ready for `io.json.write_jsonl`.
    terms: dict[str, list[Term]] = field(default_factory=dict)
    # The fitted variables, in the order they were solved for.
    variables: list[BondVariable] = field(default_factory=list)
    # Per variable, aligned with `variables`: the fitted asymptote height `h` in
    # eV in the `asymptote` modes, the multiplier on `k` in `mode="k"`.
    scales: list[float] = field(default_factory=list)
    # Reaction name -> min(H_reactant, H_product) - E_reference at its
    # transition state.  Positive is fittable.
    margins: dict[str, float] = field(default_factory=dict)
    # The same, before anything was refitted.
    margins_before: dict[str, float] = field(default_factory=dict)
    # Which of the two `scales` means, since they are reported in different
    # units and the reader has no way to tell them apart from the numbers alone.
    mode: str = "asymptote"
    # Fitted Morse depth per `(template, r0, k)`, in eV.
    # Keyed on the *input* `(r0, k)` because that is what `BondVariable` carries.
    # The inner solve moves `D` and `r0`; `k` is the outer search's.
    depths: dict[tuple[str, float, float], float] = field(default_factory=dict)
    # Force-constant scale per bond type; all 1.0 unless `mode` fits `k`.
    k_scales: list[float] = field(default_factory=list)
    # Total second derivative along each bond type at its template's reference
    # geometry, in eV/A**2, aligned with `variables`.  Reported separately from
    # the fitted `k` because they are no longer the same quantity: `ZBL` adds
    # curvature at the bond length that the bonded parameters do not know
    # about, and it is roughly as large as the bond's own.  A report that
    # quotes only the fitted `k` understates the real stiffness by about 2x.
    curvatures: list[float] = field(default_factory=list)


def bond_types(terms: list[Term]) -> list[tuple[float, float]]:
    """The distinct `(r0, k)` a template's bonds carry, in term order."""
    types: list[tuple[float, float]] = []
    for term in terms:
        if term["type"] != "bond":
            continue
        key = (term["kwargs"]["r0"], term["kwargs"]["k"])
        if key not in types:
            types.append(key)
    return types


def bond_depths(terms: list[Term]) -> list[float]:
    """Fitted Morse depth of each bond type, in eV, in `bond_types` order."""
    depths: list[float] = []
    seen: list[tuple[float, float]] = []
    for term in terms:
        if term["type"] != "bond":
            continue
        key = (term["kwargs"]["r0"], term["kwargs"]["k"])
        if key in seen:
            continue
        seen.append(key)
        depths.append(term["kwargs"]["D"])
    return depths


def scale_force_constants(terms: list[Term], scales: list[float]) -> list[Term]:
    """Copy of `terms` with each bond type's `k` multiplied by its scale.

    `scales` is aligned with `bond_types(terms)`.  `D` is left alone; it is the
    inner solve's variable, not this one's.
    """
    types = bond_types(terms)
    out: list[Term] = []
    for term in terms:
        if term["type"] != "bond":
            out.append(term)
            continue
        kwargs = dict(term["kwargs"])
        index = types.index((kwargs["r0"], kwargs["k"]))
        kwargs["k"] = kwargs["k"] * scales[index]
        out.append({**term, "kwargs": kwargs})
    return out


# Upper bound on a fitted per-bond asymptote height `h`, in eV.  The lower bound
# is `bond_asymptote` itself: below it the curve is *deeper* at every stretch,
# which is the wrong direction, and it is the height the twobody crossings were
# placed against.
DEFAULT_MAX_ASYMPTOTE: float = 10.0


def set_asymptotes(terms: list[Term], values: list[float]) -> list[Term]:
    """Copy of `terms` with each bond type's asymptote `h` set, in eV.

    `values` is in `bond_types` order and in eV, as `bond_asymptote` is.  Unlike
    `scale_force_constants` this *sets* rather than scales, because the search
    is over absolute heights bounded below by `bond_asymptote`.
    """
    order = {pair: index for index, pair in enumerate(bond_types(terms))}
    out: list[Term] = []
    for term in terms:
        if term["type"] != "bond":
            out.append(term)
            continue
        kwargs = dict(term["kwargs"])
        kwargs["h"] = float(values[order[(kwargs["r0"], kwargs["k"])]])
        out.append({**term, "kwargs": kwargs})
    return out


def frequency(force_constant: float, mass_a: float, mass_b: float) -> float:
    """Harmonic wavenumber (cm^-1) of a diatomic with this force constant.

    `force_constant` is in eV/A**2 and the masses in amu.
    The report is in wavenumbers because that is where the cost of refitting `k`
    is legible: a factor of six in `k` reads as a factor of six, and as a factor
    of 2.5 in a number that can be compared against a spectrum.
    """
    return total_wavenumber(force_constant, mass_a, mass_b)


def total_wavenumber(curvature: float, mass_a: float, mass_b: float) -> float:
    """Harmonic wavenumber (cm^-1) of a diatomic with this *total* curvature.

    `frequency` above answers the same question about a bonded force constant,
    which used to be the same number and need not be.  The nonbonded terms add
    curvature at the bond length (68.5 eV/A**2 at O-H before the `ZBL` taper,
    on its own worth 4431 cm^-1), and `fit_bond_lengths` can displace `r0` off
    the bond, where the Morse's own curvature is no longer `k`.

    This is the number the timestep is set by: a stable velocity-Verlet run
    wants ~15 steps per period, so a step of `dt` femtoseconds needs every mode
    under `33356 / (15 dt)` cm^-1 -- 4450 at 0.5 fs.

    `curvature` in eV/A**2, masses in amu.
    """
    return wavenumber(curvature, mass_a * mass_b / (mass_a + mass_b))


def wavenumber(curvature: float, reduced_mass: float) -> float:
    """`total_wavenumber` given the reduced mass directly, in amu.

    The form the objective uses, because `reduced_masses` below precomputes the
    reduction once for the whole search and there is nothing to be gained by
    undoing it and redoing it a few hundred thousand times.
    """
    stiffness = curvature * units._e / 1e-20
    if stiffness <= 0.0:  # a bond that is not at a minimum has no wavenumber
        return 0.0
    return float(
        np.sqrt(stiffness / (reduced_mass * units._amu)) / (2 * np.pi * units._c * 1e2)
    )


def reduced_masses(variables: list["BondVariable"]) -> np.ndarray:
    """Reduced mass in amu of each variable's bond, aligned with `variables`."""
    return np.array(
        [
            (lambda a, b: a * b / (a + b))(
                *(atomic_masses[atomic_numbers[e]] for e in variable.elements)
            )
            for variable in variables
        ]
    )


def diabatic_energy(reaction_set, frame: Atoms, positions) -> float:
    """Total energy of `frame`'s connectivity evaluated at `positions`.

    This is the diabat: the bonding topology of one endpoint, scored at the
    geometry of another.  Both of a reaction's endpoints evaluated at its
    transition state are what the coupling fit inverts, and what
    `fit_force_constants` is trying to push above the reference barrier.

    Includes the nonbonded term for the same reason `bonded_energy` does, and
    more sharply: the barrier it is compared against is a reference energy that
    contains everything, so a bonded-only diabat made `reaction_margins` --
    and therefore the entire force-constant objective -- an inequality between
    two different quantities.
    """
    from DynamicTopology.core.topology import Topology

    atoms = frame.copy()
    atoms.calc = None
    atoms.positions = positions
    topology = Topology.from_atoms(atoms)
    topology.set_terms(reaction_set.get_terms(topology))
    energy = _QFORCE(atoms.positions, atoms.pbc, atoms.cell, topology.term_dict)[0]
    return energy + nonbonded_energy(atoms, topology.term_dict)


def install_templates(
    reaction_set,
    templates: list[tuple[str, Atoms, list[Term]]],
    fitted: list[list[Term]],
    keys: list[str] | None = None,
) -> None:
    """Swap refitted terms into a loaded `ReactionSet`, in memory.

    The outer solve evaluates hundreds of parameter sets and every one of them
    has to be scored through the real `ReactionSet.get_terms` path -- writing
    the files and reloading is both slow and a side effect on the dataset the
    fit has not yet decided to keep.  Only `kwargs` change, so the stored
    `Topology` graphs and hashes stay valid; the remapped-term cache does not,
    which is why this routes through `ReactionSet.set_template_terms`.

    The fitted lists must keep the exclusion terms `refit.load_templates`
    derived: installing a template without them leaves the whole-system ZBL and
    12-6 uncancelled inside `nonbonded_energy`, and every margin then read
    feasible by hundreds of eV (rxn_14 by +2131) whatever the force constants
    were.

    `keys` is `template_keys(templates)`, for a caller installing many times.
    """
    if keys is None:
        keys = template_keys(templates)
    for key, terms in zip(keys, fitted):
        reaction_set.set_template_terms(key, terms)


def template_keys(templates: list[tuple[str, Atoms, list[Term]]]) -> list[str]:
    """Each template's `ReactionSet` key: the WL hash of its bond graph."""
    from DynamicTopology.core.topology import Topology

    return [
        Topology.from_terms(original, atoms).hash() for _, atoms, original in templates
    ]


def reaction_margins(
    reaction_set, reactions: list[tuple[str, list[Atoms]]]
) -> dict[str, float]:
    """`min(H_reactant, H_product) - E_reference` at each transition state.

    Negative means no real coupling amplitude reproduces that barrier, because
    the EVB ground state of a two-level block lies below both diabats by
    construction.  Reactions whose transition state carries no reference energy
    are skipped -- there is nothing to be feasible against.
    """
    margins: dict[str, float] = {}
    for name, frames in reactions:
        transition = frames[len(frames) // 2]
        if transition.calc is None:
            continue
        reactant = diabatic_energy(reaction_set, frames[0], transition.positions)
        product = diabatic_energy(reaction_set, frames[-1], transition.positions)
        margins[name] = min(reactant, product) - transition.get_potential_energy()
    return margins


def bond_asymptotes(terms: list[Term]) -> list[float]:
    """Asymptote height `h` of each bond type in eV, in `bond_types` order.

    A bond without its own `h` reads as `bond_asymptote`, as `QForce` does.
    """
    heights: list[float] = []
    seen: list[tuple[float, float]] = []
    for term in terms:
        if term["type"] != "bond":
            continue
        key = (term["kwargs"]["r0"], term["kwargs"]["k"])
        if key in seen:
            continue
        seen.append(key)
        height = term["kwargs"].get("h")
        heights.append(active().bond_asymptote if height is None else float(height))
    return heights


def fit_force_constants(
    reaction_set,
    templates: list[tuple[str, Atoms, list[Term]]],
    reactions: list[tuple[str, list[Atoms]]],
    margin: float = DEFAULT_MARGIN,
    frequency_weight: float = DEFAULT_FREQUENCY_WEIGHT,
    max_scale: float = DEFAULT_MAX_SCALE,
    mode: str = "asymptote",
    geometry_weight: float = DEFAULT_GEOMETRY_WEIGHT,
    max_wavenumber: float = DEFAULT_MAX_WAVENUMBER,
    curvature_weight: float = DEFAULT_CURVATURE_WEIGHT,
    max_asymptote: float = DEFAULT_MAX_ASYMPTOTE,
) -> ForceConstantFit:
    """Refit every template's bonds so the couplings become fittable.

    Outer variables are, per `BondVariable`, the asymptote height `h` in eV
    (`mode="asymptote"`, the default), `log(k-scale)` (`mode="k"`), or both
    (`mode="asymptote-k"`).  The inner solve is `fit_template`, which re-derives
    each template's `D` scale and bond lengths so its atomization energy and
    geometry are reproduced exactly whatever the outer solve is trying.  The atomization condition is therefore a constraint the
    objective cannot trade away, not a term competing with the barriers.

    The objective is a hinge, not a least squares:

        J = sum_r max(0, margin - margin_r)**2
          + geometry_weight  * sum_b (leftover force at the reference geometry)**2
          + curvature_weight * sum_b max(0, nu_b - max_wavenumber)**2
          + frequency_weight * sum_b (x_b / box_b)**2

    A margin past the target is worth nothing -- the barrier is already
    reproduced exactly by the amplitude, which is free per reaction -- so the
    hinge stops pushing as soon as a channel is feasible and spends nothing more
    on it.  The same shape is used for the wavenumber: a mode under the cap is
    free, and one over it pays.

    The third term is the timestep: the stiffness the integrator sees is the
    total curvature at the reference geometry, not the fitted `k`.  See
    `DEFAULT_MAX_WAVENUMBER` and `_bonded_curvature`.

    Args:
        reaction_set: a loaded `ReactionSet` for the same dataset.  **Mutated**:
            on return its templates carry the fitted terms, so the caller can
            fit couplings against them without writing files first.
        templates: `(name, atoms, terms)` per molecule, atoms carrying the
            reference atomization energy.
        reactions: `(name, frames)` per reaction, frames in reactant / TS /
            product order.
        margin: how far below the reference barrier a diabat must sit.
        frequency_weight: pull back towards the original force constants.
        max_scale: hard bound on each k-scale, in both directions.
        mode: which bonded parameters the outer search moves.
        max_wavenumber: stretching modes above this cost the objective.
        curvature_weight: how much they cost.  Zero restores the old objective,
            which is the vacuity check for the cap.
        max_asymptote: upper bound on each `h`, in eV; the lower bound is
            `bond_asymptote`.  Setting it there freezes `h`, which is the
            vacuity check for the asymptote -- plain Morse must make no
            progress.
    """
    keys = template_keys(templates)
    variables: list[BondVariable] = []
    for name, atoms, terms in templates:
        symbols = {
            (term["kwargs"]["r0"], term["kwargs"]["k"]): tuple(
                atoms[index].symbol for index in term["atoms"].values()
            )
            for term in terms
            if term["type"] == "bond"
        }
        for r0, force_constant in bond_types(terms):
            variables.append(
                BondVariable(
                    template=name,
                    r0=r0,
                    k=force_constant,
                    elements=symbols[(r0, force_constant)],
                )
            )

    # Which slice of the variable vector belongs to which template, so a
    # template's scales can be handed to `scale_force_constants` in the order
    # `bond_types` produced them.
    spans: list[tuple[int, int]] = []
    start = 0
    for _, _, terms in templates:
        stop = start + len(bond_types(terms))
        spans.append((start, stop))
        start = stop

    modes = ("asymptote", "k", "asymptote-k")
    if mode not in modes:
        raise ValueError(f"mode must be one of {modes}, got {mode!r}")

    # The search vector is laid out in fixed-width blocks of one entry per bond
    # type, so a `spans` slice indexes into any block with the same offsets:
    #
    #     asymptote     [ h ]
    #     k             [ log k-scale ]
    #     asymptote-k   [ h | log k-scale ]
    #
    # `h` is the absolute asymptote height in eV, not a scale: its lower bound
    # is `bond_asymptote`, which is also what a bond without one reads as.
    width = len(variables)
    lifted = mode in ("asymptote", "asymptote-k")
    k_at = width if mode == "asymptote-k" else 0
    base_asymptote = active().bond_asymptote

    def apply(terms: list[Term], x: np.ndarray, lo: int, hi: int) -> list[Term]:
        if mode != "asymptote":
            terms = scale_force_constants(terms, list(np.exp(x[k_at + lo : k_at + hi])))
        if lifted:
            terms = set_asymptotes(terms, [float(value) for value in x[lo:hi]])
        return terms

    # Both constant across the entire search: the elements do not change, and
    # the nonbonded curvature is a function of the fixed template geometry.
    # Measured once here rather than per objective evaluation, which is the
    # difference between a hinge that costs nothing and one that doubles the
    # cost of the fit.
    masses = reduced_masses(variables)
    frozen_nonbonded = [
        nonbonded_curvatures(atoms, terms) for _, atoms, terms in templates
    ]

    def refit(x: np.ndarray, rounds: int = LENGTH_DEPTH_ROUNDS) -> list[list[Term]]:
        """Every template's geometry and depth re-solved at this point.

        Always best-effort on the geometry: where no bond length cancels the
        repulsion, `fit_bond_lengths` leaves the closest it can and `score`
        charges for what is left.  See `DEFAULT_GEOMETRY_WEIGHT` for why this
        is a penalty and not a constraint.
        """
        return [
            fit_template(atoms, apply(terms, x, lo, hi), strict=False, rounds=rounds)
            for (_, atoms, terms), (lo, hi) in zip(templates, spans)
        ]

    # Per-variable normalizer for the regularizer: the width of each variable's
    # box, so `used` below is "fraction of the freedom taken", and an origin at
    # the *unchanged* force field -- `h = bond_asymptote` is plain Morse as every
    # term file without an `h` reads, and a zero log k-scale is q-force's own
    # constant.
    log_bound = max(float(np.log(max_scale)), 1e-12)
    height_scale = max(max_asymptote - base_asymptote, 1e-12)
    blocks = {
        "asymptote": [np.full(width, height_scale)],
        "k": [np.full(width, log_bound)],
        "asymptote-k": [np.full(width, height_scale), np.full(width, log_bound)],
    }[mode]
    scale_of = np.concatenate(blocks)
    origin = np.zeros(scale_of.size)
    if lifted:
        origin[:width] = base_asymptote

    def score(x: np.ndarray) -> float:
        try:
            # Two rounds, not four.  The inner solve is most of this fit's
            # runtime and the objective only reads *margins* and a residual
            # force, neither of which a 1e-4 eV/A refinement moves.  What gets
            # written out is refit at full accuracy below.
            fitted = refit(x, rounds=2)
            install_templates(reaction_set, templates, fitted, keys)
        except DissociationFitError:
            # These force constants leave some template with no depth scale that
            # reproduces its atomization energy.  Not a failure of the fit, just
            # a point the outer solve should walk away from.
            return np.inf
        leftover = np.array(
            [
                value
                for (_, atoms, _), terms in zip(templates, fitted)
                for value in stretch_forces(atoms, terms)
            ]
        )
        shortfall = [
            max(0.0, margin - value)
            for value in reaction_margins(reaction_set, reactions).values()
        ]
        # Stretching wavenumbers, in `variables` order.  `stretch_curvatures`
        # rather than `bond_curvatures`: same number on every mode the cap can
        # bind on, at a hundredth of the cost.  See its docstring for the one
        # row where they differ and why it does not matter here.
        overshoot = np.array(
            [
                max(0.0, wavenumber(curvature, mass) - max_wavenumber)
                for mass, curvature in zip(
                    masses,
                    (
                        value
                        for (_, atoms, _), terms, nonbonded in zip(
                            templates, fitted, frozen_nonbonded
                        )
                        for value in stretch_curvatures(atoms, terms, nonbonded)
                    ),
                )
            ]
        )
        # The regularizer is on the *fraction of the available box* each
        # variable uses, not on its raw value.  `h` in eV and `log(k-scale)`
        # live on scales that differ by an order of magnitude, so penalizing raw
        # magnitudes would silently reweight the objective whenever either box
        # changed.  (It did once: under the retired shape term, raising a decay
        # bound multiplied this penalty by ~200 for no physical reason.)
        used = (x - origin) / scale_of
        return float(
            np.dot(shortfall, shortfall)
            + geometry_weight * np.dot(leftover, leftover)
            + curvature_weight * np.dot(overshoot, overshoot)
            + frequency_weight * np.dot(used, used)
        )

    # `x = start` is the *unchanged* force field, so the "before" column reports
    # on the force field the templates arrived with.  The geometry solve is
    # best-effort there, for the reason `DEFAULT_GEOMETRY_WEIGHT` gives.
    start = origin.copy()
    install_templates(reaction_set, templates, refit(start), keys)
    before = reaction_margins(reaction_set, reactions)

    frozen = {
        "asymptote": max_asymptote <= base_asymptote,
        "k": max_scale <= 1.0,
        "asymptote-k": max_asymptote <= base_asymptote and max_scale <= 1.0,
    }[mode]
    if frozen or not variables:
        # No freedom at all.  Say so by returning the unrefitted point rather
        # than handing a degenerate box to the optimizer; this is also the path
        # the vacuity check takes.
        solution = start
    else:
        # A k-scale is bounded symmetrically in the log, so that halving and
        # doubling are the same distance from the starting point.  `h` only
        # goes up: below `bond_asymptote` the curve is deeper at every stretch.
        bound = float(np.log(max_scale))
        heights = [(base_asymptote, max(max_asymptote, base_asymptote))] * width
        box = {
            "asymptote": heights,
            "k": [(-bound, bound)] * width,
            "asymptote-k": heights + [(-bound, bound)] * width,
        }[mode]
        # **The search converges on its own; there is nothing here to cap.**
        # Worth writing down because it was twice diagnosed wrongly.  Swept over
        # the retired shape term's `mode="shape"` from plain Morse, which was the
        # slowest configuration (the asymptote fit converges in ~16 s):
        #
        #     budget         time    nfev   objective   channels
        #     maxfev=200     8.0s     200   135.18145   12 -> 14
        #     maxfev=400    16.0s     400     6.05919   12 -> 15
        #     maxfev=800    32.3s     800     5.72266   12 -> 14
        #     maxfev=2000   49.5s    1219     5.71545   12 -> 15
        #
        # The last row is the answer: `nfev` came in *under* its cap, so Powell
        # stopped on `xtol`/`ftol` at 1219 evaluations and about fifty seconds.
        # Everything past 400 is noise -- the objective moves 6.059 to 5.715 and
        # the channel count wanders between 14 and 15 while improving, because
        # the hinge, the regularizer and the geometry penalty will trade a
        # marginal channel against each other.
        #
        # `maxiter` was the wrong knob regardless: it counts Powell *sweeps*
        # while `maxfev` counts evaluations, and with `maxfev` defaulting to
        # `N * 1000` the sweep cap is never what binds.  Capping it at 30 and
        # then at 10 changed nothing, which is what sent this looking in the
        # wrong place to begin with.
        result = minimize(
            score,
            start,
            method="Powell",
            bounds=box,
            options={"maxiter": 200, "xtol": 1e-3, "ftol": 1e-5},
        )
        solution = np.asarray(result.x, dtype=float)
        logger.info(
            "force-constant fit: %d evaluations, objective %.6g",
            result.nfev,
            result.fun,
        )

    fitted = refit(solution)
    install_templates(reaction_set, templates, fitted, keys)
    margins = reaction_margins(reaction_set, reactions)

    # **Refuse a solution that reproduces fewer channels than it started with.**
    #
    # The objective is not monotone in the channel count and was never going to
    # be: the hinge is `max(0, margin - margin_target)**2`, so a channel already
    # past the target contributes exactly nothing and the regularizer is free to
    # spend it.  That is the right trade when the margin bought is another
    # channel's feasibility and the wrong one when it is only frequency, and the
    # objective cannot tell the difference -- the note above the optimizer call
    # records the count wandering between 14 and 15 while the objective improved.
    #
    # It was unreachable while the starting point was poor, because there was
    # always more to gain by fixing an infeasible channel than to save by
    # dropping a feasible one.  `params.bond_asymptote` made the starting point
    # good -- 16 of 19 before any force-constant fit at all -- and `mode="k"`
    # then traded `rxn_16` from +0.191 to -0.095 for the frequency term, giving
    # up the one channel the asymptote had just bought.
    #
    # `before` is measured at `refit(start)`, so this compares the solution
    # against the same point the search began from rather than against the
    # dataset on disk, and the fallback is a point already inside the box.
    feasible = sum(value > 0.0 for value in margins.values())
    baseline = sum(value > 0.0 for value in before.values())
    if feasible < baseline:
        logger.warning(
            "force-constant fit reproduced %d channels against %d at its "
            "starting point; keeping the starting point. The hinge is blind to "
            "a margin already past --margin, so the regularizer can spend one; "
            "lower --frequency-weight or raise --margin to make the fit value "
            "what it is giving up.",
            feasible,
            baseline,
        )
        solution = start
        fitted = refit(solution)
        install_templates(reaction_set, templates, fitted, keys)
        margins = reaction_margins(reaction_set, reactions)

    curvatures = [
        value
        for (_, atoms, _), terms in zip(templates, fitted)
        for value in bond_curvatures(atoms, terms)
    ]
    return ForceConstantFit(
        terms={name: terms for (name, _, _), terms in zip(templates, fitted)},
        variables=variables,
        scales=(
            [float(value) for value in solution[:width]]
            if lifted
            else [float(value) for value in np.exp(solution)]
        ),
        k_scales=(
            [float(value) for value in np.exp(solution[k_at : k_at + width])]
            if mode != "asymptote"
            else [1.0] * width
        ),
        mode=mode,
        curvatures=curvatures,
        margins=margins,
        margins_before=before,
        depths={
            (name, r0, k): depth
            for (name, _, original), terms in zip(templates, fitted)
            for (r0, k), depth in zip(bond_types(original), bond_depths(terms))
        },
    )
