"""Tapered screened-nuclear repulsion, over every pair in the system.

**What this term is for.**  ACKS2 has no repulsive branch.  Its Coulomb kernel
`erf(2r)/r` is finite at contact rather than divergent -- it tends to 2.257, so
the charges saturate and there is no runaway in the solve itself -- but the
interaction it produces between two atoms of different electronegativity is a
smooth, monotone, ~4 eV attractive funnel all the way to zero separation:

    r (A)   2.0    1.5    1.2   0.96    0.6    0.2   0.05
    Q_H     0.110  0.212  0.259 0.286  0.317  0.347  0.353
    E (eV) -0.09  -0.43  -0.80 -1.21  -2.19  -3.71  -4.03

No minimum, no wall.  At 3000 K, kT is 0.26 eV.  A 200-atom H2/O2 box run
against ACKS2 alone reached 0.60 A intermolecular contacts with 55 pairs inside
1.2 A, reading -46.6 eV of nonbonded energy -- 0.85 eV per pair, which is this
table.  This is the term that opposes it.

**Why it is topology-independent, and why that is the whole design.**  The
obvious repulsion to reach for is the Lennard-Jones already in the templates,
excluded between bonded atoms the way any fixed-topology force field excludes
it.  That was tried four times.  It fails because q-force's 12-6 is enormous at
the separations reactive chemistry actually visits -- 504 eV at the H2 bond
length, 930 eV at O-H, 1348 eV at O-O -- so a diabatic state that has *broken* a
bond pays hundreds of eV for a pair that is merely close.  Every attempt to
strip that penalty made the repulsion differ between diabatic states, and each
one failed in its own way: stripping it from the parent as well let the box fuse
at 0.60 A; gating it on the coupling left a 19 eV plateau across rxn_16's
reaction path where a wall was half-removed; and transition states with close
non-bonded contacts came back with EVB amplitudes of -72 to -108 eV.

What made that survivable was not abandoning the 12-6 but shrinking it: since
`lj.switch` the term is three to four orders of magnitude smaller at a bond
length, small enough to need no exclusions, and it is back in the force field on
the same terms this one is on.  The paragraphs below are why this module carries
the short range and that one does not.

This term takes no topology at all.  Its only parameter is the atomic number,
which it reads from `numbers` rather than from a term list, so there is no
template, no remapping and no per-state path by which it could acquire a state
dependence.  That is deliberate and it is structural: a term identical across
every diabatic state adds the same constant to every EVB diagonal, and a common
shift of the diagonal moves `np.linalg.eigh`'s eigenvalue by exactly that
constant while leaving the eigenvectors untouched.  So this term *cannot*
produce a plateau, a spurious coupling amplitude, a pivot dependence, or a
discontinuity at the bimolecular cutoff -- not "does not", cannot.  Each of the
four previous failures lived in the degree of freedom this removes.

The same property means it cancels exactly out of every energy *difference*,
including the diabatic margins `fit/dissociation.py` scores channels on.  It
buys stability and it buys nothing at all for fittability; that work belongs to
the bonded fit.

**Form.**  The ZBL universal potential (Ziegler, Biersack and Littmark), which
is what reactive potentials -- ReaxFF, Tersoff/ZBL -- use for exactly this job.
Screened Coulomb between bare nuclei, fitted across the periodic table, with no
free parameters beyond Z.  Values it takes at the separations that matter here,
against the 12-6 it replaces:

    pair  r (A)  what it is                12-6      ZBL
    H-H   0.777  the H2 bond length      504 eV    2.0 eV
    O-H   0.960  the O-H bond length     930 eV    5.4 eV
    O-O   1.210  the O2 bond length     1348 eV   11.6 eV
    O-H   0.600  the observed fusion    2.6e5 eV  20.9 eV

It is applied to bonded pairs too, since it knows nothing about bonds.  Those
values are absorbed by the fitted Morse depths -- `fit/dissociation.py` solves
against `E_QForce + E_nonbonded`, and this term is part of `E_nonbonded` -- so a
template still reproduces its own reference atomization energy.  O2 is the
stress case: +11.6 eV and +36.6 eV/A at its own bond length.

**The taper, and why the term had to acquire one.**  ZBL is a screened
*nuclear* potential.  Its screening function was fitted where two nuclei are
close enough that the electrons between them barely intervene -- the keV
stopping-power regime -- and it carries no information at all about the range
where chemistry happens.  Used unmodified out to infinity, as this module did,
its two longest exponentials (`B` = 0.4029 and 0.20162, decay lengths of 0.45
and 0.89 A for an O-H pair) reach straight into the hydrogen bond:

    O...H at 1.94 A, the H-bond contact in a water dimer      +0.55 eV
    O...O at 2.91 A, the same dimer's oxygen separation       +0.21 eV

A hydrogen bond is worth -0.218 eV.  Measured on the assembled surface, at the
dimer's own geometry `ZBL` contributed **+0.72 eV** against ACKS2's -0.12 eV, so
the water dimer was unbound by +0.60 eV and had no minimum at any separation.
In a 64-water box at 997 kg/m**3 the intermolecular part of this term read
+172 eV and **+251 kbar**, which is most of the reason that box would not hold
together.  None of that is a defect of the ZBL form; it is the form being
evaluated a factor of ten outside the range it was fitted in.

Every hybrid ZBL potential in the literature -- Tersoff/ZBL, ReaxFF -- switches
ZBL off with a Fermi function before it reaches bonding distances and hands over
to something else.  This module does the same, with the difference that there is
nothing to hand over to: `ACKS2` and the bonded terms are the whole of the rest.
So the taper is placed as far out as the wall can afford rather than at the ~1 A
those potentials use.  `TAPER_RADIUS` and `TAPER_WIDTH` are the two constants,
and what they buy at a 64-water box and at the dimer:

    quantity                              untapered   tapered
    intermolecular ZBL, 64 waters          172 eV     0.16 eV
    intermolecular ZBL pressure           +251 kbar   +1 kbar
    water dimer minimum                    none       -0.116 eV at 2.85 A
    H2 + O2 wall, 4.0 A -> 0.6 A           21.8 eV    20.8 eV

and what they cost at the separations the bonded fit leans on, as the retained
fraction `f(r)`:

    H-H at 0.741 A (the H2 bond)     0.998
    O-H at 0.958 A (the O-H bond)    0.989
    O-O at 1.208 A (the O2 bond)     0.918
    H...H at 1.51 A (water's 1-3)    0.479

The first three are the ones absorbed into the fitted Morse depths, and they are
nearly untouched -- which is why this change is a refit rather than a rebuild.
**It is still a refit.**  `fit/dissociation.py` solves against
`E_QForce + E_nonbonded` with this term inside `E_nonbonded`, so every `.jsonl`
in every dataset was fitted against the untapered form and has to be regenerated
by `scripts/fit.py` after any change to the two constants below.

**And the refit is not free, though the bill lands on HCombustion rather than on
water.**  A transition state is where close intermolecular contacts are, and so
where the removed tail was largest, which means the taper lowers the diabats at
exactly the geometries `fit.coupling` inverts the secular equation at.
HCombustion went from 14 of 19 fittable channels to **12** -- rxn_06, rxn_11 and
rxn_16 are now decoupled, and two of those three had margins under 0.1 eV before
it.  `--frequency-weight` at 200x the default buys none of them back, so this is
structural and not a knob setting.  It also made H2 stiff enough that the fit
overshot its own wavenumber cap, so HCombustion is now fitted with
`--max-wavenumber 4200` (landing at 4325 cm^-1, and 4314 after the 12-6 came
back) rather than the 4400 default.
Water lost nothing: still 3 of 3, still under the cap.

The taper is a function of `r` alone.  It multiplies a pair potential that
already took no topology, so it cannot introduce one, and every structural
guarantee in the paragraphs above survives it unchanged.

**A cut in `r`, not in the reduced coordinate `x = r/a`.**  The natural-looking
choice is to taper at a fixed `x`, which would be element-transferable for free.
It is wrong: bond lengths scale with covalent radii and `a` scales with
`Z**-0.23`, so a single `x` cut lands at 1.97 A for H-H and 1.22 A for O-O --
that is, it would leave H-H walled well past the H2 bond while cutting O-O
straight through the O2 bond.  A fixed radius in Angstrom tracks bond lengths
much better than the screening length does.

**What the taper does not fix, and what now does.**  With it, the dimer binds at
-0.112 eV against a -0.218 eV reference, of which `ACKS2` supplies -0.121 eV on
its own.  So the surface was under-attractive *and* under-repulsive -- and the
second was the larger error, because the taper removes the intermolecular wall
and, for a while, nothing replaced it.  Measured then: the pressure of a
64-water box at 997 kg/m**3 went from about +165 kbar to **-7.6 kbar**, crossing
zero, and the equation-of-state crossing moved from ~250-350 kg/m**3 to ~1250.
Against an experimental 997 that is a change from three-to-four times too low to
roughly a quarter too high -- a large improvement and an overshoot, in that
order.

`forcefield/lj.py` is what replaced it, and this is the term it hands over to.
The 12-6 is switched *on* at 2.2 A, so above that radius the repulsion and the
dispersion are its and below it they are this module's.  The two switches are
the same Fermi function with the same `TAPER_WIDTH`; they are deliberately
**not** placed at the same radius, and `lj.SWITCH_RADIUS` carries the
measurement that says why.  See `production/density-300K/README.md` for what the
pair did to the density.

Angstrom and eV throughout, unlike `QForce` and `LennardJones`, because the ZBL
constants are stated in those units and converting them would put a unit slip
between this module and its own literature.
"""

import numpy as np


# Screening length prefactor, `0.8854 * a_0`, in Angstrom.
SCREENING_LENGTH: float = 0.46850

# Coulomb constant in eV*Angstrom, matching `ACKS2.CCOUL` to the digits ASE uses.
CCOUL: float = 14.399645

# The universal screening function `phi(x) = sum_k C[k] * exp(-B[k] * x)`.
PHI_C: tuple[float, ...] = (0.18175, 0.50986, 0.28022, 0.02817)
PHI_B: tuple[float, ...] = (3.19980, 0.94229, 0.40290, 0.20162)


# Where the screened-nuclear form stops being evaluated, in Angstrom, and how
# sharply it is switched off there.  See the module docstring for the numbers
# these produce; this comment is for why they are these numbers.
#
# `TAPER_RADIUS` is bounded from both sides and the window is narrow:
#
#   from below  the wall has to stay ahead of the ACKS2 contact funnel at every
#               separation, or two molecules drift through each other.  At 1.2 A
#               the H2 + O2 approach of `tests/test_collapse.py` already reads
#               +0.52 eV against +3.14 eV untapered, and by 1.0 A that approach
#               is downhill -- the collapse this module exists to prevent.
#   from above  every 0.1 A of extra reach puts roughly another 0.2 eV onto the
#               hydrogen bond.  Measured on the water dimer, the minimum moves
#               -0.148 eV at 2.70 A (1.4) -> -0.116 eV at 2.85 A (1.5) ->
#               -0.090 eV at 3.00 A (1.6), against a reference of -0.218 eV at
#               2.91 A.  1.5 puts the minimum at the right *separation* and
#               takes the depth deficit as a known residual, which is the right
#               way round: the position is structure and the depth is not.
#
# `TAPER_WIDTH` is the smallest that keeps the term smooth enough to integrate.
# The switch contributes `-f(1-f)/w` to `du/dr`, which peaks at `1/(4w)` times
# the potential there; at 0.12 A that peak sits at 1.5 A where O-H ZBL is
# 1.15 eV, so it adds 2.4 eV/A of force -- large, but far under the 17.8 eV/A
# the unmodified term already carries at the O-H bond length, and continuous in
# every derivative because a Fermi function is analytic.
TAPER_RADIUS: float = 1.5
TAPER_WIDTH: float = 0.12


def taper(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The Fermi switch `f = 1/(1 + exp((r - rc)/w))` and its derivative.

    Returns `(f, df/dr)`.  `f` runs from 1 well inside `TAPER_RADIUS` to 0 well
    outside it, and `df/dr = -f (1 - f) / w` is expressed through `f` itself so
    that it costs one extra multiply rather than a second `exp`.

    The exponent is clipped before `np.exp` sees it.  Without that, an r far
    outside the window overflows to `inf` and the `f (1 - f)` product becomes
    `0 * inf = nan` in the derivative -- at 8 A the exponent is already 54, and
    `ZBL.__call__` evaluates every pair in the box including ones half a cell
    apart.  Clipping at 500 is far outside anything the switch resolves and
    keeps `f` exactly 0 or 1 there, which is what the analytic limit is anyway.
    """
    z = np.clip((r - TAPER_RADIUS) / TAPER_WIDTH, -500.0, 500.0)
    f = 1.0 / (1.0 + np.exp(z))
    return f, -f * (1.0 - f) / TAPER_WIDTH


def pair_potential(
    r: np.ndarray, z1: np.ndarray, z2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The repulsion and its radial derivative, `(u, du/dr)`, in eV and eV/A.

    Tapered ZBL: the screened-nuclear form multiplied by `taper`.  This is the
    potential the force field actually evaluates, not the textbook ZBL -- see
    the module docstring for why the two differ and what the difference is worth.
    Callers wanting the bare form can divide by `taper(r)[0]`, but nothing does.

    `r` must be strictly positive; callers holding a full distance matrix should
    substitute anything on the diagonal and zero the result there, as
    `ZBL.__call__` does.

    Angstrom and eV, like the rest of the `forcefield` package and like
    `lj.pair_potential`: `SCREENING_LENGTH` is in Angstrom and `CCOUL` in
    eV*Angstrom, so `r` has to be in Angstrom and the result comes back in eV.
    `TAPER_RADIUS` and `TAPER_WIDTH` are in Angstrom for the same reason.
    """
    a = SCREENING_LENGTH / (z1**0.23 + z2**0.23)
    x = r / a

    phi = np.zeros_like(r)
    dphi_dx = np.zeros_like(r)
    for c, b in zip(PHI_C, PHI_B):
        term = c * np.exp(-b * x)
        phi = phi + term
        dphi_dx = dphi_dx - b * term

    k = CCOUL * z1 * z2
    u = k * phi / r
    # d/dr [ k phi(r/a) / r ] = k ( phi'(x)/a / r  -  phi(x) / r^2 )
    du_dr = k * (dphi_dx / (a * r) - phi / r**2)

    # Product rule, and it has to be applied here rather than in `__call__`:
    # everything downstream -- the forces, the virial, `fit/dissociation.py`'s
    # curvatures -- differentiates whatever this function returns, so the
    # switch and its derivative have to travel together or the analytic
    # gradients silently stop matching the energy.
    f, df_dr = taper(r)
    return f * u, f * du_dr + df_dr * u


class ZBL:
    """Tapered ZBL repulsion summed over every pair, with no exclusions.

    Deliberately *not* wired like `ACKS2` and `LennardJones`.  Those read their
    per-atom parameters out of `term_dict`, which means they carry the term-order
    versus global-order distinction and, more importantly, that a state could in
    principle hand them a different parameter set.  This one takes `numbers`
    straight from the `Atoms`, so the only thing it can be a function of is the
    geometry and the elements -- see the module docstring for why that matters
    more than the consistency.
    """

    def __call__(
        self,
        pos: np.ndarray,
        numbers: np.ndarray,
        pbc: np.ndarray,
        cell: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        vecs = pos[:, None, :] - pos[None, :, :]
        if np.any(pbc):
            f = vecs @ np.linalg.inv(cell)
            vecs = vecs - (pbc * np.floor(f + 0.5)) @ cell

        rij = np.sqrt(np.sum(vecs * vecs, -1))
        diag = np.diag_indices(len(pos))
        r = rij.copy()
        r[diag] = 1.0  # excluded below; only keeps the division finite

        z = np.asarray(numbers, dtype=float)
        u, du_dr = pair_potential(r, z[:, None], z[None, :])
        u[diag] = 0.0
        du_dr[diag] = 0.0

        # The 0.5 cancels for the forces because both (i, j) and (j, i)
        # contribute to dE/d(pos_i) -- the convention `ACKS2.compute_coulomb`
        # and `LennardJones.__call__` both use.
        energy = 0.5 * float(np.sum(u))
        nij = vecs / r[:, :, None]
        forces = -np.sum(du_dr[:, :, None] * nij, axis=1)

        # Virial.  Under a homogeneous strain every minimum-image separation
        # maps `v -> (I + e) v`, so `dr_ij/de_ab = v_a v_b / r` and
        #
        #     W_ab = dE/de_ab = 0.5 * sum_ij du_dr * v_a v_b / r
        #
        # The 0.5 is the energy's, not the force's: `W` differentiates `energy`
        # directly, so the (i, j) / (j, i) double counting has to be halved here
        # exactly as it is two lines above.  It does *not* cancel the way it does
        # for the forces, because `v_a v_b` is even under swapping i and j.
        virial = 0.5 * np.einsum("ij,ija,ijb->ab", du_dr / r, vecs, vecs)
        return energy, forces, virial
