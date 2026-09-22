"""Removing the near-neighbour part of the three whole-system pair sums.

`ZBL`, `LennardJones` and the electrostatics all run over *every* pair with no
reference to the bond graph.  That is what makes them cheap and, more to the
point, what makes them identical on every diabatic state.  It is also wrong for
a pair that is bonded, or one or two bonds further along: a molecule's internal
geometry is set by its bonded terms, and a screened-nuclear repulsion and a 12-6
evaluated at a bond length would set it a second time.  This module takes that
part back off.

**One mask drives all three.**  `Parameters.exclusions` is the boolean
`(n, n)` matrix of pairs within `EXCLUSION_DEPTH` bonds of each other, built by
`topology.exclusion_mask` from the state's own bond graph.  It is the only
topology this module reads, and it is the *second* bond-graph-dependent
contribution to a diabatic diagonal -- the bonded terms being the first, and the
only other one.

**The two halves are not applied the same way, and cannot be.**  The additive
ones subtract; the Coulomb one screens.

  * `additive` evaluates `-u_ZBL` and `-u_126` through the very functions
    `zbl.pair_potential` and `lj.pair_potential` that the whole-system sums use,
    switches included, so the cancellation is exact rather than close.  It was
    open-coded once and the copy silently stopped matching the moment
    `lj.pair_potential` gained its core continuation: the two halves disagreed
    by 1609 eV on an H2 template.

  * Electrostatics cannot simply subtract, because the charges come from a
    solve whose matrix *contains* the kernel.  Masking the kernel inside that
    solve would make the charges a function of the bond graph, which is
    disallowed -- the charges have to be identical on every state, and the
    measured cost of violating it is 0.88 eV of dependence on the arbitrary
    choice of reference state.  So the charges are solved once from the unmasked
    kernel and the exclusion is applied afterwards, as a weight `S` on the
    energy contraction.  See `screen` and `coulomb_correction`.

Angstrom and eV, like the rest of the `forcefield` package.
"""

import numpy as np

from . import lj as _lj
from . import zbl as _zbl
from .ewald import CCOUL

# How far along the bond graph the three whole-system pair sums are switched
# off, in bonds.  3 excludes 1-2, 1-3 and 1-4 pairs, so they act between atoms
# four or more bonds apart and between atoms in different molecules -- GROMACS's
# `nrexcl = 3`, and q-force's own convention.
#
# On HCombustion this is indistinguishable from excluding whole molecules, since
# no template has graph diameter above 3 (H2O2's H...H is the one 1-4 pair, at
# 2.593 A).  It bites on anything larger.
EXCLUSION_DEPTH: int = 3


# Whether the Coulomb sum is screened at all.  Setting it false leaves the
# additive exclusions in place and keeps the full electrostatic contraction,
# which is a different force field and not a cheaper evaluation of this one --
# see `screen` for what the screening is worth.
EXCLUDE_COULOMB: bool = True


def pair_indices(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The excluded pairs as `(i, j)` index arrays with `i < j`, each once.

    `mask` is symmetric with a false diagonal, so taking the upper triangle is
    what turns it into a list of pairs rather than of ordered pairs.
    """
    i, j = np.nonzero(np.triu(np.asarray(mask, dtype=bool), 1))
    return i, j


def lj_parameters(term_dict: dict, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-atom `(sigma, eps)` in global atom order, in Angstrom and eV.

    The `lennardjones` block is in *term* order like every other per-atom block;
    this scatters it back, because `additive` indexes it by the global atom
    numbers the exclusion mask is stated in.  Atoms with no term keep zero,
    which is the value that makes their 12-6 exclusion vanish.
    """
    sigma = np.zeros(n_atoms)
    eps = np.zeros(n_atoms)
    block = term_dict.get("lennardjones")
    if block is not None:
        indices = np.asarray(block["atoms"])[:, 0]
        sigma[indices] = block["kwargs"]["sigma"]
        eps[indices] = block["kwargs"]["eps"]
    return sigma, eps


def _displacements(pos, pbc, cell, i, j):
    """Minimum-image `pos[j] - pos[i]` for a list of pairs."""
    v = pos[j] - pos[i]
    if np.any(pbc):
        frac = v @ np.linalg.inv(cell)
        v = v - (np.asarray(pbc, dtype=float) * np.floor(frac + 0.5)) @ cell
    return v


def additive(
    pos: np.ndarray,
    numbers: np.ndarray,
    pbc: np.ndarray,
    cell: np.ndarray,
    mask: np.ndarray,
    sigma: np.ndarray | None = None,
    eps: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """`-u_ZBL - u_126` over the excluded pairs: energy, forces and virial.

    Both are evaluated through the same `pair_potential` the whole-system sums
    go through, so the switches, the ZBL taper and the 12-6 core continuation
    all cancel exactly rather than approximately.  That exactness is not a
    nicety: `fit` scores a template against its own reference atomization
    energy, and at a bond length the two halves are hundreds of eV apiece.

    A 12-6 exclusion is present only for pairs whose atoms both carry a nonzero
    `sigma` and `eps` -- where either vanishes the whole-system 12-6 is
    identically zero for that pair and there is nothing to cancel.  ZBL admits
    no such exemption: it has no free parameters, only atomic numbers.

    `mask` is in global atom order, and so are `sigma`, `eps` and the returned
    forces.  Pass `sigma=None` for a field with no 12-6 at all.
    """
    i, j = pair_indices(mask)
    forces = np.zeros_like(pos)
    virial = np.zeros((3, 3))
    if len(i) == 0:
        return 0.0, forces, virial

    v = _displacements(pos, pbc, cell, i, j)
    r = np.sqrt(np.sum(v * v, -1))

    z = np.asarray(numbers, dtype=float)
    u, du_dr = _zbl.pair_potential(r, z[i], z[j])

    if sigma is not None and eps is not None:
        sig = np.sqrt(sigma[i] * sigma[j])
        epsilon = np.sqrt(eps[i] * eps[j])
        # `pair_potential` would divide by a zero core radius on a pair with no
        # parameters, and the answer it would be heading for is zero anyway.
        live = (sig > 0.0) & (epsilon > 0.0)
        if np.any(live):
            u_lj, du_lj = _lj.pair_potential(r[live], sig[live], epsilon[live])
            u[live] += u_lj
            du_dr[live] += du_lj

    # The exclusion is the negative of what the whole-system sums charged.
    energy = -float(np.sum(u))
    dE_dv = (-du_dr / r)[:, None] * v

    # v = pos[j] - pos[i], so dE/d(pos_j) = +dE_dv and dE/d(pos_i) = -dE_dv.
    np.add.at(forces, j, -dE_dv)
    np.add.at(forces, i, dE_dv)

    # Every pair is listed once here, unlike the `(i, j)` / `(j, i)` double
    # count the whole-system sums halve, so there is no 0.5 to apply.
    virial = np.einsum("na,nb->ab", v, dE_dv)
    return energy, forces, virial


def screen(masks, weights, n_atoms: int) -> np.ndarray:
    """`S_ij = 1 - sum_s w_s M_ij^s`, the weight the Coulomb sum is contracted on.

    `masks` is one exclusion mask per state and `weights` the states'
    ground-state weights; a single-state force field passes one mask and a
    weight of one, and gets a plain 0/1 screen.

    **Fractional entries are the normal case and are not an interpolation.**
    Each state's correction is linear in its own mask and the contraction is
    linear in its weight, so the Hellmann-Feynman sum `sum_s w_s dE_s/dr`
    collapses into this single weight matrix exactly.  A pair bonded in every
    state comes out at exactly `S = 0`; a pair bonded in some of them is removed
    in proportion.  `S` is then held fixed under the derivative, which is what
    Hellmann-Feynman prescribes for eigenvector components.

    **What the screening is worth.**  It removes the intramolecular half of the
    polarization response -- about 0.07 eV of a hydrogen bond.  The water-dimer
    well moves from 0.1677 eV at 2.85 A to 0.1024 eV at 2.91 A when it is
    applied: two approaching molecules polarize each other (`q_H` rises from
    +0.30399 to +0.31264 at 2.85 A) and the *intramolecular* Coulomb energy
    falls along with the intermolecular one, and booking that intramolecular
    gain is exactly what this prevents.  The charges themselves are untouched --
    an isolated water still gives `q_H = +0.30399` -- so no molecular dipole
    moves, and `eta` is the lever that pays the depth back.
    """
    s = np.ones((n_atoms, n_atoms))
    for mask, weight in zip(masks, weights, strict=True):
        if mask is None:
            continue
        s -= float(weight) * np.asarray(mask, dtype=float)
    return s


def coulomb_correction(Q, kernel_matrix, mask, ccoul: float = CCOUL) -> float:
    """`-ccoul * sum_{(i,j) in excl} Q_i Q_j K_ij`, one state's diagonal term.

    This is the piece that lets the exclusion influence *which* bonding pattern
    is lower, which the screened contraction on its own cannot do: the screen is
    built from the weights and so is downstream of the eigenproblem the diagonal
    feeds.  The two are the same quantity written twice -- summing this over the
    states with their weights and adding the unscreened contraction reproduces
    `(ccoul/2) sum_ij S_ij Q_i Q_j K_ij` identically -- which is why `EVB` puts
    this on the diagonal and evaluates the energy and forces from the screen.

    `Q`, `kernel_matrix` and `mask` are all in the same order; `ACKS2` and
    `Coulomb` work in term order, so the caller restricts the global mask to the
    electrostatic block's atoms before calling.
    """
    i, j = pair_indices(mask)
    if len(i) == 0:
        return 0.0
    return -ccoul * float(np.sum(Q[i] * Q[j] * np.asarray(kernel_matrix)[i, j]))
