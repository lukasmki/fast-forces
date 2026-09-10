"""Universal screened-nuclear repulsion"""

import numpy as np

# Screening length prefactor, `0.8854 * a_0`, in Angstrom.
SCREENING_LENGTH: float = 0.46850

# Coulomb constant in eV*Angstrom, matching `ACKS2.CCOUL` to the digits ASE uses.
CCOUL: float = 14.399645

# The universal screening function `phi(x) = sum_k C[k] * exp(-B[k] * x)`.
PHI_C: tuple[float, ...] = (0.18175, 0.50986, 0.28022, 0.02817)
PHI_B: tuple[float, ...] = (3.19980, 0.94229, 0.40290, 0.20162)


def pair_potential(
    r: np.ndarray, z1: np.ndarray, z2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """ZBL and its radial derivative, `(u, du/dr)`, in eV and eV/Angstrom.

    `r` must be strictly positive; callers holding a full distance matrix should
    substitute anything on the diagonal and zero the result there, as
    `ZBL.__call__` does.

    Not unit-agnostic, unlike `lj.pair_potential`: `SCREENING_LENGTH` is in
    Angstrom and `CCOUL` in eV*Angstrom, so `r` has to be in Angstrom and the
    result comes back in eV.  There is only one caller and it works in ASE
    units, so there is nothing for this to drift against.
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
    return u, du_dr


class ZBL:
    """ZBL repulsion summed over every pair, with no exclusions.

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
    ) -> tuple[float, np.ndarray]:
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
        return energy, forces
