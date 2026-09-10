"""Lennard-Jones dispersion/repulsion."""

import numpy as np


def pair_potential(
    r: np.ndarray, sigma: np.ndarray, eps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """LJ and its radial derivative, `(u, du/dr)`.

    `r` must be strictly positive; callers holding a full distance matrix should
    substitute anything on the diagonal and zero the result there, as
    `LennardJones.__call__` does.

    Unit-agnostic, unlike `zbl.pair_potential`: `sigma` carries the length unit
    and `eps` the energy unit, so whatever the caller works in comes back out.
    """
    s6 = (sigma / r) ** 6
    s12 = s6 * s6
    u = 4.0 * eps * (s12 - s6)
    # d/dr [4 eps (s12 - s6)] = (24 eps / r) (s6 - 2 s12)
    du_dr = (24.0 * eps / r) * (s6 - 2.0 * s12)
    return u, du_dr


class LennardJones:
    """LJ summed over every pair, minus the bonded exclusions.

    Mixing is geometric in *both* parameters -- `sigma_ij = sqrt(sigma_i
    sigma_j)`, `eps_ij = sqrt(eps_i eps_j)` -- rather than the more usual
    Lorentz-Berthelot arithmetic mean on `sigma`.  That is what the export
    formats specify: the `CustomNonbondedForce` in the OpenMM XML carries
    `A=sqrt(A1*A2)` in its energy expression, and the DynamicTopology `.jsonl`
    is the same force field.  Mixing differently here would make the exported
    file disagree with the calculator it was fit with.

    Wired like `ACKS2` rather than `ZBL`: the per-atom parameters come out of
    `term_dict`, so the term-order versus global-order distinction applies here
    too -- `indices[k]` is the global index of the k-th term, and the forces are
    scattered back to global order at the end.

    `exclusions` is a boolean `(n_atoms, n_atoms)` mask, in *global* order, that
    is True for pairs whose interaction is already described by a bonded term
    (1-2, 1-3 and 1-4 pairs).  It matches the `<Exclusions>` block of the
    exported XML.
    """

    def __init__(self, exclusions: np.ndarray | None = None):
        self.exclusions: np.ndarray | None = (
            None if exclusions is None else np.asarray(exclusions, dtype=bool)
        )

    def __call__(
        self,
        pos: np.ndarray,
        pbc: np.ndarray,
        cell: np.ndarray,
        term_dict: dict,
    ) -> tuple[float, np.ndarray]:
        lj_params = term_dict.get("lennardjones")
        if lj_params is None:
            return 0.0, np.zeros_like(pos)

        vecs = pos[:, None, :] - pos[None, :, :]
        if np.any(pbc):
            f = vecs @ np.linalg.inv(cell)
            vecs = vecs - (pbc * np.floor(f + 0.5)) @ cell

        indices = lj_params["atoms"][:, 0]
        sigma = lj_params["kwargs"]["sigma"]
        eps = lj_params["kwargs"]["eps"]

        # Both axes in term order, so the parameter vectors line up with them.
        sub = np.ix_(indices, indices)
        vecs = vecs[sub]
        rij = np.sqrt(np.sum(vecs * vecs, -1))

        diag = np.diag_indices(len(indices))
        r = rij.copy()
        r[diag] = 1.0  # excluded below; only keeps the division finite

        sigma_ij = np.sqrt(sigma[:, None] * sigma[None, :])
        eps_ij = np.sqrt(eps[:, None] * eps[None, :])
        u, du_dr = pair_potential(r, sigma_ij, eps_ij)

        keep = np.ones_like(u, dtype=bool)
        keep[diag] = False
        if self.exclusions is not None:
            keep &= ~self.exclusions[sub]
        u = np.where(keep, u, 0.0)
        du_dr = np.where(keep, du_dr, 0.0)

        # The 0.5 cancels for the forces because both (i, j) and (j, i)
        # contribute to dE/d(pos_i) -- the convention `ZBL.__call__` and
        # `ACKS2.compute_coulomb` both use.
        energy = 0.5 * float(np.sum(u))
        nij = vecs / r[:, :, None]
        f_tot = -np.sum(du_dr[:, :, None] * nij, axis=1)

        # scatter term-ordered forces back to global atom order
        forces = np.zeros_like(pos)
        forces[indices] = f_tot
        return energy, forces
