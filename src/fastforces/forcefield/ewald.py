"""The ACKS2 charge kernel, with and without the periodic lattice sum.

ACKS2 needs the smeared-charge kernel

    g(r) = erf(GAMMA * r) / r

in three places: as the Coulomb block of its linear system, as the energy
`E = CCOUL/2 * Q.K.Q`, and as the `dE/dQ` that drives the charge-response
adjoint.  Under periodic boundary conditions each of those is a lattice sum
over every image rather than a nearest-image pair term, so this module puts the
two behind one interface and `acks2.py` never branches on `pbc`:

    kernel.matrix()    -> `K`, the (n, n) kernel matrix
    kernel.contract(W) -> `(dS/dr_i, dS/de_ab)` for `S = sum_ij W_ij K_ij`

`contract` is what makes the abstraction worth having.  Both places ACKS2
differentiates the kernel -- the explicit Coulomb force and the
`-lam^T (dA/dr) x` response term -- are of exactly that form and differ only in
the symmetric weight matrix `W` they contract against, so the periodic
derivatives are written once instead of once per caller.  Under `MinimumImage`
that contraction is the pair sum the old inline code did; under `Ewald` it also
carries a reciprocal-space term that is *not* a sum over pair separations, and
which is why the kernel had to become an object rather than a matrix.

**The splitting.**  Writing `erf(a r) = 1 - erfc(a r)` twice turns the periodic
sum of `g` into a standard Ewald sum of `1/r` minus a short-ranged remainder,
and the two real-space pieces merge:

    K_ij = sum_n' [erf(GAMMA |r_ij + n|) - erf(kappa |r_ij + n|)] / |r_ij + n|
         + (4 pi / V) sum_{k != 0} exp(-k^2 / 4 kappa^2) / k^2 * cos(k . r_ij)
         - delta_ij * 2 kappa / sqrt(pi)

The prime excludes `n = 0` when `i == j`: an atom does not interact with itself,
but it *does* interact with its own images, and that is the diagonal `K_ii`
below -- a term with no counterpart in the open-boundary kernel, carried on the
diagonal of the ACKS2 matrix alongside the atomic hardness.

**Charge neutrality is assumed, not enforced here.**  The `k = 0` term of the
reciprocal sum is the divergent one; regularizing it against a neutralizing
background leaves a constant `-pi / (V kappa^2)` in every entry of `K`.  Adding
any constant to every entry of `K` changes the energy by that constant times
`(sum_i q_i)^2` and the ACKS2 rows by that constant times `sum_i q_i`, so with
`sum_i q_i = 0` -- which `ACKS2.build_system` imposes as a hard constraint -- it
drops out of the energy, the forces, the virial and the solved charges alike.
It is therefore omitted.  A charged system would need it back, and would need a
physical justification for what the compensating background is.

**Only the Coulomb block is summed over images.**  ACKS2's other
geometry-dependent block, the bond softness, decays as `exp(-r / tau)` with
`tau ~ 0.3 A`; at the nearest-image cutoff of a 9 A cell that is `e^-15`, so it
stays a nearest-image pair term and no periodic treatment is needed.
"""

import numpy as np
from scipy.special import erf


# Charge-smearing width of the ACKS2 kernel `erf(GAMMA r) / r`, in 1/Angstrom.
GAMMA = 2.0

# Target relative error of the truncated lattice sum.  It sets `kappa` (through
# the real-space sum, which is truncated at the nearest image) and the
# reciprocal cutoff together, so both halves are converged to the same level.
#
# Do not loosen this casually.  The cost is a cube root -- the reciprocal
# vector count scales as `(-log(accuracy))^3` -- but the *stress* pays for it
# linearly: `kappa` is derived from the cell, so a strained cell is summed with
# a slightly different splitting, and the residual `d(truncation error)/d(cell)`
# is a spurious contribution to the finite-difference virial that the analytic
# expression (taken at fixed `kappa`, correctly, since the exact sum does not
# depend on it) has no counterpart for.  That error is amplified by
# `2 log(1/accuracy)` relative to the truncation error itself.  At 1e-8 it lands
# around 1e-6 eV, comfortably under what `tests/test_stress.py` asserts; at 1e-6
# it would be at the tolerance.
ACCURACY = 1e-8


def _screened(rij, r, alpha):
    """`erf(alpha r) / r` and its radial derivative, on every pair at once.

    `r` is `rij` shifted off zero so the diagonal divides safely.  Both results
    are zeroed there explicitly: the kernel vanishes with its numerator anyway,
    but the derivative does not -- it tends to `2 alpha / sqrt(pi) / eps`, which
    is `4e15` rather than something a stray multiply would forgive.
    """
    diag = np.diag_indices_from(rij)
    k = erf(alpha * rij) / r
    dk = (2 * alpha / np.sqrt(np.pi)) * np.exp(-((alpha * rij) ** 2)) / r - k / r
    k[diag] = 0.0
    dk[diag] = 0.0
    return k, dk


def contract_pairs(coeff, vecs, r):
    """`dS/dr_i` and `dS/de_ab` for a pair sum `S = sum_ij W_ij f(r_ij)`.

    `coeff` is `W * f'(r)`, already multiplied out by the caller, which is what
    lets the softness block in `acks2.py` share this with the Coulomb kernels.

    The factor of two on the force is the pair double count: entries `(i, j)`
    and `(j, i)` both move with `r_i`.  The virial has no such factor because
    `v_a v_b` is even under the swap and so has nothing to cancel against --
    the same asymmetry `acks2.py` documents at length, kept in one place here.
    """
    nij = vecs / r[:, :, None]
    dS_dr = 2.0 * np.sum(coeff[:, :, None] * nij, axis=1)
    dS_de = np.einsum("ij,ija,ijb->ab", coeff / r, vecs, vecs)
    return dS_dr, dS_de


class MinimumImage:
    """`erf(GAMMA r) / r` over the nearest image of each pair and nothing else.

    The open-boundary kernel, and the fallback for a partially periodic cell:
    Ewald in fewer than three dimensions is a different summation altogether,
    so a slab keeps the behaviour it had rather than silently getting a 3D sum.

    `K_ii` is zero -- with no images there is nothing for an atom to interact
    with but the other atoms.
    """

    def __init__(self, rij, vecs=None):
        self.rij = rij
        self.vecs = vecs
        self.r = rij + np.finfo(np.float64).eps

    def matrix(self):
        return _screened(self.rij, self.r, GAMMA)[0]

    def contract(self, W):
        dk = _screened(self.rij, self.r, GAMMA)[1]
        return contract_pairs(W * dk, self.vecs, self.r)


class Ewald:
    """Cell-dependent setup for the periodic kernel; `bind` attaches a geometry.

    Split in two because the expensive part -- choosing `kappa` and enumerating
    the reciprocal vectors -- depends only on the cell, while the trigonometric
    tables depend on the positions.  `ACKS2` keeps one of these per cell.

    **Why the reciprocal vectors are chosen by integer index and not by `|k|`.**
    The obvious selection, `|k| <= k_max`, is discontinuous in the cell: a
    strain of 1e-6 moves a whole degenerate shell of a cubic lattice across the
    cutoff at once, and the energy jumps.  Selecting an *integer* ellipsoid
    whose semi-axes come from `ceil(k_max / |b_i|)` instead makes the set
    piecewise constant in the cell, and the pieces meet where the vectors that
    join or leave weigh `accuracy` apiece.  The ellipsoid contains the sphere,
    so this only ever sums more than asked for.
    """

    def __init__(self, cell, accuracy=ACCURACY):
        self.cell = np.asarray(cell, dtype=float)
        self.volume = abs(np.linalg.det(self.cell))
        if self.volume <= 0.0:
            raise ValueError("Ewald summation needs a cell with nonzero volume")

        # Rows of `recip` are the reciprocal lattice vectors b_j, so that
        # a_i . b_j = 2 pi delta_ij and k = m . recip for an integer triple m.
        recip = 2 * np.pi * np.linalg.inv(self.cell).T
        blen = np.linalg.norm(recip, axis=1)

        # The real-space sum is truncated at the nearest image, so the cutoff
        # is half the *perpendicular* width of the cell -- 2 pi / |b_i|, which
        # for a skewed cell is shorter than any lattice vector.
        span = np.sqrt(-np.log(accuracy))
        cutoff = 0.5 * (2 * np.pi / blen).min()

        # kappa is the smallest splitting that leaves the nearest image enough:
        # smallest, because every reciprocal vector is paid for on every atom
        # pair, so the reciprocal sum is the expensive half here.  Capping it at
        # GAMMA covers the cell so small that no splitting converges in one
        # image; there the real-space term vanishes identically and the whole
        # kernel is summed in reciprocal space, more slowly but correctly.
        self.kappa = min(span / cutoff, GAMMA)

        kmax = 2 * self.kappa * span
        nmax = np.maximum(np.ceil(kmax / blen).astype(int), 1)
        axes = [np.arange(-n, n + 1) for n in nmax]
        m = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

        # Half of reciprocal space, doubled in `weight` below: every quantity
        # here is even in k, so k and -k contribute identically.  The half-space
        # test (first nonzero component positive) drops k = 0 along with it.
        half = (m[:, 0] > 0) | (
            (m[:, 0] == 0) & ((m[:, 1] > 0) | ((m[:, 1] == 0) & (m[:, 2] > 0)))
        )
        inside = np.sum((m / nmax) ** 2, axis=1) <= 1.0
        self.kvecs = m[half & inside] @ recip
        self.k2 = np.sum(self.kvecs**2, axis=1)

        # 4 pi / (V k^2) is the Fourier transform of 1/r, the Gaussian is the
        # k-space image of the screening, and the 2 is the half-space fold.
        self.weight = (
            (8 * np.pi / self.volume) * np.exp(-self.k2 / (4 * self.kappa**2)) / self.k2
        )

        # Removes the k = 0 image of an atom's own screening charge, which the
        # reciprocal sum includes and which is not a real interaction.
        self.self_term = -2 * self.kappa / np.sqrt(np.pi)

    def bind(self, pos, vecs, rij):
        return EwaldKernel(self, pos, vecs, rij)


class EwaldKernel:
    """The periodic kernel at one geometry.

    `pos` is in ACKS2's term order, the same order as `vecs` and `rij`; only
    differences of positions ever enter, and `cos(k . r)` is invariant under a
    lattice translation, so neither the origin nor whether an atom has been
    wrapped into the cell matters.
    """

    def __init__(self, setup, pos, vecs, rij):
        self.setup = setup
        self.vecs = vecs
        self.rij = rij
        self.r = rij + np.finfo(np.float64).eps
        self._matrix = None

        # The structure factor, factorized: cos(k . r_ij) = c_i c_j + s_i s_j
        # turns every k-sum below into a matrix product over atoms, which is
        # what keeps the cost at one dgemm rather than n^2 trigonometric calls.
        kr = pos @ setup.kvecs.T
        self.cos = np.cos(kr)
        self.sin = np.sin(kr)

    def matrix(self):
        """`K_ij`, the kernel summed over every image.  Memoized: `ACKS2` asks
        for it once to build its linear system and once more for `dE/dQ`."""
        if self._matrix is None:
            setup = self.setup
            short = (
                _screened(self.rij, self.r, GAMMA)[0]
                - _screened(self.rij, self.r, setup.kappa)[0]
            )
            weighted_cos = self.cos * setup.weight
            weighted_sin = self.sin * setup.weight
            K = short + weighted_cos @ self.cos.T + weighted_sin @ self.sin.T
            K[np.diag_indices_from(K)] += setup.self_term
            self._matrix = K
        return self._matrix

    def contract(self, W):
        """`dS/dr_i` and `dS/de_ab` for `S = sum_ij W_ij K_ij`, `W` symmetric.

        The reciprocal half is where this stops being a pair sum.  Its virial
        comes from the two ways a strain reaches it: the `1/V` prefactor, and
        the reciprocal vectors themselves, which transform *inversely* to the
        positions -- `k -> (I - e) k`, so `d(k^2)/de_ab = -2 k_a k_b`.  The
        structure factor is untouched, since `k . r` is invariant, which is why
        no analogue of the real-space `v_a v_b` term appears.
        """
        setup = self.setup

        dshort = (
            _screened(self.rij, self.r, GAMMA)[1]
            - _screened(self.rij, self.r, setup.kappa)[1]
        )
        dS_dr, dS_de = contract_pairs(W * dshort, self.vecs, self.r)

        Wc = W @ self.cos
        Ws = W @ self.sin
        # Per-k contribution to S, from  sum_ij W_ij cos(k . r_ij).
        Sk = setup.weight * (
            np.einsum("ik,ik->k", self.cos, Wc) + np.einsum("ik,ik->k", self.sin, Ws)
        )

        # d/dr_i sum_ij W_ij cos(k . r_ij) = -2 k [s_i (Wc)_i - c_i (Ws)_i]
        amp = self.sin * Wc - self.cos * Ws
        dS_dr = dS_dr - 2.0 * ((amp * setup.weight) @ setup.kvecs)

        coeff = 2.0 * Sk * (1.0 / (4 * setup.kappa**2) + 1.0 / setup.k2)
        dS_de = (
            dS_de
            + np.einsum("k,ka,kb->ab", coeff, setup.kvecs, setup.kvecs)
            - np.eye(3) * Sk.sum()
        )
        return dS_dr, dS_de
