import numpy as np
from typing import Callable


def _kabsch(frozen: np.ndarray, mobile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation superposing `mobile` onto `frozen`, batched."""
    n = frozen.shape[0]
    center_f = np.sum(frozen, axis=0) / n
    center_m = np.sum(mobile, axis=1) / n
    shifted_f = frozen - center_f
    shifted_m = mobile - center_m[:, None, :]

    # Diamond's M (eq. 16), then Q (17), V (18) and the 4x4 P (22) whose
    # dominant eigenvector is the optimal rotation as a quaternion.
    M = np.einsum("mni,nj->mij", shifted_m, shifted_f)
    trace = np.trace(M, axis1=1, axis2=2)
    Q = M + M.transpose(0, 2, 1) - 2.0 * np.eye(3) * trace[:, None, None]
    V = np.stack(
        (
            M[:, 1, 2] - M[:, 2, 1],
            M[:, 2, 0] - M[:, 0, 2],
            M[:, 0, 1] - M[:, 1, 0],
        ),
        axis=-1,
    )

    P = np.zeros(mobile.shape[:1] + (4, 4))
    P[:, :3, :3] = Q
    P[:, 3, :3] = V
    P[:, :3, 3] = V

    if n < 2:
        # A single point has no orientation to solve for; Superpose3D's own
        # `singular` branch falls back to the identity quaternion here.
        p = np.zeros(mobile.shape[:1] + (4,))
        p[:, 3] = 1.0
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(P)
        index = np.argmax(eigenvalues, axis=-1)
        p = np.take_along_axis(eigenvectors, index[:, None, None], axis=-1)[:, :, 0]
        p = p / np.linalg.norm(p, axis=-1, keepdims=True)

    p0, p1, p2, p3 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    R = np.empty(mobile.shape[:1] + (3, 3))
    R[:, 0, 0] = p0 * p0 - p1 * p1 - p2 * p2 + p3 * p3
    R[:, 1, 1] = -p0 * p0 + p1 * p1 - p2 * p2 + p3 * p3
    R[:, 2, 2] = -p0 * p0 - p1 * p1 + p2 * p2 + p3 * p3
    R[:, 0, 1] = 2.0 * (p0 * p1 - p2 * p3)
    R[:, 1, 0] = 2.0 * (p0 * p1 + p2 * p3)
    R[:, 1, 2] = 2.0 * (p1 * p2 - p0 * p3)
    R[:, 2, 1] = 2.0 * (p1 * p2 + p0 * p3)
    R[:, 0, 2] = 2.0 * (p0 * p2 + p1 * p3)
    R[:, 2, 0] = 2.0 * (p0 * p2 - p1 * p3)

    T = center_f - np.einsum("mij,mj->mi", R, center_m)
    return R, T


class EVBCoupling:
    def __init__(self):
        pass

    def __call__(
        self,
        pos: np.ndarray,
        pbc: np.ndarray,
        cell: np.ndarray,
        ensemble: np.ndarray,
        term_dict: dict,
        inv_cell: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Coupling of one fragment, or of a stack of them.

        `pos` is (n, 3) for a single fragment or (m, n, 3) for a batch that
        shares a reaction template; the returns carry the same leading axis.
        Every channel of one template in one force call has the same `n` and the
        same `ensemble`, which is what makes the batch worth forming -- see
        `_kabsch`.

        `inv_cell` is the caller's cached `inv(cell)`.  The cell is fixed for a
        whole force call while this is entered once per channel, so inverting it
        here was a 3x3 `solve` repeated thousands of times per step.
        """
        pos = np.asarray(pos, dtype=float)
        batched = pos.ndim == 3
        if not batched:
            pos = pos[None]

        if np.any(pbc):  # unwrap coordinates
            if inv_cell is None:
                inv_cell = np.linalg.inv(cell)
            frac = pos @ inv_cell
            diffs = np.diff(frac, axis=1)
            shift = diffs.round()
            frac[:, 1:] = frac[:, :1] + np.cumsum(diffs - shift, axis=1)
            pos = frac @ np.asarray(cell)

        e = np.zeros(pos.shape[0])
        f = np.zeros_like(pos)
        for term_type, param_dict in term_dict.items():
            fn: Callable | None = getattr(self, f"compute_{term_type}", None)
            if fn is None:
                continue
            de, df = fn(pos, ensemble, param_dict["atoms"], **param_dict["kwargs"])
            e += de
            f += df

        w = -np.einsum("mna,mnb->mab", pos, f)

        if not batched:
            return float(e[0]), f[0], w[0]
        return e, f, w

    def compute_rmsd(self, pos, ensemble, atoms, A, a):
        """Gaussian in the optimally superposed RMSD to each template frame.

        `pos` is (m, n, 3) and `ensemble` is (E, n, 3); returns the ensemble
        mean energy (m,) and its force (m, n, 3).
        """
        nbatch, natoms, _ = pos.shape
        rmsd = np.empty((nbatch, len(ensemble)))
        drmsd = np.empty((nbatch, len(ensemble), natoms, 3))
        for i in range(len(ensemble)):
            R, T = _kabsch(ensemble[i], pos)
            ppos = np.einsum("mni,mji->mnj", pos, R) + T[:, None, :]

            distsq = np.sum(np.square(ensemble[i] - ppos), -1)
            rmsd[:, i] = np.sqrt(np.mean(distsq, axis=-1))
            # d(rmsd)/d(pos) = (1/(n*rmsd)) * (ppos - ensemble) @ R
            drmsd[:, i] = (
                (1.0 / (rmsd[:, i] + np.finfo(np.float64).eps))[:, None, None]
                / natoms
                * np.einsum("mni,mij->mnj", ppos - ensemble[i], R)
            )
        e = A * np.exp(-a * rmsd**2)
        # The ensemble axis is broadcast explicitly rather than relied upon:
        # `e` is (m, E) against a (m, E, n, 3) gradient, which numpy would only
        # line up by accident at E = 1 or E = 3.
        f = (-1 * e * -a * 2 * rmsd)[..., None, None] * drmsd
        return np.mean(e, axis=-1), np.mean(f, axis=1)

    def compute_twobody(self, pos, ensemble, atoms, A, a, r0):
        i, j = atoms[:, 0], atoms[:, 1]
        v = pos[:, j, :] - pos[:, i, :]  # (m, t, 3)  vec from atom0 -> atom1
        r = np.sqrt(np.sum(v * v, -1))  # (m, t)
        dr = r - r0
        e = A * np.exp(-a * dr**2)
        # dE/dr = -2*a*dr*E, and dr/dpos_j = +v/r, so f_j = -dE/dpos_j = +2*a*dr*E*v/r.
        dv = (2.0 * a * dr * e / r)[..., None] * v
        f = np.zeros_like(pos)
        np.add.at(f, (slice(None), j), dv)
        np.add.at(f, (slice(None), i), -dv)
        return np.sum(e, axis=-1), f

    def compute_threebody(self, pos, ensemble, atoms, A, a, ra0, rb0, t0):
        d_i, h_i, a_i = atoms[:, 0], atoms[:, 1], atoms[:, 2]
        va = pos[:, h_i, :] - pos[:, d_i, :]  # (m, t, 3)  donor  -> proton
        vb = pos[:, h_i, :] - pos[:, a_i, :]  # acceptor -> proton
        vd = pos[:, a_i, :] - pos[:, d_i, :]  # donor -> acceptor
        ra = np.sqrt(np.sum(va * va, -1))
        rb = np.sqrt(np.sum(vb * vb, -1))
        d = np.sqrt(np.sum(vd * vd, -1))

        # Law of cosines on the reference triangle.  `t0` is the angle at the
        # proton, so the side opposite it is the donor-acceptor separation.
        d0 = np.sqrt(ra0**2 + rb0**2 - 2.0 * ra0 * rb0 * np.cos(t0))

        g = (ra - ra0) ** 2 + (rb - rb0) ** 2 + (d - d0) ** 2
        e = A * np.exp(-a * g)

        # dE/dg = -a*E, and dg/dra = 2*(ra - ra0), so the force along each side
        # is `2*a*E*(r - r0)/r` times that side's vector.
        ca = (2.0 * a * e * (ra - ra0) / ra)[..., None]
        cb = (2.0 * a * e * (rb - rb0) / rb)[..., None]
        cd = (2.0 * a * e * (d - d0) / d)[..., None]

        f = np.zeros_like(pos)
        np.add.at(f, (slice(None), h_i), ca * va + cb * vb)
        np.add.at(f, (slice(None), d_i), -ca * va - cd * vd)
        np.add.at(f, (slice(None), a_i), -cb * vb + cd * vd)
        return np.sum(e, axis=-1), f
