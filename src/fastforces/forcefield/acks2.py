import numpy as np

from .ewald import CCOUL, KernelCache, MinimumImage, contract_pairs, coulomb_sum


class ACKS2:
    """Charge-equilibration electrostatics.

    Two index spaces meet here and must not be confused.  The per-atom
    parameters (`mu`, `eta`, `soft_amp`, `soft_decay`) arrive in *term order* --
    the order the `atom` terms were collected -- while `pos` and the returned
    forces are in *global* atom order.  `indices[k]` is the global index of the
    k-th term.  Everything below is built in term order, and the forces are
    scattered back to global order only at the very end.

    Getting this wrong is silent: term order coincides with global order
    whenever the atom terms happen to be collected in index order, which is the
    common case, so the error only appears once a molecule is matched onto the
    live system in a different order.

    **The charge kernel is an object, not a matrix**, and everything that
    touches the geometry goes through it -- see `forcefield/ewald.py`.  Under
    open or partially periodic boundaries it is `MinimumImage`, the nearest-image
    `erf(2 r) / r` this class used to inline.  Under full periodicity it is
    `Ewald`, which sums that kernel over every image; the two differ in the
    energy, in the forces, and in the fact that the periodic `K_ii` is nonzero,
    because an atom does interact with its own images even though it does not
    interact with itself.  A method here that needs the kernel takes it as an
    optional argument and falls back to `MinimumImage`, so calling any of them
    with a bare `rij` still means the open-boundary problem.
    """

    # eV*Angstrom.  Shared with `Coulomb` and with `export.openmm`, so it is
    # defined in `ewald` and only re-exposed here.
    CCOUL = CCOUL

    def __init__(self):
        self.Q = None
        self.u = None
        self.A = None
        self.state_hash = None
        self.kernels = KernelCache()

    def build_system(self, rij, params, kernel=None):
        """Assemble the ACKS2 linear system `A x = b`, in term order.

        `x` is `[Q, u, lambda_total, lambda_KS]`: the charges, the Kohn-Sham
        potentials conjugate to them, and the two constraint multipliers.  `A`
        is symmetric, which is what lets the force adjoint below reuse it
        untransposed.

        Only two blocks of `A` depend on the geometry -- the Coulomb block and
        the softness block -- and `compute_response_forces` differentiates
        exactly those two.  Anything geometry-dependent added here must be
        differentiated there as well, or the forces stop being the gradient of
        the energy.

        The hardness is *added* to the Coulomb diagonal rather than overwriting
        it.  With open boundaries there is nothing there to overwrite, but the
        periodic kernel carries an atom's interaction with its own images on
        that diagonal, and it belongs in the equilibration alongside `2 eta`.
        """
        natoms = rij.shape[0]
        neqns = 2 * natoms + 2
        diag = np.diag_indices(natoms)
        atom = np.arange(natoms)
        if kernel is None:
            kernel = MinimumImage(rij)

        A = np.zeros((neqns, neqns))
        b = np.zeros(neqns)

        # interaction
        A[:natoms, :natoms] = kernel.matrix()

        # softness
        amp, decay = params["soft_amp"], params["soft_decay"]
        X0 = amp[:, None] * amp[None, :]
        tau = 0.5 * (decay[:, None] + decay[None, :])
        bsoft = X0 * np.exp(-rij / tau)
        bsoft[diag] = 0.0
        bsoft[diag] = -1 * np.sum(bsoft, axis=1)
        A[natoms : 2 * natoms, natoms : 2 * natoms] = bsoft

        # coupling
        A[:natoms, natoms : 2 * natoms] = -np.eye(natoms)
        A[natoms : 2 * natoms, :natoms] = -np.eye(natoms)

        # diagonal
        A[atom, atom] += 2.0 * params["eta"]
        b[:natoms] = -params["mu"]

        # Constraints
        # KS coeffs
        A[-1, atom + natoms] = -1
        A[atom + natoms, -1] = -1
        b[-1] = 0.0

        # total charge
        A[-2, atom] = -1
        A[atom, -2] = -1
        b[-2] = 0.0

        return A, b

    def solve_charges(self, rij, params, kernel=None):
        """Charges, KS potentials and the system matrix.  All in term order."""
        natoms = rij.shape[0]
        A, b = self.build_system(rij, params, kernel)
        x = np.linalg.solve(A, b)
        return x[:natoms], x[natoms : 2 * natoms], A

    def compute_charges(self, rij, params, kernel=None):
        """Solve the ACKS2 linear system.  All arguments are in term order."""
        return self.solve_charges(rij, params, kernel)[0]

    def compute_coulomb(self, Q, rij, vecs, kernel=None):
        """Coulomb energy, forces and virial.  All arguments and results in term order.

        The charges are held fixed here, so this is only the explicit part of
        the gradient.  `compute_response_forces` supplies the dQ/dr part, and
        `__call__` adds the two; this method on its own is not the gradient of
        its own energy.  That is the whole difference between this term and
        `Coulomb`, which shares the sum below and stops there because its
        charges really are fixed.
        """
        if kernel is None:
            kernel = MinimumImage(rij, vecs)
        return coulomb_sum(Q, kernel, self.CCOUL)

    def compute_response_forces(self, Q, u, A, rij, vecs, params, kernel=None):
        """The dQ/dr part of the force, and its virial.  All arguments and results in term order.

        The charges are not independent of the geometry: they solve `A(r) x = b`
        with `b` geometry-free, so moving an atom moves every charge.  The
        energy `E = CCOUL/2 * Q.K.Q` is not stationary in `Q` -- the ACKS2
        functional is, but this Coulomb piece alone is not -- so that motion
        contributes to `dE/dr` and cannot be dropped.  Omitting it is what made
        the electrostatic force disagree with its own energy for every species
        with nonzero charges, and NVE energy drift the symptom.

        Differentiating the solve gives `dx/dr = -A^-1 (dA/dr) x`, so

            dE/dr = (dE/dr)|_Q  +  (dE/dx) . dx/dr
                  = (dE/dr)|_Q  -  lam^T (dA/dr) x,   lam = A^-1 (dE/dx)

        which needs one extra solve rather than one per coordinate.  `A` is
        symmetric, so no transpose is required.

        `dA/dr` is nonzero only in the two blocks `build_system` builds from the
        geometry, and each is differentiated by contracting it against the
        symmetric weight matrix that `-lam^T (dA/dr) x` puts on it.  The
        softness block needs care: `X_ii = -sum_j X_ij`, so each off-diagonal
        `bsoft_ij` appears in four entries of `X` and all four contribute.
        """
        natoms = len(Q)
        diag = np.diag_indices(natoms)
        r = rij + np.finfo(np.float64).eps
        if kernel is None:
            kernel = MinimumImage(rij, vecs)

        # dE/dx, nonzero only on the charge block.  The multiplier rows are
        # geometry-free and the energy does not depend on u.
        gradient = np.zeros(A.shape[0])
        gradient[:natoms] = self.CCOUL * (kernel.matrix() @ Q)
        lam = np.linalg.solve(A, gradient)
        lam_q, lam_u = lam[:natoms], lam[natoms : 2 * natoms]

        # -lam^T (dA/dr) x for the Coulomb block, as a weight on the kernel.
        # Entry (i, j) and entry (j, i) each hold the whole pair term, so the
        # weight is halved to match the convention `compute_coulomb` uses.
        W = -0.5 * (lam_q[:, None] * Q[None, :] + lam_q[None, :] * Q[:, None])
        coulomb_dr, coulomb_de = kernel.contract(W)

        # The same for the softness block, which stays a nearest-image pair
        # term at every boundary condition -- see `ewald.py`.
        amp, decay = params["soft_amp"], params["soft_decay"]
        tau = 0.5 * (decay[:, None] + decay[None, :])
        dbsoft_dr = -(amp[:, None] * amp[None, :]) * np.exp(-rij / tau) / tau
        dbsoft_dr[diag] = 0.0
        W_soft = -0.5 * (
            lam_u[:, None] * u[None, :]
            + lam_u[None, :] * u[:, None]
            - lam_u[:, None] * u[:, None]
            - lam_u[None, :] * u[None, :]
        )
        soft_dr, soft_de = contract_pairs(W_soft * dbsoft_dr, vecs, r)

        # Both blocks are contracted as `dS/dr`; the force is minus that.
        return -(coulomb_dr + soft_dr), coulomb_de + soft_de

    def get_kernel(self, pos, vecs, rij, pbc, cell):
        """The charge kernel for these boundary conditions, in term order."""
        return self.kernels.get(pos, vecs, rij, pbc, cell)

    def __call__(
        self, pos, pbc, cell, term_dict: dict
    ) -> tuple[float, np.ndarray, np.ndarray]:
        pbc = np.asarray(pbc, dtype=bool)
        vecs = pos[:, None, :] - pos[None, :, :]
        if np.any(pbc):
            F = vecs @ np.linalg.inv(cell)
            vecs = vecs - (pbc * np.floor(F + 0.5)) @ cell
        atom_params = term_dict.get("atom")
        if atom_params is None:
            raise KeyError("No atom parameters set")
        indices = atom_params["atoms"][:, 0]
        params = atom_params["kwargs"]

        # Both axes in term order, so the parameter vectors line up with them.
        sub = np.ix_(indices, indices)
        vecs = vecs[sub]
        rij = np.sqrt(np.sum(vecs * vecs, -1))
        kernel = self.get_kernel(pos[indices], vecs, rij, pbc, cell)

        # Cache on the parameters and the cell as well as the geometry: the same
        # positions with a different set of atom terms is a different problem,
        # and so is the same system in a cell a barostat has just rescaled.
        state_hash = hash(
            (pos.tobytes(), indices.tobytes(), np.asarray(cell, dtype=float).tobytes())
        )
        if self.Q is None or state_hash != self.state_hash:
            self.Q, self.u, self.A = self.solve_charges(rij, params, kernel)
            self.state_hash = state_hash

        e_tot, f_tot, w_tot = self.compute_coulomb(self.Q, rij, vecs, kernel)
        f_resp, w_resp = self.compute_response_forces(
            self.Q, self.u, self.A, rij, vecs, params, kernel
        )
        f_tot = f_tot + f_resp
        w_tot = w_tot + w_resp

        # scatter term-ordered forces back to global atom order.  The virial
        # needs no scatter: it is a single 3x3 sum over pairs, not a per-atom
        # quantity, so term order and global order give the same matrix.
        forces = np.zeros_like(pos)
        forces[indices] = f_tot
        return e_tot, forces, w_tot
