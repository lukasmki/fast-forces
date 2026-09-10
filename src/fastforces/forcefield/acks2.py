from ase import units
from scipy.special import erf
import numpy as np


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
    """

    CCOUL = 14.4  # eV

    def __init__(self):
        self.Q = None
        self.u = None
        self.A = None
        self.state_hash = None

    def build_system(self, rij, params):
        """Assemble the ACKS2 linear system `A x = b`, in term order.

        `x` is `[Q, u, lambda_total, lambda_KS]`: the charges, the Kohn-Sham
        potentials conjugate to them, and the two constraint multipliers.  `A`
        is symmetric, which is what lets the force adjoint below reuse it
        untransposed.

        Only two blocks of `A` depend on the geometry -- the off-diagonal
        Coulomb block and the softness block -- and `compute_response_forces`
        differentiates exactly those two.  Anything geometry-dependent added
        here must be differentiated there as well, or the forces stop being the
        gradient of the energy.
        """
        natoms = rij.shape[0]
        neqns = 2 * natoms + 2
        diag = np.diag_indices(natoms)
        atom = np.arange(natoms)

        A = np.zeros((neqns, neqns))
        b = np.zeros(neqns)

        # interaction
        A[:natoms, :natoms] = erf(2 * rij) / (rij + np.finfo(np.float64).eps)

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
        A[atom, atom] = 2.0 * params["eta"]
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

    def solve_charges(self, rij, params):
        """Charges, KS potentials and the system matrix.  All in term order."""
        natoms = rij.shape[0]
        A, b = self.build_system(rij, params)
        x = np.linalg.solve(A, b)
        return x[:natoms], x[natoms : 2 * natoms], A

    def compute_charges(self, rij, params):
        """Solve the ACKS2 linear system.  All arguments are in term order."""
        return self.solve_charges(rij, params)[0]

    def compute_coulomb(self, Q, rij, vecs):
        """Coulomb energy and forces.  All arguments and results in term order.

        The charges are held fixed here, so this is only the explicit part of
        the gradient.  `compute_response_forces` supplies the dQ/dr part, and
        `__call__` adds the two; this method on its own is not the gradient of
        its own energy.
        """
        diag = np.diag_indices(len(Q))
        qiqj = Q[:, None] * Q[None, :]
        qiqj[diag] = 0.0
        r = rij + np.finfo(np.float64).eps
        kernel = erf(2 * rij) / r
        e = self.CCOUL * qiqj * kernel
        e[diag] = 0.0  # zero the diagonal
        e_tot = 0.5 * np.sum(e) * units.eV

        # forces: F_i = CCOUL * sum_j qi*qj * (pos_i-pos_j)/rij^3
        # The energy 0.5-factor cancels because both e[i,j] and e[j,i] contribute to dE/d(pos_i)
        dkernel_dr = (4 / np.sqrt(np.pi)) * np.exp(-4 * rij**2) / r - kernel / r
        nij = vecs / (r[:, :, None])
        f = -nij * self.CCOUL * qiqj[:, :, None] * dkernel_dr[:, :, None]
        f_tot = np.sum(f, 1) * units.eV / units.Angstrom
        return e_tot, f_tot

    def compute_response_forces(self, Q, u, A, rij, vecs, params):
        """The dQ/dr part of the force.  All arguments and results in term order.

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

        `dA/dr` is nonzero only in the two blocks `build_system` builds from
        `rij`.  The softness block needs care: `X_ii = -sum_j X_ij`, so each
        off-diagonal `bsoft_ij` appears in four entries of `X` and all four
        contribute.
        """
        natoms = len(Q)
        diag = np.diag_indices(natoms)
        r = rij + np.finfo(np.float64).eps
        kernel = erf(2 * rij) / r

        # dE/dx, nonzero only on the charge block.  The multiplier rows are
        # geometry-free and the energy does not depend on u.
        dkernel = kernel.copy()
        dkernel[diag] = 0.0
        gradient = np.zeros(A.shape[0])
        gradient[:natoms] = self.CCOUL * (dkernel @ Q)
        lam = np.linalg.solve(A, gradient)
        lam_q, lam_u = lam[:natoms], lam[natoms : 2 * natoms]

        # d(A block)/d r_ij for the two geometry-dependent blocks
        dkernel_dr = (4 / np.sqrt(np.pi)) * np.exp(-4 * rij**2) / r - kernel / r
        dkernel_dr[diag] = 0.0
        amp, decay = params["soft_amp"], params["soft_decay"]
        tau = 0.5 * (decay[:, None] + decay[None, :])
        dbsoft_dr = -(amp[:, None] * amp[None, :]) * np.exp(-rij / tau) / tau
        dbsoft_dr[diag] = 0.0

        # -lam^T (dA/dr) x, accumulated per pair.  Entry (i, j) and entry
        # (j, i) each hold the whole pair term, so the sum is halved to match
        # the convention compute_coulomb uses for the explicit part.
        coulomb = (
            -(lam_q[:, None] * Q[None, :] + lam_q[None, :] * Q[:, None]) * dkernel_dr
        )
        softness = (
            -(
                lam_u[:, None] * u[None, :]
                + lam_u[None, :] * u[:, None]
                - lam_u[:, None] * u[:, None]
                - lam_u[None, :] * u[None, :]
            )
            * dbsoft_dr
        )
        dE_dr = 0.5 * (coulomb + softness)

        nij = vecs / (r[:, :, None])
        return -2.0 * np.sum(dE_dr[:, :, None] * nij, axis=1)

    def __call__(self, pos, pbc, cell, term_dict: dict) -> tuple[float, np.ndarray]:
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

        # Cache on the parameters as well as the geometry: the same positions
        # with a different set of atom terms is a different problem.
        state_hash = hash((pos.tobytes(), indices.tobytes()))
        if self.Q is None or state_hash != self.state_hash:
            self.Q, self.u, self.A = self.solve_charges(rij, params)
            self.state_hash = state_hash

        e_tot, f_tot = self.compute_coulomb(self.Q, rij, vecs)
        f_tot = f_tot + self.compute_response_forces(
            self.Q, self.u, self.A, rij, vecs, params
        )

        # scatter term-ordered forces back to global atom order
        forces = np.zeros_like(pos)
        forces[indices] = f_tot
        return e_tot, forces
