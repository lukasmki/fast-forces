import numpy as np
from typing import Callable


class QForce:
    """Bonded force field.  Works internally in ASE units (eV, Angstrom);
    parameters passed via `term_dict` must already be in these units.

    `bond_form` selects the bond functional form:

      "morse"      D*(1 - exp(-a*dr))**2 - D,  a = sqrt(k/2D)
      "harmonic"   0.5*k*dr**2

    Morse is the default and is required for reactive work.  It is bounded
    above by its dissociation asymptote, so a product state whose newly formed
    bond is still several Angstrom long costs at most D rather than the
    unbounded 0.5*k*dr**2 that the harmonic form charges; and its well depth
    places bound and dissociated topologies on a physically ordered scale
    instead of leaving bond breaking free.  The harmonic form is retained for
    non-reactive use and for comparison.
    """

    def __init__(self, bond_form: str = "morse"):
        if bond_form not in ("morse", "harmonic"):
            raise ValueError(
                f"bond_form must be 'morse' or 'harmonic', got {bond_form!r}"
            )
        self.bond_form: str = bond_form

    def __call__(
        self, pos: np.ndarray, pbc: np.ndarray, cell: np.ndarray, term_dict: dict
    ) -> tuple[float, np.ndarray]:
        # compute all distance vectors
        # vecs[1, 0] - vector from atom_0 to atom_1
        vecs = pos[:, None, :] - pos[None, :, :]
        if np.any(pbc):
            F = vecs @ np.linalg.inv(cell)
            vecs = vecs - (pbc * np.floor(F + 0.5)) @ cell

        # compute terms
        e = 0.0
        f = np.zeros_like(pos)
        for term_type, param_dict in term_dict.items():
            fn: Callable | None = getattr(self, f"compute_{term_type}", None)
            if fn is None:
                continue
            de, df = fn(vecs, param_dict["atoms"], **param_dict["kwargs"])
            e += de
            f += df
        return e, f

    def _accumulate_forces(self, f, atoms_col, grad):
        """
        Scatter force gradients into the global force array.

        grad shape: (n_terms, 3)
        atoms_col: integer column index into `atoms` selecting which atom receives `grad`.
        The sign convention follows F = -dE/dr, but since vecs[j,i] = pos_i - pos_j,
        dE/d(pos_i) contributions are passed in directly and negated for pos_j.
        """
        np.add.at(f, atoms_col, grad)

    def compute_bond(self, vecs, atoms, D, r0, k, c=0.0, b=2.5):
        if self.bond_form == "morse":
            return self._bond_morse(vecs, atoms, D, r0, k, c, b)
        return self._bond_harmonic(vecs, atoms, D, r0, k)

    def _bond_morse(self, vecs, atoms, D, r0, k, c=0.0, b=2.5):
        """Morse with a one-sided Hulburt-Hirschfelder shape term.

        s = a*max(dr, 0),  a = sqrt(k / 2D)
        E = D * [ (1 - exp(-a*dr))**2 - 1 + c * s**3 * exp(-b*s) ]
        """
        v = vecs[atoms[:, 1], atoms[:, 0]]  # (n, 3)  vec from atom0->atom1
        r = np.sqrt(np.sum(v * v, -1))  # (n,)
        dr = r - r0
        al = np.sqrt(k / (2 * D))  # (n,)  1/Angstrom
        exp_term = np.exp(-al * dr)  # (n,)
        e = D * (1 - exp_term) ** 2 - D
        # dE/dr  =  2*D*(1 - exp)*al*exp
        de_dr = 2 * D * (1 - exp_term) * al * exp_term  # (n,)

        # Stretched branch only; `np.maximum` rather than a mask so that the
        # zero-`c` case stays a single vectorised expression.
        s = al * np.maximum(dr, 0.0)  # (n,)
        decay = np.exp(-b * s)
        e = e + D * c * s * s * s * decay
        # d/ds [s**3 exp(-b s)] = (3 s**2 - b s**3) exp(-b s),  ds/dr = al (or 0)
        de_dr = de_dr + D * c * al * s * s * (3.0 - b * s) * decay

        e_tot = np.sum(e)
        # dr/dv = v/r,  v = pos_atom1 - pos_atom0
        dv = (de_dr / r)[:, None] * v  # (n, 3)

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        # F = -dE/d(pos)
        np.add.at(f, atoms[:, 0], dv)
        np.add.at(f, atoms[:, 1], -dv)
        return e_tot, f

    def _bond_harmonic(self, vecs, atoms, D, r0, k):
        """Harmonic potential, E = 0.5*k*dr**2.  `D` is unused."""
        v = vecs[atoms[:, 1], atoms[:, 0]]  # (n, 3)  vec from atom0->atom1
        r = np.sqrt(np.sum(v * v, -1))  # (n,)
        dr = r - r0
        e = 0.5 * k * dr * dr
        e_tot = np.sum(e)

        de_dr = k * dr  # (n,)
        dv = (de_dr / r)[:, None] * v  # (n, 3)  force direction

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        # F = -dE/d(pos)
        np.add.at(f, atoms[:, 0], dv)  # atom0:  v points away from atom1
        np.add.at(f, atoms[:, 1], -dv)  # atom1
        return e_tot, f

    def compute_angle(self, vecs, atoms, theta0, k):
        va = vecs[atoms[:, 0], atoms[:, 1]]  # (n, 3)
        vb = vecs[atoms[:, 2], atoms[:, 1]]  # (n, 3)
        ra = np.sqrt(np.sum(va * va, -1, keepdims=True))  # (n,1)
        rb = np.sqrt(np.sum(vb * vb, -1, keepdims=True))
        na = va / ra  # unit vectors
        nb = vb / rb
        costheta = np.sum(na * nb, -1)  # (n,)
        e = 0.5 * k * np.square(costheta - np.cos(theta0))
        e_tot = np.sum(e)

        # dE/d(cos) = k*(cos - cos0)
        dE_dcos = k * (costheta - np.cos(theta0))  # (n,)

        # d(cos)/d(va) = (nb - cos*na) / |va|
        # d(cos)/d(vb) = (na - cos*nb) / |vb|
        costheta_k = costheta[:, None]
        dcos_dva = (nb - costheta_k * na) / ra  # (n, 3)
        dcos_dvb = (na - costheta_k * nb) / rb

        # chain rule: dE/d(va) = dE/d(cos) * dcos/d(va)
        dE_dva = dE_dcos[:, None] * dcos_dva  # (n, 3)
        dE_dvb = dE_dcos[:, None] * dcos_dvb

        # va = pos[atom0] - pos[atom1]  =>  dE/d(pos_a0)=+dE_dva, dE/d(pos_a1)=-dE_dva
        # vb = pos[atom2] - pos[atom1]  =>  dE/d(pos_a2)=+dE_dvb, dE/d(pos_a1)-= dE_dvb
        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        np.add.at(f, atoms[:, 0], -dE_dva)
        np.add.at(f, atoms[:, 1], dE_dva + dE_dvb)
        np.add.at(f, atoms[:, 2], -dE_dvb)
        return e_tot, f

    def compute_bondbond(self, vecs, atoms, r1_0, r2_0, k):
        v1 = vecs[atoms[:, 0], atoms[:, 1]]  # (n, 3)
        v2 = vecs[atoms[:, 2], atoms[:, 3]]
        r1 = np.sqrt(np.sum(v1 * v1, -1))  # (n,)
        r2 = np.sqrt(np.sum(v2 * v2, -1))
        raw = k * (r1 - r1_0) * (r2 - r2_0)
        e = np.clip(raw, -10, None)
        e_tot = np.sum(e)

        # gradient only where not clipped
        mask = (raw > -10).astype(float)[:, None]
        # dE/d(r1) = k*(r2-r2_0),  dE/d(r2) = k*(r1-r1_0)
        dE_dr1 = (k * (r2 - r2_0))[:, None] * mask  # (n,1)
        dE_dr2 = (k * (r1 - r1_0))[:, None] * mask

        # dr/dv = v/r
        dv1 = dE_dr1 * v1 / r1[:, None]
        dv2 = dE_dr2 * v2 / r2[:, None]

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        np.add.at(f, atoms[:, 0], -dv1)
        np.add.at(f, atoms[:, 1], dv1)
        np.add.at(f, atoms[:, 2], -dv2)
        np.add.at(f, atoms[:, 3], dv2)
        return e_tot, f

    def compute_bondangle(self, vecs, atoms, theta0, r0, k):
        # angle part (atoms 0,1,2)
        va = vecs[atoms[:, 0], atoms[:, 1]]
        vb = vecs[atoms[:, 2], atoms[:, 1]]
        ra = np.sqrt(np.sum(va * va, -1, keepdims=True))
        rb = np.sqrt(np.sum(vb * vb, -1, keepdims=True))
        na = va / ra
        nb = vb / rb
        costheta = np.sum(na * nb, -1)  # (n,)
        dcos = costheta - np.cos(theta0)

        # bond part (atoms 3,4)
        vc = vecs[atoms[:, 3], atoms[:, 4]]
        rc = np.sqrt(np.sum(vc * vc, -1))  # (n,)
        dr = rc - r0

        raw = k * dr * dcos
        e = np.clip(raw, -20, None)
        e_tot = np.sum(e)

        mask = (raw > -20).astype(float)[:, None]

        # dE/d(cos) = k * dr
        dE_dcos = (k * dr)[:, None] * mask
        costheta_k = costheta[:, None]
        dcos_dva = (nb - costheta_k * na) / ra
        dcos_dvb = (na - costheta_k * nb) / rb

        dE_dva = dE_dcos * dcos_dva
        dE_dvb = dE_dcos * dcos_dvb

        # dE/d(rc) = k * dcos
        dE_drc = (k * dcos)[:, None] * mask
        dE_dvc = dE_drc * vc / rc[:, None]  # (n, 3)

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        # angle vectors: va = pos[a0]-pos[a1], vb = pos[a2]-pos[a1]
        np.add.at(f, atoms[:, 0], -dE_dva)
        np.add.at(f, atoms[:, 1], dE_dva + dE_dvb)
        np.add.at(f, atoms[:, 2], -dE_dvb)
        # bond vector: vc = pos[a3]-pos[a4]
        np.add.at(f, atoms[:, 3], -dE_dvc)
        np.add.at(f, atoms[:, 4], dE_dvc)
        return e_tot, f

    def compute_angleangle(self, vecs, atoms, theta1_0, theta2_0, k):
        va = vecs[atoms[:, 0], atoms[:, 1]]
        vb = vecs[atoms[:, 2], atoms[:, 1]]
        vc = vecs[atoms[:, 3], atoms[:, 4]]
        vd = vecs[atoms[:, 5], atoms[:, 4]]
        ra = np.sqrt(np.sum(va * va, -1, keepdims=True))
        rb = np.sqrt(np.sum(vb * vb, -1, keepdims=True))
        rc = np.sqrt(np.sum(vc * vc, -1, keepdims=True))
        rd = np.sqrt(np.sum(vd * vd, -1, keepdims=True))
        na, nb = va / ra, vb / rb
        nc, nd = vc / rc, vd / rd
        ct1 = np.sum(na * nb, -1)  # (n,)
        ct2 = np.sum(nc * nd, -1)
        dct1 = ct1 - np.cos(theta1_0)
        dct2 = ct2 - np.cos(theta2_0)
        e = k * dct1 * dct2
        e_tot = np.sum(e)

        # dE/d(ct1) = k * dct2,  dE/d(ct2) = k * dct1
        dE_dct1 = (k * dct2)[:, None]
        dE_dct2 = (k * dct1)[:, None]

        ct1_k = ct1[:, None]
        ct2_k = ct2[:, None]
        dct1_dva = (nb - ct1_k * na) / ra
        dct1_dvb = (na - ct1_k * nb) / rb
        dct2_dvc = (nd - ct2_k * nc) / rc
        dct2_dvd = (nc - ct2_k * nd) / rd

        dE_dva = dE_dct1 * dct1_dva
        dE_dvb = dE_dct1 * dct1_dvb
        dE_dvc = dE_dct2 * dct2_dvc
        dE_dvd = dE_dct2 * dct2_dvd

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        # angle 1: va=pos[0]-pos[1], vb=pos[2]-pos[1]
        np.add.at(f, atoms[:, 0], -dE_dva)
        np.add.at(f, atoms[:, 1], dE_dva + dE_dvb)
        np.add.at(f, atoms[:, 2], -dE_dvb)
        # angle 2: vc=pos[3]-pos[4], vd=pos[5]-pos[4]
        np.add.at(f, atoms[:, 3], -dE_dvc)
        np.add.at(f, atoms[:, 4], dE_dvc + dE_dvd)
        np.add.at(f, atoms[:, 5], -dE_dvd)
        return e_tot, f

    def _dihedral_phi_and_grads(self, va, vb, vc):
        """
        Returns (phi, dphi/d(pos_a0..3)) for a batch of dihedrals.

        Vector convention:
            va = vecs[a0, a1] = pos_a0 - pos_a1
            vb = vecs[a2, a1] = pos_a2 - pos_a1  (central bond)
            vc = vecs[a3, a2] = pos_a3 - pos_a2

        phi = atan2(S, C),  S = (axis x u)·v,  C = u·v
        axis = vb/|vb|,  u = P·va,  v = P·vc  (P = I - axis⊗axis, perp projection)

        Analytic vec-level gradients (derived via chain rule through u,v):
            dphi/dva  = (C*(v x axis) - S*v) / (S²+C²)
            dphi/dvc  = (C*(axis x u) - S*u) / (S²+C²)
            dphi/dvb  = -(va·axis/|vb|)*dphi/dva - (vc·axis/|vb|)*dphi/dvc

        Position-level chain rule (va=-pos_a1, vb=pos_a2-pos_a1, vc=-pos_a2):
            dphi/d(pos_a0) =  dphi/dva
            dphi/d(pos_a1) = -dphi/dva - dphi/dvb
            dphi/d(pos_a2) =  dphi/dvb - dphi/dvc
            dphi/d(pos_a3) =  dphi/dvc
        """
        rb = np.sqrt(np.sum(vb * vb, -1, keepdims=True))
        axis = vb / rb
        u = va - np.sum(va * axis, -1, keepdims=True) * axis  # va perp to axis
        v = vc - np.sum(vc * axis, -1, keepdims=True) * axis  # vc perp to axis
        axu = np.cross(axis, u, -1)

        S = np.sum(axu * v, -1)  # sin-like
        C = np.sum(u * v, -1)  # cos-like
        r2 = np.maximum(S**2 + C**2, 1e-30)
        phi = np.arctan2(S, C)

        # vec-level gradients via chain rule through u and v
        # dphi/dva: dS/dva = v×axis,  dC/dva = v
        dphi_dva = (C[:, None] * np.cross(v, axis, -1) - S[:, None] * v) / r2[:, None]
        # dphi/dvc: dS/dvc = axu,  dC/dvc = u
        dphi_dvc = (C[:, None] * axu - S[:, None] * u) / r2[:, None]
        # dphi/dvb: chain rule through axis = vb/|vb| (verified numerically)
        va_dot = np.sum(va * axis, -1, keepdims=True)
        vc_dot = np.sum(vc * axis, -1, keepdims=True)
        dphi_dvb = -(va_dot / rb) * dphi_dva - (vc_dot / rb) * dphi_dvc

        # position-level gradients via chain rule:
        #   va = pos_a0 - pos_a1,  vb = pos_a2 - pos_a1,  vc = pos_a3 - pos_a2
        dphi_dpos0 = dphi_dva
        dphi_dpos1 = -dphi_dva - dphi_dvb
        dphi_dpos2 = dphi_dvb - dphi_dvc
        dphi_dpos3 = dphi_dvc

        return phi, dphi_dpos0, dphi_dpos1, dphi_dpos2, dphi_dpos3

    def compute_periodicdihedral(self, vecs, atoms, phi0, n, k):
        va = vecs[atoms[:, 0], atoms[:, 1]]
        vb = vecs[atoms[:, 2], atoms[:, 1]]
        vc = vecs[atoms[:, 3], atoms[:, 2]]

        phi, dp0, dp1, dp2, dp3 = self._dihedral_phi_and_grads(va, vb, vc)

        e = k * (1 + np.cos(n * phi - phi0))
        e_tot = np.sum(e)

        # dE/dphi = -k * n * sin(n*phi - phi0)
        dE_dphi = (-k * n * np.sin(n * phi - phi0))[:, None]  # (n,1)

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        # F = -dE/d(pos) = -dE/dphi * dphi/d(pos)
        np.add.at(f, atoms[:, 0], -dE_dphi * dp0)
        np.add.at(f, atoms[:, 1], -dE_dphi * dp1)
        np.add.at(f, atoms[:, 2], -dE_dphi * dp2)
        np.add.at(f, atoms[:, 3], -dE_dphi * dp3)
        return e_tot, f

    def compute_dihedralbond(self, vecs, atoms, phi0, n, k, r0):
        # dihedral part (atoms 0..3)
        va = vecs[atoms[:, 0], atoms[:, 1]]
        vb = vecs[atoms[:, 2], atoms[:, 1]]
        vc = vecs[atoms[:, 3], atoms[:, 2]]
        phi, dp0, dp1, dp2, dp3 = self._dihedral_phi_and_grads(va, vb, vc)
        cos_term = 1 + np.cos(n * phi - phi0)  # (n,)

        # bond part (atoms 4,5)
        vd = vecs[atoms[:, 4], atoms[:, 5]]
        rd = np.sqrt(np.sum(vd * vd, -1))  # (n,)
        dr = rd - r0

        e = k * dr * cos_term
        e_tot = np.sum(e)

        # dE/dphi  = k * dr * (-n * sin(n*phi-phi0))
        dE_dphi = (k * dr * (-n * np.sin(n * phi - phi0)))[:, None]
        # dE/d(rd) = k * cos_term
        dE_drd = (k * cos_term)[:, None]
        dE_dvd = dE_drd * vd / rd[:, None]

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        np.add.at(f, atoms[:, 0], -dE_dphi * dp0)
        np.add.at(f, atoms[:, 1], -dE_dphi * dp1)
        np.add.at(f, atoms[:, 2], -dE_dphi * dp2)
        np.add.at(f, atoms[:, 3], -dE_dphi * dp3)
        np.add.at(f, atoms[:, 4], -dE_dvd)  # vd = pos[a4]-pos[a5]
        np.add.at(f, atoms[:, 5], dE_dvd)
        return e_tot, f

    def compute_dihedralangle(self, vecs, atoms, phi0, n, k, theta0):
        # dihedral (atoms 0..3)
        va = vecs[atoms[:, 0], atoms[:, 1]]
        vb = vecs[atoms[:, 2], atoms[:, 1]]
        vc = vecs[atoms[:, 3], atoms[:, 2]]
        phi, dp0, dp1, dp2, dp3 = self._dihedral_phi_and_grads(va, vb, vc)
        cos_term = 1 + np.cos(n * phi - phi0)

        # angle (atoms 4,5,6)
        vp = vecs[atoms[:, 4], atoms[:, 5]]
        vq = vecs[atoms[:, 6], atoms[:, 5]]
        rp = np.sqrt(np.sum(vp * vp, -1, keepdims=True))
        rq = np.sqrt(np.sum(vq * vq, -1, keepdims=True))
        np_ = vp / rp
        nq = vq / rq
        costheta = np.sum(np_ * nq, -1)  # (n,)
        dcos = costheta - np.cos(theta0)

        e = k * dcos * cos_term
        e_tot = np.sum(e)

        dE_dphi = (k * dcos * (-n * np.sin(n * phi - phi0)))[:, None]
        dE_dcos = (k * cos_term)[:, None]

        costheta_k = costheta[:, None]
        dcos_dvp = (nq - costheta_k * np_) / rp
        dcos_dvq = (np_ - costheta_k * nq) / rq

        dE_dvp = dE_dcos * dcos_dvp
        dE_dvq = dE_dcos * dcos_dvq

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        np.add.at(f, atoms[:, 0], -dE_dphi * dp0)
        np.add.at(f, atoms[:, 1], -dE_dphi * dp1)
        np.add.at(f, atoms[:, 2], -dE_dphi * dp2)
        np.add.at(f, atoms[:, 3], -dE_dphi * dp3)
        np.add.at(f, atoms[:, 4], -dE_dvp)
        np.add.at(f, atoms[:, 5], dE_dvp + dE_dvq)
        np.add.at(f, atoms[:, 6], -dE_dvq)
        return e_tot, f

    def compute_dihedralangleangle(self, vecs, atoms, phi0, n, k, theta0_1, theta0_2):
        # dihedral (atoms 0..3)
        va = vecs[atoms[:, 0], atoms[:, 1]]
        vb = vecs[atoms[:, 2], atoms[:, 1]]
        vc = vecs[atoms[:, 3], atoms[:, 2]]
        phi, dp0, dp1, dp2, dp3 = self._dihedral_phi_and_grads(va, vb, vc)
        cos_term = 1 + np.cos(n * phi - phi0)

        # angle 1 (atoms 0,1,2) - same vectors as dihedral start
        va1 = vecs[atoms[:, 0], atoms[:, 1]]
        vb1 = vecs[atoms[:, 2], atoms[:, 1]]
        ra1 = np.sqrt(np.sum(va1 * va1, -1, keepdims=True))
        rb1 = np.sqrt(np.sum(vb1 * vb1, -1, keepdims=True))
        na1, nb1 = va1 / ra1, vb1 / rb1
        costheta1 = np.sum(na1 * nb1, -1)
        dcos1 = costheta1 - np.cos(theta0_1)

        # angle 2 (atoms 1,2,3)
        va2 = vecs[atoms[:, 1], atoms[:, 2]]
        vb2 = vecs[atoms[:, 3], atoms[:, 2]]
        ra2 = np.sqrt(np.sum(va2 * va2, -1, keepdims=True))
        rb2 = np.sqrt(np.sum(vb2 * vb2, -1, keepdims=True))
        na2, nb2 = va2 / ra2, vb2 / rb2
        costheta2 = np.sum(na2 * nb2, -1)
        dcos2 = costheta2 - np.cos(theta0_2)

        e = k * dcos1 * dcos2 * cos_term
        e_tot = np.sum(e)

        dE_dphi = (k * dcos1 * dcos2 * (-n * np.sin(n * phi - phi0)))[:, None]
        dE_dcos1 = (k * dcos2 * cos_term)[:, None]
        dE_dcos2 = (k * dcos1 * cos_term)[:, None]

        ct1k = costheta1[:, None]
        dcos1_dva1 = (nb1 - ct1k * na1) / ra1
        dcos1_dvb1 = (na1 - ct1k * nb1) / rb1

        ct2k = costheta2[:, None]
        dcos2_dva2 = (nb2 - ct2k * na2) / ra2
        dcos2_dvb2 = (na2 - ct2k * nb2) / rb2

        # angle1: va1=pos[0]-pos[1], vb1=pos[2]-pos[1]
        # angle vector va1=pos[0]-pos[1]: F_pos0 += -dE/dva1, F_pos1 += +dE/dva1 ... chain rule:
        # E_ang1 contrib: dE/d(pos_0) = dE_dcos1 * dcos1_dva1 (since va1=pos0-pos1, d/dpos0=+I)
        # F = -dE/d(pos), so F_pos0 -= dE_dcos1 * dcos1_dva1
        dE_ang1_pos0 = dE_dcos1 * dcos1_dva1
        dE_ang1_pos1 = -dE_dcos1 * (dcos1_dva1 + dcos1_dvb1)
        dE_ang1_pos2 = dE_dcos1 * dcos1_dvb1

        # angle2: va2=pos[1]-pos[2], vb2=pos[3]-pos[2]
        dE_ang2_pos1 = dE_dcos2 * dcos2_dva2
        dE_ang2_pos2 = -dE_dcos2 * (dcos2_dva2 + dcos2_dvb2)
        dE_ang2_pos3 = dE_dcos2 * dcos2_dvb2

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        # dihedral forces
        np.add.at(f, atoms[:, 0], -dE_dphi * dp0)
        np.add.at(f, atoms[:, 1], -dE_dphi * dp1)
        np.add.at(f, atoms[:, 2], -dE_dphi * dp2)
        np.add.at(f, atoms[:, 3], -dE_dphi * dp3)
        # angle1 forces
        np.add.at(f, atoms[:, 0], -dE_ang1_pos0)
        np.add.at(f, atoms[:, 1], -dE_ang1_pos1)
        np.add.at(f, atoms[:, 2], -dE_ang1_pos2)
        # angle2 forces
        np.add.at(f, atoms[:, 1], -dE_ang2_pos1)
        np.add.at(f, atoms[:, 2], -dE_ang2_pos2)
        np.add.at(f, atoms[:, 3], -dE_ang2_pos3)
        return e_tot, f
