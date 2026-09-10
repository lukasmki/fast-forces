import numpy as np
from typing import Callable

# Default decay rate of the Morse shape term, `c * s**3 * exp(-b * s)`, used for
# any bond whose term file predates `b` being a parameter.
#
# **`b` is now fitted per bond type, and this is only the fallback.**  It was
# fixed at 4 on the argument that the decay "sets *where* the correction acts,
# and that is a property of the chemistry rather than of any one bond".  That is
# measurably false.  Writing `s = a*(r - r0)`, the three populations the shape
# term has to serve sit in three separate bands:
#
#     bonds at their own reference geometry   s = 0.180 - 0.382   (the cost)
#     metathesis transition states            s = 0.352 - 1.049, median 0.681
#     homolysis mid-dissociation geometries   s = 1.081 - 1.525, median 1.358
#
# and `s**3 exp(-b s)` peaks at `s = 3/b`, so one `b` can be aimed at exactly one
# of them.  Four was aimed at the second.  Refitting the whole pipeline from the
# q-force baseline at each fixed `b` -- metathesis channels the coupling fit can
# invert, root-mean-square error of the reactant diabat against the reference
# dissociation curve, and the fastest stretching mode:
#
#     b     c_max   metathesis   dissociation rms   fastest mode      dt
#     2.0    1.31      12/13         0.700 eV         4517 cm^-1   0.492 fs
#     2.5    3.84      13/13         0.783            4402         0.505
#     3.0    7.80      13/13         1.098            4352         0.511
#     4.0   19.33      13/13         1.710            4400         0.505
#     5.0   34.50      13/13         2.660            4402         0.505
#     6.0   52.20      12/13         3.181            4432         0.502
#
# The timestep is flat across the whole range -- the curvature cap absorbs
# whatever `b` does -- so `b` costs nothing and 4 was simply the worst reachable
# value for the dissociation curves.  Hence `fit.dissociation` fits it.
#
# The bound `c_max(b)` above is the monotonicity limit (`fit.dissociation.
# shape_bound`), and it is why low `b` is not free either: at 2.0 it is 1.31 and
# binds on five of eight bonds, so that row is limited by the constraint rather
# than by the fit.
SHAPE_DECAY: float = 4.0


class QForce:
    """Bonded force field, in ASE units (eV, Angstrom) throughout.

    Nothing is converted anywhere in here.  The term dict arrives in the units
    `Parameters.terms` holds, which are the units the fit works in and the units
    `ACKS2`, `ZBL` and `LennardJones` all use; the jsonl's nm and kJ/mol are
    turned into these once at load time by `export/units.py`.  That is what lets
    the per-term `compute_*` methods be called directly -- by `fit`, and by the
    tests -- without a caller having to know which unit system it is in.

    One consequence worth naming: the `-10` and `-20` clamps in
    `compute_bondbond` and `compute_bondangle` are eV, which is what
    `export.openmm.CLIP_BONDBOND` already converts them as.

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
    ) -> tuple[float, np.ndarray, np.ndarray]:
        # compute all distance vectors
        # vecs[1, 0] - vector from atom_0 to atom_1
        vecs = pos[:, None, :] - pos[None, :, :]
        if np.any(pbc):
            F = vecs @ np.linalg.inv(cell)
            vecs = vecs - (pbc * np.floor(F + 0.5)) @ cell

        # compute terms
        e = 0.0
        f = np.zeros_like(pos)
        w = np.zeros((3, 3))
        for term_type, param_dict in term_dict.items():
            fn: Callable | None = getattr(self, f"compute_{term_type}", None)
            if fn is None:
                continue
            de, df, dw = fn(vecs, param_dict["atoms"], **param_dict["kwargs"])
            e += de
            f += df
            w += dw
        return e, f, w

    def _accumulate_forces(self, f, atoms_col, grad):
        """
        Scatter force gradients into the global force array.

        grad shape: (n_terms, 3)
        atoms_col: integer column index into `atoms` selecting which atom receives `grad`.
        The sign convention follows F = -dE/dr, but since vecs[j,i] = pos_i - pos_j,
        dE/d(pos_i) contributions are passed in directly and negated for pos_j.
        """
        np.add.at(f, atoms_col, grad)

    @staticmethod
    def _virial(*pairs):
        """Sum of `v (x) dE/dv` over a term's internal displacement vectors.

        Every term here is a function of displacement vectors only, so under a
        homogeneous strain `v -> (I + e) v` and `dE/de_ab = sum v_a (dE/dv)_b`.
        Each `compute_*` already forms `dE/dv` on its way to scattering the
        forces -- the sign convention being that `dE/dv` is the gradient with
        respect to the vector as `vecs` defines it, so it is scattered as `-dE/dv`
        onto the atom the vector points *to*.  Passing the same arrays here costs
        one outer product per vector and needs no new derivatives.

        The result is an energy, in eV: `v` is in Angstrom and `dE/dv` in
        eV/Angstrom, so the product carries no length.
        """
        w = np.zeros((3, 3))
        for v, dE_dv in pairs:
            w += np.einsum("na,nb->ab", v, dE_dv)
        return w

    def compute_bond(self, vecs, atoms, D, r0, k, c=0.0, b=SHAPE_DECAY):
        if self.bond_form == "morse":
            return self._bond_morse(vecs, atoms, D, r0, k, c, b)
        return self._bond_harmonic(vecs, atoms, D, r0, k)

    def _bond_morse(self, vecs, atoms, D, r0, k, c=0.0, b=SHAPE_DECAY):
        """Morse with a one-sided Hulburt-Hirschfelder shape term.

            s = a*max(dr, 0),  a = sqrt(k / 2D)
            E = D * [ (1 - exp(-a*dr))**2 - 1 + c * s**3 * exp(-b*s) ]

        The `-D` offset puts the dissociated limit at zero, so a topology's
        energy carries the depth of the bonds it contains and breaking a bond
        costs `+D` rather than nothing.

        **Why the third parameter exists.**  Plain Morse (`c = 0`) is exact at
        the minimum and at dissociation and has nothing left over in between:
        `D` is pinned by the atomization energy, `r0` by the geometry, and `k`
        by the vibrational frequency.  It came out 0.55-1.83 eV too deep at the
        stretched geometries where reactions happen, which put every reference
        barrier *above* both diabats -- and `fit_amplitude` has a real root only
        below both, so eighteen of nineteen coupling channels could not be fitted
        at all.  Buying the depth back by inflating `k` works and costs the
        frequency: reaching even 17 of 19 needed H2 at 12402 cm^-1 against an
        experimental 4401, and no force constant whatever reached 18.

        `c` is the freedom that has no other job.  The correction is `O(s**3)`,
        so it vanishes to second order at `dr = 0` and leaves `D`, `r0` and the
        curvature there exactly as they were, and it decays to zero, so the
        dissociation limit is untouched too.  `c = 0` is plain Morse, which is
        what every term file that predates this reads as.

        **That is a statement about `dr = 0`, not about the molecule.**  It was
        read as "the shape term is free of frequency" for as long as `r0` was
        where the bond sat.  It is not: `fit.dissociation.fit_bond_lengths` now
        displaces `r0` inside the reference bond length so the Morse can lean
        against the repulsion, and this term's second derivative at that
        displacement, `D c a**2 (6s - 6b s**2 + b**2 s**3) exp(-b s)`, is the
        largest single contribution to the stiffness of most of HCombustion's
        bonds.  See `fit.dissociation._bonded_curvature`.

        `b` is per bond type and fitted alongside `c`; `SHAPE_DECAY` is only
        the fallback for a term file written before it was a parameter.  See
        `SHAPE_DECAY`'s comment for the measurement that made it one, and
        `fit.dissociation.shape_bound` for the `c <= c_max(b)` constraint
        that couples the two.

        **Why it is one-sided.**  `s**3 * exp(-b*s)` continued to `dr < 0` grows
        without bound against a repulsive wall that only grows like
        `exp(-2a*dr)`, so the compressed branch would turn over and run to minus
        infinity -- an atom pushed hard enough would fall through the nucleus.
        Clamping at `dr = 0` costs nothing in smoothness precisely because the
        term is cubic there: value, slope and curvature are all zero, so the
        join is C2 and the forces never see it.
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
        return e_tot, f, self._virial((v, dv))

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
        return e_tot, f, self._virial((v, dv))

    def compute_reference(self, vecs, atoms, E0):
        """Constant per-molecule reference energy (the EVB alpha shift).

        Geometry-independent, so it contributes no force.  It exists to put
        different bonding topologies on a common absolute energy scale: without
        it the diabatic energies are each measured from their own minimum and
        are not comparable, which makes every EVB eigenvalue meaningless.

        Set by ReactionSet at load time as the residual between the template's
        reference atomization energy and the depth its Morse bonds already
        supply, so it is small and Morse carries the physics.
        """
        # Geometry-independent, so it is strain-independent too: a constant
        # shift moves no atom and stores no stress.
        return np.sum(E0), np.zeros((vecs.shape[0], 3)), np.zeros((3, 3))

    def compute_exclusion(self, vecs, atoms, sigma, eps):
        """Cancels the global Lennard-Jones term between near neighbours.

        **Dormant.**  The repulsion is `forcefield/zbl.py`, which has no
        exclusions, and `ReactionSet.load` no longer derives `exclusion` terms --
        so nothing in the calculator reaches this method.  It is kept because
        `lj.with_exclusions` still builds those terms on demand, for the tests
        that check the Lennard-Jones decomposition still holds, and because a
        dataset shipping explicit `exclusion` terms would still be honoured.

        `forcefield/lj.py` sums 12-6 over *every* pair in the system, including
        pairs that are bonded to each other, because that sum is the same for
        every diabatic state and can therefore be evaluated once outside the EVB.
        What is topology-dependent is which pairs should not have been counted,
        and that is the pairs within `lj.EXCLUSION_DEPTH` bonds of each other --
        a per-molecule quantity, which is what makes it expressible as a term.

        The functional form must match `LennardJones` exactly, combining rule
        included, or an isolated template stops reproducing its own energy.
        `sigma` and `eps` are therefore the already-combined pair values, worked
        out once when the template is loaded rather than twice from different
        code -- and the form itself comes from `lj.pair_potential` for the same
        reason.  It was open-coded here once, and the copy silently stopped
        matching the moment `pair_potential` gained its short-range linear
        continuation: the two halves of the cancellation disagreed by 1609 eV on
        an H2 template.
        """
        from .lj import pair_potential

        v = vecs[atoms[:, 1], atoms[:, 0]]  # (n, 3)  vec from atom0->atom1
        r = np.sqrt(np.sum(v * v, -1))  # (n,)
        # In Angstrom, like everything else in this class -- which is what
        # `pair_potential` requires, since `lj.switch` fixes its length unit,
        # and what makes this half of the cancellation the same function of the
        # same numbers as the other half.
        u, du_dr = pair_potential(r, sigma, eps)
        e_tot = -np.sum(u)

        # e = -u, so de_dr = -du/dr
        de_dr = -du_dr
        dv = (de_dr / r)[:, None] * v  # (n, 3)

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        # F = -dE/d(pos)
        np.add.at(f, atoms[:, 0], dv)
        np.add.at(f, atoms[:, 1], -dv)
        return e_tot, f, self._virial((v, dv))

    def compute_angle(self, vecs, atoms, theta0, k):
        """`E = 0.5*k*(cos(theta) - cos(theta0))**2`.

        The 1/2 is the convention the example files and `export.openmm.ANGLE`
        are written in, so a `k` read from a jsonl means the same well on both
        sides of the export and `export.units` has nothing to rescale.
        """
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
        return e_tot, f, self._virial((va, dE_dva), (vb, dE_dvb))

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
        return e_tot, f, self._virial((v1, dv1), (v2, dv2))

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
        return e_tot, f, self._virial((va, dE_dva), (vb, dE_dvb), (vc, dE_dvc))

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
        return (
            e_tot,
            f,
            self._virial((va, dE_dva), (vb, dE_dvb), (vc, dE_dvc), (vd, dE_dvd)),
        )

    def _dihedral_phi_and_grads(self, va, vb, vc):
        """
        Returns (phi, dphi/d(pos_a0..3), dphi/d(va, vb, vc)) for a batch of
        dihedrals -- position-level gradients for the forces, vector-level ones
        for the virial.

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

        # The vec-level gradients are returned alongside the position-level
        # ones because the virial needs them and they would otherwise have to be
        # rebuilt from the position gradients, which is not possible: the map
        # from three vectors to four positions is not invertible (it drops the
        # centre of mass).  The forces use the position form, the virial the
        # vector form, and both come from the one derivation above.
        return (
            phi,
            dphi_dpos0,
            dphi_dpos1,
            dphi_dpos2,
            dphi_dpos3,
            dphi_dva,
            dphi_dvb,
            dphi_dvc,
        )

    def compute_periodicdihedral(self, vecs, atoms, phi0, n, k):
        va = vecs[atoms[:, 0], atoms[:, 1]]
        vb = vecs[atoms[:, 2], atoms[:, 1]]
        vc = vecs[atoms[:, 3], atoms[:, 2]]

        phi, dp0, dp1, dp2, dp3, dva, dvb, dvc = self._dihedral_phi_and_grads(
            va, vb, vc
        )

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
        return (
            e_tot,
            f,
            self._virial((va, dE_dphi * dva), (vb, dE_dphi * dvb), (vc, dE_dphi * dvc)),
        )

    def compute_dihedralbond(self, vecs, atoms, phi0, n, k, r0):
        # dihedral part (atoms 0..3)
        va = vecs[atoms[:, 0], atoms[:, 1]]
        vb = vecs[atoms[:, 2], atoms[:, 1]]
        vc = vecs[atoms[:, 3], atoms[:, 2]]
        phi, dp0, dp1, dp2, dp3, dva, dvb, dvc = self._dihedral_phi_and_grads(
            va, vb, vc
        )
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
        return (
            e_tot,
            f,
            self._virial(
                (va, dE_dphi * dva),
                (vb, dE_dphi * dvb),
                (vc, dE_dphi * dvc),
                (vd, dE_dvd),
            ),
        )

    def compute_dihedralangle(self, vecs, atoms, phi0, n, k, theta0):
        # dihedral (atoms 0..3)
        va = vecs[atoms[:, 0], atoms[:, 1]]
        vb = vecs[atoms[:, 2], atoms[:, 1]]
        vc = vecs[atoms[:, 3], atoms[:, 2]]
        phi, dp0, dp1, dp2, dp3, dva, dvb, dvc = self._dihedral_phi_and_grads(
            va, vb, vc
        )
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
        return (
            e_tot,
            f,
            self._virial(
                (va, dE_dphi * dva),
                (vb, dE_dphi * dvb),
                (vc, dE_dphi * dvc),
                (vp, dE_dvp),
                (vq, dE_dvq),
            ),
        )

    def compute_dihedralangleangle(self, vecs, atoms, phi0, n, k, theta0_1, theta0_2):
        # dihedral (atoms 0..3)
        va = vecs[atoms[:, 0], atoms[:, 1]]
        vb = vecs[atoms[:, 2], atoms[:, 1]]
        vc = vecs[atoms[:, 3], atoms[:, 2]]
        phi, dp0, dp1, dp2, dp3, dva, dvb, dvc = self._dihedral_phi_and_grads(
            va, vb, vc
        )
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
        # The two angles reuse the dihedral's own atoms but not its vectors:
        # angle 1 shares va/vb, while angle 2 is built from va2 = vecs[a1, a2]
        # and vb2 = vecs[a3, a2], neither of which is va, vb or vc.  Each
        # displacement vector is listed once with its own total gradient.
        w = self._virial(
            (va, dE_dphi * dva),
            (vb, dE_dphi * dvb),
            (vc, dE_dphi * dvc),
            (va1, dE_dcos1 * dcos1_dva1),
            (vb1, dE_dcos1 * dcos1_dvb1),
            (va2, dE_dcos2 * dcos2_dva2),
            (vb2, dE_dcos2 * dcos2_dvb2),
        )
        return e_tot, f, w
