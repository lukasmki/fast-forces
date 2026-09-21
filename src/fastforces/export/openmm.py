"""Export to an OpenMM `System`, and to the XML `XmlSerializer` writes.

The System is built through the OpenMM API rather than by assembling XML by
hand, so the serialized file is valid by construction.

The energy expressions here are transcriptions of the `forcefield` module's
`compute_*` methods, not of the example XML -- where the two disagree, the
calculator wins, because an exported file that does not reproduce the
calculator it was fit with is worse than useless.  Where that matters:

  * `dihedralangle` and `dihedralangleangle` use `cos(angle) - cos(theta0)`,
    matching the implementation; the example XML uses the raw angle difference
    there, while using the cosine form for `bondangle` and `angleangle`.
  * The `bondbond` and `bondangle` clamps are `-10` and `-20` *eV*, converted
    here; the example XML carries the bare numbers, which OpenMM would read as
    kJ/mol.
  * The Morse bond keeps its `-D` offset and its Hulburt-Hirschfelder `c` term.
  * `angle` is `0.5*k*(cos-cos0)^2`, the convention the example files are
    written in, so an angle `k` means the same well on both sides.
  * Both nonbonded pair terms carry a Fermi switch, and both are summed over
    *every* pair with no exclusions -- `zbl.taper` switching ZBL off outside
    1.5 A, `lj.switch` switching the 12-6 on outside 2.2 A, and
    `lj.CORE_FRACTION` replacing the `r**-12` divergence with a tangent.  A
    transcription that dropped any of those would be a different force field at
    exactly the separations bonded pairs sit at.

Electrostatics is the one place the export cannot be faithful, and only for one
of the two terms.  ACKS2 solves for the charges at every geometry; OpenMM has no
charge-equilibration force, so an `atom` block is solved once at the geometry
passed in and baked in as fixed values.  A simulation run from that exported
file has fixed charges, and will drift from the FastForces calculator as the
geometry moves away from the reference.

A `coulomb` block has nothing to bake: its charges *are* fixed, so the exported
force is the term itself and needs no geometry.  Both export to the same
`erf(beta r)/r` expression -- which is why that is the kernel
`forcefield/coulomb.py` uses.
"""

import numpy as np
from ase.data import atomic_masses

from ..forcefield.acks2 import ACKS2
from ..forcefield.ewald import CCOUL
from ..forcefield.lj import CORE_FRACTION, SWITCH_RADIUS, SWITCH_WIDTH
from ..forcefield.qforce import SHAPE_DECAY
from ..forcefield.zbl import CCOUL as ZBL_CCOUL
from ..forcefield.zbl import PHI_B, PHI_C, SCREENING_LENGTH, TAPER_RADIUS, TAPER_WIDTH
from . import units as u

# Clamps from `QForce.compute_bondbond` / `compute_bondangle`, in eV.
CLIP_BONDBOND = -10.0
CLIP_BONDANGLE = -20.0

# The Hulburt-Hirschfelder decay `QForce.compute_bond` defaults to.  The export
# formats have no slot for it, so it has to stay at the default on both sides;
# it is imported rather than restated so the two cannot drift.
HH_DECAY = SHAPE_DECAY

# Both Fermi switches overflow `exp` if handed a raw exponent -- these forces are
# evaluated with no cutoff, so `r` reaches the box diagonal.  The evaluators clip
# at 500 in numpy; Lepton has no `clip`, so `max(min(...))` does it here.  300 is
# well past where either switch resolves anything (`exp(-300)` is 5e-131) and
# stays inside double range, which 500 does not.
SWITCH_CLAMP = 300.0

# Screening width of the smeared Coulomb kernel, `erf(2 r)/r` with r in
# Angstrom.  `forcefield.ewald.GAMMA`, under the name the expression uses.
ACKS2_BETA = 2.0

MORSE = (
    "D*((1-exp(-a*(r-r0)))^2 - 1 + c*s*s*s*exp(-hh_decay*s));"
    " s=a*max(r-r0,0); a=sqrt(k/(2*D))"
)
# `QForce.compute_angle`, transcribed -- 1/2 included, which is also the
# convention the example XML and jsonl are written in.
ANGLE = "0.5*k*(cos(theta)-cos(theta0))^2"
BONDBOND = "max(k*(distance(p1,p2)-r1_0)*(distance(p3,p4)-r2_0), clip_bb)"
BONDANGLE = "max(k*(cos(angle(p1,p2,p3))-cos(theta0))*(distance(p4,p5)-r0), clip_ba)"
ANGLEANGLE = (
    "k*(cos(angle(p1,p2,p3))-cos(theta1_0))*(cos(angle(p4,p5,p6))-cos(theta2_0))"
)
DIHEDRALANGLE = (
    "k*(1+cos(n*dihedral(p1,p2,p3,p4)-phi0))*(cos(angle(p5,p6,p7))-cos(theta0))"
)
DIHEDRALBOND = "k*(1+cos(n*dihedral(p1,p2,p3,p4)-phi0))*(distance(p5,p6)-r0)"
DIHEDRALANGLEANGLE = (
    "k*(1+cos(n*dihedral(p1,p2,p3,p4)-phi0))"
    "*(cos(angle(p1,p2,p3))-cos(theta0_1))*(cos(angle(p2,p3,p4))-cos(theta0_2))"
)
PERIODICDIHEDRAL = "k*(1+cos(n*theta-phi0))"
# `lj.pair_potential`, transcribed: a 12-6 that is switched *on* outside
# `SWITCH_RADIUS`, and that continues along its own tangent inside
# `CORE_FRACTION * sigma` instead of diverging as `r**-12`.  Both pieces matter
# here and not only in the calculator -- the term is applied to bonded pairs, so
# the exported system evaluates it at bond lengths too.
LENNARDJONES = (
    "g*(u+du*min(r-rc,0));"
    " g=1/(1+exp(zs)); zs=max(-sw_clamp,min(sw_clamp,-(r-sw_r)/sw_w));"
    " u=4*B*(A12/re12-A6/re6); du=-(24*B/re)*(2*A12/re12-A6/re6);"
    " re12=re6*re6; re6=re^6; re=max(r,rc); rc=core_frac*A;"
    " A12=A6*A6; A6=A^6; B=sqrt(B1*B2); A=sqrt(A1*A2)"
)
# `zbl.pair_potential`, transcribed, including `zbl.taper`: the screened-nuclear
# form is switched off outside `TAPER_RADIUS` so that it does not reach into the
# hydrogen bond.  Omitting the taper here would leave the exported system with a
# repulsion the calculator no longer has.
ZBL = (
    "f*zk*(pc0*exp(-pb0*x)+pc1*exp(-pb1*x)+pc2*exp(-pb2*x)+pc3*exp(-pb3*x))/r;"
    " f=1/(1+exp(zt)); zt=max(-sw_clamp,min(sw_clamp,(r-taper_r)/taper_w));"
    " x=r/a; a=screen/(z1^0.23+z2^0.23); zk=zbl_ccoul*z1*z2"
)
# `acks2_ccoul` and `beta` keep their ACKS2-flavoured names although both
# electrostatic terms export to this expression now: the names are pinned by
# `tests/acetonitrile.xml` and by every file the exporter has already written,
# and renaming them would invalidate those for nothing but a label.
COULOMB = "acks2_ccoul*q1*q2*erf(beta*r)/r"

# Which compound-bond expression goes with how many particles.
COMPOUND = {
    "bondbond": (BONDBOND, 4, ("r1_0", "r2_0", "k")),
    "bondangle": (BONDANGLE, 5, ("theta0", "r0", "k")),
    "angleangle": (ANGLEANGLE, 6, ("theta1_0", "theta2_0", "k")),
    "dihedralangle": (DIHEDRALANGLE, 7, ("k", "theta0", "n", "phi0")),
    "dihedralbond": (DIHEDRALBOND, 6, ("k", "r0", "n", "phi0")),
    "dihedralangleangle": (
        DIHEDRALANGLEANGLE,
        4,
        ("k", "theta0_1", "theta0_2", "n", "phi0"),
    ),
}


def _converted(params, term: str, name: str) -> np.ndarray:
    return u.to_openmm(term, name, params.terms[term]["kwargs"][name])


def build_system(params, positions: np.ndarray | None = None, box: float = 20.0):
    """An `openmm.System` reproducing `params`.

    `positions` (Angstrom) fixes the geometry the ACKS2 charges are solved at;
    without it the electrostatic force is omitted entirely rather than written
    with wrong charges.  A field carrying fixed `coulomb` charges does not need
    it.
    """
    import openmm
    from openmm import unit as omm_unit

    system = openmm.System()
    for z in params.numbers:
        system.addParticle(float(atomic_masses[int(z)]) * omm_unit.amu)
    edge = box * u.LENGTH
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(edge, 0, 0), openmm.Vec3(0, edge, 0), openmm.Vec3(0, 0, edge)
    )

    for force in _nonbonded_forces(params, positions, edge):
        system.addForce(force)
    for force in _bonded_forces(params):
        system.addForce(force)

    e0 = params.e0
    if e0:
        # A constant has nowhere else to live in an OpenMM System.  The
        # expression has no coordinate dependence, so it contributes energy and
        # no force.
        offset = openmm.CustomExternalForce("e0_per_atom")
        offset.addGlobalParameter(
            "e0_per_atom", u.to_openmm("reference", "E0", e0) / len(params.numbers)
        )
        for i in range(len(params.numbers)):
            offset.addParticle(i, [])
        offset.setName("Reference")
        system.addForce(offset)
    return system


def _nonbonded_forces(params, positions, edge):
    import openmm

    forces = []

    if "lennardjones" in params.terms:
        # No exclusions, deliberately: `LennardJones` dropped them, so adding
        # them here would make the exported system disagree with the calculator
        # it was fit with on every 1-2, 1-3 and 1-4 pair.  `params.exclusions`
        # is still carried, and is still the right mask -- nothing consumes it.
        lj = openmm.CustomNonbondedForce(LENNARDJONES)
        lj.setName("LennardJones")
        lj.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
        lj.addPerParticleParameter("A")
        lj.addPerParticleParameter("B")
        # In Angstrom in `lj`, which works in ASE units like the rest of the
        # `forcefield` package, so both take `u.LENGTH` here.  `core_frac` is a
        # fraction of sigma and converts by 1.
        lj.addGlobalParameter("sw_r", SWITCH_RADIUS * u.LENGTH)
        lj.addGlobalParameter("sw_w", SWITCH_WIDTH * u.LENGTH)
        lj.addGlobalParameter("core_frac", CORE_FRACTION)
        lj.addGlobalParameter("sw_clamp", SWITCH_CLAMP)
        sigma = _converted(params, "lennardjones", "sigma")
        eps = _converted(params, "lennardjones", "eps")
        order = np.asarray(params.terms["lennardjones"]["atoms"])[:, 0]
        by_index = {int(a): i for i, a in enumerate(order)}
        for atom in range(len(params.numbers)):
            i = by_index[atom]
            lj.addParticle([float(sigma[i]), float(eps[i])])
        forces.append(lj)

    # ZBL takes no exclusions -- see the `ZBL` docstring; it is a function of the
    # geometry and the elements alone.
    zbl = openmm.CustomNonbondedForce(ZBL)
    zbl.setName("ZBL")
    zbl.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
    zbl.addPerParticleParameter("z")
    zbl.addGlobalParameter("screen", SCREENING_LENGTH * u.LENGTH)
    zbl.addGlobalParameter("zbl_ccoul", ZBL_CCOUL * u.ENERGY * u.LENGTH)
    # In Angstrom in `zbl`, which works in ASE units throughout.
    zbl.addGlobalParameter("taper_r", TAPER_RADIUS * u.LENGTH)
    zbl.addGlobalParameter("taper_w", TAPER_WIDTH * u.LENGTH)
    zbl.addGlobalParameter("sw_clamp", SWITCH_CLAMP)
    for k, (c, b) in enumerate(zip(PHI_C, PHI_B, strict=True)):
        zbl.addGlobalParameter(f"pc{k}", c)
        zbl.addGlobalParameter(f"pb{k}", b)
    for z in params.numbers:
        zbl.addParticle([float(z)])
    forces.append(zbl)

    charges = exported_charges(params, positions)
    if charges is not None:
        coulomb = openmm.CustomNonbondedForce(COULOMB)
        coulomb.setName("Coulomb")
        coulomb.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
        coulomb.addPerParticleParameter("q")
        coulomb.addGlobalParameter("acks2_ccoul", CCOUL * u.ENERGY * u.LENGTH)
        coulomb.addGlobalParameter("beta", ACKS2_BETA / u.LENGTH)
        for q in charges:
            coulomb.addParticle([float(q)])
        forces.append(coulomb)
    return forces


def exported_charges(params, positions: np.ndarray | None) -> np.ndarray | None:
    """The fixed charges the exported Coulomb force carries, in global order.

    `None` when there are none to write, which is either a field with no
    electrostatics at all or an ACKS2 field exported without a geometry to
    solve at -- the electrostatic force is then omitted entirely rather than
    written with wrong charges.

    A `coulomb` field ignores `positions`: its charges do not depend on the
    geometry, which is the whole difference between the two terms.
    """
    term = params.electrostatics()
    if term == "coulomb":
        block = params.terms["coulomb"]
        charges = np.zeros(len(params.numbers))
        charges[np.asarray(block["atoms"])[:, 0]] = block["kwargs"]["q"]
        return charges
    if term == "atom" and positions is not None:
        return acks2_charges(params, np.asarray(positions, dtype=float))
    return None


def acks2_charges(params, positions: np.ndarray) -> np.ndarray:
    """ACKS2 charges at `positions` (Angstrom), in global atom order.

    Solved with no kernel, so under open boundaries -- see
    `ACKS2.get_kernel`.  The exported system is periodic in a 20 A box by
    default, so this is already an approximation on top of freezing the
    charges; both are the same trade `build_system` documents.
    """
    block = params.terms["atom"]
    indices = np.asarray(block["atoms"])[:, 0]
    vecs = positions[:, None, :] - positions[None, :, :]
    sub = np.ix_(indices, indices)
    rij = np.sqrt(np.sum(vecs[sub] * vecs[sub], -1))
    solved = ACKS2().compute_charges(rij, block["kwargs"])
    charges = np.zeros(len(positions))
    charges[indices] = solved
    return charges


def _bonded_forces(params):
    import openmm

    forces = []

    if "bond" in params.terms:
        bond = openmm.CustomBondForce(MORSE)
        bond.setName("Bond")
        for name in ("r0", "k", "D", "c"):
            bond.addPerBondParameter(name)
        bond.addGlobalParameter("hh_decay", HH_DECAY)
        atoms = np.asarray(params.terms["bond"]["atoms"])
        values = [_converted(params, "bond", n) for n in ("r0", "k", "D", "c")]
        for i, (a, b) in enumerate(atoms):
            bond.addBond(int(a), int(b), [float(v[i]) for v in values])
        forces.append(bond)

    if "angle" in params.terms:
        angle = openmm.CustomAngleForce(ANGLE)
        angle.setName("Angle")
        for name in ("theta0", "k"):
            angle.addPerAngleParameter(name)
        atoms = np.asarray(params.terms["angle"]["atoms"])
        values = [_converted(params, "angle", n) for n in ("theta0", "k")]
        for i, (a, b, c) in enumerate(atoms):
            angle.addAngle(int(a), int(b), int(c), [float(v[i]) for v in values])
        forces.append(angle)

    for term, (expression, n_particles, names) in COMPOUND.items():
        if term not in params.terms:
            continue
        force = openmm.CustomCompoundBondForce(n_particles, expression)
        force.setName(term)
        for name in names:
            force.addPerBondParameter(name)
        if term == "bondbond":
            force.addGlobalParameter("clip_bb", CLIP_BONDBOND * u.ENERGY)
        if term == "bondangle":
            force.addGlobalParameter("clip_ba", CLIP_BONDANGLE * u.ENERGY)
        atoms = np.asarray(params.terms[term]["atoms"])
        values = [_converted(params, term, n) for n in names]
        for i, row in enumerate(atoms):
            force.addBond([int(a) for a in row], [float(v[i]) for v in values])
        forces.append(force)

    if "periodicdihedral" in params.terms:
        torsion = openmm.CustomTorsionForce(PERIODICDIHEDRAL)
        torsion.setName("PeriodicDihedral")
        for name in ("k", "n", "phi0"):
            torsion.addPerTorsionParameter(name)
        atoms = np.asarray(params.terms["periodicdihedral"]["atoms"])
        values = [_converted(params, "periodicdihedral", n) for n in ("k", "n", "phi0")]
        for i, row in enumerate(atoms):
            torsion.addTorsion(*[int(a) for a in row], [float(v[i]) for v in values])
        forces.append(torsion)
    return forces


def write_xml(
    params, path: str, positions: np.ndarray | None = None, box: float = 20.0
):
    import openmm

    system = build_system(params, positions=positions, box=box)
    with open(path, "w") as handle:
        handle.write(openmm.XmlSerializer.serialize(system))


def system_energy(system, positions: np.ndarray) -> float:
    """Potential energy of `system` at `positions` (Angstrom), returned in eV."""
    import openmm
    from openmm import unit as omm_unit

    integrator = openmm.VerletIntegrator(1.0 * omm_unit.femtosecond)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatform("Reference")
    )
    context.setPositions((np.asarray(positions) * u.LENGTH) * omm_unit.nanometer)
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(omm_unit.kilojoule_per_mole)
    return energy / u.ENERGY
