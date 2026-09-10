"""Export to an OpenMM `System`, and to the XML `XmlSerializer` writes.

The System is built through the OpenMM API rather than by assembling XML by
hand, so the serialized file is valid by construction.

The energy expressions here are transcriptions of the `forcefield` module's
`compute_*` methods, not of the example XML -- where the two disagree, the
calculator wins, because an exported file that does not reproduce the
calculator it was fit with is worse than useless.  Three places where that
matters:

  * `dihedralangle` and `dihedralangleangle` use `cos(angle) - cos(theta0)`,
    matching the implementation; the example XML uses the raw angle difference
    there, while using the cosine form for `bondangle` and `angleangle`.
  * The `bondbond` and `bondangle` clamps are `-10` and `-20` *eV*, converted
    here; the example XML carries the bare numbers, which OpenMM would read as
    kJ/mol.
  * The Morse bond keeps its `-D` offset and its Hulburt-Hirschfelder `c` term.

Electrostatics is the one place the export cannot be faithful.  ACKS2 solves for
the charges at every geometry; OpenMM has no charge-equilibration force, so the
charges are solved once at the geometry passed in and baked in as fixed values.
A simulation run from the exported file therefore has fixed charges, and will
drift from the FastForces calculator as the geometry moves away from that
reference.
"""

import numpy as np
from ase.data import atomic_masses

from ..forcefield.acks2 import ACKS2
from ..forcefield.zbl import CCOUL as ZBL_CCOUL
from ..forcefield.zbl import PHI_B, PHI_C, SCREENING_LENGTH
from . import units as u

# Clamps from `QForce.compute_bondbond` / `compute_bondangle`, in eV.
CLIP_BONDBOND = -10.0
CLIP_BONDANGLE = -20.0

# The Hulburt-Hirschfelder decay `QForce.compute_bond` defaults to.  The export
# formats have no slot for it, so it has to stay at the default on both sides.
HH_DECAY = 2.5

# Screening width of the ACKS2 Coulomb kernel, `erf(2 r)/r` with r in Angstrom.
ACKS2_BETA = 2.0

MORSE = (
    "D*((1-exp(-a*(r-r0)))^2 - 1 + c*s*s*s*exp(-hh_decay*s));"
    " s=a*max(r-r0,0); a=sqrt(k/(2*D))"
)
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
LENNARDJONES = (
    "4*B*(A12/r12-A6/r6); r12=r6*r6; r6=r^6; A12=A6*A6; A6=A^6;"
    " B=sqrt(B1*B2); A=sqrt(A1*A2)"
)
ZBL = (
    "zk*(pc0*exp(-pb0*x)+pc1*exp(-pb1*x)+pc2*exp(-pb2*x)+pc3*exp(-pb3*x))/r;"
    " x=r/a; a=screen/(z1^0.23+z2^0.23); zk=zbl_ccoul*z1*z2"
)
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

    `positions` (Angstrom) fixes the geometry the ACKS2 charges are solved at.
    Without it the electrostatic force is omitted entirely rather than written
    with wrong charges.
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
    exclusions = params.exclusions
    pairs = (
        [
            (i, j)
            for i in range(len(params.numbers))
            for j in range(i + 1, len(params.numbers))
            if exclusions[i, j]
        ]
        if exclusions is not None
        else []
    )

    if "lennardjones" in params.terms:
        lj = openmm.CustomNonbondedForce(LENNARDJONES)
        lj.setName("LennardJones")
        lj.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
        lj.addPerParticleParameter("A")
        lj.addPerParticleParameter("B")
        sigma = _converted(params, "lennardjones", "sigma")
        eps = _converted(params, "lennardjones", "eps")
        order = np.asarray(params.terms["lennardjones"]["atoms"])[:, 0]
        by_index = {int(a): i for i, a in enumerate(order)}
        for atom in range(len(params.numbers)):
            i = by_index[atom]
            lj.addParticle([float(sigma[i]), float(eps[i])])
        for i, j in pairs:
            lj.addExclusion(i, j)
        forces.append(lj)

    # ZBL takes no exclusions -- see the `ZBL` docstring; it is a function of the
    # geometry and the elements alone.
    zbl = openmm.CustomNonbondedForce(ZBL)
    zbl.setName("ZBL")
    zbl.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
    zbl.addPerParticleParameter("z")
    zbl.addGlobalParameter("screen", SCREENING_LENGTH * u.LENGTH)
    zbl.addGlobalParameter("zbl_ccoul", ZBL_CCOUL * u.ENERGY * u.LENGTH)
    for k, (c, b) in enumerate(zip(PHI_C, PHI_B, strict=True)):
        zbl.addGlobalParameter(f"pc{k}", c)
        zbl.addGlobalParameter(f"pb{k}", b)
    for z in params.numbers:
        zbl.addParticle([float(z)])
    forces.append(zbl)

    if "atom" in params.terms and positions is not None:
        charges = acks2_charges(params, np.asarray(positions, dtype=float))
        coulomb = openmm.CustomNonbondedForce(COULOMB)
        coulomb.setName("Coulomb")
        coulomb.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
        coulomb.addPerParticleParameter("q")
        coulomb.addGlobalParameter("acks2_ccoul", ACKS2.CCOUL * u.ENERGY * u.LENGTH)
        coulomb.addGlobalParameter("beta", ACKS2_BETA / u.LENGTH)
        for q in charges:
            coulomb.addParticle([float(q)])
        forces.append(coulomb)
    return forces


def acks2_charges(params, positions: np.ndarray) -> np.ndarray:
    """ACKS2 charges at `positions` (Angstrom), in global atom order."""
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
