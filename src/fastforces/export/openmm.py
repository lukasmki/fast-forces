"""Export to an OpenMM `System`, and to the XML `XmlSerializer` writes.

The System is built through the OpenMM API rather than by assembling XML by
hand, so the serialized file is valid by construction.

The energy expressions here are transcriptions of DynamicTopology's force field
-- its `QForce.compute_*` methods, `zbl.pair_potential` and `lj.pair_potential`
-- not of the example XML.  Where the two disagree, the force field wins,
because an exported file that does not reproduce the calculator it was fit with
is worse than useless.  The constants are read from DynamicTopology's *active*
`global_params` at export time, so a field exported inside its dataset's
`params.use(...)` carries that dataset's radii.  Where that matters:

  * `dihedralangle` and `dihedralangleangle` use `cos(angle) - cos(theta0)`,
    matching the implementation; the example XML uses the raw angle difference
    there, while using the cosine form for `bondangle` and `angleangle`.
  * The `bondbond` and `bondangle` clamps are q-force's `-10` and `-20` kJ/mol
    (`qforce.CLIP_BONDBOND` / `CLIP_BONDANGLE`, held in eV).
  * The Morse bond keeps its `-D` offset and its one-sided per-bond asymptote
    `h` (`bond_asymptote` where a bond states none), which deepens the stretched
    branch only.
  * `angle` is `0.5*k*(cos-cos0)^2`, the convention the example files are
    written in, so an angle `k` means the same well on both sides.
  * Both nonbonded pair terms carry a Fermi switch -- `zbl.taper` switching ZBL
    off outside 1.5 A, `lj.switch` switching the 12-6 on outside 2.2 A, and
    `core_fraction` replacing the `r**-12` divergence with a tangent.  A
    transcription that dropped any of those would be a different force field at
    exactly the separations bonded pairs sit at.
  * All three nonbonded forces carry the pairs within `exclusion_depth` bonds as
    real OpenMM exclusions, which is the export of DynamicTopology's exclusion
    terms.  OpenMM removes an excluded pair from the sum outright, where the
    calculator sums it and subtracts it back; the two agree because both halves
    of the calculator's cancellation go through one `pair_potential`.  (An
    excluded pair with no 12-6 -- a zero sigma -- has no `exclusion` term there
    and contributes nothing here, which is the same number.)

Electrostatics is the one place the export cannot be faithful, and only for one
of the two terms.  ACKS2 solves for the charges at every geometry; OpenMM has no
charge-equilibration force, so an `atom` block is solved once at the geometry
passed in and baked in as fixed values.  A simulation run from that exported
file has fixed charges, and will drift from the FastForces calculator as the
geometry moves away from the reference.

A `charge` block has nothing to bake: its charges *are* fixed, so the exported
force is the term itself and needs no geometry.  Both export to the same
`erf(gamma r)/r` expression, DynamicTopology's charge kernel.
"""

import numpy as np
from ase import Atoms
from ase.data import atomic_masses
from DynamicTopology.forcefield.exclusions import bond_graph
from DynamicTopology.forcefield.lj import near_pairs
from DynamicTopology.forcefield.params import active
from DynamicTopology.forcefield.qforce import CLIP_BONDANGLE, CLIP_BONDBOND
from DynamicTopology.forcefield.zbl import PHI_B, PHI_C, SCREENING_LENGTH
from DynamicTopology.io import units as u

# The clamps, from `QForce.compute_bondbond` / `compute_bondangle`, imported
# rather than restated so the two sides cannot drift.  Held in eV there, so
# `u.ENERGY` takes them back to the -10 and -20 kJ/mol q-force states.

# Both Fermi switches overflow `exp` if handed a raw exponent -- these forces are
# evaluated with no cutoff, so `r` reaches the box diagonal.  The evaluators clip
# at 500 in numpy; Lepton has no `clip`, so `max(min(...))` does it here.  300 is
# well past where either switch resolves anything (`exp(-300)` is 5e-131) and
# stays inside double range, which 500 does not.
SWITCH_CLAMP = 300.0

# `Dw` is the stretched branch's deeper well, `D + h`; `step(r-r0)` picks it.
# The two branches agree in value, slope and curvature at `r = r0`, so which one
# the step hands back exactly there does not matter.
MORSE = "Dw*(1-exp(-a*(r-r0)))^2 - D; a=sqrt(k/(2*Dw)); Dw=D+h*step(r-r0)"
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
    return np.asarray(params.terms[term]["kwargs"][name]) * u.factor(term, name)


def build_system(params, positions: np.ndarray | None = None, box: float = 20.0):
    """An `openmm.System` reproducing `params`.

    `positions` (Angstrom) fixes the geometry the ACKS2 charges are solved at;
    without it the electrostatic force is omitted entirely rather than written
    with wrong charges.  A field carrying fixed `charge` terms does not need
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
            "e0_per_atom", e0 * u.factor("reference", "E0") / len(params.numbers)
        )
        for i in range(len(params.numbers)):
            offset.addParticle(i, [])
        offset.setName("Reference")
        system.addForce(offset)
    return system


def _nonbonded_forces(params, positions, edge):
    import openmm

    ff = active()
    forces = []
    excluded = exclusion_pairs(params)

    if "lennardjones" in params.terms:
        lj = openmm.CustomNonbondedForce(LENNARDJONES)
        lj.setName("LennardJones")
        lj.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
        lj.addPerParticleParameter("A")
        lj.addPerParticleParameter("B")
        # In Angstrom in DynamicTopology, so both take `u.LENGTH` here.
        # `core_frac` is a fraction of sigma and converts by 1.
        lj.addGlobalParameter("sw_r", ff.switch_radius * u.LENGTH)
        lj.addGlobalParameter("sw_w", ff.switch_width * u.LENGTH)
        lj.addGlobalParameter("core_frac", ff.core_fraction)
        lj.addGlobalParameter("sw_clamp", SWITCH_CLAMP)
        sigma = _converted(params, "lennardjones", "sigma")
        eps = _converted(params, "lennardjones", "eps")
        order = np.asarray(params.terms["lennardjones"]["atoms"])[:, 0]
        by_index = {int(a): i for i, a in enumerate(order)}
        for atom in range(len(params.numbers)):
            i = by_index[atom]
            lj.addParticle([float(sigma[i]), float(eps[i])])
        _add_exclusions(lj, excluded)
        forces.append(lj)

    zbl = openmm.CustomNonbondedForce(ZBL)
    zbl.setName("ZBL")
    zbl.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
    zbl.addPerParticleParameter("z")
    zbl.addGlobalParameter("screen", SCREENING_LENGTH * u.LENGTH)
    zbl.addGlobalParameter("zbl_ccoul", ff.zbl_ccoul * u.ENERGY * u.LENGTH)
    # In Angstrom in `zbl`, which works in ASE units throughout.
    zbl.addGlobalParameter("taper_r", ff.taper_radius * u.LENGTH)
    zbl.addGlobalParameter("taper_w", ff.taper_width * u.LENGTH)
    zbl.addGlobalParameter("sw_clamp", SWITCH_CLAMP)
    for k, (c, b) in enumerate(zip(PHI_C, PHI_B, strict=True)):
        zbl.addGlobalParameter(f"pc{k}", c)
        zbl.addGlobalParameter(f"pb{k}", b)
    for z in params.numbers:
        zbl.addParticle([float(z)])
    _add_exclusions(zbl, excluded)
    forces.append(zbl)

    charges = exported_charges(params, positions)
    if charges is not None:
        coulomb = openmm.CustomNonbondedForce(COULOMB)
        coulomb.setName("Coulomb")
        coulomb.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
        coulomb.addPerParticleParameter("q")
        coulomb.addGlobalParameter("acks2_ccoul", ff.ccoul * u.ENERGY * u.LENGTH)
        coulomb.addGlobalParameter("beta", ff.gamma / u.LENGTH)
        for q in charges:
            coulomb.addParticle([float(q)])
        if ff.exclude_coulomb:
            _add_exclusions(coulomb, excluded)
        forces.append(coulomb)
    return forces


def exclusion_pairs(params) -> list[tuple[int, int]]:
    """The pairs within `exclusion_depth` bonds, from the field's `bond` terms.

    The same pairs DynamicTopology derives its exclusion terms for -- the same
    graph, the same depth, the same shortest-path search.
    """
    graph = bond_graph(params.to_terms(), len(params.numbers))
    return sorted(near_pairs(graph, active().exclusion_depth))


def _add_exclusions(force, pairs):
    """Put the field's exclusions on a `CustomNonbondedForce` as OpenMM exclusions.

    The only faithful export available: OpenMM drops an excluded pair from the
    sum outright, where the calculator sums it and subtracts it back through
    the same `pair_potential`.  Those are the same number, so the exported
    force matches -- but only because the calculator's two halves are one
    function.
    """
    for i, j in pairs:
        force.addExclusion(int(i), int(j))


def exported_charges(params, positions: np.ndarray | None) -> np.ndarray | None:
    """The fixed charges the exported Coulomb force carries, in global order.

    `None` when there are none to write, which is either a field with no
    electrostatics at all or an ACKS2 field exported without a geometry to
    solve at -- the electrostatic force is then omitted entirely rather than
    written with wrong charges.

    A `charge` field ignores `positions`: its charges do not depend on the
    geometry, which is the whole difference between the two terms.
    """
    term = params.electrostatics()
    if term == "charge":
        block = params.terms["charge"]
        charges = np.zeros(len(params.numbers))
        charges[np.asarray(block["atoms"])[:, 0]] = block["kwargs"]["q"]
        return charges
    if term == "atom" and positions is not None:
        return acks2_charges(params, np.asarray(positions, dtype=float))
    return None


def acks2_charges(params, positions: np.ndarray) -> np.ndarray:
    """ACKS2 charges at `positions` (Angstrom), in global atom order.

    Solved by DynamicTopology under open boundaries.  The exported system is
    periodic in a 20 A box by default, so this is already an approximation on
    top of freezing the charges; both are the same trade `build_system`
    documents.
    """
    from ..calculator import breakdown

    atoms = Atoms(numbers=params.numbers, positions=positions)
    return breakdown(atoms, params).charges


def _bonded_forces(params):
    import openmm

    forces = []

    if "bond" in params.terms:
        bond = openmm.CustomBondForce(MORSE)
        bond.setName("Bond")
        for name in ("r0", "k", "D", "h"):
            bond.addPerBondParameter(name)
        atoms = np.asarray(params.terms["bond"]["atoms"])
        kwargs = params.terms["bond"]["kwargs"]
        if "h" not in kwargs:
            # A bond without its own asymptote reads `bond_asymptote`, as
            # `QForce._bond_morse` does.
            kwargs = {**kwargs, "h": np.full(len(atoms), active().bond_asymptote)}
        values = [
            np.asarray(kwargs[n]) * u.factor("bond", n) for n in ("r0", "k", "D", "h")
        ]
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
