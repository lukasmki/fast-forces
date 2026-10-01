"""Reference data generation: the geometries the force field is fit to.

Every function here takes a `calc_factory` -- a callable turning an `Atoms` into
a fresh ASE calculator -- so that any calculator that provides energies and
forces can drive the fit, which is the point of the package.  Each returns
frames carrying `energy` and `forces` on a `SinglePointCalculator`, tagged with
`info["frame_kind"]` so `io` can write them all into one file and `fit` can
weight them separately.
"""

import numpy as np
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixInternals
from ase.optimize import BFGS

# hbar * 1e10 / sqrt(e * amu): converts sqrt(eV / (Angstrom^2 amu)) to eV, the
# same constant `ase.vibrations` uses.
VIB_ENERGY_SCALE = units._hbar * 1e10 / (units._e * units._amu) ** 0.5

# Modes below this energy (eV) are translations, rotations, or numerical noise.
# The cut is on the mode *energy*, not the raw mass-weighted eigenvalue: a
# residual rotation comes out at a few meV, which is a small eigenvalue but a
# huge `sqrt(kT)/w` amplitude, and displacing along it produces geometries no
# reference method will converge.  10 meV is about 80 cm^-1, below any real
# vibration and well above the rotational contamination.
TRIVIAL_MODE_ENERGY = 0.010

# Hard cap on any single atom's displacement, in Angstrom.  A genuinely soft
# mode (the H2O2 torsion is 31 meV) still earns a large amplitude; this only
# stops the tail of the Gaussian from producing a dissociated geometry.
MAX_DISPLACEMENT = 0.35


def label(atoms: Atoms, calc_factory, kind: str, strict: bool = True) -> Atoms | None:
    """Evaluate `atoms` with a fresh calculator and freeze the result onto it.

    With `strict=False` a calculator failure -- an SCF that will not converge on
    a strained geometry, most often -- drops the frame instead of ending the
    run.  Sampling deliberately visits geometries near the edge of what a
    reference method can handle, so losing a few of them is normal; losing the
    equilibrium geometry or the Hessian is not, and those callers stay strict.
    """
    frame = atoms.copy()
    frame.calc = calc_factory(frame)
    try:
        energy = frame.get_potential_energy()
        forces = frame.get_forces()
    except Exception:
        if strict:
            raise
        return None
    # Keep whatever the calculator wrote onto the frame (pyscf writes
    # connectivity and bond orders, both write `mulliken` charges) but drop the
    # live calculator.  What it wrote replaces an array `atoms` already carried:
    # a Hessian frame is a copy of the labelled equilibrium, and a fragment can
    # be sliced out of a labelled parent, so the incoming array is a stale value
    # from another geometry.
    info = dict(frame.info)
    arrays = {
        k: v for k, v in frame.arrays.items() if k not in ("numbers", "positions")
    }
    out = atoms.copy()
    out.info.update(info)
    for k, v in arrays.items():
        out.set_array(k, v)
    out.info["frame_kind"] = kind
    out.calc = SinglePointCalculator(out, energy=energy, forces=forces)
    return out


def optimize(
    atoms: Atoms,
    calc_factory,
    fmax: float = 1e-3,
    steps: int = 500,
    kind: str = "equilibrium",
) -> Atoms:
    """Relax to the equilibrium geometry every equilibrium value is read from.

    `fmax` is tighter than a typical geometry optimization on purpose: `r0` and
    `theta0` are taken straight off this structure, and the Hessian is finite
    differenced about it, so residual forces here leak into every fitted term.

    `kind` is the frame tag, and anything relaxing something other than the
    parent molecule has to pass one.  `TrainingSet.equilibrium` is the *first*
    frame tagged `"equilibrium"`, so a dissociation fragment left at the default
    could shadow the molecule and quietly corrupt every fitted term.
    """
    work = atoms.copy()
    work.calc = calc_factory(work)
    BFGS(work, logfile=None).run(fmax=fmax, steps=steps)
    out = atoms.copy()
    out.set_positions(work.get_positions())
    return label(out, calc_factory, kind)


def hessian(
    atoms: Atoms, calc_factory, delta: float = 0.01
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Atoms]]:
    """Central-difference Hessian, plus its mass-weighted normal modes.

    Returns `(H, energies, modes, frames)`: the `(3N, 3N)` Cartesian Hessian in
    eV/Angstrom^2, the mode energies in eV, the mass-weighted eigenvectors as
    `(n_modes, N, 3)` Cartesian displacements, and the `6N` displaced frames --
    which are themselves training data and cost nothing extra.

    Deliberately not `ase.vibrations.Vibrations`, which insists on a directory
    of cache files.
    """
    n = len(atoms)
    pos = atoms.get_positions()
    h = np.zeros((3 * n, 3 * n))
    frames: list[Atoms] = []

    for i in range(n):
        for axis in range(3):
            forces = []
            for sign in (+1, -1):
                shifted = atoms.copy()
                shifted.positions = pos.copy()
                shifted.positions[i, axis] += sign * delta
                frame = label(shifted, calc_factory, "hessian")
                # Tagged so `io.rebuild_hessian` can reconstruct H from the file
                # alone -- the Hessian never has to be serialized separately.
                frame.info["hessian_column"] = 3 * i + axis
                frame.info["hessian_sign"] = sign
                frame.info["hessian_delta"] = delta
                frames.append(frame)
                forces.append(frame.get_forces())
            # H = -dF/dR
            h[3 * i + axis] = -(forces[0] - forces[1]).ravel() / (2 * delta)

    h = 0.5 * (h + h.T)

    masses = atoms.get_masses()
    inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)
    h_mass = h * inv_sqrt_m[:, None] * inv_sqrt_m[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(h_mass)

    energies = VIB_ENERGY_SCALE * np.sqrt(np.abs(eigenvalues)) * np.sign(eigenvalues)
    modes = (eigenvectors.T * inv_sqrt_m).reshape(-1, n, 3)
    return h, energies, modes, frames


def normal_mode_frames(
    atoms: Atoms,
    calc_factory,
    hessian_matrix: np.ndarray,
    n_frames: int = 40,
    temperature: float = 500.0,
    seed: int = 0,
) -> list[Atoms]:
    """Frames displaced along the normal modes, Boltzmann-distributed.

    In mass-weighted coordinates the harmonic potential is `0.5 w^2 q^2`, so the
    classical thermal distribution is `q ~ N(0, sqrt(kT)/w)`: stiff modes get
    small displacements and soft ones large.  That is what makes the linear
    refine well conditioned -- isotropic Cartesian noise spends nearly all its
    amplitude on the stiff bond stretches and barely probes the soft
    coordinates the cross terms describe.

    Trivial modes (translations, rotations, anything imaginary) are skipped.
    """
    rng = np.random.default_rng(seed)
    masses = atoms.get_masses()
    inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)
    h_mass = hessian_matrix * inv_sqrt_m[:, None] * inv_sqrt_m[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(h_mass)

    energies = VIB_ENERGY_SCALE * np.sqrt(np.abs(eigenvalues)) * np.sign(eigenvalues)
    keep = energies > TRIVIAL_MODE_ENERGY
    if not np.any(keep):
        raise ValueError("no real vibrational modes; is the geometry a minimum?")
    omega = np.sqrt(eigenvalues[keep])
    vectors = eigenvectors[:, keep]

    kt = units.kB * temperature
    pos = atoms.get_positions()
    frames = []
    while len(frames) < n_frames:
        q = rng.normal(scale=np.sqrt(kt) / omega)
        displacement = ((vectors @ q) * inv_sqrt_m).reshape(-1, 3)
        largest = np.linalg.norm(displacement, axis=1).max()
        if largest > MAX_DISPLACEMENT:
            displacement *= MAX_DISPLACEMENT / largest
        shifted = atoms.copy()
        shifted.positions = pos + displacement
        frame = label(shifted, calc_factory, "mode", strict=False)
        if frame is not None:
            frames.append(frame)
    return frames


def conformer_frames(
    atoms: Atoms, calc_factory, n_conformers: int = 20, seed: int = 42
) -> list[Atoms]:
    """Distinct conformers, labelled with the reference calculator.

    Uses `openconf` when a SMILES string is available on the frame, and falls
    back to `molify.smiles2conformers`.  A molecule with no rotatable bonds has
    nothing to sample, and gets an empty list rather than an error.
    """
    smiles = atoms.info.get("smiles")
    if smiles is None:
        return []

    try:
        from openconf import ConformerConfig, generate_conformers

        ensemble = generate_conformers(
            smiles, config=ConformerConfig(max_conformers=n_conformers, seed=seed)
        )
        geometries = [
            np.array(ensemble.coords(i)) for i in range(ensemble.n_conformers)
        ]
    except Exception:
        from molify import smiles2conformers

        geometries = [
            c.get_positions()
            for c in smiles2conformers(smiles, n_conformers, randomSeed=seed)
        ]

    frames = []
    for geometry in geometries:
        if geometry.shape != (len(atoms), 3):
            continue  # a different atom ordering is not usable as a frame here
        shifted = atoms.copy()
        shifted.set_positions(geometry)
        frame = label(shifted, calc_factory, "conformer", strict=False)
        if frame is not None:
            frames.append(frame)
    return frames


def torsion_frames(
    atoms: Atoms,
    calc_factory,
    dihedrals: list[tuple[int, int, int, int]],
    step_deg: float = 15.0,
    fmax: float = 0.05,
    steps: int = 60,
) -> list[Atoms]:
    """Relaxed torsion scans, one per rotatable bond.

    The Hessian sees the torsion profile only to second order about the
    minimum, which is not enough to separate the `n = 1..4` terms of the
    periodic series from each other.  A relaxed scan is what makes those
    coefficients identifiable, so leaving this out quietly degrades every
    dihedral term rather than failing.
    """
    frames = []
    for a, b, c, d in dihedrals:
        for angle in np.arange(0.0, 360.0, step_deg):
            work = atoms.copy()
            try:
                work.set_dihedral(a, b, c, d, float(angle))
            except Exception:
                continue  # ring closure: the dihedral cannot be set independently
            work.set_constraint(
                FixInternals(dihedrals_deg=[[float(angle), [a, b, c, d]]])
            )
            work.calc = calc_factory(work)
            try:
                BFGS(work, logfile=None).run(fmax=fmax, steps=steps)
            except Exception:
                continue
            relaxed = atoms.copy()
            relaxed.set_positions(work.get_positions())
            frame = label(relaxed, calc_factory, "torsion", strict=False)
            if frame is not None:
                frames.append(frame)
    return frames


def rotatable_dihedrals(atoms: Atoms) -> list[tuple[int, int, int, int]]:
    """Dihedrals worth scanning, from `openconf`'s rotor perception."""
    smiles = atoms.info.get("smiles")
    if smiles is None:
        return []
    try:
        from openconf import build_rotor_model, smiles_to_mol

        model = build_rotor_model(smiles_to_mol(smiles))
        return [tuple(int(i) for i in r.dihedral_atoms) for r in model.rotors]
    except Exception:
        return []


def bridge_bonds(topology) -> dict[int, tuple[int, int]]:
    """One bond per bond class whose removal splits the molecule in two.

    Keyed by class.  A ring bond has no fragments to break into -- stretching
    it opens the ring rather than dissociating anything -- so a class of ring
    bonds only is absent, and its asymptote stays `bond_asymptote`.
    """
    import networkx as nx

    bridges = {frozenset(e) for e in nx.bridges(topology.graph)}
    out: dict[int, tuple[int, int]] = {}
    for (i, j), c in zip(topology.atoms["bond"], topology.classes["bond"]):
        if int(c) not in out and frozenset((int(i), int(j))) in bridges:
            out[int(c)] = (int(i), int(j))
    return out


def _cut_fragments(smiles: str, atoms: Atoms, graph, bond) -> list[tuple]:
    """`[(indices, 2S), (indices, 2S)]`: the two sides of cutting `bond` homolytically.

    Each side keeps the radicals its atoms carry in `smiles` and gains one
    electron per unit of the bond order it loses, all high-spin coupled -- the
    convention `Reaction.spin` and a manifest molecule's default spin follow,
    so the fragments come out at the spin the dataset fits them at (the O of
    OH -> O + H at the triplet, not the singlet the electron count allows).

    The SMILES is where the radicals and bond orders are stated, and its atoms
    are matched onto `atoms` by graph isomorphism, since nothing promises the
    two share an order.
    """
    import networkx as nx
    from networkx.algorithms.isomorphism import GraphMatcher
    from rdkit import Chem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    reference = nx.Graph()
    reference.add_nodes_from(
        (a.GetIdx(), {"Z": a.GetAtomicNum()}) for a in mol.GetAtoms()
    )
    reference.add_edges_from(
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
    )
    target = nx.Graph()
    target.add_nodes_from(
        (k, {"Z": int(z)}) for k, z in enumerate(atoms.get_atomic_numbers())
    )
    target.add_edges_from(graph.edges)
    matcher = GraphMatcher(target, reference, node_match=lambda a, b: a["Z"] == b["Z"])
    mapping = next(matcher.isomorphisms_iter(), None)  # atoms index -> SMILES index
    if mapping is None:
        raise ValueError(f"the geometry is not bonded as {smiles!r} says")

    i, j = bond
    order = int(round(mol.GetBondBetweenAtoms(mapping[i], mapping[j]).GetBondTypeAsDouble()))
    cut = graph.copy()
    cut.remove_edge(i, j)
    sides = []
    for end in (i, j):
        indices = sorted(nx.node_connected_component(cut, end))
        radicals = sum(
            mol.GetAtomWithIdx(mapping[k]).GetNumRadicalElectrons() for k in indices
        )
        sides.append((indices, radicals + order))
    return sides


def fragment_frames(equilibrium: Atoms, calc_factory, topology=None) -> list[Atoms]:
    """The two fragments of each bond class, labelled where they sit.

    What a molecule's stretched limit is measured against.  Cutting one bond
    of each class (`bridge_bonds`) leaves two fragments, each labelled as a
    single point at its frozen equilibrium geometry: the dissociated diabat an
    EVB crossing hands over to is evaluated at exactly those geometries, so
    the vertical energy is the one it has to reach, not the relaxed one.  The
    fragments carry the formal charges of their atoms and the spins
    `_cut_fragments` assigns.

    Tagged `"fragment"`, with `fragment_bond` the parent bond cut and
    `fragment_atoms` the parent indices each one holds, so `fit` can pair them
    up again.  Nothing is returned without `info["smiles"]`, which is where
    the spins come from, and a pair either of whose calculations fails is
    dropped: its class keeps the default asymptote.
    """
    from .topology import enumerate_terms, perceive

    smiles = equilibrium.info.get("smiles")
    if smiles is None or len(equilibrium) < 2:
        return []
    if topology is None:
        topology = enumerate_terms(equilibrium, perceive(equilibrium))
    frames = []
    for bond in bridge_bonds(topology).values():
        pair = []
        for indices, spin in _cut_fragments(smiles, equilibrium, topology.graph, bond):
            fragment = Atoms(
                numbers=equilibrium.get_atomic_numbers()[indices],
                positions=equilibrium.get_positions()[indices],
                charges=equilibrium.get_initial_charges()[indices],
            )
            fragment.info.update(
                spin=int(spin), fragment_bond=list(bond), fragment_atoms=list(indices)
            )
            pair.append(label(fragment, calc_factory, "fragment", strict=False))
        if all(frame is not None for frame in pair):
            frames += pair
    return frames


def atom_frames(equilibrium: Atoms, calc_factory) -> list[Atoms]:
    """One free, neutral atom of each element in the molecule, labelled.

    What a molecule's atomization energy is measured against, `E(molecule) -
    sum_i E(atom_i)`, which `fit` puts on the bonds' depths.  Each atom is at
    its ground-state spin (`elements.ATOM_SPIN`, Hund's rule), and neutral
    whatever the molecule's charge, the convention `label` uses for a dataset's
    atomization energies -- so an ion's includes its ionization energy or
    electron affinity.

    Tagged `"atom"`.  All or nothing: the sum needs every element, so if any
    one fails to label, or has no tabulated spin, none is returned and the
    depths keep their element-table values.
    """
    from .elements import ATOM_SPIN

    if len(equilibrium) < 2:
        return []
    frames = []
    for symbol in sorted(set(equilibrium.get_chemical_symbols())):
        if symbol not in ATOM_SPIN:
            return []
        atom = Atoms(symbol, positions=[[0.0, 0.0, 0.0]])
        atom.info["spin"] = ATOM_SPIN[symbol]
        frame = label(atom, calc_factory, "atom", strict=False)
        if frame is None:
            return []
        frames.append(frame)
    return frames
