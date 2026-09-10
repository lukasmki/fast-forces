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
    # connectivity and bond orders) but drop the live calculator.
    info = dict(frame.info)
    arrays = {k: v for k, v in frame.arrays.items() if k not in atoms.arrays}
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
