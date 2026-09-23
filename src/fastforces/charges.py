"""Fixed point charges from the reference electrostatic potential (Merz-Kollman).

The alternative to `fit._charge_block`'s Mulliken populations for a fixed-charge
(`pointcharge`) dataset: each template's charges are the least-squares fit to
its own electrostatic potential, at the reference level of theory and at the
template's own geometry, with the total constrained to the formal charge.

The potential is sampled on four shells at 1.4, 1.6, 1.8 and 2.0 times each
atom's Merz-Kollman radius, at 5 points per A^2, keeping only points outside
every other atom's shell.  Symmetry-equivalent atoms are then averaged within
the topology's equivalence classes (`topology.equivalence_classes`), which moves
them by very little and keeps a rotor free of a spurious electrostatic torsion.

These are gas-phase charges.  Every liquid fixed-charge water model carries a
larger dipole than this -- SPC/E 2.35 D against 1.85 D in the gas -- to stand in
for the polarization a fixed-charge model does not have.

Formerly DynamicTopology's `datasets/Water-fixed-pc/fit_charges.py`; `mk_charges`
is that script's fit and `charge_terms` turns its result into template terms.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

# Angstrom.  PySCF's own conversion (`pyscf.lib.param.BOHR`, CODATA 2010), not
# `ase.units.Bohr`, so the fitting points sit where PySCF puts the nuclei.
BOHR = 0.52917721092

# Merz-Kollman radii, in Angstrom.
MK_RADII: dict[str, float] = {
    "H": 1.20,
    "C": 1.50,
    "N": 1.50,
    "O": 1.40,
    "F": 1.35,
    "P": 1.80,
    "S": 1.75,
    "Cl": 1.70,
}
SHELLS: tuple[float, ...] = (1.4, 1.6, 1.8, 2.0)
DENSITY: float = 5.0  # points per A^2


def sphere(n: int) -> np.ndarray:
    """`n` near-uniform unit vectors on a Fibonacci spiral."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    phi = np.pi * (1.0 + 5**0.5) * k
    rho = np.sqrt(1.0 - z * z)
    return np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)


def mk_points(symbols: list[str], positions: np.ndarray) -> np.ndarray:
    """Merz-Kollman sampling points, in Angstrom."""
    try:
        radii = np.array([MK_RADII[s] for s in symbols])
    except KeyError as error:
        raise KeyError(
            f"no Merz-Kollman radius for {error.args[0]}; add it to MK_RADII"
        ) from None
    points = []
    for scale in SHELLS:
        for centre, radius in zip(positions, radii):
            r = scale * radius
            shell = centre + r * sphere(int(DENSITY * 4 * np.pi * r * r))
            d = np.linalg.norm(shell[:, None, :] - positions[None], axis=-1)
            points.append(shell[np.all(d >= scale * radii[None] - 1e-9, axis=1)])
    return np.concatenate(points)


def esp(mol, dm: np.ndarray, points_bohr: np.ndarray) -> np.ndarray:
    """Total electrostatic potential at `points_bohr`, in Hartree/e."""
    coords = mol.atom_coords()
    d = np.linalg.norm(points_bohr[:, None] - coords[None], axis=-1)
    nuclear = (mol.atom_charges()[None] / d).sum(axis=1)
    electronic = np.empty(len(points_bohr))
    for start in range(0, len(points_bohr), 256):
        chunk = points_bohr[start : start + 256]
        ints = mol.intor("int1e_grids", grids=chunk)
        electronic[start : start + 256] = np.einsum("gij,ij->g", ints, dm)
    return nuclear - electronic


def fit_esp(positions, V, points, total: int) -> np.ndarray:
    """Least-squares charges reproducing `V` on `points`, summing to `total`.

    All lengths in Bohr, `V` in Hartree/e.
    """
    inv = 1.0 / np.linalg.norm(points[:, None] - positions[None], axis=-1)
    n = len(positions)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = inv.T @ inv
    A[:n, n] = A[n, :n] = 1.0
    b = np.concatenate([inv.T @ V, [total]])
    return np.linalg.solve(A, b)[:n]


def class_average(q: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """`q` averaged within each equivalence class; the total is unchanged."""
    classes = np.asarray(classes, dtype=int)
    counts = np.bincount(classes)
    return (np.bincount(classes, weights=q) / counts)[classes]


def mk_charges(
    atoms: Atoms,
    charge: int,
    spin: int = 0,
    xc: str = "wb97x_v",
    basis: str = "aug-cc-pvtz",
    classes: np.ndarray | None = None,
    grid_level: int = 4,
) -> tuple[np.ndarray, float]:
    """Merz-Kollman charges of `atoms` and the relative RMS error of the fit.

    `classes` are the equivalence classes to average within; by default the
    topology's own (`enumerate_terms(atoms).atom_classes`).
    """
    from pyscf import dft

    from .calculators.pyscf import ase_to_pyscf

    if len(atoms) == 1:
        return np.array([float(charge)]), 0.0
    symbols = atoms.get_chemical_symbols()
    mol = ase_to_pyscf(atoms, basis=basis, charge=charge, spin=spin, verbose=0)
    mf = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
    mf.xc = xc
    if "_v" in xc.lower():
        mf.nlc = "VV10"
    mf.grids.level = grid_level
    mf.nlcgrids.level = 1
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"SCF did not converge for {atoms.get_chemical_formula()}")
    dm = mf.make_rdm1()
    if dm.ndim == 3:  # unrestricted: the potential sees the total density
        dm = dm[0] + dm[1]

    points = mk_points(symbols, atoms.positions)
    V = esp(mol, dm, points / BOHR)
    q = fit_esp(atoms.positions / BOHR, V, points / BOHR, charge)
    if classes is None:
        from .topology import enumerate_terms

        classes = enumerate_terms(atoms).atom_classes
    q = class_average(q, classes)

    inv = 1.0 / np.linalg.norm(points[:, None] / BOHR - atoms.positions[None] / BOHR, axis=-1)
    rrms = np.sqrt(np.mean((inv @ q - V) ** 2) / np.mean(V**2))
    return q, float(rrms)


def charge_terms(q: np.ndarray) -> list[dict]:
    """`q` as DynamicTopology `charge` terms (elementary charges, unconverted)."""
    from .params import block_to_terms

    q = np.asarray(q, dtype=float)
    return block_to_terms("charge", {"atoms": np.arange(len(q))[:, None], "kwargs": {"q": q}})
