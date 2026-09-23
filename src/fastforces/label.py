"""Label structures with wB97X-V atomization energies: `fast-forces label`.

    fast-forces label -i rxn.xyz -o rxn.xyz -b aug-cc-pvtz -c 1 --atom-cache atoms.json

The reference energies a DynamicTopology dataset carries on every template and
reaction frame (`energy=` in its `.xyz`), which `refine` fits the Morse depths
and the coupling amplitudes against.  Formerly DynamicTopology's
`scripts/compute.py`.

For every structure the total energy is evaluated with PySCF and the
free-atom energies of its constituent elements are subtracted, i.e.

    E_atomization = E(molecule) - sum_i E(atom_i)

(negative for a bound molecule).  Results are stored on an ASE
SinglePointCalculator in eV and written to an extxyz file.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import cast

from ase import Atoms, io, units
from ase.calculators.singlepoint import SinglePointCalculator

from pyscf import dft, gto, lib

from .calculators.pyscf import ase_to_pyscf

XC = "wb97x_v"

# Number of unpaired electrons (2S = n_alpha - n_beta) in the ground state
# of the neutral atoms, i.e. Hund's rule applied to the atomic configuration.
ATOM_SPIN: dict[str, int] = {
    "H": 1,
    "He": 0,
    "Li": 1,
    "Be": 0,
    "B": 1,
    "C": 2,
    "N": 3,
    "O": 2,
    "F": 1,
    "Ne": 0,
    "Na": 1,
    "Mg": 0,
    "Al": 1,
    "Si": 2,
    "P": 3,
    "S": 2,
    "Cl": 1,
    "Ar": 0,
    "K": 1,
    "Ca": 0,
    "Sc": 1,
    "Ti": 2,
    "V": 3,
    "Cr": 6,
    "Mn": 5,
    "Fe": 4,
    "Co": 3,
    "Ni": 2,
    "Cu": 1,
    "Zn": 0,
    "Ga": 1,
    "Ge": 2,
    "As": 3,
    "Se": 2,
    "Br": 1,
    "Kr": 0,
}


def _run_scf(mol: gto.Mole, grid_level: int, nlc_grid_level: int, df: bool) -> float:
    """Run a wB97X-V SCF and return the total energy in Hartree."""
    mf = dft.KS(mol)  # RKS for mol.spin == 0, UKS otherwise
    mf.xc = XC

    # wB97X-V contains the VV10 non-local correlation term.  Recent PySCF
    # versions switch it on automatically for this functional; setting it
    # explicitly keeps older versions honest (the b/C coefficients still come
    # from libxc's definition of wb97x_v).
    mf.nlc = "VV10"

    mf.grids.level = grid_level
    # The VV10 kernel is smooth, so a coarser grid is enough and much cheaper.
    mf.nlcgrids.level = nlc_grid_level

    if df:
        mf = mf.density_fit()

    mf.conv_tol = 1e-9
    mf.max_cycle = 100
    energy = mf.kernel()

    if not mf.converged:
        # Fall back to the second-order (Newton) solver from the current density.
        dm0 = mf.make_rdm1()
        mf = mf.newton()
        energy = mf.kernel(dm0)

    if not mf.converged:
        raise RuntimeError(f"SCF did not converge for {mol.atom_symbols()}")

    return float(energy)


def atom_energy(
    symbol: str,
    basis: str,
    cache: dict[str, float],
    grid_level: int,
    nlc_grid_level: int,
    df: bool,
    verbose: int,
) -> float:
    """Spin-polarized free-atom energy in Hartree, memoized per element."""
    if symbol in cache:
        return cache[symbol]

    try:
        spin = ATOM_SPIN[symbol]
    except KeyError as exc:
        raise KeyError(
            f"No ground-state spin known for element {symbol!r}; add it to ATOM_SPIN."
        ) from exc

    mol = gto.M(
        atom=[(symbol, (0.0, 0.0, 0.0))], basis=basis, spin=spin, verbose=verbose
    )
    # Always use an unrestricted reference for atoms (identical to RKS when
    # spin == 0, but keeps open-shell atoms variational).
    mol.build()
    energy = _run_scf(mol, grid_level, nlc_grid_level, df)

    cache[symbol] = energy
    return energy


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(prog="fast-forces label", description=__doc__)
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("-b", "--basis", default="cc-pvtz")
    parser.add_argument(
        "-c",
        "--charge",
        type=int,
        default=0,
        help="Total charge, applied to every structure in the input.",
    )
    parser.add_argument(
        "-s",
        "--spin",
        type=int,
        default=0,
        help="Number of unpaired electrons (2S = n_alpha - n_beta), applied to "
        "every structure in the input. Note this is not the multiplicity.",
    )
    parser.add_argument(
        "--atom-cache",
        type=Path,
        default=None,
        help="JSON file used to persist free-atom energies between runs.",
    )
    parser.add_argument("--grid-level", type=int, default=3)
    parser.add_argument("--nlc-grid-level", type=int, default=1)
    parser.add_argument(
        "--density-fit",
        action="store_true",
        help="Use RI/density fitting for the two-electron integrals (faster, "
        "introduces a small fitting error).",
    )
    parser.add_argument("--nthreads", type=int, default=None)
    parser.add_argument(
        "-v", "--verbose", type=int, default=0, help="PySCF verbosity level."
    )
    args = parser.parse_args(argv)

    if args.nthreads is not None:
        lib.num_threads(args.nthreads)

    atom_cache: dict[str, float] = {}
    if args.atom_cache is not None and args.atom_cache.exists():
        atom_cache = json.loads(args.atom_cache.read_text())

    atoms: list[Atoms] = cast(list[Atoms], io.read(args.input, index=":"))

    # Write incrementally so a crash halfway through does not lose everything.
    args.output.unlink(missing_ok=True)

    for i, at in enumerate(atoms):
        mol = ase_to_pyscf(
            at, args.basis, charge=args.charge, spin=args.spin, verbose=args.verbose
        )

        e_mol = _run_scf(mol, args.grid_level, args.nlc_grid_level, args.density_fit)

        e_atoms = sum(
            atom_energy(
                symbol,
                args.basis,
                atom_cache,
                args.grid_level,
                args.nlc_grid_level,
                args.density_fit,
                args.verbose,
            )
            for symbol in at.get_chemical_symbols()
        )

        e_atomization = (e_mol - e_atoms) * units.Hartree  # eV

        at.info["method"] = f"{XC}/{args.basis}"
        at.info["charge"] = args.charge
        at.info["spin"] = args.spin
        at.calc = SinglePointCalculator(at, energy=e_atomization)

        print(
            f"[{i + 1}/{len(atoms)}] {at.get_chemical_formula()}: "
            f"E = {e_mol:.8f} Ha, E_atomization = {e_atomization:.6f} eV",
            flush=True,
        )

        io.write(args.output, at, format="extxyz", append=i > 0)

        if args.atom_cache is not None:
            args.atom_cache.write_text(json.dumps(atom_cache, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
