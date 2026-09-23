"""Command line entry point: parameterize a SMILES string, or a whole manifest.

    fast-forces CC#N -o acetonitrile.jsonl
    fast-forces ../DynamicTopo/datasets/Water/Water.json --dry-run
    fast-forces refit ../DynamicTopo/datasets/Water/Water.json --force-constants
    fast-forces label molecules/ -o labelled/        # reference energies (PySCF)
    fast-forces import-qforce qforce.xml -o mol.jsonl

`refit`, `label` and `import-qforce` are the dataset tools that used to live in
DynamicTopology's `scripts/` (`fit.py`, `compute.py`, `convert.py`); each takes
`--help`.
"""

import argparse
import importlib
import sys

from . import FitConfig, build, parameterize


def manifest_main(argv: list[str]) -> None:
    """Fit every molecule and reaction a manifest lists; see `manifest`."""
    from . import manifest

    parser = argparse.ArgumentParser(
        prog="fast-forces", description=manifest_main.__doc__
    )
    parser.add_argument("manifest", help="manifest .json file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the manifest and print what would be fitted",
    )
    args = parser.parse_args(argv)

    try:
        loaded = manifest.load(args.manifest)
    except manifest.ManifestError as error:
        parser.exit(2, f"fast-forces: {error}\n")
    if args.dry_run:
        print(loaded.plan())
        return
    outcomes = manifest.run(loaded)
    if not all(o.ok for o in outcomes):
        sys.exit(1)


# Subcommand -> module whose `main(argv)` runs it.
SUBCOMMANDS = {"refit": "refit", "label": "label", "import-qforce": "qforce_xml"}


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in SUBCOMMANDS:
        module = importlib.import_module(f".{SUBCOMMANDS[argv[0]]}", __package__)
        sys.exit(module.main(argv[1:]))
    if argv and argv[0].endswith(".json"):
        manifest_main(argv)
        return

    parser = argparse.ArgumentParser(prog="fast-forces", description=__doc__)
    parser.add_argument("smiles", help="SMILES string of the molecule to fit")
    parser.add_argument("-o", "--output", default=None, help="jsonl output path")
    parser.add_argument("--xml", default=None, help="also write an OpenMM XML system")
    parser.add_argument("--training-set", default="training.xyz")
    parser.add_argument("--method", default="GFN2-xTB", help="tblite method")
    parser.add_argument("--temperature", type=float, default=FitConfig.temperature)
    parser.add_argument("--mode-frames", type=int, default=FitConfig.n_mode_frames)
    parser.add_argument(
        "--initial",
        default=None,
        help="jsonl force field to start the fit from, instead of the element table",
    )
    args = parser.parse_args(argv)

    from .calculators.tblite import TBLiteCalculator

    atoms = build(args.smiles)
    config = FitConfig(temperature=args.temperature, n_mode_frames=args.mode_frames)
    params = parameterize(
        atoms,
        lambda a: TBLiteCalculator(method=args.method, verbosity=0),
        config=config,
        training_set=args.training_set,
        initial=args.initial,
    )
    print(params.fit_report())

    output = args.output or "forcefield.jsonl"
    params.to_jsonl(output)
    print(f"wrote {output} and {args.training_set}")
    if args.xml:
        params.to_openmm_xml(args.xml, positions=atoms.get_positions())
        print(f"wrote {args.xml}")


if __name__ == "__main__":
    main()
