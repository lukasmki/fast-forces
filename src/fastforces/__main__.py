"""Command line entry point: parameterize a SMILES string."""

import argparse

from . import FitConfig, build, parameterize


def main() -> None:
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
    args = parser.parse_args()

    from tblite.ase import TBLite

    atoms = build(args.smiles)
    config = FitConfig(temperature=args.temperature, n_mode_frames=args.mode_frames)
    params = parameterize(
        atoms,
        lambda a: TBLite(method=args.method, verbosity=0),
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
