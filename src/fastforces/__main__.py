"""Command line entry point: parameterize a SMILES string, or a whole manifest.

    fast-forces fit CC#N -o acetonitrile.jsonl
    fast-forces fit-manifest examples/hydrogen-combustion/hydrogen-combustion.json --dry-run
    fast-forces refit examples/hydrogen-combustion/hydrogen-combustion.json --force-constants
    fast-forces label -i rxn.xyz -o rxn.xyz        # reference energies (PySCF)
    fast-forces import-qforce -i qforce.xml -o mol.jsonl

`refit`, `label` and `import-qforce` are the dataset tools that used to live in
DynamicTopology's `scripts/` (`fit.py`, `compute.py`, `convert.py`); each takes
`--help`.
"""

import inspect
from pathlib import Path
from typing import Annotated

import typer

from . import FitConfig, build, label, parameterize, qforce_xml, refit


def _help(doc: str | None) -> str:
    """`doc` as click help, its indented blocks (examples, tables) kept verbatim.

    Click rewraps every paragraph except one headed by a `\\b` line.
    """
    paragraphs = inspect.cleandoc(doc or "").split("\n\n")
    return "\n\n".join("\b\n" + p if p.startswith(" ") else p for p in paragraphs)


app = typer.Typer(
    help=_help(__doc__),
    no_args_is_help=True,
    rich_markup_mode=None,
    add_completion=False,
)


@app.command("fit")
def fit_smiles(
    smiles: Annotated[str, typer.Argument(help="SMILES string of the molecule to fit")],
    output: Annotated[
        str, typer.Option("-o", "--output", help="jsonl output path")
    ] = "forcefield.jsonl",
    xml: Annotated[
        str | None, typer.Option(help="also write an OpenMM XML system")
    ] = None,
    training_set: str = "training.xyz",
    method: Annotated[str, typer.Option(help="tblite method")] = "GFN2-xTB",
    temperature: float = FitConfig.temperature,
    mode_frames: int = FitConfig.n_mode_frames,
    initial: Annotated[
        str | None,
        typer.Option(
            help="jsonl force field to start the fit from, instead of the element table"
        ),
    ] = None,
) -> None:
    """Fit one molecule from a SMILES string against tblite."""
    from .calculators.tblite import TBLiteCalculator

    atoms = build(smiles)
    config = FitConfig(temperature=temperature, n_mode_frames=mode_frames)
    params = parameterize(
        atoms,
        lambda a: TBLiteCalculator(method=method, verbosity=0),
        config=config,
        training_set=training_set,
        initial=initial,
    )
    print(params.fit_report())

    params.to_jsonl(output)
    print(f"wrote {output} and {training_set}")
    if xml:
        params.to_openmm_xml(xml, positions=atoms.get_positions())
        print(f"wrote {xml}")


@app.command("fit-manifest")
def fit_manifest(
    path: Annotated[
        Path,
        typer.Argument(
            metavar="MANIFEST", exists=True, dir_okay=False, help="manifest .json file"
        ),
    ],
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="validate the manifest and print what would be fitted"
        ),
    ] = False,
) -> None:
    """Fit every molecule and reaction a manifest lists; see `manifest`."""
    from . import manifest

    try:
        loaded = manifest.load(path)
    except manifest.ManifestError as error:
        print(f"fast-forces: {error}")
        raise typer.Exit(2)
    if dry_run:
        print(loaded.plan())
        return
    outcomes = manifest.run(loaded)
    if not all(o.ok for o in outcomes):
        raise typer.Exit(1)


app.command("refit", help=_help(refit.__doc__))(refit.main)
app.command("label", help=_help(label.__doc__))(label.main)
app.command("import-qforce", help=_help(qforce_xml.__doc__))(qforce_xml.main)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
