"""Import q-force's OpenMM-style `<Forces>` XML as DynamicTopology term rows.

    fast-forces import-qforce -i qforce_xml_dir/ -o molecules/

q-force writes a molecule's force field as OpenMM `Custom*Force` blocks with
named per-term parameters.  This reads them into the `.jsonl` row format, in
the units q-force states them in -- nm and kJ/mol, which is what a `.jsonl`
stores -- so the rows are written without conversion.  Forces whose parameters
are unnamed are dropped (q-force's own `Coulomb`, among them), and the 12-6 comes
out as q-force's `A`/`B` rather than the `sigma`/`eps` the datasets carry, so an
imported file is a starting point for `refine` and not a finished template.

Formerly DynamicTopology's `io/xml.py` and `scripts/convert.py`.
"""

from argparse import ArgumentParser
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

from DynamicTopology.core.types import Term
from DynamicTopology.io.json import write_jsonl


def read_xml(path: str | Path) -> list[Term]:
    if isinstance(path, str):
        path = Path(path).resolve()
    with open(path) as fp:
        return read_xmls(fp.read())


def read_xmls(data: str) -> list[Term]:
    terms: list[Term] = []

    root = ET.fromstring(data)
    forces = root.find("Forces")
    if forces is None:
        return []

    for force in forces.findall("Force"):
        # get term name
        name = force.get("name")
        if name is None:
            raise ValueError("Term with no name")

        atom_params = force.find("PerParticleParameters")
        bond_params = force.find("PerBondParameters")
        angle_params = force.find("PerAngleParameters")
        torsion_params = force.find("PerTorsionParameters")
        # at most one of these is not None
        params = next(
            (
                p
                for p in (atom_params, bond_params, angle_params, torsion_params)
                if p is not None
            ),
            None,
        )

        parameter_map: dict[str, str] = {}
        if params is not None:
            params: Element[str]
            # map parameter indices to named parameters
            for i, param in enumerate(params.findall("Parameter")):
                param_name = param.get("name")
                if param_name is None:
                    # skip if not named
                    continue
                parameter_map[f"param{i + 1}"] = param_name

        # get term lines
        atoms = force.find("Particles")
        bonds = force.find("Bonds")
        angles = force.find("Angles")
        torsions = force.find("Torsions")
        # at most one of these is not None; skip a force that lists no terms
        dofs: Element[str] | None = next(
            (d for d in (atoms, bonds, angles, torsions) if d is not None), None
        )
        if dofs is None:
            continue

        # loop over children
        for position, dof in enumerate(dofs):
            attrib: dict[str, str] = dof.attrib
            keys: list[str] = list(attrib.keys())
            args: dict[str, float] = {
                k: float(attrib.pop(k)) for k in keys if k.startswith("param")
            }

            if params is not None:
                for k in parameter_map.keys():
                    args[parameter_map[k]] = args.pop(k)
                # A <Bond>/<Angle>/<Torsion> names its atoms explicitly ("p1",
                # "p2", ...), but a <Particle> does not: its index is its
                # position in the list, and every attribute it carries is a
                # parameter.  Reading the indices off `attrib` therefore left
                # every per-particle force with an empty `atoms` dict, which is
                # how q-force's Lennard-Jones parameters were parsed for years
                # without ever being attachable to an atom.
                if dofs.tag == "Particles":
                    idx: dict[str, int] = {"p0": position}
                else:
                    idx = {k: int(v) for k, v in attrib.items()}
                term: Term = {"type": name.lower(), "atoms": idx, "kwargs": args}
                terms.append(term)
    return terms


def convert(input_path: Path, output_path: Path) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()

    if not output_path.exists() and output_path.suffix == "":
        output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        if not output_path.is_dir():
            output_path = output_path.parent

        for path in input_path.iterdir():
            if path.is_dir() or path.suffix != ".xml":
                continue
            terms = read_xml(path)
            print(f"writing to {output_path / path.name}")
            write_jsonl(
                (output_path / path.name).with_suffix(".jsonl"), terms, convert=False
            )
    else:
        print(f"reading from {input_path}")
        terms = read_xml(input_path)
        if output_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"writing to {output_path / input_path.name}")
            write_jsonl(
                (output_path / input_path.name).with_suffix(".jsonl"),
                terms,
                convert=False,
            )
        else:
            print(f"writing to {output_path}")
            write_jsonl((output_path).with_suffix(".jsonl"), terms, convert=False)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(prog="fast-forces import-qforce", description=__doc__)
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)
    convert(args.input, args.output)
    return 0
