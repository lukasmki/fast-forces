"""Parameter container, the DynamicTopology format, and the OpenMM export."""

import json
from pathlib import Path

import numpy as np
import pytest

import fastforces as ff
from fastforces.export import units
from fastforces.params import Parameters
from fastforces.topology import enumerate_terms

# Acetonitrile, in both export formats: the same force field written twice, so
# each file is a check on the other.  `04_export_formats.py` regenerates them.
# Paths hang off `__file__` so the suite runs from any directory.
EXAMPLE_SMILES = "CC#N"
EXAMPLE_JSONL = str(Path(__file__).parent / "acetonitrile.jsonl")
EXAMPLE_XML = Path(__file__).parent / "acetonitrile.xml"


def example_rows():
    with open(EXAMPLE_JSONL) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def example_params():
    """The example force field, on the geometry the XML was exported at.

    `ff.build` is deterministic, so this reproduces that geometry from the
    SMILES alone -- there is no geometry file to keep in step with the two
    parameter files.
    """
    atoms = ff.build(EXAMPLE_SMILES)
    params = Parameters.from_jsonl(EXAMPLE_JSONL, numbers=atoms.get_atomic_numbers())
    params.exclusions = enumerate_terms(atoms).exclusions
    return atoms, params


def test_jsonl_round_trip_reproduces_the_example():
    rows = example_rows()
    rebuilt = Parameters.from_rows(rows, numbers=[6, 6, 7, 1, 1, 1]).to_rows()
    assert len(rebuilt) == len(rows)
    for original, produced in zip(rows, rebuilt, strict=True):
        assert produced["type"] == original["type"]
        assert produced["atoms"] == original["atoms"]
        for key, value in original["kwargs"].items():
            assert produced["kwargs"][key] == pytest.approx(value, rel=1e-12, abs=1e-12)


def test_round_trip_through_a_file(tmp_path):
    _, params = example_params()
    path = tmp_path / "ff.jsonl"
    params.to_jsonl(str(path))
    reloaded = Parameters.from_jsonl(str(path), numbers=params.numbers)
    for term, block in params.terms.items():
        assert np.array_equal(block["atoms"], reloaded.terms[term]["atoms"])
        for key, value in block["kwargs"].items():
            assert np.allclose(value, reloaded.terms[term]["kwargs"][key])


def test_units_convert_to_physical_values():
    """Spot-check the table against the example's own numbers."""
    # The C-N triple bond's `r0`, as `acetonitrile.jsonl` carries it.
    assert units.from_openmm("bond", "r0", 0.10319731347263705) == pytest.approx(
        1.0319731347263705
    )
    assert units.from_openmm("bond", "D", 680.6783954015332) == pytest.approx(
        7.0548, abs=1e-4
    )
    # ACKS2 parameters are the exception: both formats carry them in eV/Angstrom
    assert units.factor("atom", "mu") == 1.0
    assert units.factor("atom", "soft_decay") == 1.0


def test_angle_k_is_exported_unscaled():
    """An angle `k` is a plain energy, so only the energy unit changes.

    `QForce.compute_angle`, `export.openmm.ANGLE` and the example files are all
    `0.5*k*(cos-cos0)^2`, so the two sides of the export agree and there is
    nothing for this table to rescale.  See `test_angle_convention`.
    """
    assert units.factor("angle", "k") == pytest.approx(units.ENERGY)


def test_openmm_export_matches_the_calculator():
    """The strongest check on the exporter: same energy, independent code path.

    Validates every unit conversion and every transcribed energy expression at
    once, through OpenMM's own evaluator rather than ours.
    """
    pytest.importorskip("openmm")
    from fastforces.calculator import evaluate
    from fastforces.export.openmm import system_energy

    atoms, params = example_params()
    positions = atoms.get_positions()
    system = params.to_openmm_system(positions=positions)

    reference, _ = evaluate(atoms, params)
    assert system_energy(system, positions) == pytest.approx(reference, rel=1e-9)


def test_openmm_xml_reloads(tmp_path):
    openmm = pytest.importorskip("openmm")
    from fastforces.export.openmm import system_energy

    atoms, params = example_params()
    path = tmp_path / "system.xml"
    params.to_openmm_xml(str(path), positions=atoms.get_positions())
    system = openmm.XmlSerializer.deserialize(path.read_text())
    direct = params.to_openmm_system(positions=atoms.get_positions())
    assert system_energy(system, atoms.get_positions()) == pytest.approx(
        system_energy(direct, atoms.get_positions()), rel=1e-9
    )


# How to read one entry out of each force class the exporter emits.  OpenMM has
# no common accessor for these, so the shapes are spelled out once here.
ENTRY_ACCESSORS = {
    "CustomBondForce": ("getNumBonds", "getBondParameters"),
    "CustomAngleForce": ("getNumAngles", "getAngleParameters"),
    "CustomTorsionForce": ("getNumTorsions", "getTorsionParameters"),
    "CustomCompoundBondForce": ("getNumBonds", "getBondParameters"),
    "CustomNonbondedForce": ("getNumParticles", "getParticleParameters"),
    "CustomExternalForce": ("getNumParticles", "getParticleParameters"),
}


def flatten(value) -> list[float]:
    """Atom indices and parameters of one entry, as a flat list of numbers."""
    if isinstance(value, (int, float)):
        return [float(value)]
    return [number for item in value for number in flatten(item)]


def force_entries(force) -> list[list[float]]:
    count, parameters = ENTRY_ACCESSORS[type(force).__name__]
    return [
        flatten(getattr(force, parameters)(i)) for i in range(getattr(force, count)())
    ]


def test_openmm_export_matches_the_reference_xml():
    """`acetonitrile.xml` pins the exporter force by force, against the jsonl.

    `test_openmm_export_matches_the_calculator` says the export agrees with the
    calculator *today*; this says it still produces the same system it produced
    when the fixture was written, so a changed energy expression or a dropped
    global parameter shows up as a diff rather than as two sides moving
    together.  Regenerate the file whenever the exporter changes on purpose.

    The `Coulomb` charges are the one thing that is not pinned: ACKS2 re-solves
    them at every geometry and the export freezes whatever `positions` was
    passed, so they belong to the geometry, not to the parameters.
    """
    openmm = pytest.importorskip("openmm")

    atoms, params = example_params()
    reference = openmm.XmlSerializer.deserialize(EXAMPLE_XML.read_text())
    produced = params.to_openmm_system(positions=atoms.get_positions())

    assert produced.getNumParticles() == reference.getNumParticles()
    for i in range(reference.getNumParticles()):
        assert produced.getParticleMass(i) == reference.getParticleMass(i)

    names = [reference.getForce(i).getName() for i in range(reference.getNumForces())]
    assert [
        produced.getForce(i).getName() for i in range(produced.getNumForces())
    ] == names

    for i, name in enumerate(names):
        want, got = reference.getForce(i), produced.getForce(i)
        assert got.getEnergyFunction() == want.getEnergyFunction(), name
        assert _globals(got) == pytest.approx(_globals(want)), name
        if name == "Coulomb":
            continue
        for wanted, produced_entry in zip(
            force_entries(want), force_entries(got), strict=True
        ):
            assert produced_entry == pytest.approx(wanted, rel=1e-12, abs=1e-12), name


def _globals(force) -> dict[str, float]:
    return {
        force.getGlobalParameterName(i): force.getGlobalParameterDefaultValue(i)
        for i in range(force.getNumGlobalParameters())
    }


def test_calculator_gradient_on_the_example():
    """Analytic forces of the assembled calculator, end to end."""
    from fastforces.calculator import FastForces

    atoms, params = example_params()
    atoms.calc = FastForces(atoms, params)
    forces = atoms.get_forces()

    step = 1e-5
    numeric = np.zeros_like(forces)
    for i in range(len(atoms)):
        for axis in range(3):
            energies = []
            for sign in (+1, -1):
                probe = atoms.copy()
                probe.calc = FastForces(probe, params)
                probe.positions[i, axis] += sign * step
                energies.append(probe.get_potential_energy())
            numeric[i, axis] = -(energies[0] - energies[1]) / (2 * step)
    assert np.allclose(forces, numeric, atol=1e-5)
