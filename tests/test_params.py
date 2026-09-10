"""Parameter container, the DynamicTopology format, and the OpenMM export."""

import json

import numpy as np
import pytest
from ase.io import read

from fastforces.export import units
from fastforces.params import Parameters
from fastforces.topology import enumerate_terms

EXAMPLE_JSONL = "examples/h2o2_dynamictopology_format.jsonl"


def example_rows():
    with open(EXAMPLE_JSONL) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def example_params():
    atoms = read("examples/h2o2.xyz")
    params = Parameters.from_jsonl(EXAMPLE_JSONL, numbers=atoms.get_atomic_numbers())
    params.exclusions = enumerate_terms(atoms).exclusions
    return atoms, params


def test_jsonl_round_trip_reproduces_the_example():
    rows = example_rows()
    rebuilt = Parameters.from_rows(rows, numbers=[8, 8, 1, 1]).to_rows()
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
    assert units.from_openmm("bond", "r0", 0.142088398) == pytest.approx(1.42088398)
    assert units.from_openmm("bond", "D", 366.0) == pytest.approx(3.7933, abs=1e-4)
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
