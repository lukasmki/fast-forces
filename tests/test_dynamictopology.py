"""A DynamicTopology dataset scores the same through fast-forces as through itself.

fast-forces fits parameters for DynamicTopology's force field and evaluates them
with it, so every template of every shipped dataset, read into `Parameters` and
evaluated by `FastForces`, has to reproduce DynamicTopology's `System` on that
lone molecule -- open and periodic, energy, forces and stress.  What this pins
is the wrapper: the term-list round trip, the derived exclusions, the reference
shift, the per-bond asymptote, and the electrostatics `evaluate.surface`
selects from the block a field carries.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from fastforces.calculator import FastForces
from fastforces.params import Parameters, terms_to_blocks

DATASETS = Path(__file__).resolve().parents[2] / "DynamicTopology" / "datasets"
MANIFESTS = [
    DATASETS / "HCombustion" / "HCombustion.json",
    DATASETS / "Water" / "Water.json",
    DATASETS / "Water-fixed-pc" / "Water.json",
]

pytestmark = pytest.mark.skipif(
    not DATASETS.exists(), reason="needs the DynamicTopology checkout"
)


def _params(manifest):
    from DynamicTopology.forcefield.params import ForceFieldParams

    return ForceFieldParams.from_dict(json.loads(manifest.read_text())["global_params"])


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda m: m.parent.name)
@pytest.mark.parametrize("periodic", [False, True], ids=["open", "periodic"])
def test_every_template_scores_as_dynamictopology_scores_it(manifest, periodic):
    from DynamicTopology.core import ReactionSet, Topology
    from DynamicTopology.forcefield.params import use
    from DynamicTopology.system import System

    with use(_params(manifest)):
        reaction_set = ReactionSet(manifest)
        for template in reaction_set.data.molecules.values():
            atoms = template.atoms.copy()
            atoms.calc = None
            atoms.set_cell([14.0, 14.5, 15.0])
            atoms.center()
            atoms.pbc = periodic

            stated = [t for t in template.terms if not t["type"].endswith("exclusion")]
            params = Parameters(numbers=atoms.numbers, terms=terms_to_blocks(stated))
            work = atoms.copy()
            work.calc = FastForces(work, params)

            expected = System(
                atoms,
                Topology.from_terms(template.terms, atoms),
                reaction_set,
                evb={"max_states": 1},
            ).calculate()
            name = atoms.get_chemical_formula()
            assert work.get_potential_energy() == pytest.approx(
                expected["energy"], abs=1e-10
            ), name
            np.testing.assert_allclose(
                work.get_forces(), expected["forces"], atol=1e-10, err_msg=name
            )
            # `FastForces` publishes the symmetrized virial over the volume.
            np.testing.assert_allclose(
                work.get_stress(voigt=False) * work.get_volume(),
                0.5 * (expected["virial"] + expected["virial"].T),
                atol=1e-9,
                err_msg=name,
            )


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda m: m.parent.name)
def test_every_shipped_file_reads_and_writes_back(manifest, tmp_path):
    """`Parameters` round-trips DynamicTopology's own `.jsonl` files."""
    spec = json.loads(manifest.read_text())
    for entry in spec["molecules"]:
        path = manifest.parent / f"{entry['path']}.jsonl"
        params = Parameters.from_jsonl(str(path))
        out = tmp_path / path.name
        params.to_jsonl(str(out))
        again = Parameters.from_jsonl(str(out))
        for term, block in params.terms.items():
            for key, value in block["kwargs"].items():
                assert np.allclose(value, again.terms[term]["kwargs"][key], rtol=1e-14)
