"""Starting a fit from an existing force field.

These are the mapping tests -- do the supplied parameters land on the right
terms -- and need no reference calculator.  What a starting point does to the
result once it has landed is covered by the slow tests in `test_fit.py`.
"""

import numpy as np
import pytest
from ase import Atoms

from fastforces.fit import GEOMETRIC, _seed_from, as_parameters
from fastforces.params import Parameters
from fastforces.topology import Topology, enumerate_terms

# Geometry is irrelevant to the mapping -- the connectivity is what drives term
# enumeration -- but a realistic one keeps the fixture readable.
H2O2 = Atoms(
    "OOHH",
    positions=[
        [0.0, 0.0, 0.7104],
        [0.0, 0.0, -0.7104],
        [0.8419, 0.3455, 0.8926],
        [-0.3455, -0.8419, -0.8926],
    ],
)
H2O2.info["connectivity"] = [[0, 1, 1.0], [0, 2, 1.0], [1, 3, 1.0]]


@pytest.fixture
def topology() -> Topology:
    return enumerate_terms(H2O2)


def synthetic(topology: Topology, offset: float = 0.0) -> Parameters:
    """A force field over `topology` whose values encode their own class.

    Class `c` of parameter `name` gets a value that appears nowhere else, so a
    mis-mapping shows up as a wrong number rather than a near-miss.
    """
    terms = {}
    for term, atoms in topology.atoms.items():
        classes = topology.classes[term]
        kwargs = {n: v.copy() for n, v in topology.fixed.get(term, {}).items()}
        names = list(GEOMETRIC.get(term, {})) + ["k"]
        if term == "bond":
            names = ["r0", "k", "D"]
        for i, name in enumerate(names):
            kwargs[name] = 1.0 + offset + 0.25 * i + 0.03 * classes
        terms[term] = {"atoms": atoms.copy(), "kwargs": kwargs}
    return Parameters(numbers=topology.numbers, terms=terms)


def test_every_class_is_seeded_from_a_matching_field(topology):
    initial = synthetic(topology)
    seed = _seed_from(topology, initial, len(H2O2))

    for name, expected in (("r0", 1.0), ("k", 1.25), ("D", 1.5)):
        classes = np.arange(topology.n_classes("bond"))
        assert seed.bond[name] == pytest.approx(expected + 0.03 * classes)

    for (term, c), value in seed.k.items():
        n_geometric = len(GEOMETRIC[term])
        assert value == pytest.approx(1.0 + 0.25 * n_geometric + 0.03 * c)

    assert seed.n_seeded == sum(topology.n_classes(t) for t in topology.atoms)


def test_geometric_values_land_on_every_row_of_their_class(topology):
    seed = _seed_from(topology, synthetic(topology), len(H2O2))
    theta0 = seed.values["angle"]["theta0"]
    assert len(theta0) == len(topology.atoms["angle"])
    assert theta0 == pytest.approx(1.0 + 0.03 * topology.classes["angle"])


def test_bond_r0_seeds_the_nonlinear_block_not_the_fixed_values(topology):
    """`values["bond"]["r0"]` sets the refinement's bounds, so it stays measured."""
    seed = _seed_from(topology, synthetic(topology), len(H2O2))
    assert "bond" not in seed.values
    assert "r0" in seed.bond


def test_dihedral_rows_are_keyed_by_periodicity(topology):
    """Four rows share the same four atoms, one per `n`; they must not collide."""
    initial = synthetic(topology)
    n_values = initial.terms["periodicdihedral"]["kwargs"]["n"]
    assert len(set(n_values.tolist())) > 1

    # Scramble one periodicity's force constants; only that class may move.
    initial.terms["periodicdihedral"]["kwargs"]["k"][n_values == 1] += 5.0
    seed = _seed_from(topology, initial, len(H2O2))
    moved = [c for (t, c), v in seed.k.items() if t == "periodicdihedral" and v > 5.0]
    assert len(moved) == int(np.sum(n_values == 1))


def test_a_partial_field_leaves_the_rest_unseeded(topology):
    """Terms the initial field says nothing about come back as `nan`."""
    initial = synthetic(topology)
    for term in ("angle", "bondbond"):
        del initial.terms[term]
    keep = topology.classes["bond"] == 0
    block = initial.terms["bond"]
    block["atoms"] = block["atoms"][keep]
    block["kwargs"] = {n: v[keep] for n, v in block["kwargs"].items()}

    seed = _seed_from(topology, initial, len(H2O2))
    assert not any(term == "angle" for term, _ in seed.k)
    assert not np.isnan(seed.bond["r0"][0])
    assert np.all(np.isnan(seed.bond["r0"][1:]))


def test_nonbonded_terms_are_carried_through(topology):
    initial = synthetic(topology)
    initial.terms["lennardjones"] = {
        "atoms": np.arange(len(H2O2))[:, None],
        "kwargs": {"sigma": np.full(len(H2O2), 3.0), "eps": np.full(len(H2O2), 0.01)},
    }
    seed = _seed_from(topology, initial, len(H2O2))
    assert seed.nonbonded["lennardjones"]["kwargs"]["eps"] == pytest.approx(0.01)


def test_a_field_for_another_molecule_is_rejected(topology):
    initial = synthetic(topology)
    initial.terms["bond"]["atoms"] = initial.terms["bond"]["atoms"] + 10
    with pytest.raises(ValueError, match="different molecule"):
        _seed_from(topology, initial, len(H2O2))


def test_a_field_in_another_atom_ordering_is_rejected(topology):
    """Silently seeding nothing would look like the starting point was ignored."""
    initial = synthetic(topology)
    shifted = (initial.terms["angle"]["atoms"] + 1) % len(H2O2)
    initial.terms["angle"]["atoms"] = shifted
    with pytest.raises(ValueError, match="same atom ordering"):
        _seed_from(topology, initial, len(H2O2))


def test_a_supplied_reference_offset_is_dropped(topology):
    """`E0` is derived from the converged residual, so it has no starting point.

    It is not a fitted parameter -- every energy residual in the fit is
    mean-centered -- so carrying a supplied one into the fit would be carrying
    it into a solve that has no place to put it.
    """
    initial = synthetic(topology)
    initial.terms["reference"] = {
        "atoms": np.zeros((1, 1), dtype=int),
        "kwargs": {"E0": np.array([-12.5])},
    }
    seed = _seed_from(topology, initial, len(H2O2))
    assert not hasattr(seed, "e0")


def test_as_parameters_accepts_the_three_input_forms(topology, tmp_path):
    initial = synthetic(topology)
    assert as_parameters(None) is None
    assert as_parameters(initial) is initial

    from_dict = as_parameters(initial.terms)
    assert set(from_dict.terms) == set(initial.terms)

    path = tmp_path / "ff.jsonl"
    initial.to_jsonl(str(path))
    from_file = as_parameters(str(path))
    assert np.allclose(
        from_file.terms["bond"]["kwargs"]["r0"], initial.terms["bond"]["kwargs"]["r0"]
    )

    with pytest.raises(TypeError):
        as_parameters(42)
