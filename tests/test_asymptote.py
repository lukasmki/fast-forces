"""The per-bond asymptote: where a molecule's stretched limit sits.

A bonded diabat has to level off *above* the fragments it dissociates into, or
it never crosses theirs and no coupling can hand over.  The molecule fit puts
it `bond_asymptote` above the fragments' own reference energies, read off two
single points per bond class (`sampling.fragment_frames`).  The fragment rule
is tested without a calculator; the limit itself against GFN2-xTB.
"""

import shutil

import numpy as np
import pytest
from ase import Atoms
from DynamicTopology.forcefield.evaluate import evaluate as dt_evaluate
from DynamicTopology.forcefield.params import active

import fastforces as ff
from fastforces import io, sampling
from fastforces.topology import enumerate_terms, perceive


# ---------------------------------------------------------------------------
# which fragments, at which charge and spin
# ---------------------------------------------------------------------------


def _cuts(smiles):
    """`{(formula, formula): (2S, 2S)}` over one bond of each class."""
    atoms = ff.build(smiles)
    topology = enumerate_terms(atoms, perceive(atoms))
    out = {}
    for bond in sampling.bridge_bonds(topology).values():
        sides = sampling._cut_fragments(smiles, atoms, topology.graph, bond)
        formulas = tuple(atoms[indices].get_chemical_formula() for indices, _ in sides)
        out[formulas] = tuple(spin for _, spin in sides)
    return out


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("[H][H]", {("H", "H"): (1, 1)}),
        # both O atoms come out triplet, the ground state the manifest fits
        ("[O][O]", {("O", "O"): (2, 2)}),
        ("[OH]", {("O", "H"): (2, 1)}),
        ("O", {("HO", "H"): (1, 1)}),
        ("[O]O", {("HO", "O"): (1, 2), ("O2", "H"): (2, 1)}),
        ("OO", {("HO", "HO"): (1, 1), ("HO2", "H"): (1, 1)}),
        # a double bond releases two electrons to each side
        ("C=C", {("CH2", "CH2"): (2, 2), ("C2H3", "H"): (1, 1)}),
    ],
)
def test_a_cut_leaves_each_side_its_radicals_plus_the_bond_order(smiles, expected):
    found = {tuple(sorted(k)): tuple(sorted(v)) for k, v in _cuts(smiles).items()}
    assert found == {tuple(sorted(k)): tuple(sorted(v)) for k, v in expected.items()}


def test_a_ring_bond_has_no_fragments():
    atoms = ff.build("C1CC1")
    topology = enumerate_terms(atoms, perceive(atoms))
    bridges = sampling.bridge_bonds(topology)
    # every representative is a C-H, and the C-C class alone has none
    assert all(
        sorted(atoms[list(b)].get_chemical_symbols()) == ["C", "H"]
        for b in bridges.values()
    )
    assert len(bridges) == topology.n_classes("bond") - 1


def test_fragments_keep_their_atoms_formal_charges():
    """Homolysis moves no charge: OH- is O- and H, not O and H-."""

    def stub(atoms=None):
        from ase.calculators.lj import LennardJones

        return LennardJones()

    atoms = ff.build("[OH-]")
    atoms.info["smiles"] = "[OH-]"
    frames = sampling.fragment_frames(atoms, stub)
    by_formula = {f.get_chemical_formula(): f for f in frames}
    assert by_formula["O"].get_initial_charges().sum() == pytest.approx(-1)
    assert by_formula["H"].get_initial_charges().sum() == pytest.approx(0)
    assert (by_formula["O"].info["spin"], by_formula["H"].info["spin"]) == (1, 1)
    for frame in frames:
        assert frame.info["frame_kind"] == "fragment"
        assert sorted(np.atleast_1d(frame.info["fragment_bond"])) == [0, 1]


def test_no_smiles_means_no_fragments():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    assert sampling.fragment_frames(atoms, lambda a=None: None) == []


# ---------------------------------------------------------------------------
# the limit, against GFN2-xTB
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def h2_fit(tmp_path_factory, tblite_factory):
    path = tmp_path_factory.mktemp("h2") / "training.xyz"
    params = ff.parameterize(
        ff.build("[H][H]"),
        tblite_factory,
        config=ff.FitConfig(n_mode_frames=20, n_conformers=0),
        training_set=str(path),
    )
    return params, str(path)


@pytest.mark.slow
def test_a_diatomic_levels_off_bond_asymptote_above_its_fragments(h2_fit):
    """The whole point: pulled apart, the bonded state ends exactly
    `bond_asymptote` above the two atoms' reference energies -- not wherever
    the near-equilibrium fit happened to leave `D`."""
    params, path = h2_fit
    fragments = io.read_training_set(path).of_kind("fragment")
    assert len(fragments) == 2
    apart = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1000.0]])
    limit = dt_evaluate(apart, params.to_terms()).energy
    target = sum(f.get_potential_energy() for f in fragments) + active().bond_asymptote
    assert limit == pytest.approx(target, abs=1e-5)
    assert "asymptote_h" in params.report


@pytest.mark.slow
def test_the_fragments_are_not_fitted_to(h2_fit):
    params, path = h2_fit
    data = io.read_training_set(path)
    assert params.report["frames"] == len(data.frames) - len(
        data.of_kind("fragment", "atom")
    )
    assert "fragment" not in params.report


@pytest.mark.slow
def test_a_diatomic_needs_no_asymptote_beyond_the_default(h2_fit):
    """A diatomic's fragments are its free atoms, so once its depth carries the
    atomization energy the stretched limit is already where `bond_asymptote`
    puts it: the two solves agree, and `h` comes out at the default."""
    params, _ = h2_fit
    assert params.terms["bond"]["kwargs"]["h"] == pytest.approx(
        active().bond_asymptote, abs=1e-6
    )


@pytest.mark.slow
def test_every_bond_class_of_h2o2_gets_its_own_asymptote(h2o2_fit):
    _, params, path = h2o2_fit
    assert len(io.read_training_set(path).of_kind("fragment")) == 4  # O-H, O-O
    h = params.terms["bond"]["kwargs"]["h"]
    assert len(set(np.round(h, 8))) == 2
    assert not np.allclose(h, active().bond_asymptote)


@pytest.mark.slow
def test_an_older_training_set_is_brought_up_to_date(h2_fit, tmp_path, tblite_factory):
    """A set written before the fragments existed gets them, once."""
    _, path = h2_fit
    old = tmp_path / "old.xyz"
    shutil.copy(path, old)
    data = io.read_training_set(str(old))
    kept = [f for f in data.frames if f.info.get("frame_kind") != "fragment"]
    io.write_training_set(str(old), kept, meta=data.meta)
    assert "h" not in ff.fit_from_file(str(old)).terms["bond"]["kwargs"]

    assert ff.add_fragment_frames(str(old), tblite_factory) == 2
    assert ff.add_fragment_frames(str(old), tblite_factory) == 0
    assert "h" in ff.fit_from_file(str(old)).terms["bond"]["kwargs"]
