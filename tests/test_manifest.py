"""Manifests: one file naming a dataset's molecules, reactions and fit settings.

Most of what can go wrong with a manifest is found before any fitting, by
design, so most of these tests need no calculator: the automatic atom mapping,
the validation `load` does, and the check that supplied stationary points match
the reaction they are filed under.  The slow tests run a small manifest end to
end against GFN2-xTB.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from fastforces import manifest as M
from fastforces import reaction as R

HERE = Path(__file__).parent

WATER = (
    "[O+:1]([H:2])([H:3])[H:4].[O:5]([H:6])[H:7]"
    ">>[O:1]([H:3])[H:4].[O+:5]([H:2])([H:6])[H:7]"
)


def write(tmp_path, **sections) -> Path:
    body = {"name": "Test", "molecules": [], "reactions": []}
    body.update(sections)
    path = tmp_path / "Test.json"
    path.write_text(json.dumps(body))
    return path


# ---------------------------------------------------------------------------
# automatic atom mapping
# ---------------------------------------------------------------------------


def test_an_unmapped_transfer_maps_onto_the_hand_written_one():
    """The water channel written plainly is the channel `test_reaction` pins,
    bond for bond -- including which proton moves."""
    auto = R.parse(R.map_atoms("[OH3+].O>>O.[OH3+]"))
    hand = R.parse(WATER)
    assert list(auto.numbers) == list(hand.numbers)
    assert auto.reactant_bonds == hand.reactant_bonds
    assert auto.product_bonds == hand.product_bonds
    assert auto.channel() == ("transfer", (0, 1, 4))


@pytest.mark.parametrize(
    "smiles, channel",
    [
        ("O.O>>[OH-].[OH3+]", ("transfer", (0, 1, 3))),
        ("O.[OH-]>>[OH-].O", ("transfer", (0, 1, 3))),
        ("[Cl-].CCl>>ClC.[Cl-]", ("transfer", (5, 1, 0))),
        ("[H+].O>>[OH3+]", ("fission", ((0, 1), (1, 2, 3)))),
    ],
)
def test_the_lowest_numbered_hydrogen_is_the_one_that_moves(smiles, channel):
    """The reference datasets' convention, which is what lets a manifest's
    unmapped SMILES be checked against their stored frames."""
    assert R.parse(R.map_atoms(smiles)).channel() == channel


def test_a_mapped_smiles_is_left_alone():
    assert R.map_atoms(WATER) == WATER


@pytest.mark.parametrize(
    "smiles, message",
    [
        ("[OH3+].O>>[OH3+].O", "no bond broken or formed"),
        ("[CH3:1]Cl>>C.Cl", "maps some atoms and not others"),
        ("[H][H]>>[H].[H]", "hydrogen-hydrogen bond"),
        ("CCO>>CC.O", "do not contain the same atoms"),
        ("O.O", "expected one '>>'"),
    ],
)
def test_an_unmappable_reaction_is_refused(smiles, message):
    with pytest.raises(R.ReactionError, match=message):
        R.map_atoms(smiles)


def test_a_manifest_molecule_has_the_key_a_reaction_gives_it():
    """What lets a reaction reuse a molecule fit instead of refitting it."""
    reaction = R.parse(WATER)
    keys = {f.key for f in reaction.reactant_fragments + reaction.product_fragments}
    assert keys == {M.molecule_key("[OH3+]"), M.molecule_key("O")}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def test_the_example_manifest_loads():
    manifest = M.load(HERE.parent / "examples" / "proton-transfer" / "Water.json")
    assert len(manifest.molecules) == 5
    assert len(manifest.reactions) == 3
    assert all(e.mapped for e in manifest.reactions)
    assert manifest.calculator["name"] == "pyscf"


def test_paths_are_relative_to_the_manifest(tmp_path):
    path = write(
        tmp_path,
        molecules=[{"id": 1, "smiles": "O", "path": "molecules/h2o"}],
        fit_config={"workdir": "scratch"},
    )
    entry = M.load(path).molecules[0]
    assert entry.output == tmp_path / "molecules" / "h2o"
    assert entry.training == tmp_path / "scratch" / "molecules" / "h2o"


def test_fit_config_reaches_the_fit(tmp_path):
    path = write(tmp_path, fit_config={"n_mode_frames": 7, "temperature": 300.0})
    config = M.load(path).config
    assert (config.n_mode_frames, config.temperature) == (7, 300.0)


def test_the_package_defaults_pass_the_global_check(tmp_path):
    M.load(write(tmp_path, global_params=M.package_globals()))


@pytest.mark.parametrize(
    "global_params, message",
    [
        ({"taper_radius": 1.6}, "taper_radius=1.6"),
        ({"switch_radius": 2.2}, "switch_radius=2.2"),  # Angstrom, not nm
        ({"exclude_coulomb": False}, "exclude_coulomb"),
        ({"taper_raduis": 1.5}, "unknown global_params"),
    ],
)
def test_global_params_the_package_would_not_honour_are_refused(
    tmp_path, global_params, message
):
    with pytest.raises(M.ManifestError, match=message):
        M.load(write(tmp_path, global_params=global_params))


@pytest.mark.parametrize(
    "sections, message",
    [
        ({"fit_config": {"n_modes": 3}}, "unknown fit_config keys"),
        ({"fit_config": {"calculator": {"name": "orca"}}}, "unknown calculator"),
        ({"molecules": [{"smiles": "O", "path": "a", "charge": 0}]}, "unknown keys"),
        ({"molecules": [{"smiles": "O"}]}, "no 'path'"),
        (
            {
                "molecules": [
                    {"id": 1, "smiles": "O", "path": "a"},
                    {"id": 2, "smiles": "C", "path": "a"},
                ]
            },
            "same path",
        ),
        (
            {
                "molecules": [
                    {"id": 1, "smiles": "O", "path": "a"},
                    {"id": 1, "smiles": "C", "path": "b"},
                ]
            },
            "share id",
        ),
        ({"molecules": [{"smiles": "O.O", "path": "a"}]}, "disconnected"),
        ({"reactions": [{"smiles": "O>>O", "path": "a"}]}, "no bond broken"),
    ],
)
def test_a_malformed_manifest_is_refused_before_anything_runs(
    tmp_path, sections, message
):
    with pytest.raises(M.ManifestError, match=message):
        M.load(write(tmp_path, **sections))


# ---------------------------------------------------------------------------
# supplied stationary points
# ---------------------------------------------------------------------------


def _reaction_entry(tmp_path, smiles):
    frames = tmp_path / "frames.xyz"
    frames.write_bytes((HERE / "h3o-h2o-transfer.xyz").read_bytes())
    path = write(
        tmp_path, reactions=[{"smiles": smiles, "path": "r", "frames": "frames.xyz"}]
    )
    entry = M.load(path).reactions[0]
    return entry, R.parse(entry.mapped)


def test_stored_frames_match_the_automatic_mapping(tmp_path):
    entry, reaction = _reaction_entry(tmp_path, "[OH3+].O>>O.[OH3+]")
    frames = M._supplied_frames(entry, reaction)
    assert len(frames) == 3
    assert np.isfinite(frames[1].get_potential_energy())


def test_frames_bonded_differently_from_the_reaction_are_refused(tmp_path):
    """Same atoms, same order, but the third hydrogen moves instead of the
    first -- a mapping the file does not describe."""
    moved = (
        "[O+:1]([H:2])([H:3])[H:4].[O:5]([H:6])[H:7]"
        ">>[O:1]([H:2])[H:3].[O+:5]([H:4])([H:6])[H:7]"
    )
    entry, reaction = _reaction_entry(tmp_path, moved)
    with pytest.raises(M.ManifestError, match="bonded differently"):
        M._supplied_frames(entry, reaction)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """Three molecules -- a lone atom, a diatomic and a triatomic -- and a
    reaction whose fragments are all among them."""
    pytest.importorskip("tblite.ase")
    root = tmp_path_factory.mktemp("manifest")
    path = write(
        root,
        molecules=[
            {"id": 1, "smiles": "[Cl-]", "path": "molecules/cl"},
            {"id": 2, "smiles": "[OH-]", "path": "molecules/oh"},
            {"id": 3, "smiles": "CCl", "path": "molecules/ch3cl"},
        ],
        reactions=[{"id": 1, "smiles": "[Cl-].CCl>>ClC.[Cl-]", "path": "rxn/sn2"}],
        fit_config={
            "calculator": {"name": "tblite", "method": "GFN2-xTB"},
            "n_mode_frames": 10,
            "n_conformers": 0,
        },
    )
    outcomes = M.run(path, log=lambda *_: None)
    return root, path, outcomes


@pytest.mark.slow
def test_every_entry_is_fitted_and_written(small_run):
    root, _, outcomes = small_run
    assert [o.ok for o in outcomes] == [True] * 4, [o.detail for o in outcomes]
    for name in ("cl", "oh", "ch3cl"):
        assert (root / "molecules" / f"{name}.jsonl").exists()
        assert (root / "molecules" / f"{name}.xyz").exists()
    for suffix in (".xyz", ".jsonl", "-reactant.jsonl", "-product.jsonl"):
        assert (root / "rxn" / f"sn2{suffix}").exists()


@pytest.mark.slow
def test_the_reaction_reuses_the_molecule_fits(small_run):
    root, _, _ = small_run
    assert not (root / "training" / "fragments").exists()


@pytest.mark.slow
def test_a_rerun_reproduces_the_outputs_from_the_training_sets(small_run):
    """Only the lone atom -- which keeps no training set -- is recalculated.

    To file precision rather than bit for bit: the first run fits the coupling
    to the stationary points as Sella left them, every later one to the same
    points read back from extxyz, which stores positions to 1e-8 A.
    """
    root, path, _ = small_run
    from tblite.ase import TBLite

    def rows():
        return {
            p: [json.loads(line) for line in p.read_text().splitlines()]
            for p in sorted(root.rglob("*.jsonl"))
        }

    before = rows()
    calls = []

    def counting(atoms=None):
        calls.append(len(atoms) if atoms is not None else 0)
        return TBLite(method="GFN2-xTB", verbosity=0)

    outcomes = M.run(path, calc_factory=counting, log=lambda *_: None)
    assert all(o.ok for o in outcomes)
    assert set(calls) == {1}

    after = rows()
    assert after.keys() == before.keys()
    for p, old in before.items():
        new = after[p]
        assert [(r["type"], r["atoms"]) for r in new] == [
            (r["type"], r["atoms"]) for r in old
        ], p
        for r_new, r_old in zip(new, old, strict=True):
            for key, value in r_old["kwargs"].items():
                assert r_new["kwargs"][key] == pytest.approx(value, rel=1e-6), (p, key)
