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


# O + OH -> O2 + H: 17 electrons, three reactant radicals.
OXYGEN = "[O].[OH]>>[O][O].[H]"


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


WATER_MANIFEST = HERE.parents[1] / "DynamicTopo" / "datasets" / "Water" / "Water.json"


@pytest.mark.skipif(
    not WATER_MANIFEST.exists(), reason="needs the DynamicTopo checkout"
)
def test_the_water_dataset_manifest_loads():
    """DynamicTopology's own dataset manifest is a fast-forces manifest too."""
    manifest = M.load(WATER_MANIFEST)
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


def test_dynamictopology_s_defaults_load(tmp_path):
    from DynamicTopology.forcefield.params import DEFAULTS

    manifest = M.load(write(tmp_path, global_params=DEFAULTS.to_dict()))
    assert manifest.params == DEFAULTS


def test_stated_global_params_are_applied(tmp_path):
    """A dataset is fitted at its own `global_params`, not the defaults."""
    manifest = M.load(
        write(tmp_path, global_params={"taper_radius": 1.6, "switch_radius": 2.4})
    )
    assert manifest.params.taper_radius == 1.6
    assert manifest.params.switch_radius == 2.4


@pytest.mark.parametrize(
    "global_params, message",
    [
        ({"taper_raduis": 1.5}, "unknown `global_params`"),
        ({"electrostatics": "qeq"}, "electrostatics must be one of"),
        ({"exclusion_depth": -1}, "exclusion_depth must be >= 0"),
        ({"taper_width": 0.0}, "taper_width must be > 0"),
    ],
)
def test_global_params_dynamictopology_would_not_accept_are_refused(
    tmp_path, global_params, message
):
    with pytest.raises(M.ManifestError, match=message):
        M.load(write(tmp_path, global_params=global_params))


@pytest.mark.parametrize(
    "global_params, fit_config, expected",
    [
        ({}, {}, ("acks2", "acks2")),
        ({}, {"electrostatics": "fixed"}, ("pointcharge", "fixed")),
        ({"electrostatics": "pointcharge"}, {}, ("pointcharge", "fixed")),
        (
            {"electrostatics": "pointcharge"},
            {"electrostatics": "fixed"},
            ("pointcharge", "fixed"),
        ),
    ],
)
def test_the_two_electrostatics_settings_follow_each_other(
    tmp_path, global_params, fit_config, expected
):
    manifest = M.load(
        write(tmp_path, global_params=global_params, fit_config=fit_config)
    )
    assert (manifest.params.electrostatics, manifest.config.electrostatics) == expected


def test_contradictory_electrostatics_are_refused(tmp_path):
    with pytest.raises(M.ManifestError, match="have to agree"):
        M.load(
            write(
                tmp_path,
                global_params={"electrostatics": "acks2"},
                fit_config={"electrostatics": "fixed"},
            )
        )


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
        ({"molecules": [{"smiles": "[OH]", "path": "a", "spin": 2}]}, "parity"),
        ({"molecules": [{"smiles": "O", "path": "a", "spin": -2}]}, "non-negative"),
        ({"reactions": [{"smiles": OXYGEN, "path": "a", "spin": 2}]}, "parity"),
        ({"reactions": [{"smiles": OXYGEN, "path": "a", "spin_ts": 1.0}]}, "integer"),
        ({"reactions": [{"smiles": "OO>>[OH].[OH]", "path": "a", "spin_ts": 2}]},
         "fission"),
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


def _reaction_entry(tmp_path, smiles, **extra):
    frames = tmp_path / "frames.xyz"
    frames.write_bytes((HERE / "h3o-h2o-transfer.xyz").read_bytes())
    entry = {"smiles": smiles, "path": "r", "frames": "frames.xyz", **extra}
    path = write(tmp_path, reactions=[entry])
    entry = M.load(path).reactions[0]
    return entry, R.parse(entry.mapped)


def test_stored_frames_match_the_automatic_mapping(tmp_path):
    entry, reaction = _reaction_entry(tmp_path, "[OH3+].O>>O.[OH3+]")
    frames, source = M._supplied_frames(entry, reaction)
    assert source == "frames"
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


def test_frames_at_another_spin_are_refused(tmp_path):
    """The stored water frames state no spin, so they were computed at the
    closed-shell parity default -- not at the triplet this entry asks for."""
    entry, reaction = _reaction_entry(tmp_path, "[OH3+].O>>O.[OH3+]", spin=2)
    with pytest.raises(M.ManifestError, match="2S=0, not the 2"):
        M._supplied_frames(entry, reaction)


# ---------------------------------------------------------------------------
# stationary points already at the output path
# ---------------------------------------------------------------------------


class _Counting:
    """A stand-in reference calculator: Lennard-Jones, counting its uses.

    Its energies are nothing like the file's, which is the point -- a frame
    that comes back carrying them was relabelled.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, atoms=None):
        from ase.calculators.lj import LennardJones

        self.calls.append(atoms.info.get("spin") if atoms is not None else None)
        return LennardJones(sigma=1.0, epsilon=0.01, rc=6.0)


def _output_entry(tmp_path, frames=None, **extra):
    """The water transfer with its frames at the entry's own output path."""
    from ase.io import write as ase_write

    frames = frames or _read(HERE / "h3o-h2o-transfer.xyz")
    ase_write(str(tmp_path / "r.xyz"), frames, format="extxyz")
    path = write(tmp_path, reactions=[{"smiles": WATER, "path": "r", **extra}])
    entry = M.load(path).reactions[0]
    return entry, R.parse(entry.mapped)


def _read(path):
    from ase.io import read

    return read(str(path), index=":", format="extxyz")


def _permuted(frames, order):
    """`frames` with atom `k` taken from atom `order[k]`, connectivity renumbered."""
    new_index = {old: new for new, old in enumerate(order)}
    out = []
    for frame in frames:
        new = frame[order]
        new.info = dict(frame.info)
        new.info["connectivity"] = [
            [new_index[int(i)], new_index[int(j)], rest]
            for i, j, rest in frame.info["connectivity"]
        ]
        out.append(new)
    return out


def test_output_geometries_are_relabelled_with_the_reference(tmp_path):
    entry, reaction = _output_entry(tmp_path)
    calc = _Counting()
    frames, source = M._supplied_frames(entry, reaction, calc)
    assert source == "output"
    assert calc.calls == [0, 0, 0]

    stored = _read(tmp_path / "r.xyz")
    for frame, old, kind in zip(frames, stored, R.FRAME_KINDS):
        np.testing.assert_allclose(frame.positions, old.positions)
        work = frame.copy()
        work.calc = calc()
        assert frame.get_potential_energy() == pytest.approx(work.get_potential_energy())
        assert frame.get_potential_energy() != pytest.approx(old.get_potential_energy())
        assert frame.info["frame_kind"] == kind
        # the calculator reads the total charge off the atoms, and the file's
        # initial charges are all zero for an H3O+
        assert frame.get_initial_charges().sum() == pytest.approx(1.0)
    assert frames[0].info["connectivity"] == reaction.connectivity("reactant")
    assert frames[2].info["connectivity"] == reaction.connectivity("product")


def test_output_geometries_need_no_energies(tmp_path):
    bare = _read(HERE / "h3o-h2o-transfer.xyz")
    for frame in bare:
        frame.calc = None
        frame.info.pop("energy", None)
    entry, reaction = _output_entry(tmp_path, bare)
    frames, source = M._supplied_frames(entry, reaction, _Counting())
    assert source == "output"
    assert np.isfinite(frames[1].get_potential_energy())


def test_an_output_file_in_another_atom_order_is_renumbered(tmp_path):
    """Numbered as a dataset might number it: the water first, then the
    hydronium, each oxygen before its hydrogens."""
    original = _read(HERE / "h3o-h2o-transfer.xyz")
    entry, reaction = _output_entry(tmp_path, _permuted(original, [4, 5, 6, 0, 1, 2, 3]))
    frames, source = M._supplied_frames(entry, reaction, _Counting())
    assert source == "output"
    assert frames[0].info["connectivity"] == reaction.connectivity("reactant")
    assert frames[2].info["connectivity"] == reaction.connectivity("product")
    # The two oxygens and the moving proton have roles no other atom shares,
    # so they land exactly where they were; the spectator hydrogens on each
    # oxygen are interchangeable and may land either way round.
    for frame, old in zip(frames, original):
        np.testing.assert_allclose(frame.positions[[0, 1, 4]], old.positions[[0, 1, 4]])
        for group in ([2, 3], [5, 6]):
            assert sorted(map(tuple, frame.positions[group].round(6))) == sorted(
                map(tuple, old.positions[group].round(6))
            )


def test_an_output_file_of_another_reaction_is_refused(tmp_path):
    """The product has the proton back on the oxygen it started on: no
    renumbering of the atoms makes that this reaction."""
    frames = _read(HERE / "h3o-h2o-transfer.xyz")
    frames[2].info["connectivity"] = frames[0].info["connectivity"]
    entry, reaction = _output_entry(tmp_path, frames)
    with pytest.raises(M.ManifestError, match="is not this reaction"):
        M._supplied_frames(entry, reaction, _Counting())


def test_an_output_file_at_another_stated_spin_is_refused(tmp_path):
    entry, reaction = _output_entry(tmp_path, spin=2)
    with pytest.raises(M.ManifestError, match="the reaction's output file"):
        M._supplied_frames(entry, reaction, _Counting())


def test_the_cache_is_reused_while_it_holds_the_output_geometry(tmp_path):
    from ase.io import write as ase_write

    entry, reaction = _output_entry(tmp_path)
    labelled, _ = M._supplied_frames(entry, reaction, _Counting())
    cache = tmp_path / "training" / "r.xyz"
    cache.parent.mkdir(parents=True)
    ase_write(str(cache), labelled, format="extxyz")

    calc = _Counting()
    frames, source = M._supplied_frames(entry, reaction, calc)
    assert (source, calc.calls) == ("cache", [])
    assert frames[1].get_potential_energy() == pytest.approx(
        labelled[1].get_potential_energy()
    )

    # New geometries at the output path outrank the cache they no longer match.
    moved = _read(tmp_path / "r.xyz")
    for frame in moved:
        frame.positions[1] += 0.05
    ase_write(str(tmp_path / "r.xyz"), moved, format="extxyz")
    frames, source = M._supplied_frames(entry, reaction, calc)
    assert source == "output"
    assert len(calc.calls) == 3


def test_frames_the_manifest_names_outrank_the_output_file(tmp_path):
    from ase.io import write as ase_write

    entry, reaction = _reaction_entry(tmp_path, "[OH3+].O>>O.[OH3+]")
    ase_write(str(tmp_path / "r.xyz"), _read(HERE / "h3o-h2o-transfer.xyz"))
    calc = _Counting()
    _, source = M._supplied_frames(entry, reaction, calc)
    assert (source, calc.calls) == ("frames", [])


def test_a_fission_frame_the_reference_cannot_label_is_kept(tmp_path):
    """Only a fission's reactant is fitted, so an SCF that fails on the
    separated end costs that frame's energy and not the channel -- and the
    frame stays, since DynamicTopology reads the product's bonds off it."""
    from ase import Atoms
    from ase.calculators.calculator import CalculationFailed
    from ase.io import write as ase_write

    path = []
    for distance, bonds in ((0.97, [[0, 1, None]]), (1.5, [[0, 1, None]]), (4.0, [])):
        frame = Atoms("OH", positions=[[0, 0, 0], [0, 0, distance]])
        frame.info["connectivity"] = bonds
        path.append(frame)
    ase_write(str(tmp_path / "r.xyz"), path, format="extxyz")
    manifest = write(tmp_path, reactions=[{"smiles": "[O:1][H:2]>>[O:1].[H:2]", "path": "r"}])
    entry = M.load(manifest).reactions[0]
    reaction = R.parse(entry.mapped)

    from ase.calculators.calculator import Calculator

    class Unconverged(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, *args, **kwargs):
            raise CalculationFailed("SCF not converged")

    lj = _Counting()

    def failing(atoms=None):
        if atoms is not None and atoms.get_distance(0, 1) > 3.0:
            return Unconverged()
        return lj(atoms)

    frames, source = M._supplied_frames(entry, reaction, failing)
    assert source == "output"
    assert [f.calc is not None for f in frames] == [True, True, False]
    assert frames[2].info["connectivity"] == []

    # what the run leaves behind reads back as the cache
    cache = tmp_path / "training" / "r.xyz"
    cache.parent.mkdir(parents=True)
    ase_write(str(cache), frames, format="extxyz")
    ase_write(str(tmp_path / "r.xyz"), frames, format="extxyz")
    _, source = M._supplied_frames(entry, reaction, failing)
    assert source == "cache"


def test_the_plan_says_where_each_reaction_s_frames_come_from(tmp_path):
    entry, _ = _output_entry(tmp_path)
    manifest = M.load(tmp_path / "Test.json")
    assert "stationary points from its .xyz, relabelled" in manifest.plan()


# ---------------------------------------------------------------------------
# spin
# ---------------------------------------------------------------------------


def _spins(tmp_path, smiles=OXYGEN, **keys):
    return M.load(write(tmp_path, reactions=[{"smiles": smiles, "path": "r", **keys}]))


@pytest.mark.parametrize(
    "keys, expected",
    [
        ({}, (3, 3, 3)),
        ({"spin": 1}, (1, 1, 1)),
        ({"spin_ts": 1}, (3, 1, 3)),
        ({"spin": 1, "spin_p": 3}, (1, 1, 3)),
        ({"spin_r": 1, "spin_ts": 3, "spin_p": 1}, (1, 3, 1)),
    ],
)
def test_spin_sets_every_frame_and_the_per_frame_keys_override_it(
    tmp_path, keys, expected
):
    entry = _spins(tmp_path, **keys).reactions[0]
    assert tuple(entry.spins[k] for k in R.FRAME_KINDS) == expected


def test_a_fission_takes_its_reactant_spin(tmp_path):
    entry = _spins(tmp_path, "OO>>[OH].[OH]", spin_r=2).reactions[0]
    assert entry.spins["reactant"] == 2


def test_the_plan_shows_the_spins(tmp_path):
    assert "2S: reactant 1, transition 1, product 3" in _spins(
        tmp_path, spin=1, spin_p=3
    ).plan()


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


@pytest.mark.slow
def test_a_reaction_s_output_geometry_replaces_the_saddle_search(small_run, tmp_path):
    """With its cached stationary points gone, the reaction is fitted from the
    `.xyz` it wrote: three single points at those geometries, no Sella."""
    import shutil

    from tblite.ase import TBLite

    source, path, _ = small_run
    root = tmp_path / "copy"
    shutil.copytree(source, root)
    (root / "training" / "rxn" / "sn2.xyz").unlink()
    before = _read(root / "rxn" / "sn2.xyz")

    calls = []

    def counting(atoms=None):
        calls.append(len(atoms) if atoms is not None else 0)
        return TBLite(method="GFN2-xTB", verbosity=0)

    manifest = root / path.name
    outcomes = M.run(manifest, calc_factory=counting, log=lambda *_: None)
    assert all(o.ok for o in outcomes), [o.detail for o in outcomes]
    assert calls.count(6) == 3  # the lone Cl- is labelled on every run too
    assert set(calls) == {1, 6}

    after = _read(root / "rxn" / "sn2.xyz")
    for old, new in zip(before, after, strict=True):
        np.testing.assert_allclose(new.positions, old.positions, atol=1e-7)
        assert new.get_potential_energy() == pytest.approx(
            old.get_potential_energy(), abs=1e-5
        )
    assert (root / "training" / "rxn" / "sn2.xyz").exists()

    # and the rerun after that finds it cached
    calls.clear()
    M.run(manifest, calc_factory=counting, log=lambda *_: None)
    assert set(calls) == {1}
