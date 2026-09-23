"""Parsing a reaction SMILES, and the three index spaces it produces.

The whole module is about atom ordering, so most of these tests are about
ordering: the combined index space both sides are read into, the canonical order
two occurrences of one molecule agree on, and the order `build` happens to use
for the fit.  A force field scattered onto the wrong one of the three is a
silent error -- every array is the right shape -- which is why they are pinned
here rather than left to the end-to-end test.
"""

import numpy as np
import pytest

from fastforces import elements
from fastforces import reaction as R
from fastforces.params import Parameters

# The reference channel: a Grotthuss proton transfer in the water dimer.  The
# index order this parses into is the one `tests/h3o-h2o-transfer.xyz` stores,
# which is what makes the two comparable.
WATER = (
    "[O+:1]([H:2])([H:3])[H:4].[O:5]([H:6])[H:7]"
    ">>[O:1]([H:3])[H:4].[O+:5]([H:2])([H:6])[H:7]"
)
SN2 = (
    "[Cl-:1].[C:2]([H:3])([H:4])([H:5])[Cl:6]>>[Cl:1][C:2]([H:3])([H:4])([H:5]).[Cl-:6]"
)
FISSION = "[H:1][H:2]>>[H:1].[H:2]"


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_parses_into_map_order():
    reaction = R.parse(WATER)
    assert list(reaction.numbers) == [8, 1, 1, 1, 8, 1, 1]
    assert reaction.charge == 1


def test_both_sides_share_one_index_space():
    """The point of map order: the two bond sets are directly comparable, and
    their difference is the reaction."""
    reaction = R.parse(WATER)
    assert reaction.connectivity("reactant") == [
        [0, 1, 1.0],
        [0, 2, 1.0],
        [0, 3, 1.0],
        [4, 5, 1.0],
        [4, 6, 1.0],
    ]
    assert reaction.connectivity("product") == [
        [0, 2, 1.0],
        [0, 3, 1.0],
        [1, 4, 1.0],
        [4, 5, 1.0],
        [4, 6, 1.0],
    ]
    assert reaction.broken == frozenset([frozenset((0, 1))])
    assert reaction.formed == frozenset([frozenset((1, 4))])


def test_implicit_hydrogens_are_rejected():
    """An implicit hydrogen is an atom with no index, so it cannot be mapped.

    `[OH3+:1]` passes a map check -- every atom RDKit lists carries a number --
    while standing for three atoms it never names, and the atom that moves in a
    transfer is usually one of them.
    """
    with pytest.raises(R.ReactionError, match="leaves hydrogens implicit"):
        R.parse("[OH3+:1].[O:2]>>[O:1].[OH3+:2]")


def test_a_bare_atom_map_check_still_applies():
    with pytest.raises(R.ReactionError, match="must carry an atom map number"):
        R.parse("[O+:1]([H:2])([H])[H:4]>>[O+:1]([H:2])([H])[H:4]")


def test_sides_with_different_atoms_are_rejected():
    with pytest.raises(R.ReactionError, match="different atom map numbers"):
        R.parse("[O:1]([H:2])[H:3]>>[O:1][H:2]")


def test_a_map_number_may_not_change_element():
    with pytest.raises(R.ReactionError, match="different elements"):
        R.parse("[C:1]([H:2])([H:3])([H:4])[H:5]>>[N+:1]([H:2])([H:3])([H:4])[H:5]")


# ---------------------------------------------------------------------------
# channels
# ---------------------------------------------------------------------------


def test_a_transfer_is_routed_by_its_shared_atom():
    assert R.parse(WATER).channel() == ("transfer", (0, 1, 4))


def test_a_methyl_transfer_is_still_a_transfer():
    """The form does not care that the moving atom is a carbon: what it needs is
    one bond broken and one formed sharing an atom, which is a Walden inversion
    exactly as much as it is a proton transfer."""
    kind, triple = R.parse(SN2).channel()
    assert kind == "transfer"
    assert triple[1] == 1  # the carbon
    assert set((triple[0], triple[2])) == {0, 5}  # the two chlorines


def test_a_bond_fission_is_routed_to_the_crossing_form():
    kind, spec = R.parse(FISSION).channel()
    assert kind == "fission"
    assert spec[0] == (0, 1)


def test_two_unrelated_bond_changes_fall_back_to_rmsd():
    """Two bond changes sharing no atom have no single coordinate to be a
    function of, so neither the triangle nor the crossing form applies."""
    metathesis = "[H:1][H:2].[Cl:3][Cl:4]>>[H:1][Cl:3].[H:2][Cl:4]"
    assert R.parse(metathesis).channel() == ("rmsd", None)


# ---------------------------------------------------------------------------
# fragments
# ---------------------------------------------------------------------------


def test_one_molecule_gets_one_key_on_both_sides():
    """The water is the same water before and after, so it is fitted once."""
    reaction = R.parse(WATER)
    keys = {f.key for f in reaction.reactant_fragments}
    assert keys == {f.key for f in reaction.product_fragments}
    assert len(keys) == 2


def test_two_occurrences_of_a_molecule_agree_atom_for_atom():
    """Canonical order is what makes one fit reusable.

    The parent lists the product hydronium proton-first, so without canonical
    ordering its force field would be scattered onto `(H, O, H, H)` while the
    reactant's went onto `(O, H, H, H)` -- every array the right shape, every
    O-H bond on an H-H pair.
    """
    reaction = R.parse(WATER)
    by_key = {}
    for fragment in reaction.reactant_fragments + reaction.product_fragments:
        elements_of = tuple(int(reaction.numbers[i]) for i in fragment.indices)
        by_key.setdefault(fragment.key, []).append(elements_of)
    for occurrences in by_key.values():
        assert len(set(occurrences)) == 1


# ---------------------------------------------------------------------------
# the guess geometry
# ---------------------------------------------------------------------------


def test_the_guess_puts_every_changing_bond_at_the_contact_distance():
    """A transfer comes out symmetric by construction: both partial bonds at the
    contact distance, so the two heavy atoms sit at twice it."""
    reaction = R.parse(WATER)
    atoms = R.guess(reaction)
    assert len(atoms) == len(reaction)
    for bond in reaction.changing:
        i, j = sorted(bond)
        assert atoms.get_distance(i, j) == pytest.approx(
            R._contact(reaction.numbers, i, j), rel=0.02
        )


def test_the_guess_keeps_the_unchanged_bonds_near_equilibrium():
    reaction = R.parse(WATER)
    atoms = R.guess(reaction)
    for bond in reaction.reactant_bonds - reaction.changing:
        i, j = sorted(bond)
        assert 0.85 < atoms.get_distance(i, j) < 1.15


def test_the_guess_carries_the_formal_charge():
    assert R.guess(R.parse(WATER)).get_initial_charges().sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "smiles, spin",
    [
        (WATER, 0),
        ("[H:1][H:2].[O:3]>>[O:3][H:2].[H:1]", 2),  # O(3P) + H2: triplet
        ("[O:1].[O:2][H:3]>>[H:3].[O:1][O:2]", 3),  # parity alone says doublet
        (FISSION, 0),
    ],
)
def test_the_guess_carries_the_high_spin_of_its_reactants(smiles, spin):
    assert R.guess(R.parse(smiles)).info["spin"] == spin


def test_a_displacement_is_assigned_the_side_it_heads_for():
    """H2 + O -> OH + H: shortening H-H while stretching O-H is the way back."""
    reaction = R.parse("[H:1][H:2].[O:3]>>[O:3][H:2].[H:1]")
    ts = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [2.1, 0.0, 0.0]])
    back = ts + np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    assert R._toward(reaction, ts, back) == "reactant"
    assert R._toward(reaction, ts, 2 * ts - back) == "product"


def test_frame_spins_fill_in_the_reaction_spin():
    reaction = R.parse("[H:1][H:2].[O:3]>>[O:3][H:2].[H:1]")
    assert R.frame_spins(reaction, {"product": 0}) == {
        "reactant": 2,
        "transition": 2,
        "product": 0,
    }
    with pytest.raises(R.ReactionError, match="unknown frame kinds"):
        R.frame_spins(reaction, {"ts": 0})


def test_the_guess_does_not_overlap_atoms():
    atoms = R.guess(R.parse(SN2))
    distances = atoms.get_all_distances()
    np.fill_diagonal(distances, np.inf)
    assert distances.min() > 0.8


# ---------------------------------------------------------------------------
# merging fragment force fields into a state
# ---------------------------------------------------------------------------


def _fragment_field(smiles: str, e0: float) -> tuple:
    """A minimal fitted force field for one fragment, in `build` order.

    Each bond's `r0` encodes the pair it was fitted for, so a term scattered
    onto the wrong atoms is visible as a wrong number rather than as a wrong
    shape.
    """
    import fastforces as ff

    atoms = ff.build(smiles)
    graph = ff.perceive(atoms)
    bonds = np.array(
        sorted(tuple(sorted(e)) for e in graph.edges()), dtype=int
    ).reshape(-1, 2)
    params = Parameters(numbers=atoms.get_atomic_numbers())
    column = np.arange(len(atoms))[:, None]
    for term, kwargs in elements.defaults_for(params.numbers).items():
        params.terms[term] = {"atoms": column.copy(), "kwargs": dict(kwargs)}
    params.terms["bond"] = {
        "atoms": bonds,
        "kwargs": {
            "r0": np.arange(1, len(bonds) + 1, dtype=float),
            "k": np.full(len(bonds), 30.0),
            "D": np.full(len(bonds), 5.0),
            "c": np.zeros(len(bonds)),
            "b": np.full(len(bonds), 4.0),
        },
    }
    params.terms["reference"] = {
        "atoms": np.zeros((1, 1), dtype=int),
        "kwargs": {"E0": np.array([e0])},
    }
    return atoms, params


def test_a_state_carries_every_bond_of_its_own_topology():
    reaction = R.parse(WATER)
    fitted = {
        f.key: _fragment_field(f.smiles, -1.0 if f.charge else -2.0)
        for f in reaction.reactant_fragments
    }
    for side in ("reactant", "product"):
        params = R.state_parameters(reaction, side, fitted)
        got = {frozenset(row) for row in params.terms["bond"]["atoms"]}
        assert got == set(reaction.bonds(side))


def test_a_state_sums_the_fragment_reference_energies():
    reaction = R.parse(WATER)
    fitted = {
        f.key: _fragment_field(f.smiles, -1.0 if f.charge else -2.0)
        for f in reaction.reactant_fragments
    }
    assert R.state_parameters(reaction, "reactant", fitted).e0 == pytest.approx(-3.0)


def test_a_state_rebuilds_its_nonbonded_block_over_every_atom():
    """`atom` and `lennardjones` are element defaults, never fitted, so they are
    regenerated over the combined system rather than concatenated."""
    reaction = R.parse(WATER)
    fitted = {
        f.key: _fragment_field(f.smiles, 0.0) for f in reaction.reactant_fragments
    }
    params = R.state_parameters(reaction, "reactant", fitted)
    assert list(params.terms["atom"]["atoms"][:, 0]) == list(range(len(reaction)))
    assert list(params.terms["lennardjones"]["atoms"][:, 0]) == list(
        range(len(reaction))
    )


def test_the_two_states_differ_only_where_their_topologies_do():
    reaction = R.parse(WATER)
    fitted = {
        f.key: _fragment_field(f.smiles, 0.0)
        for f in reaction.reactant_fragments + reaction.product_fragments
    }
    states = {
        side: R.state_parameters(reaction, side, fitted)
        for side in ("reactant", "product")
    }
    bonds = {
        side: {frozenset(row) for row in states[side].terms["bond"]["atoms"]}
        for side in states
    }
    assert bonds["reactant"] - bonds["product"] == reaction.broken
    assert bonds["product"] - bonds["reactant"] == reaction.formed


@pytest.mark.parametrize(
    "smiles",
    [
        "[H:1][H:2].[O:3]>>[O:3][H:2].[H:1]",
        "[O:1][O:2][H:3].[H:4]>>[O:1][O:2].[H:3][H:4]",
    ],
)
def test_radical_fragments_map_onto_their_atoms(smiles):
    """`ase2rdkit` reads every atom back with its valence filled -- `[OH]` as
    water's oxygen, a lone `[O]` as water -- so a radical fragment only maps
    if the match leaves the radicals out."""
    reaction = R.parse(smiles)
    fragments = reaction.reactant_fragments + reaction.product_fragments
    fitted = {f.key: _fragment_field(f.smiles, 0.0) for f in fragments}
    for side in ("reactant", "product"):
        params = R.state_parameters(reaction, side, fitted)
        got = {frozenset(row) for row in params.terms["bond"]["atoms"]}
        assert got == set(reaction.bonds(side))


def test_a_missing_fragment_fit_is_an_error():
    reaction = R.parse(WATER)
    with pytest.raises(R.ReactionError, match="no force field was fitted"):
        R.state_parameters(reaction, "reactant", {})


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sn2_surface(tmp_path_factory, tblite_factory):
    """One whole EVB surface: two fragment fits, a saddle and a coupling.

    The identity SN2 rather than the water transfer, because it has a real
    gas-phase barrier -- the shared-proton geometry of a water dimer is a
    minimum, so that channel has no saddle to find.
    """
    import fastforces as ff

    workdir = tmp_path_factory.mktemp("sn2")
    return ff.parameterize_reaction(
        SN2,
        tblite_factory,
        config=ff.FitConfig(n_mode_frames=20, n_conformers=0),
        workdir=str(workdir),
    )


@pytest.mark.slow
def test_the_saddle_is_the_symmetric_one(sn2_surface):
    """An identity SN2's transition state is D3h: both C-Cl equal."""
    saddle = sn2_surface.frames[1]
    assert saddle.get_distance(0, 1) == pytest.approx(
        saddle.get_distance(1, 5), abs=0.01
    )
    assert saddle.info["frame_kind"] == "transition"


@pytest.mark.slow
def test_the_endpoints_are_the_two_topologies_the_smiles_names(sn2_surface):
    import fastforces as ff

    reaction = sn2_surface.reaction
    for frame, side in zip(
        sn2_surface.frames[::2], ("reactant", "product"), strict=True
    ):
        found = {frozenset(e) for e in ff.perceive(frame).edges()}
        assert found == set(reaction.bonds(side))


@pytest.mark.slow
def test_the_barrier_is_reproduced_exactly_at_the_saddle(sn2_surface):
    """The amplitude was fitted by inverting the 2x2 there, so this is the
    identity holding rather than an approximation converging."""
    energies = sn2_surface.energies()
    assert energies["evb"][1] == pytest.approx(energies["reference"][1], abs=1e-6)


@pytest.mark.slow
def test_the_two_diabats_swap_across_the_saddle(sn2_surface):
    """Each state is the lower one on its own side, which is what makes the
    surface a reaction path rather than two unrelated force fields."""
    energies = sn2_surface.energies()
    reactant, product = energies["diabatic"][0], energies["diabatic"][2]
    assert reactant[0] < reactant[1]
    assert product[1] < product[0]


@pytest.mark.slow
def test_an_identity_reaction_comes_out_symmetric(sn2_surface):
    energies = sn2_surface.energies()
    assert energies["reference"][0] == pytest.approx(energies["reference"][2], abs=1e-4)
    assert energies["evb"][0] == pytest.approx(energies["evb"][2], abs=1e-4)


@pytest.mark.slow
def test_the_surface_is_a_working_ase_calculator(sn2_surface):
    saddle = sn2_surface.frames[1].copy()
    saddle.calc = sn2_surface.calculator(saddle)
    assert saddle.get_forces().shape == (len(saddle), 3)
    assert saddle.calc.results["statevec"] == pytest.approx([0.5, 0.5], abs=0.05)


@pytest.mark.slow
def test_the_written_files_round_trip(sn2_surface, tmp_path):
    import fastforces as ff
    from fastforces.coupling import Coupling

    stem = tmp_path / "sn2"
    sn2_surface.write(str(stem))
    back = Coupling.from_jsonl(str(stem.with_suffix(".jsonl")))
    assert back.provenance == sn2_surface.coupling.provenance
    for side in ("reactant", "product"):
        params = ff.Parameters.from_jsonl(str(tmp_path / f"sn2-{side}.jsonl"))
        assert set(map(frozenset, params.terms["bond"]["atoms"])) == set(
            sn2_surface.reaction.bonds(side)
        )
