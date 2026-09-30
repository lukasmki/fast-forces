"""The two-state surface: the eigenvalue, its gradient, and what mixes.

The states here are built by hand rather than fitted, for the reason
`test_reaction`'s own synthetic fields give: a fit would make every number
plausible and none of them diagnostic.  Two one-bond states over three atoms are
enough to exercise everything the calculator does -- and because each state's
exclusions are derived from its own bond, the two exclude *different* pairs,
which is the case the Coulomb exclusion's per-state treatment exists for.

The last test holds the whole thing to DynamicTopology's `System` on a real
dataset's reaction, which is the claim `evb`'s module docstring makes.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, io

from fastforces import elements
from fastforces.coupling import Coupling
from fastforces.evb import EVB, diabatic_energies, diabatic_forces
from fastforces.params import Parameters, terms_to_blocks

NUMBERS = [8, 1, 8]
WATER = Path(__file__).resolve().parents[2] / "DynamicTopology" / "datasets" / "Water"


def _state(bond, e0=-1.5, charges=None, q0=None):
    """`charges` makes it a point-charge state; `q0` an ACKS2 one with those
    reference charges (zero without)."""
    params = Parameters(numbers=np.asarray(NUMBERS))
    column = np.arange(len(NUMBERS))[:, None]
    fixed = charges is not None
    defaults = elements.defaults_for(
        params.numbers, "pointcharge" if fixed else "acks2"
    )
    for term, kwargs in defaults.items():
        params.terms[term] = {"atoms": column.copy(), "kwargs": dict(kwargs)}
    if q0 is not None:
        params.terms["atom"]["kwargs"]["q0"] = np.asarray(q0, dtype=float)
    if fixed:
        params.terms["charge"] = {
            "atoms": column.copy(),
            "kwargs": {"q": np.asarray(charges, dtype=float)},
        }
    params.terms["bond"] = {
        "atoms": np.array([bond]),
        "kwargs": {
            "r0": np.array([0.98]),
            "k": np.array([40.0]),
            "D": np.array([5.0]),
        },
    }
    params.terms["reference"] = {
        "atoms": np.zeros((1, 1), dtype=int),
        "kwargs": {"E0": np.array([e0])},
    }
    return params


def _coupling(amplitude=-2.0, width=3.0):
    return Coupling(
        terms={
            "threebody": {
                "atoms": np.array([[0, 1, 2]]),
                "kwargs": {
                    "A": np.array([amplitude]),
                    "a": np.array([width]),
                    "ra0": np.array([1.2]),
                    "rb0": np.array([1.2]),
                    "t0": np.array([np.pi]),
                },
            }
        },
        provenance={"threebody": "placeholder"},
    )


@pytest.fixture
def geometry():
    rng = np.random.default_rng(3)
    return np.array(
        [[0.0, 0.0, 0.0], [1.15, 0.2, 0.05], [2.4, 0.1, -0.1]]
    ) + rng.normal(scale=0.05, size=(3, 3))


@pytest.fixture
def states():
    return [_state([0, 1]), _state([2, 1])]


@pytest.fixture
def charged_states():
    """A proton transfer whose -1 moves with the proton, as fixed charges do."""
    return [
        _state([0, 1], charges=[-0.4, 0.4, -1.0]),
        _state([2, 1], charges=[-1.0, 0.4, -0.4]),
    ]


@pytest.fixture
def ion_states():
    """The same transfer under fragment ACKS2: the -1 moves through `q0`."""
    return [
        _state([0, 1], q0=[-0.4, 0.4, -1.0]),
        _state([2, 1], q0=[-1.0, 0.4, -0.4]),
    ]


def _evb(positions, states, coupling):
    atoms = Atoms(numbers=NUMBERS, positions=positions)
    atoms.calc = EVB(atoms, states=states, coupling=coupling)
    return atoms


def _numeric_forces(geometry, states, coupling, step=1e-6):
    numeric = np.zeros_like(geometry)
    for i in range(len(geometry)):
        for axis in range(3):
            probes = []
            for sign in (+1, -1):
                shifted = geometry.copy()
                shifted[i, axis] += sign * step
                probes.append(_evb(shifted, states, coupling).get_potential_energy())
            numeric[i, axis] = -(probes[0] - probes[1]) / (2 * step)
    return numeric


def test_energy_is_the_lower_eigenvalue(geometry, states):
    atoms = _evb(geometry, states, _coupling())
    energy = atoms.get_potential_energy()
    hamiltonian = atoms.calc.results["hamiltonian"]
    assert energy == pytest.approx(np.linalg.eigvalsh(hamiltonian)[0])


def test_the_diagonal_is_each_state_on_its_own(geometry, states):
    """What `coupling.fit_amplitude` inverts has to be what the surface runs
    with, so the two are the same code path and must agree exactly."""
    atoms = _evb(geometry, states, _coupling())
    atoms.get_potential_energy()
    diagonal = np.diag(atoms.calc.results["hamiltonian"])
    assert diagonal == pytest.approx(diabatic_energies(atoms, states))


def test_the_surface_lies_below_both_diabats(geometry, states):
    atoms = _evb(geometry, states, _coupling())
    assert atoms.get_potential_energy() < min(diabatic_energies(atoms, states))


@pytest.mark.parametrize("which", ["acks2", "acks2-ion", "pointcharge"])
def test_forces_match_the_numerical_gradient(
    geometry, states, charged_states, ion_states, which
):
    """With each state on its own topology, under both electrostatic terms.

    Under fragment ACKS2 each state equilibrates over its own molecules, around
    its own reference charges -- zero, or a -1 that moves with the proton;
    under fixed charges the states differ in the charges themselves.  Either
    way the weights move with the geometry, so a Hellmann-Feynman sum assembled
    with the wrong weights is a force that no longer differentiates the energy
    it is paired with.
    """
    pair = {"acks2": states, "acks2-ion": ion_states, "pointcharge": charged_states}[
        which
    ]
    coupling = _coupling()
    forces = _evb(geometry, pair, coupling).get_forces()
    assert np.allclose(forces, _numeric_forces(geometry, pair, coupling), atol=1e-5)


def test_the_states_exclude_different_pairs(geometry, states):
    """Each diabat's exclusions follow its own bond, so the diagonals differ by
    more than the one bond term they differ in."""
    from DynamicTopology.forcefield.evaluate import evaluate
    from DynamicTopology.forcefield.exclusions import exclusion_terms

    atoms = Atoms(numbers=NUMBERS, positions=geometry)
    for params, pair in zip(states, ([0, 1], [1, 2]), strict=True):
        terms = params.to_terms()
        excluded = {
            tuple(sorted(t["atoms"].values()))
            for t in exclusion_terms(terms, atoms.numbers)
        }
        assert tuple(pair) in excluded
        assert evaluate(atoms, terms).energy == pytest.approx(
            diabatic_energies(atoms, [params, params])[0]
        )


def test_the_hellmann_feynman_sum_is_the_state_gradients(geometry, states):
    """`forces = sum_s w_s dH_ss/dr + 2 c1 c2 dV/dr`, to rounding."""
    coupling = _coupling()
    atoms = _evb(geometry, states, coupling)
    forces = atoms.get_forces()

    weights = atoms.calc.results["statevec"]
    c = np.linalg.eigh(atoms.calc.results["hamiltonian"])[1][:, 0]
    _, coupling_forces, _ = coupling(geometry, atoms.pbc, np.array(atoms.cell))
    _, per_state = diabatic_forces(atoms, states)

    collapsed = (
        weights[0] * per_state[0]
        + weights[1] * per_state[1]
        + 2.0 * c[0] * c[1] * coupling_forces
    )
    assert np.allclose(forces, collapsed, atol=1e-12)


def test_statevec_is_the_squared_ground_state_eigenvector(geometry, states):
    atoms = _evb(geometry, states, _coupling())
    atoms.get_potential_energy()
    weights = atoms.calc.results["statevec"]
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights >= 0.0)


def test_degenerate_diabats_mix_evenly():
    """At the symmetric geometry the two states are the same energy, so the
    ground state is an even mixture whatever the coupling is worth."""
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0]])
    states = [_state([0, 1]), _state([2, 1])]
    atoms = _evb(positions, states, _coupling())
    atoms.get_potential_energy()
    assert atoms.calc.results["statevec"] == pytest.approx([0.5, 0.5])


def test_the_coupling_sets_the_depth_at_a_degenerate_geometry():
    """With the diabats degenerate the stabilization is exactly `|V|`, which is
    the identity `fit_amplitude` inverts."""
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0]])
    states = [_state([0, 1]), _state([2, 1])]
    atoms = _evb(positions, states, _coupling(amplitude=-0.75))
    diabats = diabatic_energies(atoms, states)
    assert diabats[0] == pytest.approx(diabats[1])
    assert atoms.get_potential_energy() == pytest.approx(diabats[0] - 0.75)


def test_a_common_shift_moves_the_surface_by_exactly_that_shift(geometry):
    """The coupling is a function of the geometry alone, so a constant added to
    both diagonals passes straight through the eigenvalue."""
    coupling = _coupling()
    plain = _evb(geometry, [_state([0, 1]), _state([2, 1])], coupling)
    shifted = _evb(
        geometry,
        [_state([0, 1], e0=-1.5 - 3.0), _state([2, 1], e0=-1.5 - 3.0)],
        coupling,
    )
    assert shifted.get_potential_energy() == pytest.approx(
        plain.get_potential_energy() - 3.0
    )


def test_zero_coupling_is_the_lower_of_the_two_states(geometry, states):
    atoms = _evb(geometry, states, _coupling(amplitude=0.0))
    assert atoms.get_potential_energy() == pytest.approx(
        min(diabatic_energies(atoms, states))
    )


def test_two_states_are_required(geometry):
    with pytest.raises(ValueError, match="exactly two"):
        EVB(
            Atoms(numbers=NUMBERS, positions=geometry),
            states=[_state([0, 1])],
            coupling=_coupling(),
        )


def test_a_coupling_is_required(geometry, states):
    with pytest.raises(ValueError, match="needs a Coupling"):
        EVB(Atoms(numbers=NUMBERS, positions=geometry), states=states)


def test_the_states_must_share_their_electrostatics(geometry, states, charged_states):
    with pytest.raises(ValueError, match="disagree about their electrostatics"):
        EVB(
            Atoms(numbers=NUMBERS, positions=geometry),
            states=[states[0], charged_states[1]],
            coupling=_coupling(),
        )


@pytest.mark.skipif(not WATER.exists(), reason="needs the DynamicTopology checkout")
@pytest.mark.parametrize(
    "reaction", ["h3o-h2o-transfer", "h2o-oh-transfer", "h2o-autoionization"]
)
def test_the_surface_is_dynamictopology_s(reaction):
    """`EVB` is what DynamicTopology's `System` gives the same reaction.

    The two states are the dataset's own templates, remapped onto the reaction's
    atoms by `ReactionSet.get_terms` exactly as a simulation remaps them, and the
    coupling is the reaction's own fitted term.  Evaluated at the transition
    state, where both diabats are occupied and the coupling is at its largest.
    """
    from DynamicTopology.core import ReactionSet, Topology
    from DynamicTopology.forcefield.params import ForceFieldParams, use
    from DynamicTopology.system import System

    manifest = WATER / "Water.json"
    params = ForceFieldParams.from_dict(
        json.loads(manifest.read_text())["global_params"]
    )
    frames = io.read(WATER / "reactions" / f"{reaction}.xyz", index=":")
    ts = frames[len(frames) // 2].copy()
    ts.calc = None
    rows = [
        json.loads(line) for line in open(WATER / "reactions" / f"{reaction}.jsonl")
    ]

    with use(params):
        reaction_set = ReactionSet(manifest)
        states = []
        for end in (frames[0], frames[-1]):
            topology = Topology.from_atoms(end)
            terms = [
                t
                for t in reaction_set.get_terms(topology)
                if not t["type"].endswith("exclusion")
            ]
            states.append(Parameters(numbers=ts.numbers, terms=terms_to_blocks(terms)))
        coupling = Coupling.from_terms(rows)

        work = ts.copy()
        work.calc = EVB(work, states=states, coupling=coupling)
        energy, forces = work.get_potential_energy(), work.get_forces()

        seed = ts.copy()
        seed.info["connectivity"] = frames[0].info["connectivity"]
        expected = System(
            seed, Topology.from_atoms(seed), reaction_set, evb={"max_states": 2}
        ).calculate()

    (block,) = [b for b in expected["blocks"] if b["nstates"] > 1]
    assert block["nstates"] == 2
    assert energy == pytest.approx(expected["energy"], abs=1e-8)
    np.testing.assert_allclose(forces, expected["forces"], atol=1e-8)
