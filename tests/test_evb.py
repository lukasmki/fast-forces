"""The two-state surface: the eigenvalue, its gradient, and what mixes.

The states here are built by hand rather than fitted, for the reason
`test_reaction`'s own synthetic fields give: a fit would make every number
plausible and none of them diagnostic.  Two one-bond states over three atoms are
enough to exercise everything the calculator does.
"""

import networkx as nx
import numpy as np
import pytest
from ase import Atoms

from fastforces import elements
from fastforces.coupling import Coupling
from fastforces.evb import EVB, diabatic_energies
from fastforces.forcefield import exclusions
from fastforces.params import Parameters
from fastforces.topology import exclusion_mask

NUMBERS = [8, 1, 8]


def _state(bond, e0=-1.5):
    params = Parameters(numbers=np.asarray(NUMBERS))
    column = np.arange(len(NUMBERS))[:, None]
    for term, kwargs in elements.defaults_for(params.numbers).items():
        params.terms[term] = {"atoms": column.copy(), "kwargs": dict(kwargs)}
    params.terms["bond"] = {
        "atoms": np.array([bond]),
        "kwargs": {
            "r0": np.array([0.98]),
            "k": np.array([40.0]),
            "D": np.array([5.0]),
            "c": np.array([0.0]),
            "b": np.array([4.0]),
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


def _excluded_state(bond, e0=-1.5):
    """One state, carrying the exclusion mask of its own single bond.

    The two states then exclude *different* pairs, which is the case the whole
    of section 5.2 exists for and the only one where the screen has anything to
    collapse.
    """
    params = _state(bond, e0)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(NUMBERS)))
    graph.add_edge(*bond)
    params.exclusions = exclusion_mask(graph, len(NUMBERS))
    return params


@pytest.fixture
def excluded_states():
    return [_excluded_state([0, 1]), _excluded_state([2, 1])]


def _evb(positions, states, coupling):
    atoms = Atoms(numbers=NUMBERS, positions=positions)
    atoms.calc = EVB(atoms, states=states, coupling=coupling)
    return atoms


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


def test_forces_match_the_numerical_gradient(geometry, states):
    coupling = _coupling()
    atoms = _evb(geometry, states, coupling)
    forces = atoms.get_forces()

    step = 1e-6
    numeric = np.zeros_like(geometry)
    for i in range(len(geometry)):
        for axis in range(3):
            probes = []
            for sign in (+1, -1):
                shifted = geometry.copy()
                shifted[i, axis] += sign * step
                probes.append(_evb(shifted, states, coupling).get_potential_energy())
            numeric[i, axis] = -(probes[0] - probes[1]) / (2 * step)
    assert np.allclose(forces, numeric, atol=1e-5)


def test_forces_match_the_numerical_gradient_with_exclusions(
    geometry, excluded_states
):
    """The same check, with the two states excluding different pairs.

    This is the one that says the screen is right.  `S` is built from the
    ground-state weights and then held fixed under the derivative, while the
    per-state Coulomb corrections sit on the diagonal and move the weights --
    so a screen differentiated by mistake, or one assembled with the wrong
    weights, is a force that no longer differentiates the energy it is paired
    with.  Nothing else here would catch it: the energy is right either way.
    """
    coupling = _coupling()
    atoms = _evb(geometry, excluded_states, coupling)
    forces = atoms.get_forces()

    step = 1e-6
    numeric = np.zeros_like(geometry)
    for i in range(len(geometry)):
        for axis in range(3):
            probes = []
            for sign in (+1, -1):
                shifted = geometry.copy()
                shifted[i, axis] += sign * step
                probes.append(
                    _evb(shifted, excluded_states, coupling).get_potential_energy()
                )
            numeric[i, axis] = -(probes[0] - probes[1]) / (2 * step)
    assert np.allclose(forces, numeric, atol=1e-5)


def test_the_diagonal_carries_each_state_own_coulomb_correction(
    geometry, excluded_states
):
    """Two states excluding different pairs get different diagonals from it.

    The correction is what lets the exclusion decide which bonding pattern is
    lower; a screen alone could not, being downstream of the weights.
    """
    atoms = _evb(geometry, excluded_states, _coupling())
    atoms.get_potential_energy()
    with_exclusions = np.diag(atoms.calc.results["hamiltonian"])

    plain = _evb(geometry, [_state([0, 1]), _state([2, 1])], _coupling())
    plain.get_potential_energy()
    without = np.diag(plain.calc.results["hamiltonian"])

    assert not np.allclose(with_exclusions, without)
    # and the two states are moved by different amounts, since their masks differ
    shift = with_exclusions - without
    assert shift[0] != pytest.approx(shift[1])


def test_the_screen_reproduces_the_weighted_diagonal_corrections(
    geometry, excluded_states
):
    """`(ccoul/2) sum_ij S_ij Q_i Q_j K_ij  ==  E_full + sum_s w_s c_s`.

    The identity the design rests on: the per-state correction and the screened
    contraction are the same quantity summed two ways, which is why `EVB` can
    put one on the diagonal and take its energy from the eigenvalue while taking
    its gradient from the other.
    """
    atoms = _evb(geometry, excluded_states, _coupling())
    atoms.get_potential_energy()
    weights = atoms.calc.results["statevec"]

    evaluator = atoms.calc.electrostatic
    terms = excluded_states[0].terms
    pos, pbc, cell = atoms.get_positions(), atoms.pbc, np.array(atoms.cell)
    indices, Q, K = evaluator.charges_and_kernel(pos, pbc, cell, terms)

    full = evaluator(pos, pbc, cell, terms)[0]
    corrections = [
        exclusions.coulomb_correction(
            Q, K, p.exclusions[np.ix_(indices, indices)]
        )
        for p in excluded_states
    ]
    masks = [p.exclusions[np.ix_(indices, indices)] for p in excluded_states]
    screen = exclusions.screen(masks, weights, len(indices))

    assert evaluator(pos, pbc, cell, terms, screen)[0] == pytest.approx(
        full + float(np.dot(weights, corrections))
    )


def test_the_screen_collapses_the_hellmann_feynman_sum(geometry, excluded_states):
    """`sum_s w_s dH_ss/dr` is the single screened evaluation, to rounding.

    `diabatic_forces` screens per state with that state's own `1 - M^s`; `EVB`
    screens once with `1 - sum_s w_s M^s`.  They agree because the whole
    electrostatic gradient -- the explicit part and the charge-response adjoint
    alike -- is linear in the screen, which is the claim that lets the surface
    pay for one electrostatic evaluation per step instead of one per state.
    """
    from fastforces.evb import diabatic_forces

    coupling = _coupling()
    atoms = _evb(geometry, excluded_states, coupling)
    forces = atoms.get_forces()

    weights = atoms.calc.results["statevec"]
    c = np.linalg.eigh(atoms.calc.results["hamiltonian"])[1][:, 0]
    _, coupling_forces, _ = coupling(geometry, atoms.pbc, np.array(atoms.cell))
    _, per_state = diabatic_forces(atoms, excluded_states)

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
    both diagonals passes straight through the eigenvalue.

    That is what makes the placement of the topology-independent terms free, and
    it is the property the `sqrt((1+h) H1 H2)` coupling does not have.
    """
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
