"""Section 5: taking the near-neighbour part back off the three pair sums.

Two halves that behave quite differently, and the tests are grouped that way.
`ZBL` and the 12-6 are *subtracted*, so the property to pin is that the
subtraction is exact -- both halves have to be the same function of the same
numbers, to the last bit, or a template stops reproducing its own energy.  The
Coulomb sum is *screened* instead, because its charges come from a solve whose
matrix contains the kernel, so the property to pin there is that the charges
never notice.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from ase import Atoms

import fastforces as ff
from fastforces.calculator import FastForces
from fastforces.forcefield import exclusions
from fastforces.forcefield.acks2 import ACKS2
from fastforces.forcefield.coulomb import Coulomb
from fastforces.forcefield.lj import LennardJones
from fastforces.forcefield.zbl import ZBL
from fastforces.params import Parameters
from fastforces.topology import enumerate_terms, exclusion_mask

EXAMPLE_SMILES = "CC#N"
# Off `__file__`, like `test_params`, so the suite runs from any directory.
EXAMPLE_JSONL = str(Path(__file__).parent / "acetonitrile.jsonl")


@pytest.fixture
def example():
    """Acetonitrile and its fitted field, with the mask `from_jsonl` derived."""
    atoms = ff.build(EXAMPLE_SMILES)
    params = Parameters.from_jsonl(
        EXAMPLE_JSONL, numbers=atoms.get_atomic_numbers()
    )
    return atoms, params


# ---------------------------------------------------------------------------
# the mask
# ---------------------------------------------------------------------------


def test_a_loaded_field_derives_the_same_mask_the_topology_does(example):
    """`from_rows` reads the graph out of the `bond` terms it just parsed.

    A field loaded from a jsonl has to exclude what the field it was written
    from excluded, or it evaluates to a different energy than it was fit at --
    and a mask stored alongside the terms could go stale against an edited bond.
    """
    atoms, params = example
    assert params.exclusions is not None
    assert np.array_equal(params.exclusions, enumerate_terms(atoms).exclusions)


def test_the_mask_is_symmetric_with_an_empty_diagonal(example):
    """`S_ii` must survive the screen: an atom is not excluded from its images.

    Under periodic boundaries `K_ii` is a real self-image interaction, so a mask
    that marked the diagonal would delete it.
    """
    _, params = example
    mask = params.exclusions
    assert np.array_equal(mask, mask.T)
    assert not np.any(np.diagonal(mask))


# ---------------------------------------------------------------------------
# the additive half
# ---------------------------------------------------------------------------


def test_the_additive_exclusion_cancels_exactly(example):
    """Every pair of acetonitrile is inside the depth, so all three sums vanish.

    To the last bit, not approximately.  Both halves go through one
    `pair_potential` apiece -- switches, taper and core continuation included --
    which is the only reason it can be exact: at a bond length each half is tens
    of eV, and a transcription that drifted would show up here as a residue
    rather than as anything diagnosable downstream.
    """
    atoms, params = example
    pos = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    pbc, cell = np.zeros(3, dtype=bool), np.eye(3) * 14.0

    e1, f1, w1 = LennardJones()(pos, pbc, cell, params.terms)
    e2, f2, w2 = ZBL()(pos, numbers, pbc, cell)
    sigma, eps = exclusions.lj_parameters(params.terms, len(numbers))
    e3, f3, w3 = exclusions.additive(
        pos, numbers, pbc, cell, params.exclusions, sigma, eps
    )

    # the halves are individually large, which is what makes the cancellation
    # worth asserting at all
    assert abs(e1) + abs(e2) > 10.0
    assert e1 + e2 + e3 == pytest.approx(0.0, abs=1e-10)
    assert np.allclose(f1 + f2 + f3, 0.0, atol=1e-10)
    assert np.allclose(w1 + w2 + w3, 0.0, atol=1e-10)


def test_a_pair_with_no_12_6_parameters_gets_no_12_6_exclusion(example):
    """Where sigma or eps is zero the whole-system term is zero too.

    So there is nothing to cancel, and cancelling anyway would divide by a zero
    core radius.  ZBL admits no such exemption: it has no free parameters.
    """
    atoms, params = example
    numbers = atoms.get_atomic_numbers()
    pos = atoms.get_positions()
    pbc, cell = np.zeros(3, dtype=bool), np.eye(3) * 14.0
    mask = params.exclusions

    zeros = np.zeros(len(numbers))
    only_zbl = exclusions.additive(pos, numbers, pbc, cell, mask, zeros, zeros)[0]
    bare_zbl = ZBL()(pos, numbers, pbc, cell)[0]
    assert only_zbl == pytest.approx(-bare_zbl)


def test_an_empty_mask_contributes_nothing(example):
    """A field whose pairs are all far apart along the graph subtracts nothing."""
    atoms, params = example
    numbers = atoms.get_atomic_numbers()
    empty = np.zeros((len(numbers), len(numbers)), dtype=bool)
    energy, forces, virial = exclusions.additive(
        atoms.get_positions(), numbers, np.zeros(3, dtype=bool), np.eye(3) * 14.0, empty
    )
    assert energy == 0.0
    assert np.allclose(forces, 0.0)
    assert np.allclose(virial, 0.0)


# ---------------------------------------------------------------------------
# the screened half
# ---------------------------------------------------------------------------


def _acks2_terms(numbers):
    from fastforces import elements

    return {
        "atom": {
            "atoms": np.arange(len(numbers))[:, None],
            "kwargs": elements.acks2_defaults(numbers),
        }
    }


def test_the_charges_do_not_depend_on_the_screen(example):
    """The solve uses the unmasked kernel, always.

    This is the property the whole two-piece design exists to protect: the
    charges have to be identical on every diabatic state, so the mask cannot
    reach `A`.  Masking the kernel inside the solve instead costs 0.88 eV of
    dependence on the arbitrary choice of reference state.
    """
    atoms, params = example
    numbers = atoms.get_atomic_numbers()
    pos = atoms.get_positions()
    pbc, cell = np.zeros(3, dtype=bool), np.eye(3) * 14.0
    terms = _acks2_terms(numbers)

    unscreened = ACKS2()
    unscreened(pos, pbc, cell, terms)
    screened = ACKS2()
    screen = exclusions.screen([params.exclusions], [1.0], len(numbers))
    screened(pos, pbc, cell, terms, screen)
    assert np.allclose(unscreened.Q, screened.Q)


def test_the_screen_removes_exactly_the_excluded_pairs(example):
    """`E(S) = E(1) - ccoul * sum_{(i,j) in excl} Q_i Q_j K_ij`.

    The screened contraction and the per-state correction are the same quantity
    written two ways; `EVB` relies on that identity to put one on the diagonal
    and evaluate the other.
    """
    atoms, params = example
    numbers = atoms.get_atomic_numbers()
    pos = atoms.get_positions()
    pbc, cell = np.zeros(3, dtype=bool), np.eye(3) * 14.0
    terms = _acks2_terms(numbers)

    evaluator = ACKS2()
    indices, Q, K = evaluator.charges_and_kernel(pos, pbc, cell, terms)
    full = evaluator(pos, pbc, cell, terms)[0]

    mask = params.exclusions[np.ix_(indices, indices)]
    correction = exclusions.coulomb_correction(Q, K, mask)
    screen = exclusions.screen([mask], [1.0], len(indices))
    assert evaluator(pos, pbc, cell, terms, screen)[0] == pytest.approx(
        full + correction
    )


def test_a_fractional_screen_is_the_weighted_sum_of_its_states():
    """`S = 1 - sum_s w_s M^s`, and the energy is linear in it.

    Fractional entries are the normal case and are not an interpolation: this is
    what lets the Hellmann-Feynman sum collapse into a single screened
    evaluation.
    """
    n = 4
    left = exclusion_mask(nx.Graph([(0, 1)]), n)
    right = exclusion_mask(nx.Graph([(2, 3)]), n)
    weights = (0.3, 0.7)
    s = exclusions.screen([left, right], weights, n)

    assert s[0, 1] == pytest.approx(1.0 - weights[0])
    assert s[2, 3] == pytest.approx(1.0 - weights[1])
    assert s[0, 3] == pytest.approx(1.0)
    # a pair excluded in every state comes out at exactly zero
    both = exclusions.screen([left, left], weights, n)
    assert both[0, 1] == pytest.approx(0.0)


def test_fixed_charges_are_screened_by_the_same_code(example):
    """`Coulomb` takes the screen too, so the two terms cannot drift apart.

    A field should not acquire a different exclusion convention by choosing its
    electrostatics.
    """
    atoms, params = example
    numbers = atoms.get_atomic_numbers()
    pos = atoms.get_positions()
    pbc, cell = np.zeros(3, dtype=bool), np.eye(3) * 14.0
    charges = np.linspace(-0.3, 0.3, len(numbers))
    charges = charges - charges.mean()
    terms = {"coulomb": {"atoms": np.arange(len(numbers))[:, None],
                         "kwargs": {"q": charges}}}

    evaluator = Coulomb()
    indices, Q, K = evaluator.charges_and_kernel(pos, pbc, cell, terms)
    full = evaluator(pos, pbc, cell, terms)[0]
    mask = params.exclusions[np.ix_(indices, indices)]
    screen = exclusions.screen([mask], [1.0], len(indices))
    assert evaluator(pos, pbc, cell, terms, screen)[0] == pytest.approx(
        full + exclusions.coulomb_correction(Q, K, mask)
    )


# ---------------------------------------------------------------------------
# through the calculator
# ---------------------------------------------------------------------------


def _forces_match(atoms_of, positions, step=1e-6):
    numeric = np.zeros_like(positions)
    for i in range(len(positions)):
        for axis in range(3):
            probe = []
            for sign in (+1, -1):
                shifted = positions.copy()
                shifted[i, axis] += sign * step
                probe.append(atoms_of(shifted).get_potential_energy())
            numeric[i, axis] = -(probe[0] - probe[1]) / (2 * step)
    return numeric


@pytest.mark.parametrize("periodic", [False, True])
def test_the_calculator_gradient_survives_the_exclusions(example, periodic):
    """Energy and forces still agree once all three sums are being cut into.

    The screened electrostatic gradient is the one at risk: the screen enters
    `dE/dQ` and therefore the response adjoint, while `dA/dr` stays unmasked, so
    a screen applied in the wrong place is a force that no longer differentiates
    its own energy.
    """
    atoms, params = example
    numbers = atoms.get_atomic_numbers()
    cell = np.eye(3) * 14.0

    def build(positions):
        work = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=periodic)
        work.calc = FastForces(work, params)
        return work

    positions = atoms.get_positions()
    forces = build(positions).get_forces()
    assert np.allclose(forces, _forces_match(build, positions), atol=1e-5)


def test_the_calculator_virial_survives_the_exclusions(example):
    """And the stress, which is where the excluded pairs' own virial shows up."""
    atoms, params = example
    numbers = atoms.get_atomic_numbers()
    cell = np.eye(3) * 9.0
    positions = atoms.get_positions()

    def energy(scaled_positions, scaled_cell):
        work = Atoms(
            numbers=numbers, positions=scaled_positions, cell=scaled_cell, pbc=True
        )
        work.calc = FastForces(work, params)
        return work.get_potential_energy()

    work = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
    work.calc = FastForces(work, params)
    analytic = work.get_stress(voigt=False) * work.get_volume()

    step = 1e-6
    numeric = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            delta = np.zeros((3, 3))
            delta[a, b] = delta[b, a] = step / (1.0 if a == b else 2.0)
            plus, minus = np.eye(3) + delta, np.eye(3) - delta
            numeric[a, b] = (
                energy(positions @ plus.T, cell @ plus.T)
                - energy(positions @ minus.T, cell @ minus.T)
            ) / (2 * step)
    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_a_field_with_no_mask_sums_every_pair(example):
    """`exclusions is None` is "no exclusions", not "derive them for me".

    Worth pinning because the difference is large and silent: a hand-assembled
    `Parameters` gets the unexcluded sums, and the gap between the two is what
    the fitted bond depths would have to absorb.
    """
    atoms, params = example
    bare = Parameters(numbers=params.numbers, terms=params.terms)
    assert bare.exclusions is None

    work = Atoms(numbers=params.numbers, positions=atoms.get_positions())
    work.calc = FastForces(work, params)
    excluded = work.get_potential_energy()

    other = work.copy()
    other.calc = FastForces(other, bare)
    assert other.get_potential_energy() != pytest.approx(excluded)
