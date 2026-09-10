"""Gradient and convention tests for the energy terms themselves."""

import numpy as np
import pytest

from fastforces.forcefield.acks2 import ACKS2
from fastforces.forcefield.lj import LennardJones
from fastforces.forcefield.qforce import QForce
from fastforces.forcefield.zbl import ZBL

PBC = np.zeros(3, dtype=bool)
CELL = np.eye(3)

# One representative term of each type, on an 8-atom cloud, with the kwargs the
# corresponding `compute_*` takes.  `k` is small so the clamped cross terms stay
# on their linear branch.
TERMS = {
    "bond": ((0, 1), {"D": 4.0, "r0": 1.1, "k": 30.0, "c": 0.4}),
    "angle": ((0, 1, 2), {"theta0": 1.9, "k": 3.0}),
    "bondbond": ((0, 1, 1, 2), {"r1_0": 1.1, "r2_0": 1.2, "k": 0.7}),
    "bondangle": ((0, 1, 2, 1, 2), {"theta0": 1.9, "r0": 1.2, "k": 0.5}),
    "angleangle": ((0, 1, 2, 3, 4, 5), {"theta1_0": 1.9, "theta2_0": 2.0, "k": 0.6}),
    "periodicdihedral": ((0, 1, 2, 3), {"phi0": np.pi, "n": 3.0, "k": 0.8}),
    "dihedralbond": ((0, 1, 2, 3, 4, 5), {"phi0": 0.0, "n": 2.0, "k": 0.4, "r0": 1.3}),
    "dihedralangle": (
        (0, 1, 2, 3, 4, 5, 6),
        {"phi0": 0.0, "n": 2.0, "k": 0.4, "theta0": 1.9},
    ),
    "dihedralangleangle": (
        (0, 1, 2, 3),
        {"phi0": 0.0, "n": 1.0, "k": 0.5, "theta0_1": 1.9, "theta0_2": 2.0},
    ),
}


def _term_dict(term, indices, kwargs):
    return {
        term: {
            "atoms": np.array([indices]),
            "kwargs": {k: np.array([v]) for k, v in kwargs.items()},
        }
    }


def _numeric_forces(energy_of, positions, step=1e-6):
    forces = np.zeros_like(positions)
    for i in range(positions.shape[0]):
        for axis in range(3):
            plus, minus = positions.copy(), positions.copy()
            plus[i, axis] += step
            minus[i, axis] -= step
            forces[i, axis] = -(energy_of(plus) - energy_of(minus)) / (2 * step)
    return forces


@pytest.mark.parametrize("term", sorted(TERMS))
def test_qforce_gradients(term, random_positions):
    """Every analytic force is the gradient of its own energy."""
    indices, kwargs = TERMS[term]
    qforce = QForce()
    terms = _term_dict(term, indices, kwargs)

    def energy_of(pos):
        return qforce(pos, PBC, CELL, terms)[0]

    _, forces = qforce(random_positions, PBC, CELL, terms)
    numeric = _numeric_forces(energy_of, random_positions)
    assert np.allclose(forces, numeric, atol=1e-5)


def test_lennardjones_gradient(random_positions):
    n = len(random_positions)
    terms = {
        "lennardjones": {
            "atoms": np.arange(n)[:, None],
            "kwargs": {
                "sigma": np.full(n, 3.0),
                "eps": np.full(n, 0.01),
            },
        }
    }
    exclusions = np.zeros((n, n), dtype=bool)
    exclusions[0, 1] = exclusions[1, 0] = True
    lj = LennardJones(exclusions)

    _, forces = lj(random_positions, PBC, CELL, terms)
    numeric = _numeric_forces(lambda p: lj(p, PBC, CELL, terms)[0], random_positions)
    assert np.allclose(forces, numeric, atol=1e-6)


def test_zbl_gradient(random_positions):
    numbers = np.array([8, 8, 1, 1, 6, 7, 1, 1])
    zbl = ZBL()
    _, forces, _ = zbl(random_positions, numbers, PBC, CELL)
    numeric = _numeric_forces(lambda p: zbl(p, numbers, PBC, CELL)[0], random_positions)
    assert np.allclose(forces, numeric, atol=1e-5)


def test_acks2_gradient(random_positions):
    """Covers the charge-response force, which is easy to lose silently."""
    from fastforces.elements import acks2_defaults

    numbers = np.array([8, 8, 1, 1, 6, 7, 1, 1])
    n = len(numbers)
    terms = {
        "atom": {
            "atoms": np.arange(n)[:, None],
            "kwargs": acks2_defaults(numbers),
        }
    }

    def energy_of(pos):
        # a fresh solver each time: the cache is keyed on the geometry, and a
        # stale charge set would hide exactly the error this test looks for
        return ACKS2()(pos, PBC, CELL, terms)[0]

    _, forces = ACKS2()(random_positions, PBC, CELL, terms)
    numeric = _numeric_forces(energy_of, random_positions, step=1e-5)
    assert np.allclose(forces, numeric, atol=1e-5)


def test_angle_convention(random_positions):
    """`compute_angle` is `0.5*k*(cos(theta)-cos(theta0))^2`, not `k*(...)^2`.

    The example XML and jsonl both carry the 0.5 convention, so the evaluator
    has to as well or every exported angle is off by a factor of two.
    """
    theta0, k = 1.9, 3.0
    qforce = QForce()
    terms = _term_dict("angle", (0, 1, 2), {"theta0": theta0, "k": k})
    energy, _ = qforce(random_positions, PBC, CELL, terms)

    a = random_positions[0] - random_positions[1]
    b = random_positions[2] - random_positions[1]
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert energy == pytest.approx(0.5 * k * (cos - np.cos(theta0)) ** 2)


# The bond is excluded: the Morse exponent is `sqrt(k / 2D)`, so a Morse bond is
# not linear in `k`.  `test_morse_bond_is_not_linear_in_k` pins that, and it is
# the reason `fit` handles bonds in a separate nonlinear block.
LINEAR_TERMS = sorted(set(TERMS) - {"bond"})


@pytest.mark.parametrize("term", LINEAR_TERMS)
def test_linear_in_k(term, random_positions):
    """Every bonded term the fitter solves linearly scales exactly with `k`.

    This is the assumption the fitter's design matrix rests on: it builds each
    basis column by evaluating a term at `k = 1`.  If a term ever stops being
    linear in `k`, this is the test that says so.
    """
    indices, kwargs = TERMS[term]
    qforce = QForce()
    scale = 7.5

    unit = dict(kwargs, k=1.0)
    scaled = dict(kwargs, k=scale)
    e_unit, f_unit = qforce(
        random_positions, PBC, CELL, _term_dict(term, indices, unit)
    )
    e_scaled, f_scaled = qforce(
        random_positions, PBC, CELL, _term_dict(term, indices, scaled)
    )
    assert e_scaled == pytest.approx(scale * e_unit)
    assert np.allclose(f_scaled, scale * f_unit)


def test_harmonic_bond_is_linear_in_k(random_positions):
    indices, kwargs = TERMS["bond"]
    qforce = QForce(bond_form="harmonic")
    scale = 7.5
    e_unit, f_unit = qforce(
        random_positions, PBC, CELL, _term_dict("bond", indices, dict(kwargs, k=1.0))
    )
    e_scaled, f_scaled = qforce(
        random_positions, PBC, CELL, _term_dict("bond", indices, dict(kwargs, k=scale))
    )
    assert e_scaled == pytest.approx(scale * e_unit)
    assert np.allclose(f_scaled, scale * f_unit)


def test_morse_bond_is_not_linear_in_k(random_positions):
    """The Morse exponent `sqrt(k/2D)` makes the bond nonlinear in `k`.

    Documented as a test rather than a comment because the fitter's structure
    depends on it: bonds are the one term type that cannot go in the linear
    block.
    """
    indices, kwargs = TERMS["bond"]
    qforce = QForce(bond_form="morse")
    scale = 7.5
    e_unit, _ = qforce(
        random_positions, PBC, CELL, _term_dict("bond", indices, dict(kwargs, k=1.0))
    )
    e_scaled, _ = qforce(
        random_positions, PBC, CELL, _term_dict("bond", indices, dict(kwargs, k=scale))
    )
    assert e_scaled != pytest.approx(scale * e_unit)


def test_bond_forms_share_curvature_at_r0():
    """Morse and harmonic have the same second derivative at `r0`.

    That is what lets a `k` obtained under one form be used under the other.
    """
    kwargs = {"D": 4.0, "r0": 1.1, "k": 30.0, "c": 0.0}
    step = 1e-4
    curvatures = []
    for form in ("morse", "harmonic"):
        qforce = QForce(bond_form=form)

        def energy(r):
            pos = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
            return qforce(pos, PBC, CELL, _term_dict("bond", (0, 1), kwargs))[0]

        r0 = kwargs["r0"]
        curvatures.append(
            (energy(r0 + step) - 2 * energy(r0) + energy(r0 - step)) / step**2
        )
    assert curvatures[0] == pytest.approx(curvatures[1], rel=1e-4)
    assert curvatures[0] == pytest.approx(kwargs["k"], rel=1e-4)
