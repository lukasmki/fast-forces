"""Gradient and convention tests for the energy terms themselves.

There is no unit boundary anywhere in `forcefield`: all four evaluators take
Angstrom and eV and return Angstrom and eV, so every parameter below is stated
in those and `__call__` and the per-term `compute_*` methods can be checked
against each other directly.  `test_call_does_not_convert` pins that, because
the alternative -- a conversion hidden inside `__call__` -- is exactly the shape
of bug that shows up only as a force field that disagrees with its own fit.
The jsonl's nm and kJ/mol are `export/units.py`'s business alone.
"""

import warnings

import numpy as np
import pytest

from fastforces.forcefield.acks2 import ACKS2
from fastforces.forcefield.coulomb import Coulomb
from fastforces.forcefield.lj import CORE_FRACTION, LennardJones
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

    forces = qforce(random_positions, PBC, CELL, terms)[1]
    numeric = _numeric_forces(energy_of, random_positions)
    assert np.allclose(forces, numeric, atol=1e-5)


def _lj_terms(n, sigma=3.0, eps=0.01):
    """Per-atom sigma in Angstrom and epsilon in eV, which is what `lj` reads."""
    return {
        "lennardjones": {
            "atoms": np.arange(n)[:, None],
            "kwargs": {"sigma": np.full(n, sigma), "eps": np.full(n, eps)},
        }
    }


def test_lennardjones_gradient(random_positions):
    """Covers the switch and the short-range tangent, not just the bare 12-6.

    `random_positions` is a cloud about 1.6 A across, so most pairs sit inside
    `SWITCH_RADIUS` and some inside `CORE_FRACTION * sigma` -- exactly the two
    branches `pair_potential` added, and the ones a plain 12-6 test would miss.
    """
    lj = LennardJones()
    terms = _lj_terms(len(random_positions))

    separations = np.linalg.norm(
        random_positions[:, None, :] - random_positions[None, :, :], axis=-1
    )
    core = CORE_FRACTION * 3.0
    off = separations[~np.eye(len(random_positions), dtype=bool)]
    assert off.min() < core < off.max(), "the cloud has to straddle the core radius"

    forces = lj(random_positions, PBC, CELL, terms)[1]
    numeric = _numeric_forces(lambda p: lj(p, PBC, CELL, terms)[0], random_positions)
    assert np.allclose(forces, numeric, atol=1e-6)


def test_lennardjones_takes_no_exclusions(random_positions):
    """The property the whole term rests on: every pair counts, bonded or not.

    It is checked as the absence of a constructor argument *and* as a number,
    because a mask reintroduced anywhere would show up here as a bonded pair
    contributing nothing.
    """
    lj = LennardJones()
    pair = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])  # 2.5 A, past the switch
    energy = lj(pair, PBC, CELL, _lj_terms(2))[0]
    assert energy != 0.0


def test_lennardjones_is_bounded_inside_the_core():
    """`CORE_FRACTION` replaces `r**-12` with its own tangent.

    Without it a pair driven to a hundredth of sigma contributes 1e21 eV and
    takes the rest of the energy with it -- and, worse, takes the precision of
    every other pair with it, since the sum is then quantized in units of 2**22.
    The tangent bounds it instead.

    Only boundedness is asserted here, not the gradient: at 0.02 A the energy
    is a switch of 1e-8 times a core value of 1e6, so a float64 finite
    difference of it is rounding noise (it misses the exact derivative by 3.7%,
    which was checked against exact arithmetic).  `test_lennardjones_gradient`
    covers the tangent branch at separations where a difference quotient means
    something.
    """
    lj = LennardJones()
    terms = _lj_terms(2)
    for separation in (0.02, 0.2, 0.5):
        close = np.array([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]])
        energy, forces, _ = lj(close, PBC, CELL, terms)
        assert np.isfinite(energy) and abs(energy) < 1e4
        assert np.all(np.isfinite(forces))


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

    forces = ACKS2()(random_positions, PBC, CELL, terms)[1]
    numeric = _numeric_forces(energy_of, random_positions, step=1e-5)
    assert np.allclose(forces, numeric, atol=1e-5)


NUMBERS = np.array([8, 8, 1, 1, 6, 7, 1, 1])

# Charges that sum to zero, which is what `ewald` assumes of a periodic sum.
CHARGES = np.array([-0.5, -0.4, 0.25, 0.25, 0.3, -0.35, 0.2, 0.25])


def _coulomb_terms(charges=CHARGES, indices=None):
    charges = np.asarray(charges, dtype=float)
    if indices is None:
        indices = np.arange(len(charges))
    return {
        "coulomb": {
            "atoms": np.asarray(indices)[:, None],
            "kwargs": {"q": charges},
        }
    }


def test_coulomb_gradient(random_positions):
    coulomb = Coulomb()
    terms = _coulomb_terms()
    forces = coulomb(random_positions, PBC, CELL, terms)[1]
    numeric = _numeric_forces(
        lambda p: Coulomb()(p, PBC, CELL, terms)[0], random_positions
    )
    assert np.allclose(forces, numeric, atol=1e-6)


def test_coulomb_needs_no_response_term(random_positions):
    """The whole point of the term: fixed `q` makes `coulomb_sum` the gradient.

    `ACKS2` needs `compute_response_forces` on top of the same sum because its
    charges move with the geometry.  This asserts the other half of that
    statement -- that at fixed charges the explicit gradient is already exact --
    by checking `ACKS2.compute_coulomb`, the shared piece, against
    `Coulomb.__call__` on the same charges.
    """
    terms = _coulomb_terms()
    vecs = random_positions[:, None, :] - random_positions[None, :, :]
    rij = np.sqrt(np.sum(vecs * vecs, -1))
    explicit = ACKS2().compute_coulomb(CHARGES, rij, vecs)
    whole = Coulomb()(random_positions, PBC, CELL, terms)

    assert whole[0] == pytest.approx(explicit[0])
    assert np.allclose(whole[1], explicit[1])
    assert np.allclose(whole[2], explicit[2])


def test_coulomb_reproduces_the_acks2_energy_at_the_solved_charges(random_positions):
    """Both terms sum the same kernel, so the energy agrees where `Q` does.

    This is what makes the fixed-charge term a *replacement* for ACKS2 rather
    than a second, differently-screened electrostatics: hand it the charges
    ACKS2 solved for and it reproduces ACKS2's energy exactly.  The forces do
    not follow, and are not compared -- ACKS2's carry the `dQ/dr` response that
    fixed charges have no counterpart for.
    """
    from fastforces.elements import acks2_defaults

    terms = {
        "atom": {
            "atoms": np.arange(len(NUMBERS))[:, None],
            "kwargs": acks2_defaults(NUMBERS),
        }
    }
    acks2 = ACKS2()
    reference = acks2(random_positions, PBC, CELL, terms)[0]

    fixed = Coulomb()(random_positions, PBC, CELL, _coulomb_terms(acks2.Q))[0]
    assert fixed == pytest.approx(reference, rel=1e-12)


def test_coulomb_publishes_its_charges(random_positions):
    """`.Q` is how the calculator reads charges off either electrostatic term."""
    coulomb = Coulomb()
    coulomb(random_positions, PBC, CELL, _coulomb_terms())
    assert np.allclose(coulomb.Q, CHARGES)


def test_coulomb_skips_atoms_with_no_term(random_positions):
    """Only the atoms that carry a `coulomb` term are in the sum."""
    subset = np.array([0, 1, 2, 3])
    terms = _coulomb_terms(CHARGES[subset] - CHARGES[subset].mean(), subset)
    energy, forces, _ = Coulomb()(random_positions, PBC, CELL, terms)

    assert np.allclose(forces[4:], 0.0)
    # and it is the same number as the same four atoms on their own
    alone = Coulomb()(
        random_positions[subset],
        PBC,
        CELL,
        _coulomb_terms(CHARGES[subset] - CHARGES[subset].mean()),
    )[0]
    assert energy == pytest.approx(alone)


def test_coulomb_warns_once_on_a_charged_periodic_system(random_positions):
    """The Ewald `k = 0` term is omitted, which needs `sum q = 0`."""
    charged = CHARGES + 1.0 / len(CHARGES)
    coulomb = Coulomb()
    terms = _coulomb_terms(charged)
    with pytest.warns(UserWarning, match="net charge"):
        coulomb(random_positions, STRAIN_PBC, STRAIN_CELL, terms)
    # once per evaluator, not once per step of an MD run
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coulomb(random_positions, STRAIN_PBC, STRAIN_CELL, terms)


def test_coulomb_does_not_warn_under_open_boundaries(random_positions):
    """An isolated ion is an ordinary thing to evaluate; there is no `k = 0`."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Coulomb()(random_positions, PBC, CELL, _coulomb_terms(CHARGES + 0.125))


def test_angle_convention(random_positions):
    """`compute_angle` is `0.5*k*(cos(theta)-cos(theta0))^2`, 1/2 included.

    The 1/2 is the convention the example XML and jsonl are written in, so an
    angle `k` read from one of them means the same well here, in
    `export.openmm.ANGLE` and in the file it came from.  `export.units` has
    nothing to rescale, which is what `test_angle_k_is_exported_unscaled`
    asserts from the other side.
    """
    theta0, k = 1.9, 3.0
    qforce = QForce()
    terms = _term_dict("angle", (0, 1, 2), {"theta0": theta0, "k": k})
    energy = qforce(random_positions, PBC, CELL, terms)[0]

    a = random_positions[0] - random_positions[1]
    b = random_positions[2] - random_positions[1]
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert energy == pytest.approx(0.5 * k * (cos - np.cos(theta0)) ** 2)


def test_call_does_not_convert(random_positions):
    """`__call__` is Angstrom and eV in, Angstrom and eV out.

    `fit` calls the `compute_*` methods directly, with parameters straight out
    of `Parameters.terms`, while the calculator reaches the same numbers through
    `__call__`.  A conversion in either place and not the other would mean the
    force field is fit under one scale and evaluated under another, which is
    invisible in a gradient test because both halves would scale together.  So
    the two paths are checked against literally the same numbers.
    """
    qforce = QForce()
    kwargs = {"D": 4.0, "r0": 1.1, "k": 30.0, "c": 0.4}
    energy, forces, _ = qforce(
        random_positions, PBC, CELL, _term_dict("bond", (0, 1), kwargs)
    )

    vecs = random_positions[:, None, :] - random_positions[None, :, :]
    raw_e, raw_f, _ = qforce.compute_bond(
        vecs,
        np.array([[0, 1]]),
        **{name: np.array([value]) for name, value in kwargs.items()},
    )
    assert energy == pytest.approx(raw_e)
    assert np.allclose(forces, raw_f)


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
    e_unit, f_unit, _ = qforce(
        random_positions, PBC, CELL, _term_dict(term, indices, unit)
    )
    e_scaled, f_scaled, _ = qforce(
        random_positions, PBC, CELL, _term_dict(term, indices, scaled)
    )
    assert e_scaled == pytest.approx(scale * e_unit)
    assert np.allclose(f_scaled, scale * f_unit)


def test_harmonic_bond_is_linear_in_k(random_positions):
    indices, kwargs = TERMS["bond"]
    qforce = QForce(bond_form="harmonic")
    scale = 7.5
    e_unit, f_unit, _ = qforce(
        random_positions, PBC, CELL, _term_dict("bond", indices, dict(kwargs, k=1.0))
    )
    e_scaled, f_scaled, _ = qforce(
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
    e_unit = qforce(
        random_positions, PBC, CELL, _term_dict("bond", indices, dict(kwargs, k=1.0))
    )[0]
    e_scaled = qforce(
        random_positions, PBC, CELL, _term_dict("bond", indices, dict(kwargs, k=scale))
    )[0]
    assert e_scaled != pytest.approx(scale * e_unit)


def test_bond_forms_share_curvature_at_r0():
    """Morse and harmonic have the same second derivative at `r0`.

    That is what lets a `k` obtained under one form be used under the other.
    """
    kwargs = {
        name: np.array([v])
        for name, v in {"D": 4.0, "r0": 1.1, "k": 30.0, "c": 0.0}.items()
    }
    step = 1e-4
    curvatures = []
    for form in ("morse", "harmonic"):
        qforce = QForce(bond_form=form)

        def energy(r):
            # `compute_bond` rather than `__call__`, which is how `fit`
            # calls it -- and, since neither converts anything, the two would
            # give the same curvature either way.
            pos = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
            vecs = pos[:, None, :] - pos[None, :, :]
            return qforce.compute_bond(vecs, np.array([[0, 1]]), **kwargs)[0]

        r0 = float(kwargs["r0"][0])
        curvatures.append(
            (energy(r0 + step) - 2 * energy(r0) + energy(r0 - step)) / step**2
        )
    assert curvatures[0] == pytest.approx(curvatures[1], rel=1e-4)
    assert curvatures[0] == pytest.approx(float(kwargs["k"][0]), rel=1e-4)


# ---------------------------------------------------------------------------
# virials
# ---------------------------------------------------------------------------

# A cell big enough that the minimum image of the cloud is the cloud itself, so
# the virial being tested is the open-boundary one plus, for ACKS2, a real Ewald
# lattice sum.
STRAIN_CELL = np.eye(3) * 9.0
STRAIN_PBC = np.ones(3, dtype=bool)


def _numeric_virial(energy_of, positions, step=1e-6):
    """`dE/de_ab` by central differences on a homogeneous strain.

    Both the positions and the cell are mapped by `I + e`, which is what makes
    it the same derivative the analytic expressions take: every displacement
    vector, minimum image included, transforms the same way.
    """
    virial = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            delta = np.zeros((3, 3))
            delta[a, b] = delta[b, a] = step / (1.0 if a == b else 2.0)
            plus = np.eye(3) + delta
            minus = np.eye(3) - delta
            e_plus = energy_of(positions @ plus.T, STRAIN_CELL @ plus.T)
            e_minus = energy_of(positions @ minus.T, STRAIN_CELL @ minus.T)
            virial[a, b] = (e_plus - e_minus) / (2 * step)
    return virial


def test_zbl_virial(random_positions):
    zbl = ZBL()
    numbers = np.array([8, 8, 1, 1, 6, 7, 1, 1])
    analytic = zbl(random_positions, numbers, STRAIN_PBC, STRAIN_CELL)[2]
    numeric = _numeric_virial(
        lambda p, c: zbl(p, numbers, STRAIN_PBC, c)[0], random_positions
    )
    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-8)


def test_lennardjones_virial(random_positions):
    lj = LennardJones()
    terms = _lj_terms(len(random_positions))
    analytic = lj(random_positions, STRAIN_PBC, STRAIN_CELL, terms)[2]
    numeric = _numeric_virial(
        lambda p, c: lj(p, STRAIN_PBC, c, terms)[0], random_positions
    )
    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-8)


def test_acks2_virial(random_positions):
    """The one that exercises `ewald.EwaldKernel.contract`.

    With all three directions periodic the kernel is a lattice sum, and its
    strain derivative reaches the energy through the `1/V` prefactor and through
    the reciprocal vectors as well as through the pair separations -- none of
    which the open-boundary expression has a counterpart for.  A fresh solver per
    evaluation, because the charge cache is keyed on the geometry and the cell.
    """
    from fastforces.elements import acks2_defaults

    numbers = np.array([8, 8, 1, 1, 6, 7, 1, 1])
    terms = {
        "atom": {
            "atoms": np.arange(len(numbers))[:, None],
            "kwargs": acks2_defaults(numbers),
        }
    }
    analytic = ACKS2()(random_positions, STRAIN_PBC, STRAIN_CELL, terms)[2]
    numeric = _numeric_virial(
        lambda p, c: ACKS2()(p, STRAIN_PBC, c, terms)[0], random_positions
    )
    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-8)


def test_coulomb_virial(random_positions):
    """Fully periodic, so this is the Ewald lattice sum's strain derivative."""
    coulomb = Coulomb()
    terms = _coulomb_terms()
    analytic = coulomb(random_positions, STRAIN_PBC, STRAIN_CELL, terms)[2]
    numeric = _numeric_virial(
        lambda p, c: Coulomb()(p, STRAIN_PBC, c, terms)[0], random_positions
    )
    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-8)


def test_coulomb_periodic_gradient(random_positions):
    """The force under the lattice sum, which `MinimumImage` does not cover."""
    terms = _coulomb_terms()
    forces = Coulomb()(random_positions, STRAIN_PBC, STRAIN_CELL, terms)[1]
    numeric = _numeric_forces(
        lambda p: Coulomb()(p, STRAIN_PBC, STRAIN_CELL, terms)[0],
        random_positions,
        step=1e-5,
    )
    assert np.allclose(forces, numeric, atol=1e-6)


@pytest.mark.parametrize("term", sorted(TERMS))
def test_qforce_virials(term, random_positions):
    """Every bonded term's `_virial` is the strain derivative of its own energy."""
    indices, kwargs = TERMS[term]
    qforce = QForce()
    terms = _term_dict(term, indices, kwargs)

    analytic = qforce(random_positions, STRAIN_PBC, STRAIN_CELL, terms)[2]
    numeric = _numeric_virial(
        lambda p, c: qforce(p, STRAIN_PBC, c, terms)[0], random_positions
    )
    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-8)
