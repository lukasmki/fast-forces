"""The coupling fit: what each form's two numbers are pinned by.

Every form answers the same two questions -- what is the coupling worth at the
transition state, and where has it switched off -- so the tests are mostly the
same two assertions in three metrics.  The reference-file test is the one that
is not: it reproduces a fit made elsewhere, bit for bit, from the geometries
alone.
"""

import json

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from fastforces import coupling as C

REFERENCE_XYZ = "quickstart/output/h3o-h2o-transfer.xyz"
REFERENCE_JSONL = "quickstart/output/h3o-h2o-transfer.jsonl"

# Donor, transferring proton, acceptor, in the reference file's atom order.
TRIPLE = (0, 1, 4)


@pytest.fixture
def reference_frames():
    return read(REFERENCE_XYZ, index=":", format="extxyz")


@pytest.fixture
def reference_terms():
    with open(REFERENCE_JSONL) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_reproduces_the_reference_threebody_fit(reference_frames, reference_terms):
    """The whole port, checked against a fit made outside this package.

    Only the amplitude is handed in -- it came from a wB97X-V barrier this
    repository has no calculator for -- so what is being reproduced is the
    width, the centre and the format, from the three geometries alone.  Exact
    equality rather than a tolerance: every one of these is a closed-form
    function of the stored coordinates, so anything but equality is a different
    formula, not a different rounding.
    """
    expected = reference_terms[0]
    fitted = C.fit_threebody(
        reference_frames, TRIPLE, amplitude=expected["kwargs"]["A"]
    )
    row = fitted.to_rows()[0]
    assert row["type"] == expected["type"]
    assert row["atoms"] == expected["atoms"]
    assert row["kwargs"] == expected["kwargs"]


def test_the_coupling_is_the_amplitude_at_the_transition_state(reference_frames):
    """`compute_threebody` is centred on the saddle's own triangle, so `g = 0`
    there and `V = A` exactly -- the one property `fit_amplitude` needs."""
    fitted = C.fit_threebody(reference_frames, TRIPLE, amplitude=-3.5)
    assert fitted.value(reference_frames[1].get_positions()) == pytest.approx(-3.5)


def test_the_coupling_quenches_to_eps_at_the_nearer_endpoint(reference_frames):
    """The width's defining condition, at whichever end is nearer in `g`.

    This channel is symmetric, so both endpoints sit at the same deviation and
    both land on `eps`; an asymmetric one would have the other end below it.
    """
    fitted = C.fit_threebody(reference_frames, TRIPLE, amplitude=-3.5, eps=1e-3)
    for endpoint in (reference_frames[0], reference_frames[-1]):
        assert abs(fitted.value(endpoint.get_positions())) <= 1e-3 + 1e-12


def test_a_wider_eps_gives_a_wider_coupling(reference_frames):
    loose = C.fit_threebody(reference_frames, TRIPLE, amplitude=-3.5, eps=1e-2)
    tight = C.fit_threebody(reference_frames, TRIPLE, amplitude=-3.5, eps=1e-4)
    assert (
        loose.terms["threebody"]["kwargs"]["a"][0]
        < tight.terms["threebody"]["kwargs"]["a"][0]
    )


# ---------------------------------------------------------------------------
# the amplitude
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "h1,h2", [(-10.0, -10.0), (-10.0, -9.0), (-10.0, -12.5), (0.5, -0.25)]
)
def test_amplitude_inverts_the_secular_equation(h1, h2):
    """Round trip: build a ground state from a known coupling, recover it."""
    amplitude = -1.75
    mean, half_gap = 0.5 * (h1 + h2), 0.5 * (h1 - h2)
    barrier = mean - np.hypot(half_gap, amplitude)
    assert C.fit_amplitude(h1, h2, barrier) == pytest.approx(amplitude)


def test_amplitude_rejects_a_barrier_above_a_diabat():
    """The EVB ground state is below every diabat by construction."""
    with pytest.raises(C.CouplingError, match="above a diabatic energy"):
        C.fit_amplitude(-10.0, -12.0, -9.0)


def test_amplitude_rejects_a_barrier_between_the_two_diabats():
    """A barrier between the two diabats is above the lower one, so the ordering
    check catches it before the square root does.

    The discriminant test behind it is unreachable except at exact degeneracy,
    where it is a rounding guard: with the barrier below both diabats,
    `mean - barrier` always exceeds `|half_gap|`.
    """
    with pytest.raises(C.CouplingError, match="above a diabatic energy"):
        C.fit_amplitude(-10.0, -12.0, -11.0)


def test_degenerate_diabats_give_the_barrier_depth_as_the_amplitude():
    assert C.fit_amplitude(-10.0, -10.0, -10.4) == pytest.approx(-0.4)


def test_a_supplied_amplitude_is_not_recorded_as_a_fit(reference_frames):
    assert C.fit_threebody(reference_frames, TRIPLE, amplitude=-1.0).provenance == {
        "threebody": "placeholder"
    }
    assert C.fit_threebody(reference_frames, TRIPLE, amplitude=0.0).provenance == {
        "threebody": "decoupled"
    }


def test_a_decoupled_channel_still_gets_a_readable_width(reference_frames):
    """Zero amplitude has no width -- every width satisfies the condition -- so
    the width a plausible amplitude would have needed is stored instead."""
    fitted = C.fit_threebody(reference_frames, TRIPLE, amplitude=0.0)
    assert np.isfinite(fitted.terms["threebody"]["kwargs"]["a"][0])
    assert fitted.terms["threebody"]["kwargs"]["a"][0] > 0.0


# ---------------------------------------------------------------------------
# the rmsd metric
# ---------------------------------------------------------------------------


def test_rmsd_is_invariant_to_rigid_motion(random_positions):
    """The metric is superposed, so a translated and rotated copy is at zero."""
    angle = 0.7
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = random_positions @ rotation.T + np.array([3.0, -1.0, 0.5])
    assert C.rmsd(random_positions, moved) == pytest.approx(0.0, abs=1e-9)


def test_rmsd_coupling_is_the_amplitude_at_its_own_template(reference_frames):
    fitted = C.fit_rmsd(reference_frames, amplitude=-2.0)
    assert fitted.value(reference_frames[1].get_positions()) == pytest.approx(-2.0)


def test_rmsd_coupling_quenches_at_the_endpoints(reference_frames):
    fitted = C.fit_rmsd(reference_frames, amplitude=-2.0, eps=1e-3)
    for endpoint in (reference_frames[0], reference_frames[-1]):
        assert abs(fitted.value(endpoint.get_positions())) <= 1e-3 + 1e-12


def test_a_spectator_switches_the_rmsd_form_off_and_not_the_triangle(
    reference_frames,
):
    """Why the transfer channels left the RMSD form.

    The transfer atoms are held exactly at the transition state and only the
    four spectator hydrogens are displaced, by 0.173 A each -- less than thermal
    motion at 300 K.  An RMSD is a tolerance on all `3N` coordinates at once, so
    it collapses: -2.000 eV to -1.088, with none of the coordinates the reaction
    is a function of having moved.  `g` sees only the proton and its two
    oxygens, so the triangle form does not move at all.

    The gap is wider still on a channel whose endpoints are nearer its saddle in
    RMSD -- that is what sets the width -- which is why this form was replaced
    for transfers rather than merely supplemented.
    """
    transition = reference_frames[1]
    nudged = transition.copy()
    positions = nudged.get_positions()
    spectators = [i for i in range(len(nudged)) if i not in TRIPLE]
    positions[spectators] += 0.1
    nudged.set_positions(positions)

    by_rmsd = C.fit_rmsd(reference_frames, amplitude=-2.0)
    by_triangle = C.fit_threebody(reference_frames, TRIPLE, amplitude=-2.0)
    assert by_rmsd.value(positions) == pytest.approx(-1.088, abs=1e-3)
    assert by_rmsd.value(transition.get_positions()) == pytest.approx(-2.0)
    assert by_triangle.value(positions) == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# gradients and the format
# ---------------------------------------------------------------------------


def _numeric_forces(coupling, positions, step=1e-6):
    forces = np.zeros_like(positions)
    for i in range(len(positions)):
        for axis in range(3):
            shifted = []
            for sign in (+1, -1):
                probe = positions.copy()
                probe[i, axis] += sign * step
                shifted.append(coupling.value(probe))
            forces[i, axis] = -(shifted[0] - shifted[1]) / (2 * step)
    return forces


@pytest.mark.parametrize("form", ["rmsd", "threebody", "twobody"])
def test_coupling_forces_match_the_numerical_gradient(form, reference_frames):
    """Every form returns `-dV/dx`, checked away from both its centre and its
    tail so the Gaussian is neither flat nor vanishing."""
    if form == "twobody":
        coupling = C.Coupling(
            terms={
                "twobody": {
                    "atoms": np.array([[0, 4]]),
                    "kwargs": {
                        "A": np.array([-2.0]),
                        "a": np.array([1.5]),
                        "r0": np.array([2.6]),
                    },
                }
            }
        )
    elif form == "threebody":
        coupling = C.fit_threebody(reference_frames, TRIPLE, amplitude=-2.0, eps=0.3)
    else:
        coupling = C.fit_rmsd(reference_frames, amplitude=-2.0, eps=0.3)

    rng = np.random.default_rng(11)
    positions = reference_frames[1].get_positions() + rng.normal(
        scale=0.08, size=(len(reference_frames[1]), 3)
    )
    if coupling.ensemble is None:
        coupling.ensemble = np.array([reference_frames[1].get_positions()])

    _, forces, _ = coupling(positions)
    assert np.allclose(forces, _numeric_forces(coupling, positions), atol=1e-5)


def test_jsonl_round_trip_is_exact(tmp_path, reference_frames):
    """Coupling parameters are stored unconverted, so the round trip is
    equality and not a tolerance -- see `coupling`'s note on units."""
    fitted = C.fit_threebody(reference_frames, TRIPLE, amplitude=-3.5)
    path = tmp_path / "coupling.jsonl"
    fitted.to_jsonl(str(path))
    back = C.Coupling.from_jsonl(str(path))
    assert back.provenance == fitted.provenance
    for name, value in fitted.terms["threebody"]["kwargs"].items():
        assert back.terms["threebody"]["kwargs"][name] == pytest.approx(value, abs=0.0)
    assert np.array_equal(
        back.terms["threebody"]["atoms"], fitted.terms["threebody"]["atoms"]
    )


def test_too_few_frames_is_an_error():
    frames = [Atoms("H2", positions=np.zeros((2, 3)))] * 2
    with pytest.raises(C.CouplingError, match="reactant, transition state and product"):
        C.fit_rmsd(frames, amplitude=-1.0)


def test_fitting_an_amplitude_needs_the_diabatic_energies(reference_frames):
    with pytest.raises(C.CouplingError, match="needs the diabatic energies"):
        C.fit_threebody(reference_frames, TRIPLE)
