"""End-to-end fitting against a real reference calculator."""

import numpy as np
import pytest

from fastforces import (
    FitConfig,
    Parameters,
    as_parameters,
    evaluate,
    fit,
    fit_from_file,
    io,
)
from fastforces.topology import enumerate_terms

# Reference equilibrium values for H2O2, in Angstrom and radians.  They came
# from wB97X-V; GFN2-xTB lands close enough that they are a meaningful check on
# the geometry the fit is built on.  The fixture files are acetonitrile -- this
# module fits H2O2 from SMILES at runtime and reads no file, so the two do not
# have to agree.
REFERENCE_R_OH = 0.96566477
REFERENCE_R_OO = 1.42088398
REFERENCE_THETA = 1.78183281

pytestmark = pytest.mark.slow


def test_equilibrium_geometry_matches_the_reference(h2o2_fit):
    _, _, path = h2o2_fit
    equilibrium = io.read_training_set(path).equilibrium
    positions = equilibrium.get_positions()
    graph = enumerate_terms(equilibrium)
    bonds = [tuple(sorted(row)) for row in graph.atoms["bond"]]
    lengths = sorted(np.linalg.norm(positions[i] - positions[j]) for i, j in bonds)
    assert lengths[0] == pytest.approx(REFERENCE_R_OH, abs=0.02)
    assert lengths[-1] == pytest.approx(REFERENCE_R_OO, abs=0.02)


def test_fitted_angle_matches_the_reference(h2o2_fit):
    _, params, _ = h2o2_fit
    theta0 = params.terms["angle"]["kwargs"]["theta0"]
    assert np.allclose(theta0, theta0[0])  # one shared class
    assert theta0[0] == pytest.approx(REFERENCE_THETA, abs=np.radians(3))


def test_bond_r0_is_compressed_below_the_true_length(h2o2_fit):
    """The design property: `r0` balances the nonbonded baseline.

    `ZBL` contributes ~20 eV/A of repulsion at a normal bond length and has no
    switching function, so the bond has to be pre-compressed to pull back.  A fitted
    H2O2 force field's O-H `r0` is 0.70 A against a true 0.97 A for the same
    reason.
    """
    _, params, path = h2o2_fit
    equilibrium = io.read_training_set(path).equilibrium
    positions = equilibrium.get_positions()
    for row, r0 in zip(
        params.terms["bond"]["atoms"], params.terms["bond"]["kwargs"]["r0"], strict=True
    ):
        true_length = np.linalg.norm(positions[row[0]] - positions[row[1]])
        assert r0 < true_length


def test_symmetry_equivalent_bonds_share_parameters(h2o2_fit):
    _, params, _ = h2o2_fit
    kwargs = params.terms["bond"]["kwargs"]
    rows = [tuple(sorted(r)) for r in params.terms["bond"]["atoms"]]
    first, second = rows.index((0, 2)), rows.index((1, 3))
    for name in ("r0", "k", "D", "c"):
        assert kwargs[name][first] == kwargs[name][second]


def test_fit_accuracy(h2o2_fit):
    _, params, _ = h2o2_fit
    assert params.report["energy_rmse_eV"] < 0.05
    assert params.report["force_rmse_eV_A"] < 0.5


def test_reproducible_from_the_training_file_alone(h2o2_fit):
    """The README's headline claim: the file is enough to reproduce the fit.

    Not bit-for-bit -- extxyz stores forces as text, and the Hessian divides
    those differences by `2*delta`, so the last couple of digits are lost. The
    fit reproduces to well inside that.
    """
    _, params, path = h2o2_fit
    again = fit_from_file(path)
    for term, block in params.terms.items():
        for name, value in block["kwargs"].items():
            assert np.allclose(value, again.terms[term]["kwargs"][name], rtol=1e-6), (
                f"{term}.{name} did not reproduce"
            )
    assert again.e0 == pytest.approx(params.e0, rel=1e-9)


def _score(params, frames):
    energy_errors, force_errors = [], []
    for frame in frames:
        energy, forces = evaluate(frame, params)
        energy_errors.append(energy - frame.get_potential_energy())
        force_errors.append((forces - frame.get_forces()).reshape(-1))
    # Energies are scored by their spread, not their offset: `E0` is fit, so
    # only the shape of the profile is under test.
    return (
        float(np.std(energy_errors)),
        float(np.sqrt(np.mean(np.concatenate(force_errors) ** 2))),
    )


def test_held_out_torsion_angles(h2o2_fit):
    """Train on every other torsion angle, score on the ones in between.

    This is the generalization that matters in practice: the default pipeline
    always scans the torsions, so the question is whether the fitted dihedral
    series interpolates the profile rather than memorizing the scan points.
    """
    _, _, path = h2o2_fit
    data = io.read_training_set(path)
    torsions = data.of_kind("torsion")
    assert len(torsions) >= 8, "expected a torsion scan in the training set"

    training = io.TrainingSet(
        frames=data.of_kind("equilibrium", "hessian", "mode") + torsions[::2],
        meta=data.meta,
    )
    params = fit(training, enumerate_terms(training.equilibrium), FitConfig())

    energy_spread, force_rmse = _score(params, torsions[1::2])
    assert energy_spread < 0.05
    assert force_rmse < 0.5


def test_torsion_data_is_required_for_the_dihedral_terms(h2o2_fit):
    """Dropping the torsion scan entirely wrecks the dihedral terms.

    Recorded as a test because it is a real requirement on the pipeline rather
    than a tuning detail: the Hessian only sees the torsion profile to second
    order about the minimum, so nothing else in the training set constrains the
    `n = 1..4` series away from it.  Without the scan the fit still converges --
    it just extrapolates badly at the cis barrier, which is exactly the kind of
    failure that is invisible unless something checks for it.

    Scored as a *ratio* against the field that did see the scan, rather than
    against an absolute threshold.  The absolute number is not a property of the
    pipeline alone: the unexcluded `LennardJones` supplies part of the H2O2
    torsion profile directly -- the H...H 1-4 pair at ~2.7 A is the torsion
    coordinate, and it is now in the fixed nonbonded baseline rather than being
    absorbed into the fitted Fourier series -- so what the dihedral terms have
    to extrapolate is smaller than it was, and the raw score fell below the 1.0
    this used to assert.  The gap the test is really about is untouched:
    measured, 0.339 eV without the scan against 0.00068 eV with it, a factor of
    495.
    """
    _, with_scan_params, path = h2o2_fit
    data = io.read_training_set(path)
    torsions = data.of_kind("torsion")

    training = io.TrainingSet(
        frames=data.of_kind("equilibrium", "hessian", "mode"), meta=data.meta
    )
    no_scan_params = fit(training, enumerate_terms(training.equilibrium), FitConfig())

    without_scan, _ = _score(no_scan_params, torsions)
    with_scan, _ = _score(with_scan_params, torsions)
    assert without_scan > 50 * with_scan, (
        f"torsion extrapolation is unexpectedly good without the scan "
        f"({without_scan:.4g} against {with_scan:.4g} with it); if sampling "
        "changed so that the dihedral terms are constrained without a scan, "
        "this test should go"
    )


def test_openmm_matches_the_fitted_field(h2o2_fit):
    pytest.importorskip("openmm")
    from fastforces.export.openmm import system_energy

    _, params, path = h2o2_fit
    equilibrium = io.read_training_set(path).equilibrium
    positions = equilibrium.get_positions()
    system = params.to_openmm_system(positions=positions)
    reference, _ = evaluate(equilibrium, params)
    assert system_energy(system, positions) == pytest.approx(reference, rel=1e-8)


# ---------------------------------------------------------------------------
# starting from an existing force field
# ---------------------------------------------------------------------------


def _worst_relative_change(a, b) -> tuple[float, str]:
    """Largest change in any parameter, scaled by that parameter's own size."""
    worst = (0.0, "")
    for term, block in a.terms.items():
        for name, value in block["kwargs"].items():
            value = np.asarray(value, dtype=float)
            other = np.asarray(b.terms[term]["kwargs"][name], dtype=float)
            scale = max(float(np.abs(value).max()), 1e-12)
            worst = max(
                worst, (float(np.abs(other - value).max() / scale), f"{term}.{name}")
            )
    return worst


def test_a_starting_point_is_reported_and_lands_on_every_class(h2o2_fit):
    """A fitted field covers its own topology exactly, so nothing falls back."""
    _, params, path = h2o2_fit
    topology = enumerate_terms(io.read_training_set(path).equilibrium)
    again = fit_from_file(path, initial=params)

    bonded = sum(topology.n_classes(term) for term in topology.atoms)
    nonbonded = len(params.numbers) * len(["atom", "lennardjones"])
    assert again.report["seeded"] == bonded + nonbonded
    assert "seeded" not in params.report  # only reported when there was one


def test_a_starting_point_round_trips_through_jsonl(tmp_path, h2o2_fit):
    """The realistic path: fit, write the jsonl, start the next fit from it.

    The two fits are scored on what they predict, not parameter by parameter.
    The jsonl round trip perturbs the starting point by ~1e-16 -- the round-off
    of multiplying by a unit factor and dividing it out again -- and the
    alternation stops on a relative tolerance, so the shallowest parameters
    (`bond.c`, which the data barely constrains) land a few tenths of a percent
    apart.  The force fields those parameters describe are the same one: the
    RMSEs agree to five or six significant figures, against a difference in the
    second figure if the starting point had actually changed the answer.
    """
    _, params, path = h2o2_fit
    jsonl = tmp_path / "h2o2.jsonl"
    params.to_jsonl(str(jsonl))
    change, _ = _worst_relative_change(params, as_parameters(str(jsonl)))
    assert change < 1e-12, "the jsonl round trip itself should be near-exact"

    from_object = fit_from_file(path, initial=params)
    from_file = fit_from_file(path, initial=str(jsonl))
    for key in ("energy_rmse_eV", "force_rmse_eV_A"):
        assert from_file.report[key] == pytest.approx(from_object.report[key], rel=1e-4)
    change, where = _worst_relative_change(from_object, from_file)
    assert change < 0.05, f"{where} depends on how the starting point was supplied"


def test_a_starting_point_replaces_the_nonbonded_baseline(h2o2_fit):
    """The nonbonded terms are never fit, so a supplied one is used verbatim.

    The `atom` block is what carries the check rather than `lennardjones`:
    every pair of H2O2 is inside the 1-4 exclusions, so its dispersion is
    identically zero and moving it would prove nothing.
    """
    _, params, path = h2o2_fit
    n = len(params.numbers)
    initial = Parameters(numbers=params.numbers, terms={})
    # A blunt shift of the electronegativities: the ACKS2 charges are re-solved
    # from these, so the electrostatic baseline the bonded terms sit on moves.
    mu = np.asarray(params.terms["atom"]["kwargs"]["mu"], dtype=float) + 1.0
    initial.terms["atom"] = {
        "atoms": np.arange(n)[:, None],
        "kwargs": {"mu": mu},
    }
    again = fit_from_file(path, initial=initial)

    assert np.allclose(again.terms["atom"]["kwargs"]["mu"], mu)
    # the parameters it did not mention keep their element-table values
    assert np.allclose(
        again.terms["atom"]["kwargs"]["eta"], params.terms["atom"]["kwargs"]["eta"]
    )
    # and it is a real change: the bonded fit had to absorb a different baseline
    change, _ = _worst_relative_change(params, again)
    assert change > 1e-3


def test_a_starting_point_never_makes_the_fit_worse(h2o2_fit):
    """Refitting from a fitted field descends further; it does not undo anything.

    Both blocks minimize the same residual and neither is seeded outside its own
    bounds, so starting from a converged field can only continue downhill.

    "The same residual" is a *weighted combination* of the energy and force
    blocks, so neither RMSE is separately monotone -- a refit is free to trade a
    little of one for a little of the other, and does: measured, energy
    +2.7e-6 eV against force -1.2e-6 eV/A.  The tolerance is therefore relative
    rather than the absolute 1e-6 it used to be, which was calibrated when the
    energy RMSE was 0.027 eV and is 0.05% of it now that the fit reaches
    0.0022 eV.
    """
    _, params, path = h2o2_fit
    again = fit_from_file(path, initial=params)
    for key in ("force_rmse_eV_A", "energy_rmse_eV"):
        assert again.report[key] <= params.report[key] * 1.01 + 1e-9, (
            f"{key} rose from {params.report[key]:.6g} to {again.report[key]:.6g}"
        )


def test_the_fit_converges_inside_its_default_budget(h2o2_fit):
    """`cycle_tol` must be what ends the fit, not `n_cycles`.

    This is the property that makes everything below meaningful, and it is the
    one that `E0` used to break: while `E0` was fit alongside `D`, the two
    traded along a flat direction for hundreds of cycles and the budget was
    always what stopped the fit.
    """
    _, params, _ = h2o2_fit
    assert params.report["cycles"] < FitConfig.n_cycles


def test_a_converged_fit_is_idempotent(h2o2_fit):
    """Refitting a converged field from its own output returns it.

    "Returns it" means to within `cycle_tol`, which is what the companion test
    below pins down.  The refit also stops immediately -- there is nothing left
    for it to do.
    """
    _, params, path = h2o2_fit
    again = fit_from_file(path, initial=params)

    change, where = _worst_relative_change(params, again)
    assert change < 0.05, f"{where} moved {change:.3g} on a refit"
    # `E0` is derived from the converged mean residual, so it inherits whatever
    # `cycle_tol` left behind and tracks the bond depths directly: a shift of
    # `dD` on each of three bonds moves it by `3 dD`.  At the default
    # `cycle_tol` that is 0.08-0.2 eV on an offset of -235 eV, and it shrinks
    # with the tolerance (0.079 -> 0.017 -> 0.008 eV at 1e-4, 1e-6, 1e-8), which
    # is the fixed-point signature `test_the_refit_drift_is_the_stopping_
    # tolerance` exists to pin.  The bound here is on that residual, not on 0.
    assert again.e0 == pytest.approx(params.e0, abs=0.3)
    assert again.report["cycles"] <= 3, "a converged refit should stop immediately"
    assert again.report["force_rmse_eV_A"] == pytest.approx(
        params.report["force_rmse_eV_A"], rel=1e-4
    )


def test_the_refit_drift_is_the_stopping_tolerance(h2o2_fit):
    """What is left of non-idempotency is `cycle_tol`, and shrinks with it.

    This is the test that distinguishes a fixed point being approached from a
    flat direction being walked along.  Tightening `cycle_tol` by four orders of
    magnitude shrinks the drift by orders of magnitude for a handful of extra
    cycles.  Before `E0` was taken out of the fit the same experiment did the
    opposite: a 64x larger budget moved the parameters *further*, because there
    was no fixed point to approach.
    """
    _, _, path = h2o2_fit
    drifts = {}
    for tol in (1e-4, 1e-8):
        config = FitConfig(cycle_tol=tol, n_cycles=400)
        converged = fit_from_file(path, config=config)
        assert converged.report["cycles"] < config.n_cycles, "did not reach cycle_tol"
        again = fit_from_file(path, config=config, initial=converged)
        drifts[tol] = _worst_relative_change(converged, again)[0]

    assert drifts[1e-8] < drifts[1e-4] / 10, (
        f"tightening cycle_tol barely helped ({drifts}); the fit may be walking "
        "a flat direction again rather than converging"
    )
