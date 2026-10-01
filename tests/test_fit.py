"""End-to-end fitting against a real reference calculator."""

import copy

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
from fastforces.fit import NOT_FITTED
from fastforces.topology import enumerate_terms
from DynamicTopology.forcefield.params import use

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


def test_bond_r0_is_the_measured_length(h2o2_fit):
    """`r0` is read off the equilibrium frame, averaged within its class.

    It is not fitted: the bond is a harmonic column at that `r0` in the linear
    solve, so whatever nonbonded baseline survives on a bonded pair is left to
    the other terms rather than balanced by moving `r0`.
    """
    _, params, path = h2o2_fit
    positions = io.read_training_set(path).equilibrium.get_positions()
    atoms = params.terms["bond"]["atoms"]
    lengths = np.linalg.norm(positions[atoms[:, 0]] - positions[atoms[:, 1]], axis=1)
    classes = enumerate_terms(io.read_training_set(path).equilibrium).classes["bond"]
    r0 = params.terms["bond"]["kwargs"]["r0"]
    for c in np.unique(classes):
        assert r0[classes == c] == pytest.approx(lengths[classes == c].mean(), abs=1e-12)


def test_symmetry_equivalent_bonds_share_parameters(h2o2_fit):
    _, params, _ = h2o2_fit
    kwargs = params.terms["bond"]["kwargs"]
    rows = [tuple(sorted(r)) for r in params.terms["bond"]["atoms"]]
    first, second = rows.index((0, 2)), rows.index((1, 3))
    for name in ("r0", "k", "D"):
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

    The jsonl round trip perturbs the starting point by ~1e-16 -- the round-off
    of multiplying by a unit factor and dividing it out again -- and the fit is
    one linear solve, so that is all that separates the two results.
    """
    _, params, path = h2o2_fit
    jsonl = tmp_path / "h2o2.jsonl"
    params.to_jsonl(str(jsonl))
    change, _ = _worst_relative_change(params, as_parameters(str(jsonl)))
    assert change < 1e-12, "the jsonl round trip itself should be near-exact"

    from_object = fit_from_file(path, initial=params)
    from_file = fit_from_file(path, initial=str(jsonl))
    change, where = _worst_relative_change(from_object, from_file)
    assert change < 1e-9, f"{where} depends on how the starting point was supplied"


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


def test_a_fit_is_idempotent(h2o2_fit):
    """Refitting from its own output returns the same field.

    What a starting point supplies -- the equilibrium values, the bond depths
    and asymptotes, the nonbonded baseline -- is exactly what the fit held
    fixed the first time, and the force constants come out of the same linear
    solve.
    """
    _, params, path = h2o2_fit
    again = fit_from_file(path, initial=params)
    change, where = _worst_relative_change(params, again)
    assert change < 1e-9, f"{where} moved {change:.3g} on a refit"
    assert again.e0 == pytest.approx(params.e0, rel=1e-12)


def test_a_supplied_force_constant_does_not_reach_the_result(h2o2_fit):
    """The linear solve has no starting point, so any `k` in `initial` is inert."""
    _, params, path = h2o2_fit
    initial = as_parameters(copy.deepcopy(params.terms))
    for block in initial.terms.values():
        if "k" in block["kwargs"]:
            block["kwargs"] = {**block["kwargs"], "k": 3.0 * block["kwargs"]["k"] + 1.0}
    again = fit_from_file(path, initial=initial)
    change, where = _worst_relative_change(params, again)
    assert change < 1e-9, f"{where} followed the supplied force constants"


# ---------------------------------------------------------------------------
# fixed point charges
# ---------------------------------------------------------------------------


def _fixed(path, source="mulliken"):
    """`fit_from_file` under `pointcharge`, with the charges read from `source`."""
    with use(electrostatics="pointcharge"):
        return fit_from_file(
            path,
            config=FitConfig(n_mode_frames=30, n_conformers=0, electrostatics=source),
        )


def _with_mulliken(path, charges, out):
    """Copy a training set, writing `charges` onto every frame as `mulliken`.

    The fixture's reference calculator is plain `tblite.ase.TBLite`, which
    writes no `mulliken` array -- `calculators.tblite.TBLiteCalculator` and
    `calculators.pyscf.PySCFCalculator` do -- so the charges are injected here.  Their values do not matter to what these tests assert; that
    they are carried from the file into the fitted field does.
    """
    data = io.read_training_set(path)
    for frame in data.frames:
        if frame.info.get("frame_kind") in ("fragment", "atom"):
            continue  # another molecule; `fit` never reads its charges
        frame.set_array("mulliken", np.asarray(charges, dtype=float), float)
    io.write_training_set(str(out), data.frames, meta=data.meta)
    return str(out)


def test_fixed_charges_come_off_the_training_file(tmp_path, h2o2_fit):
    """Under `pointcharge` the charges are a `charge` block, and `atom` is gone."""
    _, params, path = h2o2_fit
    # H2O2 as O, O, H, H: the two oxygens are one equivalence class and the two
    # hydrogens another, and these are already symmetric so the class-wise mean
    # leaves them alone.
    charges = np.array([-0.4, -0.4, 0.4, 0.4])
    fixed = _fixed(_with_mulliken(path, charges, tmp_path / "charged.xyz"))

    assert fixed.electrostatics() == "charge"
    assert "atom" not in fixed.terms
    assert np.allclose(fixed.terms["charge"]["kwargs"]["q"], charges)
    # the ACKS2 fit of the same molecule is the other way round
    assert params.electrostatics() == "atom"


def test_fixed_charges_are_averaged_within_an_equivalence_class(tmp_path, h2o2_fit):
    """Asymmetric input charges come out symmetric, with the total preserved.

    A single geometry's Mulliken charges put slightly different values on
    symmetry-equivalent atoms, and freezing that in would give a rotor a
    spurious electrostatic torsion.
    """
    _, _, path = h2o2_fit
    lopsided = np.array([-0.5, -0.3, 0.45, 0.35])
    fixed = _fixed(_with_mulliken(path, lopsided, tmp_path / "lopsided.xyz"))

    q = np.asarray(fixed.terms["charge"]["kwargs"]["q"])
    assert q[0] == pytest.approx(q[1])
    assert q[2] == pytest.approx(q[3])
    # a class-wise mean moves no charge between classes
    assert q.sum() == pytest.approx(lopsided.sum())


def test_fixed_charges_need_charges_in_the_file(h2o2_fit):
    """A training set without them fails loudly rather than silently unscreened."""
    _, _, path = h2o2_fit
    with pytest.raises(ValueError, match="mulliken"):
        _fixed(path)


def test_tblite_writes_its_charges_onto_the_frame():
    """A manifest's `tblite` leaves xTB's charges where the fit reads them.

    Hydroxide, so the check that they sum to the formal charge is not the
    trivial one a neutral molecule would pass by symmetry alone.
    """
    import fastforces as ff
    from fastforces import sampling
    from fastforces.manifest import calculator_factory

    factory, _ = calculator_factory({"name": "tblite"})
    atoms = ff.build("[OH-]")
    frame = sampling.label(atoms, factory, "equilibrium")

    q = frame.get_array("mulliken")
    assert q.shape == (len(atoms),)
    assert q.sum() == pytest.approx(-1.0, abs=1e-6)
    assert q[0] < -0.5  # the oxygen carries it


def test_label_replaces_a_stale_array():
    """A frame copied off a labelled one gets its own charges, not the parent's.

    A Hessian frame is a copy of the labelled equilibrium, so it arrives already
    carrying a `mulliken` array from another geometry.
    """
    import fastforces as ff
    from fastforces import sampling
    from fastforces.manifest import calculator_factory

    factory, _ = calculator_factory({"name": "tblite"})
    atoms = ff.build("O")
    atoms.set_array("mulliken", np.full(len(atoms), 9.0), float)
    frame = sampling.label(atoms, factory, "hessian")

    q = frame.get_array("mulliken")
    assert not np.allclose(q, 9.0)
    assert q.sum() == pytest.approx(0.0, abs=1e-6)


def test_a_tblite_fixed_charge_fit_freezes_the_xtb_charges(tmp_path):
    """tblite alone is enough for Mulliken point charges: no injected array.

    The template's charges are the equilibrium frame's xTB charges averaged
    within each equivalence class, and carry the formal charge.
    """
    import fastforces as ff
    from fastforces.charges import class_average
    from fastforces.manifest import calculator_factory

    factory, _ = calculator_factory({"name": "tblite"})
    path = tmp_path / "water.xyz"
    with use(electrostatics="pointcharge"):
        params = ff.parameterize(
            ff.build("O"),
            factory,
            config=FitConfig(
                n_mode_frames=10, n_conformers=0, electrostatics="mulliken"
            ),
            training_set=str(path),
        )

    equilibrium = io.read_training_set(str(path)).equilibrium
    raw = equilibrium.get_array("mulliken")
    classes = enumerate_terms(equilibrium).atom_classes
    q = np.asarray(params.terms["charge"]["kwargs"]["q"])
    assert np.allclose(q, class_average(raw, classes))
    assert q.sum() == pytest.approx(0.0, abs=1e-6)
    assert q[0] < 0 < q[1]


def test_the_fixed_charge_fit_is_as_accurate(tmp_path, h2o2_fit):
    """Swapping the electrostatic baseline does not cost accuracy.

    Neither term is fitted -- both are the baseline the bonded terms are fit
    against -- so what this really says is that the bonded fit absorbs the
    different baseline, which is the claim `_nonbonded` rests on.
    """
    _, acks2_params, path = h2o2_fit
    charges = np.array([-0.4, -0.4, 0.4, 0.4])
    fixed = _fixed(_with_mulliken(path, charges, tmp_path / "accuracy.xyz"))
    data = io.read_training_set(path)
    frames = [f for f in data.frames if f.info.get("frame_kind") not in NOT_FITTED]

    fixed_energy, fixed_force = _score(fixed, frames)
    acks2_energy, acks2_force = _score(acks2_params, frames)
    assert fixed_energy < max(2.0 * acks2_energy, 0.05)
    assert fixed_force < max(2.0 * acks2_force, 0.5)


def test_a_fixed_charge_starting_point_switches_the_fit_over(tmp_path, h2o2_fit):
    """An `initial` field's electrostatics replaces the configured one.

    The two terms are alternatives, so "start from this field" has to mean its
    electrostatics too -- the alternative is a field carrying both, which
    `Parameters.electrostatics` rejects.
    """
    _, params, path = h2o2_fit
    n = len(params.numbers)
    charges = np.array([-0.45, -0.45, 0.45, 0.45])
    initial = Parameters(
        numbers=params.numbers,
        terms={
            "charge": {"atoms": np.arange(n)[:, None], "kwargs": {"q": charges}},
        },
    )
    # Under ACKS2, the default: the starting point is what moves it over.
    again = fit_from_file(path, initial=initial)

    assert again.electrostatics() == "charge"
    assert np.allclose(again.terms["charge"]["kwargs"]["q"], charges)


def test_a_diatomic_fits(tmp_path, tblite_factory):
    """A bond and nothing else: the linear block is the one harmonic column.

    The force bar is the harmonic bond's, which cannot follow the anharmonicity
    of the stretched mode frames: measured 0.0156 eV/A, where the nonlinear
    Morse fit it replaced passed a bar of 0.01.

    Hydroxide, so it also runs the whole reference-charge path from plain
    `TBLite` -- which writes no `mulliken` array -- to a `q0` carrying the -1.
    """
    import fastforces as ff

    params = ff.parameterize(
        ff.build("[OH-]"),
        tblite_factory,
        config=FitConfig(n_mode_frames=10, n_conformers=0, electrostatics="mulliken"),
        training_set=str(tmp_path / "hydroxide.xyz"),
    )
    assert set(params.terms) - {"atom", "lennardjones"} == {"bond", "reference"}
    assert params.report["force_rmse_eV_A"] < 0.03
    q0 = params.terms["atom"]["kwargs"]["q0"]
    assert q0.sum() == pytest.approx(-1.0, abs=1e-6)
    assert q0[0] < -0.5


# ---------------------------------------------------------------------------
# reference charges under ACKS2
# ---------------------------------------------------------------------------


def test_acks2_carries_the_reference_charges_as_q0(tmp_path, h2o2_fit):
    """Under ACKS2 the charges are the `atom` block's `q0`, class-averaged,
    next to the element defaults -- the same numbers `pointcharge` freezes."""
    _, _, path = h2o2_fit
    lopsided = np.array([-0.5, -0.3, 0.45, 0.35])
    params = fit_from_file(
        _with_mulliken(path, lopsided, tmp_path / "q0.xyz"),
        config=FitConfig(n_mode_frames=30, n_conformers=0, electrostatics="mulliken"),
    )
    assert params.electrostatics() == "atom"
    kwargs = params.terms["atom"]["kwargs"]
    assert np.allclose(kwargs["q0"], [-0.4, -0.4, 0.4, 0.4])
    assert {"mu", "eta", "soft_amp", "soft_decay"} <= set(kwargs)


def test_neutral_reference_charges_are_zero(h2o2_fit):
    _, params, _ = h2o2_fit
    assert np.array_equal(params.terms["atom"]["kwargs"]["q0"], np.zeros(4))


def test_a_charged_template_cannot_be_neutral(tmp_path, tblite_factory):
    """The case the reference charges exist for: an ion fitted `neutral` would
    be held at zero charge by fragment ACKS2, so it is refused."""
    import fastforces as ff

    with pytest.raises(ValueError, match="formal charge of -1"):
        ff.parameterize(
            ff.build("[OH-]"),
            tblite_factory,
            config=FitConfig(n_mode_frames=4, n_conformers=0),
            training_set=str(tmp_path / "hydroxide.xyz"),
        )


def test_neutral_point_charges_are_refused(h2o2_fit):
    _, _, path = h2o2_fit
    with pytest.raises(ValueError, match="puts no charge on any atom"):
        _fixed(path, source="neutral")


def test_charges_from_another_charge_state_are_refused(tmp_path, h2o2_fit):
    _, _, path = h2o2_fit
    with pytest.raises(ValueError, match="another\s+charge state"):
        _fixed(_with_mulliken(path, np.full(4, 0.25), tmp_path / "cation.xyz"))


@pytest.mark.parametrize(
    "stored, expected", [("acks2", "neutral"), ("fixed", "mulliken"), ("esp", "esp")]
)
def test_a_stored_config_translates_the_old_names(stored, expected):
    """A training set written before the split records the old names."""
    assert FitConfig.from_stored({"electrostatics": stored}).electrostatics == expected


def test_a_stored_config_drops_the_nonlinear_bond_fields():
    """A training set written before the bond fit went linear still refits."""
    stored = {"bond_form": "morse", "n_cycles": 200, "cycle_tol": 1e-4, "seed": 3}
    assert FitConfig.from_stored(stored) == FitConfig(seed=3)


def test_a_new_config_refuses_the_old_names():
    with pytest.raises(ValueError, match="old name for the inference method"):
        FitConfig(electrostatics="fixed")
