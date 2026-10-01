"""The bond depths carry the atomization energy.

A single-molecule fit scales every Morse depth by one factor so that `E0` is the
molecule's free atoms and nothing else: the bonds, with the rest of the field at
equilibrium, carry `E(molecule) - sum_i E(atom_i)`.  The free atoms are single
points in the training set (`sampling.atom_frames`).  Which atoms, and the
closed-form scale, are tested without a calculator; the fit against GFN2-xTB.
"""

import shutil

import numpy as np
import pytest
from ase import Atoms
from DynamicTopology.forcefield.evaluate import evaluate as dt_evaluate

import fastforces as ff
from fastforces import elements, io, sampling
from fastforces.fit import _depth_scale


def _flat(atoms=None):
    from ase.calculators.lj import LennardJones

    return LennardJones()


# ---------------------------------------------------------------------------
# which atoms
# ---------------------------------------------------------------------------


def test_one_neutral_ground_state_atom_per_element():
    """Neutral whatever the molecule's charge, as `label` references a dataset."""
    frames = sampling.atom_frames(ff.build("[OH3+]"), _flat)
    assert [f.get_chemical_symbols() for f in frames] == [["H"], ["O"]]
    for frame in frames:
        assert frame.info["frame_kind"] == "atom"
        assert frame.get_initial_charges().sum() == 0
        assert frame.info["spin"] == elements.ATOM_SPIN[frame.get_chemical_symbols()[0]]


def test_a_lone_atom_has_no_atom_frames():
    assert sampling.atom_frames(Atoms("Ar"), _flat) == []


def test_atom_frames_are_all_or_nothing():
    """The sum needs every element, so one failure drops them all."""

    from ase.calculators.lj import LennardJones

    class Unconverged(LennardJones):
        def calculate(self, *args, **kwargs):
            raise RuntimeError("SCF did not converge")

    def no_oxygen(atoms=None):
        if atoms is not None and "O" in atoms.get_chemical_symbols():
            return Unconverged()
        return _flat()

    assert sampling.atom_frames(ff.build("O"), no_oxygen) == []


# ---------------------------------------------------------------------------
# the scale
# ---------------------------------------------------------------------------


def test_the_scale_puts_e0_on_the_free_atoms():
    # bound: the fit's energy zero sits below the free atoms
    offset, depth, free_atoms = -238.0, 8.0, -226.0
    scale = _depth_scale(offset, depth, free_atoms)
    assert offset + scale * depth == pytest.approx(free_atoms)


def test_a_scale_outside_the_bracket_is_refused():
    """A molecule its fit puts above its free atoms needs a negative depth."""
    with pytest.raises(ValueError, match="scaling by"):
        _depth_scale(offset=-218.0, depth=8.0, free_atoms=-219.0)


# ---------------------------------------------------------------------------
# the fit, against GFN2-xTB
# ---------------------------------------------------------------------------


def _free_atoms(path):
    data = io.read_training_set(path)
    energies = {
        f.get_chemical_symbols()[0]: f.get_potential_energy()
        for f in data.of_kind("atom")
    }
    return data, sum(energies[s] for s in data.equilibrium.get_chemical_symbols())


@pytest.mark.slow
def test_e0_is_the_free_atoms(h2o2_fit):
    _, params, path = h2o2_fit
    data, free_atoms = _free_atoms(path)
    assert len(data.of_kind("atom")) == 2  # H, O
    assert params.e0 == pytest.approx(free_atoms, abs=1e-9)
    assert "depth_scale" in params.report


@pytest.mark.slow
def test_the_field_reproduces_the_atomization_energy(h2o2_fit):
    """To within the fit's own residual at equilibrium, which is what is left."""
    _, params, path = h2o2_fit
    data, free_atoms = _free_atoms(path)
    equilibrium = data.equilibrium
    reference = equilibrium.get_potential_energy() - free_atoms
    field = dt_evaluate(equilibrium, params.to_terms()).energy - free_atoms
    residual = params.report["equilibrium"]["E_rmse"]
    assert field == pytest.approx(reference, abs=residual + 1e-6)


@pytest.mark.slow
def test_the_atoms_are_not_fitted_to(h2o2_fit):
    _, params, path = h2o2_fit
    data = io.read_training_set(path)
    assert params.report["frames"] == len(data.frames) - len(
        data.of_kind("fragment", "atom")
    )
    assert "atom" not in params.report


@pytest.mark.slow
def test_a_set_without_atoms_keeps_the_table_depths(h2o2_fit, tmp_path, tblite_factory):
    """And is brought up to date once, the way a manifest rerun does it."""
    _, params, path = h2o2_fit
    old = tmp_path / "old.xyz"
    shutil.copy(path, old)
    data = io.read_training_set(str(old))
    kept = [f for f in data.frames if f.info.get("frame_kind") != "atom"]
    io.write_training_set(str(old), kept, meta=data.meta)

    unscaled = ff.fit_from_file(str(old))
    assert "depth_scale" not in unscaled.report
    bonds = unscaled.terms["bond"]["atoms"]
    numbers = data.equilibrium.get_atomic_numbers()
    table = elements.morse_well_depth(numbers[bonds[:, 0]], numbers[bonds[:, 1]])
    assert np.allclose(unscaled.terms["bond"]["kwargs"]["D"], table)

    assert ff.add_atom_frames(str(old), tblite_factory) == 2
    assert ff.add_atom_frames(str(old), tblite_factory) == 0
    rescaled = ff.fit_from_file(str(old))
    assert np.allclose(
        rescaled.terms["bond"]["kwargs"]["D"], params.terms["bond"]["kwargs"]["D"]
    )
