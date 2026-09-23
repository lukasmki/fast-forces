"""The PySCF calculator, with and without an implicit solvent.

The solvent enters the two evaluation modes by different routes -- PySCF's
own gradient `kernel` on the Born-Oppenheimer path, hand-added terms on the
Car-Parrinello one -- so the tests that matter are the ones that make the two
agree, and that hold either against a finite difference.  PBE in 6-31G keeps
each SCF well under a second; nothing here depends on the method.
"""

import numpy as np
import pytest

import fastforces as ff

pytest.importorskip("pyscf")
from fastforces.calculators.pyscf import PySCFCalculator  # noqa: E402

pytestmark = pytest.mark.slow

CHEAP = dict(xc="PBE", basis="6-31g")


@pytest.fixture(scope="module")
def hydroxide():
    atoms = ff.build("[OH-]")
    # off equilibrium, so the forces are large enough to compare
    atoms.positions[1] += [0.05, 0.02, 0.0]
    return atoms


def evaluate(atoms, **kwargs):
    work = atoms.copy()
    work.calc = PySCFCalculator(charge=-1, spin=0, **CHEAP, **kwargs)
    return work.get_potential_energy(), work.get_forces()


def test_the_continuum_stabilizes_an_ion(hydroxide):
    """Several eV for a bare anion -- the whole reason to use it."""
    gas, _ = evaluate(hydroxide)
    solvated, _ = evaluate(hydroxide, pcm="IEF-PCM")
    assert solvated - gas < -2.0


def test_a_weaker_dielectric_screens_less(hydroxide):
    gas, _ = evaluate(hydroxide)
    water, _ = evaluate(hydroxide, pcm="IEF-PCM")
    thf, _ = evaluate(hydroxide, pcm="IEF-PCM", pcm_eps=7.4)
    assert water < thf < gas


@pytest.mark.parametrize("pcm", [None, "IEF-PCM", "C-PCM"])
def test_forces_are_the_gradient_of_the_energy(hydroxide, pcm):
    _, forces = evaluate(hydroxide, pcm=pcm)
    delta = 1e-3
    energies = []
    for sign in (+1, -1):
        displaced = hydroxide.copy()
        displaced.positions[1, 0] += sign * delta
        energies.append(evaluate(displaced, pcm=pcm)[0])
    numeric = -(energies[0] - energies[1]) / (2 * delta)
    assert forces[1, 0] == pytest.approx(numeric, abs=1e-3)


@pytest.mark.parametrize("pcm", [None, "IEF-PCM"])
def test_car_parrinello_at_the_converged_density_is_born_oppenheimer(hydroxide, pcm):
    """At a converged `D` the CP energy and forces are the BO ones.  Without the
    solvent gradient on the CP path the forces here would be off by ~0.8 eV/A,
    and without the solvent potential its Fock matrix would be the gas-phase
    one."""
    energy, forces = evaluate(hydroxide, pcm=pcm)

    cp = PySCFCalculator(charge=-1, spin=0, cp=True, pcm=pcm, **CHEAP)
    dm = cp.scf_dm(hydroxide)
    cp.set_dm(dm)
    work = hydroxide.copy()
    work.calc = cp
    assert work.get_potential_energy() == pytest.approx(energy, abs=1e-6)
    np.testing.assert_allclose(work.get_forces(), forces, atol=1e-3)

    reference = np.asarray(cp.mf.get_fock(dm=dm))
    for spin in (0, 1):
        np.testing.assert_allclose(cp.fock[spin], reference[spin], atol=1e-10)


def test_the_manifest_labels_a_solvated_method():
    from fastforces import manifest

    _, label = manifest.calculator_factory(
        {"name": "pyscf", "xc": "PBE", "basis": "6-31g", "pcm": "IEF-PCM"}
    )
    assert label == "PBE/6-31g/IEF-PCM(eps=78.3553)"


def test_perceived_bonds_are_the_bonded_pairs_with_integer_orders():
    """Every pair has a positive Mayer bond order; a connectivity is the graph."""
    from fastforces.calculators.pyscf import perceive_bonds

    # HO2 at PBE/6-31G: O=O-ish, O-H, and a 0.02 O...H across the angle
    bond_order = np.array(
        [[0.0, 1.267, 0.022], [1.267, 0.0, 0.769], [0.022, 0.769, 0.0]]
    )
    assert perceive_bonds(bond_order) == [(0, 1, 1.0), (1, 2, 1.0)]


def test_a_stated_connectivity_is_kept(hydroxide):
    """The frame's topology is an input; the bond orders go to their own array."""
    work = hydroxide.copy()
    stated = [[0, 1, 1.0]]
    work.info["connectivity"] = stated
    work.calc = PySCFCalculator(charge=-1, spin=0, **CHEAP)
    work.get_potential_energy()
    assert work.info["connectivity"] is stated
    assert work.arrays["bond-order"].shape == (2, 2)


def test_a_perceived_connectivity_follows_the_geometry():
    """Without one, it is perceived -- and re-perceived on the next call, so a
    relaxation or an MD run sees a bond break.

    The neutral radical, because it dissociates: closed-shell hydroxide keeps a
    Mayer bond order near 0.9 out to 3 A, and at 4 A its SCF does not converge.
    """
    work = ff.build("[OH]")
    work.info.pop("connectivity", None)
    work.calc = PySCFCalculator(charge=0, spin=1, **CHEAP)
    work.get_potential_energy()
    assert [tuple(b[:2]) for b in work.info["connectivity"]] == [(0, 1)]
    work.positions[1] += [4.0, 0.0, 0.0]
    work.get_potential_energy()
    assert work.info["connectivity"] == []


def test_an_unconverged_scf_raises(hydroxide):
    """PySCF hands back whatever the last cycle had; it must not reach a
    training set.  Closed-shell hydroxide pulled 4 A apart does not converge."""
    from ase.calculators.calculator import CalculationFailed

    work = hydroxide.copy()
    work.positions[1] += [4.0, 0.0, 0.0]
    work.calc = PySCFCalculator(charge=-1, spin=0, **CHEAP)
    with pytest.raises(CalculationFailed, match="not converged"):
        work.get_potential_energy()
