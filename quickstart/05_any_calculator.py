"""05 -- Any ASE calculator can be the reference.

`parameterize` only ever asks the calculator for energies and forces, so the
reference method is a free choice: a semiempirical Hamiltonian, a DFT code, or
a machine-learned potential.  Here the same molecule is fit three times against
three different references, and the parameters move the way the underlying
physics does.

`calc_factory` is a callable, not a calculator, because some calculators bind
to a specific `Atoms` at construction and the fit builds many geometries.

    uv run python quickstart/05_any_calculator.py
"""

import numpy as np

import fastforces as ff

from _common import OUTPUT, banner

banner(__doc__)


def tblite(method):
    from tblite.ase import TBLite

    return lambda atoms=None: TBLite(method=method, verbosity=0)


def emt(atoms=None):
    """ASE's built-in effective-medium calculator: no dependencies, poor
    chemistry for molecules -- included only to show nothing here is xTB
    specific."""
    from ase.calculators.emt import EMT

    return EMT()


references = {
    "GFN2-xTB": tblite("GFN2-xTB"),
    "GFN1-xTB": tblite("GFN1-xTB"),
    "EMT": emt,
}

config = ff.FitConfig(n_mode_frames=24, n_conformers=0)
results = {}
for name, factory in references.items():
    params = ff.parameterize(
        ff.build("O"),
        factory,
        config=config,
        training_set=str(OUTPUT / f"water_{name}.xyz"),
    )
    results[name] = params

print(
    f"{'reference':10s} {'O-H r0':>8s} {'O-H k':>10s} {'HOH k':>10s} "
    f"{'E_rmse':>9s} {'F_rmse':>9s}"
)
print(
    f"{'':10s} {'(A)':>8s} {'(eV/A^2)':>10s} {'(eV)':>10s} {'(eV)':>9s} {'(eV/A)':>9s}"
)
for name, params in results.items():
    bond = params.terms["bond"]["kwargs"]
    angle = params.terms["angle"]["kwargs"]
    print(
        f"{name:10s} {bond['r0'][0]:8.4f} {bond['k'][0]:10.1f} {angle['k'][0]:10.2f} "
        f"{params.report['energy_rmse_eV']:9.4f} {params.report['force_rmse_eV_A']:9.4f}"
    )

# The fitted force field is only ever as good as what it was fit to.  Compare
# the three against each other on a stretched geometry.
atoms = ff.io.read_training_set(str(OUTPUT / "water_GFN2-xTB.xyz")).equilibrium.copy()
atoms.positions[1] += np.array([0.15, 0.0, 0.0])
energies = {n: ff.evaluate(atoms, p)[0] for n, p in results.items()}
base = {
    n: ff.evaluate(
        ff.io.read_training_set(str(OUTPUT / f"water_{n}.xyz")).equilibrium, p
    )[0]
    for n, p in results.items()
}
print("\nstretch one O-H by 0.15 A, relative to each field's own minimum:")
for name in results:
    print(f"  {name:10s} {energies[name] - base[name]:+7.3f} eV")

print("""
EMT has no business describing a water molecule, and the energy residual says
so -- over ten times the xTB fits, and a stretch penalty three times too
steep.  Its
force residual does not, because EMT's forces are simply small; `fit_report()`
gives both blocks for exactly this reason.  The two xTB fits, meanwhile, agree
with each other to well within their own residuals, which is what a working
parameterization of two similar Hamiltonians should look like.""")
