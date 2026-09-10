"""01 -- Quickstart: a force field from a SMILES string.

The whole package in six lines.  `build` turns a SMILES string into an `Atoms`
with perceived connectivity, `parameterize` generates its own reference data
and fits every term against it, and `FastForces` evaluates the result as an
ordinary ASE calculator.

    uv run python examples/01_quickstart.py
"""

import fastforces as ff
from tblite.ase import TBLite

from _common import OUTPUT, banner

banner(__doc__)

# Any ASE calculator works here; `calc_factory` is called with the atoms so
# calculators that need the system at construction time are supported too.
atoms = ff.build("CC#N")
params = ff.parameterize(
    atoms,
    calc_factory=lambda atm: TBLite(atm, method="GFN2-xTB", verbosity=0),
    training_set=str(OUTPUT / "acetonitrile.xyz"),
)

atoms.calc = ff.FastForces(atoms, params)

print(f"{atoms.get_chemical_formula()}: {params!r}\n")
print(params.fit_report())
print()
print(f"energy       {atoms.get_potential_energy():.6f} eV")
print(f"max |force|  {abs(atoms.get_forces()).max():.6f} eV/A")
print(f"charges      {atoms.get_charges().round(3)}")

params.to_jsonl(str(OUTPUT / "acetonitrile.jsonl"))
print(f"\nwrote {OUTPUT / 'acetonitrile.jsonl'} and {OUTPUT / 'acetonitrile.xyz'}")
