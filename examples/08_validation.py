"""08 -- Checking the fit against the reference on held-out geometries.

A fit report tells you how well the field describes its own training data.
This scores it on geometries it never saw.  The training set used relaxed
torsion scans at 30 degree steps; the validation scan is relaxed at 10 degree
steps, and every angle that coincides with a training angle is dropped, so what
is left is genuinely held out.

    uv run python examples/08_validation.py
"""

import numpy as np

import fastforces as ff

from _common import banner, fitted, gfn2

banner(__doc__)

config = ff.FitConfig(torsion_step_deg=30.0)
atoms, params, _ = fitted("OO", config=config, name="peroxide")
dihedral = ff.sampling.rotatable_dihedrals(ff.build("OO"))[0]
print(
    f"scanning dihedral {dihedral} of {atoms.get_chemical_formula()} "
    f"(trained on {int(360 / config.torsion_step_deg)} relaxed points)\n"
)

frames = ff.sampling.torsion_frames(atoms, gfn2, [dihedral], step_deg=10.0)
held_out = [
    frame
    for frame in frames
    if round(frame.get_dihedral(*dihedral)) % config.torsion_step_deg
]

angles = np.array([frame.get_dihedral(*dihedral) for frame in held_out])
order = np.argsort(angles)
angles = angles[order]
held_out = [held_out[i] for i in order]

reference = np.array([frame.get_potential_energy() for frame in held_out])
predicted = np.array([ff.evaluate(frame, params)[0] for frame in held_out])
force_rmse = np.array(
    [
        np.sqrt(np.mean((ff.evaluate(frame, params)[1] - frame.get_forces()) ** 2))
        for frame in held_out
    ]
)

reference -= reference.min()
predicted -= predicted.min()
errors = predicted - reference

print(f"{len(held_out)} held-out geometries")
print(f"  energy RMSE   {np.sqrt(np.mean(errors**2)):.4f} eV")
print(f"  max error     {abs(errors).max():.4f} eV")
print(f"  force RMSE    {force_rmse.mean():.4f} eV/A")
print(
    f"  training      {params.report['energy_rmse_eV']:.4f} eV, "
    f"{params.report['force_rmse_eV_A']:.4f} eV/A"
)
print(
    f"  cis barrier   reference {reference.max():.3f} eV, "
    f"fitted {predicted.max():.3f} eV\n"
)

# A plain terminal plot: the profile shape matters more than the numbers.
height, top = 16, max(reference.max(), predicted.max())
grid = [[" "] * len(angles) for _ in range(height)]
for column, (r, p) in enumerate(zip(reference, predicted)):
    for value, mark in ((r, "o"), (p, "x")):
        row = height - 1 - min(height - 1, int(value / top * (height - 1)))
        grid[row][column] = "+" if grid[row][column] not in (" ", mark) else mark

print("relative energy     o reference   x fitted   + both")
for index, row in enumerate(grid):
    print(f"  {top * (height - 1 - index) / (height - 1):5.2f} eV |{''.join(row)}")
print("           +" + "-" * len(angles))
print(
    f"            {angles[0]:.0f}"
    + " " * (len(angles) - 8)
    + f"{angles[-1]:.0f} degrees"
)

print("""
The torsion profile is the part of the fit the Hessian cannot see -- a Hessian
is quadratic about one minimum -- so it comes entirely from the relaxed scans
in the training set.  Remove them and this plot degrades badly while
`fit_report()` still looks healthy, which is why the scans are not optional.

The held-out energy RMSE lands on top of the training RMSE, which is what says
the fit generalized rather than memorized.  The forces are somewhat worse out
of sample, and the cis barrier -- the hardest point in the profile, where the
two hydrogens are eclipsed -- is overshot by around 20%.  Those are the honest
limits of this fit, and they are visible only because these geometries were
held out rather than fit.""")
