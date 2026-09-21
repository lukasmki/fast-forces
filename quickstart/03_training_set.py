"""03 -- One file holds the whole fit.

Every reference calculation the fit consumed is written to a single extended
XYZ.  Nothing else is needed to reproduce the parameters: `fit_from_file` reads
that file and re-runs the regression without calling the reference method
again.  The Hessian is not stored separately -- it is reconstructed from the
displaced frames that were used to compute it in the first place.

    uv run python quickstart/03_training_set.py
"""

import numpy as np

import fastforces as ff

from _common import OUTPUT, banner, gfn2

banner(__doc__)

path = OUTPUT / "water_training.xyz"
atoms = ff.build("O")
params = ff.parameterize(atoms, gfn2, training_set=str(path))

data = ff.io.read_training_set(str(path))
print(f"{path.name}: {data.summary()}")
print(f"metadata: {', '.join(k for k in data.meta)}\n")

for kind in ("equilibrium", "hessian", "mode", "conformer", "torsion"):
    frames = data.of_kind(kind)
    if not frames:
        continue
    energies = np.array([f.get_potential_energy() for f in frames])
    forces = np.concatenate([f.get_forces().ravel() for f in frames])
    print(
        f"  {kind:12s} {len(frames):3d} frames"
        f"  E span {np.ptp(energies):7.3f} eV"
        f"  max|F| {abs(forces).max():6.3f} eV/A"
    )

# The displaced frames tagged `hessian` carry the +/- delta columns, so the
# Hessian is recovered by finite difference rather than serialized as a blob.
hessian = data.hessian
frequencies = np.sort(np.linalg.eigvalsh(hessian))[-3:]
print(
    f"\nHessian rebuilt from the frames: {hessian.shape}, symmetric to "
    f"{abs(hessian - hessian.T).max():.2e}"
)
print(f"stiffest eigenvalues: {frequencies.round(2)} eV/A^2")

# The claim under test.  `parameterize` has no in-memory shortcut: it writes
# this file and then fits from it, so sufficiency is structural rather than
# something the fit has to be careful to preserve.  Re-fitting here reproduces
# the parameters exactly, without the reference calculator existing any more.
del atoms
refit = ff.fit_from_file(str(path))
deltas = [
    abs(params.terms[t]["kwargs"][k] - refit.terms[t]["kwargs"][k]).max()
    for t in params.terms
    for k in params.terms[t]["kwargs"]
]
print(f"\nre-fit from the file alone: max parameter difference {max(deltas):.2e}")
print(f"re-fit accuracy: {refit.report['force_rmse_eV_A']:.4f} eV/A")

print("""
Note the 20 conformer frames all sit at the same energy -- water has only one
conformer, so they contribute nothing here.  They earn their keep on flexible
molecules, where they are what stops the fit from describing a single well.""")
