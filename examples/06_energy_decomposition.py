"""06 -- What the four evaluators each contribute, and why r0 looks wrong.

`FastForces` is a sum of four pieces: the bonded terms, ACKS2 charge-
equilibration electrostatics, Lennard-Jones outside the bonded exclusions, and
ZBL screened nuclear repulsion.  Only the bonded block is fit; the other three
come from element tables and are evaluated and subtracted before the fit, so
the bonded parameters absorb only what is left over.

That subtraction has a consequence worth understanding, which is the second
half of this example.

    uv run python examples/06_energy_decomposition.py
"""

import numpy as np

import fastforces as ff
from fastforces.forcefield.acks2 import ACKS2
from fastforces.forcefield.lj import LennardJones
from fastforces.forcefield.qforce import QForce
from fastforces.forcefield.zbl import ZBL

from _common import banner, fitted

banner(__doc__)

atoms, params, training = fitted("CC#N", name="acetonitrile")
pos, cell, pbc = atoms.get_positions(), np.array(atoms.cell), atoms.pbc

pieces = {
    "bonded (QForce)": QForce(bond_form="morse")(pos, pbc, cell, params.bonded_terms()),
    "ACKS2": ACKS2()(pos, pbc, cell, params.terms),
    "Lennard-Jones": LennardJones(params.exclusions)(pos, pbc, cell, params.terms),
    "ZBL": ZBL()(pos, atoms.get_atomic_numbers(), pbc, cell),
}

print(f"{'term':18s} {'energy (eV)':>13s} {'max |force|':>13s}")
for name, (energy, forces) in pieces.items():
    print(f"{name:18s} {energy:13.4f} {abs(forces).max():13.4f}")
print(f"{'E0 (reference)':18s} {params.e0:13.4f} {0.0:13.4f}")

total = sum(e for e, _ in pieces.values()) + params.e0
print(f"{'total':18s} {total:13.4f}")
print(f"{'FastForces':18s} {ff.evaluate(atoms, params)[0]:13.4f}  (agrees)\n")

print("Lennard-Jones is exactly zero here: in a six-atom molecule every pair is")
print("within the 1-2/1-3/1-4 exclusions, so nothing is left for it to act on.\n")

# ACKS2 solves its charges at every geometry rather than carrying fixed ones.
atoms.calc = ff.FastForces(atoms, params)
print("ACKS2 charges at equilibrium:")
print(
    "  "
    + "  ".join(
        f"{a.symbol}{a.index}={q:+.3f}" for a, q in zip(atoms, atoms.get_charges())
    )
)

frames = ff.io.read_training_set(str(training)).of_kind("mode", "conformer")
spread = np.array([ff.FastForces(f, params).get_property("charges", f) for f in frames])
print(f"range over the {len(frames)} distorted training geometries:")
print(
    "  "
    + "  ".join(
        f"{a.symbol}{a.index}={spread[:, a.index].min():+.3f}..{spread[:, a.index].max():+.3f}"
        for a in atoms
    )
)
print("""  The swing is only a few thousandths of an electron: these are modest
  distortions and the ACKS2 parameters are element defaults, not fit.  What
  matters is that it is nonzero and geometry-driven at all -- the charges are
  re-solved at every step, and the response contributes real forces.
""")

# ---------------------------------------------------------------------
# Why the fitted r0 values are shorter than the bond lengths
# ---------------------------------------------------------------------
print("-" * 68)
bonds = params.terms["bond"]
lengths = np.linalg.norm(pos[bonds["atoms"][:, 0]] - pos[bonds["atoms"][:, 1]], axis=1)
print(f"{'bond':10s} {'actual (A)':>11s} {'fitted r0':>11s}")
for (i, j), length, r0 in zip(bonds["atoms"], lengths, bonds["kwargs"]["r0"]):
    label = f"{atoms[int(i)].symbol}{i}-{atoms[int(j)].symbol}{j}"
    print(f"{label:10s} {length:11.4f} {r0:11.4f}")

zbl_energy, zbl_forces = pieces["ZBL"]
print(f"""
`r0` is not the bond length, and it is not supposed to be.  ZBL is applied to
every pair with no switching function and no bonded exclusions -- a deliberate
choice -- so at these distances it still contributes {zbl_energy:+.1f} eV and up to
{abs(zbl_forces).max():.1f} eV/A of pure repulsion.  A Morse bond sitting at its own r0
exerts no force at all, so it could never balance that.  The fit therefore
compresses r0 until the bond pulls hard enough to cancel ZBL at the real
geometry, making r0 an effective parameter rather than a measurement.

The reference files in this directory are built the same way: the O-H `r0` in
h2o2_dynamictopology_format.jsonl is 0.0698 nm against a true bond length of
0.0966 nm, while the `bondangle` cross term keeps the true 0.142 nm.""")
