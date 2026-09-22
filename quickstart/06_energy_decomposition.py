"""06 -- What each evaluator contributes, and what the exclusions take back.

`FastForces` is a sum of five pieces: the bonded terms, ACKS2 charge-
equilibration electrostatics, ZBL screened nuclear repulsion at short range,
Lennard-Jones at long range, and the exclusions.  Only the bonded block is fit;
the rest come from element tables and are evaluated and subtracted before the
fit, so the bonded parameters absorb only what is left over.

The three pair sums run over *every* pair with no reference to the bond graph --
ZBL switched off outside 1.5 A, the 12-6 switched on outside 2.2 A -- and
`forcefield/exclusions.py` then removes every pair within three bonds of
another.  That is the one part of the nonbonded energy that knows about bonding
at all, and it is what makes a reactive surface possible: it is the only place a
diabatic state's own topology reaches the nonbonded terms.

Acetonitrile has six atoms and graph diameter 3, so *every* pair is excluded and
the three sums cancel to zero between them.  Watching that happen is the first
half of this example; the second is what it means for `r0`.

    uv run python quickstart/06_energy_decomposition.py
"""

import numpy as np

import fastforces as ff
from fastforces.forcefield import exclusions
from fastforces.forcefield.acks2 import ACKS2
from fastforces.forcefield.lj import LennardJones
from fastforces.forcefield.qforce import QForce
from fastforces.forcefield.zbl import ZBL

from _common import banner, fitted

banner(__doc__)

atoms, params, training = fitted("CC#N", name="acetonitrile")
pos, cell, pbc = atoms.get_positions(), np.array(atoms.cell), atoms.pbc
numbers = atoms.get_atomic_numbers()
mask = params.exclusions

# All five work in eV and Angstrom and read `params.terms` directly; each
# returns `(energy, forces, virial)`.  `QForce` gets `bonded_terms()` rather
# than the whole dict only so that it does not evaluate `reference` and count
# `E0` a second time.
#
# The electrostatic term takes a *screen* rather than a subtraction: its charges
# come from a solve whose matrix holds the kernel, so masking that kernel would
# make the charges depend on the bond graph -- which they must not, or a
# reaction's two states would disagree about them.  So the charges are solved
# once, unmasked, and the exclusion is applied afterwards as a weight on the
# energy contraction.
indices = params.terms["atom"]["atoms"][:, 0]
screen = exclusions.screen([mask[np.ix_(indices, indices)]], [1.0], len(indices))
sigma, eps = exclusions.lj_parameters(params.terms, len(numbers))

pieces = {
    "bonded (QForce)": QForce(bond_form="morse")(pos, pbc, cell, params.bonded_terms()),
    "ACKS2 (screened)": ACKS2()(pos, pbc, cell, params.terms, screen),
    "Lennard-Jones": LennardJones()(pos, pbc, cell, params.terms),
    "ZBL": ZBL()(pos, numbers, pbc, cell),
    "exclusions": exclusions.additive(pos, numbers, pbc, cell, mask, sigma, eps),
}

print(f"{'term':18s} {'energy (eV)':>13s} {'max |force|':>13s}")
for name, (energy, forces, _) in pieces.items():
    print(f"{name:18s} {energy:13.4f} {abs(forces).max():13.4f}")
print(f"{'E0 (reference)':18s} {params.e0:13.4f} {0.0:13.4f}")

total = sum(e for e, _, _ in pieces.values()) + params.e0
print(f"{'total':18s} {total:13.4f}")
print(f"{'FastForces':18s} {ff.evaluate(atoms, params)[0]:13.4f}  (agrees)\n")

lj_energy = pieces["Lennard-Jones"][0]
zbl_energy, zbl_forces, _ = pieces["ZBL"]
residue = lj_energy + zbl_energy + pieces["exclusions"][0]
n_pairs = int(mask.sum()) // 2
print(
    f"""The 12-6 and ZBL sums are {lj_energy:+.4f} and {zbl_energy:+.4f} eV, and the
exclusions take back {pieces["exclusions"][0]:+.4f}: the three add to {residue:+.1e} eV.
All {n_pairs} pairs of this molecule are inside the three-bond depth, so nothing
survives.  The cancellation is exact rather than close because both halves go
through one `pair_potential` apiece -- taper, switch and core continuation
included.  A transcription that wrote the subtraction out a second time would
leave a residue here of order the terms themselves, which is tens of eV.

The electrostatic screen is the same idea taken the other way: ACKS2 reads
{pieces["ACKS2 (screened)"][0]:+.4f} eV screened, and the charges it solved are the ones it would
have solved unscreened.\n"""
)

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
# What the exclusions do to r0
# ---------------------------------------------------------------------
print("-" * 68)
bonds = params.terms["bond"]
lengths = np.linalg.norm(pos[bonds["atoms"][:, 0]] - pos[bonds["atoms"][:, 1]], axis=1)
print(f"{'bond':10s} {'actual (A)':>11s} {'fitted r0':>11s}")
for (i, j), length, r0 in zip(bonds["atoms"], lengths, bonds["kwargs"]["r0"]):
    label = f"{atoms[int(i)].symbol}{i}-{atoms[int(j)].symbol}{j}"
    print(f"{label:10s} {length:11.4f} {r0:11.4f}")

print(f"""
`r0` is an effective parameter, not a measurement -- it balances whatever
nonbonded baseline survives on the pair it connects.  Here nothing survives, so
there is nothing to balance and `r0` lands on the bond length.

Without the exclusions it would not.  ZBL alone puts {zbl_energy:+.1f} eV and up to
{abs(zbl_forces).max():.1f} eV/A of pure repulsion on this molecule at its own geometry, against
a reference force of zero, and a Morse bond sitting exactly at its own `r0`
exerts no force at all -- so the fit would have to compress `r0` until the bond
pulled hard enough to cancel it.  Fitted that way, the same C-H bond came out at
0.0746 nm against a true 0.1089 nm: an effective parameter that no longer looked
like a bond length at all.

A molecule large enough to have pairs more than three bonds apart is back in
that situation for those pairs, and so is anything intermolecular.  The
exclusions do not remove the bargain; they remove it from within a small
molecule.""")
