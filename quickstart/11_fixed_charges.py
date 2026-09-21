"""11 -- Fixed point charges instead of charge equilibration.

`FitConfig(electrostatics="fixed")` swaps ACKS2 for the `coulomb` term: one
charge per atom, carried as an ordinary parameter, summed over the same smeared
`erf(2r)/r` kernel over the same pairs.  The two are alternatives rather than
additions -- a field carries one or the other, and `Parameters.electrostatics()`
is what says which -- so this is a different force field, not a reparametrized
one.

What it buys is that the electrostatics becomes a plain function of the
geometry: no `2n+2` linear system solved at every step, no `dQ/dr` adjoint
behind the forces, and nothing to bake in when the field is exported.  What it
costs is the thing that machinery was there for -- charges that redistribute as
bonds stretch and as molecules approach.

The charges have to come from somewhere, and they come from the reference
calculation: a per-atom `mulliken` array on the training frames.

    uv run python quickstart/11_fixed_charges.py
"""

import copy
import textwrap
import time
import warnings
from dataclasses import replace

import numpy as np
from ase.calculators.calculator import all_changes
from tblite.ase import TBLite

import fastforces as ff
from fastforces.export.openmm import exported_charges
from fastforces.forcefield import electrostatic_evaluator

from _common import OUTPUT, banner

banner(__doc__)


class ChargedTBLite(TBLite):
    """GFN2-xTB, with its partial charges left on the frame as `mulliken`.

    `calculators.pyscf.PySCFCalculator` does this itself -- it has the density
    matrix in hand anyway -- and the fit reads the array rather than asking a
    calculator for it, so a reference method is not required to be PySCF.  This
    is what any ASE calculator that reports `charges` needs to drive a
    fixed-charge fit, and `sampling.label` carries whatever a calculator writes
    onto a frame into the training file.

    The array is named for Mulliken because that is what PySCF puts there; what
    xTB reports is its own population analysis, which is a different
    approximation to the same ill-defined quantity.  Neither is more correct
    than the other, and that is the honest caveat about this whole term.
    """

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties or ["energy"], system_changes)
        # The caller's object, not `self.atoms`: the array has to land on the
        # frame `sampling.label` is about to freeze, the way PySCF writes its
        # bond orders.
        target = atoms if atoms is not None else self.atoms
        target.set_array("mulliken", np.asarray(self.results["charges"], float), float)


def charged_gfn2(atoms=None):
    return ChargedTBLite(method="GFN2-xTB", verbosity=0)


# Methanol: a hydroxyl, a methyl rotor, and enough polarity for the
# electrostatics to be worth arguing about.
config = ff.FitConfig(n_conformers=0, torsion_step_deg=30.0)
path = OUTPUT / "methanol.xyz"
if not path.exists():
    ff.parameterize(ff.build("CO"), charged_gfn2, config=config, training_set=str(path))

# One training set, fit twice.  Everything below compares two force fields
# built from exactly the same reference data, so every difference between them
# is the electrostatic term and nothing else.
data = ff.io.read_training_set(str(path))
atoms = data.equilibrium.copy()
acks2 = ff.fit_from_file(str(path), config=config)
fixed = ff.fit_from_file(str(path), config=replace(config, electrostatics="fixed"))

print(f"{atoms.get_chemical_formula()} from {path.name}: {data.summary()}\n")
for name, params in (("acks2", acks2), ("fixed", fixed)):
    absent = next(t for t in ("atom", "coulomb") if t not in params.terms)
    print(
        f"  {name:6s} electrostatics() -> {params.electrostatics()!r:10s} "
        f"and no {absent!r} block at all"
    )

# ---------------------------------------------------------------------
# where the charges come from
# ---------------------------------------------------------------------
print("\n" + "-" * 68)
raw = data.equilibrium.get_array("mulliken")
q = np.asarray(fixed.terms["coulomb"]["kwargs"]["q"])
classes = ff.enumerate_terms(
    data.equilibrium, ff.perceive(data.equilibrium)
).atom_classes

print(f"{'atom':7s} {'class':>6s} {'reference q':>12s} {'fitted q':>10s}")
for atom, cls, reference, fitted in zip(atoms, classes, raw, q):
    print(f"{atom.symbol}{atom.index:<6d} {cls:6d} {reference:12.4f} {fitted:10.4f}")
print(f"{'sum':7s} {'':6s} {raw.sum():12.4f} {q.sum():10.4f}")

spreads = [np.ptp(raw[classes == c]) for c in range(classes.max() + 1)]
worst = int(np.argmax(spreads))
members = "".join(f"{atoms[i].symbol}{i} " for i in np.flatnonzero(classes == worst))
print(f"""
`q` is the reference charge averaged within each atom equivalence class, and
class {worst} -- {members.strip()} -- is why.  Those are one class because the graph cannot
tell them apart, but the charges come out of a single geometry, and this one
puts {spreads[worst]:.3f} e between the widest pair of them.  Freezing that asymmetry in
would give a rotor with no electrostatic torsion a spurious one, which the
relaxed torsion scans in the training set would then have to fight.  A
class-wise mean moves no charge between classes, so the total is preserved
exactly -- the condition `forcefield/ewald.py` needs of it.

Neither field's charges are *fitted*.  Both are part of the fixed baseline the
bonded terms are fit against, which is the point example 06 makes about `r0`:
change the baseline and the bonded parameters absorb the difference.  Fitting
charges against reference electrostatics is future work, the same entry
`elements.py` already records for the ACKS2 softness.""")

# ---------------------------------------------------------------------
# what the two terms do as the geometry moves
# ---------------------------------------------------------------------
print("-" * 68)
frames = data.of_kind("mode", "torsion")
charges = {
    name: np.array(
        [ff.FastForces(f, params).get_property("charges", f) for f in frames]
    )
    for name, params in (("acks2", acks2), ("fixed", fixed))
}

print(f"charge range over the {len(frames)} distorted training geometries:")
print(f"{'atom':7s} {'ACKS2':>19s} {'fixed':>19s}")
for atom in atoms:
    cells = [
        f"{charges[n][:, atom.index].min():+.4f}..{charges[n][:, atom.index].max():+.4f}"
        for n in ("acks2", "fixed")
    ]
    print(f"{atom.symbol}{atom.index:<6d} {cells[0]:>19s} {cells[1]:>19s}")

swing = max(np.ptp(charges["acks2"][:, i]) for i in range(len(atoms)))
print(f"""
The fixed column has no range at all: those charges are parameters, and a
parameter does not know the geometry moved.  The ACKS2 column swings by up to
{swing:.4f} e, and that is the whole reason the term exists -- it is also where its
forces pick up a response contribution, which the fixed term has no analogue of
because it needs none.""")

# ---------------------------------------------------------------------
# what each one costs, and what each one predicts
# ---------------------------------------------------------------------
print("-" * 68)


def per_call(work, frames, repeats=5):
    """Seconds per call of `work(frame)`, averaged over the frames."""
    start = time.perf_counter()
    for _ in range(repeats):
        for frame in frames:
            work(frame)
    return (time.perf_counter() - start) / (repeats * len(frames))


def term_only(params):
    """Just the electrostatic evaluator, called the way the calculator calls it."""
    evaluator = electrostatic_evaluator(params)

    def work(frame):
        return evaluator(
            frame.get_positions(), frame.pbc, np.array(frame.cell), params.terms
        )

    return work


def whole_field(params):
    """Every term: what an MD step actually pays."""
    calc = ff.FastForces(atoms, params)

    def work(frame):
        calc.get_potential_energy(frame)
        calc.get_forces(frame)

    return work


cost = {}
print(
    f"{'field':8s} {'E_rmse (eV)':>12s} {'F_rmse (eV/A)':>14s} {'term':>10s} {'whole field':>13s}"
)
for name, params in (("acks2", acks2), ("fixed", fixed)):
    cost[name] = (
        per_call(term_only(params), frames),
        per_call(whole_field(params), frames),
    )
    print(
        f"{name:8s} {params.report['energy_rmse_eV']:12.5f} "
        f"{params.report['force_rmse_eV_A']:14.5f} "
        f"{cost[name][0] * 1e3:7.3f} ms {cost[name][1] * 1e3:10.3f} ms"
    )

energy_ratio = fixed.report["energy_rmse_eV"] / acks2.report["energy_rmse_eV"]
force_ratio = fixed.report["force_rmse_eV_A"] / acks2.report["force_rmse_eV_A"]
print(f"""
The two fits land in the same ballpark rather than on top of each other: the
fixed-charge field is {energy_ratio:.1f}x the energy RMSE and {force_ratio:.1f}x the force RMSE of the
ACKS2 one here.  Since neither term is fitted, what that difference measures is
the *baseline*, not the fit -- how much of the reference's electrostatics was
already accounted for before the bonded block started work, and therefore how
much of it the bonded block had to absorb.  `tests/test_fit.py` asserts a
fixed-charge fit stays within 2x of the ACKS2 one for exactly that reason.

What it rests on is that each field was fit against the term it is evaluated
with.  Fit against ACKS2 and evaluate with fixed charges and every force
constant is off by the difference, which is why `fit._nonbonded` asks
`Parameters.electrostatics()` rather than assuming, and why refitting with the
other setting produces a different force field rather than a reparametrized
one.

The term itself is {cost["acks2"][0] / cost["fixed"][0]:.0f}x cheaper: a pair sum with no `2n+2` solve in front
of it and no adjoint behind it.  The whole field is only {cost["acks2"][1] / cost["fixed"][1]:.2f}x cheaper, because
on a six-atom molecule the bonded block is most of the work -- but the solve is
the piece that scales worst, so that ratio is a floor rather than the number to
expect on a real system.""")

# ---------------------------------------------------------------------
# exporting
# ---------------------------------------------------------------------
print("-" * 68)
positions = atoms.get_positions()
print("the charges `export.openmm` writes into the exported system:")
print(f"  acks2, no geometry     {exported_charges(acks2, None)}")
print(f"  acks2, at equilibrium  {exported_charges(acks2, positions).round(4)}")
print(f"  fixed, no geometry     {exported_charges(fixed, None).round(4)}")

jsonl = OUTPUT / "methanol_fixed.jsonl"
fixed.to_jsonl(str(jsonl))
restored = ff.as_parameters(str(jsonl))
print(
    f"\nwrote {jsonl.name}; read back -> {restored.electrostatics()!r}, charges "
    f"preserved: {np.allclose(restored.terms['coulomb']['kwargs']['q'], q)}"
)

print("""
This is the one place the two terms are not equally good.  OpenMM has no
charge-equilibration force, so exporting ACKS2 means solving its charges once
at the geometry passed in and baking them in: an exported system that drifts
from the calculator it came from as the geometry moves, and one that cannot be
written at all without a geometry to solve at -- hence the `None` above.  Fixed
charges have nothing to bake, because the exported `erf(beta*r)/r` force *is*
the term.  That is the third and deciding reason `forcefield/coulomb.py` smears
its kernel instead of using a bare `1/r`: the exported system has always had
this form, and a term with a different kernel would export to something that is
not itself.""")

# ---------------------------------------------------------------------
# the one thing the term will not check for you
# ---------------------------------------------------------------------
print("-" * 68)
periodic = atoms.copy()
periodic.set_cell([15.0, 15.0, 15.0])
periodic.pbc = True

# The same field with a proton's worth of charge spread over it: an ion, and
# nothing about the term objects.
ion = copy.deepcopy(fixed)
ion.terms["coulomb"]["kwargs"]["q"] = q + 1.0 / len(q)

caught_text = None
for name, params in (("neutral", fixed), ("net +1 e", ion)):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        energy = ff.evaluate(periodic, params)[0]
    # Other libraries warn about other things; this is the one being shown.
    ours = [w for w in caught if "net charge" in str(w.message)]
    if ours:
        caught_text = str(ours[0].message)
    print(f"  periodic, {name:9s} {energy:10.4f} eV   {'warned' if ours else 'quiet'}")
print(
    "\n" + textwrap.fill(caught_text, 68, initial_indent="  ", subsequent_indent="  ")
)

print("""
ACKS2 constrains its charges to sum to zero; this term carries whatever it is
handed.  That matters only under periodic boundaries, where `ewald` omits the
`k = 0` reciprocal term -- legitimate for a neutral cell, and for a charged one
an energy missing its neutralizing background.  It warns once per evaluator
rather than raising, because an isolated ion is a perfectly ordinary thing to
evaluate under open boundaries, where that term does not exist at all.  Charges
read off a neutral molecule sum to zero by construction, so the warning is
about what is done with them afterwards.

One further limit, and it is why this example fits a molecule rather than a
reaction: `reaction.state_parameters` refuses a fragment carrying fixed
charges.  `EVB` evaluates electrostatics once for both states, which is exact
for ACKS2 -- the charges equilibrate over whatever they are handed, so there is
nothing state-specific about them -- and simply wrong for two fragments whose
fixed charges differ on the two sides of a reaction.  Until electrostatics
moves onto each state's diagonal, example 10's path stays on ACKS2.  Which is
the trade in one line: fixed charges for a fixed topology, and charge
equilibration for chemistry that changes one.""")
