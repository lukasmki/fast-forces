"""11 -- Fixed point charges instead of charge equilibration.

`FitConfig(electrostatics="fixed")` swaps ACKS2 for the `charge` term --
DynamicTopology's `pointcharge` electrostatics: one charge per atom, carried as
an ordinary parameter, summed over the same smeared `erf(2r)/r` kernel over the
same pairs.  The two are alternatives rather than
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
import time
from dataclasses import replace

import numpy as np

import fastforces as ff
from DynamicTopology.forcefield.electrostatics import Electrostatics
from DynamicTopology.forcefield.evaluate import surface
from DynamicTopology.forcefield.ewald import Ewald
from DynamicTopology.forcefield.params import active

from fastforces.calculator import term_dict_for
from fastforces.calculators.tblite import TBLiteCalculator
from fastforces.export.openmm import exported_charges

from _common import OUTPUT, banner

banner(__doc__)


def charged_gfn2(atoms=None):
    # Writes xTB's charges onto each frame as `mulliken`, as PySCFCalculator
    # does -- what a manifest's `tblite` calculator builds, too.
    return TBLiteCalculator(method="GFN2-xTB", verbosity=0)


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
    absent = next(t for t in ("atom", "charge") if t not in params.terms)
    print(
        f"  {name:6s} electrostatics() -> {params.electrostatics()!r:10s} "
        f"and no {absent!r} block at all"
    )

# ---------------------------------------------------------------------
# where the charges come from
# ---------------------------------------------------------------------
print("\n" + "-" * 68)
raw = data.equilibrium.get_array("mulliken")
q = np.asarray(fixed.terms["charge"]["kwargs"]["q"])
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
exactly -- the template carries its formal charge, as DynamicTopology's point
charges need it to.

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
    """Just the electrostatic term, called the way DynamicTopology calls it."""
    evaluator = Electrostatics()
    td = term_dict_for(params)

    def work(frame):
        with surface(td):
            return evaluator(frame.get_positions(), frame.pbc, np.array(frame.cell), td)

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
The two fits are {energy_ratio:.2f}x and {force_ratio:.2f}x of each other -- the same fit.  That is
not a coincidence of this training set.  Methanol's fifteen atom pairs are all
within three bonds of each other, and DynamicTopology excludes every such pair
from the electrostatics, so *within* this molecule neither term contributes
anything: the two baselines are identical, and so is everything fitted against
them.  The choice of term decides what happens between molecules and across a
reaction -- where fixed charges move with a proton and ACKS2's cannot -- and it
reaches a single molecule's fit only through pairs four or more bonds apart.

What that rests on is that each field is fit against the term it is evaluated
with.  Fit against ACKS2 and evaluate with fixed charges and, in a molecule that
has such pairs, every force constant is off by the difference, which is why
`fit._nonbonded` scores through DynamicTopology's `evaluate`, which picks the
term from the block the field carries rather than assuming.

The term itself is {cost["acks2"][0] / cost["fixed"][0]:.1f}x cheaper: a pair sum with no `2n+2` solve in front
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
    f"preserved: {np.allclose(restored.terms['charge']['kwargs']['q'], q)}"
)

print("""
This is the one place the two terms are not equally good.  OpenMM has no
charge-equilibration force, so exporting ACKS2 means solving its charges once
at the geometry passed in and baking them in: an exported system that drifts
from the calculator it came from as the geometry moves, and one that cannot be
written at all without a geometry to solve at -- hence the `None` above.  Fixed
charges have nothing to bake, because the exported `erf(beta*r)/r` force *is*
the term.  That is one reason DynamicTopology's point charges use the smeared
kernel rather than a bare `1/r`: a term with a different kernel would export to
something that is not itself.""")

# ---------------------------------------------------------------------
# a charged cell, and what it costs
# ---------------------------------------------------------------------
print("-" * 68)
periodic = atoms.copy()
periodic.set_cell([15.0, 15.0, 15.0])
periodic.pbc = True

# The same field with a proton's worth of charge spread over it: an ion, and
# nothing about the term objects.
ion = copy.deepcopy(fixed)
ion.terms["charge"]["kwargs"]["q"] = q + 1.0 / len(q)

energies = {}
for name, params in (("neutral", fixed), ("net +1 e", ion)):
    energies[name] = ff.evaluate(periodic, params)[0]
    print(f"  periodic, {name:9s} {energies[name]:10.4f} eV")

# What the `k = 0` background charges an ion, from the splitting the cell fixes.
setup = Ewald(np.array(periodic.cell))
background = 0.5 * active().ccoul * setup.background * 1.0**2
print(f"\n  the k=0 background at this cell   {background:+10.4f} eV")
print(f"  difference, ion minus neutral    "
      f"{energies['net +1 e'] - energies['neutral']:+10.4f} eV")

print("""
ACKS2 constrains its charges to sum to zero; this term carries whatever it is
handed.  That used to matter under periodic boundaries, where the `k = 0`
reciprocal term is the divergent one and was simply omitted -- legitimate for a
neutral cell, and for a charged one an energy quietly missing its neutralizing
background.  It is omitted still, but the background it leaves behind,
`-pi / (kappa^2 V)` in every entry of the kernel, is now carried explicitly, so
a charged cell means the standard thing: the energy against a uniform
compensating background, which is the only thing a periodic sum over a charged
cell can mean.  The number above is that term, `ccoul/2 * background * (sum q)^2`
-- a part of the difference between the two rows, not all of it, since shifting
every charge also moves the contraction itself.

The reason it had to come back is not ions at all.  The exclusions contract the
kernel against a non-neutral weight, so each `K_ij` has to be well defined on its
own, and a constant that "cancels anyway" no longer does.

Fixed charges also work for reactions.  DynamicTopology gives every template
its own charges and puts each state's Coulomb energy on that state's diagonal,
so a proton transfer carries its excess charge with the proton -- which ACKS2's
single sum-zero constraint cannot do.  `reaction.state_parameters` scatters each
side's fragment charges onto that side, and `EVB` evaluates each diabat under
them, exactly as a DynamicTopology simulation of the same dataset would.""")
