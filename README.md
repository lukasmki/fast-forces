# fast-forces

Fast automatic parameterization of the [DynamicTopology](http://github.com/lukasmki/DynamicTopology) force field.

## Features

- Uses any ASE calculator: Fit to MLIPs, QM codes, TBLite, etc! any calculator that provides energies/forces
- Automatic conformer generation using `openconf`: You only need a SMILES string and a reference method to get a fully parameterized force field
- Writes DynamicTopology datasets directly (`.jsonl` in eV and Å, the units DynamicTopology holds them in), and exports to OpenMM
- Use the included FastForces calculator to immediately start running simulations
- All training data is saved into one extended XYZ file: Everything necessary to reproduce the fit is contained in one file
- Two electrostatics models, chosen by DynamicTopology's `global_params.electrostatics`: fragment ACKS2 charge equilibration (`acks2`), whose charges are re-solved at every geometry around each atom's reference charge `q0`, or fixed point charges (`pointcharge`). Either way the per-atom charges come from `FitConfig.electrostatics`: the reference calculation's Mulliken populations (`"mulliken"`), Merz-Kollman charges fitted to its electrostatic potential (`"esp"`, PySCF only), or zero (`"neutral"`, the default, neutral molecules only)
- Start a fit from an existing force field with `initial=`, rather than from the element table
- Parameterize a *reaction* from an atom-mapped reaction SMILES: a force field per fragment, a Sella transition state, and a fitted EVB off-diagonal coupling

## Example Usage

```python
import fastforces as ff
from tblite.ase import TBLite

atoms = ff.build("CC#N")
params = ff.parameterize(atoms, calc_factory=lambda atm: TBLite(atm))
atoms.calc = FastForces(atoms, params)

# do stuff!
```

## Reactions

`parameterize_reaction` takes an atom-mapped reaction SMILES and returns a
two-state EVB surface: a force field for each molecule involved, a transition
state located with [Sella](https://github.com/zadorlab/sella), and the
off-diagonal coupling that joins the two diabatic states.

```python
rxn = ff.parameterize_reaction(
    "[Cl-:1].[C:2]([H:3])([H:4])([H:5])[Cl:6]"
    ">>[Cl:1][C:2]([H:3])([H:4])([H:5]).[Cl-:6]",
    calc_factory=lambda atm: TBLite(atm),
)
atoms.calc = rxn.calculator(atoms)   # an ASE calculator that goes over the barrier
```

Every hydrogen must be written out and mapped: an implicit hydrogen is an atom
with no index, and in a transfer the atom that moves is usually one of them.

Three stages, and each is usable on its own:

1. **Fragments.** Every distinct molecule on either side gets its own
   `parameterize` against the same reference calculator -- the same relaxation,
   Hessian, thermal frames, conformers and torsion scans a standalone molecule
   would get. This is the expensive stage, and it is cached per molecule, so a
   species appearing on both sides is fitted once.
2. **Stationary points.** The saddle guess is built from the changing bonds --
   each fragment embedded on its own, the breaking bond opened out to the
   contact distance, the leaving group docked along it -- then refined with
   Sella at `order=1`. The two endpoints are relaxed *down* from the saddle
   along its imaginary mode, so all three frames are on one reaction path in one
   atom order.
3. **Coupling.** The connectivity change picks the form: `threebody` for an atom
   transfer, `twobody` for a barrierless fission, `rmsd` for anything else. The
   amplitude comes from inverting the 2x2 secular equation at the saddle against
   the reference barrier, so the barrier is reproduced by construction; the
   width comes from requiring the coupling to have quenched to `eps` at the
   nearer endpoint, where the diabatic picture is already correct.

The endpoints are *not* fitted, and are the honest test of the surface: the
coupling has switched off there, so what is left is the two diabatic force
fields and the unfitted nonbonded baseline between them.

A channel whose diabats never separate has no saddle to find --
`[H3O]+ + H2O` is a single well in the gas phase -- and `stationary_points`
raises rather than fitting a coupling to a path that does not exist. Supply the
three frames yourself and call `coupling.fit_threebody` directly when the path
has to be constrained.

## Datasets from a manifest

A whole dataset of molecules and reactions is described by one JSON manifest --
the same file DynamicTopology loads, with a `fit_config` block DynamicTopology
ignores; see
[`examples/hydrogen-combustion/hydrogen-combustion.json`](examples/hydrogen-combustion/hydrogen-combustion.json)
-- and fitted in one command:

```bash
fast-forces fit-manifest examples/hydrogen-combustion/hydrogen-combustion.json --dry-run   # validate, show the plan
fast-forces fit-manifest examples/hydrogen-combustion/hydrogen-combustion.json             # fit, overwriting the dataset
```

Each entry's `path` is an output stem relative to the manifest: a molecule
writes `<path>.jsonl` and `<path>.xyz`, a reaction `<path>.xyz`, `<path>.jsonl`
and one jsonl per diabatic state. The `fit_config` section takes any `FitConfig`
field plus:

| key | default | meaning |
| --- | --- | --- |
| `calculator` | `{"name": "tblite"}` | `tblite` (`method`; its xTB charges are what `electrostatics: "mulliken"` reads) or `pyscf` (`xc`, `basis`, and `pcm`/`pcm_eps` for implicit solvent, e.g. `"IEF-PCM"`); other keys go to the constructor |
| `workdir` | `"training"` | where the full training sets go, mirroring each `path` |
| `embed_seed` | 42 | seed for conformer embedding and the saddle guess |
| `eps` | 1e-3 | coupling quench at the nearer endpoint |

Reaction SMILES need not be atom-mapped: heavy atoms correspond by order of
appearance and hydrogens are assigned to change as few bonds as possible
(`reaction.map_atoms`), so `[OH3+].O>>O.[OH3+]` is the Grotthuss transfer.
Molecules are fitted first and reactions reuse them as fragments. A reaction
entry may give `"frames"` -- reactant, TS and product in one extxyz -- to skip
the saddle search, which is what a barrierless gas-phase channel needs, and
`"amplitude"` to fix the coupling amplitude.

A reaction whose `<path>.xyz` is already in place also skips the search: its
reactant, TS and product are taken as the reaction's geometry. That is how
[`examples/hydrogen-combustion`](examples/hydrogen-combustion) is set up, with
the stationary points of DynamicTopology's HCombustion dataset in `reactions/`.
Only the geometries are used. They are renumbered onto the mapped SMILES by
matching elements and the bonds of both ends, since a dataset's atom order is
rarely the SMILES's. Each frame is then relabelled with the manifest's
calculator at the entry's spin, because a barrier from another method would
not sit on the same zero as the fragment fits. The labelled frames are cached
in `workdir` like a searched path. To search again, delete both the cached
frames and `<path>.xyz`. An explicit `"frames"` file outranks the output file.

Spin is `2S`, the number of unpaired electrons, and is used only for fitting --
DynamicTopology reads nothing from an entry but its `id` and `path`. A molecule
entry may give `"spin"`; otherwise it is read from the SMILES radicals. A
reaction's frames default to every reactant radical high-spin coupled
(`[O].[OH]` runs at 3, not the doublet the electron count alone would pick). On
a reaction entry `"spin"` sets the transition-state search and both endpoint
relaxations, and `"spin_r"`, `"spin_ts"` and `"spin_p"` override the reactant,
the transition state and the product individually:

```json
{"id": 2, "smiles": "[H:1][H:2].[O:3]>>[O:3][H:2].[H:1]", "path": "reactions/rxn_02",
 "spin": 2, "spin_p": 0}
```

A fission is fitted from its reactant alone, so it takes `"spin"` or `"spin_r"`
and refuses the other two. A value the electron count does not allow is refused
when the manifest is loaded, and a cached training set or `"frames"` file at a
different spin is refused rather than reused. So is an output `<path>.xyz`
that states a different spin, since its geometries are stationary points of
another surface.

Training sets already in `workdir` are reused, so a rerun refits without the
reference calculator; delete one to regenerate it. A molecule training set
written before the per-bond asymptote was fitted (see below) gets its fragment
single points added on the next run, and every other frame in it is kept.
`global_params` is
DynamicTopology's own block (`forcefield.params.ForceFieldParams`) and it is
*applied*: the whole manifest is fitted under it, so the dataset is fitted at
exactly the constants it states. The two `electrostatics` keys are the two
halves of one choice: `global_params.electrostatics` is how a simulation uses
the charges (`"acks2"`, the default, or `"pointcharge"`), and
`fit_config.electrostatics` is where they come from (`"mulliken"`, `"esp"` or
`"neutral"`, the default). Under `acks2` they are each atom's reference charge
`q0`, which is what keeps an ion's formal charge on it; under `pointcharge` they
are the fixed charges. A charged molecule needs `"mulliken"` or `"esp"`, and so
does `pointcharge`.

## Running on a CPU node

A fit with a DFT reference is almost entirely independent single points -- the
`6N` Hessian displacements, the thermal frames, every point of every torsion
scan, the fragments and the free atoms -- and a reference code does not scale
across a node by itself. PySCF B3LYP/6-31G* on PABA is 18.6 s per energy and
gradient on 4 threads and 6.9 s on 32, and *slower* again past that. So
`fastforces.parallel` runs the calculations side by side, each in its own
forked process:

```bash
fast-forces fit-manifest Water.json --workers 32 --threads 4
FASTFORCES_WORKERS=32 FASTFORCES_THREADS=4 python my_fit.py   # any script
```

or `fastforces.parallel.configure(workers=32, threads=4)` before the first
reference calculation. `examples/fit-manifest.slurm` is a whole-node job.
Within a molecule the Hessian, thermal, conformer, torsion, fragment and atom
frames are each one batch, and the two downhill relaxations from a saddle run
together; a manifest fits its molecules side by side and then its reactions,
dividing the node between them. A serial chain -- a relaxation, a Sella search --
runs in one worker with up to 32 threads.

| B3LYP/6-31G*, 128-core Milan | serial, 32 threads | 32 workers x 4 threads |
|---|---|---|
| ethanol, whole fit | 1204 s | 285 s |
| of which torsion scans (48 relaxations) | 1006 s | 207 s |
| of which Hessian (54 single points) | 70 s | 15 s |
| of which thermal frames (40) | 54 s | 12 s |
| H2O2, whole fit | 124 s | 31 s |

What does not parallelize is a serial chain. PABA's equilibrium relaxation alone
(BFGS to `fmax = 1e-3`) took about 32 minutes on 32 threads, the same either
way. A batch is as slow as its slowest wave: ethanol's 48 torsion points on 32
workers run as two, so a molecule with many scan points can do better with more,
narrower workers (`--workers 64 --threads 2`).

**The equilibrium relaxation can start from GFN2-xTB.** `"prerelax":
"GFN2-xTB"` in a manifest's `fit_config` (or `FitConfig(prerelax=...)`)
relaxes the molecule with tblite first, finite-differences tblite's Hessian
there, and starts the reference's BFGS from that geometry with that Hessian as
its first guess -- in place of BFGS's uniform 70 eV/A^2, which is right for no
mode of a real molecule. Translations and rotations keep the 70: a DFT grid is
not rotationally invariant, so the reference's forces carry a small torque
that a soft guess would answer with full-length rotations. The reference
still converges to its own `fmax`, so the minimum is the same to within it.

| relaxation to `fmax = 1e-3` | plain | pre-relaxed |
|---|---|---|
| ethanol, B3LYP/6-31G*, 32 threads | 63 evaluations, 118 s | 11 evaluations, 51 s |
| PABA, same, SCF `conv_tol = 1e-9` | not converged after 35 | 11 evaluations |
| PABA, GFN1-xTB as the reference | 96 evaluations | 7 evaluations |

The two ethanol minima agree to 1e-4 A RMSD. It is opt-in because a different
method can carry a flexible molecule whose input geometry sits between two
conformers into the other well.

**The PySCF calculator converges its SCF to `conv_tol = 1e-9`** (PySCF's own
default; a `"conv_tol"` key in `fit_config.calculator` overrides it).
It used to be 1e-6, where the SCF stops at an orbital gradient of ~1e-3 and
the forces inherit an error of that order -- the size of `fmax = 1e-3`, so
near convergence BFGS updated its Hessian from differences that were mostly
noise. PABA, plain or pre-relaxed, was still wandering between 1e-2 and 5e-2
eV/A after 75 evaluations at 1e-6, which is most likely where its 32-minute
relaxation went; at 1e-9 the pre-relaxed run converged in 11, each evaluation
3-4x dearer but a tenth as many. Every frame's energy moves in its last digits
with the change, so a training set labelled at 1e-6 and one labelled now are
not byte-identical.

**The training sets are the serial ones.** Every frame is labelled by a fresh
calculator either way, so at one thread per calculation a parallel fit is byte
for byte the serial fit -- training file and `.jsonl` -- which
`tests/test_parallel.py` checks. At more threads the last digits move with the
thread count, as they do between two serial runs at different
`OMP_NUM_THREADS`: tblite reduces in parallel, and PySCF's SCF converges to its
`conv_tol` along a slightly different path (energies within ~1e-5 eV at the
old 1e-6; less at 1e-9).

Two constraints come with it. The workers are forked, so the calculator factory
may be a lambda or a closure; but OpenMP runtimes are not fork-safe, so the
parent process holds its own to one thread and never runs a reference
calculation itself -- configure workers *before* anything in the process has
run a multi-threaded tblite or PySCF calculation, which is why the environment
variables are read at import. And anything a factory does as a side effect
happens in a worker: four `tests/test_manifest.py` tests that count calculator
constructions through a closure fail under `FASTFORCES_WORKERS`, and every
other test passes.

## Examples

Numbered, runnable examples live in [`quickstart/`](quickstart/) -- start with
`quickstart/01_quickstart.py` and read `quickstart/README.md` for the index.
`quickstart/10_reaction.py` walks through the reaction pipeline end to end, and
`quickstart/11_fixed_charges.py` fits one molecule with each electrostatics
model against the same reference data and compares them. A complete fitted
dataset -- manifest, molecules, reactions and the training sets they were fitted
from -- is in [`examples/hydrogen-combustion/`](examples/hydrogen-combustion/);
it is the input to the manifest command above and loads straight into
DynamicTopology.

## Workflow Overview

1. Assemble force field terms from bond graph
2. Compute atomization energy and set bond dissociation energies
3. [optional] Openconf generate_conformers with "ensemble" preset
4. [optional] Run molecular dynamics to obtain extra structure
5. Fit or use reference nonbonded potential parameters
6. Fit all force field terms via regression

## Where a bond ends up when it breaks

A Morse bond's stretched branch climbs a well `Dw = D + h` and levels off `h`
above zero. An EVB fission needs the bonded state to finish *above* its
fragments, so that the two diabats cross. Near-equilibrium frames cannot see
where the curve ends, and `D` is scaled so that all the bonds together carry
the molecule's atomization energy, which says nothing about where any one bond's
fragments sit. Left alone, `h = bond_asymptote` put the limit wherever `D`
happened to fall: 0.28 eV *below* two H atoms for H2 at B3LYP, whose diabats
then never crossed.

So each molecule's training set also carries the two fragments of each bond
class. They are two single points at the frozen equilibrium geometry, with
each fragment keeping its atoms' formal charges and its SMILES radicals plus
the electrons the cut releases (the O of OH comes out triplet). The fit holds
each class's `Dw` so that pulling the bond apart ends exactly `bond_asymptote`
above those two energies, which is where DynamicTopology's defaults put it
for a bond whose `D` is its dissociation energy. `h = Dw - D` is written to
the `.jsonl`, and the fit report lists it as `asymptote_h`. It can come out
negative when `D` overshoots the fragments. Ring bonds, and molecules built
without a SMILES, keep `bond_asymptote`.

## Starting from an existing force field

`fit`, `fit_from_file` and `parameterize` take `initial=` -- a `Parameters`, a
term dict, or a path to a jsonl file:

```python
params = ff.parameterize(atoms, calc_factory=..., initial="previous.jsonl")
```

Terms are matched to the topology by their atom slots, and anything the supplied
field does not cover keeps its ordinary default. Only what the fit holds fixed
can be supplied: the equilibrium values (bond `r0` excepted, which is always
measured), the bond depths `D` and asymptotes, and the nonbonded baseline. Every
force constant comes from one global least squares that has no starting point,
so a supplied `k` does not reach the result.

### How the bonds are fitted

Every bonded term is a force constant times a function of the geometry, so with
the equilibrium values fixed every `k` is one bounded linear least-squares solve.
The bond joins that solve as a harmonic spring `k*dr**2/2` about its measured
`r0`, and is written as the Morse bond a reactive simulation needs: a Morse bond
`D*(1 - exp(-a*dr))**2 - D`, `a = sqrt(k/2D)`, is `k*dr**2/2 - D` to second
order, so the harmonic `k` is its curvature at `r0` whatever `D` is. `h` comes
from the fragments (above).

A molecule's total energy is the sum of its force field terms plus `E0`, a
fixed scalar that shifts its whole potential energy surface. The shift is what
keeps the relative energies of molecules right: a reactant state and a product
state, each the sum of its molecules, have to carry the reference offset
between them at the two ends of a reaction coordinate, so every molecule has
to sit on the reference calculator's absolute scale. `E0` is the offset that
puts it there.

It is not in the solve. Every energy residual is mean-centered, and `E0` is
read off the leftover mean afterwards, plus the `-D` per bond the Morse form
carries at its minimum. So a refit from a field's own output returns it to
round-off, and so does one from a field with every `k` scrambled.

### Where the bond depths come from

The frames near equilibrium see `D` only through anharmonicity, so it is not
fitted to them. It starts from the element table (or `initial=`), and every
depth is then scaled by one factor so the bonds carry the molecule's
atomization energy, `E(molecule) - sum_i E(atom_i)`. The free atoms are one
single point per element in the training set, neutral and at their Hund's-rule
spin, computed with the same reference calculator. Because `E0` moves with the
depths, a scale changes how much of the molecule's total the bonds hold and how
much the shift holds, never the total, so no relative energy moves. The scale
is closed form: it is the one that leaves `E0` equal to the summed free-atom
energies, which is the same for reactants and products, so the offset between
them is carried by the terms alone. "Carry" means
the whole field at equilibrium, as `refine` matches it, so the depths add up to
the atomization energy less what the angles, cross terms and nonbonded
baseline hold there. The fit report lists the factor as `depth_scale`.

For an ion the atoms are still neutral, as `fast-forces label` references a
dataset, so its atomization energy includes an ionization energy or electron
affinity. A training set without atom frames keeps the unscaled table depths;
`add_atom_frames` adds them, and a manifest rerun does so on its cached sets.

The cost is anharmonicity. The harmonic spring cannot follow a bond stretched
0.1 A in a 500 K mode frame, and the written Morse bond adds curvature the fit
never saw. On H2O2 against GFN2-xTB:

| bond fit | energy RMSE (eV) | force RMSE (eV/A) | mode-frame force RMSE (eV/A) |
|---|---|---|---|
| nonlinear Morse `(r0, k, D)`, alternated with the linear block | 0.0017 | 0.025 | 0.040 |
| linear harmonic `k`, measured `r0`, `D` carrying the atomization energy | 0.0111 | 0.096 | 0.154 |

The equilibrium, Hessian and torsion frames move by at most 2.5 meV and 0.011
eV/A; the rest of the difference is on the stretched frames.
`quickstart/09_starting_point.py` shows a refit returning its own output.
