# fast-forces

Fast automatic parameterization of force fields -- for
[DynamicTopology](../DynamicTopo).

**The two packages split one job.** DynamicTopology is the inference package:
the force field itself (bonded terms, ACKS2 or fixed point charges, tapered ZBL,
switched 12-6, the intramolecular exclusions, the EVB couplings) and the
reactive MD that runs on it. fast-forces is everything that produces its
parameters: sampling a reference calculator, fitting each molecule, locating
reactions and fitting their couplings, and writing a dataset DynamicTopology
loads. fast-forces carries no force field of its own -- every energy it fits
against is `DynamicTopology.forcefield.evaluate`, the single-topology sum
DynamicTopology's `System` puts on a diabat, so a template is scored during the
fit exactly as the simulation will score it. The equations are in
DynamicTopology's `src/DynamicTopology/forcefield/REFERENCE.md`.

## Features

- Uses any ASE calculator: Fit to MLIPs, QM codes, TBLite, etc! any calculator that provides energies/forces
- Automatic conformer generation using `openconf`: You only need a SMILES string and a reference method to get a fully parameterized force field
- Writes DynamicTopology datasets directly (`.jsonl` in eV and Å, the units DynamicTopology holds them in), and exports to OpenMM
- Use the included FastForces calculator to immediately start running simulations
- All training data is saved into one extended XYZ file: Everything necessary to reproduce the fit is contained in one file
- Two electrostatics models: ACKS2 charge equilibration, whose charges are re-solved at every geometry, or fixed point charges taken from the reference calculation's Mulliken populations (`FitConfig(electrostatics="fixed")`, DynamicTopology's `pointcharge`)
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
| `calculator` | `{"name": "tblite"}` | `tblite` (`method`; its xTB charges are what `electrostatics: "fixed"` freezes) or `pyscf` (`xc`, `basis`, and `pcm`/`pcm_eps` for implicit solvent, e.g. `"IEF-PCM"`); other keys go to the constructor |
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
different spin is refused rather than reused.

Training sets already in `workdir` are reused, so a rerun refits without the
reference calculator; delete one to regenerate it. `global_params` is
DynamicTopology's own block (`forcefield.params.ForceFieldParams`) and it is
*applied*: the whole manifest is fitted under it, so the dataset is fitted at
exactly the constants it states (REFERENCE.md §7.2). Its `electrostatics` and
`fit_config.electrostatics` name the same choice; either may be given, and both
have to agree.

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

## Starting from an existing force field

`fit`, `fit_from_file` and `parameterize` take `initial=` -- a `Parameters`, a
term dict, or a path to a jsonl file:

```python
params = ff.parameterize(atoms, calc_factory=..., initial="previous.jsonl")
```

Terms are matched to the topology by their atom slots, and anything the supplied
field does not cover keeps its ordinary default. The bond parameters start a
local nonlinear solve, and the equilibrium values and nonbonded baseline are
held fixed rather than fit, so supplying any of those changes the result; the
remaining force constants come from a global least squares that has no starting
point. `quickstart/09_starting_point.py` works through the distinction.

### Is fitting idempotent?

Yes, to within `cycle_tol`. Refitting a converged field from its own output
stops after two cycles and predicts the same thing to five figures; the
parameters land within a few percent on the shallowest cross terms and a
fraction of a percent on the well-determined ones. Tightening `cycle_tol` from
its default `1e-4` to `1e-8` shrinks that drift ~80x for about 40 extra cycles
-- drift that shrinks with the tolerance is a fixed point being approached,
where a flat direction being walked would not.

That holds because `E0` is not fitted. Every energy residual in the fit is
mean-centered, so a constant shift of the whole profile costs nothing anywhere,
and `E0` is read off the leftover mean once the fit has converged. Fitting it
alongside `D` put a flat direction straight through the parameter space: the
Morse form carries a constant `-D` per bond, and with `a = sqrt(k/2D)` its well
is `k*dr**2/2` to second order, so near equilibrium `D` reaches the energy
through that constant and little else. The nonlinear block would raise `D` and
the linear block would raise `E0` to compensate, for hundreds of cycles, with
the residual barely moving.

Removing the offset from the fit is what fixed it, and it costs nothing:

| | cycles to converge | energy RMSE (eV) | force RMSE (eV/A) |
|---|---|---|---|
| fitting `E0` | 620 | 0.0273 | 0.280 |
| deriving `E0` at the end | 21 | 0.0269 | 0.258 |

`n_cycles` now defaults to 200 rather than 25. It is a cap the fit exits well
inside, and it has to be one: an interrupted fit is neither idempotent nor a
converged residual for `E0` to be read off. Note that a training file records
the config it was written with and `fit_from_file` prefers that recorded one, so
files written before this change keep the old budget unless you pass a config.

One thing to watch: `D` now reaches the energy only through the anharmonicity it
describes, and `D -> infinity` is the harmonic limit. Where the reference data
does not constrain anharmonicity strongly, `D` runs to its upper bound of 200 eV
and stops meaning a dissociation energy. The fit is still good -- that is the
better-fitting row above -- but a bond whose `D` is pinned at 200 will not
describe dissociation if you pull it apart in MD.

`quickstart/09_starting_point.py` measures all of it.
