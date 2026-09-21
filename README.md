# fast-forces

Fast automatic parameterization of force fields

## Features

- Uses any ASE calculator: Fit to MLIPs, QM codes, TBLite, etc! any calculator that provides energies/forces
- Automatic conformer generation using `openconf`: You only need a SMILES string and a reference method to get a fully parameterized force field
- Export parameters in OpenMM or DynamicTopology format
- Use the included FastForces calculator to immediately start running simulations
- All training data is saved into one extended XYZ file: Everything necessary to reproduce the fit is contained in one file
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

## Examples

Numbered, runnable examples live in [`examples/`](examples/) -- start with
`examples/01_quickstart.py` and read `examples/README.md` for the index.
`examples/10_reaction.py` walks through the reaction pipeline end to end.

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
point. `examples/09_starting_point.py` works through the distinction.

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

`examples/09_starting_point.py` measures all of it.