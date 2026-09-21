"""10 -- Reactions: two diabatic states and the coupling between them.

`parameterize_reaction` takes an atom-mapped reaction SMILES and does three
things: fits a force field for every molecule involved, finds the saddle
between the two sides with Sella, and fits the EVB off-diagonal coupling that
joins them.  The result is an ASE calculator whose surface goes over the
barrier instead of stopping at it.

Three parts: the identity SN2 end to end, what happens to a channel with no
gas-phase saddle, and a scan across the finished surface.

    uv run python examples/10_reaction.py
"""

import numpy as np
from ase.io import read

import fastforces as ff
from fastforces import coupling as coupling_module
from fastforces import reaction as reaction_module

from _common import OUTPUT, banner, gfn2

banner(__doc__)

# Every hydrogen is written out and mapped.  That is not decoration: an implicit
# hydrogen is an atom with no index, and the atom that moves in a transfer is
# usually one of them.
SN2 = (
    "[Cl-:1].[C:2]([H:3])([H:4])([H:5])[Cl:6]>>[Cl:1][C:2]([H:3])([H:4])([H:5]).[Cl-:6]"
)
WATER = (
    "[O+:1]([H:2])([H:3])[H:4].[O:5]([H:6])[H:7]"
    ">>[O:1]([H:3])[H:4].[O+:5]([H:2])([H:6])[H:7]"
)

# ---------------------------------------------------------------------------
print("1. The identity SN2, end to end")
print("-" * 72)

reaction = reaction_module.parse(SN2)
kind, spec = reaction.channel()
print(f"channel      {kind}, spanning atoms {spec}")
print(f"broken       {sorted(map(sorted, reaction.broken))}")
print(f"formed       {sorted(map(sorted, reaction.formed))}")
print(f"fragments    {[f.smiles for f in reaction.reactant_fragments]}")

# The guess is built from the changing bonds rather than interpolated: each
# fragment is embedded on its own, the breaking bond is opened out to the
# contact distance and the leaving group is docked along it.  For a transfer
# that is a symmetric saddle guess by construction.
start = reaction_module.guess(reaction)
print(
    f"\nguess        Cl-C {start.get_distance(0, 1):.3f} A, "
    f"C-Cl {start.get_distance(1, 5):.3f} A"
)

rxn = ff.parameterize_reaction(
    SN2,
    gfn2,
    config=ff.FitConfig(n_mode_frames=20, n_conformers=0),
    workdir=str(OUTPUT / "sn2"),
)
saddle = rxn.frames[1]
print(
    f"Sella        Cl-C {saddle.get_distance(0, 1):.3f} A, "
    f"C-Cl {saddle.get_distance(1, 5):.3f} A   (D3h, as it should be)"
)
print(f"\n{rxn.report()}\n")

rxn.write(str(OUTPUT / "sn2-chloride"))
print(f"wrote {OUTPUT / 'sn2-chloride'}.xyz, .jsonl and one jsonl per state")

# The transition state is reproduced exactly, because that is the equation the
# amplitude inverted.  The endpoints are not fitted and are the honest test:
# the coupling has quenched to `eps` there, so what is left is the two diabatic
# force fields alone -- and their intermolecular part is `elements`' unfitted
# nonbonded table, which is where the residual error sits.
energies = rxn.energies()
print(
    "\nendpoint error is the unfitted intermolecular term, not the coupling: "
    f"{energies['evb'][0] - energies['reference'][0]:+.3f} eV"
)

# ---------------------------------------------------------------------------
print("\n\n2. A channel with no gas-phase saddle")
print("-" * 72)
print(
    "The water Grotthuss transfer has no barrier at contact: the shared-proton\n"
    "Zundel geometry is a minimum, not a saddle. Sella still lands on it -- it is\n"
    "where the reaction goes -- but relaxing along its (3.6 meV, i.e. numerical)\n"
    "imaginary mode falls back into the same well from both sides, so the two\n"
    "topologies the SMILES names are never reached. The pipeline says so rather\n"
    "than fitting a coupling to a path that does not exist."
)

water = reaction_module.parse(WATER)
try:
    reaction_module.stationary_points(water, gfn2)
except reaction_module.ReactionError as error:
    print(f"\n  ReactionError: {str(error).splitlines()[0]}")

print(
    "\nThe frames can also be supplied: a constrained path, a literature geometry,\n"
    "or a higher-level calculation. `coupling.fit_threebody` then needs only the\n"
    "three geometries and an amplitude."
)

frames = read(str(OUTPUT / "h3o-h2o-transfer.xyz"), index=":", format="extxyz")
stored = coupling_module.Coupling.from_jsonl(str(OUTPUT / "h3o-h2o-transfer.jsonl"))
amplitude = float(stored.terms["threebody"]["kwargs"]["A"][0])
refit = coupling_module.fit_threebody(frames, water.channel()[1], amplitude=amplitude)

print(f"\n  stored   {stored.to_rows()[0]['kwargs']}")
print(f"  refitted {refit.to_rows()[0]['kwargs']}")
print(f"  identical: {refit.to_rows()[0]['kwargs'] == stored.to_rows()[0]['kwargs']}")
print(
    f"\n  V(reactant)   {refit.value(frames[0].get_positions()):+.6f} eV"
    f"   <- quenched to eps by construction"
)
print(
    f"  V(saddle)     {refit.value(frames[1].get_positions()):+.6f} eV   <- exactly A"
)
print(f"  V(product)    {refit.value(frames[2].get_positions()):+.6f} eV")

# ---------------------------------------------------------------------------
print("\n\n3. Across the finished surface")
print("-" * 72)
print(
    "Sliding the methyl group along the Cl-C-Cl axis. `statevec` is how much of\n"
    "each diabatic state the geometry is made of: it hands over where the two\n"
    "diabats cross, which is what the coupling is there to smooth.\n"
)

calculator = rxn.calculator()
axis = saddle.get_positions()
direction = (axis[5] - axis[0]) / np.linalg.norm(axis[5] - axis[0])
methyl = [1, 2, 3, 4]

print(
    f"{'shift/A':>8s} {'E_evb/eV':>11s} {'H_react':>11s} {'H_prod':>11s} {'state':>16s}"
)
for shift in np.linspace(-0.6, 0.6, 13):
    probe = saddle.copy()
    positions = probe.get_positions()
    positions[methyl] += direction * shift
    probe.set_positions(positions)
    probe.calc = rxn.calculator(probe)
    energy = probe.get_potential_energy()
    h1, h2 = ff.evb.diabatic_energies(probe, rxn.state_list)
    weights = probe.calc.results["statevec"]
    print(
        f"{shift:8.2f} {energy:11.4f} {h1:11.4f} {h2:11.4f} "
        f"  {weights[0]:.2f} / {weights[1]:.2f}"
    )

print(
    "\nThe surface is smooth across the handover and never follows either diabat\n"
    "up its own dissociation wall, which is the whole point of the off-diagonal."
)
