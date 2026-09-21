"""02 -- Topology perception, symmetry typing, and term enumeration.

No reference calculator and no fitting: this is the bookkeeping layer that
decides *which* terms exist and *which of them share parameters*.  Atoms in the
same symmetry class share a type, so both O-H bonds of hydrogen peroxide and
all three methyl hydrogens of acetonitrile are fit as one parameter each.

    uv run python quickstart/02_topology.py
"""

import fastforces as ff

from _common import banner

banner(__doc__)

for smiles in ("OO", "CC#N", "c1ccccc1C"):
    atoms = ff.build(smiles)
    graph = ff.perceive(atoms)
    topology = ff.enumerate_terms(atoms, graph)

    print(f"{smiles}  ({atoms.get_chemical_formula()})")
    print("  " + topology.summary().replace("\n", "\n  "))

    # Atom classes are found by Morgan-style refinement of the bond graph, so
    # they fall out of connectivity alone -- no SMARTS patterns, no atom-type
    # dictionary to maintain.
    by_class: dict[int, list[str]] = {}
    for index, cls in enumerate(topology.atom_classes):
        by_class.setdefault(int(cls), []).append(f"{atoms[index].symbol}{index}")
    print("  atom classes: " + " | ".join(",".join(v) for v in by_class.values()))

    n_excluded = int(topology.exclusions.sum()) // 2
    print(f"  nonbonded exclusions: {n_excluded} pairs (1-2, 1-3, 1-4)\n")

print("""Sharing parameters across a class is what keeps the fit determined: a
molecule provides far more reference forces than it does independent
parameters only because equivalent terms collapse onto one another.""")
