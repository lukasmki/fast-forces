"""Topology perception, symmetry typing, and term enumeration."""

import json
from pathlib import Path

import numpy as np

import fastforces as ff
from fastforces.topology import enumerate_terms, equivalence_classes, perceive

EXAMPLE_SMILES = "CC#N"
EXAMPLE_JSONL = Path(__file__).parent / "acetonitrile.jsonl"

# `acetonitrile.jsonl` is the specification for what a complete set of terms
# means, so the counts come out of the file itself rather than being copied
# alongside it.  `atom`, `lennardjones` and `reference` are per-atom bookkeeping
# rather than enumerated terms, so they are not in `Topology.atoms`.
NOT_ENUMERATED = {"atom", "lennardjones", "reference"}


def example_term_counts():
    counts: dict[str, int] = {}
    for line in EXAMPLE_JSONL.read_text().splitlines():
        if not line.strip():
            continue
        term = json.loads(line)["type"]
        if term not in NOT_ENUMERATED:
            counts[term] = counts.get(term, 0) + 1
    return counts


def acetonitrile():
    """The molecule the example files describe: H3C-C#N, atoms 0-5.

    `ff.build` is deterministic, so the atom ordering here is the ordering the
    indices in `acetonitrile.jsonl` refer to.
    """
    return ff.build(EXAMPLE_SMILES)


def test_term_counts_match_the_example():
    top = enumerate_terms(acetonitrile())
    assert {t: len(a) for t, a in top.atoms.items()} == example_term_counts()


def test_symmetry_classes():
    """The three methyl hydrogens are equivalent; the two carbons are not."""
    atoms = acetonitrile()
    classes = equivalence_classes(perceive(atoms), atoms.get_atomic_numbers())
    assert classes[3] == classes[4] == classes[5]  # the methyl hydrogens
    assert classes[0] != classes[1]  # methyl carbon vs nitrile carbon
    assert classes[0] != classes[3]


def test_equivalent_bonds_share_a_class():
    top = enumerate_terms(acetonitrile())
    bonds = {
        tuple(sorted(row)): int(c)
        for row, c in zip(top.atoms["bond"], top.classes["bond"])
    }
    assert bonds[(0, 3)] == bonds[(0, 4)] == bonds[(0, 5)]  # the three C-H bonds
    assert bonds[(0, 1)] != bonds[(0, 3)]  # C-C is its own class
    assert bonds[(1, 2)] != bonds[(0, 1)]  # and so is C#N


def test_methyl_hydrogens_are_equivalent():
    top = enumerate_terms(acetonitrile())
    hydrogens = [i for i, z in enumerate(top.numbers) if z == 1]
    assert len({int(top.atom_classes[i]) for i in hydrogens}) == 1


def test_exclusions_cover_1_2_1_3_and_1_4():
    atoms = acetonitrile()
    top = enumerate_terms(atoms)
    # acetonitrile is small enough that every pair is within three bonds -- the
    # longest path, H-C-C#N, is exactly a 1-4
    from DynamicTopology.forcefield.exclusions import exclusion_terms

    excluded = {
        tuple(sorted(t["atoms"].values()))
        for t in exclusion_terms([], atoms.numbers, graph=top.graph)
        if t["type"] == "zblexclusion"
    }
    n = len(atoms)
    assert excluded == {(i, j) for i in range(n) for j in range(i + 1, n)}


def test_perceive_prefers_stored_connectivity():
    atoms = acetonitrile()
    graph = perceive(atoms)
    assert sorted(tuple(sorted(e)) for e in graph.edges) == [
        (0, 1),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 2),
    ]


def test_canonical_slot_order_is_deterministic():
    """Two runs give byte-identical topologies, so classes are reproducible."""
    a, b = enumerate_terms(acetonitrile()), enumerate_terms(acetonitrile())
    for term in a.atoms:
        assert np.array_equal(a.atoms[term], b.atoms[term])
        assert np.array_equal(a.classes[term], b.classes[term])
