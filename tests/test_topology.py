"""Topology perception, symmetry typing, and term enumeration."""

import numpy as np
from ase.io import read

from fastforces.topology import enumerate_terms, equivalence_classes, perceive

# Counts implied by `examples/h2o2_dynamictopology_format.jsonl`, which is the
# specification for what a complete set of terms means.
H2O2_TERM_COUNTS = {
    "bond": 3,
    "angle": 2,
    "bondbond": 2,
    "bondangle": 4,
    "angleangle": 1,
    "dihedralangle": 8,
    "dihedralbond": 12,
    "dihedralangleangle": 1,
    "periodicdihedral": 4,
}


def h2o2():
    return read("examples/h2o2.xyz")


def test_term_counts_match_the_example():
    top = enumerate_terms(h2o2())
    assert {t: len(a) for t, a in top.atoms.items()} == H2O2_TERM_COUNTS


def test_symmetry_classes():
    """The two oxygens are equivalent, and so are the two hydrogens."""
    atoms = h2o2()
    classes = equivalence_classes(perceive(atoms), atoms.get_atomic_numbers())
    assert classes[0] == classes[1]
    assert classes[2] == classes[3]
    assert classes[0] != classes[2]


def test_equivalent_bonds_share_a_class():
    top = enumerate_terms(h2o2())
    bonds = {
        tuple(sorted(row)): int(c)
        for row, c in zip(top.atoms["bond"], top.classes["bond"])
    }
    assert bonds[(0, 2)] == bonds[(1, 3)]  # the two O-H bonds
    assert bonds[(0, 1)] != bonds[(0, 2)]  # O-O is its own class


def test_methyl_hydrogens_are_equivalent():
    import fastforces as ff

    top = enumerate_terms(ff.build("CC#N"))
    hydrogens = [i for i, z in enumerate(top.numbers) if z == 1]
    assert len({int(top.atom_classes[i]) for i in hydrogens}) == 1


def test_exclusions_cover_1_2_1_3_and_1_4():
    top = enumerate_terms(h2o2())
    # H2O2 is small enough that every pair is within three bonds
    off_diagonal = ~np.eye(4, dtype=bool)
    assert np.all(top.exclusions[off_diagonal])


def test_perceive_prefers_stored_connectivity():
    atoms = h2o2()
    graph = perceive(atoms)
    assert sorted(tuple(sorted(e)) for e in graph.edges) == [(0, 1), (0, 2), (1, 3)]


def test_canonical_slot_order_is_deterministic():
    """Two runs give byte-identical topologies, so classes are reproducible."""
    a, b = enumerate_terms(h2o2()), enumerate_terms(h2o2())
    for term in a.atoms:
        assert np.array_equal(a.atoms[term], b.atoms[term])
        assert np.array_equal(a.classes[term], b.classes[term])
