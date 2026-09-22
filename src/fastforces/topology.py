"""Connectivity perception, symmetry typing, and force field term enumeration.

This module decides which terms exist and which of them share parameters.  It is
pure data in, pure data out -- no reference calculator, no fitting.

Term enumeration follows `tests/acetonitrile.jsonl` slot for slot; that file is
the specification for what a "complete" set of terms means here, and the counts
it implies for acetonitrile (5 bonds, 7 angles, 7 bondbond, 14 bondangle,
15 angleangle, 12 proper dihedrals) are asserted in the tests.
"""

from dataclasses import dataclass, field
from itertools import combinations

import networkx as nx
import numpy as np
from ase import Atoms

from .forcefield.exclusions import EXCLUSION_DEPTH

# The periodic series carried by every dihedral term.  Order and phase follow
# the example file: descending `n`, with `phi0 = pi` for even `n` and `0` for
# odd.
DIHEDRAL_ORDERS: tuple[int, ...] = (4, 3, 2, 1)

# `dihedralangleangle` is the one dihedral cross term the example gives a single
# order rather than the full series.
DIHEDRAL_ANGLEANGLE_ORDERS: tuple[int, ...] = (1,)

# Atom-slot permutations that leave each term's energy unchanged.  Used to pick
# one canonical slot order per term, so that two terms in the same class always
# have their per-slot parameters (`r1_0` versus `r2_0`, and so on) defined the
# same way round.  Parameters are read off the geometry *after* canonicalization,
# so no parallel permutation of parameter names is needed.
TERM_SYMMETRY: dict[str, tuple[tuple[int, ...], ...]] = {
    "bond": ((0, 1), (1, 0)),
    "angle": ((0, 1, 2), (2, 1, 0)),
    "bondbond": (
        (0, 1, 2, 3),
        (1, 0, 2, 3),
        (0, 1, 3, 2),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (3, 2, 0, 1),
        (2, 3, 1, 0),
        (3, 2, 1, 0),
    ),
    "bondangle": (
        (0, 1, 2, 3, 4),
        (2, 1, 0, 3, 4),
        (0, 1, 2, 4, 3),
        (2, 1, 0, 4, 3),
    ),
    "angleangle": (
        (0, 1, 2, 3, 4, 5),
        (2, 1, 0, 3, 4, 5),
        (0, 1, 2, 5, 4, 3),
        (2, 1, 0, 5, 4, 3),
        (3, 4, 5, 0, 1, 2),
        (5, 4, 3, 0, 1, 2),
        (3, 4, 5, 2, 1, 0),
        (5, 4, 3, 2, 1, 0),
    ),
    "periodicdihedral": ((0, 1, 2, 3), (3, 2, 1, 0)),
    "dihedralbond": (
        (0, 1, 2, 3, 4, 5),
        (3, 2, 1, 0, 4, 5),
        (0, 1, 2, 3, 5, 4),
        (3, 2, 1, 0, 5, 4),
    ),
    "dihedralangle": (
        (0, 1, 2, 3, 4, 5, 6),
        (3, 2, 1, 0, 4, 5, 6),
        (0, 1, 2, 3, 6, 5, 4),
        (3, 2, 1, 0, 6, 5, 4),
    ),
    "dihedralangleangle": ((0, 1, 2, 3), (3, 2, 1, 0)),
}


@dataclass
class Topology:
    """Every term of the force field, with its parameter-sharing classes.

    `atoms[t]` is an `(n_terms, n_slots)` integer array of global atom indices,
    in the canonical slot order defined by `TERM_SYMMETRY`.  `classes[t]` maps
    each row to a class index; rows sharing a class share fitted parameters.
    `fixed[t]` holds the per-row parameters that are neither fit nor derived
    from the geometry -- `n` and `phi0` for the dihedral series.
    """

    numbers: np.ndarray
    graph: nx.Graph
    atom_classes: np.ndarray
    atoms: dict[str, np.ndarray] = field(default_factory=dict)
    classes: dict[str, np.ndarray] = field(default_factory=dict)
    fixed: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    class_keys: dict[str, list] = field(default_factory=dict)
    exclusions: np.ndarray | None = None

    def n_classes(self, term_type: str) -> int:
        return len(self.class_keys.get(term_type, ()))

    def summary(self) -> str:
        lines = [
            f"{len(self.numbers)} atoms, {self.atom_classes.max() + 1} atom classes"
        ]
        for t in self.atoms:
            lines.append(
                f"  {t:20s} {len(self.atoms[t]):4d} terms  {self.n_classes(t):3d} classes"
            )
        return "\n".join(lines)


def perceive(atoms: Atoms) -> nx.Graph:
    """Molecular graph for `atoms`.

    Prefers `atoms.info["connectivity"]`, which is what `build`, the training
    files `io.write_training_set` writes and `calculators.pyscf.PySCFCalculator`
    all carry -- a perceived bond order beats a distance cutoff.  Falls back to
    `molify.ase2networkx`.
    """
    connectivity = atoms.info.get("connectivity")
    if connectivity is not None and len(connectivity):
        graph = nx.Graph()
        graph.add_nodes_from(range(len(atoms)))
        for entry in connectivity:
            i, j = int(entry[0]), int(entry[1])
            order = entry[2] if len(entry) > 2 else None
            graph.add_edge(i, j, order=order)
        return graph

    from molify import ase2networkx

    graph = ase2networkx(atoms, pbc=bool(np.any(atoms.pbc)))
    return nx.Graph(graph)


def equivalence_classes(
    graph: nx.Graph, numbers: np.ndarray, charges: np.ndarray | None = None
) -> np.ndarray:
    """Per-atom symmetry classes by Morgan-style refinement.

    Atoms start out labelled by element and formal charge, and each round
    relabels an atom by its own label plus the sorted multiset of its
    neighbours'.  Iterating to a fixed point separates atoms that any local
    graph environment can tell apart, and leaves genuinely equivalent atoms --
    the two oxygens of H2O2, the three hydrogens of a methyl -- sharing a label.

    This is the graph automorphism approximation every class-2 force field uses.
    It is exact for the topologies this package targets and cheap; it can in
    principle merge two atoms that only a global symmetry distinguishes, which
    would tie parameters that ought to be free.  That errs toward fewer
    parameters, which is the safe direction for a least-squares fit.
    """
    numbers = np.asarray(numbers, dtype=int)
    if charges is None:
        charges = np.zeros(len(numbers), dtype=int)
    labels = [
        (int(z), int(round(float(q)))) for z, q in zip(numbers, charges, strict=True)
    ]

    n_prev = -1
    while True:
        keys = {lab: i for i, lab in enumerate(sorted(set(labels)))}
        compact = [keys[lab] for lab in labels]
        if len(keys) == n_prev:
            return np.array(compact, dtype=int)
        n_prev = len(keys)
        labels = [
            (compact[i], tuple(sorted(compact[j] for j in graph.neighbors(i))))
            for i in range(len(numbers))
        ]


def _canonicalize(term_type: str, indices: tuple[int, ...], atom_classes: np.ndarray):
    """Pick the canonical slot order for one term, and return `(atoms, key)`.

    Ordered by class tuple first so that equivalent terms land in the same
    class, then by atom index so the choice is deterministic within a class.
    """
    best = None
    for perm in TERM_SYMMETRY[term_type]:
        permuted = tuple(indices[p] for p in perm)
        key = tuple(int(atom_classes[a]) for a in permuted)
        candidate = (key, permuted)
        if best is None or candidate < best:
            best = candidate
    key, permuted = best
    return permuted, key


def _angles(graph: nx.Graph) -> list[tuple[int, int, int]]:
    """`(i, j, k)` with `j` the vertex -- the slot order `compute_angle` wants."""
    out = []
    for j in sorted(graph.nodes):
        for i, k in combinations(sorted(graph.neighbors(j)), 2):
            out.append((i, j, k))
    return out


def _dihedrals(graph: nx.Graph) -> list[tuple[int, int, int, int]]:
    """Proper dihedrals `(a, b, c, d)` about each bond `b-c`."""
    out = []
    for b, c in sorted(tuple(sorted(e)) for e in graph.edges):
        for a in sorted(graph.neighbors(b)):
            if a == c:
                continue
            for d in sorted(graph.neighbors(c)):
                if d == b or d == a:  # d == a would be a 3-ring, not a dihedral
                    continue
                out.append((a, b, c, d))
    return out


def _pack(
    term_type: str,
    raw: list[tuple[int, ...]],
    atom_classes: np.ndarray,
    orders: tuple[int, ...] | None = None,
):
    """Canonicalize, deduplicate, assign classes, and expand the `n` series."""
    seen: dict[tuple[int, ...], tuple] = {}
    for indices in raw:
        permuted, key = _canonicalize(term_type, indices, atom_classes)
        seen.setdefault(permuted, (permuted, key))

    rows, keys = [], []
    for permuted, key in seen.values():
        for n in orders or (None,):
            rows.append(permuted)
            keys.append(key if n is None else (key, n))

    order = sorted(range(len(rows)), key=lambda i: (keys[i], rows[i]))
    rows = [rows[i] for i in order]
    keys = [keys[i] for i in order]

    unique = sorted(set(keys))
    index_of = {k: i for i, k in enumerate(unique)}
    atoms = np.array(rows, dtype=int).reshape(len(rows), -1)
    classes = np.array([index_of[k] for k in keys], dtype=int)

    fixed: dict[str, np.ndarray] = {}
    if orders is not None:
        n_arr = np.array([k[1] for k in keys], dtype=float)
        fixed["n"] = n_arr
        fixed["phi0"] = np.where(n_arr % 2 == 0, np.pi, 0.0)
    return atoms, classes, unique, fixed


def exclusion_mask(
    graph: nx.Graph, n_atoms: int, depth: int = EXCLUSION_DEPTH
) -> np.ndarray:
    """Boolean mask, True for pairs within `depth` bonds (1-2, 1-3, 1-4).

    This is the mask `forcefield/exclusions.py` takes off all three whole-system
    pair sums, so the depth is that module's and is imported rather than
    restated -- a topology that excluded to a different depth than the
    evaluators do would leave the difference in the energy with nothing to
    cancel it.
    """
    mask = np.zeros((n_atoms, n_atoms), dtype=bool)
    for i, reachable in nx.all_pairs_shortest_path_length(graph, cutoff=depth):
        for j in reachable:
            if i != j:
                mask[i, j] = True
    return mask


def enumerate_terms(atoms: Atoms, graph: nx.Graph | None = None) -> Topology:
    """Every term of the force field for `atoms`, with parameter-sharing classes."""
    graph = perceive(atoms) if graph is None else graph
    numbers = atoms.get_atomic_numbers()
    atom_classes = equivalence_classes(graph, numbers, atoms.get_initial_charges())

    bonds = [tuple(sorted(e)) for e in graph.edges]
    angles = _angles(graph)
    dihedrals = _dihedrals(graph)

    raw: dict[str, list[tuple[int, ...]]] = {}
    raw["bond"] = list(bonds)
    raw["angle"] = [tuple(a) for a in angles]
    # A bondbond term is a pair of bonds sharing an atom -- the same geometry an
    # angle spans, which is why H2O2 has as many of these as it has angles.
    raw["bondbond"] = [(i, j, j, k) for i, j, k in angles]
    # Each angle crossed with each of its own two legs.
    raw["bondangle"] = [
        leg for i, j, k in angles for leg in ((i, j, k, i, j), (i, j, k, j, k))
    ]
    # Angle pairs sharing a leg.  The example's single term crosses two angles
    # with *different* vertices that share the central O-O bond, so the usual
    # "same vertex" rule of COMPASS is too narrow to reproduce it; sharing a leg
    # covers both that case and the same-vertex one.
    raw["angleangle"] = [
        a + b
        for a, b in combinations(angles, 2)
        if {frozenset(a[:2]), frozenset(a[1:])} & {frozenset(b[:2]), frozenset(b[1:])}
    ]
    raw["periodicdihedral"] = list(dihedrals)
    # Each dihedral crossed with each of its three bonds, then with each of its
    # two angles.
    raw["dihedralbond"] = [
        d + bond
        for d in dihedrals
        for bond in ((d[0], d[1]), (d[1], d[2]), (d[2], d[3]))
    ]
    raw["dihedralangle"] = [d + ang for d in dihedrals for ang in (d[:3], d[1:])]
    raw["dihedralangleangle"] = list(dihedrals)

    series = {
        "periodicdihedral": DIHEDRAL_ORDERS,
        "dihedralbond": DIHEDRAL_ORDERS,
        "dihedralangle": DIHEDRAL_ORDERS,
        "dihedralangleangle": DIHEDRAL_ANGLEANGLE_ORDERS,
    }

    topology = Topology(
        numbers=numbers,
        graph=graph,
        atom_classes=atom_classes,
        exclusions=exclusion_mask(graph, len(atoms)),
    )
    for term_type, entries in raw.items():
        if not entries:
            continue
        packed, classes, keys, fixed = _pack(
            term_type, entries, atom_classes, series.get(term_type)
        )
        topology.atoms[term_type] = packed
        topology.classes[term_type] = classes
        topology.class_keys[term_type] = keys
        topology.fixed[term_type] = fixed
    return topology
