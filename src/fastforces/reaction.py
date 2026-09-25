"""From an atom-mapped reaction SMILES to the pieces an EVB surface needs.

Three things have to come out of a reaction SMILES before a coupling can be
fitted, and they are independent of each other:

  the states      one force field per side of the arrow, both defined over the
                  *same* atom indices.  A diabatic state is a bonding topology,
                  so the two differ only in which bonded terms they carry, and
                  `state_parameters` builds each by fitting every fragment once
                  and scattering the fitted terms onto the combined indices.

  the geometry    reactant, transition state and product, in that one index
                  order.  `stationary_points` finds the saddle first and walks
                  down from it, rather than building the endpoints and guessing
                  a saddle between them -- see its docstring.

  the channel     which of the three coupling forms the reaction gets, decided
                  by its own connectivity change and nothing else.  `channel`
                  answers that; `coupling.fit` acts on the answer.

**Every atom must be mapped, on both sides, including hydrogens.** A transfer is
a statement about one atom moving between two others, and an unmapped hydrogen
is exactly the atom that usually moves.  Written out, the water Grotthuss
channel is

    [O+:1]([H:2])([H:3])[H:4].[O:5]([H:6])[H:7]
        >> [O:1]([H:3])[H:4].[O+:5]([H:2])([H:6])[H:7]

and the map numbers, sorted, are the combined index order: atom `i` is the one
carrying map number `i + 1` if the maps are `1..n`, and in general the `i`-th
smallest map number.  Both sides are read in that order, so `reactant_bonds` and
`product_bonds` are directly comparable and their difference is the reaction.
`map_atoms` writes that form from a plain `[OH3+].O>>O.[OH3+]`, by two rules its
docstring states.
"""

import numpy as np
from ase import Atoms
from ase.data import covalent_radii

from . import sampling
from .params import Parameters


def build(smiles: str, seed: int = 42) -> Atoms:
    """`fastforces.build`, imported lazily because the package imports this."""
    from . import build as _build

    return _build(smiles, seed=seed)


# Contact distance for a bond that is being made or broken, as a multiple of the
# sum of the two covalent radii.  The guess geometry puts *every* changing bond
# here, so a transfer comes out with its two partial bonds equal and its heavy
# atoms at twice this -- for the water dimer, O-H 1.24 A and O-O 2.48 A against
# a wB97X-V saddle at 1.225 and 2.45.  It only has to land inside the saddle's
# basin for `Sella`; it is this close because a symmetric guess costs nothing.
CHANGING_BOND_SCALE: float = 1.30

# Displacement (A, largest single-atom) along the imaginary mode that separates
# the two endpoint relaxations.  Large enough to commit to a side -- at the
# saddle the gradient is zero, so a plain relaxation started there goes nowhere
# -- and small enough to stay on the reaction path rather than skipping into a
# neighbouring basin.
MODE_STEP: float = 0.25

# Mode energy (eV) below which a negative curvature is taken for numerical noise
# rather than for a reaction coordinate.  Much lower than
# `sampling.TRIVIAL_MODE_ENERGY`, which screens *real* soft modes out of thermal
# sampling: a saddle can legitimately be this flat -- the shared-proton geometry
# of a water dimer comes out at 3.6 meV imaginary and is, correctly, not a
# saddle at all -- so what actually decides whether a geometry is this
# reaction's transition state is where relaxing along the mode ends up, which
# `endpoints` checks against the two topologies afterwards.
IMAGINARY_MODE_ENERGY: float = 0.002


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


class ReactionError(ValueError):
    """Raised when a reaction SMILES does not describe a usable channel."""


def _side(smiles: str, label: str):
    """One side of the arrow as an RDKit mol, renumbered into map order.

    `removeHs=False` is the whole reason this is not `MolFromSmiles(smiles)`:
    the default parser folds `[H:2]` into its neighbour's implicit hydrogen
    count and the map number goes with it, which silently turns a proton
    transfer into a reaction with no mapped atoms at all.
    """
    from rdkit import Chem

    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ReactionError(f"RDKit could not parse the {label} side: {smiles!r}")

    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    if 0 in maps:
        unmapped = [
            f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}"
            for i, m in enumerate(maps)
            if m == 0
        ]
        raise ReactionError(
            f"every atom on the {label} side must carry an atom map number; "
            f"{', '.join(unmapped)} do not. Hydrogens have to be written out "
            "explicitly and mapped -- in a transfer the moving atom is usually "
            "one of them."
        )
    if len(set(maps)) != len(maps):
        raise ReactionError(f"the {label} side repeats an atom map number")

    # An implicit hydrogen is an atom that exists, has no index and cannot be
    # mapped, so a SMILES carrying one describes more atoms than it names.  The
    # check above cannot see them -- every atom RDKit *lists* is mapped -- which
    # is why `[OH3+:1].[O:2]` gets this far looking well formed while standing
    # for five atoms it never mentions.
    implicit = [
        f"{a.GetSymbol()}:{a.GetAtomMapNum()}"
        for a in mol.GetAtoms()
        if a.GetTotalNumHs() > 0
    ]
    if implicit:
        raise ReactionError(
            f"the {label} side leaves hydrogens implicit on {', '.join(implicit)}. "
            "Write every hydrogen out and map it -- `[O+:1]([H:2])([H:3])[H:4]` "
            "rather than `[OH3+:1]` -- because an unnamed atom cannot be given "
            "an index, and in a transfer the moving atom is usually one of them."
        )

    order = [i for i, _ in sorted(enumerate(maps), key=lambda p: p[1])]
    return Chem.RenumberAtoms(mol, order), sorted(maps)


def _bonds(mol) -> frozenset:
    return frozenset(
        frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        for bond in mol.GetBonds()
    )


class Fragment:
    """One connected molecule of one side, in combined indices.

    `indices` is in fragment-template order: the force field fitted for this
    fragment is defined over `0..len(indices)-1`, and `indices[k]` is where its
    `k`-th atom lives in the combined system.  `key` is what two occurrences of
    the same molecule share, so the fit happens once -- the water in
    `H3O+ + H2O` is the same water on both sides of the arrow.
    """

    __slots__ = ("indices", "smiles", "charge", "mol")

    def __init__(self, indices, smiles: str, charge: int, mol=None):
        self.indices = tuple(int(i) for i in indices)
        self.smiles = smiles
        self.charge = int(charge)
        # The fragment as RDKit sees it, in the same canonical order as
        # `indices`.  Kept so `_mapping` can match it onto a freshly built
        # conformer by graph isomorphism rather than by re-deriving a canonical
        # order from the SMILES and trusting the two to agree.
        self.mol = mol

    @property
    def key(self) -> tuple[str, int]:
        return (self.smiles, self.charge)

    def __repr__(self) -> str:
        return f"Fragment({self.smiles!r}, charge={self.charge}, {self.indices})"


class Reaction:
    """A parsed reaction SMILES: two bonding topologies over one atom list."""

    def __init__(
        self,
        smiles: str,
        numbers: np.ndarray,
        charges: np.ndarray,
        reactant_bonds: frozenset,
        product_bonds: frozenset,
        reactant_fragments: list,
        product_fragments: list,
        spin: int = 0,
    ):
        self.smiles = smiles
        self.numbers = np.asarray(numbers, dtype=int)
        self.charges = np.asarray(charges, dtype=float)
        self.reactant_bonds = reactant_bonds
        self.product_bonds = product_bonds
        self.reactant_fragments = reactant_fragments
        self.product_fragments = product_fragments
        self.spin = int(spin)

    # -- access ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.numbers)

    @property
    def charge(self) -> int:
        return int(round(float(self.charges.sum())))

    def bonds(self, side: str) -> frozenset:
        return self.reactant_bonds if side == "reactant" else self.product_bonds

    def fragments(self, side: str) -> list:
        return self.reactant_fragments if side == "reactant" else self.product_fragments

    @property
    def broken(self) -> frozenset:
        return self.reactant_bonds - self.product_bonds

    @property
    def formed(self) -> frozenset:
        return self.product_bonds - self.reactant_bonds

    @property
    def changing(self) -> frozenset:
        return self.broken | self.formed

    def connectivity(self, side: str) -> list:
        """The `[i, j, order]` list an extxyz frame carries, for one side."""
        return [[int(i), int(j), 1.0] for i, j in sorted(map(sorted, self.bonds(side)))]

    # -- classification -------------------------------------------------

    def channel(self) -> tuple[str, tuple | None]:
        """Which coupling form this reaction gets, and the atoms it spans.

        `("transfer", (donor, moving, acceptor))` for one bond broken and one
        formed sharing an atom; `("fission", ((i, j), moving_fragment))` for a
        single bond broken or formed with nothing else changing; `("rmsd", None)`
        for everything else.

        The routing is the reaction's own connectivity change and nothing else,
        because that is what decides whether the channel *has* the data each
        form needs -- see `coupling`'s module docstring, which is where the
        three forms are argued.
        """
        return classify(self.reactant_bonds, self.product_bonds, len(self))


def classify(
    reactant_bonds: frozenset, product_bonds: frozenset, n: int
) -> tuple[str, tuple | None]:
    """`Reaction.channel` for any two bond sets over the same `n` atoms.

    Bonds are `frozenset({i, j})` pairs.  Shared with `refit`, which classifies
    a dataset's reactions from their endpoint frames rather than from a SMILES.
    """
    broken = reactant_bonds - product_bonds
    formed = product_bonds - reactant_bonds

    if len(broken) == 1 and len(formed) == 1:
        gone, made = next(iter(broken)), next(iter(formed))
        shared = gone & made
        if len(shared) == 1:
            moving = next(iter(shared))
            return "transfer", (
                next(iter(gone - shared)),
                moving,
                next(iter(made - shared)),
            )

    if len(broken) == 1 and not formed:
        changed, separated = next(iter(broken)), product_bonds
    elif len(formed) == 1 and not broken:
        changed, separated = next(iter(formed)), reactant_bonds
    else:
        return "rmsd", None

    first, second = sorted(changed)
    moving = _component(separated, n, second)
    if first in moving:
        # Both ends stay in one piece, so cutting this bond separates nothing:
        # a ring opening, which keeps a saddle and takes the RMSD route rather
        # than the crossing-centred one.
        return "rmsd", None
    return "fission", ((first, second), tuple(sorted(moving)))


def _component(bonds: frozenset, n: int, start: int) -> set:
    """The connected component of `bonds` containing `start`."""
    neighbours: dict[int, set] = {i: set() for i in range(n)}
    for bond in bonds:
        i, j = sorted(bond)
        neighbours[i].add(j)
        neighbours[j].add(i)
    seen, stack = {start}, [start]
    while stack:
        for nxt in neighbours[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def map_atoms(smiles: str) -> str:
    """Atom-map a reaction SMILES written without map numbers.

    `parse` needs every atom mapped, hydrogens included, which is the honest
    input but not the natural one: `[OH3+].O>>O.[OH3+]` says what the reaction
    is as plainly as the fourteen-bracket form does.  The mapping is recovered
    by two rules, and they are the whole contract:

      * Every atom other than hydrogen keeps its place: the `k`-th oxygen on
        the left is the `k`-th oxygen on the right, in order of appearance.
        The order the product side is written in is therefore what names the
        reaction -- `[OH3+].O>>O.[OH3+]` is a transfer, `[OH3+].O>>[OH3+].O`
        is nothing at all.
      * Hydrogens, which a SMILES cannot tell apart, are assigned to break and
        form as few bonds as possible.  That is a linear assignment -- a
        hydrogen's only bonds are to atoms the first rule already placed -- so
        it is exact and cheap.  Among equally good assignments the
        lowest-numbered hydrogen is the one that moves, which is the convention
        the reference datasets use.

    The combined order is each heavy atom followed by its own hydrogens, in the
    order the reactant side is written, so the result reads like the mapped
    SMILES a person would write -- for the water channel, exactly the one in
    this module's docstring.

    A SMILES that is already fully mapped is returned unchanged.  One with
    hydrogen-hydrogen bonds on either side is refused: the second rule does not
    hold there, and such a reaction has to be mapped by hand.
    """
    from rdkit import Chem
    from scipy.optimize import linear_sum_assignment

    if smiles.count(">>") != 1:
        raise ReactionError(
            f"expected one '>>' in the reaction SMILES, got {smiles.count('>>')}"
        )
    sides = smiles.split(">>")

    raw = Chem.SmilesParserParams()
    raw.removeHs = False
    listed = [Chem.MolFromSmiles(side, raw) for side in sides]
    if any(mol is None for mol in listed):
        raise ReactionError(f"RDKit could not parse {smiles!r}")
    maps = [a.GetAtomMapNum() for mol in listed for a in mol.GetAtoms()]
    if all(maps):
        return smiles
    if any(maps):
        raise ReactionError(
            "the reaction SMILES maps some atoms and not others. Map every atom, "
            "hydrogens included, or none -- a partial mapping would have to be "
            "completed by a guess the caller already chose not to leave to us"
        )

    reactant, product = (Chem.AddHs(mol) for mol in listed)
    for mol in (reactant, product):
        for bond in mol.GetBonds():
            if bond.GetBeginAtom().GetAtomicNum() == bond.GetEndAtom().GetAtomicNum() == 1:
                raise ReactionError(
                    f"{smiles!r} has a hydrogen-hydrogen bond, which the automatic "
                    "mapping cannot place; write the reaction atom-mapped instead"
                )

    # combined order: each heavy atom, then its hydrogens; a bare hydrogen
    # (a proton, an H atom) where it is written
    order: list[int] = []
    for atom in reactant.GetAtoms():
        if atom.GetAtomicNum() != 1:
            order.append(atom.GetIdx())
            order += sorted(
                n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1
            )
        elif atom.GetDegree() == 0:
            order.append(atom.GetIdx())
    combined = {idx: c for c, idx in enumerate(order)}

    def by_element(mol) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for atom in mol.GetAtoms():
            out.setdefault(atom.GetAtomicNum(), []).append(atom.GetIdx())
        return out

    left, right = by_element(reactant), by_element(product)
    if {z: len(v) for z, v in left.items()} != {z: len(v) for z, v in right.items()}:
        raise ReactionError(
            f"the two sides of {smiles!r} do not contain the same atoms -- an EVB "
            "state pair is two bonding topologies over one set of atoms"
        )

    # heavy atoms: by element, in order of appearance
    placed: dict[int, int] = {}  # product atom index -> combined index
    for z, indices in left.items():
        if z != 1:
            for r, p in zip(indices, right[z], strict=True):
                placed[p] = combined[r]

    # hydrogens: the assignment that changes the fewest bonds
    r_h = sorted(left.get(1, []), key=combined.__getitem__)
    p_h = right.get(1, [])
    # A hydrogen's neighbours are all heavy atoms (H-H was refused above), so
    # both sets are in combined indices already.
    r_sets = [
        {combined[n.GetIdx()] for n in reactant.GetAtomWithIdx(h).GetNeighbors()}
        for h in r_h
    ]
    p_sets = [
        {placed[n.GetIdx()] for n in product.GetAtomWithIdx(h).GetNeighbors()}
        for h in p_h
    ]
    p_rank = np.argsort(np.argsort([min(s, default=-1) for s in p_sets], kind="stable"))
    cost = np.zeros((len(r_h), len(p_h)))
    for i, r_set in enumerate(r_sets):
        for j, p_set in enumerate(p_sets):
            changed = len(r_set ^ p_set)
            # Tie-breaks, each far below one bond: a moving hydrogen is the
            # lowest-numbered one available, and the rest keep their order.
            cost[i, j] = changed + 1e-3 * i * (changed > 0) + 1e-6 * abs(i - p_rank[j])
    rows, cols = linear_sum_assignment(cost)
    for i, j in zip(rows, cols, strict=True):
        placed[p_h[j]] = combined[r_h[i]]

    for atom in reactant.GetAtoms():
        atom.SetAtomMapNum(combined[atom.GetIdx()] + 1)
    for atom in product.GetAtoms():
        atom.SetAtomMapNum(placed[atom.GetIdx()] + 1)
    mapped = f"{Chem.MolToSmiles(reactant)}>>{Chem.MolToSmiles(product)}"

    parsed = parse(mapped)
    if not parsed.changing:
        raise ReactionError(
            f"{smiles!r} maps onto itself with no bond broken or formed. Heavy "
            "atoms are matched in order of appearance, so the product side has "
            "to be written with the moving groups in their new places -- "
            "`[OH3+].O>>O.[OH3+]`, not `[OH3+].O>>[OH3+].O` -- or atom-mapped"
        )
    return mapped


def parse(smiles: str) -> Reaction:
    """Parse an atom-mapped reaction SMILES into a `Reaction`.

    Both sides are renumbered into map order, so the two bond sets live in one
    index space and their difference is the reaction.  The element list and the
    formal charges are read from the reactant side and checked against the
    product side: a map number that changes element between the two sides is a
    mapping error, not a reaction.
    """

    if smiles.count(">>") != 1:
        raise ReactionError(
            f"expected one '>>' in the reaction SMILES, got {smiles.count('>>')}"
        )
    reactant_smiles, product_smiles = smiles.split(">>")
    reactant, reactant_maps = _side(reactant_smiles, "reactant")
    product, product_maps = _side(product_smiles, "product")

    if reactant_maps != product_maps:
        missing = set(reactant_maps) ^ set(product_maps)
        raise ReactionError(
            "the two sides carry different atom map numbers "
            f"(unpaired: {sorted(missing)}). Every atom has to appear on both "
            "sides -- an EVB state pair is two bonding topologies over one set "
            "of atoms, so nothing may be created or destroyed."
        )

    numbers = np.array([a.GetAtomicNum() for a in reactant.GetAtoms()], dtype=int)
    product_numbers = np.array(
        [a.GetAtomicNum() for a in product.GetAtoms()], dtype=int
    )
    if not np.array_equal(numbers, product_numbers):
        wrong = np.flatnonzero(numbers != product_numbers)
        raise ReactionError(
            f"atom map numbers {[reactant_maps[i] for i in wrong]} name different "
            "elements on the two sides"
        )

    charges = np.array(
        [float(a.GetFormalCharge()) for a in reactant.GetAtoms()], dtype=float
    )
    # `2S` of the complex: every radical on the reactant side, high-spin
    # coupled.  Electron-count parity -- what a calculator falls back to -- puts
    # `[H][H].[O]` on the singlet surface and `[O].[OH]` on the doublet, which
    # is neither the surface the channel runs on nor the one the fragments were
    # fitted on, so the fragment `E0`s and the barrier would not share a zero.
    spin = sum(a.GetNumRadicalElectrons() for a in reactant.GetAtoms())

    return Reaction(
        smiles=smiles,
        numbers=numbers,
        charges=charges,
        reactant_bonds=_bonds(reactant),
        product_bonds=_bonds(product),
        reactant_fragments=_fragments(reactant),
        product_fragments=_fragments(product),
        spin=spin,
    )


def _fragments(mol) -> list:
    """The connected molecules of one side, each in combined indices.

    Two things here are what let one fit serve several occurrences of the same
    molecule, and both are needed:

    The per-fragment SMILES is stripped of map numbers, so it is the molecule's
    identity rather than its role in this particular reaction -- the water in
    `H3O+ + H2O` is the same water on both sides of the arrow and gets one fit.

    `indices` is put in *canonical rank* order rather than in the order the
    atoms happen to appear in the parent.  That order is a property of the
    molecular graph alone, so two occurrences of one molecule agree on it atom
    for atom, and the fitted force field can be scattered onto either by
    position.  Without it the product-side hydronium of the water channel --
    which the parent lists as `(H, O, H, H)`, the transferred proton first --
    would silently receive a hydronium force field fitted in `(O, H, H, H)`
    order, putting the O-H bonds on H-H pairs.
    """
    from rdkit import Chem

    groups = Chem.GetMolFrags(mol)
    mols = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    out = []
    for indices, fragment in zip(groups, mols, strict=True):
        clean = Chem.Mol(fragment)
        for atom in clean.GetAtoms():
            atom.SetAtomMapNum(0)
        # `GetMolFrags` hands back unsanitized pieces, and canonical ranking
        # reads ring membership.  `FastFindRings` is the cheap half of
        # sanitization and the only half needed here.
        Chem.FastFindRings(clean)
        ranks = list(Chem.CanonicalRankAtoms(clean, breakTies=True))
        order = [i for i, _ in sorted(enumerate(ranks), key=lambda p: p[1])]
        charge = sum(a.GetFormalCharge() for a in fragment.GetAtoms())
        out.append(
            Fragment(
                [indices[i] for i in order],
                Chem.MolToSmiles(clean),
                charge,
                Chem.RenumberAtoms(clean, order),
            )
        )
    return out


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def _contact(numbers: np.ndarray, i: int, j: int) -> float:
    """Target length (A) for a bond that is being made or broken."""
    return CHANGING_BOND_SCALE * (
        covalent_radii[numbers[i]] + covalent_radii[numbers[j]]
    )


def _unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0]) if fallback is None else fallback
    return v / norm


def _open_direction(pos: np.ndarray, neighbours, i: int) -> np.ndarray:
    """The direction atom `i` has a free valence in, from its own geometry.

    Away from the centroid of everything already bonded to it -- the lone-pair
    side of a water oxygen, the outward side of a terminal hydrogen.  An atom
    with no neighbours (a bare ion) has no such direction and gets `+x`, which
    is as good as any other for a sphere.
    """
    others = [k for k in neighbours[i]]
    if not others:
        return np.array([1.0, 0.0, 0.0])
    return _unit(pos[i] - pos[others].mean(axis=0))


def _rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation taking unit vector `source` onto unit vector `target`."""
    axis = np.cross(source, target)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.dot(source, target))
    if sine < 1e-8:
        if cosine > 0.0:
            return np.eye(3)
        # Antiparallel: any axis perpendicular to `source` turns it around.
        perpendicular = np.cross(source, [1.0, 0.0, 0.0])
        if np.linalg.norm(perpendicular) < 1e-8:
            perpendicular = np.cross(source, [0.0, 1.0, 0.0])
        axis, sine, cosine = _unit(perpendicular), 0.0, -1.0
        k = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        return np.eye(3) + 2.0 * (k @ k)
    axis = axis / sine
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + sine * k + (1.0 - cosine) * (k @ k)


def _spin(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _unit(axis)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _embed(mol, seed: int) -> np.ndarray:
    """A 3D conformer for one fragment, in its own atom order."""
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

    work = Chem.Mol(mol)
    if work.GetNumAtoms() == 1:
        return np.zeros((1, 3))
    # `GetMolFrags` returns unsanitized pieces and distance geometry reads ring
    # membership; without this RDKit complains to stderr and carries on.
    Chem.FastFindRings(work)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    if rdDistGeom.EmbedMolecule(work, params) != 0:
        raise ReactionError(f"RDKit could not embed {Chem.MolToSmiles(mol)}")
    # Cleaning up the distance-geometry output is worth doing but not worth
    # failing over: the reference calculator relaxes these geometries anyway,
    # and MMFF has no parameters for plenty of the ions a reaction involves.
    for refine in (
        rdForceFieldHelpers.MMFFOptimizeMolecule,
        rdForceFieldHelpers.UFFOptimizeMolecule,
    ):
        try:
            refine(work)
            break
        except Exception:
            continue
    return np.array(work.GetConformer().GetPositions())


def _neighbour_table(bonds, n: int) -> dict:
    table: dict[int, list] = {i: [] for i in range(n)}
    for bond in bonds:
        i, j = sorted(bond)
        table[i].append(j)
        table[j].append(i)
    return table


def _stretch(pos: np.ndarray, bonds, i: int, j: int, target: float) -> None:
    """Move the `j` side of bond `i-j` so the bond is `target` long, in place.

    `pos` and `bonds` are both in the fragment's own indices.  Cutting the bond
    splits the fragment in two and the side holding `j` moves rigidly along the
    bond axis.  A breaking bond opened out to the contact distance is most of
    what makes the guess a saddle guess rather than a reactant geometry, and
    moving a whole side rather than one atom keeps the rest of the fragment's
    geometry intact.
    """
    remaining = frozenset(b for b in bonds if b != frozenset((i, j)))
    moving = sorted(_component(remaining, len(pos), j))
    if i in moving:
        return  # a ring bond: cutting it separates nothing to move
    offset = pos[j] - pos[i]
    pos[moving] += _unit(offset) * (target - float(np.linalg.norm(offset)))


def guess(reaction: Reaction, seed: int = 42) -> Atoms:
    """A transition-state guess for `reaction`, in combined index order.

    Built rather than interpolated, because there is nothing to interpolate
    between yet: the two endpoints of a bimolecular channel are complexes whose
    relative placement is the very thing that has to be decided, and a reactant
    complex relaxed without knowing where the reaction goes is as likely to
    settle into some other hydrogen bond as into the reactive one.

    So the changing bonds are used directly.  Each fragment is embedded on its
    own, every *breaking* bond is opened out to `CHANGING_BOND_SCALE` times the
    covalent sum, and the fragments are then docked along those same bonds --
    each incoming fragment brought in along the free-valence direction of the
    atom it is bonding to, turned so its own free valence points back, and spun
    about that axis to the angle that keeps it furthest from everything already
    placed.

    For a transfer that puts both partial bonds at the contact distance and the
    two heavy atoms at twice it, which is a symmetric saddle guess by
    construction.  Forming bonds *inside* a fragment are not opened out -- doing
    so means changing a torsion, not a bond length -- and are left to the saddle
    optimizer.

    Fragments no changing bond reaches are spectators; they are parked clear of
    the bounding sphere of everything else rather than left overlapping it.
    """
    from rdkit import Chem

    reactant, _ = _side(reaction.smiles.split(">>")[0], "reactant")
    mols = Chem.GetMolFrags(reactant, asMols=True, sanitizeFrags=False)
    groups = Chem.GetMolFrags(reactant)

    # `_fragments` reordered each fragment canonically; `_embed` works on the
    # parent's order, so the conformer is permuted to match `Fragment.indices`
    # rather than the other way round.
    local_positions, local_index = [], []
    for fragment, mol, group in zip(
        reaction.reactant_fragments, mols, groups, strict=True
    ):
        conformer = _embed(mol, seed)
        where = {atom: k for k, atom in enumerate(group)}
        local_positions.append(
            np.array([conformer[where[a]] for a in fragment.indices])
        )
        local_index.append(list(fragment.indices))

    # Open out every breaking bond, inside the fragment that still holds it.
    for f, local in enumerate(local_index):
        members = set(local)
        inside = frozenset(b for b in reaction.reactant_bonds if set(b) <= members)
        own = frozenset(frozenset(_renumber(b, local)) for b in inside)
        for bond in reaction.broken & inside:
            i, j = sorted(bond)
            _stretch(
                local_positions[f],
                own,
                local.index(i),
                local.index(j),
                _contact(reaction.numbers, i, j),
            )

    # Dock the fragments along the changing bonds that join them.
    positions = np.zeros((len(reaction), 3))
    owner = {a: f for f, idx in enumerate(local_index) for a in idx}
    placed: set[int] = set()

    def deposit(f: int, xyz: np.ndarray) -> None:
        positions[local_index[f]] = xyz
        placed.add(f)

    deposit(0, local_positions[0] - local_positions[0].mean(axis=0))
    pending = [
        bond
        for bond in sorted(map(sorted, reaction.changing))
        if owner[bond[0]] != owner[bond[1]]
    ]
    progress = True
    while progress:
        progress = False
        for bond in list(pending):
            i, j = bond
            if owner[i] not in placed and owner[j] not in placed:
                continue
            if owner[i] in placed and owner[j] in placed:
                pending.remove(bond)
                continue
            if owner[j] in placed:
                i, j = j, i
            deposit(
                owner[j],
                _dock(
                    reaction,
                    positions,
                    placed,
                    local_index,
                    local_positions,
                    anchor=i,
                    incoming=j,
                    owner=owner,
                ),
            )
            pending.remove(bond)
            progress = True

    for f in range(len(local_index)):
        if f in placed:
            continue
        settled = positions[sorted({a for g in placed for a in local_index[g]})]
        radius = float(np.linalg.norm(settled - settled.mean(axis=0), axis=1).max())
        block = local_positions[f] - local_positions[f].mean(axis=0)
        own = float(np.linalg.norm(block, axis=1).max())
        offset = np.array([radius + own + 3.0, 0.0, 0.0])
        deposit(f, block + settled.mean(axis=0) + offset)

    atoms = Atoms(numbers=reaction.numbers, positions=positions)
    atoms.set_initial_charges(reaction.charges)
    atoms.info["smiles"] = reaction.smiles
    atoms.info["spin"] = reaction.spin
    return atoms


def _renumber(bond, local) -> tuple:
    return tuple(local.index(a) for a in bond)


def _dock(
    reaction, positions, placed, local_index, local_positions, anchor, incoming, owner
) -> np.ndarray:
    """Rigidly place the fragment holding `incoming` against `anchor`."""
    f = owner[incoming]
    block = local_positions[f]
    local = local_index[f]

    settled = sorted({a for g in placed for a in local_index[g]})
    outward = _open_direction(
        positions, _neighbour_table(reaction.reactant_bonds, len(reaction)), anchor
    )
    target = positions[anchor] + outward * _contact(reaction.numbers, anchor, incoming)

    members = set(local)
    own_bonds = frozenset(
        frozenset(_renumber(b, local))
        for b in reaction.reactant_bonds
        if set(b) <= members
    )
    inward = _open_direction(
        block, _neighbour_table(own_bonds, len(local)), local.index(incoming)
    )
    turned = (block - block[local.index(incoming)]) @ _rotation(inward, -outward).T

    # Spin about the docking axis to whichever angle keeps the new fragment
    # furthest from what is already there.  The axis is a free coordinate of the
    # placement -- rotating about it moves nothing that the bond constrains --
    # so this is the cheapest clash removal available and costs one scan.
    best, best_clearance = turned, -np.inf
    for angle in np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False):
        candidate = turned @ _spin(outward, float(angle)).T + target
        gaps = np.linalg.norm(
            candidate[:, None, :] - positions[settled][None, :, :], axis=-1
        )
        clearance = float(gaps.min())
        if clearance > best_clearance:
            best, best_clearance = candidate, clearance
    return best


# ---------------------------------------------------------------------------
# stationary points
# ---------------------------------------------------------------------------


def transition_state(
    atoms: Atoms,
    calc_factory,
    fmax: float = 0.01,
    steps: int = 300,
    internal: bool = False,
    logfile=None,
) -> Atoms:
    """Refine `atoms` to a first-order saddle with Sella.

    `order=1` is the whole point: Sella follows the lowest curvature *uphill*
    and everything else downhill, so it converges on a geometry with exactly one
    negative Hessian eigenvalue instead of on a minimum.  Cartesian coordinates
    by default -- `internal=True` is faster on a covalently bound molecule, and
    wrong for the dissociated complexes a bimolecular channel passes through,
    where the internal-coordinate set has to span two pieces that are not bonded
    to each other.

    Whether the result really is a saddle is not asserted here; `endpoints` has
    to diagonalize the Hessian anyway and raises there if it is not.
    """
    from sella import Sella

    work = atoms.copy()
    work.calc = calc_factory(work)
    Sella(work, order=1, internal=internal, logfile=logfile).run(fmax=fmax, steps=steps)
    out = atoms.copy()
    out.set_positions(work.get_positions())
    return sampling.label(out, calc_factory, "transition")


def endpoints(
    ts: Atoms,
    calc_factory,
    reaction: Reaction,
    fmax: float = 1e-3,
    steps: int = 500,
    delta: float = 0.01,
    step: float = MODE_STEP,
    spins: dict | None = None,
) -> tuple[Atoms, Atoms]:
    """The two minima the saddle `ts` connects, relaxed and in reaction order.

    Walked down from the saddle rather than built independently, which is what
    makes them *this* reaction's endpoints: a complex assembled and relaxed on
    its own is a minimum, but nothing says it is the one on the far side of this
    barrier, and a coupling fitted to endpoints the saddle does not connect is
    fitted to the wrong path.  Displacing along the imaginary mode and relaxing
    is the cheap end of an IRC and gives the same two basins.

    Which relaxation is the reactant is decided by perceived connectivity, not
    by the sign of the displacement -- the sign is an eigenvector phase, and
    which way it points is up to LAPACK.

    `spins` (see `frame_spins`) sets the reactant's and the product's spin.
    When they differ, each displacement is relaxed on the surface of the side
    it heads for (`_toward`), and that guess has to survive the connectivity
    check: a displacement that relaxes into the other basin was relaxed on the
    wrong surface.
    """
    from .topology import perceive

    _, energies, modes, _ = sampling.hessian(ts, calc_factory, delta=delta)
    if energies[0] > -IMAGINARY_MODE_ENERGY:
        raise ReactionError(
            f"the refined geometry has no imaginary mode (lowest {energies[0]:+.4f} "
            "eV), so it is not a transition state. Sella converged on a minimum "
            "or a flat region; try a different guess, or a channel with a "
            "barrier -- a barrierless fission has no saddle to find and is "
            "fitted by `coupling.fit_twobody` instead."
        )
    mode = modes[0]
    mode = mode / float(np.linalg.norm(mode, axis=1).max()) * step

    spins = frame_spins(reaction, spins)
    relaxed, heading = [], []
    for sign in (+1, -1):
        displaced = ts.copy()
        displaced.info.pop("connectivity", None)
        displaced.set_positions(ts.get_positions() + sign * mode)
        heading.append(_toward(reaction, ts.get_positions(), displaced.get_positions()))
        displaced.info["spin"] = spins[heading[-1]]
        relaxed.append(
            sampling.optimize(
                displaced, calc_factory, fmax=fmax, steps=steps, kind="endpoint"
            )
        )

    found = [
        frozenset(frozenset(e) for e in perceive(frame).edges()) for frame in relaxed
    ]
    for first, second in ((0, 1), (1, 0)):
        if found[first] == reaction.reactant_bonds:
            if found[second] != reaction.product_bonds:
                break
            if spins["reactant"] != spins["product"] and heading[first] != "reactant":
                raise ReactionError(
                    "the saddle's two downhill relaxations each ended in the basin "
                    "the other was headed for, so each endpoint was relaxed on the "
                    f"other's spin surface (reactant 2S={spins['reactant']}, "
                    f"product 2S={spins['product']}). Supply the frames instead."
                )
            return _tag(relaxed[first], reaction, "reactant"), _tag(
                relaxed[second], reaction, "product"
            )
    raise ReactionError(
        "the saddle does not connect the two topologies the reaction SMILES "
        f"names. Relaxing along its imaginary mode gave {_show(found[0])} and "
        f"{_show(found[1])}; the SMILES asks for "
        f"{_show(reaction.reactant_bonds)} and {_show(reaction.product_bonds)}."
    )


def _show(bonds) -> str:
    return "{" + ", ".join(f"{i}-{j}" for i, j in sorted(map(sorted, bonds))) + "}"


def _tag(frame: Atoms, reaction: Reaction, side: str) -> Atoms:
    """Stamp a relaxed frame with the side it belongs to and that side's graph.

    The connectivity written here is the *reaction's*, not the perceived one.
    They agree -- `endpoints` only gets this far having checked that -- and
    writing the reaction's makes the frame carry the diabatic state it stands
    for rather than a distance cutoff's opinion of it, which is what
    `state_parameters` builds a force field from.
    """
    out = frame.copy()
    out.calc = frame.calc
    out.info["frame_kind"] = side
    out.info["connectivity"] = reaction.connectivity(side)
    return out


FRAME_KINDS = ("reactant", "transition", "product")


def frame_spins(reaction: Reaction, spins: dict | None = None) -> dict:
    """`2S` for each of `FRAME_KINDS`: `spins` where it says, `reaction.spin` else.

    One spin for the whole path is the usual case and the default, but a
    channel may cross between surfaces -- a spin-forbidden recombination is
    searched on one surface and relaxed on another -- so each frame can be given
    its own.
    """
    spins = dict(spins or {})
    unknown = sorted(set(spins) - set(FRAME_KINDS))
    if unknown:
        raise ReactionError(f"unknown frame kinds {unknown}; expected {FRAME_KINDS}")
    return {kind: int(spins.get(kind, reaction.spin)) for kind in FRAME_KINDS}


def _toward(reaction: Reaction, ts: np.ndarray, displaced: np.ndarray) -> str:
    """Which side a displacement from the saddle heads for, before relaxing it.

    Towards the reactant the bonds the reaction breaks shorten and the ones it
    forms lengthen.  `endpoints` needs the answer before the relaxation rather
    than after, when the endpoints are on different spin surfaces: the spin is
    part of what the relaxation is run on.
    """

    def stretch(bonds) -> float:
        return sum(
            float(np.linalg.norm(displaced[i] - displaced[j]))
            - float(np.linalg.norm(ts[i] - ts[j]))
            for i, j in map(tuple, bonds)
        )

    shorter = stretch(reaction.broken) < stretch(reaction.formed)
    return "reactant" if shorter else "product"


def stationary_points(
    reaction: Reaction,
    calc_factory,
    seed: int = 42,
    fmax: float = 1e-3,
    ts_fmax: float = 0.01,
    ts_steps: int = 300,
    hessian_delta: float = 0.01,
    initial: Atoms | None = None,
    logfile=None,
    spins: dict | None = None,
) -> list[Atoms]:
    """`[reactant, transition state, product]`, in one atom order.

    The order and the count are what `coupling.fit` reads: it takes the middle
    frame as the transition state and the two ends as the endpoints, exactly as
    the `rxn_*.xyz` layout does.

    `initial` overrides the built guess, for a channel whose saddle is known.
    `spins` sets `2S` per frame (`frame_spins`): the saddle search runs on the
    `"transition"` one, and each endpoint is relaxed on its own.
    """
    spins = frame_spins(reaction, spins)
    start = (initial if initial is not None else guess(reaction, seed=seed)).copy()
    start.info["spin"] = spins["transition"]
    saddle = transition_state(
        start, calc_factory, fmax=ts_fmax, steps=ts_steps, logfile=logfile
    )
    reactant, product = endpoints(
        saddle, calc_factory, reaction, fmax=fmax, delta=hessian_delta, spins=spins
    )
    # The saddle's own connectivity is ambiguous by construction -- that is what
    # a transition state is -- so it carries the reactant's, which is the
    # convention the datasets use and what makes the middle frame readable.
    saddle = _tag(saddle, reaction, "reactant")
    saddle.info["frame_kind"] = "transition"
    return [reactant, saddle, product]


# ---------------------------------------------------------------------------
# diabatic states
# ---------------------------------------------------------------------------


def _mapping(fragment: Fragment, built: Atoms) -> np.ndarray:
    """`mapping[k]` is where built atom `k` lives in the combined system.

    Three atom orders meet here and only two of them are ours.  `Fragment`
    numbers its atoms canonically, which is the order two occurrences of one
    molecule agree on; `build` numbers them however `molify` does, which is the
    order the *fit* happens in and the order `sampling.conformer_frames` assumes
    when it hands back a geometry for the same SMILES.  Matching the two by
    graph isomorphism is what lets the fit keep molify's order -- so every
    existing sampler stays valid -- while the merge uses the canonical one.
    """
    from molify import ase2rdkit
    from rdkit import Chem

    # `ase2rdkit` fills every atom's valence with implicit hydrogens, so a
    # radical comes back closed-shell -- `[OH]` as water's oxygen, `[O]` and
    # `[H]` as water and H2 -- and RDKit will not match an atom with no radical
    # electrons onto one with some.  The match is only there to line up atoms
    # the graph already identifies, so the radicals are left out of it.
    query = Chem.RWMol(fragment.mol)
    for atom in query.GetAtoms():
        atom.SetNumRadicalElectrons(0)
    match = ase2rdkit(built).GetSubstructMatch(query, useChirality=False)
    if len(match) != fragment.mol.GetNumAtoms():
        raise ReactionError(
            f"could not match the built conformer of {fragment.smiles!r} onto the "
            "fragment as the reaction SMILES writes it. The two describe "
            "different molecular graphs, which usually means a missing explicit "
            "hydrogen in the reaction SMILES."
        )
    mapping = np.empty(len(built), dtype=int)
    for k, index in enumerate(match):
        mapping[index] = fragment.indices[k]
    return mapping


def fit_fragments(
    reaction: Reaction,
    calc_factory,
    config=None,
    workdir: str = ".",
    seed: int = 42,
    known: dict | None = None,
) -> dict:
    """Fit one force field per distinct molecule in `reaction`.

    Keyed by `(SMILES, charge)`, so a molecule appearing on both sides -- or
    twice on one side -- is fitted once.  Each entry is `(atoms, params)` in the
    order `build` produced, which is the order the fit ran in; `state_parameters`
    is what moves them into combined indices.

    Every fit goes through `parameterize`, so each fragment gets the same
    treatment a standalone molecule would: its own relaxation, Hessian, thermal
    frames, conformers and torsion scans, against the same reference calculator
    the reaction's stationary points are found with.  That shared reference is
    what puts the diabatic energies and the reference barrier on one zero, which
    is the condition `coupling.fit_amplitude` needs.

    `known` holds fits already made, keyed and shaped the same way; a fragment
    found there is used as it stands.  That is how several reactions over the
    same molecules share one fit each, and the shared-reference condition above
    is the caller's to keep.
    """
    from pathlib import Path

    from . import parameterize

    out: dict = {}
    known = known or {}
    for side in ("reactant", "product"):
        for fragment in reaction.fragments(side):
            if fragment.key in out:
                continue
            if fragment.key in known:
                out[fragment.key] = known[fragment.key]
                continue
            built = build(fragment.smiles, seed=seed)
            if len(built) < 2:
                electrostatics = (config.electrostatics if config else "acks2")
                out[fragment.key] = (
                    built,
                    _lone_atom(built, calc_factory, electrostatics=electrostatics),
                )
                continue
            training = Path(workdir) / f"{_stem(fragment)}.xyz"
            training.parent.mkdir(parents=True, exist_ok=True)
            params = parameterize(
                built, calc_factory, config=config, training_set=str(training)
            )
            out[fragment.key] = (built, params)
    return out


def _lone_atom(
    atoms: Atoms,
    calc_factory,
    frame: Atoms | None = None,
    electrostatics: str = "acks2",
) -> Parameters:
    """The force field of a single atom, which has nothing to fit.

    A monatomic fragment -- the leaving halide of an SN2, a bare proton -- has
    no internal coordinate, so there is no geometry to relax, no Hessian to
    difference and no normal mode to sample along; `sampling.normal_mode_frames`
    says as much by refusing to run. What it does have is the two things a
    diabatic state needs from it: the element-table nonbonded parameters, and an
    `E0` that puts its energy on the reference calculator's absolute scale.

    `E0` is set the same way `fit` sets it -- the reference energy with the
    nonbonded baseline removed -- so that a `FastForces` total reproduces the
    reference energy here exactly, as it does approximately for a fitted
    molecule.

    `frame` is `atoms` already labelled as `"equilibrium"`, for a caller that
    wants to keep it; without one, `atoms` is labelled here.

    Under `electrostatics="fixed"` the atom carries its formal charge as its
    `charge` -- the only charge a lone atom can have, and what lets a halide
    leave an SN2 carrying the -1 the fixed-charge surface puts on it.
    """
    import networkx as nx

    from . import elements
    from .fit import _nonbonded

    params = Parameters(numbers=atoms.get_atomic_numbers())
    index_column = np.arange(len(atoms))[:, None]
    for term, kwargs in elements.defaults_for(params.numbers, electrostatics).items():
        params.terms[term] = {"atoms": index_column.copy(), "kwargs": dict(kwargs)}
    if electrostatics == "fixed":
        params.terms["charge"] = {
            "atoms": index_column.copy(),
            "kwargs": {"q": np.asarray(atoms.get_initial_charges(), dtype=float)},
        }

    if frame is None:
        frame = sampling.label(atoms, calc_factory, "equilibrium")
    baseline, _ = _nonbonded(params, [frame], nx.empty_graph(len(atoms)))
    e0 = float(frame.get_potential_energy() - baseline[0])
    params.terms["reference"] = {
        "atoms": np.zeros((1, 1), dtype=int),
        "kwargs": {"E0": np.array([e0])},
    }
    params.report = {"energy_rmse_eV": 0.0, "force_rmse_eV_A": 0.0, "lone_atom": True}
    return params


def _stem(fragment: Fragment) -> str:
    """A filesystem-safe stem for one fragment's training file."""
    safe = "".join(c if c.isalnum() else "_" for c in fragment.smiles)
    return f"fragment_{safe}"


def state_parameters(reaction: Reaction, side: str, fitted: dict) -> Parameters:
    """One side's force field, over the combined atom indices.

    A diabatic state is a bonding topology and nothing else, so this is the
    fragment fits scattered onto the combined system -- and the two states
    differ in exactly the bonded terms their two topologies differ in.

    The nonbonded blocks are *rebuilt* from the combined element list rather
    than merged from the fragments.  They are `elements.defaults_for`'s output,
    never fitted, and ACKS2's per-atom parameters describe atoms rather than
    molecules; concatenating the fragments' copies would produce the same
    numbers by a longer route and would leave the block's ordering depending on
    which side it came from.  What is *not* the same is the energy they add:
    ACKS2 equilibrates over whatever system it is handed, so the combined value
    is not the sum of the fragments'.  That difference is identical in both
    states -- the ACKS2 charges have no topology -- so it cancels exactly in
    the diabatic gap, and what it leaves in the mean is absorbed by the fitted
    coupling amplitude, which is fitted against these very diagonals.

    **Fixed charges are the exception, and are carried per side.**  A `charge`
    block is not an element default: the fragments' charges are molecular, so
    the two sides of a proton transfer genuinely carry different ones -- which
    is the point of them, since that is how the excess charge moves with the
    proton.  DynamicTopology evaluates exactly that (each template its own
    charges, on the diagonal), so the fragments' blocks are scattered onto the
    combined indices like every bonded term.  Every fragment of a side has to
    carry them, or none: a side half on ACKS2 and half on fixed charges has no
    single `global_params.electrostatics` to be evaluated under.

    `E0` is summed: it is a constant per molecule, so a system of several is
    their sum.  The exclusions are not built here: DynamicTopology derives them
    from each side's `bond` terms when it evaluates the state.
    """
    from . import elements

    numbers = reaction.numbers
    params = Parameters(numbers=numbers)
    index_column = np.arange(len(numbers))[:, None]
    fragments = list(reaction.fragments(side))
    for fragment in fragments:
        if fragment.key not in fitted:
            raise ReactionError(f"no force field was fitted for {fragment.smiles!r}")
    fixed = {"charge" in fitted[f.key][1].terms for f in fragments}
    if len(fixed) > 1:
        raise ReactionError(
            f"the {side} fragments disagree about their electrostatics: some "
            "carry fixed `charge` terms and some ACKS2, and a state is evaluated "
            "under one.  Refit them with the same `FitConfig.electrostatics`"
        )
    fixed_charges = fixed == {True}

    defaults = elements.defaults_for(
        numbers, "fixed" if fixed_charges else "acks2"
    )
    for term, kwargs in defaults.items():
        params.terms[term] = {"atoms": index_column.copy(), "kwargs": dict(kwargs)}

    collected: dict[str, dict] = {}
    e0 = 0.0
    for fragment in fragments:
        built, fragment_params = fitted[fragment.key]
        mapping = _mapping(fragment, built)
        e0 += fragment_params.e0
        for term, block in fragment_params.terms.items():
            if term in ("atom", "lennardjones", "reference"):
                continue
            entry = collected.setdefault(term, {"atoms": [], "kwargs": {}})
            entry["atoms"].append(mapping[np.asarray(block["atoms"])])
            for name, value in block["kwargs"].items():
                entry["kwargs"].setdefault(name, []).append(np.asarray(value))

    for term, entry in collected.items():
        params.terms[term] = {
            "atoms": np.concatenate(entry["atoms"], axis=0),
            "kwargs": {
                name: np.concatenate(values) for name, values in entry["kwargs"].items()
            },
        }
    params.terms["reference"] = {
        "atoms": np.zeros((1, 1), dtype=int),
        "kwargs": {"E0": np.array([e0])},
    }
    return params


# ---------------------------------------------------------------------------
# the whole thing
# ---------------------------------------------------------------------------


class ReactionParameters:
    """Everything a reaction's EVB surface is made of, and how to check it.

    `states` holds the two diabatic force fields, keyed `"reactant"` and
    `"product"`; `coupling` the off-diagonal; `frames` the three stationary
    points the coupling was fitted to; `fragments` the per-molecule fits the
    states were assembled from, kept because they are the expensive part and are
    reusable on their own.
    """

    def __init__(self, reaction, states, coupling, frames, fragments):
        self.reaction = reaction
        self.states = states
        self.coupling = coupling
        self.frames = frames
        self.fragments = fragments

    def __repr__(self) -> str:
        kind, _ = self.reaction.channel()
        return (
            f"ReactionParameters({kind}, {len(self.reaction)} atoms, {self.coupling!r})"
        )

    @property
    def state_list(self) -> list:
        return [self.states["reactant"], self.states["product"]]

    def calculator(self, atoms: Atoms = None):
        """An `EVB` calculator for this reaction."""
        from .evb import EVB

        return EVB(atoms=atoms, states=self.state_list, coupling=self.coupling)

    def energies(self) -> dict:
        """Reference and EVB energies at the three stationary points, in eV.

        The check the whole fit exists to pass: `evb` at the transition state
        should reproduce `reference` there, because that is the equation
        `coupling.fit_amplitude` inverted.  The two endpoints are *not* fitted
        and are the honest test -- the coupling has quenched to `eps` there by
        construction, so what is left is the diabatic force fields alone.
        """
        out = {"reference": [], "evb": [], "diabatic": []}
        calc = self.calculator()
        for frame in self.frames:
            work = frame.copy()
            work.calc = calc
            # nan for a fission frame the reference could not label (see
            # `manifest._relabel`); the fit never reads one.
            reference = frame.get_potential_energy() if frame.calc is not None else np.nan
            out["reference"].append(float(reference))
            out["evb"].append(float(work.get_potential_energy()))
            # The Hamiltonian's diagonal is each state's own energy, unmixed.
            out["diabatic"].append(tuple(map(float, np.diag(calc.results["hamiltonian"]))))
        return out

    def report(self) -> str:
        energies = self.energies()
        reference, evb = energies["reference"], energies["evb"]
        lines = [
            f"{'':12s} {'reference':>12s} {'EVB':>12s} {'error':>10s}"
            f" {'H_react':>10s} {'H_prod':>10s}"
        ]
        # A fission carries its reactant alone, so there is no barrier to report.
        names = ("reactant", "transition", "product")
        for name, i in zip(names, range(len(self.frames))):
            h1, h2 = energies["diabatic"][i]
            lines.append(
                f"{name:12s} {reference[i]:12.5f} {evb[i]:12.5f} "
                f"{evb[i] - reference[i]:10.5f} {h1:10.4f} {h2:10.4f}"
            )
        if len(self.frames) == 3:
            lines.append(
                f"{'barrier':12s} {reference[1] - reference[0]:12.5f} "
                f"{evb[1] - evb[0]:12.5f} "
                f"{(evb[1] - evb[0]) - (reference[1] - reference[0]):10.5f}"
            )
        lines.append(f"coupling     {self.coupling!r}")
        return "\n".join(lines)

    def write(self, stem: str) -> None:
        """`<stem>.xyz`, `<stem>.jsonl` and one jsonl per diabatic state.

        The layout the reference datasets use: the three frames in one extended
        XYZ, each carrying its own connectivity, and the coupling in a jsonl of
        its own beside them.
        """
        from pathlib import Path

        from .io import write_frames

        stem_path = Path(stem)
        stem_path.parent.mkdir(parents=True, exist_ok=True)
        write_frames(stem_path.with_suffix(".xyz"), self.frames)
        self.coupling.to_jsonl(str(stem_path.with_suffix(".jsonl")))
        for side, params in self.states.items():
            params.to_jsonl(str(stem_path.parent / f"{stem_path.name}-{side}.jsonl"))


def parameterize(
    smiles: str,
    calc_factory,
    config=None,
    workdir: str = ".",
    seed: int = 42,
    eps: float = 1e-3,
    amplitude: float | None = None,
    initial: Atoms | None = None,
    logfile=None,
    fragments: dict | None = None,
    frames: list[Atoms] | None = None,
    spins: dict | None = None,
):
    """Fit a whole EVB surface from an atom-mapped reaction SMILES.

    Three independent stages, in the only order they can run in:

    1. **Fragments.** Every distinct molecule on either side gets its own
       `fastforces.parameterize` against `calc_factory` -- the same relaxation,
       Hessian, thermal frames, conformers and torsion scans a standalone
       molecule would get.  This is the expensive stage and the reusable one.
    2. **Stationary points.** The saddle is built from the changing bonds and
       refined with Sella, and the two endpoints are relaxed down from it along
       its imaginary mode, so all three are on one reaction path in one atom
       order.
    3. **Coupling.** The channel's own connectivity change picks the form; the
       amplitude comes from inverting the 2x2 at the saddle against the same
       reference calculator's barrier, and the width from quenching to `eps` at
       the nearer endpoint.

    All three stages use `calc_factory`, and that is not incidental: `E0` is
    fitted per fragment from its energies, so a diabatic total is on its
    absolute scale, and the reference barrier is on the same one.  Fitting the
    fragments against one method and the barrier against another puts two
    different zeros into `fit_amplitude` and the amplitude absorbs the
    difference.

    Either expensive stage can be handed in instead.  `fragments` is fits
    already made, as `fit_fragments` returns them; any molecule it lacks is
    still fitted.  `frames` is the stationary points -- `[reactant, transition
    state, product]`, or the reactant alone for a fission -- from a previous run,
    a constrained path or a higher-level calculation, in this reaction's atom
    order and carrying reference energies.  Both still have to share one
    reference with each other, for the reason above.

    `spins` is `2S` per frame kind (`frame_spins`); anything it leaves out is
    `Reaction.spin`, every reactant radical high-spin coupled.  It reaches only
    the frames computed here: supplied `frames` carry the spin they were
    computed at, and fragments are fitted at their own.
    """
    from . import coupling as coupling_module

    reaction = parse(smiles)
    fragments = fit_fragments(
        reaction,
        calc_factory,
        config=config,
        workdir=workdir,
        seed=seed,
        known=fragments,
    )
    states = {
        side: state_parameters(reaction, side, fragments)
        for side in ("reactant", "product")
    }

    kind, _ = reaction.channel()
    if frames is not None:
        frames = list(frames)
    elif kind == "fission":
        # A barrierless fission has no saddle to find, and `fit_twobody` needs
        # none: it walks the two diabats apart along the breaking bond instead.
        # The reactant geometry is still wanted, and it is the relaxed complex.
        start = (guess(reaction, seed=seed) if initial is None else initial).copy()
        start.info["spin"] = frame_spins(reaction, spins)["reactant"]
        reactant = sampling.optimize(
            start,
            calc_factory,
            fmax=1e-3,
            kind="reactant",
        )
        frames = [_tag(reactant, reaction, "reactant")]
    else:
        frames = stationary_points(
            reaction,
            calc_factory,
            seed=seed,
            initial=initial,
            logfile=logfile,
            spins=spins,
        )

    fitted = coupling_module.fit(
        reaction,
        frames,
        [states["reactant"], states["product"]],
        amplitude=amplitude,
        eps=eps,
    )
    return ReactionParameters(reaction, states, fitted, frames, fragments)
