"""Fitting a whole dataset from one manifest file.

A manifest names every molecule and reaction of a dataset, where each one's
force field goes, and how they are all fitted:

    {
        "name": "Water",
        "molecules": [{"id": 1, "smiles": "O", "path": "molecules/h2o"}, ...],
        "reactions": [{"id": 1, "smiles": "[OH3+].O>>O.[OH3+]",
                       "path": "reactions/h3o-h2o-transfer"}, ...],
        "global_params": {"bond_asymptote": 1.0, ...},
        "fit_config": {"calculator": {"name": "tblite"}, "workdir": "training"}
    }

Every `path` is an output stem relative to the manifest's own directory, and is
written in the layout the reference datasets already use:

    molecule   <path>.jsonl     the force field
               <path>.xyz       its equilibrium frame, with the metadata
    reaction   <path>.xyz       reactant, transition state and product
               <path>.jsonl     the coupling
               <path>-reactant.jsonl, <path>-product.jsonl   the two states

The full training sets go under `fit_config.workdir` instead, mirroring the
same paths.  They are what make a rerun cheap: a molecule whose training set is
already there is refit from it without the reference calculator, and a
reaction whose stationary points are there skips the saddle search.  Delete the
file to regenerate it.

A reaction's `<path>.xyz` is also an *input*.  One already at the output path
-- a published dataset's reactant, TS and product, say -- is taken as the
reaction's geometry and the saddle search is skipped; the frames are
relabelled with the reference calculator, since their energies came from
something else, and the labelled frames become the cache (`_frame_source`).  A
searched path is written there too, so regenerating one means deleting both
files.  A change to the sampling half of `fit_config` --
`n_mode_frames`, `temperature` -- therefore does not reach a molecule until its
training set is deleted; a change to the fitting half does.

Spin is `2S` and fitting-only; DynamicTopology ignores it.  A molecule entry's
`spin` defaults to its SMILES radicals.  A reaction entry's `spin` sets every
frame the reaction computes -- the saddle search and both endpoint relaxations
-- and `spin_r`, `spin_ts` and `spin_p` set one each (`_frame_spins`); anything
left unset is every reactant radical high-spin coupled (`Reaction.spin`).

Molecules are fitted before reactions, and a reaction's fragments are taken
from the molecule fits wherever the manifest lists that molecule, so the water
in three proton transfers is fitted once.  A fragment it does not list is
fitted anyway, into `<workdir>/fragments`.

**`global_params` is DynamicTopology's, and it is applied.**  A manifest is the
same file DynamicTopology loads, and its `global_params` are the constants the
dataset is fitted at -- DynamicTopology's `forcefield/REFERENCE.md` §7.2 is
explicit that a parameter set is valid only at those values.  `run` fits every
entry under them (`DynamicTopology.forcefield.params.use`), so a dataset is
fitted on exactly the surface it will be simulated on.  Values it leaves out
take DynamicTopology's defaults.

Everything is validated when the manifest is loaded -- unknown keys, duplicate
outputs, reaction SMILES that cannot be mapped -- because the alternative is a
typo found hours into a DFT run.
"""

import json
import traceback
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
from DynamicTopology.forcefield.params import ForceFieldParams, use

from . import io, sampling
from . import reaction as reaction_module
from .fit import FitConfig


class ManifestError(ValueError):
    """Raised for a manifest that cannot be run as written."""


# ---------------------------------------------------------------------------
# global parameters
# ---------------------------------------------------------------------------

# `FitConfig.electrostatics` <-> `global_params.electrostatics`.
_ELECTROSTATICS = {"acks2": "acks2", "fixed": "pointcharge"}


def resolve_globals(
    stated: dict | None, config: FitConfig, source: str, config_states: bool = False
):
    """The dataset's `ForceFieldParams`, reconciled with `fit_config`.

    `global_params` is DynamicTopology's own block -- `ForceFieldParams` parses
    it, unknown keys and bad values included -- and it is *applied*: `run`
    fits the whole manifest under it, so every template is fitted on the
    surface the dataset says it was fitted on.

    `electrostatics` is the one field both blocks can state: DynamicTopology's
    `"acks2"` / `"pointcharge"` and `FitConfig`'s `"acks2"` / `"fixed"`.  Either
    may be given alone and the other follows; both, and they have to agree.
    `config_states` says whether `fit_config` stated it or `config` merely
    carries the default.
    """
    try:
        params = ForceFieldParams.from_dict(stated or {}, source=source)
    except ValueError as error:
        raise ManifestError(str(error)) from error
    wanted = _ELECTROSTATICS[config.electrostatics]
    if "electrostatics" not in (stated or {}):
        return replace(params, electrostatics=wanted)
    if config_states and params.electrostatics != wanted:
        raise ManifestError(
            f"global_params.electrostatics={params.electrostatics!r} but "
            f"fit_config.electrostatics={config.electrostatics!r}; they name "
            "the same choice and have to agree"
        )
    reverse = {v: k for k, v in _ELECTROSTATICS.items()}
    config.electrostatics = reverse[params.electrostatics]
    return params


# ---------------------------------------------------------------------------
# the reference calculator
# ---------------------------------------------------------------------------


def _charge(atoms) -> int:
    return int(round(float(np.sum(atoms.get_initial_charges()))))


def _unpaired(atoms) -> int:
    """`2S` for `atoms`: as stated on it, or the lowest the electron count allows.

    A molecule entry states it -- from its SMILES radicals, or explicitly -- and
    it is set on the atoms before they are fitted; a reaction's frames carry
    theirs from `reaction.frame_spins`.  Anything else falls back to the parity
    of the electron count, which is right for every closed-shell system.
    """
    if "spin" in atoms.info:
        return int(atoms.info["spin"])
    return int((int(atoms.get_atomic_numbers().sum()) - _charge(atoms)) % 2)


def calculator_factory(spec: dict):
    """`(calc_factory, method label)` for a `fit_config.calculator` block.

    `name` picks the backend and every other key is passed to its constructor.
    Charge and spin are read off each geometry rather than fixed here, because
    one manifest fits cations, anions and radicals against the same method.
    """
    spec = dict(spec)
    name = spec.pop("name", "tblite")
    if name == "tblite":
        method = spec.pop("method", "GFN2-xTB")
        spec.setdefault("verbosity", 0)

        def factory(atoms=None):
            from .calculators.tblite import TBLiteCalculator

            if atoms is None:
                return TBLiteCalculator(method=method, **spec)
            return TBLiteCalculator(
                method=method,
                charge=_charge(atoms),
                multiplicity=_unpaired(atoms) + 1,
                **spec,
            )

        return factory, method

    if name == "pyscf":
        xc = spec.pop("xc", "HYB_GGA_XC_WB97X_V")
        basis = spec.pop("basis", "cc-pvtz")

        def factory(atoms=None):
            from .calculators.pyscf import PySCFCalculator

            if atoms is None:
                return PySCFCalculator(xc=xc, basis=basis, **spec)
            return PySCFCalculator(
                charge=_charge(atoms),
                spin=_unpaired(atoms),
                xc=xc,
                basis=basis,
                **spec,
            )

        label = f"{xc}/{basis}"
        if spec.get("pcm"):
            label += f"/{spec['pcm']}(eps={spec.get('pcm_eps', 78.3553)})"
        return factory, label

    raise ManifestError(
        f"unknown calculator {name!r}; expected 'tblite' or 'pyscf', or pass "
        "`calc_factory` to `run` for anything else"
    )


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

# fit_config keys that are not `FitConfig` fields
RUN_KEYS = ("calculator", "workdir", "embed_seed", "eps")
MOLECULE_KEYS = ("id", "smiles", "path", "spin")
REACTION_KEYS = ("id", "smiles", "path", "frames", "amplitude", "spin")
REACTION_KEYS += ("spin_r", "spin_ts", "spin_p")
# Per-frame spin keys of a reaction entry, and the frame kind each one sets.
# `spin` sets all three and these override it.
FRAME_SPIN_KEYS = {"spin_r": "reactant", "spin_ts": "transition", "spin_p": "product"}


@dataclass
class Entry:
    """One molecule or reaction, with its paths resolved against the manifest."""

    kind: str
    id: object
    smiles: str
    output: Path  # a stem: `_file` adds each output's suffix
    training: Path  # the same stem, under the workdir
    options: dict = field(default_factory=dict)
    # molecules: the `Fragment.key` a reaction would give it
    key: tuple | None = None
    # reactions: the fully mapped SMILES `map_atoms` produced
    mapped: str | None = None
    # reactions: `2S` per frame kind, resolved by `_frame_spins`
    spins: dict | None = None

    @property
    def label(self) -> str:
        return f"{self.kind} {self.id} ({self.smiles})"


@dataclass
class Manifest:
    path: Path
    name: str
    molecules: list[Entry]
    reactions: list[Entry]
    config: FitConfig
    workdir: Path
    calculator: dict
    embed_seed: int = 42
    eps: float = 1e-3
    meta: dict = field(default_factory=dict)
    # The dataset's `global_params`, resolved; see `resolve_globals`.
    params: ForceFieldParams = field(default_factory=ForceFieldParams)

    @property
    def root(self) -> Path:
        return self.path.parent

    def plan(self) -> str:
        """What `run` would do, one line per entry."""
        lines = [f"{self.name}: {self.path}", f"  training sets under {self.workdir}"]
        for entry in self.molecules + self.reactions:
            notes = []
            if entry.kind == "reaction":
                notes.extend(self._frame_notes(entry))
            elif _cached(entry):
                notes.append("training set present")
            if any(entry.output.parent.glob(entry.output.name + ".*")):
                notes.append("overwrites existing output")
            note = f" ({', '.join(notes)})" if notes else ""
            lines.append(f"  {entry.label} -> {_relative(entry.output, self.root)}{note}")
            if entry.mapped and entry.mapped != entry.smiles:
                lines.append(f"      mapped as {entry.mapped}")
            if entry.spins is not None:
                shown = ", ".join(f"{k} {v}" for k, v in entry.spins.items())
                lines.append(f"      2S: {shown}")
        return "\n".join(lines)

    def _frame_notes(self, entry: Entry) -> list[str]:
        reaction = reaction_module.parse(entry.mapped)
        try:
            source, path, _ = _frame_source(entry, reaction)
        except ManifestError as error:  # `run` reports it against the entry
            return [str(error)]
        if source == "cache":
            return ["training set present"]
        if source == "frames":
            return [f"stationary points from {_relative(path, self.root)}"]
        if source == "output":
            return ["stationary points from its .xyz, relabelled"]
        if reaction.channel()[0] == "fission":
            return ["reactant relaxed"]
        return ["saddle search"]


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _file(stem: Path, suffix: str) -> Path:
    """`<stem><suffix>`.  Not `with_suffix`, which would eat a dot in the name."""
    return stem.parent / (stem.name + suffix)


def _cached(entry: Entry) -> bool:
    return _file(entry.training, ".xyz").exists()


def _entries(raw: list, kind: str, allowed, root: Path, workdir: Path) -> list:
    entries, ids = [], set()
    for i, item in enumerate(raw):
        where = f"{kind} #{i + 1}"
        if not isinstance(item, dict):
            raise ManifestError(f"{where} is not an object")
        unknown = sorted(set(item) - set(allowed))
        if unknown:
            raise ManifestError(f"{where} has unknown keys {unknown}; allowed: {allowed}")
        for key in ("smiles", "path"):
            if not item.get(key):
                raise ManifestError(f"{where} has no {key!r}")
        entry_id = item.get("id", i + 1)
        if entry_id in ids:
            raise ManifestError(f"two {kind}s share id {entry_id!r}")
        ids.add(entry_id)
        options = {k: v for k, v in item.items() if k not in ("id", "smiles", "path")}
        if "frames" in options:
            options["frames"] = root / options["frames"]
        entries.append(
            Entry(
                kind=kind,
                id=entry_id,
                smiles=item["smiles"],
                output=root / item["path"],
                training=workdir / item["path"],
                options=options,
            )
        )
    return entries


def _checked_spin(entry: Entry, key: str, electrons: int) -> int:
    """`entry.options[key]` as a `2S` the electron count allows."""
    value = entry.options[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestError(
            f"{entry.label}: {key!r} is 2S, a non-negative integer; got {value!r}"
        )
    if (electrons - value) % 2:
        raise ManifestError(
            f"{entry.label}: {key!r} = {value} is impossible with {electrons} "
            f"electrons -- 2S has the parity of the electron count"
        )
    return value


def _electrons(smiles: str) -> int:
    from rdkit import Chem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    return sum(a.GetAtomicNum() - a.GetFormalCharge() for a in mol.GetAtoms())


def _frame_spins(entry: Entry, reaction) -> dict:
    """A reaction entry's `2S` per frame kind: its `spin*` keys over the default.

    The default is `Reaction.spin`, every reactant radical high-spin coupled.  A
    fission has only its reactant frame -- `fit_twobody` walks the diabats apart
    instead of searching for a saddle -- so a transition-state or product spin
    on one would be silently dropped, and is refused instead.
    """
    electrons = int(reaction.numbers.sum()) - reaction.charge
    spins = {}
    if "spin" in entry.options:
        value = _checked_spin(entry, "spin", electrons)
        spins = dict.fromkeys(reaction_module.FRAME_KINDS, value)
    for key, kind in FRAME_SPIN_KEYS.items():
        if key in entry.options:
            spins[kind] = _checked_spin(entry, key, electrons)
    channel, _ = reaction.channel()
    stray = [k for k in ("spin_ts", "spin_p") if k in entry.options]
    if channel == "fission" and stray:
        raise ManifestError(
            f"{entry.label}: a fission is fitted from its reactant frame alone, so "
            f"{stray} would set the spin of a frame that is never computed; use "
            "'spin' or 'spin_r'"
        )
    return reaction_module.frame_spins(reaction, spins)


def load(path) -> Manifest:
    """Read and validate a manifest, without fitting anything."""
    path = Path(path).resolve()
    with open(path) as handle:
        raw = json.load(handle)
    root = path.parent

    fit_config = dict(raw.get("fit_config", {}))
    known = [f.name for f in fields(FitConfig)]
    unknown = sorted(set(fit_config) - set(known) - set(RUN_KEYS))
    if unknown:
        raise ManifestError(
            f"unknown fit_config keys {unknown}; allowed: {list(RUN_KEYS) + known}"
        )
    config = FitConfig(**{k: v for k, v in fit_config.items() if k in known})
    workdir = root / fit_config.get("workdir", "training")
    calculator = dict(fit_config.get("calculator", {"name": "tblite"}))
    calculator_factory(calculator)  # refuse an unknown backend now

    params = resolve_globals(
        raw.get("global_params"),
        config,
        str(path),
        config_states="electrostatics" in fit_config,
    )

    molecules = _entries(raw.get("molecules", []), "molecule", MOLECULE_KEYS, root, workdir)
    reactions = _entries(raw.get("reactions", []), "reaction", REACTION_KEYS, root, workdir)

    outputs = [e.output for e in molecules + reactions]
    clashes = sorted({str(p) for p in outputs if outputs.count(p) > 1})
    if clashes:
        raise ManifestError(f"several entries write to the same path: {clashes}")

    for entry in molecules:
        entry.key = molecule_key(entry.smiles)
        if "spin" in entry.options:
            _checked_spin(entry, "spin", _electrons(entry.smiles))
    for entry in reactions:
        try:
            entry.mapped = reaction_module.map_atoms(entry.smiles)
            parsed = reaction_module.parse(entry.mapped)
        except reaction_module.ReactionError as error:
            raise ManifestError(f"{entry.label}: {error}") from error
        entry.spins = _frame_spins(entry, parsed)

    meta = {
        k: v
        for k, v in raw.items()
        if k not in ("molecules", "reactions", "global_params", "fit_config")
    }
    return Manifest(
        path=path,
        name=str(raw.get("name", path.stem)),
        molecules=molecules,
        reactions=reactions,
        config=config,
        workdir=workdir,
        calculator=calculator,
        embed_seed=int(fit_config.get("embed_seed", 42)),
        eps=float(fit_config.get("eps", 1e-3)),
        meta=meta,
        params=params,
    )


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    entry: Entry
    ok: bool
    written: list[Path] = field(default_factory=list)
    detail: str = ""


def molecule_key(smiles: str) -> tuple[str, int]:
    """The `Fragment.key` a reaction would give this molecule.

    Computed by the same `_fragments` a reaction side goes through, so a
    molecule written `[OH3+]` in the manifest and `[O+:1]([H:2])...` inside a
    reaction are recognized as one.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ManifestError(f"RDKit could not parse {smiles!r}")
    fragments = reaction_module._fragments(Chem.AddHs(mol))
    if len(fragments) != 1:
        raise ManifestError(
            f"molecule {smiles!r} is {len(fragments)} disconnected pieces; list "
            "each as its own molecule"
        )
    return fragments[0].key


def _radicals(smiles: str) -> int:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    return int(sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()))


def fit_molecule(manifest: Manifest, entry: Entry, calc_factory, method: str):
    """Fit one molecule entry; `(written paths, (equilibrium, params))`."""
    import fastforces as ff

    atoms = ff.build(entry.smiles, seed=manifest.embed_seed)
    atoms.info["spin"] = int(entry.options.get("spin", _radicals(entry.smiles)))
    if method is not None:
        atoms.info["method"] = method
    training = _file(entry.training, ".xyz")

    if len(atoms) < 2:
        frame = ff.sampling.label(atoms, calc_factory, "equilibrium")
        params = reaction_module._lone_atom(
            atoms,
            calc_factory,
            frame=frame,
            electrostatics=manifest.config.electrostatics,
        )
        equilibrium = frame
        meta = {
            "method": method or type(calc_factory(atoms)).__name__,
            "smiles": entry.smiles,
            "charge": _charge(atoms),
            "spin": atoms.info["spin"],
        }
    else:
        if training.exists():
            cached = io.read_training_set(str(training)).meta.get("spin")
            if cached is not None and int(cached) != atoms.info["spin"]:
                raise ManifestError(
                    f"{entry.label}: the training set {training} was computed at "
                    f"2S={cached}, not the {atoms.info['spin']} asked for; delete "
                    "it to regenerate"
                )
            # A set from before the asymptote was fitted has no fragments.
            ff.add_fragment_frames(str(training), calc_factory)
            params = ff.fit_from_file(str(training), config=manifest.config)
        else:
            training.parent.mkdir(parents=True, exist_ok=True)
            params = ff.parameterize(
                atoms, calc_factory, config=manifest.config, training_set=str(training)
            )
        data = io.read_training_set(str(training))
        equilibrium, meta = data.equilibrium, data.meta

    jsonl, xyz = _file(entry.output, ".jsonl"), _file(entry.output, ".xyz")
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    params.to_jsonl(str(jsonl))
    io.write_training_set(str(xyz), [equilibrium], meta=meta)
    return [jsonl, xyz], (equilibrium, params)


def _read_frames(path: Path) -> list:
    return io.read_frames(path)


def _same_geometry(first: list, second: list) -> bool:
    """Whether two frame lists are one path: same atoms, positions to file precision."""
    return len(first) == len(second) and all(
        np.array_equal(a.get_atomic_numbers(), b.get_atomic_numbers())
        and np.allclose(a.get_positions(), b.get_positions(), rtol=0.0, atol=1e-6)
        for a, b in zip(first, second)
    )


def _pairs(connectivity) -> frozenset:
    """An extxyz `connectivity` list as the bond set `Reaction.bonds` compares to."""
    return frozenset(frozenset((int(i), int(j))) for i, j, *_ in connectivity)


def _in_reaction_order(entry: Entry, reaction, frames: list, path: Path) -> list:
    """`frames` renumbered onto the reaction's atoms, matched by bonds, not index.

    An output geometry is usually someone else's file -- a DynamicTopology
    dataset, which matches templates by graph isomorphism and so never cared
    what order its atoms are in -- so its numbering is rarely the one the
    mapped SMILES gives, and the states and the coupling are written in that
    one.  The atoms are matched on element and on the bonds of both ends
    together: an atom's reactant bonds alone would not say which of two
    hydrogens on one oxygen is the one that leaves.  Any match is as good as
    any other, since two only differ by a symmetry of the reaction, but the file's
    own numbering is kept whenever it is one, so that a file this run wrote
    reads back unchanged and still matches its cache.

    Frames stating no connectivity are left alone, and `_check_frames` holds
    them to the reaction's index order as it stands.
    """
    import networkx as nx
    from networkx.algorithms.isomorphism import GraphMatcher

    ends = [(frames[0], "reactant")]
    if len(frames) > 1:
        ends.append((frames[-1], "product"))
    if any("connectivity" not in f.info for f, _ in ends):
        return frames
    found = {side: _pairs(f.info["connectivity"]) for f, side in ends}
    numbers = frames[0].get_atomic_numbers()
    if np.array_equal(numbers, reaction.numbers) and all(
        bonds == reaction.bonds(side) for side, bonds in found.items()
    ):
        return frames

    def graph(numbers, bonds_by_side) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from((i, {"Z": int(z)}) for i, z in enumerate(numbers))
        for side, bonds in bonds_by_side.items():
            for i, j in map(tuple, bonds):
                sides = g.edges[i, j]["sides"] if g.has_edge(i, j) else frozenset()
                g.add_edge(i, j, sides=sides | {side})
        return g

    matcher = GraphMatcher(
        graph(numbers, found),
        graph(reaction.numbers, {side: reaction.bonds(side) for side in found}),
        node_match=lambda a, b: a["Z"] == b["Z"],
        edge_match=lambda a, b: a["sides"] == b["sides"],
    )
    mapping = next(matcher.isomorphisms_iter(), None)  # file index -> reaction index
    if mapping is None:
        raise ManifestError(
            f"{entry.label}: {path} is not this reaction -- no numbering of its "
            f"atoms gives the reactant and product bonds of {entry.mapped}"
        )
    order = sorted(mapping, key=mapping.get)  # reaction index -> file index
    reordered = []
    for frame in frames:
        new = frame[order]
        new.info = dict(frame.info)
        if "connectivity" in frame.info:
            new.info["connectivity"] = [
                [mapping[int(i)], mapping[int(j)], *rest]
                for i, j, *rest in frame.info["connectivity"]
            ]
        reordered.append(new)
    return reordered


def _frame_source(entry: Entry, reaction):
    """`(source, path, frames)`: where a reaction's stationary points come from.

    In order: `"frames"`, a file the manifest names, taken as it stands;
    `"output"`, a `<path>.xyz` already at the entry's output path, whose
    *geometries* are taken -- renumbered into the reaction's order
    (`_in_reaction_order`) -- and relabelled with the reference calculator;
    `"cache"`, the training set an earlier run left in the workdir; and `None`,
    nothing, so the saddle is searched for.

    The output file outranks the cache because it is the one a person puts
    there -- the geometries of a published dataset, a constrained path -- and
    the cache is only ever this run's own.  A cache is still used when it holds
    the output's geometries, since it is then those same frames already
    labelled: every run leaves the two in that state, so only the first run
    after the output file changes pays for the relabelling.
    """
    if "frames" in entry.options:
        path = Path(entry.options["frames"])
        if not path.exists():
            raise ManifestError(f"{entry.label}: frames file {path} does not exist")
        return "frames", path, _read_frames(path)
    output, cache = _file(entry.output, ".xyz"), _file(entry.training, ".xyz")
    if output.exists():
        frames = _in_reaction_order(entry, reaction, _read_frames(output), output)
        if cache.exists():
            cached = _read_frames(cache)
            if _same_geometry(frames, cached):
                return "cache", cache, cached
        return "output", output, frames
    if cache.exists():
        return "cache", cache, _read_frames(cache)
    return None, None, None


_SOURCES = {
    "frames": "a file the manifest names",
    "output": "the reaction's output file",
    "cache": "a cached training set; delete it to regenerate",
}


def _frame_kinds(n: int) -> list[str]:
    """The `FRAME_KINDS` entry of each of `n` frames: reactant first, product last.

    Three frames are exactly `FRAME_KINDS`.  A fission may carry a single
    reactant, or a longer scan, whose inner frames are taken as the path's
    middle.
    """
    if n == 1:
        return ["reactant"]
    return ["reactant"] + ["transition"] * (n - 2) + ["product"]


def _supplied_frames(entry: Entry, reaction, calc_factory=None):
    """`(frames, source)`: the stationary points this entry already has.

    `source` is `_frame_source`'s, and `(None, None)` means there are none.
    Frames from the output file are relabelled with `calc_factory`, and without
    one only checked: their energies are whatever produced them, which is
    rarely this manifest's calculator, and `parameterize` needs the barrier on
    the same zero as the fragment fits.
    """
    source, path, frames = _frame_source(entry, reaction)
    if source is None:
        return None, None
    _check_frames(entry, reaction, frames, path, source)
    if source == "output" and calc_factory is not None:
        frames = _relabel(entry, reaction, frames, calc_factory)
    return frames, source


def _relabel(entry: Entry, reaction, frames: list, calc_factory) -> list:
    """`frames`' geometries, labelled afresh at the entry's charge and spins.

    Only the numbers, positions and cell survive: whatever else the file
    carried -- its energies, a stated method, a perceived connectivity -- came
    from somewhere else.  Each frame is stamped with the reaction's own graph,
    as `stationary_points` stamps a searched path, the middle ones with the
    reactant's.

    A fission's coupling is fitted from its reactant alone (`coupling.fit`);
    the rest of its path is there for DynamicTopology, which reads the
    product's topology off the last frame, and for the report.  Two fragments
    several angstroms apart are where an SCF is least likely to converge -- the
    4 A frame of O2 -> 2 O does not under B3LYP -- so a fission frame past the
    reactant that fails keeps its geometry unlabelled rather than costing the
    channel.
    """
    from ase import Atoms

    fission = reaction.channel()[0] == "fission"
    labelled = []
    for frame, kind in zip(frames, _frame_kinds(len(frames))):
        atoms = Atoms(
            numbers=frame.get_atomic_numbers(),
            positions=frame.get_positions(),
            cell=frame.get_cell(),
            pbc=frame.get_pbc(),
        )
        atoms.set_initial_charges(reaction.charges)
        atoms.info["spin"] = entry.spins[kind]
        strict = not fission or kind == "reactant"
        atoms = sampling.label(atoms, calc_factory, kind, strict=strict) or atoms
        atoms = reaction_module._tag(
            atoms, reaction, "product" if kind == "product" else "reactant"
        )
        atoms.info["frame_kind"] = kind
        labelled.append(atoms)
    return labelled


def _check_frames(entry: Entry, reaction, frames: list, path: Path, source: str):
    """Refuse frames that are not this reaction's path at this entry's spins."""
    kind, _ = reaction.channel()
    if (kind == "fission" and not frames) or (kind != "fission" and len(frames) != 3):
        needs = "the reactant" if kind == "fission" else "reactant, TS and product"
        raise ManifestError(
            f"{entry.label}: {path} holds {len(frames)} frames; a {kind} channel "
            f"needs {needs}"
        )

    checks = [(frames[0], "reactant")]
    if len(frames) > 1:
        checks.append((frames[-1], "product"))
    for frame, side in checks:
        if not np.array_equal(frame.get_atomic_numbers(), reaction.numbers):
            raise ManifestError(
                f"{entry.label}: the atoms in {path} are not in this reaction's "
                "order -- write the reaction atom-mapped to match the file"
            )
        stored = frame.info.get("connectivity")
        if stored is not None and _pairs(stored) != reaction.bonds(side):
            raise ManifestError(
                f"{entry.label}: the {side} frame in {path} is bonded differently "
                f"from the mapped reaction {entry.mapped} -- write the reaction "
                "atom-mapped to match the file"
            )
        # Output geometries are relabelled, so what they carry does not matter,
        # and a fission is fitted from its reactant alone (see `_relabel`).
        needed = source != "output" and not (kind == "fission" and side == "product")
        if needed and (frame.calc is None or "energy" not in frame.calc.results):
            raise ManifestError(f"{entry.label}: {path} carries no reference energies")
    # A frame computed at one spin is not reused for another.  One without a
    # stated spin was computed at whatever the calculator fell back to -- unless
    # it is an output geometry, which is relabelled at the spin asked for, and
    # is refused only for stating a different one: it is a stationary point of
    # that surface, not of this one.
    for frame, kind in zip(frames, _frame_kinds(len(frames))):
        if source == "output" and "spin" not in frame.info:
            continue
        found, wanted = _unpaired(frame), entry.spins[kind]
        if found != wanted:
            raise ManifestError(
                f"{entry.label}: the {kind} frame in {path} is at 2S={found}, not "
                f"the {wanted} asked for ({_SOURCES[source]})"
            )


def fit_reaction(manifest: Manifest, entry: Entry, calc_factory, fitted: dict):
    """Fit one reaction entry; the written paths."""
    reaction = reaction_module.parse(entry.mapped)
    frames, source = _supplied_frames(entry, reaction, calc_factory)
    rxn = reaction_module.parameterize(
        entry.mapped,
        calc_factory,
        config=manifest.config,
        workdir=str(manifest.workdir / "fragments"),
        seed=manifest.embed_seed,
        eps=manifest.eps,
        amplitude=entry.options.get("amplitude"),
        fragments=fitted,
        frames=frames,
        spins=entry.spins,
    )
    # A searched path, or output geometries just labelled, is what the next
    # run should find in the cache.
    if source in (None, "output"):
        cache = _file(entry.training, ".xyz")
        cache.parent.mkdir(parents=True, exist_ok=True)
        io.write_frames(cache, rxn.frames)

    rxn.write(str(entry.output))
    written = [
        _file(entry.output, suffix)
        for suffix in (".xyz", ".jsonl", "-reactant.jsonl", "-product.jsonl")
    ]
    return written, rxn


def run(manifest, calc_factory=None, log=print) -> list[Outcome]:
    """Fit every molecule, then every reaction, and write each where it says.

    `manifest` is a `Manifest` or a path to one.  `calc_factory` overrides the
    manifest's `calculator` block, for a reference method that is not one of
    the named backends.

    An entry that fails is recorded and the run goes on: one reaction with no
    gas-phase saddle should not cost the rest of the dataset.  The outcomes are
    returned in manifest order, and `ok` on each says which is which.
    """
    if not isinstance(manifest, Manifest):
        manifest = load(manifest)
    with use(manifest.params):
        return _run(manifest, calc_factory, log)


def _run(manifest, calc_factory, log) -> list[Outcome]:
    factory, method = calculator_factory(manifest.calculator)
    if calc_factory is not None:
        # `parameterize` names the method after the calculator class
        factory, method = calc_factory, None

    log(manifest.plan())
    outcomes: list[Outcome] = []
    fitted: dict = {}

    def attempt(entry, work):
        log(f"\n[{entry.kind} {entry.id}] {entry.smiles}")
        try:
            written, detail = work()
        except Exception as error:  # noqa: BLE001 -- recorded, see the docstring
            log(f"  FAILED: {type(error).__name__}: {error}")
            outcomes.append(Outcome(entry, False, detail=traceback.format_exc()))
            return None
        for path in written:
            log(f"  wrote {_relative(path, manifest.root)}")
        outcomes.append(Outcome(entry, True, written=written, detail=detail))
        return detail

    for entry in manifest.molecules:
        def work(entry=entry):
            written, (equilibrium, params) = fit_molecule(
                manifest, entry, factory, method
            )
            fitted[entry.key] = (equilibrium, params)
            return written, params.fit_report()

        report = attempt(entry, work)
        if report:
            log("  " + report.replace("\n", "\n  "))

    for entry in manifest.reactions:
        def work(entry=entry):
            written, rxn = fit_reaction(manifest, entry, factory, fitted)
            return written, rxn.report()

        report = attempt(entry, work)
        if report:
            log("  " + report.replace("\n", "\n  "))

    failed = [o for o in outcomes if not o.ok]
    log(f"\n{len(outcomes) - len(failed)} of {len(outcomes)} entries fitted")
    for outcome in failed:
        log(f"  failed: {outcome.entry.label}")
    return outcomes
