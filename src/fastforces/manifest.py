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
file to regenerate it.  A change to the sampling half of `fit_config` --
`n_mode_frames`, `temperature` -- therefore does not reach a molecule until its
training set is deleted; a change to the fitting half does.

Molecules are fitted before reactions, and a reaction's fragments are taken
from the molecule fits wherever the manifest lists that molecule, so the water
in three proton transfers is fitted once.  A fragment it does not list is
fitted anyway, into `<workdir>/fragments`.

**`global_params` is checked, not applied.**  Those values are module constants
the evaluators and the OpenMM exporter read at import, and REFERENCE.md §7.2 is
explicit that a parameter set is valid only at the values it was fitted at.  A
manifest that states a value the package does not use is refused before any
fitting starts, rather than producing force fields that silently disagree with
the dataset they are filed under.  Values it leaves out take the defaults.

Everything is validated when the manifest is loaded -- unknown keys, duplicate
outputs, reaction SMILES that cannot be mapped -- because the alternative is a
typo found hours into a DFT run.
"""

import json
import traceback
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from . import io
from . import reaction as reaction_module
from .fit import FitConfig


class ManifestError(ValueError):
    """Raised for a manifest that cannot be run as written."""


# ---------------------------------------------------------------------------
# global parameters
# ---------------------------------------------------------------------------


def package_globals() -> dict:
    """The global parameters this package evaluates at, in REFERENCE.md units.

    Read from the modules rather than copied, so this cannot drift from what
    the evaluators actually use.  `switch_radius` and `switch_width` are stated
    in nm by the table and held in Angstrom by `lj`.
    """
    from .forcefield import ewald, exclusions, lj, qforce, zbl

    return {
        "bond_asymptote": qforce.BOND_ASYMPTOTE,
        "shape_decay": qforce.SHAPE_DECAY,
        "taper_radius": zbl.TAPER_RADIUS,
        "taper_width": zbl.TAPER_WIDTH,
        "switch_radius": lj.SWITCH_RADIUS / 10.0,
        "switch_width": lj.SWITCH_WIDTH / 10.0,
        "core_fraction": lj.CORE_FRACTION,
        "exclusion_depth": exclusions.EXCLUSION_DEPTH,
        "exclude_coulomb": exclusions.EXCLUDE_COULOMB,
        "gamma": ewald.GAMMA,
        "accuracy": ewald.ACCURACY,
        "ccoul": ewald.CCOUL,
        "zbl_ccoul": zbl.CCOUL,
    }


def check_globals(stated: dict) -> None:
    """Refuse a `global_params` block this package would not honour."""
    actual = package_globals()
    unknown = sorted(set(stated) - set(actual))
    if unknown:
        raise ManifestError(
            f"unknown global_params {unknown}; known: {sorted(actual)}"
        )
    wrong = []
    for key, value in stated.items():
        expected = actual[key]
        if isinstance(expected, bool):
            same = value is expected
        else:
            same = np.isclose(float(value), float(expected), rtol=1e-9, atol=0.0)
        if not same:
            wrong.append(f"{key}={value!r} (package: {expected!r})")
    if wrong:
        raise ManifestError(
            "global_params differ from the values this package evaluates at: "
            + ", ".join(wrong)
            + ". They are module constants, not fit settings -- a force field "
            "fitted here would not be valid at the manifest's values"
        )


# ---------------------------------------------------------------------------
# the reference calculator
# ---------------------------------------------------------------------------


def _charge(atoms) -> int:
    return int(round(float(np.sum(atoms.get_initial_charges()))))


def _unpaired(atoms) -> int:
    """`2S` for `atoms`: as stated on it, or the lowest the electron count allows.

    A molecule entry states it -- from its SMILES radicals, or explicitly -- and
    it is set on the atoms before they are fitted.  A reaction's combined
    geometry does not carry one, so it falls back to the parity of the electron
    count, which is right for every closed-shell channel.
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
            from tblite.ase import TBLite

            if atoms is None:
                return TBLite(method=method, **spec)
            return TBLite(
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
REACTION_KEYS = ("id", "smiles", "path", "frames", "amplitude")


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

    @property
    def root(self) -> Path:
        return self.path.parent

    def plan(self) -> str:
        """What `run` would do, one line per entry."""
        lines = [f"{self.name}: {self.path}", f"  training sets under {self.workdir}"]
        for entry in self.molecules + self.reactions:
            notes = []
            if _cached(entry):
                notes.append("training set present")
            if any(entry.output.parent.glob(entry.output.name + ".*")):
                notes.append("overwrites existing output")
            note = f" ({', '.join(notes)})" if notes else ""
            lines.append(f"  {entry.label} -> {_relative(entry.output, self.root)}{note}")
            if entry.mapped and entry.mapped != entry.smiles:
                lines.append(f"      mapped as {entry.mapped}")
        return "\n".join(lines)


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

    check_globals(raw.get("global_params", {}))

    molecules = _entries(raw.get("molecules", []), "molecule", MOLECULE_KEYS, root, workdir)
    reactions = _entries(raw.get("reactions", []), "reaction", REACTION_KEYS, root, workdir)

    outputs = [e.output for e in molecules + reactions]
    clashes = sorted({str(p) for p in outputs if outputs.count(p) > 1})
    if clashes:
        raise ManifestError(f"several entries write to the same path: {clashes}")

    for entry in molecules:
        entry.key = molecule_key(entry.smiles)
    for entry in reactions:
        try:
            entry.mapped = reaction_module.map_atoms(entry.smiles)
            reaction_module.parse(entry.mapped)
        except reaction_module.ReactionError as error:
            raise ManifestError(f"{entry.label}: {error}") from error

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
        params = reaction_module._lone_atom(atoms, calc_factory, frame=frame)
        equilibrium = frame
        meta = {
            "method": method or type(calc_factory(atoms)).__name__,
            "smiles": entry.smiles,
            "charge": _charge(atoms),
            "spin": atoms.info["spin"],
        }
    else:
        if training.exists():
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


def _supplied_frames(entry: Entry, reaction):
    """Stationary points given in the manifest or left by an earlier run."""
    from ase.io import read

    path = entry.options.get("frames") or _file(entry.training, ".xyz")
    if not Path(path).exists():
        if "frames" in entry.options:
            raise ManifestError(f"{entry.label}: frames file {path} does not exist")
        return None

    frames = read(str(path), index=":", format="extxyz")
    kind, _ = reaction.channel()
    if (kind == "fission" and not frames) or (kind != "fission" and len(frames) != 3):
        needs = "the reactant" if kind == "fission" else "reactant, TS and product"
        raise ManifestError(
            f"{entry.label}: {path} holds {len(frames)} frames; a {kind} channel "
            f"needs {needs}"
        )

    def pairs(connectivity) -> set:
        return {frozenset((int(i), int(j))) for i, j, *_ in connectivity}

    checks = [(frames[0], "reactant")]
    if len(frames) == 3:
        checks.append((frames[2], "product"))
    for frame, side in checks:
        if not np.array_equal(frame.get_atomic_numbers(), reaction.numbers):
            raise ManifestError(
                f"{entry.label}: the atoms in {path} are not in this reaction's "
                "order -- write the reaction atom-mapped to match the file"
            )
        stored = frame.info.get("connectivity")
        if stored is not None and pairs(stored) != reaction.bonds(side):
            raise ManifestError(
                f"{entry.label}: the {side} frame in {path} is bonded differently "
                f"from the mapped reaction {entry.mapped} -- write the reaction "
                "atom-mapped to match the file"
            )
        if frame.calc is None or "energy" not in frame.calc.results:
            raise ManifestError(f"{entry.label}: {path} carries no reference energies")
    return frames


def fit_reaction(manifest: Manifest, entry: Entry, calc_factory, fitted: dict):
    """Fit one reaction entry; the written paths."""
    from ase.io import write

    reaction = reaction_module.parse(entry.mapped)
    frames = _supplied_frames(entry, reaction)
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
    )
    if frames is None:
        cache = _file(entry.training, ".xyz")
        cache.parent.mkdir(parents=True, exist_ok=True)
        write(str(cache), rxn.frames, format="extxyz")

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
