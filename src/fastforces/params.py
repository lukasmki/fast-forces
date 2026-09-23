"""The fitted force field: one container, three representations.

Internally the parameters are held as a `term_dict` in ASE units (eV,
Angstrom) -- the units the fit works in, because its residuals come from an ASE
calculator's own energies and forces, and the units every DynamicTopology
evaluator reads.  Nothing in the evaluation path converts anything.

The force field itself is DynamicTopology's.  This package fits parameters for
it and evaluates them through it -- `DynamicTopology.forcefield.evaluate`, the
single-topology sum `System` puts on a diabat -- so a template scores here
exactly as it will in a simulation.  `to_terms` is the bridge: the same
parameters as the term list DynamicTopology holds in memory.

The two file formats are the `.jsonl` DynamicTopology loads, converted to nm and
kJ/mol by `DynamicTopology.io.units` (the one table of what converts how), and
an OpenMM System (`export.openmm`).
"""

from dataclasses import dataclass, field

import numpy as np
from DynamicTopology.io.json import read_jsonl, write_jsonl
from DynamicTopology.io.units import term_from_disk, term_to_disk

# The two electrostatic terms, which are alternatives rather than additions:
# `atom` is the ACKS2 per-atom block whose charges are re-solved at every
# geometry, `charge` the fixed per-atom charges.  Both sum the same kernel over
# the same pairs, so a field carrying both would count electrostatics twice.
# Each names the value of DynamicTopology's `global_params.electrostatics`
# that evaluates it.
ELECTROSTATIC_TERMS: dict[str, str] = {"atom": "acks2", "charge": "pointcharge"}

# The order terms are written in, and the slot names each uses for its atom
# indices -- per-atom terms `p0`, bonded terms `p1..pN`, `reference` `a1`, as
# the DynamicTopology datasets carry them.
TERM_ORDER: tuple[str, ...] = (
    "atom",
    "charge",
    "lennardjones",
    "bond",
    "angle",
    "bondbond",
    "bondangle",
    "angleangle",
    "dihedralangle",
    "dihedralbond",
    "dihedralangleangle",
    "periodicdihedral",
    "reference",
)
SLOT_PREFIX: dict[str, str] = {"reference": "a"}
SLOT_START: dict[str, int] = {
    "atom": 0,
    "charge": 0,
    "lennardjones": 0,
    "reference": 1,
}


def slot_names(term: str, n_slots: int) -> list[str]:
    prefix = SLOT_PREFIX.get(term, "p")
    start = SLOT_START.get(term, 1)
    return [f"{prefix}{start + i}" for i in range(n_slots)]


def block_to_terms(term: str, block: dict) -> list[dict]:
    """One `term_dict` block as a list of DynamicTopology terms, in ASE units."""
    atoms = np.asarray(block["atoms"])
    names = slot_names(term, atoms.shape[1])
    return [
        {
            "type": term,
            "atoms": {n: int(a) for n, a in zip(names, atoms[i], strict=True)},
            "kwargs": {key: float(value[i]) for key, value in block["kwargs"].items()},
        }
        for i in range(len(atoms))
    ]


def terms_to_blocks(terms: list[dict]) -> dict[str, dict]:
    """Invert `block_to_terms` over a whole term list."""
    grouped: dict[str, list[dict]] = {}
    for term in terms:
        grouped.setdefault(term["type"], []).append(term)
    blocks = {}
    for kind, entries in grouped.items():
        atoms = np.array(
            [[int(v) for v in e["atoms"].values()] for e in entries], dtype=int
        )
        kwargs = {
            key: np.array([e["kwargs"][key] for e in entries], dtype=float)
            for key in entries[0]["kwargs"]
        }
        blocks[kind] = {"atoms": atoms, "kwargs": kwargs}
    return blocks


@dataclass
class Parameters:
    """A complete force field for one topology.

    `terms` is the `term_dict`: `{type: {"atoms": (n, slots) int array,
    "kwargs": {name: (n,) float array}}}`, in eV and Angstrom.

    The 1-2/1-3/1-4 exclusions are not stored.  DynamicTopology derives them
    from the `bond` terms whenever it evaluates a term list -- `to_terms` hands
    it exactly that -- so a field excludes what its bonds say, and an edited
    bond cannot leave a stale exclusion behind.
    """

    numbers: np.ndarray
    terms: dict[str, dict] = field(default_factory=dict)
    report: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # access
    # ------------------------------------------------------------------

    def to_term_dict(self) -> dict:
        return self.terms

    @property
    def e0(self) -> float:
        """The single reference energy offset, in eV."""
        reference = self.terms.get("reference")
        return 0.0 if reference is None else float(reference["kwargs"]["E0"].sum())

    def electrostatics(self) -> str | None:
        """Which electrostatic term this field carries, or `None` for neither.

        `atom` means ACKS2, `charge` means fixed charges.  They are mutually
        exclusive -- see `ELECTROSTATIC_TERMS` -- and every consumer asks here
        rather than testing for a term name itself, so the exclusion is checked
        once instead of being assumed in four places.
        """
        present = [t for t in ELECTROSTATIC_TERMS if t in self.terms]
        if len(present) > 1:
            raise ValueError(
                f"a force field carries one electrostatic term, not {present}: "
                "`atom` (ACKS2) and `charge` (fixed charges) sum the same "
                "kernel over the same pairs, so keeping both double counts it"
            )
        return present[0] if present else None

    def global_electrostatics(self) -> str:
        """The `global_params.electrostatics` this field is evaluated under.

        DynamicTopology's `evaluate` picks the same one from the block present
        (`evaluate.surface`), so this is for the checks that two fields, or a
        field and a manifest, agree -- not something an evaluation has to set.
        """
        return ELECTROSTATIC_TERMS.get(self.electrostatics(), "acks2")

    def to_terms(self) -> list[dict]:
        """The term list DynamicTopology holds in memory: ASE units, `TERM_ORDER`."""
        out: list[dict] = []
        for term in TERM_ORDER:
            if term in self.terms:
                out.extend(block_to_terms(term, self.terms[term]))
        unknown = sorted(set(self.terms) - set(TERM_ORDER))
        if unknown:
            raise KeyError(f"no DynamicTopology term type {unknown}")
        return out

    def fit_report(self) -> str:
        if not self.report:
            return "no fit diagnostics recorded"
        lines = []
        for key, value in self.report.items():
            if isinstance(value, dict):
                inner = "  ".join(f"{k}={v:.5g}" for k, v in value.items())
                lines.append(f"{key:14s} {inner}")
            elif isinstance(value, float):
                lines.append(f"{key:14s} {value:.5g}")
            else:
                lines.append(f"{key:14s} {value}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        counts = ", ".join(f"{t}={len(v['atoms'])}" for t, v in self.terms.items())
        return f"Parameters({len(self.numbers)} atoms; {counts})"

    # ------------------------------------------------------------------
    # DynamicTopology jsonl
    # ------------------------------------------------------------------

    def to_rows(self) -> list[dict]:
        """The `.jsonl` representation: one row per term, nm and kJ/mol."""
        return [term_to_disk(term) for term in self.to_terms()]

    def to_jsonl(self, path: str) -> None:
        write_jsonl(path, self.to_terms(), exist_ok=True)

    @classmethod
    def from_rows(cls, rows: list[dict], numbers: np.ndarray | None = None):
        """Invert `to_rows`, converting back to ASE units."""
        return cls.from_terms([term_from_disk(row) for row in rows], numbers)

    @classmethod
    def from_terms(cls, terms: list[dict], numbers: np.ndarray | None = None):
        """Invert `to_terms`.  Without `numbers` every atom reads as element 0."""
        terms = terms_to_blocks(terms)
        if numbers is None:
            n_atoms = (
                max(
                    (
                        int(np.max(b["atoms"]))
                        for b in terms.values()
                        if len(b["atoms"])
                    ),
                    default=-1,
                )
                + 1
            )
            numbers = np.zeros(n_atoms, dtype=int)
        return cls(numbers=np.asarray(numbers), terms=terms)

    @classmethod
    def from_jsonl(cls, path: str, numbers: np.ndarray | None = None):
        return cls.from_terms(read_jsonl(path), numbers)

    # ------------------------------------------------------------------
    # OpenMM
    # ------------------------------------------------------------------

    def to_openmm_system(self, positions=None, box: float = 20.0):
        """An `openmm.System`.

        `positions` (Angstrom) fixes the geometry the ACKS2 charges are frozen
        at; without it the exported system has no electrostatics at all.  See
        `export.openmm` for why that trade-off exists.
        """
        from .export.openmm import build_system

        return build_system(self, positions=positions, box=box)

    def to_openmm_xml(self, path: str, positions=None, box: float = 20.0) -> None:
        from .export.openmm import write_xml

        write_xml(self, path, positions=positions, box=box)
