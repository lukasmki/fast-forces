"""The fitted force field: one container, three representations.

Internally the parameters are held as a `term_dict` in ASE units -- exactly what
`QForce`, `ACKS2` and `LennardJones` consume, so the calculator needs no
translation layer.  The two export formats are OpenMM-flavoured and are produced
through `export`, which owns every unit conversion.
"""

import json
from dataclasses import dataclass, field

import numpy as np

from .export import units as export_units

# Terms the bonded evaluator does not implement, and that the calculator handles
# itself.  `QForce.__call__` silently skips unknown keys, so these ride along in
# the same container without disturbing it.
NON_QFORCE_TERMS = ("atom", "lennardjones", "reference")


@dataclass
class Parameters:
    """A complete force field for one topology.

    `terms` is the `term_dict`: `{type: {"atoms": (n, slots) int array,
    "kwargs": {name: (n,) float array}}}`, in eV and Angstrom.
    """

    numbers: np.ndarray
    terms: dict[str, dict] = field(default_factory=dict)
    exclusions: np.ndarray | None = None
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

    def bonded_terms(self) -> dict:
        """Just the terms `QForce` evaluates."""
        return {k: v for k, v in self.terms.items() if k not in NON_QFORCE_TERMS}

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
        """The DynamicTopology representation: one row per term, OpenMM units."""
        rows = []
        for term in export_units.TERM_ORDER:
            block = self.terms.get(term)
            if block is None:
                continue
            atoms = np.asarray(block["atoms"])
            names = export_units.slot_names(term, atoms.shape[1])
            for i in range(len(atoms)):
                rows.append(
                    {
                        "type": term,
                        "atoms": {
                            n: int(a) for n, a in zip(names, atoms[i], strict=True)
                        },
                        "kwargs": {
                            key: float(export_units.to_openmm(term, key, value[i]))
                            for key, value in block["kwargs"].items()
                        },
                    }
                )
        return rows

    def to_jsonl(self, path: str) -> None:
        with open(path, "w") as handle:
            for row in self.to_rows():
                handle.write(json.dumps(row) + "\n")

    @classmethod
    def from_rows(cls, rows: list[dict], numbers: np.ndarray | None = None):
        """Invert `to_rows`, converting back to ASE units."""
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["type"], []).append(row)

        terms = {}
        for term, entries in grouped.items():
            atoms = np.array(
                [[int(v) for v in e["atoms"].values()] for e in entries], dtype=int
            )
            kwargs = {
                key: np.array(
                    [
                        export_units.from_openmm(term, key, e["kwargs"][key])
                        for e in entries
                    ],
                    dtype=float,
                )
                for key in entries[0]["kwargs"]
            }
            terms[term] = {"atoms": atoms, "kwargs": kwargs}

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
        with open(path) as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        return cls.from_rows(rows, numbers)

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
