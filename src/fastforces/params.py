"""The fitted force field: one container, three representations.

Internally the parameters are held as a `term_dict` in ASE units (eV,
Angstrom) -- the units the fit works in, because its residuals come from an ASE
calculator's own energies and forces.

All four evaluators want them that way, so `terms` is handed to each of them as
it stands and nothing in the evaluation path converts anything.  The two export
formats are the only OpenMM-flavoured representations, and `export.units` is the
only place that conversion is written down, so the calculator and the two
exporters cannot drift apart.
"""

import json
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .export import units as export_units

# Terms the bonded evaluator does not implement, and that the calculator handles
# itself.  `QForce.__call__` silently skips unknown keys, so these ride along in
# the same container without disturbing it.
NON_QFORCE_TERMS = ("atom", "coulomb", "lennardjones", "reference")

# The two electrostatic terms, which are alternatives rather than additions:
# `atom` is the ACKS2 per-atom block whose charges are re-solved at every
# geometry, `coulomb` the fixed per-atom charges.  Both sum the same kernel over
# the same pairs, so a field carrying both would count electrostatics twice.
ELECTROSTATIC_TERMS = ("atom", "coulomb")



def _exclusion_mask(terms: dict, n_atoms: int) -> np.ndarray:
    """The 1-2/1-3/1-4 mask implied by a term dict's `bond` block.

    Derived rather than stored.  A `.jsonl` states the bond graph once, in its
    bond terms, and a mask written alongside it could go stale against an edited
    radius or a removed bond -- which would show up only as a field no longer
    reproducing its own energy.  The graph here is built the same way
    `Topology.from_terms` builds it, and the depth is `forcefield.exclusions`'s,
    so a loaded field excludes exactly what the field it was written from did.
    """
    from .forcefield.exclusions import EXCLUSION_DEPTH
    from .topology import exclusion_mask

    graph = nx.Graph()
    graph.add_nodes_from(range(n_atoms))
    bonds = terms.get("bond")
    if bonds is not None:
        graph.add_edges_from(
            (int(a), int(b)) for a, b in np.asarray(bonds["atoms"])[:, :2]
        )
    return exclusion_mask(graph, n_atoms, EXCLUSION_DEPTH)


@dataclass
class Parameters:
    """A complete force field for one topology.

    `terms` is the `term_dict`: `{type: {"atoms": (n, slots) int array,
    "kwargs": {name: (n,) float array}}}`, in eV and Angstrom.
    """

    numbers: np.ndarray
    terms: dict[str, dict] = field(default_factory=dict)
    # The 1-2/1-3/1-4 mask from `Topology`, and the only topology the three
    # whole-system pair sums ever see.  `forcefield/exclusions.py` takes every
    # pair it marks back off all three -- `-u_ZBL` and `-u_126` additively, the
    # Coulomb contraction through a screen -- and `export.openmm` writes it into
    # the exported system as real OpenMM exclusions.
    #
    # `None` means "no exclusions", which is a force field that charges every
    # bonded pair two nonbonded terms and will not reproduce its own reference
    # energy.  Both routes in set it: `fit` from the topology it fit against,
    # `from_rows` from the `bond` terms it just read.
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

    def electrostatics(self) -> str | None:
        """Which electrostatic term this field carries, or `None` for neither.

        `atom` means ACKS2, `coulomb` means fixed charges.  They are mutually
        exclusive -- see `ELECTROSTATIC_TERMS` -- and every consumer asks here
        rather than testing for a term name itself, so the exclusion is checked
        once instead of being assumed in four places.
        """
        present = [t for t in ELECTROSTATIC_TERMS if t in self.terms]
        if len(present) > 1:
            raise ValueError(
                f"a force field carries one electrostatic term, not {present}: "
                "`atom` (ACKS2) and `coulomb` (fixed charges) sum the same "
                "kernel over the same pairs, so keeping both double counts it"
            )
        return present[0] if present else None

    def bonded_terms(self) -> dict:
        """Just the terms `QForce` evaluates, in ASE units.

        `reference` is excluded even though `QForce.compute_reference` would
        evaluate it: `E0` is added once by the calculator, and letting `QForce`
        pick it up as well would count it twice.
        """
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
        numbers = np.asarray(numbers)
        return cls(
            numbers=numbers,
            terms=terms,
            exclusions=_exclusion_mask(terms, len(numbers)),
        )

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
