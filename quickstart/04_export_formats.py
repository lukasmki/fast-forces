"""04 -- Exporting: DynamicTopology jsonl and OpenMM XML.

The same parameters go out in two formats: the jsonl in the ASE units
(Angstrom, eV) the runtime works in, and OpenMM in its own (nm, kJ/mol), through
DynamicTopology's single conversion table.  The jsonl round-trips exactly;
the OpenMM system is checked by evaluating it with OpenMM's own engine and
comparing against `FastForces`.

    uv run python quickstart/04_export_formats.py
"""

import json

import numpy as np

import fastforces as ff

from _common import OUTPUT, banner, fitted

banner(__doc__)

atoms, params, _ = fitted("CC#N", name="acetonitrile")

# ---------------------------------------------------------------- jsonl
jsonl = OUTPUT / "acetonitrile.jsonl"
params.to_jsonl(str(jsonl))
rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
kinds: dict[str, int] = {}
for row in rows:
    kinds[row["type"]] = kinds.get(row["type"], 0) + 1
print(f"{jsonl.name}: {len(rows)} rows -- one per term")
print("  " + ", ".join(f"{k}={n}" for k, n in kinds.items()))
print(f"  example: {json.dumps(rows[0])[:96]}...")

reloaded = ff.Parameters.from_jsonl(str(jsonl), numbers=params.numbers)
delta = max(
    abs(params.terms[t]["kwargs"][k] - reloaded.terms[t]["kwargs"][k]).max()
    for t in params.terms
    for k in params.terms[t]["kwargs"]
)
print(f"  round trip: max parameter difference {delta:.2e}\n")

# `bond r0` is the clearest place to see the units: Angstrom inside and in the
# jsonl, nanometres only on the way out to OpenMM.
r0_internal = params.terms["bond"]["kwargs"]["r0"]
r0_jsonl = [row["kwargs"]["r0"] for row in rows if row["type"] == "bond"]
print(f"bond r0 internal (A): {np.round(r0_internal, 4)}")
print(f"bond r0 jsonl    (A): {np.round(r0_jsonl, 4)}\n")

# ---------------------------------------------------------------- OpenMM
try:
    from openmm import XmlSerializer

    from fastforces.export.openmm import system_energy
except ImportError:
    print("openmm is not installed -- skipping the cross-check.")
    print("  uv sync --extra openmm")
    raise SystemExit

xml = OUTPUT / "acetonitrile.xml"
positions = atoms.get_positions()
params.to_openmm_xml(str(xml), positions=positions)

system = XmlSerializer.deserialize(xml.read_text())
openmm_energy = system_energy(system, positions)
reference_energy = ff.evaluate(atoms, params)[0]

print(
    f"{xml.name}: {system.getNumForces()} forces, {system.getNumParticles()} particles"
)
print(f"  FastForces : {reference_energy:.8f} eV")
print(f"  OpenMM     : {openmm_energy:.8f} eV")
print(f"  difference : {abs(openmm_energy - reference_energy):.2e} eV")

print("""
One limitation to know about: ACKS2 re-solves its charges at every geometry,
and OpenMM has no charge-equilibration force.  The exported system therefore
freezes the charges solved at the geometry passed to `to_openmm_xml`, which is
right at and near that structure and drifts as the molecule distorts.""")
