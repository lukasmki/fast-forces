# Examples

Two kinds of thing live here.

**Reference data** — hand-written files showing the two export formats, used as
regression fixtures by the test suite:

| file | what it is |
|---|---|
| `h2o2.xyz` | a labelled frame in the extended-XYZ training format |
| `h2o2_dynamictopology_format.jsonl` | the same force field, one JSON row per term |
| `h2o2_openmm_format.xml` | the same force field as a serialized OpenMM system |

**Numbered scripts** — each one standalone and runnable:

```bash
uv run python examples/01_quickstart.py
```

| script | shows | runtime |
|---|---|---|
| `01_quickstart.py` | SMILES to a simulation-ready force field, the README example | 30 s |
| `02_topology.py` | bond-graph perception, symmetry typing, term enumeration | 1 s |
| `03_training_set.py` | the single training file, and re-fitting from it alone | 2 s |
| `04_export_formats.py` | jsonl and OpenMM XML, with an OpenMM cross-check | 25 s |
| `05_any_calculator.py` | the same molecule fit against three different references | 5 s |
| `06_energy_decomposition.py` | what each of the four evaluators contributes, and charges | 25 s |
| `07_molecular_dynamics.py` | NVE and Langevin dynamics, and the speedup over the reference | 45 s |
| `08_validation.py` | scoring the fit on geometries it never saw | 35 s |
| `09_starting_point.py` | fitting from an existing force field, and why refitting is idempotent | 20 s |

They are meant to be read in order — 02 and 06 explain results that look
surprising in the earlier ones — but each runs on its own.

Everything the scripts write goes to `examples/output/`. Scripts 04, 06, 07, 08 and 09 reuse the training files cached there, re-fitting from them rather than
calling the reference calculator again. That saves little here — GFN2-xTB is
cheap enough that the regression, not the reference method, is most of the
runtime — but it is what makes the pattern worth having when the reference is a
DFT code. Delete `examples/output/` to start over; the whole set takes about
three minutes from cold.

`04_export_formats.py` needs the optional OpenMM dependency:

```bash
uv sync --extra openmm
```

The other scripts need only what `pyproject.toml` already installs. They fit
against GFN2-xTB via `tblite` because it is fast enough to make the examples
runnable; nothing in the pipeline is specific to it, which is the point of
`05_any_calculator.py`.
