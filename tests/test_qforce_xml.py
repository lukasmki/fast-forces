"""q-force's XML imports into `.jsonl` rows, converted to eV and Angstrom."""

import json
from pathlib import Path

from DynamicTopology.io.json import read_jsonl

from fastforces.qforce_xml import convert, read_xml

DATA = Path(__file__).parent / "data"


def test_every_qforce_file_parses():
    files = sorted(DATA.glob("*_qforce.xml"))
    assert files
    for path in files:
        terms = read_xml(path)
        assert any(t["type"] == "bond" for t in terms), path.name
        # A <Particle> has no atom attributes; its index is its position.
        per_atom = [t for t in terms if t["type"] == "lennardjones"]
        assert [t["atoms"] for t in per_atom] == [
            {"p0": i} for i in range(len(per_atom))
        ]


def test_the_import_writes_ev_and_angstrom(tmp_path):
    """q-force states nm and kJ/mol; the `.jsonl` carries eV and Angstrom."""
    source = DATA / "h2o_qforce.xml"
    out = tmp_path / "h2o.jsonl"
    convert(source, out)
    rows = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert rows == read_xml(source)
    bond = next(r for r in rows if r["type"] == "bond")
    assert 0.9 < bond["kwargs"]["r0"] < 1.0  # O-H in Angstrom, not nm
    # And the file loads: DynamicTopology refuses a bond `r0` still in nm.
    assert read_jsonl(out) == rows
