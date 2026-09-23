"""q-force's XML imports into `.jsonl` rows, in the units q-force states them in."""

import json
from pathlib import Path

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


def test_the_import_writes_q_force_s_own_numbers(tmp_path):
    """No unit conversion: the file is already in the `.jsonl`'s nm and kJ/mol."""
    source = DATA / "h2o_qforce.xml"
    out = tmp_path / "h2o.jsonl"
    convert(source, out)
    rows = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert rows == read_xml(source)
    bond = next(r for r in rows if r["type"] == "bond")
    assert bond["kwargs"]["r0"] < 0.2  # nm, not Angstrom
