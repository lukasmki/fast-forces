"""extxyz round trips of the frames fast-forces reads back and writes again."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from fastforces import io


def _fission_product(energy=-2056.08):
    """Two atoms and no bond: the frame whose empty connectivity ASE mangles."""
    frame = Atoms("OH", positions=[[0, 0, 0], [0, 0, 3.0]])
    frame.info.update(spin=1, connectivity=[], frame_kind="product")
    frame.calc = SinglePointCalculator(frame, energy=energy, forces=np.zeros((2, 3)))
    return frame


def test_an_empty_connectivity_survives_being_read_and_written_back(tmp_path):
    """Twice through, which is what broke: the first read turns `[]` into an
    empty array, and plain ASE writes that as a bare `connectivity=`."""
    path = tmp_path / "rxn.xyz"
    io.write_frames(path, [_fission_product()])
    for _ in range(3):
        frames = io.read_frames(path)
        io.write_frames(path, frames)
    frame = io.read_frames(path)[0]
    assert frame.info["connectivity"] == []
    assert frame.info["frame_kind"] == "product"
    assert frame.get_potential_energy() == pytest.approx(-2056.08)


@pytest.mark.parametrize(
    "tail, check",
    [
        # the two headers HCombustion's rxn_06 and rxn_07 were written with
        ('pbc="F F F"', lambda f: not any(f.pbc)),
        (
            'frame_kind=product energy=-2056.08 pbc="F F F"',
            lambda f: f.info["frame_kind"] == "product"
            and f.get_potential_energy() == pytest.approx(-2056.08),
        ),
        (
            'energy=-2056.08 pbc="F F F"',
            lambda f: f.get_potential_energy() == pytest.approx(-2056.08),
        ),
    ],
)
def test_a_file_plain_ase_already_mangled_reads_back_whole(tmp_path, tail, check):
    """Whichever key the bare `connectivity=` swallowed is put back."""
    path = tmp_path / "mangled.xyz"
    path.write_text(
        "2\n"
        f"Properties=species:S:1:pos:R:3 spin=1 connectivity= {tail}\n"
        "O 0.0 0.0 0.0\n"
        "H 0.0 0.0 3.0\n"
    )
    repaired = io.read_frames(path)[0]
    assert repaired.info["connectivity"] == []
    assert repaired.info["spin"] == 1
    assert check(repaired)


def test_bonds_come_back_as_plain_lists(tmp_path):
    frame = _fission_product()
    frame.info["connectivity"] = [[0, 1, 1.0]]
    path = tmp_path / "bonded.xyz"
    io.write_frames(path, [frame])
    io.write_frames(path, io.read_frames(path))
    assert io.read_frames(path)[0].info["connectivity"] == [[0, 1, 1.0]]
