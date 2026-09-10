"""The training set: one extended XYZ file holding everything the fit needs.

Per the README, "everything necessary to reproduce the fit is contained in one
file".  That is taken literally here: the frames, their reference energies and
forces, the connectivity, the reference method, and the fit configuration all
live in the file, and the Hessian is *reconstructed* from the displaced frames
rather than stored -- those frames are training data regardless, so storing the
matrix too would be a second copy that could disagree with them.
"""

import json
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from ase.io import read, write

# info keys written onto the first frame only; they describe the whole set.
META_KEYS = ("method", "smiles", "charge", "spin", "fit_config")


@dataclass
class TrainingSet:
    frames: list[Atoms]
    meta: dict = field(default_factory=dict)

    def of_kind(self, *kinds: str) -> list[Atoms]:
        return [f for f in self.frames if f.info.get("frame_kind") in kinds]

    @property
    def equilibrium(self) -> Atoms:
        frames = self.of_kind("equilibrium")
        if not frames:
            raise ValueError("training set has no equilibrium frame")
        return frames[0]

    @property
    def hessian(self) -> np.ndarray:
        return rebuild_hessian(self.equilibrium, self.of_kind("hessian"))

    def summary(self) -> str:
        kinds: dict[str, int] = {}
        for frame in self.frames:
            kind = frame.info.get("frame_kind", "unknown")
            kinds[kind] = kinds.get(kind, 0) + 1
        parts = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
        return f"{len(self.frames)} frames ({parts})"


def rebuild_hessian(equilibrium: Atoms, frames: list[Atoms]) -> np.ndarray:
    """Reassemble the Cartesian Hessian from the tagged +/-delta frames."""
    n = len(equilibrium)
    columns: dict[int, dict[int, np.ndarray]] = {}
    delta = None
    for frame in frames:
        column = frame.info.get("hessian_column")
        if column is None:
            continue
        columns.setdefault(int(column), {})[int(frame.info["hessian_sign"])] = (
            frame.get_forces()
        )
        delta = float(frame.info["hessian_delta"])

    if delta is None or len(columns) != 3 * n:
        raise ValueError(
            f"expected {3 * n} Hessian displacement frames, found {len(columns)}"
        )

    h = np.zeros((3 * n, 3 * n))
    for column, pair in columns.items():
        h[column] = -(pair[+1] - pair[-1]).ravel() / (2 * delta)
    return 0.5 * (h + h.T)


def write_training_set(
    path: str, frames: list[Atoms], meta: dict | None = None
) -> None:
    """Write every frame to one extended XYZ, metadata on the first frame."""
    if not frames:
        raise ValueError("refusing to write an empty training set")
    # The metadata is written onto the first frame in place rather than onto a
    # copy: `Atoms.copy()` drops the calculator, and the calculator is where the
    # extxyz writer reads `energy` and `forces` from.
    for key in META_KEYS:
        value = (meta or {}).get(key)
        if value is not None:
            frames[0].info[key] = (
                json.dumps(value) if isinstance(value, (dict, list)) else value
            )
    write(path, frames, format="extxyz")


def read_training_set(path: str) -> TrainingSet:
    """Read back what `write_training_set` wrote."""
    frames = read(path, index=":", format="extxyz")
    frames = frames if isinstance(frames, list) else [frames]
    meta = {}
    for key in META_KEYS:
        if key in frames[0].info:
            value = frames[0].info[key]
            if key == "fit_config" and isinstance(value, str):
                value = json.loads(value)
            meta[key] = value
    return TrainingSet(frames=frames, meta=meta)
