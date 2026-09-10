"""Shared helpers for the numbered examples -- not part of the package API.

Running any example as `python examples/0N_name.py` puts this directory on
`sys.path`, which is how the plain `from _common import ...` below resolves.
"""

from pathlib import Path

# Everything the examples write lands here, so the repo stays clean.
OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)


def banner(doc: str) -> None:
    """Print an example's docstring as its header, minus the usage line."""
    lines = [line for line in doc.strip().splitlines() if "uv run python" not in line]
    title = lines[0]
    print(f"\n{title}\n{'=' * len(title)}")
    print("\n".join(lines[1:]).strip())
    print()


def gfn2(atoms=None):
    """The reference calculator the examples fit against.

    GFN2-xTB is fast enough that a full parameterization takes seconds, which
    is the only reason these examples are runnable.  Swap it for a DFT code or
    an MLIP and nothing else in the pipeline changes.
    """
    from tblite.ase import TBLite

    return TBLite(method="GFN2-xTB", verbosity=0)


def fitted(smiles: str, config=None, name: str | None = None):
    """A fitted force field for `smiles`, cached through its training file.

    The first call runs the reference calculator; every later call re-fits from
    the extended XYZ alone, which is the README's headline claim doing real
    work -- the examples after this one never touch the reference method.

    Returns `(equilibrium atoms, parameters, training file path)`.
    """
    import fastforces as ff

    path = OUTPUT / f"{name or smiles}.xyz"
    if path.exists():
        params = ff.fit_from_file(str(path), config=config)
    else:
        params = ff.parameterize(
            ff.build(smiles), gfn2, config=config, training_set=str(path)
        )
    atoms = ff.io.read_training_set(str(path)).equilibrium.copy()
    return atoms, params, path
