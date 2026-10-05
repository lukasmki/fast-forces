"""`fastforces.parallel`: the same fit, with the reference calculations farmed out.

Each check runs in a fresh interpreter.  Forking is only safe in a process
whose OpenMP runtimes have never started a thread pool, and a pytest session
has usually run tblite in-process by the time it gets here -- which is exactly
the hazard `parallel` exists to keep its own parent out of.
"""

import os
import subprocess
import sys
import textwrap

import pytest

FIT = textwrap.dedent(
    """
    import sys
    import fastforces as ff
    from tblite.ase import TBLite

    atoms = ff.build("OO")
    params = ff.parameterize(
        atoms,
        lambda a: TBLite(a, method="GFN2-xTB", verbosity=0),
        config=ff.FitConfig(n_mode_frames=12, n_conformers=0),
        training_set=sys.argv[1] + ".xyz",
    )
    params.to_jsonl(sys.argv[1] + ".jsonl")
    """
)


def _run(code: str, *args, **env) -> subprocess.CompletedProcess:
    environment = {**os.environ, "OMP_NUM_THREADS": "1", **env}
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )


@pytest.mark.slow
def test_a_parallel_fit_is_the_serial_fit(tmp_path):
    """Byte for byte, at one thread per calculation.

    tblite reduces in parallel, and past one thread is not even reproducible
    run to run, so the comparison is at one thread on both sides -- which is
    also what `_SPREAD = False` holds the workers' lone relaxations to.
    """
    pytest.importorskip("tblite.ase")
    serial, farmed = tmp_path / "serial", tmp_path / "parallel"
    _run(FIT, str(serial))
    _run(
        "import fastforces.parallel as p; p._SPREAD = False\n" + FIT,
        str(farmed),
        FASTFORCES_WORKERS="4",
        FASTFORCES_THREADS="1",
    )
    for suffix in (".xyz", ".jsonl"):
        assert (serial.with_suffix(suffix)).read_bytes() == (
            farmed.with_suffix(suffix)
        ).read_bytes(), suffix


def test_map_runs_in_workers_and_keeps_the_order():
    code = textwrap.dedent(
        """
        import os
        from fastforces import parallel

        parallel.configure(workers=4, threads=1)
        parent = os.getpid()
        out = parallel.map(lambda x: (x * x, os.getpid()), range(9))
        assert [value for value, _ in out] == [x * x for x in range(9)]
        assert all(pid != parent for _, pid in out)
        assert parallel.call(lambda: os.getpid()) != parent

        # a worker runs what it is handed in itself, rather than forking again
        inner = parallel.map(
            lambda _: parallel.map(lambda _: os.getpid(), [0, 0]), [0, 1]
        )
        assert all(len(set(pids)) == 1 for pids in inner)

        # a nested batch divides the pool, and its items fork their own
        nested = parallel.map(
            lambda _: (os.getpid(), parallel.map(lambda _: os.getpid(), [0, 0])),
            [0, 1],
            nested=True,
        )
        assert all(pid not in pids for pid, pids in nested)
        """
    )
    _run(code)


def test_an_exception_in_a_worker_is_raised_in_the_parent():
    code = textwrap.dedent(
        """
        from fastforces import parallel

        parallel.configure(workers=2, threads=1)

        def fail(x):
            if x == 1:
                raise ValueError("frame 1")
            return x

        try:
            parallel.map(fail, [0, 1, 2])
        except ValueError as error:
            assert str(error) == "frame 1"
        else:
            raise AssertionError("no exception")
        """
    )
    _run(code)


def test_one_worker_is_serial():
    code = textwrap.dedent(
        """
        import os
        from fastforces import parallel

        assert not parallel.active()
        assert parallel.map(lambda _: os.getpid(), [0, 1]) == [os.getpid()] * 2
        """
    )
    _run(code)
