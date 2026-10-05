"""Running the reference calculations side by side on one node.

Everything a fit spends its time on, with a DFT reference, is independent single
points: the `6N` Hessian displacements, the thermal and conformer frames, every
point of every torsion scan, the fragments and the free atoms.  None reads
another's result, so they can all run at once -- and on a CPU node they have to,
because a reference code does not scale across a node by itself.  Measured with
PySCF B3LYP/6-31G* on PABA (17 atoms) on a 128-core Milan node, one single point
(energy and forces) takes 18.6 s on 4 threads and 6.87 s on 32, and nothing
more threads can buy (see `SPREAD_CAP`) -- so 32 frames four threads apiece take
18.6 s, where one after another on 32 threads they take 220.

    from fastforces import parallel
    parallel.configure(workers=32, threads=4)

or `FASTFORCES_WORKERS` / `FASTFORCES_THREADS` in the environment, which is read
when `fastforces` is imported.  With one worker -- the default -- nothing here
runs and every call is the serial one it always was.

**The results are the serial results, bit for bit, at the same thread count.**
Every frame is labelled by a fresh calculator built from `calc_factory`,
exactly as `sampling.label` always did, so a frame cannot see which process its
neighbour ran in; the only thing that changes is the wall clock.  What can move
the last digits is the thread count itself -- tblite and PySCF both reduce in
parallel -- and a batch smaller than the pool, or a lone relaxation, is given
the idle cores' threads.  That is the difference a serial run sees under a
different `OMP_NUM_THREADS`: about 1e-15 relative under tblite (which past one
thread is not even reproducible run to run), and under PySCF up to the SCF
tolerance -- 1e-5 eV at the `conv_tol = 1e-6` the PySCF calculator used to
run at, less at its 1e-9 now -- since the SCF path itself moves.  At
one thread per calculation a parallel fit was byte-identical to the serial one,
training file and `.jsonl` both, on H2O2 and PABA.  Where the serial code consumed frames in
order until it had enough (`normal_mode_frames` drops the ones whose SCF fails
and draws again), the parallel one labels a batch and keeps the same frames in
the same order.

**Why `fork`, and why the parent never runs the reference itself.**  The
factories a fit is handed are closures and lambdas, which `spawn` cannot send to
another process, so the workers are forked and find the task already in their
memory.  But both reference codes are OpenMP programs, and GNU libgomp is not
fork-safe: a parent that has run one parallel region keeps a pool of parked
threads, the forked child inherits the pool's bookkeeping but not its threads,
and its first parallel region waits for them forever.  Measured: PySCF and
tblite both hang in the child that way.  So while workers are configured, the
parent holds its own OpenMP runtimes to one thread -- one thread never starts a
pool -- and hands every calculation, the serial chains (a relaxation, a Sella
search) included, to a child that raises its own thread count after the fork.
A chain is one task with as many of the node's threads as help it
(`SPREAD_CAP`), since it has nothing to run beside it.

This is configured before the first reference calculation in the process, which
is why the environment variables are read at import.  A process that has already
run a multi-threaded OpenMP region cannot fork safely, and `configure` cannot
undo that.
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterable

# Workers and the OpenMP/BLAS threads each one runs.  One worker means serial.
_WORKERS: int = 1
_THREADS: int = 1
# True inside a worker: whatever it calls runs in that process, never forks again.
_IN_WORKER: bool = False
# The batch the next fork carries: `(function, items)`.  Read by the children
# out of their copy of this module, which is what lets the function be a closure.
_TASK: tuple[Callable, list] | None = None
# Whether a batch smaller than the pool, or a lone chain, gets the idle cores'
# threads.  Off, every calculation runs `_THREADS` threads, which is what a test
# comparing against a serial run at that thread count needs: the reference codes
# sum in parallel, so their last bits move with the thread count.
_SPREAD: bool = True
# The most threads one calculation is given when spreading.  Past it PySCF gets
# *slower*, not merely no faster -- B3LYP/6-31G* single points (energy and
# forces) on a 128-core Milan node, seconds:
#
#     threads          4      8     16     32     64    128
#     H2O2  (4)      0.65   0.61   0.55   0.56   1.04   2.83
#     EtOH  (9)      1.79   1.49   1.30   1.22   1.59   6.01
#     PABA (17)     18.6   11.4    7.83   6.87   8.47  22.4
#
# Handing a lone relaxation every thread on the node made H2O2's 3.6x slower
# than the serial fit's at 32.
SPREAD_CAP: int = 32

_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _loaded_libraries(*fragments: str) -> list[str]:
    """Paths of every shared library mapped into this process matching a fragment."""
    paths: list[str] = []
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                path = line.rsplit(None, 1)[-1]
                if path.startswith("/") and any(f in path for f in fragments):
                    if path not in paths:
                        paths.append(path)
    except OSError:
        pass
    return paths


def set_threads(n: int) -> None:
    """Hold every OpenMP and BLAS runtime in this process to `n` threads.

    A reference code reads `OMP_NUM_THREADS` when its runtime is loaded, so the
    variable alone does nothing for one already imported -- and the two the fit
    uses bring separate copies (PySCF vendors its own libgomp and OpenBLAS,
    tblite links the system libgomp, numpy has its own OpenBLAS).  So the
    variables are set for whatever loads next, and each runtime already mapped
    in is told directly.
    """
    n = max(int(n), 1)
    for name in _THREAD_VARIABLES:
        os.environ[name] = str(n)
    for path in _loaded_libraries("libgomp", "libiomp", "libomp."):
        try:
            ctypes.CDLL(path).omp_set_num_threads(n)
        except (OSError, AttributeError):
            pass
    for path in _loaded_libraries("openblas"):
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for symbol in (
            "openblas_set_num_threads",
            "openblas_set_num_threads64_",
            "scipy_openblas_set_num_threads64_",
            "scipy_openblas_set_num_threads",
        ):
            setter = getattr(lib, symbol, None)
            if setter is not None:
                setter(n)
                break


def configure(workers: int | None = None, threads: int | None = None) -> None:
    """Run reference calculations on `workers` processes of `threads` threads each.

    `None` leaves a setting as it is.  `workers=1` is serial and restores
    nothing: the parent's OpenMP runtimes stay where an earlier call put them.
    """
    global _WORKERS, _THREADS
    if workers is not None:
        _WORKERS = max(int(workers), 1)
    if threads is not None:
        _THREADS = max(int(threads), 1)
    if active():
        # The parent only orchestrates from here on; see the module docstring.
        set_threads(1)


def configure_from_environment() -> None:
    """`configure` from `FASTFORCES_WORKERS` and `FASTFORCES_THREADS`, if set."""
    workers = os.environ.get("FASTFORCES_WORKERS")
    threads = os.environ.get("FASTFORCES_THREADS")
    if workers or threads:
        configure(
            workers=int(workers) if workers else None,
            threads=int(threads) if threads else None,
        )


def active() -> bool:
    """Is a batch going to be farmed out, rather than run here?"""
    return _WORKERS > 1 and not _IN_WORKER


def workers() -> int:
    return _WORKERS if active() else 1


def threads() -> int:
    return _THREADS


def _start(workers: int, threads: int) -> None:
    """A fresh worker: either a leaf, or the parent of a smaller pool.

    A leaf (`workers == 1`) runs whatever it is handed in its own process at
    `threads`.  A nested worker -- one molecule of a manifest, say -- is handed
    a share of the node and orchestrates it exactly as the top-level parent
    does: one OpenMP thread of its own, everything farmed out again.
    """
    global _IN_WORKER, _WORKERS, _THREADS
    _WORKERS, _THREADS = workers, threads
    _IN_WORKER = workers <= 1
    set_threads(threads if _IN_WORKER else 1)


def _run(index: int):
    function, items = _TASK  # type: ignore[misc]
    return function(items[index])


def _fork(function: Callable, items: list, processes: int, initargs: tuple) -> list:
    """`function` over `items` on `processes` forked workers, in order.

    `concurrent.futures` rather than `multiprocessing.Pool` because a `Pool`'s
    workers are daemonic and may not fork in turn, which a nested map needs.
    Under `fork` every worker is started at the first submission, so each one
    finds `_TASK` in its copy of this module.
    """
    global _TASK
    _TASK = (function, items)
    try:
        with ProcessPoolExecutor(
            processes,
            mp_context=mp.get_context("fork"),
            initializer=_start,
            initargs=initargs,
        ) as pool:
            return list(pool.map(_run, range(len(items))))
    finally:
        _TASK = None


def _share(processes: int) -> int:
    """The threads each of `processes` workers gets out of the node."""
    if not _SPREAD:
        return _THREADS
    return max(_THREADS, min((_WORKERS * _THREADS) // processes, SPREAD_CAP))


def map(function: Callable[[Any], Any], items: Iterable, nested: bool = False) -> list:
    """`[function(x) for x in items]`, one worker per item, results in order.

    Serial when workers are not configured, inside a worker, or for a single
    item -- a lone item gets `call`'s thread count instead.  An exception raised
    for any item is raised here, as the serial loop would.

    `nested` is for items that are themselves whole fits, each farming out
    batches of its own: the node is divided between them, and each item's
    worker runs its share as a pool rather than as threads.
    """
    items = list(items)
    if not active():
        return [function(x) for x in items]
    if not items:
        return []
    if len(items) == 1:
        return [call(function, items[0])] if not nested else [function(items[0])]
    processes = min(_WORKERS, len(items))
    if nested:
        workers = max(1, _WORKERS // processes)
        threads = _THREADS if workers > 1 else _share(processes)
        return _fork(function, items, processes, (workers, threads))
    # A batch smaller than the pool leaves cores idle; give them to the items it
    # has.  `threads` is a floor, not a share.
    return _fork(function, items, processes, (1, _share(processes)))


def call(function: Callable, *args, **kwargs):
    """`function(*args, **kwargs)` in one worker holding every useful thread.

    For the serial chains -- a relaxation, a saddle search -- which have nothing
    to run beside them, and must still not run in the parent.
    """
    if not active():
        return function(*args, **kwargs)
    return _fork(lambda _: function(*args, **kwargs), [None], 1, (1, _share(1)))[0]
