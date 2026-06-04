"""Parallel batch helpers for label conversion."""

import os
from multiprocessing import Pool
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, TypeVar

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional at runtime
    tqdm = None

T = TypeVar("T")
R = TypeVar("R")


def default_workers() -> int:
    """Return the number of available CPUs for parallel workers."""
    return max(1, os.cpu_count() or 1)


def _pool_initializer() -> None:
    """Safe matplotlib backend for forked worker processes."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except ImportError:
        pass


def _imap_chunksize(num_tasks: int, workers: int) -> int:
    return max(1, min(32, num_tasks // max(workers * 4, 1) or 1))


def run_parallel(
    tasks: Sequence[T],
    worker: Callable[[T], R],
    workers: int = 1,
    desc: str = "Converting",
    show_progress: bool = True,
    max_tasks_per_child: Optional[int] = 50,
) -> List[R]:
    """Run ``worker`` over ``tasks`` with optional multiprocessing and tqdm."""
    if not tasks:
        return []

    workers = max(1, workers)
    use_progress = show_progress and tqdm is not None

    if workers == 1:
        iterator: Iterable[T] = tasks
        if use_progress:
            iterator = tqdm(tasks, desc=desc, unit="file")
        return [worker(task) for task in iterator]

    # Stream tasks to workers instead of submitting all futures upfront (e.g. 170k+).
    chunksize = _imap_chunksize(len(tasks), workers)
    with Pool(
        processes=workers,
        initializer=_pool_initializer,
        maxtasksperchild=max_tasks_per_child,
    ) as pool:
        iterator = pool.imap_unordered(worker, tasks, chunksize=chunksize)
        if use_progress:
            iterator = tqdm(iterator, total=len(tasks), desc=desc, unit="file")
        return list(iterator)


def summarize_results(
    results: List[Tuple[bool, str]],
) -> Tuple[int, int, List[str]]:
    ok = fail = 0
    errors: List[str] = []
    for success, msg in results:
        if success:
            ok += 1
        else:
            fail += 1
            if msg:
                errors.append(msg)
    return ok, fail, errors
