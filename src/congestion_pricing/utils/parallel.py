"""Parallelization utilities."""

from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def chunked(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    """Split an iterable into chunks of a given size.

    Args:
        iterable: Input iterable
        size: Chunk size

    Yields:
        Lists of at most `size` elements
    """
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    n_jobs: int = -1,
    backend: str = "loky",
    verbose: int = 0,
) -> list[R]:
    """Apply a function to items in parallel.

    Args:
        func: Function to apply
        items: Iterable of items
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        backend: Joblib backend
        verbose: Verbosity level

    Returns:
        List of results
    """
    try:
        from joblib import Parallel, delayed

        return Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
            delayed(func)(item) for item in items
        )
    except ImportError:
        # Fallback to sequential if joblib not available
        return [func(item) for item in items]


def parallel_starmap(
    func: Callable[..., R],
    items: Iterable[tuple],
    n_jobs: int = -1,
    backend: str = "loky",
    verbose: int = 0,
) -> list[R]:
    """Apply a function to items in parallel with argument unpacking.

    Args:
        func: Function to apply
        items: Iterable of argument tuples
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        backend: Joblib backend
        verbose: Verbosity level

    Returns:
        List of results
    """
    try:
        from joblib import Parallel, delayed

        return Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
            delayed(func)(*args) for args in items
        )
    except ImportError:
        return [func(*args) for args in items]
