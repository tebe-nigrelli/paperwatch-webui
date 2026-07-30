from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


class RunAlreadyInProgress(RuntimeError):
    pass


class AnalysisAlreadyInProgress(RuntimeError):
    pass


@contextmanager
def _named_lock(
    database_path: str | Path,
    *,
    suffix: str,
    error_type: type[RuntimeError],
    message: str,
) -> Iterator[None]:
    path = Path(f"{Path(database_path)}.{suffix}.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle: TextIO = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise error_type(message.format(database=Path(database_path))) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@contextmanager
def run_lock(database_path: str | Path) -> Iterator[None]:
    """Prevent overlapping timer, CLI, and web fetches on the same database."""

    with _named_lock(
        database_path,
        suffix="run",
        error_type=RunAlreadyInProgress,
        message="Another paperwatch fetch is already using {database}",
    ):
        yield


@contextmanager
def analysis_lock(database_path: str | Path) -> Iterator[None]:
    """Prevent overlapping on-demand model batches across web processes."""

    with _named_lock(
        database_path,
        suffix="analysis",
        error_type=AnalysisAlreadyInProgress,
        message="Another paperwatch analysis batch is already using {database}",
    ):
        yield
