from __future__ import annotations

import threading
from collections.abc import Callable


ProgressCallback = Callable[[str, str], None]


class JobCancelled(RuntimeError):
    """Raised when a user cancels a background LLM job."""


def check_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelled("Cancelled by user")


def publish(progress: ProgressCallback | None, kind: str, value: str) -> None:
    if progress is not None:
        progress(kind, value)
