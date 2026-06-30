from __future__ import annotations

import time
from collections.abc import Callable


def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_s: float = 10.0,
    interval_s: float = 0.01,
    reason: str = "condition was not met",
) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            if predicate():
                return
        except Exception as exc:  # noqa: BLE001 - surface the last polling error in the assertion
            last_error = exc
        time.sleep(interval_s)
    if last_error is not None:
        raise AssertionError(f"{reason}; last error: {last_error}") from last_error
    raise AssertionError(reason)


def eventually(
    predicate: Callable[[], bool],
    *,
    timeout_s: float = 10.0,
    interval_s: float = 0.01,
) -> bool:
    try:
        wait_until(predicate, timeout_s=timeout_s, interval_s=interval_s)
    except AssertionError:
        return False
    return True

