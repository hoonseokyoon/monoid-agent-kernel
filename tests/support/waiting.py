from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


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


def wait_for_durable_status(
    run_root: Path,
    run_id: str,
    *,
    where: Callable[[dict[str, Any]], bool] | None = None,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """The run's durable ``status.json``, once the writer has actually produced it.

    A test that waits on ``backend._record(run_id)`` has waited on an IN-MEMORY fact, and the
    status artifact is written after it -- so a second backend constructed on the same run root
    reads whatever happens to be on disk at that instant. That is not a slow test, it is a race
    the test cannot see: it passes whenever the writer wins and fails whenever it does not, and
    the failure lands in the assertion about the SECOND backend, pointing away from the wait that
    was missing.

    ``where`` takes the parsed artifact so the wait can name the state it is waiting FOR rather
    than merely the file's existence -- a status.json from an earlier moment of the same run
    exists too, and satisfies a bare ``.exists()``.

    This raises rather than returning a bool, and it returns the artifact, so a caller cannot
    accidentally wait for something and then read it a second time from disk.
    """

    status_path = run_root / run_id / "status.json"
    seen: dict[str, Any] = {}

    def ready() -> bool:
        if not status_path.exists():
            return False
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        if where is not None and not where(payload):
            return False
        seen.clear()
        seen.update(payload)
        return True

    wait_until(
        ready,
        timeout_s=timeout_s,
        reason=f"{status_path} never reported the durable state this test waits on",
    )
    return seen


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

