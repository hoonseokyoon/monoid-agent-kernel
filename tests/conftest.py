from __future__ import annotations

# ruff: noqa: E402

import faulthandler
import os
import sys
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --- Hang guard -------------------------------------------------------------------------
# The threaded-HTTP / shared-asyncio-loop / subprocess tests have, very rarely, left a test
# blocked on a timing race. For a backgrounded, output-buffered run that looks like the whole
# suite silently stalled forever (no output, process alive but idle) — which is exactly what bit
# us. Arm a process-wide watchdog: if the run wedges past this deadline, dump every thread's
# stack (pinpointing the deadlock) and abort, instead of hanging indefinitely. The suite
# finishes well under this; override with NAR_TEST_HANG_TIMEOUT_S=0 to disable, or a custom value.
faulthandler.enable()
_HANG_TIMEOUT_S = float(os.environ.get("NAR_TEST_HANG_TIMEOUT_S", "240"))
if _HANG_TIMEOUT_S > 0:
    faulthandler.dump_traceback_later(_HANG_TIMEOUT_S, exit=True)


from monoid_agent_kernel.reference.backend.service import RunnerBackend
from support.studio_harness import studio as studio

_ACTIVE_BACKENDS: list[RunnerBackend] = []
_BACKEND_FUTURES: dict[int, list[Any]] = {}
_ORIGINAL_BACKEND_POST_INIT = RunnerBackend.__post_init__
_ORIGINAL_BACKEND_SPAWN = RunnerBackend._spawn


def _tracked_backend_post_init(self: RunnerBackend) -> None:
    _ORIGINAL_BACKEND_POST_INIT(self)
    _ACTIVE_BACKENDS.append(self)
    _BACKEND_FUTURES.setdefault(id(self), [])


def _tracked_backend_spawn(self: RunnerBackend, coro: Any) -> Any:
    future = _ORIGINAL_BACKEND_SPAWN(self, coro)
    _BACKEND_FUTURES.setdefault(id(self), []).append(future)
    return future


RunnerBackend.__post_init__ = _tracked_backend_post_init
RunnerBackend._spawn = _tracked_backend_spawn


@pytest.fixture(autouse=True)
def _drain_runner_backends_after_test() -> Any:
    yield
    try:
        for backend in list(_ACTIVE_BACKENDS):
            backend.shutdown(drain=True, drain_timeout_s=5.0)
        for backend in list(_ACTIVE_BACKENDS):
            for future in _BACKEND_FUTURES.get(id(backend), []):
                try:
                    future.result(timeout=0.5)
                except FutureTimeoutError:
                    future.cancel()
                    time.sleep(0.05)
                    try:
                        future.result(timeout=1.0)
                    except Exception:
                        pass
                except Exception:
                    pass
    finally:
        _BACKEND_FUTURES.clear()
        _ACTIVE_BACKENDS.clear()


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    del session, exitstatus
    # The run completed within the deadline — disarm so the watchdog can't fire during a
    # (bounded) interpreter shutdown and turn a green run red.
    faulthandler.cancel_dump_traceback_later()
