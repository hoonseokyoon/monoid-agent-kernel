from __future__ import annotations

# ruff: noqa: E402

import faulthandler
import os
import sys
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
# stack (pinpointing the deadlock) and abort. Re-arm it for each test so the timeout catches
# a wedged test item while healthy full-suite runs can exceed the per-test deadline. Override
# with NAR_TEST_HANG_TIMEOUT_S=0 to disable, or a custom value.
faulthandler.enable()
_HANG_TIMEOUT_S = float(os.environ.get("NAR_TEST_HANG_TIMEOUT_S", "240"))


def _arm_hang_watchdog() -> None:
    if _HANG_TIMEOUT_S <= 0:
        return
    faulthandler.cancel_dump_traceback_later()
    faulthandler.dump_traceback_later(_HANG_TIMEOUT_S, exit=True)


_arm_hang_watchdog()


def pytest_runtest_setup(item: Any) -> None:
    del item
    # Keep the watchdog as a per-test hang guard instead of a whole-suite wall clock.
    _arm_hang_watchdog()


from support.backend_factory import ManagedBackendFactory, set_current_backend_factory
from support.studio_harness import studio as studio


@pytest.fixture(autouse=True)
def backend_factory(tmp_path: Path) -> Any:
    factory = ManagedBackendFactory(tmp_path)
    set_current_backend_factory(factory)
    try:
        yield factory
    finally:
        set_current_backend_factory(None)
        factory.close()


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    del session, exitstatus
    # The run completed within the deadline — disarm so the watchdog can't fire during a
    # (bounded) interpreter shutdown and turn a green run red.
    faulthandler.cancel_dump_traceback_later()
