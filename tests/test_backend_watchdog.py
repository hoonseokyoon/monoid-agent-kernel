from __future__ import annotations

import threading
from typing import Any

from support.waiting import eventually


def test_watchdog_stop_timeout_keeps_live_owner_registered(
    backend_factory: Any,
) -> None:
    backend = backend_factory.create()
    backend.watchdog_interval_s = 0.01
    tick_entered = threading.Event()
    release_tick = threading.Event()

    def blocked_reclaim() -> list[str]:
        tick_entered.set()
        release_tick.wait(timeout=5)
        return []

    backend._reclaim_stale_runs = blocked_reclaim  # type: ignore[method-assign]
    backend.start_watchdog()
    assert tick_entered.wait(timeout=2)
    original = backend._watchdog_thread
    assert original is not None and original.is_alive()

    assert backend.stop_watchdog(timeout_s=0.01) is False
    assert backend._watchdog_thread is original

    backend.start_watchdog()
    assert backend._watchdog_thread is original

    release_tick.set()
    assert eventually(lambda: not original.is_alive(), timeout_s=2)
    assert backend.stop_watchdog(timeout_s=1) is True
    assert backend._watchdog_thread is None
