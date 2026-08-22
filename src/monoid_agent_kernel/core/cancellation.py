from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from monoid_agent_kernel.core.interruption import InterruptionCause


@dataclass
class CancellationToken:
    _event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _callbacks: dict[int, Callable[[], None]] = field(default_factory=dict, init=False, repr=False)
    _next_callback_id: int = field(default=0, init=False, repr=False)
    _cause: InterruptionCause | None = field(default=None, init=False, repr=False)

    def cancel(
        self,
        cause: InterruptionCause = InterruptionCause.USER_CANCEL,
    ) -> None:
        try:
            cause = InterruptionCause(cause)
        except (TypeError, ValueError) as exc:
            raise ValueError("cancellation cause is outside the portable vocabulary") from exc
        with self._lock:
            if self._event.is_set():
                return
            self._cause = cause
            self._event.set()
            callbacks = tuple(self._callbacks.values())
            self._callbacks.clear()
        for callback in callbacks:
            callback()

    def add_cancel_callback(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register a one-shot callback and return an idempotent unsubscribe function."""

        with self._lock:
            if self._event.is_set():
                callback_id = None
            else:
                callback_id = self._next_callback_id
                self._next_callback_id += 1
                self._callbacks[callback_id] = callback
        if callback_id is None:
            callback()

        def remove() -> None:
            if callback_id is None:
                return
            with self._lock:
                self._callbacks.pop(callback_id, None)

        return remove

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    @property
    def cause(self) -> InterruptionCause | None:
        with self._lock:
            return self._cause
