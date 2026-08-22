"""Activation-scoped write authority, separate from execution cancellation."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from monoid_agent_kernel.core.interruption import InterruptionCause
from monoid_agent_kernel.errors import RunCancelled

_LOGGER = logging.getLogger(__name__)
_T = TypeVar("_T")


class WriteAuthorityRevoked(RunCancelled):
    """Unwind an activation after its host-issued writer authority is revoked."""

    error_code = "lease_lost"
    interruption_cause = InterruptionCause.LEASE_LOST

    def __init__(self) -> None:
        super().__init__(
            "activation writer authority was revoked",
            interruption_cause=InterruptionCause.LEASE_LOST,
        )


@dataclass
class ActivationWriteAuthority:
    """A sticky, process-local capability shared by every activation mutation surface.

    Durable stores still fence the host-issued ``WriterToken`` atomically with their mutation.
    This object closes the in-process half of the boundary: retained tool contexts, recorders,
    observers, and callbacks all fail closed once the activation is revoked.
    """

    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _revoked: bool = field(default=False, init=False, repr=False)
    _callbacks: dict[int, Callable[[], None]] = field(default_factory=dict, init=False, repr=False)
    _next_callback_id: int = field(default=0, init=False, repr=False)

    @property
    def revoked(self) -> bool:
        with self._lock:
            return self._revoked

    def revoke(self) -> bool:
        """Revoke once and notify registered execution-control bridges outside the lock."""

        with self._lock:
            if self._revoked:
                return False
            self._revoked = True
            callbacks = tuple(self._callbacks.values())
            self._callbacks.clear()
        for callback in callbacks:
            try:
                callback()
            except Exception:  # noqa: BLE001 - revocation must remain sticky despite diagnostics
                _LOGGER.exception("writer-authority revoke callback failed")
        return True

    def assert_active(self) -> None:
        with self._lock:
            if self._revoked:
                raise WriteAuthorityRevoked()

    def add_revoke_callback(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register a one-shot callback and return an idempotent unsubscribe function."""

        with self._lock:
            if self._revoked:
                callback_id = None
            else:
                callback_id = self._next_callback_id
                self._next_callback_id += 1
                self._callbacks[callback_id] = callback
        if callback_id is None:
            try:
                callback()
            except Exception:  # noqa: BLE001 - see revoke()
                _LOGGER.exception("late writer-authority revoke callback failed")

        def remove() -> None:
            if callback_id is None:
                return
            with self._lock:
                self._callbacks.pop(callback_id, None)

        return remove

    def guard_local_mutation(self, operation: Callable[[], _T]) -> _T:
        """Linearize one short process-local mutation with revocation."""

        with self._lock:
            if self._revoked:
                raise WriteAuthorityRevoked()
            try:
                result = operation()
            except BaseException as exc:
                if self._revoked:
                    raise WriteAuthorityRevoked() from exc
                raise
            if self._revoked:
                raise WriteAuthorityRevoked()
            return result

    def guard_external_call(self, operation: Callable[[], _T]) -> _T:
        """Fence an arbitrary callback/I/O edge without holding the authority lock across it."""

        self.assert_active()
        try:
            return operation()
        finally:
            # Authority loss outranks the callback's own result or failure. The external effect may
            # already have happened; this check prevents every later kernel-managed publication.
            self.assert_active()


__all__ = ["ActivationWriteAuthority", "WriteAuthorityRevoked"]
