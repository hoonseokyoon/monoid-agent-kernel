"""Fenced event and terminal bridges for finite host activations."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, NoReturn, TypeVar

from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.core.authority import ActivationWriteAuthority
from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.json_ingress import is_portable_json_integer
from monoid_agent_kernel.core.outcome import TerminalOutcome
from monoid_agent_kernel.errors import NativeAgentError
from monoid_agent_kernel.hosting.commit_results import accept_fenced_commit_result
from monoid_agent_kernel.hosting.contracts import FencedRunSink, WriterToken


_PRIVATE_DELTA_EVENTS = frozenset({"model.output.delta", "model.reasoning.delta"})
TerminalCommitStatus = Literal["committed", "already_committed", "conflict"]
_T = TypeVar("_T")


def _record_only_digest(payload: dict[str, object]) -> str:
    return canonical_sha256({"record": payload, "blobs": {}})


def _retry_ambiguous_mutation(
    operation: Callable[[], _T],
    *,
    write_authority: ActivationWriteAuthority,
) -> _T:
    """Retry one identity-stable mutation once and preserve the final transport error."""

    try:
        return write_authority.guard_external_call(operation)
    except Exception:
        if write_authority.revoked:
            raise
    return write_authority.guard_external_call(operation)


def _revoke_after_unsafe_mutation(
    write_authority: ActivationWriteAuthority,
    error: Exception,
) -> NoReturn:
    """Stop every later local projection after an uncertain or inconsistent durable mutation."""

    write_authority.revoke()
    try:
        write_authority.assert_active()
    except Exception as revoked:
        raise revoked from error
    raise AssertionError("revoked authority unexpectedly remained active")


@dataclass
class FencedEventSink:
    """Publish public events through the activation's durable writer fence.

    The caller seeds ``last_sequence`` from the authoritative journal before constructing the
    loop. Every event must then be the next coordinate. Model-authored delta content uses the
    separate durable stream contract and is rejected at this public journal boundary.
    """

    sink: FencedRunSink
    writer_token: WriterToken
    write_authority: ActivationWriteAuthority
    last_sequence: int = 0
    _closed: bool = field(default=False, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.writer_token, WriterToken):
            raise TypeError("fenced event sink requires WriterToken")
        if not isinstance(self.write_authority, ActivationWriteAuthority):
            raise TypeError("fenced event sink requires ActivationWriteAuthority")
        if not is_portable_json_integer(self.last_sequence) or self.last_sequence < 0:
            raise ValueError(
                "fenced event last_sequence must be a non-negative portable integer"
            )

    def emit(self, event: AgentEvent) -> None:
        if not isinstance(event, AgentEvent):
            raise TypeError("fenced event sink requires AgentEvent")
        if event.run_id != self.writer_token.run_id:
            self.write_authority.revoke()
            self.write_authority.assert_active()
        if event.type in _PRIVATE_DELTA_EVENTS:
            raise NativeAgentError(
                "model delta content requires the durable stream channel",
                error_code="private_event_content",
            )
        self.write_authority.assert_active()
        with self._lock:
            if self._closed:
                raise NativeAgentError(
                    "fenced event sink is closed",
                    error_code="event_sink_closed",
                )
            expected_sequence = self.last_sequence + 1
            if event.seq != expected_sequence:
                raise NativeAgentError(
                    "event sequence does not continue the authoritative journal",
                    error_code="event_sequence_conflict",
                )
            try:
                result = _retry_ambiguous_mutation(
                    lambda: self.sink.append_event(event, writer_token=self.writer_token),
                    write_authority=self.write_authority,
                )
            except Exception as uncertain:
                _revoke_after_unsafe_mutation(self.write_authority, uncertain)
            status = accept_fenced_commit_result(
                result,
                write_authority=self.write_authority,
            )
            if status == "conflict":
                _revoke_after_unsafe_mutation(
                    self.write_authority,
                    NativeAgentError(
                        "event coordinate has a different durable winner",
                        error_code="event_conflict",
                    ),
                )
            expected_digest = _record_only_digest(event.to_json())
            if result.sequence is not None and result.sequence != event.seq:
                _revoke_after_unsafe_mutation(
                    self.write_authority,
                    RuntimeError("fenced event sink returned invalid sequence evidence"),
                )
            if result.content_digest and result.content_digest != expected_digest:
                _revoke_after_unsafe_mutation(
                    self.write_authority,
                    RuntimeError("fenced event sink returned invalid commit evidence"),
                )
            self.last_sequence = event.seq

    def close(self) -> None:
        self.write_authority.assert_active()
        with self._lock:
            self._closed = True


@dataclass(frozen=True, kw_only=True)
class TerminalSettlement:
    """Canonical first-writer terminal observation after one settlement attempt."""

    status: TerminalCommitStatus
    outcome: TerminalOutcome
    terminal_ref: str


@dataclass
class FencedTerminalBridge:
    """Settle a terminal outcome and return the integrity-checked canonical winner."""

    sink: FencedRunSink
    writer_token: WriterToken
    write_authority: ActivationWriteAuthority
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.writer_token, WriterToken):
            raise TypeError("fenced terminal bridge requires WriterToken")
        if not isinstance(self.write_authority, ActivationWriteAuthority):
            raise TypeError("fenced terminal bridge requires ActivationWriteAuthority")

    def settle(self, outcome: TerminalOutcome) -> TerminalSettlement:
        if not isinstance(outcome, TerminalOutcome):
            raise TypeError("terminal bridge requires TerminalOutcome")
        if outcome.run_id != self.writer_token.run_id:
            self.write_authority.revoke()
            self.write_authority.assert_active()
        self.write_authority.assert_active()
        with self._lock:
            try:
                result = _retry_ambiguous_mutation(
                    lambda: self.sink.settle_terminal(outcome, writer_token=self.writer_token),
                    write_authority=self.write_authority,
                )
            except Exception as uncertain:
                try:
                    winner = self.write_authority.guard_external_call(
                        lambda: self.sink.read_terminal(outcome.run_id)
                    )
                except Exception as read_error:
                    _revoke_after_unsafe_mutation(self.write_authority, read_error)
                if not isinstance(winner, TerminalOutcome) or winner.run_id != outcome.run_id:
                    _revoke_after_unsafe_mutation(self.write_authority, uncertain)
                return TerminalSettlement(
                    status="already_committed" if winner == outcome else "conflict",
                    outcome=winner,
                    terminal_ref=f"terminal:{winner.run_id}",
                )
            status = accept_fenced_commit_result(
                result,
                write_authority=self.write_authority,
            )
            winner = self.write_authority.guard_external_call(
                lambda: self.sink.read_terminal(outcome.run_id)
            )
            if not isinstance(winner, TerminalOutcome) or winner.run_id != outcome.run_id:
                _revoke_after_unsafe_mutation(
                    self.write_authority,
                    RuntimeError("terminal settlement has no valid canonical winner"),
                )
            winner_digest = _record_only_digest(winner.to_json())
            if status in {"committed", "already_committed"}:
                if (result.content_digest and result.content_digest != winner_digest) or (
                    winner != outcome
                ):
                    _revoke_after_unsafe_mutation(
                        self.write_authority,
                        RuntimeError(
                            "terminal settlement commit evidence disagrees with readback"
                        ),
                    )
            elif result.winner_digest and result.winner_digest != winner_digest:
                _revoke_after_unsafe_mutation(
                    self.write_authority,
                    RuntimeError("terminal settlement winner evidence disagrees with readback"),
                )
            return TerminalSettlement(
                status=status,
                outcome=winner,
                terminal_ref=f"terminal:{winner.run_id}",
            )

__all__ = [
    "FencedEventSink",
    "FencedTerminalBridge",
    "TerminalCommitStatus",
    "TerminalSettlement",
]
