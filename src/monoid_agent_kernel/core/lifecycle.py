"""Session lifecycle: a named ``AgentSession`` contract and a formal state machine.

``AgentLoop`` is the engine; an embedder (Agent Cell / Daemon) should be able to
*control* a session without importing the loop's internals. This module provides:

- ``SessionState`` — the formal lifecycle FSM. Backend lifecycle payloads expose this
  vocabulary as ``state`` plus a separate ``terminal`` boolean.
- ``state_from_suspension`` — the projection core: how a pump ``Suspension`` maps to a
  state. This is what keeps the FSM in sync with the engine without the engine knowing
  about the FSM.
- ``LEGAL_TRANSITIONS`` + ``can_transition`` / ``assert_transition`` — the legal-edge
  table the facade enforces.
- ``AgentSession`` (Protocol) + ``LoopSession`` (a thin facade over ``AgentLoop`` that
  owns the FSM and exposes ``inspect()`` / ``health()`` — added in Step 2).

The engine (``loop.py``) is intentionally untouched: the facade derives state from the
signals the loop already produces at every quiescent park (``Suspension`` + the live
``_Session.terminal`` flag), so there is no second source of truth to drift.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.result import AgentTurnResult, Suspension
from monoid_agent_kernel.errors import NativeAgentError

if TYPE_CHECKING:
    from monoid_agent_kernel.loop import AgentLoop


class SessionState(str, enum.Enum):
    """The formal lifecycle state of an agent session.

    ``str``-valued so a state serializes to its wire name verbatim (``state.value``
    and ``SessionState(value)`` round-trip through JSON). ``FAILED`` and ``COMPLETED``
    are terminal (empty out-set in ``LEGAL_TRANSITIONS``).
    """

    CREATED = "created"  # constructed, not bootstrapped (open() not called)
    IDLE = "idle"  # open + bootstrapped, no turn in flight, ready for submit
    RUNNING = "running"  # a turn is actively stepping
    AWAITING_INPUT = "awaiting_input"  # settled, parked for the next user message
    AWAITING_TASKS = "awaiting_tasks"  # parked on a hosted/external task
    PAUSED = "paused"  # cooperatively paused (Step 3)
    INTERRUPTED = "interrupted"  # turn stopped, session alive
    TURN_FAILED = "turn_failed"  # recoverable turn error, session alive
    LIMITED = "limited"  # per-submit/session budget hit (non-terminal)
    CANCELLED = "cancelled"  # terminal: cancelled by an external caller
    FAILED = "failed"  # terminal failure
    COMPLETED = "completed"  # terminal success (closed cleanly)


#: The unambiguous ``Suspension.reason -> SessionState`` edges. ``"terminal"`` is the
#: one reason that needs the suspension's status/error to disambiguate, so it is handled
#: in ``state_from_suspension`` rather than here.
REASON_TO_STATE: dict[str, SessionState] = {
    "settled": SessionState.AWAITING_INPUT,
    "awaiting_tasks": SessionState.AWAITING_TASKS,
    "limited": SessionState.LIMITED,
    "paused": SessionState.PAUSED,
    "interrupted": SessionState.INTERRUPTED,
    "turn_failed": SessionState.TURN_FAILED,
}


def state_from_suspension(suspension: Suspension) -> SessionState:
    """Project a pump ``Suspension`` onto a ``SessionState``.

    ``"terminal"`` always means the loop set ``_Session.terminal=True`` — a dead run.
    Cancel arrives as ``reason="terminal"`` with ``error_code="cancelled"`` and maps to
    the distinct ``CANCELLED`` state; any other terminal maps to ``FAILED``. (Clean
    ``COMPLETED`` is reached only via ``close()`` returning a successful
    ``AgentRunResult``, never via a ``Suspension``.)
    """
    if suspension.reason == "terminal":
        return SessionState.CANCELLED if suspension.error_code == "cancelled" else SessionState.FAILED
    try:
        return REASON_TO_STATE[suspension.reason]
    except KeyError as exc:  # pragma: no cover - guards a future unmapped reason
        raise NativeAgentError(
            f"no SessionState mapping for suspension reason {suspension.reason!r}",
            error_code="unmapped_suspension_reason",
        ) from exc


#: Legal state transitions. A facade boundary validates its computed next state against
#: this table before assigning. Terminal states have an empty out-set. The table is
#: permissive about ``COMPLETED`` / ``FAILED`` from any live state because ``close()`` and
#: ``cancel()`` can finalize a run from any non-terminal park.
_LIVE_FINALIZE = frozenset(
    {SessionState.COMPLETED, SessionState.FAILED, SessionState.CANCELLED}
)
LEGAL_TRANSITIONS: dict[SessionState, frozenset[SessionState]] = {
    SessionState.CREATED: frozenset({SessionState.IDLE}) | _LIVE_FINALIZE,
    SessionState.IDLE: frozenset({SessionState.RUNNING}) | _LIVE_FINALIZE,
    SessionState.RUNNING: frozenset(
        {
            SessionState.AWAITING_INPUT,
            SessionState.AWAITING_TASKS,
            SessionState.PAUSED,
            SessionState.INTERRUPTED,
            SessionState.TURN_FAILED,
            SessionState.LIMITED,
        }
    )
    | _LIVE_FINALIZE,
    SessionState.AWAITING_INPUT: frozenset({SessionState.RUNNING}) | _LIVE_FINALIZE,
    SessionState.AWAITING_TASKS: frozenset({SessionState.RUNNING}) | _LIVE_FINALIZE,
    SessionState.PAUSED: frozenset({SessionState.RUNNING}) | _LIVE_FINALIZE,
    SessionState.INTERRUPTED: frozenset({SessionState.RUNNING, SessionState.AWAITING_INPUT})
    | _LIVE_FINALIZE,
    SessionState.TURN_FAILED: frozenset({SessionState.RUNNING}) | _LIVE_FINALIZE,
    SessionState.LIMITED: frozenset({SessionState.RUNNING}) | _LIVE_FINALIZE,
    SessionState.CANCELLED: frozenset(),
    SessionState.FAILED: frozenset(),
    SessionState.COMPLETED: frozenset(),
}

TERMINAL_STATES: frozenset[SessionState] = frozenset(
    {SessionState.CANCELLED, SessionState.FAILED, SessionState.COMPLETED}
)


#: Maps legacy lifecycle status strings from older backend payloads and ``status.json`` onto the
#: one ``SessionState``. ``Suspension.reason`` has its own richer projection in
#: :func:`state_from_suspension`.
_STATUS_STRING_TO_STATE: dict[str, SessionState] = {
    "queued": SessionState.CREATED,
    "created": SessionState.CREATED,
    "idle": SessionState.IDLE,
    "running": SessionState.RUNNING,
    "awaiting_input": SessionState.AWAITING_INPUT,
    "awaiting_tasks": SessionState.AWAITING_TASKS,
    "waiting_for_background_jobs": SessionState.AWAITING_TASKS,
    "paused": SessionState.PAUSED,
    "interrupted": SessionState.INTERRUPTED,
    "turn_failed": SessionState.TURN_FAILED,
    "completed": SessionState.COMPLETED,
    "failed": SessionState.FAILED,
    "limited": SessionState.LIMITED,
    "cancelled": SessionState.CANCELLED,
}


def session_state_value(state: SessionState | str) -> str:
    """Return the canonical wire value for a lifecycle state."""
    return state.value if isinstance(state, SessionState) else SessionState(str(state)).value


def session_state_from_run_status(
    status: str | SessionState,
    *,
    error_code: str = "",
    terminal: bool = False,
) -> SessionState:
    """Project run lifecycle strings onto :class:`SessionState`.

    ``status`` may be an existing ``SessionState`` value or one of the legacy lifecycle status
    strings read from ``status.json`` / older backend payloads. ``terminal`` disambiguates the
    public lifecycle payload, but ``SessionState.LIMITED`` remains the vocabulary value for both a
    live budget-limited park and a terminal limited result.
    """
    if isinstance(status, SessionState):
        return status
    if error_code == "cancelled" and status in {"limited", "failed"}:
        return SessionState.CANCELLED
    return _STATUS_STRING_TO_STATE.get(str(status), SessionState.CREATED)


def lifecycle_from_status_artifact(
    payload: Mapping[str, Any] | None,
    *,
    failure_present: bool = False,
) -> tuple[SessionState, bool]:
    """Project a durable ``status.json`` payload onto ``(state, terminal)``.

    New artifacts carry ``state`` plus an explicit ``terminal`` flag. Legacy artifacts carry
    ``status`` only, so terminal result statuses are inferred there, including bare
    ``status="limited"`` from pre-``state`` terminal-limited runs.
    """
    status_payload = payload or {}
    state_value = status_payload.get("state")
    status_value = status_payload.get("status")
    raw_state = state_value or status_value
    if raw_state:
        terminal = bool(status_payload.get("terminal"))
        state = session_state_from_run_status(
            str(raw_state),
            error_code=str(status_payload.get("error_code") or ""),
            terminal=terminal,
        )
        if "terminal" not in status_payload:
            state_text = str(state_value or "")
            status_text = str(status_value or "")
            if state_text in {"completed", "failed", "cancelled"} or (
                not state_text and status_text in {"completed", "failed", "limited", "cancelled"}
            ):
                terminal = True
        return state, terminal
    if failure_present:
        return SessionState.FAILED, True
    return SessionState.CREATED, False


def to_session_state(status: str, *, error_code: str = "") -> SessionState:
    """Compatibility wrapper for legacy status readers."""
    return session_state_from_run_status(status, error_code=error_code)


def can_transition(src: SessionState, dst: SessionState) -> bool:
    """Whether ``src -> dst`` is a legal edge. A self-edge (``src == dst``) is always
    legal (an idempotent re-derivation of the same state must never fail)."""
    if src == dst:
        return True
    return dst in LEGAL_TRANSITIONS.get(src, frozenset())


def assert_transition(src: SessionState, dst: SessionState) -> None:
    """Raise if ``src -> dst`` is not a legal edge."""
    if not can_transition(src, dst):
        raise NativeAgentError(
            f"illegal session transition {src.value!r} -> {dst.value!r}",
            error_code="illegal_session_transition",
        )


# --- AgentSession contract + LoopSession facade (Step 2) ----------------------------------

#: States in which a session can accept a new user message (used by ``health()``).
_CAN_ACCEPT_INPUT: frozenset[SessionState] = frozenset(
    {
        SessionState.IDLE,
        SessionState.AWAITING_INPUT,
        SessionState.INTERRUPTED,
        SessionState.TURN_FAILED,
        SessionState.LIMITED,
        SessionState.PAUSED,
    }
)


@dataclass(frozen=True)
class SessionInspection:
    """A point-in-time view of a live session, recomputed on every ``inspect()`` call
    from the loop's in-memory state (so it can never go stale)."""

    state: SessionState
    run_id: str
    terminal: bool
    pending_tasks: bool
    awaiting_task_ids: tuple[str, ...]
    last_suspension_reason: str | None
    turn_handle: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "run_id": self.run_id,
            "terminal": self.terminal,
            "pending_tasks": self.pending_tasks,
            "awaiting_task_ids": list(self.awaiting_task_ids),
            "last_suspension_reason": self.last_suspension_reason,
            "turn_handle": self.turn_handle,
        }


@dataclass(frozen=True)
class SessionHealth:
    """Cheap liveness projection for a control plane — does the session live, and can it
    take input right now?"""

    state: SessionState
    alive: bool
    can_accept_input: bool
    has_pending_tasks: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "alive": self.alive,
            "can_accept_input": self.can_accept_input,
            "has_pending_tasks": self.has_pending_tasks,
        }


@runtime_checkable
class AgentSession(Protocol):
    """The stable embedder contract for driving and observing one session.

    ``AgentLoop`` is the engine; ``LoopSession`` is the reference facade that satisfies
    this Protocol. An Agent Daemon/Cell depends on ``AgentSession`` (+ the control
    protocol) rather than importing ``AgentLoop``.
    """

    @property
    def state(self) -> SessionState: ...

    def inspect(self) -> SessionInspection: ...

    def health(self) -> SessionHealth: ...

    def pause(self) -> None: ...

    def resume(self) -> Suspension: ...

    def cancel(self, reason: str = "") -> None: ...


@dataclass
class LoopSession:
    """A thin, FSM-owning facade over :class:`~monoid_agent_kernel.loop.AgentLoop`.

    It does not subclass or rename the loop — it wraps it, delegates execution, and
    re-derives :class:`SessionState` at each boundary from the signals the loop already
    produces (a returned ``Suspension`` + the live ``_Session.terminal`` flag). ``_state``
    is a convenience cache for synchronous callers; ``inspect()`` always recomputes from
    live loop state. Nothing new is persisted — on restore a fresh facade derives its
    state from the restored loop, never from a stored facade field.
    """

    loop: AgentLoop
    _state: SessionState = SessionState.CREATED
    _last_suspension: Suspension | None = None
    _cancel_reason: str = ""

    @property
    def state(self) -> SessionState:
        return self._state

    def _set_state(self, dst: SessionState) -> None:
        assert_transition(self._state, dst)
        self._state = dst

    def _derive_after_settle(self, turn: AgentTurnResult) -> SessionState:
        session = self.loop._session
        if session is not None and session.terminal:
            return SessionState.FAILED
        if turn.status == "limited":
            return SessionState.LIMITED
        return SessionState.AWAITING_INPUT

    # --- lifecycle delegation ---------------------------------------------------------

    def open(self) -> None:
        self.loop.open()
        session = self.loop._session
        self._set_state(
            SessionState.FAILED if (session is not None and session.terminal) else SessionState.IDLE
        )

    def submit(self, user_input: Any) -> AgentTurnResult:
        """Blocking convenience: run one user turn to settle. Mirrors ``AgentLoop.submit``."""
        self._set_state(SessionState.RUNNING)
        turn = self.loop.submit(user_input)
        self._set_state(self._derive_after_settle(turn))
        return turn

    def run_until_suspended(self, user_input: Any | None = None) -> Suspension:
        """Non-blocking pump: step until the run suspends, mapping the returned
        ``Suspension`` onto a state. With ``None`` it resumes a parked run."""
        self._set_state(SessionState.RUNNING)
        suspension = self.loop.run_until_suspended(user_input)
        self._last_suspension = suspension
        self._set_state(state_from_suspension(suspension))
        return suspension

    def close(self) -> Any:
        result = self.loop.close()
        status = getattr(result, "status", "completed")
        self._set_state(SessionState.FAILED if status == "failed" else SessionState.COMPLETED)
        return result

    # --- control: pause / resume / cancel ---------------------------------------------

    def pause(self) -> None:
        """Signal a cooperative pause. One-way, non-blocking (mirrors the loop's
        ``pause_turn``): the running turn freezes at the start of its next step and the
        driving ``run_until_suspended`` returns ``reason="paused"`` — at which point the
        facade transitions to ``PAUSED``. A no-op once the run is terminal."""
        if self._state in TERMINAL_STATES:
            return
        self.loop.pause_turn()

    def resume(self) -> Suspension:
        """Resume a paused (or task-parked) run by re-pumping with no new input. Continues
        the same turn from where it froze (``pending_observations`` were kept)."""
        return self.run_until_suspended(None)

    def cancel(self, reason: str = "") -> None:
        """Request a terminal cancel. One-way signal: the next step boundary raises and the
        driving pump settles the run terminal (the facade then maps it to ``FAILED``).
        ``reason`` is retained for reporting (``inspect``/control results)."""
        self._cancel_reason = reason
        token = self.loop.cancellation_token
        if token is None:
            token = CancellationToken()
            self.loop.cancellation_token = token
        token.cancel()

    # --- projections (always recomputed from live loop state) -------------------------

    def inspect(self) -> SessionInspection:
        session = self.loop._session
        run_id = self.loop.spec.run_id
        if session is None:
            return SessionInspection(
                state=self._state,
                run_id=run_id,
                terminal=False,
                pending_tasks=False,
                awaiting_task_ids=(),
                last_suspension_reason=None,
                turn_handle=None,
            )
        last = self._last_suspension
        awaiting = last.awaiting_task_ids if (last and last.reason == "awaiting_tasks") else ()
        return SessionInspection(
            state=self._state,
            run_id=run_id,
            terminal=session.terminal,
            pending_tasks=self.loop.has_pending_tasks(),
            awaiting_task_ids=tuple(awaiting),
            last_suspension_reason=(last.reason if last else None),
            turn_handle=session.state.previous_turn_handle,
        )

    def health(self) -> SessionHealth:
        session = self.loop._session
        terminal = bool(session.terminal) if session is not None else False
        pending = self.loop.has_pending_tasks() if session is not None else False
        return SessionHealth(
            state=self._state,
            alive=not terminal,
            can_accept_input=self._state in _CAN_ACCEPT_INPUT,
            has_pending_tasks=pending,
        )
