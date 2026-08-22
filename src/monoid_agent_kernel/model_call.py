"""Executing one model call: adapter dispatch, the cancel/deadline race, and the capture receipt.

`AgentLoop` used to own this as a family of private methods. It is a separate, reusable component
because a model call is a complete unit of work on its own -- a gateway, a batch driver or an
evaluation harness needs exactly this and none of the loop's step budget, tool surface or
conversation state.

**Why this module sits at the package root rather than under `core/`.** Everything it must name --
`ModelRequest`, `ModelTurn`, `ModelStreamChunk`, `assemble_streamed_turn` -- lives in `providers`,
and `core` never imports `providers` (`core/model_io.py` states the rule; `core/streaming.py` goes
as far as documenting `ModelStreamChunk` in prose to avoid the import). Moving those types into
`core` is not available either: `ModelRequest.tools` is a `tuple[ToolSpec, ...]` and `tools/base.py`
imports `core.content`, so it would close a `core` <-> `tools` cycle. A runner that drives adapters
genuinely depends on provider vocabulary, so it belongs in the layer above both, next to `loop` --
which is where `permissions`, `recorder` and `loop_phases` already live.

**Where the replay key's arithmetic lives.** One layer down, in
`providers/_request_identity.py`, so the replay adapter shares the exact functions this
runner stamps receipts with (this module cannot be imported from `providers` without a
cycle). The names are re-imported here for this module's own callers and their tests.

**Where retry lives: one owner per call, named by the config.** Backoff and HTTP
classification live inside the adapters (`providers/gateway.py`), and `ModelRetryConfig.layer`
names which loop owns a call. Under the default `"adapter"` layer the kernel makes exactly one
adapter call per turn; under `"kernel"` the attempt loop in this module re-dispatches, with the
adapter's copy of the config neutralized to a single attempt so a config-honoring loop cannot
multiply against it. (A loop that reads neither the config nor the layer is out of the kernel's
reach — its retries surface as `provider_retried` evidence, never as attempts; CONTRACTS states
both sides of that compliance.) That is the distinction `ModelCallReceipt.attempts` (kernel dispatches) and
`provider_retried` (a loop below the adapter boundary reported) encode — and classification is
inherited from the adapters either way: this module reads what the escaping exception carries,
it never invents its own taxonomy. (This paragraph said "the kernel makes exactly one adapter
call per turn" unconditionally until W7-3; that stopped being true when W7-0 landed the kernel
layer, and two review cycles read past it.)
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, replace
from typing import Any

from monoid_agent_kernel.core._sync_bridge import (
    CalleeCancelled,
    abandon_unwaited_call,
    await_abandonable_call,
    consume_task_outcome,
    is_async_callable,
    start_abandonable_sync_call,
)
from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.json_ingress import (
    normalize_json_ingress,
    normalize_unicode_scalars,
    portable_type_name,
)
from monoid_agent_kernel.core.model_io import (
    ModelCallAttempt,
    ModelCallReceipt,
    ModelIOSubscription,
    destination_digest,
    dispatch_model_call,
)
from monoid_agent_kernel.core.safe_evidence import is_safe_opaque_id
from monoid_agent_kernel.core.spec import ModelConfig, ModelRetryConfig
from monoid_agent_kernel.errors import (
    DurableModelCallError,
    ModelAdapterError,
    ModelCallAborted,
    ModelDispatchRefused,
    ModelEvidenceUncommitted,
    RunCancelled,
    RunTimeout,
)
from monoid_agent_kernel.model_lifecycle import (
    ModelCallLifecycleHook,
    ModelDispatchReservation,
    RecoveredModelDispatch,
    dispatch_evidence,
    durable_model_result_blob,
    durable_model_turn,
    mark_recovered_model_usage,
    raise_model_dispatch_unknown,
    recover_model_dispatch,
    reserve_model_dispatch,
    safe_failure_code,
    settle_model_dispatch,
)
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelStreamIngressNormalizer,
    ModelStreamChunk,
    ModelTurn,
    assemble_streamed_turn,
    collect_retry_reports,
    mark_provider_retried,
    mark_provider_usage,
    new_idempotency_key,
    normalize_model_request,
    normalize_model_config,
    normalize_model_turn,
    resolved_provider_name,
)
from monoid_agent_kernel.providers._common import retry_delay_s
from monoid_agent_kernel.providers._request_identity import (
    _REQUEST_DIGEST_GENERATION,
    _digest,
    _encoded_digest,
    _prompt_payload,
    _request_payload,
    effective_model_for,
)
from monoid_agent_kernel.providers._request_identity import (  # noqa: F401 - re-exported
    _PROMPT_DIGEST_GENERATION,
    _DigestResult,
    _model_identity,
    _prompt_terms,
    _tool_payload,
)

_LOGGER = logging.getLogger("monoid_agent_kernel.model_call")


DeltaConsumer = Callable[[ModelStreamChunk], None]
"""Receives every chunk of a streamed call, in order, as it arrives.

Deliberately every chunk and not a filtered subset: the two callers in the kernel want different
subsets -- a live stream relays all of them, an event-emitting run only turns text and reasoning
into events -- and a runner that filtered would have to know which caller it was serving. Filtering
is the consumer's business; ordering and completeness are this module's.
"""

ShouldAbort = Callable[[], bool]
"""Polled once per streamed chunk, after it has been delivered. See `ModelCallRunner.acall`."""


@dataclass(frozen=True)
class SettledModelCall:
    """One settled call, as `ModelCallRunner.settled_sink` receives it.

    ``receipt`` is always present -- success or failure, this is the audit fact. ``turn`` is the
    normalized provider turn on success and ``None`` on failure, because a failed call has no
    answer to record. ``request_preimage`` is the exact byte sequence ``request_digest`` was
    hashed over, present only when the runner was asked to capture it
    (`capture_request_preimage`) *and* a key was issued -- a refusal has no preimage because a
    preimage for an unissued key would be an identity for a call that does not exist.

    The preimage travels as the bytes the hasher consumed, never re-derived at the sink: the
    key's ``provider`` term comes from a resolution the runner performed against the adapter at
    call time, and re-resolving at the sink is the double-read that once let a receipt say
    ``openai`` while its key was taken under ``gateway``.
    """

    receipt: ModelCallReceipt
    request_preimage: bytes | None = None
    turn: Any | None = None


def _recordable_usage(usage: Mapping[str, Any]) -> dict[str, int]:
    """The usage counts a receipt will accept, dropping any the adapter got wrong.

    `ModelCallReceipt` refuses a negative or non-integer count, and rightly so -- usage is summed,
    so a negative silently subtracts from an aggregate. But raising *here* would fail a call the
    provider has already been paid for, on account of a counter nobody reads for control flow. So a
    malformed entry is omitted and the rest of the receipt still lands, the same trade
    `dispatch_model_call` makes when an observer raises.

    `bool` is excluded because it is an `int` subclass and a boolean token count is a bug, not a
    count of one -- the same rule the receipt applies.

    `type(value) is int` rather than `isinstance`, which is what its three siblings
    (`providers/base.py:provider_usage_of`, `providers/_common.py:usage_reported_by`,
    `core/model_io.py:ModelCallReceipt.with_error`) already spell. An `isinstance` here accepted
    every `int` subclass -- an `IntEnum` a provider SDK hands back as a token count is the real
    shape -- so one stamp read as a recordable usage on this path and as no usage at all on the
    three that consume it, and the receipt this function feeds would then reject what it accepted.
    Excluding `bool` is now implied, and kept spelled out because it is the case a reader checks
    for first.
    """

    return {
        key: value
        for key, value in usage.items()
        if isinstance(key, str)
        and not isinstance(value, bool)
        and type(value) is int
        and value >= 0
    }


def _kernel_retryable(exc: BaseException) -> bool:
    """Whether the kernel's own loop may pay for another attempt at this failure.

    Judged by the taxonomy, not by `retry_on`: that list is the adapter loop's
    provider-specific code selector (its defaults are gateway codes), while `retryable` is
    the cross-provider signal CONTRACTS names "automatic retry eligibility". Run boundaries
    (`RunCancelled`, `RunTimeout`, `ModelCallAborted`) are not `ModelAdapterError` and fall
    out structurally -- a run that stopped is not a failure to retry -- and
    `config_recoverable` refuses even when marked retryable, because re-sending cannot help
    a call whose config must change first.
    """

    if not isinstance(exc, ModelAdapterError):
        return False
    return bool(exc.retryable) and not bool(exc.config_recoverable)


def _turn_reported_retry(turn: ModelTurn) -> bool:
    """What the outcome object itself declared, probed rather than read as an attribute.

    A third-party adapter may return any turn-shaped object, and a missing flag means "did not
    retry", which is true of every adapter with no retry loop. One probe for its two readers --
    the receipt's whole-call fold in `_completed` and the answering attempt's log entry -- so
    the two cannot drift.
    """

    try:
        return bool(getattr(turn, "provider_retried", False))
    except Exception:
        return False


def _merged_usage(spent: Mapping[str, int], usage: Mapping[str, int]) -> dict[str, int]:
    """Key-wise sum of two already-normalized usage mappings.

    Both inputs arrive clean -- `provider_usage_of` and `_recordable_usage` apply the same
    exact-int rule -- so this is arithmetic, not validation. One helper for both settle
    exits, so success and failure cannot disagree about what a retried call cost.
    """

    merged = dict(usage)
    for key, value in spent.items():
        merged[key] = merged.get(key, 0) + value
    return merged


def _recovered_receipt(
    base: ModelCallReceipt,
    recovered: RecoveredModelDispatch,
) -> ModelCallReceipt:
    """Combine current private routing context with authoritative public settlement evidence."""

    try:
        evidence = recovered.receipt
        reservation = recovered.reservation
        return replace(
            base,
            request_digest=reservation.request_digest,
            digest_generation=reservation.digest_generation,
            digest_status="ok",
            idempotency_key=reservation.idempotency_key,
            stop_reason=evidence.get("stop_reason", ""),
            usage=dict(evidence.get("usage", {})),
            latency_ms=evidence.get("latency_ms", 0),
            attempts=evidence["attempts"],
            provider_retried=evidence.get("provider_retried", False),
            error_code=recovered.failure_code,
            provider_error_code=evidence.get("provider_error_code", ""),
            retryable=evidence.get("retryable", False),
            config_recoverable=evidence.get("config_recoverable", False),
            http_status=evidence.get("http_status"),
            attempt_log=(),
        )
    except Exception as exc:
        raise DurableModelCallError(
            "durable model invocation receipt is corrupt",
            error_code="durable_invocation_receipt_corrupt",
        ) from exc


def _recovered_result_matches_evidence(
    turn: ModelTurn,
    evidence: Mapping[str, Any],
) -> bool:
    """Compare only facts preserved by both durable projections.

    The public receipt describes the whole logical call, including usage absorbed by kernel
    retries and provider-retry evidence folded across attempts. The private result describes the
    final provider turn. Their usage and retry fields intentionally have different meanings. A
    public-safe stop reason, when present, is the one shared result fact that can be compared.
    Blob-address verification and ``durable_model_turn`` validate the private result itself.
    """

    return "stop_reason" not in evidence or (turn.stop_reason or "") == evidence["stop_reason"]


def _recovered_failure_can_retry(
    receipt: ModelCallReceipt,
    evidence: Mapping[str, Any],
    retry_plan: ModelRetryConfig | None,
) -> bool:
    """Whether a proven refusal can safely resume the current kernel retry loop."""

    return (
        retry_plan is not None
        and receipt.retryable
        and not receipt.config_recoverable
        and receipt.attempts < retry_plan.max_attempts
        and evidence.get("stream_committed") is False
    )


def _safe_repr(value: Any) -> str:
    """A printable stand-in for an object that may refuse to describe itself.

    The last step of the capture surface's fallback, and the one that made the fallback not a
    fallback: an object with no ``__dict__`` for ``vars()`` *and* a ``__repr__`` that raises took the
    exception all the way out through ``_publish``, discarding a turn the provider had already
    produced. A display surface must not be able to fail a call that already happened -- which is the
    reason the fallback exists in the first place.

    The type's name is read in its own attempt, because that is an attribute lookup on the type and a
    metaclass can make it raise too.
    """

    try:
        return repr(value)
    except Exception:
        pass
    try:
        return f"<unrepresentable {portable_type_name(value)}>"
    except Exception:
        return "<unrepresentable>"


def _copy_with_fields(value: Any, /, **changes: Any) -> Any:
    """Shallow-copy an extension value without invoking its convenience constructor."""

    cloned = copy(value)
    for name, replacement in changes.items():
        object.__setattr__(cloned, name, replacement)
    return cloned


# The minter itself lives in ``providers.base`` beside the rule it satisfies, because the
# runner is not its only caller: the reference gateway's service keys the upstream hop it
# drives, which has a retry loop of its own. One expression, both issuers.


def _normalize_invocation_context(context: InvocationContext) -> InvocationContext:
    attempt = context.attempt
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("invocation attempt must be an integer greater than zero")
    normalized_attributes = normalize_json_ingress(dict(context.attributes))
    changes = {
        "run_id": normalize_unicode_scalars(context.run_id),
        "skill_id": normalize_unicode_scalars(context.skill_id),
        "skill_digest": normalize_unicode_scalars(context.skill_digest),
        "step_id": normalize_unicode_scalars(context.step_id),
        "attempt": attempt,
        "batch_id": normalize_unicode_scalars(context.batch_id),
        "item_id": normalize_unicode_scalars(context.item_id),
        "case_id": normalize_unicode_scalars(context.case_id),
        "traceparent": normalize_unicode_scalars(context.traceparent),
        "tracestate": normalize_unicode_scalars(context.tracestate),
        "attributes": normalized_attributes,
    }
    try:
        return _copy_with_fields(context, **changes)
    except Exception:
        return InvocationContext(**changes)


def _call_content(request: ModelRequest, turn: ModelTurn | None) -> dict[str, Any]:
    """What an observer may be shown of one call, before any redaction.

    Keys are stable field names rather than one blob so a `RedactionPolicy` can name a field and a
    consumer can be given some fields and not others.

    `observations` are here because they are model *input*: in the by-reference request shape they
    carry the tool results the model is being shown. Omitting them did not merely give a `full`
    observer an incomplete picture -- it routed tool output around redaction entirely, since a
    policy can only mask a field it is handed. That is a disclosure hole, not a gap in coverage.

    `messages` is always a list here, even when the request carried `None` -- deliberately unlike
    `_prompt_payload`, which keeps those apart because a replay key must. This is a display surface,
    not a key: `messages` and `observations` are the two fields a consumer *walks*, and normalising
    both means neither has to special-case the by-reference shape to learn nothing, since that shape
    is already legible from `previous_turn_handle` and `observations` in this same dict. The scalars
    are not normalised to match and do not need to be -- `instruction` is handed through as `None` in
    exactly that by-reference case, because nothing iterates it.

    Not a claim that `None` would break redaction -- it does not. `RedactionPolicy` holds rules and
    answers questions (`names_a_secret`, `redact_text`); the walking is a `Redactor`'s, and the
    shipped `DefaultRedactor` returns `None` untouched. An earlier version of this note said
    otherwise and was wrong about both halves.
    """

    try:
        content: dict[str, Any] = {
            "system_prompt": request.system_prompt,
            "instruction": request.instruction,
            "messages": list(request.messages or ()),
            "observations": [observation.to_json() for observation in request.observations],
            "previous_turn_handle": request.previous_turn_handle or "",
        }
        if turn is not None:
            content["output_text"] = getattr(turn, "final_text", "") or ""
            # Probed and per-call, so one tool call the adapter built oddly costs its own entry rather
            # than the whole record. A `__slots__` object has no `__dict__` and used to raise from here,
            # which is a display surface failing a call that already happened.
            calls: list[dict[str, Any]] = []
            for call in getattr(turn, "tool_calls", ()) or ():
                try:
                    calls.append(dict(vars(call)))
                except Exception:
                    calls.append({"repr": _safe_repr(call)})
            content["tool_calls"] = calls
        return normalize_json_ingress(content)
    except Exception:
        # Capture is diagnostic.  A malformed custom object may reduce it to metadata-only,
        # while the provider result and its receipt still reach the caller.
        return {}


def _discard_hook(adapter: Any, request: Any) -> Any:
    """The adapter's chance to take back an answer the run threw away, or None.

    A run boundary that wins over a COMPLETED call discards a real result. For a
    stateless adapter that is nothing; for one holding shared state it is a silent
    corruption. The replay adapter advances a per-key cursor when it hands an answer
    over, so a discarded call permanently consumes a recorded answer and every later
    consumer of that corpus is shifted by one -- a structurally valid turn belonging to
    a different call, which is the substitution the replay work exists to prevent.

    Optional and duck-typed, so no adapter is obliged to care: absent means the old
    behaviour exactly.
    """

    hook = getattr(adapter, "discard_turn", None)
    if not callable(hook) or request is None:
        return None

    def discarded(turn: Any) -> None:
        hook(request, turn)

    return discarded


@dataclass
class ModelCallRunner:
    """Runs one model call against an adapter, whatever shape that adapter is.

    Four dispatch shapes reach the same semantics -- a streamed call relayed to a consumer,
    `anext_turn`, a coroutine `next_turn`, and a blocking `next_turn` on an abandonable daemon
    thread. All four go through the same cancel/deadline race, so an adapter's async-ness never
    changes when a run stops. A streaming-capable adapter called *without* a consumer is not a
    fifth shape: it takes one of the other three, because streaming is selected by the argument.

    Which shape is used is a function of the **call arguments**, never of state held elsewhere: a
    streamed drive happens when the caller passes a `delta_consumer` and the adapter can stream.
    That is what makes the runner testable in isolation and what stopped path selection from
    depending on whether some other object's queue happened to be active.
    """

    adapter: Any
    """The model adapter. Typed `Any` because the optional members are probed with `getattr`, as
    the protocols in `providers/base.py` intend -- declaring the optional members would make them
    required for structural typing and reject a third-party adapter implementing only `next_turn`.

    Used when `current_adapter` is unset, which is the standalone case: a caller holding one adapter
    for the runner's lifetime passes it here and never thinks about the distinction."""

    current_adapter: Callable[[], Any] | None = None
    """Returns the adapter to call **at the moment of the call**, for a host whose adapter can change
    between calls. `AgentLoop.model_adapter` is a public mutable field, and the loop reads it live
    everywhere *else* -- for `supports_multimodal`, for `wire_image_encoding`, and for the
    `provider_name` it attributes the answer to. A runner holding a snapshot therefore did not merely
    go stale: the request was shaped for and attributed to one adapter while another answered it."""

    current_cancellation_token: Callable[[], CancellationToken | None] | None = None
    """Returns the token to observe **at the moment of the call**, not a token captured at
    construction. `AgentLoop.astream` installs a token lazily on a run already in progress, so a
    runner holding a snapshot would watch a token nobody cancels and silently lose cancellation on
    the streaming path."""

    cancel_grace_s: float = 1.0
    """How long an abandoned call's worker is given to settle before it is reported as leaked.

    Used when `current_cancel_grace_s` is unset, the same way `adapter` is."""

    current_cancel_grace_s: Callable[[], float] | None = None
    """Returns the grace to apply **at the moment it is spent**, for the same reason
    `current_adapter` exists: `AgentLoop.async_model_cancel_grace_s` is a public mutable field, and
    its tool-side twin is still read live at every use. A snapshot made the model and tool halves of
    one knob disagree -- a grace raised after `open()` abandoned the model worker on the old value,
    ~150x early in the case that found it."""

    thread_name: str = "nar-model-call"
    """Thread name for a blocking `next_turn`. Carries the run id in kernel use, so a leaked worker
    in a thread dump names the run that leaked it."""

    subscriptions: Sequence[ModelIOSubscription] = ()
    """Observers of settled calls. Empty by default -- *delivery* of content is opt-in.

    Identifying the call is not: `prompt_digest` and `request_digest` are computed on every call,
    before anything looks at this field, because they describe the call whether or not anyone is
    watching. That costs two canonical-JSON encodes and two SHA-256 passes over the serialized
    prompt per call, which is why `MAX_MODEL_PAYLOAD_BYTES` bounds it. An `AgentLoop` with no model-I/O
    subscriptions still pays that; anyone trimming the cost should start there and not expect this
    field to gate it."""

    settled_sink: Callable[[SettledModelCall], None] | None = None
    """Where a settled call is recorded, as opposed to *delivered*.

    Separate from `subscriptions` rather than a subscription of its own, for three reasons. A
    subscription is governed by a `CapturePolicy`, and the narrowing a `none`-mode policy applies
    would strip the very digests a durable record exists to carry. Registering one would also defeat
    the `if self.subscriptions` gate below, so every call would assemble `_call_content` and hash
    every field of it to satisfy a consumer that reads none of that. And a kernel-owned recorder
    inserting itself into the host's own observer list is a thing the host can see and close.

    The reason a caller cannot do this from `acall`'s return value: a failed call publishes its
    receipt and re-raises without stamping it on the exception, so the return value carries receipts
    for successful calls only. A ledger built on it would record everything except the failures --
    which are exactly what an audit trail is for.

    ONE sink receiving the whole settled call -- receipt, optional preimage, optional turn --
    rather than a receipt sink beside a payload sink. Two sinks would put the receipt ledger and
    the payload corpus for one call in two deliveries, and the recorder behind them could only
    keep their line indices agreeing by cooperation across two lock acquisitions -- the exact
    index race W6-1 fixed *inside* the recorder, reopened one seam higher. A superset delivery
    makes the agreement structural: one call, one delivery, one reservation. (This superseded the
    unreleased `receipt_sink`, whose guarantees the tests re-prove on this seam.)

    Read once per call, like `subscriptions`, and expected not to raise; one that does is contained
    the same way an observer is."""

    capture_request_preimage: bool = False
    """Whether the sink's `SettledModelCall.request_preimage` is populated.

    Off by default because the preimage is the encoded request -- up to `MAX_MODEL_PAYLOAD_BYTES`
    held in memory per in-flight call -- and a sink that only files receipts (the ledger without
    the payload corpus) should not pay for bytes it never reads. The wiring that enables the
    payload recorder sets this; the digests themselves are computed either way."""

    lifecycle_hook: ModelCallLifecycleHook | None = None
    """Optional authoritative lifecycle writer for durable paid-call execution.

    The hook is synchronous because the hosting contracts it adapts are synchronous fenced
    mutations. It is opt-in and independent of ``settled_sink``: lifecycle writes control whether
    adapter work may begin or be retried, while the existing sink remains passive evidence
    delivery. Durable mode requires an explicit ``logical_call_id`` on :meth:`acall`.
    """

    def _effective_model(
        self,
        request: ModelRequest,
        adapter: Any,
    ) -> tuple[ModelConfig, ModelConfig | None]:
        """Delegates to :func:`providers._request_identity.effective_model_for`.

        A wrapper and nothing more, pinned as such: the receipt's resolution and a replay
        lookup's must be one implementation, and a body regrown here would be the twin the
        move to `_request_identity` closed.
        """

        return effective_model_for(request, adapter)

    def _resolved_destination(self, model: ModelConfig, adapter: Any) -> tuple[str, str]:
        """Where this adapter would send a call under `model`, and WHICH outcome that was.

        Probed and tolerant of failure for the same reason every other probe here is: bookkeeping,
        and an adapter that cannot answer must not thereby lose its call.

        The `getattr` is inside the `try`, not before it. Tolerating only the *call* left the
        *lookup* undefended, so an adapter exposing `resolve_destination` as a property that raised
        still lost its call -- the one shape the rule above exists to rule out, surviving in the
        probe the other two were written to imitate.

        Returning a status alongside the value is the fix for what tolerance used to cost. Three
        outcomes answered `""`: an adapter with no destination concept, one that answered with
        nothing, and one whose probe raised. The third is not a shrug -- the shipped gateway
        resolver raises *deterministically* when no URL is configured anywhere, so it usually means
        a deployment whose every call is about to fail. All three produced a valid-looking key that
        could not be told from the others, which is exactly the shape `_digest`'s own docstring
        rules out for itself: refusing is safe, but a fabricated answer returns the wrong call.
        """

        try:
            resolve = getattr(adapter, "resolve_destination", None)
            if not callable(resolve):
                return "", "not_declared"
            value = normalize_unicode_scalars(str(resolve(model) or ""))
        except Exception:
            return "", "unavailable"
        return (value, "resolved") if value else ("", "declined")

    def _token(self) -> CancellationToken | None:
        return (
            None if self.current_cancellation_token is None else self.current_cancellation_token()
        )

    def _current_adapter(self) -> Any:
        """The adapter for one call. Read **once** per call and threaded through from there.

        Reading it again per probe would let a host that swaps adapters mid-call produce a receipt
        describing a mixture of two, which is worse than describing either. One read per call is also
        exactly what the loop did before this runner existed.
        """

        return self.adapter if self.current_adapter is None else self.current_adapter()

    def _grace_s(self) -> float:
        """The grace, read where it is spent rather than per call: a call has no single moment the
        interval belongs to, and the tool half is read live at each use too."""

        return (
            self.cancel_grace_s
            if self.current_cancel_grace_s is None
            else self.current_cancel_grace_s()
        )

    def _check_cancel_or_deadline(self, deadline: float | None) -> None:
        """Check only terminal run boundaries while model I/O is in flight.

        A cooperative abort is not checked here. It is a per-chunk question on the streaming path
        and has no meaning for a one-shot call, which cannot be stopped part-way.
        """

        token = self._token()
        if token is not None and token.requested:
            raise RunCancelled("run cancelled")
        if deadline is not None and time.time() >= deadline:
            raise RunTimeout("run exceeded max duration")

    async def acall(
        self,
        request: ModelRequest,
        *,
        context: InvocationContext | None = None,
        deadline: float | None = None,
        should_abort: ShouldAbort | None = None,
        delta_consumer: DeltaConsumer | None = None,
        logical_call_id: str = "",
    ) -> tuple[ModelTurn, ModelCallReceipt]:
        """Run one call and return the turn with the receipt that describes it.

        `deadline` is absolute and run-scoped; exceeding it raises `RunTimeout`. Cancellation raises
        `RunCancelled`. Both are terminal run boundaries and are raced against the call itself, so a
        wedged provider cannot outlast them.

        `should_abort` is polled once per chunk **after** that chunk has been delivered, and only on
        the streamed path. Delivering first is the observable rule: a stop arriving while a chunk is
        in flight does not retract that chunk, it stops the one after it. Aborting raises
        `ModelCallAborted`.

        A receipt is produced whether the call succeeded or failed -- a failed call is exactly the
        one an audit trail needs -- and is delivered to every subscription before the exception is
        re-raised.

        ``logical_call_id`` is required only when ``lifecycle_hook`` is configured. It is the
        caller-owned durable address of this call; standalone anonymous calls cannot invent a
        stable address across process restore.
        """

        started = time.monotonic()
        lifecycle_hook = self.lifecycle_hook
        adapter = self._current_adapter()
        # Same tolerance as the other two adapter probes, and for the same reason. Undefended, a
        # `provider_name` property that raised -- or whose `str()` did -- lost the call before the
        # adapter was ever invoked, over a field nothing reads for control flow.
        try:
            provider = normalize_unicode_scalars(str(getattr(adapter, "provider_name", "") or ""))
        except Exception:
            provider = ""
        try:
            provisional_context = _normalize_invocation_context(
                context if context is not None else InvocationContext()
            )
        except Exception:
            provisional_context = InvocationContext()
        try:
            raw_model = getattr(request, "model", None)
            provisional_model = (
                normalize_model_config(raw_model)
                if isinstance(raw_model, ModelConfig)
                else ModelConfig()
            ) or ModelConfig()
        except Exception:
            provisional_model = ModelConfig()
        receipt = ModelCallReceipt(
            context=provisional_context,
            model=provisional_model,
            provider_name=provider,
        )
        # Assigned once the request key is taken; a failure before that point hands the sink
        # `None`, which is truthful -- there is no preimage for a call that never got a key.
        request_preimage: bytes | None = None
        with collect_retry_reports() as progress:
            # How many times the kernel reached into the adapter, which is what `attempts`
            # counts. A run already cancelled or past its deadline is refused below without
            # the adapter being touched and reports 0 -- the receipt used to carry the
            # default `attempts=1` there, telling a consumer summing the field that provider
            # work happened when none did. Under the kernel retry layer each re-dispatch
            # counts one more.
            attempts_made = 0
            # Usage the loop's swallowed attempts already paid for; merged into whichever
            # receipt settles this call, success or failure, by the same helper.
            spent_usage: dict[str, int] = {}
            # One entry per dispatch, appended at the same commit points the counters use:
            # absorbed attempts at the absorb line below, the terminal one at whichever exit
            # settles the call. Initialized beside `spent_usage` because the failure exit
            # reads both for calls refused before the loop was ever entered.
            attempt_log: list[ModelCallAttempt] = []
            # A recovered public receipt carries aggregate usage but intentionally omits the
            # private per-attempt log. Once recovery resumes a kernel retry, the final receipt
            # keeps that log empty rather than fabricating the missing historical entries.
            attempt_log_complete = True
            last_attempt_entry: ModelCallAttempt | None = None
            # The measured wait that preceded the NEXT dispatch: 0 until a backoff actually
            # runs, then re-measured after each one. Threaded into every entry-construction
            # site so the wait lands on the entry it delayed, not the one that caused it.
            pending_backoff_ms = 0
            recovered_failure_receipt: ModelCallReceipt | None = None
            durable_outcome_receipt: ModelCallReceipt | None = None
            elapsed_before_recovery_ms = 0
            try:
                if lifecycle_hook is not None and not is_safe_opaque_id(logical_call_id):
                    raise DurableModelCallError(
                        "durable model calls require an explicit bounded logical_call_id",
                        error_code="durable_invocation_identity_required",
                    )
                # Before dispatch, not only inside the race. `_aawait` reports a boundary that had
                # already been crossed, but by then the adapter has been invoked and the provider has
                # been paid for work the run had already decided not to do. Checking here also covers
                # the interval the caller cannot: building the receipt digests above happens between
                # the caller's own boundary check and this line, so a deadline can expire in between.
                #
                # Durable mode performs synchronous reserve/start commits between this check and
                # adapter entry. A boundary crossed during those commits is caught by the race once
                # provider work starts; the host must keep those mutations bounded.
                #
                # Lifted out of `_adrive` so that refusing the call and dispatching it are
                # distinguishable here; `_adrive` is called from nowhere else, so the check still
                # exists once.
                self._check_cancel_or_deadline(deadline)
                request = normalize_model_request(request)
                normalized_context = _normalize_invocation_context(
                    context if context is not None else InvocationContext()
                )
                model, dispatch_model = self._effective_model(request, adapter)
                # The kernel loops only when the effective config assigns it the loop.
                retry_plan = model.retry if model.retry.layer == "kernel" else None
                if retry_plan is not None:
                    # The dispatch copy is neutralized -- `max_attempts=1` -- so any
                    # config-honoring adapter cannot loop under this layer even if it never
                    # learned `layer` exists; the layer value itself still travels so an
                    # adapter whose loop lives outside the config (the OpenAI SDK) can
                    # comply on its own. The receipt is keyed from `model`, not this copy,
                    # so it describes the call as configured -- and the replay key excludes
                    # the retry block entirely, so neither the layer nor the neutralization
                    # can move it.
                    #
                    # Copied field-wise rather than through `dataclasses.replace`, for the
                    # reason `_copy_with_fields` exists: a public extension config with a
                    # narrower convenience constructor is supported everywhere else -- ingress
                    # normalization rewrites it this same way -- and `replace` would dispatch
                    # back through that constructor with every inherited field, raising
                    # `TypeError` before the adapter is reached. Only this layer rewrites a
                    # config after ingress, so only this layer could refuse one.
                    dispatch_model = _copy_with_fields(
                        dispatch_model if dispatch_model is not None else model,
                        retry=_copy_with_fields(model.retry, max_attempts=1),
                    )
                # One copy for both call-scoped rewrites below. ``normalize_model_request``
                # already returned a fresh instance, but the explicit copy keeps the rule
                # visible: nothing past this line writes onto a value the caller holds.
                request = copy(request)
                # The call is KEYED here, in the same breath as its digests below: one fresh
                # token per call, before the first dispatch, so every kernel re-dispatch (the
                # loop reuses this request) and every adapter-internal retry (the gateway
                # rebuilds only headers) presents the same value. The runner is the single
                # issuer -- a caller-supplied value is overwritten, because respecting it
                # would let one request object hand two calls the same retry scope, the
                # collision per-call issuance exists to prevent. Issuance is uniform across
                # adapters; presenting the token on a wire is the gateway transport's alone.
                object.__setattr__(request, "idempotency_key", new_idempotency_key())
                if dispatch_model is not None:
                    object.__setattr__(request, "model", dispatch_model)
                where, destination_status = self._resolved_destination(model, adapter)
                digest_result = _encoded_digest(
                    payload=_request_payload(
                        request,
                        model,
                        # The RESOLVED provider, not the raw declaration `provider_name` records
                        # above: the key must say who actually served the call. Declaration-only
                        # collided a fake adapter with a gateway built without one -- both declare
                        # nothing -- and `ModelConfig.provider` alone separated a direct call from
                        # a gateway relaying the same upstream, which is the one pair a corpus
                        # wants sharing a key. It also normalizes, which matters here because
                        # `provider` is the only `ModelConfig` field with no ingress validation.
                        #
                        # Resolved from the declaration THIS CALL ALREADY READ, handed in rather
                        # than probed again. The adapter is read once per call for this reason
                        # already; the declaration on it was still read twice, and a `provider_name`
                        # that answers once and then raises made the two disagree -- the receipt
                        # saying `openai` while the key had been taken under the config's `gateway`.
                        # A key whose preimage the record contradicts is the exact defect that took
                        # the destination out of this payload.
                        provider=resolved_provider_name(adapter, model, declared=provider) or "",
                    ),
                    want_preimage=self.capture_request_preimage,
                )
                request_preimage = digest_result.preimage
                receipt = replace(
                    receipt,
                    context=normalized_context,
                    model=model,
                    prompt_digest=_digest(_prompt_payload(request)),
                    request_digest=digest_result.digest,
                    digest_generation=_REQUEST_DIGEST_GENERATION,
                    # Key and status arrive as one result object: the two fields describing one
                    # encoder decision cannot be computed twice and disagree, and the encoder is
                    # the only party that can tell `absent` from `too_large`.
                    digest_status=digest_result.status,
                    destination_status=destination_status,
                    destination_digest=destination_digest(where),
                    idempotency_key=request.idempotency_key,
                )
                if lifecycle_hook is not None and digest_result.status != "ok":
                    raise DurableModelCallError(
                        "durable model call request could not be keyed",
                        error_code="durable_invocation_unkeyable",
                    )
                if lifecycle_hook is not None:
                    recovered = recover_model_dispatch(
                        lifecycle_hook,
                        logical_call_id=logical_call_id,
                        request_digest=receipt.request_digest,
                    )
                    if recovered is not None:
                        object.__setattr__(
                            request,
                            "idempotency_key",
                            recovered.reservation.idempotency_key,
                        )
                        try:
                            receipt = _recovered_receipt(receipt, recovered)
                        except DurableModelCallError as recovery_error:
                            mark_recovered_model_usage(recovery_error, recovered.receipt)
                            raise
                        if recovered.failure_code:
                            recovered_failure_receipt = receipt
                            recovered_error = ModelDispatchRefused(
                                "durable model dispatch restored a settled refusal",
                                error_code=recovered.failure_code,
                                provider_error_code=receipt.provider_error_code,
                                retryable=receipt.retryable,
                                config_recoverable=receipt.config_recoverable,
                                http_status=receipt.http_status,
                                provider_retried=receipt.provider_retried,
                            )
                            if not _recovered_failure_can_retry(
                                receipt,
                                recovered.receipt,
                                retry_plan,
                            ):
                                raise recovered_error
                            delay = retry_delay_s(
                                receipt.attempts,
                                retry_plan.initial_delay_s,
                                retry_plan.max_delay_s,
                                retry_plan.backoff_multiplier,
                                retry_plan.jitter_s,
                            )
                            if deadline is not None and time.time() + delay >= deadline:
                                raise recovered_error
                            attempts_made = receipt.attempts
                            spent_usage = dict(receipt.usage)
                            elapsed_before_recovery_ms = receipt.latency_ms
                            attempt_log_complete = False
                            receipt = replace(
                                receipt,
                                stop_reason="",
                                usage={},
                                latency_ms=0,
                                error_code="",
                                provider_error_code="",
                                retryable=False,
                                config_recoverable=False,
                                http_status=None,
                                attempt_log=(),
                            )
                            recovered_failure_receipt = None
                            backoff_started = time.monotonic()
                            await self._abackoff(delay, deadline)
                            pending_backoff_ms = (
                                self._ms_since(backoff_started) if delay > 0 else 0
                            )
                        else:
                            try:
                                turn = durable_model_turn(recovered.result_blob)
                            except DurableModelCallError as recovery_error:
                                mark_recovered_model_usage(recovery_error, recovered.receipt)
                                raise
                            if not _recovered_result_matches_evidence(turn, recovered.receipt):
                                recovery_error = DurableModelCallError(
                                    "durable model result conflicts with its receipt",
                                    error_code="durable_invocation_result_corrupt",
                                )
                                mark_recovered_model_usage(recovery_error, recovered.receipt)
                                raise recovery_error
                            settled = self._publish(
                                request,
                                turn,
                                receipt,
                                elapsed_ms=receipt.latency_ms,
                                request_preimage=request_preimage,
                            )
                            return turn, settled
                consumer = delta_consumer
                delivered = False
                # Installed for any consumer, not only under the kernel's loop. The flag is
                # *used* where a retry window exists -- delivery closes it -- but it is
                # *recorded* on every call, and `layer` defaults to `"adapter"`: gating the
                # wrapper on the loop that reads the flag wrote `stream_committed: false`
                # onto every shipped streaming call's ledger line while the consumer was
                # holding its chunks. The key is present either way, so a reader cannot tell
                # a definite "nothing was delivered" from "this arm never answered".
                if delta_consumer is not None:
                    inner_consumer = delta_consumer

                    def _marking_consumer(chunk: ModelStreamChunk) -> None:
                        # Delivery is what closes the retry window (see the loop below),
                        # and delivery means the consumer received it -- the same line the
                        # `acall` docstring draws for `should_abort`.
                        nonlocal delivered
                        delivered = True
                        inner_consumer(chunk)

                    consumer = _marking_consumer
                reservation: ModelDispatchReservation | None = None
                while True:
                    next_attempt = attempts_made + 1
                    if lifecycle_hook is not None:
                        reservation = reserve_model_dispatch(
                            lifecycle_hook,
                            logical_call_id=logical_call_id,
                            dispatch_attempt=next_attempt,
                            request_digest=receipt.request_digest,
                            idempotency_key=request.idempotency_key,
                        )
                        # A restored reservation owns the key. The request digest excludes this
                        # carriage field, so replacing it does not invalidate the identity already
                        # checked above.
                        object.__setattr__(
                            request, "idempotency_key", reservation.idempotency_key
                        )
                        receipt = replace(
                            receipt,
                            idempotency_key=reservation.idempotency_key,
                        )
                        # The commit sits immediately before adapter entry. A hook failure leaves
                        # attempts_made unchanged, so the receipt does not claim provider work.
                        lifecycle_hook.dispatch_started(reservation)
                    attempts_made = next_attempt
                    reports_before = progress.count
                    attempt_started = time.monotonic()
                    try:
                        turn = await self._adrive(
                            request, deadline, should_abort, consumer, adapter
                        )
                        if lifecycle_hook is not None:
                            # Durable mode classifies a malformed terminal inside the started
                            # dispatch. The default path keeps its historical settle accounting
                            # below unchanged.
                            turn = normalize_model_turn(turn)
                        break
                    except BaseException as exc:
                        # Every fact the entry needs, read through `with_error` on a throwaway
                        # receipt -- the one census-pinned reader of what an exception carries,
                        # so the log cannot drift from the receipt built beside it. Probed HERE,
                        # before the outer handler's whole-call `mark_provider_retried` can
                        # colour the exception: what this reads is attempt-scoped.
                        probe = ModelCallReceipt().with_error(exc)
                        last_attempt_entry = ModelCallAttempt(
                            index=attempts_made,
                            elapsed_ms=self._ms_since(attempt_started),
                            error_code=probe.error_code,
                            provider_error_code=probe.provider_error_code,
                            retryable=probe.retryable,
                            config_recoverable=probe.config_recoverable,
                            http_status=probe.http_status,
                            provider_retried=probe.provider_retried
                            or progress.count > reports_before,
                            usage=probe.usage,
                            stream_committed=delivered,
                            backoff_ms=pending_backoff_ms,
                        )
                        if (
                            lifecycle_hook is not None
                            and reservation is not None
                            and isinstance(exc, Exception)
                        ):
                            if dispatch_evidence(exc) != "refused":
                                raise_model_dispatch_unknown(
                                    lifecycle_hook,
                                    reservation,
                                    exc,
                                    failure_code=safe_failure_code(
                                        probe.error_code,
                                        default="dispatch_unknown",
                                    ),
                                    usage=(
                                        _merged_usage(spent_usage, probe.usage)
                                        if spent_usage
                                        else probe.usage
                                    ),
                                )
                            current_failure = receipt.with_error(exc)
                            durable_failure = replace(
                                current_failure,
                                attempts=attempts_made,
                                latency_ms=(
                                    elapsed_before_recovery_ms + self._ms_since(started)
                                ),
                                provider_retried=(
                                    current_failure.provider_retried or progress.retried
                                ),
                                attempt_log=(
                                    (*attempt_log, last_attempt_entry)
                                    if attempt_log_complete
                                    else ()
                                ),
                                usage=(
                                    _merged_usage(spent_usage, current_failure.usage)
                                    if spent_usage
                                    else current_failure.usage
                                ),
                            )
                            durable_outcome_receipt = durable_failure
                            settle_model_dispatch(
                                lifecycle_hook,
                                reservation,
                                durable_failure,
                                failure_code=durable_failure.error_code,
                                stream_committed=delivered,
                            )
                        if (
                            retry_plan is None
                            or attempts_made >= retry_plan.max_attempts
                            or delivered
                            or not _kernel_retryable(exc)
                        ):
                            raise
                        delay = retry_delay_s(
                            attempts_made,
                            retry_plan.initial_delay_s,
                            retry_plan.max_delay_s,
                            retry_plan.backoff_multiplier,
                            retry_plan.jitter_s,
                        )
                        # Sleeping into a boundary would waste wall clock the run has
                        # already spent and then mask the provider failure with a
                        # `RunTimeout` that names nothing; the transient error itself is
                        # the better answer.
                        if deadline is not None and time.time() + delay >= deadline:
                            raise
                        # What the absorbed attempt already cost, recorded only once the loop
                        # has committed to absorbing it -- which is here, past every `raise`
                        # that would make THIS error the call's outcome. Recorded any earlier,
                        # the error the deadline check re-raises is both the swallowed attempt
                        # and the terminal one, and `receipt.with_error` reads its stamp again:
                        # a single billed call landing on the receipt twice. The log entry
                        # commits at this same line for the same reason: appended earlier, the
                        # attempt the deadline check re-raises would be logged as absorbed AND
                        # as terminal, and the log would name one dispatch twice. Past this
                        # line the receipt is the only carrier left -- a boundary raised by
                        # the wait below reports itself, not the provider failure it
                        # interrupted.
                        spent_usage = _merged_usage(spent_usage, probe.usage)
                        if attempt_log_complete:
                            attempt_log.append(last_attempt_entry)
                        # Measured, not copied from the schedule -- a capped sleep must record
                        # what happened. Only when a wait was actually requested: for a zero
                        # schedule the boundary check is not a backoff, and timing it would
                        # stamp scheduler noise onto a wait that never ran.
                        backoff_started = time.monotonic()
                        await self._abackoff(delay, deadline)
                        pending_backoff_ms = (
                            self._ms_since(backoff_started) if delay > 0 else 0
                        )
                if lifecycle_hook is None:
                    turn = normalize_model_turn(turn)
            except BaseException as exc:
                # Process-control exceptions preserve already-observed accounting but never become
                # model outcomes or lifecycle compensation. Prefer a complete receipt prepared
                # immediately before durable settle. If the stop arrived earlier, carry the usage
                # of completed attempts without constructing an attempt log whose terminal entry
                # would falsely describe the stop as a provider result. Every diagnostic action is
                # contained so the original KeyboardInterrupt/SystemExit-shaped signal survives.
                if not isinstance(exc, (Exception, asyncio.CancelledError)):
                    crash_receipt = durable_outcome_receipt
                    if crash_receipt is None:
                        crash_usage = dict(spent_usage)
                        if (
                            last_attempt_entry is not None
                            and last_attempt_entry.index == attempts_made
                        ):
                            crash_usage = _merged_usage(crash_usage, last_attempt_entry.usage)
                        with contextlib.suppress(Exception):
                            crash_receipt = replace(
                                receipt.with_error(exc),
                                attempts=attempts_made,
                                attempt_log=(),
                                usage=crash_usage,
                            )
                    if crash_receipt is not None:
                        with contextlib.suppress(Exception):
                            self._publish(
                                request,
                                None,
                                crash_receipt,
                                elapsed_ms=(
                                    elapsed_before_recovery_ms + self._ms_since(started)
                                ),
                                request_preimage=request_preimage,
                            )
                        if crash_receipt.usage:
                            mark_provider_usage(exc, crash_receipt.usage)
                    raise
                if isinstance(exc, ModelEvidenceUncommitted):
                    # The paid dispatch and its canonical result/refusal are already settled.
                    # Required evidence is a separate recovery lane: do not publish a fabricated
                    # failed model call or let the kernel retry loop absorb this infrastructure
                    # outcome as another provider attempt.
                    raise
                if recovered_failure_receipt is not None:
                    with contextlib.suppress(Exception):
                        self._publish(
                            request,
                            None,
                            recovered_failure_receipt,
                            elapsed_ms=recovered_failure_receipt.latency_ms,
                            request_preimage=request_preimage,
                        )
                    if recovered_failure_receipt.usage:
                        mark_provider_usage(exc, recovered_failure_receipt.usage)
                    raise
                # What the adapter managed to say before this call stopped producing outcomes. A
                # boundary raised by the race is not something the adapter can stamp, and an
                # abandoned worker's eventual exception is never read, so for a call the run gave
                # up on this is the only surviving evidence that a retry happened.
                if progress.retried:
                    mark_provider_retried(exc)
                # Guarded, because the docstring promises the receipt is delivered *before the
                # exception is re-raised* -- and a raising observer made it delivered *instead of*
                # it, replacing a `ModelAdapterError` carrying `retryable` and `http_status` with
                # whatever the observer threw. Turning capture on must not change how a provider
                # failure is classified. `Exception` and not `BaseException`: a KeyboardInterrupt
                # arriving during delivery should still stop everything.
                failed = replace(receipt.with_error(exc), attempts=attempts_made)
                # The terminal entry: the attempt whose outcome this exception is -- unless it
                # interrupted the WAIT between attempts (a backoff boundary), in which case
                # every dispatched attempt is already logged and the receipt alone carries the
                # boundary's taxonomy. A refused call never dispatched, so its log stays empty
                # beside `attempts == 0`.
                if attempt_log_complete and attempts_made > len(attempt_log):
                    if (
                        last_attempt_entry is not None
                        and last_attempt_entry.index == attempts_made
                    ):
                        attempt_log.append(last_attempt_entry)
                    else:
                        # The failure happened between the dispatch's return and the settle
                        # (the normalizer refused the turn), so no attempt-scoped probe exists;
                        # the entry mirrors the receipt, which extracted the same facts from
                        # the same exception -- every fact except one. ``provider_retried`` on
                        # that receipt is the whole CALL's fold: the handler above stamps
                        # ``progress.retried`` onto the escaping error before ``with_error``
                        # reads it, so mirroring it here would credit this dispatch with a
                        # report an earlier absorbed one made. Counted on this attempt's own
                        # channel instead, the way both sibling construction sites count it.
                        # The refused turn's own declaration is not consulted: the normalizer
                        # rejected that turn, and a fact read off a value the call refused is
                        # not a fact the call may report.
                        attempt_log.append(
                            ModelCallAttempt(
                                index=attempts_made,
                                elapsed_ms=self._ms_since(attempt_started),
                                error_code=failed.error_code,
                                provider_error_code=failed.provider_error_code,
                                retryable=failed.retryable,
                                config_recoverable=failed.config_recoverable,
                                http_status=failed.http_status,
                                provider_retried=progress.count > reports_before,
                                usage=failed.usage,
                                stream_committed=delivered,
                                backoff_ms=pending_backoff_ms,
                            )
                        )
                # One ``replace`` for the same reason the answering path takes one: the entries
                # must sum to the usage they are entries for, and a two-step build holds a
                # receipt that does not -- which ``__post_init__`` now refuses.
                failed = replace(
                    failed,
                    attempt_log=tuple(attempt_log),
                    usage=_merged_usage(spent_usage, failed.usage) if spent_usage else failed.usage,
                )
                with contextlib.suppress(Exception):
                    self._publish(
                        request,
                        None,
                        failed,
                        elapsed_ms=elapsed_before_recovery_ms + self._ms_since(started),
                        request_preimage=request_preimage,
                    )
                # The terminal error leaves carrying what the whole logical call cost: the
                # loop's failure accounting reads this stamp (`_billed_usage`), and a stamp
                # naming only the last attempt under-counts every absorbed one. Stamped AFTER
                # the receipt above is built -- `with_error` reads this same stamp, and a
                # cumulative stamp read back there would land the absorbed spend on the
                # receipt twice (the exact double-count the absorb commit point exists to
                # prevent).
                if spent_usage:
                    mark_provider_usage(exc, failed.usage)
                raise
            # Read on this path too, so `report_provider_retried` means the same thing whatever
            # the call returns. Honoured only on failure, it would be a reporting seam that
            # silently stops working for adapters that retry and then succeed -- which is most of
            # the time a retry loop runs.
            completed = replace(
                self._completed(receipt, turn, retried=progress.retried),
                attempts=attempts_made,
            )
            # The answering attempt's entry, built from the settled receipt so its usage is the
            # same normalized reading `_completed` just made -- and BEFORE the absorbed spend is
            # merged below, so the entry keeps this attempt's own bill and the entries sum to
            # the receipt. Its retry flag is attempt-scoped on both channels: what the adapter
            # reported during THIS dispatch, plus what this turn itself declared -- not the
            # whole-call `progress.retried` fold the receipt carries.
            answering_entry = ModelCallAttempt(
                index=attempts_made,
                elapsed_ms=self._ms_since(attempt_started),
                provider_retried=progress.count > reports_before or _turn_reported_retry(turn),
                usage=completed.usage,
                stream_committed=delivered,
                backoff_ms=pending_backoff_ms,
            )
            # One ``replace``, not two. The log and the merged bill are the same fact stated two
            # ways, and the receipt refuses a log whose entries do not sum to its usage -- so a
            # receipt carrying the log beside the not-yet-merged usage is a state this record is
            # not allowed to hold, however briefly. ``replace`` re-runs ``__post_init__``, which
            # is what makes "briefly" indistinguishable from "at all".
            completed = replace(
                completed,
                attempt_log=(
                    (*attempt_log, answering_entry) if attempt_log_complete else ()
                ),
                usage=(
                    _merged_usage(spent_usage, completed.usage) if spent_usage else completed.usage
                ),
            )
            if lifecycle_hook is not None:
                if reservation is None:  # pragma: no cover - a started durable call reserves first
                    raise AssertionError("durable model call completed without a reservation")
                durable_completed = replace(
                    completed,
                    latency_ms=elapsed_before_recovery_ms + self._ms_since(started),
                )
                durable_outcome_receipt = durable_completed
                try:
                    result_blob = durable_model_result_blob(turn)
                except DurableModelCallError as result_error:
                    raise_model_dispatch_unknown(
                        lifecycle_hook,
                        reservation,
                        result_error,
                        failure_code=result_error.error_code,
                        usage=durable_completed.usage,
                    )
                settle_model_dispatch(
                    lifecycle_hook,
                    reservation,
                    durable_completed,
                    result_blob=result_blob,
                    stream_committed=delivered,
                )
            settled = self._publish(
                request,
                turn,
                completed,
                elapsed_ms=elapsed_before_recovery_ms + self._ms_since(started),
                request_preimage=request_preimage,
            )
        return turn, settled

    @staticmethod
    def _ms_since(started: float) -> int:
        return max(0, int((time.monotonic() - started) * 1000))

    @staticmethod
    def _completed(
        receipt: ModelCallReceipt, turn: ModelTurn, *, retried: bool = False
    ) -> ModelCallReceipt:
        # Every field read defensively, for the reason the `provider_retried` probe below already
        # gives: a third-party adapter may return any turn-shaped object. Read as hard attributes,
        # a `usage=None` -- which `examples/custom_model_adapter.py` invites by calling usage
        # "optional" -- raised `AttributeError` from inside the argument list of `_publish`, so
        # *no* receipt was produced at all and an answer the provider had already been paid for was
        # discarded over a token counter. `_recordable_usage` already refuses to fail a paid call
        # over a malformed usage *value*; this is the same rule for a malformed usage *type*.
        try:
            usage = getattr(turn, "usage", None)
            normalized_usage = (
                normalize_json_ingress(dict(usage)) if isinstance(usage, Mapping) else {}
            )
        except Exception:
            normalized_usage = {}
        try:
            stop_reason = normalize_unicode_scalars(str(getattr(turn, "stop_reason", "") or ""))
        except Exception:
            stop_reason = ""
        turn_retried = _turn_reported_retry(turn)
        return replace(
            receipt,
            stop_reason=stop_reason,
            usage=_recordable_usage(normalized_usage),
            # Probed rather than read as an attribute: a third-party adapter may return any
            # turn-shaped object, and a missing flag means "did not retry", which is true of every
            # adapter with no retry loop. Combined with what the adapter reported through the
            # channel, since either alone is a partial view and neither can contradict the other:
            # both only ever say that a retry happened.
            provider_retried=receipt.provider_retried or retried or turn_retried,
        )

    def _publish(
        self,
        request: ModelRequest,
        turn: ModelTurn | None,
        receipt: ModelCallReceipt,
        *,
        elapsed_ms: int,
        request_preimage: bytes | None = None,
    ) -> ModelCallReceipt:
        """Deliver the settled receipt to observers, then record it, on every exit.

        Recording is in a `finally` because the two callers fail in opposite directions when
        delivery raises. The failure path publishes inside `contextlib.suppress(Exception)`, so a
        record placed after `dispatch_model_call` disappears with no trace; the success path
        publishes unguarded, so the same placement fails a call the provider has already been paid
        for. `dispatch_model_call` contains its own observers, so this is a defence against the
        pipeline itself, not against a broken exporter.

        The cost, stated rather than fixed: when delivery raises, the recorded receipt is the
        pre-delivery one and `capture_downgrades` stays at the floor it was resolved to. Moving the
        record earlier to make that deterministic would make it *always* zero, which is a worse
        record -- a field that never reports a withheld capture reads as "nothing was withheld".
        """

        timed = replace(receipt, latency_ms=elapsed_ms)
        settled = timed
        try:
            if self.subscriptions:
                settled = dispatch_model_call(
                    receipt=timed,
                    content=_call_content(request, turn),
                    subscriptions=self.subscriptions,
                )
        finally:
            self._record(settled, request_preimage, turn)
        return settled

    def _record(
        self, receipt: ModelCallReceipt, request_preimage: bytes | None, turn: Any | None
    ) -> None:
        """Hand the settled call to `settled_sink`, absorbing whatever it does.

        The same containment `dispatch_model_call` gives an observer, and for a sharper version of
        the same reason: a sink is typically a durable writer, so "the disk is full" must not
        become "the answer is discarded" -- nor "this provider failure is now a RuntimeError",
        which is what an unguarded call on the failure path would do to the taxonomy an escaping
        `ModelAdapterError` carries.

        The `SettledModelCall` is assembled inside the containment: its constructor is trivial
        today, and the guard is what keeps that an implementation detail rather than a promise.
        """

        sink = self.settled_sink
        if sink is None:
            return
        try:
            sink(
                SettledModelCall(
                    receipt=receipt, request_preimage=request_preimage, turn=turn
                )
            )
        except Exception:  # noqa: BLE001 - recording must not alter the model-call outcome
            _LOGGER.debug("model call settled sink failed", exc_info=True)

    async def _adrive(
        self,
        request: ModelRequest,
        deadline: float | None,
        should_abort: ShouldAbort | None,
        delta_consumer: DeltaConsumer | None,
        adapter: Any,
    ) -> ModelTurn:
        # The pre-dispatch boundary check that used to open this method now sits in `acall`, before
        # the optional synchronous lifecycle reserve/start commits. This method is called only after
        # those commits, so reaching it is the runner's adapter-entry boundary and increments the
        # receipt's attempt count exactly once.
        astream_turn = getattr(adapter, "astream_turn", None)
        if delta_consumer is not None and astream_turn is not None:
            return await self._astream(
                astream_turn, request, deadline, delta_consumer, should_abort
            )
        anext_turn = getattr(adapter, "anext_turn", None)
        if anext_turn is not None:
            return await self._aawait(
                anext_turn(request), deadline, adapter=adapter, request=request
            )
        next_turn = adapter.next_turn
        if is_async_callable(next_turn):
            return await self._aawait(
                next_turn(request), deadline, adapter=adapter, request=request
            )
        turn = await self._aawait(
            start_abandonable_sync_call(lambda: next_turn(request), thread_name=self.thread_name),
            deadline,
            adapter=adapter,
            request=request,
        )
        # Second line of defence, the same one the tool half keeps: an adapter can be synchronous
        # and still hand back something awaitable -- it delegates to an async client, or its
        # ``next_turn`` is a callable object no predicate recognised. Returned as-is, that awaitable
        # became the turn. Nothing downstream reads a coroutine as a failure: the receipt's field
        # reads are all defensive, so it recorded a *successful* call for a provider that was never
        # invoked, and the caller got an object whose every turn field was missing.
        if inspect.isawaitable(turn):
            return await self._aawait(turn, deadline, adapter=adapter, request=request)
        return turn

    async def _astream(
        self,
        astream_turn: Callable[[ModelRequest], Any],
        request: ModelRequest,
        deadline: float | None,
        delta_consumer: DeltaConsumer,
        should_abort: ShouldAbort | None,
    ) -> ModelTurn:
        """Drive `astream_turn`, relaying each chunk and folding them into one `ModelTurn`.

        The assembled turn is identical in shape to a one-shot turn, so nothing downstream of the
        call has to know the answer arrived in pieces.

        A call that fails never produces a turn, so retry evidence seen on the wire has to be
        carried by the exception instead. `assemble_streamed_turn` handles the success half; this
        holds the same fact for the half where there is nothing to assemble -- a stream cancelled or
        aborted after a retried attempt committed. Stamping the exception is how the adapters
        already report it, and `with_error` already reads it back.
        """

        agen = astream_turn(request)
        retried = False
        # Cleared the moment this call stops driving the stream, and read before every delivery.
        #
        # The kernel can stop *waiting* for a provider; it cannot stop one. That is the premise
        # `detach_unfinished_call` and `_aclose_within_grace` are both built on, and a stream is the
        # one place where a callee outliving the run does more than leak: `consume` runs as its own
        # task, so a generator that survives the cancellation the boundary delivers goes on yielding
        # into `delta_consumer` after `acall` has already raised. That consumer belongs to one call.
        # In `AgentLoop` it is a `QueueEventSink` the *next* turn rebinds to a fresh queue, so the
        # abandoned stream's tokens arrived in the next turn's stream -- output attributed to a turn
        # that never produced it, which is worse than the leak the grace interval already accepts.
        driving = True

        async def consume() -> ModelTurn:
            nonlocal retried
            chunks: list[ModelStreamChunk] = []
            ingress = ModelStreamIngressNormalizer()
            try:
                async for chunk in agen:
                    if not driving:
                        break
                    try:
                        normalized_chunks = ingress.normalize(chunk)
                    except ModelAdapterError:
                        raise
                    except Exception as exc:
                        raise ModelAdapterError(
                            "model adapter returned a non-portable stream fragment"
                        ) from exc
                    for normalized_chunk in normalized_chunks:
                        if getattr(normalized_chunk, "provider_retried", False):
                            retried = True
                        chunks.append(normalized_chunk)
                        delta_consumer(normalized_chunk)
                    if should_abort is not None and should_abort():
                        for normalized_chunk in ingress.finish():
                            if getattr(normalized_chunk, "provider_retried", False):
                                retried = True
                            chunks.append(normalized_chunk)
                            delta_consumer(normalized_chunk)
                        raise ModelCallAborted("model call aborted")
                for normalized_chunk in ingress.finish():
                    if getattr(normalized_chunk, "provider_retried", False):
                        retried = True
                    chunks.append(normalized_chunk)
                    delta_consumer(normalized_chunk)
            except BaseException:
                # A boundary can end the stream after a provider-delivered high surrogate and
                # before its continuation arrives.  At that point the unit is definitively lone;
                # deliver its replacement before propagating the original outcome.  Preserve the
                # original failure if a diagnostic consumer also rejects the synthetic suffix.
                if driving:
                    with contextlib.suppress(Exception):
                        for normalized_chunk in ingress.finish():
                            if getattr(normalized_chunk, "provider_retried", False):
                                retried = True
                            chunks.append(normalized_chunk)
                            delta_consumer(normalized_chunk)
                raise
            finally:
                # Provider async iterators own network resources, so the iterator is closed
                # explicitly rather than left to finalization.
                #
                # Bounded and swallowed, because this runs in a ``finally``: whatever it raises or
                # however long it blocks *replaces* the call's own outcome. A provider whose close
                # raised turned a user's `interrupt_turn` into a terminal failure -- the session the
                # `ModelCallAborted` type exists to keep parked was killed by its cleanup -- and
                # destroyed successful turns whose tokens had already been relayed. One that hung
                # hung the run, with no boundary able to fire, because the abort is raised inside
                # this task and so no grace interval applies to it.
                #
                # The grace is the same one an abandoned call gets, and outliving it means being
                # abandoned the same way: detached and warned about, not waited on. See
                # `_aclose_within_grace` for why the bound cannot be `asyncio.wait_for`.
                aclose = getattr(agen, "aclose", None)
                if aclose is not None:
                    with contextlib.suppress(Exception):
                        await self._aclose_within_grace(aclose)
            return assemble_streamed_turn(chunks)

        try:
            return await self._aawait(consume(), deadline)
        except BaseException as exc:
            # Wraps `_aawait`, not just `consume`: the abort raised inside the loop is only one of
            # the ways this ends. `RunCancelled` and `RunTimeout` are raised by the race *around*
            # the stream and never pass through the adapter, so they are exactly the exceptions the
            # provider had no opportunity to stamp.
            if retried:
                mark_provider_retried(exc)
            raise
        finally:
            # A `finally` rather than a line in the handler above, and deliberately *not* because a
            # normal return can leave `consume` running -- it cannot. `await_abandonable_call`
            # returns `task.result()` only once the task is done, so returning here means the drive
            # already finished, and every other exit passes through the `except BaseException`.
            # Clearing the flag there instead is, today, exactly equivalent: a mutation that moves
            # this line into that handler survives the suite, and no test can be written against it.
            #
            # It stays here because the guarantee must not depend on that argument holding. It
            # currently rests on two facts one edit away from changing -- that the handler catches
            # every exception type and re-raises, and that `mark_provider_retried` cannot raise
            # before the clear is reached (it swallows `Exception`). Add an early return, narrow the
            # handler, or reorder those two statements, and an abandoned stream starts talking to a
            # finished call again. A `finally` costs nothing and survives all three.
            driving = False

    async def _aclose_within_grace(self, aclose: Callable[[], Any]) -> None:
        """Close a provider's stream, spending at most the grace interval on its cleanup.

        `asyncio.wait_for` is the obvious spelling and does **not** bound this. On timeout it
        cancels the close and then *awaits that cancellation to complete*, so a cleanup that
        suppresses `CancelledError` holds the run for as long as it wants: a close that ignored
        cancellation for 5s took 4.6s against a 0.05s grace, ~90x over. A real bound has to stop
        waiting, which means detaching -- `core._sync_bridge` already does exactly this for a call
        that outlives a run boundary, down to consuming the outcome so the abandoned task does not
        resurface as an unretrieved exception at collection.

        Abandoning is the lesser harm, not a good outcome, so it warns for the reason
        `detach_unfinished_call` warns: one leak per abandoned stream, on a loop that may run for
        days, is the kind of growth that has to be visible to be fixed. The grace is spent *before*
        cancelling rather than after, because unlike an in-flight call a close is already the
        cleanup -- cancelling it first would pre-empt the very work the interval exists to allow. A
        slow-but-cancellable close is therefore warned about and then reclaimed immediately; the
        message says so rather than claiming a leak that did not happen.
        """

        closing = asyncio.ensure_future(aclose())
        try:
            grace_s = self._grace_s()
            done, _pending = await asyncio.wait({closing}, timeout=max(0.0, grace_s))
            if closing in done:
                return
            _LOGGER.warning(
                "a provider stream's close outran the %.3gs grace interval: the run stopped waiting "
                "for it and cancelled it. A close that honours the cancellation is reclaimed at "
                "that point; one that suppresses it keeps the generator and any connection it holds "
                "alive, one per streamed call, with nothing left to reclaim them. Enforce a timeout "
                "at the provider's I/O edge.",
                grace_s,
            )
        finally:
            # Also covers the run boundary firing *here*: this runs inside the stream task's
            # `finally`, so the cancellation that abandons the call lands on this await too.
            if not closing.done():
                closing.cancel()
            if closing.done():
                consume_task_outcome(closing)
            else:
                closing.add_done_callback(consume_task_outcome)

    async def _abackoff(self, delay_s: float, deadline: float | None) -> None:
        """Wait between kernel attempts under the same race the attempts run under.

        The sleep is `pending` to the shared bridge: a cancellation wakes it through the
        token callback, the deadline bounds it through the wait timeout, and the boundary
        check re-raises the same `RunCancelled`/`RunTimeout` an in-flight attempt reports.
        A backoff cannot outlive a run that stopped wanting the answer.
        """

        if delay_s <= 0:
            self._check_cancel_or_deadline(deadline)
            return
        await await_abandonable_call(
            asyncio.sleep(delay_s),
            deadline=deadline,
            token=self._token(),
            grace_s=0.0,
            check_boundary=self._check_cancel_or_deadline,
        )

    async def _aawait(
        self,
        pending: Any,
        deadline: float | None,
        *,
        adapter: Any = None,
        request: ModelRequest | None = None,
    ) -> ModelTurn:
        """Await model I/O against the shared cancel/deadline race.

        Only terminal run boundaries apply while a model call is in flight. Interrupt and pause are
        step-boundary signals for a one-shot call and are the caller's to check after the model
        returns; on a streamed call the caller's `should_abort` covers the cooperative stop.

        An adapter that cancels its *own* call failed; the run did not stop. `await_abandonable_call`
        raises `CalleeCancelled` rather than `CancelledError` exactly so its two callers can each say
        what it means to them -- the tool path names it `tool_handler_cancelled` -- and this is the
        model half of that. Reported as a `ModelAdapterError` so it reaches the loop's
        `except ModelAdapterError`, which records a `model_turn` naming the failure, instead of the
        generic handler that rewraps with `str(exc)`: `CalleeCancelled` carries no message, so a run
        failed with an empty one. Every dispatch shape funnels through here, so one translation
        covers all of them.
        """

        # Resolved before the wait is entered, and not in the argument list, because `pending` is
        # already live by the time this method runs -- on the blocking path it is a daemon worker
        # inside the provider. `await_abandonable_call` sets its cleanup up inside itself, so an
        # accessor raising here used to leave that worker running with its future neither detached
        # nor consumed. Same shape as the registration failure the bridge already guards against:
        # anything that can fail after a call is live has to release it.
        #
        # The static field is what the detach is given, not `_grace_s()`: that is the accessor which
        # may have just raised.
        #
        # The discard hook is the same one the wait below is given, and this exit went without it
        # for a round. The callee cannot tell which exit its caller took: a replay adapter that
        # answers after `_grace_s()` raised has already advanced its cursor, so dropping the turn
        # here leaves the slot spent and hands the following recording to the next caller. Both
        # exits of this function carry it, and
        # `test_every_abandonable_call_site_routes_its_discards` is what keeps that true.
        try:
            token = self._token()
            grace_s = self._grace_s()
        except BaseException:
            await abandon_unwaited_call(
                pending,
                grace_s=self.cancel_grace_s,
                on_discarded=_discard_hook(adapter, request),
            )
            raise
        try:
            return await await_abandonable_call(
                pending,
                deadline=deadline,
                token=token,
                grace_s=grace_s,
                check_boundary=self._check_cancel_or_deadline,
                on_discarded=_discard_hook(adapter, request),
            )
        except CalleeCancelled as exc:
            raise ModelAdapterError(
                "model adapter cancelled its own call",
                error_code="model_adapter_cancelled",
            ) from exc
