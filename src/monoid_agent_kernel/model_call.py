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

**What it does not do: retry.** Backoff and HTTP classification live inside the adapters
(`providers/gateway.py`), so the kernel makes exactly one adapter call per turn. That is the
distinction `ModelCallReceipt.attempts` and `provider_retried` encode, and it is why this module
inherits classification without inheriting a retry loop.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

from monoid_agent_kernel.core._sync_bridge import (
    await_abandonable_call,
    start_abandonable_sync_call,
)
from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.model_io import (
    ModelCallReceipt,
    ModelIOSubscription,
    dispatch_model_call,
)
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.errors import ModelCallAborted, RunCancelled, RunTimeout
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelStreamChunk,
    ModelTurn,
    assemble_streamed_turn,
)
from monoid_agent_kernel.tools.base import ToolSpec

DeltaConsumer = Callable[[ModelStreamChunk], None]
"""Receives every chunk of a streamed call, in order, as it arrives.

Deliberately every chunk and not a filtered subset: the two callers in the kernel want different
subsets -- a live stream relays all of them, an event-emitting run only turns text and reasoning
into events -- and a runner that filtered would have to know which caller it was serving. Filtering
is the consumer's business; ordering and completeness are this module's.
"""

ShouldAbort = Callable[[], bool]
"""Polled once per streamed chunk, after it has been delivered. See `ModelCallRunner.acall`."""


def _prompt_payload(request: ModelRequest) -> dict[str, Any]:
    """The assembled prompt, as the thing `prompt_digest` identifies.

    Tool definitions and generation settings are deliberately absent: the question this digest
    answers is "did the model see the same conversation twice", which must stay true when a tool is
    added to the surface or the temperature changes around it.

    Everything that *constitutes* the conversation is present, including the by-reference shape.
    A request may carry its history as `messages`, or as a `previous_turn_handle` naming history the
    provider holds plus the `observations` produced since -- and in that second shape those two
    fields **are** the prompt. Hashing only `messages` made every by-reference continuation collide
    with every other, which is the ordinary case for a gateway client, not an edge one.
    """

    return {
        "system_prompt": request.system_prompt,
        "instruction": request.instruction,
        "messages": list(request.messages or ()),
        "previous_turn_handle": request.previous_turn_handle,
        "observations": [observation.to_json() for observation in request.observations],
    }


_MAX_DIGEST_DEPTH = 32


def _canonical_ready(value: Any, depth: int = 0) -> Any:
    """`value` reshaped so the canonical serializer can always carry it.

    Digests are computed on every call, *before* the adapter is reached, so a value the serializer
    chokes on stops the call from happening at all. A digest must never be able to do that: it is
    bookkeeping about a call, not a precondition for making one.

    Applied to the whole payload rather than to selected fields. The first attempt guarded only the
    tool definitions, which left `messages` and `observations` -- both `dict[str, Any]` a caller
    fills -- able to kill a call outright. Guarding at the one place every digest passes through is
    what makes the property universal instead of true of the fields somebody remembered.

    Mapping keys are stringified because canonical JSON sorts them, and sorting mixed key types
    raises. Depth is bounded so a cyclic or pathologically nested payload terminates rather than
    exhausting the stack.

    Anything JSON has no form for degrades to `repr`, which for a default `__repr__` embeds an
    address and so digests differently each run. That is the deliberate direction: an unstable
    replay key always misses and costs a re-run, while a stable-but-lossy one -- hashing the type
    name, say -- would let two unrelated objects claim the same call.
    """

    if depth > _MAX_DIGEST_DEPTH:
        return "<max-depth>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _canonical_ready(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_ready(item, depth + 1) for item in value]
    return repr(value)


def _digest(payload: dict[str, Any]) -> str:
    """The one place a model-call digest is taken, so normalization cannot be skipped at one."""

    return canonical_sha256(_canonical_ready(payload))


def _tool_payload(spec: ToolSpec) -> dict[str, Any]:
    """A tool definition as the replay key sees it: every field except the ones JSON cannot carry.

    Read off the dataclass rather than listing fields by hand, so a field added to `ToolSpec` joins
    the digest automatically instead of quietly falling out of it. Reducing a tool to its `id` --
    which this did first -- made two requests offering the same id with different descriptions or
    input schemas produce the same replay key, though the provider was sent different tool
    definitions.

    Erring toward *more* than the wire carries is deliberate and asymmetric: an over-sensitive
    replay key costs a miss and a re-run, an under-sensitive one hands back the wrong call.
    """

    return {
        field_.name: getattr(spec, field_.name)
        for field_ in fields(spec)
        if not callable(getattr(spec, field_.name))
    }


def _request_payload(request: ModelRequest, model: ModelConfig) -> dict[str, Any]:
    """The whole request, as the thing `request_digest` identifies -- the replay key.

    `model` is the *effective* config, resolved by the caller, not `request.model`. The request's
    is optional and the shipped adapters fall back to their own, so hashing `request.model or
    ModelConfig()` gave two calls on differently-configured adapters the same replay key -- and
    stamped the receipt with the default model rather than the one that ran.
    """

    payload = _prompt_payload(request)
    payload["tools"] = [_tool_payload(spec) for spec in request.tools]
    payload["model"] = model.to_json()
    return payload


def _recordable_usage(usage: Mapping[str, Any]) -> dict[str, int]:
    """The usage counts a receipt will accept, dropping any the adapter got wrong.

    `ModelCallReceipt` refuses a negative or non-integer count, and rightly so -- usage is summed,
    so a negative silently subtracts from an aggregate. But raising *here* would fail a call the
    provider has already been paid for, on account of a counter nobody reads for control flow. So a
    malformed entry is omitted and the rest of the receipt still lands, the same trade
    `dispatch_model_call` makes when an observer raises.

    `bool` is excluded because it is an `int` subclass and a boolean token count is a bug, not a
    count of one -- the same rule the receipt applies.
    """

    return {
        key: value
        for key, value in usage.items()
        if isinstance(key, str) and not isinstance(value, bool) and isinstance(value, int) and value >= 0
    }


def _call_content(request: ModelRequest, turn: ModelTurn | None) -> dict[str, Any]:
    """What an observer may be shown of one call, before any redaction.

    Keys are stable field names rather than one blob so a `RedactionPolicy` can name a field and a
    consumer can be given some fields and not others.

    `observations` are here because they are model *input*: in the by-reference request shape they
    carry the tool results the model is being shown. Omitting them did not merely give a `full`
    observer an incomplete picture -- it routed tool output around redaction entirely, since a
    policy can only mask a field it is handed. That is a disclosure hole, not a gap in coverage.
    """

    content: dict[str, Any] = {
        "system_prompt": request.system_prompt,
        "instruction": request.instruction,
        "messages": list(request.messages or ()),
        "observations": [observation.to_json() for observation in request.observations],
        "previous_turn_handle": request.previous_turn_handle or "",
    }
    if turn is not None:
        content["output_text"] = turn.final_text or ""
        content["tool_calls"] = [dict(call.__dict__) for call in turn.tool_calls]
    return content


@dataclass
class ModelCallRunner:
    """Runs one model call against an adapter, whatever shape that adapter is.

    Five adapter shapes reach the same semantics -- a streamed call relayed to a consumer, a
    streamed call with no consumer, `anext_turn`, a coroutine `next_turn`, and a blocking
    `next_turn` on an abandonable daemon thread. All five go through the same cancel/deadline race,
    so an adapter's async-ness never changes when a run stops.

    Which shape is used is a function of the **call arguments**, never of state held elsewhere: a
    streamed drive happens when the caller passes a `delta_consumer` and the adapter can stream.
    That is what makes the runner testable in isolation and what stopped path selection from
    depending on whether some other object's queue happened to be active.
    """

    adapter: Any
    """The model adapter. Typed `Any` because the five shapes are probed with `getattr`, exactly as
    the protocols in `providers/base.py` intend -- declaring the optional members would make them
    required for structural typing and reject a third-party adapter implementing only `next_turn`."""

    current_cancellation_token: Callable[[], CancellationToken | None] | None = None
    """Returns the token to observe **at the moment of the call**, not a token captured at
    construction. `AgentLoop.astream` installs a token lazily on a run already in progress, so a
    runner holding a snapshot would watch a token nobody cancels and silently lose cancellation on
    the streaming path."""

    cancel_grace_s: float = 1.0
    """How long an abandoned call's worker is given to settle before it is reported as leaked."""

    thread_name: str = "nar-model-call"
    """Thread name for a blocking `next_turn`. Carries the run id in kernel use, so a leaked worker
    in a thread dump names the run that leaked it."""

    subscriptions: Sequence[ModelIOSubscription] = ()
    """Observers of settled calls. Empty by default: capture is opt-in, and a runner wired to
    nothing does no digesting at all."""

    def _effective_model(self, request: ModelRequest) -> ModelConfig:
        """The config the adapter will actually run under.

        `ModelRequest.model` is optional and every shipped adapter falls back to its own
        `self.config`, so a receipt built from the request alone reports the *default* model no
        matter which one served the call -- a fabricated audit field, not merely a missing one.
        Probed via `getattr` and type-checked: see `ConfiguredModelAdapter`.
        """

        if request.model is not None:
            return request.model
        configured = getattr(self.adapter, "config", None)
        return configured if isinstance(configured, ModelConfig) else ModelConfig()

    def _token(self) -> CancellationToken | None:
        return None if self.current_cancellation_token is None else self.current_cancellation_token()

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
        """

        started = time.monotonic()
        model = self._effective_model(request)
        receipt = ModelCallReceipt(
            context=context if context is not None else InvocationContext(),
            model=model,
            provider_name=str(getattr(self.adapter, "provider_name", "") or ""),
            prompt_digest=_digest(_prompt_payload(request)),
            request_digest=_digest(_request_payload(request, model)),
        )
        try:
            turn = await self._adrive(request, deadline, should_abort, delta_consumer)
        except BaseException as exc:
            self._publish(
                request, None, receipt.with_error(exc), elapsed_ms=self._ms_since(started)
            )
            raise
        settled = self._publish(
            request,
            turn,
            self._completed(receipt, turn),
            elapsed_ms=self._ms_since(started),
        )
        return turn, settled

    @staticmethod
    def _ms_since(started: float) -> int:
        return max(0, int((time.monotonic() - started) * 1000))

    @staticmethod
    def _completed(receipt: ModelCallReceipt, turn: ModelTurn) -> ModelCallReceipt:
        return replace(
            receipt,
            stop_reason=str(turn.stop_reason or ""),
            usage=_recordable_usage(turn.usage),
            # Probed rather than read as an attribute: a third-party adapter may return any
            # turn-shaped object, and a missing flag means "did not retry", which is true of every
            # adapter with no retry loop.
            provider_retried=bool(getattr(turn, "provider_retried", False)),
        )

    def _publish(
        self,
        request: ModelRequest,
        turn: ModelTurn | None,
        receipt: ModelCallReceipt,
        *,
        elapsed_ms: int,
    ) -> ModelCallReceipt:
        timed = replace(receipt, latency_ms=elapsed_ms)
        if not self.subscriptions:
            return timed
        return dispatch_model_call(
            receipt=timed,
            content=_call_content(request, turn),
            subscriptions=self.subscriptions,
        )

    async def _adrive(
        self,
        request: ModelRequest,
        deadline: float | None,
        should_abort: ShouldAbort | None,
        delta_consumer: DeltaConsumer | None,
    ) -> ModelTurn:
        astream_turn = getattr(self.adapter, "astream_turn", None)
        if delta_consumer is not None and astream_turn is not None:
            return await self._astream(
                astream_turn, request, deadline, delta_consumer, should_abort
            )
        anext_turn = getattr(self.adapter, "anext_turn", None)
        if anext_turn is not None:
            return await self._aawait(anext_turn(request), deadline)
        next_turn = self.adapter.next_turn
        if inspect.iscoroutinefunction(next_turn):
            return await self._aawait(next_turn(request), deadline)
        return await self._aawait(
            start_abandonable_sync_call(lambda: next_turn(request), thread_name=self.thread_name),
            deadline,
        )

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
        """

        agen = astream_turn(request)

        async def consume() -> ModelTurn:
            chunks: list[ModelStreamChunk] = []
            try:
                async for chunk in agen:
                    chunks.append(chunk)
                    delta_consumer(chunk)
                    if should_abort is not None and should_abort():
                        raise ModelCallAborted("model call aborted")
            finally:
                # Provider async iterators own network resources. Cooperative cancellation enters
                # their ``finally`` and then explicitly closes the iterator; stubborn cleanup is
                # detached by ``_aawait`` after its bounded grace interval.
                aclose = getattr(agen, "aclose", None)
                if aclose is not None:
                    await aclose()
            return assemble_streamed_turn(chunks)

        return await self._aawait(consume(), deadline)

    async def _aawait(self, pending: Any, deadline: float | None) -> ModelTurn:
        """Await model I/O against the shared cancel/deadline race.

        Only terminal run boundaries apply while a model call is in flight. Interrupt and pause are
        step-boundary signals for a one-shot call and are the caller's to check after the model
        returns; on a streamed call the caller's `should_abort` covers the cooperative stop.
        """

        return await await_abandonable_call(
            pending,
            deadline=deadline,
            token=self._token(),
            grace_s=self.cancel_grace_s,
            check_boundary=self._check_cancel_or_deadline,
        )
