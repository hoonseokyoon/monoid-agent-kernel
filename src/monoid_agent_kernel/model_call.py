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

import asyncio
import contextlib
import hashlib
import inspect
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

from monoid_agent_kernel.core._sync_bridge import (
    CalleeCancelled,
    await_abandonable_call,
    consume_task_outcome,
    start_abandonable_sync_call,
)
from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.model_io import (
    ModelCallReceipt,
    ModelIOSubscription,
    dispatch_model_call,
)
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.errors import (
    ModelAdapterError,
    ModelCallAborted,
    RunCancelled,
    RunTimeout,
)
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelStreamChunk,
    ModelTurn,
    assemble_streamed_turn,
    collect_retry_reports,
    mark_provider_retried,
)
from monoid_agent_kernel.tools.base import ToolSpec

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

    `messages` keeps `None` apart from `()`, because the wire does. Both shipped adapters select the
    request shape with `messages is not None` -- an empty tuple sends an empty conversation and
    drops the instruction, `None` sends the instruction or the handle. `or ()` read the field for
    emptiness when the field's own meaning is presence, so two requests the provider answers
    differently were handed one replay key.
    """

    return {
        "system_prompt": request.system_prompt,
        "instruction": request.instruction,
        "messages": None if request.messages is None else list(request.messages),
        "previous_turn_handle": request.previous_turn_handle,
        "observations": [observation.to_json() for observation in request.observations],
    }


# Bounds the encoder's output, not the input's shape. Comfortably above a resolved multimodal
# request (base64 image parts ride in ``messages``) and far below the point where a deliberately
# shared payload could expand without end.
_MAX_DIGEST_BYTES = 4 * 1024 * 1024

# The settings ``core._util.canonical_sha256`` serializes with, and the same object does the
# encoding and the hashing here. A guard that checks one encoding while another does the hashing
# is exactly how a payload once passed validation and then raised mid-hash.
_CANONICAL_ENCODER = json.JSONEncoder(
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    check_circular=True,
)


def _digest(payload: dict[str, Any]) -> str:
    """The canonical-JSON digest of `payload`, or `""` when canonical JSON cannot carry it.

    Streamed through the standard encoder rather than normalized first. Four rounds of review went
    into a hand-written normalizer that reshaped anything into something hashable -- stringified
    mapping keys, `<cycle:n>` markers, `repr` for values JSON had no form for -- and each fix
    revealed another way for two different requests to land on one digest: a `repr` shared by
    unrelated objects, a marker a caller could type as ordinary text, a lone surrogate that passed
    the type check and then failed at encode.

    The premise was wrong. A `ModelRequest` carries what will be sent to a provider over HTTP, so a
    payload canonical JSON cannot carry was never going to reach a model either. Reshaping it into
    something hashable invented an identity for a request that does not exist, and inventing
    identities is the one thing a replay key must not do.

    So: hash what encodes, and issue no key for what does not. Refusing is safe -- the call still
    happens, it simply is not replayable -- while a fabricated key returns the wrong call. An empty
    digest means *no key*; two unreplayable calls both carry `""` and are not thereby the same call.

    Every encoder failure means the same thing here -- no key -- so the clause catches `Exception`
    rather than a list of types. Naming them was itself a bug found four times over: circular
    references, unencodable primitives, unserializable objects, then a `dict` subclass whose
    `items()` raises. The question is never which exception the encoder chose, only whether it
    finished. `BaseException` is deliberately not caught: a cancellation or an interrupt is not a
    statement about the payload.

    Output is capped so a payload built from shared references cannot expand without bound; passing
    the cap also means no key, since a prefix would stand for the whole.
    """

    hasher = hashlib.sha256()
    encoded = 0
    try:
        for chunk in _CANONICAL_ENCODER.iterencode(payload):
            raw = chunk.encode("utf-8")
            encoded += len(raw)
            if encoded > _MAX_DIGEST_BYTES:
                return ""
            hasher.update(raw)
    except Exception:
        return ""
    return hasher.hexdigest()


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


def _request_payload(
    request: ModelRequest, model: ModelConfig, *, provider: str, destination: str
) -> dict[str, Any]:
    """The whole request, as the thing `request_digest` identifies -- the replay key.

    `model` is the *effective* config, resolved by the caller, not `request.model`. The request's is
    optional and the shipped adapters fall back to their own, so hashing `request.model or
    ModelConfig()` gave two calls on differently-configured adapters the same replay key.

    `provider` and `destination` are here because the same request answered by a different service
    is a different call. A config alone does not identify the service: an adapter may route by a
    per-instance override or an environment variable, so two adapters holding identical configs can
    address different hosts. `destination` is hashed and never recorded, so an internal hostname
    stays internal, and it is empty for an adapter that does not expose one -- see
    `AddressedModelAdapter`.
    """

    payload = _prompt_payload(request)
    payload["tools"] = [_tool_payload(spec) for spec in request.tools]
    payload["model"] = model.to_json()
    payload["provider"] = provider
    payload["destination"] = destination
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
                calls.append({"repr": repr(call)})
        content["tool_calls"] = calls
    return content


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
    prompt per call, which is why `_MAX_DIGEST_BYTES` bounds it. `AgentLoop` builds its runner with
    no subscriptions and still pays that; anyone trimming the cost should start there and not
    expect this field to gate it."""

    def _effective_model(self, request: ModelRequest, adapter: Any) -> ModelConfig:
        """The config the adapter will actually run under.

        `ModelRequest.model` is optional and every shipped adapter falls back to its own
        `self.config`, so a receipt built from the request alone reports the *default* model no
        matter which one served the call -- a fabricated audit field, not merely a missing one.
        Probed via `getattr` and type-checked: see `ConfiguredModelAdapter`.
        """

        if request.model is not None:
            return request.model
        # Tolerant of a raising probe for the reason `_destination` gives: a replay key is
        # bookkeeping, and an adapter that cannot answer must not thereby lose its call. Plain
        # `getattr(..., None)` swallowed only `AttributeError`, so a `config` property that raised
        # anything else took the whole call down.
        #
        # A raising probe and an absent one both land on the default here, because the return type
        # admits nothing else -- the receipt cannot say "unknown". That is a known limit of this
        # field, not a distinction being drawn.
        try:
            configured = getattr(adapter, "config", None)
        except Exception:
            configured = None
        return configured if isinstance(configured, ModelConfig) else ModelConfig()

    def _destination(self, model: ModelConfig, adapter: Any) -> str:
        """Where this adapter would send a call under `model`, or `""` if it does not say.

        Probed and tolerant of failure for the same reason every other probe here is: a replay key
        is bookkeeping, and an adapter that cannot answer must not thereby lose its call.

        The `getattr` is inside the `try`, not before it. Tolerating only the *call* left the
        *lookup* undefended, so an adapter exposing `resolve_destination` as a property that raised
        still lost its call -- the one shape the rule above exists to rule out, surviving in the
        probe the other two were written to imitate.
        """

        try:
            resolve = getattr(adapter, "resolve_destination", None)
            if not callable(resolve):
                return ""
            return str(resolve(model) or "")
        except Exception:
            return ""

    def _token(self) -> CancellationToken | None:
        return None if self.current_cancellation_token is None else self.current_cancellation_token()

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
        adapter = self._current_adapter()
        model = self._effective_model(request, adapter)
        # Same tolerance as the other two adapter probes, and for the same reason. Undefended, a
        # `provider_name` property that raised -- or whose `str()` did -- lost the call before the
        # adapter was ever invoked, over a field nothing reads for control flow.
        try:
            provider = str(getattr(adapter, "provider_name", "") or "")
        except Exception:
            provider = ""
        receipt = ModelCallReceipt(
            context=context if context is not None else InvocationContext(),
            model=model,
            provider_name=provider,
            prompt_digest=_digest(_prompt_payload(request)),
            request_digest=_digest(
                _request_payload(
                    request,
                    model,
                    provider=provider,
                    destination=self._destination(model, adapter),
                )
            ),
        )
        with collect_retry_reports() as progress:
            try:
                turn = await self._adrive(request, deadline, should_abort, delta_consumer, adapter)
            except BaseException as exc:
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
                with contextlib.suppress(Exception):
                    self._publish(
                        request, None, receipt.with_error(exc), elapsed_ms=self._ms_since(started)
                    )
                raise
            # Read on this path too, so `report_provider_retried` means the same thing whatever
            # the call returns. Honoured only on failure, it would be a reporting seam that
            # silently stops working for adapters that retry and then succeed -- which is most of
            # the time a retry loop runs.
            settled = self._publish(
                request,
                turn,
                self._completed(receipt, turn, retried=progress.retried),
                elapsed_ms=self._ms_since(started),
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
        usage = getattr(turn, "usage", None)
        return replace(
            receipt,
            stop_reason=str(getattr(turn, "stop_reason", "") or ""),
            usage=_recordable_usage(usage if isinstance(usage, Mapping) else {}),
            # Probed rather than read as an attribute: a third-party adapter may return any
            # turn-shaped object, and a missing flag means "did not retry", which is true of every
            # adapter with no retry loop. Combined with what the adapter reported through the
            # channel, since either alone is a partial view and neither can contradict the other:
            # both only ever say that a retry happened.
            provider_retried=retried or bool(getattr(turn, "provider_retried", False)),
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
        adapter: Any,
    ) -> ModelTurn:
        # Before dispatch, not only inside the race. `_aawait` reports a boundary that had already
        # been crossed, but by then the adapter has been invoked and the provider has been paid for
        # work the run had already decided not to do. Checking here also covers the interval the
        # caller cannot: building the receipt digests happens between the caller's own boundary
        # check and this line, so a deadline can expire in between.
        #
        # Nothing awaits between here and the dispatch below, so the check cannot go stale within
        # this task. A boundary crossed *after* dispatch is the race's business, which is why this
        # is an addition to it rather than a replacement.
        self._check_cancel_or_deadline(deadline)
        astream_turn = getattr(adapter, "astream_turn", None)
        if delta_consumer is not None and astream_turn is not None:
            return await self._astream(
                astream_turn, request, deadline, delta_consumer, should_abort
            )
        anext_turn = getattr(adapter, "anext_turn", None)
        if anext_turn is not None:
            return await self._aawait(anext_turn(request), deadline)
        next_turn = adapter.next_turn
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
            try:
                async for chunk in agen:
                    if not driving:
                        break
                    if getattr(chunk, "provider_retried", False):
                        retried = True
                    chunks.append(chunk)
                    delta_consumer(chunk)
                    if should_abort is not None and should_abort():
                        raise ModelCallAborted("model call aborted")
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

    async def _aawait(self, pending: Any, deadline: float | None) -> ModelTurn:
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

        try:
            return await await_abandonable_call(
                pending,
                deadline=deadline,
                token=self._token(),
                grace_s=self._grace_s(),
                check_boundary=self._check_cancel_or_deadline,
            )
        except CalleeCancelled as exc:
            raise ModelAdapterError(
                "model adapter cancelled its own call",
                error_code="model_adapter_cancelled",
            ) from exc
