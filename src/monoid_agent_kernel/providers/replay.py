"""An adapter that answers from a recorded corpus, and refuses everything it cannot prove.

W6-4b B3. ``core/payload_replay.py`` owns reading; this owns *standing where a provider
stood*: the impersonation that makes recorded keys reachable, the reconstruction that hands
the loop real ``ModelTurn``/``ToolCall`` objects verbatim (``raw={}`` -- an honest "this is a
replay"), and the miss semantics D-c/D-d fixed: fail by default with a typed, content-free
:class:`ReplayMiss`, or fall through to exactly one live ``inner`` adapter.

**Impersonation is derived from evidence, not copied from a field (D-h).** The corpus's
``provider`` term is a *resolved* value -- the original adapter's declaration when it had
one, else its config's provider -- so the term alone cannot say whether the original
*declared*. The loop's reasoning re-injection gate reads only the declaration
(``loop.py``'s assistant-message append), which makes the difference load-bearing: declare
for a corpus whose original did not, and every second-turn preimage grows a reasoning block
the recorded ones never had -- a silent 100% miss from turn two. So the adapter reads the
evidence: recorded request messages carrying the injected block mean the original declared
(declare); recorded answers carrying reasoning where a recorded request *had a turn behind
it* and still carried no injected trace mean it did not (do not declare -- the key's provider
term is then authored by the replay run's config, which the preflight checks); anything else
declares, because declaring is inert for injection and pins the key's provider term
independent of the run config.

That third branch includes the corpus that cannot testify. The block is appended *after* a
call, so only a request with an assistant message in front of it could ever have carried one:
a corpus whose every recorded run settled in a single turn is silent on the question, not
evidence for the negative. Reading its silence as a refusal broke the shipped gateway default
outright -- ``GatewayModelAdapter`` declares the relayed provider (``openai``) while
``ModelConfig.provider`` names the transport (``gateway``), so the recorded key term is
``openai`` and a non-declaring replay computes ``gateway``: every lookup in the run misses,
and the preflight refuses a config and a corpus that are both correct.

**Heterogeneous sources are rejected at construction.** One adapter serves one provider's
answers; a family that mixed providers cannot exist under the kernel (children share the
parent's adapter instance), so arriving here it is an assembly mistake, said early.

``astream_turn`` is deliberately absent: the runner's structural fallback serves streamed runs
one-shot, and ``docs/CLI.md`` names that degradation under the v1 limits rather than leaving
it to be discovered. ``resolve_destination`` is deliberately absent too (D-f):
``not_declared`` is the honest answer, and the ledger delta it causes against an original
whose destination resolved is documented rather than imitated.

**Under fallthrough the wrapper still speaks for the call.** The receipt is built by probing
this adapter, so a live answer the inner produced is stamped with whatever *this* adapter
declares -- the corpus-derived provider, or nothing at all where the derivation declined, which
the receipt records as ``""`` -- and with ``not_declared``. The inner's own declaration and
destination do not reach it, and its ``supports_multimodal``/``astream_turn`` do not either. That is forced rather than chosen: the
declaration is what makes recorded keys reachable, so it cannot also report who served a miss.
``docs/CONTRACTS.md`` carries it as the third ledger delta, the one only fallthrough produces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from monoid_agent_kernel.core.payload_replay import (
    MISS_ABSENT,
    MISS_NO_KEY,
    MISS_NOT_RECORDED,
    REPLAY_MISS_REASONS,
    ReplayCorpus,
    ReplayMissReason,
    ReplayedResponse,
)
from monoid_agent_kernel.core._sync_bridge import is_async_callable
from monoid_agent_kernel.core.model_payloads import RECORDED_TURN_FIELDS
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.providers._request_identity import (
    _REQUEST_DIGEST_GENERATION,
    replay_lookup,
)
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelTurn,
    ToolCall,
    unportable_usage_key,
)

_AUTO: Any = object()
"""Derive from corpus evidence. Distinct from ``None``, which is an explicit non-answer."""


class ReplayMiss(ModelAdapterError):
    """The corpus cannot truthfully answer this call, and no fallthrough was configured.

    ``error_code="replay_miss"`` with ``retryable=False`` -- asking the same disk the same
    question is not a transient condition, and the W7 kernel retry must not spin on it --
    and ``config_recoverable=True``: the remedy is operator-shaped (fix the config the
    preflight named, add the missing family run directory, or rerun live), which is the
    park-not-kill classification a 4xx gets. ``provider_error_code`` carries the sub-reason
    and is a door, not a convention: only the six approved reasons pass.
    """

    def __init__(self, message: str, *, provider_error_code: str) -> None:
        if provider_error_code not in REPLAY_MISS_REASONS:
            raise ValueError(
                f"unknown replay miss reason {provider_error_code!r}; "
                f"the vocabulary is {', '.join(REPLAY_MISS_REASONS)}"
            )
        super().__init__(
            message,
            error_code="replay_miss",
            retryable=False,
            config_recoverable=True,
            provider_error_code=provider_error_code,
        )


def _is_injected_reasoning(message: Any) -> bool:
    """The loop's re-injected reasoning block, as the assistant-message append writes it."""

    if not isinstance(message, Mapping):
        return False
    block = message.get("reasoning")
    return isinstance(block, Mapping) and "provider" in block and "items" in block


def _is_assistant_message(message: Any) -> bool:
    """A recorded request that already had a turn behind it.

    The witness for whether the corpus can testify at all. The loop appends the reasoning
    block *after* a call, so only a request with an assistant message in front of it could
    ever have carried one -- a corpus of first turns is silent on the question, not evidence
    for the negative.
    """

    return isinstance(message, Mapping) and message.get("role") == "assistant"


def _is_media_block(part: Any) -> bool:
    """The neutral resolved media block (``core/media.py``'s base64 source shape)."""

    if not isinstance(part, Mapping):
        return False
    source = part.get("source")
    return isinstance(source, Mapping) and source.get("type") == "base64"


class ReplayModelAdapter:
    """Serves ``next_turn`` from a :class:`ReplayCorpus`; see the module docstring.

    ``sources`` is a sequence of run directories (the family union -- children record to
    their own run dirs, so replaying a spawning run means naming the children too) or an
    already-loaded corpus. Construction is where everything early-checkable fails: unreadable
    sources, heterogeneous providers, an inner adapter this wrapper could not drive or could
    not release.
    """

    wire_image_encoding: str

    def __init__(
        self,
        sources: Sequence[Path] | ReplayCorpus,
        *,
        inner: Any = None,
        provider_name: Any = _AUTO,
        supports_multimodal: Any = _AUTO,
        wire_image_encoding: str = "base64",
        config: ModelConfig | None = None,
    ) -> None:
        corpus = (
            sources
            if isinstance(sources, ReplayCorpus)
            else ReplayCorpus.load([Path(source) for source in sources])
        )
        self._corpus = corpus
        self._inner = inner
        self._inner_opener = None
        self._inner_closer = None
        if inner is not None:
            next_turn = getattr(inner, "next_turn", None)
            if not callable(next_turn) or is_async_callable(next_turn):
                # `callable` is true for an `async def`, and `ModelCallRunner` documents a
                # coroutine `next_turn` as one of its four dispatch shapes -- so without the
                # second half this wrapper accepts an inner it cannot drive, hands `_adrive` a
                # coroutine that has done no provider work, and spends the refused slot on it
                # before the awaited call has a chance to fail.
                raise ValueError(
                    "the fallthrough inner adapter exposes no synchronous next_turn; this "
                    "wrapper is synchronous and cannot drive an anext_turn-only adapter"
                )
            # Both halves resolved before either is ever called -- the CLI's own lifecycle
            # doctrine, enforced here because the CLI's probe only sees this wrapper.
            opener = getattr(inner, "open", None)
            closer = getattr(inner, "close", None)
            if callable(opener) != callable(closer):
                raise ValueError(
                    f"inner adapter {type(inner).__name__} exposes open() or close() "
                    "without its pair; nothing would release what open() allocates"
                )
            if callable(opener):
                self._inner_opener = opener
                self._inner_closer = closer

        providers: list[str] = []
        injected_reasoning = False
        assistant_history = False
        media_seen = False
        for terms in corpus.request_terms_view():
            provider = terms.get("provider")
            if isinstance(provider, str) and provider not in providers:
                providers.append(provider)
            messages = terms.get("messages")
            for message in messages if isinstance(messages, list) else ():
                if _is_injected_reasoning(message):
                    injected_reasoning = True
                if _is_assistant_message(message):
                    assistant_history = True
                content = message.get("content") if isinstance(message, Mapping) else None
                if isinstance(content, list) and any(_is_media_block(part) for part in content):
                    media_seen = True
        if len(providers) > 1:
            raise ValueError(
                "replay sources recorded more than one provider "
                f"({', '.join(sorted(providers))}); replay one provider per adapter"
            )

        if provider_name is _AUTO:
            recorded = providers[0] if providers else None
            if injected_reasoning:
                declared = recorded
            elif assistant_history and any(
                body.get("reasoning") for body in corpus.response_bodies_view()
            ):
                # Answers carried reasoning, and a recorded request had a turn behind it in
                # which the block would have appeared: the original did not declare, so
                # neither may this adapter -- declaring would make the loop inject blocks the
                # recorded preimages never had.
                declared = None
            else:
                # Including the corpus that cannot testify: every recorded request is a first
                # turn, so no injected block could exist either way. Silence is not evidence
                # for the negative, and reading it as one breaks the shipped gateway default,
                # whose key term is the RELAYED provider ("openai") while the config names the
                # transport ("gateway") -- declining there misses every lookup in the run.
                declared = recorded
        else:
            declared = provider_name
        if declared:
            self.provider_name = declared

        self.supports_multimodal = (
            media_seen if supports_multimodal is _AUTO else bool(supports_multimodal)
        )
        self.wire_image_encoding = wire_image_encoding
        if config is not None:
            # Inert under the loop (it always authors request.model); a standalone
            # ModelCallRunner caller may want the effective-config probe to answer.
            self.config = config

    # --- the call ----------------------------------------------------------------------------

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        lookup = replay_lookup(request, self)
        if lookup.result.status != "ok":
            return self._serve_miss(
                request,
                ReplayMissReason(
                    MISS_NO_KEY,
                    "no replay key could be issued for this request "
                    f"({lookup.result.status}); an unkeyable call was never recorded either",
                ),
                digest=None,
                held=None,
            )
        digest = lookup.result.digest
        outcome, held = self._replayed_turn_or_miss(digest)
        if isinstance(outcome, ReplayMissReason):
            if outcome.reason == MISS_ABSENT:
                outcome = self._corpus.diagnose(
                    lookup.payload,
                    generation=_REQUEST_DIGEST_GENERATION,
                    digest=digest,
                )
            return self._serve_miss(request, outcome, digest=digest, held=held)
        return outcome

    def _replayed_turn_or_miss(
        self, digest: str
    ) -> tuple[ModelTurn | ReplayMissReason, int | None]:
        """The turn, or the refusal -- and the slot the corpus is holding open for us, if any.

        ``consume`` no longer advances on a refusal, so a miss it returns holds nothing; a
        record it *did* hand over and reconstruction then rejected holds exactly one slot,
        which the caller must either spend (it served the call live) or release (the turn
        parks and will be re-attempted).
        """

        outcome = self._corpus.consume(digest, generation=_REQUEST_DIGEST_GENERATION)
        if isinstance(outcome, ReplayMissReason):
            return outcome, None
        turn = self._reconstruct(outcome)
        return turn, (outcome.slot if isinstance(turn, ReplayMissReason) else None)

    def _reconstruct(self, hit: ReplayedResponse) -> ModelTurn | ReplayMissReason:
        """The recorded body as a real ``ModelTurn``, or the refusal it earns.

        Strict on arrival for the same reason the reader re-hashes chunks: the loop consumes
        ``ToolCall`` attributes and ``turn.reasoning`` items structurally, and a corpus is
        run-directory data -- replaying a fabrication is the one thing this adapter exists
        to never do. Every raise below names structure, never values; the text lands on
        public surfaces.
        """

        body = hit.body
        try:
            missing = [name for name in RECORDED_TURN_FIELDS if name not in body]
            if missing:
                # Without this the answer below reconstructs into an *empty* turn, which the
                # loop rejects as "neither final text nor tool calls" -- a `model_error` that
                # kills the run and blames a model that was never called. A recorded turn
                # carries every field the writer declares; anything else is not one.
                raise ValueError(
                    "the recorded answer is missing the fields a recorded turn carries "
                    f"({', '.join(missing)})"
                )
            calls: list[ToolCall] = []
            for call in body.get("tool_calls") or ():
                if (
                    not isinstance(call, Mapping)
                    or not isinstance(call.get("id"), str)
                    or not isinstance(call.get("name"), str)
                    or not isinstance(call.get("arguments"), dict)
                ):
                    raise ValueError("a recorded tool call is missing its id/name/arguments triple")
                calls.append(
                    ToolCall(id=call["id"], name=call["name"], arguments=dict(call["arguments"]))
                )
            reasoning: list[dict[str, Any]] = []
            for item in body.get("reasoning") or ():
                if not isinstance(item, Mapping):
                    raise ValueError("a recorded reasoning item is not an object")
                reasoning.append(dict(item))
            usage = body.get("usage") or {}
            if not isinstance(usage, Mapping):
                raise ValueError("the recorded usage is not an object")
            unportable = unportable_usage_key(usage)
            if unportable is not None:
                # The loop refuses these too, three layers down, as `model_bad_response` --
                # a kill, reported against the adapter. Refusing here keeps a corrupt corpus
                # inside the miss vocabulary, where the operator can act on it.
                raise ValueError(f"the recorded usage {unportable} is not a non-negative integer")
            for name in ("response_id", "final_text", "stop_reason"):
                if body.get(name) is not None and not isinstance(body.get(name), str):
                    raise ValueError(f"the recorded {name} is neither null nor a string")
            return ModelTurn(
                response_id=body.get("response_id"),
                final_text=body.get("final_text"),
                tool_calls=tuple(calls),
                usage=dict(usage),
                raw={},
                reasoning=tuple(reasoning),
                stop_reason=body.get("stop_reason"),
                provider_retried=bool(body.get("provider_retried", False)),
            )
        except Exception as error:  # noqa: BLE001 - one unreplayable record, one refusal
            return ReplayMissReason(
                MISS_NOT_RECORDED,
                f"the recorded answer could not be reconstructed ({error}); "
                f"run {hit.run_id} call_index {hit.call_index}",
            )

    def _serve_miss(
        self,
        request: ModelRequest,
        miss: ReplayMissReason,
        *,
        digest: str | None,
        held: int | None,
    ) -> ModelTurn:
        """Fall through to the inner adapter, or refuse -- and settle the refused slot.

        Not two exits but three, and only one of them is a call that happened. Serving it live
        moves the conversation past this slot, so the slot is spent and the next call meets the
        next recording. Raising -- because there is no inner, or because the inner raised --
        parks the turn for an idempotent re-attempt, so the slot goes back: the re-attempt must
        earn the same refusal, not the answer that belonged to the call after it.

        The two unserved exits settle through one function rather than two hand-kept sites,
        because that is the seam this repair was needed at twice: the rule was written for the
        refusal that leaves the cursor standing and never bound on the record ``consume``
        handed over and reconstruction then rejected.
        """

        if self._inner is not None:
            try:
                turn = self._inner.next_turn(request)
            except BaseException:
                self._settle_unserved(digest, held)
                raise
            # Only now. Spending before the call would claim the conversation moved past this
            # slot on the strength of an attempt: a live adapter that raises recoverably (a
            # 429 is enough) parks the turn for an idempotent re-attempt, and the re-attempt
            # would be answered with the next call's recording -- the same silent substitution
            # the two-phase consume exists to prevent, reached through the other exit.
            if digest is not None and held is None and miss.slot is not None:
                self._corpus.spend_refused(digest, miss.slot)
            return turn
        self._settle_unserved(digest, held)
        raise ReplayMiss(
            f"replay miss ({miss.reason}): {miss.detail}", provider_error_code=miss.reason
        )

    def _settle_unserved(self, digest: str | None, held: int | None) -> None:
        """Give back a slot ``consume`` handed over for a call that then did not happen.

        A refusal that left the cursor standing holds nothing and needs nothing; only a record
        reconstruction rejected has already moved the cursor, and only that one has to be put
        back before the turn parks.
        """

        if digest is not None and held is not None:
            self._corpus.release(digest, held)

    # --- lifecycle ----------------------------------------------------------------------------

    def open(self) -> None:
        """Forwarded when the inner has a lifecycle; a no-op otherwise, so the CLI's
        open/close probe drives the wrapper it sees and reaches what that wrapper wraps."""

        if self._inner_opener is not None:
            self._inner_opener()

    def close(self) -> None:
        if self._inner_closer is not None:
            self._inner_closer()
