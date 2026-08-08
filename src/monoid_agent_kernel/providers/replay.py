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

import inspect
import threading
import weakref
from pathlib import Path
from typing import Any, Mapping, Sequence

from monoid_agent_kernel.core.payload_replay import (
    _short,
    _where,
    MISS_ABSENT,
    MISS_NO_KEY,
    MISS_NOT_RECORDED,
    REPLAY_MISS_REASONS,
    ReplayCorpus,
    ReplayMissReason,
    ReplayTake,
    ReplayedResponse,
)
from monoid_agent_kernel.core._sync_bridge import dispose_unawaited, is_async_callable
from monoid_agent_kernel.core.media import WIRE_MEDIA_CARRIERS
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
            if not callable(next_turn):
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
            # Every callable this wrapper forwards, gated by one predicate over the census
            # rather than by a hand-picked member. `next_turn` earned this check and `open`/
            # `close` inherited nothing from it, so an `async def` lifecycle pair passed the
            # pairing test above and then neither half ran: `open()` called the coroutine
            # function, discarded the coroutine, and reported success to a CLI probe that has
            # only this wrapper to ask -- the exact outcome the pairing check exists to
            # prevent. Three hand-written checks can be half-applied; a census cannot.
            asynchronous = [
                name
                for name, member in (
                    ("next_turn", next_turn),
                    ("open", opener),
                    ("close", closer),
                )
                if callable(member) and is_async_callable(member)
            ]
            if asynchronous:
                raise ValueError(
                    f"the fallthrough inner adapter's {', '.join(asynchronous)} "
                    f"{'is' if len(asynchronous) == 1 else 'are'} asynchronous; this wrapper "
                    "is synchronous and would hand back a coroutine nothing awaits"
                )
            if callable(opener):
                self._inner_opener = opener
                self._inner_closer = closer

        providers: list[str] = []
        injected_reasoning = False
        media_seen = False
        runs_with_history: set[str] = set()
        for run_id, terms in corpus.request_terms_view():
            provider = terms.get("provider")
            if isinstance(provider, str) and provider not in providers:
                providers.append(provider)
            messages = terms.get("messages")
            for message in messages if isinstance(messages, list) else ():
                if _is_injected_reasoning(message):
                    injected_reasoning = True
                if _is_assistant_message(message):
                    runs_with_history.add(run_id)
                for carrier in WIRE_MEDIA_CARRIERS:
                    parts = message.get(carrier) if isinstance(message, Mapping) else None
                    if isinstance(parts, list) and any(_is_media_block(p) for p in parts):
                        media_seen = True
        if len(providers) > 1:
            raise ValueError(
                "replay sources recorded more than one provider "
                f"({_short(', '.join(sorted(providers)))}); replay one provider per adapter"
            )

        if provider_name is _AUTO:
            recorded = providers[0] if providers else None
            if injected_reasoning:
                declared = recorded
            elif runs_with_history & {
                run_id for run_id, body in corpus.response_bodies_view() if body.get("reasoning")
            }:
                # Answers carried reasoning, and a recorded request had a turn behind it in
                # which the block would have appeared: the original did not declare, so
                # neither may this adapter -- declaring would make the loop inject blocks the
                # recorded preimages never had.
                #
                # Both halves must come from ONE run. "The block would have appeared" is a claim
                # about a continuation of the conversation the reasoning came from; of a
                # continuation in some other run it is simply false. Two global `any`s let a
                # union combine a single-turn source that recorded reasoning with a multi-turn
                # source that recorded none -- neither evidence alone, both declaring alone --
                # and decline, which drops the declaration and, under the shipped gateway
                # default, makes every recomputed key name the transport instead of the relayed
                # provider: the preflight then refuses a corpus and a config that are both right.
                #
                # The run is a NARROWING, not a proof. One run can still hold a single-turn call
                # that recorded reasoning alongside a continuation of a different conversation
                # whose own upstream answer carried none. Proving it needs the continuation
                # matched to the answer it continues from -- message-prefix against the recorded
                # requests -- which the corpus does not model today. Both error directions fail
                # closed (a wrong decline refuses at preflight; a wrong declaration injects
                # blocks the preimages never had and misses), so this buys the reachable half.
                declared = None
            elif corpus.unreadable_requests():
                # A request record the reader could not parse is present, hashes correctly, and
                # says nothing -- so this corpus is not silent, it is NARROWED, and the two must
                # not take the same horn. The evidence for not declaring lives in the requests,
                # so the ones that went missing are exactly the ones that could have carried it:
                # taking the "cannot testify" horn here concludes more confidently from strictly
                # less. Measured -- with both requests readable the derivation declines; with the
                # one carrying assistant history unreadable it declared, and the loop would
                # inject blocks the recorded preimages never had.
                declared = None
            else:
                # Including the corpus that cannot testify: every recorded request is a first
                # turn, so no injected block could exist either way. Silence is not evidence
                # for the negative, and reading it as one breaks the shipped gateway default,
                # whose key term is the RELAYED provider ("openai") while the config names the
                # transport ("gateway") -- declining there misses every lookup in the run.
                # Reachable only when every request record was readable; see the branch above.
                declared = recorded
        else:
            declared = provider_name
        if declared:
            self.provider_name = declared

        self.supports_multimodal = (
            media_seen if supports_multimodal is _AUTO else bool(supports_multimodal)
        )
        self._served_slots: dict[str, list[tuple[int, weakref.ref[ModelTurn]]]] = {}
        self._pending_releases: dict[str, set[int]] = {}
        self._served_lock = threading.Lock()
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
                    # Labelled, because the statuses overlap the miss vocabulary by name:
                    # an unlabelled `(absent)` two words after `(no_key)` reads as a second,
                    # contradicting reason rather than as the key derivation's own verdict.
                    f"(key status: {lookup.result.status}); an unkeyable call was never "
                    "recorded either",
                ),
                take=None,
            )
        digest = lookup.result.digest
        # Everything from here settles by leaving the block. The two ways a take goes unusable
        # settle in opposite directions -- a standing refusal is spent forward, a rejected
        # record is given back -- and choosing between them at the call site is what every
        # route into the substitution failure got wrong. The take owns that choice now; this
        # function only ever says whether the call happened.
        with self._corpus.take(digest, generation=_REQUEST_DIGEST_GENERATION) as take:
            if take.hit is not None:
                outcome = self._reconstruct(take.hit)
                if isinstance(outcome, ModelTurn):
                    take.served()
                    self._note_served(digest, take.hit.slot, outcome)
                    return outcome
                miss = outcome
            else:
                miss = take.miss
                if miss.reason == MISS_ABSENT:
                    # Re-authored for the operator, not re-settled: the take still remembers
                    # the slot the original refusal stood on, so a diagnosis cannot drop it.
                    miss = self._corpus.diagnose(
                        lookup.payload,
                        generation=_REQUEST_DIGEST_GENERATION,
                        digest=digest,
                    )
            return self._serve_miss(request, miss, take=take)

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
            # One table rather than three hand-written guards, because the first repair here
            # bounded `provider_retried` alone and left `or ()` / `or {}` standing three lines
            # up: a damaged `false`, `0` or `""` became an empty container, and for `tool_calls`
            # that reconstructs into a perfectly successful final-text turn -- the corpus's
            # damage answered rather than refused. The writer projects these through `list(...)`
            # and `dict(...)`, so no recorded body legitimately carries anything else.
            for name, kind, noun in (
                ("tool_calls", list, "a list"),
                ("reasoning", list, "a list"),
                ("usage", Mapping, "an object"),
            ):
                if not isinstance(body.get(name), kind):
                    raise ValueError(f"the recorded {name} is not {noun}")
            calls: list[ToolCall] = []
            for call in body["tool_calls"]:
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
            for item in body["reasoning"]:
                if not isinstance(item, Mapping):
                    raise ValueError("a recorded reasoning item is not an object")
                reasoning.append(dict(item))
            usage = body["usage"]
            unportable = unportable_usage_key(usage)
            if unportable is not None:
                # The loop refuses these too, three layers down, as `model_bad_response` --
                # a kill, reported against the adapter. Refusing here keeps a corrupt corpus
                # inside the miss vocabulary, where the operator can act on it.
                raise ValueError(
                    f"the recorded usage {_short(str(unportable))} is not a non-negative integer"
                )
            for name in ("response_id", "final_text", "stop_reason"):
                if body.get(name) is not None and not isinstance(body.get(name), str):
                    raise ValueError(f"the recorded {name} is neither null nor a string")
            retried = body.get("provider_retried", False)
            if not isinstance(retried, bool):
                # ``bool()`` accepts every JSON scalar, and the string "false" is truthy: coercing
                # replays a recorded "not retried" as RETRIED. Bodies are deliberately open-shaped
                # in the payload schema, so a damaged corpus reaches here, and inventing an audit
                # value from one is the single thing every neighbour above refuses to do.
                raise ValueError("the recorded provider_retried is not a boolean")
            return ModelTurn(
                response_id=body.get("response_id"),
                final_text=body.get("final_text"),
                tool_calls=tuple(calls),
                usage=dict(usage),
                raw={},
                reasoning=tuple(reasoning),
                stop_reason=body.get("stop_reason"),
                provider_retried=retried,
            )
        except Exception as error:  # noqa: BLE001 - one unreplayable record, one refusal
            return ReplayMissReason(
                MISS_NOT_RECORDED,
                f"the recorded answer could not be reconstructed ({_short(str(error))}); "
                + _where(hit),
            )

    def _note_served(self, digest: str, slot: int, turn: ModelTurn) -> None:
        """Remember which slot produced ``turn``, weakly and per in-flight call.

        Per call, not per key: nothing serialises ``next_turn`` against a shared adapter, and a
        family of sibling subagents shares one. A single note per key let whichever of two
        concurrent calls served last overwrite the other's turn-to-slot association, so discarding
        both gave back at most one slot and the lost one stayed consumed but undelivered -- the
        next caller then received the answer recorded for a different call.

        Weakly, because the note must not be the reason a recorded body stays in memory. Holding
        the turn made every accepted hit a permanent retention: one ``ModelTurn`` per key, with its
        copied tool-call/reasoning/usage containers, for the adapter's whole life, and recorded
        bodies reach the 8 MB payload ceiling. The turn is alive at the only moment this is read --
        the runner holds it across the boundary check that discards it -- so a weak reference does
        the job and stops doing it afterwards, with no accept-side hook to invent.

        Dead references are pruned here rather than on a timer: the next call on a key is the only
        moment that key's list can grow, so it is the moment to shrink it.
        """

        with self._served_lock:
            held = [pair for pair in self._served_slots.get(digest, ()) if pair[1]() is not None]
            held.append((slot, weakref.ref(turn)))
            self._served_slots[digest] = held

    def discard_turn(self, request: ModelRequest, turn: ModelTurn) -> None:
        """Take back an answer the run threw away without ever seeing it.

        ``consume`` advances the cursor when the answer is handed over, so by the time a run
        boundary wins the race the recording is already spent. Without this the next call on this
        corpus -- a sibling subagent, or a later call in a run that reused the adapter -- is
        served the FOLLOWING recording: a structurally valid turn belonging to a different call,
        exit 0, ledger success, ``monoid validate`` clean. That is the whole failure class.

        Keyed by recomputing the request's own digest rather than by object identity, so a
        recycled ``id()`` can never release a slot belonging to a different key. The digest alone
        is not enough, though, and the first version of this shipped believing it was: the entry
        is cleared by nothing on success, so it outlives its call, and a LATER call on the same
        key -- exhausted by then, so served live through the fallthrough -- would pop the earlier
        call's slot when discarded. ``release`` cannot catch that, because an exhausted call never
        moved the cursor, so ``cursor == slot + 1`` still holds for the delivered slot and the
        rewind succeeds. A recording already handed to one call gets handed to another: this
        module's whole failure class, arriving through the repair meant to close it.

        So the served turn travels with the slot and is compared by identity here. The dict holds
        the object, which is what makes identity sound -- an ``id()`` cannot be recycled while the
        entry that would match it is still reachable.

        A live fallthrough answer is deliberately NOT taken back -- that call reached a provider
        and was paid for, and the corpus has no unspend primitive for a refusal spent forward --
        and the identity check is now what enforces that, rather than the docstring. See
        ``docs/CONTRACTS.md``.
        """

        lookup = replay_lookup(request, self)
        if lookup.result.status != "ok":
            return
        digest = lookup.result.digest
        slot: int | None = None
        with self._served_lock:
            remaining: list[tuple[int, weakref.ref[ModelTurn]]] = []
            for pair in self._served_slots.get(digest, ()):
                alive = pair[1]()
                if alive is None:
                    continue
                if slot is None and alive is turn:
                    slot = pair[0]
                    continue
                remaining.append(pair)
            if remaining:
                self._served_slots[digest] = remaining
            else:
                self._served_slots.pop(digest, None)
        if slot is None:
            return
        with self._served_lock:
            self._pending_releases.setdefault(digest, set()).add(slot)
        self._compact_releases(digest)

    def _compact_releases(self, digest: str) -> None:
        """Give back every consecutively discarded slot the cursor can still reach.

        ``release`` takes only the slot the cursor stands one past, so discard order decided
        whether an answer came back at all: two concurrent calls on one key take slots 0 and 1,
        and a slot-0 discard arriving first was simply dropped -- the cursor was already at 2 --
        leaving that answer consumed and undelivered while the next caller took a later one.
        Concurrent identical calls arrive in no order at all, so honouring one of the two is
        honouring neither.

        A discard that cannot be honoured yet therefore stays pending, and every release retries
        the highest pending slot: releasing 1 puts the cursor where 0 becomes releasable, and the
        loop walks the whole consecutive run back. A slot the cursor can never reach again --
        because the answer above it was delivered and kept -- simply stays in the set, which is
        bounded by the key's own queue length.

        The corpus call stays outside ``_served_lock``: it takes its own, and this is the only
        place that would hold both. Each iteration re-reads the set under the lock, so two
        concurrent discards can interleave without either losing its slot -- a release that loses
        the race fails rather than corrupting, and the winner's own loop picks the slot up.
        """

        while True:
            with self._served_lock:
                pending = self._pending_releases.get(digest)
                if not pending:
                    return
                slot = max(pending)
            if not self._corpus.release(digest, slot):
                return
            with self._served_lock:
                remaining = self._pending_releases.get(digest)
                if remaining is not None:
                    remaining.discard(slot)
                    if not remaining:
                        self._pending_releases.pop(digest, None)

    def _serve_miss(
        self,
        request: ModelRequest,
        miss: ReplayMissReason,
        *,
        take: ReplayTake | None,
    ) -> ModelTurn:
        """Fall through to the inner adapter, or refuse.

        Three exits, and only one of them is a call that happened. Serving it live moves the
        conversation past this slot, so the take is told the call happened and the next call
        meets the next recording. Raising -- because there is no inner, or because the inner
        raised, or because it handed back an awaitable this wrapper cannot drive -- parks
        the turn for an idempotent re-attempt, and leaving the block without the declaration is
        what gives the slot back. An inner returning some *other* non-turn is not refused here:
        it goes to the runner, which classifies it terminally, and the slot is spent because the
        call did happen. Two review axes read the earlier wording as a promise to check the
        shape. There is nothing to remember here and nothing to choose between: this
        function knows only whether the call happened, which is the only thing it can know.

        ``take`` is None for the one call that took nothing: an unkeyable request never
        reached the corpus, so no slot is owed either way.
        """

        if self._inner is not None:
            turn = self._inner.next_turn(request)
            if inspect.isawaitable(turn):
                # `_adrive` awaits whatever a synchronous adapter hands back, so this return
                # means the provider has not been called yet -- the awaited call still has
                # every chance to fail, and declaring it served would pay for a call that has
                # not happened. No declaration-side gate can see this shape:
                # `iscoroutinefunction` is False for a plain `def` that returns a coroutine,
                # so the result is the only place left to ask.
                # Not on a loop thread: this is the abandonable daemon worker. On the live-
                # loop default the disposal itself raises under asyncio debug mode and
                # REPLACES the typed error below, losing the config_recoverable
                # classification the guard exists to carry. Disposal is hygiene; it must
                # never become the verdict.
                dispose_unawaited(turn, on_live_loop=False)
                raise ModelAdapterError(
                    "the fallthrough inner adapter returned an awaitable from a "
                    "synchronous next_turn; this wrapper cannot drive it",
                    retryable=False,
                    # The remedy is operator-shaped -- build the wrapper around an inner it
                    # can drive -- so the session survives to be resumed against a fixed
                    # configuration, rather than dying as an unclassified kill that takes the
                    # checkpoints with it.
                    config_recoverable=True,
                )
            if take is not None:
                take.served()
            return turn
        raise ReplayMiss(
            f"replay miss ({miss.reason}): {miss.detail}", provider_error_code=miss.reason
        )

    # --- lifecycle ----------------------------------------------------------------------------

    def _forward_lifecycle(self, name: str, member: Any) -> None:
        """Drive one forwarded lifecycle half, refusing a result nothing will await.

        The constructor's census already rejects an async pair, so this is the second binding
        of one rule rather than its only one: an inner that resolves ``open`` to a coroutine
        function at call time -- a ``__getattr__``, a hot-swapped attribute -- is invisible to
        a check that ran at construction, and it fails the same way. The coroutine is
        discarded, the inner is never entered, and the CLI's synchronous probe has only this
        wrapper to ask, so it reports success on a lifecycle that did not happen.
        """

        result = member()
        if inspect.isawaitable(result):
            # The CLI thread, not a loop thread -- same reason as the call site above.
            dispose_unawaited(result, on_live_loop=False)
            raise ValueError(
                f"the fallthrough inner adapter's {name}() returned an awaitable; this "
                "wrapper is synchronous and nothing would await it, so the inner would never "
                f"be {'opened' if name == 'open' else 'closed'}"
            )

    def open(self) -> None:
        """Forwarded when the inner has a lifecycle; a no-op otherwise, so the CLI's
        open/close probe drives the wrapper it sees and reaches what that wrapper wraps."""

        if self._inner_opener is not None:
            self._forward_lifecycle("open", self._inner_opener)

    def close(self) -> None:
        if self._inner_closer is not None:
            self._forward_lifecycle("close", self._inner_closer)
