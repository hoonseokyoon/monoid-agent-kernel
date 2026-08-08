"""The replay adapter: recorded answers as a model, misses as typed refusals, nothing invented.

W6-4b B3. ``tests/test_payload_replay.py`` pins the corpus reader; this pins the adapter that
stands where a provider stood: impersonation derived from evidence (D-h), reconstruction that
is verbatim or refused, a fallthrough that is the only path to a live call, and lifecycle
forwarding that keeps the CLI's open/close probe honest about what the wrapper wraps.
"""

from __future__ import annotations

import asyncio
import gc
import json
from pathlib import Path
from typing import Any

import pytest

from monoid_agent_kernel.core._util import sha256_bytes
from monoid_agent_kernel.core.model_payloads import (
    MODEL_PAYLOADS_FILENAME,
    model_response_record,
)
from monoid_agent_kernel.core.payload_replay import (
    MISS_ABSENT,
    MISS_IDENTITY_MISMATCH,
    MISS_NO_KEY,
    MISS_NOT_RECORDED,
    ReplayCorpus,
)
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.model_call import ModelCallRunner, SettledModelCall
from monoid_agent_kernel.providers._request_identity import replay_lookup
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn, ToolCall
from monoid_agent_kernel.providers.replay import ReplayMiss, ReplayModelAdapter
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.recorder import AgentRecorder

_MARKER = "SECRET-CONVERSATION-4X"


class _OriginalAdapter:
    """The recording-side adapter; declares only what a test hands it."""

    def __init__(self, turns: list[ModelTurn], *, provider_name: str | None = None):
        self.turns = turns
        if provider_name is not None:
            self.provider_name = provider_name

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        del request
        return self.turns.pop(0)


class _CountingInner:
    def __init__(self) -> None:
        self.calls = 0
        self.opened = 0
        self.closed = 0

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        del request
        self.calls += 1
        return ModelTurn(final_text="live answer")

    def open(self) -> None:
        self.opened += 1

    def close(self) -> None:
        self.closed += 1


def _record(
    base: Path, adapter: Any, requests: list[ModelRequest], *, run_id: str = "run-1"
) -> list[str]:
    recorder = AgentRecorder(
        base / "runs",
        run_id,
        status_file=False,
        model_calls_file=True,
        model_payload_file=True,
    )
    digests: list[str] = []

    def sink(call: SettledModelCall) -> None:
        digests.append(call.receipt.request_digest)
        recorder.record_settled_call(call)

    runner = ModelCallRunner(adapter=adapter, settled_sink=sink, capture_request_preimage=True)
    for request in requests:
        try:
            asyncio.run(runner.acall(request))
        except ModelAdapterError:
            pass
    recorder.close()
    return digests


def _request(text: str = "hi", **changes: Any) -> ModelRequest:
    return ModelRequest(instruction=text, system_prompt="sys", tools=(), **changes)


def _replay(base: Path, *run_ids: str, **kwargs: Any) -> ReplayModelAdapter:
    sources = [base / "runs" / run_id for run_id in (run_ids or ("run-1",))]
    return ReplayModelAdapter(sources, **kwargs)


def _call(adapter: Any, request: ModelRequest) -> ModelTurn:
    turn, _receipt = asyncio.run(ModelCallRunner(adapter=adapter).acall(request))
    return turn


def _prepend_refused_answer(
    base: Path,
    digest: str,
    *,
    run_id: str = "run-1",
    body: Any = None,
    unrecorded_reason: str = "too_large",
    after: int = 0,
) -> None:
    """Put an unusable record in front of the recorded answer, so slot 0 cannot be given back
    and slot 1 can.

    The two ways a slot goes unusable reach the corpus differently and must both be driven:
    ``unrecorded_reason`` is refused before the cursor moves, while a body that only
    *reconstruction* rejects is handed over first and has to be given back.
    """

    path = base / "runs" / run_id / MODEL_PAYLOADS_FILENAME
    refused = json.dumps(
        model_response_record(
            body,
            call_index=0,
            request_digest=digest,
            unrecorded_reason=unrecorded_reason,
            run_id=run_id,
            root_run_id=run_id,
            recorded_at="2026-08-08T00:00:00Z",
        ),
        sort_keys=True,
    )
    out: list[str] = []
    inserted = False
    seen = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if json.loads(line).get("kind") == "model_response":
            if not inserted and seen == after:
                out.append(refused)
                inserted = True
            seen += 1
        out.append(line)
    assert inserted
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# --- hits are verbatim ---------------------------------------------------------------------


def test_a_recorded_run_replays_with_zero_live_calls(tmp_path: Path) -> None:
    original = [
        ModelTurn(
            response_id="r1",
            tool_calls=(ToolCall(id="c1", name="fs_list", arguments={"path": "."}),),
            usage={"input_tokens": 7, "output_tokens": 2},
        ),
        ModelTurn(
            response_id="r2",
            final_text="done",
            usage={"input_tokens": 9, "output_tokens": 3},
            reasoning=({"type": "reasoning", "encrypted_content": "OPAQUE"},),
            stop_reason="stop",
        ),
    ]
    first = _request(_MARKER)
    second = _request(_MARKER, previous_turn_handle="r1")
    _record(tmp_path, _OriginalAdapter(list(original)), [first, second])

    inner = _CountingInner()
    adapter = _replay(tmp_path, inner=inner)
    replayed_first = _call(adapter, _request(_MARKER))
    replayed_second = _call(adapter, _request(_MARKER, previous_turn_handle="r1"))

    assert inner.calls == 0
    assert replayed_first.response_id == "r1"
    assert isinstance(replayed_first.tool_calls[0], ToolCall)
    assert replayed_first.tool_calls[0].__dict__ == {
        "id": "c1",
        "name": "fs_list",
        "arguments": {"path": "."},
    }
    assert replayed_second.final_text == "done"
    assert replayed_second.usage["output_tokens"] == 3
    assert replayed_second.reasoning == ({"type": "reasoning", "encrypted_content": "OPAQUE"},)
    assert replayed_second.stop_reason == "stop"
    assert replayed_second.provider_retried is False
    assert replayed_second.raw == {}, "raw={} is the honest 'this is a replay' statement"


# --- misses are typed, content-free, and park-shaped -----------------------------------------


def test_a_miss_is_a_replay_miss_with_the_approved_flags(tmp_path: Path) -> None:
    _record(tmp_path, _OriginalAdapter([ModelTurn(final_text="x")]), [_request(_MARKER)])
    adapter = _replay(tmp_path)

    with pytest.raises(ReplayMiss) as caught:
        _call(adapter, _request("a different conversation " + _MARKER))

    miss = caught.value
    assert miss.error_code == "replay_miss"
    assert miss.retryable is False
    assert miss.config_recoverable is True
    assert miss.provider_error_code == MISS_ABSENT
    assert _MARKER not in str(miss), "the exception message is a public surface"


def test_an_identity_miss_names_the_diverging_config(tmp_path: Path) -> None:
    _record(tmp_path, _OriginalAdapter([ModelTurn(final_text="x")]), [_request(_MARKER)])
    adapter = _replay(tmp_path)

    with pytest.raises(ReplayMiss) as caught:
        _call(adapter, _request(_MARKER, model=ModelConfig(model="elsewhere-9")))

    assert caught.value.provider_error_code == MISS_IDENTITY_MISMATCH
    assert "elsewhere-9" in str(caught.value)
    assert _MARKER not in str(caught.value)


def test_an_unkeyable_request_is_no_key(tmp_path: Path) -> None:
    _record(tmp_path, _OriginalAdapter([ModelTurn(final_text="x")]), [_request()])
    adapter = _replay(tmp_path)
    hostile = _request(messages=[{"role": "user", "content": {1, 2}}])  # a set never encodes

    with pytest.raises(ReplayMiss) as caught:
        adapter.next_turn(hostile)

    assert caught.value.provider_error_code == MISS_NO_KEY


def test_a_corrupt_recorded_body_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    """A tool call without its triple cannot become a real ToolCall; replaying a fabrication
    is the one thing the adapter must never do, so the record reads as unrecoverable."""

    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    envelope = {"run_id": "run-1", "root_run_id": "run-1", "recorded_at": "2026-08-08T00:00:00Z"}
    digest = sha256_bytes(b"corrupt-body")
    body = {
        "response_id": "r",
        "final_text": None,
        "tool_calls": [{"id": "c1", "arguments": {}}],  # no name
        "reasoning": [],
        "usage": {},
        "stop_reason": None,
        "provider_retried": False,
    }
    (run_dir / MODEL_PAYLOADS_FILENAME).write_text(
        json.dumps(
            model_response_record(
                body, call_index=0, request_digest=digest, unrecorded_reason="", **envelope
            ),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    corpus = ReplayCorpus.load([run_dir])
    adapter = ReplayModelAdapter(corpus, provider_name=None)

    outcome, held = adapter._replayed_turn_or_miss(digest)

    assert not isinstance(outcome, ModelTurn)
    assert outcome.reason == MISS_NOT_RECORDED
    assert held == 0, "reconstruction rejected a record the corpus had handed over"


def test_the_vocabulary_is_a_door_not_a_convention() -> None:
    with pytest.raises(ValueError, match="replay miss reason"):
        ReplayMiss("nope", provider_error_code="made_up_reason")


# --- fallthrough and lifecycle ----------------------------------------------------------------


def test_fallthrough_hands_only_misses_to_the_inner(tmp_path: Path) -> None:
    _record(tmp_path, _OriginalAdapter([ModelTurn(final_text="recorded")]), [_request()])
    inner = _CountingInner()
    adapter = _replay(tmp_path, inner=inner)

    hit = _call(adapter, _request())
    live = _call(adapter, _request("never recorded"))

    assert hit.final_text == "recorded"
    assert live.final_text == "live answer"
    assert inner.calls == 1


def test_the_lifecycle_pair_forwards_to_the_inner(tmp_path: Path) -> None:
    _record(tmp_path, _OriginalAdapter([ModelTurn(final_text="x")]), [_request()])
    inner = _CountingInner()
    adapter = _replay(tmp_path, inner=inner)

    adapter.open()
    adapter.close()

    assert (inner.opened, inner.closed) == (1, 1)


def test_a_half_lifecycled_inner_is_rejected_at_construction(tmp_path: Path) -> None:
    _record(tmp_path, _OriginalAdapter([ModelTurn(final_text="x")]), [_request()])

    class _OpenOnly(_CountingInner):
        close = None  # type: ignore[assignment]

    with pytest.raises(ValueError, match="open"):
        _replay(tmp_path, inner=_OpenOnly())


def test_an_async_only_inner_is_rejected_at_construction(tmp_path: Path) -> None:
    _record(tmp_path, _OriginalAdapter([ModelTurn(final_text="x")]), [_request()])

    class _AsyncOnly:
        async def anext_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(final_text="never")

    with pytest.raises(ValueError, match="next_turn"):
        _replay(tmp_path, inner=_AsyncOnly())


# --- impersonation is derived from evidence (D-h) ---------------------------------------------


def test_a_reasoning_free_corpus_declares_the_recorded_provider(tmp_path: Path) -> None:
    """Shape (c): no reasoning anywhere, so declaring is safe and makes the key term
    independent of the replay run's config.provider -- pinned by replaying under a config
    that names a different provider and still hitting."""

    _record(
        tmp_path,
        _OriginalAdapter([ModelTurn(final_text="x")], provider_name="openai"),
        [_request()],
    )
    adapter = _replay(tmp_path)

    assert getattr(adapter, "provider_name", None) == "openai"
    turn = _call(adapter, _request(model=ModelConfig(provider="gateway")))
    assert turn.final_text == "x"


def test_reinjected_reasoning_in_the_record_means_declare(tmp_path: Path) -> None:
    """Shape (a): the recorded conversation itself carries the loop's re-injected reasoning
    block, which only a declared original produces -- so the replay declares too."""

    injected = {
        "role": "assistant",
        "content": "",
        "reasoning": {"provider": "openai", "model": "gpt-5.5", "items": [{"type": "reasoning"}]},
    }
    _record(
        tmp_path,
        _OriginalAdapter(
            [ModelTurn(final_text="x", reasoning=({"type": "reasoning"},))],
            provider_name="openai",
        ),
        [_request(messages=[{"role": "user", "content": "hi"}, injected])],
    )
    adapter = _replay(tmp_path)

    assert getattr(adapter, "provider_name", None) == "openai"


def test_reasoning_answers_with_no_reinjected_trace_mean_do_not_declare(tmp_path: Path) -> None:
    """Shape (b): answers carry reasoning, and a recorded request that HAD a turn behind it
    still carried no injected block -- the undeclared-original shape. Declaring here would
    make the loop inject blocks the original preimages never had, so the adapter must not.

    The assistant history is the whole evidence. The loop appends the block after a call, so
    only a request with a turn in front of it could ever have carried one; a corpus of first
    turns proves nothing either way (see the sibling test below).
    """

    reasoning = ({"type": "reasoning", "id": "opaque"},)
    _record(
        tmp_path,
        _OriginalAdapter(
            [
                ModelTurn(final_text="x", reasoning=reasoning),
                ModelTurn(final_text="y", reasoning=reasoning),
            ]
        ),
        [_request(), _request(messages=[{"role": "assistant", "content": "prior turn"}])],
    )
    adapter = _replay(tmp_path)

    assert getattr(adapter, "provider_name", None) is None
    turn = _call(adapter, _request())
    assert turn.reasoning == reasoning


def test_a_corpus_of_first_turns_cannot_testify_and_so_it_declares(tmp_path: Path) -> None:
    """The undecidable case, and the horn the derivation has to pick.

    Every recorded run settled in one turn, so no request could carry an injected block
    whether the original declared or not. Reading that silence as "did not declare" breaks the
    shipped gateway default outright: the gateway declares the RELAYED provider while
    `ModelConfig.provider` names the transport, so the recorded key term is `openai` and a
    non-declaring replay computes `gateway` -- every lookup misses, and the preflight refuses
    a config and a corpus that are both correct.
    """

    _record(
        tmp_path,
        _OriginalAdapter(
            [ModelTurn(final_text="x", reasoning=({"type": "reasoning", "id": "opaque"},))],
            provider_name="openai",
        ),
        [_request(model=ModelConfig(provider="gateway"))],
    )
    adapter = _replay(tmp_path)

    assert getattr(adapter, "provider_name", None) == "openai"
    assert _call(adapter, _request(model=ModelConfig(provider="gateway"))).final_text == "x"


def test_history_without_a_reply_still_cannot_testify(tmp_path: Path) -> None:
    """The witness is an assistant *reply*, not the presence of a message list.

    A first call can carry prior user messages -- a session that queued two before the model
    ever answered. No turn has happened, so no injected block could exist, and the corpus is
    as silent as an empty history. Treating "has messages" as the witness would decline to
    declare here and miss every lookup.
    """

    _record(
        tmp_path,
        _OriginalAdapter(
            [ModelTurn(final_text="x", reasoning=({"type": "reasoning", "id": "opaque"},))],
            provider_name="openai",
        ),
        [
            _request(
                model=ModelConfig(provider="gateway"),
                messages=[
                    {"role": "user", "content": "first"},
                    {"role": "user", "content": "and another"},
                ],
            )
        ],
    )
    adapter = _replay(tmp_path)

    assert getattr(adapter, "provider_name", None) == "openai"


def test_an_explicit_provider_name_overrides_the_derivation(tmp_path: Path) -> None:
    _record(tmp_path, _OriginalAdapter([ModelTurn(final_text="x")]), [_request()])

    declared = _replay(tmp_path, provider_name="custom-name")
    undeclared = _replay(tmp_path, provider_name=None)

    assert declared.provider_name == "custom-name"
    assert getattr(undeclared, "provider_name", None) is None


def test_heterogeneous_provider_sources_are_rejected(tmp_path: Path) -> None:
    _record(
        tmp_path,
        _OriginalAdapter([ModelTurn(final_text="a")], provider_name="openai"),
        [_request()],
        run_id="run-1",
    )
    _record(
        tmp_path,
        _OriginalAdapter([ModelTurn(final_text="b")], provider_name="anthropic"),
        [_request("other")],
        run_id="run-2",
    )

    with pytest.raises(ValueError, match="one provider"):
        _replay(tmp_path, "run-1", "run-2")


def test_multimodal_declaration_follows_the_recorded_messages(tmp_path: Path) -> None:
    media_block = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "aGk="},
    }
    _record(
        tmp_path,
        _OriginalAdapter([ModelTurn(final_text="seen")]),
        [_request(messages=[{"role": "user", "content": [media_block]}])],
        run_id="run-1",
    )
    _record(
        tmp_path,
        _OriginalAdapter([ModelTurn(final_text="plain")]),
        [_request("text only")],
        run_id="run-2",
    )

    assert _replay(tmp_path, "run-1").supports_multimodal is True
    assert _replay(tmp_path, "run-2").supports_multimodal is False


# --- the lookup the adapter serves is the runner's ---------------------------------------------


def test_the_adapter_answers_the_exact_key_the_runner_stamps(tmp_path: Path) -> None:
    """Belt to the recompute==stamp brace in test_request_identity: through the runner, the
    digest the replay adapter consumed is the digest the replay run's own receipt carries."""

    [recorded_digest] = _record(
        tmp_path, _OriginalAdapter([ModelTurn(final_text="x")]), [_request()]
    )
    adapter = _replay(tmp_path)
    seen: list[str] = []
    original_consume = adapter._corpus.consume

    def spying(digest: str, **kwargs: Any) -> Any:
        seen.append(digest)
        return original_consume(digest, **kwargs)

    adapter._corpus.consume = spying  # type: ignore[method-assign]
    _turn, receipt = asyncio.run(ModelCallRunner(adapter=adapter).acall(_request()))

    assert seen == [recorded_digest]
    assert receipt.request_digest == recorded_digest


def test_replay_lookup_agrees_with_the_recorded_corpus_key(tmp_path: Path) -> None:
    [recorded_digest] = _record(
        tmp_path, _OriginalAdapter([ModelTurn(final_text="x")]), [_request()]
    )
    adapter = _replay(tmp_path)

    lookup = replay_lookup(_request(), adapter)

    assert lookup.result.digest == recorded_digest


# --- a refused slot belongs to the call that earned it ---------------------------------------


def test_a_re_attempt_earns_the_same_refusal_not_the_next_call_s_answer(tmp_path: Path) -> None:
    """A replay miss parks the turn, and the loop re-attempts a ``config_recoverable`` failure
    idempotently. So asking twice must refuse twice: a refusal that advanced the sequence
    would hand the re-attempt of call N the answer recorded for call N+1 -- a different call's
    words, arriving as a valid turn, with nothing anywhere saying so.
    """

    [digest] = _record(
        tmp_path,
        _OriginalAdapter([ModelTurn(response_id="r-good", final_text="recovered")]),
        [_request()],
    )
    _prepend_refused_answer(tmp_path, digest)
    adapter = _replay(tmp_path)

    details = []
    for _ in range(3):
        with pytest.raises(ReplayMiss) as caught:
            _call(adapter, _request())
        assert caught.value.provider_error_code == MISS_NOT_RECORDED
        details.append(str(caught.value))

    assert details[0] == details[1] == details[2]
    assert "too_large" in details[0]
    assert "recovered" not in details[0]


def test_a_record_reconstruction_rejects_is_given_back_for_the_re_attempt(tmp_path: Path) -> None:
    """The other route to an unusable slot, and the one that has to be *given back*.

    An ``unrecorded_reason`` is refused before the cursor moves; a body that only
    reconstruction rejects has already been handed over, so the adapter must release it before
    it parks. Otherwise the re-attempt of this call is answered with the next call's recording
    -- the same silent substitution, reached by the other door.
    """

    [digest] = _record(
        tmp_path,
        _OriginalAdapter([ModelTurn(response_id="r-good", final_text="recovered")]),
        [_request()],
    )
    _prepend_refused_answer(
        tmp_path,
        digest,
        unrecorded_reason="",
        body={
            "response_id": "r-bad",
            "final_text": None,
            "tool_calls": [{"id": "c1", "name": "fs_list"}],  # no arguments: not a triple
            "reasoning": [],
            "usage": {},
            "stop_reason": None,
            "provider_retried": False,
        },
    )
    adapter = _replay(tmp_path)

    details = []
    for _ in range(2):
        with pytest.raises(ReplayMiss) as caught:
            _call(adapter, _request())
        assert caught.value.provider_error_code == MISS_NOT_RECORDED
        details.append(str(caught.value))

    assert details[0] == details[1]
    assert "recovered" not in details[0]


def test_serving_a_refused_call_live_moves_the_sequence_past_it(tmp_path: Path) -> None:
    """The other exit from the same refusal. Falling through means this call really was
    answered, so the slot is spent and the next call meets the next recording -- which is the
    alignment ``unrecorded_reason`` records exist to preserve."""

    [digest] = _record(
        tmp_path,
        _OriginalAdapter([ModelTurn(response_id="r-good", final_text="recovered")]),
        [_request()],
    )
    _prepend_refused_answer(tmp_path, digest)
    inner = _CountingInner()
    adapter = _replay(tmp_path, inner=inner)

    first = _call(adapter, _request())
    second = _call(adapter, _request())

    assert (first.final_text, inner.calls) == ("live answer", 1)
    assert (second.final_text, inner.calls) == ("recovered", 1)


def test_a_body_that_is_not_a_recorded_turn_is_a_miss_not_an_empty_turn(tmp_path: Path) -> None:
    """A JSON object is not a recorded answer. Reconstruction used to accept any dict, so a
    corrupt or foreign body became `ModelTurn(final_text=None, tool_calls=())` -- which the
    loop rejects as "neither final text nor tool calls": a `model_error` that kills the run
    instead of parking it, and blames a model that was never called.
    """

    [digest] = _record(
        tmp_path,
        _OriginalAdapter([ModelTurn(response_id="r-good", final_text="recovered")]),
        [_request()],
    )
    for body in ({}, {"surprise": 1}, {"final_text": "half a turn"}):
        _prepend_refused_answer(tmp_path, digest, unrecorded_reason="", body=body)
        with pytest.raises(ReplayMiss) as caught:
            _call(_replay(tmp_path), _request())
        assert caught.value.provider_error_code == MISS_NOT_RECORDED
        assert caught.value.config_recoverable is True
        assert "fields a recorded turn carries" in str(caught.value)


@pytest.mark.parametrize("count", [-1, "ZZ", 1.0, True, {"nested": 1}])
def test_a_recorded_count_the_loop_would_refuse_is_refused_here(tmp_path: Path, count: Any) -> None:
    """Usage was the one reconstructed field checked for its container and not its contents,
    so a corrupt count escaped the miss vocabulary and surfaced three layers down as
    `model_bad_response` -- again a kill, again against the adapter rather than the corpus.
    One predicate now decides portability for the loop and for this reader."""

    [digest] = _record(
        tmp_path,
        _OriginalAdapter([ModelTurn(response_id="r-good", final_text="recovered")]),
        [_request()],
    )
    _prepend_refused_answer(
        tmp_path,
        digest,
        unrecorded_reason="",
        body={
            "response_id": "r-bad",
            "final_text": "text",
            "tool_calls": [],
            "reasoning": [],
            "usage": {"input_tokens": count},
            "stop_reason": None,
            "provider_retried": False,
        },
    )

    with pytest.raises(ReplayMiss) as caught:
        _call(_replay(tmp_path), _request())

    assert caught.value.provider_error_code == MISS_NOT_RECORDED
    assert "input_tokens is not a non-negative integer" in str(caught.value)


def test_a_failed_live_serve_does_not_move_the_sequence(tmp_path: Path) -> None:
    """Falling through spends the refused slot because the call was answered -- so the spend
    has to wait until it was. A live adapter that raises recoverably (a 429 is enough) parks
    the turn for an idempotent re-attempt, and a slot spent on the attempt would answer that
    re-attempt with the next call's recording: the same silent substitution the two-phase
    consume exists to prevent, reached through the other exit.
    """

    [digest] = _record(
        tmp_path,
        _OriginalAdapter([ModelTurn(response_id="r-good", final_text="recovered")]),
        [_request()],
    )
    _prepend_refused_answer(tmp_path, digest)

    class _FailsOnce:
        def __init__(self) -> None:
            self.calls = 0

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            self.calls += 1
            if self.calls == 1:
                raise ModelAdapterError(
                    "upstream is busy", error_code="rate_limited", config_recoverable=True
                )
            return ModelTurn(final_text="live answer")

    inner = _FailsOnce()
    adapter = _replay(tmp_path, inner=inner)

    with pytest.raises(ModelAdapterError) as caught:
        _call(adapter, _request())
    assert caught.value.error_code == "rate_limited"

    served = _call(adapter, _request())

    assert served.final_text == "live answer", (
        "the re-attempt was answered from the corpus, so the failed attempt spent a slot"
    )
    assert inner.calls == 2
    assert _call(adapter, _request()).final_text == "recovered"


def test_a_failed_live_serve_gives_back_a_held_slot_too(tmp_path: Path) -> None:
    """The same rule on the other kind of unusable slot, which is where it was missing.

    Falling through has two shapes of refusal behind it. A ``unrecorded_reason`` is refused
    before the cursor moves, so a failed live serve simply must not *spend* it. A body only
    reconstruction rejects was already handed over -- the cursor is past it -- so a failed live
    serve must *give it back*. Same failure either way: the call did not happen, the
    conversation did not move, and a re-attempt answered from the corpus is the next call's
    recording arriving as this call's answer.
    """

    digest, repeat = _record(
        tmp_path,
        _OriginalAdapter(
            [
                ModelTurn(response_id="r-2", final_text="answer for call two"),
                ModelTurn(response_id="r-3", final_text="answer for call three"),
            ]
        ),
        [_request(), _request()],
    )
    assert digest == repeat, "two identical calls are one key with two recorded answers"
    _prepend_refused_answer(
        tmp_path,
        digest,
        unrecorded_reason="",
        body={
            "response_id": "r-bad",
            "final_text": None,
            "tool_calls": [{"id": "c1", "name": "fs_list"}],  # no arguments: not a triple
            "reasoning": [],
            "usage": {},
            "stop_reason": None,
            "provider_retried": False,
        },
    )

    class _FailsOnce:
        def __init__(self) -> None:
            self.calls = 0

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            self.calls += 1
            if self.calls == 1:
                raise ModelAdapterError(
                    "upstream is busy", error_code="rate_limited", config_recoverable=True
                )
            return ModelTurn(final_text="live answer")

    inner = _FailsOnce()
    adapter = _replay(tmp_path, inner=inner)

    with pytest.raises(ModelAdapterError) as caught:
        _call(adapter, _request())
    assert caught.value.error_code == "rate_limited"

    served = _call(adapter, _request())

    assert served.final_text == "live answer", (
        "the re-attempt was answered from the corpus, so the failed attempt kept a slot it "
        "never served"
    )
    assert inner.calls == 2
    assert _call(adapter, _request()).final_text == "answer for call two"


def test_a_message_that_is_not_an_assistant_reply_is_not_a_turn_behind_it(tmp_path: Path) -> None:
    """The D-h witness, in the direction that costs a whole run.

    "A request that already had a turn behind it" is what makes the corpus able to testify
    about impersonation at all. Widening it from `role == "assistant"` to `role != "user"`
    catches the `role: "tool"` messages the loop writes after every tool call -- and then a
    corpus of first turns whose answers carry reasoning is read as evidence that the original
    declined to declare. The adapter declines too, its key term stops being the relayed
    provider, and every lookup in the run misses: the 100%-miss shape.
    """

    messages = [{"role": "tool", "tool_call_id": "c1", "content": "output"}]
    _record(
        tmp_path,
        _OriginalAdapter(
            [
                ModelTurn(
                    response_id="r1",
                    final_text="answered",
                    reasoning=({"type": "reasoning", "encrypted_content": "OPAQUE"},),
                )
            ],
            provider_name="openai",
        ),
        [_request(_MARKER, messages=messages)],
    )

    adapter = _replay(tmp_path)

    assert adapter.provider_name == "openai", (
        "no request carried an assistant reply, so the corpus is silent on impersonation -- "
        "and silence is not evidence for the negative"
    )
    assert _call(adapter, _request(_MARKER, messages=messages)).final_text == "answered"


def test_a_partial_reasoning_block_is_not_an_injected_one(tmp_path: Path) -> None:
    """The other witness, in its own negative direction.

    `_is_injected_reasoning` describes the block "as the assistant-message append writes it" --
    both keys. Accepting either one alone makes any foreign or third-party corpus with a
    `reasoning` key look like a re-injected trace, and the adapter then declares where the
    evidence says the original did not: the loop injects blocks the recorded preimages never
    had, and every key moves.
    """

    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "a reply", "reasoning": {"provider": "openai"}},
    ]
    _record(
        tmp_path,
        _OriginalAdapter(
            [
                ModelTurn(
                    response_id="r1",
                    final_text="answered",
                    reasoning=({"type": "reasoning", "encrypted_content": "OPAQUE"},),
                )
            ],
            provider_name="openai",
        ),
        [_request(_MARKER, messages=messages)],
    )

    adapter = _replay(tmp_path)

    assert not hasattr(adapter, "provider_name"), (
        "a block missing `items` is not the block the append writes, so the corpus shows an "
        "assistant reply that carried no injected trace -- the original did not declare"
    )


def test_a_media_source_that_is_not_base64_is_not_media(tmp_path: Path) -> None:
    """`supports_multimodal` moves the preimage, so deriving it loosely moves every key.

    The neutral resolved block is `source.type == "base64"`. Counting any mapping named
    `source` as media flips the flag on a text-only corpus whose messages happen to carry a
    by-reference part, and the recorded keys stop matching the ones this run computes.
    """

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "source": {"type": "url", "url": "http://x/y.png"}}],
        }
    ]
    _record(
        tmp_path,
        _OriginalAdapter([ModelTurn(response_id="r1", final_text="answered")]),
        [_request(_MARKER, messages=messages)],
    )

    adapter = _replay(tmp_path)

    assert adapter.supports_multimodal is False


def test_a_retried_call_replays_as_a_retried_call(tmp_path: Path) -> None:
    """`provider_retried` is registered in the carriage conformance census as a field this
    adapter carries, with the note that dropping it "would replay a retried call as a clean
    one, the exact lie W6-2 recorded the field to prevent".

    That registration is a file-level census: it reads which modules mention the name, not what
    any of them does with the value. Hard-coding `provider_retried=False` in the reconstruction
    passes it, and passes every replay suite.
    """

    [digest] = _record(
        tmp_path,
        _OriginalAdapter([ModelTurn(response_id="r-1", final_text="answered after a retry")]),
        [_request()],
    )
    path = tmp_path / "runs" / "run-1" / MODEL_PAYLOADS_FILENAME
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") == "model_response":
            record["response"]["provider_retried"] = True
        out.append(json.dumps(record, sort_keys=True))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    del digest

    turn = _call(_replay(tmp_path), _request())

    assert turn.final_text == "answered after a retry"
    assert turn.provider_retried is True, "the replay reported a retried call as a clean one"


def test_a_refusal_after_the_first_spends_its_own_slot(tmp_path: Path) -> None:
    """The spend coordinate, driven where it can be wrong.

    Every refusal any fixture builds stands at slot 0, and there `spend_refused(digest, slot)`
    and `spend_refused(digest, 0)` agree -- so a `slot` that is simply the constant zero passes
    the whole suite. At any later slot the guard `cursor == slot` is then false, the spend does
    nothing, and the sequence never realigns: every remaining turn of the run is paid for live
    while its recording sits unread one position back.
    """

    digest, repeat = _record(
        tmp_path,
        _OriginalAdapter(
            [
                ModelTurn(response_id="r-1", final_text="recorded first"),
                ModelTurn(response_id="r-3", final_text="recorded third"),
            ]
        ),
        [_request(), _request()],
    )
    assert digest == repeat, "two identical calls are one key with two recorded answers"
    _prepend_refused_answer(tmp_path, digest, after=1)

    inner = _CountingInner()
    adapter = _replay(tmp_path, inner=inner)

    served = [_call(adapter, _request()).final_text for _ in range(3)]

    assert served == ["recorded first", "live answer", "recorded third"], (
        "the refused slot was not spent, so the corpus never realigned"
    )
    assert inner.calls == 1


def test_an_inner_whose_next_turn_is_a_coroutine_function_is_refused(tmp_path: Path) -> None:
    """The constructor's synchronous-inner check asks `callable(next_turn)`, which is true for
    an `async def`.

    `ModelCallRunner` documents a coroutine `next_turn` as one of its four dispatch shapes, and
    `_adrive` awaits whatever this wrapper hands back -- *after* `_serve_miss` has already
    spent the refused slot on a coroutine object that has done no provider work. The awaited
    call then raises, the turn parks, and the re-attempt meets the next recording. No shipped
    inner has this shape (the CLI builds only gateway and OpenAI adapters, both blocking), so
    the door is a library embedder's; the check the constructor already wanted to make is the
    one that closes it.
    """

    _record(
        tmp_path,
        _OriginalAdapter([ModelTurn(response_id="r-1", final_text="recorded")]),
        [_request()],
    )

    class _AsyncInner:
        async def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(final_text="live")

    with pytest.raises(ValueError) as caught:
        _replay(tmp_path, inner=_AsyncInner())

    assert "anext_turn-only" in str(caught.value) or "coroutine" in str(caught.value)


class _AwaitableInner:
    """A *synchronous* ``next_turn`` that hands back an awaitable.

    `_adrive`'s fourth documented shape: "an adapter can be synchronous and still hand back
    something awaitable -- it delegates to an async client". `inspect.iscoroutinefunction` is
    False for a plain `def`, so no declaration-side gate can see this one; only the result can.
    """

    def __init__(self) -> None:
        self.calls = 0

    def next_turn(self, request: ModelRequest) -> Any:
        del request
        self.calls += 1

        async def _later() -> ModelTurn:
            raise RuntimeError("the live call failed after the wrapper had already returned")

        return _later()


def _serve_or_error(adapter: Any, request: ModelRequest) -> str:
    """The answer, or a stand-in naming the failure -- so a sequence can be asserted whole."""

    try:
        text = _call(adapter, request).final_text
    except BaseException as error:  # noqa: BLE001 - the failure is the observation
        return f"<{type(error).__name__}>"
    return text if text is not None else "<no text>"


def test_an_awaitable_from_a_sync_inner_does_not_spend_the_refused_slot(tmp_path: Path) -> None:
    """The spend guard is bound on the blocking inner only, and one inner shape defeats it.

    `_serve_miss` spends the refused slot the moment `next_turn` *returns*. An inner that
    returns an awaitable returns immediately, having done no provider work: the slot is paid
    for, then `_adrive` awaits the call and the call fails. The turn parks for an idempotent
    re-attempt -- and the re-attempt meets the recording that belongs to the call after it,
    exit 0, a valid turn, no surface reporting the substitution.

    `except BaseException` cannot help here: nothing raises inside the try. The settlement has
    to be a fact about the *result*, not about how the call returned.
    """

    digest, repeat = _record(
        tmp_path,
        _OriginalAdapter(
            [
                ModelTurn(response_id="r-1", final_text="recorded first"),
                ModelTurn(response_id="r-3", final_text="recorded third"),
            ]
        ),
        [_request(), _request()],
    )
    assert digest == repeat, "two identical calls are one key with two recorded answers"
    _prepend_refused_answer(tmp_path, digest, after=1)

    inner = _AwaitableInner()
    adapter = _replay(tmp_path, inner=inner)

    served = [_serve_or_error(adapter, _request()) for _ in range(3)]

    assert served[0] == "recorded first"
    assert served[2] != "recorded third", (
        "the refused slot was spent on a call that never happened, so the parked turn's "
        "re-attempt was answered with the next call's recording"
    )


def test_an_awaitable_from_a_sync_inner_gives_back_the_held_slot(tmp_path: Path) -> None:
    """The release half of the same shape, defeated the same way.

    A record `consume` handed over and reconstruction rejected is held, and `_settle_unserved`
    puts it back when the inner raises. An inner that hands back an awaitable never raises
    inside the try, so the held slot is never given back: the cursor stands one past the
    refusal and the re-attempt is served the *following* recording instead of earning the same
    `not_recorded` refusal it must earn every time.
    """

    digest, repeat = _record(
        tmp_path,
        _OriginalAdapter(
            [
                ModelTurn(response_id="r-1", final_text="first"),
                ModelTurn(response_id="r-3", final_text="third"),
            ]
        ),
        [_request(), _request()],
    )
    assert digest == repeat
    _prepend_refused_answer(
        tmp_path,
        digest,
        after=1,
        unrecorded_reason="",
        body={
            "response_id": "r-bad",
            "final_text": None,
            "tool_calls": [{"id": "c1", "name": "fs_list"}],  # no arguments: not a triple
            "reasoning": [],
            "usage": {},
            "stop_reason": None,
            "provider_retried": False,
        },
    )

    inner = _AwaitableInner()
    adapter = _replay(tmp_path, inner=inner)

    served = [_serve_or_error(adapter, _request()) for _ in range(3)]

    assert served[0] == "first"
    assert served[2] != "third", (
        "the held slot was never given back, so the re-attempt was handed the recording "
        "belonging to the call after the one it asked about"
    )


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_the_refused_awaitable_is_disposed_rather_than_left_to_warn(tmp_path: Path) -> None:
    """Refusing an awaitable is not enough; something has to close it.

    A coroutine that is never awaited surfaces at collection as a RuntimeWarning attributed to
    whatever line happened to trigger the collection -- noise pointing at innocent code, on a
    path the operator reaches by misconfiguration rather than by fault. The wrapper made the
    thing and refused it, so the wrapper disposes of it.
    """

    digest, _repeat = _record(
        tmp_path,
        _OriginalAdapter(
            [
                ModelTurn(response_id="r-1", final_text="recorded"),
                ModelTurn(response_id="r-2", final_text="recorded again"),
            ]
        ),
        [_request(), _request()],
    )
    _prepend_refused_answer(tmp_path, digest, after=1)

    adapter = _replay(tmp_path, inner=_AwaitableInner())
    assert _call(adapter, _request()).final_text == "recorded"
    with pytest.raises(ModelAdapterError) as caught:
        _call(adapter, _request())

    assert "awaitable" in str(caught.value)
    assert caught.value.config_recoverable, (
        "the remedy is to build the wrapper around an inner it can drive, so the session "
        "survives to be resumed rather than dying as an unclassified kill"
    )
    gc.collect()


def test_an_inner_whose_lifecycle_is_async_is_refused(tmp_path: Path) -> None:
    """The pairing check is one of three probed callables, and only one of them is gated.

    `open`/`close` are checked for *pairing* and not for async-ness, so an `async def` pair
    passes: `open()` calls the coroutine function and discards the coroutine, the inner is
    never opened, `close()` releases nothing, and the CLI's synchronous probe reports success --
    the exact outcome the pairing check exists to prevent. One predicate over the census of
    probed callables, rather than one hand-picked member.
    """

    _record(
        tmp_path,
        _OriginalAdapter([ModelTurn(response_id="r-1", final_text="recorded")]),
        [_request()],
    )

    class _AsyncLifecycleInner:
        def __init__(self) -> None:
            self.entered = 0

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(final_text="live")

        async def open(self) -> None:
            self.entered += 1

        async def close(self) -> None:
            self.entered -= 1

    with pytest.raises(ValueError) as caught:
        _replay(tmp_path, inner=_AsyncLifecycleInner())

    assert "open" in str(caught.value) or "close" in str(caught.value)


def test_a_slot_after_the_first_is_given_back_too(tmp_path: Path) -> None:
    """The release coordinate, driven where it can be wrong.

    Every other test holds slot 0, and at slot 0 `release(digest, held)` and
    `release(digest, 0)` agree -- so a `held` that is simply the constant zero passes them all
    while, at any later slot, the guard `cursor == slot + 1` is false, nothing is given back,
    and the parked turn's re-attempt is served the call AFTER the one it asked about.
    """

    digest, repeat = _record(
        tmp_path,
        _OriginalAdapter(
            [
                ModelTurn(response_id="r-1", final_text="first"),
                ModelTurn(response_id="r-3", final_text="third"),
            ]
        ),
        [_request(), _request()],
    )
    assert digest == repeat, "two identical calls are one key with two recorded answers"
    _prepend_refused_answer(
        tmp_path,
        digest,
        after=1,
        unrecorded_reason="",
        body={
            "response_id": "r-bad",
            "final_text": None,
            "tool_calls": [{"id": "c1", "name": "fs_list"}],  # no arguments: not a triple
            "reasoning": [],
            "usage": {},
            "stop_reason": None,
            "provider_retried": False,
        },
    )
    # A separate instance for the coordinate, because reaching past `_serve_miss` takes the
    # slot without settling it and would move the sequence this test is about.
    probe = _replay(tmp_path)
    assert _call(probe, _request()).final_text == "first"
    _outcome, held = probe._replayed_turn_or_miss(digest)
    assert held == 1, "the slot the corpus is holding is the one it handed over, not zero"

    adapter = _replay(tmp_path)
    assert _call(adapter, _request()).final_text == "first"
    for _ in range(2):
        with pytest.raises(ReplayMiss) as caught:
            _call(adapter, _request())
        assert caught.value.provider_error_code == MISS_NOT_RECORDED
        assert "third" not in str(caught.value)
