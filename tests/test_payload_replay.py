"""The corpus reader: file-order consumption, typed misses, and refusals that stay refusals.

W6-4b B2. ``tests/test_model_payloads.py`` pins what a run writes; this pins what a replay can
truthfully take back out. The reader's conclusions substitute for paid provider calls, so every
rule the writer holds by construction is re-established here on arrival: lines through the same
lenient verified-descriptor reader the collector uses, references through the same trichotomy
the validator reports through, chunk bytes re-hashed before they are believed.

Two vocabularies are pinned here. The miss taxonomy is exactly six (D-i) -- a reason the
adapter cannot name precisely lands on a neighbour and misdirects the operator's next step.
And diagnosis speaks config and structure only: identity terms are named with expected/actual
values (the ledger already records them in plaintext), prompt terms are named by term name and
digest, never by content.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from monoid_agent_kernel.core import model_payloads, payload_gc, payload_replay
from monoid_agent_kernel.core._util import sha256_bytes
from monoid_agent_kernel.core.model_io import ModelCallReceipt
from monoid_agent_kernel.core.model_payloads import (
    MODEL_PAYLOADS_DIRNAME,
    MODEL_PAYLOADS_FILENAME,
    MODEL_REQUEST_KIND,
    PAYLOAD_CHUNK_REF_KEY,
    PAYLOAD_OFFLOAD_THRESHOLD_BYTES,
    chunk_record,
    model_request_record,
    model_response_record,
    read_corpus_records,
)
from monoid_agent_kernel.core.payload_replay import (
    MISS_ABSENT,
    MISS_EXHAUSTED,
    MISS_GENERATION_MISMATCH,
    MISS_IDENTITY_MISMATCH,
    MISS_NO_KEY,
    MISS_NOT_RECORDED,
    REPLAY_MISS_REASONS,
    ReplayCorpus,
    ReplayMissReason,
    ReplayedResponse,
)
from monoid_agent_kernel.core.schemas import validate_run_dir
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.model_call import ModelCallRunner, SettledModelCall
from monoid_agent_kernel.providers._request_identity import (
    _REQUEST_DIGEST_GENERATION,
    _request_payload,
    replay_lookup,
)
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn, normalize_model_request
from monoid_agent_kernel.recorder import AgentRecorder

_GEN = _REQUEST_DIGEST_GENERATION


def _recorder(base: Path, *, run_id: str = "run-1", reopen: bool = False) -> AgentRecorder:
    return AgentRecorder(
        base / "runs",
        run_id,
        status_file=False,
        model_calls_file=True,
        model_payload_file=True,
        reopen=reopen,
    )


class _ScriptedAdapter:
    """Answers from a list; raising entries raise."""

    def __init__(self, turns: list[Any]):
        self.turns = turns
        self.requests: list[ModelRequest] = []

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        answer = self.turns.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _drive(recorder: AgentRecorder, adapter: Any, requests: list[ModelRequest]) -> list[str]:
    """Run real calls through the real runner into the real recorder; return request digests."""

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
    return digests


def _request(text: str = "hi", **changes: Any) -> ModelRequest:
    return ModelRequest(instruction=text, system_prompt="sys", tools=(), **changes)


def _load(base: Path, *run_ids: str) -> ReplayCorpus:
    return ReplayCorpus.load([base / "runs" / run_id for run_id in (run_ids or ("run-1",))])


def _write_corpus(run_dir: Path, records: list[dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records]
    (run_dir / MODEL_PAYLOADS_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _envelope(run_id: str = "run-1") -> dict[str, str]:
    return {"run_id": run_id, "root_run_id": run_id, "recorded_at": "2026-08-08T00:00:00Z"}


def _recorded_pair(
    run_dir: Path,
    *,
    generation: str = _GEN,
    body: dict[str, Any] | None = None,
    answers: list[str] | None = None,
) -> str:
    """A hand-built, self-consistent request + N answers; returns the request digest.

    ``answers`` names the final texts when a test needs more than one recording under the key
    -- the only shape in which a slot coordinate is distinguishable from the constant zero.
    """

    preimage_value = {generation: {"instruction": "hand-built", "provider": "gateway"}}
    preimage = model_payloads._encoded(preimage_value)
    digest = sha256_bytes(preimage)

    def _body(final_text: str, response_id: str) -> dict[str, Any]:
        return {
            "response_id": response_id,
            "final_text": final_text,
            "tool_calls": [],
            "reasoning": [],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            "stop_reason": "stop",
            "provider_retried": False,
        }

    if body is not None:
        bodies = [body]
    elif answers is not None:
        bodies = [_body(text, f"r-{index}") for index, text in enumerate(answers)]
    else:
        bodies = [_body("hand answer", "r-hand")]
    _write_corpus(
        run_dir,
        [
            model_request_record(
                preimage_value,
                refs=False,
                request_digest=digest,
                digest_generation=generation,
                **_envelope(),
            ),
            *(
                model_response_record(
                    response,
                    call_index=index,
                    request_digest=digest,
                    unrecorded_reason="",
                    **_envelope(),
                )
                for index, response in enumerate(bodies)
            ),
        ],
    )
    return digest


# --- one reader, shared ------------------------------------------------------------------


def test_the_reader_and_gc_read_lines_through_one_function() -> None:
    """The collector's lenient reader and the replay corpus's are one function, by identity.

    A mirror here is the twin-drift shape this repo keeps re-earning: two line loops that agree
    today and diverge the first time one grows a rule.
    """

    assert payload_gc._corpus_records is model_payloads.read_corpus_records
    assert ReplayCorpus._read is read_corpus_records


# --- hits: verbatim, in file order, each once ---------------------------------------------


def test_a_recorded_call_replays_its_answer_verbatim(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    turn = ModelTurn(
        response_id="r1",
        final_text="the answer",
        usage={"input_tokens": 3, "output_tokens": 5},
        reasoning=({"type": "reasoning", "encrypted_content": "OPAQUE"},),
        stop_reason="stop",
    )
    [digest] = _drive(recorder, _ScriptedAdapter([turn]), [_request()])
    recorder.close()

    corpus = _load(tmp_path)
    hit = corpus.consume(digest, generation=_GEN)

    assert isinstance(hit, ReplayedResponse)
    assert hit.body["response_id"] == "r1"
    assert hit.body["final_text"] == "the answer"
    assert hit.body["reasoning"] == [{"type": "reasoning", "encrypted_content": "OPAQUE"}]
    assert hit.body["stop_reason"] == "stop"
    assert hit.call_index == 0
    assert hit.run_id == "run-1"

    second = corpus.consume(digest, generation=_GEN)
    assert isinstance(second, ReplayMissReason)
    assert second.reason == MISS_EXHAUSTED


def test_answers_replay_in_file_order_each_once(tmp_path: Path) -> None:
    """Sequence, not set: models are not functions, and the corpus records what happened."""

    recorder = _recorder(tmp_path)
    adapter = _ScriptedAdapter(
        [
            ModelTurn(response_id="a", final_text="first"),
            ModelTurn(response_id="b", final_text="second"),
        ]
    )
    digests = _drive(recorder, adapter, [_request(), _request()])
    recorder.close()
    assert digests[0] == digests[1], "one request twice must carry one key"

    corpus = _load(tmp_path)
    first = corpus.consume(digests[0], generation=_GEN)
    second = corpus.consume(digests[0], generation=_GEN)
    third = corpus.consume(digests[0], generation=_GEN)

    assert isinstance(first, ReplayedResponse) and first.body["final_text"] == "first"
    assert isinstance(second, ReplayedResponse) and second.body["final_text"] == "second"
    assert isinstance(third, ReplayMissReason) and third.reason == MISS_EXHAUSTED
    assert "2" in third.detail


def test_a_refused_answer_keeps_its_slot_until_the_caller_moves_past_it(tmp_path: Path) -> None:
    """[P3, refined in round-1 review] A record whose body was refused still owns its turn in
    the order -- but the corpus does not spend it on the refusal itself.

    Both halves are the same rule seen from the two exits a miss has. Asking again without
    having served the call must earn the *same* refusal: a replay miss parks the turn, and
    the loop's contract for a ``config_recoverable`` failure is an idempotent re-attempt, so
    an advancing refusal would answer the re-attempt with the next call's recording. Once the
    caller says the conversation has moved past this call -- ``spend_refused``, which is what
    serving it live means -- answer N+1 belongs to call N+1 again.
    """

    recorder = _recorder(tmp_path)
    digest = sha256_bytes(b"slot")
    hostile = SimpleNamespace(
        response_id="r-bad",
        final_text=None,
        tool_calls=(SimpleNamespace(id=1, name="x", arguments={}),),  # id not a str
        usage={},
        reasoning=(),
        stop_reason=None,
        provider_retried=False,
    )
    for turn in (hostile, ModelTurn(response_id="r-good", final_text="recovered")):
        recorder.record_settled_call(
            SettledModelCall(
                receipt=ModelCallReceipt(request_digest=digest, digest_generation=_GEN),
                turn=turn,
            )
        )
    recorder.close()

    corpus = _load(tmp_path)
    first = corpus.consume(digest, generation=_GEN)
    again = corpus.consume(digest, generation=_GEN)

    assert isinstance(first, ReplayMissReason)
    assert first.reason == MISS_NOT_RECORDED
    assert "unencodable" in first.detail
    assert isinstance(again, ReplayMissReason)
    assert (again.reason, again.detail) == (first.reason, first.detail)
    assert first.slot == again.slot == 0, "a refusal names the position it stands on"

    corpus.spend_refused(digest, first.slot)
    after = corpus.consume(digest, generation=_GEN)

    assert isinstance(after, ReplayedResponse)
    assert after.body["final_text"] == "recovered"
    assert after.slot == 1


def test_a_released_answer_is_handed_out_again_and_only_that_one(tmp_path: Path) -> None:
    """``release`` is the other half of the same accounting: a slot handed over and then found
    unusable goes back, so the re-attempt meets it -- but only while nothing else has moved,
    because rewinding past a concurrent taker would hand one answer to two calls."""

    run_dir = tmp_path / "runs" / "run-1"
    digest = _recorded_pair(run_dir, answers=["first", "second"])
    corpus = ReplayCorpus.load([run_dir])

    first = corpus.consume(digest, generation=_GEN)
    second = corpus.consume(digest, generation=_GEN)
    assert isinstance(first, ReplayedResponse) and isinstance(second, ReplayedResponse)
    assert (first.slot, second.slot) == (0, 1)

    corpus.release(digest, second.slot)
    again = corpus.consume(digest, generation=_GEN)

    # Two answers, so "give back that slot" and "rewind to the start" are different acts. On a
    # one-answer queue they are the same, which is why this test used to pass against a corpus
    # that simply reset the cursor to zero.
    assert isinstance(again, ReplayedResponse)
    assert (again.slot, again.body["final_text"]) == (1, "second")

    # And the concurrency clause: once another caller has moved, an older slot is not ours to
    # give back -- rewinding then would hand one answer to two calls.
    third = corpus.consume(digest, generation=_GEN)
    corpus.release(digest, first.slot)
    assert isinstance(third, ReplayMissReason)
    assert isinstance(corpus.consume(digest, generation=_GEN), ReplayMissReason)


def test_a_failed_call_leaves_a_request_with_no_answer(tmp_path: Path) -> None:
    """[P6] The original call failed: its key and preimage were recorded, its answer was not
    (there was none). Replay finds the request known and the answer absent -- with a detail
    that says which of the two situations this is."""

    recorder = _recorder(tmp_path)
    [digest] = _drive(
        recorder,
        _ScriptedAdapter([ModelAdapterError("upstream refused", retryable=True)]),
        [_request()],
    )
    recorder.close()

    records = read_corpus_records(tmp_path / "runs" / "run-1" / MODEL_PAYLOADS_FILENAME)[1]
    assert any(r.get("kind") == MODEL_REQUEST_KIND for r in records)

    miss = _load(tmp_path).consume(digest, generation=_GEN)
    assert isinstance(miss, ReplayMissReason)
    assert miss.reason == MISS_ABSENT
    assert "no answer" in miss.detail


# --- the resumed-run shape [P5] ------------------------------------------------------------


def test_a_resumed_run_reads_as_one_sequence(tmp_path: Path) -> None:
    """call_index restarts per activation and the seen-set is activation-local, so a durable
    resume writes duplicate request records and a second index-0 -- the reader's file-order
    rule is what makes that corpus mean one run."""

    first = _recorder(tmp_path)
    [digest] = _drive(
        first, _ScriptedAdapter([ModelTurn(response_id="a", final_text="before")]), [_request()]
    )
    first.close()
    resumed = _recorder(tmp_path, reopen=True)
    [again] = _drive(
        resumed, _ScriptedAdapter([ModelTurn(response_id="b", final_text="after")]), [_request()]
    )
    resumed.close()
    assert digest == again

    records = read_corpus_records(tmp_path / "runs" / "run-1" / MODEL_PAYLOADS_FILENAME)[1]
    request_records = [r for r in records if r.get("kind") == MODEL_REQUEST_KIND]
    assert len(request_records) == 2, "the duplicate is the fixture's point"

    corpus = _load(tmp_path)
    hits = [corpus.consume(digest, generation=_GEN) for _ in range(2)]
    assert [h.body["final_text"] for h in hits] == ["before", "after"]
    assert [h.call_index for h in hits] == [0, 0]


def test_duplicate_request_records_keep_the_first(tmp_path: Path) -> None:
    """First-wins is observable through the identity profiles a preflight compares against."""

    run_dir = tmp_path / "runs" / "run-1"
    value_a = {_GEN: {"instruction": "a", "provider": "gateway", "model": {"model": "m-first"}}}
    value_b = {_GEN: {"instruction": "a", "provider": "gateway", "model": {"model": "m-second"}}}
    preimage = model_payloads._encoded(value_a)
    digest = sha256_bytes(preimage)
    _write_corpus(
        run_dir,
        [
            model_request_record(
                value_a, refs=False, request_digest=digest, digest_generation=_GEN, **_envelope()
            ),
            model_request_record(
                value_b, refs=False, request_digest=digest, digest_generation=_GEN, **_envelope()
            ),
        ],
    )

    profiles = ReplayCorpus.load([run_dir]).identity_profiles()

    assert [profile["model"] for profile in profiles] == [{"model": "m-first"}]


# --- typed misses ---------------------------------------------------------------------------


def test_the_vocabulary_is_exactly_six() -> None:
    """[D-i] approved vocabulary; a seventh member or a lost one is a contract change."""

    assert set(REPLAY_MISS_REASONS) == {
        MISS_NO_KEY,
        MISS_ABSENT,
        MISS_NOT_RECORDED,
        MISS_IDENTITY_MISMATCH,
        MISS_EXHAUSTED,
        MISS_GENERATION_MISMATCH,
    }
    assert len(REPLAY_MISS_REASONS) == 6


def test_generation_mismatch_names_both_generations(tmp_path: Path) -> None:
    """The whole-corpus miss with one cause gets one message naming recorded and computing
    generations -- the killer of the silent 100%-miss run."""

    other = "monoid.model-request-digest.v0"
    run_dir = tmp_path / "runs" / "run-1"
    _recorded_pair(run_dir, generation=other)

    live_digest = sha256_bytes(b"never-recorded")
    miss = ReplayCorpus.load([run_dir]).consume(live_digest, generation=_GEN)

    assert isinstance(miss, ReplayMissReason)
    assert miss.reason == MISS_GENERATION_MISMATCH
    assert other in miss.detail and _GEN in miss.detail


def test_diagnosis_refines_identity_and_names_config_not_content(tmp_path: Path) -> None:
    """Identity terms are compared by value (the ledger already records them in plaintext);
    the conversation never appears."""

    marker = "SECRET-CONTENT-9Q"
    recorder = _recorder(tmp_path)
    _drive(recorder, _ScriptedAdapter([ModelTurn(final_text="ok")]), [_request(marker)])
    recorder.close()

    live = _request_payload(
        normalize_model_request(_request(marker)),
        ModelConfig(model="elsewhere-9"),
        provider="gateway",
    )
    corpus = _load(tmp_path)
    miss = corpus.diagnose(live, generation=_GEN)

    assert miss.reason == MISS_IDENTITY_MISMATCH
    assert "elsewhere-9" in miss.detail and "gpt-5.5" in miss.detail
    assert marker not in miss.detail


def test_a_prompt_divergence_stays_absent_and_names_the_term(tmp_path: Path) -> None:
    """Same identity, different conversation (the nondeterministic-tool shape): the reason
    stays absent and the diagnosis names the diverging terms by name and digest only."""

    marker = "SECRET-OBS-77"
    recorder = _recorder(tmp_path)
    adapter = _ScriptedAdapter([ModelTurn(final_text="ok")])
    _drive(recorder, adapter, [_request("stable instruction")])
    recorder.close()

    lookup = replay_lookup(
        normalize_model_request(
            _request("stable instruction", messages=[{"role": "user", "content": marker}])
        ),
        adapter,
    )
    corpus = _load(tmp_path)
    miss = corpus.diagnose(lookup.payload, generation=_GEN)

    assert miss.reason == MISS_ABSENT
    assert "messages" in miss.detail
    assert marker not in miss.detail


# --- refusals stay refusals ------------------------------------------------------------------


def test_the_trichotomy_names_the_third_arm() -> None:
    """[P7] The classification itself, pinned directly. The defense-in-depth sha checks in
    both resolvers would still refuse these inputs if this arm vanished (measured: mutating
    it away stays green against downstream substrings alone), so the arm's falsifiable value
    is the *classification* -- corruption told apart from a resolution failure -- and that is
    what this test binds."""

    from monoid_agent_kernel.core.model_payloads import (
        RESPONSE_INLINE,
        RESPONSE_MALFORMED,
        RESPONSE_REFERENCE,
        response_reference,
    )

    sha = "0" * 64
    assert response_reference({PAYLOAD_CHUNK_REF_KEY: sha}) == (RESPONSE_REFERENCE, sha)
    assert response_reference({PAYLOAD_CHUNK_REF_KEY: "../../escape"}) == (
        RESPONSE_MALFORMED,
        None,
    )
    assert response_reference({PAYLOAD_CHUNK_REF_KEY: 123}) == (RESPONSE_MALFORMED, None)
    assert response_reference(None) == (RESPONSE_INLINE, None)
    assert response_reference({"final_text": "x"}) == (RESPONSE_INLINE, None)
    assert response_reference({PAYLOAD_CHUNK_REF_KEY: sha, "extra": 1}) == (
        RESPONSE_INLINE,
        None,
    )


def test_a_malformed_reference_is_refused_by_reader_and_validator_alike(tmp_path: Path) -> None:
    """[P7] One trichotomy, both consumers -- and each consumer must say WHICH arm spoke.

    The assertions here are arm-exact on purpose: a malformed reference reported in the
    resolution-failure vocabulary means the front door was bypassed and a deeper guard
    caught it, which is the redundant-enforcement shape that made a first draft of this
    test survive the front door's deletion."""

    run_dir = tmp_path / "runs" / "run-1"
    digest = _recorded_pair(run_dir)
    with (run_dir / MODEL_PAYLOADS_FILENAME).open("a", encoding="utf-8") as handle:
        for bad in ("../../escape", 123):
            handle.write(
                json.dumps(
                    model_response_record(
                        {PAYLOAD_CHUNK_REF_KEY: bad},
                        call_index=1,
                        request_digest=digest,
                        unrecorded_reason="",
                        **_envelope(),
                    ),
                    sort_keys=True,
                )
                + "\n"
            )

    corpus = ReplayCorpus.load([run_dir])
    assert isinstance(corpus.consume(digest, generation=_GEN), ReplayedResponse)
    for _ in range(2):
        miss = corpus.consume(digest, generation=_GEN)
        assert isinstance(miss, ReplayMissReason)
        assert miss.reason == MISS_NOT_RECORDED
        assert miss.detail.startswith("a response reference is not a content-addressed name")

    issues = [issue for issue in validate_run_dir(run_dir) if "reference" in issue.message]
    assert [issue.message for issue in issues] == [
        "response reference is not a content-addressed name",
        "response reference is not a content-addressed name",
    ]


def test_an_offloaded_answer_resolves_verified_or_not_at_all(tmp_path: Path) -> None:
    big = "x" * (PAYLOAD_OFFLOAD_THRESHOLD_BYTES + 4096)
    recorder = _recorder(tmp_path)
    [digest] = _drive(
        recorder, _ScriptedAdapter([ModelTurn(response_id="r", final_text=big)]), [_request()]
    )
    recorder.close()
    run_dir = tmp_path / "runs" / "run-1"

    hit = _load(tmp_path).consume(digest, generation=_GEN)
    assert isinstance(hit, ReplayedResponse)
    assert hit.body["final_text"] == big

    stored = next((run_dir / MODEL_PAYLOADS_DIRNAME).iterdir())
    stored.write_bytes(b"not the recorded bytes")
    tampered = _load(tmp_path).consume(digest, generation=_GEN)
    assert isinstance(tampered, ReplayMissReason)
    assert tampered.reason == MISS_NOT_RECORDED

    stored.unlink()
    missing = _load(tmp_path).consume(digest, generation=_GEN)
    assert isinstance(missing, ReplayMissReason)
    assert missing.reason == MISS_NOT_RECORDED


def test_an_inline_chunk_that_lies_about_its_sha_is_not_believed(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    body_bytes = model_payloads._encoded({"response_id": "r", "final_text": "real"})
    sha = sha256_bytes(body_bytes)
    chunk = chunk_record(body_bytes, **_envelope())
    chunk["text"] = json.dumps({"response_id": "r", "final_text": "forged"})
    digest = sha256_bytes(b"inline-lie")
    _write_corpus(
        run_dir,
        [
            chunk,
            model_response_record(
                {PAYLOAD_CHUNK_REF_KEY: sha},
                call_index=0,
                request_digest=digest,
                unrecorded_reason="",
                **_envelope(),
            ),
        ],
    )

    miss = ReplayCorpus.load([run_dir]).consume(digest, generation=_GEN)

    assert isinstance(miss, ReplayMissReason)
    assert miss.reason == MISS_NOT_RECORDED


def test_load_refuses_a_directory_without_a_corpus(tmp_path: Path) -> None:
    """Construction-time, not mid-run: a replay source that recorded nothing is an operator
    mistake, and the tenth turn is the wrong place to hear about it."""

    empty = tmp_path / "runs" / "not-recorded"
    empty.mkdir(parents=True)

    with pytest.raises(ValueError, match="not-recorded"):
        ReplayCorpus.load([empty])


def test_damaged_lines_are_counted_not_fatal(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    digest = _recorded_pair(run_dir)
    with (run_dir / MODEL_PAYLOADS_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")

    corpus = ReplayCorpus.load([run_dir])

    assert corpus.damaged_lines == 1
    assert isinstance(corpus.consume(digest, generation=_GEN), ReplayedResponse)


# --- the family union ------------------------------------------------------------------------


def test_a_union_replays_across_run_directories_in_argument_order(tmp_path: Path) -> None:
    for run_id, answer in (("run-1", "parent"), ("run-2", "child")):
        recorder = _recorder(tmp_path, run_id=run_id)
        [digest] = _drive(
            recorder,
            _ScriptedAdapter([ModelTurn(response_id=run_id, final_text=answer)]),
            [_request()],
        )
        recorder.close()

    corpus = _load(tmp_path, "run-1", "run-2")
    first = corpus.consume(digest, generation=_GEN)
    second = corpus.consume(digest, generation=_GEN)

    assert isinstance(first, ReplayedResponse)
    assert (first.body["final_text"], first.run_id) == ("parent", "run-1")
    assert isinstance(second, ReplayedResponse)
    assert (second.body["final_text"], second.run_id) == ("child", "run-2")
    assert set(corpus.run_ids()) == {"run-1", "run-2"}


@pytest.mark.parametrize("spelling", ["identical", "via-parent", "relative"])
def test_one_directory_named_twice_is_one_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str
) -> None:
    """Each answer once is a property of the *corpus*, not of the argument list.

    A directory reaches the union by more than one name routinely -- as a run id and as a
    path, through a relative and an absolute spelling, through a link. Indexing it twice
    would append every answer to its queue again, so the call that has earned ``exhausted``
    receives a stale recorded body instead, and nothing anywhere says the source was doubled.
    """

    run_dir = tmp_path / "runs" / "run-1"
    digest = _recorded_pair(run_dir)
    if spelling == "relative":
        monkeypatch.chdir(tmp_path)
    again = {
        "identical": run_dir,
        "via-parent": run_dir.parent / ".." / "runs" / run_dir.name,
        "relative": Path("runs") / "run-1",
    }[spelling]
    if spelling != "identical":
        # `pathlib` collapses a lone "." and `tmp_path` is already resolved, so the obvious
        # spellings are the SAME STRING and a string-keyed dedupe would pass them. These two
        # differ as text and name one file, which is the only version of this test that can
        # tell the two implementations apart.
        assert str(again) != str(run_dir)

    corpus = ReplayCorpus.load([run_dir, again])

    assert corpus.response_count() == 1
    assert corpus.repeated_sources == 1
    assert isinstance(corpus.consume(digest, generation=_GEN), ReplayedResponse)
    exhausted = corpus.consume(digest, generation=_GEN)
    assert isinstance(exhausted, ReplayMissReason)
    assert exhausted.reason == MISS_EXHAUSTED


def test_two_directories_holding_the_same_bytes_are_two_sources(tmp_path: Path) -> None:
    """The rule is file identity, not file content: two runs that happen to record the same
    answer are two answers, and a union of them must serve both."""

    first = tmp_path / "runs" / "run-1"
    second = tmp_path / "runs" / "run-2"
    digest = _recorded_pair(first)
    assert _recorded_pair(second) == digest

    corpus = ReplayCorpus.load([first, second])

    assert corpus.response_count() == 2
    assert corpus.repeated_sources == 0
    assert isinstance(corpus.consume(digest, generation=_GEN), ReplayedResponse)
    assert isinstance(corpus.consume(digest, generation=_GEN), ReplayedResponse)


# --- what the reader refuses to read at all ---------------------------------------------------


def test_a_record_from_another_schema_version_is_not_served(tmp_path: Path) -> None:
    """The validator enforces the version on these bytes; a reader that did not would serve a
    corpus the kernel's own `monoid validate` calls corrupt -- and, after a bump, serve v2
    answers under v1 field semantics. A version is a promise about what the other fields mean.
    """

    run_dir = tmp_path / "runs" / "run-1"
    digest = _recorded_pair(run_dir)
    path = run_dir / MODEL_PAYLOADS_FILENAME
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") == "model_response":
            record["schema_version"] = "monoid.model-payloads.v2"
        lines.append(json.dumps(record, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    corpus = ReplayCorpus.load([run_dir])

    assert corpus.rejected_records == 1
    assert corpus.response_count() == 0
    miss = corpus.consume(digest, generation=_GEN)
    assert isinstance(miss, ReplayMissReason)
    assert miss.reason == MISS_ABSENT


def test_a_retired_generation_is_nameable_before_the_run_starts(tmp_path: Path) -> None:
    """The same sentence the miss diagnosis gives, available to the preflight.

    A corpus retired by a generation bump can match nothing at all, so "before the run starts"
    is where an operator should hear it -- not at turn one, after a run directory and a
    checkpoint already exist, and not at all under `--replay-fallthrough`, where the whole run
    would otherwise go live and billed in silence.
    """

    run_dir = tmp_path / "runs" / "run-1"
    _recorded_pair(run_dir, generation="monoid.model-request-digest.v0")
    corpus = ReplayCorpus.load([run_dir])

    divergence = corpus.generation_divergence(_GEN)

    assert divergence is not None
    assert "monoid.model-request-digest.v0" in divergence
    assert _GEN in divergence
    assert corpus.generation_divergence("monoid.model-request-digest.v0") is None


def test_concurrent_refusals_of_one_slot_spend_it_once(tmp_path: Path) -> None:
    """``consume`` deliberately does not advance on a refusal, so every concurrent caller
    meets the SAME refused entry. A blind increment would then spend one slot per caller and
    skip recorded answers no caller ever sees -- which is worse than the ordering
    nondeterminism the module docstring does disclaim, because those answers are simply lost.

    Naming the slot makes duplicate refusals idempotent, which is what they are: one slot,
    refused once, however many callers heard about it.
    """

    run_dir = tmp_path / "runs" / "run-1"
    digest = _recorded_pair(run_dir)
    path = run_dir / MODEL_PAYLOADS_FILENAME
    answered = path.read_text(encoding="utf-8").splitlines()
    refused = json.dumps(
        model_response_record(
            None,
            call_index=0,
            request_digest=digest,
            unrecorded_reason="too_large",
            **_envelope(),
        ),
        sort_keys=True,
    )
    path.write_text("\n".join([answered[0], refused, answered[1]]) + "\n", encoding="utf-8")
    corpus = ReplayCorpus.load([run_dir])

    barrier = threading.Barrier(4)
    misses: list[Any] = []

    def refuse() -> None:
        barrier.wait()
        misses.append(corpus.consume(digest, generation=_GEN))

    threads = [threading.Thread(target=refuse) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(isinstance(miss, ReplayMissReason) for miss in misses)
    assert {miss.slot for miss in misses} == {0}
    for miss in misses:
        corpus.spend_refused(digest, miss.slot)

    served = corpus.consume(digest, generation=_GEN)
    assert isinstance(served, ReplayedResponse), "four refusals of one slot spent four slots"
    assert served.body["final_text"] == "hand answer"


def test_a_keyless_answer_is_unjoinable_not_damage(tmp_path: Path) -> None:
    """A `model_response` with an empty `request_digest` is deliberate and legal -- a keyless
    call still records its answer. The reader cannot index it (nothing can ask for it by key),
    but declining to index a healthy record is not damage, and calling it damage told the
    operator their corpus was broken when `monoid validate` says it is clean.
    """

    run_dir = tmp_path / "runs" / "run-1"
    digest = _recorded_pair(run_dir)
    with (run_dir / MODEL_PAYLOADS_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                model_response_record(
                    {
                        "response_id": "r-keyless",
                        "final_text": "answered without a key",
                        "tool_calls": [],
                        "reasoning": [],
                        "usage": {},
                        "stop_reason": "stop",
                        "provider_retried": False,
                    },
                    call_index=1,
                    request_digest="",
                    unrecorded_reason="",
                    **_envelope(),
                ),
                sort_keys=True,
            )
            + "\n"
        )

    corpus = ReplayCorpus.load([run_dir])

    assert corpus.unjoinable_records == 1
    assert corpus.rejected_records == 0, "a legal record the reader cannot key is not damage"
    assert corpus.damaged_lines == 0
    assert isinstance(corpus.consume(digest, generation=_GEN), ReplayedResponse)


def test_a_volume_without_inodes_does_not_collapse_the_union(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`payload_gc` states this repo's rule -- an inode number is evidence only where the
    platform supplies one -- and enforces it. The dedupe is the second consumer of
    `file_identity`, and without the same gate two DISTINCT corpora that both report `(0, 0)`
    compare equal, so every source after the first is discarded as a repeat. On the family
    union, which `docs/CLI.md` documents as the required shape for a spawning run, that drops
    the child's answers and blames the operator for a spelling mistake they did not make.
    """

    first = tmp_path / "runs" / "run-1"
    second = tmp_path / "runs" / "run-2"
    digest = _recorded_pair(first)
    assert _recorded_pair(second) == digest
    monkeypatch.setattr(
        payload_replay,
        "file_identity",
        lambda metadata: payload_replay.VerifiedFileIdentity(device=0, inode=0),
    )

    corpus = ReplayCorpus.load([first, second])

    assert corpus.repeated_sources == 0
    assert corpus.response_count() == 2, "an unnameable source is indexed, not discarded"
    assert isinstance(corpus.consume(digest, generation=_GEN), ReplayedResponse)
    assert isinstance(corpus.consume(digest, generation=_GEN), ReplayedResponse)
