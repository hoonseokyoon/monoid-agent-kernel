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
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from monoid_agent_kernel.core import model_payloads, payload_gc
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
    run_dir: Path, *, generation: str = _GEN, body: dict[str, Any] | None = None
) -> str:
    """A hand-built, self-consistent request+response pair; returns the request digest."""

    preimage_value = {generation: {"instruction": "hand-built", "provider": "gateway"}}
    preimage = model_payloads._encoded(preimage_value)
    digest = sha256_bytes(preimage)
    response = (
        body
        if body is not None
        else {
            "response_id": "r-hand",
            "final_text": "hand answer",
            "tool_calls": [],
            "reasoning": [],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            "stop_reason": "stop",
            "provider_retried": False,
        }
    )
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
            model_response_record(
                response,
                call_index=0,
                request_digest=digest,
                unrecorded_reason="",
                **_envelope(),
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


def test_an_unrecorded_answer_occupies_its_slot(tmp_path: Path) -> None:
    """[P3] A record whose body was refused still spends its turn in the order; skipping it
    would hand answer N+1 to call N and lie about what happened when."""

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
    second = corpus.consume(digest, generation=_GEN)

    assert isinstance(first, ReplayMissReason)
    assert first.reason == MISS_NOT_RECORDED
    assert "unencodable" in first.detail
    assert isinstance(second, ReplayedResponse)
    assert second.body["final_text"] == "recovered"


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


@pytest.mark.parametrize("spelling", ["identical", "dot-suffixed", "absolute"])
def test_one_directory_named_twice_is_one_source(tmp_path: Path, spelling: str) -> None:
    """Each answer once is a property of the *corpus*, not of the argument list.

    A directory reaches the union by more than one name routinely -- as a run id and as a
    path, through a relative and an absolute spelling, through a link. Indexing it twice
    would append every answer to its queue again, so the call that has earned ``exhausted``
    receives a stale recorded body instead, and nothing anywhere says the source was doubled.
    """

    run_dir = tmp_path / "runs" / "run-1"
    digest = _recorded_pair(run_dir)
    again = {
        "identical": run_dir,
        "dot-suffixed": run_dir / ".",
        "absolute": run_dir.resolve(),
    }[spelling]

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
