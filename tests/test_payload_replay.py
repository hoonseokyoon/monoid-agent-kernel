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

import ast
import asyncio
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from monoid_agent_kernel.core import model_payloads, payload_gc, payload_replay, schemas
from monoid_agent_kernel.core._util import sha256_bytes
from monoid_agent_kernel.core.model_io import ModelCallReceipt
from monoid_agent_kernel.core.model_payloads import (
    MODEL_PAYLOADS_DIRNAME,
    MODEL_RESPONSE_KIND,
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
from monoid_agent_kernel.providers.replay import ReplayModelAdapter
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.model_call import ModelCallRunner, SettledModelCall
from monoid_agent_kernel.providers._request_identity import (
    _REQUEST_DIGEST_GENERATION,
    _request_payload,
    replay_lookup,
)
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelTurn,
    ToolCall,
    normalize_model_request,
)
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


def _envelope(run_id: str = "run-1", root_run_id: str | None = None) -> dict[str, str]:
    return {
        "run_id": run_id,
        "root_run_id": root_run_id or run_id,
        "recorded_at": "2026-08-08T00:00:00Z",
    }


def _recorded_pair(
    run_dir: Path,
    *,
    generation: str = _GEN,
    body: dict[str, Any] | None = None,
    answers: list[str] | None = None,
    run_id: str = "run-1",
    root_run_id: Any = None,
    terms: dict[str, Any] | None = None,
) -> str:
    """A hand-built, self-consistent request + N answers; returns the request digest.

    ``answers`` names the final texts when a test needs more than one recording under the key
    -- the only shape in which a slot coordinate is distinguishable from the constant zero.
    """

    preimage_value = {
        generation: terms
        if terms is not None
        else {"instruction": "hand-built", "provider": "gateway"}
    }
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
                **_envelope(run_id, root_run_id),
            ),
            *(
                model_response_record(
                    response,
                    call_index=index,
                    request_digest=digest,
                    unrecorded_reason="",
                    **_envelope(run_id, root_run_id),
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


def test_a_miss_whose_diverging_term_is_the_identity_says_identity_mismatch(
    tmp_path: Path,
) -> None:
    """`identity_divergence` answers "some recorded identity matches", which is not the same
    question as "the record this call would have used was recorded under this identity".

    In a union of two identities the run's config reaches one of them, so the preflight passes
    with a warning and the whole-corpus check finds nothing -- and then a call recorded under
    the *other* one misses. Today that lands in the term-by-term branch, which prefixes its
    sentence with "identity matches" and then names `model` as the diverging term, in digest
    form, while `provider_error_code` reads `absent`. Every one of those is the wrong answer:
    an automation choosing between "fix the config" and "the conversation diverged" is told the
    second, and an operator is told the identity is fine by the same sentence that says it is
    not.
    """

    run_dir = tmp_path / "runs" / "run-1"
    # This run's config reaches `m-here`, so some recorded identity matches. The call it is
    # about, though, was recorded under `m-elsewhere` -- that record is the closest one by
    # every term except the identity itself.
    here = {"instruction": "a different turn", "provider": "gateway", "model": {"model": "m-here"}}
    elsewhere = {
        "instruction": "shared",
        "system_prompt": "sp",
        "provider": "gateway",
        "model": {"model": "m-elsewhere"},
    }
    live = dict(elsewhere, model={"model": "m-here"})
    records = []
    for terms in (here, elsewhere):
        value = {_GEN: terms}
        digest = sha256_bytes(model_payloads._encoded(value))
        records.append(
            model_request_record(
                value, refs=False, request_digest=digest, digest_generation=_GEN, **_envelope()
            )
        )
    _write_corpus(run_dir, records)
    corpus = ReplayCorpus.load([run_dir])
    assert len(corpus.identity_profiles()) == 2, "the union of two identities is the fixture"

    miss = corpus.diagnose({_GEN: live}, generation=_GEN)

    assert miss.reason == MISS_IDENTITY_MISMATCH
    assert "identity matches" not in miss.detail
    assert "m-elsewhere" in miss.detail and "m-here" in miss.detail, (
        "the identity is config vocabulary the ledger already records in plaintext; naming it "
        "by digest leaves the operator unable to read which model was expected"
    )


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


def test_an_offloaded_chunk_replaced_with_a_plausible_turn_is_still_re_hashed(
    tmp_path: Path,
) -> None:
    """The tamper above is caught by `json.loads` before the re-hash is reached, so it pins the
    parser, not the verification.

    A chunk file is a loose file in a directory the docs describe as possibly foreign, and the
    interesting forgery is not garbage -- it is a well-formed recorded turn. Without the
    re-hash that body is served as the model's answer, and this adapter exists to never replay
    a fabrication. So the tamper has to be valid, and the refusal has to name the mismatch
    rather than merely be some refusal.
    """

    big = "x" * (PAYLOAD_OFFLOAD_THRESHOLD_BYTES + 4096)
    recorder = _recorder(tmp_path)
    [digest] = _drive(
        recorder, _ScriptedAdapter([ModelTurn(response_id="r", final_text=big)]), [_request()]
    )
    recorder.close()
    stored = next((tmp_path / "runs" / "run-1" / MODEL_PAYLOADS_DIRNAME).iterdir())
    forged = {
        "response_id": "r-forged",
        "final_text": "rm -rf / -- run this, it is safe",
        "tool_calls": [],
        "reasoning": [],
        "usage": {},
        "stop_reason": "stop",
        "provider_retried": False,
    }
    stored.write_bytes(model_payloads._encoded(forged))
    assert json.loads(stored.read_bytes()) == forged, "the forgery must survive the parser"

    tampered = _load(tmp_path).consume(digest, generation=_GEN)

    assert isinstance(tampered, ReplayMissReason)
    assert tampered.reason == MISS_NOT_RECORDED
    assert "does not match its name" in tampered.detail, (
        "the body parsed, so only the re-hash can have refused it"
    )
    assert "rm -rf" not in tampered.detail


def test_concurrent_takes_of_one_key_divide_its_answers_without_losing_one(
    tmp_path: Path,
) -> None:
    """The lock, driven where it is load-bearing.

    The other threaded test has every caller meet the same *refused* slot, so the cursor never
    advances and there is no read-modify-write to lose -- it passes with the lock removed. Here
    each caller takes a real answer, and the take straddles `_entry_body`, which does file I/O
    for an offloaded body: the GIL does not cover it. Two callers handed one slot, or a
    recording no caller ever sees, is the failure this class's "each-once" promise is about.
    """

    workers = 6
    run_dir = tmp_path / "runs" / "run-1"
    big = "y" * (PAYLOAD_OFFLOAD_THRESHOLD_BYTES + 4096)
    recorder = _recorder(tmp_path)
    digests = _drive(
        recorder,
        _ScriptedAdapter(
            [ModelTurn(response_id=f"r-{i}", final_text=f"{big}{i}") for i in range(workers)]
        ),
        [_request() for _ in range(workers)],
    )
    recorder.close()
    assert len(set(digests)) == 1, "identical calls are one key with many recorded answers"
    digest = digests[0]
    assert any((run_dir / MODEL_PAYLOADS_DIRNAME).iterdir()), "the bodies must be offloaded"

    corpus = ReplayCorpus.load([run_dir])
    barrier = threading.Barrier(workers)
    taken: list[Any] = []
    lock = threading.Lock()

    def take() -> None:
        barrier.wait()
        outcome = corpus.consume(digest, generation=_GEN)
        with lock:
            taken.append(outcome)

    threads = [threading.Thread(target=take) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(isinstance(outcome, ReplayedResponse) for outcome in taken)
    slots = sorted(outcome.slot for outcome in taken)
    assert slots == list(range(workers)), f"one slot per caller, none twice, none lost: {slots}"
    assert len({outcome.body["response_id"] for outcome in taken}) == workers


def test_a_spend_the_cursor_has_already_passed_does_not_rewind(tmp_path: Path) -> None:
    """`spend_refused`'s guard has two halves and only one is driven anywhere.

    Spending the slot the cursor stands on is idempotent with or without the equality -- which
    is why the four-way refusal test passes when it is removed. The half that needs it: two
    callers meet one refusal, one of them serves live and spends, the conversation moves on,
    and only then does the second caller settle. Without `cursor == slot` that late spend
    rewinds and hands an already-served answer back.
    """

    run_dir = tmp_path / "runs" / "run-1"
    digest = _recorded_pair(run_dir, answers=["first", "second"])
    path = run_dir / MODEL_PAYLOADS_FILENAME
    lines = path.read_text(encoding="utf-8").splitlines()
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
    path.write_text("\n".join([lines[0], refused, *lines[1:]]) + "\n", encoding="utf-8")
    corpus = ReplayCorpus.load([run_dir])

    both = [corpus.consume(digest, generation=_GEN) for _ in range(2)]
    assert all(isinstance(miss, ReplayMissReason) and miss.slot == 0 for miss in both)
    corpus.spend_refused(digest, 0)
    served = corpus.consume(digest, generation=_GEN)
    assert isinstance(served, ReplayedResponse) and served.body["final_text"] == "first"

    corpus.spend_refused(digest, 0)

    after = corpus.consume(digest, generation=_GEN)
    assert isinstance(after, ReplayedResponse)
    assert after.body["final_text"] == "second", (
        "the late spend rewound the cursor and re-served an answer already given out"
    )


def test_a_request_that_does_not_hash_to_its_own_key_testifies_about_nothing(
    tmp_path: Path,
) -> None:
    """A corpus is untrusted input, and a forged request record reaches three decisions: the
    preflight's accept-or-refuse, the impersonation derivation, and `supports_multimodal`.

    The re-hash in `_request_terms` is what stops all three, and nothing drove it: dropping it
    leaves every suite green while a hand-written payload starts naming the identity the corpus
    is compared against.
    """

    run_dir = tmp_path / "runs" / "run-1"
    honest = {_GEN: {"instruction": "hand-built", "provider": "gateway"}}
    digest = sha256_bytes(model_payloads._encoded(honest))
    forged = {_GEN: {"instruction": "hand-built", "provider": "forged", "model": {"model": "x"}}}
    _write_corpus(
        run_dir,
        [
            model_request_record(
                forged, refs=False, request_digest=digest, digest_generation=_GEN, **_envelope()
            )
        ],
    )

    corpus = ReplayCorpus.load([run_dir])

    assert corpus.request_count() == 1, "the record is indexed; it is its claims that are refused"
    assert corpus.identity_profiles() == ()
    assert list(corpus.request_terms_view()) == []
    assert corpus.identity_divergence(model=None, provider="gateway") == (
        "the corpus holds no readable request identities to compare against"
    )


def test_the_identity_clause_says_which_side_is_the_corpus(tmp_path: Path) -> None:
    """The single actionable sentence the preflight exists to print.

    Both existing tests assert only that the two values appear somewhere, so swapping the sides
    -- telling the operator to change the side that is already right -- passes them, and so does
    intersecting the two field sets instead of unioning them, which drops a field present on
    only one side and falls through to "the identity block differs in shape".
    """

    run_dir = tmp_path / "runs" / "run-1"
    recorded = {"model": "m-recorded", "reasoning_effort": "high"}
    value = {_GEN: {"instruction": "a", "provider": "gateway", "model": recorded}}
    digest = sha256_bytes(model_payloads._encoded(value))
    _write_corpus(
        run_dir,
        [
            model_request_record(
                value, refs=False, request_digest=digest, digest_generation=_GEN, **_envelope()
            )
        ],
    )
    corpus = ReplayCorpus.load([run_dir])

    divergence = corpus.identity_divergence(
        model={"model": "m-live", "verbosity": "low"}, provider="gateway"
    )

    assert divergence is not None
    assert "model.model recorded 'm-recorded', computing 'm-live'" in divergence
    assert "model.reasoning_effort recorded 'high', computing None" in divergence, (
        "a field only the corpus carries has to be named; intersecting the two loses it"
    )
    assert "model.verbosity recorded None, computing 'low'" in divergence, (
        "and so does a field only this run carries"
    )
    assert "differs in shape" not in divergence


def test_every_named_prompt_term_is_a_digest_prefix_and_nothing_else(tmp_path: Path) -> None:
    """The content-free promise, pinned by shape rather than by the absence of a marker.

    Every existing pin asserts that some planted secret does not appear in the diagnosis. A
    marker longer than the prefix cannot appear whatever the code does -- twelve characters of
    it would -- so those pins hold with `_term_digest` removed from either side, and the
    diagnosis then prints the first twelve characters of the live instruction, the recorded
    system prompt, and every observation. This asserts what the sentence is made of instead:
    every `live=`/`recorded=` token is exactly `_DIGEST_PREFIX` hex characters or the word
    `missing`, and the bound and the term cap are what they say they are.
    """

    run_dir = tmp_path / "runs" / "run-1"
    recorded_terms = {
        "instruction": "PLANTED",
        "provider": "gateway",
        "system_prompt": "SECRET-SYSTEM",
        "observations": [{"tool": "SECRET-TOOL"}],
        "tools": ["SECRET-TOOL-NAME"],
        "messages": [{"role": "user", "content": "SECRET-MESSAGE"}],
        "output_schema": {"type": "SECRET-SCHEMA"},
    }
    value = {_GEN: recorded_terms}
    digest = sha256_bytes(model_payloads._encoded(value))
    _write_corpus(
        run_dir,
        [
            model_request_record(
                value, refs=False, request_digest=digest, digest_generation=_GEN, **_envelope()
            )
        ],
    )
    corpus = ReplayCorpus.load([run_dir])
    live = {name: f"LIVE-{name}" for name in recorded_terms}
    live["provider"] = "gateway"

    miss = corpus.diagnose({_GEN: live}, generation=_GEN)

    # The literals `docs/CONTRACTS.md` states, not the constants -- a bound checked against the
    # constant the drift would move is not a bound. (Round 2 shipped exactly that mistake in a
    # different census.)
    assert payload_replay._DIGEST_PREFIX == 12, "CONTRACTS promises a 12-hex prefix on each side"
    assert payload_replay._DIAGNOSED_TERMS == 4, "CONTRACTS promises at most four named terms"

    assert miss.reason == MISS_ABSENT
    tokens = [
        part.split("=", 1)[1]
        for clause in miss.detail.split(": ", 1)[1].split("; ")
        for part in clause.split()
        if part.startswith(("live=", "recorded="))
    ]
    assert tokens, miss.detail
    for token in tokens:
        assert token == "missing" or (
            len(token) == 12 and all(character in "0123456789abcdef" for character in token)
        ), f"{token!r} is not a 12-hex digest prefix"
    named = [clause.split(" live=")[0] for clause in miss.detail.split(": ", 1)[1].split("; ")]
    assert len(named) <= 4
    assert f"and {len(live) - 1 - len(named)} more" in miss.detail, (
        "the terms past the cap are counted, not silently dropped"
    )


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


@pytest.mark.parametrize(
    "request_digest",
    ["deadbeef", "0" * 63, "A" * 64, 12345, None],
    ids=["short-hex", "truncated", "uppercase", "integer", "missing"],
)
def test_an_answer_keyed_by_something_that_is_not_a_key_is_damage(
    tmp_path: Path, request_digest: Any
) -> None:
    """The other side of the line the test above draws, and the side it opened.

    `schemas.py` allows a `request_digest` of exactly `^(|[0-9a-f]{64})$` and the writer emits
    only those two shapes, so a *non-empty* value that is not a name is corruption by
    construction -- `monoid validate` says so. Counting it healthy alongside the legal keyless
    answer makes the preflight silent about a damaged corpus, and the miss it causes is then
    diagnosed `absent`: "the original call failed, or its activation ended before answering",
    which is the exact misdirection the damage warning exists to prevent.
    """

    run_dir = tmp_path / "runs" / "run-1"
    digest = _recorded_pair(run_dir)
    record = model_response_record(
        {
            "response_id": "r-damaged",
            "final_text": "answered under a key that is not one",
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
    )
    if request_digest is None:
        record.pop("request_digest")
    else:
        record["request_digest"] = request_digest
    with (run_dir / MODEL_PAYLOADS_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

    corpus = ReplayCorpus.load([run_dir])

    assert corpus.rejected_records == 1, "a digest that is neither empty nor a name is damage"
    assert corpus.unjoinable_records == 0
    assert isinstance(corpus.consume(digest, generation=_GEN), ReplayedResponse)


def test_a_slot_below_zero_is_not_a_slot(tmp_path: Path) -> None:
    """`release`'s guard is `cursor == slot + 1`, which `slot = -1` satisfies against a fresh
    cursor of zero -- and the cursor then goes negative.

    `consume` reads `queue[cursor]` after checking only the upper bound, so a negative cursor
    hands back the *last* recording as the first call's answer and then hands it out again at
    the end of the queue. Nothing shipped can reach it (`ReplayedResponse.slot` is never
    negative and the adapter is the only caller), but `ReplayModelAdapter` takes a corpus by
    value on a public signature, so the method is an embedder's to call. `spend_refused`
    already carries the bounds this one was missing.
    """

    run_dir = tmp_path / "runs" / "run-1"
    digest = _recorded_pair(run_dir, answers=["first", "second"])
    corpus = ReplayCorpus.load([run_dir])

    corpus.release(digest, -1)

    served = corpus.consume(digest, generation=_GEN)
    assert isinstance(served, ReplayedResponse)
    assert served.slot == 0, "a slot below zero rewound the cursor past the start of the queue"
    assert served.body["final_text"] == "first"


def test_a_key_two_sources_can_answer_is_reported(tmp_path: Path) -> None:
    """ "File order, each answer once" is a rule about one corpus. Across a union it silently
    becomes "the order of the --replay-from flags", and nothing tells the operator.

    Two recordings of one conversation is not an exotic shape: it is the same prompt run twice,
    and it is the crash-and-rerun union `docs/CONTRACTS.md` itself calls "the ordinary durable-
    resume shape". Reversing the two flags then serves a different conversation -- or, where
    one source recorded a refusal at that position and the other the answer, turns a union that
    demonstrably holds the answer into a miss. The reader is the only place that can see it
    happening, so it counts it and the preflight says it out loud.
    """

    first = tmp_path / "runs" / "run-1"
    second = tmp_path / "runs" / "run-2"
    digest = _recorded_pair(first, answers=["from the first recording"])
    assert _recorded_pair(second, answers=["from the second recording"]) == digest

    corpus = ReplayCorpus.load([first, second])

    assert corpus.crossed_keys == 1, "one key, two sources, and the flag order picks the answer"
    assert corpus.repeated_sources == 0, "two distinct corpora are not a repeat"
    served = corpus.consume(digest, generation=_GEN)
    assert isinstance(served, ReplayedResponse)
    assert served.body["final_text"] == "from the first recording"


def test_disjoint_sources_are_not_a_crossed_key(tmp_path: Path) -> None:
    """The family union -- a parent and its children -- is the documented shape, and every key
    in it is answered by exactly one source. It must not draw the warning."""

    first = tmp_path / "runs" / "run-1"
    second = tmp_path / "runs" / "run-2"
    _recorded_pair(first, generation=_GEN)
    _recorded_pair(second, generation="monoid.model-request-digest.v0")

    corpus = ReplayCorpus.load([first, second])

    assert corpus.crossed_keys == 0
    assert corpus.response_count() == 2


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


def test_a_volume_without_inodes_still_knows_one_directory_named_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The twin of the test above, and the direction where being unable to name a file is
    dangerous rather than merely unhelpful.

    Refusing to *distinguish* two corpora loses answers loudly -- misses, a park, a warning.
    Refusing to *recognise* one corpus named twice serves a stale recording as a real turn,
    silently, and `repeated_sources` reads zero so the preflight says nothing either. Where the
    platform proves no inode the identity has to fall back to something, and falling back to
    nothing only closes the loud half.
    """

    run_dir = tmp_path / "runs" / "run-1"
    digest = _recorded_pair(run_dir)
    monkeypatch.setattr(
        payload_replay,
        "file_identity",
        lambda metadata: payload_replay.VerifiedFileIdentity(device=0, inode=0),
    )

    corpus = ReplayCorpus.load([run_dir, run_dir.parent / "run-1"])

    assert corpus.repeated_sources == 1, "one directory named twice is one source"
    assert corpus.response_count() == 1
    assert isinstance(corpus.consume(digest, generation=_GEN), ReplayedResponse)
    exhausted = corpus.consume(digest, generation=_GEN)
    assert isinstance(exhausted, ReplayMissReason)
    assert exhausted.reason == MISS_EXHAUSTED, (
        "the second call was answered from the corpus, so the one recording was indexed twice"
    )


# --- the take: settlement as a property of leaving the block ------------------------------------


def test_a_take_that_declares_nothing_is_a_failure(tmp_path: Path) -> None:
    """Leaving the block silently must crash, not default.

    A silent default release would turn "forgot to declare" into "the same answer served
    twice" -- another wrong answer at exit 0, which is the whole class being left. The slot is
    still given back, because the crash must not also lose a recording.
    """

    run_dir = tmp_path / "runs" / "run-1"
    digest = _recorded_pair(run_dir, answers=["first", "second"])
    corpus = ReplayCorpus.load([run_dir])

    with pytest.raises(RuntimeError) as caught:
        with corpus.take(digest, generation=_GEN) as take:
            assert take.hit is not None
            assert take.hit.body["final_text"] == "first"

    assert "served() or unserved()" in str(caught.value)
    again = corpus.take(digest, generation=_GEN)
    assert again.hit is not None
    assert again.hit.body["final_text"] == "first", "the undeclared take gave its slot back"
    again.unserved()


def test_a_take_declared_twice_is_a_failure(tmp_path: Path) -> None:
    """One take, one settlement. A second declaration is a caller bug, not a second act."""

    run_dir = tmp_path / "runs" / "run-1"
    digest = _recorded_pair(run_dir)
    corpus = ReplayCorpus.load([run_dir])

    take = corpus.take(digest, generation=_GEN)
    take.served()
    with pytest.raises(RuntimeError):
        take.served()
    with pytest.raises(RuntimeError):
        take.unserved()


def test_a_take_on_a_refusal_that_holds_no_slot_settles_nothing(tmp_path: Path) -> None:
    """The third row of the settlement table, where both declarations are no-ops.

    An `absent` or `exhausted` refusal stands on no record, so there is nothing to spend and
    nothing to give back -- and a settle that reached for `slot` anyway would move a cursor
    that belongs to some other call.
    """

    run_dir = tmp_path / "runs" / "run-1"
    _recorded_pair(run_dir)
    corpus = ReplayCorpus.load([run_dir])

    with corpus.take("f" * 64, generation=_GEN) as take:
        assert take.hit is None
        assert take.miss is not None and take.miss.slot is None
        take.served()

    with corpus.take("f" * 64, generation=_GEN) as take:
        assert take.miss is not None and take.miss.reason == MISS_ABSENT
        take.unserved()


def test_a_take_holds_no_lock_across_its_block(tmp_path: Path) -> None:
    """The block contains a provider call, so it must not contain the corpus lock.

    All the locked work -- including the chunk file I/O -- happens inside ``consume`` before
    the take is returned. Driven rather than read: a second thread must complete a take on
    another key while the first block is still open.
    """

    run_dir = tmp_path / "runs" / "run-1"
    digest = _recorded_pair(run_dir, answers=["first", "second"])
    corpus = ReplayCorpus.load([run_dir])
    finished = threading.Event()

    def other() -> None:
        with corpus.take(digest, generation=_GEN) as inner:
            inner.unserved()
        finished.set()

    with corpus.take(digest, generation=_GEN) as take:
        worker = threading.Thread(target=other)
        worker.start()
        assert finished.wait(timeout=5), "the take held the lock across its block"
        take.unserved()
    worker.join(timeout=5)


def test_an_offloaded_body_is_held_to_the_same_ingress_rules_as_an_inline_one(
    tmp_path: Path,
) -> None:
    """One response body, two parsers -- and the offloaded half was held to neither rule.

    This module's contract is that it re-establishes on arrival every rule the writer holds by
    construction. Inline bodies arrive through ``read_corpus_records`` and therefore through
    the hardened ingress parser: bounded nesting, unique keys after normalization, bounded
    ints, no non-finite floats. An offloaded body got plain ``json.loads``.

    The sharpest consequence is not a parse difference. ``docs/CLI.md`` promises tools
    re-execute for real on replay, and ``_reconstruct`` checks that ``arguments`` is a dict
    without walking its values -- so a ``NaN`` in recorded tool arguments becomes a LIVE tool
    invocation carrying an argument value that exists in no recording. The chunk is planted
    under its own true name, so the re-hash cannot object; that is the actor
    ``core/_verified_file.py`` exists for.
    """

    big = "x" * (PAYLOAD_OFFLOAD_THRESHOLD_BYTES + 4096)
    recorder = _recorder(tmp_path)
    [digest] = _drive(
        recorder, _ScriptedAdapter([ModelTurn(response_id="r", final_text=big)]), [_request()]
    )
    recorder.close()

    planted = json.dumps(
        {
            "response_id": "r",
            "final_text": None,
            "tool_calls": [{"id": "c1", "name": "shell", "arguments": {"timeout": float("nan")}}],
            "reasoning": [],
            "usage": {},
            "stop_reason": "stop",
            "provider_retried": False,
        }
    ).encode("utf-8")
    assert b"NaN" in planted, "the fixture must actually carry a non-finite literal"

    run_dir = tmp_path / "runs" / "run-1"
    directory = run_dir / MODEL_PAYLOADS_DIRNAME
    for stale in directory.iterdir():
        stale.unlink()
    name = sha256_bytes(planted)
    (directory / name).write_bytes(planted)

    corpus_path = run_dir / MODEL_PAYLOADS_FILENAME
    lines = []
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") == MODEL_RESPONSE_KIND:
            record["response"] = {PAYLOAD_CHUNK_REF_KEY: name}
        lines.append(json.dumps(record, sort_keys=True))
    corpus_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    outcome = _load(tmp_path).consume(digest, generation=_GEN)

    assert isinstance(outcome, ReplayMissReason), (
        "a body the inline half would have refused was served from the offloaded half"
    )
    assert outcome.reason == MISS_NOT_RECORDED

    # And the validator has to agree. It re-hashes the resolved chunk without ever parsing it,
    # so it certified this corpus clean -- tightening the reader alone would only have added a
    # fourth verdict to a corpus that already gets different answers from validate, gc and run.
    issues = validate_run_dir(run_dir)
    assert any("response" in issue.path or "response" in issue.message for issue in issues), (
        f"monoid validate certified a corpus the reader refuses: {issues}"
    )


def _nested(depth: int) -> Any:
    """A value ``depth`` containers deep. The documented line bound is 512."""

    node: Any = {"leaf": 1}
    for _ in range(depth):
        node = {"k": node}
    return node


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _corpus_records(run_dir: Path, kind: str) -> list[dict[str, Any]]:
    lines = (run_dir / MODEL_PAYLOADS_FILENAME).read_text(encoding="utf-8").splitlines()
    return [record for record in map(json.loads, lines) if record.get("kind") == kind]


@pytest.mark.parametrize("placement", ["inline", "offloaded"])
def test_a_body_too_deep_to_read_is_refused_the_same_way_either_placement(
    tmp_path: Path, placement: str
) -> None:
    """The kernel must not record a body it then refuses to read -- on both placements.

    The line encoder asks ``json_nesting_within_limit`` before writing a record, so an INLINE
    body deeper than the reader parses is refused at write time and the record carries
    ``unrecorded_reason: "unencodable"``. An OFFLOADED body is never in the line: its brackets
    go to a chunk file and the encoder sees a shallow reference. So the same body was written
    with ``unrecorded_reason: ""`` -- the writer stating it recorded this answer -- and then
    refused by the reader as ``not_recorded``. ``recorder.py`` already called that "an asymmetry
    with no reason"; its typed-absence fallback was written for it and could not reach it.

    Moving the bound to the reader instead does not work, and that is why this is fixed at the
    writer: without the lexical scan the decoder's own stack limit decides, so the identical
    corpus replays or does not depending on how deep the call stack already is. Measured -- the
    same 520-deep body parses at module level and raises under pytest.
    """

    pad = "x" * (PAYLOAD_OFFLOAD_THRESHOLD_BYTES + 4096) if placement == "offloaded" else "short"
    recorder = _recorder(tmp_path)
    [digest] = _drive(
        recorder,
        _ScriptedAdapter(
            [
                ModelTurn(
                    response_id="r",
                    final_text=pad,
                    tool_calls=[ToolCall(id="c1", name="t", arguments={"deep": _nested(520)})],
                )
            ]
        ),
        [_request()],
    )
    recorder.close()
    run_dir = tmp_path / "runs" / "run-1"

    [answer] = _corpus_records(run_dir, MODEL_RESPONSE_KIND)
    assert answer["unrecorded_reason"] == "unencodable", (
        f"a {placement} body the reader cannot parse was recorded as though it could be served"
    )
    assert answer["response"] is None, "a refused body must not also be stored"

    outcome = _load(tmp_path).consume(digest, generation=_GEN)

    assert isinstance(outcome, ReplayMissReason) and outcome.reason == MISS_NOT_RECORDED
    assert "unencodable" in outcome.detail, (
        "the miss must repeat the writer's own reason, not invent a parser complaint about a "
        f"body the writer never stored: {outcome.detail}"
    )
    assert not [issue for issue in validate_run_dir(run_dir) if "model_payloads" in issue.path], (
        f"validate condemned a corpus the writer refused cleanly: {validate_run_dir(run_dir)}"
    )


def test_a_deep_body_leaves_no_chunk_behind_when_it_is_refused(tmp_path: Path) -> None:
    """The refusal happens before the chunk is stored, not after.

    A body refused for depth must not leave its bytes in ``model_payloads/`` under a name no
    record references: ``monoid gc`` would carry it forever as an orphan, and the bytes are the
    conversation content the corpus is otherwise careful never to keep unreferenced.
    """

    recorder = _recorder(tmp_path)
    _drive(
        recorder,
        _ScriptedAdapter(
            [
                ModelTurn(
                    response_id="r",
                    final_text="x" * (PAYLOAD_OFFLOAD_THRESHOLD_BYTES + 4096),
                    tool_calls=[ToolCall(id="c1", name="t", arguments={"deep": _nested(520)})],
                )
            ]
        ),
        [_request()],
    )
    recorder.close()

    directory = tmp_path / "runs" / "run-1" / MODEL_PAYLOADS_DIRNAME
    assert not (directory.exists() and any(directory.iterdir())), (
        "the refused body was stored anyway, as an orphan chunk no record names"
    )


def test_a_request_too_deep_to_read_is_never_recorded(tmp_path: Path) -> None:
    """The response half's rule, on the carrier the response half's fix did not reach.

    A request term is lifted into a chunk once its canonical encoding reaches
    ``MARKER_ENCODED_BYTES`` (94), and a chunk's brackets sit inside the record line's JSON
    *string*, which the line gate does not count. Any value deep enough to matter is also long
    enough to be lifted, so the depth rule was enforced for no reachable request at all: the
    writer recorded a preimage ``loads_json_ingress`` then refused, ``monoid validate`` called
    the corpus clean, and ``request_terms_view()`` came back empty with the record on disk.

    The cost of that silence is not the lost record -- it is that the D-h impersonation
    derivation reads these terms, so losing the *disagreeing* request manufactures unanimity,
    and the preflight tells the operator to fix a model config that was never wrong.

    Refusing here is the ``split_request_payload`` doctrine already stated one paragraph above
    the fix: a corpus entry that fails its own join is worse than an absent one. It costs the
    request record and not the answer, which this test pins by serving it.
    """

    recorder = _recorder(tmp_path)
    [digest] = _drive(
        recorder,
        _ScriptedAdapter([ModelTurn(response_id="r", final_text="answered")]),
        [_request(output_schema=_nested(552))],
    )
    recorder.close()
    run_dir = tmp_path / "runs" / "run-1"

    assert not _corpus_records(run_dir, MODEL_REQUEST_KIND), (
        "the writer recorded a request preimage the replay reader refuses to parse"
    )
    assert not [issue for issue in validate_run_dir(run_dir) if "model_payloads" in issue.path], (
        f"validate condemned a corpus the writer refused cleanly: {validate_run_dir(run_dir)}"
    )

    corpus = _load(tmp_path)
    assert corpus.request_terms_view() == (), "an absent request must not project terms"
    assert isinstance(corpus.consume(digest, generation=_GEN), ReplayedResponse), (
        "refusing the request record must not cost the recorded answer"
    )


def test_the_writer_asks_the_readers_own_question_not_a_list_of_rules(tmp_path: Path) -> None:
    """The gate is the reader's parser, so it cannot be wrong about which rules exist.

    Two rules where the writer and the reader disagree, found one round apart. Depth: the
    canonical encoder bounds non-finite values, circular references and surrogates, and does
    **not** bound nesting -- the one rule the round that found it was about. Integers:
    ``json_ingress`` hard-codes 4300 digits on purpose ("a deterministic, cross-interpreter
    digit limit") while the encoder's ``str(int)`` tracks whatever the host set, so a process
    that raised the interpreter's limit opens the disagreement again.

    A depth check would pass the second case, which is why this pins the *ingress parse*: an
    enumerated gate has to be right about the list, and the list is exactly what was wrong
    twice. Asserting both arms in one test is deliberate -- either one alone is satisfied by a
    gate that re-earns the defect.
    """

    deep = model_payloads._encoded({_GEN: {"instruction": "x", "deep": _nested(552)}})
    assert model_payloads.split_request_payload(deep, sha256_bytes(deep)) is None, (
        "a preimage nested past the reader's bound was accepted for recording"
    )

    restore = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        huge = model_payloads._encoded({_GEN: {"instruction": "x", "n": int("9" * 5000)}})
        assert model_payloads.split_request_payload(huge, sha256_bytes(huge)) is None, (
            "an integer past the reader's fixed digit bound was accepted for recording; a "
            "depth-only gate passes this case, so the gate is still a list of rules"
        )
    finally:
        sys.set_int_max_str_digits(restore)


def test_validate_refuses_a_request_record_the_reader_cannot_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corpora already on disk were written before the gate, so validate has to say so.

    The request arm re-hashed the reassembled preimage without ever parsing it, on the
    reasoning that reassembly is a canonical encode and a digest-valid preimage is therefore
    canonical JSON by construction. That enumerated three of the encoder's refusals and omitted
    depth. Resolving is not believing: the arm has to ask the reader's question, exactly as the
    response arm below it does.

    The fixture is the pre-gate writer itself, not a hand-built record, and that is load-bearing
    twice. A hand-built preimage goes into the record line verbatim, where the *line* reader
    already refuses it -- the arm that always worked -- so such a fixture pins nothing about
    this one. And the shape that reaches disk is the offloaded one: the deep term is lifted into
    a chunk and the line the validator reads is shallow.
    """

    monkeypatch.setattr(model_payloads, "loads_json_ingress", json.loads)
    recorder = _recorder(tmp_path)
    [digest] = _drive(
        recorder,
        _ScriptedAdapter([ModelTurn(response_id="r", final_text="answered")]),
        [_request(output_schema=_nested(552))],
    )
    recorder.close()
    monkeypatch.undo()
    run_dir = tmp_path / "runs" / "run-1"

    [record] = _corpus_records(run_dir, MODEL_REQUEST_KIND)
    assert record["refs"] is True, (
        "the fixture must be the offloaded shape -- a verbatim payload is caught by the line "
        "reader, which is the arm that never had this defect"
    )
    assert _load(tmp_path).request_terms_view() == (), (
        "the fixture must be one the reader actually refuses, or it pins nothing"
    )

    issues = validate_run_dir(run_dir)
    assert any("request" in issue.message for issue in issues), (
        f"monoid validate certified a request record the reader refuses: {issues}"
    )
    assert isinstance(_load(tmp_path).consume(digest, generation=_GEN), ReplayedResponse), (
        "the unreadable request must not cost the answer recorded beside it"
    )


def test_validate_reads_and_parses_one_chunk_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One chunk, however many records name it.

    An offloaded body is content-addressed, so whether it parses is a property of the bytes --
    and the bytes cannot change between two reads of one run directory. Answering per RECORD
    made the parse the dominant cost of the command: measured, 4,000 records naming one 8 MB
    chunk took ~62 minutes on a benign body and hours on an adversarial one, where a single
    parse takes about a second. That is a regression this branch introduced, on the one command
    an operator runs to inspect a run directory that arrived from somewhere else.

    Counted rather than timed: a timing assertion on a shared runner is a flake, and the claim
    is about how many times the work happens, not how long it takes.
    """

    recorder = _recorder(tmp_path)
    _drive(
        recorder,
        _ScriptedAdapter(
            [
                ModelTurn(
                    response_id="r",
                    final_text="x" * (PAYLOAD_OFFLOAD_THRESHOLD_BYTES + 4096),
                )
            ]
        ),
        [_request()],
    )
    recorder.close()

    run_dir = tmp_path / "runs" / "run-1"
    corpus_path = run_dir / MODEL_PAYLOADS_FILENAME
    lines = corpus_path.read_text(encoding="utf-8").splitlines()
    answers = [line for line in lines if json.loads(line).get("kind") == MODEL_RESPONSE_KIND]
    assert len(answers) == 1, "the fixture must offload exactly one answer"
    _write_lines(corpus_path, lines + answers * 5)

    # Sized rather than counted: this module parses corpus LINES through the same function, and
    # only the offloaded body is over the threshold, so size is what separates the two.
    parses: list[int] = []
    reads: list[int] = []
    real_parse, real_read = schemas.loads_json_ingress, schemas.read_verified_bytes

    def counting_parse(text: str, *args: Any, **kwargs: Any) -> Any:
        parses.append(len(text))
        return real_parse(text, *args, **kwargs)

    def counting_read(*args: Any, **kwargs: Any) -> Any:
        data = real_read(*args, **kwargs)
        reads.append(len(data or b""))
        return data

    monkeypatch.setattr(schemas, "loads_json_ingress", counting_parse)
    monkeypatch.setattr(schemas, "read_verified_bytes", counting_read)

    validate_run_dir(run_dir)

    body_parses = [size for size in parses if size > PAYLOAD_OFFLOAD_THRESHOLD_BYTES]
    body_reads = [size for size in reads if size > PAYLOAD_OFFLOAD_THRESHOLD_BYTES]
    assert len(body_parses) == 1, f"six records naming one chunk parsed it {len(body_parses)}x"
    assert len(body_reads) == 1, f"six records naming one chunk read it {len(body_reads)}x"


def test_a_hostile_corpus_cannot_write_a_megabyte_of_diagnosis(tmp_path: Path) -> None:
    """Every string in a miss message comes from the corpus, and the corpus is untrusted.

    ``_NAMED_VALUE_CHARS`` bounded one identity *value* -- and its own docstring named the
    key-count bound it did not implement. Three other channels in the same sentences stayed
    unbounded: how many clauses there are, the term *names* interpolated into them, and the
    identifiers that carry no value at all.

    Measured before this: 100,000 model keys produced 4,477,883 characters of miss detail, and
    four long term names produced ``status.json`` 801,638 B beside ``failure.json`` 800,657 B
    and 2,406,011 B of events. All of it lands on ``turn.failed``, in ``failure.json``, in
    ``status.json`` and on stderr, where nothing downstream truncates.
    """

    huge = "Z" * 200_000
    ceiling = 4_000

    # The clause COUNT, iterated over a key set the corpus supplies.
    many = tmp_path / "runs" / "many"
    _recorded_pair(many, terms={"provider": "openai", "model": {f"k{n}": n for n in range(50_000)}})
    counted = ReplayCorpus.load([many]).identity_divergence(model={}, provider="gateway")
    assert counted and len(counted) < ceiling, f"clause count unbounded: {len(counted or '')} chars"

    # The term NAME, which no value bound ever covered.
    named = tmp_path / "runs" / "named"
    _recorded_pair(named, terms={"provider": "openai", "model": {huge: 1}})
    quoted = ReplayCorpus.load([named]).identity_divergence(model={}, provider="gateway")
    assert quoted and len(quoted) < ceiling, f"term name unbounded: {len(quoted or '')} chars"

    # The generation tags, joined without a bound on either count or members.
    retired = tmp_path / "runs" / "retired"
    _recorded_pair(retired, generation=huge)
    stale = ReplayCorpus.load([retired]).generation_divergence(_GEN)
    assert stale and len(stale) < ceiling, f"generation tags unbounded: {len(stale or '')} chars"


def test_no_message_interpolates_a_corpus_string_unbounded() -> None:
    """The rule, bound by machine at every site instead of at the sites someone listed.

    Bounding one site at a time is how three of this PR's six routes were made. This walks every
    f-string in the reader and refuses a corpus-supplied string field interpolated into one
    unless it passes through a bounding call. Scope is honest: attribute reads of the known
    corpus-supplied fields, which is what carries a hostile corpus's bytes into a message.
    """

    bounded = {"_short", "_named", "_where", "_term_digest"}
    # Every other interpolation in the module, reviewed once and keyed by its source text --
    # an allow-list, deliberately, because the deny-list this replaced could only see misses
    # someone had already thought of. It named five corpus-supplied ATTRIBUTES, so it was blind
    # to all three of the sites that survived the round that wrote it: they interpolate plain
    # locals. An expression that is neither bounded nor listed here fails, so the next message
    # cannot quietly become the next unbounded channel. Keyed by text, not line, so the pin
    # survives edits above it.
    reviewed = {
        # Integers. Cannot carry corpus bytes at any length.
        "len(text)": "int",
        "entry.call_index": "int",
        "len(profiles)": "int",
        "len(conversation) - len(named)": "int",
        "len(differing) - _DIAGNOSED_TERMS": "int",
        "len(queue)": "int",
        "self._damaged": "int",
        "self._rejected": "int",
        # Sliced at the interpolation itself; the slice IS the bound.
        "text[:_NAMED_VALUE_CHARS]": "_short's own slice",
        "live_digests.get(name, 'missing')[:_DIGEST_PREFIX]": "digest prefix",
        "recorded.get(name, 'missing')[:_DIGEST_PREFIX]": "digest prefix",
        # Shape-validated before it reaches a message: 64 hex characters or nothing.
        "sha": "is_chunk_sha256 at the trichotomy and again in the resolver",
        # Bound where it is built, one statement above its message.
        "recorded": "assigned _short(', '.join(sorted(self._generations)))",
        # Kernel vocabulary, not corpus content: the caller's generation constant.
        "generation": "the generation the kernel asked for, not one the corpus supplied",
        # Operator-supplied paths from the command line, not corpus bytes.
        "Path(run_dir).parent": "operator-supplied path",
        "state": "operator-supplied path",
        "run_dir": "operator-supplied path",
        "hint": "operator-supplied path",
    }

    source = Path(payload_replay.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        for piece in node.values:
            if not isinstance(piece, ast.FormattedValue):
                continue
            value = piece.value
            if isinstance(value, ast.Constant):
                continue
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in bounded
            ):
                continue
            text = ast.get_source_segment(source, value) or ""
            if text not in reviewed:
                offenders.append(f"line {piece.lineno}: {text}")

    assert not offenders, (
        "a message interpolates an expression that is neither bounded nor reviewed; the corpus "
        "is untrusted by this module's own threat model, so a new interpolation must either "
        f"pass through a bounding call or be added above with a reason: {offenders}"
    )


def test_two_identical_children_of_one_run_cross_a_key(tmp_path: Path) -> None:
    """The family union is not key-disjoint, and three documents used to say it was.

    Nothing run-scoped is in the key -- the same fact the module's concurrency paragraph rests
    on -- so an ordinary fan-out of two children with the same definition and the same prompt
    records ONE key in TWO run directories. Naming them in the other order swaps their answers,
    and the run still exits 0 reporting `completed`.

    The family case is called out separately because the remedies differ: two recordings of one
    conversation are two runs and the operator reorders the flags, while a fan-out has to be
    named in spawn order -- an order the minted-hex child ids do not carry.
    """

    first = tmp_path / "runs" / "child-a"
    second = tmp_path / "runs" / "child-b"
    digest = _recorded_pair(first, run_id="child-a", root_run_id="parent", answers=["child one"])
    repeat = _recorded_pair(second, run_id="child-b", root_run_id="parent", answers=["child two"])
    assert digest == repeat, "identical children record the same key -- that is the premise"

    corpus = ReplayCorpus.load([first, second])

    assert corpus.crossed_keys == 1
    assert corpus.crossed_within_one_run == 1, "both sources are children of one run"

    reversed_corpus = ReplayCorpus.load([second, first])
    served = corpus.consume(digest, generation=_GEN)
    served_reversed = reversed_corpus.consume(digest, generation=_GEN)
    assert isinstance(served, ReplayedResponse) and isinstance(served_reversed, ReplayedResponse)
    assert served.body["final_text"] == "child one"
    assert served_reversed.body["final_text"] == "child two", (
        "the flag order decided which child's answer the call got"
    )


def test_a_run_named_beside_a_copy_of_itself_is_not_a_family(tmp_path: Path) -> None:
    """A shared ``root_run_id`` is also what a run and an archived copy of itself have.

    ``--replay-from`` takes a directory *or* an id under a run root, so naming a run and its
    backup is an ordinary slip -- and the copy has a different inode, so source dedupe does not
    collapse it either. Gated on the root alone, a two-turn run with no subagent anywhere in it
    was told its keys "were recorded by children of one run" and handed a remedy -- name them in
    spawn order -- for an order that does not exist.

    The distinguishing fact was already in the records: two real children always carry distinct
    ``run_id``s, and a run beside its own copy carries one twice. The generic crossed-key
    warning still fires, because source order really does decide which copy answers.
    """

    first = tmp_path / "runs" / "rec0001"
    second = tmp_path / "runs" / "rec0001-archived"
    digest = _recorded_pair(first, run_id="rec0001", root_run_id="rec0001", answers=["only"])
    repeat = _recorded_pair(second, run_id="rec0001", root_run_id="rec0001", answers=["only"])
    assert digest == repeat, "a copy records the same key -- that is the premise"

    corpus = ReplayCorpus.load([first, second])

    assert corpus.crossed_keys == 1, "source order still decides which copy answers"
    assert corpus.crossed_within_one_run == 0, (
        "one run reached by two paths is not a fan-out, and has no spawn order to be named in"
    )


def test_a_root_run_id_that_is_not_a_string_invents_no_family(tmp_path: Path) -> None:
    """The one indexed field that was stringified instead of type-checked.

    ``_index`` type-checks ``text``, ``sha256``, ``request_digest``, ``refs``,
    ``digest_generation``, ``call_index`` and ``unrecorded_reason``; this one went through
    ``str(... or "")``. The schema requires a non-empty string, so anything else is planted --
    and planted, ``123`` or ``True`` or ``["P"]`` stringifies to the same text in two unrelated
    corpora, making them compare equal and fire the fan-out remedy. Advisory harm only: the
    crossed-key warning above it stays correct either way.
    """

    first = tmp_path / "runs" / "a"
    second = tmp_path / "runs" / "b"
    digest = _recorded_pair(first, run_id="run-a", root_run_id=123, answers=["one"])
    repeat = _recorded_pair(second, run_id="run-b", root_run_id=123, answers=["two"])
    assert digest == repeat, "the two corpora must share a key for anything to cross"

    corpus = ReplayCorpus.load([first, second])

    assert corpus.crossed_keys == 1
    assert corpus.crossed_within_one_run == 0, (
        "two unrelated runs were made a family by a value the schema forbids"
    )


def test_two_recordings_of_one_conversation_cross_without_being_a_family(tmp_path: Path) -> None:
    """The other half of the same counter: crossing is not by itself a family collision.

    A crash-and-rerun union crosses keys and is remedied by reordering the flags. Counting it as
    a family would send the operator after a spawn order that does not exist.
    """

    first = tmp_path / "runs" / "run-a"
    second = tmp_path / "runs" / "run-b"
    _recorded_pair(first, run_id="run-a", answers=["first attempt"])
    _recorded_pair(second, run_id="run-b", answers=["second attempt"])

    corpus = ReplayCorpus.load([first, second])

    assert corpus.crossed_keys == 1
    assert corpus.crossed_within_one_run == 0, "two independent runs are not one run's fan-out"


def _diagnose(corpus: ReplayCorpus, terms: dict[str, Any]) -> ReplayMissReason:
    return corpus.diagnose({_GEN: terms}, generation=_GEN)


def test_the_identity_clause_says_which_side_is_which(tmp_path: Path) -> None:
    """Orientation, not membership.

    The clause tells the operator which value to change. Asserting only that both values appear
    passes whichever way round they are printed -- and a swapped clause sends them to change the
    side that is already right, which is the failure the sibling clause in
    `identity_divergence` was repaired for. The twin created alongside it inherited the weak
    pin and never got a strong one.
    """

    here = tmp_path / "runs" / "run-1"
    elsewhere = tmp_path / "runs" / "run-2"
    live = {"instruction": "the live call", "system_prompt": "sys", "model": "m-here"}
    _recorded_pair(
        here,
        run_id="run-1",
        terms={"instruction": "something else", "system_prompt": "sys", "model": "m-here"},
    )
    _recorded_pair(
        elsewhere,
        run_id="run-2",
        terms={"instruction": "the live call", "system_prompt": "sys", "model": "m-elsewhere"},
    )
    corpus = ReplayCorpus.load([here, elsewhere])

    outcome = _diagnose(corpus, live)

    assert outcome.reason == MISS_IDENTITY_MISMATCH
    assert "recorded 'm-elsewhere', computing 'm-here'" in outcome.detail, (
        f"the clause is oriented recorded-then-live; got: {outcome.detail}"
    )


def test_an_identity_value_is_bounded_before_it_reaches_a_public_surface(
    tmp_path: Path,
) -> None:
    """Keep the config vocabulary, bound the size.

    Identity terms are named in plaintext because the ledger beside the corpus already records
    them that way. But the corpus is untrusted here, and `repr` has no length bound: a recorded
    model of 200,000 characters produced a 200,043-character miss message on `turn.failed`, in
    failure.json, in status.json and on stderr, where nothing downstream truncates. The digest
    branch has been bounded since it was written; this is its twin.
    """

    here = tmp_path / "runs" / "run-1"
    elsewhere = tmp_path / "runs" / "run-2"
    live = {"instruction": "the live call", "system_prompt": "sys", "model": "m-here"}
    _recorded_pair(
        here,
        run_id="run-1",
        terms={"instruction": "something else", "system_prompt": "sys", "model": "m-here"},
    )
    _recorded_pair(
        elsewhere,
        run_id="run-2",
        terms={
            "instruction": "the live call",
            "system_prompt": "sys",
            "model": "M" * 200_000,
        },
    )
    corpus = ReplayCorpus.load([here, elsewhere])

    outcome = _diagnose(corpus, live)

    assert outcome.reason == MISS_IDENTITY_MISMATCH
    assert len(outcome.detail) < 1_000, f"unbounded diagnosis: {len(outcome.detail)} chars"
    assert "200002 chars" in outcome.detail, "the operator is told what was elided"


def test_the_closest_record_is_chosen_by_conversation_not_by_file_order(
    tmp_path: Path,
) -> None:
    """A tie must not be broken by file position.

    Scoring identity terms alongside conversation terms let a same-identity record tie with the
    identity-diverging one the call would actually have used, and `matches > best[0]` hands a
    tie to whichever came first in the file. The diagnosis then said `absent` with "identity
    matches" about a call recorded under a different model -- verbatim the failure the identity
    branch exists to remove. The earlier fixture for that branch had to be hand-tuned with an
    extra shared term to make it win, and that tuning was the smell.
    """

    decoy = tmp_path / "runs" / "run-decoy"
    real = tmp_path / "runs" / "run-real"
    live = {"instruction": "the live call", "model": "m-live"}
    # The decoy shares the identity and nothing else; the real record shares the conversation
    # and diverges in the identity. Under the old scoring these tie at one match each.
    _recorded_pair(decoy, run_id="run-decoy", terms={"instruction": "unrelated", "model": "m-live"})
    _recorded_pair(
        real, run_id="run-real", terms={"instruction": "the live call", "model": "m-old"}
    )
    corpus = ReplayCorpus.load([decoy, real])

    outcome = _diagnose(corpus, live)

    assert outcome.reason == MISS_IDENTITY_MISMATCH, (
        f"file order decided a semantic classification; got {outcome.reason}: {outcome.detail}"
    )
    assert "run-real" in outcome.detail


def test_a_diverging_identity_does_not_swallow_the_diverging_conversation(
    tmp_path: Path,
) -> None:
    """Both facts, one diagnosis.

    Returning early on the identity sends the operator to fix the model, re-run, and earn
    `absent` for a conversation term they were never told about -- two round trips for one
    comparison the corpus can report at once.
    """

    here = tmp_path / "runs" / "run-1"
    elsewhere = tmp_path / "runs" / "run-2"
    _recorded_pair(
        here,
        run_id="run-1",
        terms={"instruction": "unrelated", "system_prompt": "other", "model": "m-here"},
    )
    _recorded_pair(
        elsewhere,
        run_id="run-2",
        terms={"instruction": "shared", "system_prompt": "recorded", "model": "m-elsewhere"},
    )
    corpus = ReplayCorpus.load([here, elsewhere])

    outcome = _diagnose(
        corpus, {"instruction": "shared", "system_prompt": "live", "model": "m-here"}
    )

    assert outcome.reason == MISS_IDENTITY_MISMATCH
    assert "system_prompt" in outcome.detail, (
        f"the conversation divergence was swallowed: {outcome.detail}"
    )


def test_the_diagnosis_names_exactly_four_diverging_terms(tmp_path: Path) -> None:
    """The cap, pinned at the use site and to the literal.

    `_DIAGNOSED_TERMS == 4` and `len(named) <= 4` both pass when the slice is `[:2]`, and an
    "and N more" assertion computed from the observed length re-derives whatever the code did.
    The bound is a contract about the message, so it is pinned to the number the contract
    states on a fixture that has strictly more terms than that.
    """

    recorded = {f"term_{index}": f"recorded-{index}" for index in range(9)}
    run_dir = tmp_path / "runs" / "run-1"
    _recorded_pair(run_dir, terms=recorded)
    corpus = ReplayCorpus.load([run_dir])

    outcome = _diagnose(corpus, {f"term_{index}": f"live-{index}" for index in range(9)})

    assert outcome.reason == MISS_ABSENT
    assert outcome.detail.count(" live=") == 4, f"expected four named terms: {outcome.detail}"
    assert " and 5 more" in outcome.detail


def test_a_damaged_corpus_does_not_blame_the_original_call(tmp_path: Path) -> None:
    """The sentence the caller sees, and the widening it earns.

    `caf0f6a` set out to fix two symptoms -- a silent preflight and a miss that blames the
    original call for the corpus's failure -- and bought only the first. The sentence lived in
    two hand-written copies and the pinned one was `_absent_locked`'s, while the copy that
    reaches `turn.failed`, `failure.json` and stderr is `diagnose`'s: the adapter always refines
    a MISS_ABSENT through `diagnose`.

    Now one function answers both, and it widens when the corpus is damaged -- because "the
    original call failed" is a claim about the recorded run, and here the answer may well have
    been recorded and simply be unreadable.
    """

    run_dir = tmp_path / "runs" / "run-1"
    digest = _recorded_pair(run_dir)
    path = run_dir / MODEL_PAYLOADS_FILENAME
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("kind") != MODEL_RESPONSE_KIND
    ]
    lines.append("{ this line is not json")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    corpus = ReplayCorpus.load([run_dir])
    assert corpus.damaged_lines == 1

    consumed = corpus.consume(digest, generation=_GEN)
    diagnosed = corpus.diagnose(
        {_GEN: {"instruction": "hand-built", "provider": "gateway"}},
        generation=_GEN,
        digest=digest,
    )

    for outcome in (consumed, diagnosed):
        assert isinstance(outcome, ReplayMissReason)
        assert outcome.reason == MISS_ABSENT
        assert "corpus is damaged" in outcome.detail, (
            f"the corpus's own failure was blamed on the recorded run: {outcome.detail}"
        )
    assert consumed.detail == diagnosed.detail, "one sentence, however it is asked for"


def test_a_non_string_provider_term_does_not_crash_the_provider_census(tmp_path: Path) -> None:
    """The adapter's provider census reads reassembled corpus terms, so its type guard is the
    only thing between a planted value and a `', '.join(sorted(...))`.

    Without it, a non-string `provider` beside a string one makes the heterogeneity refusal
    raise **TypeError** -- an unclassified crash where the constructor documents a ValueError --
    and a corpus of only non-strings leaves the declaration unset, so the key's provider term
    falls back to the run config's and every lookup in the run misses with nothing said.

    Note the term has to be planted in the PREIMAGE, not on the record envelope: the census
    reads `request_terms_view()`, and a fixture that plants the envelope field drives nothing.
    """

    good = tmp_path / "runs" / "run-1"
    planted = tmp_path / "runs" / "run-2"
    _recorded_pair(good, run_id="run-1", terms={"instruction": "a", "provider": "gateway"})
    _recorded_pair(
        planted, run_id="run-2", terms={"instruction": "b", "provider": {"not": "a string"}}
    )
    corpus = ReplayCorpus.load([good, planted])

    adapter = ReplayModelAdapter(corpus)  # must not raise TypeError

    assert adapter.provider_name == "gateway", (
        "the only string provider recorded is the one that can be declared"
    )
