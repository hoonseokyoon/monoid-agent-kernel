"""What a recorded model call may say, and the two things it must never say.

W6-1 (dx-note ``2026-08-02-v0.21-contract-replay-scope.md`` §Track B, decision 1). The ledger is
the first artifact to put ``ModelCallReceipt`` data on disk, which turns two properties that were
previously live-only into durable ones.

**Mutation gate.** ``model_call_record`` and its two sub-projections are the single place a
receipt becomes a record. Mutating any of them must turn all four of these red -- if one survives,
the binding is broken:

  1. the endpoint-exclusion test here (the plaintext the keyed digest exists to prevent),
  2. the structural pin here (the projection reflects over nothing),
  3. the writer/schema key agreement here,
  4. the sidecar's end-to-end shape (``tests/test_model_calls.py``).

The structural pin carries its own mutation test, because a pin is a claim about which edits turn
it red and that claim is testable.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.model_calls import (
    MODEL_CALL_KIND,
    MODEL_CALLS_FILENAME,
    MODEL_CALLS_SCHEMA_VERSION,
    _recorded_attempt,
    _recorded_context,
    _recorded_model,
    model_call_record,
)
from monoid_agent_kernel.core.model_io import ModelCallReceipt, destination_digest
from monoid_agent_kernel.core.schemas import MODEL_CALLS_RECORD_SCHEMA, validate_run_dir
from monoid_agent_kernel.core.spec import GenerationConfig, ModelConfig

_ENDPOINT = "https://gateway.internal.example/tenant-a/llm/turns"


def _receipt(**changes: object) -> ModelCallReceipt:
    base = ModelCallReceipt(
        context=InvocationContext(run_id="run-1", step_id="turn_0001"),
        model=ModelConfig(gateway_url=_ENDPOINT),
        provider_name="gateway",
        prompt_digest="a" * 64,
        request_digest="b" * 64,
        digest_generation="monoid.model-request-digest.v1",
        digest_status="ok",
        destination_status="resolved",
        destination_digest=destination_digest(_ENDPOINT),
        stop_reason="stop",
        usage={"input_tokens": 12, "output_tokens": 3},
        latency_ms=42,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def _record(**changes: object) -> dict[str, object]:
    return model_call_record(
        _receipt(**changes),
        run_id="run-1",
        root_run_id="run-1",
        call_index=0,
        recorded_at="2026-08-06T00:00:00Z",
    )


_ATTEMPT: dict[str, object] = {
    "index": 1,
    "elapsed_ms": 0,
    "error_code": "",
    "provider_error_code": "",
    "retryable": False,
    "config_recoverable": False,
    "http_status": None,
    "provider_retried": False,
    "usage": {},
    "stream_committed": False,
}


def _errors(record: dict[str, object]) -> list[str]:
    return [
        error.message
        for error in Draft202012Validator(MODEL_CALLS_RECORD_SCHEMA).iter_errors(record)
    ]


# --- what the record must never carry ------------------------------------------------------


def test_a_recorded_line_never_carries_the_endpoint_it_hashes() -> None:
    """The receipt states the destination two ways, and only one of them may be written down.

    ``destination_digest`` is keyed precisely so a record can compare destinations without
    becoming a confirm-a-guess oracle for an internal hostname. But the receipt also carries the
    whole ``ModelConfig``, whose ``to_json`` emits ``gateway_url`` unconditionally -- and for the
    gateway adapter the configured URL *is* the resolved destination. Serializing the receipt
    therefore writes the digest's own preimage in the adjacent field, and since the key is
    per-process, one such line makes every other digest in the file confirmable.

    This is not hypothetical: the shipped scaffold configures a gateway URL, so it would be line
    one of a new agent's first run.
    """
    record = _record()
    rendered = json.dumps(record, sort_keys=True)

    assert _ENDPOINT not in rendered
    assert "gateway.internal.example" not in rendered
    assert "gateway_url" not in rendered
    # The status survives, because it is the half that stays true across a restart and it is what
    # an auditor actually asks: did the probe resolve, decline, fail, or never run.
    assert record["destination_status"] == "resolved"
    assert "destination_digest" not in record


def test_a_recorded_line_omits_the_per_subscription_redaction_digest() -> None:
    """A field that is structurally empty on this seam would be recorded as a false statement.

    ``redaction_digest`` is set only by the per-subscription narrowing, and the receipt handed to
    the recording seam never passes through it. Writing the field would put "no redaction rules
    were applied" on every line -- including the lines for calls a redacted consumer really did
    apply rules to. A constant that reads as a fact is worse than an absent key.
    """
    assert "redaction_digest" not in _record()
    # Even when a receipt somehow carries one, the projection is a declared list and does not
    # forward it: the exclusion is structural, not a filter on the value.
    assert "redaction_digest" not in _record(redaction_digest="c" * 64)


def test_a_recorded_line_omits_transport_policy_and_derived_properties() -> None:
    """Two smaller exclusions, asserted so a later edit has to argue with them.

    Transport policy says how a call was carried, not what was asked for; the gateway wire already
    omits it so each hop owns its own. The derived properties (``succeeded``/``trace_id``/
    ``span_id``) are computed from ``error_code`` and ``traceparent``, both of which are recorded
    -- and a stored derivation is a value that can come to disagree with its source.
    """
    record = _record()

    assert set(record["model"]) <= {"provider", "model", "reasoning", "generation"}  # type: ignore[arg-type]
    for absent in ("timeout_s", "retry", "succeeded", "trace_id", "span_id"):
        assert absent not in record
        assert absent not in record["model"]  # type: ignore[operator]


# --- what the record must carry ------------------------------------------------------------


def test_the_record_carries_the_replay_key_and_the_failure_taxonomy_together() -> None:
    """A ledger that dropped either would answer half of the question it exists for."""
    failed = _record(
        error_code="model_adapter_error",
        provider_error_code="rate_limit",
        retryable=True,
        config_recoverable=False,
        http_status=429,
        attempts=1,
    )

    assert failed["request_digest"] == "b" * 64
    assert failed["digest_status"] == "ok"
    assert failed["digest_generation"] == "monoid.model-request-digest.v1"
    assert failed["provider_error_code"] == "rate_limit"
    assert failed["http_status"] == 429
    assert failed["retryable"] is True


def test_the_recorded_model_names_the_configured_provider_the_replay_key_leaves_out() -> None:
    """The two hand-listed projections differ deliberately, and this is where.

    The replay key carries a *resolved* provider as a sibling term and omits
    ``ModelConfig.provider`` from its model block. A record wants the configured value as well,
    because ``receipt.provider_name or receipt.model.provider`` is how every other reader of a
    receipt spells "who served this".
    """
    from monoid_agent_kernel.model_call import _model_identity

    model = ModelConfig(provider="openai", gateway_url=_ENDPOINT)

    assert _recorded_model(model)["provider"] == "openai"
    assert "provider" not in _model_identity(model)


def test_the_generation_block_is_omitted_when_the_caller_configured_none() -> None:
    """Omit-when-absent, the rule the replay key and the runtime-config hash both hold."""
    assert "generation" not in _recorded_model(ModelConfig())

    configured = ModelConfig(generation=GenerationConfig(temperature=0.5))
    assert _recorded_model(configured)["generation"]["temperature"] == 0.5


def test_the_recorded_context_carries_the_lineage_the_kernel_writes_into_attributes() -> None:
    """``attributes`` is caller data, and one of the callers is AgentLoop.

    Every subagent call gets ``root_run_id`` / ``parent_run_id`` / ``parent_task_id`` /
    ``subagent_definition_id`` / ``subagent_depth`` in this map. Dropping it as "open caller data"
    would lose subagent lineage from the artifact whose job is lineage.
    """
    context = InvocationContext(
        run_id="run-1_sub_2",
        attributes={"parent_run_id": "run-1", "subagent_depth": "1"},
    )

    assert _recorded_context(context)["attributes"] == {
        "parent_run_id": "run-1",
        "subagent_depth": "1",
    }


# --- writer and schema agree, in both directions -------------------------------------------


def test_the_writer_and_the_schema_declare_the_same_keys() -> None:
    """``additionalProperties: False`` makes a writer-only key a validation failure, not an extra.

    Checked both ways. A schema key the writer never emits is a required field that fails every
    real record; a writer key the schema does not declare fails every record too, just later.

    ``required`` is two explicit keys short of ``properties``: ``validate_run_dir`` sweeps run
    directories that earlier writers filled, and requiring ``attempt_log`` would fail every ledger
    written before that field existed, just as requiring ``idempotency_key`` would fail every
    line a pre-W7-3 v0.21 build wrote. The writer still always emits both -- the ``set(record)``
    equality above is the writer-side pin -- so absence keeps meaning exactly one thing, a
    writer that predates the field. The optional set is pinned exactly: a future key cannot
    slip into it without arguing with this test.
    """
    record = _record()

    assert set(record) == set(MODEL_CALLS_RECORD_SCHEMA["properties"])
    assert set(MODEL_CALLS_RECORD_SCHEMA["properties"]) - set(
        MODEL_CALLS_RECORD_SCHEMA["required"]
    ) == {"attempt_log", "idempotency_key"}
    assert _errors(record) == []
    assert record["schema_version"] == MODEL_CALLS_SCHEMA_VERSION
    assert record["kind"] == MODEL_CALL_KIND


@pytest.mark.parametrize(
    "key",
    [
        pytest.param("", id="unkeyed"),
        pytest.param("idem_0123456789abcdef", id="minted-shape"),
        pytest.param("A", id="single-character"),
        pytest.param("A" * 128, id="at-the-bound"),
    ],
)
def test_the_ledger_accepts_a_key_the_kernel_could_have_issued(key: str) -> None:
    """Empty included, and admitted explicitly rather than by leaving the field unconstrained:
    a refused call was never keyed, which is the same reason ``prompt_digest``'s pattern spells
    ``^(|...)$`` instead of demanding a digest."""

    assert _errors(_record(idempotency_key=key)) == []


@pytest.mark.parametrize(
    "key",
    [
        pytest.param("ok\r\n X-Injected: yes", id="obs-fold"),
        pytest.param("ok\nforged", id="lf"),
        pytest.param("A" * 129, id="over-the-bound"),
        pytest.param("-leading-punctuation", id="bad-first-character"),
        pytest.param("key with spaces", id="space"),
        pytest.param("k\u00e9", id="non-ascii"),
        pytest.param("...", id="punctuation-only"),
    ],
)
def test_the_ledger_refuses_a_key_no_issuer_could_have_minted(key: str) -> None:
    """``monoid validate`` certifies imported and third-party directories, so a line whose key
    the rest of the kernel would refuse must not be certified here either. Before this pattern
    the field was an open string and every one of these validated clean."""

    assert _errors(_record(idempotency_key=key))


def test_the_ledgers_key_pattern_is_derived_from_the_rule_the_kernel_enforces() -> None:
    """A retyped twin regex is a drift waiting to happen, and this is the check that refuses it.

    ``core`` cannot import ``providers``, so the rule lives in ``core/model_io.py`` and both
    enforcers read it: the schema states it to ``monoid validate``, and
    ``is_valid_idempotency_key`` states it to a request being built. Verified as agreement on
    behaviour and not merely as a shared substring, because two spellings of "the same" regex
    can still disagree at the edges.

    The candidates are a *product*, not a list, and that is the correction this census needed.
    Its first form checked ten strings chosen by hand, and none of them ended in a newline -- so
    it certified agreement while the schema's ``$`` was matching just before a trailing newline
    under ``jsonschema``'s Python engine and the kernel's ``\\Z`` was not. A hand-picked lattice
    proves agreement on the disagreements its author already imagined; the suffix axis exists so
    the edge no one pictured is generated anyway.
    """

    from monoid_agent_kernel.core.model_io import (
        IDEMPOTENCY_KEY_JSON_PATTERN,
        is_valid_idempotency_key,
    )

    assert MODEL_CALLS_RECORD_SCHEMA["properties"]["idempotency_key"] == {
        "type": "string",
        "pattern": IDEMPOTENCY_KEY_JSON_PATTERN,
    }

    bodies = (
        "",
        "idem_0123456789abcdef",
        "A",
        "A" * 128,
        "A" * 129,
        "-leading",
        "has space",
        "k\u00e9",
        "...",
        "ok\r\n folded",
    )
    suffixes = ("", "\n", "\r", "\r\n", "\n\n", "\nX", "\t", " ", "\u2028", "\u0085", "\x00")

    for body in bodies:
        for suffix in suffixes:
            candidate = body + suffix
            schema_accepts = not _errors(_record(idempotency_key=candidate))
            rule_accepts = candidate == "" or is_valid_idempotency_key(candidate)
            assert schema_accepts is rule_accepts, {
                "candidate": candidate,
                "schema_accepts": schema_accepts,
                "rule_accepts": rule_accepts,
                "hint": "the ledger and the kernel disagree about what a key may be",
            }


def test_the_schema_advertises_one_namespace_because_the_ledger_has_only_ever_had_one() -> None:
    """``schema_version_property`` would emit the legacy namespace too, and this artifact never
    existed under it. The registry pin refuses the mismatch; this states the reason."""

    assert MODEL_CALLS_RECORD_SCHEMA["properties"]["schema_version"] == {
        "enum": ["monoid.model-calls.v1"]
    }


@pytest.mark.parametrize(
    "mutation",
    [
        {"digest_status": "okay"},
        {"destination_status": "somewhere"},
        {"call_index": -1},
        {"attempts": -1},
        {"recorded_at": "2026-08-06T00:00:00"},
        {"usage": {"input_tokens": -1}},
        {"idempotency_key": None},
        {"unexpected": True},
    ],
)
def test_a_malformed_record_is_refused_by_the_schema(mutation: dict[str, object]) -> None:
    assert _errors({**_record(), **mutation})


def test_the_attempt_log_rides_the_record_and_legacy_lines_stay_valid() -> None:
    """The record carries the receipt's per-dispatch log; a line written before the field
    existed carries no key and stays valid -- the sweep validator reads directories that
    v0.20 writers filled, and a required key there would fail every one of them.

    The same rule holds one level down since W7-2: an entry written before ``backoff_ms``
    existed carries ten keys and stays valid, because requiring the eleventh would fail every
    ledger the W7-1 writer filled -- while a writer that knows its wait records it and the
    schema admits it (``additionalProperties: false`` would refuse an undeclared key)."""
    from monoid_agent_kernel.core.model_io import ModelCallAttempt

    # Two dispatches, because a recorded wait can only ride the entry it delayed: the first
    # dispatch has nothing before it, so its wait is 0 and the 7ms belongs to the retry.
    first = ModelCallAttempt(index=1, elapsed_ms=5, usage={"input_tokens": 12}, backoff_ms=0)
    entry = ModelCallAttempt(
        index=2,
        elapsed_ms=30,
        usage={"output_tokens": 3},
        stream_committed=True,
        backoff_ms=7,
    )
    record = _record(attempts=2, attempt_log=(first, entry))

    assert record["attempt_log"] == [first.to_json(), entry.to_json()]
    assert record["attempt_log"][1]["backoff_ms"] == 7
    assert _errors(record) == []

    ten_key_entry = {name: value for name, value in entry.to_json().items() if name != "backoff_ms"}
    assert _errors({**_record(), "attempts": 1, "attempt_log": [ten_key_entry]}) == []

    legacy = _record()
    del legacy["attempt_log"]
    assert _errors(legacy) == []


@pytest.mark.parametrize(
    "attempt_log",
    [
        "two",
        [{"index": 1}],
        [
            {
                "index": 0,
                "elapsed_ms": 0,
                "error_code": "",
                "provider_error_code": "",
                "retryable": False,
                "config_recoverable": False,
                "http_status": None,
                "provider_retried": False,
                "usage": {},
                "stream_committed": False,
            }
        ],
        [
            {
                "index": 1,
                "elapsed_ms": 0,
                "error_code": "",
                "provider_error_code": "",
                "retryable": False,
                "config_recoverable": False,
                "http_status": None,
                "provider_retried": False,
                "usage": {"output_tokens": -1},
                "stream_committed": False,
            }
        ],
        [
            {
                "index": 1,
                "elapsed_ms": 0,
                "error_code": "",
                "provider_error_code": "",
                "retryable": False,
                "config_recoverable": False,
                "http_status": None,
                "provider_retried": False,
                "usage": {},
                "stream_committed": False,
                "unexpected": True,
            }
        ],
        [{**_ATTEMPT, "backoff_ms": -1}],
        [{**_ATTEMPT, "backoff_ms": None}],
        [{**_ATTEMPT, "backoff_ms": "3"}],
    ],
)
def test_a_malformed_attempt_log_is_refused_by_the_schema(attempt_log: object) -> None:
    """An entry is written whole or not at all: a partial one, a zero index, a negative count,
    or a stray key is a writer bug the schema refuses -- the same closed-shape rule the record
    itself follows."""

    assert _errors({**_record(), "attempt_log": attempt_log})


def test_an_empty_digest_is_a_valid_record_because_a_status_explains_it() -> None:
    """No key is a real outcome -- refused before dispatch, over the cap, unencodable -- and
    ``digest_status`` names which. A schema requiring 64 hex characters would refuse the record
    that documents the refusal."""

    keyless = _record(prompt_digest="", request_digest="", digest_status="absent")

    assert _errors(keyless) == []


def test_validate_run_dir_treats_model_calls_as_optional(tmp_path: Path) -> None:
    issues = validate_run_dir(tmp_path)

    assert not any(issue.path.startswith(MODEL_CALLS_FILENAME) for issue in issues)


def test_validate_run_dir_reports_a_malformed_ledger_line(tmp_path: Path) -> None:
    (tmp_path / MODEL_CALLS_FILENAME).write_text(
        json.dumps({**_record(), "digest_status": "okay"}) + "\n",
        encoding="utf-8",
    )

    issues = validate_run_dir(tmp_path)

    assert any(issue.path.startswith(MODEL_CALLS_FILENAME) for issue in issues)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            {
                "attempts": 2,
                "attempt_log": [
                    {**_ATTEMPT, "index": 1, "usage": {"input_tokens": 12}},
                    {**_ATTEMPT, "index": 1, "usage": {"output_tokens": 3}},
                ],
            },
            "exactly once",
        ),
        (
            {
                "attempts": 1,
                "attempt_log": [{**_ATTEMPT, "index": 1, "usage": {"output_tokens": 3}}],
            },
            "sum",
        ),
        (
            {
                "latency_ms": 7,
                "attempts": 2,
                # Every entry records its wait, so this arm turns on the check existing and
                # nothing else. The legacy arm below is the one that turns on reading an
                # absent wait as zero -- kept apart so a mutant cannot satisfy both at once.
                "attempt_log": [
                    {
                        **_ATTEMPT,
                        "index": 1,
                        "elapsed_ms": 5,
                        "backoff_ms": 0,
                        "usage": {"input_tokens": 12},
                    },
                    {
                        **_ATTEMPT,
                        "index": 2,
                        "elapsed_ms": 3,
                        "backoff_ms": 40,
                        "usage": {"output_tokens": 3},
                    },
                ],
            },
            "latency_ms",
        ),
        (
            # The same claim on a line with no `backoff_ms` anywhere: a W7-1 writer's shape,
            # where the dispatches alone already outlast the call. Absent waits count as zero
            # rather than as unknown, so the check still reaches a legacy line -- what it cannot
            # do is accuse one of a wait it never recorded.
            {
                "latency_ms": 7,
                "attempts": 2,
                "attempt_log": [
                    {**_ATTEMPT, "index": 1, "elapsed_ms": 5, "usage": {"input_tokens": 12}},
                    {**_ATTEMPT, "index": 2, "elapsed_ms": 9, "usage": {"output_tokens": 3}},
                ],
            },
            "latency_ms",
        ),
        (
            # A wait recorded before the dispatch that had nothing before it. The totals fit
            # inside `latency_ms`, so the timeline claim above has nothing to say about this
            # line -- it is well-formed in every way except that no runner writes it.
            {
                "attempts": 2,
                "attempt_log": [
                    {
                        **_ATTEMPT,
                        "index": 1,
                        "elapsed_ms": 5,
                        "backoff_ms": 3,
                        "usage": {"input_tokens": 12},
                    },
                    {
                        **_ATTEMPT,
                        "index": 2,
                        "elapsed_ms": 3,
                        "backoff_ms": 0,
                        "usage": {"output_tokens": 3},
                    },
                ],
            },
            "first",
        ),
    ],
    ids=["indices", "usage", "timeline", "timeline_legacy", "first_backoff"],
)
def test_validate_run_dir_reports_an_attempt_log_that_contradicts_its_own_line(
    tmp_path: Path, mutation: dict[str, object], message: str
) -> None:
    """The four cross-entry invariants, refused on the sweep as well.

    A JSON Schema validates each entry against its own shape and cannot relate one entry to
    another, so `attempts: 2` under indices `[1, 1]`, entries billing 3 beside a receipt billing
    15, dispatches that outlast the call that made them, and a wait booked before the first
    dispatch are all structurally perfect lines. Nothing constructs a receipt on the way through
    ``monoid validate``, which reads the ledger as JSON, so a directory the record could not have
    produced was reported clean -- the one answer a validator must never give.

    Three of the four the record also refuses at construction. The timeline one is this surface's
    alone, and deliberately: ``attempt_log`` is attached while ``latency_ms`` is still its default, because
    ``_publish`` stamps the measured duration afterwards on every exit (``model_call.py`` builds
    the log at the failure and answering exits, then times the receipt inside ``_publish``). A
    constructor check would therefore fire on every retried call, comparing real dispatch
    durations against a latency of zero. The ledger line is the first place both facts are
    settled and present, which makes it the first place the claim can be checked at all.

    Semantic passes are what ``validate_run_dir`` already does for the surfaces that have
    cross-record claims -- manifest against workspace index, proposal against its hashes,
    settled text against its digests, payloads against their keys. The ledger had none because
    until this field it made no claim spanning two values; now it makes three.
    """

    (tmp_path / MODEL_CALLS_FILENAME).write_text(
        json.dumps({**_record(), **mutation}) + "\n", encoding="utf-8"
    )

    issues = validate_run_dir(tmp_path)

    assert [issue for issue in issues if issue.path.startswith(MODEL_CALLS_FILENAME)], (
        f"expected a {message!r} issue, got {issues}"
    )
    assert any(
        message in issue.message for issue in issues if issue.path.startswith(MODEL_CALLS_FILENAME)
    ), [issue.message for issue in issues]


def test_validate_run_dir_accepts_a_ledger_line_whose_log_adds_up(tmp_path: Path) -> None:
    """The other side of the pin: the writer's own output passes the sweep.

    A semantic check that refused a legitimate line would be worse than none, and the record
    the runner actually writes is the one case that must never be flagged -- entries indexed
    ``1..attempts`` whose usage sums to the receipt's, which is what ``__post_init__`` already
    guarantees at construction.
    """
    from monoid_agent_kernel.core.model_io import ModelCallAttempt

    # Entry 1 predates `backoff_ms` and omits the key; entry 2 records a 30ms wait. Their
    # dispatches and that wait total 38ms under the record's 42ms, which is the shape the
    # runner's own clock guarantees -- and an absent wait must not be read as anything but
    # a wait this line cannot report.
    line = _record(
        attempts=2,
        attempt_log=(
            ModelCallAttempt(index=1, elapsed_ms=5, usage={"input_tokens": 12}),
            ModelCallAttempt(index=2, elapsed_ms=3, backoff_ms=30, usage={"output_tokens": 3}),
        ),
    )
    (tmp_path / MODEL_CALLS_FILENAME).write_text(json.dumps(line) + "\n", encoding="utf-8")

    issues = validate_run_dir(tmp_path)

    assert not [issue for issue in issues if issue.path.startswith(MODEL_CALLS_FILENAME)]

    # And a line written before the field existed still sweeps clean: no key, no claim to check.
    legacy = _record()
    del legacy["attempt_log"]
    (tmp_path / MODEL_CALLS_FILENAME).write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    assert not [
        issue for issue in validate_run_dir(tmp_path) if issue.path.startswith(MODEL_CALLS_FILENAME)
    ]


# --- the projection reflects over nothing ---------------------------------------------------
#
# The one claim no behavioural test can express. The tests above say which fields are recorded;
# they cannot say *how* the projection decided, and a reflective implementation would satisfy all
# of them while re-opening the hazard the hand-listing closed -- a field added to `ModelConfig`
# for an unrelated reason silently joining the audit record, `gateway_url` first among them.


_PROJECTION_ALLOWLIST = {
    # the model block
    "provider",
    "model",
    "reasoning",
    "generation",
    "effort",
    "summary",
    "on_unsupported",
    "temperature",
    "top_p",
    "max_output_tokens",
    "is_default",
    # the context block
    "run_id",
    "skill_id",
    "skill_digest",
    "step_id",
    "attempt",
    "batch_id",
    "item_id",
    "case_id",
    "traceparent",
    "tracestate",
    "attributes",
    # the receipt's own recorded fields
    "context",
    "provider_name",
    "prompt_digest",
    "request_digest",
    "digest_generation",
    "digest_status",
    "destination_status",
    "stop_reason",
    "usage",
    "latency_ms",
    "attempts",
    "provider_retried",
    "error_code",
    "provider_error_code",
    "retryable",
    "config_recoverable",
    "http_status",
    "capture_downgrades",
    # W7-3: recorded as issued, not as sent -- the writer docstring fixes the meaning.
    "idempotency_key",
    # the attempt-log block (W7-1): the taxonomy names above are shared with the receipt's
    # own recorded fields; these four are the entry's alone. `backoff_ms` joined in W7-2,
    # deliberately -- the wait the kernel imposed before that dispatch.
    "attempt_log",
    "index",
    "elapsed_ms",
    "stream_committed",
    "backoff_ms",
}


def _assert_the_projection_reflects_over_nothing(source: str) -> None:
    tree = ast.parse(textwrap.dedent(source))

    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    } | {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    reflective = called & {"to_json", "asdict", "vars", "fields", "getattr", "dir"}
    assert reflective == set(), {
        "reflective_calls": sorted(reflective),
        "hint": "a serialized or enumerated config makes every field an author of the audit record",
    }

    read = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
    }
    assert read <= _PROJECTION_ALLOWLIST, {
        "outside_the_allowlist": sorted(read - _PROJECTION_ALLOWLIST),
        "hint": "a field the ledger records must be listed here, deliberately",
    }


@pytest.mark.parametrize(
    "projection",
    [model_call_record, _recorded_model, _recorded_context, _recorded_attempt],
    ids=["record", "model", "context", "attempt"],
)
def test_the_recorded_call_projection_is_hand_listed(projection: object) -> None:
    _assert_the_projection_reflects_over_nothing(inspect.getsource(projection))  # type: ignore[arg-type]


def test_the_projection_carries_the_wait_only_when_the_entry_knows_it() -> None:
    """`backoff_ms` on the artifact mirrors the entry's two writer generations: a measured wait
    is recorded, and an entry parsed from a pre-W7-2 line (None) omits the key. Emitting null
    would manufacture a value no writer ever wrote -- the exact corrupt shape the null-refusal
    rule exists to catch -- and emitting 0 would claim a measurement that never happened."""

    from monoid_agent_kernel.core.model_io import ModelCallAttempt

    measured = _recorded_attempt(ModelCallAttempt(index=1, backoff_ms=7))
    assert measured["backoff_ms"] == 7

    legacy = _recorded_attempt(ModelCallAttempt(index=1))
    assert "backoff_ms" not in legacy


def test_the_projection_pin_moves_on_the_edits_it_claims_to_catch() -> None:
    """A structural pin is a claim about which edits turn it red, and that claim is testable."""

    source = inspect.getsource(_recorded_model)
    _assert_the_projection_reflects_over_nothing(source)

    def _mutant(old: str, new: str) -> str:
        mutated = source.replace(old, new)
        assert mutated != source, {"hint": "the mutation did not apply; the anchor moved"}
        return mutated

    for old, new in (
        ('"model": model.model,', '"model": model.to_json(),'),
        ('"model": model.model,', '"gateway_url": model.gateway_url,'),
        ('"provider": model.provider,', '"timeout_s": model.timeout_s,'),
    ):
        with pytest.raises(AssertionError):
            _assert_the_projection_reflects_over_nothing(_mutant(old, new))


def test_the_ledger_does_not_share_the_replay_key_s_projection() -> None:
    """Two hand-listed projections of one config, and they must not become one function.

    Sharing looks like deduplication and is the opposite: the replay key's list answers "what was
    the provider asked for" and the ledger's answers "what was this call", so the day the ledger
    wants one more field, every recorded replay key would move -- the exact defect W6-0 closed.

    Read as code, not as text. The module docstring names ``model_call._model_identity`` on
    purpose -- documenting *why* the two are separate is the point -- and a substring scan would
    make the explanation itself the violation, which is how a rule ends up undocumented.
    """
    from monoid_agent_kernel.core import model_calls

    tree = ast.parse(inspect.getsource(model_calls))

    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    # Both homes, because the projection has moved once already: it lived in `model_call`
    # when this guard was written and now lives in `providers._request_identity`, and for the
    # length of that move `from ..._request_identity import _model_identity as _mi` satisfied
    # every assertion here -- wrong module prefix for the first, and an `ast.alias` is neither
    # a Name nor an Attribute, so the imported name never reached the second.
    # The current home is *derived*, so a third move cannot silently empty this guard the way
    # the first move emptied its predecessor. The old home stays listed by history: a ledger
    # reaching for `model_call` is reaching for the projection wherever it re-exports from.
    from monoid_agent_kernel.providers._request_identity import _model_identity

    home = inspect.getmodule(_model_identity)
    assert home is not None
    forbidden = ("monoid_agent_kernel.model_call", home.__name__)
    assert not any(name.startswith(forbidden) for name in imported), {
        "imported": sorted(imported),
        "hint": "the ledger must not reach into the replay key's module",
    }

    referenced = (
        {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        | {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        | {
            name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
            for name in (alias.name, alias.asname)
            if name
        }
    )
    assert "_model_identity" not in referenced
