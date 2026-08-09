"""ModelCallReceipt: what happened on one model call, without any of what was said."""

from __future__ import annotations

import inspect
import json
from http import HTTPStatus
from dataclasses import fields, replace
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints

import pytest

from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.model_io import (
    DESTINATION_STATUSES,
    DIGEST_STATUSES,
    ModelCallAttempt,
    ModelCallReceipt,
)
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.core.wire_validation import WireValidationError
from monoid_agent_kernel.errors import ModelAdapterError, RunCancelled

TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def test_a_receipt_carries_no_content() -> None:
    """The property that makes a receipt safe for every consumer regardless of capture policy.

    Asserted structurally rather than by inspecting one instance: any field that could hold prompt or
    response text would be a disclosure channel that bypasses `CapturePolicy` entirely.
    """
    fields = set(ModelCallReceipt().to_json())

    assert not fields & {"prompt", "messages", "final_text", "reasoning_text", "content", "response"}


def test_defaults_describe_a_successful_single_attempt_call() -> None:
    receipt = ModelCallReceipt()

    assert receipt.succeeded is True
    assert receipt.attempts == 1
    assert receipt.provider_retried is False
    assert receipt.capture_downgrades == 0
    assert dict(receipt.usage) == {}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"attempts": -1}, "attempts must not be negative"),
        ({"latency_ms": -1}, "latency_ms must not be negative"),
        ({"capture_downgrades": -1}, "capture_downgrades must not be negative"),
    ],
)
def test_impossible_counts_are_rejected(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ModelCallReceipt(**kwargs)  # type: ignore[arg-type]


def test_zero_attempts_is_a_real_count_not_an_impossible_one() -> None:
    """A call refused before the adapter is reached made no adapter call, and says so.

    `attempts` used to be required to be at least 1, which forced a refused call -- a run already
    cancelled or past its deadline when the call was requested -- to claim one adapter call. A
    consumer summing the field then counted provider work that provably never happened. Dropping the
    receipt instead was the other option and the wrong one: a refused call is part of the audit
    trail, which is exactly why one is written for it.
    """
    refused = ModelCallReceipt(attempts=0, error_code="cancelled")
    assert refused.attempts == 0
    assert refused.succeeded is False
    # And it survives the wire, since an audit record that cannot be read back is not one.
    assert ModelCallReceipt.from_json(refused.to_json()).attempts == 0
    # A payload that never mentions the field still reads as one call, for older records.
    assert ModelCallReceipt.from_json({}).attempts == 1


def test_usage_must_be_whole_token_counts() -> None:
    with pytest.raises(WireValidationError, match="mapping of str to int"):
        ModelCallReceipt(usage={"input_tokens": 1.5})  # type: ignore[dict-item]
    # ``bool`` is an ``int`` subclass, and a boolean token count is a bug, not a count of one.
    with pytest.raises(WireValidationError, match="mapping of str to int"):
        ModelCallReceipt(usage={"input_tokens": True})


def test_usage_counts_must_not_be_negative() -> None:
    """The sign check the other three counters already had. Usage is the one that gets *summed*, so
    a negative slipping through subtracts from an aggregate instead of failing visibly."""
    with pytest.raises(WireValidationError, match="'input_tokens' must not be negative"):
        ModelCallReceipt(usage={"input_tokens": -100})
    with pytest.raises(WireValidationError, match="'output_tokens' must not be negative"):
        ModelCallReceipt.from_json({"usage": {"input_tokens": 5, "output_tokens": -1}})

    # Zero is a real count -- a cache-hit call reports it -- so "reject anything not positive" is
    # not the rule and does not pass here.
    assert dict(ModelCallReceipt(usage={"input_tokens": 0, "output_tokens": 7}).usage) == {
        "input_tokens": 0,
        "output_tokens": 7,
    }


def test_usage_is_copied_away_from_the_caller() -> None:
    supplied = {"input_tokens": 5}
    receipt = ModelCallReceipt(usage=supplied)

    supplied["input_tokens"] = 999

    assert dict(receipt.usage) == {"input_tokens": 5}


def test_attempts_and_provider_retried_are_independent_facts() -> None:
    """A gateway can retry three times inside one call the kernel counts as one attempt.

    A receipt with only ``attempts`` would report that as a clean single call, which is exactly the
    thing an operator investigating latency needs to see.
    """
    internal = ModelCallReceipt(attempts=1, provider_retried=True)
    kernel_level = ModelCallReceipt(attempts=3, provider_retried=False)

    assert (internal.attempts, internal.provider_retried) == (1, True)
    assert (kernel_level.attempts, kernel_level.provider_retried) == (3, False)


def test_trace_ids_come_from_the_invocation_context() -> None:
    """One source of truth: the context is the call's own span, so the receipt does not restate it."""
    receipt = ModelCallReceipt(context=InvocationContext(traceparent=TRACEPARENT))

    assert receipt.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert receipt.span_id == "00f067aa0ba902b7"


# --- with_error -----------------------------------------------------------------------------


def test_with_error_takes_the_taxonomy_a_model_error_already_classified() -> None:
    """The providers raise a classified error, so the runner must not re-derive any of this."""
    receipt = ModelCallReceipt().with_error(
        ModelAdapterError(
            "upstream rejected the request",
            provider_error_code="rate_limit_exceeded",
            retryable=True,
            http_status=429,
        )
    )

    assert receipt.succeeded is False
    assert receipt.error_code == "model_error"
    assert receipt.provider_error_code == "rate_limit_exceeded"
    assert receipt.retryable is True
    assert receipt.http_status == 429


def test_with_error_records_that_a_failure_was_config_fixable() -> None:
    """The sixth fact. ``retryable`` says waiting may help; this one says a config change will.

    A receipt that carried only ``retryable`` could not tell an auditor why an exhausted retry
    budget was never going to succeed — the two are independent, and a config-fixable failure is
    usually the one that is NOT retryable.
    """
    receipt = ModelCallReceipt().with_error(
        ModelAdapterError(
            "the configured model is not available to this account",
            provider_error_code="model_not_found",
            retryable=False,
            config_recoverable=True,
            http_status=404,
        )
    )

    assert receipt.retryable is False
    assert receipt.config_recoverable is True
    assert receipt.to_json()["config_recoverable"] is True
    # An exception that does not classify itself reads as False rather than raising.
    assert ModelCallReceipt().with_error(ValueError("boom")).config_recoverable is False


def test_with_error_records_an_unclassified_exception_by_type_not_message() -> None:
    """An arbitrary exception's message can carry request content, and the receipt holds none."""
    receipt = ModelCallReceipt().with_error(ValueError("prompt was: my social security number is"))

    assert receipt.error_code == "ValueError"
    assert "social security" not in json.dumps(receipt.to_json())


def test_with_error_keeps_a_kernel_error_code() -> None:
    receipt = ModelCallReceipt().with_error(RunCancelled("cancelled"))

    assert receipt.error_code == RunCancelled.error_code
    assert receipt.retryable is False


def test_with_error_preserves_everything_else() -> None:
    original = ModelCallReceipt(
        prompt_digest="sha-prompt",
        usage={"input_tokens": 5},
        latency_ms=120,
        attempts=2,
    )

    failed = original.with_error(ModelAdapterError("boom"))

    assert failed.prompt_digest == "sha-prompt"
    assert dict(failed.usage) == {"input_tokens": 5}
    assert (failed.latency_ms, failed.attempts) == (120, 2)
    # A failed call still gets its own receipt rather than mutating the one it came from.
    assert original.succeeded is True


# --- serialization --------------------------------------------------------------------------


def test_a_too_large_receipt_survives_the_round_trip() -> None:
    """Self-evidencing red for the fifth digest status. The vocabulary pins beside this file are
    construction-enumerated -- they iterate the tuples -- so a member added to `DIGEST_STATUSES`
    passes them without any writer ever producing it. This is the write that proves the value is
    accepted on both sides of the wire."""
    receipt = ModelCallReceipt(digest_status="too_large")

    restored = ModelCallReceipt.from_json(json.loads(json.dumps(receipt.to_json())))

    assert restored.digest_status == "too_large"
    assert restored.request_digest == ""


def test_json_round_trip_preserves_every_field() -> None:
    """Every field, enumerated by construction -- so a new one that is not added here leaves the
    test green while its name becomes a lie. The guard below turns that failure mode into a
    failing test."""

    receipt = ModelCallReceipt(
        context=InvocationContext(run_id="run_1", skill_id="lecture-note", traceparent=TRACEPARENT),
        model=ModelConfig(provider="openai", model="gpt-5.5", timeout_s=42),
        provider_name="openai",
        prompt_digest="sha-prompt",
        request_digest="sha-request",
        stop_reason="tool_calls",
        usage={"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
        latency_ms=1234,
        attempts=2,
        provider_retried=True,
        error_code="model_error",
        provider_error_code="rate_limit_exceeded",
        retryable=True,
        config_recoverable=True,
        http_status=429,
        redaction_digest="sha-policy",
        capture_downgrades=1,
        digest_generation="monoid.model-request-digest.v1",
        digest_status="ok",
        destination_status="resolved",
        destination_digest="sha-destination",
        attempt_log=(
            ModelCallAttempt(
                index=1,
                elapsed_ms=800,
                error_code="model_error",
                provider_error_code="rate_limit_exceeded",
                retryable=True,
                config_recoverable=False,
                http_status=429,
                provider_retried=True,
                usage={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                stream_committed=False,
            ),
            ModelCallAttempt(
                index=2,
                elapsed_ms=434,
                usage={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
                stream_committed=True,
            ),
        ),
    )

    assert ModelCallReceipt.from_json(json.loads(json.dumps(receipt.to_json()))) == receipt


def test_the_round_trip_above_names_every_field_there_is() -> None:
    """The enumeration is only a contract while it is complete.

    `test_json_round_trip_preserves_every_field` builds a receipt by construction, so a field
    added without touching it round-trips at its default and the test passes without covering it.
    This reads the dataclass instead: every declared field must appear in the wire shape, and the
    round trip above must set each one to something other than its default.
    """

    declared = {field_.name for field_ in fields(ModelCallReceipt)}
    assert set(ModelCallReceipt().to_json()) == declared

    source = inspect.getsource(test_json_round_trip_preserves_every_field)
    missing = sorted(name for name in declared if f"{name}=" not in source)
    assert missing == [], {
        "never_exercised_by_the_round_trip": missing,
        "hint": "a field the enumeration does not name is a field it does not cover",
    }


def test_an_attempt_entry_round_trips_and_refuses_impossible_counts() -> None:
    """One kernel dispatch as the log records it: same metadata-only rule, same usage
    validation, same error_code-empty success convention as the receipt that carries it."""

    entry = ModelCallAttempt(
        index=2,
        elapsed_ms=125,
        error_code="model_error",
        provider_error_code="overloaded",
        retryable=True,
        config_recoverable=False,
        http_status=529,
        provider_retried=True,
        usage={"total_tokens": 5},
        stream_committed=True,
    )

    assert ModelCallAttempt.from_json(json.loads(json.dumps(entry.to_json()))) == entry
    assert entry.succeeded is False
    assert ModelCallAttempt().succeeded is True

    with pytest.raises(ValueError, match="index"):
        ModelCallAttempt(index=0)
    with pytest.raises(ValueError, match="elapsed_ms"):
        ModelCallAttempt(elapsed_ms=-1)
    with pytest.raises(WireValidationError):
        ModelCallAttempt(usage={"total_tokens": -5})


def test_the_attempt_entry_wire_shape_names_every_field_there_is() -> None:
    """The same completeness guard the receipt's round trip carries, for the entry type:
    a field added to the dataclass without joining the wire shape reads back at its default
    and every historical log silently drops it."""

    declared = {field_.name for field_ in fields(ModelCallAttempt)}
    assert set(ModelCallAttempt().to_json()) == declared


def test_the_attempt_log_is_empty_or_names_every_attempt() -> None:
    """`len(attempt_log)` is 0 (a legacy or refused receipt) or exactly `attempts` -- a log
    naming some attempts but not others cannot answer the question it exists for, and a sum
    over its usage would silently disagree with the receipt's."""

    entry = ModelCallAttempt(index=1)

    assert ModelCallReceipt(attempts=1, attempt_log=(entry,)).attempt_log == (entry,)
    assert ModelCallReceipt(attempts=0).attempt_log == ()
    assert ModelCallReceipt(attempts=3).attempt_log == ()

    with pytest.raises(ValueError, match="attempt_log"):
        ModelCallReceipt(attempts=2, attempt_log=(entry,))
    with pytest.raises(WireValidationError):
        ModelCallReceipt(attempts=1, attempt_log=({"index": 1},))  # type: ignore[arg-type]


def test_the_attempt_log_names_each_dispatch_once_and_in_order() -> None:
    """"Exactly once" is what the refusal says; counting was all it did.

    A length check accepts `[1, 1]` for `attempts=2` -- the right number of entries naming the
    same dispatch twice, with dispatch two absent. A consumer of that record cannot answer the
    question the log exists for (what did attempt 2 do) and cannot tell that it is unanswerable,
    because every other field is well-formed. The indices are the record's own claim about which
    dispatch each row is, so they carry the claim: `1..attempts`, ascending, no gaps, no repeats.
    """

    def _log(*indices: int) -> tuple[ModelCallAttempt, ...]:
        return tuple(ModelCallAttempt(index=index) for index in indices)

    assert ModelCallReceipt(attempts=2, attempt_log=_log(1, 2)).attempts == 2

    for indices in ((1, 1), (2, 1), (2, 3), (1, 3)):
        with pytest.raises(ValueError, match="attempt_log"):
            ModelCallReceipt(attempts=2, attempt_log=_log(*indices))


def test_the_attempt_logs_usage_adds_up_to_the_receipts() -> None:
    """The entries are the receipt's breakdown of its bill, so they add up to it or they are not.

    A record whose entries say 3 while its total says 99 cannot be reconciled by its reader, and
    carries nothing that says which number to believe. This was the stated reason the log is
    all-or-nothing and it went unchecked, so the class documented a property it did not hold.

    The writer builds the log and the merged total in a single ``replace`` for exactly this
    reason: ``__post_init__`` re-runs on every ``replace``, so a two-step build would have to
    pass through a receipt that carries its entries beside a total they do not sum to.
    """

    def _entry(index: int, output: int) -> ModelCallAttempt:
        return ModelCallAttempt(index=index, usage={"output_tokens": output})

    ModelCallReceipt(
        attempts=2,
        attempt_log=(_entry(1, 5), _entry(2, 7)),
        usage={"output_tokens": 12},
    )

    with pytest.raises(ValueError, match="sum"):
        ModelCallReceipt(
            attempts=2,
            attempt_log=(_entry(1, 5), _entry(2, 7)),
            usage={"output_tokens": 99},
        )
    # And a key on one side only is a disagreement too: a total nobody's dispatch reported is
    # exactly the shape a reader cannot attribute.
    with pytest.raises(ValueError, match="sum"):
        ModelCallReceipt(
            attempts=1,
            attempt_log=(_entry(1, 5),),
            usage={"output_tokens": 5, "input_tokens": 4},
        )


def test_a_legacy_receipt_without_an_attempt_log_still_reads() -> None:
    """Absent on every receipt written before the field existed, which is legal and reads as
    an empty log beside an intact `attempts` count; present-but-mistyped is refused, like
    every other field here."""

    payload = ModelCallReceipt(attempts=3).to_json()
    del payload["attempt_log"]

    restored = ModelCallReceipt.from_json(payload)

    assert restored.attempt_log == ()
    assert restored.attempts == 3

    with pytest.raises(WireValidationError):
        ModelCallReceipt.from_json({**payload, "attempt_log": "two"})


def test_an_attempt_entry_is_read_whole_or_refused() -> None:
    """The closed shape the schema already declares, enforced by the reader that builds the record.

    `attempt_log` is all-or-nothing one level up because absence there means one thing -- a writer
    that predates the field. An *entry* has no such predecessor: it arrived whole or it did not
    arrive, and the ledger schema says so by requiring all ten keys. The reader defaulted every
    one of them, so `{}` deserialized into a plausible lie -- a successful, zero-duration,
    unbilled dispatch numbered 1 -- and `attempts=1` beside it satisfied both cross-entry
    invariants. A corrupt audit record that reads as data is worse than one that fails to read.

    Enumerated from the dataclass rather than listed here, so a field added later is covered by
    this test on the day it is added rather than on the day someone remembers to extend a list.
    """

    from dataclasses import fields as dataclass_fields

    whole = ModelCallAttempt(index=1).to_json()
    declared = {field.name for field in dataclass_fields(ModelCallAttempt)}

    assert set(whole) == declared, {"projection_and_fields_disagree": set(whole) ^ declared}
    assert ModelCallAttempt.from_json(whole) == ModelCallAttempt(index=1)

    for key in sorted(declared):
        partial = {name: value for name, value in whole.items() if name != key}
        with pytest.raises(WireValidationError, match=key):
            ModelCallAttempt.from_json(partial)


def test_with_error_never_fails_the_call_it_is_reporting() -> None:
    """A receipt already holding its breakdown keeps its total, rather than raising over it.

    ``with_error`` exists to mark a receipt failed, and every read inside it is guarded for one
    reason: a surface that describes a failure must not be able to *replace* that failure with an
    error of its own. The usage-sum invariant put a way to do exactly that back in -- the method
    overwrites ``usage`` from the exception's stamp and does not touch ``attempt_log``, so calling
    it on a settled receipt that carries entries raised ``ValueError`` out of the reporting path.

    Not reached from ``src`` today (the runner probes a fresh receipt and merges in one
    ``replace``), which is what makes it worth pinning: the safety is an ordering the runner
    happens to have, and both the class and this method are public.

    The entries are the receipt's own per-dispatch breakdown; a total read off an exception cannot
    restate them, so the existing "a receipt that already carried counts keeps them" rule extends
    to the receipt that already carried rows.
    """

    from monoid_agent_kernel.providers.base import mark_provider_usage, provider_usage_of

    settled = ModelCallReceipt(attempts=1, attempt_log=(ModelCallAttempt(index=1),))
    later = RuntimeError("failed after the log was built")
    mark_provider_usage(later, {"input_tokens": 4})

    failed = settled.with_error(later)

    assert failed.error_code == "RuntimeError"
    assert failed.usage == {}
    assert failed.attempt_log == settled.attempt_log
    # The stamp is not lost, only not written over a breakdown that would contradict it: it is
    # still on the exception, which is where every reader of it looks.
    assert provider_usage_of(later) == {"input_tokens": 4}

    # A log-less receipt still adopts the stamp -- the path the runner actually takes.
    adopted = ModelCallReceipt().with_error(later)
    assert adopted.usage == {"input_tokens": 4}


def test_a_receipt_that_never_reached_the_probe_says_so() -> None:
    """The default is a fact, not a blank.

    A call refused before dispatch -- cancelled, past its deadline, rejected at ingress -- gets a
    receipt built before any key or probe happened. Defaulting those fields to `""` would have
    reported "no destination" and "no key" for a call that never asked either question.
    """

    provisional = ModelCallReceipt()

    assert provisional.digest_status == "not_reached"
    assert provisional.destination_status == "not_reached"
    assert provisional.request_digest == ""
    assert provisional.digest_generation == ""


@pytest.mark.parametrize(
    ("status_field", "witness_field", "witnessed"),
    (
        ("digest_status", "request_digest", "ok"),
        ("destination_status", "destination_digest", "resolved"),
    ),
)
def test_an_absent_status_never_contradicts_the_digest_it_describes(
    status_field: str, witness_field: str, witnessed: str
) -> None:
    """A receipt written before these fields existed keeps the answer it already recorded.

    `not_reached` claims the call was refused *before* the value was ever computed. Read over a
    payload that carries the value, that is not a cautious default -- it is a record denying its own
    contents, and `to_json` writes the denial back, so the first read/write makes it permanent and a
    consumer that asks the status whether a replay key exists throws away a real one.

    Both pairs, because the reader is one rule and the shape is one shape: a digest that is there is
    proof of the only outcome that produces it. `destination_digest` is non-empty on the `resolved`
    arm alone -- `_resolved_destination` answers `""` for the other three -- so the same inference
    is available and the same contradiction is manufacturable.
    """

    legacy = ModelCallReceipt().to_json()
    for field_name in ("digest_status", "digest_generation", "destination_status"):
        del legacy[field_name]
    del legacy["destination_digest"]
    legacy[witness_field] = "sha-witness"

    parsed = ModelCallReceipt.from_json(legacy)

    assert getattr(parsed, status_field) == witnessed
    assert getattr(parsed, witness_field) == "sha-witness"
    # Inferred from silence, never over a statement: a writer that says `not_reached` beside a
    # digest is reporting a bug, and rewriting it to `ok` would hide the writer that has one.
    assert (
        getattr(
            ModelCallReceipt.from_json(legacy | {status_field: "not_reached"}),
            status_field,
        )
        == "not_reached"
    )
    # No value, no claim. `absent` and `not_reached` are not distinguishable on a legacy record and
    # are not guessed at; the default stands wherever it contradicts nothing.
    assert (
        getattr(ModelCallReceipt.from_json(legacy | {witness_field: ""}), status_field)
        == "not_reached"
    )


def test_a_legacy_key_is_never_promoted_to_a_generation_it_was_not_taken_in() -> None:
    """The status says a key exists; the generation says which rules made it. Only one is inferable.

    A pre-W6-0 key was taken over a different payload -- an endpoint inside it, a serialized config
    rather than the projection -- so reading a generation onto it would hand a replay consumer a key
    it cannot reproduce. Empty is the honest answer, and it is what makes `ok` safe to infer.
    """

    legacy = ModelCallReceipt().to_json()
    for field_name in ("digest_status", "digest_generation"):
        del legacy[field_name]
    legacy["request_digest"] = "sha-request"

    parsed = ModelCallReceipt.from_json(legacy)

    assert parsed.digest_status == "ok"
    assert parsed.digest_generation == ""


@pytest.mark.parametrize(
    ("status_field", "witness_field"),
    (
        ("digest_status", "request_digest"),
        ("destination_status", "destination_digest"),
    ),
)
def test_an_explicit_null_status_is_a_malformed_value_not_an_absent_field(
    status_field: str, witness_field: str
) -> None:
    """A key that is *there* holding `null` is a corrupt record, not a legacy one.

    Every other string on this receipt already draws that line -- `parse_str` separates a missing
    key from a present one holding the wrong type, and `http_status` is nullable only because it is
    declared `int | None`. Reading `null` as absence would admit the corrupt record, hand it a
    status inferred from its digest that it never carried, and let `to_json` write that back out as
    a stated one.
    """

    stated_null = ModelCallReceipt().to_json() | {status_field: None}

    # With a digest to infer from -- the case where absence and null diverge, and where reading
    # null as absence fabricates a status rather than merely defaulting one.
    with pytest.raises(WireValidationError, match=status_field):
        ModelCallReceipt.from_json(stated_null | {witness_field: "sha-witness"})

    # And with nothing to infer from: the refusal is about the value, not about the inference.
    with pytest.raises(WireValidationError, match=status_field):
        ModelCallReceipt.from_json(stated_null)


@pytest.mark.parametrize("field_name", ("digest_status", "destination_status"))
def test_a_status_the_reader_would_refuse_cannot_be_constructed(field_name: str) -> None:
    """A receipt cannot be born unreadable by its own class.

    `from_json` refused a non-member and `to_json` emitted one, so `ModelCallReceipt(
    digest_status="okay")` wrote an audit record this same class rejects on the way back in. A
    record that can be written and not read fails in the consumer, long after the writer that
    caused it is gone -- and the writer is the only place the mistake is fixable.

    `ModelCallCapture.__post_init__` already refuses a `mode` outside `CAPTURE_MODES`; these two
    were the closed enums on this side of the file that did not. Constructor and reader now share
    one function, so the two cannot drift back apart.
    """

    with pytest.raises(WireValidationError, match=field_name):
        ModelCallReceipt(**{field_name: "okay"})

    # `replace` re-runs `__post_init__`, and it is the path the subscription narrowing and
    # `with_error` both take -- a rule the direct constructor alone enforced would miss them.
    with pytest.raises(WireValidationError, match=field_name):
        replace(ModelCallReceipt(), **{field_name: "okay"})


def test_each_status_field_admits_exactly_its_own_vocabulary() -> None:
    """The refusal must not be stricter than the enum, and must not accept the other one's.

    One helper now serves both fields, which is precisely the shape that lets a transposed
    argument pass every "a bad value is refused" test: `not_reached` is a member of both sets, so
    a swapped pair would still refuse `"okay"` and still accept the default.
    """

    for value in DIGEST_STATUSES:
        assert ModelCallReceipt(digest_status=value).digest_status == value
    for value in DESTINATION_STATUSES:
        assert ModelCallReceipt(destination_status=value).destination_status == value

    for value in set(DESTINATION_STATUSES) - set(DIGEST_STATUSES):
        with pytest.raises(WireValidationError, match="digest_status"):
            ModelCallReceipt(digest_status=value)
    for value in set(DIGEST_STATUSES) - set(DESTINATION_STATUSES):
        with pytest.raises(WireValidationError, match="destination_status"):
            ModelCallReceipt(destination_status=value)


@pytest.mark.parametrize("field_name", ("digest_status", "destination_status"))
def test_an_unknown_status_is_refused_rather_than_absorbed(field_name: str) -> None:
    """Closed kernel enums, unlike ``stop_reason``.

    ``stop_reason`` is an open string because a *provider* may invent a fifth value and a receipt
    must stay recordable. These two are written only by this kernel, so an unknown value is a bug
    in a writer -- absorbing it would let a typo travel as data.
    """

    payload = ModelCallReceipt().to_json() | {field_name: "probably"}

    with pytest.raises(WireValidationError, match=field_name):
        ModelCallReceipt.from_json(payload)


def test_an_unknown_stop_reason_survives_a_round_trip() -> None:
    """A receipt is an audit record: a provider that adds a fifth stop reason must be recordable
    without a kernel change, which is why the field is not the provider ``Literal``."""
    receipt = ModelCallReceipt(stop_reason="content_filter")

    assert ModelCallReceipt.from_json(receipt.to_json()).stop_reason == "content_filter"


def test_from_json_distinguishes_an_absent_http_status_from_zero() -> None:
    assert ModelCallReceipt.from_json({}).http_status is None
    assert ModelCallReceipt.from_json({"http_status": None}).http_status is None
    assert ModelCallReceipt.from_json({"http_status": 0}).http_status == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"context": []},
        {"context": False},
        {"context": 0},
        {"model": []},
        {"model": "gpt-5.5"},
        {"usage": 0},
    ],
)
def test_from_json_rejects_a_falsy_malformed_nested_payload(payload: dict[str, object]) -> None:
    """A falsy wrong type used to be read as "absent", so `{"context": []}` was accepted as an
    anonymous invocation and a corrupt audit record silently lost its run and trace attribution.

    `{"model": []}` additionally used to raise `AttributeError` out of `ModelConfig.from_json`, which
    assumes dict-or-None — a crash on untrusted input rather than a rejection.
    """
    with pytest.raises(WireValidationError):
        ModelCallReceipt.from_json(payload)


@pytest.mark.parametrize("key", ["context", "model", "usage"])
def test_from_json_treats_an_absent_nested_payload_as_a_default(key: str) -> None:
    """The half of the old contract that is real: a writer that predates the field.

    ``183e197`` settled this as "absent and null keep their defaults; anything present reaches
    validation", and the absent half is still exactly right -- it is what lets a receipt written
    before a field existed be read back.
    """
    assert ModelCallReceipt.from_json({}) == ModelCallReceipt()

    payload = ModelCallReceipt().to_json()
    del payload[key]

    assert ModelCallReceipt.from_json(payload) == ModelCallReceipt()


@pytest.mark.parametrize("key", ["context", "model", "usage"])
def test_from_json_refuses_a_nested_payload_that_is_present_and_null(key: str) -> None:
    """The other half of ``183e197`` is withdrawn here, and the file had already withdrawn it.

    That commit lumped ``null`` in with absent while reasoning about *falsy wrong types*
    (``{"context": []}`` read as an anonymous invocation). At the time the conflation was
    harmless: both landed on the same default, and no default meant anything beyond "empty".

    ``_parsed_status`` is where this record stopped believing that, in its own words -- "a key
    present and holding ``null`` is a corrupt record ... conflating the two was harmless while
    both landed on the default and stopped being harmless the moment absence began to infer".
    Two fields since made the defaults load-bearing in exactly that way: ``attempt_log``'s
    absence *infers* a writer predating the ledger, so a null there erases a real ledger and
    reads as legacy; and ``usage``'s default is an empty mapping that then satisfies the
    receipt's own cross-entry sum invariant, so a nulled total agrees with any breakdown.

    ``http_status`` stays nullable and is covered by its own test above, because it is declared
    ``int | None`` -- the annotation is what decides, not this list.
    """
    with pytest.raises(WireValidationError):
        ModelCallReceipt.from_json({key: None})


@pytest.mark.parametrize(
    "model",
    [
        {"reasoning": []},
        {"reasoning": "high"},
        {"retry": []},
        {"timeout_s": "soon"},
    ],
)
def test_from_json_rejects_a_malformed_nested_model_config(model: dict[str, object]) -> None:
    """Checking the outer object is not enough.

    `ModelConfig` and its nested `ReasoningConfig` / `ModelRetryConfig` are typed `dict | None` and
    trust it, so a malformed nested object reached `.get` on a list and raised `AttributeError`. A
    consumer handling corrupt audit records through the documented validation exception would crash
    instead — so this module translates at its own boundary, which is the one that made the promise.
    """
    with pytest.raises(WireValidationError, match="model must be a valid model config"):
        ModelCallReceipt.from_json({"model": model})


def test_from_json_rejects_an_oversized_numeric_exponent() -> None:
    """`OverflowError` is an `ArithmeticError`, not a `ValueError`, so it escaped the translation.

    JSON decodes `1e999` to `inf` and `int(inf)` raises it — so a corrupt audit record could crash the
    consumer whose job is to reject corrupt audit records.
    """
    payload = json.loads('{"model": {"timeout_s": 1e999}}')

    with pytest.raises(WireValidationError, match="model must be a valid model config"):
        ModelCallReceipt.from_json(payload)


def test_from_json_still_accepts_a_well_formed_nested_model_config() -> None:
    """The containment must not have been bought by rejecting valid nested configs."""
    restored = ModelCallReceipt.from_json(
        {"model": {"provider": "openai", "model": "gpt-5.5", "reasoning": {}, "retry": {}}}
    )

    assert (restored.model.provider, restored.model.model) == ("openai", "gpt-5.5")


def test_from_json_rejects_a_non_object_and_a_mistyped_field() -> None:
    with pytest.raises(WireValidationError):
        ModelCallReceipt.from_json("not-an-object")
    with pytest.raises(WireValidationError):
        ModelCallReceipt.from_json({"attempts": "2"})
    with pytest.raises(WireValidationError):
        ModelCallReceipt.from_json({"usage": "not-an-object"})
    with pytest.raises(WireValidationError):
        ModelCallReceipt.from_json({"provider_retried": "yes"})
    with pytest.raises(WireValidationError):
        ModelCallReceipt.from_json({"config_recoverable": "yes"})


def test_from_json_tolerates_a_receipt_written_before_config_recoverable_existed() -> None:
    """Absence is legal and reads False; present-but-mistyped is refused above."""

    payload = ModelCallReceipt(config_recoverable=True).to_json()
    del payload["config_recoverable"]

    assert ModelCallReceipt.from_json(payload).config_recoverable is False


def _admits(annotation: object, wanted: object) -> bool:
    """Whether ``annotation`` names ``wanted`` directly or as a union member.

    Deliberately re-derived here rather than imported from the module under test: an enumeration
    that shares its definition with the code it audits agrees with that code by construction, and
    would go green with it if the definition were the thing that broke.
    """
    if annotation is wanted:
        return True
    return get_origin(annotation) in (Union, UnionType) and wanted in get_args(annotation)


def _nullable_names(cls: type) -> set[str]:
    hints = get_type_hints(cls)
    return {entry.name for entry in fields(cls) if _admits(hints.get(entry.name), type(None))}


def _numeric_names(cls: type) -> list[str]:
    hints = get_type_hints(cls)
    return sorted(entry.name for entry in fields(cls) if _admits(hints.get(entry.name), int))


@pytest.mark.parametrize("record", [ModelCallAttempt, ModelCallReceipt], ids=["attempt", "receipt"])
def test_every_wire_field_that_is_not_nullable_refuses_an_explicit_null(record: type) -> None:
    """Derived census: "absent" and "present and null" are different answers on both records.

    ``from_json`` read a key holding ``null`` as if the key were missing, so an explicit null
    landed on the field's legacy default. On ``usage`` that produced an empty mapping which then
    *satisfied* the receipt's cross-entry sum invariant; on ``attempt_log`` it erased the whole
    per-dispatch ledger and read as a writer predating the field -- while the ledger schema
    rejects null and no writer that ever existed emitted one.

    The per-field required-key pin added for the earlier partial-entry finding was green through
    every one of these, and structurally had to be: it asks whether the key is *in* the payload,
    and a key holding null is in the payload. Presence and non-nullness are different questions,
    and only the first had an instrument.

    Enumerated over each record's own fields rather than over the two the review named, because
    the receipt's ``usage``, ``context`` and ``model`` had the identical shape and were not
    named. ``http_status`` is declared ``int | None``, so a wire null is a value there rather
    than a collapse -- read off the annotation, so a field that becomes nullable tomorrow stops
    being required here on the same day.
    """
    complete = json.loads(json.dumps(record().to_json()))
    nullable = _nullable_names(record)
    collapsed = []
    for key in sorted(complete):
        if key in nullable:
            continue
        payload = dict(complete)
        payload[key] = None
        try:
            record.from_json(payload)
        except WireValidationError:
            continue
        collapsed.append(key)

    assert collapsed == [], {
        "fields_reading_an_explicit_null_as_their_default": collapsed,
        "nullable_and_therefore_exempt": sorted(nullable),
        "hint": "absent = legacy default; present-but-null = refused",
    }


def test_with_error_still_accepts_the_int_subclass_it_documents() -> None:
    """The bool rule must not become an exact-type rule. It did, and this is that regression.

    ``with_error`` reads ``http_status`` off an arbitrary exception and guards it with
    ``isinstance(http_status, bool) or not isinstance(http_status, int)`` -- excluding ``bool``
    by name while deliberately admitting every other ``int`` subclass, because an
    ``http.HTTPStatus`` is exactly what an HTTP client hands back. The count census added in
    ``a8faa8f`` then spelled ``type(value) is not int`` and the ``replace()`` inside ``with_error``
    re-ran it, so this record refused a value its own reader had just accepted.

    The shape is worth naming: ``http_status`` was correctly exempted from the *null* rule
    (it is declared ``int | None``) and then handed the *bool* rule without anyone asking
    whether it had an acceptance of its own. One field, two rules, exempted from one and not
    the other.

    The repair narrows the predicate to the defect it was written for -- ``bool`` is refused,
    every other ``int`` is not -- rather than exempting the one field that was noticed. That
    restores the prior acceptance on all seven enumerated counts at once, which is the only
    version of this fix that cannot leave a second field broken the same way.

    ``usage`` is deliberately NOT covered by this: its own loop keeps ``type(value) is not int``
    because its four sibling readers spell the same, and an ``IntEnum`` token count accepted here
    would be dropped by every one of them.
    """
    failed = ModelCallReceipt().with_error(
        ModelAdapterError("rate limited", http_status=HTTPStatus.TOO_MANY_REQUESTS)
    )

    assert failed.http_status == 429
    # And it still leaves as a JSON integer, which is the only thing the wire cares about.
    assert json.loads(json.dumps(failed.to_json()))["http_status"] == 429


@pytest.mark.parametrize("record", [ModelCallAttempt, ModelCallReceipt], ids=["attempt", "receipt"])
def test_every_numeric_field_on_both_records_refuses_a_boolean(record: type) -> None:
    """Derived census: one predicate for every count, not a comparison written per field.

    ``__post_init__`` validated its counts by comparison alone, and ``True < 1`` is ``False`` --
    so ``ModelCallAttempt(index=True)`` was stored, and ``to_json`` emitted a JSON boolean where
    ``MODEL_CALLS_RECORD_SCHEMA`` requires an integer: a record this kernel writes and its own
    schema refuses. ``True == 1`` also satisfied the receipt's ordered-index invariant, so a
    bool-indexed entry passed the one check that exists to make the log answerable.

    The rule was already written down in this same file, one dataclass over: the usage loop on
    both records spells ``type(value) is not int`` and carries a comment explaining that ``bool``
    is an ``int`` subclass and a boolean count is a bug. It was applied to the mapping values and
    to none of the scalar counts beside them -- the rule stated on one sibling and not the other.

    Enumerated from the annotations, so the seventh count added tomorrow is covered the day it
    exists. The review named two of the seven that were open; ``http_status`` on both records was
    not validated at all.
    """
    numeric = _numeric_names(record)
    assert numeric, f"{record.__name__} has no numeric fields to enumerate"

    accepted = []
    for name in numeric:
        try:
            built = record(**{name: True})
        except WireValidationError:
            continue
        accepted.append((name, built.to_json()[name]))

    assert accepted == [], {
        "numeric_fields_that_stored_a_boolean": accepted,
        "enumerated": numeric,
        "hint": "type(value) is not int -- the rule the usage loop already spells",
    }
