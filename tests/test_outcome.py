from __future__ import annotations

from typing import get_args

import pytest

from monoid_agent_kernel.core.outcome import (
    InterruptionCause,
    RetryEligibility,
    TERMINAL_OUTCOME_SCHEMA_VERSION,
    TerminalOutcome,
    TerminalOutcomeKind,
    terminal_outcome_from_suspension,
)
from monoid_agent_kernel.core.result import Suspension


def test_terminal_outcome_round_trips_the_portable_record() -> None:
    outcome = TerminalOutcome(
        run_id="run_1",
        kind="interrupted",
        retry_eligibility="safe",
        interruption_cause="graceful_drain",
        checkpoint_seq=7,
        partial_output_ref="blob:partial",
        last_evidence_ref="invocation:call_1:3",
        error_code="worker_draining",
        http_status=503,
    )

    assert TerminalOutcome.from_json(outcome.to_json()) == outcome
    assert outcome.to_json()["schema_version"] == TERMINAL_OUTCOME_SCHEMA_VERSION


def test_terminal_outcome_reads_legacy_namespace_and_writes_canonical_namespace() -> None:
    payload = TerminalOutcome(
        run_id="run_1",
        kind="completed",
        retry_eligibility="not_applicable",
    ).to_json()
    payload["schema_version"] = "native-agent-runner.terminal-outcome.v1"

    restored = TerminalOutcome.from_json(payload)

    assert restored.schema_version == "native-agent-runner.terminal-outcome.v1"
    assert restored.to_json()["schema_version"] == TERMINAL_OUTCOME_SCHEMA_VERSION


@pytest.mark.parametrize("kind", get_args(TerminalOutcomeKind))
def test_terminal_outcome_accepts_every_declared_kind(kind: TerminalOutcomeKind) -> None:
    outcome = TerminalOutcome(
        run_id="run_1",
        kind=kind,
        retry_eligibility="forbidden",
    )

    assert TerminalOutcome.from_json(outcome.to_json()).kind == kind


@pytest.mark.parametrize("eligibility", tuple(RetryEligibility))
def test_terminal_outcome_accepts_every_declared_retry_eligibility(
    eligibility: RetryEligibility,
) -> None:
    assert (
        TerminalOutcome(
            run_id="run_1",
            kind="failed_terminal",
            retry_eligibility=eligibility,
        ).retry_eligibility
        == eligibility
    )


@pytest.mark.parametrize(
    "eligibility",
    (RetryEligibility.AFTER_RECONCILIATION, RetryEligibility.FORBIDDEN),
)
def test_dispatch_unknown_accepts_only_nonautomatic_retry_eligibility(
    eligibility: RetryEligibility,
) -> None:
    outcome = TerminalOutcome(
        run_id="run_1",
        kind="dispatch_unknown",
        retry_eligibility=eligibility,
    )

    assert outcome.retry_eligibility is eligibility


@pytest.mark.parametrize(
    "eligibility",
    (
        RetryEligibility.NOT_APPLICABLE,
        RetryEligibility.SAFE,
        RetryEligibility.AFTER_CONFIGURATION,
    ),
)
def test_dispatch_unknown_rejects_automatic_retry_eligibility(
    eligibility: RetryEligibility,
) -> None:
    with pytest.raises(ValueError, match="requires reconciliation"):
        TerminalOutcome(
            run_id="run_1",
            kind="dispatch_unknown",
            retry_eligibility=eligibility,
        )


@pytest.mark.parametrize(
    "eligibility",
    (
        RetryEligibility.NOT_APPLICABLE,
        RetryEligibility.SAFE,
        RetryEligibility.AFTER_CONFIGURATION,
        RetryEligibility.AFTER_RECONCILIATION,
    ),
)
def test_limited_outcome_rejects_retry_eligibility(
    eligibility: RetryEligibility,
) -> None:
    with pytest.raises(ValueError, match="limited forbids retry"):
        TerminalOutcome(
            run_id="run_1",
            kind="limited",
            retry_eligibility=eligibility,
        )


@pytest.mark.parametrize("cause", tuple(InterruptionCause))
def test_terminal_outcome_accepts_every_declared_interruption_cause(
    cause: InterruptionCause,
) -> None:
    assert (
        TerminalOutcome(
            run_id="run_1",
            kind="interrupted",
            retry_eligibility="forbidden",
            interruption_cause=cause,
        ).interruption_cause
        == cause
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", "monoid.terminal-outcome.v2"),
        ("run_id", ""),
        ("kind", "surprise"),
        ("retry_eligibility", "maybe"),
        ("interruption_cause", "worker_stop"),
        ("checkpoint_seq", True),
        ("checkpoint_seq", -1),
        ("http_status", True),
        ("http_status", -1),
        ("http_status", 99),
        ("http_status", 600),
        ("error_code", 500),
    ),
)
def test_terminal_outcome_rejects_values_outside_the_contract(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "run_id": "run_1",
        "kind": "failed_terminal",
        "retry_eligibility": "forbidden",
    }
    kwargs[field] = value

    with pytest.raises((TypeError, ValueError)):
        TerminalOutcome(**kwargs)  # type: ignore[arg-type]


def test_terminal_outcome_rejects_an_unserializable_checkpoint_sequence() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        TerminalOutcome(
            run_id="run_1",
            kind="completed",
            retry_eligibility="not_applicable",
            checkpoint_seq=10**5000,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("run_id", "private run text"),
        ("run_id", "r" * 257),
        ("final_output_ref", "secret"),
        ("final_output_ref", "model output text"),
        ("partial_output_ref", "line\nbreak"),
        ("last_evidence_ref", "évidence"),
        ("final_output_ref", "r" * 257),
    ),
)
def test_terminal_outcome_rejects_free_text_and_unbounded_references(
    field: str,
    value: str,
) -> None:
    kwargs = {
        "run_id": "run_1",
        "kind": "failed_terminal",
        "retry_eligibility": "forbidden",
        field: value,
    }

    with pytest.raises(ValueError, match="bounded opaque"):
        TerminalOutcome(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("error_code", "raw exception message"),
        ("provider_error_code", "bad/code"),
        ("error_code", "line\nbreak"),
        ("provider_error_code", "c" * 129),
    ),
)
def test_terminal_outcome_rejects_free_text_and_unbounded_error_codes(
    field: str,
    value: str,
) -> None:
    kwargs = {
        "run_id": "run_1",
        "kind": "failed_terminal",
        "retry_eligibility": "forbidden",
        field: value,
    }

    with pytest.raises(ValueError, match="bounded taxonomy code"):
        TerminalOutcome(**kwargs)  # type: ignore[arg-type]


def test_terminal_outcome_reader_requires_a_versioned_nonempty_identity() -> None:
    payload = TerminalOutcome(
        run_id="run_1",
        kind="completed",
        retry_eligibility="not_applicable",
    ).to_json()

    for broken in ({**payload, "schema_version": ""}, {**payload, "run_id": ""}):
        with pytest.raises(ValueError):
            TerminalOutcome.from_json(broken)


def test_terminal_outcome_rejects_equal_but_non_string_kind() -> None:
    class EqualToCompleted:
        def __eq__(self, other: object) -> bool:
            return other == "completed"

    with pytest.raises(ValueError, match="kind"):
        TerminalOutcome(
            run_id="run_1",
            kind=EqualToCompleted(),  # type: ignore[arg-type]
            retry_eligibility="not_applicable",
        )

    payload = TerminalOutcome(
        run_id="run_1",
        kind="completed",
        retry_eligibility="not_applicable",
    ).to_json()
    payload["kind"] = EqualToCompleted()
    with pytest.raises(ValueError, match="kind"):
        TerminalOutcome.from_json(payload)


@pytest.mark.parametrize(
    "field",
    ("prompt", "request_body", "rawResponse", "unknown_future_field"),
)
def test_terminal_outcome_strict_reader_rejects_unknown_top_level_fields(field: str) -> None:
    payload = TerminalOutcome(
        run_id="run_1",
        kind="completed",
        retry_eligibility="not_applicable",
    ).to_json()
    payload[field] = "private content"

    with pytest.raises(ValueError, match="outside its closed schema"):
        TerminalOutcome.from_json(payload)


def test_terminal_outcome_has_no_content_or_raw_exception_channel() -> None:
    fields = TerminalOutcome.__dataclass_fields__

    assert not {
        "prompt",
        "response",
        "reasoning",
        "raw",
        "raw_exception",
        "replay_payload",
    }.intersection(fields)


def test_terminal_outcome_constructor_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        TerminalOutcome(  # type: ignore[misc]
            "run_1",
            "completed",
            "not_applicable",
        )


def test_evidence_uncommitted_suspension_maps_to_safe_sink_only_recovery() -> None:
    outcome = terminal_outcome_from_suspension(
        Suspension(
            reason="turn_failed",
            status="failed",
            error="private infrastructure detail",
            error_code="evidence_uncommitted",
            retryable=True,
        ),
        run_id="run_1",
        checkpoint_seq=8,
        last_evidence_ref="invocation:call_1:3",
    )

    assert outcome == TerminalOutcome(
        run_id="run_1",
        kind="evidence_uncommitted",
        retry_eligibility=RetryEligibility.SAFE,
        checkpoint_seq=8,
        last_evidence_ref="invocation:call_1:3",
        error_code="evidence_uncommitted",
    )
    assert "private infrastructure detail" not in str(outcome.to_json())


def test_dispatch_unknown_suspension_requires_reconciliation_before_retry() -> None:
    outcome = terminal_outcome_from_suspension(
        Suspension(
            reason="terminal",
            status="failed",
            error_code="dispatch_unknown",
            retryable=True,
        ),
        run_id="run_1",
    )

    assert outcome.kind == "dispatch_unknown"
    assert outcome.retry_eligibility is RetryEligibility.AFTER_RECONCILIATION


@pytest.mark.parametrize(
    ("suspension", "kind", "retry", "cause"),
    (
        (
            Suspension(reason="settled", status="completed"),
            "completed",
            RetryEligibility.NOT_APPLICABLE,
            None,
        ),
        (
            Suspension(reason="paused", status="completed"),
            "paused",
            RetryEligibility.NOT_APPLICABLE,
            None,
        ),
        (
            Suspension(
                reason="limited",
                status="limited",
                error_code="max_steps_exceeded",
            ),
            "limited",
            RetryEligibility.FORBIDDEN,
            None,
        ),
        (
            Suspension(reason="interrupted", status="completed"),
            "interrupted",
            RetryEligibility.SAFE,
            None,
        ),
        (
            Suspension(
                reason="turn_failed",
                status="failed",
                error_code="bad_configuration",
                config_recoverable=True,
            ),
            "failed_config",
            RetryEligibility.AFTER_CONFIGURATION,
            None,
        ),
        (
            Suspension(
                reason="turn_failed",
                status="failed",
                error_code="rate_limited",
                retryable=True,
            ),
            "failed_retryable",
            RetryEligibility.SAFE,
            None,
        ),
        (
            Suspension(
                reason="terminal",
                status="limited",
                error_code="cancelled",
            ),
            "cancelled",
            RetryEligibility.FORBIDDEN,
            None,
        ),
        (
            Suspension(
                reason="terminal",
                status="limited",
                error_code="run_timeout",
            ),
            "cancelled",
            RetryEligibility.FORBIDDEN,
            None,
        ),
    ),
)
def test_suspension_outcome_projection_uses_the_portable_classification(
    suspension: Suspension,
    kind: TerminalOutcomeKind,
    retry: RetryEligibility,
    cause: InterruptionCause | None,
) -> None:
    outcome = terminal_outcome_from_suspension(suspension, run_id="run_1")

    assert outcome.kind == kind
    assert outcome.retry_eligibility is retry
    assert outcome.interruption_cause is cause
