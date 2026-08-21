from __future__ import annotations

from typing import get_args

import pytest

from monoid_agent_kernel.core.outcome import (
    InterruptionCause,
    RetryEligibility,
    TERMINAL_OUTCOME_SCHEMA_VERSION,
    TerminalOutcome,
    TerminalOutcomeKind,
)


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


def test_terminal_outcome_reader_requires_a_versioned_nonempty_identity() -> None:
    payload = TerminalOutcome(
        run_id="run_1",
        kind="completed",
        retry_eligibility="not_applicable",
    ).to_json()

    for broken in ({**payload, "schema_version": ""}, {**payload, "run_id": ""}):
        with pytest.raises(ValueError):
            TerminalOutcome.from_json(broken)


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
