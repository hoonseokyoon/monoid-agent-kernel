"""The reusable Redactor contract, run against every redactor this repo ships.

The suite is the same object an external implementer runs, so the negative cases below matter as
much as the positive one: a contract that cannot fail proves nothing about the implementations that
pass it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from monoid_agent_kernel.conformance.contracts import run_redactor_contract
from monoid_agent_kernel.core.model_io import DefaultRedactor, RedactionPolicy, Redactor

FACTORIES: list[Any] = [pytest.param(DefaultRedactor, id="default")]


@pytest.fixture(params=FACTORIES)
def redactor_factory(request: pytest.FixtureRequest) -> Callable[[], Redactor]:
    return request.param  # type: ignore[no-any-return]


def test_shipped_redactors_satisfy_the_contract(redactor_factory: Callable[[], Redactor]) -> None:
    outcomes = run_redactor_contract(redactor_factory)

    assert [outcome.rule_id for outcome in outcomes] == [
        "REDACTOR-01-DETERMINISTIC",
        "REDACTOR-02-NO-DEFAULT-SECRET-LEAK",
        "REDACTOR-03-FAILURE-IS-CONTAINED",
        "REDACTOR-04-PRESERVES-THE-VALUE-SHAPE",
    ]
    assert all(outcome.status == "passed" for outcome in outcomes), [
        (outcome.rule_id, outcome.status, outcome.error) for outcome in outcomes
    ]


def _statuses(outcomes: tuple[Any, ...]) -> dict[str, str]:
    return {outcome.rule_id: outcome.status for outcome in outcomes}


def test_the_contract_catches_a_nondeterministic_redactor() -> None:
    class Counting:
        """Masks correctly but tags each result, the way a redactor that stamped a timestamp or a
        uuid into its output would."""

        def __init__(self) -> None:
            self.calls = 0

        def redact(self, value: Any, *, policy: RedactionPolicy) -> Any:
            self.calls += 1
            redacted = DefaultRedactor().redact(value, policy=policy)
            if isinstance(redacted, dict):
                return {**redacted, "_pass": str(self.calls)}
            return redacted

    # Passed as the class, not as a shared instance. The determinism rule calls one redactor twice,
    # because a CapturePolicy holds its redactor for the life of the policy -- constructing a second
    # instance would hide exactly the per-instance state production would hit.
    statuses = _statuses(run_redactor_contract(Counting))

    assert statuses["REDACTOR-01-DETERMINISTIC"] == "failed"
    assert statuses["REDACTOR-02-NO-DEFAULT-SECRET-LEAK"] == "passed"


def test_the_contract_catches_a_redactor_that_ignores_the_default_secret_keys() -> None:
    class TextOnly:
        """Applies the free-text rules and nothing else — a plausible implementation that forgets
        structured payloads carry secrets under named keys."""

        def redact(self, value: Any, *, policy: RedactionPolicy) -> Any:
            if isinstance(value, str):
                return policy.redact_text(value)
            if isinstance(value, dict):
                return {key: self.redact(item, policy=policy) for key, item in value.items()}
            if isinstance(value, list):
                return [self.redact(item, policy=policy) for item in value]
            return value

    statuses = _statuses(run_redactor_contract(TextOnly))

    assert statuses["REDACTOR-01-DETERMINISTIC"] == "passed"
    assert statuses["REDACTOR-02-NO-DEFAULT-SECRET-LEAK"] == "failed"


def test_the_contract_catches_a_redactor_that_masks_everything() -> None:
    """"Redact the whole payload" would otherwise satisfy every leak rule trivially."""

    class MaskAll:
        def redact(self, value: Any, *, policy: RedactionPolicy) -> Any:
            return policy.replacement

    statuses = _statuses(run_redactor_contract(MaskAll))

    assert statuses["REDACTOR-02-NO-DEFAULT-SECRET-LEAK"] == "failed"
    # And it hands the pipeline a scalar where fields are needed, which is its own rule: the pipeline
    # fails closed on that, so a redactor tripping it silently loses its consumer's content.
    assert statuses["REDACTOR-04-PRESERVES-THE-VALUE-SHAPE"] == "failed"


def test_a_raising_redactor_reports_an_error_rather_than_taking_the_suite_down() -> None:
    class Failing:
        def redact(self, value: Any, *, policy: RedactionPolicy) -> Any:
            raise RuntimeError("classifier unavailable")

    outcomes = run_redactor_contract(Failing)
    statuses = _statuses(outcomes)

    assert statuses["REDACTOR-01-DETERMINISTIC"] == "error"
    assert statuses["REDACTOR-02-NO-DEFAULT-SECRET-LEAK"] == "error"
    # Containment holds -- nothing propagated -- but the rule still fails, because it also requires
    # that a *successful* redaction be distinguishable from a failed one, and this redactor never
    # succeeds. "Always returns None" is fail-closed and useless, and the rule says both.
    assert statuses["REDACTOR-03-FAILURE-IS-CONTAINED"] == "failed"
    assert all(outcome.error for outcome in outcomes if outcome.status == "error")


def test_the_suite_never_raises_out_of_a_broken_factory() -> None:
    def factory() -> Redactor:
        raise RuntimeError("cannot construct")

    statuses = _statuses(run_redactor_contract(factory))

    assert set(statuses.values()) == {"error"}
