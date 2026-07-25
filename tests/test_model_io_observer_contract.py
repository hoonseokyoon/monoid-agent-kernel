"""The reusable ModelIOObserver contract.

The rules are guarantees the capture pipeline gives an observer, so the negative cases here drive the
pipeline itself rather than a bad observer: an observer has almost no obligations, and a suite that
cannot fail would say nothing about the guarantees it claims to check.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from monoid_agent_kernel.conformance.contracts import run_model_io_observer_contract
from monoid_agent_kernel.core.model_io import ModelCallCapture, ModelIOObserver

RULE_IDS = [
    "MODELIO-01-PARTIAL-IMPLEMENTATION-LEGAL",
    "MODELIO-02-OBSERVER-FAILURE-CONTAINED",
    "MODELIO-03-NONE-POLICY-RECEIVES-NO-CONTENT",
]


class MinimalObserver:
    """Everything an observer is required to be: one method."""

    def on_model_call(self, capture: ModelCallCapture) -> None:
        del capture


class ClosingObserver(MinimalObserver):
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class StoringObserver(MinimalObserver):
    """The shape most integrations take: keep what arrived."""

    def __init__(self) -> None:
        self.captures: list[ModelCallCapture] = []

    def on_model_call(self, capture: ModelCallCapture) -> None:
        self.captures.append(capture)


FACTORIES: list[Any] = [
    pytest.param(MinimalObserver, id="minimal"),
    pytest.param(ClosingObserver, id="closing"),
    pytest.param(StoringObserver, id="storing"),
]


@pytest.fixture(params=FACTORIES)
def observer_factory(request: pytest.FixtureRequest) -> Callable[[], ModelIOObserver]:
    return request.param  # type: ignore[no-any-return]


def test_observer_shapes_satisfy_the_contract(
    observer_factory: Callable[[], ModelIOObserver],
) -> None:
    outcomes = run_model_io_observer_contract(observer_factory)

    assert [outcome.rule_id for outcome in outcomes] == RULE_IDS
    assert all(outcome.status == "passed" for outcome in outcomes), [
        (outcome.rule_id, outcome.status, outcome.error) for outcome in outcomes
    ]


def test_the_suite_closes_every_observer_it_constructs() -> None:
    """It builds one per rule, and only the first was being released.

    A public conformance suite gets run repeatedly, which is exactly where a per-run leak of a file
    handle, thread or network client accumulates.
    """
    closed: list[int] = []

    class Closing:
        def on_model_call(self, capture: ModelCallCapture) -> None:
            del capture

        def close(self) -> None:
            closed.append(id(self))

    outcomes = run_model_io_observer_contract(Closing)

    assert all(outcome.status == "passed" for outcome in outcomes)
    assert len(closed) == len(RULE_IDS)
    assert len(set(closed)) == len(RULE_IDS)


def test_a_raising_close_does_not_lose_the_outcomes() -> None:
    class ClosingBadly:
        def on_model_call(self, capture: ModelCallCapture) -> None:
            del capture

        def close(self) -> None:
            raise RuntimeError("already gone")

    outcomes = run_model_io_observer_contract(ClosingBadly)

    assert [outcome.rule_id for outcome in outcomes] == RULE_IDS
    assert all(outcome.status == "passed" for outcome in outcomes)


def test_a_shared_observer_instance_is_closed_once() -> None:
    """`close_model_io_subscriptions` de-duplicates by identity, so a factory returning one instance
    does not get closed three times."""
    closes = []

    class Counting:
        def on_model_call(self, capture: ModelCallCapture) -> None:
            del capture

        def close(self) -> None:
            closes.append(1)

    shared = Counting()
    run_model_io_observer_contract(lambda: shared)

    assert len(closes) == 1


def test_an_observer_that_always_raises_still_satisfies_the_contract() -> None:
    """Because containment is the pipeline's job, not the observer's. An exporter that is down is
    not a reason to fail a model call the provider has already billed for."""

    class AlwaysRaising:
        def on_model_call(self, capture: ModelCallCapture) -> None:
            del capture
            raise RuntimeError("exporter unavailable")

    outcomes = run_model_io_observer_contract(AlwaysRaising)

    assert all(outcome.status == "passed" for outcome in outcomes)


def test_a_factory_that_cannot_construct_reports_errors_rather_than_raising() -> None:
    def factory() -> ModelIOObserver:
        raise RuntimeError("cannot construct")

    outcomes = run_model_io_observer_contract(factory)

    assert [outcome.rule_id for outcome in outcomes] == RULE_IDS
    assert {outcome.status for outcome in outcomes} == {"error"}
    assert all(outcome.error for outcome in outcomes)


def test_the_contract_redacts_exception_details_from_a_broken_factory() -> None:
    secret = "observer-secret-must-not-enter-report"

    def factory() -> ModelIOObserver:
        raise RuntimeError(secret)

    outcomes = run_model_io_observer_contract(factory)

    assert all(secret not in str(outcome.to_json()) for outcome in outcomes)
