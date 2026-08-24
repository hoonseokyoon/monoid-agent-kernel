from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from monoid_agent_kernel.hosting import (
    OperationalMetric,
    OperationalSnapshot,
    record_operational_snapshot,
)


def _snapshot() -> OperationalSnapshot:
    return OperationalSnapshot(
        source="postgres",
        collected_at=datetime(2026, 8, 25, tzinfo=UTC),
        metrics=(
            OperationalMetric(
                name="monoid.postgres.invocation.count",
                value=2,
                attributes=(("state", "unknown"),),
            ),
            OperationalMetric(
                name="monoid.postgres.outbox.oldest_age",
                value=1.5,
                unit="s",
                attributes=(("queue", "activation"),),
            ),
        ),
    )


def test_operational_snapshot_is_canonical_public_json_and_records_exactly_once() -> None:
    snapshot = _snapshot()

    class Sink:
        def __init__(self) -> None:
            self.metrics: list[OperationalMetric] = []

        def record(self, metric: OperationalMetric) -> None:
            self.metrics.append(metric)

    sink = Sink()
    assert record_operational_snapshot(snapshot, sink) == 2
    assert sink.metrics == list(snapshot.metrics)
    assert snapshot.to_json() == {
        "source": "postgres",
        "collected_at": "2026-08-25T00:00:00+00:00",
        "metrics": [metric.to_json() for metric in snapshot.metrics],
    }


@pytest.mark.parametrize(
    "metric",
    (
        lambda: OperationalMetric(name="tenant.private.value", value=1),
        lambda: OperationalMetric(name="monoid.safe", value=float("nan")),
        lambda: OperationalMetric(name="monoid.safe", value=10**1000),
        lambda: OperationalMetric(name="monoid.safe", value=1, unit="tokens"),
        lambda: OperationalMetric(
            name="monoid.safe",
            value=1,
            attributes=(("run_id", "run/private"),),
        ),
        lambda: OperationalMetric(
            name="monoid.safe",
            value=1,
            attributes=(("state", "settled"), ("queue", "activation")),
        ),
    ),
)
def test_operational_metric_rejects_unbounded_or_noncanonical_carriage(
    metric: Callable[[], OperationalMetric],
) -> None:
    with pytest.raises(ValueError):
        metric()


def test_operational_snapshot_rejects_duplicate_or_unsorted_metric_identity() -> None:
    first = OperationalMetric(name="monoid.z", value=1)
    second = OperationalMetric(name="monoid.a", value=1)

    with pytest.raises(ValueError, match="canonical"):
        OperationalSnapshot(
            source="postgres",
            collected_at=datetime.now(UTC),
            metrics=(first, second),
        )
    with pytest.raises(ValueError, match="unique"):
        OperationalSnapshot(
            source="postgres",
            collected_at=datetime.now(UTC),
            metrics=(first, first),
        )
    with pytest.raises(ValueError, match="OperationalMetric"):
        OperationalSnapshot(
            source="postgres",
            collected_at=datetime.now(UTC),
            metrics=(object(),),  # type: ignore[arg-type]
        )


def test_record_operational_snapshot_propagates_export_failure() -> None:
    class FailingSink:
        def record(self, metric: OperationalMetric) -> None:
            del metric
            raise RuntimeError("export unavailable")

    with pytest.raises(RuntimeError, match="export unavailable"):
        record_operational_snapshot(_snapshot(), FailingSink())
