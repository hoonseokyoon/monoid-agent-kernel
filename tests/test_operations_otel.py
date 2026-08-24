from __future__ import annotations

import pytest


pytest.importorskip("opentelemetry.sdk.metrics")

from opentelemetry.sdk.metrics import MeterProvider  # noqa: E402
from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: E402

from monoid_agent_kernel.hosting import OperationalMetric  # noqa: E402
from monoid_agent_kernel.observability import OtelOperationalMetricSink  # noqa: E402


def _points(reader: InMemoryMetricReader) -> dict[tuple[str, tuple[tuple[str, object], ...]], object]:
    data = reader.get_metrics_data()
    return {
        (metric.name, tuple(sorted(point.attributes.items()))): point.value
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        for point in metric.data.data_points
    }


def test_otel_operational_sink_exports_latest_public_gauges() -> None:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=(reader,))
    sink = OtelOperationalMetricSink(meter_provider=provider)

    sink.record(
        OperationalMetric(
            name="monoid.postgres.outbox.count",
            value=2,
            attributes=(("queue", "activation"), ("state", "pending")),
        )
    )
    sink.record(
        OperationalMetric(
            name="monoid.postgres.outbox.count",
            value=3,
            attributes=(("queue", "activation"), ("state", "pending")),
        )
    )
    sink.record(
        OperationalMetric(
            name="monoid.postgres.stream.chunk.bytes",
            value=10,
            unit="By",
        )
    )

    points = _points(reader)
    assert points[
        (
            "monoid.postgres.outbox.count",
            (("queue", "activation"), ("state", "pending")),
        )
    ] == 3
    assert points[("monoid.postgres.stream.chunk.bytes", ())] == 10
    provider.shutdown()


def test_otel_operational_sink_rejects_unit_drift() -> None:
    sink = OtelOperationalMetricSink(meter_provider=MeterProvider())
    sink.record(OperationalMetric(name="monoid.postgres.object.bytes", value=1, unit="By"))

    with pytest.raises(ValueError, match="unit"):
        sink.record(OperationalMetric(name="monoid.postgres.object.bytes", value=1, unit="1"))
