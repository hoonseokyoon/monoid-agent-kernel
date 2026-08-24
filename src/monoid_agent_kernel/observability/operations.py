"""OpenTelemetry export for public operational snapshots."""

from __future__ import annotations

from threading import RLock
from typing import Any

from monoid_agent_kernel.hosting.operations import OperationalMetric


class OtelOperationalMetricSink:
    """Retain the latest aggregate values behind OTel observable gauges.

    OpenTelemetry stays optional and host-configured. The sink imports only the API when it is
    instantiated and accepts an explicit meter provider for isolated embedding and tests.
    """

    def __init__(
        self,
        *,
        meter_name: str = "monoid_agent_kernel.operations",
        meter_provider: Any = None,
    ) -> None:
        if type(meter_name) is not str or not meter_name or len(meter_name) > 255:
            raise ValueError("OTel operational meter_name must be a non-empty bounded string")
        try:
            from opentelemetry import metrics
            from opentelemetry.metrics import Observation
        except ImportError as exc:  # pragma: no cover - exercised in installed import smoke
            raise RuntimeError(
                "OtelOperationalMetricSink requires opentelemetry; "
                "install monoid-agent-kernel[otel]"
            ) from exc
        self._Observation = Observation
        self._meter = metrics.get_meter(meter_name, meter_provider=meter_provider)
        self._lock = RLock()
        self._units: dict[str, str] = {}
        self._values: dict[str, dict[tuple[tuple[str, str], ...], int | float]] = {}
        self._instruments: dict[str, object] = {}

    def _observe(self, name: str) -> tuple[Any, ...]:
        with self._lock:
            values = tuple(sorted(self._values[name].items()))
        return tuple(
            self._Observation(value, attributes=dict(attributes))
            for attributes, value in values
        )

    def record(self, metric: OperationalMetric) -> None:
        """Publish the latest value for one fixed-cardinality metric identity."""

        if not isinstance(metric, OperationalMetric):
            raise TypeError("OTel operational sink requires OperationalMetric")
        with self._lock:
            known_unit = self._units.get(metric.name)
            if known_unit is not None and known_unit != metric.unit:
                raise ValueError("OTel operational metric name cannot change unit")
            self._units[metric.name] = metric.unit
            self._values.setdefault(metric.name, {})[metric.attributes] = metric.value
            if metric.name not in self._instruments:
                def callback(_options: object, name: str = metric.name) -> tuple[Any, ...]:
                    return self._observe(name)

                self._instruments[metric.name] = self._meter.create_observable_gauge(
                    metric.name,
                    callbacks=(callback,),
                    unit=metric.unit,
                    description="Monoid Agent Kernel aggregate operational state",
                )


__all__ = ["OtelOperationalMetricSink"]
