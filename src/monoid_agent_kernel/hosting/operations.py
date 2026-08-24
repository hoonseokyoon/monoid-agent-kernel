"""Dependency-neutral, public-safe operational metric snapshots."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable


_METRIC_NAME = re.compile(r"monoid\.[a-z0-9_.]{1,191}\Z", re.ASCII)
_ATTRIBUTE_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)
_ATTRIBUTE_NAMES = frozenset(
    {
        "component",
        "operation",
        "queue",
        "result",
        "state",
        "status",
        "type",
    }
)
_UNITS = frozenset({"1", "By", "s"})


@dataclass(frozen=True, kw_only=True)
class OperationalMetric:
    """One bounded aggregate measurement with public attribute keys.

    The caller owns value cardinality; adapter collectors use fixed value vocabularies.
    """

    name: str
    value: int | float
    unit: str = "1"
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if type(self.name) is not str or _METRIC_NAME.fullmatch(self.name) is None:
            raise ValueError("operational metric name must use the bounded monoid.* vocabulary")
        if (
            type(self.value) not in {int, float}
            or isinstance(self.value, bool)
            or not math.isfinite(float(self.value))
        ):
            raise ValueError("operational metric value must be a finite number")
        if self.unit not in _UNITS:
            raise ValueError("operational metric unit is outside the supported vocabulary")
        if type(self.attributes) is not tuple or len(self.attributes) > 8:
            raise ValueError("operational metric attributes must be a bounded tuple")
        if tuple(sorted(self.attributes)) != self.attributes:
            raise ValueError("operational metric attributes must use canonical key order")
        names: set[str] = set()
        for item in self.attributes:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("operational metric attribute must be a key/value pair")
            key, value = item
            if type(key) is not str or key not in _ATTRIBUTE_NAMES:
                raise ValueError("operational metric attribute name is outside the public vocabulary")
            if type(value) is not str or _ATTRIBUTE_VALUE.fullmatch(value) is None:
                raise ValueError("operational metric attribute value is invalid")
            if key in names:
                raise ValueError("operational metric attribute names must be unique")
            names.add(key)

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True, kw_only=True)
class OperationalSnapshot:
    """A point-in-time aggregate report suitable for metrics and structured logs."""

    source: str
    collected_at: datetime
    metrics: tuple[OperationalMetric, ...]

    def __post_init__(self) -> None:
        if type(self.source) is not str or _ATTRIBUTE_VALUE.fullmatch(self.source) is None:
            raise ValueError("operational snapshot source must be a bounded public identity")
        if not isinstance(self.collected_at, datetime) or self.collected_at.tzinfo is None:
            raise ValueError("operational snapshot collected_at must be timezone-aware")
        if type(self.metrics) is not tuple or not self.metrics:
            raise ValueError("operational snapshot must contain a non-empty metric tuple")
        identities = tuple((metric.name, metric.attributes) for metric in self.metrics)
        if len(set(identities)) != len(identities):
            raise ValueError("operational snapshot metric identities must be unique")
        if tuple(sorted(identities)) != identities:
            raise ValueError("operational snapshot metrics must use canonical identity order")

    def to_json(self) -> dict[str, object]:
        return {
            "source": self.source,
            "collected_at": self.collected_at.isoformat(),
            "metrics": [metric.to_json() for metric in self.metrics],
        }


@runtime_checkable
class OperationalMetricSink(Protocol):
    """Host-owned exporter for an explicit operational collection pass."""

    def record(self, metric: OperationalMetric) -> None: ...


def record_operational_snapshot(
    snapshot: OperationalSnapshot,
    sink: OperationalMetricSink,
) -> int:
    """Record one canonical snapshot and return its exact measurement count."""

    if not isinstance(snapshot, OperationalSnapshot):
        raise TypeError("record_operational_snapshot requires OperationalSnapshot")
    record = getattr(sink, "record", None)
    if not callable(record):
        raise TypeError("operational metric sink must provide record(metric)")
    for metric in snapshot.metrics:
        record(metric)
    return len(snapshot.metrics)


__all__ = [
    "OperationalMetric",
    "OperationalSnapshot",
    "OperationalMetricSink",
    "record_operational_snapshot",
]
