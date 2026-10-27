"""Content-free termination facts and explicitly reported token observations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, get_args

from monoid_agent_kernel.core.json_ingress import is_portable_json_integer
from monoid_agent_kernel.core.wire_validation import require_object, require_only_fields

CallOutcome = Literal["completed", "interrupted", "failed", "cancelled", "timed_out"]
StopRequester = Literal[
    "user_cancel", "turn_interrupt", "graceful_drain", "deadline", "host_shutdown", "caller_cancel"
]
ReadEnd = Literal[
    "provider_terminal",
    "provider_eof",
    "provider_error",
    "provider_cancelled",
    "stop_request",
    "deadline",
    "drain_timeout",
    "lease_lost",
    "local_failure",
]
UsageSource = Literal["provider", "adapter", "legacy_ambiguous"]
UsageCollection = Literal["normal", "drain"]
USAGE_COUNTERS = (
    "audio_input_tokens",
    "audio_output_tokens",
    "audio_tokens",
    "cached_input_tokens",
    "cache_creation_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)


def _literal(value: object, vocabulary: Any, label: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if type(value) is not str or value not in get_args(vocabulary):
        raise ValueError(f"{label} is outside the public vocabulary")


def _object(payload: Any, keys: tuple[str, ...], label: str) -> dict[str, Any]:
    payload = require_object(payload, label)
    require_only_fields(payload, frozenset(keys), label)
    if set(payload) != set(keys):
        raise ValueError(f"{label} must contain every declared field")
    return payload


def _reported(value: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError("reported usage must be an object")
    result = dict(value)
    if any(
        type(key) is not str
        or key not in USAGE_COUNTERS
        or not is_portable_json_integer(count)
        or count < 0
        for key, count in result.items()
    ):
        raise ValueError(
            "reported usage requires known counters and non-negative portable integers"
        )
    return result


@dataclass(frozen=True, kw_only=True)
class ModelCallTermination:
    """The call result, first accepted stop request, and actual end of provider reading."""

    outcome: CallOutcome
    ended_by: ReadEnd
    requested_by: StopRequester | None = None

    def __post_init__(self) -> None:
        _literal(self.outcome, CallOutcome, "termination outcome")
        _literal(self.ended_by, ReadEnd, "termination ended_by")
        _literal(self.requested_by, StopRequester, "termination requested_by", nullable=True)

    def to_json(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "requested_by": self.requested_by,
            "ended_by": self.ended_by,
        }

    @classmethod
    def from_json(cls, payload: Any) -> ModelCallTermination:
        return cls(**_object(payload, ("outcome", "requested_by", "ended_by"), "termination"))


@dataclass(frozen=True, kw_only=True)
class UsageObservation:
    """One dispatch's last explicit snapshot; an empty report makes no counter claim."""

    reported: Mapping[str, int] = field(default_factory=dict)
    source: UsageSource | None = None
    collection: UsageCollection | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reported", MappingProxyType(_reported(self.reported)))
        _literal(self.source, UsageSource, "usage source", nullable=True)
        _literal(self.collection, UsageCollection, "usage collection", nullable=True)
        if self.reported:
            if self.source not in {"provider", "adapter"} or self.collection is None:
                raise ValueError("reported usage requires an explicit source and collection")
        elif self.source not in {None, "legacy_ambiguous"} or self.collection is not None:
            raise ValueError("unreported usage must have no collection or asserted source")

    def to_json(self) -> dict[str, Any]:
        return {
            "reported": dict(self.reported),
            "source": self.source,
            "collection": self.collection,
        }

    @classmethod
    def from_json(cls, payload: Any) -> UsageObservation:
        return cls(**_object(payload, ("reported", "source", "collection"), "usage observation"))

    def validate_usage(self, usage: Mapping[str, int]) -> None:
        if any(usage.get(key, 0) != count for key, count in self.reported.items()):
            raise ValueError("usage observation must agree with dispatch usage")


@dataclass(frozen=True, kw_only=True)
class UsageEvidenceAttempt:
    """An observation at its one-based kernel dispatch index."""

    index: int
    observation: UsageObservation

    def __post_init__(self) -> None:
        if not is_portable_json_integer(self.index) or self.index < 1:
            raise ValueError("usage evidence attempt index must be a positive integer")
        if not isinstance(self.observation, UsageObservation):
            raise ValueError("usage evidence attempt requires an observation")

    def to_json(self) -> dict[str, Any]:
        return {"index": self.index, **self.observation.to_json()}

    @classmethod
    def from_json(cls, payload: Any) -> UsageEvidenceAttempt:
        raw = _object(
            payload, ("index", "reported", "source", "collection"), "usage evidence attempt"
        )
        return cls(
            index=raw["index"],
            observation=UsageObservation.from_json(
                {key: raw[key] for key in ("reported", "source", "collection")},
            ),
        )


@dataclass(frozen=True, kw_only=True)
class UsageEvidence:
    """Counter coverage across kernel dispatches, independent of the numeric accounting path."""

    attempts: tuple[UsageEvidenceAttempt, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "attempts", tuple(self.attempts))
        if any(
            not isinstance(entry, UsageEvidenceAttempt) or entry.index != index
            for index, entry in enumerate(self.attempts, 1)
        ):
            raise ValueError("usage evidence must name every attempt once in order")
        _reported(self.reported)

    @property
    def reported(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for entry in self.attempts:
            for key, count in entry.observation.reported.items():
                totals[key] = totals.get(key, 0) + count
        return totals

    @property
    def completeness(self) -> dict[str, str]:
        result = {}
        for key in USAGE_COUNTERS:
            count = sum(key in entry.observation.reported for entry in self.attempts)
            result[key] = (
                "no_dispatch"
                if not self.attempts
                else "complete"
                if count == len(self.attempts)
                else "partial"
                if count
                else "unknown"
            )
        return result

    def validate_usage(self, usage: Mapping[str, int], attempts: int) -> None:
        if len(self.attempts) != attempts:
            raise ValueError("usage_evidence must cover the receipt attempt count")
        coverage = self.completeness
        for key, count in self.reported.items():
            numeric = usage.get(key, 0)
            if count > numeric or (coverage[key] == "complete" and count != numeric):
                raise ValueError("usage_evidence must agree with numeric usage and coverage")

    def to_json(self) -> dict[str, Any]:
        return {
            "reported": self.reported,
            "completeness": self.completeness,
            "attempts": [entry.to_json() for entry in self.attempts],
        }

    @classmethod
    def from_json(cls, payload: Any) -> UsageEvidence:
        raw = _object(payload, ("reported", "completeness", "attempts"), "usage_evidence")
        if not isinstance(raw["attempts"], list):
            raise ValueError("usage_evidence attempts must be an array")
        value = cls(
            attempts=tuple(UsageEvidenceAttempt.from_json(item) for item in raw["attempts"])
        )
        if (
            _reported(raw["reported"]) != value.reported
            or raw["completeness"] != value.completeness
        ):
            raise ValueError("usage_evidence aggregate must equal its observations")
        return value


def mark_call_facts(
    error: BaseException, termination: ModelCallTermination | None, evidence: UsageEvidence | None,
) -> None:
    """Carry the already-settled logical-call facts through exception translation."""
    try:
        error.model_call_termination = termination  # type: ignore[attr-defined]
        error.model_usage_evidence = evidence  # type: ignore[attr-defined]
    except Exception:
        pass


def call_facts_of(error: BaseException) -> tuple[ModelCallTermination | None, UsageEvidence | None]:
    try:
        termination = getattr(error, "model_call_termination", None)
        evidence = getattr(error, "model_usage_evidence", None)
        return (
            termination if isinstance(termination, ModelCallTermination) else None,
            evidence if isinstance(evidence, UsageEvidence) else None,
        )
    except Exception:
        return None, None


def mark_read_end(value: Any, ended_by: ReadEnd) -> None:
    """The provider/read owner records its known end before an error is translated."""
    try:
        object.__setattr__(value, "_model_read_end", ended_by)
    except Exception:
        pass


def read_end_of(value: Any) -> ReadEnd | None:
    try:
        ended_by = getattr(value, "_model_read_end", None)
        return ended_by if ended_by in get_args(ReadEnd) else None
    except Exception:
        return None
