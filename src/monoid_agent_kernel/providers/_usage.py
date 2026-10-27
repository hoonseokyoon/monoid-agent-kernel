"""One dispatch's observed usage, retained across an abandoned wait and bounded cleanup."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from threading import Lock
from typing import Any

from monoid_agent_kernel.core.json_ingress import is_portable_json_integer
from monoid_agent_kernel.core.model_facts import UsageObservation
from monoid_agent_kernel.providers._common import normalize_usage, usage_reported_by


def select_usage_snapshot(
    current: tuple[dict[str, int], UsageObservation],
    incoming: tuple[dict[str, int], UsageObservation],
) -> tuple[dict[str, int], UsageObservation]:
    """Replace explicit snapshots together; an unreported terminal cannot erase one."""
    usage, observation = incoming
    if not observation.reported and not any(usage.values()):
        return (current[0], observation) if not current[1].reported else current
    if observation.reported or (
        not current[1].reported and (any(usage.values()) or not any(current[0].values()))
    ):
        return dict(usage), observation
    return current


class UsageProgress:
    def __init__(self) -> None:
        self.snapshot: tuple[dict[str, int], UsageObservation] = (
            {}, UsageObservation(source="legacy_ambiguous"),
        )
        self.collection = "normal"
        self.closed = False
        self.lock = Lock()

    def begin_drain(self) -> None:
        with self.lock:
            if not self.closed:
                self.collection = "drain"

    def capture(
        self, usage: Mapping[str, int], observation: UsageObservation, *, preserve: bool,
    ) -> UsageObservation:
        with self.lock:
            if observation.reported and not preserve:
                observation = replace(observation, collection=self.collection)
            if not self.closed:
                self.snapshot = select_usage_snapshot(self.snapshot, (dict(usage), observation))
            return observation

    def close(self) -> None:
        with self.lock:
            self.closed = True


_USAGE_PROGRESS: ContextVar[UsageProgress | None] = ContextVar("monoid_usage_progress", default=None)


@contextmanager
def collect_usage_reports() -> Iterator[UsageProgress]:
    progress = UsageProgress()
    token = _USAGE_PROGRESS.set(progress)
    try:
        yield progress
    finally:
        progress.close()
        _USAGE_PROGRESS.reset(token)


def capture_usage(
    usage: Mapping[str, int], observation: UsageObservation | None, *, preserve: bool = False,
) -> UsageObservation:
    observation = observation or UsageObservation(source="legacy_ambiguous")
    if not isinstance(observation, UsageObservation):
        raise ValueError("model usage_evidence must be a UsageObservation")
    observation.validate_usage(usage)
    progress = _USAGE_PROGRESS.get()
    return observation if progress is None else progress.capture(usage, observation, preserve=preserve)


def current_usage_progress() -> UsageProgress | None:
    return _USAGE_PROGRESS.get()


def provider_snapshot(
    raw: Any, *, lenient: bool = False, observed: UsageObservation | None = None,
) -> tuple[dict[str, int], UsageObservation]:
    """Use the same alias precedence for numeric and sparse first-party source readings."""
    if lenient:
        raw = raw if isinstance(raw, dict) else {}
        detail_fields = {
            "input_tokens_details", "prompt_tokens_details",
            "output_tokens_details", "completion_tokens_details",
        }
        raw = {
            key: ({k: v for k, v in value.items() if is_portable_json_integer(v) and v >= 0}
                  if key in detail_fields else value)
            for key, value in raw.items()
            if (isinstance(value, dict) if key in detail_fields
                else is_portable_json_integer(value) and value >= 0)
        }
    reported = normalize_usage(raw, legacy_aliases=True, sparse=True)
    usage = (
        {**usage_reported_by({"usage": raw}), **reported} if lenient
        else normalize_usage(raw, legacy_aliases=True)
    )
    observation = UsageObservation(
        reported=reported, source="provider" if reported else None,
        collection="normal" if reported else None,
    )
    if observed is not None and observed.reported == reported:
        return usage, capture_usage(usage, observed, preserve=True)
    return usage, capture_usage(usage, observation)


def mark_usage_observation(error: BaseException, observation: UsageObservation) -> None:
    try:
        error.provider_usage_evidence = observation  # type: ignore[attr-defined]
    except Exception:
        pass


def usage_observation_of(error: BaseException) -> UsageObservation | None:
    try:
        value = getattr(error, "provider_usage_evidence", None)
        return value if isinstance(value, UsageObservation) else None
    except Exception:
        return None
