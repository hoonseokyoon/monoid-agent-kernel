"""Closed provenance and evidence types for external conformance reports.

The harness remains the trusted translator. These types create a public, secret-minimized
representation whose exact bytes can be retained and checked independently of the harness.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from monoid_agent_kernel.identifiers import namespaced_id

CONFORMANCE_EVIDENCE_VERSION = namespaced_id("conformance-evidence.v1")
CONFORMANCE_EVIDENCE_MEDIA_TYPE = (
    "application/vnd.monoid.conformance-evidence.v1+json"
)
CONFORMANCE_EVIDENCE_KIND = "minimal-agent-lifecycle"
MAX_CONFORMANCE_EVIDENCE_BYTES = 8 * 1024 * 1024

_SAFE_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}\Z", re.ASCII)
_DIGEST_ALGORITHM_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,31}\Z", re.ASCII)
_LOWER_HEX_RE = re.compile(r"[a-f0-9]+\Z", re.ASCII)
_MEDIA_TYPE_RE = re.compile(
    r"[a-z0-9][a-z0-9!#$&^_.+-]{0,126}/[a-z0-9][a-z0-9!#$&^_.+-]{0,126}\Z",
    re.ASCII,
)
_KNOWN_DIGEST_HEX_LENGTHS = {"sha256": 64, "sha512": 128}


@dataclass(frozen=True, kw_only=True)
class ConformanceDigest:
    """One algorithm-qualified lowercase hexadecimal content digest."""

    algorithm: str
    value: str

    def __post_init__(self) -> None:
        _require_digest_algorithm(self.algorithm)
        _require_digest_value(self.algorithm, self.value)


@dataclass(frozen=True, kw_only=True)
class ConformanceResource:
    """A path-free content descriptor used for target sources and evidence."""

    name: str
    digests: tuple[ConformanceDigest, ...]
    media_type: str = ""
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        _require_safe_token(self.name, "resource name")
        digests = _typed_tuple(self.digests, ConformanceDigest, "resource digests")
        if not digests or any(not isinstance(item, ConformanceDigest) for item in digests):
            raise ValueError("resource digests must contain typed digest entries")
        algorithms = tuple(item.algorithm for item in digests)
        if len(set(algorithms)) != len(algorithms):
            raise ValueError("resource digest algorithms must be unique")
        if "sha256" not in algorithms:
            raise ValueError("resource digests must include sha256")
        if self.media_type:
            _require_media_type(self.media_type)
        if self.size_bytes is not None:
            _require_nonnegative_int(self.size_bytes, "resource size_bytes")
        object.__setattr__(self, "digests", tuple(sorted(digests, key=lambda item: item.algorithm)))

    def digest(self, algorithm: str) -> str | None:
        return next(
            (item.value for item in self.digests if item.algorithm == algorithm),
            None,
        )

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "digest": {item.algorithm: item.value for item in self.digests},
        }
        if self.media_type:
            payload["media_type"] = self.media_type
        if self.size_bytes is not None:
            payload["size_bytes"] = self.size_bytes
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> ConformanceResource:
        parsed = _closed_object(
            payload,
            "conformance resource",
            required={"name", "digest"},
            optional={"media_type", "size_bytes"},
        )
        digest_payload = _closed_mapping(parsed["digest"], "resource digest")
        digests = tuple(
            ConformanceDigest(
                algorithm=_required_string(algorithm, "digest algorithm"),
                value=_required_string(value, "digest value"),
            )
            for algorithm, value in digest_payload.items()
        )
        size_bytes = parsed.get("size_bytes")
        if size_bytes is not None:
            _require_nonnegative_int(size_bytes, "resource size_bytes")
        return cls(
            name=_required_string(parsed["name"], "resource name"),
            digests=digests,
            media_type=_optional_string(parsed.get("media_type"), "resource media_type"),
            size_bytes=size_bytes,
        )


@dataclass(frozen=True, kw_only=True)
class ConformanceTarget:
    """Sanitized identity for the tested implementation and trusted adapter."""

    implementation_id: str
    adapter_id: str
    implementation_version: str = ""
    adapter_version: str = ""
    source: ConformanceResource | None = None

    def __post_init__(self) -> None:
        _require_safe_token(self.implementation_id, "target implementation_id")
        _require_safe_token(self.adapter_id, "target adapter_id")
        if self.implementation_version:
            _require_safe_token(self.implementation_version, "target implementation_version")
        if self.adapter_version:
            _require_safe_token(self.adapter_version, "target adapter_version")
        if self.source is not None and not isinstance(self.source, ConformanceResource):
            raise ValueError("target source must be a typed resource")

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "implementation_id": self.implementation_id,
            "implementation_version": self.implementation_version,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
        }
        if self.source is not None:
            payload["source"] = self.source.to_json()
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> ConformanceTarget:
        parsed = _closed_object(
            payload,
            "conformance target",
            required={"implementation_id", "adapter_id"},
            optional={"implementation_version", "adapter_version", "source"},
        )
        source_payload = parsed.get("source")
        return cls(
            implementation_id=_required_string(
                parsed["implementation_id"], "target implementation_id"
            ),
            implementation_version=_optional_string(
                parsed.get("implementation_version"),
                "target implementation_version",
            ),
            adapter_id=_required_string(parsed["adapter_id"], "target adapter_id"),
            adapter_version=_optional_string(
                parsed.get("adapter_version"),
                "target adapter_version",
            ),
            source=(
                ConformanceResource.from_json(
                    _closed_mapping(source_payload, "target source")
                )
                if source_payload is not None
                else None
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ConformanceEvent:
    """One secret-minimized event identity retained as public evidence."""

    seq: int
    event_type: str

    def __post_init__(self) -> None:
        _require_positive_int(self.seq, "event seq")
        _require_optional_safe_token(self.event_type, "event type")

    def to_json(self) -> dict[str, Any]:
        return {"seq": self.seq, "type": self.event_type}

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> ConformanceEvent:
        parsed = _closed_object(
            payload,
            "conformance event",
            required={"seq", "type"},
        )
        _require_positive_int(parsed["seq"], "event seq")
        return cls(
            seq=parsed["seq"],
            event_type=_optional_string(parsed["type"], "event type"),
        )


@dataclass(frozen=True, kw_only=True)
class ConformanceEvidenceBundle:
    """A frozen minimal-lifecycle projection used to support conformance rules."""

    profile_id: str
    target: ConformanceTarget
    case_id_sha256: str
    run_id_present: bool
    submitted: bool
    states: tuple[str, ...]
    result_run_id_matches: bool
    result_status: str
    events: tuple[ConformanceEvent, ...]
    events_complete: bool
    next_seq: int

    def __post_init__(self) -> None:
        _require_safe_token(self.profile_id, "evidence profile_id")
        if not isinstance(self.target, ConformanceTarget):
            raise ValueError("evidence target must be typed target metadata")
        _require_sha256(self.case_id_sha256, "evidence case_id_sha256")
        _require_bool(self.run_id_present, "evidence run_id_present")
        _require_bool(self.submitted, "evidence submitted")
        states = _sequence_tuple(self.states, "evidence states")
        for state in states:
            _require_optional_safe_token(state, "evidence lifecycle state")
        _require_bool(self.result_run_id_matches, "evidence result_run_id_matches")
        _require_optional_safe_token(self.result_status, "evidence result_status")
        events = _typed_tuple(self.events, ConformanceEvent, "evidence events")
        if any(not isinstance(event, ConformanceEvent) for event in events):
            raise ValueError("evidence events must contain typed event entries")
        _require_bool(self.events_complete, "evidence events_complete")
        _require_nonnegative_int(self.next_seq, "evidence next_seq")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "events", events)

    @property
    def schema_version(self) -> str:
        return CONFORMANCE_EVIDENCE_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "target": self.target.to_json(),
            "case": {
                "case_id_sha256": self.case_id_sha256,
                "run_id_present": self.run_id_present,
                "submitted": self.submitted,
                "states": list(self.states),
                "result": {
                    "run_id_matches": self.result_run_id_matches,
                    "status": self.result_status,
                },
                "events": [event.to_json() for event in self.events],
                "events_complete": self.events_complete,
                "next_seq": self.next_seq,
            },
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> ConformanceEvidenceBundle:
        parsed = _closed_object(
            payload,
            "conformance evidence",
            required={"schema_version", "profile_id", "target", "case"},
        )
        if parsed["schema_version"] != CONFORMANCE_EVIDENCE_VERSION:
            raise ValueError("unsupported conformance evidence schema")
        case = _closed_object(
            _closed_mapping(parsed["case"], "conformance evidence case"),
            "conformance evidence case",
            required={
                "case_id_sha256",
                "run_id_present",
                "submitted",
                "states",
                "result",
                "events",
                "events_complete",
                "next_seq",
            },
        )
        result = _closed_object(
            _closed_mapping(case["result"], "conformance evidence result"),
            "conformance evidence result",
            required={"run_id_matches", "status"},
        )
        states = _required_list(case["states"], "evidence states")
        events = _required_list(case["events"], "evidence events")
        _require_bool(case["run_id_present"], "evidence run_id_present")
        _require_bool(case["submitted"], "evidence submitted")
        _require_bool(result["run_id_matches"], "evidence result run_id_matches")
        _require_bool(case["events_complete"], "evidence events_complete")
        _require_nonnegative_int(case["next_seq"], "evidence next_seq")
        return cls(
            profile_id=_required_string(parsed["profile_id"], "evidence profile_id"),
            target=ConformanceTarget.from_json(
                _closed_mapping(parsed["target"], "evidence target")
            ),
            case_id_sha256=_required_string(
                case["case_id_sha256"], "evidence case_id_sha256"
            ),
            run_id_present=case["run_id_present"],
            submitted=case["submitted"],
            states=tuple(
                _optional_string(state, "evidence lifecycle state") for state in states
            ),
            result_run_id_matches=result["run_id_matches"],
            result_status=_optional_string(result["status"], "evidence result status"),
            events=tuple(
                ConformanceEvent.from_json(
                    _closed_mapping(event, "conformance event")
                )
                for event in events
            ),
            events_complete=case["events_complete"],
            next_seq=case["next_seq"],
        )


@dataclass(frozen=True, kw_only=True)
class ConformanceEvidenceReference:
    """A report-facing content descriptor for one retained evidence artifact."""

    evidence_id: str
    kind: str
    schema_version: str
    resource: ConformanceResource
    record_count: int

    def __post_init__(self) -> None:
        _require_safe_token(self.evidence_id, "evidence reference id")
        if self.kind != CONFORMANCE_EVIDENCE_KIND:
            raise ValueError("unsupported conformance evidence kind")
        if self.schema_version != CONFORMANCE_EVIDENCE_VERSION:
            raise ValueError("unsupported conformance evidence reference schema")
        if not isinstance(self.resource, ConformanceResource):
            raise ValueError("evidence reference resource must be typed")
        if self.resource.media_type != CONFORMANCE_EVIDENCE_MEDIA_TYPE:
            raise ValueError("unsupported conformance evidence media type")
        if any(
            digest.algorithm not in _KNOWN_DIGEST_HEX_LENGTHS
            for digest in self.resource.digests
        ):
            raise ValueError("unsupported conformance evidence digest algorithm")
        if self.resource.size_bytes is None:
            raise ValueError("evidence reference requires size_bytes")
        _require_nonnegative_int(self.record_count, "evidence reference record_count")

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.evidence_id,
            "kind": self.kind,
            "schema_version": self.schema_version,
            **self.resource.to_json(),
            "record_count": self.record_count,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> ConformanceEvidenceReference:
        parsed = _closed_object(
            payload,
            "conformance evidence reference",
            required={
                "id",
                "kind",
                "schema_version",
                "name",
                "digest",
                "media_type",
                "size_bytes",
                "record_count",
            },
        )
        return cls(
            evidence_id=_required_string(parsed["id"], "evidence reference id"),
            kind=_required_string(parsed["kind"], "evidence reference kind"),
            schema_version=_required_string(
                parsed["schema_version"], "evidence reference schema_version"
            ),
            resource=ConformanceResource.from_json(
                {
                    "name": parsed["name"],
                    "digest": parsed["digest"],
                    "media_type": parsed["media_type"],
                    "size_bytes": parsed["size_bytes"],
                }
            ),
            record_count=parsed["record_count"],
        )


def case_id_sha256(run_id: str) -> str:
    """Return a domain-separated case fingerprint without retaining the raw run id."""

    if not isinstance(run_id, str):
        raise ValueError("run id must be a string")
    return hashlib.sha256(
        b"monoid.conformance.case.v1\0" + run_id.encode("utf-8")
    ).hexdigest()


def serialize_conformance_evidence(
    bundle: ConformanceEvidenceBundle,
    *,
    max_bytes: int = MAX_CONFORMANCE_EVIDENCE_BYTES,
) -> bytes:
    """Serialize the evidence using the v1 exact-byte form.

    The form is compact UTF-8 JSON with sorted ASCII schema keys and exactly one trailing LF.
    Consumers verify these bytes before parsing; this is deliberately narrower than claiming an
    RFC 8785 implementation.
    """

    if not isinstance(bundle, ConformanceEvidenceBundle):
        raise ValueError("conformance evidence bundle must be typed")
    _require_nonnegative_int(max_bytes, "evidence max_bytes")
    data = (
        json.dumps(
            bundle.to_json(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if len(data) > max_bytes:
        raise ValueError("conformance evidence exceeds the byte limit")
    return data


def build_evidence_reference(
    bundle: ConformanceEvidenceBundle,
    *,
    evidence_id: str,
    artifact_name: str,
) -> ConformanceEvidenceReference:
    """Describe the exact bytes produced for one evidence bundle."""

    data = serialize_conformance_evidence(bundle)
    return ConformanceEvidenceReference(
        evidence_id=evidence_id,
        kind=CONFORMANCE_EVIDENCE_KIND,
        schema_version=bundle.schema_version,
        resource=ConformanceResource(
            name=artifact_name,
            digests=(
                ConformanceDigest(
                    algorithm="sha256",
                    value=hashlib.sha256(data).hexdigest(),
                ),
            ),
            media_type=CONFORMANCE_EVIDENCE_MEDIA_TYPE,
            size_bytes=len(data),
        ),
        record_count=len(bundle.events),
    )


def verify_conformance_evidence(
    reference: ConformanceEvidenceReference,
    data: bytes,
    *,
    max_bytes: int = MAX_CONFORMANCE_EVIDENCE_BYTES,
) -> ConformanceEvidenceBundle:
    """Verify exact bytes, parse the closed schema, and return the frozen bundle."""

    if not isinstance(reference, ConformanceEvidenceReference):
        raise ValueError("conformance evidence reference must be typed")
    if not isinstance(data, bytes):
        raise ValueError("conformance evidence must be bytes")
    _require_nonnegative_int(max_bytes, "evidence max_bytes")
    if len(data) > max_bytes:
        raise ValueError("conformance evidence exceeds the byte limit")
    if reference.resource.size_bytes != len(data):
        raise ValueError("conformance evidence size mismatch")
    for digest in reference.resource.digests:
        if digest.algorithm == "sha256":
            actual = hashlib.sha256(data).hexdigest()
        elif digest.algorithm == "sha512":
            actual = hashlib.sha512(data).hexdigest()
        else:  # guarded by ConformanceEvidenceReference
            raise ValueError("unsupported conformance evidence digest algorithm")
        if not hmac.compare_digest(digest.value, actual):
            raise ValueError("conformance evidence digest mismatch")
    payload = _load_evidence_json(data)
    bundle = ConformanceEvidenceBundle.from_json(payload)
    if not hmac.compare_digest(
        data,
        serialize_conformance_evidence(bundle, max_bytes=max_bytes),
    ):
        raise ValueError("conformance evidence does not use the exact-byte form")
    if reference.schema_version != bundle.schema_version:
        raise ValueError("conformance evidence schema mismatch")
    if reference.record_count != len(bundle.events):
        raise ValueError("conformance evidence record count mismatch")
    return bundle


def _load_evidence_json(data: bytes) -> Mapping[str, Any]:
    try:
        text = data.decode("utf-8")
        payload = json.loads(text, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError("invalid conformance evidence JSON") from exc
    return _closed_mapping(payload, "conformance evidence")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("conformance evidence JSON contains duplicate keys")
        payload[key] = value
    return payload


def _closed_object(
    payload: Mapping[str, Any],
    label: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    parsed = dict(_closed_mapping(payload, label))
    allowed = required | (optional or set())
    if set(parsed) - allowed:
        raise ValueError(f"{label} contains unsupported fields")
    if required - set(parsed):
        raise ValueError(f"{label} is missing required fields")
    return parsed


def _closed_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return value


def _required_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _sequence_tuple(value: Any, label: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list or tuple")
    return tuple(value)


def _typed_tuple(value: Any, item_type: type[Any], label: str) -> tuple[Any, ...]:
    items = _sequence_tuple(value, label)
    if any(not isinstance(item, item_type) for item in items):
        raise ValueError(f"{label} contains invalid entries")
    return items


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_string(value: Any, label: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _require_safe_token(value: str, label: str) -> None:
    if not isinstance(value, str) or _SAFE_TOKEN_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a bounded ASCII token")


def _require_optional_safe_token(value: str, label: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if value and _SAFE_TOKEN_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be empty or a bounded ASCII token")


def _require_media_type(value: str) -> None:
    if not isinstance(value, str) or _MEDIA_TYPE_RE.fullmatch(value) is None:
        raise ValueError("resource media_type must be a bounded lowercase media type")


def _require_digest_algorithm(value: str) -> None:
    if not isinstance(value, str) or _DIGEST_ALGORITHM_RE.fullmatch(value) is None:
        raise ValueError("digest algorithm must be a bounded lowercase token")


def _require_digest_value(algorithm: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) > 256
        or len(value) % 2 != 0
        or _LOWER_HEX_RE.fullmatch(value) is None
    ):
        raise ValueError("digest value must be bounded lowercase hexadecimal")
    expected_length = _KNOWN_DIGEST_HEX_LENGTHS.get(algorithm)
    if expected_length is not None and len(value) != expected_length:
        raise ValueError(f"{algorithm} digest has an invalid hexadecimal length")


def _require_sha256(value: str, label: str) -> None:
    try:
        _require_digest_value("sha256", value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a sha256 digest") from exc


def _require_bool(value: Any, label: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")


def _require_positive_int(value: Any, label: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _require_nonnegative_int(value: Any, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


__all__ = [
    "CONFORMANCE_EVIDENCE_MEDIA_TYPE",
    "CONFORMANCE_EVIDENCE_KIND",
    "CONFORMANCE_EVIDENCE_VERSION",
    "MAX_CONFORMANCE_EVIDENCE_BYTES",
    "ConformanceDigest",
    "ConformanceEvent",
    "ConformanceEvidenceBundle",
    "ConformanceEvidenceReference",
    "ConformanceResource",
    "ConformanceTarget",
    "build_evidence_reference",
    "case_id_sha256",
    "serialize_conformance_evidence",
    "verify_conformance_evidence",
]
