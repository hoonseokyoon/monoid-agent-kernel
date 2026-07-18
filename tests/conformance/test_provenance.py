from __future__ import annotations

import hashlib
import json

import pytest

from monoid_agent_kernel.conformance.provenance import (
    CONFORMANCE_EVIDENCE_KIND,
    CONFORMANCE_EVIDENCE_MEDIA_TYPE,
    CONFORMANCE_EVIDENCE_VERSION,
    ConformanceDigest,
    ConformanceEvent,
    ConformanceEvidenceBundle,
    ConformanceEvidenceReference,
    ConformanceResource,
    ConformanceTarget,
    build_evidence_reference,
    case_id_sha256,
    serialize_conformance_evidence,
    verify_conformance_evidence,
)


def _target(*, version: str = "1.2.3") -> ConformanceTarget:
    return ConformanceTarget(
        implementation_id="vendor.agent-runtime",
        implementation_version=version,
        adapter_id="vendor.monoid-adapter",
        adapter_version="2.0.0",
        source=ConformanceResource(
            name="agent-runtime.whl",
            digests=(
                ConformanceDigest(algorithm="sha256", value="a" * 64),
                ConformanceDigest(algorithm="sha512", value="b" * 128),
            ),
            media_type="application/zip",
            size_bytes=4096,
        ),
    )


def _bundle(
    *,
    target: ConformanceTarget | None = None,
    event_type: str = "run.started",
) -> ConformanceEvidenceBundle:
    return ConformanceEvidenceBundle(
        profile_id="minimal-agent",
        target=target or _target(),
        case_id_sha256=case_id_sha256("run_private_123"),
        run_id_present=True,
        submitted=True,
        states=("submitted", "running", "completed"),
        result_run_id_matches=True,
        result_status="completed",
        events=(
            ConformanceEvent(seq=1, event_type=event_type),
            ConformanceEvent(seq=2, event_type="run.finished"),
        ),
        events_complete=True,
        next_seq=3,
    )


def _reference_for_bytes(
    data: bytes,
    *,
    record_count: int = 2,
) -> ConformanceEvidenceReference:
    return ConformanceEvidenceReference(
        evidence_id="minimal-agent.lifecycle",
        kind=CONFORMANCE_EVIDENCE_KIND,
        schema_version=CONFORMANCE_EVIDENCE_VERSION,
        resource=ConformanceResource(
            name="minimal-agent.evidence.json",
            digests=(
                ConformanceDigest(
                    algorithm="sha256",
                    value=hashlib.sha256(data).hexdigest(),
                ),
            ),
            media_type=CONFORMANCE_EVIDENCE_MEDIA_TYPE,
            size_bytes=len(data),
        ),
        record_count=record_count,
    )


def test_exact_evidence_bytes_build_and_verify_reference() -> None:
    bundle = _bundle()
    data = serialize_conformance_evidence(bundle)
    reference = build_evidence_reference(
        bundle,
        evidence_id="minimal-agent.lifecycle",
        artifact_name="minimal-agent.evidence.json",
    )

    assert data.endswith(b"\n")
    assert not data.endswith(b"\n\n")
    assert reference.resource.size_bytes == len(data)
    assert reference.resource.digest("sha256") == hashlib.sha256(data).hexdigest()
    assert reference.record_count == 2
    assert verify_conformance_evidence(reference, data) == bundle
    assert ConformanceEvidenceReference.from_json(reference.to_json()) == reference
    assert ConformanceEvidenceBundle.from_json(bundle.to_json()) == bundle


def test_exact_serialization_is_stable_across_digest_input_order() -> None:
    first = _target()
    source = first.source
    assert source is not None
    reversed_source = ConformanceResource(
        name=source.name,
        digests=tuple(reversed(source.digests)),
        media_type=source.media_type,
        size_bytes=source.size_bytes,
    )
    second = ConformanceTarget(
        implementation_id=first.implementation_id,
        implementation_version=first.implementation_version,
        adapter_id=first.adapter_id,
        adapter_version=first.adapter_version,
        source=reversed_source,
    )

    assert serialize_conformance_evidence(_bundle(target=first)) == (
        serialize_conformance_evidence(_bundle(target=second))
    )


def test_digest_binds_target_case_and_event_content() -> None:
    base = build_evidence_reference(
        _bundle(),
        evidence_id="minimal-agent.lifecycle",
        artifact_name="minimal-agent.evidence.json",
    ).resource.digest("sha256")
    changed_target = build_evidence_reference(
        _bundle(target=_target(version="1.2.4")),
        evidence_id="minimal-agent.lifecycle",
        artifact_name="minimal-agent.evidence.json",
    ).resource.digest("sha256")
    changed_event = build_evidence_reference(
        _bundle(event_type="run.changed"),
        evidence_id="minimal-agent.lifecycle",
        artifact_name="minimal-agent.evidence.json",
    ).resource.digest("sha256")

    assert len({base, changed_target, changed_event}) == 3
    assert "run_private_123" not in serialize_conformance_evidence(_bundle()).decode()


def test_verifier_rejects_tampering_size_and_record_count() -> None:
    bundle = _bundle()
    data = serialize_conformance_evidence(bundle)
    reference = build_evidence_reference(
        bundle,
        evidence_id="minimal-agent.lifecycle",
        artifact_name="minimal-agent.evidence.json",
    )

    with pytest.raises(ValueError, match="size mismatch"):
        verify_conformance_evidence(reference, data + b" ")
    tampered = data.replace(b"run.started", b"run.changed")
    assert len(tampered) == len(data)
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_conformance_evidence(reference, tampered)
    with pytest.raises(ValueError, match="record count mismatch"):
        verify_conformance_evidence(_reference_for_bytes(data, record_count=1), data)


def test_verifier_checks_every_recognized_digest() -> None:
    data = serialize_conformance_evidence(_bundle())
    reference = ConformanceEvidenceReference(
        evidence_id="minimal-agent.lifecycle",
        kind=CONFORMANCE_EVIDENCE_KIND,
        schema_version=CONFORMANCE_EVIDENCE_VERSION,
        resource=ConformanceResource(
            name="minimal-agent.evidence.json",
            digests=(
                ConformanceDigest(
                    algorithm="sha256",
                    value=hashlib.sha256(data).hexdigest(),
                ),
                ConformanceDigest(algorithm="sha512", value="0" * 128),
            ),
            media_type=CONFORMANCE_EVIDENCE_MEDIA_TYPE,
            size_bytes=len(data),
        ),
        record_count=2,
    )

    with pytest.raises(ValueError, match="digest mismatch"):
        verify_conformance_evidence(reference, data)


def test_evidence_reference_rejects_unknown_digest_claims() -> None:
    data = serialize_conformance_evidence(_bundle())

    with pytest.raises(ValueError, match="digest algorithm"):
        ConformanceEvidenceReference(
            evidence_id="minimal-agent.lifecycle",
            kind=CONFORMANCE_EVIDENCE_KIND,
            schema_version=CONFORMANCE_EVIDENCE_VERSION,
            resource=ConformanceResource(
                name="minimal-agent.evidence.json",
                digests=(
                    ConformanceDigest(
                        algorithm="sha256",
                        value=hashlib.sha256(data).hexdigest(),
                    ),
                    ConformanceDigest(algorithm="blake3", value="0" * 64),
                ),
                media_type=CONFORMANCE_EVIDENCE_MEDIA_TYPE,
                size_bytes=len(data),
            ),
            record_count=2,
        )


def test_verifier_hashes_original_bytes_before_enforcing_exact_form() -> None:
    canonical = serialize_conformance_evidence(_bundle())
    alternate = json.dumps(_bundle().to_json(), indent=2, sort_keys=False).encode() + b"\r\n"
    reference = _reference_for_bytes(alternate)

    assert hashlib.sha256(alternate).hexdigest() == reference.resource.digest("sha256")
    with pytest.raises(ValueError, match="exact-byte form"):
        verify_conformance_evidence(reference, alternate)
    assert verify_conformance_evidence(_reference_for_bytes(canonical), canonical) == _bundle()


def test_writer_and_reference_reject_unsupported_bounds_or_kind() -> None:
    with pytest.raises(ValueError, match="byte limit"):
        serialize_conformance_evidence(_bundle(), max_bytes=1)
    with pytest.raises(ValueError, match="evidence kind"):
        ConformanceEvidenceReference(
            evidence_id="minimal-agent.lifecycle",
            kind="arbitrary",
            schema_version=CONFORMANCE_EVIDENCE_VERSION,
            resource=_reference_for_bytes(
                serialize_conformance_evidence(_bundle())
            ).resource,
            record_count=2,
        )


def test_verifier_rejects_duplicate_json_keys_after_digest_verification() -> None:
    data = serialize_conformance_evidence(_bundle())
    duplicate = data.replace(
        b'"profile_id":"minimal-agent"',
        b'"profile_id":"minimal-agent","profile_id":"minimal-agent"',
        1,
    )

    with pytest.raises(ValueError, match="invalid conformance evidence JSON"):
        verify_conformance_evidence(_reference_for_bytes(duplicate), duplicate)


def test_closed_schemas_reject_secret_shaped_extra_fields_without_echoing_values() -> None:
    secret = "Authorization: Bearer provenance-secret"
    payload = _bundle().to_json()
    payload["target"]["provider_error"] = secret

    with pytest.raises(ValueError) as captured:
        ConformanceEvidenceBundle.from_json(payload)

    assert secret not in str(captured.value)


def test_frozen_types_detach_from_caller_sequences() -> None:
    states = ["submitted", "completed"]
    events = [ConformanceEvent(seq=1, event_type="run.finished")]
    digests = [ConformanceDigest(algorithm="sha256", value="c" * 64)]
    resource = ConformanceResource(name="runtime.whl", digests=digests)  # type: ignore[arg-type]
    bundle = ConformanceEvidenceBundle(
        profile_id="minimal-agent",
        target=ConformanceTarget(
            implementation_id="vendor.runtime",
            adapter_id="vendor.adapter",
            source=resource,
        ),
        case_id_sha256=case_id_sha256("run_1"),
        run_id_present=True,
        submitted=True,
        states=states,  # type: ignore[arg-type]
        result_run_id_matches=True,
        result_status="completed",
        events=events,  # type: ignore[arg-type]
        events_complete=True,
        next_seq=2,
    )

    states.append("secret")
    events.append(ConformanceEvent(seq=2, event_type="secret"))
    digests.append(ConformanceDigest(algorithm="sha512", value="d" * 128))

    assert bundle.states == ("submitted", "completed")
    assert tuple(event.seq for event in bundle.events) == (1,)
    assert tuple(item.algorithm for item in resource.digests) == ("sha256",)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: ConformanceDigest(algorithm="sha256", value="A" * 64),
            "lowercase hexadecimal",
        ),
        (
            lambda: ConformanceEvent(seq=True, event_type="run.started"),
            "positive integer",
        ),
        (
            lambda: ConformanceTarget(
                implementation_id="vendor.runtime\nsecret",
                adapter_id="vendor.adapter",
            ),
            "bounded ASCII token",
        ),
        (
            lambda: ConformanceResource(
                name="runtime.whl",
                digests=(
                    ConformanceDigest(algorithm="sha256", value="a" * 64),
                    ConformanceDigest(algorithm="sha256", value="b" * 64),
                ),
            ),
            "unique",
        ),
        (
            lambda: ConformanceEvidenceBundle(
                profile_id="minimal-agent",
                target=_target(),
                case_id_sha256=case_id_sha256("run_1"),
                run_id_present=True,
                submitted=True,
                states="running",  # type: ignore[arg-type]
                result_run_id_matches=True,
                result_status="completed",
                events=(),
                events_complete=True,
                next_seq=0,
            ),
            "list or tuple",
        ),
        (
            lambda: ConformanceResource(
                name="runtime.whl",
                digests=(ConformanceDigest(algorithm="sha256", value="a" * 64),),
                media_type=f"{'a' * 128}/json",
            ),
            "bounded lowercase media type",
        ),
    ],
)
def test_primitives_reject_ambiguous_or_unsafe_values(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]
