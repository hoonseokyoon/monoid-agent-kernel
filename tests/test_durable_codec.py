from __future__ import annotations

import copy

from monoid_agent_kernel.core.durable_codec import DurableCodec, parse_artifact_version


def test_artifact_version_parser_accepts_canonical_and_legacy_namespaces() -> None:
    current = parse_artifact_version("monoid.checkpoint.v2")
    legacy = parse_artifact_version("native-agent-runner.backend-run.v1")

    assert current is not None and (current.namespace, current.family, current.version) == (
        "monoid",
        "checkpoint",
        2,
    )
    assert legacy is not None and (legacy.namespace, legacy.family, legacy.version) == (
        "native-agent-runner",
        "backend-run",
        1,
    )
    assert parse_artifact_version("monoid.checkpoint.v0") is None
    assert parse_artifact_version("checkpoint-v1") is None


def test_codec_distinguishes_loaded_corrupt_and_unsupported() -> None:
    codec = DurableCodec[dict](family="demo", current_schema="monoid.demo.v1")

    loaded = codec.decode({"schema_version": "monoid.demo.v1", "value": 1}, dict)
    legacy = codec.decode({"schema_version": "native-agent-runner.demo.v1", "value": 1}, dict)
    corrupt = codec.decode({"schema_version": "bad"}, dict)
    future = codec.decode({"schema_version": "monoid.demo.v2"}, dict)

    assert loaded.status == "loaded" and loaded.value == {"schema_version": "monoid.demo.v1", "value": 1}
    assert legacy.status == "loaded" and legacy.observed_schema == "native-agent-runner.demo.v1"
    assert corrupt.status == "corrupt" and corrupt.error_code == "demo_corrupt"
    assert future.status == "unsupported_version" and future.error_code == "demo_unsupported_version"


def test_ordered_migrations_are_pure_deterministic_and_preserve_unknown_fields() -> None:
    def v1_to_v2(payload: dict) -> dict:
        payload["second"] = payload.pop("first")
        return payload

    def v2_to_v3(payload: dict) -> dict:
        payload["third"] = payload.pop("second")
        return payload

    codec = DurableCodec[dict](
        family="demo",
        current_schema="monoid.demo.v3",
        migrations={1: v1_to_v2, 2: v2_to_v3},
    )
    source = {"schema_version": "monoid.demo.v1", "first": "x", "unknown": {"keep": True}}
    original = copy.deepcopy(source)

    first = codec.decode(source, dict)
    second = codec.decode(source, dict)

    assert source == original
    assert first == second
    assert first.status == "migrated"
    assert first.migrations == ("v1->v2", "v2->v3")
    assert first.value == {
        "schema_version": "monoid.demo.v3",
        "third": "x",
        "unknown": {"keep": True},
    }
    assert codec.decode(first.value, dict).status == "loaded"


def test_migration_failure_is_corrupt_and_does_not_mutate_source() -> None:
    def broken(payload: dict) -> dict:
        payload["secret"] = "changed"
        raise RuntimeError("payload content must not escape")

    codec = DurableCodec[dict](
        family="demo",
        current_schema="monoid.demo.v2",
        migrations={1: broken},
    )
    source = {"schema_version": "monoid.demo.v1", "secret": "original"}

    result = codec.decode(source, dict)

    assert source["secret"] == "original"
    assert result.status == "corrupt"
    assert "payload content" not in result.message
