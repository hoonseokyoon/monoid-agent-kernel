"""Helpers for durable backend recovery metadata."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from monoid_agent_kernel.core._util import write_json_atomic
from monoid_agent_kernel.core.agents import AgentRuntimeConfig
from monoid_agent_kernel.core.durable_codec import DurableCodec, DurableLoadResult
from monoid_agent_kernel.identifiers import accepted_namespaced_ids, namespaced_id

RUN_METADATA_SCHEMA_VERSION = namespaced_id("backend-run.v1")
ACCEPTED_RUN_METADATA_SCHEMA_VERSIONS = accepted_namespaced_ids("backend-run.v1")
RUN_METADATA_FILENAME = "run.json"


class RunMetadataStore(Protocol):
    """Shared durable metadata operations exposed by checkpoint stores."""

    def put_run_metadata(self, run_id: str, metadata: Mapping[str, Any]) -> None: ...

    def run_metadata(self, run_id: str) -> dict[str, Any] | None: ...


RUN_METADATA_CODEC = DurableCodec[dict[str, Any]](
    family="backend-run",
    current_schema=RUN_METADATA_SCHEMA_VERSION,
)


_RUN_METADATA_STRING_FIELDS = frozenset(
    {
        "tenant_id",
        "user_id",
        "workspace_root",
        "mode",
        "workspace_backend",
        "title",
        "runtime_config_hash",
        "runtime_config_issuer",
        "runtime_config_reason",
    }
)


def _require_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"backend-run {field_name} must be a non-empty string")


def _require_nonnegative_int(value: object, field_name: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"backend-run {field_name} must be an integer >= {minimum}")


def _require_finite_nonnegative_number(value: object, field_name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"backend-run {field_name} must be a finite non-negative number")


def _run_metadata_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    _require_nonempty_string(payload.get("run_id"), "run_id")
    for field_name in _RUN_METADATA_STRING_FIELDS:
        if field_name in payload and not isinstance(payload[field_name], str):
            raise ValueError(f"backend-run {field_name} must be a string")
    if "multi_turn" in payload and not isinstance(payload["multi_turn"], bool):
        raise ValueError("backend-run multi_turn must be boolean")
    if "runtime_config_version" in payload:
        _require_nonnegative_int(payload["runtime_config_version"], "runtime_config_version")
    if "metadata_generation" in payload:
        _require_nonnegative_int(payload["metadata_generation"], "metadata_generation", minimum=1)
    for field_name in ("created_at", "runtime_config_committed_at"):
        if field_name in payload:
            _require_finite_nonnegative_number(payload[field_name], field_name)
    if (
        "runtime_config" in payload
        and payload["runtime_config"] is not None
        and not isinstance(payload["runtime_config"], dict)
    ):
        raise ValueError("backend-run runtime_config must be an object or null")
    if (
        "permission_policy" in payload
        and payload["permission_policy"] is not None
        and not isinstance(payload["permission_policy"], dict)
    ):
        raise ValueError("backend-run permission_policy must be an object or null")
    if "limits" in payload:
        limits = payload["limits"]
        if not isinstance(limits, dict):
            raise ValueError("backend-run limits must be an object")
        for field_name in ("max_steps", "max_tool_calls", "max_bytes_read"):
            if field_name in limits:
                _require_nonnegative_int(limits[field_name], f"limits.{field_name}")
        if "max_duration_s" in limits and limits["max_duration_s"] is not None:
            _require_finite_nonnegative_number(limits["max_duration_s"], "limits.max_duration_s")
    return dict(payload)


def decode_run_metadata(payload: object) -> DurableLoadResult[dict[str, Any]]:
    return RUN_METADATA_CODEC.decode(payload, _run_metadata_from_payload)


def bind_run_metadata_result(
    result: DurableLoadResult[dict[str, Any]], run_id: str
) -> DurableLoadResult[dict[str, Any]]:
    """Bind decoded metadata to the key/path used to look it up."""

    if not result.ok:
        return result
    metadata = result.value
    if not isinstance(metadata, dict) or metadata.get("run_id") != run_id:
        return RUN_METADATA_CODEC.corrupt(
            "backend-run metadata run_id does not match the requested run",
            observed_schema=result.observed_schema,
        )
    return result


def validate_recovery_metadata(
    metadata: Mapping[str, Any], *, expected_run_id: str
) -> dict[str, Any]:
    """Validate fields required to rebuild a Reference run after restart."""

    value = _run_metadata_from_payload(dict(metadata))
    if value["run_id"] != expected_run_id:
        raise ValueError("backend-run metadata run_id does not match the recovery run")
    for field_name in ("tenant_id", "user_id", "workspace_root"):
        _require_nonempty_string(value.get(field_name), field_name)
    if not isinstance(value.get("runtime_config"), dict):
        raise ValueError("backend-run runtime_config is required for recovery")
    runtime_config_from_metadata(value)
    return value


def validate_recovery_metadata_result(
    result: DurableLoadResult[dict[str, Any]], run_id: str
) -> DurableLoadResult[dict[str, Any]]:
    """Convert a structurally readable but non-recoverable descriptor to checked corrupt."""

    bound = bind_run_metadata_result(result, run_id)
    if not bound.ok:
        return bound
    assert bound.value is not None
    try:
        validate_recovery_metadata(bound.value, expected_run_id=run_id)
    except (TypeError, ValueError) as exc:
        return RUN_METADATA_CODEC.corrupt(
            f"backend-run recovery metadata validation failed ({exc})",
            observed_schema=bound.observed_schema,
        )
    return bound


def metadata_payload_for_write(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate accepted input and emit only the canonical current writer version."""
    decoded = decode_run_metadata(dict(metadata))
    if not decoded.ok or decoded.value is None:
        raise ValueError(decoded.message or "run metadata is invalid")
    canonical = dict(decoded.value)
    canonical["schema_version"] = RUN_METADATA_SCHEMA_VERSION
    return canonical


def validate_run_metadata(payload: Any) -> dict[str, Any] | None:
    """Compatibility wrapper returning supported run recovery metadata."""
    return decode_run_metadata(payload).value


def read_run_metadata_checked(
    run_dir: Path, *, expected_run_id: str | None = None
) -> DurableLoadResult[dict[str, Any]]:
    """Read local recovery metadata without collapsing bad state into missing."""
    path = run_dir / RUN_METADATA_FILENAME
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return RUN_METADATA_CODEC.missing()
    except OSError:
        return RUN_METADATA_CODEC.corrupt("backend-run metadata could not be read")
    try:
        payload = json.loads(raw)
    except ValueError:
        return RUN_METADATA_CODEC.corrupt("backend-run metadata is not valid JSON")
    decoded = decode_run_metadata(payload)
    return (
        bind_run_metadata_result(decoded, expected_run_id)
        if expected_run_id is not None
        else decoded
    )


def read_run_metadata(
    run_dir: Path, *, expected_run_id: str | None = None
) -> dict[str, Any] | None:
    """Read local run recovery metadata if it is present and schema-compatible."""
    return read_run_metadata_checked(run_dir, expected_run_id=expected_run_id).value


def runtime_config_from_metadata(meta: Mapping[str, Any]) -> AgentRuntimeConfig:
    """Rebuild and verify the runtime config carried by recovery metadata."""
    config_payload = meta.get("runtime_config")
    if not isinstance(config_payload, dict):
        raise ValueError("run metadata is missing runtime_config")
    config = AgentRuntimeConfig.from_json(config_payload)
    expected_hash = str(meta.get("runtime_config_hash") or config_payload.get("config_hash") or "")
    if expected_hash and expected_hash != config.config_hash:
        raise ValueError("runtime config hash mismatch in run metadata")
    return config


@dataclass(frozen=True)
class DurableMetadataCommitter:
    """Commit and read backend-owned recovery metadata."""

    checkpoint_store: Any | None = None

    def write_initial_metadata(
        self, run_dir: Path, run_id: str, metadata: Mapping[str, Any]
    ) -> dict[str, Any]:
        meta = metadata_payload_for_write(metadata)
        meta["metadata_generation"] = 1
        validate_recovery_metadata(meta, expected_run_id=run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(run_dir / RUN_METADATA_FILENAME, meta)
        self.store_shared_metadata(run_id, meta)
        return meta

    def commit_metadata_update(
        self, run_dir: Path, run_id: str, metadata: Mapping[str, Any]
    ) -> dict[str, Any]:
        current = self.read_recovery_metadata_checked(run_dir, run_id)
        if not current.ok or current.value is None:
            raise ValueError(current.message or "run metadata is not ready for update")
        current_generation = current.value.get("metadata_generation")
        generation = current_generation if isinstance(current_generation, int) else 0
        meta = metadata_payload_for_write(metadata)
        meta["metadata_generation"] = generation + 1
        validate_recovery_metadata(meta, expected_run_id=run_id)
        self.store_shared_metadata(run_id, meta)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(run_dir / RUN_METADATA_FILENAME, meta)
        return meta

    def commit_runtime_config_update(
        self,
        run_dir: Path,
        run_id: str,
        config: AgentRuntimeConfig,
        *,
        issuer: str,
        reason: str,
        committed_at: float,
    ) -> dict[str, Any]:
        checked = self.read_recovery_metadata_checked(run_dir, run_id)
        if not checked.ok or checked.value is None:
            raise ValueError(checked.message or "run metadata is not ready")
        meta = dict(checked.value)
        meta["runtime_config"] = config.to_json()
        meta["runtime_config_version"] = config.config_version
        meta["runtime_config_hash"] = config.config_hash
        meta["runtime_config_issuer"] = issuer
        meta["runtime_config_reason"] = reason
        meta["runtime_config_committed_at"] = committed_at
        return self.commit_metadata_update(run_dir, run_id, meta)

    def store_shared_metadata(self, run_id: str, metadata: Mapping[str, Any]) -> None:
        validate_recovery_metadata(metadata, expected_run_id=run_id)
        if self.checkpoint_store is None:
            return
        put_metadata = getattr(self.checkpoint_store, "put_run_metadata", None)
        if callable(put_metadata):
            put_metadata(run_id, dict(metadata))

    def read_recovery_metadata(self, run_dir: Path, run_id: str) -> dict[str, Any] | None:
        return self.read_recovery_metadata_checked(run_dir, run_id).value

    def read_recovery_metadata_checked(
        self, run_dir: Path, run_id: str
    ) -> DurableLoadResult[dict[str, Any]]:
        local = read_run_metadata_checked(run_dir, expected_run_id=run_id)
        if local.status not in {"loaded", "migrated", "missing"}:
            return local
        if self.checkpoint_store is None:
            return validate_recovery_metadata_result(local, run_id)
        checked_reader = getattr(self.checkpoint_store, "run_metadata_checked", None)
        if callable(checked_reader):
            shared = checked_reader(run_id)
            if not isinstance(shared, DurableLoadResult):
                shared = RUN_METADATA_CODEC.corrupt(
                    "metadata store returned an invalid checked result"
                )
        else:
            raw_reader = getattr(self.checkpoint_store, "run_metadata", None)
            # Legacy readers have no checked result that can prove an artifact is
            # corrupt. Let read failures propagate so recovery can retry them.
            stored = raw_reader(run_id) if callable(raw_reader) else None
            shared = RUN_METADATA_CODEC.missing() if stored is None else decode_run_metadata(stored)
        shared = bind_run_metadata_result(shared, run_id)
        if shared.status not in {"loaded", "migrated", "missing"}:
            return shared
        selected = local
        materialize_local = False
        materialize_shared = False
        if local.status == "missing":
            selected = shared
            materialize_local = shared.ok
        elif shared.status == "missing":
            selected = local
            materialize_shared = local.ok and local.value is not None and (
                local.value.get("metadata_generation") is not None
            )
        else:
            assert local.value is not None and shared.value is not None
            local_generation = local.value.get("metadata_generation")
            shared_generation = shared.value.get("metadata_generation")
            if local_generation is None and shared_generation is None:
                # Historical split-brain policy: before generations existed, the local
                # descriptor was authoritative whenever both copies were readable.
                selected = local
            elif local_generation == shared_generation:
                if local.value != shared.value:
                    return RUN_METADATA_CODEC.corrupt(
                        "backend-run metadata copies diverge at the same generation",
                        observed_schema=local.observed_schema,
                    )
                selected = local
            elif (shared_generation or 0) > (local_generation or 0):
                selected = shared
                materialize_local = True
            else:
                selected = local
                materialize_shared = True
        selected = validate_recovery_metadata_result(selected, run_id)
        if selected.ok and materialize_shared:
            assert selected.value is not None
            self.store_shared_metadata(run_id, selected.value)
        if selected.ok and materialize_local:
            assert selected.value is not None
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(run_dir / RUN_METADATA_FILENAME, selected.value)
        return selected
