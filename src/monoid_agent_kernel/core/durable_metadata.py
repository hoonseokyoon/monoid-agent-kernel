"""Helpers for durable backend recovery metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from monoid_agent_kernel.core._util import write_json_atomic
from monoid_agent_kernel.core.agents import AgentRuntimeConfig
from monoid_agent_kernel.identifiers import accepted_namespaced_ids, namespaced_id

RUN_METADATA_SCHEMA_VERSION = namespaced_id("backend-run.v1")
ACCEPTED_RUN_METADATA_SCHEMA_VERSIONS = accepted_namespaced_ids("backend-run.v1")
RUN_METADATA_FILENAME = "run.json"


class RunMetadataStore(Protocol):
    """Shared durable metadata operations exposed by checkpoint stores."""

    def put_run_metadata(self, run_id: str, metadata: Mapping[str, Any]) -> None: ...

    def run_metadata(self, run_id: str) -> dict[str, Any] | None: ...


def validate_run_metadata(payload: Any) -> dict[str, Any] | None:
    """Return a copy of supported run recovery metadata."""
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") not in ACCEPTED_RUN_METADATA_SCHEMA_VERSIONS:
        return None
    return dict(payload)


def read_run_metadata(run_dir: Path) -> dict[str, Any] | None:
    """Read local run recovery metadata if it is present and schema-compatible."""
    try:
        payload = json.loads((run_dir / RUN_METADATA_FILENAME).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return None
    return validate_run_metadata(payload)


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

    def write_initial_metadata(self, run_dir: Path, run_id: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
        meta = dict(metadata)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(run_dir / RUN_METADATA_FILENAME, meta)
        self.store_shared_metadata(run_id, meta)
        return meta

    def commit_metadata_update(self, run_dir: Path, run_id: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
        meta = dict(metadata)
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
        meta = read_run_metadata(run_dir)
        if meta is None:
            raise ValueError("run metadata is not ready")
        meta["runtime_config"] = config.to_json()
        meta["runtime_config_version"] = config.config_version
        meta["runtime_config_hash"] = config.config_hash
        meta["runtime_config_issuer"] = issuer
        meta["runtime_config_reason"] = reason
        meta["runtime_config_committed_at"] = committed_at
        return self.commit_metadata_update(run_dir, run_id, meta)

    def store_shared_metadata(self, run_id: str, metadata: Mapping[str, Any]) -> None:
        if self.checkpoint_store is None:
            return
        put_metadata = getattr(self.checkpoint_store, "put_run_metadata", None)
        if callable(put_metadata):
            put_metadata(run_id, dict(metadata))

    def read_recovery_metadata(self, run_dir: Path, run_id: str) -> dict[str, Any] | None:
        meta = read_run_metadata(run_dir)
        if meta is not None or self.checkpoint_store is None:
            return meta
        read_metadata = getattr(self.checkpoint_store, "run_metadata", None)
        stored = read_metadata(run_id) if callable(read_metadata) else None
        meta = validate_run_metadata(stored)
        if meta is None:
            return None
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(run_dir / RUN_METADATA_FILENAME, meta)
        return meta
