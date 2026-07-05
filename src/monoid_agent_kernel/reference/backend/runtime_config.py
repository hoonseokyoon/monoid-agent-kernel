from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from monoid_agent_kernel.core.agents import AgentRuntimeConfig, validate_runtime_config
from monoid_agent_kernel.core.checkpoint import CheckpointStore
from monoid_agent_kernel.core.durable_metadata import DurableMetadataCommitter, runtime_config_from_metadata
from monoid_agent_kernel.reference.backend.ports import RunRecordPort
from monoid_agent_kernel.reference.backend.run_state import record_terminal as _record_terminal


def runtime_config_from_meta(meta: Mapping[str, Any]) -> AgentRuntimeConfig:
    return runtime_config_from_metadata(meta)


@dataclass(frozen=True)
class RuntimeConfigContext:
    authorize_run: Callable[[str, str], None]
    record: Callable[[str], RunRecordPort]
    with_record_lock: Callable[[Callable[[], Any]], Any]
    checkpoint_store_provider: Callable[[], CheckpointStore | None]
    builtin_tool_specs_provider: Callable[[], tuple[Any, ...]]
    now: Callable[[], float] = time.time


class RuntimeConfigService:
    """Runtime config projection and hot-swap operations for the Reference backend."""

    def __init__(self, context: RuntimeConfigContext) -> None:
        self._context = context

    def current_runtime_config(self, run_id: str) -> AgentRuntimeConfig | None:
        record = self._context.record(run_id)

        def _read() -> AgentRuntimeConfig | None:
            return record.runtime_config

        return self._context.with_record_lock(_read)

    def runtime_config(self, run_id: str, token: str) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        record = self._context.record(run_id)

        def _read() -> dict[str, Any]:
            config = record.runtime_config
            if config is None:
                return {
                    "run_id": record.run_id,
                    "tenant_id": record.tenant_id,
                    "ready": False,
                    "config_version": 0,
                    "config_hash": "",
                    "issuer": record.runtime_config_issuer,
                    "reason": record.runtime_config_reason,
                    "committed_at": record.runtime_config_committed_at,
                }
            return {
                "run_id": record.run_id,
                "tenant_id": record.tenant_id,
                "ready": True,
                "issuer": record.runtime_config_issuer,
                "reason": record.runtime_config_reason,
                "committed_at": record.runtime_config_committed_at,
                "config": config.to_json(),
                "config_version": config.config_version,
                "config_hash": config.config_hash,
            }

        return self._context.with_record_lock(_read)

    def replace_runtime_config(
        self,
        run_id: str,
        token: str,
        *,
        expected_version: int,
        issuer: str,
        reason: str,
        config: AgentRuntimeConfig,
    ) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        validate_runtime_config(config, self._context.builtin_tool_specs_provider())
        record = self._context.record(run_id)

        def _replace() -> dict[str, Any]:
            nonlocal config
            if _record_terminal(record):
                raise ValueError("cannot update runtime config for a terminal run")
            current_version = record.runtime_config.config_version if record.runtime_config else 0
            if expected_version != current_version:
                raise ValueError(
                    f"runtime config version mismatch: expected {expected_version}, current {current_version}"
                )
            if config.config_version <= current_version:
                # replace() copies all fields, so a new config field cannot be silently dropped.
                config = replace(config, config_version=current_version + 1)
            committed_at = self._context.now()
            self.write_runtime_config_run_meta(
                record,
                config,
                issuer=issuer,
                reason=reason,
                committed_at=committed_at,
            )
            record.runtime_config = config
            record.runtime_config_issuer = issuer
            record.runtime_config_reason = reason
            record.runtime_config_committed_at = committed_at
            return {
                "run_id": record.run_id,
                "tenant_id": record.tenant_id,
                "ready": True,
                "issuer": issuer,
                "reason": reason,
                "committed_at": committed_at,
                "config": config.to_json(),
                "config_version": config.config_version,
                "config_hash": config.config_hash,
            }

        return self._context.with_record_lock(_replace)

    def write_runtime_config_run_meta(
        self,
        record: RunRecordPort,
        config: AgentRuntimeConfig,
        *,
        issuer: str,
        reason: str,
        committed_at: float,
    ) -> None:
        DurableMetadataCommitter(self._checkpoint_store()).commit_runtime_config_update(
            record.run_dir,
            record.run_id,
            config,
            issuer=issuer,
            reason=reason,
            committed_at=committed_at,
        )

    def _checkpoint_store(self) -> CheckpointStore:
        checkpoint_store = self._context.checkpoint_store_provider()
        assert checkpoint_store is not None
        return checkpoint_store
