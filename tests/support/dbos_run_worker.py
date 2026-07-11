"""Subprocess worker for DBOS run-lifecycle crash/restart acceptance tests."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from support.runtime import runtime_config, runtime_provider, tool_binding

from monoid_agent_kernel.core.agents import AgentRuntimeConfig
from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore
from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.errors import ModelAdapterError, NativeAgentError
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.reference.dbos import (
    DbosResumeCommand,
    DbosRunConfig,
    DbosRunDriver,
)
from monoid_agent_kernel.tools.base import ToolContext, ToolResult, ToolSpec

_RUN_ID = "run_dbos_resume"
_COMMAND_ID = "resume_once"
_APPLICATION_VERSION = "monoid-dbos-run-test-v1"
_EXECUTOR_ID = "stable-run-slot"


class _RetryableAdapter:
    def next_turn(self, request: Any) -> ModelTurn:
        del request
        raise ModelAdapterError(
            "retry after restart",
            error_code="provider_unavailable",
            retryable=True,
        )


class _EffectProvider:
    def __init__(
        self,
        effect_db: Path,
        run_id: str,
        *,
        committed_path: Path | None = None,
    ) -> None:
        self._effect_db = effect_db
        self._run_id = run_id
        self._committed_path = committed_path

    def get_tools(self, context: ToolContext | None = None) -> list[ToolSpec]:
        del context

        def apply_effect(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            del ctx
            key = str(args.get("idempotency_key") or "")
            with sqlite3.connect(self._effect_db) as connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS semantic_effects ("
                    "run_id TEXT NOT NULL, effect_key TEXT NOT NULL, "
                    "PRIMARY KEY (run_id, effect_key))"
                )
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO semantic_effects(run_id, effect_key) VALUES (?, ?)",
                    (self._run_id, key),
                )
                connection.commit()
            if self._committed_path is not None:
                self._committed_path.write_text("semantic effect committed", encoding="utf-8")
                while True:
                    time.sleep(0.1)
            return ToolResult(ok=True, content={"applied": cursor.rowcount == 1})

        return [
            ToolSpec(
                id="test.semantic_effect",
                description="Apply one externally visible idempotent effect.",
                input_schema={
                    "type": "object",
                    "properties": {"idempotency_key": {"type": "string"}},
                    "required": ["idempotency_key"],
                    "additionalProperties": False,
                },
                capability="",
                side_effect="write",
                annotations={
                    "external_side_effect": True,
                    "side_effect_delivery": "idempotent",
                    "idempotency_key_arg": "idempotency_key",
                },
                handler=apply_effect,
            )
        ]


def _runtime_config() -> AgentRuntimeConfig:
    base = runtime_config(
        bindings=(
            tool_binding(
                "test.semantic_effect",
                runtime={
                    "external_side_effect": True,
                    "side_effect_delivery": "idempotent",
                    "idempotency_key_arg": "idempotency_key",
                },
            ),
        )
    )
    return AgentRuntimeConfig(
        definition_id=base.definition_id,
        config_version=base.config_version,
        model=base.model,
        prompt=base.prompt,
        tools=base.tools,
        tool_search=base.tool_search,
        output_validators=base.output_validators,
        metadata={"tool_side_effect_policy": {"mode": "strict"}},
    )


def _driver_config(db_path: Path) -> DbosRunConfig:
    return DbosRunConfig(
        system_database_url=f"sqlite:///{db_path}",
        name="monoid-dbos-run-test",
        application_version=_APPLICATION_VERSION,
        executor_id=_EXECUTOR_ID,
        polling_interval_s=0.01,
    )


def seed(run_root: Path, workspace: Path, effect_db: Path, output_path: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    store = LocalFsCheckpointStore(run_root)
    loop = AgentLoop(
        spec=AgentRunSpec(
            run_id=_RUN_ID,
            workspace_root=workspace,
            run_root=run_root,
        ),
        model_adapter=_RetryableAdapter(),
        runtime_config_provider=runtime_provider(_runtime_config()),
        checkpoint_store=store,
        tool_providers=(_EffectProvider(effect_db, _RUN_ID),),
    )
    loop.open()
    suspension = loop.run_until_suspended("apply the effect after recovery")
    assert suspension.reason == "turn_failed"
    loop.release_parked()
    checkpoint = store.latest(_RUN_ID)
    assert checkpoint is not None
    _write_json(
        output_path,
        {
            "run_id": _RUN_ID,
            "command_id": _COMMAND_ID,
            "checkpoint_seq": checkpoint.seq,
            "suspension": checkpoint.checkpoint.last_suspension,
        },
    )


def _make_driver(
    db_path: Path,
    run_root: Path,
    workspace: Path,
    effect_db: Path,
    *,
    started_path: Path | None = None,
    fault_phase: str | None = None,
) -> DbosRunDriver:
    store = LocalFsCheckpointStore(run_root)

    def loop_factory(command: DbosResumeCommand) -> AgentLoop:
        adapter = FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="r_effect",
                    tool_calls=(
                        fake_tool_call(
                            "test_semantic_effect",
                            {"idempotency_key": command.command_id},
                            "call_effect",
                        ),
                    ),
                ),
                ModelTurn(response_id="r_done", final_text="resumed exactly once"),
            ]
        )
        return AgentLoop(
            spec=AgentRunSpec(
                run_id=command.run_id,
                workspace_root=workspace,
                run_root=run_root,
            ),
            model_adapter=adapter,
            runtime_config_provider=runtime_provider(_runtime_config()),
            tool_providers=(
                _EffectProvider(
                    effect_db,
                    command.run_id,
                    committed_path=(
                        started_path if fault_phase == "effect_committed" else None
                    ),
                ),
            ),
        )

    def fault_hook(phase: str, command: DbosResumeCommand) -> None:
        del command
        if (
            phase != "boundary_committed"
            or fault_phase != "boundary_committed"
            or started_path is None
        ):
            return
        started_path.write_text("boundary committed", encoding="utf-8")
        while True:
            time.sleep(0.1)

    return DbosRunDriver(
        _driver_config(db_path),
        store,
        loop_factory,
        fault_hook=fault_hook if fault_phase == "boundary_committed" else None,
    )


def crash(
    db_path: Path,
    run_root: Path,
    workspace: Path,
    effect_db: Path,
    started_path: Path,
    fault_phase: str,
) -> None:
    driver = _make_driver(
        db_path,
        run_root,
        workspace,
        effect_db,
        started_path=started_path,
        fault_phase=fault_phase,
    )
    driver.launch()
    command = DbosResumeCommand(_RUN_ID, _COMMAND_ID, 1)
    driver.enqueue_resume(command)
    driver.wait_for_receipt(command, timeout_s=60)


def recover(
    db_path: Path,
    run_root: Path,
    workspace: Path,
    effect_db: Path,
    output_path: Path,
) -> None:
    driver = _make_driver(db_path, run_root, workspace, effect_db)
    driver.launch()
    command = DbosResumeCommand(_RUN_ID, _COMMAND_ID, 1)
    completed = driver.wait_for_receipt(command, timeout_s=30)
    duplicate = driver.enqueue_resume(command)
    if duplicate.status == "pending":
        duplicate = driver.wait_for_receipt(command, timeout_s=30)
    stale_command = DbosResumeCommand(_RUN_ID, "resume_from_stale_checkpoint", 1)
    stale = driver.enqueue_resume(stale_command)
    if stale.status == "pending":
        stale = driver.wait_for_receipt(stale_command, timeout_s=30)
    conflict_code = ""
    try:
        driver.enqueue_resume(DbosResumeCommand(_RUN_ID, _COMMAND_ID, 2))
    except NativeAgentError as exc:
        conflict_code = exc.error_code

    store = LocalFsCheckpointStore(run_root)
    latest = store.latest(_RUN_ID)
    assert latest is not None
    with sqlite3.connect(effect_db) as connection:
        effect_count = connection.execute(
            "SELECT COUNT(*) FROM semantic_effects WHERE run_id = ? AND effect_key = ?",
            (_RUN_ID, _COMMAND_ID),
        ).fetchone()[0]
    workflow_id = DbosRunDriver.workflow_id(_RUN_ID, _COMMAND_ID)
    stale_workflow_id = DbosRunDriver.workflow_id(
        _RUN_ID,
        stale_command.command_id,
    )
    with sqlite3.connect(db_path) as connection:
        terminal_receipt_rows = connection.execute(
            "SELECT COUNT(*) FROM workflow_status WHERE workflow_uuid = ? AND status = 'SUCCESS'",
            (workflow_id,),
        ).fetchone()[0]
        stale_workflow_success_rows = connection.execute(
            "SELECT COUNT(*) FROM workflow_status WHERE workflow_uuid = ? AND status = 'SUCCESS'",
            (stale_workflow_id,),
        ).fetchone()[0]
    driver.close()
    _write_json(
        output_path,
        {
            "completed": completed.to_json(),
            "conflict_code": conflict_code,
            "duplicate": duplicate.to_json(),
            "effect_count": effect_count,
            "latest_seq": latest.seq,
            "marker_count": latest.checkpoint.applied_input_ids.count(command.checkpoint_marker),
            "stale": stale.to_json(),
            "stale_workflow_success_rows": stale_workflow_success_rows,
            "terminal_receipt_rows": terminal_receipt_rows,
        },
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seed", "crash", "recover"))
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--effect-db", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--started", type=Path)
    parser.add_argument(
        "--fault-phase",
        choices=("effect_committed", "boundary_committed"),
    )
    args = parser.parse_args()
    if args.mode == "seed":
        assert args.output is not None
        seed(args.run_root, args.workspace, args.effect_db, args.output)
    elif args.mode == "crash":
        assert args.started is not None
        assert args.fault_phase is not None
        crash(
            args.db,
            args.run_root,
            args.workspace,
            args.effect_db,
            args.started,
            args.fault_phase,
        )
    else:
        assert args.output is not None
        recover(args.db, args.run_root, args.workspace, args.effect_db, args.output)


if __name__ == "__main__":
    main()
