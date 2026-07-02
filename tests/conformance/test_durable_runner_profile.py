from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from support.runtime import runtime_config, tool_binding
from support.waiting import eventually

from monoid_agent_kernel.conformance.profiles.durable_runner import (
    assert_durable_runner_event_sequence_profile,
    assert_durable_runner_recovery_metadata_profile,
    assert_durable_runner_subagent_diagnostics_profile,
)
from monoid_agent_kernel.core.agents import AgentRuntimeConfig, PromptSpec, SubagentDefinition
from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore
from monoid_agent_kernel.core.control import ControlCommand
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.backend.service import BackendRunRequest, RunnerBackend


class _ReferenceDurableRunnerHarness:
    def __init__(
        self,
        tmp_path: Path,
        *,
        workspace: Path | None = None,
        run_root: Path | None = None,
        checkpoint_store: LocalFsCheckpointStore | None = None,
        token_manager: TokenManager | None = None,
    ) -> None:
        self.tmp_path = tmp_path
        self.workspace = workspace or tmp_path / "workspace"
        self.workspace.mkdir(exist_ok=True)
        self.workspace.joinpath("notes.md").write_text("notes\n", encoding="utf-8")
        self.run_root = run_root or tmp_path / "runs"
        self.checkpoint_store = checkpoint_store or LocalFsCheckpointStore(tmp_path / "shared-checkpoints")
        self.token_manager = token_manager or TokenManager.from_secret("x" * 32)

        def factory(spec: Any, llm_gateway_token: str) -> FakeModelAdapter:
            del llm_gateway_token
            scenario = str(spec.metadata.get("scenario") or "")
            if scenario == "subagent-foreground":
                return FakeModelAdapter(
                    turns=[
                        ModelTurn(
                            tool_calls=(
                                fake_tool_call(
                                    "agent_spawn",
                                    {"subagent_type": "researcher", "prompt": "summarize notes"},
                                    "spawn_1",
                                ),
                            )
                        ),
                        ModelTurn(final_text="child found answer", usage={"total_tokens": 10}),
                        ModelTurn(final_text="parent done"),
                    ]
                )
            return FakeModelAdapter(turns=[ModelTurn(response_id=f"r_{uuid.uuid4().hex[:8]}", final_text="first")])

        self.backend = RunnerBackend(
            run_root=self.run_root,
            token_manager=self.token_manager,
            allowed_workspace_roots=(self.workspace,),
            llm_gateway_url="http://llm-gateway.internal/v1/turns",
            model_adapter_factory=factory,
            checkpoint_store=self.checkpoint_store,
            subagent_definitions={"researcher": _child_def()},
        )
        self.backend.idle_timeout_s = 30.0
        self.backend.max_recover_attempts = 10_000

    @property
    def harness_id(self) -> str:
        return "reference-durable-runner"

    @property
    def supported_profiles(self) -> tuple[str, ...]:
        return ("durable-runner",)

    def submit_run(self, request: dict[str, Any]) -> dict[str, Any]:
        scenario = str(request["scenario"])
        multi_turn = scenario in {"multi-turn", "recoverable-multi-turn"}
        config = (
            AgentRuntimeConfig(
                definition_id="parent",
                prompt=PromptSpec(persona_segments=("[[ROLE=PARENT]]",)),
                tools=(tool_binding("agent.spawn"), tool_binding("run.finish")),
            )
            if scenario == "subagent-foreground"
            else runtime_config("fs.read", "fs.write", "run.finish")
        )
        submission = self.backend.submit_run(
            BackendRunRequest(
                tenant_id="tenant_a",
                user_id="user_a",
                workspace_root=self.workspace,
                instruction=f"{scenario} run",
                runtime_config=config,
                multi_turn=multi_turn,
                metadata={"scenario": scenario},
            )
        )
        if scenario in {"completed", "subagent-foreground"}:
            assert self.backend.wait_for_run(submission.run_id, timeout_s=20) == "completed"
        elif multi_turn:
            assert eventually(lambda: self.backend._record(submission.run_id).status == "awaiting_input")
            assert eventually(lambda: self.checkpoint_store.latest(submission.run_id) is not None)
        else:
            raise AssertionError(f"unsupported durable runner scenario: {scenario}")
        return {"run_id": submission.run_id, "token": submission.run_token}

    def status(self, run_id: str, token: str) -> dict[str, Any]:
        return self.backend.status(run_id, token)

    def events(self, run_id: str, token: str, *, from_seq: int = 0, limit: int | None = None) -> dict[str, Any]:
        return self.backend.events(run_id, token, from_seq=from_seq, limit=limit)

    def descendant_events(
        self,
        run_id: str,
        token: str,
        descendant_run_id: str,
        *,
        from_seq: int = 0,
        limit: int | None = None,
    ) -> dict[str, Any]:
        return self.backend.descendant_events(
            run_id,
            token,
            descendant_run_id,
            from_seq=from_seq,
            limit=limit,
        )

    def diagnostics(self, run_id: str, token: str, *, event_limit: int = 50) -> dict[str, Any]:
        return self.backend.diagnostics(run_id, token, event_limit=event_limit)

    def result(self, run_id: str, token: str) -> dict[str, Any]:
        return self.backend.result(run_id, token)

    def runtime_config(self, run_id: str, token: str) -> dict[str, Any]:
        return self.backend.runtime_config(run_id, token)

    def replace_runtime_config(
        self,
        run_id: str,
        token: str,
        config: dict[str, Any],
        *,
        expected_version: int,
        issuer: str,
        reason: str,
    ) -> dict[str, Any]:
        return self.backend.replace_runtime_config(
            run_id,
            token,
            expected_version=expected_version,
            issuer=issuer,
            reason=reason,
            config=AgentRuntimeConfig.from_json(dict(config)),
        )

    def resume_run(self, run_id: str, token: str) -> dict[str, Any]:
        return self.backend.resume_run(run_id, token)

    def recover_runs(self) -> tuple[str, ...]:
        return tuple(self.backend.recover_runs())

    def restart(self, *, local_state: str = "same") -> _ReferenceDurableRunnerHarness:
        if local_state == "same":
            run_root = self.run_root
        elif local_state == "empty":
            run_root = self.tmp_path / f"empty-runs-{uuid.uuid4().hex[:8]}"
        else:
            raise AssertionError(f"unsupported restart local_state: {local_state}")
        return _ReferenceDurableRunnerHarness(
            self.tmp_path,
            workspace=self.workspace,
            run_root=run_root,
            checkpoint_store=self.checkpoint_store,
            token_manager=self.token_manager,
        )

    def task_result(self, run_id: str, token: str, task_id: str) -> dict[str, Any]:
        self.backend.status(run_id, token)
        task_path = self.backend._record(run_id).run_dir / "artifacts" / "tasks" / task_id / "task.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        return {"result": task["result"]}

    def dispatch(self, command: dict[str, Any]) -> dict[str, Any]:
        return self.backend.dispatch(ControlCommand.from_json(dict(command))).to_json()


def _child_def() -> SubagentDefinition:
    return SubagentDefinition(
        description="Researcher",
        prompt=PromptSpec(persona_segments=("[[ROLE=CHILD]]",)),
        tools=(),
    )


def test_reference_backend_satisfies_durable_runner_event_sequence_profile(tmp_path: Path) -> None:
    assert_durable_runner_event_sequence_profile(_ReferenceDurableRunnerHarness(tmp_path))


def test_reference_backend_satisfies_durable_runner_recovery_metadata_profile(tmp_path: Path) -> None:
    assert_durable_runner_recovery_metadata_profile(_ReferenceDurableRunnerHarness(tmp_path))


def test_reference_backend_satisfies_subagent_diagnostics_profile(tmp_path: Path) -> None:
    assert_durable_runner_subagent_diagnostics_profile(_ReferenceDurableRunnerHarness(tmp_path))
