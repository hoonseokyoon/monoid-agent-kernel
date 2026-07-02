from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from support.runtime import runtime_config
from support.waiting import eventually

from monoid_agent_kernel.conformance.profiles.control_plane import assert_control_plane_decision_profile
from monoid_agent_kernel.core.control import ControlCommand
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.backend.service import BackendRunRequest, RunnerBackend


class _ReferenceControlPlaneHarness:
    def __init__(self, tmp_path: Path) -> None:
        self.workspace = tmp_path / "workspace"
        self.workspace.mkdir()
        self.workspace.joinpath("notes.md").write_text("notes\n", encoding="utf-8")

        def factory(spec: Any, llm_gateway_token: str) -> FakeModelAdapter:
            del spec, llm_gateway_token
            return FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="first")])

        self.backend = RunnerBackend(
            run_root=tmp_path / "runs",
            token_manager=TokenManager.from_secret("x" * 32),
            allowed_workspace_roots=(self.workspace,),
            llm_gateway_url="http://llm-gateway.internal/v1/turns",
            model_adapter_factory=factory,
        )
        self.backend.idle_timeout_s = 10.0

    @property
    def harness_id(self) -> str:
        return "reference-control-plane"

    @property
    def supported_profiles(self) -> tuple[str, ...]:
        return ("control-plane",)

    def submit_run(self, request: dict[str, Any]) -> dict[str, Any]:
        assert request["scenario"] == "parked-hitl"
        submission = self.backend.submit_run(
            BackendRunRequest(
                tenant_id="tenant_a",
                user_id="user_a",
                workspace_root=self.workspace,
                instruction="hello",
                runtime_config=runtime_config("fs.read", "fs.write", "run.finish"),
                multi_turn=True,
            )
        )
        assert eventually(lambda: self.backend._record(submission.run_id).status == "awaiting_input")
        return {"run_id": submission.run_id, "token": submission.run_token}

    def status(self, run_id: str, token: str) -> dict[str, Any]:
        return self.backend.status(run_id, token)

    def events(self, run_id: str, token: str, *, from_seq: int = 0, limit: int | None = None) -> dict[str, Any]:
        return self.backend.events(run_id, token, from_seq=from_seq, limit=limit)

    def diagnostics(self, run_id: str, token: str, *, event_limit: int = 50) -> dict[str, Any]:
        return self.backend.diagnostics(run_id, token, event_limit=event_limit)

    def task_result(self, run_id: str, token: str, task_id: str) -> dict[str, Any]:
        self.backend.status(run_id, token)
        task_path = self.backend._record(run_id).run_dir / "artifacts" / "tasks" / task_id / "task.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        return {"result": task["result"]}

    def dispatch(self, command: dict[str, Any]) -> dict[str, Any]:
        return self.backend.dispatch(ControlCommand.from_json(dict(command))).to_json()


def test_reference_backend_satisfies_control_plane_decision_profile(tmp_path: Path) -> None:
    assert_control_plane_decision_profile(_ReferenceControlPlaneHarness(tmp_path))
