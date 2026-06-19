"""Unit tests for the non-blocking pump (AgentLoop.run_until_suspended)."""

from __future__ import annotations

from pathlib import Path

from conftest import runtime_config, runtime_provider

from native_agent_runner.core.spec import AgentRunSpec, RunLimits
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call


def _loop(tmp_path: Path, adapter: FakeModelAdapter, *tool_ids: str, limits: RunLimits | None = None) -> AgentLoop:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs", limits=limits or RunLimits())
    return AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config(*(tool_ids or ("fs.write",)))),
    )


def test_run_until_suspended_settles(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="done")])
    loop = _loop(tmp_path, adapter)
    loop.open()
    s = loop.run_until_suspended("hello")
    loop.close()

    assert s.reason == "settled"
    assert s.has_external is False
    assert s.awaiting_task_ids == ()
    assert s.turn is not None
    assert s.turn.status == "completed"
    assert s.turn.final_text == "done"


def test_run_until_suspended_parks_on_external_task_then_resumes(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(
        turns=[ModelTurn(response_id="r1", tool_calls=(fake_tool_call("hitl_request", {"prompt": "Pick"}, "c1"),))]
    )
    loop = _loop(tmp_path, adapter, "hitl.request")
    loop.open()

    # First pump parks on the hitl task WITHOUT blocking.
    s1 = loop.run_until_suspended("ask the human")
    assert s1.reason == "awaiting_tasks"
    assert s1.has_external is True
    assert len(s1.awaiting_task_ids) == 1
    assert s1.turn is None

    # Deliver the answer out of band, then resume.
    loop.report_task_result(s1.awaiting_task_ids[0], {"answer": "Ada"})
    s2 = loop.run_until_suspended(None)
    loop.close()

    assert s2.reason == "settled"
    assert s2.turn is not None
    hitl_obs = [
        obs for request in adapter.requests for obs in request.observations if obs.tool_name == "human_input"
    ]
    assert hitl_obs and hitl_obs[0].output["answer"] == "Ada"


def test_run_until_suspended_reports_limited(tmp_path: Path) -> None:
    # A tool call with a 1-step budget never settles -> limited.
    adapter = FakeModelAdapter(
        turns=[ModelTurn(response_id="r1", tool_calls=(fake_tool_call("fs_write", {"path": "A.md", "content": "a"}, "c1"),))]
    )
    loop = _loop(tmp_path, adapter, "fs.write", limits=RunLimits(max_steps=1))
    loop.open()
    s = loop.run_until_suspended("go")
    loop.close()

    assert s.reason == "limited"
    assert s.error_code == "max_steps_exceeded"
    assert s.turn is not None
    assert s.turn.status == "limited"
