"""Human-in-the-loop in-process PoC.

The agent calls ``hitl.request``; the run parks waiting for a human answer that
arrives on another thread via ``report_task_result``; the answer is injected as a
user message and the model continues.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from conftest import runtime_config, runtime_provider

from native_agent_runner.core.spec import AgentRunSpec
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call


def _build_loop(tmp_path: Path, adapter: FakeModelAdapter) -> AgentLoop:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("hitl.request")),
    )


def _answer_when_parked(loop: AgentLoop, manager, answer: str, captured: dict) -> None:
    for _ in range(400):
        pending = [t for t in manager.jobs.values() if t.kind == "hitl" and t.status == "running"]
        if pending:
            captured["task_id"] = pending[0].job_id
            loop.report_task_result(pending[0].job_id, {"answer": answer})
            return
        time.sleep(0.01)


def test_hitl_request_parks_and_resumes_with_user_message(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("hitl_request", {"prompt": "Pick a name"}, "c1"),)),
            ModelTurn(response_id="r2"),  # no tool calls -> the run parks on the hitl task
            ModelTurn(response_id="r3", final_text="thanks"),
        ]
    )
    loop = _build_loop(tmp_path, adapter)
    loop.open()
    manager = loop._session.res.context.job_manager  # type: ignore[union-attr]

    captured: dict = {}
    responder = threading.Thread(target=_answer_when_parked, args=(loop, manager, "Ada", captured))
    responder.start()
    turn = loop.submit("Name the project, asking me if unsure.")
    responder.join(timeout=10)
    result = loop.close()

    assert captured.get("task_id"), "responder never observed the parked hitl task"
    assert turn.status == "completed"
    assert turn.final_text == "thanks"

    # The human answer was injected as a user message (is_background=True) carrying
    # the answer, and reached the model on a later turn.
    hitl_obs = [
        obs
        for request in adapter.requests
        for obs in request.observations
        if obs.tool_name == "human_input"
    ]
    assert hitl_obs, "the hitl answer was never delivered to the model"
    assert hitl_obs[0].is_background is True
    assert hitl_obs[0].output["answer"] == "Ada"
    assert result.status == "completed"


def test_hitl_answer_can_be_delivered_as_tool_result(tmp_path: Path) -> None:
    # Flip the injector to deliver the answer as a tool result instead of a user
    # message (both shapes are supported; the backend chooses per kind).
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("hitl_request", {"prompt": "Approve?"}, "c1"),)),
            ModelTurn(response_id="r2"),
            ModelTurn(response_id="r3", final_text="done"),
        ]
    )
    loop = _build_loop(tmp_path, adapter)
    loop.open()
    manager = loop._session.res.context.job_manager  # type: ignore[union-attr]
    manager.injectors["hitl"].as_user_message = False

    captured: dict = {}
    responder = threading.Thread(target=_answer_when_parked, args=(loop, manager, "yes", captured))
    responder.start()
    loop.submit("Approve the plan, ask me first.")
    responder.join(timeout=10)
    loop.close()

    hitl_obs = [
        obs
        for request in adapter.requests
        for obs in request.observations
        if obs.tool_name == "human_input"
    ]
    assert hitl_obs
    assert hitl_obs[0].is_background is False
    assert hitl_obs[0].output["answer"] == "yes"
