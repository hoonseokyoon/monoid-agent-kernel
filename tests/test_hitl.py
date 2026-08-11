"""Human-in-the-loop in-process PoC.

The agent calls ``hitl.request``; the run parks waiting for a human answer that
arrives on another thread via ``report_task_result``; the answer is injected as a
user message and the model continues.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.errors import ToolExecutionError
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call


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
    # A single hitl.request turn: once the model has nothing else to do, the run
    # parks on the task until the human answers (FakeModelAdapter then settles).
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("hitl_request", {"prompt": "Pick a name"}, "c1"),)),
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


def test_report_result_is_idempotent_first_report_wins(tmp_path: Path) -> None:
    # A duplicate hosted-task result report (e.g. a callback retry) must be a safe no-op: it neither
    # clobbers the recorded result nor re-publishes to the reentry queue (which would make the agent
    # observe the result twice). Mirrors the inbox's dedup-by-id.
    loop = _build_loop(tmp_path, FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="x")]))
    loop.open()
    manager = loop._session.res.context.job_manager  # type: ignore[union-attr]
    task = manager.start_task("hitl", {"prompt": "Pick a name"})

    first = manager.report_result(task.job_id, {"answer": "Ada"})
    assert first["delivered"] is True and first["duplicate"] is False
    assert manager._reentry_queue.count(task.job_id) == 1
    assert task.result == {"answer": "Ada"} and task.status == "answered"

    # Re-report with a different (stale) result: rejected as a duplicate, state unchanged.
    second = manager.report_result(task.job_id, {"answer": "STALE"}, status="answered")
    assert second["delivered"] is False and second["duplicate"] is True
    assert task.result == {"answer": "Ada"}  # not clobbered
    assert manager._reentry_queue.count(task.job_id) == 1  # no double reentry

    loop.close()


def test_an_unportable_task_request_or_result_is_refused_at_ingress(tmp_path: Path) -> None:
    """The tool-result refusal's census twins ③ and ④: hosted-task payloads are Python objects too.

    ``start_task`` and ``report_result`` both take dicts that never crossed a JSON parse — an
    in-process reporter can hand them ``bytes`` or an integer past the portable digit bound, which
    the normalizer deliberately leaves alone and ``task.json``'s writer then cannot serialize. The
    refusal fires before any state moves *on the path that moves state*: a refused report leaves the
    task running, unclobbered, and a correct report afterwards still lands. A task that has already
    finished has no such writer ahead of it and is answered without judging the payload at all --
    the two pins below.
    """
    loop = _build_loop(tmp_path, FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="x")]))
    loop.open()
    manager = loop._session.res.context.job_manager  # type: ignore[union-attr]

    with pytest.raises(ToolExecutionError) as refused_request:
        manager.start_task("hitl", {"prompt": b"\x00"})
    assert refused_request.value.error_code == "task_request_unportable"

    task = manager.start_task("hitl", {"prompt": "Pick a name"})
    with pytest.raises(ToolExecutionError) as refused_result:
        manager.report_result(task.job_id, {"answer": 10**4700})
    assert refused_result.value.error_code == "task_result_unportable"
    assert task.result is None, "the refused report half-landed"
    assert task.status == "running"

    delivered = manager.report_result(task.job_id, {"answer": "Ada"})
    assert delivered["delivered"] is True

    loop.close()


def test_a_duplicate_report_is_a_no_op_even_when_its_body_is_unportable(tmp_path: Path) -> None:
    """A refusal protects a writer, and this path reaches none.

    The retry's body is not the reporter's choice -- it is whatever the first attempt sent, resent
    -- so judging it turns "first report wins" into an error for exactly the callers idempotency
    exists for. Nothing here is stored or published, so there is nothing for the refusal to defend:
    the no-op answers first.
    """
    loop = _build_loop(tmp_path, FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="x")]))
    loop.open()
    manager = loop._session.res.context.job_manager  # type: ignore[union-attr]
    task = manager.start_task("hitl", {"prompt": "Pick a name"})
    assert manager.report_result(task.job_id, {"answer": "Ada"})["delivered"] is True

    for stale in ({"answer": b"\x00"}, {"answer": 10**4700}, {"answer": object()}):
        duplicate = manager.report_result(task.job_id, stale)
        assert duplicate["duplicate"] is True and duplicate["delivered"] is False
        assert duplicate["status"] == "answered"

    assert task.result == {"answer": "Ada"}, "the refused duplicate clobbered the stored result"
    assert manager._reentry_queue.count(task.job_id) == 1, "the no-op re-published the task"

    loop.close()


def test_a_cancelled_task_answers_a_late_report_rather_than_judging_its_body(tmp_path: Path) -> None:
    """The already-finished branch is not reached only by duplicates.

    ``cancel`` sets ``finished_at`` and a cancelled result, so a reporter that was already working
    when the cancellation landed takes this branch on its **first and only** report. The answer it
    gets back -- ``status: "cancelled"`` -- is what tells it to stop; a reporter that decides that
    from "did this raise" instead never does.
    """
    loop = _build_loop(tmp_path, FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="x")]))
    loop.open()
    manager = loop._session.res.context.job_manager  # type: ignore[union-attr]
    task = manager.start_task("hitl", {"prompt": "Pick a name"})
    manager.cancel(task.job_id)
    cancelled_result = task.result
    assert task.finished_at is not None and cancelled_result is not None

    answered = manager.report_result(task.job_id, {"answer": b"\x00"})

    assert answered["duplicate"] is True and answered["delivered"] is False
    assert answered["status"] == "cancelled", "the late reporter never learns why it was refused"
    assert task.result == cancelled_result

    loop.close()


def test_hitl_answer_can_be_delivered_as_tool_result(tmp_path: Path) -> None:
    # Flip the injector to deliver the answer as a tool result instead of a user
    # message (both shapes are supported; the backend chooses per kind).
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("hitl_request", {"prompt": "Approve?"}, "c1"),)),
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
