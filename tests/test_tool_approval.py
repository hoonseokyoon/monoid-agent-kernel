from __future__ import annotations

from monoid_agent_kernel.core.tool_approval import (
    TOOL_APPROVAL_TASK_KIND,
    approval_replay_from_task,
    build_tool_approval_task_request,
    denied_tool_approval_observation,
    normalize_tool_approval_result,
    tool_approval_key,
)
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.tasks import HostedTask
from monoid_agent_kernel.tools.base import ToolResult, ToolSpec


def _spec() -> ToolSpec:
    def handler(_ctx, _args):
        return ToolResult(ok=True)

    return ToolSpec(
        id="demo.approval",
        description="demo",
        input_schema={"type": "object"},
        capability="",
        side_effect="write",
        handler=handler,
    )


def test_tool_approval_task_request_redacts_secret_arguments() -> None:
    request = build_tool_approval_task_request(
        spec=_spec(),
        binding_id="demo.approval",
        model_name="demo_approval",
        call_name="demo_approval",
        call_id="call_1",
        arguments={"api_key": "secret", "value": "ok", "nested": {"token": "secret"}},
        reason="sensitive write",
        turn_id="turn_0001",
        tool_event_id="event_1",
    )

    assert request["arguments"]["api_key"] == "secret"
    assert request["arguments_preview"]["api_key"] == "[redacted]"
    assert request["arguments_preview"]["nested"]["token"] == "[redacted]"
    assert request["approval_key"]


def test_tool_approval_result_normalizes_approve_answer() -> None:
    normalized = normalize_tool_approval_result({"answer": "Approve"}, task_id="task_1")

    assert normalized["approved"] is True
    assert normalized["answer"] == "Approve"


def test_tool_approval_result_parses_approved_strings_fail_closed() -> None:
    assert normalize_tool_approval_result({"approved": "true"}, task_id="task_1")["approved"] is True
    assert normalize_tool_approval_result({"approved": "yes"}, task_id="task_1")["approved"] is True
    assert normalize_tool_approval_result({"approved": "approve"}, task_id="task_1")["approved"] is True
    assert normalize_tool_approval_result({"approved": "false"}, task_id="task_1")["approved"] is False
    assert normalize_tool_approval_result({"approved": "no"}, task_id="task_1")["approved"] is False
    assert normalize_tool_approval_result({"approved": "0"}, task_id="task_1")["approved"] is False
    assert normalize_tool_approval_result({"approved": "surprise"}, task_id="task_1")["approved"] is False


def test_denied_tool_approval_observation_strips_replay_material() -> None:
    request = {"tool_id": "demo.approval", "binding_id": "demo.approval", "call_name": "demo_approval"}
    denied = denied_tool_approval_observation(
        request,
        {"answer": "Deny", "approved": False, "reason": "policy"},
        task_id="task_1",
    )

    assert denied["approved"] is False
    assert denied["status"] == "denied"
    assert denied["reason"] == "policy"
    assert "arguments" not in denied


def test_approval_replay_requires_approved_result() -> None:
    request = {
        "call_name": "demo_approval",
        "call_id": "call_1",
        "arguments": {"value": "ok"},
        "binding_id": "demo.approval",
        "tool_id": "demo.approval",
    }

    assert approval_replay_from_task(request, {"approved": False}, task_id="task_1") is None
    replay = approval_replay_from_task(request, {"approved": True}, task_id="task_1")
    assert replay is not None
    assert replay["arguments"] == {"value": "ok"}
    assert replay["approval_key"] == tool_approval_key(replay)


def test_tool_approval_public_payload_hides_raw_arguments(tmp_path) -> None:
    request = build_tool_approval_task_request(
        spec=_spec(),
        binding_id="demo.approval",
        model_name="demo_approval",
        call_name="demo_approval",
        call_id="call_1",
        arguments={"api_key": "secret", "value": "ok"},
        reason="sensitive write",
        turn_id="turn_0001",
        tool_event_id="event_1",
    )
    task = HostedTask(
        job_id="task_1",
        kind=TOOL_APPROVAL_TASK_KIND,
        prompt="Approve tool call",
        status="running",
        started_at=1.0,
        resume_on_exit=True,
        job_path=tmp_path / "task.json",
        cancel_path=tmp_path / "cancel.requested",
        request=request,
    )

    public = task.public_payload(tmp_path, PermissionPolicy())

    assert "arguments" not in public["request"]
    assert public["request"]["arguments_preview"]["api_key"] == "[redacted]"
