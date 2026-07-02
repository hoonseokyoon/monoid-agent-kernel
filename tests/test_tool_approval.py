from __future__ import annotations

from monoid_agent_kernel.core.tool_approval import (
    approval_replay_from_task,
    build_tool_approval_task_request,
    denied_tool_approval_observation,
    normalize_tool_approval_result,
)
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
