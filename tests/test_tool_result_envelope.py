from __future__ import annotations

from native_agent_runner.errors import (
    PermissionDenied,
    ToolExecutionError,
    WorkspaceError,
)
from native_agent_runner.loop import _failure_result
from native_agent_runner.tools.base import ToolResult


def test_success_envelope_keeps_content_under_result_no_collision() -> None:
    # A handler whose content contains envelope-looking keys must not corrupt the envelope.
    obs = ToolResult(ok=True, content={"ok": "shadow", "error": "shadow"}).to_observation()
    assert obs == {"ok": True, "result": {"ok": "shadow", "error": "shadow"}}


def test_failure_envelope_shape() -> None:
    obs = ToolResult(
        ok=False,
        content={"path": "x"},
        error="boom",
        error_code="tool_handler_error",
        retryable=True,
        category="tool",
    ).to_observation()
    assert obs == {
        "ok": False,
        "result": {"path": "x"},
        "error": {
            "message": "boom",
            "code": "tool_handler_error",
            "category": "tool",
            "retryable": True,
        },
    }


def test_failure_result_maps_exception_retry_signal() -> None:
    tool = _failure_result(ToolExecutionError("nope"))
    assert (tool.ok, tool.error_code, tool.category, tool.retryable) == (
        False,
        "tool_handler_error",
        "tool",
        True,
    )

    perm = _failure_result(PermissionDenied("denied"))
    assert (perm.error_code, perm.category, perm.retryable) == ("permission_denied", "policy", False)

    ws = _failure_result(WorkspaceError("bad path"))
    assert (ws.error_code, ws.category, ws.retryable) == ("workspace_error", "workspace", False)

    # Raw ValueError/TypeError become tool handler errors (retryable, "tool").
    raw = _failure_result(ValueError("bad arg"))
    assert (raw.error_code, raw.category, raw.retryable) == ("tool_handler_error", "tool", True)
