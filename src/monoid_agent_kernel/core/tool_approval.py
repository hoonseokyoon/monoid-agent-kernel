from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.tools.base import ToolSpec

TOOL_APPROVAL_TASK_KIND = "tool_approval"
TOOL_APPROVAL_RESULT_TYPE = "tool_approval_result"

_REDACTED = "[redacted]"
_SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "credential",
    "password",
    "secret",
    "token",
)
_APPROVE_VALUES = {"1", "allow", "allowed", "approve", "approved", "true", "y", "yes"}
_DENY_VALUES = {"0", "deny", "denied", "false", "n", "no", "reject", "rejected"}


def build_tool_approval_task_request(
    *,
    spec: ToolSpec,
    binding_id: str,
    model_name: str,
    call_name: str,
    call_id: str,
    arguments: Mapping[str, Any],
    reason: str,
    turn_id: str,
    tool_event_id: str | None,
) -> dict[str, Any]:
    """Build the durable hosted-task request for one model-requested tool approval."""
    sanitized_arguments = _jsonish(dict(arguments))
    request = {
        "prompt": f"Approve tool call {call_name}",
        "tool_id": spec.id,
        "binding_id": binding_id,
        "model_name": model_name,
        "call_name": call_name,
        "call_id": call_id,
        "arguments": sanitized_arguments,
        "arguments_preview": redact_tool_arguments(sanitized_arguments),
        "reason": reason,
        "side_effect": spec.side_effect,
        "turn_id": turn_id,
        "tool_event_id": tool_event_id,
    }
    request["approval_key"] = tool_approval_key(request)
    return request


def tool_approval_key(request: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "tool_id": request.get("tool_id"),
            "binding_id": request.get("binding_id"),
            "call_name": request.get("call_name"),
            "call_id": request.get("call_id"),
            "arguments": request.get("arguments"),
        }
    )


def redact_tool_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): (_REDACTED if _is_secret_key(str(key)) else _redact_value(value))
        for key, value in arguments.items()
    }


def normalize_tool_approval_result(
    result: Mapping[str, Any] | None,
    *,
    task_id: str,
    default_reason: str = "",
) -> dict[str, Any]:
    payload = dict(result or {})
    answer = str(payload.get("answer") or "").strip()
    if "approved" in payload and payload.get("approved") is not None:
        approved_bool = _parse_approval_bool(payload.get("approved")) is True
    else:
        approved_bool = _parse_approval_bool(answer) is True
    reason = str(payload.get("reason") or default_reason or ("approved" if approved_bool else "denied"))
    return {
        "type": TOOL_APPROVAL_RESULT_TYPE,
        "task_id": task_id,
        "approved": approved_bool,
        "answer": "Approve" if approved_bool else "Deny",
        "reason": reason,
    }


def approval_replay_from_task(
    request: Mapping[str, Any] | None,
    result: Mapping[str, Any] | None,
    *,
    task_id: str,
) -> dict[str, Any] | None:
    if not isinstance(request, Mapping):
        return None
    normalized = normalize_tool_approval_result(result, task_id=task_id)
    if not normalized["approved"]:
        return None
    return {
        "call_name": str(request.get("call_name") or ""),
        "call_id": str(request.get("call_id") or ""),
        "arguments": dict(request.get("arguments") or {}),
        "binding_id": str(request.get("binding_id") or ""),
        "tool_id": str(request.get("tool_id") or ""),
        "task_id": task_id,
        "approval_key": str(request.get("approval_key") or tool_approval_key(request)),
    }


def denied_tool_approval_observation(
    request: Mapping[str, Any] | None,
    result: Mapping[str, Any] | None,
    *,
    task_id: str,
    reason: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_tool_approval_result(result, task_id=task_id)
    deny_reason = str(reason or normalized.get("reason") or "denied")
    return {
        **normalized,
        "approved": False,
        "answer": "Deny",
        "reason": deny_reason,
        "status": "denied",
        "tool_id": str((request or {}).get("tool_id") or ""),
        "binding_id": str((request or {}).get("binding_id") or ""),
        "call_name": str((request or {}).get("call_name") or ""),
    }


def _redact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return redact_tool_arguments(value)
    if isinstance(value, list | tuple):
        return [_redact_value(item) for item in value]
    return value


def _is_secret_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(part in lowered for part in _SECRET_KEY_PARTS)


def _parse_approval_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _APPROVE_VALUES:
            return True
        if normalized in _DENY_VALUES:
            return False
    return None


def _jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonish(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonish(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)
