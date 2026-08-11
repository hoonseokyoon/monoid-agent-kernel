from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.core.model_io import DEFAULT_SECRET_KEY_PARTS, REDACTION_PLACEHOLDER
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.public_view import (
    APPROVAL_BYTE_BUDGET,
    APPROVAL_BYTE_THRESHOLD,
    APPROVAL_PAYLOAD_BYTE_BUDGET,
    UNMASKED,
    PayloadBudget,
    preview_value,
    public_mapping,
    touches_redacted_path,
)
from monoid_agent_kernel.tools.base import ToolSpec

TOOL_APPROVAL_TASK_KIND = "tool_approval"
TOOL_APPROVAL_RESULT_TYPE = "tool_approval_result"

# The secret-key list and placeholder now live in ``core.model_io``, where the model-I/O capture
# policies need the same answer. Approvals had them first; two copies would drift, and the drift
# would be invisible until one surface masked something the other did not.
_REDACTED = REDACTION_PLACEHOLDER
_SECRET_KEY_PARTS = DEFAULT_SECRET_KEY_PARTS
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
    policy: PermissionPolicy | None = None,
) -> dict[str, Any]:
    """Build the durable hosted-task request for one model-requested tool approval.

    ``arguments`` stays raw: it is what the approver replays and what ``approval_key`` is taken
    over, so a truncated copy would key a different call than the one being approved. Only
    ``arguments_preview`` — the projection every public surface reads — is bounded.
    """
    sanitized_arguments = _jsonish(dict(arguments))
    request = {
        "prompt": f"Approve tool call {call_name}",
        "tool_id": spec.id,
        "binding_id": binding_id,
        "model_name": model_name,
        "call_name": call_name,
        "call_id": call_id,
        "arguments": sanitized_arguments,
        "arguments_preview": redact_tool_arguments(
            sanitized_arguments,
            prose_keys=_PROSE_KEYS_BY_PREVIEW_KIND.get(spec.preview_kind, frozenset()),
            policy=policy,
        ),
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


# Tool arguments that are the model's own prose, keyed by the tool's preview kind. Mirrors
# ``public_view._FINISH_CONTENT_KEYS``: an approval request republishes ``arguments_preview`` on
# ``task.started``, so it is a SECOND public route for the same values. Redacting only in
# ``_tool_start_data`` left that half unbound, and binding a tool to ``authorization="ask"`` put
# the text straight back on ``events.jsonl``.
_PROSE_KEYS_BY_PREVIEW_KIND: dict[str, frozenset[str]] = {
    "finish": frozenset({"summary", "notes"}),
}


def redact_tool_arguments(
    arguments: Mapping[str, Any],
    *,
    prose_keys: frozenset[str] = frozenset(),
    policy: PermissionPolicy | None = None,
) -> dict[str, Any]:
    """The approval projection of a tool call's arguments — a *decision* surface, not a log.

    Runs on ``public_view.preview_value``'s traversal rather than its own. It used to have one of
    its own, and the two carried disjoint halves of the same policy: this one masked secret- and
    prose-named keys but applied no length, depth or item cap at all, so an ``ask``-gated
    ``fs.write`` published an unbounded file body on ``task.started``, in ``task.json``, and back to
    the model through ``job.list``. Meanwhile ``preview_value`` capped everything but knew nothing
    about secrets.

    What it takes from that traversal is the **bounds**, not the trace surface's *policy*. The
    budget here is far larger, and ``decision_surface=True`` keeps file-content fields visible,
    because someone reads this to decide whether a call may run: a command cut mid-string hides the
    part that matters (with the model choosing where that part sits), and a card rendering
    ``{"redacted": true}`` where the body should be asks a human to authorize a write they cannot
    see. Both were measured on this branch before being corrected. The result is still strictly
    tighter than before the release — bounded rather than unbounded — just not blanked.

    Routing this *through* ``preview_value`` rather than *replacing* it with ``args_preview`` is the
    load-bearing detail. ``args_preview`` is the generic branch of a four-way dispatch on
    ``spec.preview_kind``, and the request does not record which kind it came from — so swapping it
    in would have dropped secret masking (``api_key`` published verbatim), dropped the ``run.finish``
    prose redaction that closed the settle leak one stage ago, and dropped the shell/web shaping.
    Measured, not assumed.

    ``value is not None`` mirrors ``public_view.finish_args_preview``. Without it the two halves
    moved in opposite directions on ``notes: null`` — a legal call shape — and the approval preview
    badged an absent value as withheld, which is the exact behaviour the other half was changed to
    stop doing.
    """
    resolved = policy if policy is not None else PermissionPolicy()
    # The operator's explicit policy outranks the decision surface's exemption -- and it has to
    # outrank it for the *whole* call, not field by field. Blanking only the path produced a card
    # showing a private key's contents above `{"redacted": true}` where the destination should be:
    # it broke the approver's judgement and protected nothing, since the body was right there.
    # Blanking only the content is the same trade in the other direction. So: if `redact_patterns`
    # matches any path this call touches, this is not a decision surface, and the approver decides
    # from the tool name and the withheld-value markers. That is the operator asking for exactly
    # this, which is different from the kernel's own `_is_content_field` default -- a blanket rule
    # the approver is entitled to see past.
    withheld = touches_redacted_path(arguments, resolved)

    def mask(key: str, value: Any) -> Any:
        if _is_secret_key(key) or (value is not None and key.lower() in prose_keys):
            return _REDACTED
        return UNMASKED

    # The approval surface gets its *own* payload ceiling, for the same reason it gets its own
    # per-value budget: the traversal is shared, so a budget added there is inherited here, and
    # letting the trace constant become the approval card's ceiling would cut arguments a person
    # needs to read. Far higher, still bounded -- a card is read by a human, not paged by one.
    payload_budget = PayloadBudget(APPROVAL_PAYLOAD_BYTE_BUDGET)
    return public_mapping(
        arguments,
        lambda key, value: preview_value(
            key,
            value,
            resolved,
            mask=mask,
            threshold=APPROVAL_BYTE_THRESHOLD,
            budget=APPROVAL_BYTE_BUDGET,
            _payload_budget=payload_budget,
            # This is the decision surface: `content`/`old`/`new` are blanked on the trace surface
            # and *shown* here, bounded by the budget above. An approval card that renders
            # `{"redacted": true}` where the file body should be asks a human to authorize a write
            # they cannot see -- which trains people to approve blindly and is a worse outcome than
            # the logging it buys. The bound still applies, so this publishes at most
            # `APPROVAL_BYTE_BUDGET` rather than the unbounded body that reached here before.
            #
            # ...unless the operator redacted a path this call touches; see `withheld` above.
            decision_surface=not withheld,
        ),
        threshold=APPROVAL_BYTE_THRESHOLD,
        budget=APPROVAL_BYTE_BUDGET,
        payload_budget=payload_budget,
    )


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


# Tool arguments nest a handful of levels in practice. The bound exists because ``arguments`` is
# model-controlled and is stored raw -- it is the replay copy and what ``approval_key`` is taken
# over, so it cannot be truncated the way the preview is. Rejecting is the only honest answer: a call
# whose arguments cannot be recorded faithfully cannot be faithfully approved.
MAX_ARGUMENT_DEPTH = 64
# Known gap, deliberately not closed in this release. This bound is reached only through
# `build_tool_approval_task_request`, i.e. the `ask` path. On `allow`, the arguments still enter
# the message history and reach `RunCheckpoint.to_json`, whose `dataclasses.asdict` recurses in
# pure Python and raises `RecursionError` around depth 500 -- while `json.loads`/`json.dumps`
# handle 900 -- surfacing as `_CheckpointPersistError` out of `run_once`: the run lost, from one
# model-authored argument. Guarding tool *dispatch* does not close it (the turn is already in
# history by then); the fixes are either rejecting the turn at ingestion or dropping `asdict`,
# and the latter also drops its deep copy, so a checkpoint would start sharing mutable state
# with the live loop. Both are decisions for the durability surface, not for a content-egress
# release, and neither file is in its diff.


def _jsonish(value: Any, _depth: int = 0) -> Any:
    # Raises ``ValueError``, which the tool-call handler already turns into a tool error the model
    # can read and correct. Left unbounded this raised ``RecursionError`` at ~496 -- and
    # ``RecursionError`` is a ``RuntimeError``, so it fell straight through the
    # ``(NativeAgentError, ValueError, TypeError)`` handler and out of tool dispatch entirely. The
    # read side learned this same lesson already (``core.schemas`` catches it explicitly); an
    # uncaught crash reachable from one model-authored argument is worse than a rejected call.
    if _depth > MAX_ARGUMENT_DEPTH:
        raise ValueError(
            f"tool arguments nest deeper than {MAX_ARGUMENT_DEPTH} levels; "
            "flatten the payload or pass it as a workspace file"
        )
    if isinstance(value, Mapping):
        return {str(key): _jsonish(item, _depth + 1) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonish(item, _depth + 1) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)
