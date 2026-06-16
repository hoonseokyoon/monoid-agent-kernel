from __future__ import annotations

from typing import Any

from native_agent_runner.permissions import PermissionPolicy
from native_agent_runner.web import public_query_preview, public_url_preview

REDACTED_PATH = "[redacted-path]"


def public_path(path: str, policy: PermissionPolicy) -> str:
    return REDACTED_PATH if policy.is_path_redacted(path) else path


def public_error_message(error: str) -> str:
    if not error:
        return ""
    if "PRIVATE KEY" in error.upper():
        return "[redacted-sensitive-error]"
    return error


def public_result_content(content: dict[str, Any], policy: PermissionPolicy) -> dict[str, Any]:
    public: dict[str, Any] = {}
    for key, value in content.items():
        if key == "content":
            public[key] = redacted_value(value)
        elif key == "path" and isinstance(value, str):
            public[key] = public_path(value, policy)
        else:
            public[key] = preview_value(key, value, policy)
    return public


def public_proposal_payload(payload: dict[str, Any], policy: PermissionPolicy) -> dict[str, Any]:
    files = [file for file in payload.get("files", []) if isinstance(file, dict)]
    return {
        "path": "proposal.json",
        "mode": payload.get("mode"),
        "proposal_hash": payload.get("proposal_hash"),
        "diff_path": payload.get("diff_path"),
        "diff_bytes": payload.get("diff_bytes"),
        "diff_sha256": payload.get("diff_sha256"),
        "changed_paths": [public_path(str(path), policy) for path in payload.get("changed_paths", [])],
        "files": [public_proposal_file(file, policy) for file in files],
    }


def public_proposal_file(file: dict[str, Any], policy: PermissionPolicy) -> dict[str, Any]:
    path = str(file.get("path", ""))
    redacted = policy.is_path_redacted(path)
    return {
        "path": public_path(path, policy),
        "kind": file.get("kind"),
        "size": file.get("size"),
        "sha256": file.get("sha256"),
        "base_sha256": file.get("base_sha256"),
        "proposed_sha256": file.get("proposed_sha256"),
        "snapshot_sha256": file.get("snapshot_sha256"),
        "change_kind": file.get("change_kind"),
        "snapshot_path": REDACTED_PATH if redacted else file.get("snapshot_path"),
    }


def args_preview(arguments: dict[str, Any], policy: PermissionPolicy) -> dict[str, Any]:
    return {key: preview_value(key, value, policy) for key, value in arguments.items()}


def shell_args_preview(arguments: dict[str, Any], policy: PermissionPolicy) -> dict[str, Any]:
    env = arguments.get("env") if isinstance(arguments.get("env"), dict) else {}
    return {
        "command_preview": preview_value("command_preview", str(arguments.get("command") or ""), policy),
        "cwd": preview_value("cwd", arguments.get("cwd", "."), policy),
        "timeout_s": arguments.get("timeout_s"),
        "max_output_bytes": arguments.get("max_output_bytes"),
        "startup_wait_s": arguments.get("startup_wait_s"),
        "background": bool(arguments.get("background", False)),
        "resume_on_exit": bool(arguments.get("resume_on_exit", True)),
        "env_keys": sorted(str(key) for key in env),
    }


def web_args_preview(arguments: dict[str, Any], policy: PermissionPolicy) -> dict[str, Any]:
    del policy
    preview: dict[str, Any] = {}
    if "query" in arguments:
        preview["query_preview"] = public_query_preview(str(arguments.get("query") or ""))
    if "url" in arguments:
        preview["url_preview"] = public_url_preview(str(arguments.get("url") or ""))
    for key in (
        "max_results",
        "max_tokens",
        "max_urls",
        "max_snippets",
        "timeout_s",
        "max_bytes",
        "recency_days",
        "locale",
        "format",
    ):
        if key in arguments:
            preview[key] = arguments[key]
    if "allowed_domains" in arguments:
        preview["allowed_domains"] = arguments.get("allowed_domains") or []
    if "blocked_domains" in arguments:
        preview["blocked_domains"] = arguments.get("blocked_domains") or []
    return preview


def preview_value(key: str, value: Any, policy: PermissionPolicy) -> Any:
    lowered = key.lower()
    if _is_content_field(lowered):
        return redacted_value(value)
    if lowered in {"path", "root", "cwd"} and isinstance(value, str) and policy.is_path_redacted(value):
        return redacted_value(value)
    if isinstance(value, dict):
        return {str(child_key): preview_value(str(child_key), child_value, policy) for child_key, child_value in value.items()}
    if isinstance(value, list):
        preview = [preview_value(key, item, policy) for item in value[:20]]
        if len(value) > 20:
            preview.append({"truncated_items": len(value) - 20})
        return preview
    if isinstance(value, str):
        encoded_len = len(value.encode("utf-8"))
        if encoded_len > 240:
            return {"type": "str", "preview": value[:160], "bytes": encoded_len, "truncated": True}
        return value
    return value


def redacted_value(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"redacted": True, "type": "str", "bytes": len(value.encode("utf-8"))}
    if isinstance(value, bytes):
        return {"redacted": True, "type": "bytes", "bytes": len(value)}
    return {"redacted": True, "type": type(value).__name__}


def _is_content_field(lowered_key: str) -> bool:
    # File-content fields are kept out of the public event stream; full content
    # lives only in the private transcript/proposal artifacts. Secret redaction
    # beyond this (and PermissionPolicy.redact_patterns) is the integrator's job.
    return lowered_key in {"content", "old", "new", "old_text", "new_text"}
