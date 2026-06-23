from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from native_agent_runner.core._util import canonical_sha256
from native_agent_runner.workspace.paths import normalize_workspace_path


EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["schema_version", "event_id", "seq", "run_id", "timestamp", "type", "level", "data"],
    "properties": {
        "schema_version": {"const": "native-agent-runner.event.v1"},
        "event_id": {"type": "string", "minLength": 1},
        "seq": {"type": "integer", "minimum": 1},
        "run_id": {"type": "string", "minLength": 1},
        "turn_id": {"type": ["string", "null"]},
        "parent_id": {"type": ["string", "null"]},
        "timestamp": {"type": "string", "pattern": "Z$"},
        "type": {"type": "string", "pattern": "^[a-z]+(\\.[a-z_]+)+$"},
        "level": {"enum": ["debug", "info", "warning", "error"]},
        "data": {"type": "object"},
    },
    "additionalProperties": False,
}

# Per-event-type `data` schemas. The envelope above covers everything except
# `data`, which is the implicit contract the two consumers (recorder.StatusJsonSink
# and core.projections) read. These schemas pin that contract per event type so
# drift between producer (emit) and consumers is caught by validate_run_dir.
#
# Strictness is staged. Events whose payload is fully enumerable at the emit site
# use `additionalProperties: False`. Events whose payload is assembled from
# to_public_json()/snapshot/merged dicts (shell, web, approval, job, proposal
# lifecycle, workspace snapshots) use `additionalProperties: True` and document
# only the load-bearing keys; tightening them is a follow-up. `required` lists
# only keys that are always emitted AND read by a consumer or otherwise essential.
_STR: dict[str, Any] = {"type": "string"}
_STR_NULL: dict[str, Any] = {"type": ["string", "null"]}
_INT: dict[str, Any] = {"type": "integer"}
_NUM: dict[str, Any] = {"type": "number"}
_BOOL: dict[str, Any] = {"type": "boolean"}
_OBJ: dict[str, Any] = {"type": "object"}
_STR_ARRAY: dict[str, Any] = {"type": "array", "items": {"type": "string"}}
_OBJ_ARRAY: dict[str, Any] = {"type": "array", "items": {"type": "object"}}


def _data_schema(
    properties: dict[str, Any],
    *,
    required: tuple[str, ...] = (),
    additional: bool = False,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": additional,
    }


EVENT_DATA_SCHEMAS: dict[str, dict[str, Any]] = {
    "run.started": _data_schema(
        {
            "workspace": _STR,
            "run_dir": _STR,
            "manifest_path": _STR,
            "mode": _STR,
            "workspace_backend": _STR,
            "workspace_base_path": _STR,
            "model_provider": _STR,
            "model": _STR,
            "reasoning_effort": _STR,
            "visible_bindings": _STR_ARRAY,
            "agent_config_hash": _STR,
        },
        required=("mode", "workspace_backend", "model"),
    ),
    "run.finished": _data_schema(
        {
            "status": _STR,
            "error": _STR,
            "error_code": _STR,
            "final_text": _STR,
            "duration_s": _NUM,
            "diff_path": _STR,
            "proposal_path": _STR,
            "metrics_path": _STR,
        },
        required=("status",),
    ),
    "run.failed": _data_schema(
        {"error": _STR, "error_code": _STR, "type": _STR},
        required=("error_code",),
    ),
    "run.waiting": _data_schema(
        {"reason": _STR, "jobs": _OBJ_ARRAY},
    ),
    "run.resumed": _data_schema(
        {"reason": _STR, "job_ids": _STR_ARRAY, "count": _INT},
    ),
    "run.awaiting_input": _data_schema(
        {"reason": _STR, "task_ids": _STR_ARRAY, "prompt": _STR_NULL},
        required=("reason",),
    ),
    "turn.settled": _data_schema(
        {"status": _STR, "final_text": _STR, "error_code": _STR, "changed_paths": _STR_ARRAY},
        required=("status",),
    ),
    "checkpoint.committed": _data_schema(
        {"workspace_backend": _STR, "changed_paths": _STR_ARRAY},
    ),
    "agent.config.updated": _data_schema(
        {
            "definition_id": _STR,
            "config_version": _INT,
            "config_hash": _STR,
            "previous_config_version": {"type": ["integer", "null"]},
            "previous_config_hash": _STR_NULL,
            "diff": _OBJ,
        },
        required=("definition_id", "config_version", "config_hash"),
    ),
    "model.turn.started": _data_schema(
        {"step": _INT, "previous_turn_handle": _STR_NULL},
        required=("step",),
    ),
    "model.turn.finished": _data_schema(
        {
            "step": _INT,
            "response_id": _STR_NULL,
            "tool_calls": _INT,
            "has_final": _BOOL,
            "usage": _OBJ,
        },
        required=("step",),
    ),
    "turn.failed": _data_schema(
        {
            "error": _STR,
            "error_code": _STR,
            "provider_error_code": _STR,
            "http_status": {"type": ["integer", "null"]},
            "retryable": _BOOL,
        },
        required=("error_code",),
    ),
    "turn.interrupted": _data_schema(
        {"reason": _STR},
        required=(),
    ),
    "model.output.delta": _data_schema(
        {"text": _STR},
        required=("text",),
    ),
    "model.reasoning.delta": _data_schema(
        {"text": _STR},
        required=("text",),
    ),
    "model.input.degraded": _data_schema(
        {"dropped_part_types": _STR_ARRAY, "reason": _STR},
        required=("reason",),
    ),
    "metrics.updated": _data_schema(
        {
            "step": _INT,
            "tool_calls": _INT,
            "input_tokens": _INT,
            "output_tokens": _INT,
            "total_tokens": _INT,
            "web_search_calls": _INT,
            "web_fetch_calls": _INT,
            "web_context_calls": _INT,
            "web_failed_calls": _INT,
        },
    ),
    "tool.call.started": _data_schema(
        {
            "call_id": _STR,
            "tool": _STR,
            "capability": _STR_NULL,
            "side_effect": _STR_NULL,
            "paths": _STR_ARRAY,
            "args_preview": _OBJ,
        },
        required=("call_id", "tool"),
    ),
    "tool.call.finished": _data_schema(
        {"call_id": _STR, "tool": _STR, "ok": _BOOL, "error": _STR, "error_code": _STR},
        required=("call_id", "tool", "ok"),
    ),
    "tool.call.failed": _data_schema(
        {"call_id": _STR, "tool": _STR, "ok": _BOOL, "error": _STR, "error_code": _STR},
        required=("call_id", "tool", "ok"),
    ),
    "tool.surface.updated": _data_schema(
        {
            "surface_hash": _STR,
            "immediate_binding_ids": _STR_ARRAY,
            "searchable_count": _INT,
            "hidden_count": _INT,
            "delta_notice": _STR,
        },
        required=("surface_hash", "immediate_binding_ids", "searchable_count", "hidden_count"),
    ),
    "tool.approval.requested": _data_schema({}, additional=True),
    "tool.approval.approved": _data_schema({}, additional=True),
    "tool.approval.denied": _data_schema({}, additional=True),
    "shell.exec.started": _data_schema({}, additional=True),
    "shell.exec.finished": _data_schema({}, additional=True),
    "shell.exec.failed": _data_schema({}, additional=True),
    "job.started": _data_schema({"job_id": _STR_NULL}, additional=True),
    "job.output.updated": _data_schema({"job_id": _STR_NULL}, additional=True),
    "job.finished": _data_schema({"job_id": _STR_NULL}, additional=True),
    "job.timed_out": _data_schema({"job_id": _STR_NULL}, additional=True),
    "job.cancelled": _data_schema({"job_id": _STR_NULL}, additional=True),
    "job.output_limited": _data_schema({"job_id": _STR_NULL}, additional=True),
    "job.failed": _data_schema({"job_id": _STR_NULL}, additional=True),
    "task.started": _data_schema({"task_id": _STR_NULL, "kind": _STR}, additional=True),
    "task.finished": _data_schema({"task_id": _STR_NULL, "kind": _STR}, additional=True),
    "task.cancelled": _data_schema({"task_id": _STR_NULL, "kind": _STR}, additional=True),
    "task.timed_out": _data_schema({"task_id": _STR_NULL, "kind": _STR}, additional=True),
    "task.failed": _data_schema({"task_id": _STR_NULL, "kind": _STR}, additional=True),
    "subagent.started": _data_schema(
        {"subagent_type": _STR, "child_run_id": _STR, "depth": _INT, "background": _BOOL},
        additional=True,
    ),
    "subagent.finished": _data_schema(
        {"subagent_type": _STR, "child_run_id": _STR, "status": _STR, "usage": _OBJ},
        additional=True,
    ),
    "subagent.failed": _data_schema(
        {"subagent_type": _STR, "child_run_id": _STR, "status": _STR, "usage": _OBJ},
        additional=True,
    ),
    "skill.activated": _data_schema(
        {"name": _STR, "resource_count": _INT},
        additional=True,
    ),
    "web.search.started": _data_schema({}, additional=True),
    "web.search.finished": _data_schema({}, additional=True),
    "web.search.failed": _data_schema({}, additional=True),
    "web.fetch.started": _data_schema({}, additional=True),
    "web.fetch.finished": _data_schema({}, additional=True),
    "web.fetch.failed": _data_schema({}, additional=True),
    "web.context.started": _data_schema({}, additional=True),
    "web.context.finished": _data_schema({}, additional=True),
    "web.context.failed": _data_schema({}, additional=True),
    "permission.denied": _data_schema(
        {
            "call_id": _STR,
            "tool": _STR,
            "requested_tool": _STR,
            "error": _STR,
            "error_code": _STR,
            "surface_decision": _STR_NULL,
            "surface_reason": _STR_NULL,
        },
        required=("tool",),
    ),
    "workspace.file.read": _data_schema(
        {"tool": _STR, "paths": _STR_ARRAY},
        required=("tool",),
    ),
    "workspace.file.changed": _data_schema(
        {
            "tool": _STR,
            "job_id": _STR_NULL,
            "paths": _STR_ARRAY,
            "result": _OBJ,
            "mode": _STR,
        },
        additional=True,
    ),
    "workspace.diff.updated": _data_schema(
        {"path": _STR, "bytes": _INT, "changed_paths": _STR_ARRAY},
    ),
    "workspace.proposal.updated": _data_schema(
        {"changed_paths": _STR_ARRAY, "proposal_hash": _STR_NULL, "diff_sha256": _STR_NULL},
        additional=True,
    ),
    "proposal.ready": _data_schema(
        {"proposal_hash": _STR_NULL, "diff_sha256": _STR_NULL, "changed_paths": _STR_ARRAY},
    ),
    "proposal.package.exported": _data_schema(
        {"package_hash": _STR, "package_path": _STR},
        required=("package_hash",),
        additional=True,
    ),
    "proposal.approved": _data_schema(
        {"approval_hash": _STR, "package_hash": _STR},
        required=("approval_hash",),
        additional=True,
    ),
    "proposal.rejected": _data_schema(
        {"approval_hash": _STR, "package_hash": _STR},
        required=("approval_hash",),
        additional=True,
    ),
    "proposal.applied": _data_schema(
        {
            "status": _STR,
            "approval_hash": _STR_NULL,
            "package_hash": _STR_NULL,
            "applied_paths": _STR_ARRAY,
            "conflicts": _OBJ_ARRAY,
        },
        required=("status",),
        additional=True,
    ),
    "proposal.conflict": _data_schema(
        {
            "status": _STR,
            "approval_hash": _STR_NULL,
            "package_hash": _STR_NULL,
            "applied_paths": _STR_ARRAY,
            "conflicts": _OBJ_ARRAY,
        },
        required=("status",),
        additional=True,
    ),
    "proposal.stale": _data_schema({}, additional=True),
    "artifact.emitted": _data_schema(
        {"artifact_id": _STR, "path": _STR, "kind": _STR},
        required=("artifact_id",),
    ),
    "plan.updated": _data_schema(
        {"items": _OBJ_ARRAY},
        required=("items",),
    ),
}

MANIFEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version",
        "run_id",
        "created_at",
        "mode",
        "workspace_backend",
        "workspace_root",
        "workspace_base_path",
        "model_provider",
        "model",
        "reasoning_effort",
        "limits",
        "permission_policy",
        "tool_surface",
        "agent_config",
        "tool_specs",
        "metadata",
        "workspace_index_path",
    ],
    "properties": {
        "schema_version": {"const": "native-agent-runner.manifest.v1"},
        "run_id": {"type": "string", "minLength": 1},
        "created_at": {"type": "string", "pattern": "Z$"},
        "mode": {"enum": ["read-only", "propose", "apply"]},
        "workspace_backend": {"enum": ["overlay", "staging"]},
        "workspace_root": {"type": "string"},
        "workspace_base_path": {"type": "string"},
        "model_provider": {"type": "string"},
        "model": {"type": "string"},
        "reasoning_effort": {"type": "string"},
        "limits": {"type": "object"},
        "permission_policy": {"type": "object"},
        "tool_surface": {"type": "object"},
        "agent_config": {"type": "object"},
        "tool_specs": {"type": "array", "items": {"type": "object"}},
        "metadata": {"type": "object"},
        "workspace_index_path": {"type": "string"},
    },
    "additionalProperties": False,
}

WORKSPACE_BASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version",
        "run_id",
        "created_at",
        "workspace_root",
        "workspace_backend",
        "entries",
        "excluded",
    ],
    "properties": {
        "schema_version": {"const": "native-agent-runner.workspace-base.v1"},
        "run_id": {"type": "string", "minLength": 1},
        "created_at": {"type": "string", "pattern": "Z$"},
        "workspace_root": {"type": "string"},
        "workspace_backend": {"enum": ["overlay", "staging"]},
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["path", "kind", "size", "sha256"],
                "properties": {
                    "path": {"type": "string"},
                    "kind": {"enum": ["file", "dir", "other"]},
                    "size": {"type": "integer", "minimum": 0},
                    "sha256": {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"},
                },
                "additionalProperties": False,
            },
        },
        "excluded": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["path", "reason"],
                "properties": {
                    "path": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

WORKSPACE_INDEX_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version",
        "run_id",
        "generated_at",
        "workspace_root",
        "max_entries",
        "max_hash_bytes",
        "truncated",
        "entries",
        "excluded",
    ],
    "properties": {
        "schema_version": {"const": "native-agent-runner.workspace-index.v1"},
        "run_id": {"type": "string", "minLength": 1},
        "generated_at": {"type": "string", "pattern": "Z$"},
        "workspace_root": {"type": "string"},
        "max_entries": {"type": "integer", "minimum": 1},
        "max_hash_bytes": {"type": "integer", "minimum": 0},
        "truncated": {"type": "boolean"},
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["path", "kind", "size", "sha256", "hash_status"],
                "properties": {
                    "path": {"type": "string"},
                    "kind": {"enum": ["file", "dir", "other"]},
                    "size": {"type": "integer", "minimum": 0},
                    "sha256": {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"},
                    "hash_status": {"enum": ["hashed", "too_large", "not_file", "error"]},
                },
                "additionalProperties": False,
            },
        },
        "excluded": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["path", "reason"],
                "properties": {
                    "path": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

TRANSCRIPT_RECORD_SCHEMA: dict[str, Any] = {
    "oneOf": [
        {
            "type": "object",
            "required": ["kind", "step", "previous_turn_handle", "observations"],
            "properties": {
                "kind": {"const": "model_request"},
                "step": {"type": "integer", "minimum": 1},
                "previous_turn_handle": {"type": ["string", "null"]},
                "observations": {"type": "array", "items": {"type": "object"}},
            },
            "additionalProperties": True,
        },
        {
            "type": "object",
            "required": ["kind", "step", "response_id", "final_text", "tool_calls", "usage"],
            "properties": {
                "kind": {"const": "model_turn"},
                "step": {"type": "integer", "minimum": 1},
                "response_id": {"type": ["string", "null"]},
                "final_text": {"type": ["string", "null"]},
                "tool_calls": {"type": "array", "items": {"type": "object"}},
                "usage": {"type": "object"},
                "error": {"type": "string"},
                "error_code": {"type": "string"},
                "provider_error_code": {"type": "string"},
                "retryable": {"type": "boolean"},
                "http_status": {"type": ["integer", "null"]},
            },
            "additionalProperties": True,
        },
        {
            "type": "object",
            "required": ["kind", "step", "call_id", "tool", "output"],
            "properties": {
                "kind": {"const": "tool_observation"},
                "step": {"type": "integer", "minimum": 1},
                "call_id": {"type": "string"},
                "tool": {"type": "string"},
                "output": {"type": "object"},
            },
            "additionalProperties": True,
        },
        {
            "type": "object",
            "required": [
                "kind",
                "step",
                "turn_id",
                "definition_id",
                "config_version",
                "config_hash",
                "binding_ids",
                "tool_ids",
                "prompt_hash",
            ],
            "properties": {
                "kind": {"const": "agent_runtime_config_snapshot"},
                "step": {"type": "integer", "minimum": 1},
                "turn_id": {"type": "string"},
                "definition_id": {"type": "string"},
                "config_version": {"type": "integer"},
                "config_hash": {"type": "string"},
                "binding_ids": {"type": "array", "items": {"type": "string"}},
                "tool_ids": {"type": "array", "items": {"type": "string"}},
                "prompt_hash": {"type": "string"},
                "model": {"type": ["string", "null"]},
            },
            "additionalProperties": True,
        },
        {
            "type": "object",
            "required": [
                "kind",
                "step",
                "turn_id",
                "surface_hash",
                "immediate_tools",
                "searchable_tools",
                "search_entries",
                "hidden_tool_ids",
                "authorizations",
            ],
            "properties": {
                "kind": {"const": "tool_surface_snapshot"},
                "step": {"type": "integer", "minimum": 1},
                "turn_id": {"type": "string"},
                "surface_hash": {"type": "string"},
                "immediate_tools": {"type": "array", "items": {"type": "object"}},
                "searchable_tools": {"type": "array", "items": {"type": "object"}},
                "search_entries": {"type": "array", "items": {"type": "object"}},
                "hidden_tool_ids": {"type": "array", "items": {"type": "string"}},
                "authorizations": {"type": "object"},
                "delta_notice": {"type": "string"},
                "surface_warnings": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": True,
        },
    ]
}

PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version",
        "run_id",
        "updated_at",
        "mode",
        "proposal_hash",
        "diff_path",
        "diff_bytes",
        "diff_sha256",
        "changed_paths",
        "files",
    ],
    "properties": {
        "schema_version": {"const": "native-agent-runner.proposal.v2"},
        "run_id": {"type": "string", "minLength": 1},
        "updated_at": {"type": "number"},
        "mode": {"enum": ["read-only", "propose", "apply"]},
        "proposal_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "diff_path": {"type": "string"},
        "diff_bytes": {"type": "integer", "minimum": 0},
        "diff_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "changed_paths": {"type": "array", "items": {"type": "string"}},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["path", "kind", "size", "change_kind"],
                "properties": {
                    "path": {"type": "string"},
                    "kind": {"type": "string"},
                    "size": {"type": "integer", "minimum": 0},
                    "sha256": {"type": ["string", "null"]},
                    "base_sha256": {"type": ["string", "null"]},
                    "proposed_sha256": {"type": ["string", "null"]},
                    "snapshot_path": {"type": "string"},
                    "snapshot_sha256": {"type": "string"},
                    "change_kind": {"enum": ["created", "modified", "deleted", "directory"]},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

METRICS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["run_id", "started_at", "finished_at", "status", "duration_s", "error_code"],
    "properties": {
        "run_id": {"type": "string", "minLength": 1},
        "started_at": {"type": "number"},
        "finished_at": {"type": "number"},
        "status": {"enum": ["completed", "failed", "limited"]},
        "duration_s": {"type": "number", "minimum": 0},
        "error": {"type": "string"},
        "error_code": {"type": "string"},
    },
    "additionalProperties": True,
}

STATUS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["run_id", "status", "last_event_seq", "last_event_type", "updated_at"],
    "properties": {
        "run_id": {"type": "string", "minLength": 1},
        "status": {"type": "string"},
        "last_event_seq": {"type": "integer", "minimum": 1},
        "last_event_type": {"type": "string"},
        "updated_at": {"type": "string"},
    },
    "additionalProperties": True,
}

JOB_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version",
        "job_id",
        "command",
        "command_preview",
        "cwd",
        "status",
        "started_at",
        "duration_s",
        "stdout_path",
        "stderr_path",
        "stdout_bytes",
        "stderr_bytes",
        "effective_timeout_s",
        "effective_max_output_bytes",
        "effective_startup_wait_s",
        "execution_workspace",
        "resume_on_exit",
    ],
    "properties": {
        "schema_version": {"const": "native-agent-runner.background-job.v1"},
        "job_id": {"type": "string", "minLength": 1},
        "command": {"type": "string"},
        "command_preview": {"type": "string"},
        "cwd": {"type": "string"},
        "status": {"enum": ["running", "exited", "timed_out", "cancelled", "output_limited", "failed"]},
        "started_at": {"type": "number"},
        "finished_at": {"type": ["number", "null"]},
        "duration_s": {"type": "number", "minimum": 0},
        "exit_code": {"type": ["integer", "null"]},
        "timed_out": {"type": "boolean"},
        "output_truncated": {"type": "boolean"},
        "error": {"type": "string"},
        "changed_paths": {"type": "array", "items": {"type": "string"}},
        "stdout_path": {"type": "string"},
        "stderr_path": {"type": "string"},
        "stdout_bytes": {"type": "integer", "minimum": 0},
        "stderr_bytes": {"type": "integer", "minimum": 0},
        "requested_timeout_s": {"type": ["integer", "null"]},
        "effective_timeout_s": {"type": "integer", "minimum": 1},
        "requested_max_output_bytes": {"type": ["integer", "null"]},
        "effective_max_output_bytes": {"type": "integer", "minimum": 1},
        "requested_startup_wait_s": {"type": ["integer", "null"]},
        "effective_startup_wait_s": {"type": "integer", "minimum": 0},
        "execution_workspace": {"enum": ["isolated-copy", "direct"]},
        "resume_on_exit": {"type": "boolean"},
    },
    "additionalProperties": False,
}

PACKAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version",
        "run_id",
        "created_at",
        "proposal_hash",
        "diff_sha256",
        "files",
        "package_hash",
    ],
    "properties": {
        "schema_version": {"const": "native-agent-runner.proposal-package.v1"},
        "run_id": {"type": "string", "minLength": 1},
        "created_at": {"type": "string"},
        "proposal_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "diff_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "package_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["path", "role", "size", "sha256"],
                "properties": {
                    "path": {"type": "string"},
                    "role": {"type": "string"},
                    "workspace_path": {"type": "string"},
                    "size": {"type": "integer", "minimum": 0},
                    "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

APPROVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version",
        "approval_id",
        "decision",
        "package_hash",
        "proposal_hash",
        "approved_paths",
        "rejected_paths",
        "approver_id",
        "approved_at",
        "note",
        "approval_hash",
    ],
    "properties": {
        "schema_version": {"const": "native-agent-runner.approval.v1"},
        "approval_id": {"type": "string"},
        "decision": {"enum": ["approved", "rejected"]},
        "package_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "proposal_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "approved_paths": {"type": "array", "items": {"type": "string"}},
        "rejected_paths": {"type": "array", "items": {"type": "string"}},
        "approver_id": {"type": "string"},
        "approved_at": {"type": "string"},
        "note": {"type": "string"},
        "approval_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    },
    "additionalProperties": False,
}

APPLY_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version",
        "status",
        "applied_paths",
        "skipped_paths",
        "conflicts",
        "approval_hash",
        "package_hash",
        "apply_hash",
    ],
    "properties": {
        "schema_version": {"const": "native-agent-runner.apply-result.v1"},
        "status": {"enum": ["dry_run", "applied", "conflict", "rejected"]},
        "applied_paths": {"type": "array", "items": {"type": "string"}},
        "skipped_paths": {"type": "array", "items": {"type": "string"}},
        "conflicts": {"type": "array", "items": {"type": "object"}},
        "approval_hash": {"type": "string"},
        "package_hash": {"type": "string"},
        "apply_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    },
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str


def validate_run_dir(run_dir: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    required_files = (
        "manifest.json",
        "workspace.index.json",
        "workspace.base.json",
        "events.jsonl",
        "transcript.jsonl",
        "metrics.json",
        "proposal.json",
        "diff.patch",
    )
    for name in required_files:
        if not run_dir.joinpath(name).exists():
            issues.append(ValidationIssue(name, "missing required file"))
    _validate_json_file(run_dir / "manifest.json", MANIFEST_SCHEMA, issues)
    _validate_json_file(run_dir / "workspace.index.json", WORKSPACE_INDEX_SCHEMA, issues)
    _validate_json_file(run_dir / "workspace.base.json", WORKSPACE_BASE_SCHEMA, issues)
    _validate_json_file(run_dir / "metrics.json", METRICS_SCHEMA, issues)
    _validate_json_file(run_dir / "proposal.json", PROPOSAL_SCHEMA, issues)
    _validate_manifest_workspace_index(run_dir, issues)
    _validate_manifest_workspace_base(run_dir, issues)
    _validate_proposal_hashes(run_dir, issues)
    status_path = run_dir / "status.json"
    if status_path.exists():
        _validate_json_file(status_path, STATUS_SCHEMA, issues)
    package_path = run_dir / "proposal.package.json"
    if package_path.exists():
        _validate_json_file(package_path, PACKAGE_SCHEMA, issues)
        _validate_package_hashes(run_dir, issues)
    approval_path = run_dir / "approval.json"
    if approval_path.exists():
        _validate_json_file(approval_path, APPROVAL_SCHEMA, issues)
        _validate_canonical_hash(approval_path, "approval_hash", issues)
    apply_result_path = run_dir / "apply-result.json"
    if apply_result_path.exists():
        _validate_json_file(apply_result_path, APPLY_RESULT_SCHEMA, issues)
        _validate_canonical_hash(apply_result_path, "apply_hash", issues)
    events_path = run_dir / "events.jsonl"
    if events_path.exists():
        _validate_jsonl_file(events_path, EVENT_SCHEMA, issues)
        _validate_event_data(events_path, issues)
    transcript_path = run_dir / "transcript.jsonl"
    if transcript_path.exists():
        _validate_jsonl_file(transcript_path, TRANSCRIPT_RECORD_SCHEMA, issues)
    jobs_dir = run_dir / "artifacts" / "jobs"
    if jobs_dir.exists():
        for job_path in sorted(jobs_dir.glob("*/job.json")):
            _validate_json_file(job_path, JOB_SCHEMA, issues)
    return issues


def _validate_json_file(path: Path, schema: dict[str, Any], issues: list[ValidationIssue]) -> None:
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(ValidationIssue(path.name, f"invalid JSON: {exc.msg}"))
        return
    _validate_object(payload, schema, issues, path.name)


def _validate_object(payload: Any, schema: dict[str, Any], issues: list[ValidationIssue], label: str) -> None:
    validator = Draft202012Validator(schema)
    for error in sorted(validator.iter_errors(payload), key=lambda item: list(item.path)):
        suffix = ".".join(str(part) for part in error.path)
        issue_path = f"{label}.{suffix}" if suffix else label
        issues.append(ValidationIssue(issue_path, error.message))


def _validate_jsonl_file(path: Path, schema: dict[str, Any], issues: list[ValidationIssue]) -> None:
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(ValidationIssue(f"{path.name}:{index}", f"invalid JSON: {exc.msg}"))
            continue
        _validate_object(payload, schema, issues, f"{path.name}:{index}")


def _validate_event_data(path: Path, issues: list[ValidationIssue]) -> None:
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # envelope pass already reported malformed lines
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        schema = EVENT_DATA_SCHEMAS.get(event_type) if isinstance(event_type, str) else None
        if schema is None:
            issues.append(
                ValidationIssue(f"{path.name}:{index}", f"no data schema for event type: {event_type!r}")
            )
            continue
        data = event.get("data")
        _validate_object(data if isinstance(data, dict) else {}, schema, issues, f"{path.name}:{index}.data")


def _validate_manifest_workspace_index(run_dir: Path, issues: list[ValidationIssue]) -> None:
    _validate_manifest_relative_file(run_dir, issues, "workspace_index_path", "workspace index file missing")


def _validate_manifest_workspace_base(run_dir: Path, issues: list[ValidationIssue]) -> None:
    _validate_manifest_relative_file(run_dir, issues, "workspace_base_path", "workspace base file missing")


def _validate_manifest_relative_file(
    run_dir: Path,
    issues: list[ValidationIssue],
    key: str,
    missing_message: str,
) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if not isinstance(manifest, dict):
        return
    rel = manifest.get(key)
    if not isinstance(rel, str):
        return
    try:
        safe_rel = normalize_workspace_path(rel)
    except Exception as exc:
        issues.append(ValidationIssue(f"manifest.json.{key}", str(exc)))
        return
    if safe_rel != rel.replace("\\", "/"):
        issues.append(ValidationIssue(f"manifest.json.{key}", f"{key} is not normalized"))
        return
    if not (run_dir / safe_rel).exists():
        issues.append(ValidationIssue(f"manifest.json.{key}", missing_message))


def _validate_proposal_hashes(run_dir: Path, issues: list[ValidationIssue]) -> None:
    proposal_path = run_dir / "proposal.json"
    if not proposal_path.exists():
        return
    try:
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if not isinstance(proposal, dict):
        return
    expected_proposal_hash = proposal.get("proposal_hash")
    actual_proposal_hash = canonical_sha256(proposal, drop=("proposal_hash", "updated_at"))
    if expected_proposal_hash != actual_proposal_hash:
        issues.append(ValidationIssue("proposal.json.proposal_hash", "proposal hash mismatch"))
    diff_rel = proposal.get("diff_path")
    if isinstance(diff_rel, str):
        diff_path = run_dir / diff_rel
        if diff_path.exists():
            actual_diff_hash = hashlib.sha256(diff_path.read_bytes()).hexdigest()
            if proposal.get("diff_sha256") != actual_diff_hash:
                issues.append(ValidationIssue("proposal.json.diff_sha256", "diff hash mismatch"))
    files = proposal.get("files")
    if isinstance(files, list):
        for index, file_info in enumerate(files):
            if not isinstance(file_info, dict):
                continue
            snapshot_path = file_info.get("snapshot_path")
            if not isinstance(snapshot_path, str):
                continue
            path = run_dir / snapshot_path
            if not path.exists():
                issues.append(ValidationIssue(f"proposal.json.files.{index}.snapshot_path", "snapshot missing"))
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if file_info.get("snapshot_sha256") != actual:
                issues.append(ValidationIssue(f"proposal.json.files.{index}.snapshot_sha256", "snapshot hash mismatch"))


def _validate_package_hashes(run_dir: Path, issues: list[ValidationIssue]) -> None:
    package_path = run_dir / "proposal.package.json"
    try:
        package = json.loads(package_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if not isinstance(package, dict):
        return
    if package.get("package_hash") != canonical_sha256(package, drop=("package_hash",)):
        issues.append(ValidationIssue("proposal.package.json.package_hash", "package hash mismatch"))
    seen: set[str] = set()
    for index, file_info in enumerate(package.get("files") or []):
        if not isinstance(file_info, dict):
            continue
        rel = file_info.get("path")
        if not isinstance(rel, str):
            continue
        if rel in seen:
            issues.append(ValidationIssue(f"proposal.package.json.files.{index}.path", "duplicate package path"))
        seen.add(rel)
        path = run_dir / rel
        if not path.exists() or not path.is_file():
            issues.append(ValidationIssue(f"proposal.package.json.files.{index}.path", "package file missing"))
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if file_info.get("sha256") != actual:
            issues.append(ValidationIssue(f"proposal.package.json.files.{index}.sha256", "package file hash mismatch"))


def _validate_canonical_hash(path: Path, hash_key: str, issues: list[ValidationIssue]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if not isinstance(payload, dict):
        return
    expected = payload.get(hash_key)
    actual = canonical_sha256(payload, drop=(hash_key,))
    if expected != actual:
        issues.append(ValidationIssue(f"{path.name}.{hash_key}", f"{hash_key} mismatch"))
