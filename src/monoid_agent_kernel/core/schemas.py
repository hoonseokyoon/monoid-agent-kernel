from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from monoid_agent_kernel.core._event_log import iter_committed_jsonl_records
from monoid_agent_kernel.core._json_schema import END_OF_INPUT
from monoid_agent_kernel.core._util import canonical_sha256, sha256_bytes
from monoid_agent_kernel.core.json_ingress import loads_json_ingress
from monoid_agent_kernel.core.model_calls import (
    MODEL_CALL_KIND,
    MODEL_CALLS_FILENAME,
    MODEL_CALLS_SCHEMA_VERSION,
)
from monoid_agent_kernel.core.model_content import MODEL_CONTENT_FILENAME
from monoid_agent_kernel.core.model_payloads import (
    MODEL_PAYLOADS_DIRNAME,
    MODEL_PAYLOADS_FILENAME,
    MODEL_PAYLOADS_SCHEMA_VERSION,
    MODEL_REQUEST_KIND,
    MODEL_RESPONSE_KIND,
    PAYLOAD_CHUNK_KIND,
    RESPONSE_MALFORMED,
    RESPONSE_REFERENCE,
    UNRECORDED_REASONS,
    is_chunk_sha256,
    reassemble_request_preimage,
    response_reference,
)
from monoid_agent_kernel.core._verified_file import read_verified_bytes
from monoid_agent_kernel.core.model_io import (
    DESTINATION_STATUSES,
    DIGEST_STATUSES,
    IDEMPOTENCY_KEY_JSON_PATTERN,
    MAX_MODEL_PAYLOAD_BYTES,
    RECORDED_DIGEST_BODY,
)
from monoid_agent_kernel.identifiers import namespaced_id, schema_version_property
from monoid_agent_kernel.workspace.paths import normalize_workspace_path


TIMESTAMP_PATTERN = rf"Z{END_OF_INPUT}"
"""Ends with the UTC designator -- unanchored at the front on purpose, the way it always was."""

EVENT_TYPE_PATTERN = rf"^[a-z]+(\.[a-z_]+)+{END_OF_INPUT}"
"""A dotted lowercase event name, whole and nothing after it."""

SHA256_PATTERN = rf"^{RECORDED_DIGEST_BODY}{END_OF_INPUT}"
"""A digest that must be present."""

OPTIONAL_SHA256_PATTERN = rf"^(|{RECORDED_DIGEST_BODY}){END_OF_INPUT}"
"""A digest that may not have been issued, where empty is the recorded spelling of absence.

Both forms compose the one body ``core/model_io.py`` owns (W7-4), the way the idempotency-key
forms do: these schema patterns and the ledger's mint guard are enforcers of one rule, and a
retyped twin regex is how enforcers drift."""


EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version",
        "event_id",
        "seq",
        "run_id",
        "timestamp",
        "type",
        "level",
        "data",
    ],
    "properties": {
        "schema_version": schema_version_property("event.v1"),
        "event_id": {"type": "string", "minLength": 1},
        "seq": {"type": "integer", "minimum": 1},
        "run_id": {"type": "string", "minLength": 1},
        "turn_id": {"type": ["string", "null"]},
        "parent_id": {"type": ["string", "null"]},
        "timestamp": {"type": "string", "pattern": TIMESTAMP_PATTERN},
        "type": {"type": "string", "pattern": EVENT_TYPE_PATTERN},
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
            # ``final_text`` stays accepted after v0.20 stops emitting model-authored text here.
            # ``validate_run_dir`` replays committed logs against these schemas, so removing the
            # property would fail every run directory written before the change — and kernel
            # strings ("Stopped after reaching max steps.") keep travelling inline regardless.
            "final_text": _STR,
            # Set instead of ``final_text`` when the text is the model's. The digest is
            # ``core.model_io.content_digest`` — canonical JSON, NOT a bare sha256 of the text —
            # and the text itself is resolved from the run-dir settled-text record.
            "final_text_digest": _STR,
            "final_text_len": _INT,
            "duration_s": _NUM,
            "diff_path": _STR,
            "proposal_path": _STR,
            "metrics_path": _STR,
        },
        required=("status",),
    ),
    "run.failed": _data_schema(
        {
            "error": _STR,
            "error_code": _STR,
            "type": _STR,
            "provider_error_code": _STR,
            "http_status": {"type": ["integer", "null"]},
            # The terminal twin of ``turn.failed`` carries the same classification it does:
            # ``fail_recoverable`` promotes one into the other, so a config-fixable failure that
            # a driver gave up on must still say it was config-fixable in the record of giving up.
            "retryable": _BOOL,
            "config_recoverable": _BOOL,
        },
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
    "session.state.changed": _data_schema(
        {"state": _STR, "from": _STR, "reason": _STR},
        required=("state",),
    ),
    "turn.settled": _data_schema(
        {
            "status": _STR,
            # Retained for the same reasons as on ``run.finished`` above.
            "final_text": _STR,
            "final_text_digest": _STR,
            "final_text_len": _INT,
            "error_code": _STR,
            "changed_paths": _STR_ARRAY,
            "output_validators": _INT,
            "output_retries": _INT,
        },
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
            "config_recoverable": _BOOL,
            # Whether the adapter's own retry budget was already spent before this park.
            "provider_retried": _BOOL,
            # What the refused call already cost. A failure *after* a billed answer is an
            # ordinary shape (the applied-parameters proof refusals are exactly that), and the
            # transcript twin written on the same failure has always recorded it. Named for the
            # kernel fact, not for the gateway wire's compat-frozen ``usage`` alias — the event
            # spells ``provider_error_code`` for the same reason.
            "provider_usage": _OBJ,
        },
        required=("error_code",),
    ),
    # ``reason`` here is a CAUSE vocabulary — what stopped the turn ("user_stop") — and it is
    # deliberately NOT ``Suspension.reason``, which is a PARK vocabulary naming the state the
    # session came to rest in ("interrupted"). One key name, two domains, on purpose: the event
    # answers "why did this stop", the park answers "where is the run now". A reader that joins
    # them by name is reading two different questions. See docs/CONTRACTS.md, event reads.
    "turn.interrupted": _data_schema(
        {"reason": _STR},
        required=(),
    ),
    # The interrupt's twin, and the same cause vocabulary ("user_pause"). The pause park used to
    # emit no event of its own — only a ``session.state.changed`` — so two sibling parks were not
    # observable the same way: a consumer watching the turn lane saw the stop and missed the
    # pause. Observability only; no projection consumes it.
    "turn.paused": _data_schema(
        {"reason": _STR},
        required=("reason",),
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
            # The priced sub-counts, each present only when the adapter reported one. All four
            # are billed differently from a plain input token, so a live consumer that sees only
            # ``reasoning_tokens`` cannot show what a cache-heavy run actually cost.
            "cache_read_tokens": _INT,
            "cache_creation_tokens": _INT,
            "reasoning_tokens": _INT,
            "audio_tokens": _INT,
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
            "immediate_tools": _OBJ_ARRAY,
            "searchable_count": _INT,
            "searchable_tools": _OBJ_ARRAY,
            "hidden_count": _INT,
            "hidden_binding_ids": _STR_ARRAY,
            "authorizations": _OBJ,
            "delta_notice": _STR,
            "surface_warnings": _STR_ARRAY,
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
    "capability.requested": _data_schema(
        {"capability": _STR, "binding_id": _STR, "request_id": _STR, "reason": _STR, "scope": _OBJ},
        required=("capability",),
    ),
    "capability.granted": _data_schema(
        {
            "capability": _STR,
            "binding_id": _STR,
            "lease_id": _STR,
            "expires_at": _NUM,
            "scope": _OBJ,
        },
        required=("capability",),
    ),
    "capability.denied": _data_schema(
        {"capability": _STR, "binding_id": _STR, "reason": _STR, "retryable": _BOOL},
        required=("capability",),
    ),
    "capability.revoked": _data_schema(
        {"capability": _STR, "lease_id": _STR, "reason": _STR, "scope": _OBJ},
        required=("capability",),
    ),
    "capability.rotated": _data_schema(
        {"capability": _STR, "old_lease_id": _STR, "new_lease_id": _STR, "expires_at": _NUM},
        required=("capability",),
    ),
    "control.command.received": _data_schema(
        {
            "command_id": _STR,
            "command": _STR,
            "target_run_id": _STR,
            "actor": _STR,
            "reason": _STR,
            "token_sha256": _STR,
            "idempotency_key": _STR,
            "args_keys": _STR_ARRAY,
        },
        required=("command_id", "command", "target_run_id"),
    ),
    "control.command.completed": _data_schema(
        {
            "command_id": _STR,
            "command": _STR,
            "target_run_id": _STR,
            "actor": _STR,
            "idempotency_key": _STR,
            "token_sha256": _STR,
            "status": _STR,
            "result_code": _STR,
            "state": _STR_NULL,
            "duration_ms": _NUM,
        },
        required=(
            "command_id",
            "command",
            "target_run_id",
            "idempotency_key",
            "status",
            "result_code",
        ),
    ),
    "control.command.failed": _data_schema(
        {
            "command_id": _STR,
            "command": _STR,
            "target_run_id": _STR,
            "actor": _STR,
            "idempotency_key": _STR,
            "token_sha256": _STR,
            "status": _STR,
            "error": _STR,
            "error_code": _STR,
            "failure_code": _STR,
            "duration_ms": _NUM,
        },
        required=(
            "command_id",
            "command",
            "target_run_id",
            "idempotency_key",
            "status",
            "error_code",
            "failure_code",
        ),
    ),
    "outbox.requested": _data_schema(
        {"request_id": _STR, "destination": _STR, "capability": _STR, "traceparent": _STR},
        required=("request_id",),
    ),
    "outbox.dispatched": _data_schema(
        {
            "request_id": _STR,
            "destination": _STR,
            "reference": _STR,
            "attempts": _NUM,
            "traceparent": _STR,
        },
        required=("request_id",),
    ),
    "outbox.failed": _data_schema(
        {
            "request_id": _STR,
            "destination": _STR,
            "reason": _STR,
            "attempts": _NUM,
            "traceparent": _STR,
        },
        required=("request_id",),
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
        {"artifact_id": _STR, "path": _STR, "kind": _STR, "metadata": {"type": "object"}},
        required=("artifact_id",),
    ),
    "plan.updated": _data_schema(
        {"items": _OBJ_ARRAY, "truncated_items": _INT},
        required=("items",),
    ),
    "output.validator.satisfied": _data_schema(
        {"validator_id": _STR},
        required=("validator_id",),
    ),
    "output.validation.failed": _data_schema(
        # Carries either a refusal/truncation ``reason`` or a validator ``attempt`` + ``failures``.
        {"reason": _STR, "attempt": _INT, "failures": _OBJ_ARRAY},
        additional=True,
    ),
    "output.validator.skipped": _data_schema(
        {"validator_id": _STR, "reason": _STR},
        required=("validator_id",),
    ),
    "output.validator.error": _data_schema(
        {"validator_id": _STR, "error": _STR},
        required=("validator_id",),
    ),
    "output.validator.exhausted": _data_schema(
        # Roll-up + per-attempt history of which validators kept failing (diagnose contradictions).
        {"retries": _INT, "failures_by_validator": _OBJ, "history": _OBJ_ARRAY},
        additional=True,
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
        "schema_version": schema_version_property("manifest.v1"),
        "run_id": {"type": "string", "minLength": 1},
        "created_at": {"type": "string", "pattern": TIMESTAMP_PATTERN},
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
        "schema_version": schema_version_property("workspace-base.v1"),
        "run_id": {"type": "string", "minLength": 1},
        "created_at": {"type": "string", "pattern": TIMESTAMP_PATTERN},
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
                    "sha256": {"type": ["string", "null"], "pattern": SHA256_PATTERN},
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
        "schema_version": schema_version_property("workspace-index.v1"),
        "run_id": {"type": "string", "minLength": 1},
        "generated_at": {"type": "string", "pattern": TIMESTAMP_PATTERN},
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
                    "sha256": {"type": ["string", "null"], "pattern": SHA256_PATTERN},
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
                # The failure record's writer has always emitted this beside ``retryable``; the
                # branch only stayed valid because ``additionalProperties`` is True here.
                "config_recoverable": {"type": "boolean"},
                # Written by BOTH model_turn records (success and failure): the private replay
                # artifact of a retried-then-successful call used to read as a clean single
                # attempt, which is exactly the case where the retry evidence matters most.
                "provider_retried": {"type": "boolean"},
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
        {
            # Model-authored text a run settled on, keyed by its content digest so a settle event
            # carrying ``final_text_digest`` can resolve back to it. Distinct from ``model_turn``
            # above: that records one model response per step, while ``state.final_text`` is
            # frequently not the last of those (a ``run.finish`` summary, a validator repair), and
            # is what the settle events actually publish.
            "type": "object",
            "required": ["kind", "final_text", "final_text_digest", "final_text_len"],
            "properties": {
                "kind": {"const": "settled_text"},
                "final_text": {"type": "string"},
                # ``core.model_io.content_digest`` — canonical JSON under a shape key, NOT a bare
                # sha256 of the text. Recompute with that function or the join silently misses.
                "final_text_digest": {"type": "string"},
                "final_text_len": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": True,
        },
    ]
}

MODEL_CONTENT_RECORD_SCHEMA: dict[str, Any] = {
    "oneOf": [
        {
            "type": "object",
            "required": [
                "schema_version",
                "kind",
                "run_id",
                "root_run_id",
                "turn_id",
                "stream_id",
                "step",
                "provider",
                "model",
                "started_at",
            ],
            "properties": {
                "schema_version": schema_version_property("model-content.v1"),
                "kind": {"const": "stream_opened"},
                "run_id": {"type": "string", "minLength": 1},
                "root_run_id": {"type": "string", "minLength": 1},
                "turn_id": {"type": "string", "minLength": 1},
                "stream_id": {"type": "string", "minLength": 1},
                "step": {"type": "integer", "minimum": 1},
                "provider": {"type": ["string", "null"]},
                "model": {"type": ["string", "null"]},
                "started_at": {"type": "string", "pattern": TIMESTAMP_PATTERN},
            },
            "additionalProperties": False,
        },
        {
            "type": "object",
            "required": [
                "schema_version",
                "kind",
                "run_id",
                "stream_id",
                "segment_index",
                "channel",
                "text",
                "text_len",
                "emitted_at",
            ],
            "properties": {
                "schema_version": schema_version_property("model-content.v1"),
                "kind": {"const": "stream_segment"},
                "run_id": {"type": "string", "minLength": 1},
                "stream_id": {"type": "string", "minLength": 1},
                "segment_index": {"type": "integer", "minimum": 0},
                "channel": {"enum": ["output", "reasoning"]},
                "text": {"type": "string"},
                "text_len": {"type": "integer", "minimum": 0},
                "emitted_at": {"type": "string", "pattern": TIMESTAMP_PATTERN},
            },
            "additionalProperties": False,
        },
        {
            "type": "object",
            "required": [
                "schema_version",
                "kind",
                "run_id",
                "stream_id",
                "status",
                "final_text",
                "usage",
                "error_code",
                "finished_at",
            ],
            "properties": {
                "schema_version": schema_version_property("model-content.v1"),
                "kind": {"const": "stream_closed"},
                "run_id": {"type": "string", "minLength": 1},
                "stream_id": {"type": "string", "minLength": 1},
                "status": {
                    "enum": ["completed", "interrupted", "failed", "cancelled", "timed_out"]
                },
                "final_text": {"type": ["string", "null"]},
                "usage": {"type": ["object", "null"]},
                "error_code": {"type": ["string", "null"]},
                "retryable": {"type": "boolean"},
                # Additive and optional: a sidecar written before this key existed still
                # validates, and the reader defaults it to False. Declared in the same change as
                # the writer because ``additionalProperties`` is False here — a record key with
                # no schema slot is a validation failure, not a forward-compatible extra.
                "config_recoverable": {"type": "boolean"},
                "finished_at": {"type": "string", "pattern": TIMESTAMP_PATTERN},
            },
            "additionalProperties": False,
        },
        {
            "type": "object",
            "required": [
                "schema_version",
                "kind",
                "run_id",
                "final_text",
                "final_text_digest",
                "final_text_len",
                "recorded_at",
            ],
            "properties": {
                "schema_version": schema_version_property("model-content.v1"),
                "kind": {"const": "settled_text"},
                "run_id": {"type": "string", "minLength": 1},
                "final_text": {"type": "string"},
                "final_text_digest": {"type": "string", "pattern": SHA256_PATTERN},
                "final_text_len": {"type": "integer", "minimum": 0},
                "recorded_at": {"type": "string", "pattern": TIMESTAMP_PATTERN},
            },
            "additionalProperties": False,
        },
    ]
}

# The private model-call ledger. A literal single-element enum rather than
# ``schema_version_property``: that helper emits the legacy namespace beside the current one, and
# this artifact has never existed under it — advertising a reader for records that cannot exist is
# a false compatibility claim, and the ledger's own reader-version pin refuses it.
_MODEL_CALL_CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["provider", "model", "reasoning"],
    "properties": {
        "provider": {"type": "string"},
        "model": {"type": "string"},
        "reasoning": {
            "type": "object",
            "required": ["effort", "summary", "on_unsupported"],
            "properties": {
                "effort": {"type": "string"},
                "summary": {"type": "string"},
                "on_unsupported": {"enum": ["fail", "omit"]},
            },
            "additionalProperties": False,
        },
        # Omitted when the caller configured no sampling control, which is why it is not required.
        "generation": {
            "type": "object",
            "required": ["temperature", "top_p", "max_output_tokens", "on_unsupported"],
            "properties": {
                "temperature": {"type": ["number", "null"]},
                "top_p": {"type": ["number", "null"]},
                "max_output_tokens": {"type": ["integer", "null"]},
                "on_unsupported": {"enum": ["fail", "omit"]},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

MODEL_CALLS_RECORD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version",
        "kind",
        "run_id",
        "root_run_id",
        "call_index",
        "recorded_at",
        "context",
        "model",
        "provider_name",
        "prompt_digest",
        "request_digest",
        "digest_generation",
        "digest_status",
        "destination_status",
        "stop_reason",
        "usage",
        "latency_ms",
        "attempts",
        "provider_retried",
        "error_code",
        "provider_error_code",
        "retryable",
        "config_recoverable",
        "http_status",
        "capture_downgrades",
    ],
    "properties": {
        "schema_version": {"enum": [MODEL_CALLS_SCHEMA_VERSION]},
        "kind": {"const": MODEL_CALL_KIND},
        "run_id": {"type": "string", "minLength": 1},
        "root_run_id": {"type": "string", "minLength": 1},
        "call_index": {"type": "integer", "minimum": 0},
        "recorded_at": {"type": "string", "pattern": TIMESTAMP_PATTERN},
        "context": {
            "type": "object",
            "required": [
                "run_id",
                "skill_id",
                "skill_digest",
                "step_id",
                "attempt",
                "batch_id",
                "item_id",
                "case_id",
                "traceparent",
                "tracestate",
                "attributes",
            ],
            "properties": {
                "run_id": {"type": "string"},
                "skill_id": {"type": "string"},
                "skill_digest": {"type": "string"},
                "step_id": {"type": "string"},
                "attempt": {"type": "integer", "minimum": 1},
                "batch_id": {"type": "string"},
                "item_id": {"type": "string"},
                "case_id": {"type": "string"},
                "traceparent": {"type": "string"},
                "tracestate": {"type": "string"},
                "attributes": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
            "additionalProperties": False,
        },
        "model": _MODEL_CALL_CONFIG_SCHEMA,
        "provider_name": {"type": "string"},
        # Empty is a valid answer and ``digest_status`` says which reason it is, so the
        # pattern admits both the key and its absence rather than requiring one.
        "prompt_digest": {"type": "string", "pattern": OPTIONAL_SHA256_PATTERN},
        "request_digest": {"type": "string", "pattern": OPTIONAL_SHA256_PATTERN},
        "digest_generation": {"type": "string"},
        "digest_status": {"enum": list(DIGEST_STATUSES)},
        # Declared and not required: the writer always emits it -- the in-band empty string
        # is its absence spelling -- so absence on a line means exactly one thing, a writer
        # that predates the field, and ``validate_run_dir`` keeps passing directories
        # pre-W7-3 builds filled. (``attempt_log`` below spells absence by omitting the key
        # instead; the asymmetry is each field's own rule.)
        #
        # Format-constrained like the two digests beside it rather than left an open string:
        # this key is a token the kernel MINTS to a closed shape, not an open vocabulary a
        # provider may extend (which is why ``stop_reason`` and ``provider_error_code`` carry
        # no pattern and this does). The pattern is DERIVED from the same body
        # ``is_valid_idempotency_key`` compiles, so an imported or third-party line cannot be
        # certified against a rule the rest of the kernel does not hold. Empty is admitted
        # explicitly, the ``^(|...)$`` idiom the digests use: a refused call was never keyed.
        "idempotency_key": {"type": "string", "pattern": IDEMPOTENCY_KEY_JSON_PATTERN},
        "destination_status": {"enum": list(DESTINATION_STATUSES)},
        "stop_reason": {"type": "string"},
        "usage": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 0}},
        "latency_ms": {"type": "integer", "minimum": 0},
        "attempts": {"type": "integer", "minimum": 0},
        # Declared but not required: the sweep validator reads ledgers earlier v0.21 builds
        # filled, and absence means nothing was itemized -- a writer that predates the field,
        # a refused call that never dispatched, or a receipt built without a log (this writer
        # omits an empty log; the ones before it spelled the same value ``[]``, which stays
        # legal and is why no ``minItems`` appears here). What a JSON Schema cannot state is
        # how a *non-empty* log stands to the line around it, which is
        # ``_validate_model_call_attempt_logs``'s job. A present entry is written whole or
        # refused: the closed shape is the record's own rule, one level down.
        "attempt_log": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "index",
                    "elapsed_ms",
                    "error_code",
                    "provider_error_code",
                    "retryable",
                    "config_recoverable",
                    "http_status",
                    "provider_retried",
                    "usage",
                    "stream_committed",
                ],
                "properties": {
                    "index": {"type": "integer", "minimum": 1},
                    "elapsed_ms": {"type": "integer", "minimum": 0},
                    "error_code": {"type": "string"},
                    "provider_error_code": {"type": "string"},
                    "retryable": {"type": "boolean"},
                    "config_recoverable": {"type": "boolean"},
                    "http_status": {"type": ["integer", "null"]},
                    "provider_retried": {"type": "boolean"},
                    "usage": {
                        "type": "object",
                        "additionalProperties": {"type": "integer", "minimum": 0},
                    },
                    "stream_committed": {"type": "boolean"},
                    # W7-2: declared and not required -- an entry a W7-1 writer filled carries
                    # ten keys and stays valid, absence meaning the line predates the field.
                    # Integer only, never null: no writer omits by writing null, and the reader
                    # (``ModelCallAttempt.from_json``) refuses it under the same rule.
                    "backoff_ms": {"type": "integer", "minimum": 0},
                },
                "additionalProperties": False,
            },
        },
        "provider_retried": {"type": "boolean"},
        "error_code": {"type": "string"},
        "provider_error_code": {"type": "string"},
        "retryable": {"type": "boolean"},
        "config_recoverable": {"type": "boolean"},
        "http_status": {"type": ["integer", "null"]},
        "capture_downgrades": {"type": "integer", "minimum": 0},
    },
    "additionalProperties": False,
}


def _payloads_envelope(kind: str) -> dict[str, Any]:
    return {
        "schema_version": {"enum": [MODEL_PAYLOADS_SCHEMA_VERSION]},
        "kind": {"const": kind},
        "run_id": {"type": "string", "minLength": 1},
        "root_run_id": {"type": "string", "minLength": 1},
        "recorded_at": {"type": "string", "pattern": TIMESTAMP_PATTERN},
    }


_PAYLOADS_ENVELOPE_KEYS = ["schema_version", "kind", "run_id", "root_run_id", "recorded_at"]

# Three kinds under one namespace, discriminated the way MODEL_CONTENT_RECORD_SCHEMA's four are:
# oneOf with a const kind per branch, additionalProperties refused per branch, and a literal
# single-element schema_version enum because this artifact never existed under the legacy prefix
# (the model-calls ledger states the same rule).
MODEL_PAYLOADS_RECORD_SCHEMA: dict[str, Any] = {
    "oneOf": [
        {
            "type": "object",
            "required": [*_PAYLOADS_ENVELOPE_KEYS, "sha256", "text"],
            "properties": {
                **_payloads_envelope(PAYLOAD_CHUNK_KIND),
                "sha256": {"type": "string", "pattern": SHA256_PATTERN},
                "text": {"type": "string"},
            },
            "additionalProperties": False,
        },
        {
            "type": "object",
            "required": [
                *_PAYLOADS_ENVELOPE_KEYS,
                "request_digest",
                "digest_generation",
                "refs",
                "payload",
            ],
            "properties": {
                **_payloads_envelope(MODEL_REQUEST_KIND),
                # Never empty: a keyless call has nothing to file a preimage under, so the
                # record simply does not exist (unlike the response branch below).
                "request_digest": {"type": "string", "pattern": SHA256_PATTERN},
                "digest_generation": {"type": "string", "minLength": 1},
                "refs": {"type": "boolean"},
                # Deliberately untyped. The recipe arm always produces an object, but the
                # verbatim arm exists for "a preimage the recipe shape does not fit", and a
                # future digest generation need not wrap its terms at all. What makes a
                # request record valid is that it reassembles to its digest, which
                # ``_validate_model_payload_digests`` checks; a type here would reject a
                # faithful record for the shape of bytes it is faithful to.
                "payload": {},
            },
            "additionalProperties": False,
        },
        {
            "type": "object",
            "required": [
                *_PAYLOADS_ENVELOPE_KEYS,
                "call_index",
                "request_digest",
                "unrecorded_reason",
                "response",
            ],
            "properties": {
                **_payloads_envelope(MODEL_RESPONSE_KIND),
                "call_index": {"type": "integer", "minimum": 0},
                # Empty is legal here: the ledger line this index joins says why there was no
                # key (its ``digest_status``), and this record still names the answer.
                "request_digest": {"type": "string", "pattern": OPTIONAL_SHA256_PATTERN},
                "unrecorded_reason": {"enum": list(UNRECORDED_REASONS)},
                # The inline body, a chunk reference to an offloaded one, or null with
                # ``unrecorded_reason`` saying why. Body keys are content, not contract, so the
                # object arm stays open the way the request ``payload`` does.
                "response": {"type": ["object", "null"]},
            },
            "additionalProperties": False,
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
        "schema_version": schema_version_property("proposal.v2"),
        "run_id": {"type": "string", "minLength": 1},
        "updated_at": {"type": "number"},
        "mode": {"enum": ["read-only", "propose", "apply"]},
        "proposal_hash": {"type": "string", "pattern": SHA256_PATTERN},
        "diff_path": {"type": "string"},
        "diff_bytes": {"type": "integer", "minimum": 0},
        "diff_sha256": {"type": "string", "pattern": SHA256_PATTERN},
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
        # The failure classification, declared with its writer (the ``stream_closed``
        # precedent: declare even under ``additionalProperties: True``, because an open cap is
        # a tolerance, not a declaration). The code/status pair is written whenever the run
        # recorded provider detail; the two booleans only on a failed run, where the state
        # they are read from is classified fresh.
        "provider_error_code": {"type": "string"},
        "provider_http_status": {"type": ["integer", "null"]},
        "retryable": {"type": "boolean"},
        "config_recoverable": {"type": "boolean"},
    },
    "additionalProperties": True,
}

STATUS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["run_id", "state", "terminal", "last_event_seq", "last_event_type", "updated_at"],
    "properties": {
        "run_id": {"type": "string", "minLength": 1},
        "state": {"type": "string"},
        "terminal": {"type": "boolean"},
        # ``minimum: 0``, not 1: the event sink always writes >= 1, but the failure-quarantine
        # writer (``run_state.write_failure_status_artifact``) can mint this artifact over a
        # run that never wrote status.json, and its honest seed is 0 — "no committed event
        # known to this writer". Every reader already accepts 0 (and reconciles against the
        # committed log tail).
        "last_event_seq": {"type": "integer", "minimum": 0},
        "last_event_type": {"type": "string"},
        "updated_at": {"type": "string"},
        # The classification a parked ``turn.failed`` writes into this artifact (declared
        # with the writer, the same rule as METRICS_SCHEMA above). Cleared on unpark and
        # healed at a non-failed terminal, so absence means "no live failure to classify" —
        # which is also what absence on a pre-v0.21 artifact meant.
        "provider_error_code": {"type": "string"},
        "http_status": {"type": ["integer", "null"]},
        "retryable": {"type": "boolean"},
        "config_recoverable": {"type": "boolean"},
        "provider_retried": {"type": "boolean"},
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
        "schema_version": schema_version_property("background-job.v1"),
        "job_id": {"type": "string", "minLength": 1},
        # `BackgroundJob.to_json` has written `kind` since the tool bundle was widened, and this
        # schema is `additionalProperties: false` -- so `monoid validate` reported
        # "Additional properties are not allowed ('kind' was unexpected)" on every run that started
        # a background job, and no test noticed because none validated a run directory that had
        # one. Declared optional rather than required: a `background-job.v1` artifact written
        # before that change has no `kind`, and this schema still has to read it.
        "kind": {"type": "string"},
        "command": {"type": "string"},
        "command_preview": {"type": "string"},
        "cwd": {"type": "string"},
        "status": {
            "enum": ["running", "exited", "timed_out", "cancelled", "output_limited", "failed"]
        },
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

PUBLIC_JOB_SCHEMA_VERSION = namespaced_id("public-background-job.v1")
_PUBLIC_PATH_PREVIEW_SCHEMA: dict[str, Any] = {
    "oneOf": [
        {"type": "string"},
        {
            "type": "object",
            "required": ["redacted", "type", "bytes"],
            "properties": {
                "redacted": {"const": True},
                "type": {"const": "str"},
                "bytes": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": False,
        },
        {
            "type": "object",
            "required": ["type", "preview", "bytes", "truncated"],
            "properties": {
                "type": {"const": "str"},
                "preview": {"type": "string"},
                "bytes": {"type": "integer", "minimum": 0},
                "truncated": {"const": True},
            },
            "additionalProperties": False,
        },
    ]
}

# The public projection is a different wire shape from the durable artifact: it drops `command`,
# transforms paths, and identifies the input version separately. Giving the transformed object the
# durable `background-job.v1` discriminator made every response invalid against its own schema.
PUBLIC_JOB_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version",
        "artifact_schema_version",
        "job_id",
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
        "schema_version": {"enum": [PUBLIC_JOB_SCHEMA_VERSION]},
        "artifact_schema_version": schema_version_property("background-job.v1"),
        "job_id": JOB_SCHEMA["properties"]["job_id"],
        "kind": JOB_SCHEMA["properties"]["kind"],
        "command_preview": {"type": "string"},
        "cwd": _PUBLIC_PATH_PREVIEW_SCHEMA,
        "status": JOB_SCHEMA["properties"]["status"],
        "started_at": JOB_SCHEMA["properties"]["started_at"],
        "finished_at": JOB_SCHEMA["properties"]["finished_at"],
        "duration_s": JOB_SCHEMA["properties"]["duration_s"],
        "exit_code": JOB_SCHEMA["properties"]["exit_code"],
        "timed_out": JOB_SCHEMA["properties"]["timed_out"],
        "output_truncated": JOB_SCHEMA["properties"]["output_truncated"],
        "error": {"type": "string"},
        "changed_paths": JOB_SCHEMA["properties"]["changed_paths"],
        "stdout_path": {"type": "string"},
        "stderr_path": {"type": "string"},
        "stdout_bytes": JOB_SCHEMA["properties"]["stdout_bytes"],
        "stderr_bytes": JOB_SCHEMA["properties"]["stderr_bytes"],
        "requested_timeout_s": JOB_SCHEMA["properties"]["requested_timeout_s"],
        "effective_timeout_s": JOB_SCHEMA["properties"]["effective_timeout_s"],
        "requested_max_output_bytes": JOB_SCHEMA["properties"]["requested_max_output_bytes"],
        "effective_max_output_bytes": JOB_SCHEMA["properties"]["effective_max_output_bytes"],
        "requested_startup_wait_s": JOB_SCHEMA["properties"]["requested_startup_wait_s"],
        "effective_startup_wait_s": JOB_SCHEMA["properties"]["effective_startup_wait_s"],
        "execution_workspace": JOB_SCHEMA["properties"]["execution_workspace"],
        "resume_on_exit": JOB_SCHEMA["properties"]["resume_on_exit"],
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
        "schema_version": schema_version_property("proposal-package.v1"),
        "run_id": {"type": "string", "minLength": 1},
        "created_at": {"type": "string"},
        "proposal_hash": {"type": "string", "pattern": SHA256_PATTERN},
        "diff_sha256": {"type": "string", "pattern": SHA256_PATTERN},
        "package_hash": {"type": "string", "pattern": SHA256_PATTERN},
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
                    "sha256": {"type": "string", "pattern": SHA256_PATTERN},
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
        "schema_version": schema_version_property("approval.v1"),
        "approval_id": {"type": "string"},
        "decision": {"enum": ["approved", "rejected"]},
        "package_hash": {"type": "string", "pattern": SHA256_PATTERN},
        "proposal_hash": {"type": "string", "pattern": SHA256_PATTERN},
        "approved_paths": {"type": "array", "items": {"type": "string"}},
        "rejected_paths": {"type": "array", "items": {"type": "string"}},
        "approver_id": {"type": "string"},
        "approved_at": {"type": "string"},
        "note": {"type": "string"},
        "approval_hash": {"type": "string", "pattern": SHA256_PATTERN},
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
        "schema_version": schema_version_property("apply-result.v1"),
        "status": {"enum": ["dry_run", "applied", "conflict", "rejected"]},
        "applied_paths": {"type": "array", "items": {"type": "string"}},
        "skipped_paths": {"type": "array", "items": {"type": "string"}},
        "conflicts": {"type": "array", "items": {"type": "object"}},
        "approval_hash": {"type": "string"},
        "package_hash": {"type": "string"},
        "apply_hash": {"type": "string", "pattern": SHA256_PATTERN},
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
        _validate_event_file(events_path, issues)
    transcript_path = run_dir / "transcript.jsonl"
    if transcript_path.exists():
        _validate_jsonl_file(transcript_path, TRANSCRIPT_RECORD_SCHEMA, issues)
        _validate_settled_text_digests(transcript_path, issues)
    model_content_path = run_dir / MODEL_CONTENT_FILENAME
    if model_content_path.exists():
        # Both content-classified artifacts, so both redact: binding this on one of the two
        # would be the twin miss this package keeps making.
        _validate_jsonl_file(
            model_content_path, MODEL_CONTENT_RECORD_SCHEMA, issues, redact_instance=True
        )
        _validate_settled_text_digests(model_content_path, issues)
    # Optional like the content sidecar beside it, and for the same reason: it exists only for a
    # run that asked for it, so its absence is a configuration, not a defect. No digest
    # recomputation pass -- the ledger holds no content-addressed field -- but it does now carry
    # claims that span two of its own values, and a schema cannot relate one entry to another.
    model_calls_path = run_dir / MODEL_CALLS_FILENAME
    if model_calls_path.exists():
        _validate_jsonl_file(model_calls_path, MODEL_CALLS_RECORD_SCHEMA, issues)
        _validate_model_call_attempt_logs(model_calls_path, issues)
    # Optional for the same reason as its two sidecar siblings. Unlike the ledger, this one DOES
    # get a recomputation pass: the corpus's whole contract is that every request record
    # reassembles to the exact bytes its key was taken over, and a validator that only
    # shape-checked would bless a corpus that cannot honor it.
    model_payloads_path = run_dir / MODEL_PAYLOADS_FILENAME
    if model_payloads_path.exists():
        _validate_jsonl_file(
            model_payloads_path, MODEL_PAYLOADS_RECORD_SCHEMA, issues, redact_instance=True
        )
        _validate_model_payload_digests(run_dir, issues)
    jobs_dir = run_dir / "artifacts" / "jobs"
    if jobs_dir.exists():
        for job_path in sorted(jobs_dir.glob("*/job.json")):
            _validate_json_file(job_path, JOB_SCHEMA, issues)
    return issues


def _validate_settled_text_digests(path: Path, issues: list[ValidationIssue]) -> None:
    """Recompute each ``settled_text`` record's digest and length.

    The schema can only say ``final_text_digest`` is *a string*, but the reader
    (``reference.backend.content_hydration``) rejects any record whose text does not hash to the
    digest it claims. Without this check the two disagree in the worst direction: a record whose
    text was altered while its digest was left alone stays schema-valid, so ``monoid validate``
    reports the run clean while an entitled reader silently resolves nothing and the final answer
    is gone. Validation must not certify a record the reader will refuse.

    Uses ``content_digest``/``content_length`` — the same functions the writer and the reader use —
    rather than reimplementing the hash, so the three cannot drift apart.
    """
    from monoid_agent_kernel.core.model_io import content_digest, content_length

    try:
        raw = path.read_bytes()
    except OSError:
        return  # already reported by the schema pass
    for index, raw_line in enumerate(raw.split(b"\n"), start=1):
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            continue  # already reported by the schema pass
        if not line.strip():
            continue
        try:
            record = loads_json_ingress(line)
        except (ValueError, RecursionError):
            continue  # already reported by the schema pass
        if not isinstance(record, dict) or record.get("kind") != "settled_text":
            continue
        text = record.get("final_text")
        if not isinstance(text, str):
            continue  # shape is the schema's job
        label = f"{path.name}:{index}"
        claimed = record.get("final_text_digest")
        if claimed != content_digest(text):
            issues.append(ValidationIssue(label, "settled_text digest does not match final_text"))
        claimed_len = record.get("final_text_len")
        if claimed_len != content_length(text):
            issues.append(ValidationIssue(label, "settled_text length does not match final_text"))


def _validate_model_call_attempt_logs(path: Path, issues: list[ValidationIssue]) -> None:
    """Relate each ledger line's ``attempt_log`` to the three record fields it itemizes, to
    the one rule its entries owe each other, and to the one spelling an empty log may take.

    A JSON Schema validates every entry against its own shape and can say nothing about how one
    entry stands to another, or to the record around it. ``ModelCallReceipt.__post_init__``
    refuses three cross-entry claims -- indices exactly ``1..attempts`` in order, entry usage
    summing to the receipt's, and no wait recorded before the first dispatch -- and nothing
    constructs a receipt on the way through ``monoid validate``, which reads the ledger as JSON.
    So a line the record could not have produced (``attempts: 2`` under indices ``[1, 1]``;
    entries billing 3 beside a total of 99; a wait booked ahead of the call's own first reach
    into the adapter) passed the sweep clean, which is the one answer a validator must never
    give about a corrupt artifact.

    One claim is this surface's alone -- the dispatches, plus the waits between them, fitting
    inside the call's own ``latency_ms`` -- which is a fact about when the two values exist
    rather than an omission. ``model_call.py`` attaches the log at its failure and its
    answering exit while ``latency_ms`` is still the field's default; ``_publish`` stamps the
    measured duration afterwards, on every exit. A constructor check would therefore fire on
    every retried call, weighing real dispatch durations against a latency of zero. This line is
    the first place both values are settled and present together. Consumers that lay the entries
    out on a timeline -- the OTel preset's per-attempt children -- bound their own arithmetic
    rather than trust the record, because reporting a corrupt line is not the same as stopping
    one from being read back.

    The relationship pass the ledger did not have, alongside the ones its sidecar siblings do
    (manifest against workspace index, proposal against its hashes, settled text against its
    digests, payload records against their keys). An unitemized log makes no claim to check,
    in either of its two spellings: absence, which is what this build writes and what every
    record predating the field carries, and a present ``[]``, which is what builds between
    W7-1 and W7-4 wrote for the same value at whatever ``attempts`` the receipt held. Both
    are read as "nothing itemized" and neither is reported -- the claims below are about the
    entries a log actually names.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return  # already reported by the schema pass
    for index, raw_line in enumerate(raw.split(b"\n"), start=1):
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            continue  # already reported by the schema pass
        if not line.strip():
            continue
        try:
            record = loads_json_ingress(line)
        except (ValueError, RecursionError):
            continue  # already reported by the schema pass
        if not isinstance(record, dict):
            continue
        entries = record.get("attempt_log")
        if not isinstance(entries, list):
            continue  # absent, or shape the schema already refused
        label = f"{path.name}:{index}"
        attempts = record.get("attempts")
        counted = isinstance(attempts, int) and not isinstance(attempts, bool)
        if not entries:
            # Nothing to relate, at any count. ``[]`` is what every build before W7-4 wrote
            # for an empty log -- the projection emitted the key unconditionally, and a
            # receipt without entries is legal at any ``attempts`` -- so the count beside it
            # says nothing about the writer. Reporting the positive-count arm would convict
            # directories the previous build filled while certifying the zero arm written by
            # the same line of code.
            continue
        if not all(isinstance(entry, dict) for entry in entries):
            continue  # shape is the schema's job
        if counted:
            named = [entry.get("index") for entry in entries]
            if named != list(range(1, attempts + 1)):
                issues.append(
                    ValidationIssue(
                        label, "attempt_log must name every attempt exactly once, in order"
                    )
                )
        # The wait that separates two dispatches cannot precede the first one. The record refuses
        # this itself; repeated here because nothing constructs a record on this path, and a line
        # can satisfy every other claim -- indices in order, usage summing, durations fitting --
        # while still reporting a wait before the call had done anything to wait after.
        first_backoff = entries[0].get("backoff_ms")
        if (
            isinstance(first_backoff, int)
            and not isinstance(first_backoff, bool)
            and first_backoff != 0
        ):
            issues.append(
                ValidationIssue(
                    label,
                    "attempt_log first entry backoff_ms must be 0: "
                    "nothing precedes the first dispatch",
                )
            )
        # Checked before the usage block, which returns early on a shape the schema owns: two
        # independent claims about one line, and the second must not be skipped by the first's
        # excuse for leaving.
        latency_ms = record.get("latency_ms")
        occupied: int | None = _attempt_timeline_ms(entries)
        if (
            occupied is not None
            and isinstance(latency_ms, int)
            and not isinstance(latency_ms, bool)
            and occupied > latency_ms
        ):
            issues.append(
                ValidationIssue(
                    label, "attempt_log dispatches and waits must fit inside the line's latency_ms"
                )
            )
        usage = record.get("usage")
        if not isinstance(usage, dict):
            continue  # shape is the schema's job
        summed: dict[str, int] | None = _summed_attempt_usage(entries)
        if summed is not None and summed != usage:
            issues.append(
                ValidationIssue(label, "attempt_log usage must sum to the record's usage")
            )


def _summed_attempt_usage(entries: list[Any]) -> dict[str, int] | None:
    """Key-wise total of the entries' usage, or ``None`` when a count is not one.

    ``None`` rather than a partial sum: a malformed count is the schema's finding, already
    reported, and a total computed around it would be a second and *wrong* finding on the same
    line -- a validator inventing a disagreement out of corruption it did not cause.
    """
    summed: dict[str, int] = {}
    for entry in entries:
        counts = entry.get("usage")
        if not isinstance(counts, dict):
            return None
        for key, value in counts.items():
            if not isinstance(value, int) or isinstance(value, bool):
                return None
            summed[key] = summed.get(key, 0) + value
    return summed


def _attempt_timeline_ms(entries: list[Any]) -> int | None:
    """Total time the entries account for: every dispatch plus every wait they recorded.

    ``None`` rather than a partial total when a duration is not a count, for the reason
    ``_summed_attempt_usage`` gives -- a figure computed around corruption the schema already
    reported would be a second and wrong finding on the same line.

    An absent ``backoff_ms`` contributes zero and is not corruption: it is a line written before
    the field, whose unrecorded waits can only make the true total larger than this one. That
    direction is the safe one for a check that reports totals which are *too large*, so a legacy
    line is never accused on the strength of what it could not say.
    """
    total = 0
    for entry in entries:
        elapsed = entry.get("elapsed_ms")
        if not isinstance(elapsed, int) or isinstance(elapsed, bool):
            return None
        backoff = entry.get("backoff_ms", 0)
        if backoff is None:
            backoff = 0
        if not isinstance(backoff, int) or isinstance(backoff, bool):
            return None
        total += elapsed + backoff
    return total


def _read_json_artifact(path: Path) -> tuple[Any, ValidationIssue | None]:
    """Decode and parse one JSON artifact, returning the problem instead of raising it.

    The single loader for every JSON artifact read in this module — schema validation *and* the
    relationship/hash checks that re-read the same files afterwards. Hardening one reader and
    leaving its siblings is precisely how this file kept crashing ``monoid validate`` on the
    corruption it exists to report: the schema pass recorded the issue and returned, and a
    downstream check then re-read the same bytes with a bare ``read_text()``.

    Decoding is explicit rather than left to ``read_text`` because a torn multi-byte sequence
    raises out of the *read*, not out of ``json.loads`` — and ``RecursionError`` is not a
    ``ValueError``, so a deeply nested document escapes a ``ValueError``-only handler.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, ValidationIssue(path.name, f"unreadable: {exc}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, ValidationIssue(path.name, "invalid UTF-8")
    try:
        return loads_json_ingress(text), None
    except json.JSONDecodeError as exc:
        return None, ValidationIssue(path.name, f"invalid JSON: {exc.msg}")
    except (ValueError, RecursionError):
        # Same catch-set and label as both JSONL halves: a deeply nested document exceeds the C
        # scanner's stack, and ``json.loads`` raises other ValueErrors (the digit-conversion cap)
        # that are decoder limits too.
        return None, ValidationIssue(path.name, "invalid JSON: decoder limit exceeded")


def _validate_json_file(path: Path, schema: dict[str, Any], issues: list[ValidationIssue]) -> None:
    if not path.exists():
        return
    payload, issue = _read_json_artifact(path)
    if issue is not None:
        issues.append(issue)
        return
    _validate_object(payload, schema, issues, path.name)


def _validate_object(
    payload: Any,
    schema: dict[str, Any],
    issues: list[ValidationIssue],
    label: str,
    *,
    redact_instance: bool = False,
) -> None:
    """Report one object's schema errors, optionally without quoting the object back.

    ``redact_instance`` is for the content-classified artifacts. jsonschema builds its message out
    of the *instance* -- a top-level ``oneOf`` failure prints the whole record -- and
    ``monoid validate``'s issues go to a terminal and into ``--json`` output, so an unmatched line
    of ``model_payloads.jsonl`` or ``model-content.jsonl`` would republish a conversation, or a
    whole system prompt, out of the private run directory. The trigger is not an attack but the
    compatibility policy: both artifacts pin a literal ``schema_version`` enum of v1 spellings only, so
    the first version bump makes every line of every retained run directory fail at once. The
    failing keyword and the path locate the problem; the value is the payload.
    """

    validator = Draft202012Validator(schema)
    for error in sorted(validator.iter_errors(payload), key=lambda item: list(item.path)):
        suffix = ".".join(str(part) for part in error.path)
        issue_path = f"{label}.{suffix}" if suffix else label
        if not redact_instance:
            issues.append(ValidationIssue(issue_path, error.message))
            continue
        # Both redacted schemas are a bare top-level ``oneOf``, so the outer error is always
        # ``oneOf`` at the root: reporting the keyword alone would give a maintainer one identical
        # line per record and no way to tell a version bump from a truncated write. Descend to the
        # branch the record *claims* to be -- its ``kind`` is a schema literal, so naming it costs
        # nothing -- and report that branch's keyword and path. Both come from the schema and the
        # instance's key structure, never from a value. ``best_match`` alone is not enough here:
        # its relevance heuristic picks whichever branch failed shallowest, which for a bumped
        # ``schema_version`` is a *different* kind's missing-required error.
        detail = best_match(_discriminated_errors(payload, schema) or error.context or [error])
        detail = detail if detail is not None else error
        location = detail.json_path
        where = "" if location in ("$", "") else f" at {location}"
        issues.append(
            ValidationIssue(
                issue_path, f"does not satisfy the {detail.validator} constraint{where}"
            )
        )


def _discriminated_errors(payload: Any, schema: dict[str, Any]) -> list[Any]:
    """The errors of the ``oneOf`` branch ``payload``'s ``kind`` names, if it names one.

    The two content artifacts discriminate their branches with ``{"kind": {"const": ...}}``, which
    is exactly the information a redacted report may use: it is a schema literal, and it is the
    difference between "this record is broken somehow" and "this record's ``schema_version`` is
    from a version you do not read".
    """

    if not isinstance(payload, dict):
        return []
    kind = payload.get("kind")
    for branch in schema.get("oneOf", ()):
        if branch.get("properties", {}).get("kind", {}).get("const") == kind:
            return list(Draft202012Validator(branch).iter_errors(payload))
    return []


def _validate_model_payload_digests(run_dir: Path, issues: list[ValidationIssue]) -> None:
    """Re-verify the corpus's self-verification: chunks hash to their names, request records
    reassemble to the bytes their key was taken over, response references resolve.

    Reads the file leniently -- lines the schema pass already reported are skipped here rather
    than reported twice -- and treats every reassembly failure as an issue on the record that
    cannot honor its digest, naming the line. Unreferenced files in the chunk directory are NOT
    issues: a crashed write may orphan one, and reclaiming it is ``monoid gc``'s job, not this
    pass's -- integrity is only what a record references. The collector keeps the converse
    promise (it deletes nothing this pass resolves), bound by a spy test over this function's
    reader rather than by sharing its code.
    """

    path = run_dir / MODEL_PAYLOADS_FILENAME
    chunk_dir = run_dir / MODEL_PAYLOADS_DIRNAME
    records: list[tuple[int, dict[str, Any]]] = []
    chunks: dict[str, bytes] = {}
    try:
        lines = path.read_bytes().split(b"\n")
    except OSError:
        return
    for index, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            payload = loads_json_ingress(raw_line.decode("utf-8"))
        except Exception:
            continue  # the schema pass already reported this line
        if not isinstance(payload, dict):
            continue
        records.append((index, payload))
        if payload.get("kind") == PAYLOAD_CHUNK_KIND:
            text = payload.get("text")
            sha = payload.get("sha256")
            if not isinstance(text, str) or not isinstance(sha, str):
                continue
            data = text.encode("utf-8")
            if sha256_bytes(data) != sha:
                issues.append(
                    ValidationIssue(f"{path.name}:{index}", "chunk text does not match its sha256")
                )
                continue
            chunks[sha] = data

    # One slot, deliberately. Caching every resolved chunk beside the inline ones turned this
    # command's footprint from O(largest chunk) into O(total offloaded corpus) -- measured at
    # 42.1 MB against 3.2 MB over forty 1 MB chunks, and with an 8 MB ceiling per chunk and no
    # bound on the count, a large run directory costs gigabytes on the one command an operator
    # runs before trusting it. Records that name one chunk are adjacent, so a single slot keeps
    # the re-read the memo was added to stop, without keeping the corpus.
    last_resolved: tuple[str, bytes] | None = None

    def resolve(sha: str) -> bytes:
        nonlocal last_resolved
        if sha in chunks:
            return chunks[sha]
        if last_resolved is not None and last_resolved[0] == sha:
            return last_resolved[1]
        # A reference becomes a filename here, so this is where the writer's constraint has to be
        # re-established: everything it writes is 64 hex, and an absolute or ``..``-relative string
        # joined onto ``chunk_dir`` discards the base and names any file on the machine. The hash
        # check below cannot stand in for it -- it happens after the read.
        if not is_chunk_sha256(sha):
            raise ValueError("chunk reference is not a content-addressed name")
        data = read_verified_bytes(chunk_dir / sha, max_bytes=MAX_MODEL_PAYLOAD_BYTES)
        if data is None:
            raise ValueError(f"offloaded chunk {sha} is not a readable run-directory file")
        if sha256_bytes(data) != sha:
            raise ValueError(f"offloaded chunk {sha} does not match its name")
        # Held for the next caller only, because N records may name ONE chunk and every one of
        # them used to re-read it. A content-addressed name means the bytes cannot have changed
        # between two reads of the same run directory.
        last_resolved = (sha, data)
        return data

    parsed_bodies: dict[str, str | None] = {}
    for index, payload in records:
        kind = payload.get("kind")
        if kind == MODEL_REQUEST_KIND:
            digest = payload.get("request_digest")
            refs = payload.get("refs")
            if not isinstance(digest, str) or not isinstance(refs, bool):
                continue  # shape issues are the schema pass's report
            try:
                rebuilt = reassemble_request_preimage(payload.get("payload"), resolve, refs=refs)
            except Exception:
                issues.append(
                    ValidationIssue(f"{path.name}:{index}", "request payload cannot be reassembled")
                )
                continue
            if sha256_bytes(rebuilt) != digest:
                issues.append(
                    ValidationIssue(
                        f"{path.name}:{index}",
                        "request payload does not reassemble to its request_digest",
                    )
                )
                continue
            # Resolving is not believing -- on this half too. What stood here instead was a
            # comment declining this arm, on the reasoning that reassembly is a canonical
            # encode and a digest-valid preimage is therefore canonical JSON by construction.
            # That enumerated three of the encoder's refusals (non-finite values, over-long
            # ints, surrogates) and omitted nesting, which the encoder does not bound and the
            # reader does. The writer now refuses such a preimage, but corpora written before
            # that gate still hold these records, and this is the command an operator runs on a
            # run directory that arrived from somewhere else. A record that reassembles and
            # hashes correctly and *still* cannot be read testifies to nothing, and saying so
            # here is the difference between an unreadable corpus and a silent one.
            try:
                loads_json_ingress(rebuilt.decode("utf-8"))
            except Exception as error:
                issues.append(
                    ValidationIssue(
                        f"{path.name}:{index}",
                        # Bounded: these messages come from the parser's own fixed vocabulary
                        # and carry positions rather than content, but the corpus is untrusted
                        # input and a validator report is not the place to find that out.
                        f"request payload is not readable by the replay reader: {str(error)[:200]}",
                    )
                )
        elif kind == MODEL_RESPONSE_KIND:
            # Through the shared trichotomy, not an inline shape test: the replay reader
            # refuses through the same function, so the two consumers cannot disagree about
            # which objects are references. The ``malformed`` arm is new strictness this
            # gained from the share -- a single-key marker object carrying a non-sha value
            # used to be skipped as data here while being unmistakably writer-shaped.
            shape, sha = response_reference(payload.get("response"))
            if shape == RESPONSE_MALFORMED:
                issues.append(
                    ValidationIssue(
                        f"{path.name}:{index}",
                        "response reference is not a content-addressed name",
                    )
                )
            elif shape == RESPONSE_REFERENCE:
                if sha in parsed_bodies:
                    # The verdict is a property of the bytes and the bytes are named by their
                    # hash, so a second record naming this chunk needs neither the read nor the
                    # re-hash that produced it. Skipping the resolve entirely -- rather than
                    # resolving and then not parsing -- is what keeps the bytes transient.
                    problem = parsed_bodies[sha]
                    if problem is not None:
                        issues.append(ValidationIssue(f"{path.name}:{index}", problem))
                    continue
                try:
                    resolved = resolve(sha)
                except Exception:
                    issues.append(
                        ValidationIssue(
                            f"{path.name}:{index}",
                            "response reference does not resolve to a recorded chunk",
                        )
                    )
                    continue
                # Resolving is not believing. Re-hashing proves the bytes are the ones the
                # writer named, not that they are a body any reader will accept: the sha names
                # whatever was planted, so an offloaded body could carry JSON the ingress rules
                # forbid and still pass every check here. The replay reader refuses such a body;
                # without this arm ``monoid validate`` would certify the corpus clean and the
                # operator would meet the refusal at run time with a green integrity report.
                #
                # Memoized by sha, because the answer is a property of the bytes and the bytes
                # are named by their hash. Parsing per RECORD instead of per CHUNK made this the
                # dominant cost of the command: 4,000 records naming one 8 MB chunk took ~62
                # minutes, where one parse takes about a second.
                if sha not in parsed_bodies:
                    try:
                        loads_json_ingress(resolved.decode("utf-8"))
                        parsed_bodies[sha] = None
                    except Exception:
                        parsed_bodies[sha] = (
                            "response body is not JSON this kernel's readers accept"
                        )
                problem = parsed_bodies[sha]
                if problem is not None:
                    issues.append(ValidationIssue(f"{path.name}:{index}", problem))


def _validate_jsonl_file(
    path: Path,
    schema: dict[str, Any],
    issues: list[ValidationIssue],
    *,
    redact_instance: bool = False,
) -> None:
    # Decode per line and REPORT undecodable bytes, matching the twin ``_validate_event_file``.
    #
    # Strict whole-file decoding raised ``UnicodeDecodeError`` out of ``monoid validate`` — the
    # ``try`` below covers only ``json.loads`` — so a torn transcript crashed the validator. But
    # ``errors="replace"`` is the wrong repair: a *complete* record whose string value holds an
    # undecodable byte then parses, validates, and the file is reported clean. A validator that
    # turns detected corruption into silence is worse than one that crashes, because the crash at
    # least stops the caller. The twin detects and reports; do the same.
    for index, raw_line in enumerate(path.read_bytes().split(b"\n"), start=1):
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            issues.append(ValidationIssue(f"{path.name}:{index}", "invalid UTF-8"))
            continue
        if not line.strip():
            continue
        try:
            payload = loads_json_ingress(line)
        except json.JSONDecodeError as exc:
            issues.append(ValidationIssue(f"{path.name}:{index}", f"invalid JSON: {exc.msg}"))
            continue
        except (ValueError, RecursionError) as exc:
            # Wider than JSONDecodeError: a deeply nested line exceeds the C scanner's stack, and
            # `json.loads` raises other ValueErrors too. A validator whose job is to report
            # corruption must not be stopped by it. Same catch-set and same message shape as
            # ``_validate_event_file`` — identical corruption should not be labelled two ways.
            message = exc.msg if isinstance(exc, json.JSONDecodeError) else "decoder limit exceeded"
            issues.append(ValidationIssue(f"{path.name}:{index}", f"invalid JSON: {message}"))
            continue
        _validate_object(
            payload, schema, issues, f"{path.name}:{index}", redact_instance=redact_instance
        )


def _validate_event_file(path: Path, issues: list[ValidationIssue]) -> None:
    for index, record in enumerate(iter_committed_jsonl_records(path), start=1):
        if not record.raw_bytes.strip():
            continue
        try:
            event = loads_json_ingress(record.raw_bytes.decode("utf-8"))
        except UnicodeDecodeError:
            issues.append(ValidationIssue(f"{path.name}:{index}", "invalid UTF-8"))
            continue
        except (ValueError, RecursionError) as exc:
            # ``RecursionError`` is NOT a ``ValueError``: a deeply nested line exceeds the C
            # scanner's stack and escaped this clause entirely, crashing ``monoid validate`` on the
            # very corruption it exists to report. ``events.jsonl`` is validated before
            # ``transcript.jsonl``, so hardening only the transcript half left this reachable first.
            message = exc.msg if isinstance(exc, json.JSONDecodeError) else "decoder limit exceeded"
            issues.append(ValidationIssue(f"{path.name}:{index}", f"invalid JSON: {message}"))
            continue
        _validate_object(event, EVENT_SCHEMA, issues, f"{path.name}:{index}")
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        schema = EVENT_DATA_SCHEMAS.get(event_type) if isinstance(event_type, str) else None
        if schema is None:
            issues.append(
                ValidationIssue(
                    f"{path.name}:{index}", f"no data schema for event type: {event_type!r}"
                )
            )
            continue
        data = event.get("data")
        _validate_object(
            data if isinstance(data, dict) else {}, schema, issues, f"{path.name}:{index}.data"
        )


def _validate_manifest_workspace_index(run_dir: Path, issues: list[ValidationIssue]) -> None:
    _validate_manifest_relative_file(
        run_dir, issues, "workspace_index_path", "workspace index file missing"
    )


def _validate_manifest_workspace_base(run_dir: Path, issues: list[ValidationIssue]) -> None:
    _validate_manifest_relative_file(
        run_dir, issues, "workspace_base_path", "workspace base file missing"
    )


def _validate_manifest_relative_file(
    run_dir: Path,
    issues: list[ValidationIssue],
    key: str,
    missing_message: str,
) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest, issue = _read_json_artifact(manifest_path)
    if issue is not None:
        return  # already recorded by the schema pass; do not double-report
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
    proposal, issue = _read_json_artifact(proposal_path)
    if issue is not None:
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
                issues.append(
                    ValidationIssue(
                        f"proposal.json.files.{index}.snapshot_path", "snapshot missing"
                    )
                )
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if file_info.get("snapshot_sha256") != actual:
                issues.append(
                    ValidationIssue(
                        f"proposal.json.files.{index}.snapshot_sha256", "snapshot hash mismatch"
                    )
                )


def _validate_package_hashes(run_dir: Path, issues: list[ValidationIssue]) -> None:
    package_path = run_dir / "proposal.package.json"
    package, issue = _read_json_artifact(package_path)
    if issue is not None:
        return
    if not isinstance(package, dict):
        return
    if package.get("package_hash") != canonical_sha256(package, drop=("package_hash",)):
        issues.append(
            ValidationIssue("proposal.package.json.package_hash", "package hash mismatch")
        )
    seen: set[str] = set()
    for index, file_info in enumerate(package.get("files") or []):
        if not isinstance(file_info, dict):
            continue
        rel = file_info.get("path")
        if not isinstance(rel, str):
            continue
        if rel in seen:
            issues.append(
                ValidationIssue(
                    f"proposal.package.json.files.{index}.path", "duplicate package path"
                )
            )
        seen.add(rel)
        path = run_dir / rel
        if not path.exists() or not path.is_file():
            issues.append(
                ValidationIssue(f"proposal.package.json.files.{index}.path", "package file missing")
            )
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if file_info.get("sha256") != actual:
            issues.append(
                ValidationIssue(
                    f"proposal.package.json.files.{index}.sha256", "package file hash mismatch"
                )
            )


def _validate_canonical_hash(path: Path, hash_key: str, issues: list[ValidationIssue]) -> None:
    payload, issue = _read_json_artifact(path)
    if issue is not None:
        return
    if not isinstance(payload, dict):
        return
    expected = payload.get(hash_key)
    actual = canonical_sha256(payload, drop=(hash_key,))
    if expected != actual:
        issues.append(ValidationIssue(f"{path.name}.{hash_key}", f"{hash_key} mismatch"))
