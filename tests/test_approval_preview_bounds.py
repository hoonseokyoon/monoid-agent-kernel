"""The approval preview is bounded *and* still masks — asserted on the routes that publish it.

Every pre-existing assertion on this surface reads `build_tool_approval_task_request`'s return value
directly. That is the one place a regression cannot be seen: the value is published through
`HostedTask.public_request()` onto `task.started`, into `artifacts/tasks/<id>/task.json`, and back to
the *model* through `job.list`/`job.status`. A change that swapped the projection for a different one
would keep every build-output assertion green while the published copy lost its masking.

So these read the public payload, and they assert the two halves together. "Bounded" and "still
masks" are separate properties, and the obvious way to get the first — route the preview through
`public_view.args_preview`, which has all the caps — silently drops the second: `args_preview` is the
generic branch of a four-way dispatch on `spec.preview_kind`, the request never records which kind it
came from, and the generic branch knows nothing about secret names. Measured before this commit:
routing that way published `api_key` as `sk-live-123` and republished the whole `run.finish` summary
that the previous stage had just removed from the settle events.
"""

from __future__ import annotations

import json

import pytest
from support.runtime import runtime_config, runtime_provider, tool_binding

from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore
from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.core.tool_approval import (
    MAX_ARGUMENT_DEPTH,
    TOOL_APPROVAL_TASK_KIND,
    build_tool_approval_task_request,
)
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.public_view import PREVIEW_BYTE_BUDGET
from monoid_agent_kernel.tasks import HostedTask
from monoid_agent_kernel.tools.base import ToolContext, ToolResult, ToolSpec

FILE_BODY = "SENTINEL-file-body " * 40  # 760 bytes of "file content" an approver must not publish
SUMMARY = "SENTINEL-the-final-answer"


def _spec(tool_id: str = "fs.write", preview_kind: str = "") -> ToolSpec:
    def handler(_ctx, _args):
        return ToolResult(ok=True)

    kwargs = {"preview_kind": preview_kind} if preview_kind else {}
    return ToolSpec(
        id=tool_id,
        description="demo",
        input_schema={"type": "object"},
        capability="",
        side_effect="write",
        handler=handler,
        **kwargs,
    )


def _published(tmp_path, arguments, *, spec=None, policy=None) -> dict:
    """The `arguments_preview` as a public surface actually receives it."""
    request = build_tool_approval_task_request(
        spec=spec or _spec(),
        binding_id="b",
        model_name="m",
        call_name="c",
        call_id="call_1",
        arguments=arguments,
        reason="sensitive write",
        turn_id="turn_0001",
        tool_event_id="event_1",
        policy=policy,
    )
    task = HostedTask(
        job_id="task_1",
        kind=TOOL_APPROVAL_TASK_KIND,
        prompt="Approve tool call",
        status="running",
        started_at=1.0,
        resume_on_exit=True,
        job_path=tmp_path / "task.json",
        cancel_path=tmp_path / "cancel.requested",
        request=request,
    )
    return task.public_payload(tmp_path, policy or PermissionPolicy())["request"]["arguments_preview"]


def test_an_ask_gated_write_no_longer_publishes_the_file_body(tmp_path) -> None:
    """The leak this commit exists to close, on the surface that leaked it.

    `redact_tool_arguments` masked secret-*named* keys and nothing else, so binding `fs.write` to
    `authorization="ask"` put the entire file body on `task.started` — wider than the settle-event
    leak the previous stage closed.
    """
    preview = _published(tmp_path, {"path": "notes/a.md", "content": FILE_BODY})

    assert preview["content"] == {"redacted": True, "type": "str", "bytes": len(FILE_BODY.encode())}
    assert preview["path"] == "notes/a.md"  # metadata an approver needs is untouched


def test_secret_named_keys_are_still_masked_at_every_depth(tmp_path) -> None:
    """The half a "just use args_preview" simplification would silently drop."""
    preview = _published(
        tmp_path,
        {
            "api_key": "sk-live-123",
            "env": {"AWS_SECRET_ACCESS_KEY": "abc123", "PATH": "/usr/bin"},
            "headers": [{"authorization": "Bearer xyz"}],
        },
    )

    assert preview["api_key"] == "[redacted]"
    assert preview["env"]["AWS_SECRET_ACCESS_KEY"] == "[redacted]"
    assert preview["env"]["PATH"] == "/usr/bin"
    assert preview["headers"][0]["authorization"] == "[redacted]"


def test_run_finish_prose_is_still_redacted_on_the_public_route(tmp_path) -> None:
    """`preview_kind="finish"` prose masking, asserted where it is published rather than where it
    is built. Nothing covered this route before, so dropping it would have shipped green."""
    preview = _published(
        tmp_path,
        {"summary": SUMMARY, "notes": "SENTINEL-notes", "outputs": ["n.md"]},
        spec=_spec("run.finish", preview_kind="finish"),
    )

    assert preview["summary"] == "[redacted]"
    assert preview["notes"] == "[redacted]"
    assert preview["outputs"] == ["n.md"]


def test_a_null_prose_value_is_left_alone_rather_than_badged_as_withheld(tmp_path) -> None:
    preview = _published(
        tmp_path,
        {"summary": SUMMARY, "notes": None},
        spec=_spec("run.finish", preview_kind="finish"),
    )

    assert preview["notes"] is None


def test_long_non_ascii_arguments_are_bounded_here_too(tmp_path) -> None:
    """The approval preview shares one truncator with the trace preview, so it inherits the
    byte-correct bound rather than carrying a second copy that can drift."""
    command = "echo " + "가" * 200

    preview = _published(tmp_path, {"command": command})

    assert preview["command"]["truncated"] is True
    assert len(preview["command"]["preview"].encode()) <= PREVIEW_BYTE_BUDGET
    assert command not in str(preview)


def test_the_builder_honours_the_policy_it_is_given(tmp_path) -> None:
    """`redact_patterns` is operator configuration; a preview built with `PermissionPolicy()`
    keeps every cap and silently drops exactly the redaction someone asked for."""
    policy = PermissionPolicy(redact_patterns=("secret/**",))

    preview = _published(tmp_path, {"path": "secret/report.txt"}, policy=policy)

    assert preview["path"] == {"redacted": True, "type": "str", "bytes": len("secret/report.txt")}


def test_the_run_s_own_policy_actually_reaches_the_preview(tmp_path) -> None:
    """Driven through a real loop, because the test above cannot see the wiring.

    It hands the builder a policy directly, so it stays green even if `loop.py` passes none at all —
    which a mutation run demonstrated. The parameter existing and the parameter being *supplied* are
    two properties, and only this one covers the second. Same shape as the gap that let the whole
    approval-preview leak survive: everything asserted where the value is built, nothing where it is
    published.
    """

    class Provider:
        def get_tools(self, context: ToolContext) -> list[ToolSpec]:
            del context

            def handler(_ctx: ToolContext, args: dict) -> ToolResult:
                return ToolResult(ok=True, content={"path": args["path"]})

            return [
                ToolSpec(
                    id="demo.gated",
                    description="ask-gated write",
                    input_schema={"type": "object", "additionalProperties": True},
                    capability="",
                    side_effect="write",
                    handler=handler,
                )
            ]

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    loop = AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace,
            run_root=tmp_path / "runs",
            permission_policy=PermissionPolicy(redact_patterns=("secret/**",)),
        ),
        model_adapter=FakeModelAdapter(
            turns=[
                ModelTurn(tool_calls=(fake_tool_call("demo_gated", {"path": "secret/report.txt"}, "c1"),)),
                ModelTurn(final_text="done"),
            ]
        ),
        runtime_config_provider=runtime_provider(
            runtime_config(bindings=(tool_binding("demo.gated", authorization="ask"),))
        ),
        tool_providers=(Provider(),),
    )
    loop.open()
    parked = loop.run_until_suspended("go")
    loop.report_task_result(parked.awaiting_task_ids[0], {"approved": False})
    loop.run_until_suspended(None)
    result = loop.close()

    started = [
        json.loads(line)
        for line in (result.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["type"] == "task.started"
    ]
    preview = started[-1]["data"]["request"]["arguments_preview"]
    assert preview["path"] == {"redacted": True, "type": "str", "bytes": len("secret/report.txt")}


def test_the_published_digest_resolves_to_the_raw_arguments(tmp_path) -> None:
    """What makes the bounded preview acceptable: the approver can still get the full text.

    The record carries `put_blob`'s own return value rather than a separately computed hash, so the
    digest on the record and the key the blob is stored under cannot disagree — the failure mode
    would be a handle that looks authoritative and resolves to nothing.
    """

    class Provider:
        def get_tools(self, context: ToolContext) -> list[ToolSpec]:
            del context

            def handler(_ctx: ToolContext, args: dict) -> ToolResult:
                return ToolResult(ok=True, content={})

            return [
                ToolSpec(
                    id="demo.gated",
                    description="ask-gated write",
                    input_schema={"type": "object", "additionalProperties": True},
                    capability="",
                    side_effect="write",
                    handler=handler,
                )
            ]

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = LocalFsCheckpointStore(tmp_path / "runs")
    arguments = {"path": "notes/a.md", "content": FILE_BODY}
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=FakeModelAdapter(
            turns=[
                ModelTurn(tool_calls=(fake_tool_call("demo_gated", arguments, "c1"),)),
                ModelTurn(final_text="done"),
            ]
        ),
        runtime_config_provider=runtime_provider(
            runtime_config(bindings=(tool_binding("demo.gated", authorization="ask"),))
        ),
        tool_providers=(Provider(),),
        checkpoint_store=store,
    )
    loop.open()
    parked = loop.run_until_suspended("go")

    task = json.loads(
        (loop.spec.run_root / loop.spec.run_id / "artifacts" / "tasks" / parked.awaiting_task_ids[0] / "task.json").read_text(encoding="utf-8")
    )
    digest = task["request"]["arguments_digest"]

    assert json.loads(store.get_blob(loop.spec.run_id, digest)) == arguments
    assert digest == canonical_sha256(arguments), "the handle must be the digest of what it addresses"
    # And the body is still absent from the preview the same record publishes.
    assert task["request"]["arguments_preview"]["content"]["redacted"] is True


def test_raw_arguments_stay_intact_for_replay(tmp_path) -> None:
    """Only the preview is bounded. `arguments` is what the approver replays and what
    `approval_key` is taken over, so a truncated copy would key a different call."""
    request = build_tool_approval_task_request(
        spec=_spec(),
        binding_id="b",
        model_name="m",
        call_name="c",
        call_id="call_1",
        arguments={"content": FILE_BODY, "api_key": "sk-live-123"},
        reason="r",
        turn_id="t",
        tool_event_id=None,
    )

    assert request["arguments"] == {"content": FILE_BODY, "api_key": "sk-live-123"}


def test_arguments_nested_past_the_limit_are_rejected_rather_than_crashing() -> None:
    """`RecursionError` is a `RuntimeError`, so it fell through the tool-call handler's
    `(NativeAgentError, ValueError, TypeError)` chain and out of dispatch entirely — one
    model-authored argument could end the run. A `ValueError` becomes a tool error it can correct."""
    deep: dict = {"leaf": "x"}
    for _ in range(600):
        deep = {"n": deep}

    with pytest.raises(ValueError, match="nest deeper"):
        build_tool_approval_task_request(
            spec=_spec(),
            binding_id="b",
            model_name="m",
            call_name="c",
            call_id="call_1",
            arguments=deep,
            reason="r",
            turn_id="t",
            tool_event_id=None,
        )


def test_ordinary_nesting_is_still_accepted() -> None:
    """The guard is a ceiling on hostile input, not a restriction on real tool calls."""
    nested: dict = {"leaf": "x"}
    for _ in range(MAX_ARGUMENT_DEPTH - 2):
        nested = {"n": nested}

    request = build_tool_approval_task_request(
        spec=_spec(),
        binding_id="b",
        model_name="m",
        call_name="c",
        call_id="call_1",
        arguments=nested,
        reason="r",
        turn_id="t",
        tool_event_id=None,
    )

    assert request["approval_key"]
