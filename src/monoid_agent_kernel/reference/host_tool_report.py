"""Offline typed report, approval and fresh-process recovery recipe.

Run against the installed package: ``python -m monoid_agent_kernel.reference.host_tool_report``.
Use ``--phase prepare --root PATH``, then ``--phase resume --root PATH --run-id ID --task-id ID``
to carry the printed host identities between interpreters. Approval permits reading the proposal;
applying the proposal to the user workspace is a separate host operation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from monoid_agent_kernel import AgentLoop, AgentRunSpec, LoopSession, ToolContext, tool
from monoid_agent_kernel.core.agents import AgentRuntimeConfig, SubagentDefinition, ToolBinding
from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore
from monoid_agent_kernel.core.tool_surface import ToolScope
from monoid_agent_kernel.providers.fake import FakeModelAdapter, ModelTurn, fake_tool_call
from monoid_agent_kernel.tool_loader import FunctionToolProvider
from monoid_agent_kernel.tools.host_bindings import (
    HostToolActivation,
    HostToolBinding,
    HostToolKey,
    HostToolResource,
    ToolWorkspace,
    require_host_tool_binding,
)

REPORTS = HostToolKey[ToolWorkspace](
    "reports.workspace", accepts=lambda view: isinstance(view, ToolWorkspace)
)


class ReportDraft(BaseModel):
    name: str
    title: str
    body: str


class SavedReport(BaseModel):
    path: str
    sha256: str


@tool(id="report.save", side_effect="write")
def save_report(ctx: ToolContext, draft: ReportDraft) -> SavedReport:
    workspace = require_host_tool_binding(ctx, REPORTS)
    path = f"reports/{draft.name}.md"
    digest = workspace.write_bytes(
        path, f"# {draft.title}\n\n{draft.body}\n".encode(), create_dirs=True
    )
    return SavedReport(path=path, sha256=digest)


@tool(id="report.read", side_effect="read")
def read_report(ctx: ToolContext, path: str) -> dict[str, str]:
    content, digest = require_host_tool_binding(ctx, REPORTS).read_bytes(path)
    return {"body": content.decode(), "sha256": digest}


def _report_loop(
    spec: AgentRunSpec, adapter: FakeModelAdapter, opened: list[str], reads: list[str],
) -> AgentLoop:
    def open_reports(activation: HostToolActivation) -> HostToolResource[ToolWorkspace]:
        opened.append(activation.run_id)
        return activation.workspace_resource()

    def counted_read(ctx: ToolContext, args: dict[str, Any]) -> Any:
        reads.append(args["path"])
        return read_report.handler(ctx, args)

    scope = ToolScope(allowed_paths=("reports/**",))
    config = AgentRuntimeConfig(
        definition_id="typed-reports",
        tools=(
            ToolBinding.for_tool("report.save", scope=scope),
            ToolBinding.for_tool("report.read", scope=scope, authorization="ask"),
        ),
    )
    provider = FunctionToolProvider(
        lambda _ctx: (save_report, replace(read_report, handler=counted_read))
    )
    return AgentLoop.from_config(
        spec, adapter, config, tool_providers=(provider,),
        host_tool_bindings=(HostToolBinding(REPORTS, open=open_reports),),
    )


def prepare_report(root: Path) -> dict[str, Any]:
    """Commit one approval park and release its process-local activation."""
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    spec = AgentRunSpec(workspace_root=workspace, run_root=root / "runs", mode="propose")
    opened: list[str] = []
    reads: list[str] = []
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                tool_calls=(
                    fake_tool_call(
                        "report_save",
                        {
                            "draft": {
                                "name": "weekly",
                                "title": "Weekly",
                                "body": "Ready for approval.",
                            }
                        },
                        "save",
                    ),
                ),
            ),
            ModelTurn(
                usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                tool_calls=(
                    fake_tool_call("report_read", {"path": "reports/weekly.md"}, "read"),
                ),
            ),
            ModelTurn(
                final_text="Waiting for report approval.",
                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            ),
        ]
    )
    source = _report_loop(spec, adapter, opened, reads)
    try:
        source.open()
        parked = source.run_until_suspended(
            "Draft a report and ask for approval before reading it."
        )
        assert parked.reason == "awaiting_tasks" and len(parked.awaiting_task_ids) == 1
        checked = LocalFsCheckpointStore(spec.run_root).latest_checked(spec.run_id)
        if not checked.ok or checked.value is None:
            raise RuntimeError(f"report checkpoint could not be loaded: {checked.status}")
        record = checked.value
        assert len(opened) == 1 and not reads and len(adapter.requests) == 3
        assert not (workspace / "reports/weekly.md").exists()
        source.release_parked()
    except BaseException:
        try:
            source.discard_uncommitted()
        except BaseException:
            pass
        raise
    return {
        "status": parked.reason, "run_id": spec.run_id, "task_id": parked.awaiting_task_ids[0],
        "pid": os.getpid(), "resource_opens": len(opened), "read_calls": len(reads),
        "model_calls": len(adapter.requests), "total_usage": dict(record.checkpoint.total_usage),
        "proposal_only": not (workspace / "reports/weekly.md").exists(),
    }


def resume_report(root: Path, run_id: str, task_id: str) -> dict[str, Any]:
    """Restore the host's original run/task and consume one approval."""
    spec = AgentRunSpec(
        run_id=run_id, workspace_root=root / "workspace", run_root=root / "runs", mode="propose",
    )
    checked = LocalFsCheckpointStore(spec.run_root).latest_checked(run_id)
    if not checked.ok or checked.value is None:
        raise RuntimeError(f"report checkpoint could not be loaded: {checked.status}")
    record = checked.value
    adapter = FakeModelAdapter(turns=[ModelTurn(
        final_text="Approved report restored and read.",
        usage={"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
    )])
    opened: list[str] = []
    reads: list[str] = []
    restored = _report_loop(spec, adapter, opened, reads)
    try:
        restored.restore(record.checkpoint, blobs=record.blob)
        inspection = LoopSession(restored).inspect()
        if (
            inspection.last_suspension_reason != "awaiting_tasks"
            or task_id not in inspection.awaiting_task_ids
        ):
            raise ValueError("task is not awaiting approval for this run")
        first = restored.report_task_result(task_id, {"approved": True})
        duplicate = restored.report_task_result(task_id, {"approved": False})
        assert first["delivered"] and not first["duplicate"]
        assert not duplicate["delivered"] and duplicate["duplicate"]
        settled = restored.run_until_suspended(None)
        assert settled.reason == "settled"
        result = restored.close()
    except BaseException:
        try:
            restored.discard_uncommitted()
        except BaseException:
            pass
        raise
    observations = [obs for request in adapter.requests for obs in request.observations]
    read = next(obs for obs in observations if obs.call_id == "tool_approval_replay:read")
    outcome = read.output["result"]
    assert outcome["ok"] and outcome["result"]["body"] == "# Weekly\n\nReady for approval.\n"
    assert len(opened) == 1 and len(reads) == 1 and len(adapter.requests) == 1
    return {
        "status": result.status, "run_id": run_id, "task_id": task_id, "pid": os.getpid(),
        "resource_opens": len(opened), "read_calls": len(reads), "model_calls": len(adapter.requests),
        "report_body": outcome["result"]["body"], "duplicate_ignored": duplicate["duplicate"],
        "total_usage": {key: result.metrics[key] for key in (
            "input_tokens", "output_tokens", "total_tokens",
        )},
        "proposal_only": not (spec.workspace_root / "reports/weekly.md").exists(),
    }


def run_report(root: Path) -> dict[str, Any]:
    prepared = prepare_report(root)
    resumed = resume_report(root, prepared["run_id"], prepared["task_id"])
    return {
        "status": resumed["status"],
        "activations": prepared["resource_opens"] + resumed["resource_opens"],
        "approved_read": resumed["read_calls"] == 1,
        "proposal_only": prepared["proposal_only"] and resumed["proposal_only"],
    }


def run_child_reports(root: Path, *, context: str = "fresh") -> dict[str, Any]:
    """Delegate a report, then use the parent's independent resource after child cleanup."""
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    opened: list[HostToolActivation] = []
    adapters: list[HostToolResource[ToolWorkspace]] = []

    def open_reports(activation: HostToolActivation) -> HostToolResource[ToolWorkspace]:
        opened.append(activation)
        adapter = activation.workspace_resource()
        adapters.append(adapter)
        return adapter

    def save(name: str) -> ModelTurn:
        return ModelTurn(
            tool_calls=(
                fake_tool_call(
                    "report_save",
                    {"draft": {"name": name, "title": name, "body": f"{name} report"}},
                    f"save-{name}",
                ),
            )
        )

    def read(name: str) -> ModelTurn:
        return ModelTurn(
            tool_calls=(
                fake_tool_call(
                    "report_read",
                    {"path": f"reports/{name}.md"},
                    f"read-{name}",
                ),
            )
        )

    model = FakeModelAdapter(
        turns=[
            ModelTurn(
                tool_calls=(
                    fake_tool_call(
                        "agent_spawn",
                        {"subagent_type": "reporter", "prompt": "Write the child report."},
                        "spawn",
                    ),
                )
            ),
            save("child"),
            read("child"),
            ModelTurn(final_text="Child report ready."),
            save("parent"),
            read("parent"),
            ModelTurn(final_text="Both reports ready."),
        ]
    )
    scope = ToolScope(allowed_paths=("reports/**",))
    loop = AgentLoop.from_config(
        AgentRunSpec(workspace_root=workspace, run_root=root / "runs", mode="propose"),
        model,
        AgentRuntimeConfig(
            definition_id="report-delegation",
            tools=(
                ToolBinding.for_tool("agent.spawn"),
                ToolBinding.for_tool("report.save", scope=scope),
                ToolBinding.for_tool("report.read", scope=scope),
            ),
        ),
        tool_providers=(FunctionToolProvider(lambda _ctx: (save_report, read_report)),),
        subagent_definitions={
            "reporter": SubagentDefinition(
                context=context,
                tools=("report.save", "report.read"),
            )
        },
        host_tool_bindings=(
            HostToolBinding(
                REPORTS,
                open=open_reports,
                inherit_to_children=True,
            ),
        ),
    )
    result = loop.run_once("Delegate one report, then write the parent report.")
    observations = {
        obs.call_id: obs.output for request in model.requests for obs in request.observations
    }
    for name in ("child", "parent"):
        read_result = observations[f"read-{name}"]
        assert read_result["ok"] and f"{name} report" in read_result["result"]["body"]
    assert len(opened) == 2 and adapters[0] is not adapters[1]
    assert opened[1].parent_run_id == opened[0].run_id == opened[1].root_run_id
    assert not (workspace / "reports").exists()
    return {"status": result.status, "context": context, "activations": 2, "proposal_only": True}


def main(argv: Sequence[str] = ()) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "resume"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--task-id")
    args = parser.parse_args(argv)
    if args.phase is None:
        if args.root is not None or args.run_id is not None or args.task_id is not None:
            parser.error("--root, --run-id and --task-id require --phase")
        with tempfile.TemporaryDirectory(prefix="monoid-host-report-") as directory:
            root = Path(directory)
            output = {
                "approval": run_report(root / "approval"),
                "delegation": [
                    run_child_reports(root / mode, context=mode) for mode in ("fresh", "fork")
                ],
            }
    else:
        if args.root is None:
            parser.error("--root is required with --phase")
        if args.phase == "prepare":
            if args.run_id is not None or args.task_id is not None:
                parser.error("--run-id and --task-id belong to --phase resume")
            output = prepare_report(args.root)
        else:
            if not args.run_id or not args.task_id:
                parser.error("--run-id and --task-id are required with --phase resume")
            output = resume_report(args.root, args.run_id, args.task_id)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
