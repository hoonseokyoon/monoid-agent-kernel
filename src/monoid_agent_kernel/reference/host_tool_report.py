"""Offline typed report, approval and fresh-activation recovery recipe.

Run against the installed package: ``python -m monoid_agent_kernel.reference.host_tool_report``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from monoid_agent_kernel import AgentLoop, AgentRunSpec, ToolContext, tool
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


def run_report(root: Path) -> dict[str, Any]:
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    spec = AgentRunSpec(workspace_root=workspace, run_root=root / "runs", mode="propose")
    opened: list[str] = []

    def open_reports(activation: HostToolActivation) -> HostToolResource[ToolWorkspace]:
        opened.append(activation.run_id)
        return activation.workspace_resource()

    bindings = (HostToolBinding(REPORTS, open=open_reports),)
    scope = ToolScope(allowed_paths=("reports/**",))
    config = AgentRuntimeConfig(
        definition_id="typed-reports",
        tools=(
            ToolBinding.for_tool("report.save", scope=scope),
            ToolBinding.for_tool("report.read", scope=scope, authorization="ask"),
        ),
    )
    provider = FunctionToolProvider(lambda _ctx: (save_report, read_report))
    source = AgentLoop.from_config(
        spec,
        FakeModelAdapter(
            turns=[
                ModelTurn(
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
                    )
                ),
                ModelTurn(
                    tool_calls=(
                        fake_tool_call("report_read", {"path": "reports/weekly.md"}, "read"),
                    )
                ),
            ]
        ),
        config,
        tool_providers=(provider,),
        host_tool_bindings=bindings,
    )
    source.open()
    try:
        parked = source.run_until_suspended(
            "Draft a report and ask for approval before reading it."
        )
        assert parked.reason == "awaiting_tasks" and len(parked.awaiting_task_ids) == 1
        record = LocalFsCheckpointStore(spec.run_root).latest(spec.run_id)
        assert record is not None
        assert not (workspace / "reports/weekly.md").exists()
        source.release_parked()
    finally:
        source.discard_uncommitted()

    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="Approved report restored and read.")])
    restored = AgentLoop.from_config(
        spec, adapter, config, tool_providers=(provider,), host_tool_bindings=bindings
    )
    restored.restore(record.checkpoint, blobs=record.blob)
    try:
        restored.report_task_result(parked.awaiting_task_ids[0], {"approved": True})
        settled = restored.run_until_suspended(None)
        assert settled.reason == "settled"
        result = restored.close()
    finally:
        restored.discard_uncommitted()
    observations = [obs for request in adapter.requests for obs in request.observations]
    read = next(obs for obs in observations if obs.call_id == "tool_approval_replay:read")
    outcome = read.output["result"]
    assert outcome["ok"] and "Ready for approval." in outcome["result"]["body"]
    assert len(opened) == 2
    return {
        "status": result.status,
        "activations": len(opened),
        "approved_read": True,
        "proposal_only": not (workspace / "reports/weekly.md").exists(),
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


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="monoid-host-report-") as directory:
        root = Path(directory)
        output = {
            "approval": run_report(root / "approval"),
            "delegation": [
                run_child_reports(root / mode, context=mode) for mode in ("fresh", "fork")
            ],
        }
        print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
