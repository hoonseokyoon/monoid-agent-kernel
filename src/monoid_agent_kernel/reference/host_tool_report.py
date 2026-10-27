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
from monoid_agent_kernel.core.agents import AgentRuntimeConfig, ToolBinding
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


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="monoid-host-report-") as directory:
        print(json.dumps(run_report(Path(directory)), sort_keys=True))


if __name__ == "__main__":
    main()
