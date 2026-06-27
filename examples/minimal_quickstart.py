"""Smallest possible run: one agent, one workspace, NO servers.

This is the "first turn" example — it uses ``FakeModelAdapter`` (a scripted model) so it
runs offline with no LLM gateway, no API key, and no HTTP stack. Swap the adapter for
``GatewayModelAdapter(...)`` (or your own ``ModelAdapter``) and the rest is unchanged.

Run it::

    python examples/minimal_quickstart.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make the example runnable from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from native_agent_runner import (  # noqa: E402
    AgentLoop,
    AgentRunSpec,
    AgentRuntimeConfig,
    FakeModelAdapter,
    RegistryToolRef,
    ToolBinding,
    tool_ids,
)
from native_agent_runner.providers.base import ModelTurn  # noqa: E402
from native_agent_runner.providers.fake import fake_tool_call  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "workspace"
        workspace.mkdir()
        (workspace / "notes.md").write_text("alpha beta gamma\n", encoding="utf-8")

        # 1) Session descriptor: where the run executes and under what limits.
        #    mode="apply" writes straight to the workspace. The default "propose" instead
        #    stages changes as a diff.patch + proposal for review (nothing is mutated).
        spec = AgentRunSpec(workspace_root=workspace, run_root=Path(tmp) / "runs", mode="apply")

        # 2) Runtime config: which tools the agent may use this run. (Bindings map a
        #    registry tool id to an agent-facing tool.) Use the tool_ids constants for
        #    autocomplete + typo-safety instead of bare strings.
        config = AgentRuntimeConfig(
            definition_id="quickstart",
            tools=(
                ToolBinding(binding_id="fs.write", ref=RegistryToolRef(tool_ids.FS_WRITE)),
                ToolBinding(binding_id="run.finish", ref=RegistryToolRef(tool_ids.RUN_FINISH)),
            ),
        )

        # 3) A scripted model: turn 1 writes a file via fs.write, turn 2 settles with text.
        adapter = FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="t1",
                    tool_calls=(
                        fake_tool_call(
                            "fs_write",
                            {"path": "SUMMARY.md", "content": "# Summary\n3 words.\n"},
                            "c1",
                        ),
                    ),
                ),
                ModelTurn(response_id="t2", final_text="Wrote SUMMARY.md."),
            ]
        )

        # 4) from_config wraps the bare config in a provider — one call instead of six.
        result = AgentLoop.from_config(spec, adapter, config).run_once("Summarize notes.md")

        print("status     :", result.status)
        print("final_text :", result.final_text)
        print("SUMMARY.md :", (workspace / "SUMMARY.md").exists())


if __name__ == "__main__":
    main()
