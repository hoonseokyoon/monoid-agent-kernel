"""Provider-backed memory tools with local filesystem storage.

This example runs offline with a scripted model. It attaches the same
``LocalFilesystemMemoryProvider`` as both a tool provider and a context provider, writes a memory
file in one run, then reads it from a second run.

Run it::

    python examples/memory_quickstart.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make the example runnable from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from monoid_agent_kernel import AgentLoop, AgentRunSpec, AgentRuntimeConfig  # noqa: E402
from monoid_agent_kernel.memory import LocalFilesystemMemoryProvider  # noqa: E402
from monoid_agent_kernel.providers.base import ModelTurn  # noqa: E402
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call  # noqa: E402


def _config(provider: LocalFilesystemMemoryProvider) -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        definition_id="memory-quickstart",
        tools=provider.tool_bindings(),
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        workspace = root / "workspace"
        run_root = root / "runs"
        workspace.mkdir()

        provider = LocalFilesystemMemoryProvider(
            root / "memory",
            write_authorization="allow",
        )
        config = _config(provider)

        writer = FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="write",
                    tool_calls=(
                        fake_tool_call(
                            "memory_create",
                            {
                                "path": "/memories/project/notes.md",
                                "file_text": "Project prefers concise release checklists.\n",
                            },
                            "memory-write",
                        ),
                    ),
                ),
                ModelTurn(response_id="write-done", final_text="Stored project memory."),
            ]
        )
        AgentLoop.from_config(
            AgentRunSpec(workspace_root=workspace, run_root=run_root / "writer"),
            writer,
            config,
            tool_providers=(provider,),
            context_providers=(provider,),
        ).run_once("Remember the project preference.")

        reader = FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="read",
                    tool_calls=(
                        fake_tool_call(
                            "memory_view",
                            {"path": "/memories/project/notes.md"},
                            "memory-read",
                        ),
                    ),
                ),
                ModelTurn(response_id="read-done", final_text="Read project memory."),
            ]
        )
        result = AgentLoop.from_config(
            AgentRunSpec(workspace_root=workspace, run_root=run_root / "reader"),
            reader,
            config,
            tool_providers=(provider,),
            context_providers=(provider,),
        ).run_once("Read the saved project preference.")

        saved = provider.store.view("/memories/project/notes.md")
        print("status     :", result.status)
        print("final_text :", result.final_text)
        print("memory     :", saved["content"].strip())


if __name__ == "__main__":
    main()
