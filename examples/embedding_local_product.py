"""Offline golden path for embedding Monoid directly in a local product.

Run from a checkout with::

    python examples/embedding_local_product.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from monoid_agent_kernel import (  # noqa: E402
    AgentLoop,
    AgentRunSpec,
    AgentRuntimeConfig,
    ToolBinding,
)
from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore  # noqa: E402
from monoid_agent_kernel.core.lifecycle import LoopSession  # noqa: E402
from monoid_agent_kernel.providers.base import ModelTurn  # noqa: E402
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call  # noqa: E402
from monoid_agent_kernel.tools import tool_ids  # noqa: E402


def run_local_product(root: Path) -> dict[str, Any]:
    """Run one apply-mode agent and verify its durable local artifacts."""

    workspace = root / "workspace"
    run_root = root / "runs"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "request.txt").write_text("Prepare an offline release note.\n", encoding="utf-8")

    config = AgentRuntimeConfig(
        definition_id="embedding-local-product",
        tools=(
            ToolBinding.for_tool(tool_ids.FS_WRITE),
            ToolBinding.for_tool(tool_ids.RUN_FINISH),
        ),
    )
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="local-write",
                tool_calls=(
                    fake_tool_call(
                        "fs_write",
                        {
                            "path": "RELEASE_NOTE.md",
                            "content": "# Release note\n\nOffline embedding is ready.\n",
                        },
                        "call-write",
                    ),
                ),
            ),
            ModelTurn(response_id="local-done", final_text="Created RELEASE_NOTE.md."),
        ]
    )
    checkpoints = LocalFsCheckpointStore(run_root)
    loop = AgentLoop.from_config(
        AgentRunSpec(workspace_root=workspace, run_root=run_root, mode="apply"),
        adapter,
        config,
        checkpoint_store=checkpoints,
    )
    session = LoopSession(loop)
    session.open()
    session.submit("Create the requested release note.")
    checkpoint = checkpoints.latest(loop.spec.run_id)
    result = session.close()

    events_path = run_root / result.run_id / "events.jsonl"
    event_count = len(events_path.read_text(encoding="utf-8").splitlines())
    return {
        "status": result.status,
        "run_id": result.run_id,
        "output_exists": (workspace / "RELEASE_NOTE.md").is_file(),
        "checkpoint_seq": checkpoint.seq if checkpoint is not None else 0,
        "event_count": event_count,
        "network_required": False,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="monoid-local-embedding-") as tmp:
        print(json.dumps(run_local_product(Path(tmp)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
