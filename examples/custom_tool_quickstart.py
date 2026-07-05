"""Author a custom tool and watch the model call it — in one `from_tools` call.

The shortest path from a `@tool` function to a run where the model actually invokes it. No
hand-wrapped ToolProvider, no hand-written ToolBinding: `AgentLoop.from_tools` registers the
tool and generates its binding for you. Runs offline with a scripted `FakeModelAdapter` (no
gateway, no API key).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make the example runnable from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from monoid_agent_kernel import AgentLoop, AgentRunSpec, tool  # noqa: E402
from monoid_agent_kernel.providers.base import ModelTurn  # noqa: E402
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call  # noqa: E402


# A custom tool. @tool derives the input schema from the type hints; its model-facing name is
# the id with dots → underscores (here, "skill_word_count").
@tool(id="skill.word_count", capability="run.control", side_effect="run")
def word_count(text: str) -> dict:
    """Count whitespace-separated words in a text string."""
    return {"words": len(text.split())}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "workspace"
        workspace.mkdir()
        spec = AgentRunSpec(workspace_root=workspace, run_root=Path(tmp) / "runs")

        # Scripted model: turn 1 calls the custom tool (by its exported name), turn 2 settles
        # with final text (a run settles on final text with no tool calls — no run.finish needed).
        adapter = FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="t1",
                    tool_calls=(
                        fake_tool_call("skill_word_count", {"text": "alpha beta gamma"}, "c1"),
                    ),
                ),
                ModelTurn(response_id="t2", final_text="Counted 3 words."),
            ]
        )

        # One call: register the tool + generate its binding + run.
        result = AgentLoop.from_tools(spec, adapter, [word_count]).run_once("Count the words.")

        # The tool result reached the model as an observation on the next request.
        observations = [obs for req in adapter.requests for obs in req.observations]
        print("status     :", result.status)
        print("final_text :", result.final_text)
        print("tool_result:", observations[0].output if observations else None)


if __name__ == "__main__":
    main()
