"""Use domain models in a custom tool through one `from_tools` call.

`AgentLoop.from_tools` registers the tool and generates its binding. A scripted
`FakeModelAdapter` invokes it offline and receives the typed result on its next request.
Install this checkout: python -m pip install .
Run: python examples/custom_tool_quickstart.py
"""

import tempfile
from pathlib import Path

from pydantic import BaseModel

from monoid_agent_kernel import AgentLoop, AgentRunSpec, tool
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call


class Customer(BaseModel):
    name: str


class ReportDraft(BaseModel):
    customer: Customer
    title: str


class ReportSummary(BaseModel):
    customer: str
    title: str


# @tool derives the JSON input schema, passes validated model instances to the function,
# and serializes its top-level model result in JSON mode.
@tool(id="skill.summarize_report", capability="run.control", side_effect="run")
def summarize_report(draft: ReportDraft) -> ReportSummary:
    """Summarize a report draft for its customer."""
    return ReportSummary(customer=draft.customer.name, title=draft.title)


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
                        fake_tool_call(
                            "skill_summarize_report",
                            {"draft": {"customer": {"name": "Ada"}, "title": "Weekly report"}},
                            "c1",
                        ),
                    ),
                ),
                ModelTurn(response_id="t2", final_text="Prepared Ada's weekly report."),
            ]
        )

        # One call: register the tool + generate its binding + run.
        result = AgentLoop.from_tools(spec, adapter, [summarize_report]).run_once("Summarize the draft.")

        # The tool result reached the model as an observation on the next request.
        expected = {"ok": True, "result": {"customer": "Ada", "title": "Weekly report"}}
        assert result.status == "completed", result.error
        observation = adapter.requests[1].observations[0].output
        assert observation == expected, observation
        print("status     :", result.status)
        print("final_text :", result.final_text)
        print("tool_result:", observation)


if __name__ == "__main__":
    main()
