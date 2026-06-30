from __future__ import annotations

from pathlib import Path

from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core.context import TurnContext
from monoid_agent_kernel.core.prompt import compose_system_prompt
from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call


class _MarkerProvider:
    def __init__(self, static: str | None = None, dynamic: str | None = None) -> None:
        self._static = static
        self._dynamic = dynamic

    def static_segment(self) -> str | None:
        return self._static

    def dynamic_segment(self, turn: TurnContext) -> str | None:
        if self._dynamic is None:
            return None
        return f"{self._dynamic} step={turn.step} remaining_steps={turn.remaining_steps}"


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.md").write_text("hi\n", encoding="utf-8")
    return workspace


def _finish_only() -> FakeModelAdapter:
    return FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("run_finish", {"summary": "done"}, "c"),),
            )
        ]
    )


def _spec(tmp_path: Path, workspace: Path) -> AgentRunSpec:
    return AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")


def _provider(*tool_ids: str):
    return runtime_provider(runtime_config(*(tool_ids or ("run.finish",))))


def test_static_segment_folded_into_every_turn(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    adapter = _finish_only()
    AgentLoop(
        spec=_spec(tmp_path, workspace),
        model_adapter=adapter,
        context_providers=(_MarkerProvider(static="STATIC-MARKER"),),
        runtime_config_provider=_provider(),
    ).run_once("go")
    assert "STATIC-MARKER" in adapter.requests[0].system_prompt
    assert adapter.requests[0].system_prompt == compose_system_prompt(
        persona_segments=("STATIC-MARKER",)
    )


def test_dynamic_segment_appended_per_turn_with_live_turn_data(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("fs_list", {"path": "."}, "c1"),)),
            ModelTurn(response_id="r2", tool_calls=(fake_tool_call("run_finish", {"summary": "ok"}, "c2"),)),
        ]
    )
    AgentLoop(
        spec=_spec(tmp_path, workspace),
        model_adapter=adapter,
        context_providers=(_MarkerProvider(dynamic="DYN"),),
        runtime_config_provider=_provider("fs.list", "run.finish"),
    ).run_once("go")
    # Static prompt is unchanged; the dynamic segment is appended and reflects the live step.
    assert adapter.requests[0].system_prompt.startswith(compose_system_prompt())
    assert "DYN step=1 remaining_steps=" in adapter.requests[0].system_prompt
    assert "DYN step=2 remaining_steps=" in adapter.requests[1].system_prompt


def test_no_dynamic_keeps_prompt_equal_to_static(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    adapter = _finish_only()
    AgentLoop(
        spec=_spec(tmp_path, workspace),
        model_adapter=adapter,
        # static only; dynamic_segment returns None
        context_providers=(_MarkerProvider(static="S"),),
        runtime_config_provider=_provider(),
    ).run_once("go")
    expected_static = compose_system_prompt(persona_segments=("S",))
    assert adapter.requests[0].system_prompt == expected_static


def test_inject_workspace_index_adds_file_listing(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    adapter = _finish_only()
    AgentLoop(
        spec=_spec(tmp_path, workspace),
        model_adapter=adapter,
        inject_workspace_index=True,
        runtime_config_provider=_provider(),
    ).run_once("go")
    prompt = adapter.requests[0].system_prompt
    assert "Workspace files (initial snapshot):" in prompt
    assert "notes.md" in prompt
