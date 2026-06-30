from __future__ import annotations

from pathlib import Path

from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core.spec import AgentRunSpec, RunLimits
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call


def _provider(*tool_ids: str):
    return runtime_provider(runtime_config(*(tool_ids or ("fs.write",))))


def _write_turn(handle: str, path: str, content: str, call_id: str) -> ModelTurn:
    return ModelTurn(
        response_id=handle,
        tool_calls=(fake_tool_call("fs_write", {"path": path, "content": content}, call_id),),
    )


def test_multi_turn_threads_handle_and_uses_third_shape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            _write_turn("r1", "A.md", "alpha\n", "c1"),
            ModelTurn(response_id="r2", final_text="wrote A"),
            _write_turn("r3", "B.md", "beta\n", "c2"),
            ModelTurn(response_id="r4", final_text="wrote B"),
        ]
    )
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=_provider("fs.write"),
    )

    loop.open()
    first = loop.submit("first")
    second = loop.submit("second")
    result = loop.close()

    assert first.status == "completed"
    assert first.final_text == "wrote A"
    assert "A.md" in first.changed_paths
    assert first.turn_handle == "r2"

    # Workspace changes accumulate across submits (no commit between them).
    assert second.final_text == "wrote B"
    assert set(second.changed_paths) == {"A.md", "B.md"}

    # First turn ever: first-turn shape (no handle, instruction set).
    assert adapter.requests[0].instruction == "first"
    assert adapter.requests[0].previous_turn_handle is None
    # Second submit's first turn: third shape (handle from prior submit + new message).
    third_shape = next(r for r in adapter.requests if r.instruction == "second")
    assert third_shape.previous_turn_handle == "r2"
    assert third_shape.observations == ()

    # Tool-continuation turns carry observations, never the user message.
    continuation = next(r for r in adapter.requests if r.observations)
    assert continuation.instruction is None

    assert result.status == "completed"
    assert result.final_turn_handle == "r4"


def test_max_steps_budget_resets_per_submit(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            # submit 1: a tool call with only a 1-step budget -> never settles -> limited
            _write_turn("r1", "A.md", "alpha\n", "c1"),
            # submit 2: settles immediately on a fresh 1-step budget
            ModelTurn(response_id="r2", final_text="done"),
        ]
    )
    loop = AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace,
            run_root=tmp_path / "runs",
            limits=RunLimits(max_steps=1),
        ),
        model_adapter=adapter,
        runtime_config_provider=_provider("fs.write"),
    )

    loop.open()
    first = loop.submit("first")
    second = loop.submit("second")
    loop.close()

    assert first.status == "limited"
    assert first.error_code == "max_steps_exceeded"
    # The session survives a per-submit step exhaustion; the next submit runs fresh.
    assert second.status == "completed"
    assert second.final_text == "done"


def _rebaseline_loop(tmp_path: Path, backend: str) -> AgentLoop:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            _write_turn("r1", "A.md", "alpha\n", "c1"),
            ModelTurn(response_id="r2", final_text="wrote A"),
            _write_turn("r3", "B.md", "beta\n", "c2"),
            ModelTurn(response_id="r4", final_text="wrote B"),
        ]
    )
    return AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace,
            run_root=tmp_path / "runs",
            workspace_backend=backend,  # type: ignore[arg-type]
        ),
        model_adapter=adapter,
        runtime_config_provider=_provider("fs.write"),
    )


def test_commit_checkpoint_rebaselines_overlay(tmp_path: Path) -> None:
    loop = _rebaseline_loop(tmp_path, "overlay")
    loop.open()
    first = loop.submit("write A")
    assert "A.md" in first.changed_paths

    loop.commit_checkpoint()

    second = loop.submit("write B")
    # After the commit, only post-commit changes are reported.
    assert second.changed_paths == ("B.md",)
    loop.close()


def test_commit_checkpoint_rebaselines_staging(tmp_path: Path) -> None:
    loop = _rebaseline_loop(tmp_path, "staging")
    loop.open()
    loop.submit("write A")
    loop.commit_checkpoint()
    second = loop.submit("write B")
    assert second.changed_paths == ("B.md",)
    loop.close()


def test_unchanged_checkpoints_have_stable_proposal_hash(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", final_text="nothing to do"),
            ModelTurn(response_id="r2", final_text="still nothing"),
        ]
    )
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=_provider("fs.write"),
    )

    loop.open()
    first = loop.submit("one")
    second = loop.submit("two")
    loop.close()

    # No workspace change between settles -> identical content -> identical hash
    # (updated_at is excluded from the proposal hash).
    assert first.changed_paths == ()
    assert second.proposal_hash == first.proposal_hash


def test_submit_requires_open(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=FakeModelAdapter(),
        runtime_config_provider=_provider("fs.write"),
    )
    try:
        loop.submit("hello")
    except Exception as exc:  # NativeAgentError
        assert "not open" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("submit() should require open()")
