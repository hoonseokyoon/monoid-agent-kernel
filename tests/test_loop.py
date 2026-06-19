from __future__ import annotations

import json
from pathlib import Path

from conftest import runtime_config, runtime_provider, tool_binding

from native_agent_runner.core.cancellation import CancellationToken
from native_agent_runner.core.schemas import validate_run_dir
from native_agent_runner.core.spec import AgentRunSpec, RunLimits
from native_agent_runner.core.tool_surface import ToolQuota
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call
from native_agent_runner.workspace.local import default_local_workspace_factory, sha256_bytes


DEFAULT_TOOLS = (
    "fs.read",
    "fs.write",
    "fs.patch",
    "fs.list",
    "fs.tree",
    "fs.stat",
    "fs.glob",
    "fs.copy",
    "fs.move",
    "fs.delete",
    "run.finish",
)


def _provider(*tool_ids: str):
    return runtime_provider(runtime_config(*(tool_ids or DEFAULT_TOOLS)))


def _python_command(code: str) -> str:
    return f'python -c "{code.replace(chr(34), chr(92) + chr(34))}"'


def _finish_only_adapter() -> FakeModelAdapter:
    return FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("run_finish", {"summary": "done"}, "call_finish"),),
            ),
        ]
    )


def test_message_log_cap_settles_run_as_limited(tmp_path: Path) -> None:
    # A by-value conversation log that outgrows max_message_log_bytes settles the run as
    # ``limited`` (a safe stop, not a drop) before the over-limit log is ever sent.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = _finish_only_adapter()
    spec = AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        limits=RunLimits(max_message_log_bytes=10),
    )

    result = AgentLoop(
        spec=spec, model_adapter=adapter, runtime_config_provider=_provider("run.finish")
    ).run_once("This instruction is clearly longer than ten bytes.")

    assert result.status == "limited"
    assert result.error_code == "message_log_bytes_exceeded"
    assert adapter.requests == []  # the over-limit log is never sent to the model


def test_default_system_prompt_is_composed_base(tmp_path: Path) -> None:
    from native_agent_runner.core.prompt import compose_system_prompt

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = _finish_only_adapter()
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")

    AgentLoop(spec=spec, model_adapter=adapter, runtime_config_provider=_provider("run.finish")).run_once(
        "Inspect."
    )

    assert adapter.requests[0].system_prompt == compose_system_prompt()


def test_run_finish_surfaces_outputs_and_notes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("rough notes\n", encoding="utf-8")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call(
                        "run_finish",
                        {
                            "summary": "Reviewed the notes",
                            "outputs": ["notes.md", "SUMMARY.md"],
                            "notes": "No changes were necessary.",
                        },
                        "call_finish",
                    ),
                ),
            ),
        ]
    )

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=_provider("run.finish"),
    ).run_once("Review.")

    assert result.status == "completed"
    assert result.final_text == "Reviewed the notes"
    assert result.final_outputs == ("notes.md", "SUMMARY.md")
    assert result.final_notes == "No changes were necessary."


def test_loop_read_write_finish_happy_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("rough notes\n", encoding="utf-8")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call("fs_read", {"path": "notes.md"}, "call_read"),
                    fake_tool_call(
                        "fs_write",
                        {"path": "SUMMARY.md", "content": "Clean summary\n", "create_dirs": False},
                        "call_write",
                    ),
                ),
            ),
            ModelTurn(
                response_id="r2",
                tool_calls=(fake_tool_call("run_finish", {"summary": "Created SUMMARY.md"}, "call_finish"),),
            ),
        ]
    )

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=_provider(),
    ).run_once("Clean.")

    assert result.status == "completed"
    assert not workspace.joinpath("SUMMARY.md").exists()
    assert "+Clean summary" in result.diff_path.read_text(encoding="utf-8")
    proposal = json.loads(result.proposal_path.read_text(encoding="utf-8"))
    assert proposal["files"][0]["path"] == "SUMMARY.md"
    manifest = json.loads(result.run_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert manifest["agent_config"]["definition_id"] == "test-agent"
    assert any(tool["id"] == "fs.write" for tool in manifest["tool_specs"])
    assert validate_run_dir(result.run_dir) == []


def test_loop_staging_backend_records_base_hash(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_bytes(b"old\n")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call(
                        "fs_write",
                        {"path": "notes.md", "content": "new\n", "create_dirs": False},
                        "call_write",
                    ),
                    fake_tool_call("run_finish", {"summary": "Updated notes."}, "call_finish"),
                ),
            ),
        ]
    )
    result = AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace,
            run_root=tmp_path / "runs",
            workspace_backend="staging",
        ),
        model_adapter=adapter,
        runtime_config_provider=_provider("fs.write", "run.finish"),
    ).run_once("Update.")

    assert result.status == "completed"
    assert workspace.joinpath("notes.md").read_text(encoding="utf-8") == "new\n"
    file_info = json.loads(result.proposal_path.read_text(encoding="utf-8"))["files"][0]
    assert file_info["base_sha256"] == sha256_bytes(b"old\n")


def test_loop_uses_injected_workspace_factory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")
    seen: list[AgentRunSpec] = []

    def factory(run_spec: AgentRunSpec):
        seen.append(run_spec)
        return default_local_workspace_factory(run_spec)

    result = AgentLoop(
        spec=spec,
        model_adapter=_finish_only_adapter(),
        workspace_factory=factory,
        runtime_config_provider=_provider("run.finish"),
    ).run_once("noop")

    assert result.status == "completed"
    assert seen == [spec]


def test_unknown_tool_is_recorded_as_observation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("missing_tool", {}, "call_missing"),)),
            ModelTurn(final_text="done"),
        ]
    )

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=_provider("run.finish"),
    ).run_once("Do it.")

    assert result.status == "completed"
    assert "unknown tool" in result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")


def test_absent_binding_means_tool_unavailable(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_write", {"path": "x.md", "content": "x"}, "c1"),),
            ),
            ModelTurn(final_text="done"),
        ]
    )

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=_provider("fs.read", "run.finish"),
    ).run_once("Write.")

    assert result.status == "completed"
    assert "fs.write" not in {tool.id for tool in adapter.requests[0].tools}
    assert "unknown tool" in result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")


def test_binding_authorization_and_quota_are_enforced(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("old.md").write_text("old\n", encoding="utf-8")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call("fs_read", {"path": "old.md"}, "read1"),
                    fake_tool_call("fs_read", {"path": "old.md"}, "read2"),
                    fake_tool_call("fs_delete", {"path": "old.md"}, "delete1"),
                ),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    config = runtime_config(
        bindings=(
            tool_binding("fs.read", quota=ToolQuota(max_calls_per_run=1)),
            tool_binding("fs.delete", authorization="ask"),
            tool_binding("run.finish"),
        )
    )

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
    ).run_once("Try tools.")

    transcript = result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")
    assert "tool_quota_exceeded" in transcript
    assert "tool_approval_required" in transcript
    assert workspace.joinpath("old.md").exists()


def test_shell_binding_auto_approve_updates_proposal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call(
                        "shell_exec",
                        {
                            "command": _python_command(
                                "from pathlib import Path; Path('SHELL.md').write_text('shell\\n', encoding='utf-8')"
                            )
                        },
                        "c1",
                    ),
                ),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    config = runtime_config(
        bindings=(
            tool_binding(
                "shell.exec",
                runtime={"shell": {"approval_mode": "auto-approve", "default_timeout_s": 30}},
            ),
            tool_binding("run.finish"),
        )
    )

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
    ).run_once("Use shell.")

    assert result.status == "completed"
    assert not workspace.joinpath("SHELL.md").exists()
    assert result.run_dir.joinpath("proposal", "files", "SHELL.md").read_text(encoding="utf-8") == "shell\n"
    events = result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8")
    assert "tool.approval.approved" in events
    assert "shell.exec.finished" in events


def test_loop_limits_and_cancellation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    limited = AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace,
            run_root=tmp_path / "limited",
            limits=RunLimits(max_steps=1),
        ),
        model_adapter=FakeModelAdapter(
            turns=[
                ModelTurn(response_id="r1", tool_calls=(fake_tool_call("fs_list", {"path": "."}, "c1"),)),
                ModelTurn(response_id="r2"),
            ]
        ),
        runtime_config_provider=_provider("fs.list", "run.finish"),
    ).run_once("Loop.")
    assert limited.status == "limited"
    assert limited.error_code == "max_steps_exceeded"

    token = CancellationToken()
    token.cancel()
    cancelled = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "cancelled"),
        model_adapter=FakeModelAdapter(turns=[ModelTurn(final_text="done")]),
        runtime_config_provider=_provider("run.finish"),
        cancellation_token=token,
    ).run_once("Finish.")
    assert cancelled.status == "limited"
    assert cancelled.error_code == "cancelled"
