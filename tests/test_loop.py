from __future__ import annotations

import json
from pathlib import Path

from native_agent_runner.core.cancellation import CancellationToken
from native_agent_runner.core.schemas import validate_run_dir
from native_agent_runner.core.spec import AgentRunSpec, RunLimits
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call
from native_agent_runner.shell import ShellPolicy
from native_agent_runner.tools.policy import ToolPolicy
from native_agent_runner.workspace.local import sha256_bytes


def _python_command(code: str) -> str:
    escaped = code.replace('"', '\\"')
    return f'python -c "{escaped}"'


def test_loop_read_write_finish_happy_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.md").write_text("rough notes\n", encoding="utf-8")
    run_root = tmp_path / "runs"
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
                tool_calls=(
                    fake_tool_call(
                        "run_finish",
                        {"summary": "Created SUMMARY.md", "outputs": ["SUMMARY.md"]},
                        "call_finish",
                    ),
                ),
            ),
        ]
    )
    spec = AgentRunSpec(
        instruction="Clean the notes.",
        workspace_root=workspace,
        run_root=run_root,
        mode="propose",
    )

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    assert result.status == "completed"
    assert result.final_text == "Created SUMMARY.md"
    assert not workspace.joinpath("SUMMARY.md").exists()
    assert "+Clean summary" in result.diff_path.read_text(encoding="utf-8")
    proposal = json.loads(result.proposal_path.read_text(encoding="utf-8"))
    assert proposal["schema_version"] == "native-agent-runner.proposal.v2"
    assert proposal["proposal_hash"]
    assert proposal["diff_sha256"]
    assert proposal["files"][0]["path"] == "SUMMARY.md"
    assert proposal["files"][0]["change_kind"] == "created"
    assert proposal["files"][0]["base_sha256"] is None
    assert proposal["files"][0]["proposed_sha256"] == proposal["files"][0]["snapshot_sha256"]
    assert result.run_dir.joinpath("proposal", "files", "SUMMARY.md").read_text(encoding="utf-8") == "Clean summary\n"
    assert result.run_dir.joinpath("events.jsonl").exists()
    assert result.run_dir.joinpath("transcript.jsonl").exists()
    exposed = {tool.id for tool in adapter.requests[0].tools}
    assert {"fs.copy", "fs.move", "fs.delete"}.issubset(exposed)
    manifest = json.loads(result.run_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "native-agent-runner.manifest.v1"
    assert manifest["run_id"] == result.run_id
    assert manifest["model"] == "gpt-5.5"
    assert manifest["workspace_backend"] == "overlay"
    assert manifest["workspace_base_path"] == "workspace.base.json"
    assert "fs.read" in manifest["capabilities"]
    assert manifest["tool_policy"]["visible_tools"]
    assert any(tool["id"] == "fs.write" for tool in manifest["tool_specs"])
    assert manifest["workspace_index_path"] == "workspace.index.json"
    workspace_index = json.loads(result.run_dir.joinpath("workspace.index.json").read_text(encoding="utf-8"))
    assert workspace_index["run_id"] == result.run_id
    assert any(entry["path"] == "notes.md" for entry in workspace_index["entries"])
    workspace_base = json.loads(result.run_dir.joinpath("workspace.base.json").read_text(encoding="utf-8"))
    assert workspace_base["schema_version"] == "native-agent-runner.workspace-base.v1"
    assert workspace_base["workspace_backend"] == "overlay"
    assert any(entry["path"] == "notes.md" for entry in workspace_base["entries"])
    transcript = [
        json.loads(line)
        for line in result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(item["step"] >= 1 for item in transcript)
    metrics = json.loads(result.run_dir.joinpath("metrics.json").read_text(encoding="utf-8"))
    assert metrics["tool_calls"] == 3
    assert metrics["error_code"] == ""
    assert validate_run_dir(result.run_dir) == []


def test_loop_staging_backend_writes_to_staging_workspace_and_records_proposal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.md").write_bytes(b"old\n")
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
    spec = AgentRunSpec(
        instruction="Update notes.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        mode="propose",
        workspace_backend="staging",
    )

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    assert result.status == "completed"
    assert workspace.joinpath("notes.md").read_text(encoding="utf-8") == "new\n"
    proposal = json.loads(result.proposal_path.read_text(encoding="utf-8"))
    assert proposal["files"][0]["path"] == "notes.md"
    assert proposal["files"][0]["change_kind"] == "modified"
    assert proposal["files"][0]["base_sha256"] == sha256_bytes(b"old\n")
    manifest = json.loads(result.run_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert manifest["workspace_backend"] == "staging"
    metrics = json.loads(result.run_dir.joinpath("metrics.json").read_text(encoding="utf-8"))
    assert metrics["workspace_backend"] == "staging"
    assert validate_run_dir(result.run_dir) == []


def test_validate_run_dir_rejects_malformed_transcript(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    result = AgentLoop(
        spec=AgentRunSpec(instruction="Finish.", workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
    ).run()

    result.run_dir.joinpath("transcript.jsonl").write_text(
        '{"kind":"model_turn","response_id":"r1","final_text":"done","tool_calls":[],"usage":{}}\n',
        encoding="utf-8",
    )

    issues = validate_run_dir(result.run_dir)
    assert any(issue.path.startswith("transcript.jsonl:1") for issue in issues)


def test_loop_records_unknown_tool_as_observation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("missing_tool", {}, "call_missing"),),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    spec = AgentRunSpec(instruction="Do it.", workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    assert result.status == "completed"
    transcript = result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")
    assert "unknown tool" in transcript


def test_loop_proposal_records_modified_file_base_and_proposed_hashes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_bytes(b"alpha\n")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call(
                        "fs_write",
                        {"path": "notes.md", "content": "beta\n", "create_dirs": False},
                        "call_write",
                    ),
                ),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    spec = AgentRunSpec(instruction="Rewrite.", workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    assert workspace.joinpath("notes.md").read_text(encoding="utf-8") == "alpha\n"
    proposal = json.loads(result.proposal_path.read_text(encoding="utf-8"))
    file_info = proposal["files"][0]
    assert file_info["path"] == "notes.md"
    assert file_info["change_kind"] == "modified"
    assert file_info["base_sha256"] == sha256_bytes(b"alpha\n")
    assert file_info["proposed_sha256"] == sha256_bytes(b"beta\n")
    assert file_info["snapshot_sha256"] == file_info["proposed_sha256"]


def test_loop_limits_steps(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("fs_list", {"path": "."}, "c1"),)),
            ModelTurn(response_id="r2", tool_calls=(fake_tool_call("fs_list", {"path": "."}, "c2"),)),
        ]
    )
    spec = AgentRunSpec(
        instruction="Loop.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        limits=RunLimits(max_steps=1),
    )

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    assert result.status == "limited"
    assert "max steps" in result.final_text
    assert result.error_code == "max_steps_exceeded"
    metrics = json.loads(result.run_dir.joinpath("metrics.json").read_text(encoding="utf-8"))
    assert metrics["error_code"] == "max_steps_exceeded"


def test_loop_permission_denied_for_write_in_read_only(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call(
                        "fs_write",
                        {"path": "x.md", "content": "x", "create_dirs": False},
                        "c1",
                    ),
                ),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    spec = AgentRunSpec(
        instruction="Write.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        mode="read-only",
    )

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    assert result.status == "completed"
    assert not workspace.joinpath("x.md").exists()
    assert "fs.write" not in {tool.id for tool in adapter.requests[0].tools}
    assert "fs.copy" not in {tool.id for tool in adapter.requests[0].tools}
    assert "fs.move" not in {tool.id for tool in adapter.requests[0].tools}
    assert "fs.delete" not in {tool.id for tool in adapter.requests[0].tools}
    manifest = json.loads(result.run_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert "fs.write" not in {tool["id"] for tool in manifest["tool_specs"]}
    assert {"tool": "fs.write", "reason": "missing_capability"} in manifest["tool_policy"]["hidden_tools"]
    assert {"tool": "fs.delete", "reason": "missing_capability"} in manifest["tool_policy"]["hidden_tools"]
    assert "missing capability" in result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")


def test_loop_tool_policy_filters_model_visible_tools(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("hello\n", encoding="utf-8")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("fs_read", {"path": "notes.md"}, "c1"),)),
            ModelTurn(response_id="r2", tool_calls=(fake_tool_call("run_finish", {"summary": "done"}, "c2"),)),
        ]
    )
    spec = AgentRunSpec(
        instruction="Read only.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        tool_policy=ToolPolicy(allowed_tools=("fs.read", "run.finish")),
    )

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    assert result.status == "completed"
    exposed = {tool.id for tool in adapter.requests[0].tools}
    assert exposed == {"fs.read", "run.finish"}
    manifest = json.loads(result.run_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert {tool["id"] for tool in manifest["tool_specs"]} == {"fs.read", "run.finish"}


def test_loop_tool_policy_denies_stale_tool_call(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call(
                        "fs_write",
                        {"path": "x.md", "content": "x", "create_dirs": False},
                        "c1",
                    ),
                ),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    spec = AgentRunSpec(
        instruction="Try write.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        tool_policy=ToolPolicy(allowed_tools=("fs.read", "run.finish")),
    )

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    assert result.status == "completed"
    assert not workspace.joinpath("x.md").exists()
    transcript = result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")
    assert "tool_policy_denied" in transcript
    events = [
        json.loads(line)
        for line in result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    denied = [event for event in events if event["type"] == "permission.denied"]
    assert denied
    assert denied[0]["data"]["policy_decision"] == "deny"
    assert denied[0]["data"]["policy_reason"] == "not_in_tool_allowlist"


def test_loop_tool_policy_ask_requires_approval(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call(
                        "fs_write",
                        {"path": "x.md", "content": "x", "create_dirs": False},
                        "c1",
                    ),
                ),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    spec = AgentRunSpec(
        instruction="Try write.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        tool_policy=ToolPolicy(ask_tools=("fs.write",)),
    )

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    assert result.status == "completed"
    assert "fs.write" in {tool.id for tool in adapter.requests[0].tools}
    assert not workspace.joinpath("x.md").exists()
    transcript = result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")
    assert "tool_approval_required" in transcript


def test_loop_tool_policy_deny_and_ask_for_delete(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("old.md").write_text("old\n", encoding="utf-8")
    deny_adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_delete", {"path": "old.md"}, "c1"),),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    deny_spec = AgentRunSpec(
        instruction="Delete.",
        workspace_root=workspace,
        run_root=tmp_path / "deny-runs",
        tool_policy=ToolPolicy(denied_tools=("fs.delete",)),
    )

    deny_result = AgentLoop(spec=deny_spec, model_adapter=deny_adapter).run()

    assert deny_result.status == "completed"
    assert "fs.delete" not in {tool.id for tool in deny_adapter.requests[0].tools}
    assert workspace.joinpath("old.md").exists()
    assert "tool_policy_denied" in deny_result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")

    ask_adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_delete", {"path": "old.md"}, "c1"),),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    ask_spec = AgentRunSpec(
        instruction="Delete.",
        workspace_root=workspace,
        run_root=tmp_path / "ask-runs",
        tool_policy=ToolPolicy(ask_tools=("fs.delete",)),
    )

    ask_result = AgentLoop(spec=ask_spec, model_adapter=ask_adapter).run()

    assert ask_result.status == "completed"
    assert "fs.delete" in {tool.id for tool in ask_adapter.requests[0].tools}
    assert workspace.joinpath("old.md").exists()
    assert "tool_approval_required" in ask_result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")


def test_loop_shell_disabled_hides_tool_and_stale_call_reports_shell_disabled(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("shell_exec", {"command": "echo hi"}, "c1"),),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    spec = AgentRunSpec(instruction="Try shell.", workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    assert result.status == "completed"
    assert "shell.exec" not in {tool.id for tool in adapter.requests[0].tools}
    transcript = result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")
    assert "shell_disabled" in transcript
    manifest = json.loads(result.run_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert manifest["shell_policy"]["enabled"] is False
    assert {"tool": "shell.exec", "reason": "missing_capability"} in manifest["tool_policy"]["hidden_tools"]


def test_loop_shell_enabled_auto_approve_updates_proposal_without_base_write(tmp_path: Path) -> None:
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
    spec = AgentRunSpec(
        instruction="Use shell.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        shell_policy=ShellPolicy(enabled=True, approval_mode="auto-approve"),
    )

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    assert result.status == "completed"
    assert "shell.exec" in {tool.id for tool in adapter.requests[0].tools}
    assert not workspace.joinpath("SHELL.md").exists()
    proposal = json.loads(result.proposal_path.read_text(encoding="utf-8"))
    assert proposal["changed_paths"] == ["SHELL.md"]
    assert result.run_dir.joinpath("proposal", "files", "SHELL.md").read_text(encoding="utf-8") == "shell\n"
    events = [
        json.loads(line)
        for line in result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_types = [event["type"] for event in events]
    assert "tool.approval.requested" in event_types
    assert "tool.approval.approved" in event_types
    assert "shell.exec.finished" in event_types
    metrics = json.loads(result.run_dir.joinpath("metrics.json").read_text(encoding="utf-8"))
    assert metrics["shell_calls"] == 1
    assert metrics["failed_shell_calls"] == 0
    assert validate_run_dir(result.run_dir) == []


def test_loop_background_shell_job_reenters_with_result_observation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.md").write_text("notes\n", encoding="utf-8")
    command = _python_command(
        "import time; from pathlib import Path; "
        "time.sleep(0.2); Path('job-result.txt').write_text('job done\\n', encoding='utf-8')"
    )
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call(
                        "shell_exec",
                        {"command": command, "background": True, "resume_on_exit": True},
                        "call_shell",
                    ),
                ),
            ),
            ModelTurn(response_id="r2"),
            ModelTurn(
                response_id="r3",
                tool_calls=(
                    fake_tool_call(
                        "run_finish",
                        {"summary": "background job completed", "outputs": ["job-result.txt"]},
                        "call_finish",
                    ),
                ),
            ),
        ]
    )
    spec = AgentRunSpec(
        instruction="Run a background job and summarize it.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        mode="propose",
        workspace_backend="staging",
        shell_policy=ShellPolicy(enabled=True, approval_mode="auto-approve"),
    )

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    assert result.status == "completed"
    assert result.final_text == "background job completed"
    assert workspace.joinpath("job-result.txt").read_text(encoding="utf-8") == "job done\n"
    assert any(
        obs.output.get("job_id") and obs.output.get("status") == "running"
        for obs in adapter.requests[1].observations
    )
    assert any(
        obs.output.get("type") == "background_job_result" and obs.output.get("status") == "exited"
        for obs in adapter.requests[2].observations
    )
    events = [
        json.loads(line)["type"]
        for line in result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert "job.started" in events
    assert "run.waiting" in events
    assert "job.finished" in events
    assert "run.resumed" in events
    metrics = json.loads(result.run_dir.joinpath("metrics.json").read_text(encoding="utf-8"))
    assert metrics["background_jobs_started"] == 1
    assert metrics["background_jobs_finished"] == 1
    proposal = json.loads(result.run_dir.joinpath("proposal.json").read_text(encoding="utf-8"))
    assert "job-result.txt" in proposal["changed_paths"]
    assert validate_run_dir(result.run_dir) == []


def test_loop_background_shell_timeout_reenters_with_timed_out_status(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = _python_command("import time; time.sleep(5)")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call(
                        "shell_exec",
                        {"command": command, "background": True, "timeout_s": 1},
                        "call_shell",
                    ),
                ),
            ),
            ModelTurn(response_id="r2"),
            ModelTurn(
                response_id="r3",
                tool_calls=(
                    fake_tool_call(
                        "run_finish",
                        {"summary": "background job timed out", "outputs": []},
                        "call_finish",
                    ),
                ),
            ),
        ]
    )
    spec = AgentRunSpec(
        instruction="Run a timeout background job.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        shell_policy=ShellPolicy(
            enabled=True,
            approval_mode="auto-approve",
            default_timeout_s=1,
            max_timeout_s=1,
        ),
    )

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    assert result.status == "completed"
    assert any(
        obs.output.get("type") == "background_job_result" and obs.output.get("status") == "timed_out"
        for obs in adapter.requests[2].observations
    )
    metrics = json.loads(result.run_dir.joinpath("metrics.json").read_text(encoding="utf-8"))
    assert metrics["background_jobs_started"] == 1
    assert metrics["background_jobs_failed"] == 1
    events = [
        json.loads(line)["type"]
        for line in result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert "job.timed_out" in events
    assert validate_run_dir(result.run_dir) == []


def test_loop_tool_policy_deny_and_ask_for_shell(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    deny_adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("shell_exec", {"command": "echo hi"}, "c1"),)),
            ModelTurn(final_text="done"),
        ]
    )
    deny_spec = AgentRunSpec(
        instruction="Shell.",
        workspace_root=workspace,
        run_root=tmp_path / "deny-runs",
        shell_policy=ShellPolicy(enabled=True, approval_mode="auto-approve"),
        tool_policy=ToolPolicy(denied_tools=("shell.exec",)),
    )

    deny_result = AgentLoop(spec=deny_spec, model_adapter=deny_adapter).run()

    assert deny_result.status == "completed"
    assert "shell.exec" not in {tool.id for tool in deny_adapter.requests[0].tools}
    assert "tool_policy_denied" in deny_result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")

    ask_adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("shell_exec", {"command": "echo hi"}, "c1"),)),
            ModelTurn(final_text="done"),
        ]
    )
    ask_spec = AgentRunSpec(
        instruction="Shell.",
        workspace_root=workspace,
        run_root=tmp_path / "ask-runs",
        shell_policy=ShellPolicy(enabled=True, approval_mode="auto-approve"),
        tool_policy=ToolPolicy(ask_tools=("shell.exec",)),
    )

    ask_result = AgentLoop(spec=ask_spec, model_adapter=ask_adapter).run()

    assert ask_result.status == "completed"
    assert "shell.exec" in {tool.id for tool in ask_adapter.requests[0].tools}
    assert "tool_approval_required" in ask_result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")


def test_loop_limits_duration_before_model_turn(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    spec = AgentRunSpec(
        instruction="Finish.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        limits=RunLimits(max_duration_s=0),
    )

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    assert result.status == "limited"
    assert result.error_code == "run_timeout"
    assert adapter.requests == []
    metrics = json.loads(result.run_dir.joinpath("metrics.json").read_text(encoding="utf-8"))
    assert metrics["error_code"] == "run_timeout"


def test_loop_honors_cancellation_token(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    token = CancellationToken()
    token.cancel()
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    spec = AgentRunSpec(instruction="Finish.", workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop(spec=spec, model_adapter=adapter, cancellation_token=token).run()

    assert result.status == "limited"
    assert result.error_code == "cancelled"
    assert adapter.requests == []
