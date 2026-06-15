from __future__ import annotations

import json
import inspect
from pathlib import Path

from click.testing import CliRunner

from native_agent_runner.cli import main
from native_agent_runner.core.events import EventBus
from native_agent_runner.core.projections import project_run_status
from native_agent_runner.core.spec import AgentRunSpec, RunLimits
from native_agent_runner.loop import AgentLoop
from native_agent_runner.permissions import PermissionPolicy
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call
from native_agent_runner.recorder import JsonlEventSink, MemoryEventSink, StatusJsonSink


def _events(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in run_dir.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _isolated_cli_runner() -> tuple[CliRunner, bool]:
    if "mix_stderr" in inspect.signature(CliRunner).parameters:
        return CliRunner(mix_stderr=False), True
    return CliRunner(), False


def test_event_bus_schema_sequence_and_memory_sink() -> None:
    memory = MemoryEventSink()
    bus = EventBus("run_test", (memory,))

    first = bus.emit("run.started", data={"mode": "propose"})
    second = bus.emit("run.finished", data={"status": "completed"})
    bus.close()

    assert first.seq == 1
    assert second.seq == 2
    assert first.event_id != second.event_id
    assert first.timestamp.endswith("Z")
    assert memory.events == [first, second]
    payload = first.to_json()
    assert payload["schema_version"] == "native-agent-runner.event.v1"
    assert payload["type"] == "run.started"
    assert "kind" not in payload


def test_jsonl_and_status_sinks_flush_and_update(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    status_path = tmp_path / "status.json"
    bus = EventBus("run_sink", (JsonlEventSink(events_path), StatusJsonSink(status_path)))

    bus.emit("run.started", data={"workspace": "w", "mode": "propose", "model": "gpt-5.5"})
    assert events_path.read_text(encoding="utf-8").strip()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "running"

    bus.emit("run.finished", data={"status": "completed", "final_text": "done"})
    bus.close()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["last_event_type"] == "run.finished"


def test_loop_events_are_ordered_and_status_file_exists(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.md").write_text("rough notes\n", encoding="utf-8")
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
            ModelTurn(final_text="done"),
        ]
    )
    spec = AgentRunSpec(instruction="Clean notes.", workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    types = [event["type"] for event in _events(result.run_dir)]
    assert types[0] == "run.started"
    assert "model.turn.started" in types
    assert "tool.call.started" in types
    assert "tool.call.finished" in types
    assert "workspace.file.changed" in types
    assert "workspace.diff.updated" in types
    assert "workspace.proposal.updated" in types
    assert types[-1] == "run.finished"
    status = json.loads(result.run_dir.joinpath("status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["proposal"]["path"] == "proposal.json"
    assert "+Clean summary" in result.diff_path.read_text(encoding="utf-8")
    proposal = json.loads(result.run_dir.joinpath("proposal.json").read_text(encoding="utf-8"))
    assert proposal["files"][0]["snapshot_path"] == "proposal/files/SUMMARY.md"
    assert proposal["proposal_hash"]
    assert status["manifest_path"] == "manifest.json"
    workspace_index = json.loads(result.run_dir.joinpath("workspace.index.json").read_text(encoding="utf-8"))
    assert workspace_index["schema_version"] == "native-agent-runner.workspace-index.v1"
    assert any(entry["path"] == "notes.md" for entry in workspace_index["entries"])
    projection = project_run_status(result.run_dir)
    assert projection["status"] == "completed"
    assert projection["proposal_hash"] == proposal["proposal_hash"]
    assert projection["changed_paths"] == ["SUMMARY.md"]


def test_public_events_redact_tool_arguments_and_policy_redacted_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.md").write_text("alpha\n", encoding="utf-8")
    (workspace / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    private_key = "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----"
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call(
                        "fs_write",
                        {"path": "SECRET.md", "content": private_key, "create_dirs": False},
                        "call_write",
                    ),
                    fake_tool_call(
                        "fs_patch",
                        {
                            "path": "notes.md",
                            "replacements": [{"old": "alpha", "new": "beta-token-value"}],
                        },
                        "call_patch",
                    ),
                    fake_tool_call("fs_read", {"path": ".env"}, "call_env"),
                ),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    spec = AgentRunSpec(instruction="Edit notes.", workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        permission_policy=PermissionPolicy(redact_patterns=(".env",)),
    ).run()

    events_text = result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8")
    assert "BEGIN PRIVATE KEY" not in events_text
    assert "alpha" not in events_text
    assert "beta-token-value" not in events_text
    assert ".env" not in events_text
    assert "TOKEN=secret" not in events_text
    assert "[redacted-path]" in events_text
    assert '"redacted": true' in events_text
    transcript = result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")
    assert "BEGIN PRIVATE KEY" in transcript
    index_text = result.run_dir.joinpath("workspace.index.json").read_text(encoding="utf-8")
    assert ".env" in index_text
    assert "TOKEN=secret" not in index_text


def test_status_projection_redacts_paths_from_manifest_policy(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call(
                        "fs_write",
                        {"path": ".env", "content": "TOKEN=secret\n", "create_dirs": False},
                        "call_write",
                    ),
                ),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    spec = AgentRunSpec(instruction="Create env.", workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        permission_policy=PermissionPolicy(redact_patterns=(".env",)),
    ).run()

    proposal = json.loads(result.run_dir.joinpath("proposal.json").read_text(encoding="utf-8"))
    assert proposal["changed_paths"] == [".env"]
    assert proposal["files"][0]["path"] == ".env"
    projection = project_run_status(result.run_dir)
    assert projection["changed_paths"] == ["[redacted-path]"]


def test_loop_records_unknown_malformed_and_permission_failures_as_events(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text("x", encoding="utf-8")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call("missing_tool", {}, "call_missing"),
                    fake_tool_call("fs_read", {}, "call_bad_args"),
                    fake_tool_call("fs_read", {"path": ".env"}, "call_denied"),
                ),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    spec = AgentRunSpec(instruction="Try tools.", workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        permission_policy=PermissionPolicy(deny_patterns=(".env",)),
    ).run()

    events = _events(result.run_dir)
    failed = [event for event in events if event["type"] == "tool.call.failed"]
    assert [event["data"]["call_id"] for event in failed] == [
        "call_missing",
        "call_bad_args",
        "call_denied",
    ]
    assert [event["data"]["error_code"] for event in failed] == [
        "tool_unknown",
        "tool_args_invalid",
        "permission_denied",
    ]
    assert any(event["type"] == "permission.denied" for event in events)
    assert events[-1]["type"] == "run.finished"
    assert events[-1]["data"]["status"] == "completed"


def test_loop_limited_status_is_public_event(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[ModelTurn(response_id="r1", tool_calls=(fake_tool_call("fs_list", {"path": "."}, "c1"),))]
    )
    spec = AgentRunSpec(
        instruction="Loop.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        limits=RunLimits(max_steps=1, max_tool_calls=0),
    )

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    final_event = _events(result.run_dir)[-1]
    assert result.status == "limited"
    assert final_event["type"] == "run.finished"
    assert final_event["data"]["status"] == "limited"
    assert final_event["data"]["error_code"] == "max_tool_calls_exceeded"


def test_cli_stream_json_normal_output_watch_and_custom_sink(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sink_output = tmp_path / "sink-events.txt"
    sink_module = tmp_path / "sink_module.py"
    sink_module.write_text(
        """
import os
from pathlib import Path

class Sink:
    def __init__(self):
        self.path = Path(os.environ["NAR_TEST_SINK_PATH"])

    def emit(self, event):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(event.type + "\\n")
            handle.flush()

    def close(self):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("closed\\n")

def make_sink():
    return Sink()
""",
        encoding="utf-8",
    )

    class FakeCliGatewayAdapter:
        def __init__(self, _config, **_kwargs):
            self._adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="done")])

        def next_turn(self, request):
            return self._adapter.next_turn(request)

    monkeypatch.setattr("native_agent_runner.cli.GatewayModelAdapter", FakeCliGatewayAdapter)
    monkeypatch.setenv("NAR_TEST_SINK_PATH", str(sink_output))
    runner, has_separate_stderr = _isolated_cli_runner()
    run_root = tmp_path / "runs"

    result = runner.invoke(
        main,
        [
            "run",
            "--workspace",
            str(workspace),
            "--instruction",
            "Finish.",
            "--run-root",
            str(run_root),
            "--run-id",
            "cli_stream",
            "--stream-json",
            "--event-sink-module",
            f"{sink_module}:make_sink",
        ],
    )

    stderr_text = result.stderr if has_separate_stderr else result.output
    stdout_text = result.stdout if has_separate_stderr else result.output
    assert result.exit_code == 0, stderr_text
    stdout_events = [
        json.loads(line)
        for line in stdout_text.splitlines()
        if line.strip().startswith("{")
    ]
    assert stdout_events[0]["type"] == "run.started"
    assert stdout_events[-1]["type"] == "run.finished"
    assert "run_id: cli_stream" in stderr_text
    assert sink_output.read_text(encoding="utf-8").splitlines()[-1] == "closed"

    watch_result = runner.invoke(
        main,
        ["watch", "cli_stream", "--run-root", str(run_root), "--from-start", "--json"],
    )
    assert watch_result.exit_code == 0
    watched = [json.loads(line) for line in watch_result.stdout.splitlines() if line.strip()]
    assert [event["type"] for event in watched] == [event["type"] for event in stdout_events]

    validate_result = runner.invoke(
        main,
        ["validate", "cli_stream", "--run-root", str(run_root), "--json"],
    )
    assert validate_result.exit_code == 0
    assert json.loads(validate_result.stdout)["ok"] is True

    status_result = runner.invoke(
        main,
        ["status", "cli_stream", "--run-root", str(run_root), "--json"],
    )
    assert status_result.exit_code == 0
    status_payload = json.loads(status_result.stdout)
    assert status_payload["status"] == "completed"
    assert status_payload["last_event_type"] == "run.finished"


def test_cli_normal_mode_prints_run_identity_before_completion(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class FakeCliGatewayAdapter:
        def __init__(self, _config, **_kwargs):
            self._adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="done")])

        def next_turn(self, request):
            return self._adapter.next_turn(request)

    monkeypatch.setattr("native_agent_runner.cli.GatewayModelAdapter", FakeCliGatewayAdapter)
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--workspace",
            str(workspace),
            "--instruction",
            "Finish.",
            "--run-root",
            str(tmp_path / "runs"),
            "--run-id",
            "cli_normal",
        ],
    )

    assert result.exit_code == 0
    lines = result.output.splitlines()
    assert lines[0] == "run_id: cli_normal"
    assert lines[1].startswith("run_dir: ")
    assert "status: completed" in result.output


def test_cli_proposal_command_reads_snapshot_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call(
                        "fs_write",
                        {"path": "SUMMARY.md", "content": "Clean summary\n", "create_dirs": False},
                        "call_write",
                    ),
                ),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    spec = AgentRunSpec(instruction="Write summary.", workspace_root=workspace, run_root=tmp_path / "runs")
    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    runner = CliRunner()
    summary = runner.invoke(main, ["proposal", str(result.run_dir), "--file", "SUMMARY.md", "--json"])

    assert summary.exit_code == 0
    payload = json.loads(summary.stdout)
    assert payload["encoding"] == "utf-8"
    assert payload["content"] == "Clean summary\n"
