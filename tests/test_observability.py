from __future__ import annotations

import json
import inspect
from pathlib import Path

from click.testing import CliRunner

from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.cli import main
from monoid_agent_kernel.core.events import EventBus
from monoid_agent_kernel.core.projections import project_run_status
from monoid_agent_kernel.core.spec import AgentRunSpec, RunLimits
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.public_view import args_preview
from monoid_agent_kernel.recorder import JsonlEventSink, MemoryEventSink, StatusJsonSink


DEFAULT_TOOLS = (
    "fs.read",
    "fs.write",
    "fs.patch",
    "fs.list",
    "run.finish",
)


def _provider(*tool_ids: str):
    return runtime_provider(runtime_config(*(tool_ids or DEFAULT_TOOLS)))


def _runtime_config_file(tmp_path: Path, *tool_ids: str) -> Path:
    path = tmp_path / "runtime-config.json"
    path.write_text(json.dumps(runtime_config(*(tool_ids or ("run.finish",))).to_json()), encoding="utf-8")
    return path


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
    assert payload["schema_version"] == "monoid.event.v1"
    assert payload["type"] == "run.started"
    assert "kind" not in payload


def test_jsonl_and_status_sinks_flush_and_update(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    status_path = tmp_path / "status.json"
    bus = EventBus("run_sink", (JsonlEventSink(events_path), StatusJsonSink(status_path)))

    bus.emit("run.started", data={"workspace": "w", "mode": "propose", "model": "gpt-5.5"})
    assert events_path.read_text(encoding="utf-8").strip()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["state"] == "running"
    assert status["terminal"] is False

    bus.emit("run.finished", data={"status": "completed", "final_text": "done"})
    bus.close()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    assert status["terminal"] is True
    assert status["last_event_type"] == "run.finished"


def test_emit_after_close_is_a_noop(tmp_path: Path) -> None:
    # A background job thread can deliver a terminal event after the run closed the
    # recorder. That late emit must be a benign no-op, not a write to a closed file
    # handle (which surfaced as a flaky PytestUnhandledThreadExceptionWarning).
    events_path = tmp_path / "events.jsonl"
    bus = EventBus("run_late", (JsonlEventSink(events_path),))
    bus.emit("run.started", data={"workspace": "w", "mode": "propose", "model": "gpt-5.5"})
    bus.close()
    bytes_before = events_path.read_bytes()

    event = bus.emit("task.completed", data={"job_id": "late"})  # must not raise

    assert event.type == "task.completed"  # return contract preserved
    assert events_path.read_bytes() == bytes_before  # the closed sink is not written


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
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop(spec=spec, model_adapter=adapter, runtime_config_provider=_provider()).run_once(
        "Clean notes."
    )

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
    assert status["state"] == "completed"
    assert status["terminal"] is True
    assert status["proposal"]["path"] == "proposal.json"
    assert "+Clean summary" in result.diff_path.read_text(encoding="utf-8")
    proposal = json.loads(result.run_dir.joinpath("proposal.json").read_text(encoding="utf-8"))
    assert proposal["files"][0]["snapshot_path"] == "proposal/files/SUMMARY.md"
    assert proposal["proposal_hash"]
    assert status["manifest_path"] == "manifest.json"
    workspace_index = json.loads(result.run_dir.joinpath("workspace.index.json").read_text(encoding="utf-8"))
    assert workspace_index["schema_version"] == "monoid.workspace-index.v1"
    assert any(entry["path"] == "notes.md" for entry in workspace_index["entries"])
    projection = project_run_status(result.run_dir)
    assert projection["state"] == "completed"
    assert projection["terminal"] is True
    assert projection["proposal_hash"] == proposal["proposal_hash"]
    assert projection["changed_paths"] == ["SUMMARY.md"]


def test_otel_event_sink_emits_genai_span_tree(tmp_path: Path) -> None:
    """OtelEventSink turns the run event tree into invoke_agent / chat / execute_tool spans.
    A local in-memory exporter keeps this off any global provider or network."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from monoid_agent_kernel.observability.otel import OtelEventSink

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.md").write_text("rough notes\n", encoding="utf-8")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_write", {"path": "OUT.md", "content": "hi\n"}, "c1"),),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")
    AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=_provider("fs.write", "run.finish"),
        event_sinks=(OtelEventSink(tracer_provider=provider),),
    ).run_once("Write OUT.md.")

    names = [span.name for span in exporter.get_finished_spans()]
    assert "invoke_agent" in names
    assert any(n.startswith("chat") for n in names)
    assert any(n.startswith("execute_tool") for n in names)


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
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        permission_policy=PermissionPolicy(redact_patterns=(".env",)),
        runtime_config_provider=_provider(),
    ).run_once("Edit notes.")

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


def test_args_preview_redacts_content_fields_only_not_secret_named_keys() -> None:
    preview = args_preview(
        {
            "content": "file body",
            "old_text": "before",
            "api_key": "sk-secret",
            "authorization": "Bearer xyz",
            "note": "has PRIVATE KEY here",
        },
        PermissionPolicy(),
    )
    # (a) file-content fields stay redacted
    assert preview["content"] == {"redacted": True, "type": "str", "bytes": len(b"file body")}
    assert preview["old_text"]["redacted"] is True
    # (b) removed: secret-named keys and PRIVATE-KEY values are NOT scrubbed by the core
    assert preview["api_key"] == "sk-secret"
    assert preview["authorization"] == "Bearer xyz"
    assert preview["note"] == "has PRIVATE KEY here"


def test_example_redacting_event_sink_scrubs_secret_named_values() -> None:
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "examples" / "redacting_event_sink.py"
    module_spec = importlib.util.spec_from_file_location("example_redacting_sink", path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    inner = MemoryEventSink()
    bus = EventBus(run_id="r1", sinks=(module.RedactingEventSink(inner),))
    bus.emit(
        "tool.call.started",
        data={"args_preview": {"api_key": "sk-secret", "path": "a.txt"}},
    )
    bus.emit("tool.call.finished", data={"note": "-----BEGIN PRIVATE KEY-----"})

    preview = inner.events[0].data["args_preview"]
    assert preview["api_key"] == "[redacted]"
    assert preview["path"] == "a.txt"
    assert inner.events[1].data["note"] == "[redacted]"


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
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        permission_policy=PermissionPolicy(redact_patterns=(".env",)),
        runtime_config_provider=_provider(),
    ).run_once("Create env.")

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
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        permission_policy=PermissionPolicy(deny_patterns=(".env",)),
        runtime_config_provider=_provider(),
    ).run_once("Try tools.")

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
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        limits=RunLimits(max_steps=1, max_tool_calls=0),
    )

    result = AgentLoop(
        spec=spec, model_adapter=adapter, runtime_config_provider=_provider("fs.list", "run.finish")
    ).run_once("Loop.")

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

    monkeypatch.setattr("monoid_agent_kernel.cli.GatewayModelAdapter", FakeCliGatewayAdapter)
    monkeypatch.setenv("NAR_TEST_SINK_PATH", str(sink_output))
    runner, has_separate_stderr = _isolated_cli_runner()
    run_root = tmp_path / "runs"
    config_file = _runtime_config_file(tmp_path, "run.finish")

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
            "--runtime-config-file",
            str(config_file),
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
    assert status_payload["state"] == "completed"
    assert status_payload["terminal"] is True
    assert status_payload["last_event_type"] == "run.finished"


def test_cli_normal_mode_prints_run_identity_before_completion(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class FakeCliGatewayAdapter:
        def __init__(self, _config, **_kwargs):
            self._adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="done")])

        def next_turn(self, request):
            return self._adapter.next_turn(request)

    monkeypatch.setattr("monoid_agent_kernel.cli.GatewayModelAdapter", FakeCliGatewayAdapter)
    runner = CliRunner()
    config_file = _runtime_config_file(tmp_path, "run.finish")
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
            "--runtime-config-file",
            str(config_file),
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
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")
    result = AgentLoop(
        spec=spec, model_adapter=adapter, runtime_config_provider=_provider("fs.write", "run.finish")
    ).run_once("Write summary.")

    runner = CliRunner()
    summary = runner.invoke(main, ["proposal", str(result.run_dir), "--file", "SUMMARY.md", "--json"])

    assert summary.exit_code == 0
    payload = json.loads(summary.stdout)
    assert payload["encoding"] == "utf-8"
    assert payload["content"] == "Clean summary\n"
