from __future__ import annotations

import json
import inspect
from pathlib import Path

import pytest
from click.testing import CliRunner

from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.cli import _read_watch_batch, main
from monoid_agent_kernel.core._event_log import EventLogChanged, inspect_event_log_tail
from monoid_agent_kernel.core.events import AgentEvent, EventBus
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
    path.write_text(
        json.dumps(runtime_config(*(tool_ids or ("run.finish",))).to_json()), encoding="utf-8"
    )
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


def test_event_bus_normalizes_python_values_before_any_sink_sees_them() -> None:
    memory = MemoryEventSink()
    bus = EventBus("run_\ud800", (memory,))

    event = bus.emit(
        "metrics.updated",
        data={"\ud800": [float("nan"), float("inf"), -float("inf")]},
    )

    assert event.run_id == "run_�"
    assert event.data == {"�": [None, None, None]}
    assert memory.events == [event]


def test_event_bus_rechecks_authority_between_sink_callbacks() -> None:
    authority_lost = False
    first_events: list[AgentEvent] = []
    second = MemoryEventSink()

    class LosingSink:
        def emit(self, event: AgentEvent) -> None:
            nonlocal authority_lost
            first_events.append(event)
            authority_lost = True

        def close(self) -> None:
            return None

    def check_authority() -> None:
        if authority_lost:
            raise RuntimeError("writer authority lost")

    bus = EventBus(
        "run_fenced",
        (LosingSink(), second),
        check_authority=check_authority,
    )

    with pytest.raises(RuntimeError, match="writer authority lost"):
        bus.emit("run.started", data={"mode": "propose"})

    assert len(first_events) == 1
    assert second.events == []


def test_jsonl_sink_rejects_non_finite_values_if_an_ingress_is_bypassed(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    sink = JsonlEventSink(path)
    event = AgentEvent(
        schema_version="monoid.event.v1",
        event_id="evt_1",
        seq=1,
        run_id="run_1",
        timestamp="2026-07-30T00:00:00Z",
        type="metrics.updated",
        data={"value": float("nan")},
    )

    try:
        with pytest.raises(ValueError, match="Out of range float values"):
            sink.emit(event)
    finally:
        sink.close()

    assert path.read_bytes() == b""


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


def test_status_projection_withholds_uncommitted_terminal_event(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_partial_projection"
    run_dir.mkdir()
    events_path = run_dir / "events.jsonl"
    started = {"seq": 1, "type": "run.started", "data": {}}
    finished = {"seq": 2, "type": "run.finished", "data": {"status": "completed"}}
    events_path.write_text(
        json.dumps(started) + "\n" + json.dumps(finished),
        encoding="utf-8",
    )

    before_commit = project_run_status(run_dir)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    after_commit = project_run_status(run_dir)

    assert before_commit["state"] == "running"
    assert before_commit["terminal"] is False
    assert before_commit["last_event_seq"] == 1
    assert after_commit["state"] == "completed"
    assert after_commit["terminal"] is True
    assert after_commit["last_event_seq"] == 2


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


def test_event_bus_closes_each_sink_once_even_when_one_close_fails() -> None:
    class ClosingSink:
        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail
            self.close_count = 0

        def emit(self, event: AgentEvent) -> None:
            del event

        def close(self) -> None:
            self.close_count += 1
            if self.fail:
                raise RuntimeError("sink close failed")

    sinks = (ClosingSink(), ClosingSink(fail=True), ClosingSink())
    bus = EventBus("run-close-once", sinks)

    with pytest.raises(RuntimeError, match="sink close failed"):
        bus.close()
    bus.close()

    assert [sink.close_count for sink in sinks] == [1, 1, 1]


def test_event_bus_rechecks_authority_between_sink_close_callbacks() -> None:
    authority_lost = False

    class ClosingSink:
        def __init__(self, *, lose_authority: bool = False) -> None:
            self.lose_authority = lose_authority
            self.close_count = 0

        def emit(self, event: AgentEvent) -> None:
            del event

        def close(self) -> None:
            nonlocal authority_lost
            self.close_count += 1
            if self.lose_authority:
                authority_lost = True

    def check_authority() -> None:
        if authority_lost:
            raise RuntimeError("writer authority lost")

    sinks = (ClosingSink(lose_authority=True), ClosingSink())
    bus = EventBus("run-close-fenced", sinks, check_authority=check_authority)

    with pytest.raises(RuntimeError, match="writer authority lost"):
        bus.close()
    bus.close()

    assert [sink.close_count for sink in sinks] == [1, 0]


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

    result = AgentLoop(
        spec=spec, model_adapter=adapter, runtime_config_provider=_provider()
    ).run_once("Clean notes.")

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
    workspace_index = json.loads(
        result.run_dir.joinpath("workspace.index.json").read_text(encoding="utf-8")
    )
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
                tool_calls=(
                    fake_tool_call("fs_write", {"path": "OUT.md", "content": "hi\n"}, "c1"),
                ),
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


@pytest.mark.parametrize(
    ("stored_pattern", "encoding", "changed_path", "expected_path"),
    [
        ("!private", None, "!private", "[redacted-path]"),
        (r"\!private", None, "!private", "!private"),
        ("!private", "monoid.literal-bang.v1", "!private", "[redacted-path]"),
        ("secret//file", None, "secret/file", "[redacted-path]"),
        ("secret/./file", None, "secret/file", "[redacted-path]"),
        ("[", None, "[", "[redacted-path]"),
        ("public\u00a0", None, "public\u00a0", "[redacted-path]"),
    ],
)
def test_status_projection_reads_legacy_and_current_literal_bang_patterns(
    tmp_path: Path,
    stored_pattern: str,
    encoding: str | None,
    changed_path: str,
    expected_path: str,
) -> None:
    run_dir = tmp_path / "runs" / "run_legacy"
    run_dir.mkdir(parents=True)
    policy = {"deny_patterns": [], "redact_patterns": [stored_pattern]}
    if encoding is not None:
        policy["path_pattern_encoding"] = encoding
    run_dir.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "monoid.manifest.v1",
                "permission_policy": policy,
            }
        ),
        encoding="utf-8",
    )
    run_dir.joinpath("proposal.json").write_text(
        json.dumps({"run_id": "run_legacy", "changed_paths": [changed_path]}),
        encoding="utf-8",
    )

    projection = project_run_status(run_dir)

    assert projection["run_id"] == "run_legacy"
    assert projection["changed_paths"] == [expected_path]


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
        turns=[
            ModelTurn(
                response_id="r1", tool_calls=(fake_tool_call("fs_list", {"path": "."}, "c1"),)
            )
        ]
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
        json.loads(line) for line in stdout_text.splitlines() if line.strip().startswith("{")
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


def test_cli_watch_withholds_uncommitted_event_tail(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_watch_partial"
    run_dir.mkdir()
    events_path = run_dir / "events.jsonl"
    committed = {"seq": 1, "type": "run.started"}
    uncommitted = {"seq": 2, "type": "run.finished", "data": {"secret": "withheld"}}
    events_path.write_text(
        json.dumps(committed) + "\n" + json.dumps(uncommitted),
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["watch", str(run_dir), "--from-start", "--json"])

    assert result.exit_code == 0
    assert [json.loads(line) for line in result.stdout.splitlines()] == [committed]
    assert "withheld" not in result.stdout


def test_cli_watch_follows_repaired_partial_tail_by_committed_offset(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    first = b'{"seq":1,"type":"run.started"}\n'
    events_path.write_bytes(first + (b"x" * 1_000))
    tail = inspect_event_log_tail(events_path)
    identity = (tail.device, tail.inode)
    offset = tail.committed_end

    records, offset, drained = _read_watch_batch(events_path, offset, identity)
    assert records == []
    assert drained is True

    second = b'{ "seq" : 2, "type":"run.finished", "text":"\\u2603" }\n'
    events_path.write_bytes(first + second)
    records, offset, drained = _read_watch_batch(events_path, offset, identity)
    assert [record.raw_json for record in records] == [second.decode().removesuffix("\n")]
    assert drained is True

    third = b'{"seq":3,"type":"run.finished","padding":"' + (b"y" * 2_000) + b'"}\r\n'
    with events_path.open("ab") as handle:
        handle.write(third)
    records, offset, drained = _read_watch_batch(events_path, offset, identity)
    assert [record.raw_json for record in records] == [third.decode().removesuffix("\r\n")]
    assert drained is True
    assert _read_watch_batch(events_path, offset, identity)[0] == []


def test_cli_watch_batches_large_history_without_skip_or_duplicate(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    events_path.write_bytes(
        b"".join(f'{{"seq":{seq},"type":"run.started"}}\n'.encode() for seq in range(1, 1_001))
    )
    tail = inspect_event_log_tail(events_path)
    identity = (tail.device, tail.inode)
    offset = 0
    seen: list[int] = []
    drained = False

    while not drained:
        records, offset, drained = _read_watch_batch(events_path, offset, identity)
        assert len(records) <= 256
        seen.extend(record.seq for record in records)

    assert seen == list(range(1, 1_001))
    assert offset == tail.committed_end


def test_cli_watch_follow_starts_at_committed_end_before_partial_tail(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run_watch_follow_partial"
    run_dir.mkdir()
    events_path = run_dir / "events.jsonl"
    events_path.write_bytes(b"x" * 1_000)
    replacement = b'{ "seq" : 1, "type":"run.started" }\n'
    sleeps = 0

    class StopWatch(Exception):
        pass

    def advance_then_stop(_seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps == 1:
            events_path.write_bytes(replacement)
            return
        raise StopWatch

    monkeypatch.setattr("monoid_agent_kernel.cli.time.sleep", advance_then_stop)

    result = CliRunner().invoke(main, ["watch", str(run_dir), "--follow", "--json"])

    assert isinstance(result.exception, StopWatch)
    assert result.stdout.splitlines() == [replacement.decode().removesuffix("\n")]


def test_cli_watch_fails_closed_on_committed_truncation(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    events_path.write_bytes(b'{"seq":1}\n{"seq":2}\n')
    tail = inspect_event_log_tail(events_path)
    identity = (tail.device, tail.inode)
    offset = tail.committed_end
    events_path.write_bytes(b'{"seq":1}\n')

    with pytest.raises(EventLogChanged, match="truncated"):
        _read_watch_batch(events_path, offset, identity)


def test_cli_watch_fails_closed_on_file_replacement(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    events_path.write_bytes(b'{"seq":1}\n')
    tail = inspect_event_log_tail(events_path)
    identity = (tail.device, tail.inode)
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(b'{"seq":2}\n')
    replacement.replace(events_path)

    with pytest.raises(EventLogChanged, match="replaced"):
        _read_watch_batch(events_path, tail.committed_end, identity)


def test_cli_watch_reports_committed_corruption(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_watch_corrupt"
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_bytes(b'{"seq":\n')

    result = CliRunner().invoke(main, ["watch", str(run_dir), "--json"])

    assert result.exit_code != 0
    assert "committed event log record is not valid JSON" in result.output


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
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=_provider("fs.write", "run.finish"),
    ).run_once("Write summary.")

    runner = CliRunner()
    summary = runner.invoke(
        main, ["proposal", str(result.run_dir), "--file", "SUMMARY.md", "--json"]
    )

    assert summary.exit_code == 0
    payload = json.loads(summary.stdout)
    assert payload["encoding"] == "utf-8"
    assert payload["content"] == "Clean summary\n"


# --- corrupt event log ---------------------------------------------------------------------
#
# `monoid watch` catches `EventLogCorruption` and prints one line. The two *projection* readers
# did not, so `monoid status --json` printed a 4.8 KB traceback and Studio's chat catch-up died
# inside `do_GET`, which has no handler, taking the connection with it. Both now read leniently
# and publish the reason; these pin that the reason is published rather than swallowed, which is
# the half that matters -- a projection that stops at the damage without saying so reads as a
# complete, shorter run.


def _corrupt_event_log(run_dir: Path) -> Path:
    """A log whose *interior* is damaged: valid, garbage, valid.

    Interior rather than trailing on purpose. A truncated tail is the ordinary case -- a crash
    mid-append -- and the reader already withholds an uncommitted final record. What escaped was a
    committed record that does not decode, with good records after it.
    """
    events_path = run_dir / "events.jsonl"
    events_path.write_text(
        json.dumps({"seq": 1, "type": "run.started", "data": {}})
        + "\n"
        + "{not json at all\n"
        + json.dumps({"seq": 3, "type": "run.finished", "data": {"status": "completed"}})
        + "\n",
        encoding="utf-8",
    )
    return events_path


def test_status_projection_degrades_on_a_corrupt_event_log(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_corrupt"
    run_dir.mkdir()
    _corrupt_event_log(run_dir)

    projection = project_run_status(run_dir)

    assert "not valid JSON" in projection["event_log_error"]
    # Everything before the damage survives...
    assert projection["last_event_seq"] == 1
    # ...and nothing after it does. This is exactly why the flag has to be published: the file
    # says the run completed and the projection cannot see it, so a poller reading `state` alone
    # waits forever on a run that already finished.
    assert projection["state"] == "running"
    assert projection["terminal"] is False


def test_status_projection_reports_no_error_on_a_clean_log(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_clean"
    run_dir.mkdir()
    run_dir.joinpath("events.jsonl").write_text(
        json.dumps({"seq": 1, "type": "run.started", "data": {}}) + "\n",
        encoding="utf-8",
    )

    projection = project_run_status(run_dir)

    assert projection["event_log_error"] == ""
    assert projection["state"] == "running"


def test_status_projection_reports_no_error_when_there_is_no_log(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_empty"
    run_dir.mkdir()

    assert project_run_status(run_dir)["event_log_error"] == ""


def _event(event_type: str, data: dict[str, object]) -> AgentEvent:
    return AgentEvent(
        schema_version="monoid.event.v1",
        event_id="evt_1",
        seq=1,
        run_id="run-1",
        timestamp="2026-08-03T00:00:00Z",
        type=event_type,
        data=data,
    )


def _write_events(run_dir: Path, *events: dict[str, object]) -> None:
    run_dir.joinpath("events.jsonl").write_text(
        "".join(
            json.dumps({"seq": index + 1, **event}) + "\n" for index, event in enumerate(events)
        ),
        encoding="utf-8",
    )


def test_the_offline_projection_sees_the_user_input_park_not_only_the_job_wait(
    tmp_path: Path,
) -> None:
    """Two parks travel this stream and this reader handled one of them.

    `run.waiting` (background jobs) was projected; `run.awaiting_input` — the park a hosted task
    or a multi-turn session sits in — was not, so `monoid status` reported a parked run as still
    running. The live sink (`StatusJsonSink`) had always handled both.
    """
    run_dir = tmp_path / "run_awaiting"
    run_dir.mkdir()
    _write_events(
        run_dir,
        {"type": "run.started", "data": {}},
        {"type": "run.awaiting_input", "data": {"reason": "task", "task_ids": ["t-1"]}},
    )

    projection = project_run_status(run_dir)

    assert projection["state"] == "awaiting_input"
    assert projection["terminal"] is False


# The full classification `turn.failed` emits (loop.py), minus the metering-only
# `provider_usage`. One literal shared by the park/heal/unpark tests below so they cannot
# drift onto different subsets of the same fact family.
_TURN_FAILED_CLASSIFICATION = {
    "error": "model rejected the key",
    "error_code": "model_error",
    "provider_error_code": "insufficient_quota",
    "http_status": 422,
    "retryable": False,
    "config_recoverable": True,
    "provider_retried": True,
}


def test_the_offline_projection_reports_the_classification_a_parked_turn_carries(
    tmp_path: Path,
) -> None:
    """`turn.failed` carries the whole taxonomy and no status reader consumed any of it.

    State is deliberately untouched by this branch: `turn.failed` is not terminal, and the park
    that follows it names the state. What the event uniquely carries is *why* — and
    `config_recoverable` alone cannot separate an `insufficient_quota` (fix config) from a
    `rate_limit` (wait), so the reader carries the full set, not a fragment.
    """
    run_dir = tmp_path / "run_turn_failed"
    run_dir.mkdir()
    _write_events(
        run_dir,
        {"type": "run.started", "data": {}},
        {"type": "turn.failed", "data": dict(_TURN_FAILED_CLASSIFICATION)},
        {"type": "run.awaiting_input", "data": {"reason": "turn_failed"}},
    )

    projection = project_run_status(run_dir)

    for key, value in _TURN_FAILED_CLASSIFICATION.items():
        assert projection[key] == value, key
    # The park that follows still owns the state.
    assert projection["state"] == "awaiting_input"


def test_a_model_turn_starting_clears_both_parks_on_both_readers(tmp_path: Path) -> None:
    """The clear was bound on one park and one reader each; it is one rule on two readers now.

    The sink cleared `AWAITING_INPUT` only, so a run that had parked on background jobs read as
    parked while the turn that unparked it was already running; the offline projection cleared
    neither and depended entirely on `run.resumed`, which nothing emits after a user-input park.
    """
    run_dir = tmp_path / "run_unpark"
    run_dir.mkdir()
    _write_events(
        run_dir,
        {"type": "run.started", "data": {}},
        {"type": "run.waiting", "data": {"jobs": []}},
        {"type": "model.turn.started", "data": {"step": 2}},
    )

    projection = project_run_status(run_dir)

    assert projection["state"] == "running"
    assert projection["waiting_for_background_jobs"] is False

    # The live twin, driven through the same two events.
    sink = StatusJsonSink(tmp_path / "status.json")
    for state in ("awaiting_tasks", "awaiting_input"):
        sink.state["state"] = state
        sink.emit(
            _event("model.turn.started", {"step": 2})
        )
        assert sink.state["state"] == "running", state
        assert sink.state["terminal"] is False


def test_a_settled_recovery_clears_the_interruption_cause_on_both_readers(
    tmp_path: Path,
) -> None:
    events = (
        {
            "type": "turn.interrupted",
            "data": {"reason": "user_stop", "interruption_cause": "user_cancel"},
        },
        {
            "type": "turn.settled",
            "data": {"status": "completed", "interruption_cause": ""},
        },
    )
    run_dir = tmp_path / "run_recovered_settle"
    run_dir.mkdir()
    _write_events(run_dir, *events)

    assert project_run_status(run_dir)["interruption_cause"] is None

    sink = StatusJsonSink(tmp_path / "status.json")
    for payload in events:
        sink.emit(_event(payload["type"], dict(payload["data"])))
    assert "interruption_cause" not in sink.state


def test_a_legacy_interruption_clears_an_older_typed_cause_on_both_readers(
    tmp_path: Path,
) -> None:
    legacy_payloads = (
        {"reason": "user_stop"},
        {"reason": "user_stop", "interruption_cause": ""},
        {"reason": "user_stop", "interruption_cause": 1},
        {"reason": "user_stop", "interruption_cause": "cancel"},
    )
    for index, legacy_payload in enumerate(legacy_payloads):
        events = (
            {
                "type": "turn.interrupted",
                "data": {"reason": "drain", "interruption_cause": "graceful_drain"},
            },
            {"type": "turn.interrupted", "data": legacy_payload},
        )
        run_dir = tmp_path / f"run_legacy_interrupt_{index}"
        run_dir.mkdir()
        _write_events(run_dir, *events)

        assert project_run_status(run_dir)["interruption_cause"] is None

        sink = StatusJsonSink(tmp_path / f"status_{index}.json")
        for payload in events:
            sink.emit(_event(payload["type"], dict(payload["data"])))
        assert "interruption_cause" not in sink.state


def test_a_pause_supersedes_an_old_interruption_cause_on_both_readers(
    tmp_path: Path,
) -> None:
    events = (
        {
            "type": "turn.interrupted",
            "data": {"reason": "user_stop", "interruption_cause": "user_cancel"},
        },
        {"type": "turn.paused", "data": {"reason": "user_pause"}},
    )
    run_dir = tmp_path / "run_reparked_as_pause"
    run_dir.mkdir()
    _write_events(run_dir, *events)

    assert project_run_status(run_dir)["interruption_cause"] is None

    sink = StatusJsonSink(tmp_path / "status.json")
    for payload in events:
        sink.emit(_event(payload["type"], dict(payload["data"])))
    assert "interruption_cause" not in sink.state


def test_the_status_sink_records_the_classification_of_a_recoverable_turn_failure(
    tmp_path: Path,
) -> None:
    """The sink's half of the same convergence: `turn.failed` was unread here too."""

    sink = StatusJsonSink(tmp_path / "status.json")
    sink.emit(_event("turn.failed", dict(_TURN_FAILED_CLASSIFICATION)))

    for key, value in _TURN_FAILED_CLASSIFICATION.items():
        assert sink.state[key] == value, key
    # Not a lifecycle change: the park that follows owns the state.
    assert "state" not in sink.state


_CLASSIFICATION_KEYS = (
    "provider_error_code",
    "http_status",
    "retryable",
    "config_recoverable",
    "provider_retried",
)


def _assert_no_stale_failure(projection: dict) -> None:
    assert projection["error"] == ""
    assert projection["error_code"] == ""
    assert projection["provider_error_code"] == ""
    assert projection["http_status"] is None
    assert projection["retryable"] is False
    assert projection["config_recoverable"] is False
    assert projection["provider_retried"] is False


def test_a_clean_terminal_settle_heals_the_stale_classification_on_both_readers(
    tmp_path: Path,
) -> None:
    """The empirically traced or-fallback staleness: a completed run kept a dead turn's error.

    `run.started -> turn.failed -> run.awaiting_input -> model.turn.started ->
    run.finished{completed, error:"", error_code:""}` used to project
    `error="model rejected the key", error_code="model_error"`, because the terminal branches
    or-ed event data over the stale value and `run.finished` never touched `error` at all.
    Terminal branches ASSIGN now, on the offline projection and on its live sink twin.
    """
    run_dir = tmp_path / "run_or_fallback"
    run_dir.mkdir()
    events = (
        {"type": "run.started", "data": {}},
        {"type": "turn.failed", "data": dict(_TURN_FAILED_CLASSIFICATION)},
        {"type": "run.awaiting_input", "data": {"reason": "turn_failed"}},
        {"type": "model.turn.started", "data": {"step": 2}},
        {"type": "run.finished", "data": {"status": "completed", "error": "", "error_code": ""}},
    )
    _write_events(run_dir, *events)

    projection = project_run_status(run_dir)

    assert projection["state"] == "completed"
    assert projection["terminal"] is True
    _assert_no_stale_failure(projection)

    # The live twin, fed the identical stream.
    sink = StatusJsonSink(tmp_path / "status.json")
    for payload in events:
        sink.emit(_event(payload["type"], dict(payload["data"])))
    assert sink.state["error"] == ""
    assert sink.state["error_code"] == ""
    for key in _CLASSIFICATION_KEYS:
        assert key not in sink.state, key


def test_the_terminal_heal_does_not_wait_for_an_unpark(tmp_path: Path) -> None:
    """A completed run must not keep `retryable`/`config_recoverable` even with no retry turn.

    The sequence above also rides the unpark clear (`model.turn.started`); this one goes
    straight from the park to the clean terminal, so only the terminal heal can clear it.
    """
    run_dir = tmp_path / "run_terminal_heal"
    run_dir.mkdir()
    events = (
        {"type": "run.started", "data": {}},
        {"type": "turn.failed", "data": dict(_TURN_FAILED_CLASSIFICATION)},
        {"type": "run.finished", "data": {"status": "completed", "error": "", "error_code": ""}},
    )
    _write_events(run_dir, *events)

    projection = project_run_status(run_dir)

    assert projection["state"] == "completed"
    _assert_no_stale_failure(projection)

    sink = StatusJsonSink(tmp_path / "status.json")
    for payload in events:
        sink.emit(_event(payload["type"], dict(payload["data"])))
    assert sink.state["error"] == ""
    for key in _CLASSIFICATION_KEYS:
        assert key not in sink.state, key


def test_a_failed_run_keeps_the_classification_its_terminal_event_carries(
    tmp_path: Path,
) -> None:
    """The heal must not overshoot: `run.failed` owns the terminal classification.

    `run.finished{status:"failed"}` follows `run.failed` on the same stream, and popping the
    classification there would undo the terminal record one event after it was written.
    `provider_retried` is the exception on purpose — it is a per-call fact the terminal
    vocabulary deliberately drops (see test_carriage_conformance's promotion pin).
    """
    run_dir = tmp_path / "run_failed_keeps"
    run_dir.mkdir()
    events = (
        {"type": "run.started", "data": {}},
        {"type": "turn.failed", "data": dict(_TURN_FAILED_CLASSIFICATION)},
        {
            "type": "run.failed",
            "data": {
                "error": "model rejected the key",
                "error_code": "model_error",
                "type": "ModelAdapterError",
                "provider_error_code": "insufficient_quota",
                "http_status": 422,
                "retryable": False,
                "config_recoverable": True,
            },
        },
        {
            "type": "run.finished",
            "data": {
                "status": "failed",
                "error": "model rejected the key",
                "error_code": "model_error",
            },
        },
    )
    _write_events(run_dir, *events)

    projection = project_run_status(run_dir)

    assert projection["state"] == "failed"
    assert projection["error"] == "model rejected the key"
    assert projection["error_code"] == "model_error"
    assert projection["provider_error_code"] == "insufficient_quota"
    assert projection["http_status"] == 422
    assert projection["config_recoverable"] is True
    assert projection["provider_retried"] is False

    sink = StatusJsonSink(tmp_path / "status.json")
    for payload in events:
        sink.emit(_event(payload["type"], dict(payload["data"])))
    assert sink.state["error"] == "model rejected the key"
    assert sink.state["error_type"] == "ModelAdapterError"
    assert sink.state["provider_error_code"] == "insufficient_quota"
    assert sink.state["http_status"] == 422
    assert sink.state["config_recoverable"] is True
    assert "provider_retried" not in sink.state


def test_a_model_turn_starting_clears_the_parked_failure_on_both_readers(
    tmp_path: Path,
) -> None:
    """`turn.failed -> model.turn.started` (retry/recovery) must not keep the dead turn's error.

    While PARKED the classification must remain — that is the point of carrying it — so the
    clear rides the unpark, not the park. The retry path never passes through a parked state
    (the driver re-pumps straight from `turn_failed`), so the clear cannot hide behind the
    parked-state guard.
    """
    run_dir = tmp_path / "run_unpark_clear"
    run_dir.mkdir()
    events = (
        {"type": "run.started", "data": {}},
        {"type": "turn.failed", "data": dict(_TURN_FAILED_CLASSIFICATION)},
        {"type": "model.turn.started", "data": {"step": 2}},
    )
    _write_events(run_dir, *events)

    projection = project_run_status(run_dir)

    assert projection["state"] == "running"
    _assert_no_stale_failure(projection)

    sink = StatusJsonSink(tmp_path / "status.json")
    for payload in events:
        sink.emit(_event(payload["type"], dict(payload["data"])))
    assert "error" not in sink.state
    assert "error_code" not in sink.state
    for key in _CLASSIFICATION_KEYS:
        assert key not in sink.state, key


def test_a_paused_run_is_visible_on_both_durable_readers(tmp_path: Path) -> None:
    """While paused, status.json and the offline projection said state="running".

    The pause park emits two events; the session-lane `session.state.changed{state:"paused"}`
    is the carrier here because it names the lifecycle state these readers project
    (`turn.paused` stays a turn-lane cause event no projection consumes). A model turn
    starting is the unpark, exactly as for the input/task parks.
    """
    run_dir = tmp_path / "run_paused"
    run_dir.mkdir()
    paused_events = (
        {"type": "run.started", "data": {}},
        {"type": "turn.paused", "data": {"reason": "user_pause"}},
        {
            "type": "session.state.changed",
            "data": {"state": "paused", "from": "running", "reason": "pause_requested"},
        },
    )
    _write_events(run_dir, *paused_events)

    projection = project_run_status(run_dir)
    assert projection["state"] == "paused"
    assert projection["terminal"] is False

    # ...and the resumed pump unparks it on the same event the other parks use.
    _write_events(
        run_dir,
        *paused_events,
        {"type": "model.turn.started", "data": {"step": 2}},
    )
    assert project_run_status(run_dir)["state"] == "running"

    sink = StatusJsonSink(tmp_path / "status.json")
    for payload in paused_events:
        sink.emit(_event(payload["type"], dict(payload["data"])))
    assert sink.state["state"] == "paused"
    assert sink.state["terminal"] is False
    sink.emit(_event("model.turn.started", {"step": 2}))
    assert sink.state["state"] == "running"
    assert sink.state["terminal"] is False


def test_cli_status_json_prints_the_projection_then_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_cli_corrupt"
    run_dir.mkdir()
    _corrupt_event_log(run_dir)

    runner, split_stderr = _isolated_cli_runner()
    result = runner.invoke(main, ["status", str(run_dir), "--json"])

    # Non-zero, because `state` is not trustworthy and a script must not read it as current.
    assert result.exit_code != 0
    # The partial answer is still printed: the caller asked for it and it is all there is.
    payload = json.loads(result.stdout.splitlines()[0])
    assert payload["state"] == "running"
    assert "not valid JSON" in payload["event_log_error"]
    # One clean line, not a traceback.
    stderr = result.stderr if split_stderr else result.output
    assert "Traceback" not in stderr
    assert "not valid JSON" in stderr


def test_cli_status_human_output_also_fails_on_a_corrupt_log(tmp_path: Path) -> None:
    """The `--json` branch returns early, so the human branch is a separate exit path -- the
    shape that shipped a rule bound to one of two siblings all through this release."""
    run_dir = tmp_path / "run_cli_corrupt_human"
    run_dir.mkdir()
    _corrupt_event_log(run_dir)

    runner, split_stderr = _isolated_cli_runner()
    result = runner.invoke(main, ["status", str(run_dir)])

    assert result.exit_code != 0
    assert "run_id:" in result.stdout
    stderr = result.stderr if split_stderr else result.output
    assert "Traceback" not in stderr
    assert "not valid JSON" in stderr


def test_cli_status_succeeds_on_a_clean_log(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_cli_clean"
    run_dir.mkdir()
    run_dir.joinpath("events.jsonl").write_text(
        json.dumps({"seq": 1, "type": "run.started", "data": {}}) + "\n",
        encoding="utf-8",
    )

    runner, _ = _isolated_cli_runner()
    result = runner.invoke(main, ["status", str(run_dir), "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["event_log_error"] == ""


def test_studio_chat_catch_up_degrades_on_a_corrupt_event_log(tmp_path: Path) -> None:
    from monoid_agent_kernel.reference.studio.chat_projection import ChatProjection

    run_dir = tmp_path / "run_studio_corrupt"
    run_dir.mkdir()
    run_dir.joinpath("events.jsonl").write_text(
        json.dumps({"seq": 1, "type": "run.failed", "data": {"error": "before the damage"}})
        + "\n"
        + "{not json at all\n"
        + json.dumps({"seq": 3, "type": "run.failed", "data": {"error": "after the damage"}})
        + "\n",
        encoding="utf-8",
    )

    body = ChatProjection(run_dir).catch_up("run-corrupt")

    contents = [message["content"] for message in body["messages"]]
    assert "before the damage" in contents
    assert "after the damage" not in contents
    assert "not valid JSON" in body["event_log_error"]


def test_studio_chat_catch_up_reports_no_error_on_a_clean_log(tmp_path: Path) -> None:
    from monoid_agent_kernel.reference.studio.chat_projection import ChatProjection

    run_dir = tmp_path / "run_studio_clean"
    run_dir.mkdir()
    run_dir.joinpath("events.jsonl").write_text(
        json.dumps({"seq": 1, "type": "run.failed", "data": {"error": "only message"}}) + "\n",
        encoding="utf-8",
    )

    body = ChatProjection(run_dir).catch_up("run-clean")

    assert [message["content"] for message in body["messages"]] == ["only message"]
    assert body["event_log_error"] == ""


def test_an_uncommitted_tail_is_not_reported_as_corruption(tmp_path: Path) -> None:
    """The distinction the flag lives or dies on.

    A run that crashed mid-append leaves a final record with no newline. That is the *ordinary*
    case, the reader already withholds it, and reporting it as damage would put a scary field on
    a large fraction of interrupted runs -- which is how a warning stops being read.
    """
    from monoid_agent_kernel.core._event_log import read_committed_event_payloads

    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        json.dumps({"seq": 1, "type": "run.started", "data": {}}) + "\n" + '{"seq": 2, "type": "ru',
        encoding="utf-8",
    )

    read = read_committed_event_payloads(events_path)

    assert [payload["seq"] for payload in read.payloads] == [1]
    assert read.corruption == ""


def test_a_missing_or_empty_event_log_is_not_corruption(tmp_path: Path) -> None:
    from monoid_agent_kernel.core._event_log import read_committed_event_payloads

    absent = read_committed_event_payloads(tmp_path / "nope.jsonl")
    assert absent.payloads == [] and absent.corruption == ""

    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("", encoding="utf-8")
    empty = read_committed_event_payloads(empty_path)
    assert empty.payloads == [] and empty.corruption == ""


@pytest.mark.parametrize("sequence", [(1, 1), (2, 1)])
def test_lenient_event_read_stops_at_a_non_increasing_sequence(
    tmp_path: Path, sequence: tuple[int, int]
) -> None:
    from monoid_agent_kernel.core._event_log import read_committed_event_payloads

    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        "".join(
            json.dumps({"seq": seq, "type": "run.started", "data": {}}) + "\n" for seq in sequence
        ),
        encoding="utf-8",
    )

    read = read_committed_event_payloads(events_path)

    assert [payload["seq"] for payload in read.payloads] == [sequence[0]]
    assert "sequence is not increasing" in read.corruption


def test_status_and_studio_degrade_when_json_decoder_hits_recursion_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from monoid_agent_kernel.core import _event_log
    from monoid_agent_kernel.reference.studio.chat_projection import ChatProjection

    run_dir = tmp_path / "run_deep_event"
    run_dir.mkdir()
    prefix = json.dumps({"seq": 1, "type": "run.failed", "data": {"error": "safe prefix"}})
    deeply_nested = '{"seq":2,"type":"run.failed","data":{"nested":' + "[[0]]" + "}}"
    run_dir.joinpath("events.jsonl").write_text(
        prefix + "\n" + deeply_nested + "\n", encoding="utf-8"
    )

    real_loads = json.loads

    def recursion_error_for_nested_event(
        payload: str | bytes | bytearray,
        *args: object,
        **kwargs: object,
    ) -> object:
        if '"nested"' in str(payload):
            raise RecursionError("simulated JSON decoder recursion limit")
        return real_loads(payload, *args, **kwargs)

    monkeypatch.setattr(_event_log, "loads_json_ingress", recursion_error_for_nested_event)

    status = project_run_status(run_dir)
    transcript = ChatProjection(run_dir).catch_up("run-deep-event")

    assert "not valid JSON" in status["event_log_error"]
    assert status["state"] == "failed"
    assert [message["content"] for message in transcript["messages"]] == ["safe prefix"]
    assert "not valid JSON" in transcript["event_log_error"]


def test_degraded_status_can_mix_snapshot_and_event_prefix_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_mixed_projection"
    run_dir.mkdir()
    run_dir.joinpath("status.json").write_text(
        json.dumps(
            {
                "run_id": "run-mixed",
                "state": "completed",
                "terminal": True,
                "last_event_seq": 3,
                "last_event_type": "run.finished",
            }
        ),
        encoding="utf-8",
    )
    _corrupt_event_log(run_dir)

    projection = project_run_status(run_dir)

    assert "not valid JSON" in projection["event_log_error"]
    # Snapshot metadata can be newer than the valid event prefix. The error flag marks the whole
    # mixed projection as diagnostic instead of promising one coherent point in time.
    assert projection["last_event_seq"] == 3
    assert projection["last_event_type"] == "run.finished"
    assert projection["state"] == "running"
