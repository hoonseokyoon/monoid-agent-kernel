from __future__ import annotations

import json
from pathlib import Path

from conftest import runtime_config, runtime_provider, tool_binding

from native_agent_runner.core.cancellation import CancellationToken
from native_agent_runner.core.schemas import validate_run_dir
from native_agent_runner.core.spec import AgentRunSpec, RunLimits
from native_agent_runner.core.tool_surface import ToolQuota
from native_agent_runner.errors import ModelAdapterError
from native_agent_runner.loop import AgentLoop, _recoverable_turn_error
from native_agent_runner.providers.base import ModelTurn, ReasoningDelta, TextDelta, TurnComplete
from native_agent_runner.providers.fake import (
    FakeModelAdapter,
    FakeStreamingModelAdapter,
    fake_tool_call,
)
from native_agent_runner.recorder import MemoryEventSink
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


def test_workspace_delta_cap_settles_run_as_limited(tmp_path: Path) -> None:
    # A workspace delta that outgrows the cap settles the run ``limited`` at the next
    # turn's start, before the over-cap delta is persisted into a checkpoint.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_write", {"path": "big.txt", "content": "x" * 50}, "c1"),),
            ),
            ModelTurn(
                response_id="r2",
                tool_calls=(fake_tool_call("run_finish", {"summary": "done"}, "c2"),),
            ),
        ]
    )
    spec = AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        limits=RunLimits(max_delta_file_bytes=10),
    )

    result = AgentLoop(
        spec=spec, model_adapter=adapter, runtime_config_provider=_provider("fs.write", "run.finish")
    ).run_once("write a big file")

    assert result.status == "limited"
    assert result.error_code == "workspace_delta_file_bytes_exceeded"
    assert len(adapter.requests) == 1  # turn 2 is never sent (settled at its start)


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


# --- recoverable turn errors -----------------------------------------------------------


class _ScriptedAdapter:
    """Drives a script of turns/exceptions: a ModelTurn is returned, a BaseException is raised."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.requests: list = []

    def next_turn(self, request):  # noqa: ANN001
        self.requests.append(request)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _loop_with(tmp_path: Path, adapter, *tool_ids: str) -> tuple[AgentLoop, MemoryEventSink, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    run_root = tmp_path / "runs"
    sink = MemoryEventSink()
    spec = AgentRunSpec(workspace_root=workspace, run_root=run_root)
    loop = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=_provider(*(tool_ids or ("run.finish",))),
        event_sinks=(sink,),
    )
    return loop, sink, run_root


def test_metrics_surface_reasoning_tokens_when_reported(tmp_path: Path) -> None:
    # R10: reasoning tokens reach the metrics.updated event when the adapter reports them, so the
    # studio meter can show the reasoning share.
    adapter = FakeModelAdapter(
        turns=[ModelTurn(final_text="done", usage={"input_tokens": 5, "output_tokens": 9, "total_tokens": 14, "reasoning_tokens": 7})]
    )
    loop, sink, _ = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        loop.run_until_suspended("hi")
        metrics = [e for e in sink.events if e.type == "metrics.updated"]
        assert metrics and metrics[-1].data["reasoning_tokens"] == 7
    finally:
        loop.close()


def test_metrics_omit_reasoning_tokens_when_absent(tmp_path: Path) -> None:
    # A non-reasoning model reports none → the key is omitted (no "🧠0" noise in the meter).
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done", usage={"input_tokens": 5, "output_tokens": 9, "total_tokens": 14})])
    loop, sink, _ = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        loop.run_until_suspended("hi")
        metrics = [e for e in sink.events if e.type == "metrics.updated"]
        assert metrics and "reasoning_tokens" not in metrics[-1].data
    finally:
        loop.close()


def test_recoverable_turn_error_classifier() -> None:
    assert _recoverable_turn_error(ModelAdapterError("x", http_status=400))
    assert _recoverable_turn_error(ModelAdapterError("x", http_status=401))
    assert _recoverable_turn_error(ModelAdapterError("x", http_status=429, retryable=True))
    assert _recoverable_turn_error(ModelAdapterError("x", retryable=True))  # any status
    assert not _recoverable_turn_error(ModelAdapterError("x", http_status=500))
    assert not _recoverable_turn_error(RuntimeError("x"))


def test_turn_failed_suspension_is_non_terminal(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter([ModelAdapterError("bad effort", http_status=400, error_code="model_error")])
    loop, sink, run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        susp = loop.run_until_suspended("hello")
        assert susp.reason == "turn_failed"
        assert susp.retryable is False
        assert susp.http_status == 400
        assert loop._session is not None and loop._session.terminal is False
        types = [e.type for e in sink.events]
        assert "turn.failed" in types
        assert "run.failed" not in types
        assert list(run_root.rglob("failure.json")) == []  # not a terminal failure
    finally:
        loop.close()


def test_turn_failed_is_idempotent_on_reentry(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        [
            ModelAdapterError("transient", http_status=503, retryable=True),
            ModelTurn(response_id="r2", final_text="recovered"),
        ]
    )
    loop, _sink, _run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        first = loop.run_until_suspended("hi")
        assert first.reason == "turn_failed"
        assert loop._session is not None and loop._session.state.pending_observations == ()
        second = loop.run_until_suspended(None)  # re-issue the same turn
        assert second.reason == "settled"
        # The re-attempt sent the identical message log — no duplicated user message.
        assert adapter.requests[0].messages == adapter.requests[1].messages
    finally:
        loop.close()


def test_non_recoverable_model_error_is_terminal(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        [ModelAdapterError("server boom", http_status=500, provider_error_code="server_error")]
    )
    loop, sink, run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        susp = loop.run_until_suspended("hi")
        assert susp.reason == "terminal"
        assert susp.status == "failed"
        assert loop._session is not None and loop._session.terminal is True
        failed = [e for e in sink.events if e.type == "run.failed"]
        assert failed, "run.failed event emitted"
        # The public failure event carries the provider detail (not just a generic message), so
        # logs and the UI can see the real cause.
        assert failed[0].data["provider_error_code"] == "server_error"
        assert failed[0].data["http_status"] == 500
        assert list(run_root.rglob("failure.json"))
    finally:
        loop.close()


def test_generic_model_error_is_terminal(tmp_path: Path) -> None:
    # A raw exception is wrapped into a non-retryable ModelAdapterError -> still terminal.
    adapter = _ScriptedAdapter([RuntimeError("kaboom")])
    loop, sink, _run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        susp = loop.run_until_suspended("hi")
        assert susp.reason == "terminal" and susp.status == "failed"
        assert loop._session is not None and loop._session.terminal is True
    finally:
        loop.close()


def test_turn_failed_after_tool_round_clears_observations(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        [
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("fs_write", {"path": "a.md", "content": "x"}, "c1"),)),
            ModelAdapterError("transient", http_status=503, retryable=True),
            ModelTurn(response_id="r3", final_text="done"),
        ]
    )
    loop, _sink, _run_root = _loop_with(tmp_path, adapter, "fs.write", "run.finish")
    loop.open()
    try:
        first = loop.run_until_suspended("write a.md")  # tool runs, then turn 2 model call fails
        assert first.reason == "turn_failed"
        assert loop._session is not None and loop._session.state.pending_observations == ()
        second = loop.run_until_suspended(None)
        assert second.reason == "settled"
        # The post-tool message log is re-sent verbatim — the function_call_output isn't duplicated.
        assert adapter.requests[1].messages == adapter.requests[2].messages
    finally:
        loop.close()


def test_fail_recoverable_promotes_to_terminal(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter([ModelAdapterError("bad", http_status=400)])
    loop, sink, run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        assert loop.run_until_suspended("hi").reason == "turn_failed"
        loop.fail_recoverable("gave up after retries", error_code="model_error")
        assert loop._session is not None and loop._session.terminal is True
        assert "run.failed" in [e.type for e in sink.events]
        assert list(run_root.rglob("failure.json"))
    finally:
        loop.close()


def test_promotion_preserves_provider_details_from_turn_failed(tmp_path: Path) -> None:
    # A recoverable provider failure records provider detail on the turn.failed; promoting it with
    # fail_recoverable() (a fresh error with no provider fields) must NOT blank that detail.
    adapter = _ScriptedAdapter(
        [ModelAdapterError("bad request", http_status=400, provider_error_code="invalid_request_error")]
    )
    loop, sink, run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        assert loop.run_until_suspended("hi").reason == "turn_failed"
        loop.fail_recoverable("gave up after retries", error_code="model_error")
        failed = [e for e in sink.events if e.type == "run.failed"]
        assert failed, "run.failed emitted"
        assert failed[0].data["provider_error_code"] == "invalid_request_error"
        assert failed[0].data["http_status"] == 400
    finally:
        loop.close()


def test_fresh_terminal_failure_clears_stale_provider_details(tmp_path: Path) -> None:
    # A recoverable turn.failed records provider detail; if the re-issued turn then fails terminally
    # for an UNRELATED reason, run.failed must reflect that new cause, not the stale detail.
    adapter = _ScriptedAdapter(
        [
            ModelAdapterError("rate limited", http_status=429, provider_error_code="rate_limit_exceeded", retryable=True),
            ModelAdapterError("server boom", http_status=500),  # terminal, no provider code
        ]
    )
    loop, sink, _run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        assert loop.run_until_suspended("hi").reason == "turn_failed"
        assert loop.run_until_suspended(None).reason == "terminal"
        failed = [e for e in sink.events if e.type == "run.failed"]
        assert failed, "run.failed emitted"
        assert failed[0].data["http_status"] == 500
        assert failed[0].data["provider_error_code"] == ""  # not the stale rate_limit_exceeded
    finally:
        loop.close()


# --- DX-9: turn-level interrupt (a "stop" that keeps the session alive) -----------------


class _SelfInterruptingAdapter:
    """First turn calls a tool and flips the loop's interrupt flag, so the next step boundary
    (before the second model call) trips — simulating a user "stop" mid-turn. A later call
    settles, proving the session survived the stop."""

    def __init__(self) -> None:
        self.loop = None
        self.calls = 0

    def next_turn(self, request):  # noqa: ANN001
        self.calls += 1
        if self.calls == 1:
            self.loop.interrupt_turn()
            return ModelTurn(response_id="r1", tool_calls=(fake_tool_call("fs_list", {"path": "."}, "c1"),))
        return ModelTurn(response_id="r2", final_text="resumed ok")


def test_interrupt_parks_turn_without_terminating(tmp_path: Path) -> None:
    adapter = _SelfInterruptingAdapter()
    loop, sink, run_root = _loop_with(tmp_path, adapter, "fs.list", "run.finish")
    adapter.loop = loop
    loop.open()
    try:
        susp = loop.run_until_suspended("go")
        assert susp.reason == "interrupted"
        assert loop._session is not None and loop._session.terminal is False
        assert adapter.calls == 1  # the second model call never ran — the turn was stopped
        types = [e.type for e in sink.events]
        assert "turn.interrupted" in types
        assert "run.failed" not in types
        assert list(run_root.rglob("failure.json")) == []  # not a terminal failure
        # The session is alive: re-issuing the turn (the interrupt flag is consumed) settles.
        again = loop.run_until_suspended(None)
        assert again.reason == "settled"
        assert again.final_text == "resumed ok"
        assert adapter.calls == 2
    finally:
        loop.close()


def test_stale_interrupt_does_not_kill_next_turn(tmp_path: Path) -> None:
    # interrupt_turn() with no turn in flight is a no-op: the next submit clears the flag.
    adapter = _ScriptedAdapter([ModelTurn(response_id="r1", final_text="ok")])
    loop, _sink, _run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        loop.interrupt_turn()  # stale stop
        susp = loop.run_until_suspended("hi")
        assert susp.reason == "settled"
        assert susp.final_text == "ok"
    finally:
        loop.close()


# --- DX-8: autonomous-drive token streaming (model.output.delta) ------------------------


def _streaming_loop(tmp_path: Path, adapter, *, emit: bool):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    sink = MemoryEventSink()
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=_provider("run.finish"),
        event_sinks=(sink,),
        emit_output_deltas=emit,
    )
    return loop, sink


def test_autonomous_drive_emits_output_deltas(tmp_path: Path) -> None:
    adapter = FakeStreamingModelAdapter(
        chunk_turns=[[TextDelta("Hel"), TextDelta("lo"), TurnComplete(response_id="r1", usage={"total_tokens": 3})]]
    )
    loop, sink = _streaming_loop(tmp_path, adapter, emit=True)
    loop.open()
    try:
        susp = loop.run_until_suspended("hi")
        assert susp.reason == "settled"
        assert susp.final_text == "Hello"  # assembled identically to the one-shot path
        deltas = [e for e in sink.events if e.type == "model.output.delta"]
        assert [d.data["text"] for d in deltas] == ["Hel", "lo"]
    finally:
        loop.close()


def test_autonomous_drive_emits_reasoning_deltas(tmp_path: Path) -> None:
    # DX-13b: reasoning summary fragments surface as model.reasoning.delta (display-only) and
    # are NOT folded into the assembled final_text (that stays the answer text alone).
    adapter = FakeStreamingModelAdapter(
        chunk_turns=[
            [
                ReasoningDelta("thinking… "),
                ReasoningDelta("almost there"),
                TextDelta("Answer"),
                TurnComplete(response_id="r1"),
            ]
        ]
    )
    loop, sink = _streaming_loop(tmp_path, adapter, emit=True)
    loop.open()
    try:
        susp = loop.run_until_suspended("hi")
        assert susp.reason == "settled"
        assert susp.final_text == "Answer"  # reasoning is not part of the answer
        reasoning = [e.data["text"] for e in sink.events if e.type == "model.reasoning.delta"]
        assert reasoning == ["thinking… ", "almost there"]
        answer = [e.data["text"] for e in sink.events if e.type == "model.output.delta"]
        assert answer == ["Answer"]
    finally:
        loop.close()


def test_no_output_deltas_when_disabled(tmp_path: Path) -> None:
    # Off by default: the same streaming adapter falls back to next_turn (no delta events).
    adapter = FakeStreamingModelAdapter(chunk_turns=[[TextDelta("hi"), TurnComplete()]])
    loop, sink = _streaming_loop(tmp_path, adapter, emit=False)
    loop.open()
    try:
        susp = loop.run_until_suspended("hi")
        assert susp.final_text == "hi"
        assert not [e for e in sink.events if e.type == "model.output.delta"]
    finally:
        loop.close()


class _StreamThenStopAdapter:
    """Streams text fragments and flips the loop's interrupt flag after the first one, so the
    next post-yield check aborts the stream mid-generation (immediate stop)."""

    def __init__(self) -> None:
        self.loop = None

    async def astream_turn(self, request):  # noqa: ANN001
        yield TextDelta("part1 ")
        self.loop.interrupt_turn()  # a "stop" arrives mid-stream
        yield TextDelta("part2 ")
        yield TextDelta("part3 ")  # must NOT be reached — the stream is aborted first
        yield TurnComplete(response_id="r1")

    def next_turn(self, request):  # noqa: ANN001
        return ModelTurn(final_text="unused")


def test_interrupt_aborts_stream_mid_generation(tmp_path: Path) -> None:
    adapter = _StreamThenStopAdapter()
    loop, sink = _streaming_loop(tmp_path, adapter, emit=True)
    adapter.loop = loop
    loop.open()
    try:
        susp = loop.run_until_suspended("go")
        assert susp.reason == "interrupted"
        assert loop._session is not None and loop._session.terminal is False
        texts = [e.data["text"] for e in sink.events if e.type == "model.output.delta"]
        assert texts == ["part1 ", "part2 "]  # part3 never streamed: aborted mid-generation
        assert "turn.interrupted" in [e.type for e in sink.events]
    finally:
        loop.close()


def test_from_tools_wires_a_custom_tool_end_to_end(tmp_path: Path) -> None:
    from native_agent_runner.tools.decorator import tool

    @tool(id="custom.echo", side_effect="read")
    def echo(text: str) -> dict:
        """Echo the input text."""
        return {"echoed": text}

    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("custom_echo", {"text": "hello"}, "c1"),),
            ),
            ModelTurn(response_id="r2", final_text="done"),
        ]
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop.from_tools(spec, adapter, [echo]).run_once("echo hello")

    assert result.status == "completed"
    assert result.final_text == "done"
    # The custom tool was exposed to the model under its derived exported name...
    exported = {t.exported_name for t in adapter.requests[0].tools}
    assert "custom_echo" in exported
    # ...and its result came back as an observation.
    observations = [obs for req in adapter.requests for obs in req.observations]
    assert any(obs.output.get("result") == {"echoed": "hello"} for obs in observations)
