from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import monoid_agent_kernel.loop_phases as phases
from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core.authority import ActivationWriteAuthority, WriteAuthorityRevoked
from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore
from monoid_agent_kernel.core.model_io import CapturePolicy, ModelIOSubscription
from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.recorder import AgentRecorder, JsonlEventSink

pytestmark = pytest.mark.serial


class _Observer:
    def __init__(self, on_close: Callable[[], None] | None = None) -> None:
        self.close_count = 0
        self.on_close = on_close

    def on_model_call(self, capture: Any) -> None:
        pass

    def close(self) -> None:
        self.close_count += 1
        if self.on_close is not None:
            self.on_close()


@pytest.fixture
def handles(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    opened: list[Any] = []
    repair = AgentRecorder._terminate_torn_transcript_tail
    sink_init = JsonlEventSink.__post_init__

    def capture_transcript(recorder: AgentRecorder) -> None:
        opened.append(recorder._transcript_file)
        repair(recorder)

    def capture_events(sink: JsonlEventSink) -> None:
        sink_init(sink)
        opened.append(sink._handle)

    monkeypatch.setattr(AgentRecorder, "_terminate_torn_transcript_tail", capture_transcript)
    monkeypatch.setattr(JsonlEventSink, "__post_init__", capture_events)
    yield opened
    # A red regression must not leave its deliberately retained file handles open.
    for handle in opened:
        handle.close()


def _spec(tmp_path: Path) -> AgentRunSpec:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")


def _loop(spec: AgentRunSpec, **kwargs: Any) -> AgentLoop:
    return AgentLoop(
        spec=spec,
        model_adapter=FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="park")]),
        runtime_config_provider=runtime_provider(runtime_config("hitl.request")),
        **kwargs,
    )


def _assert_released(loop: AgentLoop) -> None:
    assert loop._session is None
    assert loop._bootstrap_resources is None
    assert loop._stream_sink is None
    assert loop._authority_unsubscribe is None
    assert loop._owned_loop is None
    assert loop._finalized


def _source_checkpoint(spec: AgentRunSpec):  # noqa: ANN202
    source = AgentLoop(
        spec=spec,
        model_adapter=FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="r1",
                    tool_calls=(fake_tool_call("hitl_request", {"prompt": "Choose"}, "c1"),),
                )
            ]
        ),
        runtime_config_provider=runtime_provider(runtime_config("hitl.request")),
    )
    source.open()
    suspension = source.run_until_suspended("ask")
    assert suspension.reason == "awaiting_tasks"
    record = LocalFsCheckpointStore(spec.run_root).latest(spec.run_id)
    assert record is not None
    cancel_path = (
        spec.run_root
        / spec.run_id
        / "artifacts"
        / "tasks"
        / suspension.awaiting_task_ids[0]
        / "cancel.requested"
    )
    source.release_parked()
    return record, cancel_path


def test_constructor_guard_postcheck_closes_owned_handles(
    tmp_path: Path,
    handles: list[Any],
) -> None:
    class RevokeAfterBody(ActivationWriteAuthority):
        def guard_local_mutation(self, operation: Callable[[], Any]) -> Any:
            def finish_then_revoke() -> Any:
                result = operation()
                self.revoke()
                return result

            return super().guard_local_mutation(finish_then_revoke)

    sink = _Observer()
    with pytest.raises(WriteAuthorityRevoked):
        AgentRecorder(
            tmp_path / "runs",
            "constructor-postcheck",
            write_authority=RevokeAfterBody(),
            extra_event_sinks=(sink,),
        )
    assert len(handles) == 2
    assert all(handle.closed for handle in handles)
    assert sink.close_count == 0


@pytest.mark.parametrize("operation", ["open", "restore"])
@pytest.mark.parametrize("failure_stage", ["recorder_return", "observer", "control"])
def test_prepublication_failure_releases_first_owned_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    handles: list[Any],
    operation: str,
    failure_stage: str,
) -> None:
    spec = _spec(tmp_path)
    record, cancel_path = _source_checkpoint(spec) if operation == "restore" else (None, None)
    handles.clear()  # The source activation was already released.
    authority = ActivationWriteAuthority()
    sink = _Observer()
    observer = _Observer()
    original = phases.AgentRecorder

    def construct(*args: Any, **kwargs: Any) -> AgentRecorder:
        recorder = original(*args, **kwargs)
        if failure_stage == "recorder_return":
            authority.revoke()
        return recorder

    def factory() -> None:
        if failure_stage == "control":
            raise asyncio.CancelledError("bootstrap control")
        authority.revoke()

    monkeypatch.setattr(phases, "AgentRecorder", construct)
    loop = _loop(
        spec,
        write_authority=authority,
        event_sinks=(sink,),
        model_stream_observer_factories=(factory,),
        model_io_subscriptions=(ModelIOSubscription(observer, CapturePolicy(mode="digest")),),
    )
    expected = asyncio.CancelledError if failure_stage == "control" else WriteAuthorityRevoked
    try:
        with pytest.raises(expected):
            if record is None:
                loop.open()
            else:
                loop.restore(record.checkpoint, blobs=record.blob)
        assert len(handles) == 2
        assert all(handle.closed for handle in handles)
        _assert_released(loop)
        assert sink.close_count == 0
        assert observer.close_count == (1 if failure_stage == "control" else 0)
        loop.discard_uncommitted()
        assert sink.close_count == 0
        if record is not None:
            latest = LocalFsCheckpointStore(spec.run_root).latest(spec.run_id)
            assert latest is not None and latest.checkpoint == record.checkpoint
            assert cancel_path is not None and not cancel_path.exists()
    finally:
        loop.discard_uncommitted()


@pytest.mark.parametrize("operation", ["close", "discard_uncommitted", "release_parked"])
def test_subscription_revoke_finishes_first_activation_teardown(
    tmp_path: Path,
    operation: str,
) -> None:
    spec = _spec(tmp_path)
    authority = ActivationWriteAuthority()
    event_path = spec.run_root / spec.run_id / "events.jsonl"
    events_at_revoke: list[bytes] = []

    def revoke() -> None:
        events_at_revoke.append(event_path.read_bytes())
        authority.revoke()

    first, second = _Observer(revoke), _Observer()
    loop = _loop(
        spec,
        write_authority=authority,
        model_io_subscriptions=tuple(
            ModelIOSubscription(observer, CapturePolicy(mode="digest"))
            for observer in (first, second)
        ),
    )
    loop.open()
    loop.run_until_suspended("park")
    recorder = loop._session.res.recorder
    owned_loop = loop._owned_loop
    assert owned_loop is not None
    try:
        with pytest.raises(WriteAuthorityRevoked):
            getattr(loop, operation)()
        assert (first.close_count, second.close_count) == (1, 0)
        assert recorder._transcript_file.closed
        _assert_released(loop)
        assert owned_loop.is_closed()
        assert LocalFsCheckpointStore(spec.run_root).latest(spec.run_id) is not None
        assert event_path.read_bytes() == events_at_revoke[0]
        assert (b'"type": "run.finished"' in events_at_revoke[0]) == (operation == "close")
        loop.discard_uncommitted()
        assert (first.close_count, second.close_count) == (1, 0)
    finally:
        loop.discard_uncommitted()


@pytest.mark.parametrize("stage", ["bootstrap", "restore", "finalize"])
def test_escaping_failure_survives_cleanup_revoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    spec = _spec(tmp_path)
    record, cancel_path = _source_checkpoint(spec) if stage == "restore" else (None, None)
    authority = ActivationWriteAuthority()
    observer = _Observer(authority.revoke)
    loop = _loop(
        spec,
        write_authority=authority,
        model_io_subscriptions=(ModelIOSubscription(observer, CapturePolicy(mode="digest")),),
    )
    primary = RuntimeError("original escaping failure")

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise primary

    if stage == "bootstrap":
        loop.workspace_factory = fail
        invoke = loop.open
    elif stage == "restore":
        monkeypatch.setattr(loop, "_crashed_shell_observations", fail)
        assert record is not None
        def invoke() -> None:
            loop.restore(record.checkpoint, blobs=record.blob)
    else:
        loop.open()
        loop.run_until_suspended("park")
        monkeypatch.setattr(loop, "_finalize", fail)
        invoke = loop.close
    try:
        with pytest.raises(RuntimeError) as caught:
            invoke()
        assert caught.value is primary
        assert observer.close_count == 1
        _assert_released(loop)
        if record is not None:
            latest = LocalFsCheckpointStore(spec.run_root).latest(spec.run_id)
            assert latest is not None and latest.checkpoint == record.checkpoint
            assert cancel_path is not None and not cancel_path.exists()
    finally:
        loop.discard_uncommitted()


@pytest.mark.parametrize("stage", ["provider", "config"])
def test_recordable_bootstrap_failure_keeps_recorder_until_close(
    tmp_path: Path,
    stage: str,
) -> None:
    spec = _spec(tmp_path)
    loop = _loop(spec)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("recordable bootstrap failure")

    if stage == "provider":
        from monoid_agent_kernel.tool_loader import FunctionToolProvider

        loop.tool_providers = (FunctionToolProvider(fail),)
    else:

        class FailingConfig:
            def current_config(self, run_id: str) -> Any:
                return fail(run_id)

        loop.runtime_config_provider = FailingConfig()
    loop.open()
    assert loop._session is not None and loop._session.terminal
    recorder = loop._session.res.recorder
    try:
        assert not recorder._transcript_file.closed
        result = loop.close()
        assert result.status == "failed"
        assert result.error == "recordable bootstrap failure"
        assert recorder._transcript_file.closed
        _assert_released(loop)
        events = (spec.run_root / spec.run_id / "events.jsonl").read_text(encoding="utf-8")
        assert '"type": "run.finished"' in events
        assert loop.write_authority.revoked is False
    finally:
        loop.discard_uncommitted()
