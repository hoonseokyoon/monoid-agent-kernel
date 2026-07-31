from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore
from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.model_io import (
    CapturePolicy,
    ModelCallCapture,
    ModelIOSubscription,
)
from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.core.trace_context import new_traceparent
from monoid_agent_kernel.errors import ModelAdapterError, NativeAgentError
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.workspace.local import default_local_workspace_factory


class RecordingObserver:
    def __init__(self) -> None:
        self.captures: list[ModelCallCapture] = []
        self.close_count = 0

    def on_model_call(self, capture: ModelCallCapture) -> None:
        self.captures.append(capture)

    def close(self) -> None:
        self.close_count += 1


class RecordingEventSink:
    def __init__(self) -> None:
        self.close_count = 0

    def emit(self, event: AgentEvent) -> None:
        del event

    def close(self) -> None:
        self.close_count += 1


def _loop(
    tmp_path: Path,
    adapter: FakeModelAdapter,
    observer: RecordingObserver,
    *,
    context: InvocationContext | None = None,
    duplicate_policy: bool = False,
) -> AgentLoop:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subscriptions = [
        ModelIOSubscription(observer, CapturePolicy(mode="full")),
    ]
    if duplicate_policy:
        subscriptions.append(ModelIOSubscription(observer, CapturePolicy(mode="digest")))
    return AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace,
            run_root=tmp_path / "runs",
            run_id="run-model-io",
        ),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("fs.write")),
        invocation_context=context,
        model_io_subscriptions=tuple(subscriptions),
    )


def test_agent_loop_delivers_every_turn_with_durable_context_and_closes_once(
    tmp_path: Path,
) -> None:
    observer = RecordingObserver()
    traceparent = new_traceparent()
    loop = _loop(
        tmp_path,
        FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="r1",
                    tool_calls=(
                        fake_tool_call(
                            "fs_write",
                            {"path": "answer.txt", "content": "first"},
                            "call-1",
                        ),
                    ),
                ),
                ModelTurn(response_id="r2", final_text="done"),
            ]
        ),
        observer,
        context=InvocationContext(
            run_id="caller-run",
            skill_id="summarize",
            skill_digest="sha256:skill",
            step_id="caller-step",
            attempt=9,
            batch_id="batch-1",
            item_id="item-2",
            case_id="case-3",
            traceparent=traceparent,
            tracestate="vendor=value",
            attributes={"tenant": "alpha"},
        ),
        duplicate_policy=True,
    )

    result = loop.run_once("write and finish")
    loop.discard_uncommitted()

    assert result.status == "completed"
    # Two policies observe each of two model calls. Identity remains identical across the views.
    assert len(observer.captures) == 4
    contexts = [observer.captures[index].receipt.context for index in (0, 2)]
    assert [context.step_id for context in contexts] == [
        "caller-step/turn_0001",
        "caller-step/turn_0002",
    ]
    for context in contexts:
        assert context.run_id == "run-model-io"
        assert context.attempt == 9
        assert context.skill_id == "summarize"
        assert context.skill_digest == "sha256:skill"
        assert context.batch_id == "batch-1"
        assert context.item_id == "item-2"
        assert context.case_id == "case-3"
        assert context.traceparent == traceparent
        assert context.tracestate == "vendor=value"
        assert context.attributes == {"tenant": "alpha"}
    assert observer.captures[0].content is not None
    assert observer.captures[0].content["instruction"] == "write and finish"
    assert observer.captures[2].content is not None
    assert observer.captures[2].content["output_text"] == "done"
    # The same exporter can be registered under multiple policies but remains one owned resource.
    assert observer.close_count == 1


def test_agent_loop_delivers_a_failed_receipt_before_terminal_cleanup(tmp_path: Path) -> None:
    class FailingAdapter(FakeModelAdapter):
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            raise ModelAdapterError(
                "provider unavailable",
                provider_error_code="overloaded",
                retryable=False,
                http_status=503,
            )

    observer = RecordingObserver()
    loop = _loop(tmp_path, FailingAdapter(), observer)

    result = loop.run_once("fail")

    assert result.status == "failed"
    assert len(observer.captures) == 1
    receipt = observer.captures[0].receipt
    assert receipt.succeeded is False
    assert receipt.context.run_id == "run-model-io"
    assert receipt.context.step_id == "turn_0001"
    assert receipt.error_code == "model_error"
    assert receipt.provider_error_code == "overloaded"
    assert receipt.http_status == 503
    assert observer.close_count == 1


def test_discard_closes_model_io_even_when_no_call_was_made(tmp_path: Path) -> None:
    observer = RecordingObserver()
    loop = _loop(tmp_path, FakeModelAdapter(), observer)

    loop.open()
    loop.discard_uncommitted()
    loop.discard_uncommitted()

    assert observer.captures == []
    assert observer.close_count == 1


def test_bootstrap_failure_before_resources_closes_owned_model_io(tmp_path: Path) -> None:
    observer = RecordingObserver()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def fail_workspace(_spec: AgentRunSpec) -> Any:
        raise RuntimeError("workspace unavailable")

    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=FakeModelAdapter(),
        runtime_config_provider=runtime_provider(runtime_config()),
        workspace_factory=fail_workspace,
        model_io_subscriptions=(ModelIOSubscription(observer, CapturePolicy(mode="digest")),),
    )

    try:
        loop.open()
    except RuntimeError as exc:
        assert str(exc) == "workspace unavailable"
    else:  # pragma: no cover - the workspace factory defines the failure boundary
        raise AssertionError("bootstrap failure was ignored")

    assert observer.close_count == 1


def test_failed_close_discards_activation_and_rejects_further_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = RecordingObserver()
    loop = _loop(tmp_path, FakeModelAdapter(), observer)
    loop.open()

    def fail_finalize(*_args: Any) -> Any:
        raise RuntimeError("terminal writer unavailable")

    monkeypatch.setattr(loop, "_finalize", fail_finalize)

    with pytest.raises(RuntimeError, match="terminal writer unavailable"):
        loop.close()

    assert observer.close_count == 1
    with pytest.raises(NativeAgentError) as exc_info:
        loop.submit("must not run")
    assert exc_info.value.error_code == "run_not_open"


def test_closed_subscription_activation_cannot_be_reopened(tmp_path: Path) -> None:
    observer = RecordingObserver()
    loop = _loop(
        tmp_path,
        FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="done")]),
        observer,
    )

    assert loop.run_once("finish").status == "completed"

    with pytest.raises(NativeAgentError) as exc_info:
        loop.open()
    assert exc_info.value.error_code == "model_io_subscriptions_closed"
    assert observer.close_count == 1


def test_partial_bootstrap_recording_failure_discards_published_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    observer = RecordingObserver()

    def fail_config(_run_id: str) -> Any:
        raise RuntimeError("runtime config unavailable")

    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=FakeModelAdapter(),
        runtime_config_provider=fail_config,
        model_io_subscriptions=(ModelIOSubscription(observer, CapturePolicy(mode="digest")),),
    )

    def fail_record(*_args: Any) -> None:
        raise OSError("failure writer unavailable")

    monkeypatch.setattr(loop, "_record_failure", fail_record)

    with pytest.raises(OSError, match="failure writer unavailable"):
        loop.open()

    assert observer.close_count == 1
    assert loop._session is None
    assert loop._bootstrap_resources is None


def test_checkpoint_delete_failure_ends_activation_without_double_closing_sinks(
    tmp_path: Path,
) -> None:
    class FailingDeleteStore(LocalFsCheckpointStore):
        def delete(self, run_id: str) -> None:
            del run_id
            raise OSError("checkpoint delete unavailable")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    observer = RecordingObserver()
    event_sink = RecordingEventSink()
    loop = AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace,
            run_root=tmp_path / "runs",
            run_id="run-delete-failure",
        ),
        model_adapter=FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="done")]),
        runtime_config_provider=runtime_provider(runtime_config()),
        checkpoint_store=FailingDeleteStore(tmp_path / "runs"),
        event_sinks=(event_sink,),
        model_io_subscriptions=(ModelIOSubscription(observer, CapturePolicy(mode="digest")),),
    )
    loop.open()
    loop.submit("finish")

    with pytest.raises(OSError, match="checkpoint delete unavailable"):
        loop.close()

    assert event_sink.close_count == 1
    assert observer.close_count == 1
    assert loop._session is None
    assert loop._bootstrap_resources is None


def test_cancelled_aopen_waits_for_bootstrap_then_discards_without_resurrection(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    observer = RecordingObserver()
    entered = threading.Event()
    release = threading.Event()

    def blocking_workspace(spec: AgentRunSpec) -> Any:
        entered.set()
        assert release.wait(timeout=10)
        return default_local_workspace_factory(spec)

    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=FakeModelAdapter(),
        runtime_config_provider=runtime_provider(runtime_config()),
        workspace_factory=blocking_workspace,
        model_io_subscriptions=(ModelIOSubscription(observer, CapturePolicy(mode="digest")),),
    )

    async def exercise() -> None:
        task = asyncio.create_task(loop.aopen())
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    assert observer.close_count == 1
    assert loop._session is None
    assert loop._bootstrap_resources is None


def test_raising_optional_close_probe_cannot_change_outcome_or_skip_later_observer(
    tmp_path: Path,
) -> None:
    class BrokenCloseProbeObserver:
        def on_model_call(self, capture: ModelCallCapture) -> None:
            del capture

        @property
        def close(self) -> Any:
            raise RuntimeError("close probe failed")

    healthy = RecordingObserver()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="done")]),
        runtime_config_provider=runtime_provider(runtime_config()),
        model_io_subscriptions=(
            ModelIOSubscription(BrokenCloseProbeObserver(), CapturePolicy(mode="digest")),
            ModelIOSubscription(healthy, CapturePolicy(mode="digest")),
        ),
    )

    result = loop.run_once("finish")

    assert result.status == "completed"
    assert healthy.close_count == 1
    assert loop._session is None
