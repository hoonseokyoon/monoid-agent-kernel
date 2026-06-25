"""Outbox: durable, capability-gated egress staged in core and drained at the edge."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from conftest import runtime_config, runtime_provider, tool_binding

from native_agent_runner.core.capability import AutoGrantBroker
from native_agent_runner.core.outbox import Outbox, OutboxReceipt, OutboxRequest
from native_agent_runner.core.spec import AgentRunSpec
from native_agent_runner.core.tool_surface import ToolScope
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call
from native_agent_runner.reference._shared.tokens import TokenManager
from native_agent_runner.reference.backend.service import BackendRunRequest, RunnerBackend
from native_agent_runner.reference.outbox import (
    FailingOutboxSender,
    OutboxToolProvider,
    RecordingOutboxSender,
)


# --- core types + holder ------------------------------------------------------------------


def test_outbox_request_round_trips() -> None:
    req = OutboxRequest(destination="email", payload={"to": "x@a.edu"}, capability="outbox.send", token_ref="auto:outbox.send")
    payload = req.to_json()
    assert payload["protocol"] == "native-agent-runner.outbox-request.v1"
    assert payload["idempotency_key"] == req.id  # defaults to the request id
    back = OutboxRequest.from_json(payload)
    assert back.destination == "email"
    assert back.token_ref == "auto:outbox.send"
    assert back.status == "pending"


def test_outbox_holder_pending_mark_export_import() -> None:
    box = Outbox()
    a = box.append(OutboxRequest(destination="email", id="o1"))
    box.append(OutboxRequest(destination="webhook", id="o2"))
    assert {r.id for r in box.pending()} == {"o1", "o2"}
    box.mark("o1", status="dispatched", attempts=1, reference="ref-1")
    assert {r.id for r in box.pending()} == {"o2"}  # dispatched is no longer pending
    assert a.reference == "ref-1"

    fresh = Outbox()
    fresh.import_(box.export())
    assert fresh.get("o1").status == "dispatched"  # full state round-trips
    assert {r.id for r in fresh.pending()} == {"o2"}


# --- backend e2e: capability-gated staging + edge drain -----------------------------------


def _wait(predicate: Any, tries: int = 1000) -> bool:
    for _ in range(tries):
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _outbox_backend(
    tmp_path: Path,
    turns: list[ModelTurn],
    *,
    sender: Any,
    broker: Any = None,
) -> tuple[RunnerBackend, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("n\n", encoding="utf-8")

    def factory(spec: Any, llm_gateway_token: str) -> FakeModelAdapter:
        del spec, llm_gateway_token
        return FakeModelAdapter(turns=list(turns))

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=TokenManager.from_secret("x" * 32),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
        tool_providers=(OutboxToolProvider(),),
        capability_broker_factory=(lambda req: broker) if broker is not None else None,
        outbox_sender_factory=lambda req: sender,
    )
    backend.idle_timeout_s = 10.0
    return backend, workspace


def _run(backend: RunnerBackend, workspace: Path, *, multi_turn: bool = False) -> tuple[str, str]:
    binding = tool_binding("outbox.send", runtime={"requires_lease": True}, scope=ToolScope())
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="go",
            runtime_config=runtime_config(bindings=(binding,)),
            multi_turn=multi_turn,
        )
    )
    return submission.run_id, submission.run_token


_SEND = ModelTurn(response_id="r1", tool_calls=(fake_tool_call("outbox_send", {"destination": "email", "payload": {"to": "x@a.edu"}}, "c1"),))
_DONE = ModelTurn(response_id="rN", final_text="done")


def test_outbox_staged_then_dispatched_by_edge_with_lease_handle(tmp_path: Path) -> None:
    sender = RecordingOutboxSender()
    backend, workspace = _outbox_backend(tmp_path, [_SEND, _DONE], sender=sender, broker=AutoGrantBroker())
    run_id, _token = _run(backend, workspace)
    assert backend.wait_for_run(run_id, timeout_s=20) == "completed"

    # The edge drained the staged request, and it carried the capability lease handle (not a secret).
    assert [r.destination for r in sender.sent] == ["email"]
    assert sender.sent[0].token_ref == "auto:outbox.send"

    events = [json.loads(line) for line in (backend._record(run_id).run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["type"] == "outbox.requested" for e in events)
    dispatched = [e for e in events if e["type"] == "outbox.dispatched"]
    assert dispatched and dispatched[0]["data"]["destination"] == "email"

    # The request carried a W3C trace from staging; the dispatched event surfaces it, and the edge
    # sender attached a *child* span (same trace-id, new span-id) for the actual outbound call.
    from native_agent_runner.core.trace_context import parse_traceparent

    tp = sender.sent[0].traceparent
    assert parse_traceparent(tp) is not None
    assert dispatched[0]["data"]["traceparent"] == tp
    child = parse_traceparent(sender.child_traceparents[0])
    assert child is not None and child["trace_id"] == parse_traceparent(tp)["trace_id"]
    assert child["span_id"] != parse_traceparent(tp)["span_id"]


def test_outbox_not_staged_when_capability_denied(tmp_path: Path) -> None:
    from native_agent_runner.reference.capability import DenyAllBroker

    sender = RecordingOutboxSender()
    backend, workspace = _outbox_backend(tmp_path, [_SEND, _DONE], sender=sender, broker=DenyAllBroker())
    run_id, _token = _run(backend, workspace)
    backend.wait_for_run(run_id, timeout_s=20)
    # The gate denied the capability -> the tool never ran -> nothing was staged or dispatched.
    assert sender.sent == []


def test_outbox_retryable_failure_then_dead_letters(tmp_path: Path) -> None:
    # A retryable sender keeps the request pending across drains; bound the attempts so it
    # eventually dead-letters as failed rather than looping forever.
    sender = FailingOutboxSender(retryable=True)
    backend, workspace = _outbox_backend(tmp_path, [_SEND, _DONE], sender=sender, broker=AutoGrantBroker())
    backend.outbox_max_attempts = 1  # fail on the first attempt
    run_id, _token = _run(backend, workspace)
    backend.wait_for_run(run_id, timeout_s=20)
    events = [json.loads(line) for line in (backend._record(run_id).run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    failed = [e for e in events if e["type"] == "outbox.failed"]
    assert failed and failed[0]["data"]["destination"] == "email"


# --- crash-safety: a pending request survives restart and dispatches ----------------------


def test_pending_outbox_survives_snapshot_restore(tmp_path: Path) -> None:
    # Stage a request in one loop (no sender -> stays pending), snapshot, restore into a fresh loop,
    # and confirm the request (with its lease handle) round-tripped and is still pending to dispatch.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    binding = tool_binding("outbox.send", runtime={"requires_lease": True}, scope=ToolScope())
    loop1 = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=FakeModelAdapter(turns=[_SEND, ModelTurn(response_id="rw", final_text="staged")]),
        runtime_config_provider=runtime_provider(runtime_config(bindings=(binding,))),
        tool_providers=(OutboxToolProvider(),),
        capability_broker=AutoGrantBroker(),
    )
    loop1.open()
    loop1.run_until_suspended("go")
    pending = loop1.pending_outbox()
    assert [r.destination for r in pending] == ["email"]
    cp = loop1.snapshot()
    assert cp is not None and len(cp.outbox_requests) == 1
    blobs = loop1.collect_checkpoint_blobs()
    run_id = loop1.spec.run_id

    loop2 = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs", run_id=run_id),
        model_adapter=FakeModelAdapter(turns=[ModelTurn(response_id="rb", final_text="done")]),
        runtime_config_provider=runtime_provider(runtime_config(bindings=(binding,))),
        tool_providers=(OutboxToolProvider(),),
        capability_broker=AutoGrantBroker(),
    )
    loop2.restore(cp, blobs=blobs)
    restored = loop2.pending_outbox()
    assert [r.destination for r in restored] == ["email"]
    assert restored[0].token_ref == "auto:outbox.send"  # the handle survived the restart

    # The edge can now dispatch the restored request.
    assert loop2.record_outbox_result(restored[0].id, OutboxReceipt(ok=True, reference="r")) == "dispatched"
    assert loop2.pending_outbox() == []
    loop2.close()
