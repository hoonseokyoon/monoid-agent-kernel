"""Outbox: durable, capability-gated egress staged in core and drained at the edge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from support.runtime import runtime_config, runtime_provider, tool_binding
from support.waiting import eventually

from monoid_agent_kernel.core.agents import AgentRuntimeConfig
from monoid_agent_kernel.core.capability import AutoGrantBroker
from monoid_agent_kernel.core.outbox import Outbox, OutboxReceipt, OutboxRequest
from monoid_agent_kernel.core.external_agent_envelope import (
    external_agent_envelope_to_inbox_message,
    validate_external_agent_envelope,
)
from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.core.tool_surface import ToolScope
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.backend.service import BackendRunRequest, RunnerBackend
from monoid_agent_kernel.reference.outbox import (
    FailingOutboxSender,
    InboxRoutingOutboxSender,
    OutboxToolProvider,
    RecordingOutboxSender,
)


# --- core types + holder ------------------------------------------------------------------


def test_outbox_request_round_trips() -> None:
    req = OutboxRequest(destination="email", payload={"to": "x@a.edu"}, capability="outbox.send", token_ref="auto:outbox.send")
    payload = req.to_json()
    assert payload["protocol"] == "monoid.outbox-request.v1"
    assert payload["idempotency_key"] == req.id  # defaults to the request id
    back = OutboxRequest.from_json(payload)
    assert back.destination == "email"
    assert back.token_ref == "auto:outbox.send"
    assert back.status == "pending"


def test_outbox_request_from_json_accepts_legacy_protocol_id() -> None:
    payload = OutboxRequest(destination="email", id="o1").to_json()
    payload["protocol"] = "native-agent-runner.outbox-request.v1"

    assert OutboxRequest.from_json(payload).id == "o1"


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
_SEND_ACK = ModelTurn(
    response_id="r1",
    tool_calls=(fake_tool_call("outbox_send", {"destination": "email", "payload": {"to": "x@a.edu"}, "expect_ack": True}, "c1"),),
)
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
    from monoid_agent_kernel.core.trace_context import parse_traceparent

    tp = sender.sent[0].traceparent
    assert parse_traceparent(tp) is not None
    assert dispatched[0]["data"]["traceparent"] == tp
    child = parse_traceparent(sender.child_traceparents[0])
    assert child is not None and child["trace_id"] == parse_traceparent(tp)["trace_id"]
    assert child["span_id"] != parse_traceparent(tp)["span_id"]


def test_strict_outbox_side_effect_stages_and_dispatches(tmp_path: Path) -> None:
    sender = RecordingOutboxSender()
    backend, workspace = _outbox_backend(tmp_path, [_SEND, _DONE], sender=sender, broker=AutoGrantBroker())
    binding = tool_binding(
        "outbox.send",
        runtime={
            "requires_lease": True,
            "external_side_effect": True,
            "side_effect_delivery": "outbox",
        },
        scope=ToolScope(),
    )
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="go",
            runtime_config=AgentRuntimeConfig(
                definition_id="test-agent",
                tools=(binding,),
                metadata={"tool_side_effect_policy": {"mode": "strict"}},
            ),
        )
    )
    assert backend.wait_for_run(submission.run_id, timeout_s=20) == "completed"

    assert [request.destination for request in sender.sent] == ["email"]
    events = [
        json.loads(line)
        for line in (backend._record(submission.run_id).run_dir / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(event["type"] == "outbox.requested" for event in events)
    assert any(event["type"] == "outbox.dispatched" for event in events)


def test_outbox_not_staged_when_capability_denied(tmp_path: Path) -> None:
    from monoid_agent_kernel.reference.capability import DenyAllBroker

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


# --- backoff scheduling + watchdog redrive ------------------------------------------------


def test_outbox_request_next_attempt_at_round_trips() -> None:
    req = OutboxRequest(destination="email", id="o1", next_attempt_at=1234.5)
    assert OutboxRequest.from_json(req.to_json()).next_attempt_at == 1234.5
    # An old payload without the field defaults to 0.0 (due immediately) — back-compat.
    assert OutboxRequest.from_json({"destination": "email", "id": "o2"}).next_attempt_at == 0.0


def test_backoff_delay_is_capped_with_full_jitter(tmp_path: Path) -> None:
    sender = RecordingOutboxSender()
    backend, _ws = _outbox_backend(tmp_path, [_DONE], sender=sender)
    backend.outbox_retry_base_s, backend.outbox_retry_factor, backend.outbox_retry_cap_s = 1.0, 2.0, 10.0
    backend._outbox_rng.seed(1234)
    # Full jitter: each delay lands within [0, ceiling]; the ceiling grows with attempts but is capped.
    for attempts in range(1, 12):
        ceiling = min(10.0, 1.0 * 2.0**attempts)
        assert 0.0 <= backend._outbox_backoff_delay(attempts) <= ceiling
    assert all(backend._outbox_backoff_delay(20) <= 10.0 for _ in range(50))  # never exceeds the cap


def test_retryable_failure_stamps_future_schedule_and_is_not_due(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    binding = tool_binding("outbox.send", runtime={"requires_lease": True}, scope=ToolScope())
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=FakeModelAdapter(turns=[_SEND, ModelTurn(response_id="rw", final_text="staged")]),
        runtime_config_provider=runtime_provider(runtime_config(bindings=(binding,))),
        tool_providers=(OutboxToolProvider(),),
        capability_broker=AutoGrantBroker(),
    )
    loop.open()
    loop.run_until_suspended("go")
    [req] = loop.pending_outbox()

    now = 1000.0
    status = loop.record_outbox_result(
        req.id, OutboxReceipt(ok=False, error="x", retryable=True), next_attempt_at=now + 60
    )
    assert status == "pending"
    assert loop.due_outbox(now) == []  # scheduled into the future — not due yet
    assert [r.id for r in loop.due_outbox(now + 60)] == [req.id]  # due once its time arrives
    assert [r.id for r in loop.pending_outbox()] == [req.id]  # still in the full pending set (snapshot)
    loop.close()


def test_watchdog_redrives_due_request_while_run_is_idle(tmp_path: Path) -> None:
    # The first send fails (retryable); with base=0 the retry is immediately due. The run then parks
    # (idle) — and the watchdog redrive tick dispatches the due request without any run activity.
    class _FlakyOnce:
        calls = 0

        def send(self, request: Any) -> OutboxReceipt:
            self.calls += 1
            if self.calls == 1:
                return OutboxReceipt(ok=False, error="flaky", retryable=True)
            return OutboxReceipt(ok=True, reference=f"ok:{request.id}")

    sender = _FlakyOnce()
    backend, workspace = _outbox_backend(
        tmp_path,
        [_SEND, ModelTurn(response_id="rw", final_text="staged")],
        sender=sender,
        broker=AutoGrantBroker(),
    )
    backend.outbox_retry_base_s = 0.0  # next_attempt_at == now -> immediately due on redrive
    backend.watchdog_interval_s = 0.05
    run_id, token = _run(backend, workspace, multi_turn=True)
    assert eventually(lambda: backend._record(run_id).state.value == "awaiting_input")
    assert eventually(lambda: sender.calls >= 1)  # the park-time drain attempted (and failed) once

    backend.start_watchdog()
    # Redrive resends the now-due request while the run sits idle (no turn drove this).
    assert eventually(lambda: sender.calls >= 2)
    backend.stop_watchdog()

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


# --- ack-back (non-park): the receipt comes back as a correlated inbox message ------------


def test_outbox_ack_delivered_to_run_inbox_and_consumed(tmp_path: Path) -> None:
    from monoid_agent_kernel.core.inbox import is_inbox_envelope

    sender = RecordingOutboxSender()
    backend, workspace = _outbox_backend(tmp_path, [_SEND_ACK, _DONE, _DONE], sender=sender, broker=AutoGrantBroker())

    # Capture the exact ack envelope at staging (wrap put_nowait around the original call — no race,
    # since the put happens synchronously on the shared loop before the parked driver can consume it).
    captured: list[dict] = []
    original_stage = backend._stage_outbox_ack

    def stage_spy(record: Any, request: Any, status: str, receipt: Any) -> None:
        real_put = record.message_queue.put_nowait

        def put_spy(item: Any) -> None:
            if is_inbox_envelope(item) and item.get("type") == "outbox_ack":
                captured.append(item)
            real_put(item)

        record.message_queue.put_nowait = put_spy  # type: ignore[method-assign]
        try:
            original_stage(record, request, status, receipt)
        finally:
            record.message_queue.put_nowait = real_put  # type: ignore[method-assign]

    backend._stage_outbox_ack = stage_spy  # type: ignore[method-assign]

    run_id, token = _run(backend, workspace, multi_turn=True)
    request_id = sender.sent[0].id if eventually(lambda: sender.sent) else ""
    # The ack is delivered and *consumed* (the run takes a turn on it) — its id lands in the seen-set.
    assert eventually(lambda: f"ack_{request_id}" in backend._record(run_id).seen_inbox_ids)

    assert captured, "no outbox_ack envelope was staged"
    ack = captured[0]
    assert ack["type"] == "outbox_ack" and ack["source"] == "outbox"
    assert ack["correlation_id"] == request_id  # correlated to the request's flow
    assert ack["causation_id"] == request_id  # the send is the direct cause of the ack
    assert "dispatched" in ack["content"]
    assert ack["traceparent"] == sender.sent[0].traceparent  # the trace rides the ack

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_outbox_without_expect_ack_delivers_no_inbox_message(tmp_path: Path) -> None:
    sender = RecordingOutboxSender()
    backend, workspace = _outbox_backend(tmp_path, [_SEND, _DONE], sender=sender, broker=AutoGrantBroker())
    run_id, _token = _run(backend, workspace)
    assert backend.wait_for_run(run_id, timeout_s=20) == "completed"
    assert sender.sent and not sender.sent[0].expect_ack
    # No ack id was ever marked seen (nothing was delivered back).
    assert not any(sid.startswith("ack_") for sid in backend._record(run_id).seen_inbox_ids)


# --- A2A: route one agent's outbox into another agent's inbox -----------------------------


def _send_to(peer: str, text: str, call_id: str) -> ModelTurn:
    return ModelTurn(
        tool_calls=(fake_tool_call("outbox_send", {"destination": peer, "payload": {"text": text}}, call_id),)
    )


def test_a2a_outbox_routes_into_peer_inbox_bidirectional(tmp_path: Path) -> None:
    """Agent-to-agent over the durable fabric: planner stages an ``outbox.send`` to ``worker``; the
    routing sender delivers it into worker's idempotent inbox, worker consumes it as a turn and
    replies, and the reply is routed back into planner's inbox. Proven by each peer emitting an
    ``outbox.dispatched`` addressed to the other (worker could only reply if it received)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("n\n", encoding="utf-8")

    directory: dict[str, str] = {}  # agent name -> run_id
    tokens: dict[str, str] = {}
    holder: dict[str, Any] = {}
    delivered_sources: list[tuple[str, str]] = []

    def deliver(destination, payload, *, message_id, correlation_id, causation_id, traceparent):
        run_id = directory.get(destination)
        if not run_id:  # peer not registered yet -> retryable; the backend redrives
            raise LookupError(f"no agent {destination!r}")
        envelope = validate_external_agent_envelope(dict(payload))
        message = external_agent_envelope_to_inbox_message(envelope, run_id=run_id)
        delivered_sources.append((destination, message.source))
        holder["backend"].send_message(
            run_id,
            tokens[run_id],
            message.content,
            message_id=message.id or message_id,
            source=message.source,
            correlation_id=message.correlation_id or correlation_id,
            causation_id=message.causation_id or causation_id,
            traceparent=message.traceparent or traceparent,
            message_type=message.type,
            metadata=message.metadata,
        )
        return f"a2a:{run_id}"

    # Worker bootstraps first (we wait for it below), so this ordered script queue is deterministic.
    # Worker: settle the opening turn, then on receiving planner's message reply back to planner.
    worker_script = [ModelTurn(final_text="standing by"), _send_to("planner", "done: ok", "w1")]
    planner_script = [_send_to("worker", "please do X", "p1"), ModelTurn(final_text="sent")]
    pending = [worker_script, planner_script]

    def factory(spec: Any, llm_gateway_token: str) -> FakeModelAdapter:
        del spec, llm_gateway_token
        return FakeModelAdapter(turns=list(pending.pop(0) if pending else []))

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=TokenManager.from_secret("x" * 32),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
        tool_providers=(OutboxToolProvider(),),
        capability_broker_factory=lambda req: AutoGrantBroker(),
        outbox_sender_factory=lambda req: InboxRoutingOutboxSender(
            deliver=deliver,
            source_peer_id=str(req.metadata.get("message_fabric_peer_id") or ""),
        ),
    )
    backend.idle_timeout_s = 10.0
    holder["backend"] = backend

    def spawn(name: str, instruction: str) -> str:
        binding = tool_binding("outbox.send", runtime={"requires_lease": True}, scope=ToolScope())
        sub = backend.submit_run(
            BackendRunRequest(
                tenant_id="tenant_a", user_id="user_a", workspace_root=workspace,
                instruction=instruction, runtime_config=runtime_config(bindings=(binding,)),
                multi_turn=True, metadata={"message_fabric_peer_id": name},
            )
        )
        tokens[sub.run_id] = sub.run_token
        return sub.run_id

    try:
        worker_id = spawn("worker", "stand by for planner")
        directory["worker"] = worker_id
        assert eventually(lambda: backend.status(worker_id, tokens[worker_id]).get("state") == "awaiting_input")
        planner_id = spawn("planner", "collaborate with worker")
        directory["planner"] = planner_id

        def dispatched_to(run_id: str, dest: str) -> bool:
            path = backend._record(run_id).run_dir / "events.jsonl"
            if not path.exists():
                return False
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            return any(
                e["type"] == "outbox.dispatched" and e["data"].get("destination") == dest for e in events
            )

        assert eventually(lambda: dispatched_to(planner_id, "worker"))   # A -> B
        assert eventually(lambda: dispatched_to(worker_id, "planner"))   # B -> A (so B received A's message)
        assert ("worker", "external-agent:planner") in delivered_sources
        assert ("planner", "external-agent:worker") in delivered_sources

        # The inbox is idempotent: once a message id has been processed, re-delivering it is a no-op.
        assert backend.send_message(worker_id, tokens[worker_id], "dup", message_id="m-dup")["status"] == "queued"
        assert eventually(lambda: "m-dup" in backend._record(worker_id).seen_inbox_ids)
        again = backend.send_message(worker_id, tokens[worker_id], "dup", message_id="m-dup")
        assert again["status"] == "duplicate"
    finally:
        backend.shutdown(drain=True)
