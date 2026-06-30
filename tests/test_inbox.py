"""Inbox message envelope (inbox-message.v1) + idempotent message ingress."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from support.runtime import runtime_config
from support.waiting import eventually

from monoid_agent_kernel.core.checkpoint import RunCheckpoint
from monoid_agent_kernel.core.inbox import (
    INBOX_PROTOCOL_VERSION,
    InboxMessage,
    is_inbox_envelope,
)
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.backend.service import (
    BackendRunRequest,
    RunnerBackend,
    _queued_message_to_loop_input,
)


# --- envelope contract --------------------------------------------------------------------


def test_inbox_message_round_trips() -> None:
    env = InboxMessage(content="hello", id="m1", source="http", run_id="run_1")
    payload = env.to_json()
    assert payload["protocol"] == INBOX_PROTOCOL_VERSION
    assert payload["correlation_id"] == "m1"  # a root message correlates to its own id
    back = InboxMessage.from_json(payload)
    assert back.id == "m1"
    assert back.content == "hello"
    assert back.source == "http"


def test_is_inbox_envelope_discriminates() -> None:
    assert is_inbox_envelope(InboxMessage(content="x").to_json())
    assert not is_inbox_envelope("raw text")  # legacy raw str
    assert not is_inbox_envelope([{"type": "text", "text": "x"}])  # legacy raw parts
    assert not is_inbox_envelope({"foo": "bar"})  # an unrelated dict


def test_queued_message_unwraps_envelope_and_passes_legacy_through() -> None:
    # An envelope unwraps to its content; a legacy raw str/list passes through (back-compat).
    assert _queued_message_to_loop_input(InboxMessage(content="hi").to_json()) == "hi"
    assert _queued_message_to_loop_input("legacy text") == "legacy text"


def test_checkpoint_carries_seen_ids_and_envelope_queue() -> None:
    cp = RunCheckpoint(
        run_id="run_1",
        queued_messages=[InboxMessage(content="q", id="m2").to_json()],
        inbox_seen_ids=["m1"],
    )
    back = RunCheckpoint.from_json(cp.to_json())
    assert back is not None
    assert back.inbox_seen_ids == ["m1"]
    assert is_inbox_envelope(back.queued_messages[0])
    # An old checkpoint without the field decodes to the empty default (back-compat).
    legacy = RunCheckpoint.from_json({"run_id": "r", "schema_version": cp.schema_version})
    assert legacy is not None and legacy.inbox_seen_ids == []


# --- idempotent ingress through the backend -----------------------------------------------


def _backend(tmp_path: Path, turns: list[ModelTurn]) -> tuple[RunnerBackend, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("notes\n", encoding="utf-8")

    def factory(spec: Any, llm_gateway_token: str) -> FakeModelAdapter:
        del spec, llm_gateway_token
        return FakeModelAdapter(turns=list(turns))

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=TokenManager.from_secret("x" * 32),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
    )
    backend.idle_timeout_s = 10.0
    return backend, workspace


def test_duplicate_message_id_is_processed_once(tmp_path: Path) -> None:
    backend, workspace = _backend(
        tmp_path,
        [ModelTurn(response_id="r1", final_text="first"), ModelTurn(response_id="r2", final_text="second")],
    )
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hello",
            runtime_config=runtime_config("fs.read", "run.finish"),
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert eventually(lambda: backend._record(run_id).status == "awaiting_input")

    # First send with a stable id is accepted and processed (the run takes a turn, parks again).
    first = backend.send_message(run_id, token, content="go", message_id="m1")
    assert first["status"] == "queued"
    assert eventually(lambda: "m1" in backend._record(run_id).seen_inbox_ids)
    assert eventually(lambda: backend._record(run_id).status == "awaiting_input")

    # Re-sending the same id is an idempotent no-op — no second turn.
    dup = backend.send_message(run_id, token, content="go", message_id="m1")
    assert dup["status"] == "duplicate"

    # A different id is accepted.
    other = backend.send_message(run_id, token, content="again", message_id="m2")
    assert other["status"] == "queued"

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_send_message_propagates_trace_context_onto_envelope(tmp_path: Path) -> None:
    backend, workspace = _backend(tmp_path, [ModelTurn(response_id="r1", final_text="first")])
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hello",
            runtime_config=runtime_config("fs.read", "run.finish"),
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert eventually(lambda: backend._record(run_id).status == "awaiting_input")

    # Capture the enqueued envelope synchronously (the parked run would otherwise consume + unwrap it
    # before the test thread could read the queue). _call_soon receives the to_json() dict verbatim.
    captured: list[dict] = []
    original = backend._call_soon

    def spy(fn: Any, *args: Any) -> None:
        for a in args:
            if is_inbox_envelope(a):
                captured.append(a)
        original(fn, *args)

    backend._call_soon = spy  # type: ignore[method-assign]
    tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    backend.send_message(run_id, token, content="go", message_id="m1", traceparent=tp, tracestate="v=1")
    assert captured and captured[0]["traceparent"] == tp and captured[0]["tracestate"] == "v=1"

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_message_without_id_gets_a_generated_envelope_id(tmp_path: Path) -> None:
    backend, workspace = _backend(tmp_path, [ModelTurn(response_id="r1", final_text="first")])
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hello",
            runtime_config=runtime_config("fs.read", "run.finish"),
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert eventually(lambda: backend._record(run_id).status == "awaiting_input")
    result = backend.send_message(run_id, token, content="go")
    assert result["status"] == "queued"
    assert result["message_id"].startswith("inbox_")  # the edge minted one
    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)
