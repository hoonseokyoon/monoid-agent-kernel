from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from support.fenced_hosting import DeterministicFencedRunHarness
from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core.model_invocation import (
    DurableModelInvocation,
    logical_model_call_id,
)
from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.errors import (
    AgentConfigError,
    DurableModelCallError,
    ModelDispatchRefused,
)
from monoid_agent_kernel.hosting import CommitResult, ModelInvocationRecord, WriterToken
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.model_lifecycle import durable_model_turn
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn, mark_provider_usage


RUN_ID = "run-durable-recovery"
LOGICAL_CALL_ID = logical_model_call_id(RUN_ID, "turn_0001")


class _HardCrash(BaseException):
    """Simulate process loss without activating the loop's Exception handlers."""


@dataclass
class _ScriptedAdapter:
    outcomes: list[ModelTurn | BaseException]
    requests: list[ModelRequest]

    def __init__(self, *outcomes: ModelTurn | BaseException) -> None:
        self.outcomes = list(outcomes)
        self.requests = []

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@dataclass
class _CrashAfterInvocationCommit:
    inner: Any
    target_state: str
    armed: bool = True

    @property
    def capabilities(self):
        return self.inner.capabilities

    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        result = self.inner.commit_invocation(
            invocation,
            blobs,
            writer_token=writer_token,
        )
        if (
            self.armed
            and result.status in {"committed", "already_committed"}
            and invocation.dispatch_state == self.target_state
        ):
            self.armed = False
            raise _HardCrash(invocation.dispatch_state)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@dataclass
class _RejectCheckpointCommit:
    inner: Any

    @property
    def capabilities(self):
        return self.inner.capabilities

    def commit_checkpoint(self, *args: Any, **kwargs: Any) -> CommitResult:
        return CommitResult(status="fenced")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@dataclass
class _TamperInvocationResultLoad:
    inner: Any

    @property
    def capabilities(self):
        return self.inner.capabilities

    def load_invocation(self, run_id: str, logical_call_id: str):
        loaded = self.inner.load_invocation(run_id, logical_call_id)
        if loaded.value is None:
            return loaded
        record = loaded.value
        return replace(
            loaded,
            value=ModelInvocationRecord(
                revision=record.revision,
                invocation=record.invocation,
                _blob_reader=lambda _sha256: b"tampered",
            ),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _spec(tmp_path: Path) -> AgentRunSpec:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        run_id=RUN_ID,
    )


def _loop(
    tmp_path: Path,
    adapter: _ScriptedAdapter,
    *,
    sink: Any,
    writer_token: WriterToken,
    checkpoint_persist_callback: Any = None,
) -> AgentLoop:
    return AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config()),
        run_sink=sink,
        writer_token=writer_token,
        checkpoint_persist_callback=checkpoint_persist_callback,
    )


def _crash_at(
    tmp_path: Path,
    harness: DeterministicFencedRunHarness,
    adapter: _ScriptedAdapter,
    state: str,
    *,
    user_input: str = "hello",
):
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=_CrashAfterInvocationCommit(harness.sink, state),
        writer_token=token,
    )
    loop.open()
    baseline = loop.snapshot()
    assert baseline is not None
    try:
        with pytest.raises(_HardCrash):
            loop.run_until_suspended(user_input)
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()
    loaded = harness.sink.load_invocation(RUN_ID, LOGICAL_CALL_ID)
    assert loaded.ok and loaded.value is not None
    assert loaded.value.invocation.dispatch_state == state
    return baseline, loaded.value


def _restore(
    tmp_path: Path,
    harness: DeterministicFencedRunHarness,
    adapter: _ScriptedAdapter,
    baseline,
    *,
    user_input: str = "hello",
    sink: Any = None,
):
    token = harness.claim_writer(RUN_ID, "worker-2")
    loop = _loop(
        tmp_path,
        adapter,
        sink=harness.sink if sink is None else sink,
        writer_token=token,
    )
    loop.restore(baseline)
    try:
        suspension = loop.run_until_suspended(user_input)
        return suspension, loop.snapshot()
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()


def test_reserved_dispatch_resumes_once_with_stored_idempotency_key(tmp_path: Path) -> None:
    harness = DeterministicFencedRunHarness()
    adapter = _ScriptedAdapter(ModelTurn(final_text="done", stop_reason="stop"))
    baseline, reserved = _crash_at(tmp_path, harness, adapter, "reserved")

    suspension, checkpoint = _restore(tmp_path, harness, adapter, baseline)

    assert suspension.reason == "settled"
    assert suspension.turn is not None and suspension.turn.final_text == "done"
    assert len(adapter.requests) == 1
    assert adapter.requests[0].idempotency_key == reserved.invocation.idempotency_key
    assert checkpoint is not None
    assert checkpoint.last_model_invocation is not None
    assert checkpoint.last_model_invocation["dispatch_state"] == "settled"


def test_started_dispatch_becomes_unknown_without_provider_reentry(tmp_path: Path) -> None:
    harness = DeterministicFencedRunHarness()
    adapter = _ScriptedAdapter(ModelTurn(final_text="must not run"))
    baseline, _started = _crash_at(tmp_path, harness, adapter, "dispatch_started")

    suspension, checkpoint = _restore(tmp_path, harness, adapter, baseline)

    assert suspension.reason == "terminal"
    assert suspension.error_code == "dispatch_unknown"
    assert adapter.requests == []
    loaded = harness.sink.load_invocation(RUN_ID, LOGICAL_CALL_ID)
    assert loaded.ok and loaded.value is not None
    assert loaded.value.invocation.dispatch_state == "unknown"
    assert checkpoint is not None
    assert checkpoint.last_model_invocation is not None
    assert checkpoint.last_model_invocation["dispatch_state"] == "unknown"


def test_settled_success_replays_private_result_without_provider_call(tmp_path: Path) -> None:
    harness = DeterministicFencedRunHarness()
    turn = ModelTurn(
        response_id="response-1",
        final_text="durable answer",
        usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        raw={"private": "provider payload"},
        stop_reason="stop",
    )
    adapter = _ScriptedAdapter(turn)
    baseline, settled = _crash_at(tmp_path, harness, adapter, "settled")
    result_sha = settled.invocation.result_ref.removeprefix("blob:")
    result_blob = settled.blob(result_sha)

    suspension, checkpoint = _restore(tmp_path, harness, adapter, baseline)

    assert suspension.reason == "settled"
    assert suspension.turn is not None
    assert suspension.turn.final_text == "durable answer"
    assert len(adapter.requests) == 1
    assert b"provider payload" not in result_blob
    assert checkpoint is not None
    assert checkpoint.total_usage == turn.usage


def test_settled_failure_replays_classification_without_provider_call(tmp_path: Path) -> None:
    harness = DeterministicFencedRunHarness()
    refusal = ModelDispatchRefused(
        "provider refused the request",
        error_code="rate_limited",
        provider_error_code="rate_limit",
        retryable=True,
        http_status=429,
        provider_retried=True,
        config_recoverable=True,
    )
    mark_provider_usage(
        refusal,
        {"input_tokens": 7, "output_tokens": 0, "total_tokens": 7},
    )
    adapter = _ScriptedAdapter(refusal)
    baseline, _settled = _crash_at(tmp_path, harness, adapter, "settled")

    suspension, checkpoint = _restore(tmp_path, harness, adapter, baseline)

    assert suspension.reason == "turn_failed"
    assert suspension.error_code == "rate_limited"
    assert suspension.provider_error_code == "rate_limit"
    assert suspension.http_status == 429
    assert suspension.retryable is True
    assert suspension.config_recoverable is True
    assert suspension.provider_retried is True
    assert len(adapter.requests) == 1
    assert checkpoint is not None
    assert checkpoint.total_usage == {
        "input_tokens": 7,
        "output_tokens": 0,
        "total_tokens": 7,
    }


def test_changed_request_is_blocked_before_provider_reentry(tmp_path: Path) -> None:
    harness = DeterministicFencedRunHarness()
    adapter = _ScriptedAdapter(ModelTurn(final_text="first"))
    baseline, _settled = _crash_at(tmp_path, harness, adapter, "settled")

    suspension, _checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        baseline,
        user_input="different input",
    )

    assert suspension.reason == "terminal"
    assert suspension.error_code == "durable_invocation_request_conflict"
    assert len(adapter.requests) == 1


def test_tampered_settled_result_is_blocked_before_provider_reentry(tmp_path: Path) -> None:
    harness = DeterministicFencedRunHarness()
    adapter = _ScriptedAdapter(ModelTurn(final_text="first"))
    baseline, settled = _crash_at(tmp_path, harness, adapter, "settled")
    result_sha = settled.invocation.result_ref.removeprefix("blob:")
    assert hashlib.sha256(settled.blob(result_sha)).hexdigest() == result_sha

    suspension, _checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        baseline,
        sink=_TamperInvocationResultLoad(harness.sink),
    )

    assert suspension.reason == "terminal"
    assert suspension.error_code == "durable_invocation_result_corrupt"
    assert len(adapter.requests) == 1


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("tool_calls", {}),
        ("reasoning", ["private text"]),
        ("usage", {"total_tokens": True}),
        ("final_text", 7),
        ("provider_retried", "yes"),
        ("unexpected_private_field", "private"),
    ],
)
def test_private_result_decoder_rejects_a_correctly_hashed_malformed_body(
    field_name: str,
    bad_value: Any,
) -> None:
    body = {
        "response_id": "response-1",
        "final_text": "answer",
        "tool_calls": [],
        "reasoning": [],
        "usage": {"total_tokens": 1},
        "stop_reason": "stop",
        "provider_retried": False,
    }
    body[field_name] = bad_value
    blob = json.dumps(body, separators=(",", ":")).encode()

    with pytest.raises(DurableModelCallError) as caught:
        durable_model_turn(blob)

    assert caught.value.error_code == "durable_invocation_result_corrupt"


@pytest.mark.parametrize(
    ("fault", "expected_code"),
    [
        ("corrupt", "durable_invocation_corrupt"),
        ("unsupported_version", "durable_invocation_unsupported_version"),
    ],
)
def test_unreadable_invocation_head_fails_closed(
    tmp_path: Path,
    fault: str,
    expected_code: str,
) -> None:
    harness = DeterministicFencedRunHarness()
    adapter = _ScriptedAdapter(ModelTurn(final_text="must not run"))
    baseline, _reserved = _crash_at(tmp_path, harness, adapter, "reserved")
    harness.inject_authoritative_load_fault(
        "invocation",
        RUN_ID,
        fault,  # type: ignore[arg-type]
        logical_call_id=LOGICAL_CALL_ID,
    )

    suspension, _checkpoint = _restore(tmp_path, harness, adapter, baseline)

    assert suspension.reason == "terminal"
    assert suspension.error_code == expected_code
    assert adapter.requests == []


def test_fenced_checkpoint_commit_escapes_without_local_fallback(tmp_path: Path) -> None:
    harness = DeterministicFencedRunHarness()
    token = harness.claim_writer(RUN_ID, "worker-1")
    adapter = _ScriptedAdapter(ModelTurn(final_text="done"))
    loop = _loop(
        tmp_path,
        adapter,
        sink=_RejectCheckpointCommit(harness.sink),
        writer_token=token,
    )
    loop.open()
    try:
        with pytest.raises(RuntimeError, match="checkpoint persistence failed"):
            loop.run_until_suspended("hello")
        assert not (tmp_path / "runs" / RUN_ID / "checkpoint.json").exists()
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()


def test_stale_writer_cannot_reserve_or_fall_back_to_a_local_checkpoint(tmp_path: Path) -> None:
    harness = DeterministicFencedRunHarness()
    stale_token = harness.claim_writer(RUN_ID, "worker-1")
    adapter = _ScriptedAdapter(ModelTurn(final_text="must not run"))
    loop = _loop(
        tmp_path,
        adapter,
        sink=harness.sink,
        writer_token=stale_token,
    )
    loop.open()
    harness.claim_writer(RUN_ID, "worker-2")
    try:
        with pytest.raises(RuntimeError, match="checkpoint persistence failed"):
            loop.run_until_suspended("hello")
        assert adapter.requests == []
        assert harness.sink.load_invocation(RUN_ID, LOGICAL_CALL_ID).status == "missing"
        assert not (tmp_path / "runs" / RUN_ID / "checkpoint.json").exists()
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()


def test_durable_hosting_configuration_is_all_or_nothing(tmp_path: Path) -> None:
    harness = DeterministicFencedRunHarness()
    token = harness.claim_writer(RUN_ID, "worker-1")
    adapter = _ScriptedAdapter(ModelTurn(final_text="done"))

    with pytest.raises(AgentConfigError, match="configured together"):
        AgentLoop(
            spec=_spec(tmp_path),
            model_adapter=adapter,
            runtime_config_provider=runtime_provider(runtime_config()),
            run_sink=harness.sink,
        )
    with pytest.raises(AgentConfigError, match="cannot be combined"):
        _loop(
            tmp_path,
            adapter,
            sink=harness.sink,
            writer_token=token,
            checkpoint_persist_callback=lambda _checkpoint, _blobs: True,
        )
    with pytest.raises(AgentConfigError, match="must match"):
        _loop(
            tmp_path,
            adapter,
            sink=harness.sink,
            writer_token=WriterToken(run_id="another-run", owner_id="worker-1", generation=1),
        )
    harness.sink.capabilities = replace(
        harness.sink.capabilities,
        durable_invocations=False,
    )
    with pytest.raises(AgentConfigError, match="durable_invocations"):
        _loop(tmp_path, adapter, sink=harness.sink, writer_token=token)
