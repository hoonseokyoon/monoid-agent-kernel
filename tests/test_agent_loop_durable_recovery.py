from __future__ import annotations

import asyncio
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

from monoid_agent_kernel.core.content import ImagePart, TextPart
from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.model_invocation import (
    DurableModelInvocation,
    logical_model_call_id,
    model_dispatch_id,
)
from monoid_agent_kernel.core.spec import AgentRunSpec, ModelConfig, ModelRetryConfig, RunLimits
from monoid_agent_kernel.errors import (
    AgentConfigError,
    DurableModelCallError,
    ModelDispatchRefused,
    NativeAgentError,
    TurnNotSettled,
)
from monoid_agent_kernel.hosting import CommitResult, ModelInvocationRecord, WriterToken
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.model_call import ModelCallRunner
from monoid_agent_kernel.model_lifecycle import (
    ModelDispatchReservation,
    RecoveredModelDispatch,
    durable_model_result_blob,
    durable_model_turn,
)
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelTurn,
    ToolCall,
    mark_provider_usage,
    provider_usage_of,
)


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


class _RaisingDynamicContextProvider:
    def static_segment(self) -> None:
        return None

    def dynamic_segment(self, turn: Any) -> None:
        del turn
        raise AssertionError("dynamic context must not run before stored outcome application")


class _InterruptingDynamicContextProvider:
    loop: AgentLoop | None = None

    def static_segment(self) -> None:
        return None

    def dynamic_segment(self, turn: Any) -> None:
        del turn
        assert self.loop is not None
        self.loop.interrupt_turn()
        return None


@dataclass
class _CrashAfterInvocationCommit:
    inner: Any
    target_state: str
    target_failure_code: str | None = None
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
            and (
                self.target_failure_code is None
                or invocation.failure_code == self.target_failure_code
            )
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
class _RejectModelEvidence:
    inner: Any
    failures: int = 1

    @property
    def capabilities(self):
        return self.inner.capabilities

    def commit_model_evidence(self, *args: Any, **kwargs: Any) -> CommitResult:
        if self.failures > 0:
            self.failures -= 1
            return CommitResult(status="conflict")
        return self.inner.commit_model_evidence(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@dataclass
class _RejectNthModelEvidence:
    inner: Any
    reject_on: int
    calls: int = 0

    @property
    def capabilities(self):
        return self.inner.capabilities

    def commit_model_evidence(self, *args: Any, **kwargs: Any) -> CommitResult:
        self.calls += 1
        if self.calls == self.reject_on:
            return CommitResult(status="conflict")
        return self.inner.commit_model_evidence(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@dataclass
class _RejectAtomicEvidenceStage:
    inner: Any

    @property
    def capabilities(self):
        return self.inner.capabilities

    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
        stage_evidence: bool = False,
    ) -> CommitResult:
        if stage_evidence:
            return CommitResult(status="conflict", sequence=invocation.revision)
        return self.inner.commit_invocation(
            invocation,
            blobs,
            writer_token=writer_token,
        )

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


@dataclass
class _RecoveredResultHook:
    result_blob: bytes
    receipt: Mapping[str, Any]

    def recover(self, query: Any) -> RecoveredModelDispatch:
        return RecoveredModelDispatch(
            reservation=ModelDispatchReservation(
                logical_call_id=query.logical_call_id,
                dispatch_attempt=1,
                dispatch_id=model_dispatch_id(query.logical_call_id, 1),
                request_digest=query.request_digest,
                digest_generation=query.digest_generation,
                idempotency_key="idem_recovered_result",
            ),
            receipt=self.receipt,
            result_blob=self.result_blob,
        )


def _spec(tmp_path: Path, *, limits: RunLimits | None = None) -> AgentRunSpec:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        run_id=RUN_ID,
        limits=limits or RunLimits(),
    )


def _loop(
    tmp_path: Path,
    adapter: _ScriptedAdapter,
    *,
    sink: Any,
    writer_token: WriterToken,
    checkpoint_persist_callback: Any = None,
    invocation_context: InvocationContext | None = None,
    model: ModelConfig | None = None,
    model_evidence_policy: str = "passive",
    limits: RunLimits | None = None,
    context_providers: tuple[Any, ...] = (),
    model_calls_file: bool = False,
) -> AgentLoop:
    return AgentLoop(
        spec=_spec(tmp_path, limits=limits),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config(model=model)),
        run_sink=sink,
        writer_token=writer_token,
        checkpoint_persist_callback=checkpoint_persist_callback,
        invocation_context=invocation_context,
        model_evidence_policy=model_evidence_policy,  # type: ignore[arg-type]
        context_providers=context_providers,
        model_calls_file=model_calls_file,
    )


def _crash_at(
    tmp_path: Path,
    harness: DeterministicFencedRunHarness,
    adapter: _ScriptedAdapter,
    state: str,
    *,
    user_input: str | None = "hello",
    invocation_context: InvocationContext | None = None,
    target_failure_code: str | None = None,
    model: ModelConfig | None = None,
    model_evidence_policy: str = "passive",
    limits: RunLimits | None = None,
):
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=_CrashAfterInvocationCommit(
            harness.sink,
            state,
            target_failure_code=target_failure_code,
        ),
        writer_token=token,
        invocation_context=invocation_context,
        model=model,
        model_evidence_policy=model_evidence_policy,
        limits=limits,
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
    user_input: str | None = "hello",
    sink: Any = None,
    invocation_context: InvocationContext | None = None,
    model: ModelConfig | None = None,
    model_evidence_policy: str = "passive",
    limits: RunLimits | None = None,
    context_providers: tuple[Any, ...] = (),
    model_calls_file: bool = False,
):
    token = harness.claim_writer(RUN_ID, "worker-2")
    loop = _loop(
        tmp_path,
        adapter,
        sink=harness.sink if sink is None else sink,
        writer_token=token,
        invocation_context=invocation_context,
        model=model,
        model_evidence_policy=model_evidence_policy,
        limits=limits,
        context_providers=context_providers,
        model_calls_file=model_calls_file,
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
        usage={
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "cache_creation_tokens": 2,
            "audio_tokens": 1,
        },
        raw={"private": "provider payload"},
        stop_reason="provider stop reason",
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
    assert settled.invocation.receipt is not None
    assert "stop_reason" not in settled.invocation.receipt
    assert settled.invocation.receipt["usage"] == turn.usage
    assert checkpoint is not None
    assert checkpoint.total_usage == turn.usage


def test_recovery_identity_ignores_changed_caller_provenance(tmp_path: Path) -> None:
    harness = DeterministicFencedRunHarness()
    adapter = _ScriptedAdapter(ModelTurn(final_text="durable answer", stop_reason="stop"))
    baseline, settled = _crash_at(
        tmp_path,
        harness,
        adapter,
        "settled",
        invocation_context=InvocationContext(step_id="pipeline-step-before-crash"),
    )

    suspension, _checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        baseline,
        invocation_context=InvocationContext(step_id="different-restored-step"),
    )

    assert suspension.reason == "settled"
    assert suspension.turn is not None and suspension.turn.final_text == "durable answer"
    assert settled.invocation.logical_call_id == LOGICAL_CALL_ID
    assert len(adapter.requests) == 1


def test_recovered_receipt_keeps_whole_call_usage_across_kernel_retry(tmp_path: Path) -> None:
    harness = DeterministicFencedRunHarness()
    absorbed = ModelDispatchRefused("transient overload", retryable=True)
    mark_provider_usage(
        absorbed,
        {
            "input_tokens": 2,
            "output_tokens": 3,
            "total_tokens": 5,
            "cache_creation_tokens": 1,
        },
    )
    final_turn = ModelTurn(
        final_text="recovered",
        usage={
            "input_tokens": 3,
            "output_tokens": 4,
            "total_tokens": 7,
            "audio_tokens": 2,
        },
        stop_reason="stop",
    )
    adapter = _ScriptedAdapter(absorbed, final_turn)
    model = ModelConfig(
        retry=ModelRetryConfig(
            layer="kernel",
            max_attempts=2,
            initial_delay_s=0.0,
            jitter_s=0.0,
        )
    )
    baseline, settled = _crash_at(
        tmp_path,
        harness,
        adapter,
        "settled",
        target_failure_code="",
        model=model,
    )

    suspension, checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        baseline,
        model=model,
    )

    whole_call_usage = {
        "input_tokens": 5,
        "output_tokens": 7,
        "total_tokens": 12,
        "cache_creation_tokens": 1,
        "audio_tokens": 2,
    }
    assert suspension.reason == "settled"
    assert suspension.turn is not None and suspension.turn.final_text == "recovered"
    result_sha = settled.invocation.result_ref.removeprefix("blob:")
    assert durable_model_turn(settled.blob(result_sha)).usage == final_turn.usage
    assert settled.invocation.receipt is not None
    assert settled.invocation.receipt["usage"] == whole_call_usage
    assert checkpoint is not None and checkpoint.total_usage == whole_call_usage
    assert len(adapter.requests) == 2


def test_recovered_retryable_refusal_resumes_remaining_kernel_attempt(tmp_path: Path) -> None:
    harness = DeterministicFencedRunHarness()
    refusal = ModelDispatchRefused("transient overload", retryable=True)
    mark_provider_usage(
        refusal,
        {
            "input_tokens": 2,
            "output_tokens": 1,
            "total_tokens": 3,
            "cache_read_tokens": 1,
        },
    )
    final_turn = ModelTurn(
        final_text="recovered after crash",
        usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        stop_reason="stop",
    )
    adapter = _ScriptedAdapter(refusal, final_turn)
    model = ModelConfig(
        retry=ModelRetryConfig(
            layer="kernel",
            max_attempts=2,
            initial_delay_s=0.0,
            jitter_s=0.0,
        )
    )
    baseline, failed = _crash_at(
        tmp_path,
        harness,
        adapter,
        "settled",
        model=model,
    )

    suspension, checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        baseline,
        model=model,
    )

    expected_usage = {
        "input_tokens": 5,
        "output_tokens": 3,
        "total_tokens": 8,
        "cache_read_tokens": 1,
    }
    assert failed.invocation.receipt is not None
    assert failed.invocation.receipt["attempts"] == 1
    assert failed.invocation.receipt["retryable"] is True
    assert failed.invocation.receipt["stream_committed"] is False
    assert suspension.reason == "settled"
    assert suspension.turn is not None
    assert suspension.turn.final_text == "recovered after crash"
    assert checkpoint is not None and checkpoint.total_usage == expected_usage
    assert len(adapter.requests) == 2

    loaded = harness.sink.load_invocation(RUN_ID, LOGICAL_CALL_ID)
    assert loaded.ok and loaded.value is not None
    assert loaded.value.invocation.dispatch_attempt == 2
    assert loaded.value.invocation.receipt is not None
    assert loaded.value.invocation.receipt["attempts"] == 2
    assert loaded.value.invocation.receipt["usage"] == expected_usage


def test_required_evidence_retries_only_projection_and_deduplicates_usage(
    tmp_path: Path,
) -> None:
    harness = DeterministicFencedRunHarness()
    sink = _RejectModelEvidence(harness.sink, failures=2)
    usage = {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}
    adapter = _ScriptedAdapter(
        ModelTurn(final_text="durable answer", usage=usage, stop_reason="stop")
    )
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=token,
        model_evidence_policy="required",
        model_calls_file=True,
    )
    loop.open()
    try:
        first = loop.run_until_suspended("hello")
        first_checkpoint = loop.snapshot()
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    assert first.reason == "turn_failed"
    assert first.error_code == "evidence_uncommitted"
    assert first.retryable is True
    assert first_checkpoint is not None and first_checkpoint.total_usage == usage
    assert len(adapter.requests) == 1

    second, second_checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        first_checkpoint,
        user_input=None,
        sink=sink,
        model_evidence_policy="required",
        model_calls_file=True,
    )
    assert second.reason == "turn_failed", second
    assert second.error_code == "evidence_uncommitted"
    assert second_checkpoint is not None and second_checkpoint.total_usage == usage
    assert len(adapter.requests) == 1

    settled, final_checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        second_checkpoint,
        user_input=None,
        sink=sink,
        model_evidence_policy="required",
        model_calls_file=True,
    )
    assert settled.reason == "settled"
    assert settled.turn is not None and settled.turn.final_text == "durable answer"
    assert final_checkpoint is not None and final_checkpoint.total_usage == usage
    assert len(adapter.requests) == 1
    assert (RUN_ID, LOGICAL_CALL_ID, 3) in harness.sink._model_evidence
    transcript_path = next((tmp_path / "runs").rglob("transcript.jsonl"))
    model_turns = [
        record
        for line in transcript_path.read_text(encoding="utf-8").splitlines()
        if (record := json.loads(line)).get("kind") == "model_turn"
    ]
    assert [record["usage"] for record in model_turns] == [usage, {}]
    assert sum(record["usage"].get("total_tokens", 0) for record in model_turns) == 6
    model_calls_path = next((tmp_path / "runs").rglob("model_calls.jsonl"))
    model_calls = [
        json.loads(line)
        for line in model_calls_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(model_calls) == 1
    assert model_calls[0]["stop_reason"] == "stop"
    assert model_calls[0]["usage"] == usage
    events_path = next((tmp_path / "runs").rglob("events.jsonl"))
    failed_turns = [
        event
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if (event := json.loads(line)).get("type") == "turn.failed"
    ]
    assert [event["data"]["provider_usage"] for event in failed_turns] == [usage, {}]


def test_required_evidence_recovery_preserves_policy_and_identity_across_config_changes(
    tmp_path: Path,
) -> None:
    harness = DeterministicFencedRunHarness()
    sink = _RejectModelEvidence(harness.sink)
    adapter = _ScriptedAdapter(ModelTurn(final_text="durable answer", stop_reason="stop"))
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=token,
        model=ModelConfig(model="original-model"),
        model_evidence_policy="required",
    )
    loop.open()
    try:
        evidence_park = loop.run_until_suspended("hello")
        checkpoint = loop.snapshot()
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    assert evidence_park.error_code == "evidence_uncommitted"
    assert checkpoint is not None

    restored, final_checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        checkpoint,
        user_input=None,
        sink=sink,
        model=ModelConfig(model="changed-model"),
        model_evidence_policy="passive",
    )

    assert restored.reason == "settled"
    assert restored.turn is not None and restored.turn.final_text == "durable answer"
    assert final_checkpoint is not None
    assert len(adapter.requests) == 1
    assert (RUN_ID, LOGICAL_CALL_ID, 3) in harness.sink._model_evidence


def test_run_once_releases_required_evidence_park_for_sink_only_recovery(
    tmp_path: Path,
) -> None:
    harness = DeterministicFencedRunHarness()
    sink = _RejectModelEvidence(harness.sink)
    adapter = _ScriptedAdapter(ModelTurn(final_text="durable answer", stop_reason="stop"))
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=token,
        model_evidence_policy="required",
    )

    with pytest.raises(TurnNotSettled) as raised:
        loop.run_once("hello")

    assert raised.value.reason == "turn_failed"
    assert raised.value.suspension.error_code == "evidence_uncommitted"
    assert RUN_ID not in harness.sink._terminals
    loaded = harness.sink.latest_checked(RUN_ID)
    assert loaded.ok and loaded.value is not None
    checkpoint = loaded.value.checkpoint
    assert checkpoint.terminal is False
    assert checkpoint.last_suspension is not None
    assert checkpoint.last_suspension["error_code"] == "evidence_uncommitted"

    restored, final_checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        checkpoint,
        user_input=None,
        sink=sink,
        model_evidence_policy="passive",
    )

    assert restored.reason == "settled"
    assert restored.turn is not None and restored.turn.final_text == "durable answer"
    assert final_checkpoint is not None and final_checkpoint.terminal is False
    assert (RUN_ID, LOGICAL_CALL_ID, 3) in harness.sink._model_evidence
    assert len(adapter.requests) == 1


def test_required_evidence_recovery_applies_stored_final_before_dynamic_context(
    tmp_path: Path,
) -> None:
    harness = DeterministicFencedRunHarness()
    sink = _RejectModelEvidence(harness.sink)
    adapter = _ScriptedAdapter(ModelTurn(final_text="durable answer", stop_reason="stop"))
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=token,
        model_evidence_policy="required",
    )
    loop.open()
    try:
        evidence_park = loop.run_until_suspended("hello")
        checkpoint = loop.snapshot()
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    assert evidence_park.error_code == "evidence_uncommitted"
    assert checkpoint is not None

    restored, final_checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        checkpoint,
        user_input=None,
        sink=sink,
        model_evidence_policy="required",
        context_providers=(_RaisingDynamicContextProvider(),),
    )

    assert restored.reason == "settled"
    assert restored.turn is not None and restored.turn.final_text == "durable answer"
    assert final_checkpoint is not None
    assert final_checkpoint.messages[-1]["content"] == "durable answer"
    assert len(adapter.requests) == 1


def test_required_evidence_recovery_replays_multimodal_result_without_rebuilding_request(
    tmp_path: Path,
) -> None:
    harness = DeterministicFencedRunHarness()
    sink = _RejectModelEvidence(harness.sink)
    adapter = _ScriptedAdapter(ModelTurn(final_text="durable answer", stop_reason="stop"))
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=token,
        model_evidence_policy="required",
    )
    loop.open()
    try:
        evidence_park = loop.run_until_suspended(
            (
                TextPart("describe this"),
                ImagePart(source_ref="asset:image-1", mime_type="image/png"),
            )
        )
        checkpoint = loop.snapshot()
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    assert evidence_park.error_code == "evidence_uncommitted"
    assert checkpoint is not None
    assert adapter.requests[0].instruction == "describe this"
    assert isinstance(adapter.requests[0].messages[0]["content"], list)

    restored, _final_checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        checkpoint,
        user_input=None,
        sink=sink,
        model_evidence_policy="required",
    )

    assert restored.reason == "settled"
    assert restored.turn is not None and restored.turn.final_text == "durable answer"
    assert len(adapter.requests) == 1


def test_required_evidence_recovery_does_not_consume_an_extra_step(tmp_path: Path) -> None:
    harness = DeterministicFencedRunHarness()
    sink = _RejectModelEvidence(harness.sink)
    adapter = _ScriptedAdapter(ModelTurn(final_text="durable answer", stop_reason="stop"))
    limits = RunLimits(max_steps=1)
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=token,
        model_evidence_policy="required",
        limits=limits,
    )
    loop.open()
    try:
        evidence_park = loop.run_until_suspended("hello")
        checkpoint = loop.snapshot()
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    assert evidence_park.error_code == "evidence_uncommitted"
    assert checkpoint is not None and checkpoint.submit_local_step == 1

    restored, final_checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        checkpoint,
        user_input=None,
        sink=sink,
        model_evidence_policy="required",
        limits=limits,
    )

    assert restored.reason == "settled"
    assert restored.turn is not None and restored.turn.final_text == "durable answer"
    assert final_checkpoint is not None and final_checkpoint.session_step == 1
    assert len(adapter.requests) == 1


def test_required_evidence_recovery_bypasses_budget_exhausted_by_settled_call(
    tmp_path: Path,
) -> None:
    harness = DeterministicFencedRunHarness()
    sink = _RejectModelEvidence(harness.sink)
    usage = {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}
    adapter = _ScriptedAdapter(
        ModelTurn(final_text="durable answer", usage=usage, stop_reason="stop")
    )
    limits = RunLimits(max_total_tokens=1)
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=token,
        model_evidence_policy="required",
        limits=limits,
    )
    loop.open()
    try:
        evidence_park = loop.run_until_suspended("hello")
        checkpoint = loop.snapshot()
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    assert evidence_park.error_code == "evidence_uncommitted"
    assert checkpoint is not None and checkpoint.total_usage == usage

    restored, final_checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        checkpoint,
        user_input=None,
        sink=sink,
        model_evidence_policy="required",
        limits=limits,
    )

    assert restored.reason == "settled"
    assert restored.turn is not None and restored.turn.final_text == "durable answer"
    assert final_checkpoint is not None and final_checkpoint.total_usage == usage
    assert len(adapter.requests) == 1


def test_required_evidence_recovery_completes_before_a_requested_pause(
    tmp_path: Path,
) -> None:
    harness = DeterministicFencedRunHarness()
    sink = _RejectModelEvidence(harness.sink)
    adapter = _ScriptedAdapter(ModelTurn(final_text="durable answer", stop_reason="stop"))
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=token,
        model_evidence_policy="required",
    )
    loop.open()
    try:
        evidence_park = loop.run_until_suspended("hello")
        checkpoint = loop.snapshot()
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    assert evidence_park.error_code == "evidence_uncommitted"
    assert checkpoint is not None
    current = harness.claim_writer(RUN_ID, "worker-2")
    restored_loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=current,
        model_evidence_policy="required",
    )
    restored_loop.restore(checkpoint)
    restored_loop.pause_turn()
    try:
        settled = restored_loop.run_until_suspended(None)
        final_checkpoint = restored_loop.snapshot()
    finally:
        with suppress(BaseException):
            restored_loop.discard_uncommitted()

    assert settled.reason == "settled"
    assert settled.turn is not None and settled.turn.final_text == "durable answer"
    assert final_checkpoint is not None and final_checkpoint.last_suspension is not None
    assert final_checkpoint.last_suspension["reason"] == "settled"
    assert len(adapter.requests) == 1


def test_required_evidence_recovery_survives_interrupt_before_result_application(
    tmp_path: Path,
) -> None:
    harness = DeterministicFencedRunHarness()
    sink = _RejectModelEvidence(harness.sink)
    adapter = _ScriptedAdapter(ModelTurn(final_text="durable answer", stop_reason="stop"))
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=token,
        model_evidence_policy="required",
    )
    loop.open()
    try:
        evidence_park = loop.run_until_suspended("hello")
        checkpoint = loop.snapshot()
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    assert evidence_park.error_code == "evidence_uncommitted"
    assert checkpoint is not None
    current = harness.claim_writer(RUN_ID, "worker-2")
    restored_loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=current,
        model_evidence_policy="required",
    )
    restored_loop.restore(checkpoint)
    restored_loop.interrupt_turn()
    try:
        interrupted = restored_loop.run_until_suspended(None)
        interrupted_checkpoint = restored_loop.snapshot()
        settled = restored_loop.run_until_suspended(None)
    finally:
        with suppress(BaseException):
            restored_loop.discard_uncommitted()

    assert interrupted.reason == "interrupted"
    assert interrupted_checkpoint is not None
    assert interrupted_checkpoint.last_suspension is not None
    assert interrupted_checkpoint.last_suspension["reason"] == "interrupted"
    assert settled.reason == "settled"
    assert settled.turn is not None and settled.turn.final_text == "durable answer"
    assert len(adapter.requests) == 1


def test_projected_recovered_tool_calls_resume_after_context_interrupt(
    tmp_path: Path,
) -> None:
    harness = DeterministicFencedRunHarness()
    sink = _RejectModelEvidence(harness.sink)
    adapter = _ScriptedAdapter(
        ModelTurn(
            tool_calls=(ToolCall(id="missing-1", name="missing.tool", arguments={}),),
            stop_reason="tool_calls",
        ),
        ModelTurn(final_text="done after resumed tool", stop_reason="stop"),
    )
    first_token = harness.claim_writer(RUN_ID, "worker-1")
    first_loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=first_token,
        model_evidence_policy="required",
    )
    first_loop.open()
    try:
        evidence_park = first_loop.run_until_suspended("hello")
        evidence_checkpoint = first_loop.snapshot()
    finally:
        with suppress(BaseException):
            first_loop.discard_uncommitted()

    assert evidence_park.error_code == "evidence_uncommitted"
    assert evidence_checkpoint is not None
    interrupting_provider = _InterruptingDynamicContextProvider()
    second_token = harness.claim_writer(RUN_ID, "worker-2")
    interrupted_loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=second_token,
        model_evidence_policy="required",
        context_providers=(interrupting_provider,),
    )
    interrupting_provider.loop = interrupted_loop
    interrupted_loop.restore(evidence_checkpoint)
    try:
        interrupted = interrupted_loop.run_until_suspended(None)
        interrupted_checkpoint = interrupted_loop.snapshot()
        with pytest.raises(NativeAgentError) as raised:
            interrupted_loop.run_until_suspended("replace the unfinished tool turn")
    finally:
        with suppress(BaseException):
            interrupted_loop.discard_uncommitted()

    assert interrupted.reason == "interrupted"
    assert raised.value.error_code == "evidence_recovery_requires_resume"
    assert interrupted.model_tool_calls_pending is True
    assert interrupted_checkpoint is not None
    assert interrupted_checkpoint.last_suspension is not None
    assert interrupted_checkpoint.last_suspension["model_tool_calls_pending"] is True
    projected_tool_turns = [
        message
        for message in interrupted_checkpoint.messages
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    assert len(projected_tool_turns) == 1
    assert len(adapter.requests) == 1

    settled, final_checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        interrupted_checkpoint,
        user_input=None,
        sink=sink,
        model_evidence_policy="passive",
    )

    assert settled.reason == "settled"
    assert settled.turn is not None
    assert settled.turn.final_text == "done after resumed tool"
    assert final_checkpoint is not None
    assert final_checkpoint.last_suspension is not None
    assert final_checkpoint.last_suspension["model_tool_calls_pending"] is False
    final_tool_turns = [
        message
        for message in final_checkpoint.messages
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    assert len(final_tool_turns) == 1
    assert len(adapter.requests) == 2
    assert {observation.call_id for observation in adapter.requests[1].observations} == {
        "missing-1"
    }


def test_projected_recovered_tool_calls_skip_checkpointed_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = DeterministicFencedRunHarness()
    sink = _RejectModelEvidence(harness.sink)
    adapter = _ScriptedAdapter(
        ModelTurn(
            tool_calls=(
                ToolCall(id="missing-1", name="missing.tool", arguments={}),
                ToolCall(id="missing-2", name="missing.tool", arguments={}),
            ),
            stop_reason="tool_calls",
        ),
        ModelTurn(final_text="done after both tools", stop_reason="stop"),
    )
    first_token = harness.claim_writer(RUN_ID, "worker-1")
    first_loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=first_token,
        model_evidence_policy="required",
    )
    first_loop.open()
    try:
        evidence_park = first_loop.run_until_suspended("hello")
        evidence_checkpoint = first_loop.snapshot()
    finally:
        with suppress(BaseException):
            first_loop.discard_uncommitted()

    assert evidence_park.error_code == "evidence_uncommitted"
    assert evidence_checkpoint is not None
    second_token = harness.claim_writer(RUN_ID, "worker-2")
    interrupted_loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=second_token,
        model_evidence_policy="required",
    )
    interrupted_loop.restore(evidence_checkpoint)
    execute_tool_call = interrupted_loop._aexecute_tool_call

    async def interrupt_after_first_tool(**kwargs: Any):
        observation = await execute_tool_call(**kwargs)
        if kwargs["call_id"] == "missing-1":
            interrupted_loop.interrupt_turn()
        return observation

    monkeypatch.setattr(interrupted_loop, "_aexecute_tool_call", interrupt_after_first_tool)
    try:
        interrupted = interrupted_loop.run_until_suspended(None)
        interrupted_checkpoint = interrupted_loop.snapshot()
    finally:
        with suppress(BaseException):
            interrupted_loop.discard_uncommitted()

    assert interrupted.reason == "interrupted"
    assert interrupted_checkpoint is not None
    assert [
        observation["call_id"] for observation in interrupted_checkpoint.pending_observations
    ] == ["missing-1"]
    assert interrupted_checkpoint.total_tool_calls == 1

    settled, final_checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        interrupted_checkpoint,
        user_input=None,
        sink=sink,
        model_evidence_policy="passive",
    )

    assert settled.reason == "settled"
    assert final_checkpoint is not None
    assert final_checkpoint.total_tool_calls == 2
    assert len(adapter.requests) == 2
    assert [
        observation.call_id for observation in adapter.requests[1].observations
    ] == ["missing-1", "missing-2"]


def test_interrupted_evidence_recovery_preserves_tool_follow_up_request(
    tmp_path: Path,
) -> None:
    harness = DeterministicFencedRunHarness()
    sink = _RejectNthModelEvidence(harness.sink, reject_on=2)
    adapter = _ScriptedAdapter(
        ModelTurn(
            tool_calls=(ToolCall(id="missing-1", name="missing.tool", arguments={}),),
            stop_reason="tool_calls",
        ),
        ModelTurn(final_text="done after observation", stop_reason="stop"),
    )
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=token,
        model_evidence_policy="required",
    )
    loop.open()
    try:
        evidence_park = loop.run_until_suspended("hello")
        checkpoint = loop.snapshot()
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    assert evidence_park.error_code == "evidence_uncommitted"
    assert checkpoint is not None and checkpoint.pending_observations
    current = harness.claim_writer(RUN_ID, "worker-2")
    restored_loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=current,
        model_evidence_policy="required",
    )
    restored_loop.restore(checkpoint)
    restored_loop.interrupt_turn()
    try:
        interrupted = restored_loop.run_until_suspended(None)
        interrupted_checkpoint = restored_loop.snapshot()
        settled = restored_loop.run_until_suspended(None)
    finally:
        with suppress(BaseException):
            restored_loop.discard_uncommitted()

    assert interrupted.reason == "interrupted"
    assert interrupted_checkpoint is not None
    assert interrupted_checkpoint.pending_observations == checkpoint.pending_observations
    assert settled.reason == "settled"
    assert settled.turn is not None
    assert settled.turn.final_text == "done after observation"
    assert len(adapter.requests) == 2


def test_new_input_can_abandon_a_recovered_result_after_interrupt(tmp_path: Path) -> None:
    harness = DeterministicFencedRunHarness()
    sink = _RejectModelEvidence(harness.sink)
    adapter = _ScriptedAdapter(
        ModelTurn(final_text="interrupted answer", stop_reason="stop"),
        ModelTurn(final_text="replacement answer", stop_reason="stop"),
    )
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=token,
        model_evidence_policy="required",
    )
    loop.open()
    try:
        evidence_park = loop.run_until_suspended("hello")
        checkpoint = loop.snapshot()
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    assert evidence_park.error_code == "evidence_uncommitted"
    assert checkpoint is not None
    current = harness.claim_writer(RUN_ID, "worker-2")
    restored_loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=current,
        model_evidence_policy="required",
    )
    restored_loop.restore(checkpoint)
    restored_loop.interrupt_turn()
    try:
        interrupted = restored_loop.run_until_suspended(None)
        replacement = restored_loop.run_until_suspended("replace it")
    finally:
        with suppress(BaseException):
            restored_loop.discard_uncommitted()

    assert interrupted.reason == "interrupted"
    assert replacement.reason == "settled"
    assert replacement.turn is not None
    assert replacement.turn.final_text == "replacement answer"
    assert len(adapter.requests) == 2


@pytest.mark.parametrize(
    ("checkpoint_change", "expected_error_code"),
    [
        ({"remaining_duration_s": 0.0}, "run_timeout"),
        ({"cancellation_requested": True}, "cancelled"),
    ],
)
def test_required_evidence_commits_before_terminal_run_boundary(
    tmp_path: Path,
    checkpoint_change: dict[str, Any],
    expected_error_code: str,
) -> None:
    harness = DeterministicFencedRunHarness()
    sink = _RejectModelEvidence(harness.sink)
    adapter = _ScriptedAdapter(ModelTurn(final_text="durable answer", stop_reason="stop"))
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=token,
        model_evidence_policy="required",
    )
    loop.open()
    try:
        evidence_park = loop.run_until_suspended("hello")
        checkpoint = loop.snapshot()
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    assert evidence_park.error_code == "evidence_uncommitted"
    assert checkpoint is not None
    checkpoint = replace(checkpoint, **checkpoint_change)

    terminal, _terminal_checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        checkpoint,
        user_input=None,
        sink=sink,
        model_evidence_policy="required",
    )

    assert terminal.reason == "terminal"
    assert terminal.error_code == expected_error_code
    assert (RUN_ID, LOGICAL_CALL_ID, 3) in harness.sink._model_evidence
    assert len(adapter.requests) == 1


def test_required_evidence_recovery_preserves_tool_observations_without_replaying_provider(
    tmp_path: Path,
) -> None:
    harness = DeterministicFencedRunHarness()
    sink = _RejectNthModelEvidence(harness.sink, reject_on=2)
    adapter = _ScriptedAdapter(
        ModelTurn(
            tool_calls=(ToolCall(id="missing-1", name="missing.tool", arguments={}),),
            stop_reason="tool_calls",
        ),
        ModelTurn(final_text="done after observation", stop_reason="stop"),
    )
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=token,
        model_evidence_policy="required",
    )
    loop.open()
    try:
        evidence_park = loop.run_until_suspended("hello")
        checkpoint = loop.snapshot()
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    assert evidence_park.error_code == "evidence_uncommitted"
    assert checkpoint is not None and checkpoint.pending_observations
    assert len(adapter.requests) == 2

    restored, _final_checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        checkpoint,
        user_input=None,
        sink=sink,
        model_evidence_policy="required",
    )

    assert restored.reason == "settled"
    assert restored.turn is not None and restored.turn.final_text == "done after observation"
    assert len(adapter.requests) == 2


def test_required_evidence_park_rejects_new_input_without_mutating_recovery_state(
    tmp_path: Path,
) -> None:
    harness = DeterministicFencedRunHarness()
    sink = _RejectModelEvidence(harness.sink)
    adapter = _ScriptedAdapter(ModelTurn(final_text="durable answer", stop_reason="stop"))
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=token,
        model_evidence_policy="required",
    )
    loop.open()
    try:
        evidence_park = loop.run_until_suspended("hello")
        checkpoint = loop.snapshot()
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    assert evidence_park.error_code == "evidence_uncommitted"
    assert checkpoint is not None
    current = harness.claim_writer(RUN_ID, "worker-2")
    restored_loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=current,
        model_evidence_policy="required",
    )
    restored_loop.restore(checkpoint)
    try:
        with pytest.raises(NativeAgentError, match="before accepting new input") as raised:
            restored_loop.run_until_suspended("new input")
        unchanged = restored_loop.snapshot()
        settled = restored_loop.run_until_suspended(None)
    finally:
        with suppress(BaseException):
            restored_loop.discard_uncommitted()

    assert raised.value.error_code == "evidence_recovery_requires_resume"
    assert unchanged is not None and unchanged.messages == checkpoint.messages
    assert settled.reason == "settled"
    assert len(adapter.requests) == 1


def test_required_evidence_recovery_restores_provider_refusal_without_double_billing(
    tmp_path: Path,
) -> None:
    harness = DeterministicFencedRunHarness()
    sink = _RejectModelEvidence(harness.sink)
    refusal = ModelDispatchRefused(
        "provider refused",
        error_code="rate_limited",
        provider_error_code="rate_limit",
        retryable=True,
        http_status=429,
        config_recoverable=True,
    )
    usage = {"input_tokens": 5, "output_tokens": 0, "total_tokens": 5}
    mark_provider_usage(refusal, usage)
    adapter = _ScriptedAdapter(refusal)
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=token,
        model_evidence_policy="required",
        model_calls_file=True,
    )
    loop.open()
    try:
        evidence_park = loop.run_until_suspended("hello")
        checkpoint = loop.snapshot()
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    assert evidence_park.error_code == "evidence_uncommitted"
    assert checkpoint is not None and checkpoint.total_usage == usage

    restored, final_checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        checkpoint,
        user_input=None,
        sink=sink,
        model_evidence_policy="required",
        model_calls_file=True,
    )
    assert restored.reason == "turn_failed"
    assert restored.error_code == "rate_limited"
    assert restored.provider_error_code == "rate_limit"
    assert restored.http_status == 429
    assert final_checkpoint is not None and final_checkpoint.total_usage == usage
    assert len(adapter.requests) == 1
    model_calls_path = next((tmp_path / "runs").rglob("model_calls.jsonl"))
    model_calls = [
        json.loads(line)
        for line in model_calls_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(model_calls) == 1
    assert model_calls[0]["error_code"] == "rate_limited"
    assert model_calls[0]["usage"] == usage


def test_required_evidence_recovery_surfaces_retryable_refusal_without_paid_retry(
    tmp_path: Path,
) -> None:
    harness = DeterministicFencedRunHarness()
    sink = _RejectModelEvidence(harness.sink)
    refusal = ModelDispatchRefused("transient overload", retryable=True)
    mark_provider_usage(
        refusal,
        {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
    )
    adapter = _ScriptedAdapter(
        refusal,
        ModelTurn(
            final_text="second attempt",
            usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            stop_reason="stop",
        ),
    )
    model = ModelConfig(
        retry=ModelRetryConfig(
            layer="kernel",
            max_attempts=2,
            initial_delay_s=0.0,
            jitter_s=0.0,
        )
    )
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=token,
        model=model,
        model_evidence_policy="required",
    )
    loop.open()
    try:
        evidence_park = loop.run_until_suspended("hello")
        checkpoint = loop.snapshot()
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    assert evidence_park.error_code == "evidence_uncommitted"
    assert checkpoint is not None
    assert checkpoint.total_usage == {
        "input_tokens": 2,
        "output_tokens": 1,
        "total_tokens": 3,
    }

    restored, final_checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        checkpoint,
        user_input=None,
        sink=sink,
        model=model,
        model_evidence_policy="required",
    )
    assert restored.reason == "turn_failed"
    assert restored.error_code == "model_error"
    assert restored.retryable is True
    assert final_checkpoint is not None
    assert final_checkpoint.total_usage == {
        "input_tokens": 2,
        "output_tokens": 1,
        "total_tokens": 3,
    }
    assert len(adapter.requests) == 1


def test_outbox_policy_stages_evidence_with_the_settled_invocation(tmp_path: Path) -> None:
    harness = DeterministicFencedRunHarness()
    harness.sink.capabilities = replace(
        harness.sink.capabilities,
        transactional_outbox=True,
    )
    adapter = _ScriptedAdapter(ModelTurn(final_text="done", stop_reason="stop"))
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=harness.sink,
        writer_token=token,
        model_evidence_policy="outbox",
    )
    loop.open()
    try:
        settled = loop.run_until_suspended("hello")
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    assert settled.reason == "settled"
    assert len(adapter.requests) == 1
    assert (RUN_ID, LOGICAL_CALL_ID, 3) in harness.sink._model_evidence_outbox


def test_outbox_atomic_failure_publishes_neither_settlement_nor_evidence(tmp_path: Path) -> None:
    harness = DeterministicFencedRunHarness()
    harness.sink.capabilities = replace(
        harness.sink.capabilities,
        transactional_outbox=True,
    )
    adapter = _ScriptedAdapter(ModelTurn(final_text="done", stop_reason="stop"))
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=_RejectAtomicEvidenceStage(harness.sink),
        writer_token=token,
        model_evidence_policy="outbox",
    )
    loop.open()
    try:
        failed = loop.run_until_suspended("hello")
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    assert failed.reason == "terminal"
    assert failed.error_code == "dispatch_unknown"
    loaded = harness.sink.load_invocation(RUN_ID, LOGICAL_CALL_ID)
    assert loaded.ok and loaded.value is not None
    assert loaded.value.invocation.dispatch_state == "unknown"
    assert harness.sink._model_evidence_outbox == {}
    assert len(adapter.requests) == 1


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
    turn = ModelTurn(
        final_text="first",
        usage={
            "input_tokens": 4,
            "output_tokens": 2,
            "total_tokens": 6,
            "cache_creation_tokens": 1,
        },
    )
    adapter = _ScriptedAdapter(turn)
    baseline, settled = _crash_at(tmp_path, harness, adapter, "settled")
    result_sha = settled.invocation.result_ref.removeprefix("blob:")
    assert hashlib.sha256(settled.blob(result_sha)).hexdigest() == result_sha

    suspension, checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        baseline,
        sink=_TamperInvocationResultLoad(harness.sink),
    )

    assert suspension.reason == "terminal"
    assert suspension.error_code == "durable_invocation_result_corrupt"
    assert checkpoint is not None and checkpoint.total_usage == turn.usage
    assert len(adapter.requests) == 1


def test_stale_writer_cannot_expose_a_recovered_settlement(tmp_path: Path) -> None:
    harness = DeterministicFencedRunHarness()
    adapter = _ScriptedAdapter(ModelTurn(final_text="must stay fenced", stop_reason="stop"))
    baseline, _settled = _crash_at(tmp_path, harness, adapter, "settled")
    stale_token = WriterToken(run_id=RUN_ID, owner_id="worker-1", generation=1)
    harness.claim_writer(RUN_ID, "worker-2")
    loop = _loop(
        tmp_path,
        adapter,
        sink=harness.sink,
        writer_token=stale_token,
    )
    loop.restore(baseline)
    try:
        with pytest.raises(RuntimeError, match="checkpoint persistence failed"):
            loop.run_until_suspended("hello")
        assert len(adapter.requests) == 1
        assert loop._session is not None
        assert not any(
            message.get("role") == "assistant" for message in loop._session.state.messages
        )
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()


@pytest.mark.parametrize(
    "result_blob",
    [
        pytest.param(b"not-json", id="malformed-result"),
        pytest.param(
            durable_model_result_blob(ModelTurn(final_text="answer", stop_reason="length")),
            id="receipt-conflict",
        ),
    ],
)
def test_recovered_result_integrity_failure_carries_authoritative_usage(
    result_blob: bytes,
) -> None:
    usage = {
        "input_tokens": 5,
        "output_tokens": 3,
        "total_tokens": 8,
        "audio_tokens": 2,
    }
    hook = _RecoveredResultHook(
        result_blob=result_blob,
        receipt={"attempts": 1, "stop_reason": "stop", "usage": usage},
    )
    adapter = _ScriptedAdapter(ModelTurn(final_text="must not run"))
    runner = ModelCallRunner(adapter=adapter, lifecycle_hook=hook)

    with pytest.raises(DurableModelCallError) as caught:
        asyncio.run(
            runner.acall(
                ModelRequest(instruction="hello", system_prompt="system", tools=()),
                logical_call_id=LOGICAL_CALL_ID,
            )
        )

    assert caught.value.error_code == "durable_invocation_result_corrupt"
    assert provider_usage_of(caught.value) == usage
    assert adapter.requests == []


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
    with pytest.raises(AgentConfigError, match="need run_sink and writer_token"):
        AgentLoop(
            spec=_spec(tmp_path),
            model_adapter=adapter,
            runtime_config_provider=runtime_provider(runtime_config()),
            model_evidence_policy="required",
        )
    with pytest.raises(AgentConfigError, match="passive, required, or outbox"):
        AgentLoop(
            spec=_spec(tmp_path),
            model_adapter=adapter,
            runtime_config_provider=runtime_provider(runtime_config()),
            model_evidence_policy="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(AgentConfigError, match="transactional_outbox"):
        _loop(
            tmp_path,
            adapter,
            sink=harness.sink,
            writer_token=token,
            model_evidence_policy="outbox",
        )
    harness.sink.capabilities = replace(
        harness.sink.capabilities,
        durable_invocations=False,
    )
    with pytest.raises(AgentConfigError, match="durable_invocations"):
        _loop(tmp_path, adapter, sink=harness.sink, writer_token=token)
