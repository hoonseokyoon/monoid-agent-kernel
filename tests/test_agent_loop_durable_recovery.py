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

from monoid_agent_kernel.core.authority import ActivationWriteAuthority
from monoid_agent_kernel.core.content import ImagePart, TextPart
from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.model_invocation import (
    DurableModelInvocation,
    logical_model_call_id,
    model_dispatch_id,
)
from monoid_agent_kernel.core.model_content import read_model_content
from monoid_agent_kernel.core.model_stream import ModelStreamOutcome
from monoid_agent_kernel.core.outcome import InterruptionCause
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
    ToolObservation,
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
class _RevokeAfterInvocationCommit:
    inner: Any
    write_authority: ActivationWriteAuthority
    target_state: str = "settled"
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
        stage_evidence: bool = False,
    ) -> CommitResult:
        result = self.inner.commit_invocation(
            invocation,
            blobs,
            writer_token=writer_token,
            stage_evidence=stage_evidence,
        )
        if (
            self.armed
            and invocation.dispatch_state == self.target_state
            and result.status in {"committed", "already_committed"}
        ):
            self.armed = False
            self.write_authority.revoke()
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
    calls: int = 0

    @property
    def capabilities(self):
        return self.inner.capabilities

    def commit_model_evidence(self, *args: Any, **kwargs: Any) -> CommitResult:
        self.calls += 1
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
    model_content_file: bool = False,
    model_payload_file: bool = False,
    model_stream_observer_factories: tuple[Any, ...] = (),
    model_io_subscriptions: tuple[Any, ...] = (),
    tool_ids: tuple[str, ...] = (),
    cancellation_token: CancellationToken | None = None,
    write_authority: ActivationWriteAuthority | None = None,
) -> AgentLoop:
    return AgentLoop(
        spec=_spec(tmp_path, limits=limits),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config(*tool_ids, model=model)),
        run_sink=sink,
        writer_token=writer_token,
        checkpoint_persist_callback=checkpoint_persist_callback,
        invocation_context=invocation_context,
        model_evidence_policy=model_evidence_policy,  # type: ignore[arg-type]
        context_providers=context_providers,
        model_calls_file=model_calls_file,
        model_content_file=model_content_file,
        model_payload_file=model_payload_file,
        model_stream_observer_factories=model_stream_observer_factories,
        model_io_subscriptions=model_io_subscriptions,
        cancellation_token=cancellation_token,
        write_authority=write_authority or ActivationWriteAuthority(),
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
    tool_ids: tuple[str, ...] = (),
    model_io_subscriptions: tuple[Any, ...] = (),
    model_stream_observer_factories: tuple[Any, ...] = (),
    model_payload_file: bool = False,
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
        tool_ids=tool_ids,
        model_io_subscriptions=model_io_subscriptions,
        model_stream_observer_factories=model_stream_observer_factories,
        model_payload_file=model_payload_file,
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


@pytest.mark.parametrize("policy", ["passive", "required", "outbox"])
@pytest.mark.parametrize("refused", [False, True])
def test_fresh_v2_settlement_preserves_sparse_facts_after_process_loss(tmp_path: Path, policy, refused) -> None:
    from support.fenced_hosting import _record_digest
    from monoid_agent_kernel.core.model_facts import UsageObservation
    from monoid_agent_kernel.core.model_io import CapturePolicy, ModelIOSubscription
    from monoid_agent_kernel.providers._usage import mark_usage_observation

    harness = DeterministicFencedRunHarness()
    harness.sink.capabilities = replace(harness.sink.capabilities, transactional_outbox=True)
    usage = {"input_tokens": 0, "output_tokens": 2, "total_tokens": 2}
    observation = UsageObservation(reported={"input_tokens": 0, "output_tokens": 2}, source="adapter", collection="normal")
    result = ModelDispatchRefused("refused", error_code="provider_refused") if refused else ModelTurn(
        final_text="stored answer", usage=usage, usage_evidence=observation,
    )
    if refused:
        mark_provider_usage(result, usage)
        mark_usage_observation(result, observation)
    adapter = _ScriptedAdapter(result)
    baseline, stored = _crash_at(tmp_path, harness, adapter, "settled", model_evidence_policy=policy)
    payload = stored.invocation.to_json()
    assert payload["schema_version"] == "monoid.model-invocation.v2"
    evidence = payload["receipt"]["usage_evidence"]
    assert evidence["reported"] == {"input_tokens": 0, "output_tokens": 2}
    assert evidence["completeness"]["input_tokens"] == "complete"
    assert evidence["completeness"]["total_tokens"] == "unknown"
    assert evidence["attempts"][0]["source"] == "adapter"
    captures = []

    class Observer:
        def on_model_call(self, capture):
            captures.append(capture.receipt)

    coord = (RUN_ID, LOGICAL_CALL_ID, stored.revision)
    original_digest = harness.sink._invocations[coord][0]
    for _ in range(2):
        suspension, checkpoint = _restore(
            tmp_path, harness, adapter, baseline, model_evidence_policy=policy,
            model_io_subscriptions=(ModelIOSubscription(Observer(), CapturePolicy(mode="none")),),
        )
        assert suspension.reason == ("terminal" if refused else "settled")
        assert checkpoint.total_usage == usage
        assert checkpoint.last_model_invocation["receipt"]["usage_evidence"] == evidence
        assert captures[-1].usage_evidence.to_json() == evidence
        assert captures[-1].termination.to_json() == payload["receipt"]["termination"]
        assert len(adapter.requests) == 1
        assert harness.sink._invocations[coord][0] == original_digest
    if policy != "passive":
        delivered = harness.sink._model_evidence if policy == "required" else harness.sink._model_evidence_outbox
        assert len(delivered) == 1 and delivered[coord][0] == _record_digest(payload)


def test_required_evidence_obligation_survives_pre_checkpoint_settlement_crash(
    tmp_path: Path,
) -> None:
    harness = DeterministicFencedRunHarness()
    adapter = _ScriptedAdapter(ModelTurn(final_text="durable answer", stop_reason="stop"))
    baseline, settled = _crash_at(
        tmp_path,
        harness,
        adapter,
        "settled",
        model_evidence_policy="required",
    )
    evidence_key = (RUN_ID, LOGICAL_CALL_ID, settled.invocation.revision)

    assert settled.invocation.evidence_policy == "required"
    assert evidence_key not in harness.sink._model_evidence

    suspension, checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        baseline,
        model_evidence_policy="passive",
    )

    assert suspension.reason == "settled"
    assert suspension.turn is not None
    assert suspension.turn.final_text == "durable answer"
    assert checkpoint is not None
    assert len(adapter.requests) == 1
    assert evidence_key in harness.sink._model_evidence


def test_pre_checkpoint_required_evidence_commits_before_rebuilt_request_drift(
    tmp_path: Path,
) -> None:
    harness = DeterministicFencedRunHarness()
    adapter = _ScriptedAdapter(ModelTurn(final_text="durable answer", stop_reason="stop"))
    baseline, settled = _crash_at(
        tmp_path,
        harness,
        adapter,
        "settled",
        model=ModelConfig(model="original-model"),
        model_evidence_policy="required",
    )
    evidence_key = (RUN_ID, LOGICAL_CALL_ID, settled.invocation.revision)

    suspension, _checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        baseline,
        model=ModelConfig(model="replacement-model"),
        model_evidence_policy="passive",
    )

    assert suspension.reason == "terminal"
    assert suspension.error_code == "durable_invocation_request_conflict"
    assert evidence_key in harness.sink._model_evidence
    assert len(adapter.requests) == 1


def test_required_policy_cannot_upgrade_a_passive_settled_invocation(
    tmp_path: Path,
) -> None:
    harness = DeterministicFencedRunHarness()
    adapter = _ScriptedAdapter(ModelTurn(final_text="durable answer", stop_reason="stop"))
    baseline, settled = _crash_at(tmp_path, harness, adapter, "settled")

    suspension, _checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        baseline,
        model_evidence_policy="required",
    )

    assert settled.invocation.evidence_policy == "passive"
    assert suspension.reason == "terminal"
    assert suspension.error_code == "durable_invocation_evidence_policy_conflict"
    assert len(adapter.requests) == 1
    assert harness.sink._model_evidence == {}


def test_required_policy_cannot_upgrade_a_passive_reservation_before_provider_entry(
    tmp_path: Path,
) -> None:
    harness = DeterministicFencedRunHarness()
    adapter = _ScriptedAdapter(ModelTurn(final_text="must not run"))
    baseline, reserved = _crash_at(tmp_path, harness, adapter, "reserved")

    suspension, _checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        baseline,
        model_evidence_policy="required",
    )

    assert reserved.invocation.evidence_policy == "passive"
    assert suspension.reason == "terminal"
    assert suspension.error_code == "durable_invocation_evidence_policy_conflict"
    assert adapter.requests == []
    loaded = harness.sink.load_invocation(RUN_ID, LOGICAL_CALL_ID)
    assert loaded.ok and loaded.value is not None
    assert loaded.value.invocation == reserved.invocation


def test_required_policy_change_still_closes_a_started_passive_dispatch_as_unknown(
    tmp_path: Path,
) -> None:
    harness = DeterministicFencedRunHarness()
    adapter = _ScriptedAdapter(ModelTurn(final_text="must not run"))
    baseline, _started = _crash_at(tmp_path, harness, adapter, "dispatch_started")

    suspension, _checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        baseline,
        model_evidence_policy="required",
    )

    assert suspension.reason == "terminal"
    assert suspension.error_code == "dispatch_unknown"
    assert adapter.requests == []
    loaded = harness.sink.load_invocation(RUN_ID, LOGICAL_CALL_ID)
    assert loaded.ok and loaded.value is not None
    assert loaded.value.invocation.dispatch_state == "unknown"


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


@pytest.mark.parametrize(("with_v2_facts", "cancel_backoff"), [(False, False), (True, False), (True, True)])
def test_recovered_retryable_refusal_resumes_remaining_kernel_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_v2_facts: bool, cancel_backoff: bool,
) -> None:
    from monoid_agent_kernel.core.model_io import CapturePolicy, ModelIOSubscription

    captures = []

    class Observer:
        def on_model_call(self, capture):
            captures.append(capture.receipt)

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

    historical_coordinate = (RUN_ID, LOGICAL_CALL_ID, failed.revision)
    if with_v2_facts:
        from support.fenced_hosting import _record_digest
        from support.model_facts import facts
        from monoid_agent_kernel.core.model_invocation import decode_model_invocation

        payload = failed.invocation.to_json()
        payload["schema_version"] = "monoid.model-invocation.v2"
        payload["receipt"].update(facts(dict(failed.invocation.receipt["usage"]), outcome="failed"))
        checked = decode_model_invocation(payload)
        assert checked.status == "loaded"
        failed = replace(failed, invocation=checked.value)
        harness.sink._invocations[historical_coordinate] = (_record_digest(payload), failed)
    historical_digest = harness.sink._invocations[historical_coordinate][0]
    if cancel_backoff:
        from monoid_agent_kernel.errors import RunCancelled

        async def cancelled_wait(self, delay_s, deadline):
            raise RunCancelled("cancelled during recovered backoff")

        monkeypatch.setattr(ModelCallRunner, "_abackoff", cancelled_wait)

    suspension, checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        baseline,
        model=model,
        model_io_subscriptions=(ModelIOSubscription(Observer(), CapturePolicy(mode="none")),),
    )

    receipt = captures[-1]
    assert receipt.usage_evidence.attempts[0].observation.source == (
        "provider" if with_v2_facts else "legacy_ambiguous"
    )
    assert len(receipt.usage_evidence.attempts) == (1 if cancel_backoff else 2)
    assert receipt.termination.outcome == ("cancelled" if cancel_backoff else "completed")
    if with_v2_facts:
        assert receipt.usage_evidence.reported == failed.invocation.receipt["usage"]
        assert receipt.usage_evidence.completeness["input_tokens"] == (
            "complete" if cancel_backoff else "partial"
        )

    if cancel_backoff:
        assert suspension.reason == "terminal"
        assert suspension.error_code == "cancelled"
        assert checkpoint.total_usage == failed.invocation.receipt["usage"]
        assert len(adapter.requests) == 1
        assert harness.sink._invocation_heads[(RUN_ID, LOGICAL_CALL_ID)] == failed.revision
        assert harness.sink._invocations[historical_coordinate][0] == historical_digest
        return

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
    assert loaded.value.invocation.schema_version == "monoid.model-invocation.v1"
    assert harness.sink._invocations[historical_coordinate][0] == historical_digest


def test_recovered_v2_facts_reach_the_public_receipt_and_subscription() -> None:
    from support.model_facts import facts
    from monoid_agent_kernel.core.model_io import CapturePolicy, ModelIOSubscription

    usage = {"input_tokens": 2, "output_tokens": 0}
    wire_facts = facts(usage)
    hook = _RecoveredResultHook(
        result_blob=durable_model_result_blob(ModelTurn(final_text="answer", stop_reason="stop")),
        receipt={"attempts": 1, "stop_reason": "stop", "usage": usage, **wire_facts},
    )
    received = []

    class Observer:
        def on_model_call(self, capture):
            received.append(capture)

    adapter = _ScriptedAdapter(ModelTurn(final_text="must not run"))
    runner = ModelCallRunner(adapter=adapter, lifecycle_hook=hook, subscriptions=(
        ModelIOSubscription(observer=Observer(), policy=CapturePolicy(mode="none")),
    ))
    _, receipt = asyncio.run(runner.acall(
        ModelRequest(instruction="hello", system_prompt="system", tools=()),
        logical_call_id=LOGICAL_CALL_ID,
    ))
    assert receipt.to_json().get("termination") == wire_facts["termination"]
    assert receipt.to_json().get("usage_evidence") == wire_facts["usage_evidence"]
    assert received[0].receipt.usage_evidence == receipt.usage_evidence
    assert receipt.attempt_log == ()
    assert adapter.requests == []


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


def test_required_evidence_projection_failure_preserves_completed_model_stream(
    tmp_path: Path,
) -> None:
    class _Writer:
        def __init__(self) -> None:
            self.outcomes: list[ModelStreamOutcome] = []

        def push(self, _delta: Any) -> None:
            return None

        def close(self, outcome: ModelStreamOutcome) -> None:
            self.outcomes.append(outcome)

    class _Observer:
        def __init__(self, writer: _Writer) -> None:
            self.writer = writer

        def open(self, _context: Any) -> _Writer:
            return self.writer

    harness = DeterministicFencedRunHarness()
    usage = {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}
    adapter = _ScriptedAdapter(
        ModelTurn(final_text="durable answer", usage=usage, stop_reason="stop")
    )
    writer = _Writer()
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=_RejectModelEvidence(harness.sink),
        writer_token=token,
        model_evidence_policy="required",
        model_content_file=True,
        model_stream_observer_factories=(lambda: _Observer(writer),),
    )
    loop.open()
    try:
        evidence_park = loop.run_until_suspended("hello")
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    expected = ModelStreamOutcome(
        status="completed",
        final_text="durable answer",
        usage=usage,
    )
    assert evidence_park.error_code == "evidence_uncommitted"
    assert len(adapter.requests) == 1
    assert len(writer.outcomes) == 1
    outcome = writer.outcomes[0]
    assert replace(outcome, termination=None, usage_evidence=None) == expected
    assert outcome.termination.to_json() == {
        "outcome": "completed", "requested_by": None, "ended_by": "provider_terminal",
    }
    assert outcome.usage_evidence.completeness["input_tokens"] == "unknown"
    content_path = next((tmp_path / "runs").rglob("model-content.jsonl"))
    snapshots = read_model_content(content_path).snapshots
    assert len(snapshots) == 1
    assert snapshots[0].status == "completed"
    assert snapshots[0].final_text == "durable answer"
    assert snapshots[0].usage == usage


def test_post_settlement_interrupt_accounts_usage_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = DeterministicFencedRunHarness()
    usage = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    adapter = _ScriptedAdapter(
        ModelTurn(final_text="settled answer", usage=usage, stop_reason="stop")
    )
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(tmp_path, adapter, sink=harness.sink, writer_token=token)
    original_acall_model = loop._acall_model

    async def interrupt_after_settlement(*args: Any, **kwargs: Any):
        outcome = await original_acall_model(*args, **kwargs)
        loop.interrupt_turn()
        return outcome

    monkeypatch.setattr(loop, "_acall_model", interrupt_after_settlement)
    loop.open()
    try:
        interrupted = loop.run_until_suspended("hello")
        interrupted_checkpoint = loop.snapshot()
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    assert interrupted.reason == "interrupted"
    assert interrupted.interruption_cause is InterruptionCause.USER_CANCEL
    assert interrupted_checkpoint is not None
    assert interrupted_checkpoint.interruption_cause == InterruptionCause.USER_CANCEL.value
    assert interrupted_checkpoint.total_usage == usage
    assert not any(
        message.get("role") == "assistant" for message in interrupted_checkpoint.messages
    )

    settled, final_checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        interrupted_checkpoint,
        user_input=None,
    )

    assert settled.reason == "settled"
    assert settled.turn is not None and settled.turn.final_text == "settled answer"
    assert final_checkpoint is not None and final_checkpoint.total_usage == usage
    assert len(adapter.requests) == 1


def test_internal_safety_checkpoint_reuses_step_and_delivers_required_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = DeterministicFencedRunHarness()
    crash_after_settle = _CrashAfterInvocationCommit(
        harness.sink,
        target_state="settled",
    )
    adapter = _ScriptedAdapter(
        ModelTurn(final_text="settled after replay", stop_reason="stop"),
        ModelTurn(final_text="must not dispatch", stop_reason="stop"),
    )
    token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=crash_after_settle,
        writer_token=token,
        model_evidence_policy="required",
    )
    loop.open()
    session = loop._require_open()
    session.state.pending_tool_approval_replays = (
        {
            "call_name": "demo_approval",
            "call_id": "approved-1",
            "arguments": {"value": "approved"},
            "binding_id": "demo.approval",
            "tool_id": "demo.approval",
            "task_id": "task-1",
            "approval_key": "unused-by-injected-replay",
        },
    )

    async def replay_approved_call(*_args: Any, **_kwargs: Any) -> ToolObservation:
        return ToolObservation(
            call_id="tool_approval_replay:approved-1",
            tool_name="demo_approval",
            output={"type": "tool_approval_replay_result", "value": "approved"},
            is_background=True,
        )

    monkeypatch.setattr(loop, "_aexecute_tool_approval_replay", replay_approved_call)
    try:
        with pytest.raises(_HardCrash, match="^settled$"):
            loop.run_until_suspended("hello")
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    checkpoint_load = harness.sink.latest_checked(RUN_ID)
    assert checkpoint_load.ok and checkpoint_load.value is not None
    safety_checkpoint = checkpoint_load.value.checkpoint
    assert safety_checkpoint.session_step == 1
    assert safety_checkpoint.submit_local_step == 1
    assert safety_checkpoint.last_suspension is None
    assert safety_checkpoint.pending_tool_approval_replays == []

    settled_load = harness.sink.load_invocation(RUN_ID, LOGICAL_CALL_ID)
    assert settled_load.ok and settled_load.value is not None
    settled_invocation = settled_load.value.invocation
    evidence_key = (RUN_ID, LOGICAL_CALL_ID, settled_invocation.revision)
    assert settled_invocation.evidence_policy == "required"
    assert evidence_key not in harness.sink._model_evidence

    current = harness.claim_writer(RUN_ID, "worker-2")
    restored_loop = _loop(
        tmp_path,
        adapter,
        sink=harness.sink,
        writer_token=current,
        model_evidence_policy="required",
    )
    restored_loop.restore(safety_checkpoint)
    try:
        recovered = restored_loop.run_until_suspended(None)
    finally:
        with suppress(BaseException):
            restored_loop.discard_uncommitted()

    assert recovered.reason == "terminal"
    assert recovered.error_code == "durable_invocation_request_conflict"
    assert evidence_key in harness.sink._model_evidence
    assert len(adapter.requests) == 1
    next_call_id = logical_model_call_id(RUN_ID, "turn_0002")
    next_load = harness.sink.load_invocation(RUN_ID, next_call_id)
    assert next_load.status == "missing"


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
    assert settled.interruption_cause is None
    assert settled.turn is not None
    assert settled.turn.final_text == "done after resumed tool"
    assert final_checkpoint is not None
    assert final_checkpoint.interruption_cause == ""
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


def test_checkpointed_recovered_finish_restores_kernel_tool_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = DeterministicFencedRunHarness()
    sink = _RejectModelEvidence(harness.sink)
    adapter = _ScriptedAdapter(
        ModelTurn(
            tool_calls=(
                ToolCall(
                    id="finish-1",
                    name="run_finish",
                    arguments={
                        "summary": "durable finish",
                        "outputs": ["result.md"],
                        "notes": "kept across restore",
                    },
                ),
            ),
            stop_reason="tool_calls",
        )
    )
    first_token = harness.claim_writer(RUN_ID, "worker-1")
    first_loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=first_token,
        model_evidence_policy="required",
        tool_ids=("run.finish",),
    )
    first_loop.open()
    try:
        evidence_park = first_loop.run_until_suspended("finish the run")
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
        tool_ids=("run.finish",),
    )
    interrupted_loop.restore(evidence_checkpoint)
    execute_tool_call = interrupted_loop._aexecute_tool_call

    async def interrupt_after_finish(**kwargs: Any):
        observation = await execute_tool_call(**kwargs)
        interrupted_loop.interrupt_turn()
        return observation

    monkeypatch.setattr(interrupted_loop, "_aexecute_tool_call", interrupt_after_finish)
    try:
        interrupted = interrupted_loop.run_until_suspended(None)
        interrupted_checkpoint = interrupted_loop.snapshot()
    finally:
        with suppress(BaseException):
            interrupted_loop.discard_uncommitted()

    assert interrupted.reason == "interrupted"
    assert interrupted_checkpoint is not None
    assert interrupted_checkpoint.pending_finish == {
        "summary": "durable finish",
        "outputs": ["result.md"],
        "notes": "kept across restore",
    }
    assert len(adapter.requests) == 1

    settled, final_checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        interrupted_checkpoint,
        user_input=None,
        sink=sink,
        model_evidence_policy="passive",
        tool_ids=("run.finish",),
    )

    assert settled.reason == "settled"
    assert settled.turn is not None
    assert settled.turn.final_text == "durable finish"
    assert final_checkpoint is not None
    assert final_checkpoint.pending_finish is not None
    assert final_checkpoint.pending_finish["outputs"] == ["result.md"]
    assert final_checkpoint.total_tool_calls == 1
    assert len(adapter.requests) == 1


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


def test_lease_loss_precedes_required_evidence_recovery_mutation(tmp_path: Path) -> None:
    harness = DeterministicFencedRunHarness()
    sink = _RejectModelEvidence(harness.sink)
    adapter = _ScriptedAdapter(ModelTurn(final_text="durable answer", stop_reason="stop"))
    first_writer = harness.claim_writer(RUN_ID, "worker-1")
    first_loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=first_writer,
        model_evidence_policy="required",
    )
    first_loop.open()
    try:
        evidence_park = first_loop.run_until_suspended("hello")
        checkpoint = first_loop.snapshot()
    finally:
        with suppress(BaseException):
            first_loop.discard_uncommitted()

    assert evidence_park.error_code == "evidence_uncommitted"
    assert checkpoint is not None
    assert sink.calls == 1
    previous_checkpoint = harness.sink.latest_checked(RUN_ID)

    cancellation_token = CancellationToken()
    replacement_writer = harness.claim_writer(RUN_ID, "worker-2")
    restored_loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=replacement_writer,
        model_evidence_policy="required",
        cancellation_token=cancellation_token,
    )
    restored_loop.restore(checkpoint)
    cancellation_token.cancel(InterruptionCause.GRACEFUL_DRAIN)
    restored_loop.lose_writer_authority()
    try:
        suspension = restored_loop.run_until_suspended(None)
    finally:
        with suppress(BaseException):
            restored_loop.discard_uncommitted()

    assert suspension.reason == "interrupted"
    assert suspension.interruption_cause is InterruptionCause.LEASE_LOST
    assert cancellation_token.cause is InterruptionCause.GRACEFUL_DRAIN
    assert restored_loop.write_authority.revoked is True
    assert sink.calls == 1
    assert (RUN_ID, LOGICAL_CALL_ID, 3) not in harness.sink._model_evidence
    assert harness.sink.latest_checked(RUN_ID) == previous_checkpoint
    assert len(adapter.requests) == 1


def test_lease_loss_after_invocation_settle_blocks_all_stale_publication(
    tmp_path: Path,
) -> None:
    class _Writer:
        def __init__(self) -> None:
            self.outcomes: list[ModelStreamOutcome] = []

        def push(self, _delta: Any) -> None:
            return None

        def close(self, outcome: ModelStreamOutcome) -> None:
            self.outcomes.append(outcome)

    class _Observer:
        def __init__(self, writer: _Writer) -> None:
            self.writer = writer

        def open(self, _context: Any) -> _Writer:
            return self.writer

    harness = DeterministicFencedRunHarness()
    cancellation_token = CancellationToken()
    write_authority = ActivationWriteAuthority()
    sink = _RevokeAfterInvocationCommit(harness.sink, write_authority)
    usage = {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
    adapter = _ScriptedAdapter(
        ModelTurn(final_text="durable answer", usage=usage, stop_reason="stop")
    )
    writer = _Writer()
    writer_token = harness.claim_writer(RUN_ID, "worker-1")
    loop = _loop(
        tmp_path,
        adapter,
        sink=sink,
        writer_token=writer_token,
        cancellation_token=cancellation_token,
        write_authority=write_authority,
        model_calls_file=True,
        model_stream_observer_factories=(lambda: _Observer(writer),),
    )
    loop.open()
    try:
        suspension = loop.run_until_suspended("hello")
        assert loop._session is not None
        stale_total_usage = dict(loop._session.state.total_usage)
    finally:
        with suppress(BaseException):
            loop.discard_uncommitted()

    assert suspension.reason == "interrupted"
    assert suspension.interruption_cause is InterruptionCause.LEASE_LOST
    invocation = harness.sink.load_invocation(RUN_ID, LOGICAL_CALL_ID)
    assert invocation.ok and invocation.value is not None
    assert invocation.value.invocation.dispatch_state == "settled"
    assert invocation.value.invocation.receipt is not None
    assert invocation.value.invocation.receipt["usage"] == usage
    assert stale_total_usage == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    events_path = next((tmp_path / "runs").rglob("events.jsonl"))
    event_types = {
        json.loads(line)["type"]
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert "metrics.updated" not in event_types
    assert "model.turn.completed" not in event_types
    assert list((tmp_path / "runs").rglob("model_calls.jsonl")) == []
    assert writer.outcomes == []
    assert harness.sink.latest_checked(RUN_ID).status == "missing"
    assert harness.read_terminal(RUN_ID) is None


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from support.fenced_hosting import _record_digest
    from support.model_facts import facts
    from monoid_agent_kernel.core.model_invocation import decode_model_invocation
    from monoid_agent_kernel.core.model_facts import call_facts_of

    failures = []
    original_record = AgentLoop._record_failed_model_turn

    def record(self, *args, **kwargs):
        failures.append(kwargs["exc"])
        return original_record(self, *args, **kwargs)

    monkeypatch.setattr(AgentLoop, "_record_failed_model_turn", record)
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
        model_content_file=True,
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
    refusal_content_path = next((tmp_path / "runs").rglob("model-content.jsonl"))
    refusal_streams = read_model_content(refusal_content_path).snapshots
    assert len(refusal_streams) == 1
    assert refusal_streams[0].status == "failed"
    assert refusal_streams[0].error_code == "evidence_uncommitted"

    loaded = harness.sink.load_invocation(RUN_ID, LOGICAL_CALL_ID).value
    payload = loaded.invocation.to_json()
    payload["schema_version"] = "monoid.model-invocation.v2"
    payload["receipt"].update(facts(usage, outcome="failed"))
    checked = decode_model_invocation(payload)
    assert checked.status == "loaded"
    harness.sink._invocations[(RUN_ID, LOGICAL_CALL_ID, loaded.revision)] = (
        _record_digest(payload), replace(loaded, invocation=checked.value),
    )

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
    termination, evidence = call_facts_of(failures[-1])
    assert termination.to_json() == payload["receipt"]["termination"]
    assert evidence.to_json() == payload["receipt"]["usage_evidence"]
    assert provider_usage_of(failures[-1]) == {}
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


def test_outbox_reservation_survives_replacement_passive_activation(tmp_path: Path) -> None:
    harness = DeterministicFencedRunHarness()
    harness.sink.capabilities = replace(
        harness.sink.capabilities,
        transactional_outbox=True,
    )
    adapter = _ScriptedAdapter(ModelTurn(final_text="done", stop_reason="stop"))
    baseline, reserved = _crash_at(
        tmp_path,
        harness,
        adapter,
        "reserved",
        model_evidence_policy="outbox",
    )

    assert reserved.invocation.evidence_policy == "outbox"
    assert adapter.requests == []

    settled, _checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        baseline,
        model_evidence_policy="passive",
    )

    assert settled.reason == "settled"
    assert len(adapter.requests) == 1
    loaded = harness.sink.load_invocation(RUN_ID, LOGICAL_CALL_ID)
    assert loaded.ok and loaded.value is not None
    assert loaded.value.invocation.evidence_policy == "outbox"
    evidence_key = (RUN_ID, LOGICAL_CALL_ID, loaded.value.invocation.revision)
    assert evidence_key in harness.sink._model_evidence_outbox


def test_recovered_outbox_reservation_checks_capability_before_dispatch(tmp_path: Path) -> None:
    harness = DeterministicFencedRunHarness()
    harness.sink.capabilities = replace(
        harness.sink.capabilities,
        transactional_outbox=True,
    )
    adapter = _ScriptedAdapter(ModelTurn(final_text="must not dispatch", stop_reason="stop"))
    baseline, reserved = _crash_at(
        tmp_path,
        harness,
        adapter,
        "reserved",
        model_evidence_policy="outbox",
    )
    harness.sink.capabilities = replace(
        harness.sink.capabilities,
        transactional_outbox=False,
    )

    failed, _checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        baseline,
        model_evidence_policy="passive",
    )

    assert failed.reason == "terminal"
    assert failed.error_code == "durable_invocation_evidence_policy_conflict"
    assert adapter.requests == []
    loaded = harness.sink.load_invocation(RUN_ID, LOGICAL_CALL_ID)
    assert loaded.ok and loaded.value is not None
    assert loaded.value.invocation == reserved.invocation
    assert harness.sink._model_evidence_outbox == {}


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


@pytest.mark.parametrize("fault", ["integrity", "evidence_success", "evidence_refusal", "evidence_changed_request", "evidence_changed_model"])
def test_tampered_settled_result_is_blocked_before_provider_reentry(tmp_path: Path, fault, monkeypatch) -> None:
    from support.fenced_hosting import _record_digest
    from support.model_facts import facts
    from monoid_agent_kernel.core.model_invocation import decode_model_invocation
    from monoid_agent_kernel.core.model_io import CapturePolicy, ModelIOSubscription

    projection_only = fault != "integrity"
    refusal = fault == "evidence_refusal"
    changed_request = fault == "evidence_changed_request"
    changed_model = fault == "evidence_changed_model"
    recorded = []
    original_record = ModelCallRunner._record

    def record(self, receipt, request_preimage, turn):
        recorded.append((receipt, request_preimage))
        original_record(self, receipt, request_preimage, turn)

    monkeypatch.setattr(ModelCallRunner, "_record", record)
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
    provider_result = ModelDispatchRefused("refused", error_code="rate_limited", retryable=True) if refusal else turn
    if refusal:
        mark_provider_usage(provider_result, turn.usage)
    adapter = _ScriptedAdapter(provider_result)
    baseline, settled = _crash_at(tmp_path, harness, adapter, "settled")
    payload = settled.invocation.to_json()
    payload["schema_version"] = "monoid.model-invocation.v2"
    if projection_only:
        payload["evidence_policy"] = "required"
    payload["receipt"].update(facts(dict(settled.invocation.receipt["usage"]), outcome="failed" if refusal else "completed"))
    checked = decode_model_invocation(payload)
    assert checked.status == "loaded"
    coord = (RUN_ID, LOGICAL_CALL_ID, settled.revision)
    settled = replace(settled, invocation=checked.value)
    harness.sink._invocations[coord] = (_record_digest(payload), settled)
    outcomes = []
    captures = []

    class Observer:
        def open(self, context):
            return self

        def push(self, delta):
            pass

        def close(self, outcome):
            outcomes.append(outcome)

        def on_model_call(self, capture):
            captures.append(capture)

    result_sha = settled.invocation.result_ref.removeprefix("blob:")
    if not refusal:
        assert hashlib.sha256(settled.blob(result_sha)).hexdigest() == result_sha

    suspension, checkpoint = _restore(
        tmp_path,
        harness,
        adapter,
        baseline,
        user_input="different new input" if changed_request else "hello",
        model=ModelConfig(provider="openai", model="replacement-model", gateway_url="https://replacement.invalid") if changed_model else None,
        sink=_RejectModelEvidence(harness.sink) if projection_only else _TamperInvocationResultLoad(harness.sink),
        model_evidence_policy="required" if projection_only else "passive",
        model_stream_observer_factories=(Observer,),
        model_payload_file=True,
        model_io_subscriptions=(ModelIOSubscription(Observer(), CapturePolicy(mode="full")),),
    )

    assert suspension.reason == ("turn_failed" if projection_only else "terminal")
    assert suspension.error_code == ("evidence_uncommitted" if projection_only else "durable_invocation_result_corrupt")
    assert checkpoint is not None and checkpoint.total_usage == turn.usage
    assert len(adapter.requests) == 1
    assert len(outcomes) == 1
    assert harness.sink._invocations[coord][0] == _record_digest(payload)
    if changed_request or changed_model:
        # Current routing cannot describe the historical call. The close still
        # reports the evidence barrier while accounting retains the checked bill.
        assert outcomes[0].status == "failed"
        assert outcomes[0].error_code == "evidence_uncommitted"
        assert outcomes[0].final_text is None and outcomes[0].usage is None
        assert outcomes[0].termination is None and outcomes[0].usage_evidence is None
        assert captures == []
        assert recorded == []
        return
    assert outcomes[0].status == ("completed" if projection_only and not refusal else "failed")
    assert outcomes[0].error_code == ("evidence_uncommitted" if refusal else None if projection_only else "durable_invocation_result_corrupt")
    assert outcomes[0].usage == (None if projection_only and not refusal else turn.usage)
    assert outcomes[0].final_text is None
    assert outcomes[0].termination.ended_by == ("provider_error" if refusal else "provider_terminal" if projection_only else "local_failure")
    assert outcomes[0].usage_evidence.to_json() == payload["receipt"]["usage_evidence"]
    assert len(captures) == 1
    receipt = captures[0].receipt
    assert receipt.usage == turn.usage and receipt.attempts == 1
    assert receipt.usage_evidence.to_json() == payload["receipt"]["usage_evidence"]
    assert receipt.termination == outcomes[0].termination
    assert receipt.idempotency_key == payload["idempotency_key"]
    assert receipt.request_digest == payload["request_digest"]
    assert receipt.error_code == ("rate_limited" if refusal else "" if projection_only else "durable_invocation_result_corrupt")
    assert recorded[-1][1] is not None


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
        suspension = loop.run_until_suspended("hello")
        assert suspension.interruption_cause is InterruptionCause.LEASE_LOST
        assert loop.write_authority.revoked is True
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
@pytest.mark.parametrize("with_v2_facts", [False, True])
def test_recovered_result_integrity_failure_carries_authoritative_usage(
    result_blob: bytes, with_v2_facts: bool,
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
    if with_v2_facts:
        from support.model_facts import facts

        hook.receipt = {**hook.receipt, **facts(usage)}
    received = []
    adapter = _ScriptedAdapter(ModelTurn(final_text="must not run"))
    runner = ModelCallRunner(adapter=adapter, lifecycle_hook=hook, settled_sink=received.append)

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
    assert received[0].receipt.usage == usage
    if with_v2_facts:
        assert received[0].receipt.attempts == 1
        assert received[0].receipt.attempt_log == ()
        assert received[0].receipt.usage_evidence.to_json() == hook.receipt["usage_evidence"]
        assert received[0].receipt.termination.outcome == "failed"
        assert received[0].receipt.termination.ended_by == "local_failure"
        assert hook.receipt["termination"]["outcome"] == "completed"


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
        suspension = loop.run_until_suspended("hello")
        assert suspension.interruption_cause is InterruptionCause.LEASE_LOST
        assert loop.write_authority.revoked is True
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
        suspension = loop.run_until_suspended("hello")
        assert suspension.interruption_cause is InterruptionCause.LEASE_LOST
        assert loop.write_authority.revoked is True
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
