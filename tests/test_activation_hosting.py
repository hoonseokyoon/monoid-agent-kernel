from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from support.fenced_hosting import (
    DeterministicFencedRunHarness,
    DeterministicFencedRunSink,
)
from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.core.authority import (
    ActivationWriteAuthority,
    WriteAuthorityRevoked,
)
from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.checkpoint import (
    LocalFsCheckpointStore,
    RunCheckpoint,
    decode_checkpoint,
)
from monoid_agent_kernel.core.content import TextPart
from monoid_agent_kernel.core.events import EVENT_SCHEMA_VERSION, AgentEvent
from monoid_agent_kernel.core.interruption import InterruptionCause
from monoid_agent_kernel.core.outcome import RetryEligibility, TerminalOutcome
from monoid_agent_kernel.core.result import Suspension
from monoid_agent_kernel.core.spec import AgentRunSpec, RunLimits
from monoid_agent_kernel.errors import ModelAdapterError, NativeAgentError
from monoid_agent_kernel.hosting import CommitResult, WriterToken
from monoid_agent_kernel.hosting.activation import (
    ACTIVATION_COMMAND_SCHEMA_VERSION,
    ACTIVATION_RECEIPT_SCHEMA_VERSION,
    ActivationCommand,
    ActivationDriver,
    ActivationLoopConfigurationError,
    ActivationReceipt,
    ActivationRuntime,
    ResolvedActivationInput,
)
from monoid_agent_kernel.hosting.execution import FencedEventSink, FencedTerminalBridge
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.recorder import AgentRecorder


_PRIVATE_FINAL_TEXT = "private final text that must stay out of the activation receipt"


class _RetryableAdapter:
    def next_turn(self, request: Any) -> ModelTurn:
        del request
        raise ModelAdapterError(
            "private provider failure",
            error_code="provider_unavailable",
            retryable=True,
        )


@dataclass
class _CountingFinalAdapter:
    calls: int = 0

    def next_turn(self, request: Any) -> ModelTurn:
        del request
        self.calls += 1
        return ModelTurn(final_text=_PRIVATE_FINAL_TEXT)


class _ForbiddenAdapter:
    def next_turn(self, request: Any) -> ModelTurn:
        del request
        raise AssertionError("terminal recovery must not call the model")


@dataclass
class _CommitThenFaultEventSink:
    inner: DeterministicFencedRunSink
    faults_remaining: int = 2

    @property
    def capabilities(self):  # noqa: ANN201 - adapter facade preserves the exact capability value
        return self.inner.capabilities

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def append_event(self, event: AgentEvent, *, writer_token: WriterToken) -> CommitResult:
        result = self.inner.append_event(event, writer_token=writer_token)
        if event.type == "model.turn.finished" and self.faults_remaining:
            self.faults_remaining -= 1
            raise TimeoutError("ambiguous event commit")
        return result


@dataclass
class _CommitThenFaultTerminalSink:
    inner: DeterministicFencedRunSink
    faults_remaining: int = 2

    @property
    def capabilities(self):  # noqa: ANN201 - adapter facade preserves the exact capability value
        return self.inner.capabilities

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def settle_terminal(
        self,
        outcome: TerminalOutcome,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        result = self.inner.settle_terminal(outcome, writer_token=writer_token)
        if self.faults_remaining:
            self.faults_remaining -= 1
            raise TimeoutError("ambiguous terminal commit")
        return result


@dataclass
class _CaptureEventSink:
    fail: bool = False
    events: list[AgentEvent] | None = None

    def __post_init__(self) -> None:
        if self.events is None:
            self.events = []

    def emit(self, event: AgentEvent) -> None:
        if self.fail:
            raise RuntimeError("authoritative journal unavailable")
        assert self.events is not None
        self.events.append(event)

    def close(self) -> None:
        return None


class _MixedTaskBoundaryLoop:
    """Minimal loop double for a park containing local and external resume tasks."""

    def __init__(self) -> None:
        self.pump_calls = 0
        self.wait_calls = 0
        self._committed_boundary = False

    def run_until_suspended(self, user_input: object = None) -> Suspension:
        assert user_input is None
        self.pump_calls += 1
        if self.pump_calls == 2:
            self._committed_boundary = True
        return Suspension(
            reason="awaiting_tasks",
            status="completed",
            awaiting_task_ids=("external-task",),
            has_external=True,
        )

    def at_quiescent_park(self) -> bool:
        return self._committed_boundary

    def wait_for_pending_tasks(self, timeout_s: float) -> bool:
        assert timeout_s > 0
        self.wait_calls += 1
        return True


def _workspace(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _seed_checkpoint(
    tmp_path: Path,
    *,
    run_id: str,
    cancellation_requested: bool = False,
) -> tuple[DeterministicFencedRunHarness, WriterToken, RunCheckpoint, AgentRunSpec]:
    spec = AgentRunSpec(
        run_id=run_id,
        workspace_root=_workspace(tmp_path / "workspace"),
        run_root=tmp_path / "source-runs",
    )
    store = LocalFsCheckpointStore(spec.run_root)
    source = AgentLoop(
        spec=spec,
        model_adapter=_RetryableAdapter(),
        runtime_config_provider=runtime_provider(runtime_config("fs.write")),
        checkpoint_store=store,
    )
    source.open()
    assert source.run_until_suspended("resume after durable failure").reason == "turn_failed"
    source.release_parked()
    record = store.latest(run_id)
    assert record is not None
    checkpoint = record.checkpoint
    if cancellation_requested:
        checkpoint = replace(
            checkpoint,
            cancellation_requested=True,
            interruption_cause=InterruptionCause.USER_CANCEL.value,
        )

    harness = DeterministicFencedRunHarness()
    token = harness.claim_writer(run_id, "owner-a")
    committed = harness.sink.commit_checkpoint(checkpoint, {}, writer_token=token)
    assert committed.status == "committed"
    return harness, token, checkpoint, spec


def _seed_settled_checkpoint(
    tmp_path: Path,
    *,
    run_id: str,
) -> tuple[DeterministicFencedRunHarness, WriterToken, RunCheckpoint, AgentRunSpec]:
    spec = AgentRunSpec(
        run_id=run_id,
        workspace_root=_workspace(tmp_path / "workspace"),
        run_root=tmp_path / "source-runs",
    )
    store = LocalFsCheckpointStore(spec.run_root)
    source = AgentLoop(
        spec=spec,
        model_adapter=_CountingFinalAdapter(),
        runtime_config_provider=runtime_provider(runtime_config("fs.write")),
        checkpoint_store=store,
    )
    source.open()
    assert source.run_until_suspended("initial input").reason == "settled"
    source.release_parked()
    record = store.latest(run_id)
    assert record is not None
    harness = DeterministicFencedRunHarness()
    token = harness.claim_writer(run_id, "owner-a")
    assert (
        harness.sink.commit_checkpoint(record.checkpoint, {}, writer_token=token).status
        == "committed"
    )
    return harness, token, record.checkpoint, spec


def _command(
    checkpoint: RunCheckpoint,
    *,
    command_id: str = "command-1",
    command_sequence: int = 1,
) -> ActivationCommand:
    request_digest = canonical_sha256({"command_id": command_id})
    return ActivationCommand(
        run_id=checkpoint.run_id,
        command_id=command_id,
        command_sequence=command_sequence,
        kind="control",
        source_checkpoint_seq=checkpoint.seq,
        source_checkpoint_sha256=canonical_sha256(checkpoint.to_json()),
        request_digest=request_digest,
        payload_ref=f"blob:{request_digest}",
    )


def _loop_factory(
    tmp_path: Path,
    spec: AgentRunSpec,
    adapter: object,
    factory_calls: list[str],
    *,
    run_root_name: str = "replacement-runs",
    limits: RunLimits | None = None,
):
    def build(command: ActivationCommand, runtime: ActivationRuntime) -> AgentLoop:
        factory_calls.append(command.identity_sha256)
        return AgentLoop(
            spec=AgentRunSpec(
                run_id=command.run_id,
                workspace_root=spec.workspace_root,
                run_root=tmp_path / run_root_name,
                limits=limits or spec.limits,
            ),
            model_adapter=adapter,
            runtime_config_provider=runtime_provider(runtime_config("fs.write")),
            run_sink=runtime.run_sink,
            writer_token=runtime.writer_token,
            write_authority=runtime.write_authority,
            cancellation_token=runtime.cancellation_token,
            authoritative_event_sinks=(runtime.event_sink,),
            event_sequence_seed=runtime.event_sequence_seed,
            status_file=False,
        )

    return build


def _event(run_id: str, sequence: int, *, event_type: str = "run.resumed") -> AgentEvent:
    return AgentEvent(
        schema_version=EVENT_SCHEMA_VERSION,
        event_id=f"event-{sequence}",
        seq=sequence,
        run_id=run_id,
        timestamp="2026-08-24T00:00:00Z",
        type=event_type,
        data={"sequence": sequence},
    )


def _terminal(run_id: str, *, failed: bool = False) -> TerminalOutcome:
    return TerminalOutcome(
        run_id=run_id,
        kind="failed_terminal" if failed else "completed",
        retry_eligibility=(
            RetryEligibility.FORBIDDEN if failed else RetryEligibility.NOT_APPLICABLE
        ),
        checkpoint_seq=2,
        error_code="terminal_failure" if failed else "",
    )


def _public_event_payloads(
    harness: DeterministicFencedRunHarness,
    run_id: str,
    event_cursor: int,
) -> list[dict[str, Any]]:
    payloads = []
    for sequence in range(1, event_cursor + 1):
        event = harness.read_event(run_id, sequence)
        if event is not None:
            payloads.append(event.to_json())
    return payloads


def test_activation_command_is_strict_and_retry_stable() -> None:
    checkpoint = RunCheckpoint(run_id="strict-command", seq=3)
    command = _command(checkpoint)

    assert ActivationCommand.from_json(command.to_json()) == command
    assert command.identity_sha256 == ActivationCommand.from_json(command.to_json()).identity_sha256

    unknown = {**command.to_json(), "attempt": 2}
    with pytest.raises(ValueError, match="closed schema"):
        ActivationCommand.from_json(unknown)
    tampered = {**command.to_json(), "identity_sha256": "0" * 64}
    with pytest.raises(ValueError, match="identity digest mismatch"):
        ActivationCommand.from_json(tampered)

    zero_source = replace(
        command,
        source_checkpoint_seq=0,
        source_checkpoint_sha256=canonical_sha256(
            RunCheckpoint(run_id=command.run_id, seq=0).to_json()
        ),
    )
    assert ActivationCommand.from_json(zero_source.to_json()) == zero_source

    legacy = command.to_json()
    legacy["schema_version"] = ACTIVATION_COMMAND_SCHEMA_VERSION.replace(
        "monoid.", "native-agent-runner.", 1
    )
    assert ActivationCommand.from_json(legacy).to_json() == command.to_json()


def test_activation_replacement_returns_same_content_free_receipt(tmp_path: Path) -> None:
    harness, token, checkpoint, spec = _seed_checkpoint(
        tmp_path,
        run_id="activation-replacement",
    )
    command = _command(checkpoint)
    adapter = _CountingFinalAdapter()
    factory_calls: list[str] = []
    first = ActivationDriver(
        sink=harness.sink,
        writer_token=token,
        loop_factory=_loop_factory(tmp_path, spec, adapter, factory_calls),
    ).drive(command)

    assert first.boundary_reason == "settled"
    assert first.terminal is False
    assert first.event_cursor > 0
    assert first == ActivationReceipt.from_json(first.to_json())
    legacy_receipt = first.to_json()
    legacy_receipt["schema_version"] = ACTIVATION_RECEIPT_SCHEMA_VERSION.replace(
        "monoid.", "native-agent-runner.", 1
    )
    assert ActivationReceipt.from_json(legacy_receipt).to_json() == first.to_json()
    assert _PRIVATE_FINAL_TEXT not in json.dumps(first.to_json(), sort_keys=True)
    with pytest.raises(ValueError, match="closed schema"):
        ActivationReceipt.from_json({**first.to_json(), "final_text": _PRIVATE_FINAL_TEXT})
    assert adapter.calls == 1
    assert len(factory_calls) == 1
    stored = harness.sink.latest_checked(command.run_id).value
    assert stored is not None
    assert stored.checkpoint.last_suspension is not None
    assert stored.checkpoint.last_suspension["final_text"] == _PRIVATE_FINAL_TEXT
    public_events = _public_event_payloads(harness, command.run_id, first.event_cursor)
    assert _PRIVATE_FINAL_TEXT not in json.dumps(public_events, sort_keys=True)

    replacement_token = harness.claim_writer(command.run_id, "owner-b")

    def forbidden_factory(command: ActivationCommand, runtime: ActivationRuntime) -> AgentLoop:
        del command, runtime
        raise AssertionError("an applied command must bypass loop construction")

    duplicate = ActivationDriver(
        sink=harness.sink,
        writer_token=replacement_token,
        loop_factory=forbidden_factory,
    ).drive(command)

    assert duplicate == first
    assert adapter.calls == 1
    assert len(factory_calls) == 1
    assert harness.sink.latest_event_sequence(command.run_id) == first.event_cursor


def test_older_command_keeps_exact_receipt_after_later_input(tmp_path: Path) -> None:
    harness, token, checkpoint, spec = _seed_checkpoint(
        tmp_path,
        run_id="activation-historical-receipt",
    )
    first_command = _command(checkpoint)
    first = ActivationDriver(
        sink=harness.sink,
        writer_token=token,
        loop_factory=_loop_factory(
            tmp_path,
            spec,
            _CountingFinalAdapter(),
            [],
            run_root_name="first-command-runs",
        ),
    ).drive(first_command)
    first_head = harness.sink.latest_checked(first_command.run_id).value
    assert first_head is not None
    first_private_receipt = first_head.checkpoint.applied_input_receipts[
        first_command.checkpoint_marker
    ]
    assert first_private_receipt["checkpoint_sha256"] == first.checkpoint_sha256

    second_command = _command(
        first_head.checkpoint,
        command_id="command-2",
        command_sequence=2,
    )
    second_command = replace(second_command, kind="input")
    second = ActivationDriver(
        sink=harness.sink,
        writer_token=token,
        loop_factory=_loop_factory(
            tmp_path,
            spec,
            _CountingFinalAdapter(),
            [],
            run_root_name="second-command-runs",
        ),
        input_resolver=lambda observed: ResolvedActivationInput(
            request_digest=observed.request_digest,
            payload_ref=observed.payload_ref,
            parts=(TextPart("later private input"),),
        ),
    ).drive(second_command)

    assert second.checkpoint_seq > first.checkpoint_seq
    assert second.event_cursor > first.event_cursor
    duplicate = ActivationDriver(
        sink=harness.sink,
        writer_token=token,
        loop_factory=lambda command, runtime: (_ for _ in ()).throw(
            AssertionError((command, runtime))
        ),
        input_resolver=lambda command: (_ for _ in ()).throw(AssertionError(command)),
    ).drive(first_command)

    assert duplicate == first


def test_input_activation_resolves_private_payload_once(tmp_path: Path) -> None:
    harness, token, checkpoint, spec = _seed_settled_checkpoint(
        tmp_path,
        run_id="activation-private-input",
    )
    command = replace(_command(checkpoint), kind="input")
    adapter = _CountingFinalAdapter()
    resolver_calls = 0
    private_input = "private user input resolved from opaque storage"

    def resolver(observed: ActivationCommand) -> ResolvedActivationInput:
        nonlocal resolver_calls
        resolver_calls += 1
        assert observed == command
        resolved = ResolvedActivationInput(
            request_digest=observed.request_digest,
            payload_ref=observed.payload_ref,
            parts=(TextPart(private_input),),
        )
        assert private_input not in repr(resolved)
        return resolved

    receipt = ActivationDriver(
        sink=harness.sink,
        writer_token=token,
        loop_factory=_loop_factory(tmp_path, spec, adapter, []),
        input_resolver=resolver,
    ).drive(command)

    assert receipt.boundary_reason == "settled"
    assert resolver_calls == 1
    assert adapter.calls == 1
    assert private_input not in json.dumps(receipt.to_json(), sort_keys=True)
    record = harness.sink.latest_checked(command.run_id).value
    assert record is not None
    assert any(message.get("content") == private_input for message in record.checkpoint.messages)
    public_events = _public_event_payloads(harness, command.run_id, receipt.event_cursor)
    assert private_input not in json.dumps(public_events, sort_keys=True)

    replacement_token = harness.claim_writer(command.run_id, "owner-b")
    duplicate = ActivationDriver(
        sink=harness.sink,
        writer_token=replacement_token,
        loop_factory=lambda command, runtime: (_ for _ in ()).throw(
            AssertionError((command, runtime))
        ),
        input_resolver=lambda command: (_ for _ in ()).throw(AssertionError(command)),
    ).drive(command)

    assert duplicate == receipt
    assert resolver_calls == 1
    assert adapter.calls == 1


def test_input_activation_rejects_resolved_identity_mismatch(tmp_path: Path) -> None:
    harness, token, checkpoint, spec = _seed_settled_checkpoint(
        tmp_path,
        run_id="activation-input-mismatch",
    )
    command = replace(_command(checkpoint), kind="input")

    with pytest.raises(NativeAgentError) as raised:
        ActivationDriver(
            sink=harness.sink,
            writer_token=token,
            loop_factory=_loop_factory(tmp_path, spec, _ForbiddenAdapter(), []),
            input_resolver=lambda observed: ResolvedActivationInput(
                request_digest="0" * 64,
                payload_ref=observed.payload_ref,
                parts=(TextPart("private"),),
            ),
        ).drive(command)

    assert raised.value.error_code == "activation_payload_mismatch"


def test_input_activation_accepts_initial_checkpoint_and_requires_resolver(
    tmp_path: Path,
) -> None:
    run_id = "activation-initial-checkpoint"
    checkpoint = RunCheckpoint(run_id=run_id, seq=0)
    harness = DeterministicFencedRunHarness()
    token = harness.claim_writer(run_id, "owner-a")
    assert harness.sink.commit_checkpoint(checkpoint, {}, writer_token=token).status == "committed"
    command = replace(_command(checkpoint), kind="input")
    spec = AgentRunSpec(
        run_id=run_id,
        workspace_root=_workspace(tmp_path / "workspace"),
        run_root=tmp_path / "unused-source-runs",
    )
    factory_calls: list[str] = []

    with pytest.raises(NativeAgentError) as raised:
        ActivationDriver(
            sink=harness.sink,
            writer_token=token,
            loop_factory=_loop_factory(
                tmp_path,
                spec,
                _CountingFinalAdapter(),
                factory_calls,
            ),
        ).drive(command)

    assert raised.value.error_code == "activation_input_resolver_missing"
    assert factory_calls == []

    receipt = ActivationDriver(
        sink=harness.sink,
        writer_token=token,
        loop_factory=_loop_factory(
            tmp_path,
            spec,
            _CountingFinalAdapter(),
            factory_calls,
        ),
        input_resolver=lambda observed: ResolvedActivationInput(
            request_digest=observed.request_digest,
            payload_ref=observed.payload_ref,
            parts=(TextPart("first private input"),),
        ),
    ).drive(command)

    assert receipt.checkpoint_seq > 0
    assert receipt.boundary_reason == "settled"
    assert factory_calls == [command.identity_sha256]


def test_boundary_commit_crash_recovers_without_reexecution(tmp_path: Path) -> None:
    harness, token, checkpoint, spec = _seed_checkpoint(
        tmp_path,
        run_id="activation-boundary-crash",
    )
    command = _command(checkpoint)
    adapter = _CountingFinalAdapter()
    factory_calls: list[str] = []

    def crash_after_boundary(phase: str, observed: ActivationCommand) -> None:
        assert observed == command
        if phase == "boundary_committed":
            raise RuntimeError("simulated process crash")

    with pytest.raises(RuntimeError, match="simulated process crash"):
        ActivationDriver(
            sink=harness.sink,
            writer_token=token,
            loop_factory=_loop_factory(tmp_path, spec, adapter, factory_calls),
            fault_hook=crash_after_boundary,
        ).drive(command)

    replacement_token = harness.claim_writer(command.run_id, "owner-b")
    recovered = ActivationDriver(
        sink=harness.sink,
        writer_token=replacement_token,
        loop_factory=lambda command, runtime: (_ for _ in ()).throw(
            AssertionError((command, runtime))
        ),
    ).drive(command)

    assert recovered.command_identity_sha256 == command.identity_sha256
    assert recovered.boundary_reason == "settled"
    assert adapter.calls == 1
    assert len(factory_calls) == 1


def test_crash_before_restore_leaves_command_unapplied_for_replacement(tmp_path: Path) -> None:
    harness, token, checkpoint, spec = _seed_checkpoint(
        tmp_path,
        run_id="activation-before-restore-crash",
    )
    command = _command(checkpoint)
    adapter = _CountingFinalAdapter()
    factory_calls: list[str] = []

    def crash_before_restore(phase: str, observed: ActivationCommand) -> None:
        assert observed == command
        if phase == "before_restore":
            raise RuntimeError("crash before restore")

    with pytest.raises(RuntimeError, match="crash before restore"):
        ActivationDriver(
            sink=harness.sink,
            writer_token=token,
            loop_factory=_loop_factory(tmp_path, spec, adapter, factory_calls),
            fault_hook=crash_before_restore,
        ).drive(command)

    assert adapter.calls == 0
    assert factory_calls == []
    replacement_token = harness.claim_writer(command.run_id, "owner-b")
    receipt = ActivationDriver(
        sink=harness.sink,
        writer_token=replacement_token,
        loop_factory=_loop_factory(tmp_path, spec, adapter, factory_calls),
    ).drive(command)

    assert receipt.boundary_reason == "settled"
    assert adapter.calls == 1
    assert factory_calls == [command.identity_sha256]


def test_crash_before_return_reconstructs_committed_receipt(tmp_path: Path) -> None:
    harness, token, checkpoint, spec = _seed_checkpoint(
        tmp_path,
        run_id="activation-before-return-crash",
    )
    command = _command(checkpoint)
    adapter = _CountingFinalAdapter()

    def crash_before_return(phase: str, observed: ActivationCommand) -> None:
        assert observed == command
        if phase == "before_return":
            raise RuntimeError("response lost before return")

    with pytest.raises(RuntimeError, match="response lost before return"):
        ActivationDriver(
            sink=harness.sink,
            writer_token=token,
            loop_factory=_loop_factory(tmp_path, spec, adapter, []),
            fault_hook=crash_before_return,
        ).drive(command)

    assert adapter.calls == 1
    replacement_token = harness.claim_writer(command.run_id, "owner-b")
    receipt = ActivationDriver(
        sink=harness.sink,
        writer_token=replacement_token,
        loop_factory=lambda command, runtime: (_ for _ in ()).throw(
            AssertionError((command, runtime))
        ),
    ).drive(command)

    assert receipt.boundary_reason == "settled"
    assert adapter.calls == 1


def test_ambiguous_event_commit_replacement_reuses_settled_model_result(tmp_path: Path) -> None:
    harness, token, checkpoint, spec = _seed_checkpoint(
        tmp_path,
        run_id="activation-event-ambiguity",
    )
    command = _command(checkpoint)
    adapter = _CountingFinalAdapter()
    faulting_sink = _CommitThenFaultEventSink(harness.sink)

    with pytest.raises(WriteAuthorityRevoked):
        ActivationDriver(
            sink=faulting_sink,
            writer_token=token,
            loop_factory=_loop_factory(
                tmp_path,
                spec,
                adapter,
                [],
                run_root_name="faulting-process",
            ),
        ).drive(command)

    event_cursor_after_crash = harness.sink.latest_event_sequence(command.run_id)
    assert event_cursor_after_crash > 0
    assert adapter.calls == 1

    replacement_token = harness.claim_writer(command.run_id, "owner-b")
    receipt = ActivationDriver(
        sink=harness.sink,
        writer_token=replacement_token,
        loop_factory=_loop_factory(
            tmp_path,
            spec,
            adapter,
            [],
            run_root_name="replacement-process",
        ),
    ).drive(command)

    assert receipt.boundary_reason == "settled"
    assert receipt.event_cursor > event_cursor_after_crash
    assert adapter.calls == 1


def test_terminal_checkpoint_crash_is_settled_by_replacement(tmp_path: Path) -> None:
    harness, token, checkpoint, spec = _seed_checkpoint(
        tmp_path,
        run_id="activation-terminal-crash",
        cancellation_requested=True,
    )
    command = _command(checkpoint)

    def crash_before_terminal(phase: str, observed: ActivationCommand) -> None:
        assert observed == command
        if phase == "boundary_committed":
            raise RuntimeError("crash before terminal settlement")

    with pytest.raises(RuntimeError, match="crash before terminal settlement"):
        ActivationDriver(
            sink=harness.sink,
            writer_token=token,
            loop_factory=_loop_factory(tmp_path, spec, _ForbiddenAdapter(), []),
            fault_hook=crash_before_terminal,
        ).drive(command)

    assert harness.sink.read_terminal(command.run_id) is None
    replacement_token = harness.claim_writer(command.run_id, "owner-b")
    recovered = ActivationDriver(
        sink=harness.sink,
        writer_token=replacement_token,
        loop_factory=lambda command, runtime: (_ for _ in ()).throw(
            AssertionError((command, runtime))
        ),
    ).drive(command)

    assert recovered.terminal is True
    assert recovered.terminal_ref == f"terminal:{command.run_id}"
    assert recovered.outcome_kind == "cancelled"
    assert recovered.retry_eligibility is RetryEligibility.FORBIDDEN
    assert harness.sink.read_terminal(command.run_id) is not None


def test_terminal_commit_response_loss_reconstructs_canonical_winner(tmp_path: Path) -> None:
    harness, token, checkpoint, spec = _seed_checkpoint(
        tmp_path,
        run_id="activation-terminal-return-crash",
        cancellation_requested=True,
    )
    command = _command(checkpoint)

    def crash_after_terminal(phase: str, observed: ActivationCommand) -> None:
        assert observed == command
        if phase == "terminal_committed":
            raise RuntimeError("terminal response lost")

    with pytest.raises(RuntimeError, match="terminal response lost"):
        ActivationDriver(
            sink=harness.sink,
            writer_token=token,
            loop_factory=_loop_factory(tmp_path, spec, _ForbiddenAdapter(), []),
            fault_hook=crash_after_terminal,
        ).drive(command)

    winner = harness.sink.read_terminal(command.run_id)
    assert winner is not None
    replacement_token = harness.claim_writer(command.run_id, "owner-b")
    receipt = ActivationDriver(
        sink=harness.sink,
        writer_token=replacement_token,
        loop_factory=lambda command, runtime: (_ for _ in ()).throw(
            AssertionError((command, runtime))
        ),
    ).drive(command)

    assert receipt.terminal is True
    assert receipt.outcome_kind == winner.kind
    assert receipt.retry_eligibility is winner.retry_eligibility


def test_activation_preserves_terminal_tool_limit_receipt(tmp_path: Path) -> None:
    harness, token, checkpoint, spec = _seed_checkpoint(
        tmp_path,
        run_id="activation-terminal-tool-limit",
    )
    command = _command(checkpoint)
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                tool_calls=(
                    fake_tool_call("fs_write", {"path": "first.txt", "content": "one"}, "c1"),
                    fake_tool_call("fs_write", {"path": "second.txt", "content": "two"}, "c2"),
                )
            )
        ]
    )

    first = ActivationDriver(
        sink=harness.sink,
        writer_token=token,
        loop_factory=_loop_factory(
            tmp_path,
            spec,
            adapter,
            [],
            limits=RunLimits(max_tool_calls=1),
        ),
    ).drive(command)
    duplicate = ActivationDriver(
        sink=harness.sink,
        writer_token=token,
        loop_factory=lambda command, runtime: (_ for _ in ()).throw(
            AssertionError((command, runtime))
        ),
    ).drive(command)

    assert duplicate == first
    assert first.boundary_reason == "limited"
    assert first.state == "limited"
    assert first.terminal is True
    assert first.outcome_kind == "limited"
    assert first.retry_eligibility is RetryEligibility.FORBIDDEN
    assert first.error_code == "max_tool_calls_exceeded"
    assert harness.sink.read_terminal(command.run_id) is not None


def test_activation_rejects_stale_source_before_constructing_loop(tmp_path: Path) -> None:
    harness, token, checkpoint, _ = _seed_checkpoint(
        tmp_path,
        run_id="activation-stale-source",
    )
    command = replace(_command(checkpoint), source_checkpoint_sha256="0" * 64)

    with pytest.raises(NativeAgentError) as raised:
        ActivationDriver(
            sink=harness.sink,
            writer_token=token,
            loop_factory=lambda command, runtime: (_ for _ in ()).throw(
                AssertionError((command, runtime))
            ),
        ).drive(command)

    assert raised.value.error_code == "activation_source_mismatch"


@pytest.mark.parametrize("status", ("corrupt", "unsupported_version"))
def test_activation_refuses_unreadable_checkpoint_head(tmp_path: Path, status: str) -> None:
    harness, token, checkpoint, _ = _seed_checkpoint(
        tmp_path,
        run_id=f"activation-load-{status}",
    )
    command = _command(checkpoint)
    harness.inject_authoritative_load_fault("checkpoint", command.run_id, status)

    with pytest.raises(NativeAgentError) as raised:
        ActivationDriver(
            sink=harness.sink,
            writer_token=token,
            loop_factory=lambda command, runtime: (_ for _ in ()).throw(
                AssertionError((command, runtime))
            ),
        ).drive(command)

    assert raised.value.error_code == f"checkpoint_{status}"


def test_activation_refuses_missing_checkpoint() -> None:
    harness = DeterministicFencedRunHarness()
    run_id = "activation-missing"
    token = harness.claim_writer(run_id, "owner-a")
    command = ActivationCommand(
        run_id=run_id,
        command_id="command-1",
        command_sequence=1,
        kind="control",
        source_checkpoint_seq=1,
        source_checkpoint_sha256="0" * 64,
        request_digest="1" * 64,
        payload_ref=f"blob:{'1' * 64}",
    )

    with pytest.raises(NativeAgentError) as raised:
        ActivationDriver(
            sink=harness.sink,
            writer_token=token,
            loop_factory=lambda command, runtime: (_ for _ in ()).throw(
                AssertionError((command, runtime))
            ),
        ).drive(command)

    assert raised.value.error_code == "checkpoint_missing"


def test_activation_continues_matching_internal_checkpoint(tmp_path: Path) -> None:
    harness, token, checkpoint, spec = _seed_checkpoint(
        tmp_path,
        run_id="activation-running-marker",
    )
    command = _command(checkpoint)
    internal = replace(
        checkpoint,
        seq=checkpoint.seq + 1,
        last_suspension=None,
        active_input={
            "input_id": command.checkpoint_marker,
            "source_seq": command.source_checkpoint_seq,
            "phase": "running",
        },
    )
    assert harness.sink.commit_checkpoint(internal, {}, writer_token=token).status == "committed"
    adapter = _CountingFinalAdapter()

    receipt = ActivationDriver(
        sink=harness.sink,
        writer_token=token,
        loop_factory=_loop_factory(tmp_path, spec, adapter, []),
    ).drive(command)

    assert receipt.checkpoint_seq > internal.seq
    assert receipt.boundary_reason == "settled"
    assert adapter.calls == 1


def test_activation_rejects_competing_running_marker(tmp_path: Path) -> None:
    harness, token, checkpoint, _ = _seed_checkpoint(
        tmp_path,
        run_id="activation-competing-marker",
    )
    command = _command(checkpoint)
    internal = replace(
        checkpoint,
        seq=checkpoint.seq + 1,
        last_suspension=None,
        active_input={
            "input_id": "monoid.activation/" + "f" * 64,
            "source_seq": command.source_checkpoint_seq,
            "phase": "running",
        },
    )
    assert harness.sink.commit_checkpoint(internal, {}, writer_token=token).status == "committed"

    with pytest.raises(NativeAgentError) as raised:
        ActivationDriver(
            sink=harness.sink,
            writer_token=token,
            loop_factory=lambda command, runtime: (_ for _ in ()).throw(
                AssertionError((command, runtime))
            ),
        ).drive(command)

    assert raised.value.error_code == "prior_activation_incomplete"


def test_activation_driver_rejects_loop_without_exact_host_bindings(tmp_path: Path) -> None:
    harness, token, checkpoint, spec = _seed_checkpoint(
        tmp_path,
        run_id="activation-loop-binding",
    )
    command = _command(checkpoint)

    def unbound_factory(command: ActivationCommand, runtime: ActivationRuntime) -> AgentLoop:
        del runtime
        return AgentLoop(
            spec=AgentRunSpec(
                run_id=command.run_id,
                workspace_root=spec.workspace_root,
                run_root=tmp_path / "unbound-runs",
            ),
            model_adapter=_ForbiddenAdapter(),
            runtime_config_provider=runtime_provider(runtime_config("fs.write")),
        )

    with pytest.raises(RuntimeError, match="exact write authority"):
        ActivationDriver(
            sink=harness.sink,
            writer_token=token,
            loop_factory=unbound_factory,
        ).drive(command)


def test_activation_driver_rejects_duplicate_authoritative_event_sink(tmp_path: Path) -> None:
    harness, token, checkpoint, spec = _seed_checkpoint(
        tmp_path,
        run_id="activation-duplicate-event-sink",
    )
    command = _command(checkpoint)

    def duplicate_factory(command: ActivationCommand, runtime: ActivationRuntime) -> AgentLoop:
        return AgentLoop(
            spec=AgentRunSpec(
                run_id=command.run_id,
                workspace_root=spec.workspace_root,
                run_root=tmp_path / "duplicate-event-runs",
            ),
            model_adapter=_ForbiddenAdapter(),
            runtime_config_provider=runtime_provider(runtime_config("fs.write")),
            run_sink=runtime.run_sink,
            writer_token=runtime.writer_token,
            write_authority=runtime.write_authority,
            cancellation_token=runtime.cancellation_token,
            authoritative_event_sinks=(runtime.event_sink, runtime.event_sink),
            event_sequence_seed=runtime.event_sequence_seed,
            status_file=False,
        )

    with pytest.raises(RuntimeError, match="exactly once as authoritative"):
        ActivationDriver(
            sink=harness.sink,
            writer_token=token,
            loop_factory=duplicate_factory,
        ).drive(command)


def test_activation_driver_requires_exact_cancellation_token(tmp_path: Path) -> None:
    harness, token, checkpoint, spec = _seed_checkpoint(
        tmp_path,
        run_id="activation-cancellation-token",
    )
    command = _command(checkpoint)

    def wrong_token_factory(command: ActivationCommand, runtime: ActivationRuntime) -> AgentLoop:
        return AgentLoop(
            spec=AgentRunSpec(
                run_id=command.run_id,
                workspace_root=spec.workspace_root,
                run_root=tmp_path / "wrong-token-runs",
            ),
            model_adapter=_ForbiddenAdapter(),
            runtime_config_provider=runtime_provider(runtime_config("fs.write")),
            run_sink=runtime.run_sink,
            writer_token=runtime.writer_token,
            write_authority=runtime.write_authority,
            cancellation_token=CancellationToken(),
            authoritative_event_sinks=(runtime.event_sink,),
            event_sequence_seed=runtime.event_sequence_seed,
            status_file=False,
        )

    with pytest.raises(ActivationLoopConfigurationError, match="exact cancellation token"):
        ActivationDriver(
            sink=harness.sink,
            writer_token=token,
            loop_factory=wrong_token_factory,
        ).drive(command)


def test_activation_driver_waits_for_committed_mixed_task_boundary() -> None:
    harness = DeterministicFencedRunHarness()
    token = harness.claim_writer("mixed-task-boundary", "owner-a")
    driver = ActivationDriver(
        sink=harness.sink,
        writer_token=token,
        loop_factory=lambda command, runtime: (_ for _ in ()).throw(
            AssertionError((command, runtime))
        ),
    )
    loop = _MixedTaskBoundaryLoop()

    suspension = driver._drive_to_durable_boundary(loop, None)  # type: ignore[arg-type]

    assert suspension.reason == "awaiting_tasks"
    assert suspension.has_external is True
    assert loop.wait_calls == 1
    assert loop.pump_calls == 2


def test_activation_driver_rejects_legacy_delta_channel(tmp_path: Path) -> None:
    harness, token, checkpoint, spec = _seed_checkpoint(
        tmp_path,
        run_id="activation-legacy-deltas",
    )
    command = _command(checkpoint)

    def delta_factory(command: ActivationCommand, runtime: ActivationRuntime) -> AgentLoop:
        loop = _loop_factory(tmp_path, spec, _ForbiddenAdapter(), [])(command, runtime)
        loop.emit_output_deltas = True
        return loop

    with pytest.raises(RuntimeError, match="durable stream channel"):
        ActivationDriver(
            sink=harness.sink,
            writer_token=token,
            loop_factory=delta_factory,
        ).drive(command)


def test_fenced_event_sink_enforces_cursor_content_and_authority() -> None:
    harness = DeterministicFencedRunHarness()
    run_id = "event-bridge"
    token = harness.claim_writer(run_id, "owner-a")
    authority = ActivationWriteAuthority()
    sink = FencedEventSink(harness.sink, token, authority)

    sink.emit(_event(run_id, 1))
    assert sink.last_sequence == 1
    replacement = FencedEventSink(
        harness.sink,
        token,
        ActivationWriteAuthority(),
        last_sequence=0,
    )
    replacement.emit(_event(run_id, 1))
    assert replacement.last_sequence == 1
    conflicting = FencedEventSink(
        harness.sink,
        token,
        ActivationWriteAuthority(),
        last_sequence=0,
    )
    conflict_authority = conflicting.write_authority
    with pytest.raises(WriteAuthorityRevoked) as conflict:
        conflicting.emit(replace(_event(run_id, 1), data={"winner": False}))
    assert isinstance(conflict.value.__cause__, NativeAgentError)
    assert conflict.value.__cause__.error_code == "event_conflict"
    assert conflict_authority.revoked is True
    with pytest.raises(NativeAgentError) as delta:
        replacement.emit(_event(run_id, 2, event_type="model.output.delta"))
    assert delta.value.error_code == "private_event_content"

    harness.claim_writer(run_id, "owner-b")
    with pytest.raises(WriteAuthorityRevoked):
        sink.emit(_event(run_id, 2))
    assert authority.revoked is True


def test_fenced_event_sink_reconciles_one_ambiguous_commit() -> None:
    harness = DeterministicFencedRunHarness()
    run_id = "event-bridge-ambiguity"
    token = harness.claim_writer(run_id, "owner-a")
    faulting = _CommitThenFaultEventSink(harness.sink, faults_remaining=1)
    bridge = FencedEventSink(faulting, token, ActivationWriteAuthority())

    bridge.emit(_event(run_id, 1, event_type="model.turn.finished"))

    assert bridge.last_sequence == 1
    assert harness.sink.latest_event_sequence(run_id) == 1


def test_terminal_bridge_returns_canonical_winner_and_closes_event_journal() -> None:
    harness = DeterministicFencedRunHarness()
    run_id = "terminal-bridge"
    token = harness.claim_writer(run_id, "owner-a")
    event = _event(run_id, 1)
    assert harness.sink.append_event(event, writer_token=token).status == "committed"
    winner = _terminal(run_id)
    assert harness.sink.settle_terminal(winner, writer_token=token).status == "committed"

    bridge = FencedTerminalBridge(harness.sink, token, ActivationWriteAuthority())
    settlement = bridge.settle(_terminal(run_id, failed=True))

    assert settlement.status == "conflict"
    assert settlement.outcome == winner
    assert settlement.terminal_ref == f"terminal:{run_id}"
    assert harness.sink.append_event(event, writer_token=token).status == "already_committed"
    assert harness.sink.append_event(_event(run_id, 2), writer_token=token).status == "conflict"


def test_terminal_bridge_reads_winner_after_repeated_response_loss() -> None:
    harness = DeterministicFencedRunHarness()
    run_id = "terminal-bridge-ambiguity"
    token = harness.claim_writer(run_id, "owner-a")
    outcome = _terminal(run_id)
    authority = ActivationWriteAuthority()
    faulting = _CommitThenFaultTerminalSink(harness.sink)

    settlement = FencedTerminalBridge(faulting, token, authority).settle(outcome)

    assert settlement.status == "already_committed"
    assert settlement.outcome == outcome
    assert authority.revoked is False


def test_event_projection_is_durable_first_and_uses_authoritative_seed(tmp_path: Path) -> None:
    failing = _CaptureEventSink(fail=True)
    recorder = AgentRecorder(
        run_root=tmp_path / "failed-projection",
        run_id="durable-first",
        status_file=False,
        authoritative_event_sinks=(failing,),
    )
    with pytest.raises(RuntimeError, match="authoritative journal unavailable"):
        recorder.event_bus.emit("run.resumed")
    assert (recorder.run_dir / "events.jsonl").read_text(encoding="utf-8") == ""
    recorder.close()

    capture = _CaptureEventSink()
    seeded = AgentRecorder(
        run_root=tmp_path / "seeded-projection",
        run_id="seeded-events",
        status_file=False,
        authoritative_event_sinks=(capture,),
        event_sequence_seed=7,
    )
    event = seeded.event_bus.emit("run.resumed")
    seeded.close()

    assert event.seq == 8
    assert capture.events is not None and capture.events[0].seq == 8
    local = json.loads((seeded.run_dir / "events.jsonl").read_text(encoding="utf-8"))
    assert local["seq"] == 8


def test_optional_commit_evidence_remains_compatible() -> None:
    harness = DeterministicFencedRunHarness()
    run_id = "optional-evidence"
    token = harness.claim_writer(run_id, "owner-a")

    class EvidenceElidingSink:
        capabilities = harness.sink.capabilities

        def __getattr__(self, name: str) -> Any:
            return getattr(harness.sink, name)

        def append_event(self, event: AgentEvent, *, writer_token: WriterToken) -> CommitResult:
            result = harness.sink.append_event(event, writer_token=writer_token)
            return CommitResult(status=result.status)

        def settle_terminal(
            self,
            outcome: TerminalOutcome,
            *,
            writer_token: WriterToken,
        ) -> CommitResult:
            result = harness.sink.settle_terminal(outcome, writer_token=writer_token)
            return CommitResult(status=result.status)

    sink = EvidenceElidingSink()
    event_bridge = FencedEventSink(sink, token, ActivationWriteAuthority())
    event_bridge.emit(_event(run_id, 1))
    terminal = _terminal(run_id)
    settlement = FencedTerminalBridge(sink, token, ActivationWriteAuthority()).settle(terminal)

    assert event_bridge.last_sequence == 1
    assert settlement.status == "committed"
    assert settlement.outcome == terminal


def test_invalid_commit_evidence_revokes_activation_authority() -> None:
    harness = DeterministicFencedRunHarness()

    class InvalidEventEvidenceSink:
        capabilities = harness.sink.capabilities

        def __getattr__(self, name: str) -> Any:
            return getattr(harness.sink, name)

        def append_event(self, event: AgentEvent, *, writer_token: WriterToken) -> CommitResult:
            result = harness.sink.append_event(event, writer_token=writer_token)
            return CommitResult(status=result.status, sequence=event.seq + 1)

    event_run_id = "invalid-event-evidence"
    event_token = harness.claim_writer(event_run_id, "owner-a")
    event_authority = ActivationWriteAuthority()
    with pytest.raises(WriteAuthorityRevoked) as event_error:
        FencedEventSink(InvalidEventEvidenceSink(), event_token, event_authority).emit(
            _event(event_run_id, 1)
        )
    assert isinstance(event_error.value.__cause__, RuntimeError)
    assert event_authority.revoked is True

    class InvalidTerminalEvidenceSink:
        capabilities = harness.sink.capabilities

        def __getattr__(self, name: str) -> Any:
            return getattr(harness.sink, name)

        def settle_terminal(
            self,
            outcome: TerminalOutcome,
            *,
            writer_token: WriterToken,
        ) -> CommitResult:
            result = harness.sink.settle_terminal(outcome, writer_token=writer_token)
            return CommitResult(status=result.status, content_digest="0" * 64)

    terminal_run_id = "invalid-terminal-evidence"
    terminal_token = harness.claim_writer(terminal_run_id, "owner-a")
    terminal_authority = ActivationWriteAuthority()
    with pytest.raises(WriteAuthorityRevoked) as terminal_error:
        FencedTerminalBridge(
            InvalidTerminalEvidenceSink(),
            terminal_token,
            terminal_authority,
        ).settle(_terminal(terminal_run_id))
    assert isinstance(terminal_error.value.__cause__, RuntimeError)
    assert terminal_authority.revoked is True


def test_checkpoint_decoder_rejects_tampered_activation_receipt(tmp_path: Path) -> None:
    harness, token, checkpoint, spec = _seed_checkpoint(
        tmp_path,
        run_id="activation-receipt-tamper",
    )
    command = _command(checkpoint)
    ActivationDriver(
        sink=harness.sink,
        writer_token=token,
        loop_factory=_loop_factory(tmp_path, spec, _CountingFinalAdapter(), []),
    ).drive(command)
    record = harness.sink.latest_checked(command.run_id).value
    assert record is not None
    payload = record.checkpoint.to_json()
    payload["applied_input_receipts"][command.checkpoint_marker]["event_cursor"] = -1

    decoded = decode_checkpoint(payload)

    assert decoded.status == "corrupt"
    assert decoded.value is None
    assert decoded.error_code == "checkpoint_corrupt"


def test_receipt_projection_rejects_valid_but_inconsistent_outcome_metadata(
    tmp_path: Path,
) -> None:
    harness, token, checkpoint, spec = _seed_checkpoint(
        tmp_path,
        run_id="activation-receipt-outcome-tamper",
    )
    command = _command(checkpoint)
    ActivationDriver(
        sink=harness.sink,
        writer_token=token,
        loop_factory=_loop_factory(tmp_path, spec, _CountingFinalAdapter(), []),
    ).drive(command)
    record = harness.sink.latest_checked(command.run_id).value
    assert record is not None
    payload = record.checkpoint.to_json()
    original_checkpoint_sha256 = payload["applied_input_receipts"][
        command.checkpoint_marker
    ]["checkpoint_sha256"]
    payload["applied_input_receipts"][command.checkpoint_marker][
        "checkpoint_sha256"
    ] = "0" * 64
    digest_tamper = decode_checkpoint(payload)
    assert digest_tamper.value is not None
    with pytest.raises(NativeAgentError) as digest_error:
        ActivationReceipt.from_checkpoint(command, digest_tamper.value)
    assert digest_error.value.error_code == "invalid_activation_receipt"

    payload["applied_input_receipts"][command.checkpoint_marker][
        "checkpoint_sha256"
    ] = original_checkpoint_sha256
    payload["applied_input_receipts"][command.checkpoint_marker]["retry_eligibility"] = "safe"
    decoded = decode_checkpoint(payload)
    assert decoded.value is not None

    with pytest.raises(NativeAgentError) as raised:
        ActivationReceipt.from_checkpoint(command, decoded.value)

    assert raised.value.error_code == "invalid_activation_receipt"


def test_fake_sink_type_still_satisfies_activation_protocol() -> None:
    sink: object = DeterministicFencedRunSink({})
    assert hasattr(sink, "latest_event_sequence")
    assert hasattr(sink, "read_terminal")
