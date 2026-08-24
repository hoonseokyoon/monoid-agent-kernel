from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.hosting.activation import ActivationCommand, ActivationReceipt
from monoid_agent_kernel.hosting.admission import (
    ADMISSION_RECEIPT_SCHEMA_VERSION,
    ADMISSION_REQUEST_SCHEMA_VERSION,
    ADMITTED_COMMAND_SCHEMA_VERSION,
    AdmittedCommand,
    AdmissionReceipt,
    AdmissionRequest,
    CommandAdmissionStore,
    CommandDispatchStore,
    CommandOutboxDispatcher,
    CommandTransport,
    DispatchClaim,
    DispatchResult,
    DispatchToken,
    MAX_COMMAND_RETRY_DELAY_S,
)
from monoid_agent_kernel.hosting.contracts import WriterToken


def _request(command_id: str = "command-1") -> AdmissionRequest:
    digest = canonical_sha256({"command_id": command_id})
    return AdmissionRequest(
        run_id="run-1",
        command_id=command_id,
        kind="input",
        request_digest=digest,
        payload_ref=f"blob:{digest}",
    )


def _command(command_id: str = "command-1", sequence: int = 1) -> AdmittedCommand:
    return AdmittedCommand.from_request(_request(command_id), sequence)


def _activation(command: AdmittedCommand) -> ActivationCommand:
    return ActivationCommand(
        run_id=command.run_id,
        command_id=command.command_id,
        command_sequence=command.command_sequence,
        kind=command.kind,
        source_checkpoint_seq=3,
        source_checkpoint_sha256="1" * 64,
        request_digest=command.request_digest,
        payload_ref=command.payload_ref,
    )


def _completion(activation: ActivationCommand) -> ActivationReceipt:
    return ActivationReceipt(
        run_id=activation.run_id,
        command_id=activation.command_id,
        command_sequence=activation.command_sequence,
        command_identity_sha256=activation.identity_sha256,
        checkpoint_seq=4,
        checkpoint_sha256="2" * 64,
        checkpoint_ref=f"checkpoint:{activation.run_id}/4",
        state="awaiting_input",
        boundary_reason="settled",
        terminal=False,
        terminal_ref="",
        applied_input_ref=activation.applied_input_ref,
        event_cursor=8,
        stream_cursor=0,
        outcome_kind="completed",
        retry_eligibility="not_applicable",  # type: ignore[arg-type]
    )


@dataclass
class _FakeDispatchStore:
    command: AdmittedCommand
    claim_available: bool = True
    operations: list[tuple[str, object]] = field(default_factory=list)

    def claim_dispatch(
        self,
        owner_id: str,
        claim_id: str,
        *,
        lease_s: float,
    ) -> DispatchClaim | None:
        self.operations.append(("claim", (owner_id, claim_id, lease_s)))
        if not self.claim_available:
            return None
        self.claim_available = False
        return DispatchClaim(
            token=DispatchToken(
                run_id=self.command.run_id,
                command_id=self.command.command_id,
                owner_id=owner_id,
                claim_id=claim_id,
                generation=1,
            ),
            command=self.command,
            attempt=1,
        )

    def acknowledge_dispatch(
        self,
        token: DispatchToken,
        result: DispatchResult,
    ) -> AdmissionReceipt:
        self.operations.append(("ack", (token, result)))
        return AdmissionReceipt(
            command=self.command,
            state="dispatched",
            attempt_count=1,
            dispatch_ref=result.dispatch_ref,
        )

    def retry_dispatch(
        self,
        token: DispatchToken,
        *,
        error_code: str,
        delay_s: float,
    ) -> AdmissionReceipt:
        self.operations.append(("retry", (token, error_code, delay_s)))
        return AdmissionReceipt(
            command=self.command,
            state="prepared",
            attempt_count=1,
            error_code=error_code,
        )

    def reject_dispatch(
        self,
        token: DispatchToken,
        *,
        error_code: str,
    ) -> AdmissionReceipt:
        self.operations.append(("reject", (token, error_code)))
        return AdmissionReceipt(
            command=self.command,
            state="dead_letter",
            attempt_count=1,
            error_code=error_code,
        )


@dataclass
class _FakeTransport:
    result: object
    commands: list[AdmittedCommand] = field(default_factory=list)

    def dispatch(self, command: AdmittedCommand) -> Any:
        self.commands.append(command)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _ProtocolStore(_FakeDispatchStore):
    def admit(self, request: AdmissionRequest) -> AdmissionReceipt:
        return AdmissionReceipt(command=AdmittedCommand.from_request(request, 1), state="prepared")

    def receipt(self, run_id: str, command_id: str) -> AdmissionReceipt | None:
        if (run_id, command_id) != (self.command.run_id, self.command.command_id):
            return None
        return AdmissionReceipt(command=self.command, state="prepared")

    def bind_activation(
        self,
        command: AdmittedCommand,
        *,
        writer_token: WriterToken,
    ) -> ActivationCommand:
        assert writer_token.run_id == command.run_id
        return _activation(command)


def test_admission_request_and_admitted_command_are_strict_retry_stable() -> None:
    request = _request()
    command = AdmittedCommand.from_request(request, 7)

    assert AdmissionRequest.from_json(request.to_json()) == request
    assert AdmittedCommand.from_json(command.to_json()) == command
    assert command.request == request

    unknown = {**request.to_json(), "payload": "private"}
    with pytest.raises(ValueError, match="closed schema"):
        AdmissionRequest.from_json(unknown)
    tampered = {**command.to_json(), "identity_sha256": "0" * 64}
    with pytest.raises(ValueError, match="identity digest mismatch"):
        AdmittedCommand.from_json(tampered)
    conflicting_request = {**command.to_json(), "request_identity_sha256": "0" * 64}
    with pytest.raises(ValueError, match="request identity mismatch"):
        AdmittedCommand.from_json(conflicting_request)

    legacy_request = request.to_json()
    legacy_request["schema_version"] = ADMISSION_REQUEST_SCHEMA_VERSION.replace(
        "monoid.", "native-agent-runner.", 1
    )
    assert AdmissionRequest.from_json(legacy_request).to_json() == request.to_json()
    legacy_command = command.to_json()
    legacy_command["schema_version"] = ADMITTED_COMMAND_SCHEMA_VERSION.replace(
        "monoid.", "native-agent-runner.", 1
    )
    assert AdmittedCommand.from_json(legacy_command).to_json() == command.to_json()


def test_admission_receipt_states_bind_exact_activation_identity() -> None:
    command = _command()
    activation = _activation(command)
    completion = _completion(activation)
    receipts = (
        AdmissionReceipt(command=command, state="prepared"),
        AdmissionReceipt(
            command=command,
            state="dispatched",
            attempt_count=1,
            dispatch_ref="temporal:run-1",
        ),
        AdmissionReceipt(
            command=command,
            state="activation_claimed",
            attempt_count=1,
            dispatch_ref="temporal:run-1",
            activation_command=activation,
        ),
        AdmissionReceipt(
            command=command,
            state="completed",
            attempt_count=1,
            dispatch_ref="temporal:run-1",
            activation_command=activation,
            activation_receipt=completion,
        ),
        AdmissionReceipt(
            command=command,
            state="dead_letter",
            attempt_count=3,
            error_code="dispatch_rejected",
        ),
    )

    for receipt in receipts:
        assert AdmissionReceipt.from_json(receipt.to_json()) == receipt
    for attempt_count in (0, 2):
        terminal = AdmissionReceipt(
            command=command,
            state="run_terminal",
            attempt_count=attempt_count,
            error_code="run_terminal",
        )
        assert AdmissionReceipt.from_json(terminal.to_json()) == terminal

    bad_version = receipts[0].to_json()
    bad_version["schema_version"] = "monoid.admission-receipt.v999"
    with pytest.raises(ValueError, match="unsupported admission receipt"):
        AdmissionReceipt.from_json(bad_version)
    legacy = receipts[0].to_json()
    legacy["schema_version"] = ADMISSION_RECEIPT_SCHEMA_VERSION.replace(
        "monoid.", "native-agent-runner.", 1
    )
    assert AdmissionReceipt.from_json(legacy).to_json() == receipts[0].to_json()

    with pytest.raises(ValueError, match="activation binding is inconsistent"):
        replace(receipts[2], activation_command=replace(activation, command_id="other"))
    with pytest.raises(ValueError, match="completion is inconsistent"):
        replace(receipts[3], activation_receipt=replace(completion, command_id="other"))
    for field_name, invalid_value in (
        ("checkpoint_seq", activation.source_checkpoint_seq),
        ("checkpoint_ref", "checkpoint:run-1/999"),
        ("applied_input_ref", "input:run-1/other"),
    ):
        invalid = receipts[3].to_json()
        raw_completion = invalid["activation_receipt"]
        assert isinstance(raw_completion, dict)
        raw_completion[field_name] = invalid_value
        with pytest.raises(ValueError, match="completion is inconsistent"):
            AdmissionReceipt.from_json(invalid)
    for receipt in receipts[1:]:
        invalid = receipt.to_json()
        invalid["attempt_count"] = 0
        with pytest.raises(ValueError, match="requires a dispatch attempt"):
            AdmissionReceipt.from_json(invalid)
    unclaimed_error = receipts[0].to_json()
    unclaimed_error["error_code"] = "transport_busy"
    with pytest.raises(ValueError, match="unclaimed admission receipt"):
        AdmissionReceipt.from_json(unclaimed_error)
    for receipt in receipts[1:4]:
        invalid = receipt.to_json()
        invalid["error_code"] = "transport_busy"
        with pytest.raises(ValueError, match="evidence is inconsistent"):
            AdmissionReceipt.from_json(invalid)
    with pytest.raises(ValueError, match="terminal-run admission receipt"):
        AdmissionReceipt(command=command, state="run_terminal", error_code="terminal")


@pytest.mark.parametrize(
    ("result", "operation", "state"),
    (
        (DispatchResult(status="accepted", dispatch_ref="temporal:workflow-1"), "ack", "dispatched"),
        (DispatchResult(status="retry", error_code="transport_busy"), "retry", "prepared"),
        (DispatchResult(status="rejected", error_code="unsupported_command"), "reject", "dead_letter"),
    ),
)
def test_finite_dispatcher_routes_content_free_results(
    result: DispatchResult,
    operation: str,
    state: str,
) -> None:
    command = _command()
    store = _FakeDispatchStore(command)
    transport = _FakeTransport(result)
    dispatcher = CommandOutboxDispatcher(
        store=store,
        transport=transport,
        owner_id="worker-1",
        retry_delay_s=lambda attempt: float(attempt + 2),
        claim_id_factory=lambda: "claim-1",
    )

    receipt = dispatcher.dispatch_once()

    assert receipt is not None and receipt.state == state
    assert transport.commands == [command]
    assert [name for name, _ in store.operations] == ["claim", operation]
    assert dispatcher.dispatch_once() is None


@pytest.mark.parametrize(
    "delay_s",
    (MAX_COMMAND_RETRY_DELAY_S + 1, 10**1000),
)
def test_dispatcher_caps_retry_delay_before_store_settlement(delay_s: float | int) -> None:
    command = _command()
    store = _FakeDispatchStore(command)
    receipt = CommandOutboxDispatcher(
        store=store,
        transport=_FakeTransport(DispatchResult(status="retry", error_code="transport_busy")),
        owner_id="worker-1",
        retry_delay_s=lambda attempt: delay_s,
        claim_id_factory=lambda: "claim-1",
    ).dispatch_once()

    assert receipt is not None and receipt.state == "prepared"
    operation, payload = store.operations[-1]
    assert operation == "retry"
    assert isinstance(payload, tuple) and payload[2] == MAX_COMMAND_RETRY_DELAY_S


def test_dispatcher_sanitizes_transport_exception_and_invalid_result() -> None:
    command = _command()
    for result, expected_code in (
        (RuntimeError("private transport details"), "dispatch_transport_error"),
        ({"accepted": True}, "invalid_dispatch_result"),
    ):
        store = _FakeDispatchStore(command)
        receipt = CommandOutboxDispatcher(
            store=store,
            transport=_FakeTransport(result),
            owner_id="worker-1",
            retry_delay_s=lambda attempt: 0.0,
            claim_id_factory=lambda: "claim-1",
        ).dispatch_once()

        assert receipt is not None and receipt.error_code == expected_code
        assert "private transport details" not in repr(store.operations)


def test_dispatcher_rejects_invalid_host_controls() -> None:
    store = _FakeDispatchStore(_command())
    with pytest.raises(ValueError, match="retry delay"):
        CommandOutboxDispatcher(
            store=store,
            transport=_FakeTransport(DispatchResult(status="retry", error_code="busy")),
            owner_id="worker-1",
            retry_delay_s=lambda attempt: float("nan"),
            claim_id_factory=lambda: "claim-1",
        ).dispatch_once()
    with pytest.raises(ValueError, match="claim factory"):
        CommandOutboxDispatcher(
            store=_FakeDispatchStore(_command()),
            transport=_FakeTransport(
                DispatchResult(status="accepted", dispatch_ref="temporal:workflow-1")
            ),
            owner_id="worker-1",
            claim_id_factory=lambda: "bad claim",
        ).dispatch_once()


def test_protocols_accept_independent_structural_implementations() -> None:
    command = _command()
    store = _ProtocolStore(command)
    transport = _FakeTransport(
        DispatchResult(status="accepted", dispatch_ref="temporal:workflow-1")
    )

    assert isinstance(store, CommandAdmissionStore)
    assert isinstance(store, CommandDispatchStore)
    assert isinstance(transport, CommandTransport)
