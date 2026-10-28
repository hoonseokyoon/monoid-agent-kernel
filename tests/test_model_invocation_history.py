"""Recovery preserves the submitted record identity, including accepted historical spellings."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from support.fenced_hosting import DeterministicFencedRunHarness, _record_digest
from support.model_facts import facts

from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.core.authority import WriteAuthorityRevoked
from monoid_agent_kernel.core.checkpoint import RunCheckpoint, checkpoint_payload_for_write
from monoid_agent_kernel.core.model_invocation import (
    MODEL_REQUEST_DIGEST_GENERATION,
    DurableModelInvocation,
    decode_model_invocation,
    model_invocation_payload_for_write,
)
from monoid_agent_kernel.hosting.contracts import ModelInvocationRecord
from monoid_agent_kernel.hosting.model_calls import FencedModelCallLifecycle
from monoid_agent_kernel.model_lifecycle import (
    ModelDispatchRecoveryQuery,
    durable_model_result_blob,
)
from monoid_agent_kernel.providers.base import ModelTurn


CASES = [
    (policy, variant)
    for policy in ("passive", "required", "outbox")
    for variant in ("canonical", "schema_alias", "digest_alias", "omitted_scalar", "v2")
]
CASES += [("passive", "omitted_policy"), ("passive", "legacy_flag"), ("required", "legacy_flag")]


@pytest.mark.parametrize("success", [False, True])
@pytest.mark.parametrize(("policy", "variant"), CASES)
def test_historical_settlement_recovers_without_rewriting_its_coordinate(
    policy: str,
    variant: str,
    success: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blob = durable_model_result_blob(ModelTurn(final_text="answer", stop_reason="stop"))
    sha = hashlib.sha256(blob).hexdigest()
    blobs = {sha: blob} if success else {}
    payload = DurableModelInvocation(
        schema_version="monoid.model-invocation.v1",
        run_id="history",
        logical_call_id="call_1",
        revision=3,
        dispatch_id="dispatch_1",
        dispatch_attempt=1,
        idempotency_key="idem_1",
        dispatch_state="settled",
        request_digest="a" * 64,
        digest_generation=MODEL_REQUEST_DIGEST_GENERATION,
        evidence_policy=policy,
        receipt={"attempts": 1, "usage": {"input_tokens": 0}},
        result_ref=f"blob:{sha}" if success else "",
        failure_code="" if success else "provider_refused",
    ).to_json()
    if variant == "schema_alias":
        payload["schema_version"] = "native-agent-runner.model-invocation.v1"
    elif variant == "digest_alias":
        payload["digest_generation"] = "native-agent-runner.model-request-digest.v1"
    elif variant == "omitted_scalar":
        del payload["failure_code" if success else "result_ref"]
    elif variant == "omitted_policy":
        del payload["evidence_policy"]
    elif variant == "legacy_flag":
        payload["requires_evidence"] = policy == "required"
        del payload["evidence_policy"]
    elif variant == "v2":
        payload["schema_version"] = "monoid.model-invocation.v2"
        payload["receipt"].update(
            facts({"input_tokens": 0}, outcome="completed" if success else "failed")
        )
    original_digest = _record_digest(payload, blobs)
    decoded = decode_model_invocation(payload)
    assert decoded.status == "loaded"
    assert decoded.value is not None
    invocation = decoded.value
    harness = DeterministicFencedRunHarness()
    harness.sink.capabilities = replace(harness.sink.capabilities, transactional_outbox=True)
    token = harness.claim_writer("history", "worker_1")
    coord = ("history", "call_1", 3)
    harness.sink._invocations[coord] = (
        original_digest,
        ModelInvocationRecord(
            3,
            invocation,
            _blob_reader=blobs.__getitem__,
        ),
    )
    harness.sink._invocation_heads[("history", "call_1")] = 3
    harness.sink._blobs.update({("history", key): value for key, value in blobs.items()})
    query = ModelDispatchRecoveryQuery(
        logical_call_id="call_1",
        request_digest="a" * 64,
        digest_generation=MODEL_REQUEST_DIGEST_GENERATION,
    )
    lifecycle = FencedModelCallLifecycle(harness.sink, token)
    if policy == "required" and variant == "v2":
        from monoid_agent_kernel.core.model_facts import call_facts_of
        from monoid_agent_kernel.errors import ModelEvidenceUncommitted

        def unavailable(*args, **kwargs):
            raise RuntimeError("evidence store unavailable after restart")

        with monkeypatch.context() as patch:
            patch.setattr(harness.sink, "commit_model_evidence", unavailable)
            for _ in range(2):
                with pytest.raises(ModelEvidenceUncommitted) as caught:
                    lifecycle.recover(query)
                termination, evidence = call_facts_of(caught.value)
                assert termination is not None and evidence is not None
                assert termination.to_json() == payload["receipt"]["termination"]
                assert evidence.to_json() == payload["receipt"]["usage_evidence"]
                assert harness.sink._invocations[coord][0] == original_digest
    for _ in range(2):
        recovered = lifecycle.recover(query)
        assert recovered is not None
        assert recovered.result_blob == (blob if success else None)
        assert recovered.receipt == invocation.receipt
        assert harness.sink._invocations[coord][0] == original_digest
        assert len(harness.sink._invocations) == 1
    if policy != "passive":
        evidence = (
            harness.sink._model_evidence
            if policy == "required"
            else harness.sink._model_evidence_outbox
        )
        assert evidence[coord][0] == _record_digest(payload)
        assert len(evidence) == 1
    nested = checkpoint_payload_for_write(
        RunCheckpoint(run_id="history", last_model_invocation=payload)
    )
    assert canonical_sha256(nested["last_model_invocation"]) == canonical_sha256(payload)
    harness.claim_writer("history", "worker_2")
    with pytest.raises(WriteAuthorityRevoked):
        lifecycle.recover(query)
    assert harness.sink._invocations[coord][0] == original_digest


def test_loaded_storage_payload_is_detached_and_nested_mutation_is_refused() -> None:
    from monoid_agent_kernel.conformance.fixtures import load_compatibility_fixtures

    raw = next(
        item.payload
        for item in load_compatibility_fixtures()
        if item.fixture_id == "model-invocation-v025-facts-v2"
    )
    invocation = decode_model_invocation(raw).value
    assert invocation is not None
    exported = model_invocation_payload_for_write(invocation)
    exported["receipt"]["usage"]["input_tokens"] = 5
    assert model_invocation_payload_for_write(invocation) == raw
    invocation.receipt["usage"]["input_tokens"] = 5
    invocation.receipt["usage_evidence"]["reported"]["input_tokens"] = 5
    invocation.receipt["usage_evidence"]["attempts"][0]["reported"]["input_tokens"] = 5
    with pytest.raises(ValueError, match="mutated"):
        model_invocation_payload_for_write(invocation)


def test_v2_facts_do_not_expand_v1_or_unknown_receipt_vocabulary() -> None:
    from monoid_agent_kernel.conformance.fixtures import load_compatibility_fixtures

    raw = next(
        item.payload
        for item in load_compatibility_fixtures()
        if item.fixture_id == "model-invocation-v025-facts-v2"
    )
    assert (
        decode_model_invocation({**raw, "schema_version": "monoid.model-invocation.v1"}).status
        == "corrupt"
    )
    assert (
        decode_model_invocation(
            {**raw, "schema_version": "native-agent-runner.model-invocation.v2"}
        ).status
        == "unsupported_version"
    )
    assert (
        decode_model_invocation({**raw, "schema_version": "monoid.model-invocation.v999"}).status
        == "unsupported_version"
    )
    assert decode_model_invocation({**raw, "dispatch_state": "unknown"}).status == "corrupt"
    raw["receipt"]["usage_evidence"]["attempts"][0]["prompt"] = "secret"
    assert decode_model_invocation(raw).status == "corrupt"


@pytest.mark.parametrize("schema", ["monoid.model-invocation.v1", "monoid.model-invocation.v2"])
def test_new_revision_from_loaded_history_selects_v2_writer_and_drops_source_envelope(schema) -> None:
    from monoid_agent_kernel.conformance.fixtures import load_compatibility_fixtures

    raw = next(
        item.payload
        for item in load_compatibility_fixtures()
        if item.fixture_id == "model-invocation-v025-facts-v2"
    )
    raw.update(schema_version=schema, revision=1, dispatch_state="reserved", receipt=None, failure_code="")
    invocation = decode_model_invocation(raw).value
    harness = DeterministicFencedRunHarness()
    token = harness.claim_writer(raw["run_id"], "worker")
    assert harness.sink.commit_invocation(invocation, {}, writer_token=token).status == "committed"
    from monoid_agent_kernel.model_lifecycle import ModelDispatchReservation

    reservation = ModelDispatchReservation(
        **{
            key: raw[key]
            for key in (
                "logical_call_id",
                "dispatch_id",
                "dispatch_attempt",
                "request_digest",
                "digest_generation",
                "idempotency_key",
            )
        }
    )
    FencedModelCallLifecycle(harness.sink, token).dispatch_started(reservation)
    newer = harness.sink.load_invocation(raw["run_id"], raw["logical_call_id"]).value.invocation
    assert newer.revision == 2
    assert (
        model_invocation_payload_for_write(newer)["schema_version"] == "monoid.model-invocation.v2"
    )
    assert model_invocation_payload_for_write(invocation) == raw
