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
    MODEL_REQUEST_DIGEST_GENERATION, DurableModelInvocation, decode_model_invocation,
)
from monoid_agent_kernel.hosting.contracts import ModelInvocationRecord
from monoid_agent_kernel.hosting.model_calls import FencedModelCallLifecycle
from monoid_agent_kernel.model_lifecycle import ModelDispatchRecoveryQuery, durable_model_result_blob
from monoid_agent_kernel.providers.base import ModelTurn


CASES = [(policy, variant) for policy in ("passive", "required", "outbox")
         for variant in ("canonical", "schema_alias", "digest_alias", "omitted_scalar", "v2")]
CASES += [("passive", "omitted_policy"), ("passive", "legacy_flag"), ("required", "legacy_flag")]


@pytest.mark.parametrize("success", [False, True])
@pytest.mark.parametrize(("policy", "variant"), CASES)
def test_historical_settlement_recovers_without_rewriting_its_coordinate(
    policy: str, variant: str, success: bool,
) -> None:
    blob = durable_model_result_blob(ModelTurn(final_text="answer", stop_reason="stop"))
    sha = hashlib.sha256(blob).hexdigest()
    blobs = {sha: blob} if success else {}
    payload = DurableModelInvocation(
        run_id="history", logical_call_id="call_1", revision=3, dispatch_id="dispatch_1",
        dispatch_attempt=1, idempotency_key="idem_1", dispatch_state="settled",
        request_digest="a" * 64, digest_generation=MODEL_REQUEST_DIGEST_GENERATION,
        evidence_policy=policy, receipt={"attempts": 1, "usage": {"input_tokens": 0}},
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
        payload["receipt"].update(facts({"input_tokens": 0}, outcome="completed" if success else "failed"))
    original_digest = _record_digest(payload, blobs)
    decoded = decode_model_invocation(payload)
    assert decoded.status == "loaded"
    assert decoded.value is not None
    invocation = decoded.value
    harness = DeterministicFencedRunHarness()
    harness.sink.capabilities = replace(harness.sink.capabilities, transactional_outbox=True)
    token = harness.claim_writer("history", "worker_1")
    coord = ("history", "call_1", 3)
    harness.sink._invocations[coord] = (original_digest, ModelInvocationRecord(
        3, invocation, _blob_reader=blobs.__getitem__,
    ))
    harness.sink._invocation_heads[("history", "call_1")] = 3
    harness.sink._blobs.update({("history", key): value for key, value in blobs.items()})
    query = ModelDispatchRecoveryQuery(logical_call_id="call_1", request_digest="a" * 64,
                                      digest_generation=MODEL_REQUEST_DIGEST_GENERATION)
    lifecycle = FencedModelCallLifecycle(harness.sink, token)
    for _ in range(2):
        recovered = lifecycle.recover(query)
        assert recovered is not None
        assert recovered.result_blob == (blob if success else None)
        assert recovered.receipt == invocation.receipt
        assert harness.sink._invocations[coord][0] == original_digest
        assert len(harness.sink._invocations) == 1
    if policy != "passive":
        evidence = harness.sink._model_evidence if policy == "required" else harness.sink._model_evidence_outbox
        assert evidence[coord][0] == _record_digest(payload)
        assert len(evidence) == 1
    nested = checkpoint_payload_for_write(RunCheckpoint(run_id="history", last_model_invocation=payload))
    assert canonical_sha256(nested["last_model_invocation"]) == canonical_sha256(payload)
    harness.claim_writer("history", "worker_2")
    with pytest.raises(WriteAuthorityRevoked):
        lifecycle.recover(query)
    assert harness.sink._invocations[coord][0] == original_digest
