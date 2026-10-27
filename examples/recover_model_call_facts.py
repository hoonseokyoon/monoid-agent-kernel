"""Offline JSON round trip through the public lifecycle hooks.

The memory journal demonstrates the hook contract. A production host supplies durable,
fenced storage; see the FencedModelCallLifecycle adapter and its service tests.
Run: python examples/recover_model_call_facts.py
"""

import asyncio
import hashlib
import json

from monoid_agent_kernel import CapturePolicy, ModelIOSubscription, UsageObservation
from monoid_agent_kernel.core.model_invocation import DurableModelInvocation, decode_model_invocation
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.model_call import ModelCallRunner
from monoid_agent_kernel.model_lifecycle import ModelDispatchReservation, RecoveredModelDispatch
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn


class MemoryJournal:
    def __init__(self):
        self.payload = None
        self.result = None

    def reserve(self, proposed):
        return proposed

    def dispatch_started(self, reservation):
        pass

    def settled(self, settlement):
        reservation = settlement.reservation
        self.result = settlement.result_blob
        invocation = DurableModelInvocation(
            run_id="example", logical_call_id=reservation.logical_call_id, revision=3,
            dispatch_id=reservation.dispatch_id, dispatch_attempt=reservation.dispatch_attempt,
            idempotency_key=reservation.idempotency_key, dispatch_state="settled",
            request_digest=reservation.request_digest, digest_generation=reservation.digest_generation,
            receipt=settlement.receipt, failure_code=settlement.failure_code,
            result_ref=f"blob:{hashlib.sha256(self.result).hexdigest()}" if self.result is not None else "",
        )
        self.payload = json.loads(json.dumps(invocation.to_json()))

    def unknown(self, unknown):
        raise RuntimeError("The example stops on an ambiguous dispatch")

    def recover(self, query):
        if self.payload is None:
            return None
        checked = decode_model_invocation(self.payload)
        assert checked.ok and checked.value is not None
        invocation = checked.value
        assert query.request_digest == invocation.request_digest
        return RecoveredModelDispatch(
            reservation=ModelDispatchReservation(
                logical_call_id=invocation.logical_call_id, dispatch_attempt=invocation.dispatch_attempt,
                dispatch_id=invocation.dispatch_id, idempotency_key=invocation.idempotency_key,
                request_digest=invocation.request_digest, digest_generation=invocation.digest_generation,
            ),
            receipt=invocation.receipt, result_blob=self.result, failure_code=invocation.failure_code,
        )


class OfflineAdapter:
    provider_name = "fake"

    def __init__(self):
        self.calls = 0

    def next_turn(self, request):
        self.calls += 1
        assert self.calls == 1, "Recovery must use the checked result"
        return ModelTurn(final_text="Ready", usage={"input_tokens": 0, "output_tokens": 3},
                         usage_evidence=UsageObservation(reported={"input_tokens": 0, "output_tokens": 3},
                                                         source="adapter", collection="normal"))


class Diagnostics:
    def __init__(self):
        self.receipts = []

    def on_model_call(self, capture):
        self.receipts.append(capture.receipt)


async def main():
    journal, adapter, diagnostics = MemoryJournal(), OfflineAdapter(), Diagnostics()
    for _ in range(2):
        runner = ModelCallRunner(adapter=adapter, lifecycle_hook=journal,
                                 subscriptions=(ModelIOSubscription(diagnostics, CapturePolicy(mode="none")),))
        await runner.acall(ModelRequest(instruction="hello", system_prompt="", tools=(),
                                        model=ModelConfig(provider="fake", model="offline")),
                           logical_call_id="example_call")
    original, recovered = diagnostics.receipts
    assert recovered.termination == original.termination
    assert recovered.usage_evidence == original.usage_evidence
    assert recovered.idempotency_key == original.idempotency_key
    print(json.dumps({"schema": journal.payload["schema_version"], "provider_calls": adapter.calls,
                      "known_usage": dict(recovered.usage_evidence.reported),
                      "total_coverage": recovered.usage_evidence.completeness["total_tokens"]}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
