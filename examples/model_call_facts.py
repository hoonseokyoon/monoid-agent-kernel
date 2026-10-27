"""Run offline: python examples/model_call_facts.py. No provider credentials are needed."""

import asyncio
import json

from monoid_agent_kernel import CapturePolicy, ModelIOSubscription, UsageObservation
from monoid_agent_kernel.model_call import ModelCallRunner
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn


class ExampleAdapter:
    def next_turn(self, request: ModelRequest) -> ModelTurn:
        return ModelTurn(
            final_text="Ready",
            usage={"input_tokens": 0, "output_tokens": 3},
            usage_evidence=UsageObservation(
                reported={"input_tokens": 0, "output_tokens": 3},
                source="adapter", collection="normal",
            ),
        )


class Diagnostics:
    def on_model_call(self, capture):
        receipt = capture.receipt
        print(json.dumps({
            "termination": receipt.termination.to_json(),
            "known_usage": dict(receipt.usage_evidence.reported),
            "input_coverage": receipt.usage_evidence.completeness["input_tokens"],
            "total_coverage": receipt.usage_evidence.completeness["total_tokens"],
        }, indent=2))


async def main():
    runner = ModelCallRunner(
        adapter=ExampleAdapter(),
        subscriptions=(ModelIOSubscription(Diagnostics(), CapturePolicy(mode="none")),),
    )
    await runner.acall(ModelRequest(instruction="hello", system_prompt="", tools=()))


if __name__ == "__main__":
    asyncio.run(main())
