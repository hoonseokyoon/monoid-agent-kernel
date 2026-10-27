"""Observe Stop, drain and the final bill offline: python examples/termination_budget.py."""

import asyncio
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from monoid_agent_kernel import (
    AgentLoop,
    AgentRunSpec,
    AgentRuntimeConfig,
    CapturePolicy,
    ModelIOSubscription,
    RunLimits,
    UsageObservation,
)
from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.providers.base import TextDelta, TurnComplete


async def main():
    release_bill = asyncio.Event()
    bill = {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
    receipts = []

    class Adapter:
        async def astream_turn(self, request):
            yield TextDelta("Working")
            await release_bill.wait()
            yield TurnComplete(
                usage=bill,
                usage_evidence=UsageObservation(reported=bill, source="adapter", collection="normal"),
            )

    class Diagnostics:
        def on_model_call(self, capture):
            receipts.append(capture.receipt)

    with tempfile.TemporaryDirectory() as folder:
        workspace = Path(folder) / "workspace"
        workspace.mkdir()
        token = CancellationToken()
        loop = AgentLoop.from_config(
            AgentRunSpec(
                workspace_root=workspace, run_root=Path(folder) / "runs",
                limits=RunLimits(max_duration_s=30),
            ),
            Adapter(), AgentRuntimeConfig(definition_id="termination-example"),
            cancellation_token=token, stream_model_calls=True,
            async_model_abort_drain_s=2, async_model_cancel_grace_s=0.2,
            model_io_subscriptions=(ModelIOSubscription(Diagnostics(), CapturePolicy(mode="none")),),
        )
        await loop.aopen()
        stream = loop.astream("Start a response")
        during_exit = []
        stopped = asyncio.Event()

        async def progress():
            # A host keeps this task alive while the stream context is exiting.
            # Reading the snapshot starts no timer and requests no cancellation.
            while not stopped.is_set():
                snapshot = stream.termination_budget()
                if snapshot.stream_exit.phase == "cooperative_wait":
                    during_exit.append(snapshot)
                    release_bill.set()
                    return
                await asyncio.sleep(0.01)

        watcher = asyncio.create_task(progress())
        try:
            async with stream:
                async for item in stream:
                    if isinstance(item, TextDelta):
                        token.cancel()  # The product's Stop button.
                        break
            await watcher
            assert during_exit and during_exit[0].calls
            assert during_exit[0].calls[0].drain_remaining_s is not None
            assert stream.termination_budget().stream_exit.phase == "settled"
            assert len(receipts) == 1
            receipt = receipts[0]
            assert receipt.termination.requested_by == "user_cancel"
            assert receipt.termination.ended_by == "provider_terminal", receipt
            assert dict(receipt.usage) == bill
            print(json.dumps({
                "during_exit": asdict(during_exit[0]),
                "after_exit": asdict(stream.termination_budget()),
                "termination": receipt.termination.to_json(),
                "usage": dict(receipt.usage),
            }, indent=2))
        finally:
            stopped.set()
            release_bill.set()
            await watcher
            await loop.aclose()


if __name__ == "__main__":
    asyncio.run(main())
