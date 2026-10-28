"""Request-owned streaming and reported usage, offline. Try --scenario silent_stop."""

import argparse
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
from monoid_agent_kernel.errors import TurnNotSettled
from monoid_agent_kernel.providers.base import TextDelta, TurnComplete


async def run_request(scenario="break"):
    """Own one request; a disconnected consumer receives its CancelledError after cleanup."""
    if scenario not in {"silent_stop", "break", "consumer_cancel"}:
        raise ValueError(f"unknown scenario: {scenario}")
    entered, release, cancelled, stopped, exit_seen = (asyncio.Event() for _ in range(5))
    reported = {"input_tokens": 0, "output_tokens": 3}  # The provider supplied no total.
    receipts, delivered, during_exit = [], [], []

    class Adapter:
        async def astream_turn(self, request):
            entered.set()
            if scenario == "break":
                yield TextDelta("Working")
            await release.wait()
            yield TextDelta("late output")
            yield TurnComplete(
                usage=reported,
                usage_evidence=UsageObservation(
                    reported=reported, source="adapter", collection="normal",
                ),
            )

    class Diagnostics:
        def on_model_call(self, capture):
            receipts.append(capture.receipt)

    with tempfile.TemporaryDirectory() as folder:
        workspace = Path(folder) / "workspace"
        workspace.mkdir()
        token = CancellationToken()
        remove_callback = token.add_cancel_callback(cancelled.set)
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
        owner = asyncio.current_task()

        async def progress():
            # Observation never changes the cancellation state or releases the provider.
            while not stopped.is_set():
                snapshot = stream.termination_budget()
                if (snapshot.stream_exit.phase == "cooperative_wait" and not during_exit
                        and any(call.drain_remaining_s is not None for call in snapshot.calls)):
                    during_exit.append(snapshot)
                    exit_seen.set()
                await asyncio.sleep(0.01)

        async def control():
            await entered.wait()  # Stop works before the provider emits any text.
            if scenario == "silent_stop":
                token.cancel()
            elif scenario == "consumer_cancel":
                owner.cancel("request disconnected")
            await cancelled.wait()  # Early break lets the stream context request cancellation.
            try:
                if scenario == "break":
                    await asyncio.wait_for(exit_seen.wait(), 1)
            finally:
                release.set()

        helpers = [asyncio.create_task(progress()), asyncio.create_task(control())]
        failed = False
        try:
            async with stream:
                async for item in stream:
                    if isinstance(item, TextDelta):
                        delivered.append(item.text)
                        if scenario == "break":
                            break
            if stream.suspension is not None:
                raise TurnNotSettled(stream.suspension)
        except BaseException:
            failed = True
            raise
        finally:
            stopped.set()
            release.set()
            remove_callback()
            for helper in helpers:
                if not helper.done():
                    helper.cancel()
            outcomes = await asyncio.gather(*helpers, return_exceptions=True)
            try:
                await loop.aclose()
            except BaseException:
                if not failed:
                    raise
            if not failed:
                for outcome in outcomes:
                    if isinstance(outcome, Exception):
                        raise outcome

        receipt, = receipts
        assert delivered == (["Working"] if scenario == "break" else [])
        assert receipt.termination.requested_by == "user_cancel"
        assert receipt.termination.ended_by == "provider_terminal"
        assert dict(receipt.usage_evidence.reported) == reported
        return {
            "scenario": scenario,
            "delivered_text": delivered,
            "during_exit": asdict(during_exit[0]) if during_exit else None,
            "after_exit": asdict(stream.termination_budget()),
            "termination": receipt.termination.to_json(),
            "known_usage": dict(receipt.usage_evidence.reported),
            "total_coverage": receipt.usage_evidence.completeness["total_tokens"],
            "receipt_count": len(receipts),
        }


async def main(scenario="break"):
    print(json.dumps(await run_request(scenario), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("break", "silent_stop", "consumer_cancel"), default="break")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.scenario))
    except asyncio.CancelledError as exc:
        # The CLI supervisor displays the cancellation received from its request owner.
        print(json.dumps({"consumer_cancelled": str(exc)}))
