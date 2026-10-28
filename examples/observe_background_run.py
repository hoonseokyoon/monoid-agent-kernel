"""Observe a real background run, reconnect, then explicitly cancel it. Runs offline."""

import asyncio
import json
import tempfile
import threading
import time
from pathlib import Path

from monoid_agent_kernel import AgentRuntimeConfig
from monoid_agent_kernel.providers.base import TextDelta, TurnComplete
from monoid_agent_kernel.reference.backend import (
    BackendRunRequest, LiveModelStreamBroker, RunnerBackend, TokenManager,
)


def run_example(root: Path) -> dict:
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    first_sent, second_sent, overflow_sent = (threading.Event() for _ in range(3))
    next_chunk, overflow, finish = (threading.Event() for _ in range(3))

    class Adapter:
        async def astream_turn(self, request):
            yield TextDelta("A")
            first_sent.set()
            assert await asyncio.to_thread(next_chunk.wait, 10), "next chunk was not released"
            yield TextDelta("B")
            second_sent.set()
            assert await asyncio.to_thread(overflow.wait, 10), "overflow was not released"
            yield TextDelta("C")
            yield TextDelta("D")
            overflow_sent.set()
            assert await asyncio.to_thread(finish.wait, 10), "terminal was not released"
            yield TextDelta("late output")
            yield TurnComplete()

    broker = LiveModelStreamBroker(max_frames=2)
    backend = RunnerBackend(
        run_root=root / "runs", allowed_workspace_roots=(workspace,),
        token_manager=TokenManager.from_secret("offline-example-token-secret-32bytes"),
        llm_gateway_url="http://unused.invalid",
        model_adapter_factory=lambda spec, token: Adapter(),
        stream_model_calls=True, model_stream_broker=broker,
    )
    submission = None
    readers = []

    def subscribe(cursor=None):
        reader = backend.subscribe_model_stream(
            submission.run_id, submission.run_token, after_cursor=cursor,
        )
        readers.append(reader)
        return reader

    def wait_for_result():
        # A terminal state can precede close/publication; ready follows the completed close.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = backend.result(submission.run_id, submission.run_token)
            if result["ready"]:
                return result
            time.sleep(0.01)
        raise TimeoutError("background run did not publish its final result")

    def cleanup():
        for gate in (next_chunk, overflow, finish):
            gate.set()
        for reader in readers:
            reader.close()
        try:
            backend.shutdown(drain=True, drain_timeout_s=10)
        finally:
            if submission is not None:
                wait_for_result()  # Also required when an observation/assertion failed.

    try:
        submission = backend.submit_run(BackendRunRequest(
            tenant_id="example", user_id="developer", workspace_root=workspace,
            instruction="Stream a short response", max_duration_s=30,
            runtime_config=AgentRuntimeConfig(definition_id="background-observation"),
        ))
        assert first_sent.wait(10), "adapter did not start"
        first, peer = subscribe(), subscribe()
        assert [frame.text for frame in first.poll() if frame.kind == "delta"] == ["A"]
        assert [frame.text for frame in peer.poll() if frame.kind == "delta"] == ["A"]
        cursor = first.cursor
        first.close()  # A UI disconnect releases only this reader.
        assert first.poll() == ()

        next_chunk.set()
        assert second_sent.wait(10), "active run did not continue after reader close"
        peer_text = "".join(frame.text for frame in peer.poll() if frame.kind == "delta")
        assert peer_text == "B"
        assert backend.status(submission.run_id, submission.run_token)["terminal"] is False
        resumed = subscribe(cursor)
        replayed = "".join(frame.text for frame in resumed.poll() if frame.kind == "delta")
        assert replayed == "B"  # after_cursor excludes text the client already consumed.
        resumed.close()

        overflow.set()
        assert overflow_sent.wait(10), "adapter did not publish the bounded-ring example"
        stale = subscribe(cursor)
        frames = stale.poll()
        assert frames[0].kind == "reset" and frames[0].reason == "cursor_gap"
        retained = "".join(frame.text for frame in frames if frame.kind == "delta")
        assert retained == "CD"
        assert [frame.cursor.sequence for frame in frames] == sorted(
            frame.cursor.sequence for frame in frames
        )

        cancel = backend.cancel_run(submission.run_id, submission.run_token)
        assert cancel["cancel_requested"] is True
        finish.set()
        state = backend.wait_for_run(submission.run_id, timeout_s=10).value
        assert state == "cancelled"
        wait_for_result()
        result = {
            "state": state, "reader_closed": first.closed,
            "continued_after_disconnect": True, "peer_text": peer_text,
            "replayed_text": replayed, "reset_reason": frames[0].reason,
            "history_complete": False, "retained_text": retained,
            "cancel_requested": cancel["cancel_requested"],
        }
    except BaseException:
        try:
            cleanup()
        except BaseException:
            pass  # Preserve the primary observation failure after attempting owned cleanup.
        raise
    else:
        cleanup()  # Successful work exposes a cleanup failure to its caller.
        return result


def main():
    with tempfile.TemporaryDirectory(prefix="monoid-background-observation-") as directory:
        print(json.dumps(run_example(Path(directory)), indent=2))


if __name__ == "__main__":
    main()
