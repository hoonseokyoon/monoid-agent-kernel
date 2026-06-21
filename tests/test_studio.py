"""R0 coverage for the Agent Studio reference app.

Studio is a reference example, but it is the pressure test for "build an app from the surface
alone", so it gets the same regression coverage the other reference services have. These tests
drive the Studio server through its Python API (no browser / no Chromium window) against the
offline echo model, so they are deterministic and key-less.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from native_agent_runner.providers.base import ModelRequest
from native_agent_runner.reference.llm_gateway.providers import EchoModelAdapter
from native_agent_runner.reference.studio.server import StudioConfig, StudioServer


def _settled(server: StudioServer, run_id: str) -> list[dict]:
    events = server.poll_events(run_id, 0).get("events", [])
    return [e for e in events if e.get("type") == "turn.settled"]


def _wait_settled(server: StudioServer, run_id: str, n: int, timeout: float = 10.0) -> list[dict]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        settled = _settled(server, run_id)
        if len(settled) >= n:
            return settled
        time.sleep(0.1)
    return _settled(server, run_id)


@pytest.fixture
def studio(tmp_path: Path):
    server = StudioServer(
        StudioConfig(
            workspace=tmp_path / "ws",
            host="127.0.0.1",
            port=0,
            provider="offline",
            run_root=tmp_path / "runs",
        )
    )
    server.start()
    try:
        yield server
    finally:
        server.shutdown()


def test_echo_adapter_replies_with_latest_user_text() -> None:
    adapter = EchoModelAdapter()
    request = ModelRequest(
        instruction="hello there",
        system_prompt="",
        tools=(),
        messages=({"role": "user", "content": "hello there"},),
    )
    turn = adapter.next_turn(request)
    assert turn.final_text
    assert "hello there" in turn.final_text
    assert turn.tool_calls == ()
    assert turn.usage["total_tokens"] > 0


def test_offline_chat_produces_assistant_reply(studio: StudioServer) -> None:
    result = studio.start_chat("summarize the workspace")
    run_id = result["run_id"]
    settled = _wait_settled(studio, run_id, 1)
    assert len(settled) == 1
    assert settled[0]["data"]["final_text"]


def test_multi_turn_session_yields_a_reply_per_message(studio: StudioServer) -> None:
    run_id = studio.start_chat("first")["run_id"]
    assert len(_wait_settled(studio, run_id, 1)) == 1
    studio.continue_chat(run_id, "second")
    assert len(_wait_settled(studio, run_id, 2)) == 2
    studio.continue_chat(run_id, "third")
    assert len(_wait_settled(studio, run_id, 3)) == 3
    # The session stays open for the next message rather than going terminal.
    assert studio.run_status(run_id)["status"] not in {"completed", "failed", "limited"}


def test_run_tokens_are_not_exposed_to_callers(studio: StudioServer) -> None:
    # The BFF holds run tokens server-side; start_chat returns only the run id + status.
    result = studio.start_chat("hello")
    assert set(result) == {"run_id", "status"}
    assert "run_token" not in result
