"""R0 coverage for the Agent Studio reference app.

Studio is a reference example, but it is the pressure test for "build an app from the surface
alone", so it gets the same regression coverage the other reference services have. These tests
drive the Studio server through its Python API (no browser / no Chromium window) against the
offline echo model, so they are deterministic and key-less.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from monoid_agent_kernel.errors import ModelAdapterError, NativeAgentError
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn, TextDelta, TurnComplete
from monoid_agent_kernel.providers.fake import (
    FakeModelAdapter,
    FakeStreamingModelAdapter,
    fake_tool_call,
)
from monoid_agent_kernel.reference.llm_gateway.providers import EchoModelAdapter
from monoid_agent_kernel.reference.studio.activity import describe_event
from monoid_agent_kernel.reference.studio.server import (
    _ALL_CAPABILITIES,
    StudioConfig,
    StudioServer,
    _agent_runtime_config,
    _runtime_config_for,
)
from support.process import python_command as _python_command

__all__ = [
    'EchoModelAdapter',
    'FakeModelAdapter',
    'FakeStreamingModelAdapter',
    'ModelAdapterError',
    'ModelRequest',
    'ModelTurn',
    'NativeAgentError',
    'Path',
    'StudioConfig',
    'StudioServer',
    'TextDelta',
    'TurnComplete',
    '_ALL_CAPABILITIES',
    '_BlockingThenToolAdapter',
    '_RaiseThenAdapter',
    '_agent_runtime_config',
    '_mcp_studio',
    '_python_command',
    '_runtime_config_for',
    '_settled',
    '_shell_studio',
    '_wait_event',
    '_wait_proposal',
    '_wait_settled',
    'annotations',
    'describe_event',
    'fake_tool_call',
    'pytest',
    'studio',
    'threading',
    'time',
]


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


def _wait_proposal(server: StudioServer, run_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        proposal = server.proposal(run_id)
        if proposal.get("ready") and proposal.get("diff"):
            return proposal
        time.sleep(0.1)
    return server.proposal(run_id)


def _wait_event(server: StudioServer, run_id: str, etype: str, timeout: float = 10.0) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for event in server.poll_events(run_id, 0).get("events", []):
            if event.get("type") == etype:
                return event
        time.sleep(0.1)
    return None


def _shell_studio(tmp_path: Path, turns: list) -> StudioServer:
    fake = FakeModelAdapter(turns=turns)
    server = StudioServer(
        StudioConfig(workspace=tmp_path / "ws", host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
    server.start()
    return server


class _RaiseThenAdapter:
    """A gateway provider that raises scripted exceptions, otherwise returns turns."""

    def __init__(self, script: list) -> None:
        self.script = list(script)

    def next_turn(self, request):  # noqa: ANN001
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _BlockingThenToolAdapter:
    """Turn 1 calls a tool; turn 2 blocks until released (giving the test a window to call
    interrupt_chat while a turn is in flight), then calls a tool so the next step boundary
    trips the interrupt. Turn 3 (after resume) settles."""

    def __init__(self) -> None:
        self.calls = 0
        self.reached_block = threading.Event()
        self.release = threading.Event()

    def next_turn(self, request):  # noqa: ANN001
        self.calls += 1
        if self.calls == 1:
            return ModelTurn(response_id="r1", tool_calls=(fake_tool_call("fs_read", {"path": "notes.md"}, "c1"),))
        if self.calls == 2:
            self.reached_block.set()
            self.release.wait(5.0)
            return ModelTurn(response_id="r2", tool_calls=(fake_tool_call("fs_read", {"path": "notes.md"}, "c2"),))
        return ModelTurn(response_id="r3", final_text="resumed ok")


def _mcp_studio(tmp_path: Path, *, provider_factory=None) -> StudioServer:
    return StudioServer(
        StudioConfig(workspace=tmp_path / "ws", host="127.0.0.1", port=0, run_root=tmp_path / "runs", mcp=True),
        provider_factory=provider_factory,
    )
