"""P4b-②: backend transport-neutral streaming output (HTTP SSE + programmatic seam).

The whole path is exercised with a `FakeStreamingModelAdapter` injected via the backend's
`model_adapter_factory` seam — no gateway, no API key. HTTP frames are read with a stdlib
`urlopen` streaming read; the in-process seam is consumed directly with `asyncio.run`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from support.http import serving
from support.runtime import runtime_config

from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.providers.base import ModelRequest, ModelStreamChunk, TextDelta, TurnComplete
from monoid_agent_kernel.providers.fake import FakeStreamingModelAdapter
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.backend.http import create_backend_server
from monoid_agent_kernel.reference.backend.service import BackendRunRequest, RunnerBackend


def _token_manager() -> TokenManager:
    return TokenManager.from_secret("x" * 32)


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


def _backend(tmp_path: Path, workspace: Path, factory) -> RunnerBackend:
    return RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
    )


def _streaming_backend(tmp_path: Path, workspace: Path, chunks: list[ModelStreamChunk]) -> RunnerBackend:
    return _backend(tmp_path, workspace, lambda spec, token: FakeStreamingModelAdapter(chunk_turns=[list(chunks)]))


def _request(workspace: Path) -> BackendRunRequest:
    return BackendRunRequest(
        tenant_id="tenant_a",
        user_id="user_a",
        workspace_root=workspace,
        instruction="go",
        runtime_config=runtime_config("run.finish"),
    )


def _run_payload(workspace: Path) -> dict[str, Any]:
    return {
        "tenant_id": "tenant_a",
        "user_id": "user_a",
        "workspace_root": str(workspace),
        "instruction": "go",
        "runtime_config": runtime_config("run.finish").to_json(),
    }


def _read_sse(base_url: str, payload: dict[str, Any], *, token: str = "admin") -> list[dict[str, Any]]:
    request = Request(
        f"{base_url}/v1/runs/stream",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=15) as response:
        assert response.headers.get("Content-Type", "").startswith("text/event-stream")
        raw = response.read().decode("utf-8")
    frames: list[dict[str, Any]] = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if block.startswith("data:"):
            frames.append(json.loads(block[len("data:") :].strip()))
    return frames


async def _collect(backend: RunnerBackend, request: BackendRunRequest) -> list[dict[str, Any]]:
    return [frame async for frame in backend.astream_run(request)]


# --- HTTP SSE transport ----------------------------------------------------------------


def test_backend_streams_run_over_sse(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    chunks = [TextDelta("Hel"), TextDelta("lo"), TurnComplete(response_id="prov", usage={"total_tokens": 5})]
    backend = _streaming_backend(tmp_path, workspace, chunks)
    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    with serving(server) as base_url:
        frames = _read_sse(base_url, _run_payload(workspace))

    kinds = [f["kind"] for f in frames]
    # Leading meta frame carries run id + token (mirrors BackendRunSubmission).
    assert kinds[0] == "meta"
    assert frames[0]["run_id"] and "run_token" in frames[0]
    # Token deltas stream and concatenate to the settled text.
    assert "".join(f["text"] for f in frames if f["kind"] == "delta" and f.get("type") == "text_delta") == "Hello"
    # Orchestration events stream too.
    event_types = {f["type"] for f in frames if f["kind"] == "event"}
    assert {"model.turn.started", "model.turn.finished"} <= event_types
    # Exactly one terminal result frame, last.
    assert kinds.count("result") == 1
    assert frames[-1] == frames[-1] and frames[-1]["kind"] == "result"
    assert frames[-1]["status"] == "completed"
    assert frames[-1]["final_text"] == "Hello"


def test_backend_stream_rejects_non_admin(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    backend = _streaming_backend(tmp_path, workspace, [TextDelta("x"), TurnComplete()])
    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    with serving(server) as base_url:
        with pytest.raises(HTTPError) as excinfo:
            _read_sse(base_url, _run_payload(workspace), token="wrong")
    assert excinfo.value.code == 401


# --- In-process programmatic seam (no HTTP) --------------------------------------------


def test_astream_run_programmatic_seam(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    backend = _streaming_backend(tmp_path, workspace, [TextDelta("done"), TurnComplete(response_id="prov")])
    frames = asyncio.run(_collect(backend, _request(workspace)))

    assert frames[0]["kind"] == "meta"
    assert any(f["kind"] == "delta" and f.get("type") == "text_delta" for f in frames)
    assert frames[-1]["kind"] == "result"
    assert frames[-1]["status"] == "completed"
    assert frames[-1]["final_text"] == "done"


def test_astream_run_emits_failed_result_on_adapter_error(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    class BoomAdapter:
        def next_turn(self, request: ModelRequest):  # pragma: no cover - astream_turn preferred
            raise AssertionError("astream_turn should be used")

        async def astream_turn(self, request: ModelRequest):
            if True:
                raise ModelAdapterError("provider blew up", provider_error_code="gateway_server_error")
            yield  # pragma: no cover - present only to make this an async generator

    backend = _backend(tmp_path, workspace, lambda spec, token: BoomAdapter())
    frames = asyncio.run(_collect(backend, _request(workspace)))

    # Exactly one terminal result frame, marking failure.
    assert sum(1 for f in frames if f["kind"] == "result") == 1
    assert frames[-1]["kind"] == "result"
    assert frames[-1]["status"] == "failed"
