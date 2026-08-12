"""P4b-②: backend transport-neutral streaming output (HTTP SSE + programmatic seam).

The whole path is exercised with a `FakeStreamingModelAdapter` injected via the backend's
`model_adapter_factory` seam — no gateway, no API key. HTTP frames are read with a stdlib
`urlopen` streaming read; the in-process seam is consumed directly with `asyncio.run`.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from monoid_agent_kernel.core.content import TextPart
from support.http import LINE_SEPARATORS, serving, sse_data_frames_by_line
from support.runtime import runtime_config

from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.providers.base import ModelRequest, ModelStreamChunk, ModelTurn, TextDelta, TurnComplete
from monoid_agent_kernel.providers.fake import FakeModelAdapter, FakeStreamingModelAdapter
from monoid_agent_kernel.reference.backend.http import create_backend_server
from monoid_agent_kernel.reference.backend.service import BackendRunRequest, RunnerBackend

_SETTLE_TYPES = {"turn.settled", "run.finished"}


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


def _backend(backend_factory: Any, workspace: Path, factory) -> RunnerBackend:
    return backend_factory.create(workspace=workspace, model_adapter_factory=factory)


def _streaming_backend(
    backend_factory: Any, workspace: Path, chunks: list[ModelStreamChunk]
) -> RunnerBackend:
    return _backend(
        backend_factory,
        workspace,
        lambda spec, token: FakeStreamingModelAdapter(chunk_turns=[list(chunks)]),
    )


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


def test_backend_streams_run_over_sse(tmp_path: Path, backend_factory: Any) -> None:
    workspace = _workspace(tmp_path)
    chunks = [TextDelta("Hel"), TextDelta("lo"), TurnComplete(response_id="prov", usage={"total_tokens": 5})]
    backend = _streaming_backend(backend_factory, workspace, chunks)
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


def test_a_line_splitting_client_reads_a_run_stream_whose_model_emitted_a_separator(
    tmp_path: Path, backend_factory: Any
) -> None:
    """SSE is a line protocol and the two ends did not agree on what a line is.

    The writer dumped with ``ensure_ascii=False``, so U+2028, U+2029 and U+0085 went onto the wire
    as themselves and a ``str.splitlines`` reader -- httpx's ``aiter_lines``, what any third-party
    consumer of this route reads with -- broke the frame there. The client then parsed JSON that
    stopped mid-string and saw a failed run the server had produced, framed and already metered.

    Both carriers are pinned because they are separate frames: the ``delta`` frames are raw model
    text, and the terminal ``result`` frame carries ``final_text`` even when no delta did. The
    reader is the point of the test -- :func:`_read_sse` above splits on ``\\n\\n`` and passes
    against the bug.
    """

    pytest.importorskip("httpx")
    answer = f"before{LINE_SEPARATORS}after"
    workspace = _workspace(tmp_path)
    chunks = [
        TextDelta(answer),
        TurnComplete(response_id="prov", usage={"total_tokens": 5}),
    ]
    backend = _streaming_backend(backend_factory, workspace, chunks)
    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    with serving(server) as base_url:
        frames = sse_data_frames_by_line(
            f"{base_url}/v1/runs/stream",
            token="admin",
            payload=_run_payload(workspace),
        )

    deltas = [f for f in frames if f["kind"] == "delta" and f.get("type") == "text_delta"]
    assert "".join(f["text"] for f in deltas) == answer
    assert frames[-1]["kind"] == "result"
    assert frames[-1]["status"] == "completed"
    assert frames[-1]["final_text"] == answer


def test_backend_stream_rejects_non_admin(tmp_path: Path, backend_factory: Any) -> None:
    workspace = _workspace(tmp_path)
    backend = _streaming_backend(backend_factory, workspace, [TextDelta("x"), TurnComplete()])
    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    with serving(server) as base_url:
        with pytest.raises(HTTPError) as excinfo:
            _read_sse(base_url, _run_payload(workspace), token="wrong")
    assert excinfo.value.code == 401


# --- In-process programmatic seam (no HTTP) --------------------------------------------


def test_astream_run_hydrates_event_frames_and_not_delta_frames(
    tmp_path: Path, backend_factory: Any, monkeypatch: Any
) -> None:
    """The live stream is a read path of its own, and it degrades asymmetrically without this.

    ``kind:event`` frames carry the settle events whose model text moves to the run-dir record;
    ``kind:delta`` frames carry live token text that no turn-end record can supply, and
    ``kind:result`` reads off ``AgentRunResult`` rather than the event stream. Hydrating the wrong
    set is invisible while the events still carry their text, so the wiring is asserted directly
    rather than through an observable outcome.
    """
    from monoid_agent_kernel.reference.backend import run_execution

    hydrated_kinds: list[Any] = []
    hydrated_dirs: list[Any] = []
    hydrated_threads: list[int] = []
    aliased: list[bool] = []
    streamed_items: list[Any] = []
    original = run_execution.hydrate_settled_text
    original_frame = run_execution.stream_item_frame

    def frame_spy(item: Any) -> Any:
        streamed_items.append(item)
        return original_frame(item)

    def spy(events: Any, run_dir: Any) -> Any:
        hydrated_threads.append(threading.get_ident())
        hydrated_dirs.append(run_dir)
        for event in events:
            hydrated_kinds.append(event.get("kind"))
            # The frame's ``data`` must not BE the AgentEvent's own dict. Filling it in place
            # would write hydrated text into the event the bus owns and every sink shares.
            aliased.append(
                any(event.get("data") is getattr(item, "data", None) for item in streamed_items)
            )
        return original(events, run_dir)

    monkeypatch.setattr(run_execution, "stream_item_frame", frame_spy)
    monkeypatch.setattr(run_execution, "hydrate_settled_text", spy)
    # Driven by real emissions. This previously monkeypatched ``needs_settled_text`` to claim every
    # ``kind: event`` frame needed text, because nothing carried a digest until the emit change
    # landed. Now that settle events do, that patch would be a mask — it would mark frames the real
    # gate skips, so the test would keep passing if the gate broke.

    workspace = _workspace(tmp_path)
    backend = _streaming_backend(
        backend_factory, workspace, [TextDelta("done"), TurnComplete(response_id="prov")]
    )
    frames = asyncio.run(_collect(backend, _request(workspace)))

    # The run really did stream both shapes, so "only events were hydrated" is a choice rather
    # than an artefact of nothing else being there.
    assert any(frame["kind"] == "delta" for frame in frames)
    assert any(frame["kind"] == "event" for frame in frames)
    assert hydrated_kinds and set(hydrated_kinds) == {"event"}
    # Resolved against the run's OWN directory. Passing the run *root* would resolve nothing and
    # the live stream would silently lose its text.
    run_id = frames[0]["run_id"]
    assert hydrated_dirs and all(directory.name == run_id for directory in hydrated_dirs)
    assert aliased and not any(aliased)
    # Resolved OFF the event loop. The scan has no positional bound — any window drops text a
    # reader legitimately asked for — so inline it blocked the shared run loop for a whole
    # transcript read, delaying every other concurrently streaming run.
    assert hydrated_threads and threading.get_ident() not in hydrated_threads


def test_astream_run_hops_threads_only_for_the_frames_carrying_a_digest(
    tmp_path: Path, backend_factory: Any, monkeypatch: Any
) -> None:
    """Only the two settle frames need resolving; the rest must not pay for a thread hop.

    The executor the hop targets is process-wide and bounded (32 workers), and runs parked awaiting
    hosted tasks hold a worker each for up to `task_wait_poll_s`. An unconditional hop would queue
    every event frame's delivery behind them to do no work at all.

    This asserted ``hydrated == []`` when nothing on the wire carried a digest. That premise died
    with the emit change, and the assertion would have failed loudly — but the *property* it was
    protecting is unchanged and still worth pinning, so it is now stated as a ratio against the
    frames that really streamed rather than as an absolute zero.
    """
    from monoid_agent_kernel.reference.backend import run_execution

    hydrated: list[Any] = []
    original = run_execution.hydrate_settled_text

    def spy(events: Any, run_dir: Any) -> Any:
        hydrated.append(events)
        return original(events, run_dir)

    monkeypatch.setattr(run_execution, "hydrate_settled_text", spy)

    workspace = _workspace(tmp_path)
    backend = _streaming_backend(
        backend_factory, workspace, [TextDelta("done"), TurnComplete(response_id="prov")]
    )
    frames = asyncio.run(_collect(backend, _request(workspace)))

    event_frames = [frame for frame in frames if frame["kind"] == "event"]
    settled = [frame for frame in event_frames if frame.get("type") in _SETTLE_TYPES]
    # Non-empty first, so the ratio below cannot hold vacuously at 0 == 0. Only ``turn.settled``
    # reaches this stream — ``run.finished`` is emitted after it closes — so the count is derived
    # from the frames rather than hardcoded.
    assert settled, [frame.get("type") for frame in event_frames]
    # The discrimination is the point: exactly one hop per frame carrying a digest, and strictly
    # fewer than the event frames. Equality with `len(event_frames)` would mean the gate does
    # nothing; zero would mean it never resolves.
    assert len(hydrated) == len(settled)
    assert len(hydrated) < len(event_frames)
    # And the frame the reader receives really did get its text back.
    assert settled[0]["data"]["final_text"] == "done"
    assert settled[0]["data"]["final_text_digest"]


def test_astream_run_programmatic_seam(tmp_path: Path, backend_factory: Any) -> None:
    workspace = _workspace(tmp_path)
    backend = _streaming_backend(
        backend_factory, workspace, [TextDelta("done"), TurnComplete(response_id="prov")]
    )
    frames = asyncio.run(_collect(backend, _request(workspace)))

    assert frames[0]["kind"] == "meta"
    assert any(f["kind"] == "delta" and f.get("type") == "text_delta" for f in frames)
    assert frames[-1]["kind"] == "result"
    assert frames[-1]["status"] == "completed"
    assert frames[-1]["final_text"] == "done"


def test_astream_run_result_projection_ready_when_result_frame_emits(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    backend = _streaming_backend(
        backend_factory,
        workspace,
        [TextDelta("ok"), TurnComplete(response_id="prov", usage={"total_tokens": 7})],
    )
    frames = asyncio.run(_collect(backend, _request(workspace)))
    meta = frames[0]

    result = backend.result(meta["run_id"], meta["run_token"])

    assert frames[-1]["kind"] == "result"
    assert result["ready"] is True
    assert result["terminal"] is True
    assert result["state"] == "completed"
    assert result["final_text"] == "ok"
    assert result["metrics"]["total_tokens"] == 7


def test_astream_run_emits_failed_result_on_adapter_error(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)

    class BoomAdapter:
        def next_turn(self, request: ModelRequest):  # pragma: no cover - astream_turn preferred
            raise AssertionError("astream_turn should be used")

        async def astream_turn(self, request: ModelRequest):
            if True:
                raise ModelAdapterError("provider blew up", provider_error_code="gateway_server_error")
            yield  # pragma: no cover - present only to make this an async generator

    backend = _backend(backend_factory, workspace, lambda spec, token: BoomAdapter())
    frames = asyncio.run(_collect(backend, _request(workspace)))

    # Exactly one terminal result frame, marking failure.
    assert sum(1 for f in frames if f["kind"] == "result") == 1
    assert frames[-1]["kind"] == "result"
    assert frames[-1]["status"] == "failed"
    assert frames[-1]["error_code"] == "model_error"
    meta = frames[0]
    result = backend.result(meta["run_id"], meta["run_token"])
    assert result["ready"] is True
    assert result["terminal"] is True
    assert result["state"] == "failed"
    assert result["error_code"] == "model_error"
    assert (backend._record(meta["run_id"]).run_dir / "failure.json").exists()


def test_astream_run_worker_failure_after_meta_releases_slot(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    calls = 0

    def factory(spec: Any, token: str) -> FakeStreamingModelAdapter:
        nonlocal calls
        del spec, token
        calls += 1
        if calls == 1:
            raise RuntimeError("factory blew up")
        return FakeStreamingModelAdapter(chunk_turns=[[TextDelta("next"), TurnComplete(response_id="ok")]])

    backend = backend_factory.create(
        workspace=workspace,
        model_adapter_factory=factory,
        max_concurrent_runs=1,
    )

    failed = asyncio.run(_collect(backend, _request(workspace)))
    failed_meta = failed[0]
    assert failed[-1]["kind"] == "result"
    assert failed[-1]["status"] == "failed"
    assert failed[-1]["error_code"] == "internal_error"
    failed_result = backend.result(failed_meta["run_id"], failed_meta["run_token"])
    assert failed_result["ready"] is False
    assert failed_result["terminal"] is True
    assert failed_result["state"] == "failed"
    assert (backend._record(failed_meta["run_id"]).run_dir / "failure.json").exists()

    completed = asyncio.run(_collect(backend, _request(workspace)))
    assert completed[-1]["kind"] == "result"
    assert completed[-1]["status"] == "completed"
    assert completed[-1]["final_text"] == "next"


def test_first_turn_input_parts_reach_submit_and_stream_paths(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    submit_adapters: list[FakeModelAdapter] = []
    stream_adapters: list[FakeStreamingModelAdapter] = []

    submit_backend = backend_factory.create(
        workspace=workspace,
        model_adapter_factory=lambda spec, token: submit_adapters.append(
            FakeModelAdapter(turns=[ModelTurn(response_id="submit", final_text="submit done")])
        )
        or submit_adapters[-1],
    )
    submission = submit_backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="ignored title",
            input_parts=(TextPart("from parts"),),
            runtime_config=runtime_config("run.finish"),
        )
    )
    assert submit_backend.wait_for_run(submission.run_id, timeout_s=20) == "completed"
    assert submit_adapters[0].requests[0].messages[0]["content"] == "from parts"

    stream_backend = backend_factory.create(
        workspace=workspace,
        model_adapter_factory=lambda spec, token: stream_adapters.append(
            FakeStreamingModelAdapter(chunk_turns=[[TextDelta("stream done"), TurnComplete()]])
        )
        or stream_adapters[-1],
    )
    frames = asyncio.run(
        _collect(
            stream_backend,
            BackendRunRequest(
                tenant_id="tenant_a",
                user_id="user_a",
                workspace_root=workspace,
                instruction="ignored title",
                input_parts=(TextPart("stream parts"),),
                runtime_config=runtime_config("run.finish"),
            ),
        )
    )
    assert frames[-1]["status"] == "completed"
    assert stream_adapters[0].requests[0].messages[0]["content"] == "stream parts"
