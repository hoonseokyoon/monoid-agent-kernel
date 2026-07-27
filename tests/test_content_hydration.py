"""Settled-text hydration, at every transport that reads events.

The failure this guards against is silent. Every frontend read normalizes an absent field to `""`
and hides on truthiness, so a transport that forgets to hydrate renders an empty transcript with
no throw, no log, and nothing in the console. The only way to know a path is covered is to strip
the text from its events and watch it come back.

The events are rewritten into the shape the emit change will produce — ``final_text_digest`` and
no ``final_text`` — because until that lands the events still carry the text and hydration is
correctly a no-op. Testing the mechanism before it is load-bearing is the point: the alternative
is discovering a missed transport after the text is already gone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from support.backend_harness import (
    BackendRunRequest,
    RunnerBackend,
    _backend,
    _default_config,
    _json_get,
    _wait_http_ready,
    _workspace,
    create_backend_server,
    threading,
    Request,
    urlopen,
)

from monoid_agent_kernel.core.model_io import content_digest
from monoid_agent_kernel.reference.backend.content_hydration import (
    MAX_SCANNED_LINES,
    hydrate_settled_text,
)
from monoid_agent_kernel.reference.studio.chat_projection import ChatProjection

# Tier and the serial marker come from ``support/test_tiers.py`` (this module is registered in
# ``_INTEGRATION_MODULES``), so no per-module decorator here — the policy is what keeps the
# required PR shard deterministic.

SETTLE_TYPES = {"turn.settled", "run.finished"}


# --- unit: the resolver's own contract -------------------------------------------------------


def _write_record(run_dir: Path, text: str, **overrides: Any) -> str:
    digest = content_digest(text)
    record = {
        "kind": "settled_text",
        "final_text": text,
        "final_text_digest": digest,
        "final_text_len": len(text),
    }
    record.update(overrides)
    with (run_dir / "transcript.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record) + "\n")
    return digest


def _event(**data: Any) -> dict[str, Any]:
    return {"type": "turn.settled", "seq": 1, "data": data}


def test_absent_text_is_filled_from_the_record(tmp_path: Path) -> None:
    digest = _write_record(tmp_path, "the model wrote this")
    events = [_event(status="completed", final_text_digest=digest)]

    hydrate_settled_text(events, tmp_path)

    assert events[0]["data"]["final_text"] == "the model wrote this"


def test_present_text_is_never_overwritten_on_a_mixed_page(tmp_path: Path) -> None:
    """Kernel-authored text keeps travelling inline and must survive a page that also needs work.

    The page has to be *mixed*. With every event already carrying text there is nothing to
    resolve, so hydration returns before the fill loop and a version that overwrites
    indiscriminately passes anyway — verified: the single-event version of this test survived a
    mutant that deleted the fill loop's guard.
    """
    digest = _write_record(tmp_path, "the model wrote this")
    kernel = _event(status="limited", final_text="Stopped after reaching max steps.", final_text_digest=digest)
    needs_fill = _event(status="completed", final_text_digest=digest)
    events = [kernel, needs_fill]

    hydrate_settled_text(events, tmp_path)

    assert kernel["data"]["final_text"] == "Stopped after reaching max steps."
    assert needs_fill["data"]["final_text"] == "the model wrote this"


def test_a_missing_transcript_never_fails_the_read(tmp_path: Path) -> None:
    # Durability is best-effort: no fsync, no append-tail repair. A committed event whose digest
    # resolves to nothing must degrade to an absent field, not a dead endpoint.
    events = [_event(status="completed", final_text_digest="a" * 64)]

    hydrate_settled_text(events, tmp_path / "does-not-exist")

    assert "final_text" not in events[0]["data"]


def test_a_malformed_line_does_not_hide_a_later_record(tmp_path: Path) -> None:
    with (tmp_path / "transcript.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("{not json\n")
        handle.write(json.dumps({"kind": "model_turn", "final_text": "wrong kind"}) + "\n")
    digest = _write_record(tmp_path, "found anyway")
    events = [_event(status="completed", final_text_digest=digest)]

    hydrate_settled_text(events, tmp_path)

    assert events[0]["data"]["final_text"] == "found anyway"


def test_malformed_event_shapes_are_skipped(tmp_path: Path) -> None:
    _write_record(tmp_path, "unused")
    events: list[Any] = ["not a dict", {"data": "not a mapping"}, {"no": "data"}]

    hydrate_settled_text(events, tmp_path)  # must not raise
    hydrate_settled_text(None, tmp_path)
    hydrate_settled_text({"events": []}, tmp_path)


def test_the_transcript_is_not_opened_when_nothing_wants_text(tmp_path: Path, monkeypatch: Any) -> None:
    """Today every event still carries its text, so this is the only path taken in production.

    Asserted rather than assumed: a resolver that ran unconditionally would read the transcript on
    every page of every run for no benefit.
    """
    from monoid_agent_kernel.reference.backend import content_hydration

    calls: list[Any] = []
    monkeypatch.setattr(content_hydration, "_resolve", lambda *a, **k: calls.append(a) or {})

    content_hydration.hydrate_settled_text([_event(status="completed", final_text="present")], tmp_path)

    assert calls == []


def test_the_scan_is_bounded(tmp_path: Path) -> None:
    # An unbounded lookup would regress the resource bounds this repo applies to its other readers.
    with (tmp_path / "transcript.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        for index in range(MAX_SCANNED_LINES + 5):
            handle.write(json.dumps({"kind": "model_turn", "step": index}) + "\n")
    digest = _write_record(tmp_path, "beyond the cap")
    events = [_event(status="completed", final_text_digest=digest)]

    hydrate_settled_text(events, tmp_path)

    assert "final_text" not in events[0]["data"]


# --- integration: every transport that reads events ------------------------------------------


def _strip_text_from_settle_events(run_dir: Path, text: str) -> str:
    """Rewrite committed settle events into the shape the emit change will produce.

    Asserts that it actually stripped something. If the run's settled text ever stops matching
    ``text`` this becomes a no-op, and every transport test below would then "pass" by finding
    text that was never removed — the tests would be measuring nothing and still be green.
    """
    digest = content_digest(text)
    path = run_dir / "events.jsonl"
    rewritten: list[str] = []
    stripped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        data = event.get("data") or {}
        if event.get("type") in SETTLE_TYPES and data.get("final_text") == text:
            data.pop("final_text")
            data["final_text_digest"] = digest
            data["final_text_len"] = len(text)
            event["data"] = data
            stripped += 1
        rewritten.append(json.dumps(event, ensure_ascii=False, sort_keys=True))
    assert stripped == 2, f"expected to strip turn.settled and run.finished, stripped {stripped}"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(rewritten) + "\n")
    return digest


def _completed_run(tmp_path: Path) -> tuple[RunnerBackend, Any]:
    workspace = _workspace(tmp_path)
    backend = _backend(tmp_path, workspace, [])
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Run.",
            runtime_config=_default_config(),
        )
    )
    assert backend.wait_for_run(submission.run_id, timeout_s=10).value == "completed"
    return backend, submission


def _settled_texts(events: list[dict[str, Any]]) -> list[str | None]:
    return [
        event["data"].get("final_text")
        for event in events
        if event.get("type") in SETTLE_TYPES
    ]


def test_the_record_is_written_by_a_real_run(tmp_path: Path) -> None:
    # Guards the premise of every test below: if the run stopped writing the record, the
    # transports would "pass" by finding nothing to hydrate and returning nothing.
    _backend_obj, submission = _completed_run(tmp_path)

    transcript = (submission.run_dir / "transcript.jsonl").read_text(encoding="utf-8")
    kinds = [json.loads(line)["kind"] for line in transcript.splitlines() if line.strip()]
    assert "settled_text" in kinds


def test_backend_projection_hydrates(tmp_path: Path) -> None:
    backend, submission = _completed_run(tmp_path)
    _strip_text_from_settle_events(submission.run_dir, "done")

    page = backend.events(submission.run_id, submission.run_token)

    assert _settled_texts(page["events"]) == ["done", "done"]


def test_backend_http_json_twin_hydrates(tmp_path: Path) -> None:
    backend, submission = _completed_run(tmp_path)
    _strip_text_from_settle_events(submission.run_dir, "done")
    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        _wait_http_ready(base_url)
        page = _json_get(
            f"{base_url}/v1/runs/{submission.run_id}/events?from_seq=0",
            token=submission.run_token,
        )
        assert _settled_texts(page["events"]) == ["done", "done"]
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_backend_http_sse_twin_hydrates(tmp_path: Path) -> None:
    """The twin the original plan missed entirely.

    Same route, same handler, different Accept header — and a different code path from the JSON
    branch. A test that only covered the JSON twin would have declared this transport safe.
    """
    backend, submission = _completed_run(tmp_path)
    _strip_text_from_settle_events(submission.run_dir, "done")
    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        _wait_http_ready(base_url)
        request = Request(
            f"{base_url}/v1/runs/{submission.run_id}/events?from_seq=0",
            headers={
                "Authorization": f"Bearer {submission.run_token}",
                "Accept": "text/event-stream",
            },
        )
        with urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8")
    finally:
        server.shutdown()
        thread.join(timeout=5)

    texts: list[str] = []
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            frame = json.loads(line[len("data:") :].strip())
        except ValueError:
            continue
        if isinstance(frame, dict) and frame.get("type") in SETTLE_TYPES:
            texts.append((frame.get("data") or {}).get("final_text"))
    assert texts == ["done", "done"], body[:400]


def test_studio_chat_projection_catch_up_hydrates(tmp_path: Path) -> None:
    """Studio's catch-up reads events.jsonl directly, bypassing the backend projection.

    Its consumer drops an assistant message whose content is empty, so a miss here is not a blank
    bubble — it is a message that silently never appears.
    """
    _backend_obj, submission = _completed_run(tmp_path)
    _strip_text_from_settle_events(submission.run_dir, "done")

    body = ChatProjection(submission.run_dir).catch_up(submission.run_id)

    assistant = [message for message in body["messages"] if message["role"] == "assistant"]
    assert [message["content"] for message in assistant] == ["done"]


def test_descendant_feed_hydrates_from_the_child_run_dir(tmp_path: Path) -> None:
    """Hydration must use the *child's* run dir, not the parent's.

    This endpoint exists because Studio cannot reach a child run dir at all, which is why the
    plan's Studio-side hydration site was unimplementable. Resolving against the parent would
    find nothing and the assistant text would vanish for every subagent.
    """
    backend, submission = _completed_run(tmp_path)
    child_run_id = f"{submission.run_id}.sub.task1"
    child_dir = submission.run_dir.parent / child_run_id
    child_dir.mkdir(parents=True)
    digest = _write_record(child_dir, "the child answered")
    event = {
        "schema_version": "monoid.event.v1",
        "event_id": "evt_child",
        "seq": 1,
        "run_id": child_run_id,
        "timestamp": "2026-07-27T00:00:00Z",
        "type": "turn.settled",
        "level": "info",
        "data": {"status": "completed", "final_text_digest": digest, "final_text_len": 18},
    }
    with (child_dir / "events.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event) + "\n")

    page = backend.descendant_events(submission.run_id, submission.run_token, child_run_id)

    assert _settled_texts(page["events"]) == ["the child answered"]
