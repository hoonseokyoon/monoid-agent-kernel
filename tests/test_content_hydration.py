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
from monoid_agent_kernel.reference.backend.content_hydration import hydrate_settled_text
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
    # Durability is best-effort: no fsync, and the only repair confines a torn line rather than
    # recovering it. A committed event whose digest
    # resolves to nothing must degrade to an absent field, not a dead endpoint.
    events = [_event(status="completed", final_text_digest="a" * 64)]

    hydrate_settled_text(events, tmp_path / "does-not-exist")

    assert "final_text" not in events[0]["data"]


def test_a_malformed_line_does_not_hide_a_later_record(tmp_path: Path) -> None:
    digest = content_digest("found anyway")
    with (tmp_path / "transcript.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("{not json\n")
        # The decoy carries the wanted digest, so it reaches the digest check rather than being
        # rejected earlier on `digest is None` — which is all the previous version tested.
        #
        # It does NOT pin the `kind` check, and nothing can: digest verification subsumes it. A
        # wrong-kind record can only be accepted if its text hashes to the digest asked for, which
        # makes it byte-identical to the record wanted. Verified — dropping the kind check leaves
        # this green. The check stays as defence-in-depth against a future record shape that
        # carries `final_text_digest` for another purpose; it is deliberately not observable today.
        handle.write(
            json.dumps(
                {
                    "kind": "model_turn",
                    "final_text": "found anyway",
                    "final_text_digest": digest,
                }
            )
            + "\n"
        )
    _write_record(tmp_path, "found anyway")
    events = [_event(status="completed", final_text_digest=digest)]

    hydrate_settled_text(events, tmp_path)

    assert events[0]["data"]["final_text"] == "found anyway"


def test_only_the_wanted_digest_is_resolved_when_several_records_exist(tmp_path: Path) -> None:
    """A two-turn run writes several records; the page must get the one it asked for.

    The `digest not in wanted` filter is not tidiness — with it removed, the scan takes the FIRST
    settled-text record it meets and the wanted digest is never found, so the field silently stays
    absent. No other fixture writes more than one record, so nothing else pins this.
    """
    first = _write_record(tmp_path, "first answer")
    second = _write_record(tmp_path, "second answer")
    assert first != second
    events = [_event(status="completed", final_text_digest=second)]

    hydrate_settled_text(events, tmp_path)

    assert events[0]["data"]["final_text"] == "second answer"


def test_a_page_wanting_two_digests_resolves_both(tmp_path: Path) -> None:
    # The early exit stops at `len(found) == len(wanted)`; with one wanted digest that is
    # indistinguishable from stopping at the first match.
    first = _write_record(tmp_path, "first answer")
    second = _write_record(tmp_path, "second answer")
    events = [
        _event(status="completed", final_text_digest=first),
        _event(status="completed", final_text_digest=second),
    ]

    hydrate_settled_text(events, tmp_path)

    assert [event["data"]["final_text"] for event in events] == ["first answer", "second answer"]


def test_a_torn_utf8_sequence_does_not_fail_the_read(tmp_path: Path) -> None:
    """A crash can tear a multi-byte sequence mid-write.

    Decoding is lazy, so strict UTF-8 raises ``UnicodeDecodeError`` from the iterator — a
    ``ValueError``, which slips past the ``OSError`` handler and turns every read needing a
    digest into a failed request. The tear has to come *before* the wanted record: the scan exits
    as soon as every digest is found, so a tear in the trailing bytes is never decoded at all and
    proves nothing.
    """
    digest = content_digest("survived the tear")
    record = json.dumps(
        {
            "kind": "settled_text",
            "final_text": "survived the tear",
            "final_text_digest": digest,
            "final_text_len": len("survived the tear"),
        }
    )
    torn = "한".encode("utf-8")[:2] + b"\n"
    (tmp_path / "transcript.jsonl").write_bytes(torn + record.encode("utf-8") + b"\n")
    events = [_event(status="completed", final_text_digest=digest)]

    hydrate_settled_text(events, tmp_path)

    assert events[0]["data"]["final_text"] == "survived the tear"


def test_a_record_whose_text_does_not_match_its_digest_is_refused(tmp_path: Path) -> None:
    """The digest names the content, so the join verifies rather than trusts.

    A torn or character-replaced line can still decode to a well-formed record whose text is no
    longer what its digest names. Handing that back would be worse than handing back nothing.
    """
    digest = content_digest("the real answer")
    _write_record(tmp_path, "the real answer", final_text="tampered text")
    events = [_event(status="completed", final_text_digest=digest)]

    hydrate_settled_text(events, tmp_path)

    assert "final_text" not in events[0]["data"]


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

    # The event carries a digest AND the text. Without the digest this short-circuited on the
    # missing digest instead of the text-present guard, so deleting that guard left it green.
    digest = _write_record(tmp_path, "already inline")
    content_hydration.hydrate_settled_text(
        [_event(status="completed", final_text="present", final_text_digest=digest)], tmp_path
    )

    assert calls == []


def _pad(run_dir: Path, byte_target: int) -> None:
    with (run_dir / "transcript.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        written = 0
        index = 0
        while written < byte_target:
            line = json.dumps({"kind": "model_turn", "step": index, "pad": "x" * 200}) + "\n"
            handle.write(line)
            written += len(line)
            index += 1


def test_position_in_the_transcript_does_not_decide_whether_text_resolves(tmp_path: Path) -> None:
    """Any record for a wanted digest must be found, wherever it sits in the file.

    Two positional bounds were tried and both lost text in mirrored ways: a line cap counted from
    the START dropped the newest settled text, and the same budget anchored at the END dropped the
    oldest — which broke Studio catch-up, since it hydrates every committed event of a run at once
    and so wants digests spanning the whole session.

    The bound was anchored to the *file*; the reader's need is anchored to the *event set it was
    asked about*. This pins the invariant that replaced it, from both ends of a padded transcript.
    """
    oldest = _write_record(tmp_path, "the first answer")
    _pad(tmp_path, 200_000)
    newest = _write_record(tmp_path, "the last answer")
    _pad(tmp_path, 200_000)
    events = [
        _event(status="completed", final_text_digest=oldest),
        _event(status="completed", final_text_digest=newest),
    ]

    hydrate_settled_text(events, tmp_path)

    assert [event["data"]["final_text"] for event in events] == [
        "the first answer",
        "the last answer",
    ]


def test_an_unresolvable_event_is_not_filled_with_another_events_text(tmp_path: Path) -> None:
    """Serving the wrong text is the worst outcome available here, worse than serving none.

    Needs a MIXED page: with nothing resolvable at all there is no other text to mis-fill from, so
    a hydrator that ignores the digest entirely and takes whatever it found still passes. Verified
    — the single-event version of this survived exactly that mutant.
    """
    resolvable = _write_record(tmp_path, "the answer that exists")
    events = [
        _event(status="completed", final_text_digest=resolvable),
        _event(status="completed", final_text_digest=content_digest("lost to a crash")),
    ]

    hydrate_settled_text(events, tmp_path)

    assert events[0]["data"]["final_text"] == "the answer that exists"
    assert "final_text" not in events[1]["data"]


def test_a_digest_with_no_record_anywhere_stays_absent(tmp_path: Path) -> None:
    # The counterweight to "position never excludes": absence must still read as absence, or the
    # test above would pass against a resolver that invented text.
    _write_record(tmp_path, "a different answer")
    _pad(tmp_path, 50_000)
    events = [_event(status="completed", final_text_digest=content_digest("never recorded"))]

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
