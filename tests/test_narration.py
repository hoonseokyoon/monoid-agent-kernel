"""The shared event->narration projection (used by the watch CLI and the Studio feed)."""

from __future__ import annotations

from native_agent_runner.narration import EventNarration, narrate_event


def test_narrate_tool_started_read() -> None:
    narration = narrate_event(
        {"type": "tool.call.started", "data": {"tool": "fs_read", "args_preview": {"path": "notes.md"}}}
    )
    assert narration == EventNarration(category="tool", action="read", target="notes.md", status="start")


def test_narrate_tool_started_write_and_unknown_tool() -> None:
    write = narrate_event(
        {"type": "tool.call.started", "data": {"tool": "fs_write", "args_preview": {"path": "a.md", "content": "x"}}}
    )
    assert write is not None and write.action == "write" and write.target == "a.md"
    # An unrecognized tool falls back to a generic "run", targeting the tool name.
    unknown = narrate_event({"type": "tool.call.started", "data": {"tool": "mystery_tool", "args_preview": {}}})
    assert unknown is not None and unknown.action == "run" and unknown.target == "mystery_tool"


def test_narrate_finished_ok_is_none_but_failure_carries_detail() -> None:
    assert narrate_event({"type": "tool.call.finished", "data": {"tool": "fs_read", "ok": True}}) is None
    failed = narrate_event({"type": "tool.call.failed", "data": {"tool": "fs_read", "error": "boom"}})
    assert failed is not None and failed.status == "error" and failed.detail == "boom"
    finished_bad = narrate_event(
        {"type": "tool.call.finished", "data": {"tool": "fs_write", "ok": False, "error_code": "E_DISK"}}
    )
    assert finished_bad is not None and finished_bad.status == "error" and finished_bad.detail == "E_DISK"


def test_narrate_target_from_paths_and_url() -> None:
    listing = narrate_event({"type": "tool.call.started", "data": {"tool": "fs_tree", "paths": ["src", "docs"]}})
    assert listing is not None and listing.action == "list" and listing.target == "src, docs"
    fetch = narrate_event(
        {"type": "tool.call.started", "data": {"tool": "web_fetch", "args_preview": {"url": "http://x"}}}
    )
    assert fetch is not None and fetch.action == "fetch" and fetch.target == "http://x"


def test_narrate_non_tool_events_return_none() -> None:
    for event_type in ("run.started", "turn.settled", "model.turn.started", "workspace.file.read"):
        assert narrate_event({"type": event_type, "data": {}}) is None
