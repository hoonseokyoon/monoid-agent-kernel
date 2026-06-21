"""Turn raw ``AgentEvent`` dicts into a human-readable activity feed line.

This is the reference answer to "how do I show what the agent is doing?" — it maps the public
event stream to short status lines for a UI. Kept server-side (not in the browser JS) so it is a
tested Python artifact and a copyable example.

DX note (see DX_NOTES.md, DX-3): there is no presentation-ready field on events — no human
``summary`` and no typed ``(verb, target)`` — so this has to hand-maintain a verb table keyed by
the wire tool name and heuristically dig the target out of ``args_preview`` / ``paths``. A core
``event.summary`` (or a structured tool-activity event) would remove this guesswork.
"""

from __future__ import annotations

from typing import Any

# Wire tool name (dots already sanitized to underscores) -> present-progressive verb.
_TOOL_VERBS = {
    "fs_read": "Reading",
    "fs_read_media": "Reading",
    "fs_tree": "Listing",
    "fs_write": "Writing",
    "fs_patch": "Editing",
    "fs_delete": "Deleting",
    "fs_move": "Moving",
    "fs_copy": "Copying",
    "fs_mkdir": "Creating directory",
    "shell_exec": "Running",
    "web_search": "Searching the web for",
    "web_fetch": "Fetching",
    "web_context": "Researching",
}

# args_preview keys to surface as the action target, in priority order.
_TARGET_KEYS = ("path", "url", "query", "command", "cmd", "pattern", "src", "dest")


def _target(data: dict[str, Any]) -> str:
    args = data.get("args_preview") or {}
    if isinstance(args, dict):
        for key in _TARGET_KEYS:
            value = args.get(key)
            if value:
                return str(value)
    paths = data.get("paths")
    if isinstance(paths, (list, tuple)) and paths:
        return ", ".join(str(p) for p in paths)
    return ""


def describe_event(event: dict[str, Any]) -> str | None:
    """A short activity line for ``event``, or ``None`` if it should not appear in the feed.

    Only tool activity is surfaced; chat text (``turn.settled`` / ``run.finished``) and internal
    lifecycle events are left to the chat transcript and return ``None`` here.
    """
    etype = event.get("type") or ""
    data = event.get("data") or {}
    if not isinstance(data, dict):
        return None
    if etype == "tool.call.started":
        tool = str(data.get("tool") or "tool")
        verb = _TOOL_VERBS.get(tool)
        target = _target(data)
        if verb:
            return f"{verb} {target}".strip() if target else verb
        return f"Running {tool}" + (f" ({target})" if target else "")
    if etype == "tool.call.finished":
        if data.get("ok", True):
            return None  # success is implied by the next step; don't clutter the feed
        tool = str(data.get("tool") or "tool")
        error = str(data.get("error") or data.get("error_code") or "failed")
        return f"⚠ {tool} failed: {error}"
    return None
