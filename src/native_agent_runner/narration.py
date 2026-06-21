"""Project a single event into a neutral, structured narration for UIs.

Events are typed and structured (see ``core/events.py``); turning one into something a person
reads is a presentation concern. So — exactly as ``OtelEventSink`` maps the event stream to
spans — this maps one event to a small *neutral* descriptor that any renderer (the ``watch`` CLI,
a web activity feed, …) formats in its own style and locale. It deliberately does **not** bake a
localized sentence into the event itself, matching how agent UI protocols (AG-UI, the Vercel AI
SDK, the OTel GenAI conventions) keep events typed and render at the edge.

Shared by the ``native-agent watch`` CLI and the Studio activity feed, so the "which events
matter + what is the verb/target" logic lives in one place instead of drifting across renderers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Wire tool name (registry id with dots sanitized to underscores) -> neutral action token.
_TOOL_ACTIONS = {
    "fs_read": "read",
    "fs_read_media": "read",
    "fs_tree": "list",
    "fs_write": "write",
    "fs_patch": "edit",
    "fs_delete": "delete",
    "fs_move": "move",
    "fs_copy": "copy",
    "fs_mkdir": "mkdir",
    "shell_exec": "run",
    "web_search": "search",
    "web_fetch": "fetch",
    "web_context": "research",
}
# args_preview keys to surface as the action's primary target, in priority order.
_TARGET_KEYS = ("path", "url", "query", "command", "cmd", "pattern", "src", "dest")


@dataclass(frozen=True)
class EventNarration:
    """A renderer-neutral description of one event. Formatters turn this into their own text."""

    category: str  # coarse grouping, e.g. "tool"
    action: str  # neutral verb token: read / write / search / run / ...
    target: str  # the primary object (path / url / query / command), or ""
    status: str  # "start" | "ok" | "error"
    level: str = "info"  # passthrough of the event level
    detail: str = ""  # extra context, e.g. an error message


def _target(data: Mapping[str, Any]) -> str:
    args = data.get("args_preview")
    if isinstance(args, Mapping):
        for key in _TARGET_KEYS:
            value = args.get(key)
            if value:
                return str(value)
    paths = data.get("paths")
    if isinstance(paths, (list, tuple)) and paths:
        return ", ".join(str(p) for p in paths)
    return ""


def narrate_event(event: Mapping[str, Any]) -> EventNarration | None:
    """A structured narration for a tool-activity event, or ``None`` for events with no natural
    narration (lifecycle, model turns, low-level workspace events — left to the renderer)."""
    etype = event.get("type") or ""
    data = event.get("data")
    if not isinstance(data, Mapping):
        data = {}
    level = str(event.get("level") or "info")
    if etype == "tool.call.started":
        tool = str(data.get("tool") or "")
        action = _TOOL_ACTIONS.get(tool, "run")
        target = _target(data) or (tool if action == "run" else "")
        return EventNarration("tool", action, target, "start", level)
    if etype in ("tool.call.finished", "tool.call.failed"):
        ok = data.get("ok", etype == "tool.call.finished")
        if ok:
            return None  # success is implied by the next step; not narrated
        tool = str(data.get("tool") or "")
        action = _TOOL_ACTIONS.get(tool, "run")
        detail = str(data.get("error") or data.get("error_code") or "failed")
        return EventNarration("tool", action, tool, "error", level, detail)
    return None
