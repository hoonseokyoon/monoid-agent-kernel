"""Format an event into a friendly activity-feed line for the Studio UI.

The "which events matter + what is the verb/target" logic lives once in
``monoid_agent_kernel.narration`` (shared with the ``watch`` CLI). This module is just the
Studio-flavored *formatter* over that neutral narration — present-tense, user-facing prose.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from monoid_agent_kernel.narration import narrate_event

# Neutral action token -> Studio's present-tense phrasing.
_PRESENT = {
    "read": "Reading",
    "write": "Writing",
    "edit": "Editing",
    "delete": "Deleting",
    "move": "Moving",
    "copy": "Copying",
    "mkdir": "Creating directory",
    "list": "Listing",
    "run": "Running",
    "ask": "Asking the human",
    "search": "Searching the web for",
    "fetch": "Fetching",
    "research": "Researching",
}


def describe_event(event: Mapping[str, Any]) -> str | None:
    """A friendly activity line for the feed, or ``None`` if the event isn't shown there."""
    narration = narrate_event(event)
    if narration is None:
        return None
    if narration.status == "error":
        if narration.detail:
            return f"⚠ {narration.target} failed: {narration.detail}"
        return f"⚠ {narration.target} failed"
    # Provider tools carry no path/query target, so the generic narration is bare ("Running
    # skill"). Surface the skill name / mark MCP tools for a clearer feed (the R5 narration
    # lesson: a tool family the narrator doesn't know gets a thin studio-side branch).
    data = event.get("data") or {}
    tool = str(data.get("tool") or "")
    args = data.get("args_preview") or {}
    if tool == "skill" and args.get("name"):
        return f"Using skill: {args['name']}"
    if tool.startswith("mcp_"):
        return f"Calling MCP tool: {tool}"
    verb = _PRESENT.get(narration.action, narration.action.capitalize())
    return f"{verb} {narration.target}".strip() if narration.target else verb
