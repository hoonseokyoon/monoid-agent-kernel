"""Format an event into a friendly activity-feed line for the Studio UI.

The "which events matter + what is the verb/target" logic lives once in
``native_agent_runner.narration`` (shared with the ``watch`` CLI). This module is just the
Studio-flavored *formatter* over that neutral narration — present-tense, user-facing prose.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from native_agent_runner.narration import narrate_event

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
    verb = _PRESENT.get(narration.action, narration.action.capitalize())
    return f"{verb} {narration.target}".strip() if narration.target else verb
