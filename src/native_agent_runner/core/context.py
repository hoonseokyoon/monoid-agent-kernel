"""Context providers — pluggable static and per-turn system context.

A ``ContextProvider`` lets an integrator inject extra system context without
touching the loop or the model adapter, in two layers:

* ``static_segment()`` is called once at bootstrap and folded into the composed
  system prompt alongside persona segments (see ``core/prompt.py``). Use it for
  context fixed for the whole run (environment notes, conventions).
* ``dynamic_segment(turn)`` is called every turn and appended to that turn's
  system prompt. Use it for context that changes as the run progresses (plan
  state, remaining budget) — analogous to per-turn "system reminders".

Both return ``None`` (or blank) to contribute nothing. With no providers the
per-turn prompt is byte-identical to the static composed prompt, so the default
behavior is unchanged.

Note: the contract deliberately carries only plain data (``TurnContext``), not
the ``Workspace`` object, keeping the workspace out of the integration surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class TurnContext:
    """Read-only snapshot of run progress handed to ``dynamic_segment`` each turn."""

    step: int
    remaining_steps: int
    remaining_tool_calls: int
    deadline_s: float | None  # seconds left until the run deadline, or None if unbounded
    plan: tuple[dict[str, Any], ...]
    pending_observation_count: int


class ContextProvider(Protocol):
    def static_segment(self) -> str | None: ...

    def dynamic_segment(self, turn: TurnContext) -> str | None: ...


def render_workspace_index_segment(index: dict[str, Any], *, max_files: int = 50) -> str | None:
    """Render a compact workspace-file listing from a workspace-index dict
    (``core/workspace_index.build_workspace_index``) for optional prompt injection.
    Returns None when there are no files to show."""
    entries = index.get("entries") or []
    files = [e for e in entries if e.get("kind") == "file"]
    if not files:
        return None
    shown = files[:max_files]
    lines = [f"- {entry['path']} ({entry.get('size', 0)} bytes)" for entry in shown]
    remaining = len(files) - len(shown)
    if remaining > 0:
        lines.append(f"- … and {remaining} more file(s)")
    return "Workspace files (initial snapshot):\n" + "\n".join(lines)
