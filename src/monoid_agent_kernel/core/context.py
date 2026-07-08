"""Context providers — pluggable static and per-turn system context.

A ``ContextProvider`` lets an integrator inject extra system context without
touching the loop or the model adapter, in two layers:

* ``static_segment()`` is called once at bootstrap and folded into the composed
  system prompt alongside persona segments (see ``core/prompt.py``). Use it for
  context fixed for the whole run (environment notes, conventions).
* ``dynamic_segment(turn)`` is called every turn and appended to that turn's
  system prompt. Use it for context that changes as the run progresses (plan
  state, remaining budget) — analogous to per-turn "system reminders". Because
  ``turn`` carries ``bound_tools`` (the tool ids actually bound this turn), a
  provider can **gate its segment on the live config** — e.g. a Skills catalog
  appears only while the ``skill`` tool is bound, and disappears the turn after
  the capability is toggled off (a hot-swap). This is the per-run gating that
  ``static_segment`` (composed once at bootstrap) cannot express.

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
    # Registry tool ids bound for THIS turn (after runtime-config + tool-surface resolution).
    # Lets a dynamic_segment gate itself on what's actually available now — the seam that makes
    # a provider's context hot-swappable with the config. Empty by default (back-compat).
    bound_tools: frozenset[str] = frozenset()
    # Registry tool ids that are immediately callable without approval. None means the caller
    # did not provide authorization-aware data, preserving older provider tests and embedders.
    allowed_tools: frozenset[str] | None = None


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
