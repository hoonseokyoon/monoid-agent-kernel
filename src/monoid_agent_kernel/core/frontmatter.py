"""Minimal, dependency-free frontmatter parser.

Splits a document into a leading ``---`` metadata block and the body. The metadata
is a small YAML *subset* — enough for subagent (``.claude/agents``-style) and Skill
(``SKILL.md``) files without taking a PyYAML dependency (the project is zero-dep by
design). Supported:

- ``key: value`` scalars (strings, ints, ``true``/``false``, ``null``/``~``)
- single/double quoted strings (quotes stripped)
- inline lists: ``key: [a, b, c]``
- block lists::

      key:
        - a
        - b

- full-line ``#`` comments and blank lines (ignored)

NOT supported (deliberately): nested maps, multi-line scalars, anchors. Frontmatter
for these files is flat, so this covers it; anything fancier should be JSON.
"""

from __future__ import annotations

from typing import Any

__all__ = ["parse_frontmatter", "parse_scalar"]


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return ``(metadata, body)``. If the text has no leading ``---`` block, the
    metadata is empty and the whole text is the body."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---\n") and normalized.strip() != "---":
        return {}, text
    lines = normalized.split("\n")
    # lines[0] is the opening '---'; find the closing '---'.
    close_idx: int | None = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            close_idx = idx
            break
    if close_idx is None:
        # No closing fence — treat the whole document as body (no frontmatter).
        return {}, text
    meta = _parse_block(lines[1:close_idx])
    body = "\n".join(lines[close_idx + 1 :])
    # Drop a single leading blank line after the fence for a clean body.
    if body.startswith("\n"):
        body = body[1:]
    return meta, body


def _parse_block(lines: list[str]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if ":" not in raw:
            i += 1
            continue
        key, _, rest = raw.partition(":")
        key = key.strip()
        rest = rest.strip()
        if rest:
            meta[key] = parse_scalar(rest)
            i += 1
            continue
        # Empty value -> maybe a block list on following indented '- ' lines.
        items: list[Any] = []
        j = i + 1
        while j < n:
            item_line = lines[j]
            item_stripped = item_line.strip()
            if not item_stripped or item_stripped.startswith("#"):
                j += 1
                continue
            if item_stripped.startswith("- "):
                items.append(parse_scalar(item_stripped[2:].strip()))
                j += 1
                continue
            break
        meta[key] = items if items else None
        i = j if items else i + 1
    return meta


def parse_scalar(token: str) -> Any:
    """Parse one scalar token: quoted string, inline list, bool, null, int, or str."""
    token = token.strip()
    if not token:
        return ""
    if (token[0] == token[-1]) and token[0] in {'"', "'"} and len(token) >= 2:
        return token[1:-1]
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part) for part in _split_inline_list(inner)]
    lowered = token.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "~"}:
        return None
    if _is_int(token):
        return int(token)
    return token


def _split_inline_list(inner: str) -> list[str]:
    """Split ``a, "b, c", d`` on commas not inside quotes."""
    parts: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    for ch in inner:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in {'"', "'"}:
            quote = ch
            buf.append(ch)
        elif ch == ",":
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _is_int(token: str) -> bool:
    candidate = token[1:] if token[:1] in {"-", "+"} else token
    return candidate.isdigit()
