"""Resolve settled text back onto event payloads from the run-dir record.

v0.20 stops publishing model-authored ``final_text`` on the fan-out event stream — ``events.jsonl``
is documented as public/redacted (``docs/OBSERVABILITY.md``) and was carrying raw model output.
The text moves to a ``settled_text`` record in ``transcript.jsonl``, the private debug/replay
artifact, and the event carries ``final_text_digest`` instead. Readers that are entitled to the
text join the two back together here.

**Applied at the reader seam, not at call sites.** Every consumer that reaches events through the
backend projection — the backend's own REST and SSE twins, the Studio BFF, ``monoid studio
accept`` — inherits hydration from one place. Patching individual call sites was the shape this
was first planned as, and it missed two transports outright.

**Fills absent fields only.** An event that still carries ``final_text`` is left exactly as it is,
which is what makes this a no-op until the emit change lands, and what keeps kernel-authored text
(``"Stopped after reaching max steps."``, which never leaves the event) untouched.

**Never fails a read.** Durability of the record is best-effort — no fsync, no append-tail repair —
so a crash can leave a committed event whose digest resolves to nothing. Content-missing is a
tolerated outcome: the field stays absent and the reader sees what it would have seen anyway.
Raising here would turn a cosmetic gap into a dead endpoint.
"""

from __future__ import annotations

import json
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core.model_io import content_digest

TRANSCRIPT_FILE_NAME = "transcript.jsonl"
SETTLED_TEXT_KIND = "settled_text"
DIGEST_FIELD = "final_text_digest"
TEXT_FIELD = "final_text"

# The record is keyed by content digest, while every reader pages by ``seq``, so resolving a page
# means a lookup rather than an offset. Bound it: this repo bounds its other readers (watch
# batches, event-index slots) and an unbounded scan of an attacker-influenced file would regress
# that.
#
# The bound is applied from the END of the file, which is the correction to a cap that used to
# count lines from the start. ``settled_text`` records are *appended*, and the transcript
# accumulates across every step and every restore of a run, so a front-anchored cap truncated
# exactly the region holding the newest settled text: old text resolved and the current run's did
# not. Scanning the tail also means the early-exit on "every wanted digest found" is a latency
# optimisation rather than the thing keeping the cap honest.
MAX_SCAN_BYTES = 8_000_000


def hydrate_settled_text(events: Any, run_dir: Path) -> Any:
    """Fill absent ``final_text`` on ``events`` in place, from ``run_dir``'s settled-text records.

    Returns ``events`` so callers can wrap a read expression directly.
    """
    wanted = _wanted_digests(events)
    if not wanted:
        # The common case, and today the *only* case: nothing asked for text, so the transcript is
        # never opened. Hydration costs one pass over an in-memory page until the emit change.
        return events
    resolved = _resolve(run_dir / TRANSCRIPT_FILE_NAME, wanted)
    if not resolved:
        return events
    for event in _event_data(events):
        if TEXT_FIELD in event:
            continue
        text = resolved.get(event.get(DIGEST_FIELD))
        if text is not None:
            event[TEXT_FIELD] = text
    return events


def _event_data(events: Any) -> list[MutableMapping[str, Any]]:
    """The mutable ``data`` payloads of ``events``, skipping anything malformed.

    Defensive about shape on purpose: this runs on the read path of several transports, and a
    surprising payload should cost a skipped hydration, not a failed request.
    """
    if not isinstance(events, list):
        return []
    payloads: list[MutableMapping[str, Any]] = []
    for event in events:
        if not isinstance(event, MutableMapping):
            continue
        data = event.get("data")
        if isinstance(data, MutableMapping):
            payloads.append(data)
    return payloads


def _wanted_digests(events: Any) -> set[str]:
    wanted: set[str] = set()
    for data in _event_data(events):
        if TEXT_FIELD in data:
            continue
        digest = data.get(DIGEST_FIELD)
        if isinstance(digest, str) and digest:
            wanted.add(digest)
    return wanted


def _resolve(transcript_path: Path, wanted: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    try:
        # Read bytes and decode per line rather than opening in text mode. Two reasons: seeking to
        # a byte offset is only well-defined on a binary handle, and a crash can tear a multi-byte
        # sequence mid-write — decoding is lazy, so strict text mode raises ``UnicodeDecodeError``
        # from the iterator itself, which is a ``ValueError`` and slips past the ``OSError``
        # handler below. That turned every read needing a digest into a failed request rather than
        # the promised absent field. A replaced line then fails to parse as JSON, or fails the
        # digest check, and is skipped like any other malformed record.
        size = transcript_path.stat().st_size
        with transcript_path.open("rb") as handle:
            if size > MAX_SCAN_BYTES:
                handle.seek(size - MAX_SCAN_BYTES)
                handle.readline()  # discard the partial line the seek landed inside
            for raw_line in handle:
                digest, text = _settled_text_entry(raw_line.decode("utf-8", errors="replace"))
                if digest is None or text is None or digest not in wanted:
                    continue
                if content_digest(text) != digest:
                    # The join's whole premise is that the digest names the content, so verify
                    # rather than trust. A torn or character-replaced line can still decode to a
                    # well-formed record whose text is no longer what its digest names, and
                    # handing that back would be worse than handing back nothing.
                    continue
                found[digest] = text
                if len(found) == len(wanted):
                    break
    except OSError:
        # Missing transcript (an older run dir, a run dir copied without it) or an unreadable one.
        return found
    return found


def _settled_text_entry(line: str) -> tuple[str | None, str | None]:
    line = line.strip()
    if not line:
        return None, None
    try:
        record = json.loads(line)
    except ValueError:
        # A torn tail, or a line another writer is mid-way through. The transcript has no
        # append-tail repair, so a malformed line is expected rather than exceptional.
        return None, None
    if not isinstance(record, dict) or record.get("kind") != SETTLED_TEXT_KIND:
        return None, None
    digest = record.get(DIGEST_FIELD)
    text = record.get(TEXT_FIELD)
    if isinstance(digest, str) and isinstance(text, str):
        return digest, text
    return None, None
