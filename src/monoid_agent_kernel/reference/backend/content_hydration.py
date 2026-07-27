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

**Never fails a read.** Durability of the record is best-effort — no fsync, and the transcript's
only repair confines a torn line rather than recovering it — so a committed event's digest can
resolve to nothing. A crash is not the only cause: on a shared run root a second node can resume a
run a live peer still owns (neither ``recover_runs`` nor ``resume_run`` consults the lease store),
and two recorders then append to one transcript. Content-missing is a tolerated outcome either
way: the field stays absent and the reader sees what it would have seen anyway. Raising here would
turn a cosmetic gap into a dead endpoint.
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

# No positional bound on the scan. Two earlier attempts both lost text, in mirrored ways: a cap
# counting lines from the START dropped the newest settled text (the transcript grows by append),
# and anchoring the same budget at the END dropped the oldest — which broke Studio catch-up, since
# ``_read_committed_events`` hydrates every committed event of a run in one call and therefore
# wants digests spanning the whole session.
#
# The mistake both times was anchoring the bound to the *file* when the reader's need is anchored
# to the *event set it was asked about*. Any positional window is wrong for a caller that asks
# about events outside it, and hydration silently returning less than it holds is worse than being
# slow: every consumer normalises an absent field to "" and hides it, so the loss is invisible.
#
# What is bounded is the working set, not the file position: the transcript is read streaming a
# line at a time, and only the wanted digests are retained — so peak memory is the combined size
# of the answers actually being resolved, not the transcript.
#
# The cost is honestly one full pass in the common case, not just the worst one. The early exit
# fires only when every wanted digest has been found, so a page whose record was appended last
# still reads to EOF, and a digest whose record was lost (a tolerated outcome, see above) never
# terminates early at all. Callers on an event loop must therefore not call this inline —
# ``run_execution`` hands it to a thread. Paid only once the emit change makes events carry
# digests, and only for pages actually missing text.


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


def needs_settled_text(events: Any) -> bool:
    """Whether ``events`` would cause ``hydrate_settled_text`` to touch the filesystem.

    Exists so a caller on an event loop can keep the cheap check on the loop and offload only the
    scan. Until the emit change lands nothing carries a digest, so this is ``False`` for every
    frame and an unconditional thread hop would queue delivery behind a shared, bounded executor
    for no work at all.

    Takes the same shape as ``hydrate_settled_text`` — a list — deliberately. As a variadic it
    accepted ``needs_settled_text(page)`` and silently answered ``False``, because the list nested
    inside a list and every entry was skipped as malformed. A gate that fails closed by returning
    "no work" is the exact one-more-caller trap this module has been bitten by repeatedly.
    """
    return bool(_wanted_digests(events))


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
        # Read bytes and decode per line rather than opening in text mode: a crash can tear a
        # multi-byte sequence mid-write, and decoding is lazy, so strict text mode raises
        # ``UnicodeDecodeError`` from the iterator itself — a ``ValueError``, which slips past the
        # ``OSError`` handler below and turned every read needing a digest into a failed request
        # rather than the promised absent field. A replaced line then fails to parse as JSON, or
        # fails the digest check, and is skipped like any other malformed record.
        with transcript_path.open("rb") as handle:
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
    except (ValueError, RecursionError):
        # ``RecursionError`` as well as ``ValueError``: a deeply nested line makes the C scanner
        # exceed the interpreter's stack, and the read path runs on a deeper HTTP-handler stack
        # than the writer did. This module promises never to fail a read, and a corrupted or
        # foreign run dir is exactly the case that promise exists for.
        # A torn tail, or a line another writer is mid-way through. The transcript has no
        # repair that RECOVERS a torn line — the recorder's only one confines a tear to the record
        # it tore — so a malformed line is expected rather than exceptional.
        return None, None
    if not isinstance(record, dict) or record.get("kind") != SETTLED_TEXT_KIND:
        return None, None
    digest = record.get(DIGEST_FIELD)
    text = record.get(TEXT_FIELD)
    if isinstance(digest, str) and isinstance(text, str):
        return digest, text
    return None, None
