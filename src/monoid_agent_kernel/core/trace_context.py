"""W3C Trace Context helpers (``traceparent`` / ``tracestate``).

Pure functions, zero dependencies — used by the inbox/outbox envelopes to carry a distributed-trace
id across a checkpoint/restart and the edge's outbound send. This is *observability* metadata: it
**complements** the envelope's ``correlation_id``/``causation_id`` (the domain identity that routing
and reply-matching depend on) and application behavior must never depend on it — a missing or
malformed ``traceparent`` is simply ignored.

``traceparent`` format (W3C Trace Context, version ``00``)::

    00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
    │  └ trace-id (16 bytes / 32 hex)    └ span-id (8 / 16) └ flags (1 / 2)
    └ version (1 byte / 2 hex)

``tracestate`` is an opaque vendor list propagated verbatim; we never parse it, only carry it.
"""

from __future__ import annotations

import secrets

TRACE_VERSION = "00"
_FLAG_SAMPLED = "01"
_FLAG_UNSAMPLED = "00"


def _is_hex(s: str, length: int) -> bool:
    if len(s) != length:
        return False
    try:
        int(s, 16)
    except ValueError:
        return False
    return True


def parse_traceparent(value: str | None) -> dict[str, str] | None:
    """Parse + validate a ``traceparent``. Returns ``{version, trace_id, span_id, flags}`` or
    ``None`` if the string is absent or malformed (wrong shape, non-hex, or an all-zero trace/span
    id, both invalid per spec). Tolerant by design — a bad header never raises."""
    if not value:
        return None
    parts = value.split("-")
    if len(parts) != 4:
        return None
    version, trace_id, span_id, flags = parts
    if not (_is_hex(version, 2) and _is_hex(trace_id, 32) and _is_hex(span_id, 16) and _is_hex(flags, 2)):
        return None
    if int(trace_id, 16) == 0 or int(span_id, 16) == 0:
        return None
    return {"version": version, "trace_id": trace_id, "span_id": span_id, "flags": flags}


def new_traceparent(*, sampled: bool = True) -> str:
    """Mint a fresh root ``traceparent`` — a new 128-bit trace-id and 64-bit span-id."""
    trace_id = secrets.token_hex(16)
    span_id = secrets.token_hex(8)
    flags = _FLAG_SAMPLED if sampled else _FLAG_UNSAMPLED
    return f"{TRACE_VERSION}-{trace_id}-{span_id}-{flags}"


def child_traceparent(parent: str | None) -> str:
    """Derive a child span of ``parent``: same trace-id, a new span-id (the caller becomes the
    parent). If ``parent`` is missing or malformed, mint a fresh root instead so the result is always
    a valid ``traceparent``."""
    parsed = parse_traceparent(parent)
    if parsed is None:
        return new_traceparent()
    span_id = secrets.token_hex(8)
    return f"{parsed['version']}-{parsed['trace_id']}-{span_id}-{parsed['flags']}"


def trace_id_of(value: str | None) -> str:
    """The trace-id of a ``traceparent`` (the stable end-to-end id), or ``""`` if unparseable."""
    parsed = parse_traceparent(value)
    return parsed["trace_id"] if parsed is not None else ""
