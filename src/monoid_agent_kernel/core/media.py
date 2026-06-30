"""Kernel-side resolution of multimodal ``source_ref`` references to bytes.

A multimodal part (``ImagePart``/``DocumentPart``) carries only a symbolic
``source_ref`` + ``mime_type`` — never bytes. The durable message log and the
checkpoint stay by-reference (small, resumable). Only the kernel can read the
workspace, so the reference is resolved to bytes **kernel-side, at wire-build
time**, by a ``MediaResolver``; the provider adapter then maps the resolved bytes
to its wire shape (base64 in v1).

v1 resolves workspace-path references via ``Workspace.read_bytes``. Opaque/remote
references (URLs, provider file-ids) are deferred and raise ``MediaResolveError``.
"""

from __future__ import annotations

import base64
import math
import re
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, TypeVar

from monoid_agent_kernel.core._util import sha256_bytes
from monoid_agent_kernel.core.workspace import Workspace
from monoid_agent_kernel.errors import NativeAgentError

# The capability marker a binding/run grants to enable multimodal media reads. Kept
# here (and referenced) so it is a live constant rather than a dead one.
MEDIA_INPUT_CAPABILITY = "media.input"

# Non-text part ``type`` values the wire-build resolves and forwards today. ``audio``/``video``
# join as their provider mapping lands.
WIRE_FORWARDABLE_PART_TYPES = frozenset({"image", "document"})

# Image token accounting (Anthropic patch model: tokens ≈ ⌈w/28⌉·⌈h/28⌉, clamped to a
# per-model native cap). The legacy (w·h)/750 approximation is deprecated.
_IMAGE_PATCH_PX = 28
NATIVE_IMAGE_TOKEN_CAP_DEFAULT = 1568
NATIVE_IMAGE_TOKEN_CAP_HIGH_RES = 4784
# Substrings of high-resolution model ids that get the larger native token budget.
_HIGH_RES_MODEL_MARKERS = ("opus-4-7", "opus-4-8", "opus4.7", "opus4.8", "fable-5", "mythos-5")

# More than this many image+document blocks in one request triggers the provider's
# stricter per-image dimension limit (Anthropic). The loop warns past it.
MAX_FORWARDABLE_BLOCKS = 20


def native_image_token_cap(model: str | None) -> int:
    """The per-image native token budget for ``model`` (high-res models get more)."""
    name = (model or "").lower()
    if any(marker in name for marker in _HIGH_RES_MODEL_MARKERS):
        return NATIVE_IMAGE_TOKEN_CAP_HIGH_RES
    return NATIVE_IMAGE_TOKEN_CAP_DEFAULT


def estimate_image_tokens(width: int, height: int, *, cap: int = NATIVE_IMAGE_TOKEN_CAP_DEFAULT) -> int:
    """Estimate an image's input token cost via the 28×28 patch formula, clamped to ``cap``."""
    if width <= 0 or height <= 0:
        return 0
    patches = math.ceil(width / _IMAGE_PATCH_PX) * math.ceil(height / _IMAGE_PATCH_PX)
    return min(patches, cap)


def image_dimensions(data: bytes, mime_type: str) -> tuple[int, int] | None:
    """Best-effort ``(width, height)`` from raw PNG/JPEG header bytes; ``None`` if unknown.

    Zero-dependency: parses the PNG IHDR chunk and JPEG SOF markers directly. Returns
    ``None`` for unsupported formats so callers treat token accounting as best-effort.
    """
    mime = (mime_type or "").lower()
    if mime == "image/png" and data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    if mime in ("image/jpeg", "image/jpg") and data[:2] == b"\xff\xd8":
        return _jpeg_dimensions(data)
    return None


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    # Walk JPEG markers to the start-of-frame (SOF0..SOF15, excluding DHP/DAC/RSTn).
    i = 2
    size = len(data)
    while i + 9 <= size:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[i + 5 : i + 9])
            return int(width), int(height)
        segment_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
        i += 2 + segment_len
    return None

# A ``scheme://`` prefix where the scheme is not ``workspace``: a remote/opaque ref.
_SCHEME_RE = re.compile(r"^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*)://")
_WORKSPACE_PREFIX_RE = re.compile(r"^workspace:(//)?")

# A durable, content-addressed media reference: ``blob:<sha256>``. The bytes live in the
# checkpoint blob store (write-once, content-addressed), so a blob ref survives a restart and a
# base re-provisioning — unlike a workspace path, whose bytes vanish if the base is re-cloned.
BLOB_SCHEME = "blob:"
# Inline ingress: an embedder may hand media by value as a ``data:`` URI (what a browser
# FileReader produces). The loop normalizes it to a ``blob:`` ref at ingestion, so the durable
# message log never carries the bytes inline.
_DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;,]*)(?P<b64>;base64)?,(?P<data>.*)$", re.DOTALL)

_PartT = TypeVar("_PartT")


def parse_data_uri(source_ref: str) -> tuple[str, bytes] | None:
    """Decode a ``data:[<mime>][;base64],<payload>`` URI into ``(mime_type, bytes)``; ``None`` if
    ``source_ref`` is not a data URI. Only base64 payloads are accepted (a non-base64 data URI is
    rejected loudly — the inline-media path is for binary attachments)."""
    match = _DATA_URI_RE.match(source_ref)
    if match is None:
        return None
    if not match.group("b64"):
        raise MediaResolveError("inline media must be a base64 data URI (data:<mime>;base64,...)")
    try:
        data = base64.b64decode(match.group("data"), validate=True)
    except (ValueError, TypeError) as exc:
        raise MediaResolveError(f"inline media is not valid base64: {exc}") from exc
    return (match.group("mime") or "application/octet-stream"), data


def _blobify_data_uri(source_ref: Any, store: dict[str, bytes]) -> str | None:
    """If ``source_ref`` is an inline ``data:`` URI, persist its bytes into ``store`` keyed by
    sha256 and return the durable ``blob:<sha256>`` reference; otherwise ``None``."""
    if not isinstance(source_ref, str) or not source_ref.startswith("data:"):
        return None
    parsed = parse_data_uri(source_ref)
    if parsed is None:  # pragma: no cover - startswith("data:") guarantees a match
        return None
    data = parsed[1]
    sha = sha256_bytes(data)
    store[sha] = data
    return f"{BLOB_SCHEME}{sha}"


def normalize_inline_media_part(part: _PartT, store: dict[str, bytes]) -> _PartT:
    """If ``part`` (a ``ContentPart`` dataclass) carries inline bytes (a ``data:`` ``source_ref``),
    persist the bytes into ``store`` and return a copy whose ``source_ref`` is the durable
    ``blob:<sha256>`` reference. Non-media / already-by-reference parts pass through unchanged.
    ``store`` is the loop's content-addressed media-blob map, so an inline image becomes durable
    the moment it is ingested."""
    blob_ref = _blobify_data_uri(getattr(part, "source_ref", None), store)
    if blob_ref is None:
        return part
    return replace(part, source_ref=blob_ref)  # type: ignore[type-var]


def normalize_inline_media_dicts(
    parts: list[dict[str, Any]], store: dict[str, bytes]
) -> list[dict[str, Any]]:
    """Dict form of :func:`normalize_inline_media_part`, for the by-value message media lists
    (``ToolObservation.media`` / a user message's ``content`` parts are already JSON dicts). Any
    part with an inline ``data:`` source is rewritten to a ``blob:<sha>`` ref with the bytes in
    ``store`` — so tool-returned inline media is symmetric with user-input inline media."""
    out: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, dict):
            blob_ref = _blobify_data_uri(part.get("source_ref"), store)
            if blob_ref is not None:
                out.append({**part, "source_ref": blob_ref})
                continue
        out.append(part)
    return out


def blob_shas_in_messages(messages: tuple[dict[str, Any], ...]) -> set[str]:
    """Every ``blob:<sha>`` referenced by a by-reference message log — user-content part lists
    and tool-message ``media`` lists. Used on restore to know which durable blobs to rehydrate
    into the loop's in-memory media-blob map."""
    shas: set[str] = set()
    for message in messages:
        for carrier in ("content", "media"):
            parts = message.get(carrier)
            if not isinstance(parts, list):
                continue
            for part in parts:
                ref = part.get("source_ref") if isinstance(part, dict) else None
                if isinstance(ref, str) and ref.startswith(BLOB_SCHEME):
                    shas.add(ref[len(BLOB_SCHEME):])
    return shas


class MediaResolveError(NativeAgentError):
    """Raised when a ``source_ref`` cannot be resolved to bytes."""

    error_code = "media_resolve_error"
    category = "workspace"


@dataclass(frozen=True)
class ResolvedMedia:
    """A ``source_ref`` resolved to concrete bytes, ready for provider mapping."""

    data: bytes
    sha256: str
    mime_type: str


class MediaResolver(Protocol):
    """The seam that turns a symbolic ``source_ref`` into bytes at wire-build time."""

    def resolve(self, source_ref: str, mime_type: str, *, max_bytes: int | None = None) -> ResolvedMedia:
        ...


@dataclass(frozen=True)
class WorkspaceMediaResolver:
    """Resolve media references to bytes at wire-build time.

    Three reference kinds are resolvable: a ``blob:<sha256>`` content-addressed reference (read from
    the in-memory ``blobs`` map first, then the optional ``blob_reader`` fallback — typically the
    checkpoint blob store, so a blob a *peer* persisted, e.g. the backend normalizing a queued inline
    message, still resolves); and a workspace path (bare, or with a ``workspace:`` / ``workspace://``
    prefix), read via ``Workspace.read_bytes``. Any other URI scheme (``http://``, ``file_id:`` …) is
    a deferred remote/opaque handle and raises ``MediaResolveError``.
    """

    workspace: Workspace
    blobs: Mapping[str, bytes] = field(default_factory=dict)
    # Fallback for a blob: ref absent from the in-memory map — e.g. ``store.get_blob(run_id, sha)``.
    blob_reader: Callable[[str], bytes] | None = None

    def resolve(self, source_ref: str, mime_type: str, *, max_bytes: int | None = None) -> ResolvedMedia:
        if source_ref.startswith(BLOB_SCHEME):
            sha = source_ref[len(BLOB_SCHEME):]
            data = self.blobs.get(sha)
            if data is None and self.blob_reader is not None:
                try:
                    data = self.blob_reader(sha)
                except KeyError:
                    data = None
            if data is None:
                raise MediaResolveError(f"cannot resolve media blob {source_ref!r}: not in the blob store")
            if max_bytes is not None and len(data) > max_bytes:
                raise MediaResolveError(f"media blob exceeds max read size: {source_ref!r}")
            return ResolvedMedia(data=data, sha256=sha, mime_type=mime_type)
        path = self._workspace_path(source_ref)
        try:
            data, digest = self.workspace.read_bytes(path, max_bytes=max_bytes)
        except NativeAgentError as exc:
            # Surface a missing/oversized workspace read loudly, with an actionable remedy — the
            # cap is a run-level knob a caller would not associate with media (gap 2b).
            raise MediaResolveError(
                f"cannot resolve media source_ref {source_ref!r}: {exc}. If this is a size limit, "
                f"raise the run's max_bytes_read, or downsample/re-ingest the media below the cap."
            ) from exc
        return ResolvedMedia(data=data, sha256=digest, mime_type=mime_type)

    @staticmethod
    def _workspace_path(source_ref: str) -> str:
        stripped = _WORKSPACE_PREFIX_RE.sub("", source_ref, count=1)
        scheme = _SCHEME_RE.match(stripped)
        if scheme is not None:
            raise MediaResolveError(
                f"unsupported media source_ref scheme {scheme.group('scheme')!r}: "
                f"only workspace paths are resolvable (got {source_ref!r})"
            )
        return stripped


def media_block_base64(
    part_type: str, resolved: ResolvedMedia, *, filename: str | None = None
) -> dict[str, Any]:
    """Neutral resolved media block (the Anthropic image/document shape).

    Adapters map this further to their provider envelope (e.g. OpenAI ``input_image`` /
    ``input_file``). ``filename`` is carried on document blocks because OpenAI ``input_file``
    requires it; the Anthropic shape ignores the extra key.
    """
    block: dict[str, Any] = {
        "type": part_type,
        "source": {
            "type": "base64",
            "media_type": resolved.mime_type,
            "data": base64.b64encode(resolved.data).decode("ascii"),
        },
    }
    if filename:
        block["filename"] = filename
    return block


def _document_filename(source_ref: str) -> str:
    """Basename of a document ``source_ref`` (workspace prefix stripped) for OpenAI input_file."""
    path = _WORKSPACE_PREFIX_RE.sub("", source_ref, count=1)
    return path.rsplit("/", 1)[-1] or "document.pdf"


def count_tool_result_images(messages: tuple[dict[str, Any], ...]) -> int:
    """Count forwardable image parts on ``tool``-role messages' ``media`` lists. Eviction is
    image-specific (diminishing-value screenshots), so only image-type parts are counted even
    though the ``media`` carrier may also hold documents."""
    return sum(
        1
        for message in messages
        if message.get("role") == "tool" and isinstance(message.get("media"), list)
        for part in message["media"]
        if isinstance(part, dict) and part.get("type") in WIRE_FORWARDABLE_PART_TYPES
    )


def evict_tool_result_images(
    messages: tuple[dict[str, Any], ...],
    keep_n: int | None,
    *,
    chunk: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Keep only the most-recent ``keep_n`` tool-result images, dropping older ones.

    Operates on the by-reference message list (cheap — runs before resolution). Targets only
    image parts on ``role == "tool"`` messages' ``media`` lists; user-content images and any
    non-image media (e.g. documents) are never touched (mirrors the Anthropic computer-use
    nesting rule, and keeps PDFs from being aged out like stale screenshots). Removal is
    **cache-aligned**: the count is rounded down to a multiple of ``chunk`` (default
    ``keep_n``) so the wire prefix stays byte-stable between turns until a whole chunk ages
    out. ``keep_n=None`` is a no-op.
    """
    if keep_n is None:
        return messages
    chunk = chunk or keep_n or 1
    total = count_tool_result_images(messages)
    to_remove = total - keep_n
    if to_remove <= 0:
        return messages
    to_remove -= to_remove % chunk
    if to_remove <= 0:
        return messages
    removed = 0
    result: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "tool" or not isinstance(message.get("media"), list):
            result.append(message)
            continue
        kept: list[Any] = []
        for part in message["media"]:
            is_image = isinstance(part, dict) and part.get("type") in WIRE_FORWARDABLE_PART_TYPES
            if is_image and removed < to_remove:
                removed += 1
                continue
            kept.append(part)
        result.append({**message, "media": kept})
    return tuple(result)


def resolve_wire_messages(
    messages: tuple[dict[str, Any], ...],
    resolver: MediaResolver,
    *,
    encoding: str = "base64",
) -> tuple[dict[str, Any], ...]:
    """Build the ephemeral wire copy of a by-reference message log.

    Text content passes through. A by-reference media part whose ``type`` is
    forwardable is resolved to bytes and replaced with a neutral resolved block;
    non-forwardable media parts are dropped (the loop already emitted a degraded
    warning for them). The durable ``messages`` are never mutated.
    """
    if encoding != "base64":
        raise MediaResolveError(f"unsupported wire image encoding: {encoding!r}")
    resolved_messages: list[dict[str, Any]] = []
    for message in messages:
        new_message = message
        # User multimodal turns carry parts in a ``content`` list.
        content = message.get("content")
        if isinstance(content, list):
            new_message = {**new_message, "content": _resolve_part_list(content, resolver)}
        # Tool messages carry returned media in a top-level ``media`` list.
        media = message.get("media")
        if isinstance(media, list):
            new_message = {**new_message, "media": _resolve_part_list(media, resolver)}
        resolved_messages.append(new_message)
    return tuple(resolved_messages)


def _resolve_part_list(parts: list[Any], resolver: MediaResolver) -> list[dict[str, Any]]:
    """Resolve forwardable by-reference media parts to neutral base64 blocks; text passes
    through; non-forwardable media is dropped (already degraded-warned)."""
    resolved: list[dict[str, Any]] = []
    for part in parts:
        part_type = part.get("type") if isinstance(part, dict) else None
        if part_type == "text":
            resolved.append(part)
        elif part_type in WIRE_FORWARDABLE_PART_TYPES:
            media = resolver.resolve(str(part["source_ref"]), str(part["mime_type"]))
            filename = _document_filename(str(part["source_ref"])) if part_type == "document" else None
            resolved.append(media_block_base64(str(part_type), media, filename=filename))
    return resolved
