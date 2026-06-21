"""Runner-side resolution of multimodal ``source_ref`` references to bytes.

A multimodal part (``ImagePart``/``DocumentPart``) carries only a symbolic
``source_ref`` + ``mime_type`` — never bytes. The durable message log and the
checkpoint stay by-reference (small, resumable). Only the runner can read the
workspace, so the reference is resolved to bytes **runner-side, at wire-build
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
from dataclasses import dataclass
from typing import Any, Protocol

from native_agent_runner.core.workspace import Workspace
from native_agent_runner.errors import NativeAgentError

# The capability marker a binding/run grants to enable multimodal media reads. Kept
# here (and referenced) so it is a live constant rather than a dead one.
MEDIA_INPUT_CAPABILITY = "media.input"

# Non-text part ``type`` values the wire-build resolves and forwards today. Image only
# for now; ``document``/``audio``/``video`` join as their provider mapping lands.
WIRE_FORWARDABLE_PART_TYPES = frozenset({"image"})

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
    """Resolve workspace-path references via ``Workspace.read_bytes``.

    Accepts bare workspace paths and an optional ``workspace:`` / ``workspace://``
    prefix. Any other URI scheme (``http://``, ``https://``, ``file_id:`` …) is a
    deferred remote/opaque handle and raises ``MediaResolveError``.
    """

    workspace: Workspace

    def resolve(self, source_ref: str, mime_type: str, *, max_bytes: int | None = None) -> ResolvedMedia:
        path = self._workspace_path(source_ref)
        try:
            data, digest = self.workspace.read_bytes(path, max_bytes=max_bytes)
        except NativeAgentError as exc:
            # Surface a missing/oversized blob loudly under the resolver contract,
            # preserving the underlying cause (e.g. on resume with a deleted file).
            raise MediaResolveError(f"cannot resolve media source_ref {source_ref!r}: {exc}") from exc
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


def media_block_base64(part_type: str, resolved: ResolvedMedia) -> dict[str, Any]:
    """Neutral resolved media block (the Anthropic image/document shape).

    Adapters map this further to their provider envelope (e.g. OpenAI ``input_image``).
    """
    return {
        "type": part_type,
        "source": {
            "type": "base64",
            "media_type": resolved.mime_type,
            "data": base64.b64encode(resolved.data).decode("ascii"),
        },
    }


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
        content = message.get("content")
        if not isinstance(content, list):
            resolved_messages.append(message)
            continue
        new_content: list[dict[str, Any]] = []
        for part in content:
            part_type = part.get("type") if isinstance(part, dict) else None
            if part_type == "text":
                new_content.append(part)
            elif part_type in WIRE_FORWARDABLE_PART_TYPES:
                resolved = resolver.resolve(str(part["source_ref"]), str(part["mime_type"]))
                new_content.append(media_block_base64(str(part_type), resolved))
            # else: non-forwardable media — dropped from the wire (already degraded-warned).
        resolved_messages.append({**message, "content": new_content})
    return tuple(resolved_messages)
