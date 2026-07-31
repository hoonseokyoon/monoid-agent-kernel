"""Provider-neutral content parts for multimodal input.

This module defines the stable input shape for multimodal work. ``TextPart``,
``ImagePart`` and ``DocumentPart`` are forwarded to models that support it (the
loop resolves them to provider blocks). ``AudioPart`` and ``VideoPart`` round-trip
through JSON and survive checkpoints, but are not yet forwarded — the loop emits a
``model.input.degraded`` warning for any part type it cannot forward and proceeds
with the rest. Audio/video forwarding is provider-thin (Gemini-native) and left to a
later adapter; the contract here keeps them first-class so an integrator can carry them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from monoid_agent_kernel.core.json_ingress import normalize_unicode_scalars


@dataclass(frozen=True)
class TextPart:
    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class ImagePart:
    """A by-reference image input. Forwarded to multimodal providers: the loop resolves
    ``source_ref`` to bytes at wire-build time (``WIRE_FORWARDABLE_PART_TYPES``)."""

    source_ref: str  # workspace path or opaque handle; resolution is deferred
    mime_type: str
    type: Literal["image"] = "image"


@dataclass(frozen=True)
class DocumentPart:
    """A by-reference document input (e.g. PDF). Forwarded to multimodal providers, same as
    ``ImagePart`` (resolved at wire-build time)."""

    source_ref: str
    mime_type: str
    type: Literal["document"] = "document"


@dataclass(frozen=True)
class AudioPart:
    """Contract-only: a reference to audio input. Round-trips but not yet forwarded."""

    source_ref: str
    mime_type: str
    type: Literal["audio"] = "audio"


@dataclass(frozen=True)
class VideoPart:
    """Contract-only: a reference to video input. Round-trips but not yet forwarded."""

    source_ref: str
    mime_type: str
    type: Literal["video"] = "video"


ContentPart = TextPart | ImagePart | DocumentPart | AudioPart | VideoPart


def normalize_content_part(part: ContentPart) -> ContentPart:
    """Copy one externally supplied part into the portable Unicode domain."""

    if isinstance(part, TextPart):
        return TextPart(text=normalize_unicode_scalars(part.text))
    if isinstance(part, ImagePart):
        return ImagePart(
            source_ref=normalize_unicode_scalars(part.source_ref),
            mime_type=normalize_unicode_scalars(part.mime_type),
        )
    if isinstance(part, DocumentPart):
        return DocumentPart(
            source_ref=normalize_unicode_scalars(part.source_ref),
            mime_type=normalize_unicode_scalars(part.mime_type),
        )
    if isinstance(part, AudioPart):
        return AudioPart(
            source_ref=normalize_unicode_scalars(part.source_ref),
            mime_type=normalize_unicode_scalars(part.mime_type),
        )
    if isinstance(part, VideoPart):
        return VideoPart(
            source_ref=normalize_unicode_scalars(part.source_ref),
            mime_type=normalize_unicode_scalars(part.mime_type),
        )
    raise ValueError(f"unsupported content part: {part!r}")


def non_text_part_types(parts: tuple[ContentPart, ...]) -> list[str]:
    """Distinct ``type`` values of the non-text parts, in first-seen order.
    Empty when every part is text (the only kind forwarded today)."""
    seen: list[str] = []
    for part in parts:
        if not isinstance(part, TextPart) and part.type not in seen:
            seen.append(part.type)
    return seen


def content_part_to_json(part: ContentPart) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        return {"type": "image", "source_ref": part.source_ref, "mime_type": part.mime_type}
    if isinstance(part, DocumentPart):
        return {"type": "document", "source_ref": part.source_ref, "mime_type": part.mime_type}
    if isinstance(part, AudioPart):
        return {"type": "audio", "source_ref": part.source_ref, "mime_type": part.mime_type}
    if isinstance(part, VideoPart):
        return {"type": "video", "source_ref": part.source_ref, "mime_type": part.mime_type}
    raise ValueError(f"unsupported content part: {part!r}")


def content_part_from_json(payload: dict[str, Any]) -> ContentPart:
    if not isinstance(payload, dict):
        raise ValueError("content part must be an object")

    def required_text(field_name: str) -> str:
        value = payload.get(field_name)
        if type(value) is not str:
            raise ValueError(f"content part {field_name} must be a string")
        return normalize_unicode_scalars(value)

    kind = payload.get("type")
    if kind == "text":
        return TextPart(text=required_text("text"))
    if kind == "image":
        return ImagePart(
            source_ref=required_text("source_ref"), mime_type=required_text("mime_type")
        )
    if kind == "document":
        return DocumentPart(
            source_ref=required_text("source_ref"), mime_type=required_text("mime_type")
        )
    if kind == "audio":
        return AudioPart(
            source_ref=required_text("source_ref"), mime_type=required_text("mime_type")
        )
    if kind == "video":
        return VideoPart(
            source_ref=required_text("source_ref"), mime_type=required_text("mime_type")
        )
    raise ValueError(f"unknown content part type: {kind!r}")
