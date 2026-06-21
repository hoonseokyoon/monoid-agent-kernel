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


@dataclass(frozen=True)
class TextPart:
    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class ImagePart:
    """Contract-only: a reference to image input. Not yet forwarded to providers."""

    source_ref: str  # workspace path or opaque handle; resolution is deferred
    mime_type: str
    type: Literal["image"] = "image"


@dataclass(frozen=True)
class DocumentPart:
    """Contract-only: a reference to document input (e.g. PDF). Not yet forwarded."""

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
    kind = payload.get("type")
    if kind == "text":
        return TextPart(text=str(payload["text"]))
    if kind == "image":
        return ImagePart(source_ref=str(payload["source_ref"]), mime_type=str(payload["mime_type"]))
    if kind == "document":
        return DocumentPart(source_ref=str(payload["source_ref"]), mime_type=str(payload["mime_type"]))
    if kind == "audio":
        return AudioPart(source_ref=str(payload["source_ref"]), mime_type=str(payload["mime_type"]))
    if kind == "video":
        return VideoPart(source_ref=str(payload["source_ref"]), mime_type=str(payload["mime_type"]))
    raise ValueError(f"unknown content part type: {kind!r}")
