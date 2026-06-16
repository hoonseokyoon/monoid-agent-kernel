"""Provider-neutral content parts for multimodal input.

This is the **contract surface only** for multimodal work (see REFACTOR_PLAN
Round 2, W2). The types and JSON codecs land now so the input shape is stable;
only ``TextPart`` is actually forwarded to a model today. ``ImagePart`` /
``DocumentPart`` are accepted and round-trip, but the runner does not yet thread
them to providers and no provider advertises multimodal support — when non-text
parts are present the loop emits a ``model.input.degraded`` warning and proceeds
with text only.

Deferred (explicit follow-up): threading parts into ``ModelRequest``, provider
multimodal payload mapping, and ``fs.read`` extraction of non-text files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

# Capability that gates multimodal input. Not in any default set — off unless an
# integrator grants it explicitly via AgentRunSpec.capabilities.
MEDIA_INPUT_CAPABILITY = "media.input"


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


ContentPart = TextPart | ImagePart | DocumentPart


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
    raise ValueError(f"unsupported content part: {part!r}")


def content_part_from_json(payload: dict[str, Any]) -> ContentPart:
    kind = payload.get("type")
    if kind == "text":
        return TextPart(text=str(payload["text"]))
    if kind == "image":
        return ImagePart(source_ref=str(payload["source_ref"]), mime_type=str(payload["mime_type"]))
    if kind == "document":
        return DocumentPart(source_ref=str(payload["source_ref"]), mime_type=str(payload["mime_type"]))
    raise ValueError(f"unknown content part type: {kind!r}")
