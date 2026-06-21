from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import struct

from native_agent_runner.core.media import (
    NATIVE_IMAGE_TOKEN_CAP_DEFAULT,
    NATIVE_IMAGE_TOKEN_CAP_HIGH_RES,
    MediaResolveError,
    ResolvedMedia,
    WorkspaceMediaResolver,
    count_tool_result_images,
    estimate_image_tokens,
    evict_tool_result_images,
    image_dimensions,
    native_image_token_cap,
)
from native_agent_runner.workspace.local import LocalWorkspaceBackend

# A tiny PNG header + NUL bytes — binary content fs.read would reject as non-text.
_PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"


def _workspace_with_image(tmp_path: Path) -> LocalWorkspaceBackend:
    root = tmp_path / "workspace"
    root.mkdir()
    root.joinpath("img.png").write_bytes(_PNG_BYTES)
    return LocalWorkspaceBackend(root)


def test_workspace_media_resolver_reads_bytes(tmp_path: Path) -> None:
    resolver = WorkspaceMediaResolver(workspace=_workspace_with_image(tmp_path))

    resolved = resolver.resolve("img.png", "image/png")

    assert isinstance(resolved, ResolvedMedia)
    assert resolved.data == _PNG_BYTES
    assert resolved.mime_type == "image/png"
    assert resolved.sha256 == hashlib.sha256(_PNG_BYTES).hexdigest()


def test_workspace_media_resolver_strips_workspace_prefix(tmp_path: Path) -> None:
    resolver = WorkspaceMediaResolver(workspace=_workspace_with_image(tmp_path))

    bare = resolver.resolve("img.png", "image/png")
    prefixed = resolver.resolve("workspace://img.png", "image/png")
    short_prefixed = resolver.resolve("workspace:img.png", "image/png")

    assert bare.data == prefixed.data == short_prefixed.data == _PNG_BYTES


@pytest.mark.parametrize("ref", ["https://example.com/x.png", "http://h/x", "gs://bucket/x", "s3://b/k"])
def test_resolver_rejects_opaque_handle(tmp_path: Path, ref: str) -> None:
    resolver = WorkspaceMediaResolver(workspace=_workspace_with_image(tmp_path))

    with pytest.raises(MediaResolveError, match="scheme"):
        resolver.resolve(ref, "image/png")


def test_resolver_missing_file_raises_media_error(tmp_path: Path) -> None:
    resolver = WorkspaceMediaResolver(workspace=_workspace_with_image(tmp_path))

    with pytest.raises(MediaResolveError, match="cannot resolve media source_ref"):
        resolver.resolve("nope.png", "image/png")


def test_image_token_estimate_patch_formula() -> None:
    assert estimate_image_tokens(28, 28) == 1
    assert estimate_image_tokens(29, 28) == 2
    assert estimate_image_tokens(56, 56) == 4
    # Clamped to the native cap for very large images.
    assert estimate_image_tokens(100_000, 100_000) == NATIVE_IMAGE_TOKEN_CAP_DEFAULT
    assert estimate_image_tokens(0, 100) == 0


def test_native_image_token_cap_is_higher_for_opus() -> None:
    assert native_image_token_cap("claude-opus-4-8") == NATIVE_IMAGE_TOKEN_CAP_HIGH_RES
    assert native_image_token_cap("gpt-5.5") == NATIVE_IMAGE_TOKEN_CAP_DEFAULT
    assert native_image_token_cap(None) == NATIVE_IMAGE_TOKEN_CAP_DEFAULT


def _tool_image_msg(ref: str) -> dict:
    return {
        "role": "tool",
        "call_id": ref,
        "content": {"ok": True, "result": {}},
        "media": [{"type": "image", "source_ref": ref, "mime_type": "image/png"}],
    }


def _refs(messages) -> list[str]:
    return [p["source_ref"] for m in messages for p in m.get("media", [])]


def test_evict_keeps_chunk_aligned_recent() -> None:
    messages = tuple(_tool_image_msg(f"img{i}.png") for i in range(5))

    evicted = evict_tool_result_images(messages, keep_n=2)

    # Chunk-aligned (chunk=keep_n=2): remove 3 → round down to 2; keeps the 3 most recent.
    assert _refs(evicted) == ["img2.png", "img3.png", "img4.png"]
    assert count_tool_result_images(evicted) == 3


def test_evict_targets_only_tool_images() -> None:
    user_msg = {"role": "user", "content": [{"type": "image", "source_ref": "user.png", "mime_type": "image/png"}]}
    messages = (user_msg, _tool_image_msg("t0.png"), _tool_image_msg("t1.png"), _tool_image_msg("t2.png"))

    evicted = evict_tool_result_images(messages, keep_n=1)

    # User-content image untouched; only the most-recent tool image survives.
    assert evicted[0]["content"][0]["source_ref"] == "user.png"
    assert _refs([m for m in evicted if m["role"] == "tool"]) == ["t2.png"]


def test_evict_off_by_default() -> None:
    messages = tuple(_tool_image_msg(f"img{i}.png") for i in range(5))
    assert evict_tool_result_images(messages, None) == messages


def test_image_dimensions_png_and_jpeg() -> None:
    png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 4, 3)
    assert image_dimensions(png, "image/png") == (4, 3)

    # Minimal JPEG: SOI + SOF0 marker carrying height=7, width=5.
    jpeg = b"\xff\xd8" + b"\xff\xc0" + struct.pack(">H", 17) + b"\x08" + struct.pack(">HH", 7, 5)
    assert image_dimensions(jpeg, "image/jpeg") == (5, 7)

    assert image_dimensions(b"not an image", "image/gif") is None
