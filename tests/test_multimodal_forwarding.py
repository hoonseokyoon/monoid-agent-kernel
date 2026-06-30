from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from support.runtime import runtime_config, runtime_provider

from native_agent_runner.core.content import AudioPart, DocumentPart, ImagePart, TextPart
from native_agent_runner.core.spec import AgentRunSpec, ModelConfig, RunLimits
from native_agent_runner.errors import WorkspaceError
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelRequest, ModelTurn, ToolObservation
from native_agent_runner.providers.fake import (
    FakeModelAdapter,
    FakeMultimodalModelAdapter,
    fake_tool_call,
)
from native_agent_runner.providers.gateway import GatewayModelAdapter
from native_agent_runner.providers.openai import _message_to_input_items
from native_agent_runner.tools.builtin import builtin_tools
from native_agent_runner.workspace.local import LocalWorkspaceBackend

_PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
# Includes a NUL byte (as real PDFs do in stream data) so fs.read rejects it as non-text,
# while the leading "%PDF-" magic still identifies it.
_PDF_BYTES = b"%PDF-1.4\n1 0 obj<<>>stream\n\x00\x01\x02binary\nendstream\n%%EOF\n"


def _workspace_with_image(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("img.png").write_bytes(_PNG_BYTES)
    return workspace


def _workspace_with_pdf(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("doc.pdf").write_bytes(_PDF_BYTES)
    return workspace


def _events(result) -> list:
    return [
        json.loads(line)
        for line in result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _finish_turn() -> ModelTurn:
    return ModelTurn(
        response_id="r1",
        tool_calls=(fake_tool_call("run_finish", {"summary": "done"}, "c"),),
    )


def test_fake_adapter_forwards_resolved_base64(tmp_path: Path) -> None:
    """End-to-end: a multimodal adapter receives the workspace image resolved to a base64
    wire block, and no degraded warning is emitted."""
    workspace = _workspace_with_image(tmp_path)
    adapter = FakeMultimodalModelAdapter(turns=[_finish_turn()])
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
    ).run_once((TextPart("describe"), ImagePart(source_ref="img.png", mime_type="image/png")))

    user = [m for m in adapter.requests[0].messages if m["role"] == "user"][0]
    blocks = [p for p in user["content"] if p.get("type") == "image"]
    assert len(blocks) == 1
    source = blocks[0]["source"]
    assert source["type"] == "base64"
    assert source["media_type"] == "image/png"
    assert base64.b64decode(source["data"]) == _PNG_BYTES

    events = [
        json.loads(line)
        for line in result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert not [e for e in events if e["type"] == "model.input.degraded"]


def test_text_only_adapter_still_degrades(tmp_path: Path) -> None:
    """A non-multimodal adapter drops the image and emits the degraded warning."""
    workspace = _workspace_with_image(tmp_path)
    adapter = FakeModelAdapter(turns=[_finish_turn()])
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
    ).run_once((TextPart("describe"), ImagePart(source_ref="img.png", mime_type="image/png")))

    events = [
        json.loads(line)
        for line in result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    degraded = [e for e in events if e["type"] == "model.input.degraded"]
    assert len(degraded) == 1
    assert degraded[0]["data"]["reason"] == "adapter_lacks_multimodal"
    # The image was never resolved to a base64 wire block; it stays by-reference, and the
    # real text-only adapter projects it to text on its own send path.
    user = [m for m in adapter.requests[0].messages if m["role"] == "user"][0]
    images = [p for p in user["content"] if p.get("type") == "image"]
    assert images and images[0].get("source_ref") == "img.png"
    assert "source" not in images[0]


def test_openai_message_to_input_items_maps_image() -> None:
    """OpenAI mapping: a neutral base64 image block becomes a Responses input_image data-URL."""
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
            },
        ],
    }

    items = _message_to_input_items(message)

    assert len(items) == 1
    content = items[0]["content"]
    assert {"type": "input_text", "text": "describe"} in content
    assert {"type": "input_image", "image_url": "data:image/png;base64,QUJD"} in content


def test_oversize_wire_settles_limited(tmp_path: Path) -> None:
    """A multimodal turn whose resolved (base64) payload exceeds the wire-size cap settles
    ``limited`` with ``wire_bytes_exceeded`` rather than being sent."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("big.png").write_bytes(_PNG_BYTES + b"\x00" * 5000)
    adapter = FakeMultimodalModelAdapter(turns=[_finish_turn()])
    # Durable by-reference log is tiny (well under the cap); the resolved base64 payload
    # (~6.7KB) is what trips the guard.
    spec = AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        limits=RunLimits(max_message_log_bytes=2000),
    )

    result = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
    ).run_once((TextPart("describe"), ImagePart(source_ref="big.png", mime_type="image/png")))

    assert result.status == "limited"
    assert result.error_code == "wire_bytes_exceeded"


def test_block_count_cliff_warns(tmp_path: Path) -> None:
    """More than 20 resolved image blocks in one request emits the cliff warning."""
    workspace = _workspace_with_image(tmp_path)
    adapter = FakeMultimodalModelAdapter(turns=[_finish_turn()])
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")
    parts = tuple(ImagePart(source_ref="img.png", mime_type="image/png") for _ in range(21))

    result = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
    ).run_once((TextPart("many"), *parts))

    events = [
        json.loads(line)
        for line in result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cliff = [
        e
        for e in events
        if e["type"] == "model.input.degraded" and e["data"].get("reason") == "image_block_count_cliff"
    ]
    assert cliff and cliff[0]["data"]["block_count"] == 21


def test_fs_read_media_returns_image_part(tmp_path: Path) -> None:
    """fs.read_media returns a by-reference ImagePart for an image; fs.read still rejects it."""
    workspace = LocalWorkspaceBackend(_workspace_with_image(tmp_path))
    tools = {tool.id: tool for tool in builtin_tools(workspace)}

    result = tools["fs.read_media"].handler(None, {"path": "img.png"})  # type: ignore[arg-type]
    assert result.ok
    assert len(result.media) == 1
    assert result.media[0].source_ref == "img.png"
    assert result.media[0].mime_type == "image/png"
    assert result.content["mime_type"] == "image/png"

    # fs.read with no media capability (context None) still rejects the binary.
    with pytest.raises(WorkspaceError):
        tools["fs.read"].handler(None, {"path": "img.png"})  # type: ignore[arg-type]


def test_fs_read_binary_points_at_read_media(tmp_path: Path) -> None:
    workspace = LocalWorkspaceBackend(_workspace_with_image(tmp_path))
    tools = {tool.id: tool for tool in builtin_tools(workspace)}

    # Binary/non-utf8 → an actionable error naming fs.read_media (which enforces its own scope,
    # quota, and authorization), not a bare reject and not media read under fs.read's binding.
    with pytest.raises(WorkspaceError, match="fs.read_media"):
        tools["fs.read"].handler(None, {"path": "img.png"})  # type: ignore[arg-type]


def test_fs_read_text_is_unchanged(tmp_path: Path) -> None:
    ws_dir = tmp_path / "workspace"
    ws_dir.mkdir()
    ws_dir.joinpath("a.txt").write_text("hello world\n", encoding="utf-8")
    workspace = LocalWorkspaceBackend(ws_dir)
    tools = {tool.id: tool for tool in builtin_tools(workspace)}

    result = tools["fs.read"].handler(None, {"path": "a.txt"})  # type: ignore[arg-type]
    assert result.ok
    assert "hello world" in result.content["content"]
    assert result.media == ()


def test_tool_media_observation_round_trips() -> None:
    """ToolObservation media survive checkpoint serialization (by reference)."""
    obs = ToolObservation(
        call_id="c1",
        tool_name="fs.read_media",
        output={"ok": True, "result": {"path": "img.png"}},
        media=({"type": "image", "source_ref": "img.png", "mime_type": "image/png"},),
    )
    assert ToolObservation.from_json(obs.to_json()) == obs


def test_tool_observation_from_json_accepts_legacy_images_key() -> None:
    """A checkpoint written before the rename (``images`` key) still restores its media."""
    legacy = {
        "call_id": "c1",
        "tool_name": "fs.read_media",
        "output": {"ok": True},
        "images": [{"type": "image", "source_ref": "img.png", "mime_type": "image/png"}],
    }
    restored = ToolObservation.from_json(legacy)
    assert restored.media == ({"type": "image", "source_ref": "img.png", "mime_type": "image/png"},)


def test_openai_tool_message_splits_to_followup_user() -> None:
    """A tool message with resolved images → function_call_output + a follow-up user image."""
    message = {
        "role": "tool",
        "call_id": "c1",
        "content": {"ok": True, "result": {"path": "img.png"}},
        "media": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}
        ],
    }

    items = _message_to_input_items(message)

    assert items[0]["type"] == "function_call_output"
    assert items[0]["call_id"] == "c1"
    assert items[1]["role"] == "user"
    assert {"type": "input_image", "image_url": "data:image/png;base64,QUJD"} in items[1]["content"]


def test_tool_image_forwarded_end_to_end(tmp_path: Path) -> None:
    """A tool returning an image: the resolved wire tool message carries the base64 image."""
    workspace = _workspace_with_image(tmp_path)
    adapter = FakeMultimodalModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_read_media", {"path": "img.png"}, "c1"),),
            ),
            _finish_turn(),
        ]
    )
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")

    AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("fs.read_media", "run.finish")),
    ).run_once("show me the image")

    tool_messages = [
        m for req in adapter.requests for m in req.messages if m.get("role") == "tool" and m.get("media")
    ]
    assert tool_messages, "expected a tool message carrying resolved media"
    block = tool_messages[0]["media"][0]
    assert block["source"]["type"] == "base64"
    assert base64.b64decode(block["source"]["data"]) == _PNG_BYTES


def test_tool_images_evicted_on_wire(tmp_path: Path) -> None:
    """With keep_recent_tool_images set, older tool-result images are dropped from the wire."""
    workspace = _workspace_with_image(tmp_path)
    adapter = FakeMultimodalModelAdapter(
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("fs_read_media", {"path": "img.png"}, "c1"),)),
            ModelTurn(response_id="r2", tool_calls=(fake_tool_call("fs_read_media", {"path": "img.png"}, "c2"),)),
            _finish_turn(),
        ]
    )
    spec = AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        limits=RunLimits(keep_recent_tool_images=1),
    )

    AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("fs.read_media", "run.finish")),
    ).run_once("read it twice")

    # The final request saw two tool images accumulate; eviction keeps only the most recent.
    last = adapter.requests[-1]
    total = sum(len(m.get("media", [])) for m in last.messages if m.get("role") == "tool")
    assert total == 1


def test_gateway_payload_forwards_image_block_verbatim() -> None:
    """Gateway wire: a multimodal GatewayModelAdapter forwards resolved image blocks in
    ``messages`` verbatim (no text projection) for the downstream gateway to map."""
    adapter = GatewayModelAdapter(config=ModelConfig(provider="gateway", model="gpt-5.5"))
    block = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}
    request = ModelRequest(
        instruction=None,
        system_prompt="sys",
        tools=(),
        messages=({"role": "user", "content": [{"type": "text", "text": "hi"}, block]},),
    )

    payload = adapter._payload(request)

    assert payload["messages"][0]["content"] == [{"type": "text", "text": "hi"}, block]


# ---- P6a/P6c: documents (PDF) ------------------------------------------------------

def test_document_input_forwarded(tmp_path: Path) -> None:
    """A user-submitted PDF reaches the wire as a base64 document block (+ filename); no degrade."""
    workspace = _workspace_with_pdf(tmp_path)
    adapter = FakeMultimodalModelAdapter(turns=[_finish_turn()])
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
    ).run_once((TextPart("summarize"), DocumentPart(source_ref="doc.pdf", mime_type="application/pdf")))

    user = [m for m in adapter.requests[0].messages if m["role"] == "user"][0]
    blocks = [p for p in user["content"] if p.get("type") == "document"]
    assert len(blocks) == 1
    source = blocks[0]["source"]
    assert source["type"] == "base64"
    assert source["media_type"] == "application/pdf"
    assert base64.b64decode(source["data"]) == _PDF_BYTES
    assert blocks[0]["filename"] == "doc.pdf"
    assert not [e for e in _events(result) if e["type"] == "model.input.degraded"]


def test_openai_maps_document_to_input_file() -> None:
    """A neutral base64 document block becomes a Responses input_file with a filename."""
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "see"},
            {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": "QUJD"},
                "filename": "report.pdf",
            },
        ],
    }

    items = _message_to_input_items(message)

    content = items[0]["content"]
    assert {
        "type": "input_file",
        "filename": "report.pdf",
        "file_data": "data:application/pdf;base64,QUJD",
    } in content


def test_fs_read_media_returns_document_part_for_pdf(tmp_path: Path) -> None:
    workspace = LocalWorkspaceBackend(_workspace_with_pdf(tmp_path))
    tools = {tool.id: tool for tool in builtin_tools(workspace)}

    result = tools["fs.read_media"].handler(None, {"path": "doc.pdf"})  # type: ignore[arg-type]
    assert result.ok
    assert len(result.media) == 1
    assert isinstance(result.media[0], DocumentPart)
    assert result.media[0].mime_type == "application/pdf"
    assert result.content["mime_type"] == "application/pdf"

    with pytest.raises(WorkspaceError):
        tools["fs.read"].handler(None, {"path": "doc.pdf"})  # type: ignore[arg-type]


def test_tool_result_document_forwarded(tmp_path: Path) -> None:
    """P6c: a tool returning a PDF (fs.read_media) reaches the wire as a resolved document block."""
    workspace = _workspace_with_pdf(tmp_path)
    adapter = FakeMultimodalModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_read_media", {"path": "doc.pdf"}, "c1"),),
            ),
            _finish_turn(),
        ]
    )
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")

    AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("fs.read_media", "run.finish")),
    ).run_once("read the pdf")

    tool_messages = [
        m for req in adapter.requests for m in req.messages if m.get("role") == "tool" and m.get("media")
    ]
    assert tool_messages, "expected a tool message carrying resolved media"
    block = tool_messages[0]["media"][0]
    assert block["type"] == "document"
    assert base64.b64decode(block["source"]["data"]) == _PDF_BYTES


def test_document_degrades_on_text_only_adapter(tmp_path: Path) -> None:
    workspace = _workspace_with_pdf(tmp_path)
    adapter = FakeModelAdapter(turns=[_finish_turn()])
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
    ).run_once((TextPart("x"), DocumentPart(source_ref="doc.pdf", mime_type="application/pdf")))

    degraded = [e for e in _events(result) if e["type"] == "model.input.degraded"]
    assert len(degraded) == 1
    assert degraded[0]["data"]["reason"] == "adapter_lacks_multimodal"
    assert degraded[0]["data"]["dropped_part_types"] == ["document"]


def test_audio_degrades_even_on_multimodal_adapter(tmp_path: Path) -> None:
    """P6b: audio/video round-trip but are not forwarded — a multimodal adapter still degrades
    them (reason ``type_not_forwarded``) and they never reach the wire."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("clip.mp3").write_bytes(b"ID3\x03\x00fake-audio")
    adapter = FakeMultimodalModelAdapter(turns=[_finish_turn()])
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
    ).run_once((TextPart("describe"), AudioPart(source_ref="clip.mp3", mime_type="audio/mpeg")))

    degraded = [e for e in _events(result) if e["type"] == "model.input.degraded"]
    assert len(degraded) == 1
    assert degraded[0]["data"]["reason"] == "type_not_forwarded"
    assert degraded[0]["data"]["dropped_part_types"] == ["audio"]
    # The audio part is dropped from the wire; only text reaches the model.
    user = [m for m in adapter.requests[0].messages if m["role"] == "user"][0]
    assert all(p.get("type") != "audio" for p in user["content"] if isinstance(p, dict))


# --- inline ingress -> content-addressed blob normalization (gap 2a) --------------------


def test_normalize_inline_media_part_and_blob_resolver(tmp_path: Path) -> None:
    from native_agent_runner.core.media import (
        WorkspaceMediaResolver,
        blob_shas_in_messages,
        normalize_inline_media_part,
        parse_data_uri,
    )

    data_uri = "data:image/png;base64," + base64.b64encode(_PNG_BYTES).decode()
    assert parse_data_uri(data_uri) == ("image/png", _PNG_BYTES)
    assert parse_data_uri("img.png") is None

    store: dict[str, bytes] = {}
    part = ImagePart(source_ref=data_uri, mime_type="image/png")
    norm = normalize_inline_media_part(part, store)
    assert norm.source_ref.startswith("blob:")
    sha = norm.source_ref[len("blob:"):]
    assert store[sha] == _PNG_BYTES
    # A by-reference (workspace) part passes through untouched.
    ws_part = ImagePart(source_ref="img.png", mime_type="image/png")
    assert normalize_inline_media_part(ws_part, store) is ws_part

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    resolver = WorkspaceMediaResolver(LocalWorkspaceBackend(root=workspace), blobs=store)
    resolved = resolver.resolve(norm.source_ref, "image/png")
    assert resolved.data == _PNG_BYTES and resolved.sha256 == sha

    msgs = ({"role": "user", "content": [{"type": "image", "source_ref": norm.source_ref, "mime_type": "image/png"}]},)
    assert sha in blob_shas_in_messages(msgs)


def test_inline_media_ingested_to_blob_survives_restore(tmp_path: Path) -> None:
    # The durability claim: an inline (by-value) image is normalized to a content-addressed blob,
    # so it survives a restart AND a base re-provisioning (the new workspace has NO attachment file)
    # — resolution reads from the rehydrated blob, not the workspace.
    data_uri = "data:image/png;base64," + base64.b64encode(_PNG_BYTES).decode()
    workspace = tmp_path / "workspace"
    workspace.mkdir()  # deliberately empty: no img.png on disk

    adapter_a = FakeMultimodalModelAdapter(turns=[_finish_turn()])
    spec_a = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")
    loop_a = AgentLoop(
        spec=spec_a,
        model_adapter=adapter_a,
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
    )
    loop_a.open()
    loop_a.submit((TextPart("describe"), ImagePart(source_ref=data_uri, mime_type="image/png")))

    # The inline bytes were forwarded (resolved from the blob, since no workspace file exists).
    user_a = [m for m in adapter_a.requests[0].messages if m["role"] == "user"][0]
    img_a = [p for p in user_a["content"] if p.get("type") == "image"][0]
    assert base64.b64decode(img_a["source"]["data"]) == _PNG_BYTES

    checkpoint = loop_a.snapshot()
    assert checkpoint is not None
    blobs = loop_a.collect_checkpoint_blobs()
    # The DURABLE log holds a blob: ref, not the bytes; the bytes live in the content blob.
    logged_user = [m for m in checkpoint.messages if m["role"] == "user"][0]
    blob_ref = logged_user["content"][1]["source_ref"]
    assert blob_ref.startswith("blob:")
    assert blob_ref[len("blob:"):] in blobs
    loop_a.close()

    # Fresh process: a re-provisioned, EMPTY workspace; restore only from the checkpoint + blobs.
    workspace2 = tmp_path / "workspace2"
    workspace2.mkdir()
    adapter_b = FakeMultimodalModelAdapter(turns=[_finish_turn()])
    spec_b = AgentRunSpec(workspace_root=workspace2, run_root=tmp_path / "runs2", run_id=spec_a.run_id)
    loop_b = AgentLoop(
        spec=spec_b,
        model_adapter=adapter_b,
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
    )
    loop_b.restore(checkpoint, blobs=blobs)
    loop_b.submit("again")

    # The resent log STILL forwards the image, resolved from the rehydrated blob.
    user_b = [m for m in adapter_b.requests[0].messages if m["role"] == "user"][0]
    img_b = [p for p in user_b["content"] if p.get("type") == "image"][0]
    assert base64.b64decode(img_b["source"]["data"]) == _PNG_BYTES
    loop_b.close()


# --- close-out: resolver blob fallback, tool inline media, gap-2b eager guard ----------


def test_resolver_blob_reader_fallback(tmp_path: Path) -> None:
    from native_agent_runner.core.media import WorkspaceMediaResolver

    (tmp_path / "workspace").mkdir()
    workspace = LocalWorkspaceBackend(tmp_path / "workspace")
    sha = __import__("hashlib").sha256(_PNG_BYTES).hexdigest()
    # Not in the in-memory map; only reachable via the blob_reader (e.g. the checkpoint store).
    resolver = WorkspaceMediaResolver(
        workspace, blobs={}, blob_reader=lambda s: _PNG_BYTES if s == sha else (_ for _ in ()).throw(KeyError(s))
    )
    resolved = resolver.resolve(f"blob:{sha}", "image/png")
    assert resolved.data == _PNG_BYTES and resolved.sha256 == sha
    # A blob absent from both map and reader raises.
    with pytest.raises(Exception):
        resolver.resolve("blob:" + "0" * 64, "image/png")


def test_observation_message_normalizes_inline_tool_media() -> None:
    # A tool that returns inline (data:) media gets it normalized to a durable blob — symmetric
    # with user-input media (gap: tool-result inline media).
    from native_agent_runner.loop import _observation_message
    from native_agent_runner.providers.base import ToolObservation

    data_uri = "data:image/png;base64," + base64.b64encode(_PNG_BYTES).decode()
    obs = ToolObservation(
        tool_name="snap", call_id="c1", output="took a screenshot",
        media=({"type": "image", "source_ref": data_uri, "mime_type": "image/png"},),
    )
    store: dict[str, bytes] = {}
    msg = _observation_message(obs, store)
    ref = msg["media"][0]["source_ref"]
    assert ref.startswith("blob:")
    assert store[ref[len("blob:"):]] == _PNG_BYTES


def test_fs_read_media_rejects_oversized_eagerly(tmp_path: Path) -> None:
    # gap 2b: media that would not fit the wire-build cap (max_bytes_read) is rejected at the tool
    # call — adjacent to the cause — with an actionable message, not late at wire-build.
    ws_dir = tmp_path / "workspace"
    ws_dir.mkdir()
    ws_dir.joinpath("big.png").write_bytes(_PNG_BYTES + b"\x00" * 2000)
    workspace = LocalWorkspaceBackend(ws_dir, max_bytes_read=500)
    tools = {tool.id: tool for tool in builtin_tools(workspace)}
    with pytest.raises(WorkspaceError) as exc:
        # max_bytes arg larger than the run cap would otherwise let a doomed reference through.
        tools["fs.read_media"].handler(None, {"path": "big.png", "max_bytes": 10000})  # type: ignore[arg-type]
    assert "max_bytes_read" in str(exc.value)
