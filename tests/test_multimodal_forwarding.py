from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from conftest import runtime_config, runtime_provider

from native_agent_runner.core.content import ImagePart, TextPart
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


def _workspace_with_image(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("img.png").write_bytes(_PNG_BYTES)
    return workspace


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
    assert len(result.images) == 1
    assert result.images[0].source_ref == "img.png"
    assert result.images[0].mime_type == "image/png"
    assert result.content["mime_type"] == "image/png"

    # fs.read is unchanged — still rejects the binary as non-text.
    with pytest.raises(WorkspaceError):
        tools["fs.read"].handler(None, {"path": "img.png"})  # type: ignore[arg-type]


def test_tool_image_observation_round_trips() -> None:
    """ToolObservation images survive checkpoint serialization (by reference)."""
    obs = ToolObservation(
        call_id="c1",
        tool_name="fs.read_media",
        output={"ok": True, "result": {"path": "img.png"}},
        images=({"type": "image", "source_ref": "img.png", "mime_type": "image/png"},),
    )
    assert ToolObservation.from_json(obs.to_json()) == obs


def test_openai_tool_message_splits_to_followup_user() -> None:
    """A tool message with resolved images → function_call_output + a follow-up user image."""
    message = {
        "role": "tool",
        "call_id": "c1",
        "content": {"ok": True, "result": {"path": "img.png"}},
        "images": [
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
        m for req in adapter.requests for m in req.messages if m.get("role") == "tool" and m.get("images")
    ]
    assert tool_messages, "expected a tool message carrying resolved images"
    block = tool_messages[0]["images"][0]
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
    total = sum(len(m.get("images", [])) for m in last.messages if m.get("role") == "tool")
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
