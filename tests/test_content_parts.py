from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import runtime_config, runtime_provider

from native_agent_runner.core.content import (
    DocumentPart,
    ImagePart,
    TextPart,
    content_part_from_json,
    content_part_to_json,
    non_text_part_types,
)
from native_agent_runner.core.spec import AgentRunSpec, text_from_parts
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call


def _provider():
    return runtime_provider(runtime_config("run.finish"))


@pytest.mark.parametrize(
    "part",
    [
        TextPart("hello"),
        ImagePart(source_ref="img.png", mime_type="image/png"),
        DocumentPart(source_ref="doc.pdf", mime_type="application/pdf"),
    ],
)
def test_content_part_codec_round_trip(part) -> None:
    assert content_part_from_json(content_part_to_json(part)) == part


def test_unknown_part_type_rejected() -> None:
    with pytest.raises(ValueError, match="unknown content part type"):
        content_part_from_json({"type": "video", "source_ref": "x"})


def test_effective_input_synthesizes_text_from_instruction() -> None:
    spec = AgentRunSpec(workspace_root=Path("/ws"), run_root=Path("runs"))
    assert spec.input == ()
    assert spec.effective_input == ()


def test_effective_input_uses_explicit_parts() -> None:
    parts = (TextPart("a"), DocumentPart(source_ref="d.pdf", mime_type="application/pdf"))
    spec = AgentRunSpec(
        workspace_root=Path("/ws"),
        run_root=Path("runs"),
        input=parts,
    )
    assert spec.effective_input == parts


def test_effective_text_instruction_uses_explicit_text_parts() -> None:
    parts = (
        TextPart("first"),
        ImagePart(source_ref="i.png", mime_type="image/png"),
        TextPart("second"),
    )

    assert text_from_parts(parts) == "first\n\nsecond"


def test_effective_text_instruction_falls_back_when_input_has_no_text() -> None:
    parts = (ImagePart(source_ref="i.png", mime_type="image/png"),)

    assert text_from_parts(parts) == ""


def test_non_text_part_types_helper() -> None:
    parts = (
        TextPart("a"),
        ImagePart(source_ref="i.png", mime_type="image/png"),
        DocumentPart(source_ref="d.pdf", mime_type="application/pdf"),
        ImagePart(source_ref="j.png", mime_type="image/png"),
    )
    assert non_text_part_types(parts) == ["image", "document"]
    assert non_text_part_types((TextPart("only text"),)) == []


def test_spec_round_trip_preserves_input_parts() -> None:
    spec = AgentRunSpec(
        workspace_root=Path("/ws"),
        run_root=Path("runs"),
        input=(TextPart("hi"), DocumentPart(source_ref="d.pdf", mime_type="application/pdf")),
    )
    restored = AgentRunSpec.from_json(json.loads(json.dumps(spec.to_json())))
    assert restored == spec


def test_non_text_input_emits_degraded_warning(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("run_finish", {"summary": "done"}, "c"),),
            )
        ]
    )
    spec = AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
    )

    result = AgentLoop(spec=spec, model_adapter=adapter, runtime_config_provider=_provider()).run_once(
        (TextPart("describe the image"), ImagePart(source_ref="i.png", mime_type="image/png"))
    )

    events = [
        json.loads(line)
        for line in result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    degraded = [e for e in events if e["type"] == "model.input.degraded"]
    assert len(degraded) == 1
    assert degraded[0]["level"] == "warning"
    assert degraded[0]["data"]["dropped_part_types"] == ["image"]
    assert degraded[0]["data"]["reason"] == "adapter_lacks_multimodal"


def test_explicit_text_input_is_sent_to_model(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("run_finish", {"summary": "done"}, "c"),),
            )
        ]
    )
    spec = AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
    )

    AgentLoop(spec=spec, model_adapter=adapter, runtime_config_provider=_provider()).run_once(
        (TextPart("explicit text"),)
    )

    assert adapter.requests[0].instruction == "explicit text"


def test_text_only_input_emits_no_degraded_warning(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("run_finish", {"summary": "done"}, "c"),),
            )
        ]
    )
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop(spec=spec, model_adapter=adapter, runtime_config_provider=_provider()).run_once(
        "plain text"
    )

    events = [
        json.loads(line)
        for line in result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert not [e for e in events if e["type"] == "model.input.degraded"]
