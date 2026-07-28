"""Model-authored structures are bounded on the fan-out stream, not just on the way in.

``run.update_plan`` and ``artifact.emit`` each publish their argument twice: once through
``tool.call.started.data.args_preview``, which caps it, and once through a dedicated event
(``plan.updated`` / ``artifact.emitted``) which republished it raw. Same value, same run, one route
capped and the other not — the twin-miss shape, with the uncapped half being the one an operator
actually reads.

So the assertions below are mostly *equality between the two routes* rather than a bound restated
by hand. A bound restated by hand drifts: that is precisely how the two halves came apart. If a
future change moves the cap, these fail rather than silently protecting one door again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core.schemas import validate_run_dir
from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.public_view import PREVIEW_BYTE_BUDGET, PREVIEW_MAX_ITEMS, PREVIEW_MAX_KEYS

LONG_STEP = "가" * 200  # 600 bytes: over the threshold, under the old 160-character slice.


def _spec(tmp_path: Path) -> AgentRunSpec:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")


def _events(run_dir: Path, event_type: str) -> list[dict[str, Any]]:
    text = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [record for record in records if record["type"] == event_type]


def _run(spec: AgentRunSpec, calls: list[tuple[str, dict[str, Any]]], *tool_ids: str) -> Any:
    turns = [
        ModelTurn(response_id=f"r{index}", tool_calls=[fake_tool_call(name, args)])
        for index, (name, args) in enumerate(calls)
    ]
    turns.append(ModelTurn(response_id="rz", final_text="done"))
    loop = AgentLoop(
        spec=spec,
        model_adapter=FakeModelAdapter(turns=turns),
        runtime_config_provider=runtime_provider(runtime_config(*tool_ids, "run.finish")),
    )
    return loop.run_once("go")


def test_plan_updated_carries_the_same_capped_items_as_the_tool_call_event(tmp_path: Path) -> None:
    items = [{"step": f"{LONG_STEP}-{index}", "status": "pending"} for index in range(30)]

    result = _run(_spec(tmp_path), [("run.update_plan", {"items": items})], "run.update_plan")

    published = _events(result.run_dir, "plan.updated")[-1]["data"]["items"]
    on_the_call = _events(result.run_dir, "tool.call.started")[0]["data"]["args_preview"]["items"]
    assert published == on_the_call, "the two routes for the same value disagree again"

    # And the bound really bound: capped in width, and each step capped in bytes.
    assert len(published) == PREVIEW_MAX_ITEMS + 1
    assert published[-1] == {"truncated_items": len(items) - PREVIEW_MAX_ITEMS}
    assert len(published[0]["step"]["preview"].encode()) <= PREVIEW_BYTE_BUDGET
    assert LONG_STEP not in (result.run_dir / "events.jsonl").read_text(encoding="utf-8")


def test_plan_items_stay_objects_so_the_run_dir_still_validates(tmp_path: Path) -> None:
    """``plan.updated.items`` is ``_OBJ_ARRAY``; a step replaced by a preview dict is fine, an
    *item* replaced by one is not. ``WorkspaceInspector`` reads ``items[].step`` directly, so this
    also pins that the renderer still finds a row rather than a blank."""
    items = [{"step": LONG_STEP, "status": "pending"}]

    result = _run(_spec(tmp_path), [("run.update_plan", {"items": items})], "run.update_plan")

    assert validate_run_dir(result.run_dir) == []
    published = _events(result.run_dir, "plan.updated")[-1]["data"]["items"]
    assert isinstance(published[0], dict)
    assert published[0]["status"] == "pending"
    assert published[0]["step"]["truncated"] is True


def test_status_json_gets_the_capped_plan_rather_than_a_second_uncapped_copy(tmp_path: Path) -> None:
    """``StatusJsonSink`` copies ``plan.updated.items`` verbatim and the projection service serves
    ``status.json`` wholesale, so capping at the emit site is what closes this one too."""
    items = [{"step": LONG_STEP, "status": "pending"}]

    result = _run(_spec(tmp_path), [("run.update_plan", {"items": items})], "run.update_plan")

    raw = (result.run_dir / "status.json").read_text(encoding="utf-8")
    assert LONG_STEP not in raw
    assert json.loads(raw)["plan"] == _events(result.run_dir, "plan.updated")[-1]["data"]["items"]


def test_artifact_metadata_is_capped_on_the_event_but_not_in_the_model_s_own_result(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    (spec.workspace_root / "note.md").write_text("hi", encoding="utf-8")
    metadata = {f"k{index}": LONG_STEP for index in range(PREVIEW_MAX_KEYS + 5)}

    result = _run(
        spec,
        [("artifact.emit", {"path": "note.md", "kind": "note", "metadata": metadata})],
        "artifact.emit",
    )

    published = _events(result.run_dir, "artifact.emitted")[-1]["data"]["metadata"]
    assert published["truncated_keys"] == 5
    assert len(published) == PREVIEW_MAX_KEYS + 1
    assert LONG_STEP not in (result.run_dir / "events.jsonl").read_text(encoding="utf-8")

    on_the_call = _events(result.run_dir, "tool.call.started")[0]["data"]["args_preview"]["metadata"]
    assert published == on_the_call


def test_ordinary_plans_and_metadata_are_published_unchanged(tmp_path: Path) -> None:
    """The cap is a ceiling on hostile input, not a reshaping of normal tool use.

    Without this, "cap everything" would pass every assertion above while turning every plan an
    operator reads into preview dicts.
    """
    items = [{"step": "Read the file", "status": "done"}, {"step": "Write it back", "status": "pending"}]

    result = _run(_spec(tmp_path), [("run.update_plan", {"items": items})], "run.update_plan")

    assert _events(result.run_dir, "plan.updated")[-1]["data"]["items"] == items
