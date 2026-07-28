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
from monoid_agent_kernel.public_view import (
    PREVIEW_BYTE_BUDGET,
    PREVIEW_MAX_ITEMS,
    PREVIEW_MAX_KEYS,
    TRUNCATION_SUFFIX,
)

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


def test_a_truncated_path_says_so_instead_of_reading_as_an_exact_filename(tmp_path: Path) -> None:
    """`paths` entries stay strings, so the cut has to be marked *in* the string.

    `narration._target` prefers `args_preview[path]`, but skips it when it is a preview dict — which
    is exactly what a long path becomes — and falls back to joining `data["paths"]`. So an unmarked
    prefix is what the CLI and Studio present to an operator as the exact target of a write, with
    nothing distinguishing it from a short path that was published whole.

    Asserted on two paths that share the truncated prefix, because that is the case a bound alone
    does not cover: both entries are identical after the cut, and the marker is the only thing left
    telling a reader the name is incomplete rather than a real file called `.../nested/nested`.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    shared = "deeply/" + "nested/" * 60
    calls = [
        ("fs.write", {"path": f"{shared}alpha.txt", "content": "a"}),
        ("fs.write", {"path": f"{shared}beta.txt", "content": "b"}),
    ]

    result = _run(_spec(tmp_path), calls, "fs.write")

    assert validate_run_dir(result.run_dir) == []
    published = [
        event["data"]["paths"][0]
        for event in _events(result.run_dir, "tool.call.started")
        if event["data"].get("paths")
    ]
    assert len(published) == 2
    for entry in published:
        assert isinstance(entry, str)
        assert entry.endswith(TRUNCATION_SUFFIX), f"{entry!r} reads as a complete filename"
        assert len(entry.encode()) <= PREVIEW_BYTE_BUDGET + len(TRUNCATION_SUFFIX.encode())
    assert published[0] == published[1], (
        "the two paths do collide after the cut — which is why the marker, not the prefix, "
        "is what tells a reader the name is partial"
    )


def test_a_short_path_is_published_whole_and_unmarked(tmp_path: Path) -> None:
    """The other side of the bound. Without this, marking *every* path would pass the test above
    while making every ordinary filename in the UI end in an ellipsis it did not earn."""
    result = _run(
        _spec(tmp_path), [("fs.write", {"path": "notes.md", "content": "x"})], "fs.write"
    )

    started = [e for e in _events(result.run_dir, "tool.call.started") if e["data"].get("paths")]
    assert started[0]["data"]["paths"] == ["notes.md"]


def test_plan_updated_carries_the_same_capped_items_as_the_tool_call_event(tmp_path: Path) -> None:
    items = [{"step": f"{LONG_STEP}-{index}", "status": "pending"} for index in range(30)]

    result = _run(_spec(tmp_path), [("run.update_plan", {"items": items})], "run.update_plan")

    plan_data = _events(result.run_dir, "plan.updated")[-1]["data"]
    published = plan_data["items"]
    on_the_call = _events(result.run_dir, "tool.call.started")[0]["data"]["args_preview"]["items"]
    # The two routes carry the same *values* under the same bounds; they differ only in where the
    # truncation count is written. `plan.updated.items` is a typed array a renderer iterates, so its
    # count is a sibling key; `args_preview` is a generic JSON blob, where the marker element is the
    # signal and there is no typed consumer to confuse. Pinned as an exact identity rather than
    # loosened to a prefix match, so neither route can quietly move its cap.
    assert on_the_call == [*published, {"truncated_items": plan_data["truncated_items"]}]

    # And the bound really bound: capped in width, and each step capped in bytes.
    assert len(published) == PREVIEW_MAX_ITEMS
    assert plan_data["truncated_items"] == len(items) - PREVIEW_MAX_ITEMS
    # A *string*, not a preview dict. `WorkspaceInspector.svelte` renders `{item.step}` directly, so
    # a dict here renders `[object Object]` — and `svelte-check` cannot catch it, because
    # `run-state.ts` casts the items through `unknown` to `PlanItem[]`.
    assert isinstance(published[0]["step"], str)
    assert published[0]["step"].endswith(TRUNCATION_SUFFIX)
    assert len(published[0]["step"].encode()) <= PREVIEW_BYTE_BUDGET + len(TRUNCATION_SUFFIX.encode())
    assert LONG_STEP not in (result.run_dir / "events.jsonl").read_text(encoding="utf-8")


def test_plan_items_stay_renderable_objects_with_string_steps(tmp_path: Path) -> None:
    """Two separate properties, and an earlier version of this test asserted the second one wrong.

    ``plan.updated.items`` is ``_OBJ_ARRAY``, so each *item* must stay an object — that is the
    schema half. The renderer half is that ``items[].step`` must stay a **string**: the previous
    implementation replaced it with a ``{"preview": ...}`` dict, which `validate_run_dir` accepts
    and `WorkspaceInspector.svelte:52` renders as ``[object Object]``. The test passed while
    claiming the renderer "still finds a row rather than a blank"; it found a row reading
    ``[object Object]``. Both halves are asserted here.
    """
    items = [{"step": LONG_STEP, "status": "pending"}]

    result = _run(_spec(tmp_path), [("run.update_plan", {"items": items})], "run.update_plan")

    assert validate_run_dir(result.run_dir) == []
    published = _events(result.run_dir, "plan.updated")[-1]["data"]["items"]
    assert isinstance(published[0], dict)
    assert published[0]["status"] == "pending"
    assert isinstance(published[0]["step"], str)
    assert LONG_STEP[:20] in published[0]["step"], "the step is truncated, not replaced"


def test_a_long_plan_is_capped_without_a_foreign_element_inside_the_typed_array(
    tmp_path: Path,
) -> None:
    """The *other* half of the same renderer contract, and the half the first fix missed.

    Capping the step keeps each item renderable; capping the list used to append a
    ``{"truncated_items": n}`` element, which is an object (so ``_OBJ_ARRAY`` accepts it) but not a
    *plan item*. ``run-state.ts:265`` casts the array to ``PlanItem[]`` wholesale, so that element
    reached ``WorkspaceInspector.svelte`` as a row with no ``step`` — a blank line — and, because
    line 49 divides by ``plan.length``, it also inflated the progress denominator: a 25-step plan
    with every step done rendered ``20/21``, permanently one short of finished.

    So the count moves to a sibling key and the array stays homogeneous. Asserted on the shape a
    consumer actually relies on — *every* element carries the two fields the renderer reads — rather
    than on the last element only, which is how the marker survived the first pass.
    """
    items = [{"step": f"step {index}", "status": "pending"} for index in range(25)]

    result = _run(_spec(tmp_path), [("run.update_plan", {"items": items})], "run.update_plan")

    assert validate_run_dir(result.run_dir) == []
    data = _events(result.run_dir, "plan.updated")[-1]["data"]
    published = data["items"]
    assert len(published) == 20
    assert all(isinstance(item, dict) for item in published)
    assert all(isinstance(item.get("step"), str) for item in published), "no marker object smuggled in"
    assert all(item.get("status") == "pending" for item in published)
    assert data["truncated_items"] == 5, "the drop is reported, just not as a plan item"

    # `status.json` is the twin surface: it copies `items` out of this same event, so moving the
    # count out of the array silently shortened the plan there too unless the count travels with it.
    # Capping is fine; capping without saying so reads as "that was the whole plan".
    status = json.loads((result.run_dir / "status.json").read_text(encoding="utf-8"))
    assert len(status["plan"]) == 20
    assert status["plan_truncated_items"] == 5


def test_a_later_shorter_plan_clears_the_stale_truncation_count(tmp_path: Path) -> None:
    """`status.json` is reassigned per event, not merged, and the count has to follow that.

    A run that publishes a long plan and then replaces it with a short one would otherwise leave
    `plan_truncated_items` behind from the first, reporting a drop against a list that no longer has
    one — a stale number being worse than an absent one, since a reader cannot tell it is stale.
    """
    long_plan = [{"step": f"step {index}", "status": "pending"} for index in range(25)]
    short_plan = [{"step": "only step", "status": "pending"}]

    result = _run(
        _spec(tmp_path),
        [("run.update_plan", {"items": long_plan}), ("run.update_plan", {"items": short_plan})],
        "run.update_plan",
    )

    status = json.loads((result.run_dir / "status.json").read_text(encoding="utf-8"))
    assert len(status["plan"]) == 1
    assert "plan_truncated_items" not in status


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
