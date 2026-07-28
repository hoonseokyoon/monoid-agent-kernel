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
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.public_view import (
    PREVIEW_BYTE_BUDGET,
    PREVIEW_MAX_ITEMS,
    PREVIEW_MAX_KEYS,
    REDACTED_PATH,
    TRUNCATION_SUFFIX,
    redacted_value,
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


def test_a_bad_path_argument_fails_the_call_instead_of_the_run(tmp_path: Path) -> None:
    """The consequence half of the fail-closed rule, which the unit test cannot reach.

    `is_path_redacted` normalizes before matching and raises on an absolute or `..` path. Every
    builder that calls it sits inside event construction, so the raise escaped `_emit_tool_started`
    *before* validation and the error handler retried the same emission — turning one malformed,
    model-authored argument into a terminated run, for any operator who had configured
    `redact_patterns` and no one else.

    So this asserts on the run surviving, not on the redaction. `test_preview_bounds` already pins
    that `public_path` returns `[redacted-path]`; what could not be seen from there is that the
    difference between "redacted" and "raises" is the difference between a tool error the model can
    correct and no result at all.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    spec = AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        permission_policy=PermissionPolicy(redact_patterns=("secrets/**",)),
    )

    result = _run(spec, [("fs.write", {"path": "/etc/passwd", "content": "x"})], "fs.write")

    assert result.final_text == "done", "the run ended early instead of continuing past a bad call"
    assert validate_run_dir(result.run_dir) == []
    # The model gets an observation it can correct, which is the whole difference being asserted.
    assert _events(result.run_dir, "tool.call.failed"), "no failed-call observation was recorded"

    # And the *projections* carry the marker rather than the argument. Scoped to these fields on
    # purpose: `tool.call.failed.data.error` does name the rejected path, and that is deliberate —
    # `docs/OBSERVABILITY.md` lists error messages among the surfaces that carry paths, because an
    # operator debugging a denied write needs to know which write was denied.
    started = _events(result.run_dir, "tool.call.started")[0]["data"]
    assert started["paths"] == [REDACTED_PATH]
    assert started["args_preview"]["path"] == redacted_value("/etc/passwd")


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
    # A prefix check alone passes on an uncapped step -- it is true of the whole value too, so this
    # test held with the cap removed entirely. The bound is what makes it a *truncation*.
    assert published[0]["step"] != LONG_STEP, "the step was published whole"
    assert published[0]["step"].endswith(TRUNCATION_SUFFIX)
    assert len(published[0]["step"].encode()) <= PREVIEW_BYTE_BUDGET + len(TRUNCATION_SUFFIX.encode())
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


def test_a_list_nested_inside_a_plan_item_keeps_its_own_truncation_marker(tmp_path: Path) -> None:
    """Suppressing the marker is scoped to the array with the typed consumer, not to every depth.

    The first version of this propagated `list_marker=False` all the way down, on the reasoning that
    a rule bound at one site should be bound at its twins. Nested lists are not that twin: only the
    root `items` array is cast to `PlanItem[]` and iterated by element shape. A list *inside* an item
    is an ordinary JSON blob, and dropping its marker deleted elements that nothing reported — the
    sibling `truncated_items` count measures the root only, so `len(items) - len(published)` was 0
    while ten entries had disappeared. A silent cap, which is the failure this release exists to
    stop, introduced by the fix for a different one.
    """
    items = [
        {"step": f"step {index}", "status": "pending", "evidence": [f"ref-{n}" for n in range(30)]}
        for index in range(25)
    ]

    result = _run(_spec(tmp_path), [("run.update_plan", {"items": items})], "run.update_plan")

    assert validate_run_dir(result.run_dir) == []
    data = _events(result.run_dir, "plan.updated")[-1]["data"]
    published = data["items"]

    # The root array is still typed and still reports its drop out-of-band.
    assert all(isinstance(item.get("step"), str) for item in published)
    assert data["truncated_items"] == 5

    # The nested blob keeps the in-band marker, so its drop is reported too.
    nested = published[0]["evidence"]
    assert len(nested) == PREVIEW_MAX_ITEMS + 1
    assert nested[-1] == {"truncated_items": 30 - PREVIEW_MAX_ITEMS}


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

    # The first plan has to have *set* a count, or "it is absent at the end" is true for the wrong
    # reason: with the cap regressed no count is ever written and this test passes having observed
    # nothing. Asserting the intermediate state is what makes the clearing observable.
    first = _events(result.run_dir, "plan.updated")[0]["data"]
    assert first["truncated_items"] == 25 - PREVIEW_MAX_ITEMS, "no count was ever set to clear"

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

    # "...but not in the model's own result" -- the half the name promises, and the half that makes
    # the cap a *publication* boundary rather than data loss. `tool.call.finished` deliberately
    # carries no result, so the model's copy is observable only where it actually goes: the private
    # transcript. Capping the tool's return value would have left this test green.
    # The `tool_observation` record specifically, not the file. `LONG_STEP` is also in the
    # `model_turn` record -- the arguments the model *sent* -- so a whole-file search is satisfied
    # by the model's own outbound copy and stays green with the return value capped, which is the
    # one thing this half exists to catch.
    observations = [
        json.loads(line)
        for line in (result.run_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("kind") == "tool_observation"
    ]
    assert observations, "no tool observation was recorded"
    returned = observations[0]["output"]["result"]["artifact"]["metadata"]
    assert returned == metadata, "the model wrote this and must get it back whole"


def test_ordinary_plans_and_metadata_are_published_unchanged(tmp_path: Path) -> None:
    """The cap is a ceiling on hostile input, not a reshaping of normal tool use.

    Without this, "cap everything" would pass every assertion above while turning every plan an
    operator reads into preview dicts.
    """
    items = [{"step": "Read the file", "status": "done"}, {"step": "Write it back", "status": "pending"}]

    result = _run(_spec(tmp_path), [("run.update_plan", {"items": items})], "run.update_plan")

    assert _events(result.run_dir, "plan.updated")[-1]["data"]["items"] == items


def _redacting_spec(tmp_path: Path) -> AgentRunSpec:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        permission_policy=PermissionPolicy(redact_patterns=("secrets/**", "secrets/*")),
    )


def test_a_declared_path_argument_is_redacted_in_the_preview_and_not_only_in_paths(
    tmp_path: Path,
) -> None:
    """The redaction was defeated by the field beside it, on the same event.

    `preview_value` matched a hardcoded `{path, root, cwd}` while the registry declares path
    arguments per tool, so `fs.move` published `paths: ["[redacted-path]"]` next to
    `args_preview.source_path: "secrets/creds.txt"`. An operator reading either field alone would
    draw opposite conclusions about whether their `redact_patterns` worked.
    """
    spec = _redacting_spec(tmp_path)
    (spec.workspace_root / "secrets").mkdir()
    (spec.workspace_root / "secrets" / "creds.txt").write_text("k", encoding="utf-8")

    result = _run(
        spec,
        [("fs.move", {"source_path": "secrets/creds.txt", "destination_path": "public/out.txt"})],
        "fs.move",
    )

    started = _events(result.run_dir, "tool.call.started")[0]["data"]
    assert started["paths"][0] == REDACTED_PATH
    assert started["args_preview"]["source_path"] == redacted_value("secrets/creds.txt")
    assert "secrets/creds.txt" not in json.dumps(started)


# The three files a run publishes. Asserting over the set, not over `events.jsonl` alone, is the
# point: `metrics.json` was the one nobody checked, and it was the one that did not redact.
PUBLIC_FILES = ("events.jsonl", "status.json", "metrics.json")


def _public_text(run_dir: Path) -> dict[str, str]:
    return {name: (run_dir / name).read_text(encoding="utf-8") for name in PUBLIC_FILES if (run_dir / name).exists()}


def test_every_public_file_redacts_a_matched_path_not_just_the_two_that_were_checked(
    tmp_path: Path,
) -> None:
    """`metrics.json` published `changed_paths` raw while its two siblings redacted.

    Both callers of `build_metrics` apply `public_path` to the same list a few lines later for the
    events they emit, so an operator who configured `redact_patterns`, looked at `events.jsonl` and
    `status.json`, and saw `[redacted-path]` had every reason to conclude it had worked — with the
    whole path sitting in the third file.

    Found by driving a run and grepping its output rather than by reading the emit sites, which is
    why several passes over the surrounding code walked past it.
    """
    spec = _redacting_spec(tmp_path)
    (spec.workspace_root / "secrets").mkdir()

    result = _run(spec, [("fs.write", {"path": "secrets/creds.txt", "content": "x"})], "fs.write")

    metrics = json.loads((result.run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["changed_paths"] == [REDACTED_PATH], "metrics.json published the path raw"
    assert validate_run_dir(result.run_dir) == []
    for name, text in _public_text(result.run_dir).items():
        assert "secrets/creds.txt" not in text, f"{name} carries the redacted path"


def test_a_free_text_argument_that_looks_like_an_enum_is_still_bounded(tmp_path: Path) -> None:
    """`artifact.emit`'s `kind` is `{"type": "string"}` with no enum — free model text.

    It sat unbounded between three bounded neighbours (`label` truncates, `metadata` is previewed,
    `path` is a run-dir pointer), because it reads like an enum. So do `timeout_s`,
    `max_output_bytes` and `startup_wait_s` on `shell.exec`, which are declared
    `["integer", "null"]` and were copied verbatim — `tool.call.started` is emitted *before*
    `validate_args` rejects the call, so the schema does not protect this surface.
    """
    spec = _spec(tmp_path)
    (spec.workspace_root / "note.md").write_text("hi", encoding="utf-8")
    sentinel = "기밀" * 400

    result = _run(spec, [("artifact.emit", {"path": "note.md", "kind": sentinel})], "artifact.emit")

    # The assertion the six older tests in this file all carry, and that the four added with
    # these fixes all omitted -- which is exactly why the suite stayed green while `kind`
    # published a preview envelope into a field its schema declares as a string.
    assert validate_run_dir(result.run_dir) == []

    for name, text in _public_text(result.run_dir).items():
        assert sentinel not in text, f"{name} carries the whole `kind`"


def test_an_unknown_job_id_fails_the_call_instead_of_the_run(tmp_path: Path) -> None:
    """`TaskManager` raises `KeyError`, which tool dispatch does not catch.

    A model asking about a job that already finished — or inventing an id — terminated the run and
    republished its own argument into `run.failed`, `status.json` and `metrics.json`. Same shape as
    the `WorkspaceError` that ended runs for operators with `redact_patterns` configured, on four
    twins (`job.status` / `logs` / `cancel` / `wait`) that guard never reached.
    """
    sentinel = "기밀" * 300

    result = _run(_spec(tmp_path), [("job.status", {"job_id": sentinel})], "job.status")

    assert validate_run_dir(result.run_dir) == []

    assert result.status == "completed", "one bad argument ended the run"
    assert result.final_text == "done"
    for name, text in _public_text(result.run_dir).items():
        assert sentinel not in text, f"{name} republished the argument"


def test_a_provider_supplied_response_id_is_bounded(tmp_path: Path) -> None:
    """`response_id` and `previous_turn_handle` arrive from the gateway — outside the trust boundary.

    Real ids are short, so this needs a hostile or buggy proxy rather than a hostile model. It is
    here because no amount of reviewing *tool arguments* would ever reach it: the value never passes
    through a tool, a preview builder, or the permission policy.
    """
    spec = _spec(tmp_path)
    sentinel = "기밀" * 3_000
    loop = AgentLoop(
        spec=spec,
        model_adapter=FakeModelAdapter(turns=[ModelTurn(response_id=sentinel, final_text="done")]),
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
    )
    result = loop.run_once("go")

    assert validate_run_dir(result.run_dir) == []
    for name, text in _public_text(result.run_dir).items():
        assert sentinel not in text, f"{name} carries the whole response_id"


def test_the_terminal_error_is_filtered_on_every_public_artifact_not_three_of_four(
    tmp_path: Path,
) -> None:
    """`public_error_message` was bound on three surfaces and missed on the fourth.

    `events.jsonl`, `status.json` and `failure.json` rendered `[redacted-sensitive-error]` while
    `metrics.json` carried the message whole — and `_error_from_status_body` embeds the *entire* LLM
    gateway HTTP response body in it, so a 400 from a misconfigured gateway put whatever that body
    held into a public run artifact.

    The `changed_paths` list in the same function had been routed through `public_path` one commit
    earlier, under a comment calling `metrics.json` "the only one of the three that never redacted".
    The list got bound; the error string twenty-six lines below it did not.

    `AgentRunResult.error` stays raw on purpose: the embedding application is inside the trust
    boundary and needs the whole message. The reference backend filters it again before serving it
    over HTTP, because a provider's response body is not the run's own data to hand back.
    """
    secret = "-----BEGIN PRIVATE KEY-----" + "기밀" * 200

    class FailingAdapter:
        def next_turn(self, request: Any) -> Any:
            raise ModelAdapterError(f"LLM gateway returned HTTP 400: {secret}", retryable=False)

    spec = _spec(tmp_path)
    loop = AgentLoop(
        spec=spec,
        model_adapter=FailingAdapter(),
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
    )
    result = loop.run_once("go")

    for name in (*PUBLIC_FILES, "failure.json"):
        path = result.run_dir / name
        if path.exists():
            assert secret not in path.read_text(encoding="utf-8"), f"{name} carries the raw error"

    assert secret in str(result.error), "the in-process caller must still get the whole message"
