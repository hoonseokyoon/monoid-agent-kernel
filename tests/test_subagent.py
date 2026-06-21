from __future__ import annotations

import json
from pathlib import Path

from conftest import tool_binding

from native_agent_runner.core.agents import AgentRuntimeConfig, PromptSpec
from native_agent_runner.core.spec import AgentRunSpec, RunLimits
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelRequest, ModelTurn
from native_agent_runner.providers.fake import fake_tool_call
from native_agent_runner.recorder import MemoryEventSink

# Routing fake adapter: parent and child runs share one adapter (the child inherits
# ``model_adapter``), so we route by a marker baked into each config's persona segment
# to give each role its own independent turn script — no shared pop list, no races.

PARENT_MARK = "[[ROLE=PARENT]]"
CHILD_MARK = "[[ROLE=CHILD]]"


class RoutingAdapter:
    def __init__(self, parent: list[ModelTurn], child: list[ModelTurn]) -> None:
        self.scripts = {"PARENT": list(parent), "CHILD": list(child)}
        self.requests: list[ModelRequest] = []

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        role = "CHILD" if CHILD_MARK in request.system_prompt else "PARENT"
        script = self.scripts[role]
        if not script:
            return ModelTurn(final_text=f"{role} idle")
        return script.pop(0)


def _parent_config(*, tools: tuple = (("agent.spawn",))) -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        definition_id="parent",
        prompt=PromptSpec(persona_segments=(PARENT_MARK,)),
        tools=tuple(tool_binding(t) for t in tools),
    )


def _child_config(*tool_ids: str) -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        definition_id="child",
        prompt=PromptSpec(persona_segments=(CHILD_MARK,)),
        tools=tuple(tool_binding(t) for t in tool_ids),
    )


def _spawn_call(prompt: str, *, background: bool = False, call_id: str = "c1") -> ModelTurn:
    args: dict[str, object] = {"subagent_type": "child", "prompt": prompt}
    if background:
        args["background"] = True
    return ModelTurn(tool_calls=(fake_tool_call("agent_spawn", args, call_id),))


def _loop(
    tmp_path: Path,
    adapter: RoutingAdapter,
    parent: AgentRuntimeConfig,
    *,
    limits: RunLimits | None = None,
    child: AgentRuntimeConfig | None = None,
    event_sinks: tuple = (),
) -> AgentLoop:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace,
            run_root=tmp_path / "runs",
            limits=limits or RunLimits(),
        ),
        model_adapter=adapter,
        runtime_config_provider=parent,
        subagent_definitions={"child": child or _child_config()},
        event_sinks=event_sinks,
    )


def _all_observation_outputs(adapter: RoutingAdapter) -> list[dict]:
    return [obs.output for req in adapter.requests for obs in req.observations]


def test_foreground_returns_child_final_message(tmp_path: Path) -> None:
    adapter = RoutingAdapter(
        parent=[_spawn_call("do X"), ModelTurn(final_text="parent done")],
        child=[ModelTurn(final_text="CHILD_OUTPUT")],
    )
    loop = _loop(tmp_path, adapter, _parent_config())

    result = loop.run_once("go")

    assert result.status == "completed"
    assert result.final_text == "parent done"
    # The child actually ran (its prompt drove a child-role request).
    assert any(r.instruction == "do X" and CHILD_MARK in r.system_prompt for r in adapter.requests)
    # The child's final message came back to the parent as the spawn tool result.
    outputs = json.dumps(_all_observation_outputs(adapter))
    assert "CHILD_OUTPUT" in outputs


def test_child_workspace_is_isolated_from_parent(tmp_path: Path) -> None:
    adapter = RoutingAdapter(
        parent=[_spawn_call("write a file"), ModelTurn(final_text="parent done")],
        child=[
            ModelTurn(tool_calls=(fake_tool_call("fs_write", {"path": "child.txt", "content": "hi"}, "w1"),)),
            ModelTurn(final_text="wrote child.txt"),
        ],
    )
    loop = _loop(tmp_path, adapter, _parent_config(), child=_child_config("fs.write"))

    result = loop.run_once("go")

    assert result.status == "completed"
    # The child's overlay write must NOT surface in the parent's proposal (isolation).
    assert "child.txt" not in result.metrics.get("changed_paths", [])


def test_depth_cap_rejects_spawn(tmp_path: Path) -> None:
    adapter = RoutingAdapter(
        parent=[_spawn_call("too deep"), ModelTurn(final_text="parent done")],
        child=[ModelTurn(final_text="should not run")],
    )
    # max depth 0: a top-level spawn (depth 0) is already at the cap.
    loop = _loop(tmp_path, adapter, _parent_config(), limits=RunLimits(max_subagent_depth=0))

    result = loop.run_once("go")

    assert result.status == "completed"
    outputs = json.dumps(_all_observation_outputs(adapter))
    assert "subagent_depth_exceeded" in outputs
    # The child never ran.
    assert not any(CHILD_MARK in r.system_prompt for r in adapter.requests)


def test_fanout_cap_rejects_second_spawn(tmp_path: Path) -> None:
    adapter = RoutingAdapter(
        parent=[
            _spawn_call("first", call_id="c1"),
            _spawn_call("second", call_id="c2"),
            ModelTurn(final_text="parent done"),
        ],
        child=[ModelTurn(final_text="CHILD_ONE")],
    )
    loop = _loop(tmp_path, adapter, _parent_config(), limits=RunLimits(max_subagents=1))

    result = loop.run_once("go")

    assert result.status == "completed"
    outputs = json.dumps(_all_observation_outputs(adapter))
    assert "CHILD_ONE" in outputs  # first spawn succeeded
    assert "subagent_fanout_exceeded" in outputs  # second was rejected


def test_background_spawn_returns_started_then_delivers_result(tmp_path: Path) -> None:
    # Background: the spawn returns "started" immediately (the parent gets a turn
    # before the child finishes), and the child's result is later delivered as a
    # user-message follow-up that drives another parent turn.
    adapter = RoutingAdapter(
        parent=[
            _spawn_call("bg task", background=True),
            ModelTurn(final_text="continued without child"),
            ModelTurn(final_text="got child"),
        ],
        child=[ModelTurn(final_text="CHILD_BG_OUTPUT")],
    )
    loop = _loop(tmp_path, adapter, _parent_config())

    result = loop.run_once("go")

    # The spawn CALL's own result (call_id c1) was a background "started" ack — not the
    # child's final message (which is delivered separately, later, as a follow-up).
    spawn_results = [
        obs.output for req in adapter.requests for obs in req.observations if obs.call_id == "c1"
    ]
    assert spawn_results, "the spawn tool result should have been observed"
    started = json.dumps(spawn_results).lower()
    assert '"background": true' in started or '"background":true' in started
    assert "child_bg_output" not in started  # result was not returned synchronously
    # The detached child's result was delivered later and drove the final parent turn.
    assert result.final_text == "got child"
    assert any(CHILD_MARK in r.system_prompt for r in adapter.requests)


def test_subagent_events_correlate_to_spawn_call(tmp_path: Path) -> None:
    sink = MemoryEventSink()
    adapter = RoutingAdapter(
        parent=[_spawn_call("do X", call_id="c1"), ModelTurn(final_text="parent done")],
        child=[ModelTurn(final_text="CHILD_OUTPUT")],
    )
    loop = _loop(tmp_path, adapter, _parent_config(), event_sinks=(sink,))

    loop.run_once("go")

    by_type = {e.type: e for e in sink.events}
    assert "subagent.started" in by_type
    assert "subagent.finished" in by_type
    started = by_type["subagent.started"]
    finished = by_type["subagent.finished"]
    # started nests under the spawn tool call; finished nests under started (close pairing).
    tool_starts = {e.event_id: e for e in sink.events if e.type == "tool.call.started"}
    assert started.parent_id in tool_starts
    assert tool_starts[started.parent_id].data.get("tool") == "agent_spawn"
    assert finished.parent_id == started.event_id
    assert finished.data["status"] == "completed"
    assert "usage" in finished.data
    # The child run records to its OWN run dir; the parent's external sink never sees the
    # child's internal events (only the subagent.* summary on the parent stream).
    child_run_id = started.data["child_run_id"]
    assert all(e.run_id != child_run_id for e in sink.events)


def test_child_at_max_depth_has_no_spawn_tool(tmp_path: Path) -> None:
    # max depth 1: the parent (depth 0) may spawn, but the child (depth 1) is at the cap,
    # so its agent.spawn binding is stripped and the tool is absent from its requests.
    adapter = RoutingAdapter(
        parent=[_spawn_call("delegate", call_id="c1"), ModelTurn(final_text="parent done")],
        child=[ModelTurn(final_text="child done")],
    )
    # The child config DOES bind agent.spawn — without depth-hiding it would be exposed.
    child_cfg = _child_config("agent.spawn")
    loop = _loop(
        tmp_path,
        adapter,
        _parent_config(),
        limits=RunLimits(max_subagent_depth=1),
        child=child_cfg,
    )

    result = loop.run_once("go")

    assert result.status == "completed"
    child_requests = [r for r in adapter.requests if CHILD_MARK in r.system_prompt]
    assert child_requests, "the child must have run"
    for req in child_requests:
        assert all(getattr(t, "id", "") != "agent.spawn" for t in req.tools)
