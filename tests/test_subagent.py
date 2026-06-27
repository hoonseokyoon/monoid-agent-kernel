from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from conftest import tool_binding

from native_agent_runner.core.agents import AgentRuntimeConfig, PromptSpec, SubagentDefinition
from native_agent_runner.core.spec import AgentRunSpec, ModelConfig, RunLimits
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelRequest, ModelTurn
from native_agent_runner.providers.fake import fake_tool_call
from native_agent_runner.recorder import MemoryEventSink
from native_agent_runner.tools.base import ToolContext, ToolResult, ToolSpec

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


def _parent_config(*tool_ids: str, model: ModelConfig | None = None) -> AgentRuntimeConfig:
    # The parent always binds agent.spawn (so it can delegate) plus any extra tools the
    # children should be able to inherit (children can never exceed the parent).
    ids = ("agent.spawn", *tool_ids)
    return AgentRuntimeConfig(
        definition_id="parent",
        model=model,
        prompt=PromptSpec(persona_segments=(PARENT_MARK,)),
        tools=tuple(tool_binding(t) for t in ids),
    )


def _child_def(
    *,
    tools: tuple[str, ...] | None = None,
    disallowed: tuple[str, ...] = (),
    model: ModelConfig | None = None,
    mode=None,
    limits: RunLimits | None = None,
    description: str = "",
    context: str = "fresh",
) -> SubagentDefinition:
    return SubagentDefinition(
        description=description,
        prompt=PromptSpec(persona_segments=(CHILD_MARK,)),
        tools=tools,
        disallowed_tools=disallowed,
        model=model,
        mode=mode,
        limits=limits,
        context=context,  # type: ignore[arg-type]
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
    child: SubagentDefinition | None = None,
    event_sinks: tuple = (),
    tool_providers: tuple = (),
) -> AgentLoop:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace,
            run_root=tmp_path / "runs",
            limits=limits or RunLimits(),
        ),
        model_adapter=adapter,
        runtime_config_provider=parent,
        subagent_definitions={"child": child or _child_def()},
        event_sinks=event_sinks,
        tool_providers=tool_providers,
    )


def _all_observation_outputs(adapter: RoutingAdapter) -> list[dict]:
    return [obs.output for req in adapter.requests for obs in req.observations]


def _child_tool_ids(adapter: RoutingAdapter) -> set[str]:
    ids: set[str] = set()
    for req in adapter.requests:
        if CHILD_MARK in req.system_prompt:
            ids.update(getattr(t, "id", "") for t in req.tools)
    return ids


# --- core behavior (P1) ------------------------------------------------------------


def test_foreground_returns_child_final_message(tmp_path: Path) -> None:
    adapter = RoutingAdapter(
        parent=[_spawn_call("do X"), ModelTurn(final_text="parent done")],
        child=[ModelTurn(final_text="CHILD_OUTPUT")],
    )
    loop = _loop(tmp_path, adapter, _parent_config())

    result = loop.run_once("go")

    assert result.status == "completed"
    assert result.final_text == "parent done"
    assert any(r.instruction == "do X" and CHILD_MARK in r.system_prompt for r in adapter.requests)
    assert "CHILD_OUTPUT" in json.dumps(_all_observation_outputs(adapter))


def test_child_workspace_is_isolated_from_parent(tmp_path: Path) -> None:
    adapter = RoutingAdapter(
        parent=[_spawn_call("write a file"), ModelTurn(final_text="parent done")],
        child=[
            ModelTurn(tool_calls=(fake_tool_call("fs_write", {"path": "child.txt", "content": "hi"}, "w1"),)),
            ModelTurn(final_text="wrote child.txt"),
        ],
    )
    # Parent must expose fs.write for the child to inherit it (hard ceiling).
    loop = _loop(tmp_path, adapter, _parent_config("fs.write"))

    result = loop.run_once("go")

    assert result.status == "completed"
    assert "child.txt" not in result.metrics.get("changed_paths", [])


def test_depth_cap_rejects_spawn(tmp_path: Path) -> None:
    adapter = RoutingAdapter(
        parent=[_spawn_call("too deep"), ModelTurn(final_text="parent done")],
        child=[ModelTurn(final_text="should not run")],
    )
    loop = _loop(tmp_path, adapter, _parent_config(), limits=RunLimits(max_subagent_depth=0))

    result = loop.run_once("go")

    assert result.status == "completed"
    assert "subagent_depth_exceeded" in json.dumps(_all_observation_outputs(adapter))
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
    assert "CHILD_ONE" in outputs
    assert "subagent_fanout_exceeded" in outputs


def test_background_spawn_returns_started_then_delivers_result(tmp_path: Path) -> None:
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

    spawn_results = [
        obs.output for req in adapter.requests for obs in req.observations if obs.call_id == "c1"
    ]
    assert spawn_results, "the spawn tool result should have been observed"
    started = json.dumps(spawn_results).lower()
    assert '"background": true' in started or '"background":true' in started
    assert "child_bg_output" not in started
    assert result.final_text == "got child"
    assert any(CHILD_MARK in r.system_prompt for r in adapter.requests)


# --- observability (P2) ------------------------------------------------------------


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
    tool_starts = {e.event_id: e for e in sink.events if e.type == "tool.call.started"}
    assert started.parent_id in tool_starts
    assert tool_starts[started.parent_id].data.get("tool") == "agent_spawn"
    assert finished.parent_id == started.event_id
    assert finished.data["status"] == "completed"
    assert "usage" in finished.data
    child_run_id = started.data["child_run_id"]
    assert all(e.run_id != child_run_id for e in sink.events)


# --- Claude-parity permissions (P2.5) ----------------------------------------------


def test_child_inherits_all_parent_tools_by_default(tmp_path: Path) -> None:
    adapter = RoutingAdapter(
        parent=[_spawn_call("inherit"), ModelTurn(final_text="parent done")],
        child=[ModelTurn(final_text="child done")],
    )
    loop = _loop(tmp_path, adapter, _parent_config("fs.read", "fs.write"), child=_child_def())

    loop.run_once("go")

    child_tools = _child_tool_ids(adapter)
    assert {"agent.spawn", "fs.read", "fs.write"} <= child_tools


def test_allowlist_restricts_to_subset(tmp_path: Path) -> None:
    adapter = RoutingAdapter(
        parent=[_spawn_call("read only"), ModelTurn(final_text="parent done")],
        child=[ModelTurn(final_text="child done")],
    )
    loop = _loop(
        tmp_path,
        adapter,
        _parent_config("fs.read", "fs.write"),
        child=_child_def(tools=("fs.read",)),
    )

    loop.run_once("go")

    child_tools = _child_tool_ids(adapter)
    assert "fs.read" in child_tools
    assert "fs.write" not in child_tools
    assert "agent.spawn" not in child_tools  # not in the allowlist


def test_disallowed_tools_win_over_inherit(tmp_path: Path) -> None:
    adapter = RoutingAdapter(
        parent=[_spawn_call("no writes"), ModelTurn(final_text="parent done")],
        child=[ModelTurn(final_text="child done")],
    )
    loop = _loop(
        tmp_path,
        adapter,
        _parent_config("fs.read", "fs.write"),
        child=_child_def(disallowed=("fs.write",)),
    )

    loop.run_once("go")

    child_tools = _child_tool_ids(adapter)
    assert "fs.read" in child_tools
    assert "fs.write" not in child_tools


def test_child_cannot_exceed_parent_ceiling(tmp_path: Path) -> None:
    # Parent does NOT expose fs.write; an allowlist asking for it yields nothing.
    adapter = RoutingAdapter(
        parent=[_spawn_call("want write"), ModelTurn(final_text="parent done")],
        child=[ModelTurn(final_text="child done")],
    )
    loop = _loop(
        tmp_path,
        adapter,
        _parent_config("fs.read"),
        child=_child_def(tools=("fs.write",)),
    )

    loop.run_once("go")

    assert "fs.write" not in _child_tool_ids(adapter)


def test_child_at_max_depth_has_no_spawn_tool(tmp_path: Path) -> None:
    adapter = RoutingAdapter(
        parent=[_spawn_call("delegate", call_id="c1"), ModelTurn(final_text="parent done")],
        child=[ModelTurn(final_text="child done")],
    )
    # max depth 1: the child (depth 1) is at the cap, so agent.spawn is stripped even
    # though it would otherwise inherit it from the parent.
    loop = _loop(tmp_path, adapter, _parent_config(), limits=RunLimits(max_subagent_depth=1))

    result = loop.run_once("go")

    assert result.status == "completed"
    assert any(CHILD_MARK in r.system_prompt for r in adapter.requests)
    assert "agent.spawn" not in _child_tool_ids(adapter)


class _DemoToolProvider:
    """A stand-in for an MCP / custom tool provider: yields one tool the parent can bind."""

    def get_tools(self, context: ToolContext):
        def handler(ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(ok=True, content={"pong": True})

        return [
            ToolSpec(
                id="mcp.demo.ping",
                description="demo ping",
                input_schema={"type": "object", "properties": {}, "additionalProperties": True},
                capability="mcp.demo.ping",
                side_effect="read",
                handler=handler,
            )
        ]


def test_mcp_custom_provider_tools_inherited_by_child(tmp_path: Path) -> None:
    adapter = RoutingAdapter(
        parent=[_spawn_call("use mcp"), ModelTurn(final_text="parent done")],
        child=[ModelTurn(final_text="child done")],
    )
    # Parent binds the provider tool; child inherits via "mcp.*" allowlist — proving both
    # provider inheritance (the tool is in the child registry) and pattern matching.
    loop = _loop(
        tmp_path,
        adapter,
        _parent_config("mcp.demo.ping"),
        child=_child_def(tools=("mcp.*",)),
        tool_providers=(_DemoToolProvider(),),
    )

    result = loop.run_once("go")

    assert result.status == "completed"
    assert "mcp.demo.ping" in _child_tool_ids(adapter)


def test_child_model_inherits_and_overrides(tmp_path: Path) -> None:
    # Inherit: child def has no model -> child requests use the parent's model.
    adapter = RoutingAdapter(
        parent=[_spawn_call("x"), ModelTurn(final_text="parent done")],
        child=[ModelTurn(final_text="child done")],
    )
    loop = _loop(
        tmp_path,
        adapter,
        _parent_config(model=ModelConfig(model="M-parent")),
        child=_child_def(),
    )
    loop.run_once("go")
    child_models = {r.model.model for r in adapter.requests if CHILD_MARK in r.system_prompt and r.model}
    assert child_models == {"M-parent"}

    # Override: child def sets its own model.
    adapter2 = RoutingAdapter(
        parent=[_spawn_call("x"), ModelTurn(final_text="parent done")],
        child=[ModelTurn(final_text="child done")],
    )
    loop2 = _loop(
        tmp_path / "b",
        adapter2,
        _parent_config(model=ModelConfig(model="M-parent")),
        child=_child_def(model=ModelConfig(model="M-child")),
    )
    loop2.run_once("go")
    child_models2 = {r.model.model for r in adapter2.requests if CHILD_MARK in r.system_prompt and r.model}
    assert child_models2 == {"M-child"}


# --- P3: directory discovery, context fork, usage reporting -------------------------


def test_loader_reads_markdown_directory(tmp_path: Path) -> None:
    from native_agent_runner.subagent_loader import load_subagent_definitions

    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "reviewer.md").write_text(
        "---\n"
        "name: reviewer\n"
        "description: Reviews code\n"
        "tools: [fs.read]\n"
        "disallowedTools: [shell.exec]\n"
        "model: M-sub\n"
        "---\n"
        "You are a careful reviewer.\n",
        encoding="utf-8",
    )
    (agents / "brancher.md").write_text(
        "---\ndescription: continue in a branch\ncontext: fork\n---\nbody\n",
        encoding="utf-8",
    )

    defs = load_subagent_definitions(agents)

    assert set(defs) == {"reviewer", "brancher"}  # brancher id from filename (no name field)
    rev = defs["reviewer"]
    assert rev.description == "Reviews code"
    assert rev.tools == ("fs.read",)  # explicit allowlist
    assert rev.disallowed_tools == ("shell.exec",)
    assert rev.model is not None and rev.model.model == "M-sub"
    assert "careful reviewer" in rev.prompt.persona_segments[0]
    assert defs["brancher"].context == "fork"
    assert defs["brancher"].tools is None  # omitted -> inherit


def test_loader_duplicate_id_warns_and_first_wins(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from native_agent_runner.subagent_loader import load_subagent_definitions

    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "a.md").write_text(
        "---\nname: dup\ndescription: first\n---\nbody a\n", encoding="utf-8"
    )
    (agents / "b.md").write_text(
        "---\nname: dup\ndescription: second\n---\nbody b\n", encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING, logger="native_agent_runner.subagent_loader"):
        defs = load_subagent_definitions(agents)

    # sorted path order: a.md before b.md, so "first" wins...
    assert defs["dup"].description == "first"
    # ...and the dropped file is logged, not silently skipped.
    assert any(
        "duplicate subagent id" in r.message and "dup" in r.message for r in caplog.records
    ), caplog.text


class InstructionAdapter:
    """Routes by ModelRequest.instruction (not by system prompt) — needed for fork, where
    the child inherits the parent's system prompt and the marker can't distinguish them."""

    def __init__(self, by_instruction: dict[str, ModelTurn], default_final: str) -> None:
        self.by_instruction = by_instruction
        self.default_final = default_final
        self.requests: list[ModelRequest] = []

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        if request.instruction in self.by_instruction:
            return self.by_instruction[request.instruction]
        return ModelTurn(final_text=self.default_final)


def test_fork_inherits_parent_context_prompt_and_tools(tmp_path: Path) -> None:
    adapter = InstructionAdapter(
        by_instruction={
            "go": _spawn_call("fork directive"),
            "fork directive": ModelTurn(final_text="FORK_CHILD_DONE"),
        },
        default_final="parent done",
    )
    loop = _loop(
        tmp_path,
        adapter,
        _parent_config("fs.read", model=ModelConfig(model="M-parent")),
        child=_child_def(context="fork"),
    )

    result = loop.run_once("go")
    assert result.status == "completed"

    fork_reqs = [r for r in adapter.requests if r.instruction == "fork directive"]
    assert fork_reqs, "the fork child should have run with the directive as its instruction"
    fr = fork_reqs[0]
    # Inherits the parent's system prompt (NOT the child definition's persona).
    assert PARENT_MARK in fr.system_prompt
    assert CHILD_MARK not in fr.system_prompt
    # Inherits the parent's conversation snapshot (seeded messages contain the parent's "go").
    assert "go" in json.dumps(list(fr.messages))
    # Inherits the parent's tools (fs.read) and model.
    assert any(getattr(t, "id", "") == "fs.read" for t in fr.tools)
    assert fr.model is not None and fr.model.model == "M-parent"


def test_fresh_subagent_starts_with_empty_context(tmp_path: Path) -> None:
    adapter = RoutingAdapter(
        parent=[_spawn_call("do X"), ModelTurn(final_text="parent done")],
        child=[ModelTurn(final_text="child done")],
    )
    loop = _loop(tmp_path, adapter, _parent_config(), child=_child_def(context="fresh"))

    loop.run_once("go")

    child_first = next(r for r in adapter.requests if r.instruction == "do X" and CHILD_MARK in r.system_prompt)
    # A fresh child sees only its own task prompt — none of the parent's conversation.
    assert "go" not in json.dumps(list(child_first.messages or ()))


def test_metrics_report_subagent_usage_separately(tmp_path: Path) -> None:
    adapter = RoutingAdapter(
        parent=[_spawn_call("do X"), ModelTurn(final_text="parent done")],
        child=[ModelTurn(final_text="child done", usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})],
    )
    loop = _loop(tmp_path, adapter, _parent_config())

    result = loop.run_once("go")

    assert result.metrics["subagent_count"] == 1
    assert result.metrics["subagent_usage"]["total_tokens"] == 10
    # The child's tokens must NOT inflate the parent's own usage (context accounting).
    assert result.metrics.get("total_tokens", 0) == 0
