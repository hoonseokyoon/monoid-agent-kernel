from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from support.runtime import runtime_config, runtime_provider, tool_binding

from monoid_agent_kernel.core.context import TurnContext
from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.recorder import MemoryEventSink
from monoid_agent_kernel.skills import SkillDefinition, SkillProvider, load_skill_definitions
from monoid_agent_kernel.skills.definition import SKILL_FILENAME


# --- fixtures / helpers ------------------------------------------------------------


def _write_skill(
    root: Path,
    name: str,
    *,
    description: str = "",
    allowed_tools: str = "",
    context: str = "",
    body: str = "Do the thing.",
    resources: dict[str, str] | None = None,
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {name}"]
    if description:
        lines.append(f"description: {description}")
    if allowed_tools:
        lines.append(f"allowed-tools: {allowed_tools}")
    if context:
        lines.append(f"context: {context}")
    lines += ["---", "", body, ""]
    (skill_dir / SKILL_FILENAME).write_text("\n".join(lines), encoding="utf-8")
    for rel, content in (resources or {}).items():
        target = skill_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return skill_dir


def _specs(provider: SkillProvider) -> dict[str, object]:
    return {spec.id: spec for spec in provider.get_tools()}  # type: ignore[attr-defined]


# --- loader (directory discovery) --------------------------------------------------


def test_loader_parses_skill_directories(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "pdf-fill",
        description="Fill PDF forms",
        allowed_tools="Read Glob",
        body="Use the bundled script.",
        resources={"references/FORMS.md": "FORMS_CONTENT"},
    )
    _write_skill(tmp_path, "commit-msg", description="Write commit messages")

    definitions = load_skill_definitions(tmp_path)

    assert set(definitions) == {"pdf-fill", "commit-msg"}
    pdf = definitions["pdf-fill"]
    assert pdf.description == "Fill PDF forms"
    assert pdf.instructions == "Use the bundled script."
    assert pdf.allowed_tools == ("Read", "Glob")
    assert pdf.directory == tmp_path / "pdf-fill"


def test_loader_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="skills directory not found"):
        load_skill_definitions(tmp_path / "nope")


def test_loader_name_falls_back_to_directory_name(tmp_path: Path) -> None:
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / SKILL_FILENAME).write_text("---\ndescription: d\n---\nbody\n", encoding="utf-8")

    definitions = load_skill_definitions(tmp_path)

    assert set(definitions) == {"my-skill"}
    assert definitions["my-skill"].description == "d"


def test_loader_duplicate_name_first_wins(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write_skill(tmp_path / "a", "dup", description="first")
    _write_skill(tmp_path / "b", "dup", description="second")

    with caplog.at_level(logging.WARNING, logger="monoid_agent_kernel.skills.loader"):
        definitions = load_skill_definitions(tmp_path)

    # sorted path order: a/ before b/, so "first" wins.
    assert definitions["dup"].description == "first"
    # ...and the dropped file is no longer silent — a WARNING names the collision.
    assert any(
        "duplicate skill name" in r.message and "dup" in r.message for r in caplog.records
    ), caplog.text


def test_from_frontmatter_allowed_tools_inline_list() -> None:
    definition = SkillDefinition.from_frontmatter(
        {"name": "x", "allowed-tools": ["fs.read", "shell.exec"]}, "body"
    )
    assert definition.allowed_tools == ("fs.read", "shell.exec")


# --- L1: catalog (dynamic_segment, gated on the skill tool being bound) ------------


def _turn(*, bound_tools: frozenset[str] = frozenset()) -> TurnContext:
    return TurnContext(
        step=1,
        remaining_steps=10,
        remaining_tool_calls=10,
        deadline_s=None,
        plan=(),
        pending_observation_count=0,
        bound_tools=bound_tools,
    )


def test_catalog_emitted_only_when_skill_tool_is_bound(tmp_path: Path) -> None:
    _write_skill(tmp_path, "pdf-fill", description="Fill PDF forms")
    _write_skill(tmp_path, "commit-msg", description="Write commit messages")
    provider = SkillProvider(load_skill_definitions(tmp_path))

    # The catalog is config-gated (DX-17): it is a per-turn dynamic segment, present only when
    # the `skill` tool is bound this turn — so it disappears when the capability is toggled off.
    assert provider.static_segment() is None
    assert provider.dynamic_segment(_turn(bound_tools=frozenset())) is None

    segment = provider.dynamic_segment(_turn(bound_tools=frozenset({"skill"})))
    assert segment is not None
    assert "- pdf-fill: Fill PDF forms" in segment
    assert "- commit-msg: Write commit messages" in segment
    assert "`skill`" in segment  # tells the model how to load one


def test_empty_provider_is_inert() -> None:
    provider = SkillProvider({})
    assert provider.static_segment() is None
    assert provider.dynamic_segment(_turn(bound_tools=frozenset({"skill"}))) is None
    assert list(provider.get_tools()) == []
    assert provider.tool_bindings() == ()


# --- L2: skill tool (load instructions) --------------------------------------------


def test_skill_tool_returns_instructions_resources_and_advisory(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "pdf-fill",
        description="Fill PDF forms",
        allowed_tools="Read Glob",
        body="Step 1. Step 2.",
        resources={"references/FORMS.md": "x", "scripts/fill.py": "print(1)"},
    )
    provider = SkillProvider(load_skill_definitions(tmp_path))
    skill = _specs(provider)["skill"]

    result = skill.handler(None, {"name": "pdf-fill"})

    assert result.ok
    assert result.content["instructions"] == "Step 1. Step 2."
    assert result.content["allowed_tools"] == ["Read", "Glob"]  # advisory
    assert set(result.content["resources"]) == {"references/FORMS.md", "scripts/fill.py"}


def test_inline_allowed_tools_is_advisory_and_does_not_narrow_parent_surface(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        "notes-helper",
        description="Help with notes",
        allowed_tools="fs.read",
        body="Use fs.read when you need existing notes.",
    )
    provider = SkillProvider(load_skill_definitions(skills_root))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("skill", {"name": "notes-helper"}, "c1"),)),
            ModelTurn(
                tool_calls=(
                    fake_tool_call(
                        "fs_write",
                        {"path": "created.md", "content": "ok\n", "create_dirs": False},
                        "c2",
                    ),
                )
            ),
            ModelTurn(final_text="done"),
        ]
    )
    bindings = (*provider.tool_bindings(), tool_binding("fs.write"), tool_binding("run.finish"))

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        context_providers=(provider,),
        tool_providers=(provider,),
        runtime_config_provider=runtime_provider(runtime_config(bindings=bindings)),
    ).run_once("load the helper and write")

    assert result.status == "completed"
    outputs = [obs.output for req in adapter.requests for obs in req.observations]
    assert any(
        isinstance(output, dict)
        and output.get("ok")
        and output.get("result", {}).get("path") == "created.md"
        for output in outputs
    )


def test_skill_tool_unknown_name(tmp_path: Path) -> None:
    _write_skill(tmp_path, "pdf-fill", description="d")
    provider = SkillProvider(load_skill_definitions(tmp_path))
    skill = _specs(provider)["skill"]

    result = skill.handler(None, {"name": "ghost"})

    assert not result.ok
    assert result.error_code == "skill_unknown"


def test_skill_tool_enum_constrains_names(tmp_path: Path) -> None:
    _write_skill(tmp_path, "pdf-fill", description="d")
    provider = SkillProvider(load_skill_definitions(tmp_path))
    skill = _specs(provider)["skill"]
    assert skill.input_schema["properties"]["name"]["enum"] == ["pdf-fill"]


# --- L3: skill.read_file (read a bundled resource) ---------------------------------


def test_read_file_returns_bundled_content(tmp_path: Path) -> None:
    _write_skill(tmp_path, "pdf-fill", resources={"references/FORMS.md": "FORMS_CONTENT"})
    provider = SkillProvider(load_skill_definitions(tmp_path))
    read = _specs(provider)["skill.read_file"]

    result = read.handler(None, {"name": "pdf-fill", "path": "references/FORMS.md"})

    assert result.ok
    assert result.content["content"] == "FORMS_CONTENT"


def test_read_file_rejects_traversal(tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("TOPSECRET", encoding="utf-8")
    _write_skill(tmp_path, "pdf-fill")
    provider = SkillProvider(load_skill_definitions(tmp_path))
    read = _specs(provider)["skill.read_file"]

    result = read.handler(None, {"name": "pdf-fill", "path": "../secret.txt"})

    assert not result.ok
    assert result.error_code == "skill_path_invalid"


def test_read_file_rejects_skill_md(tmp_path: Path) -> None:
    _write_skill(tmp_path, "pdf-fill")
    provider = SkillProvider(load_skill_definitions(tmp_path))
    read = _specs(provider)["skill.read_file"]

    result = read.handler(None, {"name": "pdf-fill", "path": SKILL_FILENAME})

    assert not result.ok
    assert result.error_code == "skill_path_invalid"


def test_read_file_missing_resource(tmp_path: Path) -> None:
    _write_skill(tmp_path, "pdf-fill")
    provider = SkillProvider(load_skill_definitions(tmp_path))
    read = _specs(provider)["skill.read_file"]

    result = read.handler(None, {"name": "pdf-fill", "path": "references/NOPE.md"})

    assert not result.ok
    assert result.error_code == "skill_resource_missing"


# --- bindings ----------------------------------------------------------------------


def test_tool_bindings_cover_all_tools(tmp_path: Path) -> None:
    _write_skill(tmp_path, "pdf-fill")
    provider = SkillProvider(load_skill_definitions(tmp_path))
    bound = {b.ref.tool_id for b in provider.tool_bindings()}
    assert bound == {"skill", "skill.read_file", "skill.run_script"}


# --- E2E: progressive disclosure through a real run --------------------------------


def test_e2e_model_loads_skill_then_reads_resource(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        "pdf-fill",
        description="Fill PDF forms",
        body="INSTRUCTIONS_BODY",
        resources={"references/FORMS.md": "FORMS_CONTENT"},
    )
    provider = SkillProvider(load_skill_definitions(skills_root))

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("skill", {"name": "pdf-fill"}, "c1"),)),
            ModelTurn(
                tool_calls=(
                    fake_tool_call(
                        "skill_read_file",
                        {"name": "pdf-fill", "path": "references/FORMS.md"},
                        "c2",
                    ),
                )
            ),
            ModelTurn(final_text="done"),
        ]
    )
    AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        context_providers=(provider,),
        tool_providers=(provider,),
        runtime_config_provider=runtime_provider(runtime_config(bindings=provider.tool_bindings())),
    ).run_once("go")

    # L1: the catalog is in the very first system prompt.
    assert "pdf-fill: Fill PDF forms" in adapter.requests[0].system_prompt
    # L2 + L3: the instructions and the on-demand resource both reached the model.
    outputs = json.dumps([obs.output for req in adapter.requests for obs in req.observations])
    assert "INSTRUCTIONS_BODY" in outputs
    assert "FORMS_CONTENT" in outputs


# --- P2: observability -------------------------------------------------------------


def _run_skill_activation(tmp_path: Path, *, event_sinks: tuple = ()):
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        "pdf-fill",
        description="Fill PDF forms",
        body="BODY",
        resources={"references/FORMS.md": "X"},
    )
    provider = SkillProvider(load_skill_definitions(skills_root))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("skill", {"name": "pdf-fill"}, "c1"),)),
            ModelTurn(final_text="done"),
        ]
    )
    return AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        context_providers=(provider,),
        tool_providers=(provider,),
        runtime_config_provider=runtime_provider(runtime_config(bindings=provider.tool_bindings())),
        event_sinks=event_sinks,
    ).run_once("go")


def test_skill_activation_emits_correlated_event_and_metrics(tmp_path: Path) -> None:
    sink = MemoryEventSink()
    result = _run_skill_activation(tmp_path, event_sinks=(sink,))

    by_type = {e.type: e for e in sink.events}
    assert "skill.activated" in by_type
    activated = by_type["skill.activated"]
    assert activated.data["name"] == "pdf-fill"
    assert activated.data["resource_count"] == 1

    # Correlated to the skill tool call (so an OTel sink can enrich that tool span).
    tool_starts = {e.event_id: e for e in sink.events if e.type == "tool.call.started"}
    assert activated.parent_id in tool_starts
    assert tool_starts[activated.parent_id].data.get("tool") == "skill"

    # Report-only run metrics.
    assert result.metrics["skill_activation_count"] == 1
    assert result.metrics["skills_activated"] == ["pdf-fill"]


def test_no_skill_metrics_when_none_activated(tmp_path: Path) -> None:
    _write_skill(tmp_path / "skills", "pdf-fill", description="d")
    provider = SkillProvider(load_skill_definitions(tmp_path / "skills"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        context_providers=(provider,),
        tool_providers=(provider,),
        runtime_config_provider=runtime_provider(runtime_config(bindings=provider.tool_bindings())),
    ).run_once("go")
    assert "skill_activation_count" not in result.metrics


# --- P3①: skill.run_script (execute a bundled script, output-only) -----------------


def _make_script_skill(tmp_path: Path, script_rel: str, source: str) -> SkillProvider:
    skills_root = tmp_path / "skills"
    _write_skill(skills_root, "pdf-fill", description="d", body="BODY", resources={script_rel: source})
    return SkillProvider(load_skill_definitions(skills_root))


def _run_script_via_loop(tmp_path: Path, provider: SkillProvider, call_args: dict):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("skill_run_script", call_args, "c1"),)),
            ModelTurn(final_text="done"),
        ]
    )
    bindings = (
        tool_binding("skill"),
        tool_binding("skill.read_file"),
        tool_binding("skill.run_script", runtime={"shell": {"approval_mode": "auto-approve"}}),
    )
    AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        context_providers=(provider,),
        tool_providers=(provider,),
        runtime_config_provider=runtime_provider(runtime_config(bindings=bindings)),
    ).run_once("go")
    return [obs.output for req in adapter.requests for obs in req.observations]


def test_run_script_executes_and_returns_output_only(tmp_path: Path) -> None:
    # The source carries a marker that must NOT reach context; only stdout should.
    source = "# SECRET_SOURCE_MARKER\nprint('HELLO_FROM_SCRIPT')\n"
    provider = _make_script_skill(tmp_path, "scripts/hello.py", source)

    outputs = _run_script_via_loop(tmp_path, provider, {"name": "pdf-fill", "path": "scripts/hello.py"})
    dumped = json.dumps(outputs)

    assert "HELLO_FROM_SCRIPT" in dumped  # stdout reached the model
    assert "SECRET_SOURCE_MARKER" not in dumped  # source never entered context
    # exit_code surfaced in the tool result content.
    run_obs = [o for o in outputs if isinstance(o, dict) and "result" in o and "exit_code" in o.get("result", {})]
    assert run_obs and run_obs[0]["result"]["exit_code"] == 0


def test_run_script_passes_args_literally_without_a_shell(tmp_path: Path) -> None:
    # Echo argv; a shell would interpret '; touch pwned' — argv execution keeps it literal.
    source = "import sys\nprint(sys.argv[1:])\n"
    provider = _make_script_skill(tmp_path, "scripts/echo.py", source)
    injection = "; touch pwned"

    outputs = _run_script_via_loop(
        tmp_path, provider, {"name": "pdf-fill", "path": "scripts/echo.py", "args": [injection, "two words"]}
    )
    dumped = json.dumps(outputs)

    # The injection string survives verbatim as a single argv element (never shell-parsed).
    assert injection in dumped
    assert "two words" in dumped
    assert not (tmp_path / "workspace" / "pwned").exists()


def test_run_script_unsupported_extension(tmp_path: Path) -> None:
    provider = _make_script_skill(tmp_path, "data/notes.txt", "just text")
    run = _specs(provider)["skill.run_script"]
    result = run.handler(_FakeRunContext(), {"name": "pdf-fill", "path": "data/notes.txt"})
    assert not result.ok
    assert result.error_code == "skill_script_unsupported"


def test_run_script_rejects_traversal(tmp_path: Path) -> None:
    (tmp_path / "outside.py").write_text("print('x')", encoding="utf-8")
    provider = _make_script_skill(tmp_path, "scripts/hello.py", "print('x')")
    run = _specs(provider)["skill.run_script"]
    result = run.handler(_FakeRunContext(), {"name": "pdf-fill", "path": "../../outside.py"})
    assert not result.ok
    assert result.error_code == "skill_path_invalid"


class _FakeRunContext:
    """Minimal context: enough for the run_script handler's pre-flight checks (it never
    reaches run_script for the unsupported/traversal cases, so run_script is unused)."""

    def run_script(self, args: dict) -> dict:  # pragma: no cover - not reached in these tests
        raise AssertionError("run_script should not be called for rejected inputs")


def test_run_script_side_effect_is_shell(tmp_path: Path) -> None:
    provider = _make_script_skill(tmp_path, "scripts/hello.py", "print('x')")
    assert _specs(provider)["skill.run_script"].side_effect == "shell"


# --- P3 fork: a skill that runs as a subagent -------------------------------------


def test_from_frontmatter_context_fork_vs_inline() -> None:
    assert SkillDefinition.from_frontmatter({"name": "x", "context": "fork"}, "b").context == "fork"
    assert SkillDefinition.from_frontmatter({"name": "y"}, "b").context == "inline"


def test_inline_skill_has_no_subagent_definition(tmp_path: Path) -> None:
    _write_skill(tmp_path, "pdf-fill", description="d")
    provider = SkillProvider(load_skill_definitions(tmp_path))
    assert provider.subagent_definitions() == {}


def test_fork_skill_synthesizes_fresh_subagent_with_allowlist(tmp_path: Path) -> None:
    _write_skill(
        tmp_path, "researcher", description="Research", context="fork",
        allowed_tools="fs.read shell.exec", body="You are a researcher.",
    )
    provider = SkillProvider(load_skill_definitions(tmp_path))

    defs = provider.subagent_definitions()
    assert set(defs) == {"skill:researcher"}
    sub = defs["skill:researcher"]
    assert sub.context == "fresh"  # fresh subagent whose persona is the skill body
    assert sub.tools == ("fs.read", "shell.exec")  # allowed_tools -> hard ceiling
    assert sub.prompt.persona_segments == ("You are a researcher.",)


def test_fork_skill_activation_runs_as_subagent(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root, "researcher", description="Research things", context="fork",
        body="CHILD_PERSONA — research the topic and report.",
    )
    provider = SkillProvider(load_skill_definitions(skills_root))
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class _ForkAdapter:
        def __init__(self) -> None:
            self.parent = [
                ModelTurn(tool_calls=(fake_tool_call("skill", {"name": "researcher", "task": "find X"}, "c1"),)),
                ModelTurn(final_text="parent done"),
            ]
            self.requests: list = []

        def next_turn(self, request):
            self.requests.append(request)
            if "CHILD_PERSONA" in request.system_prompt:  # the fork subagent's persona
                return ModelTurn(final_text="RESEARCH_RESULT")
            return self.parent.pop(0) if self.parent else ModelTurn(final_text="parent idle")

    adapter = _ForkAdapter()
    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        context_providers=(provider,),
        tool_providers=(provider,),
        subagent_definitions=provider.subagent_definitions(),
        runtime_config_provider=runtime_provider(runtime_config(bindings=provider.tool_bindings())),
    ).run_once("go")

    assert result.status == "completed"
    # The skill ran as an isolated subagent; only its final message came back to the parent.
    outputs = json.dumps([obs.output for req in adapter.requests for obs in req.observations])
    assert "RESEARCH_RESULT" in outputs
    # The child's persona was the skill instructions; the task was its user message.
    child_reqs = [r for r in adapter.requests if "CHILD_PERSONA" in r.system_prompt]
    assert child_reqs  # the fork subagent actually ran
    # Report-only subagent metrics reflect the delegated run.
    assert result.metrics.get("subagent_count") == 1


def test_otel_skill_tool_span_enriched(tmp_path: Path) -> None:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from monoid_agent_kernel.observability.otel import OtelEventSink

    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    _run_skill_activation(tmp_path, event_sinks=(OtelEventSink(tracer_provider=tracer_provider),))

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert "execute_tool skill" in spans
    tool_span = spans["execute_tool skill"]
    assert tool_span.attributes["skill.name"] == "pdf-fill"
    assert tool_span.attributes["skill.resource_count"] == 1
