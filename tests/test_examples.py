"""Smoke tests for the programs under ``examples/``.

Examples are an integrator's first contact and the most likely thing to bit-rot when the
public API shifts, yet most were unguarded. Each test below loads an example by path
(mirroring ``test_scenario_scoring``) and exercises it in fake/offline mode — no LLM
gateway, no API key, no network — so a broken quickstart fails CI instead of a new adopter.

``messy_workspace_cleanup`` and ``full_stack_integration`` are reference-tier scenario
harnesses; the former is already covered by ``test_scenario_scoring`` and the latter has a
fake mode exercised here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load(rel: str):
    path = EXAMPLES_DIR / rel
    mod_name = "example_" + rel.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load example module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Self-contained offline programs: each ``main()`` uses an internal TemporaryDirectory and a
# scripted/echo adapter, so it neither touches the network nor pollutes the cwd.
@pytest.mark.parametrize(
    "rel",
    [
        "minimal_quickstart.py",
        "custom_model_adapter.py",
        "otel_tracing.py",
        "custom_tool_quickstart.py",
        "memory_quickstart.py",
    ],
)
def test_example_main_runs_offline(rel: str, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load(rel)
    module.main()
    out = capsys.readouterr().out
    assert "status" in out, f"{rel} produced no status line:\n{out}"


def test_custom_tools_example_builds_specs() -> None:
    module = _load("custom_tools/word_count_tool.py")

    # The @tool-decorated path: get_tools ignores its context arg and yields ToolSpecs.
    tools = module.get_tools(None)
    assert tools, "get_tools returned no tools"
    assert tools[0].id == "skill.word_count"

    # The equivalent hand-written ToolSpec builds and its handler actually counts words.
    handwritten = module._word_count_handwritten()
    assert handwritten.id == "skill.word_count"
    result = handwritten.handler(None, {"text": "alpha beta gamma"})
    assert result.ok and result.content == {"words": 3}


def test_redacting_event_sink_example() -> None:
    module = _load("redacting_event_sink.py")

    # The factory returns an EventSink-shaped object (emit/close).
    sink = module.make_sink()
    assert callable(getattr(sink, "emit", None))
    assert callable(getattr(sink, "close", None))

    # The redaction policy masks secret-looking keys and PEM bodies, leaves the rest.
    scrubbed = module._scrub({"api_key": "sk-123", "note": "hello", "blob": "-----BEGIN PRIVATE KEY-----"})
    assert scrubbed["api_key"] == module.REDACTED
    assert scrubbed["blob"] == module.REDACTED
    assert scrubbed["note"] == "hello"


def test_full_stack_integration_fake_scenario(tmp_path: Path) -> None:
    module = _load("full_stack_integration.py")
    result = module.run_scenario(mode="fake", model="gpt-5.5", reasoning_effort="low", root=tmp_path)
    # The whole reference stack (token manager → fake LLM gateway → Monoid backend) ran
    # end-to-end: a result exists, events were emitted, and nothing leaked.
    assert result["result_ready"] is True, result.get("status")
    assert result["event_types"], "no events emitted"
    assert result["secret_leak_detected"] is False
