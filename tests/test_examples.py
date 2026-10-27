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

import ast
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_local_product_embedding_golden_path(tmp_path: Path) -> None:
    module = _load("embedding_local_product.py")
    result = module.run_local_product(tmp_path)

    assert result["status"] == "completed"
    assert result["runtime_profile"] == "embedded-local"
    assert result["output_exists"] is True
    assert result["checkpoint_load_status"] == "loaded"
    assert result["checkpoint_seq"] >= 1
    assert result["event_count"] > 0
    assert result["network_required"] is False


@pytest.mark.parametrize("failure", ["adapter", "checkpoint"])
def test_local_product_failure_discards_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    from monoid_agent_kernel.errors import ModelAdapterError, TurnNotSettled

    module = _load("embedding_local_product.py")
    original_session = module.LoopSession
    original_checked = module.LocalFsCheckpointStore.latest_checked
    original_discard = module.AgentLoop.discard_uncommitted
    sessions = []
    read_error = OSError("checkpoint read failed")

    def observe_session(loop):
        session = original_session(loop)
        sessions.append(session)
        return session

    def fail_turn(self, request):
        raise ModelAdapterError("bad config", http_status=400, error_code="model_error")

    def fail_checked(self, run_id):
        raise read_error

    def discard_then_fail(loop):
        original_discard(loop)
        raise RuntimeError("cleanup failed after releasing the activation")

    monkeypatch.setattr(module, "LoopSession", observe_session)
    monkeypatch.setattr(module.AgentLoop, "discard_uncommitted", discard_then_fail)
    if failure == "adapter":
        monkeypatch.setattr(module.FakeModelAdapter, "next_turn", fail_turn)
    else:
        monkeypatch.setattr(module.LocalFsCheckpointStore, "latest_checked", fail_checked)
    try:
        with pytest.raises(TurnNotSettled if failure == "adapter" else OSError) as raised:
            module.run_local_product(tmp_path)
        if failure == "adapter":
            assert raised.value.suspension.reason == "turn_failed"
        else:
            assert raised.value is read_error
        session, = sessions
        assert session.health().alive is False
        assert session.health().can_accept_input is False
        assert session.inspect().terminal is True
        checkpoint = original_checked(
            module.LocalFsCheckpointStore(tmp_path / "runs"), session.loop.spec.run_id,
        )
        assert checkpoint.ok and checkpoint.value is not None
    finally:
        for session in sessions:
            original_discard(session.loop)


@pytest.mark.parametrize("rel", ["full_stack_integration.py", "messy_workspace_cleanup.py"])
@pytest.mark.parametrize("caller_pythonpath", [None, "caller-imports"])
def test_real_gateway_launcher_preserves_python_environment(
    rel: str, caller_pythonpath: str | None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(rel)
    launches = []

    def capture_launch(command, **kwargs):
        launches.append((command, kwargs))
        return SimpleNamespace(poll=lambda: 0)

    if caller_pythonpath is None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", caller_pythonpath)
    monkeypatch.setattr(module, "_free_port", lambda: 40123)
    monkeypatch.setattr(module, "_wait_for_health", lambda *args: None)
    monkeypatch.setattr(module.subprocess, "Popen", capture_launch)
    _, stop, _ = module._start_real_gateway_subprocess(
        token_secret="example-token", admin_token="example-admin", openai_api_key="example-key",
    )
    stop()
    (command, options), = launches
    assert command[:5] == [
        sys.executable, "-m", "monoid_agent_kernel.cli", "llm-gateway", "serve",
    ]
    actual_pythonpath = options["env"].get("PYTHONPATH")
    assert actual_pythonpath == caller_pythonpath


@pytest.mark.parametrize("rel", ["model_call_facts.py", "recover_model_call_facts.py"])
def test_model_call_fact_examples_preserve_sparse_observations(
    rel: str, capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load(rel)
    asyncio.run(module.main())
    output = json.loads(capsys.readouterr().out)
    assert output["known_usage"] == {"input_tokens": 0, "output_tokens": 3}
    assert output["total_coverage"] == "unknown"
    if rel == "model_call_facts.py":
        assert output["input_coverage"] == "complete"
        assert output["termination"] == {
            "outcome": "completed", "requested_by": None, "ended_by": "provider_terminal",
        }
    else:
        from monoid_agent_kernel.core.model_invocation import MODEL_INVOCATION_V2_SCHEMA_VERSION

        assert output["schema"] == MODEL_INVOCATION_V2_SCHEMA_VERSION
        assert output["provider_calls"] == 1


def test_termination_budget_example_observes_stop_and_settlement(
    capsys: pytest.CaptureFixture[str],
) -> None:
    asyncio.run(_load("termination_budget.py").main())
    output = json.loads(capsys.readouterr().out)
    assert output["during_exit"]["stream_exit"]["phase"] == "cooperative_wait"
    assert output["after_exit"]["stream_exit"]["phase"] == "settled"
    assert output["termination"]["requested_by"] == "user_cancel"
    assert output["termination"]["ended_by"] == "provider_terminal"
    assert output["usage"] == {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}


def test_host_report_example_preserves_approval_and_child_paths(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from monoid_agent_kernel.reference.host_tool_report import main

    main()
    output = json.loads(capsys.readouterr().out)
    assert output["approval"] == {
        "status": "completed", "activations": 2, "approved_read": True, "proposal_only": True,
    }
    assert output["delegation"] == [
        {"status": "completed", "context": mode, "activations": 2, "proposal_only": True}
        for mode in ("fresh", "fork")
    ]


def test_hosted_product_embedding_golden_path(tmp_path: Path) -> None:
    module = _load("embedding_hosted_product.py")
    result = module.run_hosted_product(tmp_path)

    assert result["status"] == "ok"
    assert result["runtime_profile"] == "reference-inbox"
    assert result["tenants"] == ["tenant_a", "tenant_b"]
    assert result["cross_tenant_access_denied"] is True
    assert set(result["checked_loads"]) == set(result["run_ids"])
    assert all(
        load == {"checkpoint": "loaded", "run_metadata": "loaded"}
        for load in result["checked_loads"].values()
    )
    assert result["initial_event_count"] > 0
    assert result["replay_count"] == 0
    assert result["status_receipt"] == "completed"
    assert result["status_result"]["type"] == "status"
    assert result["status_result"]["status"] == "ok"
    assert result["status_result"]["data"]["state"] == "awaiting_input"
    assert result["status_duplicate_matches"] is True
    assert result["task_receipt"] == "completed"
    assert result["task_result"]["type"] == "report_task_result"
    assert result["task_result"]["status"] == "ok"
    assert result["approval_receipt"] == "completed"
    assert result["approval_result"]["type"] == "approve"
    assert result["approval_result"]["status"] == "ok"
    assert result["historical_event_count"] >= result["initial_event_count"]
    assert {
        tenant: (usage["tenant_id"], usage["runs"])
        for tenant, usage in result["tenant_usage"].items()
    } == {"tenant_a": ("tenant_a", 1), "tenant_b": ("tenant_b", 1)}
    assert result["observed_gateway_token_count"] == 2
    assert result["credential_leak_detected"] is False
    assert result["network_required"] is False


def test_hosted_embedding_uses_reference_facades_without_dbos() -> None:
    tree = ast.parse((EXAMPLES_DIR / "embedding_hosted_product.py").read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)

    assert "monoid_agent_kernel.reference.backend" in modules
    assert "monoid_agent_kernel.reference.stores" in modules
    assert not any(
        module == "monoid_agent_kernel.reference.dbos"
        or module.startswith("monoid_agent_kernel.reference.dbos.")
        for module in modules
    )
    assert not any(
        module.startswith(
            (
                "monoid_agent_kernel.reference._",
                "monoid_agent_kernel.reference.backend.service",
                "monoid_agent_kernel.reference.command_inbox",
                "monoid_agent_kernel.reference.stores.sqlite",
            )
        )
        for module in modules
    )
