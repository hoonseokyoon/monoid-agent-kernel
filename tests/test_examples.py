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
import os
import subprocess
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
    assert output["delivered_text"] == ["Working"]
    assert output["known_usage"] == {"input_tokens": 0, "output_tokens": 3}
    assert output["total_coverage"] == "unknown"
    assert output["receipt_count"] == 1


def test_termination_example_stops_before_the_first_delta() -> None:
    module = _load("termination_budget.py")

    async def check():
        existing = asyncio.all_tasks()
        output = await module.run_request("silent_stop")
        assert output["delivered_text"] == []
        assert output["known_usage"] == {"input_tokens": 0, "output_tokens": 3}
        assert output["total_coverage"] == "unknown"
        assert output["receipt_count"] == 1
        assert output["termination"]["requested_by"] == "user_cancel"
        assert output["termination"]["ended_by"] == "provider_terminal"
        assert output["after_exit"]["stream_exit"]["phase"] == "settled"
        assert not {item for item in asyncio.all_tasks() - existing if not item.done()}

    asyncio.run(check())


@pytest.mark.parametrize("scenario, close_fails", [
    ("consumer_cancel", False), ("consumer_cancel", True), ("silent_stop", True),
])
def test_termination_example_preserves_request_cancellation_and_cleanup_errors(
    monkeypatch: pytest.MonkeyPatch, scenario: str, close_fails: bool,
) -> None:
    from monoid_agent_kernel import CapturePolicy, LoopSession, ModelIOSubscription

    module = _load("termination_budget.py")
    original_from_config = module.AgentLoop.from_config
    loops, receipts, closed = [], [], []
    cleanup_error = RuntimeError("close failed after releasing the activation")

    class Observer:
        def on_model_call(self, capture):
            receipts.append(capture.receipt)

        def close(self):
            closed.append(True)

    def build_loop(spec, adapter, config, **kwargs):
        kwargs["model_io_subscriptions"] += (
            ModelIOSubscription(Observer(), CapturePolicy(mode="none")),
        )
        loop = original_from_config(spec, adapter, config, **kwargs)
        original_close = loop.aclose

        async def close():
            result = await original_close()
            if close_fails:
                raise cleanup_error
            return result

        monkeypatch.setattr(loop, "aclose", close)
        loops.append(loop)
        return loop

    monkeypatch.setattr(module.AgentLoop, "from_config", build_loop)

    async def check():
        existing = asyncio.all_tasks()
        task = asyncio.create_task(module.run_request(scenario))
        try:
            done, _ = await asyncio.wait({task}, timeout=5)
            assert task in done, "request cleanup did not finish"
            expected = asyncio.CancelledError if scenario == "consumer_cancel" else RuntimeError
            with pytest.raises(expected) as raised:
                await task
            if scenario == "consumer_cancel":
                assert raised.value.args == ("request disconnected",)
            else:
                assert raised.value is cleanup_error
            receipt, = receipts
            assert receipt.termination.requested_by == "user_cancel"
            assert receipt.termination.ended_by == "provider_terminal"
            assert dict(receipt.usage_evidence.reported) == {"input_tokens": 0, "output_tokens": 3}
            assert closed == [True]
            loop, = loops
            assert LoopSession(loop).health().alive is False
            assert not {item for item in asyncio.all_tasks() - existing if not item.done()}
        finally:
            pending = [item for item in asyncio.all_tasks() - existing if not item.done()]
            for item in pending:
                item.cancel()
            await asyncio.gather(task, *pending, return_exceptions=True)
            for loop in loops:
                await loop.adiscard_uncommitted()

    asyncio.run(check())


def test_termination_example_closes_after_a_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from monoid_agent_kernel import CapturePolicy, LoopSession, ModelIOSubscription
    from monoid_agent_kernel.errors import ModelAdapterError, TurnNotSettled

    module = _load("termination_budget.py")
    original_from_config = module.AgentLoop.from_config
    loops, receipts, closed = [], [], []

    class FailingAdapter:
        async def astream_turn(self, request):
            raise ModelAdapterError("bad configuration", http_status=400)
            yield  # Make the failing method an async generator, like a real stream adapter.

    class Observer:
        def on_model_call(self, capture):
            receipts.append(capture.receipt)

        def close(self):
            closed.append(True)

    def build_loop(spec, adapter, config, **kwargs):
        kwargs["model_io_subscriptions"] += (
            ModelIOSubscription(Observer(), CapturePolicy(mode="none")),
        )
        loop = original_from_config(spec, FailingAdapter(), config, **kwargs)
        loops.append(loop)
        return loop

    monkeypatch.setattr(module.AgentLoop, "from_config", build_loop)

    async def check():
        existing = asyncio.all_tasks()
        task = asyncio.create_task(module.main())
        try:
            # Observe the old watcher hang before timeout cancellation can hide its cause.
            done, _ = await asyncio.wait({task}, timeout=3)
            assert task in done, "example waited for a Stop phase after the provider had failed"
            with pytest.raises(TurnNotSettled) as raised:
                await task
            assert raised.value.suspension.reason == "turn_failed"
            assert raised.value.suspension.http_status == 400
            assert "bad configuration" in raised.value.suspension.error
            receipt, = receipts
            assert receipt.termination.to_json() == {
                "outcome": "failed", "requested_by": None, "ended_by": "provider_error",
            }
            assert dict(receipt.usage_evidence.reported) == {}
            assert receipt.usage_evidence.completeness["total_tokens"] == "unknown"
            assert closed == [True]
            loop, = loops
            assert LoopSession(loop).health().alive is False
            assert not {item for item in asyncio.all_tasks() - existing if not item.done()}
        finally:
            # This fallback keeps a deliberately red test from leaking its real activation.
            pending = [item for item in asyncio.all_tasks() - existing if not item.done()]
            for item in pending:
                item.cancel()
            await asyncio.gather(task, *pending, return_exceptions=True)
            for loop in loops:
                await loop.adiscard_uncommitted()

    asyncio.run(check())


@pytest.mark.parametrize("fail_observation", [False, True])
def test_background_example_owns_run_cleanup_and_passive_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend_factory, fail_observation: bool,
) -> None:
    module = _load("observe_background_run.py")
    original_backend = module.RunnerBackend
    backends, submissions, readers = [], [], []
    observation_error = RuntimeError("reader failed while the run was active")

    def build_backend(**kwargs):
        backend = backend_factory.track(original_backend(**kwargs))
        backends.append(backend)
        original_submit, original_subscribe = backend.submit_run, backend.subscribe_model_stream

        def submit(request):
            submission = original_submit(request)
            submissions.append(submission)
            return submission

        def subscribe(*args, **kwargs):
            reader = original_subscribe(*args, **kwargs)
            readers.append(reader)
            if fail_observation and len(readers) == 2:
                original_poll = reader.poll

                def fail_poll(**options):
                    original_poll(**options)
                    raise observation_error

                monkeypatch.setattr(reader, "poll", fail_poll)
            return reader

        monkeypatch.setattr(backend, "submit_run", submit)
        monkeypatch.setattr(backend, "subscribe_model_stream", subscribe)
        return backend

    monkeypatch.setattr(module, "RunnerBackend", build_backend)
    if fail_observation:
        with pytest.raises(RuntimeError) as raised:
            module.run_example(tmp_path)
        assert raised.value is observation_error
    else:
        assert module.run_example(tmp_path) == {
            "state": "cancelled", "reader_closed": True, "continued_after_disconnect": True,
            "peer_text": "B", "replayed_text": "B", "reset_reason": "cursor_gap",
            "history_complete": False, "retained_text": "CD", "cancel_requested": True,
        }
    # Assert the example's own completion before the managed fixture joins its Futures.
    backend, = backends
    submission, = submissions
    assert backend.result(submission.run_id, submission.run_token)["ready"] is True
    assert readers and all(reader.closed for reader in readers)


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


def test_host_report_cli_restores_approval_in_a_fresh_process(tmp_path: Path) -> None:
    from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore

    root = tmp_path / "report"
    env = os.environ.copy()
    # conftest's sys.path affects only pytest; every CLI child must use this source revision.
    env["PYTHONPATH"] = str((EXAMPLES_DIR.parent / "src").resolve())

    def invoke(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "monoid_agent_kernel.reference.host_tool_report", *args],
            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30, check=False,
        )

    preparation = invoke("--phase", "prepare", "--root", str(root))
    assert preparation.returncode == 0, preparation.stderr
    prepared = json.loads(preparation.stdout)
    assert prepared.get("status") == "awaiting_tasks", prepared
    assert isinstance(prepared["run_id"], str) and prepared["run_id"]
    assert isinstance(prepared["task_id"], str) and prepared["task_id"]
    assert isinstance(prepared["pid"], int) and prepared["pid"] > 0
    assert prepared["pid"] != os.getpid()
    assert prepared["resource_opens"] == 1
    assert prepared["read_calls"] == 0
    assert prepared["model_calls"] == 3
    assert prepared["total_usage"] == {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
    assert prepared["proposal_only"] is True
    report_path = root / "workspace" / "reports" / "weekly.md"
    assert not report_path.exists()

    store = LocalFsCheckpointStore(root / "runs")
    checked = store.latest_checked(prepared["run_id"])
    assert checked.ok and checked.value is not None
    record = checked.value
    before_refusal = record.checkpoint.to_json()
    assert record.checkpoint.run_id == prepared["run_id"]
    assert record.checkpoint.total_usage == prepared["total_usage"]
    assert record.checkpoint.last_suspension["reason"] == "awaiting_tasks"
    assert record.checkpoint.last_suspension["awaiting_task_ids"] == [prepared["task_id"]]

    refused = invoke(
        "--phase", "resume", "--root", str(root), "--run-id", prepared["run_id"],
        "--task-id", f"{prepared['task_id']}-wrong",
    )
    assert refused.returncode != 0, refused.stdout
    assert "task is not awaiting approval" in refused.stderr
    retained = store.latest_checked(prepared["run_id"])
    assert retained.ok and retained.value is not None
    assert retained.value.seq == record.seq
    assert retained.value.checkpoint.to_json() == before_refusal
    assert not report_path.exists()

    completion = invoke(
        "--phase", "resume", "--root", str(root), "--run-id", prepared["run_id"],
        "--task-id", prepared["task_id"],
    )
    assert completion.returncode == 0, completion.stderr
    resumed = json.loads(completion.stdout)
    assert resumed["status"] == "completed"
    assert resumed["run_id"] == prepared["run_id"]
    assert resumed["task_id"] == prepared["task_id"]
    assert isinstance(resumed["pid"], int) and resumed["pid"] > 0
    assert resumed["pid"] not in {prepared["pid"], os.getpid()}
    assert resumed["resource_opens"] == 1
    assert resumed["read_calls"] == 1
    assert resumed["model_calls"] == 1
    assert resumed["total_usage"] == {"input_tokens": 9, "output_tokens": 6, "total_tokens": 15}
    assert resumed["duplicate_ignored"] is True
    assert resumed["report_body"] == "# Weekly\n\nReady for approval.\n"
    assert resumed["proposal_only"] is True
    assert not report_path.exists()


def test_host_report_restore_failure_preserves_error_and_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from monoid_agent_kernel import LoopSession
    from monoid_agent_kernel.core.checkpoint import CheckpointRecord, LocalFsCheckpointStore
    from monoid_agent_kernel.reference import host_tool_report as module

    prepared = module.prepare_report(tmp_path)
    store = LocalFsCheckpointStore(tmp_path / "runs")
    before = store.latest_checked(prepared["run_id"])
    assert before.ok and before.value is not None
    original_from_config = module.AgentLoop.from_config
    original_discard = module.AgentLoop.discard_uncommitted
    loops = []
    blob_error = OSError("saved report blob unavailable")

    def observe_loop(*args, **kwargs):
        loop = original_from_config(*args, **kwargs)
        loops.append(loop)
        return loop

    def fail_blob(self, sha256):
        raise blob_error

    def discard_then_fail(loop):
        original_discard(loop)
        raise RuntimeError("cleanup failed after releasing the activation")

    monkeypatch.setattr(module.AgentLoop, "from_config", observe_loop)
    monkeypatch.setattr(CheckpointRecord, "blob", fail_blob)
    monkeypatch.setattr(module.AgentLoop, "discard_uncommitted", discard_then_fail)
    try:
        with pytest.raises(OSError) as raised:
            module.resume_report(tmp_path, prepared["run_id"], prepared["task_id"])
        assert raised.value is blob_error
        loop, = loops
        assert LoopSession(loop).health().alive is False
        after = store.latest_checked(prepared["run_id"])
        assert after.ok and after.value is not None
        assert after.value.seq == before.value.seq
        assert after.value.checkpoint.to_json() == before.value.checkpoint.to_json()
    finally:
        for loop in loops:
            original_discard(loop)


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
