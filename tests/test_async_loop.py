"""P1 async-native core: async entry points, native async adapters, sync facade guard.

Tests drive the async API with ``asyncio.run`` from plain sync test functions, so no
pytest-asyncio plugin is needed. The existing sync suite already covers the facade ->
async-core path transitively.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from conftest import runtime_config, runtime_provider, tool_binding

from native_agent_runner.core.spec import AgentRunSpec, RunLimits
from native_agent_runner.errors import NativeAgentError
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelRequest, ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call


def _spec(tmp_path: Path, *, mode: str = "apply", limits: RunLimits | None = None) -> AgentRunSpec:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    return AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        mode=mode,
        limits=limits or RunLimits(),
    )


def _loop(
    tmp_path: Path,
    adapter: object,
    *tool_ids: str,
    limits: RunLimits | None = None,
    mode: str = "apply",
) -> AgentLoop:
    return AgentLoop(
        spec=_spec(tmp_path, mode=mode, limits=limits),
        model_adapter=adapter,  # type: ignore[arg-type]
        runtime_config_provider=runtime_provider(runtime_config(*(tool_ids or ("fs.write",)))),
    )


def test_arun_once_writes_and_completes(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="1", tool_calls=(fake_tool_call("fs_write", {"path": "A.md", "content": "x"}, "c1"),)),
            ModelTurn(response_id="2", final_text="done"),
        ]
    )
    loop = _loop(tmp_path, adapter, "fs.write", "run.finish", limits=RunLimits(max_steps=4))
    result = asyncio.run(loop.arun_once("go"))

    assert result.status == "completed"
    assert result.final_text == "done"
    assert (tmp_path / "workspace" / "A.md").read_text() == "x"


def test_native_async_adapter_is_awaited(tmp_path: Path) -> None:
    class AsyncAdapter:
        supports_multimodal = False
        awaited = False

        def next_turn(self, request: ModelRequest) -> ModelTurn:  # pragma: no cover
            raise AssertionError("anext_turn should be preferred over next_turn")

        async def anext_turn(self, request: ModelRequest) -> ModelTurn:
            await asyncio.sleep(0)
            type(self).awaited = True
            return ModelTurn(response_id="a", final_text="native-async")

    loop = _loop(tmp_path, AsyncAdapter())
    result = asyncio.run(loop.arun_once("go"))

    assert result.status == "completed"
    assert result.final_text == "native-async"
    assert AsyncAdapter.awaited is True


def test_sync_api_inside_running_loop_raises(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    loop = _loop(tmp_path, adapter)

    async def call_sync_in_loop() -> object:
        try:
            loop.run_once("go")
        except NativeAgentError as exc:
            return exc.error_code
        return None

    assert asyncio.run(call_sync_in_loop()) == "sync_in_async_loop"


def test_async_and_sync_paths_agree(tmp_path: Path) -> None:
    def make_adapter() -> FakeModelAdapter:
        return FakeModelAdapter(turns=[ModelTurn(response_id="1", final_text="settled")])

    sync_loop = _loop(tmp_path / "s", make_adapter())
    async_loop = _loop(tmp_path / "a", make_adapter())

    sync_result = sync_loop.run_once("go")
    async_result = asyncio.run(async_loop.arun_once("go"))

    assert sync_result.status == async_result.status == "completed"
    assert sync_result.final_text == async_result.final_text == "settled"


def _python_command(code: str) -> str:
    escaped = code.replace('"', '\\"')
    return f'python -c "{escaped}"'


def test_background_shell_job_completes_on_run_loop_and_reenters(tmp_path: Path) -> None:
    # B2: a background (resume_on_exit) shell job runs its asyncio subprocess monitor on the
    # run's always-on loop. The run parks, the monitor completes the subprocess on that loop
    # (no per-job thread, no 20ms poll), and the result reenters as an observation so the
    # next turn settles.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call(
                        "shell_exec",
                        {"command": _python_command("print('bg-done')"), "background": True},
                        "c1",
                    ),
                ),
            ),
            # The job is still running here, so this turn parks (awaiting_tasks) rather than
            # settling; its final_text is discarded.
            ModelTurn(response_id="r2", final_text="still running"),
            # Reentry turn: the job result has been delivered, so this settles.
            ModelTurn(response_id="r3", final_text="done"),
        ]
    )
    config = runtime_config(
        bindings=(
            tool_binding(
                "shell.exec",
                runtime={"shell": {"approval_mode": "auto-approve", "default_timeout_s": 30}},
            ),
            tool_binding("run.finish"),
        )
    )
    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
    ).run_once("Run a background job.")

    assert result.status == "completed"
    assert result.final_text == "done"
    # The run parked and resumed: the model saw a later turn carrying the job result.
    assert len(adapter.requests) >= 3
    job_results = [
        obs
        for request in adapter.requests
        for obs in request.observations
        if obs.output.get("type") == "background_job_result"
    ]
    assert job_results and job_results[0].output["status"] == "exited"
    events = result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8")
    assert "job.started" in events and "job.finished" in events


def test_concurrent_runs_interleave_on_one_loop(tmp_path: Path) -> None:
    class SlowAsyncAdapter:
        supports_multimodal = False

        def next_turn(self, request: ModelRequest) -> ModelTurn:  # pragma: no cover
            raise AssertionError("anext_turn should be used")

        async def anext_turn(self, request: ModelRequest) -> ModelTurn:
            await asyncio.sleep(0.05)
            return ModelTurn(response_id="x", final_text="ok")

    async def run_both() -> list[str]:
        a = _loop(tmp_path / "a", SlowAsyncAdapter())
        b = _loop(tmp_path / "b", SlowAsyncAdapter())
        results = await asyncio.gather(a.arun_once("go"), b.arun_once("go"))
        return [r.status for r in results]

    assert asyncio.run(run_both()) == ["completed", "completed"]
