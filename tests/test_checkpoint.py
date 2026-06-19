"""Durable-persistence serializers, checkpoint I/O, and snapshot/restore."""

from __future__ import annotations

import json
import time
from pathlib import Path

from conftest import runtime_config, runtime_provider

from native_agent_runner.core.checkpoint import (
    SCHEMA_VERSION,
    LocalFsCheckpointStore,
    RunCheckpoint,
    read_checkpoint,
    write_checkpoint,
)
from native_agent_runner.core.spec import AgentRunSpec, RunLimits
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelTurn, ToolObservation
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call
from native_agent_runner.shell import ShellExecutionOptions
from native_agent_runner.tasks import HostedTask


def _latest_checkpoint(spec: AgentRunSpec) -> RunCheckpoint | None:
    """Read the loop's last durably-committed checkpoint via the default store."""
    record = LocalFsCheckpointStore(spec.run_root).latest(spec.run_id)
    return record.checkpoint if record is not None else None


def _python_command(code: str) -> str:
    return f'python -c "{code.replace(chr(34), chr(92) + chr(34))}"'


def _hitl_parked_loop(spec: AgentRunSpec) -> tuple[AgentLoop, str, Path, Path]:
    """Open a loop, drive a turn that requests human input, and leave it parked on the
    hosted task (no close). Returns the loop, the parked task id, the run dir, and the
    artifacts dir — the setup shared by the restore tests."""
    provider = runtime_provider(runtime_config("hitl.request"))
    adapter = FakeModelAdapter(
        turns=[ModelTurn(response_id="r1", tool_calls=(fake_tool_call("hitl_request", {"prompt": "Pick"}, "c1"),))]
    )
    loop = AgentLoop(spec=spec, model_adapter=adapter, runtime_config_provider=provider)
    loop.open()
    suspension = loop.run_until_suspended("ask the human")
    assert suspension.reason == "awaiting_tasks"
    recorder = loop._session.res.recorder  # type: ignore[union-attr]
    return loop, suspension.awaiting_task_ids[0], recorder.run_dir, recorder.artifacts_dir


def test_tool_observation_round_trip() -> None:
    obs = ToolObservation(
        call_id="task_abc",
        tool_name="human_input",
        output={"type": "human_input_result", "answer": "Ada"},
        is_background=True,
    )
    assert ToolObservation.from_json(obs.to_json()) == obs


def test_hosted_task_checkpoint_round_trip(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    task = HostedTask(
        job_id="task_123",
        kind="automation",
        prompt="run the pipeline",
        status="running",
        started_at=100.0,
        resume_on_exit=True,
        job_path=artifacts / "tasks" / "task_123" / "task.json",
        cancel_path=artifacts / "tasks" / "task_123" / "cancel.requested",
        created_by="backend",
        choices=("a", "b"),
        request={"trigger": "nightly"},
        ready_for_reentry=False,
    )
    restored = HostedTask.from_checkpoint(task.checkpoint_json(), artifacts)

    assert restored.job_id == "task_123"
    assert restored.kind == "automation"
    assert restored.choices == ("a", "b")
    assert restored.request == {"trigger": "nightly"}
    assert restored.resume_on_exit is True
    # job_path/cancel_path are derived from artifacts_dir, matching HostedTaskExecutor.start.
    assert restored.job_path == artifacts / "tasks" / "task_123" / "task.json"
    assert restored.cancel_path == artifacts / "tasks" / "task_123" / "cancel.requested"


def test_run_checkpoint_round_trip_via_disk(tmp_path: Path) -> None:
    cp = RunCheckpoint(
        run_id="run_1",
        status="running",
        previous_turn_handle="turn_xyz",
        pending_observations=[{"call_id": "c1", "tool_name": "t", "output": {}, "is_background": False}],
        tool_call_counts={"fs.write": 2},
        total_tool_calls=2,
        total_usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        session_step=3,
        submit_local_step=1,
        terminal=False,
        hosted_tasks=[{"task_id": "task_1", "kind": "hitl"}],
        reentry_queue=["task_1"],
        delivered_reentry_jobs=[],
        remaining_duration_s=120.0,
        cancellation_requested=False,
        queued_messages=["next please"],
    )
    write_checkpoint(tmp_path, cp)
    restored = read_checkpoint(tmp_path)

    assert restored is not None
    assert restored == cp
    assert restored.schema_version == SCHEMA_VERSION


def test_read_checkpoint_missing_returns_none(tmp_path: Path) -> None:
    assert read_checkpoint(tmp_path) is None


def test_read_checkpoint_schema_mismatch_returns_none(tmp_path: Path) -> None:
    cp = RunCheckpoint(run_id="run_1")
    write_checkpoint(tmp_path, cp)
    # Corrupt the schema version on disk -> treated as no checkpoint, never raises.
    path = tmp_path / "checkpoint.json"
    path.write_text(path.read_text(encoding="utf-8").replace(SCHEMA_VERSION, "bogus.v0"), encoding="utf-8")
    assert read_checkpoint(tmp_path) is None


def test_local_fs_store_put_latest_seq(tmp_path: Path) -> None:
    store = LocalFsCheckpointStore(tmp_path)
    assert store.latest("run_1") is None

    store.put(RunCheckpoint(run_id="run_1", seq=1, previous_turn_handle="a"))
    store.put(RunCheckpoint(run_id="run_1", seq=2, previous_turn_handle="b"))
    record = store.latest("run_1")
    assert record is not None
    assert record.seq == 2
    assert record.checkpoint.previous_turn_handle == "b"

    # A second run is isolated; deleting run_1 leaves nothing to recover.
    store.put(RunCheckpoint(run_id="run_2", seq=1))
    store.delete("run_1")
    assert store.latest("run_1") is None
    assert store.latest("run_2") is not None


def test_local_fs_store_latest_ignores_uncommitted_seq(tmp_path: Path) -> None:
    # A manifest written for a higher seq WITHOUT flipping LATEST (a crash mid-commit)
    # must be ignored — latest() returns the last fully-committed checkpoint.
    store = LocalFsCheckpointStore(tmp_path)
    store.put(RunCheckpoint(run_id="run_1", seq=1, final_text="good"))
    seq2_dir = tmp_path / "run_1" / "checkpoints" / "2"
    seq2_dir.mkdir(parents=True)
    (seq2_dir / "manifest.json").write_text(
        json.dumps(RunCheckpoint(run_id="run_1", seq=2, final_text="half").to_json()),
        encoding="utf-8",
    )
    record = store.latest("run_1")
    assert record is not None and record.seq == 1 and record.checkpoint.final_text == "good"


def test_local_fs_store_put_flip_is_monotonic(tmp_path: Path) -> None:
    # A late writer with a lower seq (e.g. a reclaim racing a slow original worker) must
    # never regress LATEST and unpublish a newer committed checkpoint.
    store = LocalFsCheckpointStore(tmp_path)
    store.put(RunCheckpoint(run_id="run_1", seq=2, final_text="new"))
    store.put(RunCheckpoint(run_id="run_1", seq=1, final_text="stale"))
    record = store.latest("run_1")
    assert record is not None and record.seq == 2 and record.checkpoint.final_text == "new"


def test_local_fs_store_put_gcs_orphan_blob_tmp(tmp_path: Path) -> None:
    # A blob temp file left by a crashed prior write is dead weight (no LATEST flip ever
    # referenced it); the next put garbage-collects it.
    store = LocalFsCheckpointStore(tmp_path)
    blobs_dir = tmp_path / "run_1" / "checkpoints" / "blobs"
    blobs_dir.mkdir(parents=True)
    orphan = blobs_dir / ("a" * 64 + ".tmp")
    orphan.write_bytes(b"partial")
    store.put(RunCheckpoint(run_id="run_1", seq=1), blobs={"b" * 64: b"data"})
    assert not orphan.exists()
    assert (blobs_dir / ("b" * 64)).read_bytes() == b"data"


def test_snapshot_writes_checkpoint_at_hosted_park(tmp_path: Path) -> None:
    spec = AgentRunSpec(workspace_root=_mk(tmp_path / "ws"), run_root=tmp_path / "runs")
    loop, _task_id, _run_dir, _artifacts = _hitl_parked_loop(spec)

    # The persist hook committed a non-terminal checkpoint at the awaiting_tasks park.
    cp = _latest_checkpoint(spec)
    assert cp is not None
    assert cp.seq == 1
    assert cp.terminal is False
    assert cp.previous_turn_handle == "r1"
    assert len(cp.hosted_tasks) == 1
    # snapshot() is a pure read: calling it again yields an equal payload, save the
    # wall-clock fields (remaining_duration_s countdown, workspace_base.created_at).
    def _strip(payload: dict) -> dict:
        payload.pop("remaining_duration_s")
        payload.pop("workspace_base")
        return payload

    assert _strip(loop.snapshot().to_json()) == _strip(cp.to_json())  # type: ignore[union-attr]


def test_restore_resumes_parked_hitl_in_fresh_loop(tmp_path: Path) -> None:
    spec = AgentRunSpec(workspace_root=_mk(tmp_path / "ws"), run_root=tmp_path / "runs")
    loop1, task_id, _run_dir, _artifacts = _hitl_parked_loop(spec)
    cp = _latest_checkpoint(spec)
    assert cp is not None
    del loop1  # simulate process death WITHOUT close()

    # Fresh "process": brand-new loop + adapter over the same run dir.
    provider = runtime_provider(runtime_config("hitl.request"))
    adapter2 = FakeModelAdapter(turns=[ModelTurn(response_id="r2", final_text="thanks")])
    loop2 = AgentLoop(spec=spec, model_adapter=adapter2, runtime_config_provider=provider)
    loop2.restore(cp)

    # The hosted task is re-registered, so an external report wakes the parked run.
    loop2.report_task_result(task_id, {"answer": "Ada"})
    suspension = loop2.run_until_suspended(None)
    loop2.close()

    assert suspension.reason == "settled"
    assert suspension.turn is not None and suspension.turn.final_text == "thanks"
    # The hitl answer reached the model, and the conversation continued by reference
    # from the pre-restart turn handle (no transcript replay).
    hitl_obs = [
        obs for req in adapter2.requests for obs in req.observations if obs.tool_name == "human_input"
    ]
    assert hitl_obs and hitl_obs[0].output["answer"] == "Ada"
    assert adapter2.requests[0].previous_turn_handle == "r1"
    # A finalized run leaves no checkpoint behind.
    assert _latest_checkpoint(spec) is None


def test_restore_folds_crashed_shell_job_as_failed_observation(tmp_path: Path) -> None:
    spec = AgentRunSpec(workspace_root=_mk(tmp_path / "ws"), run_root=tmp_path / "runs")
    loop1, _task_id, _run_dir, artifacts = _hitl_parked_loop(spec)
    cp = _latest_checkpoint(spec)
    assert cp is not None
    del loop1

    # Plant a shell job left "running" on disk — its subprocess was lost on the crash.
    job_dir = artifacts / "jobs" / "job_dead"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps({"job_id": "job_dead", "status": "running", "command_preview": "sleep 999"}),
        encoding="utf-8",
    )

    provider = runtime_provider(runtime_config("hitl.request"))
    adapter2 = FakeModelAdapter(turns=[ModelTurn(response_id="r2", final_text="ok")])
    loop2 = AgentLoop(spec=spec, model_adapter=adapter2, runtime_config_provider=provider)
    loop2.restore(cp)

    pending = loop2._session.state.pending_observations  # type: ignore[union-attr]
    failed = [obs for obs in pending if obs.output.get("status") == "failed"]
    assert failed and failed[0].output["job_id"] == "job_dead"
    assert failed[0].output["error"] == "process lost on restart"
    loop2.close()


def test_snapshot_refuses_while_in_process_shell_job_runs(tmp_path: Path) -> None:
    spec = AgentRunSpec(
        workspace_root=_mk(tmp_path / "ws"),
        run_root=tmp_path / "runs",
        workspace_backend="staging",
    )
    provider = runtime_provider(runtime_config("fs.write"))
    loop = AgentLoop(
        spec=spec,
        model_adapter=FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="done")]),
        runtime_config_provider=provider,
    )
    loop.open()
    manager = loop._session.res.context.job_manager  # type: ignore[union-attr]
    # startup_wait_s=0 returns immediately with the job still running (no race against a
    # job that finishes during the startup wait); the long sleep outlives the snapshot check.
    job = manager.start_shell_job(
        shell_options=ShellExecutionOptions(enabled=True, approval_mode="auto-approve"),
        command=_python_command("import time; time.sleep(30)"),
        cwd=".",
        timeout_s=60,
        max_output_bytes=100_000,
        startup_wait_s=0,
        env={},
        requested_timeout_s=None,
        requested_max_output_bytes=None,
        requested_startup_wait_s=None,
        execution_workspace="direct",
        resume_on_exit=True,
    )
    # A live shell subprocess can't be restored, so the park is not durable yet.
    assert loop.snapshot() is None
    manager.cancel(job.job_id)  # stop the long-running job
    manager.wait(job.job_id)  # let the cancellation settle
    # Once only a finished (hosted-free) park remains, a snapshot is allowed.
    assert loop.snapshot() is not None
    loop.close()


def test_workspace_delta_round_trips_through_restore(tmp_path: Path) -> None:
    # Agent edits files; the checkpoint carries the delta; a fresh loop (over a
    # re-provisioned base) restores the exact same workspace state.
    base = _mk(tmp_path / "ws")
    (base / "keep.txt").write_text("original\n", encoding="utf-8")  # base file the agent deletes
    spec = AgentRunSpec(workspace_root=base, run_root=tmp_path / "runs")
    provider = runtime_provider(runtime_config("fs.write"))
    loop1 = AgentLoop(
        spec=spec,
        model_adapter=FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="done")]),
        runtime_config_provider=provider,
    )
    loop1.open()
    ws1 = loop1._session.res.workspace  # type: ignore[union-attr]
    ws1.write_bytes("new.txt", b"created\n", create_dirs=True)
    ws1.delete_path("keep.txt")
    expected = sorted(ws1.changed_paths())

    cp = loop1.snapshot()
    assert cp is not None
    blobs = loop1.collect_checkpoint_blobs()
    assert {entry["change_kind"] for entry in cp.workspace_delta} == {"created", "deleted"}
    del loop1

    # Fresh "process": re-provision the base (the deleted file is back), then restore.
    base2 = _mk(tmp_path / "ws2")
    (base2 / "keep.txt").write_text("original\n", encoding="utf-8")
    spec2 = AgentRunSpec(workspace_root=base2, run_root=tmp_path / "runs", run_id=spec.run_id)
    loop2 = AgentLoop(
        spec=spec2,
        model_adapter=FakeModelAdapter(turns=[ModelTurn(response_id="r2", final_text="ok")]),
        runtime_config_provider=runtime_provider(runtime_config("fs.write")),
    )
    loop2.restore(cp, blobs=blobs)
    ws2 = loop2._session.res.workspace  # type: ignore[union-attr]

    assert sorted(ws2.changed_paths()) == expected
    assert ws2.read_bytes("new.txt")[0] == b"created\n"
    assert not ws2.exists("keep.txt")
    loop2.close()


def test_by_value_conversation_accumulates_and_survives_restore(tmp_path: Path) -> None:
    # The core owns the conversation by value: each turn appends user/assistant messages,
    # the request carries the full log, and a restore continues it vendor-independently.
    spec = AgentRunSpec(workspace_root=_mk(tmp_path / "ws"), run_root=tmp_path / "runs")
    provider = runtime_provider(runtime_config("fs.write"))
    adapter1 = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="hi there")])
    loop1 = AgentLoop(spec=spec, model_adapter=adapter1, runtime_config_provider=provider)
    loop1.open()
    loop1.submit("hello")

    cp = loop1.snapshot()
    assert cp is not None
    logged = [(m["role"], m.get("content")) for m in cp.messages]
    assert ("user", "hello") in logged and ("assistant", "hi there") in logged
    # The request was by-value (full messages), not handle-only.
    assert adapter1.requests[0].messages is not None
    assert adapter1.requests[0].messages[0]["role"] == "user"

    # Fresh "process": restore and continue — the follow-up request carries the FULL
    # prior history plus the new user message, with no reliance on a vendor handle.
    adapter2 = FakeModelAdapter(turns=[ModelTurn(response_id="r2", final_text="bye")])
    loop2 = AgentLoop(spec=spec, model_adapter=adapter2, runtime_config_provider=provider)
    loop2.restore(cp)
    loop2.submit("more")

    sent = [(m["role"], m.get("content")) for m in adapter2.requests[0].messages]
    assert ("user", "hello") in sent
    assert ("assistant", "hi there") in sent
    assert ("user", "more") in sent
    loop2.close()


def test_failed_run_writes_failure_bundle_and_keeps_checkpoint(tmp_path: Path) -> None:
    class _RaisingAdapter:
        def next_turn(self, request):  # noqa: ANN001, ANN201
            from native_agent_runner.errors import ModelAdapterError

            raise ModelAdapterError("boom", error_code="provider_unavailable")

    spec = AgentRunSpec(workspace_root=_mk(tmp_path / "ws"), run_root=tmp_path / "runs")
    loop = AgentLoop(
        spec=spec,
        model_adapter=_RaisingAdapter(),
        runtime_config_provider=runtime_provider(runtime_config("fs.write")),
    )
    result = loop.run_once("do it")

    assert result.status == "failed"
    # The operator-facing failure bundle names the error and the restore handle.
    failure = json.loads((spec.run_root / spec.run_id / "failure.json").read_text(encoding="utf-8"))
    assert failure["type"] == "ModelAdapterError"
    assert failure["error_code"] and "restore" in failure["restore_hint"].lower()
    assert "last_good_seq" in failure
    # A failed run KEEPS its checkpoints (terminal -> the restart scanner still skips it).
    record = LocalFsCheckpointStore(spec.run_root).latest(spec.run_id)
    assert record is not None and record.checkpoint.terminal is True


def test_time_machine_restores_workspace_conversation_and_task_together(tmp_path: Path) -> None:
    # The capstone: one checkpoint, one restore — workspace edit, by-value conversation,
    # and a parked hosted task all return to the same instant in a fresh process over a
    # re-provisioned (empty) base, then the run resumes and settles vendor-independently.
    base = _mk(tmp_path / "ws")
    spec = AgentRunSpec(workspace_root=base, run_root=tmp_path / "runs")
    provider = runtime_provider(runtime_config("hitl.request"))
    adapter1 = FakeModelAdapter(
        turns=[ModelTurn(response_id="r1", tool_calls=(fake_tool_call("hitl_request", {"prompt": "ok?"}, "c1"),))]
    )
    loop1 = AgentLoop(spec=spec, model_adapter=adapter1, runtime_config_provider=provider)
    loop1.open()
    loop1._session.res.workspace.write_bytes("draft.md", b"v1\n", create_dirs=True)  # agent's file edit
    suspension = loop1.run_until_suspended("write a draft")
    assert suspension.reason == "awaiting_tasks"
    task_id = suspension.awaiting_task_ids[0]
    cp = loop1.snapshot()
    assert cp is not None
    blobs = loop1.collect_checkpoint_blobs()
    del loop1  # crash, no close

    # Fresh process over a re-provisioned base (empty — draft.md is only in the delta).
    base2 = _mk(tmp_path / "ws2")
    spec2 = AgentRunSpec(workspace_root=base2, run_root=tmp_path / "runs", run_id=spec.run_id)
    adapter2 = FakeModelAdapter(turns=[ModelTurn(response_id="r2", final_text="done")])
    loop2 = AgentLoop(spec=spec2, model_adapter=adapter2, runtime_config_provider=provider)
    loop2.restore(cp, blobs=blobs)

    # Workspace + conversation are both back at the checkpoint instant.
    assert loop2._session.res.workspace.read_bytes("draft.md")[0] == b"v1\n"
    assert any(m["role"] == "user" and m.get("content") == "write a draft" for m in loop2._session.state.messages)

    # The parked task resumes and the run settles, sending full by-value history.
    loop2.report_task_result(task_id, {"answer": "yes"})
    resumed = loop2.run_until_suspended(None)
    loop2.close()
    assert resumed.reason == "settled" and resumed.turn is not None and resumed.turn.final_text == "done"
    assert adapter2.requests[-1].messages is not None


def test_restore_carries_remaining_deadline(tmp_path: Path) -> None:
    spec = AgentRunSpec(
        workspace_root=_mk(tmp_path / "ws"),
        run_root=tmp_path / "runs",
        limits=RunLimits(max_duration_s=1000),
    )
    loop = AgentLoop(
        spec=spec,
        model_adapter=FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="done")]),
        runtime_config_provider=runtime_provider(runtime_config("fs.write")),
    )
    cp = RunCheckpoint(run_id=spec.run_id, status="completed", remaining_duration_s=300.0)
    loop.restore(cp)
    res = loop._session.res  # type: ignore[union-attr]
    # Downtime does not count against max_duration_s: the resumed deadline is ~now+remaining,
    # so the run is not immediately limited even though it was parked for a long time.
    assert 290.0 < (res.deadline - time.time()) < 300.5
    loop.close()


def _mk(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
