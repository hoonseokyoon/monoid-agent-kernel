from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore

pytest.importorskip("dbos")
pytestmark = pytest.mark.slow


def _worker_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(root / "src"), str(root / "tests"), env.get("PYTHONPATH", "")))
    )
    return env


def _worker_args(
    root: Path,
    mode: str,
    *,
    db_path: Path,
    run_root: Path,
    workspace: Path,
    effect_db: Path,
    output: Path | None = None,
    started: Path | None = None,
    fault_phase: str | None = None,
) -> list[str]:
    args = [
        sys.executable,
        "-m",
        "support.dbos_run_worker",
        mode,
        "--db",
        str(db_path),
        "--run-root",
        str(run_root),
        "--workspace",
        str(workspace),
        "--effect-db",
        str(effect_db),
    ]
    if output is not None:
        args.extend(("--output", str(output)))
    if started is not None:
        args.extend(("--started", str(started)))
    if fault_phase is not None:
        args.extend(("--fault-phase", fault_phase))
    return args


def _kill_and_reap(process: subprocess.Popen[bytes]) -> int:
    returncode: int | None = None
    try:
        if process.poll() is None:
            process.kill()
        returncode = process.wait(timeout=10)
    finally:
        try:
            if process.poll() is None:
                process.kill()
        finally:
            returncode = process.wait(timeout=10)
    assert returncode is not None
    return returncode


@pytest.mark.parametrize(
    (
        "crash_mode",
        "recover_mode",
        "crash_driver_mode",
        "recovery_driver_mode",
    ),
    (
        pytest.param(
            "crash",
            "recover",
            "standalone",
            "standalone",
            id="standalone-to-standalone",
        ),
        pytest.param(
            "hosted-crash",
            "hosted-recover",
            "hosted",
            "hosted",
            id="hosted-to-hosted",
        ),
        pytest.param(
            "crash",
            "hosted-recover",
            "standalone",
            "hosted",
            id="standalone-to-hosted",
        ),
    ),
)
@pytest.mark.parametrize("fault_phase", ("effect_committed", "boundary_committed"))
def test_dbos_run_resume_survives_kill_with_one_effect_and_one_workflow_result(
    tmp_path: Path,
    crash_mode: str,
    recover_mode: str,
    crash_driver_mode: str,
    recovery_driver_mode: str,
    fault_phase: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    db_path = tmp_path / "dbos.sqlite"
    run_root = tmp_path / "runs"
    workspace = tmp_path / "workspace"
    effect_db = tmp_path / "effects.sqlite"
    seed_output = tmp_path / "seed.json"
    transition = f"{crash_driver_mode}-to-{recovery_driver_mode}"
    started_path = tmp_path / f"{transition}-{fault_phase}.json"
    recovery_output = tmp_path / f"{transition}-{fault_phase}-recovered.json"
    crash_log = tmp_path / f"{transition}-{fault_phase}-crash.log"
    env = _worker_env(root)

    seeded = subprocess.run(
        _worker_args(
            root,
            "seed",
            db_path=db_path,
            run_root=run_root,
            workspace=workspace,
            effect_db=effect_db,
            output=seed_output,
        ),
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert seeded.returncode == 0, seeded.stderr or seeded.stdout
    seed_result = json.loads(seed_output.read_text(encoding="utf-8"))
    assert seed_result["checkpoint_seq"] == 1
    command_marker = seed_result["checkpoint_marker"]

    crashed = None
    try:
        with crash_log.open("w", encoding="utf-8") as crash_stream:
            crashed = subprocess.Popen(
                _worker_args(
                    root,
                    crash_mode,
                    db_path=db_path,
                    run_root=run_root,
                    workspace=workspace,
                    effect_db=effect_db,
                    started=started_path,
                    fault_phase=fault_phase,
                ),
                cwd=root,
                env=env,
                stdout=crash_stream,
                stderr=subprocess.STDOUT,
            )
        expected_marker = {
            "command_id": "resume_once",
            "fault_phase": fault_phase,
            "pid": crashed.pid,
            "run_id": "run_dbos_resume",
            "runtime_mode": crash_driver_mode,
            "workflow_id": "monoid/run/run_dbos_resume/resume/resume_once",
        }
        marker: dict[str, object] | None = None
        crash_failure = ""
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and crashed.poll() is None:
            try:
                if not started_path.exists():
                    time.sleep(0.02)
                    continue
                marker = json.loads(started_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
            else:
                break
            time.sleep(0.02)
        if marker is None:
            crash_failure = f"DBOS worker exited before reaching {expected_marker!r}"
        elif marker != expected_marker:
            crash_failure = f"DBOS worker wrote {marker!r}; expected {expected_marker!r}"
        elif crashed.poll() is not None:
            crash_failure = "DBOS worker exited after writing the crash marker"
    finally:
        if crashed is not None:
            crash_returncode = _kill_and_reap(crashed)
    if crash_failure:
        diagnostics = crash_log.read_text(encoding="utf-8", errors="replace")
        pytest.fail(f"{crash_failure}\n{diagnostics}")
    assert crash_returncode != 0

    workflow_id = str(expected_marker["workflow_id"])
    with sqlite3.connect(db_path) as connection:
        workflow_statuses = connection.execute(
            "SELECT status FROM workflow_status WHERE workflow_uuid = ?",
            (workflow_id,),
        ).fetchall()
    with sqlite3.connect(effect_db) as connection:
        effect_count_before_recovery = connection.execute(
            "SELECT COUNT(*) FROM semantic_effects WHERE run_id = ? AND effect_key = ?",
            ("run_dbos_resume", "resume_once"),
        ).fetchone()[0]
    checkpoint_before_recovery = LocalFsCheckpointStore(run_root).latest(
        "run_dbos_resume"
    )
    assert checkpoint_before_recovery is not None
    expected_checkpoint_state = {
        "effect_committed": (1, 0),
        "boundary_committed": (2, 1),
    }
    expected_seq, expected_marker_count = expected_checkpoint_state[fault_phase]
    assert workflow_statuses == [("PENDING",)]
    assert effect_count_before_recovery == 1
    assert checkpoint_before_recovery.seq == expected_seq
    assert (
        checkpoint_before_recovery.checkpoint.applied_input_ids.count(command_marker)
        == expected_marker_count
    )

    recovered = subprocess.run(
        _worker_args(
            root,
            recover_mode,
            db_path=db_path,
            run_root=run_root,
            workspace=workspace,
            effect_db=effect_db,
            output=recovery_output,
        ),
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert recovered.returncode == 0, recovered.stderr or recovered.stdout
    result = json.loads(recovery_output.read_text(encoding="utf-8"))

    assert result["recovery_driver_mode"] == recovery_driver_mode
    hosted_recovery = recovery_driver_mode == "hosted"
    assert result["ambient_rejected"] is hosted_recovery
    assert result["host_closed"] is hosted_recovery
    assert result["host_running_after_launch"] is hosted_recovery
    assert result["host_running_after_rejection"] is hosted_recovery
    assert result["replacement_constructed"] is hosted_recovery
    assert result["shared_runtime"] is hosted_recovery
    assert result["completed"] == result["duplicate"]
    assert result["completed"]["run_id"] == "run_dbos_resume"
    assert result["completed"]["command_id"] == "resume_once"
    assert result["completed"]["status"] == "completed"
    assert result["conflict_code"] == "command_id_conflict"
    assert result["completed"]["checkpoint_seq"] == 2
    assert result["completed"]["state"] == "awaiting_input"
    assert result["effect_count"] == 1
    assert result["latest_seq"] == 2
    assert result["marker_count"] == 1
    assert result["stale"]["status"] == "failed"
    assert result["stale"]["error_code"] == "stale_resume_checkpoint"
    assert result["stale"]["error"] == "DBOS run resume was safely rejected"
    assert result["stale_workflow_success_rows"] == 1
    assert result["workflow_success_rows"] == 1
