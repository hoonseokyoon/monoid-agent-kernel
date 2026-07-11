from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("fault_phase", ("effect_committed", "boundary_committed"))
def test_dbos_run_resume_survives_kill_with_one_effect_and_one_terminal_receipt(
    tmp_path: Path,
    fault_phase: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    db_path = tmp_path / "dbos.sqlite"
    run_root = tmp_path / "runs"
    workspace = tmp_path / "workspace"
    effect_db = tmp_path / "effects.sqlite"
    seed_output = tmp_path / "seed.json"
    started_path = tmp_path / f"{fault_phase}.txt"
    recovery_output = tmp_path / "recovered.json"
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
    assert json.loads(seed_output.read_text(encoding="utf-8"))["checkpoint_seq"] == 1

    crashed = subprocess.Popen(
        _worker_args(
            root,
            "crash",
            db_path=db_path,
            run_root=run_root,
            workspace=workspace,
            effect_db=effect_db,
            started=started_path,
            fault_phase=fault_phase,
        ),
        cwd=root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not started_path.exists() and crashed.poll() is None:
        time.sleep(0.02)
    if not started_path.exists():
        if crashed.poll() is None:
            crashed.kill()
        crashed.wait(timeout=5)
        pytest.fail(f"DBOS worker exited before reaching crash phase {fault_phase!r}")
    crashed.kill()
    crashed.wait(timeout=10)

    recovered = subprocess.run(
        _worker_args(
            root,
            "recover",
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

    assert result["completed"] == result["duplicate"]
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
    assert result["terminal_receipt_rows"] == 1
