from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from monoid_agent_kernel.reference.dbos.control_plane import _is_uninitialized_queue_table
from support.waiting import eventually

pytest.importorskip("dbos")


def test_dbos_preflight_recognizes_fresh_sqlite_and_postgres_queue_tables() -> None:
    from sqlalchemy.exc import OperationalError, ProgrammingError

    errors = (
        OperationalError(
            "SELECT * FROM queues",
            {},
            Exception("no such table: main.queues"),
        ),
        ProgrammingError(
            "SELECT * FROM dbos.queues",
            {},
            Exception('relation "dbos.queues" does not exist'),
        ),
    )

    assert all(_is_uninitialized_queue_table(error) for error in errors)


def _worker_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(root / "src"), str(root / "tests"), env.get("PYTHONPATH", "")))
    )
    return env


def _run_worker(root: Path, *args: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "support.dbos_control_worker", *args],
        cwd=root,
        env=_worker_env(root),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_dbos_control_plane_deduplicates_rejects_conflicts_and_persists_no_bearer(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    db_path = tmp_path / "dbos.sqlite"
    output_path = tmp_path / "result.json"
    secret = "raw-bearer-must-not-persist"

    completed = _run_worker(
        root,
        "exercise",
        "--db",
        str(db_path),
        "--output",
        str(output_path),
        "--secret",
        secret,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["first_status"] in {"pending", "completed"}
    assert result["duplicate_status"] in {"pending", "completed"}
    assert result["resumed"]["status"] == "completed"
    assert result["status"]["status"] == "completed"
    assert result["conflict_code"] == "command_id_conflict"
    assert result["dispatched"] == ["cmd_resume", "cmd_status"]
    assert result["max_active"] == 1
    assert result["unsafe_envelope_rejected"] is True
    assert result["persisted_envelope"]["args"] == {
        "nested": {"password": "[redacted]"},
        "safe": "visible",
    }
    assert secret not in str(result)
    for database_file in db_path.parent.glob(f"{db_path.name}*"):
        assert secret.encode() not in database_file.read_bytes()


def test_dbos_control_plane_recovers_pending_workflow_after_process_restart(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    db_path = tmp_path / "restart.sqlite"
    started_path = tmp_path / "started.txt"
    output_path = tmp_path / "recovered.json"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "support.dbos_control_worker",
            "crash",
            "--db",
            str(db_path),
            "--started",
            str(started_path),
        ],
        cwd=root,
        env=_worker_env(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert eventually(started_path.exists, timeout_s=30)
    finally:
        process.kill()
        process.wait(timeout=10)

    recovered = _run_worker(
        root,
        "recover",
        "--db",
        str(db_path),
        "--output",
        str(output_path),
        timeout=90,
    )

    assert recovered.returncode == 0, recovered.stderr or recovered.stdout
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["receipt"]["status"] == "completed"
    assert result["receipt"]["result"]["status"] == "ok"
    assert result["dispatched"] == ["cmd_restart"]


def test_dbos_control_plane_serializes_each_run_without_blocking_other_runs(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    db_path = tmp_path / "partitions.sqlite"
    output_path = tmp_path / "partitions.json"

    completed = _run_worker(
        root,
        "partitions",
        "--db",
        str(db_path),
        "--output",
        str(output_path),
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["max_active"] == 2
    assert [receipt["status"] for receipt in result["receipts"]] == [
        "completed",
        "completed",
    ]


def test_dbos_control_plane_does_not_start_a_second_command_for_the_same_run(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    db_path = tmp_path / "same-partition.sqlite"
    output_path = tmp_path / "same-partition.json"

    completed = _run_worker(
        root,
        "same-partition",
        "--db",
        str(db_path),
        "--output",
        str(output_path),
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["second_overlapped"] is False
    assert result["max_active"] == 1
    assert result["dispatched"] == ["cmd_first", "cmd_second"]
    assert [receipt["status"] for receipt in result["receipts"]] == [
        "completed",
        "completed",
    ]


def test_dbos_control_plane_keeps_dispatch_errors_and_bearers_out_of_durable_state(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    db_path = tmp_path / "failure.sqlite"
    output_path = tmp_path / "failure.json"
    secret = "private-callback-error-detail"

    completed = _run_worker(
        root,
        "failure",
        "--db",
        str(db_path),
        "--output",
        str(output_path),
        "--secret",
        secret,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["receipt"]["status"] == "failed"
    assert result["receipt"]["result"] == {
        "status": "error",
        "error": "DBOS command workflow failed",
        "error_code": "dbos_workflow_error",
    }
    assert secret not in str(result)
    assert secret not in completed.stderr
    for database_file in db_path.parent.glob(f"{db_path.name}*"):
        assert secret.encode() not in database_file.read_bytes()


def test_dbos_control_plane_releases_an_unlaunched_process_global_runtime(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    db_path = tmp_path / "lifecycle.sqlite"
    output_path = tmp_path / "lifecycle.json"

    completed = _run_worker(
        root,
        "lifecycle",
        "--db",
        str(db_path),
        "--output",
        str(output_path),
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["receipt"]["status"] == "completed"
    assert result["dispatched"] == ["cmd_lifecycle"]
    assert result["exclusive_owner_rejected"] is True
    assert result["external_runtime_rejected"] is True
    assert result["invalid_timeout_rejected"] is True


def test_dbos_control_plane_repairs_and_verifies_persisted_queue_configuration(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    db_path = tmp_path / "queue-config.sqlite"
    output_path = tmp_path / "queue-config.json"

    completed = _run_worker(
        root,
        "queue-config",
        "--db",
        str(db_path),
        "--output",
        str(output_path),
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == {
        "concurrency": 1,
        "partition_queue": True,
        "priority_enabled": False,
        "queue_name": (
            "monoid/control-queue/monoid-reference-control"
            "/version/monoid-dbos-control-test-v1"
        ),
        "worker_concurrency": 1,
    }


def test_dbos_control_plane_shutdown_timeout_stops_admission_and_retains_ownership(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    db_path = tmp_path / "shutdown.sqlite"
    output_path = tmp_path / "shutdown.json"

    completed = _run_worker(
        root,
        "shutdown",
        "--db",
        str(db_path),
        "--output",
        str(output_path),
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == {
        "admission_stopped": True,
        "dispatch_finished": True,
        "ownership_retained": True,
        "shutdown_timed_out": True,
    }


def test_dbos_control_plane_close_waits_for_an_active_dispatch_within_grace(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    db_path = tmp_path / "shutdown-success.sqlite"
    output_path = tmp_path / "shutdown-success.json"

    completed = _run_worker(
        root,
        "shutdown-success",
        "--db",
        str(db_path),
        "--output",
        str(output_path),
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["elapsed_s"] >= 0.2
    assert result["replacement_constructed"] is True


def test_dbos_profile_does_not_import_legacy_owner_or_inbox_orchestration() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/monoid_agent_kernel/reference/dbos/control_plane.py").read_text(
        encoding="utf-8"
    )

    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )

    assert not any("command_inbox" in module for module in imported)
    assert not any("stores.lease" in module for module in imported)
    assert not any("backend.recovery" in module for module in imported)
