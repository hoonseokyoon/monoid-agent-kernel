"""Subprocess worker used by the optional DBOS control-plane acceptance tests."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event, Lock, Timer
from typing import Any

from monoid_agent_kernel.core.control import ControlCommand, ControlResult
from monoid_agent_kernel.reference.dbos import (
    DbosControlConfig,
    DbosControlEnvelope,
    DbosControlPlane,
    DbosProcessOwnershipError,
    DbosShutdownTimeout,
)

_APP_VERSION = "monoid-dbos-control-test-v1"
_EXECUTOR_ID = "stable-test-executor"


def _config(
    db_path: Path,
    *,
    application_version: str = _APP_VERSION,
    shutdown_grace_s: int = 30,
) -> DbosControlConfig:
    return DbosControlConfig(
        system_database_url=f"sqlite:///{db_path}",
        name="monoid-dbos-control-test",
        application_version=application_version,
        executor_id=_EXECUTOR_ID,
        polling_interval_s=0.01,
        shutdown_grace_s=shutdown_grace_s,
    )


def _ok(envelope: DbosControlEnvelope) -> ControlResult:
    return ControlResult(
        run_id=envelope.run_id,
        type=envelope.type,
        status="ok",
        state="running" if envelope.type == "resume" else "awaiting_input",
        data={"command_id": envelope.command_id, "applied": True},
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def exercise(db_path: Path, output_path: Path, secret: str) -> None:
    dispatched: list[str] = []
    active = 0
    max_active = 0
    dispatch_lock = Lock()

    def dispatch(envelope: DbosControlEnvelope) -> ControlResult:
        nonlocal active, max_active
        dispatched.append(envelope.command_id)
        with dispatch_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.05)
            return _ok(envelope)
        finally:
            with dispatch_lock:
                active -= 1

    plane = DbosControlPlane(_config(db_path), dispatch)
    plane.launch()
    command = ControlCommand(
        type="resume",
        run_id="run_dbos",
        args={"token": secret, "nested": {"password": secret}, "safe": "visible"},
        issuer=f"operator-{secret}",
        reason=f"resume with {secret}",
        command_id="cmd_resume",
    )
    envelope = DbosControlEnvelope.from_control_command(
        command,
        tenant_id="tenant",
        user_id="user",
    )
    unsafe_envelope_rejected = False
    try:
        plane.enqueue_control(
            envelope,  # type: ignore[arg-type]
            tenant_id="tenant",
            user_id="user",
        )
    except TypeError:
        unsafe_envelope_rejected = True
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(
            plane.enqueue_control,
            command,
            tenant_id="tenant",
            user_id="user",
        )
        duplicate_future = pool.submit(
            plane.enqueue_control,
            command,
            tenant_id="tenant",
            user_id="user",
        )
        first = first_future.result(timeout=10)
        duplicate = duplicate_future.result(timeout=10)
    conflict_code = ""
    try:
        plane.enqueue_control(
            replace(command, reason="different semantic command"),
            tenant_id="tenant",
            user_id="user",
        )
    except Exception as exc:  # the parent asserts the stable error code
        conflict_code = str(getattr(exc, "error_code", ""))
    plane.enqueue_control(
        ControlCommand(type="status", run_id="run_dbos", command_id="cmd_status"),
        tenant_id="tenant",
        user_id="user",
    )
    resumed = plane.wait_for_receipt("run_dbos", "cmd_resume", timeout_s=20)
    status = plane.wait_for_receipt("run_dbos", "cmd_status", timeout_s=20)
    _write_json(
        output_path,
        {
            "first_status": first.status,
            "duplicate_status": duplicate.status,
            "resumed": resumed.to_json(),
            "status": status.to_json(),
            "conflict_code": conflict_code,
            "dispatched": dispatched,
            "max_active": max_active,
            "unsafe_envelope_rejected": unsafe_envelope_rejected,
            "persisted_envelope": envelope.to_json(),
        },
    )
    plane.close()


def crash_phase(db_path: Path, started_path: Path) -> None:
    def dispatch(envelope: DbosControlEnvelope) -> ControlResult:
        started_path.write_text(envelope.command_id, encoding="utf-8")
        while True:
            time.sleep(1)

    plane = DbosControlPlane(_config(db_path), dispatch)
    plane.launch()
    plane.enqueue_control(
        ControlCommand(type="resume", run_id="run_restart", command_id="cmd_restart"),
        tenant_id="tenant",
        user_id="user",
    )
    while True:
        time.sleep(1)


def recovery_phase(db_path: Path, output_path: Path) -> None:
    dispatched: list[str] = []

    def dispatch(envelope: DbosControlEnvelope) -> ControlResult:
        dispatched.append(envelope.command_id)
        return _ok(envelope)

    plane = DbosControlPlane(_config(db_path), dispatch)
    plane.launch()
    receipt = plane.wait_for_receipt("run_restart", "cmd_restart", timeout_s=30)
    _write_json(
        output_path,
        {"receipt": receipt.to_json(), "dispatched": dispatched},
    )
    plane.close()


def partition_phase(db_path: Path, output_path: Path) -> None:
    rendezvous = Barrier(2, timeout=10)
    lock = Lock()
    active = 0
    max_active = 0

    def dispatch(envelope: DbosControlEnvelope) -> ControlResult:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            rendezvous.wait()
            return _ok(envelope)
        finally:
            with lock:
                active -= 1

    plane = DbosControlPlane(_config(db_path), dispatch)
    plane.launch()
    for run_id, command_id in (("run_a", "cmd_a"), ("run_b", "cmd_b")):
        plane.enqueue_control(
            ControlCommand(type="status", run_id=run_id, command_id=command_id),
            tenant_id="tenant",
            user_id="user",
        )
    receipts = [
        plane.wait_for_receipt(run_id, command_id, timeout_s=20).to_json()
        for run_id, command_id in (("run_a", "cmd_a"), ("run_b", "cmd_b"))
    ]
    _write_json(output_path, {"max_active": max_active, "receipts": receipts})
    plane.close()


def same_partition_phase(db_path: Path, output_path: Path) -> None:
    first_started = Event()
    release_first = Event()
    second_started = Event()
    lock = Lock()
    active = 0
    max_active = 0
    dispatched: list[str] = []

    def dispatch(envelope: DbosControlEnvelope) -> ControlResult:
        nonlocal active, max_active
        with lock:
            dispatched.append(envelope.command_id)
            active += 1
            max_active = max(max_active, active)
        try:
            if envelope.command_id == "cmd_first":
                first_started.set()
                if not release_first.wait(timeout=10):
                    raise TimeoutError("first command was not released")
            else:
                second_started.set()
            return _ok(envelope)
        finally:
            with lock:
                active -= 1

    plane = DbosControlPlane(_config(db_path), dispatch)
    plane.launch()
    plane.enqueue_control(
        ControlCommand(type="status", run_id="run_serial", command_id="cmd_first"),
        tenant_id="tenant",
        user_id="user",
    )
    if not first_started.wait(timeout=10):
        raise TimeoutError("first command did not start")
    plane.enqueue_control(
        ControlCommand(type="status", run_id="run_serial", command_id="cmd_second"),
        tenant_id="tenant",
        user_id="user",
    )
    second_overlapped = second_started.wait(timeout=0.5)
    release_first.set()
    receipts = [
        plane.wait_for_receipt("run_serial", command_id, timeout_s=20).to_json()
        for command_id in ("cmd_first", "cmd_second")
    ]
    _write_json(
        output_path,
        {
            "dispatched": dispatched,
            "max_active": max_active,
            "receipts": receipts,
            "second_overlapped": second_overlapped,
        },
    )
    plane.close()


def failure_phase(db_path: Path, output_path: Path, secret: str) -> None:
    def dispatch(envelope: DbosControlEnvelope) -> ControlResult:
        del envelope
        raise ValueError(f"callback detail must stay private: {secret}")

    plane = DbosControlPlane(_config(db_path), dispatch)
    plane.launch()
    plane.enqueue_control(
        ControlCommand(
            type="status",
            run_id="run_failure",
            command_id="cmd_failure",
            args={"token": secret},
        ),
        tenant_id="tenant",
        user_id="user",
    )
    receipt = plane.wait_for_receipt("run_failure", "cmd_failure", timeout_s=20)
    _write_json(output_path, {"receipt": receipt.to_json()})
    plane.close()


def lifecycle_phase(db_path: Path, output_path: Path) -> None:
    import dbos

    unlaunched = DbosControlPlane(_config(db_path), _ok)
    invalid_timeout_rejected = False
    try:
        unlaunched.close(timeout_s=float("inf"))  # type: ignore[arg-type]
    except ValueError:
        invalid_timeout_rejected = True
    unlaunched.close()

    dispatched: list[str] = []

    def dispatch(envelope: DbosControlEnvelope) -> ControlResult:
        dispatched.append(envelope.command_id)
        return _ok(envelope)

    plane = DbosControlPlane(_config(db_path), dispatch)
    exclusive_owner_rejected = False
    try:
        DbosControlPlane(_config(db_path), lambda envelope: _ok(envelope))
    except DbosProcessOwnershipError:
        exclusive_owner_rejected = True
    plane.launch()
    plane.enqueue_control(
        ControlCommand(type="status", run_id="run_lifecycle", command_id="cmd_lifecycle"),
        tenant_id="tenant",
        user_id="user",
    )
    receipt = plane.wait_for_receipt("run_lifecycle", "cmd_lifecycle", timeout_s=20)
    plane.close()
    replacement = DbosControlPlane(_config(db_path), _ok)
    replacement.close()
    config = _config(db_path)
    external_runtime = dbos.DBOS(
        config={
            "name": config.name,
            "system_database_url": config.system_database_url,
            "application_version": config.application_version,
            "executor_id": config.executor_id,
            "run_admin_server": False,
        }
    )
    external_runtime_rejected = False
    try:
        DbosControlPlane(config, _ok)
    except DbosProcessOwnershipError:
        external_runtime_rejected = True
    external_runtime.destroy(destroy_registry=True, workflow_completion_timeout_sec=0)
    _write_json(
        output_path,
        {
            "receipt": receipt.to_json(),
            "dispatched": dispatched,
            "exclusive_owner_rejected": exclusive_owner_rejected,
            "external_runtime_rejected": external_runtime_rejected,
            "invalid_timeout_rejected": invalid_timeout_rejected,
        },
    )


def queue_config_phase(db_path: Path, output_path: Path) -> None:
    import dbos

    config = _config(db_path)
    queue_name = DbosControlPlane.versioned_queue_name(
        config.queue_name,
        config.application_version,
    )
    runtime = dbos.DBOS(
        config={
            "name": config.name,
            "system_database_url": config.system_database_url,
            "application_version": config.application_version,
            "executor_id": config.executor_id,
            "run_admin_server": False,
        }
    )
    runtime.launch()
    runtime.register_queue(
        queue_name,
        worker_concurrency=2,
        concurrency=2,
        priority_enabled=True,
        partition_queue=False,
        polling_interval_sec=config.polling_interval_s,
        on_conflict="always_update",
    )
    runtime.register_queue(
        "unrelated-queue",
        concurrency=4,
        polling_interval_sec=config.polling_interval_s,
        on_conflict="always_update",
    )
    runtime.destroy(destroy_registry=True, workflow_completion_timeout_sec=1)

    plane = DbosControlPlane(config, _ok)
    plane.launch()
    queue = plane._runtime.retrieve_queue(plane.registered_queue_name)
    assert queue is not None
    _write_json(
        output_path,
        {
            "concurrency": queue.concurrency,
            "listening_queues": list(plane._runtime._listening_queues or ()),
            "partition_queue": queue.partition_queue,
            "priority_enabled": queue.priority_enabled,
            "queue_name": plane.registered_queue_name,
            "worker_concurrency": queue.worker_concurrency,
        },
    )
    plane.close()


def shutdown_phase(db_path: Path, output_path: Path) -> None:
    started = Event()
    release = Event()
    finished = Event()

    def dispatch(envelope: DbosControlEnvelope) -> ControlResult:
        started.set()
        try:
            if not release.wait(timeout=10):
                raise TimeoutError("shutdown test dispatcher was not released")
            return _ok(envelope)
        finally:
            finished.set()

    plane = DbosControlPlane(_config(db_path, shutdown_grace_s=1), dispatch)
    plane.launch()
    plane.enqueue_control(
        ControlCommand(type="status", run_id="run_shutdown", command_id="cmd_shutdown"),
        tenant_id="tenant",
        user_id="user",
    )
    if not started.wait(timeout=10):
        raise TimeoutError("shutdown test dispatcher did not start")

    shutdown_timed_out = False
    try:
        plane.close()
    except DbosShutdownTimeout:
        shutdown_timed_out = True
    admission_stopped = False
    try:
        plane.enqueue_control(
            ControlCommand(type="status", run_id="run_shutdown", command_id="cmd_late"),
            tenant_id="tenant",
            user_id="user",
        )
    except RuntimeError:
        admission_stopped = True
    ownership_retained = False
    try:
        DbosControlPlane(_config(db_path), _ok)
    except DbosProcessOwnershipError:
        ownership_retained = True

    release.set()
    dispatch_finished = finished.wait(timeout=5)
    _write_json(
        output_path,
        {
            "admission_stopped": admission_stopped,
            "dispatch_finished": dispatch_finished,
            "ownership_retained": ownership_retained,
            "shutdown_timed_out": shutdown_timed_out,
        },
    )


def shutdown_success_phase(db_path: Path, output_path: Path) -> None:
    started = Event()
    release = Event()

    def dispatch(envelope: DbosControlEnvelope) -> ControlResult:
        started.set()
        if not release.wait(timeout=10):
            raise TimeoutError("graceful shutdown dispatcher was not released")
        return _ok(envelope)

    plane = DbosControlPlane(_config(db_path, shutdown_grace_s=3), dispatch)
    plane.launch()
    plane.enqueue_control(
        ControlCommand(
            type="status",
            run_id="run_shutdown_success",
            command_id="cmd_shutdown_success",
        ),
        tenant_id="tenant",
        user_id="user",
    )
    if not started.wait(timeout=10):
        raise TimeoutError("graceful shutdown dispatcher did not start")
    timer = Timer(0.25, release.set)
    timer.start()
    started_close = time.monotonic()
    plane.close()
    elapsed_s = time.monotonic() - started_close
    timer.join(timeout=2)
    replacement = DbosControlPlane(_config(db_path), _ok)
    replacement.close()
    _write_json(
        output_path,
        {"elapsed_s": elapsed_s, "replacement_constructed": True},
    )


def generated_id_phase(db_path: Path, output_path: Path) -> None:
    plane = DbosControlPlane(_config(db_path), _ok)
    plane.launch()
    admitted = plane.enqueue_control(
        ControlCommand(type="status", run_id="run_generated_id"),
        tenant_id="tenant",
        user_id="user",
    )
    completed = plane.wait_for_receipt(
        "run_generated_id",
        admitted.command_id,
        timeout_s=20,
    )
    _write_json(
        output_path,
        {"admitted": admitted.to_json(), "completed": completed.to_json()},
    )
    plane.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "exercise",
            "crash",
            "recover",
            "partitions",
            "same-partition",
            "failure",
            "lifecycle",
            "queue-config",
            "shutdown",
            "shutdown-success",
            "generated-id",
        ),
    )
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--started", type=Path)
    parser.add_argument("--secret", default="")
    args = parser.parse_args()
    if args.mode == "exercise":
        assert args.output is not None
        exercise(args.db, args.output, args.secret)
    elif args.mode == "crash":
        assert args.started is not None
        crash_phase(args.db, args.started)
    elif args.mode == "recover":
        assert args.output is not None
        recovery_phase(args.db, args.output)
    elif args.mode == "partitions":
        assert args.output is not None
        partition_phase(args.db, args.output)
    elif args.mode == "same-partition":
        assert args.output is not None
        same_partition_phase(args.db, args.output)
    elif args.mode == "failure":
        assert args.output is not None
        failure_phase(args.db, args.output, args.secret)
    elif args.mode == "lifecycle":
        assert args.output is not None
        lifecycle_phase(args.db, args.output)
    elif args.mode == "queue-config":
        assert args.output is not None
        queue_config_phase(args.db, args.output)
    elif args.mode == "shutdown":
        assert args.output is not None
        shutdown_phase(args.db, args.output)
    elif args.mode == "shutdown-success":
        assert args.output is not None
        shutdown_success_phase(args.db, args.output)
    else:
        assert args.output is not None
        generated_id_phase(args.db, args.output)


if __name__ == "__main__":
    main()
