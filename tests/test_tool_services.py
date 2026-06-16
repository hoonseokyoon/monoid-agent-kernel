from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from native_agent_runner.tool_services import CallContext, JobsService, ShellService, WebService
from native_agent_runner.web import WebPolicy


def test_call_context_holds_in_flight_ids() -> None:
    call = CallContext(tool_call_id="c1", turn_id="turn_0001", tool_event_id="e1")
    assert (call.tool_call_id, call.turn_id, call.tool_event_id) == ("c1", "turn_0001", "e1")


def test_shell_service_metrics_start_at_zero() -> None:
    service = ShellService(
        run_id="r",
        workspace=None,  # type: ignore[arg-type]
        recorder=None,  # type: ignore[arg-type]
        job_manager=None,  # type: ignore[arg-type]
        shell_policy=None,  # type: ignore[arg-type]
        permission_policy=None,  # type: ignore[arg-type]
    )
    assert service.metrics() == {
        "shell_calls": 0,
        "failed_shell_calls": 0,
        "total_shell_duration_s": 0.0,
    }


def test_web_service_metrics_keys() -> None:
    service = WebService(web_policy=WebPolicy(), recorder=None)  # type: ignore[arg-type]
    assert set(service.metrics()) == {
        "web_search_calls",
        "web_fetch_calls",
        "web_context_calls",
        "web_failed_calls",
        "web_result_count",
        "web_bytes_returned",
        "web_context_source_count",
        "web_context_bytes_returned",
    }
    assert all(value == 0 for value in service.metrics().values())


@dataclass
class _StubJobManager:
    jobs: list[dict[str, Any]]

    def list_jobs(self) -> list[dict[str, Any]]:
        return self.jobs


def test_jobs_service_background_metrics_aggregates_by_status() -> None:
    manager = _StubJobManager(
        jobs=[
            {"status": "exited", "duration_s": 1.5, "stdout_bytes": 10, "stderr_bytes": 2},
            {"status": "failed", "duration_s": 0.5, "stdout_bytes": 4, "stderr_bytes": 1},
            {"status": "cancelled", "duration_s": 0.0, "stdout_bytes": 0, "stderr_bytes": 0},
            {"status": "running", "stdout_bytes": 7, "stderr_bytes": 3},
        ]
    )
    service = JobsService(job_manager=manager)  # type: ignore[arg-type]
    metrics = service.background_metrics()
    assert metrics["background_jobs_started"] == 4
    assert metrics["background_jobs_finished"] == 1
    assert metrics["background_jobs_failed"] == 1
    assert metrics["background_jobs_cancelled"] == 1
    assert metrics["background_job_duration_s_total"] == 2.0
    assert metrics["background_job_bytes_stdout"] == 21
    assert metrics["background_job_bytes_stderr"] == 6
