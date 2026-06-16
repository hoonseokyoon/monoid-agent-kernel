from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from native_agent_runner.jobs import BackgroundJobManager


@dataclass
class JobsService:
    """Tool-facing view over the background job manager (list/status/logs/cancel/wait)."""

    job_manager: BackgroundJobManager

    def list_jobs(self) -> list[dict[str, Any]]:
        return self.job_manager.list_jobs()

    def status(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.job_manager.status(str(args["job_id"]))

    def logs(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.job_manager.logs(
            str(args["job_id"]),
            stream=str(args.get("stream") or "stdout"),  # type: ignore[arg-type]
            tail_bytes=args.get("tail_bytes"),
            offset=args.get("offset"),
        )

    def cancel(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.job_manager.cancel(str(args["job_id"]))

    def wait(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.job_manager.wait(str(args["job_id"]), timeout_s=args.get("timeout_s"))

    def background_metrics(self) -> dict[str, Any]:
        jobs = self.job_manager.list_jobs()
        terminal_jobs = [job for job in jobs if job.get("status") != "running"]
        failed_statuses = {"failed", "timed_out", "output_limited"}
        return {
            "background_jobs_started": len(jobs),
            "background_jobs_finished": sum(1 for job in terminal_jobs if job.get("status") == "exited"),
            "background_jobs_failed": sum(1 for job in terminal_jobs if job.get("status") in failed_statuses),
            "background_jobs_cancelled": sum(1 for job in terminal_jobs if job.get("status") == "cancelled"),
            "background_job_duration_s_total": sum(float(job.get("duration_s") or 0.0) for job in terminal_jobs),
            "background_job_bytes_stdout": sum(int(job.get("stdout_bytes") or 0) for job in jobs),
            "background_job_bytes_stderr": sum(int(job.get("stderr_bytes") or 0) for job in jobs),
        }
