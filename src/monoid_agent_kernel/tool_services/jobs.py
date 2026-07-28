from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from monoid_agent_kernel.errors import ToolExecutionError
from monoid_agent_kernel.public_view import public_identifier
from monoid_agent_kernel.tasks import TaskManager


@dataclass
class JobsService:
    """Tool-facing view over the task manager (list/status/logs/cancel/wait)."""

    job_manager: TaskManager

    def _job_id(self, args: dict[str, Any]) -> str:
        """The requested job id, or a tool error naming it.

        `TaskManager` raises `KeyError` for an id it does not know, and `KeyError` is not in the
        `(NativeAgentError, ValueError, TypeError)` set the tool-call handler catches -- so a model
        asking about a job that has finished, or inventing an id, terminated the whole run and
        republished its argument into `run.failed`, `status.json` and `metrics.json`. That is the
        defect `public_path`'s fail-closed guard was written to stop, on four twins nobody bound.
        Bounded in the message for the same reason the tool name is.
        """
        job_id = str(args["job_id"])
        if job_id not in self.job_manager.jobs:
            raise ToolExecutionError(
                f"unknown job_id: {public_identifier(job_id)}", error_code="job_unknown"
            )
        return job_id

    def _guarded(self, args: dict[str, Any], call: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
        """Run one job lookup with **every** `KeyError` converted, not just the id-not-found one.

        Checking membership in `job_manager.jobs` bound one of two sources and left the other:
        `read_job_log_text` raises a second `KeyError` for an id that *is* registered but has no log
        file on disk -- which is every `HostedTask` (subagent, hitl, capability, tool approval). The
        model is handed those ids by `job.list`, so `agent.spawn` followed by `job.logs` on the id it
        just read terminated the run. Three of the four twins were bound and the fourth was bound
        halfway, which is worse than not bound: it looks covered.
        """
        job_id = self._job_id(args)
        try:
            return call(job_id)
        except KeyError as exc:
            detail = str(exc.args[0]) if exc.args else job_id
            raise ToolExecutionError(
                f"job data unavailable: {public_identifier(detail)}", error_code="job_unavailable"
            ) from None

    def list_jobs(self) -> list[dict[str, Any]]:
        return self.job_manager.list_jobs()

    def status(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._guarded(args, self.job_manager.status)

    def logs(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._guarded(
            args,
            lambda job_id: self.job_manager.logs(
                job_id,
                stream=str(args.get("stream") or "stdout"),  # type: ignore[arg-type]
                tail_bytes=args.get("tail_bytes"),
                offset=args.get("offset"),
            ),
        )

    def cancel(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._guarded(args, self.job_manager.cancel)

    def wait(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._guarded(args, lambda job_id: self.job_manager.wait(job_id, timeout_s=args.get("timeout_s")))

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
