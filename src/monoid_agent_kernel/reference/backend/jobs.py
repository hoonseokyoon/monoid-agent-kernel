from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from monoid_agent_kernel.tasks import (
    get_job_artifact,
    list_job_artifacts,
    read_job_log_text,
    request_job_cancel,
)
from monoid_agent_kernel.reference.backend.ports import RunRecordPort


@dataclass(frozen=True)
class JobServiceContext:
    authorize_run: Callable[[str, str], None]
    record: Callable[[str], RunRecordPort]


class JobService:
    """Reference job artifact, log, and cancel projections."""

    def __init__(self, context: JobServiceContext) -> None:
        self._context = context

    def jobs(self, run_id: str, token: str) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        record = self._context.record(run_id)
        jobs = list_job_artifacts(record.run_dir)
        return {"run_id": run_id, "tenant_id": record.tenant_id, "jobs": jobs}

    def job_status(self, run_id: str, token: str, job_id: str) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        record = self._context.record(run_id)
        return {
            "run_id": run_id,
            "tenant_id": record.tenant_id,
            "job": get_job_artifact(record.run_dir, job_id),
        }

    def job_logs(
        self,
        run_id: str,
        token: str,
        job_id: str,
        *,
        stream: str = "stdout",
        tail_bytes: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        record = self._context.record(run_id)
        logs = read_job_log_text(
            record.run_dir,
            job_id,
            stream=stream,  # type: ignore[arg-type]
            tail_bytes=tail_bytes,
            offset=offset,
        )
        return {"run_id": run_id, "tenant_id": record.tenant_id, **logs}

    def cancel_job(self, run_id: str, token: str, job_id: str) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        record = self._context.record(run_id)
        payload = request_job_cancel(record.run_dir, job_id)
        return {"run_id": run_id, "tenant_id": record.tenant_id, **payload}
