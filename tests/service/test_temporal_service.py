from __future__ import annotations

import asyncio
import os
import uuid

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.serial, pytest.mark.service, pytest.mark.slow]
temporalio = pytest.importorskip("temporalio")

from temporalio import workflow  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import Worker  # noqa: E402


@workflow.defn
class _ServiceSmokeWorkflow:
    @workflow.run
    async def run(self, value: str) -> str:
        return value


@pytest.mark.skipif(
    os.environ.get("MONOID_SERVICE_PROFILE") not in {"temporal", "combined"},
    reason="Temporal service profile is not selected",
)
def test_pinned_temporal_local_server_executes_workflow() -> None:
    cli_version = os.environ.get("MONOID_TEMPORAL_CLI_VERSION")
    if not cli_version:
        pytest.fail("MONOID_TEMPORAL_CLI_VERSION is required for the selected service profile")
    assert cli_version.startswith("v")

    async def run() -> str:
        async with await WorkflowEnvironment.start_local(
            dev_server_download_version=cli_version,
            dev_server_log_level="warn",
        ) as environment:
            task_queue = f"monoid-v023-{uuid.uuid4()}"
            async with Worker(
                environment.client,
                task_queue=task_queue,
                workflows=[_ServiceSmokeWorkflow],
            ):
                return await environment.client.execute_workflow(
                    _ServiceSmokeWorkflow.run,
                    "service-smoke",
                    id=f"monoid-v023-{uuid.uuid4()}",
                    task_queue=task_queue,
                )

    assert asyncio.run(run()) == "service-smoke"
