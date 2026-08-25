"""Disposable Activity worker used by the PR10 hard-crash service qualification."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping

_SUPPORT_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == _SUPPORT_DIR:
    sys.path.pop(0)
sys.path.insert(0, str(_SUPPORT_DIR.parent))

from temporalio.client import Client
from temporalio.worker import Worker

from monoid_agent_kernel.adapters.postgres import (
    PostgresCommandAdmissionStore,
    PostgresConfig,
    PostgresDatabase,
    PostgresFencedRunSink,
    PostgresWriterAuthorityStore,
)
from monoid_agent_kernel.adapters.temporal.activity import (
    TemporalActivationActivity,
    TemporalActivityPolicy,
)
from monoid_agent_kernel.core.model_invocation import DurableModelInvocation
from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.hosting import (
    ActivationCommand,
    ActivationRuntime,
    CommitResult,
    WriterToken,
)
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from support.runtime import runtime_config, runtime_provider


@dataclass
class _FileCountingAdapter:
    counter_path: Path

    def next_turn(self, request: Any) -> ModelTurn:
        del request
        current = int(self.counter_path.read_text("utf-8")) if self.counter_path.exists() else 0
        self.counter_path.write_text(str(current + 1), encoding="utf-8")
        return ModelTurn(final_text="private paid model result before worker crash")


@dataclass
class _BlockAfterSettledInvocation:
    inner: PostgresFencedRunSink
    marker_path: Path

    @property
    def capabilities(self):  # noqa: ANN201 - exact adapter capability passthrough
        return self.inner.capabilities

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
        stage_evidence: bool = False,
    ) -> CommitResult:
        result = self.inner.commit_invocation(
            invocation,
            blobs,
            writer_token=writer_token,
            stage_evidence=stage_evidence,
        )
        if (
            invocation.dispatch_state == "settled"
            and result.status in {"committed", "already_committed"}
        ):
            self.marker_path.write_text(invocation.logical_call_id, encoding="utf-8")
            threading.Event().wait()
        return result


def _loop_factory(
    *,
    workspace_root: Path,
    run_root: Path,
    adapter: _FileCountingAdapter,
):
    provider = runtime_provider(runtime_config("fs.write"))

    def build(command: ActivationCommand, runtime: ActivationRuntime) -> AgentLoop:
        return AgentLoop(
            spec=AgentRunSpec(
                run_id=command.run_id,
                workspace_root=workspace_root,
                run_root=run_root,
            ),
            model_adapter=adapter,
            runtime_config_provider=provider,
            run_sink=runtime.run_sink,
            writer_token=runtime.writer_token,
            write_authority=runtime.write_authority,
            cancellation_token=runtime.cancellation_token,
            authoritative_event_sinks=(runtime.event_sink,),
            event_sequence_seed=runtime.event_sequence_seed,
            status_file=False,
        )

    return build


async def _run(args: argparse.Namespace) -> None:
    database = PostgresDatabase(
        PostgresConfig(
            dsn=args.postgres_dsn,
            schema=args.postgres_schema,
            min_pool_size=1,
            max_pool_size=8,
            pool_timeout_s=10,
            lock_timeout_s=5,
            statement_timeout_s=15,
            application_name="monoid-pr10-crash-worker",
        )
    )
    database.open()
    authority = PostgresWriterAuthorityStore(database)
    admission = PostgresCommandAdmissionStore(database)
    base_sink = PostgresFencedRunSink(database)
    authority.check_ready()
    admission.check_ready()
    base_sink.check_ready()
    sink = _BlockAfterSettledInvocation(base_sink, Path(args.marker_path))
    adapter = _FileCountingAdapter(Path(args.counter_path))
    activation = TemporalActivationActivity(
        authority_store=authority,
        admission_store=admission,
        run_sink=sink,
        loop_factory=_loop_factory(
            workspace_root=Path(args.workspace_root),
            run_root=Path(args.run_root),
            adapter=adapter,
        ),
        policy=TemporalActivityPolicy(
            writer_lease_ttl_s=2,
            writer_lease_renew_interval_s=0.4,
            heartbeat_interval_s=0.2,
            authority_call_timeout_s=2,
            driver_call_timeout_s=20,
            supervisor_join_timeout_s=2,
            local_task_wait_s=5,
        ),
    )
    client = await Client.connect(args.temporal_target)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            async with Worker(
                client,
                task_queue=args.activity_task_queue,
                activities=[activation.run],
                activity_executor=executor,
                max_concurrent_activities=2,
                max_heartbeat_throttle_interval=timedelta(seconds=0.2),
                default_heartbeat_throttle_interval=timedelta(seconds=0.2),
            ):
                await asyncio.Event().wait()
    finally:
        database.close()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal-target", required=True)
    parser.add_argument("--postgres-dsn", required=True)
    parser.add_argument("--postgres-schema", required=True)
    parser.add_argument("--activity-task-queue", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--marker-path", required=True)
    parser.add_argument("--counter-path", required=True)
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
