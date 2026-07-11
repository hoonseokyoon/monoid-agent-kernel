# DBOS Reference Control-Plane Profile

Status: experimental adoption spike for v0.18.0. The default Reference backend remains unchanged
while this profile is validated against the durable command-inbox and ownerless-recovery failure
matrix. The extra pins the validated DBOS 2.26 minor line; upgrades require rerunning these
acceptance and lifecycle probes.

Install the optional profile with:

```bash
python -m pip install "monoid-agent-kernel[reference-dbos]"
```

Importing `monoid_agent_kernel` or `monoid_agent_kernel.reference.dbos` remains lazy. DBOS is loaded
only when `DbosControlPlane` is constructed.

## Boundary

The DBOS profile is a separate Reference orchestration path. It does not compose or import:

- `LeaseStore`;
- `CommandStore` or its claim TTL;
- `RecoveryService`;
- the Reference watchdog;
- `RunnerBackend._records` as durable ownership state.

Each accepted control is a finite DBOS workflow. Its workflow ID is derived from
`(run_id, command_id)`, and a partitioned queue uses `run_id` as the partition key. DBOS serializes
commands for the same run while allowing different run partitions to progress concurrently. The
adapter scopes the queue name by `application_version`, fixes global concurrency at one per
partition, repairs and verifies the persisted queue configuration through `DBOSClient` before
runtime listeners start, and verifies it again before accepting commands. Global per-partition
concurrency is the cross-executor ordering invariant; worker concurrency is also pinned to one.
The workflow result is projected to the existing `monoid.command-receipt.v1` receipt shape.

```python
from monoid_agent_kernel.core.control import ControlCommand, ControlResult
from monoid_agent_kernel.reference.dbos import DbosControlConfig, DbosControlPlane

config = DbosControlConfig(
    system_database_url="sqlite:///monoid-dbos.sqlite",
    executor_id="stable-local-slot",
    application_version="my-product-v1",
)

def dispatch(envelope):
    # A production implementation restores the run checkpoint and applies one idempotent
    # suspension-to-suspension drive operation here.
    return ControlResult(
        run_id=envelope.run_id,
        type=envelope.type,
        status="ok",
        data={"applied": True},
    )

control = DbosControlPlane(config, dispatch)
control.launch()
receipt = control.enqueue_control(
    ControlCommand(
        type="resume",
        run_id="run_123",
        command_id="cmd_456",
        args={"token": authenticated_bearer},
    ),
    tenant_id=authenticated_claims.tenant_id,
    user_id=authenticated_claims.user_id,
)
```

The HTTP edge authenticates the bearer before calling `enqueue_control`. The control plane builds
the durable envelope internally, removes the bearer, redacts repeated bearer text, sanitizes
credential-shaped fields, and persists only its SHA-256 reference. The spike restricts the durable
vocabulary to `pause`, `resume`, `cancel`, and `status`; commands that return one-time secrets
remain outside this generic durable transport.

## Guarantees demonstrated by the spike

- A repeated `(run_id, command_id)` with the same semantic payload returns the existing workflow
  and produces one normal-case dispatch.
- DBOS workflow IDs are first-writer-wins. The adapter compares the persisted workflow input's
  canonical identity and raises `command_id_conflict` for the same ID with different content.
- Queue concurrency is one per run partition, so finite command workflows serialize without a
  process-local owner lease while different runs may progress concurrently.
- A workflow interrupted inside its command step recovers after the same stable executor slot and
  application version restart.
- Durable workflow inputs and the SQLite system database contain no raw bearer credential.

## Deliberate limitations

DBOS workflows do not make arbitrary model, tool, filesystem, or network effects exactly once.
Steps are retried when a process dies before their result is checkpointed. A dispatcher must use
the command ID as its effect idempotency key and retain Monoid's checkpoint/outbox rules. DBOS
database transactions are exactly once; arbitrary external effects are at least once.

DBOS 2.26 orders queued work by `(priority, created_at)`. This profile disables priority and
demonstrates sequential-producer ordering. Concurrent submissions can share a timestamp, and the
queue query has no unique ordering tie-breaker. Exact concurrent append-order parity with the
legacy inbox requires an atomic per-run sequence or a stronger upstream DBOS guarantee before
adoption. The spike also has no per-run queue-depth limit yet.

SQLite is a development and single-host test system database. A production multi-server profile
requires PostgreSQL. DBOS automatically recovers `PENDING` workflows only when the restarted
process uses the same `executor_id` and `application_version`. A surviving executor does not take
over another executor's pending work by itself. Automatic cross-executor recovery requires DBOS
Conductor; without it, the deployment supervisor must recreate a stable executor slot and ID.
Self-hosted production Conductor is separately licensed, so its operating cost belongs in the
production-profile decision.

DBOS owns a process-global runtime and workflow registry. This profile owns that lifecycle and
enforces one live control-plane instance per process; a second constructor fails before workflow
registration can replace the first dispatcher. `close()` stops admission and waits for active
DBOS workflows for the positive whole-second `shutdown_grace_s`, then verifies both the dispatcher
counter and DBOS 2.26 active-workflow registry are empty. A grace timeout keeps process ownership,
raises `DbosShutdownTimeout`, and requires process termination because DBOS has already stopped its
runtime resources. A future shared-host integration needs explicit runtime injection before
multiple DBOS-backed services can share one process safely.

Long-lived workflow code also has version compatibility obligations. This spike uses finite command
workflows to reduce old-version draining pressure. A production adopter must retain compatible
workers or use DBOS workflow patch/version mechanisms during rolling upgrades.

Official references:

- [Workflow guarantees and idempotent workflow IDs](https://docs.dbos.dev/python/tutorials/workflow-tutorial)
- [Queues, partition keys, and concurrency](https://docs.dbos.dev/python/tutorials/queue-tutorial)
- [SQLite and PostgreSQL system databases](https://docs.dbos.dev/python/tutorials/database-connection)
- [Executor recovery and Conductor](https://docs.dbos.dev/production/workflow-recovery)
- [Conductor hosting and licensing](https://docs.dbos.dev/production/hosting-conductor)
- [Workflow code upgrades](https://docs.dbos.dev/python/tutorials/upgrading-workflows)

## Adoption gates

This profile replaces the PR 31/33 orchestration only after all of the following pass:

1. `DbosRunDriver` restores a Monoid checkpoint and drives one suspension boundary without
   constructing the legacy lease, inbox, recovery, or watchdog services.
2. Duplicate and conflicting command IDs retain the current receipt contract.
3. Crash injection before and after workflow admission, checkpoint commit, external effects, and
   receipt commit converges to one semantic effect and one terminal receipt.
4. PostgreSQL two-process tests prove run-partition serialization and durable delivery.
5. Concurrent-producer tests prove exact per-run append ordering and enforce a per-run admission
   bound before the legacy inbox is retired.
6. The chosen production deployment provides Conductor or stable executor-slot restart semantics.
7. Host-owned runtime integration covers launch failure, shared registration, and orderly close.
8. Compatibility documentation covers DBOS `application_version`, workflow input/result schemas,
   and rollback to the legacy Reference profile.

If the implementation still needs the legacy `LeaseStore`, `CommandStore`, or in-memory receipt
repair alongside DBOS, the spike fails and must not become the production profile.
