# DBOS Reference Activation-Recovery Profile

Status: experimental v0.19.2 Reference profile. The optional extra pins the validated DBOS 2.26
minor line. DBOS owns finite-activation admission, participant-local per-run serialization, retry,
same-slot workflow recovery, and workflow results. The private hosted composition adds one shared
process-runtime lifecycle for control and run participants. The Core-defined `CheckpointStore`
owns portable semantic state, including input deduplication, committed boundary receipts,
suspension, and terminal meaning.

This profile is a recovery interoperability proof. Its v0.19.2 scope covers finite control-dispatch
and run-resume activations plus private shared-host ownership for those hosted workflows.
`RunnerBackend`, Studio, product routing, and production cutover sit outside this profile.

Install the profile with:

```bash
python -m pip install "monoid-agent-kernel[reference-dbos]"
```

Importing `monoid_agent_kernel` or `monoid_agent_kernel.reference.dbos` stays lazy. DBOS loads when
a standalone `DbosRunDriver` or `DbosControlPlane`, or the private runtime host, is constructed.

## Boundary

The DBOS profile is a separate Reference composition path. Its run driver imports or constructs
none of these legacy orchestration services:

- `LeaseStore`;
- `CommandStore` and claim TTL state;
- `RecoveryService`;
- the Reference watchdog;
- `RunnerBackend._records` as durable ownership state.

`CheckpointStore` remains the canonical persistence seam for portable run state. DBOS decides
when one activation starts and retries. The activation commits its semantic result to the
checkpoint and verifies exact readback before the DBOS workflow first records `SUCCESS` with a
copy of that receipt. Later workflow-result reads may use DBOS's cached output. The checkpoint
remains the source of the semantic receipt.

DBOS workflow IDs, queue state, executor identity, and application version are operational
metadata. Checkpoint sequence, suspension, applied-input identity, semantic effect identity, and
terminal state are Monoid semantics. DBOS dependencies and types stay inside the optional
Reference profile; the checkpoint fields remain runtime-neutral.

## Finite resume workflow

`DbosRunDriver` accepts a `DbosResumeCommand(run_id, command_id, checkpoint_seq)`. One finite,
run-partitioned DBOS workflow performs this sequence:

1. Read the checked latest checkpoint.
2. Return the immutable receipt in `applied_input_receipts` when the checkpoint already contains
   this resume identity in `applied_input_ids`.
3. For a fresh input, require the latest checkpoint to match its declared source sequence. For a
   restarted input, require the durable `active_input` to contain the same identity and original
   source sequence with `phase="running"`. Reject stale inputs and competing activations.
4. Construct a fresh `AgentLoop`, restore the latest checkpoint and blobs, and drive through local
   tasks until one durable suspension boundary.
5. Persist each internal safety checkpoint produced by `AgentLoop` with the same `active_input`
   and `phase="running"`. One activation can therefore advance from source `N` through multiple
   monotonic checkpoints before it returns a boundary.
6. Commit the returned boundary with `phase="completed"`, the exact portable suspension
   observation, the applied identity, and its immutable identity-bound receipt.
7. Read the committed checkpoint back, verify ownership, and return the same receipt as the DBOS
   workflow result. If a store publishes the checkpoint and then raises while returning its
   response, exact canonical readback reconciles the write as committed. Older, missing, or
   transiently unreadable state keeps the same DBOS step pending and retrying; a same-sequence
   mismatch or newer checkpoint fences the writer.
8. Release the parked loop's process resources without finalizing the run or deleting its
   checkpoint.

The receipt ledger preserves the result for each applied identity after newer inputs advance the
run. Retrying input A after input B returns A's stored boundary. The latest checkpoint's suspension
continues to describe B.

The queue uses `run_id` as its partition key. Queue and worker concurrency are one per partition,
so resumes for one run serialize while different runs can progress concurrently. Queue names are
scoped by `application_version`, and preflight repairs and verifies persisted queue configuration
before listeners accept work.

```python
from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore
from monoid_agent_kernel.reference.dbos import (
    DbosResumeCommand,
    DbosRunConfig,
    DbosRunDriver,
)

store = LocalFsCheckpointStore(run_root)

def build_loop(command):
    # Build an AgentLoop for command.run_id. Leave checkpoint_store and
    # checkpoint_persist_callback unset; DbosRunDriver owns this activation's commit.
    return make_agent_loop(command.run_id)

driver = DbosRunDriver(
    DbosRunConfig(
        system_database_url="sqlite:///monoid-dbos.sqlite",
        executor_id="stable-local-slot",
        application_version="my-product-v1",
    ),
    store,
    build_loop,
)
driver.launch()
command = DbosResumeCommand("run_123", "resume_456", checkpoint_seq=7)
driver.enqueue_resume(command)
receipt = driver.wait_for_receipt(command)
```

The exported `DbosControlPlane` and `DbosRunDriver` remain standalone experimental entry points.
Their Reference-private hosted forms register with one private runtime host. The host constructs
one captured DBOS runtime and registry, aggregates the participants' distinct versioned queues and
stable workflow names under one shared namespace, launches once, and owns shared admission and
shutdown. Every hosted participant must request the same database URL, application name,
application version, executor ID, and shutdown grace. Queue and retry settings remain
participant-local.

The control and run queues serialize their own partitions independently. Cross-surface per-run
serialization and control-to-run semantic routing require product integration. The host and
participant-registration factories are private Reference implementation details. The legacy
Reference command inbox remains a separate valid multi-instance assembly.

## Stable executor slot

The initial operating model is one active process per stable executor slot. A restarted slot uses
the same `executor_id` and `application_version`. The supervisor terminates and waits for the prior
process, or supplies an equivalent fencing guarantee, before starting the replacement.

DBOS 2.26 recovers `PENDING` work assigned to that same executor identity and application version.
The executor ID supplies recovery identity. The supervisor supplies single-process slot exclusion
and fencing.

Automatic takeover by an arbitrary host is outside the initial Reference scope. This design
excludes Conductor. A later multi-host requirement can add a narrow Reference recovery
coordinator that reassigns eligible DBOS work while preserving checkpoint sequence and slot
fencing. That coordinator stays smaller than a general orchestration control plane.

SQLite is the checked development and single-host acceptance database. v0.19.2 validates that
configuration only. DBOS recommends PostgreSQL for production system-database deployments;
PostgreSQL process qualification and any broader takeover contract belong to a future milestone.

## Verified recovery invariant

The recovery model permits a single activation to commit multiple internal safety checkpoints:

```text
source checkpoint N restore
  -> zero or more internal safety commits N+1 ... N+k-1
     with active input phase="running"
  -> semantic effect with stable idempotency key
  -> boundary checkpoint N+k (k >= 1)
     with phase="completed" + applied identity + identity-bound receipt
  -> process kill
  -> same executor slot/version restart
  -> same resume retry
  -> the identical stored boundary receipt, with no further checkpoint advance
```

The acceptance test asserts one semantic effect row, a monotonic final checkpoint `N+k`, one
committed resume identity, one immutable receipt for that identity, one DBOS `SUCCESS` workflow
row, and the same finite-activation workflow result for duplicate callers. Internal safety
checkpoints may make `k` greater than one. `N+1` is the simple case; the general sequence contract
is `N+k`.

The DBOS workflow result is an operational copy of the canonical checkpoint receipt. Its
`terminal` field reports whether the Monoid boundary is terminal. A successful finite activation
represents activation completion; the checkpoint records overall Monoid run terminality.

## Side-effect rule

DBOS steps can run again when a process dies before DBOS records the step result. The Monoid
boundary checkpoint commits the applied identity and its receipt together, so a recovered step
returns that receipt without another model/tool drive. An external target can still commit before
that boundary checkpoint. Strict runs therefore use a stable explicit idempotency key or durable
outbox delivery for every external effect.

The acceptance test uses `command_id` as an explicit unique effect key. A randomly generated
outbox request ID does not supply replay stability across reconstruction. Strict integrations must
provide a replay-stable idempotency identity or durable outbox staging. Neither path requires DBOS
workflow identity in Core.

DBOS transactions cover database work performed through their transaction boundary. Model, tool,
filesystem, and network effects retain their declared idempotent or outbox delivery semantics.

A checkpoint-store timeout has an ambiguous commit outcome. The run driver keeps that uncertainty
pending and retries the same canonical checkpoint inside the
live DBOS step until readback proves the commit or a conflicting writer. If shutdown grace expires
while storage remains uncertain, the supervisor terminates the process; same-slot startup recovery
then resumes the still-pending DBOS workflow.

## Runtime and version lifecycle

DBOS owns a process-global runtime and workflow registry. Standalone Reference DBOS components
claim one process owner before workflow registration. In the private hosted composition, control
and run participants register before launch and share an exact host identity. The host registers
both workflow families, preflights both queues, installs one aggregate listener set, launches the
captured runtime once, registers both queues, and then opens admission.

Identity-scoped hosted submissions require a clean external call context; an ambient DBOS workflow
context is rejected before durable enqueue. Shared close stops host and participant admission,
drains admitted facade calls, destroys the captured runtime once under one deadline, proves both
participants drained, clears DBOS globals, and releases process ownership. Uncertain launch,
drain, or shutdown fences the host and requires process termination.

Finite workflows keep version-drain pressure bounded. A deployment retains compatible workflow
code for pending work, keeps `application_version` stable across a slot restart, and changes the
version when workflow operation order or durable input/result compatibility changes.

The exported resume-command and run-receipt version constants describe this experimental
Reference profile only. They are excluded from the stable Core compatibility inventory and carry
no mixed-version rolling-reader guarantee in v0.19.2. Drain pending workflows before deploying an
incompatible record or operation-order change.

Official references:

- [Workflow guarantees and idempotent workflow IDs](https://docs.dbos.dev/python/tutorials/workflow-tutorial)
- [Queues, partition keys, and concurrency](https://docs.dbos.dev/python/tutorials/queue-tutorial)
- [SQLite and PostgreSQL system databases](https://docs.dbos.dev/python/tutorials/database-connection)
- [Workflow recovery](https://docs.dbos.dev/production/workflow-recovery)
- [Workflow code upgrades](https://docs.dbos.dev/python/tutorials/upgrading-workflows)

## v0.19.2 status and non-goals

Completed vertical-slice gates:

1. `DbosRunDriver` restores a checkpoint, persists internal safety checkpoints, and drives one
   durable suspension boundary without legacy lease, inbox, recovery, watchdog, or record-registry
   services.
2. Durable active-input ownership rejects stale or competing activation attempts. Workflow IDs and
   the immutable checkpoint receipt ledger return the original result for duplicate inputs.
3. Kill/restart recovery under the same stable executor slot produces one semantic effect and one
   finite-activation workflow result copied from the committed checkpoint receipt.
4. Core checkpoints carry the exact portable suspension observation, and `AgentLoop.release_parked`
   closes process resources without finalizing the run.
5. The DBOS 2.26 adapter binds workflow registration, identity-scoped workflow access, queues,
   launch, and shutdown to one captured runtime and verifies process-global ownership.
6. The private runtime host aggregates hosted control and run workflows under one listener set,
   launch, admission, drain, and shutdown lifecycle. Hosted control recovery and hosted run
   recovery pass same-slot process-restart acceptance.
7. Run recovery covers effect-committed and boundary-committed crashes across standalone,
   hosted, and standalone-to-hosted transitions. Real composition acceptance proves one runtime,
   two queues, one launch, one destroy, both participant receipts, and fresh replacement ownership.

The v0.19.2 profile stops at that Reference-private vertical slice. Its non-goals are:

- replacing `RunnerBackend`, its HTTP facade, or Studio;
- publishing the private runtime host or participant-registration seam as a supported product API;
- automatic standalone-component composition or control-to-run semantic routing;
- migrating submission, task-result, pause, cancel, status, or terminal artifact projection;
- claiming one physical write for proposal, metrics, `run.finished`, or other rebuildable views;
- PostgreSQL production qualification, rolling upgrades, or arbitrary-host takeover;
- retiring the accepted durable Reference command-inbox workstream.

These items sit outside the v0.19.2 release scope and require separate product-integration design
review. The private host owns DBOS lifecycle. Hosted participants add no `LeaseStore`,
`CommandStore`, `RecoveryService`, or watchdog lifecycle. `CheckpointStore` remains authoritative
for boundary and terminal receipts and other portable semantic state while DBOS remains the
activation scheduler.
