# DBOS Reference Activation-Recovery Profile

Status: experimental v0.18.0 Reference profile. The optional extra pins the validated DBOS 2.26
minor line. DBOS owns the operational execution of one finite activation: admission, per-run
serialization, retry, same-slot workflow recovery, and the workflow result. The Core-defined
`CheckpointStore` owns the portable semantic state, including input deduplication, the committed
boundary receipt, suspension, and terminal meaning.

This profile is a recovery interoperability proof. Its v0.18 scope covers one finite activation;
`RunnerBackend`, Studio, and production cutover remain separate.

Install the profile with:

```bash
python -m pip install "monoid-agent-kernel[reference-dbos]"
```

Importing `monoid_agent_kernel` or `monoid_agent_kernel.reference.dbos` stays lazy. DBOS loads when
`DbosRunDriver` or `DbosControlPlane` is constructed.

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

`DbosControlPlane` remains the isolated authenticated, credential-sanitizing transport experiment
from the earlier control-plane spike. v0.18 evaluates it separately from `DbosRunDriver`. The
legacy Reference command inbox remains a separate valid multi-instance assembly. Shared DBOS host
composition belongs to a future architecture decision.

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

SQLite is the checked development and single-host acceptance database. v0.18 validates that
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

DBOS owns a process-global runtime and workflow registry. Reference DBOS components claim one
process owner before workflow registration. `close()` stops admission, waits for active workflows,
verifies the DBOS active-workflow set, and releases ownership. A grace timeout requires process
termination.

Finite workflows keep version-drain pressure bounded. A deployment retains compatible workflow
code for pending work, keeps `application_version` stable across a slot restart, and changes the
version when workflow operation order or durable input/result compatibility changes.

Official references:

- [Workflow guarantees and idempotent workflow IDs](https://docs.dbos.dev/python/tutorials/workflow-tutorial)
- [Queues, partition keys, and concurrency](https://docs.dbos.dev/python/tutorials/queue-tutorial)
- [SQLite and PostgreSQL system databases](https://docs.dbos.dev/python/tutorials/database-connection)
- [Workflow recovery](https://docs.dbos.dev/production/workflow-recovery)
- [Workflow code upgrades](https://docs.dbos.dev/python/tutorials/upgrading-workflows)

## v0.18 status and non-goals

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

The v0.18 profile deliberately stops at that vertical slice. Its non-goals are:

- replacing `RunnerBackend`, its HTTP facade, or Studio;
- composing the control-plane experiment and run driver into a production host;
- migrating submission, task-result, pause, cancel, status, or terminal artifact projection;
- claiming one physical write for proposal, metrics, `run.finished`, or other rebuildable views;
- PostgreSQL production qualification, rolling upgrades, or arbitrary-host takeover;
- retiring the accepted durable Reference command-inbox workstream.

These items sit outside the v0.18 release scope. A future DBOS milestone must first choose one
explicit authority model. The portable model keeps `CheckpointStore` authoritative for
boundary/terminal receipts and other semantic state while DBOS remains an activation scheduler.
