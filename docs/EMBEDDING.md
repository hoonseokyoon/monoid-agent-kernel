# Production embedding handbook

This handbook defines portable production responsibilities for two product topologies and supplies
one offline executable path for each:

- [`embedding_local_product.py`](../examples/embedding_local_product.py) embeds `AgentLoop`
  directly in one trusted process through the stable contracts and Core Helper Kit.
- [`embedding_hosted_product.py`](../examples/embedding_hosted_product.py) exercises a hosted,
  multi-tenant product through the accepted Reference durable-inbox assembly.

The hosted script selects one concrete Reference composition so the path can run end to end in CI.
Production services depend on [`monoid_agent_kernel.contracts`](CONTRACTS.md), choose their own
orchestration and storage implementations, and verify observable behavior with conformance
profiles. Reference packages provide runnable examples and smoke targets outside the supported
stable surface. See [REFERENCE.md](REFERENCE.md).

Run both paths from a checkout:

```bash
python examples/embedding_local_product.py
python examples/embedding_hosted_product.py
python -m pytest -q tests/test_examples.py
```

Both scripts use `FakeModelAdapter`, require no credentials, make no network requests, and create
all state under a temporary directory.

## Preserve the authority boundary

Each deployment assigns these responsibilities explicitly:

| Layer | Authority |
|---|---|
| Product edge/API | Authenticates the caller, selects tenant/user/run scope, enforces route policy, and returns product-facing responses. |
| Activation runtime | Admits work, serializes inputs for one run, retries activations, fences competing executors, and recovers operational work. |
| Core session and `CheckpointStore` | Define run meaning: checkpoint sequence, suspension, input deduplication, committed boundary receipts, and terminal state. |
| Event and status projections | Materialize authorized events, status, metrics, and proposals; document each projection's source of truth, idempotency, and retention policy. |
| Model/tool gateways | Hold provider credentials, enforce tenant policy and quotas, and return sanitized results. |
| Product database | Maps product records to `run_id` and stores product state without duplicating the Monoid state machine. |

Use one activation runtime for a run. Its queue and workflow states are operational diagnostics.
Monoid checkpoints and terminal receipts carry portable semantic state. Externally retried effects
use a replay-stable idempotency key or durable outbox staging.

For each run, select exactly one activation authority: a product-owned runtime, the Reference inbox
assembly, or the optional experimental DBOS profile. DBOS must not coexist with `LeaseStore`,
`CommandStore`, `RecoveryService`, or watchdog lifecycle ownership for that run. Core contracts
contain no DBOS, lease, watchdog, executor, or workflow-version types.

## Choose the product topology

| Requirement | Embedded/local | Hosted/multi-tenant |
|---|---|---|
| Process model | One trusted owner process | Product API plus one chosen activation runtime |
| Product interface | Direct Python calls | Authenticated HTTP/RPC plus SSE or equivalent event delivery |
| Tenant isolation | Product process and workspace boundary | Signed principal plus isolated storage, workspace, quotas, and projections |
| Durable inputs | Direct session calls from the owner | Product-routed initial submission plus authenticated, ordered, idempotent post-submission control/callback transport |
| Recovery | Reconstruct one loop from its checked checkpoint | Runtime recovery plus stale-executor fencing and checked checkpoint restore |
| Storage | Local durable filesystem can be sufficient | Shared or runtime-addressable stores selected by the deployment |
| Operations | Process lifecycle and local backups | Admission limits, draining, health, version routing, incident response, and retention |

The embedded path fits a desktop app, appliance, or single worker. The hosted path fits products
with tenant routing, multiple API instances, external workers, callbacks, and durable control.

## Choose one hosted assembly

| Assembly | v0.19.2 position | Operational owner | Recovery scope |
|---|---|---|---|
| Product-owned runtime | Production integration target | The product's scheduler and worker control plane | Defined and qualified by the product |
| Reference inbox assembly | Runnable Reference example and CI-qualified hosted golden path | `RunnerBackend`, `LeaseStore`, `CommandStore`, `RecoveryService`, and watchdog | Shared-store stale-owner claim demonstrated with SQLite |
| Optional DBOS activation-recovery profile | Experimental private Reference composition proof | One private host owns the process-global DBOS runtime and hosted control/run lifecycle | Same `executor_id` and `application_version` after a fenced restart; hosted control and run recovery are acceptance-tested |

The hosted golden path uses owner-local Reference submission together with the durable inbox for
post-submission status, task-result, and approval commands. It also covers event projection and
tenant usage through one executable facade. `submit_run()` has no durable client-submission key;
the product edge owns idempotent initial-submission admission and routes each accepted submission
to the selected owner. Integrators place this Reference fixture behind their own product API.

The optional DBOS profile covers finite control-dispatch and run-resume activations.
`CheckpointStore` remains the run-semantic authority, and each run-resume workflow result copies
its committed boundary receipt. Control workflows return a versioned, credential-sanitized
`CommandReceipt` after dispatch. Private hosted control and run participants share one captured
runtime, listener set, launch, admission, drain, and shutdown lifecycle while retaining distinct
partitioned queues. The exported standalone components remain individual experimental entry
points.

The v0.19.2 scope excludes `RunnerBackend` replacement, hosted-golden-path routing, Studio and
terminal-projection migration, PostgreSQL production qualification, rolling upgrades,
arbitrary-host takeover, and Conductor. See [DBOS_REFERENCE.md](DBOS_REFERENCE.md) for the exact
verified invariant and non-goals.

## Golden path A: embedded/local product

The local example creates an `AgentRunSpec`, explicit `AgentRuntimeConfig`, scripted model adapter,
and `LocalFsCheckpointStore`. `LoopSession` owns the session FSM and commits a checkpoint when the
turn settles.

```python
from monoid_agent_kernel.contracts import AgentLoop, AgentRunSpec, LoopSession
from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore

checkpoints = LocalFsCheckpointStore(run_root)
loop = AgentLoop.from_config(
    AgentRunSpec(workspace_root=workspace, run_root=run_root, mode="apply"),
    adapter,
    runtime_config,
    checkpoint_store=checkpoints,
)
session = LoopSession(loop)
session.open()
session.submit("Create the requested release note.")

loaded = checkpoints.latest_checked(loop.spec.run_id)
if not loaded.ok:
    raise RuntimeError(f"checkpoint load failed: {loaded.status}")

result = session.close()
```

Use this topology when the product process is the sole run owner. Keep these locations separate:

- `workspace_root`: user-visible files the tools may read or change;
- `run_root`: events, status, proposals, checkpoints, and failure artifacts;
- application secrets: an external secret store or in-memory provider outside both roots.

Choose `mode="propose"` when a human or product policy approves file changes. Choose
`mode="apply"` after the product authorizes direct workspace mutation.

The example binds `fs.write` and `run.finish` explicitly. A tool absent from `ToolBinding` stays
outside the model surface. Keep the local surface small and call `AgentLoop.validate()` during
startup when tool configuration comes from users or deployment data.

### Local recovery

Use one durable `CheckpointStore` for the run lifetime. A local filesystem store needs a durable
mount and filesystem semantics that support atomic replace and locking. Read checkpoints through
`latest_checked()` so the product can distinguish `loaded`, `migrated`, `missing`, `corrupt`, and
`unsupported_version` outcomes. The executable example asserts a checked `loaded` outcome.
It also returns `runtime_profile="embedded-local"` so test output names the selected topology.

The bundled LocalFS and SQLite checkpoint stores are single-writer adapters. Their atomic commit
protects checkpoint integrity, and the Reference host serializes each in-process commit with its
activation write authority. A topology where an old writer can resume after another writer takes
over uses a `FencedRunSink` and passes a `WriterToken` into every canonical mutation. The storage
adapter validates owner and generation in the same transaction as the write.

Reconstruct the loop with the same run ID, compatible runtime definition, workspace mapping, and
blob store. Restore the checked checkpoint before accepting a new input. Stop recovery and surface
an actionable failure for corrupt or unsupported state.

### Finite activation hosting

Use `PostgresCommandAdmissionStore` at the product ingress to create one immutable admitted command
and its dispatch outbox row in a single transaction. The caller supplies a stable command ID,
request digest, and opaque private payload address. An exact retry returns the current receipt;
reusing the ID for a different request raises `AdmissionConflict`.

`PostgresConfig.lock_timeout_s` and `statement_timeout_s` apply transaction-local PostgreSQL
limits to every adapter operation. Their defaults are 30 and 300 seconds. Set both below the host's
outer operation timeout so a database lock or statement aborts before it can retain a worker slot.
`pool_timeout_s` independently bounds connection-pool acquisition.

```python
from monoid_agent_kernel.adapters.postgres import PostgresCommandAdmissionStore
from monoid_agent_kernel.hosting import AdmissionRequest, CommandOutboxDispatcher

admission = PostgresCommandAdmissionStore(postgres_database)
admission.check_ready()
receipt = admission.admit(
    AdmissionRequest(
        run_id=run_id,
        command_id=command_id,
        kind="input",
        request_digest=request_digest,
        payload_ref=private_payload_ref,
    )
)
dispatcher = CommandOutboxDispatcher(
    store=admission,
    transport=orchestrator_transport,
    owner_id=dispatcher_id,
)
dispatcher.dispatch_once()
```

`dispatch_once()` performs one finite claim/send/settle attempt. The host owns polling, threads,
shutdown, credentials, and health reporting. Dispatch uses database-clock leases and preserves
per-run command order across competing workers. Delivery is at least once: the transport deduplicates
`AdmittedCommand.identity_sha256`, and the activation path applies the immutable command identity
once. `lease_s` must be in the portable `(0, MAX_COMMAND_DISPATCH_LEASE_S]` range, whose maximum is
86,400 seconds. Retry policies return a finite non-negative delay; the dispatcher caps it at the
portable `MAX_COMMAND_RETRY_DELAY_S` value of 86,400 seconds before store settlement. A rejected
command enters `dead_letter` and blocks later commands in that run until an operator
resolves the lane. A canonical terminal winner excludes that run from new claims. Unbound pending,
leased, or delivered commands converge to `run_terminal` without another transport call. Active
claim finalization and settlement share the run-authority lock with terminal selection, so commit
order decides the winner and no claim, acknowledge, retry, or reject mutation can follow the
terminal commit. `receipt()` reads admission, activation, and terminal evidence from one PostgreSQL
statement snapshot, preserving monotonic state projection across concurrent binding and terminal
commits.

After the orchestrator selects a delivered command, acquire the current `WriterToken` and call
`bind_activation()`. Binding captures the latest checked checkpoint exactly once at activation time.
This allows several commands to be admitted before an earlier command advances the checkpoint.
The store requires every preceding command sequence to have its canonical checkpoint receipt before
binding the next command. Concurrent or out-of-order Activity delivery therefore cannot create two
bindings from the same source checkpoint. Replacement workers receive the same stored
`ActivationCommand`; stale writers are fenced before readback.

Use `monoid_agent_kernel.hosting.ActivationDriver` when an external scheduler, workflow engine, or
queue worker owns process replacement. The host admits one stable `ActivationCommand`, claims a
`WriterToken`, and drives the restored loop to one durable suspension boundary:

```python
from monoid_agent_kernel.hosting import ActivationCommand, ActivationDriver

command = ActivationCommand.from_json(admitted_command)
receipt = ActivationDriver(
    sink=fenced_run_sink,
    writer_token=writer_token,
    loop_factory=build_loop,
    input_resolver=resolve_private_input,
).drive(command)
```

The command contains digests and an opaque payload address. For an `input` command, the resolver
loads private content and returns `ResolvedActivationInput` whose request digest and payload address
match the admitted command. A `control` command resumes host-prepared checkpoint state without
injecting another user message. A duplicate command whose marker is already in the canonical
checkpoint returns the same content-free `ActivationReceipt` without resolving input or opening a
loop. The checkpoint keeps that command's original boundary sequence and digest inside its private
receipt, so later commands may advance the head without changing an earlier command's returned
receipt. The digest hashes the boundary checkpoint with that receipt's own digest field blanked,
which avoids a self-reference while retaining the rest of the exact boundary identity.

The loop factory receives an `ActivationRuntime` and binds its exact `run_sink`, `writer_token`,
`write_authority`, and `cancellation_token`. It also configures
`authoritative_event_sinks=(runtime.event_sink,)`, seeds `event_sequence_seed` from the runtime, and
keeps `emit_output_deltas=False` until a durable private stream sink is configured. The resulting
event order is durable journal first, then local projections. A terminal winner closes new public
event coordinates while preserving exact retries of events committed before terminal settlement.

Treat `ActivationReceipt` as an operational copy. Reconstruct it from the canonical checkpoint and
terminal readback after a response loss. Store model output and raw provider errors only in the
private checkpoint/blob channels referenced by the receipt.

### Temporal run orchestration

Install `monoid-agent-kernel[durable-host]`, check every storage adapter for readiness, and compose
the versioned per-run Workflow with the production threaded Activity:

```python
from monoid_agent_kernel.adapters.temporal import (
    TemporalRunPolicy,
    TemporalSignalWithStartTransport,
)
from monoid_agent_kernel.adapters.temporal.activity import (
    TemporalActivationActivity,
    TemporalActivityPolicy,
)
from monoid_agent_kernel.adapters.temporal.worker import TemporalWorkerGroup

authority_store.check_ready()
admission_store.check_ready()
fenced_run_sink.check_ready()

policy = TemporalRunPolicy(activity_task_queue="monoid-activation-v1")
transport = TemporalSignalWithStartTransport(
    client=temporal_client,
    event_loop=temporal_client_loop,
    workflow_task_queue="monoid-run-v1",
    run_policy=policy,
)

activation = TemporalActivationActivity(
    authority_store=authority_store,
    admission_store=admission_store,
    run_sink=fenced_run_sink,
    loop_factory=build_loop,
    input_resolver=resolve_private_input,
    policy=TemporalActivityPolicy(
        writer_lease_ttl_s=30,
        writer_lease_renew_interval_s=10,
        heartbeat_interval_s=5,
        authority_call_timeout_s=30,
        driver_call_timeout_s=3300,
        supervisor_join_timeout_s=30,
        local_task_wait_s=300,
    ),
)

workers = TemporalWorkerGroup(
    client=temporal_client,
    workflow_task_queue="monoid-run-v1",
    activity_task_queue=policy.activity_task_queue,
    activation_activity=activation,
    max_concurrent_activities=10,
    graceful_shutdown_timeout_s=30,
)

async with workers:
    await serve_until_shutdown()
```

`TemporalActivationActivity.run` is registered as the exported
`TEMPORAL_DRIVE_ACTIVATION_ACTIVITY` name with
`no_thread_cancel_exception=True`. It accepts an `AdmittedCommand.to_json()` payload and returns
`TemporalActivationResult.to_json()`. The result binds the exact command identity to a canonical
receipt ref and terminal flag. Database access, private payload resolution, provider and tool calls,
checkpoint restore, and terminal settlement remain inside this finite Activity.

The Activity derives a content-free owner ID from the Temporal task token, claims an independent
PostgreSQL writer generation, and starts a copied-context control supervisor before the potentially
blocking writer claim. The control supervisor sends empty heartbeats, observes cancellation and
worker shutdown, and enforces a conservative monotonic deadline derived from PostgreSQL lease
evidence. Claim, initial exact-token renewal, and durable activation binding run in one bounded
copied-context daemon bootstrap worker, so a stuck database lock cannot retain the Temporal
Activity executor slot. The configured authority timeout is capped by the current Activity
attempt's remaining start-to-close budget with cleanup reserve. A timed-out bootstrap cannot enter
the driver and releases a late exact token. An independent
renewal thread performs later PostgreSQL calls, so pool or row-lock waits cannot stop heartbeat or
deadline enforcement. Its first renewal is scheduled from the installed lease's remaining
monotonic budget and runs immediately when bootstrap consumed the normal safety margin. Control
propagation uses the exact
`ActivationRuntime.cancellation_token`. PostgreSQL remains the mutation authority. A heartbeat,
renewal ambiguity, or local lease deadline revokes `ActivationWriteAuthority`, and every later
checkpoint, invocation, event, and terminal publication fails closed at the PostgreSQL fence.
The `ActivationDriver` runs in a copied-context daemon worker. `driver_call_timeout_s` bounds that
worker and is further capped by the current Activity attempt's remaining start-to-close budget,
with time reserved for cleanup. Timeout or supervisor loss revokes local write authority, cancels
the exact activation token, ignores a late driver result, and returns the Temporal Activity
executor slot. Configure PostgreSQL lock and statement timeouts below this driver bound so an
in-flight fenced mutation aborts inside the database first. The control supervisor keeps
heartbeating while one bounded copied-context cleanup task joins driver and renewal work and
releases the exact writer token. `supervisor_join_timeout_s` bounds that combined cleanup; expiry
returns retryable lease loss and leaves any uncooperative cleanup thread daemonized under revoked
local authority.
A lost claim response is reconciled inside the same Activity attempt with the same unique owner and
an exact-token read. A competing owner delays the next Temporal attempt by the lease interval
observed by PostgreSQL, so short exponential retry backoffs do not exhaust attempts before expiry.
A writer fence observed during activation binding is retryable lease loss. A deterministic loop
wiring violation is a non-retryable configuration conflict.

Keep `heartbeat_interval_s` below the Workflow's `activity_heartbeat_timeout_s`; runtime caps the
effective interval to half of the actual Activity heartbeat timeout. Keep
`authority_call_timeout_s`, `driver_call_timeout_s`, and the cleanup reserve within its
`activity_start_to_close_timeout_s`; runtime caps both bootstrap and driver phases to the actual
remaining attempt budget. Keep PostgreSQL `pool_timeout_s`, `lock_timeout_s`, and
`statement_timeout_s` below the relevant Activity bound. The Activity policy requires the writer
lease TTL to cover at least two renewal intervals. A start-to-close timeout that cannot contain the
configured cleanup reserve fails as a non-retryable configuration conflict before writer claim.
Give
`graceful_shutdown_timeout_s` enough time for AgentLoop to reach and commit a safe boundary. Worker
composition requires this timeout to cover the configured heartbeat interval, authority call
timeout, and supervisor join window. Shutdown maps to `graceful_drain` by default; set
`worker_shutdown_cause=InterruptionCause.HOST_SHUTDOWN` when an orderly host termination should
retain that distinct cause.

`TemporalWorkerGroup` creates an Activity executor sized to `max_concurrent_activities` by default.
An externally owned executor must be active and expose at least that many worker threads. Reserve
that capacity for Activity work; unrelated tasks in a shared pool can still delay the initial
heartbeat and lease claim. After the Temporal graceful-drain window, an owned executor stops
accepting work and cancels queued futures without joining a stuck running call. This keeps group
exit bounded and leaves final process termination to the host supervisor. Externally owned
executors retain their host-managed lifecycle.

A drain before provider entry commits a resumable `graceful_drain` receipt. A drain after durable
`dispatch_started` and before trustworthy provider evidence commits `dispatch_unknown` with
`after_reconciliation` eligibility. This result blocks automatic paid-call replay. A worker crash
after the settled invocation commit lets the replacement generation reuse the stored result with
zero additional provider calls.

The Temporal client is asynchronous. Keep its owner event loop running and call the synchronous
`transport.dispatch()` from the PostgreSQL outbox polling thread. Async code already executing on
the owner loop calls `await transport.dispatch_async(command)`. The synchronous method rejects an
owner-loop call because waiting there would deadlock the client. Both methods return after Temporal
server acceptance; the PostgreSQL admission receipt remains the client-facing canonical handle.

Signal-With-Start uses a deterministic digest-derived Workflow ID and targets an existing running
Workflow when present. A lost response can produce another Signal. The Workflow compares the
PostgreSQL sequence and immutable admitted-command identity, buffers future sequences, and schedules
one Activity for each sequence. A closed Workflow ID rejects another start, allowing the outbox lane
to enter an operator-visible terminal/dead-letter disposition.

The Workflow checks Temporal's Continue-As-New suggestion after every completed Activity. It waits
for Signal handlers, transfers no in-flight Activity, and carries ordered pending commands, the next
sequence, latest receipt ref, policy, build, and operational counters into the new Run ID. Keep
`history_rollover_command_limit=0` in production to follow the service suggestion. A small positive
value is a deterministic qualification hook.

Temporal history and Query status contain opaque IDs, digests, refs, sequence counters, bounded
policy values, and public-safe taxonomies. Store prompt, response, reasoning, model result,
workspace bytes, checkpoint bytes, credentials, and raw exception text in private storage. Replay
`tests/fixtures/temporal_replay_v1/run-workflow-v1.json` with Temporal `Replayer` before changing the
Workflow command sequence or Activity options. Use Temporal patching or Worker Versioning for a
change that would produce different commands for an existing history.

## Golden path B: hosted/multi-tenant product

The hosted example creates two `RunnerBackend` instances over one SQLite database. This diagram is
the selected Reference fixture, rather than a portable deployment mandate:

```text
authenticated product submit router ---- owner-local submit_run() -----+
                                                                        |
callback/control worker ---> peer RunnerBackend ---> durable inbox -----+
                                                                        v
                                                     owner RunnerBackend (Reference)
                                                                        |
                                                          AgentLoop / CheckpointStore
                                                                        |
                                                          events / receipts / usage
```

Each backend receives separate `SqliteCheckpointStore`, `SqliteLeaseStore`, and
`SqliteCommandStore` objects pointing at the same database. The owner watchdog publishes and
heartbeats its lease, drains commands, redrives due outbox work, and uses atomic stale-owner claims
for Reference recovery.

The example submits runs for `tenant_a/user_a` and `tenant_b/user_b`. Every command ingress checks
the signed subject against durable run metadata. The command store persists a sanitized principal
and token SHA-256 for audit attribution. The bearer remains transient. The owner mints a fresh,
short-lived execution token after claiming a cross-instance command. The example also attempts to
address tenant A's run with tenant B's token and requires authorization to fail before append.

The script returns `runtime_profile="reference-inbox"` to make this composition choice observable.
It imports Reference types through `monoid_agent_kernel.reference.backend` and
`monoid_agent_kernel.reference.stores`; product code keeps those imports inside its composition
root. It performs checked reads of each run's checkpoint and durable metadata through a fresh
shared-store handle and reports both `loaded` outcomes.

### Hosted ownership checklist

| Owner | Required behavior |
|---|---|
| Product edge/API | Authenticate callers; bind tenant, user, run, audience, and callback scope; apply admission limits. |
| Activation runtime | Serialize mutating inputs per run, fence stale execution, retry safely, drain on shutdown, and expose operational health. |
| Semantic store | Atomically publish the last good checkpoint and blobs; preserve typed checked-load outcomes. |
| Durable input transport | Deduplicate stable input IDs, preserve order, sanitize credentials, and return identity-bound receipts. |
| Event projection | Preserve sequence ownership, gap detection, terminal drain, authorization, and reconnect cursors. |
| Gateway and broker | Hold provider/capability secrets, narrow scope, enforce quotas, and return sanitized observations. |
| Approval service | Preserve task identity and callback scope; record the decision actor and reason. |

The Reference inbox assembly realizes the activation and input rows with leases, a transactional
command store, a recovery service, and a watchdog. A product-owned scheduler can realize the same
obligations through different storage and recovery mechanisms.

### Choose the model evidence delivery policy

Hosted products that configure `AgentLoop(run_sink=..., writer_token=...)` also choose
`model_evidence_policy`:

- `passive` preserves the existing observer and sidecar behavior. Export failure does not alter the
  model result.
- `required` makes the public-safe evidence projection a checked post-settlement mutation. An
  evidence failure parks as `evidence_uncommitted`; resume the same checkpoint without a new user
  input. The loop reloads the settled invocation and retries the sink without calling the provider.
- `outbox` asks the sink to stage evidence in the invocation-settlement transaction. Enable it only
  on a sink that declares `transactional_outbox=True`. The host runs the sender and its retry or
  dead-letter policy.

Keep invocation settlement and evidence delivery in separate tables or record families for
`required`. Make `commit_model_evidence()` fence-first, idempotent on
`(run_id, logical_call_id, revision)`, and valid only for the current authoritative settled
revision. Persist the invocation's `evidence_policy` enum in the same transaction as every
invocation revision and preserve it across retries. A replacement worker must honor the journal
policy even when its configured policy is `passive`; this covers a crash before an
`evidence_uncommitted` checkpoint exists. Reject a `passive` to `required` policy change for an
existing logical call before provider or evidence mutation. Apply the stronger policy to a new
logical call. On recovery, deliver a journal-required settlement before validating a request digest
rebuilt from replacement config or dynamic context. For `outbox`, reject the complete transaction
when either
the invocation revision or the outbox entry cannot commit. A partial invocation-only commit
violates the declared capability. A recovered outbox reservation remains outbox-owned, checks
`transactional_outbox` before provider entry, and stages evidence at settlement even when the
replacement activation uses the passive default.

Treat a durable checkpoint whose `last_suspension` is null and whose model-step counter is positive
as an in-progress internal checkpoint. Resume that allocated step once before advancing counters.
The lifecycle then probes the same logical-call journal address, delivers any required evidence,
and only enters a new provider dispatch when the head is missing. This rule covers a crash after an
approval-replay safety checkpoint and after provider settlement but before an evidence-failure park.

On `evidence_uncommitted`, persist the returned checkpoint before releasing the worker. Redrive it
with the same run ID and a current writer token. The loop commits evidence from the stored logical
call ID and request digest, then applies the stored success or final refusal before consulting the
current request-building configuration. A recovered final settles immediately. A recovered
tool-call turn enters the message log before the current tool surface is resolved for execution.
Stored outcomes therefore survive runtime-config changes. Passive observers and enabled
model-call sidecars receive the authoritative call at the original settlement even when required
evidence parks afterward; recovery does not duplicate that passive publication. Live model-stream
observers and the private content sidecar close a settled provider success as `completed` with its
final text and usage while the run separately parks for evidence recovery. The checkpoint
marker and the authoritative invocation flag each force required delivery on the replacement
activation even when its configured policy is `passive`. One-shot `run_once()` releases this
committed park and raises `TurnNotSettled`; restore
the checkpoint and resume with `None`. Treat
`dispatch_unknown` separately: reconcile the journal or provider before any new paid call.
The kernel accounts an authoritative settled receipt before a stop or deadline can park its result
as unapplied. Persisted interruption checkpoints therefore include the paid usage even when the
assistant turn remains absent; resume projects the stored result without adding that usage again.
Lease loss uses the stricter ownership boundary. The runner rechecks authority immediately after
durable settlement and suppresses stale usage accounting, metrics, passive model-I/O delivery,
model-call sidecars, and model-stream completion. The replacement owner recovers the stored receipt
and publishes it without another provider dispatch.
Treat `CancellationToken.cause` as first-cause execution-control history. Inject one
`ActivationWriteAuthority` into AgentLoop and every activation-owned adapter. Revoke that authority
when lease ownership moves. Revocation wakes the loop while preserving an earlier Stop or drain
cause, and the stale activation returns an ephemeral `lease_lost` disposition. Required-evidence
recovery checks the authority before and after its lifecycle hook. Token-based deadlines use the
same `run_timeout` terminal projection as elapsed wall-clock deadlines.

Keep the host's `WriterToken(run_id, owner_id, generation)` beside the process-local authority.
Checkpoint, invocation, canonical event, and terminal adapters validate that token atomically with
each durable mutation. When any adapter reports `fenced`, revoke the shared authority immediately.
Host adapters also fence or deduplicate external shell, MCP, memory, and custom effects that can
continue after an activation loses authority.
Revocation synchronously disables the recorder's private model-content store. Stale activation
cleanup cancels pending flush timers, drops buffered deltas, and closes its handle without appending
segments or a stream terminal record.
Ordinary cleanup while authority remains active cancels hosted tasks created after the last
committed checkpoint and writes their normal cancel marker. It preserves hosted tasks already
owned by that checkpoint. Revoked cleanup releases in-process handles only and delegates hosted
task cancellation to the fenced/idempotent host adapter.
Evidence recovery surfaces stored retryable refusals without consuming a remaining kernel attempt.
Let the driver decide whether to start a later model step.
Evidence recovery is a commit barrier for a model step that already settled. The runner completes
the fenced settlement/evidence mutation before applying a pending cancellation or expired deadline.
An internal safety checkpoint taken after allocating step `N` and before a park restores step `N`
once. Recovery queries that journal coordinate before incrementing either the run step or the
submit-local step, so a crash before evidence delivery cannot strand the settled invocation behind
step `N+1` or start a second paid call.
An interrupt can still park before the recovered result is applied; a `None` resume reloads that
same logical call with its checkpointed tool observations, while a new user input intentionally
abandons the interrupted result. An interrupt after a recovered assistant tool-call turn is added
to the message log persists `model_tool_calls_pending=true` and every completed observation. Resume
with `None`: the loop reloads the settled result, skips completed call IDs, and executes the rest
without adding a second assistant turn. The loop rejects new user input with
`evidence_recovery_requires_resume` until this tool exchange completes. Give effectful tool handlers
stable idempotency keys because a process can still disappear after an external effect and before
its observation is returned. The interruption checkpoint restores the kernel-owned plan, pending
`run.finish` value, and pending `tool.search` loads before completed call IDs are skipped.

## Model and tool wiring

The offline examples inject a fake adapter, so no gateway is contacted. A hosted deployment places
provider credentials in its model gateway and passes only a short-lived, run-scoped gateway token
to the runner. The gateway validates tenant, user, run, audience, kind, expiry, model policy, and
quota before injecting the provider credential at the final boundary.

Build the tool surface from explicit bindings:

```python
from monoid_agent_kernel.contracts import AgentRuntimeConfig, ToolBinding

config = AgentRuntimeConfig(
    definition_id="support-agent-v3",
    tools=(
        ToolBinding.for_tool("fs.read"),
        ToolBinding.for_tool("run.finish"),
    ),
)
```

Product tools implement `ToolProvider` and use stable tool IDs. Declare authorization, side-effect
delivery, scope, quota, and capability requirements. Async handlers implement the async contract
directly; synchronous handlers run through the bounded synchronous boundary. See
[TOOL_SURFACE.md](TOOL_SURFACE.md).

## Checkpoints, recovery, and upgrades

Every hosted assembly follows these portable rules:

1. Read durable state through checked codecs and stop on corrupt or unsupported records.
2. Fence competing or stale activations before applying a new input.
3. Commit the input identity and its semantic receipt with the checkpoint boundary.
4. Reconstruct compatible runtime, workspace, gateway, task, and policy dependencies.
5. Give each external effect a replay-stable idempotency identity or durable outbox record.
6. Preserve terminal state and its canonical receipt independently of rebuildable projections.

For the Reference inbox assembly, every instance shares durable checkpoint and lease stores plus
one transactional command store. A fresh backend can call `recover_runs()` and start its watchdog
after an atomic stale-owner claim once the previous writer has stopped. Queue limits and claim TTLs
bound durable command admission and recovery. The bundled SQLite composition remains a
single-host, single-writer Reference fixture.

For the experimental DBOS profile, the private host is the sole DBOS lifecycle authority in one
process. A supervisor fences the previous process and restarts the same stable executor slot with
the same application version. DBOS resumes pending finite activations; the checked checkpoint
remains canonical for semantic state. The profile contains no Reference lease, command inbox,
recovery service, or watchdog lifecycle.

Every durable family has a versioned codec and compatibility-ledger entry. Upgrade in this order:

1. deploy readers that accept old and new schemas;
2. verify mixed-version conformance;
3. enable new writers;
4. monitor unsupported and corrupt load outcomes;
5. retire old readers after the rollback window closes.

Run `python -m pytest -q tests/test_compatibility_ledger.py`, then follow the mixed-version and
rollback procedures in [COMPATIBILITY.md](COMPATIBILITY.md). DBOS workflow inputs, results, executor
identity, and application-version operation remain governed by the experimental profile document;
v0.19.2 makes no production rolling-upgrade claim for that profile.

## Streaming and cursor ownership

### Durable private model streams

Use `PostgresObjectStoreDurableStreamStore` when reconnect must survive process replacement. The
store keeps generation, UTF-8 byte cursor, chunk metadata, and the final digest in PostgreSQL. A
`ContentAddressedBlobStore` keeps immutable private chunk bytes. The host supplies the same exact
`WriterToken` and activation-wide `ActivationWriteAuthority` used by checkpoint, invocation,
event, and terminal publication:

```python
from monoid_agent_kernel.adapters.postgres import PostgresObjectStoreDurableStreamStore
from monoid_agent_kernel.hosting import DurableModelStreamObserver

streams = PostgresObjectStoreDurableStreamStore(database, object_store)
streams.check_ready()

loop = make_loop(
    model_stream_observer_factories=(
        lambda: DurableModelStreamObserver(
            streams,
            writer_token=writer_token,
            write_authority=write_authority,
            chunk_bytes=64 * 1024,
            flush_interval_s=0.25,
            max_buffer_bytes=1024 * 1024,
        ),
    ),
)
```

The observer opens one `output` lane and normally opens `reasoning` when that channel first emits
content. A replacement that finds prior output also hydrates reasoning. `ModelCallRunner` signals
every actual adapter entry before durable `dispatch_started` publication. The observer resets every
pre-existing kernel lane at that boundary. A reset failure leaves the invocation `reserved` and
prevents provider entry. Provider-free settled success or failure recovery receives no dispatch
signal and preserves the committed generation. Direct `ModelCallRunner` integrations pass a
`before_dispatch` callback that calls `begin_model_stream_dispatch()`.

The loop also passes a `before_settlement` callback that calls
`prepare_model_stream_settlement()`. The durable observer flushes all accepted output and reasoning
bytes before the lifecycle publishes a recoverable success or refusal. The generation stays open
until ordinary close seals it. A crash after invocation settlement can therefore recover the full
generation and seal it without reconstructing reasoning. A preparation failure commits
`dispatch_unknown`, invokes `abort_model_stream()`, and leaves the generation unsealed for
diagnosis. The first-delta reset remains a fallback for direct integrations that omit the dispatch
callback. Host-defined private lanes use the same `DurableStreamIdentity` contract directly. Byte
and time thresholds bound coalescing; a copied-context daemon performs ordered flushes. Generic
observer factory, open, push, and close failures stay isolated. Dispatch- and settlement-aware
extensions opt into fail-closed preparation. A `fenced` store result revokes the shared activation
authority, so later kernel publication fails closed.
The durable observer derives its store address with `durable_model_stream_id(run_id, turn_id)` and
leaves `ModelStreamContext.stream_id` execution-unique for legacy sidecars. A recovered completed
call reuses and seals its prior generation; the first delta from an admitted replacement dispatch
cannot mix with the prior generation because reset already committed before adapter entry.
On a completed close, the observer compares the output lane's byte length and SHA-256 with the
settled `final_text`. A mismatch means recovery found an unflushed/truncated prefix; the observer
rebuilds output in a new generation from that authoritative final text before sealing it.

Persist reconnect state as `(generation, cursor)`. Call `read_after()` only with cursor values
returned by a prior read. `ok` returns complete UTF-8 chunks and `next_cursor`; `reset` tells the
consumer to discard the old generation and restart from cursor zero; `gap` reports an ahead or
non-boundary cursor. A stable reset ID makes process-response loss idempotent. Seal records the
final byte length and SHA-256 after checked ObjectStore reads. PostgreSQL serializes append, reset,
seal, and terminal settlement through the run authority row. An append committed before terminal
remains readable; a new append ordered after terminal returns `run_terminal`.

Each append holds the run-authority and stream-head locks through one bounded ObjectStore put;
terminal settlement, reset, takeover, and renewal then linearize after that chunk commit. Configure
ObjectStore request timeouts, lease TTL/renewal margin, and `supervisor_join_timeout_s` as one
operational budget. Seal releases those locks before its multi-chunk checked-read pass and validates
the captured head again before publication.

Stream metadata contains opaque IDs, channel taxonomy, offsets, sizes, and digests. It contains no
model text. `read_after()` returns private bytes and performs no tenant authorization; expose it
only through a product-owned authenticated projection. Derive globally unique kernel run IDs or
apply tenant scoping in that wrapper.

Use the `EventSubscription` and `SequenceCursor` contracts for reusable polling or frame iteration.
The cursor stores the next required sequence, suppresses replayed events, and raises on a gap. The
Reference facade exposes the same behavior through `subscribe_events()`:

```python
subscription = backend.subscribe_events(run_id, run_token, from_seq=1)
page = subscription.poll(limit=500)
next_seq = page["next_seq"]
```

For SSE, send each event sequence as `id`, accept `Last-Event-ID` on reconnect, and resume at the
next sequence. Heartbeats carry no event ID. A terminal subscription drains the event log once more
before emitting its end frame. Persist the cursor in the consuming product when delivery must
survive client restarts.

The hosted golden path polls twice through one subscription and asserts that the second page is
empty. It then constructs a fresh Reference backend and reads the durable event history with the
original authorized run token.

## Control commands, tasks, and approvals

`ControlCommand` and `ControlDispatcher` form the transport-independent control contract. A hosted
runtime supplies authenticated durable delivery, per-run ordering, stable input IDs, idempotent
receipts, and queue limits. Product status normally comes from an authenticated read projection.
The hosted fixture deliberately sends one `status` command through the inbox to exercise the
accepted durable command transport.

The Reference example uses:

```python
receipt = peer.enqueue_control(
    ControlCommand(
        type="status",
        run_id=run_id,
        args={"token": run_token},
        issuer="product-api",
        reason="operator requested refresh",
        command_id="product-command-01842",
    )
)
```

An identical duplicate returns the existing receipt. Reusing an ID with a different sanitized
payload, command type, principal, issuer, or reason returns `command_id_conflict`. The Reference
store executes commands in append order with one claimed head per run. A replacement Reference
owner can reclaim a stale unacknowledged head.

`create_task` returns a callback token in the immediate owner response. Give that token to the
specific external worker. The worker can call `report_task_result`, `approve`, or `deny` within its
task scope and poll the matching receipt. Durable receipts redact newly issued callback secrets.
A lost `create_task` response requires an intentional new task because the secret is absent from
durable storage.

For tool approvals, render the sanitized durable task request, preserve the task ID, and submit the
decision through the callback-scoped path. Record the approver and reason. Execute the original
durable request after approval.

The exported `DbosControlPlane` is an isolated transport experiment outside the hosted golden
facade. Its private hosted form composes with the private hosted run participant under one
Reference runtime host. Public product routing across those participants sits outside the v0.19.2
proof.

## Gateway credentials and security

Apply these boundaries:

- expose backend, model gateway, Web gateway, and MCP gateway on distinct audiences;
- keep admin capabilities on private operational routes;
- issue short TTLs and rotate signing keys through an overlap window;
- validate token kind, audience, run ID, tenant, user, task metadata, and expiry;
- keep provider keys and capability secrets in a gateway, broker, or secret manager;
- persist hashes or opaque handles in checkpoints, events, commands, receipts, and logs;
- give Python tool extensions only the method-only `ToolContext` façade; keep workspace roots,
  native paths, recorder/task handles, lifecycle state, and counters behind the trusted kernel edge;
- restrict workspace roots and resolve paths through the `Workspace` contract;
- isolate run artifacts, projections, quotas, and retention by tenant;
- run the production checklist and relevant conformance profiles before external traffic.

The hosted golden path scans its run root and SQLite database for run, callback, and observed model
gateway bearers plus the signing secret after all connections close. An unreadable durable file or
credential match fails the test.

## Observability

Consume lifecycle, model, tool, control, task, usage, and failure events by sequence. Attach an
`EventSink` for telemetry and keep exporter failures outside the run state machine. Correlate
product requests with `run_id`, stable input or `command_id`, event sequence, and trace context.
Exclude raw prompts, tool secrets, bearer tokens, and unreviewed model output from routine logs.

Publish portable metrics for:

- active runs by tenant and semantic lifecycle state;
- activation admission, attempt, retry, completion, and fencing outcomes;
- checkpoint commit latency and checked-load status;
- durable input depth, age, conflicts, failures, and receipt latency;
- event cursor lag and sequence-gap failures;
- gateway usage, retries, quota denials, and provider latency;
- approval age and callback completion latency;
- drain, shutdown, recovery, and terminal projection outcomes.

The Reference inbox assembly additionally reports lease age, stale claims, watchdog recovery, and
command claim age. A DBOS evaluation reports workflow, queue, executor-slot, and application-version
state as operational diagnostics.

## Failure handling matrix

| Failure | Portable required behavior | Selected Reference evidence |
|---|---|---|
| Model timeout or retryable provider error | Apply bounded retry policy and emit the final typed error. | Fake and gateway adapter tests cover typed outcomes. |
| Tool cancellation | Propagate cancellation, bound cleanup, and preserve the last good checkpoint. | Async tool contract tests cover cancellation and cleanup. |
| Checkpoint interruption | Publish atomically and reconcile ambiguous outcomes through checked readback. | LocalFS/SQLite fault-matrix tests cover interrupted publication. |
| Corrupt or future durable schema | Return a typed load outcome and stop recovery for that record. | Compatibility and durability tests assert `corrupt` and `unsupported_version`. |
| Competing or stale executor | Fence stale execution before a semantic commit. | Reference inbox uses lease CAS; DBOS proof uses active-input ownership and stable-slot fencing. |
| Duplicate input ID with changed payload | Reject the identity conflict. | Reference inbox returns `command_id_conflict`. |
| Activation crash after an external effect | Redrive with the same stable effect identity or durable outbox record; the target deduplicates repeated delivery. | The DBOS proof uses an explicitly idempotent target; Reference inbox tests separately prove command receipt deduplication. |
| Event client disconnect | Resume from the cursor, suppress replays, and fail on gaps. | Event subscription tests cover SSE IDs and terminal drain. |
| Lost task-secret response | Keep the secret absent from durable state and require explicit replacement. | Hosted golden path scans durable files for callback bearers. |
| Gateway credential expiry | Mint a new scoped gateway token and keep provider keys at the gateway. | Gateway contract and token tests cover expiry and scope. |
| Telemetry exporter failure | Drop, buffer, or retry telemetry without changing run semantics. | Event sink boundaries isolate exporter failure. |
| Required model-evidence delivery failure | Commit the settled invocation, publish passive call records once, park as `evidence_uncommitted`, and redrive the same checkpoint. | Fenced recovery retries only `commit_model_evidence`; it applies the stored outcome before current request-building state and keeps provider and passive-record counts fixed. |
| Transactional evidence outbox failure | Publish neither the settlement nor the outbox entry; close paid-call ambiguity through `dispatch_unknown`. | `transactional_outbox` capability gate and atomic-stage recovery tests cover the boundary. |

## Conformance and release gate

Before production traffic:

1. run both embedding golden paths offline;
2. run fast, contract, serial integration, cross-platform, and install-smoke CI tiers;
3. run the external `minimal-agent` profile against each product implementation, then call
   `run_checkpoint_store_contract(factory, root)` and `run_capability_broker_contract(factory)`
   directly for each replacement store or broker;
4. run `run_fenced_run_sink_contract(factory)` against the canonical storage transaction and lease
   boundary used by every host with overlapping or replaceable workers;
5. execute the durability fault matrix for the selected checkpoint, activation, input, and effect
   paths;
6. verify the compatibility ledger and package contents from the built wheel;
7. rehearse fenced recovery, gateway outage, cursor reconnect, drain, rollback, and credential
   rotation for the selected runtime;
8. complete [security/PRODUCTION_CHECKLIST.md](security/PRODUCTION_CHECKLIST.md).

For the Reference inbox fixture, run `tests/test_backend_command_inbox.py` and its shared-store
recovery tests. For DBOS evaluation, install `reference-dbos`, run the owned-runtime, shared-host,
hosted-control, hosted-run, and process-restart suites, and apply the scope limits in
[DBOS_REFERENCE.md](DBOS_REFERENCE.md).

`tests/test_examples.py` imports and executes both golden paths. Contract, Helper Kit, and Reference
facade drift therefore fails CI at the same integration points shown here.
