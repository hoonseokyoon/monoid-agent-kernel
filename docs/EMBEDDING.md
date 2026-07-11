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

| Assembly | v0.18 position | Operational owner | Recovery scope |
|---|---|---|---|
| Product-owned runtime | Production integration target | The product's scheduler and worker control plane | Defined and qualified by the product |
| Reference inbox assembly | Runnable Reference example and CI-qualified hosted golden path | `RunnerBackend`, `LeaseStore`, `CommandStore`, `RecoveryService`, and watchdog | Shared-store stale-owner claim demonstrated with SQLite |
| Optional DBOS activation-recovery profile | Experimental Reference recovery proof | DBOS owns finite activation admission, serialization, retry, and same-slot workflow recovery | Same `executor_id` and `application_version` after a fenced restart |

The hosted golden path uses owner-local Reference submission together with the durable inbox for
post-submission status, task-result, and approval commands. It also covers event projection and
tenant usage through one executable facade. `submit_run()` has no durable client-submission key;
the product edge owns idempotent initial-submission admission and routes each accepted submission
to the selected owner. Integrators place this Reference fixture behind their own product API.

The optional DBOS profile covers one finite resume activation. `CheckpointStore` remains the
semantic authority, and the DBOS workflow result copies the committed boundary receipt. Its v0.18
scope excludes `RunnerBackend` replacement, Studio integration, terminal artifact projection,
PostgreSQL production qualification, rolling upgrades, and arbitrary-host takeover. Conductor is
outside this profile. The isolated DBOS control experiment and run driver also remain separate.
See [DBOS_REFERENCE.md](DBOS_REFERENCE.md) for the exact verified invariant and non-goals.

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

Reconstruct the loop with the same run ID, compatible runtime definition, workspace mapping, and
blob store. Restore the checked checkpoint before accepting a new input. Stop recovery and surface
an actionable failure for corrupt or unsupported state.

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
after an atomic stale-owner claim. Queue limits and claim TTLs bound durable command admission and
recovery. The bundled SQLite composition remains a single-host Reference fixture.

For the experimental DBOS profile, a supervisor fences the previous process and restarts the same
stable executor slot with the same application version. DBOS resumes pending finite activations;
the checked checkpoint remains canonical for semantic state. The profile contains no Reference
lease, command inbox, recovery service, or watchdog.

Every durable family has a versioned codec and compatibility-ledger entry. Upgrade in this order:

1. deploy readers that accept old and new schemas;
2. verify mixed-version conformance;
3. enable new writers;
4. monitor unsupported and corrupt load outcomes;
5. retire old readers after the rollback window closes.

Run `python -m pytest -q tests/test_compatibility_ledger.py`, then follow the mixed-version and
rollback procedures in [COMPATIBILITY.md](COMPATIBILITY.md). DBOS workflow inputs, results, executor
identity, and application-version operation remain governed by the experimental profile document;
v0.18 makes no production rolling-upgrade claim for that profile.

## Streaming and cursor ownership

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

The v0.18 `DbosControlPlane` is an isolated transport experiment. It does not supply the hosted
facade used by this golden path and does not compose with `DbosRunDriver`.

## Gateway credentials and security

Apply these boundaries:

- expose backend, model gateway, Web gateway, and MCP gateway on distinct audiences;
- keep admin capabilities on private operational routes;
- issue short TTLs and rotate signing keys through an overlap window;
- validate token kind, audience, run ID, tenant, user, task metadata, and expiry;
- keep provider keys and capability secrets in a gateway, broker, or secret manager;
- persist hashes or opaque handles in checkpoints, events, commands, receipts, and logs;
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

## Conformance and release gate

Before production traffic:

1. run both embedding golden paths offline;
2. run fast, contract, serial integration, cross-platform, and install-smoke CI tiers;
3. run the external `minimal-agent` profile against each product implementation, then call
   `run_checkpoint_store_contract(factory, root)` and `run_capability_broker_contract(factory)`
   directly for each replacement store or broker;
4. execute the durability fault matrix for the selected checkpoint, activation, input, and effect
   paths;
5. verify the compatibility ledger and package contents from the built wheel;
6. rehearse fenced recovery, gateway outage, cursor reconnect, drain, rollback, and credential
   rotation for the selected runtime;
7. complete [security/PRODUCTION_CHECKLIST.md](security/PRODUCTION_CHECKLIST.md).

For the Reference inbox fixture, run `tests/test_backend_command_inbox.py` and its shared-store
recovery tests. For DBOS evaluation, install `reference-dbos`, run its focused model/driver/process
tests, and apply the scope limits in [DBOS_REFERENCE.md](DBOS_REFERENCE.md).

`tests/test_examples.py` imports and executes both golden paths. Contract, Helper Kit, and Reference
facade drift therefore fails CI at the same integration points shown here.
