# Production embedding handbook

This handbook turns the Monoid Agent Kernel contracts into two production-shaped integration
paths. Both paths run offline in CI and use the same public APIs an embedding product uses.

- [`embedding_local_product.py`](../examples/embedding_local_product.py) embeds `AgentLoop`
  directly in one trusted process.
- [`embedding_hosted_product.py`](../examples/embedding_hosted_product.py) assembles the Reference
  backend, shared SQLite stores, ownership leases, durable commands, and multi-tenant identity.

Run them from a checkout:

```bash
python examples/embedding_local_product.py
python examples/embedding_hosted_product.py
python -m pytest -q tests/test_examples.py
```

Both scripts use `FakeModelAdapter`, require no credentials, make no network requests, and create
all state under a temporary directory.

## Choose the product topology

| Requirement | Embedded/local | Hosted/multi-tenant |
|---|---|---|
| One trusted desktop, worker, or appliance process | Best fit | Additional control-plane cost |
| Multiple API instances or worker failover | Single-owner only | Shared stores and leases |
| Direct Python calls | Primary interface | Useful inside the owner worker |
| HTTP/SSE product API | Optional adapter | Reference backend API |
| Tenant/user isolation | Product process boundary | Signed subject plus durable metadata |
| Durable cross-worker controls | Unnecessary | `CommandStore` plus owner watchdog |
| Store replacement/conformance | Usually local filesystem | Required before production rollout |

The contract/core/reference boundary guides both choices:

- `monoid_agent_kernel.contracts` and the core protocols define the integration obligations.
- Core helpers own lifecycle, events, checkpoints, tools, permissions, and deterministic behavior.
- `monoid_agent_kernel.reference.*` assembles a runnable backend and gateway architecture.
- Your product owns authentication, routing, secret storage, deployment, quotas, and incident
  response.

Reference code is a production-shaped example. Copy its ownership rules and validate replacement
components with the conformance suite.

## Golden path A: embedded/local product

The local example creates an `AgentRunSpec`, explicit `AgentRuntimeConfig`, scripted model adapter,
and `LocalFsCheckpointStore`. `LoopSession` keeps the run open across turns and commits a checkpoint
when the first turn settles.

```python
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
result = session.close()
```

Use this topology when the product process is the only run owner. Keep these directories separate:

- `workspace_root`: user-visible files the tools may read or change;
- `run_root`: events, status, proposals, checkpoints, and failure artifacts;
- application secrets: an external secret store or in-memory provider, outside both roots.

Choose `mode="propose"` when a human or product policy must approve file changes. Choose
`mode="apply"` only when the product has already authorized direct workspace mutation.

The example binds `fs.write` and `run.finish` explicitly. A tool absent from `ToolBinding` is absent
from the model surface. Keep the local surface small and use `AgentLoop.validate()` during startup
when tool configuration comes from users or deployment data.

### Local recovery

Use one durable `CheckpointStore` for the run lifetime. A local filesystem store requires a durable
mount and filesystem semantics that support atomic rename and locking. Read checkpoints with
`latest_checked()` when diagnostics must distinguish missing, unsupported, and corrupt state.

The core checkpoint contains control state and content-addressed blob references. Provider API
responses and raw credentials stay outside the durable payload. Restore with the same runtime
definition and compatible codec versions recorded in [COMPATIBILITY.md](COMPATIBILITY.md).

## Golden path B: hosted/multi-tenant product

The hosted example creates an owner backend and a peer API instance over one SQLite database:

```text
client/callback worker
        |
        v
 peer RunnerBackend ---- shared command inbox ----+
        |                                          |
        +---- shared checkpoint + lease stores ----+--> owner RunnerBackend --> AgentLoop
                                                   |
                                                   +--> events / receipts / usage
```

Each backend receives separate `SqliteCheckpointStore`, `SqliteLeaseStore`, and
`SqliteCommandStore` objects pointing at the same database. The owner starts its watchdog, which
publishes a fresh lease, heartbeats live runs, reclaims stale runs, redrives the outbox, and drains
commands.

The example submits `tenant_a/user_a` and `tenant_b/user_b` runs. A run token carries that subject,
and every peer compares it with durable run metadata before enqueueing a command. The owner mints a
fresh short-lived execution token after claiming the sanitized command. The inbox stores the
caller's token SHA-256 for audit attribution and never stores the bearer.

### Ownership boundaries

| Owner | Responsibility |
|---|---|
| Product edge/API | Authenticate the human or service and choose tenant/user/run scope. |
| Token manager or identity service | Sign short-lived audience- and kind-scoped capabilities. |
| Run owner backend | Execute the loop, sequence events, drain commands, and heartbeat its lease. |
| Peer backend | Authenticate, sanitize, enqueue, and return a durable receipt. |
| Checkpoint store | Atomically publish the last good run checkpoint and blobs. |
| Lease store | Provide one atomic stale-owner claim winner. |
| Command store | Provide idempotent append, strict per-run order, claim, acknowledgement, and receipts. |
| LLM/Web gateways | Hold provider credentials, enforce tenant quotas, and return sanitized results. |
| Product database | Map product records to `run_id`; avoid duplicating the kernel state machine. |

A peer accepts a command only while the lease store reports a fresh owner. An absent or stale owner
returns `command_owner_unavailable`. `create_task` stays owner-local and requires an empty command
lane because its callback token is returned once and is redacted from durable receipts.

## Model and tool wiring

The offline examples inject `model_adapter_factory` so no gateway is contacted. A hosted deployment
normally leaves that factory unset and configures `llm_gateway_url`; the Reference backend then uses
`GatewayModelAdapter` and run-scoped LLM gateway tokens.

Keep provider API keys inside the gateway process. The runner receives a signed gateway token with
tenant, user, run, audience, kind, and expiry claims. The gateway validates those claims, applies
provider/model policy, records usage, and injects the real provider credential at the last boundary.

Build runtime tool exposure from explicit bindings:

```python
config = AgentRuntimeConfig(
    definition_id="support-agent-v3",
    tools=(
        ToolBinding.for_tool("fs.read"),
        ToolBinding.for_tool("run.finish"),
    ),
)
```

For product tools, implement `ToolProvider` and use stable tool IDs. Declare side effects, scopes,
quotas, and capability requirements. Async handlers should implement the async contract directly;
sync handlers run behind the bounded sync boundary. See [TOOL_SURFACE.md](TOOL_SURFACE.md).

## Checkpoints, recovery, and migrations

Hosted workers must share the checkpoint and lease backends. A worker that owns no live record can
recover resumable runs with:

```python
backend = RunnerBackend(
    run_root=shared_run_root,
    checkpoint_store=shared_checkpoints,
    lease_store=shared_leases,
    command_store=shared_commands,
    # token manager, workspace roots, and gateway URLs omitted here
)
recovered_run_ids = backend.recover_runs()
backend.start_watchdog()
```

Recovery follows the durable lifecycle and skips terminal checkpoints. Stale lease takeover uses an
atomic compare-and-set, so one worker wins. Bound recovery attempts and surface an unrecoverable
failure artifact instead of retrying corrupt state forever.

Every durable family has a versioned codec and compatibility-ledger entry. Upgrade in this order:

1. deploy readers that accept the old and new schemas;
2. verify mixed-version conformance;
3. enable new writers;
4. monitor unsupported/corrupt load results;
5. retire an old reader only after rollback no longer needs it.

Run `python -m pytest -q tests/test_compatibility_ledger.py` to execute the machine-checked ledger,
then follow the mixed-version playbooks in [COMPATIBILITY.md](COMPATIBILITY.md).

## Streaming and cursor ownership

Use `subscribe_events()` for reusable page polling or frame iteration. The cursor stores the next
required sequence, suppresses replays, and raises on a gap.

```python
subscription = backend.subscribe_events(run_id, run_token, from_seq=1)
page = subscription.poll(limit=500)
next_seq = page["next_seq"]
```

For SSE, send each event sequence as `id`, accept `Last-Event-ID` on reconnect, and resume at the
next sequence. Heartbeats carry no event ID. A terminal subscription drains the event log once more
before emitting its end frame. Store the cursor in the consuming product when delivery must survive
client restarts.

The hosted golden path polls twice through one subscription and asserts the second page contains no
replayed events. It also creates a new backend after shutdown and reads the terminal event history
with the original run token.

## Control commands, tasks, and approvals

Send control operations through one `ControlCommand` envelope. Supply a stable `command_id` for
external retries:

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

An identical duplicate returns the existing receipt. Reusing the ID with different sanitized
arguments, command type, principal, issuer, or reason returns `command_id_conflict`. Commands execute
in append order with one in-flight command per run. A replacement worker can reclaim the head after
its claim TTL.

`create_task` returns a callback token only in the immediate owner response. Give that token to the
specific external worker. The worker may call `report_task_result`, `approve`, or `deny` for its task
scope and poll that command's receipt. Durable receipts redact callback tokens.

For tool approvals, render the sanitized task request to the approver, preserve the task ID, and
submit the decision through the same callback-scoped path. Never reconstruct a tool call from UI
text; use the durable task payload and decision contract.

Durable command arguments redact credential-shaped keys. Owner-local commands execute from a
transient bearer-scrubbed payload. Cross-worker commands execute from the sanitized durable payload;
place credential-shaped domain data in an external store and send a durable `token_ref` or object
reference.

## Gateway credentials and security

Apply these boundaries:

- expose backend, LLM gateway, Web gateway, and MCP gateway on separate audiences;
- keep admin tokens on private operational routes;
- issue short TTLs and rotate signing secrets with an overlap window;
- validate token kind, audience, run ID, tenant, user, task metadata, and expiry;
- keep provider keys and capability secrets in gateway/broker memory or a secret manager;
- store hashes or opaque handles in checkpoints, events, commands, receipts, and logs;
- restrict `allowed_workspace_roots` and resolve paths through the workspace abstraction;
- run the production checklist and conformance profiles before enabling external traffic.

The hosted golden path scans its run root and SQLite files for run and callback bearers after all
connections close. The test fails if a credential reaches durable storage.

## Observability

Treat the run event log as the canonical ordered audit stream. Consume lifecycle, model, tool,
control, task, usage, and failure events by sequence. `control.command.received` and terminal command
events carry the authenticating credential hash, command ID, actor, and reason.

Attach an `EventSink` for product telemetry. Use the OpenTelemetry sink described in
[OBSERVABILITY.md](OBSERVABILITY.md), and keep export failures outside the run state machine. Publish
metrics for:

- active runs by tenant and lifecycle state;
- checkpoint commit latency and load status;
- lease age, stale claims, and recovery attempts;
- command queue depth, claim age, conflicts, failures, and receipt latency;
- event cursor lag and sequence-gap failures;
- gateway usage, retries, quota denials, and provider latency;
- approval age and callback completion latency.

Correlate product requests with `run_id`, `command_id`, event sequence, and trace context. Exclude raw
prompts, tool secrets, bearer tokens, and unreviewed model output from routine logs.

## Failure handling matrix

| Failure | Required behavior |
|---|---|
| Model timeout or retryable provider error | Apply bounded retry policy and emit the final typed error. |
| Tool cancellation | Propagate async cancellation; bound cleanup; preserve the last good checkpoint. |
| Checkpoint write interruption | Publish blobs/manifest first and flip the latest pointer last. |
| Unsupported or corrupt durable schema | Return a typed load result and stop recovery for that record. |
| Owner crash | Let the lease expire; one replacement wins CAS and recovers. |
| Ownerless peer command | Reject before append with `command_owner_unavailable`. |
| Duplicate command ID with changed payload | Reject with `command_id_conflict`. |
| Command handler crash after claim | Persist a failure receipt; reclaim only after stale ownership when no acknowledgement exists. |
| Event client disconnect | Resume from `Last-Event-ID`; suppress replays; fail on gaps. |
| Lost `create_task` response | Create a new task intentionally; the callback secret is unrecoverable from the receipt. |
| Gateway credential expiry | Mint a new run-scoped gateway token; keep the provider key in the gateway. |
| Telemetry exporter failure | Drop, buffer, or retry telemetry without changing run semantics. |

## Conformance and release gate

Before production traffic:

1. run both embedding golden paths offline;
2. run fast, contract, serial integration, and install-smoke CI tiers;
3. run the external conformance runner against each replacement backend/store/broker;
4. execute the durability fault matrix for checkpoints, leases, command claims, and migrations;
5. verify the compatibility ledger and package install from the built wheel;
6. complete [security/PRODUCTION_CHECKLIST.md](security/PRODUCTION_CHECKLIST.md);
7. rehearse owner crash, stale reclaim, gateway outage, cursor reconnect, and rollback.

`tests/test_examples.py` imports and executes both golden paths. Public API drift therefore breaks CI
at the same code product integrators copy from this handbook.
