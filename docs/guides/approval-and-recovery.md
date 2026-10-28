# Approval and recovery

Run a tool until it parks for approval, end that process, and restore the same run in a new one.
The packaged [report recipe](../../src/monoid_agent_kernel/reference/host_tool_report.py) demonstrates
this flow with typed tools, a checked filesystem checkpoint, fresh resources, and scripted models.

## Install and prepare

Use Python 3.11+ and install this checkout with `python -m pip install .`, or install its matching
wheel. The recipe module is included in the package. These commands use the v0.25 development
API; package metadata currently reports `0.24.0`.

Choose a root to retain between commands, then run:

```bash
python -m monoid_agent_kernel.reference.host_tool_report --phase prepare --root ./report-state
```

Prepare writes a scoped proposal, requests approval for `report.read`, commits the parked
checkpoint, and releases its activation. Keep the JSON response's `run_id` and `task_id`.
It reports `status="awaiting_tasks"`, one resource open, zero reads, three model calls, and
`proposal_only=true`. The calls save the report, request approval, and acknowledge the pending task.

## Resume in a new process

Copy the original IDs into the next command and keep the same root:

```bash
python -m monoid_agent_kernel.reference.host_tool_report --phase resume --root ./report-state --run-id RUN_ID --task-id TASK_ID
```

Each command starts a separate interpreter. Resume loads the checked checkpoint, reconstructs
the loop and resource factory, checks the pending task, and delivers approval. It executes the
original durable request and returns the proposal body after one approved read and one model call.

| Field | Prepare | Resume |
|---|---|---|
| `status` | `awaiting_tasks` | `completed` |
| `run_id` / `task_id` | Newly printed IDs | Same IDs |
| `pid` | Prepare process | Different process |
| `resource_opens` | 1 | 1 fresh resource |
| `read_calls` | 0 | 1 |
| `model_calls` | 3 | 1 |
| `total_usage` input/output/total | 5 / 3 / 8 | 9 / 6 / 15 |
| `proposal_only` | `true` | `true` |

Resume also returns `duplicate_ignored=true` and `report_body` containing
`# Weekly\n\nReady for approval.\n`. The recipe checks duplicate result delivery inside the resume
process. Use that output to verify deduplication. The workspace file remains unchanged because
this run uses proposal mode; applying the reviewed proposal is a separate host operation.

## Connect the product

The recipe gives the trusted host concrete insertion points:

- **Workspace and storage:** `prepare_report` creates `root/workspace` and uses `root/runs` for
  `LocalFsCheckpointStore`. Persist both roots across processes. `resume_report` reconstructs the
  same `AgentRunSpec` with the original run ID and mode, then loads `latest_checked(run_id)`.
- **Runtime and tools:** `_report_loop` supplies `AgentRuntimeConfig`, stable report tool IDs,
  the `reports/**` path scope, and `authorization="ask"` for reads. Reconstruct compatible
  configuration and permission policy on restore.
- **Resources:** `open_reports` returns `activation.workspace_resource()` through
  `HostToolBinding`. The new process injects that factory again. Keys, clients, and adapters
  stay process-local; read restored content inside the handler. See
  [tools and resources](tools-and-resources.md#connect-product-resources).
- **Model:** replace the scripted adapter at `_report_loop` with the product's adapter.
  [custom_model_adapter.py](../../examples/custom_model_adapter.py) shows implementation and injection;
  [BACKEND.md](../BACKEND.md) covers the hosted gateway route.
- **Approval transport:** authenticate the approver, authorize the run/task, render the durable
  request, and deliver the decision through the task-scoped callback path. The recipe assumes
  the local host already owns those IDs. Keep transport credentials in the host.

The core calls in `resume_report` are `restore(record.checkpoint, blobs=record.blob)`, session
inspection, `report_task_result(task_id, {"approved": True})`, and `run_until_suspended(None)`.
Use the complete recipe for their cleanup and error paths. A denial follows the same authorized
task-result route with `approved=false`; preserve the original task identity and decision.
Hosted command ordering, callback tokens, and durable receipts are described in
[control commands, tasks, and approvals](../EMBEDDING.md#control-commands-tasks-and-approvals).

## Recover from failures

Checked loads distinguish `loaded`, `migrated`, `missing`, `corrupt`, and `unsupported_version`.
Continue only with accepted state. Surface corrupt or unsupported state for operator repair.
See the [compatibility ledger](../COMPATIBILITY.md) and [v0.25 migration](../migrations/v0.25.md)
before changing reader/writer versions.

An unrelated task ID is refused before approval delivery, and the committed checkpoint is
preserved. Restore failures retain the original error while discarding the local activation.
The example verifies one original result reaches the model after duplicate result delivery for
the same task. Reuse the pending run/task identity when retrying delivery through the host.

The demonstrated boundary is a committed park followed by restoration. For crash recovery and
external effects, the host owns fencing, replay-stable idempotency, or durable outbox staging.
LocalFS and SQLite checkpoint stores are single-writer adapters; a host with competing activations
uses fenced storage and checks owner/generation with each canonical mutation. Follow
[local recovery](../EMBEDDING.md#local-recovery) and
[hosted recovery obligations](../EMBEDDING.md#checkpoints-recovery-and-upgrades).

Run `python -m monoid_agent_kernel.reference.host_tool_report` without arguments for the temporary
approval demo and fresh/fork child resources. Each child gets its own resource; parent resources
remain usable after child cleanup. Use the [embedding handbook](../EMBEDDING.md) to choose the
production topology and [conformance gate](../EMBEDDING.md#conformance-and-release-gate) for that host.
