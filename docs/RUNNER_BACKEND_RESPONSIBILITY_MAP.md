# RunnerBackend Responsibility Map

This is a Phase 4 implementation note for the Reference backend. It records the
current split between the public facade, extracted internal services, and the
responsibilities still owned by `RunnerBackend`.

## Current Shape

`RunnerBackend` is the public facade and composition root for the Reference
backend. It owns process-level runtime resources and wires private services with
explicit contexts:

| Service | Current responsibility |
| --- | --- |
| `RunProjectionService` | Read-only projections: status, result, events, diagnostics, list runs. |
| `BackendCommandService` | Control command dispatch and audit result mapping. |
| `BackendSessionService` | Session control actions, task callbacks, inbound messages, resume entrypoint. |
| `SessionDriveService` | Open multi-turn session driving and checkpoint park points. |

These services receive private context objects instead of depending on the
entire facade.

## Responsibilities That Stay On RunnerBackend

| Responsibility | Why it stays here |
| --- | --- |
| Public facade | Embedder-facing methods still live on one stable object. |
| Composition root | The backend creates stores, leases, services, providers, and callback contexts. |
| Shared runtime ownership | `_spawn`, `_call_soon`, drain/shutdown, semaphore ownership, and stream cancellation belong to the process-level runtime wrapper. |
| Loop construction | `_build_loop` assembles the Reference wiring around `AgentLoop`: model adapter, tool providers, context providers, validators, broker, outbox sender, event sinks. |
| Run execution entrypoints | `_run_run`, `_drive_session`, `_run_recovered`, and `astream_run` still coordinate lifecycle around the loop. |
| Event sink integration | `BackendRunStateSink`, `record_event`, and backend event append behavior keep live record state aligned with recorded events. |

## Product-Specific Logic Still On RunnerBackend

| Candidate | Current methods | Notes |
| --- | --- | --- |
| Proposal package service | `proposal`, `proposal_diff`, `proposal_file`, `export_proposal_package`, `read_run_artifact`, `approve_proposal`, `reject_proposal`, `apply_proposal` | Best next extraction target. The logic is Reference-specific, cohesive, and mostly bounded by run directory, proposal files, artifact blobs, approval records, and backend event emission. |
| Runtime config service | `current_runtime_config`, `runtime_config`, `replace_runtime_config`, `_write_runtime_config_run_meta` | Important but more sensitive. It touches live record state, validation, shared durable metadata, and hot-swap behavior. |
| Job service | `jobs`, `job_status`, `job_logs`, `cancel_job` | Small and low risk. It can be extracted opportunistically, but it does not reduce the core backend complexity much. |
| Recovery service | `recover_runs`, `_attempt_resume`, `_resume_from_checkpoint`, `_run_recovered`, recover-attempt helpers, failure bundle helpers, metadata helpers, watchdog reclaim paths | High value and high risk. It crosses durable metadata, checkpoint restore, leases, spawned run ownership, and event-loop scheduling. |

## Next Extraction Decision

Extract `ProposalService` next.

Reasons:

- It removes product-specific Reference package/apply behavior from the facade.
- It does not touch `_drive_open_session`, `_run_run`, `_run_recovered`, or shared event-loop ownership.
- It has a clear context boundary: authorize run, record lookup, checkpoint blob store, backend event emission, apply roots, and proposal file helpers.
- It gives an immediate readability gain before touching recovery or runtime config.

Recommended order after `ProposalService`:

1. `RuntimeConfigService`
2. `JobService`
3. `RecoveryService`

Recovery should follow after more product-specific surfaces are out of the
facade, because recovery is the highest-risk remaining boundary.
