# RunnerBackend Responsibility Map

This Phase 4 implementation note records the current split between the
`RunnerBackend` public facade, extracted private services, and the runtime
responsibilities that still belong on the facade/composition root.

## Current Shape

`RunnerBackend` is the single embedder-facing facade for the Reference backend.
Public methods stay on that object. Internally, the backend wires private
services with explicit context objects so each service depends on the ports it
uses instead of depending on the whole facade.

| Service | Current responsibility |
| --- | --- |
| `RunProjectionService` | Read-only projections: `status`, `result`, `events`, `descendant_events`, `diagnostics`, `list_runs`. |
| `RunPreparationService` | Request validation, workspace admission, initial runtime config validation, run token issuance, record creation, initial run metadata materialization, and submission payload assembly. |
| `RunStateMutationService` | Live run state mutation, backend/control event append decisions, terminal result/failure recording, and process-local tenant usage aggregation. |
| `BackendCommandService` | Control command dispatch and audit result mapping. |
| `BackendSessionService` | Session control actions, task callbacks, inbound messages, and `resume_run` entrypoint. |
| `SessionDriveService` | Open multi-turn session driving, message waits, checkpoint park points. |
| `RunExecutionService` | Autonomous and stream-driven run execution from a prepared run: loop attach, first turn drive, stream frames, terminal result/failure handoff, run slot accounting. |
| `BackendLoopFactory` | Reference `AgentLoop` assembly: run spec, model adapter, gateway clients, runtime-config provider, event sinks, capability broker, outbox sender, and checkpoint callback. |
| `OutboxDispatchService` | Reference outbox edge dispatch: due-request drain, retry backoff, ack staging, checkpoint persistence, watchdog redrive scheduling. |
| `ProposalService` | Reference proposal/package/artifact/approve/reject/apply operations. |
| `RuntimeConfigService` | Runtime config projection, validation, hot-swap, durable metadata commit. |
| `JobService` | Reference job artifact list/status/log/cancel projections. |
| `RecoveryService` | `recover_runs`, stale lease reclaim, checkpoint resume, recover-attempt bookkeeping, failure bundles. |

These services are private implementation modules under
`monoid_agent_kernel.reference.backend`. They are not stable contract exports.
Their context objects use private Protocol/port types from
`monoid_agent_kernel.reference.backend.ports` for stable internal shapes such as
run records, run requests, loop operations, token claims, queue snapshots, and
lease stores. Shared file readers such as proposal snapshot loading live in
private helper modules used by the services that need them.

## Responsibilities That Stay On RunnerBackend

| Responsibility | Current owner role |
| --- | --- |
| Public facade | Embedder-facing methods stay on `RunnerBackend` so callers interact with one stable object. |
| Composition root | `RunnerBackend.__post_init__` creates stores and default leases, then delegates service context wiring to private `_build_*_service` helpers. |
| Shared runtime ownership | `_spawn`, `_call_soon`, drain/shutdown, shared loop scheduling, semaphore ownership, and stream cancellation stay with the process-level runtime wrapper. |
| Request admission wiring | `submit_run`, `astream_run`, `_prepare_run_record`, `_submission_for`, `_validate_request`, `_check_workspace_allowed`, and `_write_run_meta` stay as facade compatibility and public API wrappers around `RunPreparationService`. |
| Loop factory wiring | `_build_loop_factory`, `_build_loop_build`, and `_build_loop` stay as facade compatibility and composition-root wrappers around `BackendLoopFactory`. |
| Run execution wiring | `submit_run`, `_run_run`, `_drive_session`, `astream_run`, and `_frame` stay as facade compatibility and public API wrappers around `RunExecutionService`. |
| Streaming public API | `astream_run` remains the embedder-facing stream seam; stream frame construction and run driving delegate to `RunExecutionService`. |
| Outbox dispatch wiring | `_build_outbox_dispatch_service`, `_drain_outbox`, `_stage_outbox_ack`, `_outbox_backoff_delay`, and `_redrive_outbox` stay as facade compatibility and composition-root wrappers around `OutboxDispatchService`. |
| Event/state wiring | `record_event`, `_emit_backend_event`, `_record_run_result`, `_record_run_failure`, and `tenant_usage` stay as facade compatibility and public API wrappers around `RunStateMutationService`. |
| Watchdog heartbeat | `start_watchdog`, `stop_watchdog`, `_watchdog_loop`, and `_heartbeat_own_runs` stay with process ownership. Stale reclaim delegates to `RecoveryService`; outbox redrive delegates to `OutboxDispatchService`. |
| Shared auth/record ports | `_authorize_run`, `_verify_run_token`, `_authorized_run_dir`, `_record`, `_active_record`, and token issuance helpers are backend-owned ports shared by services. |
| Recovery factories | Recovery request/record reconstruction and gateway-token reissue factories stay in the composition root; `RecoveryService` drives the recovery flow through those ports. |

## Extracted Product-Specific Logic

The main Reference product-specific surfaces have been moved out of the facade:

| Extracted service | Former backend surface |
| --- | --- |
| `RunPreparationService` | `_prepare_run_record`, `_submission_for`, `_validate_request`, `_check_workspace_allowed`, `_write_run_meta`. |
| `RunStateMutationService` | `record_event`, `_emit_backend_event`, `_record_run_result`, `_record_run_failure`, `tenant_usage`, lifecycle helper interpretation, `BackendRunStateSink`. |
| `RunProjectionService` | `status`, `result`, `events`, `descendant_events`, `diagnostics`, `list_runs`. |
| `ProposalService` | `proposal`, `proposal_diff`, `proposal_file`, `export_proposal_package`, `read_run_artifact`, `approve_proposal`, `reject_proposal`, `apply_proposal`. |
| `RuntimeConfigService` | `current_runtime_config`, `runtime_config`, `replace_runtime_config`, `_write_runtime_config_run_meta`. |
| `JobService` | `jobs`, `job_status`, `job_logs`, `cancel_job`. |
| `RecoveryService` | `recover_runs`, `_attempt_resume`, `_resume_from_checkpoint`, `_run_recovered`, recover-attempt helpers, failure bundle helpers, `_read_recovery_meta`, stale lease reclaim. |
| `BackendLoopFactory` | `_run_spec_for_request`, `_build_loop`, `_build_model_adapter`, `_llm_token_source`, `_web_gateway_client`, `_capability_broker_for`, `_outbox_sender_for`. |
| `OutboxDispatchService` | `_drain_outbox`, `_stage_outbox_ack`, `_outbox_backoff_delay`, `_redrive_outbox`. |
| `RunExecutionService` | `_run_run`, `_drive_session`, `astream_run` body, `_frame`. |

Compatibility wrappers remain on `RunnerBackend` where tests or internal call
sites use private methods. The implementation delegates through services.
Circular service callbacks have been removed: recovery resume/reclaim and
outbox ack staging call their owning service methods directly.

## Remaining Cleanup Targets

| Target | Why it remains |
| --- | --- |
| Conformance fixture decoupling | Generic profiles still name Reference scenarios in a few assertions. Those scenarios should move behind Reference harness case methods. |
| Closure docs | Public docs should link this responsibility map and summarize the Phase 4 facade/service boundary. |
| CI hardening | Xdist and coverage jobs are advisory. Promote them only after their signal is consistently clean. |
| Streaming transport adapters | HTTP SSE and Studio consumers sit outside the backend service split. Keep them transport-owned. |

## Design Position

The current structure matches the Phase 4 target:

- External callers use one stable `RunnerBackend` facade.
- Internal behavior is split by responsibility into private services.
- Services depend on explicit context/port objects.
- Private ports hide implementation details such as queue internals and service
  callback wiring.
- Core, helper, and conformance surfaces do not require this Reference backend
  decomposition or any specific storage/product deployment choice.

The next cleanup should focus on conformance fixture decoupling and documentation
closure, not more public API movement.
