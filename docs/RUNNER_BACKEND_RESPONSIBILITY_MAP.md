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
| `RunProjectionService` | Read-only projections: `status`, `result`, `events`, `diagnostics`, `list_runs`. |
| `BackendCommandService` | Control command dispatch and audit result mapping. |
| `BackendSessionService` | Session control actions, task callbacks, inbound messages, and `resume_run` entrypoint. |
| `SessionDriveService` | Open multi-turn session driving, message waits, checkpoint park points. |
| `ProposalService` | Reference proposal/package/artifact/approve/reject/apply operations. |
| `RuntimeConfigService` | Runtime config projection, validation, hot-swap, durable metadata commit. |
| `JobService` | Reference job artifact list/status/log/cancel projections. |
| `RecoveryService` | `recover_runs`, stale lease reclaim, checkpoint resume, recover-attempt bookkeeping, failure bundles. |

These services are private implementation modules under
`monoid_agent_kernel.reference.backend`. They are not stable contract exports.
Their context objects use private Protocol/port types from
`monoid_agent_kernel.reference.backend.ports` for stable internal shapes such as
run records, run requests, loop operations, token claims, and lease stores.

## Responsibilities That Stay On RunnerBackend

| Responsibility | Current owner role |
| --- | --- |
| Public facade | Embedder-facing methods stay on `RunnerBackend` so callers interact with one stable object. |
| Composition root | `RunnerBackend.__post_init__` creates stores and default leases, then delegates service context wiring to private `_build_*_service` helpers. |
| Shared runtime ownership | `_spawn`, `_call_soon`, drain/shutdown, shared loop scheduling, semaphore ownership, and stream cancellation stay with the process-level runtime wrapper. |
| Request admission | `_validate_request`, `_check_workspace_allowed`, initial runtime config validation, gateway token issuance, and initial record creation stay in the facade. |
| Loop construction | `_build_loop` assembles the Reference `AgentLoop` wiring: model adapter, tool providers, context providers, validators, capability broker, outbox sender, event sinks, and checkpoint callback. |
| Run execution entrypoints | `submit_run`, `_run_run`, `_drive_session`, and `astream_run` still coordinate cold-start execution around the loop. |
| Streaming execution | `astream_run` remains facade-owned because it combines submission metadata, event/delta/result framing, stream lifetime, and semaphore ownership. |
| Outbox edge dispatch | `_drain_outbox`, `_stage_outbox_ack`, retry backoff, and watchdog redrive stay in the Reference operational edge. |
| Event sink integration | `BackendRunStateSink`, `record_event`, `_emit_backend_event`, and run state mutation keep live record state aligned with recorded events. |
| Usage accounting | `tenant_usage` and terminal result metric aggregation remain on the backend-owned in-memory usage ledger. |
| Watchdog heartbeat | `start_watchdog`, `stop_watchdog`, `_watchdog_loop`, `_heartbeat_own_runs`, and `_redrive_outbox` stay with process ownership. Stale reclaim delegates to `RecoveryService`. |
| Shared auth/record ports | `_authorize_run`, `_verify_run_token`, `_authorized_run_dir`, `_record`, `_active_record`, and token issuance helpers are backend-owned ports shared by services. |
| Recovery factories | Recovery request/record reconstruction and gateway-token reissue factories stay in the composition root; `RecoveryService` drives the recovery flow through those ports. |

## Extracted Product-Specific Logic

The main Reference product-specific surfaces have been moved out of the facade:

| Extracted service | Former backend surface |
| --- | --- |
| `ProposalService` | `proposal`, `proposal_diff`, `proposal_file`, `export_proposal_package`, `read_run_artifact`, `approve_proposal`, `reject_proposal`, `apply_proposal`. |
| `RuntimeConfigService` | `current_runtime_config`, `runtime_config`, `replace_runtime_config`, `_write_runtime_config_run_meta`. |
| `JobService` | `jobs`, `job_status`, `job_logs`, `cancel_job`. |
| `RecoveryService` | `recover_runs`, `_attempt_resume`, `_resume_from_checkpoint`, `_run_recovered`, recover-attempt helpers, failure bundle helpers, `_read_recovery_meta`, stale lease reclaim. |

Compatibility wrappers remain on `RunnerBackend` where tests or internal call
sites use private methods. The implementation now delegates through services.

## Remaining Cleanup Targets

| Target | Why it remains |
| --- | --- |
| Service port typing | Private ports now cover the main internal shapes. Remaining `Any` usage is concentrated in dynamic JSON/tool/provider payloads and test-facing monkeypatch seams. |
| Composition root size | Service context wiring now lives in `_build_*_service` helpers. Future cleanup should keep each helper small rather than adding wiring back to `__post_init__`. |
| Loop construction | `_build_loop` is still a large Reference assembly point. It is correctly facade-owned today, but a future `LoopFactory` could make model/tool/provider wiring clearer. |
| Outbox edge service | Outbox drain/redrive/ack logic is operational edge behavior. It can become a private `OutboxDispatchService` if it grows further. |
| Event sink/state mutation | `record_event`, `_record_run_result`, `_record_run_failure`, and backend event append remain coherent as live-state ownership. They can be revisited after service port typing is stable. |
| Streaming path | `astream_run` still owns stream-driven execution. Extract it only after preserving event/result framing with focused tests. |

## Design Position

The current structure matches the Phase 4 target:

- External callers use one stable `RunnerBackend` facade.
- Internal behavior is split by responsibility into private services.
- Services depend on explicit context/port objects.
- Core, helper, and conformance surfaces do not require this Reference backend
  decomposition or any specific storage/product deployment choice.

The next cleanup should focus on private type/port clarity, not more public API
movement.
