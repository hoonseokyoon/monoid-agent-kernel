# Core Helper Kit

The Core Helper Kit is the supported runtime and helper module set for building systems that
satisfy the Monoid Agent Kernel contract. It gives backend and gateway implementers a clear path
to the Phase 1S operational rules while preserving replaceable deployment choices.

## Role

- Provide small modules for repeated contract rules.
- Keep security, durability, observability, and lifecycle semantics in readable places.
- Let Reference services assemble the helpers into runnable examples.
- Let external implementations reuse helpers or implement the same behavior and prove it with
  conformance profiles.

## Implemented Helper Surfaces

| Helper surface | Contract rules | Purpose |
| --- | --- | --- |
| `core.scope.scope_within`, `domain_patterns_within`, `effective_signed_scope` | `OR-01-SCOPE-RELATION`, `OR-02-CAPABILITY-BOUNDARY`, `OR-08-PROVIDER-CAPS` | Compare signed, requested, and granted scopes; apply numeric caps, list subsets, scalar equality, binding ids, and wildcard domain relations. |
| `core.lease_admission.validate_lease_admission`, `sanitize_denied_capability_result` | `OR-02-CAPABILITY-BOUNDARY`, `OR-03-LEASE-ADMISSION` | Reject capability mismatch and scope widening, preserve approved lease fields, and remove grant material from denied results. |
| `core.capability_revocation` | `OR-04-REVOCATION-SCOPE` | Track per-capability, per-lease, wildcard, and issued-before revocation state with import/export support. |
| `core.capability.CapabilityVault` | `OR-03-LEASE-ADMISSION`, `OR-04-REVOCATION-SCOPE`, `OR-09-SUBAGENT-BOUNDARY` | Compose lease admission, valid-token lookup, revocation checks, durable lease export, and child vault forking. |
| `core.event_sequencing.RunEventSequencer`, `read_event_page`, `diagnostic_event_summary` | `OR-05-EVENT-SEQUENCING`, `OR-06-CONTROL-AUDIT` | Keep event sequence ownership consistent across queued, live, terminal, and diagnostic read paths. |
| `core.control_audit.ControlAuditPolicy` | `OR-06-CONTROL-AUDIT` | Build redacted received, completed, and failed control audit payloads and declare callback-token command eligibility. |
| `core.durable_metadata.DurableMetadataCommitter` | `OR-07-DURABLE-METADATA` | Validate run metadata, commit runtime config updates, write shared metadata, and materialize local recovery descriptors. |
| `core.subagent_runtime.SubagentRuntimeContext`, `validate_descendant_run_id`, `subagent_diagnostics_from_events` | `OR-09-SUBAGENT-BOUNDARY` | Create child run identity, validate descendant event access, build lifecycle/result payloads, and summarize subagent diagnostics. |

## Gateway And Diagnostics Assembly

Provider gateway policy is assembled from the scope helpers and the Reference web gateway endpoint
checks. Search, fetch, and context handlers apply effective signed scope before provider invocation,
then preserve redirect and response caps through provider-specific code paths.

Lifecycle and diagnostics policy is assembled from event sequencing, durable metadata, control
audit, and subagent runtime helpers. Backend diagnostics use these helpers to expose bounded event,
control, recovery, trace, and subagent summaries.

## Promotion Rule

A guard starts as helper implementation quality. It becomes Contract language when it repeats
across implementations, affects security, durability, or observability, and can be asserted by a
helper-independent conformance profile.

## Phase 1S State

Phase 1S defines stable operational rule ids, implements the helper surfaces above, and pins their
observable behavior through conformance profiles. The full mapping from rule id to helper, profile,
Reference harness, and test coverage lives in `docs/PHASE_1S_COVERAGE.md`.
