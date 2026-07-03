# Core Helper Kit

The Core Helper Kit is the supported runtime and helper module set for building systems that
satisfy the Monoid Agent Kernel contract. It gives backend and gateway implementers a clear path
to the operational rules while preserving replaceable deployment choices.

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
| `core.tool_surface.DefaultToolSurfaceResolver`, `ToolSurfaceSnapshot` | `OR-10-TOOL-SURFACE-ADMISSION` | Build the active turn tool surface and enforce exposure, authorization, quota, scope, and binding availability before handlers run. |
| `core.tool_approval` | `OR-11-GENERIC-ASK-APPROVAL` | Build approval task payloads, redact argument previews, normalize approve/deny results, and derive replay descriptors for approved calls. |
| `core.side_effect_policy` | `OR-12-DURABLE-SIDE-EFFECT` | Read runtime side-effect policy, interpret tool/binding declarations, admit strict external side-effect calls through outbox or idempotency keys, and verify outbox staging. |
| `core.external_agent_envelope` | `OR-13-EXTERNAL-AGENT-ENVELOPE` | Build and validate the minimum transport-neutral peer-agent envelope, preserve ordered text/data parts, propagate correlation/causation/trace identity, and map outbox sends into inbox messages. |
| `core.wire_validation` | Cross-rule wire hardening | Parse JSON-native wire payloads with strict field typing, preserve missing optional defaults, and ignore unknown fields for forward compatibility. |
| `public_view.public_capability_result`, `HostedTask.public_payload` | `OR-03-LEASE-ADMISSION`, `OR-06-CONTROL-AUDIT`, `OR-11-GENERIC-ASK-APPROVAL` | Separate internal checkpoint/reentry payloads from public task, event, and diagnostics payloads. |

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

## Validation And Library Policy

Phase 2S hardens existing operational rules with stricter parsing and focused property tests.

- Pydantic v2 is available for strict local parser/helper models. Use `strict=True` and
  `extra="ignore"` at the parser boundary so present wrong-type values fail closed while unknown
  fields remain forward-compatible.
- jsonschema remains the schema validator for tool schemas, event payload schemas, manifests, and
  other JSON Schema surfaces.
- Hypothesis is a dev-time edge-case generator for pure helpers, parsers, serializers, sanitizers,
  and metadata merge helpers.
- msgspec stays a hold candidate. It can be revisited when wire model replacement has a measurable
  performance or maintenance payoff.
- portalocker stays a hold candidate. Local filesystem locking is a Reference store detail, while
  product checkpoint stores can provide stronger native concurrency control.
- tenacity stays a hold candidate. Retry timing and backoff remain edge/backend policy around
  durable outbox state.
- anyio stays a hold candidate. Async runtime adoption touches provider, gateway, Studio, and
  backend lifecycle boundaries together.

## Coverage State

The current operational rules are mapped to helper surfaces, profiles, Reference harnesses, and
tests in `docs/OPERATIONAL_RULE_COVERAGE.md`. The same document maps Phase 2S strict parser,
sanitizer, helper adoption, and property-test hardening to OR-01 through OR-13.

## Current Phase 2 And 2S State

Phase 2 provides executable `tool-agent` behavior, an optional `side-effect-tool-agent` profile,
and an optional `message-fabric` profile. `AgentLoop` uses `core.tool_approval` for
`authorization="ask"` calls and `core.side_effect_policy` for strict external side-effect
admission. Reference sender and harness code use `core.external_agent_envelope` to route peer-agent
messages over the existing inbox/outbox fabric. Backend-specific approval UI, notification policy,
external senders, transport bindings, discovery, and retry schedules remain outside these helpers.

Phase 2S adds strict wire parsing, public/private payload separation, canonical external-agent
metadata merge, and helper adoption in Reference and Studio boundaries. It keeps the current
operational rule ids and profile set fixed.
