# Core Helper Kit

The Core Helper Kit is the supported runtime and helper module set for building systems that
satisfy the Monoid Agent Kernel contract. It gives backend and gateway implementers a clear path
to the Phase 1S operational rules while preserving replaceable deployment choices.

## Role

- Provide small modules for repeated contract rules.
- Keep security, durability, observability, and lifecycle semantics in readable places.
- Let Reference services assemble the helpers into runnable examples.
- Let external implementations either reuse helpers or implement the same behavior and prove it with conformance profiles.

## Initial Helper Candidates

| Helper | Contract rules | Purpose |
| --- | --- | --- |
| `ScopeRelation` | `OR-01-SCOPE-RELATION`, `OR-02-CAPABILITY-BOUNDARY` | Compare signed scope, requested scope, grant scope, numeric caps, lists, and wildcard domains. |
| `ProviderGatewayPolicy` | `OR-01-SCOPE-RELATION`, `OR-02-CAPABILITY-BOUNDARY`, `OR-08-PROVIDER-CAPS` | Normalize provider caps and domain filters for gateway calls and redirect checks. |
| `LeaseAdmission` | `OR-03-LEASE-ADMISSION`, `OR-04-REVOCATION-SCOPE`, `OR-09-SUBAGENT-BOUNDARY` | Preserve lease policy fields, strip denied grant material, and coordinate revocation semantics. |
| `RunEventSequencer` | `OR-05-EVENT-SEQUENCING`, `OR-06-CONTROL-AUDIT` | Keep run event sequence ownership consistent across queued, live, terminal, and recovered states. |
| `ControlCommandPolicy` | `OR-03-LEASE-ADMISSION`, `OR-06-CONTROL-AUDIT` | Declare command auth modes, lifecycle requirements, audit envelopes, and decision coercion. |
| `DurableMetadataCommitter` | `OR-07-DURABLE-METADATA` | Align local descriptors, shared metadata, compatibility paths, and recovery outcomes. |
| `LifecycleProjection` | `OR-05-EVENT-SEQUENCING`, `OR-07-DURABLE-METADATA` | Project internal lifecycle states into wire status and operation predicates. |
| `DiagnosticsBuilder` | `OR-05-EVENT-SEQUENCING`, `OR-09-SUBAGENT-BOUNDARY` | Build bounded event, failure, recovery, control, and trace summaries with redaction. |
| `SubagentRuntimeContext` | `OR-04-REVOCATION-SCOPE`, `OR-09-SUBAGENT-BOUNDARY` | Keep child identity, accounting, capability slots, revocation, and traces connected. |

## Promotion Rule

A guard starts as helper implementation quality. It becomes Contract language when it repeats
across implementations, affects security/durability/observability, and can be asserted by a
helper-independent conformance profile.

## Phase 1S Boundary

The Phase 1S first slice creates docs and conformance skeletons. Helper extraction starts with later PRs,
beginning with scope relation and provider gateway policy.
