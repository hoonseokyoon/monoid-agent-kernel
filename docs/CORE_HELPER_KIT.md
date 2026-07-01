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
| `ScopeRelation` | `PH1S-R1`, `PH1S-R2` | Compare signed scope, requested scope, grant scope, numeric caps, lists, and wildcard domains. |
| `ProviderGatewayPolicy` | `PH1S-R1`, `PH1S-R2`, `PH1S-R8` | Normalize provider caps and domain filters for gateway calls and redirect checks. |
| `LeaseAdmission` | `PH1S-R3`, `PH1S-R4`, `PH1S-R9` | Preserve lease policy fields, strip denied grant material, and coordinate revocation semantics. |
| `RunEventSequencer` | `PH1S-R5`, `PH1S-R6` | Keep run event sequence ownership consistent across queued, live, terminal, and recovered states. |
| `ControlCommandPolicy` | `PH1S-R3`, `PH1S-R6` | Declare command auth modes, lifecycle requirements, audit envelopes, and decision coercion. |
| `DurableMetadataCommitter` | `PH1S-R7` | Align local descriptors, shared metadata, compatibility paths, and recovery outcomes. |
| `LifecycleProjection` | `PH1S-R5`, `PH1S-R7` | Project internal lifecycle states into wire status and operation predicates. |
| `DiagnosticsBuilder` | `PH1S-R5`, `PH1S-R9` | Build bounded event, failure, recovery, control, and trace summaries with redaction. |
| `SubagentRuntimeContext` | `PH1S-R4`, `PH1S-R9` | Keep child identity, accounting, capability slots, revocation, and traces connected. |

## Promotion Rule

A guard starts as helper implementation quality. It becomes Contract language when it repeats
across implementations, affects security/durability/observability, and can be asserted by a
helper-independent conformance profile.

## Phase 1S Boundary

The Phase 1S first slice creates docs and conformance skeletons. Helper extraction starts with later PRs,
beginning with scope relation and provider gateway policy.
