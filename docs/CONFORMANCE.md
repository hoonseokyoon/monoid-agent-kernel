# Conformance Profiles

Conformance profiles let an integrator validate the contract behavior needed for a specific agent
runtime shape. Profiles are additive: a small local chatbot can run a
small profile, while a durable multi-agent backend can run the control, capability, gateway, and
multi-agent profiles.

The reusable package lives under `monoid_agent_kernel.conformance`. Phase 1S provides profile
metadata, harness protocols, reusable assertion helpers, and a bundled Reference factory that runs
the profiles against the shipped implementation.

## Profiles

| Profile | Target runtime | Contract rules |
| --- | --- | --- |
| `minimal-agent` | Local loop or chatbot-style integration | Metadata registration profile; concrete assertions remain future profile work. |
| `tool-agent` | Agent that executes tools | `OR-10-TOOL-SURFACE-ADMISSION`, `OR-11-GENERIC-ASK-APPROVAL` |
| `side-effect-tool-agent` | Agent that executes external side-effect tools | `OR-12-DURABLE-SIDE-EFFECT` |
| `durable-runner` | Backend that survives restarts | `OR-05-EVENT-SEQUENCING`, `OR-07-DURABLE-METADATA`, `OR-09-SUBAGENT-BOUNDARY` |
| `control-plane` | Backend with external control commands | `OR-03-LEASE-ADMISSION`, `OR-05-EVENT-SEQUENCING`, `OR-06-CONTROL-AUDIT`, `OR-07-DURABLE-METADATA` |
| `capability-security` | Capability-gated runtime | `OR-01-SCOPE-RELATION`, `OR-02-CAPABILITY-BOUNDARY`, `OR-03-LEASE-ADMISSION`, `OR-04-REVOCATION-SCOPE`, `OR-06-CONTROL-AUDIT`, `OR-09-SUBAGENT-BOUNDARY` |
| `provider-gateway` | Runtime using LLM/Web gateways | `OR-01-SCOPE-RELATION`, `OR-02-CAPABILITY-BOUNDARY`, `OR-08-PROVIDER-CAPS` |
| `multi-agent` | Runtime with subagents | `OR-04-REVOCATION-SCOPE`, `OR-09-SUBAGENT-BOUNDARY` |
| `reference-full` | Bundled Reference services and Studio smoke path | All Phase 1S operational rules, Phase 2 tool-agent rules, and Reference smoke. |

## Executable Assertions

| Profile | Assertion helpers | Harnesses |
| --- | --- | --- |
| `provider-gateway` | `assert_provider_gateway_profile` | `GatewayHarness` |
| `tool-agent` | `assert_tool_agent_surface_admission_profile`; `assert_tool_agent_generic_ask_approval_profile` | `BackendHarness` |
| `side-effect-tool-agent` | `assert_side_effect_tool_agent_profile` | `SideEffectHarness` |
| `capability-security` | `assert_capability_security_lease_admission`; `assert_capability_security_revocation_profile` | `CapabilityHarness` |
| `control-plane` | `assert_control_plane_decision_profile`; `assert_control_plane_audit_sequence_profile` | `BackendHarness` |
| `durable-runner` | `assert_durable_runner_event_sequence_profile`; `assert_durable_runner_recovery_metadata_profile`; `assert_durable_runner_subagent_diagnostics_profile` | `BackendHarness` |
| `multi-agent` | `assert_multi_agent_backend_boundary_profile`; `assert_multi_agent_backend_capability_boundary_profile`; `assert_multi_agent_shared_revocation_profile` | `BackendHarness`; `CapabilityHarness` |
| `reference-full` | `assert_reference_full_profile` | `ReferenceFullFactory` with backend, capability, gateway, and Studio harnesses |

`minimal-agent` remains a registered metadata profile. `tool-agent` is executable for tool surface
admission and generic approval behavior. `side-effect-tool-agent` adds the optional durable
external side-effect profile for runtimes that expose those tools.

## Harness Roles

The conformance package defines four Protocol families:

- `BackendHarness`: submits runs, inspects status/events/diagnostics/results, dispatches control
  commands, replaces runtime config, resumes runs, and restarts against durable state.
- `SideEffectHarness`: extends `BackendHarness` with normalized durable side-effect request
  inspection for the optional side-effect profile.
- `GatewayHarness`: calls gateway surfaces with scoped inputs and reports normalized provider
  outcomes.
- `CapabilityHarness`: issues capability requests, grants, denials, revocations, callback-token
  results, and child-vault checks.

The bundled Reference implementation is the first harness target. External backends can implement
the same harness protocols and run the same profile suite.

## Reference Full

The `reference-full` profile runs the bundled Reference implementation through the concrete Phase
1S and Phase 2 profile set. It uses
`monoid_agent_kernel.reference.conformance.ReferenceConformanceFactory` to create a fresh backend,
capability, gateway, or side-effect harness for each assertion. The factory also runs an offline
Studio smoke path that boots Studio, starts a chat, observes events, and confirms session
visibility.

The profile is a release confidence target for the Reference assembly. External implementations
should run the smaller profiles directly with their own harnesses.

## Phase 1S Sequence

1. Phase 1S 1차: profile metadata, harness protocols, public rule ids, and import-smoke tests.
2. Phase 1S 2차: scope relation helper and provider gateway cap conformance.
3. Phase 1S 3차: lease admission helper, denied-result sanitization, and control decision conformance.
4. Phase 1S 4차: revocation helper, child vault sharing, and revocation conformance.
5. Phase 1S 5차: event sequencing, durable metadata, control audit helpers, and recovery conformance.
6. Phase 1S 6차: subagent runtime context, descendant event access, diagnostics, and multi-agent conformance.
7. Phase 1S 7차: Reference conformance factory, `reference-full`, and Studio smoke integration.
8. Phase 1S 8차: final coverage matrix and public docs closure.
9. Phase 2 1차: executable `tool-agent` profile and generic `authorization="ask"` approval.
10. Phase 2 2차: optional `side-effect-tool-agent` profile, strict durable side-effect policy,
    outbox conformance, and idempotency-key admission.

## Acceptance

A profile passes when the implementation satisfies the observable behavior named by the rule ids.
The tests assert behavior through harness protocols. The Core Helper Kit provides the supported
implementation path for those behaviors. `docs/PHASE_1S_COVERAGE.md` maps every Phase 1S rule to
helpers, assertions, Reference scenarios, and tests.
