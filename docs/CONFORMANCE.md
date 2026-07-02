# Conformance Profiles

Phase 1S introduces conformance profiles so an integrator can validate the contract behavior
needed for a specific agent runtime shape. Profiles are additive: a small local chatbot can
run a small profile, while a durable multi-agent backend can run the control, capability,
gateway, and multi-agent profiles.

The reusable package lives under `monoid_agent_kernel.conformance`. Phase 1S adds profile metadata,
harness protocols, reusable assertion helpers, and a bundled Reference factory that runs the
profiles against the shipped implementation.

## Profiles

| Profile | Target runtime | Contract rules |
| --- | --- | --- |
| `minimal-agent` | Local loop or chatbot-style integration | base loop, model adapter, runtime config |
| `tool-agent` | Agent that executes tools | tool binding, tool result, permissions, output validation |
| `durable-runner` | Backend that survives restarts | `OR-05-EVENT-SEQUENCING`, `OR-07-DURABLE-METADATA`, partial `OR-09-SUBAGENT-BOUNDARY` |
| `control-plane` | Backend with external control commands | `OR-03-LEASE-ADMISSION`, `OR-05-EVENT-SEQUENCING`, `OR-06-CONTROL-AUDIT`, `OR-07-DURABLE-METADATA` |
| `capability-security` | Capability-gated runtime | `OR-01-SCOPE-RELATION`, `OR-02-CAPABILITY-BOUNDARY`, `OR-03-LEASE-ADMISSION`, `OR-04-REVOCATION-SCOPE`, `OR-06-CONTROL-AUDIT`, `OR-09-SUBAGENT-BOUNDARY` |
| `provider-gateway` | Runtime using LLM/Web gateways | `OR-01-SCOPE-RELATION`, `OR-02-CAPABILITY-BOUNDARY`, `OR-08-PROVIDER-CAPS` |
| `multi-agent` | Runtime with subagents | `OR-04-REVOCATION-SCOPE`, `OR-09-SUBAGENT-BOUNDARY` |
| `reference-full` | Bundled Reference services and Studio smoke path | all Phase 1S rules plus reference smoke |

## Harness Roles

The conformance package defines three initial Protocol families:

- `BackendHarness`: submits runs, inspects status/events/diagnostics, and dispatches control commands.
- `GatewayHarness`: calls gateway surfaces with scoped inputs and reports normalized provider outcomes.
- `CapabilityHarness`: issues capability requests, grants, denials, revocations, and callback-token results.

The first implementation target is the bundled Reference backend and gateways. External backends can
implement the same harness protocols and run the same profile suite.

## Reference Full

The `reference-full` profile runs the bundled Reference implementation through the profile set. It
uses `monoid_agent_kernel.reference.conformance.ReferenceConformanceFactory` to create fresh backend,
capability, and gateway harnesses for each assertion. The factory also runs an offline Studio smoke
path that boots Studio, starts a chat, observes events, and confirms session visibility.

The profile is a release confidence target for the Reference assembly. External implementations
should run the smaller profiles directly with their own harnesses.

## Phase 1S Sequence

1. Phase 1S first slice: profile metadata, harness protocols, and import-smoke tests.
2. Phase 1S 2차: first matrix tests for scope relation and provider gateway caps.
3. Phase 1S 3-6차: lease admission, revocation, durable metadata, control audit, diagnostics, and subagent assertions.
4. Phase 1S 7차: Reference factory, `reference-full`, and Studio smoke integration.

## Acceptance

A profile passes when the implementation satisfies the observable behavior named by the rule ids.
The tests assert behavior, not use of a specific helper module. The Core Helper Kit provides the
supported implementation path for those behaviors.
