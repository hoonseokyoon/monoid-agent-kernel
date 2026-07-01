# Conformance Profiles

Phase 1S introduces conformance profiles so an integrator can validate the contract behavior
needed for a specific agent runtime shape. Profiles are additive: a small local chatbot can
run a small profile, while a durable multi-agent backend can run the control, capability,
gateway, and multi-agent profiles.

The reusable package skeleton lives under `monoid_agent_kernel.conformance`. The Phase 1S first slice adds
profile metadata and harness protocols. Concrete assertions land in later Phase 1S PRs.

## Profiles

| Profile | Target runtime | Contract rules |
| --- | --- | --- |
| `minimal-agent` | Local loop or chatbot-style integration | base loop, model adapter, runtime config |
| `tool-agent` | Agent that executes tools | tool binding, tool result, permissions, output validation |
| `durable-runner` | Backend that survives restarts | `PH1S-R5`, `PH1S-R7`, partial `PH1S-R9` |
| `control-plane` | Backend with external control commands | `PH1S-R3`, `PH1S-R5`, `PH1S-R6`, `PH1S-R7` |
| `capability-security` | Capability-gated runtime | `PH1S-R1`, `PH1S-R2`, `PH1S-R3`, `PH1S-R4`, `PH1S-R6`, `PH1S-R9` |
| `provider-gateway` | Runtime using LLM/Web gateways | `PH1S-R1`, `PH1S-R2`, `PH1S-R8` |
| `multi-agent` | Runtime with subagents | `PH1S-R4`, `PH1S-R9` |
| `reference-full` | Bundled Reference services and Studio smoke path | all Phase 1S rules plus reference smoke |

## Harness Roles

The conformance package defines three initial Protocol families:

- `BackendHarness`: submits runs, inspects status/events/diagnostics, and dispatches control commands.
- `GatewayHarness`: calls gateway surfaces with scoped inputs and reports normalized provider outcomes.
- `CapabilityHarness`: issues capability requests, grants, denials, revocations, and callback-token results.

The first implementation target is the bundled Reference backend and gateways. External backends can
implement the same harness protocols and run the same profile suite.

## Phase 1S Sequence

1. Phase 1S first slice: profile metadata, harness protocols, and import-smoke tests.
2. Phase 1S 2차: first matrix tests for scope relation and provider gateway caps.
3. Later Phase 1S PRs: control audit sequencing, lease admission, durable metadata, diagnostics, and subagent profile assertions.

## Acceptance

A profile passes when the implementation satisfies the observable behavior named by the rule ids.
The tests assert behavior, not use of a specific helper module. The Core Helper Kit provides the
supported implementation path for those behaviors.
