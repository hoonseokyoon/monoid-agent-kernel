# Conformance Profiles

Conformance profiles let an integrator validate the contract behavior needed for a specific agent
runtime shape. Profiles are additive: a small local chatbot can run a
small profile, while a durable multi-agent backend can run the control, capability, gateway, and
multi-agent profiles.

The reusable package lives under `monoid_agent_kernel.conformance`. It provides profile metadata,
harness protocols, reusable assertion helpers, and a bundled Reference factory that runs the
profiles against the shipped implementation.

## Profiles

| Profile | Target runtime | Contract rules |
| --- | --- | --- |
| `minimal-agent` | Local loop or chatbot-style integration | Metadata registration profile; concrete assertions remain future profile work. |
| `tool-agent` | Agent that executes tools | `OR-10-TOOL-SURFACE-ADMISSION`, `OR-11-GENERIC-ASK-APPROVAL` |
| `side-effect-tool-agent` | Agent that executes external side-effect tools | `OR-12-DURABLE-SIDE-EFFECT` |
| `message-fabric` | Agent runtime that exchanges peer-agent messages | `OR-13-EXTERNAL-AGENT-ENVELOPE` |
| `durable-runner` | Backend that survives restarts | `OR-05-EVENT-SEQUENCING`, `OR-07-DURABLE-METADATA`, `OR-09-SUBAGENT-BOUNDARY` |
| `control-plane` | Backend with external control commands | `OR-03-LEASE-ADMISSION`, `OR-05-EVENT-SEQUENCING`, `OR-06-CONTROL-AUDIT`, `OR-07-DURABLE-METADATA` |
| `capability-security` | Capability-gated runtime | `OR-01-SCOPE-RELATION`, `OR-02-CAPABILITY-BOUNDARY`, `OR-03-LEASE-ADMISSION`, `OR-04-REVOCATION-SCOPE`, `OR-06-CONTROL-AUDIT`, `OR-09-SUBAGENT-BOUNDARY` |
| `provider-gateway` | Runtime using LLM/Web gateways | `OR-01-SCOPE-RELATION`, `OR-02-CAPABILITY-BOUNDARY`, `OR-08-PROVIDER-CAPS` |
| `multi-agent` | Runtime with subagents | `OR-04-REVOCATION-SCOPE`, `OR-09-SUBAGENT-BOUNDARY` |
| `reference-full` | Bundled Reference services and Studio smoke path | All current operational rules and Reference smoke. |

## Executable Assertions

| Profile | Assertion helpers | Harnesses |
| --- | --- | --- |
| `provider-gateway` | `assert_provider_gateway_profile` | `GatewayHarness` |
| `tool-agent` | `assert_tool_agent_surface_admission_profile`; `assert_tool_agent_generic_ask_approval_profile` | `ToolAgentHarness` |
| `side-effect-tool-agent` | `assert_side_effect_tool_agent_profile` | `SideEffectHarness` |
| `message-fabric` | `assert_message_fabric_profile` | `MessageFabricHarness` |
| `capability-security` | `assert_capability_security_lease_admission`; `assert_capability_security_revocation_profile` | `CapabilityHarness` |
| `control-plane` | `assert_control_plane_decision_profile`; `assert_control_plane_audit_sequence_profile` | `ControlPlaneHarness` |
| `durable-runner` | `assert_durable_runner_event_sequence_profile`; `assert_durable_runner_recovery_metadata_profile`; `assert_durable_runner_subagent_diagnostics_profile` | `DurableRunnerHarness` |
| `multi-agent` | `assert_multi_agent_backend_boundary_profile`; `assert_multi_agent_backend_capability_boundary_profile`; `assert_multi_agent_shared_revocation_profile` | `MultiAgentBackendHarness`; `CapabilityHarness` |
| `reference-full` | `assert_reference_full_profile` | `ReferenceFullFactory` with backend, capability, gateway, side-effect, message-fabric, and Studio harnesses |

`minimal-agent` remains a registered metadata profile. `tool-agent` is executable for tool surface
admission and generic approval behavior. `side-effect-tool-agent` adds the optional durable
external side-effect profile for runtimes that expose those tools. `message-fabric` adds the
optional external-agent envelope profile for runtimes that exchange messages with peer agents.

## Harness Roles

The conformance package defines profile-specific Protocol families:

- `ToolAgentHarness`: runs tool surface admission and generic approval behavior cases.
- `ControlPlaneHarness`: runs decision and control audit sequencing behavior cases.
- `DurableRunnerHarness`: runs event sequence, recovery metadata, and subagent diagnostics cases.
- `MultiAgentBackendHarness`: runs backend-visible subagent boundary and capability-boundary cases.
- `SideEffectHarness`: runs durable side-effect behavior cases for the optional side-effect profile.
- `MessageFabricHarness`: runs external-agent message-fabric behavior cases for the optional
  message-fabric profile.
- `GatewayHarness`: calls gateway surfaces with scoped inputs and reports normalized provider
  outcomes.
- `CapabilityHarness`: issues capability requests, grants, denials, revocations, callback-token
  results, and child-vault checks.
- `BackendHarness`: raw backend operations kept for compatibility and custom harnesses. New generic
  profile assertions use the narrower Protocols above.

The bundled Reference implementation is the first harness target. External backends can implement
the same harness protocols and run the same profile suite.

## Reference Full

The `reference-full` profile runs the bundled Reference implementation through the concrete current
profile set. It uses
`monoid_agent_kernel.reference.conformance.ReferenceConformanceFactory` to create fresh
profile-specific backend harnesses, capability harnesses, gateway harnesses, side-effect harnesses,
and message-fabric harnesses. The factory also runs an offline Studio smoke path that boots Studio,
starts a chat, observes events, and confirms session visibility.

Reference harnesses use named scenarios internally. Those scenario names are fixture details for
the bundled implementation. External implementations satisfy conformance by returning the same
observable case results through the harness protocols.

The profile is a release confidence target for the Reference assembly. External implementations
should run the smaller profiles directly with their own harnesses.

## Testing Policy

Conformance assertions are deterministic behavior checks. They run against harness protocols and
case methods, then assert observable results such as events, lifecycle state, diagnostics,
outbox/inbox state, and public payloads. Reference scenario names stay inside
`monoid_agent_kernel.reference.conformance`.

Hypothesis property tests target pure helpers, parsers, and serializers. Use them for JSON-native
wire payloads, sanitizer helpers, metadata merge helpers, and other in-process functions with no
server, thread, clock, network, or durable backend dependency.

Backend scenarios, conformance assertions, Studio server paths, threaded services, and live
provider paths use fixed regression cases. This keeps conformance output stable and keeps fuzzing
focused on edge-case parser coverage.

## Implementation Sequence

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
11. Phase 2 3차: optional `message-fabric` profile, external-agent envelope helper, and Reference
    inbox-routing outbox sender conformance.
12. Phase 2 closure: approval hardening, optional harness narrowing, OR-13 contract narrowing, and
    current coverage closure.
13. Phase 2S 1차: strict wire parser helper and Hypothesis coverage for pure wire parsers.
14. Phase 2S 2차: public/private payload boundary and canonical external-agent metadata merge.
15. Phase 2S 3차: helper adoption hardening in web scope, backend events, durable metadata, and
    Studio subagent routing.
16. Phase 2S 4차: coverage closure and validation/library policy documentation.
17. Phase 4-4P: profile-specific harness protocols, Reference fixture decoupling, and conformance
    import-boundary guards.
18. Phase 4-4Q: Phase 4 structural closure docs and CI/advisory risk position.

## Acceptance

A profile passes when the implementation satisfies the observable behavior named by the rule ids.
The tests assert behavior through harness protocols. The Core Helper Kit provides the supported
implementation path for those behaviors. `docs/OPERATIONAL_RULE_COVERAGE.md` maps every rule to
helpers, assertions, Reference case methods, and tests.

Fast local checks:

```bash
python -m pytest tests/conformance -q
python -m pytest tests/test_wire_validation.py tests/test_external_agent_envelope_properties.py tests/test_inbox_outbox_properties.py -q
python -m pytest -q -n 4
```
