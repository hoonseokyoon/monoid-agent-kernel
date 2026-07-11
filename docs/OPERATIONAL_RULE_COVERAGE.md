# Operational Rule Coverage

This document is the current coverage index for Monoid operational rules. It maps each rule to the
helper surface, executable conformance assertion, Reference harness case, and primary tests that
prove the rule.

It is also the current index for Phase 2S hardening coverage. Phase 2S keeps the OR-01 through
OR-13 rule set fixed and tightens existing behavior with strict parsers, public payload
sanitizers, canonical metadata merge helpers, helper adoption in Reference boundaries, and
property tests for pure wire helpers.

## Rule Coverage

| Rule | Contract meaning | Core Helper Kit surface | Conformance assertion | Reference harness / case | Primary tests |
| --- | --- | --- | --- | --- | --- |
| `OR-01-SCOPE-RELATION` | Signed scope bounds request scope; request scope bounds grant scope; caps and domain patterns narrow consistently. | `core.scope.scope_within`, `domain_patterns_within`, `effective_signed_scope` | `assert_provider_gateway_profile`; `assert_capability_security_lease_admission` | `ReferenceGatewayHarness`; `ReferenceCapabilityHarness` | `tests/test_scope_relation.py`; `tests/conformance/test_provider_gateway_profile.py`; `tests/conformance/test_capability_security_profile.py` |
| `OR-02-CAPABILITY-BOUNDARY` | Capability identity, binding id, domain filters, and endpoint boundaries are preserved through gateway and lease paths. | `core.scope.effective_signed_scope`; `core.lease_admission.validate_lease_admission`; `CapabilityVault.admit` | `assert_provider_gateway_profile`; `assert_capability_security_lease_admission` | `ReferenceGatewayHarness`; `ReferenceCapabilityHarness` | `tests/test_web_gateway.py`; `tests/test_capability.py`; `tests/conformance/test_provider_gateway_profile.py` |
| `OR-03-LEASE-ADMISSION` | Approved leases preserve policy fields; grants cannot widen requests; denied decisions strip grant material. | `core.lease_admission.validate_lease_admission`; `sanitize_denied_capability_result`; `CapabilityVault.admit` | `assert_capability_security_lease_admission`; `assert_control_plane_decision_profile` | `ReferenceCapabilityHarness`; `ControlPlaneHarness.run_control_decision_case()` | `tests/test_lease_admission.py`; `tests/test_capability.py`; `tests/test_backend_control.py`; `tests/conformance/test_capability_security_profile.py`; `tests/conformance/test_control_plane_profile.py` |
| `OR-04-REVOCATION-SCOPE` | Revocation covers capability, lease id, watermark, wildcard, recovery import/export, and child vault sharing. | `core.capability_revocation`; `CapabilityVault.revoke`; `CapabilityVault.fork_for_child`; `AgentLoop.revoke_capability` | `assert_capability_security_revocation_profile`; `assert_multi_agent_shared_revocation_profile`; `assert_multi_agent_backend_capability_boundary_profile` | `ReferenceCapabilityHarness`; `MultiAgentBackendHarness.run_subagent_capability_boundary_case()` | `tests/test_capability_revocation.py`; `tests/test_capability.py`; `tests/test_subagent.py`; `tests/conformance/test_capability_revocation_profile.py`; `tests/conformance/test_multi_agent_profile.py` |
| `OR-05-EVENT-SEQUENCING` | Run event sequence ownership stays monotonic across live recorders, direct appends, terminal appends, and diagnostics. | `core.event_sequencing.RunEventSequencer`; `read_event_page`; `diagnostic_event_summary` | `assert_durable_runner_event_sequence_profile`; `assert_control_plane_audit_sequence_profile` | `DurableRunnerHarness.run_event_sequence_case()`; `ControlPlaneHarness.run_control_audit_sequence_case()` | `tests/test_event_sequencing.py`; `tests/test_backend_control.py`; `tests/conformance/test_durable_runner_profile.py`; `tests/conformance/test_control_plane_profile.py` |
| `OR-06-CONTROL-AUDIT` | Authorized commands emit redacted received/completed/failed audit events; unauthorized commands stay outside the run stream. | `core.control_audit.ControlAuditPolicy`; `core.event_sequencing.RunEventSequencer` | `assert_control_plane_decision_profile`; `assert_control_plane_audit_sequence_profile` | `ControlPlaneHarness.run_control_decision_case()`; `run_control_audit_sequence_case()` | `tests/test_backend_control.py`; `tests/conformance/test_control_plane_profile.py` |
| `OR-07-DURABLE-METADATA` | Runtime config and recovery metadata commit in an order that keeps API state and recovery state aligned. | `core.durable_metadata.DurableMetadataCommitter`; `validate_run_metadata`; `runtime_config_from_metadata` | `assert_durable_runner_recovery_metadata_profile`; `assert_control_plane_audit_sequence_profile` | `DurableRunnerHarness.run_recovery_metadata_case()`; `ControlPlaneHarness.run_control_audit_sequence_case()` | `tests/test_durable_metadata.py`; `tests/test_backend_recovery.py`; `tests/test_backend_runtime_config.py`; `tests/conformance/test_durable_runner_profile.py` |
| `OR-08-PROVIDER-CAPS` | Gateway requests apply effective signed/request/default caps on search, fetch, context, redirects, bytes, and timeouts. | `core.scope.effective_signed_scope`; Reference web gateway cap application | `assert_provider_gateway_profile` | `ReferenceGatewayHarness` | `tests/test_web_gateway.py`; `tests/conformance/test_provider_gateway_profile.py` |
| `OR-09-SUBAGENT-BOUNDARY` | Child runs have descendant identity, isolated event streams, trace linkage, accounting roll-up, diagnostics summaries, and shared revocation visibility. | `core.subagent_runtime.SubagentRuntimeContext`; `validate_descendant_run_id`; `subagent_diagnostics_from_events`; `CapabilityVault.fork_for_child` | `assert_multi_agent_backend_boundary_profile`; `assert_multi_agent_backend_capability_boundary_profile`; `assert_multi_agent_shared_revocation_profile`; `assert_durable_runner_subagent_diagnostics_profile` | `MultiAgentBackendHarness.run_subagent_boundary_case()`; `run_subagent_capability_boundary_case()`; `DurableRunnerHarness.run_subagent_diagnostics_case()`; `ReferenceCapabilityHarness.fork_child()` | `tests/test_subagent_runtime_context.py`; `tests/test_subagent.py`; `tests/test_backend_turn_recovery.py`; `tests/conformance/test_multi_agent_profile.py`; `tests/conformance/test_durable_runner_profile.py` |
| `OR-10-TOOL-SURFACE-ADMISSION` | Tool execution follows the active turn surface; unavailable, hidden, denied, and quota-exceeded bindings do not execute handlers. | `core.tool_surface.DefaultToolSurfaceResolver`; `ToolSurfaceSnapshot`; `AgentLoop` tool admission path | `assert_tool_agent_surface_admission_profile` | `ToolAgentHarness.run_tool_surface_admission_case()` | `tests/test_tool_surface.py`; `tests/test_loop.py`; `tests/conformance/test_tool_agent_profile.py` |
| `OR-11-GENERIC-ASK-APPROVAL` | `authorization="ask"` creates a durable approval task; approval revalidates the captured call before one execution; denial returns an observation without invoking the handler. | `core.tool_approval`; `TaskManager`; `AgentLoop` approval replay path | `assert_tool_agent_generic_ask_approval_profile` | `ToolAgentHarness.run_generic_ask_approval_case()` | `tests/test_tool_approval.py`; `tests/test_loop.py`; `tests/conformance/test_tool_agent_profile.py` |
| `OR-12-DURABLE-SIDE-EFFECT` | External side-effect tools declare delivery semantics; strict runtimes admit calls through durable outbox staging or explicit idempotency keys. | `core.side_effect_policy`; `core.outbox`; `ToolContext.emit_outbox`; Reference edge drain | `assert_side_effect_tool_agent_profile` | `SideEffectHarness.run_outbox_dispatched_case`; `run_pending_recovery_case`; `run_strict_rejected_case`; `run_idempotent_inline_case` | `tests/test_side_effect_policy.py`; `tests/test_outbox.py`; `tests/test_loop.py`; `tests/conformance/test_side_effect_tool_agent_profile.py` |
| `OR-13-EXTERNAL-AGENT-ENVELOPE` | Peer-agent messages preserve peer/message identity, restart-stable dedupe, correlation, causation, trace context, ordered text/data parts, and retryable pending/error state. | `core.external_agent_envelope`; `core.inbox`; `core.outbox`; Reference `InboxRoutingOutboxSender` | `assert_message_fabric_profile` | `MessageFabricHarness.run_two_peer_exchange_case`; `run_malformed_envelope_case`; `run_duplicate_after_restart_case`; `run_peer_unavailable_case` | `tests/test_external_agent_envelope.py`; `tests/test_outbox.py`; `tests/conformance/test_message_fabric_profile.py` |

## Profile Coverage

| Profile | Coverage status | Executable assertions | Harnesses |
| --- | --- | --- | --- |
| `minimal-agent` | Executable submission-to-result lifecycle profile. | `run_minimal_agent_profile` | `MinimalAgentHarness`; bundled `ReferenceBackendHarness` |
| `tool-agent` | Concrete tool surface and generic approval profile. | `assert_tool_agent_surface_admission_profile`; `assert_tool_agent_generic_ask_approval_profile` | `ToolAgentHarness` |
| `side-effect-tool-agent` | Optional profile for runtimes that expose external side-effect tools. | `assert_side_effect_tool_agent_profile` | `SideEffectHarness` |
| `message-fabric` | Optional profile for runtimes that exchange peer-agent messages. | `assert_message_fabric_profile` | `MessageFabricHarness` |
| `provider-gateway` | Concrete provider gateway profile. | `assert_provider_gateway_profile` | `GatewayHarness` |
| `capability-security` | Concrete lease admission and revocation profile. | `assert_capability_security_lease_admission`; `assert_capability_security_revocation_profile` | `CapabilityHarness` |
| `control-plane` | Concrete control decision and audit sequencing profile. | `assert_control_plane_decision_profile`; `assert_control_plane_audit_sequence_profile` | `ControlPlaneHarness` |
| `durable-runner` | Concrete event sequencing, recovery metadata, and subagent diagnostics profile. | `assert_durable_runner_event_sequence_profile`; `assert_durable_runner_recovery_metadata_profile`; `assert_durable_runner_subagent_diagnostics_profile` | `DurableRunnerHarness` |
| `multi-agent` | Concrete subagent boundary and shared revocation profile. | `assert_multi_agent_backend_boundary_profile`; `assert_multi_agent_backend_capability_boundary_profile`; `assert_multi_agent_shared_revocation_profile` | `MultiAgentBackendHarness`; `CapabilityHarness` |
| `reference-full` | Bundled Reference release-confidence profile. | Runs every concrete assertion above plus the offline Studio smoke path. | `ReferenceFullFactory` |

## Reference Harness Coverage

`monoid_agent_kernel.reference.conformance.ReferenceConformanceFactory` creates fresh harnesses for
each assertion:

- `new_tool_agent()` returns a `ToolAgentHarness` for tool surface and approval cases.
- `new_control_plane()` returns a `ControlPlaneHarness` for decision and audit cases.
- `new_durable_runner()` returns a `DurableRunnerHarness` for sequencing, recovery, and subagent
  diagnostics cases.
- `new_multi_agent()` returns a `MultiAgentBackendHarness` for subagent backend boundary cases.
- `new_backend()` remains as compatibility raw backend harness construction.
- `new_capability()` returns `ReferenceCapabilityHarness` for pure capability lease, denial,
  revocation, and child-vault checks.
- `new_gateway()` returns `ReferenceGatewayHarness` for signed web gateway scope and cap checks.
- `new_side_effect()` returns a `SideEffectHarness` with behavior case methods for durable
  side-effect tools.
- `new_message_fabric()` returns a `MessageFabricHarness` with behavior case methods for
  external-agent message exchange.
- `run_studio_smoke()` boots offline Studio, starts a chat, observes events, and confirms session
  visibility.

Reference harnesses may use scenario names internally. Public conformance assertions depend on
observable behavior returned through case methods.

## Phase 2S Hardening Coverage

Phase 2S adds edge-case coverage to existing operational rules. It introduces no new rule ids and
no new conformance profiles.

| Rule | Phase 2S hardening | Helper / parser / sanitizer surface | Primary tests |
| --- | --- | --- | --- |
| `OR-01-SCOPE-RELATION` | Web tool-service domain filtering uses the same wildcard narrowing relation as signed gateway scope. | `core.scope.effective_signed_scope`; Reference web tool-service domain filter | `tests/test_tool_services.py`; `tests/test_scope_relation.py`; `tests/test_scope_relation_properties.py` |
| `OR-02-CAPABILITY-BOUNDARY` | Domain, binding, and capability boundaries stay intact when requested web scope is narrower than signed scope. | `core.scope.effective_signed_scope`; `CapabilityLease.from_json`; strict HTTP payload parsing | `tests/test_tool_services.py`; `tests/test_web_gateway.py`; `tests/test_capability.py` |
| `OR-03-LEASE-ADMISSION` | Capability result public payloads expose safe lease summaries while internal checkpoint/reentry payloads keep raw grant material. | `public_view.public_capability_result`; `HostedTask.public_payload`; `core.wire_validation` | `tests/test_tool_approval.py`; `tests/test_capability.py`; `tests/test_checkpoint.py` |
| `OR-04-REVOCATION-SCOPE` | Lease and checkpoint payload parsing rejects present wrong-type fields before durable revocation state is trusted. | `CapabilityLease.from_json`; `HostedTask.from_checkpoint`; `core.wire_validation` | `tests/test_capability.py`; `tests/test_capability_revocation.py`; `tests/test_checkpoint.py` |
| `OR-05-EVENT-SEQUENCING` | Proposal export, approval, rejection, apply, and conflict events use the same backend event sequencer path as control audit. | `core.event_sequencing.RunEventSequencer`; `RunnerBackend._emit_backend_event` | `tests/test_proposal_package.py`; `tests/test_backend_control.py`; `tests/test_event_sequencing.py` |
| `OR-06-CONTROL-AUDIT` | Public task payloads redact approval/capability grant material, and callback results parse approval booleans fail-closed. | `core.tool_approval`; `public_view.public_result_content`; `public_capability_result` | `tests/test_tool_approval.py`; `tests/test_backend_control.py` |
| `OR-07-DURABLE-METADATA` | Run listing reads recovery metadata through the durable metadata helper and materializes local descriptors from shared metadata. | `core.durable_metadata.DurableMetadataCommitter.read_recovery_metadata` | `tests/test_durable_metadata.py`; `tests/test_backend_recovery.py`; `tests/test_backend_runtime_config.py` |
| `OR-08-PROVIDER-CAPS` | Web request domain scope adopts `effective_signed_scope`; numeric cap behavior stays on the existing bounded cap path. | `core.scope.effective_signed_scope`; Reference web gateway cap application | `tests/test_tool_services.py`; `tests/test_web_gateway.py`; `tests/conformance/test_provider_gateway_profile.py` |
| `OR-09-SUBAGENT-BOUNDARY` | Studio subagent event routing derives the root ancestor id with the core helper and rejects traversal-shaped ids. | `core.subagent_runtime.root_run_id_from_descendant`; `validate_descendant_run_id` | `tests/test_subagent_runtime_context.py`; `tests/test_studio.py`; `tests/test_studio_sessions.py` |
| `OR-10-TOOL-SURFACE-ADMISSION` | External HTTP and control payloads parse tool arguments as objects, so wrong-type args do not reach handler admission. | `core.wire_validation`; Reference backend HTTP parsers; `DefaultToolSurfaceResolver` | `tests/test_backend_http_api.py`; `tests/test_tool_surface.py`; `tests/test_loop.py` |
| `OR-11-GENERIC-ASK-APPROVAL` | Approval callback payloads parse approve/deny values fail-closed, public approval payloads hide raw args, and replay state is checkpointed. | `core.tool_approval`; `HostedTask.public_payload`; approval replay checkpoint commit | `tests/test_tool_approval.py`; `tests/test_loop.py`; `tests/test_checkpoint.py` |
| `OR-12-DURABLE-SIDE-EFFECT` | Strict side-effect admission is covered by deterministic cases, and outbox/inbox wire payloads get property-tested parser coverage. | `core.side_effect_policy`; `core.outbox.OutboxRequest.from_json`; `core.inbox.InboxMessage.from_json` | `tests/test_side_effect_policy.py`; `tests/test_outbox.py`; `tests/test_inbox_outbox_properties.py` |
| `OR-13-EXTERNAL-AGENT-ENVELOPE` | Envelope parsing rejects malformed JSON-native shapes, metadata merge protects canonical identity, and inbox/outbox roundtrips are property-tested. | `core.external_agent_envelope`; `merge_canonical_metadata`; `core.wire_validation` | `tests/test_external_agent_envelope.py`; `tests/test_external_agent_envelope_properties.py`; `tests/test_inbox_outbox_properties.py` |

## Coverage Notes

`OR-12-DURABLE-SIDE-EFFECT` and `OR-13-EXTERNAL-AGENT-ENVELOPE` are optional profiles. Runtimes
that do not expose external side-effect tools or peer-agent messaging skip those profiles.

`OR-13-EXTERNAL-AGENT-ENVELOPE` is the minimum message-fabric rule. It covers identity, dedupe,
correlation, causation, trace propagation, ordered text/data parts, and retryable delivery state.
Full A2A wire binding, Agent Card discovery, rich artifact lifecycle, push transport, and remote
auth policy remain extension work.

Some profile metadata lists neighboring rules because runtime behavior crosses boundaries. The
matrix above points to the concrete assertion that proves each behavior.
