# Operational Rule Coverage

This document is the current coverage index for Monoid operational rules. It maps each rule to the
helper surface, executable conformance assertion, Reference harness case, and primary tests that
prove the rule.

## Rule Coverage

| Rule | Contract meaning | Core Helper Kit surface | Conformance assertion | Reference harness / case | Primary tests |
| --- | --- | --- | --- | --- | --- |
| `OR-01-SCOPE-RELATION` | Signed scope bounds request scope; request scope bounds grant scope; caps and domain patterns narrow consistently. | `core.scope.scope_within`, `domain_patterns_within`, `effective_signed_scope` | `assert_provider_gateway_profile`; `assert_capability_security_lease_admission` | `ReferenceGatewayHarness`; `ReferenceCapabilityHarness` | `tests/test_scope_relation.py`; `tests/conformance/test_provider_gateway_profile.py`; `tests/conformance/test_capability_security_profile.py` |
| `OR-02-CAPABILITY-BOUNDARY` | Capability identity, binding id, domain filters, and endpoint boundaries are preserved through gateway and lease paths. | `core.scope.effective_signed_scope`; `core.lease_admission.validate_lease_admission`; `CapabilityVault.admit` | `assert_provider_gateway_profile`; `assert_capability_security_lease_admission` | `ReferenceGatewayHarness`; `ReferenceCapabilityHarness` | `tests/test_web_gateway.py`; `tests/test_capability.py`; `tests/conformance/test_provider_gateway_profile.py` |
| `OR-03-LEASE-ADMISSION` | Approved leases preserve policy fields; grants cannot widen requests; denied decisions strip grant material. | `core.lease_admission.validate_lease_admission`; `sanitize_denied_capability_result`; `CapabilityVault.admit` | `assert_capability_security_lease_admission`; `assert_control_plane_decision_profile` | `ReferenceCapabilityHarness`; `ReferenceBackendHarness` `parked-hitl` case | `tests/test_lease_admission.py`; `tests/test_capability.py`; `tests/test_backend_control.py`; `tests/conformance/test_capability_security_profile.py`; `tests/conformance/test_control_plane_profile.py` |
| `OR-04-REVOCATION-SCOPE` | Revocation covers capability, lease id, watermark, wildcard, recovery import/export, and child vault sharing. | `core.capability_revocation`; `CapabilityVault.revoke`; `CapabilityVault.fork_for_child`; `AgentLoop.revoke_capability` | `assert_capability_security_revocation_profile`; `assert_multi_agent_shared_revocation_profile`; `assert_multi_agent_backend_capability_boundary_profile` | `ReferenceCapabilityHarness`; `ReferenceBackendHarness` `subagent-capability-revoked` case | `tests/test_capability_revocation.py`; `tests/test_capability.py`; `tests/test_subagent.py`; `tests/conformance/test_capability_revocation_profile.py`; `tests/conformance/test_multi_agent_profile.py` |
| `OR-05-EVENT-SEQUENCING` | Run event sequence ownership stays monotonic across live recorders, direct appends, terminal appends, and diagnostics. | `core.event_sequencing.RunEventSequencer`; `read_event_page`; `diagnostic_event_summary` | `assert_durable_runner_event_sequence_profile`; `assert_control_plane_audit_sequence_profile` | `ReferenceBackendHarness` `multi-turn` and `completed` cases | `tests/test_event_sequencing.py`; `tests/test_backend_control.py`; `tests/conformance/test_durable_runner_profile.py`; `tests/conformance/test_control_plane_profile.py` |
| `OR-06-CONTROL-AUDIT` | Authorized commands emit redacted received/completed/failed audit events; unauthorized commands stay outside the run stream. | `core.control_audit.ControlAuditPolicy`; `core.event_sequencing.RunEventSequencer` | `assert_control_plane_decision_profile`; `assert_control_plane_audit_sequence_profile` | `ReferenceBackendHarness` `parked-hitl`, `multi-turn`, and `completed` cases | `tests/test_backend_control.py`; `tests/conformance/test_control_plane_profile.py` |
| `OR-07-DURABLE-METADATA` | Runtime config and recovery metadata commit in an order that keeps API state and recovery state aligned. | `core.durable_metadata.DurableMetadataCommitter`; `validate_run_metadata`; `runtime_config_from_metadata` | `assert_durable_runner_recovery_metadata_profile`; `assert_control_plane_audit_sequence_profile` | `ReferenceBackendHarness` `recoverable-multi-turn` case; restart modes `same` and `empty` | `tests/test_durable_metadata.py`; `tests/test_backend_recovery.py`; `tests/test_backend_runtime_config.py`; `tests/conformance/test_durable_runner_profile.py` |
| `OR-08-PROVIDER-CAPS` | Gateway requests apply effective signed/request/default caps on search, fetch, context, redirects, bytes, and timeouts. | `core.scope.effective_signed_scope`; Reference web gateway cap application | `assert_provider_gateway_profile` | `ReferenceGatewayHarness` | `tests/test_web_gateway.py`; `tests/conformance/test_provider_gateway_profile.py` |
| `OR-09-SUBAGENT-BOUNDARY` | Child runs have descendant identity, isolated event streams, trace linkage, accounting roll-up, diagnostics summaries, and shared revocation visibility. | `core.subagent_runtime.SubagentRuntimeContext`; `validate_descendant_run_id`; `subagent_diagnostics_from_events`; `CapabilityVault.fork_for_child` | `assert_multi_agent_backend_boundary_profile`; `assert_multi_agent_backend_capability_boundary_profile`; `assert_multi_agent_shared_revocation_profile`; `assert_durable_runner_subagent_diagnostics_profile` | `ReferenceBackendHarness` `subagent-foreground` and `subagent-capability-revoked` cases; `ReferenceCapabilityHarness.fork_child()` | `tests/test_subagent_runtime_context.py`; `tests/test_subagent.py`; `tests/test_backend_turn_recovery.py`; `tests/conformance/test_multi_agent_profile.py`; `tests/conformance/test_durable_runner_profile.py` |
| `OR-10-TOOL-SURFACE-ADMISSION` | Tool execution follows the active turn surface; unavailable, hidden, denied, and quota-exceeded bindings do not execute handlers. | `core.tool_surface.DefaultToolSurfaceResolver`; `ToolSurfaceSnapshot`; `AgentLoop` tool admission path | `assert_tool_agent_surface_admission_profile` | `ReferenceBackendHarness` `tool-quota-denied` case | `tests/test_tool_surface.py`; `tests/test_loop.py`; `tests/conformance/test_tool_agent_profile.py` |
| `OR-11-GENERIC-ASK-APPROVAL` | `authorization="ask"` creates a durable approval task; approval revalidates the captured call before one execution; denial returns an observation without invoking the handler. | `core.tool_approval`; `TaskManager`; `AgentLoop` approval replay path | `assert_tool_agent_generic_ask_approval_profile` | `ReferenceBackendHarness` `tool-ask-approved`, `tool-ask-denied`, and `tool-ask-stale-denied` cases | `tests/test_tool_approval.py`; `tests/test_loop.py`; `tests/conformance/test_tool_agent_profile.py` |
| `OR-12-DURABLE-SIDE-EFFECT` | External side-effect tools declare delivery semantics; strict runtimes admit calls through durable outbox staging or explicit idempotency keys. | `core.side_effect_policy`; `core.outbox`; `ToolContext.emit_outbox`; Reference edge drain | `assert_side_effect_tool_agent_profile` | `SideEffectHarness.run_outbox_dispatched_case`; `run_pending_recovery_case`; `run_strict_rejected_case`; `run_idempotent_inline_case` | `tests/test_side_effect_policy.py`; `tests/test_outbox.py`; `tests/test_loop.py`; `tests/conformance/test_side_effect_tool_agent_profile.py` |
| `OR-13-EXTERNAL-AGENT-ENVELOPE` | Peer-agent messages preserve peer/message identity, restart-stable dedupe, correlation, causation, trace context, ordered text/data parts, and retryable pending/error state. | `core.external_agent_envelope`; `core.inbox`; `core.outbox`; Reference `InboxRoutingOutboxSender` | `assert_message_fabric_profile` | `MessageFabricHarness.run_two_peer_exchange_case`; `run_malformed_envelope_case`; `run_duplicate_after_restart_case`; `run_peer_unavailable_case` | `tests/test_external_agent_envelope.py`; `tests/test_outbox.py`; `tests/conformance/test_message_fabric_profile.py` |

## Profile Coverage

| Profile | Coverage status | Executable assertions | Harnesses |
| --- | --- | --- | --- |
| `minimal-agent` | Metadata registration profile. | None. | `backend` |
| `tool-agent` | Concrete tool surface and generic approval profile. | `assert_tool_agent_surface_admission_profile`; `assert_tool_agent_generic_ask_approval_profile` | `BackendHarness` |
| `side-effect-tool-agent` | Optional profile for runtimes that expose external side-effect tools. | `assert_side_effect_tool_agent_profile` | `SideEffectHarness` |
| `message-fabric` | Optional profile for runtimes that exchange peer-agent messages. | `assert_message_fabric_profile` | `MessageFabricHarness` |
| `provider-gateway` | Concrete provider gateway profile. | `assert_provider_gateway_profile` | `GatewayHarness` |
| `capability-security` | Concrete lease admission and revocation profile. | `assert_capability_security_lease_admission`; `assert_capability_security_revocation_profile` | `CapabilityHarness` |
| `control-plane` | Concrete control decision and audit sequencing profile. | `assert_control_plane_decision_profile`; `assert_control_plane_audit_sequence_profile` | `BackendHarness` |
| `durable-runner` | Concrete event sequencing, recovery metadata, and subagent diagnostics profile. | `assert_durable_runner_event_sequence_profile`; `assert_durable_runner_recovery_metadata_profile`; `assert_durable_runner_subagent_diagnostics_profile` | `BackendHarness` |
| `multi-agent` | Concrete subagent boundary and shared revocation profile. | `assert_multi_agent_backend_boundary_profile`; `assert_multi_agent_backend_capability_boundary_profile`; `assert_multi_agent_shared_revocation_profile` | `BackendHarness`; `CapabilityHarness` |
| `reference-full` | Bundled Reference release-confidence profile. | Runs every concrete assertion above plus the offline Studio smoke path. | `ReferenceFullFactory` |

## Reference Harness Coverage

`monoid_agent_kernel.reference.conformance.ReferenceConformanceFactory` creates fresh harnesses for
each assertion:

- `new_backend()` returns `ReferenceBackendHarness` for durable runner, control plane, tool-agent,
  and multi-agent cases.
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
observable behavior returned through harness methods.

## Coverage Notes

`OR-12-DURABLE-SIDE-EFFECT` and `OR-13-EXTERNAL-AGENT-ENVELOPE` are optional profiles. Runtimes
that do not expose external side-effect tools or peer-agent messaging skip those profiles.

`OR-13-EXTERNAL-AGENT-ENVELOPE` is the minimum message-fabric rule. It covers identity, dedupe,
correlation, causation, trace propagation, ordered text/data parts, and retryable delivery state.
Full A2A wire binding, Agent Card discovery, rich artifact lifecycle, push transport, and remote
auth policy remain extension work.

Some profile metadata lists neighboring rules because runtime behavior crosses boundaries. The
matrix above points to the concrete assertion that proves each behavior.
