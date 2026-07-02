# Phase 1S Coverage

Phase 1S turns the operational rules in `docs/CONTRACTS.md` into traceable helper surfaces,
conformance assertions, and Reference smoke paths. This document is the coverage index for
that work.

## Rule Coverage

| Rule | Contract meaning | Core Helper Kit surface | Conformance assertion | Reference harness / scenario | Primary tests |
| --- | --- | --- | --- | --- | --- |
| `OR-01-SCOPE-RELATION` | Signed scope bounds request scope; request scope bounds grant scope; caps and domain patterns narrow consistently. | `core.scope.scope_within`, `domain_patterns_within`, `effective_signed_scope` | `assert_provider_gateway_profile`; `assert_capability_security_lease_admission` | `ReferenceGatewayHarness`; `ReferenceCapabilityHarness` | `tests/test_scope_relation.py`; `tests/conformance/test_provider_gateway_profile.py`; `tests/conformance/test_capability_security_profile.py` |
| `OR-02-CAPABILITY-BOUNDARY` | Capability identity, binding id, domain filters, and endpoint boundaries are preserved through gateway and lease paths. | `core.scope.effective_signed_scope`; `core.lease_admission.validate_lease_admission`; `CapabilityVault.admit` | `assert_provider_gateway_profile`; `assert_capability_security_lease_admission` | `ReferenceGatewayHarness`; `ReferenceCapabilityHarness` | `tests/test_web_gateway.py`; `tests/test_capability.py`; `tests/conformance/test_provider_gateway_profile.py` |
| `OR-03-LEASE-ADMISSION` | Approved leases preserve policy fields; grants cannot widen requests; denied decisions strip grant material. | `core.lease_admission.validate_lease_admission`; `sanitize_denied_capability_result`; `CapabilityVault.admit` | `assert_capability_security_lease_admission`; `assert_control_plane_decision_profile` | `ReferenceCapabilityHarness`; `ReferenceBackendHarness` scenario `parked-hitl` | `tests/test_lease_admission.py`; `tests/test_capability.py`; `tests/test_backend_control.py`; `tests/conformance/test_capability_security_profile.py`; `tests/conformance/test_control_plane_profile.py` |
| `OR-04-REVOCATION-SCOPE` | Revocation covers capability, lease id, watermark, wildcard, recovery import/export, and child vault sharing. | `core.capability_revocation`; `CapabilityVault.revoke`; `CapabilityVault.fork_for_child`; `AgentLoop.revoke_capability` | `assert_capability_security_revocation_profile`; `assert_multi_agent_shared_revocation_profile`; `assert_multi_agent_backend_capability_boundary_profile` | `ReferenceCapabilityHarness`; `ReferenceBackendHarness` scenario `subagent-capability-revoked` | `tests/test_capability_revocation.py`; `tests/test_capability.py`; `tests/test_subagent.py`; `tests/conformance/test_capability_revocation_profile.py`; `tests/conformance/test_multi_agent_profile.py` |
| `OR-05-EVENT-SEQUENCING` | Run event sequence ownership stays monotonic across live recorders, direct appends, terminal appends, and diagnostics. | `core.event_sequencing.RunEventSequencer`; `read_event_page`; `diagnostic_event_summary` | `assert_durable_runner_event_sequence_profile`; `assert_control_plane_audit_sequence_profile` | `ReferenceBackendHarness` scenarios `multi-turn` and `completed` | `tests/test_event_sequencing.py`; `tests/test_backend_control.py`; `tests/conformance/test_durable_runner_profile.py`; `tests/conformance/test_control_plane_profile.py` |
| `OR-06-CONTROL-AUDIT` | Authorized commands emit redacted received/completed/failed audit events; unauthorized commands stay outside the run stream. | `core.control_audit.ControlAuditPolicy`; `core.event_sequencing.RunEventSequencer` | `assert_control_plane_decision_profile`; `assert_control_plane_audit_sequence_profile` | `ReferenceBackendHarness` scenarios `parked-hitl`, `multi-turn`, and `completed` | `tests/test_backend_control.py`; `tests/conformance/test_control_plane_profile.py` |
| `OR-07-DURABLE-METADATA` | Runtime config and recovery metadata commit in an order that keeps API state and recovery state aligned. | `core.durable_metadata.DurableMetadataCommitter`; `validate_run_metadata`; `runtime_config_from_metadata` | `assert_durable_runner_recovery_metadata_profile`; `assert_control_plane_audit_sequence_profile` | `ReferenceBackendHarness` scenario `recoverable-multi-turn`; restart modes `same` and `empty` | `tests/test_durable_metadata.py`; `tests/test_backend_recovery.py`; `tests/test_backend_runtime_config.py`; `tests/conformance/test_durable_runner_profile.py` |
| `OR-08-PROVIDER-CAPS` | Gateway requests apply effective signed/request/default caps on search, fetch, context, redirects, bytes, and timeouts. | `core.scope.effective_signed_scope`; Reference web gateway cap application | `assert_provider_gateway_profile` | `ReferenceGatewayHarness` | `tests/test_web_gateway.py`; `tests/conformance/test_provider_gateway_profile.py` |
| `OR-09-SUBAGENT-BOUNDARY` | Child runs have descendant identity, isolated event streams, trace linkage, accounting roll-up, diagnostics summaries, and shared revocation visibility. | `core.subagent_runtime.SubagentRuntimeContext`; `validate_descendant_run_id`; `subagent_diagnostics_from_events`; `CapabilityVault.fork_for_child` | `assert_multi_agent_backend_boundary_profile`; `assert_multi_agent_backend_capability_boundary_profile`; `assert_multi_agent_shared_revocation_profile`; `assert_durable_runner_subagent_diagnostics_profile` | `ReferenceBackendHarness` scenarios `subagent-foreground` and `subagent-capability-revoked`; `ReferenceCapabilityHarness.fork_child()` | `tests/test_subagent_runtime_context.py`; `tests/test_subagent.py`; `tests/test_backend_turn_recovery.py`; `tests/conformance/test_multi_agent_profile.py`; `tests/conformance/test_durable_runner_profile.py` |

## Profile Coverage

| Profile | Phase 1S status | Executable assertions | Harnesses |
| --- | --- | --- | --- |
| `minimal-agent` | Metadata registration profile; concrete assertions remain future profile work. | None in Phase 1S. | `backend` |
| `tool-agent` | Metadata registration profile; concrete assertions remain future profile work. | None in Phase 1S. | `backend` |
| `provider-gateway` | Concrete provider gateway profile. | `assert_provider_gateway_profile` | `gateway` |
| `capability-security` | Concrete lease admission and revocation profile. | `assert_capability_security_lease_admission`; `assert_capability_security_revocation_profile` | `capability` |
| `control-plane` | Concrete control decision and audit sequencing profile. | `assert_control_plane_decision_profile`; `assert_control_plane_audit_sequence_profile` | `backend` |
| `durable-runner` | Concrete event sequencing, recovery metadata, and subagent diagnostics profile. | `assert_durable_runner_event_sequence_profile`; `assert_durable_runner_recovery_metadata_profile`; `assert_durable_runner_subagent_diagnostics_profile` | `backend` |
| `multi-agent` | Concrete subagent boundary and shared revocation profile. | `assert_multi_agent_backend_boundary_profile`; `assert_multi_agent_backend_capability_boundary_profile`; `assert_multi_agent_shared_revocation_profile` | `backend`; `capability` |
| `reference-full` | Bundled Reference release-confidence profile. | `assert_reference_full_profile` runs the concrete profiles above and the offline Studio smoke path. | `backend`; `capability`; `gateway`; `studio` |

## Coverage Notes

Some profile metadata declares rules that are exercised through neighboring profiles in the Phase
1S suite. `capability-security` lists `OR-06-CONTROL-AUDIT` and `OR-09-SUBAGENT-BOUNDARY` because
capability results and revocations feed those boundaries; executable backend checks for those rules
live in `control-plane`, `multi-agent`, and `durable-runner`. Provider gateway redirect boundaries
and raw byte trimming are covered by companion gateway regression tests alongside
`assert_provider_gateway_profile`.

## Reference Harness Coverage

`monoid_agent_kernel.reference.conformance.ReferenceConformanceFactory` creates fresh harnesses for
each assertion:

- `new_backend()` returns `ReferenceBackendHarness` with scenarios `completed`, `multi-turn`,
  `recoverable-multi-turn`, `parked-hitl`, `subagent-foreground`, and
  `subagent-capability-revoked`.
- `new_capability()` returns `ReferenceCapabilityHarness` for pure capability lease, denial,
  revocation, and child-vault checks.
- `new_gateway()` returns `ReferenceGatewayHarness` for signed web gateway scope and cap checks.
- `run_studio_smoke()` boots offline Studio, starts a chat, observes events, and confirms session
  visibility.

The conformance package asserts observable behavior through harness protocols. The Reference
factory is the canonical adapter for the bundled implementation.
