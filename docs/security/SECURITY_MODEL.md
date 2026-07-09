# Security Model

## Status and audience

Monoid Agent Kernel is a **pre-1.0 embeddable agent runtime**. This document
describes the security boundaries the kernel is *designed* to preserve, the
assumptions those boundaries rely on, and the responsibilities left to the
product integrator.

It is a companion to two other documents:

- [THREAT_MODEL.md](THREAT_MODEL.md) — what can go wrong, per surface.
- [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md) — what to do before deploying.

Read this as *intended guarantees*, not as a certification. Every invariant below
names how it is verified so you can check the claim against the code and tests
rather than trust the prose. The rule ids (`OR-0x`) index into
[OPERATIONAL_RULE_COVERAGE.md](../OPERATIONAL_RULE_COVERAGE.md), which maps each
rule to its helper, conformance assertion, and primary tests.

> **The reference implementation is not a production security boundary.**
> Everything under `monoid_agent_kernel.reference.*` is example wiring. Build
> production services against the contracts in [CONTRACTS.md](../CONTRACTS.md).

## Security goals

The kernel is designed to:

1. Keep provider credentials outside the kernel process.
2. Execute only tools admitted by the active turn's tool surface.
3. Use binding identity (`binding_id`) as the identity for authorization, quota,
   approval, capability requests, and audit.
4. Ensure gateway/lease scope can only narrow, never widen.
5. Keep workspace mutation proposal-first by default.
6. Keep public event streams redacted and content-light.
7. Preserve durable, auditable records of control commands, approvals, side
   effects, and recovery.
8. Stage external side effects durably before edge delivery.
9. Keep subagent identity, trace, accounting, and capability boundaries explicit.

## Non-goals

The kernel does **not** claim to:

1. Sandbox arbitrary Python, shell commands, MCP servers, or skill code by itself.
2. Prevent prompt injection or model deception.
3. Guarantee that model-visible workspace content is free of secrets — **the
   default path policy is permissive** (see
   [THREAT_MODEL → permissive by default](THREAT_MODEL.md#permissive-by-default)).
4. Make permissive local-development defaults safe for production.
5. Validate the business logic of custom tools.
6. Replace tenant authentication, authorization, rate limiting, or audit policy in
   the product backend.
7. Keep *private* run artifacts (`transcript.jsonl`, checkpoints) free of
   sensitive content — they are private by placement, not by redaction.
8. Provide production hardening for reference services.

## Actors and trust zones

| Zone | Contents |
|------|----------|
| A — client | Human user / client app |
| B — control plane | Product backend (authn/z, tenancy, policy, token issuance) |
| C — kernel runner | `AgentLoop` and the run it drives — **holds no provider keys** |
| D — storage | Workspace, checkpoint store, run artifacts |
| E — credential boundary | LLM gateway, Web gateway — **hold provider keys** |
| F — external | LLM/search providers, the open web |
| G — untrusted code | Custom tools, shell, skill bundles, MCP servers |
| H — observability | Event sinks, OTel exporters, diagnostics |

The credential boundary (Zone E) is the load-bearing one: the kernel (Zone C)
calls it with a short-lived, scoped token and never receives provider keys.

## Core invariants

Each invariant states what is guaranteed, how it is verified, and — where it
matters — what it does *not* cover.

### I-01 — Provider credentials stay outside the kernel
The kernel neither requires nor stores provider API keys. `GatewayModelAdapter`
is the default model path; core ships with no provider SDK dependency.
**Verified by:** design — `CONTRACTS.md` gateway boundary; the direct
`OpenAIModelAdapter` requires an explicit `--allow-direct-provider-api` opt-in.
**Does not cover:** how you secure the gateway process itself (Zone E).

### I-02 — Tool execution is surface-admitted
A tool handler runs only when its binding is present, immediate, authorized,
under quota, and active in the current turn's `ToolSurfaceSnapshot`. Unavailable,
hidden/searchable-only, denied, and quota-exceeded bindings do not execute.
**Verified by:** `OR-10-TOOL-SURFACE-ADMISSION` — `core.tool_surface.DefaultToolSurfaceResolver`;
`tests/test_tool_surface.py`, `tests/test_loop.py`, `tests/conformance/test_tool_agent_profile.py`.

### I-03 — Binding identity is the audit identity
Authorization, quota, approval, and capability requests are keyed by `binding_id`,
not by the underlying registry tool. One implementation exposed under two bindings
is two independently governed surfaces.
**Verified by:** `OR-10` (admission) and `OR-11` (ask-approval revalidates the
captured call exactly once) — `core.tool_approval`; `tests/test_tool_approval.py`,
`tests/test_loop.py`.

### I-04 — Scope only narrows
Signed scope bounds request scope; request scope bounds grant scope. Numeric caps
narrow by the smaller value; list caps by subset; wildcard domains by pattern
relation. A payload cannot widen a signed scope.
**Verified by:** `OR-01-SCOPE-RELATION` / `OR-08-PROVIDER-CAPS` —
`core.scope.scope_within`, `domain_patterns_within`, `effective_signed_scope`;
`tests/test_scope_relation.py`, `tests/test_scope_relation_properties.py`,
`tests/test_web_gateway.py`.

### I-05 — Leases carry handles, not raw secrets
A capability lease and an outbox request carry a `token_ref` **handle**, resolved
to the real secret only at the gateway/tool edge — the raw provider/tenant secret
never enters the core. Denied decisions strip grant material; public payloads are
reduced to redacted lease summaries.
**Verified by:** `OR-03-LEASE-ADMISSION` — `core.lease_admission.validate_lease_admission`,
`sanitize_denied_capability_result`; `CONTRACTS.md` (“a lease carries a `token_ref`
handle, never the secret”); `tests/test_lease_admission.py`, `tests/test_capability.py`.
**Does not cover:** *private* checkpoint/reentry payloads retain the full lease
record (handle + scope + policy fields) so a restart need not re-broker. That
record is sensitive (though not the raw secret) — protect run storage accordingly.

### I-06 — Public streams are content-light
Public events and diagnostics do not create a broad read surface for file
contents, bearer tokens, lease material, or raw tool arguments. Control-audit
events record a safe `token_sha256` reference, never the bearer token itself;
diagnostics use summaries, not raw event payloads.
**Verified by:** `OR-05-EVENT-SEQUENCING` (diagnostics summaries) and
`OR-06-CONTROL-AUDIT` — `core.control_audit.ControlAuditPolicy`,
`diagnostic_event_summary`; `tests/test_backend_control.py`, `tests/test_event_sequencing.py`.
**Does not cover:** redaction of secret-bearing tool *arguments* or shell
*commands* in your own event sinks — that is integrator responsibility (see
[OBSERVABILITY.md](../OBSERVABILITY.md#event-sinks)).

### I-07 — Workspace mutation is proposal-first by default
The default mode is `propose`: writes are staged and emitted as `diff.patch` /
`proposal.json` without mutating source-of-truth storage. `apply` (direct writes)
is an explicit opt-in.
**Verified by:** default config (`mode: propose`) and the `Workspace` contract
suite (`tests/test_workspace_contract.py`); overlay/staging backends in
`CONTRACTS.md`.
**Does not cover:** root containment against symlink escape is backend-dependent —
use a `Workspace` backend whose contract suite passes and document its symlink
behavior.

### I-08 — External side effects are durable
Strict runtimes stage external side effects in a durable outbox (checkpointed in
full) before edge delivery, or require an explicit idempotency key. The request is
persisted `pending` before the send and `dispatched`/`failed` after, so a crash
redispatches safely.
**Verified by:** `OR-12-DURABLE-SIDE-EFFECT` — `core.side_effect_policy`,
`core.outbox`; `tests/test_side_effect_policy.py`, `tests/test_outbox.py`,
`tests/conformance/test_side_effect_tool_agent_profile.py`.

### I-09 — Subagents have explicit descendant boundaries
Child runs have their own `run_id`, isolated event streams, parent-child trace
linkage, accounting roll-up, summarized diagnostics, and a capability vault fork
that shares revocation state with the parent.
**Verified by:** `OR-09-SUBAGENT-BOUNDARY` / `OR-04-REVOCATION-SCOPE` —
`core.subagent_runtime`, `CapabilityVault.fork_for_child`; `tests/test_subagent.py`,
`tests/conformance/test_multi_agent_profile.py`.

## Integrator responsibilities

Monoid preserves runtime boundaries; the integrator owns deployment trust. You
are responsible for:

- tenant authentication and authorization;
- workspace root selection and per-tenant isolation;
- the deny/redact path policy (there is **no** secure default — see I‑03 non-goal);
- provider credential storage and gateway operation;
- gateway signing-key rotation, token TTLs, and revocation policy;
- event/transcript/checkpoint retention and access control;
- redaction in your own event sinks and OTel exporters;
- shell sandboxing and network egress policy;
- MCP server allowlisting and skill-bundle review/signing;
- memory retention/deletion policy;
- outbox sender idempotency and dead-letter handling;
- production hardening of any reference-derived service.

See [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md) for the actionable form.

## Known limitations and future hardening

- **Coarse file-mutation authorization.** `fs.copy`/`fs.move`/`fs.delete` require
  approval, but authorization is per-binding, not yet argument-aware
  (path-scoped). Argument-aware file authorization is future hardening.
- **No built-in shell/skill/MCP sandbox.** Execution isolation is delegated to the
  environment you run the kernel in.
- **Reference services are not hardened.** Multi-tenant isolation, rate limiting,
  and secret management in `reference.*` are illustrative only.

## Verification

The invariants above are not self-attestation: each names an operational rule and
its primary tests. To re-check the model against the code, run the relevant
conformance profiles (see [CONFORMANCE.md](../CONFORMANCE.md)) and the referenced
`tests/*`. If an invariant and its tests ever disagree, the tests win and this
document is the bug.
