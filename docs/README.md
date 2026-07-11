# Monoid Documentation

Start with the top-level [README](../README.md) for the product position,
install, and the no-server quickstart. This index goes deeper on using,
building against, and securing the kernel.

## Find your path

- **I want to run an agent** → [Quickstart](../README.md#quickstart-no-servers),
  then [CLI.md](CLI.md) for the full `run` / `watch` / `jobs` surface.
- **I'm embedding the kernel in a product** → [CONTRACTS.md](CONTRACTS.md) for the
  stable Python/HTTP surface, [EMBEDDING.md](EMBEDDING.md) for portable responsibilities and
  executable local and hosted paths, [CONFORMANCE.md](CONFORMANCE.md) to check your
  implementation, [CORE_HELPER_KIT.md](CORE_HELPER_KIT.md) for the helper boundary,
  [COMPATIBILITY.md](COMPATIBILITY.md) for version and upgrade obligations.
- **I'm writing tools** → [TOOL_SURFACE.md](TOOL_SURFACE.md) for `ToolBinding`,
  exposure, authorization, scope, and quota.
- **I'm running a backend/gateway** → [BACKEND.md](BACKEND.md) for the reference
  wiring and token boundary, [OBSERVABILITY.md](OBSERVABILITY.md) for outputs,
  event sinks, and OTel, [DBOS_REFERENCE.md](DBOS_REFERENCE.md) for the experimental
  finite-activation recovery proof and explicit non-goals.
- **I'm reviewing security** → [security/SECURITY_MODEL.md](security/SECURITY_MODEL.md)
  for verified invariants and boundaries, [security/THREAT_MODEL.md](security/THREAT_MODEL.md)
  for the threat-by-threat breakdown, [security/PRODUCTION_CHECKLIST.md](security/PRODUCTION_CHECKLIST.md)
  before deploying, [SECURITY.md](../SECURITY.md) for reporting.
- **I'm extending with delegation/skills** → [SUBAGENT_DESIGN.md](SUBAGENT_DESIGN.md),
  [SKILLS_DESIGN.md](SKILLS_DESIGN.md), [FIRST_SKILL_TUTORIAL.md](FIRST_SKILL_TUTORIAL.md).
- **I'm contributing** → [CONTRIBUTING.md](../CONTRIBUTING.md), then
  [OPERATIONAL_RULE_COVERAGE.md](OPERATIONAL_RULE_COVERAGE.md) and
  [PHASE_4_CLOSURE.md](PHASE_4_CLOSURE.md) for how behavior is verified.

## All documents

| Doc | What it covers |
|-----|----------------|
| [CLI.md](CLI.md) | Full `monoid` CLI: `run`, `builder`, `watch`, `proposal`, `jobs`, modes, custom workspace backends, streaming, and path permissions. |
| [BACKEND.md](BACKEND.md) | Reference backend + LLM/Web gateway walkthrough: starting the services, creating a run over HTTP, and the token boundary. |
| [DBOS_REFERENCE.md](DBOS_REFERENCE.md) | Experimental DBOS activation-recovery profile, stable-slot crash invariant, ownership boundary, and v0.18 non-goals. |
| [OBSERVABILITY.md](OBSERVABILITY.md) | Run-directory artifact set, custom event sinks, OpenTelemetry tracing, live streaming, and metrics. |
| [security/SECURITY_MODEL.md](security/SECURITY_MODEL.md) | Intended security boundaries, non-goals, actors/trust zones, and verified core invariants (each mapped to an operational rule and its tests). |
| [security/THREAT_MODEL.md](security/THREAT_MODEL.md) | Trust boundaries, the permissive-by-default warning, threat-by-threat defenses, and integrator responsibilities. |
| [security/PRODUCTION_CHECKLIST.md](security/PRODUCTION_CHECKLIST.md) | Actionable pre-deployment checklist for gateways, workspace, tool surface, artifacts, and conformance. |
| [CONTRACTS.md](CONTRACTS.md) | The stable integration surface: Python contracts, HTTP wire contracts, wiring rules, operational rules, and the contract/core/reference boundary. |
| [EMBEDDING.md](EMBEDDING.md) | Portable production responsibilities plus executable local and hosted/multi-tenant golden paths using one explicit Reference assembly. |
| [CONFORMANCE.md](CONFORMANCE.md) | Profile-based conformance tests, harness roles, executable assertions, testing policy, and the Reference full profile. |
| [CORE_HELPER_KIT.md](CORE_HELPER_KIT.md) | Supported helper modules and validation/library policy for contract-aligned implementations. |
| [COMPATIBILITY.md](COMPATIBILITY.md) | Machine-checked wire/durable version ledger, aliases, deprecation policy, mixed-version operation, and upgrade/rollback playbooks. |
| [REFERENCE.md](REFERENCE.md) | Runnable Reference services, Studio, gateway examples, and the public Reference conformance harness. |
| [OPERATIONAL_RULE_COVERAGE.md](OPERATIONAL_RULE_COVERAGE.md) | Current rule-to-helper-to-profile-to-test coverage matrix and Phase 2S hardening matrix for OR-01 through OR-13. |
| [RUNNER_BACKEND_RESPONSIBILITY_MAP.md](RUNNER_BACKEND_RESPONSIBILITY_MAP.md) | RunnerBackend facade/service split, private service responsibilities, and remaining runtime ownership. |
| [PHASE_4_CLOSURE.md](PHASE_4_CLOSURE.md) | Phase 4 completion criteria, CI status, remaining flake risk, and structural closure position. |
| [V0_18_RELEASE_AUDIT.md](V0_18_RELEASE_AUDIT.md) | Frozen-range multi-agent release audit, cross-challenges, dispositions, and hardening evidence. |
| [PHASE_1S_COVERAGE.md](PHASE_1S_COVERAGE.md) | Historical Phase 1S coverage pointer. |
| [TOOL_SURFACE.md](TOOL_SURFACE.md) | The dynamic, binding-based tool surface — `ToolBinding`, model-name aliasing, exposure/authorization/guidance/scope/quota, and how bindings resolve against the registry. |
| [SUBAGENT_DESIGN.md](SUBAGENT_DESIGN.md) | Agent-as-tool delegation — isolated child runs via the `agent.spawn` tool, progressive disclosure through dynamic context providers. |
| [SKILLS_DESIGN.md](SKILLS_DESIGN.md) | Agent Skills — procedural knowledge delivered through a `ContextProvider`, complementing subagents (execution) and MCP (integration). |
| [FIRST_SKILL_TUTORIAL.md](FIRST_SKILL_TUTORIAL.md) | Create and smoke-test a minimal `SKILL.md` skill locally. |
