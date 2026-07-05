# Monoid Documentation

Start with the top-level [README](../README.md) for the product position,
install, quickstart, and run model. These docs go deeper on the contracts that
make Monoid embeddable in different products and runtimes:

| Doc | What it covers |
|-----|----------------|
| [CONTRACTS.md](CONTRACTS.md) | The stable integration surface: Python contracts, HTTP wire contracts, wiring rules, operational rules, and the contract/core/reference boundary. |
| [CONFORMANCE.md](CONFORMANCE.md) | Profile-based conformance tests, harness roles, executable assertions, testing policy, and the Reference full profile. |
| [CORE_HELPER_KIT.md](CORE_HELPER_KIT.md) | Supported helper modules and validation/library policy for contract-aligned implementations. |
| [REFERENCE.md](REFERENCE.md) | Runnable Reference services, Studio, gateway examples, and the public Reference conformance harness. |
| [OPERATIONAL_RULE_COVERAGE.md](OPERATIONAL_RULE_COVERAGE.md) | Current rule-to-helper-to-profile-to-test coverage matrix and Phase 2S hardening matrix for OR-01 through OR-13. |
| [RUNNER_BACKEND_RESPONSIBILITY_MAP.md](RUNNER_BACKEND_RESPONSIBILITY_MAP.md) | RunnerBackend facade/service split, private service responsibilities, and remaining runtime ownership. |
| [PHASE_4_CLOSURE.md](PHASE_4_CLOSURE.md) | Phase 4 completion criteria, CI status, remaining flake risk, and structural closure position. |
| [PHASE_1S_COVERAGE.md](PHASE_1S_COVERAGE.md) | Historical Phase 1S coverage pointer. |
| [TOOL_SURFACE.md](TOOL_SURFACE.md) | The dynamic, binding-based tool surface — `ToolBinding`, model-name aliasing, exposure/authorization/guidance/scope/quota, and how bindings resolve against the registry. |
| [SUBAGENT_DESIGN.md](SUBAGENT_DESIGN.md) | Agent-as-tool delegation — isolated child runs via the `agent.spawn` tool, progressive disclosure through dynamic context providers. |
| [SKILLS_DESIGN.md](SKILLS_DESIGN.md) | Agent Skills — procedural knowledge delivered through a `ContextProvider`, complementing subagents (execution) and MCP (integration). |
| [FIRST_SKILL_TUTORIAL.md](FIRST_SKILL_TUTORIAL.md) | Create and smoke-test a minimal `SKILL.md` skill locally. |

For observability (OpenTelemetry tracing, the streaming surface, `metrics.json`), see the
[Observability](../README.md#observability) section of the top-level README and the runnable
[`examples/otel_tracing.py`](../examples/otel_tracing.py).
