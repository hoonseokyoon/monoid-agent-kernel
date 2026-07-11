# Monoid Agent Kernel

*A lightweight durable agent kernel for embedding product-grade agents anywhere: contract-first, observable, permission-aware, and replaceable at every seam.*

[![CI](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/workflows/ci.yml/badge.svg)](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/pyproject.toml)

Monoid is the small runtime core you put inside a larger product when you need
agents to run reliably. It owns the loop, durable artifacts, tool execution,
permissions, observability, subagents, skills, and gateway contracts while leaving
deployment choices to your platform. Models, tools, workspace storage, checkpoint
stores, event sinks, capability brokers, memory, and gateway services are all
replaceable contracts.

> Throughout these docs, **"your gateway" / "your backend platform"** refers to the
> backend you operate — the credential boundary that hosts the LLM and Web gateways. The
> kernel never holds provider keys; it calls your gateway with a short-lived, scoped token.

**New here?** Run the [Quickstart](#quickstart-no-servers) (no servers, no API key),
then follow the [Documentation map](#documentation) to the path for your role.

## See it run

The bundled **Agent Studio** reference app (`monoid studio serve`) drives the kernel
through its Python API behind a single-page UI. A profile chooses the model, reasoning
level, prompt instructions, and tool surface; each profile keeps its own chat history.

![Agent Studio: a Visual Analyst profile reads sales data, writes insights, and generates an annotated revenue chart](https://raw.githubusercontent.com/hoonseokyoon/monoid-agent-kernel/main/docs/img/studio-v016-main.png)

*A real Studio run: the agent reads `sales.csv`, writes `INSIGHTS.md`, generates an
annotated `revenue_trend.svg`, and previews the artifact directly in the workspace panel.*

![Agent Studio profile builder showing the exact system prompt and tool schema preview](https://raw.githubusercontent.com/hoonseokyoon/monoid-agent-kernel/main/docs/img/studio-v016-profile-builder.png)

*The profile editor shows the exact first-turn model request boundary: system prompt,
tool schemas, model settings, and preview notes. Users can edit the profile on the left
and see what the model will receive on the right.*

## Architecture: Contract / Conformance Test / Core Helper Kit / Reference

The package is organized around four roles:

- **Contract** — the stable integration surface, collected in `monoid_agent_kernel.contracts`
  and re-exported from the top-level `monoid_agent_kernel`. These are the specs and protocols you
  depend on and implement: `AgentLoop`, `AgentRunSpec`, `AgentRuntimeConfig`, `ModelAdapter`,
  `ToolSpec` / `@tool`, `EventSink`, `CheckpointStore`, `PermissionPolicy`, and the rest. See
  [docs/CONTRACTS.md](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/CONTRACTS.md) for the Python, HTTP, wiring, and operational rules.
- **Conformance Test** — profile-based tests that check contract behavior for a chosen runtime
  shape. See [docs/CONFORMANCE.md](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/CONFORMANCE.md) for the profile model and
  [docs/OPERATIONAL_RULE_COVERAGE.md](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/OPERATIONAL_RULE_COVERAGE.md) for the
  rule-to-test and Phase 2S hardening coverage matrix.
- **Core Helper Kit** — the supported runtime and helper modules that make the contract easy to
  satisfy (`loop.py`, `core/`, `providers/`, `tools/`, `workspace/`, …). See
  [docs/CORE_HELPER_KIT.md](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/CORE_HELPER_KIT.md) for the helper boundary and
  validation/library policy.
- **Reference** — example services under `monoid_agent_kernel.reference` (`backend`,
  `llm_gateway`, `web_gateway`, `mcp_gateway`, `stores`, `studio`, `conformance`) assembled from
  the public contract and helper kit. See [docs/REFERENCE.md](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/REFERENCE.md) for the reference
  role, harnesses, and smoke targets.

For the dynamic binding-based tool surface, see
[docs/TOOL_SURFACE.md](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/TOOL_SURFACE.md).

## Install

```bash
pip install monoid-agent-kernel
```

Core has no provider SDK dependency. The direct OpenAI adapter is for local smoke tests;
hosted/product runs use `GatewayModelAdapter` through your gateway:

```bash
pip install "monoid-agent-kernel[openai]"
```

## Quickstart (no servers)

The smallest kernel run needs three of your objects — a spec, a model adapter, and a runtime
config — and `from_config` wires them in one call. `FakeModelAdapter` (a scripted model)
makes the first turn run offline, with no gateway or API key:

```python
from pathlib import Path

from monoid_agent_kernel import AgentLoop, AgentRunSpec, AgentRuntimeConfig, RegistryToolRef, ToolBinding
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call

workspace = Path("workspace")
run_root = Path("runs")
workspace.mkdir(exist_ok=True)
run_root.mkdir(exist_ok=True)
(workspace / "notes.md").write_text("alpha\nbeta\n", encoding="utf-8")

spec = AgentRunSpec(workspace_root=workspace, run_root=run_root, mode="apply")
config = AgentRuntimeConfig(
    definition_id="quickstart",
    tools=(
        ToolBinding(binding_id="fs.read", ref=RegistryToolRef("fs.read")),
        ToolBinding(binding_id="fs.write", ref=RegistryToolRef("fs.write")),
    ),
)
adapter = FakeModelAdapter(
    turns=[
        ModelTurn(tool_calls=(fake_tool_call("fs_read", {"path": "notes.md"}, "read1"),)),
        ModelTurn(
            tool_calls=(
                fake_tool_call(
                    "fs_write",
                    {"path": "SUMMARY.md", "content": "alpha and beta\n"},
                    "write1",
                ),
            )
        ),
        ModelTurn(final_text="Wrote SUMMARY.md."),
    ]
)

result = AgentLoop.from_config(spec, adapter, config).run_once("Summarize notes.md")
print(result.final_text)
print((workspace / "SUMMARY.md").read_text(encoding="utf-8"))
```

`from_config`'s `runtime_config` accepts a bare `AgentRuntimeConfig`, a
`RuntimeConfigProvider`, or a `callable(run_id) -> AgentRuntimeConfig` (hot-reload). See
[`examples/minimal_quickstart.py`](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/examples/minimal_quickstart.py) for a complete file and
[`examples/custom_model_adapter.py`](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/examples/custom_model_adapter.py) for implementing
your own `ModelAdapter`. Author tools from typed functions with the `@tool` decorator
(see [`examples/custom_tools/word_count_tool.py`](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/examples/custom_tools/word_count_tool.py));
`generated_tool_bindings(...)` then turns a set of `ToolSpec`s into bindings.

## Memory and default tools

`monoid_agent_kernel.memory` provides an optional provider-backed Memory tool surface. The
kernel treats memory as ordinary tools and context supplied by a provider; the provider owns the
storage shape. The bundled `LocalFilesystemMemoryProvider` maps the Claude-style `/memories`
virtual tree to local files and exposes `memory.search`, `memory.view`, `memory.create`,
`memory.str_replace`, `memory.insert`, `memory.delete`, and `memory.rename`. Read tools default
to `allow`; write tools default to `ask`.

Memory providers are attached explicitly by an app, backend, or Studio profile. They are
available from `monoid_agent_kernel.memory` and stay out of the top-level contract exports and
`builtin_tools(workspace)`.

The helper `default_tool_bindings(...)` in `monoid_agent_kernel.tools.defaults` creates the
standard read, write, shell, and artifact tool bundles used by Studio and the builder. The write
bundle includes `fs.write`, `fs.patch`, `fs.mkdir`, `fs.copy`, `fs.move`, and `fs.delete`;
`fs.copy`, `fs.move`, and `fs.delete` require approval by default.

## Stability

This package is pre-1.0 (`0.x`): the public surface may change between minor versions, but
breaking changes are called out in commit messages and this README.

- **Stable Contract** — the core engine and integration contracts exported from
  `monoid_agent_kernel.contracts`: `AgentLoop`, `AgentRunSpec`, `AgentRuntimeConfig` /
  `RuntimeConfigProvider`, `ModelAdapter`, `ToolSpec` / `@tool`, `EventSink`,
  `CheckpointStore`, `Workspace` / `workspace_factory`, and `PermissionPolicy`.
- **Contract Extension** — surfaces that are public but still settling: async task seams,
  session lifecycle/control, capability leases, agent-as-tool delegation, Agent Skills,
  output validation, and multimodal content parts. `ImagePart` and `DocumentPart` are
  forwarded to multimodal-capable adapters. `AudioPart` / `VideoPart` are exported
  content contracts and round-trip through core JSON/checkpoint paths; provider forwarding
  is still adapter-specific.
- **Helper Kit** — implementation helpers live under explicit modules such as
  `monoid_agent_kernel.core.*`, `monoid_agent_kernel.providers.*`,
  `monoid_agent_kernel.tools.*`, `monoid_agent_kernel.recorder`, and
  `monoid_agent_kernel.observability`.
- **Reference examples** — everything under `monoid_agent_kernel.reference.*` is example
  implementation code; build production services against the contracts.

Agent configuration is centered on `AgentDefinition` (the reusable blueprint) and the
mutable `AgentRuntimeConfig` (the current prompt and `ToolBinding` set). Backends can replace
runtime config mid-run; the kernel applies it at the next turn boundary.

## Running agents from the CLI

```bash
monoid run \
  --workspace examples/workspaces/edit_markdown_notes \
  --instruction "Read notes.md and create a clearer summary in SUMMARY.md." \
  --runtime-config-file examples/runtime-config.json \
  --llm-gateway-url http://127.0.0.1:8080/internal/llm/turns
```

- **Run spec vs runtime config are separate.** `AgentRunSpec` carries workspace, limits, and
  permission boundary; the instruction is delivered as the first user turn. `AgentRuntimeConfig`
  carries model, prompt, tool bindings, guidance, scope, quota, and shell/web runtime.
- **`propose` vs `apply`.** The default `propose` mode stages writes and emits a proposal
  package (`diff.patch`, `proposal.json`) without mutating the workspace; `--mode apply` writes
  directly.
- **Permissions are permissive by default** — dotfiles and keys are treated as normal files.
  Pass `--deny-path` / `--redact-path` if the workspace holds secrets.
  **See the [Threat Model](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/security/THREAT_MODEL.md#permissive-by-default) before exposing secret-bearing workspaces.**
- **Optional surfaces, each off unless flagged:** `--agents-directory` (subagents via
  `agent.spawn`), `--skills-directory` (Agent Skills), `--capability-broker` (leased tools).

Full CLI reference — `run`, `builder`, `watch`, `proposal`, `jobs`, custom workspace
backends, streaming JSON — is in **[docs/CLI.md](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/CLI.md)**.

## Hosted backend and gateways

For hosted, multi-tenant runs over HTTP, the reference backend issues run tokens and starts
kernel runs, while the LLM and Web gateways hold provider credentials and validate scoped
tokens. The kernel never receives OpenAI, Anthropic, or search-provider keys.

The end-to-end walkthrough — starting the gateways, creating a run with `curl`, and polling
status/result/events/proposal — is in **[docs/BACKEND.md](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/BACKEND.md)**.

## Model Provider Boundary

`GatewayModelAdapter` is the default path. It sends normalized model-turn
requests to your LLM gateway and can authenticate with
`MONOID_LLM_GATEWAY_TOKEN` or `--llm-gateway-token-file`. Provider credentials stay
inside your backend platform, where tenant usage, budgets, and rate limits
can be enforced.

`OpenAIModelAdapter` is retained for local smoke tests. CLI use requires
`runtime_config.model.provider="openai"` and `--allow-direct-provider-api`.

To target your own LLM gateway, implement the `ModelAdapter` protocol or the
`monoid.llm-turn.v1` HTTP contract documented in
[docs/CONTRACTS.md](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/CONTRACTS.md). Current protocol and schema identifiers
use `monoid.*`; `native-agent-runner.*` identifiers are accepted during migration
for existing durable artifacts and gateway requests.

## Observability

Every run emits a structured event stream (`events.jsonl`) and durable artifacts, and can
mirror that stream to OpenTelemetry — all without the core capturing prompt/response content.
`AgentLoop.astream(user_input)` exposes live token deltas, and each run writes `metrics.json`
with counters, timing, and token usage.

The run-directory artifact set, custom event sinks (including secret redaction), OTel tracing,
live streaming, and metrics are all documented in **[docs/OBSERVABILITY.md](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/OBSERVABILITY.md)**.

## Defaults

- runtime config is required for CLI and backend runs
- default model provider inside `ModelConfig`: `gateway`
- default model inside `ModelConfig`: `gpt-5.5`
- default reasoning effort inside `ModelConfig`: `medium`
- mode: `propose`
- default tool bundles are available through `monoid_agent_kernel.tools.defaults.default_tool_bindings`
- shell is available through exposed shell bindings such as `shell.exec`
- web.search/web.fetch/web.context are available through exposed web bindings and WebGateway
- file mutation tools include write, patch, mkdir, copy, move, and delete in
  `propose` and `apply` modes when bound in runtime config
- memory tools are available through an explicitly attached `MemoryProvider`
- no path deny/redact policy unless explicitly provided

## Documentation

Start with **[docs/README.md](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/README.md)** — it routes you to the right docs by role
(app developer, tool author, backend/gateway operator, security reviewer, contributor).
Quick links:

- **Use it:** [CLI reference](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/CLI.md) · [Hosted backend](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/BACKEND.md) · [Observability](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/OBSERVABILITY.md)
- **Build against it:** [Embedding handbook](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/EMBEDDING.md) · [Contracts](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/CONTRACTS.md) · [Tool surface](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/TOOL_SURFACE.md) · [Conformance](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/CONFORMANCE.md)
- **Extend it:** [Subagents](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/SUBAGENT_DESIGN.md) · [Skills](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/SKILLS_DESIGN.md) · [First skill tutorial](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/FIRST_SKILL_TUTORIAL.md)
- **Secure it:** [Security model](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/security/SECURITY_MODEL.md) · [Threat model](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/security/THREAT_MODEL.md) · [Production checklist](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/security/PRODUCTION_CHECKLIST.md) · [Security policy](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/SECURITY.md)

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/CONTRIBUTING.md) for
development setup and the lint/test workflow, and [CODE_OF_CONDUCT.md](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/CODE_OF_CONDUCT.md).
For security issues, follow [SECURITY.md](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/SECURITY.md) (do not open a public issue).

Fast local confidence checks:

```bash
python -m pytest tests/conformance -q
python -m pytest -q -n 4
python -m pytest -q --cov=monoid_agent_kernel --cov=native_agent_runner
```

CI keeps the serial suite as the required gate and runs xdist plus coverage as
advisory checks while the test seams stabilize. See
[docs/PHASE_4_CLOSURE.md](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/docs/PHASE_4_CLOSURE.md) for the current Phase 4
structure closure and CI promotion criteria.

## License

Licensed under the [Apache License 2.0](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/main/LICENSE).
