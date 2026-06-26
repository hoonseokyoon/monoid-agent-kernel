# Native Agent Runner

*A provider-neutral, permission-aware agent runtime for safe, structured file work — secrets stay outside the engine, and every seam (model, tools, workspace, checkpoint store) is replaceable.*

[![CI](https://github.com/hoonseokyoon/native-agent-runner/actions/workflows/ci.yml/badge.svg)](https://github.com/hoonseokyoon/native-agent-runner/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

> **Terminology:** Throughout these docs, **CSP** means the *Cloud Service Provider / backend
> platform you operate* — the credential boundary that hosts the LLM and Web gateways. The runner
> itself never holds provider keys; it calls your gateway with a short-lived, scoped token.

Standalone API-backed agent harness for safe, structured file work in a
workspace. The workspace is a pluggable seam: the engine ships with a
local-filesystem backend and accepts your own implementation (see
[Custom workspace backend](#custom-workspace-backend)). This research package
intentionally has no dependency on CSP runtime modules. CSP integration is a
later adapter layer.

## Boundary: contracts / core / reference

The package is layered in three tiers:

- **contracts** — the stable integration surface, collected in `native_agent_runner.contracts`
  and re-exported from the top-level `native_agent_runner`. These are the specs and protocols you
  depend on and implement: `AgentLoop`, `AgentRunSpec`, `AgentRuntimeConfig`, `ModelAdapter`,
  `ToolSpec` / `@tool`, `EventSink`, `CheckpointStore`, `PermissionPolicy`, and the rest. See
  [docs/CONTRACTS.md](docs/CONTRACTS.md) for the Python and HTTP wire contracts.
- **core** — the engine that implements those contracts: the default, batteries-included runner
  (`loop.py`, `core/`, `providers/`, `tools/`, `workspace/`, …). This is the supported
  implementation you actually run.
- **reference** — example services under `native_agent_runner.reference` (`backend`,
  `llm_gateway`, `web_gateway`, `stores`). **Not** part of the supported surface: core never
  imports them, and real integrators are expected to build their own services against the
  contracts.

For the dynamic binding-based tool surface, see
[docs/TOOL_SURFACE.md](docs/TOOL_SURFACE.md).

## Install

```bash
pip install native-agent-runner
```

Core has no provider SDK dependency. The direct OpenAI adapter (local smoke tests only;
container/CSP runs use `GatewayModelAdapter`) is an optional extra:

```bash
pip install "native-agent-runner[openai]"
```

## Quickstart (no servers)

The smallest run needs three of your objects — a spec, a model adapter, and a runtime
config — and `from_config` wires them in one call. `FakeModelAdapter` (a scripted model)
makes the first turn run offline, with no gateway or API key:

```python
from native_agent_runner import AgentLoop, AgentRunSpec, AgentRuntimeConfig, FakeModelAdapter
from native_agent_runner import RegistryToolRef, ToolBinding
from native_agent_runner.providers.base import ModelTurn

spec = AgentRunSpec(workspace_root="./workspace", mode="apply")
config = AgentRuntimeConfig(
    definition_id="quickstart",
    tools=(ToolBinding(binding_id="fs.write", ref=RegistryToolRef("fs.write")),),
)
adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])

result = AgentLoop.from_config(spec, adapter, config).run_once("Summarize notes.md")
```

`from_config`'s `runtime_config` accepts a bare `AgentRuntimeConfig`, a
`RuntimeConfigProvider`, or a `callable(run_id) -> AgentRuntimeConfig` (hot-reload). See
[`examples/minimal_quickstart.py`](examples/minimal_quickstart.py) for a complete file and
[`examples/custom_model_adapter.py`](examples/custom_model_adapter.py) for implementing
your own `ModelAdapter`. Author tools from typed functions with the `@tool` decorator
(see [`examples/custom_tools/word_count_tool.py`](examples/custom_tools/word_count_tool.py));
`generated_tool_bindings(...)` then turns a set of `ToolSpec`s into bindings.

## Stability

This package is pre-1.0 (`0.x`): the public surface may change between minor versions, but
breaking changes are called out in commit messages and this README.

- **Stable** — the core engine and the contracts it implements: `AgentLoop`, `AgentRunSpec`,
  `AgentRuntimeConfig` / `RuntimeConfigProvider`, `ModelAdapter`, `ToolSpec` / `@tool`,
  `EventSink`, `CheckpointStore`, `Workspace` / `workspace_factory`, `PermissionPolicy`, and
  the rest of `native_agent_runner.contracts`.
- **Experimental** — surfaces still settling: the async-task seams (`TaskExecutor`,
  `ResultInjector`, `TaskReporter`); the session lifecycle + control surface (`AgentSession` /
  `LoopSession`, `SessionState`, `ControlCommand` / `ControlDispatcher`); capability leases
  (`CapabilityBroker` / `CapabilityLease`); agent-as-tool delegation (`SubagentDefinition`)
  and Agent Skills (`SkillProvider`); and the multimodal content parts. `ImagePart` and
  `DocumentPart` are forwarded to multimodal-capable adapters (the gateway and OpenAI
  adapters); a text-only adapter drops them with a `model.input.degraded` warning.
  `AudioPart` / `VideoPart` round-trip as a forward-compatible contract but are not yet
  forwarded.
- **Not a contract** — everything under `native_agent_runner.reference.*` is an example
  implementation; build your own services against the contracts instead.

Agent configuration is centered on `AgentDefinition` (the reusable blueprint) and the
mutable `AgentRuntimeConfig` (the current prompt and `ToolBinding` set). Backends can replace
runtime config mid-run; the runner applies it at the next turn boundary.

## Run

```bash
native-agent run \
  --workspace examples/workspaces/edit_markdown_notes \
  --instruction "Read notes.md and create a clearer summary in SUMMARY.md." \
  --runtime-config-file examples/runtime-config.json \
  --llm-gateway-url http://127.0.0.1:8080/internal/llm/turns
```

Run spec and runtime config are separate. `AgentRunSpec` carries workspace,
limits, and permission boundary values — it no longer carries the instruction,
which is delivered as the first user turn (CLI `--instruction`, or
`AgentLoop.run_once()` / `submit()` programmatically). `AgentRuntimeConfig`
carries model, prompt, tool bindings, guidance, scope, quota, shell runtime, and
web runtime values. You can pass a run spec JSON file with a runtime config
file:

```bash
native-agent run \
  --spec examples/run-spec.json \
  --instruction "Read notes.md and create a clearer summary in SUMMARY.md." \
  --runtime-config-file examples/runtime-config.json
```

Programmatic callers drive the run with `AgentLoop.run_once(instruction)` for the
one-shot case, or `open()` → `submit(user_input)`* → `close()` for a multi-turn
session in a single run. Each `submit()` settles when the model returns final
text with no tool calls; the workspace and model continuation thread across
submits. `commit_checkpoint()` re-baselines the proposal between turns when you
want incremental apply.

The default mode is `propose`, which means the runner creates a proposal package
without committing to tenant source-of-truth storage. Local CLI runs default to
`--workspace-backend overlay`, so writes are staged in an overlay and emitted as
`runs/<run_id>/diff.patch` and `runs/<run_id>/proposal.json` without modifying
the workspace. Container/CSP-style runs can use `--workspace-backend staging`,
where tools and shell write directly to a staging workspace and the runner
compares that workspace with `workspace.base.json` to generate the proposal.
Use `--mode apply` for local direct workspace writes.

### Custom workspace backend

The runner never touches the filesystem directly — it works through a `Workspace`
(the file-storage surface in `native_agent_runner.contracts`). `AgentLoop` builds one
per run with `workspace_factory(spec)`, defaulting to `default_local_workspace_factory`,
which returns the local-filesystem backend. Supply your own factory to back a run with a
different store — a git worktree, an object store, a remote or in-memory filesystem —
without changing the engine:

```python
from native_agent_runner import AgentLoop, Workspace

def my_workspace_factory(spec) -> Workspace:
    return MyWorkspace(spec.workspace_root, mode=spec.mode)

loop = AgentLoop.from_config(spec, adapter, config, workspace_factory=my_workspace_factory)
```

A custom backend must honor the `Workspace` contract suite
(`tests/test_workspace_contract.py`) to be a drop-in: add one `pytest.param` for your
factory and the existing invariants run against it.

The default model provider is `gateway`. Container runs should call an internal
CSP LLM gateway with a short-lived run token. The runner should not receive
OpenAI, Anthropic, or other provider API keys.

Web tools are also gateway-backed. `web.search`, `web.fetch`, and `web.context`
are available when runtime config binds those registry tools. The runner calls a
CSP WebGateway with a short-lived `web_gateway` token. The runner does not
perform direct web egress and does not receive search-provider credentials.
`web.context` returns
LLM-ready grounding context through a provider-neutral ContextProvider contract.

Shell is available when runtime config binds `shell.exec`, which supports foreground
commands and run-scoped background jobs. A background call returns a `job_id` immediately;
the runner feeds the job's result back to the model when it finishes (inspect jobs with the
`jobs` / `job` CLI commands below).

Path permission defaults are permissive: the runner treats every root-contained file as a
normal workspace file, including dotfiles and keys. Backends can explicitly deny or redact
paths per run:

```bash
native-agent run \
  --workspace examples/workspaces/edit_markdown_notes \
  --instruction "Inspect this workspace." \
  --runtime-config-file examples/runtime-config.json \
  --deny-path ".env" \
  --redact-path "*.key"
```

`--permission-policy-file policy.json` accepts:

```json
{
  "deny_patterns": [".env", "*.key"],
  "redact_patterns": ["internal/**"]
}
```

`deny_patterns` blocks tool and shell access. `redact_patterns` masks paths in the public
event/status stream only; private run artifacts keep real paths and contents.

Public events are not heuristically scrubbed for secrets: the runner keeps file-content out
of the public stream and masks `redact_patterns` paths, but redacting secret-bearing tool
arguments or shell commands is the backend's responsibility (see [Event Sinks](#event-sinks)).

### Subagents, Skills, and capability gating

Three optional features on `native-agent run`, each off unless its flag is set:

- `--agents-directory DIR` — load subagent definitions (`*.md` with frontmatter) from
  `DIR`, enabling the `agent.spawn` tool so the model can delegate to isolated child runs.
- `--skills-directory DIR` — load Agent Skills (`SKILL.md` with frontmatter) from `DIR`,
  enabling the progressive-disclosure skill tools.
- `--capability-broker path.py:factory` — load a `CapabilityBroker` that gates any tool
  declaring `runtime.requires_lease` behind a scoped, short-lived lease. For local dev,
  `--auto-grant-capabilities` uses the built-in `AutoGrantBroker` (grants every request,
  scoped to its binding) instead. Pass at most one of the two.

For machine-readable real-time progress:

```bash
native-agent run \
  --workspace examples/workspaces/edit_markdown_notes \
  --instruction "Read notes.md and create a clearer summary in SUMMARY.md." \
  --runtime-config-file examples/runtime-config.json \
  --llm-gateway-url http://127.0.0.1:8080/internal/llm/turns \
  --stream-json
```

`--stream-json` writes public redacted events to stdout as JSON Lines. Human
status output goes to stderr in this mode.

## Watch

Replay or follow a run's public event stream:

```bash
native-agent watch <run_id> --run-root ./runs --from-start --json
native-agent watch <run_id> --run-root ./runs --follow
```

`--json` prints raw JSONL events. The default watch output is a compact human
view.

Inspect the current proposed output snapshot:

```bash
native-agent proposal <run_id> --run-root ./runs
native-agent proposal <run_id> --run-root ./runs --file SUMMARY.md --json
```

Inspect background shell jobs and logs:

```bash
native-agent jobs <run_id> --run-root ./runs
native-agent job status <job_id> --run <run_id> --run-root ./runs --json
native-agent job logs <job_id> --run <run_id> --stream stdout --tail-bytes 4096
native-agent job cancel <job_id> --run <run_id>
```

## Backend (reference)

> Reference example (`native_agent_runner.reference.backend`). Not part of the supported public
> surface — build your own backend against the contracts in [docs/CONTRACTS.md](docs/CONTRACTS.md).

The standalone backend issues run tokens, starts runner jobs, and exposes status,
result, event, and tenant usage APIs. It still uses the keyless gateway model
provider. Provider API keys stay outside the runner backend.

Start a local LLM gateway. This process is the provider-credential boundary:

```bash
export NAR_BACKEND_ADMIN_TOKEN="admin-dev-token"
export NAR_LLM_GATEWAY_ADMIN_TOKEN="llm-admin-dev-token"
export NAR_BACKEND_TOKEN_SECRET="replace-with-32-plus-random-bytes"

native-agent llm-gateway serve \
  --host 127.0.0.1 \
  --port 8080
```

Start the runner backend in another process. It shares the token signing secret
with the LLM and Web gateways so it can issue scoped gateway tokens:

```bash
native-agent backend serve \
  --workspace-root /workspaces \
  --run-root ./runs \
  --llm-gateway-url http://127.0.0.1:8080/internal/llm/turns \
  --web-gateway-url http://127.0.0.1:8090
```

For local contract testing, start the reference fake WebGateway:

```bash
export NAR_WEB_GATEWAY_ADMIN_TOKEN="web-admin-dev-token"

native-agent web-gateway serve \
  --host 127.0.0.1 \
  --port 8090 \
  --provider fake
```

For a real search smoke, use Brave Search for `web.search` and the gateway's
direct HTTP fetcher for `web.fetch`. Add `--context-provider brave-llm` to use
Brave's LLM Context endpoint for `web.context`, or `--context-provider
search-fetch` to build context from the configured search/fetch providers.
Provider credentials stay in the WebGateway process and are never passed to the
runner:

```bash
export BRAVE_SEARCH_API_KEY="..."

native-agent web-gateway serve \
  --host 127.0.0.1 \
  --port 8090 \
  --provider brave-http \
  --context-provider brave-llm \
  --brave-api-key-env BRAVE_SEARCH_API_KEY
```

Create a run:

```bash
curl -sS -X POST http://127.0.0.1:8765/v1/runs \
  -H "Authorization: Bearer $NAR_BACKEND_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "tenant_a",
    "user_id": "user_a",
    "workspace_root": "/workspaces/demo",
    "instruction": "Read notes.md and create SUMMARY.md.",
    "mode": "propose",
    "runtime_config": {
      "definition_id": "markdown-editor",
      "config_version": 1,
      "model": {"provider": "gateway", "model": "gpt-5.5"},
      "tools": [
        {"binding_id": "read_file", "ref": {"kind": "registry", "tool_id": "fs.read"}},
        {"binding_id": "write_file", "ref": {"kind": "registry", "tool_id": "fs.write"}},
        {"binding_id": "finish", "ref": {"kind": "registry", "tool_id": "run.finish"}}
      ],
      "tool_search": {"enabled": true, "top_k": 5}
    }
  }'
```

The response includes a `run_token`. Use that token for:

```bash
curl -H "Authorization: Bearer $RUN_TOKEN" \
  http://127.0.0.1:8765/v1/runs/$RUN_ID/status

curl -H "Authorization: Bearer $RUN_TOKEN" \
  http://127.0.0.1:8765/v1/runs/$RUN_ID/result

curl -H "Authorization: Bearer $RUN_TOKEN" \
  http://127.0.0.1:8765/v1/runs/$RUN_ID/events

curl -H "Authorization: Bearer $RUN_TOKEN" \
  http://127.0.0.1:8765/v1/runs/$RUN_ID/proposal

curl -H "Authorization: Bearer $RUN_TOKEN" \
  http://127.0.0.1:8765/v1/runs/$RUN_ID/proposal/files/SUMMARY.md

curl -H "Authorization: Bearer $RUN_TOKEN" \
  http://127.0.0.1:8765/v1/runs/$RUN_ID/runtime-config

# POST replaces the run's config (optimistic concurrency via expected_version); the runner
# applies it at the next turn boundary. See docs/CONTRACTS.md for the request schema.
curl -sS -X POST http://127.0.0.1:8765/v1/runs/$RUN_ID/runtime-config \
  -H "Authorization: Bearer $RUN_TOKEN" \
  -H "Content-Type: application/json" \
  -d @new-runtime-config.json

curl -H "Authorization: Bearer $RUN_TOKEN" \
  http://127.0.0.1:8765/v1/runs/$RUN_ID/jobs

curl -H "Authorization: Bearer $RUN_TOKEN" \
  http://127.0.0.1:8765/v1/runs/$RUN_ID/jobs/$JOB_ID/logs?stream=stdout
```

Tenant usage is admin-scoped:

```bash
curl -H "Authorization: Bearer $NAR_BACKEND_ADMIN_TOKEN" \
  http://127.0.0.1:8765/v1/tenants/tenant_a/usage
```

The backend generates a separate `llm_gateway` token for the runner-to-gateway
call. That token is passed only to `GatewayModelAdapter` and is not returned from
the run APIs. For web-enabled runs, it also generates a separate `web_gateway`
token for `WebGatewayClient`.

The LLM gateway validates `llm_gateway` tokens, calls the provider adapter, and returns only
opaque `turn_handle` values to the runner. The default by-value `messages` request is
forwarded statelessly; for handle-based continuation it stores provider continuation ids
server-side. The turn request carries the effective model from runtime config. Its usage endpoint is
admin-scoped:

```bash
curl -H "Authorization: Bearer $NAR_LLM_GATEWAY_ADMIN_TOKEN" \
  http://127.0.0.1:8080/internal/llm/tenants/tenant_a/usage
```

The WebGateway validates `web_gateway` tokens, enforces per-request binding
constraints, calls a web provider adapter, and reports tenant usage. The reference ships a
deterministic fake provider plus Brave-backed search/fetch/context providers behind the
provider-neutral `ContextProvider` seam, so the search backend can be swapped without
changing runner tools.

```bash
curl -H "Authorization: Bearer $NAR_WEB_GATEWAY_ADMIN_TOKEN" \
  http://127.0.0.1:8090/internal/web/tenants/tenant_a/usage
```

## Outputs

Each run writes:

- `events.jsonl`: public redacted event stream
- `transcript.jsonl`: private debug/replay transcript with full tool payloads
- `status.json`: latest run status for polling
- `metrics.json`: final counters and timing
- `manifest.json`: run contract, agent config metadata, binding-aware tool surface, workspace backend
- `workspace.base.json`: base snapshot used for proposal comparison
- `workspace.index.json`: context/index artifact
- `diff.patch`: proposed or applied workspace diff
- `proposal.json`: proposed output snapshot metadata
- `proposal/files/`: materialized changed-file snapshots
- `artifacts/jobs/<job_id>/`: background job status (`job.json`) and `stdout.log` / `stderr.log`

`events.jsonl` remains public/redacted. Proposed file contents are exposed only
through the run directory snapshot or run-token protected backend proposal APIs.

## Event Sinks

Programmatic callers can pass sinks to
`AgentLoop(..., runtime_config_provider=provider, event_sinks=(...))`.
CLI callers can load sinks with:

```bash
native-agent run \
  --workspace . \
  --instruction "Inspect this workspace." \
  --runtime-config-file examples/runtime-config.json \
  --event-sink-module ./my_sink.py:make_sink
```

The function must return an object with `emit(event)` and `close()` methods, or
an iterable of those objects.

`examples/redacting_event_sink.py` is a ready-to-copy sink that masks
secret-looking values before forwarding — the recommended place to add secret
redaction now that the core no longer guesses at secrets (see above):

```bash
native-agent run \
  --workspace . \
  --instruction "Inspect this workspace." \
  --runtime-config-file examples/runtime-config.json \
  --llm-gateway-url http://127.0.0.1:8080/internal/llm/turns \
  --event-sink-module examples/redacting_event_sink.py:make_sink
```

## Model Provider Boundary

`GatewayModelAdapter` is the default path. It sends normalized model-turn
requests to a CSP-owned LLM gateway and can authenticate with
`NAR_LLM_GATEWAY_TOKEN` or `--llm-gateway-token-file`. Provider credentials stay
inside CSP backend infrastructure, where tenant usage, budgets, and rate limits
can be enforced.

`OpenAIModelAdapter` is retained for local smoke tests. CLI use requires
`runtime_config.model.provider="openai"` and `--allow-direct-provider-api`.

To target your own LLM gateway, implement the `ModelAdapter` protocol or the
`native-agent-runner.llm-turn.v1` HTTP contract documented in
[docs/CONTRACTS.md](docs/CONTRACTS.md).

## Defaults

- runtime config is required for CLI and backend runs
- default model provider inside `ModelConfig`: `gateway`
- default model inside `ModelConfig`: `gpt-5.5`
- default reasoning effort inside `ModelConfig`: `medium`
- mode: `propose`
- shell is available only through an exposed `shell.exec` binding
- web.search/web.fetch/web.context are available only through exposed web bindings and WebGateway
- file mutation tools include write, patch, mkdir, copy, move, and delete in
  `propose` and `apply` modes when bound in runtime config
- no path deny/redact policy unless explicitly provided

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for
development setup and the lint/test workflow, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
For security issues, follow [SECURITY.md](SECURITY.md) (do not open a public issue).

## License

Licensed under the [Apache License 2.0](LICENSE).
