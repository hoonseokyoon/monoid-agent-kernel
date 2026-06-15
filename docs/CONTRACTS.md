# Integration Contracts

This document defines the stable surface other systems integrate against. It has two parts:

1. **Python contracts** — the types and protocols you import from `native_agent_runner.contracts`
   to embed the runner or extend it (tools, model adapters, event sinks).
2. **HTTP wire contracts** — the request/response shapes the core runner exchanges with an LLM
   gateway and a web gateway, so you can build your own gateways.

## Boundary rule

- **Core** = `native_agent_runner.contracts` plus the engine it re-exports (`loop`, `core/*`,
  `providers/*`, `tools/*`, `workspace/*`, `permissions`, `shell`, `web`).
- **Reference** = `native_agent_runner.reference.{backend,llm_gateway,web_gateway}`. These are
  **examples**. The core never imports them; you are expected to build your own.
- Invariant: nothing under the core imports `native_agent_runner.reference`. Keep it that way.

---

## 1. Python contracts

Import everything below from `native_agent_runner.contracts` (single stable surface).

### 1.1 Engine entry / result

- `AgentLoop` (`loop.py`) — the engine. Constructor:
  `AgentLoop(spec, model_adapter, tool_providers=(), event_sinks=(), status_file=True,
  permission_policy=PermissionPolicy(), cancellation_token=None, shell_approval_provider=None,
  web_gateway_client=None)`. Call `.run() -> AgentRunResult`.
- `AgentRunSpec` (`core/spec.py`) — the run definition: `instruction`, `workspace_root`, `run_root`,
  `run_id`, `mode` (`read-only`|`propose`|`apply`), `workspace_backend` (`overlay`|`staging`),
  `model: ModelConfig`, `limits: RunLimits`, `capabilities`, and the policy objects.
- `ModelConfig`, `ReasoningConfig`, `RunLimits`, `ModelRetryConfig` (`core/spec.py`) — model + run knobs.
- `AgentRunResult`, `AgentArtifact` (`core/result.py`) — `status` (`completed`|`failed`|`limited`),
  `final_text`, `run_dir`, `diff_path`, `proposal_path`, `artifacts`, `metrics`, `error`, `error_code`.

**Spec serialization.** `AgentRunSpec.to_json()` / `AgentRunSpec.from_json(dict)` round-trip the entire run
definition (model, limits, capabilities, and all four policies) as one JSON object — this is the portable spec
contract. The CLI consumes it via `native-agent run --spec spec.json` (an alternative to the individual flags;
transport flags such as gateway URLs and tokens still apply). Shape:

```json
{
  "instruction": "...", "workspace_root": "/ws", "run_root": "runs", "run_id": "...",
  "mode": "propose", "workspace_backend": "overlay",
  "model": {"provider": "gateway", "model": "gpt-5.5", "gateway_url": "...",
            "reasoning": {"effort": "medium", "summary": "off"}, "retry": {...}},
  "limits": {"max_steps": 30, "max_tool_calls": 100, "max_bytes_read": 1000000, "max_duration_s": 900},
  "capabilities": null,
  "permission_policy": {...}, "tool_policy": {...}, "shell_policy": {...}, "web_policy": {...},
  "metadata": {}
}
```
Only `instruction` and `workspace_root` are required; everything else falls back to defaults. See
`examples/run-spec.json`.

### 1.2 Model adapter contract

Implement this to plug in any model/transport. `providers/base.py`:

```python
class ModelAdapter(Protocol):
    def next_turn(self, request: ModelRequest) -> ModelTurn: ...
```

- `ModelRequest`: `instruction`, `system_prompt`, `tools: tuple[ToolSpec, ...]`,
  `previous_turn_handle: str | None`, `observations: tuple[ToolObservation, ...]`.
- `ModelTurn`: `response_id`, `final_text`, `tool_calls: tuple[ToolCall, ...]`,
  `usage: dict[str, int]`, `raw`.
- `ToolCall`: `id`, `name`, `arguments: dict`.
- `ToolObservation`: `call_id`, `tool_name`, `output: dict`, `is_background: bool` (set when the
  observation carries a completed background-job result; adapters branch on this flag rather than
  sniffing the tool name / call-id).
- **Turn threading**: first turn sends `instruction` with `previous_turn_handle=None`. Subsequent
  turns send `previous_turn_handle` (the prior `ModelTurn.response_id`) plus `observations` (tool
  results); `instruction` is omitted. The adapter is responsible for conversation continuity.
  The provider-neutral `previous_turn_handle` is a rename of the former `previous_response_id`
  (event/transcript `data` keys renamed in place under `native-agent-runner.event.v1`; no version
  bump). `OpenAIModelAdapter` maps it to OpenAI's own `previous_response_id` wire field;
  `GatewayModelAdapter` sends it as `previous_turn_handle`.
- **Errors**: raise `ModelAdapterError` (`errors.py`) with `provider_error_code`, `retryable`,
  `http_status`. `ModelRetryConfig.retry_on` (a tuple of error codes) gates which codes the core
  retries; only `retryable=True` codes present in `retry_on` are retried with backoff.

`GatewayModelAdapter` (`providers/gateway.py`) is the default HTTP implementation;
`OpenAIModelAdapter` and `FakeModelAdapter` also implement the protocol.

### 1.3 Tool contract

Add domain-specific tools by implementing a provider. `tools/base.py`:

```python
class ToolProvider(Protocol):
    def get_tools(self, context: ToolContext) -> Iterable[ToolSpec]: ...
```

- `ToolSpec`: `id`, `description`, `input_schema` (JSON Schema, validated with Draft 2020-12),
  `capability` (gating string), `side_effect` (`read`|`write`|`artifact`|`run`|`shell`), `handler`,
  `provider_name`, `path_args`. `exported_name` is `provider_name` or `id` with dots→underscores.
  Declarative engine hints (the loop branches on these, never on tool ids): `preview_kind`
  (`args`|`shell`|`web`), `emits_workspace_diff` (emit `workspace.file.changed` + proposal after a
  successful call), `changed_paths_source` (`path_args`|`result_content`), `result_payload_kind`
  (`paths`|`shell_exec`), `skip_emit_if_background`.
- `ToolHandler = Callable[[ToolContext, dict], ToolResult]`.
- `ToolResult`: `ok`, `content: dict`, `error`, `error_code`; `to_observation()` is what the model sees.
- `ToolContext` is the protocol the handler receives (artifact/plan/finish/shell/job/web operations).
- Register via `AgentLoop(..., tool_providers=(MyProvider(),))`, or from the CLI with
  `--tool-module path.py:get_tools`. Builtin tools are always present; custom tools are additive.
- **Capability gating**: a tool is visible only if its `capability` is in the run's effective
  capabilities. Defaults derive from `mode` (`default_capabilities` in `core/spec.py`); enabling
  `shell_policy`/`web_policy` adds `shell.exec`/`job.control`/`web.*`. `ToolPolicy`
  (allow/deny/ask) further filters visibility. See `examples/custom_tools/word_count_tool.py`.
  A stale call to a tool whose capability is not granted is rejected with `error_code`
  `capability_disabled` (this replaced the tool-specific `shell_disabled`/`web_disabled` codes on
  the dispatch path; those codes still surface from the runtime shell/web gateway when an enabled
  capability's service is unconfigured).

### 1.4 Event contract

Observe a run by implementing `EventSink` (`core/events.py`):

```python
class EventSink(Protocol):
    def emit(self, event: AgentEvent) -> None: ...
    def close(self) -> None: ...
```

- `AgentEvent` carries `schema_version` (`EVENT_SCHEMA_VERSION = "native-agent-runner.event.v1"`),
  `event_id`, `seq`, `run_id`, `timestamp`, plus type/level/payload. `AgentEventType` enumerates all
  event kinds (run/model/tool/shell/job/web/workspace/proposal/...); `AgentEventLevel` is
  `debug`|`info`|`warning`|`error`.
- **Per-type `data` contract**: the envelope is validated by `EVENT_SCHEMA`; each event type's
  `data` payload is pinned by `EVENT_DATA_SCHEMAS` (`core/schemas.py`), keyed by `AgentEventType`.
  `validate_run_dir` validates every event's `data` against its type schema (and flags any event
  type with no schema). Stable events use `additionalProperties: false`; events whose payload is
  assembled from `to_public_json()`/snapshots (shell/web/approval/job/proposal-lifecycle/workspace
  snapshots) are `additionalProperties: true` and will be tightened over time. Consumers
  (`StatusJsonSink`, `core.projections`) read this contracted shape.
- Pass sinks via `AgentLoop(..., event_sinks=(...))`, or the CLI `--event-sink-module path.py:make_sink`.
- Built-in sinks: `JsonlEventSink`, `MemoryEventSink`, `StatusJsonSink`, `StdoutJsonlSink`.
- **Secret handling**: public events are *not* heuristically scrubbed for secrets. The core only keeps
  file-content fields out of the public stream (full content lives in the private `transcript.jsonl` /
  `proposal`) and masks paths matching `PermissionPolicy.redact_patterns`. Any other redaction —
  secret-bearing tool arguments, tokens embedded in shell commands, etc. — is the integrator's
  responsibility; `EventSink` is the seam to add it (wrap or post-process events before they leave
  the trust boundary). See `examples/redacting_event_sink.py` for a ready-to-copy wrapping sink.

### 1.5 Policy contracts

- `PermissionPolicy` (`permissions.py`) — `deny_patterns` (block tool/shell path access),
  `redact_patterns` (mask in public events/status only; private artifacts keep real values).
- `ShellPolicy` (`shell.py`) — enable/disable shell, approval mode, timeouts, env allowlists.
- `WebPolicy` (`web.py`) — enable web and per-capability (search/fetch/context) flags, call limits,
  domain filters.
- `ToolPolicy` (`tools/policy.py`) — allow/deny/ask rules over tool visibility.

---

## 2. HTTP wire contracts

The core sends an opaque `Authorization: Bearer <token>` on every gateway call. The token scheme is
a **reference convention** (see §3) — your gateway may validate tokens however it likes.

### 2.1 LLM gateway

Source of truth: `providers/gateway.py` (`_payload`, `_parse_gateway_response`).

**Request** — `POST <gateway-url>` (default path in the reference impl: `/internal/llm/turns`):

```json
{
  "protocol": "native-agent-runner.llm-turn.v1",
  "model": "gpt-5.5",
  "system_prompt": "...",
  "tools": [
    {"id": "fs.read", "name": "fs_read", "description": "...",
     "input_schema": { /* JSON Schema */ }, "capability": "fs.read", "side_effect": "read"}
  ],
  "reasoning": {"effort": "medium", "summary": "auto"},   // omitted when default/off
  "instruction": "...",                                    // FIRST turn only
  "previous_turn_handle": "...",                           // SUBSEQUENT turns only
  "observations": [                                        // SUBSEQUENT turns only
    {"call_id": "...", "tool_name": "fs.read", "output": { /* ToolResult.to_observation() */ },
     "is_background": false}                                // true for background-job re-entry results
  ]
}
```

**Success response** (`native-agent-runner.llm-turn-result.v1`):

```json
{
  "turn_handle": "...",          // or "response_id"; either is accepted
  "final_text": "..." | null,
  "tool_calls": [
    {"id": "...", "name": "fs_read", "arguments": { /* dict or JSON string */ }}
    // "call_id" accepted as an alias for "id"
  ],
  "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
}
```

**Error response** (any of these triggers a `ModelAdapterError`):

```json
{"error": "...", "error_code": "gateway_rate_limited", "retryable": true, "http_status": 429}
```

Retryable codes used by the core: `gateway_timeout`, `gateway_network_error`,
`gateway_rate_limited`, `gateway_server_error` (also inferred from HTTP 429 / 5xx). The gateway is
expected to hold provider credentials and store provider continuation state behind `turn_handle`.

### 2.2 Web gateway

Source of truth: request bodies in `loop.py`; response bodies in
`reference/web_gateway/service.py`; client + error envelope in `web.py`.

**Endpoints**: `POST /internal/web/search`, `/internal/web/fetch`, `/internal/web/context`.

| Call | Request protocol | Key request fields | Response protocol | Key response fields |
|------|------------------|--------------------|-------------------|---------------------|
| search | `native-agent-runner.web-search.v1` | `query`, `max_results`, `allowed_domains`, `blocked_domains`, `recency_days`, `locale` | `…web-search-result.v1` | `results[]` (`title`,`url`,`domain`,`snippet`,`source`), `result_count`, `effective_max_results` |
| fetch | `native-agent-runner.web-fetch.v1` | `url`, `format`, `timeout_s`, `max_bytes` | `…web-fetch-result.v1` | `final_url`, `domain`, `title`, `content`, `content_bytes`, `truncated`, `effective_*` |
| context | `native-agent-runner.web-context.v1` | `query`, `max_tokens`, `max_urls`, `max_snippets`, `allowed_domains`, `blocked_domains`, `recency_days`, `locale` | `…web-context-result.v1` | `context`, `sources[]`, `chunks[]`, `estimated_tokens`, `effective_*` |

**Error response** (raises `WebGatewayError`): `{"error": "...", "error_code": "...", "http_status": 0}`.

---

## 3. Token model (reference only)

The reference services sign claims with HMAC-SHA256 (`reference/_shared/tokens.py`,
`TokenManager`) and use three token kinds (`run_access`, `llm_gateway`, `web_gateway`). This is a
**convention of the reference backend**, not a core requirement: the core only emits an opaque
bearer token to its gateways. When you build your own backend you may use any auth scheme — the
only contract is the `Authorization: Bearer <token>` header on gateway calls.

---

## 4. Worked examples

- `examples/full_stack_integration.py` — runner + reference LLM gateway + reference backend wired together.
- `examples/custom_tools/word_count_tool.py` — a minimal custom `ToolProvider`.
- `examples/messy_workspace_cleanup.py` — an end-to-end propose-mode run.
