# Integration Contracts

This document defines the supported integration surface for native-agent-runner.
Import Python contracts from `native_agent_runner.contracts`. Treat
`native_agent_runner.reference.*` as runnable examples for backend, LLM gateway,
and web gateway integration.

## Boundary

- Core exports the runner, contracts, providers, tools, workspace, permission,
  shell execution, and web gateway client modules.
- Reference packages implement example services. Core code has no dependency on
  `native_agent_runner.reference`.
- Agent configuration enters the engine through `AgentRuntimeConfig`. Legacy
  tool/shell/web policy inputs have left the core, backend, and CLI execution
  paths.

## Python Contracts

### AgentLoop

`AgentLoop(spec, model_adapter, *, runtime_config_provider, tool_providers=(),
event_sinks=(), status_file=True, permission_policy=PermissionPolicy(),
cancellation_token=None, shell_approval_provider=None, web_gateway_client=None)`
runs a single agent against one workspace.

`runtime_config_provider` is required. The loop reads the current config at
bootstrap and at each turn boundary. A config change applies to the next turn.
The `ToolSurfaceSnapshot` and `BoundToolCatalog` used by a turn stay fixed for
that turn.

The run lifecycle is:

- `open()` — bootstrap and idle (workspace, recorder, tool registry, manifest;
  emits `run.started`). No model turn yet.
- `submit(user_input) -> AgentTurnResult` — run one user turn: deliver
  `user_input` (a `str` or content parts) and step until the model settles (no
  tool calls + final text) or a per-submit limit. The run stays open. Each
  `submit()` gets a fresh `max_steps` budget; `max_tool_calls`, token usage, and
  `max_duration_s` are session-wide. `AgentTurnResult` carries the settle status,
  final text, the accumulated (preview) proposal, and the continuation
  `turn_handle`.
- `commit_checkpoint()` — opt-in: adopt the current proposed workspace state as
  the new diff baseline, so later proposals report only post-commit changes.
- `close() -> AgentRunResult` — finalize: cancel jobs, write the terminal
  proposal, emit `run.finished`, close the recorder.
- `run_once(user_input) -> AgentRunResult` — one-shot convenience equal to
  `open()` + `submit(user_input)` + `close()`.

### AgentRunSpec

`AgentRunSpec` is the session descriptor. It carries no user input — the
instruction(s) flow in through `submit()` / `run_once()`:

- `workspace_root`, `run_root`, `run_id`
- `mode`: `read-only`, `propose`, or `apply`
- `workspace_backend`: `overlay` or `staging`
- `limits: RunLimits`
- `permission_policy: PermissionPolicy`
- `input`: optional multimodal content-parts surface (contract-only)
- `metadata`

It does not carry model, prompt, tool, shell, or web settings. Those values live
in runtime config.

### AgentDefinition And Runtime Config

`AgentDefinition` is a reusable blueprint:

- `id`, `version`, `description`
- `model: ModelConfig | None`
- `prompt: PromptSpec`
- `tools: tuple[ToolBinding, ...]`
- `tool_search: ToolSearchConfig`
- `metadata`

`AgentRuntimeConfig` is the effective config for a run:

- `definition_id`
- `config_version`
- `model: ModelConfig | None`
- `prompt: PromptSpec`
- `tools: tuple[ToolBinding, ...]`
- `tool_search: ToolSearchConfig`
- `metadata`
- `config_hash`

`ToolBinding` is the public tool unit:

```json
{
  "binding_id": "read_notes",
  "model_name": "read_notes",
  "ref": {"kind": "registry", "tool_id": "fs.read"},
  "exposure": "immediate",
  "authorization": "allow",
  "guidance": {"summary": "Read source files before editing."},
  "scope": {"allowed_paths": ["docs/**"]},
  "quota": {"max_calls_per_run": 20},
  "runtime": {},
  "title": "Read notes",
  "summary": "Read a workspace file.",
  "risk": "read",
  "metadata": {}
}
```

The same registry tool can appear multiple times with different `binding_id`,
`model_name`, guidance, scope, quota, and runtime settings. Duplicate
`binding_id` values and duplicate resolved `model_name` values fail validation.
Unknown registry tool refs fail validation with `AgentConfigError`.

`compile_bound_tool_catalog(config, registry)` produces a `BoundToolCatalog`.
The model receives only bound model-facing `ToolSpec`s. Tool execution resolves
`model_name -> BoundTool -> base ToolSpec.handler`.

### Tool Surface

`DefaultToolSurfaceResolver` consumes a `BoundToolCatalog`, turn context,
pending binding loads, previous snapshot, and call counts. It returns a
`ToolSurfaceSnapshot`:

- `immediate_tools`: model-facing bound specs available this turn
- `searchable_tools`: bound specs indexed for `tool.search`
- `search_entries`: binding-aware search metadata
- `hidden_tool_ids`: hidden or denied binding ids
- `authorizations`: `binding_id -> ToolAuthorization`
- `surface_hash`, `delta_notice`

Unbound registry tools stay outside the surface. Hidden or denied bindings stay
outside model tools and search results. Tool search uses `binding_id` for search
results and pending loads.

### Model Adapter

Implement `ModelAdapter.next_turn(request: ModelRequest) -> ModelTurn`.

`ModelRequest` carries:

- `instruction`
- `system_prompt`
- `tools: tuple[ToolSpec, ...]`
- `previous_turn_handle`
- `observations`
- `model: ModelConfig | None`

Adapters must use `request.model` for turn-level model selection when present.
`GatewayModelAdapter` and `OpenAIModelAdapter` follow that rule.

### Tool Contract

Add tools with `ToolProvider.get_tools(context) -> Iterable[ToolSpec]`.

`ToolSpec` still describes a registry tool: id, description, JSON schema,
side-effect class, handler, provider name, path args, preview hints, guidance,
examples, and annotations. Registry specs are implementation tools. Bindings
decide model-facing names, guidance, exposure, authorization, scope, quota, and
runtime settings.

`ToolResult.to_observation()` returns:

```json
{"ok": true, "result": {"value": "..."}}
```

Failures return:

```json
{
  "ok": false,
  "result": {},
  "error": {
    "message": "...",
    "code": "tool_handler_error",
    "category": "tool",
    "retryable": true
  }
}
```

### Shell And Web Bindings

Shell availability is the presence of an exposed `shell.exec` binding.

- command allow/deny prefixes and env allowlist live in `ToolBinding.scope`
- timeout, output, startup wait, approval mode, shell kind, and execution
  workspace live in `ToolBinding.runtime.shell`
- `ShellExecutionOptions` is an internal low-level execution options object

Web availability is the presence of exposed `web.search`, `web.fetch`, and
`web.context` bindings.

- domain allow/block lists live in `ToolBinding.scope`
- result limits, context limits, timeout, response-byte limits, and call limits
  live in `ToolBinding.runtime.web`
- gateway requests include `binding_id`, `max_calls`, and effective constraints

### Async Tasks

Long-running work whose result feeds back to the model — shell background jobs,
human-in-the-loop requests, automation — flows through one generic task system.
The core (`TaskManager`) owns the queue, lifecycle, reentry, and artifacts; three
seams are pluggable:

- `TaskExecutor` — how a task kind runs and when it is done. The shell executor
  monitors a subprocess in-process; a hosted kind (hitl/automation) has no
  monitor and is completed by an external reporter.
- `ResultInjector` — how a finished task is injected into the model: as a tool
  observation (`is_background=False`) or as a new user message
  (`is_background=True`). This is the "appropriate way, defined by the integrator".
- `TaskReporter` — how the backend drives tasks in a running run: `create_task`
  and `report_result`. Transport-agnostic — only `(task_id, dict)` cross the
  boundary, so an in-process reporter and a future durable/cross-process reporter
  share the same shape.

Both the model (via tools such as `hitl.request`) and the backend can create
tasks; a completed task wakes a parked run through the shared reentry queue.

### Permission Boundary

`PermissionPolicy` remains the workspace/public-output boundary:

- `deny_patterns` block workspace path access
- `redact_patterns` mask public events and projections

It does not grant tools. Tool availability and execution constraints come from
bindings.

## HTTP Contracts

### LLM Gateway

`GatewayModelAdapter` sends `POST <gateway-url>`.

```json
{
  "protocol": "native-agent-runner.llm-turn.v1",
  "model": "gpt-5.5",
  "system_prompt": "...",
  "tools": [
    {
      "id": "read_notes",
      "name": "read_notes",
      "description": "...",
      "input_schema": {},
      "capability": "fs.read",
      "side_effect": "read"
    }
  ],
  "reasoning": {"effort": "medium", "summary": "off"},
  "instruction": "First turn text"
}
```

The turn request has three shapes, selected by `previous_turn_handle` and
`instruction`:

- **first turn** — no `previous_turn_handle`; carries `instruction`.
- **tool continuation** — `previous_turn_handle` + `observations`; no `instruction`.
- **user follow-up** — `previous_turn_handle` + `instruction` (a new user message on
  top of an existing continuation handle; `observations` is empty).

This is what lets one run accept multiple user turns: the runner threads the last
`turn_handle` into the next user message.

Successful response:

```json
{
  "protocol": "native-agent-runner.llm-turn-result.v1",
  "turn_handle": "turn_...",
  "final_text": null,
  "tool_calls": [
    {"call_id": "call_1", "name": "read_notes", "arguments": {"path": "notes.md"}}
  ],
  "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
}
```

The reference LLM gateway token authenticates run identity. The request model
selects the turn model.

### Web Gateway

The runner calls:

- `POST /internal/web/search`
- `POST /internal/web/fetch`
- `POST /internal/web/context`

Every request includes binding constraints:

```json
{
  "protocol": "native-agent-runner.web-search.v1",
  "binding_id": "search_docs",
  "query": "native runtime config",
  "max_results": 5,
  "max_calls": 20,
  "allowed_domains": ["docs.example.test"],
  "blocked_domains": []
}
```

The reference gateway enforces per-run/binding call counters and the
per-request domain/limit constraints. Web gateway tokens authenticate run
identity.

### Reference Backend

Run creation requires `agent_definition` or `runtime_config`.

`POST /v1/runs` request shape:

```json
{
  "tenant_id": "tenant_a",
  "user_id": "user_a",
  "workspace_root": "/workspaces/demo",
  "instruction": "Read notes.md and create SUMMARY.md.",
  "mode": "propose",
  "runtime_config": {
    "definition_id": "coding-agent",
    "config_version": 1,
    "model": {"provider": "gateway", "model": "gpt-5.5"},
    "prompt": {"runtime_segments": ["Prefer concise edits."]},
    "tools": [
      {"binding_id": "read_file", "ref": {"kind": "registry", "tool_id": "fs.read"}},
      {"binding_id": "finish", "ref": {"kind": "registry", "tool_id": "run.finish"}}
    ],
    "tool_search": {"enabled": true, "top_k": 5}
  }
}
```

Runtime config API:

- `GET /v1/runs/{run_id}/runtime-config`
- `POST /v1/runs/{run_id}/runtime-config`

Replacement request:

```json
{
  "expected_version": 1,
  "issuer": "backend",
  "reason": "update guidance",
  "config": {
    "definition_id": "coding-agent",
    "config_version": 2,
    "model": {"provider": "gateway", "model": "gpt-5.5"},
    "tools": [
      {
        "binding_id": "read_file",
        "ref": {"kind": "registry", "tool_id": "fs.read"},
        "guidance": {"summary": "Read the smallest relevant file first."}
      },
      {"binding_id": "finish", "ref": {"kind": "registry", "tool_id": "run.finish"}}
    ]
  }
}
```

The backend validates schema, registry resolvability, duplicate binding ids,
and duplicate model names. A version mismatch returns HTTP 400.

### Multi-turn Sessions And Tasks

The run loop is suspend-return at its core: `AgentLoop.run_until_suspended()` runs
a turn and hands control back when the run settles (awaiting the next user
message), parks on a hosted task, or hits a limit. `submit()` is the blocking
wrapper over it; the reference backend's worker uses the non-blocking form to
drive multi-turn sessions.

Set `"multi_turn": true` on the run-creation request to keep the session open
after the first turn settles (default `false` closes after one turn). While open,
the run alternates between `running` and `awaiting_input` (a new
`run.awaiting_input` event with `reason` `"user"` or `"task"`). HTTP surface:

- `POST /v1/runs/{run_id}/messages` — deliver a follow-up user message (run token).
  It is queued and consumed as the next user turn when the current one settles.
- `POST /v1/runs/{run_id}/tasks` — create a hosted task (`{"kind": "hitl" |
  "automation", "request": {...}}`). Returns `task_id` plus a scoped
  `callback_token` and `callback_url`.
- `POST /v1/runs/{run_id}/tasks/{task_id}/result` — deliver a task result
  (`{"result": {...}, "status": "answered"}`). Authenticated by the per-task
  callback token (scoped to this run+task) or the run token (operator). Reporting
  a result wakes a parked run; the result is injected per the kind's
  `ResultInjector` (a user message for hitl, an async tool result for automation).

Follow-up user messages and task results are separate channels (a message is a new
user turn; a task result completes a specific task), mirroring the
add-message-vs-submit-tool-outputs split in comparable agent servers. Session
length is bounded by idle timeout, max lifetime, and max turns.

### Durable Persistence

A checkpoint is a **complete, self-contained "save file."** A parked run survives a
process restart even when the agent's workspace is *not* durable: workspace, the
conversation, and run state all roll back to one aligned instant (a time machine).
This is a **state-snapshot at the suspend points** (the LangGraph-checkpointer
pattern, not event-sourcing replay); snapshots are only taken at clean park points,
so there is no determinism constraint and no double-side-effect risk.

**Division of responsibility:** the core defines *what* a checkpoint contains
(`RunCheckpoint`) and how to `restore()` it; the integrator decides *how* it is
stored by implementing `CheckpointStore`. The core never does storage I/O or
auto-recovery — on failure it surfaces a bundle and the last-good checkpoint, and
recovery is the integrator's call.

- `AgentLoop.snapshot() -> RunCheckpoint | None` captures, at one quiescent park:
  run state + counters + parked hosted tasks, the **workspace delta** (created/
  modified/deleted files; content travels as content-addressed blobs), the **by-value
  conversation** (`messages` — provider-neutral user/assistant/tool log, vendor-
  independent), and the **latest `runtime_config`**. It returns `None` — refusing —
  while a live in-process shell job is still running (a subprocess cannot cross a
  process boundary).
- `CheckpointStore` (protocol): `put(checkpoint, blobs)` commits **atomically** and
  flips a `LATEST` pointer last (a half-written checkpoint is never returned);
  `latest(run_id)`; `delete(run_id)`. `LocalFsCheckpointStore` is the default
  (`run_root/<id>/checkpoints/<seq>/manifest.json` + content-addressed `blobs/<sha>`);
  swap it for a mounted-volume path or an object-store/DB store. The loop advances a
  monotonic `seq` per park and deletes checkpoints only on a *completed* run — a
  failed/limited run keeps its last-good checkpoint.
- `AgentLoop.restore(checkpoint, *, blobs=...)` reopens the run: no second
  `run.started`/manifest, parked hosted tasks re-registered (so `report_task_result`
  still wakes it), the **workspace delta re-applied** on top of a re-provisioned base
  (`blobs` is a `sha256 -> bytes` reader, e.g. the store's), the conversation and
  `runtime_config` restored, remaining duration carried forward (downtime does not
  count against `max_duration_s`), and any shell job left `running` on disk folded in
  as a failed observation.
- **Failure bundle:** on failure the core writes `run_dir/failure.json`
  (`{error, error_code, type, last_good_seq, restore_hint}`) — fail loud, name the
  checkpoint to restore from. No auto-recovery.
- The reference backend writes `run_dir/run.json` (recovery descriptor: identity,
  workspace, limits, policy, resolved runtime config) and exposes `recover_runs()`,
  which scans `run_root`, skips terminal checkpoints and failed runs, rebuilds each
  run (re-issuing gateway tokens from the signing key, **re-provisioning the base
  workspace** is the deployment's job), `restore()`s the loop with the store's blobs,
  re-enqueues durably-saved follow-up messages, and resumes.

**Assumption (workspace):** the agent workspace is not durable; on restore the
deployment re-provisions the base (re-clone/re-mount) and the checkpoint re-applies
only the agent's delta (the delta always contains the agent's created/modified
files). For container durability, `run_root` (or the `CheckpointStore`) must point at
durable storage — a mounted volume needs no code change.

**Limitations (v2):** a mid-run `commit_checkpoint` re-baseline combined with delta-
restore is a documented follow-up (the common no-re-baseline case is covered).
Multimodal message parts are text-only for now. `transcript.jsonl` is a debug
artifact (the by-value `messages` in the checkpoint are the load-bearing
conversation record).

## Run Artifacts

Manifest and transcript are binding-aware:

- `manifest.json.agent_config`: definition id, config version, config hash
- `manifest.json.tool_surface`: resolver, tool search settings, bound catalog count
- `tool_surface_snapshot`: immediate/searchable bound tool specs and binding
  authorizations
- `agent_runtime_config_snapshot`: definition id, config version/hash, binding ids
- `agent.config.updated`: emitted when the loop observes a new config hash

Replay uses recorded snapshots. Current registry state does not reinterpret an
old turn.
