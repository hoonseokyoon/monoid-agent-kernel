# Integration Contracts

This document defines the supported integration surface for Monoid Agent Kernel:
a lightweight agent kernel designed to embed into many products, runtimes, and
deployment models. Import Python contracts from `monoid_agent_kernel.contracts`.
Treat `monoid_agent_kernel.reference.*` as runnable examples for backend, LLM
gateway, and web gateway integration.

## Boundary

The package is layered in three tiers:

- **contracts** — the stable integration surface, collected in
  `monoid_agent_kernel.contracts` (and re-exported from the top-level
  `monoid_agent_kernel`). These are the specs and protocols you depend on and
  implement. This document defines them.
- **core** — the engine that implements those contracts: the supported kernel runtime
  (`loop.py`, `core/`, `providers/`, `tools/`, `workspace/`, permission,
  shell execution, and web gateway client modules).
- **reference** — example services under `monoid_agent_kernel.reference`
  (`backend`, `llm_gateway`, `web_gateway`, `stores`). These examples live outside
  the supported public surface; core has no dependency on `monoid_agent_kernel.reference`.

Agent configuration enters the engine through `AgentRuntimeConfig`. Legacy
tool/shell/web policy inputs have left the core, backend, and CLI execution
paths.

### Stability

Pre-1.0 (`0.x`); breaking changes are noted in commit messages.

- **Stable**: `AgentLoop`, `AgentRunSpec`, `AgentRuntimeConfig` /
  `RuntimeConfigProvider`, `ModelAdapter`, `ToolSpec` / `tool`, `EventSink`,
  `CheckpointStore`, `Workspace` / `workspace_factory`, `PermissionPolicy`, and the
  rest of `contracts`.
- **Experimental**: async-task seams (`TaskExecutor`, `ResultInjector`,
  `TaskReporter`); the session lifecycle + control surface (`AgentSession` /
  `LoopSession`, `SessionState`, `ControlCommand` / `ControlResult` /
  `ControlDispatcher`); multimodal input — `ImagePart` and `DocumentPart` are
  forwarded to multimodal-capable adapters (the gateway and OpenAI adapters), a
  text-only adapter drops them with a `model.input.degraded` warning, and
  `AudioPart` / `VideoPart` round-trip as a forward-compatible contract but are
  not yet forwarded. Output validation — `OutputValidator` / `ValidationOutcome` /
  `FinalOutputView` / `OutputRetry` / `OutputValidatorError` (post-response conformance:
  a validator registered via `AgentLoop(output_validators=...)` runs by default and can be
  disabled per run with an `OutputValidatorBinding(enabled=False)`; on failure the loop re-prompts
  up to `RunLimits.max_output_retries`).
- **Reference examples**: `monoid_agent_kernel.reference.*` services.

## Identifier Namespace

Current wire and artifact identifiers use the `monoid.*` namespace. The runtime emits new
schema versions, protocol ids, token issuers, and service audiences with `monoid.*` values,
including `monoid.backend` and `monoid.task-callback`.

Readers, validators, and gateway parsers accept the pre-rename `native-agent-runner.*`
identifiers during migration so existing durable run artifacts and gateway clients continue
to load.

## Python Contracts

### AgentLoop

`AgentLoop(spec, model_adapter, *, runtime_config_provider, tool_providers=(),
context_providers=(), event_sinks=(), status_file=True,
permission_policy=PermissionPolicy(), cancellation_token=None,
shell_approval_provider=None, web_gateway_client=None, workspace_factory=None,
checkpoint_store=None, capability_broker=None, subagent_definitions={})` runs a single
agent against one workspace.

The optional seams let an integrator back the engine with their own implementations
without changing it: `workspace_factory` (file storage — see [Workspace](#workspace)),
`checkpoint_store` (durable run state — see [Durable Persistence](#durable-persistence)),
`capability_broker` (sensitive-tool gating — see
[Capability Request / Lease](#capability-request--lease)), and `subagent_definitions`
(agent-as-tool delegation — see [Subagents](#subagents-agent-as-tool)). Each defaults to
the local / in-process behavior. `tool_providers` and `context_providers` are the
extension seams that Skills and MCP ride (see [Skills](#skills-progressive-disclosure)).

`runtime_config_provider` is required, but accepts any of three forms — a
`RuntimeConfigProvider`, a bare `AgentRuntimeConfig`, or a
`callable(run_id) -> AgentRuntimeConfig | None` — which the loop coerces to a
provider. `AgentLoop.from_config(spec, model_adapter, runtime_config, **kwargs)`
wraps a fixed config and forwards the remaining optional seams in one call.
`StaticRuntimeConfigProvider` / `static_runtime_config(config)` are the explicit
fixed-config provider. The loop reads the current config at
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
- `input`: optional multimodal content-parts surface (image/document parts are
  forwarded to multimodal-capable adapters)
- `metadata`

It does not carry model, prompt, tool, shell, or web settings. Those values live
in runtime config.

### Workspace

The engine never touches the filesystem directly — it works through a `Workspace`
(the file-storage surface), which stores and diffs the run's files. `AgentLoop` builds
one per run by calling `workspace_factory(spec)`. The default,
`default_local_workspace_factory`, returns the local-filesystem `LocalWorkspaceBackend`.
Supply your own `workspace_factory` to back a run with a different store — a git worktree,
an object store, a remote or in-memory filesystem — without changing the engine.

A `Workspace` exposes:

- path handling — `normalize`, `path_kind`, `exists`, `resolve_existing_or_parent`
- byte IO — `read_bytes` (honors a `max_bytes` cap), `write_bytes` (optimistic
  `expected_sha256` guard), `mkdir`, `copy_path`, `move_path`, `delete_path`
- listing — `list_entries`, `glob`, `text_files`
- proposal generation — `changed_entries`, `diff_patch`,
  `snapshot_current_as_new_baseline` (re-baseline for incremental apply),
  `workspace_base_payload`

It carries `root`, `mode`, `backend_kind`, and `max_bytes_read`. The value types it
returns, `FileEntry` and `ChangedEntry`, are exported from `contracts`.

`mode` (`read-only` / `propose` / `apply`) and `backend_kind` (`overlay` / `staging`)
select how the local backend stages writes; a custom backend interprets them or pins its
own. Every backend must pass the parametrized contract suite
(`tests/test_workspace_contract.py`): write/read round-trips with their sha256, the
proposed state is observable, the optimistic and byte-cap guards hold, no path escapes the
root, the changed-entry delta tracks edits, and re-baselining collapses it. Passing it
makes a backend a drop-in.

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
  (`is_background=True`).
- `TaskReporter` — how the backend drives tasks in a running run: `create_task`
  and `report_result`. Transport-agnostic — only `(task_id, dict)` cross the
  boundary, so an in-process reporter and a future durable/cross-process reporter
  share the same shape.

Both the model (via tools such as `hitl.request`) and the backend can create
tasks; a completed task wakes a parked run through the shared reentry queue.

### Subagents (agent-as-tool)

A run can delegate a focused task to an isolated child run. This reuses the Async
Task seams above via a `subagent` task kind (`SubagentTaskExecutor`); see
`docs/SUBAGENT_DESIGN.md` for the full design.

- **Enable**: pass `AgentLoop(subagent_definitions={<id>: SubagentDefinition})`.
  When non-empty, the bootstrap registers the `agent.spawn` tool. The runtime config
  still needs an explicit binding to `agent.spawn` (e.g. `model_name: "agent_spawn"`)
  for the tool to reach the model.
- **Definition** (`SubagentDefinition`, Claude-style — everything inherits the parent
  by default): `description` (surfaced to the model for selection), `prompt`,
  `model` (None → inherit), `tools` (None → inherit ALL parent tools; a tuple is an
  allowlist), `disallowed_tools` (denylist, applied after the allowlist — deny wins),
  `mode`/`limits` (None → inherit), `tool_search` (None → inherit). `tools`/
  `disallowed_tools` entries are fnmatch patterns matched against each parent binding's
  tool id / binding id / model name (so `fs.read`, `mcp.*`, `mcp.github.*`, `*` all
  work). The allowlist is resolved **against the parent's bindings**, so a subagent can
  never exceed the parent (hard ceiling); the parent's MCP/custom tool providers are
  inherited by the child so inherited bindings resolve.
- **Tool**: `agent.spawn(subagent_type, prompt, background=false)`. `subagent_type` is
  constrained to the configured ids. The child runs in an isolated overlay workspace and
  sees only `prompt` (not the parent's conversation). Foreground (`background=false`)
  blocks and returns the child's final message as the tool result; background returns a
  `{spawned, background, task_id}` ack and the child's final message is delivered later
  as a user message.
- **Bounds** (`RunLimits`): `max_subagents` (fan-out per run, default 8) and
  `max_subagent_depth` (nesting, default 5). Enforced in the executor; a child at the
  depth cap has the `agent.spawn` binding stripped (the tool is absent, not just an
  error at call time).
- **Result shape** (`subagent_result`): `{status, final_text, message, child_run_id,
  subagent_type, usage, error}`.
- **Events**: the parent stream carries `subagent.started` (`parent_id` = the spawn
  tool-call event) and `subagent.finished`/`subagent.failed` (`parent_id` = the
  `subagent.started` event), the latter carrying the child's `usage`. The child's
  full event stream goes to its own run dir; external `event_sinks` are not shared
  with children (stateful sinks like OTel/StatusJson are per-run).
- **Usage reporting**: the parent's run metrics carry `subagent_count` and
  `subagent_usage` (the children's combined token totals). These are kept SEPARATE
  from the parent's own `total_usage` on purpose — `total_usage` also reflects the
  parent's remaining context budget, which a child's isolated tokens must not inflate.
- **Context fork** (`SubagentDefinition.context = "fork"`): instead of a fresh
  isolated context, the child inherits a snapshot of the parent's conversation AND the
  parent's prompt / tools / model (the definition's own prompt/tools/model are ignored)
  — "continue as me in an isolated branch". `"fresh"` (default) is the normal isolated
  subagent that sees only the task prompt.
- **Directory discovery**: `load_subagent_definitions(dir)` (CLI `--agents-directory`)
  scans `*.md` files with YAML frontmatter (`.claude/agents` style) into
  `SubagentDefinition`s. Frontmatter fields: `name` (id; falls back to filename),
  `description`, `tools` (omitted → inherit all), `disallowedTools`, `model` (string
  shorthand or `inherit`), `mode`, `context`; the markdown body is the system prompt
  (fresh subagents only). Parsed by `parse_frontmatter` — a zero-dependency YAML subset
  (scalars, inline/block lists, quotes), shared with Skills' `SKILL.md`.

### Skills (progressive disclosure)

A run can be equipped with **Agent Skills** — procedural how-to knowledge (Anthropic's
`SKILL.md` model) delivered to the model by *progressive disclosure*, so a large library
costs almost nothing until a skill is actually used. Skills are a knowledge layer,
complementary to subagents (execution) and MCP (integration). The whole feature attaches
through the existing `ContextProvider` + `ToolProvider` seams with **no core-loop change**
(`SkillProvider` implements both); see `docs/SKILLS_DESIGN.md`.

- **Enable**: build a `SkillProvider(definitions)` and register the one instance in both
  `AgentLoop(context_providers=(provider,), tool_providers=(provider,))`. Provider tools
  require explicit bindings; merge `provider.tool_bindings()` into the runtime config so
  the `skill` tools reach the model (mirrors the MCP provider). The CLI
  `--skills-directory` does all of this.
- **Definition** (`SkillDefinition`): `name`, `description` (both advertised at L1),
  `instructions` (the SKILL.md body, delivered at L2), `allowed_tools` (**advisory** for
  inline skills — a hint, not enforced; **enforced** for fork skills, see below), `context`
  (`"inline"` default | `"fork"`), `directory` (bundle root for L3), `metadata`.
- **Fork skills** (`context: fork`): instead of loading instructions inline, the skill runs
  as an isolated **subagent** (reusing the subagent machine) and only its final message
  returns — heavy skills keep their working noise out of the main context. The model calls
  `skill(name, task)` with `task` describing the goal; the subagent's persona is the skill's
  instructions and `task` is its first user message. A **non-empty** `allowed_tools` becomes the
  subagent's tool **allowlist** — resolved against the parent's bindings, so it is a hard
  ceiling (here `allowed-tools` is genuinely *enforced*, unlike inline skills); an empty
  `allowed_tools` inherits all of the parent's tools (no narrowing). Enable by
  merging `SkillProvider.subagent_definitions()` (namespaced `skill:<name>` ids) into
  `AgentLoop(subagent_definitions=...)`; the CLI does this automatically. The delegated run
  is reported in the usual `subagent_count`/`subagent.*` events and metrics.
- **Three levels of disclosure**:
  - **L1 — catalog** (~100 tokens/skill, emitted per-turn while the skill tool is bound):
    `SkillProvider.dynamic_segment(turn)` lists each `name: description` in the system prompt
    plus how to load one. It is config-gated — the catalog vanishes when the skill tool is
    unbound, so `static_segment()` returns `None` and the catalog rides the per-turn segment.
  - **L2 — instructions** (on trigger): the model calls the `skill(name)` tool; the result
    carries `{name, instructions, allowed_tools?, resources?}`. Model-native triggering —
    the model picks a skill by its description, no router.
  - **L3 — resources** (on demand): the model calls `skill.read_file(name, path)` to read a
    bundled file (`path` relative to the skill directory, as listed in `resources`), or
    `skill.run_script(name, path, args?)` to **execute** a bundled script and get back only
    its `{exit_code, stdout, stderr, ...}` — the script source never enters context. The
    interpreter is chosen by extension (`.py` → the kernel's Python, `.sh` → bash, `.js` →
    node, `.rb` → ruby, `.ps1` → powershell); `args` are passed to the script **verbatim as
    argv, never through a shell**, so they cannot be re-parsed/injected. The script runs in
    the workspace through the same machinery as `shell.exec` (`side_effect: "shell"`):
    approval, env scrubbing, timeout, and output limits all apply, and it is blocked in
    read-only mode. Path traversal outside the skill directory is rejected
    (`skill_path_invalid`); `SKILL.md` itself is never a readable/runnable resource (it is
    the L2 payload). **Security**: a skill script is arbitrary code — skills are
    operator-provisioned (`--skills-directory`), the same trust boundary as `--tool-module`;
    there is no extra sandbox beyond the shell machinery's defenses, so only load skills from
    trusted sources.
- **Observability**: activating a skill (L2) emits a `skill.activated` event whose
  `parent_id` is the `skill` tool call (so it is correlated to, and an OTel sink enriches,
  that tool's `execute_tool` span with `skill.name` / `skill.resource_count`); data is
  `{name, resource_count}`. The run metrics carry `skill_activation_count` and
  `skills_activated` (the list of activated skill names) — report-only, like the subagent
  roll-up. `allowed_tools` is echoed in the `skill` tool result as an advisory hint.
- **Directory discovery**: `load_skill_definitions(dir)` (CLI `--skills-directory`) scans
  recursively for `SKILL.md` files (the `<skills>/<skill-name>/SKILL.md` convention); the
  skill name is the frontmatter `name` (falling back to the directory name) and the
  SKILL.md's parent directory is the bundle root. Frontmatter fields: `name`, `description`,
  `allowed-tools` (space-separated per the spec, or an inline list), `metadata`. Parsed by
  the same zero-dependency `parse_frontmatter` used for subagents.

### Session Lifecycle (`AgentSession` + FSM)

`AgentLoop` is the engine; `AgentSession` is the embedder contract a control plane depends
on (so an Agent Daemon/Cell never imports the loop). `LoopSession` is the reference facade
that wraps an `AgentLoop`, owns the FSM, and delegates execution:

- `SessionState` — the formal lifecycle FSM (a `str`-enum): `created`, `idle`, `running`,
  `awaiting_input`, `awaiting_tasks`, `paused`, `interrupted`, `turn_failed`, `limited`,
  `cancelled`, `completed`, `failed`. `cancelled`/`completed`/`failed` are terminal.
- `state_from_suspension(suspension)` projects a pump `Suspension` onto a state (the seam that
  keeps the FSM in sync with the engine without the engine knowing about it). `LEGAL_TRANSITIONS`
  + `can_transition` / `assert_transition` define the legal edges. `to_session_state(status,
  error_code=...)` reconciles the legacy status strings (`BackendRunState` / `status.json` /
  `project_run_status`) onto the one enum.
- `LoopSession.open() / submit() / run_until_suspended() / close()` delegate to the loop and
  re-derive `state` at each boundary. `inspect() -> SessionInspection` and `health() ->
  SessionHealth` are recomputed from live loop state on every call (never stale).
- `pause()` / `resume()` / `cancel(reason)`: pause freezes the turn at the *next start-of-step*
  boundary (its in-flight `pending_observations` are kept), suspends with `reason="paused"`, and
  persists a checkpoint — so resume (a `run_until_suspended(None)` re-pump) continues the same
  turn, in-process or after a restart. Pause lands only at a step boundary (an in-flight model
  call completes first; only an interrupt aborts mid-generation under token streaming). Entering
  `paused` emits a `session.state.changed` event.

### Control Protocol

`monoid.control-command.v1` is a transport-independent envelope + a single
`dispatch` seam, so a Daemon drives a session through one entry point instead of a route per op:

- `ControlCommand(type, run_id, args, issuer, reason, command_id)` and `ControlResult(run_id,
  type, status, state, data, error, error_code)` are plain data (`status` ∈ `ok` / `not_implemented` /
  `unsupported` / `error`). `ControlDispatcher.dispatch(command) -> ControlResult` is the contract;
  `RunnerBackend.dispatch` is the reference impl, routing each command to the in-process method it
  already exposes.
- Command types: `pause`, `resume`, `cancel`, `interrupt`, `inspect`, `health`, `send_message`,
  `runtime_config`, `replace_runtime_config`, `create_task`, `report_task_result`, `status`,
  `revoke_capability`. An unknown type returns `unsupported` (the wire vocabulary stays
  forward-compatible).
- HTTP: `POST /v1/runs/{run_id}/control` with `{"type": ..., "args": {...}, "issuer": ...,
  "reason": ...}`; the bearer token authorizes the run (the route injects it into `args` so the
  envelope stays credential-free). `resume` on a *live* paused run wakes it; on a run not in
  memory (parked after a restart) it falls back to checkpoint recovery (`resume_run`).
- Audit: `RunnerBackend.dispatch` appends `control.command.received` and then either
  `control.command.completed` or `control.command.failed` to the run event log. Events include
  `command_id`, command type, target run, `issuer` as actor, reason, result status/error, duration,
  and a safe `token_sha256` reference on receipt — never the bearer token itself. A control
  `send_message` uses the command id as its inbox idempotency key.

### Event Reads

`GET /v1/runs/{run_id}/events?from_seq=N&limit=M` returns `{run_id, events, next_seq, has_more}`.
`from_seq` remains inclusive for backward compatibility. When `limit` is present, callers resume
with `from_seq=next_seq` to avoid duplicates; omitting `limit` preserves the historical "return all
events from N" behavior. `RunnerBackend.descendant_events(...)` uses the same pagination contract
for subagent event streams authorized through an ancestor run token.

### Diagnostics

`GET /v1/runs/{run_id}/diagnostics?event_limit=N` returns one token-scoped operational aggregate:
`status`, `failure` (`failure.json` when present), `recovery` attempt state, bounded recent event
summaries, control-command audit summaries, and trace ids found in recent events. Diagnostics uses
event summaries rather than raw event payloads so model text, tool arguments, bearer tokens, and
lease material do not get a new broad read surface.
### Inbox Message Envelope

`monoid.inbox-message.v1` (`core/inbox.py`, `InboxMessage`) wraps a message entering a
run so it carries **provenance** and an idempotency key. Like the control protocol it is an
edge/transport contract — the reference `RunnerBackend` wraps inbound content into it; the engine
(`AgentLoop`) never sees the envelope (it still receives unwrapped `content` via `submit`).

- Fields (CloudEvents-shaped): `id` (the dedup key), `source`, `type`, `run_id`, `created_at`,
  `correlation_id` (defaults to `id` — a flow root), `causation_id`, `traceparent`/`tracestate`,
  `content` (the JSON-native payload: a `str` or a list of
  content-part dicts), `metadata`. `is_inbox_envelope(obj)` discriminates an envelope from a legacy
  raw `str`/`list` queue entry.
- **Idempotent ingress**: `RunnerBackend.send_message(..., message_id=, source=, correlation_id=,
  traceparent=, tracestate=)`
  wraps + enqueues the envelope. A caller-supplied `message_id` makes the send idempotent — an
  already-processed id short-circuits to `status="duplicate"`, and a redelivery still in flight is
  dropped at dequeue. Processed ids are tracked per-run and **checkpointed** (`RunCheckpoint
  .inbox_seen_ids`), so dedup survives a restart (the marker rides the same checkpoint as the
  message's effects). Absent an id the edge
  mints one. HTTP `POST /v1/runs/{id}/messages` accepts optional `message_id`/`source`/
  `correlation_id`; a control `send_message` uses the command's `command_id` as the dedup key.
- Back-compat: the queue/checkpoint carry envelopes (JSON dicts), but legacy raw `str`/`list`
  entries from older checkpoints still restore and process.
- **Symmetric dedup on result ingestion**: `TaskManager.report_result` (the hosted-task result
  callback) is idempotent the same way — **first report wins**. A duplicate report (a callback
  retry) is a safe no-op that neither clobbers the recorded result nor re-publishes to the reentry
  queue (which would make the agent observe the result twice). The dedup signal is the
  already-persisted+rehydrated `ready_for_reentry`/`finished_at` job state, so it holds across a
  restart with no extra bookkeeping; the result dict carries a `duplicate` flag.

### Outbox Request

`monoid.outbox-request.v1` (`core/outbox.py`, `OutboxRequest`): a tool **stages** an
external side-effect (send an email, call a webhook) durably in the per-run `Outbox` instead of doing
the IO inline. The request is checkpointed, so it survives a restart; the engine never performs the
send.

- A tool handler calls `ToolContext.emit_outbox(destination, payload, *, capability,
  idempotency_key="")`; the request is appended to the per-run `Outbox` (checkpointed in full as
  `RunCheckpoint.outbox_requests`) and `outbox.requested` is emitted. The request carries the
  capability lease **handle** (`token_ref`, captured via `capability_token(capability)`) — never a
  secret. Bind the outbox tool with `runtime.requires_lease` so the existing capability gate
  brokers/revokes the lease *before* the send is staged (least-privilege egress).
- **Edge drains, effectively-once**: `RunnerBackend(outbox_sender_factory=lambda request: ...)`
  supplies an `OutboxSender` (`send(request) -> OutboxReceipt`); the backend drains
  `loop.pending_outbox()` at each park/settle, performing the IO (resolving `token_ref` to the real
  credential) and recording the outcome via `loop.record_outbox_result(...)` → `outbox.dispatched` /
  `outbox.failed`. The request is persisted `pending` before the send and `dispatched` after; a
  crash in between re-dispatches on recover, made safe by the `idempotency_key` the external target
  honors. A retryable failure stays `pending` and redrives up to `outbox_max_attempts`, then
  dead-letters as `failed`. No sender → requests stay durably `pending`.
- **Backoff + redrive (retry decoupled from run activity)**: a retryable failure stamps a durable
  `next_attempt_at` on the request — capped exponential backoff with **full jitter** (`uniform(0,
  min(outbox_retry_cap_s, outbox_retry_base_s * outbox_retry_factor**attempts))`). The drain only
  dispatches **due** requests (`loop.due_outbox(now)`; a freshly staged one has `next_attempt_at=0.0`
  → due immediately, so the happy path is unchanged), and because the schedule is on the checkpoint
  it survives a restart. The backend's **watchdog tick** also runs `_redrive_outbox()`: for each live
  run it marshals the drain onto the shared loop, so a due request is redispatched even while its run
  sits idle (redrive requires the watchdog running — the backend's operational background loop). The
  loop stays policy-free: the edge computes `next_attempt_at` and passes it to
  `record_outbox_result(...)`.
- Reference `reference/outbox.py`: `RecordingOutboxSender` (dev/tests), `FailingOutboxSender`
  (retry-path tests), and an `OutboxToolProvider` yielding a generic `outbox.send` tool.
- A request also carries `traceparent`/`tracestate` and
  `correlation_id`/`causation_id` (the request↔result link reused by ack-back). Per-destination
  routing is deferred.
- **Ack-back (request-reply, non-park)**: stage with `emit_outbox(..., expect_ack=True)` (the
  `outbox.send` tool exposes `expect_ack`/`reply_to`). When the send reaches a terminal outcome
  (`dispatched`/`failed`) the edge delivers the receipt **back to the run as an inbox message**
  (`type="outbox_ack"`, `correlation_id` = the request's flow, `causation_id` = the request id,
  carrying its `traceparent`) via the idempotent inbox path with a stable id (`ack_<request id>`) so a
  redelivery is a no-op. The agent observes it on its **next activation — it never parks**; a
  terminal run has no consumer, so the ack is dropped (documented). `reply_to` empty = the run's own
  inbox. Park-and-await (the agent suspending until the reply lands) is a deferred superset that
  reuses this same ack plumbing.

### Trace Context on envelopes (`traceparent` / `tracestate`)

Both envelopes carry optional W3C Trace Context (`core/trace_context.py`): `traceparent`
(`00-{trace-id}-{span-id}-{flags}`) and the opaque vendor `tracestate`. This is **observability
only** — it complements `correlation_id`/`causation_id` (the domain identity routing and
reply-matching depend on) and **application behavior never depends on it**; a missing or malformed
header is ignored.

- Helpers: `new_traceparent()` (fresh root), `child_traceparent(parent)` (same trace-id, new
  span-id), `parse_traceparent(s)` (validates shape, rejects all-zero ids, returns `None` on
  garbage), `trace_id_of(s)`.
- **Inbox (ingress)**: `send_message(..., traceparent=, tracestate=)` propagates an inbound trace
  onto the envelope. The engine unwraps the envelope before `submit`, so an outbox request can't
  auto-inherit the *causing* inbox message's trace inside the core — a fresh root is minted instead
  (cross-loop inheritance is a later edge enhancement).
- **Outbox (egress)**: `emit_outbox` stamps a fresh root `traceparent` at staging (pure, no IO) so
  the request is traced from birth; the edge sender derives a `child_traceparent` for the actual
  outbound call. The trace rides the `outbox.requested`/`outbox.dispatched`/`outbox.failed` events so
  the OTel event-sink mapper can stitch spans across a restart.

### Capability Request / Lease

Secrets stay outside the core. When a tool needs external access it carries a *capability*
requirement, and the loop acquires a scoped, expiring **lease** from a broker before running it.

- `CapabilityRequest` (`...capability-request.v1`) / `CapabilityLease` (`...capability-lease.v1`) /
  `CapabilityDenial` are plain data. A lease carries a `token_ref` **handle, never the secret** —
  the gateway/tool edge resolves it, not the core.
- `CapabilityBroker.request(req) -> CapabilityLease | CapabilityDenial` is the seam an integrator
  (Daemon/Cell) implements. `AutoGrantBroker` is the zero-config dev default; the reference
  `GatewayCapabilityBroker` mints a scoped gateway token as the lease handle (the "absorb the
  gateway" path); `DenyAllBroker` is the safe default.
- **Implicit, binding-declared**: a `ToolBinding` with `runtime.requires_lease` declares its tool's
  `capability` needs a lease; the agent just calls the tool. `AgentLoop(capability_broker=...)`
  gates the call: a cache miss requests a lease (scoped to the binding) and on grant proceeds; a
  denial raises so the call never runs and the model gets an actionable error. If no broker is
  configured, a required lease fails closed with `capability_broker_required`. For local development
  only, `runtime.requires_lease="optional"` preserves best-effort gating and lets the tool run
  without a broker. Events `capability.requested` / `capability.granted` / `capability.denied` give
  the audit trail.
- **Using the lease**: the granted handle reaches the running tool via
  `ToolContext.capability_token(capability) -> token_ref | None` (the handle, resolved at the
  edge). The reference backend provisions a per-run broker with
  `RunnerBackend(capability_broker_factory=lambda request: ...)` — scoped to the run's identity
  (e.g. a `GatewayCapabilityBroker` per tenant). `None` is only safe for bindings without required
  leases, or bindings that explicitly opt into `runtime.requires_lease="optional"`.
- **Security invariants the core enforces**: a grant may only NARROW the requested scope, never
  widen it (`CapabilityVault.admit` is fail-closed); a lease is expiry-checked before reuse; the
  per-run vault holds handles only and durable (approved) leases are checkpointed as handles, while
  ephemeral sync grants are re-brokered on restart. Any `CapabilityBroker` can be verified against
  these invariants with the parametrized `tests/test_capability_broker_contract.py` suite.
- **CLI**: `monoid run --auto-grant-capabilities` wires the built-in `AutoGrantBroker` (local
  dev), or `--capability-broker path.py:factory` loads a custom broker (`factory()` returns it).
- **Async approval (escalation)**: a broker may return `CapabilityPending` instead of granting
  synchronously — the loop then parks the run on a `capability` hosted-task (carrying the request
  AND the gated call) and hands the model a "pending" observation; when the grant is reported
  (`report_task_result` with a `lease`), the lease is admitted to the vault (fail-closed against the
  original request scope). `HumanEscalationBroker` (reference) escalates every request; a real
  policy broker auto-grants low-risk capabilities, denies forbidden ones, and escalates only the
  sensitive ones (the three-way `lease`/`denial`/`pending` outcome is the point).
- **Auto-redispatch** (`AgentLoop.capability_auto_redispatch`, default on): after the grant the loop
  re-executes the gated call automatically at the next step (through the normal tool path, real
  permission/quota/events) and delivers the result to the model — no model retry needed. If a replay
  can't run cleanly (no valid lease), it falls back to model-retry. The gated tool never executed at
  the gate, so the replay is its first and only execution (no double side effect).
- **Durable leases**: an escalation-approved lease is marked `durable` and checkpointed (the
  `token_ref` handle only, never a secret), so a restart does not re-prompt the approver; ephemeral
  sync grants are not persisted (re-brokered on restart). The gated call is captured in the durable
  hosted-task so auto-redispatch survives a restart too.
- **Revocation** (the operator/Daemon kill switch): `revoke_capability` (a Control command, or
  `AgentLoop.revoke_capability(...)`) records a revocation in the per-run vault; `get_valid` /
  `token_for` then refuse the handle **fail-closed**. Three granularities, one mechanism: per
  `capability` (authoritative — the gate refuses to even *re-broker*, so a permissive broker can't
  resurrect it), per `lease_id`, and an issued-before `before` watermark (a bulk cohort kill). Because
  a lease is only a handle the tool re-fetches per call, revocation just refuses to hand the handle
  back — instant, with no distributed secret clawback. Revocation state is checkpointed so a
  revoked capability stays dead across a restart. Emits `capability.revoked`. The shared
  `TokenManager` also supports gateway-edge revocation by token id (`jti`) or issued-before
  watermark when the deployment propagates that revocation state to the gateway verifier.
- **Rotation** (`AgentLoop.capability_rotate_skew_seconds`, default `0.0` = off): a cached lease
  within `skew` seconds of expiry is re-brokered on use — the handle/expiry refresh under a stable
  contract without a model retry or a re-prompt. Bounded by `CapabilityLease.max_expires_at`, an
  absolute ceiling so a one-time human approval is never silently auto-extended forever; past the
  ceiling the lease is left to expire (then the normal re-broker / re-escalation path applies). A
  deny/pending/scope-widening rotation leaves the still-valid current lease untouched (no
  in-flight disruption). Emits `capability.rotated`.
- **Web tools through the gate (opt-in)**: the built-in `web.search` / `web.fetch` / `web.context`
  tools declare a `capability`; set `runtime.requires_lease` on their binding and the existing gate
  brokers a lease before each call. The lease handle becomes the request's `Authorization` (threaded
  context → `WebService` → `WebGatewayClient` as a per-call credential override; absent a lease, the
  client uses its static run-start token — back-compat). The reference `GatewayCapabilityBroker`
  mints a **web-gateway-compatible** token (`kind=web_gateway`/`aud=csp.web-gateway`) for `web.*` so
  the existing web gateway accepts it unchanged. Brokered web tokens carry a signed `metadata.scope`
  containing the binding id, domain scope, and web runtime caps such as `max_calls`; the web gateway
  applies that signed scope before provider invocation. Payload constraints can narrow the signed
  scope, and requests that widen `allowed_domains`, `binding_id`, or numeric caps fail with
  `web_scope_denied`. Net effect: web access inherits rotation + revocation (an operator can
  `revoke_capability("web.search")` to kill a live run's web access without cancelling it). The LLM
  path is deliberately NOT routed this way.
- **Gateway model-token refresh** (separate from capabilities): `GatewayModelAdapter.token_provider`
  is an optional per-request token source; the reference backend wires a source that re-mints the
  `llm_gateway` token near expiry, so a run outliving the token TTL keeps LLM access without a
  restart. Default (no provider) is the static token. This is a refresh seam, not capability routing
  — the LLM hot path stays out of the broker.

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
  "protocol": "monoid.llm-turn.v1",
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

The kernel sends one of two request styles. **By-value `messages` is the default**: the
full provider-neutral conversation log (`messages`, a list of `{role, content}` user /
assistant / tool entries) travels on every turn, and the gateway forwards it statelessly —
`previous_turn_handle` and `observations` are not consulted. The conversation is
reconstructed from the checkpoint rather than a server-side handle, so this style survives a
restart.

The **handle-based** style (shown in the example above) is the fallback, used when a turn
carries no `messages`. It has three shapes, selected by `previous_turn_handle` and
`instruction`:

- **first turn** — no `previous_turn_handle`; carries `instruction`.
- **tool continuation** — `previous_turn_handle` + `observations`; no `instruction`.
- **user follow-up** — `previous_turn_handle` + `instruction` (a new user message on
  top of an existing continuation handle; `observations` is empty).

Either style lets one run accept multiple user turns: with `messages` the new user message
is appended to the log; with a handle the kernel threads the last `turn_handle` into the
next user message.

Successful response:

```json
{
  "protocol": "monoid.llm-turn-result.v1",
  "turn_handle": "turn_...",
  "final_text": null,
  "tool_calls": [
    {"call_id": "call_1", "name": "read_notes", "arguments": {"path": "notes.md"}}
  ],
  "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
}
```

`usage` always carries `input_tokens` / `output_tokens` / `total_tokens`. It MAY
additionally carry optional priced sub-counts when the provider reports them —
`cache_read_tokens`, `cache_creation_tokens`, `reasoning_tokens`, `audio_tokens` —
which the kernel sums into per-run totals and checks against the token budget. These
fields are additive; a consumer that ignores them stays correct.

The reference gateway tokens authenticate run identity. New tokens include a `kid` header. A
`TokenManager` can be built from a keyring, rotated to a new active key, and configured to accept
retired keys only until a grace-window deadline. Verification also rejects revoked token ids and
issued-before cohorts before any gateway action proceeds. The LLM request model still selects the
turn model.

### Web Gateway

The kernel calls:

- `POST /internal/web/search`
- `POST /internal/web/fetch`
- `POST /internal/web/context`

Every request includes binding constraints:

```json
{
  "protocol": "monoid.web-search.v1",
  "binding_id": "search_docs",
  "query": "monoid runtime config",
  "max_results": 5,
  "max_calls": 20,
  "allowed_domains": ["docs.example.test"],
  "blocked_domains": []
}
```

The reference gateway enforces per-run/binding call counters and signed token scope. Brokered web
tokens carry `metadata.scope`; payload domain, binding, and numeric limit constraints can narrow
that scope and cannot widen it. Scope violations fail before the provider adapter is called.

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
conversation, and run state all roll back to one aligned instant. This is a
**state-snapshot at the suspend points** (not event-sourcing replay); snapshots are
only taken at clean park points, so there is no determinism constraint and no
double-side-effect risk.

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
- The reference backend writes `run_dir/run.json` and stores the same recovery descriptor in the
  configured `CheckpointStore`: identity, workspace, limits, policy, and the authoritative resolved
  runtime config. Runtime-config hot-swaps update both copies with `runtime_config_version`,
  `runtime_config_hash`, `runtime_config_issuer`, `runtime_config_reason`, and
  `runtime_config_committed_at`; recovery verifies the hash before rebuilding providers or gateway
  token sources. A backend that never hosted the run can reclaim it from a shared lease/checkpoint
  store, read the shared descriptor when local `run.json` is absent, materialize a local copy, then
  resume. `recover_runs()` scans `run_root`; the active watchdog discovers cross-instance orphaned
  runs from the shared lease store. Recovery skips terminal checkpoints and failed runs, rebuilds
  each run (re-issuing gateway tokens from the signing key, **re-provisioning the base workspace**
  is the deployment's job), `restore()`s the loop with the store's blobs, re-enqueues durably-saved
  follow-up messages, and resumes.

**Assumption (workspace):** the agent workspace is not durable; on restore the
deployment re-provisions the base (re-clone/re-mount) and the checkpoint re-applies
only the agent's delta (the delta always contains the agent's created/modified
files). For container durability, `run_root` (or the `CheckpointStore`) must point at
durable storage — a mounted volume needs no code change.

**Limitations (v2):** a mid-run `commit_checkpoint` re-baseline combined with delta-
restore is a documented follow-up (the common no-re-baseline case is covered).
Multimodal message parts (image/document) round-trip through the checkpoint, so a
resumed run re-forwards the media. `transcript.jsonl` is a debug artifact (the
by-value `messages` in the checkpoint are the load-bearing conversation record).

## Production Hardening

Operational safety net layered on durable persistence. The core still never auto-recovers;
the active watchdog lives only in the reference backend (the operational layer).

### Failure surfacing & bounded recovery

- **Failure bundle on every failure.** Beyond the core's own `failure.json`, the reference
  backend's `_record_run_failure` also writes `run_dir/failure.json`
  (`monoid.failure.v1`: `error, error_code, type, last_good_seq, restore_hint,
  failed_at`) — the durable mark is written *before* the in-memory terminal status, so a
  worker crash that bypassed the loop's own bundle still leaves a mark and a restart never
  resumes a crashed run into a loop.
- **Bounded recovery.** `recover_runs()` logs (not swallows) a resume failure and tracks
  attempts in `run_dir/recover_attempts.json` (`{count}`); after the cap it writes a
  `failure.json` with `error_code="unrecoverable"`, so a poison checkpoint is permanently
  skipped instead of retried forever.

### Active watchdog / lease (backend only)

- `RunnerBackend.start_watchdog()` / `stop_watchdog()` run an opt-in heartbeat thread (tick
  `watchdog_interval_s`, default 5s). For each owned live run it refreshes
  `run_dir/lease.json` (`worker_id`, `pid`, `heartbeat_at`, `lease_ttl_s`; default
  `lease_ttl_s=30`), and deletes the lease on terminal.
- It reclaims a run whose lease has gone stale (`heartbeat_at + lease_ttl_s < now`) and a
  crashed worker left behind: reclaim takes the lease via a compare-and-swap, so two
  backends racing the same run produce exactly one winner, then resumes via the
  `recover_runs()` path.
- Lease storage + the CAS are a pluggable **`LeaseStore`** (default `LocalFsLeaseStore`:
  `lease.json` + `file_lock(run_dir/.reclaim.lock)`); see *Pluggable durable stores*.

### CheckpointStore robustness invariants

`LocalFsCheckpointStore` (and any conforming store):

- **Monotonic `LATEST`:** the pointer is only advanced when `checkpoint.seq` exceeds the
  current `LATEST` seq — a late or lower-seq writer can never unpublish a newer committed
  checkpoint.
- **Orphan blob GC:** crash-leftover `blobs/*.tmp` files are cleaned on `put()`/`latest()`.
- **Cross-process serialization:** `put()` holds `file_lock(checkpoints/.put.lock)`
  (`core/_util.file_lock`, O_EXCL with stale-steal); `latest()` retries a read that races a
  concurrent commit's atomic replace, so a reader never mistakes mid-commit for "no
  checkpoint."

### Pluggable durable stores

Two seams make durability and multi-node recovery pluggable without touching the loop:

- **`CheckpointStore`** (core) — `put(checkpoint, blobs)` / `latest` / `delete`.
  `CheckpointRecord.blob(sha)` is a callable, not a directory, so a store can back blobs with
  files, a DB, or an object store.
- **`LeaseStore`** (reference) — `candidate_run_ids` / `heartbeat` / `is_stale` / `try_claim`
  (atomic CAS) / `owner` / `release`. The watchdog policy stays in `RunnerBackend`; only the
  lease's storage and its claim atomicity live here.

Every store must pass the parametrized contract suites (`tests/test_checkpoint_store_contract.py`,
`tests/test_lease_store_contract.py`): atomic last-good commit, monotonic `latest`, write-once
blob dedup, run metadata round-trip, and a single-winner `try_claim`. Passing them makes a backend
a drop-in.

**SQLite reference stores** (`reference/stores/`, stdlib `sqlite3`, zero dependencies):
`SqliteCheckpointStore` and `SqliteLeaseStore`. A DB transaction supplies the invariants —
`put` commits atomically (a crash rolls back, so `latest` never sees a torn checkpoint), the
latest pointer advances monotonically via a conditional UPSERT, blobs are write-once, and
`try_claim` is a transactional CAS under `BEGIN IMMEDIATE`. `SqliteCheckpointStore` also stores the
backend run descriptor beside checkpoints, so one shared db can host **both** stores and the
recovery metadata needed to reclaim and resume a crashed peer's run across the instance boundary
(a per-host `lease.json` cannot):

```python
db = "/shared/monoid.db"
backend = RunnerBackend(
    ...,
    checkpoint_store=SqliteCheckpointStore(db),
    lease_store=SqliteLeaseStore(db),
)
```

**Limitation / follow-up:** SQLite is single-host. A true cross-*host* deployment swaps in a
networked `CheckpointStore` / `LeaseStore` (an object store or a networked DB) behind the same
seams, as an optional dependency.

### HTTP hardening & request bounds

Shared in `reference/_shared/http_util.py`, applied to the backend / llm-gateway /
web-gateway HTTP layers:

- `read_json_limited(handler)` rejects a body whose `Content-Length` exceeds
  `MAX_REQUEST_BYTES` (10 MB) with **413** before reading — a DoS/OOM guard.
- `HardenedThreadingHTTPServer` sets a per-connection `REQUEST_TIMEOUT_S` (30s) socket
  timeout and shuts down cleanly (`daemon_threads=False`, `block_on_close=True`) so a slow
  client cannot pin a thread and in-flight handlers are not abandoned.
- `redact_internal_error(...)` logs an unmapped 5xx in full server-side under a
  `correlation_id` and returns only that id to the client (never a stack trace / path);
  intentional client-facing errors (`ValueError`/`PermissionDenied`/`KeyError`) keep their
  message. `log_http_request(...)` emits a structured access line.

### Resource & DoS bounds

- **`RunLimits`** (core): `max_messages` / `max_message_log_bytes` bound the by-value
  conversation; `max_workspace_delta_bytes` / `max_delta_file_bytes` bound a checkpoint's
  workspace delta. Exceeding a cap on **capture** settles the run `limited` (a safe stop,
  not a drop — the prior good checkpoint stays the recovery point); exceeding on **restore**
  refuses the checkpoint (`workspace_delta_bytes_exceeded` /
  `workspace_delta_file_bytes_exceeded`). Defaults are generous backstops.
- **Backend:** `max_message_bytes` (reject over-large follow-up message),
  `max_message_queue_depth` (cap pending-message queue), `max_concurrent_runs` (a bounded
  semaphore; excess submissions stay `queued`, `0` = unbounded).

### Client connection retry

The gateway model adapter (`providers/gateway.py`), the web gateway client (`web.py`), and
the web upstream providers (`reference/web_gateway/providers.py`) retry transient
connection-level failures (`URLError` / `TimeoutError` / a bare `OSError` such as a
connection reset mid-read) with backoff. An `HTTPError` is a real response and is **never**
retried as a connection error. The model adapter's retry is policy-driven by
`ModelRetryConfig.retry_on` (default codes: `gateway_timeout`, `gateway_network_error`,
`gateway_rate_limited`, `gateway_server_error`).

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
