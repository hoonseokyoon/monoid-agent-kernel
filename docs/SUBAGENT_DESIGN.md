# Subagent (agent-as-tool) design

Status: P1 in progress (branch `feat/subagent-agent-as-tool`).

## Goal

Let a run delegate a focused task to a **child run** that works in an isolated
context window and returns only its final message to the parent. This is the
**agent-as-tool** pattern (Claude Code `Agent`/`Task`, OpenAI `agent.as_tool()`,
Google ADK `AgentTool`). It is the lowest-risk of the three canonical delegation
patterns for this runner because it needs **no shared cross-agent state bus**:

| Pattern | Context | New plumbing here |
|---|---|---|
| **Agent-as-tool** (chosen) | child isolated; only final message returned | ~none — reuse task machine + workspace + checkpoint |
| Handoff | child inherits history | moderate — history serialization + handback rules |
| Supervisor | shared state, central router | heavy — a shared-state bus we do not have |

## Core idea: subagent = a new task kind

The existing background-task machine already models "park the parent, run work,
re-inject the result." A subagent is just a new task **kind** plugged into it. No
changes to the pump/parking/reentry loop.

The mechanism mirrors `request_human_input` ([loop.py] `AgentToolContext`), which
calls `job_manager.start_task("hitl", …)`, and the shell foreground/background
split in `ShellService.execute` ([tool_services/shell.py]):

```
parent turn: model calls the Agent tool
  handler -> context.spawn_subagent(subagent_type, prompt, background)
    -> job_manager.start_task("subagent", {definition_id, prompt, depth, background})
       SubagentTaskExecutor.start():
         - validate: definition exists, depth < max_depth, fan-out < max_subagents
         - register a HostedTask(kind="subagent")
         - schedule run_child(...) on the run's event loop (schedule_job_coroutine)
         - return the task
    background=False (foreground):
         handler blocks on job_manager.wait(task_id) and returns the child's
         final message as the tool result            (mirrors execute_shell)
    background=True:
         handler returns "started" content; on completion the result is injected
         later as a user message via reentry          (mirrors start_shell_job)

  run_child (closure in AgentLoop bootstrap):
    - emit subagent.started on the PARENT recorder (parent_id = spawn tool-call event)
    - build an isolated child AgentRunSpec (overlay workspace, child run_id)
    - build a child AgentLoop (shared model_adapter, checkpoint_store,
      cancellation_token; runtime config from the subagent definition; depth+1).
      External event_sinks are NOT shared — see Observability below.
    - result = await child.arun_once(prompt)
    - task.result = {final_text, status, usage, child_run_id, ...}
    - emit subagent.finished on the PARENT recorder (parent_id = subagent.started)
    - manager.mark_ready(task)
```

### Why this is re-entrancy-safe

Tool handlers run **off the loop thread** (`await asyncio.to_thread(...)` in the
pump). The child runs on the run loop as a scheduled coroutine via the existing
`TaskManager.schedule_job_coroutine` (the same mechanism shell subprocess monitors
use). A foreground handler blocks its worker thread on `wait()` while the loop
keeps driving the child. The child uses the **async** path (`arun_once`), so the
sync-API re-entrancy guard ([loop.py], "sync API called inside a running event
loop") is never hit.

## Foreground vs background

Both are the same task kind; the only differences are:

- **Foreground** — `resume_on_exit=False`; the handler blocks on `wait()` and
  returns the child's final message directly as the tool result. Sequential.
- **Background** — `resume_on_exit=True`; the handler returns immediately; the
  result is injected later as a user message through the reentry queue. Because the
  reentry queue already drains many finished tasks at once, **multiple background
  subagents run concurrently for free** (parallel fan-out).

## Isolation (decided: isolated overlay)

The child gets its own `overlay` workspace backend. Its file changes stay in the
child's in-memory overlay and surface to the parent only through the returned
final message (the parent does not see intermediate edits). This matches Claude's
`isolation: worktree` model. The child run dir lives under the parent's
`run_root` keyed by a distinct child `run_id`, so checkpoints never collide.

## Safety bounds

- **Depth cap** (`RunLimits.max_subagent_depth`, default 5): enforced at spawn
  time in the executor. A child carries `subagent_depth` in `spec.metadata`;
  `spawn_subagent` passes it through and the executor rejects spawns at the cap.
- **Fan-out cap** (`RunLimits.max_subagents`, default 8): max subagent tasks per
  run, enforced in the executor.
- **Cancellation**: the parent's `cancellation_token` is shared with the child, so
  cancelling the parent settles the child and unblocks a foreground `wait()`.
- **Usage**: the child's token usage is returned in `task.result["usage"]` so the
  parent (and accounting) can attribute it.

## Observability (correlation + usage)

The child run records its full event stream to its **own** run dir via its own
recorder. External `event_sinks` are deliberately **not** shared with the child:
sinks like `OtelEventSink` and `StatusJsonSink` hold per-run state (one root span,
one status doc), so a shared instance would be clobbered by the child's
`run.started`.

Instead the parent emits two summary events on its own stream:

- `subagent.started` — `parent_id` = the spawn tool-call event id, so it nests under
  that tool call; data carries `subagent_type`, `child_run_id`, `depth`, `background`.
- `subagent.finished` / `subagent.failed` — `parent_id` = the `subagent.started`
  event id (close pairing); data carries `status`, `usage` (the child's token
  totals, for delegated-cost attribution), and `error`/`error_code`.

`OtelEventSink` maps this pair to an `execute_subagent {type}` span nested under the
spawn tool span (foreground) or under the run span (background, whose tool span has
already closed). `child_run_id` ties the summary back to the child's own trace/log.

## Tool permissions (Claude-parity, P2.5)

A subagent's tools are **derived from the parent's**, never declared independently —
so a subagent can never exceed the parent (hard ceiling). `_resolve_child_config`:

1. Start from the parent's current bindings.
2. `definition.tools is None` → inherit all of them; a tuple → keep only parent
   bindings matching the allowlist.
3. Remove any binding matching `definition.disallowed_tools` (deny wins).
4. At the depth cap, also drop the `agent.spawn` binding (tool absent, not a
   call-time error — the call-time check in the executor remains as defense in depth).

Matching is fnmatch against each binding's tool id / binding id / model name, so
`fs.read`, `mcp.*`, `mcp.github.*`, and `*` all work. Because the inherited binding
objects are reused, their `scope`/`quota`/`guidance` carry over too — a subagent's
path/command scopes are inherited, not widened.

`model`/`mode`/`limits`/`tool_search` inherit the parent's unless the definition
overrides them. The parent's `tool_providers`/`dynamic_tool_providers` (MCP/custom)
are passed to the child so the inherited bindings resolve against the same registry.

The `agent.spawn` tool spec is registered in the base registry **only when
`subagent_definitions` is non-empty**; the runtime config author still adds a binding
(`ref.tool_id = "agent.spawn"`). The tool advertises the available subagent ids +
descriptions so the model selects the right one (Claude selects by description).

## Definitions source

`AgentLoop.subagent_definitions: Mapping[str, SubagentDefinition]` — inline, or loaded
from a directory. `load_subagent_definitions(dir)` (CLI `--agents-directory`) scans
`*.md` files with YAML frontmatter (`.claude/agents` style) via the zero-dep
`core/frontmatter.py` parser (shared with Skills' `SKILL.md`). `name` is the id (falls
back to filename); frontmatter maps to `SubagentDefinition.from_frontmatter`.
`SubagentDefinition.from_runtime_config()` also adapts an explicit config.

## Context fork (P3)

`SubagentDefinition.context = "fork"` (default `"fresh"`). A fork inherits a **snapshot
of the parent's conversation** (the parent's by-value `messages` log, seeded into the
child via `arun_once(..., seed_messages=...)`) AND the parent's prompt / tools / model
(`_resolve_child_config(..., fork=True)` ignores the definition's own prompt/tools/model
and `mode`/`limits` inherit too). The child still runs in an isolated overlay workspace
and returns only its final message — so a fork is "continue as me in an isolated
branch", cheaper to reason about than re-explaining context to a fresh subagent. The
system prompt is regenerated per turn from the (inherited) config, so the fork applies
the parent's directive over the inherited history.

## Files

| File | Change |
|---|---|
| `core/spec.py` | `RunLimits.max_subagents`, `RunLimits.max_subagent_depth` |
| `tasks.py` | `SubagentTaskExecutor`; reuse `HostedTask` + `HostedResultInjector` |
| `tools/builtin.py` | `agent_spawn_tool()` (not in the default `builtin_tools` list) |
| `loop.py` | `AgentLoop.subagent_definitions`; `AgentToolContext.subagent_depth` + `spawn_subagent`; bootstrap registers the Agent spec + executor/injector + `run_child`; propagate capability to the child |
| `tests/test_subagent.py` | foreground/background/depth/fan-out/isolation |

## Phases

- **P1** (this branch): executor/injector + Agent tool + isolated overlay +
  foreground & background + depth/fan-out caps + inline definitions.
- **P2** (done): `subagent.started`/`finished` events with `parent_id` correlation
  (OTel `execute_subagent` spans) + child usage in the event/span + hide the Agent
  tool at max depth + stop sharing stateful sinks with children + `CONTRACTS.md` +
  `contracts.py` exports (`SubagentTaskExecutor`, `agent_spawn_tool`).
- **P2.5** (done): Claude-parity tool permissions — `SubagentDefinition` (inherit
  tools/model/mode/limits by default; `tools` allowlist + `disallowed_tools` denylist
  with fnmatch incl `mcp.*`), hard ceiling (allowlist resolved against parent bindings),
  parent tool-provider inheritance (MCP/custom), description-based selection in the
  `agent.spawn` tool.
- **P3** (done): directory discovery (`--agents-directory` + `core/frontmatter.py`),
  context fork (`context: "fork"` — inherits parent conversation + prompt/tools/model),
  report-only usage roll-up (`subagent_count`/`subagent_usage` in metrics, kept out of
  `total_usage` to preserve context accounting).
- **Deferred**: Skills `context: fork` wiring (needs the Skills feature), ADK-style
  state copy-in/delta-back, parent token-budget *enforcement* across the tree (only
  AutoGen does this; we report rather than enforce to keep context accounting clean).
