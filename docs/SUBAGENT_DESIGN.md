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

## Tool exposure

Tools are exposed only through explicit `ToolBinding`s in the runtime config
(`compile_bound_tool_catalog`). The `Agent` tool spec is registered in the base
registry **only when `subagent_definitions` is non-empty**, and the runtime config
author adds a binding (`ref.tool_id = "agent.spawn"`) to expose it. Depth is
enforced at call time. Additionally, when building a child that would sit **at** the
depth cap, `_run_subagent_child` strips the `agent.spawn` binding from the child's
config and gives it no definitions, so the tool is simply absent from that child
(no wasted turn on a call-time error). The call-time check remains as defense in
depth.

## Definitions source (decided: inline)

P1: `AgentLoop.subagent_definitions: Mapping[str, AgentRuntimeConfig]` — inline,
passed by the embedder. Directory discovery (`--agents-directory`, Claude
`.claude/agents` style) is P3 and will share machinery with Skills.

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
- **P3** (deferred): directory discovery (`--agents-directory`), Skills
  `context: fork` integration, optional ADK-style state copy-in/delta-back, parent
  token-budget integration (counting child tokens against the parent's RunLimits).
