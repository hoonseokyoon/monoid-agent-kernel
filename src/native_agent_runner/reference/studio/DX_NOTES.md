# Studio DX notes

A running log of developer-experience gaps found while building Agent Studio against the
contracts + reference services alone. Each entry: what hurt, where, and the proposed core fix.
Building the app is the pressure test; this file is the yield.

## Status legend
- 🔴 open — gap confirmed, not yet addressed in core
- 🟡 worked-around — Studio papers over it locally; core fix still wanted
- 🟢 fixed — addressed in core/reference

---

### DX-11 🟢 Streaming a child subagent's work to the parent UI — FIXED
Building the subagent feature with live progress. A spawned subagent is an **isolated child run**
(`AgentLoop` with run id `<parent>.sub.<task>`, its own recorder); the parent's event stream only
carries `subagent.started`/`subagent.finished` (with `child_run_id`) — the child's tool calls and
token deltas go to **its own** `run_root/<child_run_id>/events.jsonl`. The child loop also did not
inherit token streaming.
- **Core:** the child `AgentLoop` now inherits `emit_output_deltas`, so the child streams
  `model.output.delta` into its own events.jsonl too.
- **Backend (the clean fix):** `RunnerBackend.descendant_events(run_id, token, descendant_run_id)`
  — authorize via the *ancestor's* run token, verify lineage by id prefix (a subagent id always
  extends its parent's with `.sub.<task>`, at any depth) + reject path separators, then read the
  descendant's events.jsonl. A child run has no record/token of its own, so this is how an embedder
  tails it without filesystem access.
- **Studio:** `subagent_events(child_run_id)` derives the ancestor run id from the child id, looks
  up that run's token (held server-side) and calls `descendant_events` — the earlier direct
  events.jsonl read is gone. The UI polls `/api/subagent-events` to render a nested card with the
  child's live tool calls + streamed tokens. Verified live: a child streamed 54 token fragments
  that reassembled to its final text.

### DX-10 🟢 Backend config validation rejected the dynamically-registered agent.spawn — FIXED
Binding `agent.spawn` in a runtime config failed `validate_runtime_config` with "unknown registry
tool: agent.spawn". Cause: `agent.spawn` is registered by the loop bootstrap **only when the run
carries `subagent_definitions`**, but the backend validates configs (`submit_run`,
`replace_runtime_config`) against the static `builtin_tools` registry, which never includes it.
Fix: `_backend_builtin_tool_specs(subagent_definitions)` appends `agent_spawn_tool(catalog)` when
the backend has subagent definitions, so validation matches what the loop will actually register.

---

### DX-8 🟢 Token streaming isn't reachable from the autonomous (submit_run) path — FIXED
**Found** building the Tier-1 "token streaming" item. The loop only streamed when a RunStream sink
was active (`loop.astream()` / backend `astream_run`); studio drives chats via `submit_run` →
`arun_until_suspended` (autonomous, no sink), deltas went to the RunStream **queue** (not
`events.jsonl`/SSE), and `OpenAIModelAdapter` had **no `astream_turn`** (deferred P4b-③) so even the
gateway path yielded one assembled chunk. Three layers, none of which was the core *engine*:
- **Provider**: added `OpenAIModelAdapter.astream_turn` (async, `AsyncOpenAI`, `responses.create
  (stream=True)`) mapping Responses stream events → neutral `TextDelta`/`ToolCallDelta`/`TurnComplete`
  chunks. The gateway's `_stream_turn` already forwards a provider `astream_turn` when present (else
  synthesizes one chunk), so **no gateway change** was needed.
- **Core**: opt-in `AgentLoop.emit_output_deltas`. When set and the adapter supports `astream_turn`,
  the *autonomous* drive streams via a new `_acall_model_emitting_deltas` that emits each text
  fragment as a `model.output.delta` event (new type + schema) and folds chunks into the identical
  assembled `ModelTurn`. This keeps studio's existing multi-turn `submit_run`/SSE transport — no
  switch to `astream_run` (which is single-submit, no HITL-over-stream). Off by default (CLI/others
  unaffected); adapters without `astream_turn` (offline echo) fall back to `next_turn`.
- **Backend + studio**: `RunnerBackend.emit_output_deltas` plumbed into `_build_loop`; studio turns
  it on and the UI renders `model.output.delta` into one assistant bubble live (`finalizeStream`
  reconciles with the authoritative `final_text` on settle).

Verified live with real OpenAI: 10 incremental fragments (`One`, `…`, `Two`, …) that join exactly to
the settled `final_text`. This stream also makes the DX-9 Stop *immediate* (abort the async
iterator mid-token) — implemented; see DX-9's resolution note.

### DX-9 🟢 "Stop" is run-level only (no turn-level interrupt) — FIXED
Cancellation (`RunnerBackend.cancel_run` / `CancellationToken`) terminalizes the **whole run**, so
the first Stop button ended the conversation. Added a turn-level interrupt that keeps the session
alive, reusing the recoverable-turn (DX-7) park pattern:
- **Core**: `AgentLoop.interrupt_turn()` sets a per-turn flag (distinct from the run-level
  `cancellation_token`); `_check_run_boundary` raises the new `TurnInterrupted` at the next step
  boundary; `arun_until_suspended` converts it to a non-terminal `Suspension(reason="interrupted")`
  (no error, session alive) + emits `turn.interrupted`. The flag is cleared on consume and at each
  new submit (a stale stop can't kill the next turn).
- **Backend**: `interrupt_turn(run_id, token)` signals the loop; `_drive_open_session` parks the
  multi-turn session (`await_user_input`) on `reason=="interrupted"` (a one-shot run just closes).
- **Studio**: Stop button → `POST /api/interrupt` → `interrupt_chat`; the composer stays enabled and
  the next message continues the same conversation (`/api/cancel` is kept for "end the run").

**Step-boundary limit → resolved on streaming (DX-8).** Without streaming the interrupt lands at the
next step boundary (an in-flight non-streamed model call finishes first). With token streaming
(`emit_output_deltas` + an `astream_turn` adapter), `_acall_model_emitting_deltas` checks the flag
after every chunk and raises `TurnInterrupted` + `aclose()`s the generator — aborting the in-flight
generation within one token (the partial text already streamed stays). Verified live with OpenAI:
Stop mid-essay halted at ~8 fragments, no `turn.settled`, session alive. So studio's Stop is now
immediate during a streamed turn, step-boundary otherwise.

---

### DX-7 🟢 A recoverable model error killed the whole conversation
**Where:** `loop.py` terminalized the run on *any* model-turn exception; a 4xx/429/transient
error (e.g. `reasoning effort=minimal` rejected by gpt-5.5) ended the session, after which
`RunnerBackend.send_message` refused follow-ups ("cannot send a message to a terminal run").
Found live in Studio.

**Fixed (core mechanism + backend policy + reference fidelity), prior-art-aligned (OpenAI
Assistants: thread survives a failed run; LangGraph: state not advanced on failure):**
- core: `loop.py` classifies recoverable model errors and returns a non-terminal
  `Suspension(reason="turn_failed", retryable, http_status)` (idempotent re-attempt — only
  `pending_observations` is cleared), emits `turn.failed`, keeps the session alive; adds
  `fail_recoverable` for give-up. Suspension.reason / AgentEventType / schemas extended additively.
- backend: `_drive_open_session` retries transient turn failures with async backoff, parks
  config-4xx for the user to fix + resend, gives up after `max_consecutive_turn_failures`.
- reference fidelity: `providers/openai.py` now maps a provider 4xx to a classified
  `ModelAdapterError(http_status, retryable)` (body-free) instead of leaking the raw SDK error
  (which the gateway had mistranslated to a retryable 500).
- studio: `turn.failed` renders inline, the composer stays enabled, and a model/effort change
  auto-resends (plus a Retry button).

Verified live: `effort=minimal` → `turn.failed (400, retryable=false)`, session parked (not
terminal), `send_message` accepted, `effort=medium` → resend settles. Covered by
`test_loop.py` (P1), `test_backend.py` + `test_cli_and_openai.py` (P2), `test_studio.py` (P3).

---

### DX-1 🟢 LLM gateway has no key-less / fake provider seam
**Fixed:** added `reference/llm_gateway/providers.py` (`EchoModelAdapter`, `offline_provider_factory`)
— the LLM-side counterpart of `FakeWebProvider` — and a `native-agent llm-gateway serve
--provider {openai|fake}` flag. Studio now imports the gateway's offline provider instead of
shipping its own copy. Covered by `test_llm_gateway_offline_provider_answers_without_a_key`.

**Where:** `reference/llm_gateway/service.py` — `LlmGatewayBackend._build_adapter` hard-defaults
to `OpenAIModelAdapter(allow_direct_provider_api=True)` when `provider_adapter_factory is None`.

**Hurt:** To stand up *any* local run without an OpenAI key, the integrator must hand-write a
`ProviderAdapterFactory`. The WebGateway already ships `--provider fake` (`FakeWebProvider`); the
LLM gateway has no equivalent. The existing `runs/integration-real-*` artifacts even show the
failure mode of the implicit OpenAI path (`'OpenAI' object has no attribute 'responses'` → HTTP
500), i.e. the default is both key-requiring *and* fragile.

**Worked around:** Studio ships `EchoModelAdapter` + `offline_provider_factory`
(`reference/studio/provider.py`) and passes it in by default.

**Proposed core fix:** add a first-class offline/echo provider to the reference llm_gateway and a
`native-agent llm-gateway serve --provider {fake|openai}` flag, mirroring the WebGateway. Keeps
the "works with zero keys" promise symmetric across gateways.

---

### DX-2 🟢 No clean "drain & stop my active runs" on RunnerBackend
**Fixed:** added `RunnerBackend.drain(timeout_s=...)` (cancel owned runs + wake parked sessions +
wait for terminal) and a `shutdown(drain=True)` flag. Studio's shutdown is now a single
`backend.shutdown(drain=True)` instead of cancel-each + sleep. Covered by
`test_backend_drain_ends_parked_multi_turn_sessions`.

**Where:** `reference/backend/service.py` — `RunnerBackend.shutdown()` only stops the watchdog
(by design: the run loop is process-shared). Parked multi-turn sessions are left as pending
coroutines.

**Hurt:** An app that boots a backend and later stops it (Studio's "close the window → stop the
app") leaves parked session coroutines on the shared loop. At interpreter exit this surfaces as
`Task was destroyed but it is pending` / `Event loop is closed` noise. There's no single call to
"cooperatively end the runs this backend owns."

**Worked around:** `StudioServer.shutdown()` iterates its known run ids, calls `cancel_run` on
each (which enqueues the close sentinel), then sleeps briefly to let the loop drain.

**Proposed core fix:** a `RunnerBackend.drain(timeout=...)` (or a `shutdown(drain=True)` flag)
that cancels owned runs and awaits their teardown, so embedders get clean shutdown without
reaching for `cancel_run` + `sleep`.

### DX-3 🟢 Events carry no presentation-ready summary for a UI activity feed
**Fixed (by sharing a projection, not by changing the event schema):** added
`native_agent_runner.narration` — `narrate_event(event) -> EventNarration` maps an event to a
*neutral* `(category, action, target, status, level, detail)` descriptor. The `watch` CLI
(`_compact_event_line`) and the Studio feed (`activity.describe_event`) now both format that one
projection instead of each re-deriving the verb/target. This matches the prior art (AG-UI / Vercel
AI SDK / OTel keep events typed and render at the edge; baking a localized string into the event
was the wrong move). Covered by `tests/test_narration.py`.

(superseded — original finding kept for history:)

**Where:** the public event stream (`tool.call.started` / `tool.call.finished` / `workspace.*`).
Found while building the R1 activity feed.

**Hurt:** To show "what is the agent doing right now" you must hand-maintain a verb table keyed
by the *wire* tool name (`fs_read`, `shell_exec`, …) and heuristically dig the action target out
of `args_preview` / `paths`. Every integrator who wants a feed reinvents this, and it silently
drifts when tools are added/renamed. There is no human `summary` and no typed `(verb, target,
status)` on the event.

**Worked around:** `reference/studio/activity.py::describe_event` maps events → a line server-side,
attached to each SSE frame as `studio_activity`. Covered by `test_describe_event_*`.

**Proposed core fix:** have the engine attach an optional `summary` (and/or a structured
`tool_activity` shape) to tool/workspace events, derived once at the source where the verb and
args are already known — so every UI gets a feed for free and it can't drift.

### DX-4 🟢 No mid-run API for the proposal diff text
**Fixed:** added `RunnerBackend.proposal_diff(run_id, token)` and `GET
/v1/runs/{id}/proposal/diff` — the unified diff on demand, mid-run, token-scoped (GitHub serves
PR diffs the same way: an on-demand representation of the resource). Studio now calls it instead
of reading `run_dir/diff.patch`. Binary files (images/docs) appear in the patch as a
`<binary sha256=… size=…>` marker; the actual bytes are fetched via `proposal_file` (base64) for
preview/download. Covered by `test_backend_proposal_diff_returns_unified_diff`.

(superseded — original finding kept for history:)

**Where:** `RunnerBackend.proposal()` returns the proposal payload (changed paths + per-file
snapshot refs) but **not** the unified diff. The diff text is only returned by `result()`, which
is populated at run end — so a parked multi-turn session has a proposal but no API-served diff.
Found while building the R2 diff panel.

**Hurt:** To show a live diff while the session is still open, Studio reads
`run_dir / "diff.patch"` directly, coupling the app to the run-directory layout instead of going
through a token-scoped API.

**Worked around:** `StudioServer.proposal()` merges `backend.proposal(...)` with the diff text
read from the run dir. Covered by `test_agent_write_is_staged_then_applied`.

**Proposed core fix:** include the unified diff in `proposal()` (or add a `proposal_diff()` /
`/proposal/diff` endpoint), so integrators never read run artifacts off disk.

### DX-5 🟢 status.json write race fails the run on Windows (real bug)
**Where:** `recorder.py::StatusJsonSink` wrote `status.json` with an inline `tmp.replace(dst)`,
and a sink raising propagates out of `EventBus.emit` and **fails the run**. Found when the Studio
multi-turn test flaked: polling `status()` while the run rewrote `status.json` hit
`[WinError 5] Access is denied` — on Windows `os.replace` fails while another handle holds the
destination open (a concurrent reader). So any UI polling status on Windows could intermittently
kill a run.

**Fixed (core, not a workaround):**
1. `core/_util.py::write_json_atomic` now retries the replace on `PermissionError` (Windows
   reader race; POSIX never hits the retry). 
2. `StatusJsonSink` uses `write_json_atomic` and treats the status projection as **best-effort** —
   a transient write failure is logged and skipped, never failing the run (a later event rewrites
   the full state). Verified by hammering the Studio multi-turn path (0 failures where it
   previously flaked ~1 in 4).

### R3 (HITL) — no new core gap
The `hitl.request` tool + hosted-task surface (`task.started` carrying `task_id`/`prompt`/
`choices`, `report_task_result` to resume) was sufficient to build the approval gate end to end.
Studio binds `hitl.request`, renders a gate card from `task.started`, and answers via
`POST /api/hitl` → `report_task_result`. One minor naming snag: hosted tasks key their id as
`task_id` while background **jobs** use `job_id` — worth knowing but not worth a change.

### R4 (shell + background jobs) — no new core gap
Shell + jobs were buildable from the existing surface: bind `shell.exec` with
`runtime.shell.approval_mode="auto-approve"` and a `ToolScope(command_deny_prefixes=…)` for the
destructive-command gate (enforced at the scope layer → `permission.denied` /
`error_code="tool_scope_denied"` before execution), and the backend's `jobs()` / `job_logs()` for
the background-jobs panel. Minor narration learning: the shell tool's `args_preview` carries the
command under `command_preview` (not `command`), so `narration._TARGET_KEYS` includes it — a small
key inconsistency, not worth a core change.

### R5 (web tools) — no new core gap
Web tools dropped in by booting the reference `WebGatewayBackend(FakeWebProvider())` on a loopback
port (shared signing secret) and pointing `RunnerBackend(web_gateway_url=…)` at it; binding any
`web.*` tool makes the backend mint the web token automatically. Two narration learnings (both
correct behavior, no core change): web `args_preview` carries `query_preview` / `url_preview`
(not `query`/`url`), and the query is **redacted** in the public stream (a `{"redacted": True, …}`
dict), so `narration._target` now surfaces only plain-string args — a redacted query shows as
"Searching the web for" with no term, which is the right privacy outcome.

### R6 (settings window + live Agent-spec editing) — no new core gap
The runtime-config hot-swap surface was sufficient: Studio keeps an editable capability set,
builds the runtime config from it (`_runtime_config_for`), and on a settings change calls
`backend.replace_runtime_config(expected_version=current.config_version, …)` for each active run
(version auto-bumps; terminal/stale runs are skipped). The Settings page is a second small
window (`/settings`, opened via `studio settings`). `current_runtime_config` (no token, internal)
made reading the live version trivial. Optimistic versioning (expected_version) is the only sharp
edge — read the current version right before replacing.

### DX-6 🟡 Test suite can intermittently hang (now bounded + diagnosable)
**Symptom:** a backgrounded full-suite run occasionally appeared to stall forever — no output,
the pytest process alive but idle (≈6 CPU-seconds over minutes), requiring a manual kill.

**Investigation:** the core teardowns are already bounded (`_teardown_loop` joins the shared
asyncio-loop thread with `timeout=5`; conftest `serving()` joins HTTP server threads with
`timeout=10`). Single runs and several back-to-back repros all passed cleanly, so this is a **rare
timing race** in the threaded-HTTP / shared-loop / subprocess tests, not a deterministic deadlock
or a shipped-code bug. The "silently forever" part was the harness's block-buffered background
pipe: a hung pytest never flushes, so the run looks dead.

**Fix (test-infra):** `tests/conftest.py` arms `faulthandler.dump_traceback_later(240, exit=True)`
(disabled via `NAR_TEST_HANG_TIMEOUT_S=0`, cancelled on normal finish). A wedged run now dumps
every thread's stack — pinpointing the exact blocked line — and aborts, instead of hanging. So
the background can never stall indefinitely again, and the next occurrence is self-diagnosing.

<!-- Add new entries below as later rungs surface them. -->
