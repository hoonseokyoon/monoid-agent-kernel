# Studio DX notes

A running log of developer-experience gaps found while building Agent Studio against the
contracts + reference services alone. Each entry: what hurt, where, and the proposed core fix.
Building the app is the pressure test; this file is the yield.

## Status legend
- 🔴 open — gap confirmed, not yet addressed in core
- 🟡 worked-around — Studio papers over it locally; core fix still wanted
- 🟢 fixed — addressed in core/reference

---

### R11 🟢 Three low-effort surfacings — file viewer + markdown/code + OTel toggle
Engine capability that existed but wasn't reachable from the UI. All studio-side (plus one tiny
backend seam), no core changes.
- **File viewer:** clicking a workspace file opens its contents in a side panel (Shift/Ctrl-click
  still inserts the path). New `studio.read_file()` is path-guarded to the workspace root (rejects
  traversal/absolute), 256 KiB-capped, binary-refusing (NUL byte); served at `GET /api/file?path=`.
- **Markdown/code:** `renderMarkdown` now does fenced code blocks (language label + copy button,
  robust to an unclosed trailing fence mid-stream), inline code, headers, lists, blockquotes — all
  escaped. The copy button uses inline `onclick` so it survives the streaming `innerHTML` re-render.
  (No syntax *highlighting* — that needs a lib, against the zero-dep/offline ethos.)
- **OTel tracing toggle:** a settings switch installs a global OTLP/HTTP `TracerProvider`
  (`_ensure_otel_provider`, actionable error if the `opentelemetry` SDK/exporter extra is missing)
  and attaches `OtelEventSink` **per run** via a new `RunnerBackend.extra_event_sink_factories`
  seam — a factory tuple (not a shared instance) so each run gets its own span state, appended at
  loop-build time. The only core-adjacent change; the embedder seam for observability-without-a-dep.
  Verified live over the **real OTLP wire** (a local OTLP/HTTP receiver captured the
  `invoke_agent → chat → execute_tool` span tree); a real Jaeger consumes the same stream (docker
  was unavailable here to run its UI).

### R10 🟢 Studio reasoning UX — Thinking disclosure + summary toggle + reasoning-tokens meter
The studio surface for reasoning. The disclosure (collapsible "🧠 Thinking" panel, auto-collapse on
the first answer token) and the summary toggle (off/auto/detailed in the composer setup bar) shipped
with DX-13b. The remaining piece — the **reasoning-tokens meter** — surfaces the priced-but-invisible
"thinking" token share: `_accumulate_usage` already sums `reasoning_tokens` into `total_usage`, so the
loop now includes it in `metrics.updated` (only when reported) and the meter shows `🧠N` next to
`↑in ↓out · total`. Two-line plumbing; data was already there. (Cache tokens are similarly available
in `total_usage` if a meter for those is wanted later.)

### DX-13b 🟢 Reasoning wasn't *shown* to the user (display, on top of DX-13a's round-trip) — FIXED
With DX-13a carrying reasoning for correctness, the studio still showed nothing of the model's
thinking. OpenAI doesn't expose raw chain-of-thought, only a model-written **summary** (request via
`reasoning.summary`); we modeled the `summary` field but never surfaced its text. Built the same
3-layer streaming path as DX-8 (token streaming), kept strictly separate from the round-trip:
- **Core seam:** new presentational stream chunk `ReasoningDelta` (a *summary* fragment, distinct
  from `TextDelta`) + event `model.reasoning.delta` (mirrors `model.output.delta`).
  `assemble_streamed_turn` ignores it — the assembled turn (answer + the round-trippable
  `TurnComplete.reasoning`) is unchanged, so display can't corrupt correctness.
- **OpenAI:** `astream_turn` maps `response.reasoning_summary_text.delta` → `ReasoningDelta`. The
  loop's `_acall_model_emitting_deltas` emits `model.reasoning.delta` per chunk (immediate-stop
  interrupt applies for free). Also wired through the gateway (`_chunk_from_event` /
  `_frame_from_chunk`) so the container path isn't silently dropping it.
- **Studio:** a `summary` setting (off/auto/detailed, default `auto`) in the composer setup bar,
  flowing to `ReasoningConfig.summary`; a collapsible dim "🧠 Thinking" panel that streams the
  summary and **auto-collapses on the first answer token**; subagent cards get a dim reasoning line;
  `model.reasoning.delta` is trace-skipped (noise). Display is a setting (privacy: summaries can leak;
  redacted/encrypted blocks are never shown raw — only the round-trip carries them).
- Verified live (real OpenAI gpt-5.5, effort high, summary auto, no tools so the turn is
  reasoning+answer): `response.reasoning_summary_text.delta` fires (70 summary fragments / 384 chars,
  e.g. *"**Deciding on the response format**…"*) → `model.reasoning.delta` events stream ahead of the
  118 answer-token `model.output.delta` events; the turn settles. (A tool-only turn streams no summary
  text — expected.)

### DX-13a 🟢 Reasoning artifacts weren't round-tripped (latent 400 on reasoning models) — FIXED
Pressure-testing a reasoning model (gpt-5.x via the Responses API) over a multi-turn tool loop. The
model emits **reasoning items** (`{type:"reasoning", id:"rs_…", encrypted_content:"…"}`) alongside
its `function_call`s. Under the **by-value** request shape Studio uses (full `messages` resent each
turn — `ModelRequest.messages` overrides `previous_turn_handle`), OpenAI *requires* those reasoning
items to be sent back, paired with and in the original order relative to their function_calls, for
everything since the last user message — or it hard-`400`s (`Item 'rs_…' without its required
following item`) plus a ~3% tool-accuracy loss. We were **dropping reasoning entirely**: the parser
read only function_call/message/output_text, the streaming path read no reasoning, the assistant
by-value message carried only content+tool_calls, and `_payload` set neither `store` nor `include`.
- **Core seam (neutral):** `ModelTurn.reasoning` + `TurnComplete.reasoning` (+ `assemble_streamed_turn`
  threads it) carry provider-native reasoning items opaquely; adapters with no reasoning leave it
  empty, so gateway/fake are untouched.
- **OpenAI (deep, ZDR):** capture the verbatim ordered `reasoning`/`function_call`/`message`
  subsequence (parse + stream, off `response.completed`); `_payload` sets `store=false` +
  `include=["reasoning.encrypted_content"]` so reasoning travels by-value (no `previous_response_id`
  reliance); on the next turn re-inject the captured items **verbatim** (suppressing the
  reconstructed function_calls) only within the active window (since the last user message) and only
  when the message's `(provider, model)` tag matches the current model — all-or-nothing per window,
  since the model can hot-swap mid-loop. Historical reasoning is dropped (tolerated).
- **Loop:** the assistant by-value message gains a `{provider, model, items}` reasoning block when the
  adapter reports `provider_name=="openai"`; it round-trips through the checkpoint unchanged (arbitrary
  message keys already persist).
- **Gotcha found live:** echoing output items back as *input* `400`s with
  `Unknown parameter: input[..].status` — the output-only `status` field must be stripped on capture.
- Verified live (real OpenAI, gpt-5.5, effort high): a multi-turn tool loop (incl. a **parallel**
  1-reasoning→2-calls turn) settles with **zero 400** and the reasoning carried on the continuation;
  a second user turn continues cleanly (historical drop); every payload is ZDR (`store=false`,
  `include=encrypted_content`, no `previous_response_id`); and the **streaming** path delivers
  reasoning+`encrypted_content` on `response.completed`. Display of reasoning (panel/stream) is a
  separate later item (DX-13b/R10); other providers slot into the same seam with their own policy
  (Anthropic/Gemini = preserve, DeepSeek = strip).

### DX-12 🟢 Session history doesn't survive a restart (no backend run-listing API) — FIXED
Building R8 (chat history). Studio first listed only the chats it started this server run (in-memory
`_sessions`); a history that survived a **restart** couldn't be rebuilt — the run dirs persist under
`run_root`, but the backend had no "list my runs" API and each run's events were gated behind a
per-run token held only in memory, plus `events()`/`status()` required an in-memory record.
- **Backend:** `list_runs(tenant_id, *, user_id, limit)` — a trusted-host scan of run_root (like
  `recover_runs`) reading each `run.json` (now carrying `created_at` + `title`), taking status from
  a live record when present else status.json, flagging `recoverable`, and **minting a read token
  per entry** (mirrors submit_run returning one). `events()`/`status()` gained a **no-record path**
  via `_authorized_run_dir`: a signed run token authorizes reading `run_root/<run_id>` straight from
  disk (path-guarded), so a historical run with no record is still readable.
- **Studio:** `sessions()` now sources `backend.list_runs` (restart-surviving), stores the read
  tokens server-side (never sent to the browser), and overlays very-recent in-memory runs whose
  run.json isn't on disk yet. The UI is unchanged.
- Verified live across a real "restart" (a second studio process over the same run_root): it listed
  the prior chat (`recoverable=True`) and replayed its transcript with no in-memory record, and no
  read token leaked to the browser. Resuming a parked historical run (vs read-only replay) rides the
  existing `recover_runs()` and is left for when needed.

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

### R9 (per-file proposal approval + package export) — studio bypassed an existing core gate; one remote-embedder contract gap
**What shipped:** the studio now exposes two capabilities the core+backend already had but the
reference app never reached:
- **Per-file approval.** `studio.apply(run_id, approved_paths=…)` forwards a chosen subset to
  `approve_proposal`; `apply_package` writes only those files and returns the rest as
  `skipped_paths`. The UI renders the changed-path list as checkboxes (+ select-all), so a human
  can land a subset and leave the rest staged. Empty subset still = approve-all (legacy behavior).
- **Package export.** `studio.export_package` → `export_proposal_package` builds the self-verifying
  tar; a new `POST /api/export-package` route streams it with `Content-Disposition`, and the UI
  triggers a browser download. Live-verified end-to-end: partial apply skips the unselected file on
  disk; export returns a real `ustar` tar with the package-hash filename.

**DX finding (no core change needed):** the gate was never missing — the studio's `apply()` simply
hard-coded approve-all and dropped the `approved_paths` parameter. The core contract is already
strict (unknown paths and approved∩rejected overlap both raise in `create_approval`), so wiring the
subset through was pure plumbing. The lesson is that a reference app can silently *hide* a core
safety feature by not threading one kwarg — worth auditing the other backend kwargs the studio
defaults away.

**🟠 Contract gap (remote embedders):** `export_proposal_package` returns only a **server-local
filesystem path** (`run_dir/proposal.tar`), not bytes. The studio gets away with reading it off disk
because it is co-located with the backend, but that violates the same "an embedder never reads the
run dir off disk" principle the `proposal()` API was built around — a *remote* embedder has no
token-scoped way to fetch the package bytes. Closing it cleanly would mean either a backend method
that returns the tar bytes (or a streaming handle), or a generic token-scoped run-artifact GET.
Deferred (reference studio is co-located), but it's a real seam to fix before the backend is used
out-of-process.

### R12 (resume a parked session after a restart) — the read/write asymmetry was the real core gap
**Symptom (DX blocker):** the studio could *list* and *replay* a past chat after a restart (DX-12),
but typing a follow-up into a parked-but-not-in-memory session threw `KeyError: unknown run`. The
session looked alive in the sidebar yet was unwritable — a dead end for "continue an old chat".

**Root cause (Core/Contract, not UI):** the backend had an asymmetry between read and write paths:
- **Reads** (`events`, `proposal`, `status` via `_authorized_run_dir`) authorize on the *signed run
  token alone* and read straight from `run_root` — they work with no in-memory record.
- **Writes** (`send_message` via `_authorize_run` → `_record`) require an in-memory record and
  `KeyError` for any run not currently hosted (e.g. after a restart).

`recover_runs()` existed but is the wrong tool here: it is a process-global, **no-token** operator
primitive that scans and resumes *every* parked run at startup. There was no token-scoped way for a
specific caller to resume *its own* run on demand — the exact missing contract piece.

**Fix (one new core primitive + thin studio wiring):**
- `RunnerBackend.resume_run(run_id, token)` — the token-scoped, single-run analog of
  `recover_runs()`. Verifies the run token (the capability), checks its claims against the persisted
  `run.json` identity (no record to check against yet), then materializes the record from the latest
  non-terminal checkpoint via the existing `_attempt_resume`. **Idempotent**: an already-live run
  returns `resumed=False`. Rejects terminal/failed/unknown runs with the usual `ValueError`/`KeyError`.
- `studio.continue_chat` now catches the `KeyError` from `send_message`, calls `resume_run`, and
  retries the send once — so "continue an old chat" just works. The resume reconstructs the full
  conversation from the checkpoint (prior assistant turns included), then threads the new message.
- UI: parked runs surface a `⟳ resume` tag in the session list; the composer was never disabled, so
  loading one and sending already routes through the resume path. Live-verified over HTTP: park →
  evict record (simulated restart) → `/api/sessions` reports `recoverable: true` → `POST /api/chat`
  resumes and threads the follow-up as a real second turn.

**Takeaway:** the principled fix wasn't "make `send_message` auto-recover" (that conflates a heavy,
fail-able operation with a hot path); it was to make the *write* surface symmetric with the *read*
surface by giving callers a token-scoped resume primitive. The studio orchestrates the two calls,
keeping the resume observable and `send_message` single-responsibility.

<!-- Add new entries below as later rungs surface them. -->
