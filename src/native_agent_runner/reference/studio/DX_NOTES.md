# Studio DX notes

A running log of developer-experience gaps found while building Agent Studio against the
contracts + reference services alone. Each entry: what hurt, where, and the proposed core fix.
Building the app is the pressure test; this file is the yield.

## Status legend
- 🔴 open — gap confirmed, not yet addressed in core
- 🟡 worked-around — Studio papers over it locally; core fix still wanted
- 🟢 fixed — addressed in core/reference

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
