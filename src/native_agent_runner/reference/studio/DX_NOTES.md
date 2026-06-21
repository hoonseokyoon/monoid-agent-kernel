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

### DX-3 🟡 Events carry no presentation-ready summary for a UI activity feed
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

### DX-4 🟡 No mid-run API for the proposal diff text
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

<!-- Add new entries below as later rungs (R3+) surface them. -->
