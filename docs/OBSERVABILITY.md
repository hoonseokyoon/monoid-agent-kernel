# Outputs, event sinks, and observability

Every run emits a structured event stream and durable artifacts, and can mirror
that stream to OpenTelemetry. This is the reference for the run-directory artifact
set, custom event sinks, OTel tracing, live streaming, and metrics.

The event stream carries metadata and bounded previews rather than content — with a
short list of deliberate exceptions, stated under [What `events.jsonl` does and does
not carry](#what-eventsjsonl-does-and-does-not-carry) rather than left as an absolute
claim that the shipped defaults contradict.

## Outputs

Each run writes:

- `events.jsonl`: public redacted event stream
- `transcript.jsonl`: private debug/replay transcript with full tool payloads
- `status.json`: latest run lifecycle projection for polling (`state` plus `terminal`)
- `metrics.json`: final counters and timing
- `manifest.json`: run contract, agent config metadata, binding-aware tool surface, workspace backend
- `workspace.base.json`: base snapshot used for proposal comparison
- `workspace.index.json`: context/index artifact
- `diff.patch`: proposed or applied workspace diff
- `proposal.json`: proposed output snapshot metadata
- `proposal/files/`: materialized changed-file snapshots
- `artifacts/jobs/<job_id>/`: background job status (`job.json`) and `stdout.log` / `stderr.log`.
  **Written raw, published projected.** The file keeps the exact `command`, `cwd` and
  `changed_paths` — it is the run's own record and `JOB_SCHEMA` requires them — but it does *not*
  get `task.json`'s "private by location" exemption below, because two surfaces serve it over HTTP
  (`/v1/runs/<id>/jobs` and Studio's `/api/jobs`). Every reader goes through one projection
  (`public_view.public_job_artifact`, schema `monoid.public-background-job.v1`): `command` is
  dropped, `command_preview` stays bounded, `cwd` is
  previewed and `changed_paths` redacted against `redact_patterns`. That is five readers with one
  answer; it was five with three, and the raw ones meant **backgrounding a command was enough to
  route around `redact_patterns`** — the same `cwd` came out `{"redacted": true}` on
  `shell.exec.started` and as the path on `monoid jobs --json`. The log files themselves are not
  projected: they are process output, served by `/jobs/<id>/logs` to a run-token holder.
- `artifacts/tasks/<task_id>/task.json`: hosted-task record (hitl, subagent, capability, tool
  approval). **Private by location, like `transcript.jsonl`** — it is not served over HTTP and is
  not in the export allowlist, so it keeps values the event stream drops. A **tool-approval** record
  carries a bounded `arguments_preview` and not the raw arguments; the raw copy lives in the run's
  checkpoint, which is deleted when the run completes. The other kinds (hitl, subagent, capability)
  have no `arguments_preview` at all, and their `result` is written **raw** — that is what "private
  by location" buys, and it is why this file is not a public surface.

Proposed file contents are exposed only through the run directory snapshot or
run-token protected backend proposal APIs.

## What `events.jsonl` does and does not carry

Treat this as the contract, not the summary above it. `EventBus.emit` fans every event out to every
registered sink with **no level filtering**, so `level="debug"` gates nothing: whatever is listed
here is what a redacting sink, an OTel exporter, and `monoid watch --json` all see.

Not carried:

- **Model-authored settled text.** `turn.settled` and `run.finished` carry `final_text_digest` and
  `final_text_len`. Entitled readers join the text back from `transcript.jsonl`; the run result
  itself is unaffected. Kernel-authored text (for example `Stopped after reaching max steps.`) stays
  inline, because digesting it would cost an operator the line explaining why a run stopped and buy
  no privacy.
- **File contents** in tool arguments and results (`content`, `old`, `new`, `old_text`, `new_text`)
  — **on the trace surface**. The approval surface deliberately shows them, bounded; see "Carried,
  deliberately" below. This list said "file contents" unqualified while the entry 45 lines down said
  the opposite, and the preamble calls this list the contract.
- **Whole values of any length**, for every model-authored *value* and every mapping *key* that
  reaches a preview builder: those are capped by a **byte** budget, so the cap does not depend on
  the script the text is written in. Several routes bypass the builders entirely and are listed
  under "Carried, deliberately" below — read that list rather than counting exceptions here. An
  earlier revision of this line said "three exceptions" and was wrong twice over: it omitted
  `model.output.delta`, which the very next section describes as carrying raw model text, and
  `task.started.data.choices`.

Carried, deliberately:

- **`model.output.delta` and `model.reasoning.delta` carry raw model text**, and Studio enables them
  whenever the optional `httpx` extra is installed. These are durable events, not a live-only side
  channel, so the assembled answer is reconstructible from the file. **Do not use a grep for a known
  string to decide whether the answer is present**: the text is split at whatever boundaries the
  provider streamed, so a substring search finds it when the answer happened to arrive in one chunk
  and misses it when it did not. Absence of a grep hit is not absence of the content — concatenate
  `data.text` across the run's `model.output.delta` records instead. Disable with
  `MONOID_OUTPUT_DELTAS=0` (whole deployment, subagents included),
  `monoid studio serve --no-output-deltas` (the flag sits on the `serve` / `app` / `doctor`
  subcommands, not on the `studio` group), or `StudioConfig(stream_output_deltas=False)`. The
  completed answer still arrives. Two things do change, and the second is easy to be surprised by:
  live token rendering stops, and **mid-turn interruption becomes step-boundary interruption** —
  Stop waits for the in-flight model call to finish rather than aborting within a token. That
  coupling is not new and is not the switch's doing: the kernel only takes the streaming path when
  something is consuming deltas, and `emit_output_deltas` is `False` by default, so turning this off
  returns you to the kernel's own default rather than degrading past it. Studio turns it on, which
  is why the difference is visible there. The
  reason is not obvious from the emit site, so: `turn.settled` carries only `final_text_digest` on
  the durable stream either way, and `RunProjectionService.events` refills `final_text` from
  `transcript.jsonl` (`backend/content_hydration.py`) before the event reaches a reader. The Studio
  UI polls that projection, not `events.jsonl`, so its reducer still gets the text. `AgentLoop.astream`
  is unaffected, since it takes a different path.
- **Bounded previews of tool arguments** (`args_preview`, `arguments_preview`), including a preview
  of paths, commands, plan steps and artifact metadata.
- **Secret-named argument values, on the ordinary tool-call path.** The core does *not* guess at
  secrets from key names here; that heuristic was deliberately removed, and redaction beyond
  content fields is the integrating backend's job through the `EventSink` seam — see
  `examples/redacting_event_sink.py`. The approval record (`arguments_preview` on a `tool_approval`
  task) is the exception and *does* mask secret-named keys, because a human acts on it directly. So
  an `api_key` argument is masked on an `ask`-gated call and published verbatim on an `allow` call.
  If you want it masked on both, attach the example sink, or do not pass credentials as tool
  arguments. `PermissionPolicy.redact_patterns` will not help: it is a list of **gitignore-style
  wildcard patterns over workspace paths**, consulted only for fields that name a path — `path`,
  `root`, `cwd`, and any `*_path` argument such as `fs.move`'s `source_path` /
  `destination_path`. It was previously consulted for the first three only, which meant one
  `fs.move` published `paths: ["[redacted-path]"]` alongside
  `args_preview.source_path: "secrets/creds.txt"` on the same event.
  A pattern with no slash (`.env`, `*.key`) matches that name at any depth; `dir/**` covers the
  subtree and is anchored at the workspace root; `**/name` matches anywhere including the root;
  `dir/*` is direct children only. A leading `!` is rejected rather than read as negation, because
  negation makes the answer depend on pattern order and `merged` combines policies as a set; use
  `\!` for a literal leading exclamation mark. A leading `#` remains literal rather than becoming
  a gitignore comment. Each normalized path is matched independently; the
  full table is in [CLI.md](CLI.md#path-permissions). **Changed in v0.20** — `**` previously
  behaved as a single `*`, so `dir/**` covered exactly one level.
- The **approval** preview (`arguments_preview`) is bounded far more loosely than the trace preview,
  and it does **not** blank file-content fields. A person reads it to decide whether a call may run:
  a command cut mid-string hides the part that matters (with the model choosing where that part
  sits), and a card rendering `{"redacted": true}` where a file body should be asks someone to
  authorize a write they cannot inspect. So an `ask`-gated call publishes more on `task.started`
  than the same call would on `tool.call.started` — bounded, but readable. If that is not acceptable
  for a deployment, do not bind the tool to `authorization="ask"`, or attach a redacting
  `EventSink`.
- **Error messages and paths**, which can name workspace structure.
- **A subagent's answer** on `task.finished`.
- **Hosted-task prompts** (`task.started.data.prompt`) — the model-authored delegation brief or HITL
  question, **uncapped**. Model-authored prose on a public surface, neither previewed nor digested.
  It is not the only such route on this list: `model.output.delta` carries raw model text by
  construction, and an error message passes through `public_error_message`, which substitutes only
  when the text contains `PRIVATE KEY` and otherwise returns it whole.
- **The model-chosen `call_id`**, uncapped because it is a join key, not prose: `events.jsonl`,
  `task.json` and `approval_key` are matched on it, so a shortened copy would fail to correlate.
- **Hosted-task `choices`** (`task.started.data.choices`) — the options a HITL card offers, uncapped
  for the same reason as the prompt beside it: a person reads them to choose one, and a truncated
  option is one they cannot evaluate. Model-authored via `hitl.request`, whose schema bounds neither
  the number of choices nor their length.

Studio adds `studio.chat.jsonl` inside each Studio run directory as the browser-facing chat
projection. The Studio UI restores user, assistant, and error messages from
`/api/chat-transcript`, then replays `events.jsonl` for trace and activity panels.
`transcript.jsonl` remains the private model-call log.

The `/api/chat-transcript` response is `studio.chat.v2`. It requires `event_log_error`, which is
empty after a complete read and carries the failure reason when Studio returns only the committed
prefix of a corrupt `events.jsonl`. This member is the wire-format change from v1.

**`studio.chat.jsonl` carries whole model answers and whole user prompts**, and is content-bearing
and served over HTTP. It is not the only one: `RunProjectionService.events` hydrates `final_text`
out of `transcript.jsonl` before returning an events page, and `diagnostics()` returns the whole
`failure.json` — both behind a run token, which `studio.chat.jsonl`'s route is not. It has to be: a chat UI
that cannot re-render the conversation is not a chat UI, and the text is joined back out of
`transcript.jsonl` by the same hydration seam the projection uses. Three consequences worth stating
rather than leaving to be discovered:

- It is **not** covered by `MONOID_OUTPUT_DELTAS=0`. That switch stops model text reaching
  `events.jsonl`; with it engaged, the assembled answer still lands here. The switch bounds the
  *durable event stream*, not the run directory.
- The read path **writes**. Deleting the file and requesting `/api/chat-transcript` regenerates it
  from `transcript.jsonl`, so removing it does not remove the content.
- Treat it as **private by location**, like `artifacts/tasks/<id>/task.json` — with the difference
  that Studio serves it, so "do not expose the run directory" is not by itself sufficient. Studio is
  reference code and binds `127.0.0.1` by default; a deployment that fronts it needs to gate this
  route the way it gates the run directory.

## Event Sinks

Programmatic callers can pass sinks to
`AgentLoop(..., runtime_config_provider=provider, event_sinks=(...))`.
CLI callers can load sinks with:

```bash
monoid run \
  --workspace . \
  --instruction "Inspect this workspace." \
  --runtime-config-file examples/runtime-config.json \
  --event-sink-module ./my_sink.py:make_sink
```

The function must return an object with `emit(event)` and `close()` methods, or
an iterable of those objects.

`examples/redacting_event_sink.py` is a ready-to-copy sink that masks
secret-looking values before forwarding — the recommended place to add secret
redaction now that the core no longer guesses at secrets:

```bash
monoid run \
  --workspace . \
  --instruction "Inspect this workspace." \
  --runtime-config-file examples/runtime-config.json \
  --llm-gateway-url http://127.0.0.1:8080/internal/llm/turns \
  --event-sink-module examples/redacting_event_sink.py:make_sink
```

## OpenTelemetry tracing

`OtelEventSink` is an event sink that turns the run's
`run → model.turn → tool.call` event tree into a GenAI-semantic-convention span tree:

```
invoke_agent
├── chat {model}          (one span per model turn)
└── execute_tool {tool}   (one span per tool call)
```

`chat` and `execute_tool` are siblings under `invoke_agent` (linked by a `turn_id` attribute,
not nested), and spans carry GenAI attributes (`gen_ai.operation.name`, `gen_ai.request.model`,
`gen_ai.tool.name`, token usage). The zero-argument form preserves this metadata-only behavior:

```python
from monoid_agent_kernel import AgentLoop
from monoid_agent_kernel.observability.otel import OtelEventSink

loop = AgentLoop.from_config(spec, adapter, config, event_sinks=(OtelEventSink(),))
```

W9 adds three controls to the same preset:

- `parent_context` accepts an `InvocationContext` carrying W3C `traceparent` / `tracestate`, or an
  OpenTelemetry `Context`. The `invoke_agent` span becomes its child. Invalid W3C headers are
  ignored, and omitting the argument preserves OpenTelemetry's ambient-current-context behavior.
- `span_mode="agent"` is the default and retains the tree above. `span_mode="model_call"` is for a
  standalone `ModelCallRunner`: the observer emits exactly one `chat` span per settled call and its
  event-sink facet is inactive. The modes are mutually exclusive, so the preset never creates two
  inference spans for one call.
- `capture_policy` governs model-I/O attributes. Its OTel-specific default is
  `CapturePolicy(mode="none")`, retaining metadata and withholding content-derived digests,
  lengths, and payloads. `digest`, `redacted`, and `full` are explicit opt-ins.

For an AgentLoop, register both facets from the same instance. The event facet owns chat span
timing and topology; the model-I/O facet receives content only after `CapturePolicy` has been
applied and enriches that already-open span:

```python
from monoid_agent_kernel import AgentLoop, CapturePolicy, InvocationContext
from monoid_agent_kernel.observability.otel import OtelEventSink

caller = InvocationContext(traceparent=incoming_traceparent, tracestate=incoming_tracestate)
otel = OtelEventSink(
    parent_context=caller,
    span_mode="agent",
    capture_policy=CapturePolicy(mode="digest"),
)
loop = AgentLoop.from_config(
    spec,
    adapter,
    config,
    invocation_context=caller,
    event_sinks=(otel,),
    model_io_subscriptions=(otel.model_io_subscription(),),
)
```

The applied mode is recorded as `monoid.model.capture.mode`. Digest and text-length maps use
`monoid.model.capture.digests` and `monoid.model.capture.lengths`; these describe the **raw** fields
and accompany `digest`, successful `redacted`, and `full` captures. A digest of short or
low-entropy text can be matched against guessed inputs, so digest mode is correlation metadata,
not an anonymization guarantee. Successful redaction records the applied policy's digest in
`monoid.model.capture.redaction_digest`. Redaction failure records
`monoid.model.capture.downgraded_from="redacted"` and emits raw-field digest metadata without the
payload.

Redacted or full content is a JSON string in `monoid.model.capture.content`. This is an opaque,
Monoid-specific capture shape; the preset does not claim the exact OpenTelemetry
`gen_ai.system_instructions` / `gen_ai.input.messages` / `gen_ai.output.messages` content schema.
Treat those two modes as content-bearing exports and configure collector access and retention
accordingly. Capture never changes `events.jsonl`, and raw delta events are not mirrored into span
attributes.

Fresh sinks created for a restored activation lazily open `invoke_agent` on the first child event,
because recovery does not replay `run.started`. Model receipts then update the runtime provider and
model on the open chat span. Telemetry and serialization failures are contained inside the preset;
they do not fail the model call or agent run.

`OtelEventSink` depends only on `opentelemetry-api` (a no-op until your app installs an SDK +
exporter). To actually export spans, install the SDK and an OTLP exporter and configure a global
`TracerProvider`:

```bash
pip install "monoid-agent-kernel[otel-export]"
```

[`examples/otel_tracing.py`](../examples/otel_tracing.py) is a runnable, offline demo: it prints the
span tree to the console (via a local `ConsoleSpanExporter`, no collector) for a scripted run.

## Live streaming

Beyond the durable event sinks, `AgentLoop.astream(user_input)` returns a
`RunStream` — an async context manager + iterator that yields `AgentEvent` (orchestration)
interleaved with `ModelStreamChunk` (token deltas: `TextDelta` / `ReasoningDelta` /
`ToolCallDelta` / `TurnComplete`) when the adapter exposes `astream_turn`. Read `stream.result`
after the stream drains. Gateway token streaming uses Server-Sent Events and needs the
`[http-async]` extra. Studio uses complete one-shot gateway turns when that extra is absent and
enables live token deltas when it is installed.

Durable event subscriptions use `EventSubscription` over the append-only `events.jsonl` sequence.
They support page polling and SSE, sequence IDs, `Last-Event-ID` reconnects, heartbeat comments,
terminal final-event draining, recovered runs, and ancestor-authorized descendant streams. Request
`GET /v1/runs/{run_id}/events` with `Accept: text/event-stream`; a JSON request keeps the existing
inclusive `from_seq` pagination response. Studio uses the same cursor abstraction for its root SSE
feed and descendant event polling.

## Metrics

Each run writes `metrics.json` (and emits a `metrics.updated` event per turn) with
final counters and timing: `status`, `duration_s`, `tool_calls`, shell/background-job counters,
web-call counters, and token usage (`input_tokens`, `output_tokens`, `total_tokens`,
`reasoning_tokens`). See [Outputs](#outputs) for the full run-directory artifact set.
