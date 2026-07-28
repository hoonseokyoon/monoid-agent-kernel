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
- `artifacts/jobs/<job_id>/`: background job status (`job.json`) and `stdout.log` / `stderr.log`
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
- **File contents** in tool arguments and results (`content`, `old`, `new`, `old_text`, `new_text`).
- **Whole values of any length.** Previews are capped by a **byte** budget, so the cap does not
  depend on the script the text is written in.

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
  arguments. `PermissionPolicy.redact_patterns` will not help: it is a *workspace path* glob list,
  consulted only for fields that name a path — `path`, `root`, `cwd`, and any `*_path` argument
  such as `fs.move`'s `source_path` / `destination_path`. It was previously consulted for the first
  three only, which meant one `fs.move` published `paths: ["[redacted-path]"]` alongside
  `args_preview.source_path: "secrets/creds.txt"` on the same event.
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
  question, **uncapped**. Model-authored prose on a public surface, and the one route on this list
  that is neither previewed nor digested.

Studio adds `studio.chat.jsonl` inside each Studio run directory as the browser-facing chat
projection. The Studio UI restores user, assistant, and error messages from
`/api/chat-transcript`, then replays `events.jsonl` for trace and activity panels.
`transcript.jsonl` remains the private model-call log.

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
`gen_ai.tool.name`, token usage). Wire it in with one line:

```python
from monoid_agent_kernel import AgentLoop
from monoid_agent_kernel.observability.otel import OtelEventSink

loop = AgentLoop.from_config(spec, adapter, config, event_sinks=(OtelEventSink(),))
```

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
