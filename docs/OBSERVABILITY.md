# Outputs, event sinks, and observability

Every run emits a structured event stream and durable artifacts, and can mirror
that stream to OpenTelemetry. This is the reference for the run-directory artifact
set, custom event sinks, OTel tracing, live streaming, and metrics.

The event stream carries metadata and bounded previews rather than content — bounded
per piece by the preview caps and per payload by one byte budget — with a short list
of deliberate exceptions, stated under [What `events.jsonl` does and does not
carry](#what-eventsjsonl-does-and-does-not-carry) rather than left as an absolute
claim that the shipped defaults contradict.

## Outputs

The run-directory artifact set is:

- `events.jsonl`: public redacted event stream
- `transcript.jsonl`: private debug/replay transcript with full tool payloads
- `model-content.jsonl`: optional private model-stream sidecar with output/reasoning segments and
  settled text (`monoid.model-content.v1`)
- `model_calls.jsonl`: optional private ledger of settled model calls, one record per call
  including failed ones — metadata, taxonomy, and the replay key, no content
  (`monoid.model-calls.v1`)
- `model_payloads.jsonl` + `model_payloads/`: optional private replay corpus — request preimages
  as verified reassembly recipes whose chunks deduplicate per tool definition, per message and per
  observation, and settled response bodies including provider reasoning
  (`monoid.model-payloads.v1`). `monoid run --replay-from RUN_DIR_OR_ID` reads it back as the model
  ([CLI.md](CLI.md)); a `monoid run --replay-from` run's ledger lines carry
  `attributes.replay_from` naming the source run ids — the ledger is opt-in, and a run driven
  programmatically through `ReplayModelAdapter` carries no such stamp
- `status.json`: latest run lifecycle projection for polling (`state` plus `terminal`). Every
  non-terminal park is visible here, including a cooperative pause (`state: "paused"`, projected
  from the `session.state.changed` event). While a run is parked on a recoverable turn failure
  the file carries the failure's full classification beside `error`/`error_code` —
  `provider_error_code`, `http_status`, `retryable`, `config_recoverable`, `provider_retried` —
  copied off the `turn.failed` event. (A replay miss parks exactly this way:
  `error_code: "replay_miss"` with the sub-reason in `provider_error_code` and
  `config_recoverable: true` — the remedy is a config or source fix, then resend.) A model turn starting clears the block (the new turn
  supersedes the dead one), and a non-failed terminal heals it; on a failed terminal the
  `run.failed` classification remains (minus `provider_retried`, a per-call fact the terminal
  vocabulary drops). Absent keys mean "no live failure to classify" — which is also what their
  absence on a pre-v0.21 artifact meant. The offline projection (`monoid status`, reading
  `events.jsonl`) answers with the same fields under the same rules. When a run dies without a
  live recorder — the reference backend's recovery gives it up for good (unrecoverable after
  `max_recover_attempts`, or corrupt durable state), or the backend records a driver failure
  (`record_run_failure`) — the terminal statement is written here too, by one shared writer:
  `state: "failed"`, `terminal: true`, the failure bundle's error pair, plus a quarantine
  marker naming the lane (`given_up_by_recovery` for recovery's give-ups,
  `recorded_by_run_failure` for the driver-failure lane) — because none of these paths emit a
  terminal event: without the artifact, every status reader kept reporting the dead run's
  last park. All three writers go through `run_state.write_failure_status_artifact` (a writer
  census in `tests/test_carriage_conformance.py` binds every `failure.json` writer to a
  terminal statement), which also seeds the schema-required watermark keys
  (`last_event_seq: 0`, `last_event_type: ""`) when it mints the artifact over a run that
  never wrote one, so the minted file stays `STATUS_SCHEMA`-valid. The offline projection
  honors the markers over the (necessarily park-ending) event log.
- `metrics.json`: final counters and timing. On a failed run it also records the failure's
  verdict beside the provider detail it already carried: `retryable` and `config_recoverable`
  join `provider_error_code` / `provider_http_status`, so an operator holding only this
  artifact can tell "resend after a config fix" from "this will fail the same way".
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
  `final_text_len`. Entitled readers join the text from `model-content.jsonl` and fall back to
  `transcript.jsonl` for old or partially written runs; the run result itself is unaffected.
  Kernel-authored text (for example `Stopped after reaching max steps.`) stays inline, because
  digesting it would cost an operator the line explaining why a run stopped and buy no privacy.
- **File contents** in tool arguments and results (`content`, `old`, `new`, `old_text`, `new_text`)
  — **on the trace surface**. The approval surface deliberately shows them, bounded; see "Carried,
  deliberately" below. This list said "file contents" unqualified while the entry 45 lines down said
  the opposite, and the preamble calls this list the contract.
- **Whole values of any length**, for every model-authored *value* and every mapping *key* that
  reaches a preview builder: those are capped by a **byte** budget, so the cap does not depend on
  the script the text is written in. The *payload* is bounded too: each traversal-built preview
  spends one 256 KiB budget across everything it appends — keys, values and truncation markers
  alike, counted in the widest spelling any *stream* writer uses (default separators, non-ASCII
  escaped, so the ceiling holds on the SSE surfaces that escape deliberately; the pretty-printed
  `status.json` and approval files add indentation on top of it) — so neither re-expanding a
  structure shared along many paths nor chunking a payload into
  cap-obeying pieces grows an event without bound, and the cut reports itself through the same
  `truncated_keys`/`truncated_items` vocabulary the per-container caps already use. The budget
  covers what the preview builders build, not the stream: routes that bypass the builders
  (hosted-task prompts and choices, `call_id`, validator feedback, error messages, a subagent's
  answer) are listed under "Carried, deliberately" below — read that list rather than counting
  exceptions here.
- **A value the previews cannot spell is named by its type, and that name is bounded too** — 64
  characters, in the `{"truncated": true, "type": …}` and `{"redacted": true, "type": …}` markers
  and in the `type` field of `run.failed` and `failure.json`. A class name is legal at any length,
  and two of these markers are published where the budget cannot refuse them, so the name was the
  one unbounded term in a bounded payload. The name is read off the type's own slot rather than
  asked for, so a class that answers `__name__` for itself neither widens this field nor raises
  while an event is being built.

Carried, deliberately:

- **Legacy `model.output.delta` and `model.reasoning.delta` events carry raw model text** when an
  integrating application explicitly sets `AgentLoop.emit_output_deltas=True`. These remain a
  compatibility surface for existing durable-event consumers. Provider chunk boundaries are
  arbitrary; reconstruct output by concatenating `data.text` in sequence order. The operator
  switch `MONOID_OUTPUT_DELTAS=0` disables this durable mirror for the root run and its subagents.
  A live presentation can use the model-stream observer channel and leave the durable mirror
  disabled. `AgentLoop.stream_model_calls=True` keeps provider streaming and token-boundary Stop
  behavior active without writing raw deltas into `events.jsonl`. `turn.settled` still carries only
  a digest, and `RunProjectionService.events` refills the text from `model-content.jsonl`, then
  `transcript.jsonl`, before an entitled reader receives it. `AgentLoop.astream` keeps its separate
  execution-owning stream contract and suppresses the legacy durable mirror for its live call.
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
  than the same call would on `tool.call.started` — bounded, but readable. The card's payload
  total has its own, far higher ceiling (1 MiB, against the trace's 256 KiB): a pathological
  argument map cannot put megabytes on `task.started`, and an ordinary card never meets the
  accountant. If that is not acceptable
  for a deployment, do not bind the tool to `authorization="ask"`, or attach a redacting
  `EventSink`.
- **Error messages and paths**, which can name workspace structure.
- **A subagent's answer** on `task.finished`.
- **Hosted-task prompts** (`task.started.data.prompt`) — the model-authored delegation brief or HITL
  question, **uncapped**. Model-authored prose on a public surface, neither previewed nor digested.
  Legacy `model.output.delta` carries raw model text when explicitly enabled, and an error message
  passes through `public_error_message`, which substitutes only when the text contains `PRIVATE KEY`
  and otherwise returns it whole.
- **The model-chosen `call_id`**, uncapped because it is a join key, not prose: `events.jsonl`,
  `task.json` and `approval_key` are matched on it, so a shortened copy would fail to correlate.
- **Hosted-task `choices`** (`task.started.data.choices`) — the options a HITL card offers, uncapped
  for the same reason as the prompt beside it: a person reads them to choose one, and a truncated
  option is one they cannot evaluate. Model-authored via `hitl.request`, whose schema bounds neither
  the number of choices nor their length.

Studio adds `studio.chat.jsonl` inside each Studio run directory as the browser-facing chat
projection. The Studio UI restores user, assistant, and error messages from
`/api/chat-transcript`, then replays `events.jsonl` for trace and activity panels.
`transcript.jsonl` remains the private model-call log. `model-content.jsonl` is the private streamed
content and settled-text sidecar.

On catch-up, the chat projection joins interrupted and non-retryable failed root turns to their
exact private sidecar snapshots and stores available partial output with a stable stream message
id. Failed partials remain in transcript history when a later normal turn becomes the latest live
snapshot. A durable `run.resumed(reason="studio-retry")` marker suppresses only the explicitly
reissued failed turn.

Studio receives provider output and reasoning through a separate passive model-stream channel.
The browser subscribes to one root-scoped `/api/model-stream` SSE connection that multiplexes the
root run and its descendants. Frames use `monoid.model-stream.live.v1` and a
`generation:sequence` cursor. Each root ring retains at most 1,024 frames and 512 KiB, and the
broker retains at most 64 root rings with least-recently-used eviction. A reconnect replays the
retained suffix. A generation change, missing sequence, or evicted prefix emits `reset`; Studio
flushes active in-process sidecar batches, hydrates root and descendant call snapshots from the
private stores, and resumes at the broker-provided baseline. The observer order guarantees that a
visible reset follows the corresponding sidecar push. UTF-8 channel offsets make the snapshot and
retained suffix merge idempotent even when they were observed at different instants. Snapshot I/O
accepts only regular, single-link sidecars with stable path/descriptor identity. An unsafe path or
failed coordinated flush returns HTTP 503, and Studio retries the same hydration cursor.

If the root ring is already gone, the reset baseline uses a root-bound idle cursor carrying the
broker's bounded eviction epoch. A reconnect with the current cursor waits without repeating
hydration until a new generation appears, while an older or cross-root `Last-Event-ID` receives the
reset it missed on the prior connection. A bounded acknowledgement table covers the root-ring
budget; forgotten entries reset conservatively to the current broker epoch.

This SSE subscription has no execution control. Closing a tab or losing the connection closes only
that subscriber. Provider generation, durable recording, Stop, interrupt, and cancellation remain
owned by the run. Live frames enter the chat and subagent activity projections directly and never
enter `events.jsonl`, the run event reducer, Trace, or raw trace export.

The `/api/chat-transcript` response is `studio.chat.v2`. It requires `event_log_error`, which is
empty after a complete read and carries the failure reason when Studio returns only the committed
prefix of a corrupt `events.jsonl`. This member is the wire-format change from v1.

**`studio.chat.jsonl` carries whole model answers and whole user prompts**, and is content-bearing
and served over HTTP. Other content-bearing routes include `RunProjectionService.events`, which
hydrates `final_text` from `model-content.jsonl` with a `transcript.jsonl` fallback before returning
an events page, and `diagnostics()`, which returns the whole
`failure.json` — both behind a run token, which `studio.chat.jsonl`'s route is not. It has to be: a chat UI
that cannot re-render the conversation is not a chat UI, and the text is joined from
`model-content.jsonl` with a legacy `transcript.jsonl` fallback by the same hydration seam the
projection uses. Four consequences worth stating
rather than leaving to be discovered:

- A direct AgentLoop integration uses `MONOID_OUTPUT_DELTAS=0` to disable the legacy durable delta
  mirror. Studio resolves the same variable at its composition boundary as a broader content-egress
  gate: it disables the live SSE channel and `model-content.jsonl` sidecar together. The
  `--no-output-deltas` Studio flag has the same effect. Provider streaming stays selected when the
  async transport is available, preserving token-boundary Stop without exposing chunks.
- A settled answer still lands in the Studio chat projection when live/private incremental content
  is disabled because `transcript.jsonl` retains the completed model turn. Mid-turn and interrupted
  partial hydration require the model-content sidecar.
- The read path **writes**. Deleting the file and requesting `/api/chat-transcript` regenerates it
  from the private content artifacts, so removing it does not remove the content.
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
├── chat {model}            (one span per model turn)
│   └── model.attempt {i}   (one child per kernel dispatch, when there was more than one;
│                            the only node here the event stream alone cannot produce)
└── execute_tool {tool}     (one span per tool call)
```

`chat` and `execute_tool` are siblings under `invoke_agent` (linked by a `turn_id` attribute,
not nested), and spans carry GenAI attributes (`gen_ai.operation.name`, `gen_ai.request.model`,
`gen_ai.provider.name`, `gen_ai.tool.name`, token usage). The zero-argument form preserves this
metadata-only behavior, minus the `model.attempt` children, which are receipt-derived and need
the model-I/O facet:

```python
from monoid_agent_kernel import AgentLoop
from monoid_agent_kernel.observability.otel import OtelEventSink

loop = AgentLoop.from_config(spec, adapter, config, event_sinks=(OtelEventSink(),))
```

**Provider attribution through the LLM gateway (v0.21).** Every surface that names a provider
resolves it the same way — what the answering adapter declares as `provider_name`, falling back to
`ModelConfig.provider` when it declares nothing (`providers/base.py:resolved_provider_name`).
`GatewayModelAdapter` declares the provider it relays (default `"openai"`; set it per deployment
with `monoid run --llm-gateway-provider`, `monoid backend serve --llm-gateway-provider`,
`RunnerBackend(llm_gateway_provider=...)`, or — for the reference Studio, whose embedder seam is
itself such a factory — `StudioConfig(llm_gateway_provider=...)`; `none` disables it), so a call
routed through the gateway is attributed to the **model that served it** rather than to the
transport it arrived over.

Four surfaces change together, deliberately, because one expression feeds all of them:
`ModelCallReceipt.provider_name` (previously `""` on this route); `gen_ai.provider.name` on the
receipt-derived `chat` span, which reads that receipt; the model-stream context's `provider`
(previously the config's string); and the `model_provider` field of the `run.started` event, which
is where the *event-driven* sink — the zero-argument quickstart above, with no
`model_io_subscriptions` — gets `gen_ai.provider.name` from. That last one is the reason this is
stated as a mechanism rather than as a list: while `run.started` carried the raw config, one
`OtelEventSink` class produced two different answers for one call depending on how it was wired.

The agreement between those four is scoped to activations that emit `run.started`. An
event-only sink attached to a **restored** run joins after that event was written, so it has no
`model_provider` to read and reports no provider or model for the resumed turns; the
receipt-driven configuration (`model_io_subscriptions`) has no such gap, because every call
publishes its own receipt. This predates the provider-attribution change and is unaffected by
it — a sink that needs provider attribution across a restore should be wired with subscriptions.

The transport is not lost — `receipt.model.provider` still carries `"gateway"`, and `manifest.json`
records the configured `model_provider` verbatim — so a dashboard that wants to group by hop groups
by those. Deployments whose gateway fronts a different upstream should set the flag accordingly;
leaving the default mislabels the spans in exactly the way it would mislabel the reasoning
round-trip.

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
attributes. Model-stream observers are a separate live/private content channel and do not add
content to OTel spans unless an integrating observer explicitly exports it.

**Per-attempt spans (W7-2).** When the kernel retry layer dispatched a call more than once, the
model-I/O facet synthesizes one `model.attempt {index}` INTERNAL child under that call's `chat`
span at settle — one per `receipt.attempt_log` entry, in both `span_mode`s. This rides the
subscription facet (the event stream carries no attempt data, deliberately), so the zero-argument
event-only quickstart shows neither the children nor the `monoid.model.attempts` count that
summarizes them: both are read off the receipt, and the public turn events carry no attempt count.
A single-dispatch call synthesizes no child either: the chat span *is* that attempt, and a child
would restate it at double the span volume. Children carry `monoid.model.attempt.*` attributes —
`index`, `elapsed_ms`, `backoff_ms`, the failure taxonomy, per-attempt `usage` as a JSON string,
`provider_retried`, `stream_committed` — and deliberately no `gen_ai.*`, because a GenAI-aware
backend aggregating usage or operation counts over those spans would double-count the parent;
capture content never propagates down, whatever the policy, since the attempt log is metadata by
construction. A failed dispatch gets `error.type` and ERROR status under the parent's own rule
(`model_call_aborted` is an interruption, not an error). Placement walks backward from the settle
instant: each child spans its measured `elapsed_ms`, preceded by its recorded `backoff_ms` gap;
entries read from ledger lines that predate `backoff_ms` pack edge to edge — durations and order
stay exact, the unknown gaps collapse. The instants combine a wall-clock anchor with monotonic
durations, the same stated limitation as the standalone `model_call` span.

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

`AgentLoop.astream(user_input)` is the execution-owning stream API. It returns a `RunStream`, an
async context manager and iterator that yields orchestration `AgentEvent` values interleaved with
every provider `ModelStreamChunk` variant (`TextDelta`, `ReasoningDelta`, `ToolCallDelta`, and
`TurnComplete`). Read `stream.result` after the stream drains.

Autonomous runs use `AgentLoop.stream_model_calls=True` to select provider streaming independently
from egress. `model_stream_observer_factories` creates passive observers for live presentation or
private persistence. Each observer receives output/reasoning fragments and a terminal outcome for
one provider call; tool-call fragments remain inside model-turn assembly. A fresh observer is
created for every activation and subagent, and observer failures remain isolated from the model
call. `model_content_file=True` adds a writer owned by the run recorder and persists
`stream_opened`, `stream_segment`, `stream_closed`, and `settled_text` records to the optional
private `model-content.jsonl` sidecar.

`model_calls_file=True` adds a second recorder-owned writer, for `model_calls.jsonl`. It is fed
from the model runner rather than from a `ModelIOObserver`, which is what lets it record failed
calls: a failure publishes its receipt to the subscriptions and re-raises without stamping it on
the exception, so a writer driven by the loop's return value would hold only successes. Writes are
shielded four ways — the ledger opens only if its own path is a single-link regular file, never a
symlink or hard link planted where a reopened run expects its artifact, an unencodable record costs
its own line, a write error disables the handle so a torn line cannot consume the next record, and
nothing raises into the call. The switch is independent of `stream_model_calls` and
`model_content_file`; see `docs/CONTRACTS.md` for what a record deliberately cannot say. From the
shipped shapes it is `monoid run --model-calls-file`, `monoid backend serve --model-calls-file`,
or `RunnerBackend(model_calls_file=True)` — the backend carries the boolean into the submitted run
and into the activation recovery rebuilds.

One operational note for all three verified-append sidecars — `model-content.jsonl`,
`model_calls.jsonl` and `model_payloads.jsonl`, named individually because a rule stated over "the
sidecars" is a rule that reaches whichever ones the reader counts. Each is opened as a single-link
regular file on purpose: appending mutates an inode, and a second name for it is somebody else's
file. A backup or restore that hardlink-deduplicates a run directory (`cp -al`,
`rsync --link-dest`) therefore disables all three on the next activation. The run itself is
unaffected — it completes and exits zero — and in that scenario `monoid validate` reports the
directory clean, because the linked file is a valid earlier copy and each artifact is optional.
(An arbitrary planted file is a different matter: validate then reports it as unparseable.)

So the refusal is announced. Each of these three writers logs one `WARNING` naming the artifact
whenever it enters the state where it records nothing more for the rest of the activation — a
refused open, a refused chunk, an append that may have torn its line, and, for
`model-content.jsonl`, a descriptor that no longer matches its path. The loggers are
`monoid_agent_kernel.recorder` (all three artifacts, when the recorder is the one that gives up)
and `monoid_agent_kernel.core.model_content` (when the content store is); the record carries
`monoid_run_id` and `monoid_artifact` as fields, so an aggregator can key on the run without every
default stderr rendering carrying identifiers, and the message text names the artifact's filename
and nothing else — a basename, never the directory holding it. `WARNING` because it means a run lost an artifact it was configured to produce, and because
Python's last-resort handler delivers exactly that level to stderr for an operator who configured
no logging. Once per writer per activation, and not necessarily on the run's own thread — the
content store's batch flush runs on a timer.

Two boundaries. A failure that costs a single record — an unencodable value, one line that would
not serialize — is not this, and stays quiet. And a third-party model-stream observer that
*raises* is a different contract: those failures are isolated from the run and logged at `debug`
by design, because one broken exporter must not produce a line per provider token.

A stale artifact is the reason it matters: the refused file is left as it was, so what a reader
finds there afterwards is whatever the link pointed at, not this run's record.
Content-addressed files under `model_payloads/` are safe to link; those three logs must
be copied. (`transcript.jsonl` and `events.jsonl` are unaffected — they do not go through the
verified opener.)

`model_payload_file=True` adds the third recorder-owned writer, for the replay corpus. It shares
the ledger's per-call lock and index, so a response record and its ledger line name the same call
by construction, and the two arms fail independently — a disk error in one file disables that
file only. Chunk files are created write-once through the same verified-file primitives as the
JSONL handles. The corpus, unlike the ledger, is content-classified; enable it with the same care
as `model_content_file`. From the shipped shapes it is `monoid run --model-payload-file`,
`monoid backend serve --model-payload-file`, or `RunnerBackend(model_payload_file=True)`, opt-in
in all three, and a subagent inherits whichever switches its parent ran under.
(`model_content_file` has the same three: `--model-content-file` on both commands and the field.)

Chunk-directory hygiene is a separate verb. `monoid gc RUN_DIR` reports what no record in the
corpus resolves — orphaned chunks from an interrupted write, dead `*.tmp` litter left by crashed
writers in other processes — and `monoid gc RUN_DIR --apply` deletes it, exiting non-zero for
refusals and failed deletions, zero for garbage merely found, and 2 with no report at all and
nothing swept when `--min-age-s` is unusable. Never run it beside a live
writer of the same run directory: the writer takes no cross-process lock and nothing on disk can
prove a writer dead, so liveness is the operator's knowledge, exactly as it is for
`monoid validate`. Two belts bound the damage of a broken contract without licensing one: an
entry whose age has not reached `--min-age-s` (default one day) is never touched, and the
write-once store refreshes a chunk's timestamps whenever a writer accepts one that already exists
— a resumed run re-deriving what it already holds is the common case — so recent use looks
recent. Neither belt is a guarantee: the refresh is best-effort (a touch the platform refuses is
swallowed, since the chunk is stored either way), and a writer that stalls past the gate between
storing a chunk and appending the line referencing it outlives both. An incremental archiver may
answer the refreshed timestamp with one redundant re-copy, a copy and not a correctness cost. The
gate itself must be a finite, non-negative number of seconds; anything else is refused before the
directory is read, because each unusable value breaks the belt a different way. A corpus that is absent or unreadable beside stored
chunks leaves them `unjudged` and untouched (a mutilated directory and a first-call crash whose
very first chunk was directory-sized leave the same state); damaged corpus *lines* are the
opposite case — no reader parses them, so what only they referenced is collected, and the report
names the line numbers (a count, plus the first hundred). Deletion never outruns the validator:
`monoid validate` reports the same issues after a sweep as before it.

`chunk_dir_state` names what the collector found where the chunk directory should be, and the
values call for different responses: `ok`; `absent` (a run that never offloaded); `unsafe`
(something is wearing the name that is not this run's directory — a symlink, a plain file, or a
Windows junction, which needs no privilege to create and `lstat`s as an ordinary directory, so
only its reparse tag tells it apart); `unreadable` (the platform declined, which on Windows is the
everyday shape of an antivirus pass, the search indexer, or a sync engine); `unprovable` (the
volume supplies no stable file ids — `st_ino` zero, as on FAT and some network redirectors — so
no deletion here could be re-proved and none is attempted, in either mode); and `swapped` (the
directory the gate approved was replaced before the pass finished, so every entry below it
describes whatever was standing there at the time). `corpus_state` is `ok`, `absent`, or
`unreadable`. Each entry carries a `classification`: `kept` (the keep-set names it), `orphan`
(chunk-shaped and unresolvable), `temp` (a write-once temporary over a chunk-name stem), `foreign`
(anything the writer demonstrably did not mint — never touched), or `unjudged` (chunk-shaped, but
the corpus needed to judge it was absent or unreadable — never touched).

`candidate_bytes` and `reclaimed_bytes` are both sizes, and only the second is a claim about the
volume: it counts a file only when the sweep removed the inode's last name, so an orphan inside a
hardlink-deduplicated archive reports `deleted` and reclaims nothing. Per entry, `reclaimed` says
which one that was, so the two totals can always be reconciled. `swept_at` is the instant every
`age_s` is measured against; the verb writes nothing to the run's `events.jsonl`, so the report it
prints is the only record it leaves.

Run one sweep at a time — two overlapping collectors leave the loser reporting a failure per entry
although the directory reached the state it asked for. Point the verb at a *run* directory, not at
a run root: a root is not a run, so it reports `absent` and exits 0, and a fleet sweep is a loop
over its children. Subagents keep their own run directories as siblings of their parent's, named
`<parent>.sub.<task>`, so a delegating run needs one sweep per member of the family. Note also
that `_resolve_run_dir` prefers a path that exists in the working directory over `--run-root`, as
it does for `validate` and `status`; the report echoes the absolute directory it swept, which is
the one to check when a bare run id is passed.

Gateway token streaming uses Server-Sent Events and needs the `[http-async]` extra. A presentation
layer can connect its chat UI to the live observer channel while `events.jsonl` retains
operation-level events. A reconnect hydrates completed content from the sidecar; retained v0.20 and
older runs use the transcript fallback.

Durable event subscriptions use `EventSubscription` over the append-only `events.jsonl` sequence.
They support page polling and SSE, sequence IDs, `Last-Event-ID` reconnects, heartbeat comments,
terminal final-event draining, recovered runs, and ancestor-authorized descendant streams. Request
`GET /v1/runs/{run_id}/events` with `Accept: text/event-stream`; a JSON request keeps the existing
inclusive `from_seq` pagination response. Studio uses the same cursor abstraction for its root SSE
feed and descendant event polling.

## Metrics

Each run writes `metrics.json` (and emits a `metrics.updated` event per turn) with
final counters and timing: `status`, `duration_s`, `tool_calls`, shell/background-job counters,
web-call counters, and token usage (`input_tokens`, `output_tokens`, `total_tokens`, plus the
priced sub-counts `cache_read_tokens`, `cache_creation_tokens`, `reasoning_tokens` and
`audio_tokens`). Each sub-count appears on the event only when the adapter reported one, so a
run that used no cache publishes no cache columns rather than a row of zeros — a dashboard
should treat an absent sub-count as "not reported", not as zero. `metrics.updated` is emitted
once per model call, including a call that failed *after* the provider billed for it (that arm
adds the billed tokens to the totals, so it publishes them too); a failure that reached no
provider adds nothing and emits nothing. See [Outputs](#outputs) for the full run-directory
artifact set.
