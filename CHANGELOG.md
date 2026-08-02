# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and this project is
pre-1.0 (`0.x`): minor versions may include breaking changes, which are called
out in commit messages and here.

## [Unreleased]

### Fixed — internal-review pass over the W5 surface (proof chain, ingress symmetry, classification)

- **A stream that ends without a terminal frame is no longer accepted unproven.** The
  applied-parameter checks lived only on the `turn_complete` frame — which is exactly the frame
  an older gateway never sends; its stream ends cleanly, `assemble_streamed_turn` synthesizes
  `stop_reason="stop"`, and the turn was accepted with every parameter unproven while the sync
  transport refused the same server. The drain now runs the same shared checks with an absent
  echo; traffic that configures neither knob keeps the old tolerance for frameless streams.
- **The capability question is answered per call, not per adapter.** `GatewayModelAdapter`'s
  `generation_support` / `structured_output_support` declarations are now callables taking the
  effective per-call config (the probes pass it through; a raising callable still reads
  `"none"`), and the reference gateway probes them under the same config the upstream call runs
  under (`_upstream_model_config`, built once for the adapter, the request, and the proof). A
  shared factory-built adapter could previously mint proof from its standing config for a call
  it enforced under a wire-supplied `"omit"` — the copied-back-proof defect one config-source
  hop later.
- **Direct-Python reasoning configs fail closed.** `validate_reasoning_config` now exists as
  the one rule source (the codec, `normalize_model_config`, and the runtime-config ingress all
  consume it; the ingress's hand-copied enum frozensets are deleted). A Python-constructed
  `ReasoningConfig(effort="turbo")` previously sailed through normalization to die mid-run as a
  provider 400 while the JSON codec rejected the same value at config time.
- **`effort="default"` survives the gateway wire.** It is the one reasoning field whose
  omission sentinel differs from the codec's reconstruction default (`"medium"`), so a client
  asking for provider-default reasoning silently got medium — only through a gateway. The
  client payload now carries it explicitly, like the off-default policy beside it; all other
  values keep their exact wire bytes.
- **A tool-call answer on the standalone surface is an outcome, not empty text.**
  `ValidatedCallStatus` gains `"tool_calls"`, short-circuiting before validation like refusal
  and truncation; previously the validators judged `final_text or ""`, burned a paid repair
  rewriting an answer the model never gave, and with zero validators the turn read `"ok"` with
  `final_text=None`.
- **Receipts survive exceptions.** `OutputValidatorError` (and any exception escaping
  `ValidatedCallRunner.acall`) now carries the completed calls' receipts as a `receipts`
  attribute; they were dropped with the raise, contradicting the audit-trail claim.
- **An unserializable request is a classified error on every path.** The gateway body encode
  (both transports) now raises non-retryable `gateway_bad_request` instead of leaking a raw
  `TypeError`, and the OpenAI payload build moved inside its classifier on both paths — one
  rule covering `output_schema`, `messages`, and observations.
- **A proof refusal ends the turn, not the run.** `ModelAdapterError` gains
  `config_recoverable`; `gateway_generation_not_applied` / `gateway_schema_not_applied` set it,
  `AgentLoop`'s classifier honors it (the error's own remedy is config the user fixes and
  resends), and the reference gateway's HTTP layer maps such errors to 422 rather than
  laundering them into a run-killing 502 across a chained hop.
- **Repair requests carry their conversation exactly one way.** `_repair_request` clears the
  carriage fields of the shapes it did not choose, so a repair's `request_digest` describes a
  request an adapter actually sends and a stale `instruction` can never be re-read beside the
  appended messages.
- The capability probes (`structured_output_support` / `generation_support`) and
  `AttemptDeltaConsumer` are exported from `contracts` — all three are documented contract
  surface a third-party gateway or streaming caller implements against, and previously
  required importing provider modules. OpenAI 4xx classification now names the provider's
  `param` (a provider-authored field path, not user content) so a schema-subset rejection is
  distinguishable from any other bad request.
- **`submit()` / `asubmit()` / `run_once()` surface a non-settling park honestly instead of
  crashing.** A turn that parked without settling — a *recoverable* turn failure (any
  provider 4xx, an exhausted retryable error, W5's proof refusals), an interrupt, or a pause —
  produces a `Suspension` with `turn=None`; the blocking facades asserted `turn is not None`
  and crashed with a message-less `AssertionError` (silently returning `None` under
  `python -O`), including on the fork-subagent path through `arun_once`. `submit`/`asubmit`
  now raise `TurnNotSettled` (`monoid_agent_kernel.errors`), carrying the suspension and its
  classification, with the session alive exactly as the `run_until_suspended` / `astream`
  halves always kept it; `LoopSession.submit` maps the park onto the FSM exactly like its
  pump half before re-raising. `run_once` is one-shot — its own `finally` closes the run —
  so it **returns** the promoted failed `AgentRunResult` instead of raising past the close
  that recorded it (which also restores the fork-subagent `subagent.failed` event and usage
  roll-up, and a clean CLI exit). The stale "every non-awaiting reason attaches a turn"
  claim is corrected in the docstrings and `Suspension` docs.
- **`close()` promotes an unrecovered `turn_failed` park to the terminal failure record.**
  Closing a run whose last park was a recoverable turn failure previously finalized from the
  per-turn reset state: `run.finished` claimed `status=completed` with no `failure.json`,
  and the completed-run cleanup then deleted the very checkpoints the park preserves for an
  operator-driven restore. `close()` now performs the same promotion `fail_recoverable`
  offers drivers explicitly (`failure.json` + `run.failed` + checkpoints kept); a park
  recovered by a later settle still closes `completed`. The `turn.failed` event schema
  gains `config_recoverable`, so `mak validate` accepts runs containing recoverable turn
  failures (the third twin of that key, after the Suspension and the checkpoint payload).
- `Suspension` and the `turn.failed` event carry `config_recoverable`, so a driver can
  distinguish "park for a config fix" from other non-retryable turn failures without
  hard-coding provider codes; the durable park payload round-trips it (absent on
  pre-v0.21 checkpoints reads `False`).
- The unserializable-request errors on both adapters are `config_recoverable` too — the
  same mistake reported by a gateway server is an HTTP 400, which was already
  turn-recoverable; and the OpenAI twin now names the defect (`unserializable_request`)
  instead of falling through as `unclassified_provider_error`.
- `run_output_validators` hands each validator its own copy of `parsed`, keeping
  `FinalOutputView` read-only in fact: one validator's in-place mutation was previously
  judged — and surfaced as a value — by the next. An exception already stamped with an inner
  validated call's `receipts` keeps the innermost stamp instead of being overwritten.
- **A boolean can no longer prove a sampling parameter.** The `generation_applied` echo was
  compared with `==`, and Python holds `True == 1` and `False == 0.0` — so a gateway
  answering JSON booleans proved exactly the most ordinary settings this block carries
  (`max_output_tokens=1`, `top_p=1`, `temperature=0`) under the default fail-closed policy.
  The comparison is now per key and non-coercive (a number is proven only by a number,
  through the same `is_finite_json_number` rule the rest of this wire reads with), while
  still accepting either JSON spelling of one number so a non-Python gateway echoing `1` for
  `1.0` is not falsely refused.
- **`output_schema` is not rewritten by ingress.** `normalize_model_request` substituted
  non-finite floats with `null` — right for model content, wrong for a control document
  promised verbatim: `{"enum": [NaN]}` became `{"enum": [null]}`, a different constraint the
  provider then enforced, and the strict serializer that exists to refuse the value never saw
  it. Strings and containers are still normalized; the value now reaches the boundary and is
  refused there.
- **The OpenAI adapter preflights the whole request body.** `_payload` embeds `output_schema`
  without serializing it, so its classifier saw nothing: a set or a cycle inside the schema
  failed later inside the SDK as an anonymous `unclassified_provider_error` with no
  `config_recoverable` (terminalizing the run for what the gateway twin reports recoverably),
  and a `NaN` was serialized to the JSON-invalid literal `NaN` and sent. The assembled payload
  is now strict-encoded (`allow_nan=False`) inside the classification boundary on both the
  blocking and the streaming path.
- **A request too deep to encode is classified, not raw.** `json.dumps` recurses, so a
  container nested past the interpreter limit raises `RecursionError` — a `RuntimeError`
  subclass, outside the `TypeError`/`ValueError` family both encoders caught — and nothing
  upstream refuses it first (`normalize_json_ingress` is iterative by design, and the
  512-level nesting cap guards the JSON *text* parsers, not a Python-constructed value). It
  escaped raw from the gateway encoder and reached the OpenAI adapter's outer handler as an
  anonymous `unclassified_provider_error`, terminalizing the run either way. Both encoders now
  answer it like any other unsendable request.
- **The unrecovered-park promotion survives a restart.** `close()` promotes an unrecovered
  `turn_failed` park from a session field, and `restore()` rebuilt the session without it — so
  a crash-and-recover of exactly the run the park exists for (a non-retryable configuration
  failure, recovered, left idle, then closed) finalized `completed`, wrote no `failure.json`,
  and let the completed-run cleanup delete the checkpoints the park preserves for an operator
  restore. `restore()` now rehydrates it from the checkpoint's `last_suspension` (only
  `reason="turn_failed"`; a later settle clears it at pump entry, exactly as in-process).
- **A conforming deeply-nested answer is still validated.** The per-validator copy of
  `FinalOutputView.parsed` used `deepcopy`, which recurses: the strict ingress accepts JSON
  nested to 512 levels — deep enough to exhaust the interpreter's default 1000-frame stack —
  so an answer the *parser accepted* raised `RecursionError` in the copy, outside any
  classification, before a single validator ran, and `ValidatedCallRunner.acall` leaked it raw.
  The copy now goes through the kernel's iterative JSON copier with both substitutions off, so
  it is isolation and nothing else.
- **An attempt that streams nothing still announces itself.** `AttemptDeltaConsumer` carried
  the attempt boundary on chunks alone, so an attempt that produced none delivered nothing at
  all — a consumer holding a rejected attempt's text was never told to drop it and rendered it
  beside an `ok` result. Every attempt now opens with an `AttemptStarted(attempt)` event
  (exported from `contracts`), delivered before that attempt's chunks and whether or not any
  arrive; the consumer's event type is `AttemptStarted | ModelStreamChunk`.
- **A refused turn still reports the tokens it burned.** The applied-parameters refusals fire
  *after* the gateway returned a complete, billed answer with its usage — the client simply
  refuses to trust that its parameters shaped it. The failed `ModelCallReceipt` reported zero
  tokens and the loop's accumulation runs only on the returned-turn path, so a paid call
  vanished from the metrics and from the cumulative token budget, which makes that budget a
  bound that does not hold. The refusal now carries the reported usage (`mark_provider_usage`,
  the twin of `mark_provider_retried`) on both transports; `ModelCallReceipt.with_error` reads
  it back, and `AgentLoop` accumulates it on the failure path. A failure that reports no usage
  still adds nothing. **The cost survives a gateway hop too**: when a reference gateway's own
  upstream refuses a billed turn, the error envelope carries `usage` (JSON body and SSE error
  frame alike, omitted when empty so an error raised before reaching a provider keeps its exact
  wire shape), all three client error readers stamp it back onto the reconstructed exception,
  and the gateway meters the call against the tenant before re-raising instead of losing it to
  the raise.
- **`run_once` no longer reports an interrupted run as a success.** It absorbs a non-settling
  park because `close()` turns it into the record that *is* the call's result — but `close()`
  promotes only `turn_failed`. An `interrupted` or `paused` park produced no record, so the run
  finalized `completed` with no settled answer (and the completed-run cleanup deleted the
  checkpoints the park had preserved). Only the park `close()` can promote is absorbed; the
  others surface as `TurnNotSettled` after the same close.

### Added — `GenerationConfig`: per-call sampling controls (kernel types)

- `ModelConfig` gains `generation: GenerationConfig` — `temperature` (0–2), `top_p` ((0, 1]),
  `max_output_tokens` (≥ 1), and `on_unsupported` (`"fail"` default / `"omit"`). Every value
  field defaults to `None`, meaning "delegate to the provider". The JSON codec and direct-Python
  normalization share one fail-closed rule source (`validate_generation_config`), so a range
  accepted from JSON can never diverge from the range accepted from a constructor. **This
  release adds the type and its ingress only; provider and gateway threading land next**, so a
  configured value has no request-body effect yet.
- **`ModelConfig.to_json` omits the `generation` key entirely when the block was never
  configured.** That single rule is the compatibility mechanism for three consumers at once: a
  generation-free config keeps its pre-existing `request_digest` (replay key), its
  `AgentRuntimeConfig.config_hash` (durable recovery compares this hash across versions), and
  its wire shape. Setting any generation value changes all three, deliberately — pinned by
  literal-hash tests captured on v0.20.1.
- `GenerationConfig` is exported from `monoid_agent_kernel.contracts` (and the package root)
  alongside its siblings `ModelConfig` / `ModelRetryConfig` / `ReasoningConfig`, so configuring
  generation does not require reaching into `core.spec`.

### Documentation — the output-validation and model-call contracts are now written down

- `docs/CONTRACTS.md` gains an Output Validation section documenting all six exported types
  (`OutputValidator`, `ValidationOutcome`, `FinalOutputView`, `OutputRetry`,
  `OutputValidatorBinding`, `OutputValidatorError`), the exception-classification contract, the
  loop's settle orchestration, and the standalone `ValidatedCallRunner` contract — these types
  had shipped in `contracts.py` since their introduction with no per-symbol contract entry.
- The Model Adapter section documents generation-parameter and output-schema delivery, the
  applied-echo enforcement, the fail-closed `structured_output_support` probe, the two digest
  stability rules (additive fields omitted when unset; canonicalization changes are
  domain-version changes), and the `RunLimits.max_output_tokens` vs
  `GenerationConfig.max_output_tokens` distinction. The stale `ModelRequest` field list gains
  the `messages` and `output_schema` fields.
- The LLM Gateway wire contract shows the `generation` / `output_schema` request keys and the
  applied-echo response keys; `docs/COMPATIBILITY.md` records the additive-key policy for
  `monoid.llm-turn.v1`, the fail-closed version-skew behavior, and the mixed-fleet caveat for
  configured generation blocks in `config_hash`.

### Added — output-schema delivery on the standalone path (ResponseContract)

- `ModelRequest.output_schema` carries a standard, provider-neutral JSON Schema for the final
  answer. The OpenAI adapter translates it to the Responses API `text.format` json_schema
  block **verbatim — never adjusted to the provider's strict subset** — so the request digest
  identifies exactly what the provider was asked to enforce (a schema the provider rejects is
  its own error through the taxonomy). The envelope is strict mode (`strict: true` — anything
  less is not *enforced* decoding and would make `schema_applied: true` a false proof), and
  OpenAI's strict subset has requirements of its own (`additionalProperties: false` on every
  object, every property required); a schema outside it 400s with the offending `param` named
  in the classified error. The digest follows the omission rule: schema-free
  requests keep their pre-existing replay key. `AgentLoop` does not set the field; this is the
  standalone/LLM-only path only.
- Adapters opt in with a `structured_output_support = "native"` declaration, read through a
  fail-closed probe (absence and unknown values mean `"none"`). The `monoid.llm-turn.v1` wire
  gains `output_schema` (request) and a `schema_applied` boolean echo (response body and
  terminal stream frame): the reference gateway threads the schema to its upstream adapter and
  echoes `True` only when that adapter declared native enforcement, so a forwarded-but-ignored
  schema reads `False`. The client refuses an unproven schema under the same
  `on_unsupported="fail"` knob that governs the sampling-parameter echo — one policy for "the
  transport cannot prove application", deliberately not two half-settable ones.
- `FinalOutputView.parsed` gives validators a best-effort structured view of the answer when a
  schema was requested, with `parsed_ok` saying whether there was a parse at all — `parsed is
  None` cannot, because a schema permitting a root `null` yields a valid parsed `None`, and a
  validator rejecting on `parsed is None` would fail a conforming answer and spend its repair
  budget on it. The parse goes through the kernel's strict JSON ingress
  rather than bare `json.loads`, which accepts Python's non-standard `NaN` / `Infinity`
  constants: a validator reading `parsed` — a schema validator will call `NaN` a number — would
  otherwise accept an answer that is not JSON at all. `ValidatedCallRunner` populates
  it; repair calls keep the schema riding while still stripping tools. Post-hoc validation
  remains the guarantee on every adapter — native delivery only reduces repairs, and adapters
  without support keep working unchanged.

### Added — `ValidatedCallRunner`: one validated model call, outside any loop

- A caller invoking `ModelCallRunner` directly — an LLM-only skill, a gateway, a batch driver —
  gets the same validate-and-re-prompt guarantee `AgentLoop` applies at its settle points:
  dispatch, run the registered `OutputValidator`s, and repair with at most `max_repair_calls`
  (default 1) explicit follow-up calls. Exhaustion is a result (`status="unsatisfied"`), not an
  exception; refusal, truncation, and a tool-call answer (`status="tool_calls"` — this surface
  has no executor, so a turn that stopped to request tools is handed back with its calls
  rather than having its empty text judged) short-circuit **before** validation, in the same
  order the loop decides them; a validator defect raises `OutputValidatorError` and never
  re-prompts. `ValidatedCallResult` carries the receipts of every call made on every settled
  result, and an exception escaping `acall` carries the completed calls' receipts as its
  `receipts` attribute — the failing call's own receipt reaches only
  `ModelCallRunner.subscriptions`, because the adapter raised instead of returning it.
  A thin sync facade (`call`) covers callers with no event loop and
  refuses to run inside an active one. The runner is **frozen**, and `max_repair_calls` must be
  an exact non-negative `int`, like every other budget control in the kernel — the loop bound is
  `repair_calls >= budget`, which `nan` makes permanently false and `inf` never reaches, so
  neither a budget arriving from dynamically typed configuration nor one reassigned onto a
  reusable runner afterwards can authorize unbounded paid model calls.
- **A repair call never carries tools.** The standalone surface has no tool executor, and a
  validation failure must not escalate into a tool loop — inside `AgentLoop` a repair turn is
  deliberately a full agent turn; here it is deliberately not. Repair follows the shape of how
  the **incoming request** carried its conversation, never what the answer came back with:
  by-value messages append the answer and the repair prompt, a request that itself arrived on a
  continuation handle carries the repair as the next instruction on the new handle, and a
  one-shot instruction is synthesized into the by-value form. A one-shot call is never promoted
  onto the handle path because the provider returned a response id — `OpenAIModelAdapter` sends
  `store=False`, so that id was never persisted and the repair would 404, losing the whole call
  to an exception. A request that arrived **on** a continuation handle whose turn came
  back **without** a new handle has no repairable shape — the conversation is on the provider's
  side of that handle — so it settles `unsatisfied` without spending a repair call rather than
  repairing against a synthesized prompt that drops every prior message.
- **Streaming is per attempt.** `acall` takes an `AttemptDeltaConsumer`
  (`(attempt_index, chunk) -> None`) instead of a plain `DeltaConsumer`: a rejected attempt's
  text is discarded output, and a consumer that renders or accumulates chunks must be told when
  the previous attempt is retracted. Carrying the index in the signature makes the boundary
  impossible to miss rather than a convention to remember.
- The validation routine, exception classification, repair text, and failure rollup moved from
  the loop's settle module into `core.output_validator`
  (`run_output_validators` / `build_repair_message` / `failures_by_validator`) and the loop now
  imports them — one rule source for both execution surfaces, so the repair dialect and the
  defect boundary cannot drift. No behavior change on the loop path.

### Added — generation parameters reach the providers, and the gateway proves it applied them

- The OpenAI adapter sends `temperature` / `top_p` / `max_output_tokens` on the Responses API
  body when configured (one shared payload builder covers the one-shot and streamed paths). A
  direct provider call has no applied-echo, so `on_unsupported` is not enforceable there:
  `"fail"` and `"omit"` behave identically and an unsupported parameter surfaces as the
  provider's own error through the existing taxonomy.
- The `monoid.llm-turn.v1` wire carries a `generation` block (only when configured — the
  protocol id is unchanged and generation-free traffic keeps its exact previous shape), and the
  reference gateway parses it **fail-closed with the same codec the kernel uses**, threads it
  into the upstream adapter's config on both the blocking and streaming paths, and echoes
  `generation_applied` (response body, and the terminal `turn_complete` frame on the stream).
- **The echo is derived from what the upstream adapter declares, never from the request.**
  Adapters opt in with `generation_support = "native"`, read through the same fail-closed probe
  as `structured_output_support`; the reference gateway omits the echo when its upstream does
  not declare it. Echoing the requested block back would have matched exactly on the client
  and let `on_unsupported="fail"` accept sampling controls that an ignoring adapter — the
  offline echo adapter, any `provider_adapter_factory` backend — never put on a request.
  A declaration can therefore be conditional: `GatewayModelAdapter` only *forwards*, so it
  declares `"native"` under `on_unsupported="fail"` (where a returned turn is a proven turn)
  and `"none"` under `"omit"` (where it deliberately accepts an unproven one). Otherwise a
  chained gateway would mint a fresh positive echo for a call whose inner hop proved nothing.
- **The gateway client refuses a turn whose parameters cannot be proven applied.** Under the
  default `on_unsupported="fail"`, a response without a matching `generation_applied` echo —
  an older gateway that silently discarded the block, exactly the deployment this exists to
  catch — fails with non-retryable `gateway_generation_not_applied`; `"omit"` accepts
  best-effort transport. Both the sync response and the streamed terminal frame enforce it,
  through one shape rule and one policy rule shared by both: a malformed echo is
  `gateway_bad_response` on either transport whatever the policy says, and the rejection
  carries this client's own retry evidence (`provider_retried`) since no turn is returned to
  carry it.
- The gateway wire also carries `reasoning.on_unsupported` and `generation.on_unsupported` now
  (off-default only). The server rebuilds a config object from each block, so a field left off
  is not "unset" there but the *default*: a client's `"omit"` came back as `"fail"` on the
  server's copy. That becomes a live failure as soon as a gateway's upstream is another
  gateway — the next hop enforces the reset policy and rejects a turn the caller asked to
  accept best-effort — and it hits `output_schema` callers too, since the same knob gates the
  schema echo. The applied-echo comparison is untouched: it is built from
  `build_generation_payload`, which carries provider knobs only, never policy.
- `TurnComplete` gains an optional `generation_applied` field so the streamed echo has a place
  to ride; absent means the wire never mentioned it.

### Changed — `ReasoningConfig.from_json` is now fail-closed

- `effort`, `summary`, and `on_unsupported` reject values outside their documented enums with a
  field-named `ValueError`. Previously the codec accepted arbitrary values and the mistake
  surfaced later (or not at all) depending on which ingress the config travelled through.
  Payloads that only ever carried documented values are unaffected. The reference gateway's
  request parser now reuses this codec (and the generation one), so an out-of-enum reasoning
  value or an out-of-range sampling value answers 400 `gateway_bad_request` at the boundary
  instead of travelling to the upstream provider. The direct-Python half of the 0.20.1
  "retained and direct-Python controls fail closed" contract landed with the internal-review
  pass above (`validate_reasoning_config` consumed by `normalize_model_config`); this entry
  closed the codec half.

## [0.20.1] - 2026-08-01

### Fixed — Studio traces tell the operational story

- Studio hides `model.output.delta` and `model.reasoning.delta` events from Trace by default and
  reports an operation count. The token-event toggle remains available for retained runs that
  already contain legacy delta events, and the compact summary reports omitted event counts and
  UTF-8 text bytes.
- Trace export now offers two explicit contracts. Raw export preserves every source event as
  `studio.trace-export.v1`; compact export writes `studio.trace-export.compact.v1`, its omission
  summary, and operation-level events only. The side-panel preview uses the same operation filter,
  so long answers no longer displace tool, task, and lifecycle activity.

### Added — one live content path from provider to Studio

- The kernel exposes a provider-independent, passive model-stream observer for output and reasoning
  deltas. Synchronous, asynchronous, and streamed model calls share the same observer lifecycle;
  observer failures stay contained and cannot replace the model-call outcome. `AgentLoop.astream`
  remains the execution-owning stream, and `emit_output_deltas=True` retains the legacy durable
  event mirror for integrations that explicitly use it.
- Runs can opt into the private append-only `model-content.jsonl` sidecar. It batches deltas into
  bounded UTF-8 segments, records terminal outcomes and content-addressed settled text, recovers
  valid records around malformed or torn lines, and exposes interrupted or abandoned partial
  output to entitled readers. Existing transcript-only runs continue to hydrate through the
  transcript fallback.
- The Reference backend adds a bounded, root-scoped `monoid.model-stream.live.v1` broker. One
  subscription multiplexes the root run and validated descendants, replays a retained suffix after
  reconnect, and emits a reset when a generation, sequence, capacity, or frame-size boundary makes
  replay incomplete. An acknowledged no-ring cursor carries eviction resets across reconnects
  without repeating hydration. Closing the presentation stream never cancels the run.

### Added — replay-safe Studio chat recovery

- Studio renders live root and subagent output/reasoning through a separate `/api/model-stream`
  EventSource. These frames stay outside the event reducer, durable Trace, and raw operation-log
  export, so users keep token-by-token feedback without multiplying `events.jsonl` envelopes.
- A reset or missing UTF-8 byte range hydrates the latest authorized prefix from
  `/api/model-content`, then applies retained live frames from the broker baseline. Reloads,
  transient SSE gaps, broker eviction, and backend generation changes recover without duplicating
  text. Interrupted work retains its available partial output.
- Failed provider prefixes follow the retry lifecycle. A transient failed attempt disappears when
  its automatic replacement stream starts; an explicit Studio Retry durably identifies and removes
  the exact failed turn it reissues, including after a lost response or reload. Retained replay and
  hydration cannot restore that abandoned prefix. Non-retryable failures followed by a new user
  message are joined into the durable chat projection, so their partial output remains visible after
  later calls replace the live hydration snapshot. User-interrupted partials remain visible too.
- Hydration binds every root/descendant record to its run directory and verified sidecar file
  identity. Coordinated flush failures, file replacement, and writer lifecycle races return HTTP
  503, which Studio retries. Request-scoped path watchers catch complete open/close ABA cycles
  without retaining per-run registry tombstones or invalidating unrelated runs.

### Compatibility and deployment notes

- Studio now sets its legacy durable delta mirror to off and uses the live/private content channel.
  Consumers that previously read token text from a Studio run's `/api/events` stream should use
  `/api/model-stream`; direct `AgentLoop` users can continue to select durable delta events with
  `emit_output_deltas=True`.
- `MONOID_OUTPUT_DELTAS=0` and `monoid studio serve --no-output-deltas` form Studio's content-egress
  gate: both the live SSE route and private model-content sidecar are disabled. Provider streaming
  remains enabled when the async HTTP transport is installed, preserving token-boundary Stop.
- Live Studio delivery requires the `[http-async]` extra. The new wire and artifact identifiers are
  registered in the compatibility ledger, and old run directories remain readable.

### Fixed — source distributions exclude local release scratch trees

- The sdist exclusion now covers the full `.tmp/**` subtree. Git's repository-wide
  `!.env.example` exception could re-include a nested copy from an ignored release-audit directory
  when building from a developer worktree. Release archives now exclude that local scratch content
  even when an inner filename is independently unignored.

## [0.20.0] - 2026-08-01

### Fixed — Studio plan truncation is visible (gap 9)

- Studio now preserves `plan.updated.truncated_items` in its run state. The Workspace inspector
  labels the completion count as the visible subset and shows how many additional steps the
  bounded public preview omitted; the Trace view reports both the original total and the omitted
  count. A 25-step plan capped to 20 no longer reads as a complete 20-step plan.
- Rebuilt the committed `web/dist` bundle from the pinned Node/npm toolchain. The Studio source,
  reducer contract check, and packaged wheel assets therefore carry the same behavior.

### Added — policy-gated OpenTelemetry preset (W9)

- `OtelEventSink` now accepts `parent_context`, `span_mode`, and `capture_policy`. A valid
  `InvocationContext` attaches `invoke_agent` or a standalone model-call span to its W3C parent;
  malformed trace metadata is ignored and an omitted parent retains the ambient OTel context.
- The default `agent` mode preserves the existing `invoke_agent → chat/tool/subagent` tree. Its
  paired `model_io_subscription()` enriches the already-open chat span with receipt metadata and
  the policy-approved capture, keeping one inference span per call. The `model_call` mode emits one
  chat span per standalone `ModelCallRunner` receipt and ignores the agent event facet.
- OTel capture defaults to `none`, preserving the previous metadata-only and no-content contract.
  `digest`, `redacted`, and `full` are explicit opt-ins; redaction failure downgrades to digest and
  records the downgrade without exposing the raw payload. Successful redaction records the applied
  policy digest; digest/length maps describe raw fields in every content-revealing mode. The
  content attribute is an opaque Monoid JSON shape rather than the OTel GenAI content schema.
  Capture stays on the model-I/O channel and never widens `events.jsonl`.
- Recovered activations lazily create their run root when the first child event arrives, and
  runtime receipt metadata corrects the active chat span's provider/model. Close is idempotent
  across the EventSink and model-observer ownership paths. OTel exporter and serialization failures
  are contained, and error span status no longer copies public error prose into its description.

### Added — AgentLoop model-call provenance, receipts, and capture-policy subscriptions

- `AgentLoop` now sends every one-shot, async, and streamed model call through its configured
  `model_io_subscriptions`. Successful and failed calls reach observers with the
  `ModelCallReceipt` produced by `ModelCallRunner`; the existing turn usage accounting remains the
  control-flow source. Capture delivery stays opt-in and each subscription applies its own
  `CapturePolicy`.
- `AgentLoop.invocation_context` carries caller Skill, batch, trace, and attribute provenance into
  every receipt. The loop replaces `run_id` with its authoritative run id, appends its durable
  `turn_NNNN` address to a caller step id, and preserves the caller attempt. Receipts remain an
  observer surface: this change does not add them to event, transcript, result, or checkpoint
  schemas.
- Model-I/O subscriptions are run-owned resources, matching `event_sinks`: normal close, durable
  release, discard, bootstrap failure, and recovery failure close each observer once. The Reference
  backend accepts per-run subscription factories, materializes fresh ownership-unique observers for
  every activation, cleans partial factory failure, and does not let failed recovery setup leave a
  provisional record that suppresses later retry.
- Parent subscription instances are not shared with in-process subagents. Child invocation context
  retains caller provenance and adds its run/task lineage. Current in-process child calls are not
  delivered to observers; child-scoped observer composition is deferred.

### Fixed — kernel ingress and artifacts now stay inside portable JSON

- **Strings and numbers are normalized once at semantic ingress.** Valid UTF-16 surrogate pairs
  retain their Unicode scalar, lone surrogates become `U+FFFD`, and Python/model/tool non-finite
  floats become JSON `null`. The rule covers user input, seed history, standalone and AgentLoop
  model calls, streamed chunks, native tool results, hosted-task requests and results, and emitted
  event data. The iterative traversal preserves shared and cyclic container topology and rejects a
  mapping when normalized keys would collide.
- **External JSON remains strict.** Backend, LLM gateway, MCP gateway, web gateway, and Studio HTTP
  requests reject non-standard `NaN`/`Infinity` tokens, finite-range overflow such as `1e9999`,
  excessive nesting, and duplicate object keys. Escaped lone surrogates are repaired by the same
  ingress rule. Direct Python entry points apply the substitution policy before adapters or durable
  writers see a value.
- **Every run-path JSON writer now asserts the contract with `allow_nan=False`.** Canonical digests,
  atomic artifacts, event and transcript JSONL, metrics, SQLite checkpoints and metadata, HTTP/SSE
  responses, and Studio chat projection fail fast if a semantic ingress is ever missed. Event-log
  and validation readers report retained non-finite artifacts as corrupt JSON instead of accepting
  a value that standards-compliant consumers reject.
- **Retained and direct-Python controls fail closed before coercion.** Run/model configuration,
  lifecycle state, capability and tool-surface structures, shell/web execution options,
  output-validator outcomes, and recognized provider usage counters require their exact types and
  finite ranges. Operator deny/redaction rules share the same Unicode scalar domain as the content
  they govern, so normalization cannot turn a configured denial into an allow.

### Security — `**` in a path pattern covered one directory level, not a subtree

- **`deny_patterns` and `redact_patterns` now use gitignore-style wildcard syntax.**
  Each normalized workspace path is matched independently; the table in `docs/CLI.md` is the
  normative contract (directory carry-over therefore does not make `dir/*` cover grandchildren).
  `matches_path_patterns` used `PurePath.match`, where `**` is a single `*` matched right-to-left.
  So `internal/**` denied `internal/a.txt`, **not** `internal/deep/a.txt` — and did deny
  `vendor/internal/a.txt`, which nobody asked for. `**/id_rsa` never matched a bare `id_rsa` at the
  workspace root. Both of those patterns were already in the shipped documentation.
  The previous minimum deny list in `docs/security/PRODUCTION_CHECKLIST.md`
  (`.env`, `*.key`, `*.pem`, `**/id_rsa`, `.ssh/**`, `.git/**`) left three deep-path holes,
  measured:

  | path | before | after |
  | --- | --- | --- |
  | `.ssh/keys/deep/id_ed25519` | **not denied** | denied |
  | `.git/refs/heads/main` | **not denied** | denied |
  | `id_rsa` (workspace root) | **not denied** | denied |

  It also omitted the `.ssh` and `.git` directory nodes themselves, which recursive move/delete
  tools check as one root argument. The checklist now uses bare `.ssh` and `.git`, covering each
  directory and its descendants when that directory or a descendant is the checked argument.
  Recursive operations rooted at an allowed ancestor still require a backend/tool tree preflight;
  the built-in recursive copy/move/delete path check currently sees the root argument only. This
  existing limitation is explicit in the CLI guide and production checklist. Re-check any policy
  written against an earlier version. Nothing caught the depth defect because every pattern fixture
  in the repo used one-level paths.
- **This moves an access boundary in both directions, and one caller is an allow-list.**
  `matches_path_patterns` backs four policy fields: `PermissionPolicy.deny_patterns` and
  `redact_patterns`, plus a tool binding's `ToolScope.denied_paths` and `allowed_paths`. For the
  three deny-shaped fields, matching more is safer and this change closes holes. For
  `allowed_paths`, matching more is *more permissive* — a binding scoped to `internal/**` now
  admits `internal/deep/x`, which it previously refused. Review binding scopes as well as policies.
  Patterns that were anchored by accident change too: `dir/**` no longer matches that directory at
  arbitrary depth; write `**/dir/**` for that. A bare directory pattern changes as well:
  `allowed_paths=["internal"]` used to cover only that node and now covers every directory named
  `internal` plus its subtree at any depth. Use `/internal` to retain root anchoring, and audit any
  scope that relied on node-only behavior. The synthetic workspace root remains outside every
  pattern, preserving the previous fail-closed result for root-cwd allow-list checks.
- **Negation is rejected rather than silently adopted.** `.gitignore` reads a leading `!` as
  "un-match", which would have handed every deny list a way to punch holes in itself — with the
  result depending on pattern *order*, while `PermissionPolicy.merged` combines two policies by
  concatenating and de-duplicating, i.e. as a set. Fresh permission-policy files, run specs, CLI
  flags, HTTP/control runtime configs, and `ToolScope` configuration now raise on an unescaped
  negation. Write `\!name` to configure a literal leading `!`. JSON output keeps `!name` and adds
  `"path_pattern_encoding": "monoid.literal-bang.v1"`, preserving old-reader matching while making
  current round-trips unambiguous. Retained `manifest.v1`, `backend-run.v1`, `checkpoint.v1`, and
  queued `command-inbox.v1` artifacts still decode the pre-v0.20 bare `!` spelling as a literal;
  patterns accepted by the old `PurePath` grammar but rejected by fresh v0.20 configuration retain
  the historical matcher on those artifact readers. The runtime-config semantic hash excludes
  only this representation marker while continuing to hash the raw path arrays and every other
  scope field. A v0.19 reader can therefore ignore the additive marker and recompute the same hash
  during rolling recovery. An unmarked
  pre-v0.20 `\!name` retains its historical literal-backslash meaning and cannot widen an old
  allow-list to `!name`. Durable runtime-config commands now preserve the validated
  `ToolBinding.authorization` policy enum (`allow`, `ask`, or `deny`) at its exact schema path;
  credential-shaped `authorization` fields elsewhere remain redacted. A leading `#`
  likewise stays literal rather than becoming a gitignore comment and silently dropping a rule.
- **Fresh configuration spellings are validated before matching.** A leading `./` is normalized;
  a trailing `/` covers both the directory node and its subtree; root-only and malformed patterns
  fail during config load rather than at the first event. Source backslashes are rejected except
  for the documented leading `\!` wire spelling: workspace input treats backslash as a separator,
  while pathspec would reinterpret it as an escape. Workspace paths containing C0/DEL controls are
  rejected. Windows rejects trailing-dot/space aliases, alternate data streams,
  reserved device names, and 8.3 alias-shaped spellings. The matcher remains lexical: case and
  Unicode-normalization aliases, plus symlink/hardlink identity, are existing workspace-backend
  responsibilities described in the production checklist. It receives paths rather than a
  workspace root, volume metadata, and filesystem object identities.
- **New runtime dependency: `pathspec>=1.1,<2`.** The stdlib cannot express these semantics on this
  project's floor — `PurePath.full_match` and `glob.translate` are 3.13+ while `requires-python` is
  `>=3.11`, `fnmatch` lets `*` cross `/`, and `PurePath.match` is what was wrong. Using 3.13's
  built-ins where available and hand-rolling a backport otherwise would be two implementations of
  one security control. Pinned to 1.x rather than `>=0.12` because the two disagree on `dir/*`
  against a grandchild, and a deny-list hole that opens from a dependency upgrade is not
  acceptable; `tests/test_path_patterns.py` runs every case against whatever version resolves.
- `Workspace.glob` is deliberately **not** changed. It is `fnmatch` over a discovery API the model
  calls to find files, where matching too much returns files it could already list. This function
  decides access, so the two want opposite defaults.

### Fixed — `job.json` had five readers and three answers (breaking for `monoid_agent_kernel.tasks`)

- **Every reader of `artifacts/jobs/<id>/job.json` now publishes the same projection.** The right
  one existed — drop `command`, preview `cwd`, redact `changed_paths` — and reached only the event
  sink. `monoid jobs --json`, `monoid job status --json`, the reference backend's
  `/v1/runs/<id>/jobs` and (through it) Studio's `/api/jobs` re-read the artifact off disk and
  published all three verbatim, and `core.projections` had a *fourth* answer that dropped `command`
  and redacted `changed_paths` but left `cwd` exact. So the same `cwd` came out `{"redacted": true}`
  on `shell.exec.started` and as the path on `monoid jobs --json`: **backgrounding a command was
  enough to route around an operator's `redact_patterns`.** The rules now live in one function,
  `public_view.public_job_artifact`, and `BackgroundJob.public_payload` calls it like everyone else.
  The artifact on disk is unchanged — it is the run's own record, `JOB_SCHEMA` requires those
  fields, and `monoid validate` reads the file rather than a reader.
- **The transformed object now names its own contract.** Public readers write
  `schema_version: monoid.public-background-job.v1` and retain the durable input identifier as
  `artifact_schema_version`. Reusing `monoid.background-job.v1` was incorrect: that schema requires
  the removed `command` and a string `cwd`. Readers validate the durable artifact, copy only the
  classified public fields, and validate the projection before returning it. Unknown fields,
  malformed values, and out-of-run symlinks therefore fail closed.
- **Missing policy metadata and path-bearing job errors fail closed.** A missing manifest produces
  a redact-all policy; a present manifest with missing/null `permission_policy` is corrupt and
  raises. When a run declares any `redact_patterns`, a non-empty job `error` is replaced as a unit:
  shell scan failures can interpolate a sensitive path that never reaches `changed_paths`.
- **A listing skips only a job artifact that disappears during its read.** `FileNotFoundError` is
  the expected completion/cleanup race. Permission, device, and other I/O failures now propagate
  instead of silently removing a running job from CLI, backend, Studio, and status projections.
- **A backward wall-clock correction no longer invalidates a live or completed task.** Background
  jobs and hosted tasks clamp a negative timestamp-derived `duration_s` to zero, so schema
  validation cannot abort start, output, or terminal publication before a completed task reaches
  the reentry queue.
- **Breaking: `monoid_agent_kernel.tasks.list_job_artifacts` and `get_job_artifact` are gone**,
  replaced by `public_job_artifacts` and `public_job_artifact_for`, which project rather than
  return the raw artifact and read the run's policy from its own `manifest.json`. Renamed rather
  than changed in place so the break is an `ImportError` and not a field that quietly stops
  arriving; `command_preview` is the replacement for `command`. **No raw accessor ships**: keeping
  one would leave the unprojected form a single default argument away, which is how this defect
  existed at all. An embedder inside the trust boundary can read `artifacts/jobs/*/job.json`
  directly. Neither name was in `contracts.__all__`.
- `monoid jobs` / `monoid job status` change only under `--json`; their human-readable output was
  already limited to ids, status, exit code and byte counts.
- **`monoid validate` no longer reports a false issue on every run that started a background job.**
  `BackgroundJob.to_json` has written `kind` since the tool bundle was widened and `JOB_SCHEMA` is
  `additionalProperties: false` without it, so any run directory containing a `job.json` failed
  validation with `Additional properties are not allowed ('kind' was unexpected)`. Declared
  optional, so a `background-job.v1` artifact written before `kind` existed still reads. No test
  had validated a run directory that had a job in it.

### Fixed — a corrupt `events.jsonl` no longer kills the surfaces that read it

- **`project_run_status` and Studio's chat catch-up degrade instead of raising.** `monoid watch`
  caught `EventLogCorruption` and printed one clean line; `monoid status --json` printed a 4.8 KB
  traceback, and Studio's catch-up raised inside a `do_GET` that had no exception handler, so the
  request died mid-response and the session rendered as if it had no history. Both now read through
  `read_committed_event_payloads`, which keeps every record before the damage and returns the
  reason.
- **The reason is published, not swallowed.** `project_run_status` carries `event_log_error` (empty
  on a clean read) and the Studio chat response carries the same required field. **Breaking:** the
  expanded Studio response is identified as `studio.chat.v2`; strict v1 clients must upgrade to
  read the new writer. The bundled reader retains exact v1 and v2 support for reader-first rollout.
  It validates the response at the HTTP boundary before hydration: unknown versions, same-version
  envelope expansion, missing v2 fields, malformed common fields, non-object responses, and chat
  messages missing renderer-required fields fail visibly. Message ids must be non-empty and unique;
  roles, content, attachment metadata, timestamps, and optional sources are type-checked before the
  keyed Svelte view receives them. Message-level version identifiers remain optional and ungated,
  and additional message, attachment, and source fields remain compatible.
  Corruption before `run.finished` leaves a finished run projecting as `running`, and a degraded
  projection that does not say it is degraded reads as a complete, shorter run. `monoid status`
  prints what it could project and then **exits non-zero**, on both the `--json` and the human
  branch. Studio renders the same reason as a persistent **Transcript is incomplete** warning;
  replayed lifecycle events cannot clear it or turn it into a retryable model failure.
- `read_event_page` and `reference/event_reader.py` deliberately still raise. The backend already
  answers with a clean 500, and silently shortening a *paged* reader would read as end-of-stream.
- **Studio's `do_GET` now has the exception handler `do_POST` has always had**, so an unanticipated
  error answers `500 {"error": "internal error"}` instead of dropping the connection with no status
  line. Reachable today: `/api/job-logs` catches `NativeAgentError` only, and `read_job_log_text`
  raises `KeyError` for an unknown job id.

### Changed — content egress from the event stream (breaking for `events.jsonl` consumers)

- **`turn.settled` and `run.finished` no longer carry `final_text` for model-authored answers.**
  They carry `final_text_digest` (a domain-separated `content_digest`) and `final_text_len` instead.
  The text moved to `transcript.jsonl`'s `settled_text` record, which entitled readers join back —
  the Studio and backend read paths hydrate it automatically. **Kernel-authored text still arrives
  inline** (for example `Stopped after reaching max steps.`), so the field is present on some settle
  events and absent on others; branch on which key is there rather than assuming either. `monoid
  watch --json` reads raw JSONL and does *not* hydrate. `status.json` no longer carries `final_text`
  at all: it is written by a fan-out sink that no hydration seam can reach, and keeping the key would
  have written `""` on every model-answered run.
- **Public previews are now capped by bytes rather than characters.** The cap compared bytes and
  sliced characters, so any string with at most 160 characters and more than 240 bytes was published
  in full while the payload reported `truncated: true` — in practice, no cap at all for non-ASCII
  text. `shell.preview_command` had the same defect independently, with different constants.
- **The tool-approval preview is now bounded.** `arguments_preview` masked secrets but applied no
  length, depth or item cap, so an `ask`-gated `fs.write` published the entire file body on
  `task.started`, in `task.json`, and back to the model through `job.list`. It now runs on the same
  traversal as its twin `args_preview`, which had the caps, so the bounds are stated once instead of
  in two places that carried disjoint halves of the policy.
  Sharing a traversal did **not** give the ordinary `allow` path secret masking: `args_preview`
  calls it without a mask and still publishes an `api_key` argument verbatim, on purpose. See the
  exceptions list below.
  The approval preview keeps a **much larger byte budget** than the trace preview and does **not**
  blank file-content fields, because a person reads it to decide whether a call may run: a command
  cut mid-string hides the part that matters (with the model choosing where that part sits), and a
  card rendering `{"redacted": true}` where a file body should be asks someone to authorize a write
  they cannot inspect. An `ask`-gated call therefore publishes more on `task.started` than the same
  call publishes on `tool.call.started` — bounded, but readable. Deployments that cannot accept that
  should not bind the tool to `authorization="ask"`, or should attach a redacting `EventSink`.
- `web.search` / `web.fetch` and `shell.exec` previews no longer copy their descriptors raw. Both
  branches exist to withhold something (the query and URL; env *values*), and an unbounded `locale`,
  `blocked_domains` entry or env *key* let a model publish exactly what was being withheld.
  This now covers the events those services emit **either side** of `tool.call.started`. They build
  their own payloads — `ShellApprovalRequest.to_public_json`, spread across seven emit sites covering
  six shell event types, and an inline `event_data` in each of the three web builders — so the same
  20 KB env key or `blocked_domains` entry rode out
  verbatim on `tool.approval.requested`, `tool.approval.denied` (a *rejected* call still shipped it)
  and `shell.exec.started`, while `args_preview` reduced it to a preview in the same run. A `cwd`
  under `redact_patterns` likewise came out `{"redacted": true}` on one event and as the path on the
  next. Both now go through one `public_event_payload`.
- **Dict keys are bounded.** `preview_value` capped every value and emitted `str(child_key)`
  untouched, so the same 30 KB file body arrived `{"redacted": true}` in the value position and
  verbatim in the key position — past the byte cap and past `_is_content_field`, which reads the key
  to judge the value and so has nothing to match when a body *is* the key. Bounding keys makes
  distinct keys collide, and a mapping resolves a collision by dropping one silently, so collisions
  are disambiguated with a `#n` suffix rather than traded for a second silent cap.
- **`redact_patterns` now covers every field that names a path**, not a hardcoded
  `{path, root, cwd}`. The registry declares path arguments per tool, and `fs.move`/`fs.copy`
  declare `source_path`/`destination_path` — so one `fs.move` published `paths: ["[redacted-path]"]`
  next to `args_preview.source_path: "secrets/creds.txt"` on the *same event*, the redaction defeated
  by the field beside it. Matched by name (`path`, `root`, `cwd`, `*_path`), which also covers custom
  and MCP tools that never register with the builtin registry.
- **The approval card no longer hides what it is asking about.** Content redaction was switched off
  for the approver while path redaction stayed on, so under `redact_patterns` the card showed a
  private key's *contents* above `{"redacted": true}` where the destination should be. The two are
  now a single `decision_surface` switch a caller cannot half-set, and the operator's explicit
  `redact_patterns` outranks it for the whole call — unlike the kernel's own content default, which
  an approver is entitled to see past.
- A container reachable from *itself* is elided with a `circular` marker rather than re-expanded,
  tracked over ancestors on the current path so a value legitimately shared twice still renders
  twice. Measured before the guard, at fanout 7: 23 s and 377 MB of serialized JSON, from an input that fits on a line.
- **Path redaction fails closed for every path-naming field**, including `*_path`. A value that
  cannot be normalized to a workspace path counts as redacted. **This over-redacts, and it reaches a
  first-party tool on every call:** `memory.*` addresses a virtual `/memories` root, so `path`,
  `old_path` and `new_path` are absolute by contract, never normalize as workspace paths, and render
  `{"redacted": true}` for any operator with *any* pattern configured — including on the approval
  card, since `memory` write bindings default to `authorization="ask"`. An earlier draft of this
  entry claimed no builtin was affected, on the strength of checking `stdout_path`/`stderr_path`
  (which do go out relative, and are fine). The
  narrower rule — fail closed only for `path`/`root`/`cwd` — was tried and taken
  back, because `normalize_workspace_path` raises on any `..` component *before* resolving it, so
  `x/../secrets/creds.txt` raises while naming a file the pattern does match, and was published
  verbatim next to `paths: ["[redacted-path]"]` on the same event. Both failure modes are real and
  only one is silent: an over-redacted field is visible and an operator can widen the glob, while an
  under-redacted one looks exactly like a field that was checked.
- A `path` argument that cannot be normalized (absolute, or containing `..`) is now redacted rather
  than raising. Normalization raises, the preview builders sit on the emit path, and the raise ended
  the run of any operator who had configured `redact_patterns` — it escaped `_emit_tool_started`
  before validation, and the error handler retried the same emission, so one malformed
  model-authored argument terminated the run instead of producing a tool error the model could
  correct. The guard lives in `public_path` itself, which every call site goes through — fourteen of
  them, ten across `loop`, `loop_phases`, `tasks`, `tool_services.shell` and `core.projections`,
  and four inside `public_view` — and in `public_proposal_file`, the one remaining direct caller of
  the raw predicate. No unguarded call to `PermissionPolicy.is_path_redacted` is left.
- A truncated `paths` entry now ends in `…`. These stay plain strings because `narration._target`
  falls back to them and joins them, so the cut has to be marked inside the text: an unmarked prefix
  is presented to an operator as the exact target of a write, and two long paths sharing a prefix
  become one indistinguishable name.
- **`plan.updated` and `artifact.emitted` cap their model-authored payloads**, matching what
  `tool.call.started` already did with the same values. This also bounds `status.json["plan"]`.
  `plan.updated.items` stays a *typed* array through the cap: each `step` is truncated as a string
  rather than replaced by a preview object, and a plan longer than the item cap reports the drop in
  a new sibling `truncated_items` key instead of appending a marker element. As an element it was a
  valid object (so schema validation passed) but not a plan item, so the Studio inspector drew a
  blank row for it and counted it in the `completed/total` denominator. `status.json` mirrors this
  with `plan_truncated_items`, so the cap is visible on that surface too rather than silently
  shortening the plan. Suppressing the in-band marker is scoped to that one typed array: a list
  nested inside a plan item is an ordinary JSON blob and keeps its own `truncated_items` marker,
  since the sibling count measures the root only and would not have reported its loss.
- Tool arguments nested deeper than 64 levels are now rejected with a `ValueError` the model can
  read and correct. They previously raised `RecursionError`, which — being a `RuntimeError` — fell
  through the tool-call handler entirely.

### Added

- **An operator kill switch for the delta channel.** `model.output.delta` and
  `model.reasoning.delta` publish raw model text, they are durable rather than live-only, and Studio
  enables them whenever the optional `httpx` extra is installed — so for the shipped app they were on
  with no supported way to turn them off short of uninstalling a package. Set
  `MONOID_OUTPUT_DELTAS=0` (applies to a run and every subagent it spawns), pass
  `monoid studio serve --no-output-deltas` (the flag is on the `serve` / `app` / `doctor`
  subcommands, not on the `studio` group), or construct `StudioConfig(stream_output_deltas=False)`.
  The cost is live token rendering in the UI **and mid-turn interruption**: the kernel only takes
  the streaming path when something consumes deltas, so Stop lands on the next step boundary rather
  than aborting inside the in-flight model call. That coupling predates this release —
  `emit_output_deltas` is `False` by default in the kernel, so the switch returns you to the kernel
  default rather than degrading past it — but Studio turns deltas on, so it is visible there.
  `AgentLoop.astream` takes a different path and is unaffected, as are the run result and
  `transcript.jsonl`. The completed answer still reaches the Studio UI, because its event feed
  hydrates `final_text` from `transcript.jsonl`.
- `transcript.jsonl` is now registered in the compatibility ledger. It became the authoritative
  source of displayed model text, so it is no longer merely a debug artifact.
- `monoid studio doctor` validates `MONOID_OUTPUT_DELTAS` and reports the effective delta state.
  A malformed value is a startup error by design, so without this the preflight passed and the next
  `serve` died in `AgentLoop.__post_init__` — and an operator who *thought* they had disabled a
  channel carrying raw model text learned nothing from a bare `[PASS]`. The report names the actual
  cause, including the case where deltas are off because the `[http-async]` extra is absent — which
  is the base package's default, and where a report covering only the two switches said raw model
  text was about to be published when none was.

- **`metrics.json` redacts `redact_patterns` paths.** It is one of the three public run artifacts
  and was the only one that never did, so an operator who configured a pattern, checked
  `events.jsonl` and `status.json`, and saw `[redacted-path]` had every reason to think it had
  worked while the whole path sat in the third file. Both callers of `build_metrics` already applied
  `public_path` to the same list for the events they emit.
- **Identifier fields are bounded**, via one `public_identifier`: a model-chosen tool name (the
  catalog is allowed not to resolve it), a `job_id`, and the `response_id` / `previous_turn_handle`
  the gateway echoes back — the last two arriving from outside the kernel's trust boundary. "That
  field is an identifier" is the assumption that left dict keys, env keys and the tool name
  unbounded, three separate times in this release.
- **Arguments that look like enums or integers are previewed, not copied.** `artifact.emit`'s `kind`
  is declared `{"type": "string"}` with no enum; `shell.exec`'s `timeout_s`, `max_output_bytes` and
  `startup_wait_s` are declared `["integer", "null"]`. The schema does not protect this surface:
  `tool.call.started` is emitted *before* `validate_args` rejects the call, so a model that sends a
  2 KB string in `timeout_s` publishes it and is only then told the call was invalid.
- **The run's terminal error goes through the same filter on every public artifact.**
  `public_error_message` was bound on `events.jsonl`, `status.json` and `failure.json` and missed on
  `metrics.json`. Note what that filter *is*: it substitutes only when the message contains
  `PRIVATE KEY`, and applies no length bound — so this closes an inconsistency, not the underlying
  exposure. `_error_from_status_body` embeds the entire LLM gateway HTTP response body in the
  message, and that body still reaches all four artifacts unless it happens to name a private key.
  Error text is listed under "Carried, deliberately" in `docs/OBSERVABILITY.md` for exactly this
  reason. The reference backend applies the same filter before serving `status` / `result` /
  `diagnostics`, and to the `failure.json` it writes; `AgentRunResult.error` stays raw, because the
  embedding application is inside the trust boundary.
- **No `KeyError` escapes the job tools.** `job.status` / `logs` / `cancel` / `wait` raised through
  tool dispatch, which catches `(NativeAgentError, ValueError, TypeError)` and not `KeyError`, so a
  model asking about a job terminated the run. Two distinct sources: an unregistered id, and a
  registered `HostedTask` with no log file — the second reachable by calling `job.logs` on an id the
  kernel itself handed the model through `job.list` after `agent.spawn`.

**What this release does not close**, stated as an exceptions list rather than an absolute claim:

- per-tool prose in `args_preview` (only `preview_kind="finish"` redacts it);
- validator and JSON-schema error text;
- a subagent's answer on `task.finished`;
- `job.list` / `job.status` exposing hosted-task payloads to the model;
- **`task.started.data.prompt`** — the model-authored delegation brief or HITL question, uncapped and
  neither previewed nor digested;
- **secret-named argument values on the ordinary tool-call path.** `args_preview` deliberately does
  not guess at secrets from key names — that heuristic was removed on purpose and redaction beyond
  content fields is the integrating backend's job via the `EventSink` seam
  (`examples/redacting_event_sink.py`). The *approval* record does mask them, so an `api_key`
  argument is masked on an `ask`-gated call and published verbatim on an `allow` call. Documented
  rather than changed, because changing it would reverse a deliberate architectural decision;
- and the delta channel remains **on by default in Studio** unless switched off.

Four gaps remain open after this review. Five related gaps are closed above or in the portable-JSON
batch: `job.json`'s readers, the corrupt-event-log guard, lone-surrogate ingress, and non-finite
number ingress and serialization, plus Studio's plan-truncation rendering.

- **Non-string scalars bypass the byte cap entirely.** `preview_value` bounds `str` and returns
  everything else unchanged, so a 4300-digit integer — model-authored, arbitrary content in base ten
  — reaches `events.jsonl` whole: one `artifact.emit` measured 96 KB with twenty verbatim copies.
  Bounding them means returning a different *type*, which is exactly how the `artifact.emitted.kind`
  fix broke that event's schema mid-review, so this needs a design rather than a patch.
- **A deeply nested tool argument still ends the run on the `allow` path.** `MAX_ARGUMENT_DEPTH` is
  reached only through the approval-request builder, so `ask`-gated calls reject and `allow`-gated
  calls carry the argument into the message history and on to `RunCheckpoint.to_json`, whose
  `dataclasses.asdict` raises `RecursionError` around depth 500 — while `json.loads`/`json.dumps`
  handle 900 — surfacing as `_CheckpointPersistError` out of `run_once`. Guarding tool *dispatch*
  does not close it: by then the turn is already in history. The fixes are to reject the turn at
  ingestion or to stop using `asdict`, and the latter also drops its deep copy, so a checkpoint would
  begin sharing mutable state with the live loop. Both belong to the durability surface.
- **One preview's total size is bounded only by its input.** The depth, key and item caps bound a
  preview's *shape*, not its size: a value reachable by many paths is re-expanded once per path, so
  nine levels of a mapping shared five ways — ten objects, 46 key slots — produced 26 MB of JSON in 1.1 s,
  and the cost grows with fanout: 110 MB at 6, 377 MB at 7. The cycle
  guard does not help, because sharing is not a cycle. Reachable only from a Python-object caller
  (a custom or MCP tool's `ToolResult.content`), since JSON cannot express sharing. A
  `PREVIEW_MAX_NODES` budget was implemented and reverted: spent per visited value it replaced
  whole `run.update_plan` items with `{"budget_exhausted": true}` markers on a plain-JSON payload,
  turning a rare adversarial cost into a *silent* cap on an ordinary one — `truncated_items` stayed
  0, so the plan looked complete and one step short of done. The right fix bounds the budget without
  crossing the `_INLINE_TEXT_KEYS` and plan-count invariants; that is a redesign, not a patch.
- **There is no aggregate budget, so the caps can be walked around by chunking.** Every cap is
  per-value (240 bytes) or per-container (20 keys, 20 items), and nothing bounds the total, so a
  model that splits its payload into sub-threshold pieces publishes all of it while obeying every
  rule: one `run.update_plan` with 20 items × 20 extra keys × 234 bytes puts **267 KB** across the
  three public files, and one `artifact.emit` puts 187 KB. Unlike the shared-graph case above this
  needs nothing but plain JSON. "Content stays off the public stream" and "100 KB per tool call" are
  both true today, which is the honest way to state where this release got to: it closed the routes
  that leaked a value *whole*, and did not make the stream a bounded channel.
See `docs/OBSERVABILITY.md` for the full public/private split.

- Added `ModelCallRunner`, which executes one model call against any adapter shape — a blocking
  `next_turn`, a coroutine `next_turn`, `anext_turn`, or a streamed `astream_turn` — through a single
  cancel/deadline race, and returns the turn with a `ModelCallReceipt` describing it. Which shape is
  used is a function of the call's own arguments, so the runner is usable outside a run: a gateway or
  a batch driver gets deadline and cancellation semantics that previously existed only inside
  `AgentLoop`. Delivery of content to observers is opt-in through `subscriptions`, but the receipt's
  digests are computed on every call, because they identify the call whether or not anyone is
  watching. A digest is empty when canonical JSON cannot carry the payload, or when the encoded
  output passes its size cap: it is documented as an
  exact replay key, and one taken over a truncated payload would match two requests that differ only
  past the cut, so no key is issued rather than a misleading one. Read an empty digest as *no key*,
  never as a key. It lives at the package root rather than under `core/` because it names the
  provider vocabulary it drives, and `core` does not import `providers`.
- Added `provider_retried` to `ModelTurn`, `TurnComplete`, and every streamed chunk (`TextDelta`,
  `ReasoningDelta`, `ToolCallDelta`), how an adapter reports that it retried internally before
  producing a turn. The kernel counts one adapter call per turn however many attempts happen inside
  it, so without this an audit receipt records a call that failed twice and succeeded on the third
  try as a clean single attempt. It is on every chunk rather than only the terminal one because a
  stream that is cancelled mid-flight never reaches its terminal chunk, and that is exactly when the
  evidence matters; `GatewayModelAdapter` marks the retry when it *decides* to retry — before the
  backoff wait and before reconnecting — because the retry is already certain there — `_should_retry` said
  yes at the end of the previous attempt — and every later point can be missed. Adapters with no retry loop leave it `False`, which is exactly true
  of them.
- `AgentLoop.astream` consumers now see one extra empty `TextDelta` per gateway retry. It is how the
  adapter reports a retry the stream itself may never live long enough to describe, and it
  concatenates to nothing, so the assembled turn is unchanged — but a consumer counting or rendering
  raw chunks will see it. Runs that emit `model.output.delta` events are unaffected: that path
  filters on non-empty text.
- Added `report_provider_retried()`, the seam an adapter uses to say its own retry loop is about to
  make another attempt. Every other carrier of that fact belongs to an *outcome* — a turn, a chunk,
  an exception the adapter raised — and a call the run abandons produces none of them: a blocking
  `next_turn` keeps running on a thread nobody reads, and the receipt is built from the
  `RunCancelled`/`RunTimeout` the race raised, which the adapter never touched. A run that timed out
  *because* the provider was retrying is the case most likely to matter, and it was the one case
  that recorded a clean single attempt. Optional and inert by default: an adapter that never calls it
  reports no retry, which is exactly true of one with no retry loop. `ModelCallRunner` honours it on
  success and failure alike, combined with what the outcome itself reports, never over it.
- The reference LLM gateway protocol now carries `provider_retried` on the one-shot turn result, `turn_complete`, the
  error payloads (non-200 body, 200-with-`error` body, SSE error frame), and — when true — each
  streamed delta frame. Two independent retry loops
  sit on that path and the client can only observe its own, so a gateway whose backend retried,
  answering a request the client got right the first time, recorded a clean single attempt. On the
  failure side that gap showed only when the client's own retry loop did not run — a 400/401/quota,
  the ordinary failure — because otherwise the client's own marker masked it.
  `GatewayModelAdapter` combines the two facts rather than assigning its own over the wire's. A
  gateway that omits the field reads as "did not retry", which is the only thing a wire that never
  mentions it can mean. `report_provider_retried` and `mark_provider_retried` are exported from
  `contracts` and the package root: an adapter author has to be able to name the seam the docs tell
  them to use.
- Added `ModelCallAborted`, raised when a caller's `should_abort` predicate stops an in-flight
  streamed call. Distinct from `TurnInterrupted` because the runner knows nothing about turns;
  `AgentLoop` translates it at its own boundary.
- Added `MultimodalModelAdapter`, `ProviderNamedModelAdapter`, `ConfiguredModelAdapter`, and
  `AddressedModelAdapter`, opt-in protocols that declare the optional capability members the kernel
  probes with `getattr` (`supports_multimodal`, `provider_name`, `config`, and
  `resolve_destination`). Implementing them is never required; they exist so the names and meanings
  are part of the checked contract and typed callers can narrow to "an adapter that reports this".
  Each member that is a value is a read-only property, so a `ClassVar`, an instance attribute, and a
  property all satisfy it; `resolve_destination` is a method because it answers for a given
  `ModelConfig`. The last two are what let a `ModelCallReceipt` name the model a call actually ran
  under and tell two hosts apart behind identical configs — the destination is hashed into the replay
  key and never recorded, so an internal hostname stays internal.

### Fixed

- A streamed direct-OpenAI turn now releases its response even when the client is reused. Leaving an
  `async for` does not close the iterator it drove, and the call's cleanup only closed the *client*,
  and only when the call owned it — which covered this by accident for an unscoped call, since tearing
  the pool down took the response with it. Inside a scope (`async with adapter` / `aopen`) the client
  outlives the call, so every turn aborted before the stream drained — cancelled, deadlined, or
  stopped by `should_abort`, all ordinary — left its response and connection checked out until the
  whole scope ended. Measured: three aborted turns, three connections still open server-side inside
  the scope. Enough of them exhaust the pool and later calls stall waiting for a connection that never
  comes back. This only ever affected the scope feature added in this release, not the previous one.
- A live model call is no longer left running silently when the bookkeeping around its wait fails.
  `_aawait` resolved the cancel/deadline race's arguments in its own argument list, and on the
  blocking path the call is already a daemon worker inside the provider by then — so a
  `current_cancel_grace_s` that raised landed between starting the call and entering the wait that
  owns the cleanup, leaving the worker with its future neither detached nor consumed, and with no
  report. Silence is the one thing this path claims never to do. The values are resolved before the
  wait now, and a failure detaches the call through the ordinary path, so the abandonment is
  reported like any other. The same shape as the registration failure already guarded a layer down.
- An adapter whose `open()` or `close()` raises now ends `monoid run` with a reported error instead of
  a bare traceback — a connection pool failing to construct or to tear down is the ordinary way in.
  Both calls sat below the handler that normalizes every other startup failure. The teardown case
  carried a second fault: the run's status and summary were echoed *after* the adapter scope unwound,
  and an exception from a cleanup callback replaces whatever is leaving the block, so a failing
  teardown silently swallowed the outcome of a run that had **completed**. The outcome is now echoed
  before the scope is released, so a cleanup failure costs the cleanup and not the result.
  A teardown failure never supersedes a real one, either: raising from a cleanup callback replaces the
  exception leaving the block, so a failing `close()` alongside a failed run reported only
  `close() failed` and the provider error an operator needs disappeared. It is the command's error
  only when nothing else is wrong; beside a real failure it is a warning on stderr.
- A tool call that refuses to describe itself no longer discards the turn it belongs to. The capture
  surface falls back to `repr()` for an object `vars()` cannot walk — a `__slots__` object, say — but
  an object that refuses *both* took the exception out through `_publish` after the provider had
  already answered, so a paid-for turn was thrown away by the code whose purpose is to prevent
  exactly that. The entry now degrades to `<unrepresentable TypeName>`: the record still says a tool
  call was there and that it could not be described.
- A model call the kernel refuses before reaching the adapter now reports `attempts=0` instead of 1.
  A run already cancelled, or past its deadline, when the call is requested never touches the
  adapter — but the receipt carried the default 1, so a consumer summing `attempts` counted provider
  work that provably never happened, against the field's own documented meaning ("the calls the
  kernel made to the adapter"). `ModelCallReceipt` accepted no such value before: `attempts` was
  validated as ≥ 1, and **0 is now legal** — read it as "no adapter call was made", not as a missing
  value. A receipt is still written for a refused call, because that is precisely the kind of call an
  audit trail is for; a failure *while* reaching into the adapter still counts as 1. A payload that
  omits the field still reads as 1, so older records are unchanged.
- An adapter whose `next_turn` is a callable object with an async `__call__`, or a synchronous
  `next_turn` that returns an awaitable, is now driven correctly. `inspect.iscoroutinefunction`
  answers for a function and says no to both, so the call went to the synchronous worker and the
  awaitable it produced was handed back *as the turn*. Nothing downstream reads a coroutine as a
  failure — every receipt field read is defensive — so the receipt recorded a **successful model call
  for a provider that was never invoked**, and the caller got an object whose every turn field was
  missing. The tool half already defended both shapes; the question "does calling this produce an
  awaitable?" now has one answer, `is_async_callable`, shared by both dispatch halves, because asking
  it two different ways is how the halves came to disagree.
- A retried streamed gateway call now carries a freshly resolved token, as the blocking one always
  did. `astream_turn` resolved its headers once above the retry loop, alongside the URL and the body
  — but neither of those can change between attempts and a credential can. A `token_provider` that
  re-mints near expiry (what `reference.backend` supplies, at `expires_at - refresh_skew_s`) crosses
  that line during exactly the window a backoff opens: the wait runs to `max_delay_s` and the run may
  already be minutes old. The retry then replayed the expired token, came back 401 — which is
  `gateway_auth_error` and *not* retryable — and ended the whole call terminally, where the blocking
  path recovered. Pre-existing: the previous release resolved the streamed headers in the same place.
- The CLI now holds open an adapter that offers `open`/`close` without also being a context manager.
  It probed `open` and then used `with`, so the very shape the probe invites — the lifecycle pair
  `AgentLoop` and `LoopSession` use, and the one `OpenAIModelAdapter`'s own `__enter__` delegates to
  — raised `TypeError` before the first turn, outside the CLI's error handling, ending the run in a
  raw traceback after `run_id` and `run_dir` had already been printed. Adapters with no client to
  hold are still left alone. One offering `open` without a callable `close` is reported as
  misconfigured before `open()` runs, so nothing it would have allocated is left with no way to be
  released.
- An adapter held open across two *concurrently running* event loops no longer has one loop's client
  closed under it. `OpenAIModelAdapter`'s scope drops a cached async client that belongs to another
  loop, on the grounds that its sockets live there — but "belongs to another loop" was not
  distinguished from "belongs to a loop that has moved on", so a second run asking for a client was
  enough to schedule a `close()` onto the first run's still-running loop and cut off a call in
  flight. Reuse now belongs to whichever loop the scope holds; a call from another live loop gets a
  client it owns and closes, exactly as an unscoped call does, and the scope is left untouched.
- A transport failure from the streaming client's own lifecycle is classified again. Hoisting the
  `httpx.AsyncClient` out of the retry loop (below) left its construction, `__aenter__` and pool
  teardown outside the per-attempt handler, so an `httpx.CloseError` or `PoolTimeout` escaped as a
  raw `httpx` exception. That is not merely a worse message: `_recoverable_turn_error` keys off
  `retryable` and a 4xx `http_status`, neither of which a raw `httpx` error carries, so a failure
  that had ended one turn recoverably — session alive, turn re-attemptable — terminalized the whole
  run and wrote `failure.json` instead. Now classified as `gateway_network_error`, retryable unless
  deltas were already committed, matching the previous release. One difference remains by design:
  a client whose *construction* fails is no longer retried per attempt (1 attempt, not 3), since the
  client is built once per call.
- A stream the run has given up on no longer delivers its remaining chunks into the *next* turn.
  Pre-existing — it reproduces identically on the previous release, which had the same unguarded
  relay at both of its streamed drive sites — and fixed here because this release rewrote that code
  into one place. The kernel can stop waiting for a provider but cannot stop one, and the stream
  drive runs as its own task, so a generator that survives the cancellation a boundary delivers goes
  on yielding into a `delta_consumer` belonging to a call that already raised. That consumer is
  `QueueEventSink.push_delta`, and one sink serves a whole run — the next turn rebinds it to a fresh
  queue — so an abandoned turn's tokens surfaced as the following turn's output. Measured before the
  fix: 13 chunks delivered after the boundary released the call. Deliveries now stop the moment the
  driving call ends.
- Extracting the model call into `ModelCallRunner` no longer freezes two of the loop's public
  mutable fields at bootstrap. `model_adapter` and `async_model_cancel_grace_s` were captured by
  value where the loop had read them live on every call, so an adapter assigned after `open()` was
  ignored — while the request was still *shaped* for it (`supports_multimodal`,
  `wire_image_encoding`) and the answer still *attributed* to it (`provider_name` on the assistant
  message). A grace raised after `open()` was likewise ignored, abandoning a model worker ~150x
  earlier than configured, while the tool half of the same knob honoured it. Both are now read
  through callables, as the cancellation token already was; the adapter is read exactly once per
  call so a receipt cannot describe a mixture of two.
- A synchronous tool handler's call authorization now reaches threads the handler starts itself. A
  `ToolContext` operation delegated to a joined child thread is checked against the same binding
  scope as its parent, instead of seeing no call at all and widening to the run-level permission
  policy — an authorization bypass for threaded handlers. The per-call isolation that keeps an
  abandoned handler on its own scope is unchanged; the two are resolved from separate tiers because
  neither alone is correct.
- Scoped `ToolContext` operations — `path_allowed`, shell execution, web search/fetch/context — now
  refuse when no tool call is in flight, instead of applying the run-level permission policy
  unnarrowed. Every scope check narrows only under a non-empty allow/deny list, so an absent call
  read as an empty scope granted the widest authorization in the run. This bounds a thread descended
  from a handler the run has abandoned: it is refused once the parent's call ends. The narrower
  remaining case is documented in `docs/CONTRACTS.md` — while another call is live, such a thread
  borrows that call's scope, because nothing links a thread to its creator.
- `ModelAdapter` and `AsyncModelAdapter` no longer declare optional capability attributes, which had
  made them **required** members for structural typing and rejected a third-party adapter that
  implements only `next_turn` — the default assigned in a protocol body reaches explicitly
  inheriting classes, not structural implementations. `supports_multimodal` and
  `wire_image_encoding` had this effect since they were introduced; the attributes are unchanged at
  runtime, where the loop has always probed them with `getattr` and a default.

### Changed
- An abandoned *asynchronous* call is now logged the way an abandoned thread already was. The
  warning was gated on there being a synchronous call, so a callee whose cleanup outran the grace
  interval was detached in silence — with the same unbounded shape as the sync case: one task, and
  everything it holds, per abandonment, on a loop that may run for days. For a streamed model call
  that is an open provider connection pool. Measured before the fix: 400 abandonments produced 400
  pending tasks and zero log lines, while the sync half produced 400.
- A model call is no longer lost, nor left without a receipt, because an adapter returned something
  slightly off-shape. `stop_reason`, `usage`, and `tool_calls` are read defensively, the same way
  `provider_retried` already was and for the same stated reason — a third-party adapter may return
  any turn-shaped object. A `usage=None`, which `examples/custom_model_adapter.py` invites by calling
  usage "optional", raised from inside the receipt's own construction, so no receipt was produced at
  all and an answer the provider had already been paid for was discarded over a token counter.
  The `provider_name`, `config` and `resolve_destination` probes are all now tolerated at the
  *lookup* as well as the call, so an adapter exposing any of them as a property that raises keeps
  its call; `resolve_destination` previously guarded only the call.
- A synchronous adapter or tool handler that raises `StopIteration` no longer loses its failure.
  `asyncio.Future.set_exception` refuses `StopIteration` by contract; the refusal surfaced inside a
  thread-safe callback where nothing awaited it, so the awaiting future stayed pending. A deadline
  still released the run, but the callee's actual failure never arrived, the kernel's abandonment
  warning stayed silent, and a run configured without a deadline hung. It now surfaces as a
  `RuntimeError` naming the cause. Raising it is ordinary — `next(...)` on an exhausted iterator
  does.
- A provider stream whose `aclose()` raises or hangs no longer replaces the call's own outcome. It
  runs in a `finally`, so a raising close turned a caller's abort into a terminal failure — killing
  the session that `ModelCallAborted` exists to keep parked — and a hanging one hung the run, since
  the abort is raised inside the awaited task and no run boundary is pending to bound it. The close
  now gets the same grace an abandoned call gets and is then detached, and its failure is not the
  call's. The bound is a detach rather than `asyncio.wait_for`, which cancels on timeout and then
  awaits that cancellation — so a close suppressing `CancelledError` still ran ~90x past the grace.
  An abandoned close warns on the `monoid_agent_kernel.model_call` logger.
- Capture failing no longer changes how a provider failure is classified. The failure receipt is
  published under a guard, so a `ModelAdapterError` carrying `retryable` and `http_status` reaches
  the caller even when delivery raises — the docstring promised delivery *before* the re-raise, and
  it had become delivery *instead of* it.
- A bookkeeping failure in the cancel/deadline race no longer orphans a call it had already started.
  Cancellation-callback registration and the timeout arithmetic now sit inside the `try`, so the
  `finally` that cancels, detaches, and consumes the call always runs; previously a raising
  `add_cancel_callback` left the adapter running to completion behind a run that had reported a
  failure.


- **Breaking for third-party synchronous adapters and tools.** A synchronous `next_turn` and a
  synchronous tool handler now observe run cancellation and the run deadline, instead of taking
  effect only once the call returns. Both previously let a wedged provider or handler outlast every
  run boundary; a run now reports `cancelled` or `run_timeout` within its cancel-grace window and
  abandons the call. Python cannot force-stop a worker thread, so the call is abandoned rather than
  stopped: it keeps running with its late outcome discarded, an awaitable returned too late is
  closed unawaited, and an abandoned tool handler may still be writing to the workspace. Sync
  adapters and tools should still enforce a timeout at their own I/O edge. Adapters needing prompt
  resource release should expose `anext_turn` or a coroutine `next_turn`.
- The configured cancel grace now applies to a synchronous call's worker thread, so a sync adapter or
  tool handler that returns inside the grace settles normally instead of being abandoned on the spot.
  Cancelling a sync call's waiter completes it immediately — there is no coroutine to throw
  `CancelledError` into — so the previous wait granted no grace at all to the one shape that needed
  it. A handler that lands inside the window now finishes its workspace writes before the run
  finalizes rather than racing it, and is not reported as abandoned. The grace is not an extension of
  the deadline: the run still reports `cancelled` or `run_timeout`.
- An awaitable returned by an abandoned synchronous call is now disposed whatever its shape: a
  coroutine is closed, and a future or task is cancelled and its outcome consumed. Previously only
  coroutines were handled, so in a persistent backend loop a returned task kept running after the
  run was cancelled and a future completing with an exception was never consumed.
- Abandoned synchronous calls no longer run on the event loop's default executor. `asyncio.run`
  joins that executor's workers before returning, which made a run deadline enforced internally but
  unobservable to the caller — it produced its result on time, then blocked at loop shutdown until
  the provider returned on its own.
- The abandoned-synchronous-call warning now logs under `monoid_agent_kernel.core.sync_bridge`
  instead of `monoid_agent_kernel.loop`. The bridge that runs a blocking call on a daemon thread
  moved into `core` so the model-call runner can share it with the tool path, and a logger naming a
  module it no longer lives in would misdirect anyone reading the warning. Deployments filtering
  this warning by logger name need to add the new one; the message text is unchanged.
- A run whose deadline has expired or whose cancellation has been requested no longer reaches the
  provider. The boundary is checked before the adapter is dispatched, not only in the race around
  it: the race reported a boundary that had already been crossed, but by then the request was out
  and the provider had been paid for work the run had already decided not to do. The refusal still
  publishes a failure receipt, so a call the run declined is recorded rather than absent.
- `GatewayModelAdapter.astream_turn` no longer blocks the event loop while it waits to retry. The
  backoff used a blocking sleep called from inside an async generator, so the whole loop stopped for
  the length of the wait — up to `max_delay_s` per retry; the default policy reaches 1.1s on its second
  backoff, and a longer configured one was measured freezing a 100ms heartbeat for a full 4.5s wait. Nothing else in the run progressed, and the run's own
  cancellation and deadline are raced on that loop, so a run told to stop kept waiting for a provider
  it had already given up on. The wait is now awaited; the schedule is unchanged and shared with the
  sync path, which keeps its blocking sleep because it runs on a thread.
- `GatewayModelAdapter.astream_turn` builds one `httpx.AsyncClient` per *call* rather than per
  *attempt*. Constructing one is synchronous and not cheap — ~285ms warm — and inside the retry loop
  that cost was paid again on every retry, with the event loop unavailable throughout: the same
  defect as the blocking backoff above, one statement later. Retries now also reuse the connection
  pool. A host counting client constructions, or relying on a fresh pool per attempt, will see the
  difference.
- An adapter that cancels its *own* call is now reported as `ModelAdapterError`
  (`model_adapter_cancelled`) instead of raising `asyncio.CancelledError` out of the run. The two
  are different events — the run stopping versus the adapter failing — and only the second is the
  adapter's. Callers that distinguished them by catching `CancelledError` around a run should catch
  `ModelAdapterError` for this case; cancellation of the run itself is unchanged.

## [0.19.2] - 2026-07-19

### Added

- Added a Reference-private single-handle, snapshot-bounded event page reader with
  monotonic-prefix and content-verified byte-offset anchors plus raw-read source-work metrics,
  supplying the sparse index's verified scan primitive while preserving Core cursor semantics
  inside the protected append-only run-directory boundary.
- Added a Reference-private process-local sparse offset index that retains verified anchors,
  stages bounded candidates during successful page scans, performs logarithmic warm lookup, and
  rebuilds safely after detected source invalidation or process restart.
- Bounded the sparse index's retained source slots with a pinned least-recently-used policy and
  an authoritative uncached fallback for saturated admission, plus cache-capacity metrics.
- Wired one backend-owned sparse event index into root, descendant, and diagnostics projections,
  with configurable retained-source capacity and unchanged Core subscription semantics.
- Added exact-byte conformance evidence descriptors plus a checked reader that reads retained v1
  reports into the current typed provenance model with provenance explicitly unavailable.
- Added offline minimal-agent report verification of exact-byte digest integrity, sanitized
  self-asserted target metadata, profile binding, lifecycle completeness, rule-reference coverage,
  and internal report/evidence semantic consistency.
- Added opt-in v2 external reports. When `--evidence-dir` is supplied with an evidence-capable
  adapter, the runner retains content-addressed normalized evidence, binds all four rules to its
  content-addressed evidence reference, self-verifies, then publishes retained evidence and
  configured JUnit/JSON outputs before stdout. Default report-only and legacy adapter runs preserve
  the v1 output contract.
- Added a Reference-private DBOS 2.26 owned-runtime adapter that binds workflow registration,
  identity-scoped enqueue and retrieval, queue, launch, and shutdown operations to one captured
  singleton and registry, rejects Cloud/Conductor mode, and verifies global identity plus
  DBOS-owned thread cleanup before ownership release.
- Added a single-owner Reference-private DBOS runtime host that composes hosted control and run
  participants under one captured runtime, with deterministic workflow identity, an explicit
  shared surface-configuration contract, pre-launch workflow/listener aggregation, one
  launch/shutdown lifecycle, pre-destroy admission drain, deadline-bounded close, and process
  fencing when participant drain or ownership verification is uncertain.
- Added a Reference-private hosted control participant that defers workflow, queue, admission,
  and shutdown ownership to the single DBOS runtime host while preserving standalone behavior.
- Added a Reference-private hosted run participant with host-owned workflow, queue, admission,
  active-drive, and shutdown lifecycle while keeping checkpoint semantics portable.

### Fixed

- Projected MIN-03 result identity and completion checks as booleans so raw run identifiers and
  result status strings do not enter conformance report projections.
- Treated newline-terminated event-log records as committed and repaired uncommitted crash
  fragments before recorder or direct append, preventing malformed concatenation and sequence
  reuse during restart. Committed records with invalid sequence fields now fail closed.

## [0.19.1] - 2026-07-16

### Changed

- Pinned Studio frontend development and CI to Node.js 24.18.0 and npm 11.16.0, added
  fail-fast developer engine checks, and documented POSIX nvm and nvm-windows setup.

## [0.19.0] - 2026-07-13

### Added

- Rebuilt Agent Studio as a packaged Svelte 5, TypeScript, Vite, and Tailwind CSS application with
  responsive profile, run-control, approval, live-config, change-review, and request-preview
  surfaces. Released wheels and source distributions carry the compiled UI and require no Node.js
  runtime.
- Added explicit Studio pause, resume, and failed-turn retry BFF controls plus a versioned,
  runtime-resolved `ModelRequest` preview payload.

### Changed

- Moved Studio frontend authoring dependencies into `studio-ui/` and added a deterministic CI build
  check that keeps the committed Python-package assets synchronized with the Svelte source.

## [0.18.0] - 2026-07-12

- Added an experimental optional DBOS Reference activation-recovery profile. Its finite,
  run-partitioned resume workflows restore one checkpoint, drive one durable suspension boundary,
  reject stale sources, commit applied-input markers, return the stored receipt for duplicates,
  and recover after a same-slot process kill with one semantic effect and one identity-bound
  boundary receipt. `CheckpointStore` remains the semantic authority while DBOS owns operational
  admission, serialization, retry, and workflow recovery. DBOS dependencies and runtime types stay
  in the optional Reference profile; the path constructs no legacy lease, inbox, recovery, or
  watchdog services. Ambiguous checkpoint-store results reconcile by exact readback and remain
  pending until the exact commit or a conflicting writer is observed.
- Added portable durable suspension observations and `AgentLoop.release_parked()` so recovery
  drivers can return an already-committed boundary and release process resources without
  finalizing a resumable run.
- Added an executable production embedding handbook with offline-tested local and hosted,
  multi-tenant golden paths, portable deployment responsibilities, one explicit Reference inbox
  assembly, and clear separation from the optional experimental DBOS activation-recovery profile.
- Added a durable Reference command inbox with idempotent append, ordered and recoverable claims,
  acknowledgements, result receipts, queue limits, authenticated principal attribution, sanitized
  persistence, owner-side draining, and in-memory/SQLite implementations for cross-worker control.
- Added cursor-correct event subscriptions with SSE event IDs, `Last-Event-ID` resume, heartbeat
  comments, terminal final-event draining, recovered-run support, and authorized descendant feeds;
  Reference backend HTTP and Studio now share the subscription abstraction.
- Added an external minimal-agent conformance runner with stable rule IDs, typed observations,
  versioned JSON and JUnit reports, packaged compatibility fixtures, and reusable checkpoint-store
  and capability-broker implementation contracts.
- Added deterministic LocalFS/SQLite durability fault coverage for corrupt and future checkpoints,
  missing blobs, stale publication pointers, interrupted writes, metadata divergence, lease races,
  side-effect recovery, and capability revocation.

### Added
- Added versioned durable codecs with explicit loaded, migrated, missing, corrupt,
  and unsupported-version outcomes; LocalFS and SQLite checkpoint stores and
  Reference recovery now use checked checkpoint and run-metadata reads.
- Added a machine-readable compatibility registry and matching ledger for public wire and
  durable artifacts, aliases, mixed-version operation, schema evolution, and coordinated
  upgrade/rollback procedures.
- Added explicit native async model, streaming model, and async tool-handler contracts;
  async tools now execute on the run loop with deadline/cancellation propagation while
  synchronous handlers retain worker-thread compatibility.

### Changed
- Classified every test into an enforced unit, contract, or integration tier and
  replaced advisory xdist/coverage jobs with required deterministic shards, a
  coverage floor, cross-platform smoke tests, and minimal/all-extras install smoke.

### Fixed
- Hardened checkpoint and run-metadata readers with structural validation, lookup-key and
  committed-sequence binding, recovery-shape checks, and generation-based reconciliation between
  local and shared metadata copies.
- Preserved every unstarted approval replay across a process loss by consuming one durable head at
  a time and carrying completed observations into the next safety checkpoint.
- Applied cancellation and the session deadline to native async model calls and streams with
  bounded provider cleanup; synchronous adapters retain their documented provider-timeout
  responsibility.
- Closed a Reference inbox redaction gap where JSON coercion of bytes or custom objects could
  reintroduce a bearer into durable command arguments, and fenced watchdog restart after a stop
  timeout.
- Redacted raw exception bodies from conformance JSON, JUnit, and console diagnostics; strengthened
  the reusable checkpoint-store contract to prove persistence across a fresh store instance.
- Excluded the workspace-local `.tmp/` release scratch directory from source distributions.
- Kept offline Studio functional in the minimal install by falling back to complete one-shot
  gateway turns when the optional async HTTP transport is unavailable.

## [0.17.1] - 2026-07-09

### Added
- Added a GitHub Actions PyPI publishing workflow for GitHub Releases, using PyPI
  Trusted Publishing with the `pypi` environment and release-tag/version validation.

### Documentation
- Split the top-level README into role-focused guides: the full CLI
  reference moved to `docs/CLI.md`, the backend/gateway walkthrough to `docs/BACKEND.md`,
  and outputs/event-sinks/observability to `docs/OBSERVABILITY.md`. The README now focuses
  on positioning, install, the no-server quickstart, core concepts, and a documentation map.
- Added a `docs/security/` cluster: `SECURITY_MODEL.md` (intended boundaries, non-goals,
  trust zones, and core invariants — each mapped to an operational rule and its tests),
  `THREAT_MODEL.md` (trust boundaries, a prominent permissive-by-default warning, and a
  threat-by-threat table of kernel defenses vs. integrator responsibilities), and
  `PRODUCTION_CHECKLIST.md` (actionable pre-deployment steps). Expanded `SECURITY.md` to
  link the cluster and surface the permissive default.
- Reorganized `docs/README.md` around a "Find your path" persona navigation
  (app developer / integrator / tool author / operator / security reviewer / contributor).
- Updated README repository-file links to absolute GitHub URLs so the PyPI long
  description links resolve to GitHub.
- Clarified the subagent fan-out threat model to cover registered subagent definitions,
  exposed `agent.spawn` bindings/capabilities, CLI-provided subagents, fork skills, and
  Studio's `delegate` capability.

## [0.17.0] - 2026-07-08

### Added
- Optional provider-backed Memory tools via `monoid_agent_kernel.memory`, including
  `MemoryProvider`, `LocalFilesystemMemoryProvider`, filesystem-style memory operations,
  provider-owned storage, and `memory.search`.
- Default tool binding bundles for read, write, shell, and artifact capabilities, plus
  stronger builtin filesystem, shell/job, and artifact tools.
- Studio durable chat projection in `studio.chat.jsonl`, with `/api/chat-transcript`
  restoring browser-facing user, assistant, and error messages across reloads and restarts.

### Changed
- Studio exposes Memory as an available capability, disabled by default and stored under
  `run_root/studio-memory/<workspace-key>/` when enabled.
- Destructive workspace helpers `fs.copy`, `fs.move`, and `fs.delete` require approval by
  default in the generated write tool bundle.
- Studio chat replay now reads durable chat messages before replaying trace events, while
  `events.jsonl` remains the trace stream and `transcript.jsonl` remains the private
  model-call log.

### Fixed
- Reopened Studio chats preserve the initial user messages and later conversation turns
  created after this release's durable chat projection.

## [0.16.1] - 2026-07-05

### Fixed
- Updated the README quickstart so the snippet works with the current `AgentRunSpec`
  API by supplying `Path`-based `workspace_root` and `run_root` values.
- Switched README Studio screenshots to GitHub raw image URLs so the PyPI long
  description can render them.
- Ignored local Studio/log artifacts to keep source distributions clean when built
  from a working checkout.

## [0.16.0] - 2026-07-05

### Changed
- Phase 4-1 public-surface cleanup: `monoid_agent_kernel.contracts` and the
  top-level `monoid_agent_kernel` package now export only the contract surface.
  Helper/default implementations and convenience adapters are imported from their
  explicit modules.
- Phase 4-2 lifecycle vocabulary cleanup: run lifecycle payloads now use
  `state` plus `terminal` instead of legacy lifecycle `status`. Terminal
  `AgentRunResult.status`, `ControlResult.status`, proposal status, tool status,
  job status, and metrics status keep their domain meanings.
- Phase 4-3 test/CI readiness: backend tests now have a managed factory seam for
  spawned future cleanup, Studio shutdown joins owned server threads, and CI runs
  xdist plus coverage as advisory checks.
- README screenshots now show the v0.16 Studio profile workflow, including a
  data-analysis run and the exact model request preview in the profile editor.
- `AudioPart` and `VideoPart` are now exported from the contract surface to match
  the core content contract.

## [0.15.0] - 2026-07-03

### Added
- Operational rule coverage for OR-01 through OR-13, mapping each rule to Core Helper Kit
  surfaces, conformance assertions, Reference harness cases, and primary tests.
- Executable conformance profiles for tool-agent approval, optional side-effect tools,
  external-agent message fabric, and the bundled Reference full profile.
- Strict wire parsing helpers for JSON-native payloads, plus property tests for
  external-agent envelopes and inbox/outbox round-trips.
- Public/private task payload separation, including safe public capability-result summaries.
- Canonical external-agent metadata merge helpers so user metadata cannot override trusted
  peer, task, request, result, or trace identity.

### Changed
- Reference backend, web tool service, durable metadata listing, and Studio subagent event
  routing now consistently use the Core Helper Kit paths established by the operational rules.
- Approval callback parsing now fails closed for ambiguous approve/deny values while preserving
  durable replay behavior.
- Strict parsers continue to accept legacy `native-agent-runner.*` protocol ids during the
  namespace migration window.

### Fixed
- Recovered outbox requests, capability leases, and control commands created before the Monoid
  namespace rename are accepted by the new strict parsers.
- Public hosted-task payloads no longer expose raw capability grant material such as `lease` or
  `token_ref`.
- Requested web domain scope now respects wildcard narrowing rules instead of exact-match-only
  intersection.

## [0.14.0] - 2026-06-30

### Added
- Compatibility imports through `native_agent_runner` and the legacy `native-agent`
  CLI alias, so existing local integrations can migrate incrementally.
- Central identifier and environment helpers for the Monoid namespace migration.

### Changed
- Project, package, repository, docs, and examples now use **Monoid Agent Kernel**
  branding.
- Python distribution name is now `monoid-agent-kernel`; import new code from
  `monoid_agent_kernel`.
- Current wire and durable artifact identifiers now emit `monoid.*` values.
  Readers and validators continue to accept legacy `native-agent-runner.*` values.
- Environment variables now prefer `MONOID_*` names. Existing `NAR_*` names are
  accepted during migration.
- Token issuer, audience, and header values now use Monoid identifiers while
  accepting legacy values during migration.

## [0.13.0] - 2026-06-29

### Added
- **OutputValidator** — developer-supplied validation of the final response with a
  bounded re-prompt loop. Register via `AgentLoop(output_validators=...)`; validators
  run **default-on**, opt out per-run with `OutputValidatorBinding(enabled=False)`. A
  rejection re-prompts the model with the validator's feedback, bounded by the new
  `RunLimits.max_output_retries`. Adds `AgentRunResult.final_output` /
  `outputs[validator_id]` / `output_as(Model)`, the `output.validator.*` event family
  (satisfied / validation.failed / exhausted / error / skipped), OTel span events, and
  a Studio + backend (`RunnerBackend(output_validators=...)`) seam.
- `ModelTurn.stop_reason` promotion: a provider refusal or truncation now settles as
  `output_refused` / `output_truncated` instead of a generic "neither text nor tool
  calls" model error.

### Changed
- The settle path is now a pure `_decide_settle` (classification) plus a single
  `_apply_settle` (state mutation + events + Suspension), and the four run.finish
  metadata fields collapsed into one `pending_finish` value — a behavior-preserving
  refactor that makes the validation lifecycle a single atomic transition.

### Fixed
- OpenAI adapter: capture `response.incomplete` in the streamed turn so truncations and
  refusals carry the correct `stop_reason`.
- Backend: `status()` falls back to the terminal result's `final_output` for
  stream-driven runs; resilient `status.json` reads under a concurrent atomic replace.

## [0.12.0] - 2026-06-27

### Added
- `AgentLoop.from_tools(spec, adapter, tools)` — one call to run with custom
  `@tool`/`ToolSpec` objects (auto-wraps a provider and generates their bindings),
  plus a runnable `examples/custom_tool_quickstart.py`.
- `AgentLoop.validate(config)` / `collect_runtime_config_issues()` — pre-run config
  validation that collects **all** problems as readable messages instead of raising
  on the first.
- Curated `contracts.core` namespace (the ~9 must-know names), a
  `monoid_agent_kernel.tool_ids` constants module, and `list_builtin_tools()`.
- `ToolBinding.for_tool("fs.read")` one-token bindings and bare-string `ref`.
- `monoid studio doctor` preflight (port / writability / API key / browser /
  OTel checks), a Studio README, and a first-run onboarding panel.
- `otel-export` extra (OTel SDK + OTLP exporter) so Studio's OTel toggle actually
  exports; a README "Observability" section, `examples/otel_tracing.py`, and a
  `docs/` index.
- Public failure events (`run.failed` / `turn.failed`) now carry
  `provider_error_code` and `http_status`, and Studio surfaces them.
- Studio: agent-to-agent (A2A) demo over the durable outbox→inbox fabric; inline
  image preview in the file viewer; open-source project files (contributing guide,
  code of conduct, security policy, CI workflow, environment template).

### Fixed
- MCP client: honor `tools/list` pagination (`nextCursor`) so large servers aren't
  truncated to page one, and reconnect once on a session-expiry (HTTP 404).
- OpenAI adapter: classify provider errors from the response body when the SDK
  exception carries no status — a streaming `429 insufficient_quota` was being
  masked as a generic `502 gateway_bad_response`.
- `fs.read`: on a binary/non-utf8 file, returns an actionable error pointing at
  `fs.read_media` (which reads images/PDFs under its own scope and authorization) instead
  of a bare "binary file" reject.
- Subagent/skill loaders warn on a duplicate id instead of silently dropping it.
- `[otel]` extra was api-only (a no-op); the Studio OTel toggle now has a working
  install path via `[otel-export]`.

### Changed
- Vendored KaTeX locally (woff2 only) so Studio honors its no-network promise.

## [0.11.0]
- Baseline at first public preparation. See the git history for the full
  evolution of the contracts, session/control protocol, capability leases,
  inbox/outbox fabric, durable checkpoints, and the Studio reference app.
