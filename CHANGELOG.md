# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and this project is
pre-1.0 (`0.x`): minor versions may include breaking changes, which are called
out in commit messages and here.

## [Unreleased]

### Added

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

- An adapter whose `open()` or `close()` raises now ends `monoid run` with a reported error instead of
  a bare traceback — a connection pool failing to construct or to tear down is the ordinary way in.
  Both calls sat below the handler that normalizes every other startup failure. The teardown case
  carried a second fault: the run's status and summary were echoed *after* the adapter scope unwound,
  and an exception from a cleanup callback replaces whatever is leaving the block, so a failing
  teardown silently swallowed the outcome of a run that had **completed**. The outcome is now echoed
  before the scope is released, so a cleanup failure costs the cleanup and not the result.
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
