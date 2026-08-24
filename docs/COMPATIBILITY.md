# Compatibility Ledger

This ledger is the release-facing inventory for stable public wire formats, durable artifacts,
namespace aliases, and source aliases. The machine-readable source is
`monoid_agent_kernel.core.compatibility.PUBLIC_ARTIFACT_COMPATIBILITY`; the table below is
checked against it in CI.

Experimental optional Reference profiles sit outside this stable inventory. v0.19.2 treats
`DbosResumeCommand`, `DbosRunReceipt`, and the DBOS-specific control envelope as experimental
Reference operational records for the finite-activation profile. Their compatibility belongs to
the DBOS profile rather than Core. Control workflow output uses the stable writer-only
`CommandReceipt` listed below. Keep the DBOS `application_version` stable while pending work must
recover, drain that work before an incompatible workflow change, and treat exported DBOS version
constants and the internal control-envelope version as local profile identifiers without a
rolling-reader guarantee. DBOS types and upgrade policy stay inside the optional Reference
profile.

## Reader policy

- `checked` readers distinguish loaded, migrated, missing, corrupt, and unsupported-version
  outcomes. Recovery can quarantine authoritative bad state while retrying store outages.
- `strict` readers enforce the protocol identifier and payload shape in code.
- `json-schema` readers validate with a published JSON Schema.
- `permissive` readers consume the payload shape without enforcing the version discriminator.
  Their listed versions are tested compatibility targets; the current implementation may also
  accept unknown identifiers.
- `writer-only` formats have a public producer and no public serialized reader contract.

`current_writer` is the default producer identifier. The machine-readable `active_writers` tuple
lists every identifier this release can emit; most artifacts contain only `current_writer`.
`Missing id accepted` records an existing compatibility behavior. New producers write
`current_writer` unless a documented variant selects another active writer.

## Versioned artifact inventory

<!-- compatibility-registry:start -->
| Key | Kind | Current writer | Reader policy | Supported readers |
|---|---|---|---|---|
| `capability-request` | wire | `monoid.capability-request.v1` | writer-only | None (writer-only) |
| `capability-lease` | wire | `monoid.capability-lease.v1` | strict; missing id accepted | `monoid.capability-lease.v1`<br>`native-agent-runner.capability-lease.v1` |
| `control-command` | wire | `monoid.control-command.v1` | strict; missing id accepted | `monoid.control-command.v1`<br>`native-agent-runner.control-command.v1` |
| `inbox-message` | wire | `monoid.inbox-message.v1` | strict; missing id accepted | `monoid.inbox-message.v1`<br>`native-agent-runner.inbox-message.v1` |
| `outbox-request` | wire | `monoid.outbox-request.v1` | strict; missing id accepted | `monoid.outbox-request.v1`<br>`native-agent-runner.outbox-request.v1` |
| `external-agent-envelope` | wire | `monoid.external-agent-envelope.v1` | strict | `monoid.external-agent-envelope.v1`<br>`native-agent-runner.external-agent-envelope.v1` |
| `llm-turn` | wire | `monoid.llm-turn.v1` | strict | `monoid.llm-turn.v1`<br>`native-agent-runner.llm-turn.v1` |
| `llm-turn-result` | wire | `monoid.llm-turn-result.v1` | permissive; missing id accepted | `monoid.llm-turn-result.v1`<br>`native-agent-runner.llm-turn-result.v1` |
| `terminal-outcome` | wire | `monoid.terminal-outcome.v1` | strict | `monoid.terminal-outcome.v1`<br>`native-agent-runner.terminal-outcome.v1` |
| `activation-command` | wire | `monoid.activation-command.v1` | strict | `monoid.activation-command.v1`<br>`native-agent-runner.activation-command.v1` |
| `activation-receipt` | wire | `monoid.activation-receipt.v1` | strict | `monoid.activation-receipt.v1`<br>`native-agent-runner.activation-receipt.v1` |
| `model-stream-live` | wire | `monoid.model-stream.live.v1` | strict | `monoid.model-stream.live.v1` |
| `web-search` | wire | `monoid.web-search.v1` | permissive; missing id accepted | `monoid.web-search.v1`<br>`native-agent-runner.web-search.v1` |
| `web-search-result` | wire | `monoid.web-search-result.v1` | permissive; missing id accepted | `monoid.web-search-result.v1`<br>`native-agent-runner.web-search-result.v1` |
| `web-fetch` | wire | `monoid.web-fetch.v1` | permissive; missing id accepted | `monoid.web-fetch.v1`<br>`native-agent-runner.web-fetch.v1` |
| `web-fetch-result` | wire | `monoid.web-fetch-result.v1` | permissive; missing id accepted | `monoid.web-fetch-result.v1`<br>`native-agent-runner.web-fetch-result.v1` |
| `web-context` | wire | `monoid.web-context.v1` | permissive; missing id accepted | `monoid.web-context.v1`<br>`native-agent-runner.web-context.v1` |
| `web-context-result` | wire | `monoid.web-context-result.v1` | permissive; missing id accepted | `monoid.web-context-result.v1`<br>`native-agent-runner.web-context-result.v1` |
| `checkpoint` | durable | `monoid.checkpoint.v1` | checked | `monoid.checkpoint.v1`<br>`native-agent-runner.checkpoint.v1` |
| `model-invocation` | durable | `monoid.model-invocation.v1` | checked | `monoid.model-invocation.v1`<br>`native-agent-runner.model-invocation.v1` |
| `backend-run` | durable | `monoid.backend-run.v1` | checked | `monoid.backend-run.v1`<br>`native-agent-runner.backend-run.v1` |
| `event` | durable | `monoid.event.v1` | json-schema | `monoid.event.v1`<br>`native-agent-runner.event.v1` |
| `transcript` | durable | `monoid.transcript.v1` | json-schema; missing id accepted | `monoid.transcript.v1` |
| `model-content` | durable | `monoid.model-content.v1` | json-schema | `monoid.model-content.v1`<br>`native-agent-runner.model-content.v1` |
| `model-calls` | durable | `monoid.model-calls.v1` | json-schema | `monoid.model-calls.v1` |
| `model-payloads` | durable | `monoid.model-payloads.v1` | json-schema | `monoid.model-payloads.v1` |
| `manifest` | durable | `monoid.manifest.v1` | json-schema | `monoid.manifest.v1`<br>`native-agent-runner.manifest.v1` |
| `workspace-base` | durable | `monoid.workspace-base.v1` | json-schema | `monoid.workspace-base.v1`<br>`native-agent-runner.workspace-base.v1` |
| `workspace-index` | durable | `monoid.workspace-index.v1` | json-schema | `monoid.workspace-index.v1`<br>`native-agent-runner.workspace-index.v1` |
| `proposal` | durable | `monoid.proposal.v2` | json-schema | `monoid.proposal.v2`<br>`native-agent-runner.proposal.v2` |
| `background-job` | durable | `monoid.background-job.v1` | json-schema | `monoid.background-job.v1`<br>`native-agent-runner.background-job.v1` |
| `public-background-job` | wire | `monoid.public-background-job.v1` | json-schema | `monoid.public-background-job.v1` |
| `task` | durable | `monoid.task.v1` | writer-only | None (writer-only) |
| `proposal-package` | durable | `monoid.proposal-package.v1` | json-schema | `monoid.proposal-package.v1`<br>`native-agent-runner.proposal-package.v1` |
| `approval` | durable | `monoid.approval.v1` | json-schema | `monoid.approval.v1`<br>`native-agent-runner.approval.v1` |
| `apply-result` | durable | `monoid.apply-result.v1` | json-schema | `monoid.apply-result.v1`<br>`native-agent-runner.apply-result.v1` |
| `failure` | durable | `monoid.failure.v1` | permissive; missing id accepted | `monoid.failure.v1`<br>`native-agent-runner.failure.v1` |
| `command-inbox` | durable | `monoid.command-inbox.v1` | strict | `monoid.command-inbox.v1` |
| `command-receipt` | wire | `monoid.command-receipt.v1` | writer-only | None (writer-only) |
| `conformance-report` | reference | `monoid.conformance-report.v1` | checked | `monoid.conformance-report.v1`<br>`monoid.conformance-report.v2` |
| `conformance-evidence` | reference | `monoid.conformance-evidence.v1` | strict | `monoid.conformance-evidence.v1` |
| `conformance-fixtures` | reference | `monoid.conformance-fixtures.v1` | strict | `monoid.conformance-fixtures.v1` |
| `studio-chat` | reference | `studio.chat.v2` | strict | `studio.chat.v1`<br>`studio.chat.v2` |
| `studio-chat-message` | reference | `studio.chat.message.v1` | permissive; missing id accepted | `studio.chat.message.v1` |
| `studio-trace-export` | reference | `studio.trace-export.v1` | writer-only | None (writer-only) |
| `studio-trace-export-compact` | reference | `studio.trace-export.compact.v1` | writer-only | None (writer-only) |
| `studio-model-content` | reference | `studio.model-content.v1` | strict | `studio.model-content.v1` |
<!-- compatibility-registry:end -->

`monoid.terminal-outcome.v1` is a content-free final-state envelope. It carries portable outcome,
retry, and interruption vocabularies plus opaque output/evidence addresses in bounded
`scheme:locator` form. It never carries a prompt, model response, reasoning item, replay payload,
or raw provider exception. Its strict reader rejects fields outside the versioned top-level schema.
A `dispatch_unknown` outcome permits only `after_reconciliation` or `forbidden` retry eligibility,
so an ambiguous paid call cannot be classified for automatic retry.
The `limited` kind is a terminal v0.22 outcome and permits only `forbidden`; it keeps exhausted run
limits distinct from a cooperative `paused` boundary. It was added inside the unreleased v0.22
contract window, so deploy the current strict reader before a writer that can emit it.

`monoid.activation-command.v1` is the strict, orchestrator-neutral identity for one admitted input
or control activation. It carries a canonical source-checkpoint digest, request digest, and bounded
opaque payload address. Private input content, attempts, timestamps, and orchestrator identifiers
stay outside the command, so retry and process replacement preserve the same identity.
`monoid.activation-receipt.v1` is the content-free operational copy of the boundary stored in the
canonical checkpoint receipt. It carries checkpoint identity, lifecycle state, event/stream
cursors, terminal reference, and portable outcome taxonomy. Hosts reconstruct it from durable
state after response loss; later checkpoint heads retain the original per-command boundary
identity. The boundary digest blanks its own nested digest field before hashing to avoid a
self-reference. Model output and raw exceptions never enter the receipt.

`monoid.model-invocation.v1` is the checked durable record for one revision of a logical model
call. Current and retained namespace readers distinguish malformed data from future versions. The
receipt is normalized metadata; model content, request bodies, endpoints, and raw exceptions are
refused. The top-level record, receipt, and usage object each accept a closed versioned vocabulary;
unknown fields are private by default. Receipt key spellings are canonicalized. Digest, identifier,
taxonomy, timestamp, numeric, boolean, and usage values each have bounded typed validation; a
receipt request digest must equal the invocation request digest. Settled success points to a private
result blob through a bounded `scheme:locator` address, and ambiguous dispatch has no automatic
retry evidence. The `evidence_policy` enum records `passive`, `required`, or `outbox` delivery from
the first reservation through every later revision and retry. The checked reader accepts the earlier
v0.22 prerelease `requires_evidence` boolean and maps `false` to `passive` and `true` to `required`;
the canonical writer emits only `evidence_policy`. Deploy the current checked reader before this
writer because the earlier prerelease reader used a closed top-level vocabulary.

`monoid.checkpoint.v1` adds optional `last_model_invocation`, `interruption_cause`, `plan`,
`pending_finish`, and `pending_tool_loads` fields. Its optional `last_suspension` object also carries
`interruption_cause`. A v0.21 checkpoint omits these fields and restores them with empty/default
values. Current readers accept the absent suspension cause; current writers emit the normalized
typed value when an interruption has one. The writer keeps the checkpoint version and emits an
explicit field projection copied through the iterative portable JSON normalizer.

The v0.19.2 conformance rollout keeps the default external report writer on v1 and adds an opt-in
v2 evidence path after deploying its checked reader. Retained v1 reports migrate into the v2 typed
model with provenance explicitly marked unavailable. `--evidence-dir` emits v2 when an enhanced
adapter supplies retained evidence; consumers of that output need a v2 reader.

Source locations and format-specific notes are available through
`compatibility_registry()`. Integrators can serialize that result directly as JSON.

## Namespace, Python, CLI, and environment aliases

<!-- compatibility-aliases:start -->
| Surface | Current | Compatibility alias | Behavior |
|---|---|---|---|
| `python-package` | `monoid_agent_kernel` | `native_agent_runner` | deprecated; the alias package and submodules resolve to the current package and emit `DeprecationWarning` on import. |
| `cli-entry-point` | `monoid` | `native-agent` | deprecated; both entry points invoke `monoid_agent_kernel.cli:main`. New automation uses `monoid`. |
| `identifier-namespace` | `monoid.*` | `native-agent-runner.*` | compatibility; artifact-specific support appears in the version table. Writers emit `monoid.*`. |
| `environment-prefix` | `MONOID_*` | `NAR_*` | deprecated; `env.getenv` prefers the current name and falls back to the legacy name. |
| `token-issuer` | `monoid` | `native-agent-runner` | compatibility; Reference token validation accepts both issuers. |
| `token-header-type` | `MAK` | `NAR` | compatibility; newly issued tokens use `MAK`, and Reference validation accepts both header types. |
| `backend-audience` | `monoid.backend` | `native-agent-runner.backend` | compatibility; Reference token validation accepts both audiences. |
| `task-callback-audience` | `monoid.task-callback` | `native-agent-runner.task-callback` | compatibility; Reference token validation accepts both audiences. |
<!-- compatibility-aliases:end -->

## Deprecation policy

The project is pre-1.0, and every public compatibility change still requires an explicit
changelog entry and ledger update.

- The Python package, CLI, environment, token, and protocol namespace aliases remain available
  throughout the 0.x line. A future removal requires a major release and deprecation notice in
  at least two preceding minor releases.
- A durable reader alias remains until operators have a documented migration path for every
  retained artifact. A major release alone does not justify stranding checkpoints or run
  metadata.
- A reader-version removal requires fixtures for the last supported version, an upgrade path,
  and a release-note callout.
- Tightening a permissive reader is a compatibility change. Introduce strict parsing with
  compatibility fixtures and staged release notes.
- Writer-only surfaces can gain a reader contract additively. Consumers must treat their
  serialized representation as unstable until the ledger records reader support.

## Mixed-version operation

Use reader-first deployment. Deploy software that reads every currently stored or transmitted
version before any component starts writing a new version.

| Combination | Support |
|---|---|
| New reader, old writer using a listed supported version | Supported. |
| Old reader, new writer using the same schema version | Supported when the payload still satisfies the old reader's documented shape. |
| Old reader, new writer using a higher schema version | Unsupported unless the older release explicitly lists that version. |
| `monoid.*` and `native-agent-runner.*` peers at the same listed version | Supported for rows that list both identifiers. |
| Strict reader and unknown future version | Rejected without interpreting the payload. |
| Checked durable reader and unknown future version | Reported as `unsupported_version`; Reference recovery writes a diagnostic failure bundle. |
| Permissive reader and unknown future version | Behavior is unspecified; do not rely on acceptance. |

Keep Reference backend, Studio, LLM gateway, and Web gateway versions close during rolling
deployments. The strict LLM turn request is the limiting gateway edge. Canary a complete turn,
a web call, control delivery, and checkpoint resume before advancing the rollout.

## Upgrade playbook

1. Inventory schema and protocol identifiers in retained run roots, shared checkpoint stores,
   gateway clients, and exported Studio transcripts.
2. Confirm every observed identifier appears in `supported_readers` for the target release.
3. Quiesce long-running writers when the release changes a durable schema. Snapshot the run
   root and shared checkpoint/metadata store as one recovery unit.
4. Deploy readers first: gateways and backend recovery workers before clients and run workers.
5. Resume a non-terminal checkpoint canary and verify its run metadata, event sequence, queued
   messages, hosted tasks, and blob references.
6. Enable new writers. Confirm each emitted identifier appears in the registry's `active_writers`;
   use `current_writer` for the default producer path.
7. Retain the pre-upgrade snapshot until recovery, Studio projection, proposal verification,
   and gateway smoke checks pass.

Schema migrations run in memory on deep copies. Canonical writers persist the current
`monoid.*` identifier after a migrated artifact is accepted. Operators should preserve the
original backup until the upgraded run reaches a terminal state.

## Rollback playbook

Rollback safety depends on whether upgraded writers emitted a version the old release cannot
read.

1. Stop new run admission and drain or pause active writers.
2. Inspect artifacts written since upgrade. A writer-version increase marks the rollback as a
   data rollback, not a binary-only rollback.
3. For a binary-only rollback, confirm every new artifact still uses a version supported by the
   old release, then restore the old services together.
4. For a data rollback, restore the coordinated run-root and shared-store snapshot. Preserve the
   upgraded copy for diagnosis. Never rewrite signed proposal packages, approvals, apply results,
   or content-addressed checkpoint blobs in place.
5. Reissue short-lived tokens after a service rollback and verify both current and legacy
   issuer/audience acceptance as applicable.
6. Resume one checkpoint and complete one gateway turn before reopening admission.

Rolling back only the local run directory can desynchronize `run.json` from shared checkpoint
metadata. Rolling back only the checkpoint database can point `LATEST` at missing manifests or
blobs. Treat both stores as one backup boundary.

## Schema changes and existing runs

| Artifact class | Required evolution behavior |
|---|---|
| Checkpoint and backend run metadata | Register an ordered, pure migration before changing the writer. Preserve unknown fields where possible. Recovery must distinguish corrupt, unsupported, and transient store failures. |
| Append-only events, model content, and Studio chat JSONL | Readers must handle every retained record version. A file can contain records written by different releases. Keep record-level version checks. |
| Manifest, workspace snapshots, and indexes | Bump the version when a strict schema changes incompatibly. Existing run directories remain readable through the listed old-version schema or an explicit migration. |
| Proposal packages, approvals, and apply results | Content participates in hashes and approval identity. Generate a new artifact after a shape change; never mutate an existing signed or hashed artifact. |
| Hosted task and background-job projections | Recovery state lives in checkpoints. Projection schema changes must preserve operator visibility and must not be treated as checkpoint migrations. |
| Wire requests and responses | Deploy accepting readers before emitting the new version. Unknown versions fail closed at strict boundaries. |

`model-content.jsonl` is optional for retained run directories. Each new sidecar record carries
its own `monoid.model-content.v1` identifier, and readers also accept the legacy
`native-agent-runner.model-content.v1` namespace. When the sidecar is enabled during this
compatibility window, settled text is written to both it and `transcript.jsonl`; hydration reads
the sidecar first and falls back to the transcript for any unresolved digest.

`model_calls.jsonl` is optional in the same way, and single-namespace: it has never existed under
`native-agent-runner.*`, so `monoid.model-calls.v1` is the only accepted reader version. A record
is a declared projection of the in-process `ModelCallReceipt` rather than its serialization, so
the two shapes are deliberately not interchangeable — a recorded line does not round-trip through
`ModelCallReceipt.from_json`, which would supply transport defaults the call never ran under.
Adding a field to `ModelCallReceipt` therefore does not change this artifact; adding one *here*
is a schema change like any other, because `additionalProperties` is false.

`model_payloads.jsonl` follows the same two rules (optional; single-namespace, literal enum) and
adds a third that is this artifact's whole contract: every `model_request` record must reassemble
to the exact preimage of its `request_digest`, and `monoid validate` recomputes that per record —
resolving chunk references from inline records and the `model_payloads/` directory, re-encoding,
and comparing hashes. Unreferenced files in the chunk directory are not integrity issues (a
crashed write may orphan one) — reclaiming them is `monoid gc`'s job: report-only by default,
deletion under `--apply`, nothing younger than `--min-age-s` ever touched, and never run beside a
live writer of the same run directory. A referenced chunk that fails its hash, or a request
record that does not reassemble, is an integrity issue, and the collector preserves the
validator's verdict by construction — it deletes only what no record resolves, so
`monoid validate` reports the same issues after a sweep as before it. The record kinds share one
`oneOf` schema the way `model-content.v1`'s four kinds do.

A checkpoint schema bump affects every non-terminal run. The release that first writes the new
version must also read the previous version and restore its message queue, inbox dedupe set,
hosted tasks, continuation handle, runtime limits, and blob references. Keep that previous-version
reader for the documented deprecation window.

The v0.18 `monoid.backend-run.v1` writer adds `metadata_generation` without changing its schema
identifier. New descriptors start at generation one and increment on every committed update.
Recovery compares the local and shared copies, selects and repairs from the higher generation,
and reports equal-generation divergence as `corrupt`. When both historical copies omit the field,
the local descriptor retains authority. Older readers ignore this additive field.

The v0.20 path-pattern writer keeps a literal leading `!` unchanged and adds
`"path_pattern_encoding": "monoid.literal-bang.v1"` to the containing policy or tool scope.
Pre-v0.20 writers stored the literal without that marker in the same v1 artifact identifiers.
Durable readers therefore migrate the unmarked bare spelling in `manifest.v1`, `backend-run.v1`,
`checkpoint.v1`, and retained `command-inbox.v1` runtime-config payloads. Fresh operator
configuration rejects a bare `!`; `\!` remains its explicit literal spelling. An unmarked legacy
`\!` retains its old literal-backslash/PurePath meaning and is never widened to `!`.
The v0.21 `monoid.llm-turn.v1` writer adds `generation` and `output_schema` to the request and
`generation_applied` / `schema_applied` / `reasoning_applied` to the response and terminal
stream frame without
changing either protocol identifier (the `metadata_generation` precedent). Every new key is
present only when the caller configured the feature, so traffic that does not use it keeps its
exact previous wire shape (for `reasoning_applied`, "configured" means a non-default
`ModelConfig.reasoning` — the codec-default config demands no proof, and the explicit
`effort="default"` sentinel is configured and proven by an empty `{}` echo). Version skew
fails closed on the client: under the default
`generation.on_unsupported="fail"`, a server that does not echo is refused rather than allowed
to silently discard parameters; the reasoning proof is governed by its own family's
`reasoning.on_unsupported` the same way, so a deployment that wants reasoning display
preferences over a non-proving transport (the reference Studio's offline mode is one) states
`"omit"` on that field rather than losing the fail-closed default elsewhere. Separately, `ModelConfig.to_json` emits its `generation` block
only when configured — a generation-free runtime config keeps its pre-v0.21 `config_hash`, and
a *configured* one intentionally does not verify across mixed-version backend-run recovery
(configure generation only on a fully rolled fleet).

The same v0.21 writer adds `reasoning` to the response body and the terminal `turn_complete`
stream frame, again without changing the protocol identifier. Its conditionality is the mirror
image of the request keys above: those are *request*-conditional (present when the caller
configured the feature), while `reasoning` is **response**-conditional — present only when the
upstream actually produced artifacts — so traffic whose upstream produces none keeps its exact
previous wire shape either way. Skew is lossless in both directions: an old client ignores the
additive array, and a new client reading an old server (or a stream that ends without a terminal
frame) reads the absence as `()` through the permissive response reader. The only consequence of
that skew is that the provider-native reasoning round-trip does not happen on that hop — the loop
appends no reasoning block for an empty tuple, and the next turn is an ordinary untagged one.

The same writer adds `provider` beside it, on the same two carriers and again without changing the
protocol identifier. Its conditionality is a third one: not request-conditional like the echoes and
not answer-conditional like `reasoning`, but **upstream**-conditional — present only when the
upstream adapter the gateway built declares a `provider_name`, so a deployment whose upstream
declares none keeps its exact previous wire shape. Skew is lossless in both directions and the
default direction is the important one: a new client reading an old server sees no key, and an
absent key gates nothing, so it keeps trusting its own configured declaration exactly as it did
before — that is the pre-v0.21 behavior, unchanged. An old client reading a new server ignores the
key. When both ends are new and they *disagree*, the client drops that turn's relayed `reasoning`
and nothing else: the artifacts are unreadable by whichever provider is really behind the hop, the
turn's text, tool calls, usage and handle are untouched, and the client's own declaration keeps
naming the provider on the reasoning tag, the receipt, and every OTel `gen_ai.provider.name`. The
consequence of a mismatch is therefore the same one skew already has — no reasoning round-trip on
that hop — rather than a refusal or a changed attribution.

Failing *open* is scoped to **absence**, and only absence: a key that is not there proves
nothing, so nothing is refused. A key that IS there must be an array of objects, because that is
the only shape the replay path can hand back to a provider; a present-but-malformed value is
refused non-retryably as `gateway_bad_response` by both readers (body and terminal frame). One
skew case reaches that refusal in a way worth naming: the same protocol uses a `reasoning` key on
the **request** body for the reasoning *config* object, so a third-party gateway that echoes
request keys onto its response answers an array-valued key with an object and is refused. That is
the correct outcome — the value is unusable for replay either way — but the cause is not obvious
from the error, so check for a request echo before suspecting the upstream's artifacts. Since
B1 there are **three** `reasoning*` spellings on this protocol — the request's config object,
the response's artifact array, and the response's `reasoning_applied` echo object — and the
last two ride the same envelopes with different shapes, so a generic echo-the-request gateway
now trips the object/array mismatch in both directions; implement the three keys separately
rather than by prefix.

`provider` is held to the same rule by the same readers: absent is unknown and gates nothing, but
present must be a string, and a non-string is `gateway_bad_response` on both transports. Note what
that separates — a *malformed* attribution is refused, while a *disagreeing* one is not: the first
means the envelope cannot be read, the second means it was read and says something the client did
not expect, which is a fact about a deployment rather than a broken wire.

The v0.21 preview payload budget is a projection-only change: no event schema value, artifact
identifier, or wire key moves, exhaustion spends the `truncated_keys` / `truncated_items`
vocabulary that already existed, and `validate_run_dir` reads directories written on either side
of the change. The same release makes five Python-object ingress boundaries — a tool result's `content`,
`emit_artifact` metadata, a hosted task's request and result, and a model turn's tool-call
`arguments` — refuse values no portable JSON writer can spell. Four of them refuse scalars (`bytes`, integers past the 4300-digit decoder bound, arbitrary
objects) as a *classified call failure* (`tool_result_unportable` and its per-boundary siblings)
where such a value previously reached a writer that could not spell it and ended the run as
`internal_error`. Which writer depends on the boundary, and naming one for all of them was wrong: a
tool result and artifact metadata die at the transcript write, a hosted task's result at
`task.json`. The refusal fires only where that writer is actually reached — a duplicate report, or
a first report arriving after the task was cancelled, is answered as the no-op it already was,
before the payload is judged, because that path stores nothing and publishes nothing. Runs that
used to die now complete with a failed call the model can observe and correct; no retained artifact is convicted retroactively, because these values could never be
written to one — the refusal moves the failure earlier and names it, it does not change what any
reader accepts. Callers whose payloads arrive through a JSON parse are unaffected: the bounded
decoders never admitted these values in the first place.

Later in the same release those boundaries stop admitting unportable *containers* as well as
unportable scalars, under the same flag and the same per-boundary classification. A container that
is reachable from itself is refused — the normalizer preserved the cycle rather than failing on it,
so the copy reached `json.dumps` as `ValueError: Circular reference detected` and
`dataclasses.asdict` as `RecursionError` — and so is one taller than
`MAX_PORTABLE_CONTAINER_DEPTH` (64), measured as a height over the finished copy rather than as a
path depth, because the walk memoises and a shared subtree is charged at whichever reference it
reaches first. The bound is 64 and not the decoder's 512 for a measured reason: on CPython 3.11
`dataclasses.asdict` dies at 492 containers while the model-JSON decoder admits 512, so every depth
in [492, 512] cleared each gate it met and then killed the run at the checkpoint writer.

The fifth boundary is the one that reaches the checkpoint — a model turn's tool-call arguments ride
the assistant message into `state.messages` and out through `RunCheckpoint.to_json` — and it
classifies differently from the other four. It has no `*_unportable` code of its own; the refusal
escapes to `normalize_model_turn`, which answers `ModelAdapterError("model adapter returned a
non-portable response")`. That is a terminal classification, not a config-recoverable one: the loop
does not re-prompt on it, so what is bought is a named failure instead of a `_CheckpointPersistError`
from the persistence layer. One asymmetry is deliberate and is recorded here because it is
observable: when the same turn also carries a settled outcome — a `final_text`, or a `refusal` /
`length` stop reason — the surrounding leniency drops the offending call and keeps the paid answer
instead of raising. That leniency predates the bound. The call could not have run anyway (a settled
answer wins in `AgentLoop`) and dropping it keeps exactly the value the bound exists to keep out of
the checkpoint, but nothing reports the drop.

One v0.21 change moves an existing wire *answer* rather than adding a key: when the reference
gateway's shipped OpenAI upstream refuses its provider's malformed payload, the HTTP answer is
now a non-retryable 502 `openai_bad_response` (carrying the billed `usage`), where the shapes
that used to escape unclassified answered 400 `gateway_bad_request` or 500 with
`retryable: true`, and the shapes refused without a code of their own answered 502
`gateway_bad_response` — the hop's own wire named for an upstream payload defect. A client that
retried on that 500 was re-buying tokens for a payload defect; a client that read the 400 as its
own bad request was mis-remediating; and one malformed body answered two different codes
depending on which transport read it.

Read the recoverability move literally, because this kernel's own client does. `AgentLoop`
(`_recoverable_turn_error`) reads `retryable` **first**, then `config_recoverable`, and only
then the status range — treating any 4xx as a *recoverable* turn failure (the turn fails, the
session survives, the caller can fix and resend) and an un-flagged 5xx as terminal. **Both**
answers that moved therefore move from a park to a terminal run failure, by different routes:
the 400 was recoverable on its status range and its replacement is a 5xx outside that range,
and the 500 was recoverable on `retryable: true` alone — which is now `false`, so the flag that
carried it no longer does. Either way a client that previously got a suspension it could resume
from now gets a `failure.json` and a terminal session. That direction is intended. It converges
the hop with in-process behavior, where a malformed upstream payload has always ended the run,
and no amount of resending the same call fixes a payload the upstream produced. The 500 half
stops a second kind of bleeding at the same time: `retryable: true` invited a client to re-buy
exactly the tokens the defect had already spent.

The third group changed its name, and some of its members gained a key. The shapes that already
answered 502 `gateway_bad_response` — refusals minted without a code of their own — were already
`retryable: false` and already terminal for this client, and none of them moved: same status,
same `retryable` / `config_recoverable`, same verdict from `_recoverable_turn_error`. Read that
as two sub-groups, because only the first is a pure rename:

- The roughly a dozen raised **inside** the adapter's two stamped regions (the body mapping and
  the stream-terminal construction) were already carrying their billed `usage` and their
  `provider_retried` before v0.21. For these, `error_code` changed and nothing else did — an
  honest attribution, not a new recoverability, and a client keyed to anything but the code sees
  no change.
- The ones raised **around** those regions joined the group in v0.21 and gained more than a
  name: the stream's per-frame field validators and the blocking path's `_coerce_response` used
  to escape with no cost and no retry evidence at all, and both now travel the same completion
  seam. So their 502 body can carry `provider_retried: true` where it always said `false`, and
  it can carry a `usage` object where it previously had **no `usage` key at all** — the writer
  omits that key when the failure cost nothing, so a turn the upstream billed for is a wire
  *shape* change on these shapes, not just a value change. A client that sums `usage` across
  failures will now count tokens on calls it used to count as free. That is the correction: they
  really were billed.

Both sub-groups keep `retryable: false` and the same terminal verdict, so nothing about
recoverability moved for either.

If you implement this answer in your own gateway, write `"retryable": false` **and** the
`error_code` explicitly into the 502 body. The client derives each of those from the status line
only when its own key is absent, and its retry gate (`_should_retry`) needs both: a bare 502
derives `retryable: true` *and* the error code `gateway_server_error`, which is in the default
`model.retry.retry_on`. So a body omitting **both** keys is re-sent until
`model.retry.max_attempts` (3) is spent — three attempts, two retries — re-buying exactly the
tokens this change exists to stop paying for. Writing either key alone already breaks that
conjunction; write both anyway, so the body states its verdict instead of leaving half of it to
a default derived from the status line.

Raw refusals from third-party adapters keep their previous arms and their stamped-usage
carriage, and v0.21 adds one field to them: both gateway error writers now read
`provider_retried` off the escaping exception on **every** arm rather than on the
`ModelAdapterError` arm alone, so an adapter that retried and then refused in its own exception
type reports `provider_retried: true` where the body always said `false`. The key was already
written unconditionally, so this is a value change on an existing key, not a shape change.

The v0.21 gateway error writer adds `config_recoverable` to the non-200 error body and the
terminal SSE `type: "error"` frame, again without changing the protocol identifier. Unlike the
request keys above it is written unconditionally, beside `retryable` and `provider_retried`,
because a reader must not have to tell "not config-fixable" apart from "a server that never
mentions it" — both mean `false`. Skew is symmetric and lossless in both directions: an old
client ignores the additive key and keeps deriving the classification from the 4xx status as
before, and a new client reading an old server's body defaults the field to `false`, which is
the value that server's failures already carried. `TRANSCRIPT_RECORD_SCHEMA`'s `model_turn`
branch declares the same field, which the writer had always emitted under an open
`additionalProperties`; no stored transcript changes and no reader has to migrate.

Both writers of `monoid.failure.v1` — the core's `run_dir/failure.json` and the reference
backend's — add `http_status`, written as `null` when the failure never reached a provider, and
`retryable` / `config_recoverable`, the classification the `run.failed` event beside them
carries. The artifact's reader policy is permissive and its consumers read keys off the JSON, so
an older bundle simply has no such key and a reader must treat "absent" and `null`/`false`
alike. The schema identifier is unchanged.

The durable park observation inside `monoid.checkpoint.v1` (`last_suspension`) gains
`provider_error_code` and `provider_retried`, and `turn.failed` / `run.failed` /
`TRANSCRIPT_RECORD_SCHEMA`'s `model_turn` branch gain the event fields that carry the same facts
(`provider_retried`, `provider_usage`, `retryable`, `config_recoverable`). All are additive:
the park reader defaults every absent key, so a pre-v0.21 checkpoint restores exactly as before,
and the event schemas grow optional properties without changing a `required` list. No checkpoint
schema version bump, and no reader has to migrate.

`status.json` additionally becomes a recovery input without any shape change: the Reference
backend's resume paths now read its terminal projection (via the same tolerant
`lifecycle_from_status_artifact` reader, so a legacy bare `status: "limited"` keeps its
pre-`state` terminal-limited meaning) to recognize a run that already closed, where before it
was observability only. A missing or unreadable artifact changes nothing — recovery proceeds as
it always did. Relatedly, a run cancelled at a park now commits an ordinary
`monoid.checkpoint.v1` park snapshot at the ack (`cancellation_requested`, an existing field)
and a terminal one at close; pre-existing readers need no migration.

`ModelRetryConfig` grows `layer` (`"adapter"` | `"kernel"`), naming the single owner of the
retry loop for a model call. `to_json` emits the key only when it departs the default, so a
config that never chose a layer serializes byte-identically to one written before the field
existed and keeps its runtime-config hash; `from_json` reads an absent key as `"adapter"`,
which is what pre-W7 configs meant. The request-identity projection already excluded the
whole `retry` block, so recorded replay keys do not move either way. A pre-W7 reader handed
a `"kernel"` config ignores the unknown key and behaves as `"adapter"` — it retries in the
adapter loop — which is the pre-W7 behavior, never a multiplication.

`ModelCallReceipt` grows `attempt_log` — one record per kernel dispatch (index, elapsed,
the failure taxonomy, that attempt's billed usage, per-attempt `provider_retried`, and
whether a streamed chunk had committed the call when the attempt settled) — and the
`model-calls.v1` ledger line carries it. Additive on both surfaces: `from_json` reads an
absent key as an empty log beside an intact `attempts` count, which is what every record
written before the field existed means; the ledger schema declares the key without
requiring it, so `monoid validate` still passes directories older writers filled; and a
present log that does not name every attempt exactly once is refused — that shape is a
writer bug, not a legacy to absorb, and `monoid validate` now says so too rather than only
the constructor. W7-4 converges the empty corner on the writer's side only: the writers omit
an empty log, so absence is the one spelling this build produces for "nothing itemized", while
a present `[]` — what every build between W7-1 and W7-4 wrote for that same value, at whatever
`attempts` the receipt carried — is still read as an empty log by `from_json` and still passes
`monoid validate`. The readers stay put on purpose. An empty log is legal on a receipt at any
count (the log is empty *or* complete, and its empty arm was never reserved for refused calls),
the projection emitted the key unconditionally, and `AgentRecorder.record_settled_call` is
public — so the previous build wrote `[]` beside a positive count for every receipt handed to
it without entries, a default `ModelCallReceipt()` and its `attempts: 1` first among them.
Refusing that pair would have convicted the directories those builds filled; every one of them
keeps validating. Leniency stops at the key for the fields an entry shipped with: those have
no writer predating them, so every one is required and a partial entry is refused instead of
completed from defaults. (W7-2 later adds `backoff_ms` to the entry; that key has
predecessors and follows the record-level absence rule — its own paragraph below.) Run totals (`metrics.json`, `state.total_usage`, the token budget, the child
roll-up) now read the settled receipt's usage, which folds spend from attempts a kernel
retry absorbed — including on a run the boundary ended, where a cancelled, timed-out or
interrupted call's absorbed attempts now reach the totals as well; transcript `model_turn`
rows keep the turn's own usage, so a reader reconciles totals as transcript rows plus
absorbed spend. Old readers of the *totals* surfaces see only values that were always legal:
larger numbers, which every one of those readers already accepted.

The ledger is not one of those surfaces and must not be described as one. `model_calls.jsonl`
has **no released reader**: the artifact, its writer, `MODEL_CALLS_RECORD_SCHEMA` and the
`monoid.model-calls.v1` identifier all arrive in the same unreleased v0.21 line that adds
`attempt_log` — nothing at or below `v0.20.1` mentions any of them. So there is no population
of older readers to be compatible *with* here, and the additive-key reasoning that applies to
the open-`additionalProperties` surfaces above does not transfer: this record schema is closed
(`additionalProperties: false`), and by the rule stated in the `model_calls.jsonl` section
above, adding a key to it is a schema change like any other. The backward property that does
hold is the reader-side one already stated: the schema declares `attempt_log` without requiring
it, so a v0.21 validator still accepts lines a pre-`attempt_log` v0.21 build wrote.

W7-3 adds `idempotency_key` to `ModelCallReceipt` and to the same ledger line, under exactly the
reasoning of the previous paragraph and inside the same unreleased window: a schema change to a
closed schema, made where there is still no released reader to break, under the unchanged
`monoid.model-calls.v1` identifier. The schema declares the key without requiring it — the
`attempt_log` precedent — so `monoid validate` keeps passing directories pre-W7-3 v0.21 builds
filled, and absence on a line means exactly one thing: a writer that predates the field. On the
receipt's own JSON, `from_json` reads an absent key as `""` ("never keyed"), which is what every
record written before the field existed means. The key is deliberately excluded from the replay
key and from the payloads corpus, so no recorded replay identity moves; it is randomly issued
per call, so replaying a corpus issues fresh keys without touching the equivalence oracle, which
reads the payloads corpus and never opens the ledger. The value never reaches `status.json`,
`metrics.json`, or the event stream — carriage is receipt and ledger only, plus the
`Idempotency-Key` HTTP header on the gateway transport.

W7-2 adds `backoff_ms` to the attempt *entry*, on the receipt and the same ledger line — the
measured wait the kernel imposed before that dispatch, 0 on the first entry — inside the same
unreleased window and under the same closed-schema reasoning. It is the first key added to the
entry after the entry shipped, so the read-whole-or-refused rule gains its stated boundary:
keys the entry was born with stay required; `backoff_ms` is declared without being required,
absence meaning a line a W7-1 writer filled, and `monoid validate` keeps passing those
directories. `to_json` and the ledger projection omit the key when the value is unknown rather
than emitting null (a value no writer ever wrote) or 0 (a measurement never taken), so legacy
lines round-trip unchanged; a present null is refused by reader and schema alike. Timing stays
duration-only — no wall-clock instant joins the entry — and the recorded durations satisfy
`sum(elapsed_ms) + sum(backoff_ms) <= latency_ms` on one monotonic clock.

In the same window, every `pattern` across the artifact schemas stops accepting a trailing
newline. `jsonschema` evaluates `pattern` with Python's `re`, where `$` matches immediately before
a final newline, so `monoid validate` had been certifying `"<digest>\n"`, `"<timestamp>\n"`,
`"<event.type>\n"` and `"<key>\n"` on `event.v1`, `manifest.v1`, `model-calls.v1`,
`model-payloads.v1`, `workspace-*`, `approval` and `apply-result` lines. This is a **validation
tightening, not a schema-version change**: every identifier is unchanged, no writer in this
project has ever emitted such a value, and the only directories that stop validating are ones
carrying a value the rest of the kernel already refused. Third parties validating these schemas
with an ECMA-262 engine see no change at all — the new spelling is redundant there and
load-bearing only under Python's `re`.

**That window closes at v0.21.0.** Every argument above rests on the same premise — the artifact,
its writer, its schema and its identifier all arrived inside one unreleased line, so there was no
population of older readers to be compatible *with*, and a closed schema (`additionalProperties:
false`) could take a new key without the identifier moving. v0.21.0 is the release that creates
that population. From it onward `monoid.model-calls.v1` has a released reader, and so do the
schemas whose `pattern` spelling this section tightened; the next key added to any of them is an
ordinary compatibility event, decided by the rules at the top of this document rather than by the
absence of anyone to break. The reasoning above is kept as written because it was true when those
changes were made — this paragraph dates it, it does not retract it.

`status.json` and `metrics.json` grow the failure-classification keys their readers already had
event-side (`provider_error_code`, `http_status` — spelled `provider_http_status` on metrics —
`retryable`, `config_recoverable`, and on status.json while parked, `provider_retried`), and
`status.json` can now say `state: "paused"` for a cooperatively paused run. All additive under
each schema's open `additionalProperties`, declared in `STATUS_SCHEMA` / `METRICS_SCHEMA`
without an identifier change: absent keys on a pre-v0.21 artifact mean what those runs meant —
no live failure classified, no pause projected — and a reader must treat "absent" and the
default (`false` / `null` / `""`) alike.

`metrics.updated` grows the three priced sub-counts beside `reasoning_tokens`
(`cache_read_tokens`, `cache_creation_tokens`, `audio_tokens`), each emitted only when the
adapter reported one — an absent sub-count means "not reported", not zero. Both tenant-usage
JSON projections (the gateway's `/internal/llm/tenants/{id}/usage` and the backend's
`tenant_usage`) gain the same four keys. Neither is a versioned artifact in the inventory above
and neither has a serialized reader contract; the additions are new keys on a read-only
projection, so a consumer that ignores them stays correct.

The same durable readers keep pre-v0.20 `PurePath` matching for stored patterns that the current
grammar rejects, while fresh inputs remain strict. Runtime-config hashes omit only the
`path_pattern_encoding` representation marker at `tools[*].scope`; raw path arrays and every other
scope field remain hash-authoritative. A pre-v0.20 reader can ignore the additive marker and
recompute the same semantic hash, while retained legacy spellings keep distinct hashes.

The v0.18 writer adds four optional recovery fields to `monoid.checkpoint.v1`:

- `last_suspension` records the exact observable boundary represented by the checkpoint;
- `active_input` records an admitted input's identity, original source sequence, and
  `running`/`completed` phase across internal safety checkpoints;
- `applied_input_ids` records identities whose boundary is already committed;
- `applied_input_receipts` is the immutable identity-bound receipt ledger, so an old duplicate
  still returns its own boundary after newer inputs advance the run.

New readers default absent fields for older checkpoints. Older readers ignore the additive fields.
Any recovery adapter resuming a pending input must retain a reader that understands all four
fields. Dropping active-input state can admit a competing or stale activation; dropping the
identity or receipt ledger can redrive an applied input or return the wrong boundary. For the
experimental DBOS adapter, keep `application_version` stable while same-slot recovery of pending
workflow history is required. That operational version never replaces checkpoint schema/version
compatibility or the checkpoint receipt as semantic authority.

Checkpoints produced before v0.22 may encode `lease_lost` as a cancellation request. A v0.22
reader detects that value before bootstrap and revokes the activation's
`ActivationWriteAuthority`, leaving recorder, workspace replay, task restore, and extension
callbacks unopened. The public `CancellationToken.cancel()` API receives only operational causes.
Checkpoint validation rejects `cancellation_requested=true` when its cause is
`provider_failure`, `validation_failure`, or `unknown`; those values describe outcomes and cannot
drive execution cancellation. Direct `AgentLoop.restore()` applies the same check before
bootstrap.
The Reference recovery service also resolves the registered activation record before its first
authority-sensitive loop call, so a concurrent revocation still unregisters and discards the
stale activation.
