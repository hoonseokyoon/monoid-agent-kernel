# Compatibility Ledger

This ledger is the release-facing inventory for stable public wire formats, durable artifacts,
namespace aliases, and source aliases. The machine-readable source is
`monoid_agent_kernel.core.compatibility.PUBLIC_ARTIFACT_COMPATIBILITY`; the table below is
checked against it in CI.

Experimental optional Reference profiles are outside this stable inventory. In v0.18,
`DbosResumeCommand` and `DbosRunReceipt` are experimental Reference operational records for the
finite-activation proof. Core compatibility excludes them. Keep the DBOS `application_version`
stable while pending work must recover, drain that work before an incompatible workflow change,
and treat their exported version constants as local profile identifiers without a rolling-reader
guarantee. DBOS types and upgrade policy stay inside the optional Reference profile.

## Reader policy

- `checked` readers distinguish loaded, migrated, missing, corrupt, and unsupported-version
  outcomes. Recovery can quarantine authoritative bad state while retrying store outages.
- `strict` readers enforce the protocol identifier and payload shape in code.
- `json-schema` readers validate with a published JSON Schema.
- `permissive` readers consume the payload shape without enforcing the version discriminator.
  Their listed versions are tested compatibility targets; the current implementation may also
  accept unknown identifiers.
- `writer-only` formats have a public producer and no public serialized reader contract.

`Missing id accepted` records an existing compatibility behavior. New producers must always
write the current identifier.

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
| `web-search` | wire | `monoid.web-search.v1` | permissive; missing id accepted | `monoid.web-search.v1`<br>`native-agent-runner.web-search.v1` |
| `web-search-result` | wire | `monoid.web-search-result.v1` | permissive; missing id accepted | `monoid.web-search-result.v1`<br>`native-agent-runner.web-search-result.v1` |
| `web-fetch` | wire | `monoid.web-fetch.v1` | permissive; missing id accepted | `monoid.web-fetch.v1`<br>`native-agent-runner.web-fetch.v1` |
| `web-fetch-result` | wire | `monoid.web-fetch-result.v1` | permissive; missing id accepted | `monoid.web-fetch-result.v1`<br>`native-agent-runner.web-fetch-result.v1` |
| `web-context` | wire | `monoid.web-context.v1` | permissive; missing id accepted | `monoid.web-context.v1`<br>`native-agent-runner.web-context.v1` |
| `web-context-result` | wire | `monoid.web-context-result.v1` | permissive; missing id accepted | `monoid.web-context-result.v1`<br>`native-agent-runner.web-context-result.v1` |
| `checkpoint` | durable | `monoid.checkpoint.v1` | checked | `monoid.checkpoint.v1`<br>`native-agent-runner.checkpoint.v1` |
| `backend-run` | durable | `monoid.backend-run.v1` | checked | `monoid.backend-run.v1`<br>`native-agent-runner.backend-run.v1` |
| `event` | durable | `monoid.event.v1` | json-schema | `monoid.event.v1`<br>`native-agent-runner.event.v1` |
| `manifest` | durable | `monoid.manifest.v1` | json-schema | `monoid.manifest.v1`<br>`native-agent-runner.manifest.v1` |
| `workspace-base` | durable | `monoid.workspace-base.v1` | json-schema | `monoid.workspace-base.v1`<br>`native-agent-runner.workspace-base.v1` |
| `workspace-index` | durable | `monoid.workspace-index.v1` | json-schema | `monoid.workspace-index.v1`<br>`native-agent-runner.workspace-index.v1` |
| `proposal` | durable | `monoid.proposal.v2` | json-schema | `monoid.proposal.v2`<br>`native-agent-runner.proposal.v2` |
| `background-job` | durable | `monoid.background-job.v1` | json-schema | `monoid.background-job.v1`<br>`native-agent-runner.background-job.v1` |
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
| `studio-chat` | reference | `studio.chat.v1` | strict | `studio.chat.v1` |
| `studio-chat-message` | reference | `studio.chat.message.v1` | permissive; missing id accepted | `studio.chat.message.v1` |
<!-- compatibility-registry:end -->

The v0.19.2 conformance rollout deploys the v2 checked reader before changing the external runner's
v1 writer. The reader migrates retained v1 reports into a v2 typed model with provenance explicitly
marked unavailable. The next writer step can then emit target and evidence references without a
reader gap.

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
6. Enable new writers. Confirm they emit the registry's `current_writer` identifiers.
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
| Append-only events and Studio chat JSONL | Readers must handle every retained record version. A file can contain records written by different releases. Keep record-level version checks. |
| Manifest, workspace snapshots, and indexes | Bump the version when a strict schema changes incompatibly. Existing run directories remain readable through the listed old-version schema or an explicit migration. |
| Proposal packages, approvals, and apply results | Content participates in hashes and approval identity. Generate a new artifact after a shape change; never mutate an existing signed or hashed artifact. |
| Hosted task and background-job projections | Recovery state lives in checkpoints. Projection schema changes must preserve operator visibility and must not be treated as checkpoint migrations. |
| Wire requests and responses | Deploy accepting readers before emitting the new version. Unknown versions fail closed at strict boundaries. |

A checkpoint schema bump affects every non-terminal run. The release that first writes the new
version must also read the previous version and restore its message queue, inbox dedupe set,
hosted tasks, continuation handle, runtime limits, and blob references. Keep that previous-version
reader for the documented deprecation window.

The v0.18 `monoid.backend-run.v1` writer adds `metadata_generation` without changing its schema
identifier. New descriptors start at generation one and increment on every committed update.
Recovery compares the local and shared copies, selects and repairs from the higher generation,
and reports equal-generation divergence as `corrupt`. When both historical copies omit the field,
the local descriptor retains authority. Older readers ignore this additive field.

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
