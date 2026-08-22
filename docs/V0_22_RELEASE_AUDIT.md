# v0.22.0 Release Audit

Status: PR 7 release candidate.

The v0.22 campaign adds portable production boundaries for durable model invocation, fenced
checkpoint and run sinks, evidence delivery policy, typed interruption, activation-scoped write
authority, and worker drain. The default standalone execution path remains in-process and opt-in
hosting stays dependency-light.

## Release scope

| Boundary | Release result | Evidence |
|---|---|---|
| Portable outcome | `TerminalOutcome`, retry eligibility, and interruption cause use closed v1 vocabularies. | `tests/test_outcome.py`, packaged compatibility fixture |
| Paid model call | Reservation, dispatch evidence, settlement, replay, and `dispatch_unknown` survive activation replacement. | `tests/test_model_call_runner.py`, `tests/test_agent_loop_durable_recovery.py` |
| Canonical storage | `WriterToken` owner/generation is checked atomically with checkpoint, invocation, event, evidence, and terminal mutation. | `run_fenced_run_sink_contract`, `tests/conformance/test_fenced_hosting_contract.py` |
| Evidence delivery | `passive`, `required`, and `outbox` policies preserve settlement authority and retry semantics. | `tests/test_agent_loop_durable_recovery.py`, `tests/test_model_call_runner.py` |
| Process-local authority | `ActivationWriteAuthority` revokes kernel-managed mutation surfaces independently from execution cancellation. | `tests/test_authority.py`, `tests/test_backend_activation.py` |
| Typed interruption | User cancel, drain, deadline, shutdown, and lease loss retain distinct portable causes. | `tests/test_cancellation.py`, `tests/test_session_drive.py` |
| Compatibility | v0.21 checkpoints and v0.22 additive checkpoints load through the packaged fixture; future versions fail closed. | `tests/conformance/test_compatibility_fixtures.py`, `tests/test_compatibility_ledger.py` |
| Package boundary | Root, hosting, and conformance imports stay lazy and platform-neutral; the wheel carries hosting, codecs, authority, and fixtures. | `tests/test_public_surface.py`, `tests/conformance/test_import_boundaries.py`, `tools/release_wheel_audit.py` |

## Release gates

- Version: `pyproject.toml`, `_version.py`, wheel metadata, and changelog all name `0.22.0`.
- Base dependencies: `click`, `jsonschema`, `pathspec`, and `pydantic`. DBOS remains an optional
  `reference-dbos` extra. PostgreSQL, Redis, and Temporal are absent.
- Root surface: core outcome and invocation contracts are stable root exports. Hosting contracts
  remain under `monoid_agent_kernel.hosting`.
- Conformance surface: external adapters import `FencedRunSinkHarness`,
  `FencedRunSinkHarnessFactory`, and `run_fenced_run_sink_contract` from
  `monoid_agent_kernel.conformance`.
- Install smoke: minimal and all-extras installations import root, hosting, conformance, provider,
  observability, MCP, and Reference DBOS modules. Minimal installation also runs retained-evidence
  conformance and Studio acceptance.
- Publish: the release workflow audits the exact wheel built with the source distribution before
  uploading that artifact to PyPI.
- CI: lint, Python 3.11/3.12 fast and contract shards, Python 3.11/3.12 serial integration,
  coverage floor, Windows/macOS smoke, DBOS recovery, Studio assets, and install smoke must pass.

## Compatibility and rollout

`monoid.checkpoint.v1` remains the checkpoint writer id. v0.22 adds model invocation and
interruption fields. A pre-v0.22 cancellation checkpoint can omit the cause; restore maps that
legacy cancellation request to `user_cancel`. A stored legacy `lease_lost` cancellation is migrated
to revoked writer authority before bootstrap.

`monoid.terminal-outcome.v1` and `monoid.model-invocation.v1` are new versioned artifacts. Their
accepted legacy namespace aliases remain read-only compatibility paths. Writers emit the `monoid.*`
identifier. The compatibility ledger defines rolling-reader and rollback behavior.

Rollout installs the new reader before enabling fenced mode, verifies adapter capabilities, runs
the fenced sink contract against the production transaction boundary, and then enables durable
invocation per workload. Rollback first drains v0.22 writers. A v0.21 process cannot interpret the
new invocation journal as an authority for paid-call recovery.

## Security and privacy

Public outcome, evidence, event, and compatibility records exclude raw prompts, model responses,
reasoning, replay payloads, provider exception text, credentials, and tenant content. Private replay
blobs stay content-addressed behind the host adapter. Capability declarations fail closed and never
substitute for storage enforcement.

## Explicit exclusions

- Temporal orchestration
- PostgreSQL/ObjectStore reference adapters
- Direct AgentLoop ownership of canonical event and terminal sinks
- Activation-specific complete workspace or run-directory isolation
- Forced termination of already-started shell, MCP, memory, or custom Python effects
- Arbitrary Python extension sandboxing
- Multi-process promotion of LocalFS or SQLite checkpoint stores

Those items remain host or future-release work. v0.22 guarantees that a revoked activation starts no
new kernel-managed mutation, canonical durable mutation validates its `WriterToken` atomically, and
external effects rely on fenced or idempotent adapters.

## Validation record

PR 7 completed the local release gates on 2026-08-23:

- `python -m compileall -q src tests tools`: passed.
- Focused compatibility, public-surface, packaging, ledger, and import-boundary suite: 46 passed.
- Full parallel suite: 5,744 passed and 90 skipped.
- `ruff check src tests tools/release_wheel_audit.py`: passed.
- Built `monoid_agent_kernel-0.22.0-py3-none-any.whl` and ran
  `tools/release_wheel_audit.py`: passed.

The PR CI repeats the platform, dependency, contract, coverage, install, and wheel gates. The final
integration PR repeats import, wheel, compatibility, full test, and CI validation after merging the
latest `origin/develop`.
