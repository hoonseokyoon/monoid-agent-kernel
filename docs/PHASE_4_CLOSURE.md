# Phase 4 Closure

Phase 4 closes the Reference backend structure pass. The public API remains one
stable facade, and the implementation is split into private services with
explicit context/port dependencies.

## Completion Criteria

Phase 4 is complete when these conditions hold:

- External callers use one stable `RunnerBackend` facade.
- Backend internals are split into responsibility-focused private services.
- Services depend on explicit context/port objects instead of the whole facade.
- `Core`, Core Helper Kit, and conformance surfaces do not require the Reference
  backend service layout or any product-specific storage/deployment choice.
- Generic conformance profiles use profile-specific harness case methods.
- Reference scenario names and fixture vocabulary stay inside
  `monoid_agent_kernel.reference.conformance`.

## Completed Structure

Phase 4 completed these cleanup tracks:

- Public surface cleanup for `contracts` and root package exports.
- Run lifecycle vocabulary cleanup around `SessionState`, `state`, and
  `terminal`.
- Test seam and CI readiness work for backend factories, spawned future cleanup,
  advisory `xdist`, and advisory coverage.
- `RunnerBackend` projection, command, session, open-session drive, preparation,
  execution, state mutation, loop factory, outbox dispatch, proposal, runtime
  config, job, and recovery logic extracted into private services.
- Service context/port typing and composition-root cleanup.
- Conformance fixture decoupling with profile-specific harness protocols and
  import-boundary guards.

## Facade Boundary

`RunnerBackend` remains the embedder-facing facade and composition root. It owns
process-level runtime concerns: shared event loop scheduling, spawned task
tracking, run-slot semaphore ownership, shutdown/drain, watchdog heartbeat, and
the public streaming seam.

Private services live under `monoid_agent_kernel.reference.backend`. They are
Reference implementation details. They are not stable contract exports, and
external implementations do not need to copy this service layout to satisfy the
contracts or conformance profiles.

`docs/RUNNER_BACKEND_RESPONSIBILITY_MAP.md` is the current detailed map for the
facade/service split.

## CI Status (updated for v0.18.0)

The required CI gates are:

- lint with `ruff check src tests`;
- parallel unit/contract tests on Python 3.11 and 3.12;
- deterministic serial integration tests;
- branch coverage with an 80% ratchetable floor;
- Windows and macOS platform-sensitive smoke tests;
- minimal and all-supported-extras installation smoke tests.

Every collected test receives exactly one primary tier (`unit`, `contract`, or
`integration`) through the repository tier policy. `slow`, `live`, and `serial`
remain orthogonal traits. CI publishes slow-test data and coverage XML so the
coverage floor can rise with the measured baseline.

## Remaining Flake Risk

The remaining flake risk is concentrated in areas with real concurrency or
external timing:

- threaded Studio/backend server shutdown paths
- watchdog and outbox redrive timing
- long-running shell/job cleanup
- CI-only `xdist` scheduling differences
- live-provider tests outside default gates

Integration tests remain serial because they exercise shared event loops, threaded
servers, and subprocesses. Unit and contract tests are the required xdist shard.
The coverage floor is a regression guard rather than a completeness claim and
should be raised deliberately as additional behavior moves behind reusable seams.

## Phase 4 Position

Phase 4 completes the structural goal:

- one stable external facade
- private services by responsibility
- explicit internal ports
- product-neutral Core/helper/conformance surfaces

Further work should target product behavior, Studio UX, CI hardening, or
transport adapters as separate tracks.
