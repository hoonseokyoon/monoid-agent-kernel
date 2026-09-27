# v0.24.0 Stopped-Stream Drain Release Audit

> 상태: release candidate qualified on `develop`; release approval pending
>
> 작성일: 2026-09-27
>
> 기준 릴리스: `v0.23.0` / `888a90a`
>
> 개발 workflow: [`V0_23_DEVELOPMENT_WORKFLOW.md`](V0_23_DEVELOPMENT_WORKFLOW.md) §11–§12

v0.24 is a single-feature release. It adds an opt-in drain for a streamed model call stopped
cooperatively, so the usage the provider bills for that call is recorded instead of lost, and it
fixes one narrow default-path case in which a stop after the terminal chunk discarded that chunk's
usage. It adds no durable record, schema, codec, migration, or package dependency.

This audit records only evidence that was actually produced: the commands below were run on the
named sources, and the CI rows cite completed GitHub Actions runs. Rows without evidence say
"not run". `develop` → `main`, tag `v0.24.0`, GitHub release, and PyPI publish are a separate,
owner-approved stage.

## 1. release identity

| 항목 | 값 | 상태 |
|---|---|---|
| version | `0.24.0` (four places, §5.1) | passed |
| base release | `v0.23.0` tag → `888a90a38bd5e55e65fa9744aac1f10e2c905a49` | reference |
| feature PR | `#142` PR01, head `c9fbf1cf074e60a49e801b1566100689b16518bb` | merged |
| feature merge on `develop` | `442602d201ece5aeb09ef3cf669be709c6d86321` (2026-09-27T03:26:38Z) | merged |
| CI service re-pin | `#143`, head `3928ab30b643072a668ac69c80006c53d5625cf8` | merged |
| qualified `develop` source | `7569aef2727ce48fba5c24c5259376f1b74edda7` (2026-09-27T04:05:18Z) | passed, §2.3 |
| release-closure branch | `codex/v0.24-release-closure`, bump commit `b943bd7de0f81144b093f92a98fe4dc7ee0c6b67` | local gates passed, §6 |
| release PR / tag / publish | separate owner approval | pending |

## 2. authoritative evidence

### 2.1 feature PR (PR01, `#142`)

- L1 `V0.23 PR fast`: run
  [`36291124641`](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/36291124641),
  head `c9fbf1c`, conclusion `success`.
- L2 `V0.23 PR full`: run
  [`36291124660`](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/36291124660),
  head `c9fbf1c`, conclusion `success`; jobs `L2 core (Python 3.11)`, `L2 core (Python 3.12)`,
  `L2 required gate` passed; the service job was skipped (non-campaign branch, profile `core`; PR01
  touches no service adapter).
- Unresolved review threads on `#142` at audit time: `0` (GraphQL `reviewThreads`).

### 2.2 first `develop` L3 after the feature merge (red, infrastructure)

- run [`36291418703`](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/36291418703),
  head `442602d`, conclusion `failure`.
- 13 of 14 jobs passed. `V0.23 combined actual services` failed at `Start PostgreSQL and MinIO`
  with `minio Error unauthorized: access to the requested resource is not authorized`; the service
  tests did not start.
- Cause: the pinned `quay.io/minio/minio:RELEASE.2025-07-23T15-54-02Z@sha256:d249d1fb…` is no
  longer anonymously pullable (quay.io repository API and manifest both returned 401 on
  2026-09-27). Upstream MinIO archived its image distribution. Not a kernel regression.

### 2.3 qualified `develop` L3 (release candidate source)

- run [`36293292352`](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/36293292352),
  event `push`, branch `develop`, head `7569aef`, conclusion `success`, 14 of 14 jobs passed.
- Totals read from the job logs:

| job | result |
|---|---|
| lint (`ruff check src tests tools/v023_ci.py`) | passed |
| Fast + contract (Python 3.11) | `4521 passed, 22 skipped` |
| Fast + contract (Python 3.12) | `4521 passed, 22 skipped` |
| Serial integration (Python 3.11) | `1579 passed, 26 skipped, 4533 deselected` |
| Serial integration (Python 3.12) | `1579 passed, 26 skipped, 4533 deselected` |
| Coverage floor | passed; `Total coverage: 80.62%` (statements+branches) against the 80% floor; TOTAL row 44,025 statements / 7,194 missed, 14,046 branches / 2,419 partial |
| windows-latest smoke | `702 passed` |
| macos-latest smoke | `694 passed, 8 skipped` |
| Experimental DBOS activation recovery (3.11 / 3.12) | `196 passed` / `196 passed` |
| Install smoke (minimal / all-extras) | passed (exact wheel audit, install, import, conformance evidence smoke, Studio acceptance) |
| Studio UI build | passed |
| V0.23 combined actual services | `206 passed`; MinIO, PostgreSQL 16 and 18 healthy; artifact `v023-combined-evidence` (ID `10923525148`) |

The install-smoke jobs in this run built and audited the `0.23.0` wheel, because the version bump
lands on the release-closure branch. The `0.24.0` wheel evidence is local (§5.2); CI rebuilds it on
the release-closure PR and on its `develop` merge.

### 2.4 CI service re-pin (`#143`)

The combined service matrix now pulls MinIO from
`docker.io/bitnamilegacy/minio:2025.7.23-debian-12-r5@sha256:6dabb4a2088c9a79908de3bc05f4586c23ad2182c8908e7e3acbf61c1467fb20`,
Bitnami's frozen build of the same upstream commit as `RELEASE.2025-07-23T15-54-02Z` (binary commit
`7ced9663e6a7`), because upstream MinIO archived its image distribution. `tests/service/compose.yml`
and `tests/service/campaign-lock.json` changed; credentials, ports, healthcheck, profiles, and the
PostgreSQL pins did not. Run `36293292352` (§2.3) is the first green combined run on this pin.

- campaign lock canonical digest (`tools/v023_ci.py validate-lock`): `v0.23.0`
  `44e826410d2f91e334dfddf60530e38f0e11df02b90e7413bc1dbcdc6cddd2fd` →
  `7569aef` `5e536963c707778e9309936d5768edc367a6c49ece5c51223999dde46697b498` (MinIO re-pin only).
- qualification manifest `tests/service/qualification-v023.json`: unchanged, schema `2`, SHA-256
  over the committed bytes `f75e3d624556a5b5f36ee9f1229fd7816802344f746f2df13fcb0374adadfb84`.

## 3. release result

| boundary | release result | qualification |
|---|---|---|
| stopped-stream drain | opt-in `ModelCallRunner.abort_drain_s` / `current_abort_drain_s` and `AgentLoop.async_model_abort_drain_s` (keyword-only, default `0`, read when the stream opens); `should_abort`/`interrupt_turn` and the run token's `user_cancel` drain the same stream undelivered until it ends or `min(stop + budget, deadline)`, then close within the cancel grace | PR01 L1/L2, §2.3, §6 |
| stop classification | the stop still raises `ModelCallAborted` or `RunCancelled(user_cancel)`; a deadline that closes the window never reclassifies it as a timeout; `graceful_drain`, `deadline`, `host_shutdown`, and lease loss still end the call at once unless a `user_cancel` drain already started | PR01 L1/L2, §2.3 |
| drained usage | carried as the exception's provider-usage stamp onto the failed receipt, the attempt log, run totals, `metrics.updated`, and the token budget, exactly once | PR01 L1/L2, §2.3 |
| `RunStream` early exit | with the drain on, the hard-cancel wait grows from 8 s to `8 s + drain + cancel grace`; unchanged with the drain off | PR01 L1/L2 |
| default path (drain `0`) | byte-identical close at the stop, except the D2 fix below | PR01 L1/L2, §2.3 |
| D2 fix | a streamed call stopped by `should_abort` after its terminal chunk was delivered now carries that chunk's usage on `ModelCallAborted` and its receipt | PR01 L1/L2 |

### 3.1 known limits (documented, shipped as-is)

- Subagent child and fork loops keep `async_model_abort_drain_s = 0`: a token `cancel()` during a
  child's streamed call still cuts it at once and reports no usage for that call.
  `interrupt_turn()` never reaches a child, which streams to completion and bills in full.
- The token keeps its first cause: once a `user_cancel` has turned a stop into a drain, a later
  `host_shutdown` (or `graceful_drain`/`deadline` cause) is not observed; only lease loss and the
  run deadline end that drain early.
- `RunStream`'s enlarged early-exit wait is read when the stream opens; changing the knobs mid-stream
  does not resize it.
- Durable lifecycle mode keeps classifying an abort as `dispatch_unknown`; drained usage rides only
  on the exception stamp (PR01 decision D3).

## 4. migration and durable-format impact

None.

- `git diff v0.23.0 7569aef -- src/` touches only `model_call.py`, `loop.py`, `loop_phases.py`,
  and `core/streaming.py` (plus `_version.py` on the release-closure branch). No change under
  `src/monoid_agent_kernel/conformance` (including `fixtures/compatibility-v1.json`),
  `adapters/`, `hosting/`, or `tests/service/qualification-v023.json` (`git diff --quiet` exit `0`).
- No schema, migration, codec, ledger row, or record version changed. The new knobs are
  keyword-only dataclass fields with a default of `0`, so positional construction is unchanged.
- The one behavioural change on the default path is the D2 fix. `docs/COMPATIBILITY.md` records it:
  the abort receipt, its attempt log, its `model_calls.jsonl` line, the run totals, and a
  `metrics.updated` event now carry usage that was previously dropped. An abort receipt could
  already carry usage from absorbed retries, so no reader meets a new shape.
- Rollback to `0.23.0` needs no data step.

## 5. compatibility and packaging qualification

### 5.1 version places

`pyproject.toml` `version`, `src/monoid_agent_kernel/_version.py` `FALLBACK_VERSION`,
`tools/release_wheel_audit.py` `EXPECTED_VERSION`, and the `CHANGELOG.md` heading
`## [0.24.0] - 2026-09-27` all read `0.24.0`; `tests/test_release_packaging.py` enforces the match
(`8 passed`, worktree `src` import path confirmed).

### 5.2 exact local artifacts (bump commit `b943bd7`)

- build: `uv build --sdist --wheel` (uv 0.8.22)
- wheel `monoid_agent_kernel-0.24.0-py3-none-any.whl`, SHA-256
  `29123712b448a1bb232bebcd34446c7e4a699c666733d03c351771ff6810ee4f`
- sdist `monoid_agent_kernel-0.24.0.tar.gz`, SHA-256
  `e83ade5709643dda57a65ad41f497891ab7b8f55dc38c04cdb0d3ddeb5df8172`
- `python tools/release_wheel_audit.py <dist>`: `release wheel audit passed:
  monoid_agent_kernel-0.24.0-py3-none-any.whl`
- `uvx twine check` (twine 7.0.0) on wheel and sdist: `PASSED` / `PASSED`

These digests identify the bump-commit source only. This audit is tracked and ships in the sdist,
so the tagged distribution's hashes belong to the release workflow's rebuild, not to this file.

### 5.3 isolated exact-wheel install (minimal, Python 3.12.11, Windows)

- `uv venv` + `uv pip install <wheel>` with no extras; `importlib.metadata.version` → `0.24.0`.
- `import monoid_agent_kernel, monoid_agent_kernel.hosting, monoid_agent_kernel.conformance`
  loads none of `psycopg`, `psycopg_pool`, `boto3`, `botocore`, `temporalio`, `dbos`, `openai`,
  `httpx`; `httpx` is absent from the environment.
- drain surface from the installed wheel: `ModelCallRunner.abort_drain_s` default `0.0`,
  keyword-only; `current_abort_drain_s` present; `AgentLoop.async_model_abort_drain_s` default
  `0.0`, keyword-only.
- cold start, `python -X importtime -c "import monoid_agent_kernel"`, five runs after the first
  compile: cumulative 1.206 s, 1.108 s, 1.174 s, 0.834 s, 1.047 s.
- installed conformance evidence smoke
  (`python -m monoid_agent_kernel.conformance.runner --harness
  monoid_agent_kernel.reference.conformance:create_minimal_harness ...` + `verify_conformance_report`):
  runner exit `0`, `verified=True`, `report_passed=True`, 1 evidence record.
- `monoid studio accept` from the installed package: exit `0`.
- all-extras exact-wheel install: not run locally; covered for the pre-bump tree by §2.3
  `Install smoke (all-extras)`.

### 5.4 compatibility and conformance gate (worktree, bump commit)

- `pytest -q -n 4 -p no:randomly tests/conformance/test_compatibility_fixtures.py
  tests/test_compatibility_ledger.py tests/test_release_packaging.py tests/test_public_surface.py
  tests/test_v023_ci_foundation.py tests/test_namespace_compat.py
  tests/conformance/test_import_boundaries.py`: `77 passed`
- `pytest -q -n 4 -p no:randomly tests/conformance`: `438 passed, 2 skipped`
- `python tools/v023_ci.py validate-lock`: valid, digest `5e536963…b498`
- `python -m compileall -q src tests tools`: exit `0`

## 6. validation matrix

| gate | command/workflow | result | evidence |
|---|---|---|---|
| lint | `uvx ruff@0.15.6 check src tests tools/v023_ci.py tools/release_wheel_audit.py` | All checks passed | local, `b943bd7` |
| whitespace | `git diff --check` | clean | local |
| full suite, run 1 | `pytest -q -n 4 -p no:randomly` (Python 3.12, Windows) | `2 failed, 6076 passed, 97 skipped` in 960.64 s | local, `b943bd7`, run concurrently with the wheel build and isolated install |
| run 1 failures, alone | the two node ids below | `2 passed`; their modules `41 passed` | local |
| full suite, run 2 | `pytest -q -n 4 -p no:randomly`, no concurrent load | `6078 passed, 97 skipped` in 704.63 s, exit `0` | local, `b943bd7` |
| release metadata | `tests/test_release_packaging.py` | `8 passed` | local |
| compatibility/conformance | §5.4 | `77 passed`; `438 passed, 2 skipped` | local |
| wheel audit | `tools/release_wheel_audit.py` | `0.24.0` passed | local §5.2 |
| twine check | `uvx twine check` | wheel + sdist passed | local §5.2 |
| minimal install/import/cold start | §5.3 | passed | local |
| Python 3.11/3.12 partitions, coverage, OS smoke, DBOS | `ci.yml` push | passed | run `36293292352` |
| PostgreSQL 16/18 + MinIO + Temporal | `ci.yml` combined actual services | `206 passed` | run `36293292352` |
| Python 3.11 local full suite | — | not run | CI covers 3.11 (§2.3) |
| all-extras local install | — | not run | CI §2.3 (pre-bump tree) |

Run 1 failures: `tests/conformance/test_durable_runner_profile.py::test_reference_backend_satisfies_durable_runner_recovery_metadata_profile`
(`ValueError: cannot update runtime config for a terminal run`) and
`tests/test_temporal_activity.py::test_slow_activation_binding_renews_from_the_remaining_lease_deadline`
(`monoid.activation_lease_lost`). Neither module changed since `v0.23.0`, both are timing-sensitive
under load, and both passed alone and in the unloaded full run. They are recorded as load flakes, not
as release defects.

## 7. explicit exclusions

- `ModelStreamOutcome(status="interrupted").usage` for observers
- classifying a drained durable abort as settled (D3)
- reference LLM gateway draining or metering upstream on client disconnect
- exposing the drain knob through `RunnerBackend`, Studio, or the CLI
- drain inheritance for subagent child and fork loops
- provider-specific disconnect billing policy

## 8. release delivery gate

1. PR01 `#142` passed L1/L2 and merged into `develop` as `442602d`.
2. The first `develop` L3 (`36291418703`) failed on the MinIO image pull only; `#143` re-pinned the
   image and merged as `7569aef`.
3. `develop` L3 `36293292352` passed 14 of 14 jobs on `7569aef`.
4. This release-closure branch bumps the version and adds this audit; its PR into `develop` repeats
   L1/L2 and its merge repeats L3 with the `0.24.0` wheel.
5. `develop` → `main` "release: v0.24.0" (`combined` profile), annotated tag `v0.24.0` on the
   `main` merge commit, GitHub release, and `publish.yml` → PyPI require separate owner approval.

## 9. sign-off

| 승인 | 상태 | 근거 |
|---|---|---|
| feature scope (PR01 D1–D3) | merged | `#142` |
| CI service pin | merged | `#143`, run `36293292352` |
| packaging/compatibility | qualified locally | §5 |
| `develop` L3 | passed | run `36293292352` |
| release-closure PR review/CI | pending | this PR |
| final release | pending | separate `develop` → `main`, tag, and publish approval |
