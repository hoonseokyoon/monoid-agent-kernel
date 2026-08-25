# v0.23.0 Production Adapters Release Audit

> 상태: implementation campaign qualified; release approval pending
>
> 작성일: 2026-08-23
>
> qualification 갱신일: 2026-08-25
>
> 범위 승인일: 2026-08-23
>
> 기준 릴리스: `v0.22.0` / `6b4bd9f`
>
> 구현 계획: [`V0_23_IMPLEMENTATION_PLAN.md`](V0_23_IMPLEMENTATION_PLAN.md)
>
> 개발 workflow: [`V0_23_DEVELOPMENT_WORKFLOW.md`](V0_23_DEVELOPMENT_WORKFLOW.md)

v0.23은 PostgreSQL, S3-compatible ObjectStore, Temporal, durable private stream을 하나의
production hosting path로 완성한다. 이 문서는 구현 qualification과 release delivery를 분리해
기록한다. 구현 qualification과 `develop` 통합은 실제 PostgreSQL 16/18, pinned MinIO, Temporal
local server, Python/OS/package matrix에서 완료됐다. `develop` → `main`, tag, GitHub release,
PyPI publish는 별도 owner 승인 단계다.

## 1. release identity

| 항목 | 값 | 상태 |
|---|---|---|
| version | `0.23.0` | passed |
| integration baseline | `788a33ea2399573719050c68ed5cc3f0d8ee54f8` | passed |
| reviewed PR13 source | `461605b62c924b8b2297d2692d984ebfbb7d635a` | passed |
| final `develop` source | `510ffa8def8a5cae50136dbcf2d2aed2625be9cc` | passed |
| integration PR | `#138`, `codex/v0.23-production-adapters` → `develop` | merged |
| release PR/tag/publish | 별도 owner 승인 단계 | pending |
| Python | 3.11, 3.12 | passed |
| PostgreSQL | 16.15, 18.6 | passed |
| ObjectStore | pinned MinIO S3 protocol | passed |
| Temporal | SDK 1.31.0, CLI 1.8.2, embedded server 1.31.2 | passed |

Exact source identity는 GitHub L2/L3 evidence artifact가 소유한다. PR13 reviewed head, 최종
integration PR head와 merge ref, `develop` merge commit을 각 gate에서 artifact `head_sha`와
`merge_sha`에 대조했다.

## 2. authoritative evidence

### 2.1 campaign lock

- campaign lock: `tests/service/campaign-lock.json`
- lock SHA-256:
  `44e826410d2f91e334dfddf60530e38f0e11df02b90e7413bc1dbcdc6cddd2fd`
- qualification manifest: `tests/service/qualification-v023.json`
- manifest schema: `2`
- manifest SHA-256:
  `f75e3d624556a5b5f36ee9f1229fd7816802344f746f2df13fcb0374adadfb84`

The lock pins exact Python SDKs, container image digests, Temporal CLI archives, embedded server
version, PR profile ownership, and profile-filtered required tests. `tools/v023_ci.py validate-lock`
fails on missing PR profiles, invalid pins, manifest drift, unknown tests, or incomplete profile
coverage.

### 2.2 integration baseline

- PR12: `#136`
- integration commit: `788a33ea2399573719050c68ed5cc3f0d8ee54f8`
- GitHub Actions run: `32785036958`
- combined evidence artifact: `v023-combined-evidence`
- evidence `head_sha` and `merge_sha`:
  `788a33ea2399573719050c68ed5cc3f0d8ee54f8`
- coverage: 83.61% lines (`36,610 / 43,787`), 70.96% branches (`9,908 / 13,962`)

All 13 integration jobs passed: lint, Python 3.11/3.12 fast+contract, Python 3.11/3.12 serial,
coverage floor, Windows/macOS smoke, DBOS recovery, Studio assets, minimal/all-extras install, and
combined actual services.

### 2.3 final campaign closure

- PR13 Ready L2: run `32787663492`, head
  `461605b62c924b8b2297d2692d984ebfbb7d635a`, merge ref
  `876bb8c364c81a4fe5a4dcf08058bf80bf12c094`, profile `combined`
- PR13 merge: PR `#137`, integration commit
  `bcfb5e54b7831c05856de9743c6532eb82fa1e7f`
- PR13 integration L3: run `32788287342`, 13 jobs passed, evidence head/merge
  `bcfb5e54b7831c05856de9743c6532eb82fa1e7f`
- final integration review: PR `#138`, reviewed head `bcfb5e54b7`, major issues `0`, unresolved
  threads `0`
- final integration L2: run `32790257448`, head
  `bcfb5e54b7831c05856de9743c6532eb82fa1e7f`, merge ref
  `4d7df600dfcba5666bae737a3d5ee3b9380a8fcf`, profile `combined`
- `develop` merge: `510ffa8def8a5cae50136dbcf2d2aed2625be9cc`
- final `develop` L3: run `32790806355`, 14 jobs passed, evidence head/merge
  `510ffa8def8a5cae50136dbcf2d2aed2625be9cc`
- final coverage: 83.61% lines (`36,610 / 43,787`), 70.96% branches
  (`9,908 / 13,962`)

Every combined artifact carries manifest schema `2`, manifest digest
`f75e3d624556a5b5f36ee9f1229fd7816802344f746f2df13fcb0374adadfb84`, and campaign-lock digest
`44e826410d2f91e334dfddf60530e38f0e11df02b90e7413bc1dbcdc6cddd2fd`.

## 3. approved release result

| boundary | release result | qualification |
|---|---|---|
| package/import | adapter-free base import; explicit PostgreSQL/S3/Temporal extras | passed |
| writer authority | PostgreSQL DB-clock lease and monotonic generation | passed |
| PostgreSQL sink | complete `FencedRunSink` with canonical ambiguous readback | passed |
| ObjectStore | immutable checked single/multipart S3-compatible blobs | passed |
| association/GC | PostgreSQL run association and fenced dry-run-first GC | passed |
| admission | stable command identity and transactional PostgreSQL dispatch outbox | passed |
| activation | orchestrator-neutral finite activation and canonical receipt | passed |
| event/terminal | fenced execution bridge and first-writer terminal convergence | passed |
| Temporal | Signal-With-Start entity Workflow and finite threaded Activity | passed |
| stream | bounded PostgreSQL/ObjectStore durable stream v1 | passed |
| operations | migration/doctor/metrics/GC/runbook | passed |
| compatibility | versioned records, retained fixtures, replay history | passed |

## 4. PostgreSQL qualification

The actual PostgreSQL 16 and 18 matrix proves:

- fresh and idempotent migration, migration checksums, explicit apply, doctor, and advisory-lock
  serialization;
- reader and writer floors for forward additive schema operation;
- configurable schema operation and platform-stable migration bytes;
- first claim, same-token renewal, release, expiry, generation handoff, concurrent winner, and stale
  writer rejection;
- complete reusable fenced sink conformance across checkpoint, invocation, event, evidence, terminal,
  and referenced private blobs;
- fence-first precedence, monotonic checkpoint heads, terminal first-writer-wins, atomic settled
  invocation/evidence outbox, and canonical readback after commit-response ambiguity;
- durable admission ordering, dispatch claiming, response-loss reconciliation, duplicate delivery,
  activation binding, and terminal-run convergence;
- aggregate-only, read-only, repeatable-read operational snapshots taken at one database time and
  MVCC boundary.

Qualification nodes:

- `migration_rolling`
- `postgres_authority_sink`
- `operations`

The production runtime performs no automatic migration. Deployment uses a dedicated migration
identity, reviews `plan()`, calls `apply()`, then requires doctor and every store readiness check.

## 5. ObjectStore and garbage-collection qualification

Pinned MinIO exercises the S3 protocol boundary with these results:

- conditional single PUT and multipart completion converge only after digest and size verification;
- 409/412 races and response loss follow bounded reconciliation;
- checked reads classify missing, size mismatch, and digest mismatch;
- PostgreSQL run association gates run-scoped resolution;
- object-first upload followed by PostgreSQL rollback creates no canonical association;
- cross-run access and stale-writer publication fail closed;
- version inventory, conditional version delete, incomplete multipart inventory, and abort are
  bounded operator primitives;
- GC plans first, rechecks generation and association under the digest lock, and records apply
  receipts;
- association/GC and stream-publication/GC races converge to the two safe outcomes.

Qualification node: `objectstore_gc`.

## 6. admission, Temporal, worker, and paid-call qualification

The combined path proves one ordered command lifecycle from PostgreSQL admission through Temporal
Signal-With-Start and finite Activity settlement:

- stable input identity and digest conflict detection;
- one PostgreSQL sequence lane per run with multi-dispatcher ordering;
- duplicate or lost transport responses converge through stored command and activation receipts;
- one deterministic per-run Workflow, buffered future sequences, duplicate signal convergence,
  Query status, Continue-As-New safe-point transfer, and saved v1 history replay;
- PostgreSQL generation ownership independent from Temporal attempt number;
- copied-context heartbeat, renewal, cancellation, graceful drain, host shutdown, and bounded worker
  cleanup;
- worker process kill followed by higher-generation takeover with zero repeated paid calls;
- pre-provider drain produces a resumable receipt;
- post-dispatch ambiguity produces `dispatch_unknown` and zero automatic paid retries;
- settled result recovery reuses private recorded bytes and makes zero provider calls.

Qualification nodes:

- `temporal_workflow`
- `worker_crash_drain`
- `paid_call_crash_matrix`

Temporal history carries opaque identifiers, digests, refs, counters, policy, and public-safe
taxonomy. PostgreSQL and ObjectStore carry private checkpoint and model bytes.

## 7. durable stream qualification

The durable private stream contract proves:

- `open → append batch* → seal` with one generation and monotonic UTF-8 byte cursor;
- same cursor and digest idempotency plus conflicting retry rejection;
- object availability before PostgreSQL chunk publication;
- exact reconnect from every committed cursor;
- stable reset receipt, typed old-generation reset, and typed cursor gap;
- prepared reset before provider entry on replacement generation;
- success/refusal settlement flush and full prepared-generation recovery after crash;
- unknown settlement on flush failure with automatic provider retry disabled;
- terminal, reset, append, seal, lease renewal, takeover, slow object upload, and GC races;
- private chunk exclusion from public events, operational metrics, receipts, and raw/decoded Temporal
  history.

Qualification nodes:

- `durable_stream`
- `privacy_combined`

## 8. compatibility and packaging qualification

The release candidate carries these compatibility guarantees:

- retained v0.21 checkpoint defaults and v0.22 additive checkpoint/invocation/interruption fields;
- checked missing, corrupt, and unsupported-version results;
- strict v1 admission, activation, Temporal policy/state/result/status records;
- packaged current fixtures for all nine v0.23 strict records;
- saved Temporal Workflow v1 history replay;
- machine-checked compatibility ledger rows and source-owned version constants;
- base imports with no psycopg, boto3, botocore, or temporalio load;
- independent `postgres`, `object-store-s3`, and `temporal` extras plus explicit `durable-host`;
- wheel inclusion of every PostgreSQL migration, production adapter module, hosting operation module,
  OTel operational bridge, and compatibility fixture;
- wheel exclusion of vendored platform SDK packages;
- minimal and all-extras CI install from the exact wheel audited in the same job;
- publish workflow order: build distributions, audit wheel, twine check, upload exact artifacts.

PR13 exact-tree local evidence:

- compatibility/package/public/ledger/CI helper: `69 passed`;
- `ruff`, compile, campaign-lock validation, and `git diff --check`: passed;
- wheel/sdist build, wheel audit, and `twine check`: passed;
- isolated minimal exact-wheel import, 19 fixtures, and Studio acceptance: passed;
- isolated all-extras exact-wheel install/import: passed;
- wheel SHA-256:
  `3ae6e207c14eb6b3b082d5cc3403f057b660500b02ae3b95bf00a26cd3c9f947`;
- sdist SHA-256:
  `26bb7cf36e1bec6ae5192c95fa682c6bc3868350850481e44a697639a6793a0b`.

PR13 reviewed source and final `develop` source have the same Git tree
`5dabc209432bec22d7b114fa598c2dc36ff0a216`; the recorded distributions therefore correspond to the
qualified release tree. CI rebuilt and audited the exact wheel independently in minimal and
all-extras jobs.

## 9. operations, security, and privacy qualification

The release supplies:

- bounded pool acquisition, transaction-local lock timeout, and statement timeout;
- aggregate outbox backlog/age/attempt, lease, invocation, terminal, and stream metrics;
- read-only PostgreSQL and S3 doctor surfaces;
- OTel observable gauges with fixed metric names and low-cardinality attributes;
- migration, rolling deploy, rollback, drain, takeover, corruption, object inventory, multipart
  cleanup, GC, backup, and restore runbooks;
- separate migration/runtime/ObjectStore-runtime/ObjectStore-admin/Temporal-worker role guidance;
- TLS, KMS, bucket policy, credential, and private operator-route guidance;
- dry-run-first GC and automatic-repair-disabled corruption response;
- seeded privacy scans over PostgreSQL rows, canonical receipts, operational snapshots, ObjectStore
  refs, raw and decoded Temporal history, public logs/events, and traces.

Backup tooling and cloud-specific credential tests belong to the host deployment. The release
runbook defines the required PostgreSQL cut, reachable ObjectStore versions, Temporal namespace
backup, restore order, checked-read sampling, stream digest verification, GC dry run, and unknown
paid-call reconciliation. Each production deployment rehearses this procedure with its selected
backup products before serving traffic.

## 10. validation matrix

| gate | command/workflow | result | evidence |
|---|---|---|---|
| lint/compile | final `develop` CI + PR13 local | passed | run `32790806355` |
| Python 3.11 full partitions | final `develop` CI | passed | fast+contract, serial |
| Python 3.12 full partitions | final `develop` CI | passed | fast+contract, serial |
| coverage floor | final `develop` CI | 83.61% lines, 70.96% branches | `coverage-xml` |
| PostgreSQL 16/18 | final combined actual services | passed | run `32790806355` |
| pinned MinIO | final combined actual services | passed | `objectstore_gc`, `durable_stream` |
| Temporal local server | final combined actual services | passed | `temporal_workflow` |
| combined kill/restart | final combined actual services | passed | `worker_crash_drain` |
| compatibility fixtures | PR13 exact tree | 19 fixtures, 69 focused tests passed | packaged fixture |
| wheel audit | PR13 exact tree + final CI rebuild | `0.23.0` passed | exact wheel |
| minimal/all-extras install | final `develop` CI | passed | exact audited wheel |
| Workflow history replay | final combined actual services | passed | checked-in v1 history |
| privacy scan | final combined actual services | passed | `privacy_combined` |

PR13 L2, integration L3, final integration L2, and final `develop` L3 all emitted qualified combined
evidence. The final evidence source is `510ffa8def8a5cae50136dbcf2d2aed2625be9cc`.

## 11. explicit exclusions

- separate `ModelCallActivity`
- deterministic AgentLoop Workflow replay and mid-step program counter
- general tool-effect journal
- multi-region active-active authority
- Redis/Kafka adapter and `LISTEN/NOTIFY` correctness
- automatic repair and operator UI
- common-skill-pipeline product integration
- cloud credentials in blocking public CI

These items require separate contracts, compatibility policy, qualification, and owner approval.

## 12. release delivery gate

The implementation campaign completed these steps:

1. PR13 latest head passed review convergence, L1, and `ci:combined` L2.
2. PR13 merged into `codex/v0.23-production-adapters` with merge commit `bcfb5e5`.
3. The integration branch passed the complete L3 matrix and emitted exact combined evidence.
4. Latest `origin/develop` was already an ancestor of the integration branch; no synchronization
   merge was required.
5. Integration → `develop` PR `#138` passed the Codex review cycle and required checks.
6. PR `#138` merged as `510ffa8`, and the resulting `develop` L3 matrix passed.

`develop` → `main`, tag `v0.23.0`, GitHub release, and PyPI publish require the separate release
approval defined by the project workflow.

## 13. sign-off

| 승인 | 상태 | 근거 |
|---|---|---|
| implementation scope owner | approved | A01~A14, 2026-08-23 |
| storage/conformance | qualified | PostgreSQL/ObjectStore actual-service manifest |
| Temporal/replay | qualified | local-server Workflow/Activity/history evidence |
| security/privacy | qualified | combined private-content scan and runbook |
| packaging/compatibility | qualified | fixture, exact wheel, install matrix |
| implementation campaign | complete | PR `#138`, `develop` run `32790806355` |
| final release | pending | separate `develop` → `main`, tag, and publish approval |
