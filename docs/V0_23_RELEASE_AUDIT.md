# v0.23.0 Production Adapters Release Audit

> 상태: approved cut 구현 진행 중; release evidence 없음
>
> 작성일: 2026-08-23
>
> 범위 승인일: 2026-08-23
>
> 기준 릴리스: `v0.22.0` / `6b4bd9f`
>
> 구현 계획: [`V0_23_IMPLEMENTATION_PLAN.md`](V0_23_IMPLEMENTATION_PLAN.md)
>
> 개발 workflow: [`V0_23_DEVELOPMENT_WORKFLOW.md`](V0_23_DEVELOPMENT_WORKFLOW.md)

이 문서는 v0.23 release candidate에서 실제로 수행한 검증만 기록한다. 계획, 예상 결과, 구현 PR의
부분 통과는 release evidence가 아니다. 각 gate는 command, commit SHA, workflow/artifact, 결과를
가질 때만 `passed`로 바꾼다.

## 1. release identity

| 항목 | 기대값 | 실제값 | 상태 |
|---|---|---|---|
| version | `0.23.0` | 미정 | pending |
| source commit | release candidate SHA | 미정 | pending |
| integration PR | v0.23 → `develop` | 미정 | pending |
| release PR | `develop` → `main` | 미정 | pending |
| tag | `v0.23.0` | 미정 | pending |
| wheel/sdist digest | exact published artifact | 미정 | pending |
| Python | 3.11, 3.12 | 미검증 | pending |
| PostgreSQL | approved minimum and latest stable major | 미검증 | pending |
| ObjectStore | pinned MinIO + S3 protocol qualification | 미검증 | pending |
| Temporal | approved SDK/local-server versions | 미검증 | pending |

## 2. approved release cut

이 표는 2026-08-23 승인된 구현 계획 A01~A14 disposition과 일치한다. `approval`은 범위 승인을,
`release evidence`는 release candidate의 실제 검증 상태를 뜻한다.

| boundary | planned result | approval | release evidence |
|---|---|---|---|
| package/import | base install remains adapter-free; optional PostgreSQL/S3/Temporal extras | approved | pending |
| writer authority | DB-clock lease and monotonic generation | approved | pending |
| PostgreSQL sink | full `FencedRunSink` production implementation | approved | pending |
| ObjectStore | immutable checked single/multipart S3-compatible blobs | approved | pending |
| admission | stable input/control identity and PG dispatch outbox | approved | pending |
| activation | orchestrator-neutral finite activation and canonical receipt | approved | pending |
| event/terminal | fenced execution bridge and terminal first-writer convergence | approved | pending |
| Temporal | Signal-With-Start entity Workflow and finite Activity | approved | pending |
| stream | separate batched durable stream v1 | approved | pending |
| operations | migration/doctor/metrics/GC/runbook | approved | pending |

## 3. PostgreSQL gate

### 3.1 migration and startup

- [ ] Fresh database applies every bundled migration in order.
- [ ] Reapplying migrations is idempotent.
- [ ] Migration checksum drift fails closed.
- [ ] Adapter construction never auto-migrates.
- [ ] `status`, `plan`, `apply`, `doctor` report the same installed schema.
- [ ] N and N-1 compatible readers open the forward schema during the documented window.
- [ ] Unsupported reader/writer floors fail before serving traffic.
- [ ] Configurable schema and default `monoid_kernel` both pass.

Evidence: pending.

### 3.2 writer authority

- [ ] First claim creates generation 1.
- [ ] Same-token renewal retains generation and extends expiry using DB clock.
- [ ] Wrong owner, stale generation, expired token, revoked token fail closed.
- [ ] Different owner claims only after expiry or authorized release.
- [ ] Renewal response loss revokes local write authority.
- [ ] Handoff races use independent connections and processes.
- [ ] Old writer publishes zero durable mutations after the new generation wins.

Evidence: pending.

### 3.3 fenced sink

- [ ] Full reusable `run_fenced_run_sink_contract` passes against actual PostgreSQL.
- [ ] Checkpoint, invocation, event, evidence, terminal mutations fence before idempotency.
- [ ] Same coordinate/same digest is idempotent; conflicting digest is rejected.
- [ ] Delayed lower checkpoint never regresses the head.
- [ ] Invocation lifecycle, retry identity, settled result reuse, unknown dispatch converge.
- [ ] Terminal is first-writer-wins across concurrent processes.
- [ ] Settled invocation and evidence outbox commit atomically.
- [ ] Commit-response loss reconciles by canonical readback.
- [ ] Reopen/restart distinguishes missing, corrupt, unsupported version.

Evidence: pending.

## 4. ObjectStore gate

- [ ] Single PUT uses conditional create and verified lowercase SHA-256 address.
- [ ] Multipart completion uses conditional create.
- [ ] Existing-object `412` converges only after digest/size verification.
- [ ] Concurrent `409` follows the documented bounded restart rule.
- [ ] Multipart ETag is never treated as content SHA-256.
- [ ] Every checked read verifies bytes and size.
- [ ] PostgreSQL association is required for run-scoped resolution.
- [ ] Cross-run association bypass fails.
- [ ] Object upload followed by PG rollback creates no canonical metadata.
- [ ] Orphan GC respects grace, generation snapshot, association recheck, deletion receipt.
- [ ] Incomplete multipart inventory and cleanup pass.
- [ ] Missing and corrupt objects fail with typed results and expose no private content.

Evidence: pending.

## 5. admission and activation gate

- [ ] Same input ID/digest returns the same handle or stored receipt.
- [ ] Same input ID/different digest conflicts before dispatch.
- [ ] Admission and PG dispatch outbox are one transaction.
- [ ] Dispatcher response loss and duplicate delivery apply input once.
- [ ] Per-run command ordering survives multiple dispatcher processes.
- [ ] Active input crash resumes the same command identity.
- [ ] Completed input returns canonical checkpoint receipt.
- [ ] Event bridge revokes authority on fenced result.
- [ ] Terminal bridge reads and returns the canonical winner.
- [ ] Projection/evidence failure triggers zero provider re-executions.

Evidence: pending.

## 6. Temporal gate

### 6.1 Workflow

- [ ] Signal-With-Start creates or targets one deterministic per-run Workflow.
- [ ] Duplicate and out-of-order command refs converge to PG admission order.
- [ ] Workflow history contains only approved content-free fields.
- [ ] Query returns operational refs and never becomes terminal authority.
- [ ] Continue-As-New waits for active handlers and Activity completion.
- [ ] Pending commands and latest receipt refs survive Continue-As-New.
- [ ] Saved histories replay under the release candidate Workflow code.
- [ ] Workflow/schema/build compatibility metadata is documented.

Evidence: pending.

### 6.2 Activity and worker

- [ ] Activity acquires PostgreSQL generation independently from Temporal attempt number.
- [ ] Supervisor renews lease and heartbeats with copied activity context.
- [ ] User cancel, graceful drain, host shutdown, deadline, lease loss retain typed causes.
- [ ] Temporal cancellation uses cooperative unwind and cannot tear a PG transaction mid-mutation.
- [ ] Worker process kill leaves the old generation unable to publish.
- [ ] New Activity waits for safe expiry/release and obtains a higher generation.
- [ ] Graceful shutdown has a bounded timeout and leaves a canonical receipt where possible.

Evidence: pending.

### 6.3 paid-call crash matrix

| crash point | required result | dispatch count | status |
|---|---|---:|---|
| before reserve | retry same command | at most 1 eventual | pending |
| after reserve, before stream dispatch preparation | reuse key and continue | 1 eventual | pending |
| after stream preparation, before `dispatch_started` | reuse key and prepare the next generation | 1 eventual | pending |
| after `dispatch_started`, before provider evidence | `dispatch_unknown` | 0 or 1 observed; automatic retry 0 | pending |
| after provider result, before settled commit | `dispatch_unknown` unless durable result exists | 1 | pending |
| after stream settlement preparation, before settled commit | full open stream + `dispatch_unknown` | 1 | pending |
| after settled commit, before Activity response | reuse stored result | 1 | pending |
| after evidence/outbox failure | recover delivery only | 1 | pending |

## 7. durable stream gate

- [ ] `open → append batch* → seal` enforces one monotonic byte cursor.
- [ ] Same cursor/same digest is idempotent; conflict is rejected.
- [ ] UTF-8 byte offsets, chunk digest, final length/digest agree.
- [ ] PostgreSQL metadata never references an unavailable object.
- [ ] Reconnect after each committed cursor returns exact ordered bytes.
- [ ] Generation replacement reports reset/gap with a typed result.
- [ ] Process kill preserves committed chunks and drops only uncommitted buffer.
- [ ] Replacement generation reset commits before `dispatch_started`; reset failure enters no provider.
- [ ] Success/refusal settlement flushes accepted output and reasoning while the generation is open.
- [ ] Flush failure leaves the generation open, marks invocation unknown, and disables automatic retry.
- [ ] Crash after invocation settlement recovers the full prepared generation and seals without reset.
- [ ] Terminal settlement rejects late stream append/seal.
- [ ] Private channels do not appear in public event, PG notification, or Temporal history.

Evidence: pending.

## 8. combined crash, race, restart gate

- [ ] PostgreSQL + MinIO + Temporal actual services run together.
- [ ] API/admission response loss creates one canonical run.
- [ ] PG outbox/Temporal acceptance ambiguity creates no duplicate semantic input.
- [ ] Worker heartbeat timeout overlaps a new Activity without stale publication.
- [ ] Stop/completion and drain/takeover races converge to one terminal winner.
- [ ] Object corruption or deletion stops checked restore before execution.
- [ ] PostgreSQL connection drop during each mutation reconciles safely.
- [ ] Restarted dispatcher and worker resume from durable rows and refs only.
- [ ] Provider dispatch counter proves duplicate paid call count zero for every recoverable case.
- [ ] Unknown dispatch counter proves automatic paid retry count zero.

Evidence: pending.

## 9. compatibility and packaging gate

- [ ] v0.21 and v0.22 checkpoint fixtures still load under the documented policy.
- [ ] v0.23 durable records use versioned schema tags and bounded codecs.
- [ ] Future/unsupported versions fail closed.
- [ ] `import monoid_agent_kernel` loads no psycopg, boto3, botocore, or temporalio module.
- [ ] `import monoid_agent_kernel.hosting` remains dependency-light.
- [ ] Each optional adapter extra installs and imports independently.
- [ ] `durable-host` installs all adapters without import cycles.
- [ ] minimal and all-extra wheel audits pass from an external temporary directory.
- [ ] SQL migrations and required package data are present in wheel and sdist.
- [ ] exact wheel tested by release audit is the artifact selected for publish.
- [ ] Windows/macOS minimal package smoke remains green.

Evidence: pending.

## 10. security and privacy gate

- [ ] public records exclude prompt, output, reasoning, replay bytes, raw checkpoint, credential.
- [ ] raw provider/database/ObjectStore exceptions do not enter durable public state.
- [ ] ObjectStore access requires run association and host authorization wrapper.
- [ ] credentials and TLS/KMS config never appear in repr, log, metric attributes, Temporal payload.
- [ ] SQL identifiers are validated and values are parameterized.
- [ ] migration role and runtime role permissions are documented separately.
- [ ] bucket policy guidance enforces conditional writes where supported.
- [ ] GC is dry-run by default and explicit apply rechecks authority.
- [ ] stale writer cannot infer winner payload through commit status evidence.
- [ ] privacy scan covers PostgreSQL rows, Temporal history, event artifacts, logs, traces.

Evidence: pending.

## 11. performance and operations gate

- [ ] connection pool limits and timeout behavior are documented and load-smoked.
- [ ] same-run serialization and cross-run concurrency are measured.
- [ ] stream batching has bounded memory, bytes, delay, chunk count.
- [ ] outbox backlog, oldest age, attempts, lease expiry, unknown invocation metrics emit.
- [ ] doctor reports service reachability, schema floor, capability, bucket conditional behavior.
- [ ] migration, rolling deploy, rollback, drain, takeover, corruption, GC runbooks are exercised.
- [ ] backup/restore preserves PostgreSQL/ObjectStore referential integrity in the test profile.

Evidence: pending.

PR12 binds these release claims to the tracked qualification manifest at
`tests/service/qualification-v023.json`. `tools/v023_ci.py validate-lock` verifies every selected
test node and `write-evidence` embeds the manifest digest plus category mapping in the L2 artifact.
The release candidate must rerun the same manifest through the complete L2 core/service workflows
before any checkbox in this audit changes to passed.

## 12. full release commands and CI

Release candidate에서 실제 command와 결과를 채운다.

| gate | command/workflow | result | artifact |
|---|---|---|---|
| lint/compile/import | pending | pending | pending |
| Python 3.11 full suite | pending | pending | pending |
| Python 3.12 full suite | pending | pending | pending |
| coverage floor | pending | pending | pending |
| PostgreSQL 16 | pending | pending | pending |
| PostgreSQL latest supported | pending | pending | pending |
| MinIO ObjectStore | pending | pending | pending |
| Temporal local service | pending | pending | pending |
| combined kill/restart | pending | pending | pending |
| compatibility fixtures | pending | pending | pending |
| wheel/sdist audit | pending | pending | pending |
| minimal install/cold-start | pending | pending | pending |
| all adapter extras install | pending | pending | pending |
| Workflow history replay | pending | pending | pending |
| privacy scan | pending | pending | pending |

## 13. explicit exclusions

최종 release에서 실제 제외 범위를 구현 계획과 다시 대조한다. 현재 계획상 제외:

- separate `ModelCallActivity`
- deterministic AgentLoop Workflow replay와 mid-step program counter
- general tool-effect journal
- multi-region active-active authority
- Redis/Kafka adapter와 `LISTEN/NOTIFY` correctness
- automatic repair/operator UI
- CSP product integration
- cloud credential을 사용하는 blocking public CI

제외 항목이 release 중 구현되면 별도 contract, test, compatibility, owner 승인을 갖춘다.

## 14. sign-off

| 승인 | 담당 | 상태 | 근거 |
|---|---|---|---|
| 구현 범위 owner | project owner | approved | A01~A14 disposition, 2026-08-23 |
| storage/conformance | 미정 | pending | PostgreSQL/ObjectStore evidence |
| Temporal/replay | 미정 | pending | Workflow/Activity evidence |
| security/privacy | 미정 | pending | privacy and credential scan |
| packaging/compatibility | 미정 | pending | wheel, fixture, migration evidence |
| final release | 미정 | pending | exact artifact and tag |

모든 required gate가 실제 release commit과 exact artifact에서 통과한 뒤에만 이 문서 상태를
`release approved`로 바꾼다.
