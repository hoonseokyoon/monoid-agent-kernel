# v0.23 PostgreSQL·ObjectStore·Temporal 구현 전 조사와 실행 계획

> 상태: 구현 기준안; A01~A14 권장안 owner 승인 완료
>
> 작성일: 2026-08-23
>
> 승인일: 2026-08-23
>
> 기준 릴리스: `v0.22.0` / `6b4bd9f`
>
> 개발 workflow: [`V0_23_DEVELOPMENT_WORKFLOW.md`](V0_23_DEVELOPMENT_WORKFLOW.md)

## 1. 결론

v0.23의 권장 범위는 다음 세 층을 production path로 완성한다.

1. PostgreSQL이 semantic metadata, writer generation, admission/outbox, canonical terminal을
   소유한다.
2. ObjectStore가 immutable private bytes와 stream chunk를 소유한다.
3. Temporal이 run별 입력 순서, timer, retry, Signal, Continue-As-New를 소유한다.

여기에 durable chat과 장기 실행 제품이 공통으로 필요로 하는 bounded durable stream v1과
migration/doctor/GC/metrics/runbook을 포함한다. 별도 `ModelCallActivity`와 deterministic
`AgentLoop` workflow replay는 v0.24+ 설계로 둔다.

이 범위는 v0.22의 production boundary를 실제 multi-process backend에 연결한다. 완료 뒤 남는
작업은 새로운 실행 모델과 제품별 composition이다. PostgreSQL/ObjectStore/Temporal adapter의
핵심 correctness가 미완성 상태로 남지 않는다.

예상 규모는 13개 순차 구현 PR, 26~33 focused development day다. 반복 리뷰와 외부 변경에 따른
calendar time은 별도로 늘어날 수 있다.

## 2. 프로그램 배경과 범위 기준

### 2.1 common-skill-pipeline에서 드러난 공통 요구

historical Skill Lab과 durable chat은 서로 다른 제품 기능이었으나 실행 기반 요구는 같았다.

- 모델 호출을 안정된 logical identity로 주소 지정한다.
- safe semantic boundary에서 checkpoint하고 process replacement 뒤 복구한다.
- paid dispatch 전후 crash를 durable journal로 구분한다.
- run별 writer owner와 monotonic generation으로 stale writer를 fence한다.
- admission response loss가 새 run이나 새 paid call을 만들지 않게 한다.
- event, terminal, evidence, input receipt가 같은 authority에 수렴한다.
- private content를 public event, notification, workflow history에서 분리한다.
- reconnect가 ordered event와 model stream을 cursor로 다시 읽는다.
- Step, Batch, Agent 같은 고수준 orchestration은 외부 workflow/runtime가 조합한다.

v0.20은 model call identity와 observation을 만들었다. v0.21은 validation, retry ownership,
receipt ledger, private replay를 추가했다. v0.22는 durable invocation journal, fenced sink,
terminal outcome, typed interruption, activation write authority를 완성했다.

### 2.2 Monoid 독립성

v0.23은 다음 제품 개념을 포함하지 않는다.

- CSP tenant/account/RLS GUC
- Skill, Case, CatalogRevision, RunPlan
- ChatStore, SubjectStore, assistant message schema
- HTTP, SSE, GUI, browser reconnect wording
- Item, Artifact, Paper, Wiki, Interactive Note
- pricing, budget, deployment, OCI/Kubernetes policy

Monoid는 opaque `run_id`, `input_id`, content digest, outcome, stream cursor만 이해한다. host는
tenant와 product identity를 외부 mapping으로 연결한다.

### 2.3 CSP 작업 보호

사전조사는 CSP의 이미 존재하는 local refs와 파일을 read-only로 읽었다. fetch, checkout,
branch 변경, test, build, install, migration, 파일 수정을 수행하지 않았다. v0.23 구현과 CI도 CSP를
fixture, submodule, editable dependency로 사용하지 않는다.

## 3. v0.22 기준선

### 3.1 이미 완성된 contract

| 기반 | 현재 제공 기능 | v0.23 사용 방식 |
|---|---|---|
| `WriterToken` | `run_id`, `owner_id`, monotonic `generation` | PostgreSQL authority row가 발급·검증한다. |
| `FencedRunSink` | checkpoint, invocation, evidence, event, terminal mutation | PostgreSQL adapter가 전체 contract를 구현한다. |
| `CommitResult` | committed/already_committed/conflict/fenced | SQL mutation과 canonical readback 결과로 유지한다. |
| `StorageCapabilities` | CAS, fencing, durability, outbox 등 fail-closed 선언 | startup doctor와 adapter capability에 연결한다. |
| `DurableModelInvocation` | reserve/start/settle/unknown journal | Activity crash 뒤 paid-call 재실행 판단에 사용한다. |
| `TerminalOutcome` | portable terminal vocabulary | PostgreSQL first-writer-wins terminal의 payload가 된다. |
| `ActivationWriteAuthority` | process-local sticky revocation | lease loss, Temporal cancel, worker drain을 모든 mutation surface에 전파한다. |
| `RunCheckpoint` | inbox/outbox, active/applied input receipt, suspension, invocation summary | finite activation의 semantic authority가 된다. |
| tool effect policy | `outbox` 또는 `idempotent` declaration | v0.23 production outbox가 strict policy를 실행한다. |
| fenced conformance | race, handoff, blob, invocation, terminal matrix | 실제 PostgreSQL adapter에 그대로 실행한다. |

### 3.2 현재 열린 gap

1. `WriterToken`을 발급·renew·revoke하는 durable authority store가 없다.
2. 실제 PostgreSQL `FencedRunSink`가 없다.
3. immutable blob을 S3-compatible ObjectStore에 저장하는 adapter가 없다.
4. admission과 PostgreSQL → Temporal handoff의 durable outbox가 없다.
5. DBOS reference에 있는 finite activation driver가 orchestrator-neutral contract가 아니다.
6. normal `AgentEvent`와 final `TerminalOutcome`을 `FencedRunSink`에 연결하는 bridge가 없다.
7. Temporal Workflow/Activity adapter가 없다.
8. high-volume model delta를 reconnect 가능한 형태로 저장하는 stream contract가 없다.
9. migration, doctor, GC, outbox lag, crash qualification 도구가 없다.
10. 현재 CI는 모든 pull request에 전체 in-process matrix를 실행하며 실제 PostgreSQL, MinIO,
    Temporal service를 사용하지 않는다.

## 4. 목표 구조와 authority

```text
Product host
  tenant/auth · Skill/Chat/Task · HTTP/SSE/UI · product projection
                          │
                          ▼
Host-neutral admission/control API
  stable input ID · request digest · dispatch outbox · canonical receipt
                          │
                          ▼
Temporal run Workflow
  ordering · timers · Signal · retry · Continue-As-New
                          │ finite activation Activity
                          ▼
Monoid activation driver + AgentLoop
  checkpoint · invocation · event · stream · terminal
             │                              │
             ▼                              ▼
PostgreSQL                             ObjectStore
metadata · writer authority            immutable private bytes
admission · outbox · cursor            checkpoint/model/stream blobs
terminal · run/blob association
```

### 4.1 canonical owner

| 데이터/결정 | canonical owner |
|---|---|
| run writer generation과 lease expiry | PostgreSQL DB clock + authority row |
| semantic checkpoint와 model invocation journal | PostgreSQL metadata, ObjectStore private blob |
| admission identity와 dispatch state | PostgreSQL |
| event/terminal winner와 stream cursor | PostgreSQL |
| immutable private bytes | ObjectStore, PostgreSQL run association |
| ordering, timer, retry schedule, Workflow history | Temporal |
| product status/message/projection | product host |

Temporal heartbeat와 Activity attempt는 liveness/provenance다. PostgreSQL writer generation이 durable
write authority다. Workflow Query는 operational view다. PostgreSQL terminal과 receipts가 canonical
read model이다.

## 5. 권장 release 범위

### 5.1 반드시 완료할 production path

- optional adapter package/import 경계와 service CI foundation
- packaged PostgreSQL migration과 schema compatibility metadata
- DB-clock writer lease/generation lifecycle
- actual PostgreSQL `FencedRunSink`와 bytea blob profile
- generic content-addressed BlobStore와 S3-compatible implementation
- PostgreSQL + ObjectStore run association과 orphan-safe commit
- orchestrator-neutral finite activation driver
- fenced event/terminal bridge와 canonical readback
- admission/control record와 durable dispatch outbox
- Temporal Signal-With-Start dispatcher
- long-lived per-run Workflow와 finite activation Activity
- heartbeat, cooperative cancellation, worker shutdown, lease-loss mapping
- Continue-As-New와 Workflow replay/versioning test
- real PostgreSQL, MinIO, Temporal service integration
- crash/process-kill/restart/response-loss qualification

### 5.2 중간 범위에 포함할 추가 기능

- durable stream v1: open/batched append/seal, monotonic cursor, reconnect, final digest
- PostgreSQL stream metadata + ObjectStore chunk bytes
- configurable bounded coalescing과 terminal 뒤 late-delta 차단
- migration plan/status/apply API와 doctor
- outbox backlog, lease, unknown invocation, GC metrics
- object inventory, grace/recheck 기반 GC dry-run과 explicit apply
- rollout, drain, recovery, backup/restore runbook

### 5.3 v0.23 제외

- model call 하나마다 별도 Temporal Activity 실행
- AgentLoop를 Temporal Workflow 본문에서 deterministic replay
- mid-step program counter와 general effect suspension
- arbitrary tool effect journal의 새 lifecycle
- multi-region active-active writer authority
- PostgreSQL `LISTEN/NOTIFY` correctness dependency
- Redis/Kafka durable queue adapter
- Temporal Cloud 또는 AWS 계정 credential이 필요한 blocking CI
- 자동 repair, operator UI, hosted control plane
- CSP integration 또는 CSP migration

## 6. package와 dependency 경계

권장 module 구조는 다음과 같다.

```text
monoid_agent_kernel.hosting
  authority.py       neutral lease/generation contract
  blobs.py           content-addressed BlobStore contract
  admission.py       admitted input/control/dispatch contract
  activation.py      finite command/receipt/driver contract
  streams.py         durable stream v1 contract

monoid_agent_kernel.adapters.postgres
  config · pool · migrations · authority · sink · admission · stream metadata · ops

monoid_agent_kernel.adapters.object_store
  local/bytea seam · S3-compatible implementation · multipart · inventory/GC

monoid_agent_kernel.adapters.temporal
  records · workflow · activity · dispatcher · worker composition · testing
```

`hosting`은 stdlib와 기존 base dependency만 사용한다. 각 concrete module은 필요한 dependency를
lazy import한다. package root는 adapter를 re-export하지 않는다.

권장 extras:

| extra | dependency policy |
|---|---|
| `postgres` | `psycopg[binary]>=3.2,<4`, `psycopg-pool>=3.2,<4` |
| `object-store-s3` | `boto3>=1.37.32,<2` |
| `temporal` | `temporalio>=1.17,<2` |
| `durable-host` | 앞 세 extra dependency의 합집합 |

PR 1 compatibility spike는 boto3 1.37.32에서 conditional PUT/checksum/multipart API를,
temporalio 1.17.0에서 local environment, Signal-With-Start, Replayer API를 확인했다. campaign
exact SDK는 psycopg 3.3.4, psycopg-pool 3.3.1, boto3 1.43.78, transitive botocore 1.43.78,
temporalio 1.31.0이다. Temporal local service는 CLI v1.8.2와 embedded Server 1.31.2를 사용한다.
exact SDK, service image digest,
Temporal CLI archive checksum은 `tests/service/campaign-lock.json`이 소유한다.
실제 Temporal service gate는 WorkflowService `GetSystemInfo.server_version`을 조회해 lock의 embedded
Server 버전과 일치시킨 뒤에만 qualification evidence를 만든다.

별도 distribution 분리는 v0.23 완료 뒤 dependency cadence와 maintainer 부담을 근거로 결정한다.
첫 release는 한 repository의 optional modules로 API와 compatibility를 함께 검증한다.

## 7. PostgreSQL adapter

### 7.1 지원 범위

권장 최소 major는 PostgreSQL 16이다. CSP의 현재 운영 major와 맞고 2028년까지 community support가
남아 있다. release gate는 16과 현재 최신 stable major 18을 실행한다. PostgreSQL은 major별로 5년간
지원되며 현재 16, 17, 18이 지원 상태다.

출처: [PostgreSQL versioning policy](https://www.postgresql.org/support/versioning/)

`run_id`는 globally unique opaque ID로 취급한다. adapter schema에는 `tenant_id`가 없다. tenant-aware
host는 global kernel ID를 파생하거나 tenant-scoped wrapper와 별도 mapping을 사용한다.

### 7.2 connection model

v0.23 public adapter는 synchronous protocol과 `psycopg_pool.ConnectionPool`을 사용한다.

- 현재 `FencedRunSink`와 `AgentLoop` mutation 표면이 sync다.
- Temporal adapter는 synchronous threaded Activity를 사용한다.
- 한 adapter instance가 pool을 소유하거나 caller-provided compatible pool을 받는다.
- 모든 mutation은 explicit transaction context를 사용한다.
- pool 반환 전 transaction state를 reset하고 startup health check를 수행한다.
- connection/session global state에 tenant나 authority를 숨기지 않는다.

async-native PostgreSQL API는 실제 consumer와 성능 근거를 확보한 뒤 별도 additive surface로 둔다.

### 7.3 schema와 migration

권장 default schema는 `monoid_kernel`이며 config로 변경할 수 있다. adapter construction은 migration을
자동 실행하지 않는다.

- ordered immutable SQL resource를 wheel에 포함한다.
- migration checksum과 installed version을 metadata table에 기록한다.
- transaction-level advisory lock은 migration runner 한 곳에서만 사용한다.
- `status`, `plan`, `apply`, `doctor` API를 제공한다.
- release N은 additive migration과 N-1 reader compatibility를 유지한다.
- destructive cleanup은 compatibility window가 끝난 후 별도 release에서 수행한다.
- down migration에 correctness를 의존하지 않는다. rollback은 이전 compatible binary와 forward
  schema를 사용하는 방식으로 검증한다.

SQLAlchemy와 Alembic을 추가하지 않는다. adapter가 사용하는 SQL이 작고 transaction ordering이
contract의 핵심이므로 migration과 statement를 직접 review한다.

### 7.4 conceptual tables

최종 이름은 schema PR에서 고정한다.

| 관계 | key와 역할 |
|---|---|
| schema metadata | migration id, checksum, reader/writer compatibility floor |
| run authority | `run_id -> owner_id, generation, leased_until, revoked` |
| checkpoint record/head | immutable `(run_id, seq)`와 authoritative latest head |
| invocation record/head | immutable `(run_id, logical_call_id, revision)`와 latest head |
| run event | immutable `(run_id, sequence)` |
| terminal outcome | `run_id`별 first writer winner |
| model evidence/outbox | settled revision의 public-safe projection과 delivery state |
| run blob association | `(run_id, sha256)`와 locator, size, verified integrity |
| admitted input/control | stable ID, digest, monotonic run sequence, payload ref, receipt |
| orchestration outbox | Temporal delivery identity, attempt/next-at, accepted receipt |
| stream head/chunk | stream generation, byte cursor, chunk digest/ref, sealed digest |

JSONB는 versioned canonical payload를 보존한다. identity, sequence, status, digest, timestamp처럼
locking/query에 필요한 값은 typed column으로 둔다. private bytes는 bytea profile 또는 ObjectStore에
둔다.

### 7.5 writer authority

추가할 neutral contract는 다음 의미를 가진다.

```python
claim(run_id, owner_id, ttl) -> WriterLease
renew(writer_token, ttl) -> RenewResult
release(writer_token) -> ReleaseResult
read(run_id) -> WriterAuthority | None
```

- 최초 claim은 generation 1을 만든다.
- 같은 active token의 renew는 generation을 유지한다.
- 다른 owner는 lease expiry 뒤에만 새 generation을 claim한다.
- 정상 handoff는 현재 token의 release/revoke를 먼저 요구한다.
- administrative force revoke는 별도 privileged API다.
- expiry와 renewal은 DB `clock_timestamp()` 계열의 database clock을 사용한다.
- renewal 응답이 불명확하면 process-local authority를 즉시 revoke한다.
- Activity attempt number를 generation으로 사용하지 않는다.

crashed worker 뒤 takeover latency의 상한은 lease TTL이다. 짧은 TTL과 heartbeat 주기를 config로
조절한다. 안전한 early takeover를 증명할 current token이 없으면 expiry를 기다린다.

### 7.6 mutation linearization

모든 fenced mutation은 한 PostgreSQL transaction 안에서 다음 순서를 지킨다.

```text
BEGIN
  1. run authority row를 SELECT ... FOR UPDATE
  2. token.run_id binding 확인
  3. owner_id, generation, revocation, DB-clock expiry 확인
  4. durable codec와 submitted blob map 검증
  5. same-coordinate idempotency/conflict 판정
  6. lifecycle, head, terminal-first-writer 규칙 확인
  7. immutable record + head + optional outbox/association commit
COMMIT
  8. response가 불명확하면 canonical readback으로 reconcile
```

fence가 content/idempotency 판정보다 앞선다. stale writer는 기존 winner의 존재나 digest를 이용해
성공 응답을 얻지 못한다. run authority row lock은 같은 run의 linearization point다. advisory lock과
`LISTEN/NOTIFY`는 correctness에 참여하지 않는다.

PostgreSQL `INSERT ... ON CONFLICT`와 `RETURNING`은 atomic insert/result에 사용하고 row lock과
unique constraint가 lifecycle을 함께 강제한다.

출처:

- [PostgreSQL INSERT / ON CONFLICT / RETURNING](https://www.postgresql.org/docs/17/sql-insert.html)
- [Psycopg transaction management](https://www.psycopg.org/psycopg3/docs/basic/transactions.html)
- [Psycopg concurrent operations](https://www.psycopg.org/psycopg3/docs/advanced/async.html)
- [Psycopg connection pools](https://www.psycopg.org/psycopg3/docs/api/pool.html)

### 7.7 bytea profile

작은 deployment와 conformance를 위해 PostgreSQL bytea blob profile을 포함한다.

- content digest를 key로 immutable insert한다.
- run association을 별도로 기록한다.
- 같은 digest의 bytes가 다르면 corrupt/conflict로 실패한다.
- size limit은 config와 bounded default를 가진다.
- 대형 checkpoint/model/stream content는 ObjectStore profile을 사용한다.

bytea profile은 production-capable small-object path이며 ObjectStore 장애 진단에도 사용한다.

## 8. ObjectStore adapter

### 8.1 neutral contract

```python
class ContentAddressedBlobStore(Protocol):
    def put_if_absent(self, sha256: str, data: bytes) -> BlobPutResult: ...
    def stat(self, sha256: str) -> BlobStat | None: ...
    def get_checked(self, sha256: str) -> bytes: ...
```

logical address는 기존 `blob:<lowercase-sha256>`를 유지한다. physical key는 기본적으로
`sha256/<first-two>/<digest>`를 사용하고 configurable prefix를 앞에 붙인다.

### 8.2 write/read contract

- caller bytes의 SHA-256과 size를 upload 전에 계산한다.
- final key는 conditional write-once다.
- single PUT과 multipart completion 모두 `If-None-Match: *`를 사용한다.
- `412`는 existing object를 읽고 digest/size가 같을 때만 already-present로 수렴한다.
- `409`는 bounded retry한다. multipart는 새 upload ID와 part 전체 재업로드가 필요하다.
- checksum과 stored metadata를 검증한다.
- multipart ETag를 content SHA-256으로 사용하지 않는다.
- read는 bytes를 다시 hash하고 corrupt를 구분한다.
- list 권한 없이 put/get runtime path가 동작한다.
- credentials, bucket policy, encryption key는 durable public record와 log에 남기지 않는다.

AWS는 `PutObject`와 `CompleteMultipartUpload`의 conditional write를 지원하며 concurrent conflict에서
`409`, existing key에서 `412`를 반환한다.

출처:

- [S3 conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html)
- [CompleteMultipartUpload API](https://docs.aws.amazon.com/AmazonS3/latest/API/API_CompleteMultipartUpload.html)
- [S3 multipart upload](https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html)
- [S3 object integrity](https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity-upload.html)

### 8.3 PostgreSQL association과 commit 순서

physical content dedup은 global일 수 있다. authorization은 PostgreSQL `(run_id, sha256)` association이
소유한다. 다른 run의 association 없이 object를 해석하지 않는다.

```text
1. bytes digest/size 검증
2. ObjectStore conditional upload와 integrity 확인
3. PostgreSQL transaction에서 writer fence 재검증
4. run/blob association 생성
5. checkpoint/invocation/stream metadata commit
6. canonical readback
```

PG rollback 뒤 object만 남으면 orphan이다. canonical run state에는 영향을 주지 않는다.

### 8.4 multipart와 GC

중간 범위는 multipart를 포함한다. 구현은 configurable threshold와 part size, maximum object size,
bounded retry를 가진다. aborted/incomplete upload cleanup도 inventory에 포함한다.

GC는 다음 두 단계로 제공한다.

1. `plan`/`dry-run`: inventory와 PG association을 비교해 grace period를 지난 candidate를 만든다.
2. explicit `apply`: candidate generation을 확인하고 association을 다시 읽은 뒤 delete와 receipt를
   기록한다.

GC scheduler와 retention policy는 host가 소유한다. v0.23 adapter는 안전한 primitive와 runbook을
제공한다.

## 9. orchestrator-neutral activation과 execution bridge

### 9.1 DBOS reference에서 추출할 부분

현재 DBOS reference driver는 finite activation의 좋은 선례다.

- stable resume command와 checkpoint marker
- applied input receipt dedup
- active input crash recovery
- `run_until_suspended(None)` drive
- ambiguous checkpoint commit readback
- durable boundary receipt

v0.23은 이 의미를 `hosting.activation`의 neutral record/driver로 추출한다. DBOS와 Temporal이 같은
driver를 소비하게 하고 DBOS-specific workflow ID, queue, runtime type을 neutral contract에 넣지 않는다.

### 9.2 command와 receipt

`ActivationCommand`의 최소 identity:

- schema version
- run ID
- admitted input/control ID와 monotonic sequence
- source checkpoint sequence/ref
- request digest와 private payload ref
- command kind

`ActivationReceipt`의 최소 결과:

- command identity
- canonical checkpoint sequence/digest
- suspension/terminal outcome ref
- applied input receipt ref
- event/stream cursor
- retry classification과 public-safe error code

receipt는 checkpoint/PG canonical data에서 재구성할 수 있어야 한다. orchestrator return은 이
canonical receipt의 operational copy다.

### 9.3 event/terminal bridge

`FencedEventSink`는 normal `EventSink` 호출을 `FencedRunSink.append_event`로 연결한다.

- current `WriterToken`과 `ActivationWriteAuthority`를 사용한다.
- fenced 결과에서 authority를 즉시 revoke한다.
- duplicate event는 same sequence/same digest만 허용한다.
- private content delta는 durable stream으로 보낸다.
- final result/suspension/failure를 `TerminalOutcome`으로 정규화한다.
- terminal commit 뒤 canonical winner를 읽는다.
- projection failure는 settled provider result를 재실행하지 않는다.
- terminal 뒤 event/stream mutation을 거부한다.

## 10. admission, control, dispatch outbox

### 10.1 state model

```text
admit input/control
  → prepared in PostgreSQL
  → dispatch outbox published
  → Temporal accepted/acceptance unknown
  → Workflow observed command
  → activation claimed
  → checkpoint receipt committed
  → command completed
```

같은 `input_id`와 digest는 같은 handle/receipt를 반환한다. 같은 ID와 다른 digest는 dispatch 전에
conflict다. 모든 state는 retry 뒤 canonical readback으로 판정한다.

### 10.2 cross-system handoff

PostgreSQL과 Temporal 사이에 distributed transaction을 만들지 않는다.

1. admission/control record와 dispatch outbox를 한 PG transaction에 commit한다.
2. polling dispatcher가 per-run order로 row를 claim한다.
3. deterministic Workflow ID와 command ID로 Signal-With-Start를 보낸다.
4. Temporal server acceptance 뒤 delivered metadata를 기록한다.
5. response loss는 같은 command를 다시 보낸다.
6. Workflow와 checkpoint applied-input receipt가 duplicate를 제거한다.

dispatcher claim에는 `FOR UPDATE SKIP LOCKED`를 사용할 수 있다. queue delivery 권위와 run write
authority는 분리된 generation을 사용한다.

### 10.3 control intent

user cancel, pause/resume, deadline 같은 control도 stable command ID로 admission한다. Activity의 lease
supervisor가 canonical control intent를 읽고 `CancellationToken`과 `ActivationWriteAuthority`에
전파한다. worker shutdown은 Temporal worker state에서 `graceful_drain`/`host_shutdown`으로 매핑한다.

## 11. Temporal adapter

### 11.1 snapshot/activity model

첫 adapter는 per-run entity Workflow와 finite `drive_activation` Activity를 사용한다.

```text
Workflow receives content-free command refs
  → orders/deduplicates commands
  → schedules one drive_activation Activity
  → Activity acquires PostgreSQL generation
  → restores checkpoint and private refs
  → drives AgentLoop to a safe suspension/terminal
  → commits canonical receipt
  → Workflow stores only receipt refs
```

AgentLoop, provider, tool, PostgreSQL, ObjectStore 호출은 Activity에서 실행한다. Workflow에는
deterministic orchestration과 작은 serializable record만 둔다.

### 11.2 Workflow messages

권장 기본 입력은 Signal-With-Start다.

- Signal은 Temporal server가 수락하면 caller에게 응답하며 Worker 처리 완료를 기다리지 않는다.
- Signal-With-Start는 Workflow가 없으면 시작하고 있으면 같은 ID에 전달한다.
- PG admission receipt가 client-facing canonical handle이므로 synchronous Update result가 필요 없다.
- Update-With-Start는 worker availability와 acceptance/result 대기를 추가한다.
- input ID/checkpoint receipt가 exactly-once application 의미를 제공한다.

Temporal 문서는 Signal이 asynchronous state mutation이고 server acceptance에서 반환한다고 설명한다.
Update는 Worker acceptance를 기다리는 synchronous request다.

출처: [Temporal Python Workflow message passing](https://docs.temporal.io/develop/python/workflows/message-passing)

Query는 content-free operational status와 PG refs만 반환한다. product status/read API는 PostgreSQL
canonical projection을 읽는다.

### 11.3 Activity execution과 cancellation

권장 구현은 synchronous multithreaded Activity다.

- current `AgentLoop`와 `FencedRunSink`가 sync surface다.
- Temporal Python SDK도 threaded Activity를 초기 권장 형태로 안내한다.
- `@activity.defn(no_thread_cancel_exception=True)`로 임의 시점 exception injection을 끈다.
- context를 복사한 supervisor thread가 PG lease renew와 Temporal heartbeat를 수행한다.
- supervisor는 user control, Activity cancel, worker shutdown, renew ambiguity에서 local authority와
  cancellation token을 revoke/request한다.
- AgentLoop는 safe boundary로 unwind하고 canonical interruption receipt를 commit한다.
- lease가 사라진 old Activity는 PostgreSQL fence 때문에 event/checkpoint/terminal을 publish하지 못한다.

Temporal Activity API는 내부 thread에서 `activity.*`를 호출할 때 `contextvars.copy_context()`가
필요하다고 명시한다. non-local Activity는 heartbeat timeout과 주기적 heartbeat가 있어야 cancel을
받는다. worker graceful shutdown timeout도 Activity에 전달된다.

출처: [Temporal Python SDK Activities](https://github.com/temporalio/sdk-python#activities)

Activity retry policy는 bounded exponential backoff와 non-retryable error classification을 명시한다.

| canonical state | Activity retry 의미 |
|---|---|
| dispatch 전 infra failure | same command/logical call로 재시도 가능 |
| invocation `reserved` | same idempotency key로 dispatch 가능 |
| invocation `settled` | stored result 재사용 |
| `dispatch_started` without receipt | `unknown`으로 수렴, provider 자동 재호출 0 |
| corrupt/unsupported/config conflict | non-retryable result/error |
| evidence/outbox incomplete | provider 호출 없이 delivery만 복구 |

### 11.4 Continue-As-New

Workflow는 다음 safe point에서 `workflow.info().is_continue_as_new_suggested()`를 확인한다.

- active Activity가 없다.
- Signal handler가 모두 끝났다.
- pending command ID와 latest canonical receipt ref를 새 input으로 전달할 수 있다.

Continue-As-New는 같은 Workflow ID와 새 Run ID, 새 Event History를 만든다. Update/Signal handler에서
직접 실행하지 않고 main Workflow가 `all_handlers_finished`를 확인한 뒤 수행한다.

출처: [Temporal Python Continue-As-New](https://docs.temporal.io/develop/python/workflows/continue-as-new)

test hook은 작은 history threshold를 사용해 CI에서 경계를 빠르게 검증한다.

### 11.5 determinism과 versioning

- workflow record와 workflow type을 v1로 version한다.
- network/database/random/wall-clock 작업을 Activity에 둔다.
- 저장된 representative history를 `Replayer`로 검증한다.
- running Workflow를 깨는 변경은 Temporal patching 또는 Worker Versioning을 사용한다.
- release audit에 supported workflow build/schema와 replay corpus를 기록한다.

Temporal은 long-running Workflow code의 determinism을 요구하며 orchestration 변경에 patching 또는
Worker Versioning을 사용하도록 안내한다.

출처: [Temporal Python Workflow versioning](https://docs.temporal.io/develop/python/workflows/versioning)

### 11.6 payload privacy

Workflow history에는 다음만 저장한다.

- opaque run/command/input ID
- sequence와 digest
- checkpoint/receipt/outcome/stream ref
- bounded retry metadata와 public-safe error code
- workflow/schema/build version

prompt, response, reasoning, workspace bytes, model result, raw checkpoint, credential, provider exception text를
저장하지 않는다. Temporal Python SDK의 external storage는 조사 시점에 experimental이다. v0.23
correctness는 그 기능에 의존하지 않고 stable Monoid ObjectStore ref를 사용한다.

출처: [Temporal Python SDK external storage](https://github.com/temporalio/sdk-python#external-storage)

## 12. durable stream v1

### 12.1 목적

normal event table에 token마다 row를 추가하면 write amplification과 Workflow/event history가 커진다.
durable stream은 private high-volume content와 public lifecycle event를 분리한다.

### 12.2 contract

- stable stream ID와 run/logical-call lineage
- `output`, `reasoning`, host-defined private channel
- stream generation과 monotonic UTF-8 byte offset
- `open → append batch* → seal` lifecycle
- chunk sequence, start/end offset, content digest, object ref
- same cursor/same digest idempotency와 conflicting digest rejection
- reconnect `read_after(cursor, limit)`
- old generation, replay gap, reset의 typed result
- final byte length와 SHA-256
- terminal 뒤 no-late-delta
- writer fencing과 process replacement
- runner의 실제 provider-dispatch 시작 신호와 provider를 호출하지 않는 settled recovery를 구분한다.
  replacement는 `dispatch_started` commit과 adapter entry 전에 prior generation을 reset한다.
  reset 실패는 invocation을 `reserved`로 유지하고 provider entry를 막는다. provider-free
  success/failure recovery는 committed generation을 보존하며 success는 authoritative final
  output도 reconcile한다.
- provider terminal 뒤 durable settlement 전에 accepted output/reasoning buffer를 모두 flush한다.
  flush 성공은 generation을 open으로 유지하고, invocation settlement 뒤 close 또는 recovery가
  seal한다. flush 실패는 `stream_settlement_uncommitted` unknown으로 수렴하며 자동 provider
  재호출을 막고 해당 generation을 seal하지 않는다.

### 12.3 storage

- PostgreSQL은 stream head, generation, cursor, chunk metadata, final digest를 저장한다.
- ObjectStore는 batched chunk bytes를 저장한다.
- configurable byte/time threshold로 delta를 coalesce한다.
- close 전 crash는 committed cursor까지만 replay한다.
- host projection은 public-safe lifecycle event와 private authorized stream read를 합성한다.

v0.23은 one-run ordered writer와 reconnect correctness를 완성한다. adaptive compression, global fanout,
Redis pubsub, CDN delivery는 후속 최적화다.

## 13. 운영 기능

### 13.1 포함

- migration `status/plan/apply`
- adapter `doctor`와 capability report
- pool/service health와 schema compatibility check
- lease current owner/generation/expiry inspection
- outbox backlog/age/attempt count
- invocation unknown/evidence-pending count
- stream lag/chunk/object bytes metrics
- object orphan/multipart inventory와 GC dry-run/apply
- worker drain과 takeover runbook
- migration, rolling deploy, rollback, backup/restore, corruption response runbook
- OTel meter/span hook과 structured public-safe logs

### 13.2 제외

- background operator daemon의 lifecycle 소유
- automatic force takeover와 automatic corrupt-data repair
- destructive migration 자동 실행
- dashboard/UI
- multi-region replication/failover controller
- cloud-specific IAM/KMS provisioning

host가 scheduler와 credential을 소유하고 adapter가 safe operation primitive와 evidence를 제공한다.
구체적인 startup, rolling migration, drain/takeover, GC, backup/restore, corruption 절차는
[`PRODUCTION_OPERATIONS.md`](PRODUCTION_OPERATIONS.md)가 소유한다.

## 14. CI와 실제 서비스 검증

CI 실행 정책은 [`V0_23_DEVELOPMENT_WORKFLOW.md`](V0_23_DEVELOPMENT_WORKFLOW.md)가 소유한다.

### 14.1 PostgreSQL

- PostgreSQL 16 actual service container per adapter PR
- release에서 PostgreSQL 16과 18
- 서로 다른 connection/process의 race
- lease expiry/renew/response-loss/handoff
- complete fenced conformance
- migration fresh install, N→N+1, compatible old reader

GitHub의 공식 service-container pattern을 사용한다.
[PostgreSQL service container guide](https://docs.github.com/en/actions/tutorials/use-containerized-services/create-postgresql-service-containers)

### 14.2 ObjectStore

- pinned MinIO container
- conditional single/multipart put race
- 409/412 fault injection
- checksum/size/corrupt/missing/cross-run association
- upload success + PG rollback orphan
- GC grace/recheck race

### 14.3 Temporal

- `WorkflowEnvironment.start_time_skipping()` unit/integration
- `WorkflowEnvironment.start_local()` actual local service
- Signal-With-Start and duplicate delivery
- Activity retry at every invocation crash prefix
- heartbeat timeout and old/new Activity overlap
- Stop/completion and worker shutdown/drain race
- Continue-As-New before/after pending command
- saved history replay across workflow changes

### 14.4 combined qualification

```text
PG admission commit
  → dispatcher response loss
  → duplicate Temporal Signal
  → worker process kill
  → lease expiry/new generation
  → settled model result reuse or dispatch_unknown
  → stream reconnect
  → terminal first-writer convergence
```

deterministic fake provider가 dispatch count를 기록한다. qualification 기준은 다음과 같다.

- stored settled result의 provider 재호출 0
- `dispatch_unknown`의 automatic provider retry 0
- stale writer의 durable mutation 0
- same input/digest의 duplicate semantic application 0
- terminal winner 1
- public PG/Temporal/event record의 private model content 0

## 15. PR 실행 순서

각 PR은 최신 v0.23 통합 브랜치에서 순차 생성한다.

| PR | 범위 | L2 profile | 핵심 종료 조건 |
|---|---|---|---|
| 1 | CI split, service harness, adapter package/import foundation | `ci:combined` | Draft 수동 Codex review capability를 실증하고 Draft/Ready 또는 `ci:full` label gate와 pinned service smoke가 동작한다. |
| 2 | neutral writer authority + PostgreSQL config/pool/migration/lease | `ci:postgres` | 실제 PG에서 DB-clock generation handoff와 stale renew가 검증된다. |
| 3 | PostgreSQL checkpoint/invocation sink + bytea profile | `ci:postgres` | checkpoint/invocation conformance와 ambiguous readback이 통과한다. |
| 4 | PostgreSQL event/terminal/evidence/outbox + full sink conformance | `ci:postgres` | 전체 `run_fenced_run_sink_contract`가 실제 PG에서 통과한다. |
| 5 | BlobStore contract + S3-compatible single/multipart adapter | `ci:objectstore` | MinIO conditional race와 checked read가 통과한다. |
| 6 | PostgreSQL/ObjectStore association, external blob sink, GC | `ci:objectstore` | object-first/PG-rollback/cross-run/GC race가 검증된다. |
| 7 | neutral activation command/receipt/driver + event/terminal bridge | `ci:postgres` | process replacement가 같은 canonical boundary receipt로 수렴한다. |
| 8 | admission/control/outbox + transport-neutral dispatcher seam | `ci:postgres` | response loss와 duplicate delivery가 input을 한 번만 적용한다. |
| 9 | Temporal Workflow, Signal-With-Start dispatcher, Continue-As-New | `ci:temporal` | 실제 local server에서 ordering/dedup/history replay가 통과한다. |
| 10 | threaded activation Activity, lease supervisor, cancel/drain | `ci:combined` | worker kill/takeover와 paid-call crash matrix가 수렴한다. |
| 11 | durable stream v1 + PG metadata/ObjectStore chunks | `ci:combined` | reconnect, generation reset, seal, late-delta race가 통과한다. |
| 12 | operations, security/privacy, rolling migration, full qualification | `ci:combined` | doctor/GC/metrics/runbook과 complete crash matrix가 증거를 가진다. |
| 13 | compatibility, docs, wheel/install, release closure | `ci:combined` | release audit의 모든 gate가 실제 결과로 채워진다. |

PR 2~6은 storage production path, PR 7~10은 orchestration production path, PR 11~12는 승인된
중간 범위 확장이다. PR 13은 기능을 추가하지 않고 release를 닫는다.

## 16. 주요 위험과 대응

| 위험 | 대응 |
|---|---|
| PG와 ObjectStore 사이 partial commit | object-first, PG association transaction, orphan GC |
| PG와 Temporal 사이 partial delivery | transactional dispatch outbox, Signal duplicate, input receipt dedup |
| heartbeat를 writer fence로 오인 | PG generation을 canonical authority로 유지 |
| sync Activity cancel이 transaction 중간에 exception injection | cooperative cancel, `no_thread_cancel_exception`, local revocation |
| old Activity가 외부 effect를 계속 실행 | strict idempotent/outbox policy, PG fence로 이후 publication 차단 |
| Workflow history에 private content 유입 | opaque ref-only records와 privacy conformance |
| Temporal experimental external storage에 API 결합 | v0.23에서 사용하지 않음 |
| stream token-row write amplification | bounded batch chunk와 separate stream table/ObjectStore |
| schema auto-upgrade가 rollout을 깨뜨림 | explicit migration plan/apply, N-1 reader window |
| full CI가 review fix마다 반복 | Draft/Ready checkpoint, Draft review 미지원 시 Ready+`ci:full` label gate, fast/full split, concurrency cancel |
| optional dependency가 base import를 오염 | lazy adapter imports와 wheel/import audit |
| CSP schema가 generic API에 스며듦 | opaque ID와 host wrapper, CSP-independent conformance |

## 17. owner 결정 항목

2026-08-23 owner가 A01~A14 권장안 전체를 승인했다. 이 표는 v0.23 구현과 release audit의
canonical disposition이다.

| ID | 결정 | 선택지 | 권장안 | 이유 | 상태 |
|---|---|---|---|---|---|
| A01 | v0.23 exact cut | production path만 / production+stream+bounded ops / ModelCallActivity까지 | production+stream+bounded ops | durable chat reconnect와 운영 종료 조건을 채우며 mid-step replay 위험을 분리한다. | 승인 |
| A02 | 배포 구조 | 별도 adapter distributions / 한 wheel의 optional modules | 한 wheel의 optional modules | 첫 release에서 contract와 compatibility를 한 캠페인으로 검증한다. | 승인 |
| A03 | package/extras | 자유 명명 / 제안 namespace와 `postgres`, `object-store-s3`, `temporal`, `durable-host` | 제안안 | base install을 가볍게 유지하고 사용자가 필요한 backend만 설치한다. | 승인 |
| A04 | PostgreSQL 지원 | 15+ / 16+ / 17+ | 16+, release 16·18 | CSP와 정렬되고 충분한 지원 기간과 최신 major 호환성을 함께 확보한다. | 승인 |
| A05 | DB API | sync pool / sync+async 동시 / async-only | sync psycopg pool | 기존 sink/loop와 Temporal threaded Activity에 맞고 public API surface를 하나로 유지한다. | 승인 |
| A06 | migration | auto-migrate / explicit bundled SQL / Alembic | explicit bundled SQL, no auto-migrate | rollout과 rollback을 운영자가 통제하고 SQL linearization을 직접 review한다. | 승인 |
| A07 | takeover | 다른 owner 즉시 takeover / expiry 또는 current-token release 뒤 takeover | expiry/release gate | crashed writer overlap에서 stale publication을 확실히 차단한다. | 승인 |
| A08 | blob profile | S3 only / bytea only / bytea+S3 | bytea+S3, multipart 포함 | 작은 설치와 conformance 경로를 유지하면서 대형 private content를 지원한다. | 승인 |
| A09 | Temporal handoff | direct start / Update-With-Start / PG outbox+Signal-With-Start | PG outbox+Signal-With-Start | response loss와 worker unavailability를 흡수하고 canonical admission을 PG에 둔다. | 승인 |
| A10 | Activity model | async Activity / sync threaded cooperative Activity / model-call Activity | sync threaded cooperative Activity | 현재 sync kernel 경계와 맞고 cancellation/lease loss를 명시적으로 통제한다. | 승인 |
| A11 | durable stream | 제외 / event table token rows / separate batched stream v1 | separate batched stream v1 | reconnect correctness를 제공하며 canonical event write amplification을 억제한다. | 승인 |
| A12 | 운영 기능 | docs only / doctor+metrics / doctor+metrics+GC+migration runbook | 세 번째 안 | v0.23을 배포 가능한 adapter release로 닫는다. | 승인 |
| A13 | CI 비용 정책 | 모든 push full / manual full / Draft-fast + Ready-full | Draft-fast + Ready-full, Draft cancel sentinel; PR 1에서 Draft 수동 Codex review가 실패하면 Ready+`ci:full` label gate | 수십 review cycle의 비용을 억제하고 Codex 제품 동작과 CI 비용 정책을 분리하며 merge SHA의 실제 서비스 증거를 보존한다. | 승인 |
| A14 | PR/기간 | 8개 대형 PR / 13개 순차 PR / 16개 이상 소형 PR | 13개 순차 PR, 26~33 개발일 | storage/orchestration/stream 책임을 review 가능한 크기로 분리한다. | 승인 |

### 승인 기록

owner 승인 문구:

```text
A01~A14 권장안 승인. v0.23 구현 계획과 workflow를 확정하고 PR 1부터 진행.
```

변경 disposition은 해당 ID, 대체안, 이유와 새 승인일을 이 표에 기록한다.

## 18. 승인 뒤 즉시 수행할 일

1. 이 문서 상태를 `구현 기준안`으로 바꾸고 승인 disposition을 기록한다.
2. `codex/v0.23-production-adapters` 통합 브랜치를 최신 `origin/develop`에서 만든다.
3. PR 1 dx-note에서 Draft 수동 Codex review capability, 현재 CI 시간/비용 기준선과
   Temporal/boto3 최소 version spike를 고정한다.
4. capability 결과에 따라 Draft/Ready 또는 Ready+`ci:full` label gate를 선택하고 fast/full
   workflow, service image lock, Docker Compose profile을 구현한다.
5. PR 1 review와 선택된 service gate가 수렴한 뒤 PR 2를 시작한다.

구현 도중 public contract, exact release cut, authority owner가 바뀌면 이 계획을 먼저 수정하고 owner
승인을 다시 받는다. 내부 class/file 분할과 test fixture 세부 조정은 PR review 범위에서 처리한다.
