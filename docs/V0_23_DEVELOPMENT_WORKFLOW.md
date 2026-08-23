# v0.23 Production Adapters 개발 workflow

> 상태: v0.22 계승 workflow 승인, v0.23 CI 비용 정책 owner 승인 대기
>
> 작성일: 2026-08-23
>
> 기준 릴리스: `v0.22.0` / `6b4bd9f`
>
> 구현 브랜치: 범위 승인 뒤 `codex/v0.23-production-adapters` 생성

## 1. 결론

v0.23은 v0.22에서 검증한 순차 PR, PR별 사전조사, Codex 리뷰 수렴, merge commit,
통합 브랜치 동기화 방식을 유지한다. 서비스 어댑터와 분산 crash 경로가 추가되므로 다음 운영
규칙을 더한다.

1. PR은 Draft 상태에서 구현과 반복 리뷰를 진행한다.
2. 모든 fix push에는 빠른 CI만 실행한다.
3. PostgreSQL, MinIO, Temporal 실제 서비스 검증은 Ready for review 전환을 CI checkpoint로 삼아
   실행한다.
4. 한 리뷰 라운드의 지적은 한 수정 묶음으로 commit하고 push한다.
5. 오래된 workflow run은 `concurrency.cancel-in-progress`로 취소한다.
6. Ready 상태에서 코드가 바뀌면 기존 full-CI 증거를 폐기하고 Draft → Ready gate를 다시
   통과한다.
7. 같은 불변식 지적이 반복되면 push 횟수를 늘리지 않고 구조적 재검토 gate로 전환한다.
8. Ready gate 실행 중 Draft로 되돌리면 cancel-sentinel workflow가 expensive run을 즉시 취소한다.

이 정책은 리뷰 횟수와 서비스 CI 횟수를 분리한다. 수십 차례 리뷰가 필요한 PR도 실제 서비스
matrix는 구현 완료, 구조 변경 완료, 리뷰 수렴 시점에만 실행한다.

## 2. v0.22에서 이어받는 검증된 방식

v0.22 Git 이력과 [`V0_22_IMPLEMENTATION_PLAN.md`](V0_22_IMPLEMENTATION_PLAN.md)는 다음 순서를
기록한다.

- 통합 브랜치에서 각 구현 브랜치를 순차 생성했다.
- 앞 PR이 리뷰, 병합, 통합 브랜치 동기화를 마친 뒤 다음 PR을 시작했다.
- 각 PR 전에 `docs/dx-notes/`에 현재 동작, gap, 상태 전이, test matrix를 기록했다.
- 구현 PR마다 `$gh-pr-review-cycle`을 수렴시켰다.
- rebase와 force-push를 사용하지 않고 merge commit으로 리뷰 이력을 보존했다.
- 동일한 durability 지적이 반복될 때 구조적 재검토 gate를 적용했다.
- 마지막 통합 및 release PR에서 전체 test, compatibility, import, wheel, cold-start를 다시
  검증했다.

v0.22는 PR #114~#120, 통합 PR #121, release PR #122 순서로 이 workflow를 실제 수행했다.
v0.23은 같은 delivery discipline을 사용한다.

## 3. 브랜치와 문서 구조

### 3.1 브랜치

범위 승인 뒤 다음 구조를 만든다.

| 역할 | 이름 | base |
|---|---|---|
| v0.23 통합 브랜치 | `codex/v0.23-production-adapters` | 승인 시점의 최신 `origin/develop` |
| 구현 브랜치 | `codex/v0.23-prNN-<slug>` | 최신 v0.23 통합 브랜치 |
| 최종 통합 PR | v0.23 통합 브랜치 → `develop` | 최신 `origin/develop` 병합 후 |
| release PR | `develop` → `main` 또는 저장소 release 관례 | 통합 PR 병합 후 별도 승인 |

각 구현 PR은 하나의 직전 PR에만 의존한다. 아직 병합되지 않은 앞 PR commit 위에 다음 PR을
쌓지 않는다. 구현 브랜치와 통합 브랜치를 rebase하거나 force-push하지 않는다. 구현 PR은 merge
commit으로 통합한다.

### 3.2 문서 권위

| 문서 | 역할 |
|---|---|
| 이 문서 | 브랜치, PR, 리뷰, CI, merge 운영 규칙 |
| [`V0_23_IMPLEMENTATION_PLAN.md`](V0_23_IMPLEMENTATION_PLAN.md) | 승인된 기술 범위, 구조, PR 분할, test matrix |
| `docs/dx-notes/YYYY-MM-DD-v0.23-prNN-<slug>.md` | PR별 사전조사와 실행 증거; gitignored |
| [`V0_23_RELEASE_AUDIT.md`](V0_23_RELEASE_AUDIT.md) | 최종 shipped 범위, 검증 결과, 제외 범위 |

장기 계약이나 PR 경계가 바뀌면 구현 전에 구현 계획을 갱신한다. 세부 실험과 실패 기록은
dx-note에 남긴다. release audit에는 실제로 통과한 증거만 기록한다.

## 4. PR별 사전조사 gate

개발 브랜치를 만든 직후 다음 조사를 끝낸다.

1. 통합 브랜치, working tree, remote refs, `origin/develop` 진행 상태를 확인한다.
2. 변경할 public contract, durable codec, schema/migration, 호출 경로, 기존 conformance를 읽는다.
3. 정상, retry, response-loss, process-kill, stale writer, corrupt data 경로를 상태 전이로 적는다.
4. PostgreSQL transaction과 ObjectStore/Temporal 사이의 각 crash prefix를 적는다.
5. optional dependency와 root/import/wheel 경계를 확인한다.
6. 변경 파일, migration 영향, targeted test, 실제 서비스 test, compatibility gate를 확정한다.
7. 결과를 `docs/dx-notes/YYYY-MM-DD-v0.23-prNN-<slug>.md`에 기록한다.

각 dx-note에는 다음 항목이 들어간다.

- 기준 commit과 조사한 파일
- 현재 동작과 발견한 gap
- 소유권과 authority
- 상태 전이와 crash window
- 구조 결정과 대안
- 수정 파일과 migration
- local/CI test matrix
- 보안·privacy·비용 영향
- 범위 밖 항목
- 완료 조건

## 5. 한 구현 PR의 실행 루프

1. v0.23 통합 브랜치를 fetch하고 fast-forward한다.
2. `origin/develop` 전진분을 통합할 시점이면 merge하고 기본 회귀 gate를 실행한다.
3. 계획에 정한 구현 브랜치를 만든다.
4. PR별 사전조사 gate를 완료한다.
5. contract와 상태 전이를 먼저 고정하고 구현한다.
6. local targeted test, 관련 compatibility test, 문서를 갱신한다.
7. 변경 범위만 commit하고 push한다.
8. v0.23 통합 브랜치를 base로 Draft PR을 만든다.
9. Codex 리뷰 사이클과 수정 묶음을 반복한다.
10. CI checkpoint 조건을 충족하면 Ready for review로 전환한다.
11. 실제 서비스 integration과 PR 전체 gate를 green으로 만든다.
12. 최신 commit 기준 리뷰를 한 차례 더 수렴시킨다.
13. merge commit으로 통합한다.
14. 통합 브랜치를 fetch/fast-forward하고 merge 결과를 검증한다.
15. 필요한 시점에 최신 `origin/develop`을 통합하고 다음 PR을 시작한다.

## 6. Codex 리뷰 사이클

모든 구현 PR, 최종 통합 PR, release PR은 `$gh-pr-review-cycle` 절차를 사용한다.

1. PR URL, base/head, 인증, checkout, working tree를 확인한다.
2. review, unresolved thread, issue comment, review 요청 인식 상태를 한 snapshot으로 읽는다.
3. 각 지적을 현재 계약과 코드에 대조한다.
4. 필요한 수정은 관련 test와 함께 적용한다. 유지할 구현은 원 thread에 근거를 답한다.
5. 한 review snapshot에서 받은 지적을 한 수정 묶음으로 만든다.
6. local gate를 통과한 뒤 commit과 push를 한 번 수행한다.
7. 처리한 모든 thread에 변경 위치와 검증 결과를 답한다.
8. 별도 comment로 정확히 `@codex review carefully`를 남긴다.
9. 요청 인식과 새 review를 정해진 polling budget 안에서 확인한다.
10. 최신 commit 기준 actionable feedback가 없고 approved 또는 명확한 no-issues 결과가 있으면
    리뷰 수렴으로 판정한다.

review 요청 comment, thread reply, polling은 CI를 유발하지 않는다. 코드 push만 fast CI를
유발한다. reviewer에게 줄 증거는 local test 결과와 dx-note에 먼저 축적한다.

### 6.1 Draft PR 수동 리뷰 capability gate

OpenAI 공식 문서는 수동 리뷰를 PR comment의 정확한 `@codex review` trigger로 정의한다. Draft
PR 지원 여부는 별도로 명시하지 않는다. v0.23은 이 제품 동작을 추정으로 고정하지 않고 PR 1에서
다음 순서로 확인한다.

1. Draft PR의 알려진 head SHA에 `@codex review carefully`를 남긴다.
2. `$gh-pr-review-cycle`의 recognition budget 안에서 👀 reaction과 그 SHA를 대상으로 한 review를
   확인한다.
3. 반응이 없으면 repository Code review 설정, Codex cloud 연결, comment 작성자 권한, 정확한
   trigger를 확인하고 한 번만 다시 요청한다.
4. 같은 head를 Ready for review로 전환하고 같은 요청을 한 번 수행한다.
5. Draft에서는 실패하고 Ready에서는 성공하면 아래 label-gated fallback을 활성화한다.
6. 두 상태에서 모두 실패하면 Draft 제약으로 판정하지 않는다. Codex 연결 또는 서비스 장애로
   기록하고 review cycle을 중지한다.

공식 trigger와 troubleshooting 기준은
[OpenAI Codex GitHub code review 문서](https://learn.chatgpt.com/docs/third-party/github)를 따른다.

### 6.2 Draft 수동 리뷰 미지원 시 fallback

fallback은 PR을 Ready 상태로 유지하고 expensive CI의 시작 조건을 `ci:full` label로 옮긴다.

- `synchronize`는 L1 fast gate와 진행 중 L2를 취소하는 sentinel만 실행한다.
- `pull_request:labeled`에서 label이 `ci:full`일 때만 L2를 실행한다.
- `ci:postgres`, `ci:objectstore`, `ci:temporal`, `ci:combined`는 L2 service profile을 선택한다.
- 새 code push는 `ci:full`을 제거하고 같은 PR concurrency group의 진행 중 L2를 취소한다.
- review가 수렴한 최신 head에서 `ci:full`을 다시 붙여 L2 증거를 만든다.
- required L2 check는 현재 head/merge ref에 연결한다. 이전 SHA의 성공 결과는 merge 조건을
  만족하지 않는다.
- manual dispatch는 진단과 재현에 사용한다. merge gate의 정규 trigger는 PR label event다.

이 fallback은 Ready 상태의 Codex review와 실제 서비스 CI를 분리한다. review fix push 횟수가
늘어나도 L2는 명시적인 merge checkpoint에서만 실행된다.

## 7. CI 비용 제어 정책

### 7.1 네 단계의 검증

| 단계 | 실행 시점 | 검증 범위 | 원격 비용 |
|---|---|---|---|
| L0 local targeted | 구현 중, 모든 fix 전 | 바뀐 불변식의 단위/contract test, lint 대상, migration 또는 codec smoke | 없음 |
| L1 push fast gate | 모든 PR push | 짧은 lint/import/compile, curated unit·contract smoke, 변경 파일 정책 | 낮음 |
| L2 Ready service gate | Draft → Ready CI checkpoint | PR 범위의 실제 PostgreSQL·MinIO·Temporal integration, PR 전체 unit/contract | 중간 |
| L3 campaign/release gate | 통합 브랜치 merge, 최종 통합, release | 모든 adapter 결합, crash/restart matrix, Python/OS matrix, coverage, wheel/install/compatibility | 높음 |

“PR별 unit/contract”는 PR이 merge되기 전 해당 범위의 전체 unit/contract gate를 통과한다는
의미다. “adapter별 실제 서비스 integration”은 mock을 벗어나 실제 PostgreSQL, S3-compatible
MinIO, Temporal local server를 기동해 transaction, network, restart 동작을 검증한다는 의미다.
두 gate는 모든 fix push에서 실행되지 않는다.

### 7.2 Draft와 Ready 상태 전이

```text
Draft
  ├─ fix push → L1 fast gate
  ├─ review request/reply → CI 없음
  ├─ 구조 변경 → local gate + L1
  └─ 구현/리뷰 checkpoint 도달
         ↓ Ready for review
       L2 full/service gate
         ├─ green + 최신 review 수렴 → merge
         └─ code fix 필요 → Draft로 전환 + L2 취소 → 수정 묶음 → Ready 재전환
```

CI checkpoint는 다음 시점에 만든다.

1. 최초 구현과 PR 전체 local gate가 끝난 시점
2. 구조적 재검토로 설계를 바꾼 뒤 새 구조가 local gate를 통과한 시점
3. review feedback가 수렴해 merge 후보가 된 시점
4. 구현 PR merge 뒤 통합 브랜치 검증 시점
5. 최종 통합 및 release 시점

최초 implementation-ready와 review-converged가 같은 commit이면 L2를 한 번만 실행한다.

### 7.3 fix commit 규칙

- PR은 기본적으로 Draft로 생성한다.
- review 한 라운드의 지적을 모두 조사한 뒤 한 commit/push로 묶는다.
- push 전에 관련 L0 test를 통과시키고 결과를 PR thread와 dx-note에 적는다.
- 문서나 test만 바뀌어도 해당 변경을 검증하는 L0 gate를 실행한다.
- Ready 상태에서 코드 변경이 필요하면 먼저 Draft로 전환한다.
- Draft 전환은 진행 중인 L2 service run을 cancel-sentinel로 취소한다.
- 새 code commit은 이전 L2/L3 결과를 무효화한다.
- 최종 review는 최신 PR head SHA를 대상으로 한다.
- L2는 그 head와 current base의 GitHub merge ref를 검증한다.
- L2 artifact는 `github.event.pull_request.head.sha`와 `GITHUB_SHA`를 함께 기록한다.
- head 또는 base가 바뀌면 merge candidate가 바뀌므로 L2 evidence를 다시 만든다.

`[skip ci]`, `[ci skip]`은 비용 제어 수단으로 사용하지 않는다. GitHub는 skip된 required
workflow를 Pending으로 남길 수 있어 merge gate가 불명확해진다. 비용은 workflow 분리, Draft
상태, CI checkpoint, concurrency 취소로 제어한다.

공식 근거:

- [GitHub Actions pull_request activity types](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)
- [Workflow concurrency와 진행 중 run 취소](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency)
- [Skip된 required check의 Pending 동작](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/skip-workflow-runs?apiVersion=2022-11-28)

### 7.4 workflow 분리 기준

CI foundation PR에서 현재 단일 `.github/workflows/ci.yml`을 다음 의미로 분리한다.

| workflow | trigger | 내용 |
|---|---|---|
| fast PR | 모든 `pull_request` push | L1 fast gate |
| full PR | `ready_for_review`, 명시적 manual dispatch | L2 PR gate |
| full cancel sentinel | `converted_to_draft` | 같은 PR의 진행 중 L2 run 취소; code checkout/test 없음 |
| integration | v0.23 통합 브랜치와 `develop` push | 영향받은 서비스와 combined gate |
| release | release PR/tag/manual | L3 전체 matrix와 wheel audit |

PR 1 capability gate가 Draft 수동 Codex review 미지원을 확인하면 앞의 두 PR workflow를 다음과
같이 대체한다.

| fallback workflow | trigger | 내용 |
|---|---|---|
| full PR | `pull_request:labeled`의 `ci:full` | 선택된 service profile의 L2 PR gate |
| full cancel sentinel | `synchronize`, `ci:full` 제거 | 같은 PR의 진행 중 L2 취소와 stale gate label 정리 |

기본 Draft/Ready 경로에서 Ready 뒤 새 code push에는 full workflow를 자동 실행하지 않는다. 현재
SHA의 full check가 없으므로 merge가 차단된다. 개발자는 Draft로 돌려 수정 묶음을 만든 뒤 Ready로
다시 전환한다. fallback 경로에서는 PR 상태를 유지하고 `ci:full` label을 제거한 채 수정한 뒤 최신
head에 label을 다시 붙인다. 긴 review cycle에서 의도하지 않은 서비스 matrix 반복을 두 경로 모두
막는다.

fast workflow는 자체 key를 사용한다.

```yaml
concurrency:
  group: fast-pr-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true
```

full PR workflow와 cancel-sentinel workflow는 의도적으로 같은 key를 사용한다.

```yaml
concurrency:
  group: full-pr-${{ github.event.pull_request.number || inputs.pr_number || github.ref }}
  cancel-in-progress: true
```

새 Ready checkpoint는 이전 L2를 취소한다. `converted_to_draft` sentinel도 같은 key로 진행 중 L2를
취소한 뒤 즉시 끝난다. GitHub concurrency group은 같은 repository의 서로 다른 workflow 사이에도
적용된다. workflow 이름을 key에 넣지 않는 이유가 이 cross-workflow cancellation이다. release
artifact publish 단계는 별도 concurrency group과 `cancel-in-progress: false`를 사용한다.

### 7.5 서비스 profile

구현 계획은 각 PR에 필요한 profile을 지정한다. PR을 Ready로 전환하기 전에 다음 label 중 하나를
붙인다.

| label | L2 실제 서비스 |
|---|---|
| `ci:core` | 외부 서비스 없음 |
| `ci:postgres` | PostgreSQL |
| `ci:objectstore` | PostgreSQL + MinIO |
| `ci:temporal` | Temporal local server와 필요한 backing service |
| `ci:combined` | PostgreSQL + MinIO + Temporal 전체 경로 |

label은 실행 범위를 줄이는 선언이다. 계획 문서의 PR profile과 다른 label은 full gate가 실패해야
한다. 전체 required gate job은 항상 존재하며 conditional service job의 skip/result를 검증한다.
경로 필터만으로 required workflow 전체를 생략하지 않는다.

### 7.6 언어와 서비스 matrix

- L1: Python 3.12 한 버전, 외부 서비스 없음
- L2 core: Python 3.11/3.12 unit·contract
- L2 service: Python 3.12와 해당 실제 서비스
- 통합 checkpoint: Python 3.12 combined service path
- release: Python 3.11/3.12, PostgreSQL 최소/최신 지원 major, Linux service matrix,
  Windows/macOS package smoke

서비스 image, Temporal SDK, local server binary는 campaign lock에 pin한다. version bump는 별도
review 가능한 commit으로 수행한다.

### 7.7 반복 실패 budget

다음 조건에서 새 fix push와 full-CI 재실행을 멈춘다.

- 같은 불변식 지적이 두 review 라운드에서 반복된다.
- 같은 L2 service failure가 두 checkpoint에서 반복된다.
- 한 수정이 서로 다른 두 crash boundary를 번갈아 깨뜨린다.
- 조건문이 core, adapter, workflow, dispatcher 여러 층으로 퍼진다.
- mutation 결과는 green이지만 canonical readback 또는 mutant test가 실패한다.

이때 구조적 재검토 gate를 수행하고 dx-note와 구현 계획을 먼저 갱신한다. 구조가 확정된 뒤 다음
수정 묶음을 만든다.

## 8. 실제 서비스 integration 운영

v0.23에서 adapter별 실제 서비스 integration은 가능하며 release 조건으로 사용한다.

| adapter | PR/CI 서비스 | 검증할 핵심 |
|---|---|---|
| PostgreSQL | GitHub service container의 PostgreSQL 16 | row lock, transaction, lease expiry, generation handoff, response-loss readback, migration |
| ObjectStore | MinIO pinned container | conditional put, checksum, multipart, missing/corrupt, orphan과 PG rollback |
| Temporal | `WorkflowEnvironment.start_local()`이 띄운 실제 local server | Workflow/Activity/Signal, worker restart, heartbeat/cancel, Continue-As-New |
| 결합 경로 | PostgreSQL + MinIO + Temporal | admission → signal → activation → checkpoint/event/terminal → retry/reconnect |

mock/fake test는 failure injection과 상태 공간을 빠르게 넓힌다. 실제 서비스 test는 driver, protocol,
transaction, process boundary가 같은 의미를 지키는지 증명한다. cloud credential이 필요한 AWS S3나
Temporal Cloud test는 blocking public CI에 넣지 않는다.

GitHub는 PostgreSQL service container를 공식 지원한다.
[Temporal `WorkflowEnvironment`](https://python.temporal.io/temporalio.testing.WorkflowEnvironment.html)는
time-skipping server와 local development server를 시작할 수 있다. MinIO는 S3-compatible contract를
로컬 container에서 검증한다.

local 개발에는 repository-owned Docker Compose profile을 제공한다.

```text
docker compose --profile postgres up
docker compose --profile objectstore up
docker compose --profile temporal up
docker compose --profile durable-host up
```

실제 명령과 image version은 해당 adapter PR의 dx-note와 runbook에서 고정한다.

## 9. 구조적 재검토 gate

구조적 재검토는 열린 feedback를 보존한 채 다음을 다시 작성한다.

1. authority owner와 linearization point
2. 정상 및 모든 crash prefix
3. failure precedence와 retry eligibility
4. durable record identity와 compatibility
5. process replacement와 stale writer 동작
6. private bytes와 public metadata 경계
7. mutant가 증명해야 할 결함 구현

근본 원인을 하나의 type, protocol, state machine, transaction 또는 capability로 고정한다. 계획과
dx-note를 갱신한 뒤 같은 PR에서 리뷰를 재개한다.

## 10. merge와 통합 브랜치 운영

- actionable review thread, latest review approval/no-issues, required checks를 모두 확인한다.
- merge 직전 head SHA, current base SHA, L2 merge SHA가 evidence와 일치하는지 확인한다.
- merge commit으로 통합하고 remote 구현 브랜치를 삭제한다.
- 통합 브랜치를 다시 fetch/fast-forward한 뒤 merge 결과를 검증한다.
- schema 또는 public contract PR 뒤에는 통합 checkpoint를 실행한다.
- `origin/develop` 전진분은 PR 사이의 안전한 경계에서 통합한다.
- conflict resolution은 별도 commit으로 남기고 관련 regression을 다시 실행한다.

## 11. 최종 통합과 release gate

마지막 구현 PR 뒤 다음 순서로 캠페인을 닫는다.

1. 최신 `origin/develop`을 v0.23 통합 브랜치에 merge한다.
2. PostgreSQL + MinIO + Temporal combined crash/restart matrix를 실행한다.
3. Python 3.11/3.12 full suite, coverage, lint, import, compatibility를 실행한다.
4. minimal 및 모든 optional adapter extra의 wheel/install/cold-start를 검증한다.
5. migration N/N+1, rolling reader, rollback floor를 검증한다.
6. `V0_23_RELEASE_AUDIT.md`의 각 항목에 실제 command, 결과, artifact를 기록한다.
7. 통합 브랜치 → `develop` PR에서 Codex 리뷰 사이클을 수렴시킨다.
8. 병합 뒤 별도 승인으로 release PR과 tag/publish를 진행한다.

release audit의 빈 칸, 계획 상태, 예상 결과는 release 증거로 취급하지 않는다.

## 12. PR evidence 최소 형식

PR 본문 또는 마지막 summary에는 다음을 남긴다.

```text
Scope / non-scope
Authority and crash boundary changed
Migration or durable-format impact
Local targeted tests
L1 fast workflow URL/result
L2 service profile and workflow URL/result
Compatibility and wheel/import result
Review convergence status
Remaining risks and follow-up
```

provider dispatch count, unknown dispatch retry count, terminal winner, stale writer rejection처럼 비용이나
권위를 직접 증명하는 숫자는 artifact로 보존한다.

## 13. common-skill-pipeline 보호 규칙

v0.23 개발은 `monoid-agent-kernel` repository에서 독립적으로 진행한다. active
`common-skill-pipeline`은 다음 원칙으로 보호한다.

- CSP working tree, branch, refs, dependency, migration, test data를 변경하지 않는다.
- CSP를 v0.23 CI fixture나 git submodule로 사용하지 않는다.
- CSP schema, tenant/RLS, chat/Skill/Catalog type을 Monoid public contract에 넣지 않는다.
- 필요한 요구는 provider-neutral contract와 conformance case로 환원한다.
- CSP integration은 v0.23 release 뒤 CSP 소유의 별도 campaign에서 수행한다.

## 14. 이 문서의 승인 상태

사용자는 다음 workflow 방향을 승인했다.

- v0.22 순차 PR과 리뷰 workflow 계승
- PR별 사전조사 dx-note
- 실제 서비스 integration
- 최종 packaging/release gate
- v0.23 중간 범위 캠페인

이 문서가 추가로 고정한 CI 비용 정책은 구현 계획 승인과 함께 최종 확정한다. 기술 범위와 PR
profile은 [`V0_23_IMPLEMENTATION_PLAN.md`](V0_23_IMPLEMENTATION_PLAN.md)의 승인 표가 소유한다.
