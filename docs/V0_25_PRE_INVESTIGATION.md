# v0.25 통합 개발 계획과 사전조사

작성일: 2026-09-28. 통합 갱신: 2026-10-28. 최초 코드 기준: v0.24.0 / `8d56e0d`.

현재 상태: **P1–P5·DX1·DX2 구현과 develop 통합 완료** (`b27a15c`, [#170](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/170)).
아래는 최초 조사와 구현 과정의 기록이다. 출시 준비 상태는 [release audit](V0_25_RELEASE_AUDIT.md),
사용자의 업그레이드 절차는 [v0.25 migration guide](migrations/v0.25.md)에 둔다.

구현 기록: **P5 #148·DX1 #149·P3 #150–155·P1/P2 #156–161 develop 병합**.
P1P2-I #161의 develop 반영은 `2914c4366f8dfd7ed537f854388836a12a05f078`이다.
공개 API·v2 저장·복구·gateway·Studio·OTel 연결과 누적 검증을 마쳤다.
구체적인 정책·외부 근거·PR 범위는 [P1·P2 PR 계획](V0_25_P1_P2_PR_PLAN.md)에 둔다.
P4-00 [#162](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/162)는 develop에,
P4-01·02 [#163](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/163)·[#164](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/164)는
통합 브랜치 `be35567`에 병합했다. [P4-I #165](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/165)를 통해
develop `973e474`에 최종 통합했다. P1–P5와 DX1의 구현은 완료됐다.
2026-10-28 DX2-00 [#166](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/166)을 develop에,
DX2-01–03 [#167](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/167)·[#168](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/168)·[#169](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/169)를
통합 브랜치 `3bd3095`에 병합했다. 설치 진입·스트림 소유권·새 프로세스 승인 재개를 구현했으며,
최종 누적 검증과 develop 통합 기록은 [DX2 PR 계획 §9](V0_25_DX2_PR_PLAN.md#9-계획-검토와-진행-기록)에 둔다.
정책·외부 근거·회귀·리뷰·CI 기록은 [P4 PR 계획](V0_25_P4_PR_PLAN.md)에 둔다.
P3의 작업 지시는 [P3 PR 계획](V0_25_P3_PR_PLAN.md), 앞선 실행 기록은 P5·DX1 PR 계획 §9에 있다.
PR 작업 지시는 [P5·DX1 PR 계획](V0_25_P5_DX1_PR_PLAN.md), 목표 API와 후속 방향은
[개발자 경험 설계](V0_25_DEVELOPER_EXPERIENCE_PROPOSAL.ko.md)에 기록한다. 릴리스 범위의 기준은 이 문서다.

## 1. 범위와 결정 상태

v0.25의 목표는 제품 통합 개발자가 타입 있는 도구를 연결하고, 실행을 멈추고, 관측한 결과를 설명할 수 있게 하는 것이다.
기존 종료·사용량·종료 대기·도구 자원 접근 계약에 DX1 typed `@tool`과 DX2 통합 예제를 포함한다.
사용자가 결정한 P3의 v0.25 포함과 P7의 후속 이관을 유지한다. 개발 단위는 독립 작업군이며,
여러 작업군을 develop에 통합한 뒤 적절한 시점에 하나의 릴리스로 마감한다.

| 구분 | 항목 | 완료 기준 |
|---|---|---|
| 필수 | P1 종료 사실 | 기존 interruption/provider 분류와 일관된 호출 종료 정보 전달 |
| 필수 | P2 사용량 관측 | 확인된 수량과 미확인 부분을 구분하고 시도·복구·누적 집계의 의미 보존 |
| 필수 | P3 도구 확장점 | 호스트 자원의 주입·권한·수명·복구를 설명하는 공식 확장 계약 |
| 필수 | P4 종료 예산 조회 | 진행 중 호출의 유효 deadline·drain·grace를 조회하는 일관된 snapshot |
| 필수 / 첫 PR | P5 권한 상실 후 출력 | 일반 delta·terminal 정규화·잔여 flush의 전달 직전 권한 검사와 관련 경합 회귀 |
| 필수 / 첫 PR군 | DX1 typed `@tool` | 검증된 입력 타입 보존, 최상위 BaseModel 결과의 JSON 변환, 기존 실패·observation 계약 유지 |
| 필수 / 기능별 동반 | DX2 통합 예제 | 순수 도구·요청 스트리밍·승인/재시작을 공개 계약과 설치 패키지로 재현 |
| 조건부 | P8 자식 중단·drain | opt-in 상속·전파·대기 정책과 사용량 중복 집계 방지 |
| 독립 유지보수 | P6, P9 | 선택 인자 규칙·예제 점검, 확인된 커널 결함 수정, CI·릴리스 개선 중 완결된 작업 |
| 후속 | P7 실행 격리 | 별도 backend/OS 강제 능력 조사와 구현 |

위 범위를 개발 계획에 반영한다. API 이름·필드·codec 버전과 후속 작업군의 PR 분할은 구현 직전 확정한다.
첫 두 PR의 작업 단위와 수용 기준은 별도 PR 계획에 고정했다. 필수 범위를 후속 릴리스로 변경하면 범위 결정을 기록한다.
FastAPI/Vercel 어댑터, 새 Studio 진단 화면, context compaction은 후속 후보로 유지한다.

## 2. 조사 방법과 기준선

2026-09-28 조사에서는 세 GPT-6 Sol xhigh 에이전트가 P1/P2, P3, P4/P5/P6/P8을 분담했고, 부모 에이전트가
선례·CI·패키징·릴리스와 조사 결과의 일관성을 확인했다. 소스 경로·상태 전이·기존 테스트를 대조하고
관측 사실, 새 계약 제안, 구현 전 검증 항목을 구분했다. P1/P2와 P3 담당자가 서로의 설계를 검토했고,
예산·취소 담당자는 통합 문서의 범위·시간 계약·권한 보장 표현을 검토했다.

다음 표의 refs·서비스 lock·73개 baseline은 **2026-09-28의 기록**이다. 구현 착수 시 최신 develop을 다시 확인한다.

| 항목 | 확인 결과 |
|---|---|
| local/remote main | `8d56e0df790a0ca518295545a0439b214d7b2d80` |
| remote develop | `76a15101c7f2bfa0b56a3926365407d3c498642d` |
| main/develop 소스 차이 | 조사 시점 tree 차이 없음 |
| 실행 환경 | 저장소 `.venv`, Python 3.11.9, workspace `src` import 확인 |
| focused baseline | 공개 표면·호환성·CI·패키지·기존 drain/문자 경계 73 passed, 8.67s |
| service lock | `validate-lock` 성공; digest `5e536963c707778e9309936d5768edc367a6c49ece5c51223999dde46697b498` |
| P5 진단 | drain OFF + high surrogate 보류 + authority revoke에서 추가 U+FFFD 전달 재현 |

전체 pytest·실제 서비스 기동·wheel build/install·배포는 이번 조사에서 수행하지 않았다.
기능 코드를 수정하지 않았으며, 위 baseline은 새 API 구현의 검증 결과를 뜻하지 않는다.

상세 실험과 조사 과정은 로컬 gitignored PR별 조사 기록에 보존한다:

- `2026-09-28-v0.25-pre-investigation.md`: 기준선·선례·CI·릴리스 및 baseline 명령.
- `2026-09-28-v0.25-p1-p2-pre-investigation.md`: 종료·사용량 상태표와 전달 경계.
- `2026-09-28-v0.25-p3-pre-investigation.md`: 도구 확장점·권한·수명 조사.
- `2026-09-28-v0.25-p4-p5-p6-p8-pre-investigation.md`: 예산·권한 상실 재현·취소·자식 정책.

이 문서의 결론과 구현 검증 기준은 해당 내부 파일을 배포하지 않아도 이해할 수 있도록 기록한다.

2026-10-25 통합 조사에서는 같은 코드 HEAD에서 P5의 일반 delta와 drain 시작 전 잔여 출력 문제를 추가 재현했다.
Python 3.11.9 / Pydantic 2.13.4 환경의 관련 기존 테스트는 **76 passed / 10.75s**였다.
이 green baseline은 새로 재현한 경합을 잡지 못한다. 구현 PR에서 신규 회귀가 수정 전 실패하는 증거를 추가한다.
외부 이슈·고정 revision의 구현·재현 명령·실행한 테스트 목록은 [PR 계획 §2·§7](V0_25_P5_DX1_PR_PLAN.md)에 보존한다.

## 3. 통합 제품과 커널의 책임

커널은 여러 호스트가 사용할 실행 사실·확장점·호환성 계약을 제공한다.
통합 제품은 사용자 경험과 도메인 정책을 구성한다.

| 경계 | 커널 | 소비 호스트 |
|---|---|---|
| 종료 | 관측한 요청·경계·결과와 우선순위 | UI Stop·연결 단절·제품 cause 매핑 |
| 사용량 | 확인값·관측 경로·완전성·재시도/복구 의미 | 가격·청구·환불·journal과 비용 부담 정책 |
| 도구 | 실행 권한·자원 접근·수명 계약 | 도메인 객체·tenant·자원 구성과 외부 효과 |
| 배포 | 커널 패키지·fixture·서비스 검증 | 핀 상향·adapter 변경·제품 수용 검증·기능 활성화 |

커널 조사·CI는 자체 fixture와 공개 계약을 사용한다. 소비자 마이그레이션은 해당 제품의
배포 절차에 따라 검증한다.

## 4. 선례와 v0.25에 적용할 운영 방식

| 선례 | 계승할 점 | 적용 범위 |
|---|---|---|
| [v0.22 계획](V0_22_IMPLEMENTATION_PLAN.md) §14 | PR별 사전조사, 상태 전이, 리뷰 수렴, merge commit, 반복 결함의 구조 재검토 | 모든 작업 PR |
| [v0.23 workflow](V0_23_DEVELOPMENT_WORKFLOW.md) §4~12 | Draft/Ready CI, service profile, local/remote 증거 분리, 패키지·호환성 검증 | 검증·리뷰·릴리스 |
| [v0.24 감사](V0_24_RELEASE_AUDIT.md) | 작은 기능·수정 PR의 develop 직접 통합, source와 artifact 구분, 실패·미실행 기록 | 독립 작업군·마감 |

단일 PR은 최신 develop에서 시작한다. 서로 의존하는 여러 PR이 필요한 작업군만 전용 통합 브랜치를 사용한다.
그 내부에서는 선행 PR의 리뷰·병합·동기화 뒤 다음 PR을 진행한다. 다른 작업군은 별도의 완료 단위를 유지한다.

공유 파일 수정은 충돌·회귀 조정 대상이다. 새 공개 타입이나 저장 형식을 실제로 사용하는 관계만 필수 선행
의존성으로 기록한다. 버전 범프·최종 감사·배포는 develop에 필요한 결과가 모인 뒤 한 번의 릴리스 마감에서 처리한다.

현재 적용되는 AGENTS와 campaign/preflight/review 절차를 따른다. v0.23 문서의 과거 외부 Codex connector
리뷰 요청이나 외부 clean 판정을 새 작업의 gate로 복원하지 않는다. 독립 설계 리뷰와 구현 최종 head의 검증을 구분한다.

## 5. 분야별 조사 결론

### 5.1 P1·P2: 종료 사실과 사용량 증거

2026-10-27 최신 develop 조사, 실제 cancel-drain probe, v2 reader/writer·rollback 정책과 PR별 수용 기준은
[P1·P2 PR 계획](V0_25_P1_P2_PR_PLAN.md)에 구체화했다. 아래 원칙을 유지하며 세부 계약은 새 계획을 따른다.

**현재 간격.** [usage 정규화](../src/monoid_agent_kernel/providers/_common.py)는 누락된 token 항목을 0으로 채우고,
[runner](../src/monoid_agent_kernel/model_call.py)의 `_streamed_usage`는 값이 모두 0인 terminal usage를
미보고로 처리한다. 확인된 0과 누락의 구분은 adapter 입력에서 이미 사라질 수 있다.
receipt에 필드를 추가하는 작업과 함께 최초 provider 수신 경계의 관측 정보를 보존해야 한다.

[ModelCallReceipt와 attempt](../src/monoid_agent_kernel/core/model_io.py)는 현재 정수 usage와 시도별 합계의
일치를 요구한다. [ModelStreamOutcome](../src/monoid_agent_kernel/core/model_stream.py)은 다섯 status를
가지지만 최초 중단 요청과 실제 읽기를 끝낸 계기를 함께 제공하지 않는다. loop의 실패 observer outcome에도
drain에서 회수한 usage가 현재 전달되지 않는다. 이 전달 누락을 P1/P2의 수용 기준에 포함한다.

**최소 계약 권고.** 기존 숫자 usage와 provider `stop_reason`은 유지하고 다음 증거를 추가한다.

| 축 | 권고 의미 | 경계 |
|---|---|---|
| 호출 결과 | 기존 completed/interrupted/failed/cancelled/timed_out와 일관된 분류 | durable dispatch의 확인 여부는 별도 축 |
| 최초 중단 요청 | 처음 수락한 cancel/interrupt의 원인; 요청이 없으면 null | token 요청 없이 감지한 deadline을 요청 원인으로 합성하지 않음 |
| 실제 종료 계기 | provider terminal, deadline, drain 만료, lease loss 등 읽기·대기를 끝낸 경계 | cancel 후 deadline 도달도 원래 취소 결과를 보존 |
| 확인된 사용량 | provider 입력에서 확인한 항목만 담는 희소 정수 맵 | 명시적 0은 보존; 누락값을 확인값으로 승격하지 않음 |
| 출처·수집 경로 | provider/adapter 보고 출처와 일반 완료/drain 수집 경로 | recovery는 재전달 경로이며 최초 provenance를 유지 |
| 관측 범위 | 어느 dispatch에서 관측이 있었는지와 counter별 완전성을 분리 | `complete_keys`류 항목별 기준, attempts=0은 `no_dispatch` |

공개 필드와 bounded vocabulary는 P1·P2 PR 계획 §4에 정했다. 모든 dispatch에서 input만 보고했다면 input만 완전하다.
이 상태를 전체 token 사용량이 확정된 것으로 표현하지 않는다. first-party adapter의 검증된 raw 입력은
provider 보고로 기록할 수 있다. 출처를 입증할 수 없는 구형/외부 exception stamp는
`legacy_ambiguous` 또는 adapter 보고로 남긴다. malformed 입력의 부분 수용은 별도 호환성 결정이다.
추정기·가격·청구 판단은 이번 최소 계약에 포함하지 않는다.

| 대표 상태 | 보존해야 할 결과 |
|---|---|
| dispatch 전 cancel 또는 deadline | 실제 호출 0회, `no_dispatch`; 관측하지 않은 provider usage를 생성하지 않음 |
| 성공이며 provider가 명시적 0 보고 | 확인된 0과 해당 counter의 관측 사실 |
| 첫 attempt 사용량 확인, 마지막 attempt 실패·미보고 | 이전 확인값의 부분 합계와 미확인 시도의 존재 |
| user cancel 후 drain에서 terminal 수신 | 취소 결과, 최초 요청, provider terminal로 닫힌 사실, 회수한 usage |
| user cancel 후 deadline이 drain 종료 | 취소 결과 유지, `ended_by=deadline`, 확보한 부분 증거 |
| lease loss | 권한 상실 사실 유지; 권한 없는 새 출력·영속 게시 차단 |
| durable dispatch unknown | 미확정 상태와 자동 유료 재시도 금지 유지; receipt/result 합성 금지 |
| settled 복구·반복 재노출 | 최초 provenance 보존, 이미 집계한 usage의 중복 합산 방지 |

**영속 호환성은 G1의 공통 변경이다.**
[durable invocation](../src/monoid_agent_kernel/core/model_invocation.py)의 v1 receipt는 닫힌 필드 목록이고,
`unknown` 상태에는 receipt/result를 넣을 수 없다. [model-calls projection](../src/monoid_agent_kernel/core/model_calls.py)과
[JSON Schema](../src/monoid_agent_kernel/core/schemas.py)도 함께 검토해야 한다.
P1과 P2의 영속 필드를 한 번에 설계하고, 지원할 새 format의 dual-reader 준비 → 새 writer 활성화 →
혼합 버전·rollback 검증 순서를 하나의 변경 계획으로 둔다. 구형 reader가 새 format을 읽는다고 가정하지 않는다.
rollback에는 구형 writer 전환 뒤에도 새 record를 읽을 수 있는 reader 유지 또는 검증된 데이터 전환이 필요하다.

구체안은 invocation/model-calls v2와 v1 dual-reader다. immutable settled 좌표를 재게시할 때 원래 검증 payload/hash를
보존한다. reader 준비 버전을 먼저 배포하고 새 writer를 활성화한다. rollback은 v2 reader를 유지한 버전으로 수행한다.
private replay와 model-content는 기존 projection을 유지하고, live/gateway의 optional 필드와 OTel 진단을 별도로 연결한다.

`unknown`의 partial-evidence 영속 slot 신설은 v0.25 필수 범위에서 제외하는 것을 권고한다.
현재 프로세스의 exception에 남은 부분값과 재시작 후 canonical unknown의 증거 부재를 명시한다.
settled의 새 증거와 구형 fixture의 `legacy_ambiguous` 처리는 필수로 검증한다.
private replay의 마지막 provider turn usage와 논리 호출의 재시도 누적 receipt usage도 각각의 의미를 유지한다.

**변경 지도.** provider ingress → runner/exception → attempt/receipt → observer·ledger·durable codec →
loop의 회계·복구 순서로 전달을 검증한다. 추가로 `ValidatedCallRunner`, Reference gateway 오류 envelope,
backend live-stream closed frame과 Studio reader, OTel의 attempt/call 계측을 대조한다.
각 소비 면에 새 필드 전달·기존 projection 유지 중 어떤 정책을 적용할지 기록하고,
전달하는 wire에서는 schema·reader·fixture를 함께 갱신한다. 새 증거 metadata를 별도 token 가산 경로로 사용하지 않는다.

### 5.2 P3: 타입 있는 주입과 호출별 권한

2026-10-27 구체화한 정책·외부 근거·코드 재현·PR 의존성·수용 행렬은 [P3 PR 계획](V0_25_P3_PR_PLAN.md)에 둔다.
P3-01에서 기존 cleanup을 보완했고, P3-02에서 root 주입·권한·복구와 보고서 소비자,
P3-03에서 child/fork 명시 상속을 구현했다. 동기 activation adapter, 호출에 고정한 façade,
deferred workspace, 별도 P3 lifetime guard를 채택했다. 구현 실행 기록은 P3 PR 계획 §9에 둔다.

**최초 조사에서 확인한 간격.** [ToolContext](../src/monoid_agent_kernel/tools/base.py)는 method-only Protocol이고,
[extension façade](../src/monoid_agent_kernel/loop.py)는 내부 workspace·recorder를 감춘다.
[bootstrap](../src/monoid_agent_kernel/loop_phases.py)은 `workspace_factory`의 반환값을 받은 뒤
authority wrapper를 만든다. factory가 원본 workspace 참조를 도구에 따로 전달하면
그 참조에는 커널 wrapper의 revoke 검사가 적용되지 않는다. 자원 factory와 handler 사이에서도
이 권한 경계를 유지하도록 커널 소유 fixture에서 검증한다.

**구현한 공개 표면.** keyword-only `host_tool_bindings`와
`monoid_agent_kernel.tools.host_bindings`의 `HostToolKey[T]`, `HostToolBinding[T]`,
`HostToolResource[T]`, `HostToolActivation`, `HostToolCallLease`, `ToolWorkspace`,
`require_host_tool_binding`을 제공한다. identity key와 `accepts` 검증자를 사용하고,
기존 `ToolContext` 필수 멤버를 유지하는 선택적 확장 Protocol로 구조적 구현자의 호환성을 보존한다.

커널이 제공하는 workspace 기능은 `read_bytes`/`write_bytes`/`exists`다.
각 작업에서 activation·원래 tool call의 lease·binding path policy·workspace authority를 검사한다.
[AuthorityBoundWorkspace](../src/monoid_agent_kernel/core/workspace.py)만으로 호출별 path policy까지 보장할 수 없다.

교차 검토에서 **보관된 view가 다음 호출의 권한을 빌리는 문제**를 발견했다. 현재 context의 `_live_call`은
최신 fallback call을 참조할 수 있다. 새 기능은 handler lookup 시 원래 call identity와 revocable lease를
고정해야 한다. `get_tools`는 키·접근자를 보관하고, 실제 작업 가능한 view는 handler 호출 안에서 얻는다.
새 view 발급 자체도 원래 handler-bound scope에 귀속시키며 최신 fallback call을 발급 근거로 사용하지 않는다.
완료·abandon된 worker와 그 하위 thread가 보관된 view를 사용하거나 옛 context에서 새 view를 조회해
다음 call의 권한으로 작업하지 못하는 회귀를 필수로 둔다.

수명 규칙은 P3 안에서 완결한다.

- loop 생성 시 등록 목록을 snapshot하고 activation별 registry를 만든다. context 생성 뒤 연결하고 provider 등록 전에 제공한다.
- factory가 도중에 실패하면 앞서 생성한 자원과 recorder/job 자원도 정리하도록 partial cleanup을 등록한다.
- root/child/fork는 별도 registry와 view를 갖는다. 기본 상속은 끄고, 명시 상속 시 자식용 factory를 새로 실행한다.
- `inherit_to_children=True`인 등록을 각 hop에 전달하고 child/fork에서 fresh adapter를 연다.
- 상속한 자원은 child-local guard와 parent activation/writer lineage를 모두 검사한다. parent 종료·revoke 뒤 접근을 차단한다. callback 등록 없이 수명 token을 연결하며 자식 durable writer identity는 별도로 유지한다.
- restore는 호스트 재주입으로 fresh view를 만든다. factory가 checkpoint 적용 전 workspace 상태를 확정해 저장하지 않도록 순서를 정한다.
- close/discard/revoke 및 성공한 `release_parked`/`arelease_parked` 뒤 retained view를 거부한다.
- 정상 park release는 activation guard를 무효화하고 resource를 역순으로 한 번씩 정리하며 checkpoint·hosted task·재개 가능성을 보존한다.
  미commit park의 release 거부는 자원을 유지한다. cleanup 실패·async 취소 뒤 discard에서도 중복 close를 막고 남은 정리를 수행한다.
- key identity·임의 객체·secret은 checkpoint나 모델 요청에 직렬화하지 않는다. revoke 뒤 cleanup의 외부 flush는 host의 fence를 따른다.

일반 호스트 객체의 타입 검사와 등록 불변성은 내부 mutation을 자동 통제하지 않는다.
커널 view의 보장과 호스트 adapter의 외부 효과 책임을 명시한다.
[authority의 external-call fence](../src/monoid_agent_kernel/core/authority.py)는 이미 시작한 외부 효과를 되돌리지 않는다.
P3는 P8의 opt-in 중단·drain 정책과 독립적으로 새 capability의 권한·수명을 검증한다. P7 OS 격리는 후속 범위다.

### 5.3 P4: 실제 제어와 일치하는 종료 예산

**2026-10-27 최신 develop 조사.** 실제 runner에서 wall clock 전진에 따른 조기 timeout과 후퇴에 따른
기한 초과 재시도를 재현했다. 당시 기존 관련 회귀 209개는 통과했으며 01에서 수정 전 clock 실패 회귀를 고정했다.
당시 `_open_drain_budgets`는 positive-drain stream만 담았다. 02에서 one-shot·drain=0·phase별 grace·남은 시간을 담는 owner registry로 교체했다.
정책·외부 문제와 답안·파일 지도·PR별 검증은 [P4 PR 계획](V0_25_P4_PR_PLAN.md)에 둔다.

**구현 결과.** P4-01에서 core run/call/drain/shared wait를 process monotonic cutoff로 정렬했다.
공개 epoch deadline은 최초 소유 경계에서 한 번 변환하고 retry·validated repair에 같은 cutoff를 전달한다.
checkpoint의 기존 `remaining_duration_s`와 parked downtime 제외 정책, 영속 wall timestamp는 유지한다.

P4-02에서 runner가 소유한 모든 acall의 immutable snapshot을 공개하고 RunStream을 실제 소비자로 연결했다.
wait 진입·stream close·drain final cleanup의 기존 grace sampling 시점을 보존하며 현재 예약과 남은 창을 구별한다.
여러 cleanup 창은 각각 표시한다. 조회는 host getter·취소·I/O를 실행하지 않는다.
call ID는 process-local unique ID, 빈 calls는 해당 runner의 active model wait 부재다.

RunStream은 현재 call의 drain/cleanup 잔여와 exit 정책 후보에서 allowance를 정하고 병렬 call은 max로 조합한다.
최초 exit wait의 cutoff를 고정해 실제 `hard_cancel_remaining_s`를 표시한다. 강제 취소 뒤 driver 정리는 계속 await한다.
exit waiter가 취소돼 timer 집행이 중단되면 `wait_interrupted/None`으로 표시한다. 재진입 시 최초 cutoff를 재사용한다.
기존 raw `RunStream(current_extra_grace_s=)` 동작을 보존한다. 전체 driver 반환 시간이나 외부 provider 종료 ETA는 제공하지 않는다.
00 독립 PR과 01→02의 리뷰·필수 combined CI를 마쳤다. 실제 Stop→조회→receipt는
[설치 패키지 실행 예제](../examples/termination_budget.py)로 검증했다. I의 최종 누적 검증·병합 결과는 #165에 기록한다.

### 5.4 P5: 스트림 전달 직전의 권한 검사

**재현 완료, 2026-10-25 경계 보완.** `TextDelta("x\ud83d")` → revoke → stall/EOF/provider cancellation의
세 종료를 drain=0/5와 조합한 6개 probe에서 `WriteAuthorityRevoked(cause=lease_lost)`,
texts=`['x', '\ufffd']`, settled=0을 확인했다. 양수 drain 설정도 drain 시작 전의 이 경로를 막지 못한다.
`TextDelta("x")` → revoke → `TextDelta("late")` → stall의 2개 probe에서도 drain=0/5 모두
texts=`['x', 'late']`였다. 직접 runner consumer에서 권한 상실 뒤 새 출력이 전달된다.

수정 소유자는 `model_call.py::_astream`의 정규화된 조각 전달이다. 일반 delta, terminal 정규화가 만든 조각,
`deliver_remainder`를 같은 권한 확인 경계로 모은다. terminal 하나가 여러 조각을 만들 수 있으므로 각 조각의
append/consumer 전달 전에 검사하고 callback 중 revoke도 이후 전달을 차단한다. 기존 authority helper를 재사용하며
임의 consumer callback 동안 authority lock을 보유하지 않는다. 이미 진입한 callback의 외부 효과는 되돌릴 수 없다.

`driving`의 호출 수명 차단, `drain_rest`의 기존 검사, 전달 전 retry-window 표시와 `lease_lost` 우선순위를 유지한다.
정상 Unicode flush와 일반 user cancel의 기존 drain/정산은 보존한다. 새 공개 API·저장 schema·취소 정책은 추가하지 않는다.
직접 runner 회귀와 실제 loop의 다음 턴 격리를 함께 검증한다. observer·sink의 기존 차단도 별도로 확인한다.
상세 반례와 종료 기준은 [P5 PR 명세](V0_25_P5_DX1_PR_PLAN.md#3-pr-01--p5-스트림-전달-권한-검사)에 있다.

### 5.5 P6·P8·P9와 후속 범위

**P6.** 커널 token은 기본 객체 truthiness를 사용하며 조사한 Python 소스에서 token의 `or` fallback 결함을
찾지 못했다. 선택 인자의 부재는 `is None`으로 검사하는 규칙·예제를 정비한다.
구조적으로 외부 신호를 받는 공개 경로가 확인되면 falsy 신호 회귀를 추가한다.
`bool(token)`에 `TypeError`를 도입하는 변경은 영향 조사 후 별도 결정한다.

**P8 조건부.** 현재 일반 child 생성은 parent cancellation token을 공유하고 drain/grace·write authority·
turn interrupt는 그대로 상속하지 않는다. foreground wait와 background 종료 polling도 다르다.
상속과 interrupt 전파는 별도 opt-in으로 설계하고 명시 override → 상속 → 기본값 순서를 고정한다.
현재 활성 자식 generation에만 전파하고 늦게 생성된 자식에 지난 interrupt를 적용하지 않는다.
부모 대기는 동시 자식 창의 최대값과 부모/정책 상한으로 제한한다. 자식 receipt와 parent의
`subagent_usage`/`total_usage`를 이중 합산하지 않는 실패·복구 테스트가 포함 조건이다.
준비가 끝나지 않으면 후속으로 넘기고 P1~P5의 릴리스 마감을 진행한다.

**P9 독립 유지보수.** 현재 service compose와 lock은 `bitnamilegacy/minio` digest를 고정한다.
미러는 출처·접근성·재배포 조건을 확인하고, 대체 서버는 conditional write·multipart·version identity·
GC 경합·corruption·response-loss 의미를 실제 service suite로 검증한다. 대체 제품의 최신 유지 상태와
우열은 이번 조사에서 확인하지 않았으며 선정은 후속 개별 조사로 둔다.
버전 선언은 `pyproject.toml`, `_version.py` fallback, wheel audit 기대값과 CHANGELOG에 걸쳐 있고,
기존 packaging test가 일치를 검사한다. 자동화는 source/build/audit 동기화를 개선하는 독립 작업이다.

**P7 후속.** 기존 shell의 timeout·cwd·env·명령 정책을 유지한다. backend별 CPU·메모리·네트워크·
프로세스 종료 강제 능력과 플랫폼 지원은 후속 조사에서 정한다. 호스트 기능 활성화는 호스트의 배포 결정이다.

### 5.6 DX1·DX2: 도구 작성과 실제 통합 경로

**DX1.** `tools/decorator.py`의 검증된 입력을 필드 접근으로 전달해 중첩 모델·enum·기본값을 보존한다.
입력 단계에서 `model_dump()`를 호출하지 않는다. 최상위 BaseModel 반환은 `model_dump(mode="json")` 후
실제 결과가 dict이면 content, 그 외 portable 값이면 기존 `{"result": value}` content에 둔다.
RootModel과 custom serializer가 dict 이외의 결과를 만드는 경우를 포함한다. 기존 dict·scalar·명시적 ToolResult는 유지한다.
일반 직렬화 실패는 기존 `tool_result_unportable`의 category/retryable을 유지해 전달하고, wrapper 안 제어 예외는 보존한다.
반환 annotation 강제 검증, 임의 객체 트리 변환,
새 artifact 채널·출력 schema는 이번 범위에 추가하지 않는다. typed 입력에 dict를 기대하던 소비자에는 마이그레이션 안내를 제공한다.

**DX2.** DX1의 순수 typed 도구 예제와 P3의
[보고서 예제](../src/monoid_agent_kernel/reference/host_tool_report.py)를 구현했다.
보고서는 prepare/resume으로 새 interpreter에서 원래 승인 작업을 복원하고 자원을 다시 주입한다.
승인 전후 실제 읽기 0→1회, 모델 호출 3→1회, 원본 proposal·중복 승인 무시·누적 사용량을 검증한다.
P1/P2/P4의 사실·복구·Stop 예제도 기존 실행 검사와 설치 CI에 연결했다. 부모 9개와 gateway 자식 2개의
강제 source 경로를 제거했으며 silent Stop·요청 취소와 passive 구독·재연결의 수명을 예제로 보여준다.
typed 도구→요청 스트림→승인 재개의 학습 경로와 검증 기록은 [DX2 PR 계획](V0_25_DX2_PR_PLAN.md)에 있다.
기존 `reference`의 안정성 등급과 호스트의 별도 소비자 마이그레이션 책임을 유지한다. FastAPI/Vercel 어댑터는 후속이다.

## 6. 작업군과 개발 경계

| 작업군 | 대상 | 통합 방식 | 독립 완료 조건 |
|---|---|---|---|
| G1 종료·사용량 | P1/P2 | P1P2-00 독립 PR, 01–04 순차 통합 브랜치 → develop | 타입·생산자·소비자·영속 형식·복구·fixture가 함께 일치 |
| G2 도구 확장점 | P3 | 독립 작업군 → develop | 공개 주입 경로, 권한 상실·child/fork·복구·호환성 검증 |
| G3 종료 예산 | P4 | #165로 develop `973e474` 통합 완료 | monotonic 제어·복구 잔여 시간·호출 snapshot과 실제 RunStream escalation의 일치 |
| G4 출력 권한 수정 | P5 | PR-01, 독립 버그 수정 PR → develop | 일반·terminal·잔여 출력의 권한 검사와 정상 flush 보존 |
| G6 typed 도구 | DX1 | PR-02, 독립 PR → develop | serializer 없는 입력 전달, JSON 결과·기존 envelope, 실제 loop 소비 |
| G7 통합 예제 | DX2 | [DX2-00→01→02→03](V0_25_DX2_PR_PLAN.md#6-pr별-변경-범위와-의존성) 구현·개별 통합 완료; I에서 누적 검증과 develop 통합 | 첫 도구·요청/passive 수명·새 interpreter 승인 재개와 동일 wheel의 current/minimum 수용 검증 |
| G5 자식 정책 | P8 조건부 | 독립 작업군 → develop | opt-in 상속·전파·대기·회계 검증 |
| M 유지보수 | P6/P9 | 항목별 독립 PR → develop | 각각의 결함 또는 운영 개선을 단독 검증 |

P1/P2는 receipt·attempt·종료 분류를 함께 검토해야 한다. 공동 설계는 필수지만 단일 거대 PR은 요구하지 않는다.
독립적으로 유효한 reader·writer·회계 상태를 만들 수 있는 지점에서 PR을 나눈다.
G1의 구체적인 순서는 [P1P2-00 → 01 reader → 02 producer/observer → 03 writer/recovery → 04 소비자 → I](V0_25_P1_P2_PR_PLAN.md#6-pr별-범위의존성우선순위)다.
G1은 통합 브랜치 `codex/v0.25-p1-p2-call-facts`에서 완료하고 #161로 develop에 병합했다.
G3은 [P4-00 → 01 clock/restore → 02 snapshot/RunStream → I](V0_25_P4_PR_PLAN.md#6-pr별-변경-범위와-의존성)로 완료했다.
G7은 기존 코드·실행 기준선·외부 구현 조사에 따라 학습 경로와 상시 검증을 보완했다.
착수 순서는 **P5 → DX1**, 다음 우선군은 P3·P1/P2·P4다. P5와 DX1 사이에는 기술적 선행 의존성이 없다.
P3·P4·P5·DX1은 G1을 기다리는 전역 의존성이 없다. P8도 기존 계약으로 구현할 수 있으며, 새 API를 실제로 소비할 때
그 관계만 선행 조건으로 추가한다.

[PR별 변경 범위·의존성·검증 기준](V0_25_P5_DX1_PR_PLAN.md)에 첫 두 PR의 파일 지도, 제외 범위,
실제 소비자, 수정 전 실패 기준과 후속 작업군의 분할 기준을 기록한다. 공유 `model_call.py` 수정은 재기준화·리뷰 순서를 조정한다.

## 7. 공통 검증과 호환성

다음은 구현 단계의 검증 기준이다. 현재 실행된 결과는 §2와 §10에서 별도로 기록한다.

| 검증 축 | 필요한 증거 |
|---|---|
| 상태·우선순위 | 정상 완료, refusal, provider 실패, user cancel, turn interrupt, deadline, lease loss의 경합 |
| 사용량 | 미보고·명시적 0·양수, 재시도 혼합, drain 회수·미회수, 복구 후 중복 합산 방지 |
| 공개 전달 | exception → receipt/attempt → ledger → loop metrics/budget → observer의 의도된 정보 보존 |
| 영속·복구 | 구형 fixture 읽기, 새 writer round trip, unknown paid dispatch 자동 재시도 차단, settled 재노출 무과금 |
| 권한·수명 | revoke 후 커널 관리 출력·쓰기 차단, call lease 차용 방지, activation 교체, child/fork 격리·authority lineage, restore 재주입 |
| typed 도구 | sync/async 중첩 입력 타입·입력 serializer 0회·BaseModel JSON 결과·직렬화 실패 observation·기존 envelope |
| 통합 개발자 | 순수 도구·요청 스트림·승인/복구의 커널 소유 시나리오와 설치 wheel 예제 |
| 시간 | 내부 monotonic 제어와 snapshot 일치, wall-clock jump, 설정 변경, 동시 호출, cancel grace 상한, callback 정리 |
| 패키지 | root/contracts import, 기존 위치 인자·구조적 protocol 소비자, optional SDK lazy import, wheel fixture 포함 |
| 서비스 | 변경한 durable 경로의 PostgreSQL/ObjectStore/Temporal 실제 검증; 릴리스 combined gate |

대표 기반: `tests/test_model_call_runner.py`, `test_carriage_conformance.py`, `test_agent_loop_durable_recovery.py`,
`test_authority.py`, `test_subagent.py`, `test_subagent_runtime_context.py`, `test_public_surface.py`,
`test_compatibility_ledger.py`, `tests/conformance/`, `tests/service/`.

기존 schema는 서로 다른 reader 정책을 가진다. [COMPATIBILITY.md](COMPATIBILITY.md)의 ledger와
각 parser를 기준으로 필드 추가의 호환성을 판정한다. 특히 durable model-invocation receipt의 closed vocabulary,
model-calls JSON Schema, replay digest/identity, old fixture를 같은 검토 범위에 둔다.
새 enum·nullable 값·항목 생략이 허용되는지는 실제 구형 reader와 round trip으로 입증한다.

## 8. CI와 릴리스 사전조건

현재 L1은 PR push에서 비직렬 unit/contract를 실행한다. L2는 Ready 진입 또는 관련 profile label 변경에서 실행하고,
manual dispatch는 diagnostic gate로 구분한다. Ready 후 수정은 Draft → 수정·검증 → Ready checkpoint로 처리한다.

새 `codex/v0.25-…` 브랜치는 현재 profile resolver에서 label이 없으면 `core`다. 기존 v0.23 PR별 profile 강제는
새 작업군에 적용되지 않는다. durable receipt 의미를 바꾸는 G1은 실제 변경 범위를 조사해 서비스 profile을 지정하고,
최종 develop/main 검증에는 combined를 포함한다.

작업군 통합 브랜치를 사용할 때는 다음을 먼저 처리한다.

1. `.github/workflows/ci.yml`의 push 대상에 해당 통합 브랜치를 등록한다. P3-00은 `codex/v0.25-p3-host-resources`를 main/develop/v0.23 대상에 추가했다.
2. `tools/v023_ci.py`의 profile 판정에서 해당 작업군에 필요한 검증을 강제할 방식을 지정한다.
3. 기존 service lock/qualification manifest의 의미를 유지하고, 새 계약을 입증할 테스트와 증거를 추가한다.

2026-09-28 GitHub 조회에서는 ruleset `[]`, main/develop protection은 `Branch not protected`였다.
서버측 강제 상태는 2026-10-25에 재조회하지 않았다. 구현 착수 시 확인하고 설정 변경은 별도 운영 결정으로 기록한다.

릴리스는 필수 작업군의 완료, 조건부 P8의 포함 결정, 최종 전체 CI·compatibility·wheel/install 검증,
release audit, develop → main PR, annotated tag, GitHub Release, PyPI 게시 확인 순서다.
[publish.yml](../.github/workflows/publish.yml)은 GitHub Release `published`를 trigger로 삼고,
tag/version 검사·exact wheel audit·twine check 후 같은 artifact를 게시한다.

## 9. 결정 기록

| ID | 결정 대상 | 상태 / 권고 |
|---|---|---|
| D01 | P3 포함, P7 후속 | 사용자 결정 |
| D02 | 독립 작업군 → develop → 적정 시점 릴리스 | 사용자 방향 반영 |
| D03 | P1/P2 타입·저장 형식 | P1·P2 PR 계획: 기존 숫자+희소 증거, invocation/model-calls v2·dual-reader, 원본 hash 보존 복구, reader 유지 rollback; unknown 증거 slot 후속. 구현 승인 |
| D04 | P3 주입·권한·수명 | P3-02/03 구현: 선택적 확장 Protocol, call별 lease, fresh child adapter와 parent lineage, restore 재주입 |
| D05 | P4 clock·집계·조회 | 01·02 구현: monotonic 제어·기존 remaining 복구, phase별 grace 창, 불변 owner snapshot, 최초 exit cutoff 고정. 최종 통합 #165 |
| D06 | P5 | 10/25 재현에 따라 일반 delta·terminal 정규화·잔여 flush의 전달 경계로 보완; PR-01 |
| D07 | P8 | 조건부. 준비된 완결 작업을 포함하고 필수 범위 출시를 대기시키지 않음 |
| D08 | P6 bool TypeError | 현재 규칙 정비와 분리해 영향 조사 후 결정 |
| D09 | P9 S3 교체·버전 자동화 | 독립 유지보수; 완료된 결과만 포함, 필수 계약의 전제조건으로 두지 않음 |
| D10 | branch protection·새 branch CI | 현재 상태 확인; 설정 변경과 CI 변경은 구현 전 별도 운영/작업 결정 |
| D11 | DX1·DX2 포함 | 사용자 통합 요청 반영. P1–P5와 함께 v0.25 필수; 기존 기능 진입점과 예제 개선 |
| D12 | 최초 PR 순서 | P5 우선, DX1 별도 PR. 10/26 사용자 구현 요청에 따라 최신 develop에서 실행 착수 |
| D13 | 외부 사례 적용 | 전달 직전 재검사, 입력 검증/출력 직렬화 분리, 관측 불능 usage 유지; 근거·채택 범위는 PR 계획 §2 |
| D14 | DX2 경험·예제·검증 | [DX2 PR 계획](V0_25_DX2_PR_PLAN.md): 기존 예제 정리, 설치 provenance, 요청/passive 수명, 새 interpreter 승인 재개. 10/28 00–03·I 순차 구현 승인 |

이 문서는 통합 범위와 작업 경계를 기록한다. 구현 PR 사전조사에서는 최신 develop을 다시 읽고
변경 파일·상태 전이·실패 우선순위·호환성·검증 명령·종료 조건을 구체화한다.

## 10. 실제 확인 결과와 남은 검증

아래 명령·73개 baseline·단일 P5 probe는 2026-09-28 조사 기록이다. 2026-10-25의 확장 재현 8개,
관련 기존 테스트 76개, 외부 구현 비교와 통합 설계 리뷰는 [PR 계획 §7·§8](V0_25_P5_DX1_PR_PLAN.md)에 기록한다.

- Python/import 위치 확인, lock 검증, profile probe는 완료했다.
- 기존 공개 표면·호환성·패키지·CI·문자 경계 focused baseline은 73 passed / 8.67s / exit 0이었다.
- P5의 drain OFF 문제는 독립 진단으로 재현했다. 수정은 이번 조사 범위에 포함하지 않았다.
- 전체 테스트, 실제 서비스, 새 schema/codec, 새 도구 API, 새로운 cancellation 정책은 구현 단계 검증 대상이다.
- 호스트 운영 서버·실청구 성과·대체 S3 서버의 현재 유지 상태는 이번 조사에서 검증하지 않았다.

재현 가능한 baseline 명령은 다음과 같다. Python은 저장소 `.venv`의 3.11.9이고 xdist worker 상한은 4였다.

```powershell
$env:PYTEST_XDIST_AUTO_NUM_WORKERS = '4'
& ./.venv/Scripts/python.exe -m pytest -q -n auto --dist=worksteal --timeout=120 --timeout-method=thread -p no:randomly tests/test_public_surface.py tests/test_compatibility_ledger.py tests/test_v023_ci_foundation.py tests/test_release_packaging.py tests/conformance/test_import_boundaries.py tests/test_model_call_runner.py::test_a_lease_lost_in_the_tick_a_user_cancel_began_the_drain_settles_nothing tests/test_model_call_runner.py::test_stream_preserves_surrogate_pairs_across_interleaved_logical_channels tests/test_model_call_runner.py::test_stream_flushes_lone_surrogates_in_original_global_order
```

P5의 조사용 재현은 기존 private helper를 사용했다. 구현 PR에서는 독립적인 regression test로 고정한다.

```powershell
& ./.venv/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"tests"); n=runpy.run_path("tests/test_model_call_runner.py"); a=n["ActivationWriteAuthority"](); s=n["_DrainStream"](n["TextDelta"]("x\ud83d"), a.revoke, n["_STALL"]); o=n["_drain_call"](s, drain_s=0.0, authority=a, timeout_s=5.0); print("error=", type(o.error).__name__, "cause=", getattr(o.error,"interruption_cause",None)); print("texts=", [ascii(x) for x in o.texts], "settled=",len(o.settled),"yielded=",len(s.yielded),"closed=",s.closed)'
```

```text
error= WriteAuthorityRevoked cause= lease_lost
texts= ["'x'", "'\ufffd'"] settled= 0 yielded= 1 closed= True
```
