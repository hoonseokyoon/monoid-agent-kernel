# v0.25 P3: 호스트 자원 주입·권한·수명·복구 PR 계획

작성·조사일: 2026-10-27 KST. 기준: 최신 `origin/develop`
`5b08b10d82a948268d1a95b51112b1508e233e3d` (P5 #148, DX1 #149 병합).
상태: **2026-10-27 P3-00/01 develop 병합, P3-02 통합 branch 병합, P3-03 사전조사.** 사용자는 P3-00·01부터 진행하고,
P3-01은 실패 회귀를 먼저 고정한 독립 PR로 구현한 뒤 리뷰·CI를 확인하도록 요청했다.
이후 P3-02·03을 계획된 통합 브랜치에서 순차 진행한다. 릴리스·배포·CSP 변경은 범위 밖이다.

범위의 기준은 [v0.25 통합 계획](V0_25_PRE_INVESTIGATION.md#52-p3-타입-있는-주입과-호출별-권한)이다.
[개발자 API 설계](V0_25_DEVELOPER_EXPERIENCE_PROPOSAL.ko.md#4-v025-api-예시-타입-있는-도구와-자원-주입)의
P3 수명 계약을 아래 정책으로 구체화한다. P5·DX1 실행 기록은 [기존 PR 계획](V0_25_P5_DX1_PR_PLAN.md)에 둔다.

## 1. 전달할 사용자 결과와 책임

제품 통합 개발자는 typed key로 호출에 허용된 workspace와 호스트 자원을 얻는다.
보고서 도구 하나가 중첩 Pydantic 입력을 받아 허용 경로에 proposal을 만들고, 승인 대기 뒤 새 activation으로 복구되어
저장된 내용을 읽는 경로까지 공개 API와 설치 wheel로 재현한다. root·child·fork의 권한과 정리 책임을 설명할 수 있어야 한다.

| 역할 | 현재 담당·책임 |
|---|---|
| 진행·설계 작성 | root: 범위·정책·소스 대조·probe·문서·검증 runner 소유 |
| 외부 조사 | p3_external_research: 공식 문서, 채택된 수정과 미병합 제안 구분 |
| 수명 조사 | p3_lifecycle_audit: bootstrap/restore/close/park/child의 실제 순서 감사 |
| 독립 계획 검토 | 작성·조사에 참여하지 않은 fresh reviewer; §8에 대상과 판정 기록 |
| 구현 | root가 P3-00·01을 소유하고 runner·공유 파일 편집을 직렬화. 후속 PR도 착수 시 담당 기록 |

상위 저장소의 현재 운영 권위인 `docs/operations/campaign-workflow.md`와
`docs/operations/campaign-preflight-review.md`를 따른다. 권한·수명 변경은 A 등급이다.
각 구현 PR은 최신 실제 base에서 사전조사를 갱신하고 독립 사전 검토 및 서로 다른 두 코드 reviewer를 거친다.
계획 검토는 향후 코드 검토를 대체하지 않는다. 공유 파일 편집과 무거운 검사는 root가 직렬화한다.

## 2. 코드와 실제 소비자에서 확인한 간격

| 생산자 → 소비자 | 현재 사실·위치 | P3에서 정할 경계 |
|---|---|---|
| handler dispatch → extension context | [loop.py](../src/monoid_agent_kernel/loop.py) `_ainvoke_handler`는 provider와 handler에 같은 façade를 전달. `_live_call`은 ContextVar가 없는 thread에서 최신 `_call_fallback`을 사용 | 호출마다 새 façade에 원래 lease를 고정. 새 helper가 현재 전역 call을 조회하지 않음 |
| tool admission → workspace | `_aexecute_tool_call`이 surface·승인·capability를 검사. `_check_permissions`는 선언한 `path_args`만 확인. [AuthorityBoundWorkspace](../src/monoid_agent_kernel/core/workspace.py)는 activation authority만 확인 | handler가 계산한 경로에도 작업마다 call의 path scope·PermissionPolicy·workspace mode 적용 |
| workspace factory → CSP provider | 상위 CSP `pipeline_core/agents/modular_kernel.py`의 `_binding_workspace_factory`, `PaperWorkspaceToolProvider.bind_kernel_workspace`, `_tool_workspace`가 원본 backend를 보관해 직접 write | kernel 소유 보고서 fixture로 이 사용법을 대체할 seam 검증. CSP 배포·pin 변경은 별도 |
| bootstrap → partial cleanup | [loop_phases.py](../src/monoid_agent_kernel/loop_phases.py): recorder 생성 118, observer factory 136, partial resources 게시 228 순서 | 첫 owned handle부터 cleanup 책임을 확보 |
| restore → handler | `restore`의 bootstrap 3753 → `_rehydrate` 3756 → workspace checkpoint 적용 3877 순서 | factory는 descriptor와 deferred resource만 생성. 복구가 끝난 실제 handler에서 내용 읽기 |
| lifecycle → observer cleanup | `discard_uncommitted`의 subscription close 2733은 오류 수집 밖. `close`에도 subscription close가 마지막 teardown의 try/finally 앞에 있음 | cleanup 오류 뒤에도 activation 정리 완료; 원래 제어 예외 보존 |
| child/fork → inherited providers | `_run_subagent_child`는 provider·workspace factory를 전달하지만 child authority는 새 객체. parent authority 검사는 child 반환 뒤 | 자원 상속은 명시적 선택, fresh factory와 parent lineage 검사 동반 |
| hosted driver → restore/release | [activation.py](../src/monoid_agent_kernel/hosting/activation.py)는 checkpoint/command 검증 후 loop를 만들고, 중복 receipt면 loop 생성도 생략. committed boundary 뒤 release | 유효한 실제 activation마다 자원 생성·해제. 중복 명령은 factory 0회 |

위 CSP 경로는 통합 요구의 소스 근거다. 운영에서 stale write가 발생했다는 조사 결과는 없다.

### 2.1 기존 정리 결함의 제한된 재현

동일 base, Python 3.11.9에서 소스·테스트를 수정하지 않고 두 callback 순서만 주입했다.
원본 script/output은 로컬 ignored `.tmp/v025/p3_preflight_probe.py`,
`.tmp/v025/p3-preflight-probes.json`에 있다. private 필드 관찰은 정리 여부 확인을 위한 probe에 한정했다.

| 조건 | 관측 결과 | 해석·선행 조치 |
|---|---|---|
| 생성된 recorder를 계측한 뒤 stream-observer factory에서 authority revoke | `WriteAuthorityRevoked`; recorder 1개, transcript open, partial resources 없음. public discard 후에도 transcript open | recorder ownership 게시 전 실패 경로를 P3-01에서 수정. GC 시점의 해제를 정상 수명 보장으로 삼지 않음 |
| model-I/O observer.close에서 authority revoke 후 public discard | `WriteAuthorityRevoked`; observer close 1회, session·authority callback 남음, finalized=false. 두 번째 discard에서 session 해제 | 첫 cleanup 시도에서 teardown을 끝내도록 P3-01에서 수정 |

기존 관련 회귀 **13 passed in 4.21s**. 이 성공은 위 두 순서가 이미 검증됐다는 뜻으로 사용하지 않는다.
새 host resource API는 아직 없으므로 위 결과는 기존 recorder/subscription 수명 결함의 증거다.

## 3. 외부에서 확인한 문제와 채택할 답안

확인일은 2026-10-27이다. 링크의 pinned source와 PR 병합 상태를 확인했다.
아래 적용은 외부 구현과 우리 코드의 차이를 검토한 설계 결정이다.

| 문제·근거 | 채택된 해법 또는 표준 계약 | Monoid 적용 |
|---|---|---|
| PydanticAI의 실행 후 내부 순환 참조가 deps를 GC까지 유지. [#11158](https://github.com/pydantic/pydantic-ai/pull/11158), 2026-10-24 merged | [ToolManager 구현](https://github.com/pydantic/pydantic-ai/blob/13e11da85da7770f0e2a2576bd7551b984b1fe34/pydantic_ai_slim/pydantic_ai/tool_manager.py#L480)은 내부 context backlink를 해제. 앱이 보관한 context는 유지 | typed 주입, 참조 해제, 사용 권한 회수를 각각 검증. retained context/view는 call lease로 차단 |
| [PydanticAI deps](https://pydantic.dev/docs/ai/core-concepts/dependencies/)는 host가 연 client를 전달하며 deps_type은 정적 타입 힌트 | [RunContext.deps](https://github.com/pydantic/pydantic-ai/blob/13e11da85da7770f0e2a2576bd7551b984b1fe34/pydantic_ai_slim/pydantic_ai/_run_context.py#L213)는 호스트 객체 필드 | identity key + runtime accepts + adapter ownership. 일반 객체를 전달하는 것만으로 revoke 보장을 주장하지 않음 |
| durable state와 live DB/client의 수명 혼동 | [LangGraph runtime context](https://docs.langchain.com/oss/python/langgraph/graph-api#runtime-context)를 state와 분리. [checkpoint metadata](https://github.com/langchain-ai/langgraph/blob/1526b0e82df6e2f9879eb0fb005d04d3fe469ebf/libs/checkpoint/langgraph/checkpoint/base/__init__.py#L786)는 내부 runtime 제외 | checkpoint에는 복구 데이터, activation에는 fresh host injection. 복구 전 workspace 내용 snapshot 금지 |
| LangGraph 0.6.0 subgraph에서 runtime context 전달 누락: [#5700](https://github.com/langchain-ai/langgraph/issues/5700) | [#5707](https://github.com/langchain-ai/langgraph/pull/5707), 2025-07-29 merged, v0.6.1 반영. [Runtime.merge](https://github.com/langchain-ai/langgraph/blob/1526b0e82df6e2f9879eb0fb005d04d3fe469ebf/libs/langgraph/langgraph/runtime.py#L240)는 참조 선택·전달 | root/child/fork 진입점 모두 검증. Monoid는 fresh adapter + parent lineage를 추가하여 별도 수명 보장 |
| FastAPI에서 yield dependency를 함수 종료에 닫아 streaming 중 DB session 사용 실패 | [공식 변경 설명](https://fastapi.tiangolo.com/advanced/advanced-dependencies/): 0.118.0 응답 후 cleanup 복구, 0.121.0 function/request scope. [#14262](https://github.com/fastapi/fastapi/pull/14262), [#14301](https://github.com/fastapi/fastapi/pull/14301) merged | call-view와 activation-resource 수명 분리. provider·편의 constructor·restore 모두 같은 수명 정책 전달 |
| 부분 초기화 실패·여러 closer의 역순 해제 | 표준 [ExitStack](https://docs.python.org/3.11/library/contextlib.html#contextlib.ExitStack), [CPython 등록·해제 구현](https://github.com/python/cpython/blob/ffbb750015b2ae168ef5a06daed257885a3269c5/Lib/contextlib.py#L636) | 반환 직후 closer 등록, pop/closed 표시 뒤 호출. 기존 1차 오류를 보존하는 작은 wrapper만 추가 |
| 취소 중 cleanup await가 재취소되거나 sync worker가 계속 실행 | [Python cancellation](https://docs.python.org/3.11/library/asyncio-task.html#task-cancellation), [AnyIO shielding](https://anyio.readthedocs.io/en/stable/cancellation.html), [PydanticAI sync worker 설명](https://github.com/pydantic/pydantic-ai/blob/13e11da85da7770f0e2a2576bd7551b984b1fe34/pydantic_ai_slim/pydantic_ai/_run_context.py#L1008) | 기존 shielded lifecycle bridge 재사용. 먼저 view 무효화, 이후 local cleanup. 이미 시작한 외부 효과는 host fencing/idempotency 책임 |

LangGraph #5705/#6381은 closed·unmerged, PydanticAI #10479/#11255는 open·unmerged다.
해결 방향의 참고 자료이며 채택된 해결책의 증거에서 제외했다.
외부 runtime 분리가 사용자 작성 state·custom saver까지 보호한다고 일반화하지 않는다.

## 4. 채택 정책

### 4.1 등록과 소유권

- 신규 모듈은 `monoid_agent_kernel.tools.host_bindings`. 공개 이름은 `HostToolKey[T]`,
  `HostToolBinding[T]`, `HostToolResource[T]`, `HostToolCallLease`, `ToolWorkspace`,
  `require_host_tool_binding`으로 계획한다. 기존 `ToolContext` 필수 멤버를 늘리지 않는다.
  helper가 사용하는 선택적 확장 Protocol은 새 모듈에 둔다.
- `AgentLoop(..., host_tool_bindings=...)`는 keyword-only다. `from_tools`/`from_config`는 같은 인자를 전달한다.
  constructor에서 tuple snapshot과 등록 형태·identity/name 중복을 검사한다. 이 단계 factory 호출은 0회다.
- key 비교는 프로세스 내 identity다. 이름은 진단용이며 문자열 lookup·직렬화·복구 식별자로 쓰지 않는다.
  host와 handler는 같은 모듈의 같은 key 인스턴스를 공유한다. 다른 프로세스는 그 프로세스에서 새로 구성한다.
- bootstrap에서 등록 순서대로 동기 `open(activation)`을 1회 실행한다.
  descriptor는 run_id, root_run_id, parent_run_id, mode와 `workspace_resource()`만 제공한다.
  operational workspace, 원본 backend, 절대 경로, recorder와 registry lookup을 factory에 노출하지 않는다.
- 반환 resource는 `bind_call(lease) -> T`, `close() -> None`을 갖는 activation adapter다.
  kernel이 adapter를 소유한다. 앱의 공유 connection pool/client를 빌리는 adapter는 pool의 close 책임을 host에 남긴다.
  kernel-owned adapter의 close는 **local dispose**이며 외부 commit/flush를 수행하지 않는다.
- 반환 직후 cleanup 등록을 **guarded factory callback 안에서** 끝내고 외부 guard postcheck로 나간다.
  이후 구조 검사·다음 factory·provider 등록이 실패해도 회수한다. 반환 전 실패는 factory가 내부 획득분을 정리한다.
  반환 객체에 유효한 closer조차 없으면 factory 계약 위반으로 실패하며 숨겨진 내부 자원 정리를 보장하지 않는다.
- 서로 다른 key의 factory가 같은 adapter 인스턴스를 반환하면 설정 오류로 거부한다.
  activation 안에서 identity를 확인해 두 번째 closer를 등록하지 않고, 먼저 소유한 adapter를 1회 정리한다.
  공유 pool은 key·activation마다 별도 adapter가 빌린다. 일반 host adapter의 activation 간 fresh 반환은 factory 계약이며,
  kernel이 모든 프로세스의 임의 객체 identity를 추적하는 전역 registry를 추가하지 않는다.
- 실제 호출 안 첫 lookup에서 `bind_call`과 `key.accepts(view)`를 수행하고 call별·key별 view를 캐시한다.
  검사와 발급 전후에 guard를 확인한다. call 종료 때 cache 참조를 해제한다. 잘못된 타입을 generic 성공값으로 대체하지 않는다.

host factories는 context 생성 후, 기존 `_bootstrap_resources` 게시 전에 실행한다.
그 이전에는 표준 cleanup scope가 recorder/job/host resource를 소유한다. 조기 cleanup 소유권 확보가
recordable session의 게시 시점을 앞당기지 않도록 한다. host factory 실패는 전체 획득분을 정리한 뒤 안전한 오류로 전파한다.
기존 provider/runtime-config 실패가 recordable failed session을 만드는 동작은 유지한다.
이 경우 P3 자원은 즉시 종료하고, 실패 결과를 마무리할 recorder는 기존 close까지 해당 session이 소유한다.

**동기 lifecycle 선택.** v0.25의 open/bind_call/close는 동기 계약으로 한정한다. async factory/closer 또는 awaitable 반환은
고정 진단으로 거부한다. awaitable 정리는 기존 unawaited-awaitable 처리 패턴을 사용한다.
`aopen/aclose/arelease_parked`는 기존 shielded sync lifecycle bridge를 사용한다. host가 callback의 시간 상한을 책임진다.
kernel이 정리 지연의 유한 상한을 보장하거나 sync thread를 강제 종료한다고 약속하지 않는다.
async client를 쓰는 도구는 host-owned client의 adapter를 주입하고 async 메서드에서 lease를 await 전후 확인한다.

### 4.2 호출 권한과 workspace

권한의 생산자는 기존 `_aexecute_tool_call`의 승인된 dispatch다. provider 등록에는 view 발급 권한이 없다.
`_ainvoke_handler`에서 원래 call identity·허용된 ToolAuthorization·ToolScope·side_effect를 고정한 lease와 façade를 만든다.
생성과 진입 검사는 정리를 보장하는 try/finally 안에 둔다. 정상 반환, 예외, 취소, timeout, abandon에서 모두 lease를 끝낸다.
sync handler가 awaitable을 반환하면 그 awaitable의 처리까지 같은 호출 수명에 포함한다.

`require_host_tool_binding(ctx, key)`는 전달받은 façade의 lease만 사용한다.
ContextVar·`_call_fallback`·thread ancestry로 원래 호출을 재발견하지 않는다.
call 안에서 명시적으로 전달받은 ctx를 쓰고 종료 전에 join하는 thread는 허용한다.
보관된 ctx/view와 abandon된 worker·descendant thread는 다음 call B가 active여도 call A의 권한만 가지며 A 종료 뒤 거부한다.

`HostToolCallLease`는 읽기 전용 call/binding identity와 `assert_active()`,
외부 callback 전후를 검사하는 `guard_external_call(callable)`만 제공한다. revoke/renew/권한 교체 API는 노출하지 않는다.
이 helper는 이미 발생한 외부 효과를 되돌리지 않는다. 일반 adapter의 메서드별 검사 책임은 host에 있다.

`ToolWorkspace`는 runtime-checkable Protocol이며 우선 `read_bytes`, `write_bytes`, `exists`만 제공한다.
read는 기존 `(bytes, sha256)`, write는 sha256을 반환하고 `max_bytes`, `create_dirs`,
`expected_sha256`, `overwrite` 의미를 기존 backend와 일치시킨다.

각 작업은 원래 call lease → P3 activation/parent lineage → writer authority → 경로 정규화 →
immutable binding scope와 PermissionPolicy → 실제 backend 작업 → 반환 전 guard 확인을 거친다.
read/exists도 scope 밖 경로의 존재·내용을 공개하지 않는다. write는 `side_effect="write"`로 승인된 도구만 허용한다.
`read/artifact/run/shell` 선언으로 이 새 view의 write 권한을 얻지 않는다.
기존 capability gate는 dispatch가 검사하며 capability 문자열에서 새 권한 체계를 추론하지 않는다.
backend의 read-only/propose/apply 모드, 경로 traversal·symlink 정책, size 제한, expected hash 충돌 처리를 재사용한다.
scope는 [기존 CLI 계약](CLI.md#path-permissions)의 정규화된 lexical path 기준이다.
symlink·hardlink·대소문자 alias의 객체 동일성까지 판별하는 새 파일 권한 체계는 포함하지 않는다.

local workspace mutation은 call 종료·activation 종료·writer revoke와 선형화한다.
진입한 local mutation은 완료될 수 있고, 권한 종료를 관측한 뒤 성공 결과를 발행하지 않는다.
종료가 반환한 뒤 새 mutation은 시작하지 못한다.
권한 lock 안에서 arbitrary factory/bind_call/accepts/closer/network callback을 호출하지 않는다.
구현 preflight에서 guard 획득 순서와 parent revoke callback의 lock 밖 실행을 명시하고 barrier 회귀로 교착·late write를 확인한다.

기존 `ToolContext` 메서드 전부의 `_call_fallback` 계약 변경은 별도 carry다.
이번 보장은 신규 helper와 kernel workspace view를 대상으로 한다.
Python private 필드 접근을 강제로 막는 sandbox는 P7의 영역이다.

### 4.3 activation 종료·오류·복구

P3 activation guard는 host의 `ActivationWriteAuthority`와 별도 수명을 갖는다.
정상 close/park release에서 P3 guard만 끝내며, 호스트가 공유하는 writer authority를 revoke하지 않는다.
실제 writer revoke는 즉시 새 resource 접근을 막고 기존 `WriteAuthorityRevoked/lease_lost` 분류를 유지한다.

| 전환 | view·resource 처리 | durable 상태 |
|---|---|---|
| handler 종료 | call lease 종료, call cache 해제; activation adapter 유지 | 기존 tool result 경로 유지 |
| 정상 close / 실패 discard | 새 접근 차단 후 adapter 역순 dispose, callback·loop 참조 해제 | 현재 finalization·checkpoint 삭제/보존 정책 유지 |
| committed park release | 유효한 committed fingerprint 확인 후 guard 종료·dispose | checkpoint·hosted task 유지, 새 activation에서 재개 |
| uncommitted park release 거부 | guard/resource 유지 | 기존 `run_not_durably_parked` |
| writer revoke | view 즉시 거부; 종료 경로에서 local dispose | external flush/commit은 fenced host에 귀속 |
| 전파되는 bootstrap/restore 실패 | 그 시점까지 등록된 자원을 역순 정리 | source checkpoint와 이미 보존 표시된 hosted task 유지 |
| 기존 recordable bootstrap 실패 | P3 자원 종료, 실패 session의 recorder는 기존 close까지 유지 | 기존 실패 결과·finalization 계약 유지 |
| async lifecycle 취소 | 진행 중 sync lifecycle을 기다린 후 discard, 취소 재전파 | 중복 closer·세션 부활 방지 |

cleanup 상태는 callback 실행 전에 closed로 바꾸고 표준 ExitStack의 pop-before-call 방식으로 한 번만 시도한다.
한 closer의 실패 뒤 남은 closer와 recorder/job/subscription teardown을 계속한다.
cleanup 전에 이미 진행 중인 lease loss·cancel·timeout·원래 실패를 보존한다.
기존 authority fence에서 관측한 lease loss는 현재 우선순위를 유지한다.
정상 종료 중 정리만 실패하면 첫 cleanup 오류를 전파하고 activation은 종료한다.
추가 실패는 자원 값·repr·secret이 없는 제한된 진단으로 남긴다. 종료 시 registry의 adapter 참조도 해제한다.
애플리케이션이 직접 보관한 객체의 GC 완료 시점은 보장하지 않는다.
**closer 최대 1회 호출은 외부 해제 효과의 성공을 뜻하지 않는다.**

restore의 bootstrap-before-rehydrate 순서를 유지한다. factory의 workspace_resource는 deferred adapter를 만들고,
provider 등록에는 operational view를 주지 않는다. checkpoint 적용 완료 후 첫 handler에서 복구된 내용을 읽는다.
잘못된 주입은 bootstrap 실패 또는 첫 lookup의 명시적 오류로 드러난다.
필요 key 목록을 ToolSpec/checkpoint에 추가해 자동 복원하는 범위는 넣지 않는다.
key·adapter·client·secret을 kernel의 model request, event payload, manifest, checkpoint에 기록하지 않는다.
새 durable/wire codec 버전은 필요하지 않다.

설정 오류는 AgentConfigError, 호출 밖/종료된 lease는 `tool_call_not_in_flight`,
누락 key는 `host_tool_binding_missing`, 확장 미지원 ctx는 `host_tool_context_unavailable`,
view 타입 오류는 `host_tool_binding_type`, 닫힌 activation은 `host_tool_resource_closed`로 계획한다.
경로·작업 거부는 기존 PermissionDenied 계열을 사용한다.
open/bind_call/accepts의 일반 예외는 새 경계에서 단계별 고정 메시지의 안전한 공개 오류로 바꾼다.
기존 `_record_failure`/`_failure_result`가 `str(exc)`를 기록하므로 원본 callback 오류 문자열도 전달하지 않는다.
새 event·artifact·로그 진단에 원본 문자열·repr·traceback을 기록하지 않는다. 직접 발생한 제어 예외는 기존 분류를 유지한다.
구현자는 기존 제어 예외 분류를 유지하고 공개 오류 코드 테스트에서 최종 이름을 고정한다.

### 4.4 child·fork

P3-02의 기본값은 상속 OFF이고, 미완성 상속 옵션을 노출하지 않는다.
P3-03에서 `HostToolBinding(..., inherit_to_children=True)`를 추가한다. 각 hop에서 opt-in 등록만 전달하며 child와 fork에 동일 적용한다.
자식은 새 registry·adapter·call view를 갖고 factory를 자식 descriptor로 다시 실행한다.
부모 resource/view 또는 WriterToken을 전달하지 않는다. factory closure의 host-owned pool 공유는 host의 명시적 책임이다.

상속한 view는 child-local guard와 모든 parent activation/authority lineage를 검사한다.
parent revoke 또는 parent activation 종료 후 자식의 새 접근을 거부한다.
child 종료는 parent/sibling adapter를 닫지 않는다. lineage는 수명 token 검사로 연결하며 새 callback을 등록하지 않는다.
복구된 parent가 새 child를 만들면 새 lineage에 연결한다. 끝난 parent의 child를 자동 재연결하지 않는다.

이 보장은 신규 상속 자원의 사용 권한이다. 자식 모델 호출 강제 취소·drain/usage 합산은 P8에서 다룬다.
기존 child builtin 전체의 parent writer 연동 정책도 P8 사전조사에서 별도 판단한다.

### 4.5 대안과 비용 판단

| 선택지 | 총비용·한계 | 결정 |
|---|---|---|
| deps dict 또는 원본 workspace 직접 전달 | 초기 구현은 작지만 retained context·call 권한·restore freshness를 표현하지 못함 | 채택하지 않음 |
| 기존 authority wrapper만 재사용 | activation fence는 충족; call 종료·binding scope는 누락 | backend fence는 재사용하고 call façade만 추가 |
| ContextVar/fallback 기반 resource lookup | 기존 구조와 가깝지만 수동 thread와 다음 call 권한 차용을 해결하지 못함 | invocation 생산 지점에서 lease 고정 |
| 모든 bootstrap을 rehydrate 뒤로 이동 | provider·index·task 복구 순서와 실패 경로 변경 폭 큼 | deferred workspace adapter로 현재 순서 유지 |
| 범용 DI container·async owned resource graph | 의존성, cycle, async shutdown, lifetime 설정 API가 늘어남 | 동기 flat 등록 + 표준 ExitStack + host adapter |
| registry-only 기반 PR 후 소비자 연결 | 임시 API와 미검증 종료 경로가 생김 | root 보고서·restore까지 한 vertical PR에 포함 |

## 5. PR별 변경 범위·의존성·완료 기준

통합 브랜치: `codex/v0.25-p3-host-resources`. P3-00/01 병합 후 생성했다.
계획 보존 브랜치는 `codex/v0.25-p3-plan`이다. 현재 실행 브랜치·증거는 §9에 기록한다.
`pr-fast.yml`/`pr-full.yml`은 base 필터가 없지만 `ci.yml` push는 main/develop/v0.23만 등록한다.
운영 규칙에 따라 P3-00에서 새 통합 base를 먼저 등록한다.
P3-01은 기존 결함을 고치는 독립 PR로 develop에 병합하고, 두 준비 PR 후 최신 develop에서 통합 브랜치를 만든다.
P3-02 → P3-03을 통합 브랜치에 순차 merge commit으로 쌓고 P3-I에서 develop에 통합한다.

| PR | 변경 경계·파일 | 선행·완료 신호 | 크기·검토·CI |
|---|---|---|---|
| **P3-00 CI base 등록** | `.github/workflows/ci.yml` push 대상에 정확한 P3 통합 브랜치 추가. workflow 동작·service profile 확대 없음 | 독립 → develop. 통합 branch push CI 및 PR L1/L2 대상 확인 | 기능 10줄 안팎, 총 30–80줄. 검증 신뢰 경계 B; 독립 검토, YAML/기존 CI 검사 |
| **P3-01 기존 cleanup 보완** | `loop_phases.py` recorder 획득 즉시 소유권 확보, `loop.py` open/restore/close/discard cleanup finally. 필요 시 기존 recorder의 local-discard 재사용. 기존 이벤트 의미 유지 | 독립 → develop. §2.1 두 실패를 수정 전 회귀로 고정하고 첫 cleanup에서 handle·session·callback 해제 | 기능 100–250줄, 총 300–600줄. A, L0 lifecycle + L1/L2 core |
| **P3-02 root 자원·권한·복구** | 새 `tools/host_bindings.py`, `loop.py` keyword seam/호출 façade/cleanup 연결, `loop_phases.py` bootstrap ownership, 필요 최소 내부 call metadata. public export/API 검사, root report 예제·EMBEDDING/TOOL_SURFACE/CONTRACTS/COMPATIBILITY | P3-00/01. from_tools·from_config → typed 도구 → scoped proposal write → park/release → fresh restore/read가 설치 wheel에서 완결. child/fork 기본 OFF | 기능 600–1,000줄, 총 1,300–2,100줄. A, L0 권한·복구·실제 loop + L1/L2 core |
| **P3-03 child/fork 상속** | `loop.py` child 생성과 registry/lineage 연결, binding opt-in 필드, 관련 subagent/authority tests, child 소비 예제·계약 | P3-02. opt-in fresh factory, default denial, 부모 종료·revoke 뒤 자식 접근 차단, 자식 정리가 부모에 영향 없음 | 기능 150–300줄, 총 450–800줄. A, L0 실제 child/fork + L1/L2 core |
| **P3-I 최종 통합** | P3-00/01 이후 최신 develop 충돌 해소, 누적 계약·예제·실행 증거 정합성. 새 기능 추가 시 해당 PR로 돌려 독립 검토 갱신 | P3-02/03. 최종 merge candidate의 review/CI, 설치 wheel 승인·재시작 예제, compatibility 확인 | 기능 증분 0 예상. A 누적 검토·최종 full develop CI. 충돌·공유 소스 변경은 영향 범위 재검증 |

예상 총비용은 기능 약 860–1,560줄, 테스트·문서·CI 포함 약 2,100–3,700줄이다.
분할해도 누적 비용을 유지한다. 기능 2,000줄 soft limit과 총비용 1.5배 초과·반복 재작업을 재계획 신호로 삼는다.
P3-02의 root 계약을 임시 registry PR로 쪼개지 않는다. 예상 초과 시 실제 변경 지도를 보고 root 결과가 완결되는 대안을 다시 검토한다.

P1/P2/P4는 기술적 선행이 아니다. DX1은 현재 base에 있으며 보고서 예제가 이를 소비한다.
P5 authority 검사는 유지한다. P8 도입 여부와 무관하게 P3 lineage 검증을 완료한다.
CSP migration·배포·릴리스, 범용 DB/HTTP adapter, delete/list/glob 등 workspace API 확대는 이번 PR군 범위 밖이다.

## 6. 실제 통합 개발자 API 예시

아래는 **P3 구현 목표 API**다. 현재 배포 패키지에서 실행되는 예제로 표시하지 않는다.
P3-02에서 전체 example을 패키지 smoke에 포함하고 실제 import·타입·승인 경로를 확정한다.

~~~python
from pathlib import Path
from pydantic import BaseModel
from monoid_agent_kernel import AgentLoop, AgentRunSpec, ToolContext, tool
from monoid_agent_kernel.tools.host_bindings import (
    HostToolBinding, HostToolKey, ToolWorkspace, require_host_tool_binding,
)

class ReportDraft(BaseModel):
    title: str
    body: str

class SavedReport(BaseModel):
    path: str
    sha256: str

REPORT_FILES = HostToolKey[ToolWorkspace](
    "report.files", accepts=lambda value: isinstance(value, ToolWorkspace),
)

@tool(id="report.save", capability="fs.write", side_effect="write")
def save_report(ctx: ToolContext, draft: ReportDraft) -> SavedReport:
    files = require_host_tool_binding(ctx, REPORT_FILES)
    path = "reports/draft.md"
    digest = files.write_bytes(
        path, f"# {draft.title}\n\n{draft.body}\n".encode(),
        create_dirs=True,
    )
    return SavedReport(path=path, sha256=digest)

def build_loop(adapter, workspace: Path, run_root: Path) -> AgentLoop:
    return AgentLoop.from_tools(
        AgentRunSpec(workspace_root=workspace, run_root=run_root, mode="propose"),
        adapter, [save_report],
        host_tool_bindings=(
            HostToolBinding(REPORT_FILES, open=lambda a: a.workspace_resource()),
        ),
    )
~~~

`ToolWorkspace.write_bytes`는 handler가 정한 경로도 검사한다.
경로 범위와 도구 실행 전 승인은 현재 ToolBinding으로 설정한다.

~~~python
from monoid_agent_kernel import AgentRuntimeConfig, ToolBinding, ToolScope
from monoid_agent_kernel.tool_loader import FunctionToolProvider

config = AgentRuntimeConfig(
    definition_id="report-agent",
    tools=(ToolBinding.for_tool(
        "report.save",
        scope=ToolScope(allowed_paths=("reports/**",)),
        authorization="ask",
    ),),
)
# spec/adapter는 호스트가 준비한다.
loop = AgentLoop.from_config(
    spec, adapter, config,
    tool_providers=(FunctionToolProvider(lambda _ctx: (save_report,)),),
    host_tool_bindings=(
        HostToolBinding(REPORT_FILES, open=lambda a: a.workspace_resource()),
    ),
)
~~~

host는 기존 suspension/control API로 승인 명령을 전달하고, committed park를 release한 뒤 새 loop factory에서 같은 등록을 구성해 restore한다.
`authorization="ask"`는 도구 실행 전 승인이다. 생성된 proposal의 사용자 파일 적용은 기존 apply 단계다.
`run_once`만으로 승인 왕복을 처리한다고 안내하지 않는다.
P3-03의 명시적 상속은 위 등록에 `inherit_to_children=True`를 지정한다.

일반 client의 소유권 예시는 다음과 같다. client는 host-owned이며 이 adapter의 close는 공유 client를 닫지 않는다.

~~~python
class SearchView:
    def __init__(self, client, lease):
        self._client = client
        self._lease = lease

    async def search(self, query: str):
        self._lease.assert_active()
        try:
            return await self._client.search(query)
        finally:
            self._lease.assert_active()

class SearchResource:
    def __init__(self, client):
        self._client = client

    def bind_call(self, lease):
        return SearchView(self._client, lease)

    def close(self) -> None:
        self._client = None  # 이 activation이 빌린 참조 해제
~~~

host는 SDK별 tenant/path/capability 검사와 외부 효과의 fencing·idempotency를 adapter에 구현한다.
위 최소 예시는 수명 검사 패턴만 보여 준다. kernel이 임의 client 구현을 자동 검사하는 계약은 없다.

## 7. 수정 전 회귀와 검증 기준

각 구현 PR은 현재 base에서 실패하는 회귀를 먼저 고정한다.
P3-01은 실제 잘못된 결과를 확인하고, P3-02/03은 신규 API 부재에 따른 실패와 구현 후 동작 회귀를 구분해 기록한다.
guard를 제거한 작은 음성 대조가 stale 접근·scope 위반·cleanup 누락을 검출하는지도 해당 경계에서 확인한다.
독립 reviewer는 이미 유효한 동일 head의 결과를 재사용한다.

| ID / 소유 PR | 실제 입력·경합 | 통과 조건·관찰점 |
|---|---|---|
| C1 / 01 | observer factory가 recorder 생성 뒤 revoke; open와 restore. recorder constructor 내부·외부 guard postcheck의 인계도 PR preflight에서 확인 | 원래 lease_lost 유지, transcript/event handle 닫힘, callback 제거, source checkpoint 보존. 기존 recordable failed-session 결과 유지 |
| C2 / 01 | subscription.close 중 revoke; close/discard/release 각각 | 첫 시도부터 session/loop/callback 정리, 반복 discard에서도 외부 close 중복 없음; late publication 없음 |
| R1 / 02 | constructor duplicates/name collision, mutable 등록 목록 변경 | factory 0회·고정 설정 오류, constructor snapshot 유지; 기존 positional 인자 의미 불변 |
| R2 / 02 | A 생성 성공 후 B 실패, 반환 직후 revoke, 구조/type 검사 실패, 다른 두 key가 같은 adapter 반환 | 성공 반환 resource 역순 close 1회, 제어 예외·안전한 공개 실패 유지. 중복 adapter 거부 및 closer 1회, 권한 postcheck 전에 소유권 인계 확인 |
| R3 / 02 | get_tools에서 lookup, 누락 key, 기존 method-only fake ctx | provider operational view 거부·고정 오류, 기존 fake는 기존 도구에서 계속 사용 가능 |
| A1 / 02 | call A의 view와 ctx를 저장; call B가 더 넓은 scope로 실행 중 접근 | 두 경로 모두 A 종료 오류; B 권한을 빌리지 못함. 실제 handler dispatch + barrier thread 사용 |
| A2 / 02 | sync/async/sync-return-awaitable 성공·raise·cancel·timeout·abandon; descendant thread | 모든 종료 뒤 거부. call 중 명시 전달·join thread는 정상 동작; arbitrary callback lock 교착 없음 |
| A3 / 02 | handler 계산 경로, read/exists/write, binding alias와 다른 scope, read 선언 write 시도 | scope·PermissionPolicy·side_effect 거부가 backend 호출 전에 발생; scope 밖 존재도 숨김 |
| A4 / 02 | read-only/propose/apply, traversal/symlink, max_bytes, hash conflict | 기존 backend 의미 유지, 원본 파일과 proposal 내용·digest를 실제 관찰. 플랫폼 skip은 이유 기록 |
| A5 / 02 | writer revoke 또는 call 종료와 local mutation 경합 | 종료 뒤 새 write 0회, 이미 진입한 작업의 결과 발행 차단, 기존 lease_lost 우선순위 |
| L1 / 02 | close/discard/정상 release 반복; closer 하나 실패; callback 중 revoke | 즉시 접근 거부, 남은 closer 시도, 역순·각 1회, shared writer authority는 정상 종료에서 active 유지 |
| L2 / 02 | 미commit release 거부; aopen/aclose/arelease 취소·재취소 | 거부는 자원 유지, 취소는 sync lifecycle 정리 완료 뒤 전파; session 부활·중복 close 없음 |
| D1 / 02 | proposal 변경 → checkpoint → release → 새 loop.restore → read tool | fresh factory·view, 첫 read가 복구된 delta와 digest 반환; 이전 view 거부 |
| D2 / 02 | partial restore·hosted task restore 실패; duplicate/stale activation command | source checkpoint/hosted task 보존; duplicate/stale는 factory 0회; 새 activation만 생성 1회 |
| D3 / 02 | resource 안 secret sentinel·직렬화 시 실패하는 객체; open/bind_call/accepts가 sentinel 포함 ValueError 발생 | kernel request/schema/event/manifest/checkpoint와 새 진단에 자원·secret·원본 오류 문자열 없음; 승인 전 handler/write 0회 |
| H1 / 03 | root/child/fork default OFF 및 opt-in; grandchild | 기본 누락 오류, opt-in factory마다 올바른 descriptor·별도 identity. 부모 writer token 전달 없음 |
| H2 / 03 | child 작업 중 parent revoke/normal release; child 실패·종료 | 상속 view의 read/write/새 lookup 차단, parent/sibling 자원 생존, lineage callback 해제 |
| X1 / 02·I | fake adapter로 nested ReportDraft → scoped proposal → 승인/park/restart | 공개 API만 사용, wheel 설치본에서 result JSON·proposal·복구 내용 확인; CSP 모듈 import 없음 |

기존 기준 테스트는 다음 13개다. 이번 조사에서는 아래 명령으로 통과를 확인했다.
추가 구현에서는 영향을 받는 `tests/test_loop.py`, tool surface/permissions/authority,
checkpoint/model-I/O/hosting/subagent 테스트 파일을 현재 파일 목록에서 선택해 범위를 넓힌다.

~~~powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:PYTEST_XDIST_AUTO_NUM_WORKERS = '4'
python -m pytest -q -n auto --dist=worksteal --timeout=120 --timeout-method=thread `
  tests/test_checkpoint.py::test_release_parked_closes_process_resources_without_finalizing_or_deleting `
  tests/test_checkpoint.py::test_release_parked_rejects_uncommitted_state_and_leaves_hosted_cleanup_to_host `
  tests/test_checkpoint.py::test_partial_restore_failure_preserves_source_checkpoint_hosted_task `
  tests/test_checkpoint.py::test_restore_bootstrap_failure_discards_published_resources `
  tests/test_checkpoint.py::test_discard_attempts_all_cleanup_after_task_cleanup_failure `
  tests/test_loop_model_io_integration.py::test_discard_closes_model_io_even_when_no_call_was_made `
  tests/test_loop_model_io_integration.py::test_checkpoint_delete_failure_ends_activation_without_double_closing_sinks `
  tests/test_loop_model_io_integration.py::test_cancelled_aopen_waits_for_bootstrap_then_discards_without_resurrection `
  tests/test_model_stream_observer.py::test_restore_materializes_a_fresh_observer_snapshot `
  tests/test_model_stream_observer.py::test_subagent_materializes_fresh_observer_with_root_lineage `
  tests/test_activation_hosting.py::test_activation_rejects_stale_source_before_constructing_loop `
  tests/test_activation_hosting.py::test_boundary_commit_crash_recovers_without_reexecution `
  tests/test_subagent.py::test_parent_capability_revoke_blocks_child_gated_tool
~~~

실행 interpreter는 원본 checkout의 `.venv/Scripts/python.exe`였으며, PYTHONPATH는 이 worktree의 src로 고정했다.
`test_parent_capability_revoke_blocks_child_gated_tool`은 CapabilityVault 회귀다.
P3의 parent writer-authority lineage는 H2가 새로 검증한다.

L0는 위 경계별 focused tests + 변경 파일 lint·공개 API 호환 검사로 시작한다.
L1/L2 core는 현재 repository workflow의 Python 3.11/3.12를 사용한다.
root recipe는 최저 지원 Pydantic 2.6과 현재 버전도 확인한다.
P3-I는 installed wheel + 전체 기존 CI를 통과해야 한다.
새 서비스 adapter/codec 변경이 없으므로 모든 PR에 PostgreSQL/MinIO/Temporal profile을 자동 부과하지 않는다.
최종 통합에서 기존 full CI의 service 검사·coverage·tracked UI bundle 검사를 줄이지 않는다.
head/base가 바뀌면 영향을 받는 merge candidate 증거를 갱신한다.

## 8. 결정·리뷰·후속 기록

초기 설계 결정: 제한된 typed key/adapter, 호출에 고정한 façade, deferred workspace,
동기 lifecycle, 표준 cleanup, 기본 상속 OFF와 명시적 fresh child, 기존 cleanup 두 지점 선행 보완.
계획 검토 당시 구현 변경 0, 새 회귀 테스트 0. baseline probe 2건, 기존 회귀 13건 통과.
fresh 독립 reviewer `p3_plan_review`의 최초 판정은 **ready — 계획 단계 진행 가능**이다.
대상은 base `5b08b10`과 계획 SHA256 `B63FD7982E9490ECE614A81E52178C20DE3FF93F8B4F4086CFBE3A0E34D2AB05`다.
작성·조사에 참여하지 않은 reviewer가 실제 dispatch·bootstrap·close/discard·restore·CSP 소비자와 외부 근거를 대조했다.
새 테스트 실행 없이 기존 13개 결과와 두 probe 원본을 재사용했다. 구현을 차단하는 설계 결함은 없었다.

확인사항 두 건을 구체화했다: callback의 secret 포함 ValueError를 안전한 공개 오류로 바꾸는 D3,
서로 다른 key의 동일 adapter 반환을 거부하고 한 번만 정리하는 R2다. lexical path 계약도 연결했다.
작성자는 추가 소스 대조로 recordable failed-session의 recorder 수명과 host factory 실패의 cleanup 시점을 명시하고,
recorder constructor의 guard postcheck를 P3-01 사전조사 입력에 포함했다.

같은 reviewer가 최종 정책 SHA256 `4713A36EFB11D26ADC05B762804E6669CC52DF1FB6224507A2E897F3EC9AF5CF`를
targeted 재검토했고 **ready, 새 차단 지적 없음**으로 판정했다. failed-session 호환성·중복 소유권·안전한 진단·lexical scope
변경을 확인했다. 계획 commit까지의 후속 수정은 이 리뷰·문서 검증 기록뿐이었다. 해당 시점 source 변경은 0이다.
문서의 Python 예시는 AST 문법 검사만 수행했으며 새 API의 실행 성공을 주장하지 않는다.
최종 문서 검증은 5개 관련 문서의 로컬 링크·anchor **127개**, Python AST block **10개** 통과다.
diff 공백 검사와 production/test/example/CI source가 기준 commit과 동일하다는 검사도 통과했다.

| carry | 담당·다시 여는 조건 |
|---|---|
| 기존 ToolContext 전체 메서드의 mutable fallback | kernel 권한 유지보수 담당. 신규 resource 바깥에서 실제 scope 차용 재현 또는 전체 context 계약 변경 요청 시 별도 PR |
| generic client의 외부 효과·tenant 권한 | host adapter 소유자. kernel 제공 DB/HTTP adapter를 제품 범위에 넣을 때 별도 조사·실제 서비스 검증 |
| 자식 모델 중단·drain·usage 및 기존 builtin parent writer 연동 | P8 담당. P8 착수 시 현재 P3 lineage와 함께 검토 |
| async owned resource/유한 cleanup deadline | host 통합 담당. 동기 callback 계약으로 실제 소비자를 지원하지 못할 때 API·실행 모델을 다시 선택 |
| CSP raw workspace 보관 제거 | CSP 통합 담당. P3 패키지 릴리스 뒤 새 seam으로 이관 PR·pin·운영 검증 |

각 PR 착수 시 이 기록에 실제 base/head, 수정 전 실패, 수정 후 결과, 최종 리뷰·CI, 예상 대비 누적 비용을 이어 기록한다.

## 9. 구현 실행 기록

### P3-00 사전조사 — 2026-10-27

- 최신 fetch 결과 `origin/develop=5b08b10d82a948268d1a95b51112b1508e233e3d`로 계획 기준과 동일하다.
  `3599b21`은 승인된 계획 문서만 담은 commit이다. 이를 포함해 `codex/v0.25-p3-00-ci-base`에서 준비 PR을 만든다.
- 실제 생산자·소비자: `.github/workflows/ci.yml`의 push branch allowlist가 통합 브랜치의 전체 CI 실행을 결정한다.
  현재 main/develop/v0.23만 포함한다. `pr-fast.yml`/`pr-full.yml`에는 base branch 제한이 없다.
  `tools/v023_ci.py::resolve_profile`은 v0.23 외 일반 PR의 기본 core profile을 이미 지원한다.
- 선택: allowlist에 `codex/v0.25-p3-host-resources` 한 항목 추가. CI job·권한·SDK pin·service profile·검사 도구는 유지한다.
  단순 등록에 새 mirror test를 만들지 않고 YAML readback, 기존 `tests/test_v023_ci_foundation.py`, lock 검사를 사용한다.
- 현재 계획의 독립 검토는 동일 production base를 대상으로 했으므로 유효하다. 이 PR의 독립 preflight reviewer는
  `p3_00_preflight_review`다. 새로운 CI 등록 동작을 검토하고, 실제 최종 diff도 별도로 확인한다.
- 범위·비용: 기능 1줄 변경 예상. 이전 계획 문서 commit을 함께 게시하므로 PR 전체 문서량은 초기 준비 PR 추정보다 크다.
  새 구현·검증 체계를 추가하지 않으며 계획 수립 비용을 중복 계상하지 않는다. L1/L2 core 및 develop merge CI를 확인한다.

독립 P3-00 사전조사 판정은 **ready**다. 기존 모든 full-CI job이 새 branch push에도 적용되고,
PR base 제한·profile 변경이 필요 없음을 확인했다. `validate-lock`은 valid이며 lock SHA256은
`5e536963c707778e9309936d5768edc367a6c49ece5c51223999dde46697b498`이다.
`resolve-profile --head-ref codex/v0.25-p3-host-resources`는 core를 반환했다.
Ready 이후 head/base가 바뀌면 현재 후보로 L2를 다시 실행하고, 실제 통합 branch push CI도 확인한다.

P3-00 local 검증: 기존 CI foundation **18 passed / 3.78s**. `ci.yml`의 기준/현재 전체 텍스트 비교에서
정확한 push branch 한 항목만 추가되고 job·permissions가 그대로임을 확인했다. YAML parser는 로컬 환경에 없어
새 의존성을 설치하지 않았고 기존 workflow 검사와 GitHub CI를 사용한다. diff 공백 검사도 통과했다.

P3-00 최종 head `6274c20de0afcd53bb267a35e3308982dc53fd61`의 독립 diff review는 PASS다.
[PR #150](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/150)을 develop 대상으로 열었다.
현재 P3-01 브랜치는 이 준비 head에서 분기했으며, PR #150 병합 후 최신 develop을 포함한 독립 diff로 제출한다.

### P3-01 사전조사 — 2026-10-27

- source base는 `5b08b10`이며 P3-00 head `6274c20`과 production tree가 같다. 브랜치는
  `codex/v0.25-p3-01-cleanup`이다. root가 구현·runner를 소유하고 `p3_01_research`가 읽기 전용 조사를 수행했다.
- recorder 내부에서도 같은 인계 결함이 있다. `AgentRecorder.__post_init__`의 decorator가
  initialization body 반환 뒤 authority를 검사하므로, 해당 실패는 body 안의 handle cleanup 밖에서 발생한다.
  초기화 동작을 작은 내부 함수로 두고 **guard 호출 전체를 기존 handle cleanup try 안에** 넣는다.
- bootstrap은 guarded constructor callback 안에서 provisional recorder를 먼저 확보한다.
  첫 `_bootstrap_resources` 게시까지 local cleanup try를 유지하고, 그 전 실패에는 local discard를 수행한다.
  게시 시점은 유지한다. provider/config 실패를 기록하는 기존 terminal failed session과 recorder의 close 수명을 보존한다.
- 현재 게시 전 TaskManager는 executor/injector만 만들고 event loop는 지연 생성한다. 이 PR의 실소유 handle은
  recorder다. 하나의 소유권에 local try/finally와 기존 closed flag를 사용한다. 범용 cleanup manager나 새 state machine은
  추가하지 않는다. 여러 host resource를 획득할 P3-02는 표준 ExitStack 적용을 별도로 검증한다.
- escaping open/restore 실패는 best-effort discard 뒤 원래 오류를 재발생한다. ordinary Exception의 기존
  failed-session 기록 여부는 유지하고, BaseException 제어 신호는 기록 경로로 바꾸지 않는다.
- close는 성공한 recorder finalization 후 subscription close를 기존 checkpoint-delete try/finally 안으로 옮긴다.
  discard는 subscription 오류가 있어도 session/resources/stream/callback/owned loop를 정리한다.
  기존 subscription authority 오류는 앞서 모은 ordinary cleanup 오류보다 우선한다. 원래 bootstrap/restore/finalize
  오류가 이미 전파 중이면 cleanup 오류가 이를 덮지 않는다. ordinary observer 오류의 격리 정책도 유지한다.
- release_parked에는 이미 실패 시 discard fallback이 있다. 해당 경로는 baseline 실패라고 단정하지 않고 통과 대조로 둔다.
  close는 subscription close 전에 run.finished를 이미 기록하지만, release/discard는 이를 기록하지 않는다.
  revoke 뒤 checkpoint 삭제·외부 observer 후속 호출이 시작되지 않는지도 실제 파일과 callback 수로 확인한다.
- 회귀 범위: real recorder constructor guard postcheck; bootstrap external postcheck/observer revoke의 open·restore;
  close/discard/committed release의 subscription revoke; 기존 recordable failure 결과; 원래 escaping error 보존.
  restore는 실제 durable checkpoint를 남겨 hosted task 취소 표식과 checkpoint 보존도 확인한다.
  실패 회귀를 production 수정 전 별도 commit으로 고정한다.
- 기존 `test_cancellation.py` constructor capture, checkpoint/model-I/O/hosting/session lifecycle 회귀를 재사용한다.
  예상 기능 100–250줄, 전체 300–600줄을 유지하고 indentation 이동도 실제 diff 비용에 포함해 기록한다.
  A 등급 독립 preflight, 서로 다른 두 final-code reviewer, L1/L2 core, develop 통합 CI가 완료 기준이다.

P3-00은 [L1](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/40420031463),
[L2 required](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/40420031631)를 통과하고
develop `61865effbabfec28adebd85e9574c52bc572c62a`로 병합했다. P3-01은 이 최신 base로 갱신했다.
새 base의 production source는 `5b08b10`과 같다. 전체 develop CI run은 `40420547924`다.

P3-01 독립 preflight reviewer `p3_01_preflight_review`의 판정은 **ready**다.
검토 대상은 base `61865ef`와 preflight SHA256 `038001BC9396D04F66B697501728507C402FBF5B0226108E23234CA29957614A`다.
추가 constructor probe에서도 initialization body 반환 뒤 revoke하면 두 owned handle이 모두 열린 채 남음을 확인했다.
출력은 `.tmp/v025/p3-01-constructor-probe.json`이다.

**수정 전 회귀:** `tests/test_activation_cleanup.py`는 **12 failed, 3 passed / 4.31s**다.
실패는 constructor/초기화 handle 누수, 첫 종료의 session 잔존, cleanup이 원래 오류를 덮는 경계다.
committed release와 provider/config의 recordable failed-session 수명은 통과 대조다.
초기 fixture의 config 주입은 runtime-config provider Protocol에 맞춰 수정한 뒤 이 결과를 고정했다.
production source는 아직 base와 동일하다. 원본 출력은 `.tmp/v025/p3-01-red.log`다.

~~~powershell
python -m pytest -q -n auto --dist=worksteal --timeout=120 --timeout-method=thread tests/test_activation_cleanup.py --tb=short
~~~

Python은 공유 `.venv`의 3.11.9이며 PYTHONPATH는 이 worktree의 src, xdist worker 상한은 4다.
ruff는 CI와 같은 0.15.6을 `uvx`로 사용한다. 이 회귀·preflight를 먼저 commit하고 production 수정을 이어간다.

수정 전 회귀 commit은 `b4fff8f`다. constructor guard 전체를 handle cleanup 안으로 이동하고,
bootstrap의 guarded callback에서 provisional recorder 소유권을 확보했다. 첫 resource 게시 시점과
recordable failed-session 수명은 유지했다. close/discard의 subscription 실패 뒤에도 teardown을 마치며,
escaping open/restore 제어 예외도 정리 후 원래 의미로 전파한다.

동일 신규 회귀는 **15 passed / 3.61s**였다. 기존 checkpoint·취소·model-I/O·hosting·loop/session lifecycle·
stream observer·public/compat를 포함한 최종 focused 검사는 **337 passed / 12.90s**다.
원본은 `.tmp/v025/p3-01-green.log`, `.tmp/v025/p3-01-focused.log`다.

~~~powershell
python -m pytest -q -n auto --dist=worksteal --timeout=120 --timeout-method=thread tests/test_activation_cleanup.py tests/test_checkpoint.py tests/test_cancellation.py tests/test_loop_model_io_integration.py tests/test_activation_hosting.py tests/test_agent_loop_lifecycle.py tests/test_session_lifecycle.py tests/test_session_drive.py tests/test_model_stream_observer.py tests/test_public_surface.py tests/test_compatibility_ledger.py --tb=short --durations=8
~~~

기능 수정은 기존 소유권과 finally를 사용한다. 공백을 제외한 production diff는 41줄 추가/11줄 제거이며,
일반 diff에는 bootstrap cleanup 범위의 indentation 이동도 포함한다. 새 lifecycle framework는 추가하지 않았다.

P3-00 develop 전체 CI `40420547924`도 14개 job 모두 성공했다.
P3-01 첫 후보 `d3d047d`의 독립 코드 reviewer `p3_01_code_review_a/b`는 같은 P2를 재현했다.
기존 discard helper는 sink 목록에서 첫 JsonlEventSink를 선택하므로 authoritative host sink가 먼저 있으면
그 외부 sink를 닫고 owned events 파일은 남긴다. 새 prepublication caller에서도 이 문제가 도달 가능했다.
추가 회귀는 provisional/published 두 경로 모두 **2 failed / 0.70s**로 고정했다(commit `fb55581`,
`.tmp/v025/p3-01-owned-sink-red.log`). 생성자가 정확한 owned sink를 private 필드에 보관하고
discard가 그 인스턴스만 닫도록 수정했다. 정상 EventBus.close 계약은 유지한다.
같은 focused 검사에 두 회귀를 더한 결과는 **339 passed / 17.75s**다
(`.tmp/v025/p3-01-final-focused.log`). ruff 0.15.6 및 diff 공백 검사도 통과했다.
최종 후보를 두 reviewer가 표적 재검토한 뒤 독립 PR의 L1/L2를 확인한다.

두 reviewer의 최종 판정은 `47ff2cf424b88189082202173601bb66b97910f3`에 대해 PASS다.
[P3-01 PR #151](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/151)은
[L1](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/40426994584)과
[L2](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/40426994558)를 통과했다.
자동 GitHub review도 동일 head에 finding 없이 완료됐다. develop merge는
`4539296dd9633743144d75a62a562a2d636d05a8`이다. 최종 전체 diff는 619 추가/133 제거이며
bootstrap indentation과 테스트·문서가 대부분이다. 이 base에서 통합 branch
`codex/v0.25-p3-host-resources`를 생성·push하고 P3-02 branch를 분기했다.

### P3-02 사전조사 — 2026-10-27

- 최신 develop·통합 base는 `4539296dd9633743144d75a62a562a2d636d05a8`이다.
  실행 branch는 `codex/v0.25-p3-02-root-resources`. root가 구현/runner를 소유하고
  `p3_02_preflight_research`가 독립적으로 소스 연결을 조사한다. 구현 전 별도 preflight review를 받는다.
- 목표 consumer는 패키지에 포함할 offline report recipe다. 중첩 Pydantic 입력/출력, computed report path,
  scoped proposal write, 승인 대기, committed release, 새 loop restore와 읽기를 공개 API로 연결한다.
  from_tools 전달 smoke와 scoped from_config 전체 흐름을 모두 검증한다. 기존 wheel audit/install CI를 재사용한다.
- constructor는 binding tuple/중복 key identity·name/동기 callable 형태를 첫 callback 등록·파일 생성 전에 검사한다.
  key/resource는 checkpoint와 tool schema에 넣지 않는다. descriptor의 공개 타입 이름은 `HostToolActivation`으로
  확정해 factory를 정적으로 작성할 수 있게 한다. 나머지 public 이름과 helper는 §4.1을 유지한다.
- `LoopBootstrapper.bootstrap`의 context 생성 직후 registry를 loop에 연결하고 factories를 연다.
  첫 `_bootstrap_resources` 게시 전 실패는 registry → provisional recorder 순서로 정리한다.
  valid closer를 factory callback 안에서 ExitStack에 등록하고 guard postcheck를 수행한다.
  duplicate adapter identity는 두 번째 등록 전에 거부한다. 일반 callback 오류는 고정 메시지로 바꾸며
  직접 발생한 RunCancelled/RunTimeout/TurnInterrupted/TurnPaused 및 BaseException 제어 신호를 유지한다.
  awaitable 반환은 기존 `core._sync_bridge.dispose_unawaited`로 처리한 뒤 거부한다.
- `_ainvoke_handler`가 실제 admission 뒤 받은 immutable authorization과 side_effect를 새 call lease에 고정한다.
  `authorization.scope`와 `surface_scope`를 모두 검사한다. provider façade에는 활성 lease가 없다.
  기존 ContextVar enter/exit를 유지하고 새 helper만 call façade를 사용한다. 진입 assert까지 finally 범위에 둔다.
- call/key 첫 lookup의 동시성은 짧은 call-state lock 안에서 표준 `concurrent.futures.Future`를 등록한다.
  한 winner가 lock 밖에서 bind/accepts를 실행하고 다른 explicit thread는 lease를 재검사하며 결과를 기다린다.
  사용자 callback을 가진 lock 기반 lazy cache와 새로운 async resource scheduler는 도입하지 않는다.
  call 종료는 closed Event를 먼저 set하고 cache를 비운다. 실패 결과에도 원본 secret 예외를 저장하지 않는다.
- local workspace 작업의 lock 순서는 **call → activation → writer**다. 종료는 Event를 먼저 set하고 해당 lock으로
  진입한 작업만 기다린다. activation 종료는 call lock을 획득하지 않는다. writer revoke callback은 기존처럼
  writer lock 밖에서 실행된다. arbitrary open/bind/accepts/close 및 일반 client I/O는 이 lock들을 잡지 않는다.
  workspace는 기존 backend의 local-mutation 계약을 재사용하며, 일반 network adapter는 lease 전후 guard와
  host의 외부 fencing을 사용한다. scope는 normalized lexical path이며 별도 alias/OS sandbox를 추가하지 않는다.
- 정상 close는 진입 시 P3 수명을 끝내고, 기존 finalization/subscription/checkpoint 처리의 finally에서 자원을 닫는다.
  committed release는 fingerprint 검증 후 수명을 끝내며 subscription/recorder 또는 resource close 실패 시 기존 discard로
  마무리한다. discard는 P3 close 오류를 모은 뒤 job/recorder/subscription teardown을 계속한다.
  기존 원래 오류를 보존하고, 이미 관측된 authority loss 우선순위도 유지한다. recordable bootstrap 실패에서는
  P3 registry만 즉시 닫고 기존 failed-session recorder 수명을 유지한다. async lifecycle bridge는 그대로 사용한다.
- registry는 callback 전에 closed 표시 및 owned map을 비우고 ExitStack으로 역순 최대 한 번 dispose한다.
  추가 close 실패는 값·repr·traceback 없는 고정 진단만 남긴다. normal close가 host writer를 revoke하지 않는다.
  deferred workspace adapter는 실제 backend 내용에 접근하지 않고 call view에서만 연산한다.
  restore 순서와 schema를 유지한다. child/fork에는 bindings를 전달하지 않으며 상속 옵션은 P3-03에서 추가한다.
- 필요한 L0는 tuple/중복/partial-init, safe-error/async rejection, computed scope·policy·mode·hash,
  retained A ctx/view while B active, explicit thread와 sync→awaitable, native async·timeout·abandon,
  local-write barrier/revoke·close 경쟁, 정상/실패/park/restore/async-cancel cleanup과 report consumer다.
  새 API의 happy-path만으로 완료 판정하지 않는다. 기존 cleanup339·tool surface·checkpoint·public/compat 검사를 합친다.
- 예상 기능 600–1,000줄, 전체 1,300–2,100줄과 campaign 누적 한도를 유지한다. resource registry만 분리하지 않는다.
  A 등급 independent preflight, 서로 다른 두 final-code review, L1/L2 core, integration push full CI,
  실제 installed-wheel report 실행이 완료 신호다. 외부 근거는 §3의 표준 ExitStack 및 실제 merged fixes를 사용한다.
- 사전 검토가 지적한 종료 진입 경계도 포함한다. 이미 writer가 revoke된 상태에서 close/release를 호출하면
  현재 선두 assert가 teardown 밖에서 실패한다. 이 두 종료 진입점은 lease_lost를 보존하면서 기존 discard를
  best-effort 수행해 P3 자원과 나머지 owned handle을 정리한다. 일반 uncommitted release 거부는 계속 자원을 유지한다.
  두 직접 진입 회귀와 기존 stale cleanup 대조를 추가한다.

독립 reviewer `p3_02_preflight_review`는 위 보완을 포함한
SHA256 `E6D6AAFB14A9AEE0456DCAF4165281A3682391B5CF120BB75EF49CD7C56E0144`에 READY를 반환했다.
root가 소스 조사를 수행했고 추가 research worker는 초기화 대기 상태여서 결과 근거에 포함하지 않았다.
구현은 7개 공개 이름을 신규 모듈에 제공하며 기존 root export/ToolContext 필수 멤버를 유지한다.
callback close가 writer를 revoke하는 경우에도 local dispose를 모두 시도한 뒤 lease_lost를 보존한다.

초기 typed report smoke는 approval replay의 기존 `tool_approval_replay:read`/nested result envelope에 맞춰
검증 경로를 수정했다. staging backend는 host가 제공한 staging directory에 직접 쓰므로 overlay와 구분해
기존 동작을 검증했다. 신규 회귀 및 기존 cleanup/checkpoint/cancellation/hosting/session/loop/async/tool-surface/
workspace-permission/public/compat 검사는 **583 passed, 1 platform skip / 17.56s**다
(`.tmp/v025/p3-02-focused.log`). 추가 hosted duplicate-command 테스트 및 close/secret/awaitable 검사
**5 passed / 0.89s**도 확인했다. 실제 duplicate command는 host factory를 다시 실행하지 않았다.

새 wheel audit에 두 모듈을 포함하고 기존 minimal install CI에서 report recipe를 실행하도록 연결했다.
별도 venv에 wheel을 설치하고 PYTHONPATH를 비운 checkout 밖 실행에서 `site-packages` import를 확인했다.
Python3.11.9/Pydantic2.14.1 환경의 결과는
`{"activations":2,"approved_read":true,"proposal_only":true,"status":"completed"}`다.
로컬 focused 환경의 Pydantic은 2.13.4다. 의존성 범위·배포 버전은 변경하지 않았다.
최종 code review 및 PR CI 증거는 후보 확정 뒤 기록한다.

### 통합 CI fixture 보완 — 2026-10-27

P3-01 #151은 최종 head `47ff2cf`의 두 독립 review와 L1/L2를 통과해 develop `4539296`으로 병합됐다.
이후 develop CI `40427565429`와 새 P3 통합 branch CI `40427573024`가 기존
`test_cancel_ack_does_not_clobber_a_committed_park_when_the_drive_is_mid_pump`에서 실패했다.
coverage 수치는 80.67%로 기준을 충족했다. 실패 원인은 해당 테스트의 동시성 조건이다.

원래 base `61865ef`와 병합 head `4539296` 모두에서 driver의 park snapshot 직전 barrier를 삽입하면
fixture가 `last_suspension=None`을 먼저 써서 같은 sequence의 checkpoint를 오염시킨다.
`AWAITING_INPUT` 상태는 최종 snapshot보다 먼저 게시된다. backend/drive/test source는 두 revision에서 같다.
원본 probe는 P3 worktree `.tmp/v025/p3-01-cancel-triage/`에 있으며, 양쪽 revision에서 원래 실패와
직렬화 대조를 확인하는 2개 probe가 각각 3.39s/3.12s에 통과했다.

선택: fixture의 인위적인 marker 변경을 기존 `_run_on_shared_loop`로 보낸다. 앞선 driver snapshot 완료 후
변경하고 cancel ack를 실행하므로 원래 quiescence·checkpoint 보존 검증을 유지한다.
production 변경은 0줄이다. 독립 B 등급 reviewer `p3_01_code_review_a`의 사전 판정은 READY다.
별도 `codex/v0.25-p3-ci-ordering` PR로 develop에 적용하고 통합 branch가 이를 포함한다.
예상 test 변경 6줄과 이 기록; 기존 focused/module 검사·L1/L2·최종 full CI로 검증한다.

수정 뒤 `tests/test_backend_control.py` 전체는 **56 passed / 5.48s**다. 표준 4-worker pytest,
ruff0.15.6 및 diff 공백 검사를 통과했다. production source diff는 없다.

PR #152의 최종 head `bfff4b7`은 독립 B 등급 review PASS, L1 `40430303690`,
L2 `40430303478`을 통과했다. 자동 GitHub review도 동일 head에 finding 없이 완료됐다.
develop merge는 `336cb4cb59d78e9bc083cfbbdd9500817f6065da`다. 통합 branch와 P3-02가
이 prerequisite를 포함한다. P3-02 production diff는 최초 검토 후보 `9df07fb`와 동일하다.

P3-02 최종 head `73f1c3995f9ebefa42f8183dbef58fe3263a5d26`의 독립 reviewer
`p3_02_code_review_a/b`는 모두 PASS다. 추가 probe에서 bind 진행 중 종료한 call의 winner/waiter 거부,
취소를 삼킨 native async handler의 late write 거부를 각각 확인했다. 별도 installed-wheel 환경의
**Pydantic 2.6.4**에서도 같은 report recipe 결과를 확인했다. 새 의존성·버전 변경은 없다.
[PR #153](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/153)은
L1 `40431097973`, L2 `40431098042`를 통과했고 자동 review도 동일 head에 finding 없이 완료됐다.
통합 merge는 `78cd610aac3f27b02de09ad3e2136bf84e1e2583`이다. 전체 diff는 1,824 추가/28 제거,
production은 480줄 registry와 156줄 소비 예제, 기존 loop/bootstrap 연결로 계획 범위 안이다.

### P3-03 사전조사 — 2026-10-27

- 최신 fetch: develop `336cb4cb59d78e9bc083cfbbdd9500817f6065da`, 실제 구현 base는
  P3 통합 `78cd610aac3f27b02de09ad3e2136bf84e1e2583`이다. 실행 branch는
  `codex/v0.25-p3-03-child-resources`. root가 소스 조사·구현·runner를 소유하고 fresh reviewer가
  아래 설계와 실제 소비자 연결을 독립 검토한다. P3-02 integration full CI를 확인한 뒤 구현한다.
- 실제 생산 지점은 `AgentLoop._run_subagent_child` 하나다. fresh/fork 모두 이곳에서 새 loop를 만들고
  `arun_once`를 실행한다. `SubagentRuntimeContext.create`는 검증된 root/parent/task lineage를 제공한다.
  fresh는 tool allow/deny와 mode/limits를 좁히고, fork는 parent config를 복사한다. 이 정책은 유지한다.
- `HostToolBinding`에 keyword-only bool `inherit_to_children=False`를 추가한다. constructor가 정확한 bool을
  검사한다. 매 hop에서 True 등록만 자식 생성자에 전달한다. factory closure/key는 공유하고 반환 adapter,
  workspace resource, call view는 해당 자식 activation에서 새로 생성한다. public WriterToken 전달은 없다.
- 자식 생성 전에 현재 parent registry의 수명 token을 확보하고, private init=False loop 필드로 전달한다.
  registry의 `_ResourceLifetime`은 parent의 같은 수명 token만 참조한다. parent registry/workspace/adapter는
  보관하지 않는다. 연결된 token chain의 assert가 root부터 child까지 closed 상태와 writer를 검사한다.
  parent 복구 시 새 registry/token을 만들므로 이전 child가 새 activation에 연결되지 않는다.
- 사전 reviewer가 실제 공통 handler 진입의 무조건 `host_call.assert_active()`를 확인했다. 여기에 parent
  chain을 그대로 적용하면 P3 자원을 사용하지 않는 기존 child tool까지 정책이 확대된다. 수명 token의
  `assert_local_active()`로 현재 local closed+writer 검사를 분리하고 handler 진입은 이 검사를 유지한다.
  full lineage 검사는 factory·lookup·bind·lease·workspace에 적용한다. 실제 parent P3 종료 후에도 일반
  child tool은 실행되고 자원 접근은 거부되는 대조를 둔다. 기존 builtin 권한 연동 정책은 P8에 유지한다.
- 독립 실제 child/fork probe는 operator policy 전달 결함도 확인했다. 부모 생성자의 명시적
  `permission_policy`가 spec과 다르면 부모는 deny하지만 자식 생성의 두 지점은 spec policy를 다시 사용한다.
  상속 자원도 이 누락을 받으므로 child spec/constructor 모두 부모의 해석된 `self.permission_policy`를
  전달하도록 선행 보완한다. bootstrap의 기존 policy 우선순위는 유지한다. 기존 fs.write → agent.spawn
  consumer의 loop keyword/spec/override × fresh/fork 회귀를 수정 전에 고정하고, 신규 computed workspace
  path에도 같은 policy가 적용됨을 검증한다. policy 모델·병합 규칙 전체를 바꾸는 작업은 포함하지 않는다.
- callback 전파 대신 매 접근의 lineage 검사와 기존 RLock guard를 조합한다. 새 revoke callback을 등록하지
  않으므로 child 종료 시 callback 해제 장치를 추가할 필요가 없다. parent 종료·revoke는 sticky Event를 먼저
  게시한다. local 연산은 **call → root activation → root writer → ... → child activation → child writer**
  순서로 재귀 guard에 진입하고 반대로 빠져나온다. parent end는 자신의 activation lock만 기다린다.
  writer의 기존 revoke callback은 writer lock 밖에서 실행된다. 다른 call lock을 획득하지 않는다.
- open은 lineage 전체의 external pre/post guard를 사용한다. callback 반환 즉시 P3-02와 같은 closer 등록을
  완료한 다음 postcheck로 나가므로 parent가 factory 중 종료해도 fresh child adapter를 회수한다.
  bind/accepts/general client guard도 같은 lineage assert를 사용한다. 임의 host callback을 잠근 채 실행하지 않는다.
- parent 정상 close/committed release/discard는 기존 P3 lifetime만 끝낸다. child cleanup은 child adapter만
  닫으며 parent·sibling writer를 revoke하지 않는다. 부모 종료 뒤 child의 이미 진입한 local write는 끝날 수 있지만
  결과를 성공으로 발급하지 않고 후속 write를 막는다. P8의 모델 호출 취소·drain·과금 합산은 이 변경과 독립이다.
- 외부 근거를 다시 확인했다. [LangGraph #5707](https://github.com/langchain-ai/langgraph/pull/5707)은
  parent runtime 전달 누락을 고친 merged 수정이며 runtime context는 durable state와 별도다.
  이는 실제 child 진입점을 검증할 근거다. Monoid의 fresh adapter와 parent authority 회수는 우리 제품의 추가 계약이다.
  표준 RLock/context-manager 중첩과 기존 guard를 사용하고 DI graph·새 scheduler·callback registry는 추가하지 않는다.
- 첫 consumer는 패키지 offline report 예제의 자식 report 실행이다. 실제 `agent.spawn`을 사용해 fresh/fork를
  모두 실행하고 root·child descriptor, 서로 다른 adapter, child close 후 parent 접근, sibling 독립을 관찰한다.
  기존 report 승인·release·restore 예제를 유지하고 installed-wheel CI에서 child consumer도 실행한다.
- 검증은 실제 nested child/grandchild와 fork 기본 거부/opt-in, root 및 중간 parent 종료·writer revoke 뒤
  현재 call의 view/helper 거부, 새 factory 중 parent 종료의 partial cleanup, 들어간 local write와 parent end/revoke
  barrier, parent 복구 후 새 child freshness, async·explicit-thread call 보장, callback 수/역순 close를 포함한다.
  handler 반환 뒤 call 종료만으로 통과하는 검사를 lineage 증거로 사용하지 않는다. 기존 subagent·root P3·
  cleanup·checkpoint·cancellation·public/compat 회귀와 wheel current/minimum Pydantic 검사를 재사용한다.
- 예상 production 150–300줄, 총 450–800줄을 유지한다. 현재까지 구현 PR의 전체 diff 합계는 약 2,634줄
  (P3-01 752, P3-02 1,852, CI fixture 30; 최초 계획 문서 비용 별도)다. 새 기능·테스트/문서도 누적에 포함한다.
  A 등급 독립 preflight → 두 fresh code review → L1/L2 → 통합 full CI 후 P3-I로 연결한다.

독립 reviewer `p3_03_preflight_review`는 두 보완을 반영한 preflight
SHA256 `5F1AEE882658BE31DEBA24AFB21783A26E6B1BF8D7B03F6EED3E19C049D9551A`에 **READY**다.
정책 probe 원본/결과는 `.tmp/v025/p3-03-preflight-review-policy.py`와 같은 이름의 `.json`에 있다.
해당 probe는 미구현 상속 전달만 메모리에서 모사했으며 production 완료 증거로 사용하지 않는다.
CI fixture 보완 뒤 develop `40430925138` 및 integration `40430979676` 전체 CI는 모두 성공했다.

수정 전 `tests/test_host_tool_children.py`는 **6 failed, 4 passed / 3.13s**다.
4개 실패는 명시적 loop policy/override의 fresh·fork에서 실제 `fs.write`가 허용되는 기존 결함이다.
나머지 2개 실패는 `inherit_to_children` 신규 API가 아직 없어 생긴 TypeError다.
기본 상속 거부와 spec policy 경로 4개는 통과 대조다. 초기 fixture의 reports directory를 준비한 뒤
실제 child write 성공을 관찰해 고정했다. production source는 base와 동일하다.
원본은 `.tmp/v025/p3-03-red.log`. 표준 4-worker pytest로 새 test module만 실행했다.

수정 전 회귀 commit은 `f10f6de`다. P3-02 integration 전체 CI `40431588835`의 14개 job 성공을
확인한 뒤 production을 변경했다. opt-in field, fresh child factory, parent lifetime chain,
effective policy 전달과 handler의 local/resource 검사를 연결했다. 초기 10개 회귀는 **10 passed / 2.87s**다.

실제 child/fork/nested 실행, 호출 중 root·중간 parent 종료/revoke, callback 전후, 진입한 child write barrier,
sibling/parent 독립, mixed opt-in, fresh restore 및 packaged consumer까지 **35 passed / 5.96s**다.
원본은 `.tmp/v025/p3-03-green.log`. 초기 restore fixture는 공유 FakeModelAdapter의 소비된 turn list를
복사하도록 수정했고, 중간 parent revoke의 기존 `TurnNotSettled(interruption_cause=lease_lost)`도 보존했다.
기존 `snapshot()`은 실행 중인 in-process resume job을 durable park로 허용하지 않는다.
공개 committed release 회귀는 `create_task(..., resume_on_exit=False)`의 host-driven child를 사용해
정상 committed 경계에서 parent를 release하고, 아직 살아 있는 child call의 접근 거부를 검증한다.
새로운 background-task 복구 정책은 추가하지 않았다.

최종 focused 검사는 **647 passed, 1 platform skip / 20.31s**다
(`.tmp/v025/p3-03-focused.log`). 기존 P3-02 focused 목록에 새 child tests와 기존
`test_subagent.py`/`test_subagent_runtime_context.py`를 추가했다. ruff0.15.6과 diff 공백 검사도 통과했다.
메모리에서 parent token 연결만 제거한 음성 대조는 child write 도중 parent end가 즉시 반환하여
barrier 회귀가 **1 failed / 0.66s**로 검출했다. production은 변경하지 않았고 원본은
`.tmp/v025/p3-03-parent-guard-negative.py`/`.log`다.

현재 source wheel audit 및 PYTHONPATH 없는 checkout 밖 installed-wheel recipe를 다시 실행했다.
Python3.11.9/Pydantic2.14.1과 최소 계열2.6.4 모두 승인·복구 및 fresh/fork report를 완료했다.
각 흐름의 activation은 2개이고 proposal-only 결과를 확인했다. 기존 minimal install CI가 같은 module을 실행한다.
현재 P3-03 변경은 production 약 160줄, 총 약 910줄이다. 원래 총량 상한800보다 약14% 늘어난 부분은
실제 policy 회귀, nested/barrier/committed release 검증과 실행 기록이다. 누적 구현 diff는 약3,550줄로
기능2,000줄 soft limit·총비용1.5배 재계획 기준 이내다. 범용 추상화와 새 runner는 추가하지 않았다.
