# v0.25 통합 개발자 경험: API 설계와 후속 방향

작성일: 2026-10-25. 코드 기준: v0.24.0, `8d56e0df790a0ca518295545a0439b214d7b2d80`.

상태: **기존 계획에 통합 / 2026-10-27 P5·DX1 병합, P3 계획 구체화**. 대상 사용자는 제품에 에이전트를 통합하는 개발자다.
사용자의 통합 요청에 따라 DX1·DX2를 [v0.25 통합 계획](V0_25_PRE_INVESTIGATION.md)에 포함했다.
P1–P5 필수, P8 조건부, P6/P9 독립 유지보수, P7 후속이라는 기존 경계는 유지한다.
범위의 기준은 통합 계획이고 첫 작업 지시는 [P5·DX1 PR 계획](V0_25_P5_DX1_PR_PLAN.md)이다.
P3 정책·변경 범위·검증 기준은 [P3 PR 계획](V0_25_P3_PR_PLAN.md)에 있다. P3 구현은 보류 상태다.
새 API 명칭과 후속 릴리스 배치는 설계 제안이다. DX1의 순수 typed 도구는 develop의
[실행 예제](../examples/custom_tool_quickstart.py)에 반영했다. 나머지 신규 API는 후속 구현 대상이다.

## 1. 제품 방향과 릴리스 경계

**v0.25의 목표를 “타입이 있는 도구를 연결하고, 실행을 멈추고, 관측한 결과를 설명할 수 있는 통합 계약”으로 구체화한다.**
실행·복구·권한에 투자한 현재 구조를 개발자가 작은 코드로 사용할 수 있게 만드는 것이 우선이다.

| 단계 | 개발자가 얻는 결과 | 범위 |
|---|---|---|
| v0.25 통합 범위 | 도메인 타입으로 도구 작성, 호스트 자원 연결, 취소·사용량·남은 대기 조회, 이를 보여주는 실행 예제 | 기존 P1–P5 + DX1 + DX2 |
| v0.26 우선 후보 | FastAPI 제품에 채팅·백그라운드 실행을 붙이는 지원 조합 | 기존 실행/hosting 계약을 조합하는 얇은 API, Vercel AI UI 어댑터 |
| 그다음 후보 | 실패 실행을 재현하고 회귀를 비교 | Studio 진단 화면, 정책을 적용한 재현 자료 내보내기, replay/conformance 연결 |
| 별도 연구 트랙 | 긴 작업에서 컨텍스트 품질·비용 제어, 실행 환경 강제 격리 | context compaction, 기존 P7의 OS/backend 격리 |

v0.26 이후 배치는 우선순위 제안이다. 날짜나 출시 확약을 두지 않는다. Studio의 기존 reader를 P1/P2에 맞추는
호환 작업은 v0.25에 포함하고, 새로운 진단 워크벤치 화면은 후속으로 분리한다.

## 2. 코드에서 확인한 출발점

| 관측 | 개발자에게 생기는 문제 | 선택한 대응 |
|---|---|---|
| `tools/decorator.py:137`의 `model_dump()`가 검증된 입력을 다시 dict로 바꿈 | `customer: Customer` 도구에 dict가 들어와 `customer.name`에서 실패 | DX1에서 handler 인자의 타입 보존 |
| 같은 파일의 결과 정규화가 BaseModel을 JSON으로 바꾸지 않음 | 타입 있는 반환값이 `tool_result_unportable`로 거부됨 | DX1에서 지원하는 반환 형태를 명시적으로 직렬화 |
| `ToolContext`는 method-only façade이고 공개 host resource 조회가 없음 | 소비자가 provider에 workspace를 저장하는 우회를 만듦 | 기존 P3의 typed key/resource/view 계약 |
| receipt·attempt·replay·durable invocation이 이미 존재 | 사용량 숫자만으로 미보고/0, 취소/drain 완료를 구별하기 어려움 | 기존 P1/P2의 사실 전달을 모든 소비 표면에 연결 |
| `RunStream`과 passive model subscription이 이미 별도로 존재 | 통합자가 연결 단절·Stop·재접속 정책을 직접 해석해야 함 | DX2에서 두 수명을 예제로 설명; 후속 어댑터에 보존 |
| `from_tools`, `from_config`, `AgentLoop.validate`, 구조화 출력, checkpoint/approval, replay가 이미 존재 | 새 편의 계층을 크게 만들면 같은 기능의 유지 지점이 늘어남 | 기존 진입점을 개선하고 실행 조합을 지원 |

실제 소비 사례는 CSP의 `pipeline_core/agents/modular_kernel.py`에서 확인했다. provider에 workspace를 보관하고
`workspace_factory`를 감싸는 경로가 P3의 필요성을 보여준다. 커널 예제와 CI는 커널 소유 도구로 같은 통합 요구를 검증한다.
CSP 저장소 의존성·핀·운영 기능의 변경은 소비 프로젝트의 별도 작업이다.

## 3. 우선순위와 변경 범위

아래 순서는 착수·검증 순서다. P1–P5의 필수 여부는 동일하며, 독립 작업군 사이의 전역 선행조건을 만들지 않는다.
S/M/L은 소스·호환성·테스트·문서·리뷰를 포함한 상대 크기다. L은 추가 PR 분할 가능성이 높은 작업이다.

| 착수 순위 | 작업 | 기존 계획과 관계 | 변경 경계 | 크기 | 완료를 판단하는 결과 |
|---|---|---|---|---|---|
| 1 | P5 권한 상실 뒤 출력 수정 | G4 필수, PR-01 | `model_call.py`의 일반·terminal·remainder 조각 전달과 경합 회귀 | S | 전달 경계에서 revoke를 확인하면 후속 출력·settlement 차단, 정상 문자 flush 보존 |
| 1 | DX1 typed `@tool` | G6 필수 추가, PR-02 | `tools/decorator.py`, decorator/loop 테스트, 예제 | S–M | 중첩 Pydantic 입력의 타입 보존·serializer 0회, BaseModel 결과가 기존 observation까지 전달됨 |
| 2 | P3 호스트 도구 자원 | G2 필수 유지 | 공개 key/binding/helper, bootstrap·call 수명·restore·child 경로, 좁은 workspace view | L | provider에 raw workspace를 보관하지 않고 도메인 도구가 동작; 종료 뒤 접근 거부 |
| 2 | P1/P2 종료·사용량 사실 | G1 필수 유지 | runner→receipt/attempt→codec/ledger→loop/hosting/observer/Studio/OTel | L | 취소·재시도·복구에도 최초 종료 요청과 보고된 수량의 의미 유지 |
| 2 | P4 종료 예산 snapshot | G3 필수 유지 | run/call/drain/sync wait의 시간 기준, 호출별 상태, 공개 조회 | M–L | 실제 제어와 snapshot이 같은 clock/cutoff 사용; 시계 변경·동시 호출 검증 |
| 각 작업과 함께 | DX2 통합 예제와 수용 시나리오 | G7 필수 추가; 기능별 문서·검증에 결합 | `examples/`, embedding 문서, 설치 패키지 smoke/conformance | M | 사용자가 공개 API만으로 도구→중단→복구 흐름을 따라감 |
| 조건 충족 시 | P8 자식 취소·drain | G5 조건부 유지 | opt-in 상속·전파·대기와 child usage 집계 | M–L | 세대별 전파·상한·중복 집계 회귀가 완결됨 |
| 독립 | P6/P9 | 유지보수 분리 유지 | optional-token 규칙, 실제 확인된 결함, CI 이미지/릴리스 작업 | 별도 산정 | 완결된 단위만 통합; 필수 작업의 선행조건으로 확대하지 않음 |

P3와 P1/P2는 병행 가능한 핵심 작업군이다. P4도 독립적으로 진행한다. DX2의 각 예제는 자신이 사용하는 계약만 기다린다.
P8은 준비되지 않으면 후속으로 넘긴다. **P3의 child/fork 자원 격리와 parent authority 검사는 P8 이관 여부와 관계없이 필수다.**

추가 비용의 중심은 DX2의 통합 검증이다. 이를 별도 SDK 전면 개편으로 확대하지 않는다. 기존 예제에 새 계약을 연결해
중복을 줄이고, PR을 나누더라도 위 작업군 전체의 테스트·마이그레이션·리뷰 비용을 누적 기록한다.

## 4. v0.25 API 예시: 타입 있는 도구와 자원 주입

다음은 **DX1 + P3 구현 후의 목표 사용법**이다. `HostToolKey`, `HostToolBinding`, `ToolWorkspace`,
`require_host_tool_binding`, `host_tool_bindings`가 신규 제안이다. `AgentLoop.from_tools`, `AgentRunSpec`,
`ToolContext`, `@tool`, `run_once`는 기존 API다. import 위치와 이름은 P3 구현 전 검토에서 확정한다.

```python
from pathlib import Path
from pydantic import BaseModel
from monoid_agent_kernel import AgentLoop, AgentRunSpec, ToolContext, tool
from monoid_agent_kernel.tools.host_bindings import (  # 제안 모듈
    HostToolBinding,
    HostToolKey,
    ToolWorkspace,
    require_host_tool_binding,
)

class ReportDraft(BaseModel):
    title: str
    body: str

class SavedReport(BaseModel):
    path: str
    sha256: str

# ToolWorkspace는 새 runtime-checkable Protocol로 설계한다.
# isinstance 검사를 지원하도록 새 Protocol에 @runtime_checkable을 선언한다.
REPORT_FILES = HostToolKey[ToolWorkspace](
    "report.files", accepts=lambda value: isinstance(value, ToolWorkspace)
)

@tool(id="report.save", capability="fs.write", side_effect="write")
def save_report(ctx: ToolContext, draft: ReportDraft) -> SavedReport:
    files = require_host_tool_binding(ctx, REPORT_FILES)
    path = "reports/draft.md"
    content = f"# {draft.title}\n\n{draft.body}\n".encode("utf-8")
    digest = files.write_bytes(path, content, create_dirs=True)
    return SavedReport(path=path, sha256=digest)

def build_report_loop(adapter, workspace: Path, run_root: Path):
    return AgentLoop.from_tools(
        AgentRunSpec(workspace_root=workspace, run_root=run_root, mode="propose"),
        adapter,
        [save_report],
        host_tool_bindings=(
            HostToolBinding(
                REPORT_FILES,
                open=lambda activation: activation.workspace_resource(),
            ),
        ),
    )

# adapter와 두 디렉터리는 호스트가 준비한다.
# result = build_report_loop(adapter, workspace, run_root).run_once(
#     "이번 주 작업을 정리해 보고서 초안을 저장해줘."
# )
```

예제는 파일 변경을 proposal workspace에 기록한다. 사용자 파일에 반영하는 승인/apply 단계는 호스트가 이어서 수행한다.
모델에게 전달되는 인자는 `draft`이고, host resource 객체는 모델 인자나 checkpoint에 직렬화하지 않는다.

도구별 경로 범위나 실행 전 승인이 필요한 제품은 기존 `from_config`와 `ToolBinding`을 사용한다.
`from_tools`가 별도의 승인 정책 인자를 이미 지원한다고 가정하지 않는다.

```python
from monoid_agent_kernel import AgentRuntimeConfig, ToolBinding, ToolScope
from monoid_agent_kernel.tool_loader import FunctionToolProvider

config = AgentRuntimeConfig(
    definition_id="report-agent",
    tools=(
        ToolBinding.for_tool(
            "report.save",
            scope=ToolScope(allowed_paths=("reports/**",)),
            authorization="ask",
        ),
    ),
)
loop = AgentLoop.from_config(
    spec, adapter, config,  # spec은 위와 같은 propose 모드 descriptor
    tool_providers=(FunctionToolProvider(lambda _ctx: (save_report,)),),
    host_tool_bindings=(
        HostToolBinding(REPORT_FILES, open=lambda a: a.workspace_resource()),
    ),
)
```

`authorization="ask"`는 해당 도구 실행 전 승인이다. 파일 proposal의 승인·apply는 생성된 변경 내용을 반영하는 별도 단계다.
승인 대기는 기존 suspension/control 경로로 구동하며, `run_once` 하나로 승인 왕복까지 완료한다고 설명하지 않는다.

### DX1의 좁은 수정 계약

- 입력 검증은 현재 Pydantic 모델을 재사용한다. 이미 수집한 필드 이름과 `getattr`로 검증된 값을 전달해 중첩 모델·enum 등 선언 타입을 보존한다.
  입력 단계에서 `model_dump()`나 serializer를 호출하지 않는다.
  모델 annotation을 선언하면서 dict 수신에 의존하던 도구에는 마이그레이션 안내를 제공한다. dict가 필요한 도구는 dict 타입을 선언하거나
  받은 모델을 명시적으로 `model_dump()`한다. 이 동작 변경을 호환성 기록에 남긴다.
- top-level BaseModel 반환은 `model_dump(mode="json")`으로 변환한 후 기존 portable JSON 검사를 통과시킨다.
  실제 dump 결과가 dict이면 `ToolResult.content`에, 그 외 portable 값이면 기존 scalar 규칙처럼 `{"result": value}`에 둔다.
  RootModel과 비-dict 결과를 만드는 custom model serializer를 함께 검증한다. 표준 exclude/SecretStr 설정은 존중한다.
- 기존 dict, 명시적 ToolResult, scalar 반환의 envelope를 유지한다. dict 안 임의 객체를 자동 직렬화하는 범위는 추가하지 않는다.
- sync/async handler, 중첩 입력, 기본값, invalid args, JSON serialization 실패를 확인한다. 반환 annotation에 따른 새로운
  강제 검증·출력 스키마 기능은 별도 요구로 평가한다.
- 직렬화 실패는 기존 `tool_result_unportable` 도구 실패로 분류하고 짧은 고정 진단을 제공한다. 원시 예외·모델 repr이나
  문자열 성공 fallback을 추가하지 않는다. 반환 annotation으로 실제 subclass의 필드가 자동 필터링된다고 설명하지 않는다.
- 기존 ToolExecutionError의 category/retryable과 실패 이벤트를 유지한다. Pydantic wrapper의 cause에 남은
  cancel/timeout/interrupt/authority 제어 예외를 일반 직렬화 실패로 바꾸지 않는다.
  decorator에서 원래 예외를 재발생한 뒤 loop는 기존 host/run 종료와 handler-local `tool_handler_cancelled` 관찰을 구분한다.

P3 없이 실행할 순수 도구의 목표 API와 상세 회귀는 [DX1 PR 명세](V0_25_P5_DX1_PR_PLAN.md#4-pr-02--dx1-typed-도구-입력과-결과)에 있다.

### P3의 자원 수명 계약

이 절의 계약을 구체화한 [P3 정책](V0_25_P3_PR_PLAN.md#4-채택-정책)을 구현 기준으로 사용한다.
동기 open/bind_call/close, call별 façade, kernel local-dispose와 host pool 소유권,
workspace write의 승인된 `side_effect="write"` 요건, 정상 종료와 writer revoke의 분리를 포함한다.

`HostToolKey[T]`는 프로세스 내 identity를 갖고 이름은 진단용이다. 같은 프로세스의 host와 handler가 같은 키 인스턴스를 공유한다.
중복 identity/name은 자원 생성 전에 거부한다. 기존 method-only `ToolContext`의 필수 멤버는 유지하고 선택적 확장 프로토콜과 helper를 제공한다.

`open(activation)`은 activation별 `HostToolResource[T]`를 반환한다. 자원의 `bind_call(call_lease) -> T`가 실제 handler 호출에
귀속된 view를 만든다. `workspace_resource()`는 이 자원 factory다. provider 등록 중에는 동작 가능한 view를 발급하지 않는다.
커널은 생성 성공한 자원의 closer를 즉시 등록하고, 부분 초기화 실패 시 역순 정리한다.

`ToolWorkspace`는 우선 `read_bytes`/`write_bytes`와 실제 소비에 필요한 최소 조회만 제공한다. 각 작업에서 원래 call lease,
activation guard, binding path scope, workspace authority를 검사한다. 보관한 view와 보관한 context의 새 helper 조회 모두
원래 호출 종료 후 거부한다. 다음 호출의 권한을 빌릴 수 없다.

root/child/fork는 별도 registry를 갖는다. 기본 상속은 OFF이고, 명시 상속 시 자식에서 factory를 다시 실행한다.
자식의 권한과 부모 lineage를 함께 검사한다. restore에서는 새 host 주입을 수행하고 복구 완료 후 첫 호출에서 workspace 내용을 읽는다.
factory가 복구 전 workspace 내용을 읽어 저장하지 않도록 API를 제한한다.

activation 해제 경로는 `close`/`discard_uncommitted`/`release_parked`와 각각의 async façade를 모두 포함한다.
특히 `ActivationDriver`가 정상 durable boundary마다 호출하는 `release_parked()`와 `arelease_parked()`의 성공한 해제에서
P3 guard를 무효화하고 소유 resource를 역순으로 한 번씩 정리한다. checkpoint·hosted task·run의 재개 가능성은 유지한다.
commit되지 않은 park여서 해제를 거부할 때는 현재 resource와 guard를 유지한다. cleanup 실패나 async 취소로 discard fallback이
이어져도 closer 호출을 중복하지 않도록 처리 상태를 공유하며, 남은 자원 정리를 계속하고 원래 오류를 보존한다.
authority loss는 즉시 view를 무효화한다. 이때 cleanup은 외부 쓰기 없는 local 정리로 제한하고 외부 flush/commit은 fenced host에 맡긴다.

일반 DB/API client에는 호스트가 같은 guard 계약을 적용한 resource adapter를 작성한다. v0.25의 보장 범위는 registry 수명과
커널이 제공하는 guarded workspace view다. 이미 시작한 외부 효과의 취소·중복 방지는 외부 시스템과 idempotency/outbox 계약으로 다룬다.

## 5. v0.25 API 예시: 종료와 사용량의 사실

`ModelCallReceipt.termination`, `ModelCallReceipt.usage_evidence` 및 attempt의 증거 필드는 **신규 제안**이다.
기존 `usage`, `stop_reason`, `attempts`, `attempt_log`를 유지한다. 기존 attempt log를 확장해 시도별 증거를 보존한다.

| 제안 표면 | 의미 |
|---|---|
| `termination.outcome` | 기존 stream outcome과 같은 completed/interrupted/failed/cancelled/timed_out 분류 |
| `termination.requested_by` | 최초 수락한 종료 요청; 요청이 없으면 null |
| `termination.ended_by` | provider terminal/deadline/drain 만료/lease loss 등 실제 읽기·대기 종료 경계 |
| `usage_evidence.reported` | 관측된 counter만 포함하는 sparse mapping; 명시적 0 보존; 재시도는 관측된 부분의 합 |
| `usage_evidence.completeness[counter]` | 해당 counter의 complete/partial/unknown/no_dispatch; counter별로 판단 |
| attempt별 source/collection | provider 또는 adapter 보고/legacy_ambiguous, normal 또는 drain 수집; 복구 재전달에서 최초 출처 유지 |

`complete`는 커널이 관측한 adapter dispatch의 보고 범위다. provider 내부 재시도가 숨긴 작업량이나 최종 청구액까지 증명하지 않는다.
구형 record에는 새 사실이 없다. 새 필드를 null로 읽고 legacy ambiguity를 유지하며, 예전 `usage=0`에서 보고 여부를 추론하지 않는다.
새 adapter가 sparse 원본을 제공하지 못하는 경우도 `legacy_ambiguous`로 드러낸다.

```python
from monoid_agent_kernel import CapturePolicy, ModelIOSubscription

class ModelDiagnostics:
    def on_model_call(self, capture):
        receipt = capture.receipt
        facts = receipt.termination       # 제안; 구형 record이면 None
        evidence = receipt.usage_evidence # 제안; 구형 record이면 None
        print({
            "outcome": facts.outcome if facts else None,
            "requested_by": facts.requested_by if facts else None,
            "ended_by": facts.ended_by if facts else None,
            "reported_output_tokens": (
                evidence.reported.get("output_tokens") if evidence else None
            ),
            "output_completeness": (
                evidence.completeness.get("output_tokens", "unknown")
                if evidence else "unknown"
            ),
        })

# 기존 AgentLoop 생성의 선택 인자에 넣는다.
diagnostics = ModelIOSubscription(ModelDiagnostics(), CapturePolicy(mode="none"))
# AgentLoop.from_config(..., model_io_subscriptions=(diagnostics,))
```

이 observer는 진단용이다. observer 오류는 현재 커널에서 삼켜지므로, 청구 장부의 유일한 입력으로 사용하지 않는다.
호스트의 확정 집계는 canonical durable record와 안정적인 invocation identity를 기준으로 중복을 제거한다.
새 metadata를 별도 token 가산 경로로 만들지 않는다. 호스트 가격·정산·환불 정책은 이 사실을 소비한다.

| 상황 | 기대하는 공개 사실 |
|---|---|
| dispatch 전 취소 | attempts=0, no_dispatch, provider usage 생성 없음 |
| provider가 output_tokens=0을 보고 | reported에 0 존재; 해당 dispatch의 counter 관측됨 |
| provider가 output_tokens를 생략 | reported에 키 없음; unknown 또는 누적 partial |
| 첫 시도 사용량 관측, 다음 시도 미보고 | 확보한 합계 보존, completeness=partial |
| 사용자 취소 후 drain에서 terminal 회수 | outcome=cancelled, requested_by=user_cancel, ended_by=provider_terminal |
| 사용자 취소 후 deadline 도달 | 취소 outcome 유지, ended_by=deadline |
| lease loss | 새 출력·canonical settlement 금지; 새 소유자의 복구 경로 적용 |
| durable dispatch 결과가 unknown | 자동 유료 재시도 금지; receipt/result 합성 금지 |
| settled 결과 복구·재전달 | 최초 증거 보존, 이미 반영한 usage 재합산 금지 |

P1/P2는 같은 영속 schema 변경으로 설계한다. dual-reader 준비 → 새 writer 활성화 → 혼합 버전 및 rollback 검증 순서다.
rollback도 새 record를 읽는 reader를 유지하거나 검증된 변환을 거쳐야 한다. 구형 reader의 새 format 수용을 가정하지 않는다.
receipt/attempt, 예외, runner·ValidatedCallRunner, gateway 오류, durable codec·ledger, loop 집계, live closed frame,
Studio reader, OTel의 같은 사실을 대조한다. replay request identity와 provider turn usage의 의미도 유지한다.

기존 계획처럼 unknown dispatch의 부분 증거를 위한 새 영속 slot은 필수 범위에서 제외한다. 프로세스 내 예외에서 본 증거가
재시작 후 사라질 수 있음을 명시한다. 새 API는 이 한계를 가리는 완전성 값을 만들지 않는다.

## 6. v0.25 API 예시: Stop 이후 현재 대기 조회

`CancellationToken.cancel`은 기존 API이고, `loop.termination_budget()`은 **신규 제안**이다.

```python
def on_stop(loop, cancellation_token):
    cancellation_token.cancel()  # 기존 user_cancel 요청
    snapshot = loop.termination_budget()  # 제안: 읽기 전용 snapshot
    return [
        {
            "call_id": call.call_id,
            "phase": call.phase,
            "deadline_remaining_s": call.deadline_remaining_s,
            "drain_remaining_s": call.drain_remaining_s,
            "effective_grace_s": call.effective_grace_s,
        }
        for call in snapshot.calls
    ]
```

snapshot에는 관측 시각, 진행 중 call 식별자와 streaming/draining/closing phase, 현재 유효 deadline,
호출에 고정된 drain budget, 남은 drain, 적용 grace를 둔다. drain 전에는 `drain_remaining_s=None`,
deadline 미설정에는 `deadline_remaining_s=None`, active call이 없으면 빈 calls다. 동시 조회는 일관된 복사본을 반환한다.

조회값은 호스트가 Stop 상태를 설명하는 샘플이다. 다음 retry·미래 자식 호출의 예산을 현재 값에 섞지 않는다.
동시 대기는 각 호출의 상한을 정책 한도 내에서 조합하며 단순 합산하지 않는다. provider 프로세스 종료와 최종 청구 시점은 별도다.

내부 제어도 monotonic cutoff로 정렬한다. 공개 epoch deadline과 저장 시각은 유지하고 진입 경계에서 변환한다.
run guard, model call, drain, sync wait, RunStream 종료 대기가 같은 제어 기준을 사용해야 한다.
관측용 monotonic timestamp 하나만 붙이는 변경으로 이 보장을 충족하지 않는다.

## 7. DX2: 실제 통합 흐름을 릴리스의 증거로 삼기

새 통합 프레임워크를 만들기 전에 기존 예제를 다음 세 경로로 정리한다.

| 예제 | 사용하는 계약 | 수용 시나리오 |
|---|---|---|
| Python 제품의 도구 | 순수 도구는 from_tools+DX1; 자원 사용 보고서 도구는 DX1+P3+proposal | 순수 typed 도구를 PR-02에 동반; 이후 reports 경로 제한·종료 뒤 자원 접근 거부까지 확장 |
| 요청에 연결된 스트리밍 | aopen → astream, P1/P2/P4/P5 | 사용자의 Stop, 연결 단절 취소, drain 동안 UI 상태와 실제 종료 사실 |
| 승인·재시작이 있는 작업 | 기존 hosting/checkpoint/control, P3 재주입 | 승인 대기로 park, 프로세스 재구성, 승인 응답, 재개, 확정 결과·사용량 중복 반영 방지 |

커널 소유 `report.save`/가짜 외부 서비스와 FakeModelAdapter로 검증한다. 승인 도구 실행과 proposal approve/apply를 각각 검증한다.
호스트 승인 데이터는 task/call 또는 proposal identity에 연결하며 중복·다른 run·다른 tenant 응답을 확인한다.
기존 hosting owner/lease/watchdog와 ActivationDriver를 재사용한다. 단순 checkpoint 저장만으로 운영 복구가 완성됐다고 설명하지 않는다.

개발 초기에 P3의 첫 수직 경로를 붙인다. 남은 수명·복구·child 검증이 완료되기 전에는 공개 안정성 완료로 판정하지 않는다.
예제는 checkout의 `sys.path` 조작 없이 설치된 wheel로도 실행한다. dependency extra별 import와 공개 표면을 검증한다.
핵심 공개 API와 `reference.RunnerBackend`를 쓰는 호스트 예제를 문서에서 구분한다. DX2는 reference 전체의 안정성 등급을 바꾸지 않는다.

개선 측정은 초기 baseline을 먼저 수집한다: 설치 후 첫 도구 실행까지 필요한 단계, 호스트가 작성하는 glue 코드,
승인 후 재시작 재개 성공 여부, unknown 사용량을 0으로 표시하는 사례, 공개 API 밖 접근 수.
목표 숫자는 측정 뒤 정한다. v0.25의 직접적인 완료 기준은 위 세 흐름을 공개 계약으로 재현하는 것이다.

## 8. 후속 API 방향: 웹과 백그라운드 실행

다음 단계에서는 지원할 최소 hosting 조합과 FastAPI/Vercel AI UI 어댑터를 만든다. `AgentLoop`, `LoopSession`,
ActivationDriver, checkpoint/control 경로를 조합하고 기존 reference 전체를 안정 API로 승격하는 일은 피한다.
추가 의존성은 optional integration 계층에 둔다. 인증·tenant 선택·가격·queue 운영은 제품이 소유한다.

**후속 릴리스의 사용감 예시**이며, 다음 이름과 request 변환은 v0.25 범위에 포함하지 않는다.

```python
from fastapi import FastAPI, Request
from monoid_agent_kernel.integrations.fastapi import VercelAIAdapter  # 후속 제안

app = FastAPI()

@app.post("/api/chat")
async def chat(request: Request):
    principal = await authenticate(request)  # 제품의 인증 함수
    loop = create_loop_for(principal)        # 제품의 자원·권한 구성
    return await VercelAIAdapter.dispatch_request(
        request, loop=loop, disconnect="cancel"
    )
```

이 편의 경로가 책임질 범위는 UIMessage 요청 검증, 허용된 사용자 입력 변환, loop open/stream/close,
표준 SSE part와 terminal/suspension 전달, 연결 단절 시 bounded cancellation이다. 호스트는 conversation/run의 소유권과
저장된 이력을 제공하며, 클라이언트가 보낸 tool 결과·approval·전체 이력을 그대로 실행 사실로 신뢰하지 않는다.
승인 대기 시 terminal close와 park/release 수명을 구별해야 한다. 코드 예시는 초기 채팅용이며 장기 작업 재접속까지 약속하지 않는다.

| 실행 모델 | 연결이 끊어질 때 | 통합 기반 |
|---|---|---|
| 요청에 연결된 채팅 | 실행에 cancel 요청, 현재 drain/grace 안에서 종료 처리 | 기존 RunStream의 수명 + 위 편의 API |
| 장기 실행의 화면 구독 | 해당 구독만 종료, 작업은 계속 | 기존 `RunnerBackend.subscribe_events` / `subscribe_model_stream` + 후속 변환기 |

passive 경로에는 실행 시작/취소/승인 API가 별도로 존재한다. 재접속은 durable event cursor와 bounded live-content cursor의
차이를 유지한다. 현재 live ring에는 gap/reset이 있으므로 전체 텍스트 복원을 보장하려면 별도 저장·권한 정책을 결정해야 한다.
후속 변환기의 preflight는 run lifecycle/result/approval 이벤트와 model delta를 함께 다루는 입력 계약, 상관관계, cursor/reset,
중복 part 및 terminal 순서를 먼저 검증한다. model delta만 변환하는 구현으로 전체 UI protocol 지원을 선언하지 않는다.

공식 UI protocol fixture와 실제 AI SDK client로 텍스트·tool·error·finish·승인 대기·재접속을 검증하고 지원 SDK 버전을 명시한다.
별도의 scheduler/store/FSM, 전용 채팅 프런트엔드, 전역 middleware 시스템은 이 단계의 기본 범위에 넣지 않는다.

## 9. 외부 사례에서 가져올 것과 현재 설계의 선택

확인일: 2026-10-25. 아래는 공식 문서의 현재 계약을 참고한 설계 판단이다.
추가로 조사한 실제 이슈·고정 revision 구현과 P5/DX1에 채택할 해법은 [PR 계획 §2](V0_25_P5_DX1_PR_PLAN.md)에 있다.
입력 serializer 부작용, 취소 뒤 다음 턴 오염, abort 뒤 usage 미보고를 제품의 회귀 조건과 연결했다.

| 사례 | 확인한 점 | 이 프로젝트에 적용할 판단 |
|---|---|---|
| [Pydantic AI dependencies](https://pydantic.dev/docs/ai/core-concepts/dependencies/) | typed RunContext와 deps로 호스트 의존성을 연결 | 타입 있는 주입 경험을 채택; 기존 method-only ToolContext와 호출별 권한 수명을 보존하는 P3 방식 사용 |
| [Pydantic AI UI streams](https://pydantic.dev/docs/ai/integrations/ui/overview/) | 요청 안 실행과 외부 worker 이벤트 변환을 분리하며 run result도 전달 | request-bound 실행과 passive 구독을 서로 다른 어댑터 경로로 설계 |
| [Vercel stream protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol) | Python 등 custom backend, SSE data stream, v1 protocol header 지원 | 자체 프런트엔드 규약 추가보다 표준 호환 어댑터와 fixture를 우선 |
| [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) | checkpoint와 resume, 재실행되는 노드의 side effect idempotency를 명시 | 승인·복구 예제에 외부 효과 재실행과 중복 응답을 포함 |

기존 코드 재사용은 호환 비용이 가장 작다. 외부 표준은 UI 전송에서 효과적이다. custom 구현은 P3의 호출별 revoke,
P1/P2의 durable 증거처럼 현재 프로젝트의 보장에 필요한 범위로 한정한다. 외부 라이브러리의 API 모양을 그대로 옮기는 데에는
현재 checkpoint·tool authority·receipt 계약과의 변환 비용이 따른다.

## 10. 검증 계획, 증거와 남은 결정

구현 전에 각 작업군의 최신 base를 조사하고 자체 preflight를 수행한다. G1의 reader/writer/집계, G2의 registry/view/복구,
G3의 clock/snapshot처럼 유효한 중간 상태가 있는 경계에서 나눈다. 기존 campaign/review 절차를 적용하고 PR 수를 미리 고정하지 않는다.

| 검증 층 | 필요한 확인 |
|---|---|
| 단위/계약 | DX1 typed args/results와 invalid 값; P3 retained view/context, 부분 초기화 cleanup, parent/child/fork·restore 격리 |
| activation 수명 | park-release→restore에서 resource closer 횟수·이전 guard 종료·새 resource 생성; 미commit park 거부 시 유지; cleanup 실패/async 취소→discard에서 중복 close 방지 |
| 제어 경합 | P5 drain 전/중 일반·terminal·remainder 전달, drain ON/OFF·정상 surrogate flush·callback 중 revoke; P4 wall-clock jump·설정 변경·병행 호출 |
| 전달/복구 | P1/P2 pre-dispatch/zero/unknown/retry/drain, 예외·receipt·ledger·live/Studio·OTel 의미 일치, no double count |
| 호환/패키지 | 구형 fixture 읽기, 새 writer round trip, 혼합 버전·rollback, public exports, 최소 설치와 wheel 예제 |
| 서비스 통합 | 변경된 durable/service 경로의 profile, 최종 통합 head에서 owner loss·unknown dispatch·settled recovery·승인 재개 |

테스트는 소유 층의 가장 작은 유효 범위를 선택한다. 사용자 흐름 자체가 계약인 DX2만 통합 시나리오로 확인한다.
실행 prefix는 `pytest -q -n auto --dist=worksteal --timeout=120 --timeout-method=thread`를 사용한다.

이번 대화의 초기 코드 조사에서 다음을 실행했다. 관련 코드 HEAD가 동일하므로 설계 구체화에서는 그 증거를 재사용했다.

```powershell
& ./.venv/Scripts/python.exe examples/minimal_quickstart.py
& ./.venv/Scripts/python.exe examples/custom_tool_quickstart.py
& ./.venv/Scripts/python.exe examples/embedding_local_product.py
& ./.venv/Scripts/python.exe examples/embedding_hosted_product.py
& ./.venv/Scripts/python.exe -m pytest -q -n auto --dist=worksteal --timeout=120 --timeout-method=thread tests/test_tool_decorator.py tests/test_output_schema_delivery.py tests/test_model_stream_observer.py tests/test_output_validator.py tests/test_public_surface.py
```

4개 offline 예제 성공. targeted pytest는 **146 passed, 5 skipped (20.34s)**였으며 skip은 optional OpenAI 의존성 부재다.
typed `Customer` 입력 probe는 실제 AgentLoop 경로에서 dict가 전달되어 AttributeError/internal_error가 발생했다.
BaseModel 반환 probe는 tool observation이 `tool_result_unportable`였다. 반환 probe의 run은 scripted final로 완료되었으므로
도구 실패와 run 실패를 구분한다. P5의 10/25 확장 재현 8개와 추가 기존 회귀 76 passed는 [PR 계획 §7](V0_25_P5_DX1_PR_PLAN.md)에 있다.

전체 suite, 실제 유료 모델, 운영 서비스, 브라우저, 신규 API 실행 검증은 수행하지 않았다. 현재 baseline의 성공을
새 설계 구현의 성공으로 해석하지 않는다. Python 예시 5개는 AST 문법 검사를 통과했다. import나 코드 실행을 수행한 검증은 아니다.
문서의 로컬 링크 대상을 확인했다. 기존 계약 정합성과 독립 설계 리뷰도 이 문서 단계의 확인 대상이다.

남은 설계 결정은 작업군별 preflight에서 좁힌다: P3 공개 export와 host adapter 최소 프로토콜, G1 필드/codec 버전,
G3 조회 명칭과 clock 변환, 후속 web adapter의 request/history 및 cursor 계약. DX1 JSON 변환 실패는 기존 `tool_result_unportable`로 정했다.
이 결정들은 설계 명칭·구현 선택이다. P1–P5 필수 범위나 사용자의 승인된 동작 보장을 줄이는 변경은 별도 범위 결정으로 남긴다.

## 11. 설계 검토 기록

이 절은 통합 전 API 제안 버전의 검토 기록이다. 10/25 범위 통합·P5 경계 보완·DX1 상세 계약은
[PR 계획의 통합 설계 검토 기록](V0_25_P5_DX1_PR_PLAN.md#8-통합-설계-검토-기록)에서 별도로 확인한다.

Preflight/Review 스킬에 따라 초안 작성에 참여하지 않은 reviewer가 read-only로 검토했다. 시작 증거는 요청, 기존 범위·소스,
이 문서와 기록된 baseline이며 이전 검토 결론을 전달하지 않았다. 코드 base는 `8d56e0d`, 초안 SHA256은
`6B64EC63C9F8E5E9E4E8CD625929B8B6E96C88F298C9BA31456D12AA81165B00`이다.

- P2 설계 지적: 정상 activation 해제인 `release_parked()`를 P3의 수명 계약에 명시해야 한다. `hosting/activation.py:798`에서
  이 경로를 호출하며, `loop.py:2650`은 close/discard를 거치지 않고 activation을 해제한다. 이를 빠뜨리면 새 resource의 closer가 누락될 수 있다.
- 반영: §4와 §10에 sync/async park release, 미commit 경계 거부, cleanup 실패 후 discard, guard와 closer의 수용 기준을 추가했다.
  P3 내부의 보완이며 작업군을 늘리지 않는다. typed 입력 변경의 마이그레이션과 reference 호스트의 안정성 경계도 명시했다.
- 재검토: 보완본 SHA256 `5023EFDD2371BEC45CC3377DE206C67B875B0239DE1166D91E4B43B97F7AEA00`에 대한 targeted 검토는
  **ready**였다. 기존 P2 지적 해소, 새 중대 문제 없음. 이전 breadth 검토의 범위·typed 도구·사용량·Stop·stream 및 외부 사례 확인은 유지한다.
  해당 candidate 뒤의 최초 변경은 이 검토 결과 기록이었다. 현재 문서는 위에 명시한 통합 갱신을 포함한다.

이 지적은 새 설계를 그대로 구현할 때의 반례다. 현재 v0.24에 HostToolResource 누수가 있다는 주장이 아니다.
검토는 설계에 한정하며 구현·출시 판정을 대신하지 않는다.
