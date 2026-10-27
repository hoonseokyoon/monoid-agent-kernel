# v0.25 P1·P2: 종료 사실·사용량 관측·저장·복구 PR 계획

조사일: 2026-10-27 KST. 기준: fetch한 `origin/develop=4e14618b3ba68ffd9e243a9bcad2e3c9d2f5ac48`.
P5·DX1·P3가 병합된 상태다. 계획은 로컬 `94ea35a`에 기록했다.
**2026-10-27 사용자가 P1P2-00·01 구현과 리뷰·CI 후 02·03·04 순차 진행을 승인했다.**
00은 develop 대상 독립 PR, 01은 수정 전 실패 회귀부터 시작한다. 릴리스·배포·CSP 변경은 별도다.
구현 상태: 00 #156은 develop에, 01–04 #157–160은 통합 브랜치에 병합했다.
실제 공개 API와 실행 예제를 제공하며, 최종 누적 검증·수정 기록은 §9.6에 둔다.

[v0.25 통합 계획](V0_25_PRE_INVESTIGATION.md#51-p1p2-종료-사실과-사용량-증거)의 G1을 구체화한다.
기존 [API 제안](V0_25_DEVELOPER_EXPERIENCE_PROPOSAL.ko.md)의 P1·P2 예시는 아래 계약을 따른다.
상위 저장소의 `docs/operations/campaign-workflow.md`, `docs/operations/campaign-preflight-review.md`를 운영 권위로 사용한다.

## 1. 전달할 결과와 책임

통합 개발자는 한 호출에서 **어떤 결과가 났는지, 어떤 중단 요청을 수락했는지, 무엇이 읽기를 끝냈는지,
어느 token 항목이 실제 보고됐는지**를 따로 읽을 수 있다. 재시도·취소·복구를 거쳐도 이 의미를 유지한다.
정수 `usage`, provider `stop_reason`, 기존 오류 분류와 회계 경로를 보존한다.

| 역할 | 책임 |
|---|---|
| root | 코드·실제 소비자 대조, probe, 정책·PR 계획 작성, 공유 파일·검증 runner 소유 |
| p1p2_storage_research | closed schema, 해시·identity, 복구·projection·구형 reader 조사; 읽기 전용 |
| p1p2_external_research | 공식 문서와 병합된 외부 구현 조사; 미병합 제안 구분 |
| 독립 계획 reviewer | 작성·조사에 참여하지 않은 fresh reviewer가 현재 소스·계획·증거 검토; §9에 기록 |
| 향후 구현 담당 | PR마다 최신 실제 base에서 조사 갱신, 실패 회귀 고정, 독립 사전 검토·코드 리뷰·CI 수행 |

종료 우선순위·유료 호출 복구·저장 형식 변경은 A 등급이다. 계획 리뷰는 구현 최종 head의 리뷰를 대체하지 않는다.
P4 예산 API, 토큰 추정·가격·청구 정책, provider SDK retry 정책 변경, 새 provider 도입은 별도 범위다.
P5의 출력 권한 검사와 P3의 자원 수명·복구 보장을 유지한다.

## 2. 최신 코드에서 확인한 문제

| 생산자 → 소비자 | 현재 사실과 위치 | 이번 수정 경계 |
|---|---|---|
| provider 원본 → 정규화 | [providers/_common.py](../src/monoid_agent_kernel/providers/_common.py) `normalize_usage`: 누락된 기본 항목을 0으로 채우고 optional 0은 생략 | OpenAI·gateway 최초 수신 경계에서 sparse presence를 함께 생성. 정규화 이후 숫자로 출처 추론 금지 |
| terminal → drain | [model_call.py](../src/monoid_agent_kernel/model_call.py) `_reports_usage`: `any(usage.values())`; 명시적 all-zero도 미보고 취급 | valid report 존재와 값 크기 분리. 동일 attempt의 누적 snapshot 합산 금지 |
| cancel → 실제 읽기 종료 | `_AbortDrain`, `_await_draining_stream`: 최초 stop은 보존하지만 deadline/drain 만료가 같은 abort로 나감 | 사실을 아는 경계에서 종료 원인 기록. 기존 취소·lease 우선순위 유지 |
| 실패 receipt → stream observer | [loop.py](../src/monoid_agent_kernel/loop.py) `_acall_model`: 실패·취소 분기에서 `outcome_usage`를 채우지 않음 | 회수한 usage와 사실을 실패 observer까지 전달. 권한 상실 후 publish는 계속 차단 |
| attempt → receipt | [core/model_io.py](../src/monoid_agent_kernel/core/model_io.py): attempt 합계와 receipt.usage 일치 검증, 구형 empty attempt log 허용 | metadata와 회계를 분리. 새 optional attempt 필드는 기존 required-wire-key 계산에서도 제외 |
| receipt → durable invocation | [core/model_invocation.py](../src/monoid_agent_kernel/core/model_invocation.py): receipt·usage 필드 닫힘, attempt_log 의도적 생략, unknown에는 receipt/result 금지 | 명시적 v2 및 안전한 관측 projection. unknown 제약 유지 |
| settled 복구 → fenced recommit | [hosting/model_calls.py](../src/monoid_agent_kernel/hosting/model_calls.py) `recover`는 기존 settled 좌표를 다시 commit. [postgres/sink.py](../src/monoid_agent_kernel/adapters/postgres/sink.py)는 기존 payload hash와 새 `to_json()` hash 비교 | 구형 immutable 좌표 재게시의 원래 payload identity 보존. 단순 v1→v2 재직렬화 금지 |
| invocation → checkpoint → host | [core/checkpoint.py](../src/monoid_agent_kernel/core/checkpoint.py)는 nested invocation 검사. SQLite·PostgreSQL·activation receipt가 이를 소비 | standalone decode 외에 중첩 checkpoint·동일 명령 복구까지 검증 |
| receipt → ledger / content / live | model-calls와 model-content는 closed artifact, Studio live v1은 optional 추가를 허용 | 각 형식의 전달 정책을 §5에서 분리하고 reader·schema·fixture 동반 변경 |

### 2.1 제한된 실행 확인

동일 base, Windows, Python 3.11.9/Pydantic 2.13.4에서 tracked 소스·테스트를 수정하지 않고 실행했다.
로컬 ignored `.tmp/v025/p1p2_preflight_probe.py`와 `p1p2-preflight-probes.json`에 원본을 보관한다.
probe는 기존 test helper로 실제 `AgentLoop.run_until_suspended`를 실행하고 private state는 관찰에만 사용한다.

| 입력·경로 | 관측값 |
|---|---|
| usage 없음 / 기본 항목 모두 명시적 0 | 둘 다 `{input_tokens:0, output_tokens:0, total_tokens:0}` |
| `reasoning_tokens=0` | 숫자 projection에서 reasoning 키 소실 |
| `_streamed_usage([TurnComplete(usage={output_tokens:0})])` | `{}` |
| user cancel → bounded drain → 6/3/9 terminal | state usage=6/3/9, observer status=cancelled, observer usage=null, 전달 text는 `partial`만 |
| 유효 v1 settled receipt에 termination 추가 | 기존 payload `loaded`, 새 필드 추가 payload `corrupt` |

probe 첫 실행은 잘못된 `result_ref="result_1"` fixture로 실패했다. 기존 주소 형식 `blob:turn_1`로 고친 뒤 위 결과를 얻었다.
기존 관련 회귀는 **280 passed in 9.50s**다. 이 성공은 새 계약을 검증한 결과가 아니다.
PostgreSQL 해시 충돌 위험은 소스에서 도출했다. 이번 계획 단계에서는 실제 DB·ObjectStore·Temporal 이행 실험을 실행하지 않았다.

## 3. 외부에서 확인한 문제와 적용

확인일은 2026-10-27이다. 병합일은 UTC이며, 병합 사실과 배포된 패키지 버전은 구분한다.

| 외부 문제·채택 근거 | 확인한 답안 | 우리 제품의 적용·한계 |
|---|---|---|
| Vercel [#17500](https://github.com/vercel/ai/issues/17500): EOF·fatal failure가 같은 종료 callback으로 보임. [#17578](https://github.com/vercel/ai/pull/17578), 2026-08-26 merged | owner가 operation outcome을 선언. [타입](https://github.com/vercel/ai/blob/957146cf244c1543f78a527ae69f1f95648f87d9/packages/ai/src/ui-message-stream/ui-message-stream-outcome.ts)은 provider finish reason과 분리 | runner가 사실 생성; UI에서 EOF·text 유무로 성공 추론 금지. Vercel의 UI 오류 우선순위는 복사하지 않음 |
| PydanticAI [#9491](https://github.com/pydantic/pydantic-ai/pull/9491), 2026-10-20 merged: 누락 usage가 zero cost로 보임 | [usage 구현](https://github.com/pydantic/pydantic-ai/blob/1b10e6dcdc8101e86ac3279221b7f3c225a0f4c4/pydantic_ai_slim/pydantic_ai/usage.py)은 unmeasured 요청을 보존하고 명시적 0 구별 | provider ingress presence 보존. Monoid는 counter별 completeness가 필요. 가격·경고 정책은 host 책임 |
| PydanticAI [#10069](https://github.com/pydantic/pydantic-ai/pull/10069), [#10078](https://github.com/pydantic/pydantic-ai/pull/10078), [#11068](https://github.com/pydantic/pydantic-ai/pull/11068), 10월 15·16·26일 merged | 중단 ledger 누락, 재개 이중 계산, BaseException에서 partial usage 소실을 각각 수정. [재개 계산](https://github.com/pydantic/pydantic-ai/blob/70b75eb3923f6ab79cdeb232fa0f191bd847016f/pydantic_ai_slim/pydantic_ai/models/_continuation.py#L245)은 기존 기여를 제외 | 예외·observer·durable의 동일 사실 전달, 기존 `_usage_delta` 재사용. background continuation 기능은 도입하지 않음 |
| [OpenAI stream 계약](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events), [Anthropic stream 계약](https://platform.claude.com/docs/en/build-with-claude/streaming) | OpenAI는 choice 종료 뒤 usage-only chunk 가능, 중단 시 미수신 가능. Anthropic usage delta는 누적값 | 첫 stop_reason에서 읽기 종료 금지. 누적 snapshot과 별도 dispatch 합산 구분. 현재 직접 adapter는 OpenAI·gateway이므로 Anthropic adapter 추가는 범위 밖 |
| [Anthropic SDK retry 기본값](https://github.com/anthropics/anthropic-sdk-python/blob/202db72c87d7c6eaa72129c67ce67fd343a3a20c/src/anthropic/_constants.py#L11), [Vercel consumeStream](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-message-persistence#handling-client-disconnects) | SDK 내부 시도와 adapter 호출 수가 다를 수 있고 backend 소비는 UI 전달 이후에도 지속 가능 | completeness 범위는 kernel adapter dispatch. 숨은 HTTP 시도·실제 청구 완전성 보장 없음. 기존 bounded drain·authority 유지 |
| OTel GenAI [#536](https://github.com/open-telemetry/semantic-conventions-genai/pull/536), [#602](https://github.com/open-telemetry/semantic-conventions-genai/pull/602), 10월 9·21일 merged | [Development 규약](https://github.com/open-telemetry/semantic-conventions-genai/blob/e19717d65f890955777b044bca2a3cd71ae8b498/docs/gen-ai/gen-ai-token-metrics.md)은 usage counter와 operation histogram을 분리; histogram은 usage 부재 시 0 가능 | canonical evidence와 exporter projection 분리. 계측의 0·추론 error를 provider 사실로 역변환 금지. 전면 semconv migration은 별도 |
| PydanticAI [#6685](https://github.com/pydantic/pydantic-ai/pull/6685), 2026-07-28 merged; LangGraph [#5106](https://github.com/langchain-ai/langgraph/pull/5106), 2025-06-16 merged | 전자는 serialization에서 usage 필드 소실을 수정. 후자는 옛 checkpoint 필드를 명시적으로 이행하고 [historical fixture](https://github.com/langchain-ai/langgraph/blob/858891aaa164ebdacf33b8f5e7f88b38b9132211/libs/langgraph/tests/test_checkpoint_migration.py#L1474)로 복구 검증 | closed v1 유지 + 명시적 v2 reader. permissive extras를 복사하지 않음. 새 reader의 옛 데이터 지원과 rollback은 각각 검증 |

Vercel #21778·#21780 및 PydanticAI #11496은 OPEN, Vercel #17499·#15495와 PydanticAI #6724는 미병합이다.
위 제안들을 채택된 해결책의 근거로 사용하지 않는다. OTel #602의 문서·reference 변경은 확인했으나 전체 SDK 실행 검증의 증거로 확대하지 않는다.

## 4. 채택할 런타임 정책

### 4.1 종료 사실의 소유자와 공개 모양

`core/model_io.py` 계열의 공개 dataclass `ModelCallTermination`을 추가하고 receipt와 `ModelStreamOutcome`에
optional `termination`으로 전달한다. 기존 positional 인자의 위치를 유지하고 신규 항목은 끝에 둔다.
구형 record·외부 객체에는 `None`을 허용한다. 예외 전달은 기존 stamp/carry helper를 확장하여 처리한다.
새 범용 종료 상태 머신이나 예외 계층은 만들지 않는다.

| 필드 | 선택한 의미·어휘 |
|---|---|
| `outcome` | 현재 `completed / interrupted / failed / cancelled / timed_out`와 같은 호출 결과 |
| `requested_by` | 처음 **수락한** 요청: `user_cancel / turn_interrupt / graceful_drain / deadline / host_shutdown / caller_cancel`, 또는 null |
| `ended_by` | 실제 read/wait 종료: `provider_terminal / provider_eof / provider_error / provider_cancelled / stop_request / deadline / drain_timeout / lease_lost / local_failure` |

`requested_by=deadline`은 cancellation token에 실제 deadline 요청을 수락한 경우에만 기록한다.
시간 경과만 관측했다면 `requested_by=None, ended_by=deadline`이다. provider가 자신의 await를 취소한 경우와
호출자의 task 취소를 구별한다. 구별할 근거가 없는 구형 오류에는 요청 사실을 합성하지 않는다.

`provider_terminal`은 유효한 terminal을 소비하고 provider 읽기가 정상 종료됐다는 뜻이다.
terminal 후 hang→deadline은 `deadline`, terminal 후 read error는 `provider_error`, terminal 없는 EOF는 `provider_eof`다.
단순 `stop_reason` 출현은 읽기의 종료가 아니다. one-shot 정상 반환은 `provider_terminal`이다.
입력 거부·내부 validation 등은 기존 error_code와 함께 `local_failure`를 사용한다.

| 순서 | 결과·사실 |
|---|---|
| dispatch 전 user cancel | cancelled / user_cancel / stop_request, attempts=0 |
| dispatch 전 시간 만료 | timed_out / null / deadline, attempts=0 |
| cancel → drain → terminal 정상 종료 | cancelled / user_cancel / provider_terminal |
| turn interrupt → drain → deadline | interrupted / turn_interrupt / deadline |
| cancel → drain budget 만료 | cancelled / user_cancel / drain_timeout |
| cancel → provider read error | 기존 취소 결과·최초 요청 보존, ended_by=provider_error |
| lease loss | 기존 authority 우선순위 유지, ended_by=lease_lost. stale owner의 새 observer/canonical 게시 차단 |
| 실패 attempt 후 backoff 중 cancel | call 종료 사실만 갱신. 마지막 attempt의 provider 실패 분류 유지 |
| provider settled 후 required evidence 전송 실패 | 이미 확정한 provider 사실 유지. evidence delivery 실패는 별도 상태 |

최초 요청을 읽는 runner, provider read, deadline/drain 경계가 각각 사실을 기록한다.
loop·Studio·OTel이 최종 예외 이름에서 원인을 재구성하지 않는다. 같은 tick의 경합은 기존 경계 검사 순서를 따른다.
deadline과 drain window가 같은 값이면 deadline을 기록한다. 시간 측정 체계 변경은 P4로 남긴다.

### 4.2 사용량: 숫자와 관측 범위

공개 `UsageObservation`은 한 kernel attempt의 `reported`, `source`, `collection`을 나타낸다.
`UsageEvidence`는 호출의 `reported`, `completeness`, `attempts`를 가진다.
`attempts`는 `index`와 해당 observation의 안전한 projection 목록이다. `ModelCallAttempt.usage_evidence`는 해당 observation,
`ModelCallReceipt.usage_evidence`와 stream outcome의 같은 이름은 **논리 호출 전체 evidence**다.
구체적인 import/export와 JSON helper는 P1P2-01에서 기존 `model_io` public surface 규칙에 맞춰 고정한다.

- `reported`: 직접 확인한 canonical counter만 담은 희소 정수 맵. bool·음수·비정수 거부, 명시적 0 보존.
  `total_tokens`를 input+output으로 생성하지 않는다. canonical alias 변환은 허용한다.
  기존 numeric normalizer가 만든 파생 audio 합계·zero-fill은 provider 직접 보고로 승격하지 않는다.
- `source`: `provider`, `adapter`, `legacy_ambiguous`, 또는 null. first-party 원본 수신·검증 경계만 provider를 부여한다.
  gateway는 신형 envelope의 증거를 전달한다. custom adapter의 명시적 관측은 adapter, 구형 정규화 숫자·stamp는 legacy_ambiguous다.
- `collection`: `normal`, `drain`, 또는 null. recovery는 최초 수집 경로를 바꾸지 않는다.
  report 부재는 빈 reported와 null source/collection. legacy 자료는 source=legacy_ambiguous, 확인할 수 없는 reported는 빈 맵이다.
- `attempts`는 runner가 실제 adapter에 진입한 호출만 센다. SDK 내부 HTTP retry 수는 포함한다고 보장하지 않는다.
  `provider_retried` 기존 힌트와 관측 범위 설명을 유지한다.
- 기존 `usage`는 회계 호환성을 위한 정수 projection이다. 확인된 항목 합계는 evidence.reported에 별도로 표시하되
  **회계는 기존 usage 경로 한 곳만** 사용한다. optional zero는 usage에 키가 없어도 기본값 0과 일치하면 된다.

지원 adapter의 `ModelTurn`/`TurnComplete` report는 해당 attempt 전체 snapshot이다.
같은 attempt에서 마지막 유효한 비어 있지 않은 sparse report가 이전 snapshot을 교체하며, `{output_tokens:0}`도 report다.
usage 없는 terminal은 앞서 받은 report를 지우지 않는다. 반복 terminal·exception stamp를 더하지 않는다.
향후 partial delta 프로토콜 adapter는 provider 내부에서 snapshot을 조립한 뒤 이 경계를 사용한다.
현재 OpenAI·gateway 경로에서 snapshot과 numeric projection의 선택을 함께 처리하고, 다른 dispatch의 값만 합산한다.

`completeness`는 지원하는 각 counter에 대해 다음 네 값이다. supported counter vocabulary는 기존 durable usage 목록을 재사용한다.

| 상태 | 계산 규칙 |
|---|---|
| `no_dispatch` | 실제 attempts=0. 빈 reported이며 usage를 관측했다고 표현하지 않음 |
| `complete` | 모든 kernel dispatch에 그 counter의 유효한 명시 보고가 있음. 모든 값이 0이어도 해당 |
| `partial` | 일부 dispatch만 그 counter를 명시 보고. 알려진 값만 합산 |
| `unknown` | 명시 보고가 없거나 구형 자료만 있음 |

input만 매번 보고하면 input만 complete다. 전체 비용·provider 내부 retry의 완전성을 뜻하지 않는다.
receipt에 attempts>0인데 구형 attempt log가 없으면 complete를 만들지 않는다.
새 attempt log와 aggregate evidence가 함께 있으면 index·dispatch 수·reported 합계·completeness 일치를 검사한다.
기존 receipt.usage와 attempt usage 합계 불변식도 계속 검사한다.

정상 입력 validation을 느슨하게 바꾸지 않는다. 기존 실패/refusal의 lenient usage 추출은 원래 실패를 유지하면서
유효한 알려진 counter만 보존한다. malformed 항목은 보고한 것으로 세지 않는다. 원본 payload·임의 key를 evidence에 저장하지 않는다.
구형 record의 `usage` 양수·0으로 새 `reported` 사실을 역산하지 않는다.

### 4.3 실제 개발자 사용 예시

다음은 구현된 공개 subscription API다. [실행 예제](../examples/model_call_facts.py)와
[저장·복구 예제](../examples/recover_model_call_facts.py)는 provider 자격 증명 없이 실행한다.

```python
from monoid_agent_kernel import CapturePolicy, ModelIOSubscription

class ModelDiagnostics:
    def on_model_call(self, capture):
        receipt = capture.receipt
        facts = receipt.termination
        evidence = receipt.usage_evidence
        print({
            "outcome": facts.outcome if facts else None,
            "requested_by": facts.requested_by if facts else None,
            "ended_by": facts.ended_by if facts else None,
            "known_output_tokens": (
                evidence.reported.get("output_tokens") if evidence else None
            ),
            "output_coverage": (
                evidence.completeness["output_tokens"] if evidence else "unknown"
            ),
        })

diagnostics = ModelIOSubscription(ModelDiagnostics(), CapturePolicy(mode="none"))
# AgentLoop.from_config(..., model_io_subscriptions=(diagnostics,))
```

예를 들어 attempt 1이 input=5/output=2를 보고하고 attempt 2가 usage 없이 실패하면,
`reported={"input_tokens":5,"output_tokens":2}`, 두 counter는 partial, total은 unknown이다.
구현 예제와 회귀 테스트는 같은 입력으로 이 결과를 검증한다.

observer는 진단 경로이며 오류가 격리된다. 확정 회계 소비자는 canonical invocation identity로 중복을 제거한다.
`ValidatedCallRunner`의 성공·repair receipts와 실패 subscription의 구분을 유지하고, escaping exception에도 동일한 관측 stamp를 전달한다.

## 5. 저장 형식·복구·각 소비자의 정책

### 5.1 reader 먼저, 새 record writer 다음

| 표면 | 결정 |
|---|---|
| Python receipt/attempt JSON | optional termination/evidence. 없으면 None. 기존 positional·legacy JSON 및 direct custom adapter 호환 |
| canonical invocation | `monoid.model-invocation.v2`. v1과 retained namespace alias reader 유지. receipt에 termination/evidence의 hand-listed 안전 projection 추가 |
| per-attempt 복구 | v2 evidence.attempts에 index·reported·source·collection 저장. 기존 private attempt_log 전체를 canonical에 복사하지 않음. recovered receipt의 attempt_log=() 허용과 별개로 관측 provenance 복구 |
| `model_calls.jsonl` | `monoid.model-calls.v2`. v1/v2 schema·mixed-line validator 유지. 안전한 receipt/attempt projection에만 새 필드 추가 |
| checkpoint | outer checkpoint.v1 유지, nested invocation v1/v2 지원. 중첩 future-version 거부를 adapter·bootstrap 이전에 검증 |
| private replay / result blob | 현재 형식·request digest 유지. 마지막 provider turn usage를 유지하며 logical retry 누적 receipt와 구별. canonical recovery는 invocation evidence 사용 |
| `model-content.jsonl` | 기존 v1 status/usage projection 유지. 새 facts는 이 content artifact에 저장하지 않음. snapshot reader는 facts 없음으로 표시 |
| live closed frame / Studio | v1의 optional termination/evidence 추가. 기존 필드·status 의미 유지. absence fallback 검증. live facts와 기록에 없는 facts 구분 |
| gateway success / HTTP error / SSE error / terminal | additive optional evidence 전달. 모든 parser 경로 검사. 구형 peer의 zero-filled usage는 legacy_ambiguous |
| OTel | 현재 span namespace·숫자 projection 유지, `monoid.model.*`에 bounded 사실/coverage 추가. attempt child에 `gen_ai.*` 추가 금지. 새 usage counter/histogram 도입 없음. 최신 Development 규약 전면 전환은 별도 |

canonical evidence는 기존 durable counter allowlist만 담는다. endpoint·model config·prompt/content·원본 exception text·raw key를 추가하지 않는다.
private ledger의 넓은 context와 public-safe canonical receipt를 같은 serializer로 합치지 않는다.
새 facts를 지원하는 consumer는 새 reader와 함께 배포한다. 구형 v1-only reader가 v2를 읽는 호환성은 제공하지 않는다.

### 5.2 immutable payload와 rollback

단순 `DurableCodec` migration 후 항상 최신 `to_json()`을 쓰면 같은 settled 좌표의 content hash가 바뀐다.
**검증된 원래 저장 payload의 canonical representation을 동일 좌표 재게시에서 재사용**하는 방식을 채택한다.
저장 경계의 private source envelope/동등한 기존 구조에 원본을 보존하고, public receipt나 raw provider payload와 섞지 않는다.
원본 참조는 불변 좌표·검증 hash에 묶는다. 새 revision·변경된 객체에 stale 원본을 재사용할 수 없게 검사한다.
새 revision을 생성하는 시점에만 현재 writer format을 선택한다. 원본 JSON text의 공백 일치는 요구하지 않으며 기존 hash 산식의 동일성을 요구한다.

이 범위에는 v1 schema alias, digest-generation alias, legacy `requires_evidence`, 생략된 기본 항목도 포함한다.
정상적인 현재 v1 writer payload 외에 retained reader가 허용하는 과거 표현도 fixture로 검증한다.
checkpoint·activation receipt에 이미 포함된 payload를 재정규화해 기존 hash를 바꾸지 않는다.

PostgreSQL payload는 현재 `json`이고 이를 유지한다. P1/P2를 위한 새 DB column/table은 필요하지 않다.
ObjectStore content-address와 request/logical-call/dispatch/idempotency identity는 유지하고, 새 record의 content hash만 새 metadata를 포함한다.

| 배포·복구 조합 | 허용 결과 |
|---|---|
| 새 reader + v1 record | 읽기·동일 좌표 재게시·required/outbox delivery·settled 복구 성공, 새 evidence는 unavailable |
| reader 준비 PR의 binary + v2 fixture | 읽기·복구 성공, writer는 v1 유지 |
| 새 writer + v1 reserved/dispatch_started | 허용된 새 revision은 v2; 기존 v1 revision 보존 |
| 새 reader + v1/v2 혼합 | 각 schema로 검증, 안정적인 identity로 중복 제거 |
| v2 작성 뒤 writer rollback | v2 reader를 유지한 reader 준비 버전으로 롤백. 기존 v2를 읽고 새 revision만 v1으로 작성 |
| v2 작성 뒤 출시된 v1-only binary로 rollback | 지원하지 않음. 읽을 수 있는 버전을 유지해야 함 |
| 미래 schema / 손상된 사실 | decode 실패 후 provider 0회, output 0회, 임의 downgrade·자동 유료 재시도 없음 |

별도 runtime feature flag는 도입하지 않는다. reader 준비 버전과 writer 활성화 버전의 배포 순서로 통제한다.
혼합 배포에서는 v2를 볼 수 있는 모든 활성 worker를 reader 준비 버전 이상으로 교체한 뒤 writer를 켠다.
구형 binary를 pin한 worker가 남아 있으면 활성화하지 않는다. 운영 배포 집합은 구현 착수 시 host 담당자가 확인한다.
불변 기록을 덮어쓰는 다운그레이드 migration은 제공하지 않는다.

### 5.3 복구와 재전달의 의미

- `unknown`에는 receipt/result를 만들지 않는다. runtime exception의 partial evidence는 프로세스 내에서만 남을 수 있다.
  unknown의 별도 partial-evidence 영속 slot은 기존 합의대로 후속이다. 재시작 시 evidence 없음과 유료 재시도 차단을 유지한다.
- settled success/failure는 v2에 저장한 최초 source/collection을 그대로 복원한다. recovery를 새로운 provider 보고로 세지 않는다.
- 회계는 기존 `_account_model_usage`/`_usage_delta`를 사용한다. receipt 재노출·중복 명령·required evidence 재전송·outbox 재전달로
  `total_usage`를 다시 더하지 않는다. metadata를 읽기 위한 추가 provider 호출은 0회다.
- 현재 OTel 구현은 span attributes를 내보낸다. 복구된 receipt의 진단 span이 다시 관측될 수 있으며,
  exporter/외부 수집기의 exactly-once 소비 계측은 이번 보장이 아니다. 새 metadata로 token 계측을 추가하지 않는다.
  회계 소비자는 span을 단순 합산하지 않고 canonical invocation identity를 사용한다.
- private replay adapter가 구형 payload corpus를 읽으면 usage 숫자는 유지하고 새 evidence는 legacy_ambiguous다.
  private replay에 새 사실을 만들기 위해 저장 형식을 연쇄 확장하지 않는다.
- stream outcome의 기존 성공 `usage`는 마지막 provider turn의 숫자다. 실패에는 기존 exception의 논리 호출 누적 usage를 전달한다.
  새 `usage_evidence`는 항상 논리 호출 범위다. Studio·문서에서 이 범위를 표시하고 성공 usage와 무조건 같다고 검사하지 않는다.

## 6. PR별 범위·의존성·우선순위

구현 승인 뒤 사용할 통합 브랜치는 **`codex/v0.25-p1-p2-call-facts`**다.
P1과 P2는 한 저장 형식 설계를 공유하고 PR은 순서대로 검토·병합한다. P4는 선행 의존성이 없다.

```text
최신 develop ─ P1P2-00 (CI 등록, 독립 PR) ─ develop
                                           └ integration
                                              └ P1P2-01 readers/구형 복구
                                                 └ P1P2-02 producer→실제 observer
                                                    └ P1P2-03 writer/복구
                                                       └ P1P2-04 전달·운영 소비자
                                                          └ P1P2-I → develop
```

| PR | 변경 범위·실제 소비자 | 의존성·제외 | 핵심 gate·profile |
|---|---|---|---|
| **P1P2-00** | `.github/workflows/ci.yml`에 정확한 통합 브랜치 등록; `tools/v023_ci.py`의 terminal profile을 combined로 강제; 기존 resolver 검사 갱신 | 최신 develop 독립 PR. 제품 코드 변경 없음 | workflow/profile fixture, lint; 통합 브랜치 push의 combined 선택 확인 |
| **P1P2-01** | 공통 optional 타입·closed v2 reader/schema·v1 유지, immutable 원본 재게시 경계, checkpoint/SQLite/PostgreSQL reader, historical fixtures·compatibility ledger | 00 이후 integration. writer는 v1. 실제 소비자는 기존 settled success/failure를 읽는 `FencedModelCallLifecycle.recover`와 새 v2 fixture | old-reader 차이·원래 hash 보존·모든 evidence policy, nested fixture, wheel; **combined** |
| **P1P2-02** | `_common`·OpenAI·gateway ingress presence, ModelTurn/TurnComplete, stamp/carry, runner·drain·attempt/receipt, loop failure observer, ValidatedCallRunner; 공개 subscription 예제 | 01 타입 사용. durable/ledger writer는 v1 유지하며 새 facts는 runtime에만 보존. 03이 영속 완료 조건 | 수정 전 실패 회귀부터: explicit 0/missing, cancel-drain observer, retries·끝 원인·P5 권한. **core** + 필요한 durable baseline |
| **P1P2-03** | invocation/ledger v2 writer 활성화, 안전한 attempt evidence projection, recovered receipt, same-coordinate recommit, accounting·required/outbox; 설치 wheel 복구 예제 | 01 reader + 02 producer. 새 DB 테이블·private replay 버전·unknown slot 제외 | 실제 PostgreSQL16/18·ObjectStore·Temporal 재시작, writer rollback, mixed format, 회계 한 번. **combined** |
| **P1P2-04** | Reference gateway success/error/SSE 전달, backend live close·Studio decoder/표시, OTel bounded 진단, DTO·계약 문서·generated UI assets | 02 facts + 03 영속. model-content는 기존 projection 유지. UI 전면 개편·OTel 표준 전면 이행 제외 | old/new peer 조합, HTTP/SSE 3개 오류 reader, 실패·0·legacy UI, OTel 중복 없음, UI build freshness, wheel. **combined** |
| **P1P2-I** | 통합 브랜치 → 최신 develop 누적 diff·공개 예제·compatibility 검토 | 01–04의 최종 head 리뷰/CI 완료. merge commit | 누적 두 독립 reviewer, full CI·combined·current/minimum Pydantic wheel; 릴리스 별도 |

01은 reader-only로 롤백할 수 있는 실제 복구 소비자를 포함한다. 02는 개발자가 즉시 관측 가능한 런타임 결과를 제공한다.
02의 v1 writer에는 새 사실이 저장되지 않는다는 중간 상태를 명시하고 G1 완료·배포로 선언하지 않는다.
04의 gateway reader가 02의 ingress와 공유하는 파일은 root가 직렬화하며 gateway 구형 응답은 그동안 legacy로 처리한다.
각 PR은 선행 PR 병합·최신 base 동기화 뒤 자신의 사전조사를 갱신한다.

### 6.1 초기 규모와 재검토 기준

| 단위 | 기능 변경 줄 추정 | 테스트·문서·fixture 포함 총 변경 줄 추정 |
|---|---:|---:|
| 00 | 20 | 80 |
| 01 | 400 | 1,300 |
| 02 | 600 | 1,900 |
| 03 | 350 | 1,250 |
| 04 | 400 | 1,450 |
| 통합·검토 반영 여유 | 100 | 600 |
| **초기 누적** | **1,870** | **6,580** |

초기 조사·계획·독립 계획 검토는 현재 문서 약 400줄과 별도 조사 노트를 포함하며 구현 비용과 따로 기록한다.
추정은 규모 판단용이다. 2,000 기능 줄 soft limit 접근 또는 초기 총 변경·검토 비용의 1.5배 초과 시
원인과 더 작은 경계를 검토한다. PR을 나누어 누적 비용을 초기화하지 않는다. 동등한 보장을 더 작은 수정으로 달성하면 그 방식을 택한다.
반복되는 serializer·원본 envelope 복잡성은 공통 codec의 무제한 재설계로 확대하지 않고 실제 두 형식의 소비자 요구로 다시 제한한다.

## 7. 검증 행렬

각 구현 PR의 행동 변경은 **그 PR 실제 base에서 수정 전 실패하는 회귀**로 시작한다.
기존 검사가 잡지 못하는 관찰점만 추가하고 단순 필드 미러링 테스트를 늘리지 않는다.

| 축·PR | 입력·경합 | 성공 관찰점 |
|---|---|---|
| ingress / 02 | None, `{}`, input=0/output 없음, optional zero, alias, bool·음수·malformed | canonical sparse report·숫자 projection·기존 오류 일치; 추정 total 없음 |
| stream / 02 | choice finish 뒤 usage-only; 같은 snapshot 반복; 명시적 0 교체; usage 없는 terminal | 첫 finish에서 조기 종료 없음, attempt당 한 snapshot, 미보고와 0 구별 |
| retries / 02 | 앞 시도 보고+뒤 시도 미보고, input-only, attempts=0, 구형 empty log | counter별 complete/partial/unknown/no_dispatch, 기존 attempt numeric 합계 유지 |
| termination / 02 | 정상·refusal·EOF·provider cancel·caller cancel·turn interrupt·host stop·deadline | 기존 status/error와 세 종료 사실의 의미 일치 |
| drain 경합 / 02 | cancel 후 terminal/EOF/error/deadline/budget, terminal 후 hang, backoff cancel | 최초 요청 유지, 실제 끝 원인, 회수량 observer 전달, remainder·delta 권한 회귀 없음 |
| 전달 / 02·04 | BaseException, translated error, required evidence 실패, ValidatedCallRunner repair | 관측 stamp 보존, 중복 stamp 합산 없음, provider 확정 사실을 evidence 전송 실패로 덮지 않음 |
| historical / 01 | 실제 v1 성공·실패 payload+hash, namespace/정책 alias, 중첩 checkpoint | passive/required/outbox 복구·동일 좌표 게시 성공, 최초 provider 추가 호출 0회 |
| v2 / 01·03 | 새 사실 round trip, malformed/미래 schema, canonical private-field 주입 | closed schema 검증, 오류 시 bootstrap/dispatch 전 중단, 안전 projection 유지 |
| upgrade·rollback / 03 | reader-prep commit에서 v2 fixture; old reserved→new settled; mixed ledger; v2 후 writer rollback | 형식별 reader·원본 hash 유지, 기록 덮어쓰기 없음, 기존 v1-only reader 거부 확인 |
| settled / 03 | success/failure 반복 복구, duplicate command, required retry, outbox duplicate | source/collection/attempt evidence 동일; 상태 usage 기여 한 번 |
| unknown / 02·03 | cancel·lease loss 후 uncertain dispatch, 프로세스 교체 | receipt/result 없음, partial runtime evidence의 한계 명시, 자동 유료 retry 0회 |
| replay / 03 | final turn 3 tokens + 앞 실패 2 tokens | private provider turn=3, logical receipt=5, 복구 누적은 5 한 번; request digest 안정 |
| gateway / 04 | old/new client×server, HTTP fail·nonstream fail·SSE error/terminal | optional absence 호환, 신형 evidence 전달, 구형 zero는 ambiguous |
| live·snapshot / 04 | 새 frame/구형 frame, 중단·0·부분값, model-content hydration | live facts 표시, 구형/기록 없는 facts는 unavailable. 성공을 EOF로 추정하지 않음 |
| OTel / 04 | attempt retry·failed usage·recovered settlement | bounded metadata, attempt child 중복 gen_ai 없음, metadata로 별도 token 계측 생성 없음; 복구 span은 진단용 |
| 서비스 / 01·03·I | PostgreSQL16/18, 외부 result blob, Temporal worker 교체 | hash·fence·outbox·재시작 경로 일치; in-memory 성공으로 대체하지 않음 |

기존 확장 대상은 `tests/test_model_call_receipt.py`, `test_model_call_runner.py`, `test_model_invocation.py`,
`test_model_calls_schema.py`, `test_model_stream_observer.py`, `test_usage_accounting.py`, `test_validated_call.py`,
`test_agent_loop_durable_recovery.py`, `test_backend_model_stream_live.py`, `test_otel.py`, provider/gateway tests,
`tests/service/test_postgres_sink_service.py`, `test_temporal_postgres_activity_service.py`, conformance compatibility fixtures다.
실제 test 경로·profile manifest는 각 PR preflight에서 최신 소스와 확인한다.

실행 명령의 기본형은 다음과 같다. repo-local `.venv`가 없는 현재 worktree에서는 공유 Python 절대 경로를 사용했다.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:PYTEST_XDIST_AUTO_NUM_WORKERS = '4'
python -m pytest -q -n auto --dist=worksteal --timeout=120 --timeout-method=thread `
  tests/test_model_call_receipt.py tests/test_model_invocation.py `
  tests/test_model_calls_schema.py tests/test_usage_accounting.py `
  tests/test_model_stream_observer.py --tb=short
```

구현 final head는 기존 lint·공개 surface·compatibility·전체 unit/contract·선택 service profile을 통과해야 한다.
Studio 수정 PR은 `npm ci`, `npm run check`, `npm run build`와 committed dist 일치를 확인한다.
current/minimum Pydantic 설치 wheel은 구형 fixture 읽기·새 공개 예제를 실행한다.
현재 계획 단계의 280 tests/probe는 이 구현 gate의 통과로 기재하지 않는다.

## 8. 대안·보류·재개 조건

| 선택하지 않은 안 | 이유·다시 검토할 조건 |
|---|---|
| downstream에서 usage=0/error 이름으로 사실 추론 | ingress에서 정보가 소실됨. 관측 소유자에서 작은 수정을 수행 |
| v1 closed schema에 추가 field 또는 permissive extra 전환 | 출시된 reader 호환성을 깨고 안전 projection 범위를 확장. 명시적 v2 채택 |
| 새 증거로 숫자 회계를 다시 계산 | 기존 retry/recovery dedupe 경로와 이중 회계 발생. metadata만 동반 |
| 모든 replay/content artifact도 새 버전 | 실제 새 사실의 복구 권위는 canonical invocation. content snapshot의 부재를 표시하며 필요한 두 형식에 집중 |
| raw provider payload·범용 이벤트 타임라인 저장 | privacy·크기·해석 비용. bounded outcome/observation만 사용 |
| immutable 기록 전체를 변환하는 rollback | hash·audit identity 훼손. dual-reader 유지 롤백 채택 |

후속 carry의 책임은 다음과 같다. unknown partial evidence 영속화는 G1 후속 담당이 incident/restart 진단 요구가 생길 때 재개한다.
model-content snapshot의 facts 영속화는 Studio 담당이 live 이후의 상세 진단을 제품 요구로 채택할 때 독립 schema 계획으로 재개한다.
OTel semconv 전면 migration은 관측성 담당이 사용 중인 exporter/collector 버전과 대시보드 이행을 확인한 뒤 진행한다.
SDK 내부 시도 계측은 provider 담당이 지원 adapter에서 실제 요구와 신뢰할 수 있는 hook을 확인할 때 진행한다.
현재 배포 worker/보유 데이터 집합의 inventory는 조사하지 않았다. writer 활성화 전 host 담당자의 배포 호환 확인 항목이다.

## 9. 독립 검토·이번 단계 실행 기록

계획 검토 대상: 위 기준 revision의 코드와 이 문서·연결 문서 diff.
fresh reviewer `p1p2_plan_review`는 작성·두 조사 worker와 별개로 읽기 전용 검토를 수행했고 **READY**로 판정했다.
검토한 본문 SHA-256은 `E25212941F049D4BFC470054FC327BE5F72359CCB12C633A1B2892BC4951EAFC`다.
이 판정 뒤에는 이 실행 기록만 추가했다. blocking finding은 없었다.

reviewer는 provider ingress/stream/drain/receipt/observer, snapshot과 dispatch 합산, 최초 요청과 종료 계기,
동일 좌표 recommit·PostgreSQL hash, nested checkpoint·unknown·safe projection·private replay를 소스와 대조했다.
PydanticAI #9491·Vercel #17578과 공식 streaming 계약은 표본 확인했다.
원본 probe와 280-test 실행 기록을 검토했으며 별도 runner를 실행하지 않았다. 모든 외부 링크·모든 소스 분기의 전수 검증은 아니다.

| 이번 단계의 실행 | 결과 |
|---|---|
| 시작·마감 `git fetch origin develop` | 모두 `4e14618b3ba68ffd9e243a9bcad2e3c9d2f5ac48` |
| `python .tmp/v025/p1p2_preflight_probe.py` | §2.1 네 경계의 기존 동작 확인 |
| §7의 다섯 파일 pytest 명령 | 280 passed in 9.50s |
| 네 계획·색인 문서의 local links / Python 예시 AST | 119 links 존재 / 6 blocks parse 성공 |
| `git diff --check` | whitespace 오류 없음 |

문서의 OTel 범위는 자체 코드 대조 중 정확히 제한했다. 현재 span attributes를 새 usage instrument와 구별하고,
canonical state의 중복 없는 회계와 복구 진단 span의 재관측을 분리했다. 계획 검토는 이 수정을 포함한다.
실제 DB·ObjectStore·Temporal 이행과 배포 worker inventory는 구현·활성화 gate이며 이번 단계에서 통과했다고 주장하지 않는다.
이번 단계는 문서와 ignored probe만 작성했다. 구현·tracked 회귀 테스트 추가·서비스 실행·PR 게시를 수행하지 않았다.
구현 요청 이후 첫 행동은 최신 develop 재조회와 P1P2-00/01 사전조사 갱신이다.

### 9.1 P1P2-00 구현 전 사전조사

2026-10-27 fetch한 `origin/develop`은 계속 `4e14618b3ba68ffd9e243a9bcad2e3c9d2f5ac48`이다.
독립 PR 브랜치는 `codex/v0.25-p1-p2-ci`. 계획 commit `94ea35a`도 이 준비 PR로 추적 문서에 반영한다.
root가 조사·구현·runner를 소유하고 `p1p2_01_research`는 후속 01의 읽기 전용 조사를 맡는다.
00은 CI 선택 신뢰를 다루므로 B 등급으로 독립 preflight와 코드 reviewer 한 명을 배정한다.

현재 `.github/workflows/ci.yml`은 push branch allowlist에 P3까지 등록돼 있다.
`tools/v023_ci.py:CAMPAIGN_TERMINAL_PROFILES`에는 v0.23과 develop만 있으며 신규 G1 head의 기본 profile은 core다.
실제 소비자는 integration push workflow와 `pr-full.yml`의 `Resolve plan-owned service profile` 단계다.
현재 ruleset 수는 0, develop protection 조회는 `Branch not protected`(404)였다. 서버 설정은 변경하지 않는다.

수정은 정확한 `codex/v0.25-p1-p2-call-facts` 한 항목을 workflow allowlist와 terminal profile map에 추가한다.
기존 resolve_profile의 explicit weaker label/manual-profile 거부를 재사용한다. wildcard 등록이나 별도 CI 엔진은 필요하지 않다.
`tests/test_v023_ci_foundation.py`의 기존 성공·거부 parameter와 workflow gate 검사를 확장한다.
기존 integration CI가 combined actual services를 실행하는 사실은 workflow body에서 확인했다.

검증: 해당 foundation tests, `ruff check src tests tools/v023_ci.py`, `validate-lock`,
CLI의 새 branch default=combined·weaker explicit 거부·기존 unrelated default=core를 확인한다.
PR L1/L2(core) 통과와 independent 코드 PASS 후 develop에 merge commit으로 병합하고,
새 integration branch를 그 develop에서 생성해 첫 push의 full CI를 확인한다.
00의 변경 예산은 기능 20줄/총 80줄이며, 이미 검토한 계획 문서 428 변경 줄은 계획 비용으로 별도 집계한다.

독립 `p1p2_00_review` preflight는 READY였다. 검토 대상은 source `4e14618`, 문서 `94ea35a`와 §9.1,
문서 SHA-256 `1EC6BC7D4FBB41E9B06DEBB533B37DE104DE4DFB9D2E9647578C61AD01E2F06A`다.
테스트부터 추가한 실행은 **4 failed / 19 passed in 5.82s**였다. 새 branch 기본 core, label/manual로 core 허용,
push allowlist 누락의 네 관찰점에서 실패했다. 이후 정확한 branch 두 항목을 등록했다.
수정 뒤 foundation은 **23 passed in 5.60s**, lock 검증과 CLI default=combined 확인이 통과했다.
정확한 branch에서 `--requested-profile core`는 예상한 `requires ci:combined` 오류로 거부됐다.
공유 Python에는 ruff module이 없어 CI 고정 버전 `uv tool run --from ruff==0.15.6 ruff check src tests tools/v023_ci.py`를
사용했고 통과했다. final candidate의 독립 코드 리뷰와 원격 L1/L2는 다음 단계다.

00 최종 `5d99863bc7aebf8a214c1e81e0605624078626c4`의 독립 코드 리뷰는 PASS였다.
[PR #156](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/156)의
[L1 40449464064](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/40449464064)와
[required L2 40449476412](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/40449476412)가 성공했다.
expected head를 고정해 merge commit `d4e659c860ca40b8c62a9527dbda1aea9701dc50`으로 develop에 병합했다.
통합 브랜치를 그 commit에서 생성·push했고 첫 full CI는 `40450049327`이다.
00의 기능 변경은 workflow/map 두 항목, 테스트·실행 문서 포함 후속 commit 변경은 68줄이다.
최초 계획 문서 428줄은 별도 계획 비용으로 유지한다.

### 9.2 P1P2-01 구현 전 사전조사

실제 base는 최신 develop/통합 head `d4e659c860ca40b8c62a9527dbda1aea9701dc50`이다.
작업 브랜치는 `codex/v0.25-p1-p2-readers`. `git diff 4e14618..d4e659c -- src`는 비어 있어 조사한 runtime과 동일하다.
root가 구현·runner를 소유한다. `p1p2_01_research`의 읽기 전용 조사와 독립 `p1p2_01_preflight_review`를 분리했다.
A 등급으로 구현 최종 head의 서로 다른 두 코드 리뷰·combined 서비스 CI·설치 wheel 검증이 필요하다.

**관측한 기준선.** 실제 `FencedModelCallLifecycle.recover`와 deterministic fenced host에 원래 payload hash를 심은
메모리 probe는 canonical v1 failure만 복구했다. namespace alias, legacy `requires_evidence`, 생략된 policy/result_ref는
decode=loaded 뒤 `durable_invocation_conflict`, digest-generation alias는 request conflict였다.
checkpoint의 nested 재직렬화도 원래 hash를 바꿨다. v2는 unsupported였다.
원본 코드·출력은 ignored `.tmp/v025/p1p2-01-baseline-probe.py`, `p1p2-01-baseline-probe-output.txt`다.
이 probe는 passive failed settlement만 실행했다. success·required/outbox·PostgreSQL·subscription은 구현 검증에서 추가한다.

**선택한 최소 구조.**

- `DurableCodec` 두 인스턴스로 v1 retained aliases와 canonical v2를 분리한다. v1을 v2로 migration하지 않는다.
  기존 `MODEL_INVOCATION_SCHEMA_VERSION` 기본 writer는 v1로 유지한다. 일반 codec API는 바꾸지 않는다.
- invocation의 private `init=False/compare=False/repr=False` source envelope가 원래 canonical JSON과 의미 fingerprint를 보관한다.
  기존 canonical `to_json()`과 저장용 `model_invocation_payload_for_write()`를 분리한다. 기존 encoder/hash를 사용한다.
  원본 반환 전에 현재 전체 의미·receipt facts를 비교해 nested mutation을 거부한다. 반환 map은 detached다.
- 같은 좌표의 invocation/evidence 재게시와 ambiguous reconciliation, loop checkpoint capture, nested checkpoint writer에서
  저장 serializer를 사용한다. 새 revision의 네 lifecycle `replace()`는 source envelope를 버리고 현재 writer=v1을 명시한다.
- 이미 허용한 digest-generation alias는 reader의 의미 표현에서 canonical domain으로 바꾼다. 원래 spelling은 envelope에 남긴다.
  request digest와 logical/dispatch identity는 바꾸지 않는다.
- PostgreSQL의 invocation/evidence 전용 typed projection은 legacy policy/default를 처리하며 raw schema/digest binding을 유지한다.
  원래 hash를 먼저 검증하고 future `unsupported_version` 분류를 보존한다. event/checkpoint/terminal의 공통 검사를 완화하지 않는다.

**공개 사실·reader 연결.** 작은 leaf `core/model_facts.py`에 `ModelCallTermination`, `UsageObservation`,
`UsageEvidenceAttempt(index, observation)`, `UsageEvidence(attempts)`를 둔다. 마지막 타입의 reported/completeness는 순서 있는
observation 목록에서 파생하고, wire reader는 저장된 aggregate와 파생값의 일치를 검증한다. JSON attempt는 평면 index/report/source/collection이다.
counter 11개·enum은 닫힌 어휘를 사용한다. 확인된 합계는 기존 numeric usage 이하이며 complete counter는 같아야 한다.
optional zero는 `usage.get(counter, 0)`과 비교한다. 새로운 optional receipt/attempt/outcome 필드는 끝에 추가하고 None이면 생략한다.
attempt의 기존 required-wire-key 예외 목록도 갱신한다. 실제 producer 생성은 02에 남긴다.

v2 invocation safe receipt와 v2 model-calls schema를 함께 읽되 두 기본 writer는 v1을 유지한다.
ledger semantic evidence 검사는 empty/absent attempt_log의 조기 반환 전에 수행한다.
`_recovered_receipt`와 공개 subscription까지 v2 facts를 연결한다. compatibility registry의 writer는 v1, reader는 v1/v2다.
packaged historical/v2 fixture와 nested checkpoint, current/minimum Pydantic wheel 소비를 검증한다.

**독립 검토에서 고친 retry 경계.** recovered 실패를 재시도할 때 기존 코드가 usage만 비우면 새 evidence와 충돌한다.
01은 retry cycle 진입 시 working receipt의 termination/evidence도 함께 비운다. 새 v1 revision의 사실은 unavailable이며,
이전 v2 immutable record의 사실/hash와 이미 쓴 numeric usage는 보존한다. backoff 중 취소도 같은 규칙이다.
순수 settled 재전달은 v2 사실을 그대로 복구한다. recovered observation prefix와 새 시도의 합성은 02가 소유한다.
이 중간 규칙은 01의 rollback reader 범위이며 G1 최종 provenance 보장을 줄이지 않는다.

**회귀부터 고정할 항목.** historical payload의 success/failure×passive/required/outbox 실제 lifecycle 복구,
checkpoint nested hash, v2 fixture의 공개 receipt 복원, mixed ledger와 malformed/future 거부를 먼저 실패시킨다.
v2 nonzero-evidence retry와 backoff cancel은 attempts·누적 숫자·새 v1 revision·옛 v2 hash를 관측한다.
source mutation, replace(new revision), stale fence, PostgreSQL typed-row/hash/evidence·ObjectStore·Temporal은 해당 기존 suite에 추가한다.

독립 preflight 판정은 **READY**다. 검토 원문은 ignored `.tmp/v025/p1p2-01-preflight.md`,
SHA-256 `00A70152A16E64801F33E67F0ABADE2D3B0878BB2CB2FFB7D871BB220B8A99D5`다.
reviewer는 source·원본 probe를 확인했으며 별도 runner를 사용하지 않았다.
01 초기 400 기능/1,300 총 변경 줄 추정을 유지해 기록하고, 추가로 확인한 경계를 포함한 작업 예산은 500/1,700줄로 갱신한다.
캠페인 누적 예상은 1,970 기능/6,980 총 변경 줄이다. 실제 구현 규모가 soft limit에 닿으면 후속 범위를 다시 평가한다.

첫 integration full CI `40450049327`은 성공했다. 01 수정 전 실제 consumer 회귀는
**33 failed / 7 passed in 3.05s**였다. canonical v1의 세 evidence policy×성공/실패와 기존 retry는 통과했다.
historical spelling/default의 같은 좌표 재게시, v2 reader, 공개 receipt/subscription, mixed ledger에서 실패했다.
테스트 초기 stale-fence 기대 예외를 기존 `WriteAuthorityRevoked` 계약에 맞춘 뒤 다시 실행한 수치다.
이 시점 `git diff -- src`는 비어 있다. 원본 출력은 ignored `.tmp/v025/p1p2-01-red.txt`에 보관한다.

실패 회귀 commit은 `cc101b5def74e65e5a1d0affcb5a94f200c8dbd9`다. 구현 후 같은 40개는 통과했다.
관측 타입의 닫힌 값·0·coverage·합계·private key 거부, source envelope의 nested mutation 거부,
새 revision의 v1 선택, 복구 retry/backoff cancel, LocalFs·SQLite nested 기록, packaged fixture를 추가해
관련 **608 passed in 7.32s**를 확인했다. raw 출력은 `.tmp/v025/p1p2-01-targeted.txt`다.
기존 필드 완전성 테스트는 새 optional 필드의 최대/생략 모양을 모두 검사하도록 확장했다.
PostgreSQL 16/18과 ObjectStore의 historical/v2 payload·원래 hash·typed binding·evidence 복구 테스트는
combined 서비스 CI에서 실행한다. 두 독립 최종 코드 리뷰, current/minimum Pydantic 설치 wheel, 원격 L1/L2가 남아 있다.

**01 첫 코드 리뷰와 수정.** `b2fdd29`에 대한 독립 a/b 리뷰는 NEEDS_FIX였다.
명시적 0 evidence가 있는 receipt의 `with_error` 숫자 덮어쓰기, 구독자 간 mutable observation 공유,
복구된 private result 손상 시 attempts=0으로 바뀌어 원래 오류가 가려지는 경로를 재현했다.
수정 전 추가 회귀는 4 failed/2 passed, stream 마지막 turn usage와 전체 호출 evidence를 비교하는
별도 회귀는 1 failed였다. 수정은 기존 값 보존·표준 읽기 전용 mapping·기존 복구 실패 게시 경로 재사용이며,
stream의 숫자 비교는 §5.3 범위에 맞춰 receipt가 담당한다. 수정 후 관련 137 passed in 3.20s.

공개 fenced-host conformance의 nested alias 기대값도 기존 정규화 정책이 남아 있었다.
기존 suite의 10 failed/265 passed를 확인한 뒤, outer checkpoint 별칭은 정규화하고 nested invocation의
다른 저장 표현은 conflict로 검사하도록 계약과 mutation test를 갱신했다. digest alias 의미 정규화는
계획대로 reader에 한정해 직접 constructor의 기존 관찰 계약을 유지했다. 해당 conformance와 복구·facts
회귀 **400 passed in 11.28s**. 첫 L1 `40452461088`은 public export/counter census 두 등록 누락으로 실패했고
두 검사를 갱신했다. draft full workflow의 성공 표시는 실제 required L2 통과로 계산하지 않는다.

`b2fdd29` wheel을 독립 환경에 설치하고 source 경로 없이 복사한 실제 소비자 테스트를 실행했다.
import가 site-packages임을 검사했고 Pydantic 2.14.1은 120 passed/3.75s, 최소 2.6.4는 120 passed/3.77s였다.
리뷰 수정 후 wheel을 다시 빌드해 같은 검증을 갱신한다. 기존 19개와 새 3개 fixture가 wheel에 포함된다.
현재 source Python diff 상한은 525줄(주석·공백 포함), 총 변경은 약 1,500줄이다. 초기 400/1,300과
사전조사 500/1,700 추정에 함께 기록한다. G1 source 상한 예상은 1,995줄로 soft limit에 가깝다.
02 사전조사에서 누적 실제 비용과 남은 범위를 다시 점검하며, 새 범용 serializer나 상태 머신은 추가하지 않는다.

재검토 `e7bffea`에서는 보고값이 없는 evidence가 기존 numeric stamp 수집을 막는 경계가 추가로 발견됐다.
빈 보고와 서로 다른 counter의 명시적 0 두 회귀를 먼저 실패시켰다(2 failed/2.75s).
기존 stamped 후보를 evidence와 검증한 뒤 채택하도록 좁혀 unknown counter의 숫자와 확인된 0을 함께 보존한다.
관련 receipt/facts **95 passed in 3.02s**. source 상한은 529줄, 누적 예상은 1,999줄이다.
`e7bffea`의 갱신 wheel audit은 통과했고 독립 설치 환경은 current/minimum 각각 125 passed/2.96s·3.21s였다.

01 최종 `74ff289431f203cb84595f90e0e09abfec12ee72`는 서로 다른 두 코드 reviewer의 PASS를 받았다.
최종 wheel audit과 설치 환경 Pydantic 2.14.1/2.6.4가 각각 **127 passed / 3.09s·3.04s**였다.
[L1 40453578644](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/40453578644)와
[required combined L2 40453679101](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/40453679101)이
성공했다. Python 3.11/3.12와 실제 PostgreSQL·ObjectStore·Temporal job을 확인했다.
[PR #157](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/157)을 expected head로 고정해
통합 브랜치에 merge commit `ec32ca8e2adc86354cfb9acd7cd5f3b9d657b537`으로 병합했다.

### 9.3 P1P2-02 구현 전 사전조사

실제 fetch한 integration base는 `ec32ca8e2adc86354cfb9acd7cd5f3b9d657b537`, develop은 계속 `d4e659c`다.
검토한 01 head와 tree가 같다. 작업 브랜치는 `codex/v0.25-p1-p2-runtime`이다.
root가 구현·runner, `p1p2_01_research`가 읽기 전용 조사, `p1p2_02_preflight_review`가 독립 검토를 맡았다.
사전조사 `.tmp/v025/p1p2-02-preflight.md` SHA-256은
`91F74B73941E3D206CFF629BDC25ABE8D68C0AE50780243458B4D5BC19F45B54`, 판정은 **READY**다.
같은 날 확인한 §3의 pinned 외부 구현을 재사용하고, 현재 OpenAI Responses 경로를 직접 대조했다.

**확인한 upstream 경계.** OpenAI는 raw terminal을 받은 뒤 EOF·cleanup을 마쳐야 TurnComplete를 내보낸다.
terminal을 일찍 yield하면 direct consumer가 client 정리를 마치기 전에 멈출 수 있다.
기존 wait bridge는 boundary가 이기면 callee의 결과·예외를 버리므로 예외 stamp만으로도 부족하다.
기존 `collect_retry_reports`의 ContextVar·mutable carrier 패턴을 재사용하는 작은 **dispatch별 usage collector**를 둔다.
숫자 snapshot·typed observation·normal/drain phase·closed만 보관하고 lock으로 update/phase/close를 보호한다.
dispatch 종료 시 닫아 늦은 worker가 다음 관측을 바꾸지 못하게 한다. RetryProgress의 논리 호출 범위는 유지한다.
원본 수신 시 만든 observation은 지연 terminal에서도 원래 collection을 유지한다.

normalizer의 alias 선택을 sparse 추출에도 재사용하며 명시적 0과 미보고를 분리한다.
숫자와 evidence를 한 snapshot으로 선택한다. 새 explicit report가 기존 pair를 교체하고,
usage 없는 terminal·legacy numeric-only conduit는 이미 확인한 explicit snapshot을 덮지 않는다.
OpenAI의 EOF refusal은 알려진 source 경계에서 provider_eof를 stamp한다. gateway의 기존 frameless EOF 허용은 유지한다.
runner는 stop 요청과 실제 읽기 종료를 각각 기록하며, provider cancellation은 CalleeCancelled의 원래 cause에서 구별한다.

**독립 검토에서 확정한 선행 수정.** malformed OpenAI 응답의 prompt_tokens/nested reasoning은 기존 실패 stamp에서
canonical 예산 항목으로 전달되지 않는다. first-party raw failure 경계에서 유효한 sibling을 보존하며 alias를
canonical로 한 번 선택한다. generic `usage_reported_by`의 임의 key 운반과 정상 parser의 엄격한 거부는 유지한다.
또한 실제 lifecycle probe에서 refused 5 → ambiguous 2가 cumulative stamp를 다시 더해 attempt-log ValueError로
바뀌었다. 기존 숫자 settlement에서 이미 합산한 dispatch와 현재 attempt를 구별하여 원래 dispatch_unknown과 합계 7을 보존한다.

recovered v2 evidence prefix는 private attempt log와 따로 유지하고 v1의 과거 dispatch는 legacy_ambiguous로 채운다.
backoff 취소는 호출 종료 사실만 바꾼다. loop failure observer·예외 translation·required evidence failure·ValidatedCallRunner까지
동일 사실을 전달한다. invocation/ledger writer는 v1, 03이 영속화, 04가 gateway server·live·Studio·OTel을 담당한다.

수정 전 테스트는 **14 failed / 1 skipped in 4.58s**였다. shared Python의 OpenAI 미설치로 생긴 skip은
별도 Python 3.11 환경에 최소 지원 OpenAI 1.80.0을 설치해 다시 실행했고 **1 failed in 5.41s**였다.
실제 terminal→hung EOF→deadline에서 이미 보고한 120/340/460이 빈 stamp로 나오는 실패다.
raw 출력은 `.tmp/v025/p1p2-02-red.txt`, `p1p2-02-openai-red.txt`이며 이때 `git diff -- src`는 비어 있다.

02 최초 600 기능/1,900 총 변경 줄 추정은 유지하며, 숨은 수신 경계와 실패 전달을 반영한 예산은 850/2,300이다.
01 실제 source 상한 529를 포함한 누적 예상은 2,249 기능 상한/약 7,500 총 변경 줄이다.
범용 상태 머신·serializer·새 회계 경로는 추가하지 않고 기존 ingress·race·stamp·settlement에 수정을 제한한다.
기능 2,000줄 soft limit에 접근한 원인과 이 대안을 검토했다. 최종 실제 diff·리뷰 비용을 계속 누적한다.

**02 구현 후보.** 수정 전 회귀 commit은 `9e8c0ef`다. 최초 15개가 통과한 뒤 실제 observer,
재시도·복구 prefix, 명시적 0과 drain provenance, gateway 선택 reader, unknown 오류 변환을 확장했다.
준비 callback이 취소를 요청한 뒤 provider에 진입하는 기존 경계도 확인하여 진입 직전 취소 검사를 추가했다.
`attempts=0`과 provider 진입 0회를 검증한다. unknown의 기존 failed 결과와 별도로 최초 stop·read end를 보존한다.

최종 후보의 관련 suite는 **834 passed in 23.69s**였다. 원본은
`.tmp/v025/p1p2-02-candidate-tests2.txt`다. 전체 ruff와 diff check, 오프라인
`examples/model_call_facts.py`도 통과했다. 예제는 input=0의 complete와 total의 unknown을 출력한다.
runner까지 unwind하는 BaseException은 관측을 보존하며, Python event loop 자체를 끝내는
KeyboardInterrupt/SystemExit의 기존 원형 전파도 검사했다. 프로세스 종료 뒤 subscriber 실행 보장은 추가하지 않는다.
carriage census는 새 observation helper·상태 전달·실패 시 비용 보존을 기존 검사에 등록했다.
stream 종료 뒤 오류·취소·deadline 경합과 P5 권한 회귀도 같은 실행에 포함했다.
01 병합 후 integration full CI `40454581400`은 성공했다. 02는 두 독립 코드 리뷰와 최종 CI를 진행한다.

**02 리뷰 수정.** 첫 후보 `6950e44`는 두 독립 코드 리뷰에서 NEEDS_FIX였다. 기본 drain-off의 stop,
로컬 stream callback·fragment 검증을 provider error로 분류하는 경계, malformed usage의 scalar/container 혼동,
호출자 except 문맥이 새 성공 observer를 덮는 경계, 두 번째 durable dispatch의 BaseException이 이전 receipt를
재사용하는 경계를 각각 실패시켰다. source owner에서 read-end를 stamp하고, 로컬 caught exception을 보관하며,
dispatch 시작 시 이전 settlement 임시값을 비우는 수정으로 해결했다. 새로운 회계 경로는 추가하지 않았다.

복구 integrity 오류와 required evidence 재게시 오류도 실제 lifecycle·observer로 확인했다. 독립 targeted
preflight는 READY였다. 결과 검증 실패의 공개 복구 outcome은 failed/local_failure이며 저장된 원본은 유지한다.
검증된 복구 receipt 전용 carrier를 recovery hook/decoder 경계에서만 소비해 historical attempts·usage를 보존한다.
required evidence 게시 실패는 확인된 provider outcome을 유지하고, 읽지 않은 final text·마지막 turn usage는 비워 둔다.
required refusal 재전달은 원래 facts와 기존 빈 accounting delta를 함께 보존한다.
추가 회귀 원본은 `.tmp/v025/p1p2-02-*-red.txt`, 수정 후 관련 **884 passed / 22.71s**는
`p1p2-02-review-green.txt`에 있다. source 변경 상한은 약 710줄(공백·주석 포함)로 갱신 예산 850 이내다.
초기 600/1,900 추정과 누적 비용 기록은 유지한다. 최종 head 재검토·wheel·CI를 다시 확인한다.

재검토 `ef50d9c`는 기존 지적을 해소했다. 남은 required-evidence 복구 재실패의 subscription 미전달도
02의 공개 전달 계약에 포함했다. 실제 host에서 성공·refusal·손상 결과 세 회귀가 먼저 실패했다.
같은 checked carrier를 기존 게시 경로에 연결하고 원래 invocation의 failure code·key·request digest를
유지했다. 원래 provider receipt와 현재 projection 오류를 구분하며 콘텐츠는 생성하지 않는다.
관련 **42 passed / 2.77s** (`p1p2-02-recovery-subscription-green.txt`), 전체 lint 통과.
`ef50d9c` wheel audit과 current/minimum 설치 검증은 각각 151 tests PASS였고, 새 final candidate로 갱신한다.

`c979d96` 재검토에서 evidence 게시가 query digest 검증보다 먼저 실패하면 원래 receipt와 현재 prompt를
함께 공개하는 반례가 나왔다. 실제 full subscription 회귀를 먼저 실패시켰고, digest가 다른 복구 오류는
현재 요청 content·prompt digest·preimage를 생략하도록 기존 게시 함수를 연결했다. payload capture를 켠
동일/변경 요청 테스트로 원래 식별자와 필요한 preimage의 유지/생략을 검사했다(4 passed/2.35s).
`c979d96` 설치 wheel은 current/minimum 각각 152 tests PASS; 최종 변경으로 다시 갱신한다.
확장 검증 604개에서 새 recovery carrier의 config_recoverable census 등록 누락과 기존 30ms backoff의
Windows 타이밍 하한 assertion이 한 번 실패했다(602 passed). 등록을 보완했고 같은 timing case를 포함한
census 196개는 통과했다(10.23s). retry 대기 코드는 변경하지 않았다. 최종 원격 CI로 전체 계약을 확인한다.

`b8f2bd3` 설치 wheel은 current/minimum 각각 153 tests PASS였다. 재검토에서 변경된 모델 설정도
과거 사용량에 귀속되는 반례를 확인했다. journal에는 과거 ModelConfig가 없으므로 targeted preflight로
정책을 확정했다(READY). query digest가 다른 required-evidence 실패는 예외·journal의 검증된 사용량과
facts를 유지하고, 재구성한 subscription/ledger receipt 게시를 생략한다. 현재 routing으로 열린 stream은
failed/evidence_uncommitted로 정상 close하며 text·usage·termination·evidence를 비워 둔다. 같은 query의
복구 게시와 기존 회계 증분은 유지한다. 새 placeholder 모델이나 wire 형식은 추가하지 않는다.
실제 host의 모델·provider·destination 변경 회귀는 수정 전 실패했고, 동일/변경 요청 5개는 수정 후
통과했다(2.55s; `p1p2-02-recovery-attribution-{red,green}.txt`). 기존 content-only 억제를 이 정책으로 대체한다.

**02 완료.** PR #158은 최종 `7208615e91e173a2c15781dffad8eda3f6f14970`에서 두 독립 코드 리뷰 PASS,
L1 `40461636410`·필수 core L2 `40461723407` SUCCESS 후 통합 `617deadc5012c2d749b395497651a7dc69985572`에
병합했다. 최종 코드 `1cc172b` wheel은 current/minimum 각각 154 tests PASS(3.27/2.93s), 예제·audit 통과.
마지막 head는 계약 문서만 수정했으므로 같은 binary 근거를 재사용했고 문서 delta도 두 리뷰가 확인했다.
마지막 attribution 관련 recovery·observer·census 297개 통과(13.22s). 실제 source 추가 788/삭제 104줄,
전체 추가 1,501/삭제 128줄이다. 줄 수에는 공백·주석이 포함되며 기능 비용의 보수적 대용치로 기록한다.

### 9.4 P1P2-03 실제 base 사전조사

새 branch `codex/v0.25-p1-p2-storage`의 base는 위 02 병합본 `617deadc`다. 재조회한 develop은 `d4e659c`다.
사전조사 `.tmp/v025/p1p2-03-preflight.md` SHA-256
`064BC150A186FBF04FD865D42DBE5183161ED4BAB88281F0BA6AC38EB2979C71`는 독립 검토 READY다.
기존 두 codec·source envelope·bounded projection·실제 service harness를 재사용한다. v1 codec 식별자는
고정하고 version dispatch reader를 registry에 기록한다. reader-prep binary가 새 v1 revision을 쓰는 롤백에서는
그 revision이 보존하지 못한 관측을 legacy_ambiguous로 유지하며 새 provenance를 만들어 내지 않는다.
03 추정은 350 기능/1,250 총 변경 줄이다. 누적 2,187은 추가 줄 기준 예상이며 02 삭제까지 포함하면
2,291이다. 이 기준을 구분하고 더 작은 기존 경계 재사용으로 soft limit 접근 원인을 해소한다.
root가 제품 소스와 단일 테스트 runner를 담당하고 별도 worker가 지정된 세 service-test 파일을 담당한다.
모든 편집을 마친 뒤 테스트를 실행하며, 최종 두 코드 리뷰·설치 wheel·combined CI는 별도 gate다.

03 실제 base의 제품 소스를 수정하기 전 새 회귀 11개가 모두 실패했다(3.04s;
`.tmp/v025/p1p2-03-red.txt`). 실제 fenced 복구 6개, recorder 성공·실패·retry 3개,
historical v1/v2에서 새 revision 전이 2개다. service worker는 같은 경계를 기존 PG16/18,
ObjectStore·Temporal 테스트에 연결했고 아직 실행하지 않았다. 아래 테스트 전용 commit으로 red를 고정한다.

**03 후보.** 테스트 전용 commit은 `25c8fbf`다. 첫 red 중 outbox 2개에는 crash fixture의
stage_evidence 전달 누락이 있었다. fixture를 보완하고 같은 base의 설치된 02 wheel에서 전체 11개를
다시 실패시켰다(2.24s; `p1p2-03-red-base-wheel.txt`). 잘못된 동일 checkpoint 재사용도 고쳐서
반복 evidence 검증은 실제 lifecycle.recover를 사용한다. 서비스 실패 fixture는 확정 refusal을 사용한다.
제품 writer 수정 후 관련 **319 passed, 4 skipped / 4.65s** (`p1p2-03-related2.txt`)였다.
skip은 Windows 파일 symlink 권한이며 Linux CI에서 계속 검증한다. 확장 계약 545개 중 10개는
current version과 namespace alias를 같은 것으로 가정한 conformance fixture에서 실패했다.
별칭 probe를 명시적 v1 쌍으로 고정하고 일반 probe는 v2를 유지했으며 관련 **275 passed / 9.34s**였다.
bounded projection·두 closed schema·version registry를 갱신했고 공개 오프라인 JSON 복구 예제는
provider 호출 1회, v2 facts 유지와 total unknown을 출력했다. 전체 lint/diff check 통과.
두 독립 코드 리뷰·실제 binary rollback·설치 wheel·필수 combined CI를 이어간다.

03 PR #159의 첫 후보 `4eadd548` wheel audit과 current/minimum 설치 검증은 각각
**336 passed, 4 Windows symlink skipped**(5.02/3.84s)였고 복구 예제도 양쪽에서 통과했다.
제품 import는 site-packages에서 확인했다. 실제 binary 검증 스크립트는
`.tmp/v025/p1p2-03-installed-tests/binary_rollback.py`, 교환 JSON은 같은 폴더의 `binary-history.json`이다.
03 wheel → 01 reader-prep wheel → 03 wheel 순서로 별도 프로세스를 실행했다. v2 성공 기록·ledger·nested
checkpoint를 읽고 같은 좌표를 재게시해 hash를 보존했다. 01 lifecycle이 v2 reservation에서 새 v1
started/settled revision을 썼고, 03 lifecycle은 이를 읽어 새 v2 retry revision으로 이어갔다.
구형 writer가 보존하지 못한 첫 observation은 legacy_ambiguous, 새 관측은 partial로 검증했다.
이 binary 전이 검증은 저장/lifecycle 호환성을 증명하며 실제 runner의 retry 회계는 앞의 복구 테스트가 검증한다.
PyPI v0.24.0의 별도 설치 reader는 v2를 unsupported_version으로 거부하고 v1은 읽었다.
01 wheel SHA-256 `ab8e726eb0a5c79b25264088e71c1e5f54424d5fd034a2ab25fbf75505f464b4`,
03 wheel SHA-256 `18c3eebba5ec4405188623930b5feb79f1919f8acb7c37201671606c1edc9cfd`다.

두 코드 리뷰가 같은 mixed-ledger fixture 누락을 지적했다. `_record()`가 새 writer를 따르면서 두 행 모두
v2가 된 검증 결함이었다. 과거 행을 명시적 v1으로 고정하고 실제 AgentLoop/recorder 재개 시 원래 v1
bytes를 보존하며 v2를 append하는 회귀까지 보완했다(2 passed/2.46s). 제품 코드는 그대로다.

`6cb2a36`은 두 최종 코드 리뷰 PASS와 L1 `40464755363` SUCCESS였다. combined L2
`40465167666`의 두 Python 코어 job은 통과했고 실제 서비스는 226 passed/8 failed였다.
새 PG 회귀의 응답 유실 주입이 settlement 전 `load_invocation()`의 default transaction에서 먼저
발동했다. 주입을 실제 settled invocation commit 진입에 한정해 저장 완료 후 응답 유실과
sink의 canonical readback을 검증하도록 고쳤다. 제품 binary와 기존 wheel 근거는 유지한다.
테스트 delta를 두 리뷰가 재검토하고 새 head의 전체 combined CI를 다시 검증한다.

**03 완료.** 최종 `fe44f736ab7ca38b3394e9074e42a454a3c38415`는 두 독립 targeted 리뷰 PASS,
L1 `40466682625`와 필수 combined L2 `40466739358` SUCCESS였다. 실제 PG16/18·ObjectStore·Temporal
서비스까지 통과한 뒤 PR #159를 통합 `0171d032a1460ac64c615fb411f5711518dd9e79`에 merge commit으로 병합했다.

### 9.5 P1P2-04 실제 base 사전조사

`codex/v0.25-p1-p2-consumers`는 위 03 병합본에서 생성했다. 최신 develop은 재조회한 `d4e659c`다.
`.tmp/v025/p1p2-04-preflight.md` SHA256
`85002F1FAE2D9CB62ED3F8B202FEFFB72E4E2F2F659F05B84AA0E93A29CCDCEE`는 독립 검토 READY다.
같은 날 갱신한 §3 외부 구현 근거와 실제 gateway·live·Studio·OTel 소비 코드를 대조했다.

gateway는 dispatch observation을, live close는 logical-call aggregate를 전달한다. 기존 공개 raw-chunk
assembly 검증을 유지하고 runner가 이미 정규화한 chunk는 내부 fold에서 재수집 없이 사용한다.
반복 A→B terminal을 재정규화하면서 원래 collection이 덮이는 문제를 이 경계에서 해결한다.
오류 envelope의 손상된 observation은 생략하고 기존 classified error와 숫자 사용량을 보존한다.

live broker는 기존 byte 제한을 지킨다. 기존 콘텐츠가 들어가면 새 optional facts를 먼저 생략하며,
기존 콘텐츠도 초과하면 compact marker에 facts를 함께 시도한다. Studio는 정확한 호출 identity의
수신된 facts만 hydration 사이에 보존한다. 기존 terminal SSE 종료 fence를 유지하므로 durable terminal이
먼저 도착한 경우 snapshot만으로는 unavailable이다. 새 durable content 형식을 추가하지 않는다.
OTel은 기존 span의 bounded 속성만 추가하며 numeric meter·attempt child 생성 규칙을 유지한다.

root는 Python 구현·공유 테스트/build runner를, `p1p2_01_research`는 Studio 네 source/script 파일을 담당한다.
먼저 실제 소비자 회귀를 실패시킨 뒤 구현한다. 최초 400 기능/1,450 총 변경 줄 추정을 유지하며
Studio 예상 230–300 기능/200–280 테스트 줄을 포함해 실제 누적 비용을 기록한다.
완료 gate는 두 최종 코드 리뷰, actual old/new peer, UI check/build freshness, current/minimum wheel,
PG16/18·ObjectStore·Temporal combined CI다. 릴리스·배포·CSP 변경은 포함하지 않는다.

04 제품 수정 전 Python 회귀는 **24 failed/4 passed, 5.58s**였다
(`.tmp/v025/p1p2-04-red-corrected.txt`). 최초 테스트의 acall 반환 tuple 해제를 고친 뒤 다시 실행했다.
실패는 실제 HTTP/SSE 관측 누락, 반복 terminal의 drain→normal 변경, live close 정보 누락,
실제 retry OTel 속성 누락이다. 기존 비용·malformed metadata 유지 검증 4개는 통과했다.
Studio의 실제 모듈·RunView SSR·App 연결 검증 8그룹도 모두 실패했다
(`.tmp/v025/p1p2-04-studio-red.txt`). 이때 Python·Studio 제품 source diff는 비어 있다.

**04 첫 구현 후보.** red commit은 `878196c`다. 관련 기존 계약까지 537개 통과 후 새 optional error
observation의 census 등록 한 건을 보완했다. 확장 runner·receipt·consumer·OTel·census는 676개 통과했고,
추가 proof-refusal 테스트의 잘못된 `on_unsupported="error"`를 실제 vocabulary `fail`로 고친 2개도
통과했다(2.78s). 원본은 `p1p2-04-candidate1.txt`, `p1p2-04-candidate2.txt`다.
Studio status/outcome 불일치도 별도 red 후 거부하도록 수정했다. 전체 npm check는 오류·경고 0,
기존 모든 script와 새 8개 facts 그룹 PASS였고 build 결과를 함께 저장했다.
Python lint/diff check 통과. 현재 제품 source/Studio 추가 315·삭제 15줄(공백·주석 포함)로
최초 기능 추정 400 이내다. 두 독립 코드 리뷰, 실제 구형 client binary와 새 wire, 설치 wheel,
최종 combined CI는 계속 진행한다.

`24855df` wheel audit과 현재·최소 의존성의 실제 설치 검증은 각각 **406 passed / 4 skipped**
(9.69/9.02s)였다. skip은 Windows symlink 권한이며 Linux CI에서 검증한다. Pydantic 2.14.1/2.6.4,
HTTPX 0.28.1/0.27.0, OTel SDK 1.45.1/1.20.0을 사용했다. 두 공개 예제도 양쪽에서 통과했다.
wheel SHA256은 `064dc6095b06e7dc94de789af92ffd1084be4b6772efc30abd345bfa2f03ac00`이다.
`.tmp/v025/p1p2-04-actual-binary-peers.txt`는 실제 01 reader-prep wheel과 04 wheel을 별도 프로세스로
연결한 양방향 sync/SSE 성공·실패 검증이다. 구형 peer의 숫자와 오류 분류를 유지하며 관측을 합성하지 않았다.

첫 후보의 독립 리뷰 A는 PASS, B는 Studio projection의 P2 회귀를 확인했다. 현재 호출과 turn이 다른
compact close를 reducer가 거부해도 App의 후속 projection이 현재 answer/reasoning을 지웠다.
실제 decode → reduce → project 경로의 회귀를 먼저 실패시켰다
(`.tmp/v025/p1p2-04-projection-red.txt`). terminal projection도 수락된 close에만 적용하는 경계로 수정하고,
같은 identity의 late facts enrichment와 기존 terminal SSE fence를 유지한다. Studio bundle을 다시 생성하고
최종 head의 두 targeted 리뷰·wheel audit·필수 combined CI로 마무리한다.

수정 후 실제 projection 회귀와 모든 Studio 검증을 통과했다. `npm run check`는 오류·경고 0,
기존 13개 protocol/reducer/transport/integration 그룹과 8개 facts 그룹 PASS다.
`npm run build`로 서버가 제공하는 bundle도 재생성했다. 원본은
`p1p2-04-projection-check.txt`, `p1p2-04-projection-build.txt`다. 이번 수정은 Studio source·검증·bundle에
한정되므로 같은 Python runtime의 앞선 406개 설치 검증과 양방향 gateway binary 근거를 재사용한다.
새 bundle을 포함한 최종 wheel의 package audit은 별도로 수행한다.

**04 완료.** 최종 `5b2aeb74cf61d0c0ec685e2d7b62ac56c32427c2`에서 두 독립 targeted 리뷰 PASS,
L1 `40470663542`와 필수 combined L2 `40470870142` SUCCESS였다. Python 3.11/3.12와 실제 서비스 gate를
확인한 뒤 PR #160을 통합 `cc7d8480a459d9e2281fbd36d32181185bfd1cc1`에 merge commit으로 병합했다.
최종 wheel SHA256 `0056606DCAFA3C50B8CA7B55C1C9197FA0F90180AB51D62096C2D5854D40D23B`의 audit과
tracked Studio bundle 3개 파일의 byte 일치도 확인했다.

### 9.6 P1P2-I 누적 검증과 통합 수정

실제 base는 최신 develop `d4e659c`와 04 병합본 `cc7d848`다. 누적 사전조사
`.tmp/v025/p1p2-integration-preflight.md`와 아래 두 수정 경계의 독립 검토는 READY다.
두 코드 reviewer는 개별 PR 검토 경험을 밝히고 전체 `d4e659c..5b2aeb7`의 소비자 연결을 다시 검토했다.
`cc7d848`의 제품 tree는 `5b2aeb7`과 같다. 각 reviewer가 확인한 P2를 다음과 같이 수용했다.

- legacy adapter가 stop 전 3, drain에서 9 → 5 → 미보고 terminal을 보내면 기존 drain fold가 3으로
  되돌아갔다. 실제 reader-prep binary는 5를 보존했다. 이미 ingress가 선택한 dispatch collector의
  숫자 snapshot을 drain에서 사용하며, 기존 exception 우선순위와 출력 차단 경계를 유지한다.
- gateway가 성공 응답의 잘못된 observation 또는 적용 증명 누락을 거부할 때 종료 원인이 provider_error로
  표시됐다. 실제 local validation 생성 지점에서 기존 오류에 local_failure를 stamp한다. 같은 오류 코드를
  가진 원격 오류와 network/timeout은 기존 provider 경계를 유지한다. 기존 constructor와 carriage census를
  유지하는 작은 private stamper를 사용한다.

두 수정 전 실제 runner 회귀는 **16 failed / 6 passed, 3.07s**였다
(`.tmp/v025/p1p2-i-regressions-red-corrected.txt`). 실패는 legacy drain 2개와 sync/SSE local 거부 14개다.
같은 gateway_bad_response 코드를 가진 정상 원격 오류와 network control 6개는 통과했다.
첫 실행의 미보고 numeric 키 assertion을 기존 empty usage 계약에 맞게 고친 뒤 다시 실행했다.
이 회귀를 테스트 전용 commit으로 먼저 고정하고 제품 수정을 진행한다.

회귀 전용 commit은 `989feb6`이다. collector snapshot 재사용과 local gateway stamper를 적용한 뒤
runner·gateway·stream observer·회계·carriage census **750 passed / 27.40s**였다
(`.tmp/v025/p1p2-i-repair-green.txt`). 같은 오류 코드와 손상된 optional observation을 가진 원격 오류도
provider_error로 유지하는 actual runner control을 추가했다. 최종 설치 wheel과 CI는 수정된 binary를 사용한다.

비용은 최초 1,870 기능 / 6,580 전체 변경 줄 추정을 유지한다. 04까지의 non-merge commit을 모두 합산하면
runtime·UI 추가 1,598/삭제 208(1,806줄), 예제 147, compatibility fixture 160, 테스트·script 2,528,
문서 497, bundle 29로 총 5,167줄이다. 공백·주석과 같은 줄의 반복 수정을 포함한다. 최종 net runtime·UI는
1,732줄이다. 이전 넓은 source 집계에는 예제·JSON fixture가 포함됐으므로 이 수치와 구분한다.
반복 작업은 복구 귀속, 소비자 identity, drain snapshot 선택과 owner stamp의 누락에서 발생했다.
기존 collector·codec·오류 stamp 경계를 재사용해 수정을 제한하며 새 회계 경로나 범용 validator를 추가하지 않는다.

최종 두 누적 리뷰, full CI·combined 서비스, current/minimum 설치 wheel·공개 예제·package audit을
실제 수정 head에 연결한 뒤 develop에 merge commit으로 반영한다. 정확한 최종 head·CI·merge SHA는
통합 PR과 같은 패스의 프로젝트 memory에 기록한다. 배포 전 dual-reader host inventory와 §8의 후속 범위는 유지한다.

`f540d76`은 두 최종 누적 리뷰 PASS, installed wheel current/minimum 각각 **431 passed / 4 Windows skip**
(9.33/9.01s), 공개 예제·audit·실제 구형/신형 gateway·v2 → reader-prep v1 → v2 binary 복구를 통과했다.
필수 combined CI `40473753359`도 SUCCESS였다. 이어 자동 GitHub 리뷰의
[최초 요청자 경합 지적](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/161#discussion_r4386472020)을
실제 runner로 확인했다. 먼저 user_cancel/turn_interrupt를 수락한 drain에 caller task 취소가 도착하면
requested_by가 caller_cancel로 바뀌었다. 기존 CancelledError·cancelled outcome·stop_request·사용량·출력 차단은 유지됐다.

반대 순서도 확인했다. caller 취소 뒤 cleanup에서 user token을 취소하면 caller_cancel이 최초다.
최종 drain.started만 읽는 수정은 이 경로를 깨뜨린다. 독립 targeted 사전 검토 READY에 따라 shared wait의
caller 취소 관찰 지점에 optional callback을 연결하고, drain owner가 첫 관찰 때 요청자를 고정한다.
wait와 drain owner의 cleanup await에서 추가 취소가 와도 같은 사실을 전달한다. 기존 예외·회수·권한 흐름은 유지한다.
새 종료 상태 머신 없이 기존 stamp와 두 단계 wait 경계를 사용한다. 다른 wait 호출자는 callback 기본값을 유지한다.

제품 수정 전 실제 runner 7개 경합 회귀는 **4 failed / 3 passed, 3.64s**였다
(`.tmp/v025/p1p2-i-caller-order-expanded-red.txt`). 앞선 stop→caller 및 cleanup 중 첫 caller 4개가 실패하고,
caller 단독·caller→cleanup의 user stop·반복 caller control은 통과했다. 이 회귀를 먼저 commit으로 고정한다.
공유 wait primitive 변경이므로 기존 bridge·model/tool 취소·durable unknown/P5 계약과 두 리뷰를 갱신한다.

회귀 commit은 `cf8f727`이다. 수정 후 경합 7개와 관련 model/tool wait·replay·durable 복구·observer·census
**810 passed / 26.00s**, lint/diff check를 통과했다(`p1p2-i-caller-order-green.txt`,
`p1p2-i-cancellation-related.txt`). callback은 caller 취소 시점과 cleanup 취소만 관찰하며 기본값은 None이다.
runtime·UI의 수정 이력을 포함한 변경량은 이 48줄 delta까지 **1,981줄**이다. 최초 1,870줄 추정과
soft limit 접근을 다시 검토했고, 기존 wait에 선택적 관찰점을 추가하는 경계가 task/shield 재구성보다 작다.
이 최종 source로 wheel·두 targeted 리뷰·필수 CI를 갱신한다.
