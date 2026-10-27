# v0.25 P1·P2: 종료 사실·사용량 관측·저장·복구 PR 계획

조사일: 2026-10-27 KST. 기준: fetch한 `origin/develop=4e14618b3ba68ffd9e243a9bcad2e3c9d2f5ac48`.
P5·DX1·P3가 병합된 상태다. 계획은 로컬 `94ea35a`에 기록했다.
**2026-10-27 사용자가 P1P2-00·01 구현과 리뷰·CI 후 02·03·04 순차 진행을 승인했다.**
00은 develop 대상 독립 PR, 01은 수정 전 실패 회귀부터 시작한다. 릴리스·배포·CSP 변경은 별도다.

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

다음은 **계획된 API**다. 새 타입은 아직 구현되지 않았다. 현재 공개 subscription을 첫 실제 소비자로 사용한다.

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
