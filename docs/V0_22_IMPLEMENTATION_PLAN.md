# v0.22 Production Boundaries 구현 전 조사와 실행 계획

> 상태: 구현 기준안
>
> 작성일: 2026-08-21
>
> 기준 브랜치: `codex/v0.22-production-boundaries`
>
> 기준 커밋: `9dff21f` (`origin/develop`과 동일)

## 1. 결론

v0.22는 현재 구조에서 구현할 수 있다. 안전한 구현에는 다음 세 가지 구조 변경이 필요하다.

1. 모델 호출 identity와 dispatch 상태를 provider 호출 전에 durable journal에 기록한다.
2. 실행 의미를 담는 checkpoint와 호출 도중의 짧은 journal transition을 분리한다.
3. 분산 저장소 계약을 `monoid_agent_kernel.hosting`에 두고 플랫폼 구현은 host가 소유한다.

Temporal adapter는 이 계약을 소비하는 후속 작업으로 둔다. v0.22는 Temporal runtime과 workflow
replay를 포함하지 않는다.

이 계획은 다음 결과를 목표로 한다.

- provider dispatch 여부가 불명확하면 자동 유료 재호출을 막는다.
- stale writer의 checkpoint, event, invocation, terminal mutation을 막는다.
- 모델 성공과 evidence 저장 실패를 각각 보존한다.
- Stop, drain, lease loss의 의미를 복구 뒤에도 유지한다.
- 기존 in-process 실행과 v0.21 checkpoint를 계속 지원한다.
- base install에 PostgreSQL, Redis, Temporal 의존성을 추가하지 않는다.

## 2. 조사한 현재 구조

### 2.1 Checkpoint

`core/checkpoint.py`의 `RunCheckpoint`는 provider-neutral 상태 스냅샷이다. 복구는 마지막으로
commit된 스냅샷에서 시작한다. 현재 구현은 다음 특성을 가진다.

| 현재 동작 | 위치 | v0.22 영향 |
|---|---|---|
| snapshot restore 모델을 명시한다. | `core/checkpoint.py:8-10` | 전체 workflow replay를 도입하지 않는다. |
| `RunCheckpoint.to_json()`이 `dataclasses.asdict`를 사용한다. | `core/checkpoint.py:162` | 새 중첩 record를 추가하기 전에 명시적 encoder로 교체한다. |
| `CheckpointStore.put()`은 결과를 반환하지 않는다. | `core/checkpoint.py:473-503` | `committed`, `conflict`, `fenced`를 구분할 새 계약이 필요하다. |
| LocalFS는 같은 sequence의 manifest를 덮어쓴다. | `core/checkpoint.py:621-630` | 기존 store는 fenced store로 광고할 수 없다. |
| LocalFS는 낮은 sequence의 `LATEST` 회귀만 막는다. | `core/checkpoint.py:626-630` | 같은 sequence의 다른 content 충돌을 검출하지 못한다. |
| checkpoint는 park/settle 경계에서 주로 저장된다. | `loop.py:2755-2797` | 호출 도중 state transition은 별도 journal에 기록한다. |

`CHANGELOG.md`도 `asdict` 제거를 v0.22 durability 작업으로 남겨 두었다. 이 변경은 선행 작업이다.
새 invocation record와 result reference가 들어간 뒤 `asdict`의 재귀와 deep-copy 비용을 그대로
두면 checkpoint 크기와 깊이에 비례해 위험이 커진다.

### 2.2 Model call

`ModelCallRunner`는 receipt와 replay identity를 잘 제공하지만 crash-safe dispatch journal은 없다.

| 현재 동작 | 위치 | v0.22 영향 |
|---|---|---|
| loop 호출 주소는 `run_id + step_id/turn_id`로 안정적이다. | `loop.py:2448-2460` | `logical_call_id`의 입력으로 재사용한다. |
| `turn_id`는 `turn_0001` 형태의 monotonic step이다. | `loop.py:3682` | 마지막 park에서 복구하면 같은 다음 turn 주소를 다시 계산할 수 있다. |
| idempotency key는 adapter 호출 직전에 새로 발급된다. | `model_call.py:666-715` | key를 reserve record에 먼저 저장하도록 발급 위치를 이동한다. |
| caller가 넣은 key는 항상 덮어쓴다. | `model_call.py:670-674` | 기존 기본 동작을 유지하고 durable coordinator만 저장된 key를 주입한다. |
| `settled_sink` 예외는 항상 흡수된다. | `model_call.py:1010-1035` | 현재 의미를 `passive`로 고정하고 required/outbox 경로를 별도로 만든다. |
| 성공 turn은 runner가 반환한 뒤 loop state에 적용된다. | `loop.py:4114-4139` | durable settle을 이 적용보다 먼저 끝낸다. |
| replay corpus는 `ModelTurn.raw`를 제외한 결과를 이미 표현한다. | `core/model_payloads.py:527-574` | durable result codec의 필드 목록으로 재사용한다. |

현재 Gateway adapter는 `Idempotency-Key`를 전달한다. Reference gateway는 이 key로 provider 결과를
deduplicate하지 않는다. OpenAI adapter도 이 key의 exactly-once 효과를 선언하지 않는다. v0.22는
모든 shipped provider를 `idempotency unproven`으로 처리한다.

### 2.3 Outcome과 interruption

현재 종료 의미는 여러 값에 나뉘어 있다.

- `AgentRunResult.status`: `completed | failed | limited`
- `Suspension.reason`: `settled | awaiting_tasks | limited | terminal | turn_failed | interrupted | paused`
- `SessionState`: live와 terminal 상태를 함께 표현
- `CancellationToken`: requested boolean만 보존
- Reference backend drain: `cancel()`과 `error_code="cancelled"` 사용

이 구조에서는 user cancel, worker drain, lease loss를 복구 가능한 하나의 값으로 만들기 어렵다.
`TerminalOutcome`과 `InterruptionCause`가 이 조각들을 정규화한다.

### 2.4 Package와 import 경계

- base dependency에는 PostgreSQL, Redis, Temporal이 없다.
- `monoid_agent_kernel.hosting` namespace는 아직 없다.
- package root는 `contracts.__all__`의 158개 이름을 그대로 노출한다.
- `tests/test_public_surface.py`는 exact public surface와 optional/reference import 부재를 검사한다.
- conformance package는 pytest와 Reference 구현에 의존하지 않는 reusable contract 함수를 이미
  제공한다.

v0.22 core type만 `contracts`에 추가한다. Hosting type은 `monoid_agent_kernel.hosting`에서만
노출한다. package root import가 hosting, Reference, DBOS, Temporal을 불러오지 않게 유지한다.

## 3. Temporal과 LangGraph에서 가져올 원칙

### 3.1 Temporal

Temporal은 Activity가 여러 번 실행될 수 있다고 명시하고 idempotent Activity를 권장한다.
Worker가 외부 작업을 끝낸 뒤 Temporal Service에 완료를 보고하기 전에 죽으면 Activity가 다시
실행될 수 있다. Idempotency key의 실제 중복 방지는 Activity가 호출하는 외부 서비스가 맡는다.

출처:

- [Temporal Activity Definition — Idempotency](https://docs.temporal.io/activity-definition#idempotency)
- [Temporal Activity failure detection](https://docs.temporal.io/encyclopedia/detecting-activity-failures)
- [Temporal Workflow Definition](https://docs.temporal.io/workflow-definition)

v0.22 적용 원칙은 다음과 같다.

- 오케스트레이터가 provider의 exactly-once 실행을 대신 증명할 수 없다.
- provider가 key를 실제로 deduplicate한다는 capability와 conformance가 있어야 재호출을 허용한다.
- 증명이 없으면 `dispatch_started` 복구를 `dispatch_unknown`으로 끝낸다.
- Temporal adapter도 같은 `DurableModelInvocation`과 `TerminalOutcome`을 사용한다.

### 3.2 LangGraph

LangGraph는 task 결과와 graph state를 checkpoint하고 resume 시 저장된 task 결과를 재사용한다.
실패한 task는 재실행될 수 있으므로 API 호출과 side effect를 task에 넣고 idempotent하게 만들도록
요구한다. Interrupt는 node를 처음부터 다시 실행하므로 interrupt 전 side effect도 idempotency가
필요하다.

출처:

- [LangGraph Functional API — durable execution and idempotency](https://docs.langchain.com/oss/python/langgraph/functional-api)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)

v0.22 적용 원칙은 다음과 같다.

- 저장된 settled result는 provider 호출 없이 재사용한다.
- 저장되지 않은 외부 호출 결과를 성공으로 추정하지 않는다.
- 재실행 전에 request digest를 비교한다.
- full graph/node replay는 v0.22 범위에 넣지 않는다.

## 4. 핵심 구조 결정

### 4.1 일반 `RunCheckpoint`를 dispatch journal로 사용하지 않는다

현재 `RunCheckpoint`는 safe park snapshot이다. 모델 요청이 만들어진 중간 지점에는 다음 변화가
이미 발생한다.

- `session_step` 증가
- pending user input 제거
- user/tool observation을 `messages`에 추가
- 동적 context와 tool surface 계산
- wire용 multimodal payload 해석

이 시점의 checkpoint를 일반 restore 경로로 읽으면 다음 step부터 실행한다. 같은 request를 다시
만들려면 mid-step program counter, 동적 context 결과, tool surface, wire payload까지 저장해야 한다.
이 작업은 workflow-safe deterministic replay에 해당하며 v0.22 제외 범위와 충돌한다.

v0.22는 두 durability 층을 사용한다.

| 층 | 권위 데이터 | 저장 시점 |
|---|---|---|
| Semantic checkpoint | conversation, tool/task/workspace state, safe suspension | park/settle 경계 |
| Fenced run journal | invocation reserve/start/settle/unknown, event, terminal | 외부 effect 전후 |

`RunCheckpoint`에는 `last_model_invocation` optional field를 추가한다. 다음 safe checkpoint가 최신
invocation 요약을 운반한다. Crash-window 판정은 `FencedRunSink.load_invocation()`이 소유한다.

### 4.2 복구 알고리즘

마지막 safe checkpoint에서 같은 다음 `logical_call_id`를 계산한 뒤 journal을 조회한다.

이 알고리즘은 host가 admitted input을 같은 input ID로 다시 전달한다는 전제를 가진다. 현재
Reference backend는 queue/inbox metadata와 active input receipt로 이 역할을 일부 수행한다.
Production host의 atomic admission과 response-loss handoff는 v0.22 제외 범위다. Input이 없거나
달라지면 request digest 비교가 provider dispatch 전에 복구를 중단한다.

| Journal 상태 | 동작 |
|---|---|
| record 없음 | request를 만들고 reserve한다. |
| `reserved` | 저장된 request digest와 현재 digest를 비교하고 같은 key로 dispatch한다. |
| `dispatch_started` | `unknown`으로 전환하고 provider를 호출하지 않는다. |
| `settled` 성공 | 저장된 receipt와 `ModelTurn` result를 읽고 provider를 호출하지 않는다. |
| `settled` 실패 | 저장된 failure classification을 다시 표면화하고 provider를 호출하지 않는다. |
| `unknown` | `dispatch_unknown` outcome을 다시 반환한다. |

복구 중 request digest가 달라지면 `failed_terminal`과
`error_code="durable_invocation_request_conflict"`를 반환한다. Provider는 아직 호출하지 않는다.
Host는 code/config 차이를 확인한 뒤 새 run 또는 명시적 reconciliation을 선택한다.

Durable mode는 `request_digest`가 `ok`인 요청만 dispatch한다. Digest가 없거나 크기 제한을 넘으면
`error_code="durable_invocation_unkeyable"`로 dispatch 전에 멈춘다. 다른 요청을 같은 logical call로
재사용할 위험을 허용하지 않는다.

### 4.3 호출 순서

새 호출의 순서는 다음과 같다.

```text
build normalized request
  → derive logical_call_id
  → compute request_digest
  → issue idempotency_key
  → commit reserved
  → commit dispatch_started
  → invoke adapter
  → normalize receipt/result
  → commit settled + private result blob
  → deliver evidence according to policy
  → apply ModelTurn to AgentLoop state
```

`settled` commit과 private result blob은 하나의 sink mutation으로 취급한다. Blob이 먼저 저장되고
metadata commit이 실패한 경우 blob은 orphan이 된다. Metadata가 참조하기 전의 orphan은 실행
권위를 갖지 않는다.

### 4.4 ID 규칙

`logical_call_id`는 content hash가 아니다. 실행 주소에서 결정한다.

```text
logical_call_id = "mcall_" + sha256(
  canonical_json({
    "generation": "monoid.logical-model-call.v1",
    "run_id": invocation_context.run_id,
    "step_id": invocation_context.step_id
  })
)
```

- AgentLoop는 항상 non-empty `run_id`와 `step_id`를 제공한다.
- Standalone `ModelCallRunner`의 durable mode는 caller가 명시적 `logical_call_id`를 제공한다.
- `dispatch_id`는 logical call과 1-based kernel dispatch attempt에서 결정한다.
- adapter 내부 retry는 하나의 opaque kernel dispatch로 취급한다.
- `idempotency_key`는 첫 reserve에서 한 번 만들고 restore와 kernel retry에서 유지한다.
- raw run/step 주소는 durable ID 문자열에 직접 노출하지 않는다.

### 4.5 Provider idempotency

Provider 재조정 capability는 fail-closed probe로 둔다.

```text
none        같은 key의 provider deduplication을 증명하지 못함
native      adapter와 provider conformance가 같은 key의 결과 재조회/중복 억제를 증명함
```

v0.22의 Gateway, OpenAI, replay, fake adapter는 기본값 `none`을 사용한다. Header 전달만으로
`native`를 선언할 수 없다. `dispatch_started` 복구는 기본적으로 `dispatch_unknown`이다.
`native` recovery는 별도 conformance case를 통과한 third-party adapter에만 열린다.

`dispatch_started` 뒤 adapter가 예외를 반환한 경우도 dispatch evidence를 판정한다.

| Evidence | Invocation transition | Retry |
|---|---|---|
| Provider의 정상 response/terminal frame | `settled` success | 결과 재사용 |
| Provider가 반환한 명시적 HTTP/application refusal | `settled` failure | 기존 retry policy가 허용할 때 다음 dispatch attempt |
| Dispatch 전 local refusal | 현재 attempt를 시작하지 않음 | reserve에서 안전하게 재평가 |
| Connection drop, timeout, malformed terminal, 분류되지 않은 transport 예외 | `unknown` | 자동 retry 금지 |

기존 `retryable`은 “시간을 두면 성공할 가능성”을 뜻한다. Dispatch가 확실히 끝났다는 증거는 아니다.
Durable coordinator는 두 판단을 분리한다. `unknown`이 `retryable=true`보다 우선한다.
Adapter 내부 retry는 v0.22에서 opaque dispatch로 기록한다. Crash-induced 재호출은 차단하며, adapter
내부 retry 정책과 provider별 wire conformance 확대는 후속 M1에서 다룬다.

## 5. Public data contract

### 5.1 `TerminalOutcome`

새 파일: `src/monoid_agent_kernel/core/outcome.py`

```python
@dataclass(frozen=True)
class TerminalOutcome:
    schema_version: str
    run_id: str
    kind: Literal[
        "completed", "paused", "cancelled", "interrupted",
        "failed_retryable", "failed_config", "failed_terminal",
        "dispatch_unknown", "evidence_uncommitted",
    ]
    retry_eligibility: RetryEligibility
    interruption_cause: InterruptionCause | None = None
    checkpoint_seq: int | None = None
    final_output_ref: str = ""
    partial_output_ref: str = ""
    last_evidence_ref: str = ""
    error_code: str = ""
    provider_error_code: str = ""
    http_status: int | None = None
```

Outcome은 content를 inline으로 운반하지 않는다. Opaque address와 안전한 taxonomy를 운반한다.
기존 `AgentRunResult`, `AgentTurnResult`, `Suspension`에서 outcome을 만드는 helper를 제공한다.

`RetryEligibility` 값은 다음으로 고정한다.

| 값 | 의미 |
|---|---|
| `not_applicable` | 성공, pause처럼 retry 질문이 적용되지 않는다. |
| `safe` | 같은 logical call에서 외부 effect를 반복하지 않고 진행할 수 있다. |
| `after_configuration` | 설정 변경 뒤 같은 logical call을 재평가할 수 있다. |
| `after_reconciliation` | journal/provider 상태 확인 뒤 결정한다. |
| `forbidden` | 자동 retry를 허용하지 않는다. |

`InterruptionCause` 값은 다음으로 고정한다.

```text
user_cancel | graceful_drain | lease_lost | deadline
host_shutdown | provider_failure | validation_failure | unknown
```

### 5.2 `DurableModelInvocation`

새 파일: `src/monoid_agent_kernel/core/model_invocation.py`

최소 필드는 다음과 같다.

```python
@dataclass(frozen=True)
class DurableModelInvocation:
    schema_version: str
    run_id: str
    logical_call_id: str
    revision: int
    dispatch_id: str
    dispatch_attempt: int
    idempotency_key: str
    dispatch_state: Literal["reserved", "dispatch_started", "settled", "unknown"]
    request_digest: str
    digest_generation: str
    receipt: Mapping[str, Any] | None = None
    result_ref: str = ""
    failure_code: str = ""
```

추가 불변식은 다음과 같다.

- `reserved`와 `dispatch_started`에는 receipt와 result가 없다.
- 성공한 `settled`에는 receipt와 `result_ref`가 있다.
- 실패한 `settled`에는 receipt와 failure code가 있고 result가 없다.
- `unknown`은 자동 retry를 금지한다.
- `logical_call_id`, request digest, idempotency key는 lifecycle 중 바뀌지 않는다.
- 다음 dispatch attempt만 `dispatch_id`와 `dispatch_attempt`를 변경한다.
- `revision`은 logical call 안에서 1부터 연속 증가하며 load의 권위 순서를 정한다.
- receipt payload는 endpoint, prompt, response, raw exception message를 포함하지 않는다.
- result blob은 `ModelTurn.raw`를 제외하고 text, tool calls, reasoning, usage, stop reason을 보존한다.

Schema identifier는 `monoid.model-invocation.v1`로 둔다. Checked reader와 compatibility fixture를
제공한다.

### 5.3 Checkpoint 변화

`RunCheckpoint` tail에 다음 optional field를 추가한다.

```python
last_model_invocation: dict[str, Any] | None = None
interruption_cause: str = ""
```

Checkpoint schema identifier는 `monoid.checkpoint.v1`을 유지한다. 두 필드는 additive/defaulted다.
v0.21 fixture는 두 값의 기본값으로 복원한다.

`to_json()`은 hand-written field projection으로 교체한다. Reflection 기반 encoder를 새로 만들지
않는다. Checkpoint decoder의 field validators에도 두 필드를 추가한다.

## 6. Hosting contract

새 namespace:

```text
src/monoid_agent_kernel/hosting/
  __init__.py
  contracts.py
```

Package root와 `monoid_agent_kernel.contracts`는 hosting type을 re-export하지 않는다.

### 6.1 Types

```python
@dataclass(frozen=True)
class WriterToken:
    owner_id: str
    generation: int

@dataclass(frozen=True)
class CommitResult:
    status: Literal["committed", "already_committed", "conflict", "fenced"]
    sequence: int | None = None
    content_digest: str = ""
    winner_digest: str = ""

@dataclass(frozen=True)
class StorageCapabilities:
    single_writer: bool = False
    concurrent_writers: bool = False
    compare_and_set: bool = False
    lease_fencing: bool = False
    durable_checkpoints: bool = False
    durable_events: bool = False
    durable_invocations: bool = False
    terminal_first_writer_wins: bool = False
    transactional_outbox: bool = False
    cross_process_notify: bool = False
```

`WriterToken`은 credential이 아니다. Host의 owner ID와 monotonic lease epoch를 운반한다. 만료와
현재 owner 판정은 storage mutation 안에서 host adapter가 수행한다.

### 6.2 Protocol shape

```python
class FencedCheckpointStore(Protocol):
    capabilities: StorageCapabilities

    def commit_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult: ...

    def latest_checked(self, run_id: str) -> DurableLoadResult[CheckpointRecord]: ...


class FencedRunSink(FencedCheckpointStore, Protocol):
    def load_invocation(
        self, run_id: str, logical_call_id: str
    ) -> DurableLoadResult[DurableModelInvocation]: ...

    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult: ...

    def append_event(
        self, event: AgentEvent, *, writer_token: WriterToken
    ) -> CommitResult: ...

    def settle_terminal(
        self, outcome: TerminalOutcome, *, writer_token: WriterToken
    ) -> CommitResult: ...
```

`FencedRunSink`는 run journal 전체를 구현하는 host를 위한 합성 protocol이다.
`FencedCheckpointStore`는 checkpoint만 공유 저장소에 두는 host가 사용할 수 있다.

### 6.3 Commit 판정 순서

모든 mutation은 다음 순서로 판정한다.

1. writer token을 현재 owner/generation과 비교한다.
2. stale token이면 `fenced`를 반환하고 아무 mutation도 수행하지 않는다.
3. 같은 resource key와 같은 canonical content면 `already_committed`를 반환한다.
4. 같은 resource key와 다른 content면 `conflict`를 반환한다.
5. 새 mutation을 commit하고 `committed`를 반환한다.

Stale writer의 identical retry도 `fenced`다. Fencing 판정이 idempotency 판정보다 먼저다.

Resource key는 다음과 같다.

| Mutation | Resource key |
|---|---|
| checkpoint | `(run_id, checkpoint.seq)` |
| event | `(run_id, event.seq)` |
| invocation | `(run_id, logical_call_id, revision)` |
| terminal | `(run_id, "terminal")` |

Terminal의 첫 committed content가 winner다. 같은 winner 재전송은 `already_committed`, 다른 content는
`conflict`다.

Invocation transition은 다음 순서를 검증한다.

```text
attempt N reserved
  → attempt N dispatch_started
  → attempt N settled | unknown
  → attempt N+1 reserved       # settled failure가 retry policy를 허용한 경우만
```

`unknown`은 해당 logical call의 최종 journal 상태다. Revision gap, state regression, settled success
뒤 새 attempt는 `conflict`다.

### 6.4 LocalFS capability

`LocalFsCheckpointStore.capabilities`는 다음 값을 선언한다.

```text
single_writer=true
durable_checkpoints=true
compare_and_set=false
lease_fencing=false
durable_events=false
durable_invocations=false
terminal_first_writer_wins=false
```

LocalFS는 기존 `CheckpointStore`를 계속 구현한다. `FencedCheckpointStore`를 구현한다고 광고하지
않는다. Multi-process durable mode는 startup capability 검사에서 LocalFS를 거부한다.

## 7. AgentLoop와 ModelCallRunner 연결

### 7.1 새 opt-in 구성

`AgentLoop`에 다음 optional 구성을 tail field로 추가한다.

```python
run_sink: FencedRunSink | None = None
writer_token: WriterToken | None = None
model_evidence_policy: Literal["passive", "required", "outbox"] = "passive"
```

Validation 규칙:

- `run_sink`와 `writer_token`은 함께 제공한다.
- `required`와 `outbox`는 `durable_invocations=true`를 요구한다.
- `outbox`는 `transactional_outbox=true`를 요구한다.
- fenced mode는 `lease_fencing=true`를 요구한다.
- 기존 `checkpoint_persist_callback`과 fenced checkpoint writer를 동시에 구성하면 시작 전에
  설정 오류로 거부한다.
- 기존 `checkpoint_store` 경로는 구성 변화 없이 동작한다.

Runtime import 순환을 막기 위해 `loop.py`는 hosting type을 `TYPE_CHECKING`에서만 import한다.
실행 시 protocol method를 직접 호출한다.

### 7.2 Runner hook

`ModelCallRunner`에는 내부 dispatch lifecycle hook을 추가한다. Hook은 다음 순간에 호출된다.

1. request normalize와 digest 계산 뒤 `reserve`
2. adapter method 진입 직전 `dispatch_started`
3. explicit provider refusal 뒤 settled failure commit, transport ambiguity 뒤 unknown commit
4. final success/failure normalization 뒤 `settled`

Hook이 없는 runner는 기존 방식으로 key를 발급하고 기존 동작을 유지한다. Hook이 있는 runner는
reserve record의 key를 request에 넣는다.

Adapter 내부 retry는 hook 밖의 한 dispatch다. `provider_retried`와 receipt attempt evidence가 그
사실을 보존한다. Kernel retry는 attempt별 `dispatch_id`를 만든다.

### 7.3 Result 재사용

`settled` success를 복구하면 private result blob을 `ModelTurn`으로 복원한다. 복원기는 다음을
검증한다.

- 모든 필수 field 존재
- tool call ID/name/arguments shape
- reasoning item은 object
- usage는 non-negative integer map
- stop reason shape
- blob sha256 일치
- `raw={}` 강제

복원된 turn은 live adapter가 반환한 turn과 같은 normalize path를 거친다. 이후 usage 누적,
assistant message append, tool call 실행은 기존 loop 경로를 사용한다.

## 8. Sink delivery policy

현재 `settled_sink`의 의미는 `passive`다. 기존 callable과 failure containment를 유지한다.

| Policy | 동작 | Provider 재호출 |
|---|---|---|
| `passive` | observer/sidecar 실패를 log하고 model outcome을 유지 | 없음 |
| `required` | durable invocation settle 뒤 evidence commit 결과를 확인 | 없음 |
| `outbox` | invocation settle transaction에서 evidence outbox entry를 stage | 없음 |

Authoritative invocation settle과 evidence projection을 구분한다.

- Invocation settle 실패: paid call의 durable 상태를 확정하지 못했으므로 crash-safe path는
  `dispatch_unknown`이다.
- Invocation settle 성공 + required evidence 실패: `evidence_uncommitted`다.
- `evidence_uncommitted` 복구: settled invocation result를 재사용하고 evidence delivery만 다시 한다.
- Provider failure + evidence failure: settled failure receipt를 재사용하고 evidence delivery만 다시 한다.

기존 반환형을 유지하기 위해 AgentLoop는 `evidence_uncommitted`를 non-terminal
`Suspension(reason="turn_failed", error_code="evidence_uncommitted")`로 표면화한다.
`TerminalOutcome` conversion helper가 이를 `kind="evidence_uncommitted"`와
`retry_eligibility="safe"`로 정규화한다. 다음 drive는 provider를 호출하지 않고 settled record를
읽은 뒤 sink delivery를 재시도한다.

## 9. Typed interruption

`CancellationToken.cancel()`을 다음처럼 확장한다.

```python
def cancel(self, cause: InterruptionCause = InterruptionCause.USER_CANCEL) -> None: ...
```

첫 cause가 승리한다. 후속 cancel은 event를 다시 set하지 않고 cause를 덮어쓰지 않는다.
기존 no-argument caller는 `user_cancel` 의미를 유지한다.

처리 규칙은 다음과 같다.

| Cause | Loop 동작 | Durable write |
|---|---|---|
| `user_cancel` | terminal cancelled | 현재 writer token으로 terminal settle 시도 |
| `graceful_drain` | partial state를 safe park로 만들고 interrupted 반환 | checkpoint 허용, terminal 정책은 host 결정 |
| `host_shutdown` | graceful drain과 같은 handoff 기본값 | checkpoint 허용 |
| `lease_lost` | 즉시 실행 중단, in-memory interrupted outcome 반환 | checkpoint/event/terminal 금지 |
| `deadline` | timed-out terminal outcome | 현재 writer token이 유효할 때만 settle |

`Suspension`, `AgentTurnResult`, `AgentRunResult`, `RunState`, checkpoint의 additive tail field로 cause를
운반한다. 기존 wire reader는 absent cause를 `unknown` 또는 기존 cancel 문맥에서 `user_cancel`로
해석한다.

Reference backend의 `drain()`은 `graceful_drain`을 사용한다. 외부 `cancel_run`과
`LoopSession.cancel()`은 `user_cancel`을 사용한다. Lease watcher를 가진 host adapter는
`lease_lost`를 사용한다.

## 10. Deterministic conformance

### 10.1 Reusable contract functions

`monoid_agent_kernel.conformance.contracts`에 다음 함수를 추가한다.

```python
run_fenced_checkpoint_store_contract(factory, root)
run_fenced_run_sink_contract(factory, root)
run_durable_model_invocation_contract(harness)
```

함수는 pytest에 의존하지 않는다. 반환형은 기존 `ConformanceRuleOutcome`을 사용한다.
새 profile ID는 다음과 같다.

```text
fenced-checkpoint-store-contract
fenced-run-sink-contract
durable-model-invocation-contract
```

### 10.2 Crash failpoint matrix

테스트 harness는 barrier를 명시적으로 열어 다음 위치에서 drive를 멈춘다.

| Failpoint | 기대 결과 | Adapter call 수 |
|---|---|---:|
| reserve commit 전 | record 없음, restore가 새 reserve 생성 | 0 또는 restore 후 1 |
| reserve commit 후 dispatch 전 | 같은 logical ID/key로 진행 | 전체 1 |
| dispatch_started commit 후 adapter 진입 전 | `dispatch_unknown` | 0 |
| adapter 반환 후 settled commit 전 | `dispatch_unknown` | 1 |
| transport 예외 반환 | `dispatch_unknown` | 1 |
| explicit provider refusal 반환 | settled failure 재사용 또는 다음 proven retry | policy 기준 |
| settled commit 후 loop state 적용 전 | 저장된 turn 재사용 | 1 |
| settled failure commit 후 surface 전 | 저장된 failure 재사용 | 1 |
| required evidence commit 실패 | `evidence_uncommitted`, sink만 재시도 | 1 |

`sleep`으로 race를 만들지 않는다. Fake clock과 named barrier를 사용한다.

### 10.3 Fencing matrix

필수 case:

1. current token checkpoint commit → `committed`
2. 같은 token/sequence/body 재시도 → `already_committed`
3. 같은 token/sequence/different body → `conflict`
4. stale token/same body → `fenced`
5. stale token/new event → `fenced`
6. stale token/terminal → `fenced`
7. terminal same winner retry → `already_committed`
8. terminal different winner → `conflict`
9. newer generation checkpoint/event/invocation이 같은 generation을 사용
10. drain barrier 뒤 lease generation 교체 → 이전 worker terminal 없음

### 10.4 Outcome race matrix

- completion이 먼저 settle되면 후속 Stop은 같은 terminal을 바꾸지 않는다.
- Stop이 먼저 settle되면 후속 completion은 conflict다.
- graceful drain 뒤 lease loss가 오면 stale worker는 terminal을 쓰지 않는다.
- `lease_lost`는 cancelled나 failed로 자동 투영되지 않는다.
- provider failure와 required sink failure가 서로의 taxonomy를 덮지 않는다.

## 11. Compatibility와 release gate

### 11.1 Versioned artifact

Compatibility registry에 다음 row를 추가한다.

| Key | Kind | Writer | Reader |
|---|---|---|---|
| `terminal-outcome` | wire | `monoid.terminal-outcome.v1` | strict |
| `model-invocation` | durable | `monoid.model-invocation.v1` | checked |

Checkpoint는 기존 `monoid.checkpoint.v1` row를 유지한다. Fixture에는 v0.21 checkpoint와 v0.22
additive checkpoint를 모두 둔다.

### 11.2 Import와 package test

다음 조건을 자동 검사한다.

- `import monoid_agent_kernel`이 `monoid_agent_kernel.hosting`을 load하지 않는다.
- `import monoid_agent_kernel.hosting`이 Reference, DBOS, Temporal, psycopg, redis를 load하지 않는다.
- `import monoid_agent_kernel.conformance`가 Reference를 load하지 않는다.
- base dependency 목록이 늘지 않는다.
- wheel에 hosting과 새 codec은 포함되고 플랫폼 library는 포함되지 않는다.
- root `__all__`에는 core outcome/invocation type만 추가된다.
- hosting type은 root `__all__`에 없다.

### 11.3 Test 명령

모든 pytest 실행은 저장소 표준 prefix를 사용한다.

```powershell
.\.venv\Scripts\python -m pytest -q -n auto --dist=worksteal `
  --timeout=120 --timeout-method=thread `
  tests/test_outcome.py tests/test_model_invocation.py tests/test_checkpoint.py

.\.venv\Scripts\python -m pytest -q -n auto --dist=worksteal `
  --timeout=120 --timeout-method=thread `
  tests/test_model_call_runner.py tests/test_agent_loop_lifecycle.py `
  tests/test_cancellation.py tests/test_session_lifecycle.py

.\.venv\Scripts\python -m pytest -q -n auto --dist=worksteal `
  --timeout=120 --timeout-method=thread `
  tests/test_fenced_hosting.py tests/conformance tests/test_public_surface.py `
  tests/test_compatibility_ledger.py tests/test_release_packaging.py

.\.venv\Scripts\python -m pytest -q -n auto --dist=worksteal `
  --timeout=120 --timeout-method=thread
```

Fresh install 뒤 첫 parallel run의 cold-cache timeout은 warm rerun으로 확인한다.

## 12. 파일 변경표

### 새 파일

| 파일 | 책임 |
|---|---|
| `core/outcome.py` | outcome/cause/retry type, strict codec, conversion helper |
| `core/model_invocation.py` | durable invocation record와 checked codec |
| `hosting/__init__.py` | 좁은 hosting export |
| `hosting/contracts.py` | writer token, capabilities, fenced protocols, commit result |
| `tests/test_outcome.py` | outcome invariant와 conversion |
| `tests/test_model_invocation.py` | lifecycle/codec/result reference |
| `tests/test_fenced_hosting.py` | protocol과 deterministic fake sink |
| `tests/conformance/test_fenced_hosting_contract.py` | reusable conformance 실행 |
| `tests/conformance/test_durable_invocation_contract.py` | crash matrix 실행 |

### 수정 파일

| 파일 | 변경 |
|---|---|
| `core/checkpoint.py` | `asdict` 제거, additive fields, validators |
| `core/cancellation.py` | first-writer interruption cause |
| `core/result.py` | additive cause와 outcome conversion 입력 |
| `core/lifecycle.py` | typed cause mapping |
| `core/compatibility.py` | 두 artifact row |
| `providers/base.py` | provider idempotency capability probe |
| `model_call.py` | durable lifecycle hook, pre-dispatch key reuse, required delivery result |
| `loop.py` | run sink 구성, lookup/reuse, cause propagation, safe outcome |
| `loop_phases.py` | result cause와 terminal conversion 입력 보존 |
| `contracts.py` | core type export만 추가 |
| `conformance/contracts.py` | fenced/invocation reusable contract |
| `conformance/__init__.py` | 새 contract function export |
| `reference/backend/service.py` | drain과 cancel cause 분리 |
| `docs/CONTRACTS.md` | public semantics와 failure precedence |
| `docs/EMBEDDING.md` | host wiring과 capability gate |
| `docs/CONFORMANCE.md` | profile과 failpoint matrix |
| `docs/COMPATIBILITY.md` | artifact ledger와 rollout |
| `docs/README.md` | 이 계획 링크 |
| `CHANGELOG.md` | v0.22 release notes |
| `tests/test_public_surface.py` | root/hosting import boundary |
| `tests/test_compatibility_ledger.py` | registry count와 constants |
| `tests/test_checkpoint_store_contract.py` | LocalFS capability와 legacy contract 유지 |

## 13. PR 순서

### PR 1 — Serialization foundation

- `TerminalOutcome`, `InterruptionCause`, `RetryEligibility`
- `DurableModelInvocation` codec
- compatibility rows와 fixtures
- `RunCheckpoint.asdict` 제거
- v0.21 checkpoint 호환 test

종료 조건: 새 type이 독립적으로 round-trip하고 기존 checkpoint fixture가 읽힌다.

### PR 2 — Hosting contracts

- `WriterToken`, `CommitResult`, `StorageCapabilities`
- `FencedCheckpointStore`, `FencedRunSink`
- LocalFS `single_writer` capability
- deterministic fake adapter와 fencing conformance

종료 조건: stale identical mutation도 `fenced`, same-key conflict와 terminal winner가 검증된다.

### PR 3 — Durable model invocation

- runner lifecycle hook
- reserve 전 key 발급과 저장된 key reuse
- AgentLoop logical call lookup
- settled result blob과 replay
- before/after dispatch crash test

종료 조건: 모든 crash matrix에서 provider call 수가 기대값과 일치한다.

### PR 4 — Sink policy

- passive 기존 동작 고정
- required delivery 결과
- outbox capability gate
- `evidence_uncommitted` mapping과 sink-only recovery

종료 조건: evidence failure가 provider 호출을 반복하지 않는다.

### PR 5 — Typed interruption

- cancellation cause round-trip
- user cancel, graceful drain, lease loss 분기
- Reference backend drain adoption
- Stop/completion/lease-loss race test

종료 조건: lease-lost worker가 checkpoint/event/terminal mutation을 만들지 않는다.

### PR 6 — Conformance와 release closure

- public conformance exports
- embedding/contract/compatibility 문서
- import, wheel, cold-start audit
- full test gate

종료 조건: v0.22 완료 조건과 release checklist가 모두 증거를 가진다.

각 PR은 독립적으로 green 상태를 유지한다. PR 3부터 기능은 opt-in이다. Default in-process 경로는
마지막 PR까지 기존 동작을 유지한다.

## 14. 주요 위험과 대응

| 위험 | 대응 |
|---|---|
| Mid-step checkpoint가 restore 위치를 잃음 | invocation journal과 semantic checkpoint를 분리한다. |
| Header 전달을 provider deduplication으로 오인 | capability를 fail-closed로 두고 shipped adapter는 `none`을 선언한다. |
| Settled result가 없어 provider를 다시 호출 | normalized turn blob을 invocation settle과 함께 commit한다. |
| Result blob에 raw provider data가 들어감 | `ModelTurn.raw`를 제외하는 명시적 projection을 사용한다. |
| Required sink 실패가 provider failure로 바뀜 | `evidence_uncommitted`를 별도 outcome으로 반환한다. |
| Stale identical write가 idempotent success로 통과 | fence를 digest/idempotency보다 먼저 검사한다. |
| Hosting API가 package root를 키움 | hosting namespace 전용 export를 사용한다. |
| 새 checkpoint 중첩 구조가 `asdict` 비용을 키움 | PR 1에서 hand-written encoder로 교체한다. |
| Adapter 내부 retry의 개별 wire attempt를 관찰할 수 없음 | 하나의 opaque dispatch로 기록하고 `provider_retried`를 보존한다. |
| Standalone anonymous call에 stable address가 없음 | durable mode에서 explicit logical call ID를 요구한다. |
| 복구할 admitted input이 없음 | host가 같은 input ID를 재전달해야 하며 digest mismatch는 dispatch 전에 중단한다. |
| `retryable` transport error를 definite failure로 오인 | explicit dispatch evidence가 없으면 `unknown`이 우선한다. |
| LocalFS를 multi-process durable store로 오인 | capability gate에서 시작 전에 거부한다. |
| Terminal과 local JSONL projection이 다름 | FencedRunSink terminal을 host authority로 문서화한다. |

## 15. 구현 시작 전 고정한 결정

다음 항목은 구현 중 다시 넓히지 않는다.

1. PostgreSQL/ObjectStore adapter는 v0.22에 넣지 않는다.
2. Temporal dependency와 adapter는 v0.22에 넣지 않는다.
3. Full workflow deterministic replay는 v0.22에 넣지 않는다.
4. Provider idempotency 증명이 없으면 `dispatch_started`를 재호출하지 않는다.
5. Invocation result는 private content로 취급하고 public event에 넣지 않는다.
6. Existing `settled_sink`의 기본 의미는 passive다.
7. Hosting type은 package root에서 re-export하지 않는다.
8. LocalFS는 single-writer store로 남는다.
9. Stale worker의 동일 payload 재전송도 `fenced`다.
10. `lease_lost` worker는 terminal을 settle하지 않는다.

이 결정으로 v0.22는 작은 production boundary release로 유지된다. Queue, worker supervisor,
database schema, browser reconnect, CSP projection은 host와 후속 adapter가 계속 소유한다.
