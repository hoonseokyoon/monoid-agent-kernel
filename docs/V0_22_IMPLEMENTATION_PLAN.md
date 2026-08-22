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
| loop 호출 주소는 `run_id + turn_id`로 안정적이다. | `loop.py:2448-2460` | `logical_call_id`의 입력으로 재사용한다. |
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
    "run_id": spec.run_id,
    "step_id": turn_id
  })
)
```

- AgentLoop는 checkpoint가 보존하는 `session_step`에서 `turn_id`를 만들고 `spec.run_id`와 함께
  logical-call 주소로 사용한다.
- Caller의 `InvocationContext.step_id`는 관찰용 provenance다. 복구 주소에는 포함하지 않는다.
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
        "completed", "paused", "limited", "cancelled", "interrupted",
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
`Suspension(reason="limited")`는 terminal `kind="limited"`와 `retry_eligibility="forbidden"`으로
투영한다. Cooperative pause와 task wait만 `kind="paused"`로 투영한다.

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
    evidence_policy: Literal["passive", "required", "outbox"] = "passive"
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
- `evidence_policy`는 첫 reservation에서 결정하고 모든 revision과 retry에서 유지한다.
- 같은 dispatch attempt 안에서 `dispatch_id`와 `dispatch_attempt`도 바뀌지 않는다.
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
plan: list[dict[str, Any]] = field(default_factory=list)
pending_finish: dict[str, Any] | None = None
pending_tool_loads: list[str] = field(default_factory=list)
```

Checkpoint schema identifier는 `monoid.checkpoint.v1`을 유지한다. 필드는 additive/defaulted다.
v0.21 fixture는 기본값으로 복원한다.

`to_json()`은 hand-written field projection으로 교체한다. Reflection 기반 encoder를 새로 만들지
않는다. Checkpoint decoder의 field validators에도 필드를 추가한다.

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
    run_id: str
    owner_id: str
    generation: int

@dataclass(frozen=True)
class CommitResult:
    status: Literal["committed", "already_committed", "conflict", "fenced"]
    sequence: int | None = None
    content_digest: str = ""
    winner_digest: str = ""

@dataclass(frozen=True)
class ModelInvocationRecord:
    revision: int
    invocation: DurableModelInvocation

    def blob(self, sha256: str) -> bytes: ...

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

`WriterToken`은 credential이 아니다. Run ID, host owner ID, monotonic lease epoch를 함께
운반한다. Run binding은 같은 owner와 generation으로 발급된 다른 run의 token 교환을 막는다.
만료와 현재 owner 판정은 storage mutation 안에서 host adapter가 수행한다. Adapter는 run ID,
owner ID, generation을 현재 authority와 각각 비교하며 세 값이 모두 정확히 일치할 때만 mutation을
허용한다.

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
    ) -> DurableLoadResult[ModelInvocationRecord]: ...

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
`ModelInvocationRecord`는 committed invocation revision과 private result blob reader를 함께
운반한다. Crash recovery는 `result_ref`의 digest로 `record.blob()`을 호출하고 metadata와 blob을
같은 권위 경계에서 복원한다.

Reusable conformance harness는 `set_current_writer(WriterToken)`으로 정확한 현재 token을 설치한다.
이 seam은 host의 실제 generation allocator를 호출하지 않는다. 따라서 per-run counter와 global
monotonic counter를 모두 허용하면서, 같은 owner/generation에서 run binding만 독립적으로 검증한다.
`reopen()`은 같은 backing store와 host authority를 새 sink facade로 연다. Contract는 재개방 뒤
checkpoint/blob, invocation/result blob, event identity, terminal winner를 다시 읽거나 재전송한다.
`inject_authoritative_load_fault(record_family, run_id, status, logical_call_id=...)`는 정상 commit으로
만든 권위 head를 backend raw mutation 또는 동등한 decoder hook으로 `corrupt`나
`unsupported_version` 상태로 바꾼다. Fault는 `reopen()` 뒤에도 유지된다. Contract는 checkpoint와
invocation 두 family에서 두 상태와 value 부재를 모두 확인해 `missing`으로 축약하거나 손상된 값을
노출하는 adapter를 거부한다.
Harness의 `read_event(run_id, seq)` seam은 재개방한 facade에서 event 전체 canonical payload를 읽는다.
Digest만 남기고 event payload를 버리는 adapter는 durable event capability를 선언할 수 없다.
Writer generation 전환 run에서는 `seq=1`과 `seq=2`를 모두 재개방 후 읽어 두 canonical payload를
각각 비교한다. Sequence별 CAS digest만 남기고 run별 최신 event payload 하나로 덮어쓰는 adapter를
거부한다.
`read_terminal(run_id)` seam도 terminal winner 전체 canonical payload를 읽어 first-writer digest만 남긴
구현을 거부한다.
Run-isolation probe는 같은 local key를 가진 두 run의 checkpoint, event, invocation, terminal 전체
canonical payload를 모두 읽는다. Checkpoint의 `final_text`와 invocation의 dispatch ID·request digest를
run마다 다르게 제출한다. Run binding과 CAS digest만 분리하고 나머지 payload를 전역 최신값으로
덮어쓰는 adapter도 이 비교에서 실패한다.
각 harness는 `close()`로 sink facade가 소유한 DB session, client, thread pool을 해제한다. Contract는
factory instance와 명시적으로 reopened된 모든 facade를 추적한다. 다음 독립 probe를 열기 전에 직전
probe의 facade group을 역순으로 닫고, `finally`에서 마지막 group을 닫는다. 이 수명 규칙이 database
session과 pooled connection의 최대 동시 보유량을 제한한다. Lazy record blob reader는 해당 probe가
열려 있을 때 bytes 관찰로 materialize한다. Close-sensitive reference harness가 닫힌 facade의 reader를
사용하는 회귀를 거부한다.
Checkpoint와 invocation은 선택 필드가 아니라 committed canonical payload 전체의 digest를
재개방 record와 비교한다.
같은 key에 다른 well-formed blob map을 제출한 CAS loser는 `conflict`이며 blob도 공개하지 않는다.
Contract는 재개방 뒤 loser 전용 digest를 빈 map으로 참조하는 새 checkpoint와 invocation lifecycle을
시도해 둘 다 `conflict`인지 확인한다. Metadata CAS 전에 blob을 게시하는 adapter는 이 probe에서 실패한다.
같은 key의 checkpoint, event, terminal, invocation 재시도는 mutable canonical non-key field를
하나씩 독립적으로 변경해 모두 `conflict`인지 확인한다. Matrix의 field 집합은 각 record의
canonical JSON field에서 계산하며, 스키마 확장 뒤 variant가 빠지면 contract가 즉시 실패한다.
Invocation의 schema version과 digest generation은 current canonical tag로 고정한다. 허용된 legacy
alias와 current tag 사이의 같은 revision 재시도는 양방향 모두 canonical payload가 같으므로
`already_committed`다. Legacy alias를 쓴 다음 current-tagged legal revision과 current tag 뒤
legacy-tagged legal revision은 canonical stable identity가 같으므로 모두 `committed`다.
Checkpoint의 schema version과 `last_model_invocation` 안의 schema version/digest generation도 current
writer shape로 정규화한다. 세 alias 위치의 current→legacy·legacy→current 같은-sequence 재시도는 모두
`already_committed`다.
TerminalOutcome의 허용된 legacy schema도 current writer shape로 정규화한다. Current→legacy와
legacy→current의 같은 winner 재시도는 모두 `already_committed`다.
Contract 실행마다 UUID 기반 namespace를 만들고 모든 run ID와 invocation idempotency key에
적용한다. 같은 durable test service에서 반복 실행해도 이전 conformance artifact와 충돌하지 않는다.

Contract는 `race_conflicting_writes` harness hook으로 두 worker를 backend의 CAS read/publication
gap에서 조정하고 checkpoint, event, invocation, terminal의 conflicting content를 각각 경쟁시킨다.
Method 호출 시작만 맞추는 barrier는 이 seam을 충족하지 않는다. 각 경쟁은 정확히 하나의
`committed`와 하나의 `conflict`를 만들며, 재개방 뒤 winner retry는 `already_committed`, loser retry는
`conflict`다. 두 worker는 같은 backing store를 가리키는 별도 sink facade를 사용한다. Hook은 내부에서
연 facade를 직접 닫는다. Backend CAS를 검증하면서 database session이나 client facade의 cross-thread
안전성을 요구하지 않는다.
별도 writer-handoff race는 stale mutation과 generation rotation을 함께 시작한다. Rotation이 먼저
직렬화되면 stale write는 `fenced`다. Write가 먼저 직렬화되면 stale write가 `committed`다. 네 mutation
모두 stale/current payload가 다르므로 write-first의 current retry는 `conflict`다. Blob mutation의
stale callback과 current callback은 서로 다른 digest를 제출한다. Rotation-first 경로에서는 stale
전용 digest를 빈 map으로 참조할 수 없어야 하며,
write-first 경로에서는 같은 참조가 authoritative backing에서 해소되어야 한다. Current callback은
stale 전용 bytes를 제출하지 않는다. 이 probe가 metadata fencing 밖의 stale blob publication을
검출한다.
네 mutation의 handoff는 stale/current canonical payload를 서로 다르게 제출하고, 재개방한 record
전체를 linearization이 선택한 값과 비교한다. 올바른 status를 반환한 뒤 metadata만 stale·loser
값으로 덮어쓰는 adapter도 실패한다. Checkpoint와 invocation은 winner blob bytes까지 함께 비교한다.
Contract는 같은 owner가 generation만 갱신하는 lease renewal과 owner와 generation이 함께 바뀌는
reassignment를 네 mutation에서 각각 경쟁시킨다. 정적 authority matrix는 현재 owner의 stale
generation과 현재 generation의 잘못된 owner를 독립적으로 제출한다. Matrix는 existing resource와
fresh resource를 별도 run에서 검증하고, 거부 뒤 current token의 정상 commit까지 확인한다. Existing
resource는 stale token으로 같은 payload, 다른 payload, malformed blob map을 각각 제출한다. 세 경우
모두 content equality, conflict, blob validation보다 fence 판정이 먼저 실행되어 `fenced`다.
Fresh resource도 stale owner+generation, current owner+stale generation, wrong owner+current generation의
세 token에 malformed checkpoint·invocation map을 각각 교차한다. Insert 경로도 blob validation보다
fence 판정을 먼저 실행한다. 각 malformed map은 mutation별 고유한 valid stale-only blob도 함께
제출한다. 재개방 뒤 current token의 checkpoint와 invocation empty-map 참조가 모두 `conflict`여야 한다.
따라서 `fenced`를 반환하면서 blob을 먼저 게시하는 adapter도 실패한다.
Checkpoint race는 referenced workspace blob을 포함하고 invocation race는 reserved/start history 뒤
settled-success result blob을 포함한다. CAS와 writer-handoff가 끝난 뒤 재개방 record에서 정확한
bytes를 읽는다. CAS race는 `committed` 결과를 낸 좌·우 값에서 winner를 결정하고 네 mutation의
재개방 canonical payload 전체가 그 winner인지 비교한다. Metadata fencing 밖에서 stale blob 또는
loser payload를 공개하는 adapter는 이 검증을 통과하지 못한다.

### 6.3 Commit 판정 순서

모든 mutation은 다음 순서로 판정한다.

1. writer token의 run binding을 target run과 비교한 뒤 현재 owner/generation과 비교한다.
2. stale token이면 `fenced`를 반환하고 아무 mutation도 수행하지 않는다.
3. 같은 resource key와 같은 canonical content면 `already_committed`를 반환한다.
4. 같은 resource key와 다른 content면 `conflict`를 반환한다.
5. 새 mutation을 commit하고 `committed`를 반환한다.

Stale writer의 identical retry도 `fenced`다. Fencing 판정이 idempotency 판정보다 먼저다.
Stale owner+generation, current owner+stale generation, wrong owner+current generation, 다른 run token에
malformed checkpoint/invocation blob map을 각각 결합해도 `fenced`다. Blob digest 검증은 run binding과
owner/generation 검증 뒤에만 실행한다.

Resource key는 다음과 같다.

| Mutation | Resource key |
|---|---|
| checkpoint | `(run_id, checkpoint.seq)` |
| event | `(run_id, event.seq)` |
| invocation | `(run_id, logical_call_id, revision)` |
| terminal | `(run_id, "terminal")` |

Contract는 같은 harness에서 run A와 run B를 모두 authorize한 뒤 동일한 local coordinate를 각 run에
독립 commit한다. 두 run의 checkpoint/event `seq=1`, invocation `call-1/revision=1`, terminal은 모두
`committed`이고 재개방 뒤 각각 `already_committed`다. Checkpoint와 invocation load도 요청한 run의
record를 반환한다. 이 검증은 네 resource key에서 `run_id`가 빠지는 adapter를 거부한다.

Terminal의 첫 committed content가 winner다. 같은 winner 재전송은 `already_committed`, 다른 content는
`conflict`다.

`CommitResult.sequence`, `content_digest`, `winner_digest`는 선택 evidence다. Adapter가 값을 채우면
resource coordinate, canonical submitted digest, canonical winner digest와 정확히 일치해야 한다.
Contract는 네 mutation family의 `committed`, `already_committed`, `conflict` 대표 결과에서 세 필드를
각각 검증한다. Winner가 없는 status의 `winner_digest`는 비어 있어야 한다.

Canonical commit digest는 아래 값을 `canonical_sha256`으로 계산한다.

```python
{
    "record": canonical_payload,
    "blobs": {
        key: sha256(blob_bytes)
        for key, blob_bytes in sorted(submitted_blobs.items())
    },
}
```

Checkpoint의 `canonical_payload`는 current writer shape로 정규화한 payload다. Event, invocation,
terminal은 각각 current canonical `to_json()` payload를 사용한다. `winner_digest`도 winner가 commit될
때의 record와 blob 집합에 같은 계산을 적용한다.

Checkpoint와 invocation의 canonical content에는 함께 제출된 blob key와 byte digest가 포함된다.
Checked load record는 committed blob을 정확한 bytes로 돌려준다.
같은 metadata에 다른 blob key를 추가하거나 같은 key의 bytes를 바꾸면 `conflict`다.
모든 blob key는 제출 bytes의 lowercase SHA-256이다. 새 resource에 잘못된 key/bytes 조합을 제출하면
`conflict`이고 metadata, head, blob을 공개하지 않는다. Contract는 checkpoint와 invocation에서 이
거부 뒤 재개방한 durable state를 직접 검사한다. 같은 resource를 올바른 bytes로 다시 제출하면
`committed`이고 재개방 record가 정확한 bytes를 반환한다.
대문자 digest key에 올바른 bytes를 넣은 경우도 `conflict`다. Contract는 이 case-folding 결함을 별도
mutant로 검증하며, 거부된 map의 bytes가 backing에 남지 않았음을 빈-map 참조 재시도로 확인한다.
Checkpoint workspace delta, committed message log, queued raw content list, queued inbox envelope의
media `blob:` reference와 nested `last_model_invocation`의 `blob:` result reference는 제출 map이나 같은
run의 authoritative backing으로 해소된 뒤에만 commit할 수 있다. Standalone invocation의 `blob:` result도
같은 규칙을 따른다. 각 fresh missing-reference commit은 `conflict`이고 head를 유지한다. 같은 run에서 먼저
저장한 blob을 새 map 없이 참조하는 checkpoint와 invocation은 commit되며 재개방 record가 그 bytes를
반환한다. 올바른 bytes를 제출한 missing-reference 재시도도 commit된다.
`blob:` suffix는 정확한 lowercase SHA-256 digest다. Workspace `content_sha256`, committed·queued
message media `source_ref`, invocation `result_ref`의 malformed suffix는 모두 `conflict`이고 metadata와
head를 공개하지 않는다. `object:` 같은 bounded external invocation result address는 blob map 없이
commit되며 재개방한 canonical payload에 그대로 남는다.
Run A에만 존재하는 digest를 run B가 빈 map으로 참조하면 checkpoint와 invocation 모두 `conflict`다.
Content-addressed blob namespace는 run 권위 경계를 포함한다.
Run A token으로 run B payload와 valid blob을 제출한 fenced write는 run B backing에 bytes를 공개하지
않는다. Run B current token의 같은 digest empty-map 참조가 계속 `conflict`인지 checkpoint와 invocation
각각에서 확인한다.
Contract는 먼저 같은 run에서 같은 digest를 참조하는 별도 valid record를 seed한다. Malformed write
직후 repair 전에 seed record의 blob과 authoritative head를 다시 읽어 run-scoped blob 저장소에서도
기존 content-addressed row가 바뀌지 않았고 malformed metadata가 공개되지 않았음을 검증한다.
성공한 invocation의 contract fixture는 `result_ref`가 가리키는 result blob을 같은 commit에 제출한다.
Checkpoint는 높은 sequence가 먼저 도착해도 아직 비어 있는 낮은 `(run_id, seq)` 좌표를 blob과 함께
commit한다. 같은 지연 write의 재시도는 `already_committed`이고, latest head는 높은 committed
sequence를 유지한다.

Invocation transition은 다음 순서를 검증한다.

```text
attempt N reserved
  → attempt N dispatch_started
  → attempt N settled | unknown
  → attempt N+1 reserved       # settled failure가 retry policy를 허용한 경우만
```

`unknown`은 failure code 유무와 관계없이 해당 logical call의 최종 journal 상태다. Revision gap,
state regression, settled success 뒤 새 attempt는 `conflict`다. Successful settlement의 receipt에
`retryable=true`가 있어도 성공 결과가 우선하며 새 paid dispatch를 열지 않는다.
Settled failure receipt에서 `retryable` 필드가 생략되면 재시도 근거가 없다. Contract는 명시적인
`retryable=false`, 필드 생략, `retryable=true`를 별도 상태로 검증하며 마지막 경우만 다음 paid
dispatch reservation을 허용한다.
세 retryability 상태는 receipt content identity에서도 서로 다르다. 같은 revision의 세 상태를 모든
6개 방향으로 비교해 `conflict`를 요구하고, 재개방한 canonical payload가 원래 winner인지 확인한다.
Core는 accepted top-level receipt field와 nested usage counter 집합을 공개한다. Contract fixture는 이
두 집합에서 자동 파생된다. 각 필드는 fresh settled failure로 정상 commit·reopen된 뒤 같은 revision의
alternate evidence와 `conflict`하고, 두 번째 reopen에서도 처음 winner payload를 유지해야 한다.
최신 head가 revision N이면 과거의 정확한 revision 재전송은 `already_committed`이며 head는 N을
유지한다. 같은 historical coordinate에 다른 canonical payload를 제출하면 `conflict`이고 head는 N을
유지한다. Contract는 revision 4 뒤 revision 1, 2, 3의 exact retry와 dispatch identity가 다른 retry를
각각 실행한다. Revision 3에는 stable identity와 legal settled transition을 유지한 채 receipt와
failure code만 각각 바꾸는 conflict도 추가한다. 모든 경우에 head 4가 유지되어야 한다.
같은 run 안의 여러 logical call은 `(run_id, logical_call_id)`마다 독립 head를 유지한다. 둘째 call을
commit한 뒤 첫째와 둘째를 모두 재로딩하며, 존재하지 않는 logical call은 `missing`과 빈 value를
반환한다. Per-run last-invocation pointer 구현은 이 복구 검증에서 실패한다.
첫 revision은 `reserved`만 허용한다. `reserved`, `dispatch_started`, `settled`, `unknown` 사이의
문서화되지 않은 13개 인접 edge는 각각 독립 history에서 `conflict`로 검증한다.
첫 record는 `(revision=1, dispatch_attempt=1, state=reserved)`만 허용한다. revision 2, attempt 2,
둘 다 2인 초기 좌표는 독립 history에서 각각 `conflict`다.
Retryable failure 뒤 reservation은 정확히 다음 attempt와 새 dispatch ID를 함께 사용한다. 같은
attempt, 같은 dispatch ID, 둘 다 같은 조합, 건너뛴 attempt는 각각 `conflict`다.
다음 attempt의 reservation은 logical call의 `idempotency_key`와 `request_digest`를 그대로 유지한다.
Contract는 두 필드를 각각 단독 변형한 후보를 `conflict`로 거부한다.
Stable invocation identity는 `reserved → dispatch_started`, `dispatch_started → settled`,
`dispatch_started → unknown`의 모든 legal edge에서 유지된다. Terminal 두 edge는 idempotency key,
request digest, dispatch ID, dispatch attempt를 각각 변형한 8칸 행렬로 `conflict`를 검증한다.
새 dispatch ID는 해당 logical call의 모든 이전 attempt와 달라야 한다. Contract는 두 번의 retryable
failure를 만든 뒤 attempt 3에서 attempt 1의 ID를 재사용하는 history를 `conflict`로 검증한다. 같은
history에서 fresh ID를 쓴 attempt 3의 reserved, dispatch_started, settled lifecycle은 모두
`committed`다.

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

`run_sink` mode의 checkpoint write는 `commit_checkpoint(..., writer_token=...)`를 사용한다.
`committed | already_committed`만 성공이다. `conflict | fenced`, 잘못된 반환형, sink exception은
checkpoint persistence failure로 처리하며 local checkpoint store로 우회하지 않는다. Restore의
checkpoint 선택과 writer token 발급은 host가 소유한다.

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

Hook은 저장소 protocol을 직접 노출하지 않는 동기식 runner 경계다. Runner는 실행 의미를 소유하고,
hook 구현은 host의 `FencedRunSink`, `WriterToken`, revision CAS를 소유한다. Runner가 hook에 전달하는
값은 다음 세 묶음으로 제한한다.

- reservation: logical call ID, dispatch attempt/ID, request digest generation/digest, 제안 key
- settlement: reservation, 공개-safe receipt projection, 성공 result blob 또는 failure code
- unknown: reservation과 safe failure code

`reserve`는 effective reservation을 반환한다. 새 호출이면 제안 key를 commit하고, 기존 `reserved`
record를 재개하면 저장된 key를 반환한다. Runner는 반환값에서 key만 바뀔 수 있도록 검사한다.
Logical call ID, dispatch coordinate, request digest와 generation의 drift는 provider dispatch 전에
거부한다. 첫 reserve 뒤 kernel retry는 같은 key를 제안하고 새 attempt에서 파생한 dispatch ID를
사용한다.

Provider failure evidence는 fail-closed다. 명시적인 terminal/refusal evidence만 settled failure로
기록한다. Connection drop, timeout, malformed terminal, 분류되지 않은 adapter exception은
`unknown`으로 기록하고 kernel retry를 중단한다. 기존 `retryable` 값은 dispatch 완료 증거로 사용하지
않는다. Shipped adapter가 증명하지 않은 failure도 `unknown`이다. Provider별 증거 확대는 M1
conformance에서 수행한다.

상태 전이 write는 passive observer보다 먼저 실행하며 실패를 흡수하지 않는다. `reserve` 또는
`dispatch_started` write가 실패하면 adapter에 진입하지 않는다. Adapter 진입 뒤 settled write가
실패하면 runner는 `unknown` write를 시도하고 `dispatch_unknown`을 표면화한다. Unknown write 자체가
실패해도 paid-call retry는 허용하지 않는다. Recovery는 남아 있는 `dispatch_started` head를
`unknown`으로 닫는다.

성공 settlement의 result blob은 `core.model_payloads`의 canonical recorded-turn projection을 재사용한다.
`raw`는 제외하고 text, tool calls, reasoning, usage, stop reason을 보존한다. Blob을 canonical하게
encode할 수 없거나 durable size bound를 넘으면 paid call을 `unknown`으로 닫는다. Passive
`settled_sink` 전달은 authoritative settlement 뒤에 실행된다.

Process crash를 표현하는 `BaseException`은 보상 전이를 실행하지 않는다. 정상 Python `Exception`만
settle 실패 보상과 unknown 전이를 수행한다. 따라서 named failpoint가 reserve/start/adapter-return
직후의 실제 journal head를 관찰할 수 있다.

Hook은 optional recovery query를 구현할 수 있다. AgentLoop는 안정적인 logical-call ID를 만들고,
Runner는 request normalize와 digest 계산 직후 query를 호출한다. Digest의 단일 소유자는 Runner다.
Hook은 `missing | reserved`에서 새 dispatch를 계속하고, `dispatch_started`를 unknown으로 닫으며,
`unknown`을 다시 표면화하고, `settled` evidence를 반환한다. PR3 hook 구현은 recovery method 없이
계속 동작한다.

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

복구 query의 request digest와 저장된 digest가 다르면 provider 진입 전에
`durable_invocation_request_conflict`로 끝낸다. Settled failure는 저장된 failure code, provider code,
retryability, config-recoverability, HTTP status, usage를 재생성해 같은 loop failure path로 보낸다.
Settled failure receipt는 attempt count와 `stream_committed`를 함께 보존한다. 현재 정책이 kernel
retry를 소유하고, 저장된 refusal이 retryable이며 config 변경이 필요 없고, attempt budget이 남고,
`stream_committed=false`가 명시된 경우에만 다음 dispatch attempt를 이어간다. 저장된 aggregate usage와
idempotency key를 유지한다. Delivery evidence가 없으면 stored refusal을 표면화한다.

Host hook은 result blob의 content digest와 canonical body shape를 검증한다. Public receipt에
보존된 stop reason은 result와 교차 검증한다. Receipt usage와 provider-retry evidence는 logical call
전체를 나타내고 private turn의 두 필드는 최종 provider turn을 나타내므로 서로 같다고 비교하지
않는다. Settled result를 반환하기 전에 권위 revision을 동일 content로 다시 commit해 현재 writer
fence를 검증한다. Run accounting은 public receipt의 canonical usage를 사용한다. 불일치, missing
blob, corrupt/unsupported load는 provider를 호출하지 않고 typed durable error로 끝내며 이미 지불된
receipt usage를 오류에 보존한다.

## 8. Sink delivery policy

현재 `settled_sink`의 의미는 `passive`다. 기존 callable과 failure containment를 유지한다.

`required`와 `outbox`는 기존 callable을 강화하지 않는다. `FencedRunSink`가 다음 두 mutation을
제공한다.

```python
commit_model_evidence(invocation, *, writer_token) -> CommitResult
commit_invocation(
    invocation,
    blobs,
    *,
    writer_token,
    stage_evidence: bool = False,
) -> CommitResult
```

`commit_model_evidence`는 이미 authoritative한 settled invocation의 public-safe projection을
idempotently 확정한다. `stage_evidence=True`는 invocation revision과 evidence outbox entry를 같은
transaction에서 확정한다. Outbox schema, poller, retry worker는 host adapter가 소유한다.

| Policy | 동작 | Provider 재호출 |
|---|---|---|
| `passive` | observer/sidecar 실패를 log하고 model outcome을 유지 | 없음 |
| `required` | durable invocation settle 뒤 evidence commit 결과를 확인 | 없음 |
| `outbox` | invocation settle transaction에서 evidence outbox entry를 stage | 없음 |

Authoritative invocation settle과 evidence projection을 구분한다.

- Invocation settle 실패: paid call의 durable 상태를 확정하지 못했으므로 crash-safe path는
  `dispatch_unknown`이다.
- Invocation settle 성공 + required evidence 실패: `evidence_uncommitted`다.
- `evidence_uncommitted` 복구: checkpoint의 logical call ID와 request digest로 evidence delivery를
  먼저 확정하고 settled invocation result를 재사용한다. 저장된 성공 또는 최종 거절을 loop state에
  적용한 뒤 현재 runtime config, context provider, tool surface, media를 조회한다. 최종 응답은 이
  지점에서 정산하고, tool call turn은 message log에 반영한 뒤 현재 tool 실행 환경을 구성한다.
  현재 runtime config, context provider, tool surface, media wire payload는 evidence commit의 입력이 아니다.
  첫 reservation은 invocation journal에 `evidence_policy="required"`를 저장한다. Settlement와 이 의무는
  같은 journal transaction에서 확정된다. Settlement commit 뒤 evidence commit 또는 checkpoint
  publication 전에 process가 종료되어도 새 activation은 journal field를 읽고 delivery를 완료한다.
  Journal field가 required이면 replacement config와 dynamic context로 다시 계산한 request digest를
  검증하기 전에 sink delivery를 먼저 완료한다. 이후 request drift는 result 적용을 중단할 수 있지만
  required evidence를 남겨두지 않는다.
  Recovery query의 `require_evidence` marker는 이미 저장된 evidence park에서 같은 의무를 보강한다.
  두 표식 모두 새 activation의 `passive` 설정보다 우선한다.
  기존 invocation의 journal policy가 `passive`이면 현재 activation 설정만으로 `required` 또는
  `outbox`로 승격하지 않는다. `outbox` reservation은 replacement activation의 passive 기본값보다
  우선하며 provider 재진입 전에 transactional outbox capability를 확인한다. Reservation 복구는
  provider 진입 전에, settled 복구는 evidence mutation 전에
  `durable_invocation_evidence_policy_conflict`로 거부한다. 강화된 정책은 새 logical call부터 적용한다.
- 복구된 assistant tool-call turn을 message log에 반영한 뒤 interrupt가 발생하면 suspension에
  `model_tool_calls_pending=true`를 저장한다. 같은 batch에서 완료된 tool observation은
  `pending_observations`에 누적한다. `None` resume은 같은 settled result를 다시 읽고 이미 완료된 call
  ID를 건너뛴 뒤 남은 call만 실행한다. Assistant turn은 중복 추가하지 않는다. 이 tool exchange가
  완료될 때까지 새 user input은 `evidence_recovery_requires_resume`으로 거부한다.
  Checkpoint는 context-owned `plan`, pending `run.finish`, pending `tool.search` load도 저장한다.
  Process restore는 이 상태를 먼저 복원한 뒤 완료된 call ID를 건너뛴다.
- `last_suspension=null`인 durable 내부 safety checkpoint는 이미 할당된 model step을 한 번 재사용한다.
  복구는 counter를 증가시키기 전에 같은 logical call journal을 조회한다. Step N settlement와 required
  evidence 의무가 있으면 먼저 delivery를 완료하고, missing head일 때만 새 dispatch를 진행한다.
  Approval replay consumption checkpoint 뒤 provider settlement와 evidence park 사이에서 process가
  종료되어도 step N invocation이 step N+1 뒤에 숨지 않는다.
- Provider failure + evidence failure: settled failure receipt를 재사용하고 evidence delivery만 다시 한다.

Required evidence failure는 authoritative invocation settle을 `unknown`으로 되돌리지 않는다.
Lifecycle bridge는 invocation commit 성공과 evidence commit 실패를 typed
`evidence_uncommitted`로 분리한다. Recovery는 현재 writer fence를 같은 invocation으로 다시 검증한
뒤 evidence mutation만 재시도한다. Passive `settled_sink`는 required/outbox mutation과 독립이며
기존처럼 결과 분류를 바꾸지 않는다. 최초 invocation settlement에서 passive observer와 활성화된
model-call sidecar에 authoritative call을 한 번 전달한다. Required evidence 실패 뒤 복구는 이 passive
전달을 반복하지 않는다. 성공한 provider stream은 live observer와 `model-content.jsonl`에서
`completed`로 닫고 normalized final text와 usage를 보존한다. Evidence projection 실패 중에도
성공한 provider stream은 settled 상태를 유지한다. Settled provider refusal은 failed stream 분류를
유지한다.

`run_once()`는 `evidence_uncommitted`를 일반 recoverable provider failure처럼 terminal로 승격하지
않는다. Committed checkpoint boundary를 release하고 `TurnNotSettled`를 표면화한다. 새 activation은
그 checkpoint를 restore한 뒤 `None`으로 resume해 sink-only recovery를 완료한다.

첫 `evidence_uncommitted` park는 저장된 receipt usage를 run total에 반영한다. 같은 logical call의
복구는 마지막 evidence-uncommitted checkpoint가 이미 반영한 usage를 읽고 이후 aggregate receipt와의
non-negative delta만 더한다. 같은 evidence 재시도, settled failure 재표면화, 남은 kernel attempt
자동 재개는 수행하지 않는다. Transcript와 public event도 같은 delta를 기록한다. 저장된 결과를
provider 호출 없이 적용한 행의 usage는 빈 mapping이다. Retryable settled failure는 그대로
표면화하고 다음 paid model step의 시작 여부를 driver가 결정한다.

모든 authoritative settled receipt는 assistant turn projection보다 먼저 `total_usage`와 cumulative
metrics에 반영한다. 그 다음 stop/deadline boundary를 검사한다. Settlement 직후 interrupt가 발생한
checkpoint도 paid usage를 포함하며 assistant message는 아직 포함하지 않는다. Resume은 저장된 결과를
적용하고 usage delta 0을 사용한다.

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
CancellationToken은 `user_cancel | graceful_drain | lease_lost | deadline | host_shutdown`만
받는다. provider/validation/unknown cause는 failure/outcome 분류에 남고 token ingress에서 거부한다.
Lease authority는 cause와 별도인 sticky 상태다. 후속 `lease_lost` 신호는 최초 cause를 보존하면서
모든 mutation fence를 활성화하고, stale activation의 반환 observation은 `lease_lost`를 사용한다.

처리 규칙은 다음과 같다.

| Cause | Loop 동작 | Durable write |
|---|---|---|
| `user_cancel` | terminal cancelled | 현재 writer token으로 terminal settle 시도 |
| `graceful_drain` | partial state를 safe park로 만들고 interrupted 반환 | checkpoint 허용, terminal 정책은 host 결정 |
| `host_shutdown` | graceful drain과 같은 handoff 기본값 | checkpoint 허용 |
| `lease_lost` | 즉시 실행 중단, in-memory interrupted outcome 반환 | usage/metric/observer/sidecar/checkpoint/event/projection/terminal 금지 |
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
| `hosting/contracts.py` | writer token, capabilities, fenced protocols, commit/result record |
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

- `WriterToken`, `CommitResult`, `ModelInvocationRecord`, `StorageCapabilities`
- `FencedCheckpointStore`, `FencedRunSink`
- LocalFS `single_writer` capability
- deterministic fake adapter와 fencing conformance

종료 조건: capability 선언, stale identical mutation의 `fenced`, in-flight writer handoff,
concurrent same-key CAS, 재개방 durability, 반복 실행 격리, terminal winner,
checkpoint/invocation private blob round-trip, malformed fresh blob 비공개, event 전체 payload 보존,
전체 attempt history의 dispatch ID 유일성이 검증된다.

### PR 3 — ModelCallRunner lifecycle

- runner lifecycle hook
- reserve 전 key 발급과 저장된 key reuse
- reserve, dispatch_started, settled, unknown transition
- dispatch evidence와 standalone failpoint test

종료 조건: Runner 단독 crash window에서 durable transition과 provider call 수가 기대값과 일치한다.

### PR 4 — AgentLoop recovery

- AgentLoop logical call lookup
- settled result blob과 replay
- request digest mismatch 차단
- before/after dispatch crash test
- fenced checkpoint commit과 invocation summary 연결
- durable hosting capability와 writer-token run binding 검증

종료 조건: 모든 crash matrix에서 provider call 수가 기대값과 일치하고, fenced checkpoint 실패가
local store로 우회하지 않으며, 손상된 invocation/result가 provider 진입 전에 차단된다.

### PR 5 — Sink policy

- passive 기존 동작 고정
- required delivery 결과
- outbox capability gate
- `evidence_uncommitted` mapping과 sink-only recovery
- 내부 safety checkpoint의 in-progress model step 재사용과 동일 journal coordinate 복구

종료 조건: evidence failure가 provider 호출을 반복하지 않고, park 전 crash도 settled step의 required
evidence를 다음 step 뒤에 남기지 않는다.

### PR 6 — Typed interruption

- cancellation cause round-trip
- user cancel, graceful drain, lease loss 분기
- Reference backend drain adoption
- Stop/completion/lease-loss race test

구현 계약:

- cancellation cause는 first-writer-wins이며 checkpoint, suspension, result, event, metrics,
  status projection을 같은 값으로 통과한다.
- `graceful_drain`과 `host_shutdown`은 checkpoint 가능한 non-terminal interrupt를 만든다.
  Reference backend는 admission barrier를 먼저 닫고, 이 park를 terminal close로 정리한다.
- `lease_lost` activation은 in-memory park만 반환하며 checkpoint, event, projection, terminal을
  쓰지 않는다. 신호 전에 commit된 model invocation settlement는 권위 evidence로 남는다.
  Settlement 반환 직후 lease authority를 다시 검사하고 stale activation의 usage accounting,
  receipt observer, model-call sidecar, model-stream close를 차단한다.
  Reference host는 suspension projection 전에 이 park를 감지한다. autonomous, streaming,
  recovery 실행은 activation을 discard하고 result/failure/stream frame/close를 생략하며,
  identity-matched local record를 해제해 stale status/heartbeat/recovery 차단을 끝낸다.
  `RunCancelled` handler도 exception의 과거 cause를 읽기 전에 sticky lease authority를 다시
  확인한다. model-call dispatch compensation과 stream close도 같은 현재 authority를 읽으며,
  stale cancellation의 usage stamp를 stale activation accounting으로 전달하지 않는다.
  output validator, tool handler, child agent await 뒤에는 run boundary를 다시 확인하고,
  model/usage/settle/tool projection gateway는 write authority를 자체 확인한다. outer pump는
  모든 exception과 반환된 park보다 sticky lease loss를 먼저 처리한다.
  세 checkpoint surface(`FencedRunSink`, host callback, `CheckpointStore`)는 snapshot/blob 수집
  뒤와 외부 commit 반환 직후 authority를 다시 확인한다. Reference host의 queue/inbox
  augmentation도 store 반환 뒤 outbox로 넘어가기 전에 같은 fence를 적용한다. outbox send와
  receipt projection 사이에 lease를 잃으면 request는 idempotency identity를 유지한 pending
  상태로 남아 replacement owner가 reconcile/redrive한다.
  receipt policy/subscriber/sidecar와 stream delta/outcome writer fan-out은 각 callback 전·후에
  authority를 다시 확인한다. 첫 callback과 겹친 lease loss는 이후 callback을 모두 중단하며,
  best-effort observer containment은 lease-loss control flow를 삼키지 않는다.
  turn settle과 run finalization은 하나의 common projection writer를 사용한다. proposal, metrics,
  settled text, event write를 각각 전·후 fencing하며 EventBus도 emit/close sink별로 authority를
  확인한다. TaskManager는 cancel_all 내부의 각 job cancellation 전·후에 같은 fence를 적용한다.
  drain은 admission quiesce, terminal 판정, cancellation, cause stamp를 한 backend lock 구간에서
  수행하고 cancellation callback 뒤 terminal 상태를 다시 확인한다.
- Reservation과 dispatch-start commit 뒤에도 각각 authority를 다시 검사한다. Reserve 뒤 loss는
  dispatch-start를 막고, dispatch-start 뒤 loss는 provider 진입을 막는다.
- Required evidence 복구는 lifecycle recovery 호출 전·후에 lease authority를 검사한다.
- Token `deadline`과 wall-clock timeout은 `run_timeout` terminal projection을 공유한다.
  Settled park의 close는 기존 turn settlement를 재실행하지 않는다.
- Model stream 종료 상태도 typed cause를 따른다. `user_cancel`은 `cancelled`, `deadline`은
  `timed_out`, drain/shutdown은 `interrupted`로 닫는다. `lease_lost`는 stale stream close를
  금지한다.
- `turn.paused`는 현재 cause-less park로서 이전 `turn.interrupted` cause를 status와 offline
  event projection에서 지운다.
- cause가 없는 legacy `turn.interrupted`도 현재 cause-less park로 읽어 이전 typed cause를
  status와 offline projection에서 지운다.
- unknown non-empty cause도 shared portable-cause parser에서 거부하며 live status, offline,
  backend projection이 모두 현재 cause 없음으로 읽는다.
- turn-level `Stop`은 `user_cancel` cause를 가진 resumable interrupt다.

종료 조건: lease-lost worker가 usage/metric/observer/sidecar/checkpoint/event/projection/terminal
mutation을 만들지 않는다.

### PR 7 — Conformance와 release closure

- public conformance exports
- embedding/contract/compatibility 문서
- import, wheel, cold-start audit
- full test gate

종료 조건: v0.22 완료 조건과 release checklist가 모두 증거를 가진다.

실행 순서는 `PR 1 → PR 2 → PR 3 → PR 4 → PR 5 → PR 6 → PR 7`이다. 모든 PR을
순차 진행한다. 특히 PR 5와 PR 6은 겹치지 않는다. PR 5가 리뷰 수렴, 병합, 동기화까지
끝난 뒤 PR 6 브랜치를 만든다. 각 PR은 독립적으로 green 상태를 유지한다. PR 3부터 기능은
opt-in이다. Default in-process 경로는 마지막 PR까지 기존 동작을 유지한다.

## 14. 개발 브랜치와 PR 운영

### 14.1 브랜치 구조

`codex/v0.22-production-boundaries`가 v0.22 통합 브랜치다. 최초 구현 PR을 시작하기 전에 이
브랜치를 `origin`에 게시한다. 각 구현 브랜치는 최신 통합 브랜치에서 생성한다.

| 순서 | 개발 브랜치 | 대상 브랜치 |
|---|---|---|
| PR 1 | `codex/v0.22-pr1-serialization` | `codex/v0.22-production-boundaries` |
| PR 2 | `codex/v0.22-pr2-hosting-contracts` | `codex/v0.22-production-boundaries` |
| PR 3 | `codex/v0.22-pr3-model-call-lifecycle` | `codex/v0.22-production-boundaries` |
| PR 4 | `codex/v0.22-pr4-agent-loop-recovery` | `codex/v0.22-production-boundaries` |
| PR 5 | `codex/v0.22-pr5-sink-policy` | `codex/v0.22-production-boundaries` |
| PR 6 | `codex/v0.22-pr6-typed-interruption` | `codex/v0.22-production-boundaries` |
| PR 7 | `codex/v0.22-pr7-release-closure` | `codex/v0.22-production-boundaries` |

리뷰된 이력을 보존하기 위해 통합 브랜치와 구현 브랜치를 rebase하거나 force-push하지 않는다.
각 구현 PR은 merge commit으로 통합하고 원격 개발 브랜치를 삭제한다.

### 14.2 PR별 사전조사 gate

개발 브랜치를 만든 직후 구현 전에 다음 조사를 끝낸다.

1. 통합 브랜치의 최신 커밋, working tree 청결 상태, `origin/develop` 진행 여부를 확인한다.
2. 해당 PR이 변경할 계약, 호출 경로, 저장 형식, 기존 test와 compatibility fixture를 읽는다.
3. 정상 경로와 crash, retry, conflict, stale writer 경로를 상태 전이 단위로 적는다.
4. 변경 파일, 추가 test, 회귀 gate, 종료 조건을 확정한다.
5. 조사 결과를 `docs/dx-notes/YYYY-MM-DD-v0.22-prN-<slug>.md`에 남긴다.
6. 조사에서 장기 계약이 바뀌면 이 계획을 먼저 갱신하고 별도 문서 커밋으로 고정한다.

사전조사 기록에는 최소한 `현재 동작`, `발견한 gap`, `구조 결정`, `수정 파일`, `test matrix`,
`범위 밖`, `종료 조건`이 들어간다. 이 gate가 끝난 뒤 기능 코드를 수정한다.

### 14.3 한 PR의 실행 루프

각 PR은 다음 순서를 한 번에 하나씩 수행한다.

1. 통합 브랜치로 이동해 `origin`을 fetch하고 원격 통합 브랜치를 fast-forward로 반영한다.
2. `origin/develop`이 전진했으면 통합 브랜치에 merge하고 기본 회귀 test를 통과시킨 뒤 push한다.
3. 표에 정한 개발 브랜치를 생성하고 체크아웃한다.
4. 14.2의 사전조사를 완료한다.
5. 구현, targeted test, compatibility test, 문서 갱신을 완료한다.
6. 변경 범위만 commit하고 개발 브랜치를 push한다.
7. 통합 브랜치를 base로 PR을 생성한다.
8. 14.4의 리뷰 사이클을 수렴시킨다.
9. 필수 check가 green인지 확인하고 merge commit으로 통합한다.
10. 통합 브랜치를 다시 체크아웃해 fetch와 fast-forward pull을 수행하고 merge 결과를 검증한다.
11. `origin/develop` 전진분을 통합하고 필요한 회귀 test를 실행한 뒤 다음 PR을 시작한다.

PR 5 브랜치는 PR 4의 11단계가 끝난 뒤 생성한다. PR 6 브랜치는 PR 5의 11단계가 끝난 뒤
생성한다. 아직 병합되지 않은 앞 PR의 commit을 다음 PR 브랜치에 쌓지 않는다.

### 14.4 Codex 리뷰 사이클

모든 구현 PR과 최종 통합 PR은 `$gh-pr-review-cycle` 절차를 사용한다.

1. PR URL, base/head 브랜치, 인증 상태, local checkout, working tree를 확인한다.
2. helper의 `--snapshot-only`를 한 번 실행해 review, unresolved review thread, issue comment,
   request-comment reaction을 함께 읽는다.
3. 각 지적을 현재 코드와 계약에 대조한다. 필요한 수정만 적용하고 나머지는 원래 thread에
   근거를 답한다.
4. 수정하면 관련 test를 실행하고 commit과 push를 마친 뒤 각 thread에 변경 위치와 결과를
   답한다. 수정이 없으면 commit 없이 판단 근거를 답한다.
5. 별도 PR comment로 정확히 `@codex review carefully`를 남긴다.
6. helper로 `EYES` 인식을 최대 1분 확인한다. 인식이 없으면 같은 요청을 한 번만 다시 남긴다.
7. 인식 뒤 5분 동안 기다린다. 그 뒤 1분 간격으로 최대 10분 동안만 새 결과를 확인한다.
8. 새 review, thread comment, issue comment, changes-requested, approved/no-issues 신호가 나타나면
   즉시 polling을 멈추고 snapshot 단계로 돌아간다.
9. 인식 후 15분까지 결과가 없으면 마지막 snapshot을 한 번 수행하고 timeout을 보고한다.

helper가 대기하는 동안 별도 수동 polling을 실행하지 않는다. `EYES`는 요청 인식이며 승인
신호가 아니다. Timeout과 unrecognized 상태도 리뷰 수렴이 아니다. 다음 조건을 모두 만족해야
수렴으로 판정한다.

- 최신 commit 기준 actionable unresolved feedback가 없다.
- 최신 리뷰 요청 뒤 approved 또는 명확한 no-issues 결과가 있다.
- 필수 check와 해당 PR의 test gate가 green이다.
- 모든 처리한 지적에 원 thread 답변이 남아 있다.

### 14.5 구조적 재검토 gate

다음 신호가 나타나면 새 리뷰 요청과 polling을 멈추고 내부 구조 리뷰로 전환한다.

- 같은 불변식에 대한 지적이나 수정이 두 리뷰 라운드에서 반복된다.
- 하나의 예외를 막기 위한 조건문이 core, runner, loop, hosting 여러 층으로 퍼진다.
- 한 crash window 수정이 다른 crash window의 호출 수나 fencing 의미를 깨뜨린다.
- public contract, capability, failure precedence가 실제 소유권 경계와 맞지 않는다.
- test를 통과시키는 수정이 문서화된 상태 전이로 설명되지 않는다.

내부 구조 리뷰에서는 열린 지적을 보존하고 상태 전이, crash window, 소유권, failure
precedence, compatibility 영향을 다시 조사한다. 근본 원인을 하나의 type, protocol, state
machine, capability gate로 해결할 수 있는지 결정한다. 필요하면 이 계획과 해당 PR의 dx-note를
먼저 갱신한다. 구조 개선과 회귀 test를 완료한 뒤 같은 PR에서 리뷰 사이클을 새로 시작한다.

PR 2에서는 반복된 durability 지적에 이 gate를 적용했다. 반환 status 중심 검증을 durable
postcondition과 logical-call 전체 history 검증으로 확장했다. Fresh malformed blob은 거부 뒤 state를
다시 읽고 기존 shared blob bytes를 즉시 재검증한다. Malformed map도 stale/cross-run fencing 뒤에만
검사한다. Event와 terminal winner는 재개방 facade에서 전체 payload를 읽으며, dispatch ID는 직전
attempt가 아니라 모든 이전 attempt를 조회한다. 각 경계에는 결함 구현이 정확한 observation을
실패시키는 mutant test가 있다. Attempt 3 이상을 일괄 거부하는 구현도 양성 lifecycle observation이
잡아낸다. Terminal schema alias를 raw tag로 digest하는 구현도 양방향 retry observation이 잡아낸다.
Invocation raw alias digest 구현은 schema/digest-generation 각각의 양방향 same-revision observation이
잡아낸다. Event에는 accepted schema alias가 없다. `schema_version`을 포함한 모든 non-key canonical
field가 content identity에 참여한다.
Blob-bearing CAS/writer-handoff 뒤 referenced bytes를 다시 읽고, 모든 mutation/status/evidence-field
조합의 잘못된 `CommitResult` mutant를 거부한다. Blob 검증은 잘못된 bytes와 대문자 digest를 독립
축으로 검사하고, workspace·committed message·queued content list·queued envelope·nested checkpoint
invocation result·standalone invocation result 참조를 submitted map과 same-run backing 두 해소 경로에서
확인한다. Terminal invocation 재시도는
unknown의 failure-code 유무, success의
retryable tag 유무, failure의 retryable 유무를 하나의 정책 행렬로 검증한다. Malformed-map precedence는
owner와 generation의 세 invalid 조합을 모두 교차 검증한다. Handoff blob probe는 stale/current writer에
서로 다른 digest를 주고 linearization에 따라 stale 전용 digest의 backing visibility가 달라지는지
확인한다. 같은 handoff에서 재개방한 checkpoint·event·invocation·terminal 전체 payload가
linearization winner와 일치하는지도 검증한다. Run 격리는 다른 run에만 저장된 blob 참조를 두
blob-bearing mutation에서 거부하고, 두 run의 event·terminal 전체 payload를 각각 재로딩한다. Contract
harness registry는 probe
전환 때 직전 facade group을 닫고 최대 동시 facade 수를 회귀 검증한다. Checked load fault matrix는
checkpoint·invocation과 corrupt·unsupported_version을 완전 교차한다. Delayed checkpoint는 fresh
낮은 좌표의 commit·idempotent retry와 높은 latest head를 동시에 요구한다. 더 높은 checkpoint가
빈 blob map으로 delayed digest를 재참조하고 재개방 뒤 bytes를 읽어 blob 권위 보존도 확인한다.
Blob reference matrix는 workspace·committed media·두 queued media carrier·nested checkpoint
invocation·standalone invocation의 malformed digest를 거부하고 external invocation result address를
허용한다. Core의 단일 `checkpoint_blob_references()`가 checkpoint carrier 전체를 추출하며 reference
adapter와 conformance fixture가 같은 어휘를 사용한다.
Fence precedence는 existing coordinate의 동일 payload·conflicting payload·malformed blob map과 fresh
coordinate의 세 invalid-authority token·malformed blob map을 독립 축으로 검증한다.
Checkpoint·invocation의 same-key blob-map conflict 뒤 loser digest를 empty-map record로 재참조해
conflicting blob 비공개도 확인한다.
Writer generation 전환 run의 event `seq=1`·`seq=2`를 모두 재로딩해 같은 run 안의 payload 보존을
검증한다.
Fresh partial-authority 세 조합은 checkpoint·invocation의 고유 stale-only blob visibility를 모두
검증해 malformed pre-fence publication을 거부한다.
Invocation load 격리는 같은 run의 두 logical call과 unknown call을 함께 조회해 복합 키 전체를
검증한다.
Historical invocation revision 1·2·3은 exact retry와 dispatch-identity conflict를 모두 비교한다.
Historical settled revision은 receipt-only·failure-code-only conflict도 비교하고 latest head 4를
유지한다. Historical receipt conflict에는 `retryable=true` winner에 대한 false·필드 생략 후보도
포함한다. Cross-run fenced blob 제출 뒤 authorized empty-map 참조는 계속 실패해야 한다.
Invocation terminal edge identity는 settled·unknown과 네 stable identity field를 완전 교차한다.
Terminal retry 행렬은 settled failure의 `retryable=false`·필드 생략·`retryable=true`를 구분하고,
명시적인 true만 새 reservation을 허용한다. Run 격리 행렬은 checkpoint·event·invocation·terminal
네 family의 서로 다른 non-key payload를 재개방 후 각각 완전 비교한다.
Retry reservation identity 행렬은 새 dispatch coordinate에서 `idempotency_key`와 `request_digest`
drift를 각각 거부한다.
Receipt identity 행렬은 `retryable=true | false | omitted`의 모든 directed pair를 conflict로 거부하고
각 conflict 뒤 원래 winner payload를 재로딩한다.
전체 receipt identity 행렬은 공개된 top-level field와 nested usage counter를 빠짐없이 순회한다.
정상 settlement 수용, same-revision conflict, conflict 뒤 winner 보존을 독립 mutant로 검증한다.

### 14.6 최종 통합 PR

PR 7까지 통합한 뒤 다음 순서로 v0.22 캠페인을 닫는다.

1. 통합 브랜치에 최신 `origin/develop`을 merge한다.
2. 전체 test, import, wheel, cold-start, compatibility, 문서와 release gate를 실행한다.
3. 통합 브랜치를 push하고 base `develop`, head `codex/v0.22-production-boundaries`로 최종 PR을
   생성한다.
4. 최종 PR에도 14.4의 리뷰 사이클을 적용해 merge-ready 상태로 수렴시킨다.
5. PR URL, 최종 commit, test 증거, 남은 범위 밖 작업을 완료 보고에 남긴다.

최종 도달점은 review가 수렴하고 필수 check가 green인 `v0.22 → develop` PR이다. `develop`
병합과 배포는 별도 승인 뒤 수행한다.

## 15. 주요 위험과 대응

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

## 16. 구현 시작 전 고정한 결정

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
