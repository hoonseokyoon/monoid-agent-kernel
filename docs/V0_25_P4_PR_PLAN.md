# v0.25 P4: 종료 예산 조회와 시간 제어 PR 계획

조사일: 2026-10-27 KST. 기준: fetch한 `origin/develop=2914c4366f8dfd7ed537f854388836a12a05f078`.
P5·DX1·P3·P1/P2가 develop에 병합된 상태다. P1P2-I는 [#161](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/161)이다.
**2026-10-28 사용자가 P4-00·01 구현과 리뷰·CI 후 P4-02·최종 통합까지 순차 진행을 승인했다.**
00은 develop 대상 독립 PR, 01은 수정 전 실패 회귀부터 시작한다. 릴리스·배포·CSP 제품 변경은 별도다.
계획은 로컬 `e16c304`에 기록했다. 최신 구현 상태와 PR별 증거는 §9.1 이후에 둔다.

[v0.25 통합 계획](V0_25_PRE_INVESTIGATION.md#53-p4-실제-제어와-일치하는-종료-예산)의 G3과
[개발자 API 제안](V0_25_DEVELOPER_EXPERIENCE_PROPOSAL.ko.md#6-v025-api-예시-stop-이후-현재-대기-조회)을 구체화한다.
상위 저장소의 `docs/operations/campaign-workflow.md`, `docs/operations/campaign-preflight-review.md`를 운영 권위로 따른다.

## 1. 전달할 결과와 범위

통합 개발자가 Stop 이후 **어떤 호출을 기다리는지, 지금 적용된 deadline·drain·cleanup에 얼마가 남았는지,
RunStream이 언제 driver task에 강제 취소를 요청하는지**를 같은 제어 상태에서 읽게 한다.
wall clock 변경으로 조기 timeout이나 기한을 넘는 유료 재시도가 발생하는 경로를 먼저 수정한다.

| 책임 | 담당·완료 기준 |
|---|---|
| root | 실제 코드·소비자 대조, 제한된 probe, 정책·PR 계획, 공유 문서와 단일 검증 runner 소유 |
| p4_code_research | run/call/drain/grace/restore/RunStream 경계와 기존 회귀를 읽기 전용 조사 |
| p4_external_research | 공식 문서·소스·PR에서 문제와 해결책 조사; 병합·배포·열린 제안 구분 |
| 독립 계획 reviewer | 작성·조사에 참여하지 않은 fresh reviewer가 계획을 그대로 구현해도 남을 실패 검토 |
| 향후 구현 담당 | 각 PR의 실제 최신 base에서 조사 갱신, 수정 전 실패 회귀 고정, 독립 리뷰와 CI 수행 |

P1/P2의 최초 중단 요청·종료 원인·usage 회계, P5의 전달 직전 권한 검사, P3의 자원 정리·복구를 보존한다.
P4는 root runner가 소유한 호출과 core run/call/drain/shared wait를 다룬다. child 전체 예산 합산·전파는 P8,
shell/process 자체 timeout·lease TTL·capability expiry·provider HTTP timeout·DBOS/Temporal host shutdown 정책은 각 소유자 범위다.
새 타이머 프레임워크, task 전역 스캔, provider 강제 종료, Studio 화면 개편, 저장 schema 변경은 포함하지 않는다.

## 2. 최신 코드와 재현 근거

| 생산자 → 실제 소비자 | 현재 코드에서 확인한 사실 | 수정 경계 |
|---|---|---|
| run 시작 → turn·model·tool | [loop_phases.py](../src/monoid_agent_kernel/loop_phases.py) bootstrap의 `started/deadline`, [loop.py](../src/monoid_agent_kernel/loop.py)의 run guard·TurnContext·tool wait가 wall time 사용 | workspace·host resource open 뒤라는 기존 시작점을 유지하고 하나의 local monotonic cutoff 전달 |
| call → retry·repair | [model_call.py](../src/monoid_agent_kernel/model_call.py)의 `_check_cancel_or_deadline`, fresh/recovered retry-fit 검사; [validated_call.py](../src/monoid_agent_kernel/validated_call.py)가 repair마다 같은 epoch 전달 | 최초 소유 경계에서 한 번 변환. retry·backoff·repair마다 남은 시간을 다시 부여하지 않음 |
| stop → drain | `_AbortDrain.begin`의 `abort_at=time.time()`, `_await_draining_stream`의 `min(abort_at + budget, deadline)` | stop을 처음 수락한 monotonic 시각부터 측정. 같은 cutoff이면 기존 `ended_by=deadline` 우선 |
| model/tool → shared bridge | [core/_sync_bridge.py](../src/monoid_agent_kernel/core/_sync_bridge.py) `await_abandonable_call`이 wall deadline을 relative timeout으로 차감 | 모든 내부 호출자가 같은 cutoff를 사용. asyncio에는 해당 순간의 relative timeout 전달 |
| park → restore | `loop.py` checkpoint의 `remaining_duration_s`; `_rehydrate`가 새 now+remaining 구성 | 기존 parked downtime 제외 정책, None/0 의미와 저장 형식 유지 |
| phase → cleanup | `_aawait`는 **wait 진입**에 grace를 확보. `_aclose_within_grace`와 drain final cleanup은 각 **cleanup 진입**에 다시 읽음 | 단일 call-wide grace로 바꾸지 않음. 이미 적용된 예약과 아직 읽지 않은 미래 값을 분리 |
| open stream → RunStream exit | `_open_drain_budgets`는 positive-drain stream의 float 목록. `_run_stream_drain_grace_s`는 현재 설정과 열린 호출의 전체 drain 중 max + 현재 grace | drain=0·one-shot도 관측. 진행된 drain을 다시 전액 주지 않고 소유 상태에서 exit allowance 계산 |
| RunStream escalation → lifecycle | [core/streaming.py](../src/monoid_agent_kernel/core/streaming.py) `__aexit__`는 bounded wait → `task.cancel()` → task 완료까지 await. `_await_lifecycle_thread`는 정리를 끝까지 기다림 | 공개 값은 강제 취소 요청 시점까지의 예산. driver 전체 반환 시간은 보장하지 않음 |

관련 위치: bootstrap 189–194, TurnContext 293/4546, checkpoint 3648, restore 3880–3891,
model boundary 948–963, recovered/fresh retry 1286/1561, drain 2469–2481, grace 2502/2533/2636,
RunStream exit 155–169와 loop exit sizing 1931–1958. 줄 번호는 위 base 기준이다.
같은 저장소의 [Temporal lease 구현](../src/monoid_agent_kernel/adapters/temporal/activity.py)은 server wall duration을
local monotonic cutoff로 바꾼 선례다. 이 원칙을 재사용하며 core에서 Temporal adapter를 import하지 않는다.

### 2.1 제한된 실행 확인

Windows/Python 3.11.9, `PYTHONPATH=src`, 위 develop 소스를 직접 import했다. 제품·기존 테스트 파일은 수정하지 않았다.
ignored `.tmp/v025/p4_preflight_probe.py`는 실제 ModelCallRunner와 RunStream을 구동한다.
model/bridge 모듈의 wall clock view만 ±3,600초 이동시켰으며 OS 시계와 event-loop monotonic 시계는 유지했다.

| 실제 실행 | 대조군 → wall clock 변경군 |
|---|---|
| 5초 deadline, 정상 응답 | 정상 완료 1회 → +3,600초에서 즉시 RunTimeout |
| 40ms deadline, 실패 뒤 80ms retry backoff | 첫 ModelAdapterError·1회 dispatch → −3,600초에서 94ms 후 성공·**2회 dispatch** |
| drain 20ms, terminal 60ms 뒤 도착 | 약 31ms에 drain_timeout → +3,600초에서 즉시 drain_timeout |
| 같은 drain, 시계 후퇴 | drain_timeout → −3,600초에서 약 63ms에 provider_terminal·output_tokens=5 회수 |
| raw RunStream cooperative wait 10ms, 제어 가능한 지연 cleanup | hard cancel 관측 뒤에도 62ms 시점에 pending. cleanup release 후 종료 |

실측 시간은 스케줄러 영향을 포함한다. 위 수치는 새 API나 수정된 코드의 검증 결과가 아니다.
정식 회귀는 PR-01/02에서 제어 가능한 clock·barrier와 실제 생산자/소비자를 사용해 먼저 실패를 고정한다.
마지막 probe의 예상 abandonment warning 2개도 원본에 남겼다.

원본은 `.tmp/v025/p4-preflight-probes.txt`다. SHA-256:
script `936243cdd6051b7b728237afdf2ea6811b92a5beeadf57b935f29b90a0e1024b`,
output `27229c9c880973de56a483ad9a114bbc740fd29daba58db8719fde506e247cd6`.

기존 관련 기준선은 **209 passed in 13.87s** (`.tmp/v025/p4-baseline.txt`)다. 아래 조건으로 실행했다.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:PYTEST_XDIST_AUTO_NUM_WORKERS = '4'
& .tmp/v025/p1p2-01-current311/Scripts/python.exe -m pytest -q -n auto --dist=worksteal --timeout=120 --timeout-method=thread tests/test_async_loop.py tests/test_async_tools.py tests/test_model_call_runner.py tests/test_model_stream_observer.py tests/test_validated_call.py tests/test_checkpoint.py -k 'deadline or drain or grace or cancel or boundary or retry or remaining_duration or clock' --tb=short
```

이번 계획 단계에서 full suite·실제 서비스·wheel 재빌드·브라우저·유료 provider는 실행하지 않았다.
기존 회귀 성공과 위 새 재현 실패가 함께 존재하므로, 현재 green만으로 wall-clock 안전성을 판단하지 않는다.

## 3. 외부에서 확인한 문제와 채택할 답안

확인일은 2026-10-27, 병합일은 UTC다. 외부 라이브러리의 취소 의미 전체를 도입하지 않고 아래 경계를 적용한다.

| 외부에서 드러난 문제 | 확인한 답안·상태 | 우리 제품 적용 |
|---|---|---|
| CPython interpreter queue timeout이 wall clock 이동에 영향받음 | [#154156](https://github.com/python/cpython/pull/154156), 2026-08-17 merged. [b94b9c8](https://github.com/python/cpython/commit/b94b9c8886a987a324a677b5fda5bef27f15cb14)은 deadline 생성과 재검사를 함께 monotonic으로 변경 | 관측 시각만 바꾸지 않고 guard·retry·drain·shared wait 전체를 같은 기준으로 정렬 |
| clock 원점이 실행 환경마다 다름 | [Python monotonic](https://docs.python.org/3.11/library/time.html#time.monotonic), [loop.time](https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.time), [Trio 0.34.0 SystemClock](https://github.com/python-trio/trio/blob/v0.34.0/src/trio/_core/_run.py#L192); [gRPC deadline propagation](https://grpc.io/docs/guides/deadlines/)은 남은 timeout 전달 | process monotonic cutoff는 process 안에서만 사용. event loop에는 relative timeout, checkpoint에는 remaining duration 전달 |
| NaN deadline이 비교를 통과해 timeout을 무력화 | AnyIO [#1317](https://github.com/agronholm/anyio/pull/1317), 2026-10-05 merged. 최신 4.15.1 배포일 09-05 이후 main 수정 | public epoch 입력의 NaN을 provider dispatch 전에 거부. infinity 정책은 별도 명시하며 이 PR에서 모든 infinity 거부를 추론하지 않음 |
| 생성 시각·현재 task의 설정을 실제 효력으로 오인 | Trio [#3010](https://github.com/python-trio/trio/pull/3010), 2024-09-23 merged: relative scope를 진입 때 시작. [AnyIO current_effective_deadline](https://github.com/agronholm/anyio/blob/4.15.1/src/anyio/_backends/_asyncio.py#L2601)은 실제 유효 scope를 반영 | run 시작·call 진입·drain 수락·grace 확보 시점을 각각 고정. 조회가 host getter나 타이머를 시작하지 않음 |
| gRPC channel close의 전역 task/stack 탐색 비용·취약성 | [#41875](https://github.com/grpc/grpc/issues/41875), [실제 반영 commit 7269456](https://github.com/grpc/grpc/commit/72694561276a1d492b0f5c08fd870825b31fe90f): channel-owned active call 집합과 shared grace wait. [#42503](https://github.com/grpc/grpc/pull/42503)은 Copybara 반영 후 closed이며 GitHub merged_at은 null | runner 소유 registry와 한 번의 immutable snapshot. 병렬 call allowance는 max. gRPC의 [같은 thread 제약](https://grpc.github.io/grpc/python/grpc_asyncio.html#caveats)과 달리 kernel은 lock으로 cross-thread 조회 지원 |
| stop 반복으로 deadline이 늘거나 timeout을 전체 종료 보장으로 오해 | [gRPC Server.stop](https://grpc.github.io/grpc/python/grpc_asyncio.html#grpc.aio.Server.stop)은 후속 긴 grace로 기존 stop을 연장하지 않음. [Python wait_for](https://docs.python.org/3.12/library/asyncio-task.html#asyncio.wait_for)는 cancellation 완료를 기다려 timeout보다 오래 걸릴 수 있음 | exit cutoff 최초 1회 확정. `hard_cancel_remaining_s`와 driver 반환을 구분. P3 정리 thread를 강제로 버리지 않음 |
| asyncio wait_for의 완료·외부 취소 경합에서 취소 소실 | CPython [#86296](https://github.com/python/cpython/issues/86296), [#98518](https://github.com/python/cpython/pull/98518), 2023-02-16 merged. [maintainer 확인](https://github.com/python/cpython/issues/86296#issuecomment-1560279122): 3.12 수정, 3.11 미반영 | 기존 shared bridge의 cancellation 구분·detach·discard hook 유지. 최소 Python 3.11에서도 경합 검증; wait_for로 교체하지 않음 |
| 취소 뒤 thread/cleanup은 계속 실행될 수 있음 | [AnyIO thread 계약](https://anyio.readthedocs.io/en/stable/threads.html#running-a-function-in-a-worker-thread), [#1279](https://github.com/agronholm/anyio/pull/1279), 2026-08-23 merged: loop 종료 뒤 late callback 경합 수정 | owned wait 종료 뒤 registry 퇴역과 늦은 결과 처리를 검증. snapshot의 빈 calls를 provider 종료·billing 완료로 해석하지 않음 |

열린 제안은 문제 식별에만 쓴다. AnyIO [#1328](https://github.com/agronholm/anyio/pull/1328)의 진입 시 relative deadline,
PydanticAI [#10479](https://github.com/pydantic/pydantic-ai/pull/10479)의 반복 취소·shielded stream cleanup은 현재 OPEN이다.
새 scope/task 구현을 복사하지 않는다. PydanticEvals [#10291](https://github.com/pydantic/pydantic-ai/pull/10291)는
2026-10-27 병합됐으며 per-attempt timeout 범위를 명시한다. 우리는 기존 overall cutoff를 retry·repair에 유지하고
timeout 뒤 유료 재시도 정책을 새로 도입하지 않는다.

## 4. 채택할 제어 정책

### 4.1 clock·변환·복구

1. core 내부 제어는 `time.monotonic()` 기준의 private cutoff와 remaining 계산을 공유한다.
   `open()`은 event loop 밖에서, `aopen()` bootstrap은 thread에서 실행될 수 있다. process clock을 사용하고
   asyncio wait에는 `max(0, cutoff - now)`를 전달한다. 다른 loop의 absolute clock을 직접 혼용하지 않는다.
2. 공개 `ModelCallRunner.acall(deadline=epoch)`와 `ValidatedCallRunner.acall`의 epoch 의미를 유지한다.
   최초 call/validated-call 진입에서 wall·monotonic을 함께 관측해 변환하고 retry·repair에 같은 cutoff를 전달한다.
   AgentLoop는 이미 가진 run cutoff를 명시적으로 전달한다. 매 attempt 재변환과 암묵적인 child 상속은 금지한다.
   public `acall` 호출 경로와 기존 instrumentation을 유지한다. private 운반 인자의 형태는 PR-01 preflight에서
   실제 호출자·wrapper와 대조해 고정하며 기존 `tests/test_cancellation.py`의 public acall interception도 검증한다.
3. public deadline은 None과 유한 int/float epoch를 지원한다. 과거/0은 즉시 만료, +inf는 unbounded(None),
   −inf는 즉시 만료로 정규화한다. NaN·bool·비수치는 입력 오류로 provider dispatch 전에 거부한다.
   invalid deadline은 새 invocation을 시작하지 않으며 provider receipt를 합성하지 않는다. 이 입력 경계는 문서화한다.
4. run budget 시작점, 이미 elapsed된 실행 시간, authority→cancel→deadline 검사 순서를 유지한다.
   guard·fresh/recovered retry-fit·repair·tool/native/sync wait·background reentry·`wait_for_pending_tasks`를 함께 대조한다.
   delay가 남은 deadline에 들어가지 않을 때 마지막 provider error를 유지하는 현재 정책도 보존한다.
5. checkpoint는 기존 `remaining_duration_s`에 monotonic remaining을 저장한다. restore activation에서 새 local cutoff를 만든다.
   **parked downtime을 차감하지 않는다.** None/0과 구형 checkpoint 의미를 유지한다. 원점 불명 monotonic timestamp는 저장하지 않는다.
   recovered settled invocation의 evidence 재게시·중복 회계 방지 순서는 만료 판단보다 앞선 기존 위치를 유지한다.
6. 기록용 epoch timestamp는 유지한다. run `duration_s`와 TurnContext의 잔여 시간은 monotonic에서 계산한다.
   복구는 기존 bounded `max_duration_s - remaining_duration_s`에 따른 elapsed 재구성만 유지한다.
   unbounded run의 과거 elapsed와 remaining=0 이후의 초과 cleanup 시간은 현재 checkpoint에서 복원할 수 없다.
   별도 elapsed 저장 필드는 추가하지 않는다. lease/capability 등 다른 clock domain의 `time.time()`은 보존한다.

기존 `_positive_seconds`의 drain 규칙(유한 양수만, invalid/raise는 0)과 grace accessor의 fallback·오류 경로를 유지한다.
제어 helper는 기존 core에 최소 private 값/연산으로 둔다. 범용 timer service·AnyIO 의존성·취소 프레임워크는 추가하지 않는다.

### 4.2 호출 상태와 읽기 계약

`ModelCallRunner.termination_budget()`와 `AgentLoop.termination_budget()`는 동기·읽기 전용 API로 제안한다.
각 logical `acall` 진입부터 receipt/evidence 처리가 끝나거나 소유권을 내려놓는 finally까지 registry에 등록한다.
drain=0, async/sync one-shot, sync→awaitable, stream, retry backoff, early error를 모두 포함한다.
`calls=()`는 해당 owner가 기다리는 model call이 없다는 의미다. tool·host resource 정리·descendant 전체 완료는 별개다.

- `call_id`는 process 안에서 충돌 없는 일시적 ID다. 같은 request/logical ID의 동시 호출을 구별한다.
  `logical_call_id`는 기존 값이 있으면 따로 제공한다. `attempt_index`는 현재/최근 kernel dispatch index, dispatch 전에는 None이다.
  retry는 같은 call ID, validated repair의 별도 acall은 별도 ID다. SDK 내부 HTTP 시도는 세지 않는다.
- 상태 갱신은 실제 owner가 수행한다. phase는 `preparing / waiting / streaming / draining / backoff / closing / settling`이다.
  이미 close 중 stop 요청이 와도 phase는 closing이다. drain 수락 사실과 실제 읽기 phase를 혼동하지 않는다.
- 한 번의 monotonic sample과 lock 아래 복사로 immutable dataclass/tuple을 만든다. callback·host getter·I/O를 lock 안에서 실행하지 않는다.
  loop는 run resource 참조를 한 번 잡고 해당 runner와 cutoff를 함께 조회한다. open 전/close 후 owner가 없으면 빈 snapshot이다.
- query는 cancellation을 요청하거나 accessor를 재평가하지 않는다. public 값은 finite nonnegative seconds 또는 None이다.
  source state는 finally에서 퇴역하고, detach 뒤 late callback이 퇴역한 호출을 다시 등록하지 못한다.
- snapshot은 process-local 관측이다. 기록용 receipt·checkpoint·Studio wire에 새 형식으로 저장하지 않는다.

제안 타입 `TerminationBudgetSnapshot`:

| 필드 | 의미 |
|---|---|
| `observed_at_monotonic_s` | 이 복사본의 process-local 단일 관측 시각; 다른 process와 비교 금지 |
| `run_deadline_remaining_s` | AgentLoop run의 남은 시간. runner 단독·run deadline 미설정·owner 부재에서는 None |
| `calls` | immutable `CallTerminationBudget` tuple |
| `stream_exit` | RunStream 조회에서만 `StreamExitBudget`, loop/runner 조회는 None |

`CallTerminationBudget`는 위 ID·phase와 아래 필드를 갖는다.

| 필드 | 의미 |
|---|---|
| `deadline_remaining_s` | 해당 logical call의 cutoff까지 남은 시간; unbounded는 None |
| `drain_budget_s` | stream open에서 확정한 drain 예산. 미결정은 None, drain 비적용 호출은 0 |
| `drain_remaining_s` | 실제 drain 수락 뒤 해당 window의 wait owner가 살아 있는 동안 `max(0, min(stop + budget, call cutoff) - now)`; 시작 전·wait owner 해제 뒤는 None |
| `grace_windows` | 현재 owner가 확보한 `GraceBudget(source, budget_s, remaining_s)` tuple |

`GraceBudget.source`는 `wait_cancel / stream_close / drain_detach`다. `budget_s`는 해당 owner가 실제 적용할 timeout이며,
`remaining_s=None`은 예약했으나 cleanup 대기를 아직 시작하지 않았다는 뜻이다. 시작한 창은 monotonic remaining을 반환한다.
기존 `max(0, grace)` 적용 결과를 기록하므로 음수·NaN의 effective timeout은 0이다. +inf grace는
`budget_s=None, remaining_s=None`으로 unbounded임을 표현한다. 이 조합은 유한 budget의 미시작 상태와 구별된다.
잘못된 타입·getter 예외는 기존 release/fallback 경로를 따르며, 관측 hook이 새로운 예외를 만들지 않는다.
outer detach와 inner aclose가 겹치면 각각의 창을 표시한다. 둘을 합한 `effective_grace_s`나 총 종료 ETA는 만들지 않는다.
grace를 실제 소비하는 기존 bridge/close 경계에 작은 상태 갱신 hook을 연결하고, snapshot 때문에 grace sampling 시점을 바꾸지 않는다.
stream 읽기가 끝나도 consume task가 aclose를 기다리는 동안 outer drain race는 살아 있을 수 있다.
따라서 closing과 유한 `drain_remaining_s`가 함께 존재할 수 있다. phase와 timer owner 수명을 각각 관측한다.

### 4.3 RunStream의 실제 escalation 예산

`RunStream.termination_budget()`도 같은 snapshot 타입을 반환한다. AgentLoop가 만든 stream은 owner snapshot을 연결하고,
raw RunStream은 calls=(), run deadline=None을 반환한다. raw `current_extra_grace_s=` callback과 기존 동작은 호환 경로로 유지한다.
새 managed 경로와 legacy callback을 동시에 소비하지 않는다.

`StreamExitBudget`는 `phase=idle / cooperative_wait / wait_interrupted / hard_cancel_requested / settled`,
`wait_budget_s`, `hard_cancel_remaining_s`를 갖는다. exit 전 두 수치는 None, 실제 wait 시작에 한 번 확정한다.
강제 취소 요청 뒤 remaining은 0이며, 그 사실은 최종 settled에서도 0으로 유지한다.
강제 취소 요청 없이 끝났으면 remaining은 None이며 phase=settled다.
반복 Stop·snapshot 조회·설정 변경이 이 cutoff를 연장하지 않는다. 자연 종료도 settled로 표시한다.
legacy raw 입력으로 유한 예산을 표현할 수 없으면 두 수치를 None으로 관측하고 기존 실행 경로를 유지한다.
이 경우에도 실제 hard cancel이 요청되면 remaining=0 규칙이 우선한다.
호출자는 phase와 함께 해석한다. None만으로 exit 전 상태라고 판단하지 않는다.

cooperative wait 중 `__aexit__` waiter가 외부 취소되거나 wait가 실패하면 해당 timer는 집행되지 않는다.
그때는 **wait_interrupted / hard_cancel_remaining_s=None**으로 전이하고 예외 전달은 현재 동작을 유지한다.
`wait_budget_s`와 최초 cutoff는 기록으로 남긴다. 별도 background timer·driver 취소 소유자는 추가하지 않는다.
직접 `__aexit__`에 재진입하면 최초 cutoff를 재사용하며 이미 지났으면 즉시 강제 취소를 요청한다.
query나 Stop만으로는 재진입하지 않는다. driver가 나중에 끝나면 settled로 전이한다.
이미 강제 취소를 요청한 뒤에는 waiter의 외부 취소가 그 사실을 지우지 않는다.

managed stream은 request_cancel 직후 owner snapshot과 **exit 정책용** grace 후보를 한 번 읽고 아래 allowance를 확정한다.

```text
B = 기존 RunStream cancel_grace_s (AgentLoop 기본 8초)
D_i = draining이면 현재 drain_remaining_s
      streaming이면 min(열 때 확보한 drain_budget_s, 현재 deadline remaining)
      그 밖의 phase이면 0
G_i = 시작한 유한 grace 창의 remaining과 아직 시작하지 않은 유한 예약 budget 중 최대 (없으면 0)
      waiting/streaming/draining에는 exit 시 읽은 grace 후보도 max 후보에 포함
A_i = D_i + G_i                    # 한 호출에서 순차로 허용할 drain + cleanup
W = B + max(A_i, default=0)        # 동시 호출 allowance는 합하지 않음
exit_cutoff = now + W              # 이 cutoff로 실제 asyncio.wait timeout을 계산
```

unbounded call deadline은 D를 줄이지 않는다. 복수 grace 창은 max로 allowance를 확보하되, 실제 nested cleanup의
owner가 먼저 끝나면 해당 창을 퇴역한다. 이 계산은 **RunStream의 escalation 정책 예약**이다. 미래 phase의 grace 적용값과 구별한다.
exit 후보는 해당 waiting/streaming/draining call이 있을 때 한 번 읽는다.
기존 `_positive_seconds` 규칙을 적용하며 getter 실패 시 0이다. 이미 확보한 유한 창은 유지한다.
unbounded grace는 유한 escalation allowance에 더하지 않는다. snapshot에는 unbounded로 표시하고 기본 hard-cancel fallback은 유지한다.
snapshot query는 이 getter를 호출하지 않는다.
현재 active call이 없으면 W=B다. 다음 호출에 설정될 drain knob는 포함하지 않는다.
이미 진행된 drain·cleanup의 remaining을 소비하며, closing 중인 call에 drain 전체를 다시 부여하지 않는다.

기존 v0.24 sizing 대비 변경: 미래 configured drain의 가산을 없애고, one-shot의 실제 grace 예약도 반영한다.
exit 시작 이후 knob를 올리면 이후 phase가 이를 읽을 수 있지만 이미 확정된 stream cutoff는 유지한다.
기존의 phase별 설정 sampling과 hard-cancel fallback을 함께 설명한다. 임의의 이후 설정까지 기다리는 보장은 추가하지 않는다.

`task.cancel()` 뒤에는 현재처럼 driver를 await한다. P3의 cleanup thread·durable settlement를 중간에 버리지 않는다.
따라서 `hard_cancel_remaining_s=0`이고 driver가 아직 pending인 상태는 유효하다. UI는 이를 “강제 취소 요청됨·정리 중”으로 표시한다.
stalled provider thread가 외부에서 계속 실행되는 경우도 기존 detach 경고·I/O timeout 책임을 유지한다.

## 5. 통합 개발자가 쓸 API 예시

아래는 **구현 전 제안**이다. P4-02에서 공개 export·docstring·설치 wheel 예제를 함께 검증한다.

```python
def on_stop(loop, cancellation_token):
    cancellation_token.cancel()
    snapshot = loop.termination_budget()
    return {
        "run_deadline_remaining_s": snapshot.run_deadline_remaining_s,
        "calls": [
            {
                "call_id": call.call_id,
                "phase": call.phase,
                "deadline_remaining_s": call.deadline_remaining_s,
                "drain_remaining_s": call.drain_remaining_s,
                "cleanup": [
                    {
                        "source": window.source,
                        "budget_s": window.budget_s,
                        "remaining_s": window.remaining_s,
                    }
                    for window in call.grace_windows
                ],
            }
            for call in snapshot.calls
        ],
    }


def stop_progress(stream):
    # host가 보관한 RunStream을 polling. 조회는 취소/대기 시작을 유발하지 않는다.
    exit_budget = stream.termination_budget().stream_exit
    return {
        "phase": exit_budget.phase,
        "hard_cancel_remaining_s": exit_budget.hard_cancel_remaining_s,
    }
```

token.cancel 직후 아직 `__aexit__`에 진입하지 않았다면 stream phase=idle이다. 외부 취소만으로 exit timer를 생성하지 않는다.
exit waiter의 취소로 timer가 중단됐다면 wait_interrupted/None을 “종료 대기 중단”으로 표시한다.
None은 필드·phase에 따라 미설정·미시작·unbounded·집행 중단을 뜻하며 0으로 대체하지 않는다.
실제 완료·usage는 기존 P1/P2 receipt/outcome에서 읽는다. budget snapshot으로 최종 사용량을 추론하지 않는다.
P4-02 offline recipe는 실제 `aopen → astream → Stop/early exit → snapshot → receipt`를 실행하고,
독립 progress task가 `__aexit__` 대기 중의 상태를 읽도록 구성한다.

## 6. PR별 변경 범위와 의존성

승인된 순서는 `latest develop → P4-00 독립 PR → codex/v0.25-p4-termination-budget → P4-01 → P4-02 → P4-I → develop`다.
integration branch는 00 병합 뒤 최신 develop에서 만든다. 각 part도 실제 최신 integration head에서 조사한다.
공유 `loop.py / model_call.py / _sync_bridge.py` 수정은 직렬화하고 매 단계 실제 소비자가 작동하게 한다.

| PR | 변경 범위·의존성 | 완료·검증 기준 | 위험 |
|---|---|---|---|
| **P4-00** CI 등록 | develop 대상 독립 PR. `.github/workflows/ci.yml` push allowlist에 정확한 integration 이름, `tools/v023_ci.py` terminal profile=combined, 기존 CI policy tests | 다른 branch routing 불변, 명시 branch의 push·terminal combined 선택 검증. 기존 정책 검사·독립 리뷰·CI | B |
| **P4-01** monotonic control | 00 후 integration 대상. private cutoff, loop/phase/model/validated/shared bridge 경계, checkpoint remaining 계산·restore, 관련 contract 문서 | 수정 전 wall jump·retry/repair·restore 회귀부터 실패 고정. 기존 결과/usage/권한·자원 계약 통과. 실제 AgentLoop·standalone runner 소비자 연결. 종료예산 공개 타입은 다음 PR | A |
| **P4-02** snapshot + 실제 소비 | 01 후 integration 대상. runner owned registry, phase/grace owner hooks, immutable 공개 DTO와 loop/runner/RunStream 조회, managed exit sizing, public exports·CONTRACTS·EMBEDDING·DX2 offline recipe | 동시 호출·다른 thread 조회·늦은 결과·진행 중 drain·nested cleanup·실제 exit cutoff 일치. legacy raw callback 보존. wheel에서 공개 API·Stop/usage 예제 실행 | A |
| **P4-I** 누적 통합 | 최신 develop을 integration에 반영하고 cumulative diff 점검. 발견한 결함만 수정, 계획·호환성·검증 기록 마감 | final-head 독립 리뷰, full/combined CI, 최소·현재 wheel, 복구/권한/회계 회귀. merge commit으로 develop 통합 | A |

01은 clock 변경 자체로 실제 실패를 고치는 독립적으로 유효한 상태다. 02는 registry만 공개하는 중간 PR로 쪼개지 않고
snapshot·RunStream·실행 예제를 함께 전달한다. scope가 커지면 기능 보장을 유지하는 유효한 경계에서 재분할한다.

최초 비용 추정은 기능 코드 **약 800줄**, 테스트·문서·CI·리뷰 수정 포함 **약 2,450줄**이다.
대략 00 70줄, 01 900줄, 02 1,350줄, 통합 수정 130줄로 잡는다. 삭제·이동 포함 실제 diff와 reviewer 수정 비용을 누적 기록한다.
기능 변경 2,000줄 soft cap 또는 최초 전체 추정의 1.5배에 접근하면 공통 원인·API 표면·분할을 재평가한다.
이 수치는 추가 기능이나 새로운 승인 절차를 만드는 근거가 아니다. 범위·보장 변경은 사용자 결정 사항이다.

## 7. 수정 전 실패 회귀와 검증 기준

| 영역 / 소유 PR | 실제 입력·경합 | 통과 기준 |
|---|---|---|
| clock / 01 | 실제 runner 대기 뒤 wall ±3,600초, 정밀 cutoff 경계 | forward jump가 정상 호출을 조기 종료하지 않음. backward jump가 원래 cutoff 뒤 dispatch를 허용하지 않음 |
| retry·repair / 01 | fresh/recovered retry, repair validator, backoff 중 clock 변경·취소 | 같은 overall cutoff 사용; 재시도 불가 시 기존 provider error 보존, 만료 후 유료 redispatch·usage 이중 합산 없음 |
| drain / 01 | stop 전후·drain 중 clock 변경, deadline과 drain 동률, terminal/error/cancel 동시 tick | 실제 monotonic window 유지. P1/P2 requested_by/ended_by와 P5 output fence 불변 |
| run·tool·bridge / 01 | sync/async model·native tool·sync→awaitable, background reentry, open outside loop/aopen thread | 동일 남은 run cutoff. 단위가 다른 absolute clock 혼용 없음. on_discarded와 최소 Python 3.11 cancellation race 회귀 통과 |
| 입력 / 01 | None·finite·past·0·±inf·NaN·bool·비수치 public deadline | §4.1대로 정규화/거부. invalid에는 dispatch·새 invocation·합성 receipt 없음 |
| checkpoint / 01 | historical remaining=None/0/positive; park 후 장시간·다른 process clock 원점·wall 변경; 이미 settled invocation | downtime 제외, 새 active cutoff와 기존 bounded elapsed 재구성. None/0의 elapsed 정보 한계 보존. 기존 schema roundtrip, 복구 evidence barrier·원본 identity·회계 한 번 유지 |
| registry / 02 | 같은 request/logical ID 동시 acall, drain=0/positive, one-shot, sync→awaitable, backoff·입력/accessor 실패·receipt 실패 | ID 충돌·누락 없음. owner 범위·phase 정확. finally 퇴역, late callback 부활 없음. 설정 getter 호출 수 불변 |
| grace / 02 | wait 진입 후 grace 변경; drain phase 전환; close 중 stop; outer drain race·detach와 inner aclose 중첩; 0/negative/NaN/+inf/invalid getter | 확보 시점별 값·remaining 구분, closing 중 살아 있는 drain timer 표시, wait owner 해제 뒤 None, 중첩 합산 ETA 없음. lower/raise knob·fallback과 기존 sampling 일치; snapshot이 새 오류를 만들지 않음 |
| cross-thread / 02 | producer 갱신과 다른 thread의 반복 조회·loop close 경합 | immutable 일관 복사본·한 sample, 음수/NaN/inf 미노출, deadlock·host accessor side effect 없음 |
| managed exit / 02 | drain 일부 소비 후 exit, deadline이 drain보다 짧음, drain0 one-shot의 긴 grace, concurrent calls, no-active-call | §4.3 W로 실제 wait 시작. remaining 감소·0에서 cancel. 병렬 max, no-call B, configured future drain 제외 |
| exit lifecycle / 02 | early break/consumer error, Stop 반복, wait 중 knob 변경, task가 cancel을 지연하며 cleanup 수행 | cutoff 고정. hard_cancel_requested/0이면서 driver pending인 상태 정상. cleanup release 뒤 settled; P3 close/evidence 한 번 |
| exit waiter 중단 / 02 | cooperative wait의 task를 직접 cancel, host consumer 취소, wait 실패, 직접 exit 재진입, hard cancel 이후 외부 취소 | 집행 중단은 wait_interrupted/None. 재진입 시 최초 cutoff만 사용. 기존 hard cancel 사실 유지. driver의 나중 완료는 settled. 실제 timer가 없는 상태에 countdown을 표시하지 않음 |
| legacy / 02 | raw RunStream extra callback 정상/raise/비유한, 자연 종료, query before/after exit | 기존 wait 산식·fallback 유지; callback 재평가 없는 query; 유한 예산을 표현할 수 없으면 None/phase로 관측 |
| 통합 / I | P1/P2 facts/usage, P3 resource lifecycle, P5 stale output, installed wheel example, old checkpoint restore | 최종 head에서 전부 통과. schema/ledger/token 회계 변경·provider 재호출 없음 |

01의 첫 회귀 commit은 테스트만 추가하고 unmodified base에서 실패 출력·원인을 남긴다. ImportError로만 실패한 경우
기존 runner/loop를 사용하는 행동 회귀로 보완한다. 02도 현재 full-drain 재부여·grace sizing 등의 행동 실패를 먼저 고정한다.
clock probe를 곧바로 golden output으로 복사하지 않는다. fake clock와 event/barrier로 boundary를 제어하며 고정 sleep 허용폭에 의존하지 않는다.

기존 `tests/test_model_call_runner.py`, `test_async_loop.py`, `test_async_tools.py`, `test_checkpoint.py`,
`test_validated_call.py`, `test_cancellation.py`, `test_model_stream_observer.py`와 shared bridge/discard coverage를 확장한다.
실패 회귀와 직접 영향군부터 실행한 뒤 프로젝트가 요구하는 lint·full suite·conformance·package·combined profile을 통과시킨다.
01/02 모두 A 등급 독립 preflight 및 독립 코드 reviewer 2명, 최신 head 리뷰·CI가 필요하다. 00은 CI 정책 위험에 맞게 검토한다.
Ready PR의 service label과 terminal profile을 실제 PR 상태에서 확인하고, 00 등록만으로 part branch에 CI가 켜졌다고 간주하지 않는다.
한 heavy runner와 기존 resource lock을 사용하고, pytest가 소스를 읽는 동안 해당 소스를 수정하지 않는다.

wheel 검증은 checkout 밖에서 최소 지원 Python 3.11/Pydantic 2.6 계열과 현재 지원 환경으로 공개 import·offline recipe를 실행한다.
최종 서비스 combined는 기존 PostgreSQL/ObjectStore/Temporal 자격 검증을 사용한다. P4 때문에 별도 host timeout 정책은 도입하지 않는다.
Studio bundle은 소스 변경이 없으면 재생성하지 않으며 기존 CI의 freshness 검사를 통과한다.

## 8. 선택 이유와 후속 경계

| 선택지 | 판단 |
|---|---|
| sampled wall-clock snapshot부터 공개 | 실제 조기 timeout·기한 초과 retry가 남는다. monotonic control을 01에서 먼저 수정 |
| AnyIO/asyncio timeout scope로 전체 교체 | 최소 3.11의 경합, provider self-cancel 분류, detach/discard·권한 제약을 다시 풀어야 한다. 기존 bridge 확장 선택 |
| 호출마다 deadline 재변환 | retry·validated repair에서 시간 이동이 budget을 재충전한다. 최초 소유 cutoff 전달 선택 |
| 전역 task scan 또는 request ID registry | owner 외 task 노출·동시 동일 request 충돌. runner 소유 ephemeral ID 선택 |
| 한 `effective_grace_s` / 총 shutdown ETA | phase별 sampling과 중첩 cleanup·외부 thread 수명 정보를 잃는다. 실제 창과 정확한 RunStream escalation만 공개 |
| hard cancel 뒤 전체 driver detach | P3 resource safety와 durable settlement에 영향을 준다. 현재 await 정책 유지 |

후속 owner와 재검토 trigger: P8 담당은 child deadline/drain 상속을 실제 채택할 때 descendant 집계를 검토한다.
hosting 담당은 전체 process shutdown SLA가 필요해질 때 host resource close·외부 thread·provider I/O 상한을 함께 설계한다.
Studio 담당은 P4 API가 구현된 뒤 UI 표시를 확장한다. 현재 DX2 offline recipe가 첫 실제 소비자다.
P1/P2 unknown invocation의 종료 evidence slot, provider 내부 HTTP retry 회계는 기존 carry로 유지한다.

## 9. 계획 검토 기록

2026-10-27 계획 단계: 최신 develop 조사·제한된 실패 재현·기존 기준선 완료. 당시 제품 구현은 보류했다.
종료 전 다시 fetch한 develop도 `2914c4366f8dfd7ed537f854388836a12a05f078`로 동일하다.

fresh reviewer `p4_plan_review`는 작성·조사에 참여하지 않고 네 문서와 실제 source·계약·기존 테스트·raw evidence를 검토했다.
최초 계획 SHA-256 `46e6242273d347f3e763f14b140185c03a96f5045b8dc07bccf8684fd8da97bb`에서 좁은 replan을 요청했다.

| 독립 지적 | 실제 근거와 반영 |
|---|---|
| P2 blocker: exit waiter 취소 뒤 timer 집행 부재 | base raw RunStream, grace=0.1s, wait 진입 뒤 waiter.cancel(), 0.15s 뒤 driver pending/강제 취소 미요청. §4.3에 wait_interrupted/None, 재진입 시 최초 cutoff, 이미 요청한 hard cancel 사실 보존을 추가하고 §7 실제 소비자 회귀 지정 |
| P3 nonblocking: 복구 누적 duration 과도한 보장 | checkpoint는 remaining만 저장. §4.1에서 bounded 재구성과 None/0 정보 한계로 명시; schema 유지 |
| targeted: close 중 살아 있는 drain timer | `_await_draining_stream`은 aclose까지 포함한 consume task를 race한다. root의 읽기 종료 시 None 제안을 실제 wait owner 해제 시 None으로 수정. 기존 close 중 user cancel 회귀를 §7 기준에 연결 |

reviewer probe는 Python 3.11.9에서 raw timer 경계만 확인하고 driver를 release해 종료했다. managed API 구현 증거로 확대하지 않는다.
재현 코드·명령·원본 결과는 ignored `.tmp/v025/p4-plan-review.txt`에 보존한다.
reviewer는 CPython #154156, AnyIO #1317, gRPC 7269456의 실제 diff·상태도 독립 확인했다.
legacy 비유한 값도 실제 hard cancel 요청 뒤에는 remaining=0을 우선한다.
targeted 최종 판정은 **ready**, 남은 blocking/nonblocking 지적은 없다.
검토한 파일 SHA-256은 `ff02af1994188c3c1d8a82d4a2094d4b627a47d596fc2491cdcfc352f5d02016`이다.
그 뒤 이 판정·검증 기록만 추가했다. 최초 fresh breadth와 수정 부분 targeted 확인을 함께 유지한다.
계획 리뷰는 향후 구현 최종 head의 리뷰를 대체하지 않는다. 당시 구현 보류는 10/28 요청으로 해제됐다.

문서 4개의 **로컬 링크·anchor 126개, Python 예시 7개 AST 문법 검사, git diff --check 통과**.
첫 링크 검사에서 코드 안의 대괄호를 링크로 읽은 false positive는 fenced/inline code를 제외해 바로잡았다.
AST 성공은 제안 API의 실행·타입 검증을 뜻하지 않는다. 기존 소스 기준선 209-pass는 계획 문서 수정에 영향받지 않아 재실행하지 않았다.
계획 단계 변경은 문서 4개와 로컬 ignored 조사·memory 기록이다. 해당 단계에서는 구현·CI 변경·PR 생성·push·병합·릴리스를 수행하지 않았다.

### 9.1 P4-00 사전조사 — 2026-10-28

최신 fetch develop은 `2914c4366f8dfd7ed537f854388836a12a05f078`로 계획 base와 동일하다.
`codex/v0.25-p4-ci`는 이 base와 계획 문서 commit `e16c304`에서 시작했다. root가 구현·공유 runner를 소유하고,
`p4_01_research`가 다음 PR의 소스·소비자 조사를 병행한다. P4-00 독립 reviewer는 별도 `p4_00_review`다.

현재 정확한 integration branch는 push allowlist에 없으며 terminal profile도 core로 fallback한다.
기존 P1P2 등록 패턴대로 `.github/workflows/ci.yml`의 exact branch와 `CAMPAIGN_TERMINAL_PROFILES`의 combined 항목을 추가한다.
기존 foundation tests의 default/explicit combined·weaker profile 거부·push 대상에 회귀를 보강한다.
유사한 part/오타 branch는 기존 core fallback을 유지한다. wildcard·권한·SDK lock·제품 변경은 포함하지 않는다.
실제 소비자는 push CI와 `pr-full.yml`의 head/label/profile resolver다. Ready/ci label의 required L2를 병합 근거로 쓰고
manual diagnostic 또는 dormant 성공을 대체 근거로 사용하지 않는다.

독립 사전조사 검토 후 red/green foundation, validate-lock·lint·diff, B 등급 코드 리뷰와 최종 L1/L2 core를 확인한다.
상세 사전조사는 ignored `.tmp/v025/p4-00-preflight.md`다. 00 병합 뒤 최신 develop에서 통합 브랜치를 생성한다.
01/02/I의 required combined, 각 PR의 독립 preflight·A 등급 두 코드 reviewer는 별도로 유지한다.

독립 사전조사는 **READY**, 필수 수정 0건이다. 테스트-only `2a31e84`에서 **4 failed / 25 passed (9.57s)**:
새 branch의 default가 core이고 weaker label/manual core가 허용되며 push allowlist에 없음을 확인했다.
정확한 두 등록 지점을 수정한 뒤 **29 passed (6.08s)**, ruff 0.15.6·validate-lock·diff check가 통과했다.
원본은 `.tmp/v025/p4-00-red.txt`, `p4-00-green.txt`다. mandatory pytest prefix와 네 worker, 단일 runner lock을 사용했다.
[GitHub 이벤트 문서](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request)를
10/28 재확인했고 기존 Draft-fast/Ready-required 경계를 유지한다. 코드 리뷰와 최종 remote CI는 이어서 확인한다.
기능 delta는 등록 2줄이고 계획 commit의 475줄 churn과 실행 기록을 누적 비용에 포함한다.

00 [PR #162](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/162)는 final head `1be46be`에서 독립 코드 **PASS**,
[L1 40543644526](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/40543644526) 및
[required L2 core 40543818445](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/40543818445) **SUCCESS**를 확인했다.
Ready 전환으로 자동 실행된 GitHub 리뷰도 같은 head에 지적이 없었다. dormant run은 병합 근거에서 제외했다.
develop merge는 `22aae182f98b8846747b894a661dfad6e5f52bd2`다. fetch 후 reviewed head와 tree 동일성을 확인했다.
이 base에서 `codex/v0.25-p4-termination-budget`를 생성·push하고, 01 branch `codex/v0.25-p4-clock`를 만들었다.

### 9.2 P4-01 사전조사 — 2026-10-28

actual base는 00 병합 뒤 `22aae18`이며 runtime는 계획 기준 `2914c43`과 동일하다.
읽기 전용 조사자 `p4_01_research`가 내부 public acall 소비자(loop/validated)와 실제 CSP epoch 입력을 대조했다.
최소 private `_MonotonicDeadline(cutoff_s)` carrier는 공개 진입 경계만 구분하고 내부 기존 deadline 인자는 monotonic float로 유지한다.
public acall 경로·callback arity를 보존하고 `_deadline` keyword로 이미 변환된 unbounded도 전달한다.
private deadline을 epoch로 표시하는 제품 소비자는 없으므로 별도의 wall deadline 복제본은 만들지 않는다.
숫자로 표현할 수 없는 큰 정수 입력도 ValueError로 거부하며 unbounded로 바꾸지 않는다.

수정 지도는 model/validated/loop/loop_phases/shared bridge의 16개 시간 계산과 작은 core helper다.
private checkpoint assertion 1개는 monotonic 기준으로, public deadline에 monotonic 과거값을 주던 테스트 1개는 epoch로 바로잡는다.
테스트 helper가 public runner에 전달하는 epoch 입력은 그대로 유지한다. invalid는 accessor·invocation·receipt 전 검증하되,
valid-expired durable recovery는 기존 evidence 재게시 뒤 boundary 검사를 유지한다.
CPython b94b9c8·AnyIO #1317·Python 3.11 clock 문서를 10/28 재확인했다.
PR별 상세 조사 `.tmp/v025/p4-01-preflight.md`를 별도 독립 reviewer가 검토하며, 코드보다 행동 회귀를 먼저 고정한다.

독립 reviewer `p4_01_preflight_review`는 actual base `22aae18`에서 **READY**, 필수 보완 0건으로 판정했다.
검토한 preflight SHA-256은 `6846584bf88279f96cc43b0e97371f0e70cc0e732a17635359371edc33ef200f`다.
이후 root가 공유 runner·최종 구현/통합을, `p4_01_research`가 테스트 파일에 한정한 실패 회귀 작성을 맡는다.
테스트-only 실패를 실행·commit하기 전 제품 코드는 수정하지 않는다.

통합 branch 최초 push [CI 40544692527](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/40544692527)은
`22aae18`에서 **SUCCESS**, actual combined services 포함이다. 01 제품 수정 전 테스트-only `029152f`를 고정했다.
`.tmp/v025/p4-01-red-core.txt`는 **22 failed / 24 passed (8.75s)**다. wall jump의 조기 종료·retry/repair 예산 재충전·
drain 연장과 invalid 입력의 부수효과를 실제 runner에서 확인했다. 최초 테스트 fixture의 사용량 정규화와
validated durable ID 설정 오류를 교정한 뒤 기록한 결과이며, 모든 무변경 대조군이 통과한다.
root는 그 뒤 private carrier·16개 시간 계산·public 소비자 연결과 clock 계약 문서를 수정한다.

핵심 회귀 **46 passed (5.90s)**, 실제 loop/tool/restore를 포함한 영향군 **728 passed (40.22s)**다.
후자는 model/validated/async loop/tools/checkpoint/cancellation/stream observer 8개 모듈을 함께 실행했다.
원본 `.tmp/v025/p4-01-green-core.txt`, `p4-01-related-green.txt`. 추가 정밀 경계·설치 wheel과 독립 리뷰·CI는 이어서 확인한다.

추가 exact cutoff·drain 동률·서로 다른 monotonic 원점의 실제 park/restore 10개도 **PASS (5.93s)**:
`.tmp/v025/p4-01-boundary-green.txt`. P4-01 신규 행동 회귀는 81개다. ruff 0.15.6·diff check와 wheel audit가 통과했다.
