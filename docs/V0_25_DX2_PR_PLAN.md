# v0.25 DX2: 통합 예제와 개발자 경험 PR 계획

조사일: 2026-10-28 KST. 기준: fetch한 `origin/develop=973e47492ab0c7ff0747af25776476f22704fcb5`.
P5·DX1·P3·P1/P2·P4는 develop에 병합됐다. P4 최종 통합은 [#165](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/165)다.
**2026-10-28 사용자가 DX2-00·01 구현과 리뷰·CI 후 02·03·최종 통합까지 순차 진행을 승인했다.**
00은 develop 대상 독립 PR이고 01은 수정 전 실패 회귀부터 진행한다. 릴리스·배포·CSP 제품 변경은 별도다.
계획은 로컬 `88b3f9e`에 기록했다. 이 문서가 DX2의 범위·결정·검증·진행 기록을 소유하며 최신 실행 상태는 §9.1 이후에 둔다.

[v0.25 통합 계획](V0_25_PRE_INVESTIGATION.md)의 G7과
[개발자 경험 설계 §7](V0_25_DEVELOPER_EXPERIENCE_PROPOSAL.ko.md#7-dx2-실제-통합-흐름을-릴리스의-증거로-삼기)을 구체화한다.
운영 절차는 상위 저장소의 `docs/operations/campaign-workflow.md`와 `campaign-preflight-review.md`를 따른다.

## 1. 개발자에게 제공할 경험

개발자는 설치한 패키지로 도구를 실행하고, 요청을 멈췄을 때 남는 일을 설명하며, 승인 대기 작업을 새 프로세스에서 재개한다.
각 경로에는 실행 명령, 필요한 호스트 객체, 관측할 결과, 다음 단계 링크를 둔다. 첫 화면은 이 세 경로를 안내한다.

| 학습 경로 | 개발자가 직접 하는 일 | 눈으로 확인할 결과 |
|---|---|---|
| 타입 있는 도구 | Pydantic 입력·결과와 `@tool`을 선언하고 `from_tools`로 실행 | 중첩 모델 속성 접근, 다음 모델 요청에 전달된 도구 결과, 완료 상태 |
| 요청 스트리밍 | `aopen → async with astream → aclose`, 별도 호스트 신호로 Stop | 출력 중단, 종료 대기 phase, 최종 종료 사실, 보고된 사용량과 미보고 항목 |
| 승인 후 재개 | proposal 작성·읽기 승인 대기·park release 후 새 interpreter에서 restore | 원래 proposal 내용, 새 자원 주입, 승인된 읽기 1회, 중복 응답 무시, 누적 사용량 보존 |

백그라운드 작업의 구독 해제·재접속은 두 번째 경로의 짧은 비교 예제로 둔다. 실행 소유권을 고르는 데 필요한 비교다.
hosted multi-tenant 구성, child/fork, gateway 전체 시나리오는 고급 경로로 연결한다.
README에는 첫 도구와 이 경로 선택을 앞에 두고, EMBEDDING은 상세 수명·권한 계약의 참조 문서로 유지한다.

DX2는 기존 공개 계약과 명시적 Helper/Reference 표면을 조합한다. 새 convenience SDK, HTTP 서버, FastAPI/Vercel 변환기,
UI, scheduler/store, 공통 예제 프레임워크, 저장 schema 변경은 후속 작업이다. P8의 자식 취소 정책과 릴리스·CSP 변경도 별도다.

| 책임 | 담당 |
|---|---|
| 범위·정책·문서·공유 runner | root; 실제 소비자와 실행 증거 대조, 비용·의존성 관리 |
| 코드 조사 | `dx2_code_research`; 예제·API·기존 테스트·CI·복구 경계 읽기 |
| 외부 조사 | `dx2_external_research`; 공식 문서·병합 구현·열린 문제를 구분 |
| 독립 계획 검토 | 작성·조사에 참여하지 않은 fresh reviewer |
| 향후 구현 | PR별 지정 worker; 실제 base 사전조사 갱신, 예제와 소비 테스트 구현. root가 공유 문서·CI 편집 조정 |

## 2. 최신 코드에서 확인한 출발점

| 실제 표면 | 확인한 사실 | DX2가 메울 부분 |
|---|---|---|
| [custom_tool_quickstart](../examples/custom_tool_quickstart.py), [minimal_quickstart](../examples/minimal_quickstart.py) | 각각 83·80줄. typed 예제는 실제 observation까지 확인. README 첫 코드는 더 긴 builtin filesystem 구성 | typed 도구를 첫 진입으로 승격하고 중복 설명·불필요한 binding 제거 |
| `examples/` Python 14개 | 9개가 `sys.path.insert(.../src)` 사용. full-stack/messy의 real gateway 자식도 `PYTHONPATH`에 checkout/src 삽입 | 부모와 자식 모두 설치 패키지 실행을 기본으로 통일 |
| [test_examples](../tests/test_examples.py) | 7개 테스트 함수, parametrization 포함 11개 case. P1/P2/P4의 세 예제 파일 자체는 등록되지 않음 | 기존 실행 테스트에 세 파일과 결과 assertion 연결 |
| [설치 CI](../.github/workflows/ci.yml) | exact wheel build/audit, minimal/all-extras import, temp cwd의 host_tool_report 실행이 이미 있음 | 같은 wheel과 기존 job으로 선택한 예제 파일도 실행 |
| [termination_budget](../examples/termination_budget.py) | 첫 delta 안에서 token 취소·break. progress task가 adapter terminal을 해제. `final bill` 표현 사용 | silent call 중 외부 Stop, consumer task 취소, 예외 정리, sparse usage 표시. 관측 사용량으로 용어 정정 |
| [RunStream](../src/monoid_agent_kernel/core/streaming.py), [AgentLoop](../src/monoid_agent_kernel/loop.py) | CM exit가 취소 요청·대기·driver await를 소유. 결과와 suspension은 별개 | 이 수명을 그대로 사용. 예제 watcher의 종료와 원래 예외 전파를 소비 경로에서 검증 |
| [model_call_facts](../examples/model_call_facts.py), [recover_model_call_facts](../examples/recover_model_call_facts.py) | 43·104줄. 첫 파일은 출력 중심. 두 번째는 메모리 journal·같은 프로세스의 codec/hook 재전달 | 진단 예제로 유지하고 관측 결과를 assert. 실제 프로세스 복구와 역할을 구분 |
| [host_tool_report](../src/monoid_agent_kernel/reference/host_tool_report.py) | 257줄. asked read·park release·동일 interpreter의 새 loop restore와 fresh/fork를 실행 | 기존 root 경로를 prepare/resume 명령으로 재사용해 interpreter 경계까지 증명 |
| [embedding_hosted_product](../examples/embedding_hosted_product.py) | 370줄. 두 tenant·inbox·callback·historical read. 새 backend가 읽는 것은 종료 후 이력 | 고급 호스트 구성 예제로 명시. 활성 작업 복구라고 설명하지 않음 |
| [backend service](../src/monoid_agent_kernel/reference/backend/service.py) | `astream_run`은 single-submit이며 external-task park 뒤 close. `subscribe_model_stream`은 passive reader | 승인 재개는 checkpoint/hosting 경로로 연결. passive 구독은 별도 작은 예제로 보여줌 |

실제 소비자 CSP의 `pipeline_core/agents/chat_stream.py:281–355`는 Stop bridge를 별도로 유지해 조용한 모델 호출에도 취소를 전달한다.
같은 파일은 승인 suspension을 지원하지 않는 채팅 경로라고 표시한다. DX2는 이 두 소비 요구를 구분해 가르친다.
CSP 코드·의존성 핀은 읽기만 했고 변경하지 않는다.

### 2.1 실행 기준선

위 base, Windows/Python 3.11.9에서 기존 소스·테스트를 그대로 실행했다.

- example·astream·RunStream budget·passive disconnect·cursor reset·host report 관련 **58 PASS / 15.61초**.
- 기존 P4 wheel의 `monoid_agent_kernel/` 아래 286개 파일이 현재 src와 byte-identical함을 확인했다.
  SHA256 `a52b6108fec32613d57cb4b747e90215e45ec1416f714a170267fc80a74d999e`, Pydantic 2.14.1.
- checkout 밖에 복사한 typed/facts/receipt-recovery/termination 예제와 설치된 host report **5개 실행 성공**.
- 같은 installed interpreter로 checkout의 typed 예제를 import하면 `src/monoid_agent_kernel`을 읽었다.
  복사본은 `site-packages/monoid_agent_kernel`을 읽었다. 실행 cwd 변경만으로 source shadowing을 제거할 수 없다.
- 별도 ignored probe에서 process A가 승인 대기로 park/release하고 process B가 checked checkpoint를 읽어 restore했다.
  PID 2632→4240, 각 process 자원 open 1회, B의 모델 호출 1회, 중복 승인 응답 무시, 원래 proposal 읽기, 실제 workspace 무변경을 확인했다.
  기존 API로 두 단계 예제를 구성할 수 있다. probe는 새 CLI 구현·crash recovery·사용량 누계 검증을 대신하지 않는다.
- plain asyncio probe에서 첫 delta 전 Stop과 소비 task 취소를 각각 실행했다. 두 경우 늦은 텍스트 전달 0회,
  receipt 1회, `user_cancel/provider_terminal`, input 0/output 3·total 미보고를 확인했다. 소비 task 취소의 원래 CancelledError도 전파됐다.
  실제 HTTP/ASGI나 반복 cancellation에 대한 검증으로 확대하지 않는다.

원문은 `.tmp/v025/dx2-baseline-tests.txt`, `dx2-baseline-consumer.json`, `dx2-example-*`,
`dx2-restart-probe.py`, `dx2-restart-prepare.txt`, `dx2-restart-resume.txt`, `dx2-stream-probe.py/.txt`에 둔다.
연구 노트는 `dx2-code-research.md`, `dx2-external-research.md`다. 이번에는 새 전체 suite·서비스·provider 호출을 실행하지 않았다.

## 3. 외부에서 드러난 문제와 채택할 답안

2026-10-28 KST에 공식 문서와 GitHub 원문을 확인했다. 병합 여부와 배포 여부는 별개이며, 아래 변경을 위한 새 의존성은 도입하지 않는다.

| 외부 문제·근거 | 확인한 해결과 상태 | DX2에 적용할 것 |
|---|---|---|
| Vercel AI SDK [#18458](https://github.com/vercel/ai/issues/18458): resume 중 Stop 후에도 replay가 UI를 갱신 | [#18461](https://github.com/vercel/ai/pull/18461), merged `79d6195`; pending/active resume의 abort 전달, reader 정리, stale write 차단. [현재 문서](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-resume-streams)는 disconnect와 별도 stop endpoint 구분 | 요청 취소·passive 해제·명시적 작업 취소를 각각 실행. 실제 활성 run과 peer reader를 관찰하고 live cursor reset 표시 |
| PydanticAI [#10258](https://github.com/pydantic/pydantic-ai/pull/10258): async generator aclose 시 실행 소유 task teardown 문제 | merged `2338905`, 10/28 KST. cancel-and-drain 뒤 원래 제어 예외 전달; 정리가 닫힘보다 먼저 일어나는 회귀 | RunStream 소유 정리를 재사용하고 예제 보조 task의 finally 정리·소비자 취소 전파 검증 |
| LangGraph [#7780](https://github.com/langchain-ai/langgraph/issues/7780): interrupt 이전 코드 재실행으로 효과 반복 | issue OPEN. maintainer는 재실행 의미와 conditional-edge 예제 개선 제시. [공식 규칙](https://docs.langchain.com/oss/python/langgraph/interrupts#side-effects-called-before-interrupt-must-be-idempotent)은 효과의 idempotency·승인 뒤 배치 설명 | 무엇을 승인하고 어느 효과가 이미 저장됐는지 표시. 중복 응답·재개 모델 호출·누계 검증. 미채택 scratchpad 설계는 가져오지 않음 |
| PydanticAI [#11431](https://github.com/pydantic/pydantic-ai/pull/11431): 서버 이력에 client 전체 이력을 더해 중복 | merged `f45e2ab`, 10/28 KST. server-owned history 모드와 실제 JS client wire 검증; 일부 incomplete-run 문제는 후속 | hosted 예제의 tenant/run/task identity 검증 유지. UI wire와 이력 병합은 v0.26에서 실제 client로 검증 |
| PydanticAI [실행 예제 검사](https://github.com/pydantic/pydantic-ai/blob/45eb9d0d9d7b0c87a981d8af01ebb8d383a0f20c/tests/test_examples.py) | README/docs/package 예제를 임시 cwd에서 scripted model/HTTP로 실행하는 구현 확인 | 기존 `test_examples.py`와 wheel smoke 확장. 사용자가 실행하는 파일을 직접 검증 |

[Pydantic UI streams](https://pydantic.dev/docs/ai/integrations/ui/overview/)의 외부 worker 경로는 delta와 run result를 함께 전달한다.
DX2에서도 텍스트 종료, turn 결과, 승인 suspension, run 종료를 각각 읽는다. 외부 라이브러리의 이벤트/FSM을 복제하지 않는다.

## 4. 채택 정책

### 4.1 설치와 예제의 역할

v0.25 배포 전에는 checkout과 같은 revision의 wheel 또는 `pip install .`를 사용한다.
현재 PyPI의 0.24 패키지로 develop 전용 예제가 동작한다고 안내하지 않는다. 개발용 editable install은 별도로 설명한다.
일반 예제의 `sys.path` 삽입과 그 때문에 붙인 `noqa: E402`를 제거하고, 필요한 extras를 각 고급 예제에 적는다.
`full_stack_integration.py`와 `messy_workspace_cleanup.py`의 real gateway subprocess도 예제가 강제한
checkout `PYTHONPATH` 삽입을 제거한다. 기존 `sys.executable`·사용자 환경 전달과 gateway 동작은 유지한다.
이 두 실제 launch helper가 만드는 환경·interpreter로 import probe를 실행해 자식 provenance를 확인한다.
provider 호출·자격증명 없이 검사하고, 부모 예제만 installed import한 결과로 자식까지 통과시켰다고 판단하지 않는다.
14개 파일의 목적을 살피되 runtime·real-provider harness 동작을 바꾸는 대규모 정리는 피한다.

첫 도구 경로의 모범은 `custom_tool_quickstart.py`다. 실제 typed 인자·결과·다음 observation 검증을 유지한다.
README 코드는 짧게 하고 전체 실행은 이 파일로 연결한다. builtin 도구·명시적 config는 `minimal_quickstart`,
session/checkpoint는 `embedding_local_product`로 이어진다. 중복된 대형 코드를 새 매뉴얼에 옮기지 않는다.

### 4.2 요청 소유 스트림과 passive 구독

| 호스트 의도 | 사용할 기존 제어 | 예제의 결과 |
|---|---|---|
| 요청 하나를 종료하는 Stop | 별도 호스트 신호에서 `CancellationToken.cancel()` | 조용한 호출에도 요청 전달, 현재 run 취소, receipt 1회 |
| 요청 consumer의 취소·예외·early break | `async with loop.astream(...)`의 exit, 바깥 `finally`의 loop/보조 task 정리 | 출력 전달 종료, driver 정리 관측, 원래 consumer 예외 전파 |
| 같은 run에서 현재 turn만 중단하고 계속 대화 | `interrupt_turn()`과 suspension 처리 | 후속 turn을 허용하는 별도 사용법으로 링크; token 취소 후 재사용 안내를 하지 않음 |
| 백그라운드 실행의 화면 닫기 | `LiveModelStreamSubscription.close()` | 실행·다른 구독은 계속. 재접속 때 마지막 live cursor 사용 |
| 백그라운드 작업 자체의 Stop | 승인된 run token으로 기존 `cancel_run`/control 호출 | 구독 close와 구분된 실행 제어 결과 |

HTTP disconnect를 어느 제어로 옮길지는 호스트 정책이다. DX2의 요청 예제는 표준 asyncio consumer cancellation을 직접 구동한다.
ASGI/AnyIO level cancellation·실제 브라우저 연결·Vercel protocol 지원은 해당 어댑터의 후속 검증 범위다.
silent call Stop을 첫 delta 수신에 의존시키지 않는다. adapter의 gate와 사용자의 Stop 신호를 독립적으로 제어한다.

live `(generation, sequence)` cursor와 durable event의 `next_seq`를 분리한다. live ring의 `reset`/`cursor_gap`을 표시하고
전체 텍스트 복원을 약속하지 않는다. durable EventSubscription에는 close 메서드가 없으므로 polling 종료 또는 frames generator 정리를 쓴다.
passive 예제는 실제 `RunnerBackend.submit_run`이 구동한 활성 model call을 관측한다. synthetic broker writer만으로 실행 지속을 판정하지 않는다.

### 4.3 종료와 사용량을 보여 주는 방식

P4의 `phase`와 remaining 값을 함께 표시한다. `hard_cancel_remaining_s=0`, `calls=()`는 각 관측 범위의 값이다.
provider thread·도구·자식·host resource 정리와 최종 청구액의 완료 신호로 사용하지 않는다.
P1/P2의 `requested_by`, `ended_by`, sparse `usage_evidence.reported`, counter별 completeness를 출력한다.
명시적 0은 0, 보고되지 않은 수량은 unknown으로 유지한다. `bill` 변수·설명은 reported usage로 바꾼다.
UI 진행률·종료까지 정확한 초 수·청구 완료라는 새 의미를 붙이지 않는다.

짧은 예제의 성공 출력은 domain result와 핵심 사실로 제한한다. 원시 요청·토큰·호스트 자원 repr을 출력하지 않는다.
현재 자료에서 재현 가능한 API 용례는 다음과 같다. 완전한 수명 관리는 실행 예제가 소유한다.

```python
# stream은 열린 AgentLoop.astream(...)의 RunStream이다.
snapshot = stream.termination_budget()
print(snapshot.stream_exit.phase, snapshot.stream_exit.hard_cancel_remaining_s)

# receipt는 ModelIOSubscription.on_model_call에서 얻는다.
if receipt.usage_evidence is not None:
    print(dict(receipt.usage_evidence.reported))
    print(dict(receipt.usage_evidence.completeness))
```

### 4.4 승인과 새 프로세스 재개

기존 보고서 예제를 재사용한다. `report.save`는 proposal에 먼저 쓰고, `report.read`의 실행 전에 승인 대기한다.
승인 전 read handler 호출은 0회이며 실제 사용자 workspace는 승인 뒤에도 그대로다. proposal approve/apply는 별도의 제품 작업이다.
이 설명과 함께 기존 proposal 승인/apply·tenant·callback 테스트를 연결한다.

prepare는 기존 checked checkpoint의 committed park를 확인하고 `release_parked`한 뒤 종료한다.
resume은 같은 root/run identity의 최신 checked checkpoint를 읽고, 새 process-local key/provider/resource factory를 구성해 restore한다.
동일 프로세스에서 host와 도구가 같은 키 인스턴스를 공유하고, 프로세스 사이에는 key나 resource 객체를 저장하지 않는다.
CLI의 root/run/task는 로컬 호스트가 제공한다. 인증된 웹 승인 endpoint를 대체하는 예제로 사용하지 않는다.
명시한 task는 복원한 run의 대기 task와 맞춰 확인한다. duplicate는 기존 first-report-wins 결과를 관측한다.

계획하는 명령 형태는 다음과 같다. **아직 추가되지 않은 CLI 옵션이다.**

```bash
python -m monoid_agent_kernel.reference.host_tool_report --phase prepare --root ./report-demo
python -m monoid_agent_kernel.reference.host_tool_report --phase resume --root ./report-demo --run-id <run-id> --task-id <task-id>
```

run/task identity는 호스트 인자로 연결하고 kernel checkpoint 형식을 유지한다. 기존 no-argument 실행과 `run_report`,
`run_child_reports`의 결과 계약은 보존한다. child/fork는 고급 읽기 경로로 분리한다.
두 프로세스 검증은 정상 park 후 재구성을 증명한다. 강제 kill·lease takeover·분산 복구는 기존 hosting/service/DBOS 검증의 범위다.
`MemoryJournal` 예제는 checked receipt 재전달의 진단 자료로 유지한다.

## 5. 기존 예제를 다듬는 범위

| 파일·문서 | 남길 것 | 덜어내거나 정정할 것 | 추가할 것 |
|---|---|---|---|
| README, docs/README, EMBEDDING 첫 부분 | 설치·안정성 구분·계약 링크 | 첫 실행 전에 요구하는 상세 구성 설명, 중복 긴 코드 | 세 경로와 예상 출력, develop용 설치 안내, 고급 경로 링크 |
| custom_tool_quickstart | typed 입력/결과와 observation assertion | src 경로 삽입·불필요한 설명 | 첫 도구 실행 진입점 |
| minimal_quickstart, embedding_local_product | builtin 도구, session/checkpoint | 사용하지 않는 `run.finish` binding은 실제 호출 검증 후 제거 | local session의 예외 정리 |
| model_call_facts | sparse 0/unknown 예제 | 출력만으로 성공을 판단하는 방식 | 반환 요약 또는 명확한 결과 assertion |
| recover_model_call_facts | codec/hook·provider_calls=1 | durable process recovery로 오해할 안내 | 메모리/동일 프로세스 표시, 실제 재개 예제 링크 |
| termination_budget | 실제 aopen/astream, P4 snapshot, P1/P2 receipt | bill 용어, 무한 watcher 대기 가능성, 첫 delta에 종속된 Stop 설명 | 정상·외부 Stop·consumer 취소·실패의 실행 함수와 정리 |
| observe_background_run.py (신규 1개) | 기존 Reference facade·broker 사용 | — | 활성 run의 구독 종료·peer 진행·cursor 재접속·별도 cancel 비교 |
| reference.host_tool_report | root 승인/restore, no-arg 출력, fresh/fork 회귀 | 같은 interpreter 재구성을 process restart로 부르는 표현 | prepare/resume 옵션과 새 interpreter 검증 |
| embedding_hosted_product | tenant·inbox dedup·callback 권한·역사 이력 | active recovery로 오해할 안내·src 경로 삽입 | 목적과 다음 예제 링크. 기능 시나리오는 유지 |
| full_stack_integration, messy_workspace_cleanup | 기존 fake/real harness·scoring | 첫 학습 경로에서 제외, 부모 sys.path·자식 PYTHONPATH의 강제 src 삽입 제거 | 고급 목록의 목적·extra·실행 조건, gateway 자식 import 검증 |
| 나머지 custom adapter/tools, memory, OTel, redacting 예제 | 기존 기능별 용도와 실행 계약 | 같은 src 경로 우회만 정리 | 공통 설치 안내 링크 |

새 passive 파일은 요청 소유 예제와 다른 Reference 설정·수명을 설명하기 위해 한 개만 추가한다.
existing hosted 예제에 새 상태기계를 더하거나 요청 예제에 Reference 설정을 섞는 비용을 피한다.
공통 helper package나 예제 등록 registry를 만들지 않고, 100–160줄 정도의 독립된 소비 코드로 시작한다.

## 6. PR별 변경 범위와 의존성

후보 통합 브랜치: `codex/v0.25-dx2-integration-recipes`. 현재는 생성하지 않았다.
00이 최신 develop에 병합된 뒤 통합 브랜치를 만들고 01→02→03을 순서대로 병합한다.
02와 03의 의미상 선행 조건은 각각 01의 설치/검증 경로다. 실제 작업은 공유 문서·CI 충돌을 줄이기 위해 순차 진행한다.

| PR | 전달 결과·파일 경계 | 의존성·등급·CI | 검증 기준 |
|---|---|---|---|
| **DX2-00** CI 준비 | `.github/workflows/ci.yml`, `tools/v023_ci.py`, `tests/test_v023_ci_foundation.py`의 정확한 통합 branch allowlist·최종 combined profile 등록 | 최신 develop 대상 독립 PR. B, core | 새 branch의 push/PR trigger와 최종 profile 검증. 기존 branch/profile 유지. 기존 테스트 패턴으로 등록 전 실패→등록 후 통과 |
| **DX2-01** 설치 후 첫 성공 | README·문서 진입 순서, 예제 부모/자식의 강제 src 우회 제거, typed/minimal/local 정리, facts/recovery 의미 검증. `test_examples.py`와 기존 installed smoke에 typed·facts·recovery·현재 termination·host report 실행 연결 | 00 뒤 통합 branch. B, core | checkout 밖 동일 wheel import, 두 real gateway launch helper의 자식 provenance, 주요 결과 assertion, 기존 11 case 보존, 정상/오류 local cleanup. optional import·minimal install 유지 |
| **DX2-02** 스트림 소유권 | 기존 `termination_budget.py` 확장, 신규 `observe_background_run.py`, EMBEDDING의 흐름 설명, example acceptance·installed smoke | 01. B, core; runtime 제어 수정 필요 시 별도 A 수정으로 분리 | silent Stop·break·consumer cancel·adapter 실패, 늦은 출력 차단, receipt와 sparse usage, watcher 종료. 실제 활성 run의 구독 close 뒤 peer 진행·cursor/reset·명시적 run cancel |
| **DX2-03** 승인 후 새 프로세스 재개 | 기존 packaged host report에 prepare/resume, 기존 example 테스트에 두 subprocess 수용 경로, installed smoke·문서 | 01; 운영상 02 뒤. B, core; 권한/저장 동작 변경 시 A로 재조사 | 각 interpreter의 새 resource, 원본 proposal 읽기, 승인 전 read 0회·승인 후 1회, duplicate 1회만 전달, 모델 재개 호출·누계 사용량, 기존 no-arg/fresh/fork 유지 |
| **DX2-I** 최종 통합 | 누적 예제·문서·CI와 G7 수용표 대조, 누적 diff → develop | 01–03. 누적 B, combined·full push | 세 경로를 같은 최종 wheel로 실행, current/minimum 지원 Pydantic, 기존 서비스·compatibility·package gate, final-head 독립 리뷰 |

01의 path/provenance와 사용량 의미, 02의 수명·정리, 03의 승인·재개는 수정 전에 실패하는 소비 테스트를 먼저 고정한다.
이미 만족하는 경로에는 임의로 실패를 만들지 않는다. 기존 사례 보존과 새 지원 경로의 수용 검사를 구분한다.
계획 단계에서 확인된 런타임 결함은 없다. 구현 중 결함이 재현되면 실제 owner에 regression-first 독립 수정 PR을 두고 의존성을 갱신한다.
예제 안 호환 분기로 kernel 결함을 보상하지 않는다.

## 7. 검증 행렬과 명령

| 주장 | 실제 생산자·관찰점 | 재사용할 근거와 새 검증 |
|---|---|---|
| 설치한 패키지를 쓴다 | 사용자 파일을 별도 cwd에서 installed Python으로 실행; 실제 import 경로. 두 real gateway helper의 launch 환경·interpreter로 자식 import | 기존 wheel audit/job 확장. PYTHONPATH 제거·파일 byte copy·부모/자식 site-packages provenance. real provider 호출은 제외 |
| 도구 타입·결과가 연결된다 | from_tools → handler → 다음 ModelRequest.observations | 기존 typed 예제 assertion·decorator 회귀 유지; 도구가 실제 호출됐는지 확인 |
| Stop이 조용한 호출에도 도달한다 | adapter entered Event → 독립 stop 신호 → CancellationToken → 실제 astream | 첫 delta 없이 대기하는 case. Event/barrier로 순서 제어; 관측 실패가 무한 대기를 만들지 않게 finally에서 gate/task 정리 |
| 요청 취소의 의미가 보존된다 | request consumer task.cancel → RunStream exit → receipt/consumer exception | break와 task cancel을 별도 case. 보조 task 종료·단일 receipt·중단 뒤 전달 없음. 기존 test_async_loop/test_run_stream_budget의 clock/grace 검증 재사용 |
| 표시한 사용량은 관측 사실이다 | 실제 scripted TurnComplete/ModelTurn → receipt | explicit zero, absent counter, drain terminal, terminal 없는 경우. unknown을 0으로 출력하면 실패 |
| 구독 close 후 실행은 계속된다 | submit_run → 실제 adapter → 두 subscribe_model_stream reader | 한 reader close 뒤 다음 delta를 peer가 수신하고 run은 진행. broker에 만든 합성 writer만으로 판정하지 않음 |
| cursor가 의미를 유지한다 | 실제 bounded broker에 stale cursor 제출 | reset을 처리하고 다음 frame으로 진행; event next_seq와 live cursor 분리. test_backend_model_stream_live의 bounds/authority/HTTP 회귀 재사용 |
| 승인이 원래 작업에 적용된다 | process A checkpoint → B restore → report_task_result → tool read observation | 다른 PID, 새 factory, original task identity, duplicate first-report-wins, read 횟수·model 호출·누계 usage 관측 |
| 승인·권한 경계가 보존된다 | 기존 command/approval/proposal 공개 ingress | test_hitl/test_tool_approval/test_backend_command_inbox/test_backend_control 및 proposal 테스트의 다른 run·tenant·중복·apply 근거 유지. 로컬 CLI를 인증 검증 증거로 세지 않음 |

기준선 명령(소스 import, 네 worker):

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:PYTEST_XDIST_AUTO_NUM_WORKERS = '4'
python -m pytest -q -n auto --dist=worksteal --timeout=120 --timeout-method=thread tests/test_examples.py tests/test_run_stream_budget.py tests/test_async_loop.py tests/test_backend_model_stream_live.py::test_closing_one_waiting_subscription_never_stops_writer_or_peer tests/test_backend_model_stream_live.py::test_backend_http_model_stream_prefers_last_event_id_and_disconnect_is_passive tests/test_backend_model_stream_live.py::test_generation_change_and_future_cursor_return_reset_frames tests/test_host_tool_bindings.py::test_report_proposal_approval_release_and_fresh_restore
```

00은 같은 pytest prefix로 `tests/test_v023_ci_foundation.py`, 01–03은 `test_examples.py`와 위 표의 변경 경계 검사를 실행한다.
프로세스/Reference 수용 검사는 기존 integration/serial 분류를 실제 공유 자원에 맞춰 사용한다. 정적 검사는 Ruff 0.15.6과
문서 local link/Python 문법 확인을 재사용한다. 낮은 계층의 모든 행렬을 example 테스트로 다시 만들지 않는다.

설치 검증은 `python -m pip wheel --no-deps --wheel-dir dist .`와 `python tools/release_wheel_audit.py dist` 후 같은 wheel을 설치한다.
선택한 예제 파일을 temp cwd로 복사하고 source `PYTHONPATH` 없이 실행한다. wheel은 바뀐 최종 packaged 파일 기준으로 다시 만든다.
current/최소 지원 계열 Pydantic 검증은 실제 resolved version을 기록한다(이번 기준선은 2.14.1/기존 P4의 2.6 계열 2.6.4).
새 CI matrix·새 검사 runner는 만들지 않는다. 기존 최소 설치 job과 로컬 호환성 lane을 사용한다.
최종 L1/L2/full push는 실제 final head를 기록하며 diagnostic·cancelled·dormant 실행을 required gate 통과로 세지 않는다.

## 8. 비용·이월·완료 기준

최초 구현 예상은 runtime core 변경 0줄, 예제/Reference 450–700줄 변경, 테스트·CI 350–550줄, 문서 250–400줄이다.
여기에 현재 계획 문서와 연결 문서 약 300줄을 포함해 총 1,350–1,950줄의 누적 변경 규모를 예산으로 둔다.
준비 PR·실패 수정·리뷰 비용을 함께 기록하며 PR 분할로 누적 비용을 초기화하지 않는다.
첫 예제는 현재 typed 83줄보다 길어지지 않는 방향으로 정리한다. 새 passive 예제는 100–160줄을 목표로 하되 권한·정리를 줄여 맞추지 않는다.
새 공통 helper/등록 시스템 필요, core/저장 변경 발생, 최초 추정 1.5배 초과 때 범위와 대안을 다시 검토한다.

| 후속 항목 | 담당·다시 여는 조건 |
|---|---|
| FastAPI/ASGI/AnyIO·Vercel UI, 서버 이력 병합 | v0.26 웹 통합; 지원 framework/client 조합을 정할 때 실제 transport로 검증 |
| P8 child 취소·drain·usage 집계 | P8 작업군; 실제 부모-자식 정책 요구가 확정될 때 |
| P6/P9, release/version/tag/PyPI | 독립 유지보수·릴리스 작업; DX2 결과가 모인 뒤 기존 릴리스 절차로 결정 |
| CSP 실제 소비 전환 | CSP 담당 작업; kernel 배포·핀 변경·제품 정책 결정 시 |

G7 완료는 세 학습 경로와 passive 비교가 같은 최종 설치 wheel에서 실행되고, 문서가 그 검증 수준을 정확히 설명할 때다.
추가 서비스·UI가 없어도 typed 도구, Stop·관측, 승인 후 재구성을 따라 할 수 있어야 한다.
구현은 승인된 PR 순서와 각 PR의 사전조사·리뷰·CI를 따른다.

## 9. 계획 검토와 진행 기록

2026-10-28: 최신 develop 조사·기존 58개 테스트·설치 예제 5개·fresh interpreter feasibility probe를 완료했다.
코드·테스트·예제·CI·패키지 파일의 tracked 변경은 없다. 문서 계획만 작성했으며 구현 PR·push·서비스 배포는 수행하지 않았다.
독립 `dx2_plan_review`는 계획·현재 코드·외부 조사·raw 실행 증거를 검토하고 **ready**로 판정했다.
초안 검토에서 두 real gateway helper의 자식 `PYTHONPATH`가 설치본 검증 범위에 빠진 P2를 확인했다.
부모/자식의 강제 source 삽입 제거와 provider 없는 자식 provenance 검증을 01에 추가했고, reviewer가 수정본을 확인했다.
최종 검토 대상 본문 SHA256은 `D301C85403B680615743C0A7D6BB987D39698733943117870A3B752DFF109A7C`다.
이후 변경은 이 결과 기록이다. 미해결 중대 지적은 없다. 리뷰 원문은 `.tmp/v025/dx2-plan-review.md`,
재현 명령·환경·증거 범위는 `.tmp/v025/dx2-baseline-evidence.md`에 둔다.
연결 문서 4개의 local link/anchor 141개와 Python block 7개의 문법 검사를 통과했다.
계획 종료 직전 fetch에서도 develop은 `973e474`였다. 이 판정은 계획의 진행 가능성을 다뤘으며 당시 구현은 보류했다.

### 9.1 실행 착수와 DX2-00

2026-10-28 사용자가 00·01과 후속 02·03·최종 통합을 승인했다. 착수 시 fetch한 develop은 `973e474`이며
기존 계획의 source와 같다. 00 브랜치 `codex/v0.25-dx2-ci`는 계획 commit `88b3f9e`를 포함한다.
root가 00 구현·공유 문서·runner·Git/PR 통합을 맡고, `dx2_01_research`가 01 코드·외부 사례를 갱신한다.
00 사전조사는 `.tmp/v025/dx2-00-preflight.md`, 독립 검토는 `dx2_00_preflight_review`가 맡는다.
GitHub의 현재 branch filter 문서와 실제 allowlist/profile resolver를 확인했다. 적용 범위는 exact branch 두 등록과 기존 회귀 확장이다.
릴리스·버전·배포·CSP 변경은 수행하지 않는다. 다음 파트의 상태와 증거를 이 절에 이어 기록한다.

00 독립 사전 검토는 ready였다. Tests-only `a76f7fd`에서 기존 foundation 모듈은 **4 FAIL / 31 PASS**:
exact branch 기본 combined, 약한 label/manual 거부, push 등록 누락을 각각 확인했다.
두 설정 등록 후 같은 모듈 **35 PASS**, Ruff 0.15.6과 `git diff --check`를 통과했다.
명령은 §7의 pytest prefix와 `tests/test_v023_ci_foundation.py`, source PYTHONPATH·4 workers를 사용했다.
원본은 `.tmp/v025/dx2-00-red.txt`, `dx2-00-green.txt`, 독립 검토는 `dx2-00-preflight-review.md`다.
현재 누적 비용은 계획·승인 기록을 포함한 약 350줄이며 runtime core 변경은 0줄이다.
