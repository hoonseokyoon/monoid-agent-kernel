# Refactoring Plan (Round 2) — native-agent-runner

> 상태: **PR-1~5 구현 완료 (2026-06-16, 브랜치 `refactor/round-2-interaction-surface`).**
> 대상 버전: v0.11.0.
> 근거: Agent↔Tool / Agent↔Workspace / Agent↔User / Agent↔Backend / 역할 모듈화
> 5개 상호작용 축의 종합 평가(코드 전수 리뷰).
> 검증: ruff 클린 + 181 passed/4 skipped. test_web_gateway/test_backend의 HTTP 소켓·서브프로세스
> 테스트는 Windows 환경 플레이크(매 실행마다 다른 소켓 테스트가 타임아웃/ConnectionAbortedError;
> 격리 실행 시 전부 통과) — 신규 로직과 무관.

## Round 1 (Phase 0–6) — 완료 (요약)

이전 라운드는 **구조 정리**가 목표였고 완료되었다: 공유 유틸 단일화(B), 계약 표면 개명·이벤트
`data` 타입별 스키마화(C), `Workspace` Protocol 도입·루프 탈-특수화(A-1), god object 분해
(`tool_services/`·`RunState`, A-2), 모듈화 프로파일(lightweight/standard/heavyweight) 도입.
Phase 5(proposal 일반화)는 비-파일 산출물 요구가 없어 보류. 상세 단계 기록과 근거는 git 이력
(`git log --grep "Phase"`) 및 `~/.claude/plans/phase-*.md`를 참조. 이 문서는 그 Round 1 계획을
대체하며, "코드가 어떻게 생겼나" 다음 단계인 **상호작용 표면 정렬**을 다룬다.

---

## Context — 왜 Round 2인가

Round 1로 코드 구조는 제품 목표에 정렬되었다. Round 2는 **에이전트의 상호작용 표면
(interaction surface)** 을 정렬한다. 5개 축을 종합 평가한 결과:

- **축4 (Agent↔Backend, turn_handle)**: 모범 수준. 불투명 핸들 + 서버측 연속성/자격증명 격리.
  손댈 것 거의 없음(문서 보강만).
- **축1 (Agent↔Tool 반환)**: 견고(단일 `{ok,...}` 봉투·일관 에러코드·예외 차단·공개/모델/
  비공개 3계층). 단 `to_observation` 평탄화의 키 충돌 위험 + 모델용 retry 신호 부재.
- **축2·3 (Workspace 읽기 / User I/O)**: 텍스트 전용·단일 `instruction` 문자열. 멀티모달 전무 →
  목표2(범용) 격차. → **이번엔 계약만 확정, 구현 후속.**
- **축5 (역할 모듈화)**: `SYSTEM_PROMPT` 하드코딩, 오버라이드/페르소나/동적 컨텍스트 경로 전무 →
  **목표1·2 최대 격차.** → **이번에 전체 구현.**

### 제품 목표 (정렬 기준 — 불변)
1. **모듈화** — 경량~중량 인스턴스를 *구성* 가능.
2. **범용 에이전트** — 기본은 일반 에이전트, 특화는 모듈로.
3. **통합 용이성** — 안정적 계약으로 외부 통합.
4. **backend는 예시** — core는 `reference/`에 의존 안 함.

### 확정 결정 (2026-06-16)
- **멀티모달(축2·3) = 계약만 확정, 구현 후속.** content-part 모델 + capability negotiation
  골격만. PDF/이미지 추출기·provider 멀티모달 매핑은 후속. 텍스트 동작 불변.
- **역할 모듈화(축5) = 전체.** base+append 합성 + 프로파일 페르소나 + per-turn 동적 컨텍스트
  (`ContextProvider` 계약). 정적/동적 컨텍스트 계층 분리.
- **하위호환 = additive 전제.** `instruction: str`·기존 CLI/계약/골든 산출물 불변.

---

## 워크스트림 개요

```
W4 (퀵윈: outputs surface + 문서)        🟢 저  ── 독립, 먼저
W3 (도구 반환 봉투 + retry 신호)          🟠 중  ── 독립 (모델 가시 payload 변경)
W1 (역할 모듈화: 합성→페르소나→동적)      🟡 중  ── 목표1·2 핵심
W2 (멀티모달 계약만)                      🟢 저  ── 독립, 마지막
```

순서 근거: W4는 무위험이라 결과 계약을 먼저 안정화. W3는 observation 모양을 확정해 W1c가 그
위에 쌓이게 함. W1은 a·b(합성/페르소나) → c·d(동적 컨텍스트) 순. W2는 inert 타입 중심이라 마지막.

---

## W1 — 역할/시스템 프롬프트 모듈화 (최우선, 목표1·2)

### W1a. 합성 프롬프트 (base + persona 세그먼트)
- **신설 `core/prompt.py`**: `BASE_SYSTEM_PROMPT`(현 `loop.py` 문구를 범용으로 일반화) +
  `compose_system_prompt(base, persona_segments) -> str`.
- **`AgentRunSpec` 신설 필드**(둘 다 기본값 → 라운드트립·기존 호출 불변):
  `system_prompt_base: str | None = None`, `persona_segments: tuple[str, ...] = ()`.
  `from_json`/`to_json` 배선. 프로파일 폴백은 기존 `shell_policy`/`web_policy` 패턴과 동일.
- **CLI**: `--persona`(repeatable), `--system-prompt-file`. 플래그 경로는 해소된
  `agent_profile.persona_segments`를 base로(=`base_shell`/`base_web` 관용구).
- **조립 이음새**: `SYSTEM_PROMPT` 상수 삭제. `_bootstrap`에서 1회 합성 → `_RunResources.system_prompt`
  보관. `_run_steps`의 `system_prompt=SYSTEM_PROMPT`를 `res.system_prompt`로 교체.
- 리스크 🟢 저 (빈 세그먼트 ⇒ base 그대로).

### W1b. 프로파일이 페르소나 모듈 운반
- **`AgentProfile`**에 `persona_segments: tuple[str,...] = ()` 추가 (trailing default → 기존 3종
  바이트 동일). 이후 `"coding"` 등 특화 프리셋은 *additive* → "특화는 모듈로"(목표2) 실증.
- **범위 밖**: `AgentProfile`엔 `tool_policy` 부재 → 완전 "페르소나+도구+정책" 번들은 프로파일
  계약 확장 필요(후속 후보).
- 리스크 🟢 저.

### W1c. 정적 vs 동적 컨텍스트 — `ContextProvider` 계약
- **신설 `core/context.py`**: `TurnContext`(step·remaining_steps·remaining_tool_calls·deadline_s·
  plan·pending_observation_count) + `ContextProvider` Protocol(`static_segment()`,
  `dynamic_segment(ctx)`). contracts.py 노출.
- **동적 전달 이음새 = 매 턴 `system_prompt`에 append**(채택). 근거: provider가 이미 매 턴
  `system_prompt` 전송 → wire·provider 무변경. `observations` append은 오답(call_id 결속),
  `ModelRequest` 신규 필드는 어댑터 파급 → 지양.
- **provider 배선**: `AgentLoop`에 `context_providers: tuple[ContextProvider,...] = ()` 추가.
  기본 빈 ⇒ dynamic="" ⇒ 프롬프트 바이트 동일.
- 리스크 🟡 중. **가드 테스트**: provider 0개일 때 턴 `system_prompt`==정적값(`FakeModelAdapter.requests`).

### W1d. workspace_index = opt-in 정적 ContextProvider
- 현재 디스크 기록만, 프롬프트 미주입. 요약 렌더하는 빌트인 `WorkspaceIndexContextProvider` 설계
  (이미 빌드된 index 재사용). **기본 비활성** — `context_providers` 추가로 opt-in. 리스크 🟢 저.

**테스트**: 신설 `test_prompt_composition.py`; `test_loop.py`(마커 주입 + no-provider 동등성);
`test_profiles.py`·`test_spec_serialization.py` 확장.
**CONTRACTS.md**: §1.1 spec(필드 2개), §1.1a 프로파일, 신설 §1.6 "Context providers".

---

## W2 — 멀티모달 입력/파일읽기 **계약만** (목표2·3)

### W2a. content-part 모델로 `instruction` 일반화
- **신설 `core/content.py`**: `TextPart`(구현) / `ImagePart` / `DocumentPart`(계약만,
  `source_ref`+`mime_type`) = `ContentPart` 유니온 + `"type"` 키 기반 JSON 코덱.
- **`AgentRunSpec`**: `input: tuple[ContentPart,...] = ()` 추가, `instruction: str` 필수 유지.
  `from_instruction(...)` 편의 생성자 + `effective_input`(`self.input or (TextPart(instruction),)`).
  `from_json`은 `input` 옵셔널, `to_json`에 `"input"`(미사용 시 `[]`).

### W2b. `fs.read` 비텍스트 반환 이음새 (계약만)
- 미래 `fs.read`가 바이너리에 `{"content_parts":[...]}`를 `media` capability 시 반환 가능함을
  **문서화만**. `_fs_read`는 현행 raise 유지 + `# TODO(multimodal):` 경계 주석. 추출 미구현.

### W2c. 지금 vs 후속 + capability negotiation
- **지금**: `core/content.py` 타입+코덱(Text만 실사용); `AgentRunSpec.input`/`effective_input`/
  `from_instruction`; 신규 capability `"media.input"`(기본 미포함 = off); `ModelAdapter` 옵셔널
  관례 `supports_multimodal`(`getattr(...,False)`). 비텍스트 part인데 미지원이면 graceful degrade
  (텍스트만 + 경고 이벤트 `model.input.degraded`).
- **후속**: 실제 part를 `ModelRequest` 관통; gateway/OpenAI 멀티모달 매핑; `fs.read` 추출.
- **스키마 영향**: `model.input.degraded`를 지금 추가하면 `AgentEventType`+`EVENT_DATA_SCHEMAS`
  동시 추가 필수(아니면 `test_every_event_type_has_a_data_schema` 실패). manifest 스키마 무변경.
- 리스크 🟢 저~중.

**테스트**: 신설 `test_content_parts.py`; `test_spec_serialization.py` 확장.
**CONTRACTS.md**: §1.1 spec `input`; 멀티모달=계약만 노트.

---

## W3 — 도구 반환 봉투 + retry 신호 (목표3, 소규모)

### W3a. 봉투/콘텐츠 분리
- 현 `to_observation`은 `**content`를 `ok`/`error` 옆에 평탄화 → 키 충돌 위험. 신 모양:
  ```python
  def to_observation(self):
      obs = {"ok": self.ok, "result": self.content}
      if not self.ok:
          obs["error"] = {"message": self.error, "code": self.error_code, "retryable": self.retryable}
      return obs
  ```
- **파급**: gateway/openai는 불투명 통과 → 무변경. transcript 스키마(`additionalProperties:True`)
  무변경. 백그라운드 재진입은 `to_observation` 우회 → 현행 유지(`is_background`가 신호; 비대칭 문서화).
  테스트는 `output["result"][...]`로 수정.
- **하위호환**: 모델 가시 payload 변경이나 계약은 내부(게이트웨이 뒤)·output은 free-form →
  외부 스키마 무파손. pre-1.0이라 지금이 적기. 머지 전 `reference/`·`examples/` observation 키 grep.
- 리스크 🟠 중.

### W3b. gateway-retryable과 구분되는 도구 retry 신호
- `errors.py`: `NativeAgentError`에 `retryable: bool=False`/`category: str="internal"` 클래스 기본값,
  오버라이드: `ToolExecutionError`(True/"tool"), `PermissionDenied`(False/"policy"),
  `WorkspaceError`(False/"workspace"). `_execute_tool_call`에서 `ToolResult.retryable`(신설)로 전달.
- **주의**: `ModelAdapterError.retryable`(gateway 재시도)과 별개 — 모델 자기교정용 정보. 리스크 🟢 저.

**테스트**: 신설 `test_tool_result_envelope.py`(충돌 해소 + `ToolExecutionError`→`error.retryable`).
**CONTRACTS.md**: §1.3 `to_observation()` 모양, §2.1 observations[].output 예시.

---

## W4 — 퀵윈 2건

### W4a. `run.finish` outputs/notes를 결과로 surface
- 현재 폐기(`del outputs, notes`), 핸들러는 이미 전달. `AgentToolContext`에 `final_outputs`/
  `final_notes` 추가·`finish`에서 저장. `AgentRunResult`에 `final_outputs: tuple[str,...]=()`/
  `final_notes: str|None=None`. `_finalize`에서 채움. `run.finished` 이벤트 스키마
  (`additionalProperties:False`)는 미변경 → **AgentRunResult에만** 노출. 리스크 🟢 저.

### W4b. stateless provider용 turn_handle 문서화 (문서만)
- CONTRACTS.md §1.2·§2.1: stateless 게이트웨이는 불투명 `previous_turn_handle` 뒤에서 전체 대화
  히스토리를 재구성할 의무. core는 system_prompt+tools+(1턴)instruction/(이후)observations만
  전송. 핸들이 유일 연속성 키. 리스크 ⚪ 없음.

---

## PR 시퀀싱 & 순서 제약

1. **PR-1 (W4a+W4b)**: 최소·무의존. 먼저 머지해 결과 계약 안정화.
2. **PR-2 (W3)**: 봉투 reshape + errors retryable/category. W1c 전에(observation 모양 확정).
3. **PR-3 (W1a+W1b)**: `core/prompt.py`, spec/프로파일 페르소나, CLI 플래그, 부트스트랩 조립.
4. **PR-4 (W1c+W1d)**: `ContextProvider`/`TurnContext`, 턴별 세그먼트, opt-in index provider.
   **PR-3 의존(하드).**
5. **PR-5 (W2)**: content-part 타입, `input`/`effective_input`, `media.input`,
   (옵션)`model.input.degraded`(이벤트 추가 시 `Literal`+`EVENT_DATA_SCHEMAS` 동일 PR).

**제약**: PR-4→PR-3(하드). PR-2는 PR-4 앞(소프트). PR-1·PR-5 자유. W2↔W1 강제 순서 없음.

## 깨끗한 설계를 방해하는 기존 코드 (합의됨)
- `ModelRequest` frozen·단일 생성처: 동적 컨텍스트에 신규 필드 추가 금지, system_prompt append로 우회.
- 백그라운드 재진입 observation이 `to_observation` 우회 → W3 봉투 비균일(허용+문서화).
- `AgentProfile`에 `tool_policy` 부재 → 완전 번들은 프로파일 계약 확장 필요(후속).
- `run.finished`·manifest 스키마 `additionalProperties:False` → 신규 데이터는 `AgentRunResult`/spec로 우회.

---

## 검증 (end-to-end)

- **단위/계약**: `pytest`(현 155 passed/4 skip 유지 + 신설). 특히 `test_spec_serialization.py`
  라운드트립 무손실, `test_every_event_type_has_a_data_schema` 동등성.
- **린트**: `ruff check` 클린.
- **골든 회귀**: provider 0개·persona 0개일 때 `FakeModelAdapter.requests[*].system_prompt`가
  기존과 바이트 동일. `examples/*` 산출물 diff/proposal 무변화.
- **수동 스모크**: `native-agent run ... --persona "..." --profile lightweight`; `--spec`+`--profile`
  동시 에러 유지.
- **윈도우 주의**: `test_llm_gateway` HTTP 소켓 테스트는 Windows 플레이크 → 격리 재실행 확인.
