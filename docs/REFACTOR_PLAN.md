# Refactoring Plan — native-agent-runner

> 상태: 제안(draft). 코드 수정 전 합의용 문서.
> 작성 기준일: 2026-06-16. 대상 버전: v0.11.0.
> 근거: `loop.py`, `core/*`, `providers/*`, `tools/*`, `workspace/*`, `shell.py`,
> `web.py`, `jobs.py`, `cli.py`, `reference/*`, `examples/*` 전수 리뷰.

## 0. 이 문서의 목적

여러 차례 agentic 편집을 거치며 구조 감이 흐려진 코드베이스를, **제품 목표에 다시 정렬**하면서
정리하기 위한 단계별 로드맵이다. 한 번에 다 고치지 않는다. 저리스크 클린업으로 바닥을 다진 뒤,
계약 표면을 단단히 하고, 마지막에 침습적인 구조 정렬을 한다.

### 제품 목표 (정렬 기준)

1. **모듈화** — 경량~중량 에이전트 인스턴스를 *구성*할 수 있어야 함.
2. **범용 에이전트** — 기본은 코딩 특화가 아닌 일반 에이전트. 특화는 모듈로 달성.
3. **통합 용이성** — 안정적 계약(contracts)을 통해 다른 시스템에 쉽게 통합.
4. **backend는 예시** — core는 reference에 의존하지 않음. 통합자는 계약 기반 자체 backend 구성.

### 목표 대비 현재 상태 (요약)

| 목표 | 현재 | 핵심 격차 |
|------|------|-----------|
| (1) 모듈화 | 🟡 부분 | capability/policy 게이팅은 있으나 "경량/중량 프로파일"이라는 일급 구성 개념 부재. `AgentLoop`/`AgentToolContext` 거대 객체 |
| (2) 범용 | 🟡 부분 | 코딩 가정은 없으나 **파일·diff 특화**가 깊게 박힘 (SYSTEM_PROMPT, proposal 모델, 루프의 도구 ID 하드코딩) |
| (3) 통합 | 🟢 양호 | 단일 contracts 표면 + 버전드 스키마. 단 `previous_response_id` 누수, 이벤트 `data` 미검증 |
| (4) backend=예시 | 🟢 양호 | core→reference import 0건. 단 `scenario_scoring.py`가 src에 출하 (예외 1건) |

---

## 1. 단계 개요 (phasing)

```
Phase 0  베이스라인 고정         (사전작업, ~0.5d)
   │
Phase 1  안전한 클린업 (B)        저리스크, 가독성·구조 감 회복   ←── 먼저
   │     공유 유틸/죽은코드/policy 통일/scenario_scoring 이동
   ▼
Phase 2  계약 표면 정리 (C)        중리스크, 통합 표면 단단화
   │     previous_response_id 개명 · 이벤트 data 계약화
   ▼
Phase 3  구조: 경계 (A-1)          고리스크, 목표(2)(4) 핵심
   │     workspace Protocol · 루프 탈-특수화
   ▼
Phase 4  구조: 분해 (A-2)          고리스크, god object 해체
   │     AgentToolContext → 서비스 분리 · AgentLoop 슬림화
   ▼
Phase 5  proposal 일반화 (A-3)     최고리스크·탐색적 (별도 의사결정 게이트)
         리뷰 가능 산출물을 파일 diff에서 추상화
   ─ ─ ─
Phase 6  모듈화 프로파일 (선택)     목표(1) 강화: 경량/중량 프리셋
```

**순서 근거**
- Phase 1을 먼저 두는 이유: 공유 유틸(canonical-hash, 원자적 쓰기, 타임스탬프)을 한 곳으로 모으면
  이후 Phase들이 그 위에 안전하게 쌓인다. 죽은 코드를 먼저 치워야 구조 파악 노이즈가 준다.
- Phase 2를 구조 변경 앞에 두는 이유: 계약 데이터클래스(`ModelRequest` 등)를 먼저 정리해야
  Phase 3+ 구조 변경이 깨끗한 계약 위에서 이뤄진다.
- Phase 5는 가장 침습적·탐색적이라 **별도 의사결정 게이트**로 분리. 진짜로 "비-파일 산출물"을
  지원할 제품 요구가 있을 때만 착수.

---

## 2. Phase 0 — 베이스라인 고정 (사전작업)

구조 변경 전, 회귀를 잡아낼 안전망 확인.

| ID | 작업 | 비고 |
|----|------|------|
| 0.1 | `pytest` 전체 통과 확인 (현재 14파일·122 테스트) | 변경 전 녹색 상태 기록 |
| 0.2 | `ruff check` 베이스라인 기록 | 신규 위반만 추적하기 위함 |
| 0.3 | 대표 e2e 스냅샷 보존 (`runs/integration-real-*`, `examples/*` 실행 산출물) | 제안/이벤트 출력의 회귀 비교 기준 |

**완료 기준:** 녹색 테스트 + 린트 베이스라인 + 골든 산출물 확보.

---

## 3. Phase 1 — 안전한 클린업 (B) 🟢 저리스크

순수 정리. 동작·계약 변경 없음. 항목 간 독립적이라 개별 커밋 가능.

| ID | 작업 | 위치 | 목표 | 리스크 |
|----|------|------|------|--------|
| B1 | **canonical-JSON SHA256 단일화** — 3중 구현을 `core`의 공유 util 하나로. 바이트-동일 보장이 정확성-critical | `schemas.py:651`, `packages.py:724`, `recorder.py:378` | (3) | 중* |
| B2 | `utc_timestamp` / ISO-Z 포맷 헬퍼 단일화 (5곳) | `events.py:137`, `manifest.py:52`, `workspace_index.py:72`, `packages.py:762`, `workspace/local.py` | — | 저 |
| B3 | `_write_json_atomic` 단일화 (2곳 동일) | `recorder.py:317`, `packages.py:671` | — | 저 |
| B4 | 죽은 상태값 제거 — `AgentToolContext`의 `background_*` 증분 카운터(아무도 안 읽음). `background_metrics`로 일원화 | `loop.py:84-90` ↔ `326-338` | — | 저 |
| B5 | `decision_for` vs `_decision_for` 중복 제거 | `policy.py:74` ↔ `139` | — | 저 |
| B6 | diff/proposal emit 3중 중복 → 헬퍼 1개 | `loop.py:991-1005, 1254-1272, 1292-1310` | — | 저 |
| B7 | `_proposed_bytes` 죽은 분기 제거 (if/else 동일) | `workspace/local.py:851-853` | — | 저 |
| B8 | **`scenario_scoring.py`를 `examples/`로 이동** (예시 코드의 src 출하 = 목표4 위반). 유일 사용처는 `examples/messy_workspace_cleanup.py` | `src/.../scenario_scoring.py` | **(4)** | 저 |
| B9 | policy 모양 통일 — 공유 `_string_tuple`/`_dedupe` 1벌(현재 4벌, 의미 미묘하게 다름), `to_manifest` 일관화(`ShellPolicy.to_manifest`는 `to_json` 위임으로) | `permissions.py:72/81`, `shell.py:223/725`, `web.py:200/326`, `policy.py:206` | (3) | 저 |
| B10 | shell/jobs 공유 — `_file_size`(2곳 동일)·spawn 보일러플레이트를 내부 공유 모듈로. `jobs.py`가 `shell.py`의 `_`-private import 하는 것 해소 | `jobs.py:26,596`, `shell.py:604,752` | — | 중* |
| B11 | 버전 문자열 단일 출처 — 흩어진 `0.2/0.9/0.11` 하드코딩 → 한 상수 | `cli.py:1118`, `providers.py:16`, 각 `http.py` `server_version` | — | 저 |

\* B1·B10은 "저리스크 클린업"이지만 정확성-critical(해시)·경계 이동(private→public)이라 **테스트로
강하게 가드**해야 함. B1은 이동 전후 해시 동일성 단위 테스트 추가 권장.

**완료 기준:** 테스트 녹색 유지, 공개 이벤트/제안 골든 산출물 무변화, 린트 위반 감소.

---

## 4. Phase 2 — 계약 표면 정리 (C) 🟠 중리스크 ✅ 완료

목표(3)을 단단히. 통합자가 의존하는 표면을 건드리므로 **CONTRACTS.md 동시 갱신** 필수.

> **완료(2026-06-16).** C1~C5 전부 구현·커밋. 결정: C2는 권고(b 단일 reducer)를 뒤집어
> **(a) 타입별 `data` 스키마**로 진행(두 reader는 라이브 sink vs 사후 projection으로 시간 범위·
> 출력 모양이 정당하게 달라 통합 대신 발생원에서 drift 차단). C1은 **버전 범프 없이 개명**
> (event.v1 미동결). C4는 **옵션 A**(turn 프로토콜에 `is_background` 추가, wire 관통). 정정: C1은
> core 데이터클래스 + 이벤트/transcript 키 두 표면 누수였고 OpenAI wire 키는 정당하므로 유지.
> `proposal.*` 라이프사이클 이벤트는 loop가 아니라 cli/reference backend가 emit.

| ID | 작업 | 위치 | 목표 | 리스크 |
|----|------|------|------|--------|
| C1 | **`previous_response_id` → `previous_turn_handle` 개명** (core 데이터클래스). wire는 이미 중립명(`previous_turn_handle`) 사용 중 — core 계약만 OpenAI 의미 누수. 어댑터 3종·CONTRACTS.md 동반 수정 | `providers/base.py:37`, `gateway.py:136`, `openai.py:61`, `loop.py` | (3) | 중 |
| C2 | **이벤트 `data` 계약화** — 현재 envelope만 스키마 검증, `data`는 `{"type":"object"}`로 미검증. 실제 계약이 `StatusJsonSink`·`_apply_event_projection` 두 reader에 암묵 존재. (a) 이벤트 타입별 `data` 스키마 도입 **또는** (b) 두 reducer를 단일 event-reducer로 통합 | `schemas.py:27`, `recorder.py:47-127`, `projections.py:61-114` | (3) | 중 |
| C3 | adapter 공통 로직 추출 — `gateway.py`/`openai.py`의 reasoning payload·usage 정규화 중복을 공유 헬퍼로. tool-schema 모양 차이만 분기 유지 | `gateway.py:121-147`, `openai.py:48-66` | (3) | 저 |
| C4 | `OpenAIModelAdapter`의 `background_job` 문자열 스니핑 제거 → `ToolObservation` 플래그로 | `openai.py:78-91` | (2)(3) | 저 |
| C5 | `_proposal_file_payload` 도메인 로직을 core로 (CLI·backend 중복 해소) | `cli.py:1270-1297` ↔ `service.py:380-417` | (3)(4) | 저 |

**의사결정(확정):** C2는 **(a) 타입별 `data` 스키마** 채택. 당초 (b) 단일 reducer를 권고했으나,
조사 결과 두 reader는 시간 범위(실행 중 라이브 vs 사후 재구성)와 출력 모양이 정당하게 달라 통합이
가치보다 리스크가 커서 (a)로 뒤집음. `EVENT_DATA_SCHEMAS`(타입별)를 `validate_run_dir`에 배선해
drift를 발생원에서 차단하고, 엄격도(`additionalProperties`)는 점진 적용.

**완료 기준:** CONTRACTS.md와 코드 일치, 어댑터 3종 테스트 녹색, status/projection 출력 동일.

---

## 5. Phase 3 — 구조: 경계 (A-1) 🔴 고리스크 · 목표(2)(4) 핵심

"범용 엔진" 정체성을 코드에 새기는 단계. 가장 가치 높음.

| ID | 작업 | 위치 | 목표 | 리스크 |
|----|------|------|------|--------|
| A1 | **workspace Protocol 도입** — core가 `LocalWorkspaceBackend` 구상 타입 대신 인터페이스에 의존. 목표(4)가 타입 레벨에서 깨진 지점 수정 | `loop.py:55,560`, `AgentToolContext:67` | **(4)** | 고 |
| A2 | **루프 탈-특수화** — `spec.id == "shell.exec"` / `startswith("web.")` 하드코딩 제거. `ToolSpec`에 선언적 플래그 추가(`emits_diff`/`approval_required`/`preview_kind`). 루프는 도구 이름을 절대 몰라야 함 | `loop.py:1036,1038,1231,1357` | **(2)** | 고 |
| A3 | `_authorize_tool`의 ID 기반 capability 재검사 제거 (정규화 정책이 이미 필터) | `loop.py:1025-1039` ↔ `policy.py:114` | (2) | 중 |
| A4 | `builtin.py`의 workspace private 호출(`_effective_kind`) 해소 — 공개 API로 | `builtin.py:127` | — | 저 |

**전제:** A2는 Phase 1의 B6(emit 헬퍼 통합)이 끝난 뒤가 안전. A1은 A2와 함께 진행하면
`AgentToolContext`의 workspace 의존도 같이 정리됨.

**완료 기준:** core 모듈에서 `LocalWorkspaceBackend`·구체 도구 ID 문자열 참조 0건(grep 검증).
빌트인 도구 동작·이벤트 출력 무변화.

---

## 6. Phase 4 — 구조: 분해 (A-2) 🔴 고리스크

god object 해체. Phase 3 이후 경계가 정리된 상태에서 진행.

| ID | 작업 | 위치 | 목표 | 리스크 |
|----|------|------|------|--------|
| A5 | `AgentToolContext`에서 **shell/jobs/web 서비스 분리** — context는 위임만. `execute_shell`(160줄)·`execute_web_*`(이미 `_run_web_call` 공유)를 동일 패턴으로 | `loop.py:64-542` | (1)(2) | 고 |
| A6 | `AgentLoop.run`(~340줄) 슬림화 — 부트스트랩/루프/teardown 분리, `finally`의 25키 metrics dict 인라인 조립을 빌더로 | `loop.py:557-894` | (1) | 중 |
| A7 | `core/` 패키지 응집도 개선 (선택) — `config/`·`events/`·`proposal/`·`validation/` 서브패키지로 분리 검토. `cancellation.py`(17줄)와 `packages.py`(~760줄)가 한 패키지에 공존하는 grab-bag 해소 | `core/*` | (3) | 중 |
| A8 | `workspace/local.py`(959줄)의 staging change-tracker를 strategy 객체로 분리 — `_uses_overlay()` 6중 분기 해소, `changed_entries`(~100줄) 축소 | `workspace/local.py` | (1) | 중 |

**완료 기준:** `AgentToolContext`·`AgentLoop`·`local.py` 라인 수·메서드 길이 감소, 단위 테스트로
분리된 서비스 개별 검증 가능.

---

## 7. Phase 5 — proposal 일반화 (A-3) 🔴🔴 최고리스크 · 의사결정 게이트

> ⚠️ **착수 전 제품 결정 필요.** 현재 "리뷰 가능 산출물"은 워크스페이스 텍스트 diff에 하드와이어
> (`workspace.diff_patch()` → `diff.patch`/`proposal.json`). 진짜 비-파일 에이전트(예: API 호출형,
> 데이터 산출형)를 지원할 요구가 있을 때만 진행. 없으면 **보류**하고 문서로만 남긴다.

| ID | 작업 | 목표 | 리스크 |
|----|------|------|--------|
| A9 | proposal/diff 추상화 — "리뷰 가능 산출물"을 인터페이스로(파일 diff는 그 한 구현). 비-파일 side_effect도 일급 관측 경로 부여 | (2) | 매우 고 |
| A10 | `SYSTEM_PROMPT` 일반화 — "local workspace agent... modify files" 문구를 도구 세트에서 파생되도록(파일 특화 프레이밍 제거) | (2) | 중 |

---

## 8. Phase 6 — 모듈화 프로파일 (선택) · 목표(1) 강화

목표(1)의 "경량~중량 *구성*"을 일급 개념으로. 현재는 capability/policy를 수동 조립해야 함.

| ID | 작업 | 목표 |
|----|------|------|
| M1 | **AgentProfile 프리셋** — `lightweight`(읽기+검색만), `standard`(파일 쓰기), `heavyweight`(shell+web) 등 명명된 capability/limits/policy 번들. spec에서 한 줄로 선택 | (1) |
| M2 | 프로파일을 `AgentRunSpec.from_json`/CLI에 노출 (`--profile lightweight`) | (1)(3) |

> M1은 Phase 3(탈-특수화)·Phase 4(분해) 이후라야 깔끔하다. 그전엔 프리셋이 거대 객체 위에 얹히는 꼴.

---

## 9. 리스크 / 의존성 매트릭스

```
B1 ─┐
B10 ┴─→ (정확성/경계 가드 필요, 테스트 선행)
B6 ────→ A2          (emit 헬퍼 통합이 루프 탈-특수화의 전제)
B8 ────→ (목표4 즉시 개선, 독립)
C1,C2 ──→ A1,A5      (계약 정리 후 구조 변경)
A1 ─┬─→ A5           (workspace Protocol 후 context 분해)
A2 ─┘
A1,A2,A4,A5 ────────→ M1  (프로파일은 구조 정렬 후)
A9,A10 → 별도 게이트  (제품 결정 선행)
```

**되돌리기 쉬운 정도:** Phase 1 = 커밋 단위로 즉시 revert 가능. Phase 3+ = 인터페이스 변경이라
revert 비용 큼 → 작은 PR로 쪼개고 각 PR마다 골든 산출물 비교.

---

## 10. 권장 실행 순서 (PR 단위)

1. **PR-1** (Phase 0+1 일부): B2·B3·B4·B5·B7·B11 — 무위험 클린업 묶음.
2. **PR-2**: B1 (+해시 동일성 테스트) — 단독, 정확성-critical.
3. **PR-3**: B8·B9·B10 — 경계/이동/policy 통일.
4. **PR-4**: C1·C3·C4·C5 — 계약 정리.
5. **PR-5**: C2 — 이벤트 reducer 통합(의사결정 후).
6. **PR-6~**: A1·A2·A3·A4 — 경계 정렬(작게 쪼개기).
7. **PR-N**: A5·A6·A8 — 분해.
8. 이후 M1·M2, 그리고 (제품 결정 시) A9·A10.

---

## 11. 미결 의사결정 (착수 전 확정 필요)

1. **C2 방향** — 이벤트 `data` 스키마 검증(a) vs 단일 reducer 통합(b). [권장: b]
2. **Phase 5 진행 여부** — 비-파일 산출물 에이전트가 실제 로드맵에 있는가? 없으면 보류.
3. **A7 (core 서브패키지 분리)** — 호환성 영향(import 경로 변경) 감수 가능한가? contracts 재노출로
   외부 영향 차단 가능하나 내부 비용 있음.
4. **Phase 6 프로파일 명세** — 경량/중량의 구체 경계(어떤 capability 묶음)를 제품이 정의해야 함.
