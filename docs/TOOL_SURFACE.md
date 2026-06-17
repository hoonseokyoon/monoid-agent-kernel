# Tool Surface Design

## 목적

Tool Surface는 한 run 또는 한 turn에서 agent에게 드러나는 도구 인터페이스 전체를 뜻한다.

Runner는 전체 tool catalog를 보유한다. Tool Surface는 그중 어떤 tool을 모델에게 바로 보여줄지, 어떤 tool을 검색으로 찾게 할지, 어떤 tool을 숨길지, 어떤 호출을 실행할지, 어떤 사용 guidance를 줄지를 결정한다.

이 설계는 Runner가 backend 또는 상위 orchestration layer에서 내려온 정책 값을 해석할 수 있는 제어 경로를 제공한다.

```text
Backend / Orchestrator
  -> Tool surface policy values
  -> Runner ToolSurfaceResolver
  -> Per-turn ToolSurfaceSnapshot
  -> ModelRequest.tools + tool.search index + runtime enforcement
```

## 핵심 개념

### ToolCatalog

Runner가 알고 있는 전체 tool 목록이다.

구성 요소:

- builtin tools: `fs.read`, `fs.write`, `shell.exec`, `web.search`, `run.finish` 등
- custom tools: `ToolProvider`가 등록한 domain-specific tools
- dynamic tools: run 또는 turn 시점에 추가되는 tools
- remote tools: MCP, gateway, plugin, connector 등에서 발견된 tools

ToolCatalog는 “존재하는 tool 전체”를 표현한다. 모델이 볼 수 있는 목록은 ToolSurfaceSnapshot이 표현한다.

### ToolAuthorizationPolicy

Tool call을 실제로 실행할 수 있는지 결정하는 정책이다.

대표 decision:

- `allow`: 호출 가능
- `deny`: 호출 차단
- `ask`: 실행 전 승인 필요

부가 제약:

- `quota`: 호출 횟수, 비용, 결과 크기 같은 사용량 제한
- `scope`: path, domain, tenant resource, action 등 접근 범위 제한
- `approval`: 승인 주체, 승인 timeout, 승인 후 재개 방식
- `reason`: 차단 또는 승인 요구 사유

Authorization은 모델 노출 여부와 별개다. 모델에게 보이지 않은 tool도 호출 시점에는 authorization을 다시 검사한다.

### ToolExposurePolicy

Tool을 모델에게 어떤 방식으로 드러낼지 결정하는 정책이다.

권장 exposure 상태:

| Exposure | 즉시 호출 | 검색 가능 | 존재 암시 | 의미 |
|---|---:|---:|---:|---|
| `immediate` | 예 | 선택 | 예 | 핵심 tool. 매 turn schema를 바로 전달한다. |
| `searchable` | 로드 후 가능 | 예 | 예 | tool.search로 찾을 수 있다. full schema는 필요할 때 노출한다. |
| `hidden` | 아니오 | 아니오 | 아니오 | 모델의 선택 공간에서 제거한다. |

`deferred`는 exposure 상태보다 schema loading 방식으로 다룬다.

```json
{
  "tool_id": "crm.list_open_orders",
  "exposure": "searchable",
  "defer_schema": true
}
```

`hidden`과 `deny`는 다른 축이다.

```text
hidden = 모델에게 보여주지 않는다.
deny   = 호출해도 실행하지 않는다.
```

일반적으로 `hidden` tool은 `authorization=deny`도 함께 가진다. 모델이 이전 context나 추측으로 hidden tool 이름을 호출할 수 있기 때문이다.

### ToolSearch

Searchable tools를 찾는 보조 도구다.

`immediate` tool은 검색 없이 바로 호출 가능하다. Tool search는 큰 catalog에서 관련 tool을 찾는 보조 경로다.

검색 대상:

- tool id/name/title
- namespace description
- tool description
- parameter names and descriptions
- guidance tags
- risk and capability metadata

검색 결과는 full schema를 바로 줄 수도 있고, 다음 turn에 selected tools로 로드할 수도 있다. Runner v1에서는 “검색 결과를 다음 turn의 immediate tools로 승격”하는 방식이 가장 단순하다.

### ToolGuidance

모델에게 주는 tool 사용법과 사용정책 정보다.

Runner는 authorization/scope/quota 검사로 보안을 강제한다. Guidance는 모델이 올바른 tool을 고르고 올바른 인자를 구성하도록 돕는다.

Guidance 종류:

- `description`: tool 기능, 언제 쓰는지, 언제 쓰면 안 되는지
- `parameter_guidance`: parameter 의미, format, constraints
- `result_guidance`: 반환값의 의미와 후속 처리 방법
- `policy_guidance`: 승인 필요, destructive 여부, retry 가능성
- `examples`: 복잡한 tool의 대표 input examples
- `namespace_guidance`: tool group 전체에 대한 짧은 설명

권장 위치:

- immediate tool: tool description 또는 schema metadata에 병합
- searchable tool: tool search index와 search result summary에 포함
- global policy: system prompt 또는 dynamic context에 짧게 포함
- policy violation: tool error envelope로 구체적으로 반환

### ToolsetDeltaNotice

이전 turn과 비교해 tool surface가 의미 있게 바뀌었을 때 모델에게 주는 짧은 알림이다.

Delta notice는 보조 신호다. 이번 turn의 `ModelRequest.tools`와 runtime enforcement가 실제 tool surface를 정의한다.

좋은 예:

```text
Tool update: GitHub tools are now searchable. Core file tools remain available.
```

```text
Tool update: write tools are unavailable. Continue in read-only mode.
```

나쁜 예:

```text
The tool environment has changed substantially. Review all tools carefully before proceeding...
```

Delta notice는 짧고 무시 가능해야 한다. 길고 세부적인 정책은 tool descriptor와 policy object에 둔다.

## 전체 모델

```text
ToolCatalog
  -> capability and base policy filtering
  -> ToolAuthorizationPolicy
  -> ToolExposurePolicy
  -> ToolGuidance enrichment
  -> ToolSurfaceSnapshot
```

`ToolSurfaceSnapshot`은 한 turn에서 모델과 runtime이 사용하는 고정된 tool surface다.

```python
@dataclass(frozen=True)
class ToolSurfaceSnapshot:
    turn_id: str
    immediate_tools: tuple[ToolSpec, ...]
    searchable_tools: tuple[ToolSearchEntry, ...]
    hidden_tool_ids: tuple[str, ...]
    authorizations: dict[str, ToolAuthorization]
    guidance: tuple[ToolGuidance, ...]
    delta_notice: str | None
    surface_hash: str
```

Tool call은 해당 call이 발생한 turn의 snapshot을 기준으로 검증한다.

```text
Model sees immediate_tools
Model may call tool.search
Model may call an immediate tool
Runner checks snapshot authorization
Runner checks scope/quota/approval
Runner executes or returns policy error
```

## Runner 통합 지점

현재 runner는 bootstrap에서 `visible_tool_specs`를 한 번 계산한다.

```text
registry
  -> capability
  -> ToolPolicy
  -> visible_tool_specs
```

Tool Surface 도입 후에는 turn마다 snapshot을 계산한다.

```text
registry
  -> ToolSurfaceResolver.resolve(turn_context)
  -> ToolSurfaceSnapshot
  -> ModelRequest.tools = snapshot.immediate_tools + tool.search if enabled
```

### ToolSurfaceResolver

Runner 내부의 핵심 extension point다.

```python
class ToolSurfaceResolver(Protocol):
    def resolve(
        self,
        *,
        registry: ToolRegistry,
        run_spec: AgentRunSpec,
        turn: TurnContext,
        previous_snapshot: ToolSurfaceSnapshot | None,
    ) -> ToolSurfaceSnapshot:
        ...
```

기본 resolver는 기존 동작과 동일하게 동작한다.

```text
all currently visible tools -> immediate
no searchable tools
ToolPolicy allow/deny/ask 그대로 적용
no delta notice
```

이 기본값은 하위호환을 지킨다.

### Tool call 검증

Tool call 실행 시 runner는 다음 순서로 검사한다.

1. tool name을 registry에서 resolve
2. 해당 turn snapshot에 존재하는 immediate tool인지 검사
3. authorization decision 확인
4. JSON Schema validation
5. scope 검사
6. quota 검사
7. approval 필요 여부 검사
8. tool handler 실행

`hidden` 또는 `searchable` 상태의 tool이 full schema 로드 없이 호출되면 policy error를 반환한다.

```json
{
  "ok": false,
  "error": {
    "code": "tool_not_available_in_turn",
    "category": "policy",
    "retryable": false,
    "message": "Tool crm.refund_order is not available in this turn. Use tool.search or continue with available tools."
  }
}
```

## Tool search 동작

Runner는 `tool.search`를 일반 tool처럼 제공한다.

입력:

```json
{
  "query": "refund customer order",
  "max_results": 5
}
```

출력:

```json
{
  "matches": [
    {
      "tool_id": "crm.refund_order",
      "title": "Refund order",
      "summary": "Issue a refund for a paid order.",
      "risk": "write",
      "requires_approval": true,
      "load_hint": "available_next_turn"
    }
  ]
}
```

검색 결과 처리는 두 단계로 나눈다.

1. `tool.search`는 후보를 반환한다.
2. 다음 turn의 resolver가 검색된 후보를 `immediate`로 승격할지 결정한다.

이 방식은 replay와 audit을 단순하게 유지한다. Search result만 받은 상태에서 모델이 full-schema tool을 즉시 호출하는 race도 막는다.

## Guidance 설계 원칙

### Tool description에 넣을 정보

각 tool description은 “신입 엔지니어가 이것만 보고 안전하게 사용할 수 있는가?”를 기준으로 작성한다.

포함할 정보:

- tool의 목적
- 사용해야 하는 상황
- 사용하지 말아야 하는 상황
- parameter 의미와 format
- 반환값 의미
- 부작용과 위험
- 흔한 실패와 복구 방법

예:

```text
Delete a file or directory from the workspace. Use only for generated files or files the user explicitly asked to remove. Do not use for source files unless the instruction clearly requests deletion. Requires recursive=true for non-empty directories.
```

### Policy guidance에 넣을 정보

Policy guidance는 짧게 작성한다.

예:

```text
Requires approval before execution.
```

```text
Allowed only under docs/** and examples/**.
```

```text
External network access. Prefer trusted documentation domains.
```

### Examples를 넣을 때

Examples는 복잡한 tool에만 넣는다.

좋은 대상:

- nested input
- optional parameter 조합이 중요한 tool
- 비슷한 tool 사이의 선택이 어려운 경우
- domain-specific identifier format이 있는 경우

단순한 single-parameter tool에는 examples를 넣지 않는다.

## 동적 변경 알림

Runner는 snapshot 간 차이를 계산해 의미 있는 변경만 알린다.

알릴 변경:

- immediate tool 추가/제거
- searchable namespace 추가/제거
- read-only/write mode 전환
- approval policy 변경
- quota exhaustion
- scope 축소

알리지 않을 변경:

- 검색 ranking의 작은 변화
- hidden tool 내부 상태 변경
- 모델이 호출할 수 없는 내부 tool 변경
- 사용자에게 의미 없는 metadata 변경

Event 예:

```json
{
  "type": "tool.surface.updated",
  "data": {
    "surface_hash": "sha256:...",
    "immediate_tools": ["fs.read", "text.search", "run.finish", "tool.search"],
    "searchable_count": 42,
    "hidden_count": 7,
    "delta_notice": "Tool update: GitHub tools are now searchable."
  }
}
```

Model dynamic context 예:

```text
Tool update: GitHub tools are now searchable. Use tool.search for repository operations.
```

## Manifest, transcript, replay

동적 Tool Surface를 지원하면 replay 계약이 중요해진다.

Manifest에는 base catalog와 resolver 정보를 기록한다.

```json
{
  "tool_surface": {
    "resolver": "default",
    "policy_version": "tool-surface.v1",
    "dynamic": true
  }
}
```

Transcript에는 turn별 snapshot을 기록한다.

```json
{
  "kind": "tool_surface_snapshot",
  "turn_id": "turn_0003",
  "surface_hash": "sha256:...",
  "immediate_tools": ["fs.read", "run.finish", "tool.search"],
  "searchable_tools": ["crm.refund_order", "crm.list_open_orders"],
  "authorizations": {
    "crm.refund_order": {"decision": "ask", "reason": "write_operation"}
  }
}
```

Replay는 snapshot 기준으로 tool availability를 복원한다. 현재 registry 상태로 과거 run을 재해석하지 않는다.

## 권장 단계적 도입

### Phase 1: 구조만 추가

- `ToolSurfaceSnapshot` 타입 추가
- 기본 `ToolSurfaceResolver` 추가
- 기존 `visible_tool_specs` 계산을 snapshot 생성으로 감싼다
- 동작은 현재와 동일하게 유지
- manifest/transcript에 `surface_hash` 기록

### Phase 2: ExposurePolicy 추가

- `immediate/searchable/hidden` 상태 추가
- hidden은 model request와 tool search에서 제외
- tool call 시 snapshot membership 검사

### Phase 3: Tool search 추가

- builtin `tool.search` 추가
- searchable tool index 구성
- search result를 다음 turn immediate 후보로 승격
- search events 기록

### Phase 4: Guidance 확장

- `ToolSpec`에 guidance metadata 추가
- description enrichment 추가
- examples 지원
- namespace guidance 지원

### Phase 5: Authorization 확장

- per-tool quota
- ask approval lifecycle
- path/domain/resource scope
- policy-denied error envelope 정리

## 설계 원칙

1. Tool Surface는 모델에게 보이는 도구 세계다.
2. ToolCatalog는 전체 가능성이고, ToolSurfaceSnapshot은 한 turn의 현실이다.
3. 노출 정책과 실행 권한 정책은 분리한다.
4. 숨긴 tool도 실행 시점에서 deny한다.
5. Tool search는 보조 경로다. 핵심 tool은 항상 immediate로 유지한다.
6. Guidance는 모델을 돕는다. Enforcement는 runner가 수행한다.
7. 동적 변경은 snapshot과 event로 기록한다.
8. 모델에게 주는 delta notice는 짧게 유지한다.
9. Replay는 과거 snapshot을 기준으로 한다.
10. 기본 resolver는 현재 동작과 동일해야 한다.
