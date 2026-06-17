# Tool Surface

Tool Surface는 한 turn에서 모델에게 드러나는 도구 인터페이스다. Native runtime 모델에서는
`AgentRuntimeConfig.tools`의 `ToolBinding` 목록이 surface의 유일한 public 입력이다.

## Flow

```text
AgentRuntimeConfig
  + ToolRegistry
  -> BoundToolCatalog
  -> ToolSurfaceSnapshot
  -> ModelRequest.tools
  -> model tool call
  -> model_name -> BoundTool -> base ToolSpec.handler
```

Runner는 registry에 전체 builtin/custom tool을 보관한다. Registry tool은 구현체다.
`ToolBinding`은 구현체를 agent-facing tool로 노출하는 public 단위다.

## ToolBinding

```json
{
  "binding_id": "read_notes",
  "model_name": "read_notes",
  "ref": {"kind": "registry", "tool_id": "fs.read"},
  "exposure": "immediate",
  "authorization": "allow",
  "guidance": {"summary": "Read notes and source files before editing."},
  "scope": {"allowed_paths": ["docs/**"]},
  "quota": {"max_calls_per_run": 20},
  "runtime": {},
  "title": "Read notes",
  "summary": "Read a workspace file.",
  "risk": "read",
  "metadata": {}
}
```

- `binding_id` is the audit and runtime identity.
- `model_name` is the model-facing function name. It defaults to
  `binding_id` with dots replaced by underscores.
- `ref.tool_id` points to a registered `ToolSpec`.
- `exposure` controls visibility: `immediate`, `searchable`, or `hidden`.
- `authorization` controls execution: `allow`, `ask`, or `deny`.
- `guidance` enriches model-facing descriptions and search entries.
- `scope`, `quota`, and `runtime` drive enforcement.

The same registry tool can be bound multiple times. Each binding can have a
different name, guidance, scope, quota, and runtime. Duplicate `binding_id` and
duplicate resolved `model_name` values are invalid.

## BoundToolCatalog

`compile_bound_tool_catalog(config, registry)` validates runtime config and
produces:

- `BoundTool.binding`
- `BoundTool.base_spec`
- `BoundTool.model_spec`
- `BoundTool.model_name`
- `BoundTool.authorization`

The model sees `model_spec`. Execution uses `base_spec.handler`.

Unbound registry tools stay outside the catalog. Runner does not create hidden
deny rules for unbound tools.

## Exposure

| Exposure | In model tools | In search | Callable this turn |
|---|---:|---:|---:|
| `immediate` | yes | no | yes |
| `searchable` | no | yes | after selected for next turn |
| `hidden` | no | no | no |

`searchable` bindings appear in `tool.search` results. When the model selects a
search result, the binding id is queued as a pending binding load. The resolver
can promote it to `immediate` at the next turn boundary. Mid-turn changes do
not affect the current snapshot.

## Authorization

Authorization is keyed by `binding_id`.

- `allow`: execute when the binding is immediate in the current snapshot
- `ask`: return an approval-required denial for now
- `deny`: keep the binding out of the callable surface and reject stale calls

Quota is also keyed by `binding_id`. Binding a registry tool twice gives each
binding separate call counts.

## Scope And Runtime

Scope is declarative enforcement input:

- `allowed_paths`, `denied_paths`
- `allowed_domains`, `blocked_domains`
- `command_allow_prefixes`, `command_deny_prefixes`
- `env_allowlist`

Runtime holds implementation options. Shell bindings read `runtime.shell`.
Web bindings read `runtime.web`.

Example shell binding:

```json
{
  "binding_id": "run_tests",
  "ref": {"kind": "registry", "tool_id": "shell.exec"},
  "model_name": "run_tests",
  "scope": {
    "command_allow_prefixes": ["pytest", "python -m pytest"],
    "env_allowlist": ["PYTHONPATH"]
  },
  "runtime": {
    "shell": {
      "approval_mode": "auto-approve",
      "default_timeout_s": 120,
      "max_output_bytes": 200000,
      "execution_workspace": "isolated-copy"
    }
  }
}
```

Example web binding:

```json
{
  "binding_id": "search_docs",
  "ref": {"kind": "registry", "tool_id": "web.search"},
  "scope": {"allowed_domains": ["docs.example.test"]},
  "runtime": {
    "web": {
      "max_calls": 10,
      "max_results": 5,
      "timeout_s": 10
    }
  }
}
```

## Tool Search

`ToolSearchConfig` controls the synthetic `tool.search` binding:

```json
{"enabled": true, "top_k": 5, "binding_id": "tool.search", "model_name": "tool_search"}
```

`tool.search` appears only when search is enabled and at least one binding is
searchable. Results use binding identity:

```json
{
  "matches": [
    {
      "binding_id": "search_docs",
      "tool_id": "web.search",
      "exported_name": "search_docs",
      "title": "Search docs",
      "summary": "Search trusted documentation.",
      "risk": "read",
      "requires_approval": false,
      "load_hint": "available_next_turn"
    }
  ]
}
```

## Snapshot

`ToolSurfaceSnapshot` is immutable for a turn:

```json
{
  "kind": "tool_surface_snapshot",
  "turn_id": "turn_0002",
  "surface_hash": "sha256:...",
  "immediate_tools": [{"id": "read_notes", "exported_name": "read_notes"}],
  "searchable_tools": [{"id": "search_docs", "exported_name": "search_docs"}],
  "search_entries": [{"binding_id": "search_docs", "tool_id": "web.search"}],
  "hidden_tool_ids": [],
  "authorizations": {
    "read_notes": {"binding_id": "read_notes", "decision": "allow"}
  },
  "delta_notice": ""
}
```

Tool execution checks the snapshot captured for the turn:

1. Resolve call name to a bound tool by `model_name` or `binding_id`.
2. Confirm the binding is immediate in the snapshot.
3. Confirm authorization allows execution.
4. Validate JSON arguments against the bound model spec.
5. Check binding scope and quota.
6. Execute the base registry handler.

## Runtime Updates

The loop reads runtime config at turn start. If the hash changed, it emits
`agent.config.updated` and writes an `agent_runtime_config_snapshot` transcript
record. The current turn continues with its existing `BoundToolCatalog` and
`ToolSurfaceSnapshot`.

This gives backend-driven mutation a clear rule: config replacement is accepted
immediately by the backend and observed by the runner at the next turn boundary.

## Replay

Replay relies on recorded snapshots:

- runtime config snapshot records config hash and binding ids
- tool surface snapshot records model-facing specs and authorizations
- tool calls resolve through the recorded model-facing binding identity

The registry can evolve after a run. Recorded snapshots preserve the old turn
surface for audit.
