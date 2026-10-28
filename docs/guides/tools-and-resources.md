# Tools and host resources

Use typed tools to accept validated inputs and return results to the model. Add host resources
when a handler needs a workspace adapter or client whose lifetime belongs to the host activation.
The full executable paths are the [typed tool example](../../examples/custom_tool_quickstart.py)
and the packaged [report recipe](../../src/monoid_agent_kernel/reference/host_tool_report.py).

## Run the first tool

Use Python 3.11+ and install this checkout or a wheel built from the same revision. From the
checkout root:

```bash
python -m pip install .
python examples/custom_tool_quickstart.py
```

The example creates temporary workspace and run directories, calls the tool through a scripted
model, and checks the observation in the model's next request. Expected output:

```text
status     : completed
final_text : Prepared Ada's weekly report.
tool_result: {'ok': True, 'result': {'customer': 'Ada', 'title': 'Weekly report'}}
```

The 0.25.0 API requires a matching installation. Repository examples remain in
the checkout; the `monoid_agent_kernel.reference.host_tool_report` module is packaged.

## Use domain models

This excerpt defines the tool used by the quickstart. The full example supplies its run spec,
adapter, and `AgentLoop.from_tools` call:

```python
from pydantic import BaseModel
from monoid_agent_kernel import tool

class Customer(BaseModel):
    name: str

class ReportDraft(BaseModel):
    customer: Customer
    title: str

class ReportSummary(BaseModel):
    customer: str
    title: str

@tool(id="skill.summarize_report", capability="run.control", side_effect="run")
def summarize_report(draft: ReportDraft) -> ReportSummary:
    return ReportSummary(customer=draft.customer.name, title=draft.title)
```

`@tool` validates JSON arguments and passes Python values to sync or async handlers: nested models,
models inside lists or dictionaries, enums, and defaults retain their Python types. Use attribute
access for model fields. Input serializers are skipped. Invalid inputs produce `invalid_tool_args`
and skip the handler.

A top-level `BaseModel` result uses `model_dump(mode="json")`, including datetime, UUID, and enum
conversion. A dictionary dump becomes `ToolResult.content`; other shapes use `content={"result": value}`.
For example, `RootModel[list[int]]([1, 2])` produces `{"ok":true,"result":{"result":[1,2]}}`.
Custom serializers follow their actual output shape. Explicit `ToolResult`, dictionaries, and
scalars retain their envelopes. Convert models nested inside an arbitrary dictionary or explicit
`ToolResult.content` yourself.

Choose a result type containing the data the model should receive. The returned instance's
exclusion and `SecretStr` rules apply, including subclass fields. The return annotation adds no
validation or filtering. See the [tool contract](../CONTRACTS.md#tool-contract) and
[v0.25 source migration](../migrations/v0.25.md#update-application-code).

## Connect product resources

The embedding product supplies these inputs:

| Input | Where to connect it |
|---|---|
| Workspace and durable run storage | Set `AgentRunSpec.workspace_root` and `run_root`; the [local product example](../../examples/embedding_local_product.py) also supplies a checked checkpoint store. |
| Model adapter | Pass it to `AgentLoop.from_tools` or `from_config`. [custom_model_adapter.py](../../examples/custom_model_adapter.py) implements and injects `EchoModelAdapter`; [BACKEND.md](../BACKEND.md) covers hosted gateways. |
| Tool exposure and policy | Use `AgentRuntimeConfig.tools` and `ToolBinding` with explicit authorization, scope, and quotas. |
| Host factory | Pass `host_tool_bindings` when constructing the loop. Supply the factories again after restore. |

A shared process-local key joins the factory and handler. This wiring excerpt assumes the product
has supplied `spec` and `adapter`:

```python
from monoid_agent_kernel import AgentLoop, ToolContext, tool
from monoid_agent_kernel.tools.host_bindings import (
    HostToolBinding, HostToolKey, ToolWorkspace, require_host_tool_binding,
)

REPORTS = HostToolKey[ToolWorkspace](
    "reports.workspace", accepts=lambda view: isinstance(view, ToolWorkspace),
)

@tool(id="report.save", side_effect="write")
def save_report(ctx: ToolContext, title: str) -> dict:
    workspace = require_host_tool_binding(ctx, REPORTS)
    digest = workspace.write_bytes("reports/draft.md", title.encode(), create_dirs=True)
    return {"sha256": digest}

loop = AgentLoop.from_tools(
    spec, adapter, [save_report],
    host_tool_bindings=(HostToolBinding(REPORTS, open=lambda activation: activation.workspace_resource()),),
)
```

For complete scoped wiring, read `_report_loop` in the
[report recipe](../../src/monoid_agent_kernel/reference/host_tool_report.py). Its `open_reports`
factory returns `activation.workspace_resource()`. Its runtime config binds report paths to
`reports/**` and requires approval for `report.read`. Follow
[approval and recovery](approval-and-recovery.md) to run that flow across two processes.

## Enforce authority and lifetime

Each workspace operation checks the binding and current surface scopes, operator `PermissionPolicy`,
original call, activation lifetime, and writer authority. Writes require an admitted
`side_effect="write"` tool. Reads and existence checks also respect path scopes, using the existing
normalized lexical path matching contract.

Factories run once per activation before checkpoint restoration. Read restored content inside
handlers. `read_bytes` returns `(bytes, sha256)`; `write_bytes` returns a digest and preserves
`create_dirs`, `expected_sha256`, `overwrite`, and backend mode behavior. Overlay writes remain
proposals; staging backends use the host's staging directory.

Custom resources implement synchronous `bind_call(lease)` and local `close()`. Their methods check
`lease.assert_active()` or use `lease.guard_external_call(callback)` before and after client
operations. Async client methods check before and after `await`. The host owns shared pools,
external fencing, and idempotency. Callbacks run without kernel authority locks; the host controls
callback duration and already-started external effects.

Adapters close in reverse order once on close, discard, and committed park release, including
partial factory failure. Normal cleanup leaves shared host writer authority active. Call views
expire on return, failure, cancellation, timeout, and abandonment. A retained context keeps its
original call identity. Pass that context explicitly to joined helper threads.

Children and forks receive no bindings by default. Opt in per binding:

```python
HostToolBinding(
    REPORTS, open=lambda activation: activation.workspace_resource(),
    inherit_to_children=True,
)
```

Every hop opens fresh adapters and call views. Operations check the child and each original
ancestor's activation and writer authority. Parent close, committed release, or revoke stops
new descendant access. Child cleanup leaves parent and sibling adapters open; a restored parent
creates a new lineage. Children inherit the parent's effective operator policy, including a
constructor override. Model-call cancellation and drain settings retain their separate controls.

## Handle failures

Missing, mistyped, or expired resources produce fixed diagnostics such as `host_tool_binding_missing`,
`host_tool_binding_type`, `host_tool_context_unavailable`, `tool_call_not_in_flight`, and
`host_tool_resource_closed`. Factory, binder, type-check, and close failures use
`host_tool_binding_open/bind/accepts/close`. Recreate resources through the host factory after recovery.

Ordinary result-serialization failures produce `tool_result_unportable`, category `tool`,
`retryable=true`, and a fixed diagnostic. That flag informs the model; it does not automatically
rerun the tool. Direct run-control exceptions and control causes retained by
`PydanticSerializationError` preserve their classification. Handler-local `asyncio.CancelledError`
becomes `tool_handler_cancelled`; host/run cancellation follows the run's control boundary.

The [typed-tool qualification record](../V0_25_P5_DX1_PR_PLAN.md) confirms cause preservation for
normal nested fields with Pydantic 2.6.0 and 2.13.4. A custom serializer that calls another
model's `model_dump()` can discard a cause upstream;
the decorator then reports the available serialization failure. Prefer normal nested fields when
that cause must remain observable.

For detailed policy, use [ToolBinding](../TOOL_SURFACE.md#toolbinding),
[scope and runtime](../TOOL_SURFACE.md#scope-and-runtime), and the
[embedding authority map](../EMBEDDING.md#preserve-the-authority-boundary).
