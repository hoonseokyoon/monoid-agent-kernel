# Documentation index

Start with the top-level [README](../README.md) for install, quickstart, and the run model.
These docs go deeper on specific surfaces:

| Doc | What it covers |
|-----|----------------|
| [CONTRACTS.md](CONTRACTS.md) | The stable integration surface — the Python and HTTP wire contracts you depend on and implement (`AgentLoop`, `ModelAdapter`, `Workspace`, `CheckpointStore`, the `*.v1` envelopes, run artifacts). The contracts/core/reference boundary. |
| [TOOL_SURFACE.md](TOOL_SURFACE.md) | The dynamic, binding-based tool surface — `ToolBinding`, model-name aliasing, exposure/authorization/guidance/scope/quota, and how bindings resolve against the registry. |
| [SUBAGENT_DESIGN.md](SUBAGENT_DESIGN.md) | Agent-as-tool delegation — isolated child runs via the `agent.spawn` tool, progressive disclosure through dynamic context providers. |
| [SKILLS_DESIGN.md](SKILLS_DESIGN.md) | Agent Skills — procedural knowledge delivered through a `ContextProvider`, complementing subagents (execution) and MCP (integration). |

For observability (OpenTelemetry tracing, the streaming surface, `metrics.json`), see the
[Observability](../README.md#observability) section of the top-level README and the runnable
[`examples/otel_tracing.py`](../examples/otel_tracing.py).
