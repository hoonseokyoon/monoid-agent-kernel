# Monoid Documentation

Start with the [offline quickstart](../README.md#quickstart-no-servers): install this revision,
run a typed tool, and inspect its result in the model's next request. Choose the next guide by
what your product needs to do.

## Start and choose a topology

| Task | Read or run | Result |
|---|---|---|
| Run the first tool | [Quickstart](../README.md#quickstart-no-servers), [example](../examples/custom_tool_quickstart.py) | Completed run and typed result observation |
| Embed a local session | [Local product example](../examples/embedding_local_product.py) | File output and checked checkpoint |
| Choose execution and storage ownership | [Embedding handbook](EMBEDDING.md#choose-a-product-topology) | Embedded or hosted composition with explicit owners |
| Run through the CLI | [CLI reference](CLI.md) | Runtime configuration, modes, proposals, jobs and streaming JSON |
| Explore Studio and Reference services | [Reference guide](REFERENCE.md) | Runnable examples of the contracts |

Development examples use the same-revision installation described in the quickstart. Repository
scripts live in `examples/`; the report recipe also ships as an importable package module.

## Integrate product capabilities

| Task | Practical path | Detailed reference |
|---|---|---|
| Write typed tools and inject resources | [Tools and resources](guides/tools-and-resources.md) | [Tool surface](TOOL_SURFACE.md), [tool contract](CONTRACTS.md#tool-contract) |
| Stream a request, show Stop progress, or observe background work | [Streaming and Stop](guides/streaming-and-stop.md) | [Model adapter](CONTRACTS.md#model-adapter), [observability](OBSERVABILITY.md) |
| Resume approval in a new process | [Approval and recovery](guides/approval-and-recovery.md) | [Checked recovery](EMBEDDING.md#checkpoints-recovery-and-upgrades), [control contract](CONTRACTS.md#control-protocol) |
| Supply your model adapter | [Custom adapter example](../examples/custom_model_adapter.py) | [Model adapter contract](CONTRACTS.md#model-adapter) |
| Connect a backend and gateways | [Backend walkthrough](BACKEND.md), [full-stack example](../examples/full_stack_integration.py) | [HTTP contracts](CONTRACTS.md#http-contracts) |
| Delegate to child agents | [Subagent design](SUBAGENT_DESIGN.md) | [Subagent contract](CONTRACTS.md#subagents-agent-as-tool) |
| Attach skills | [First skill tutorial](FIRST_SKILL_TUTORIAL.md), [skills design](SKILLS_DESIGN.md) | [Skills contract](CONTRACTS.md#skills-progressive-disclosure) |
| Attach memory tools | [Memory example](../examples/memory_quickstart.py) | [Tool surface](TOOL_SURFACE.md) |

The [embedding handbook](EMBEDDING.md) connects these paths to workspace/run storage, host authority,
resource factories, runtime configuration, and transports. It also covers production PostgreSQL,
ObjectStore, and Temporal composition. The [DBOS profile](DBOS_REFERENCE.md) remains an experimental
Reference assembly with its own recovery and lifecycle scope.

## Look up a contract

- [Integration contracts](CONTRACTS.md#find-a-contract): Python APIs, model/tool behavior, HTTP,
  durable state, control, and operational invariants.
- [Compatibility ledger](COMPATIBILITY.md): current writers, retained readers, aliases, and
  upgrade/rollback rules.
- [Core Helper Kit](CORE_HELPER_KIT.md): supported implementations and validation/library policy.
- [Conformance](CONFORMANCE.md): profiles, harnesses, and checks for your implementation.
- [Operational rule coverage](OPERATIONAL_RULE_COVERAGE.md): rule-to-helper-to-test evidence.
- [RunnerBackend responsibility map](RUNNER_BACKEND_RESPONSIBILITY_MAP.md): Reference service ownership.

## Operate and secure a deployment

| Need | Reference |
|---|---|
| Startup, migration, draining, backup/restore, retention, incident response | [Production operations](PRODUCTION_OPERATIONS.md) |
| Events, artifacts, usage, tracing and live streams | [Observability](OBSERVABILITY.md) |
| Gateway startup, submission, tokens and result retrieval | [Backend walkthrough](BACKEND.md) |
| Verified boundaries and integrator responsibilities | [Security model](security/SECURITY_MODEL.md), [threat model](security/THREAT_MODEL.md) |
| Deployment readiness | [Production checklist](security/PRODUCTION_CHECKLIST.md) |
| Report a vulnerability | [Security policy](../SECURITY.md) |

## Upgrade retained runs

Read [Upgrade from v0.24 to v0.25](migrations/v0.25.md) for typed tool changes, invocation/ledger v2,
reader-first or coordinated deployment, and rollback. The [v0.25 audit](V0_25_RELEASE_AUDIT.md)
separates integrated implementation from pending documentation and release gates.
For other format transitions, use the [compatibility playbook](COMPATIBILITY.md#upgrade-playbook).

## Contribute and inspect development records

Follow [Contributing](../CONTRIBUTING.md) for development setup, test selection, and PR workflow.
Current contract coverage remains in [Conformance](CONFORMANCE.md) and
[Operational rule coverage](OPERATIONAL_RULE_COVERAGE.md).

The [development history index](history/README.md) groups prior plans, design choices, and
qualification records by version while preserving their original paths. The
[v0.25 release audit](V0_25_RELEASE_AUDIT.md) tracks the current release preparation.
