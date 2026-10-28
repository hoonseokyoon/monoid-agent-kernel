# Monoid Agent Kernel

*A durable agent runtime for embedding tools, model calls, and recoverable workflows in your product.*

[![CI](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/workflows/ci.yml/badge.svg)](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/pyproject.toml)

Monoid owns the agent loop, tool execution, permissions, checkpoints, and observation contracts.
Your product supplies the model adapter, workspace, storage, and execution host. Start in one
Python process, then use the same contracts with your backend, gateways, and durable workers.

## Quickstart (no servers)

Run a typed tool and inspect the result delivered to the model. From this checkout's root,
use a Python 3.11+ virtual environment:

```bash
python -m pip install .
python examples/custom_tool_quickstart.py
```

This documentation describes version `0.25.0`.
Install this revision or its matching wheel for these examples. The `examples/` scripts come
from the repository. For editable development, see
[Contributing](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/CONTRIBUTING.md).

Expected output:

```text
status     : completed
final_text : Prepared Ada's weekly report.
tool_result: {'ok': True, 'result': {'customer': 'Ada', 'title': 'Weekly report'}}
```

The [complete example](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/examples/custom_tool_quickstart.py)
defines nested Pydantic input and a typed result, decorates `summarize_report` with `@tool`, and
calls `AgentLoop.from_tools(spec, adapter, [summarize_report]).run_once(...)`. Its scripted
`FakeModelAdapter` invokes the tool and receives the serialized result in the next request.
The example asserts that result and runs offline without credentials.

Choose the next integration step:

| Task | Run after installing this checkout | What to inspect |
|---|---|---|
| [Tools and host resources](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/guides/tools-and-resources.md) | `python examples/custom_tool_quickstart.py` | Validated model inputs and the tool result |
| [Streaming and Stop](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/guides/streaming-and-stop.md) | `python examples/termination_budget.py` | Early break, exit progress, final receipt and reported usage |
| [Background observation](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/guides/streaming-and-stop.md#observe-a-background-run) | `python examples/observe_background_run.py` | Reader disconnect, peer progress, replay/reset and explicit cancel |
| [Approval and recovery](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/guides/approval-and-recovery.md) | `python -m monoid_agent_kernel.reference.host_tool_report --phase prepare --root ./report-state` | Parked approval and original run/task IDs for resume |

The approval guide supplies the second command for a new process. The packaged report recipe
also runs directly from an installed wheel. For a local session with builtin tools, run
`python examples/embedding_local_product.py`; inspect its file output, checked checkpoint, and
completed run. The [embedding handbook](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/EMBEDDING.md)
connects these paths to product topology and hosted storage.

## Install

After version 0.25.0 is published on PyPI, install it with:

```bash
python -m pip install monoid-agent-kernel==0.25.0
```

Before publication, use the [same-revision quickstart](#quickstart-no-servers) or its qualified wheel.
Optional extras work with either package installation or a checkout, for example
`python -m pip install ".[http-async]"`.

| Extra | Use |
|---|---|
| `http-async` | Async gateway streaming, including Studio's live output |
| `openai` | Direct OpenAI adapter for local smoke tests |
| `postgres` | PostgreSQL persistence |
| `object-store-s3` | S3-compatible content-addressed blobs |
| `temporal` | Temporal orchestration |
| `durable-host` | PostgreSQL, ObjectStore, and Temporal composition |

Core keeps provider and platform SDKs optional. Follow the
[hosted composition](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/EMBEDDING.md#choose-one-hosted-assembly)
and [operations guide](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/PRODUCTION_OPERATIONS.md)
when installing production adapters.

## Model Provider Boundary

Replace the quickstart's fake adapter at the `AgentLoop.from_tools` or `from_config` call.
The [custom adapter example](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/examples/custom_model_adapter.py)
implements `next_turn(request) -> ModelTurn` and injects an `EchoModelAdapter` through `from_config`.
For hosted runs, `GatewayModelAdapter` sends model turns to the gateway you operate using a scoped
token; provider credentials stay at that gateway. The
[model adapter contract](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/CONTRACTS.md#model-adapter)
covers streaming, usage, and cancellation.

## Architecture: Contract / Conformance Test / Core Helper Kit / Reference

- **Contract:** import supported specs and protocols from `monoid_agent_kernel.contracts` or the
  package root. See the [contract index](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/CONTRACTS.md#find-a-contract).
- **Conformance Test:** qualify your adapters and runtime through the
  [existing profiles](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/CONFORMANCE.md).
- **Core Helper Kit:** use the [supported helpers](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/CORE_HELPER_KIT.md)
  for workspace, providers, tools, recording, and other contract implementations.
- **Reference:** learn from the [runnable services](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/REFERENCE.md)
  under `monoid_agent_kernel.reference`, including backend, gateways, Studio, and conformance harnesses.

## Stability

Monoid is pre-1.0: minor versions may change public behavior. The
[changelog](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/CHANGELOG.md) and
[compatibility ledger](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/COMPATIBILITY.md)
record source and durable-format changes. The [contract stability categories](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/CONTRACTS.md#stability)
distinguish stable APIs, evolving extensions, helpers, and Reference examples.

For persistent runs, read [Upgrade from v0.24 to v0.25](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/migrations/v0.25.md).
The released v0.24 reader rejects the new invocation/ledger v2 records. The
[release audit](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/V0_25_RELEASE_AUDIT.md)
records source and artifact qualification and publication status. Documentation and image links
target `v0.25.0`; until that tag is created, use the corresponding files in this checkout.

## Running agents from the CLI

The [CLI guide](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/CLI.md)
covers `monoid run`, runtime configuration, workspace modes, proposals, jobs, and streaming JSON.
The default `propose` mode stages file changes for review. Configure path permissions explicitly
for your workspace; defaults permit paths. See the
[workspace threat model](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/security/THREAT_MODEL.md#permissive-by-default).

## Hosted backend and gateways

Use the [backend walkthrough](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/BACKEND.md)
for authenticated submission, run tokens, status/result/events, and proposal retrieval.
The [full-stack example](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/examples/full_stack_integration.py)
connects the Reference backend and gateway; its fake mode exercises the flow offline.
Production hosts own tenant isolation, external effects, storage, and transport lifecycle.

## Observability

Read model-call receipts for termination facts and usage evidence, and `termination_budget()`
for current Stop progress. Numeric usage remains the accounting input. The
[streaming guide](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/guides/streaming-and-stop.md)
shows the observation points and cleanup limits.

The [observability reference](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/OBSERVABILITY.md)
describes events, metrics, tracing, and the optional private content, model-call, and replay
artifacts. It defines their access and content boundaries. Use the
[operations guide](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/PRODUCTION_OPERATIONS.md)
for readiness, migration, drain, backup/restore, and incident handling.

## Memory and default tools

Bind tools explicitly through runtime configuration. The
[tool surface reference](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/TOOL_SURFACE.md)
covers builtin bundles, exposure, authorization, scope, and quotas. Optional memory providers
supply ordinary tools and context; the [memory example](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/examples/memory_quickstart.py)
shows explicit attachment. Delegation and skills have their own
[design guides](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/README.md#integrate-product-capabilities).

## Defaults

`AgentRunSpec` selects workspace, limits, and permissions. `AgentRuntimeConfig` selects the model,
prompt, and tool bindings and can change at a turn boundary. See the
[configuration contract](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/CONTRACTS.md#agentdefinition-and-runtime-config)
and [builtin configuration example](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/examples/minimal_quickstart.py)
for current defaults and explicit configuration.

## See it run

The bundled Agent Studio (`monoid studio serve`) runs the kernel through its Python API and
provides profile editing, chat, approval, and proposal review. Its precompiled UI ships in the
package. See the [Reference guide](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/REFERENCE.md)
for setup.

![Agent Studio showing an approval and proposed report](https://raw.githubusercontent.com/hoonseokyoon/monoid-agent-kernel/v0.25.0/docs/img/studio-v019-main.jpg)

The agent reads a sample CSV, requests approval, and stages a report and chart for review.

## Documentation

Use the [documentation index](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/README.md)
to choose a guide, look up a contract, operate a deployment, or upgrade retained runs.

## Contributing

Follow [Contributing](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/CONTRIBUTING.md)
for setup, test selection, review, and CI. The
[operational rule coverage](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/docs/OPERATIONAL_RULE_COVERAGE.md)
connects contract guarantees to their checks.

## License

Licensed under the [Apache License 2.0](https://github.com/hoonseokyoon/monoid-agent-kernel/blob/v0.25.0/LICENSE).
