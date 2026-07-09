# CLI: run, watch, and inspect

The `monoid` CLI drives a single kernel run from the command line. This is the
long-form reference for the `run`, `builder`, `watch`, `proposal`, and `jobs`
commands. For the smallest programmatic run, see the
[Quickstart in the README](../README.md#quickstart-no-servers); for the hosted,
multi-tenant path, see [BACKEND.md](BACKEND.md).

## Run

```bash
monoid run \
  --workspace examples/workspaces/edit_markdown_notes \
  --instruction "Read notes.md and create a clearer summary in SUMMARY.md." \
  --runtime-config-file examples/runtime-config.json \
  --llm-gateway-url http://127.0.0.1:8080/internal/llm/turns
```

Run spec and runtime config are separate. `AgentRunSpec` carries workspace,
limits, and permission boundary values — it no longer carries the instruction,
which is delivered as the first user turn (CLI `--instruction`, or
`AgentLoop.run_once()` / `submit()` programmatically). `AgentRuntimeConfig`
carries model, prompt, tool bindings, guidance, scope, quota, shell runtime, and
web runtime values. You can pass a run spec JSON file with a runtime config
file:

```bash
monoid run \
  --spec examples/run-spec.json \
  --instruction "Read notes.md and create a clearer summary in SUMMARY.md." \
  --runtime-config-file examples/runtime-config.json
```

Use the builder CLI to scaffold and preflight those files:

```bash
monoid builder init --target ./my-agent
monoid builder config validate \
  --runtime-config-file ./my-agent/runtime-config.json
monoid builder tools list \
  --runtime-config-file ./my-agent/runtime-config.json
```

`monoid builder init --custom-tool-template` also writes a small `tools.py` provider.
Pass it explicitly when validating or running custom tools:

```bash
monoid builder tools list \
  --tool-module ./my-agent/tools.py:get_tools \
  --runtime-config-file ./my-agent/runtime-config.json
```

Programmatic callers drive the run with `AgentLoop.run_once(instruction)` for the
one-shot case, or `open()` → `submit(user_input)` → `close()` for a multi-turn
session in a single run. Each `submit()` settles when the model returns final
text with no tool calls; the workspace and model continuation thread across
submits. `commit_checkpoint()` re-baselines the proposal between turns when you
want incremental apply.

## Modes: propose vs apply

The default mode is `propose`, which means the kernel creates a proposal package
without committing to tenant source-of-truth storage. Local CLI runs default to
`--workspace-backend overlay`, so writes are staged in an overlay and emitted as
`runs/<run_id>/diff.patch` and `runs/<run_id>/proposal.json` without modifying
the workspace. Container/hosted runs can use `--workspace-backend staging`,
where tools and shell write directly to a staging workspace and the kernel
compares that workspace with `workspace.base.json` to generate the proposal.
Use `--mode apply` for local direct workspace writes.

## Custom workspace backend

Monoid never touches the filesystem directly — it works through a `Workspace`
(the file-storage surface in `monoid_agent_kernel.contracts`). `AgentLoop` builds one
per run with `workspace_factory(spec)`, defaulting to `default_local_workspace_factory`,
which returns the local-filesystem backend. Supply your own factory to back a run with a
different store — a git worktree, an object store, a remote or in-memory filesystem —
without changing the engine:

```python
from monoid_agent_kernel import AgentLoop, Workspace

def my_workspace_factory(spec) -> Workspace:
    return MyWorkspace(spec.workspace_root, mode=spec.mode)

loop = AgentLoop.from_config(spec, adapter, config, workspace_factory=my_workspace_factory)
```

A custom backend must honor the `Workspace` contract suite
(`tests/test_workspace_contract.py`) to be a drop-in: add one `pytest.param` for your
factory and the existing invariants run against it.

## Model, web, and shell surfaces

The default model provider is `gateway`. Hosted runs should call an internal
LLM gateway with a short-lived run token. The kernel should not receive
OpenAI, Anthropic, or other provider API keys.

Web tools are also gateway-backed. `web.search`, `web.fetch`, and `web.context`
are available when runtime config binds those registry tools. The kernel calls
your WebGateway with a short-lived `web_gateway` token. The kernel does not
perform direct web egress and does not receive search-provider credentials.
`web.context` returns
LLM-ready grounding context through a provider-neutral ContextProvider contract.

Shell is available when runtime config binds `shell.exec`, which supports foreground
commands and run-scoped background jobs. A background call returns a `job_id` immediately;
the kernel feeds the job's result back to the model when it finishes (inspect jobs with the
`jobs` / `job` CLI commands below).

## Path permissions

Path permission defaults are permissive: the kernel treats every root-contained file as a
normal workspace file, including dotfiles and keys. **Read the
[Threat Model](security/THREAT_MODEL.md) before exposing a workspace that holds secrets.**
Backends can explicitly deny or redact paths per run:

```bash
monoid run \
  --workspace examples/workspaces/edit_markdown_notes \
  --instruction "Inspect this workspace." \
  --runtime-config-file examples/runtime-config.json \
  --deny-path ".env" \
  --redact-path "*.key"
```

`--permission-policy-file policy.json` accepts:

```json
{
  "deny_patterns": [".env", "*.key"],
  "redact_patterns": ["internal/**"]
}
```

`deny_patterns` blocks tool and shell access. `redact_patterns` masks paths in the public
event/status stream only; private run artifacts keep real paths and contents.

Public events keep file content out of the stream and mask `redact_patterns` paths.
Your backend owns any extra redaction for secret-bearing tool arguments or shell commands
(see [OBSERVABILITY.md](OBSERVABILITY.md#event-sinks)).

## Subagents, Skills, and capability gating

Three optional features on `monoid run`, each off unless its flag is set:

- `--agents-directory DIR` — load subagent definitions (`*.md` with frontmatter) from
  `DIR`, enabling the `agent.spawn` tool so the model can delegate to isolated child runs.
- `--skills-directory DIR` — load Agent Skills (`SKILL.md` with frontmatter) from `DIR`,
  enabling the progressive-disclosure skill tools.
- `--capability-broker path.py:factory` — load a `CapabilityBroker` that gates any tool
  declaring `runtime.requires_lease` behind a scoped, short-lived lease. Required leases fail
  closed when no broker is configured. For local dev, `--auto-grant-capabilities` uses the built-in
  `AutoGrantBroker` (grants every request, scoped to its binding) instead. Pass at most one of the
  two.

See [SUBAGENT_DESIGN.md](SUBAGENT_DESIGN.md) and [SKILLS_DESIGN.md](SKILLS_DESIGN.md)
for the design of these surfaces.

## Streaming JSON

For machine-readable real-time progress:

```bash
monoid run \
  --workspace examples/workspaces/edit_markdown_notes \
  --instruction "Read notes.md and create a clearer summary in SUMMARY.md." \
  --runtime-config-file examples/runtime-config.json \
  --llm-gateway-url http://127.0.0.1:8080/internal/llm/turns \
  --stream-json
```

`--stream-json` writes public redacted events to stdout as JSON Lines. Human
status output goes to stderr in this mode.

## Watch

Replay or follow a run's public event stream:

```bash
monoid watch <run_id> --run-root ./runs --from-start --json
monoid watch <run_id> --run-root ./runs --follow
```

`--json` prints raw JSONL events. The default watch output is a compact human
view.

Inspect the current proposed output snapshot:

```bash
monoid proposal <run_id> --run-root ./runs
monoid proposal <run_id> --run-root ./runs --file SUMMARY.md --json
```

Inspect background shell jobs and logs:

```bash
monoid jobs <run_id> --run-root ./runs
monoid job status <job_id> --run <run_id> --run-root ./runs --json
monoid job logs <job_id> --run <run_id> --stream stdout --tail-bytes 4096
monoid job cancel <job_id> --run <run_id>
```

For the full run-directory artifact set (`events.jsonl`, `transcript.jsonl`,
`diff.patch`, `proposal.json`, …), see [OBSERVABILITY.md](OBSERVABILITY.md#outputs).
