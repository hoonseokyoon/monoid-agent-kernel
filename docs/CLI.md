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

`--llm-gateway-provider` names the **upstream** provider that gateway relays (default
`openai`, matching the reference gateway's own upstream). It is not the transport: it
tags the provider-native reasoning artifacts the gateway carries back, so they only
replay to a matching provider, and it is the provider attributed on the model-call
receipt and OTel's `gen_ai.provider.name`. Pass `--llm-gateway-provider none` for a
gateway whose upstream has no reasoning artifacts — that disables tagging. A deployment
whose gateway fronts something other than OpenAI must set this, or the tag names a
provider that cannot read the items back.

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

Both use **gitignore-style wildcard syntax**, with each normalized workspace-relative path matched
independently. This table is the exact contract:

| pattern | covers | does not cover |
| --- | --- | --- |
| `.env`, `*.key` | that name at **any** depth (`a/b/.env`) | `.envx` |
| `internal/**` | everything under `internal/`, any depth | `internal` itself; `vendor/internal/x` |
| `internal` | the directory and its contents, at any depth | `internals` |
| `internal/` | canonicalized to `internal`; the node and its contents | `internals` |
| `internal/*` | direct children only | `internal/deep/x` |
| `**/id_rsa` | that name anywhere, root included | `id_rsa_backup` |
| `**/secrets` | a `secrets` directory and its contents, wherever it appears | `secret` |

A leading `!` (negation) is **rejected**: it would make the result depend on pattern order, and
merging two policies treats their patterns as a set.
A literal leading `!` is written as `\!` in operator configuration. Serialized JSON keeps the
literal `!` and adds `"path_pattern_encoding": "monoid.literal-bang.v1"`, so current readers can
distinguish it from negation without changing how pre-v0.20 readers match the pattern.
A leading `#` stays literal; it does not turn the rest of the pattern into a gitignore comment.
Leading `./` is normalized. Backslash is a workspace path separator rather than a pattern escape;
the leading `\!` configuration spelling is its only accepted source-level use. Root-only (`/`, `.`,
`./`), malformed, and control-character patterns are rejected during configuration load. The
synthetic workspace root (`.` or an empty path) is not a workspace entry and matches no pattern;
scopes grant paths below it explicitly.

Workspace paths containing C0/DEL controls are rejected. Windows additionally rejects ambiguous
Win32 aliases (trailing dot/space, alternate data streams, reserved devices, and 8.3 alias-shaped
segments). Case and Unicode-normalization aliases, plus symlink and hardlink identity, belong to the
workspace backend because this matcher compares normalized lexical paths and has no workspace root
or volume metadata. This includes the built-in local backend when it runs on a case-insensitive or
normalization-insensitive volume. Hosted deployments should test, canonicalize or reject aliases,
and document those relations as required by the production checklist.

Path rules evaluate the path arguments presented to the policy. The built-in recursive
`fs.copy`/`fs.move`/`fs.delete` flow currently checks its root argument, not every descendant. An
allowed ancestor can therefore contain a denied descendant. Deployments that expose recursive
operations must add backend/tool tree preflight or disallow those operations across mixed-policy
trees.

> **Changed in v0.20.** These were previously matched with `PurePath.match`, where `**` behaved as
> a single `*`. `internal/**` covered one level and missed `internal/deep/x`, while also matching
> `vendor/internal/x`, and `**/id_rsa` never matched a bare `id_rsa` at the root. If you relied on
> a `dir/**` pattern matching that directory at *any* depth, write `**/dir/**`.

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

## Recording model calls

Two opt-in artifacts, off unless their flag is set, on both `monoid run` and
`monoid backend serve`:

- `--model-calls-file` — `model_calls.jsonl`, one metadata record per settled model call,
  including the failed ones: timings, token usage, failure taxonomy and the replay key. It
  carries no content and no endpoint.
- `--model-payload-file` — `model_payloads.jsonl` plus a `model_payloads/` directory of
  content-addressed chunks: the exact request bytes each replay key was hashed over, and the
  settled response bodies with provider reasoning included.

```bash
monoid run \
  --workspace examples/workspaces/edit_markdown_notes \
  --instruction "Read notes.md and create a clearer summary in SUMMARY.md." \
  --runtime-config-file examples/runtime-config.json \
  --llm-gateway-url http://127.0.0.1:8080/internal/llm/turns \
  --model-calls-file --model-payload-file
```

Four things to know before turning the second one on.

**It is content, and `--redact-path` does not reach it.** Path redaction masks the public event
and status stream; private run artifacts keep real paths and contents, as `transcript.jsonl`
already does. A workspace file your redaction policy hides from events is in the corpus in full
if the model was shown it. `--deny-path` helps but does not cover this: it refuses tool and shell
*path arguments*, and removes denied files from an isolated-copy shell workspace — a shell command
that reads a denied file by other means still returns the bytes to the model, and from there to
the corpus.

**It grows with the conversation and nothing deletes it.** A turn's request is the whole
conversation so far, so a long run's corpus lands in the same order of magnitude as its
`transcript.jsonl`, and enabling it can roughly double a run directory. Repeated content is
stored once per activation (tool definitions, messages and observations are content-addressed),
but there is no per-run cap and no retention verb: `monoid gc` collects only chunks *no record
resolves*, so a healthy corpus is never collected. Deleting one means deleting the files.

**Subagents inherit it, into their own directories.** A child run records into its own run
directory beside the parent's, joined by `root_run_id`. `monoid validate` and `monoid gc` each
take one run directory, so a run tree needs one invocation per member.

**Verify with `monoid validate RUN_DIR`** — it re-checks every record against its schema and
re-derives each request digest from the stored bytes, reporting issues rather than showing
content. **Sweep crash litter with `monoid gc RUN_DIR --apply`**, never beside a live writer of
the same directory. Both verbs are covered in
[OBSERVABILITY.md](OBSERVABILITY.md), which also documents the hardlink hazard: a backup that
deduplicates a run directory disables these writers on the next activation, and each says so with
one `WARNING` naming the artifact it will not write.

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

`watch` reads `events.jsonl` directly and does **not** join settled text back in. Since v0.20 a
settle event carries `final_text_digest` instead of model-authored `final_text`, so `--json` shows
the digest where the backend and Studio show the text — they read through a hydrating projection
and `watch` deliberately does not, because "raw JSONL events" is what this flag is for. Kernel
messages such as `Stopped after reaching max steps.` are still inline. To resolve a digest, match it
against the `settled_text` records in the run's `transcript.jsonl`.

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

`jobs` and `job status` print the **public projection** of `artifacts/jobs/<id>/job.json` rather
than the artifact: `command` is dropped and `command_preview` carries the bounded rendering, and
`cwd` and `changed_paths` are redacted against the run's `permission_policy.redact_patterns`. Read
the artifact off disk if you are inside the trust boundary and need the exact command. The response
uses `monoid.public-background-job.v1`; `artifact_schema_version` records the validated durable
input version. A missing manifest redacts every path, and malformed policy metadata is an error.

`monoid status` exits non-zero when the run's `events.jsonl` is corrupt, after printing what it
could project. The projection carries the reason in `event_log_error`. Its other fields combine
the latest readable run artifacts with the valid event prefix and may therefore describe different
points in time. Treat the entire degraded projection as diagnostic; do not poll `state` without
checking the error field.

For the full run-directory artifact set (`events.jsonl`, `transcript.jsonl`,
`diff.patch`, `proposal.json`, …), see [OBSERVABILITY.md](OBSERVABILITY.md#outputs).
