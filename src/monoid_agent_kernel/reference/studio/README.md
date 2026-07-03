# Agent Studio (reference app)

Agent Studio is the bundled "installable agent app" — it boots the reference LLM gateway, Monoid
backend, and a single-page UI in one process, so you can watch a real agent plan, run code in a
workspace, and report back. It drives the kernel through its Python API behind a thin
backend-for-frontend (BFF); the browser never sees a provider key.

> **Reference example.** Studio lives under `monoid_agent_kernel.reference.*`.
> Core never imports it; build production apps against the contracts in
> [docs/CONTRACTS.md](../../../../docs/CONTRACTS.md).

## Launch

```bash
monoid studio serve          # start the server, keep it running (window detachable)
monoid studio app            # server + a desktop window bound together
monoid studio open           # open a window for an already-running server
monoid studio doctor         # preflight: check ports, dirs, keys, browser, OTel
monoid studio accept         # deterministic offline acceptance check, emits JSON
```

Run `studio doctor` first if anything looks off — it turns late, cryptic setup failures into an
upfront pass/fail checklist with remediation.

### Flags & defaults

| Flag | Default | Meaning |
|------|---------|---------|
| `--host` | `127.0.0.1` | Bind address for the UI. |
| `--port` | `8799` | UI port. |
| `--workspace` | `studio-workspace` | Folder the agent works in (created if missing). |
| `--run-root` | `runs` | Where run artifacts (events, proposals, metrics) are written. |
| `--provider` | `offline` | `offline` = keyless echo model; `openai` = `OpenAIModelAdapter` (needs `OPENAI_API_KEY`). |
| `--skills-directory` | bundled sample | Directory of Agent Skills (`SKILL.md` files). |
| `--no-skills` | off | Disable Agent Skills entirely. |
| `--mcp` | off | Attach the bundled offline reference MCP server and expose its tools. |
| `--env-file` | `.env` | Load simple `KEY=VALUE` entries before provider checks and server start. |
| `--no-env-file` | off | Skip env-file loading. |

**Offline vs. live.** With `--provider offline` (the default), the model is a keyless *echo* model:
it replies but does not reason or call tools — handy for a zero-setup look at the UI. For a real
agent that plans, writes files, and runs tools, launch with `--provider openai` and an
`OPENAI_API_KEY` in the environment.

`studio serve`, `studio app`, and `studio doctor` load `.env` by default when it exists. Existing
process environment variables take precedence over env-file values. Use `--env-file <path>` to point
at another file, or `--no-env-file` to rely only on the process environment.

## Panels

- **Agent Configuration / Profile** — the left panel contains the active profile switcher and
  profile-scoped chat history. Built-in profiles are `default`, `reviewer`, and `builder`.
- **Agent Chat** — the center panel contains the conversation and composer. A first-run
  empty-state offers a few one-click prompts; it clears on your first message. Streamed tokens and
  tool activity appear inline.
- **Side Panel** — the right panel has tabs for Workspace, Trace, and Live Config. Workspace holds
  files, jobs, proposed changes, and file previews. Trace shows the live event tree. Live Config
  exposes model, reasoning, OTel, and capability toggles.

Profile state is lightweight Studio metadata. The profile catalog is fixed in Studio, and a small
`studio-profiles.json` sidecar under the run root maps run ids to profile ids so scoped history
survives a restart. Profiles are Studio-only history scopes; the kernel contract and runtime config
stay unchanged.

Stable test hooks are present on the main shell (`data-testid="studio-shell"`), left config panel,
profile switcher/list, chat log, composer, right-panel tabs, settings/config surfaces, and
capability toggles.

## Acceptance

`monoid studio accept` starts Studio on an ephemeral port with the offline provider, checks the key
static/API routes, verifies settings/capabilities/profile history, runs one deterministic chat, and
prints JSON. Browser smoke is optional and stays outside the default command.

## Capabilities → tools

Studio's settings expose capabilities; toggling one binds its tools for the next turn:

| Capability | Tools it binds |
|------------|----------------|
| Read files | `fs.read` |
| Write files (staged as a proposal) | `fs.write` |
| Ask the human for approval | `hitl.request` |
| Run shell commands + background jobs | `shell.exec` |
| Search & fetch the web | `web.search`, `web.fetch`, `web.context` |
| Delegate subtasks to a subagent | `agent.spawn` |
| Use Agent Skills *(when enabled)* | progressive-disclosure skill tools |
| Use a connected MCP server *(with `--mcp`)* | the MCP server's tools |

`run.update_plan` is always bound so the agent's plan is observable in the trace.

## Observability

Toggle OpenTelemetry export in settings to emit GenAI spans (`invoke_agent → chat / execute_tool`)
to an OTLP collector; install the exporter with `pip install 'monoid-agent-kernel[otel-export]'`.
See the top-level [Observability](../../../../README.md#observability) section for the full story.
