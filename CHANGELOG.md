# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and this project is
pre-1.0 (`0.x`): minor versions may include breaking changes, which are called
out in commit messages and here.

## [Unreleased]

## [0.12.0] - 2026-06-27

### Added
- `AgentLoop.from_tools(spec, adapter, tools)` — one call to run with custom
  `@tool`/`ToolSpec` objects (auto-wraps a provider and generates their bindings),
  plus a runnable `examples/custom_tool_quickstart.py`.
- `AgentLoop.validate(config)` / `collect_runtime_config_issues()` — pre-run config
  validation that collects **all** problems as readable messages instead of raising
  on the first.
- Curated `contracts.core` namespace (the ~9 must-know names), a
  `native_agent_runner.tool_ids` constants module, and `list_builtin_tools()`.
- `ToolBinding.for_tool("fs.read")` one-token bindings and bare-string `ref`.
- `native-agent studio doctor` preflight (port / writability / API key / browser /
  OTel checks), a Studio README, and a first-run onboarding panel.
- `otel-export` extra (OTel SDK + OTLP exporter) so Studio's OTel toggle actually
  exports; a README "Observability" section, `examples/otel_tracing.py`, and a
  `docs/` index.
- Public failure events (`run.failed` / `turn.failed`) now carry
  `provider_error_code` and `http_status`, and Studio surfaces them.
- Studio: agent-to-agent (A2A) demo over the durable outbox→inbox fabric; inline
  image preview in the file viewer; open-source project files (contributing guide,
  code of conduct, security policy, CI workflow, environment template).

### Fixed
- MCP client: honor `tools/list` pagination (`nextCursor`) so large servers aren't
  truncated to page one, and reconnect once on a session-expiry (HTTP 404).
- OpenAI adapter: classify provider errors from the response body when the SDK
  exception carries no status — a streaming `429 insufficient_quota` was being
  masked as a generic `502 gateway_bad_response`.
- `fs.read`: graceful binary→media fallback (delegates to `fs.read_media` when the
  `media.input` capability is held; otherwise an actionable error) instead of a hard
  reject.
- Subagent/skill loaders warn on a duplicate id instead of silently dropping it.
- `[otel]` extra was api-only (a no-op); the Studio OTel toggle now has a working
  install path via `[otel-export]`.

### Changed
- Vendored KaTeX locally (woff2 only) so Studio honors its no-network promise.

## [0.11.0]
- Baseline at first public preparation. See the git history for the full
  evolution of the contracts, session/control protocol, capability leases,
  inbox/outbox fabric, durable checkpoints, and the Studio reference app.
