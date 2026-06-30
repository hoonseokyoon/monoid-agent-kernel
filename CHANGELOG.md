# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and this project is
pre-1.0 (`0.x`): minor versions may include breaking changes, which are called
out in commit messages and here.

## [Unreleased]

## [0.13.0] - 2026-06-29

### Added
- **OutputValidator** — developer-supplied validation of the final response with a
  bounded re-prompt loop. Register via `AgentLoop(output_validators=...)`; validators
  run **default-on**, opt out per-run with `OutputValidatorBinding(enabled=False)`. A
  rejection re-prompts the model with the validator's feedback, bounded by the new
  `RunLimits.max_output_retries`. Adds `AgentRunResult.final_output` /
  `outputs[validator_id]` / `output_as(Model)`, the `output.validator.*` event family
  (satisfied / validation.failed / exhausted / error / skipped), OTel span events, and
  a Studio + backend (`RunnerBackend(output_validators=...)`) seam.
- `ModelTurn.stop_reason` promotion: a provider refusal or truncation now settles as
  `output_refused` / `output_truncated` instead of a generic "neither text nor tool
  calls" model error.

### Changed
- The settle path is now a pure `_decide_settle` (classification) plus a single
  `_apply_settle` (state mutation + events + Suspension), and the four run.finish
  metadata fields collapsed into one `pending_finish` value — a behavior-preserving
  refactor that makes the validation lifecycle a single atomic transition.

### Fixed
- OpenAI adapter: capture `response.incomplete` in the streamed turn so truncations and
  refusals carry the correct `stop_reason`.
- Backend: `status()` falls back to the terminal result's `final_output` for
  stream-driven runs; resilient `status.json` reads under a concurrent atomic replace.

## [0.12.0] - 2026-06-27

### Added
- `AgentLoop.from_tools(spec, adapter, tools)` — one call to run with custom
  `@tool`/`ToolSpec` objects (auto-wraps a provider and generates their bindings),
  plus a runnable `examples/custom_tool_quickstart.py`.
- `AgentLoop.validate(config)` / `collect_runtime_config_issues()` — pre-run config
  validation that collects **all** problems as readable messages instead of raising
  on the first.
- Curated `contracts.core` namespace (the ~9 must-know names), a
  `monoid_agent_kernel.tool_ids` constants module, and `list_builtin_tools()`.
- `ToolBinding.for_tool("fs.read")` one-token bindings and bare-string `ref`.
- `monoid studio doctor` preflight (port / writability / API key / browser /
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
- `fs.read`: on a binary/non-utf8 file, returns an actionable error pointing at
  `fs.read_media` (which reads images/PDFs under its own scope and authorization) instead
  of a bare "binary file" reject.
- Subagent/skill loaders warn on a duplicate id instead of silently dropping it.
- `[otel]` extra was api-only (a no-op); the Studio OTel toggle now has a working
  install path via `[otel-export]`.

### Changed
- Vendored KaTeX locally (woff2 only) so Studio honors its no-network promise.

## [0.11.0]
- Baseline at first public preparation. See the git history for the full
  evolution of the contracts, session/control protocol, capability leases,
  inbox/outbox fabric, durable checkpoints, and the Studio reference app.
