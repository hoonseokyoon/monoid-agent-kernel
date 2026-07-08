# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and this project is
pre-1.0 (`0.x`): minor versions may include breaking changes, which are called
out in commit messages and here.

## [Unreleased]

## [0.17.0] - 2026-07-08

### Added
- Optional provider-backed Memory tools via `monoid_agent_kernel.memory`, including
  `MemoryProvider`, `LocalFilesystemMemoryProvider`, filesystem-style memory operations,
  provider-owned storage, and `memory.search`.
- Default tool binding bundles for read, write, shell, and artifact capabilities, plus
  stronger builtin filesystem, shell/job, and artifact tools.
- Studio durable chat projection in `studio.chat.jsonl`, with `/api/chat-transcript`
  restoring browser-facing user, assistant, and error messages across reloads and restarts.

### Changed
- Studio exposes Memory as an available capability, disabled by default and stored under
  `run_root/studio-memory/<workspace-key>/` when enabled.
- Destructive workspace helpers `fs.copy`, `fs.move`, and `fs.delete` require approval by
  default in the generated write tool bundle.
- Studio chat replay now reads durable chat messages before replaying trace events, while
  `events.jsonl` remains the trace stream and `transcript.jsonl` remains the private
  model-call log.

### Fixed
- Reopened Studio chats preserve the initial user messages and later conversation turns
  created after this release's durable chat projection.

## [0.16.1] - 2026-07-05

### Fixed
- Updated the README quickstart so the snippet works with the current `AgentRunSpec`
  API by supplying `Path`-based `workspace_root` and `run_root` values.
- Switched README Studio screenshots to GitHub raw image URLs so the PyPI long
  description can render them.
- Ignored local Studio/log artifacts to keep source distributions clean when built
  from a working checkout.

## [0.16.0] - 2026-07-05

### Changed
- Phase 4-1 public-surface cleanup: `monoid_agent_kernel.contracts` and the
  top-level `monoid_agent_kernel` package now export only the contract surface.
  Helper/default implementations and convenience adapters are imported from their
  explicit modules.
- Phase 4-2 lifecycle vocabulary cleanup: run lifecycle payloads now use
  `state` plus `terminal` instead of legacy lifecycle `status`. Terminal
  `AgentRunResult.status`, `ControlResult.status`, proposal status, tool status,
  job status, and metrics status keep their domain meanings.
- Phase 4-3 test/CI readiness: backend tests now have a managed factory seam for
  spawned future cleanup, Studio shutdown joins owned server threads, and CI runs
  xdist plus coverage as advisory checks.
- README screenshots now show the v0.16 Studio profile workflow, including a
  data-analysis run and the exact model request preview in the profile editor.
- `AudioPart` and `VideoPart` are now exported from the contract surface to match
  the core content contract.

## [0.15.0] - 2026-07-03

### Added
- Operational rule coverage for OR-01 through OR-13, mapping each rule to Core Helper Kit
  surfaces, conformance assertions, Reference harness cases, and primary tests.
- Executable conformance profiles for tool-agent approval, optional side-effect tools,
  external-agent message fabric, and the bundled Reference full profile.
- Strict wire parsing helpers for JSON-native payloads, plus property tests for
  external-agent envelopes and inbox/outbox round-trips.
- Public/private task payload separation, including safe public capability-result summaries.
- Canonical external-agent metadata merge helpers so user metadata cannot override trusted
  peer, task, request, result, or trace identity.

### Changed
- Reference backend, web tool service, durable metadata listing, and Studio subagent event
  routing now consistently use the Core Helper Kit paths established by the operational rules.
- Approval callback parsing now fails closed for ambiguous approve/deny values while preserving
  durable replay behavior.
- Strict parsers continue to accept legacy `native-agent-runner.*` protocol ids during the
  namespace migration window.

### Fixed
- Recovered outbox requests, capability leases, and control commands created before the Monoid
  namespace rename are accepted by the new strict parsers.
- Public hosted-task payloads no longer expose raw capability grant material such as `lease` or
  `token_ref`.
- Requested web domain scope now respects wildcard narrowing rules instead of exact-match-only
  intersection.

## [0.14.0] - 2026-06-30

### Added
- Compatibility imports through `native_agent_runner` and the legacy `native-agent`
  CLI alias, so existing local integrations can migrate incrementally.
- Central identifier and environment helpers for the Monoid namespace migration.

### Changed
- Project, package, repository, docs, and examples now use **Monoid Agent Kernel**
  branding.
- Python distribution name is now `monoid-agent-kernel`; import new code from
  `monoid_agent_kernel`.
- Current wire and durable artifact identifiers now emit `monoid.*` values.
  Readers and validators continue to accept legacy `native-agent-runner.*` values.
- Environment variables now prefer `MONOID_*` names. Existing `NAR_*` names are
  accepted during migration.
- Token issuer, audience, and header values now use Monoid identifiers while
  accepting legacy values during migration.

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
