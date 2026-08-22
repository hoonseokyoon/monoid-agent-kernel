# Integration Contracts

This document defines the supported integration surface for Monoid Agent Kernel:
a lightweight agent kernel designed to embed into many products, runtimes, and
deployment models. Import Python contracts from `monoid_agent_kernel.contracts`.
Treat `monoid_agent_kernel.reference.*` as runnable examples for backend, LLM
gateway, and web gateway integration.

## Boundary

The package is organized around four public roles:

- **Contract** — the stable integration surface, collected in
  `monoid_agent_kernel.contracts` (and re-exported from the top-level
  `monoid_agent_kernel`). These are the specs and protocols you depend on and
  implement. This document defines the Python, HTTP, wiring, and operational rules.
- **Conformance Test** — profile-based tests that check contract behavior for a chosen
  runtime shape. See `docs/CONFORMANCE.md`.
- **Core Helper Kit** — the supported runtime and helper modules that make the contract
  easy to satisfy: `loop.py`, `core/`, `providers/`, `tools`, `workspace`,
  permission, shell execution, and gateway client modules. See `docs/CORE_HELPER_KIT.md`.
- **Reference** — example services under `monoid_agent_kernel.reference`
  (`backend`, `llm_gateway`, `web_gateway`, `mcp_gateway`, `stores`, `studio`,
  `conformance`). These examples are assembled from the public contract and helper kit; core has no
  dependency on `monoid_agent_kernel.reference`. See `docs/REFERENCE.md`.

Agent configuration enters the engine through `AgentRuntimeConfig`. Legacy
tool/shell/web policy inputs have left the core, backend, and CLI execution
paths.

### Stability

Pre-1.0 (`0.x`); breaking changes are noted in commit messages.

- **Stable Contract**: `AgentLoop`, `AgentRunSpec`, `AgentRuntimeConfig` /
  `RuntimeConfigProvider`, `ModelAdapter`, `ToolSpec` / `tool`, `EventSink`,
  `CheckpointStore`, `Workspace` / `workspace_factory`, and `PermissionPolicy`.
- **Contract Extension**: async-task seams (`TaskExecutor`, `ResultInjector`,
  `TaskReporter`); the session lifecycle + control surface (`AgentSession` /
  `LoopSession`, `SessionState`, `ControlCommand` / `ControlResult` /
  `ControlDispatcher`); capability leases; agent-as-tool delegation; Agent Skills;
  output validation and the standalone validated call (`ValidatedCallRunner` /
  `ValidatedCallResult`); generation-parameter, reasoning-parameter and output-schema
  delivery with the gateway applied echoes; model-stream observation; and multimodal input. `ImagePart` and
  `DocumentPart` are forwarded
  to multimodal-capable adapters. `AudioPart` / `VideoPart` are exported content
  contracts and round-trip through core JSON/checkpoint paths; provider forwarding is
  adapter-specific.
- **Helper Kit**: implementation helpers live under explicit modules such as
  `monoid_agent_kernel.core.*`, `monoid_agent_kernel.providers.*`,
  `monoid_agent_kernel.tools.*`, `monoid_agent_kernel.recorder`, and
  `monoid_agent_kernel.observability`.
- **Reference examples**: `monoid_agent_kernel.reference.*` services.

## Operational Rules

Operational rule ids name the semantics that keep agent systems
durable, observable, and safe across backend and gateway implementations. These rules are
contract language: a backend may use the Core Helper Kit or its own implementation path, then
prove the same behavior through conformance profiles.

Phase 2S keeps this rule list fixed. It tightens existing rule coverage through strict wire
parsers, public payload sanitizers, canonical metadata merge, helper adoption in Reference
boundaries, and property tests for pure helper/parser surfaces.

| Rule ID | Contract rule | Primary profiles | Helper surfaces |
| --- | --- | --- | --- |
| `OR-01-SCOPE-RELATION` | Scope relation is defined once: signed scope bounds request scope, request scope bounds grant scope, numeric caps narrow by smaller values, list caps narrow by subset, and wildcard domains narrow by pattern relation. | `capability-security`, `provider-gateway` | `core.scope.scope_within`, `domain_patterns_within`, `effective_signed_scope` |
| `OR-02-CAPABILITY-BOUNDARY` | Capability identity and binding boundaries are preserved through gateway calls, including endpoint capability matching and domain filters for provider and redirect checks. | `provider-gateway`, `capability-security` | `core.scope.effective_signed_scope`, `core.lease_admission.validate_lease_admission`, `CapabilityVault.admit` |
| `OR-03-LEASE-ADMISSION` | Lease admission preserves policy fields and decision semantics: approved leases keep `lease_id`, `issued_at`, `expires_at`, `max_expires_at`, and `scope`; denied decisions strip grant material. | `capability-security`, `control-plane` | `core.lease_admission.validate_lease_admission`, `sanitize_denied_capability_result`, `CapabilityVault.admit` |
| `OR-04-REVOCATION-SCOPE` | Revocation covers time and child-runtime boundaries, including revoke-now watermarks, wildcard revocation, child-held leases, and shared revocation state. | `capability-security`, `multi-agent` | `core.capability_revocation`, `CapabilityVault.revoke`, `CapabilityVault.fork_for_child` |
| `OR-05-EVENT-SEQUENCING` | Run event sequence ownership follows lifecycle state: live recorders own live sequence, queued direct appends seed later recorders, terminal appends use guarded fallback, and diagnostics use the newest sequence. | `durable-runner`, `control-plane` | `core.event_sequencing.RunEventSequencer`, `read_event_page`, `diagnostic_event_summary` |
| `OR-06-CONTROL-AUDIT` | Control audit follows authorization, lifecycle, and ownership policy: valid target authorization gates run-stream audit, failed authorized commands leave failure audit, and callback-token commands are declared. | `control-plane`, `capability-security` | `core.control_audit.ControlAuditPolicy`, `core.event_sequencing.RunEventSequencer` |
| `OR-07-DURABLE-METADATA` | Durable metadata writes keep API results and recovery results aligned through schema validation, shared-store compatibility, and commit ordering. | `durable-runner`, `control-plane` | `core.durable_metadata.DurableMetadataCommitter`, `validate_run_metadata`, `runtime_config_from_metadata` |
| `OR-08-PROVIDER-CAPS` | Provider gateways apply effective caps on request and response paths, including signed caps, request caps, defaults, redirect boundaries, byte caps, and timeout caps. | `provider-gateway` | `core.scope.effective_signed_scope`, Reference web gateway cap application |
| `OR-09-SUBAGENT-BOUNDARY` | Subagent runtime links identity, capability, and trace boundaries: child runs have their own identity/accounting, isolated live lease slots, shared revocation, and parent-child diagnostics linkage. | `multi-agent`, `capability-security`, `durable-runner` | `core.subagent_runtime.SubagentRuntimeContext`, `validate_descendant_run_id`, `subagent_diagnostics_from_events`, `CapabilityVault.fork_for_child` |
| `OR-10-TOOL-SURFACE-ADMISSION` | Tool execution follows the active turn surface: unavailable tools, hidden/searchable-only tools, denied bindings, and quota-exceeded bindings do not execute handlers. | `tool-agent` | `DefaultToolSurfaceResolver`, `ToolSurfaceSnapshot`, `AgentLoop` tool admission path |
| `OR-11-GENERIC-ASK-APPROVAL` | `authorization="ask"` creates a durable approval task; approval revalidates the captured call before one execution, and denial returns an observation without invoking the handler. | `tool-agent`, `control-plane` | `core.tool_approval`, `TaskManager`, `AgentLoop` approval replay path |
| `OR-12-DURABLE-SIDE-EFFECT` | External side-effect tools declare their delivery semantics; strict runtimes admit them through durable outbox staging or explicit idempotency keys, and outbox-declared handlers stage a durable request before success. | `side-effect-tool-agent` | `core.side_effect_policy`, `core.outbox`, `ToolContext.emit_outbox`, Reference edge drain |
| `OR-13-EXTERNAL-AGENT-ENVELOPE` | External agent messages preserve peer/message identity, restart-stable dedupe, correlation, causation, trace context, ordered text/data parts, and retryable pending/error state across inbox/outbox boundaries. | `message-fabric` | `core.external_agent_envelope`, `core.inbox`, `core.outbox`, Reference inbox-routing outbox sender |

## Identifier Namespace

Current wire and artifact identifiers use the `monoid.*` namespace. The runtime emits new
schema versions, protocol ids, token issuers, and service audiences with `monoid.*` values,
including `monoid.backend` and `monoid.task-callback`.

Readers, validators, and gateway parsers accept the pre-rename `native-agent-runner.*`
identifiers during migration so existing durable run artifacts and gateway clients continue
to load where listed. The exact per-artifact reader policy, including permissive and
writer-only exceptions, is maintained in [COMPATIBILITY.md](COMPATIBILITY.md).

## Python Contracts

### AgentLoop

`AgentLoop(spec, model_adapter, *, runtime_config_provider, tool_providers=(),
context_providers=(), event_sinks=(), status_file=True,
permission_policy=PermissionPolicy(), cancellation_token=None,
shell_approval_provider=None, web_gateway_client=None, workspace_factory=None,
checkpoint_store=None, capability_broker=None, subagent_definitions={})` runs a single
agent against one workspace.

The optional seams let an integrator back the engine with their own implementations
without changing it: `workspace_factory` (file storage — see [Workspace](#workspace)),
`checkpoint_store` (durable run state — see [Durable Persistence](#durable-persistence)),
`capability_broker` (sensitive-tool gating — see
[Capability Request / Lease](#capability-request--lease)), and `subagent_definitions`
(agent-as-tool delegation — see [Subagents](#subagents-agent-as-tool)). Each defaults to
the local / in-process behavior. `tool_providers` and `context_providers` are the
extension seams that Skills and MCP ride (see [Skills](#skills-progressive-disclosure)).

`runtime_config_provider` is required, but accepts any of three forms — a
`RuntimeConfigProvider`, a bare `AgentRuntimeConfig`, or a
`callable(run_id) -> AgentRuntimeConfig | None` — which the loop coerces to a
provider. `AgentLoop.from_config(spec, model_adapter, runtime_config, **kwargs)`
wraps a fixed config and forwards the remaining optional seams in one call.
`StaticRuntimeConfigProvider` / `static_runtime_config(config)` are the explicit
fixed-config provider. The loop reads the current config at
bootstrap and at each turn boundary. A config change applies to the next turn.
The `ToolSurfaceSnapshot` and `BoundToolCatalog` used by a turn stay fixed for
that turn.

The run lifecycle is:

- `open()` — bootstrap and idle (workspace, recorder, tool registry, manifest;
  emits `run.started`). No model turn yet.
- `submit(user_input) -> AgentTurnResult` — run one user turn: deliver
  `user_input` (a `str` or content parts) and step until the model settles (no
  tool calls + final text) or a per-submit limit. The run stays open. Each
  `submit()` gets a fresh `max_steps` budget; `max_tool_calls`, token usage, and
  `max_duration_s` are session-wide. `AgentTurnResult` carries the settle status,
  final text, the accumulated (preview) proposal, and the continuation
  `turn_handle`. A park that settles nothing — a *recoverable* turn failure
  (`turn_failed`), an interrupt, or a pause — raises `TurnNotSettled`
  (`monoid_agent_kernel.errors`): the session stays alive, and the exception's
  `suspension` carries the reason plus the `retryable` / `http_status` /
  `config_recoverable` / `provider_error_code` / `provider_retried`
  classification — all five re-stamped onto the exception itself, because a
  driver on this facade holds an exception and nothing else. The last two are
  what separates an `insufficient_quota` (a human fixes the billing) from a
  `rate_limit_exceeded` (back off and re-issue) and an exhausted adapter retry
  budget from an untried call. The non-blocking pump
  (`run_until_suspended`) returns the same park as a `Suspension` with
  `turn=None` instead of raising; `astream` ends the stream with it as
  `stream.suspension`.
- `commit_checkpoint()` — opt-in: adopt the current proposed workspace state as
  the new diff baseline, so later proposals report only post-commit changes.
- `close() -> AgentRunResult` — finalize: cancel jobs, write the terminal
  proposal, emit `run.finished`, close the recorder. A cancellation acknowledged
  while the run sat at a quiescent park (no pump stepping to raise in) is
  promoted here through the mid-run vocabulary — the result carries
  `status="limited"`, `error_code="cancelled"`, a terminal park checkpoint is
  committed, and the checkpoints are kept (only a clean completion deletes them).
  A cancel does not take the answer with it: if the park had settled text (the
  run answered, then sat awaiting input), `result.final_text` keeps that answer —
  the cancel statement lives in `error`/`error_code` — and only a park with no
  text of its own (a mid-turn cancel) carries the kernel's
  "Stopped because the run was cancelled." notice. A close over a **mid-turn park**
  (`paused`/`interrupted` — a turn that never settled) must not finalize a clean
  success either: it promotes to `status="limited"`,
  `error_code="closed_unsettled"` (one code for both variants), with an empty
  failure classification (nothing here is a provider failure) and its checkpoints
  kept, so the frozen turn's only restore point survives the close. A settled
  `awaiting_input` park is not this — its turn completed and close finalizes the
  success it was; an acknowledged cancel is the operator's stronger verdict and
  wins over the unsettled promotion.
- `run_once(user_input) -> AgentRunResult` — one-shot convenience equal to
  `open()` + `submit(user_input)` + `close()`. Unlike `submit`, an ordinary recoverable
  provider/config `turn_failed` park does not raise here: the closing `finally` promotes it
  to the terminal failure record and returns that failed `AgentRunResult`.
  `evidence_uncommitted` surfaces as `TurnNotSettled` after releasing the committed park
  without terminalizing it, so another activation can complete sink-only recovery. An
  `interrupted` or `paused` park also surfaces as
  `TurnNotSettled` after the same close, and that close records the honest
  outcome underneath the raise: `run.finished` carries `status="limited"`,
  `error_code="closed_unsettled"` (the turn never settled — a user stop is not
  a success either) and the checkpoints are kept — a caller that wants to
  resume an interrupted turn uses the multi-turn facades, where the session
  stays alive. `close()` performs the same promotion
  for any driver (the explicit form is `fail_recoverable`): a run closed on an
  unrecovered recoverable failure finalizes `failed` with `failure.json`
  written and its checkpoints kept, never as a clean success. Both promotions
  survive a restart: `restore()` rehydrates the pending failure and the
  mid-turn park marker from the checkpoint's `last_suspension`
  (`reason="turn_failed"` and `reason="paused"`/`"interrupted"` respectively; a
  later settle clears both), so a recovered run left idle and then closed
  records the park it died in rather than a clean success that deletes its own
  checkpoints.

### AgentRunSpec

`AgentRunSpec` is the session descriptor. It carries no user input — the
instruction(s) flow in through `submit()` / `run_once()`:

- `workspace_root`, `run_root`, `run_id`
- `mode`: `read-only`, `propose`, or `apply`
- `workspace_backend`: `overlay` or `staging`
- `limits: RunLimits`
- `permission_policy: PermissionPolicy`
- `input`: optional multimodal content-parts surface (image/document parts are
  forwarded to multimodal-capable adapters)
- `metadata`

It does not carry model, prompt, tool, shell, or web settings. Those values live
in runtime config.

### Workspace

The engine never touches the filesystem directly — it works through a `Workspace`
(the file-storage surface), which stores and diffs the run's files. `AgentLoop` builds
one per run by calling `workspace_factory(spec)`. The default,
`default_local_workspace_factory`, returns the local-filesystem `LocalWorkspaceBackend`.
Supply your own `workspace_factory` to back a run with a different store — a git worktree,
an object store, a remote or in-memory filesystem — without changing the engine.

A `Workspace` exposes:

- path handling — `normalize`, `path_kind`, `exists`, `resolve_existing_or_parent`
- byte IO — `read_bytes` (honors a `max_bytes` cap), `write_bytes` (optimistic
  `expected_sha256` guard), `mkdir`, `copy_path`, `move_path`, `delete_path`
- listing — `list_entries`, `glob`, `text_files`
- proposal generation — `changed_entries`, `diff_patch`,
  `snapshot_current_as_new_baseline` (re-baseline for incremental apply),
  `workspace_base_payload`

It carries `root`, `mode`, `backend_kind`, and `max_bytes_read`. The value types it
returns, `FileEntry` and `ChangedEntry`, are exported from `contracts`.

`mode` (`read-only` / `propose` / `apply`) and `backend_kind` (`overlay` / `staging`)
select how the local backend stages writes; a custom backend interprets them or pins its
own. Every backend must pass the parametrized contract suite
(`tests/test_workspace_contract.py`): write/read round-trips with their sha256, the
proposed state is observable, the optimistic and byte-cap guards hold, no path escapes the
root, the changed-entry delta tracks edits, and re-baselining collapses it. Passing it
makes a backend a drop-in.

### AgentDefinition And Runtime Config

`AgentDefinition` is a reusable blueprint:

- `id`, `version`, `description`
- `model: ModelConfig | None`
- `prompt: PromptSpec`
- `tools: tuple[ToolBinding, ...]`
- `tool_search: ToolSearchConfig`
- `metadata`

`AgentRuntimeConfig` is the effective config for a run:

- `definition_id`
- `config_version`
- `model: ModelConfig | None`
- `prompt: PromptSpec`
- `tools: tuple[ToolBinding, ...]`
- `tool_search: ToolSearchConfig`
- `metadata`
- `config_hash`

`config_hash` identifies the parsed runtime configuration rather than every JSON representation
detail. It hashes every policy value, including the raw `allowed_paths` and `denied_paths` arrays,
while omitting only `tools[*].scope.path_pattern_encoding`. That additive field disambiguates a
literal leading `!` for fresh JSON readers and does not change the in-memory `ToolScope` meaning.
This normalization lets pre-v0.20 readers ignore the field and recompute the same hash during a
rolling deployment.

`ModelConfig.to_json` emits its `generation` block only when a generation value is configured, so
a generation-free config — the entire pre-generation population — serializes and hashes
identically before and after the field existed. A config that *sets* a generation value hashes
differently by design: sampling parameters change model behavior, so the
`path_pattern_encoding`-style hash exclusion would make the hash lie about what the run does.
Consequence for rolling deployments: backend-run recovery metadata written with a configured
generation block does not verify against a pre-generation reader (and vice versa) — configure
generation values only on a fleet that has fully rolled past the version that introduced them.

`ToolBinding` is the public tool unit:

```json
{
  "binding_id": "read_notes",
  "model_name": "read_notes",
  "ref": {"kind": "registry", "tool_id": "fs.read"},
  "exposure": "immediate",
  "authorization": "allow",
  "guidance": {"summary": "Read source files before editing."},
  "scope": {"allowed_paths": ["docs/**"]},
  "quota": {"max_calls_per_run": 20},
  "runtime": {},
  "title": "Read notes",
  "summary": "Read a workspace file.",
  "risk": "read",
  "metadata": {}
}
```

The same registry tool can appear multiple times with different `binding_id`,
`model_name`, guidance, scope, quota, and runtime settings. Duplicate
`binding_id` values and duplicate resolved `model_name` values fail validation.
Unknown registry tool refs fail validation with `AgentConfigError`.

`compile_bound_tool_catalog(config, registry)` produces a `BoundToolCatalog`.
The model receives only bound model-facing `ToolSpec`s. Tool execution resolves
`model_name -> BoundTool -> base ToolSpec.handler`.

### Tool Surface

`DefaultToolSurfaceResolver` consumes a `BoundToolCatalog`, turn context,
pending binding loads, previous snapshot, and call counts. It returns a
`ToolSurfaceSnapshot`:

- `immediate_tools`: model-facing bound specs available this turn
- `searchable_tools`: bound specs indexed for `tool.search`
- `search_entries`: binding-aware search metadata
- `hidden_tool_ids`: hidden or denied binding ids
- `authorizations`: `binding_id -> ToolAuthorization`
- `surface_hash`, `delta_notice`

Unbound registry tools stay outside the surface. Hidden or denied bindings stay
outside model tools and search results. Tool search uses `binding_id` for search
results and pending loads.

### Model Adapter

Choose one one-shot contract:

- `ModelAdapter.next_turn(request: ModelRequest) -> ModelTurn` for synchronous adapters. The
  loop executes it in a worker thread.
- `AsyncModelAdapter.anext_turn(request: ModelRequest) -> ModelTurn` for native async adapters.
  The loop awaits it directly. An adapter that exposes both uses `anext_turn`.

Add `StreamingModelAdapter.astream_turn(request) -> AsyncIterator[ModelStreamChunk]` to either
one-shot contract for token streaming. `AgentLoop.astream` prefers the streaming method and folds
its chunks into the same `ModelTurn`, event, error, and checkpoint path. Autonomous runs use the
stream when `stream_model_calls=True`, `emit_output_deltas=True`, private model-content persistence
is enabled, or a model-stream observer is configured.

Four further opt-in protocols declare optional capability members:

- `MultimodalModelAdapter.supports_multimodal: bool` — the loop resolves by-reference media in the
  by-value `messages` log to wire blocks before the call. A multimodal adapter may also expose
  `wire_image_encoding` (default `"base64"`); that attribute is not a protocol member because it
  parameterizes the capability rather than declaring it.
- `ProviderNamedModelAdapter.provider_name: str | None` — tags captured `ModelTurn.reasoning` with
  provider+model so those items only round-trip back to a matching adapter and model.
  Omitting it means "do not tag", and `None` says the same thing explicitly — which is what a
  deployment needs when its gateway fronts an upstream with no reasoning artifacts, hence the
  optional type rather than `str`. A *forwarding* adapter declares the provider whose artifacts it
  relays, not itself: `GatewayModelAdapter.provider_name` names the gateway's **upstream** and
  defaults to `"openai"`, matching the reference gateway's own default upstream. A deployment
  whose `provider_adapter_factory` routes elsewhere must set it to that upstream, through
  `monoid run --llm-gateway-provider`, `monoid backend serve --llm-gateway-provider`,
  `RunnerBackend(llm_gateway_provider=...)`, or — for the reference Studio, whose embedder seam
  is exactly such a factory — `StudioConfig(llm_gateway_provider=...)`; those spell `None` as
  `none`, and leaving them unset means "derive". The same attribute
  names the provider on every observability surface that probes an adapter for one — via
  `resolved_provider_name(adapter, config)`, which is `provider_name` else `ModelConfig.provider`
  (including when the declaration is unreadable — the tolerance path still falls back rather
  than answering nothing). A caller that has already read the declaration passes it as `declared`
  so it is not read a second time: one model call resolves it once, and the receipt's
  `provider_name` and the provider term in its `request_digest` therefore cannot disagree even if
  the adapter's property does not answer the same way twice.
  and feeds `ModelCallReceipt.provider_name`, the model-stream context's `provider`, and
  `run.started`'s `model_provider` (and so every OTel `gen_ai.provider.name`). A call routed
  through the gateway is therefore attributed to the model that served it, with the transport
  still legible beside it as `ModelConfig.provider` and on the run manifest.
  A forwarding adapter's declaration is a *guess about someone else's deployment*, so it is
  **verified, never adopted**: the gateway names the upstream it actually relayed (`provider` on
  the success body and terminal frame, below), and when that disagrees with what the client
  declared, the client drops that turn's relayed `reasoning` artifacts and keeps everything else —
  including the declaration itself, on every surface above. Adopting the server's answer per turn
  would make one call's provider question have two answers again, which is what
  `resolved_provider_name` exists to prevent; dropping is what the replay filter already does with
  a tag that does not match, decided one hop earlier. A gateway that names no upstream (an older
  one, or one whose upstream declares nothing) changes nothing.
- `ConfiguredModelAdapter.config: ModelConfig` — the adapter's own fallback, used when
  `ModelRequest.model` is absent. A `ModelCallReceipt` reads it so it records the model the call
  actually ran under rather than a default the call never used.
- `AddressedModelAdapter.resolve_destination(config) -> str` — where a call under `config` would
  actually be sent. Two adapters holding identical configs can address different hosts, so the
  receipt records that fact *beside* the replay key rather than inside it: a
  `destination_status` naming which outcome the probe had (`not_declared` / `declined` /
  `resolved` / `unavailable`, or `not_reached` when the call was refused first) and a keyed
  `destination_digest` that lets two calls be compared. The value itself is never recorded, so an
  internal hostname stays internal — which is precisely why it cannot be key material: a key taken
  over a preimage no record may hold cannot be recomputed or diagnosed. Raising is permitted and is
  reported as `unavailable`, distinct from an adapter that simply does not route by host.

Implementing them is never required. The loop probes each attribute with `getattr` and a neutral
default, and behaves identically whether an adapter declares it or omits it, so the attributes are
deliberately **not** members of `ModelAdapter` / `AsyncModelAdapter`: a protocol member is required
for structural typing even when the protocol body assigns it a default, which would reject an
adapter that implements only `next_turn`. Each member that is a *value* is declared as a read-only
property so a `ClassVar`, an instance attribute, and a property all satisfy it;
`resolve_destination` is a method because it answers for a given `ModelConfig`.

Three further capability declarations are plain attributes rather than Protocol members:
`structured_output_support` (the adapter translates `ModelRequest.output_schema` into
provider-enforced constrained decoding), `generation_support` (the adapter puts
`ModelConfig.generation` on its provider request) and `reasoning_support` (the adapter puts
`ModelConfig.reasoning` there). The exact string `"native"` declares the capability; all three
probes (exported from `contracts` as `structured_output_support(adapter, config=None)` /
`generation_support(adapter, config=None)` / `reasoning_support(adapter, config=None)`) are
fail-closed through one shared
implementation — absence, any other value, and a declaration that raises (on read **or on
call**) all read as `"none"` — so a consumer can never over-trust an adapter that did not
explicitly claim it. These are what a transport in front of an adapter is allowed to base an
applied-parameters proof on: a gateway can see what it *forwarded*, never what the adapter
behind it did with it.

The probe reads the **instance**: the declaration is a `ClassVar` when the answer is fixed, and
a **callable taking the effective per-call `ModelConfig`** when it depends on policy — the
probe passes its `config` argument through to a callable declaration, and `None` probes the
adapter's standing configuration. (Before v0.21.0 the conditional convention was a *property*;
a property still probes fine but cannot receive the per-call config, so a property-declared
conditional adapter silently answers from its standing configuration — migrate conditional
declarations to the callable form.) The claim and the enforcement must read the same config:
enforcement runs under `request.model or self.config`, so a claim probed off the standing
config alone would let a shared adapter mint proof for a call it enforces under a
wire-supplied `"omit"`. `OpenAIModelAdapter` applies the parameters itself and declares all
three unconditionally. `GatewayModelAdapter` only *forwards*, so its claim is worth exactly the
proof it insists on: it answers `"native"` when the effective config says
`on_unsupported="fail"`, where a returned turn is a proven turn, and `"none"` under `"omit"`,
where it deliberately accepts an unproven one — reading the governing knob of the claim's own
feature family (`generation.on_unsupported` for the generation/schema pair,
`reasoning.on_unsupported` for `reasoning_support`), because a claim answered off another
family's policy would mint proof for a call whose own policy said best-effort. Without that, a
chained gateway would mint a fresh positive echo out of a static declaration and report proof
for a call whose inner hop had none.

An adapter with its own retry loop should call `report_provider_retried()` when it decides to make
another attempt. The kernel counts one adapter call per turn however many attempts happen inside it,
so without this a call that failed twice and succeeded on the third try is recorded as a clean
single attempt. Report on the *decision*, before waiting or reconnecting: a call the run cancels or
times out mid-retry never returns an outcome to carry the fact, and for a blocking `next_turn` the
worker's eventual result is discarded entirely. Calling it is optional and inert outside a run. Under `ModelRetryConfig.layer="kernel"`
this channel keeps exactly that meaning: the kernel's own re-dispatches are counted by
`attempts`, never reported here.

`ModelCallReceipt.attempts` may be **0**. A run whose cancellation or deadline was already past when
the call was requested is refused before the adapter is reached, and a receipt is still written
because a refused call belongs in the audit trail — so a consumer summing `attempts` must treat 0 as
"no adapter call was made" rather than as a missing value. A failure *while* reaching into the
adapter still counts as 1: the kernel did begin the call there. Under the kernel retry
layer each re-dispatch adds one, so N means N adapter calls for one logical request. A payload that omits the field reads
as 1, which is what older records mean.

`ModelCallReceipt.attempt_log` itemizes what `attempts` counts: per dispatch, the index, the
elapsed time, the failure taxonomy `with_error` reads, that attempt's own billed usage, whether
the adapter's loop reported *during that dispatch* (the receipt's `provider_retried` stays the
whole call's fold), and whether a delivered chunk had closed the retry window when the attempt
settled — possible only on the final entry. The log is empty or complete, and the record enforces
both halves rather than trusting its writer: either there are no entries (a refused call, or a
record written before the field existed) or their indices are exactly `1..attempts` in order, and
their usage sums to the receipt's `usage`. A log that names one dispatch twice has the right
length and still cannot answer what the second one did; a log whose rows do not add up to the bill
they itemize leaves a reader with two numbers and no way to tell which to believe. Both are
refused at construction, which is also why the runner builds the log and the merged total in a
single `replace` — and refused again by `monoid validate`, which reads the ledger as JSON and
constructs no receipt, so a JSON Schema that can only judge one entry at a time would have
reported such a line clean. On the wire an unitemized call has one spelling produced and two
read (W7-4): the writers omit an empty log, so absence covers a record that predates the field,
a call with zero dispatches, and a receipt built without one alike — and a present `[]`, which
every build between W7-1 and W7-4 wrote for all three because the projection emitted the key
unconditionally, reads as the same value and passes `monoid validate` beside any `attempts`.
Neither reader refuses that pair. The runner never writes it — it fills one entry per dispatch
on every terminal path — but the runner is not the only writer: `record_settled_call` is public
and a receipt handed to it carries whatever log it was built with, the field's own default being
none, so refusing the pair would convict lines the previous build wrote through its own API.
`idempotency_key` converges the other way, and the asymmetry is each field's own rule: its
absence spelling is the in-band empty string, so the key travels on every line, while the log's
is a missing key, so an empty one is simply not written. The log is optional; an *entry*'s original keys are not: they have
no writer predating them, so every key the entry shipped with is required on the wire and a
partial one is refused rather than completed from defaults — defaults there turn a corrupt line
into a plausible dispatch. A key added after the entry shipped follows the record-level absence
rule instead, named per key: `backoff_ms` (W7-2) is the measured wait the kernel imposed *between*
that dispatch and the one before it — so the first entry has nothing to report, and any other
value there is refused by the record and by `monoid validate` alike, because a line claiming the
kernel waited before its own first reach into the adapter is one no runner
writes — absent on lines a W7-1 writer filled, where absence means
exactly what the whole log's absence means one level up, and refused when null, by reader and
schema alike, because no writer omits by writing null. Entries carry no wall-clock instant — the
receipt's own rule; the ledger line's `recorded_at` anchors the call. Every entry duration is
the floor of the same monotonic clock, and floors sum to at most the floor of the sum, so
`sum(elapsed_ms) + sum(backoff_ms) <= latency_ms` holds exactly; the remainder is the keying and
settle overhead that falls outside the dispatch loop. `monoid validate` enforces that inequality
on the ledger line, where breaking it means a record no runner wrote. The constructor cannot: the
runner attaches the log while `latency_ms` is still the field's default and `_publish` stamps the
measured duration afterwards, so the line is the first place both values are settled together —
a receipt check would fire on every retried call, weighing real durations against a latency of
zero. A consumer that lays the entries on a timeline bounds its own arithmetic by the same
inequality rather than assuming it: the OTel preset's per-attempt children never begin before the
call they belong to, whatever a hand-built or corrupted receipt claims.

`ModelCallReceipt.idempotency_key` is the retry-scope token the call presented (W7-3). The runner
mints one per call (`idem_` + hex) in the same block that computes the digests — before the first
dispatch — and it is constant across kernel re-dispatches and adapter-internal retries, because
both loops reuse the request the token rides on. The runner is the single issuer: a caller-set
`ModelRequest.idempotency_key` is overwritten, since a respected caller value would let one
request object hand two calls the same retry scope. The token is deliberately outside the replay
key — two byte-identical requests share a replay slot precisely because content cannot separate
them, so the token that separates their provider work is random, never derived — and the recorded
value means *issued*, not *sent*. Three limits are the contract, not gaps: carriage is
retry-scoped and **not exactly-once** — the reference gateway logs and echoes the header
(`Idempotency-Key`, on every response including errors) but does not dedupe on it, and does not
relay it upstream: instead it *issues its own* for the upstream hop it drives, because that hop
has a retry loop of its own and relaying would stitch two retry scopes into one and misdescribe
both. `ModelCallRunner` is not the only issuer for that reason, and both read the same
`new_idempotency_key`; a
**resumed run reissues** — a call never spans a park, so recovery re-runs the step and the rerun
is a new call with a new key; and only the **gateway transport presents it** — the OpenAI adapter
does not read the field, so nothing is sent there. A receipt whose key is empty was never keyed:
the call was refused before the keying block, or the record predates the field.

A key must be a bounded ASCII token — 1–128 characters from `[A-Za-z0-9._+-]`, starting with a
letter or digit — and that rule is enforced at every edge the value crosses, not only where it is
minted. The `model-calls.v1` schema states the same rule as a pattern (empty admitted, the
`^(|...)` idiom `prompt_digest` uses), because `monoid validate` certifies imported and
third-party run directories and a line whose key the rest of the kernel would refuse must not be
certified. Both enforcers derive from one body in `core/model_io.py`: `core` cannot import
`providers`, so a rule owned on the provider side could only have been copied, and a retyped twin
regex drifts. It is the one field on this record that reaches a **transport header** rather than a JSON
string: JSON escapes a control character and an HTTP header does not, and neither `http.client`
nor `httpx` refuses an obsolete folded value (`"a\r\n b"`). So request ingress *refuses* a
non-conforming caller-supplied key, the gateway transport *omits* one (an adapter must not lose a
paid call over a bookkeeping token), and the reference gateway treats a non-conforming inbound key
as absent — logging that one was dropped, never its bytes, because that route logs before the
service authenticates. Absence on this field is spelled by the **empty string and nothing else**:
a caller who supplies `None`, `False` or `0` supplied a value, and ingress refuses it rather than
reading it as "no key" and letting the transport drop it silently.

The key shares one more rule with the two digests beside it (W7-4). These are the three
format-constrained fields whose values `ModelCallReceipt.from_json` deliberately does **not**
judge: the reader transports what it was given, so a receipt with a damaged digest can still be
loaded, inspected and repaired — and a parsed receipt is therefore not a certified one.
Certification has two enforcers, each deriving from one body in `core/model_io.py`: the schema
patterns `monoid validate` runs, and `model_call_record`, which refuses to *mint* a ledger line
the sweep would then convict — checking all three under the same empty-or-valid rule the schema
states, so it can never fire on a runner-built receipt (a refused call's line is empty and
explained by its status fields). The class is exactly these three, and the census derives it
from the schema rather than naming it: a fourth patterned receipt field joins the rule or fails
the suite, and a reader that quietly starts judging one of the three fails it too.

**Every `pattern` in every artifact schema ends at end of *input*, not at `$`.** JSON Schema calls
`pattern` an ECMA-262 expression; `jsonschema` runs it through Python's `re`, where `$` also
matches immediately before a single trailing newline. Under that engine a bare `^…$` certified
`"<digest>\n"`, `"<timestamp>\n"` and `"<key>\n"` — values every other edge in the kernel refuses —
so `monoid validate` would have passed a line no writer here could produce. The schemas assert
`END_OF_INPUT` (`core/_json_schema.py`) instead: `$` followed by "and no character may follow",
which is load-bearing under `re` and redundant under ECMA-262, so the same value is refused by a
Python validator and a JavaScript one alike. `\Z` would have been the Python spelling and is an
identity escape in ECMA-262 — a published schema ending in `\Z` demands a literal `Z`.

#### Generation parameters, reasoning, output schema, and the applied echoes

`ModelConfig.generation: GenerationConfig` carries per-call sampling controls — `temperature`
(0–2), `top_p` ((0, 1]), `max_output_tokens` (≥ 1) — where `None` means "delegate to the
provider default". Both ingresses (the JSON codec and direct-Python normalization) share one
fail-closed rule source, so a range accepted from JSON can never diverge from the range accepted
from a constructor. One projection (`build_generation_payload`, emitting only the set keys)
produces the OpenAI request-body parameters, the gateway wire block, and the gateway's
`generation_applied` echo, so all three agree on what "applied" means by construction.

`ModelConfig.reasoning` has the same one-projection property (`build_reasoning_payload` →
the OpenAI `reasoning` request block and the gateway's `reasoning_applied` echo), with one
deliberate difference in the gate: the DEFAULT reasoning config projects a non-empty provider
block (`{"effort": "medium"}`), so "did the caller configure reasoning" is answered by
`ReasoningConfig.is_default` rather than by the payload — and because `effort="default"`
projects an empty block that is still a configured value, the reasoning echo may legitimately
be `{}`, which proves exactly "an empty block was forwarded". That empty-object proof is the
value-drift catcher: a hop that quietly rebuilt `"medium"` out of an omitted effort echoes
`{"effort": "medium"}` against an expected `{}` and fails the match.

`GenerationConfig.on_unsupported` (`"fail"` default / `"omit"`) governs what happens when a
transport cannot prove the parameters were applied. It is enforced where proof exists: the
gateway client compares the `generation_applied` echo (and the `schema_applied` boolean for
`output_schema`, and the `reasoning_applied` block for a configured `ModelConfig.reasoning` —
that one governed by `ReasoningConfig.on_unsupported`, its own family's knob) against what it
sent, on both the sync response and the terminal stream frame
— and when a stream ends cleanly **without** a terminal frame, the drain runs the same checks
with an absent echo, so a frameless stream (the older-gateway shape
`assemble_streamed_turn` otherwise tolerates by synthesizing `stop_reason="stop"`) is refused
under `"fail"` exactly like a frameless sync response; plain traffic that configures no
knob keeps the pre-W5 tolerance. The refusal is non-retryable `gateway_generation_not_applied`
/ `gateway_schema_not_applied` / `gateway_reasoning_not_applied`, flagged
`config_recoverable`: resending the same call cannot
help, but the remedy is configuration (`"omit"`, or a proving transport), so `AgentLoop`
classifies it like a 4xx — the turn fails, the session survives — and the reference gateway's
HTTP layer maps it to 422 rather than 502 so the same classification survives a chained hop.
A refused turn was still generated and billed, so the refusal carries the usage the provider
reported: it reaches the failed `ModelCallReceipt` and the run's cumulative token totals on
both transports. A budget that skipped refused calls would not be a bound. The cost also
survives a hop — the gateway error envelope carries `usage` (present only when the failed call
actually spent tokens, so an error raised before reaching a provider keeps its previous wire
shape), and a gateway meters a billed failure against the tenant rather than losing it to the
raise. The classification survives it too: the envelope carries `config_recoverable` as its own
key, written unconditionally, so a client one hop out reads the remedy as a statement instead of
inferring it from the 422 — which matters most for the refusals that carry no HTTP status of
their own.
One streaming caveat is inherent to enforcing at the terminal frame: every delta has already
been delivered to the consumer when the refusal raises, so a streaming consumer of a `"fail"`
call sees the unproven text before the error arrives; the sync transport delivers nothing on
refusal. A malformed echo (a non-object `generation_applied` or `reasoning_applied`, a
non-boolean `schema_applied`)
is a wire-shape error, not a policy question: it answers `gateway_bad_response` on both
transports regardless of `on_unsupported` — with one precedence caveat when a single envelope is
both unproven and malformed, since the sync reader validates each echo inside its own checker and
runs the three in sequence (so an earlier key's *policy* refusal, e.g. `gateway_generation_not_applied`,
preempts a later key's *shape* refusal) while the stream validates all three shapes at frame parse
before any policy branch.

The echo comparison is not Python equality. A requested **number is proven only by a number**:
`True == 1` and `False == 0.0` in Python, so a plain dict comparison let a server answering
JSON booleans prove the most ordinary settings this block carries (`max_output_tokens=1`,
`top_p=1`, `temperature=0`). Numbers still compare across JSON's single numeric type — a
non-Python gateway re-serializes `1.0` as `1`, and refusing that would be a false refusal —
so `1` proves `1.0`, while `true` proves nothing.

A server may only emit these proofs from what its **upstream adapter declares** (the
`generation_support` / `structured_output_support` / `reasoning_support` probes), never from
what the request asked
for: copying the requested block back would match exactly on the client and let `"fail"` accept
parameters an adapter silently ignored. The reference gateway therefore omits
`generation_applied` and `reasoning_applied` and echoes `schema_applied: false` when its
upstream does not declare the
capability, and the client refuses the turn. A direct provider call has no echo, so `"fail"`
and `"omit"` behave identically there and an unsupported parameter surfaces as the provider's
own error. The knob granularity is one per **feature family**, not one per proof and not one
for everything: `generation.on_unsupported` deliberately governs both the generation and
schema echoes — "how to treat a parameter the transport cannot prove was applied" is one
question within that family, and two half-settable knobs would be a new asymmetry surface —
while reasoning is a separate family whose `reasoning.on_unsupported` (a field the request
wire already carries separately) governs the reasoning echo, its checker, and a forwarding
adapter's `reasoning_support` claim alike.

`ModelRequest.output_schema` is a standard, provider-neutral JSON Schema for the final answer.
An adapter declaring `structured_output_support = "native"` translates it into its provider's
constrained-decoding dialect **verbatim — never adjusted** toward a provider subset — so the
request digest identifies exactly what the provider was asked to enforce; a schema the provider
rejects is the provider's own error. Concretely for the shipped adapter: `OpenAIModelAdapter`
delivers the schema in a strict-mode envelope (`text.format` with `strict: true` — anything
less is not *enforced* decoding, and `schema_applied: true` would be a false proof), and
OpenAI's strict mode has subset requirements of its own — every object needs
`additionalProperties: false`, every listed property must be `required`, and some keywords are
unsupported. A schema outside that subset is rejected by the provider with an HTTP 400 whose
classified error names the offending `param` (e.g. `text.format.schema`); the kernel never
rewrites the schema to fit. "Never rewrites" includes ingress: unlike model content, a schema
keeps its non-finite floats through `normalize_model_request` (substituting `NaN` → `null`
would silently change `{"enum": [NaN]}` into a different constraint), so the value reaches the
strict serializer both adapters run over the assembled request body and is refused there as a
non-retryable, `config_recoverable` bad request (`gateway_bad_request` /
`unserializable_request`) — the same answer either adapter gives any value that cannot be sent.
**The same rule governs a tool's `input_schema`** (`normalize_tool_spec`, and the reference
gateway's server-side ingress over the `tools` it forwards): that schema is both the constraint
the registry validates calls against and the definition the provider is sent, so it is delivered
as its author wrote it or refused — never quietly changed. Strings and containers are still
normalized on both schemas; only the substitution is dropped. A **record** of a schema is a
different boundary and keeps the substitution, because portable JSON cannot carry the value at
all: the run manifest, the transcript's tool-surface snapshot, and the event log store `null`
there, so a schema no provider will accept fails the calls it rides on rather than the run's
durability.
Adapters without the declaration ignore the field, and post-hoc
output validation remains the guarantee on every adapter: native delivery only reduces
repairs. `AgentLoop` never sets the field; it belongs to standalone `ModelCallRunner` /
`ValidatedCallRunner` callers.

Two same-named fields are different controls:

| field | what it bounds | enforced by |
|---|---|---|
| `RunLimits.max_output_tokens` | a run's cumulative output-token budget, checked against API-reported usage after each turn | the kernel (settles `limited`) |
| `GenerationConfig.max_output_tokens` | one call's maximum generated length | the provider, per request |

#### Request digests and identity stability

`ModelCallReceipt.prompt_digest` covers the assembled prompt and stays stable when tool
definitions or generation settings change around it; `request_digest` covers the whole request
and is the exact replay key. Both are computed on the **raw** request, before any redaction or
capture policy, so consumers on different policies agree on the identity of what the provider
was sent. An empty digest means *no key was issued* (the payload could not be canonically
encoded, or exceeded the size cap) and must never be read as a key. `digest_status` says which
of five things happened, because the empty string used to be the answer to all of them:
`absent` (canonical JSON could not carry the payload — a defect in the payload), `too_large`
(the payload exceeded `MAX_MODEL_PAYLOAD_BYTES`). The cap is set to the same number as the
default `max_message_log_bytes`, but it bounds one call's whole identity payload — system prompt,
tool definitions, instruction and observations included — while the run limit sums only the
message log and the resolved-wire guard runs only for a multimodal adapter, so a request can pass
every run limit and still be refused a key. What the shared number buys is that such a call is
*named* rather than reported as `absent`, not that the case is gone; the cap is a build-time
constant, and a `too_large` call contributes no request record to the corpus at all), `withheld`
(a key was issued and a `none`-mode capture policy removed it), `not_reached` (the call was
refused before a key was computed), and `ok`.
`digest_generation` records the domain the key was taken in, so a consumer holding a key can tell
which rules produced it.

A receipt written before those fields existed carries neither, and `from_json` reads a *missing key*
as `not_reached` only where nothing contradicts it: a payload whose `request_digest` is non-empty reads
`ok`, and one whose `destination_digest` is non-empty reads `resolved`, because that is the only
probe outcome producing a value. The alternative is a record that denies its own contents, which the
next `to_json` makes permanent. `digest_generation` is *not* inferred alongside it — a legacy key was
taken under rules the record cannot name, and empty is what stops a replay consumer treating it as
reproducible. A status the payload actually states is kept verbatim even where it disagrees with its
digest: that pair is a bug in the writer, not something to repair silently on read. Silence means the
key is absent; a key present and holding `null` is refused, like every other string on the receipt.
Both enums are closed at construction as well as on the wire, through the same check, so a receipt
cannot be built that its own reader would reject.

**What the replay key is made of is a declared list, not a serialized object.** The model config
enters as a hand-listed projection — the model name, the reasoning block, and the generation block
when configured — rather than as `ModelConfig.to_json()`. `timeout_s`, `retry` and `gateway_url`
are absent: none of them reaches a provider (the gateway wire emits only model/reasoning/generation
and each hop owns its own transport policy), so an operational change to how a call is carried does
not invalidate keys describing what was asked. The `provider` term is the provider that actually
served the call — the adapter's declaration, else the config's — so a gateway relaying an upstream
and a direct call to the same upstream share a key, while two adapters that declare nothing are
still told apart.

Two rules keep digests stable across kernel versions:

1. **Additive request fields are omitted when unset.** `generation` and `output_schema` appear
   in the digest payload only when configured, so a request that does not use them keeps the
   digest it had before the field existed. Setting one changes the digest — deliberately, since
   the request's meaning changed. The projection restates this rule rather than inheriting it
   from the serializer, so a field added to `ModelConfig` for an unrelated reason cannot rekey a
   corpus by accident.
2. **Canonicalization changes are generation changes.** Each digest is taken in its own named
   domain, carried as the single wrapper key of the payload that is hashed:
   `monoid.model-prompt-digest.v1` and `monoid.model-request-digest.v1`. Changing what the
   payload is made of — not merely adding an omitted-when-unset field, which is *not* a
   generation change — bumps that tag, so two incompatible rule sets can never collide in one
   key space and a corpus a change has invalidated is disowned in one edit rather than silently.
   The two digests are separated by their domains rather than by their field lists happening to
   differ, which is why neither can be read as the other.

#### AgentLoop model-I/O subscriptions

`AgentLoop.model_io_subscriptions` is the opt-in bridge from agent turns to
`ModelIOObserver`. Every adapter shape reaches the same `ModelCallRunner`, so observers receive one
settled capture for success, provider failure, cancellation, deadline, or streamed interruption.
Observer failure remains contained and never changes the run outcome.

`AgentLoop.invocation_context` is a base context supplied by the caller. For each call, the loop:

- sets `run_id` to `AgentRunSpec.run_id`;
- appends the durable `turn_NNNN` id to a non-empty caller `step_id`, or uses it directly;
- preserves the caller's `attempt`, Skill/batch/case fields, trace context, and attributes.

Caller provenance is observational. If a caller force-mutates an `InvocationContext` into an
invalid state after construction, AgentLoop drops those caller-supplied fields and still records its
authoritative run and turn identity; malformed metadata cannot prevent the adapter call.

The receipt is delivered through subscriptions only. AgentLoop continues to use the provider turn
for usage/budget accounting. Receipt data stays out of `events.jsonl`, `transcript.jsonl`,
`model-content.jsonl`, the run result, and checkpoints. A subscription's `CapturePolicy` governs
that model-I/O delivery; the private content records and separate event-sink channel keep their own
policies.

Subscriptions passed to AgentLoop are owned by that run activation, like `event_sinks`. The loop
identity-de-duplicates and closes observers at normal close, successful durable release, discard,
and bootstrap/restore failure. A loop whose subscriptions have closed rejects reactivation; build a
fresh AgentLoop with fresh observers. `RunnerBackend.model_io_subscription_factories` is the
Reference composition seam and each factory must return an ownership-unique subscription on every
call. Partial construction is cleaned before the build error propagates.

In-process subagents do not inherit a parent's observer instances: simultaneous parent/child calls
would share callbacks and a child close could terminate the parent's exporter. Their
`InvocationContext` does preserve the caller unit and attempt while appending a subagent task segment
and child turn id, and adds root/parent/task/definition/depth attributes. Current in-process child
calls are therefore not delivered to model-I/O observers. A child-scoped observer composition seam
is deferred to a later change.

#### AgentLoop model-stream observers

`AgentLoop.model_stream_observer_factories` exposes provider-independent output and reasoning
fragments during autonomous runs. Each factory returns a `ModelStreamObserver`; the observer opens
one `ModelStreamWriter` for every provider call using a `ModelStreamContext`. The writer receives
ordered `ModelStreamDelta` values and one terminal `ModelStreamOutcome`. `TextDelta` maps to the
`output` channel and `ReasoningDelta` maps to `reasoning`. Tool-call fragments stay inside model
turn assembly. A failed outcome's `retryable` flag carries the provider's transient-failure signal
used for automatic retry eligibility; `false` does not prevent an explicit user reissue after a
configuration change.

Factories materialize a fresh observer set for every activation and every in-process subagent.
This ownership prevents a restored run or child from closing another activation's live channel.
Factory, open, push, and close failures are contained; an observer cannot change a paid model
call's result. `safe_open_model_stream` applies the same failure shield to custom observers.

`AgentLoop.stream_model_calls=True` selects `astream_turn` without selecting an egress surface.
This keeps token-boundary interruption responsive while durable events remain compact.
`emit_output_deltas=True` preserves the legacy opt-in behavior that writes raw
`model.output.delta` and `model.reasoning.delta` events. `AgentLoop.astream` keeps its existing
execution-owning stream contract, continues to expose all `ModelStreamChunk` variants, and takes
precedence over the legacy durable mirror for that call.

`AgentLoop.model_content_file=True` writes the optional private `model-content.jsonl` sidecar.
Records use `monoid.model-content.v1` and the kinds `stream_opened`, `stream_segment`,
`stream_closed`, and `settled_text`. Existing run directories may omit the file. While this option
is enabled during the compatibility window, settled text is written to both the sidecar and
`transcript.jsonl`. Entitled readers resolve a digest from the sidecar before falling back to the
transcript. `flush_active_model_content(path)` gives an in-process presentation layer a coordinated
flush point for currently buffered segments while a stream remains open. Ordinary sidecar reads
remain side-effect free. A `stream_closed` record carries the boolean `retryable` signal; readers
treat its absence as `false` for records written before that field was introduced.

`AgentLoop.model_calls_file=True` writes the optional private `model_calls.jsonl` ledger: one
`monoid.model-calls.v1` record of kind `model_call` per settled call, **including failed ones**.
Existing run directories may omit the file, and it is independent of `model_content_file` and
`stream_model_calls` — a ledger of what was called selects no provider streaming. A subagent
inherits the switch and records into its own run directory, so every record carries `root_run_id`
as the join key for a run tree.

A record is a **declared projection** of `ModelCallReceipt`, not its serialization, and does not
round-trip back through `ModelCallReceipt.from_json`: that reader would fill `timeout_s`,
`gateway_url` and `retry` with defaults and produce a receipt asserting a transport policy the
call never ran under. Four consequences follow from the projection being declared rather than
derived:

- The configured endpoint is absent. `ModelConfig.to_json()` emits `gateway_url` and the gateway
  adapter resolves its destination from that same field, so a serialized receipt would write the
  endpoint in the clear beside the digest that conceals it. The rule the ledger holds is that a
  record carries statuses and digests but
  never the preimage of a digest it also records — and because `destination_digest` is keyed
  under a per-process secret, one such line would make every other digest in the file confirmable.
- `destination_digest` is therefore absent as well, while `destination_status` is recorded. The
  digest identifies a destination within one process; the file outlives that process, so a durable
  copy would name one destination two ways across a recovery and read as a deployment change that
  never happened. The status names which of the four probe outcomes happened and stays true.
- `redaction_digest` is absent. It is a per-subscription fact set only by the capture narrowing, so
  on the recording seam it is always empty; writing it would state "no redaction rules were
  applied" on lines where a redacted consumer applied rules.
- Transport policy (`timeout_s`, `retry`) and the derived `succeeded` / `trace_id` / `span_id`
  properties are absent. The derivations' inputs — `error_code` and `traceparent` — are recorded.

`call_index` counts within one activation and restarts at zero when a durable run reopens its
directory and appends to the ledger it already has. It is a gap detector, not a join key: a
restart is self-evident because the index drops while `recorded_at` advances.

`AgentLoop.model_payload_file=True` writes the optional private replay corpus:
`model_payloads.jsonl` (`monoid.model-payloads.v1`; kinds `chunk`, `model_request`,
`model_response`) plus a `model_payloads/` directory of content-addressed files for values past
256 KiB. It is independent of every other switch — in particular it does not join the streaming
selection, so a corpus-only run keeps non-streamed cancellation granularity — and a subagent
inherits it, recording into its own run directory with `root_run_id` as the join. Unlike the
ledger beside it, this artifact is **content**: request records carry the conversation and the
tool definitions, response records carry model output and provider reasoning artifacts, and the
whole file inherits the run directory's private access boundary.

A `model_request` record is a **recipe, not a copy**: every value at least as large as the
reference that would replace it is lifted out as a content-addressed chunk -- per tool definition,
per message and per observation. Tools and history are resent unchanged turn after turn, so those
two carry the deduplication; observations are a per-turn delta, elementwise for the within-turn
half of the argument, so one oversized tool result does not drag its siblings onto the line with
it. The record verifiably
reassembles to the exact bytes the key was taken over — substitute each chunk's decoded value,
re-encode through the canonical encoder, and the SHA-256 equals `request_digest`. The writer
performs that reassembly *before* writing and falls back to a verbatim payload (`refs: false`,
never walked) when anything disagrees; `monoid validate` re-verifies every request record against
its own digest. Caller data shaped like a chunk reference does not reach that arm: a reference is
a fixed size, so anything that could be one is large enough to be lifted into a chunk, and a
resolved value is never re-walked. Records are set-keyed by digest — a run that issues the same request twice writes
one request record, and both ledger lines join to it — while `model_response` records are
sequence-keyed by `call_index`, because models are not functions: every answer is recorded, and
choosing which one a replay returns is the replay adapter's policy, not the corpus's. Both keys
are **per activation**, like the ledger's: a durable reopen starts a fresh recorder, so a
re-issued in-flight request appends a second request record and `call_index` restarts at zero. The
two records share a digest and a payload, not a line — every record carries its own `recorded_at`
— so a consumer collapsing duplicates keys on `request_digest` (and on `sha256` for chunks), never
on the line. What joins a response record to its ledger line across that boundary is the
pair — the two arms record one call under one lock and read the clock once, so `call_index` *and*
`recorded_at` agree by construction. A failed
call records its request — when a key was issued for it — and no response; the ledger line carries
its taxonomy. A call refused a key records no *request* either: the preimage it would stand for
has no key to be filed under. Whether its **answer** is recorded depends on whether there was one,
and the three keyless statuses differ:

- `absent` and `too_large` are refusals of the *key*, not of the call — the request is sent, the
  model answers, and that answer lands as a response record with an empty `request_digest`. So a
  keyless call is not a call whose content stays off disk, and this artifact's retention
  classification does not change with `digest_status`.
- `not_reached` means the call was refused before a key was computed, so it never settles a turn
  and contributes nothing to the corpus at all; the ledger line is its only trace.
(`withheld` never appears on either sidecar: the capture narrowing that produces it builds a
per-subscription copy, exactly as it does for `redaction_digest` above, and the recording seam sees
the un-narrowed receipt.) A response the canonical encoder cannot carry, one past
`MAX_MODEL_PAYLOAD_BYTES`, or one whose assembled record line nests deeper than this artifact's own
reader parses, costs its own record a typed `unrecorded_reason` — never a truncation, and never a
dropped line.

Whether media is *present* depends on the adapter, and an operator sizing or classifying this
artifact should read that first. When the adapter declares `supports_multimodal` the loop resolves
by-reference attachments into inline bytes **before** it builds the request, so the preimage — and
therefore the request record — carries the image itself, base64 and all: the largest thing this
artifact can hold, and one more reason it is content-classified. A text-only adapter never reaches
that resolution, so its corpus carries references and no image bytes. The `observations` term is
by-reference either way, because it is hashed as the tool returned it; a `workspace:` reference is
not content-addressed, so re-reading one later can yield different bytes under a digest that has
not changed.

Two deliberate absences. `ModelTurn.raw` is not recorded: it has no shape contract and no
consumer outside the provider layer, so a replayed turn answers `raw={}` — an honest statement
that this is a replay, not a gap. And the configured endpoint is absent here for the same
structural reason as in the ledger: the preimage is built from the replay key's own projection,
which excludes `gateway_url` by construction. `reasoning` *is* recorded, encrypted entries
included, because the loop re-injects it into the next by-value turn — a corpus without it would
derail one turn after every replayed answer.

#### Replaying the corpus

`ReplayModelAdapter` (`monoid_agent_kernel.providers`, W6-4b; `monoid run --replay-from` is its
CLI face, [CLI.md](CLI.md)) serves `next_turn` from one or more recorded run directories and
refuses everything it cannot prove. The contract:

- **Selection is file order, each answer once.** The corpus records what happened; the adapter
  hands answers back in that order. Consuming a record whose body cannot be given back
  (`unrecorded_reason`, an unresolvable reference, a body that is not a recorded turn) is a
  `not_recorded` miss that **leaves the cursor standing**. The loop's contract for a
  `config_recoverable` failure is an idempotent re-attempt of the *same* call, so a refusal that
  advanced would answer that re-attempt with the recording belonging to the call after it —
  silently, as a structurally valid turn. A slot is spent only when the caller moves the
  conversation past it by serving that call another way (`--replay-fallthrough`); a record the
  reader handed over that reconstruction then rejected is given back.
  Both settlements are properties of *leaving a take*, not calls a caller remembers to make:
  `ReplayCorpus.take(digest, generation=...)` is a context manager whose block declares
  `served()` when the call happened and settles unserved by every other exit. The two
  directions are opposite — a standing refusal is spent forward, a rejected record is given
  back — and choosing between them at the call site is what repeatedly went wrong.
  Duplicate request records and a restarting `call_index` are the ordinary durable-resume shape
  and collapse by digest, exactly as the previous section specifies.
  Across a union, "file order" spans the sources in the order they were named, and it is
  decisive wherever two sources can answer one key. That includes families: keys are disjoint
  across a family only when no two children share a definition and a prompt, because nothing
  run-scoped is in the key, so an ordinary fan-out of two identical children records one key in
  two run directories. The reader counts the keys more than one source can answer
  (`crossed_keys`) and the CLI preflight warns; it is not otherwise visible.
- **Misses are typed and content-free.** Six reasons, fixed: `no_key` (the live request could
  not be keyed), `absent` (nothing recorded under the key — including the failed-original-call
  shape, whose request record has no answer beside it), `not_recorded` (an answer slot exists
  but cannot yield a turn), `identity_mismatch`, `exhausted`, `generation_mismatch`. Without a
  fallthrough adapter a miss raises `ReplayMiss` — `error_code: "replay_miss"`, the sub-reason
  in `provider_error_code`, `retryable: false`, `config_recoverable: true`, so it parks a
  session and promotes to the failure record only when a one-shot facade closes. The message
  names generations, config values (which the ledger already records in plaintext) and term
  names with digests — never conversation content, and that is pinned adversarially on the
  public `turn.failed` payload.
- **The replay run's config authors the key's model identity** (the loop always sets
  `request.model`), so the commonest total miss is a config that does not match the recording.
  The CLI preflights exactly this comparison before the run starts, through the same function
  the miss diagnosis uses.
- **Impersonation is derived from evidence.** The corpus `provider` term is a resolved value
  that cannot say whether the original adapter *declared* a provider, and the loop's reasoning
  re-injection reads only the declaration — so the adapter declares when the recorded requests
  carry re-injected reasoning blocks, refuses to declare when answers carry reasoning that no
  recorded request re-injected, and declares (for key stability) when there is no reasoning at
  all. Unions that recorded more than one provider are rejected at construction; a kernel-driven
  family cannot produce one (children share the parent's adapter instance).
- **Tools re-execute for real.** Only model answers are replayed; a tool that answers
  differently than it did changes the next request and that turn misses `absent`, the diagnosis
  naming the diverging terms. The same mechanism bounds families: children replay from the
  union of their run directories (naming them is a requirement, not a convenience), and the
  parent's first post-spawn turn is a documented v1 limit — the spawn observation embeds
  per-run identifiers a replay honestly cannot reproduce, and fabricating them would be exactly
  the invented identity the key doctrine forbids.
- **Ledger deltas, five, all here on purpose** (the fourth, `attributes.replay_from`, is the
  provenance stamp described under Provenance in [CLI.md](CLI.md) and is added by `monoid run`
  rather than by the adapter)**:** the adapter declares no `resolve_destination`,
  so a replay run's `destination_status` reads `not_declared` even when the original resolved;
  when the no-reasoning rule declares a provider an undeclared original did not, the replay
  ledger's `provider_name` is non-empty where the original's was `""`; and under
  `--replay-fallthrough` a call the *inner* adapter served is still stamped with whatever the
  wrapper declares — the corpus's provider term, or `""` where the derivation declined to
  declare at all — and with `not_declared`, because the declaration is what makes recorded keys
  reachable and cannot simultaneously report who answered a miss. Whether a corpus recorded
  through fallthrough is interchangeable with a live recording therefore depends on which
  branch the derivation took: where it declares, the term it declares *is* the original's
  resolved provider and the keys agree; where it declined, the live calls are keyed under a
  term a correctly-configured live run will not compute. The fifth is the retry layer's: an
  original that retried under `ModelRetryConfig.layer="kernel"` records `attempts=N`, an
  N-entry `attempt_log`, and usage summing every billed attempt, while its replay — served
  from disk without re-spending — records the one dispatch it actually made. Both lines are
  true; the ledger describes its own run's transport, and recording is settle-driven, so the
  corpus the two runs share is shape-identical either way.
- **A miss message names run ids as well as terms.** The content-free rule bounds *values*, not
  identifiers. Three message families carry those identifiers. A term-by-term diagnosis
  names up to four diverging terms with a 12-hex digest prefix on each side and the run id of
  the record it compared against; a `not_recorded` refusal names the run id and `call_index` of
  the record it refused; and an `identity_mismatch` diagnosis names the run id of the closest
  recorded request together with the identity values on each side — in plaintext, because they
  are config vocabulary the ledger already records in the clear, and bounded to 120 characters
  per value because the corpus is not trusted to keep them short. All three reach the public
  `turn.failed` payload, in `failure.json`, and on CLI stderr. Run ids are minted hex, but they
  are foreign run ids when the corpus is foreign, so an event stream relayed to end users
  carries them.
- **Sources are read, never written**, and they are content: replaying a foreign run directory
  means reading that run's conversation bytes, so the corpus's privacy classification travels
  with the replay. A replay run's own recording switches write into its own directory only.

The generation rule above is what retires a corpus: a composition change bumps
`monoid.model-request-digest.v1`, and a replay whose sources carry **only** the retired tag
answers `generation_mismatch` naming both tags, at the CLI preflight and again from the miss
diagnosis, instead of silently missing.

Two limits on that sentence, because it used to be written without them. The comparison is
whether this run's tag appears **anywhere in the union**, so a union mixing a current source
with a retired one does not report `generation_mismatch` at all: the current tag is present, the
retired source's keys were composed differently and therefore match nothing, and its calls come
back `absent` with a term-by-term diagnosis that names conversation terms rather than the
generation. And the tag is consulted when a key is *looked up*, not when an answer is handed
back — a record whose own `digest_generation` differs is not re-checked at the moment it is
served, because after a real bump its digest cannot match the key being asked for in the first
place. Both are visible only on a corpus whose tags were edited without recomposing its keys.

The Reference backend can attach one `LiveModelStreamBroker` through
`RunnerBackend.model_stream_broker`. Its observer factory is bound to the authoritative root run
and inherited by in-process descendants, producing one passive root-multiplexed presentation
channel. This observer does not write durable events and holds no run control handle. The separate
`RunnerBackend.stream_model_calls`, `model_content_file`, `model_calls_file` and
`model_payload_file` switches let a host select provider streaming, private retention, the
per-call ledger and the replay corpus; live delivery is the separate `model_stream_broker` above.
The two recording switches are independent of the streaming selection and of each other;
`model_content_file` is not, because retaining a content stream implies selecting one. Each
boolean is read when the backend builds an activation — the submitted run and the one recovery
rebuilds alike — so it is a property of the host that builds, not of the run: a run reclaimed by a
node configured differently records differently from there on. The CLI exposes the two recording
switches on both shapes, as `monoid run --model-calls-file` / `monoid run --model-payload-file`
and `monoid backend serve --model-calls-file` / `monoid backend serve --model-payload-file`.

Run cancellation and the session deadline cancel an in-flight native `anext_turn`, coroutine
`next_turn`, or `astream_turn`. Stream cancellation closes the async iterator and runs its cleanup;
cleanup may use at most `AgentLoop.async_model_cancel_grace_s` before the provider task is detached
so a cancellation-suppressing adapter cannot block the run result. Turn interrupt and pause remain
step-boundary signals for non-streamed model calls. A synchronous `next_turn` observes the same two
run boundaries: Python cannot force-stop its worker thread, so exceeding a boundary *abandons* the
call rather than stopping it. The grace interval applies to the worker itself — a call that returns
inside `async_model_cancel_grace_s` settles normally and is not abandoned, so the boundary is
reported once the worker has stopped rather than while it races run finalization. Only a call still
running when the grace expires is abandoned: the run reports `cancelled` or `run_timeout` while the
worker keeps going and its late outcome is discarded. A settled worker does not change the outcome —
the grace is not an extension of the deadline.

"Discarded" is about the *result*, not about everything the call touched on its way there, and
an adapter holding shared state needs a way to hear about it. Two routes reach a discarded
outcome, and neither is the worker: a boundary is checked **before** the completed task's result
is read — deliberately, so a boundary landing in the same tick still wins — so even a call that
finished can have its answer thrown away.

`ModelCallRunner` therefore offers adapters an optional `discard_turn(request, turn)`, invoked at
exactly that point, because it is the only place that knows both that a result exists and that
nobody will see it. Absent, behaviour is unchanged. `ReplayModelAdapter` is the shipped
implementer: `consume` advances its per-key cursor when the answer is handed over, so without the
hook a discarded call permanently consumed a recorded answer and every later consumer of that
corpus was served the following one — a structurally valid turn belonging to a different call,
reachable wherever the adapter outlives the call, which a subagent family already does. The hook
releases the slot, keyed by recomputing the request's own digest so a recycled object identity
cannot release a slot belonging to another key.

One residue, stated rather than closed: a **fallthrough** answer served live is not taken back.
That call reached a provider and was paid for, and the corpus has no unspend primitive for a
refusal spent forward. Sync adapters should still enforce their own provider
I/O timeout and idempotency policy, because the kernel can stop waiting for a call it cannot stop.

Abandonment is not free, and this is a known limitation rather than a settled guarantee: nothing can
reclaim the thread of a call that never returns, and the run no longer blocks to throttle the next
attempt, so an implementation that wedges *permanently* accumulates one thread per abandoned call
across runs. Each abandonment is logged as a warning on the `monoid_agent_kernel.core.sync_bridge`
logger — both the synchronous and the asynchronous half — so the growth is visible; there is
currently no cap on outstanding abandoned calls. A streamed call whose `aclose()` outruns the same
grace is abandoned too, and warns on `monoid_agent_kernel.model_call`.

Nor is there a bound on *healthy* concurrent sync calls. A dedicated daemon thread per call is what
makes abandonment possible, but it gives up the thread-pool bound a shared executor provided: within
one run sync calls are sequential, so this is a per-process concern for a host driving many runs at
once, where a burst can reach the process thread limit and fail calls that would otherwise succeed.
Hosts that run many concurrent sessions with synchronous adapters or tools should bound admission
themselves until the kernel does. Both bounds belong with per-call resource policy rather than with
the dispatch helper, and are tracked for a later release.

`ModelRequest` carries:

- `instruction`
- `system_prompt`
- `tools: tuple[ToolSpec, ...]`
- `previous_turn_handle`
- `observations`
- `model: ModelConfig | None`
- `messages` — the by-value conversation log; when set it overrides the handle path
- `output_schema` — provider-neutral response schema (see the delivery section above)

Adapters must use `request.model` for turn-level model selection when present.
`GatewayModelAdapter` and `OpenAIModelAdapter` follow that rule.

**By-value is the only continuation route on `OpenAIModelAdapter`.** It sends `store=False` on
every request (zero data retention, paired with `include=["reasoning.encrypted_content"]` so the
reasoning round-trip travels by value), which means no response is ever persisted for a handle
to name. A request carrying `previous_turn_handle` — with `messages` unset, so the by-reference
shape is the one selected — is therefore refused at the adapter boundary with a non-retryable,
`config_recoverable` `ModelAdapterError` (`unsupported_request_shape`) naming `messages` as the
supported route. It is a fail-closed refusal of a shape that cannot work here, not a claim about
the shape in general: an adapter whose provider does persist responses is free to support it, and
a stale handle riding *beside* `messages` is unaffected, since `messages` selects the shape.

That by-value round-trip now survives the gateway hop in **both** directions. The request half
always did — `messages` are forwarded verbatim, so a tagged reasoning block reaches the upstream
adapter unchanged — but the response half did not, so a run routed through the gateway captured
nothing to replay. The `reasoning` key on the success envelope (documented with the LLM gateway
wire below) closes that: the artifacts cross the hop on both transports, are tagged with the
relaying adapter's upstream `provider_name`, and are replayed to it on the next turn. The one
shape that cannot carry them is a stream that ends without a terminal frame — it has no
end-of-turn metadata channel at all, exactly as it has none for `usage` or the turn handle — and
that absence is tolerated rather than refused: the turn reads no artifacts and the next turn
replays none.

**Replay is guaranteed only for the active window, and requests are pruned to it.** The active
window is everything after the last `user` message — the in-flight tool loop. A captured
reasoning block is replayable while it sits inside that window, and once a new user message
lands it is outside forever, because the window only moves forward. "Last `user` message" means
the last message *with role `user`*, whoever wrote it — the end user's next turn, and equally a
user-role message the **kernel itself** authors: the `OutputValidator`'s repair prompt, and the
observation messages a background or HITL result arrives on. An operator reading "a new user
turn" would guess those do not count; they always have, because the adapter's replay filter
reads the same role. So they are window boundaries, and everything before them is dead.

The kernel therefore builds each request from a wire copy with the `reasoning` key removed from
every message before the window start. Both of its request builders do — the loop's per-turn
wire copy and the standalone validated call's repair request, which appends an assistant turn
and a user-role repair prompt and so kills every block it was carrying. What the provider sees is
unchanged in both — verified byte-identical on the repair path: outside the window the adapter
already reconstructed those turns from `content`/`tool_calls` and never read the block. What changes is
size — the un-pruned request re-sent one dead block per user turn on every later request, which
grows with the conversation and is paid to the provider, to the gateway in between (twice on
that route), and to the resolved-wire guard that `max_message_log_bytes` bounds. Note that on a
media-free run the durable-log check reads the same limit over a strictly larger log, so it is
that check, not the wire one, that a runaway conversation trips first; the prune buys request
bytes rather than a new headroom regime. The rule lives in one function
(`providers/_common.reasoning_replay_window_start`), read by both the adapter that decides what
to replay and the kernel that decides what to send. **The durable message log and the checkpoint
are not pruned** — they keep every captured block verbatim, so a restored run and a forensic
reader both see what the model produced. Both routes benefit: the direct OpenAI one (bytes to
the provider) and the gateway one, where the block would otherwise cross two hops. `prompt_digest`
identifies the conversation the model actually saw, so it is taken over the pruned request; the
same conversation digests differently than it did before this rule, which is the digest staying
honest rather than a compatibility break — nothing compares digests across versions.

### Output Validation

A developer-supplied `OutputValidator` guarantees the final response conforms to a shape the
caller defines — a JSON schema, business rules, cross-checks against produced files. The engine
owns the bounded re-prompt orchestration; the validator owns the judgment. Two execution
surfaces consume one shared routine (`run_output_validators` / `build_repair_message` in
`core.output_validator`), so the exception classification and the repair dialect cannot drift
between them.

The types:

- `OutputValidator` — protocol: `id: str`, `schema: dict | None` (carried, inert — the
  delivery mechanism is `ModelRequest.output_schema`), and a synchronous
  `validate(view: FinalOutputView) -> ValidationOutcome`.
- `ValidationOutcome` — `ok` + `value` (the validated value, surfaced as `final_output`; must
  be JSON-serializable to persist) + `feedback` (the repair text on rejection).
- `FinalOutputView` — the read-only composite a validator sees: `final_text` (always),
  `artifacts`, `final_outputs`, a jailed size-capped `read_bytes`, and `parsed` (best-effort
  JSON view when the call carried an `output_schema`; a convenience, never the guarantee).
  `parsed` goes through the kernel's strict JSON ingress, not bare `json.loads`, so Python's
  non-standard `NaN` / `Infinity` constants — which a schema validator would happily call
  numbers — leave it unparsed like prose does, along with duplicate keys, unbounded integers,
  and runaway nesting. **`parsed_ok` is the authority, not `parsed is None`**: a schema
  permitting a root `null` produces a valid parsed value of `None`, so a validator rejecting on
  `parsed is None` would fail a conforming answer. Every field but `final_text` defaults
  (`parsed_ok` to `False`), so the view is constructible with zero loop context.
- `OutputRetry` — raising it from `validate` equals returning a rejection with feedback.
- `OutputValidatorBinding` — the per-run opt-out: a registered validator runs by default
  (registration = activation); a binding with `enabled=False` disables it for that run.
- `OutputValidatorError` (`error_code="output_validator_error"`) — a validator *defect*.

**Exception classification is the protocol contract.** `OutputRetry` and `ValueError` (which
covers `pydantic.ValidationError`) are rejections whose text becomes repair feedback; any other
exception is a defect — the model cannot fix a validator bug, so the run terminalizes (or the
standalone call raises) rather than re-prompting, and the exception text is never fed back to
the model.

**In `AgentLoop`** validators run at both settle points (natural text settle and `run.finish`),
after the refusal/truncation branch: a refusal terminalizes `output_refused`, a
length-truncated answer settles `limited`/`output_truncated` **without validating** — a cut-off
but well-formed prefix must not pass as success. All active validators run and all failures are
collected into one re-prompt; `RunLimits.max_output_retries` (default 1) bounds re-prompts, and
exhaustion settles `limited` with `output_validator_unsatisfied`. A repair turn is a full agent
turn: it may call tools and shares the global step/tool/token budgets, so `max_output_retries`
bounds settle attempts, not total cost. The retry counter is checkpointed; a mid-repair restart
does not re-grant the budget.

**Standalone** (`ValidatedCallRunner` / `ValidatedCallResult`): the same guarantee for a caller
invoking `ModelCallRunner` directly, with the opposite tool posture — **a repair call never
carries tools** (there is no executor behind the surface, and a validation failure must not
escalate into a tool loop). `max_repair_calls` (default 1) bounds explicit repair calls;
exhaustion is a *result* (`status="unsatisfied"`), and three outcomes short-circuit before any
validator runs, in the loop's ordering: refusal (`"refusal"`), truncation (`"truncated"`), and
a **tool-call answer** (`status="tool_calls"`, keyed on `turn.tool_calls` as well as the
stop reason) — this surface has no executor, so a turn that stopped to request tools is handed
back with its calls rather than having its empty text judged and repaired. `receipts` carries
every call made on every settled result, and an exception escaping `acall` carries the
completed calls' receipts as its `receipts` attribute (declared on `OutputValidatorError`,
stamped best-effort on anything else); the failing call's own receipt exists only on
`ModelCallRunner.subscriptions`, because the adapter raised instead of returning it. Repair
follows the shape of **how the incoming request carried its conversation**, never what the
answer came back with (by-value messages append; a request that itself arrived on a
continuation handle carries the repair as the next instruction on the new handle; a one-shot
instruction is synthesized into by-value form), preserves `output_schema`, and **clears the
carriage fields of the shapes it did not choose** — a repair request carries its conversation
exactly one way, so its `request_digest` describes a request an adapter actually sends. A
one-shot call is never promoted to the handle path just because
the provider returned a response id — `OpenAIModelAdapter` sends `store=False`, so that id was
never persisted and the repair would 404 (today it never leaves: that adapter refuses the
by-reference shape outright, so a promoted repair would fail before the call). A request that
came in *on* a continuation handle whose turn came back *without* a new handle has no fourth
shape — the conversation lives on the provider's side of that handle — so it settles
`unsatisfied` without repairing rather than repairing against a prompt the model never saw;
`repair_calls_used < max_repair_calls` on an `unsatisfied` result is that signal.

Streaming is per attempt: `acall` takes an `AttemptDeltaConsumer`
(`(attempt_index, event) -> None`, `0` = the original call) rather than a plain
`DeltaConsumer`, because a rejected attempt's text is discarded output. When the index
advances, everything the consumer holds from the previous index is retracted; the signature
carries the boundary so a consumer cannot concatenate a rejected answer onto the accepted one.
Every attempt opens with an `AttemptStarted(attempt)` event, delivered before any of that
attempt's chunks and **whether or not any arrive** — the index alone could not carry the
boundary, since it rides chunks and an attempt may produce none (a non-streaming adapter, a
stream carrying only its terminal frame, a frameless gateway stream accepted under `"omit"`).
Without it a consumer went on rendering a rejected attempt's text beside an `ok` result. The
events a consumer sees are therefore `AttemptStarted | ModelStreamChunk`.
The sync facade `call` refuses to run inside an active event loop, so it takes no consumer.
`ValidatedCallRunner` is frozen: `max_repair_calls` must be an exact non-negative `int`, and a
budget checked once at construction would not be a budget if a reusable runner could be
reassigned past the check. Reconfigure with `dataclasses.replace`, which revalidates.

### Tool Contract

Add tools with `ToolProvider.get_tools(context) -> Iterable[ToolSpec]`.

`ToolSpec` still describes a registry tool: id, description, JSON schema,
side-effect class, handler, provider name, path args, preview hints, guidance,
examples, and annotations. Registry specs are implementation tools. Bindings
decide model-facing names, guidance, exposure, authorization, scope, quota, and
runtime settings.

Handlers implement either `SyncToolHandler` or `AsyncToolHandler`:

```python
def handler(context: ToolContext, args: dict) -> ToolResult: ...

async def handler(context: ToolContext, args: dict) -> ToolResult: ...
```

The `@tool` decorator preserves `async def` functions and normalizes their awaited return value
the same way as synchronous functions. Native async handlers run on the run loop. Synchronous
handlers run in a worker thread. Tool calls from one model turn execute sequentially in model
order.

Authorization, scope, quota, approval, capability leases, and side-effect admission complete
before the handler starts. `tool.call.started` precedes handler execution; one
`tool.call.finished` or `tool.call.failed` event follows it. Approved and capability-granted
replays use the same async execution path.

Approved calls use durable at-most-once replay delivery. Before each approved handler starts, the
loop checkpoints consumption of that head only; unstarted approvals and observations from earlier
completed heads remain in the durable tail. A process loss after the consume checkpoint does not
retry the uncertain head, so its effect may have happened zero or one time. Handlers that need a
stronger effect guarantee require a stable idempotency key or transactional outbox at their
external boundary.

Run cancellation and the run deadline cancel an in-flight native async handler and preserve the
run-level `cancelled` or `run_timeout` result. Cleanup has a bounded
`AgentLoop.async_tool_cancel_grace_s` window; a handler that suppresses cancellation is detached
after that window so it cannot block the run result. A synchronous handler observes the same two
boundaries and the same window, but cannot be force-stopped, so it is detached without ever
receiving a cancellation to clean up after — the position a cancellation-suppressing async handler
already ends in. The window applies to the worker thread: a handler that returns inside it settles
normally and is never detached, which is what keeps its workspace writes ahead of run finalization
instead of racing it. A handler still running when the window expires is detached and may still be
writing to the workspace after the run stopped waiting for it. An awaitable it returns too late is
disposed rather than left dangling — a coroutine is closed, and a future or task is cancelled and
its outcome consumed. Sync tools that perform external I/O should apply their own operation timeout
and idempotency policy, because the kernel can stop waiting for a handler it cannot stop.

A handler's call authorization follows the handler, including into threads it starts itself: a
`ToolContext` operation delegated to a joined child thread is checked against the same binding scope
as the parent. It also stays valid for a handler the run has abandoned, so a detached handler keeps
its own scope rather than picking up whichever call the run moved on to.

Outside a tool call there is no binding whose scope could authorize anything, so scoped `ToolContext`
operations — path checks, shell execution, web access — **refuse** rather than fall back to the
run-level permission policy. Every scope check narrows only under a non-empty allow/deny list, so
treating "no call" as an empty scope would grant the widest authorization in the run at the moment it
is least warranted. This is what bounds a thread descended from an *abandoned* handler: once the run
gives up on the parent, the thread is refused. One narrower case remains: while some *other* call is
live, such a thread reads that call's authorization instead of its own, because nothing links a
thread to its creator — Python exposes no parent edges — so it cannot be told apart from the live
call's own child thread. It borrows a scope rather than escaping scoping. Handlers that outlive their
run and fan out to further threads should re-check their own authorization rather than relying on
`ToolContext` to narrow for them.

`ToolExecutionError`, `PermissionDenied`, validation failures, and other controlled contract
errors become failed tool observations. A handler-local `CancelledError` maps to
`tool_handler_cancelled`; run-token cancellation and deadlines retain their run-level outcome.
Unexpected handler exceptions fail the run through the normal recording boundary. Cancellation
cleanup runs before the call context is cleared.

`ToolResult.to_observation()` returns:

```json
{"ok": true, "result": {"value": "..."}}
```

Failures return:

```json
{
  "ok": false,
  "result": {},
  "error": {
    "message": "...",
    "code": "tool_handler_error",
    "category": "tool",
    "retryable": true
  }
}
```

### Shell And Web Bindings

Shell availability is the presence of an exposed `shell.exec` binding.

- command allow/deny prefixes and env allowlist live in `ToolBinding.scope`
- timeout, output, startup wait, approval mode, shell kind, and execution
  workspace live in `ToolBinding.runtime.shell`
- `ShellExecutionOptions` is an internal low-level execution options object

Web availability is the presence of exposed `web.search`, `web.fetch`, and
`web.context` bindings.

- domain allow/block lists live in `ToolBinding.scope`
- result limits, context limits, timeout, response-byte limits, and call limits
  live in `ToolBinding.runtime.web`
- gateway requests include `binding_id`, `max_calls`, and effective constraints

### Async Tasks

Long-running work whose result feeds back to the model — shell background jobs,
human-in-the-loop requests, automation — flows through one generic task system.
The core (`TaskManager`) owns the queue, lifecycle, reentry, and artifacts; three
seams are pluggable:

- `TaskExecutor` — how a task kind runs and when it is done. The shell executor
  monitors a subprocess in-process; a hosted kind (hitl/automation) has no
  monitor and is completed by an external reporter.
- `ResultInjector` — how a finished task is injected into the model: as a tool
  observation (`is_background=False`) or as a new user message
  (`is_background=True`).
- `TaskReporter` — how the backend drives tasks in a running run: `create_task`
  and `report_result`. Transport-agnostic — only `(task_id, dict)` cross the
  boundary, so an in-process reporter and a future durable/cross-process reporter
  share the same shape.

Both the model (via tools such as `hitl.request`) and the backend can create
tasks; a completed task wakes a parked run through the shared reentry queue.

### Subagents (agent-as-tool)

A run can delegate a focused task to an isolated child run. This reuses the Async
Task seams above via a `subagent` task kind (`SubagentTaskExecutor`); see
`docs/SUBAGENT_DESIGN.md` for the full design.

- **Enable**: pass `AgentLoop(subagent_definitions={<id>: SubagentDefinition})`.
  When non-empty, the bootstrap registers the `agent.spawn` tool. The runtime config
  still needs an explicit binding to `agent.spawn` (e.g. `model_name: "agent_spawn"`)
  for the tool to reach the model.
- **Definition** (`SubagentDefinition`, Claude-style — everything inherits the parent
  by default): `description` (surfaced to the model for selection), `prompt`,
  `model` (None → inherit), `tools` (None → inherit ALL parent tools; a tuple is an
  allowlist), `disallowed_tools` (denylist, applied after the allowlist — deny wins),
  `mode`/`limits` (None → inherit), `tool_search` (None → inherit). `tools`/
  `disallowed_tools` entries are fnmatch patterns matched against each parent binding's
  tool id / binding id / model name (so `fs.read`, `mcp.*`, `mcp.github.*`, `*` all
  work). The allowlist is resolved **against the parent's bindings**, so a subagent can
  never exceed the parent (hard ceiling); the parent's MCP/custom tool providers are
  inherited by the child so inherited bindings resolve.
- **Tool**: `agent.spawn(subagent_type, prompt, background=false)`. `subagent_type` is
  constrained to the configured ids. The child runs in an isolated overlay workspace and
  sees only `prompt` (not the parent's conversation). Foreground (`background=false`)
  blocks and returns the child's final message as the tool result; background returns a
  `{spawned, background, task_id}` ack and the child's final message is delivered later
  as a user message.
- **Bounds** (`RunLimits`): `max_subagents` (fan-out per run, default 8) and
  `max_subagent_depth` (nesting, default 5). Enforced in the executor; a child at the
  depth cap has the `agent.spawn` binding stripped (the tool is absent, not just an
  error at call time).
- **Result shape** (`subagent_result`): `{status, final_text, message, root_run_id,
  parent_run_id, child_run_id, task_id, definition_id, depth, traceparent, subagent_type, usage,
  error}`.
- **Events**: the parent stream carries `subagent.started` (`parent_id` = the spawn
  tool-call event) and `subagent.finished`/`subagent.failed` (`parent_id` = the
  `subagent.started` event), each carrying `root_run_id`, `parent_run_id`,
  `child_run_id`, `task_id`, `definition_id`, `depth`, and `traceparent`; finish events also carry
  the child's `usage`. The child's full event stream goes to its own run dir; external
  `event_sinks` are not shared with children (stateful sinks like OTel/StatusJson are per-run).
- **Usage reporting**: the parent's run metrics carry `subagent_count` and
  `subagent_usage` (the children's combined token totals). Descendant usage is also
  added to root `total_usage`, so run token budgets and backend tenant usage include
  delegated work.
- **Capability boundary**: child loops inherit the parent's broker and share the parent's
  capability vault. A parent-level capability revoke is visible to child gated tools before
  a broker request or gateway call can happen.
- **Context fork** (`SubagentDefinition.context = "fork"`): instead of a fresh
  isolated context, the child inherits a snapshot of the parent's conversation AND the
  parent's prompt / tools / model (the definition's own prompt/tools/model are ignored)
  — "continue as me in an isolated branch". `"fresh"` (default) is the normal isolated
  subagent that sees only the task prompt.
- **Directory discovery**: `load_subagent_definitions(dir)` (CLI `--agents-directory`)
  scans `*.md` files with YAML frontmatter (`.claude/agents` style) into
  `SubagentDefinition`s. Frontmatter fields: `name` (id; falls back to filename),
  `description`, `tools` (omitted → inherit all), `disallowedTools`, `model` (string
  shorthand or `inherit`), `mode`, `context`; the markdown body is the system prompt
  (fresh subagents only). Parsed by `parse_frontmatter` — a zero-dependency YAML subset
  (scalars, inline/block lists, quotes), shared with Skills' `SKILL.md`.

### Skills (progressive disclosure)

A run can be equipped with **Agent Skills** — procedural how-to knowledge (Anthropic's
`SKILL.md` model) delivered to the model by *progressive disclosure*, so a large library
costs almost nothing until a skill is actually used. Skills are a knowledge layer,
complementary to subagents (execution) and MCP (integration). The whole feature attaches
through the existing `ContextProvider` + `ToolProvider` seams with **no core-loop change**
(`SkillProvider` implements both); see `docs/SKILLS_DESIGN.md`.

- **Enable**: build a `SkillProvider(definitions)` and register the one instance in both
  `AgentLoop(context_providers=(provider,), tool_providers=(provider,))`. Provider tools
  require explicit bindings; merge `provider.tool_bindings()` into the runtime config so
  the `skill` tools reach the model (mirrors the MCP provider). The CLI
  `--skills-directory` does all of this.
- **Definition** (`SkillDefinition`): `name`, `description` (both advertised at L1),
  `instructions` (the SKILL.md body, delivered at L2), `allowed_tools` (advisory for
  inline skills and enforced for fork skills, see below), `context`
  (`"inline"` default | `"fork"`), `directory` (bundle root for L3), `metadata`.
- **Fork skills** (`context: fork`): the skill runs as an isolated **subagent** (reusing
  the subagent machine) and only its final message returns. Heavy skills keep their working
  noise out of the main context. The model calls `skill(name, task)` with `task` describing
  the goal; the subagent's persona is the skill's instructions and `task` is its first user
  message. A **non-empty** `allowed_tools` becomes the subagent's tool **allowlist**,
  resolved against the parent's bindings as a hard ceiling. An empty `allowed_tools`
  inherits all of the parent's tools. Enable by
  merging `SkillProvider.subagent_definitions()` (namespaced `skill:<name>` ids) into
  `AgentLoop(subagent_definitions=...)`; the CLI does this automatically. The delegated run
  is reported in the usual `subagent_count`/`subagent.*` events and metrics.
- **Three levels of disclosure**:
  - **L1 — catalog** (~100 tokens/skill, emitted per-turn while the skill tool is bound):
    `SkillProvider.dynamic_segment(turn)` lists each `name: description` in the system prompt
    plus how to load one. It is config-gated — the catalog vanishes when the skill tool is
    unbound, so `static_segment()` returns `None` and the catalog rides the per-turn segment.
  - **L2 — instructions** (on trigger): the model calls the `skill(name)` tool; the result
    carries `{name, instructions, allowed_tools?, resources?}`. Model-native triggering —
    the model picks a skill by its description, no router.
  - **L3 — resources** (on demand): the model calls `skill.read_file(name, path)` to read a
    bundled file (`path` relative to the skill directory, as listed in `resources`), or
    `skill.run_script(name, path, args?)` to **execute** a bundled script and get back only
    its `{exit_code, stdout, stderr, ...}` — the script source never enters context. The
    interpreter is chosen by extension (`.py` → the kernel's Python, `.sh` → bash, `.js` →
    node, `.rb` → ruby, `.ps1` → powershell); `args` are passed to the script **verbatim as
    argv, never through a shell**, so they cannot be re-parsed/injected. The script runs in
    the workspace through the same machinery as `shell.exec` (`side_effect: "shell"`):
    approval, env scrubbing, timeout, and output limits all apply, and it is blocked in
    read-only mode. Path traversal outside the skill directory is rejected
    (`skill_path_invalid`); `SKILL.md` itself is never a readable/runnable resource (it is
    the L2 payload). **Security**: a skill script is arbitrary code — skills are
    operator-provisioned (`--skills-directory`), the same trust boundary as `--tool-module`;
    there is no extra sandbox beyond the shell machinery's defenses, so only load skills from
    trusted sources.
- **Observability**: activating a skill (L2) emits a `skill.activated` event whose
  `parent_id` is the `skill` tool call (so it is correlated to, and an OTel sink enriches,
  that tool's `execute_tool` span with `skill.name` / `skill.resource_count`); data is
  `{name, resource_count}`. The run metrics carry `skill_activation_count` and
  `skills_activated` (the list of activated skill names) — report-only, like the subagent
  roll-up. `allowed_tools` is echoed in the `skill` tool result as an advisory hint.
- **Directory discovery**: `load_skill_definitions(dir)` (CLI `--skills-directory`) scans
  recursively for `SKILL.md` files (the `<skills>/<skill-name>/SKILL.md` convention); the
  skill name is the frontmatter `name` (falling back to the directory name) and the
  SKILL.md's parent directory is the bundle root. Frontmatter fields: `name`, `description`,
  `allowed-tools` (space-separated per the spec, or an inline list), `context`, `metadata`. Parsed by
  the same zero-dependency `parse_frontmatter` used for subagents.

### Session Lifecycle (`AgentSession` + FSM)

`AgentLoop` is the engine; `AgentSession` is the embedder contract a control plane depends
on (so an Agent Daemon/Cell never imports the loop). `LoopSession` is the reference facade
that wraps an `AgentLoop`, owns the FSM, and delegates execution:

- `SessionState` — the formal lifecycle FSM (a `str`-enum): `created`, `idle`, `running`,
  `awaiting_input`, `awaiting_tasks`, `paused`, `interrupted`, `turn_failed`, `limited`,
  `cancelled`, `completed`, `failed`. `cancelled`/`completed`/`failed` are terminal.
  Public run lifecycle payloads expose `state` plus `terminal`. A terminal limit result is
  represented as `state="limited", terminal=true`; a live budget-limited park is
  `state="limited", terminal=false`.
- `state_from_suspension(suspension)` projects a pump `Suspension` onto a state (the seam that
  keeps the FSM in sync with the engine without the engine knowing about it). `LEGAL_TRANSITIONS`
  + `can_transition` / `assert_transition` define the legal edges.
  `session_state_value(state)` serializes the lifecycle value, and
  `session_state_from_run_status(status, error_code=..., terminal=...)` is the tolerant reader for
  older `status.json` payloads.
- `LoopSession.open() / submit() / run_until_suspended() / close()` delegate to the loop and
  re-derive `state` at each boundary. A fresh facade constructed over a loop that already has a
  live session — a restored loop, or one opened before it was wrapped — seeds its state from
  that session at construction (the terminal outcome for a terminal session, the last committed
  park for a parked one, `idle` for an open-but-unpumped one), so `resume()`/`submit()` continue
  a restored run where it parked; a fresh un-opened loop still yields `created`, and an
  explicitly passed `_state` always wins over derivation. The blocking and pump halves share one
  terminal settle
  precedence (cancelled first, then a terminal `status="limited"` to `limited`, else `failed`),
  so a budget-terminal or cancelled `submit()` answers exactly what `run_until_suspended` answers
  for the identical run. `inspect() -> SessionInspection` and `health() -> SessionHealth` are
  recomputed from live loop state on every call (never stale), and both bind to actual loop
  liveness: `can_accept_input` is `false` (and `alive` is `false`) whenever the loop can no
  longer pump — its session is terminal, or the activation was torn down by `close()` /
  `release_parked()` / `discard_uncommitted()` (each records a loop-side finalization fact at
  the same site where it drops the session). Deadness is that recorded fact, never a guess from
  the facade's own state: a findable loop whose `open()` has not yet assigned its session — the
  backend's `aopen` window — answers alive, not dead. A `submit()`/`resume()` over a dead loop is
  refused with the loop's own typed error (`run_not_open` / `run_terminal`) *before* any FSM
  write, so a refused pump never moves the facade; a `close()` that raises lands the facade on
  `failed` rather than leaving an input-accepting park state over the dead loop.
- `pause()` / `resume()` / `cancel(reason)`: pause freezes the turn at the *next start-of-step*
  boundary (its in-flight `pending_observations` are kept), suspends with `reason="paused"`, and
  persists a checkpoint — so resume (a `run_until_suspended(None)` re-pump) continues the same
  turn, in-process or after a restart. Pause lands only at a step boundary (an in-flight model
  call completes first; only an interrupt aborts mid-generation under token streaming). Entering
  `paused` emits a `session.state.changed` event.

### Control Protocol

`monoid.control-command.v1` is a transport-independent envelope + a single
`dispatch` seam, so a Daemon drives a session through one entry point instead of a route per op:

- `ControlCommand(type, run_id, args, issuer, reason, command_id)` and `ControlResult(run_id,
  type, status, state, data, error, error_code)` are plain data (`status` ∈ `ok` / `not_implemented` /
  `unsupported` / `error`). `ControlDispatcher.dispatch(command) -> ControlResult` is the contract;
  `RunnerBackend.dispatch` is the reference impl, routing each command to the in-process method it
  already exposes.
  `ControlResult.status` is command outcome. Run lifecycle appears as `state` plus `terminal` in
  successful command data when the command returns lifecycle information.
- Command types: `pause`, `resume`, `cancel`, `approve`, `deny`, `interrupt`, `inspect`,
  `health`, `send_message`, `runtime_config`, `replace_runtime_config`, `create_task`,
  `report_task_result`, `status`, `revoke_capability`. `approve` and `deny` are explicit
  hosted-task decision aliases over `report_task_result`. An unknown type returns `unsupported`
  (the wire vocabulary stays
  forward-compatible).
- HTTP: `POST /v1/runs/{run_id}/control` with `{"type": ..., "args": {...}, "issuer": ...,
  "reason": ...}`; the bearer token authorizes the run (the route injects it into `args` so the
  envelope stays credential-free). `resume` on a locally owned paused run wakes it immediately.
  A peer with a fresh remote owner appends to the durable command inbox. An absent or stale owner
  returns `command_owner_unavailable`; checkpoint recovery must establish a live owner before a
  control command can be admitted.
- Audit: `RunnerBackend.dispatch` appends `control.command.received` and then either
  `control.command.completed` or `control.command.failed` to the run event log. Events include
  `command_id`, command type, target run, `issuer` as actor, reason, idempotency key,
  result/failure code, result status/error, duration, and a safe `token_sha256` reference — never
  the bearer token itself. A control `send_message` uses the command id as its inbox idempotency
  key.

### Durable Command Inbox

`CommandStore` is the Reference multi-instance transport for control commands. The bundled
`InMemoryCommandStore` serves one-process deployments; `SqliteCommandStore` supplies transactional
idempotent append, ordered claim, stale-claim recovery, acknowledgement, result receipts, and
per-run queue limits across backend instances. Configure every instance with a command store over
the same database used by its shared checkpoint and lease stores.

`POST /v1/runs/{run_id}/control` authenticates the submitted bearer token before enqueueing. The
stored `monoid.command-inbox.v1` envelope contains the command ID, sanitized arguments, reason, and
authenticated tenant/user principal. It never contains the bearer token. The owner mints a fresh
short-lived internal run token when it drains the command. A local owner drains immediately and
the route preserves the historical `ControlResult` response. A remote owner yields a `202`
`monoid.command-receipt.v1`; poll
`GET /v1/runs/{run_id}/control/{command_id}` until `completed` or `failed`.

Task callback bearer tokens are accepted for `approve`, `deny`, and `report_task_result` at enqueue
time and are never stored; the owner executes with its fresh run token. Durable capability
`token_ref` values remain executable handles, consistent with checkpoint durability. Credential
fields in command results, including newly issued callback tokens, are redacted in durable
receipts. The immediate local response returns the original callback token once; a lost secret
response cannot be recovered from the command store.
For that reason, `create_task` is accepted only by the instance currently owning the run; a peer
returns `command_requires_owner` instead of creating a task whose callback credential cannot be
delivered. Route that command to the owner or use the dedicated task API there.
Task callback credentials may poll the receipt for the same callback command and task scope. They
do not gain access to receipts for ordinary run-token commands or commands for another task.
Owner-local commands execute from a transient payload after removing the authenticated bearer, so
legitimate domain fields such as `password` remain intact without entering the inbox. Cross-worker
commands execute from the durable sanitized payload; use durable references such as `token_ref`
for credential-shaped domain data that must cross that boundary.
`create_task` also requires an empty per-run command lane so its one-time callback credential can
be returned by the submitting owner thread. Peers accept durable commands only while the run has a
fresh ownership lease; an absent or stale owner returns `command_owner_unavailable` before append.

Append is idempotent by `(run_id, command_id)`. An identical duplicate receives the existing
receipt and does not execute a second command. Reusing the ID for a different type, sanitized
arguments, principal, issuer, or reason returns `command_id_conflict`. Claims follow append order
per run, with one in-flight command; a later command cannot skip an unacknowledged head command. A
crashed claimant becomes eligible
after `command_claim_ttl_s`; command handlers therefore retain their existing idempotency
obligations under crash-after-effect/before-ack recovery. `command_queue_limit` bounds pending plus
claimed commands per run. Owner watchdogs drain inboxes alongside lease recovery and outbox
redrive.

### Event Reads

**`data.reason` is two vocabularies, on purpose.** On `turn.interrupted` and `turn.paused` it is a
**cause** — what stopped the turn (`"user_stop"`, `"user_pause"`). On `Suspension.reason` (and on
the durable `last_suspension` payload) it is a **park** — the state the session came to rest in
(`"interrupted"`, `"paused"`, `"turn_failed"`, `"awaiting_tasks"`, `"settled"`, `"limited"`,
`"terminal"`). The two sets are disjoint and neither is derivable from the other: the event answers
*why did this stop*, the park answers *where is the run now*. A reader that joins them by field
name is joining two different questions. The two turn-lane stop events are symmetric — a pause
emits `turn.paused` exactly as a stop emits `turn.interrupted` — so a consumer watching the turn
lane sees both parks; the `session.state.changed` event beside them carries the lifecycle
projection, and it is the carrier the status readers consume: `status.json` and the offline
projection show `state: "paused"` from it (the backend record's pause state is owned by the
session driver, which observes the `paused` Suspension directly).

**Status readers carry the whole failure classification, under one rule.** The three consumers
of the run event stream (`status.json`'s live sink, the offline `events.jsonl` projection, and
the backend record) copy the full set — `provider_error_code`, `http_status`, `retryable`,
`config_recoverable`, `provider_retried` — beside `error`/`error_code` when a `turn.failed`
parks the run, because `config_recoverable` alone cannot separate an `insufficient_quota` (fix
the config) from a `rate_limit` (wait). The classification remains for as long as the park does;
a `model.turn.started` clears it (the new turn supersedes the dead one, including on the
no-park retry path), and terminal events assign rather than or-fallback, so a completed run
never keeps a recovered turn's error. A failed terminal keeps the `run.failed` classification —
minus `provider_retried`, the per-call fact the terminal vocabulary deliberately drops.
`GET /v1/runs/{id}/status` and `/result` serve the same five off the record, on the live branch
and on the post-restart (status.json-backed) branch alike; `provider_usage` on `turn.failed` is
metering, not classification, and stays off every status surface.

`GET /v1/runs/{run_id}/events?from_seq=N&limit=M` returns `{run_id, events, next_seq, has_more}`.
`from_seq` remains inclusive for backward compatibility. When `limit` is present, callers resume
with `from_seq=next_seq` to avoid duplicates; omitting `limit` preserves the historical "return all
events from N" behavior. `RunnerBackend.descendant_events(...)` uses the same pagination contract
for subagent event streams authorized through an ancestor run token. Stored event sequences are
positive integers; `from_seq=0` remains the valid cursor for reading from before the first event.

The physical JSONL commit marker is the terminating newline. Readers withhold any trailing bytes
after the last newline, including an otherwise valid JSON object. Before a recorder or guarded
direct append reopens the log, it removes that uncommitted suffix only when the status watermark
does not advertise a later event. A later watermark or a malformed newline-terminated tail fails
closed without changing the file. Logical `seq` values remain the public cursor; binary offsets
are private storage details. Tail preparation runs only after the existing queued, live, or
terminal sequence-owner decision; it validates storage and does not elect another writer. Status
acts only as a contradiction guard. Writer restart continues to validate all committed sequences
and fails on duplicate or decreasing physical values; the derived read index never seeds writers.

The Reference package also provides an internal snapshot-bounded page-scan primitive that can
begin at a content-verified `(seq, byte offset, next byte offset, record digest)` anchor. Tail
capture, bounded scan, and tail-witness verification share one open file description, so pathname
replacement cannot mix records from different files in one page. A concurrent append stays outside
the captured committed prefix. The result reports decoded records, logical scan span, and raw bytes
fetched by the Reference reader, including fixed-size buffer read-ahead, for deterministic scale
tests.

The run directory is a protected append-only trust boundary. Deployments reserve run-artifact write
access for runtime event and metadata owners. Committed event records are immutable, and tool
workspaces and untrusted processes have no write access to run artifacts. Persistent anchors derive
transitively from a contiguous prefix originally verified from byte zero. Each warm seek rechecks
the source identity and nonshrinking committed extent, then verifies the anchored record at its byte
offset. The current snapshot's committed tail witness is verified before new anchors are minted.
The in-memory primitive accepts only anchors minted while scanning a strictly increasing prefix.
Within this boundary, every skipped prefix sequence remains below the anchor sequence, preserving
Core page results.
Each proof is process-local and bound to the normalized source path, open-file identity, and captured
source metadata; copied or field-tampered anchors, cross-log anchors, truncated sources, and
same-size sources with a changed modification timestamp fail closed. A persisted index row remains
an untrusted candidate after process restart and must be reverified from byte zero before minting a
fresh in-memory proof.

A same-inode prefix rewrite followed by suffix growth violates this trust boundary and is
indistinguishable from a valid append through portable path, inode, timestamp, and size metadata. A
live warm anchor can therefore skip rewritten earlier records. A new process verifies the current
bytes from zero. Detecting this mutation while retaining bounded warm I/O requires future
writer-authenticated generation lineage. The Reference projection injects this reader behind its
private page-reader seam; Core subscriptions and the authoritative from-zero reader remain
storage-neutral.

`ReferenceEventOffsetIndex` is the internal warm-read coordinator for those anchors. It retains the
first verified record, sparse anchors at fixed byte or record strides, and the newest verified
record as strong process-local references. One per-source lock single-flights cold construction;
different event logs remain independent. A page scan stages only lightweight stride-selected
candidates plus its newest safe-prefix candidate. It mints and publishes anchor capabilities after
the captured snapshot passes final verification. Path replacement, committed-prefix truncation,
physical shrink, a detected same-size rewrite, or an expired proof clears the derived state and
permits one authoritative from-zero retry. Committed event-log corruption remains authoritative and
propagates unchanged.

The index retains at most 128 source slots by default. Its `max_sources` constructor setting defines
that hard retained-slot capacity; `RunnerBackend.event_index_max_sources` configures the owned
Reference instance. Page reads pin admitted slots before taking the per-source I/O lock, so
concurrent readers for one retained cold source share one construction. A miss replaces the
least-recently-used idle slot. When every retained slot is active, a miss reads one authoritative
snapshot from byte zero without creating a slot or anchor capabilities. This bypass preserves event
delivery and the hard metadata bound. It uses an independent snapshot, and another read can repeat
that cold work until capacity becomes available. Capacity zero routes every read through this
authoritative uncached path.

`cache_stats()` exposes retained sources, pins, page-read hits and misses, idle-slot evictions, and
saturated bypasses. `stats()` and `invalidate()` create no slots and leave recency unchanged. An
evicted source verifies from byte zero on its next read, so deployments size `max_sources` for the
hot polled-source working set. Source capacity bounds retained slot and capability metadata;
per-source sparse-anchor density remains governed by the byte and record strides.

A new process starts with an empty offset index. Its first relevant read verifies JSONL from byte
zero while rebuilding sparse anchors; later pages and append-only-conforming same-process appends
extend from the retained tail. Persisting candidate offsets cannot reduce that required verification
under the current process-local proof contract. Constant-work restart remains coupled to the future
writer-authenticated generation lineage.

Each `RunnerBackend` owns one index and injects its page reader into `RunProjectionService`. Root
events, authorized descendant events, and diagnostics share that instance. Authorization and
descendant-lineage validation complete before cache admission. Backend JSON, SSE, Studio transports,
`SequenceCursor`, Last-Event-ID resume, heartbeat emission, and terminal final draining continue to
consume the same `{events, next_seq, has_more}` page shape. Backend shutdown leaves the cache alone
because shutdown is a non-terminal operational stop and Studio ingress can still be active; bounded
capacity supplies the process-lifetime memory control.

`SequenceCursor` and `EventSubscription` turn that inclusive page API into a reusable next-sequence
subscription. A cursor advances only after an event is presented, suppresses replayed sequences,
and raises `EventSequenceGap` when a resumed stream skips required data. `RunnerBackend` exposes
`subscribe_events(...)` for live and recovered root runs and `subscribe_descendant_events(...)`
for lineage-authorized child streams.

The same HTTP events route returns SSE when the request accepts `text/event-stream`. Each event
frame carries `id: <seq>`; reconnects send `Last-Event-ID`, which takes precedence over the initial
`from_seq` query and resumes at the following sequence. Idle streams emit `: keep-alive` comments.
Terminal streams re-read the event page after observing terminal lifecycle state, verify the
lifecycle watermark has been drained, then emit one named `end` frame and close.

`GET /v1/runs/{root_run_id}/model-stream` exposes the optional Reference live-content broker as
SSE after validating the root run token. `Last-Event-ID` takes precedence over the initial
`cursor` query. Named `model-stream` frames use `monoid.model-stream.live.v1` and carry an
`id: <generation>:<sequence>` cursor. `opened`, `delta`, and `closed` frames multiplex the root and
all validated descendant run ids. A delta includes separate output/reasoning-channel UTF-8 byte
`start_offset` and `end_offset` values; clients use them to merge private snapshots with a replayed
suffix without duplicating content. A `closed` frame and the corresponding Studio hydration
snapshot carry the same optional boolean `retryable` signal, with absence interpreted as `false`.

Each root ring retains at most 1,024 frames and 512 KiB. The broker retains 64 root rings with
least-recently-used eviction. A missing sequence, stale/ahead cursor, generation replacement, or
oversized frame produces a `reset` with the retained baseline and latest cursor. The client hydrates
the missing prefix from an entitled private store, then resumes from that baseline. When no ring
remains, the reset baseline uses a root-bound idle cursor carrying the broker's bounded eviction
epoch; reconnecting with the current cursor waits without repeating hydration until a new ring
generation appears. The broker retains acknowledgement epochs for at most the root-ring budget;
an older forgotten acknowledgement resets conservatively to the current epoch. A cursor-free
reader resets for a recently evicted root and waits as pre-publication only for an unseen root.
Closing the SSE subscription removes only that reader. It never interrupts or cancels model
execution. Backend
shutdown closes the broker, wakes blocking subscribers, and makes later observer writes inert.

Studio composes this into `/api/model-stream` and exposes a separate authorized
`/api/model-content?run_id=<root>` snapshot route for reset hydration. The snapshot response uses
`studio.model-content.v1`, verifies every root/descendant sidecar context against its run directory,
and returns output/reasoning text with their UTF-8 end offsets. Studio flushes active in-process
sidecar batches before reading the snapshot, so the snapshot covers the broker prefix represented
by a reset cursor. Sidecar access requires a regular, single-link file whose path and open file
descriptor keep the same identity across each read, write, and flush. An unsafe or unavailable
sidecar, including an active-flush failure, returns HTTP 503; the browser retries hydration without
advancing its broker cursor. Both routes reject content requests with HTTP 403 when Studio's
live/private egress gate is disabled. Their frames and snapshots stay out of the durable event
reducer and Trace.

Studio removes a retryable failed partial only after a newer root stream is accepted. The manual
retry endpoint returns the exact failed `turn_id`, and the session driver durably emits
the existing `run.resumed` v1 shape with `reason="studio-retry"` and that envelope-level `turn_id`
before starting the replacement turn. The chat projection joins a non-retryable `turn.failed` to
the exact root/run/turn private sidecar snapshot and stores its available output beside the error.
This keeps an earlier failed prefix after a normal new turn replaces the bounded live-hydration
snapshot. The projection and browser suppress only the exact turn named by an explicit Studio
Retry, so a lost HTTP response, reload, retained live frames, or hydration cannot restore that
abandoned prefix. If recovery reused a turn id and left more than one eligible private stream, the
projection omits the ambiguous partial rather than attaching content to the wrong terminal event.
Run-terminal calls stay the latest bounded hydration snapshot because no later turn can replace
them. Interrupted partials remain visible.

### Diagnostics

`GET /v1/runs/{run_id}/diagnostics?event_limit=N` returns one token-scoped operational aggregate:
`status` (the run lifecycle payload with `state` and `terminal`), `failure` (`failure.json` when present), `recovery` attempt state, bounded recent event
summaries, control-command audit summaries, and trace ids found in recent events. Diagnostics uses
event summaries rather than raw event payloads so model text, tool arguments, bearer tokens, and
lease material do not get a new broad read surface.
### Inbox Message Envelope

`monoid.inbox-message.v1` (`core/inbox.py`, `InboxMessage`) wraps a message entering a
run so it carries **provenance** and an idempotency key. Like the control protocol it is an
edge/transport contract — the reference `RunnerBackend` wraps inbound content into it; the engine
(`AgentLoop`) never sees the envelope (it still receives unwrapped `content` via `submit`).

- Fields (CloudEvents-shaped): `id` (the dedup key), `source`, `type`, `run_id`, `created_at`,
  `correlation_id` (defaults to `id` — a flow root), `causation_id`, `traceparent`/`tracestate`,
  `content` (the JSON-native payload: a `str` or a list of
  content-part dicts), `metadata`. `is_inbox_envelope(obj)` discriminates an envelope from a legacy
  raw `str`/`list` queue entry.
- **Idempotent ingress**: `RunnerBackend.send_message(..., message_id=, source=, correlation_id=,
  traceparent=, tracestate=)`
  wraps + enqueues the envelope. A caller-supplied `message_id` makes the send idempotent — an
  already-processed id short-circuits to `status="duplicate"`, and a redelivery still in flight is
  dropped at dequeue. Processed ids are tracked per-run and **checkpointed** (`RunCheckpoint
  .inbox_seen_ids`), so dedup survives a restart (the marker rides the same checkpoint as the
  message's effects). Absent an id the edge
  mints one. HTTP `POST /v1/runs/{id}/messages` accepts optional `message_id`/`source`/
  `correlation_id`; a control `send_message` uses the command's `command_id` as the dedup key.
- Back-compat: the queue/checkpoint carry envelopes (JSON dicts), but legacy raw `str`/`list`
  entries from older checkpoints still restore and process.
- **Symmetric dedup on result ingestion**: `TaskManager.report_result` (the hosted-task result
  callback) is idempotent the same way — **first report wins**. A duplicate report (a callback
  retry) is a safe no-op that neither clobbers the recorded result nor re-publishes to the reentry
  queue (which would make the agent observe the result twice). The dedup signal is the
  already-persisted+rehydrated `ready_for_reentry`/`finished_at` job state, so it holds across a
  restart with no extra bookkeeping; the result dict carries a `duplicate` flag.

### Outbox Request

`monoid.outbox-request.v1` (`core/outbox.py`, `OutboxRequest`): a tool **stages** an
external side-effect (send an email, call a webhook) durably in the per-run `Outbox` instead of doing
the IO inline. The request is checkpointed, so it survives a restart; the engine never performs the
send.

- A tool handler calls `ToolContext.emit_outbox(destination, payload, *, capability,
  idempotency_key="")`; the request is appended to the per-run `Outbox` (checkpointed in full as
  `RunCheckpoint.outbox_requests`) and `outbox.requested` is emitted. The request carries the
  capability lease **handle** (`token_ref`, captured via `capability_token(capability)`) — never a
  secret. Bind the outbox tool with `runtime.requires_lease` so the existing capability gate
  brokers/revokes the lease *before* the send is staged (least-privilege egress).
- **Edge drains, effectively-once**: `RunnerBackend(outbox_sender_factory=lambda request: ...)`
  supplies an `OutboxSender` (`send(request) -> OutboxReceipt`); the backend drains
  `loop.pending_outbox()` at each park/settle, performing the IO (resolving `token_ref` to the real
  credential) and recording the outcome via `loop.record_outbox_result(...)` → `outbox.dispatched` /
  `outbox.failed`. The request is persisted `pending` before the send and `dispatched` after; a
  crash in between re-dispatches on recover, made safe by the `idempotency_key` the external target
  honors. A retryable failure stays `pending` and redrives up to `outbox_max_attempts`, then
  dead-letters as `failed`. No sender → requests stay durably `pending`.
- **Backoff + redrive (retry decoupled from run activity)**: a retryable failure stamps a durable
  `next_attempt_at` on the request — capped exponential backoff with **full jitter**
  (`uniform(0, ceiling)`, the ceiling being `outbox_retry_base_s * outbox_retry_factor**attempts`
  bounded by `outbox_retry_cap_s`). The cap binds the **exponent**, not only the product it
  multiplies out to, and the schedule is the model-retry loops' own
  (`providers/_common.capped_backoff`): these four fields carry no validation, computing the power
  first lets it leave the float range before the cap is ever consulted, and the schedule is
  evaluated *after* `sender.send` returns — so an arithmetic error there would lose the receipt for
  a side effect that already happened. Hence one rule, applied ahead of every shortcut that reasons
  about the base or the cap: **growth the schedule cannot resolve resolves upward, to the cap** — a
  `NaN` factor (which raises rather than orders), an infinite one, and a zero base under either
  (`0 * inf` is `nan`, not zero). The other end of that range is a zero ceiling, `uniform(0, 0)`,
  an unthrottled resend against the endpoint that just refused. A **cap** the schedule cannot
  resolve goes the same direction, to the largest representable wait (`sys.float_info.max`), and it
  is settled ahead of every other arm because each of them reads `cap_s`: `NaN` and `-inf` land on
  exactly that zero ceiling, while `+inf` stamps a `next_attempt_at` that `due_outbox` never
  selects (`next_attempt_at <= now` is False for `inf`) and that `OutboxRequest.to_json` refuses
  (`parse_float` — finite only), in the checkpoint export that runs *after* the send. An unlimited
  cap stays unlimited: for every product the schedule can represent, `min(inf, product)` and
  `min(float_info.max, product)` are the same number. The drain only
  dispatches **due** requests (`loop.due_outbox(now)`; a freshly staged one has `next_attempt_at=0.0`
  → due immediately, so the happy path is unchanged), and because the schedule is on the checkpoint
  it survives a restart. The backend's **watchdog tick** also runs `_redrive_outbox()`: for each live
  run it marshals the drain onto the shared loop, so a due request is redispatched even while its run
  sits idle (redrive requires the watchdog running — the backend's operational background loop). The
  loop stays policy-free: the edge computes `next_attempt_at` and passes it to
  `record_outbox_result(...)`.
- Reference `reference/outbox.py`: `RecordingOutboxSender` (dev/tests), `FailingOutboxSender`
  (retry-path tests), and an `OutboxToolProvider` yielding a generic `outbox.send` tool.
- A request also carries `traceparent`/`tracestate` and
  `correlation_id`/`causation_id` (the request↔result link reused by ack-back). Per-destination
  routing is deferred.
- **Ack-back (request-reply, non-park)**: stage with `emit_outbox(..., expect_ack=True)` (the
  `outbox.send` tool exposes `expect_ack`/`reply_to`). When the send reaches a terminal outcome
  (`dispatched`/`failed`) the edge delivers the receipt **back to the run as an inbox message**
  (`type="outbox_ack"`, `correlation_id` = the request's flow, `causation_id` = the request id,
  carrying its `traceparent`) via the idempotent inbox path with a stable id (`ack_<request id>`) so a
  redelivery is a no-op. The agent observes it on its **next activation — it never parks**; a
  terminal run has no consumer, so the ack is dropped (documented). `reply_to` empty = the run's own
  inbox. Park-and-await (the agent suspending until the reply lands) is a deferred superset that
  reuses this same ack plumbing.

### External Agent Envelope

`monoid.external-agent-envelope.v1` (`core/external_agent_envelope.py`) gives peer-agent messages
a transport-neutral helper shape above the inbox/outbox primitives. It carries the minimum meaning
an edge preserves when one agent sends work or a reply to another agent.

- Required Phase 2 meaning: `peer_id`, `message_id`, `correlation_id`, `causation_id`,
  `traceparent`/`tracestate`, ordered text/data `parts`, and retryable delivery state.
- `message_id` is the dedupe key. A receiving backend maps it to `InboxMessage.id`, so redelivery
  is processed once and the processed id survives restart through `RunCheckpoint.inbox_seen_ids`.
- `parts` are ordered text/data records for the Phase 2 contract. Rich artifacts, terminal result
  payloads, and full A2A task lifecycle mapping remain extension points.
- Raw bearer secrets stay outside envelopes, checkpoints, diagnostics, and public event payloads.
- Helpers: `external_agent_envelope_from_outbox_request`,
  `external_agent_envelope_to_inbox_message`, `validate_external_agent_envelope`, and
  `normalize_external_agent_error`. Import them from `monoid_agent_kernel.core.external_agent_envelope`.
- Reference `InboxRoutingOutboxSender` adapts `OutboxRequest` to `ExternalAgentEnvelope` and routes
  it into a peer run's idempotent inbox. This is the Reference message-fabric adapter.

### Trace Context on envelopes (`traceparent` / `tracestate`)

Inbox, outbox, and external-agent envelopes carry optional W3C Trace Context (`core/trace_context.py`): `traceparent`
(`00-{trace-id}-{span-id}-{flags}`) and the opaque vendor `tracestate`. This is **observability
only** — it complements `correlation_id`/`causation_id` (the domain identity routing and
reply-matching depend on) and **application behavior never depends on it**; a missing or malformed
header is ignored.

- Helpers: `new_traceparent()` (fresh root), `child_traceparent(parent)` (same trace-id, new
  span-id), `parse_traceparent(s)` (validates shape, rejects all-zero ids, returns `None` on
  garbage), `trace_id_of(s)`.
- **Inbox (ingress)**: `send_message(..., traceparent=, tracestate=)` propagates an inbound trace
  onto the envelope. The engine unwraps the envelope before `submit`, so an outbox request can't
  auto-inherit the *causing* inbox message's trace inside the core — a fresh root is minted instead
  (cross-loop inheritance is a later edge enhancement).
- **Outbox (egress)**: `emit_outbox` stamps a fresh root `traceparent` at staging (pure, no IO) so
  the request is traced from birth; the edge sender derives a `child_traceparent` for the actual
  outbound call. The trace rides the `outbox.requested`/`outbox.dispatched`/`outbox.failed` events so
  the OTel event-sink mapper can stitch spans across a restart.

### Capability Request / Lease

Secrets stay outside the core. When a tool needs external access it carries a *capability*
requirement, and the loop acquires a scoped, expiring **lease** from a broker before running it.

- `CapabilityRequest` (`...capability-request.v1`) / `CapabilityLease` (`...capability-lease.v1`) /
  `CapabilityDenial` are plain data. A lease carries a `token_ref` **handle, never the secret** —
  the gateway/tool edge resolves it, not the core.
- `CapabilityBroker.request(req) -> CapabilityLease | CapabilityDenial` is the seam an integrator
  (Daemon/Cell) implements. `AutoGrantBroker` is the zero-config dev default; the reference
  `GatewayCapabilityBroker` mints a scoped gateway token as the lease handle (the "absorb the
  gateway" path); `DenyAllBroker` is the safe default.
- **Implicit, binding-declared**: a `ToolBinding` with `runtime.requires_lease` declares its tool's
  `capability` needs a lease; the agent just calls the tool. `AgentLoop(capability_broker=...)`
  gates the call: a cache miss requests a lease (scoped to the binding) and on grant proceeds; a
  denial raises so the call never runs and the model gets an actionable error. If no broker is
  configured, a required lease fails closed with `capability_broker_required`. For local development
  only, `runtime.requires_lease="optional"` preserves best-effort gating and lets the tool run
  without a broker. Events `capability.requested` / `capability.granted` / `capability.denied` give
  the audit trail.
- **Using the lease**: the granted handle reaches the running tool via
  `ToolContext.capability_token(capability) -> token_ref | None` (the handle, resolved at the
  edge). The reference backend provisions a per-run broker with
  `RunnerBackend(capability_broker_factory=lambda request: ...)` — scoped to the run's identity
  (e.g. a `GatewayCapabilityBroker` per tenant). `None` is only safe for bindings without required
  leases, or bindings that explicitly opt into `runtime.requires_lease="optional"`.
- **Security invariants the core enforces**: a grant may only NARROW the requested scope, never
  widen it (`CapabilityVault.admit` is fail-closed); a lease is expiry-checked before reuse; the
  per-run vault holds handles only and durable (approved) leases are checkpointed as handles, while
  ephemeral sync grants are re-brokered on restart. Any `CapabilityBroker` can be verified against
  these invariants with the parametrized `tests/test_capability_broker_contract.py` suite.
- **CLI**: `monoid run --auto-grant-capabilities` wires the built-in `AutoGrantBroker` (local
  dev), or `--capability-broker path.py:factory` loads a custom broker (`factory()` returns it).
- **Async approval (escalation)**: a broker may return `CapabilityPending` instead of granting
  synchronously — the loop then parks the run on a `capability` hosted-task (carrying the request
  AND the gated call) and hands the model a "pending" observation; when the grant is reported
  (`report_task_result` with a `lease`), the lease is admitted to the vault (fail-closed against the
  original request scope). `HumanEscalationBroker` (reference) escalates every request; a real
  policy broker auto-grants low-risk capabilities, denies forbidden ones, and escalates only the
  sensitive ones (the three-way `lease`/`denial`/`pending` outcome is the point).
- **Auto-redispatch** (`AgentLoop.capability_auto_redispatch`, default on): after the grant the loop
  re-executes the gated call automatically at the next step (through the normal tool path, real
  permission/quota/events) and delivers the result to the model — no model retry needed. If a replay
  can't run cleanly (no valid lease), it falls back to model-retry. The gated tool never executed at
  the gate, so failure-free replay executes it once. Capability replay recovery is at-least-once: a
  process loss before the next ordinary checkpoint restores the durable replay batch. Effectful
  handlers therefore require a stable idempotency policy at their external boundary.
- **Durable leases**: an escalation-approved lease is marked `durable` and checkpointed (the
  `token_ref` handle only, never a secret), so a restart does not re-prompt the approver; ephemeral
  sync grants are not persisted (re-brokered on restart). The gated call is captured in the durable
  hosted-task so auto-redispatch survives a restart too.
- **Revocation** (the operator/Daemon kill switch): `revoke_capability` (a Control command, or
  `AgentLoop.revoke_capability(...)`) records a revocation in the per-run vault; `get_valid` /
  `token_for` then refuse the handle **fail-closed**. Three granularities, one mechanism: per
  `capability` (authoritative — the gate refuses to even *re-broker*, so a permissive broker can't
  resurrect it), per `lease_id`, and an issued-before `before` watermark (a bulk cohort kill). Because
  a lease is only a handle the tool re-fetches per call, revocation just refuses to hand the handle
  back — instant, with no distributed secret clawback. Revocation state is checkpointed so a
  revoked capability stays dead across a restart. Emits `capability.revoked`. The shared
  `TokenManager` also supports gateway-edge revocation by token id (`jti`) or issued-before
  watermark when the deployment propagates that revocation state to the gateway verifier.
- **Rotation** (`AgentLoop.capability_rotate_skew_seconds`, default `0.0` = off): a cached lease
  within `skew` seconds of expiry is re-brokered on use — the handle/expiry refresh under a stable
  contract without a model retry or a re-prompt. Bounded by `CapabilityLease.max_expires_at`, an
  absolute ceiling so a one-time human approval is never silently auto-extended forever; past the
  ceiling the lease is left to expire (then the normal re-broker / re-escalation path applies). A
  deny/pending/scope-widening rotation leaves the still-valid current lease untouched (no
  in-flight disruption). Emits `capability.rotated`.
- **Web tools through the gate (opt-in)**: the built-in `web.search` / `web.fetch` / `web.context`
  tools declare a `capability`; set `runtime.requires_lease` on their binding and the existing gate
  brokers a lease before each call. The lease handle becomes the request's `Authorization` (threaded
  context → `WebService` → `WebGatewayClient` as a per-call credential override; absent a lease, the
  client uses its static run-start token — back-compat). The reference `GatewayCapabilityBroker`
  mints a **web-gateway-compatible** token (`kind=web_gateway`/`aud=csp.web-gateway`) for `web.*` so
  the existing web gateway accepts it unchanged. Brokered web tokens carry a signed `metadata.scope`
  containing the binding id, domain scope, and web runtime caps such as `max_calls`; the web gateway
  applies that signed scope before provider invocation. Payload constraints can narrow the signed
  scope, and requests that widen `allowed_domains`, `binding_id`, or numeric caps fail with
  `web_scope_denied`. Net effect: web access inherits rotation + revocation (an operator can
  `revoke_capability("web.search")` to kill a live run's web access without cancelling it). The LLM
  path is deliberately NOT routed this way.
- **Gateway model-token refresh** (separate from capabilities): `GatewayModelAdapter.token_provider`
  is an optional per-request token source; the reference backend wires a source that re-mints the
  `llm_gateway` token near expiry, so a run outliving the token TTL keeps LLM access without a
  restart. Default (no provider) is the static token. This is a refresh seam, not capability routing
  — the LLM hot path stays out of the broker.

### Permission Boundary

`PermissionPolicy` remains the workspace/public-output boundary:

- `deny_patterns` block workspace path access
- `redact_patterns` mask public events and projections

It does not grant tools. Tool availability and execution constraints come from
bindings.

## HTTP Contracts

### LLM Gateway

`GatewayModelAdapter` sends `POST <gateway-url>`.

```json
{
  "protocol": "monoid.llm-turn.v1",
  "model": "gpt-5.5",
  "system_prompt": "...",
  "tools": [
    {
      "id": "read_notes",
      "name": "read_notes",
      "description": "...",
      "input_schema": {},
      "capability": "fs.read",
      "side_effect": "read"
    }
  ],
  "reasoning": {"effort": "medium", "summary": "off"},
  "generation": {"temperature": 0.2, "top_p": 0.9, "max_output_tokens": 256},
  "output_schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
  "instruction": "First turn text"
}
```

`generation` and `output_schema` are additive and present only when configured — traffic that
configures neither keeps its exact pre-existing request shape, and the protocol identifier is
unchanged. Both `generation` and `reasoning` carry their `on_unsupported` when it is
off-default: the server rebuilds a config object from the block, so a field left off is not
"unset" there but the *default*, and a caller's `"omit"` would come back as `"fail"` on the
server's copy — which the next hop then enforces when a gateway's upstream is another gateway.
For the same reconstruction reason `reasoning` carries `effort` explicitly when it is
`"default"`: that is the one field whose omission sentinel differs from the codec's
reconstruction default (`"medium"`), so leaving it off silently asked the server's upstream
for medium reasoning on a call that asked for the provider default. Every other value keeps
its pre-existing wire bytes, and digests never read this wire.
The applied-echo comparisons are unaffected: they are built from `build_generation_payload`
and `build_reasoning_payload`, which carry provider knobs only, never policy. The
server parses both blocks with the kernel's own fail-closed codecs: an out-of-range or
out-of-enum value answers 400 `gateway_bad_request` at this boundary instead of travelling to
the upstream provider.

The kernel sends one of two request styles. **By-value `messages` is the default**: the
full provider-neutral conversation log (`messages`, a list of `{role, content}` user /
assistant / tool entries) travels on every turn, and the gateway forwards it statelessly —
`previous_turn_handle` and `observations` are not consulted. The conversation is
reconstructed from the checkpoint rather than a server-side handle, so this style survives a
restart.

The **handle-based** style (shown in the example above) is the fallback, used when a turn
carries no `messages`. It has three shapes, selected by `previous_turn_handle` and
`instruction`:

- **first turn** — no `previous_turn_handle`; carries `instruction`.
- **tool continuation** — `previous_turn_handle` + `observations`; no `instruction`.
- **user follow-up** — `previous_turn_handle` + `instruction` (a new user message on
  top of an existing continuation handle; `observations` is empty).

Either style lets one run accept multiple user turns: with `messages` the new user message
is appended to the log; with a handle the kernel threads the last `turn_handle` into the
next user message.

The handle-based style only works when the gateway's **upstream** persists its responses, since
the server translates the opaque `turn_handle` into the upstream's own response id. The reference
gateway's default upstream is `OpenAIModelAdapter`, which does not (`store=False`), so against
that default a handle-based continuation is refused upstream and answered as `422` with
`unsupported_request_shape` — a classified bad request the client survives, where it previously
travelled on to become an opaque provider 404. Configure a `provider_adapter_factory` whose
upstream keeps responses to use this style, or send `messages`.

Successful response:

```json
{
  "protocol": "monoid.llm-turn-result.v1",
  "turn_handle": "turn_...",
  "final_text": null,
  "tool_calls": [
    {"call_id": "call_1", "name": "read_notes", "arguments": {"path": "notes.md"}}
  ],
  "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
  "reasoning": [{"type": "reasoning", "id": "rs_...", "encrypted_content": "..."}],
  "provider": "openai"
}
```

`reasoning` carries the upstream provider's own reasoning artifacts, relayed verbatim: the
provider's output-item subsequence, adjacency-preserving — the `reasoning` items **plus the
`function_call` or `message` items they are paired with**, because the provider validates that
pairing when the subsequence is sent back. Only the `reasoning`-type entries are encrypted; a
`message` entry carries the model's plaintext answer text and a `function_call` entry carries
plaintext tool arguments, both of which also appear in `final_text` / `tool_calls` on the same
envelope. Nothing here is interpreted by the gateway or the kernel, but **a redaction or logging
policy must treat this array as model content**, not as an opaque blob: it duplicates content the
rest of the envelope already bounds, so a surface that truncates `final_text` and dumps
`reasoning` raw has not truncated anything. Sizewise it roughly doubles a small body when
populated, and it is the payload the kernel prunes off historical turns (see the replay-window
rule above).

Note that `reasoning` names two different things on the two halves of this protocol, and they
are not the same shape: on the **request** it is the reasoning *config* object
(`{"effort": ..., "summary": ...}`, documented above), and on the **response** it is this
artifact *array*. A server that echoes request keys back onto its response body therefore
answers an array-valued key with an object, which the response reader refuses as
`gateway_bad_response`. A third `reasoning*` spelling lives beside them since v0.21:
`reasoning_applied`, the response-side applied-parameters *echo* (an object — the forwarded
config projection, documented with the other echoes below), which shares a prefix with the
artifact array and nothing else. Keep the three apart when implementing: config object on the
request, artifact array and echo object on the response.

The artifacts ride the response body and
the terminal `turn_complete` stream frame — the same two writers, built from one function — and
exist so the provider-native reasoning round-trip survives this hop: the kernel captures them,
tags them with the relaying adapter's `provider_name` plus the model, carries them in the
by-value `messages` log, and the upstream adapter replays them on the next turn. The key is
present **only when the upstream produced artifacts**, which makes it conditional on the *answer*
rather than on the request (unlike the applied echoes below). It is additive and ignorable:
absence reads as "no artifacts", which is what an older gateway, a non-reasoning upstream, and a
stream that ended without a terminal frame all honestly mean, and a run that reconstructs none
simply replays none.

`provider` names the upstream those artifacts came from, and rides the same two writers. Only the
upstream adapter's own `provider_name` declaration is written here — never the gateway's own
`ModelConfig.provider`, which is a hop-local constant (`"openai"` for every call this reference
gateway serves) and would therefore *name* OpenAI for an upstream that is not. An upstream that
declares nothing is unknown, and unknown is written by **omission**, which is a third
conditionality beside the request-conditional echoes and the answer-conditional `reasoning`.

The key exists because the client's own declaration is a guess: `GatewayModelAdapter.provider_name`
defaults to `"openai"` — correct for this gateway's default upstream and wrong for any deployment
whose `provider_adapter_factory` routes elsewhere without setting it — and it is what the captured
artifacts get tagged with. A client reading `provider` **verifies** against it: on a mismatch it
drops that turn's artifacts (they are unusable under either name) and changes nothing else, keeping
its own declaration on the reasoning tag and on every observability surface. Absence gates nothing,
so an older gateway and an undeclared upstream behave exactly as before the key existed.

`usage` always carries `input_tokens` / `output_tokens` / `total_tokens`. It MAY
additionally carry optional priced sub-counts when the provider reports them —
`cache_read_tokens`, `cache_creation_tokens`, `reasoning_tokens`, `audio_tokens` —
which the kernel sums into per-run totals and checks against the token budget. These
fields are additive; a consumer that ignores them stays correct.

The sub-counts travel the whole reporting chain, not just the totals: `metrics.updated`
publishes each one it has (omitting the ones the adapter did not report), both tenant ledgers
(the gateway's and the reference backend's) sum them as their own columns, and a subagent's
sub-counts roll up into its parent through the same normalized vocabulary. `total_tokens` is
still whatever the provider reported and is never re-derived from the sub-counts, so a call
priced *only* in sub-counts reports `total_tokens: 0` and is visible in the columns it was
actually expressed in. The token BUDGET is unchanged — it reads the three headline counts.

Error response (the non-200 body, and — minus the `type` tag — the terminal SSE `error` frame,
which is written from the same definition so the two transports cannot drift):

```json
{
  "error": "upstream refused an unproven turn",
  "error_code": "gateway_generation_not_applied",
  "retryable": false,
  "config_recoverable": true,
  "http_status": 422,
  "provider_retried": false,
  "usage": {"input_tokens": 120, "output_tokens": 40, "total_tokens": 160}
}
```

`error_code` is the **provider** code (the kernel-level `ModelAdapterError.error_code` has no
wire slot and reconstructs to its class default). `retryable` forecasts a future attempt;
`config_recoverable` says the remedy is the caller's configuration instead, so the two are
independent and a client that reads only the status has to guess at the second. `provider_retried`
records attempts the gateway's own backend already made. All three booleans and `http_status` are
written unconditionally, so absence means "an older gateway" and reads as the default; `usage` is
the one omitted-when-empty key, because an error raised before a provider was reached costs
nothing and keeps its pre-`usage` wire shape.

**Applied echoes.** When the request carried a `generation` block **and the upstream adapter
declared `generation_support = "native"` for this call's config**, the response body and the
terminal `turn_complete` stream frame echo `generation_applied` — the exact block the gateway
forwarded upstream; a non-declaring upstream gets **no** generation echo at all, which a
fail-closed client refuses (emitting one unconditionally is the copied-back-proof defect the
declaration gate exists to rule out). When the request carried `output_schema`, they echo
`schema_applied` (boolean; `true` only when the upstream adapter declared native schema
enforcement, so a forwarded-but-ignored schema honestly reads `false`). When the request
carried a **non-default** reasoning config and the upstream adapter declared
`reasoning_support = "native"`, they echo `reasoning_applied` — the forwarded reasoning
projection, in the generation echo's object shape; because `effort="default"` forwards an
empty projection, the reasoning echo may legitimately be `{}`, and a client must treat that
empty object as a proof, not an absence. All proofs are probed
under the **per-call** config the turn runs under, not the adapter's standing one. All three
echo keys are additive and absent for requests that configured none. The client refuses an
unproven turn under the default `on_unsupported="fail"` — `generation.on_unsupported`
governing the generation and schema proofs, `reasoning.on_unsupported` governing the
reasoning proof — with non-retryable, config-recoverable `gateway_generation_not_applied` /
`gateway_schema_not_applied` / `gateway_reasoning_not_applied`, on the
sync response, on the streamed terminal frame, and on a stream that ends without one;
`"omit"` accepts best-effort transport. An older server that ignores the new request keys
sends no echo and therefore fails closed at the client rather than silently misapplying.

The reference gateway tokens authenticate run identity. New tokens include a `kid` header. A
`TokenManager` can be built from a keyring, rotated to a new active key, and configured to accept
retired keys only until a grace-window deadline. Verification also rejects revoked token ids and
issued-before cohorts before any gateway action proceeds. The LLM request model still selects the
turn model.

### Web Gateway

The kernel calls:

- `POST /internal/web/search`
- `POST /internal/web/fetch`
- `POST /internal/web/context`

Every request includes binding constraints:

```json
{
  "protocol": "monoid.web-search.v1",
  "binding_id": "search_docs",
  "query": "monoid runtime config",
  "max_results": 5,
  "max_calls": 20,
  "allowed_domains": ["docs.example.test"],
  "blocked_domains": []
}
```

The reference gateway enforces per-run/binding call counters and signed token scope. Brokered web
tokens carry `metadata.scope`; payload domain, binding, and numeric limit constraints can narrow
that scope and cannot widen it. Scope violations fail before the provider adapter is called.

### Reference Backend

Run creation requires `agent_definition` or `runtime_config`.

`POST /v1/runs` request shape:

```json
{
  "tenant_id": "tenant_a",
  "user_id": "user_a",
  "workspace_root": "/workspaces/demo",
  "instruction": "Read notes.md and create SUMMARY.md.",
  "mode": "propose",
  "runtime_config": {
    "definition_id": "coding-agent",
    "config_version": 1,
    "model": {"provider": "gateway", "model": "gpt-5.5"},
    "prompt": {"runtime_segments": ["Prefer concise edits."]},
    "tools": [
      {"binding_id": "read_file", "ref": {"kind": "registry", "tool_id": "fs.read"}},
      {"binding_id": "finish", "ref": {"kind": "registry", "tool_id": "run.finish"}}
    ],
    "tool_search": {"enabled": true, "top_k": 5}
  }
}
```

Runtime config API:

- `GET /v1/runs/{run_id}/runtime-config`
- `POST /v1/runs/{run_id}/runtime-config`

Replacement request:

```json
{
  "expected_version": 1,
  "issuer": "backend",
  "reason": "update guidance",
  "config": {
    "definition_id": "coding-agent",
    "config_version": 2,
    "model": {"provider": "gateway", "model": "gpt-5.5"},
    "tools": [
      {
        "binding_id": "read_file",
        "ref": {"kind": "registry", "tool_id": "fs.read"},
        "guidance": {"summary": "Read the smallest relevant file first."}
      },
      {"binding_id": "finish", "ref": {"kind": "registry", "tool_id": "run.finish"}}
    ]
  }
}
```

The backend validates schema, registry resolvability, duplicate binding ids,
and duplicate model names. A version mismatch returns HTTP 400.

### Multi-turn Sessions And Tasks

The run loop is suspend-return at its core: `AgentLoop.run_until_suspended()` runs
a turn and hands control back when the run settles (awaiting the next user
message), parks on a hosted task, or hits a limit. `submit()` is the blocking
wrapper over it; the reference backend's worker uses the non-blocking form to
drive multi-turn sessions.

Set `"multi_turn": true` on the run-creation request to keep the session open
after the first turn settles (default `false` closes after one turn). While open,
the run alternates between `running` and `awaiting_input` (a new
`run.awaiting_input` event with `reason` `"user"` or `"task"`). HTTP surface:

- `POST /v1/runs/{run_id}/messages` — deliver a follow-up user message (run token).
  It is queued and consumed as the next user turn when the current one settles.
- `POST /v1/runs/{run_id}/tasks` — create a hosted task (`{"kind": "hitl" |
  "automation", "request": {...}}`). Returns `task_id` plus a scoped
  `callback_token` and `callback_url`.
- `POST /v1/runs/{run_id}/tasks/{task_id}/result` — deliver a task result
  (`{"result": {...}, "status": "answered"}`). Authenticated by the per-task
  callback token (scoped to this run+task) or the run token (operator). Reporting
  a result wakes a parked run; the result is injected per the kind's
  `ResultInjector` (a user message for hitl, an async tool result for automation).

Follow-up user messages and task results are separate channels (a message is a new
user turn; a task result completes a specific task), mirroring the
add-message-vs-submit-tool-outputs split in comparable agent servers. Session
length is bounded by idle timeout, max lifetime, and max turns.

### Durable Persistence

A checkpoint is a **complete, self-contained "save file."** A parked run survives a
process restart even when the agent's workspace is *not* durable: workspace, the
conversation, and run state all roll back to one aligned instant. This is a
**state snapshot at clean checkpointable recovery boundaries**. Recovery reads snapshots directly. An
activation can publish internal safety checkpoints before returning its observable suspension
boundary. Restore continues from the latest committed snapshot; work before that snapshot is
already represented in it. A crash inside an activation can redeliver effects that completed
before a later checkpoint commit.
`OR-12-DURABLE-SIDE-EFFECT` requires a stable idempotency key or durable outbox for those effects.

**Division of responsibility:** the core defines *what* a checkpoint contains
(`RunCheckpoint`) and how to `restore()` it; the integrator decides *how* it is
stored by implementing `CheckpointStore`. Checkpoint I/O crosses the explicit store seam.
Auto-recovery belongs to the integrator. On failure, Core surfaces a bundle and the last-good
checkpoint.

**Portable recovery semantics:** every committed snapshot advances a monotonic checkpoint
sequence. Each internal safety checkpoint for an admitted input records its identity, original
source sequence, and `running` phase. Such checkpoints can advance from source `N` to
`N+1 ... N+k-1` while retaining that same active-input identity. Recovery continues only that
input and rejects stale or competing inputs. The returned suspension boundary commits `N+k` with
the exact observable suspension
(`reason`, status, awaiting task ids, terminal/error and retry classification), the applied input
identity, a `completed` phase, and one immutable identity-bound receipt. Repeating an applied input
returns its own stored receipt and performs zero new model/tool drives, including after later inputs
have advanced the run. A terminal boundary therefore has one terminal receipt for that input.
These observable rules are portable. Process placement, liveness, fencing, and recovery
coordination are deployment policy outside the Core contract.
An ambiguous checkpoint write remains unacknowledged until exact canonical readback proves the
commit. A caller cannot convert an unknown write outcome into a terminal receipt or admit a
competing input.

- `AgentLoop.snapshot() -> RunCheckpoint | None` captures one safe recovery boundary:
  run state + counters + parked hosted tasks, the **workspace delta** (created/
  modified/deleted files; content travels as content-addressed blobs), the **by-value
  conversation** (`messages` — provider-neutral user/assistant/tool log, vendor-
  independent), and the **latest `runtime_config`**. It returns `None` — refusing —
  while a live in-process shell job is still running (a subprocess cannot cross a
  process boundary). The park itself is recorded as `last_suspension`, the durable
  observation of one `Suspension`: `reason`, `status`, `final_text`, `error`,
  `error_code`, `awaiting_task_ids`, `has_external`, and the full failure
  classification (`retryable`, `http_status`, `config_recoverable`,
  `provider_error_code`, `provider_retried`). `turn` is excluded by design — it is a
  projection artifact of local paths and metrics, and a recovery driver needs only the
  boundary facts to return the same park. `reason` and `status` are required and must be members
  of the park and durable-status vocabularies; every other key is optional on read, and an absent
  one takes the default, which is what a checkpoint written before that key existed meant. The
  payload is schema-checked at the recovery boundary (types, vocabularies) rather than at the
  reader, and the same schema applies to the copy stored under
  `applied_input_receipts[<input_id>].suspension`.
- `CheckpointStore` (protocol): `put(checkpoint, blobs)` commits **atomically** and
  flips a `LATEST` pointer last (a half-written checkpoint is never returned);
  `latest(run_id)`; `delete(run_id)`. `LocalFsCheckpointStore` is the default
  (`run_root/<id>/checkpoints/<seq>/manifest.json` + content-addressed `blobs/<sha>`);
  swap it for a mounted-volume path or an object-store/DB store. The loop advances a
  monotonic `seq` per committed snapshot and deletes checkpoints only on a *completed* run — a
  failed/limited run keeps its last-good checkpoint.
- `CheckedCheckpointStore` is the additive checked-read extension. Its
  `latest_checked(run_id)` result distinguishes `loaded`, `migrated`, `missing`,
  `corrupt`, and `unsupported_version`, including the observed schema and committed
  sequence when available. A loaded record binds its embedded run ID and sequence to the lookup
  key and committed pointer; structural or identity mismatches are `corrupt`.
  `load_latest_checked()` adapts legacy stores, so existing `CheckpointStore` implementations
  remain source-compatible.
- Durable readers use `core.durable_codec.DurableCodec`: artifact versions parse as
  `<namespace>.<family>.vN`, accepted older versions migrate through pure ordered
  `dict -> dict` steps, and writers always emit the canonical current `monoid.*`
  version. A migration or validation failure performs no write and leaves `LATEST`
  pointing at the prior committed checkpoint.
- `AgentLoop.restore(checkpoint, *, blobs=...)` reopens the run: no second
  `run.started`/manifest, parked hosted tasks re-registered (so `report_task_result`
  still wakes it), the **workspace delta re-applied** on top of a re-provisioned base
  (`blobs` is a `sha256 -> bytes` reader, e.g. the store's), the conversation and
  `runtime_config` restored, remaining duration carried forward (downtime does not
  count against `max_duration_s`), and any shell job left `running` on disk folded in
  as a failed observation.
- **Durable cancellation:** `cancellation_requested` is written into every snapshot whenever the
  run's cancellation token was cancelled, and `restore()` honors it *unconditionally* — if the
  restored loop has no `cancellation_token`, one is minted and cancelled, so the next boundary
  check raises `RunCancelled`. The flag is the request; the token is only the channel a boundary
  check reads it through. A recovery driver that rebuilds an `AgentLoop` without passing a token
  (the ordinary shape) therefore cannot silently un-cancel a run whose cancellation was durable.
  An embedder that deliberately re-runs a cancelled checkpoint clears
  `checkpoint.cancellation_requested` before restoring. On the Reference backend, `cancel_run`
  of a QUIESCENT (parked) run commits a fresh park checkpoint carrying the flag before the ack
  returns, so the acknowledged cancel survives a crash that lands before the terminal record; a
  cancel that arrives while a turn is stepping stays in-memory until the pump's boundary check
  writes its own terminal park (that residual window is the mid-turn crash exposure).
- The snapshot also carries the **output-validator repair state** (`output_retries` *and*
  `output_failure_history`, so a restored mid-repair run continues its attempt numbering and
  keeps `failures_by_validator` in `metrics.json`) and the **delegation roll-ups**
  (`subagent_count`, `subagent_usage`, `skill_activation_count`, `skills_activated`), which are
  restored onto the rebuilt tool context so `metrics.json` reports one epoch rather than
  pre-restart token totals beside post-restart subagent counts. The per-activation tool
  *service* counters (shell/web call counts) are not durable and restart at zero.
- `AgentLoop.release_parked()` closes recorder/sink and process-local loop resources after a
  durable boundary commit. It preserves the checkpoint and hosted-task state and emits no
  `run.finished`. A non-terminal boundary can be restored by the next activation in a fresh
  process. Terminal artifact finalization remains the caller's responsibility.
- **Failure bundle:** on failure the core writes `run_dir/failure.json`
  (`{error, error_code, provider_error_code, http_status, retryable, config_recoverable, type,
  last_good_seq, restore_hint}`) —
  fail loud, name the checkpoint to restore from. `http_status` is the provider status the
  `run.failed` event beside it carries, written as `null` when the failure never reached a
  provider. `retryable` / `config_recoverable` are that same event's classification, read from
  the same run state: `fail_recoverable` (and `close()` on an unrecovered park) promotes a
  classified `turn.failed` into this record, and the promotion keeps the classification —
  including across a restart, where the checkpoint's `last_suspension` is where it is read back
  from. No auto-recovery.

#### Reference operational scopes

- The legacy Reference backend writes `run_dir/run.json` and stores the same recovery descriptor in
  the configured `CheckpointStore`: identity, workspace, limits, policy, and the authoritative
  resolved runtime config. Runtime-config hot-swaps update both copies with
  `runtime_config_version`, `runtime_config_hash`, `runtime_config_issuer`,
  `runtime_config_reason`, and `runtime_config_committed_at`; recovery verifies the hash before
  rebuilding providers or gateway token sources. `metadata_generation` starts at one and advances
  on each update. Recovery selects the higher generation, repairs the stale copy, and classifies
  different payloads at the same generation as `corrupt`. Two legacy copies without a generation
  retain local-authority behavior. Its optional lease/watchdog profile allows a backend that never
  hosted the run to reclaim it from a shared lease/checkpoint store, read the shared descriptor
  when local `run.json` is absent, materialize a local copy, then resume. Checked metadata reads
  use the same five outcomes as checkpoints; corrupt or unsupported local metadata remains
  authoritative bad state. `recover_runs()`
  writes an actionable failure bundle for corrupt or unsupported durable state. It scans
  `run_root`; the active watchdog discovers cross-instance orphaned
  runs from the shared lease store. Recovery skips terminal checkpoints, failed runs, and runs
  whose durable status artifact records a terminal outcome — a run that CLOSED limited keeps a
  non-terminal park checkpoint (a live-limited park is resumable by design) and no failure.json,
  so `status.json`'s terminal reading (including legacy bare `status="limited"` dirs) is the
  marker that stops it from being re-driven on every pass. It then rebuilds
  each run (re-issuing gateway tokens from the signing key, **re-provisioning the base workspace**
  is the deployment's job), `restore()`s the loop with the store's blobs, re-enqueues durably-saved
  follow-up messages, and resumes. Both give-up paths (unrecoverable after
  `max_recover_attempts`; corrupt/unsupported durable state) also meter the run's
  checkpointed/projected spend into the tenant ledger through the same high-water seam the
  failure path uses — a run recovery abandons is still a run that billed.
- The experimental v0.19.2 DBOS activation-recovery profile has a narrow operational scope. One
  stable executor slot has one active process; a restart reuses the same executor identity and
  application version after the prior process terminates or is fenced. One private Reference host
  owns a captured process-global DBOS runtime. Its private control and run participants register
  distinct workflow families, preflight their queue names, and contribute one aggregate listener
  set before one launch. The host registers participant queue objects after launch, then opens
  shared admission. The participants also share host drain and shutdown. DBOS schedules and retries
  finite control-dispatch and run-resume activations. The configured `CheckpointStore` remains
  authoritative for checkpoint sequence, input deduplication, committed boundary receipts,
  suspension, and terminal meaning. The profile scope covers same-slot finite-activation recovery
  and private lifecycle composition. The Reference backend, terminal artifact finalization, HTTP,
  Studio, product routing, and arbitrary-host takeover sit outside this scope. Portable recovery
  semantics remain unchanged.

**Assumption (workspace):** the agent workspace is not durable; on restore the
deployment re-provisions the base (re-clone/re-mount) and the checkpoint re-applies
only the agent's delta (the delta always contains the agent's created/modified
files). For container durability, `run_root` (or the `CheckpointStore`) must point at
durable storage — a mounted volume needs no code change.

**Limitations (v2):** a mid-run `commit_checkpoint` re-baseline combined with delta-
restore is a documented follow-up (the common no-re-baseline case is covered).
Multimodal message parts (image/document) round-trip through the checkpoint, so a
resumed run re-forwards the media. The by-value `messages` in the checkpoint remain
the load-bearing record for *resuming* a run. Since v0.20, settle events carry a digest
instead of the model's text. In v0.20.1, an entitled reader resolves that digest from
`model-content.jsonl` first and falls back to `transcript.jsonl` for older or partially written
runs. Deleting both private content artifacts costs displayed answers as well as debuggability.

## Legacy Reference Production Hardening

This section applies to the legacy `RunnerBackend` lease/watchdog profile. The DBOS Reference
profile has a private host-owned runtime lifecycle and no lease/watchdog stack. Core recovery
requires explicit host orchestration.

### Failure surfacing & bounded recovery

- **Failure bundle on every failure.** Beyond the core's own `failure.json`, the reference
  backend's `_record_run_failure` also writes `run_dir/failure.json`
  (`monoid.failure.v1`: `error, error_code, http_status, retryable, config_recoverable, type,
  last_good_seq, restore_hint, failed_at`) — the durable mark is written *before* the in-memory
  terminal state, so a worker crash that bypassed the loop's own bundle still leaves a mark and
  a restart never resumes a crashed run into a loop. The status and the two classification flags
  are read off the failing exception by name and never coerced; the recovery-path writers
  (unrecoverable, invalid durable state) hold no provider verdict and leave them at
  `null` / `false`. It writes the terminal statement into `status.json` in the same breath —
  through the one shared quarantine writer, with this lane's `recorded_by_run_failure` marker
  and the failure's *own* `error_code` — because a driver failure has no live recorder and
  therefore no terminal event: without the artifact, every status surface kept serving the
  run's last park after a restart while `recover_runs` skipped its dir on `failure.json`. A
  failure bundle beside a non-terminal artifact is also honored reader-side
  (`lifecycle_from_status_artifact`): the pair reads `failed`/terminal even for dirs
  quarantined before this writer existed, while a *terminal* artifact (a genuine close) still
  wins over the bundle.
- **Bounded recovery.** `recover_runs()` logs (not swallows) a resume failure and tracks
  attempts in `run_dir/recover_attempts.json` (`{count}`); after the cap it writes a
  `failure.json` with `error_code="unrecoverable"`, so a poison checkpoint is permanently
  skipped instead of retried forever. Both give-up paths (unrecoverable after the cap;
  corrupt/unsupported durable state) also write the terminal statement into `status.json`
  through the same shared writer (`state="failed"`, `terminal=true`, the bundle's error pair,
  an empty classification, and a `given_up_by_recovery` marker) — without it, `status()`,
  `list_runs` and the offline projection kept answering the run's last park
  (`awaiting_input`, `terminal=false`) for a permanently dead run while `resume_run` refused
  it as unrecoverable. The bundle's `restore_hint` names the actual operator flow: delete
  `failure.json` to lift the quarantine, then `recover_runs` (or `resume_run`) restores the
  last good checkpoint — the quarantine markers (`given_up_by_recovery`,
  `recorded_by_run_failure`) are what keep the closed-run status guard from mistaking a
  quarantine statement for a close once the quarantine is lifted.
- **Typed resume refusals.** `attempt_resume` answers a `ResumeOutcome`, not a bool, and
  `resume_run` maps it: `resumed` — the caller now owns a live run; `closed` (the durable
  status artifact or a terminal checkpoint records the run's own end — e.g. a close while
  budget-limited, whose park checkpoint is non-terminal by design) refuses with the loop's
  own terminal vocabulary, `NativeAgentError(error_code="run_terminal")`, instead of a hint
  at a `failure.json` that does not exist; `already_live` (a concurrent resume won the
  atomic record claim — the double-click shape) answers the same already-live success shape
  as the record-exists branch, `resumed: false`, because the run *is* being resumed; only
  `failed` — a genuine non-resume — keeps the inspect-logs/failure.json error. `list_runs`'
  `recoverable` consults the same close-recording artifact fact through the same function
  (`core.projections.status_artifact_records_close`), so the listing never advertises a run
  `resume_run` will refuse as closed.

### Active watchdog / lease (legacy backend only)

- `RunnerBackend.start_watchdog()` / `stop_watchdog()` run an opt-in heartbeat thread (tick
  `watchdog_interval_s`, default 5s). For each owned live run it refreshes
  `run_dir/lease.json` (`worker_id`, `pid`, `heartbeat_at`, `lease_ttl_s`; default
  `lease_ttl_s=30`), and deletes the lease on terminal.
- It reclaims a run whose lease has gone stale (`heartbeat_at + lease_ttl_s < now`) and a
  crashed worker left behind: reclaim takes the lease via a compare-and-swap, so two
  backends racing the same run produce exactly one winner, then resumes via the
  `recover_runs()` path.
- Lease storage + the CAS are a pluggable **`LeaseStore`** (default `LocalFsLeaseStore`:
  `lease.json` + `file_lock(run_dir/.reclaim.lock)`); see *Pluggable durable stores*.

### CheckpointStore robustness invariants

`LocalFsCheckpointStore` (and any conforming store):

- **Monotonic `LATEST`:** the pointer is only advanced when `checkpoint.seq` exceeds the
  current `LATEST` seq — a late or lower-seq writer can never unpublish a newer committed
  checkpoint.
- **Orphan blob GC:** crash-leftover `blobs/*.tmp` files are cleaned on `put()`/`latest()`.
- **Cross-process serialization:** `put()` holds `file_lock(checkpoints/.put.lock)`
  (`core/_util.file_lock`, O_EXCL with stale-steal); `latest()` retries a read that races a
  concurrent commit's atomic replace, so a reader never mistakes mid-commit for "no
  checkpoint."

### Pluggable durable stores

The legacy backend uses two seams for pluggable durability and multi-node recovery:

- **`CheckpointStore`** (core) — `put(checkpoint, blobs)` / `latest` / `delete`.
  `CheckpointRecord.blob(sha)` is a callable, not a directory, so a store can back blobs with
  files, a DB, or an object store.
- **`LeaseStore`** (reference) — `candidate_run_ids` / `heartbeat` / `is_stale` / `try_claim`
  (atomic CAS) / `owner` / `release`. The watchdog policy stays in `RunnerBackend`; only the
  lease's storage and its claim atomicity live here.

Every store must pass the parametrized contract suites (`tests/test_checkpoint_store_contract.py`,
`tests/test_lease_store_contract.py`): atomic last-good commit, monotonic `latest`, write-once
blob dedup, run metadata round-trip, and a single-winner `try_claim`. Passing them makes a backend
a drop-in.

**Legacy SQLite reference stores** (`reference/stores/`, stdlib `sqlite3`, zero dependencies):
`SqliteCheckpointStore` and `SqliteLeaseStore`. A DB transaction supplies the invariants —
`put` commits atomically (a crash rolls back, so `latest` never sees a torn checkpoint), the
latest pointer advances monotonically via a conditional UPSERT, blobs are write-once, and
`try_claim` is a transactional CAS under `BEGIN IMMEDIATE`. `SqliteCheckpointStore` also stores the
backend run descriptor beside checkpoints, so one shared db can host **both** stores and the
recovery metadata needed to reclaim and resume a crashed peer's run across the instance boundary
(a per-host `lease.json` cannot):

```python
db = "/shared/monoid.db"
backend = RunnerBackend(
    ...,
    checkpoint_store=SqliteCheckpointStore(db),
    lease_store=SqliteLeaseStore(db),
)
```

**Legacy profile limitation / follow-up:** SQLite is single-host. A legacy cross-host deployment
can supply a networked `CheckpointStore` / `LeaseStore` behind these seams. This capability is a
Reference implementation option; the Core contract and the initial DBOS profile do not promise
automatic host takeover.

### HTTP hardening & request bounds

Shared in `reference/_shared/http_util.py`, applied to the backend / llm-gateway /
web-gateway HTTP layers:

- `read_json_limited(handler)` rejects a body whose `Content-Length` exceeds
  `MAX_REQUEST_BYTES` (10 MB) with **413** before reading — a DoS/OOM guard.
- `HardenedThreadingHTTPServer` sets a per-connection `REQUEST_TIMEOUT_S` (30s) socket
  timeout and shuts down cleanly (`daemon_threads=False`, `block_on_close=True`) so a slow
  client cannot pin a thread and in-flight handlers are not abandoned.
- `redact_internal_error(...)` logs an unmapped 5xx in full server-side under a
  `correlation_id` and returns only that id to the client (never a stack trace / path);
  intentional client-facing errors (`ValueError`/`PermissionDenied`/`KeyError`) keep their
  message. `log_http_request(...)` emits a structured access line.

### Resource & DoS bounds

- **`RunLimits`** (core): `max_messages` / `max_message_log_bytes` bound the by-value
  conversation; `max_workspace_delta_bytes` / `max_delta_file_bytes` bound a checkpoint's
  workspace delta. Exceeding a cap on **capture** settles the run `limited` (a safe stop,
  not a drop — the prior good checkpoint stays the recovery point); exceeding on **restore**
  refuses the checkpoint (`workspace_delta_bytes_exceeded` /
  `workspace_delta_file_bytes_exceeded`). Defaults are generous backstops.
- **Backend:** `max_message_bytes` (reject over-large follow-up message),
  `max_message_queue_depth` (cap pending-message queue), `max_concurrent_runs` (a bounded
  semaphore; excess submissions stay `queued`, `0` = unbounded).
- **Container depth** (`MAX_PORTABLE_CONTAINER_DEPTH`, 64): a Python-object ingress refuses a
  container taller than this, and refuses one that is reachable from itself, at the five
  refusing boundaries — a tool result's `content`, `emit_artifact` metadata, a hosted task's
  request and result, and a model turn's tool-call `arguments`. The same number is the approval
  request's `MAX_ARGUMENT_DEPTH` and the preview walk's `MAX_JSONISH_DEPTH`, so an `ask`-gated
  and an `allow`-gated call are bounded alike; note the two behave differently at the bound by
  design — approval and ingress **raise**, the preview **elides**. It is deliberately below the
  bounded decoder's 512 nesting limit: `dataclasses.asdict` (the checkpoint writer) dies at 492
  containers on CPython 3.11, so a value in [492, 512] would clear every gate and then fail the
  run at persistence. Payloads that arrive through a JSON parse are unaffected — the decoder's
  own limit is stricter than anything it will hand on.

### Client connection retry

The gateway model adapter (`providers/gateway.py`), the web gateway client (`web.py`), and
the web upstream providers (`reference/web_gateway/providers.py`) retry transient
connection-level failures (`URLError` / `TimeoutError` / a bare `OSError` such as a
connection reset mid-read) with backoff. An `HTTPError` is a real response and is **never**
retried as a connection error. The model adapter's retry is policy-driven by
`ModelRetryConfig.retry_on` (default codes: `gateway_timeout`, `gateway_network_error`,
`gateway_rate_limited`, `gateway_server_error`).

### The retry layer (`ModelRetryConfig.layer`)

`layer` names the single owner of the retry loop for a model call, so two loops cannot
multiply attempts — a guarantee whose reach the compliance paragraph below states: one
process, and an adapter that honors either the neutralized config or the layer value itself.
The default, `"adapter"`, is the behavior above: the adapter's own loop
retries and the kernel makes exactly one adapter call. Under `"kernel"` the
`ModelCallRunner` owns the loop: it re-dispatches the already-keyed request on a retryable,
non-config-recoverable `ModelAdapterError` — the taxonomy, not `retry_on`, which stays the
adapter loop's provider-specific code selector — waits on the same schedule fields under the
run's cancel/deadline race, and refuses to back off into a deadline it cannot fit,
re-raising the transient error instead of converting it into a `RunTimeout` that names
nothing. A delivered stream chunk closes the retry window: once a consumer holds output, a
retry would replay it — the same commit line the gateway's streamed loop and the OpenAI
SDK's pre-stream window already draw.

Compliance is two-sided. The runner hands the adapter a dispatch copy whose policy is
neutralized to `max_attempts=1` (silencing any config-honoring loop, including a
third-party adapter that never learned `layer` exists) with the layer value preserved, and
a compliant adapter whose loop lives outside the config honors the value itself: the
gateway client answers one attempt under `"kernel"` on both its loops, and the OpenAI
adapter passes `max_retries=0` to the SDK. The receipt keeps the caller's configured
policy, and the replay key excludes the whole retry block, so neither the layer nor the
neutralization moves any recorded identity. Layers below one process stay out of scope: an
upstream gateway deployment owns its own transport policy, and its retries surface as
`provider_retried` evidence, not as attempts.

Under `"kernel"`, `ModelCallReceipt.attempts` counts every dispatch, `provider_retried`
keeps meaning a loop *below* the adapter boundary ran, and usage stamped on attempts the
loop swallowed is summed into the receipt's `usage` on either settle exit — the receipt is
the per-call audit surface for what the whole logical call cost, and `attempt_log` itemizes
it per dispatch. The run's accounting consumes it: the settled path accumulates the
receipt's usage, and the failure path restamps the terminal error with the merged total
before it escapes, so `metrics.json`, `state.total_usage`, the cumulative token budget and
the child roll-up all carry absorbed spend. The budget stays a pre-call gate — checked
before a turn begins — and `max_attempts` is the bound on what one logical call may spend
inside itself. Spend absorbed by an adapter's *own* loop is different: the client never
sees those attempts' bills — each hop meters its own wire calls, the way the reference
gateway's tenant meter counts them server-side — so a deployment that wants absorbed spend
in run accounting assigns the loop to the kernel. A call the run's own boundary ends —
cancelled, timed out, or a turn interrupted mid-call — counts the same: every arm the loop
re-raises a model failure through reaches the accounting, not only the one typed to
`ModelAdapterError`. The attempts a kernel retry absorbed before the boundary are completed,
billed wire calls that finished before anything was cancelled, and the interrupt does not
un-spend them; what a cancelled run still does not count is the terminating dispatch's own
bill, which no layer has ever reported for a call that never settled.

That holds below `Exception` too: a host cancelling the task that drives `asubmit`/`arun_once`
raises `asyncio.CancelledError` into the call, and a `KeyboardInterrupt` arrives the same way —
neither is a `NativeAgentError`, neither is an `Exception`, and both leave a live run behind that
the host still finalizes and still reports totals for. Both account, and both are re-raised
unchanged: accounting there is guarded so a raising observer cannot replace the stop, because a
coroutine that swallows a cancellation is a broken coroutine. `SystemExit` and `GeneratorExit`
deliberately do **not** account — teardown is where a recorder's sinks are closing, and a meter
nobody will read is not worth touching a closing file for.

### Durable `ModelCallRunner` lifecycle

`ModelCallRunner.lifecycle_hook` activates authoritative paid-call journaling. The default value is
`None`, so existing in-process calls keep their per-call key issuance, adapter dispatch, retry,
receipt, observer, and passive `settled_sink` behavior.

An activated lifecycle requires `acall(logical_call_id=...)`. The ID is a stable execution address,
not a request hash. `core.model_invocation.logical_model_call_id(run_id, step_id)` derives the
AgentLoop form without exposing the raw run or step string. Each kernel dispatch uses
`model_dispatch_id(logical_call_id, attempt)`. A standalone durable caller supplies its own stable
logical ID.

The synchronous lifecycle sequence is:

```text
normalize + request digest
  -> reserve effective key
  -> commit dispatch_started
  -> enter adapter
  -> commit settled success/refusal OR commit unknown
  -> deliver passive observer/sidecar evidence
```

`reserve()` may substitute the idempotency key from an existing reservation. The runner rejects any
change to logical ID, dispatch ID/attempt, request digest, or digest generation before adapter
entry. The first committed key remains fixed across restore and kernel retry. Durable mode accepts
only `digest_status="ok"`; an absent or oversized request digest raises
`durable_invocation_unkeyable` before reserve and provider dispatch.

Failure evidence is explicit and fail-closed. An ordinary `ModelAdapterError` is ambiguous.
`ModelDispatchRefused` is the typed proof of a definite terminal refusal. A typed refusal commits a
settled failure and may enter the existing kernel retry policy when its receipt is retryable.
Connection loss, timeout, malformed terminal data, and every other exception type commit `unknown`,
raise `dispatch_unknown`, and forbid an automatic paid-call retry. Provider-specific refusal proof
remains opt-in; HTTP status and `retryable=True` do not infer it.

Successful settlement stores the canonical `core.model_payloads.response_record_body()` bytes as a
private result blob. The body preserves final text, tool calls, reasoning, usage, stop reason, and
provider retry evidence. It excludes `ModelTurn.raw`. An unencodable or oversized result becomes an
unknown dispatch because provider work has already happened.

Lifecycle writes control execution and their exceptions are not observer failures. Reserve/start
failure prevents adapter entry. Settle failure attempts an unknown transition and surfaces
`dispatch_unknown` even when that transition also fails. The next recovery pass interprets a
remaining `dispatch_started` head as unknown. Python `BaseException` represents a process-stop
failpoint and bypasses in-process compensation, leaving the last committed head for recovery.

The lifecycle value types live in `monoid_agent_kernel.model_lifecycle`. They stay outside the
stable package root and contain no storage, queue, lease, database, or Temporal dependency. A host
adapter binds them to `FencedRunSink` and `WriterToken`; the runner imports neither hosting type.

### AgentLoop fenced recovery

`AgentLoop(run_sink=..., writer_token=...)` activates fenced checkpoint and model-invocation
recovery for one run. Both values are required together. The token's `run_id` must match the run,
and the sink must declare `lease_fencing`, `durable_checkpoints`, and `durable_invocations`.
`checkpoint_persist_callback` cannot be combined with this mode. The default AgentLoop path keeps
the existing local checkpoint behavior and does not import the hosting package during a root or
core-only import.

AgentLoop derives each logical call from `AgentRunSpec.run_id` and the durable `turn_XXXX`
coordinate backed by checkpoint `session_step`. Caller-supplied `InvocationContext.step_id` remains
observability provenance and cannot change the recovery address. Restoring the same checkpoint
recreates the same next turn coordinate, so the new activation queries the same invocation head
after it computes the canonical request digest. The host adapter applies this table before provider
entry:

| Authoritative head | Recovery action | Provider calls during recovery |
|---|---|---:|
| missing | reserve a new dispatch | 1 |
| `reserved` | reuse the stored idempotency key and continue | 1 |
| `dispatch_started` | commit `unknown`, raise `dispatch_unknown` | 0 |
| `unknown` | raise `dispatch_unknown` | 0 |
| successful `settled` | verify and replay the private result blob | 0 |
| failed `settled` | restore attempt/usage evidence; resume an explicitly safe kernel retry or surface the refusal | 0 or remaining policy-bound attempts |

Every existing head must match the current request digest and digest generation. A mismatch raises
`durable_invocation_request_conflict` before provider entry. Corrupt and unsupported invocation
heads fail closed. Before exposing either settled arm, recovery re-commits the exact authoritative
revision through `commit_invocation()`. The fence check precedes idempotency in that mutation, so a
stale activation cannot replay a successful turn or execute its tool calls. Successful result
recovery verifies the `blob:<sha256>` address, blob bytes, and strict recorded-turn shape. It
compares stop reason when that fact survived the public-safe receipt projection. Recovered turns
contain an empty `raw` mapping. The public receipt describes the whole logical call, including usage
absorbed by kernel retries and provider-retry evidence folded across attempts. The private result
describes the final provider turn. Recovery keeps these two evidence scopes separate, and run
accounting consumes the public receipt's canonical usage counters, including
`cache_creation_tokens` and `audio_tokens`. Missing, tampered, undecodable, and receipt-conflicting
results raise a typed integrity error carrying that already-billed public usage. Failed result
recovery preserves the failure code, provider code, HTTP status, retryability, configuration
recoverability, provider-retry fact, attempt count, usage, and explicit `stream_committed` evidence
recorded in the public-safe receipt. A current kernel retry policy may continue from the next
dispatch attempt only when the refusal is retryable, configuration-independent, below its attempt
limit, and `stream_committed` is explicitly `false`. Missing historical delivery evidence fails
closed and surfaces the stored refusal. Resumed receipts retain aggregate usage and leave the
unavailable historical per-attempt log empty. Provider exception text is never reconstructed.

Every non-task suspension commits its checkpoint through `FencedRunSink.commit_checkpoint()`.
AgentLoop accepts an exact `CommitResult` with `committed` or `already_committed`; fenced, conflict,
invalid, and raising results escape as checkpoint-persistence failures. Durable mode has no local
checkpoint fallback. `RunCheckpoint.last_model_invocation` carries the latest compact summary for
diagnostics and blob reachability. The invocation head loaded from `FencedRunSink` remains the
authority for recovery.

The concrete adapter is `monoid_agent_kernel.hosting.model_calls.FencedModelCallLifecycle`. Its
module path is explicit while stable hosting import expansion remains an M2 decision. It performs
checked loads, monotonic revision writes, content-addressed result settlement, and writer-token
fencing through the host-owned sink. It contains no database, queue, or Temporal runtime.

### Model evidence delivery policy

`AgentLoop.model_evidence_policy` selects one of three opt-in host delivery contracts:

| Policy | Durable mutation | Failure result |
|---|---|---|
| `passive` | No additional host mutation. Existing `settled_sink` observers remain failure-contained. | The model result keeps its original classification. |
| `required` | Commit the invocation first, then call `FencedRunSink.commit_model_evidence()` for that exact settled revision. | `Suspension(reason="turn_failed", error_code="evidence_uncommitted")` |
| `outbox` | Call `commit_invocation(..., stage_evidence=True)` so the settled revision and host-owned evidence outbox entry share one transaction. | A rejected settlement follows the existing `dispatch_unknown` path. |

`required` and `outbox` need `run_sink` plus `writer_token`. `outbox` also needs the sink to
declare `transactional_outbox=True`; configuration fails before the run opens when that guarantee
is absent. The core defines the atomic mutation flag. The host owns the outbox schema, poller,
backoff, dead-letter handling, and destination credentials.

An invocation settlement remains authoritative when a later required evidence commit fails.
The first reservation persists `DurableModelInvocation.requires_evidence=true`, and every
invocation revision and retry preserves that stable field. Settlement and the obligation therefore
share the authoritative invocation journal transaction. A crash after settlement commit and before
evidence delivery or checkpoint publication leaves enough durable state for a replacement
activation to finish required delivery, even when that activation uses the passive default.

Recovery uses the journal obligation or the checkpointed logical-call ID and request digest to
re-commit the exact invocation revision and required evidence before it reads current runtime
config, context providers, tool surface, or media. It then applies a stored success or final refusal
to loop state before those current request-building dependencies run. A recovered final settles
from the stored outcome; a recovered tool-call turn enters the canonical message log before current
tool resolution and execution setup. Runtime-config drift cannot block evidence delivery or
application of the stored outcome. The provider call count does not increase. Passive model-I/O
observers and requested model-call sidecars receive the authoritative call during its original
settlement, including when required evidence fails afterward. Evidence recovery does not publish
the passive call a second time. The checkpoint marker sets
`ModelDispatchRecoveryQuery.require_evidence=True` as a second recovery path for an already
published evidence park.
`run_once()` releases this durable park and surfaces `TurnNotSettled` instead of closing it into a
terminal checkpoint. Repeated evidence parks, transcript
rows, and public events carry only the non-negative usage delta beyond the amount already projected
by the prior park. Recovery surfaces a stored retryable refusal after evidence delivery and starts
no automatic paid continuation. A later driver-controlled retry begins at a new model-step boundary.

An interrupt may arrive after a recovered assistant tool-call turn enters the canonical message
log and before all of its tools finish. The interruption suspension persists
`model_tool_calls_pending=true`. `pending_observations` then carries the completed calls in that
batch. A `None` resume reloads the same settled model result, recognizes the already-projected
assistant turn, skips tool call IDs with completed observations, and executes the remaining calls.
The assistant turn is never appended twice. New user input is rejected with
`evidence_recovery_requires_resume` until a `None` resume completes the pending tool exchange.
The same checkpoint carries the context-owned plan, pending `run.finish` value, and unconsumed
`tool.search` binding loads. A process-level restore therefore reconstructs these kernel effects
before it skips calls with completed observations. Tool handlers still need stable external
idempotency for process loss before a call returns an observation.

`core.outcome.terminal_outcome_from_suspension()` projects an evidence park to
`TerminalOutcome(kind="evidence_uncommitted", retry_eligibility="safe")`. It projects
`dispatch_unknown` to `after_reconciliation`. The projection copies safe taxonomy fields and
opaque references; it has no raw prompt, model output, reasoning, replay body, or exception-text
field.

## Run Artifacts

Manifest and transcript are binding-aware. Streamed model content has a separate private sidecar:

- `manifest.json.agent_config`: definition id, config version, config hash
- `manifest.json.tool_surface`: resolver, tool search settings, bound catalog count
- `tool_surface_snapshot`: immediate/searchable bound tool specs and binding
  authorizations
- `agent_runtime_config_snapshot`: definition id, config version/hash, binding ids
- `agent.config.updated`: emitted when the loop observes a new config hash
- `model-content.jsonl`: optional `monoid.model-content.v1` stream lifecycle, output/reasoning
  segments, and settled-text records; old run directories remain valid without it

Replay uses recorded snapshots. Current registry state does not reinterpret an
old turn.
