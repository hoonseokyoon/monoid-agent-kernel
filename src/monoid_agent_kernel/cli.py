from __future__ import annotations

import contextlib
import dataclasses
import json
import time
from dataclasses import replace
from pathlib import Path
from collections.abc import Callable
from typing import Any

import click

from monoid_agent_kernel._version import user_agent
from monoid_agent_kernel.core._event_log import (
    EventLogChanged,
    EventLogCorruption,
    EventLogRecord,
    inspect_event_log_tail,
    iter_committed_event_records,
)
from monoid_agent_kernel.core.agents import (
    AgentDefinition,
    AgentRuntimeConfig,
    StaticRuntimeConfigProvider,
)
from monoid_agent_kernel.core.json_ingress import loads_json_ingress
from monoid_agent_kernel.reference.backend.http import create_backend_server
from monoid_agent_kernel.reference.backend.service import RunnerBackend
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.narration import narrate_event
from monoid_agent_kernel.core.spec import (
    AgentRunSpec,
    ModelConfig,
    RunLimits,
)
from monoid_agent_kernel.core.payload_gc import UnusableAgeGate, collect_payload_garbage
from monoid_agent_kernel.core.schemas import validate_run_dir
from monoid_agent_kernel.core.packages import (
    apply_package,
    create_approval,
    export_package,
    import_package,
    inspect_package,
    verify_package,
    write_apply_result,
    write_approval,
)
from monoid_agent_kernel.core.projections import project_run_status
from monoid_agent_kernel.core.proposal_file import ProposalFileError, read_proposal_file_payload
from monoid_agent_kernel.event_loader import load_event_sinks
from monoid_agent_kernel.tasks import (
    public_job_artifact_for,
    public_job_artifacts,
    read_job_log_text,
    request_job_cancel,
)
from monoid_agent_kernel.reference.llm_gateway.http import create_llm_gateway_server
from monoid_agent_kernel.reference.llm_gateway.providers import offline_provider_factory
from monoid_agent_kernel.reference.llm_gateway.service import LlmGatewayBackend
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.payload_replay import ReplayCorpus
from monoid_agent_kernel.providers._request_identity import _model_identity
from monoid_agent_kernel.providers.base import (
    ModelAdapter,
    normalize_model_config,
    resolved_provider_name,
)
from monoid_agent_kernel.providers.gateway import (
    DEFAULT_RELAYED_PROVIDER as _DEFAULT_RELAYED_PROVIDER,
    GatewayModelAdapter,
    resolve_relayed_provider,
)
from monoid_agent_kernel.providers.openai import OpenAIModelAdapter
from monoid_agent_kernel.providers.replay import ReplayModelAdapter
from monoid_agent_kernel.recorder import StdoutJsonlSink, append_event_to_run
from monoid_agent_kernel.skills import SkillProvider, load_skill_definitions
from monoid_agent_kernel.subagent_loader import load_subagent_definitions
from monoid_agent_kernel.tool_loader import load_capability_broker, load_tool_provider
from monoid_agent_kernel.core.capability import AutoGrantBroker
from monoid_agent_kernel.env import env_name_for_error, getenv
from monoid_agent_kernel.web import WebGatewayClient
from monoid_agent_kernel.reference.web_gateway.http import create_web_gateway_server
from monoid_agent_kernel.reference.web_gateway.providers import (
    BraveLlmContextProvider,
    BraveSearchProvider,
    CompositeWebProvider,
    HttpFetchProvider,
    SearchFetchContextProvider,
)
from monoid_agent_kernel.reference.web_gateway.service import FakeWebProvider, WebGatewayBackend
from monoid_agent_kernel.reference.studio.cli import studio as studio_group
from monoid_agent_kernel.builder import builder_group

_WATCH_BATCH_MAX_RECORDS = 256
_WATCH_BATCH_MAX_BYTES = 1024 * 1024


@click.group()
def main() -> None:
    """Run Monoid Agent Kernel."""


main.add_command(studio_group)
main.add_command(builder_group)


@main.command()
@click.option(
    "--spec",
    "spec_file",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Load run-specific values from a JSON file (AgentRunSpec.to_json shape). "
        "When set, individual spec flags are ignored; runtime flags "
        "(runtime config, gateway URLs/tokens, --event-sink-module, --stream-json, "
        "--no-status-file, --tool-module, --model-calls-file, --model-payload-file) still apply."
    ),
)
@click.option("--agent-definition-file", type=click.Path(path_type=Path), default=None)
@click.option("--runtime-config-file", type=click.Path(path_type=Path), default=None)
@click.option("--workspace", type=click.Path(path_type=Path), default=None)
@click.option("--instruction", type=str, default="")
@click.option("--instruction-file", type=click.Path(path_type=Path), default=None)
@click.option("--llm-gateway-url", type=str, default=None, help="Internal LLM gateway URL.")
@click.option(
    "--llm-gateway-token-env",
    type=str,
    default="MONOID_LLM_GATEWAY_TOKEN",
    show_default=True,
    help="Environment variable containing a short-lived gateway token.",
)
@click.option(
    "--llm-gateway-token-file",
    type=click.Path(path_type=Path),
    default=None,
    help="File containing a short-lived gateway token.",
)
@click.option(
    "--llm-gateway-provider",
    type=str,
    default=_DEFAULT_RELAYED_PROVIDER,
    show_default=True,
    help=(
        "Names the UPSTREAM provider the gateway relays, never the gateway hop itself. "
        "It tags the provider-native reasoning artifacts the gateway carries back (they only "
        "replay to a matching provider) and it is the provider attributed on the observability "
        'surfaces: the model-call receipt and OTel\'s gen_ai.provider.name. Pass "none" to '
        "disable tagging, which is right for a gateway whose upstream has no reasoning artifacts."
    ),
)
@click.option(
    "--allow-direct-provider-api",
    is_flag=True,
    help="Allow direct provider API access for local smoke tests only.",
)
@click.option(
    "--mode",
    type=click.Choice(["read-only", "propose", "apply"]),
    default="propose",
    show_default=True,
)
@click.option(
    "--workspace-backend",
    type=click.Choice(["overlay", "staging"]),
    default="overlay",
    show_default=True,
)
@click.option(
    "--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True
)
@click.option("--run-id", type=str, default=None, help="Use a specific run id.")
@click.option("--max-steps", type=int, default=30, show_default=True)
@click.option("--max-tool-calls", type=int, default=100, show_default=True)
@click.option("--max-bytes-read", type=int, default=1_000_000, show_default=True)
@click.option("--max-duration-s", type=int, default=900, show_default=True)
@click.option("--tool-module", multiple=True, help="Load custom tools from path.py:function.")
@click.option(
    "--agents-directory",
    type=click.Path(path_type=Path),
    default=None,
    help="Load subagent definitions (*.md with frontmatter) from a directory, enabling agent.spawn.",
)
@click.option(
    "--skills-directory",
    type=click.Path(path_type=Path),
    default=None,
    help="Load Agent Skills (SKILL.md with frontmatter) from a directory, enabling the skill tools.",
)
@click.option(
    "--capability-broker",
    type=str,
    default=None,
    help="Load a CapabilityBroker from path.py:factory to gate tools that declare requires_lease.",
)
@click.option(
    "--auto-grant-capabilities",
    is_flag=True,
    help="Use the built-in AutoGrantBroker (local dev): grant any requires_lease tool, scoped to its binding.",
)
@click.option(
    "--deny-path", multiple=True, help="Deny workspace paths matching a backend-provided glob."
)
@click.option(
    "--redact-path", multiple=True, help="Redact matching paths from public events and projections."
)
@click.option("--permission-policy-file", type=click.Path(path_type=Path), default=None)
@click.option("--web-gateway-url", type=str, default=None, help="Internal WebGateway base URL.")
@click.option(
    "--web-gateway-token-env",
    type=str,
    default="MONOID_WEB_GATEWAY_TOKEN",
    show_default=True,
    help="Environment variable containing a short-lived WebGateway token.",
)
@click.option(
    "--web-gateway-token-file",
    type=click.Path(path_type=Path),
    default=None,
    help="File containing a short-lived WebGateway token.",
)
@click.option(
    "--event-sink-module", multiple=True, help="Load custom event sinks from path.py:function."
)
@click.option(
    "--model-calls-file",
    is_flag=True,
    help="Record a per-call model ledger (model_calls.jsonl) in the run directory.",
)
@click.option(
    "--model-payload-file",
    is_flag=True,
    help=(
        "Record the private replay corpus (model_payloads.jsonl plus its chunk directory). "
        "Carries request and response content in full, including what --redact-path masks from "
        "events, and grows with the conversation with no retention verb. Verify with "
        "`monoid validate RUN_DIR`; sweep its crash litter with `monoid gc RUN_DIR --apply`."
    ),
)
@click.option(
    "--model-content-file",
    is_flag=True,
    help=(
        "Record the private model-content sidecar (model-content.jsonl): the streamed output and "
        "reasoning text of each call. Content, like the replay corpus. Selects provider streaming."
    ),
)
@click.option(
    "--replay-from",
    "replay_from",
    multiple=True,
    type=str,
    help=(
        "Serve model calls from a recorded run's replay corpus (RUN_DIR_OR_ID under "
        "--run-root) instead of a live provider. Repeatable: a run that spawned subagents "
        "records each child in its own run directory, so name the children too. Tools "
        "re-execute for real; only the model answers are replayed."
    ),
)
@click.option(
    "--replay-fallthrough",
    is_flag=True,
    help=(
        "On a replay miss, fall through to the live adapter this command would have built "
        "without --replay-from. Without this flag a miss fails the turn (error_code "
        "replay_miss) and no provider is ever contacted."
    ),
)
@click.option("--stream-json", is_flag=True, help="Stream public events as JSONL on stdout.")
@click.option("--no-status-file", is_flag=True, help="Disable status.json updates.")
@click.pass_context
def run(
    ctx: click.Context,
    *,
    spec_file: Path | None,
    agent_definition_file: Path | None,
    runtime_config_file: Path | None,
    workspace: Path | None,
    instruction: str,
    instruction_file: Path | None,
    llm_gateway_url: str | None,
    llm_gateway_token_env: str,
    llm_gateway_token_file: Path | None,
    llm_gateway_provider: str | None,
    allow_direct_provider_api: bool,
    mode: str,
    workspace_backend: str,
    run_root: Path,
    run_id: str | None,
    max_steps: int,
    max_tool_calls: int,
    max_bytes_read: int,
    max_duration_s: int,
    tool_module: tuple[str, ...],
    agents_directory: Path | None,
    skills_directory: Path | None,
    capability_broker: str | None,
    auto_grant_capabilities: bool,
    deny_path: tuple[str, ...],
    redact_path: tuple[str, ...],
    permission_policy_file: Path | None,
    web_gateway_url: str | None,
    web_gateway_token_env: str,
    web_gateway_token_file: Path | None,
    event_sink_module: tuple[str, ...],
    model_calls_file: bool,
    model_payload_file: bool,
    model_content_file: bool,
    replay_from: tuple[str, ...],
    replay_fallthrough: bool,
    stream_json: bool,
    no_status_file: bool,
) -> None:
    """Run an agent against a local workspace."""
    del ctx
    runtime_config = _load_agent_runtime_config(runtime_config_file, agent_definition_file)
    # The instruction is the first user turn, delivered via run_once(); the spec no
    # longer carries it, so it is required for both --spec and --workspace paths.
    if instruction_file is not None:
        instruction = instruction_file.read_text(encoding="utf-8")
    if not instruction.strip():
        raise click.ClickException("--instruction or --instruction-file is required")
    if spec_file is not None:
        if workspace is not None:
            raise click.ClickException(
                "--spec cannot be combined with --workspace; the spec file is authoritative"
            )
        try:
            spec = AgentRunSpec.from_json(loads_json_ingress(spec_file.read_text(encoding="utf-8")))
        except Exception as exc:
            raise click.ClickException(f"failed to load --spec: {exc}") from exc
        if run_id is not None:
            spec = replace(spec, run_id=run_id)
    else:
        if workspace is None:
            raise click.ClickException("--workspace (or --spec) is required")

        resolved_limits = RunLimits(
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            max_bytes_read=max_bytes_read,
            max_duration_s=max_duration_s,
        )

        try:
            permission_policy = _load_permission_policy(
                permission_policy_file,
                deny_path=deny_path,
                redact_path=redact_path,
            )
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc

        spec_kwargs: dict[str, Any] = {}
        if run_id is not None:
            spec_kwargs["run_id"] = run_id
        spec = AgentRunSpec(
            workspace_root=workspace,
            run_root=run_root,
            mode=mode,  # type: ignore[arg-type]
            workspace_backend=workspace_backend,  # type: ignore[arg-type]
            limits=resolved_limits,
            permission_policy=permission_policy,
            **spec_kwargs,
        )

    if _runtime_config_uses_web(runtime_config) and not web_gateway_url:
        raise click.ClickException("runtime config binds web tools; --web-gateway-url is required")
    _human_echo(f"run_id: {spec.run_id}", stream_json=stream_json)
    _human_echo(f"run_dir: {spec.run_root / spec.run_id}", stream_json=stream_json)

    try:
        providers = tuple(load_tool_provider(item) for item in tool_module)
        subagent_definitions = (
            load_subagent_definitions(agents_directory) if agents_directory is not None else {}
        )
        skill_provider: SkillProvider | None = None
        if skills_directory is not None:
            skill_definitions = load_skill_definitions(skills_directory)
            if skill_definitions:
                skill_provider = SkillProvider(skill_definitions)
                # Provider tools are not auto-bound; expose them by merging their bindings
                # into the runtime config (mirrors the MCP provider).
                runtime_config = replace(
                    runtime_config, tools=runtime_config.tools + skill_provider.tool_bindings()
                )
                # Fork skills (context: fork) run as subagents; register their synthesized
                # definitions (namespaced ids, so no collision with --agents-directory).
                subagent_definitions = {
                    **subagent_definitions,
                    **skill_provider.subagent_definitions(),
                }
        extra_sinks = []
        if stream_json:
            extra_sinks.append(StdoutJsonlSink())
        for item in event_sink_module:
            extra_sinks.extend(load_event_sinks(item))
        if capability_broker and auto_grant_capabilities:
            raise ValueError(
                "use either --capability-broker or --auto-grant-capabilities, not both"
            )
        broker = (
            load_capability_broker(capability_broker)
            if capability_broker
            else AutoGrantBroker()
            if auto_grant_capabilities
            else None
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    replay_corpus: ReplayCorpus | None = None
    if replay_from:
        # Pure replay deliberately BYPASSES the live-adapter branch and its gates: an
        # offline replay needs no gateway URL, no --allow-direct-provider-api, and no
        # recognized provider name -- demanding any of them would block exactly the runs
        # this flag exists for. The live branch is built only as a fallthrough inner.
        try:
            replay_corpus = ReplayCorpus.load(
                [_resolve_run_dir(item, run_root) for item in replay_from]
            )
            model_adapter = ReplayModelAdapter(
                replay_corpus,
                inner=(
                    _model_adapter(
                        runtime_config.model or ModelConfig(),
                        llm_gateway_url=llm_gateway_url
                        or (runtime_config.model.gateway_url if runtime_config.model else None),
                        llm_gateway_token_env=llm_gateway_token_env,
                        llm_gateway_token_file=llm_gateway_token_file,
                        llm_gateway_provider=llm_gateway_provider,
                        allow_direct_provider_api=allow_direct_provider_api,
                    )
                    if replay_fallthrough
                    else None
                ),
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        _replay_preflight(
            replay_corpus,
            runtime_config.model or ModelConfig(),
            model_adapter,
            fallthrough=replay_fallthrough,
        )
    else:
        model_adapter = _model_adapter(
            runtime_config.model or ModelConfig(),
            llm_gateway_url=llm_gateway_url
            or (runtime_config.model.gateway_url if runtime_config.model else None),
            llm_gateway_token_env=llm_gateway_token_env,
            llm_gateway_token_file=llm_gateway_token_file,
            llm_gateway_provider=llm_gateway_provider,
            allow_direct_provider_api=allow_direct_provider_api,
        )
    # One adapter serves every turn of this run, so an adapter that can hold its provider client
    # open across turns should: the direct-OpenAI one builds a client per call otherwise, which
    # costs far more than the request it carries. Duck-typed because only some adapters have a
    # client to hold -- the gateway adapter owns its httpx client per call by design.
    #
    # Driven through the pair that is probed. ``enter_context`` needed ``__enter__``/``__exit__``
    # instead, so the ``open``/``close`` adapter this probe invites -- the lifecycle pair the rest
    # of the kernel uses, on ``AgentLoop`` and ``LoopSession`` -- raised ``TypeError`` before the
    # first turn, outside the handler above, killing the run with a bare traceback.
    #
    # Both halves are resolved *before* either is called. Registering the bound ``close`` after
    # ``open()`` looked like it failed early enough, and did not: ``open()`` had already allocated
    # whatever it allocates -- a connection pool, for the adapter this exists for -- and the
    # ``AttributeError`` from resolving the missing ``close`` then escaped past the handler above
    # with nothing left able to release it. An adapter that offers one half of the pair is
    # misconfigured, so it is reported as such, before it holds anything.
    with contextlib.ExitStack() as adapter_scope:
        opener = getattr(model_adapter, "open", None)
        if callable(opener):
            closer = getattr(model_adapter, "close", None)
            if not callable(closer):
                raise click.ClickException(
                    f"model adapter {type(model_adapter).__name__} exposes open() without a "
                    "callable close(); nothing would release what open() allocates"
                )
            # Reported the way every other startup failure is. These two calls sit below the
            # `except Exception` that normalizes the setup above, so an adapter whose pool
            # construction or teardown raised ended the command in a bare traceback.
            try:
                opener()
            except Exception as exc:
                raise click.ClickException(f"model adapter open() failed: {exc}") from exc
            adapter_scope.push(_adapter_teardown(closer))
        result = AgentLoop(
            spec=spec,
            subagent_definitions=subagent_definitions,
            model_adapter=model_adapter,
            tool_providers=providers + ((skill_provider,) if skill_provider is not None else ()),
            context_providers=(skill_provider,) if skill_provider is not None else (),
            capability_broker=broker,
            event_sinks=tuple(extra_sinks),
            model_calls_file=model_calls_file,
            model_payload_file=model_payload_file,
            model_content_file=model_content_file,
            # Run-level provenance (D-e): which corpora served this run, as the corpus
            # envelopes name them -- comma-joined (run ids carry no commas), landing
            # verbatim on every ledger line and inherited by children. Never in the
            # replay run's own corpus envelope; provenance is the ledger's business.
            invocation_context=(
                InvocationContext(
                    attributes={"replay_from": ",".join(replay_corpus.run_ids())}
                )
                if replay_corpus is not None
                else None
            ),
            status_file=not no_status_file,
            permission_policy=spec.permission_policy,
            runtime_config_provider=StaticRuntimeConfigProvider(runtime_config),
            web_gateway_client=(
                WebGatewayClient(
                    web_gateway_url,
                    token_env=web_gateway_token_env,
                    token_file=web_gateway_token_file,
                )
                if _runtime_config_uses_web(runtime_config) and web_gateway_url
                else None
            ),
        ).run_once(instruction)
        # All inside the scope, because an exception from a cleanup callback replaces whatever is
        # leaving the block. With the echoes outside, a teardown that raised swallowed the result of a
        # run that had completed. With the failure raised outside, it was never reached at all: the
        # teardown error left the block first and `result.error` -- the provider failure an operator
        # needs -- was simply never reported. Raised in here, it is the exception already on its way
        # out, and the teardown demotes itself to a warning beside it.
        _human_echo(f"status: {result.status}", stream_json=stream_json)
        if result.final_text:
            _human_echo(f"summary: {result.final_text}", stream_json=stream_json)
        if result.error:
            raise click.ClickException(result.error)


@main.command()
@click.argument("run_dir_or_id", type=str)
@click.option(
    "--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True
)
@click.option("--from-start", is_flag=True, help="Read events from the beginning of the file.")
@click.option("--follow", is_flag=True, help="Keep waiting for new events.")
@click.option("--json", "json_output", is_flag=True, help="Print raw JSONL events.")
def watch(
    run_dir_or_id: str, run_root: Path, from_start: bool, follow: bool, json_output: bool
) -> None:
    """Watch a run's public events."""
    events_path = _resolve_events_path(run_dir_or_id, run_root)
    if not events_path.exists():
        raise click.ClickException(f"events.jsonl not found: {events_path}")

    try:
        initial_tail = inspect_event_log_tail(events_path)
        source_identity = (initial_tail.device, initial_tail.inode)
        offset = 0 if from_start or not follow else initial_tail.committed_end
        while True:
            records, offset, drained = _read_watch_batch(events_path, offset, source_identity)
            for record in records:
                click.echo(record.raw_json if json_output else _compact_event_line(record.raw_json))
            if not drained:
                continue
            if not follow:
                break
            time.sleep(0.25)
    except EventLogCorruption as exc:
        raise click.ClickException(str(exc)) from exc


def _read_watch_batch(
    events_path: Path,
    offset: int,
    source_identity: tuple[int, int],
) -> tuple[list[EventLogRecord], int, bool]:
    before = inspect_event_log_tail(events_path)
    if not before.exists or (before.device, before.inode) != source_identity:
        raise EventLogChanged(f"event log was replaced while watching: {events_path}")
    if before.committed_end < offset:
        raise EventLogChanged(f"event log was truncated while watching: {events_path}")

    records: list[EventLogRecord] = []
    batch_bytes = 0
    drained = True
    for record in iter_committed_event_records(events_path, start_offset=offset):
        record_bytes = record.next_byte_offset - record.byte_offset
        if records and (
            len(records) >= _WATCH_BATCH_MAX_RECORDS
            or batch_bytes + record_bytes > _WATCH_BATCH_MAX_BYTES
        ):
            drained = False
            break
        records.append(record)
        batch_bytes += record_bytes

    next_offset = offset
    if records:
        next_offset = records[-1].next_byte_offset
    if drained:
        next_offset = max(next_offset, before.committed_end)

    after = inspect_event_log_tail(events_path)
    if not after.exists or (after.device, after.inode) != source_identity:
        raise EventLogChanged(f"event log was replaced while watching: {events_path}")
    if after.committed_end < next_offset:
        raise EventLogChanged(f"event log was truncated while watching: {events_path}")
    return records, next_offset, drained


@main.command("status")
@click.argument("run_dir_or_id", type=str)
@click.option(
    "--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True
)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def status_command(run_dir_or_id: str, run_root: Path, json_output: bool) -> None:
    """Project a run directory into compact status state."""
    run_dir = _resolve_run_dir(run_dir_or_id, run_root)
    payload = project_run_status(run_dir)
    # The projection degrades rather than raising, so this surface decides what that means. It
    # prints the partial snapshot-and-prefix answer first -- a caller asked for it and it is the
    # only answer available -- and then fails, because `state` may name a run that has since
    # finished and a script polling this must not read that as still running.
    event_log_error = str(payload.get("event_log_error") or "")
    if json_output:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
        if event_log_error:
            raise click.ClickException(event_log_error)
        return
    click.echo(f"run_id: {payload.get('run_id', '')}")
    click.echo(f"state: {payload.get('state', '')}")
    click.echo(f"terminal: {str(bool(payload.get('terminal'))).lower()}")
    if payload.get("error_code"):
        click.echo(f"error_code: {payload['error_code']}")
    if payload.get("current_step") is not None:
        click.echo(f"current_step: {payload['current_step']}")
    if payload.get("current_tool"):
        click.echo(f"current_tool: {payload['current_tool']}")
    if payload.get("waiting_for_background_jobs"):
        click.echo("waiting_for_background_jobs: true")
    if payload.get("running_jobs"):
        click.echo(f"running_jobs: {len(payload['running_jobs'])}")
    if payload.get("proposal_hash"):
        click.echo(f"proposal_hash: {payload['proposal_hash']}")
    if payload.get("changed_paths"):
        click.echo(f"changed_paths: {', '.join(map(str, payload['changed_paths']))}")
    if event_log_error:
        raise click.ClickException(event_log_error)


@main.command("jobs")
@click.argument("run_dir_or_id", type=str)
@click.option(
    "--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True
)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def jobs_command(run_dir_or_id: str, run_root: Path, json_output: bool) -> None:
    """List background shell jobs for a run."""
    run_dir = _resolve_run_dir(run_dir_or_id, run_root)
    payload = {"run_dir": str(run_dir), "jobs": public_job_artifacts(run_dir)}
    if json_output:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return
    for job in payload["jobs"]:
        click.echo(
            f"{job.get('job_id', '')} {job.get('status', '')} "
            f"exit={job.get('exit_code', '')} duration={float(job.get('duration_s') or 0):.3f}s"
        )


@main.group("job")
def job_group() -> None:
    """Inspect or control one background shell job."""


@job_group.command("status")
@click.argument("job_id", type=str)
@click.option("--run", "run_dir_or_id", type=str, required=True)
@click.option(
    "--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True
)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def job_status_command(job_id: str, run_dir_or_id: str, run_root: Path, json_output: bool) -> None:
    """Show one background job status."""
    run_dir = _resolve_run_dir(run_dir_or_id, run_root)
    payload = public_job_artifact_for(run_dir, job_id)
    if json_output:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return
    click.echo(f"job_id: {payload.get('job_id', '')}")
    click.echo(f"status: {payload.get('status', '')}")
    click.echo(f"exit_code: {payload.get('exit_code', '')}")
    click.echo(f"duration_s: {payload.get('duration_s', '')}")
    click.echo(f"stdout_bytes: {payload.get('stdout_bytes', 0)}")
    click.echo(f"stderr_bytes: {payload.get('stderr_bytes', 0)}")


@job_group.command("logs")
@click.argument("job_id", type=str)
@click.option("--run", "run_dir_or_id", type=str, required=True)
@click.option(
    "--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True
)
@click.option(
    "--stream",
    "stream_name",
    type=click.Choice(["stdout", "stderr"]),
    default="stdout",
    show_default=True,
)
@click.option("--tail-bytes", type=int, default=None)
@click.option("--offset", type=int, default=None)
@click.option("--follow", is_flag=True)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def job_logs_command(
    job_id: str,
    run_dir_or_id: str,
    run_root: Path,
    stream_name: str,
    tail_bytes: int | None,
    offset: int | None,
    follow: bool,
    json_output: bool,
) -> None:
    """Read stdout or stderr for one background job."""
    run_dir = _resolve_run_dir(run_dir_or_id, run_root)
    next_offset = offset
    while True:
        payload = read_job_log_text(
            run_dir,
            job_id,
            stream=stream_name,  # type: ignore[arg-type]
            tail_bytes=tail_bytes if next_offset is None else None,
            offset=next_offset,
        )
        if json_output:
            click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
        elif payload.get("content"):
            click.echo(payload["content"], nl=False)
        next_offset = int(payload.get("next_offset") or 0)
        if not follow:
            break
        time.sleep(0.5)


@job_group.command("cancel")
@click.argument("job_id", type=str)
@click.option("--run", "run_dir_or_id", type=str, required=True)
@click.option(
    "--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True
)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def job_cancel_command(job_id: str, run_dir_or_id: str, run_root: Path, json_output: bool) -> None:
    """Request cancellation for one background job."""
    run_dir = _resolve_run_dir(run_dir_or_id, run_root)
    payload = request_job_cancel(run_dir, job_id)
    if json_output:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    else:
        click.echo(f"cancel_requested: {payload['job_id']}")


@main.command()
@click.argument("run_dir_or_id", type=str)
@click.option(
    "--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True
)
@click.option(
    "--file", "file_path", type=str, default=None, help="Show one proposed file's snapshot content."
)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def proposal(run_dir_or_id: str, run_root: Path, file_path: str | None, json_output: bool) -> None:
    """Inspect a run's proposal snapshot."""
    run_dir = _resolve_run_dir(run_dir_or_id, run_root)
    proposal_path = run_dir / "proposal.json"
    if not proposal_path.exists():
        raise click.ClickException(f"proposal.json not found: {proposal_path}")
    payload = loads_json_ingress(proposal_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise click.ClickException("proposal.json must contain an object")
    if file_path is not None:
        file_payload = _proposal_file_payload(run_dir, payload, file_path)
        if json_output:
            click.echo(
                json.dumps(file_payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
            )
        else:
            click.echo(file_payload["content"])
        return
    if json_output:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return
    click.echo(f"run_id: {payload.get('run_id', '')}")
    click.echo(f"mode: {payload.get('mode', '')}")
    click.echo(f"diff: {payload.get('diff_path', '')} ({payload.get('diff_bytes', 0)} bytes)")
    files = payload.get("files") if isinstance(payload.get("files"), list) else []
    for file in files:
        if isinstance(file, dict):
            click.echo(
                f"{file.get('change_kind', file.get('kind', '?')):>9} "
                f"{file.get('size', 0):>8} {file.get('path', '')}"
            )


@main.command("validate")
@click.argument("run_dir_or_id", type=str)
@click.option(
    "--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True
)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def validate(run_dir_or_id: str, run_root: Path, json_output: bool) -> None:
    """Validate a run directory's public contract artifacts."""
    run_dir = _resolve_run_dir(run_dir_or_id, run_root)
    issues = validate_run_dir(run_dir)
    payload = {
        "run_dir": str(run_dir),
        "ok": not issues,
        "issues": [issue.__dict__ for issue in issues],
    }
    if json_output:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    elif issues:
        for issue in issues:
            click.echo(f"{issue.path}: {issue.message}")
    else:
        click.echo("ok")
    if issues:
        raise click.ClickException("run directory validation failed")


@main.command("gc")
@click.argument("run_dir_or_id", type=str)
@click.option(
    "--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True
)
@click.option(
    "--min-age-s",
    "min_age_s",
    type=float,
    default=86400.0,
    show_default=True,
    help="Never touch an entry younger than this many seconds.",
)
@click.option(
    "--apply",
    "apply_deletes",
    is_flag=True,
    help="Delete the candidates; the default only reports them.",
)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
@click.pass_context
def gc_command(
    ctx: click.Context,
    run_dir_or_id: str,
    run_root: Path,
    min_age_s: float,
    apply_deletes: bool,
    json_output: bool,
) -> None:
    """Collect a run's unreferenced replay-corpus chunks and dead write temporaries.

    Report-only by default; --apply deletes. Never run this against a run whose writer may
    still be alive -- liveness is the operator's knowledge, exactly as it is for validate.
    Referenced chunks are protected by membership, not age; --min-age-s additionally spares
    every entry whose recorded age has not reached it, whatever that entry is.
    """
    if not run_dir_or_id.strip():
        # ``Path("")`` is ``Path(".")``, which exists and is a directory, so the guard below let
        # an unset shell variable through and swept the working directory -- in exactly the
        # scripted nightly sweep this verb is built for.
        raise click.ClickException("a run directory or run id is required")
    run_dir = _resolve_run_dir(run_dir_or_id, run_root)
    if not run_dir.is_dir():
        # A typo'd run id must fail loudly, not come back as a clean empty report.
        raise click.ClickException(f"run directory not found: {run_dir}")
    # Resolved before the sweep, and reported that way. ``_resolve_run_dir`` prefers a path that
    # exists in the working directory over ``--run-root``, which the read-only sibling verbs can
    # afford and a deleter cannot: an operator passing a bare run id deserves to see which
    # directory actually lost files.
    run_dir = run_dir.resolve()
    try:
        report = collect_payload_garbage(run_dir, min_age_s=min_age_s, apply=apply_deletes)
    except UnusableAgeGate as exc:
        # Click's FLOAT accepts "inf" and "nan", and a negative gate parses fine, so the option
        # layer cannot refuse these on type alone. Refusing here -- before any output -- keeps a
        # bad flag from sweeping first and only then failing to report what it swept. Caught by
        # its own type, never by ``ValueError``: this neighbourhood raises those as control flow,
        # and rendering a mid-sweep one as "bad --min-age-s" would blame the flag for deletions
        # that had already happened.
        raise click.BadParameter(str(exc), param_hint="--min-age-s") from exc
    # A refusal or a failed deletion exits non-zero so scripted sweeps notice -- via ctx.exit
    # after the payload, never ClickException, whose Error line joins the payload wherever the
    # two streams merge (a `2>&1` pipeline, or the CliRunner harness that pins this) and leaves
    # --json unparseable; the builder validate precedent. Garbage merely *found* is exit 0:
    # finding it is the verb working.
    failed = (
        report.chunk_dir_state not in ("absent", "ok")
        # A corpus the collector refused to read is a refusal in its own right, wherever the
        # chunk directory stands: scoping this to ``chunk_dir_state == "ok"`` re-opened the very
        # hole it was added to close, one state over -- a run that never offloaded (no chunk
        # directory at all, the common shape, since offload needs a size threshold) reported
        # ``corpus_state: unreadable`` and exited 0. ``unreadable`` and not ``!= "ok"``: an
        # absent corpus is the ordinary state of a run that never enabled the artifact, and
        # alarming on it made a swept-clean run alert forever.
        or report.corpus_state == "unreadable"
        # Still load-bearing with the clause above narrowed: an *absent* corpus also makes
        # chunk-shaped files unjudgeable, and that is a fault when there are such files.
        or any(entry.classification == "unjudged" for entry in report.entries)
        or any(entry.error for entry in report.entries)
    )
    if json_output:
        # ``ensure_ascii=True`` here, unlike every other payload this module prints: those carry
        # values the kernel produced, this one carries directory entry names, which a filesystem
        # may hand back with unpaired surrogates (POSIX surrogateescape for undecodable bytes,
        # NTFS by permission). Emitted verbatim, such a name either fails to write to a strict
        # UTF-8 stream -- the default for the piped consumer this mode exists for -- or gets
        # substituted, silently renaming the file being reported. The text mode's ``!r`` is the
        # same rule; this is its twin.
        click.echo(
            json.dumps(
                dataclasses.asdict(report), ensure_ascii=True, sort_keys=True, allow_nan=False
            )
        )
    else:
        # Every string below that came from the filesystem is rendered ``!r``. The rule was
        # written for ``entry.name`` and stated as if it covered the mode -- it did not reach
        # ``run_dir``, whose surrogate-bearing spelling killed the whole text report *after* the
        # sweep, losing the record of what had just been deleted. Enumerated here rather than
        # summarized: ``run_dir`` and ``name`` are caller/filesystem text; ``error`` embeds a path
        # under ``OSError.__str__``; the states and the numbers are ours.
        click.echo(f"run_dir: {report.run_dir!r}")
        click.echo(f"swept_at: {report.swept_at}")
        click.echo(f"chunk_dir: {report.chunk_dir_state}  corpus: {report.corpus_state}")
        click.echo(
            f"mode: {'apply' if report.applied else 'report-only'}"
            f"  min_age_s: {report.min_age_s!r}"
        )
        kept = sum(1 for entry in report.entries if entry.classification == "kept")
        click.echo(f"kept: {kept}")
        for entry in report.entries:
            # A kept entry is not listed -- a healthy corpus would scroll -- but an error on one
            # is still a fault the exit code reports, so it must not be the one thing the text
            # mode silently drops.
            if entry.classification == "kept" and not entry.error:
                continue
            # ``age_s`` is here because without it the default mode cannot say *why* an entry is
            # or is not a candidate: a month-old orphan and one written a second ago rendered
            # identically, and the gate is the only thing standing between them.
            line = (
                f"{entry.classification:>8} {entry.size:>10} {entry.age_s:>12.1f}s "
                f"{entry.name!r}"
            )
            if entry.deleted:
                line += f"  deleted (freed {entry.reclaimed})"
            if entry.error:
                line += f"  [{entry.error!r}]"
            click.echo(line)
        if report.damaged_line_count:
            shown = ", ".join(map(str, report.damaged_lines))
            more = report.damaged_line_count - len(report.damaged_lines)
            click.echo(
                f"damaged_lines ({report.damaged_line_count}): {shown}"
                + (f", and {more} more" if more else "")
            )
        click.echo(f"candidate_bytes: {report.candidate_bytes}")
        click.echo(f"reclaimed_bytes: {report.reclaimed_bytes}")
    if failed:
        ctx.exit(1)


@main.group("package")
def package_group() -> None:
    """Export, approve, and apply proposal packages."""


@package_group.command("export")
@click.argument("run_dir_or_id", type=str)
@click.option(
    "--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True
)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def package_export(run_dir_or_id: str, run_root: Path, output: Path, json_output: bool) -> None:
    """Export a run directory as a deterministic proposal tar package."""
    run_dir = _resolve_run_dir(run_dir_or_id, run_root)
    try:
        payload = export_package(run_dir, output)
        append_event_to_run(
            run_dir,
            "proposal.package.exported",
            data={"package_hash": payload["package_hash"], "package_path": str(output)},
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    if json_output:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    else:
        click.echo(f"package: {output}")
        click.echo(f"package_hash: {payload['package_hash']}")


@package_group.command("verify")
@click.argument("package_or_run_dir", type=str)
@click.option(
    "--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True
)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def package_verify(package_or_run_dir: str, run_root: Path, json_output: bool) -> None:
    """Verify proposal package hashes and required files."""
    source = _resolve_package_source(package_or_run_dir, run_root)
    result = verify_package(source)
    payload = {
        "ok": result.ok,
        "issues": list(result.issues),
        "source_kind": result.source_kind,
        "package": result.package,
    }
    if json_output:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    elif result.ok:
        click.echo("ok")
        click.echo(f"package_hash: {result.package.get('package_hash', '')}")
    else:
        for issue in result.issues:
            click.echo(issue)
    if not result.ok:
        raise click.ClickException("package verification failed")


@package_group.command("inspect")
@click.argument("package_or_run_dir", type=str)
@click.option(
    "--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True
)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def package_inspect(package_or_run_dir: str, run_root: Path, json_output: bool) -> None:
    """Inspect a proposal package summary."""
    source = _resolve_package_source(package_or_run_dir, run_root)
    payload = inspect_package(source)
    if json_output:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return
    click.echo(f"ok: {payload['ok']}")
    click.echo(f"package_hash: {payload.get('package', {}).get('package_hash', '')}")
    click.echo(
        f"changed_paths: {', '.join(map(str, payload.get('proposal', {}).get('changed_paths', [])))}"
    )


@package_group.command("import")
@click.argument("package_or_run_dir", type=str)
@click.option(
    "--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True
)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def package_import(
    package_or_run_dir: str, run_root: Path, output: Path, json_output: bool
) -> None:
    """Import a proposal package into a verified staging directory."""
    source = _resolve_package_source(package_or_run_dir, run_root)
    try:
        payload = import_package(source, output)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    if json_output:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    else:
        click.echo(f"imported: {payload['output']}")
        click.echo(f"package_hash: {payload['package_hash']}")


@package_group.command("approve")
@click.argument("package_or_run_dir", type=str)
@click.option(
    "--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True
)
@click.option("--approver", type=str, required=True)
@click.option(
    "--path", "approved_path", multiple=True, help="Approve one changed workspace path. Repeatable."
)
@click.option("--note", type=str, default="")
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def package_approve(
    package_or_run_dir: str,
    run_root: Path,
    approver: str,
    approved_path: tuple[str, ...],
    note: str,
    output: Path | None,
    json_output: bool,
) -> None:
    """Create an approval record for a package."""
    source = _resolve_package_source(package_or_run_dir, run_root)
    try:
        approval = create_approval(
            source,
            approver_id=approver,
            approved_paths=approved_path or None,
            note=note,
        )
        output_path = output or (
            _source_run_dir(source) / "approval.json" if source.is_dir() else Path("approval.json")
        )
        write_approval(output_path, approval)
        _append_package_event_if_run_dir(
            source,
            "proposal.approved",
            {"approval_hash": approval["approval_hash"], "package_hash": approval["package_hash"]},
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    if json_output:
        click.echo(json.dumps(approval, ensure_ascii=False, sort_keys=True, allow_nan=False))
    else:
        click.echo(f"approval: {output_path}")
        click.echo(f"approval_hash: {approval['approval_hash']}")


@package_group.command("reject")
@click.argument("package_or_run_dir", type=str)
@click.option(
    "--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True
)
@click.option("--approver", type=str, required=True)
@click.option("--reason", type=str, required=True)
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def package_reject(
    package_or_run_dir: str,
    run_root: Path,
    approver: str,
    reason: str,
    output: Path | None,
    json_output: bool,
) -> None:
    """Create a rejection record for a package."""
    source = _resolve_package_source(package_or_run_dir, run_root)
    try:
        approval = create_approval(source, approver_id=approver, decision="rejected", note=reason)
        output_path = output or (
            _source_run_dir(source) / "approval.json" if source.is_dir() else Path("approval.json")
        )
        write_approval(output_path, approval)
        _append_package_event_if_run_dir(
            source,
            "proposal.rejected",
            {"approval_hash": approval["approval_hash"], "package_hash": approval["package_hash"]},
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    if json_output:
        click.echo(json.dumps(approval, ensure_ascii=False, sort_keys=True, allow_nan=False))
    else:
        click.echo(f"approval: {output_path}")
        click.echo(f"approval_hash: {approval['approval_hash']}")


@package_group.command("apply")
@click.argument("package_or_run_dir", type=str)
@click.option(
    "--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True
)
@click.option("--approval", "approval_path", type=click.Path(path_type=Path), required=True)
@click.option("--target", type=click.Path(path_type=Path), required=True)
@click.option("--dry-run", is_flag=True)
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def package_apply(
    package_or_run_dir: str,
    run_root: Path,
    approval_path: Path,
    target: Path,
    dry_run: bool,
    output: Path | None,
    json_output: bool,
) -> None:
    """Apply an approved package to a local reference target."""
    source = _resolve_package_source(package_or_run_dir, run_root)
    try:
        result = apply_package(source, approval=approval_path, target=target, dry_run=dry_run)
        output_path = output or (
            _source_run_dir(source) / "apply-result.json"
            if source.is_dir()
            else Path("apply-result.json")
        )
        write_apply_result(output_path, result)
        event_type = "proposal.conflict" if result.status == "conflict" else "proposal.applied"
        _append_package_event_if_run_dir(
            source,
            event_type,
            {
                "status": result.status,
                "approval_hash": result.approval_hash,
                "package_hash": result.package_hash,
                "applied_paths": list(result.applied_paths),
                "conflicts": [conflict.to_json() for conflict in result.conflicts],
            },
            level="warning" if result.status == "conflict" else "info",
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    payload = result.to_json()
    if json_output:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    else:
        click.echo(f"status: {payload['status']}")
        click.echo(f"apply_result: {output_path}")


@main.group()
def backend() -> None:
    """Run the reference Monoid backend."""


def _bind_or_report(
    build: Callable[[], Any],
    *,
    host: str,
    port: int,
    release: Callable[[], Any] | None = None,
) -> Any:
    """Bind a server socket, or fail the way every other startup failure in this CLI fails.

    Shared by all three ``serve`` commands. A bound port is the everyday failure of each of them
    and it happens *after* the thing being served is constructed, so a per-command `try` around
    `serve_forever` is one command too late: the constructed object goes to nobody and the
    operator gets a bare traceback. Written once because the first version of this was bound on
    `backend serve` alone while its two siblings, eighty and two hundred lines below, kept the
    behaviour the commit message said had been fixed.

    ``OverflowError`` is in the catch because `click`'s ``int`` accepts 99999 and the socket layer
    rejects it with that, not with ``OSError`` -- measured. ``release`` failing must not replace
    the bind failure the operator needs to read.
    """

    try:
        return build()
    except (OSError, OverflowError) as exc:
        if release is not None:
            try:
                release()
            except Exception as teardown:  # noqa: BLE001 - never replace the bind failure
                # Demoted to a warning beside the real error, the way `_adapter_teardown` handles
                # the same problem in this file. Fully silent, a failed release leaves a watchdog
                # thread and whatever it holds with nothing to say so.
                click.echo(f"warning: releasing the server failed: {teardown}", err=True)
        raise click.ClickException(f"could not listen on {host}:{port}: {exc}") from exc


@backend.command("serve")
@click.option("--host", type=str, default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8765, show_default=True)
@click.option(
    "--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True
)
@click.option(
    "--workspace-root",
    type=click.Path(path_type=Path),
    multiple=True,
    required=True,
    help="Allowed workspace root. Repeat for multiple roots.",
)
@click.option(
    "--apply-root",
    type=click.Path(path_type=Path),
    multiple=True,
    help="Allowed local reference apply root. Repeat for multiple roots.",
)
@click.option("--llm-gateway-url", type=str, required=True, help="Internal LLM gateway URL.")
@click.option(
    "--llm-gateway-provider",
    type=str,
    default=_DEFAULT_RELAYED_PROVIDER,
    show_default=True,
    help=(
        "Names the UPSTREAM provider that gateway relays, never the gateway hop itself. It tags "
        "the provider-native reasoning artifacts the gateway carries back (they only replay to a "
        "matching provider) and it is the provider attributed on the model-call receipt and "
        'OTel\'s gen_ai.provider.name. Pass "none" to disable tagging.'
    ),
)
@click.option("--web-gateway-url", type=str, default=None, help="Internal WebGateway base URL.")
@click.option(
    "--model-calls-file",
    is_flag=True,
    help="Record a per-call model ledger (model_calls.jsonl) in every run this backend serves.",
)
@click.option(
    "--model-payload-file",
    is_flag=True,
    help=(
        "Record the private replay corpus (model_payloads.jsonl plus its chunk directory) in "
        "every run this backend serves. Carries request and response content for every tenant, "
        "with no per-run override and no retention verb."
    ),
)
@click.option(
    "--model-content-file",
    is_flag=True,
    help=(
        "Record the private model-content sidecar (model-content.jsonl) in every run this "
        "backend serves. Content, and deployment-wide like the two above. Unlike them it also "
        "selects provider streaming, so every tenant's model call moves to the streaming "
        "dispatch."
    ),
)
@click.option(
    "--admin-token-env",
    type=str,
    default="MONOID_BACKEND_ADMIN_TOKEN",
    show_default=True,
    help="Environment variable containing the backend admin token.",
)
@click.option(
    "--token-secret-env",
    type=str,
    default="MONOID_BACKEND_TOKEN_SECRET",
    show_default=True,
    help="Environment variable containing a 32+ byte HMAC signing secret.",
)
@click.option(
    "--ephemeral-token-secret",
    is_flag=True,
    help="Use an in-memory signing secret for local development.",
)
def backend_serve(
    *,
    host: str,
    port: int,
    run_root: Path,
    workspace_root: tuple[Path, ...],
    apply_root: tuple[Path, ...],
    llm_gateway_url: str,
    llm_gateway_provider: str | None,
    web_gateway_url: str | None,
    model_calls_file: bool,
    model_payload_file: bool,
    model_content_file: bool,
    admin_token_env: str,
    token_secret_env: str,
    ephemeral_token_secret: bool,
) -> None:
    """Serve token issuance, run submission, status, result, and events APIs."""
    admin_token = getenv(admin_token_env)
    if not admin_token:
        raise click.ClickException(f"{env_name_for_error(admin_token_env)} is required")
    if ephemeral_token_secret:
        token_manager = TokenManager.ephemeral()
    else:
        signing_secret = getenv(token_secret_env)
        if not signing_secret:
            raise click.ClickException(
                f"{env_name_for_error(token_secret_env)} is required, or pass --ephemeral-token-secret for local development"
            )
        token_manager = TokenManager.from_secret(signing_secret)

    runner_backend = RunnerBackend(
        run_root=run_root,
        token_manager=token_manager,
        allowed_workspace_roots=workspace_root,
        allowed_apply_roots=apply_root,
        llm_gateway_url=llm_gateway_url,
        llm_gateway_provider=llm_gateway_provider,
        web_gateway_url=web_gateway_url,
        model_calls_file=model_calls_file,
        model_payload_file=model_payload_file,
        model_content_file=model_content_file,
    )
    server = _bind_or_report(
        lambda: create_backend_server(
            runner_backend, host=host, port=port, admin_token=admin_token
        ),
        host=host,
        port=port,
        release=runner_backend.shutdown,
    )
    click.echo(f"Monoid backend listening on http://{host}:{port}")
    click.echo(
        f"allowed workspace roots: {', '.join(str(path.resolve()) for path in workspace_root)}"
    )
    if apply_root:
        click.echo(f"allowed apply roots: {', '.join(str(path.resolve()) for path in apply_root)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("Monoid backend stopped")
    finally:
        server.server_close()
        # Stops this backend's watchdog and closes its live-stream broker, if it has either --
        # this command starts no watchdog and sets no broker, so today it is close to a no-op,
        # kept because both are backend-owned and a future flag could turn one on. What it
        # deliberately does NOT do: the run loop is process-shared and survives (stopping it here
        # would break other backends in the process), and `drain=False` leaves parked sessions on
        # it. Two earlier versions of this comment claimed otherwise in opposite directions.
        runner_backend.shutdown()


@main.group("llm-gateway")
def llm_gateway() -> None:
    """Run the standalone LLM gateway backend."""


@llm_gateway.command("serve")
@click.option("--host", type=str, default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8080, show_default=True)
@click.option(
    "--admin-token-env",
    type=str,
    default="MONOID_LLM_GATEWAY_ADMIN_TOKEN",
    show_default=True,
    help="Environment variable containing the LLM gateway admin token.",
)
@click.option(
    "--token-secret-env",
    type=str,
    default="MONOID_BACKEND_TOKEN_SECRET",
    show_default=True,
    help="Environment variable containing the shared 32+ byte HMAC signing secret.",
)
@click.option(
    "--ephemeral-token-secret",
    is_flag=True,
    help="Use an in-memory signing secret for local development.",
)
@click.option(
    "--provider",
    type=click.Choice(["openai", "fake"]),
    default="openai",
    show_default=True,
    help="openai = direct OpenAIModelAdapter (needs OPENAI_API_KEY); "
    "fake = key-less offline echo model for local development.",
)
def llm_gateway_serve(
    *,
    host: str,
    port: int,
    admin_token_env: str,
    token_secret_env: str,
    ephemeral_token_secret: bool,
    provider: str,
) -> None:
    """Serve the internal LLM turn API consumed by GatewayModelAdapter."""
    admin_token = getenv(admin_token_env)
    if not admin_token:
        raise click.ClickException(f"{env_name_for_error(admin_token_env)} is required")
    if ephemeral_token_secret:
        token_manager = TokenManager.ephemeral()
    else:
        signing_secret = getenv(token_secret_env)
        if not signing_secret:
            raise click.ClickException(
                f"{env_name_for_error(token_secret_env)} is required, or pass --ephemeral-token-secret for local development"
            )
        token_manager = TokenManager.from_secret(signing_secret)

    provider_factory = offline_provider_factory if provider == "fake" else None
    gateway = LlmGatewayBackend(
        token_manager=token_manager, provider_adapter_factory=provider_factory
    )
    server = _bind_or_report(
        lambda: create_llm_gateway_server(gateway, host=host, port=port, admin_token=admin_token),
        host=host,
        port=port,
    )
    click.echo(f"LLM gateway listening on http://{host}:{port}")
    click.echo("turn endpoint: /internal/llm/turns")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("LLM gateway stopped")
    finally:
        server.server_close()


@main.group("web-gateway")
def web_gateway() -> None:
    """Run the standalone reference WebGateway backend."""


@web_gateway.command("serve")
@click.option("--host", type=str, default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8090, show_default=True)
@click.option(
    "--provider",
    type=click.Choice(["fake", "brave-http"]),
    default="fake",
    show_default=True,
    help="Web provider implementation. brave-http uses Brave for search and direct HTTP for fetch.",
)
@click.option(
    "--context-provider",
    type=click.Choice(["none", "search-fetch", "brave-llm"]),
    default="none",
    show_default=True,
    help="Optional LLM context provider for /internal/web/context.",
)
@click.option(
    "--brave-api-key-env",
    type=str,
    default="BRAVE_SEARCH_API_KEY",
    show_default=True,
    help="Environment variable containing the Brave Search API key.",
)
@click.option("--brave-country", type=str, default="US", show_default=True)
@click.option("--brave-search-lang", type=str, default="en", show_default=True)
@click.option(
    "--brave-llm-context-endpoint",
    type=str,
    default="https://api.search.brave.com/res/v1/llm/context",
    show_default=True,
)
@click.option("--provider-timeout-s", type=int, default=10, show_default=True)
@click.option("--fetch-timeout-s", type=int, default=20, show_default=True)
@click.option("--fetch-max-raw-bytes", type=int, default=2_000_000, show_default=True)
@click.option("--fetch-user-agent", type=str, default=None)
@click.option(
    "--admin-token-env",
    type=str,
    default="MONOID_WEB_GATEWAY_ADMIN_TOKEN",
    show_default=True,
    help="Environment variable containing the WebGateway admin token.",
)
@click.option(
    "--token-secret-env",
    type=str,
    default="MONOID_BACKEND_TOKEN_SECRET",
    show_default=True,
    help="Environment variable containing the shared 32+ byte HMAC signing secret.",
)
@click.option(
    "--ephemeral-token-secret",
    is_flag=True,
    help="Use an in-memory signing secret for local development.",
)
def web_gateway_serve(
    *,
    host: str,
    port: int,
    provider: str,
    context_provider: str,
    brave_api_key_env: str,
    brave_country: str,
    brave_search_lang: str,
    brave_llm_context_endpoint: str,
    provider_timeout_s: int,
    fetch_timeout_s: int,
    fetch_max_raw_bytes: int,
    fetch_user_agent: str | None,
    admin_token_env: str,
    token_secret_env: str,
    ephemeral_token_secret: bool,
) -> None:
    """Serve the internal web.search/web.fetch/web.context API consumed by WebGatewayClient."""
    admin_token = getenv(admin_token_env)
    if not admin_token:
        raise click.ClickException(f"{env_name_for_error(admin_token_env)} is required")
    if ephemeral_token_secret:
        token_manager = TokenManager.ephemeral()
    else:
        signing_secret = getenv(token_secret_env)
        if not signing_secret:
            raise click.ClickException(
                f"{env_name_for_error(token_secret_env)} is required, or pass --ephemeral-token-secret for local development"
            )
        token_manager = TokenManager.from_secret(signing_secret)

    try:
        web_provider = _build_web_provider(
            provider,
            context_provider=context_provider,
            brave_api_key_env=brave_api_key_env,
            brave_country=brave_country,
            brave_search_lang=brave_search_lang,
            brave_llm_context_endpoint=brave_llm_context_endpoint,
            provider_timeout_s=provider_timeout_s,
            fetch_timeout_s=fetch_timeout_s,
            fetch_max_raw_bytes=fetch_max_raw_bytes,
            fetch_user_agent=fetch_user_agent,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    gateway = WebGatewayBackend(token_manager=token_manager, provider=web_provider)
    server = _bind_or_report(
        lambda: create_web_gateway_server(gateway, host=host, port=port, admin_token=admin_token),
        host=host,
        port=port,
    )
    click.echo(f"WebGateway listening on http://{host}:{port}")
    click.echo(f"provider: {provider}")
    click.echo(f"context provider: {context_provider}")
    click.echo("search endpoint: /internal/web/search")
    click.echo("fetch endpoint: /internal/web/fetch")
    click.echo("context endpoint: /internal/web/context")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("WebGateway stopped")
    finally:
        server.server_close()


def _adapter_teardown(closer: Any) -> Any:
    """Release the adapter, reporting a teardown failure without ever replacing a real one.

    Registered with ``ExitStack.push`` rather than ``callback`` so it can see whether an exception is
    already leaving the block. Raising from a cleanup callback *replaces* that exception: a failing
    ``close()`` superseded the run's own failure, and the provider error an operator actually needs
    disappeared behind "close() failed". Measured -- a run failed by a dead provider, reported only as
    a teardown error.

    With nothing else on its way out, the failure is still the command's error, because then it is the
    only signal there is. Alongside a real failure it is a footnote, on stderr so it neither
    supersedes the error nor lands in ``--stream-json`` output.
    """

    def _exit(exc_type: Any, exc: Any, tb: Any) -> bool:
        del exc, tb
        try:
            closer()
        except Exception as close_exc:
            message = f"model adapter close() failed: {close_exc}"
            if exc_type is not None:
                click.echo(f"warning: {message}", err=True)
                return False
            raise click.ClickException(message) from close_exc
        return False

    return _exit


def _human_echo(message: str, *, stream_json: bool) -> None:
    click.echo(message, err=stream_json)


def _build_web_provider(
    provider: str,
    *,
    context_provider: str,
    brave_api_key_env: str,
    brave_country: str,
    brave_search_lang: str,
    brave_llm_context_endpoint: str,
    provider_timeout_s: int,
    fetch_timeout_s: int,
    fetch_max_raw_bytes: int,
    fetch_user_agent: str | None,
):
    if provider == "fake":
        return FakeWebProvider()
    if provider == "brave-http":
        search_provider = BraveSearchProvider.from_env(
            api_key_env=brave_api_key_env,
            country=brave_country,
            search_lang=brave_search_lang,
            timeout_s=provider_timeout_s,
        )
        fetch_provider = HttpFetchProvider(
            timeout_s=fetch_timeout_s,
            max_raw_bytes=fetch_max_raw_bytes,
            user_agent=fetch_user_agent or user_agent("monoid-agent-kernel-webgateway"),
        )
        selected_context_provider = None
        if context_provider == "search-fetch":
            selected_context_provider = SearchFetchContextProvider(
                search_provider=search_provider,
                fetch_provider=fetch_provider,
            )
        elif context_provider == "brave-llm":
            selected_context_provider = BraveLlmContextProvider.from_env(
                api_key_env=brave_api_key_env,
                endpoint=brave_llm_context_endpoint,
                country=brave_country,
                search_lang=brave_search_lang,
                timeout_s=provider_timeout_s,
            )
        return CompositeWebProvider(
            search_provider=search_provider,
            fetch_provider=fetch_provider,
            context_provider=selected_context_provider,
        )
    raise ValueError(f"unsupported web provider: {provider}")


def _replay_preflight(
    corpus: ReplayCorpus,
    config: ModelConfig,
    adapter: Any,
    *,
    fallthrough: bool,
) -> None:
    """Refuse (or warn into) a run whose every lookup is already doomed.

    The replay key's model identity is authored by the RUN'S runtime config, not by the
    corpus (the loop always sets ``request.model``), so the most common total miss is a
    config that does not match what was recorded -- discoverable before the run starts, by
    the same comparison ``diagnose`` uses on a live miss (one function, two moments). A
    zero-identity match is a rejection in fail mode and a warning under
    ``--replay-fallthrough`` (an all-live run is a valid run); a partial match -- several
    recorded identities, at least one reachable -- warns, because the unreachable ones will
    miss and the operator should hear it here rather than at turn N.
    """

    model = normalize_model_config(config) or ModelConfig()
    provider = resolved_provider_name(adapter, model) or ""
    divergence = corpus.identity_divergence(model=_model_identity(model), provider=provider)
    if divergence is None:
        if len(corpus.identity_profiles()) > 1:
            click.echo(
                "warning: replay preflight: the corpus recorded "
                f"{len(corpus.identity_profiles())} model identities and this run's config "
                "reaches one of them; calls recorded under the others will miss",
                err=True,
            )
        return
    message = (
        "replay preflight: no recorded request matches this run's model identity -- "
        f"{divergence}"
    )
    if fallthrough:
        click.echo(f"warning: {message}", err=True)
        return
    raise click.ClickException(
        f"{message}. Fix the runtime config to match the recorded run, or pass "
        "--replay-fallthrough to serve the misses live."
    )


def _model_adapter(
    config: ModelConfig,
    *,
    llm_gateway_url: str | None,
    llm_gateway_token_env: str,
    llm_gateway_token_file: Path | None,
    llm_gateway_provider: str | None = _DEFAULT_RELAYED_PROVIDER,
    allow_direct_provider_api: bool,
) -> ModelAdapter:
    if config.provider == "gateway":
        return GatewayModelAdapter(
            config,
            gateway_url=llm_gateway_url,
            token_env=llm_gateway_token_env,
            token_file=llm_gateway_token_file,
            provider_name=resolve_relayed_provider(llm_gateway_provider),
        )
    if config.provider == "openai":
        if not allow_direct_provider_api:
            raise click.ClickException(
                "OpenAI runtime configs require --allow-direct-provider-api; "
                "container runs should use a gateway runtime config"
            )
        return OpenAIModelAdapter(config, allow_direct_provider_api=True)
    raise click.ClickException(f"unsupported model provider: {config.provider}")


def _load_agent_runtime_config(
    runtime_config_file: Path | None,
    agent_definition_file: Path | None,
) -> AgentRuntimeConfig:
    if runtime_config_file is not None and agent_definition_file is not None:
        raise click.ClickException(
            "--runtime-config-file and --agent-definition-file are mutually exclusive"
        )
    if runtime_config_file is None and agent_definition_file is None:
        raise click.ClickException("--runtime-config-file or --agent-definition-file is required")
    config_file = runtime_config_file or agent_definition_file
    assert config_file is not None
    try:
        payload = loads_json_ingress(config_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"invalid agent config JSON: {exc.msg}") from exc
    try:
        if runtime_config_file is not None:
            return AgentRuntimeConfig.from_json(payload)
        return AgentRuntimeConfig.from_definition(AgentDefinition.from_json(payload))
    except Exception as exc:
        raise click.ClickException(f"failed to load agent runtime config: {exc}") from exc


def _runtime_config_uses_web(config: AgentRuntimeConfig) -> bool:
    return any(binding.ref.tool_id.startswith("web.") for binding in config.tools)


def _load_permission_policy(
    policy_file: Path | None,
    *,
    deny_path: tuple[str, ...],
    redact_path: tuple[str, ...],
) -> PermissionPolicy:
    policy = PermissionPolicy()
    if policy_file is not None:
        try:
            payload = loads_json_ingress(policy_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid permission policy JSON: {exc.msg}") from exc
        policy = PermissionPolicy.from_json(payload)
    return policy.merged(deny_patterns=deny_path, redact_patterns=redact_path)


def _resolve_events_path(run_dir_or_id: str, run_root: Path) -> Path:
    return _resolve_run_dir(run_dir_or_id, run_root) / "events.jsonl"


def _resolve_run_dir(run_dir_or_id: str, run_root: Path) -> Path:
    candidate = Path(run_dir_or_id)
    return candidate if candidate.exists() else run_root / run_dir_or_id


def _resolve_package_source(package_or_run_dir: str, run_root: Path) -> Path:
    candidate = Path(package_or_run_dir)
    return candidate if candidate.exists() else run_root / package_or_run_dir


def _source_run_dir(source: Path) -> Path:
    return source.resolve()


def _append_package_event_if_run_dir(
    source: Path,
    event_type: str,
    data: dict[str, Any],
    *,
    level: str = "info",
) -> None:
    if source.is_dir():
        append_event_to_run(source.resolve(), event_type, data=data, level=level)


def _proposal_file_payload(
    run_dir: Path, proposal: dict[str, Any], file_path: str
) -> dict[str, Any]:
    try:
        return read_proposal_file_payload(run_dir, proposal, file_path)
    except ProposalFileError as exc:
        raise click.ClickException(str(exc)) from exc


def _compact_event_line(line: str) -> str:
    try:
        event = loads_json_ingress(line)
    except json.JSONDecodeError:
        return line.rstrip("\n")
    # Tool activity goes through the shared narration projection (the same one the Studio feed
    # uses), so the verb/target extraction lives in one place. Other events keep a generic dump.
    narration = narrate_event(event)
    if narration is not None:
        suffix = f" {narration.action}"
        if narration.target:
            suffix += f" {narration.target}"
        if narration.status == "error":
            suffix += f" [error: {narration.detail}]" if narration.detail else " [error]"
    else:
        data = event.get("data") or {}
        suffix = ""
        if "status" in data:
            suffix = f" status={data['status']}"
        elif "job_id" in data:
            suffix = f" job={data['job_id']}"
        elif "paths" in data:
            suffix = f" paths={','.join(map(str, data['paths']))}"
        elif "error" in data and data["error"]:
            suffix = f" error={data['error']}"
    return f"{event.get('seq', '?'):>4} {event.get('type', '?')}{suffix}"


if __name__ == "__main__":
    main()
