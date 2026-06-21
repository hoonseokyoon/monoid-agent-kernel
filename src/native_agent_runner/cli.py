from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import click

from native_agent_runner.core.agents import (
    AgentDefinition,
    AgentRuntimeConfig,
    StaticRuntimeConfigProvider,
)
from native_agent_runner.reference.backend.http import create_backend_server
from native_agent_runner.reference.backend.service import RunnerBackend
from native_agent_runner.reference._shared.tokens import TokenManager
from native_agent_runner.narration import narrate_event
from native_agent_runner.core.spec import (
    AgentRunSpec,
    ModelConfig,
    RunLimits,
)
from native_agent_runner.core.schemas import validate_run_dir
from native_agent_runner.core.packages import (
    apply_package,
    create_approval,
    export_package,
    import_package,
    inspect_package,
    verify_package,
    write_apply_result,
    write_approval,
)
from native_agent_runner.core.projections import project_run_status
from native_agent_runner.core.proposal_file import ProposalFileError, read_proposal_file_payload
from native_agent_runner.event_loader import load_event_sinks
from native_agent_runner.tasks import (
    get_job_artifact,
    list_job_artifacts,
    read_job_log_text,
    request_job_cancel,
)
from native_agent_runner.reference.llm_gateway.http import create_llm_gateway_server
from native_agent_runner.reference.llm_gateway.providers import offline_provider_factory
from native_agent_runner.reference.llm_gateway.service import LlmGatewayBackend
from native_agent_runner.loop import AgentLoop
from native_agent_runner.permissions import PermissionPolicy
from native_agent_runner.providers.base import ModelAdapter
from native_agent_runner.providers.gateway import GatewayModelAdapter
from native_agent_runner.providers.openai import OpenAIModelAdapter
from native_agent_runner.recorder import StdoutJsonlSink, append_event_to_run
from native_agent_runner.skills import SkillProvider, load_skill_definitions
from native_agent_runner.subagent_loader import load_subagent_definitions
from native_agent_runner.tool_loader import load_tool_provider
from native_agent_runner.web import WebGatewayClient
from native_agent_runner.reference.web_gateway.http import create_web_gateway_server
from native_agent_runner.reference.web_gateway.providers import (
    BraveLlmContextProvider,
    BraveSearchProvider,
    CompositeWebProvider,
    HttpFetchProvider,
    SearchFetchContextProvider,
)
from native_agent_runner.reference.web_gateway.service import FakeWebProvider, WebGatewayBackend
from native_agent_runner.reference.studio.cli import studio as studio_group


@click.group()
def main() -> None:
    """Run a standalone native agent harness."""


main.add_command(studio_group)


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
        "--no-status-file, --tool-module) still apply."
    ),
)
@click.option("--agent-definition-file", type=click.Path(path_type=Path), default=None)
@click.option("--runtime-config-file", type=click.Path(path_type=Path), default=None)
@click.option("--workspace", type=click.Path(path_type=Path), default=None)
@click.option("--instruction", type=str, default="")
@click.option("--instruction-file", type=click.Path(path_type=Path), default=None)
@click.option("--llm-gateway-url", type=str, default=None, help="Internal CSP LLM gateway URL.")
@click.option(
    "--llm-gateway-token-env",
    type=str,
    default="NAR_LLM_GATEWAY_TOKEN",
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
@click.option("--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True)
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
@click.option("--deny-path", multiple=True, help="Deny workspace paths matching a backend-provided glob.")
@click.option("--redact-path", multiple=True, help="Redact matching paths from public events and projections.")
@click.option("--permission-policy-file", type=click.Path(path_type=Path), default=None)
@click.option("--web-gateway-url", type=str, default=None, help="Internal CSP WebGateway base URL.")
@click.option(
    "--web-gateway-token-env",
    type=str,
    default="NAR_WEB_GATEWAY_TOKEN",
    show_default=True,
    help="Environment variable containing a short-lived WebGateway token.",
)
@click.option(
    "--web-gateway-token-file",
    type=click.Path(path_type=Path),
    default=None,
    help="File containing a short-lived WebGateway token.",
)
@click.option("--event-sink-module", multiple=True, help="Load custom event sinks from path.py:function.")
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
    deny_path: tuple[str, ...],
    redact_path: tuple[str, ...],
    permission_policy_file: Path | None,
    web_gateway_url: str | None,
    web_gateway_token_env: str,
    web_gateway_token_file: Path | None,
    event_sink_module: tuple[str, ...],
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
            raise click.ClickException("--spec cannot be combined with --workspace; the spec file is authoritative")
        try:
            spec = AgentRunSpec.from_json(json.loads(spec_file.read_text(encoding="utf-8")))
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
        raise click.ClickException(
            "runtime config binds web tools; --web-gateway-url is required"
        )
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
                subagent_definitions = {**subagent_definitions, **skill_provider.subagent_definitions()}
        extra_sinks = []
        if stream_json:
            extra_sinks.append(StdoutJsonlSink())
        for item in event_sink_module:
            extra_sinks.extend(load_event_sinks(item))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    result = AgentLoop(
        spec=spec,
        subagent_definitions=subagent_definitions,
        model_adapter=_model_adapter(
            runtime_config.model or ModelConfig(),
            llm_gateway_url=llm_gateway_url or (runtime_config.model.gateway_url if runtime_config.model else None),
            llm_gateway_token_env=llm_gateway_token_env,
            llm_gateway_token_file=llm_gateway_token_file,
            allow_direct_provider_api=allow_direct_provider_api,
        ),
        tool_providers=providers + ((skill_provider,) if skill_provider is not None else ()),
        context_providers=(skill_provider,) if skill_provider is not None else (),
        event_sinks=tuple(extra_sinks),
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
    _human_echo(f"status: {result.status}", stream_json=stream_json)
    if result.final_text:
        _human_echo(f"summary: {result.final_text}", stream_json=stream_json)
    if result.error:
        raise click.ClickException(result.error)


@main.command()
@click.argument("run_dir_or_id", type=str)
@click.option("--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True)
@click.option("--from-start", is_flag=True, help="Read events from the beginning of the file.")
@click.option("--follow", is_flag=True, help="Keep waiting for new events.")
@click.option("--json", "json_output", is_flag=True, help="Print raw JSONL events.")
def watch(run_dir_or_id: str, run_root: Path, from_start: bool, follow: bool, json_output: bool) -> None:
    """Watch a run's public events."""
    events_path = _resolve_events_path(run_dir_or_id, run_root)
    if not events_path.exists():
        raise click.ClickException(f"events.jsonl not found: {events_path}")

    start_from_beginning = from_start or not follow
    with events_path.open("r", encoding="utf-8") as handle:
        if not start_from_beginning:
            handle.seek(0, 2)
        while True:
            line = handle.readline()
            if line:
                click.echo(line.rstrip("\n") if json_output else _compact_event_line(line))
                continue
            if not follow:
                break
            time.sleep(0.25)


@main.command("status")
@click.argument("run_dir_or_id", type=str)
@click.option("--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def status_command(run_dir_or_id: str, run_root: Path, json_output: bool) -> None:
    """Project a run directory into compact status state."""
    run_dir = _resolve_run_dir(run_dir_or_id, run_root)
    payload = project_run_status(run_dir)
    if json_output:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    click.echo(f"run_id: {payload.get('run_id', '')}")
    click.echo(f"status: {payload.get('status', '')}")
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


@main.command("jobs")
@click.argument("run_dir_or_id", type=str)
@click.option("--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def jobs_command(run_dir_or_id: str, run_root: Path, json_output: bool) -> None:
    """List background shell jobs for a run."""
    run_dir = _resolve_run_dir(run_dir_or_id, run_root)
    payload = {"run_dir": str(run_dir), "jobs": list_job_artifacts(run_dir)}
    if json_output:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
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
@click.option("--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def job_status_command(job_id: str, run_dir_or_id: str, run_root: Path, json_output: bool) -> None:
    """Show one background job status."""
    run_dir = _resolve_run_dir(run_dir_or_id, run_root)
    payload = get_job_artifact(run_dir, job_id)
    if json_output:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
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
@click.option("--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True)
@click.option("--stream", "stream_name", type=click.Choice(["stdout", "stderr"]), default="stdout", show_default=True)
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
            click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        elif payload.get("content"):
            click.echo(payload["content"], nl=False)
        next_offset = int(payload.get("next_offset") or 0)
        if not follow:
            break
        time.sleep(0.5)


@job_group.command("cancel")
@click.argument("job_id", type=str)
@click.option("--run", "run_dir_or_id", type=str, required=True)
@click.option("--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def job_cancel_command(job_id: str, run_dir_or_id: str, run_root: Path, json_output: bool) -> None:
    """Request cancellation for one background job."""
    run_dir = _resolve_run_dir(run_dir_or_id, run_root)
    payload = request_job_cancel(run_dir, job_id)
    if json_output:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        click.echo(f"cancel_requested: {payload['job_id']}")


@main.command()
@click.argument("run_dir_or_id", type=str)
@click.option("--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True)
@click.option("--file", "file_path", type=str, default=None, help="Show one proposed file's snapshot content.")
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def proposal(run_dir_or_id: str, run_root: Path, file_path: str | None, json_output: bool) -> None:
    """Inspect a run's proposal snapshot."""
    run_dir = _resolve_run_dir(run_dir_or_id, run_root)
    proposal_path = run_dir / "proposal.json"
    if not proposal_path.exists():
        raise click.ClickException(f"proposal.json not found: {proposal_path}")
    payload = json.loads(proposal_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise click.ClickException("proposal.json must contain an object")
    if file_path is not None:
        file_payload = _proposal_file_payload(run_dir, payload, file_path)
        if json_output:
            click.echo(json.dumps(file_payload, ensure_ascii=False, sort_keys=True))
        else:
            click.echo(file_payload["content"])
        return
    if json_output:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
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
@click.option("--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True)
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
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    elif issues:
        for issue in issues:
            click.echo(f"{issue.path}: {issue.message}")
    else:
        click.echo("ok")
    if issues:
        raise click.ClickException("run directory validation failed")


@main.group("package")
def package_group() -> None:
    """Export, approve, and apply proposal packages."""


@package_group.command("export")
@click.argument("run_dir_or_id", type=str)
@click.option("--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True)
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
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        click.echo(f"package: {output}")
        click.echo(f"package_hash: {payload['package_hash']}")


@package_group.command("verify")
@click.argument("package_or_run_dir", type=str)
@click.option("--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True)
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
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
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
@click.option("--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def package_inspect(package_or_run_dir: str, run_root: Path, json_output: bool) -> None:
    """Inspect a proposal package summary."""
    source = _resolve_package_source(package_or_run_dir, run_root)
    payload = inspect_package(source)
    if json_output:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    click.echo(f"ok: {payload['ok']}")
    click.echo(f"package_hash: {payload.get('package', {}).get('package_hash', '')}")
    click.echo(f"changed_paths: {', '.join(map(str, payload.get('proposal', {}).get('changed_paths', [])))}")


@package_group.command("import")
@click.argument("package_or_run_dir", type=str)
@click.option("--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def package_import(package_or_run_dir: str, run_root: Path, output: Path, json_output: bool) -> None:
    """Import a proposal package into a verified staging directory."""
    source = _resolve_package_source(package_or_run_dir, run_root)
    try:
        payload = import_package(source, output)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    if json_output:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        click.echo(f"imported: {payload['output']}")
        click.echo(f"package_hash: {payload['package_hash']}")


@package_group.command("approve")
@click.argument("package_or_run_dir", type=str)
@click.option("--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True)
@click.option("--approver", type=str, required=True)
@click.option("--path", "approved_path", multiple=True, help="Approve one changed workspace path. Repeatable.")
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
        output_path = output or (_source_run_dir(source) / "approval.json" if source.is_dir() else Path("approval.json"))
        write_approval(output_path, approval)
        _append_package_event_if_run_dir(
            source,
            "proposal.approved",
            {"approval_hash": approval["approval_hash"], "package_hash": approval["package_hash"]},
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    if json_output:
        click.echo(json.dumps(approval, ensure_ascii=False, sort_keys=True))
    else:
        click.echo(f"approval: {output_path}")
        click.echo(f"approval_hash: {approval['approval_hash']}")


@package_group.command("reject")
@click.argument("package_or_run_dir", type=str)
@click.option("--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True)
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
        output_path = output or (_source_run_dir(source) / "approval.json" if source.is_dir() else Path("approval.json"))
        write_approval(output_path, approval)
        _append_package_event_if_run_dir(
            source,
            "proposal.rejected",
            {"approval_hash": approval["approval_hash"], "package_hash": approval["package_hash"]},
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    if json_output:
        click.echo(json.dumps(approval, ensure_ascii=False, sort_keys=True))
    else:
        click.echo(f"approval: {output_path}")
        click.echo(f"approval_hash: {approval['approval_hash']}")


@package_group.command("apply")
@click.argument("package_or_run_dir", type=str)
@click.option("--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True)
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
        output_path = output or (_source_run_dir(source) / "apply-result.json" if source.is_dir() else Path("apply-result.json"))
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
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        click.echo(f"status: {payload['status']}")
        click.echo(f"apply_result: {output_path}")


@main.group()
def backend() -> None:
    """Run the standalone runner backend."""


@backend.command("serve")
@click.option("--host", type=str, default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8765, show_default=True)
@click.option("--run-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True)
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
@click.option("--llm-gateway-url", type=str, required=True, help="Internal CSP LLM gateway URL.")
@click.option("--web-gateway-url", type=str, default=None, help="Internal CSP WebGateway base URL.")
@click.option(
    "--admin-token-env",
    type=str,
    default="NAR_BACKEND_ADMIN_TOKEN",
    show_default=True,
    help="Environment variable containing the backend admin token.",
)
@click.option(
    "--token-secret-env",
    type=str,
    default="NAR_BACKEND_TOKEN_SECRET",
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
    web_gateway_url: str | None,
    admin_token_env: str,
    token_secret_env: str,
    ephemeral_token_secret: bool,
) -> None:
    """Serve token issuance, run submission, status, result, and events APIs."""
    admin_token = os.environ.get(admin_token_env)
    if not admin_token:
        raise click.ClickException(f"{admin_token_env} is required")
    if ephemeral_token_secret:
        token_manager = TokenManager.ephemeral()
    else:
        signing_secret = os.environ.get(token_secret_env)
        if not signing_secret:
            raise click.ClickException(
                f"{token_secret_env} is required, or pass --ephemeral-token-secret for local development"
            )
        token_manager = TokenManager.from_secret(signing_secret)

    runner_backend = RunnerBackend(
        run_root=run_root,
        token_manager=token_manager,
        allowed_workspace_roots=workspace_root,
        allowed_apply_roots=apply_root,
        llm_gateway_url=llm_gateway_url,
        web_gateway_url=web_gateway_url,
    )
    server = create_backend_server(runner_backend, host=host, port=port, admin_token=admin_token)
    click.echo(f"runner backend listening on http://{host}:{port}")
    click.echo(f"allowed workspace roots: {', '.join(str(path.resolve()) for path in workspace_root)}")
    if apply_root:
        click.echo(f"allowed apply roots: {', '.join(str(path.resolve()) for path in apply_root)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("runner backend stopped")
    finally:
        server.server_close()
        runner_backend.shutdown()  # stop the shared run loop + watchdog


@main.group("llm-gateway")
def llm_gateway() -> None:
    """Run the standalone LLM gateway backend."""


@llm_gateway.command("serve")
@click.option("--host", type=str, default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8080, show_default=True)
@click.option(
    "--admin-token-env",
    type=str,
    default="NAR_LLM_GATEWAY_ADMIN_TOKEN",
    show_default=True,
    help="Environment variable containing the LLM gateway admin token.",
)
@click.option(
    "--token-secret-env",
    type=str,
    default="NAR_BACKEND_TOKEN_SECRET",
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
    admin_token = os.environ.get(admin_token_env)
    if not admin_token:
        raise click.ClickException(f"{admin_token_env} is required")
    if ephemeral_token_secret:
        token_manager = TokenManager.ephemeral()
    else:
        signing_secret = os.environ.get(token_secret_env)
        if not signing_secret:
            raise click.ClickException(
                f"{token_secret_env} is required, or pass --ephemeral-token-secret for local development"
            )
        token_manager = TokenManager.from_secret(signing_secret)

    provider_factory = offline_provider_factory if provider == "fake" else None
    gateway = LlmGatewayBackend(token_manager=token_manager, provider_adapter_factory=provider_factory)
    server = create_llm_gateway_server(gateway, host=host, port=port, admin_token=admin_token)
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
    default="NAR_WEB_GATEWAY_ADMIN_TOKEN",
    show_default=True,
    help="Environment variable containing the WebGateway admin token.",
)
@click.option(
    "--token-secret-env",
    type=str,
    default="NAR_BACKEND_TOKEN_SECRET",
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
    admin_token = os.environ.get(admin_token_env)
    if not admin_token:
        raise click.ClickException(f"{admin_token_env} is required")
    if ephemeral_token_secret:
        token_manager = TokenManager.ephemeral()
    else:
        signing_secret = os.environ.get(token_secret_env)
        if not signing_secret:
            raise click.ClickException(
                f"{token_secret_env} is required, or pass --ephemeral-token-secret for local development"
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
    server = create_web_gateway_server(gateway, host=host, port=port, admin_token=admin_token)
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
                user_agent=fetch_user_agent or "native-agent-runner-webgateway/0.11",
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


def _model_adapter(
    config: ModelConfig,
    *,
    llm_gateway_url: str | None,
    llm_gateway_token_env: str,
    llm_gateway_token_file: Path | None,
    allow_direct_provider_api: bool,
) -> ModelAdapter:
    if config.provider == "gateway":
        return GatewayModelAdapter(
            config,
            gateway_url=llm_gateway_url,
            token_env=llm_gateway_token_env,
            token_file=llm_gateway_token_file,
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
        raise click.ClickException("--runtime-config-file and --agent-definition-file are mutually exclusive")
    if runtime_config_file is None and agent_definition_file is None:
        raise click.ClickException("--runtime-config-file or --agent-definition-file is required")
    config_file = runtime_config_file or agent_definition_file
    assert config_file is not None
    try:
        payload = json.loads(config_file.read_text(encoding="utf-8"))
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
            payload = json.loads(policy_file.read_text(encoding="utf-8"))
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


def _proposal_file_payload(run_dir: Path, proposal: dict[str, Any], file_path: str) -> dict[str, Any]:
    try:
        return read_proposal_file_payload(run_dir, proposal, file_path)
    except ProposalFileError as exc:
        raise click.ClickException(str(exc)) from exc


def _compact_event_line(line: str) -> str:
    try:
        event = json.loads(line)
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
