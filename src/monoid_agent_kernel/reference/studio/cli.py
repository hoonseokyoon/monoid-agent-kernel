"""``monoid studio`` — run the bundled reference agent app.

Three launch shapes, matching the two lifecycle models:

* ``studio serve`` — start the server and keep it running (no window, or ``--open`` once).
  The window is detachable: re-open it any time with ``studio open``. Ctrl-C stops the server.
* ``studio app`` — start the server *and* a desktop window bound together; closing the window
  stops the server. This is the "double-click the app" shape.
* ``studio open`` — open a window pointing at an already-running ``studio serve`` server.
"""

from __future__ import annotations

import os
import json
import socket
import tempfile
import time
from pathlib import Path
from urllib.parse import quote
from urllib import request as urlrequest

import click

from monoid_agent_kernel.core.json_ingress import loads_json_ingress

from monoid_agent_kernel.core.model_io import content_digest
from monoid_agent_kernel.env import OUTPUT_DELTAS_ENV, getenv_bool
from monoid_agent_kernel.reference.studio import window
from monoid_agent_kernel.reference.studio.chat_projection import (
    CHAT_SCHEMA_VERSION,
    is_supported_chat_response,
)
from monoid_agent_kernel.reference.studio.server import (
    _SAMPLE_SKILLS_DIR,
    StudioConfig,
    StudioServer,
    _gateway_streaming_available,
    load_env_file,
)
from monoid_agent_kernel.reference.studio.window import open_app_window


def _http_json(
    url: str, *, method: str = "GET", payload: dict | None = None, timeout: float = 5.0
) -> dict:
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
    req = urlrequest.Request(url, data=data, method=method)
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    if not raw:
        return {}
    payload = loads_json_ingress(raw)
    if not isinstance(payload, dict):
        raise ValueError("Studio response must be a JSON object")
    return payload


def _http_text(url: str, *, timeout: float = 5.0) -> str:
    with urlrequest.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def run_acceptance(
    *,
    workspace: Path,
    run_root: Path,
    host: str = "127.0.0.1",
    timeout_s: float = 10.0,
) -> dict:
    """Run Studio's deterministic offline acceptance check and return a JSON-serializable result."""
    server = StudioServer(
        StudioConfig(
            workspace=workspace,
            host=host,
            port=0,
            provider="offline",
            run_root=run_root,
        )
    )
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), **({"detail": detail} if detail else {})})

    try:
        base_url = server.start()
        check("healthz", _http_json(f"{base_url}/healthz").get("ok") is True)
        index_html = _http_text(f"{base_url}/")
        check(
            "index-static-shell",
            '<div id="app"></div>' in index_html and "/assets/" in index_html,
        )
        settings_html = _http_text(f"{base_url}/settings")
        check(
            "settings-static-shell",
            '<div id="app"></div>' in settings_html and "/assets/" in settings_html,
        )
        cfg = _http_json(f"{base_url}/api/config")
        check("config-route", cfg.get("offline") is True and cfg.get("provider") == "offline")
        settings = _http_json(f"{base_url}/api/settings")
        check(
            "settings-route",
            bool(settings.get("available")) and "read" in settings.get("capabilities", []),
        )
        catalog = _http_json(f"{base_url}/api/capabilities-catalog")
        check("capabilities-catalog-route", "skills" in catalog and "mcp_tools" in catalog)
        profiles = _http_json(f"{base_url}/api/profiles")
        default_profile = str(profiles.get("default_profile_id") or "default")
        check(
            "profiles-route",
            any(p.get("id") == default_profile for p in profiles.get("profiles", [])),
        )
        before_sessions = _http_json(f"{base_url}/api/sessions?profile_id={default_profile}")
        check("profile-sessions-route", before_sessions.get("profile_id") == default_profile)
        chat = _http_json(
            f"{base_url}/api/chat",
            method="POST",
            payload={"message": "Studio acceptance ping", "profile_id": default_profile},
        )
        run_id = str(chat.get("run_id") or "")
        check("chat-start", bool(run_id) and "run_token" not in chat)
        deadline = time.time() + timeout_s
        final_text = ""
        settled_digest = ""
        state = ""
        while run_id and time.time() < deadline:
            events = server.poll_events(run_id, 0).get("events", [])
            settled = [event for event in events if event.get("type") == "turn.settled"]
            if settled:
                settled_data = settled[-1].get("data") or {}
                final_text = str(settled_data.get("final_text") or "")
                settled_digest = str(settled_data.get("final_text_digest") or "")
                state = str(server.run_status(run_id).get("state") or "")
                if state != "running":
                    break
            time.sleep(0.1)
        check("deterministic-chat", bool(final_text), final_text[:120])
        # Both halves of the settled-text join, from the one surface that sees them together.
        # ``poll_events`` reads through the hydration seam, so a settle event arriving with *both* a
        # digest and matching text proves the emit side published the digest AND the reader joined
        # the transcript record back. Checking only ``deterministic-chat`` above cannot tell a
        # working join from a flip that never happened; checking only the digest cannot tell a
        # published digest from an unresolvable one.
        check(
            "settled-text-digest",
            bool(settled_digest) and settled_digest == content_digest(final_text),
            settled_digest[:16],
        )
        transcript = (
            _http_json(f"{base_url}/api/chat-transcript?run_id={quote(run_id)}") if run_id else {}
        )
        transcript_messages = (
            transcript.get("messages") if isinstance(transcript.get("messages"), list) else []
        )
        transcript_roles = [str(message.get("role") or "") for message in transcript_messages]
        check(
            "chat-transcript",
            transcript.get("schema_version") == CHAT_SCHEMA_VERSION
            and is_supported_chat_response(transcript)
            and transcript.get("run_id") == run_id
            and transcript_roles[:2] == ["user", "assistant"],
            ",".join(transcript_roles),
        )
        scoped_sessions = _http_json(f"{base_url}/api/sessions?profile_id={default_profile}")
        check(
            "profile-history",
            any(
                s.get("run_id") == run_id and s.get("profile_id") == default_profile
                for s in scoped_sessions.get("sessions", [])
            ),
        )
        ok = all(item["ok"] for item in checks)
        return {
            "ok": ok,
            "base_url": base_url,
            "workspace": str(server.workspace),
            "run_root": str(run_root),
            "checks": checks,
            "chat": {
                "run_id": run_id,
                "state": state,
                "final_text": final_text,
                "transcript_messages": len(transcript_messages),
            },
        }
    except Exception as exc:  # pragma: no cover - defensive CLI surface
        checks.append({"name": "acceptance", "ok": False, "detail": str(exc)})
        return {
            "ok": False,
            "base_url": server.base_url,
            "workspace": str(workspace),
            "run_root": str(run_root),
            "checks": checks,
            "chat": {},
        }
    finally:
        server.shutdown()


def _workspace_option(fn):
    return click.option(
        "--workspace",
        type=click.Path(path_type=Path),
        default=Path("studio-workspace"),
        show_default=True,
        help="Folder the agent works in (created if missing).",
    )(fn)


def _common_server_options(fn):
    fn = click.option(
        "--no-output-deltas",
        is_flag=True,
        default=False,
        help=(
            "Stop publishing model.output.delta / model.reasoning.delta to events.jsonl. "
            "The run result and transcript.jsonl are unaffected. Costs live token rendering, and "
            "makes Stop wait for the in-flight model call instead of aborting mid-token. "
            "MONOID_OUTPUT_DELTAS=0 does the same for every run in a deployment."
        ),
    )(fn)
    fn = click.option("--host", type=str, default="127.0.0.1", show_default=True)(fn)
    fn = click.option("--port", type=int, default=8799, show_default=True)(fn)
    fn = click.option(
        "--provider",
        type=click.Choice(["offline", "openai"]),
        default="offline",
        show_default=True,
        help="offline = key-less echo model; openai = OpenAIModelAdapter (needs OPENAI_API_KEY).",
    )(fn)
    fn = click.option(
        "--run-root",
        type=click.Path(path_type=Path),
        default=Path("runs"),
        show_default=True,
    )(fn)
    fn = click.option(
        "--skills-directory",
        type=click.Path(path_type=Path),
        default=_SAMPLE_SKILLS_DIR,
        show_default="bundled sample skill",
        help="Directory of Agent Skills (SKILL.md files). Defaults to a bundled sample.",
    )(fn)
    fn = click.option("--no-skills", is_flag=True, help="Disable Agent Skills entirely.")(fn)
    fn = click.option(
        "--mcp",
        is_flag=True,
        help="Attach the bundled offline reference MCP server and expose its tools.",
    )(fn)
    fn = click.option(
        "--no-env-file",
        is_flag=True,
        help="Do not load a local env file before starting or checking the provider.",
    )(fn)
    fn = click.option(
        "--env-file",
        type=click.Path(path_type=Path),
        default=Path(".env"),
        show_default=True,
        help="Env file loaded without overriding existing environment variables.",
    )(fn)
    fn = _workspace_option(fn)
    return fn


def _studio_config(
    *,
    workspace: Path,
    host: str,
    port: int,
    provider: str,
    run_root: Path,
    skills_directory: Path,
    no_skills: bool,
    mcp: bool,
    env_file: Path,
    no_env_file: bool,
    no_output_deltas: bool = False,
) -> StudioConfig:
    return StudioConfig(
        workspace=workspace,
        host=host,
        port=port,
        provider=provider,
        run_root=run_root,
        skills_directory=None if no_skills else skills_directory,
        mcp=mcp,
        env_file=None if no_env_file else env_file,
        stream_output_deltas=not no_output_deltas,
    )


@click.group("studio")
def studio() -> None:
    """Run the bundled Studio reference app (LLM gateway + Monoid backend + UI)."""


@studio.command("serve")
@_common_server_options
@click.option("--open", "open_window", is_flag=True, help="Open a window once after starting.")
def studio_serve(
    *,
    workspace: Path,
    host: str,
    port: int,
    provider: str,
    run_root: Path,
    skills_directory: Path,
    no_skills: bool,
    mcp: bool,
    env_file: Path,
    no_env_file: bool,
    no_output_deltas: bool,
    open_window: bool,
) -> None:
    """Start the Studio server and keep it running (window is detachable)."""
    server = StudioServer(
        _studio_config(
            workspace=workspace,
            host=host,
            port=port,
            provider=provider,
            run_root=run_root,
            skills_directory=skills_directory,
            no_skills=no_skills,
            mcp=mcp,
            env_file=env_file,
            no_env_file=no_env_file,
            no_output_deltas=no_output_deltas,
        )
    )
    url = server.start()
    click.echo(f"Agent Studio serving on {url}  (workspace: {server.workspace})")
    click.echo(f"Open a window any time with:  monoid studio open --url {url}")
    if open_window:
        if open_app_window(url) is None:
            click.echo("No Chromium browser found; open the URL above in your browser.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        click.echo("Studio stopped")
    finally:
        server.shutdown()


@studio.command("app")
@_common_server_options
def studio_app(
    *,
    workspace: Path,
    host: str,
    port: int,
    provider: str,
    run_root: Path,
    skills_directory: Path,
    no_skills: bool,
    mcp: bool,
    env_file: Path,
    no_env_file: bool,
    no_output_deltas: bool,
) -> None:
    """Start the server and a desktop window; closing the window stops the server."""
    server = StudioServer(
        _studio_config(
            workspace=workspace,
            host=host,
            port=port,
            provider=provider,
            run_root=run_root,
            skills_directory=skills_directory,
            no_skills=no_skills,
            mcp=mcp,
            env_file=env_file,
            no_env_file=no_env_file,
            no_output_deltas=no_output_deltas,
        )
    )
    url = server.start()
    click.echo(f"Agent Studio app on {url}  (workspace: {server.workspace})")
    window = open_app_window(url)
    try:
        if window is None:
            click.echo("No Chromium browser found; serving headless. Ctrl-C to stop.")
            while True:
                time.sleep(3600)
        else:
            window.wait()  # block until the window is closed
            click.echo("Window closed; stopping Studio")
    except KeyboardInterrupt:
        click.echo("Studio stopped")
    finally:
        server.shutdown()


@studio.command("open")
@click.option("--url", type=str, default="http://127.0.0.1:8799", show_default=True)
def studio_open(*, url: str) -> None:
    """Open a window pointing at an already-running Studio server."""
    window = open_app_window(url)
    if window is None:
        raise click.ClickException(f"No Chromium browser found; open {url} manually.")
    window.wait()


@studio.command("settings")
@click.option("--url", type=str, default="http://127.0.0.1:8799", show_default=True)
def studio_settings(*, url: str) -> None:
    """Open the small Settings window for an already-running Studio server."""
    win = open_app_window(url.rstrip("/") + "/settings", width=520, height=660)
    if win is None:
        raise click.ClickException(f"No Chromium browser found; open {url}/settings manually.")
    win.wait()


@studio.command("accept")
@click.option("--host", type=str, default="127.0.0.1", show_default=True)
@click.option(
    "--run-root",
    type=click.Path(path_type=Path),
    default=Path("runs/studio-acceptance"),
    show_default=True,
)
@click.option("--timeout", "timeout_s", type=float, default=10.0, show_default=True)
@_workspace_option
def studio_accept(
    *,
    workspace: Path,
    host: str,
    run_root: Path,
    timeout_s: float,
) -> None:
    """Run deterministic offline Studio acceptance checks and print JSON."""
    result = run_acceptance(
        workspace=workspace,
        run_root=run_root,
        host=host,
        timeout_s=timeout_s,
    )
    click.echo(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if not result.get("ok"):
        raise SystemExit(1)


def _port_free(host: str, port: int) -> bool:
    """True if ``host:port`` can be bound (i.e. is free). Port 0 is always free (ephemeral)."""
    if port == 0:
        return True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _dir_writable(path: Path) -> bool:
    """True if ``path`` exists-or-can-be-created and a file can be written there. Uses a unique
    temp file (O_EXCL) so the diagnostic can never clobber an existing user file."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        fd, probe = tempfile.mkstemp(prefix=".nar-doctor-", dir=path)
        os.close(fd)
        os.unlink(probe)
        return True
    except OSError:
        return False


def _otel_export_importable() -> bool:
    """True if the OTel SDK + OTLP/HTTP exporter (the [otel-export] extra) are importable —
    the same imports _ensure_otel_provider needs for the Studio OTel toggle."""
    try:
        from opentelemetry import trace  # noqa: F401
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: F401
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource  # noqa: F401
        from opentelemetry.sdk.trace import TracerProvider  # noqa: F401
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: F401
    except ImportError:
        return False
    return True


def _openai_sdk_importable() -> bool:
    """True if the installed ``openai`` SDK exposes the exact surface OpenAIModelAdapter uses:
    the ``OpenAI``/``AsyncOpenAI`` clients and the Responses API (``client.responses.create``).
    A bare ``import openai`` succeeds on legacy versions that predate the Responses API, so the
    adapter would still fail on the first turn — probe the real symbols, not just the package.
    ``responses`` is a ``cached_property`` on the client class, so ``hasattr`` on the class sees
    it without needing an API key to instantiate."""
    try:
        from openai import AsyncOpenAI, OpenAI
    except ImportError:
        return False
    return hasattr(OpenAI, "responses") and hasattr(AsyncOpenAI, "responses")


@studio.command("doctor")
@_common_server_options
def studio_doctor(
    *,
    workspace: Path,
    host: str,
    port: int,
    provider: str,
    run_root: Path,
    skills_directory: Path,
    no_skills: bool,
    mcp: bool,
    env_file: Path,
    no_env_file: bool,
    no_output_deltas: bool,
) -> None:
    """Preflight the common setup failures and print pass/fail with exact remediation.

    Exits non-zero if a hard requirement fails (busy port, unwritable dir, missing API key),
    so it doubles as a CI/launch gate. Browser and OTel gaps are warnings — ``serve`` still runs."""
    hard_failures = 0
    loaded_env = load_env_file(None if no_env_file else env_file)

    def report(status: bool | None, label: str, remedy: str = "") -> None:
        mark = {True: "PASS", False: "FAIL", None: "WARN"}[status]
        click.echo(f"[{mark}] {label}")
        if remedy and status is not True:
            click.echo(f"       -> {remedy}")

    # --- hard requirements ---
    if _port_free(host, port):
        report(True, f"port {host}:{port} is free")
    else:
        hard_failures += 1
        report(
            False,
            f"port {host}:{port} is in use",
            "stop the process using it or pass --port <other>",
        )

    for label, directory in (("workspace", workspace), ("run root", run_root)):
        if _dir_writable(directory):
            report(True, f"{label} {directory} is writable")
        else:
            hard_failures += 1
            report(False, f"{label} {directory} is not writable", "pick a writable path")

    if provider == "openai":
        if os.environ.get("OPENAI_API_KEY"):
            source = f" from {env_file}" if "OPENAI_API_KEY" in loaded_env else ""
            report(True, f"OPENAI_API_KEY is set{source}")
        else:
            hard_failures += 1
            report(
                False,
                "OPENAI_API_KEY is not set",
                "export OPENAI_API_KEY=... or use --provider offline",
            )
        if _openai_sdk_importable():
            report(True, "the openai SDK is installed")
        else:
            hard_failures += 1
            report(
                False,
                "the openai SDK is not installed",
                "pip install 'monoid-agent-kernel[openai]' or use --provider offline",
            )
    else:
        report(True, "provider 'offline' (no API key needed)")

    # A hard check, not a warning: a malformed value here is a startup error by design, so leaving
    # it out meant `doctor` could report every hard requirement passing and `serve` still die in
    # `AgentLoop.__post_init__` on the next command. `doctor` loads the same `.env` this reads from,
    # which is exactly where the typo lives.
    #
    # The effective state is reported even when the value parses, because this switch's failure mode
    # is silent: an operator who believes they disabled a channel that publishes raw model text, and
    # did not, learns nothing from "PASS". The env var can only turn deltas off — it is ANDed with
    # the loop's own setting — so both inputs are named.
    try:
        deltas_env_permits = getenv_bool(OUTPUT_DELTAS_ENV, default=True)
    except ValueError as exc:
        hard_failures += 1
        report(
            False,
            f"{OUTPUT_DELTAS_ENV} is set to a value that is not a boolean",
            str(exc),
        )
    else:
        # Three inputs decide this, and `StudioServer.start` ANDs all three:
        # `_gateway_streaming_available() and config.stream_output_deltas`, with the loop applying
        # the env var on top. Reporting on two of them told a minimal install -- no `httpx`, which
        # is the base package's own default -- that raw model text "will be published" when the
        # server was about to use one-shot turns and publish none. A preflight that names the wrong
        # cause is worse than one that stays quiet: it sends someone to disable a switch that was
        # never the reason.
        if not deltas_env_permits:
            report(True, f"model-text deltas are disabled by {OUTPUT_DELTAS_ENV}")
        elif no_output_deltas:
            report(True, "model-text deltas are disabled by --no-output-deltas")
        elif not _gateway_streaming_available():
            report(
                True,
                "model-text deltas are off because the async transport is not installed "
                "(Studio uses one-shot turns without the [http-async] extra)",
            )
        else:
            report(
                True,
                "model.output.delta / model.reasoning.delta will be published to events.jsonl "
                "(these carry raw model text)",
            )

    # --- soft checks (warnings only) ---
    if window.find_chromium() is not None:
        report(True, "a Chromium-family browser is available")
    else:
        report(
            None,
            "no Chromium browser found",
            "install Chrome/Edge, or use 'studio serve' and open the URL manually",
        )

    if _otel_export_importable():
        report(True, "OpenTelemetry SDK + OTLP exporter are importable")
    else:
        report(
            None,
            "OTel export deps not installed",
            "pip install 'monoid-agent-kernel[otel-export]' (only needed for the OTel toggle)",
        )

    click.echo("")
    if hard_failures:
        click.echo(f"{hard_failures} hard check(s) failed.")
        raise SystemExit(1)
    click.echo("All hard checks passed.")
