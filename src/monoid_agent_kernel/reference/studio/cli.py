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
import socket
import tempfile
import time
from pathlib import Path

import click

from monoid_agent_kernel.env import load_dotenv
from monoid_agent_kernel.reference.studio import window
from monoid_agent_kernel.reference.studio.server import _SAMPLE_SKILLS_DIR, StudioConfig, StudioServer
from monoid_agent_kernel.reference.studio.window import open_app_window


def _workspace_option(fn):
    return click.option(
        "--workspace",
        type=click.Path(path_type=Path),
        default=Path("studio-workspace"),
        show_default=True,
        help="Folder the agent works in (created if missing).",
    )(fn)


def _common_server_options(fn):
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
) -> StudioConfig:
    return StudioConfig(
        workspace=workspace,
        host=host,
        port=port,
        provider=provider,
        run_root=run_root,
        skills_directory=None if no_skills else skills_directory,
        mcp=mcp,
    )


def _load_studio_dotenv(provider: str) -> dict[str, str]:
    if provider != "openai":
        return {}
    return load_dotenv(".env", override=True, keys=("OPENAI_API_KEY",))


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
    open_window: bool,
) -> None:
    """Start the Studio server and keep it running (window is detachable)."""
    _load_studio_dotenv(provider)
    server = StudioServer(
        _studio_config(
            workspace=workspace, host=host, port=port, provider=provider, run_root=run_root,
            skills_directory=skills_directory, no_skills=no_skills, mcp=mcp,
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
) -> None:
    """Start the server and a desktop window; closing the window stops the server."""
    _load_studio_dotenv(provider)
    server = StudioServer(
        _studio_config(
            workspace=workspace, host=host, port=port, provider=provider, run_root=run_root,
            skills_directory=skills_directory, no_skills=no_skills, mcp=mcp,
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
) -> None:
    """Preflight the common setup failures and print pass/fail with exact remediation.

    Exits non-zero if a hard requirement fails (busy port, unwritable dir, missing API key),
    so it doubles as a CI/launch gate. Browser and OTel gaps are warnings — ``serve`` still runs."""
    hard_failures = 0
    loaded_dotenv = _load_studio_dotenv(provider)

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
        report(False, f"port {host}:{port} is in use", "stop the process using it or pass --port <other>")

    for label, directory in (("workspace", workspace), ("run root", run_root)):
        if _dir_writable(directory):
            report(True, f"{label} {directory} is writable")
        else:
            hard_failures += 1
            report(False, f"{label} {directory} is not writable", "pick a writable path")

    if provider == "openai":
        if os.environ.get("OPENAI_API_KEY"):
            source = " from .env" if "OPENAI_API_KEY" in loaded_dotenv else ""
            report(True, f"OPENAI_API_KEY is set{source}")
        else:
            hard_failures += 1
            report(False, "OPENAI_API_KEY is not set", "export OPENAI_API_KEY=... or use --provider offline")
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

    # --- soft checks (warnings only) ---
    if window.find_chromium() is not None:
        report(True, "a Chromium-family browser is available")
    else:
        report(None, "no Chromium browser found", "install Chrome/Edge, or use 'studio serve' and open the URL manually")

    if _otel_export_importable():
        report(True, "OpenTelemetry SDK + OTLP exporter are importable")
    else:
        report(None, "OTel export deps not installed", "pip install 'monoid-agent-kernel[otel-export]' (only needed for the OTel toggle)")

    click.echo("")
    if hard_failures:
        click.echo(f"{hard_failures} hard check(s) failed.")
        raise SystemExit(1)
    click.echo("All hard checks passed.")
