"""``native-agent studio`` — run the bundled reference agent app.

Three launch shapes, matching the two lifecycle models:

* ``studio serve`` — start the server and keep it running (no window, or ``--open`` once).
  The window is detachable: re-open it any time with ``studio open``. Ctrl-C stops the server.
* ``studio app`` — start the server *and* a desktop window bound together; closing the window
  stops the server. This is the "double-click the app" shape.
* ``studio open`` — open a window pointing at an already-running ``studio serve`` server.
"""

from __future__ import annotations

import time
from pathlib import Path

import click

from native_agent_runner.reference.studio.server import StudioConfig, StudioServer
from native_agent_runner.reference.studio.window import open_app_window


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
    fn = _workspace_option(fn)
    return fn


@click.group("studio")
def studio() -> None:
    """Run the bundled Studio reference app (LLM gateway + runner backend + UI)."""


@studio.command("serve")
@_common_server_options
@click.option("--open", "open_window", is_flag=True, help="Open a window once after starting.")
def studio_serve(
    *, workspace: Path, host: str, port: int, provider: str, run_root: Path, open_window: bool
) -> None:
    """Start the Studio server and keep it running (window is detachable)."""
    server = StudioServer(StudioConfig(workspace=workspace, host=host, port=port,
                                       provider=provider, run_root=run_root))
    url = server.start()
    click.echo(f"Agent Studio serving on {url}  (workspace: {server.workspace})")
    click.echo(f"Open a window any time with:  native-agent studio open --url {url}")
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
    *, workspace: Path, host: str, port: int, provider: str, run_root: Path
) -> None:
    """Start the server and a desktop window; closing the window stops the server."""
    server = StudioServer(StudioConfig(workspace=workspace, host=host, port=port,
                                       provider=provider, run_root=run_root))
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
    window = open_app_window(url.rstrip("/") + "/settings", width=520, height=660)
    if window is None:
        raise click.ClickException(f"No Chromium browser found; open {url}/settings manually.")
    window.wait()
