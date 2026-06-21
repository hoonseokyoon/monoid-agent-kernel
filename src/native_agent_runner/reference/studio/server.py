"""Studio all-in-one server: LLM gateway + runner backend + UI/BFF in one process.

This is the reference "installable agent app" — it boots the reference ``LlmGatewayBackend`` and
``RunnerBackend`` behind a shared signing secret, then serves a single-page UI plus a thin
backend-for-frontend (BFF). The browser talks only to the BFF; run tokens and the admin token
stay server-side. The browser never sees a provider key.

Topology inside one process:

    browser ── HTTP ──> Studio BFF ──(Python calls)──> RunnerBackend ──(loopback HTTP)──> LLM gateway

The runner is driven via its Python API (not its own HTTP surface), which is the most
representative path for an embedder bundling the engine into an app.

Not part of the supported surface — a reference example you copy and own.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from native_agent_runner.core.agents import (
    AgentRuntimeConfig,
    RegistryToolRef,
    ToolBinding,
)
from native_agent_runner.errors import NativeAgentError
from native_agent_runner.reference._shared.tokens import TokenManager
from native_agent_runner.reference.backend.service import BackendRunRequest, RunnerBackend
from native_agent_runner.reference.llm_gateway.http import create_llm_gateway_server
from native_agent_runner.reference.llm_gateway.providers import offline_provider_factory
from native_agent_runner.reference.llm_gateway.service import LlmGatewayBackend
from native_agent_runner.reference.studio.activity import describe_event

_LOGGER = logging.getLogger("native_agent_runner.studio")

_WEB_DIR = Path(__file__).parent / "web"
_MAX_BODY_BYTES = 1_000_000
# Studio is a single-user local app; the tenant/user are fixed placeholders.
_TENANT = "studio"
_USER = "local"

# Directories never shown in the file tree (and not worth walking).
_TREE_SKIP = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
_TREE_MAX_ENTRIES = 2000


def _read_runtime_config() -> AgentRuntimeConfig:
    """R1 capability set: read-only filesystem access (chat + read, no mutation yet)."""
    return AgentRuntimeConfig(
        definition_id="studio-agent",
        tools=(
            ToolBinding(binding_id="fs.read", model_name="fs_read", ref=RegistryToolRef("fs.read")),
        ),
    )


@dataclass
class StudioConfig:
    workspace: Path
    host: str = "127.0.0.1"
    port: int = 8799
    # "offline" -> echo model (no key). "openai" -> reference OpenAIModelAdapter (needs OPENAI_API_KEY).
    provider: str = "offline"
    run_root: Path = field(default_factory=lambda: Path("runs"))


class StudioServer:
    """Boots the bundled stack and serves the UI + BFF. Use :meth:`start` / :meth:`shutdown`."""

    def __init__(self, config: StudioConfig, *, provider_factory: Any = None) -> None:
        self.config = config
        self.workspace = config.workspace.resolve()
        # Optional override for the gateway's model provider (an embedder seam; tests inject a
        # tool-calling fake here). Defaults to the offline/openai choice in config.provider.
        self._provider_factory_override = provider_factory
        self._token_manager = TokenManager.ephemeral()
        self._admin_token = secrets.token_hex(16)
        self._gateway_server: ThreadingHTTPServer | None = None
        self._gateway_thread: threading.Thread | None = None
        self._ui_server: ThreadingHTTPServer | None = None
        self._ui_thread: threading.Thread | None = None
        self._backend: RunnerBackend | None = None
        # run_id -> run access token (held server-side, never sent to the browser).
        self._run_tokens: dict[str, str] = {}
        self._lock = threading.RLock()
        self._base_url = ""

    @property
    def offline(self) -> bool:
        return self.config.provider == "offline"

    @property
    def base_url(self) -> str:
        return self._base_url

    # --- lifecycle ----------------------------------------------------------------------

    def start(self) -> str:
        """Boot gateway + backend + UI. Returns the UI base URL."""
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.config.run_root.mkdir(parents=True, exist_ok=True)

        provider_factory = self._provider_factory_override or (
            offline_provider_factory if self.offline else None
        )
        gateway = LlmGatewayBackend(
            token_manager=self._token_manager,
            provider_adapter_factory=provider_factory,
        )
        # Bind the gateway on an ephemeral loopback port; only the backend (this process) calls it.
        self._gateway_server = create_llm_gateway_server(
            gateway, host="127.0.0.1", port=0, admin_token=self._admin_token
        )
        gateway_port = self._gateway_server.server_address[1]
        self._gateway_thread = threading.Thread(
            target=self._gateway_server.serve_forever, name="studio-llm-gateway", daemon=True
        )
        self._gateway_thread.start()

        self._backend = RunnerBackend(
            run_root=self.config.run_root,
            token_manager=self._token_manager,
            allowed_workspace_roots=(self.workspace,),
            llm_gateway_url=f"http://127.0.0.1:{gateway_port}/internal/llm/turns",
        )

        self._ui_server = ThreadingHTTPServer(
            (self.config.host, self.config.port), _make_handler(self)
        )
        ui_port = self._ui_server.server_address[1]
        self._ui_thread = threading.Thread(
            target=self._ui_server.serve_forever, name="studio-ui", daemon=True
        )
        self._ui_thread.start()

        self._base_url = f"http://{self.config.host}:{ui_port}"
        _LOGGER.info("Studio listening on %s (provider=%s)", self._base_url, self.config.provider)
        return self._base_url

    def shutdown(self) -> None:
        # Cooperatively end this backend's runs in one call, so the process-shared run loop has
        # no parked session coroutines at exit (DX-2: drain instead of cancel-each + sleep).
        if self._backend is not None:
            self._backend.shutdown(drain=True)
        for server in (self._ui_server, self._gateway_server):
            if server is not None:
                try:
                    server.shutdown()
                    server.server_close()
                except Exception:  # pragma: no cover - best-effort teardown
                    _LOGGER.debug("error during server shutdown", exc_info=True)

    # --- chat operations (called by the handler) ----------------------------------------

    def start_chat(self, message: str) -> dict[str, Any]:
        """Open a new multi-turn session in the workspace and deliver the first message."""
        assert self._backend is not None
        runtime_config = _read_runtime_config()  # R1: chat + read-only filesystem
        request = BackendRunRequest(
            tenant_id=_TENANT,
            user_id=_USER,
            workspace_root=self.workspace,
            instruction=message,
            mode="propose",
            multi_turn=True,
            runtime_config=runtime_config,
        )
        submission = self._backend.submit_run(request)
        with self._lock:
            self._run_tokens[submission.run_id] = submission.run_token
        return {"run_id": submission.run_id, "status": submission.status}

    def continue_chat(self, run_id: str, message: str) -> dict[str, Any]:
        assert self._backend is not None
        token = self._token_for(run_id)
        return self._backend.send_message(run_id, token, message)

    def cancel_chat(self, run_id: str) -> dict[str, Any]:
        assert self._backend is not None
        token = self._token_for(run_id)
        return self._backend.cancel_run(run_id, token)

    def poll_events(self, run_id: str, from_seq: int) -> dict[str, Any]:
        assert self._backend is not None
        token = self._token_for(run_id)
        return self._backend.events(run_id, token, from_seq=from_seq)

    def run_status(self, run_id: str) -> dict[str, Any]:
        assert self._backend is not None
        token = self._token_for(run_id)
        return self._backend.status(run_id, token)

    def list_files(self) -> list[dict[str, Any]]:
        """A flat, sorted listing of the workspace for the file-tree panel (read-only;
        skips VCS/cache dirs and is bounded so a huge tree can't stall the UI)."""
        root = self.workspace
        entries: list[dict[str, Any]] = []
        if not root.exists():
            return entries
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if any(part in _TREE_SKIP for part in rel.parts):
                continue
            entries.append({"path": rel.as_posix(), "is_dir": path.is_dir()})
            if len(entries) >= _TREE_MAX_ENTRIES:
                break
        return entries

    def _token_for(self, run_id: str) -> str:
        with self._lock:
            token = self._run_tokens.get(run_id)
        if token is None:
            raise NativeAgentError(f"unknown run_id: {run_id}")
        return token


_TERMINAL = {"completed", "failed", "limited"}


def _make_handler(studio: StudioServer) -> type[BaseHTTPRequestHandler]:
    class StudioHandler(BaseHTTPRequestHandler):
        server_version = "NativeAgentRunnerStudio/0.1"
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # quiet by default
            _LOGGER.debug("studio http: " + args[0], *args[1:])

        # --- GET ---------------------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._serve_file(_WEB_DIR / "index.html", "text/html; charset=utf-8")
                return
            if parsed.path == "/healthz":
                self._write_json({"ok": True})
                return
            if parsed.path == "/api/config":
                self._write_json(
                    {
                        "workspace": str(studio.workspace),
                        "provider": studio.config.provider,
                        "offline": studio.offline,
                    }
                )
                return
            if parsed.path == "/api/files":
                self._write_json({"workspace": str(studio.workspace), "files": studio.list_files()})
                return
            if parsed.path == "/api/events":
                self._stream_events(parse_qs(parsed.query))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "not found")

        # --- POST --------------------------------------------------------------------
        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/chat":
                    body = self._read_json()
                    message = str(body.get("message") or "").strip()
                    if not message:
                        self._write_json({"error": "message is required"}, HTTPStatus.BAD_REQUEST)
                        return
                    run_id = body.get("run_id")
                    if run_id:
                        result = studio.continue_chat(str(run_id), message)
                    else:
                        result = studio.start_chat(message)
                    self._write_json(result)
                    return
                if parsed.path == "/api/cancel":
                    body = self._read_json()
                    run_id = str(body.get("run_id") or "")
                    self._write_json(studio.cancel_chat(run_id))
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
            except NativeAgentError as exc:
                self._write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except ValueError as exc:
                self._write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception:  # pragma: no cover - defensive
                _LOGGER.exception("studio POST failed")
                self._write_json({"error": "internal error"}, HTTPStatus.INTERNAL_SERVER_ERROR)

        # --- SSE ---------------------------------------------------------------------
        def _stream_events(self, query: dict[str, list[str]]) -> None:
            run_id = (query.get("run_id") or [""])[0]
            cursor = int((query.get("from") or ["0"])[0])
            if not run_id:
                self.send_error(HTTPStatus.BAD_REQUEST, "run_id required")
                return
            # SSE has no Content-Length; closing the connection at stream end is the
            # unambiguous framing that plain HTTP clients (and EventSource) both handle.
            self.close_connection = True
            try:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
            except OSError:
                return
            idle = 0.0
            try:
                while True:
                    payload = studio.poll_events(run_id, cursor)
                    events = payload.get("events", [])
                    if events:
                        idle = 0.0
                        for event in events:
                            seq = int(event.get("seq") or 0)
                            cursor = max(cursor, seq + 1)
                            self._sse_send(event)
                    else:
                        idle += 0.25
                    # Stop once the run is terminal and we've drained its events.
                    status = studio.run_status(run_id).get("status")
                    if status in _TERMINAL and not events:
                        self._sse_send({"type": "studio.stream.end", "data": {"status": status}})
                        return
                    if idle >= 15.0:  # heartbeat so proxies/clients keep the stream open
                        self._sse_comment("keep-alive")
                        idle = 0.0
                    time.sleep(0.25)
            except (BrokenPipeError, ConnectionError, OSError):
                return  # client disconnected
            except NativeAgentError:
                return

        def _sse_send(self, event: dict[str, Any]) -> None:
            # Annotate tool activity with a human-readable line for the UI feed (DX-3).
            summary = describe_event(event)
            if summary:
                event = {**event, "studio_activity": summary}
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
            self.wfile.flush()

        def _sse_comment(self, text: str) -> None:
            self.wfile.write(f": {text}\n\n".encode("utf-8"))
            self.wfile.flush()

        # --- helpers -----------------------------------------------------------------
        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length > _MAX_BODY_BYTES:
                raise ValueError("request body too large")
            raw = self.rfile.read(length) if length else b""
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        def _write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_file(self, path: Path, content_type: str) -> None:
            try:
                body = path.read_bytes()
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return StudioHandler
