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
from native_agent_runner.core.spec import ModelConfig, ReasoningConfig
from native_agent_runner.core.tool_surface import ToolScope
from native_agent_runner.errors import NativeAgentError
from native_agent_runner.reference._shared.tokens import TokenManager
from native_agent_runner.reference.backend.service import BackendRunRequest, RunnerBackend
from native_agent_runner.reference.llm_gateway.http import create_llm_gateway_server
from native_agent_runner.reference.llm_gateway.providers import offline_provider_factory
from native_agent_runner.reference.llm_gateway.service import LlmGatewayBackend
from native_agent_runner.reference.web_gateway.http import create_web_gateway_server
from native_agent_runner.reference.web_gateway.service import FakeWebProvider, WebGatewayBackend
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


# Obvious destructive command prefixes the shell binding refuses outright (a binding-level
# safety gate, enforced regardless of approval mode). Matched as command.strip().startswith.
_SHELL_DENY_PREFIXES = (
    "rm ", "rmdir", "del ", "rd ", "format", "mkfs", "dd ", "sudo ", "shutdown", "reboot",
)


# The Agent's capabilities, each mapping to the tool bindings it enables. Editable live from the
# Settings window (R6) — toggling one hot-swaps the running session's runtime config.
_ALL_CAPABILITIES = ("read", "write", "hitl", "shell", "web")
_CAPABILITY_LABELS = {
    "read": "Read files",
    "write": "Write files (staged as a proposal)",
    "hitl": "Ask the human for approval",
    "shell": "Run shell commands + background jobs",
    "web": "Search & fetch the web",
}


def _capability_bindings(capability: str) -> tuple[ToolBinding, ...]:
    if capability == "read":
        return (ToolBinding(binding_id="fs.read", model_name="fs_read", ref=RegistryToolRef("fs.read")),)
    if capability == "write":
        return (ToolBinding(binding_id="fs.write", model_name="fs_write", ref=RegistryToolRef("fs.write")),)
    if capability == "hitl":
        return (
            ToolBinding(binding_id="hitl.request", model_name="hitl_request", ref=RegistryToolRef("hitl.request")),
        )
    if capability == "shell":
        return (
            ToolBinding(
                binding_id="shell.exec",
                model_name="shell_exec",
                ref=RegistryToolRef("shell.exec"),
                scope=ToolScope(command_deny_prefixes=_SHELL_DENY_PREFIXES),
                runtime={"shell": {"approval_mode": "auto-approve"}},
            ),
        )
    if capability == "web":
        return (
            ToolBinding(binding_id="web.search", model_name="web_search", ref=RegistryToolRef("web.search")),
            ToolBinding(binding_id="web.fetch", model_name="web_fetch", ref=RegistryToolRef("web.fetch")),
            ToolBinding(binding_id="web.context", model_name="web_context", ref=RegistryToolRef("web.context")),
        )
    return ()


# Model + reasoning effort are quick-editable from the chat composer's setup bar.
_DEFAULT_MODEL = "gpt-5.5"
_DEFAULT_EFFORT = "medium"
# Offered in the UI — the reasoning efforts the default model (gpt-5.5) accepts:
# none/low/medium/high/xhigh (NOT "minimal", which is gpt-5-only and was replaced by "none"
# in the 5.1+ series). Effort support is model-dependent; a non-default model may differ.
_EFFORT_CHOICES = ("none", "low", "medium", "high", "xhigh")
# Validation superset = the engine's full ReasoningEffort literal (some values suit other models).
_ALL_EFFORTS = ("default", "none", "minimal", "low", "medium", "high", "xhigh")


def _runtime_config_for(
    capabilities: list[str], model: str = _DEFAULT_MODEL, effort: str = _DEFAULT_EFFORT
) -> AgentRuntimeConfig:
    """Build the runtime config for an enabled-capability set (order-stable, deduped) plus the
    chosen model + reasoning effort. The model flows to the gateway as the effective model name
    (ignored by the offline echo provider)."""
    enabled = set(capabilities)
    tools: list[ToolBinding] = []
    for capability in _ALL_CAPABILITIES:
        if capability in enabled:
            tools.extend(_capability_bindings(capability))
    return AgentRuntimeConfig(
        definition_id="studio-agent",
        model=ModelConfig(model=model, reasoning=ReasoningConfig(effort=effort)),  # provider="gateway"
        tools=tuple(tools),
    )


def _agent_runtime_config() -> AgentRuntimeConfig:
    """The full capability set (chat + read + write + HITL + shell + web)."""
    return _runtime_config_for(list(_ALL_CAPABILITIES))


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
        self._web_gateway_server: ThreadingHTTPServer | None = None
        self._web_gateway_thread: threading.Thread | None = None
        self._ui_server: ThreadingHTTPServer | None = None
        self._ui_thread: threading.Thread | None = None
        self._backend: RunnerBackend | None = None
        # The live-editable Agent capability set (Settings window, R6). Defaults to everything.
        self._capabilities: list[str] = list(_ALL_CAPABILITIES)
        # Live-editable model + reasoning effort (chat composer setup bar).
        self._model: str = _DEFAULT_MODEL
        self._effort: str = _DEFAULT_EFFORT
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

        # Web gateway (R5): the fake corpus provider, so web tools work with no egress/keys.
        web_gateway = WebGatewayBackend(token_manager=self._token_manager, provider=FakeWebProvider())
        self._web_gateway_server = create_web_gateway_server(
            web_gateway, host="127.0.0.1", port=0, admin_token=self._admin_token
        )
        web_port = self._web_gateway_server.server_address[1]
        self._web_gateway_thread = threading.Thread(
            target=self._web_gateway_server.serve_forever, name="studio-web-gateway", daemon=True
        )
        self._web_gateway_thread.start()

        self._backend = RunnerBackend(
            run_root=self.config.run_root,
            token_manager=self._token_manager,
            allowed_workspace_roots=(self.workspace,),
            # Allow applying an approved proposal back into the workspace (R2).
            allowed_apply_roots=(self.workspace,),
            llm_gateway_url=f"http://127.0.0.1:{gateway_port}/internal/llm/turns",
            web_gateway_url=f"http://127.0.0.1:{web_port}",
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
        for server in (self._ui_server, self._gateway_server, self._web_gateway_server):
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
        runtime_config = _runtime_config_for(self._capabilities, self._model, self._effort)
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

    def interrupt_chat(self, run_id: str) -> dict[str, Any]:
        """Turn-level stop: halt the current turn but keep the session alive (the next
        message continues the chat). Contrast cancel_chat, which ends the run."""
        assert self._backend is not None
        token = self._token_for(run_id)
        return self._backend.interrupt_turn(run_id, token)

    def poll_events(self, run_id: str, from_seq: int) -> dict[str, Any]:
        assert self._backend is not None
        token = self._token_for(run_id)
        return self._backend.events(run_id, token, from_seq=from_seq)

    def run_status(self, run_id: str) -> dict[str, Any]:
        assert self._backend is not None
        token = self._token_for(run_id)
        return self._backend.status(run_id, token)

    def proposal(self, run_id: str) -> dict[str, Any]:
        """The current proposed changes for the run: changed files + the unified diff, both via
        token-scoped backend APIs (no reading run artifacts off disk)."""
        assert self._backend is not None
        token = self._token_for(run_id)
        payload = self._backend.proposal(run_id, token)
        payload["diff"] = self._backend.proposal_diff(run_id, token).get("diff", "")
        return payload

    def apply(self, run_id: str) -> dict[str, Any]:
        """Approve and apply the current proposal into the workspace (the propose→apply step)."""
        assert self._backend is not None
        token = self._token_for(run_id)
        self._backend.approve_proposal(run_id, token, approver_id=_USER)
        result = self._backend.apply_proposal(run_id, token, target=self.workspace)
        return result

    def jobs(self, run_id: str) -> dict[str, Any]:
        """Background shell jobs for the run (running + finished)."""
        assert self._backend is not None
        return self._backend.jobs(run_id, self._token_for(run_id))

    def job_logs(self, run_id: str, job_id: str, *, stream: str = "stdout") -> dict[str, Any]:
        """Tail of a background job's stdout/stderr log."""
        assert self._backend is not None
        return self._backend.job_logs(run_id, self._token_for(run_id), job_id, stream=stream, tail_bytes=20_000)

    def answer_hitl(self, run_id: str, task_id: str, answer: str) -> dict[str, Any]:
        """Deliver the human's decision for a parked ``hitl.request`` (the approval gate). The
        answer is handed back to the agent as the tool's result and the run resumes."""
        assert self._backend is not None
        token = self._token_for(run_id)
        return self._backend.report_task_result(
            run_id, token, task_id=task_id, result={"answer": answer}, status="answered"
        )

    def settings(self) -> dict[str, Any]:
        """Current Studio settings: provider, capability set, and the model + reasoning effort."""
        return {
            "provider": self.config.provider,
            "offline": self.offline,
            "capabilities": list(self._capabilities),
            "available": [{"key": cap, "label": _CAPABILITY_LABELS[cap]} for cap in _ALL_CAPABILITIES],
            "model": self._model,
            "effort": self._effort,
            "efforts": list(_EFFORT_CHOICES),
        }

    def update_settings(
        self,
        *,
        capabilities: list[str] | None = None,
        model: str | None = None,
        effort: str | None = None,
    ) -> dict[str, Any]:
        """Change the Agent's capabilities / model / reasoning effort. Only provided fields
        change. New chats use the result; active sessions are hot-swapped in place via
        runtime-config replacement (applied at their next turn)."""
        assert self._backend is not None
        if capabilities is not None:
            self._capabilities = [cap for cap in _ALL_CAPABILITIES if cap in set(capabilities)]
        if model is not None and model.strip():
            self._model = model.strip()
        if effort is not None and effort in _ALL_EFFORTS:
            self._effort = effort
        new_config = _runtime_config_for(self._capabilities, self._model, self._effort)
        with self._lock:
            active = list(self._run_tokens.items())
        applied = 0
        for run_id, token in active:
            current = self._backend.current_runtime_config(run_id)
            if current is None:
                continue
            try:
                self._backend.replace_runtime_config(
                    run_id,
                    token,
                    expected_version=current.config_version,
                    issuer="studio-settings",
                    reason="settings change",
                    config=new_config,
                )
                applied += 1
            except (NativeAgentError, ValueError):
                pass  # terminal or stale run — skip
        return {
            "capabilities": list(self._capabilities),
            "model": self._model,
            "effort": self._effort,
            "applied_runs": applied,
        }

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
            if parsed.path == "/settings":
                self._serve_file(_WEB_DIR / "settings.html", "text/html; charset=utf-8")
                return
            if parsed.path == "/api/settings":
                self._write_json(studio.settings())
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
            if parsed.path == "/api/proposal":
                run_id = (parse_qs(parsed.query).get("run_id") or [""])[0]
                try:
                    self._write_json(studio.proposal(run_id))
                except NativeAgentError as exc:
                    self._write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if parsed.path == "/api/jobs":
                run_id = (parse_qs(parsed.query).get("run_id") or [""])[0]
                try:
                    self._write_json(studio.jobs(run_id))
                except NativeAgentError as exc:
                    self._write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if parsed.path == "/api/job-logs":
                query = parse_qs(parsed.query)
                run_id = (query.get("run_id") or [""])[0]
                job_id = (query.get("job_id") or [""])[0]
                stream = (query.get("stream") or ["stdout"])[0]
                try:
                    self._write_json(studio.job_logs(run_id, job_id, stream=stream))
                except NativeAgentError as exc:
                    self._write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
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
                if parsed.path == "/api/interrupt":
                    body = self._read_json()
                    run_id = str(body.get("run_id") or "")
                    self._write_json(studio.interrupt_chat(run_id))
                    return
                if parsed.path == "/api/apply":
                    body = self._read_json()
                    run_id = str(body.get("run_id") or "")
                    self._write_json(studio.apply(run_id))
                    return
                if parsed.path == "/api/hitl":
                    body = self._read_json()
                    run_id = str(body.get("run_id") or "")
                    task_id = str(body.get("task_id") or "")
                    answer = str(body.get("answer") or "")
                    self._write_json(studio.answer_hitl(run_id, task_id, answer))
                    return
                if parsed.path == "/api/settings":
                    body = self._read_json()
                    kwargs: dict[str, Any] = {}
                    if "capabilities" in body:
                        caps = body.get("capabilities")
                        if not isinstance(caps, list):
                            self._write_json({"error": "capabilities must be a list"}, HTTPStatus.BAD_REQUEST)
                            return
                        kwargs["capabilities"] = [str(c) for c in caps]
                    if "model" in body:
                        kwargs["model"] = str(body.get("model") or "")
                    if "effort" in body:
                        kwargs["effort"] = str(body.get("effort") or "")
                    self._write_json(studio.update_settings(**kwargs))
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
