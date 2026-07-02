"""Studio all-in-one server: LLM gateway + Monoid backend + UI/BFF in one process.

This is the reference "installable agent app" — it boots the reference ``LlmGatewayBackend`` and
``RunnerBackend`` behind a shared signing secret, then serves a single-page UI plus a thin
backend-for-frontend (BFF). The browser talks only to the BFF; run tokens and the admin token
stay server-side. The browser never sees a provider key.

Topology inside one process:

    browser ── HTTP ──> Studio BFF ──(Python calls)──> RunnerBackend ──(loopback HTTP)──> LLM gateway

The kernel is driven via its Python API (not its own HTTP surface), which is the most
representative path for an embedder bundling the engine into an app.

Not part of the supported surface — a reference example you copy and own.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from monoid_agent_kernel.core.agents import (
    AgentRuntimeConfig,
    PromptSpec,
    RegistryToolRef,
    SubagentDefinition,
    ToolBinding,
)
from monoid_agent_kernel.env import getenv
from monoid_agent_kernel.core.capability import AutoGrantBroker
from monoid_agent_kernel.core.content import ContentPart, DocumentPart, ImagePart, TextPart
from monoid_agent_kernel.core.spec import ModelConfig, ReasoningConfig
from monoid_agent_kernel.core.tool_surface import ToolScope
from monoid_agent_kernel.errors import NativeAgentError
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.backend.service import BackendRunRequest, RunnerBackend
from monoid_agent_kernel.reference.outbox import InboxRoutingOutboxSender, OutboxToolProvider
from monoid_agent_kernel.reference.llm_gateway.http import create_llm_gateway_server
from monoid_agent_kernel.reference.llm_gateway.providers import offline_provider_factory
from monoid_agent_kernel.reference.llm_gateway.service import LlmGatewayBackend
from monoid_agent_kernel.reference.web_gateway.http import create_web_gateway_server
from monoid_agent_kernel.reference.web_gateway.service import FakeWebProvider, WebGatewayBackend
from monoid_agent_kernel.reference.studio.activity import describe_event
from monoid_agent_kernel.reference._shared.http_util import wait_http_ready
from monoid_agent_kernel.reference.mcp_gateway import FakeMcpServer, create_mcp_server
from monoid_agent_kernel.mcp import McpError, McpToolProvider
from monoid_agent_kernel.skills import SkillProvider, load_skill_definitions

_LOGGER = logging.getLogger("monoid_agent_kernel.studio")

_WEB_DIR = Path(__file__).parent / "web"
# Per-attachment cap. An attachment rides a base64 ``data:`` URI inside the JSON body (handed to
# the kernel by value; the core normalizes it to a content-addressed blob), so the HTTP body limit
# is sized to clear one inflated image plus the JSON envelope.
_MAX_ATTACH_BYTES = 8 * 1024 * 1024
_MAX_BODY_BYTES = _MAX_ATTACH_BYTES + 2 * 1024 * 1024  # room for base64 inflation + JSON envelope
# Studio is a single-user local app; the tenant/user are fixed placeholders.
_TENANT = "studio"
_USER = "local"

# Directories never shown in the file tree (and not worth walking).
_TREE_SKIP = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
_TREE_MAX_ENTRIES = 2000
_VIEW_MAX_BYTES = 256 * 1024  # file-viewer read cap — keep a huge file from stalling the UI
# File-viewer image preview: these extensions are served as raw bytes via /api/file-raw and
# rendered with an <img>, instead of being refused as binary by the text viewer.
_IMAGE_EXTS = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
    ".webp": "image/webp", ".bmp": "image/bmp", ".ico": "image/x-icon", ".svg": "image/svg+xml",
}
_IMAGE_VIEW_MAX_BYTES = 8 * 1024 * 1024  # raw-image preview cap

# Vendored static assets (e.g. the locally-bundled KaTeX) served under /vendor/<path>. Content
# types for the extensions we actually ship; anything else falls back to octet-stream.
_VENDOR_DIR = _WEB_DIR / "vendor"
_VENDOR_CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".ttf": "font/ttf",
    ".map": "application/json",
}


# Obvious destructive command prefixes the shell binding refuses outright (a binding-level
# safety gate, enforced regardless of approval mode). Matched as command.strip().startswith.
_SHELL_DENY_PREFIXES = (
    "rm ", "rmdir", "del ", "rd ", "format", "mkfs", "dd ", "sudo ", "shutdown", "reboot",
)


# The Agent's capabilities, each mapping to the tool bindings it enables. Editable live from the
# Settings window (R6) — toggling one hot-swaps the running session's runtime config.
_ALL_CAPABILITIES = ("read", "write", "hitl", "shell", "web", "delegate")
_CAPABILITY_LABELS = {
    "read": "Read files",
    "write": "Write files (staged as a proposal)",
    "hitl": "Ask the human for approval",
    "shell": "Run shell commands + background jobs",
    "web": "Search & fetch the web",
    "delegate": "Delegate subtasks to a subagent",
}
# Provider-backed capabilities (Skills, MCP) — present only when their provider is attached at
# boot. Unlike the static capabilities above, their bindings come from a provider instance
# (skill_provider.tool_bindings() / mcp_provider.tool_bindings()), so they are threaded into the
# runtime config as provider_bindings rather than resolved by _capability_bindings.
_PROVIDER_CAPABILITY_LABELS = {
    "skills": "Use Agent Skills (progressive-disclosure playbooks)",
    "mcp": "Use tools from the connected MCP server",
}
# Bundled sample skill, so `studio serve` demonstrates Skills with zero configuration.
_SAMPLE_SKILLS_DIR = Path(__file__).parent / "sample-skills"

# Agent-as-tool: a read-only "researcher" subagent. Its tool allowlist is intersected with the
# parent's bindings (a subagent can never exceed the parent), so it gets read + web only — no
# write/shell. The child runs in isolation and returns just its final message; its live work is
# observable via run_root/<child_run_id>/events.jsonl (see StudioServer.subagent_events).
_SUBAGENT_DEFINITIONS = {
    "researcher": SubagentDefinition(
        description=(
            "Investigate a focused question read-only — reads files and searches the web, then "
            "reports findings. Cannot edit files or run shell commands."
        ),
        prompt=PromptSpec(
            system_prompt_base=(
                "You are a research subagent working in isolation. Investigate the task using the "
                "read and web tools available to you, then return a concise findings summary as "
                "your final message. You cannot edit files or run commands."
            )
        ),
        tools=("fs.read", "text.search", "web.search", "web.fetch", "web.context", "run.update_plan"),
        context="fresh",
    ),
}


def load_env_file(path: Path | None) -> dict[str, str]:
    """Load simple KEY=VALUE pairs into os.environ without overriding existing values."""
    if path is None or not path.exists():
        return {}
    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def _capability_bindings(capability: str) -> tuple[ToolBinding, ...]:
    # ToolBinding.for_tool derives binding_id + model_name from the tool id; only the shell
    # binding needs extra scope/runtime, which pass through as keyword arguments.
    if capability == "read":
        return (ToolBinding.for_tool("fs.read"),)
    if capability == "write":
        return (ToolBinding.for_tool("fs.write"),)
    if capability == "hitl":
        return (ToolBinding.for_tool("hitl.request"),)
    if capability == "shell":
        return (
            ToolBinding.for_tool(
                "shell.exec",
                scope=ToolScope(command_deny_prefixes=_SHELL_DENY_PREFIXES),
                runtime={"shell": {"approval_mode": "auto-approve"}},
            ),
        )
    if capability == "web":
        return (
            ToolBinding.for_tool("web.search"),
            ToolBinding.for_tool("web.fetch"),
            ToolBinding.for_tool("web.context"),
        )
    if capability == "delegate":
        # Only effective when the backend carries subagent_definitions (the loop bootstrap
        # registers agent.spawn then); studio always does.
        return (ToolBinding.for_tool("agent.spawn"),)
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
# Reasoning *summary* visibility (DX-13b): "auto" surfaces a model-written summary of its thinking
# in the chat's collapsible "Thinking" panel; "off" hides it. Display-only — independent of the
# reasoning round-trip, which always travels by-value.
_DEFAULT_SUMMARY = "auto"
_SUMMARY_CHOICES = ("off", "auto", "detailed")

# OTel tracing (Tier-3): when toggled on, runs emit GenAI spans via OtelEventSink to an OTLP
# collector (default = a local Jaeger's OTLP/HTTP port). The exporter + global provider are set
# up once, lazily; the sink is a no-op until then.
_OTEL_ENDPOINT = getenv("MONOID_OTEL_ENDPOINT") or "http://localhost:4318/v1/traces"
_otel_provider_ready = False


def _ensure_otel_provider(endpoint: str) -> None:
    """Install a global OTel TracerProvider with an OTLP/HTTP span exporter (idempotent). Raises
    NativeAgentError with an actionable hint if the OTel SDK/exporter extras aren't installed."""
    global _otel_provider_ready
    if _otel_provider_ready:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise NativeAgentError(
            "OTel tracing needs the opentelemetry SDK + OTLP/HTTP exporter "
            "(pip install 'monoid-agent-kernel[otel-export]')"
        ) from exc
    provider = TracerProvider(resource=Resource.create({"service.name": "agent-studio"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _otel_provider_ready = True


# System prompt: introduce the agent + nudge it to keep a live plan (Plan panel). The plan is
# pure observability — the model self-reports via run.update_plan; the engine never enforces it.
_SYSTEM_PROMPT = (
    "You are an agent working inside a user's project workspace. For any task that takes more "
    "than one step, keep a short running plan with the run_update_plan tool: pass items as "
    "{step, status} where status is one of pending, in_progress, or completed, and update it as "
    "you make progress (mark a step in_progress when you start it and completed when done). "
    "Keep the plan concise — a handful of steps, not a transcript. "
    "For a focused, self-contained subtask (especially read-only research), you may delegate to "
    "a subagent with the agent_spawn tool; it works in isolation and returns only its final "
    "message."
)
# Always available (observability, not a gated capability): the plan tool the Plan panel renders.
_PLAN_BINDING = ToolBinding(
    binding_id="run.update_plan", model_name="run_update_plan", ref=RegistryToolRef("run.update_plan")
)


def _runtime_config_for(
    capabilities: list[str],
    model: str = _DEFAULT_MODEL,
    effort: str = _DEFAULT_EFFORT,
    summary: str = _DEFAULT_SUMMARY,
    *,
    provider_bindings: dict[str, tuple[ToolBinding, ...]] | None = None,
) -> AgentRuntimeConfig:
    """Build the runtime config for an enabled-capability set (order-stable, deduped) plus the
    chosen model + reasoning effort + summary visibility. The model flows to the gateway as the
    effective model name (ignored by the offline echo provider).

    ``provider_bindings`` maps a provider-backed capability ("skills", "mcp") to the bindings its
    attached provider exposes; they are merged when the capability is enabled, so the same
    config-build path feeds both new chats and the settings hot-swap."""
    enabled = set(capabilities)
    tools: list[ToolBinding] = [_PLAN_BINDING]  # plan tool is always bound (observability)
    for capability in _ALL_CAPABILITIES:
        if capability in enabled:
            tools.extend(_capability_bindings(capability))
    for capability, bindings in (provider_bindings or {}).items():
        if capability in enabled:
            tools.extend(bindings)
    return AgentRuntimeConfig(
        definition_id="studio-agent",
        model=ModelConfig(model=model, reasoning=ReasoningConfig(effort=effort, summary=summary)),
        prompt=PromptSpec(system_prompt_base=_SYSTEM_PROMPT),
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
    # Agent Skills directory (progressive disclosure). Defaults to the bundled sample skill so
    # `studio serve` demonstrates Skills with zero config; None disables Skills entirely.
    skills_directory: Path | None = field(default_factory=lambda: _SAMPLE_SKILLS_DIR)
    # Attach the bundled offline reference MCP server (fake, loopback) and expose its tools.
    mcp: bool = False
    # Optional env file loaded at server start without overriding process env.
    env_file: Path | None = None


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
        # Tool/context providers attached at boot (Skills, MCP). The MCP server is a bundled fake
        # gateway on a loopback port (see start()); its provider holds a live connection closed on
        # shutdown. Both are instances shared across runs (the backend's provider seam).
        self._skill_provider: SkillProvider | None = None
        self._mcp_provider: Any = None
        self._mcp_server: ThreadingHTTPServer | None = None
        self._mcp_thread: threading.Thread | None = None
        # The live-editable Agent capability set (Settings window, R6). Defaults to every available
        # capability; the provider-backed ones ("skills", "mcp") are appended in start() once their
        # provider is attached.
        self._capabilities: list[str] = list(_ALL_CAPABILITIES)
        # Live-editable model + reasoning effort + summary visibility (chat composer setup bar).
        self._model: str = _DEFAULT_MODEL
        self._effort: str = _DEFAULT_EFFORT
        self._summary: str = _DEFAULT_SUMMARY
        self._otel: bool = False  # OTel span export toggle (Tier-3)
        # run_id -> run access token (held server-side, never sent to the browser).
        self._run_tokens: dict[str, str] = {}
        # Chat sessions started this server run (newest first): the history list. In-memory only —
        # see DX note: a cross-restart history would need a backend "list runs" API.
        self._sessions: list[dict[str, Any]] = []
        # A2A demo (agent-to-agent messaging): a logical agent name -> run_id directory the routing
        # outbox sender resolves so one peer can address another by name. The peer's run token comes
        # from _run_tokens. One shared sender drains every run's outbox into the addressed peer's
        # idempotent inbox via the backend's send_message.
        self._agent_directory: dict[str, str] = {}
        self._a2a_sender = InboxRoutingOutboxSender(deliver=self._a2a_deliver)
        self._lock = threading.RLock()
        self._base_url = ""

    @property
    def offline(self) -> bool:
        return self.config.provider == "offline"

    @property
    def base_url(self) -> str:
        return self._base_url

    # --- provider-backed capabilities (Skills / MCP) ------------------------------------

    def _provider_bindings(self) -> dict[str, tuple[ToolBinding, ...]]:
        """Bindings for each attached provider-backed capability (cached — no I/O: the MCP
        provider's tool_bindings reuses its boot-time discovery)."""
        bindings: dict[str, tuple[ToolBinding, ...]] = {}
        if self._skill_provider is not None:
            bindings["skills"] = self._skill_provider.tool_bindings()
        if self._mcp_provider is not None:
            bindings["mcp"] = self._mcp_provider.tool_bindings()
        return bindings

    def _available_capabilities(self) -> list[str]:
        """All capabilities offered in settings: the static set plus provider-backed ones whose
        provider is attached this boot."""
        return list(_ALL_CAPABILITIES) + list(self._provider_bindings())

    def _capability_labels(self) -> dict[str, str]:
        labels = dict(_CAPABILITY_LABELS)
        for cap in self._provider_bindings():
            labels[cap] = _PROVIDER_CAPABILITY_LABELS.get(cap, cap)
        return labels

    def _build_config(self) -> AgentRuntimeConfig:
        """The runtime config for the current settings, including any enabled provider tools."""
        return _runtime_config_for(
            self._capabilities,
            self._model,
            self._effort,
            self._summary,
            provider_bindings=self._provider_bindings(),
        )

    def _load_skill_provider(self) -> SkillProvider | None:
        """Load Agent Skills from the configured directory into one provider (None if disabled or
        empty). Offline and synchronous — no boot-ordering concern (unlike MCP)."""
        directory = self.config.skills_directory
        if directory is None:
            return None
        try:
            definitions = load_skill_definitions(Path(directory))
        except ValueError as exc:  # missing/unreadable directory → run without Skills
            _LOGGER.warning("skills directory %s not loadable: %s", directory, exc)
            return None
        if not definitions:
            return None
        _LOGGER.info("loaded %d skill(s) from %s", len(definitions), directory)
        return SkillProvider(definitions)

    def _start_fake_mcp(self) -> McpToolProvider | None:
        """Boot the bundled offline MCP gateway on a loopback port and connect a provider to it.

        Strict ordering (the boot-ordering risk): serve -> wait until /healthz answers ->
        construct the provider -> force discovery eagerly so its tools are cached before any run
        validates against them. On any failure, tear down and return None — MCP degrades to
        off, never crashing studio boot."""
        server = create_mcp_server(
            FakeMcpServer(), host="127.0.0.1", port=0, admin_token=self._admin_token
        )
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, name="studio-mcp-gateway", daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{port}"
        provider: McpToolProvider | None = None
        try:
            wait_http_ready(base_url, timeout_s=10)
            provider = McpToolProvider(f"{base_url}/mcp", server="studio", token=self._admin_token)
            provider.tool_bindings()  # force discovery now so it's cached, not lazy at first run
        except (McpError, TimeoutError, OSError) as exc:
            _LOGGER.warning("fake MCP gateway unavailable; continuing without MCP: %s", exc)
            if provider is not None:
                provider.close()
            server.shutdown()
            server.server_close()
            return None
        self._mcp_server = server
        self._mcp_thread = thread
        _LOGGER.info("fake MCP gateway on %s (%d tools)", base_url, len(provider.tool_bindings()))
        return provider

    # --- lifecycle ----------------------------------------------------------------------

    def start(self) -> str:
        """Boot gateway + backend + UI. Returns the UI base URL."""
        load_env_file(self.config.env_file)
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

        # Agent Skills (offline, no I/O): load the bundled/configured directory into one
        # SkillProvider, attached to the backend as both a tool and a context provider.
        self._skill_provider = self._load_skill_provider()

        # MCP (opt-in): boot the bundled fake MCP gateway on loopback, then connect a provider to
        # it. Strictly ordered (serve -> wait ready -> discover) and degrades to no-MCP on failure.
        if self.config.mcp:
            self._mcp_provider = self._start_fake_mcp()

        # Enable the provider-backed capabilities by default when their provider is attached.
        self._capabilities = self._available_capabilities()

        subagent_definitions = dict(_SUBAGENT_DEFINITIONS)
        if self._skill_provider is not None:
            # Fork skills (context: fork) run as subagents; register their synthesized definitions
            # (namespaced ids, so no collision with the built-in researcher). No-op for the sample.
            subagent_definitions.update(self._skill_provider.subagent_definitions())

        provider_instances = tuple(p for p in (self._skill_provider, self._mcp_provider) if p is not None)
        # A2A demo: the generic outbox.send tool is always available (its binding is added only for
        # the demo peers, so a normal chat never sees it). Its tools are declared to config
        # validation through the same provider seam as Skills/MCP.
        provider_instances = provider_instances + (OutboxToolProvider(),)

        self._backend = RunnerBackend(
            run_root=self.config.run_root,
            token_manager=self._token_manager,
            allowed_workspace_roots=(self.workspace,),
            # Allow applying an approved proposal back into the workspace (R2).
            allowed_apply_roots=(self.workspace,),
            llm_gateway_url=f"http://127.0.0.1:{gateway_port}/internal/llm/turns",
            web_gateway_url=f"http://127.0.0.1:{web_port}",
            # Stream tokens live: emit model.output.delta events the UI renders incrementally
            # (effective for adapters that support astream_turn — the gateway/openai path).
            emit_output_deltas=True,
            # Follow-up attachments ride a base64 data: URI through send_message, so the message
            # size limit must clear an inline image (the core normalizes it to a blob downstream).
            max_message_bytes=_MAX_BODY_BYTES,
            # Agent-as-tool: makes agent.spawn available (bound via the "delegate" capability).
            # Plus any fork-skill subagents synthesized by the skill provider.
            subagent_definitions=subagent_definitions,
            # The provider seam: Skills (tool + context) and MCP (tool) attach here, shared across
            # runs and re-attached on resume. Their tools are declared to config validation too.
            tool_providers=provider_instances,
            context_providers=(self._skill_provider,) if self._skill_provider is not None else (),
            # A2A demo: drain each run's outbox into the addressed peer's inbox, and gate outbox.send
            # behind a capability lease (the binding declares requires_lease). AutoGrantBroker grants
            # every request — a dev/demo broker, never production — so the lease gate is *exercised*
            # (brokered handle on the request + capability.* events) while the actual cross-agent
            # transport uses Studio's server-side run token. Both are no-ops for a normal chat: a
            # plain chat binds neither outbox.send nor any requires_lease tool.
            outbox_sender_factory=lambda req: self._a2a_sender,
            capability_broker_factory=lambda req: AutoGrantBroker(),
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
        # Close the MCP provider's live connection before stopping its server (DX-2 ethos).
        if self._mcp_provider is not None:
            try:
                self._mcp_provider.close()
            except Exception:  # pragma: no cover - best-effort teardown
                _LOGGER.debug("error closing MCP provider", exc_info=True)
        for server in (self._ui_server, self._gateway_server, self._web_gateway_server, self._mcp_server):
            if server is not None:
                try:
                    server.shutdown()
                    server.server_close()
                except Exception:  # pragma: no cover - best-effort teardown
                    _LOGGER.debug("error during server shutdown", exc_info=True)

    # --- chat operations (called by the handler) ----------------------------------------

    def _parts_from_attachments(
        self, message: str, attachments: Sequence[dict[str, Any]]
    ) -> tuple[ContentPart, ...]:
        """Build the multimodal content parts for a user message. Each attachment is
        ``{name, mime, data_b64}``; ``image/*`` becomes an ``ImagePart``, anything else (e.g.
        ``application/pdf``) a ``DocumentPart``. The bytes are handed in **by value** as a ``data:``
        URI — the kernel normalizes that to a durable content-addressed blob at ingestion, so
        the studio manages no attachment files and the image survives restart/re-provisioning.
        Returns ``()`` when there are no attachments (the caller uses the plain-text path)."""
        if not attachments:
            return ()
        parts: list[ContentPart] = []
        if message.strip():
            parts.append(TextPart(message.strip()))
        for att in attachments:
            name = str(att.get("name") or "file")
            mime = str(att.get("mime") or "application/octet-stream")
            data_b64 = str(att.get("data_b64") or "")
            try:
                raw = base64.b64decode(data_b64, validate=True)
            except (ValueError, TypeError) as exc:
                raise NativeAgentError(f"attachment {name!r} is not valid base64") from exc
            if not raw:
                raise NativeAgentError(f"attachment {name!r} is empty")
            if len(raw) > _MAX_ATTACH_BYTES:
                raise NativeAgentError(f"attachment {name!r} exceeds the {_MAX_ATTACH_BYTES}-byte limit")
            source_ref = f"data:{mime};base64,{data_b64}"
            if mime.startswith("image/"):
                parts.append(ImagePart(source_ref=source_ref, mime_type=mime))
            else:
                parts.append(DocumentPart(source_ref=source_ref, mime_type=mime))
        return tuple(parts)

    def start_chat(self, message: str, attachments: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
        """Open a new multi-turn session in the workspace and deliver the first message (with any
        image/document attachments as multimodal content parts)."""
        assert self._backend is not None
        runtime_config = self._build_config()
        parts = self._parts_from_attachments(message, attachments)
        request = BackendRunRequest(
            tenant_id=_TENANT,
            user_id=_USER,
            workspace_root=self.workspace,
            instruction=message or "[attachment]",
            input_parts=parts,
            mode="propose",
            multi_turn=True,
            runtime_config=runtime_config,
        )
        submission = self._backend.submit_run(request)
        title = " ".join(message.split())[:60] or "(empty)"
        with self._lock:
            self._run_tokens[submission.run_id] = submission.run_token
            self._sessions.insert(0, {"run_id": submission.run_id, "title": title, "created_at": time.time()})
        return {"run_id": submission.run_id, "status": submission.status}

    # --- A2A demo (agent-to-agent durable messaging) ------------------------------------

    def _a2a_deliver(
        self,
        destination: str,
        payload: dict[str, Any],
        *,
        message_id: str,
        correlation_id: str,
        causation_id: str,
        traceparent: str,
    ) -> str:
        """Deliver one staged outbox send into the addressed peer's inbox (the routing sender's
        edge IO). Resolves the peer name through the agent directory, then hands the message to the
        backend's idempotent ingress. Raises (→ a retryable send) when the peer isn't registered yet
        or its run can't accept the message, so the backend redrives until it can. Runs on the shared
        backend loop inside _drain_outbox; send_message only schedules the enqueue, so no blocking."""
        assert self._backend is not None
        with self._lock:
            run_id = self._agent_directory.get(destination)
            token = self._run_tokens.get(run_id or "")
        if not run_id or not token:
            raise LookupError(f"no agent registered as {destination!r}")
        text = str(payload.get("text") or json.dumps(payload))
        result = self._backend.send_message(
            run_id,
            token,
            text,
            message_id=message_id,
            source="agent",
            correlation_id=correlation_id,
            causation_id=causation_id,
            traceparent=traceparent,
        )
        return f"a2a:{run_id}:{result.get('message_id', '')}"

    def _a2a_peer_config(self, name: str, peer: str) -> AgentRuntimeConfig:
        """A peer's runtime config: the current settings' tools plus a lease-gated outbox.send
        binding, with a persona segment naming the agent and how to message its peer."""
        base = self._build_config()
        outbox_binding = ToolBinding(
            binding_id="outbox.send",
            model_name="outbox_send",
            ref=RegistryToolRef("outbox.send"),
            # The capability gate brokers an outbox.send lease before each send is staged.
            runtime={"requires_lease": True},
        )
        persona = (
            f"You are the '{name}' agent, collaborating with your peer agent '{peer}'. "
            f"To send a message to your peer, call the outbox_send tool with "
            f"destination='{peer}' and payload={{\"text\": <your message>}}. A message from your "
            f"peer arrives as a new user turn. When the task is complete, reply to your peer and "
            f"then give a short final summary."
        )
        prompt = PromptSpec(system_prompt_base=f"{_SYSTEM_PROMPT}\n\n{persona}")
        return replace(base, prompt=prompt, tools=base.tools + (outbox_binding,))

    def _spawn_peer(self, name: str, peer: str, *, instruction: str) -> str:
        assert self._backend is not None
        request = BackendRunRequest(
            tenant_id=_TENANT,
            user_id=_USER,
            workspace_root=self.workspace,
            instruction=instruction,
            mode="propose",
            multi_turn=True,
            runtime_config=self._a2a_peer_config(name, peer),
        )
        submission = self._backend.submit_run(request)
        with self._lock:
            self._run_tokens[submission.run_id] = submission.run_token
            self._agent_directory[name] = submission.run_id
            self._sessions.insert(
                0,
                {"run_id": submission.run_id, "title": f"A2A · {name}", "created_at": time.time()},
            )
        return submission.run_id

    def start_a2a_demo(self, task: str) -> dict[str, Any]:
        """Spin up two peer agents (planner + worker) wired to message each other through the
        durable outbox→inbox fabric, and seed the planner with ``task``. The worker is started first
        so its inbox exists before the planner addresses it. Returns both run ids.

        Note: a real exchange needs a tool-calling model (the openai provider, or a scripted fake in
        tests) — the offline echo provider won't emit outbox_send calls on its own."""
        assert self._backend is not None
        task = task.strip() or "Break a small task into steps and complete it together."
        worker_id = self._spawn_peer(
            "worker", "planner", instruction="Stand by for a task from planner."
        )
        planner_id = self._spawn_peer("planner", "worker", instruction=task)
        return {"planner": planner_id, "worker": worker_id}

    def sessions(self) -> dict[str, Any]:
        """The chat history (newest first), restart-surviving via the backend (DX-12).

        Source of truth is ``backend.list_runs`` (scans run_root → titles/status + a read token
        per run), so the list — and loading a past chat — works after a restart. The per-run read
        tokens are stored server-side (never sent to the browser). Very-recent runs whose run.json
        isn't on disk yet are overlaid from the in-memory ``_sessions`` so a new chat appears
        immediately."""
        assert self._backend is not None
        listing = self._backend.list_runs(_TENANT, user_id=_USER).get("runs", [])
        with self._lock:
            for run in listing:
                self._run_tokens.setdefault(run["run_id"], run["read_token"])
            known = {run["run_id"] for run in listing}
            recents = [s for s in self._sessions if s["run_id"] not in known]
        out = [
            {
                "run_id": run["run_id"],
                "title": run["title"] or run["run_id"],
                "status": run["status"],
                "created_at": run["created_at"],
                "recoverable": run["recoverable"],
            }
            for run in listing
        ]
        out += [
            {"run_id": s["run_id"], "title": s["title"], "status": "running",
             "created_at": s["created_at"], "recoverable": False}
            for s in recents
        ]
        out.sort(key=lambda entry: entry["created_at"], reverse=True)
        return {"sessions": out}

    def continue_chat(
        self, run_id: str, message: str, attachments: Sequence[dict[str, Any]] = ()
    ) -> dict[str, Any]:
        """Deliver a follow-up message (optionally with image/document attachments). If the run is
        no longer live in memory (a parked session surviving a restart), transparently resume it
        from its checkpoint first, then send — so "continue an old chat" just works. ``send_message``
        raises KeyError for a non-in-memory run; we resume on that signal and retry once."""
        assert self._backend is not None
        token = self._token_for(run_id)
        parts = self._parts_from_attachments(message, attachments)
        payload: str | tuple[ContentPart, ...] = parts if parts else message
        try:
            return self._backend.send_message(run_id, token, payload)
        except KeyError:
            self._backend.resume_run(run_id, token)
            return self._backend.send_message(run_id, token, payload)

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

    def subagent_events(self, child_run_id: str, from_seq: int = 0) -> dict[str, Any]:
        """Stream a child subagent run's work for the parent UI.

        A spawned subagent is an isolated child run (id ``<parent>.sub.<task>``) the backend
        doesn't expose via a record. We derive the ancestor run studio submitted from the id and
        read the descendant's events through ``backend.descendant_events`` (authorized by the
        ancestor's run token, which the BFF holds server-side) — no filesystem access from the UI
        path. This is the DX-11 fix: the descendant-events API replaced studio's earlier direct
        events.jsonl read."""
        assert self._backend is not None
        if ".sub." not in child_run_id:
            return {"events": []}
        parent_run_id = child_run_id.split(".sub.", 1)[0]
        try:
            token = self._token_for(parent_run_id)
            return self._backend.descendant_events(parent_run_id, token, child_run_id, from_seq=from_seq)
        except NativeAgentError:
            return {"events": []}

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

    def apply(self, run_id: str, *, approved_paths: tuple[str, ...] = ()) -> dict[str, Any]:
        """Approve and apply the current proposal into the workspace (the propose→apply step).

        ``approved_paths`` empty = approve every changed path (the legacy all-or-nothing behavior).
        A non-empty subset records a partial approval, so apply_package writes only those files and
        reports the rest as skipped — the per-file approval gate the core has always supported but
        the studio used to bypass."""
        assert self._backend is not None
        token = self._token_for(run_id)
        self._backend.approve_proposal(
            run_id, token, approver_id=_USER, approved_paths=approved_paths
        )
        result = self._backend.apply_proposal(run_id, token, target=self.workspace)
        return result

    def export_package(self, run_id: str) -> dict[str, Any]:
        """Build the portable proposal package and return its RECEIPT (``digest`` + name + size).
        The bytes are fetched separately by digest via :meth:`read_artifact` — no run_dir path ever
        crosses the boundary, so this works identically whether the backend is co-located or remote
        (closes the R9 contract gap)."""
        assert self._backend is not None
        token = self._token_for(run_id)
        return self._backend.export_proposal_package(run_id, token)

    def read_artifact(self, run_id: str, digest: str) -> tuple[bytes, str]:
        """Fetch a run artifact's bytes by sha256 digest (the data-returning seam). Returns
        ``(bytes, download_name)``."""
        assert self._backend is not None
        token = self._token_for(run_id)
        data = self._backend.read_run_artifact(run_id, token, digest)
        return data, f"proposal-{digest[:12]}.tar"

    def proposal_image(self, run_id: str, path: str) -> tuple[bytes, str]:
        """Bytes + content-type for an image in the run's PROPOSAL snapshot — lets the proposal
        panel preview a generated image *before* it is applied. Mirrors :meth:`read_image` but
        sources the bytes from the token-scoped backend ``proposal_file`` API (base64 for binary,
        utf-8 for an SVG), never the live workspace (the image isn't there until apply)."""
        assert self._backend is not None
        mime = _IMAGE_EXTS.get(Path(path).suffix.lower())
        if mime is None:
            raise NativeAgentError("not an image file")
        token = self._token_for(run_id)
        payload = self._backend.proposal_file(run_id, token, path)
        content = str(payload.get("content") or "")
        data = base64.b64decode(content) if payload.get("encoding") == "base64" else content.encode("utf-8")
        if len(data) > _IMAGE_VIEW_MAX_BYTES:
            raise NativeAgentError(f"image exceeds the {_IMAGE_VIEW_MAX_BYTES}-byte preview limit")
        return data, mime

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
        labels = self._capability_labels()
        return {
            "provider": self.config.provider,
            "offline": self.offline,
            "capabilities": list(self._capabilities),
            "available": [{"key": cap, "label": labels[cap]} for cap in self._available_capabilities()],
            "model": self._model,
            "effort": self._effort,
            "efforts": list(_EFFORT_CHOICES),
            "summary": self._summary,
            "summaries": list(_SUMMARY_CHOICES),
            "otel": self._otel,
        }

    def capabilities_catalog(self) -> dict[str, Any]:
        """Read-only catalog of the attached providers' offerings, for a UI list: the available
        Agent Skills (name + description), the connected MCP server's tools (id + description), and
        the output validators registered on the backend (id). Each empty when none is attached."""
        skills = self._skill_provider.catalog() if self._skill_provider is not None else []
        mcp_tools: list[dict[str, str]] = []
        if self._mcp_provider is not None:
            mcp_tools = [
                {"id": spec.id, "description": spec.description}
                for spec in self._mcp_provider.get_tools()
            ]
        output_validators = [
            {"id": validator.id}
            for validator in (self._backend.output_validators if self._backend is not None else ())
        ]
        return {"skills": skills, "mcp_tools": mcp_tools, "output_validators": output_validators}

    def update_settings(
        self,
        *,
        capabilities: list[str] | None = None,
        model: str | None = None,
        effort: str | None = None,
        summary: str | None = None,
        otel: bool | None = None,
    ) -> dict[str, Any]:
        """Change the Agent's capabilities / model / reasoning effort / summary / OTel tracing.
        Only provided fields change. New chats use the result; active sessions are hot-swapped in
        place via runtime-config replacement (applied at their next turn)."""
        assert self._backend is not None
        if capabilities is not None:
            requested = set(capabilities)
            self._capabilities = [cap for cap in self._available_capabilities() if cap in requested]
        if model is not None and model.strip():
            self._model = model.strip()
        if effort is not None and effort in _ALL_EFFORTS:
            self._effort = effort
        if summary is not None and summary in _SUMMARY_CHOICES:
            self._summary = summary
        if otel is not None:
            self._set_otel(otel)
        new_config = self._build_config()
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
            "summary": self._summary,
            "otel": self._otel,
            "applied_runs": applied,
        }

    def _set_otel(self, enabled: bool) -> None:
        """Toggle OTel span export: install the global provider on enable and attach/detach the
        per-run OtelEventSink factory on the backend (new runs pick it up)."""
        assert self._backend is not None
        if enabled:
            _ensure_otel_provider(_OTEL_ENDPOINT)
            from monoid_agent_kernel.observability.otel import OtelEventSink

            self._backend.extra_event_sink_factories = (OtelEventSink,)
        else:
            self._backend.extra_event_sink_factories = ()
        self._otel = enabled

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

    def _resolve_workspace_file(self, rel_path: str) -> Path:
        """Resolve a viewer path to a real file inside the workspace, rejecting empty paths,
        traversal / absolute paths that escape the root, and non-files."""
        if not rel_path:
            raise NativeAgentError("path is required")
        root = self.workspace.resolve()
        candidate = (root / rel_path).resolve()
        if candidate != root and root not in candidate.parents:
            raise NativeAgentError("path escapes the workspace")
        if not candidate.is_file():
            raise NativeAgentError("not a file")
        return candidate

    def read_file(self, rel_path: str) -> dict[str, Any]:
        """Read a workspace file for the file viewer. Path-guarded to the workspace root (rejects
        traversal / absolute paths) and size-capped. An image file is flagged for inline ``<img>``
        preview (its bytes are fetched via :meth:`read_image` / ``/api/file-raw``); other binary
        content (NUL byte) is refused."""
        candidate = self._resolve_workspace_file(rel_path)
        mime = _IMAGE_EXTS.get(candidate.suffix.lower())
        if mime is not None:
            return {"path": rel_path, "image": True, "mime": mime, "binary": False,
                    "truncated": False, "content": ""}
        raw = candidate.read_bytes()
        truncated = len(raw) > _VIEW_MAX_BYTES
        raw = raw[:_VIEW_MAX_BYTES]
        if b"\x00" in raw:
            return {"path": rel_path, "binary": True, "image": False, "truncated": False, "content": ""}
        return {
            "path": rel_path,
            "binary": False,
            "image": False,
            "truncated": truncated,
            "content": raw.decode("utf-8", errors="replace"),
        }

    def read_image(self, rel_path: str) -> tuple[bytes, str]:
        """Bytes + content-type for an image workspace file — the ``/api/file-raw`` seam that backs
        the inline preview. Same path guard as :meth:`read_file`, restricted to known image
        extensions and size-capped."""
        candidate = self._resolve_workspace_file(rel_path)
        mime = _IMAGE_EXTS.get(candidate.suffix.lower())
        if mime is None:
            raise NativeAgentError("not an image file")
        data = candidate.read_bytes()
        if len(data) > _IMAGE_VIEW_MAX_BYTES:
            raise NativeAgentError(f"image exceeds the {_IMAGE_VIEW_MAX_BYTES}-byte preview limit")
        return data, mime

    def _token_for(self, run_id: str) -> str:
        with self._lock:
            token = self._run_tokens.get(run_id)
        if token is None:
            raise NativeAgentError(f"unknown run_id: {run_id}")
        return token


_TERMINAL = {"completed", "failed", "limited"}


def _make_handler(studio: StudioServer) -> type[BaseHTTPRequestHandler]:
    class StudioHandler(BaseHTTPRequestHandler):
        server_version = "MonoidStudio/0.1"
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
            if parsed.path == "/api/capabilities-catalog":
                self._write_json(studio.capabilities_catalog())
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
            if parsed.path == "/api/file":
                rel = (parse_qs(parsed.query).get("path") or [""])[0]
                try:
                    self._write_json(studio.read_file(rel))
                except NativeAgentError as exc:
                    self._write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if parsed.path == "/api/file-raw":
                # Raw image bytes for the inline <img> preview (read_file flags image=true).
                rel = (parse_qs(parsed.query).get("path") or [""])[0]
                try:
                    data, mime = studio.read_image(rel)
                except NativeAgentError as exc:
                    self._write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                else:
                    self._serve_bytes(data, mime)
                return
            if parsed.path == "/api/artifact":
                query = parse_qs(parsed.query)
                run_id = (query.get("run_id") or [""])[0]
                digest = (query.get("digest") or [""])[0]
                try:
                    data, name = studio.read_artifact(run_id, digest)
                except KeyError:
                    self.send_error(HTTPStatus.NOT_FOUND, "artifact not found")
                except (NativeAgentError, ValueError) as exc:
                    self._write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                else:
                    # Content-addressed → immutable + cacheable (ETag = the digest).
                    self._serve_bytes(
                        data, "application/x-tar", download_name=name,
                        headers={"ETag": f'"{digest}"', "Cache-Control": "immutable, max-age=31536000"},
                    )
                return
            if parsed.path == "/api/sessions":
                self._write_json(studio.sessions())
                return
            if parsed.path == "/api/proposal":
                run_id = (parse_qs(parsed.query).get("run_id") or [""])[0]
                try:
                    self._write_json(studio.proposal(run_id))
                except NativeAgentError as exc:
                    self._write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if parsed.path == "/api/proposal-file-raw":
                # Raw image bytes for a PROPOSED (not-yet-applied) file — the proposal-panel preview.
                query = parse_qs(parsed.query)
                run_id = (query.get("run_id") or [""])[0]
                rel = (query.get("path") or [""])[0]
                try:
                    data, mime = studio.proposal_image(run_id, rel)
                except KeyError:
                    self.send_error(HTTPStatus.NOT_FOUND, "not found")
                except (NativeAgentError, ValueError) as exc:
                    self._write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                else:
                    self._serve_bytes(data, mime)
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
            if parsed.path == "/api/subagent-events":
                query = parse_qs(parsed.query)
                child_run_id = (query.get("run_id") or [""])[0]
                from_seq = int((query.get("from") or ["0"])[0] or 0)
                self._write_json(studio.subagent_events(child_run_id, from_seq))
                return
            if parsed.path.startswith("/vendor/"):
                self._serve_vendor(parsed.path[len("/vendor/"):])
                return
            self.send_error(HTTPStatus.NOT_FOUND, "not found")

        # --- POST --------------------------------------------------------------------
        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/chat":
                    body = self._read_json()
                    message = str(body.get("message") or "").strip()
                    raw_attachments = body.get("attachments")
                    attachments = (
                        [a for a in raw_attachments if isinstance(a, dict)]
                        if isinstance(raw_attachments, list)
                        else []
                    )
                    if not message and not attachments:
                        self._write_json({"error": "message or attachment is required"}, HTTPStatus.BAD_REQUEST)
                        return
                    run_id = body.get("run_id")
                    if run_id:
                        result = studio.continue_chat(str(run_id), message, attachments)
                    else:
                        result = studio.start_chat(message, attachments)
                    self._write_json(result)
                    return
                if parsed.path == "/api/a2a-demo":
                    body = self._read_json()
                    task = str(body.get("task") or "").strip()
                    self._write_json(studio.start_a2a_demo(task))
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
                    raw = body.get("approved_paths")
                    approved = tuple(str(p) for p in raw) if isinstance(raw, list) else ()
                    self._write_json(studio.apply(run_id, approved_paths=approved))
                    return
                if parsed.path == "/api/export-package":
                    # Build → return the RECEIPT (digest + size + name). The bytes are fetched
                    # separately via GET /api/artifact?digest=… — no run_dir path crosses the wire.
                    body = self._read_json()
                    run_id = str(body.get("run_id") or "")
                    self._write_json(studio.export_package(run_id))
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
                    if "summary" in body:
                        kwargs["summary"] = str(body.get("summary") or "")
                    if "otel" in body:
                        kwargs["otel"] = bool(body.get("otel"))
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

        def _serve_vendor(self, rel: str) -> None:
            """Serve a vendored static asset from web/vendor/, guarding against path traversal."""
            base = _VENDOR_DIR.resolve()
            try:
                target = (base / rel).resolve()
                target.relative_to(base)
            except (ValueError, OSError):
                self.send_error(HTTPStatus.FORBIDDEN, "forbidden")
                return
            if not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
                return
            content_type = _VENDOR_CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
            self._serve_file(target, content_type)

        def _serve_file(self, path: Path, content_type: str, *, download_name: str = "") -> None:
            try:
                body = path.read_bytes()
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
                return
            self._serve_bytes(body, content_type, download_name=download_name)

        def _serve_bytes(
            self,
            body: bytes,
            content_type: str,
            *,
            download_name: str = "",
            headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if download_name:
                self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

    return StudioHandler
