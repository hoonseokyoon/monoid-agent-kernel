from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from jsonschema import Draft202012Validator, ValidationError

from native_agent_runner.core.content import ContentPart
from native_agent_runner.errors import ToolExecutionError

ToolSideEffect = Literal["read", "write", "artifact", "run", "shell"]
ToolPreviewKind = Literal["args", "shell", "web"]
ToolChangedPathsSource = Literal["path_args", "result_content"]
ToolResultPayloadKind = Literal["paths", "shell_exec"]


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    error_code: str = ""
    retryable: bool = False
    category: str = "tool"
    # Non-text media the tool produced (chart/screenshot/read image/document). Carried by
    # reference (source_ref), not bytes; resolved at wire-build time and delivered to the
    # model per provider (a follow-up user message for OpenAI/gateway). Typed ``ContentPart``
    # so a tool can return an image OR a document (PDF), not images only.
    media: tuple[ContentPart, ...] = ()

    def to_observation(self) -> dict[str, Any]:
        """Model-facing payload. The handler's ``content`` lives under ``result`` so
        domain keys can never collide with the ``ok``/``error`` envelope. Media travel
        separately (see ``ToolObservation.media``), not inside this dict."""
        obs: dict[str, Any] = {"ok": self.ok, "result": self.content}
        if not self.ok:
            obs["error"] = {
                "message": self.error,
                "code": self.error_code,
                "category": self.category or "tool",
                "retryable": self.retryable,
            }
        return obs


class ToolContext(Protocol):
    def emit_artifact(self, path: str, kind: str, label: str | None, metadata: dict[str, Any]) -> dict[str, Any]:
        ...

    def list_artifacts(self) -> list[dict[str, Any]]:
        ...

    def update_plan(self, items: list[dict[str, Any]]) -> None:
        ...

    def finish(self, summary: str, outputs: list[str], notes: str | None) -> None:
        ...

    def execute_shell(self, args: dict[str, Any]) -> dict[str, Any]:
        ...

    def run_script(self, args: dict[str, Any]) -> dict[str, Any]:
        ...

    def list_jobs(self) -> list[dict[str, Any]]:
        ...

    def job_status(self, args: dict[str, Any]) -> dict[str, Any]:
        ...

    def job_logs(self, args: dict[str, Any]) -> dict[str, Any]:
        ...

    def job_cancel(self, args: dict[str, Any]) -> dict[str, Any]:
        ...

    def job_wait(self, args: dict[str, Any]) -> dict[str, Any]:
        ...

    def request_human_input(self, args: dict[str, Any]) -> dict[str, Any]:
        ...

    def spawn_subagent(self, args: dict[str, Any]) -> dict[str, Any]:
        ...

    def execute_web_search(self, args: dict[str, Any]) -> dict[str, Any]:
        ...

    def execute_web_fetch(self, args: dict[str, Any]) -> dict[str, Any]:
        ...

    def execute_web_context(self, args: dict[str, Any]) -> dict[str, Any]:
        ...

    def search_tools(self, args: dict[str, Any]) -> dict[str, Any]:
        ...

    def capability_token(self, capability: str) -> str | None:
        """The access handle (``token_ref``) of the lease the loop acquired for ``capability``
        before invoking this tool, or ``None`` if no broker/lease applies. The handle is resolved
        to the real secret at the edge (gateway), never in the core."""
        ...

    def capability_available(self, capability: str) -> bool:
        """Whether the run holds ``capability`` this turn — a broker lease was acquired for it OR a
        tool declaring it is bound. Use this (not ``capability_token``) to gate optional handler
        behavior that should work in the common no-broker config, e.g. ``fs.read`` falling back to
        the media path when ``fs.read_media`` (``media.input``) is bound."""
        ...

    def emit_outbox(
        self,
        destination: str,
        payload: dict[str, Any],
        *,
        capability: str = "",
        idempotency_key: str = "",
        expect_ack: bool = False,
        reply_to: str = "",
    ) -> dict[str, Any]:
        """Stage a durable outbound side-effect (send an email, call a webhook) instead of doing the
        IO inline. The request is appended to the run's outbox (checkpointed) and an *edge* drains it
        later; it carries the capability lease handle for ``capability`` (never a secret). With
        ``expect_ack`` the edge delivers the send's receipt back as an inbox message (non-park).
        Returns ``{"status": "staged", "request_id": ...}``."""
        ...


ToolHandler = Callable[[ToolContext, dict[str, Any]], ToolResult]


@dataclass(frozen=True)
class ToolSpec:
    """A registered tool: its identity, JSON-Schema input, and handler.

    ``input_schema`` is a JSON Schema (Draft 2020-12) the registry validates calls against;
    ``handler`` is a ``(ToolContext, args) -> ToolResult`` callable. ``side_effect`` and the
    declarative hint fields let the engine drive previews/diffs without branching on tool
    identity. Author one by hand, or generate it from a typed Python function with the
    :func:`~native_agent_runner.tool` decorator (``tools/decorator.py``).
    """

    id: str
    description: str
    input_schema: dict[str, Any]
    capability: str
    side_effect: ToolSideEffect
    handler: ToolHandler
    provider_name: str | None = None
    path_args: tuple[str, ...] = ()
    # Declarative hints the engine uses instead of branching on tool identity.
    preview_kind: ToolPreviewKind = "args"
    emits_workspace_diff: bool = False
    changed_paths_source: ToolChangedPathsSource = "path_args"
    result_payload_kind: ToolResultPayloadKind = "paths"
    skip_emit_if_background: bool = False
    guidance: dict[str, Any] = field(default_factory=dict)
    examples: tuple[dict[str, Any], ...] = ()
    annotations: dict[str, Any] = field(default_factory=dict)

    @property
    def exported_name(self) -> str:
        return self.provider_name or self.id.replace(".", "_")


class ToolProvider(Protocol):
    def get_tools(self, context: ToolContext) -> Iterable[ToolSpec]:
        ...


class DynamicToolProvider(Protocol):
    def get_tools_for_turn(self, context: ToolContext, turn: Any) -> Iterable[ToolSpec]:
        ...


@dataclass
class ToolRegistry:
    _by_id: dict[str, ToolSpec] = field(default_factory=dict)
    _by_exported_name: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        if spec.id in self._by_id:
            raise ValueError(f"duplicate tool id: {spec.id}")
        if spec.exported_name in self._by_exported_name:
            raise ValueError(f"duplicate exported tool name: {spec.exported_name}")
        self._by_id[spec.id] = spec
        self._by_exported_name[spec.exported_name] = spec

    def register_many(self, specs: Iterable[ToolSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def resolve(self, name: str) -> ToolSpec:
        if name in self._by_id:
            return self._by_id[name]
        if name in self._by_exported_name:
            return self._by_exported_name[name]
        raise ToolExecutionError(f"unknown tool: {name}", error_code="tool_unknown")

    def validate_args(self, spec: ToolSpec, args: dict[str, Any]) -> None:
        try:
            Draft202012Validator(spec.input_schema).validate(args)
        except ValidationError as exc:
            raise ToolExecutionError(
                f"invalid arguments for {spec.id}: {exc.message}",
                error_code="tool_args_invalid",
            ) from exc

    def specs(self) -> list[ToolSpec]:
        return list(self._by_id.values())

