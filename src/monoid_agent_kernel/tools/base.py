from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from jsonschema import Draft202012Validator, ValidationError

from monoid_agent_kernel.core.content import ContentPart, normalize_content_part
from monoid_agent_kernel.core.json_ingress import normalize_json_ingress, normalize_unicode_scalars
from monoid_agent_kernel.errors import ToolExecutionError

ToolSideEffect = Literal["read", "write", "artifact", "run", "shell"]
ToolPreviewKind = Literal["args", "shell", "web", "finish"]
ToolChangedPathsSource = Literal["path_args", "result_content"]
ToolResultPayloadKind = Literal["paths", "shell_exec"]


def _copy_with_fields(value: Any, /, **changes: Any) -> Any:
    """Preserve extension subclasses whose convenience ``__init__`` omits base fields."""

    cloned = copy(value)
    for name, replacement in changes.items():
        object.__setattr__(cloned, name, replacement)
    return cloned


def _require_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return normalize_unicode_scalars(value)


def _optional_text(value: Any, field_name: str) -> str | None:
    normalized = normalize_json_ingress(value)
    if normalized is None:
        return None
    if not isinstance(normalized, str):
        raise ValueError(f"{field_name} must be a string or null")
    return normalized


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


def normalize_tool_result(result: ToolResult) -> ToolResult:
    """Copy a handler result into the kernel's portable JSON/Unicode domain."""

    if not isinstance(result, ToolResult):
        raise ValueError("tool handler must return ToolResult")
    if not isinstance(result.content, dict):
        raise ValueError("tool result content must be an object")
    if not isinstance(result.media, (list, tuple)):
        raise ValueError("tool result media must be an array")
    content = normalize_json_ingress(result.content)
    if not isinstance(content, dict):
        raise ValueError("tool result content must be an object")
    return _copy_with_fields(
        result,
        ok=_require_bool(result.ok, "tool result ok"),
        content=content,
        error=_required_text(result.error, "tool result error"),
        error_code=_required_text(result.error_code, "tool result error_code"),
        retryable=_require_bool(result.retryable, "tool result retryable"),
        category=_required_text(result.category, "tool result category"),
        media=tuple(normalize_content_part(part) for part in result.media),
    )


class ToolContext(Protocol):
    def emit_artifact(
        self, path: str, kind: str, label: str | None, metadata: dict[str, Any]
    ) -> dict[str, Any]: ...

    def list_artifacts(self) -> list[dict[str, Any]]: ...

    def update_plan(self, items: list[dict[str, Any]]) -> None: ...

    def finish(self, summary: str, outputs: list[str], notes: str | None) -> None: ...

    def execute_shell(self, args: dict[str, Any]) -> dict[str, Any]: ...

    def run_script(self, args: dict[str, Any]) -> dict[str, Any]: ...

    def list_jobs(self) -> list[dict[str, Any]]: ...

    def job_status(self, args: dict[str, Any]) -> dict[str, Any]: ...

    def job_logs(self, args: dict[str, Any]) -> dict[str, Any]: ...

    def job_cancel(self, args: dict[str, Any]) -> dict[str, Any]: ...

    def job_wait(self, args: dict[str, Any]) -> dict[str, Any]: ...

    def request_human_input(self, args: dict[str, Any]) -> dict[str, Any]: ...

    def spawn_subagent(self, args: dict[str, Any]) -> dict[str, Any]: ...

    def execute_web_search(self, args: dict[str, Any]) -> dict[str, Any]: ...

    def execute_web_fetch(self, args: dict[str, Any]) -> dict[str, Any]: ...

    def execute_web_context(self, args: dict[str, Any]) -> dict[str, Any]: ...

    def path_allowed(self, path: str, operation: str = "read") -> bool: ...

    def search_tools(self, args: dict[str, Any]) -> dict[str, Any]: ...

    def capability_token(self, capability: str) -> str | None:
        """The access handle (``token_ref``) of the lease the loop acquired for ``capability``
        before invoking this tool, or ``None`` if no broker/lease applies. The handle is resolved
        to the real secret at the edge (gateway), never in the core."""
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
        """Stage an outbound side-effect in the current tool invocation instead of doing the IO
        inline. A valid successful tool result commits the request to the run's durable outbox for
        an *edge* to drain; any failed, malformed, cancelled, or timed-out invocation discards it.
        The request carries the capability lease handle for ``capability`` (never a secret). With
        ``expect_ack`` the edge delivers the send's receipt back as an inbox message (non-park).
        Returns ``{"status": "staged", "request_id": ...}``."""
        ...


SyncToolHandler = Callable[[ToolContext, dict[str, Any]], ToolResult]
AsyncToolHandler = Callable[[ToolContext, dict[str, Any]], Awaitable[ToolResult]]
ToolHandler = SyncToolHandler | AsyncToolHandler


@dataclass(frozen=True)
class ToolSpec:
    """A registered tool: its identity, JSON-Schema input, and handler.

    ``input_schema`` is a JSON Schema (Draft 2020-12) the registry validates calls against;
    ``handler`` is a synchronous or async ``(ToolContext, args) -> ToolResult`` callable.
    Async handlers are awaited on the run loop; synchronous handlers run in a worker thread.
    ``side_effect`` and the declarative hint fields let the engine drive previews/diffs without
    branching on tool identity. Author one by hand, or generate it from a typed Python function
    with the :func:`~monoid_agent_kernel.tool` decorator (``tools/decorator.py``).
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


def normalize_tool_spec(spec: ToolSpec) -> ToolSpec:
    """Copy one model-visible tool definition into the portable JSON domain."""

    examples = normalize_json_ingress(spec.examples)
    return _copy_with_fields(
        spec,
        id=_required_text(spec.id, "tool id"),
        description=_required_text(spec.description, "tool description"),
        input_schema=normalize_json_ingress(spec.input_schema),
        capability=_required_text(spec.capability, "tool capability"),
        side_effect=_required_text(spec.side_effect, "tool side_effect"),
        provider_name=_optional_text(spec.provider_name, "tool provider_name"),
        path_args=tuple(_required_text(value, "tool path_args item") for value in spec.path_args),
        preview_kind=_required_text(spec.preview_kind, "tool preview_kind"),
        emits_workspace_diff=_require_bool(
            spec.emits_workspace_diff,
            "tool emits_workspace_diff",
        ),
        changed_paths_source=_required_text(
            spec.changed_paths_source,
            "tool changed_paths_source",
        ),
        result_payload_kind=_required_text(
            spec.result_payload_kind,
            "tool result_payload_kind",
        ),
        skip_emit_if_background=_require_bool(
            spec.skip_emit_if_background,
            "tool skip_emit_if_background",
        ),
        guidance=normalize_json_ingress(spec.guidance),
        examples=tuple(examples),
        annotations=normalize_json_ingress(spec.annotations),
    )


class ToolProvider(Protocol):
    def get_tools(self, context: ToolContext) -> Iterable[ToolSpec]: ...


class DynamicToolProvider(Protocol):
    def get_tools_for_turn(self, context: ToolContext, turn: Any) -> Iterable[ToolSpec]: ...


@dataclass
class ToolRegistry:
    _by_id: dict[str, ToolSpec] = field(default_factory=dict)
    _by_exported_name: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        spec = normalize_tool_spec(spec)
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
