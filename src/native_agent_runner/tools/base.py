from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from jsonschema import Draft202012Validator, ValidationError

from native_agent_runner.errors import ToolExecutionError
from native_agent_runner.tools.policy import NormalizedToolPolicy, ToolPolicy, normalize_tool_policy

ToolSideEffect = Literal["read", "write", "artifact", "run", "shell"]


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    error_code: str = ""

    def to_observation(self) -> dict[str, Any]:
        if self.ok:
            return {"ok": True, **self.content}
        return {"ok": False, **self.content, "error": self.error, "error_code": self.error_code}


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

    def execute_web_search(self, args: dict[str, Any]) -> dict[str, Any]:
        ...

    def execute_web_fetch(self, args: dict[str, Any]) -> dict[str, Any]:
        ...

    def execute_web_context(self, args: dict[str, Any]) -> dict[str, Any]:
        ...


ToolHandler = Callable[[ToolContext, dict[str, Any]], ToolResult]


@dataclass(frozen=True)
class ToolSpec:
    id: str
    description: str
    input_schema: dict[str, Any]
    capability: str
    side_effect: ToolSideEffect
    handler: ToolHandler
    provider_name: str | None = None
    path_args: tuple[str, ...] = ()

    @property
    def exported_name(self) -> str:
        return self.provider_name or self.id.replace(".", "_")


class ToolProvider(Protocol):
    def get_tools(self, context: ToolContext) -> Iterable[ToolSpec]:
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

    def policy_view(self, policy: ToolPolicy, capabilities: frozenset[str]) -> NormalizedToolPolicy:
        return normalize_tool_policy(policy, self.specs(), capabilities)

    def visible_specs(self, policy: NormalizedToolPolicy) -> list[ToolSpec]:
        visible = set(policy.visible_tools)
        return [spec for spec in self.specs() if spec.id in visible]

    def provider_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": spec.exported_name,
                "description": spec.description,
                "parameters": spec.input_schema,
            }
            for spec in self.specs()
        ]
