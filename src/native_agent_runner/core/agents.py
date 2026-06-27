from __future__ import annotations

import difflib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, Union

from native_agent_runner.core._util import canonical_sha256
from native_agent_runner.core.spec import ModelConfig, RunLimits, RunMode
from native_agent_runner.core.tool_surface import (
    ToolAuthorization,
    ToolAuthorizationDecision,
    ToolExposure,
    ToolGuidance,
    ToolQuota,
    ToolScope,
)
from native_agent_runner.errors import AgentConfigError
from native_agent_runner.tools.base import ToolRegistry, ToolSpec

ToolRefKind = Literal["registry"]
SubagentContext = Literal["fresh", "fork"]


@dataclass(frozen=True)
class PromptSpec:
    system_prompt_base: str | None = None
    persona_segments: tuple[str, ...] = ()
    runtime_segments: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, payload: dict[str, Any] | None) -> PromptSpec:
        if payload is None:
            return cls()
        if isinstance(payload, PromptSpec):
            return payload
        if not isinstance(payload, dict):
            raise ValueError("prompt must be an object")
        base = payload.get("system_prompt_base", payload.get("base"))
        return cls(
            system_prompt_base=None if base is None else str(base),
            persona_segments=_str_tuple(payload.get("persona_segments", payload.get("persona"))),
            runtime_segments=_str_tuple(payload.get("runtime_segments", payload.get("runtime"))),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "system_prompt_base": self.system_prompt_base,
            "persona_segments": list(self.persona_segments),
            "runtime_segments": list(self.runtime_segments),
        }


@dataclass(frozen=True)
class RegistryToolRef:
    tool_id: str
    kind: ToolRefKind = "registry"

    @classmethod
    def from_json(cls, payload: dict[str, Any] | str) -> RegistryToolRef:
        if isinstance(payload, str):
            return cls(tool_id=payload)
        if isinstance(payload, RegistryToolRef):
            return payload
        if not isinstance(payload, dict):
            raise ValueError("tool ref must be an object or string")
        kind = str(payload.get("kind") or "registry")
        if kind != "registry":
            raise ValueError("only registry tool refs are supported")
        tool_id = str(payload.get("tool_id") or payload.get("id") or "").strip()
        if not tool_id:
            raise ValueError("registry tool ref requires tool_id")
        return cls(tool_id=tool_id)

    def to_json(self) -> dict[str, str]:
        return {"kind": self.kind, "tool_id": self.tool_id}


@dataclass(frozen=True)
class ToolBinding:
    binding_id: str
    ref: RegistryToolRef | str
    model_name: str | None = None
    exposure: ToolExposure = "immediate"
    authorization: ToolAuthorizationDecision = "allow"
    guidance: ToolGuidance = field(default_factory=ToolGuidance)
    scope: ToolScope = field(default_factory=ToolScope)
    quota: ToolQuota = field(default_factory=ToolQuota)
    title: str = ""
    summary: str = ""
    risk: str = ""
    requires_approval: bool | None = None
    reason: str = ""
    runtime: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ergonomics: accept a bare tool-id string for ``ref`` and normalize it to a
        # RegistryToolRef, so ``ToolBinding(binding_id="x", ref="fs.read")`` works (frozen-safe).
        if isinstance(self.ref, str):
            object.__setattr__(self, "ref", RegistryToolRef(self.ref))

    @classmethod
    def for_tool(
        cls,
        tool_id: str,
        *,
        binding_id: str | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> ToolBinding:
        """One-token binding for a registry tool: ``ToolBinding.for_tool("fs.read")``.

        Defaults ``binding_id`` to ``tool_id`` and derives ``model_name`` from it (dots →
        underscores), matching :func:`generated_tool_bindings`. Extra binding fields (scope,
        runtime, exposure, …) pass through as keyword arguments."""
        resolved_binding_id = binding_id or tool_id
        return cls(
            binding_id=resolved_binding_id,
            ref=RegistryToolRef(tool_id),
            model_name=model_name or resolved_binding_id.replace(".", "_"),
            **kwargs,
        )

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ToolBinding:
        if isinstance(payload, ToolBinding):
            return payload
        if not isinstance(payload, dict):
            raise ValueError("tool binding must be an object")
        ref_payload = payload.get("ref")
        if ref_payload is None:
            ref_payload = {"tool_id": payload.get("tool") or payload.get("tool_id") or payload.get("id")}
        ref = RegistryToolRef.from_json(ref_payload)
        binding_id = str(payload.get("binding_id") or payload.get("id") or ref.tool_id).strip()
        if not binding_id:
            raise ValueError("tool binding requires binding_id")
        model_name_raw = payload.get("model_name")
        model_name = None if model_name_raw is None else str(model_name_raw).strip()
        if model_name == "":
            raise ValueError("tool binding model_name cannot be empty")
        exposure = str(payload.get("exposure") or "immediate")
        if exposure not in {"immediate", "searchable", "hidden"}:
            raise ValueError("tool binding exposure must be immediate, searchable, or hidden")
        authorization = str(payload.get("authorization") or "allow")
        if authorization not in {"allow", "ask", "deny"}:
            raise ValueError("tool binding authorization must be allow, ask, or deny")
        runtime = payload.get("runtime") or {}
        metadata = payload.get("metadata") or {}
        if not isinstance(runtime, Mapping):
            raise ValueError("tool binding runtime must be an object")
        if not isinstance(metadata, Mapping):
            raise ValueError("tool binding metadata must be an object")
        return cls(
            binding_id=binding_id,
            ref=ref,
            model_name=model_name,
            exposure=exposure,  # type: ignore[arg-type]
            authorization=authorization,  # type: ignore[arg-type]
            guidance=ToolGuidance.from_json(payload.get("guidance")),
            scope=ToolScope.from_json(payload.get("scope")),
            quota=ToolQuota.from_json(payload.get("quota")),
            title=str(payload.get("title") or ""),
            summary=str(payload.get("summary") or ""),
            risk=str(payload.get("risk") or ""),
            requires_approval=(
                None if "requires_approval" not in payload else bool(payload["requires_approval"])
            ),
            reason=str(payload.get("reason") or ""),
            runtime=dict(runtime),
            metadata=dict(metadata),
        )

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "binding_id": self.binding_id,
            "ref": self.ref.to_json(),
            "exposure": self.exposure,
            "authorization": self.authorization,
            "guidance": self.guidance.to_json(),
            "scope": self.scope.to_json(),
            "quota": self.quota.to_json(),
            "runtime": dict(self.runtime),
            "metadata": dict(self.metadata),
        }
        if self.model_name is not None:
            payload["model_name"] = self.model_name
        if self.title:
            payload["title"] = self.title
        if self.summary:
            payload["summary"] = self.summary
        if self.risk:
            payload["risk"] = self.risk
        if self.requires_approval is not None:
            payload["requires_approval"] = self.requires_approval
        if self.reason:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True)
class ToolSearchConfig:
    enabled: bool = True
    top_k: int = 5
    binding_id: str = "tool.search"
    model_name: str = "tool_search"

    @classmethod
    def from_json(cls, payload: dict[str, Any] | None) -> ToolSearchConfig:
        if payload is None:
            return cls()
        if isinstance(payload, ToolSearchConfig):
            return payload
        if not isinstance(payload, dict):
            raise ValueError("tool_search must be an object")
        top_k = int(payload.get("top_k", payload.get("search_top_k", 5)))
        if top_k < 1:
            raise ValueError("tool_search.top_k must be positive")
        binding_id = str(payload.get("binding_id") or "tool.search").strip()
        model_name = str(payload.get("model_name") or "tool_search").strip()
        if not binding_id or not model_name:
            raise ValueError("tool_search binding_id and model_name are required")
        return cls(
            enabled=bool(payload.get("enabled", True)),
            top_k=top_k,
            binding_id=binding_id,
            model_name=model_name,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "top_k": self.top_k,
            "binding_id": self.binding_id,
            "model_name": self.model_name,
        }


@dataclass(frozen=True)
class AgentDefinition:
    id: str
    version: str = "1"
    description: str = ""
    model: ModelConfig | None = None
    prompt: PromptSpec = field(default_factory=PromptSpec)
    tools: tuple[ToolBinding, ...] = ()
    tool_search: ToolSearchConfig = field(default_factory=ToolSearchConfig)
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> AgentDefinition:
        if not isinstance(payload, dict):
            raise ValueError("agent_definition must be an object")
        agent_id = str(payload.get("id") or payload.get("name") or "").strip()
        if not agent_id:
            raise ValueError("agent_definition.id is required")
        model_payload = payload.get("model")
        return cls(
            id=agent_id,
            version=str(payload.get("version") or "1"),
            description=str(payload.get("description") or ""),
            model=ModelConfig.from_json(model_payload) if model_payload is not None else None,
            prompt=PromptSpec.from_json(payload.get("prompt")),
            tools=tuple(ToolBinding.from_json(item) for item in payload.get("tools") or ()),
            tool_search=ToolSearchConfig.from_json(payload.get("tool_search")),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "description": self.description,
            "model": None if self.model is None else self.model.to_json(),
            "prompt": self.prompt.to_json(),
            "tools": [binding.to_json() for binding in self.tools],
            "tool_search": self.tool_search.to_json(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentRuntimeConfig:
    """Effective configuration for a single run turn: *what* the agent is right now.

    This is what a :class:`RuntimeConfigProvider` returns and the engine reads at
    bootstrap and at every turn boundary — the model, system prompt, and tool bindings
    that decide which registry tools are visible/searchable/executable this turn. It is
    deliberately separate from :class:`AgentRunSpec` (the immutable session descriptor:
    workspace, limits, permissions), so returning a different config between turns hot-
    reloads the agent. ``config_hash`` drives change detection; build one quickly from a
    reusable blueprint with :meth:`from_definition`.
    """

    definition_id: str
    config_version: int = 1
    model: ModelConfig | None = None
    prompt: PromptSpec = field(default_factory=PromptSpec)
    tools: tuple[ToolBinding, ...] = ()
    tool_search: ToolSearchConfig = field(default_factory=ToolSearchConfig)
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> AgentRuntimeConfig:
        if isinstance(payload, AgentRuntimeConfig):
            return payload
        if not isinstance(payload, dict):
            raise ValueError("runtime config must be an object")
        model_payload = payload.get("model")
        return cls(
            definition_id=str(payload.get("definition_id") or payload.get("agent_id") or "default"),
            config_version=int(payload.get("config_version") or payload.get("version") or 1),
            model=ModelConfig.from_json(model_payload) if model_payload is not None else None,
            prompt=PromptSpec.from_json(payload.get("prompt")),
            tools=tuple(ToolBinding.from_json(item) for item in payload.get("tools") or ()),
            tool_search=ToolSearchConfig.from_json(payload.get("tool_search")),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def from_definition(
        cls,
        definition: AgentDefinition,
        *,
        config_version: int = 1,
    ) -> AgentRuntimeConfig:
        return cls(
            definition_id=definition.id,
            config_version=config_version,
            model=definition.model,
            prompt=definition.prompt,
            tools=definition.tools,
            tool_search=definition.tool_search,
            metadata={"agent_definition_version": definition.version, **definition.metadata},
        )

    @property
    def config_hash(self) -> str:
        return canonical_sha256(self._json_payload())

    def to_json(self) -> dict[str, Any]:
        payload = self._json_payload()
        payload["config_hash"] = self.config_hash
        return payload

    def _json_payload(self) -> dict[str, Any]:
        return {
            "definition_id": self.definition_id,
            "config_version": self.config_version,
            "model": None if self.model is None else self.model.to_json(),
            "prompt": self.prompt.to_json(),
            "tools": [binding.to_json() for binding in self.tools],
            "tool_search": self.tool_search.to_json(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SubagentDefinition:
    """How a parent run delegates to a subagent (agent-as-tool). Mirrors the Claude
    Code subagent model: tools/model/mode/limits default to *inheriting* the parent's,
    and ``tools``/``disallowed_tools`` filter the parent's tool set — a subagent can
    never exceed the parent (the allowlist is resolved against the parent's bindings; see
    ``AgentLoop._resolve_child_config``).

    - ``tools=None`` inherits ALL of the parent's tool bindings; a tuple is an allowlist
      matched (fnmatch) against each parent binding's tool id / binding id / model name,
      so patterns like ``"mcp.*"`` or ``"fs.read"`` work.
    - ``disallowed_tools`` is a denylist applied after the allowlist (deny wins).
    - ``model``/``mode``/``limits``/``tool_search`` are ``None`` to inherit the parent's.
    - ``description`` is surfaced to the model in the ``agent.spawn`` tool so it can pick
      the right subagent (Claude selects subagents by their description).
    - ``context="fork"`` makes the child inherit the parent's conversation snapshot AND
      the parent's prompt/tools/model (the definition's own prompt/tools/model are
      ignored) — "continue as me in an isolated branch". ``"fresh"`` (default) is the
      normal isolated subagent that only sees the task prompt.
    """

    description: str = ""
    prompt: PromptSpec = field(default_factory=PromptSpec)
    model: ModelConfig | None = None
    tools: tuple[str, ...] | None = None
    disallowed_tools: tuple[str, ...] = ()
    mode: RunMode | None = None
    limits: RunLimits | None = None
    tool_search: ToolSearchConfig | None = None
    context: SubagentContext = "fresh"
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_frontmatter(cls, meta: Mapping[str, Any], body: str) -> SubagentDefinition:
        """Build a definition from a parsed ``.claude/agents``-style file: the YAML
        frontmatter ``meta`` plus the markdown ``body`` (which becomes the system
        prompt for a fresh subagent; ignored for a fork). Mirrors the Claude field
        names — ``tools`` omitted means inherit all parent tools."""
        sentinel = object()
        model_raw = meta.get("model")
        if model_raw in (None, "inherit", ""):
            model = None
        elif isinstance(model_raw, Mapping):
            model = ModelConfig.from_json(dict(model_raw))
        else:
            model = ModelConfig(model=str(model_raw))

        tools_raw = meta.get("tools", sentinel)
        if tools_raw is sentinel or tools_raw is None:
            tools: tuple[str, ...] | None = None
        elif isinstance(tools_raw, str):
            tools = (tools_raw,)
        else:
            tools = tuple(str(item) for item in tools_raw)

        disallowed_raw = meta.get("disallowed_tools", meta.get("disallowedTools", ()))
        if isinstance(disallowed_raw, str):
            disallowed = (disallowed_raw,)
        else:
            disallowed = tuple(str(item) for item in disallowed_raw or ())

        mode_raw = meta.get("mode") or meta.get("permissionMode")
        mode = mode_raw if mode_raw in ("read-only", "propose", "apply") else None

        context_raw = str(meta.get("context") or "fresh")
        context: SubagentContext = "fork" if context_raw == "fork" else "fresh"

        body_text = body.strip()
        prompt = PromptSpec(persona_segments=(body_text,)) if body_text else PromptSpec()
        return cls(
            description=str(meta.get("description") or ""),
            prompt=prompt,
            model=model,
            tools=tools,
            disallowed_tools=disallowed,
            mode=mode,
            context=context,
        )

    @classmethod
    def from_runtime_config(
        cls, config: AgentRuntimeConfig, *, description: str = ""
    ) -> SubagentDefinition:
        """Adapt an explicit runtime config into a definition: its bindings' tool ids
        become a fixed allowlist and its prompt/model/tool_search carry over. The
        allowlist is still ceiling-bound — the parent must also expose those tools."""
        return cls(
            description=description,
            prompt=config.prompt,
            model=config.model,
            tools=tuple(binding.ref.tool_id for binding in config.tools),
            tool_search=config.tool_search,
            metadata=dict(config.metadata),
        )


@dataclass(frozen=True)
class BoundTool:
    binding: ToolBinding
    base_spec: ToolSpec
    model_spec: ToolSpec
    model_name: str
    authorization: ToolAuthorization

    @property
    def binding_id(self) -> str:
        return self.binding.binding_id

    @property
    def exposure(self) -> ToolExposure:
        return self.binding.exposure

    @property
    def runtime(self) -> dict[str, Any]:
        return self.binding.runtime


@dataclass(frozen=True)
class BoundToolCatalog:
    tools: tuple[BoundTool, ...]
    tool_search: ToolSearchConfig
    search_tool: ToolSpec | None = None

    @property
    def by_binding_id(self) -> dict[str, BoundTool]:
        return {tool.binding_id: tool for tool in self.tools}

    @property
    def by_model_name(self) -> dict[str, BoundTool]:
        by_name: dict[str, BoundTool] = {}
        for tool in self.tools:
            by_name[tool.model_name] = tool
            by_name[tool.binding_id] = tool
        return by_name

    def resolve_model_call(self, name: str) -> BoundTool | None:
        return self.by_model_name.get(name)


class RuntimeConfigProvider(Protocol):
    """Supplies the per-run :class:`AgentRuntimeConfig` to the engine.

    The engine calls :meth:`current_config` at bootstrap and at each turn boundary, so a
    dynamic implementation can hot-reload the model/prompt/tools mid-run (multi-tenant
    routing, live tool toggles, A/B rollout). Returning ``None`` fails the run with
    ``agent_config_missing``. For a fixed config use :class:`StaticRuntimeConfigProvider`
    or pass a bare :class:`AgentRuntimeConfig` / a ``lambda run_id: config`` wherever a
    provider is expected — :class:`~native_agent_runner.AgentLoop` coerces all three.
    """

    def current_config(self, run_id: str) -> AgentRuntimeConfig | None:
        ...


@dataclass(frozen=True)
class StaticRuntimeConfigProvider(RuntimeConfigProvider):
    """A :class:`RuntimeConfigProvider` that always returns the same config.

    The simplest provider: ignores ``run_id`` and serves a fixed
    :class:`AgentRuntimeConfig`. Enough for a one-shot run or a single fixed agent.
    """

    config: AgentRuntimeConfig

    def current_config(self, run_id: str) -> AgentRuntimeConfig | None:
        del run_id
        return self.config


@dataclass(frozen=True)
class _CallableRuntimeConfigProvider(RuntimeConfigProvider):
    """Adapts a ``Callable[[str], AgentRuntimeConfig | None]`` to the provider protocol."""

    func: Callable[[str], AgentRuntimeConfig | None]

    def current_config(self, run_id: str) -> AgentRuntimeConfig | None:
        return self.func(run_id)


# What callers may pass anywhere a RuntimeConfigProvider is expected. AgentLoop coerces
# all of these to a RuntimeConfigProvider via coerce_runtime_config_provider().
RuntimeConfigSource = Union[
    RuntimeConfigProvider,
    AgentRuntimeConfig,
    Callable[[str], "AgentRuntimeConfig | None"],
]


def static_runtime_config(config: AgentRuntimeConfig) -> StaticRuntimeConfigProvider:
    """Wrap a fixed :class:`AgentRuntimeConfig` in a :class:`StaticRuntimeConfigProvider`."""
    return StaticRuntimeConfigProvider(config)


def coerce_runtime_config_provider(source: RuntimeConfigSource) -> RuntimeConfigProvider:
    """Normalize a config source to a :class:`RuntimeConfigProvider`.

    Accepts a provider (anything with ``current_config``), a bare
    :class:`AgentRuntimeConfig` (wrapped static), or a ``callable(run_id) -> config``.
    """
    if hasattr(source, "current_config"):
        return source  # type: ignore[return-value]
    if isinstance(source, AgentRuntimeConfig):
        return StaticRuntimeConfigProvider(source)
    if callable(source):
        return _CallableRuntimeConfigProvider(source)
    raise TypeError(
        "runtime_config_provider must be a RuntimeConfigProvider, an AgentRuntimeConfig, "
        f"or a callable(run_id) -> AgentRuntimeConfig; got {type(source).__name__}"
    )


def generated_tool_bindings(
    tool_specs: Iterable[ToolSpec],
    *,
    exposure: ToolExposure = "immediate",
    authorization: ToolAuthorizationDecision = "allow",
) -> tuple[ToolBinding, ...]:
    return tuple(
        ToolBinding(
            binding_id=tool.id,
            ref=RegistryToolRef(tool.id),
            model_name=tool.exported_name,
            exposure=exposure,
            authorization=authorization,
            guidance=ToolGuidance.from_json(tool.guidance),
            title=tool.id,
            summary=tool.description.split("\n", 1)[0],
            risk=_risk_for(tool),
        )
        for tool in tool_specs
        if tool.id != "tool.search"
    )


def _suggest(name: str, candidates: Iterable[str]) -> str:
    """Return a ' Did you mean 'x'?' hint for the closest candidate, or '' if none close."""
    matches = difflib.get_close_matches(name, list(candidates), n=1, cutoff=0.6)
    return f" Did you mean '{matches[0]}'?" if matches else ""


def _unknown_tool_message(tool_id: str, available: Iterable[str]) -> str:
    ids = sorted(available)
    listed = ", ".join(ids) if ids else "(none registered)"
    return (
        f"runtime config references unknown registry tool: {tool_id}."
        f"{_suggest(tool_id, ids)} Available tools: {listed}"
    )


def compile_bound_tool_catalog(config: AgentRuntimeConfig, registry: ToolRegistry) -> BoundToolCatalog:
    specs = {tool.id: tool for tool in registry.specs()}
    search_tool = specs.get("tool.search")
    bound: list[BoundTool] = []
    seen_binding_ids: set[str] = set()
    seen_model_names: set[str] = set()
    seen_call_names: dict[str, str] = {}

    def reserve_call_name(name: str, owner: str) -> None:
        previous = seen_call_names.get(name)
        if previous is not None and previous != owner:
            raise AgentConfigError(f"duplicate tool call name: {name}")
        seen_call_names[name] = owner

    if config.tool_search.enabled:
        reserve_call_name(config.tool_search.binding_id, "tool_search")
        reserve_call_name(config.tool_search.model_name, "tool_search")

    for binding in config.tools:
        if binding.binding_id in seen_binding_ids:
            raise AgentConfigError(f"duplicate tool binding_id: {binding.binding_id}")
        if config.tool_search.enabled and binding.binding_id == config.tool_search.binding_id:
            raise AgentConfigError(f"duplicate tool binding_id: {binding.binding_id}")
        seen_binding_ids.add(binding.binding_id)
        reserve_call_name(binding.binding_id, binding.binding_id)
        spec = specs.get(binding.ref.tool_id)
        if spec is None:
            raise AgentConfigError(_unknown_tool_message(binding.ref.tool_id, specs.keys()))
        model_name = _resolved_model_name(binding, spec)
        if model_name in seen_model_names:
            raise AgentConfigError(f"duplicate tool model_name: {model_name}")
        if config.tool_search.enabled and model_name == config.tool_search.model_name:
            raise AgentConfigError(f"duplicate tool model_name: {model_name}")
        seen_model_names.add(model_name)
        reserve_call_name(model_name, binding.binding_id)
        _validate_binding_runtime(binding)
        authorization = ToolAuthorization(
            tool_id=spec.id,
            binding_id=binding.binding_id,
            model_name=model_name,
            decision=binding.authorization,
            reason=binding.reason or f"{binding.authorization}_by_tool_binding",
            exposure=binding.exposure,
            quota=binding.quota,
            scope=binding.scope,
            surface_scope=binding.scope,
            runtime=dict(binding.runtime),
        )
        bound.append(
            BoundTool(
                binding=binding,
                base_spec=spec,
                model_spec=_model_tool_spec(spec, binding, model_name),
                model_name=model_name,
                authorization=authorization,
            )
        )
    return BoundToolCatalog(tools=tuple(bound), tool_search=config.tool_search, search_tool=search_tool)


def validate_runtime_config(config: AgentRuntimeConfig, registry_or_specs: ToolRegistry | Iterable[ToolSpec]) -> None:
    if isinstance(registry_or_specs, ToolRegistry):
        registry = registry_or_specs
    else:
        registry = ToolRegistry()
        registry.register_many(registry_or_specs)
    compile_bound_tool_catalog(config, registry)


def collect_runtime_config_issues(
    config: AgentRuntimeConfig, registry_or_specs: ToolRegistry | Iterable[ToolSpec]
) -> list[str]:
    """Validate a runtime config against a tool registry, collecting **all** problems as readable
    messages instead of raising on the first (the accumulating sibling of
    :func:`validate_runtime_config`). Returns an empty list when the config is valid. Mirrors the
    same checks as :func:`compile_bound_tool_catalog`: unknown tool id, duplicate binding_id /
    model_name / call name, and invalid binding runtime."""
    if isinstance(registry_or_specs, ToolRegistry):
        registry = registry_or_specs
    else:
        registry = ToolRegistry()
        registry.register_many(registry_or_specs)
    specs = {tool.id: tool for tool in registry.specs()}
    issues: list[str] = []
    seen_binding_ids: set[str] = set()
    seen_model_names: set[str] = set()
    seen_call_names: dict[str, str] = {}

    def reserve_call_name(name: str, owner: str) -> None:
        previous = seen_call_names.get(name)
        if previous is not None and previous != owner:
            issues.append(f"duplicate tool call name: {name}")
        else:
            seen_call_names[name] = owner

    if config.tool_search.enabled:
        reserve_call_name(config.tool_search.binding_id, "tool_search")
        reserve_call_name(config.tool_search.model_name, "tool_search")

    for binding in config.tools:
        search_binding_clash = (
            config.tool_search.enabled and binding.binding_id == config.tool_search.binding_id
        )
        if binding.binding_id in seen_binding_ids or search_binding_clash:
            issues.append(f"duplicate tool binding_id: {binding.binding_id}")
        else:
            seen_binding_ids.add(binding.binding_id)
            reserve_call_name(binding.binding_id, binding.binding_id)
        spec = specs.get(binding.ref.tool_id)
        if spec is None:
            issues.append(_unknown_tool_message(binding.ref.tool_id, specs.keys()))
            continue
        try:
            model_name = _resolved_model_name(binding, spec)
        except AgentConfigError as exc:
            # e.g. an empty/whitespace model_name — collect it instead of letting validate() throw.
            issues.append(str(exc))
            continue
        search_model_clash = (
            config.tool_search.enabled and model_name == config.tool_search.model_name
        )
        if model_name in seen_model_names or search_model_clash:
            issues.append(f"duplicate tool model_name: {model_name}")
        else:
            seen_model_names.add(model_name)
            reserve_call_name(model_name, binding.binding_id)
        try:
            _validate_binding_runtime(binding)
        except AgentConfigError as exc:
            issues.append(str(exc))
    return issues


def runtime_config_diff(
    previous: AgentRuntimeConfig | None,
    current: AgentRuntimeConfig,
) -> dict[str, Any]:
    current_tools = {binding.binding_id: binding for binding in current.tools}
    if previous is None:
        return {
            "initial": True,
            "added_bindings": sorted(current_tools),
            "removed_bindings": [],
            "changed_bindings": [],
            "prompt_changed": True,
            "model_changed": current.model is not None,
        }
    previous_tools = {binding.binding_id: binding for binding in previous.tools}
    added = sorted(set(current_tools) - set(previous_tools))
    removed = sorted(set(previous_tools) - set(current_tools))
    changed = sorted(
        binding_id
        for binding_id in set(current_tools) & set(previous_tools)
        if current_tools[binding_id].to_json() != previous_tools[binding_id].to_json()
    )
    return {
        "initial": False,
        "added_bindings": added,
        "removed_bindings": removed,
        "changed_bindings": changed,
        "prompt_changed": current.prompt.to_json() != previous.prompt.to_json(),
        "model_changed": (
            (None if current.model is None else current.model.to_json())
            != (None if previous.model is None else previous.model.to_json())
        ),
    }


def transcript_config_snapshot(
    config: AgentRuntimeConfig,
    *,
    step: int,
    turn_id: str,
) -> dict[str, Any]:
    return {
        "kind": "agent_runtime_config_snapshot",
        "step": step,
        "turn_id": turn_id,
        "definition_id": config.definition_id,
        "config_version": config.config_version,
        "config_hash": config.config_hash,
        "binding_ids": [binding.binding_id for binding in config.tools],
        "tool_ids": [binding.ref.tool_id for binding in config.tools],
        "prompt_hash": canonical_sha256(config.prompt.to_json()),
        "model": None if config.model is None else config.model.model,
    }


def _resolved_model_name(binding: ToolBinding, spec: ToolSpec) -> str:
    del spec
    model_name = binding.model_name or binding.binding_id.replace(".", "_")
    model_name = model_name.strip()
    if not model_name:
        raise AgentConfigError(f"tool binding {binding.binding_id} resolves to an empty model name")
    return model_name


def _model_tool_spec(spec: ToolSpec, binding: ToolBinding, model_name: str) -> ToolSpec:
    guidance = ToolGuidance.from_json(spec.guidance)
    if spec.examples:
        guidance = guidance.merged(ToolGuidance(examples=tuple(dict(item) for item in spec.examples)))
    if spec.annotations:
        guidance = guidance.merged(ToolGuidance(annotations=dict(spec.annotations)))
    guidance = guidance.merged(binding.guidance)
    description = spec.description.rstrip()
    text = guidance.short_text()
    if text:
        description = f"{description}\n\n{text}"
    return replace(
        spec,
        id=binding.binding_id,
        provider_name=model_name,
        description=description,
        guidance=guidance.to_json(),
        examples=guidance.examples,
        annotations={**guidance.annotations, "binding_id": binding.binding_id, **binding.metadata},
    )


def _validate_binding_runtime(binding: ToolBinding) -> None:
    if binding.ref.tool_id == "shell.exec":
        shell_runtime = binding.runtime.get("shell", binding.runtime)
        if shell_runtime is not None and not isinstance(shell_runtime, Mapping):
            raise AgentConfigError(f"shell binding runtime must be an object: {binding.binding_id}")
    if binding.ref.tool_id.startswith("web."):
        web_runtime = binding.runtime.get("web", binding.runtime)
        if web_runtime is not None and not isinstance(web_runtime, Mapping):
            raise AgentConfigError(f"web binding runtime must be an object: {binding.binding_id}")


def _risk_for(spec: ToolSpec) -> str:
    if spec.side_effect in {"write", "shell", "run"}:
        return "side_effect"
    if spec.side_effect == "artifact":
        return "artifact"
    return "read"


def _str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError("expected an array of strings")
    return tuple(str(item) for item in value)
