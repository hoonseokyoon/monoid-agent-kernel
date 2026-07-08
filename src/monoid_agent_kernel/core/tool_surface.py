from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, runtime_checkable

from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.tools.base import ToolSpec

ToolExposure = Literal["immediate", "searchable", "hidden"]
ToolAuthorizationDecision = Literal["allow", "ask", "deny"]


@dataclass(frozen=True)
class ToolGuidance:
    summary: str = ""
    policy: str = ""
    examples: tuple[dict[str, Any], ...] = ()
    annotations: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: Any) -> ToolGuidance:
        if payload is None:
            return cls()
        if isinstance(payload, ToolGuidance):
            return payload
        if isinstance(payload, str):
            return cls(summary=payload)
        if not isinstance(payload, Mapping):
            raise ValueError("tool guidance must be an object or string")
        examples = payload.get("examples") or ()
        if not isinstance(examples, list | tuple):
            raise ValueError("tool guidance examples must be an array")
        annotations = payload.get("annotations") or {}
        if not isinstance(annotations, Mapping):
            raise ValueError("tool guidance annotations must be an object")
        return cls(
            summary=str(payload.get("summary") or ""),
            policy=str(payload.get("policy") or ""),
            examples=tuple(dict(item) for item in examples if isinstance(item, Mapping)),
            annotations=dict(annotations),
        )

    def merged(self, override: ToolGuidance) -> ToolGuidance:
        return ToolGuidance(
            summary=override.summary or self.summary,
            policy=override.policy or self.policy,
            examples=override.examples or self.examples,
            annotations={**self.annotations, **override.annotations},
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "policy": self.policy,
            "examples": [dict(item) for item in self.examples],
            "annotations": dict(self.annotations),
        }

    def short_text(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"Guidance: {_shorten(self.summary)}")
        if self.policy:
            parts.append(f"Policy: {_shorten(self.policy)}")
        return " ".join(parts)


@dataclass(frozen=True)
class ToolQuota:
    max_calls_per_run: int | None = None

    @classmethod
    def from_json(cls, payload: Any) -> ToolQuota:
        if payload is None:
            return cls()
        if isinstance(payload, ToolQuota):
            return payload
        if not isinstance(payload, Mapping):
            raise ValueError("tool quota must be an object")
        raw = payload.get("max_calls_per_run", payload.get("max_calls"))
        if raw is None:
            return cls()
        value = int(raw)
        if value < 0:
            raise ValueError("tool quota max_calls_per_run must be non-negative")
        return cls(max_calls_per_run=value)

    def to_json(self) -> dict[str, Any]:
        return {"max_calls_per_run": self.max_calls_per_run}


@dataclass(frozen=True)
class ToolScope:
    allowed_paths: tuple[str, ...] = ()
    denied_paths: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    command_allow_prefixes: tuple[str, ...] = ()
    command_deny_prefixes: tuple[str, ...] = ()
    env_allowlist: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, payload: Any) -> ToolScope:
        if payload is None:
            return cls()
        if isinstance(payload, ToolScope):
            return payload
        if not isinstance(payload, Mapping):
            raise ValueError("tool scope must be an object")
        return cls(
            allowed_paths=_str_tuple(payload.get("allowed_paths")),
            denied_paths=_str_tuple(payload.get("denied_paths")),
            allowed_domains=_str_tuple(payload.get("allowed_domains")),
            blocked_domains=_str_tuple(payload.get("blocked_domains")),
            command_allow_prefixes=_str_tuple(payload.get("command_allow_prefixes")),
            command_deny_prefixes=_str_tuple(payload.get("command_deny_prefixes")),
            env_allowlist=_str_tuple(payload.get("env_allowlist")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "allowed_paths": list(self.allowed_paths),
            "denied_paths": list(self.denied_paths),
            "allowed_domains": list(self.allowed_domains),
            "blocked_domains": list(self.blocked_domains),
            "command_allow_prefixes": list(self.command_allow_prefixes),
            "command_deny_prefixes": list(self.command_deny_prefixes),
            "env_allowlist": list(self.env_allowlist),
        }


@dataclass(frozen=True)
class ToolAuthorization:
    tool_id: str
    binding_id: str
    model_name: str
    decision: ToolAuthorizationDecision
    reason: str
    exposure: ToolExposure
    quota: ToolQuota = field(default_factory=ToolQuota)
    scope: ToolScope = field(default_factory=ToolScope)
    surface_scope: ToolScope = field(default_factory=ToolScope)
    runtime: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "binding_id": self.binding_id,
            "model_name": self.model_name,
            "decision": self.decision,
            "reason": self.reason,
            "exposure": self.exposure,
            "quota": self.quota.to_json(),
            "scope": self.scope.to_json(),
            "surface_scope": self.surface_scope.to_json(),
            "runtime": dict(self.runtime),
        }


@dataclass(frozen=True)
class ToolSearchEntry:
    binding_id: str
    tool_id: str
    exported_name: str
    title: str
    summary: str
    risk: str
    requires_approval: bool
    namespace: str = ""
    groups: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    load_hint: str = "available_next_turn"
    guidance: ToolGuidance = field(default_factory=ToolGuidance)
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "binding_id": self.binding_id,
            "tool_id": self.tool_id,
            "exported_name": self.exported_name,
            "title": self.title,
            "summary": self.summary,
            "risk": self.risk,
            "requires_approval": self.requires_approval,
            "namespace": self.namespace,
            "groups": list(self.groups),
            "tags": list(self.tags),
            "load_hint": self.load_hint,
            "guidance": self.guidance.to_json(),
            "annotations": dict(self.annotations),
        }


@dataclass(frozen=True)
class ToolSurfaceSnapshot:
    turn_id: str
    immediate_tools: tuple[ToolSpec, ...]
    searchable_tools: tuple[ToolSpec, ...]
    search_entries: tuple[ToolSearchEntry, ...]
    hidden_tool_ids: tuple[str, ...]
    authorizations: dict[str, ToolAuthorization]
    delta_notice: str = ""
    surface_hash: str = ""
    surface_warnings: tuple[str, ...] = ()

    def authorization_for(self, binding_id: str) -> ToolAuthorization | None:
        return self.authorizations.get(binding_id)

    def to_public_json(self) -> dict[str, Any]:
        def _tool_payload(tool: ToolSpec) -> dict[str, Any]:
            auth = self.authorizations.get(tool.id)
            return {
                "binding_id": tool.id,
                "tool_id": auth.tool_id if auth is not None else tool.id,
                "exported_name": tool.exported_name,
                "capability": tool.capability,
                "side_effect": tool.side_effect,
                "authorization": auth.decision if auth is not None else None,
                "exposure": auth.exposure if auth is not None else None,
                "summary": str(tool.guidance.get("summary") or tool.description.split("\n", 1)[0]),
                "risk": str(tool.guidance.get("risk") or _risk_for(tool)),
                "tags": list(tool.guidance.get("tags") or ()),
                "annotations": dict(tool.annotations),
            }

        return {
            "surface_hash": self.surface_hash,
            "immediate_binding_ids": [tool.id for tool in self.immediate_tools],
            "immediate_tools": [_tool_payload(tool) for tool in self.immediate_tools],
            "searchable_count": len(self.searchable_tools),
            "searchable_tools": [entry.to_json() for entry in self.search_entries],
            "hidden_count": len(self.hidden_tool_ids),
            "hidden_binding_ids": list(self.hidden_tool_ids),
            "authorizations": {
                binding_id: authorization.to_json()
                for binding_id, authorization in sorted(self.authorizations.items())
            },
            "delta_notice": self.delta_notice,
            "surface_warnings": list(self.surface_warnings),
        }

    def to_transcript_json(self) -> dict[str, Any]:
        return {
            "kind": "tool_surface_snapshot",
            "turn_id": self.turn_id,
            "surface_hash": self.surface_hash,
            "immediate_tools": [_tool_spec_payload(tool) for tool in self.immediate_tools],
            "searchable_tools": [_tool_spec_payload(tool) for tool in self.searchable_tools],
            "search_entries": [entry.to_json() for entry in self.search_entries],
            "hidden_tool_ids": list(self.hidden_tool_ids),
            "authorizations": {
                binding_id: authorization.to_json()
                for binding_id, authorization in sorted(self.authorizations.items())
            },
            "delta_notice": self.delta_notice,
            "surface_warnings": list(self.surface_warnings),
        }


def visible_registry_tool_ids(
    snapshot: ToolSurfaceSnapshot,
    bound_catalog: Any,
) -> frozenset[str]:
    """Return registry tool ids for bindings visible in the model-facing surface."""
    visible_binding_ids = {
        tool.id for tool in (*snapshot.immediate_tools, *snapshot.searchable_tools)
    }
    return _registry_tool_ids_for_bindings(visible_binding_ids, bound_catalog)


def immediate_registry_tool_ids(
    snapshot: ToolSurfaceSnapshot,
    bound_catalog: Any,
) -> frozenset[str]:
    """Return registry tool ids for bindings immediately callable this turn."""
    immediate_binding_ids = {tool.id for tool in snapshot.immediate_tools}
    return _registry_tool_ids_for_bindings(immediate_binding_ids, bound_catalog)


def allowed_immediate_registry_tool_ids(
    snapshot: ToolSurfaceSnapshot,
    bound_catalog: Any,
) -> frozenset[str]:
    """Return immediate registry tool ids whose effective authorization is allow."""
    allowed_binding_ids = {
        tool.id
        for tool in snapshot.immediate_tools
        if (auth := snapshot.authorization_for(tool.id)) is not None and auth.decision == "allow"
    }
    return _registry_tool_ids_for_bindings(allowed_binding_ids, bound_catalog)


def _registry_tool_ids_for_bindings(binding_ids: set[str], bound_catalog: Any) -> frozenset[str]:
    by_binding_id = getattr(bound_catalog, "by_binding_id", {})
    return frozenset(
        bound.base_spec.id
        for binding_id in binding_ids
        if (bound := by_binding_id.get(binding_id)) is not None
    )


@runtime_checkable
class ToolSurfaceResolver(Protocol):
    name: str

    def resolve(
        self,
        *,
        bound_catalog: Any,
        turn: Any,
        pending_binding_loads: tuple[str, ...] = (),
        previous_snapshot: ToolSurfaceSnapshot | None = None,
        call_counts: Mapping[str, int] | None = None,
    ) -> ToolSurfaceSnapshot:
        ...


@dataclass(frozen=True)
class DefaultToolSurfaceResolver:
    name: str = "default"

    def resolve(
        self,
        *,
        bound_catalog: Any,
        turn: Any,
        pending_binding_loads: tuple[str, ...] = (),
        previous_snapshot: ToolSurfaceSnapshot | None = None,
        call_counts: Mapping[str, int] | None = None,
    ) -> ToolSurfaceSnapshot:
        counts = call_counts or {}
        pending = set(pending_binding_loads)
        available_binding_ids = {tool.binding_id for tool in bound_catalog.tools}
        hidden: list[str] = []
        immediate: list[ToolSpec] = []
        searchable: list[ToolSpec] = []
        search_entries: list[ToolSearchEntry] = []
        authorizations: dict[str, ToolAuthorization] = {}
        refused_loads: list[str] = sorted(pending - available_binding_ids)
        surface_warnings: list[str] = []

        for bound in bound_catalog.tools:
            auth = bound.authorization
            exposure = bound.exposure
            max_calls = auth.quota.max_calls_per_run
            if max_calls is not None and counts.get(bound.binding_id, 0) >= max_calls:
                hidden.append(bound.binding_id)
                authorizations[bound.binding_id] = replace(
                    auth,
                    decision="deny",
                    exposure="hidden",
                    reason="quota_exhausted",
                )
                surface_warnings.append(f"{bound.binding_id} hidden because its quota is exhausted")
                if bound.binding_id in pending:
                    refused_loads.append(bound.binding_id)
                continue
            if auth.decision == "deny" or exposure == "hidden":
                hidden.append(bound.binding_id)
                authorizations[bound.binding_id] = replace(auth, exposure="hidden")
                if bound.binding_id in pending:
                    refused_loads.append(bound.binding_id)
                continue
            if bound.binding_id in pending and exposure == "searchable":
                exposure = "immediate"
                auth = replace(auth, exposure=exposure, reason="loaded_from_tool_search")
            authorizations[bound.binding_id] = auth if auth.exposure == exposure else replace(auth, exposure=exposure)
            if exposure == "searchable":
                searchable.append(bound.model_spec)
                search_entries.append(_search_entry(bound, auth))
                continue
            immediate.append(bound.model_spec)

        if bound_catalog.tool_search.enabled and searchable and bound_catalog.search_tool is not None:
            search_spec = _search_tool_spec(bound_catalog.search_tool, bound_catalog.tool_search)
            search_auth = ToolAuthorization(
                tool_id="tool.search",
                binding_id=bound_catalog.tool_search.binding_id,
                model_name=bound_catalog.tool_search.model_name,
                decision="allow",
                reason="enabled_by_tool_search_config",
                exposure="immediate",
            )
            authorizations[bound_catalog.tool_search.binding_id] = search_auth
            immediate.append(search_spec)

        delta_notice = _delta_notice(
            previous_snapshot,
            immediate_binding_ids=tuple(tool.id for tool in immediate),
            searchable_count=len(searchable),
            hidden_count=len(hidden),
            refused_loads=tuple(refused_loads),
        )
        payload = {
            "turn_id": getattr(turn, "turn_id", ""),
            "immediate": [tool.id for tool in immediate],
            "searchable": [tool.id for tool in searchable],
            "hidden": sorted(hidden),
            "search_entries": [entry.to_json() for entry in search_entries],
            "authorizations": {
                binding_id: authorization.to_json()
                for binding_id, authorization in sorted(authorizations.items())
            },
        }
        return ToolSurfaceSnapshot(
            turn_id=str(getattr(turn, "turn_id", "")),
            immediate_tools=tuple(immediate),
            searchable_tools=tuple(searchable),
            search_entries=tuple(search_entries),
            hidden_tool_ids=tuple(sorted(hidden)),
            authorizations=authorizations,
            delta_notice=delta_notice,
            surface_hash=canonical_sha256(payload),
            surface_warnings=tuple(surface_warnings),
        )


def tool_surface_manifest(
    *,
    resolver: ToolSurfaceResolver,
    tool_search: Any,
    dynamic_enabled: bool,
    initial_catalog_count: int,
) -> dict[str, Any]:
    return {
        "resolver": resolver.name,
        "dynamic_enabled": dynamic_enabled,
        "initial_catalog_count": initial_catalog_count,
        "tool_search": tool_search.to_json() if hasattr(tool_search, "to_json") else {},
    }


def _search_entry(bound: Any, auth: ToolAuthorization) -> ToolSearchEntry:
    binding = bound.binding
    spec = bound.model_spec
    title = binding.title or binding.binding_id
    summary = binding.summary or spec.description.split("\n", 1)[0]
    risk = binding.risk or _risk_for(bound.base_spec)
    requires_approval = (
        bool(binding.requires_approval)
        if binding.requires_approval is not None
        else auth.decision == "ask"
    )
    search_metadata = _tool_search_metadata(binding.metadata)
    namespace = str(
        search_metadata.get("namespace")
        or _namespace_for(binding.binding_id)
        or _namespace_for(bound.base_spec.id)
        or "tools"
    )
    groups = _string_tuple(search_metadata.get("groups", search_metadata.get("group")))
    if not groups:
        groups = (_namespace_for(bound.base_spec.capability) or namespace,)
    tags = _dedupe_strings(
        (
            *_string_tuple(search_metadata.get("tags", search_metadata.get("tag"))),
            risk,
            bound.base_spec.side_effect,
            bound.base_spec.capability,
        )
    )
    return ToolSearchEntry(
        binding_id=binding.binding_id,
        tool_id=bound.base_spec.id,
        exported_name=bound.model_name,
        title=title,
        summary=summary,
        risk=risk,
        requires_approval=requires_approval,
        namespace=namespace,
        groups=groups,
        tags=tags,
        guidance=binding.guidance,
        annotations=dict(spec.annotations),
    )


def _search_tool_spec(search_tool: ToolSpec, config: Any) -> ToolSpec:
    return replace(
        search_tool,
        id=config.binding_id,
        provider_name=config.model_name,
        annotations={**search_tool.annotations, "binding_id": config.binding_id},
    )


def _risk_for(spec: ToolSpec) -> str:
    if spec.side_effect in {"write", "shell", "run"}:
        return "side_effect"
    if spec.side_effect == "artifact":
        return "artifact"
    return "read"


def _delta_notice(
    previous: ToolSurfaceSnapshot | None,
    *,
    immediate_binding_ids: tuple[str, ...],
    searchable_count: int,
    hidden_count: int,
    refused_loads: tuple[str, ...],
) -> str:
    notices: list[str] = []
    if previous is not None:
        old = {tool.id for tool in previous.immediate_tools}
        new = set(immediate_binding_ids)
        added = sorted(new - old)
        removed = sorted(old - new)
        if added or removed or searchable_count != len(previous.searchable_tools):
            parts: list[str] = []
            if added:
                parts.append(f"{len(added)} immediate binding added")
            if removed:
                parts.append(f"{len(removed)} immediate binding removed")
            if searchable_count != len(previous.searchable_tools):
                parts.append(f"{searchable_count} searchable bindings available")
            notices.append("Tool surface changed: " + "; ".join(parts) + ".")
    if refused_loads:
        notices.append("Some requested binding loads are unavailable under the current config.")
    return " ".join(notices)


def _tool_spec_payload(tool: ToolSpec) -> dict[str, Any]:
    return {
        "id": tool.id,
        "exported_name": tool.exported_name,
        "description": tool.description,
        "input_schema": tool.input_schema,
        "capability": tool.capability,
        "side_effect": tool.side_effect,
        "path_args": list(tool.path_args),
        "guidance": dict(tool.guidance),
        "examples": [dict(item) for item in tool.examples],
        "annotations": dict(tool.annotations),
    }


def _str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError("expected an array of strings")
    return tuple(str(item) for item in value)


def _tool_search_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = metadata.get("tool_search")
    return payload if isinstance(payload, Mapping) else {}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, list | tuple):
        return tuple(text for item in value if (text := str(item).strip()))
    return ()


def _dedupe_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)


def _namespace_for(value: str) -> str:
    text = value.strip()
    if "." not in text:
        return ""
    return text.split(".", 1)[0].strip()


def _shorten(text: str, limit: int = 320) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "..."
