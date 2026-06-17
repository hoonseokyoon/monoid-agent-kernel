from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from fnmatch import fnmatchcase
from typing import Any, Literal, Protocol, runtime_checkable

from native_agent_runner.core._util import canonical_sha256
from native_agent_runner.tools.base import ToolRegistry, ToolSpec
from native_agent_runner.tools.policy import NormalizedToolPolicy

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

    def merged(self, override: ToolScope) -> ToolScope:
        return ToolScope(
            allowed_paths=override.allowed_paths or self.allowed_paths,
            denied_paths=(*self.denied_paths, *override.denied_paths),
            allowed_domains=override.allowed_domains or self.allowed_domains,
            blocked_domains=(*self.blocked_domains, *override.blocked_domains),
            command_allow_prefixes=override.command_allow_prefixes or self.command_allow_prefixes,
            command_deny_prefixes=(*self.command_deny_prefixes, *override.command_deny_prefixes),
            env_allowlist=override.env_allowlist or self.env_allowlist,
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
class ToolExposureRule:
    tool: str
    exposure: ToolExposure = "immediate"
    authorization: ToolAuthorizationDecision | None = None
    guidance: ToolGuidance = field(default_factory=ToolGuidance)
    quota: ToolQuota = field(default_factory=ToolQuota)
    scope: ToolScope = field(default_factory=ToolScope)
    title: str = ""
    summary: str = ""
    risk: str = ""
    requires_approval: bool | None = None
    reason: str = ""

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ToolExposureRule:
        if not isinstance(payload, dict):
            raise ValueError("tool surface rule must be an object")
        tool = str(payload.get("tool") or payload.get("id") or payload.get("name") or "").strip()
        if not tool:
            raise ValueError("tool surface rule requires tool")
        exposure = str(payload.get("exposure") or "immediate")
        if exposure not in {"immediate", "searchable", "hidden"}:
            raise ValueError("tool surface rule exposure must be immediate, searchable, or hidden")
        authorization = payload.get("authorization")
        if authorization is not None:
            authorization = str(authorization)
            if authorization not in {"allow", "ask", "deny"}:
                raise ValueError("tool surface rule authorization must be allow, ask, or deny")
        return cls(
            tool=tool,
            exposure=exposure,  # type: ignore[arg-type]
            authorization=authorization,  # type: ignore[arg-type]
            guidance=ToolGuidance.from_json(payload.get("guidance")),
            quota=ToolQuota.from_json(payload.get("quota")),
            scope=ToolScope.from_json(payload.get("scope")),
            title=str(payload.get("title") or ""),
            summary=str(payload.get("summary") or ""),
            risk=str(payload.get("risk") or ""),
            requires_approval=(
                None if "requires_approval" not in payload else bool(payload["requires_approval"])
            ),
            reason=str(payload.get("reason") or ""),
        )

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tool": self.tool,
            "exposure": self.exposure,
            "guidance": self.guidance.to_json(),
            "quota": self.quota.to_json(),
            "scope": self.scope.to_json(),
        }
        if self.authorization is not None:
            payload["authorization"] = self.authorization
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
class ToolSurfacePolicy:
    version: str = "tool-surface.v1"
    rules: tuple[ToolExposureRule, ...] = ()
    default_exposure: ToolExposure = "immediate"
    tool_search_enabled: bool = True
    search_top_k: int = 5

    @classmethod
    def from_json(cls, payload: dict[str, Any] | None) -> ToolSurfacePolicy:
        if payload is None:
            return cls()
        if isinstance(payload, ToolSurfacePolicy):
            return payload
        if not isinstance(payload, dict):
            raise ValueError("tool_surface_policy must be an object")
        default_exposure = str(payload.get("default_exposure") or "immediate")
        if default_exposure not in {"immediate", "searchable", "hidden"}:
            raise ValueError("tool_surface_policy.default_exposure is invalid")
        rules = payload.get("rules", payload.get("exposure_rules") or ())
        if not isinstance(rules, list | tuple):
            raise ValueError("tool_surface_policy.rules must be an array")
        search_top_k = int(payload.get("search_top_k", 5))
        if search_top_k < 1:
            raise ValueError("tool_surface_policy.search_top_k must be positive")
        return cls(
            version=str(payload.get("version") or "tool-surface.v1"),
            rules=tuple(ToolExposureRule.from_json(dict(item)) for item in rules),
            default_exposure=default_exposure,  # type: ignore[arg-type]
            tool_search_enabled=bool(payload.get("tool_search_enabled", True)),
            search_top_k=search_top_k,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "rules": [rule.to_json() for rule in self.rules],
            "default_exposure": self.default_exposure,
            "tool_search_enabled": self.tool_search_enabled,
            "search_top_k": self.search_top_k,
        }


@dataclass(frozen=True)
class ToolAuthorization:
    tool_id: str
    decision: ToolAuthorizationDecision
    reason: str
    exposure: ToolExposure
    quota: ToolQuota = field(default_factory=ToolQuota)
    scope: ToolScope = field(default_factory=ToolScope)
    surface_scope: ToolScope = field(default_factory=ToolScope)

    def to_json(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "decision": self.decision,
            "reason": self.reason,
            "exposure": self.exposure,
            "quota": self.quota.to_json(),
            "scope": self.scope.to_json(),
            "surface_scope": self.surface_scope.to_json(),
        }


@dataclass(frozen=True)
class ToolSearchEntry:
    tool_id: str
    exported_name: str
    title: str
    summary: str
    risk: str
    requires_approval: bool
    load_hint: str = "available_next_turn"
    guidance: ToolGuidance = field(default_factory=ToolGuidance)
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "exported_name": self.exported_name,
            "title": self.title,
            "summary": self.summary,
            "risk": self.risk,
            "requires_approval": self.requires_approval,
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
    policy_warnings: tuple[str, ...] = ()

    def authorization_for(self, tool_id: str) -> ToolAuthorization | None:
        return self.authorizations.get(tool_id)

    def to_public_json(self) -> dict[str, Any]:
        return {
            "surface_hash": self.surface_hash,
            "immediate_tool_ids": [tool.id for tool in self.immediate_tools],
            "searchable_count": len(self.searchable_tools),
            "hidden_count": len(self.hidden_tool_ids),
            "delta_notice": self.delta_notice,
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
                tool_id: authorization.to_json()
                for tool_id, authorization in sorted(self.authorizations.items())
            },
            "delta_notice": self.delta_notice,
            "policy_warnings": list(self.policy_warnings),
        }


@runtime_checkable
class ToolSurfaceResolver(Protocol):
    name: str

    def resolve(
        self,
        *,
        registry: ToolRegistry,
        run_spec: Any,
        turn: Any,
        legacy_tool_policy: NormalizedToolPolicy,
        capabilities: frozenset[str],
        pending_tool_loads: tuple[str, ...] = (),
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
        registry: ToolRegistry,
        run_spec: Any,
        turn: Any,
        legacy_tool_policy: NormalizedToolPolicy,
        capabilities: frozenset[str],
        pending_tool_loads: tuple[str, ...] = (),
        previous_snapshot: ToolSurfaceSnapshot | None = None,
        call_counts: Mapping[str, int] | None = None,
    ) -> ToolSurfaceSnapshot:
        del call_counts
        policy = ToolSurfacePolicy.from_json(getattr(run_spec, "tool_surface_policy", None))
        pending = set(pending_tool_loads)
        hidden: list[str] = []
        immediate: list[ToolSpec] = []
        searchable: list[ToolSpec] = []
        search_entries: list[ToolSearchEntry] = []
        authorizations: dict[str, ToolAuthorization] = {}
        refused_loads: list[str] = []
        tool_search_spec: ToolSpec | None = None

        for original in registry.specs():
            if original.id == "tool.search":
                tool_search_spec = original
                continue
            rule = _matching_rule(policy.rules, original)
            auth, exposure = _resolve_authorization(
                original,
                rule,
                policy,
                legacy_tool_policy,
                capabilities,
                run_spec,
            )
            if original.id in pending and exposure == "searchable" and auth.decision != "deny":
                exposure = "immediate"
                auth = replace(auth, exposure=exposure, reason="loaded_from_tool_search")
            elif original.id in pending and (exposure == "hidden" or auth.decision == "deny"):
                refused_loads.append(original.id)
            authorizations[original.id] = auth if auth.exposure == exposure else replace(auth, exposure=exposure)
            if exposure == "hidden":
                hidden.append(original.id)
                continue
            guidance = _effective_guidance(original, rule)
            enriched = _with_guidance(original, guidance)
            if exposure == "searchable":
                searchable.append(enriched)
                search_entries.append(_search_entry(enriched, rule, guidance, auth))
                continue
            immediate.append(enriched)

        if tool_search_spec is not None:
            search_auth, search_exposure = _resolve_tool_search(
                tool_search_spec,
                policy,
                legacy_tool_policy,
                capabilities,
                has_searchable=bool(searchable),
            )
            authorizations[tool_search_spec.id] = search_auth
            if search_exposure == "immediate":
                immediate.append(tool_search_spec)
            else:
                hidden.append(tool_search_spec.id)

        delta_notice = _delta_notice(
            previous_snapshot,
            immediate_tool_ids=tuple(tool.id for tool in immediate),
            searchable_count=len(searchable),
            hidden_count=len(hidden),
            refused_loads=tuple(refused_loads),
        )
        payload = {
            "turn_id": getattr(turn, "turn_id", ""),
            "immediate": [tool.id for tool in immediate],
            "searchable": [tool.id for tool in searchable],
            "hidden": sorted(hidden),
            "authorizations": {
                tool_id: authorization.to_json()
                for tool_id, authorization in sorted(authorizations.items())
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
            policy_warnings=legacy_tool_policy.policy_warnings,
        )


def tool_surface_manifest(
    *,
    resolver: ToolSurfaceResolver,
    policy: ToolSurfacePolicy,
    dynamic_enabled: bool,
    initial_catalog_count: int,
) -> dict[str, Any]:
    return {
        "resolver": resolver.name,
        "policy_version": policy.version,
        "dynamic_enabled": dynamic_enabled,
        "initial_catalog_count": initial_catalog_count,
        "default_exposure": policy.default_exposure,
        "tool_search_enabled": policy.tool_search_enabled,
        "search_top_k": policy.search_top_k,
    }


def _resolve_authorization(
    spec: ToolSpec,
    rule: ToolExposureRule | None,
    policy: ToolSurfacePolicy,
    legacy_tool_policy: NormalizedToolPolicy,
    capabilities: frozenset[str],
    run_spec: Any,
) -> tuple[ToolAuthorization, ToolExposure]:
    legacy_decision = legacy_tool_policy.decision_for(spec.id)
    exposure = rule.exposure if rule is not None else policy.default_exposure
    reason = legacy_decision.reason
    decision: ToolAuthorizationDecision = legacy_decision.decision
    if spec.capability not in capabilities:
        decision = "deny"
        exposure = "hidden"
        reason = "missing_capability"
    elif spec.id in legacy_tool_policy.hidden_tools:
        decision = "deny"
        exposure = "hidden"
        reason = legacy_tool_policy.hidden_tools[spec.id]
    elif rule is not None and rule.authorization is not None:
        decision = rule.authorization
        reason = rule.reason or f"{decision}_by_tool_surface_policy"
    if exposure == "hidden":
        decision = "deny"
        reason = reason if reason != "allowed_by_tool_policy" else "hidden_by_tool_surface_policy"
    if decision == "deny":
        exposure = "hidden"
    quota = rule.quota if rule is not None else ToolQuota()
    scope = _base_scope_for(spec, run_spec)
    surface_scope = rule.scope if rule is not None else ToolScope()
    scope = scope.merged(surface_scope)
    return (
        ToolAuthorization(
            tool_id=spec.id,
            decision=decision,
            reason=reason,
            exposure=exposure,
            quota=quota,
            scope=scope,
            surface_scope=surface_scope,
        ),
        exposure,
    )


def _base_scope_for(spec: ToolSpec, run_spec: Any) -> ToolScope:
    if run_spec is None:
        return ToolScope()
    permission_policy = getattr(run_spec, "permission_policy", None)
    web_policy = getattr(run_spec, "web_policy", None)
    shell_policy = getattr(run_spec, "shell_policy", None)
    command_allow: list[str] = []
    command_deny: list[str] = []
    for rule in getattr(shell_policy, "command_rules", ()) or ():
        action = getattr(rule, "action", "")
        prefix = str(getattr(rule, "prefix", ""))
        if not prefix:
            continue
        if action == "allow":
            command_allow.append(prefix)
        elif action == "deny":
            command_deny.append(prefix)
    return ToolScope(
        denied_paths=tuple(getattr(permission_policy, "deny_patterns", ()) or ()),
        allowed_domains=tuple(getattr(web_policy, "allowed_domains", ()) or ()),
        blocked_domains=tuple(getattr(web_policy, "blocked_domains", ()) or ()),
        command_allow_prefixes=tuple(command_allow) if spec.preview_kind == "shell" else (),
        command_deny_prefixes=tuple(command_deny) if spec.preview_kind == "shell" else (),
        env_allowlist=(
            tuple(getattr(shell_policy, "env_allowlist", ()) or ())
            if spec.preview_kind == "shell"
            else ()
        ),
    )


def _resolve_tool_search(
    spec: ToolSpec,
    policy: ToolSurfacePolicy,
    legacy_tool_policy: NormalizedToolPolicy,
    capabilities: frozenset[str],
    *,
    has_searchable: bool,
) -> tuple[ToolAuthorization, ToolExposure]:
    rule = _matching_rule(policy.rules, spec)
    auth, exposure = _resolve_authorization(spec, rule, policy, legacy_tool_policy, capabilities, None)
    if not policy.tool_search_enabled or not has_searchable:
        return replace(auth, decision="deny", reason="no_searchable_tools", exposure="hidden"), "hidden"
    if spec.capability not in capabilities:
        return replace(auth, decision="deny", reason="missing_capability", exposure="hidden"), "hidden"
    if auth.decision == "deny" or exposure == "hidden":
        return replace(auth, exposure="hidden"), "hidden"
    return replace(auth, exposure="immediate"), "immediate"


def _matching_rule(rules: Iterable[ToolExposureRule], spec: ToolSpec) -> ToolExposureRule | None:
    matched: ToolExposureRule | None = None
    for rule in rules:
        if _matches(rule.tool, spec):
            matched = rule
    return matched


def _matches(reference: str, spec: ToolSpec) -> bool:
    if any(char in reference for char in "*?["):
        return fnmatchcase(spec.id, reference) or fnmatchcase(spec.exported_name, reference)
    return reference == spec.id or reference == spec.exported_name


def _effective_guidance(spec: ToolSpec, rule: ToolExposureRule | None) -> ToolGuidance:
    guidance = ToolGuidance.from_json(spec.guidance)
    if spec.examples:
        guidance = guidance.merged(ToolGuidance(examples=tuple(dict(item) for item in spec.examples)))
    if spec.annotations:
        guidance = guidance.merged(ToolGuidance(annotations=dict(spec.annotations)))
    if rule is not None:
        guidance = guidance.merged(rule.guidance)
    return guidance


def _with_guidance(spec: ToolSpec, guidance: ToolGuidance) -> ToolSpec:
    text = guidance.short_text()
    if not text:
        return spec
    description = f"{spec.description.rstrip()}\n\n{text}"
    return replace(
        spec,
        description=description,
        guidance=guidance.to_json(),
        examples=guidance.examples,
        annotations=guidance.annotations,
    )


def _search_entry(
    spec: ToolSpec,
    rule: ToolExposureRule | None,
    guidance: ToolGuidance,
    auth: ToolAuthorization,
) -> ToolSearchEntry:
    title = rule.title if rule is not None and rule.title else spec.id
    summary = rule.summary if rule is not None and rule.summary else spec.description.split("\n", 1)[0]
    risk = rule.risk if rule is not None and rule.risk else _risk_for(spec)
    requires_approval = (
        bool(rule.requires_approval)
        if rule is not None and rule.requires_approval is not None
        else auth.decision == "ask"
    )
    return ToolSearchEntry(
        tool_id=spec.id,
        exported_name=spec.exported_name,
        title=title,
        summary=summary,
        risk=risk,
        requires_approval=requires_approval,
        guidance=guidance,
        annotations=dict(spec.annotations),
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
    immediate_tool_ids: tuple[str, ...],
    searchable_count: int,
    hidden_count: int,
    refused_loads: tuple[str, ...],
) -> str:
    notices: list[str] = []
    if previous is not None:
        old = {tool.id for tool in previous.immediate_tools}
        new = set(immediate_tool_ids)
        added = sorted(new - old)
        removed = sorted(old - new)
        if added or removed or searchable_count != len(previous.searchable_tools):
            parts: list[str] = []
            if added:
                parts.append(f"{len(added)} immediate tool added")
            if removed:
                parts.append(f"{len(removed)} immediate tool removed")
            if searchable_count != len(previous.searchable_tools):
                parts.append(f"{searchable_count} searchable tools available")
            notices.append("Tool surface changed: " + "; ".join(parts) + ".")
    if refused_loads:
        notices.append("Some requested tool loads are unavailable under the current policy.")
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


def _shorten(text: str, limit: int = 320) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "..."
