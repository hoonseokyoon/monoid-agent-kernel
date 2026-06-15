from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Literal, Protocol

from native_agent_runner.errors import ToolPolicyError

ToolPolicyDecisionKind = Literal["allow", "ask", "deny"]


class PolicyToolSpec(Protocol):
    id: str
    capability: str

    @property
    def exported_name(self) -> str:
        ...


@dataclass(frozen=True)
class ToolPolicy:
    allowed_tools: tuple[str, ...] = ()
    denied_tools: tuple[str, ...] = ()
    ask_tools: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, payload: dict[str, Any] | None) -> ToolPolicy:
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ToolPolicyError("tool_policy must be an object")
        return cls(
            allowed_tools=_tuple_from_payload(payload, "allowed_tools", "allow_tools", "allow"),
            denied_tools=_tuple_from_payload(payload, "denied_tools", "deny_tools", "deny"),
            ask_tools=_tuple_from_payload(payload, "ask_tools", "ask"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "allowed_tools": list(self.allowed_tools),
            "denied_tools": list(self.denied_tools),
            "ask_tools": list(self.ask_tools),
        }

    def merged(self, *, allow: Iterable[str] = (), deny: Iterable[str] = (), ask: Iterable[str] = ()) -> ToolPolicy:
        return ToolPolicy(
            allowed_tools=_dedupe((*self.allowed_tools, *allow)),
            denied_tools=_dedupe((*self.denied_tools, *deny)),
            ask_tools=_dedupe((*self.ask_tools, *ask)),
        )


@dataclass(frozen=True)
class ToolPolicyDecision:
    decision: ToolPolicyDecisionKind
    reason: str
    matched_rule: str | None = None


@dataclass(frozen=True)
class NormalizedToolPolicy:
    allowed_tools: tuple[str, ...] = ()
    denied_tools: tuple[str, ...] = ()
    ask_tools: tuple[str, ...] = ()
    visible_tools: tuple[str, ...] = ()
    hidden_tools: dict[str, str] = field(default_factory=dict)
    policy_warnings: tuple[str, ...] = ()
    _allowed_set: frozenset[str] = field(default_factory=frozenset, repr=False)
    _denied_set: frozenset[str] = field(default_factory=frozenset, repr=False)
    _ask_set: frozenset[str] = field(default_factory=frozenset, repr=False)

    def decision_for(self, tool_id: str) -> ToolPolicyDecision:
        if tool_id in self._denied_set:
            return ToolPolicyDecision("deny", "denied_by_tool_policy", tool_id)
        if tool_id in self._ask_set:
            return ToolPolicyDecision("ask", "tool_approval_required", tool_id)
        if self._allowed_set and tool_id not in self._allowed_set:
            return ToolPolicyDecision("deny", "not_in_tool_allowlist", None)
        return ToolPolicyDecision("allow", "allowed_by_tool_policy", tool_id if self._allowed_set else None)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "allowed_tools": list(self.allowed_tools),
            "denied_tools": list(self.denied_tools),
            "ask_tools": list(self.ask_tools),
            "visible_tools": list(self.visible_tools),
            "hidden_tools": [
                {"tool": tool_id, "reason": reason}
                for tool_id, reason in sorted(self.hidden_tools.items())
            ],
            "policy_warnings": list(self.policy_warnings),
        }


def normalize_tool_policy(
    policy: ToolPolicy,
    specs: Iterable[PolicyToolSpec],
    capabilities: frozenset[str],
) -> NormalizedToolPolicy:
    ordered_specs = list(specs)
    by_id = {spec.id: spec for spec in ordered_specs}
    allowed = _resolve_references("allowed_tools", policy.allowed_tools, ordered_specs)
    denied = _resolve_references("denied_tools", policy.denied_tools, ordered_specs)
    ask = _resolve_references("ask_tools", policy.ask_tools, ordered_specs)
    allowed_set = frozenset(allowed)
    denied_set = frozenset(denied)
    ask_set = frozenset(ask)

    visible: list[str] = []
    hidden: dict[str, str] = {}
    for spec in ordered_specs:
        if spec.capability not in capabilities:
            hidden[spec.id] = "missing_capability"
            continue
        decision = _decision_for(spec.id, allowed_set, denied_set, ask_set)
        if decision.decision == "deny":
            hidden[spec.id] = decision.reason
            continue
        visible.append(spec.id)

    canonical_allowed = tuple(tool_id for tool_id in by_id if tool_id in allowed_set)
    canonical_denied = tuple(tool_id for tool_id in by_id if tool_id in denied_set)
    canonical_ask = tuple(tool_id for tool_id in by_id if tool_id in ask_set)
    return NormalizedToolPolicy(
        allowed_tools=canonical_allowed,
        denied_tools=canonical_denied,
        ask_tools=canonical_ask,
        visible_tools=tuple(visible),
        hidden_tools=hidden,
        policy_warnings=(),
        _allowed_set=allowed_set,
        _denied_set=denied_set,
        _ask_set=ask_set,
    )


def _decision_for(
    tool_id: str,
    allowed_set: frozenset[str],
    denied_set: frozenset[str],
    ask_set: frozenset[str],
) -> ToolPolicyDecision:
    if tool_id in denied_set:
        return ToolPolicyDecision("deny", "denied_by_tool_policy", tool_id)
    if tool_id in ask_set:
        return ToolPolicyDecision("ask", "tool_approval_required", tool_id)
    if allowed_set and tool_id not in allowed_set:
        return ToolPolicyDecision("deny", "not_in_tool_allowlist", None)
    return ToolPolicyDecision("allow", "allowed_by_tool_policy", tool_id if allowed_set else None)


def _resolve_references(
    field_name: str,
    references: tuple[str, ...],
    specs: list[PolicyToolSpec],
) -> tuple[str, ...]:
    if not references:
        return ()
    matched: list[str] = []
    for reference in references:
        ref = reference.strip()
        if not ref:
            raise ToolPolicyError(f"{field_name} contains an empty tool reference")
        matches = _matches_reference(ref, specs)
        if not matches:
            raise ToolPolicyError(f"{field_name} reference matched no registered tools: {ref}")
        matched.extend(matches)
    return _dedupe(matched)


def _matches_reference(reference: str, specs: list[PolicyToolSpec]) -> list[str]:
    if _is_glob(reference):
        return [
            spec.id
            for spec in specs
            if fnmatchcase(spec.id, reference) or fnmatchcase(spec.exported_name, reference)
        ]
    return [
        spec.id
        for spec in specs
        if spec.id == reference or spec.exported_name == reference
    ]


def _is_glob(reference: str) -> bool:
    return any(char in reference for char in "*?[")


def _tuple_from_payload(payload: dict[str, Any], *keys: str) -> tuple[str, ...]:
    for key in keys:
        if key not in payload:
            continue
        value = payload[key]
        if value is None:
            return ()
        if not isinstance(value, list):
            raise ToolPolicyError(f"tool_policy.{key} must be an array")
        if not all(isinstance(item, str) for item in value):
            raise ToolPolicyError(f"tool_policy.{key} entries must be strings")
        return tuple(value)
    return ()


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)
