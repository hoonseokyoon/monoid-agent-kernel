"""Canonical wire identifiers for Monoid Agent Kernel.

New artifacts and wire messages use the ``monoid`` namespace. Readers and validators accept
the pre-rename ``native-agent-runner`` namespace so durable artifacts remain recoverable.
"""

from __future__ import annotations

from collections.abc import Iterable

CURRENT_NAMESPACE = "monoid"
LEGACY_NAMESPACE = "native-agent-runner"

TOKEN_ISSUER = CURRENT_NAMESPACE
LEGACY_TOKEN_ISSUER = LEGACY_NAMESPACE
ACCEPTED_TOKEN_ISSUERS = (TOKEN_ISSUER, LEGACY_TOKEN_ISSUER)

BACKEND_AUDIENCE = f"{CURRENT_NAMESPACE}.backend"
LEGACY_BACKEND_AUDIENCE = f"{LEGACY_NAMESPACE}.backend"
BACKEND_AUDIENCES = (BACKEND_AUDIENCE, LEGACY_BACKEND_AUDIENCE)

TASK_CALLBACK_AUDIENCE = f"{CURRENT_NAMESPACE}.task-callback"
LEGACY_TASK_CALLBACK_AUDIENCE = f"{LEGACY_NAMESPACE}.task-callback"
TASK_CALLBACK_AUDIENCES = (TASK_CALLBACK_AUDIENCE, LEGACY_TASK_CALLBACK_AUDIENCE)


def namespaced_id(name: str, *, namespace: str = CURRENT_NAMESPACE) -> str:
    return f"{namespace}.{name}"


def legacy_namespaced_id(name: str) -> str:
    return namespaced_id(name, namespace=LEGACY_NAMESPACE)


def accepted_namespaced_ids(name: str) -> tuple[str, str]:
    return (namespaced_id(name), legacy_namespaced_id(name))


def accepts_namespaced_id(value: object, name: str) -> bool:
    return isinstance(value, str) and value in accepted_namespaced_ids(name)


def schema_version_property(name: str) -> dict[str, list[str]]:
    return {"enum": list(accepted_namespaced_ids(name))}


def normalize_audiences(audience: str | Iterable[str]) -> set[str]:
    if isinstance(audience, str):
        return {audience}
    return {str(item) for item in audience}
