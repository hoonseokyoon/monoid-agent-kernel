from __future__ import annotations

import os

LEGACY_ENV_ALIASES: dict[str, str] = {
    "MONOID_LLM_GATEWAY_URL": "NAR_LLM_GATEWAY_URL",
    "MONOID_LLM_GATEWAY_TOKEN": "NAR_LLM_GATEWAY_TOKEN",
    "MONOID_WEB_GATEWAY_TOKEN": "NAR_WEB_GATEWAY_TOKEN",
    "MONOID_BACKEND_ADMIN_TOKEN": "NAR_BACKEND_ADMIN_TOKEN",
    "MONOID_LLM_GATEWAY_ADMIN_TOKEN": "NAR_LLM_GATEWAY_ADMIN_TOKEN",
    "MONOID_WEB_GATEWAY_ADMIN_TOKEN": "NAR_WEB_GATEWAY_ADMIN_TOKEN",
    "MONOID_BACKEND_TOKEN_SECRET": "NAR_BACKEND_TOKEN_SECRET",
    "MONOID_ALLOW_DIRECT_PROVIDER_API": "NAR_ALLOW_DIRECT_PROVIDER_API",
    "MONOID_OTEL_ENDPOINT": "NAR_OTEL_ENDPOINT",
}


def getenv(name: str) -> str | None:
    value = os.environ.get(name)
    if value is not None:
        return value
    legacy = LEGACY_ENV_ALIASES.get(name)
    if legacy:
        return os.environ.get(legacy)
    return None


def env_name_for_error(name: str) -> str:
    legacy = LEGACY_ENV_ALIASES.get(name)
    if legacy:
        return f"{name} (legacy {legacy} is also accepted)"
    return name
