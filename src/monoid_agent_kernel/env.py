from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

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


def load_dotenv(
    path: str | Path = ".env",
    *,
    override: bool = False,
    keys: Iterable[str] | None = None,
) -> dict[str, str]:
    """Load a small dotenv file into ``os.environ``.

    This intentionally supports the common ``KEY=value`` shape without adding a runtime
    dependency. Values are returned for observability/testing; callers must not print them.
    """

    dotenv_path = Path(path)
    try:
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}
    allowed = set(keys) if keys is not None else None
    loaded: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not name or (allowed is not None and name not in allowed):
            continue
        parsed = _parse_dotenv_value(value)
        if override or name not in os.environ:
            os.environ[name] = parsed
            loaded[name] = parsed
    return loaded


def _parse_dotenv_value(value: str) -> str:
    parsed = value.strip()
    if len(parsed) >= 2 and parsed[0] == parsed[-1] and parsed[0] in {"'", '"'}:
        return parsed[1:-1]
    return parsed


def env_name_for_error(name: str) -> str:
    legacy = LEGACY_ENV_ALIASES.get(name)
    if legacy:
        return f"{name} (legacy {legacy} is also accepted)"
    return name
