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


# Operator kill switch for the durable ``model.output.delta`` / ``model.reasoning.delta`` channel.
# Named here rather than in ``loop`` so the documentation and the Studio CLI reference one constant.
OUTPUT_DELTAS_ENV = "MONOID_OUTPUT_DELTAS"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def getenv_bool(name: str, *, default: bool) -> bool:
    """Read a boolean switch, raising on a value that is neither true nor false.

    Deliberately not the `getenv(...) != "1"` shape used for direct-provider permission: that one is
    fail-closed, which is right for a permission and wrong for a switch, because every typo reads as
    "off". Here a typo reads as neither, and an operator who set the variable at all meant to change
    something — so an unrecognized value is an error at startup rather than a setting that silently
    did nothing.
    """
    raw = getenv(name)
    normalized = "" if raw is None else raw.strip().lower()
    if not normalized:
        # An empty value reads as unset, not as an error. `MONOID_FOO=` is the ordinary way to blank
        # a key in a dotenv file, and `load_env_file` copies empty values into `os.environ` — so
        # rejecting it would turn a routine edit into a hard failure of every run.
        return default
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{env_name_for_error(name)}={raw!r} is not a boolean; "
        f"use one of {sorted(_TRUE_VALUES)} or {sorted(_FALSE_VALUES)}"
    )


def env_name_for_error(name: str) -> str:
    legacy = LEGACY_ENV_ALIASES.get(name)
    if legacy:
        return f"{name} (legacy {legacy} is also accepted)"
    return name
