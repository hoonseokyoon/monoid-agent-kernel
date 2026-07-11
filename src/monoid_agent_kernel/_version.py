from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

PACKAGE_NAME = "monoid-agent-kernel"
FALLBACK_VERSION = "0.18.0"


def package_version() -> str:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return FALLBACK_VERSION


def user_agent(product: str = "monoid-agent-kernel") -> str:
    return f"{product}/{package_version()}"
