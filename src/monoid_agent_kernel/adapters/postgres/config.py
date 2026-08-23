"""Validated PostgreSQL adapter configuration without optional imports."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_SCHEMA_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,62}\Z", re.ASCII)


@dataclass(frozen=True, kw_only=True)
class PostgresConfig:
    """Connection and pool policy for one PostgreSQL adapter family."""

    dsn: str = field(repr=False)
    schema: str = "monoid_kernel"
    min_pool_size: int = 1
    max_pool_size: int = 10
    connect_timeout_s: int = 10
    pool_timeout_s: float = 30.0
    application_name: str = "monoid-agent-kernel"

    def __post_init__(self) -> None:
        if type(self.dsn) is not str or not self.dsn.strip() or len(self.dsn) > 8192:
            raise ValueError("PostgreSQL dsn must be a non-empty bounded string")
        if "\x00" in self.dsn:
            raise ValueError("PostgreSQL dsn cannot contain NUL")
        if type(self.schema) is not str or _SCHEMA_PATTERN.fullmatch(self.schema) is None:
            raise ValueError(
                "PostgreSQL schema must be an ASCII identifier of at most 63 characters"
            )
        if type(self.min_pool_size) is not int or self.min_pool_size < 0:
            raise ValueError("PostgreSQL min_pool_size must be a non-negative integer")
        if (
            type(self.max_pool_size) is not int
            or self.max_pool_size < 1
            or self.max_pool_size < self.min_pool_size
        ):
            raise ValueError(
                "PostgreSQL max_pool_size must be positive and at least min_pool_size"
            )
        if type(self.connect_timeout_s) is not int or self.connect_timeout_s < 1:
            raise ValueError("PostgreSQL connect_timeout_s must be a positive integer")
        if (
            type(self.pool_timeout_s) not in {int, float}
            or isinstance(self.pool_timeout_s, bool)
            or not 0 < float(self.pool_timeout_s) <= 3600
        ):
            raise ValueError("PostgreSQL pool_timeout_s must be in the range (0, 3600]")
        if (
            type(self.application_name) is not str
            or not self.application_name
            or len(self.application_name.encode("utf-8")) > 63
            or not all(character.isprintable() for character in self.application_name)
        ):
            raise ValueError("PostgreSQL application_name must be a bounded printable string")


__all__ = ["PostgresConfig"]
