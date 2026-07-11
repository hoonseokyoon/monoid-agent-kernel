"""Versioned, side-effect-free decoding for durable JSON artifact families."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

T = TypeVar("T")
U = TypeVar("U")

DurableLoadStatus = Literal["loaded", "migrated", "missing", "corrupt", "unsupported_version"]
Migration = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ArtifactVersion:
    namespace: str
    family: str
    version: int
    raw: str


def parse_artifact_version(value: object) -> ArtifactVersion | None:
    """Parse ``<namespace>.<family>.vN`` without assuming a specific namespace."""
    if not isinstance(value, str):
        return None
    parts = value.rsplit(".", 2)
    if len(parts) != 3 or not parts[0] or not parts[1]:
        return None
    version_token = parts[2]
    version_digits = version_token[1:] if version_token.startswith("v") else ""
    if not version_digits or any(digit < "0" or digit > "9" for digit in version_digits):
        return None
    version = int(version_digits)
    if version < 1:
        return None
    return ArtifactVersion(namespace=parts[0], family=parts[1], version=version, raw=value)


@dataclass(frozen=True)
class DurableLoadResult(Generic[T]):
    status: DurableLoadStatus
    family: str
    current_schema: str
    value: T | None = None
    observed_schema: str | None = None
    migrations: tuple[str, ...] = ()
    error_code: str = ""
    message: str = ""
    sequence: int | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"loaded", "migrated"} and self.value is not None

    def map(self, mapper: Callable[[T], U]) -> DurableLoadResult[U]:
        if self.value is None:
            return DurableLoadResult(
                status=self.status,
                family=self.family,
                current_schema=self.current_schema,
                observed_schema=self.observed_schema,
                migrations=self.migrations,
                error_code=self.error_code,
                message=self.message,
                sequence=self.sequence,
            )
        return DurableLoadResult(
            status=self.status,
            family=self.family,
            current_schema=self.current_schema,
            value=mapper(self.value),
            observed_schema=self.observed_schema,
            migrations=self.migrations,
            error_code=self.error_code,
            message=self.message,
            sequence=self.sequence,
        )


class DurableCodec(Generic[T]):
    """Decode one artifact family and apply registered one-version migrations in order."""

    def __init__(
        self,
        *,
        family: str,
        current_schema: str,
        accepted_namespaces: tuple[str, ...] = ("monoid", "native-agent-runner"),
        migrations: Mapping[int, Migration] | None = None,
    ) -> None:
        current = parse_artifact_version(current_schema)
        if current is None or current.family != family:
            raise ValueError("current_schema must be a valid version for the codec family")
        self.family = family
        self.current_schema = current_schema
        self.current_version = current.version
        self.accepted_namespaces = frozenset(accepted_namespaces)
        self._migrations = dict(migrations or {})

    def missing(self) -> DurableLoadResult[T]:
        return DurableLoadResult(status="missing", family=self.family, current_schema=self.current_schema)

    def corrupt(
        self,
        message: str,
        *,
        observed_schema: str | None = None,
        sequence: int | None = None,
    ) -> DurableLoadResult[T]:
        return DurableLoadResult(
            status="corrupt",
            family=self.family,
            current_schema=self.current_schema,
            observed_schema=observed_schema,
            error_code=f"{self.family.replace('-', '_')}_corrupt",
            message=message,
            sequence=sequence,
        )

    def unsupported(
        self,
        observed_schema: str,
        *,
        sequence: int | None = None,
    ) -> DurableLoadResult[T]:
        return DurableLoadResult(
            status="unsupported_version",
            family=self.family,
            current_schema=self.current_schema,
            observed_schema=observed_schema,
            error_code=f"{self.family.replace('-', '_')}_unsupported_version",
            message=(
                f"unsupported {self.family} schema {observed_schema!r}; "
                f"current reader writes {self.current_schema!r}"
            ),
            sequence=sequence,
        )

    def decode(self, payload: object, loader: Callable[[dict[str, Any]], T]) -> DurableLoadResult[T]:
        if not isinstance(payload, Mapping):
            return self.corrupt(f"{self.family} payload must be an object")
        source = copy.deepcopy(dict(payload))
        version = parse_artifact_version(source.get("schema_version"))
        if version is None:
            return self.corrupt(f"{self.family} schema_version is missing or malformed")
        if version.family != self.family:
            return self.corrupt(
                f"expected {self.family} artifact, found {version.family}",
                observed_schema=version.raw,
            )
        if version.namespace not in self.accepted_namespaces or version.version > self.current_version:
            return self.unsupported(version.raw)

        migrated: list[str] = []
        working = source
        current_version = version.version
        while current_version < self.current_version:
            migration = self._migrations.get(current_version)
            if migration is None:
                return self.unsupported(version.raw)
            try:
                migrated_payload = migration(copy.deepcopy(working))
            except Exception as exc:
                return self.corrupt(
                    f"{self.family} migration v{current_version}->v{current_version + 1} "
                    f"failed ({type(exc).__name__})",
                    observed_schema=version.raw,
                )
            if not isinstance(migrated_payload, dict):
                return self.corrupt(
                    f"{self.family} migration v{current_version}->v{current_version + 1} "
                    "did not return an object",
                    observed_schema=version.raw,
                )
            current_version += 1
            working = copy.deepcopy(migrated_payload)
            working["schema_version"] = f"monoid.{self.family}.v{current_version}"
            migrated.append(f"v{current_version - 1}->v{current_version}")

        try:
            value = loader(copy.deepcopy(working))
        except Exception as exc:
            return self.corrupt(
                f"{self.family} payload validation failed ({type(exc).__name__})",
                observed_schema=version.raw,
            )
        return DurableLoadResult(
            status="migrated" if migrated else "loaded",
            family=self.family,
            current_schema=self.current_schema,
            value=value,
            observed_schema=version.raw,
            migrations=tuple(migrated),
        )
