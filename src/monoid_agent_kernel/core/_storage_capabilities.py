"""Internal home for the storage capability value re-exported by ``hosting``."""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True, kw_only=True)
class StorageCapabilities:
    """Fail-closed declaration of the guarantees a storage adapter actually provides."""

    single_writer: bool = False
    concurrent_writers: bool = False
    compare_and_set: bool = False
    lease_fencing: bool = False
    durable_checkpoints: bool = False
    durable_events: bool = False
    durable_invocations: bool = False
    terminal_first_writer_wins: bool = False
    transactional_outbox: bool = False
    cross_process_notify: bool = False

    def __post_init__(self) -> None:
        for declared_field in fields(self):
            if type(getattr(self, declared_field.name)) is not bool:
                raise ValueError(f"storage capability {declared_field.name} must be a boolean")


__all__ = ["StorageCapabilities"]
