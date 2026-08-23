"""Provider-neutral contracts for durable run writer authority."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Protocol, get_args

from monoid_agent_kernel.core.authority import ActivationWriteAuthority
from monoid_agent_kernel.hosting.contracts import WriterToken


_RenewStatus = Literal["renewed", "fenced"]
_ReleaseStatus = Literal["released", "already_released", "fenced"]


def _require_aware_datetime(value: object, field_name: str) -> None:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime")


def _require_ttl(ttl: object) -> None:
    if type(ttl) is not timedelta or ttl <= timedelta(0):
        raise ValueError("writer lease ttl must be a positive timedelta")


@dataclass(frozen=True, kw_only=True)
class WriterAuthority:
    """One database-observed writer generation for a run.

    ``observed_at`` and ``leased_until`` come from the authority store's clock. Callers must not
    compare ``leased_until`` with a process-local wall clock when deciding whether this snapshot
    was active at observation time.
    """

    writer_token: WriterToken
    leased_until: datetime
    observed_at: datetime
    revoked: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.writer_token, WriterToken):
            raise TypeError("writer authority writer_token must be WriterToken")
        _require_aware_datetime(self.leased_until, "writer authority leased_until")
        _require_aware_datetime(self.observed_at, "writer authority observed_at")
        if type(self.revoked) is not bool:
            raise ValueError("writer authority revoked must be a boolean")

    @property
    def active(self) -> bool:
        """Whether the authority was active at the store's observation instant."""

        return not self.revoked and self.leased_until > self.observed_at


@dataclass(frozen=True, kw_only=True)
class WriterLease:
    """An active writer token plus database-clock expiry evidence."""

    writer_token: WriterToken
    leased_until: datetime
    observed_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.writer_token, WriterToken):
            raise TypeError("writer lease writer_token must be WriterToken")
        _require_aware_datetime(self.leased_until, "writer lease leased_until")
        _require_aware_datetime(self.observed_at, "writer lease observed_at")
        if self.leased_until <= self.observed_at:
            raise ValueError("writer lease must be active at its observation instant")

    @property
    def authority(self) -> WriterAuthority:
        return WriterAuthority(
            writer_token=self.writer_token,
            leased_until=self.leased_until,
            observed_at=self.observed_at,
        )


@dataclass(frozen=True, kw_only=True)
class RenewResult:
    """Exact-token lease renewal result.

    ``fenced`` covers a missing, wrong-owner, stale-generation, expired, or revoked token. The
    optional authority is an observation aid and never grants the caller that authority.
    """

    status: _RenewStatus
    lease: WriterLease | None = None
    authority: WriterAuthority | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in get_args(_RenewStatus):
            raise ValueError("writer renew status is outside the portable vocabulary")
        if self.lease is not None and not isinstance(self.lease, WriterLease):
            raise TypeError("writer renew lease must be WriterLease")
        if self.authority is not None and not isinstance(self.authority, WriterAuthority):
            raise TypeError("writer renew authority must be WriterAuthority")
        if self.status == "renewed":
            if self.lease is None or self.authority is not None:
                raise ValueError("renewed writer result requires only an active lease")
        elif self.lease is not None:
            raise ValueError("fenced writer renewal cannot carry a granted lease")


@dataclass(frozen=True, kw_only=True)
class ReleaseResult:
    """Exact-token release result with response-loss-safe idempotency."""

    status: _ReleaseStatus
    authority: WriterAuthority | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in get_args(_ReleaseStatus):
            raise ValueError("writer release status is outside the portable vocabulary")
        if self.authority is not None and not isinstance(self.authority, WriterAuthority):
            raise TypeError("writer release authority must be WriterAuthority")
        if self.status in {"released", "already_released"}:
            if self.authority is None or not self.authority.revoked:
                raise ValueError("accepted writer release requires revoked authority evidence")


class WriterLeaseUnavailable(RuntimeError):
    """Raised when another active claim identity owns the run."""

    def __init__(self, authority: WriterAuthority) -> None:
        if not isinstance(authority, WriterAuthority):
            raise TypeError("unavailable writer lease authority must be WriterAuthority")
        if not authority.active:
            raise ValueError("unavailable writer lease evidence must be active")
        super().__init__("run writer lease is held by another active claim identity")
        self.authority = authority


class WriterAuthorityStore(Protocol):
    """Canonical monotonic generation and lease store for fenced run writers.

    A claim identity must be unique for every potentially concurrent activation. Repeating
    ``claim`` with the same run and active owner is response-loss reconciliation: it returns the
    existing token without extending its expiry. Only exact-token ``renew`` extends a lease.
    """

    def claim(self, run_id: str, owner_id: str, ttl: timedelta) -> WriterLease: ...

    def renew(self, writer_token: WriterToken, ttl: timedelta) -> RenewResult: ...

    def release(self, writer_token: WriterToken) -> ReleaseResult: ...

    def read(self, run_id: str) -> WriterAuthority | None: ...


def renew_writer_lease(
    store: WriterAuthorityStore,
    writer_token: WriterToken,
    ttl: timedelta,
    *,
    write_authority: ActivationWriteAuthority,
) -> RenewResult:
    """Renew once and revoke process-local authority on every ambiguous or fenced outcome."""

    _require_ttl(ttl)
    try:
        result = store.renew(writer_token, ttl)
    except BaseException:
        write_authority.revoke()
        raise
    if not isinstance(result, RenewResult):
        write_authority.revoke()
        raise TypeError("writer authority store returned an invalid renew result")
    if (
        result.status == "renewed"
        and result.lease is not None
        and result.lease.writer_token != writer_token
    ):
        write_authority.revoke()
        raise TypeError("writer authority store renewed a different writer token")
    if result.status != "renewed":
        write_authority.revoke()
    return result


__all__ = [
    "WriterAuthority",
    "WriterLease",
    "RenewResult",
    "ReleaseResult",
    "WriterLeaseUnavailable",
    "WriterAuthorityStore",
    "renew_writer_lease",
]
