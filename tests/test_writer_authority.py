from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import get_type_hints

import pytest

from monoid_agent_kernel.core.authority import ActivationWriteAuthority
from monoid_agent_kernel.hosting import (
    ReleaseResult,
    RenewResult,
    WriterAuthority,
    WriterAuthorityStore,
    WriterLease,
    WriterLeaseUnavailable,
    WriterToken,
    claim_writer_lease,
    renew_writer_lease,
)


_NOW = datetime(2026, 8, 24, tzinfo=UTC)
_TOKEN = WriterToken(run_id="run-authority", owner_id="worker-1", generation=1)


def _lease() -> WriterLease:
    return WriterLease(
        writer_token=_TOKEN,
        observed_at=_NOW,
        leased_until=_NOW + timedelta(seconds=30),
    )


def _authority(*, revoked: bool = False, expired: bool = False) -> WriterAuthority:
    return WriterAuthority(
        writer_token=_TOKEN,
        observed_at=_NOW,
        leased_until=_NOW - timedelta(seconds=1) if expired else _NOW + timedelta(seconds=30),
        revoked=revoked,
    )


def test_writer_authority_uses_store_observation_time() -> None:
    assert _authority().active is True
    assert _authority(expired=True).active is False
    assert _authority(revoked=True).active is False
    assert _lease().authority == _authority()


@pytest.mark.parametrize(
    ("observed_at", "leased_until"),
    [
        (_NOW.replace(tzinfo=None), _NOW + timedelta(seconds=1)),
        (_NOW, _NOW.replace(tzinfo=None)),
    ],
)
def test_writer_authority_rejects_naive_time(
    observed_at: datetime,
    leased_until: datetime,
) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        WriterAuthority(
            writer_token=_TOKEN,
            observed_at=observed_at,
            leased_until=leased_until,
        )


def test_writer_lease_must_be_active_at_database_observation() -> None:
    with pytest.raises(ValueError, match="active"):
        WriterLease(writer_token=_TOKEN, observed_at=_NOW, leased_until=_NOW)

    with pytest.raises(TypeError, match="WriterToken"):
        WriterLease(  # type: ignore[arg-type]
            writer_token=object(),
            observed_at=_NOW,
            leased_until=_NOW + timedelta(seconds=1),
        )


@pytest.mark.parametrize("status", ["", "ok", 1, True])
def test_writer_result_statuses_are_closed(status: object) -> None:
    with pytest.raises(ValueError, match="renew status"):
        RenewResult(status=status)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="release status"):
        ReleaseResult(status=status)  # type: ignore[arg-type]


def test_writer_result_evidence_matches_status() -> None:
    lease = _lease()
    revoked = _authority(revoked=True)
    assert RenewResult(status="renewed", lease=lease).lease is lease
    assert RenewResult(status="fenced", authority=_authority()).status == "fenced"
    assert ReleaseResult(status="released", authority=revoked).status == "released"
    assert ReleaseResult(status="already_released", authority=revoked).status == "already_released"
    assert ReleaseResult(status="fenced", authority=_authority()).status == "fenced"

    with pytest.raises(ValueError, match="active lease"):
        RenewResult(status="renewed")
    with pytest.raises(ValueError, match="revoked authority"):
        ReleaseResult(status="released", authority=_authority())
    with pytest.raises(TypeError, match="WriterLease"):
        RenewResult(status="renewed", lease=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="WriterAuthority"):
        ReleaseResult(status="fenced", authority=object())  # type: ignore[arg-type]


def test_unavailable_claim_requires_active_evidence() -> None:
    error = WriterLeaseUnavailable(_authority())
    assert error.authority.writer_token == _TOKEN
    with pytest.raises(ValueError, match="must be active"):
        WriterLeaseUnavailable(_authority(expired=True))


class _ClaimingStore:
    def __init__(
        self,
        outcomes: list[object],
        *,
        observation: WriterAuthority | None = None,
    ) -> None:
        self.outcomes = outcomes
        self.observation = observation
        self.claim_calls = 0
        self.read_calls = 0

    def claim(self, run_id: str, owner_id: str, ttl: timedelta) -> object:
        del run_id, owner_id, ttl
        self.claim_calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def read(self, run_id: str) -> WriterAuthority | None:
        del run_id
        self.read_calls += 1
        return self.observation


def test_claim_helper_reuses_the_same_owner_after_response_loss() -> None:
    store = _ClaimingStore([ConnectionError("claim response lost"), _lease()])

    claimed = claim_writer_lease(  # type: ignore[arg-type]
        store,
        _TOKEN.run_id,
        _TOKEN.owner_id,
        timedelta(seconds=30),
    )

    assert claimed == _lease()
    assert store.claim_calls == 2
    assert store.read_calls == 0


def test_claim_helper_recovers_the_exact_token_after_two_ambiguous_responses() -> None:
    store = _ClaimingStore(
        [ConnectionError("first response lost"), ConnectionError("second response lost")],
        observation=_authority(),
    )

    claimed = claim_writer_lease(  # type: ignore[arg-type]
        store,
        _TOKEN.run_id,
        _TOKEN.owner_id,
        timedelta(seconds=30),
    )

    assert claimed == _lease()
    assert store.claim_calls == 2
    assert store.read_calls == 1


def test_claim_helper_does_not_retry_a_definitive_competing_owner() -> None:
    unavailable = WriterLeaseUnavailable(_authority())
    store = _ClaimingStore([unavailable])

    with pytest.raises(WriterLeaseUnavailable) as raised:
        claim_writer_lease(  # type: ignore[arg-type]
            store,
            _TOKEN.run_id,
            "worker-2",
            timedelta(seconds=30),
        )

    assert raised.value is unavailable
    assert store.claim_calls == 1
    assert store.read_calls == 0


class _RenewingStore:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome

    def renew(self, writer_token: WriterToken, ttl: timedelta) -> object:
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def test_renew_helper_preserves_local_authority_only_for_typed_success() -> None:
    local = ActivationWriteAuthority()
    result = renew_writer_lease(
        _RenewingStore(RenewResult(status="renewed", lease=_lease())),  # type: ignore[arg-type]
        _TOKEN,
        timedelta(seconds=10),
        write_authority=local,
    )

    assert result.status == "renewed"
    assert local.revoked is False


def test_renew_helper_revokes_if_store_grants_a_different_token() -> None:
    local = ActivationWriteAuthority()
    wrong_lease = WriterLease(
        writer_token=WriterToken(run_id="run-authority", owner_id="worker-2", generation=2),
        observed_at=_NOW,
        leased_until=_NOW + timedelta(seconds=30),
    )

    with pytest.raises(TypeError, match="different writer token"):
        renew_writer_lease(
            _RenewingStore(RenewResult(status="renewed", lease=wrong_lease)),  # type: ignore[arg-type]
            _TOKEN,
            timedelta(seconds=10),
            write_authority=local,
        )

    assert local.revoked is True


@pytest.mark.parametrize(
    "outcome",
    [
        RenewResult(status="fenced", authority=_authority()),
        object(),
        ConnectionError("renew response lost"),
    ],
)
def test_renew_helper_revokes_on_fence_invalid_result_or_ambiguity(outcome: object) -> None:
    local = ActivationWriteAuthority()

    if isinstance(outcome, BaseException):
        with pytest.raises(ConnectionError, match="response lost"):
            renew_writer_lease(
                _RenewingStore(outcome),  # type: ignore[arg-type]
                _TOKEN,
                timedelta(seconds=10),
                write_authority=local,
            )
    elif isinstance(outcome, RenewResult):
        assert (
            renew_writer_lease(
                _RenewingStore(outcome),  # type: ignore[arg-type]
                _TOKEN,
                timedelta(seconds=10),
                write_authority=local,
            ).status
            == "fenced"
        )
    else:
        with pytest.raises(TypeError, match="invalid renew result"):
            renew_writer_lease(
                _RenewingStore(outcome),  # type: ignore[arg-type]
                _TOKEN,
                timedelta(seconds=10),
                write_authority=local,
            )

    assert local.revoked is True


def test_writer_authority_protocol_annotations_resolve() -> None:
    claim = get_type_hints(WriterAuthorityStore.claim)
    renew = get_type_hints(WriterAuthorityStore.renew)
    release = get_type_hints(WriterAuthorityStore.release)
    read = get_type_hints(WriterAuthorityStore.read)

    assert claim["ttl"] is timedelta
    assert claim["return"] is WriterLease
    assert renew["writer_token"] is WriterToken
    assert renew["return"] is RenewResult
    assert release["return"] is ReleaseResult
    assert read["return"] == WriterAuthority | None
