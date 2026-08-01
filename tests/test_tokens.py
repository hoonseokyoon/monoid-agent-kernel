from __future__ import annotations

import base64
import json

import pytest

from monoid_agent_kernel.reference._shared import tokens as token_module
from monoid_agent_kernel.reference._shared.tokens import TokenError, TokenManager


def _header(token: str) -> dict[str, object]:
    raw = token.split(".", 1)[0]
    padding = "=" * (-len(raw) % 4)
    return json.loads(base64.urlsafe_b64decode((raw + padding).encode("ascii")).decode("utf-8"))


def _issue(manager: TokenManager, *, ttl_s: int = 600) -> str:
    return manager.issue(
        kind="web_gateway",
        audience="csp.web-gateway",
        run_id="run_1",
        tenant_id="tenant_a",
        user_id="user_a",
        ttl_s=ttl_s,
    )


def test_token_manager_issues_kid_header() -> None:
    manager = TokenManager.from_keyring({"kid-a": "a" * 32}, active_kid="kid-a")

    token = _issue(manager)

    assert _header(token)["kid"] == "kid-a"
    claims = manager.verify(token, kind="web_gateway", audience="csp.web-gateway", run_id="run_1")
    assert claims.run_id == "run_1"


def test_token_manager_rejects_key_ids_before_lossy_coercion() -> None:
    with pytest.raises(TokenError, match="token key id must be a non-empty string"):
        TokenManager.from_keyring(
            {1: "a" * 32, "1": "b" * 32},  # type: ignore[dict-item]
            active_kid="1",
        )

    manager = TokenManager.from_keyring({"1": "a" * 32}, active_kid="1")
    with pytest.raises(TokenError, match="rotation key id must be a non-empty string"):
        manager.rotate_key(
            key_id=1,  # type: ignore[arg-type]
            secret="b" * 32,
            grace_s=30,
            now=100,
        )


def test_token_manager_rotation_accepts_old_key_only_during_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"t": 1000.0}
    monkeypatch.setattr(token_module.time, "time", lambda: clock["t"])
    manager = TokenManager.from_keyring({"kid-a": "a" * 32}, active_kid="kid-a")
    old_token = _issue(manager, ttl_s=600)

    rotated = manager.rotate_key(key_id="kid-b", secret="b" * 32, grace_s=30, now=1000)
    new_token = _issue(rotated, ttl_s=600)

    assert _header(new_token)["kid"] == "kid-b"
    assert rotated.verify(old_token, kind="web_gateway", audience="csp.web-gateway").token_id
    clock["t"] = 1031.0
    with pytest.raises(TokenError, match="signing key"):
        rotated.verify(old_token, kind="web_gateway", audience="csp.web-gateway")
    assert (
        rotated.verify(new_token, kind="web_gateway", audience="csp.web-gateway").run_id == "run_1"
    )


def test_token_manager_revokes_specific_token_and_issue_cohort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"t": 2000.0}
    monkeypatch.setattr(token_module.time, "time", lambda: clock["t"])
    manager = TokenManager.from_secret("x" * 32)
    token = _issue(manager, ttl_s=600)
    claims = manager.verify(token, kind="web_gateway", audience="csp.web-gateway")

    revoked_one = manager.revoke_token_id(claims.token_id)
    with pytest.raises(TokenError, match="revoked"):
        revoked_one.verify(token, kind="web_gateway", audience="csp.web-gateway")

    revoked_cohort = manager.revoke_issued_before(claims.issued_at + 1)
    with pytest.raises(TokenError, match="revoked"):
        revoked_cohort.verify(token, kind="web_gateway", audience="csp.web-gateway")


def test_token_manager_ceil_fractional_revoke_watermark(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"t": 2000.1}
    monkeypatch.setattr(token_module.time, "time", lambda: clock["t"])
    manager = TokenManager.from_secret("x" * 32)
    token = _issue(manager, ttl_s=600)
    claims = manager.verify(token, kind="web_gateway", audience="csp.web-gateway")
    assert claims.issued_at == 2000

    revoked = manager.revoke_issued_before(2000.9)

    assert revoked.revoked_before == 2001
    with pytest.raises(TokenError, match="revoked"):
        revoked.verify(token, kind="web_gateway", audience="csp.web-gateway")


def test_token_manager_wraps_malformed_header_as_token_error() -> None:
    manager = TokenManager.from_secret("x" * 32)

    with pytest.raises(TokenError, match="invalid token header"):
        manager.verify("bm90LWpzb24.e30.signature", kind="web_gateway", audience="csp.web-gateway")


def test_token_manager_normalizes_direct_metadata_before_signing() -> None:
    manager = TokenManager.from_secret("x" * 32)
    token = manager.issue(
        kind="web_gateway",
        audience="csp.web-gateway",
        run_id="run\ud800",
        tenant_id="tenant\ud800",
        user_id="user\ud800",
        ttl_s=600,
        metadata={"score": float("nan"), "label": "bad\ud800"},
    )

    claims = manager.verify(
        token,
        kind="web_gateway",
        audience="csp.web-gateway",
        run_id="run\ud800",
    )
    assert claims.run_id == "run\ufffd"
    assert claims.tenant_id == "tenant\ufffd"
    assert claims.user_id == "user\ufffd"
    assert claims.metadata == {"score": None, "label": "bad\ufffd"}

    with pytest.raises(TokenError, match="ttl_s must be a positive integer"):
        manager.issue(
            kind="web_gateway",
            audience="csp.web-gateway",
            run_id="run_1",
            tenant_id="tenant",
            user_id="user",
            ttl_s=float("nan"),  # type: ignore[arg-type]
        )


def test_token_manager_rejects_metadata_key_collisions() -> None:
    manager = TokenManager.from_secret("x" * 32)
    metadata = {"\ud800": 1}
    metadata["\ufffd"] = 2

    with pytest.raises(ValueError, match="keys collide"):
        manager.issue(
            kind="web_gateway",
            audience="csp.web-gateway",
            run_id="run_1",
            tenant_id="tenant",
            user_id="user",
            ttl_s=600,
            metadata=metadata,
        )


def test_token_manager_verify_accepts_lazy_audience_iterables() -> None:
    manager = TokenManager.from_secret("x" * 32)
    token = _issue(manager)

    claims = manager.verify(
        token,
        kind="web_gateway",
        audience=(value for value in ("legacy.example", "csp.web-gateway")),
    )

    assert claims.audience == "csp.web-gateway"


def test_token_manager_verify_rejects_non_string_lazy_audience_entries() -> None:
    manager = TokenManager.from_secret("x" * 32)
    token = _issue(manager)

    with pytest.raises(TokenError, match="expected token audience must be a non-empty string"):
        manager.verify(
            token,
            kind="web_gateway",
            audience=(value for value in ("csp.web-gateway", 7)),  # type: ignore[arg-type]
        )


def test_token_manager_rejects_string_where_accepted_issuer_collection_is_required() -> None:
    with pytest.raises(TokenError, match="accepted token issuers must be a list"):
        TokenManager(
            secret=b"x" * 32,
            accepted_issuers="abc",  # type: ignore[arg-type]
        )
    with pytest.raises(TokenError, match="accepted token issuers must be a list"):
        TokenManager.from_keyring(
            {"kid-a": b"x" * 32},
            active_kid="kid-a",
            accepted_issuers="abc",  # type: ignore[arg-type]
        )


def test_token_manager_rejects_string_where_revoked_id_collection_is_required() -> None:
    manager = TokenManager.from_secret("x" * 32)
    token = _issue(manager)
    token_id = manager.verify(
        token,
        kind="web_gateway",
        audience="csp.web-gateway",
    ).token_id

    with pytest.raises(TokenError, match="revoked token ids must be a list"):
        TokenManager(
            secret=manager.secret,
            revoked_token_ids=token_id,  # type: ignore[arg-type]
        )
    with pytest.raises(TokenError, match="revoked token ids must be a list"):
        TokenManager.from_keyring(
            {"kid-a": manager.secret},
            active_kid="kid-a",
            revoked_token_ids=token_id,  # type: ignore[arg-type]
        )

    revoked = TokenManager(secret=manager.secret, revoked_token_ids=[token_id])
    with pytest.raises(TokenError, match="token revoked"):
        revoked.verify(token, kind="web_gateway", audience="csp.web-gateway")


@pytest.mark.parametrize(
    ("accepted_issuers", "revoked_token_ids"),
    [
        (["legacy.example"], ["jti_1"]),
        (("legacy.example",), ("jti_1",)),
        ({"legacy.example"}, {"jti_1"}),
        (frozenset({"legacy.example"}), frozenset({"jti_1"})),
    ],
    ids=("list", "tuple", "set", "frozenset"),
)
def test_token_manager_accepts_explicit_text_collection_types(
    accepted_issuers: object,
    revoked_token_ids: object,
) -> None:
    manager = TokenManager(
        secret=b"x" * 32,
        accepted_issuers=accepted_issuers,  # type: ignore[arg-type]
        revoked_token_ids=revoked_token_ids,  # type: ignore[arg-type]
    )

    assert manager.accepted_issuers == ("legacy.example",)
    assert manager.revoked_token_ids == frozenset({"jti_1"})


@pytest.mark.parametrize("invalid", [b"abc", {"abc": True}])
def test_token_manager_rejects_bytes_and_mappings_as_text_collections(
    invalid: object,
) -> None:
    with pytest.raises(TokenError, match="must be a list, tuple, set, or frozenset"):
        TokenManager(
            secret=b"x" * 32,
            accepted_issuers=invalid,  # type: ignore[arg-type]
        )


def test_token_manager_rejects_integer_signing_secrets() -> None:
    with pytest.raises(TokenError, match="must be exactly str or bytes"):
        TokenManager(secret=32)  # type: ignore[arg-type]

    with pytest.raises(TokenError, match="must be exactly str or bytes"):
        TokenManager.from_keyring(
            {"kid-a": 32},  # type: ignore[dict-item]
            active_kid="kid-a",
        )
