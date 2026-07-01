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


def test_token_manager_rotation_accepts_old_key_only_during_grace(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert rotated.verify(new_token, kind="web_gateway", audience="csp.web-gateway").run_id == "run_1"


def test_token_manager_revokes_specific_token_and_issue_cohort(monkeypatch: pytest.MonkeyPatch) -> None:
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
