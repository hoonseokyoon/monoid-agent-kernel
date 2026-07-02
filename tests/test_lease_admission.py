from __future__ import annotations

import pytest

from monoid_agent_kernel.core.lease_admission import (
    LeaseAdmissionError,
    sanitize_denied_capability_result,
    validate_lease_admission,
)


def test_validate_lease_admission_accepts_narrower_scope() -> None:
    validate_lease_admission(
        "web.search",
        {"allowed_domains": ["*.example.test"], "max_results": 5},
        "web.search",
        {"allowed_domains": ["docs.example.test"], "max_results": 3},
    )


@pytest.mark.parametrize(
    ("lease_scope", "reason"),
    [
        ({"allowed_domains": ["*.example.test"]}, "scope_widened"),
        ({"max_results": 6}, "scope_widened"),
        ({"regions": ["us", "eu"]}, "scope_widened"),
    ],
)
def test_validate_lease_admission_rejects_scope_widening(
    lease_scope: dict[str, object],
    reason: str,
) -> None:
    with pytest.raises(LeaseAdmissionError) as exc_info:
        validate_lease_admission(
            "web.search",
            {"allowed_domains": ["docs.example.test"], "max_results": 5, "regions": ["us"]},
            "web.search",
            lease_scope,
        )

    assert exc_info.value.reason == reason


def test_validate_lease_admission_rejects_capability_mismatch() -> None:
    with pytest.raises(LeaseAdmissionError) as exc_info:
        validate_lease_admission("web.search", {}, "web.fetch", {})

    assert exc_info.value.reason == "capability_mismatch"


def test_sanitize_denied_capability_result_strips_grant_material() -> None:
    sanitized = sanitize_denied_capability_result(
        {
            "answer": "Approve",
            "approved": True,
            "granted": True,
            "lease": {"capability": "web.search", "token_ref": "secret-ref://lease"},
            "token_ref": "secret-ref://lease",
        },
        reason="policy denied",
    )

    assert sanitized["answer"] == "Deny"
    assert sanitized["approved"] is False
    assert sanitized["granted"] is False
    assert sanitized["reason"] == "policy denied"
    assert "lease" not in sanitized
    assert "token_ref" not in sanitized
