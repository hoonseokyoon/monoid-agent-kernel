from __future__ import annotations

import pytest

from monoid_agent_kernel.core.scope import (
    ScopePolicyError,
    domain_patterns_within,
    effective_signed_scope,
    scope_within,
)


def test_scope_within_preserves_list_numeric_and_scalar_rules() -> None:
    assert scope_within({"allowed_domains": ["a.edu"]}, {"allowed_domains": ["a.edu", "b.edu"]})
    assert not scope_within({"allowed_domains": ["c.edu"]}, {"allowed_domains": ["a.edu"]})
    assert scope_within({"max_results": 5}, {"max_results": 10})
    assert not scope_within({"max_results": 15}, {"max_results": 10})
    assert scope_within({"region": "us"}, {"region": "us"})
    assert not scope_within({"region": "eu"}, {"region": "us"})
    assert scope_within({"allowed_domains": ["x"]}, {})


def test_scope_within_allows_wildcard_domain_narrowing() -> None:
    assert scope_within(
        {"allowed_domains": ["*.docs.example.test"]},
        {"allowed_domains": ["*.example.test"]},
    )
    assert scope_within({"allowed_domains": ["docs.example.test"]}, {"allowed_domains": ["*.example.test"]})
    assert not scope_within({"allowed_domains": ["*.example.test"]}, {"allowed_domains": ["*.docs.example.test"]})


def test_domain_patterns_within_handles_exact_wildcard_and_global_patterns() -> None:
    assert domain_patterns_within(("docs.example.test",), ("docs.example.test",))
    assert domain_patterns_within(("*.docs.example.test",), ("*.example.test",))
    assert domain_patterns_within(("anything.test",), ("*",))
    assert not domain_patterns_within(("*",), ("*.example.test",))
    assert not domain_patterns_within(("blog.example.test",), ("*.docs.example.test",))


def test_effective_signed_scope_applies_omitted_signed_caps_and_domains() -> None:
    effective = effective_signed_scope(
        {
            "binding_id": "search_docs",
            "max_results": 2,
            "allowed_domains": ["docs.example.test"],
            "blocked_domains": ["blog.example.test"],
        },
        {"query": "binding"},
    )

    assert effective["binding_id"] == "search_docs"
    assert effective["max_results"] == 2
    assert effective["allowed_domains"] == ["docs.example.test"]
    assert effective["blocked_domains"] == ["blog.example.test"]


def test_effective_signed_scope_allows_requested_narrowing() -> None:
    effective = effective_signed_scope(
        {"max_results": 5, "allowed_domains": ["*.example.test"]},
        {"max_results": 2, "allowed_domains": ["*.docs.example.test"]},
    )

    assert effective["max_results"] == 2
    assert effective["allowed_domains"] == ["*.docs.example.test"]


@pytest.mark.parametrize(
    ("scope", "payload", "key", "reason"),
    [
        ({"max_results": 1}, {"max_results": 2}, "max_results", "exceeds"),
        ({"max_calls": 1}, {"max_calls": 0}, "max_calls", "request_not_positive"),
        ({"max_calls": 0}, {}, "max_calls", "signed_not_positive"),
        ({"binding_id": "a"}, {"binding_id": "b"}, "binding_id", "exceeds"),
        (
            {"allowed_domains": ["docs.example.test"]},
            {"allowed_domains": ["blog.example.test"]},
            "allowed_domains",
            "exceeds",
        ),
    ],
)
def test_effective_signed_scope_reports_policy_violations(
    scope: dict[str, object],
    payload: dict[str, object],
    key: str,
    reason: str,
) -> None:
    with pytest.raises(ScopePolicyError) as exc_info:
        effective_signed_scope(scope, payload)

    assert exc_info.value.key == key
    assert exc_info.value.reason == reason
