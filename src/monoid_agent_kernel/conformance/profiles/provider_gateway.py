"""Provider-gateway profile metadata."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from monoid_agent_kernel.conformance.harness import GatewayHarness

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="provider-gateway",
    title="Provider Gateway",
    summary="Gateway runtime with signed scopes, domain boundaries, redirect checks, and effective caps.",
    rule_ids=("OR-01-SCOPE-RELATION", "OR-02-CAPABILITY-BOUNDARY", "OR-08-PROVIDER-CAPS"),
    harnesses=("gateway",),
)


def assert_provider_gateway_profile(harness: GatewayHarness) -> None:
    """Run the Phase 1S provider-gateway conformance smoke matrix."""
    search = harness.call_gateway(
        "web.search",
        {"query": "binding"},
        signed_scope={
            "binding_id": "search_docs",
            "max_calls": 2,
            "max_results": 1,
            "allowed_domains": ["docs.example.test"],
            "blocked_domains": ["blog.example.test"],
        },
    )
    assert search["effective_max_results"] == 1
    assert search["result_count"] == 1
    assert {result["domain"] for result in search["results"]} == {"docs.example.test"}

    narrowed = harness.call_gateway(
        "web.search",
        {"binding_id": "search_docs", "query": "binding", "max_results": 1},
        signed_scope={
            "binding_id": "search_docs",
            "max_calls": 2,
            "max_results": 3,
            "allowed_domains": ["*.example.test"],
        },
    )
    assert narrowed["effective_max_results"] == 1

    _assert_raises(
        lambda: harness.call_gateway(
            "web.search",
            {"binding_id": "search_docs", "query": "binding", "max_results": 2},
            signed_scope={"binding_id": "search_docs", "max_calls": 2, "max_results": 1},
        ),
        "max_results exceeds signed token scope",
    )
    _assert_raises(
        lambda: harness.call_gateway(
            "web.fetch",
            {"url": "https://docs.example.test/monoid-agent-kernel/web"},
            signed_capability="web.search",
            signed_scope={"binding_id": "search_docs", "allowed_domains": ["docs.example.test"]},
        ),
        "capability does not match endpoint",
    )

    fetched = harness.call_gateway(
        "web.fetch",
        {"url": "https://docs.example.test/monoid-agent-kernel/web"},
        signed_scope={
            "binding_id": "fetch_docs",
            "max_calls": 2,
            "max_bytes": 12,
            "timeout_s": 2,
            "allowed_domains": ["docs.example.test"],
        },
    )
    assert fetched["effective_max_bytes"] == 12
    assert fetched["effective_timeout_s"] == 2
    assert fetched["content_bytes"] <= 12

    context = harness.call_gateway(
        "web.context",
        {"query": "binding"},
        signed_scope={
            "binding_id": "context_docs",
            "max_calls": 2,
            "max_tokens": 2,
            "max_urls": 1,
            "max_snippets": 1,
            "allowed_domains": ["docs.example.test"],
        },
    )
    assert context["effective_max_tokens"] == 2
    assert context["effective_max_urls"] == 1
    assert context["effective_max_snippets"] == 1


def _assert_raises(operation: Callable[[], Any], message: str) -> None:
    try:
        operation()
    except Exception as exc:
        if message not in str(exc):
            raise AssertionError(f"expected error containing {message!r}, got {exc!r}") from exc
        return
    raise AssertionError(f"expected error containing {message!r}")
