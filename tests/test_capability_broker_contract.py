"""Contract every ``CapabilityBroker`` must honor, regardless of its policy.

A broker returns exactly one of three outcomes — a `CapabilityLease` (grant), a
`CapabilityDenial`, or a `CapabilityPending` (escalate). Whatever it returns, the core relies
on these invariants: a granted lease never *widens* the requested scope (least privilege),
carries a future expiry and a non-empty handle, and every outcome names the requested
capability. Parametrized over a broker factory — a new broker is verified by adding one
``pytest.param``.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from monoid_agent_kernel.core.capability import (
    AutoGrantBroker,
    CapabilityBroker,
    CapabilityDenial,
    CapabilityLease,
    CapabilityPending,
    CapabilityRequest,
    scope_within,
)
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.capability import (
    DenyAllBroker,
    GatewayCapabilityBroker,
    HumanEscalationBroker,
)

BrokerFactory = Callable[[], CapabilityBroker]

BROKER_FACTORIES = [
    pytest.param(lambda: AutoGrantBroker(), id="auto_grant"),
    pytest.param(lambda: DenyAllBroker(), id="deny_all"),
    pytest.param(lambda: HumanEscalationBroker(), id="human_escalation"),
    pytest.param(
        lambda: GatewayCapabilityBroker(
            token_manager=TokenManager.from_secret("x" * 32), tenant_id="t", user_id="u"
        ),
        id="gateway",
    ),
]


@pytest.fixture(params=BROKER_FACTORIES)
def broker(request: pytest.FixtureRequest) -> CapabilityBroker:
    factory: BrokerFactory = request.param
    return factory()


def _request() -> CapabilityRequest:
    return CapabilityRequest(
        capability="web.search",
        scope={"allowed_domains": ["a.edu"]},
        run_id="run_1",
        ttl_seconds=300,
    )


def test_returns_a_grant_union_member(broker: CapabilityBroker) -> None:
    grant = broker.request(_request())
    assert isinstance(grant, (CapabilityLease, CapabilityDenial, CapabilityPending))


def test_outcome_names_the_requested_capability(broker: CapabilityBroker) -> None:
    request = _request()
    grant = broker.request(request)
    if isinstance(grant, CapabilityPending):
        assert grant.request.capability == request.capability
    else:
        assert grant.capability == request.capability


def test_granted_lease_never_widens_scope_and_is_usable(broker: CapabilityBroker) -> None:
    request = _request()
    grant = broker.request(request)
    if not isinstance(grant, CapabilityLease):
        pytest.skip("broker did not grant a lease for this request")
    # Least privilege: the grant is no broader than what was requested.
    assert scope_within(grant.scope, request.scope)
    # Usable: a future expiry and a non-empty handle (never the secret itself here).
    now = time.time()
    assert grant.expires_at > now and grant.is_valid(now)
    assert grant.token_ref
