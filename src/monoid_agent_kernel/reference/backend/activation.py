"""Recognition of activation-local writer-authority revocation."""

from __future__ import annotations

from monoid_agent_kernel.core.authority import WriteAuthorityRevoked


def is_activation_lease_loss(exc: BaseException) -> bool:
    """Return whether control flow is retiring a stale activation."""

    return isinstance(exc, WriteAuthorityRevoked)


__all__ = [
    "is_activation_lease_loss",
]
