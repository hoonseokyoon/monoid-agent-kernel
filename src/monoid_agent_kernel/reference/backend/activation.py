"""Activation-local control flow for hosted writer-authority loss."""

from __future__ import annotations

from monoid_agent_kernel.core.outcome import InterruptionCause
from monoid_agent_kernel.core.result import Suspension
from monoid_agent_kernel.errors import NativeAgentError


class ActivationLeaseLost(Exception):
    """Unwind a stale hosted activation without projecting a run outcome."""


def raise_on_lease_loss(suspension: Suspension | None) -> None:
    """Reject a lease-loss observation before a host mutates run projections."""

    if (
        suspension is not None
        and suspension.interruption_cause is InterruptionCause.LEASE_LOST
    ):
        raise ActivationLeaseLost("hosted activation lost writer authority")


def is_activation_lease_loss(exc: BaseException) -> bool:
    """Recognize both the host unwind signal and a core write-fence refusal."""

    return isinstance(exc, ActivationLeaseLost) or (
        isinstance(exc, NativeAgentError) and exc.error_code == "lease_lost"
    )


__all__ = [
    "ActivationLeaseLost",
    "is_activation_lease_loss",
    "raise_on_lease_loss",
]
