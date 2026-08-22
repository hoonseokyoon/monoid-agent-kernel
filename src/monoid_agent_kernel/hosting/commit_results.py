"""Shared interpretation of fenced durable mutation results."""

from __future__ import annotations

from typing import Literal

from monoid_agent_kernel.core.authority import ActivationWriteAuthority
from monoid_agent_kernel.hosting.contracts import CommitResult

AcceptedCommitStatus = Literal["committed", "already_committed", "conflict"]


def accept_fenced_commit_result(
    result: object,
    *,
    write_authority: ActivationWriteAuthority,
) -> AcceptedCommitStatus:
    """Validate a sink result and convert ``fenced`` into activation-wide revocation."""

    if not isinstance(result, CommitResult):
        raise TypeError("fenced sink returned an invalid commit result")
    if result.status == "fenced":
        write_authority.revoke()
        write_authority.assert_active()
        raise AssertionError("revoked authority unexpectedly remained active")
    if result.status not in {"committed", "already_committed", "conflict"}:
        # ``CommitResult`` currently makes this unreachable. Keep the adapter boundary explicit so
        # a future vocabulary extension cannot silently acquire success semantics here.
        raise TypeError("fenced sink returned an unsupported commit status")
    return result.status


__all__ = ["accept_fenced_commit_result"]
