"""Reusable conformance rules for fenced durable stream stores."""

from __future__ import annotations

import hashlib
from typing import Protocol

from monoid_agent_kernel.conformance.report import (
    ConformanceRuleOutcome,
    observation,
    outcome_from_observations,
    safe_exception_summary,
)
from monoid_agent_kernel.hosting import (
    DurableStreamIdentity,
    DurableStreamStore,
    WriterToken,
)


DURABLE_STREAM_STORE_PROFILE = "durable-stream-store-contract"


class DurableStreamStoreHarness(Protocol):
    store: DurableStreamStore
    writer_token: WriterToken

    def replace_writer(self) -> WriterToken: ...


class DurableStreamStoreHarnessFactory(Protocol):
    def __call__(self, run_id: str) -> DurableStreamStoreHarness: ...


def _identity(run_id: str, suffix: str) -> DurableStreamIdentity:
    return DurableStreamIdentity(
        run_id=run_id,
        stream_id=f"contract-stream-{suffix}",
        logical_call_id=f"contract-call-{suffix}",
        channel="output",
    )


def _error(rule_id: str, exc: Exception) -> ConformanceRuleOutcome:
    return ConformanceRuleOutcome(
        rule_id=rule_id,
        profile_id=DURABLE_STREAM_STORE_PROFILE,
        status="error",
        error=safe_exception_summary(exc),
    )


def run_durable_stream_store_contract(
    factory: DurableStreamStoreHarnessFactory,
) -> tuple[ConformanceRuleOutcome, ...]:
    """Verify cursor replay, generation reset, final seal, and exact-token fencing."""

    outcomes: list[ConformanceRuleOutcome] = []
    try:
        run_id = "contract-stream-cursor"
        harness = factory(run_id)
        identity = _identity(run_id, "cursor")
        opened = harness.store.open(identity, writer_token=harness.writer_token)
        reopened = harness.store.open(identity, writer_token=harness.writer_token)
        first = harness.store.append(
            identity,
            generation=1,
            start_offset=0,
            data=b"first ",
            writer_token=harness.writer_token,
        )
        repeated = harness.store.append(
            identity,
            generation=1,
            start_offset=0,
            data=b"first ",
            writer_token=harness.writer_token,
        )
        conflict = harness.store.append(
            identity,
            generation=1,
            start_offset=0,
            data=b"other",
            writer_token=harness.writer_token,
        )
        gap = harness.store.append(
            identity,
            generation=1,
            start_offset=100,
            data=b"gap",
            writer_token=harness.writer_token,
        )
        second = harness.store.append(
            identity,
            generation=1,
            start_offset=6,
            data=b"second",
            writer_token=harness.writer_token,
        )
        page_one = harness.store.read_after(identity, generation=1, cursor=0, limit=1)
        page_two = harness.store.read_after(
            identity,
            generation=1,
            cursor=page_one.next_cursor,
            limit=10,
        )
        outcomes.append(
            outcome_from_observations(
                "STREAM-01-CURSOR-RECONNECT",
                DURABLE_STREAM_STORE_PROFILE,
                (
                    observation("open", expected="opened", actual=opened.status),
                    observation(
                        "idempotent-open",
                        expected="already_open",
                        actual=reopened.status,
                    ),
                    observation("first-append", expected="committed", actual=first.status),
                    observation(
                        "idempotent-append",
                        expected="already_committed",
                        actual=repeated.status,
                    ),
                    observation("conflicting-digest", expected="conflict", actual=conflict.status),
                    observation("ahead-cursor", expected="gap", actual=gap.status),
                    observation("second-append", expected="committed", actual=second.status),
                    observation(
                        "paged-replay",
                        expected=hashlib.sha256(b"first second").hexdigest(),
                        actual=hashlib.sha256(
                            b"".join(
                                chunk.data
                                for chunk in page_one.chunks + page_two.chunks
                            )
                        ).hexdigest(),
                    ),
                    observation(
                        "misaligned-cursor",
                        expected="gap",
                        actual=harness.store.read_after(
                            identity,
                            generation=1,
                            cursor=1,
                        ).status,
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("STREAM-01-CURSOR-RECONNECT", exc))

    try:
        run_id = "contract-stream-reset"
        harness = factory(run_id)
        identity = _identity(run_id, "reset")
        harness.store.open(identity, writer_token=harness.writer_token)
        harness.store.append(
            identity,
            generation=1,
            start_offset=0,
            data=b"old generation",
            writer_token=harness.writer_token,
        )
        reset = harness.store.reset(
            identity,
            expected_generation=1,
            reset_id="contract-reset-2",
            writer_token=harness.writer_token,
        )
        repeated = harness.store.reset(
            identity,
            expected_generation=1,
            reset_id="contract-reset-2",
            writer_token=harness.writer_token,
        )
        old_append = harness.store.append(
            identity,
            generation=1,
            start_offset=len(b"old generation"),
            data=b"stale",
            writer_token=harness.writer_token,
        )
        replacement = harness.store.append(
            identity,
            generation=2,
            start_offset=0,
            data=b"replacement",
            writer_token=harness.writer_token,
        )
        outcomes.append(
            outcome_from_observations(
                "STREAM-02-GENERATION-RESET",
                DURABLE_STREAM_STORE_PROFILE,
                (
                    observation("reset", expected="reset", actual=reset.status),
                    observation(
                        "idempotent-reset",
                        expected="already_reset",
                        actual=repeated.status,
                    ),
                    observation(
                        "old-generation-read",
                        expected="reset",
                        actual=harness.store.read_after(
                            identity,
                            generation=1,
                            cursor=0,
                        ).status,
                    ),
                    observation(
                        "old-generation-append",
                        expected="old_generation",
                        actual=old_append.status,
                    ),
                    observation(
                        "replacement-append",
                        expected="committed",
                        actual=replacement.status,
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("STREAM-02-GENERATION-RESET", exc))

    try:
        run_id = "contract-stream-seal-fence"
        harness = factory(run_id)
        identity = _identity(run_id, "seal-fence")
        stale = harness.writer_token
        harness.store.open(identity, writer_token=stale)
        data = b"sealed stream"
        harness.store.append(
            identity,
            generation=1,
            start_offset=0,
            data=data,
            writer_token=stale,
        )
        digest = hashlib.sha256(data).hexdigest()
        sealed = harness.store.seal(
            identity,
            generation=1,
            final_size_bytes=len(data),
            final_sha256=digest,
            writer_token=stale,
        )
        repeated = harness.store.seal(
            identity,
            generation=1,
            final_size_bytes=len(data),
            final_sha256=digest,
            writer_token=stale,
        )
        late = harness.store.append(
            identity,
            generation=1,
            start_offset=len(data),
            data=b"late",
            writer_token=stale,
        )
        current = harness.replace_writer()
        fenced = harness.store.append(
            identity,
            generation=1,
            start_offset=len(data),
            data=b"stale writer",
            writer_token=stale,
        )
        current_open = harness.store.open(identity, writer_token=current)
        outcomes.append(
            outcome_from_observations(
                "STREAM-03-SEAL-AND-FENCE",
                DURABLE_STREAM_STORE_PROFILE,
                (
                    observation("seal", expected="sealed", actual=sealed.status),
                    observation(
                        "idempotent-seal",
                        expected="already_sealed",
                        actual=repeated.status,
                    ),
                    observation("late-append", expected="sealed", actual=late.status),
                    observation("stale-writer", expected="fenced", actual=fenced.status),
                    observation(
                        "replacement-observes-seal",
                        expected="sealed",
                        actual=current_open.status,
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("STREAM-03-SEAL-AND-FENCE", exc))
    return tuple(outcomes)


__all__ = [
    "DURABLE_STREAM_STORE_PROFILE",
    "DurableStreamStoreHarness",
    "DurableStreamStoreHarnessFactory",
    "run_durable_stream_store_contract",
]
