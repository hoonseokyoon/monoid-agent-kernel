from __future__ import annotations

import pytest

from monoid_agent_kernel.core.safe_evidence import (
    is_safe_opaque_ref,
    is_safe_taxonomy_code,
    is_safe_utc_timestamp,
)


@pytest.mark.parametrize(
    "value",
    ("run_1", "blob:turn_1", "invocation:call_1:3", "provider/id-1", "a" * 256),
)
def test_safe_opaque_ref_accepts_bounded_machine_addresses(value: str) -> None:
    assert is_safe_opaque_ref(value)


@pytest.mark.parametrize(
    "value",
    ("", "private output text", "line\nbreak", "évidence", "a" * 257, True, 1),
)
def test_safe_opaque_ref_rejects_free_text_and_unbounded_values(value: object) -> None:
    assert not is_safe_opaque_ref(value)


@pytest.mark.parametrize("value", ("dispatch_unknown", "provider.refused", "HTTP_429", "a" * 128))
def test_safe_taxonomy_code_accepts_bounded_machine_codes(value: str) -> None:
    assert is_safe_taxonomy_code(value)


@pytest.mark.parametrize(
    "value",
    ("", "raw exception message", "bad/code", "line\nbreak", "a" * 129, True, 1),
)
def test_safe_taxonomy_code_rejects_free_text_and_unbounded_values(value: object) -> None:
    assert not is_safe_taxonomy_code(value)


@pytest.mark.parametrize(
    "value",
    ("2026-08-21T10:00:00Z", "2026-08-21T10:00:00.123456Z"),
)
def test_safe_utc_timestamp_accepts_real_bounded_instants(value: str) -> None:
    assert is_safe_utc_timestamp(value)


@pytest.mark.parametrize(
    "value",
    (
        "2026-02-30T10:00:00Z",
        "2026-08-21 10:00:00Z",
        "2026-08-21T10:00:00+00:00",
        "private timestamp",
        True,
    ),
)
def test_safe_utc_timestamp_rejects_invalid_or_free_text_values(value: object) -> None:
    assert not is_safe_utc_timestamp(value)
