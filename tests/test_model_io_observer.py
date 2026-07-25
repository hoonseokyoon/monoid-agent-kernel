"""The model-I/O capture pipeline: per-observer policy, digest-before-redaction, fail closed."""

from __future__ import annotations

import json
from typing import Any

import pytest

from monoid_agent_kernel.core._util import sha256_bytes
from monoid_agent_kernel.core.model_io import (
    REDACTION_PLACEHOLDER,
    CapturePolicy,
    ClosableModelIOObserver,
    ModelCallCapture,
    ModelCallReceipt,
    ModelIOObserver,
    ModelIOSubscription,
    RedactionPolicy,
    close_model_io_subscriptions,
    content_digest,
    content_length,
    dispatch_model_call,
)
from monoid_agent_kernel.core.wire_validation import WireValidationError

CONTENT = {"final_text": "the key is sk-abc123", "api_key": "sk-live-secret"}


class Recorder:
    def __init__(self) -> None:
        self.captures: list[ModelCallCapture] = []

    def on_model_call(self, capture: ModelCallCapture) -> None:
        self.captures.append(capture)


class FailingRedactor:
    def redact(self, value: Any, *, policy: RedactionPolicy) -> Any:
        raise RuntimeError("classifier unavailable")


def _dispatch(*policies: CapturePolicy, content: dict[str, Any] | None = None):
    recorders = [Recorder() for _ in policies]
    receipt = dispatch_model_call(
        receipt=ModelCallReceipt(),
        content=CONTENT if content is None else content,
        subscriptions=tuple(
            ModelIOSubscription(recorder, policy)
            for recorder, policy in zip(recorders, policies, strict=True)
        ),
    )
    return receipt, [recorder.captures[0] for recorder in recorders]


# --- per-consumer policy -------------------------------------------------------------------


def test_one_call_serves_a_different_view_to_each_consumer() -> None:
    """The reason the policy is attached per registration rather than set globally."""
    _receipt, (full, digest, none_) = _dispatch(
        CapturePolicy(mode="full"),
        CapturePolicy(mode="digest"),
        CapturePolicy(mode="none"),
    )

    assert full.content == CONTENT
    assert digest.content is None and digest.digests != {}
    assert none_.content is None and none_.digests == {}


def test_none_mode_withholds_even_the_digest() -> None:
    """A digest of a short prompt is a guessable one, so `none` means metadata about the call and
    nothing about its content."""
    _receipt, (capture,) = _dispatch(CapturePolicy(mode="none"))

    assert (capture.mode, capture.content, dict(capture.digests), dict(capture.lengths)) == (
        "none",
        None,
        {},
        {},
    )
    # The receipt still arrives: it holds no content, so it is safe at every mode.
    assert capture.receipt is not None


def test_redacted_mode_masks_secret_named_keys() -> None:
    _receipt, (capture,) = _dispatch(CapturePolicy(mode="redacted"))

    assert capture.content == {
        "final_text": "the key is sk-abc123",
        "api_key": REDACTION_PLACEHOLDER,
    }
    assert capture.mode == "redacted"
    assert capture.was_downgraded is False


def test_a_subscription_cannot_widen_its_own_grant() -> None:
    """The policy lives on the subscription, so the same observer object registered twice gets
    exactly the two views it was registered with."""
    shared = Recorder()

    dispatch_model_call(
        receipt=ModelCallReceipt(),
        content=CONTENT,
        subscriptions=(
            ModelIOSubscription(shared, CapturePolicy(mode="full")),
            ModelIOSubscription(shared, CapturePolicy(mode="none")),
        ),
    )

    assert [capture.mode for capture in shared.captures] == ["full", "none"]


# --- digest before redaction ---------------------------------------------------------------


def test_digests_describe_the_raw_content_under_every_policy() -> None:
    """Named explicitly because v0.21 replay depends on it.

    Two consumers on different policies see different text but must agree on the identity of what the
    provider was sent, so a digest can join a redacted record to a full one.
    """
    _receipt, (full, redacted) = _dispatch(
        CapturePolicy(mode="full"),
        CapturePolicy(mode="redacted"),
    )

    assert full.digests == redacted.digests
    assert redacted.digests["api_key"] == sha256_bytes(b"sk-live-secret")
    # The view differs even though the identity does not.
    assert full.content != redacted.content


def test_the_same_content_digests_identically_under_two_redaction_policies() -> None:
    strict = CapturePolicy(mode="redacted", redaction=RedactionPolicy(literals=("sk-abc123",)))
    lax = CapturePolicy(mode="redacted", redaction=RedactionPolicy(patterns=()))

    _receipt, (first, second) = _dispatch(strict, lax)

    assert first.digests == second.digests
    assert first.content != second.content


def test_lengths_measure_text_and_are_omitted_for_structured_fields() -> None:
    """"Length of the canonical JSON" would measure the serialization, not the content, and an
    operator reading it as a content size would be wrong."""
    _receipt, (capture,) = _dispatch(
        CapturePolicy(mode="digest"),
        content={"final_text": "abcde", "messages": [{"role": "user"}]},
    )

    assert dict(capture.lengths) == {"final_text": 5}
    assert set(capture.digests) == {"final_text", "messages"}


def test_content_digest_does_not_collide_across_shapes() -> None:
    assert content_digest("x") != content_digest(["x"])
    assert content_digest({"a": 1, "b": 2}) == content_digest({"b": 2, "a": 1})
    assert content_length("abc") == 3
    assert content_length({"a": 1}) is None


# --- fail closed ---------------------------------------------------------------------------


def test_a_failing_redactor_downgrades_only_its_own_subscription() -> None:
    """The scenario the fail-closed design exists for: three consumers, the middle one's redactor
    dies, and the other two must not notice."""
    receipt, (full, broken, digest) = _dispatch(
        CapturePolicy(mode="full"),
        CapturePolicy(mode="redacted", redactor=FailingRedactor()),
        CapturePolicy(mode="digest"),
    )

    assert broken.mode == "digest"
    assert broken.downgraded_from == "redacted"
    assert broken.was_downgraded is True
    assert broken.content is None
    # Never the raw value: a redaction failure must not become a disclosure. Checked against the
    # whole capture, not just ``content``, so no other field can carry it out instead.
    assert "sk-live-secret" not in json.dumps(
        {"content": broken.content, "digests": dict(broken.digests), "mode": broken.mode}
    )
    assert full.content == CONTENT and full.was_downgraded is False
    assert digest.mode == "digest" and digest.downgraded_from == ""
    assert receipt.capture_downgrades == 1


def test_every_observer_sees_the_same_downgrade_count() -> None:
    """Resolved in a first pass before any delivery. Delivering as we go would hand the first
    observer zero and the last the true total, and a receipt that disagrees with itself across
    consumers is worse than no count at all."""
    receipt, captures = _dispatch(
        CapturePolicy(mode="full"),
        CapturePolicy(mode="redacted", redactor=FailingRedactor()),
        CapturePolicy(mode="redacted", redactor=FailingRedactor()),
        CapturePolicy(mode="digest"),
    )

    assert receipt.capture_downgrades == 2
    assert {capture.receipt.capture_downgrades for capture in captures} == {2}


def test_downgrades_accumulate_onto_a_receipt_that_already_has_some() -> None:
    receipt = dispatch_model_call(
        receipt=ModelCallReceipt(capture_downgrades=3),
        content=CONTENT,
        subscriptions=(
            ModelIOSubscription(Recorder(), CapturePolicy(mode="redacted", redactor=FailingRedactor())),
        ),
    )

    assert receipt.capture_downgrades == 4


# --- observer failure ----------------------------------------------------------------------


def test_an_observer_that_raises_does_not_fail_the_call_or_starve_the_others() -> None:
    """The call already happened and the provider has already been paid."""

    class Raising:
        def on_model_call(self, capture: ModelCallCapture) -> None:
            del capture
            raise RuntimeError("exporter unavailable")

    after = Recorder()

    receipt = dispatch_model_call(
        receipt=ModelCallReceipt(),
        content=CONTENT,
        subscriptions=(
            ModelIOSubscription(Raising(), CapturePolicy(mode="full")),
            ModelIOSubscription(after, CapturePolicy(mode="full")),
        ),
    )

    assert len(after.captures) == 1
    assert receipt.capture_downgrades == 0


def test_dispatch_with_no_subscriptions_returns_the_receipt_untouched() -> None:
    receipt = ModelCallReceipt(capture_downgrades=2)

    assert dispatch_model_call(receipt=receipt, content=CONTENT, subscriptions=()) is receipt


# --- capture value semantics ---------------------------------------------------------------


def test_a_capture_copies_its_mappings_away_from_the_caller() -> None:
    digests = {"final_text": "sha"}
    capture = ModelCallCapture(receipt=ModelCallReceipt(), mode="digest", digests=digests)

    digests["final_text"] = "tampered"

    assert dict(capture.digests) == {"final_text": "sha"}


def test_a_capture_rejects_an_unknown_mode() -> None:
    with pytest.raises(WireValidationError, match="capture mode must be one of"):
        ModelCallCapture(receipt=ModelCallReceipt(), mode="partial")  # type: ignore[arg-type]


# --- close is optional ---------------------------------------------------------------------


def test_close_is_optional_and_failures_are_tolerated() -> None:
    """`close` is declared on a separate opt-in protocol, because a member with a default in a
    Protocol body is still *required* for structural typing -- so putting it on `ModelIOObserver`
    would reject every observer that does not define one."""

    class Closing:
        def __init__(self) -> None:
            self.closed = False

        def on_model_call(self, capture: ModelCallCapture) -> None:
            del capture

        def close(self) -> None:
            self.closed = True

    class ClosingBadly(Closing):
        def close(self) -> None:
            raise RuntimeError("already gone")

    closing, badly, plain = Closing(), ClosingBadly(), Recorder()

    close_model_io_subscriptions(
        tuple(
            ModelIOSubscription(observer, CapturePolicy())
            for observer in (badly, closing, plain)
        )
    )

    assert closing.closed is True


def test_the_protocols_split_the_required_member_from_the_optional_one() -> None:
    assert isinstance(Recorder(), ModelIOObserver)
    assert not isinstance(Recorder(), ClosableModelIOObserver)
