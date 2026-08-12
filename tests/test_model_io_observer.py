"""The model-I/O capture pipeline: per-observer policy, digest-before-redaction, fail closed."""

from __future__ import annotations

import json
from typing import Any

import pytest

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
    redacted_fields_or_none,
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


def test_none_mode_strips_the_receipts_content_derived_digests_too() -> None:
    """Clearing the per-field digests is not enough: `prompt_digest` walks straight past them.

    `none` promises no content metadata, and a digest of a short prompt is recoverable by hashing
    candidates, so a `none`-mode consumer holding `prompt_digest` has the guarantee in name only.
    """
    recorder = Recorder()

    returned = dispatch_model_call(
        receipt=ModelCallReceipt(
            prompt_digest="sha-prompt",
            request_digest="sha-request",
            redaction_digest="sha-policy",
            usage={"input_tokens": 5},
            latency_ms=12,
        ),
        content=CONTENT,
        subscriptions=(ModelIOSubscription(recorder, CapturePolicy(mode="none")),),
    )
    delivered = recorder.captures[0].receipt

    assert delivered.prompt_digest == ""
    assert delivered.request_digest == ""
    # Metadata about the *call* rather than about what was said still arrives. Withholding it would
    # break an accounting consumer for no privacy gain.
    assert dict(delivered.usage) == {"input_tokens": 5}
    assert delivered.latency_ms == 12
    # ``redaction_digest`` is empty because this subscription applied no rules -- see
    # ``test_the_redaction_digest_names_each_subscriptions_own_rules``. The caller's value is not
    # passed through: it cannot be meaningful at call level, where there is no single policy.
    assert delivered.redaction_digest == ""
    # The receipt returned to the kernel keeps them: it computed them.
    assert (returned.prompt_digest, returned.request_digest) == ("sha-prompt", "sha-request")


def test_a_none_mode_consumer_is_told_the_key_was_withheld_not_that_there_was_none() -> None:
    """Two facts that used to be one empty string.

    `none` clears the key, and a payload that could not be encoded never had one. A consumer
    holding a keyless receipt could not tell which had happened -- so a corpus reader could not
    tell a policy decision from a defect. `withheld` is the policy; `absent` is the defect.
    """
    withheld, genuinely_absent = Recorder(), Recorder()

    dispatch_model_call(
        receipt=ModelCallReceipt(request_digest="sha-request", digest_status="ok"),
        content=CONTENT,
        subscriptions=(ModelIOSubscription(withheld, CapturePolicy(mode="none")),),
    )
    dispatch_model_call(
        receipt=ModelCallReceipt(request_digest="", digest_status="absent"),
        content=CONTENT,
        subscriptions=(ModelIOSubscription(genuinely_absent, CapturePolicy(mode="none")),),
    )

    assert withheld.captures[0].receipt.digest_status == "withheld"
    assert withheld.captures[0].receipt.request_digest == ""
    # Not overwritten: no policy removed a key that no policy ever saw.
    assert genuinely_absent.captures[0].receipt.digest_status == "absent"


def test_a_none_mode_consumer_still_learns_where_the_call_went() -> None:
    """The destination fields are deployment metadata, not content.

    `none` promises the consumer learns nothing about what was said. Where the call was routed is
    not what was said, and the digest is keyed, so it discloses nothing about the endpoint either --
    withholding it would break an operational consumer for no privacy gain, which is the same rule
    that keeps token counts and timings on a `none`-mode receipt.
    """
    recorder = Recorder()

    dispatch_model_call(
        receipt=ModelCallReceipt(
            destination_status="resolved", destination_digest="sha-destination"
        ),
        content=CONTENT,
        subscriptions=(ModelIOSubscription(recorder, CapturePolicy(mode="none")),),
    )
    delivered = recorder.captures[0].receipt

    assert delivered.destination_status == "resolved"
    assert delivered.destination_digest == "sha-destination"


def test_only_none_mode_loses_the_receipt_digests() -> None:
    recorders = [Recorder() for _ in range(3)]
    dispatch_model_call(
        receipt=ModelCallReceipt(prompt_digest="sha-prompt"),
        content=CONTENT,
        subscriptions=tuple(
            ModelIOSubscription(recorder, CapturePolicy(mode=mode))
            for recorder, mode in zip(recorders, ("full", "digest", "redacted"), strict=True)
        ),
    )

    assert [r.captures[0].receipt.prompt_digest for r in recorders] == ["sha-prompt"] * 3


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
    assert redacted.digests["api_key"] == content_digest("sk-live-secret")
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


def test_text_is_domain_separated_from_the_structured_wrapper() -> None:
    """The regression that made text hash under its own shape key.

    Hashing text as bare UTF-8 bytes while wrapping structured values meant a text field whose content
    happened to equal the wrapper serialization collided with the value it wraps — the exact collision
    the wrapper existed to prevent, just moved.
    """
    assert content_digest('{"value":["x"]}') != content_digest(["x"])
    assert content_digest('{"text":"x"}') != content_digest("x")
    # Still stable and still order-independent, which is what a join key needs.
    assert content_digest("abc") == content_digest("abc")


def test_the_redaction_digest_names_each_subscriptions_own_rules() -> None:
    """It is a per-subscription fact, not a per-call one.

    There is no single applied policy at call level — that is the point of attaching one per
    registration — so two redacted consumers with different rules would otherwise get identical audit
    records, and neither could say which rules produced its view.
    """
    strict = RedactionPolicy(literals=("alpha",))
    lax = RedactionPolicy(literals=("beta",))

    _receipt, (first, second) = _dispatch(
        CapturePolicy(mode="redacted", redaction=strict),
        CapturePolicy(mode="redacted", redaction=lax),
        content={"final_text": "alpha beta"},
    )

    assert first.receipt.redaction_digest == strict.digest
    assert second.receipt.redaction_digest == lax.digest
    assert first.content != second.content


@pytest.mark.parametrize("mode", ["none", "digest", "full"])
def test_a_mode_that_applies_no_rules_reports_no_redaction_digest(mode: str) -> None:
    _receipt, (capture,) = _dispatch(
        CapturePolicy(mode=mode, redaction=RedactionPolicy(literals=("alpha",)))  # type: ignore[arg-type]
    )

    assert capture.receipt.redaction_digest == ""


def test_a_downgraded_subscription_reports_no_redaction_digest() -> None:
    """It applied no rules at all. Stamping the policy it *failed* to apply would read as "these rules
    were applied", which `downgraded_from` already reports correctly."""
    _receipt, (capture,) = _dispatch(
        CapturePolicy(
            mode="redacted",
            redaction=RedactionPolicy(literals=("alpha",)),
            redactor=FailingRedactor(),
        )
    )

    assert (capture.mode, capture.downgraded_from) == ("digest", "redacted")
    assert capture.receipt.redaction_digest == ""


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


def test_a_policy_restored_without_its_custom_redactor_downgrades() -> None:
    """Missing machinery is a redaction failure, not a weaker redaction.

    A custom redactor cannot round-trip through JSON. Applying the built-in rules to a policy that
    knows it had one would be the worst outcome available: the consumer is told it received redacted
    content while the classifier that masked more than key names and regexes is simply absent.
    """

    class Classifier:
        """Masks far more than the built-in rules do."""

        def redact(self, value: Any, *, policy: RedactionPolicy) -> Any:
            return dict.fromkeys(value, "[classified]")

    restored = CapturePolicy.from_json(
        CapturePolicy(mode="redacted", redactor=Classifier()).to_json()
    )
    assert restored.restored_without_redactor is True and restored.redactor is None

    receipt, (capture,) = _dispatch(restored)

    assert capture.mode == "digest"
    assert capture.downgraded_from == "redacted"
    assert capture.content is None
    assert receipt.capture_downgrades == 1


def test_a_policy_with_its_redactor_reattached_redacts_normally() -> None:
    """The downgrade is about the absence, not about having come from JSON."""

    class Classifier:
        def redact(self, value: Any, *, policy: RedactionPolicy) -> Any:
            return dict.fromkeys(value, "[classified]")

    restored = CapturePolicy.from_json(
        CapturePolicy(mode="redacted", redactor=Classifier()).to_json()
    )
    reattached = CapturePolicy(
        mode=restored.mode, redaction=restored.redaction, redactor=Classifier()
    )

    _receipt, (capture,) = _dispatch(reattached)

    assert capture.mode == "redacted"
    assert capture.content == dict.fromkeys(CONTENT, "[classified]")


def test_a_redactor_returning_a_non_mapping_is_contained_not_raised() -> None:
    """Resolution happens outside the per-observer guard, so this used to escape `dispatch_model_call`
    and fail a model call the provider had already been paid for.

    `Redactor.redact` is typed `Any -> Any`, and "mask the whole payload" is a tempting one-liner that
    satisfies every leak rule, so this is a reachable third-party shape rather than a contrived one.
    """

    class MaskAll:
        def redact(self, value: Any, *, policy: RedactionPolicy) -> Any:
            return policy.replacement

    class ReturnsAList:
        def redact(self, value: Any, *, policy: RedactionPolicy) -> Any:
            return list(value)

    receipt, (scalar, listed, unaffected) = _dispatch(
        CapturePolicy(mode="redacted", redactor=MaskAll()),
        CapturePolicy(mode="redacted", redactor=ReturnsAList()),
        CapturePolicy(mode="full"),
    )

    assert (scalar.mode, scalar.downgraded_from, scalar.content) == ("digest", "redacted", None)
    assert (listed.mode, listed.downgraded_from, listed.content) == ("digest", "redacted", None)
    assert unaffected.content == CONTENT
    assert receipt.capture_downgrades == 2


def test_a_redactor_returning_a_mapping_subclass_is_accepted() -> None:
    """The check is on the shape the pipeline needs, not on `dict` exactly."""
    from collections import OrderedDict

    class Ordered:
        def redact(self, value: Any, *, policy: RedactionPolicy) -> Any:
            return OrderedDict((key, "[masked]") for key in value)

    _receipt, (capture,) = _dispatch(CapturePolicy(mode="redacted", redactor=Ordered()))

    assert capture.mode == "redacted"
    assert capture.content == dict.fromkeys(CONTENT, "[masked]")


def test_redacted_fields_or_none_reports_both_failure_modes_as_none() -> None:
    class Scalar:
        def redact(self, value: Any, *, policy: RedactionPolicy) -> Any:
            return "flattened"

    assert (
        redacted_fields_or_none(CONTENT, policy=RedactionPolicy(), redactor=FailingRedactor())
        is None
    )
    assert redacted_fields_or_none(CONTENT, policy=RedactionPolicy(), redactor=Scalar()) is None
    assert redacted_fields_or_none(CONTENT, policy=RedactionPolicy()) == {
        "final_text": "the key is sk-abc123",
        "api_key": REDACTION_PLACEHOLDER,
    }


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


def test_full_capture_content_shares_no_nested_structure_with_the_caller() -> None:
    """`dict(content)` copied only the outer mapping, so a nested message list stayed shared.

    A caller mutating its own payload after dispatch changed captures observers had already retained,
    while the digests kept describing the pre-mutation value. A capture is meant to be a settled record.
    """
    content = {"messages": [{"role": "user", "text": "original"}], "final_text": "done"}
    recorder = Recorder()

    dispatch_model_call(
        receipt=ModelCallReceipt(),
        content=content,
        subscriptions=(ModelIOSubscription(recorder, CapturePolicy(mode="full")),),
    )
    content["messages"][0]["text"] = "mutated after the fact"  # type: ignore[index]

    retained = recorder.captures[0].content
    assert retained == {"messages": [{"role": "user", "text": "original"}], "final_text": "done"}


def test_an_in_place_redactor_touches_neither_the_caller_nor_its_peers() -> None:
    """Nothing in the `Redactor` contract forbids editing mappings in place, and it is the natural way
    to write one — so each redacted subscription is handed a payload it owns.

    With only the outer mapping copied, such a redactor mutated the caller's settled payload *and* the
    input the next redacted subscription saw, which made the first consumer's rules everyone's.
    """

    class InPlace:
        def redact(self, value: Any, *, policy: RedactionPolicy) -> Any:
            for message in value.get("messages", []):
                message["text"] = "[masked]"
            return value

    class Observes:
        def redact(self, value: Any, *, policy: RedactionPolicy) -> Any:
            return {"saw": [dict(message) for message in value.get("messages", [])]}

    content: dict[str, Any] = {"messages": [{"text": "original"}]}
    first, second = Recorder(), Recorder()

    dispatch_model_call(
        receipt=ModelCallReceipt(),
        content=content,
        subscriptions=(
            ModelIOSubscription(first, CapturePolicy(mode="redacted", redactor=InPlace())),
            ModelIOSubscription(second, CapturePolicy(mode="redacted", redactor=Observes())),
        ),
    )

    assert content == {"messages": [{"text": "original"}]}
    assert first.captures[0].content == {"messages": [{"text": "[masked]"}]}
    assert second.captures[0].content == {"saw": [{"text": "original"}]}


def test_an_all_none_dispatch_hashes_nothing() -> None:
    """Hashing walks every field and, for a value with no JSON form, materializes a string of it — so a
    run wired to nothing but a `none`-mode observer was paying to digest resolved media it discarded."""
    rendered: list[str] = []

    class Loud:
        def __repr__(self) -> str:
            rendered.append("rendered")
            return "loud"

    dispatch_model_call(
        receipt=ModelCallReceipt(),
        content={"blob": Loud()},
        subscriptions=(ModelIOSubscription(Recorder(), CapturePolicy(mode="none")),),
    )

    assert rendered == []


def test_a_downgraded_subscription_still_gets_its_metadata_computed() -> None:
    """Keyed on the *resolved* modes: a subscription downgraded from `redacted` lands on `digest`, which
    does expose this metadata, so keying on the declared mode would have withheld it."""
    recorder = Recorder()

    dispatch_model_call(
        receipt=ModelCallReceipt(),
        content=CONTENT,
        subscriptions=(
            ModelIOSubscription(
                recorder, CapturePolicy(mode="redacted", redactor=FailingRedactor())
            ),
        ),
    )

    capture = recorder.captures[0]
    assert capture.mode == "digest"
    assert capture.digests == {key: content_digest(value) for key, value in CONTENT.items()}


def test_full_capture_content_is_detached_once_not_per_observer() -> None:
    """Content can carry resolved media, so copying per subscriber is real cost for a case an observer
    is already forbidden to cause — treating the capture as read-only is part of its contract, whereas
    a caller mutating its own dict violates nothing."""
    first, second = Recorder(), Recorder()

    dispatch_model_call(
        receipt=ModelCallReceipt(),
        content={"messages": [{"text": "original"}]},
        subscriptions=(
            ModelIOSubscription(first, CapturePolicy(mode="full")),
            ModelIOSubscription(second, CapturePolicy(mode="full")),
        ),
    )

    assert first.captures[0].content == second.captures[0].content
    assert first.captures[0].content["messages"] is second.captures[0].content["messages"]  # type: ignore[index]


def test_content_a_deepcopy_refuses_still_reaches_the_observer() -> None:
    """Degraded isolation is survivable; failing a model call the provider has already been paid for is
    not."""

    class Uncopyable:
        def __deepcopy__(self, memo: dict[int, Any]) -> Any:
            raise TypeError("cannot copy")

    sentinel = Uncopyable()
    recorder = Recorder()

    dispatch_model_call(
        receipt=ModelCallReceipt(),
        content={"handle": sentinel},
        subscriptions=(ModelIOSubscription(recorder, CapturePolicy(mode="full")),),
    )

    assert recorder.captures[0].content == {"handle": sentinel}  # type: ignore[comparison-overlap]


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


def test_a_shared_observer_is_closed_once() -> None:
    """Registering one exporter under two policies is a shape `ModelIOSubscription` supports, so
    closing per subscription would ask a `close` that flushes or commits to be idempotent — and the
    second call's exception is swallowed by the guard that makes a broken exporter survivable."""

    class CountingClose:
        def __init__(self) -> None:
            self.closes = 0

        def on_model_call(self, capture: ModelCallCapture) -> None:
            del capture

        def close(self) -> None:
            self.closes += 1

    shared, other = CountingClose(), CountingClose()

    close_model_io_subscriptions(
        (
            ModelIOSubscription(shared, CapturePolicy(mode="full")),
            ModelIOSubscription(other, CapturePolicy(mode="full")),
            ModelIOSubscription(shared, CapturePolicy(mode="none")),
        )
    )

    assert (shared.closes, other.closes) == (1, 1)


def test_the_protocols_split_the_required_member_from_the_optional_one() -> None:
    assert isinstance(Recorder(), ModelIOObserver)
    assert not isinstance(Recorder(), ClosableModelIOObserver)


def test_a_cyclic_digest_payload_is_elided_rather_than_expanded() -> None:
    """The depth bound alone made this strictly worse than no bound.

    `_jsonish` caps at `MAX_JSONISH_DEPTH`, and a container reachable from itself is re-expanded
    once per edge per level — so a mapping with two self-referencing keys becomes ~2**64 nodes. The
    cap converted a fast `RecursionError` into a hang, in the digest path that runs *before* a
    completed provider turn is returned. Reachable from a third-party adapter handing back a cyclic
    tool-call argument.

    Third traversal in this tree to need an ancestor guard: `public_view.preview_value` and
    `public_view.touches_redacted_path` were both given one during review and this one was not.
    """
    import time

    from monoid_agent_kernel.core.model_io import _jsonish

    cyclic: dict[str, object] = {}
    for index in range(20):
        cyclic[f"k{index}"] = cyclic

    start = time.monotonic()
    published = _jsonish(cyclic)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, f"still re-expanding: {elapsed:.2f}s"
    assert set(published.values()) == {"[circular-elided]"}
    # Ancestors on the current path only: a value shared twice without a cycle still renders twice.
    shared = {"x": 1}
    assert _jsonish({"a": shared, "b": shared}) == {"a": {"x": 1}, "b": {"x": 1}}
    # And an ordinary payload is untouched.
    assert _jsonish({"n": 1, "s": "hi", "l": [1, 2]}) == {"n": 1, "s": "hi", "l": [1, 2]}
