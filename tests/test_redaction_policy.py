"""RedactionPolicy, CapturePolicy and the default redactor."""

from __future__ import annotations

import json
from typing import Any

import pytest

from monoid_agent_kernel.core.model_io import (
    DEFAULT_SECRET_KEY_PARTS,
    REDACTION_PLACEHOLDER,
    CapturePolicy,
    DefaultRedactor,
    RedactionPolicy,
    redacted_or_none,
)
from monoid_agent_kernel.core.wire_validation import WireValidationError

# --- RedactionPolicy ------------------------------------------------------------------------


def test_the_empty_policy_already_masks_secret_named_keys() -> None:
    """A policy nobody configured still has to be worth having."""
    assert RedactionPolicy().secret_key_parts == DEFAULT_SECRET_KEY_PARTS
    assert RedactionPolicy().names_a_secret("api_key") is True


@pytest.mark.parametrize(
    "key",
    ["api_key", "API_KEY", "X-Api-Key", "x_api_key", "authorization", "refresh_token", "db_password"],
)
def test_secret_key_matching_folds_case_and_hyphens_and_matches_substrings(key: str) -> None:
    assert RedactionPolicy().names_a_secret(key) is True


@pytest.mark.parametrize("key", ["prompt", "temperature", "keyring", "tokenizer_config"])
def test_a_key_that_only_looks_adjacent_is_not_a_secret(key: str) -> None:
    # "keyring" contains neither "api_key" nor "secret"; "tokenizer_config" does contain "token",
    # so it *is* matched -- which is the documented substring behaviour, not a bug. Assert the ones
    # that genuinely must not match.
    if key == "tokenizer_config":
        assert RedactionPolicy().names_a_secret(key) is True
        return
    assert RedactionPolicy().names_a_secret(key) is False


def test_from_json_tells_an_absent_list_apart_from_an_empty_one() -> None:
    """The default is non-empty, so conflating the two would ignore a deliberate opt-out."""
    assert RedactionPolicy.from_json({}).secret_key_parts == DEFAULT_SECRET_KEY_PARTS
    assert RedactionPolicy.from_json({"secret_key_parts": []}).secret_key_parts == ()


def test_from_json_normalizes_and_dedupes_secret_key_parts() -> None:
    policy = RedactionPolicy.from_json({"secret_key_parts": ["API_KEY", " api_key ", "Secret"]})

    assert policy.secret_key_parts == ("api_key", "secret")


def test_a_programmatic_rule_is_normalized_too() -> None:
    """The regression: normalizing only in `from_json` left every programmatic caller broken.

    `names_a_secret` folds the *candidate* key to lowercase, so an un-normalized rule could never
    match — `secret_key_parts=("API_KEY",)` matched nothing and the value it was written to mask was
    delivered inside a `redacted` capture. The built-in list is already lowercase, which is exactly
    what hid it.
    """
    policy = RedactionPolicy(secret_key_parts=("API_KEY", " api_key ", "Nonce", ""))

    assert policy.secret_key_parts == ("api_key", "nonce")
    assert policy.names_a_secret("api_key") is True
    assert policy.names_a_secret("API_KEY") is True
    assert policy.names_a_secret("x-nonce") is True
    assert DefaultRedactor().redact({"API_KEY": "sk-leaks"}, policy=policy) == {
        "API_KEY": REDACTION_PLACEHOLDER
    }


def test_merged_normalizes_the_rules_it_adds() -> None:
    widened = RedactionPolicy(secret_key_parts=()).merged(secret_key_parts=("Nonce", "NONCE"))

    assert widened.secret_key_parts == ("nonce",)


def test_two_policies_differing_only_in_rule_case_are_equal_and_digest_alike() -> None:
    """A consequence of normalizing in the constructor, and the behaviour an audit record wants: the
    same rules written two ways are the same rules."""
    assert RedactionPolicy(secret_key_parts=("API_KEY",)) == RedactionPolicy(
        secret_key_parts=("api_key",)
    )
    assert (
        RedactionPolicy(secret_key_parts=("API_KEY",)).digest
        == RedactionPolicy(secret_key_parts=("api_key",)).digest
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"secret_key_parts": "api_key"},  # a bare string is not an array
        {"patterns": "sk-"},
        {"patterns": [""]},  # an empty regex matches everywhere
        {"literals": ["  "]},  # a blank literal would mask between every character
    ],
)
def test_from_json_rejects_malformed_rule_lists(payload: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RedactionPolicy.from_json(payload)


@pytest.mark.parametrize("payload", [{"patterns": ""}, {"literals": ""}, {"patterns": 0}])
def test_from_json_rejects_a_falsy_value_of_the_wrong_type(payload: dict[str, Any]) -> None:
    """The `payload.get(key) or ()` idiom read these as an empty list, so a malformed configuration
    silently disabled text masking while still producing captures labelled `redacted`."""
    with pytest.raises(ValueError, match="must be an array of strings"):
        RedactionPolicy.from_json(payload)


@pytest.mark.parametrize("key", ["patterns", "literals"])
def test_from_json_still_treats_absent_and_null_rule_lists_as_empty(key: str) -> None:
    assert RedactionPolicy.from_json({}) == RedactionPolicy()
    assert RedactionPolicy.from_json({key: None}) == RedactionPolicy()


def test_an_invalid_regex_fails_when_the_policy_is_built() -> None:
    """Not mid-call: a policy is configuration, and configuration errors belong at load time."""
    with pytest.raises(ValueError, match="invalid redaction pattern"):
        RedactionPolicy(patterns=("sk-[",))


def test_json_round_trip() -> None:
    policy = RedactionPolicy(
        secret_key_parts=("api_key",),
        patterns=(r"sk-[A-Za-z0-9]+",),
        literals=("hunter2",),
        replacement="***",
    )

    assert RedactionPolicy.from_json(json.loads(json.dumps(policy.to_json()))) == policy


def test_from_json_accepts_none_as_the_default_policy() -> None:
    assert RedactionPolicy.from_json(None) == RedactionPolicy()


def test_merged_only_widens() -> None:
    policy = RedactionPolicy(patterns=("a",), literals=("x",), replacement="***")

    widened = policy.merged(secret_key_parts=("nonce",), patterns=("a", "b"), literals=("y",))

    assert widened.patterns == ("a", "b")  # deduped, order preserved
    assert widened.literals == ("x", "y")
    assert widened.secret_key_parts == (*DEFAULT_SECRET_KEY_PARTS, "nonce")
    assert widened.replacement == "***"
    assert policy.patterns == ("a",)


def test_digest_identifies_the_rules_without_disclosing_them() -> None:
    policy = RedactionPolicy(literals=("hunter2",))

    # A receipt records the digest precisely because a literal is a secret by construction.
    assert "hunter2" not in policy.digest
    assert policy.digest == RedactionPolicy(literals=("hunter2",)).digest
    assert policy.digest != RedactionPolicy(literals=("hunter3",)).digest
    assert policy.digest != RedactionPolicy(literals=("hunter2",), replacement="***").digest


def test_redact_text_applies_literals_then_patterns_in_order() -> None:
    policy = RedactionPolicy(patterns=(r"sk-[a-z0-9]+",), literals=("hunter2",))

    redacted = policy.redact_text("key sk-abc123 pass hunter2 end")

    assert redacted == f"key {REDACTION_PLACEHOLDER} pass {REDACTION_PLACEHOLDER} end"
    # Determinism is what lets a digest be taken over a redacted view.
    assert redacted == policy.redact_text("key sk-abc123 pass hunter2 end")


# --- DefaultRedactor -----------------------------------------------------------------------


def test_default_redactor_masks_secret_keys_at_every_depth() -> None:
    policy = RedactionPolicy()

    redacted = DefaultRedactor().redact(
        {
            "api_key": "sk-live-1",
            "nested": {"Authorization": "Bearer t", "keep": "visible"},
            "items": [{"password": "p"}, "visible too"],
        },
        policy=policy,
    )

    assert redacted == {
        "api_key": REDACTION_PLACEHOLDER,
        "nested": {"Authorization": REDACTION_PLACEHOLDER, "keep": "visible"},
        "items": [{"password": REDACTION_PLACEHOLDER}, "visible too"],
    }


def test_default_redactor_leaves_non_text_scalars_alone() -> None:
    """Coercing an int to check it for substrings would reshape every payload to catch nothing."""
    policy = RedactionPolicy(literals=("7",))

    assert DefaultRedactor().redact({"count": 7, "flag": True, "none": None}, policy=policy) == {
        "count": 7,
        "flag": True,
        "none": None,
    }


def test_default_redactor_does_not_treat_bytes_as_a_sequence_of_items() -> None:
    policy = RedactionPolicy()

    assert DefaultRedactor().redact(b"raw", policy=policy) == b"raw"


def test_default_redactor_rewrites_free_text_where_keys_cannot_help() -> None:
    """Model output is a paragraph; there are no key names in it."""
    policy = RedactionPolicy(patterns=(r"\d{3}-\d{2}-\d{4}",))

    assert DefaultRedactor().redact("ssn 123-45-6789 ok", policy=policy) == (
        f"ssn {REDACTION_PLACEHOLDER} ok"
    )


# --- redacted_or_none: the fail-closed primitive -------------------------------------------


def test_redacted_or_none_returns_nothing_when_the_redactor_raises() -> None:
    class Failing:
        def redact(self, value: Any, *, policy: RedactionPolicy) -> Any:
            raise RuntimeError("classifier unavailable")

    assert redacted_or_none({"a": "b"}, policy=RedactionPolicy(), redactor=Failing()) is None


def test_redacted_or_none_distinguishes_failure_from_redacting_to_empty() -> None:
    """Otherwise a caller cannot tell a downgrade from genuinely empty content."""
    policy = RedactionPolicy(literals=("everything",))

    assert redacted_or_none("everything", policy=policy, redactor=None) == REDACTION_PLACEHOLDER
    assert redacted_or_none("", policy=policy) == ""


def test_redacted_or_none_defaults_to_the_built_in_redactor() -> None:
    assert redacted_or_none({"token": "t"}, policy=RedactionPolicy()) == {
        "token": REDACTION_PLACEHOLDER
    }


# --- CapturePolicy -------------------------------------------------------------------------


def test_the_default_mode_is_full() -> None:
    """The documented release decision: a kernel whose own Studio renders empty bubbles gets capture
    turned on globally, which is a broader grant than any consumer needed."""
    assert CapturePolicy().mode == "full"
    assert CapturePolicy().captures_content is True


@pytest.mark.parametrize(
    ("mode", "captures"),
    [("none", False), ("digest", False), ("redacted", True), ("full", True)],
)
def test_captures_content_separates_content_from_metadata(mode: str, captures: bool) -> None:
    assert CapturePolicy(mode=mode).captures_content is captures  # type: ignore[arg-type]


def test_an_unknown_mode_is_rejected() -> None:
    with pytest.raises(WireValidationError, match="capture mode must be one of"):
        CapturePolicy(mode="partial")  # type: ignore[arg-type]


def test_redaction_survives_a_mode_change_to_digest() -> None:
    """The fail-closed downgrade produces a digest-mode policy that still carries the redaction it
    could not apply, so rejecting that pair would make the downgrade unrepresentable."""
    policy = CapturePolicy(mode="redacted", redaction=RedactionPolicy(literals=("x",)))

    downgraded = CapturePolicy(mode="digest", redaction=policy.redaction, redactor=policy.redactor)

    assert downgraded.mode == "digest"
    assert downgraded.redaction == policy.redaction


def test_effective_accessors_fill_in_the_defaults() -> None:
    policy = CapturePolicy(mode="redacted")

    assert policy.effective_redaction == RedactionPolicy()
    assert isinstance(policy.effective_redactor, DefaultRedactor)


def test_json_round_trip_names_a_custom_redactor_without_carrying_it() -> None:
    """A redactor is live code. A round trip must not quietly turn custom redaction into the
    default, so the payload records that one was attached and the restored policy says so."""

    class Custom:
        def redact(self, value: Any, *, policy: RedactionPolicy) -> Any:
            return value

    policy = CapturePolicy(mode="redacted", redaction=RedactionPolicy(), redactor=Custom())
    payload = json.loads(json.dumps(policy.to_json()))

    assert payload["redactor"] == "custom"

    restored = CapturePolicy.from_json(payload)

    assert restored.mode == "redacted"
    assert restored.redaction == RedactionPolicy()
    assert restored.redactor is None
    assert restored.restored_without_redactor is True


def test_the_lost_redactor_marker_survives_repeated_round_trips() -> None:
    """One hop set the marker; the second used to clear it and fall back to the built-in rules.

    A restored policy has `redactor is None`, so keying the serialized marker on the redactor alone
    wrote `null` on the way back out. A policy that crosses two services — config store to gateway to
    kernel — is one hop, not zero, so this was reachable by the deployment the marker exists for.
    """

    class Custom:
        def redact(self, value: Any, *, policy: RedactionPolicy) -> Any:
            return dict.fromkeys(value, "[classified]")

    policy = CapturePolicy(mode="redacted", redactor=Custom())

    for _hop in range(4):
        policy = CapturePolicy.from_json(policy.to_json())
        assert policy.restored_without_redactor is True
        assert policy.to_json()["redactor"] == "custom"


def test_from_json_without_a_redactor_does_not_claim_one_was_lost() -> None:
    restored = CapturePolicy.from_json({"mode": "digest"})

    assert restored.mode == "digest"
    assert restored.redaction is None
    assert restored.restored_without_redactor is False


def test_from_json_treats_absent_and_null_redaction_alike_but_still_validates_a_present_one() -> None:
    assert CapturePolicy.from_json({"mode": "full"}).redaction is None
    assert CapturePolicy.from_json({"mode": "full", "redaction": None}).redaction is None
    with pytest.raises(WireValidationError):
        CapturePolicy.from_json({"mode": "full", "redaction": "not-an-object"})


def test_from_json_accepts_none_and_rejects_an_unknown_mode() -> None:
    assert CapturePolicy.from_json(None) == CapturePolicy()
    with pytest.raises(WireValidationError, match="must be one of"):
        CapturePolicy.from_json({"mode": "partial"})
