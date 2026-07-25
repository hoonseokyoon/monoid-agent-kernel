"""What a consumer is allowed to see of a model call's prompt and output.

Model I/O is the most sensitive data the kernel handles, and different consumers warrant different
answers: a Studio bubble needs the text, an OTel exporter usually needs a digest, a compliance sink
needs a redacted view. So the decision is not global — a `CapturePolicy` is attached per consumer
registration, and one call can serve `full` to one observer and `digest` to another.

The split of responsibilities:

- `RedactionPolicy` is *what* counts as sensitive. Frozen, JSON-serializable, and digestible, so a
  receipt can record which policy was applied without recording the policy's contents.
- `Redactor` is *how* to remove it. A protocol, because the default is substring-and-regex matching
  and any real deployment eventually wants its own detector.
- `CapturePolicy` binds a mode to those two for one consumer.

Redaction runs *after* the digest is computed, never before. The digest identifies the content that
actually went to the provider, so it stays stable across policy changes — that is what makes it
usable as a join key and, later, as a replay key. Redaction changes the view, never the identity.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Protocol, runtime_checkable

from monoid_agent_kernel._policy_util import dedupe, str_tuple
from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.core.wire_validation import (
    WireValidationError,
    parse_literal,
    parse_str,
    require_object,
)

REDACTION_PLACEHOLDER = "[redacted]"

# Substrings that mark a *key* as naming a secret. Matching is on the lowercased key with hyphens
# folded to underscores, and it is a substring test, so "x_api_key" and "X-Api-Key" both match
# "api_key". Promoted from ``core.tool_approval``, which has masked tool-call arguments with this list
# since approvals shipped; model I/O needs the same answer, and two copies would drift.
DEFAULT_SECRET_KEY_PARTS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "credential",
    "password",
    "secret",
    "token",
)

CaptureMode = Literal["none", "digest", "redacted", "full"]
CAPTURE_MODES: tuple[CaptureMode, ...] = ("none", "digest", "redacted", "full")


@lru_cache(maxsize=512)
def _compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


@dataclass(frozen=True)
class RedactionPolicy:
    """What counts as sensitive in a model call's prompt and output.

    Shaped like `permissions.PermissionPolicy` — `from_json` / `to_json` / `merged` — because an
    integrator configuring one has already learned the other. Three axes, because model I/O has two
    very different shapes and key names only help with one of them:

    - `secret_key_parts` masks a mapping *value* whose key names a secret. This is what catches
      structured payloads: tool-call arguments, request message fields, provider extras.
    - `patterns` are regexes applied to free text, which is the only thing that helps with model
      output — there are no key names in a paragraph.
    - `literals` are exact strings to mask, so a caller that already knows a secret's value does not
      have to regex-escape it.

    `secret_key_parts` defaults to `DEFAULT_SECRET_KEY_PARTS`, which makes the empty policy a useful
    one. That is why `from_json` distinguishes an absent key from an explicitly empty list, unlike
    `PermissionPolicy` where both mean "nothing": here they mean "the defaults" and "genuinely
    nothing", and silently reading `[]` as the defaults would ignore a deliberate opt-out.

    Regexes are compiled at construction so a bad pattern fails when the policy is built rather than
    mid-call. They are *not* checked for catastrophic backtracking: patterns come from the
    integrator's own configuration, never from model output, so this is the integrator's own foot to
    shoot — but it is worth knowing the foot is there.
    """

    secret_key_parts: tuple[str, ...] = DEFAULT_SECRET_KEY_PARTS
    patterns: tuple[str, ...] = ()
    literals: tuple[str, ...] = ()
    replacement: str = REDACTION_PLACEHOLDER

    def __post_init__(self) -> None:
        for pattern in self.patterns:
            try:
                _compiled(pattern)
            except re.error as exc:
                raise ValueError(f"invalid redaction pattern {pattern!r}: {exc}") from exc

    @classmethod
    def from_json(cls, payload: Any) -> RedactionPolicy:
        if payload is None:
            return cls()
        payload = require_object(payload, "redaction_policy")
        raw_parts = payload.get("secret_key_parts")
        secret_key_parts = (
            DEFAULT_SECRET_KEY_PARTS
            if raw_parts is None
            else str_tuple(
                raw_parts,
                type_error="secret_key_parts must be an array of strings",
                normalize=True,
            )
        )
        return cls(
            secret_key_parts=dedupe(secret_key_parts),
            patterns=str_tuple(
                payload.get("patterns") or (),
                type_error="patterns must be an array of strings",
                empty_error="empty redaction pattern is not allowed",
            ),
            literals=str_tuple(
                payload.get("literals") or (),
                type_error="literals must be an array of strings",
                empty_error="empty redaction literal is not allowed",
            ),
            replacement=parse_str(payload, "replacement", default=REDACTION_PLACEHOLDER),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "secret_key_parts": list(self.secret_key_parts),
            "patterns": list(self.patterns),
            "literals": list(self.literals),
            "replacement": self.replacement,
        }

    def merged(
        self,
        *,
        secret_key_parts: tuple[str, ...] = (),
        patterns: tuple[str, ...] = (),
        literals: tuple[str, ...] = (),
    ) -> RedactionPolicy:
        """This policy widened by the given rules. Redaction only ever adds, never subtracts."""
        return RedactionPolicy(
            secret_key_parts=dedupe((*self.secret_key_parts, *secret_key_parts)),
            patterns=dedupe((*self.patterns, *patterns)),
            literals=dedupe((*self.literals, *literals)),
            replacement=self.replacement,
        )

    @property
    def digest(self) -> str:
        """A stable id for this policy, for a receipt to record *which* rules were applied.

        The receipt records the digest rather than the policy because the policy itself can name
        secrets — a `literals` entry is a secret by construction.
        """
        return canonical_sha256(self.to_json())

    def names_a_secret(self, key: str) -> bool:
        """Whether `key` names a secret under this policy."""
        lowered = key.lower().replace("-", "_")
        return any(part in lowered for part in self.secret_key_parts)

    def redact_text(self, text: str) -> str:
        """Apply `literals` then `patterns` to free text.

        The order is fixed and the tuples are ordered, which is what makes redaction deterministic:
        the same text under the same policy always yields the same output, so a digest taken over a
        redacted view is comparable across processes.
        """
        for literal in self.literals:
            text = text.replace(literal, self.replacement)
        for pattern in self.patterns:
            text = _compiled(pattern).sub(self.replacement, text)
        return text


@runtime_checkable
class Redactor(Protocol):
    """How to remove what a `RedactionPolicy` names.

    One method, so a partial implementation is not a thing that can exist. The policy is a parameter
    rather than constructor state because the same redactor serves consumers with different policies.

    Two obligations, both checked by `conformance.run_redactor_contract`: the result must be
    deterministic for a given value and policy, and a value under a key the policy calls a secret
    must not survive into the output. A redactor that raises does not violate the contract — the
    kernel treats a failure as fail-closed and drops to a digest — but it must not *silently* return
    unredacted content.
    """

    def redact(self, value: Any, *, policy: RedactionPolicy) -> Any: ...


class DefaultRedactor:
    """Substring-and-regex redaction over JSON-shaped values.

    Recurses into mappings and sequences, masking a mapping value outright when its key names a
    secret and rewriting free text otherwise. Non-text scalars pass through unchanged: an `int` has
    no substring to match, and coercing one to a string to check it would change the payload's shape
    for every caller in order to catch nothing.
    """

    def redact(self, value: Any, *, policy: RedactionPolicy) -> Any:
        if isinstance(value, str):
            return policy.redact_text(value)
        if isinstance(value, Mapping):
            return {
                str(key): (
                    policy.replacement
                    if policy.names_a_secret(str(key))
                    else self.redact(item, policy=policy)
                )
                for key, item in value.items()
            }
        # `str` is a Sequence, and bytes have no text semantics we may assume, so both are excluded.
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self.redact(item, policy=policy) for item in value]
        return value


def redacted_or_none(
    value: Any,
    *,
    policy: RedactionPolicy,
    redactor: Redactor | None = None,
) -> Any | None:
    """Redact `value`, or return `None` if the redactor failed.

    The fail-closed primitive the capture pipeline is built on. A redactor is integrator code that
    may call out to a classifier, a model, or a regex engine on adversarial input, so it *will*
    sometimes raise or hang. When it raises, the only safe answer is to produce nothing: falling
    through to the raw value would turn a redaction failure into a disclosure, which is the opposite
    of what the policy asked for. `None` is distinguishable from a successful redaction to `""`, so
    the caller can report the downgrade rather than mistake it for empty content.
    """
    try:
        return (redactor or DefaultRedactor()).redact(value, policy=policy)
    except Exception:
        return None


@dataclass(frozen=True)
class CapturePolicy:
    """How much of a model call one consumer may see.

    - `none` — nothing. The consumer still gets the receipt's metadata.
    - `digest` — content digests and lengths only. Enough to correlate and to detect drift, with no
      disclosure.
    - `redacted` — content rewritten by `redactor` under `redaction`, dropping to `digest` if that
      fails.
    - `full` — content verbatim.

    `full` is the default because the alternative is worse in practice: a kernel that defaults to
    withholding content makes its own reference Studio render empty bubbles out of the box, and the
    first thing every operator does is turn capture on globally — a broader grant than they needed.
    Defaulting to `full` and attaching a narrower policy per consumer puts the choice where the
    consumer is registered, which is the only place that knows what the consumer does with it.

    `redaction` and `redactor` are read only in `redacted` mode, and are deliberately *not* rejected
    in the others: the fail-closed downgrade produces a `digest`-mode policy that still carries the
    redaction it failed to apply, so the pair must survive a mode change.

    `to_json` carries `mode` and `redaction`. A `redactor` is live code and cannot round-trip, so a
    policy restored from JSON falls back to `DefaultRedactor`; code that wires a custom redactor must
    re-attach it, and `restored_without_redactor` says whether that is needed.
    """

    mode: CaptureMode = "full"
    redaction: RedactionPolicy | None = None
    redactor: Redactor | None = None
    restored_without_redactor: bool = False

    def __post_init__(self) -> None:
        if self.mode not in CAPTURE_MODES:
            raise WireValidationError(f"capture mode must be one of: {', '.join(CAPTURE_MODES)}")

    @property
    def captures_content(self) -> bool:
        """Whether this policy yields content at all, as opposed to metadata about it."""
        return self.mode in {"redacted", "full"}

    @property
    def effective_redaction(self) -> RedactionPolicy:
        return self.redaction if self.redaction is not None else RedactionPolicy()

    @property
    def effective_redactor(self) -> Redactor:
        return self.redactor if self.redactor is not None else DefaultRedactor()

    @classmethod
    def from_json(cls, payload: Any) -> CapturePolicy:
        if payload is None:
            return cls()
        payload = require_object(payload, "capture_policy")
        # Absent and explicit ``null`` both mean "no policy of its own", which resolves to the default
        # via ``effective_redaction``. Only a present object is parsed, so a malformed one still
        # raises rather than silently becoming the default.
        raw_redaction = payload.get("redaction")
        return cls(
            mode=parse_literal(payload, "mode", CAPTURE_MODES, default="full"),
            redaction=(
                RedactionPolicy.from_json(require_object(raw_redaction, "redaction"))
                if raw_redaction is not None
                else None
            ),
            restored_without_redactor=payload.get("redactor") is not None,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "redaction": self.redaction.to_json() if self.redaction is not None else None,
            # Names the fact that a redactor was attached without claiming to carry it, so a
            # round-trip cannot quietly turn custom redaction into the default.
            "redactor": "custom" if self.redactor is not None else None,
        }
