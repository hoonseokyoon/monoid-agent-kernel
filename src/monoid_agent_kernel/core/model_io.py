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
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Literal, Protocol, runtime_checkable

from monoid_agent_kernel._policy_util import dedupe, str_tuple
from monoid_agent_kernel.core._util import canonical_sha256, sha256_bytes
from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.core.wire_validation import (
    WireValidationError,
    parse_bool,
    parse_int,
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


def content_digest(value: Any) -> str:
    """A stable digest of one captured content field, computed on the raw value.

    Text hashes its UTF-8 bytes; anything else hashes as canonical JSON, so a digest does not depend
    on mapping order. Wrapped in a single-key object rather than hashed directly, so a string and a
    list containing that string cannot collide.
    """
    if isinstance(value, str):
        return sha256_bytes(value.encode("utf-8"))
    return canonical_sha256({"value": _jsonish(value)})


def content_length(value: Any) -> int | None:
    """Character length for text, `None` for anything else.

    Deliberately not "length of the canonical JSON": for a structured payload that number measures
    the serialization, not the content, and an operator reading it as a content size would be wrong.
    A missing answer is better than a misleading one.
    """
    return len(value) if isinstance(value, str) else None


def _jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonish(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonish(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(frozen=True)
class ModelCallReceipt:
    """What happened on one model call, without any of what was said.

    The receipt is metadata only: digests, counts, timings, taxonomy. That is what makes it safe to
    hand to every consumer regardless of its `CapturePolicy` — a `none`-mode observer still gets a
    full receipt, because nothing in here discloses content. Content travels separately, gated by the
    policy.

    Two digests, because they answer different questions. `prompt_digest` covers the assembled prompt
    and stays stable when tool definitions or generation settings change around it, which is what you
    want when asking "did the model see the same thing twice". `request_digest` covers the whole
    request and is the exact replay key. Both are computed on the raw request, *before* redaction, so
    they identify what actually went to the provider and stay comparable across policy changes.

    `attempts` and `provider_retried` are not the same fact and neither implies the other.
    `attempts` counts the calls the kernel made to the adapter. `provider_retried` is the adapter
    reporting that *it* retried internally — a gateway can retry three times inside one call the
    kernel counts as one attempt, and a receipt that only had `attempts` would show that as a clean
    single call.

    `stop_reason` is a plain string rather than the provider `Literal`. A receipt is an audit record:
    a provider that starts returning a fifth stop reason must be recordable without a kernel change,
    and `core` cannot depend on `providers` in any case.
    """

    context: InvocationContext = field(default_factory=lambda: InvocationContext())
    model: ModelConfig = field(default_factory=lambda: ModelConfig())
    provider_name: str = ""
    prompt_digest: str = ""
    request_digest: str = ""
    stop_reason: str = ""
    usage: Mapping[str, int] = field(default_factory=dict)
    latency_ms: int = 0
    attempts: int = 1
    provider_retried: bool = False
    error_code: str = ""
    provider_error_code: str = ""
    retryable: bool = False
    http_status: int | None = None
    redaction_digest: str = ""
    capture_downgrades: int = 0

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("model call attempts must be 1 or greater")
        if self.latency_ms < 0:
            raise ValueError("model call latency_ms must not be negative")
        if self.capture_downgrades < 0:
            raise ValueError("model call capture_downgrades must not be negative")
        for key, value in self.usage.items():
            if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int):
                raise WireValidationError("model call usage must be a mapping of str to int")
        object.__setattr__(self, "usage", dict(self.usage))

    @property
    def succeeded(self) -> bool:
        """Whether the call produced a turn. A failed call still gets a receipt."""
        return self.error_code == ""

    @property
    def trace_id(self) -> str:
        return self.context.trace_id

    @property
    def span_id(self) -> str:
        return self.context.span_id

    def with_error(self, exc: BaseException) -> ModelCallReceipt:
        """This receipt marked failed, carrying whatever taxonomy the exception exposes.

        `ModelAdapterError` classifies itself — provider code, retryability, HTTP status — and the
        providers already raise it, so the runner does not re-derive any of that. Anything else is
        recorded by its type name rather than its message: an arbitrary exception's message can carry
        request content, and the whole point of the receipt is that it holds none.
        """
        error_code = getattr(exc, "error_code", "") or type(exc).__name__
        return replace(
            self,
            error_code=str(error_code),
            provider_error_code=str(getattr(exc, "provider_error_code", "") or ""),
            retryable=bool(getattr(exc, "retryable", False)),
            http_status=getattr(exc, "http_status", None),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "context": self.context.to_json(),
            "model": self.model.to_json(),
            "provider_name": self.provider_name,
            "prompt_digest": self.prompt_digest,
            "request_digest": self.request_digest,
            "stop_reason": self.stop_reason,
            "usage": dict(self.usage),
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
            "provider_retried": self.provider_retried,
            "error_code": self.error_code,
            "provider_error_code": self.provider_error_code,
            "retryable": self.retryable,
            "http_status": self.http_status,
            "redaction_digest": self.redaction_digest,
            "capture_downgrades": self.capture_downgrades,
        }

    @classmethod
    def from_json(cls, payload: Any) -> ModelCallReceipt:
        payload = require_object(payload, "model_call_receipt")
        raw_usage = payload.get("usage")
        usage = require_object(raw_usage, "usage") if raw_usage is not None else {}
        raw_status = payload.get("http_status")
        return cls(
            context=InvocationContext.from_json(payload.get("context") or {}),
            model=ModelConfig.from_json(payload.get("model")),
            provider_name=parse_str(payload, "provider_name"),
            prompt_digest=parse_str(payload, "prompt_digest"),
            request_digest=parse_str(payload, "request_digest"),
            stop_reason=parse_str(payload, "stop_reason"),
            usage=usage,
            latency_ms=parse_int(payload, "latency_ms"),
            attempts=parse_int(payload, "attempts", default=1),
            provider_retried=parse_bool(payload, "provider_retried"),
            error_code=parse_str(payload, "error_code"),
            provider_error_code=parse_str(payload, "provider_error_code"),
            retryable=parse_bool(payload, "retryable"),
            http_status=None if raw_status is None else parse_int(payload, "http_status"),
            redaction_digest=parse_str(payload, "redaction_digest"),
            capture_downgrades=parse_int(payload, "capture_downgrades"),
        )


@dataclass(frozen=True)
class ModelCallCapture:
    """One model call as a single consumer is allowed to see it.

    `mode` is the mode *actually applied*, which is not always the mode the policy asked for:
    `downgraded_from` is set when redaction failed and this consumer dropped to `digest`. The pair is
    what lets a consumer tell "there was no content" apart from "there was content I was not given",
    which a single mode field cannot express.

    `digests` and `lengths` describe the **raw** content, never the redacted view. That is the whole
    reason the digest is taken first: two consumers on different policies see different text but agree
    on the identity of what the provider was sent, so a digest can join a redacted record to a full
    one — and, later, key a replay.
    """

    receipt: ModelCallReceipt
    mode: CaptureMode = "none"
    downgraded_from: str = ""
    content: Mapping[str, Any] | None = None
    digests: Mapping[str, str] = field(default_factory=dict)
    lengths: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in CAPTURE_MODES:
            raise WireValidationError(f"capture mode must be one of: {', '.join(CAPTURE_MODES)}")
        object.__setattr__(self, "digests", dict(self.digests))
        object.__setattr__(self, "lengths", dict(self.lengths))
        if self.content is not None:
            object.__setattr__(self, "content", dict(self.content))

    @property
    def was_downgraded(self) -> bool:
        return self.downgraded_from != ""


@runtime_checkable
class ModelIOObserver(Protocol):
    """A consumer of settled model calls.

    One required member. `close` is declared separately by `ClosableModelIOObserver`, because a member
    with a default in a `Protocol` body reaches only classes that explicitly inherit it — for
    structural typing it is *required*, and declaring `close` here would reject every observer that
    does not define one. The pipeline probes for it with `getattr`.

    An observer must not raise, and if it does the kernel swallows it: an exporter that is down is not
    a reason to fail a model call the provider has already billed for. It must also treat the capture
    as read-only; each consumer gets its own capture, but the receipt inside is shared.
    """

    def on_model_call(self, capture: ModelCallCapture) -> None: ...


@runtime_checkable
class ClosableModelIOObserver(Protocol):
    """An observer that owns a resource to release. Opt-in; see `ModelIOObserver`."""

    def close(self) -> None: ...


@dataclass(frozen=True)
class ModelIOSubscription:
    """One observer and the policy that governs what it sees.

    The policy lives here rather than on the observer, so registering the same exporter twice under
    two policies is a normal thing to do — and so an observer cannot widen its own grant.
    """

    observer: ModelIOObserver
    policy: CapturePolicy = field(default_factory=lambda: CapturePolicy())


def _resolve_capture(policy: CapturePolicy, content: Mapping[str, Any]) -> tuple[CaptureMode, str, Mapping[str, Any] | None]:
    """Decide what one subscription sees, as `(mode, downgraded_from, content)`."""
    if policy.mode == "none":
        return "none", "", None
    if policy.mode == "digest":
        return "digest", "", None
    if policy.mode == "full":
        return "full", "", dict(content)
    redacted = redacted_or_none(
        dict(content), policy=policy.effective_redaction, redactor=policy.redactor
    )
    if redacted is None:
        # Fail closed. This consumer asked for redacted content and the redactor could not produce it,
        # so it gets what ``digest`` would have given it -- never the raw value.
        return "digest", "redacted", None
    return "redacted", "", dict(redacted)


def dispatch_model_call(
    *,
    receipt: ModelCallReceipt,
    content: Mapping[str, Any],
    subscriptions: Sequence[ModelIOSubscription],
) -> ModelCallReceipt:
    """Deliver one settled model call to every subscription under its own policy.

    Returns the receipt the observers were given: the same one, plus `capture_downgrades`. The count
    is resolved in a first pass *before* any delivery, so every observer sees the same receipt.
    Delivering as we go would hand the first observer a count of zero and the last the true total, and
    a receipt that disagrees with itself across consumers is worse than no count at all.

    Digests and lengths are computed once, on the raw content, and shared. Beyond the cost, that is
    what makes them comparable: a per-observer digest taken after redaction would differ by policy and
    could not join anything.

    An observer that raises is skipped and the rest still run. The call already happened and the
    provider has already been paid; a broken exporter does not get to undo that.
    """
    if not subscriptions:
        return receipt

    digests = {key: content_digest(value) for key, value in content.items()}
    lengths = {
        key: length for key, value in content.items() if (length := content_length(value)) is not None
    }

    resolved = [_resolve_capture(subscription.policy, content) for subscription in subscriptions]
    downgrades = sum(1 for _mode, downgraded_from, _payload in resolved if downgraded_from)
    settled = replace(receipt, capture_downgrades=receipt.capture_downgrades + downgrades)

    for subscription, (mode, downgraded_from, payload) in zip(subscriptions, resolved, strict=True):
        reveals_metadata = mode != "none"
        capture = ModelCallCapture(
            receipt=settled,
            mode=mode,
            downgraded_from=downgraded_from,
            content=payload,
            digests=digests if reveals_metadata else {},
            lengths=lengths if reveals_metadata else {},
        )
        try:
            subscription.observer.on_model_call(capture)
        except Exception:
            continue
    return settled


def close_model_io_subscriptions(subscriptions: Sequence[ModelIOSubscription]) -> None:
    """Release every observer that declared a `close`, tolerating failures.

    Probed with `getattr` rather than required, which is what keeps `close` off the base protocol.
    """
    for subscription in subscriptions:
        close = getattr(subscription.observer, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception:
            continue
