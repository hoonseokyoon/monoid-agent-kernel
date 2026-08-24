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

import copy
import re
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from functools import lru_cache
from types import UnionType
from typing import (
    Any,
    Literal,
    Protocol,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)

from pydantic import TypeAdapter, ValidationError

from monoid_agent_kernel._policy_util import dedupe
from monoid_agent_kernel.core.authority import WriteAuthorityRevoked
from monoid_agent_kernel.core._json_schema import END_OF_INPUT
from monoid_agent_kernel.core._util import canonical_hmac_sha256, canonical_sha256
from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.json_ingress import (
    MAX_PORTABLE_CONTAINER_DEPTH,
    exact_elements,
    exact_items,
    exact_text,
    normalize_json_ingress,
    normalize_unicode_scalars,
    portable_type_name,
)
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


# Keys ``RedactionPolicy.digest``. Minted once per process on first use and never exported: a
# receipt consumer is meant to compare digests, not reproduce them. Lazy creation keeps importing
# pure record validators deterministic inside Temporal's Workflow sandbox.
@lru_cache(maxsize=1)
def _digest_key() -> bytes:
    return secrets.token_bytes(32)


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


def _folded_key(text: str) -> str:
    """The form secret-key matching happens in: stripped, lowercased, hyphens as underscores.

    One function, used for both the *rule* and the *candidate*. Keeping two copies of "how a key is
    normalized" is what produced this bug twice: first the rule was not lowercased, so `("API_KEY",)`
    matched nothing; then the rule kept its hyphens, so `("api-key",)` matched nothing — including the
    literal key `api-key`, since the candidate had already become `api_key`. Both sides now fold here or
    neither does.
    """
    return normalize_unicode_scalars(text).strip().lower().replace("-", "_")


def _normalized_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if type(value) is not tuple or not all(type(item) is str for item in value):
        raise ValueError(f"{field_name} must be a tuple of strings")
    return tuple(normalize_unicode_scalars(item) for item in value)


def _string_rules(payload: Mapping[str, Any], key: str, *, reject_blank: bool) -> tuple[str, ...]:
    """A validated tuple of rule strings: absent or null give `()`, anything malformed raises.

    Element-wise, not just container-wise. `_policy_util.str_tuple` coerces with `str()`, so
    `{"patterns": [None]}` became the pattern `"None"` — a rule that matches the literal text "None" and
    nothing an operator intended — and the policy was accepted while captures went on being labelled
    `redacted`. For a security policy, a malformed rule has to be a load error, not a rule that silently
    does something else.
    """
    value = _absent_as_empty(payload, key)
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise WireValidationError(f"{key} must be an array of strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise WireValidationError(f"{key} must be an array of strings")
        if reject_blank and not item.strip():
            raise WireValidationError(f"empty {key} entry is not allowed")
        items.append(item)
    return tuple(items)


def _absent_as_empty(payload: Mapping[str, Any], key: str) -> Any:
    """`()` when `key` is absent or explicitly null, the raw value otherwise.

    Deliberately not `payload.get(key) or ()`, the idiom used elsewhere in this repo. That reads a
    *falsy wrong type* as an empty list, so `{"patterns": ""}` silently disabled text masking and still
    produced captures labelled `redacted` — the validator downstream never saw the bad value. Here the
    only shortcut is for genuinely absent or null; anything present reaches the type check.
    """
    value = payload.get(key, None)
    return () if value is None else value


def _optional_object(payload: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    """A present object, or `None` when absent/null. A falsy wrong type still raises.

    Same hazard as `_absent_as_empty`, one level up: `{"context": []}` read as "no context" would
    accept a corrupt audit record as an anonymous invocation and quietly drop its run and trace
    attribution.
    """
    value = payload.get(key, None)
    return None if value is None else require_object(value, key)


_MODEL_CONFIG_ADAPTER = TypeAdapter(ModelConfig)


def _parsed_model_config(payload: Mapping[str, Any] | None) -> ModelConfig:
    """`ModelConfig.from_json`, with malformed input turned into a wire-validation failure.

    `ModelConfig` and its nested `ReasoningConfig` / `ModelRetryConfig` are typed `dict | None` and
    trust it, and they validate nothing: a malformed nested object reaches `.get` on a list and raises
    `AttributeError`, while a bad *value* — `{"provider": []}`, an unsupported reasoning effort — is
    accepted outright and yields an object whose fields contradict their own `Literal` annotations.
    Checking the outer object catches neither.

    Both are handled without naming a single field, so this cannot go stale when `ModelConfig` grows
    one: exceptions are translated, and the parsed result is re-validated through a pydantic
    `TypeAdapter` over its serialized form — the same tool `core.wire_validation` already uses, applied
    recursively by the type's own annotations. Validating the *instance* would not work; pydantic
    trusts an already-constructed dataclass and skips it.

    What survives is `from_json`'s deliberate coercion: `{"model": []}` becomes `"[]"` and
    `{"retry": {"retry_on": "x"}}` becomes `("x",)`. Those produce correctly-typed values, so they are a
    question about that type's leniency rather than a validation gap here, and tightening them changes
    a public contract type used by agent runtime config.
    """
    try:
        parsed = ModelConfig.from_json(payload)
    except WireValidationError:
        raise
    # ``OverflowError`` is an ``ArithmeticError``, not a ``ValueError``, so it needs naming: JSON decodes
    # an oversized exponent such as ``1e999`` to ``inf``, and ``int(inf)`` raises it. A corrupt audit
    # record must not be able to crash a consumer that rejects corrupt audit records.
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise WireValidationError(f"model must be a valid model config: {exc}") from exc
    try:
        _MODEL_CONFIG_ADAPTER.validate_python(parsed.to_json())
    except ValidationError as exc:
        raise WireValidationError(
            f"model must be a valid model config: {exc.errors()[0]['msg']}"
        ) from exc
    return parsed


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
        secret_key_parts = _normalized_string_tuple(
            self.secret_key_parts,
            "redaction secret_key_parts",
        )
        patterns = _normalized_string_tuple(self.patterns, "redaction patterns")
        literals = _normalized_string_tuple(self.literals, "redaction literals")
        if type(self.replacement) is not str:
            raise ValueError("redaction replacement must be a string")
        replacement = normalize_unicode_scalars(self.replacement)
        for pattern in patterns:
            try:
                _compiled(pattern)
            except re.error as exc:
                raise ValueError(f"invalid redaction pattern {pattern!r}: {exc}") from exc
        # Normalized here, not only in ``from_json``. ``names_a_secret`` folds the *candidate* key to
        # lowercase, so an un-normalized rule can never match: ``secret_key_parts=("API_KEY",)`` was a
        # rule that silently matched nothing, and the value it was written to mask was delivered in a
        # ``redacted`` capture. The defaults are already lowercase, which is exactly what hid it.
        # Doing it in the constructor covers ``merged`` and every programmatic caller at once, and
        # makes two policies that differ only in rule case compare -- and digest -- equal.
        object.__setattr__(
            self,
            "secret_key_parts",
            dedupe(_folded_key(part) for part in secret_key_parts if part.strip()),
        )
        object.__setattr__(self, "patterns", patterns)
        object.__setattr__(self, "literals", literals)
        object.__setattr__(self, "replacement", replacement)

    @classmethod
    def from_json(cls, payload: Any) -> RedactionPolicy:
        if payload is None:
            return cls()
        payload = require_object(payload, "redaction_policy")
        secret_key_parts = (
            DEFAULT_SECRET_KEY_PARTS
            if payload.get("secret_key_parts") is None
            else _string_rules(payload, "secret_key_parts", reject_blank=False)
        )
        return cls(
            # Folding and de-duplication happen in ``__post_init__``, so a policy built in code gets
            # exactly what a policy loaded from JSON gets.
            secret_key_parts=secret_key_parts,
            patterns=_string_rules(payload, "patterns", reject_blank=True),
            literals=_string_rules(payload, "literals", reject_blank=True),
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
        """An id for this policy, for a receipt to record *which* rules were applied.

        The receipt records the digest rather than the policy because the policy itself can name
        secrets — a `literals` entry is a secret by construction.

        Keyed, under a key minted once per process. An unkeyed digest would have defeated the very
        purpose above: the observer that receives `redaction_digest` also knows the canonical JSON
        shape, so it could hash candidate literals until one matched, and a `literals` entry is
        exactly the kind of low-entropy secret — a PIN, a password, a tenant token — that a
        ten-thousand-guess search recovers instantly. Keying removes the oracle without weakening
        what a receipt consumer actually needs, which is to tell two policies apart within a run.

        The cost is that the digest identifies a policy *within one process*, not across processes
        or restarts. That is not a property that can be recovered by choosing a better hash: any
        digest that both is reproducible by an outsider and distinguishes policies by their secret
        values is a guessing oracle for those values. A deployment that needs cross-process joins
        needs a shared key, which is a deployment-time secret rather than a kernel default.
        """
        return canonical_hmac_sha256(self.to_json(), _digest_key())

    def names_a_secret(self, key: str) -> bool:
        """Whether `key` names a secret under this policy.

        Folds the candidate through `_folded_key`, the same function `__post_init__` folds the rules
        through — the two must agree, and they have twice not.
        """
        folded = _folded_key(key)
        return any(part in folded for part in self.secret_key_parts)

    def redact_text(self, text: str) -> str:
        """Apply `literals` then `patterns` to free text.

        The order is fixed and the tuples are ordered, which is what makes redaction deterministic:
        the same text under the same policy always yields the same output, so a digest taken over a
        redacted view is comparable across processes.
        """
        if type(text) is not str:
            raise ValueError("redaction text must be a string")
        text = normalize_unicode_scalars(text)
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
            # `exact_text` and `exact_items`, not `str(key)` and `value.items()`: the
            # key answering `lower()` decides whether this field is a secret, and the
            # mapping answering `items()` decides which fields are judged at all.
            return {
                exact_text(key): (
                    policy.replacement
                    if policy.names_a_secret(exact_text(key))
                    else self.redact(item, policy=policy)
                )
                for key, item in exact_items(value)
            }
        # `str` is a Sequence, and bytes have no text semantics we may assume, so both are excluded.
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self.redact(item, policy=policy) for item in exact_elements(value)]
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
    # ``is None``, not truthiness. A redactor is an arbitrary object, and one backed by a rule set that
    # defines ``__len__`` is falsy when it holds no rules -- which is precisely when substituting the
    # weaker built-in rules is least acceptable, and it would be reported as a successful redaction.
    applied = DefaultRedactor() if redactor is None else redactor
    try:
        return applied.redact(value, policy=policy)
    except Exception:
        return None


# The one ceiling the digest gate and the payload-recording gate share: whatever gets a key gets
# recorded, and what exceeds it is refused whole -- never truncated. Decimal, agreeing with
# ``AgentRunSpec.max_message_log_bytes`` so the two numbers cannot drift, but that knob is a
# different owner's and measures a different thing: it sums a run's ``messages``, while this
# bounds one call's whole identity payload -- system prompt, tool definitions, instruction and
# observations included. So a request can still pass every run limit and exceed this; what the
# shared number buys is that such a call is now a NAMED condition (``too_large``) rather than an
# unexplained ``absent``, not that the case is gone. Raising this only turns refusals into keys;
# lowering it orphans every corpus recorded above the new value, so it moves up or not at all.
MAX_MODEL_PAYLOAD_BYTES = 8_000_000

DIGEST_STATUSES = ("not_reached", "ok", "absent", "withheld", "too_large")
"""Why ``ModelCallReceipt.request_digest`` holds what it holds.

``absent`` means no key was issued because canonical JSON could not carry the payload -- a defect
in the payload. ``too_large`` means no key was issued because the payload exceeded
:data:`MAX_MODEL_PAYLOAD_BYTES` -- an operational condition rather than a defect, though not one an
operator can configure their way out of: that constant is a build-time value, and it bounds the
whole identity payload rather than the message log the run limits bound, so a request can pass
every limit and still be refused a key. The two were one value, and a consumer holding a keyless
record could not tell a payload to file a bug about from one to make smaller. ``withheld`` means a key was issued and a ``none``-mode policy
removed it. ``not_reached`` means the call was refused before a key was computed at all.
"""

DESTINATION_STATUSES = (
    "not_reached",
    "not_declared",
    "declined",
    "resolved",
    "unavailable",
)
"""Which of the destination probe's outcomes happened.

``not_declared`` is an adapter that routes on config alone and never offered the member;
``declined`` is one that offered it and answered with nothing; ``unavailable`` is one whose probe
raised. That last is not a transient condition to shrug at: the shipped gateway resolver raises
deterministically when no URL is configured anywhere, so ``unavailable`` is usually a deployment
whose every call is about to fail. All three used to be the same empty string, and the collapse was
invisible because each produced a key that looked fine.
"""


_IDEMPOTENCY_KEY_BODY = r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}"
"""What an idempotency key may be spelled as, once, so its two enforcers cannot drift.

The rule lives in ``core`` rather than beside its first caller because it has two enforcers on
opposite sides of an import boundary that cannot be crossed the other way: ``core/schemas.py``
states it to ``monoid validate``, and ``providers/base.py`` states it to a request being built.
``core`` never imports ``providers``, so a rule owned there could only have been copied -- and a
hand-copied twin regex is a drift waiting to happen. Both forms below derive from this body.

Bounded at 128 characters and free of control characters because this is the one field on a model
call that reaches a *transport header* rather than a JSON string: JSON escapes a control
character, an HTTP header does not, and neither ``http.client`` nor ``httpx`` refuses an obsolete
folded value.
"""

IDEMPOTENCY_KEY_PATTERN = re.compile(rf"{_IDEMPOTENCY_KEY_BODY}\Z", re.ASCII)
"""The Python form, for validating a value in hand."""

IDEMPOTENCY_KEY_JSON_PATTERN = rf"^(|{_IDEMPOTENCY_KEY_BODY}){END_OF_INPUT}"
"""The ECMA-262 form for JSON Schema, empty-allowed the way ``prompt_digest``'s pattern is.

Empty is a legal recorded value -- a refused call was never keyed -- so the ledger admits it
explicitly rather than by omitting the constraint, exactly as the optional-digest pattern does for
a digest that may not have been issued. ``\\Z`` and ``re.ASCII`` do not exist in ECMA-262, which is
why this is derived from the body rather than from the compiled pattern's source -- and why the
end of input is asserted by :data:`~monoid_agent_kernel.core._json_schema.END_OF_INPUT` rather than
by a bare ``$``, which under ``jsonschema``'s Python engine would also have matched just before a
trailing newline.
"""


def is_valid_idempotency_key(value: Any) -> bool:
    """Whether ``value`` may be presented as an idempotency key on a header or written to a log.

    Empty is *not* valid here: absence is spelled by not calling this, and each caller says what
    absence means for it. Bounded so an unauthenticated client cannot choose the length of a log
    line.
    """

    return isinstance(value, str) and IDEMPOTENCY_KEY_PATTERN.fullmatch(value) is not None


RECORDED_DIGEST_BODY = r"[0-9a-f]{64}"
"""What a recorded content digest may be spelled as, once, so its enforcers cannot drift.

Same argument as ``_IDEMPOTENCY_KEY_BODY`` above, one field family over: ``core/schemas.py``
states this rule to ``monoid validate`` (both of its digest pattern forms compose this body),
and ``model_call_record`` states it to a receipt being minted into a ledger line (W7-4). The
producers already hold the shape by construction -- every digest here is hex SHA-256 output --
so the spelling exists for values a producer did not mint: a receipt loaded from foreign JSON,
whose reader deliberately transports what it was given. (``model_payloads.is_chunk_sha256``
states its own 64-hex rule on purpose and is not a projection of this one: a chunk reference
becomes a *filename*, so that check is a path-safety boundary every reader re-establishes,
with its own reasons written beside it.)
"""

RECORDED_DIGEST_PATTERN = re.compile(rf"{RECORDED_DIGEST_BODY}\Z", re.ASCII)
"""The Python form, for validating a value in hand."""


def is_recorded_digest(value: Any) -> bool:
    """Whether ``value`` is a digest in hand: exactly 64 lowercase hex characters.

    Empty is *not* valid here, the same line ``is_valid_idempotency_key`` draws: absence is
    the in-band empty string on every optional-digest field, a status field says why, and
    each caller states what absence means where it stands -- the ledger's mint guard admits
    it (a refused call's line is empty and explained), a caller comparing two digests in
    hand does not.
    """

    return isinstance(value, str) and RECORDED_DIGEST_PATTERN.fullmatch(value) is not None


def is_absent_or_valid(value: Any, is_valid: Callable[[Any], bool]) -> bool:
    """Whether ``value`` is this repo's in-band absence -- the empty string -- or something
    ``is_valid`` certifies. Judged without consulting the object's own opinion of itself.

    Type before value, the rule ``_validate_counts`` states one field family over: a
    comparison asks the *value*, and a value may answer. A guard spelling the absence arm as
    ``value != ""`` is False for a ``str`` subclass whose ``__ne__`` returns False, so the
    pattern check behind it never runs -- while ``json.dumps`` and an HTTP header both go on
    reading the underlying string, which is exactly the value the guard existed to refuse.
    Requiring the exact ``str`` type first makes the emptiness comparison and the pattern the
    string's own answers, and refuses every non-string in the same breath.

    Both boundaries that admit an empty spelling ask through here -- the ledger mint in
    ``core/model_calls.py`` and request ingress in ``providers/base.py`` -- so the rule cannot
    hold at one and drift at the other. The positive-gate callers (``providers/gateway.py``,
    the reference gateway's header reader) need nothing: asking ``is_valid`` directly never
    consults equality, and they omit what they cannot certify rather than raising.
    """

    return type(value) is str and (value == "" or is_valid(value))


def destination_digest(value: str) -> str:
    """An id for a call's destination, for a receipt to record *where* without recording *what*.

    Keyed, under the same per-process key and for the same reason as :attr:`RedactionPolicy.digest`
    -- read that docstring first, because the argument is identical and this preimage is weaker. A
    hostname is drawn from a far smaller space than a redaction literal: an unkeyed digest of one is
    a confirm-a-guess oracle for anyone holding a candidate list, which is the disclosure the
    "hashed, never recorded" rule exists to prevent. Domain separation does not help; guessing is
    not a collision.

    The cost is the same cost: this identifies a destination **within one process**, not across
    restarts. That is what a live receipt consumer needs -- "are these two calls going to different
    places" -- and it is not what a durable corpus needs. A record that must be joined after a
    restart needs a deployment-supplied key, which is a deployment-time secret rather than a kernel
    default, and choosing that is deliberately left to whoever first persists these receipts.
    """

    return canonical_hmac_sha256({"destination": value}, _digest_key()) if value else ""


def _validated_status(value: Any, key: str, allowed: tuple[str, ...]) -> str:
    """Refuse a closed kernel enum's non-member, wherever the value entered.

    One function for the constructor and the reader, because they used to disagree: `from_json`
    refused a non-member while `to_json` happily emitted one, so ``ModelCallReceipt(
    digest_status="okay")`` wrote an audit record this same class rejects on the way back in. A
    record that can be written and not read fails in the consumer, long after the writer that
    caused it is gone -- and the writer is the only place the mistake was ever fixable.
    """

    if not isinstance(value, str) or value not in allowed:
        raise WireValidationError(f"model call {key} must be one of {allowed}")
    return value


def _parsed_status(
    payload: Mapping[str, Any],
    key: str,
    allowed: tuple[str, ...],
    *,
    witness: str,
    witnessed: str,
) -> str:
    """Read a closed kernel enum: unknown is refused, and a *missing key* reads as the default
    unless the field it describes says otherwise.

    ``witness`` names the digest this status explains, and ``witnessed`` the one outcome that could
    have produced a non-empty one. Both defaults are ``not_reached`` -- "the call was refused before
    we got that far" -- which is the right reading of silence on a receipt written before these
    fields existed, and the wrong reading of silence on one that carries the value. That record
    would deny its own contents, ``to_json`` writes the denial back, and the first read/write makes
    it permanent: a consumer asking the status whether a replay key exists would discard a real one.

    Inferred from silence only. A payload that *states* a status keeps it verbatim even where it
    contradicts its digest, because that combination is a bug in whatever wrote it and quietly
    repairing it here would hide the writer that has one.

    Silence is a key that is not there -- ``key not in payload``, not ``payload.get(key) is None``.
    A key present and holding ``null`` is a corrupt record, and every other string on this receipt
    already refuses one (``parse_str`` separates missing from present-and-mistyped; ``http_status``
    is nullable only because it is declared ``int | None``). Conflating the two was harmless while
    both landed on the default and stopped being harmless the moment absence began to infer: it
    would have handed a malformed payload a status it never carried.
    """

    if key not in payload:
        return witnessed if parse_str(payload, witness) else allowed[0]
    return _validated_status(payload[key], key, allowed)


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

    `to_json` carries `mode` and `redaction`. A `redactor` is live code and cannot round-trip, so
    `restored_without_redactor` marks a policy that came back from JSON knowing it *had* one. Such a
    policy does **not** fall back to `DefaultRedactor`: the capture pipeline treats it as a redaction
    failure and downgrades to `digest` until the redactor is re-attached. Applying the built-in rules
    instead would be the worst outcome available — the consumer would be told it received redacted
    content while the classifier that masked more than key names and regexes is simply absent.
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
    def effective_redactor(self) -> Redactor | None:
        """The redactor to apply, or `None` when redaction cannot be performed at all.

        `None` for a policy restored from JSON that knows it had a custom redactor. There is no correct
        redactor to hand back in that case, and handing back `DefaultRedactor` is the disclosure the
        marker exists to prevent — an external consumer reading this accessor would apply the built-in
        rules and believe it had honoured the policy, while the pipeline downgraded the very same policy
        to `digest`. One answer, not two.
        """
        if self.redactor is not None:
            return self.redactor
        return None if self.restored_without_redactor else DefaultRedactor()

    @classmethod
    def from_json(cls, payload: Any) -> CapturePolicy:
        if payload is None:
            return cls()
        payload = require_object(payload, "capture_policy")
        # Absent and explicit ``null`` both mean "no policy of its own", which resolves to the default
        # via ``effective_redaction``. Anything else present is parsed, so a malformed value -- falsy
        # ones included -- still raises rather than silently becoming the default.
        redaction_payload = _optional_object(payload, "redaction")
        return cls(
            mode=parse_literal(payload, "mode", CAPTURE_MODES, default="full"),
            redaction=(
                RedactionPolicy.from_json(redaction_payload)
                if redaction_payload is not None
                else None
            ),
            restored_without_redactor=payload.get("redactor") is not None,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "redaction": self.redaction.to_json() if self.redaction is not None else None,
            # Names the fact that a redactor was attached without claiming to carry it, so a
            # round-trip cannot quietly turn custom redaction into the default. Set when the marker is
            # already set too, not only when a redactor is attached: a restored policy has
            # ``redactor is None``, so keying on that alone wrote ``null`` and the *second* hop cleared
            # the marker and fell back to the built-in rules. A policy that crosses two services --
            # config store to gateway to kernel -- is one hop, not zero.
            "redactor": (
                "custom" if (self.redactor is not None or self.restored_without_redactor) else None
            ),
        }


def content_digest(value: Any) -> str:
    """A stable digest of one captured content field, computed on the raw value.

    Everything is hashed as canonical JSON under a key that names its shape, so a digest does not
    depend on mapping order and two differently-shaped values cannot collide. The shape key is the
    domain separator and it is why text does *not* simply hash its own UTF-8 bytes: a text field whose
    content happened to equal the wrapper serialization of a structured value — `'{"value":["x"]}'` —
    hashed identically to that value, which is the collision the wrapper was supposed to prevent.

    Consequence for anything that records these: recompute with this function, not with a bare
    `sha256sum` of the text. **This is now frozen.** It was free to change while nothing persisted a
    digest; `recorder.settled_text` writes one into every `settled_text` record in
    `transcript.jsonl`, `_validate_settled_text_digests` verifies it, and `monoid.transcript.v1` is
    registered in the compatibility ledger precisely because the digest became the durable join key
    that a settle event's `final_text_digest` resolves against. Changing this function now
    invalidates every transcript already on disk.
    """
    if isinstance(value, str):
        return canonical_sha256({"text": value})
    return canonical_sha256({"value": _jsonish(value)})


def content_length(value: Any) -> int | None:
    """Character length for text, `None` for anything else.

    Deliberately not "length of the canonical JSON": for a structured payload that number measures
    the serialization, not the content, and an operator reading it as a content size would be wrong.
    A missing answer is better than a misleading one.
    """
    return len(value) if isinstance(value, str) else None


# Bound on the structural depth this will normalize. Reached from `content_digest`, which
# `dispatch_model_call` computes over a turn's raw tool-call arguments -- model-controlled, and
# `json.loads` will hand us up to ~994 levels. Unbounded, this raised `RecursionError` from ~495,
# and `RecursionError` is a `RuntimeError`, not a `ValueError`, so it escapes the handlers that would
# otherwise turn a bad payload into a failed call. `core.tool_approval` has the same-named function
# with the same shape; bounding one and not the other is precisely the twin-miss this release keeps
# finding, and this side fires *earlier* -- during the model-call publish, before tool dispatch.
# Imported rather than spelled again -- three sites bound the same model-authored
# argument, and the shared constant is what keeps them from drifting apart. This site
# ELIDES where the other two raise, which is deliberate and explained below: a marker is
# what a digest wants, and it is also what makes a depth cap dangerous on a cyclic input,
# so the cycle guard below is not optional here the way it is where the bound raises.
MAX_JSONISH_DEPTH = MAX_PORTABLE_CONTAINER_DEPTH

# What replaces a subtree past the bound. A marker rather than a raise: this runs inside digest and
# observer publication, where the caller wants an identifier for the payload, not an exception.
# Distinct and constant, so two payloads differing only below the bound still digest identically --
# which is honest, because at that depth the digest genuinely cannot distinguish them.
_DEPTH_ELIDED = "[depth-elided]"

# What replaces a container reachable from itself. Distinct from ``_DEPTH_ELIDED`` so a digest can
# tell "too deep to distinguish" from "cyclic", and so the marker says which bound was hit.
_CIRCULAR_ELIDED = "[circular-elided]"


def _jsonish(value: Any, _depth: int = 0, _ancestors: frozenset[int] = frozenset()) -> Any:
    if _depth > MAX_JSONISH_DEPTH:
        return _DEPTH_ELIDED
    if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
        # The depth bound alone made this *worse*, not better. A cyclic mapping with two
        # self-referencing keys is re-expanded once per edge per level, so the bound turned a fast
        # ``RecursionError`` into ~2**64 nodes -- a hang, before the completed provider turn is
        # returned. Reachable from a third-party adapter handing back a cyclic tool-call argument,
        # which is the same reachability class as ``public_view.preview_value``'s guard.
        #
        # Third traversal in this tree to need this. ``preview_value`` and
        # ``public_view.touches_redacted_path`` were both guarded during review; this one was not,
        # which is the release's own defect shape one more time.
        #
        # Its two siblings are clean, and the reason is worth knowing before touching either.
        # ``core.tool_approval._jsonish`` *raises* past its bound and ``conformance.contracts``
        # has no bound at all, so both abort the whole traversal on the first over-deep path —
        # measured at 0.002 s for the same cyclic input that hung this one. **The elision is what
        # makes a depth cap dangerous**: returning a marker lets every sibling branch keep
        # expanding. Converting either of those to return a marker instead of raising reintroduces
        # this defect there, and would need this guard first.
        if id(value) in _ancestors:
            return _CIRCULAR_ELIDED
        _ancestors = _ancestors | {id(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonish(item, _depth + 1, _ancestors) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonish(item, _depth + 1, _ancestors) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _annotation_admits(annotation: Any, wanted: Any) -> bool:
    """Whether ``annotation`` names ``wanted`` outright or as a union member.

    A parameterized container is not a match: ``Mapping[str, int]`` has ``int`` among its args and
    is not an integer field. Only a bare annotation or a union of them counts, which is what keeps
    the two censuses below off ``usage`` -- whose *values* are governed by their own loop.
    """

    if annotation is wanted:
        return True
    return get_origin(annotation) in (Union, UnionType) and wanted in get_args(annotation)


@lru_cache(maxsize=None)
def _field_names_admitting(cls: type, wanted: Any) -> frozenset[str]:
    """The record's own fields whose declared type admits ``wanted``.

    Derived from the annotations rather than restated beside them, for the reason the wire-key
    tuple below already gives: a field added tomorrow joins the rule the day it exists, not the
    day someone remembers a list. Cached because both censuses run on every construction and
    every read of these records.
    """

    hints = get_type_hints(cls)
    return frozenset(
        entry.name
        for entry in dataclass_fields(cls)
        if _annotation_admits(hints.get(entry.name), wanted)
    )


def _validate_counts(record: Any, label: str) -> None:
    """Every count on ``record`` is a real integer -- one predicate, applied by enumeration.

    ``bool`` is an ``int`` subclass, so a count validated by comparison alone admits ``True``:
    ``True < 1`` is ``False``, so a bounds check passes it, ``True == 1`` satisfies the receipt's
    ordered-index invariant, and ``to_json`` then emits a JSON boolean where
    ``MODEL_CALLS_RECORD_SCHEMA`` requires an integer -- a record this kernel writes and its own
    schema refuses to read back.

    The rule itself is not new here. The usage loops on both records already spell
    ``type(value) is not int`` and explain why; it had simply been applied to the mapping values
    and to none of the scalar counts standing beside them. Stated once and enumerated so the two
    records cannot drift apart again, which is exactly how they got here.

    A field declared nullable may hold ``None``; that too is read off the annotation.
    """

    nullable = _field_names_admitting(type(record), type(None))
    for name in sorted(_field_names_admitting(type(record), int)):
        value = getattr(record, name)
        if value is None and name in nullable:
            continue
        # ``isinstance`` minus ``bool``, NOT ``type(value) is int``. The first spelling of this
        # rule was the stricter one and it broke a documented acceptance: ``with_error`` reads
        # ``http_status`` off an arbitrary exception under
        # ``isinstance(http_status, bool) or not isinstance(http_status, int)`` -- excluding
        # ``bool`` by name while admitting every other ``int`` subclass, because an
        # ``http.HTTPStatus`` is what an HTTP client hands back -- and the ``replace()`` inside
        # it then re-ran this check and refused the value its own reader had just accepted.
        #
        # Narrowed to the defect this census was written for rather than exempting the one field
        # that was noticed: the finding was ``bool``, and only ``bool``. A per-field exemption
        # would have left the same over-reach standing on the other six counts.
        #
        # ``usage`` keeps the stricter ``type(value) is not int`` in its own loop, deliberately
        # and for a reason that does not apply here: its four sibling readers spell the same, so
        # an ``IntEnum`` token count accepted there would be dropped by every one of them. These
        # scalars have no such readers -- they are emitted straight to JSON, where an ``IntEnum``
        # serializes as the integer it is.
        if isinstance(value, bool) or not isinstance(value, int):
            raise WireValidationError(f"{label} {name} must be an integer")


def _refuse_null_wire_values(payload: Mapping[str, Any], cls: type, label: str) -> None:
    """A key present and holding ``null`` is a corrupt record, not an absent one.

    Absence on these records means one thing -- a writer that predates the field -- and the
    defaults reconstruct what that writer meant. ``null`` means a writer that had the field and
    wrote nothing into it, which no writer here has ever done: ``to_json`` emits an object for
    every one of them. Collapsing the two put an explicit null onto the field's legacy default,
    and the defaults are load-bearing: an empty ``usage`` still satisfies the receipt's
    cross-entry sum invariant, and an empty ``attempt_log`` reads as "no ledger was ever written"
    rather than as the erasure it is.

    Enumerated over the record's own fields, minus the ones declared nullable, so this cannot be
    the rule for the fields someone remembered. The per-field required-key pin is a different
    question and remains one: it asks whether the key is present, and a key holding null is.
    """

    nullable = _field_names_admitting(cls, type(None))
    required = frozenset(entry.name for entry in dataclass_fields(cls)) - nullable
    nulls = sorted(key for key in required if key in payload and payload[key] is None)
    if nulls:
        raise WireValidationError(f"{label} fields must not be null: {', '.join(nulls)}")


@dataclass(frozen=True)
class ModelCallAttempt:
    """One kernel dispatch inside a settled call, as the receipt's log records it.

    Same rule as the receipt that carries it: metadata only — taxonomy, counts, timings — so the
    log is as safe to hand to a ``none``-mode observer as the receipt is. Success is spelled the
    way the receipt spells it: an empty ``error_code``. There is no separate outcome enum, because
    the receipt's own convention already answers the question and a second vocabulary for the same
    fact is a divergence waiting to be recorded.

    No wall-clock instant, deliberately — the receipt carries ``latency_ms`` and no instant, and
    the ledger line's ``recorded_at`` is the anchor for the whole call (see
    ``core/model_calls.py:model_call_record``). ``elapsed_ms`` covers the dispatch only;
    ``backoff_ms`` is the measured wait the kernel imposed *between* this dispatch and the one
    before it, so the first entry has none to report and a receipt refuses any other value there
    (``None`` is the field's absence -- a record predating it -- and stays legal). Every
    duration here is the floor of the same
    monotonic clock, and floors sum to at most the floor of the sum, so
    ``sum(elapsed_ms) + sum(backoff_ms) <= latency_ms`` exactly — the remainder is the keying
    and settle overhead that falls outside the dispatch loop.

    ``provider_retried`` here is what the adapter reported through the progress channel *during
    this attempt's dispatch*, plus what this attempt's own outcome object declared. The receipt's
    flag additionally folds whole-call evidence, so the entry can read ``False`` where the receipt
    reads ``True`` — the entry is the per-attempt attribution, the receipt is the call's.

    ``stream_committed`` is whether a streamed chunk had been delivered when this attempt settled.
    Delivery closes the retry window, so it can only be ``True`` on the final entry.
    """

    index: int = 1
    elapsed_ms: int = 0
    error_code: str = ""
    provider_error_code: str = ""
    retryable: bool = False
    config_recoverable: bool = False
    http_status: int | None = None
    provider_retried: bool = False
    usage: Mapping[str, int] = field(default_factory=dict)
    stream_committed: bool = False
    # The measured wait between this dispatch and the one before it (W7-2) -- so on the first
    # entry there is nothing to measure and the receipt refuses anything but 0. A duration and
    # never an instant -- the entry's own timing rule -- and measured around the wait rather
    # than copied from the schedule, so a capped sleep records what happened. ``None`` means the
    # record predates the field, which is why ``to_json`` omits the key instead of inventing a
    # null no writer ever wrote or a 0 that claims a measurement never taken. Appended last
    # under the positional-stability rule the receipt states.
    backoff_ms: int | None = None

    def __post_init__(self) -> None:
        # Type before bounds: a comparison cannot tell an integer from a bool, so the bounds
        # checks below are only meaningful once every count is known to be one.
        _validate_counts(self, "model call attempt")
        if self.index < 1:
            raise ValueError("model call attempt index must be positive")
        if self.elapsed_ms < 0:
            raise ValueError("model call attempt elapsed_ms must not be negative")
        if self.backoff_ms is not None and self.backoff_ms < 0:
            raise ValueError("model call attempt backoff_ms must not be negative")
        # The same usage rule the receipt enforces, spelled the same way: a log whose entries
        # admitted what the receipt refuses could not honor the sum invariant the runner pins
        # (entry usage totals equal the receipt's usage on either settle exit).
        for key, value in self.usage.items():
            if not isinstance(key, str) or type(value) is not int:
                raise WireValidationError("model call attempt usage must be a mapping of str to int")
            if value < 0:
                raise WireValidationError(
                    f"model call attempt usage {key!r} must not be negative"
                )
        object.__setattr__(self, "usage", dict(self.usage))

    @property
    def succeeded(self) -> bool:
        """Whether this dispatch produced the turn — the receipt's own convention."""
        return self.error_code == ""

    def to_json(self) -> dict[str, Any]:
        payload = {
            "index": self.index,
            "elapsed_ms": self.elapsed_ms,
            "error_code": self.error_code,
            "provider_error_code": self.provider_error_code,
            "retryable": self.retryable,
            "config_recoverable": self.config_recoverable,
            "http_status": self.http_status,
            "provider_retried": self.provider_retried,
            "usage": dict(self.usage),
            "stream_committed": self.stream_committed,
        }
        # Omitted, not nulled: absence is the wire spelling of "written before the field
        # existed", and it must survive a round trip -- a legacy line re-serialized with an
        # unconditional key would refuse itself on the next read.
        if self.backoff_ms is not None:
            payload["backoff_ms"] = self.backoff_ms
        return payload

    @classmethod
    def from_json(cls, payload: Any) -> ModelCallAttempt:
        payload = require_object(payload, "model_call_attempt")
        # An entry is read whole or refused, which is not the rule one level up and must not be.
        # ``attempt_log`` itself is optional because its absence means exactly one thing -- a
        # writer that predates the field -- and defaults there reconstruct what that writer meant.
        # The keys an entry was BORN with have no predecessor to be lenient toward: they were
        # written by a writer that knew all of them or the entry was not written by this record
        # at all. Defaulting them turned `{}` into a successful, zero-duration, unbilled dispatch
        # numbered 1, which then satisfied both of the receipt's cross-entry invariants -- a
        # corrupt audit line reading as data, the one outcome an audit surface may not produce.
        # The ledger schema has required every one of them since the field shipped; this is the
        # reader agreeing with it. A key added AFTER the entry shipped is the other generation:
        # it does have predecessors -- every line the earlier writer filled -- so it follows the
        # record-level absence rule instead, named per key in ``_ATTEMPT_OPTIONAL_WIRE_KEYS``.
        missing = [
            name
            for name in _ATTEMPT_WIRE_KEYS
            if name not in payload and name not in _ATTEMPT_OPTIONAL_WIRE_KEYS
        ]
        if missing:
            raise WireValidationError(
                "model call attempt is missing required fields: " + ", ".join(missing)
            )
        # Present-and-null is the other half of the same question, and the pin above cannot ask
        # it: a key holding ``null`` satisfies ``name in payload``.
        _refuse_null_wire_values(payload, cls, "model call attempt")
        raw_status = payload.get("http_status")
        return cls(
            index=parse_int(payload, "index", default=1),
            elapsed_ms=parse_int(payload, "elapsed_ms"),
            error_code=parse_str(payload, "error_code"),
            provider_error_code=parse_str(payload, "provider_error_code"),
            retryable=parse_bool(payload, "retryable"),
            config_recoverable=parse_bool(payload, "config_recoverable"),
            http_status=None if raw_status is None else parse_int(payload, "http_status"),
            provider_retried=parse_bool(payload, "provider_retried"),
            usage=_optional_object(payload, "usage") or {},
            stream_committed=parse_bool(payload, "stream_committed"),
            # Absent means the line predates the field; present must be an integer. ``null`` is
            # neither generation -- no writer omits by writing null -- and ``parse_int`` refuses
            # it, which keeps this reader saying what the schema says (``"type": "integer"``)
            # instead of re-opening the reader-lenient/schema-strict split one field over.
            backoff_ms=(
                None if "backoff_ms" not in payload else parse_int(payload, "backoff_ms")
            ),
        )


# Derived from the record, not restated beside it: a field added to ``ModelCallAttempt`` becomes
# required on the wire the moment it exists, rather than the moment someone remembers a list.
_ATTEMPT_WIRE_KEYS: tuple[str, ...] = tuple(
    entry.name for entry in dataclass_fields(ModelCallAttempt)
)

# The exemption is a policy with a reason, not a forgotten field: a key added after the entry
# shipped (``backoff_ms``, W7-2) has predecessors -- every line the earlier writer filled -- and
# requiring it would refuse ledgers this same package wrote. Absence of a key in this set reads
# as "written before the field existed", the record-level rule applied per key; everything not
# named here stays required the moment it exists, which is the derivation above doing its job.
_ATTEMPT_OPTIONAL_WIRE_KEYS: frozenset[str] = frozenset({"backoff_ms"})


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

    `attempts` may be **0**: a run whose cancellation or deadline was already past when the call was
    requested is refused before the adapter is reached, and a receipt is still written because a
    refused call is part of the audit trail. It used to carry the default 1, which told a consumer
    summing `attempts` that provider work happened when none did. 0 means exactly that — no adapter
    call was made. A failure *while* reaching into the adapter still counts as 1: the kernel did begin
    the call there.

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
    # The sixth fact the exception carries. ``retryable`` says "waiting may help";
    # ``config_recoverable`` says "changing configuration will". A receipt that recorded only the
    # first could not tell an auditor why an exhausted retry budget was never going to succeed.
    config_recoverable: bool = False
    http_status: int | None = None
    redaction_digest: str = ""
    capture_downgrades: int = 0
    # What the replay key was taken under, and whether one was issued at all. An empty
    # ``request_digest`` used to be the answer to four different questions -- the payload could not
    # be canonically encoded, it exceeded the size cap, the call was refused before a key was ever
    # computed, or a ``none``-mode policy withheld it -- and nothing downstream could tell them
    # apart. A consumer holding a keyless record could not say whether it was looking at a defect
    # or at a policy.
    digest_generation: str = ""
    digest_status: str = "not_reached"
    # Where the call was going, as a fact beside the key rather than inside it. The endpoint is
    # deliberately never recorded in plaintext, which is exactly why hashing it into the replay key
    # made that key unreproducible: nothing a record holds could reconstruct the preimage. The
    # status names which of the four probe outcomes happened; the digest is keyed (see
    # :func:`destination_digest`) so that comparing two calls stays possible without turning the
    # receipt into a guessing oracle for an internal hostname.
    destination_status: str = "not_reached"
    destination_digest: str = ""
    # One entry per kernel dispatch, in order. Empty on receipts written before the field
    # existed, on refused calls (``attempts == 0``), and on any receipt a caller builds
    # without one -- the empty arm is not reserved for zero dispatches, which is why neither
    # wire reader may treat an empty log beside a positive ``attempts`` as impossible.
    # Otherwise one entry per attempt, so the log is either absent or complete. Both halves of "complete" are enforced below rather
    # than left to the writer: the indices are ``1..attempts`` in order (a log naming some
    # attempts twice and others not at all could not answer the question it exists for, and
    # counting alone cannot tell the two apart), and the entries' usage sums to this receipt's
    # (a breakdown that disagrees with its total leaves a reader nothing to believe). Appended
    # last so positional construction predating this field keeps meaning what it meant.
    attempt_log: tuple[ModelCallAttempt, ...] = ()
    # The retry-scope token the call was keyed with -- issued by the runner in the same block
    # that computes the digests, once per call and before the first dispatch, so it is constant
    # across kernel re-dispatches and adapter-internal retries and reissued on resume. Recorded
    # as ISSUED, not as sent: only the gateway transport presents it on the wire, so a key on a
    # fake or replay call's receipt says the call was keyed, nothing more. Deliberately outside
    # the replay key: two identical requests share a replay slot precisely because content
    # cannot tell them apart, and a token meant to separate their provider work must therefore
    # be content-independent. Empty means the call never reached the keying block (refused by
    # the cancel/deadline check or by ingress normalization) or the record predates the field.
    # Appended after ``attempt_log`` under the same positional-stability rule it states.
    idempotency_key: str = ""

    def __post_init__(self) -> None:
        # Type before bounds, for the reason the attempt record gives: ``True < 0`` is ``False``,
        # so a comparison admits the bool it exists to bound.
        _validate_counts(self, "model call")
        if self.attempts < 0:
            raise ValueError("model call attempts must not be negative")
        if type(self.attempt_log) is not tuple:
            object.__setattr__(self, "attempt_log", tuple(self.attempt_log))
        for entry in self.attempt_log:
            if not isinstance(entry, ModelCallAttempt):
                raise WireValidationError(
                    "model call attempt_log entries must be ModelCallAttempt records"
                )
        # "Exactly once" is the claim, so the indices carry it rather than the count. A length
        # check accepts a log of the right size naming one dispatch twice and another not at
        # all -- well-formed in every other field, and unanswerable for the question the log
        # exists for, with nothing in the record to say so. The empty half of the rule has a
        # wire side this constructor cannot see -- absence and ``[]`` parse to the same tuple
        # -- and neither reader tries to: both spellings mean an unitemized call, the second
        # because every build before W7-4 wrote it for exactly that. Only the writers
        # converged; see ``to_json``.
        if self.attempt_log and tuple(entry.index for entry in self.attempt_log) != tuple(
            range(1, self.attempts + 1)
        ):
            raise ValueError(
                "model call attempt_log must be empty or name every attempt exactly once"
            )
        # A wait is what separates two dispatches, so the first entry has none to report:
        # nothing of this call precedes its first dispatch, and a line saying otherwise claims
        # the kernel waited for something it had not yet done. Checkable here -- unlike the
        # timeline inequality the same field takes part in, which needs a `latency_ms` the
        # runner has not stamped yet when it attaches the log -- because this one reads the
        # entries alone, and their own waits are final by the time they arrive. `None` is the
        # field's absence and not a wait of zero, so it stays legal: a record that predates the
        # field says nothing here rather than saying nothing happened.
        first_backoff = self.attempt_log[0].backoff_ms if self.attempt_log else None
        if first_backoff is not None and first_backoff != 0:
            raise ValueError(
                "model call attempt_log first entry backoff_ms must be 0: "
                "nothing precedes the first dispatch"
            )
        if self.latency_ms < 0:
            raise ValueError("model call latency_ms must not be negative")
        if self.capture_downgrades < 0:
            raise ValueError("model call capture_downgrades must not be negative")
        # The two closed enums, refused here as well as on the wire and through the same function.
        # `from_json` rejected a non-member while `to_json` emitted one, so this class could write
        # an audit record it would not read back -- a failure that surfaces in the consumer, long
        # after the writer that caused it. `ModelCallCapture` has always refused a `mode` outside
        # `CAPTURE_MODES` in its own `__post_init__`; these two were the pair that did not.
        # `replace` re-runs this, which is what puts the subscription narrowing and `with_error`
        # under the same rule rather than only the direct constructor.
        _validated_status(self.digest_status, "digest_status", DIGEST_STATUSES)
        _validated_status(self.destination_status, "destination_status", DESTINATION_STATUSES)
        for key, value in self.usage.items():
            # ``type(value) is not int`` rather than ``isinstance``: the same "what is a
            # countable int" its four sibling readers spell (``provider_usage_of``,
            # ``usage_reported_by``, ``with_error``, ``_recordable_usage``). ``isinstance``
            # here accepted every ``int`` subclass -- an ``IntEnum`` a provider SDK hands back
            # as a token count is the real shape -- so this constructor admitted a count that
            # every reader of the stamp had just refused. Excluding ``bool`` is now implied.
            if not isinstance(key, str) or type(value) is not int:
                raise WireValidationError("model call usage must be a mapping of str to int")
            # Every other counter on this receipt refuses a negative, and usage is the one that gets
            # summed: a single ``{"input_tokens": -100}`` in an audit payload silently subtracts from
            # an aggregate, so a budget check undercounts rather than visibly failing.
            if value < 0:
                raise WireValidationError(f"model call usage {key!r} must not be negative")
        object.__setattr__(self, "usage", dict(self.usage))
        # The entries are this receipt's own breakdown of its bill, so a log that does not add up
        # to that bill is not a breakdown of it -- and nothing else in the record says which of
        # the two is wrong. The docstring above gave "a sum over its usage would silently
        # disagree with the receipt's" as the reason the log is all-or-nothing; unchecked, that
        # was a rationale rather than a rule. Read after the normalization above, so the sum and
        # the total are compared on the same terms.
        if self.attempt_log:
            summed: dict[str, int] = {}
            for entry in self.attempt_log:
                for key, value in entry.usage.items():
                    summed[key] = summed.get(key, 0) + value
            if summed != self.usage:
                raise ValueError("model call attempt_log usage must sum to the receipt's usage")

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
        # Read once, before the `try`, and read so that it cannot raise: the fallback below is an
        # `except` handler, and spelling it `type(exc).__name__` made the handler re-raise -- an
        # exception escaping the one object documented to survive any exception. The same read also
        # bounds the name, which reaches the wire as `error_code`: measured, 1,000,000 characters.
        type_name = portable_type_name(exc)
        try:
            error_code = getattr(exc, "error_code", "") or type_name
            # `exact_text`, not `str`: a `str` subclass answering `__str__` forges this code.
            normalized_error_code = normalize_unicode_scalars(exact_text(error_code))
        except Exception:
            normalized_error_code = type_name
        try:
            provider_error_code = normalize_unicode_scalars(
                str(getattr(exc, "provider_error_code", "") or "")
            )
        except Exception:
            provider_error_code = ""
        try:
            retryable_value = getattr(exc, "retryable", False)
            retryable = retryable_value if type(retryable_value) is bool else False
        except Exception:
            retryable = False
        try:
            config_recoverable_value = getattr(exc, "config_recoverable", False)
            config_recoverable = (
                config_recoverable_value if type(config_recoverable_value) is bool else False
            )
        except Exception:
            config_recoverable = False
        try:
            http_status = getattr(exc, "http_status", None)
        except Exception:
            http_status = None
        if isinstance(http_status, bool) or not isinstance(http_status, int):
            http_status = None
        try:
            provider_retried_value = getattr(exc, "provider_retried", False)
            provider_retried = (
                provider_retried_value if type(provider_retried_value) is bool else False
            )
        except Exception:
            provider_retried = False
        # What the provider already produced and billed for before the call was refused. A
        # failure after a complete answer is a real shape -- the applied-parameters proof
        # refusals parse a turn, read its usage, and only then reject it -- and a receipt that
        # reports zero there drops a paid call out of the metrics and out of the cumulative
        # token budget. Same guarded read as `provider_retried`, and the same combine rule: a
        # receipt that already carried counts keeps them.
        try:
            stamped_usage = getattr(exc, "provider_usage", None)
        except Exception:
            stamped_usage = None
        usage = self.usage
        # ``and not self.attempt_log``: the entries are this receipt's own per-dispatch breakdown
        # of its bill, and a total read off an exception cannot restate them -- writing one over
        # them produces a record ``__post_init__`` refuses, which would come out of this method as
        # a ``ValueError`` *instead of* the failure it was called to report. Every read here is
        # guarded for that reason; the sum invariant added one more way to break the same promise.
        # The same rule the line below already follows for ``provider_retried``, one field wider:
        # a receipt that already carried the fact keeps what it carried.
        if not usage and not self.attempt_log and isinstance(stamped_usage, Mapping):
            try:
                usage = {
                    str(key): int(value)
                    for key, value in stamped_usage.items()
                    if type(value) is int and value >= 0
                }
            except Exception:
                usage = self.usage
        return replace(
            self,
            error_code=normalized_error_code,
            provider_error_code=provider_error_code,
            retryable=retryable,
            config_recoverable=config_recoverable,
            http_status=http_status,
            usage=usage,
            # A failed call is the one most likely to have been retried, so the marker has to
            # survive the failure path too -- recording it only on success would deny retries in
            # exactly the exhausted-budget case.
            #
            # Combined with what the receipt already holds rather than assigned over it. Today's one
            # caller always passes a freshly built receipt, so nothing is lost yet; but every other
            # place this fact travels had to learn the same rule, and a receipt that had recorded a
            # retry before failing would silently unrecord it here.
            provider_retried=self.provider_retried or provider_retried,
        )

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
            "config_recoverable": self.config_recoverable,
            "http_status": self.http_status,
            "redaction_digest": self.redaction_digest,
            "capture_downgrades": self.capture_downgrades,
            "digest_generation": self.digest_generation,
            "digest_status": self.digest_status,
            "destination_status": self.destination_status,
            "destination_digest": self.destination_digest,
            "idempotency_key": self.idempotency_key,
        }
        # Emitted only when there is something to itemize. Presence IS the claim -- one entry
        # per dispatch, indices ``1..attempts`` -- so this writer gives an empty log no
        # spelling of its own: absence covers a record that predates the field, a call with
        # zero dispatches, and a receipt built without a log, which coincide in every readable
        # fact. Emitting ``[]`` unconditionally gave that one value a second spelling and made
        # a parsed pre-field receipt come back wearing it. The readers still accept both --
        # every build before this one wrote ``[]`` for an empty log, at whatever ``attempts``
        # the receipt carried, and those lines are honest records of unitemized calls.
        # ``idempotency_key`` above keeps the opposite rule on purpose: its absence spelling
        # is the in-band empty string, so the key itself always travels.
        if self.attempt_log:
            payload["attempt_log"] = [entry.to_json() for entry in self.attempt_log]
        return payload

    @classmethod
    def from_json(cls, payload: Any) -> ModelCallReceipt:
        payload = require_object(payload, "model_call_receipt")
        # Absence on this record is legacy and reads as the field's default; ``null`` is a writer
        # that had the field and wrote nothing, which no writer here has ever done. Applied over
        # the record's own fields, so ``context``, ``model`` and ``usage`` -- which collapsed the
        # same way ``attempt_log`` did -- are covered by the rule rather than by three checks.
        _refuse_null_wire_values(payload, cls, "model call receipt")
        usage = _optional_object(payload, "usage") or {}
        # Absent on every receipt written before this field existed, which is legal and reads
        # as an empty log beside an intact ``attempts`` count; present-but-mistyped is refused,
        # like every other field here. A present log of the wrong length is refused by
        # ``__post_init__`` — that shape is a bug in a writer, not a legacy to absorb. A
        # present-but-EMPTY log is read, not refused, at every ``attempts`` count: it is the
        # spelling every build before W7-4 gave an unitemized call, because ``to_json``
        # emitted the key unconditionally and an empty log is a legal receipt at any count
        # (empty *or* complete — the empty arm was never reserved for refused calls). The
        # runner never wrote that pair, but the runner is not the only writer: a receipt
        # handed straight to ``AgentRecorder.record_settled_call`` carries whatever log it
        # was built with, and the default is none. Refusing it here would convict the lines
        # the previous build wrote through its own public API. The convergence W7-4 makes is
        # the writer's alone (see ``to_json``): one spelling produced, both still read.
        raw_attempt_log = payload.get("attempt_log")
        if raw_attempt_log is None:
            attempt_log: tuple[ModelCallAttempt, ...] = ()
        elif isinstance(raw_attempt_log, list):
            attempt_log = tuple(ModelCallAttempt.from_json(item) for item in raw_attempt_log)
        else:
            raise WireValidationError("attempt_log must be an array")
        context_payload = _optional_object(payload, "context")
        model_payload = _optional_object(payload, "model")
        raw_status = payload.get("http_status")
        return cls(
            context=(
                InvocationContext.from_json(context_payload)
                if context_payload is not None
                else InvocationContext()
            ),
            model=_parsed_model_config(model_payload),
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
            # Absent on every receipt written before this field existed, which is legal and
            # reads as False; present-but-mistyped is refused, like every other bool here.
            config_recoverable=parse_bool(payload, "config_recoverable"),
            http_status=None if raw_status is None else parse_int(payload, "http_status"),
            redaction_digest=parse_str(payload, "redaction_digest"),
            capture_downgrades=parse_int(payload, "capture_downgrades"),
            # Absent on every receipt written before these fields existed, which is legal; the
            # statuses then read as the "we never got that far" default *except* where the digest
            # they describe is there to say otherwise (see :func:`_parsed_status`), and the
            # generation stays empty because a legacy key was taken under rules this record cannot
            # name. Present-but-mistyped is refused, like every other string here. The statuses are
            # closed kernel enums rather than free strings -- unlike ``stop_reason``, whose openness
            # exists because a *provider* may add a value -- so an unknown one is a bug in a writer,
            # not a provider surprise to absorb.
            digest_generation=parse_str(payload, "digest_generation"),
            digest_status=_parsed_status(
                payload,
                "digest_status",
                DIGEST_STATUSES,
                witness="request_digest",
                witnessed="ok",
            ),
            destination_status=_parsed_status(
                payload,
                "destination_status",
                DESTINATION_STATUSES,
                # The only arm that answers with a value: `not_declared`, `declined` and
                # `unavailable` all produce an empty digest, so a non-empty one names `resolved`
                # and nothing else.
                witness="destination_digest",
                witnessed="resolved",
            ),
            destination_digest=parse_str(payload, "destination_digest"),
            attempt_log=attempt_log,
            # Absent on every receipt written before the field existed, which is legal and reads
            # as "never keyed"; present-but-mistyped is refused, like every other string here.
            idempotency_key=parse_str(payload, "idempotency_key"),
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
    as read-only. Each consumer gets its own capture and its own receipt: the counts and taxonomy
    agree across consumers, but a `none`-mode receipt has its content-derived digests cleared, so two
    receipts from one call are not necessarily equal.
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


def redacted_fields_or_none(
    content: Mapping[str, Any],
    *,
    policy: RedactionPolicy,
    redactor: Redactor | None = None,
) -> Mapping[str, Any] | None:
    """Redact a field mapping, or return `None` if the redactor failed *or* misbehaved.

    `Redactor.redact` is typed `Any -> Any`, so a third-party implementation can return a scalar or a
    list for a mapping input — plausibly, since "mask the whole payload" is a tempting one-liner. The
    pipeline needs fields back to deliver fields, and converting a non-mapping result here rather than
    at the call site is what keeps that failure inside the fail-closed path: raised outside it, the
    exception would escape `dispatch_model_call` and fail a model call the provider has already been
    paid for.

    A shape violation is treated exactly like a raise. It is a contract violation either way, and the
    consumer that asked for redacted content gets metadata instead of a guess.

    The redactor is handed a **fully detached** payload it may treat as its own, so an implementation
    that edits mappings and lists in place is legal. Nothing in the `Redactor` contract forbids that, and
    it is the natural way to write one — but with only the outer mapping copied it mutated the caller's
    settled payload *and* the input the next redacted subscription would see, so the first consumer's
    rules silently became everyone's. Detached per call rather than once per dispatch, precisely because
    an in-place redactor would otherwise contaminate its peers.
    """
    redacted = redacted_or_none(_detached_content(content), policy=policy, redactor=redactor)
    if not isinstance(redacted, Mapping):
        return None
    try:
        normalized = normalize_json_ingress(
            {exact_text(key): value for key, value in exact_items(redacted)}
        )
    except Exception:
        return None
    return normalized if isinstance(normalized, Mapping) else None


def _detached_content(content: Mapping[str, Any]) -> Mapping[str, Any]:
    """A copy of `content` that shares no nested structure with the caller's payload.

    `dict(content)` copies only the outer mapping, so a nested message list stayed shared: a caller that
    mutated its own payload after dispatch changed captures observers had already retained, while the
    digests kept describing the pre-mutation value. A capture is meant to be a settled record.

    For `full` mode this is done once per dispatch and shared by those subscriptions rather than copied
    per observer. Content can carry resolved media, and copying that per subscriber is real cost for a
    case an observer is already forbidden to cause — treating the capture as read-only is part of the
    `ModelIOObserver` contract, whereas a caller mutating its own dict violates nothing. Redaction is the
    opposite: an in-place redactor is legal, so each redacted subscription gets its own copy.

    A payload holding something `deepcopy` refuses falls back to the shallow copy: degraded isolation is
    survivable, failing a model call the provider has already been paid for is not.
    """
    try:
        return copy.deepcopy(dict(content))
    except Exception:
        return dict(content)


def _resolve_capture(
    policy: CapturePolicy, content: Mapping[str, Any]
) -> tuple[CaptureMode, str, Mapping[str, Any] | None]:
    """Decide what one subscription sees, as `(mode, downgraded_from, content)`."""
    if policy.mode == "none":
        return "none", "", None
    if policy.mode == "digest":
        return "digest", "", None
    if policy.mode == "full":
        return "full", "", content
    if policy.effective_redactor is None:
        # A policy that came back from JSON knowing it *had* a custom redactor, and no longer has one.
        # Silently applying the built-in rules would be the worst outcome available: the consumer is
        # told it received redacted content while a classifier that masked more than key names and
        # regexes is simply absent. Missing machinery is a redaction failure, not a weaker redaction.
        return "digest", "redacted", None
    redacted = redacted_fields_or_none(
        content, policy=policy.effective_redaction, redactor=policy.redactor
    )
    if redacted is None:
        # Fail closed. This consumer asked for redacted content and the redactor could not produce it,
        # so it gets what ``digest`` would have given it -- never the raw value.
        return "digest", "redacted", None
    return "redacted", "", redacted


# Receipt fields derived from the call's content. A ``none``-mode consumer must not receive these:
# ``none`` promises no content metadata, and a digest of a short prompt is recoverable by hashing
# candidates. Token counts, timings and taxonomy stay -- they are metadata about the call rather than
# about what was said, and withholding them would leave an accounting or alerting consumer unable to
# do its job for no privacy gain.
_CONTENT_DERIVED_RECEIPT_FIELDS = ("prompt_digest", "request_digest")


def _receipt_for_subscription(
    receipt: ModelCallReceipt, *, mode: CaptureMode, policy: CapturePolicy
) -> ModelCallReceipt:
    """The receipt one subscription receives: narrowed to its mode, and naming its own rules.

    ``redaction_digest`` is a *per-subscription* fact, not a per-call one. There is no single applied
    policy at call level -- that is the whole point of attaching one per registration -- so the
    caller's receipt cannot carry a meaningful value and two redacted consumers with different rules
    would otherwise get identical audit records. It is set only when redaction actually ran: a
    downgraded subscription applied no rules at all, and stamping the policy it *failed* to apply
    would read as "these rules were applied", which ``downgraded_from`` already reports correctly.
    """
    changes: dict[str, Any] = {
        "redaction_digest": policy.effective_redaction.digest if mode == "redacted" else "",
    }
    if mode == "none":
        changes.update(dict.fromkeys(_CONTENT_DERIVED_RECEIPT_FIELDS, ""))
        # Say that the key was taken away rather than leaving the consumer to read the empty
        # string as "there was never one". Only when there *was* one: overwriting ``absent`` with
        # ``withheld`` would claim a policy removed a key that no policy ever saw.
        if receipt.digest_status == "ok":
            changes["digest_status"] = "withheld"
    return replace(receipt, **changes)


def dispatch_model_call(
    *,
    receipt: ModelCallReceipt,
    content: Mapping[str, Any],
    subscriptions: Sequence[ModelIOSubscription],
    check_authority: Callable[[], None] | None = None,
) -> ModelCallReceipt:
    """Deliver one settled model call to every subscription under its own policy.

    Returns the caller's receipt plus `capture_downgrades`. The count is resolved in a first pass
    *before* any delivery, so every observer agrees on it. Delivering as we go would hand the first
    observer a count of zero and the last the true total, and a receipt that disagrees with itself
    across consumers is worse than no count at all.

    What each observer receives is that receipt narrowed to its mode: a `none`-mode consumer gets the
    content-derived digests cleared, because `none` promises no content metadata and the receipt's
    `prompt_digest` would otherwise walk straight past the per-field digests this function withholds.
    The receipt *returned* keeps them — the caller is the kernel, which computed them.

    Digests and lengths are computed once, on the raw content, and shared. Beyond the cost, that is
    what makes them comparable: a per-observer digest taken after redaction would differ by policy and
    could not join anything.

    An observer that raises is skipped and the rest still run. The call already happened and the
    provider has already been paid; a broken exporter does not get to undo that.

    A durable host may pass ``check_authority``. It runs around every policy resolver and observer
    callback so one blocking subscriber cannot keep the rest of the fan-out alive after writer
    ownership moves. Authority exceptions are never contained as observer failures.
    """
    if not subscriptions:
        return receipt

    if check_authority is not None:
        check_authority()

    full_content = (
        _detached_content(content)
        if any(subscription.policy.mode == "full" for subscription in subscriptions)
        else content
    )

    resolved: list[tuple[CaptureMode, str, Mapping[str, Any] | None]] = []
    for subscription in subscriptions:
        if check_authority is not None:
            check_authority()
        resolved.append(
            _resolve_capture(
                subscription.policy,
                full_content if subscription.policy.mode == "full" else content,
            )
        )
        if check_authority is not None:
            check_authority()
    downgrades = sum(1 for _mode, downgraded_from, _payload in resolved if downgraded_from)

    # Only if somebody will actually see them. Hashing walks every field and, for a value with no JSON
    # form, materializes a string of it -- so a run wired to nothing but a ``none``-mode observer was
    # paying to digest resolved media it then discarded. Keyed on the *resolved* modes, not the declared
    # ones: a subscription downgraded from ``redacted`` lands on ``digest`` and does see this metadata.
    reveals_any_metadata = any(mode != "none" for mode, _downgraded_from, _payload in resolved)
    digests = (
        {key: content_digest(value) for key, value in content.items()}
        if reveals_any_metadata
        else {}
    )
    lengths = (
        {
            key: length
            for key, value in content.items()
            if (length := content_length(value)) is not None
        }
        if reveals_any_metadata
        else {}
    )
    settled = replace(receipt, capture_downgrades=receipt.capture_downgrades + downgrades)

    for subscription, (mode, downgraded_from, payload) in zip(subscriptions, resolved, strict=True):
        if check_authority is not None:
            check_authority()
        reveals_metadata = mode != "none"
        capture = ModelCallCapture(
            receipt=_receipt_for_subscription(settled, mode=mode, policy=subscription.policy),
            mode=mode,
            downgraded_from=downgraded_from,
            content=payload,
            digests=digests if reveals_metadata else {},
            lengths=lengths if reveals_metadata else {},
        )
        try:
            subscription.observer.on_model_call(capture)
        except WriteAuthorityRevoked:
            raise
        except Exception:
            pass
        if check_authority is not None:
            check_authority()
    return settled


def close_model_io_subscriptions(
    subscriptions: Sequence[ModelIOSubscription],
    *,
    check_authority: Callable[[], None] | None = None,
) -> None:
    """Release every observer that declared a `close`, once each, tolerating failures.

    Probed with `getattr` rather than required, which is what keeps `close` off the base protocol.

    De-duplicated by identity, because registering one exporter under two policies is a shape
    `ModelIOSubscription` explicitly supports. Closing it once per subscription asks a `close` that
    flushes or commits to be idempotent, and the failure mode is quiet: the second call's exception is
    swallowed by the same guard that makes a broken exporter survivable.
    """
    seen: set[int] = set()
    for subscription in subscriptions:
        if check_authority is not None:
            check_authority()
        observer = subscription.observer
        if id(observer) in seen:
            continue
        seen.add(id(observer))
        try:
            close = getattr(observer, "close", None)
        except WriteAuthorityRevoked:
            raise
        except Exception:
            # ``close`` is optional and may be exposed through a descriptor. A broken probe is an
            # observer failure, not permission to alter the run outcome or skip later observers.
            if check_authority is not None:
                check_authority()
            continue
        if not callable(close):
            continue
        try:
            close()
        except WriteAuthorityRevoked:
            raise
        except Exception:
            if check_authority is not None:
                check_authority()
            continue
        if check_authority is not None:
            check_authority()
