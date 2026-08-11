"""Normalize values as they cross into the kernel's JSON domain.

Python's ``json`` module accepts two values that the durable and wire contracts cannot
carry portably: lone UTF-16 surrogate code units and non-finite floats.  Normalize them
once at semantic ingress instead of teaching every preview, digest, and writer a different
recovery rule.

The traversal is iterative.  Tool results can be deeper than Python's call stack, and a
normalizer must not turn that existing durability limit into an earlier ``RecursionError``.
Shared containers and cycles retain their topology; strict JSON writers remain responsible
for rejecting cycles and other values outside the JSON domain.
"""

from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from typing import Any


_MAX_JSON_NESTING = 512
_MAX_JSON_INTEGER_DIGITS = 4300
# Below this, an integer is small enough that no configured limit can refuse its spelling, so the
# common case answers without consulting the interpreter at all.
_UNCONDITIONALLY_SPELLABLE_INT = 10**17


@lru_cache(maxsize=8)
def _digit_bound_magnitude(digits: int) -> int:
    """``10**digits``: |n| < this iff n has at most ``digits`` decimal digits.

    Cached because the bound is now read per call rather than fixed at import — the interpreter's
    limit is settable at runtime — and the distinct values it takes are few.
    """
    return 10**digits


def _spellable_integer_digits() -> int:
    """The digit budget an integer must fit for *this* process to publish it portably.

    Two bounds, and the answer is the smaller. The portable one is 4300, exactly
    ``parse_bounded_json_int``'s rule on the way in, so what one side admits the other can read.
    The local one is ``sys.get_int_max_str_digits()``: a host that sets ``PYTHONINTMAXSTRDIGITS``
    or calls ``sys.set_int_max_str_digits`` below 4300 — a documented hardening knob — makes its
    own ``json.dumps`` raise on integers this predicate would otherwise admit, which put the
    ``ValueError`` back at the transcript write that the refusing boundaries exist to prevent.
    Zero disables the interpreter's limit and leaves only the portable ceiling: a process that can
    spell a 4301-digit integer still must not hand one to a reader that cannot.
    """
    configured = sys.get_int_max_str_digits()
    if configured <= 0:
        return _MAX_JSON_INTEGER_DIGITS
    return min(_MAX_JSON_INTEGER_DIGITS, configured)


class UnportableScalarError(ValueError):
    """A scalar no portable JSON writer can spell reached a refusing ingress.

    Its own class so the four refusing boundaries can convert exactly this into their classified
    error and leave the normalizer's *other* ``ValueError`` — colliding keys after normalization —
    on the classification it already had.
    """


def _refuse_unportable_scalar(value: Any) -> None:
    """Raise for a scalar no portable JSON writer in this process can spell.

    Portable is the decoder's own vocabulary, two-sided: what `parse_bounded_json_int` and the
    strict loaders admit on the way in is what this refuses to admit past a Python-object ingress
    on the way through. ``bool`` before ``int`` because it is one; ``float`` is total here (the
    non-finite substitution is the caller's separate, documented choice); everything else —
    ``bytes``, ``Decimal``, arbitrary objects — is named by type only, never asked to repr itself.

    The magnitude is read through ``int.__index__``, the base slot, because the question is what a
    *writer* will spell and ``json.dumps`` spells an ``int`` subclass by its base value. Asked with
    ``<`` instead, the subclass answers: one that raises ends the run with an unclassified
    exception where this boundary promises a classified refusal — measured on a plain ``5``, which
    every writer here handles — and one that merely understates itself is declared portable and
    dies at the writer this exists to protect. Inside the ``isinstance``, deliberately:
    ``int.__index__`` raises ``TypeError`` for anything else, which would trade the classified type
    refusal below for exactly the bare crash this function is here to stop.
    """

    if value is None or isinstance(value, (str, bool, float)):
        return
    if isinstance(value, int):
        numeric = int.__index__(value)
        if -_UNCONDITIONALLY_SPELLABLE_INT < numeric < _UNCONDITIONALLY_SPELLABLE_INT:
            return
        digits = _spellable_integer_digits()
        bound = _digit_bound_magnitude(digits)
        if -bound < numeric < bound:
            return
        raise UnportableScalarError(f"integer exceeds the JSON bound of {digits} digits")
    raise UnportableScalarError(f"value of type {type(value).__name__} is not portable JSON")


def exact_text(value: Any) -> str:
    """The base ``str``'s own value, so a subclass cannot answer questions about itself.

    A Python-object ingress carries whatever a tool handler built, and a ``str`` subclass may
    override ``encode``, ``split``, ``lower`` or ``__str__``. Every cap in this codebase decides by
    *asking the value* — ``len(value.encode("utf-8"))`` — so the object gets to state its own size,
    and the writer that publishes it does not: ``json.dumps`` spells the base value. Measured: an
    ``encode`` returning one byte published 5,000 characters through a 160-byte cap, and made
    ``redacted_value`` report ``"bytes": 1`` for it; an ``encode`` that raises ends the run from
    inside event construction. Same rule as ``int.__index__`` on the integer guards, and the same
    idiom ``_exact_json_text`` already applies to JSON document text.

    Free for the ordinary case: ``str.__str__`` on an exact ``str`` returns the object itself, and
    the type check below skips even that call. Non-strings get ``str()``, so this is a drop-in for
    the ``str(key)`` conversions that used to do the same job by accident — by accident, because
    they route through ``type(value).__str__`` and an override defeats them.
    """

    if type(value) is str:
        return value
    if isinstance(value, str):
        return str.__str__(value)
    return str(value)


def normalize_unicode_scalars(value: str) -> str:
    """Return ``value`` with surrogate pairs combined and lone surrogates replaced.

    JSON decoders can expose ``"\\ud83d\\ude00"`` as two UTF-16 code units even though
    Python normally stores the corresponding scalar as one character.  Combining the pair
    preserves its meaning; replacing an unmatched code unit with U+FFFD keeps later UTF-8
    encoding total and deterministic.

    The *scan* reads the base text, so a ``str`` subclass with a hostile ``__iter__`` cannot raise
    from inside the four refusing boundaries — this runs on the ingress path, where an
    unclassified exception is the run-killing crash those boundaries exist to prevent.

    A value that needs no repair comes back **as it arrived**, subclass and all. Normalizing to an
    exact ``str`` here looked like the tidier half of the rule — one pass at the boundary instead
    of a rule each guard remembers — and it is wrong: this kernel carries ``str`` subclasses
    through here on purpose. ``permissions._LegacyPathPattern`` marks a retained pre-v0.20 pattern
    that needs the historical matcher, and stripping it made a replayed pre-v0.20 tool scope fail
    validation as *"escaped leading ! is a configuration spelling"*. A normalizer that silently
    destroys a marker is a worse defect than the one it was closing, and the one it was closing
    belongs at the guards anyway: ``exact_text`` at the site that measures or decides, where the
    question is actually asked.
    """

    scanned = exact_text(value)
    if not any(0xD800 <= ord(char) <= 0xDFFF for char in scanned):
        return value  # unrepaired: the caller's own object, marker and all
    # Past here the string is rebuilt, so the exact text is what gets indexed and what comes back.
    # A repaired value is a new string either way, and `__len__`/`__getitem__` are two more
    # questions this scan has no reason to ask the value.
    normalized: list[str] = []
    index = 0
    while index < len(scanned):
        codepoint = ord(scanned[index])
        if 0xD800 <= codepoint <= 0xDBFF and index + 1 < len(scanned):
            low = ord(scanned[index + 1])
            if 0xDC00 <= low <= 0xDFFF:
                normalized.append(chr(0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)))
                index += 2
                continue
        normalized.append("\ufffd" if 0xD800 <= codepoint <= 0xDFFF else scanned[index])
        index += 1
    return "".join(normalized)


def is_finite_json_number(value: Any) -> bool:
    """Return whether ``value`` is an exact JSON number representable as a finite float.

    Several wire and durable fields are stored or compared as floats. Converting an arbitrarily
    large JSON integer directly can raise ``OverflowError`` before those boundaries can return
    their documented validation error. Keep that conversion total and reject booleans and
    subclasses so callers share one exact, non-coercive rule.
    """

    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _normalize_scalar(
    value: Any,
    *,
    substitute_nonfinite: bool,
    normalize_strings: bool,
    refuse_unportable: bool,
) -> Any:
    if refuse_unportable:
        _refuse_unportable_scalar(value)
    if isinstance(value, str):
        return normalize_unicode_scalars(value) if normalize_strings else value
    if substitute_nonfinite and isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def normalize_json_ingress(
    value: Any,
    *,
    substitute_nonfinite: bool = True,
    normalize_strings: bool = True,
    refuse_unportable_scalars: bool = False,
) -> Any:
    """Copy and normalize a JSON-domain value without recursive Python calls.

    ``dict`` keys are normalized as well as values.  If two keys become equal after
    normalization, the input is rejected rather than silently overwriting one meaning.
    Tuples become JSON arrays. Non-container values outside the JSON domain are left alone by
    default; the boundaries where such a value can only crash a later writer — the four
    Python-object ingress points: a tool result's content, ``emit_artifact`` metadata, and a
    hosted task's request and result — pass ``refuse_unportable_scalars=True`` and turn the
    ``ValueError`` into their own classified refusal instead of carrying the value to the crash.
    """

    root: list[Any] = [None]
    memo: dict[int, Any] = {}
    pending: list[tuple[Any, Any, Any]] = [(value, root, 0)]

    while pending:
        source, destination, slot = pending.pop()
        if not isinstance(source, (dict, list, tuple)):
            destination[slot] = _normalize_scalar(
                source,
                substitute_nonfinite=substitute_nonfinite,
                normalize_strings=normalize_strings,
                refuse_unportable=refuse_unportable_scalars,
            )
            continue

        source_id = id(source)
        if source_id in memo:
            destination[slot] = memo[source_id]
            continue

        if isinstance(source, dict):
            copied: dict[Any, Any] = {}
            memo[source_id] = copied
            destination[slot] = copied
            prepared: list[tuple[Any, Any]] = []
            seen: set[Any] = set()
            for key, child in source.items():
                if not isinstance(key, str):
                    raise ValueError("JSON object keys must be strings")
                normalized_key = normalize_unicode_scalars(key)
                if normalized_key in seen:
                    raise ValueError("JSON object keys collide after ingress normalization")
                seen.add(normalized_key)
                prepared.append((normalized_key, child))
            for normalized_key, child in reversed(prepared):
                pending.append((child, copied, normalized_key))
            continue

        copied_list: list[Any] = [None] * len(source)
        memo[source_id] = copied_list
        destination[slot] = copied_list
        for index in range(len(source) - 1, -1, -1):
            pending.append((source[index], copied_list, index))

    return root[0]


def reject_nonfinite_json_constant(value: str) -> Any:
    """``json.loads(parse_constant=...)`` callback for strict RFC JSON input."""

    raise json.JSONDecodeError(f"non-finite number {value} is not valid JSON", value, 0)


def parse_finite_json_float(value: str) -> float:
    """Parse a JSON number while rejecting exponents outside Python's finite range."""

    parsed = float(value)
    if not math.isfinite(parsed):
        raise json.JSONDecodeError(f"non-finite number {value} is not valid JSON", value, 0)
    return parsed


def parse_bounded_json_int(value: str) -> int:
    """Parse a JSON integer under a deterministic, cross-interpreter digit limit."""

    digits = value[1:] if value.startswith("-") else value
    if len(digits) > _MAX_JSON_INTEGER_DIGITS:
        raise json.JSONDecodeError("JSON integer decoder limit exceeded", value, 0)
    try:
        return int(value)
    except ValueError as exc:
        raise json.JSONDecodeError("invalid JSON integer", value, 0) from exc


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build an object while rejecting raw and normalized duplicate keys."""

    decoded: dict[str, Any] = {}
    for key, value in pairs:
        normalized_key = normalize_unicode_scalars(key)
        if normalized_key in decoded:
            raise json.JSONDecodeError(
                "duplicate JSON object key after ingress normalization",
                repr(pairs),
                0,
            )
        decoded[normalized_key] = value
    return decoded


def _enforce_json_nesting_limit(text: str, *, message: str) -> None:
    """Reject documents whose container depth exceeds the portable parser limit.

    CPython's decoder nesting limit differs across supported interpreter versions.  A
    small lexical pass makes the contract deterministic while leaving syntax validation
    to ``json.loads``.  Delimiters inside strings, including escaped quotes, do not count.
    """

    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > _MAX_JSON_NESTING:
                raise json.JSONDecodeError(message, text, index)
        elif character in "]}" and depth:
            depth -= 1


def json_nesting_within_limit(text: str) -> bool:
    """Whether :func:`loads_json_ingress` would accept ``text``'s container depth.

    For writers of artifacts this package also reads back. The lexical bound belongs to the reader,
    so a writer that never asks it can emit a line no reader of the same file can parse -- and a
    line a validator cannot parse is a line whose contents it never checks. Asking before writing
    is what lets "every record is re-verified" be a property of the file rather than of the records
    that happened to be shallow.
    """

    try:
        _enforce_json_nesting_limit(text, message="JSON nesting exceeds the parser limit")
    except json.JSONDecodeError:
        return False
    return True


def _exact_json_text(text: Any) -> str:
    """Return an exact ``str`` so subclasses cannot override the lexical scan."""

    if not isinstance(text, str):
        raise TypeError("JSON document must be a string")
    return exact_text(text)


def loads_json_ingress(text: str) -> Any:
    """Parse external JSON strictly, then normalize every decoded string."""

    text = _exact_json_text(text)
    _enforce_json_nesting_limit(text, message="JSON nesting exceeds the parser limit")
    try:
        decoded = json.loads(
            text,
            parse_constant=reject_nonfinite_json_constant,
            parse_float=parse_finite_json_float,
            parse_int=parse_bounded_json_int,
            object_pairs_hook=_unique_json_object,
        )
    except json.JSONDecodeError:
        raise
    except RecursionError as exc:
        raise json.JSONDecodeError("JSON nesting exceeds the parser limit", text, 0) from exc
    except ValueError as exc:
        raise json.JSONDecodeError("invalid JSON value", text, 0) from exc
    try:
        return normalize_json_ingress(decoded)
    except ValueError as exc:
        raise json.JSONDecodeError("invalid JSON value", text, 0) from exc


def _loads_model_json_ingress(
    text: str,
    *,
    substitute_nonfinite: bool,
    normalize_strings: bool = True,
) -> Any:
    text = _exact_json_text(text)
    _enforce_json_nesting_limit(text, message="model JSON nesting exceeds the parser limit")
    try:
        decoded = json.loads(
            text,
            parse_int=parse_bounded_json_int,
            object_pairs_hook=_unique_json_object,
        )
    except json.JSONDecodeError:
        raise
    except RecursionError as exc:
        raise json.JSONDecodeError("model JSON nesting exceeds the parser limit", text, 0) from exc
    except ValueError as exc:
        raise json.JSONDecodeError("invalid model JSON value", text, 0) from exc
    try:
        return normalize_json_ingress(
            decoded,
            substitute_nonfinite=substitute_nonfinite,
            normalize_strings=normalize_strings,
        )
    except ValueError as exc:
        raise json.JSONDecodeError("invalid model JSON value", text, 0) from exc


def loads_model_json_ingress(text: str) -> Any:
    """Parse model-origin JSON and apply the model/tool substitution policy.

    Provider tool arguments are model content rather than an external control document. Python's
    decoder therefore accepts its legacy non-finite constants and exponent overflow, after which
    semantic ingress deterministically substitutes them with ``null``.
    """

    return _loads_model_json_ingress(text, substitute_nonfinite=True)


def loads_model_envelope_json_ingress(text: str) -> Any:
    """Parse a mixed model envelope while retaining non-finite markers for field validation.

    Gateway envelopes contain strict control fields alongside model-authored tool arguments.
    Keeping non-finite floats intact lets the envelope parser reject them in controls, while the
    parser can still normalize the model-authored argument subtrees separately.
    """

    return _loads_model_json_ingress(text, substitute_nonfinite=False)


def loads_model_stream_envelope_json_ingress(text: str) -> Any:
    """Parse a stream envelope while deferring content-fragment Unicode repair.

    Text, reasoning, and tool-argument fragments can split a UTF-16 surrogate pair across
    successive frames. Their chunk ingress buffers must see the original code units so they can
    preserve the scalar. Object keys are still normalized and collision-checked here; control
    strings are normalized by the gateway field parser before use.
    """

    return _loads_model_json_ingress(
        text,
        substitute_nonfinite=False,
        normalize_strings=False,
    )
