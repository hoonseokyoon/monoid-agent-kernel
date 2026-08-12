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


class UnportableValueError(ValueError):
    """A value no portable JSON writer can carry reached a refusing ingress.

    Its own class so the refusing boundaries can convert exactly this into their classified error
    and leave the normalizer's *other* ``ValueError`` — colliding keys after normalization — on the
    classification it already had. One base and not two independent classes, because a boundary
    that caught one and not the other would be this repository's own recurring defect: a rule bound
    to one of two parallel halves.
    """


class UnportableScalarError(UnportableValueError):
    """A scalar no portable JSON writer in this process can spell."""


class UnportableContainerError(UnportableValueError):
    """A container whose *shape*, not whose scalars, no portable writer can carry.

    Separate from the scalar refusal because the two are found by different machinery and at
    different times: a scalar is judged where it sits, and a shape is only knowable once the whole
    copy exists. Both are ``UnportableValueError``, so no boundary has to know that.
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
    # `portable_type_name`, not `type(value).__name__`: this f-string is the last thing that runs
    # before the classified refusal exists, and the plain read lets the value's metaclass raise
    # *here* -- an unclassified exception thrown by the error path of the mechanism that exists to
    # keep unclassified exceptions off this boundary. Measured through a real run: RuntimeError,
    # status failed, internal_error, and no `tool.call.failed` at all.
    raise UnportableScalarError(f"value of type {portable_type_name(value)} is not portable JSON")


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


_BASE_TYPE_NAME = type.__dict__["__name__"].__get__
TYPE_NAME_MAX_ESCAPED_BYTES = 64


def _escaped_cost(text: str) -> int:
    """The bytes ``text`` costs where it is charged: escaped, without the quotes.

    The payload accountant prices a fragment the way the *widest* sink spells it, non-ASCII
    escaped, so this is the unit a bound on published text has to be written in. Escaping is
    per-character with no context, so a per-character cost sums to the whole string's.
    """

    return len(json.dumps(text, ensure_ascii=True).encode("utf-8")) - 2


def portable_class_name(cls: type) -> str:
    """A class's own name, bounded, with nothing on the way that gets to answer for it.

    ``cls.__name__`` is an attribute read on a *class*, so it dispatches to the **metaclass** — the
    same shape as ``value.encode()`` one level down, and reached the same way: an in-process tool or
    task hands back an object, and its type is whatever built it. Three separate answers had to be
    taken away, and each line here takes one:

    * ``type.__dict__["__name__"].__get__`` is the base getset slot, so neither a metaclass
      ``__getattribute__`` nor a metaclass ``__name__`` property runs. ``type.__getattribute__``
      is *not* enough — it bypasses the first and still finds the second.
    * ``exact_text`` on the result, because ``cls.__name__ = <str subclass>`` is accepted
      (``type_set_name`` checks ``PyUnicode_Check``, which admits subclasses, and stores the object
      it was given). Without it the base slot only moves the question from the metaclass to the name
      object: measured, a name whose ``__str__`` raises still took down the f-string that used it.
    * The cap, because a class name is legal at any length. Measured: a 1,000,000-character
      name published a 1,000,038-byte payload against a 262,144-byte ceiling — 3.8× — through the
      two fallbacks that publish it uncharged, and it made an unbounded ``error_code`` and a
      1,000,077-character ``tool.call.failed.error``. The cap is in **escaped bytes**, the unit the
      accountant charges in, and not in characters: written in characters it was no bound at all on
      the surface that pays for it, because a 64-character Hangul name costs 415 bytes where the
      same length of ASCII costs 95 — measured, 4.4×, and multiplied by every sibling fallback a
      fixed-field builder emits after exhaustion, which spend through `charge_marker` and so are
      deducted unconditionally. 64 bytes leaves every ASCII name this repository defines untouched
      (the longest is under a quarter of it) and prices every script the same.

    No ``try``/``except`` and no non-``str`` arm: with ``cls`` coming from ``type(value)`` there is
    nothing here that can raise. 5,759 distinct type objects were swept — every stdlib and kernel
    module, ctypes and enum metaclasses among them — plus every hostile construction that defeats
    the plain read; the base slot raised zero times and returned a non-``str`` zero times, and the
    only arguments that do raise are non-types, which ``type(value)`` cannot produce. An unreachable
    branch presented as a defence is worse than no branch: it reads as a guarded site to the next
    person and is never exercised.
    """

    name = exact_text(_BASE_TYPE_NAME(cls))
    if _escaped_cost(name) <= TYPE_NAME_MAX_ESCAPED_BYTES:
        return name
    kept: list[str] = []
    spent = 0
    for character in name:
        cost = _escaped_cost(character)
        if spent + cost > TYPE_NAME_MAX_ESCAPED_BYTES:
            break
        kept.append(character)
        spent += cost
    return "".join(kept)


def portable_type_name(value: Any) -> str:
    """The name of ``value``'s type. See ``portable_class_name`` for why it is not asked for."""

    return portable_class_name(type(value))


_BASE_DICT_ITEMS = dict.items
_BASE_DICT_LEN = dict.__len__
_BASE_LIST_LEN = list.__len__
_BASE_LIST_ITEM = list.__getitem__
_BASE_TUPLE_LEN = tuple.__len__
_BASE_TUPLE_ITEM = tuple.__getitem__
_BASE_STR_LEN = str.__len__
_BASE_BYTES_LEN = bytes.__len__


def exact_items(mapping: Any) -> Any:
    """A mapping's own entries, so a subclass does not choose which ones a walk sees.

    The container generation of the rule ``exact_text`` and ``portable_class_name`` already apply
    one level down, and it needs a different argument than they did. Theirs was "a writer spells
    the base value, so a guard must read the base value" -- measured, that does not settle this
    one: ``json.dumps`` reads a ``list`` subclass's real storage and takes a ``dict`` subclass's
    overridden ``items()``, so the two halves of "what will a writer spell" answer opposite ways.

    What settles it is that the COPY is the record. ``normalize_json_ingress`` is the last place a
    caller's object is seen; the checkpoint's ``asdict``, the transcript, the preview and the
    operator's redact patterns all read what the walk produced. A container that answers one way
    here and another way to the walk that publishes it does not make one of them wrong -- it makes
    the published record contradict the durable one, which is the failure neither walk can detect.

    Only ``dict`` has a base slot to read. A third-party ``Mapping`` implements ``items`` as its
    storage rather than as an override of one, so asking it is the only read there is.
    """

    if isinstance(mapping, dict):
        return _BASE_DICT_ITEMS(mapping)
    return mapping.items()


def exact_length(container: Any) -> int:
    """A container's own size. See :func:`exact_items` for why it is not asked for."""

    if isinstance(container, list):
        return _BASE_LIST_LEN(container)
    if isinstance(container, tuple):
        return _BASE_TUPLE_LEN(container)
    if isinstance(container, dict):
        return _BASE_DICT_LEN(container)
    if isinstance(container, str):
        return _BASE_STR_LEN(container)
    if isinstance(container, bytes):
        return _BASE_BYTES_LEN(container)
    return len(container)


def exact_item(sequence: Any, index: int) -> Any:
    """A sequence's own element at ``index``. See :func:`exact_items`."""

    if isinstance(sequence, list):
        return _BASE_LIST_ITEM(sequence, index)
    if isinstance(sequence, tuple):
        return _BASE_TUPLE_ITEM(sequence, index)
    return sequence[index]


def exact_elements(sequence: Any, stop: int | None = None) -> tuple[Any, ...]:
    """A sequence's own elements, up to ``stop`` of them.

    ``stop`` exists because a caller with a width cap already looks at only the first few, and
    reading the base storage should not turn a twenty-element preview into a million-element copy.
    """

    size = exact_length(sequence)
    if stop is not None and stop < size:
        size = stop
    return tuple(exact_item(sequence, index) for index in range(size))


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


MAX_PORTABLE_CONTAINER_DEPTH = 64
"""How deep a container may nest before no writer downstream can be trusted with it.

One number for the three sites that bound container nesting, because they judge the *same*
model-authored argument on its way to three different writers and a bound proven at one of them is
this repository's recurring defect. They do not take the same ACTION, and each says which it takes
where it stands: this one and ``core.tool_approval`` RAISE, ``core.model_io._jsonish`` returns a
marker. That difference is load-bearing rather than an inconsistency to tidy away -- a marker lets
sibling branches keep expanding on a cyclic input, which is how a fast ``RecursionError`` there
once became a hang, and it is why the shape refusal here raises.

64 rather than the parsers' 512: the writer that fails first is ``dataclasses.asdict`` at
``RunCheckpoint.to_json``, measured dead at 492 containers under the default recursion limit, so a
value in [492, 512] cleared every gate it met and killed the run at the checkpoint. 64 leaves an
enormous margin over anything real -- the deepest structure this repository defines anywhere is 10,
and ``PREVIEW_MAX_DEPTH`` is 8 -- while sitting far below the first writer that dies.
"""


def _refuse_unportable_shape(root: Any) -> None:
    """Refuse a copy whose shape a portable JSON writer cannot carry.

    Runs over the finished copy rather than during the walk, and that is not a convenience. The
    walk memoises a container *before* descending into it, so the second reference to a shared
    subtree short-circuits — which makes a depth counter carried through the walk unsound, not
    merely imprecise. Measured: a subtree referenced once near the root and once 250 levels down
    is charged its depth at whichever reference the walk reaches first, so the counter peaked at
    252 while the copy it cleared was 501 containers tall and killed ``dataclasses.asdict``. Over
    the finished copy the same sharing is free instead: each node is settled once, and a node
    referenced again is already settled.

    Three colours. ``on_path`` is the ancestor set — entered and not yet settled — so a hit there
    is a back edge and therefore a cycle. ``settled`` is everything already finished, so a hit
    there is a cross or forward edge and therefore ordinary sharing, which is accepted and left
    shared: the preview renders a value shared twice twice, and refusing every second visit would
    convict each DAG that ever reaches a tool result.

    Why refuse a cycle at all: the walk returns a self-referential copy, and the writers it is
    handed to disagree with it later and elsewhere — ``json.dumps`` raises ``ValueError: Circular
    reference detected`` and ``dataclasses.asdict`` raises ``RecursionError``, neither at the
    boundary that accepted it. Refusing here is the same trade the scalar refusal made: a
    classified failure of one call instead of an unclassified death of the run.

    Depth rides the same pass, as a HEIGHT computed on the way out. A height is the right number
    because it is what a recursive writer walks: ``dataclasses.asdict`` has no memo, so a subtree
    referenced twice is descended twice and its full height counts from each reference. Computed
    bottom-up over the DAG it costs one visit per node, which is the property the reverted
    ``PREVIEW_MAX_NODES`` budget lacked; refusing on the first node that exceeds is equivalent to
    refusing on the root's height, since the root's is the largest, and it exits earlier.
    """

    # A height counts the root container as 1, so the bound is the container count directly. That
    # is the same admission set the ask path has, but not for the reason it looks like: that side
    # counts the root container as depth 0, which would leave it one more permissive -- except it
    # descends into the leaf SCALAR too and charges it a level of its own, so both sides admit at
    # most `MAX_PORTABLE_CONTAINER_DEPTH` containers on any path. Written down because the first
    # attempt here was `+ 1`, derived from the depth-vs-height difference alone, and the pin that
    # compares the two paths' verdict *lists* is what caught it -- an off-by-one is a differing
    # element there, where two separately stated bounds would both have looked right.
    tallest_allowed = MAX_PORTABLE_CONTAINER_DEPTH

    settled: dict[int, int] = {}
    on_path: set[int] = set()
    stack: list[tuple[Any, bool]] = [(root, False)]

    while stack:
        node, leaving = stack.pop()
        if not isinstance(node, (dict, list)):
            continue
        key = id(node)
        if leaving:
            on_path.discard(key)
            children = node.values() if isinstance(node, dict) else node
            tallest_child = max(
                (settled[id(child)] for child in children if isinstance(child, (dict, list))),
                default=0,
            )
            settled[key] = tallest_child + 1
            if settled[key] > tallest_allowed:
                raise UnportableContainerError(
                    f"container nests deeper than {MAX_PORTABLE_CONTAINER_DEPTH} levels; "
                    "flatten the payload or pass it as a workspace file"
                )
            continue
        if key in on_path:
            raise UnportableContainerError(
                "container is reachable from itself and cannot be written as portable JSON"
            )
        if key in settled:
            continue
        on_path.add(key)
        stack.append((node, True))
        children = node.values() if isinstance(node, dict) else node
        for child in children:
            stack.append((child, False))


def normalize_json_ingress(
    value: Any,
    *,
    substitute_nonfinite: bool = True,
    normalize_strings: bool = True,
    refuse_unportable: bool = False,
) -> Any:
    """Copy and normalize a JSON-domain value without recursive Python calls.

    ``dict`` keys are normalized as well as values.  If two keys become equal after
    normalization, the input is rejected rather than silently overwriting one meaning.
    The copy's *shape* is read through the base slots (:func:`exact_items`, :func:`exact_length`,
    :func:`exact_item`) rather than asked of the container, because this copy is what every later
    reader gets and a subclass answering for itself was writing the record.
    Tuples become JSON arrays. Non-container values outside the JSON domain are left alone by
    default; the boundaries where such a value can only crash a later writer — the four
    Python-object ingress points: a tool result's content, ``emit_artifact`` metadata, a hosted
    task's request and result, and a model turn's tool-call arguments — pass
    ``refuse_unportable=True`` and turn the ``UnportableValueError`` into their own classified
    refusal instead of carrying the value to the crash. The same flag refuses a *shape* no
    writer can carry (:func:`_refuse_unportable_shape`), because a boundary that refused one
    and not the other would be a rule bound to one of two halves.
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
                refuse_unportable=refuse_unportable,
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
            for key, child in exact_items(source):
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

        # One read of the size, and the elements from the same base slots. Asked of the value, the
        # two questions have no consistent answer: a `__len__` reporting more than it holds sized
        # the copy past the last real index and the walk raised a raw `IndexError` -- out of a
        # boundary whose whole job is to hand back a classified refusal, and reachable from any
        # custom or MCP tool handler.
        size = exact_length(source)
        copied_list: list[Any] = [None] * size
        memo[source_id] = copied_list
        destination[slot] = copied_list
        for index in range(size - 1, -1, -1):
            pending.append((exact_item(source, index), copied_list, index))

    if refuse_unportable:
        _refuse_unportable_shape(root[0])
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
