"""Model-supplied scalars that answer questions about themselves.

An in-process tool handler, a plan item and a task payload are Python objects, not JSON, so a
value can be an ``int`` subclass with its own ordering. ``json.dumps`` spells such a subclass by
its **base** value, so every guard deciding "can a writer spell this" has to ask the base value
too — asked through ``<`` or unary ``-``, the object answers, and it can answer by raising or by
lying. Defined once and imported by both pins, because the rule is one rule bound at two sites:
the refusing ingress boundary and the preview threshold.
"""

from __future__ import annotations

from typing import Any


class ExplodingComparisons(int):
    """Raises from every ordering. A plain ``5`` that no writer has any trouble with."""

    def __lt__(self, other: Any) -> bool:
        raise RuntimeError("hostile __lt__")

    def __gt__(self, other: Any) -> bool:
        raise RuntimeError("hostile __gt__")

    def __neg__(self) -> int:
        raise RuntimeError("hostile __neg__")


class UnderstatedInteger(int):
    """Claims to sit inside every bound it is asked about, however large it really is."""

    def __lt__(self, other: Any) -> bool:
        return True

    def __gt__(self, other: Any) -> bool:
        return True


class ExplodingText(str):
    """Raises from every question about its size or shape. The text itself is ordinary."""

    def encode(self, *args: Any, **kwargs: Any) -> bytes:
        raise RuntimeError("hostile encode")

    def split(self, *args: Any, **kwargs: Any) -> list[str]:
        raise RuntimeError("hostile split")

    def lower(self) -> str:
        raise RuntimeError("hostile lower")


class UnderstatedText(str):
    """Reports a single byte however long it really is — the cap's own question, answered wrong."""

    def encode(self, *args: Any, **kwargs: Any) -> bytes:
        return b"x"


class MisreportingKey(str):
    """Spells one name and answers ``lower()`` with another, so no rule matches what it really is."""

    def lower(self) -> str:
        return "harmless"


class MisreportingText(str):
    """Answers ``__str__`` with something else, which is what every bare ``str(x)`` conversion asks."""

    def __str__(self) -> str:
        return "harmless"


class ShoutingText(str):
    """Answers ``upper()`` with something else -- the one question ``public_error_message`` asks."""

    def upper(self) -> str:
        return "NOTHING TO SEE HERE"


class EmptyClaimingPath(str):
    """Claims to equal the empty string. ``normalize_workspace_path`` asks ``raw == ""`` first, so
    this is how a path talks its way past the operator's ``redact_patterns``."""

    def __eq__(self, other: Any) -> bool:
        return other == ""

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return str.__hash__(self)


class UniterableText(str):
    """Refuses to be iterated. ``normalize_unicode_scalars`` scans code units, and this runs on the
    ingress path where an unclassified exception is a dead run."""

    def __iter__(self) -> Any:
        raise RuntimeError("hostile __iter__")


# ---------------------------------------------------------------------------
# Types that answer for themselves. `type(value).__name__` is an attribute read on the *class*,
# so it dispatches to the metaclass -- the same shape as `value.encode()` one level up.
# ---------------------------------------------------------------------------


class RaisingNameAccess(type):
    """Raises from ``__getattribute__`` when anything reads ``__name__``."""

    def __getattribute__(cls, name: str) -> Any:
        if name == "__name__":
            raise RuntimeError("hostile metaclass __getattribute__")
        return super().__getattribute__(name)


class RaisingNameProperty(type):
    """Puts a raising descriptor at ``__name__``. ``type.__getattribute__`` still finds this one,
    which is why the base slot -- and not the reviewer's first suggestion -- is the answer."""

    @property
    def __name__(cls) -> str:
        raise RuntimeError("hostile metaclass __name__ property")


class LyingNameProperty(type):
    """Answers ``__name__`` with a portable type's name, so the marker names the wrong thing."""

    @property
    def __name__(cls) -> str:
        return "str"


class HiddenName(metaclass=RaisingNameAccess):
    pass


class ExplodingName(metaclass=RaisingNameProperty):
    pass


class ImpersonatingName(metaclass=LyingNameProperty):
    pass


class _ExplodingNameText(str):
    """A ``str`` subclass for use *as* a class's ``__name__``."""

    def __str__(self) -> str:
        raise RuntimeError("hostile __str__ on the name itself")


class RenamedByAHostileString:
    """``cls.__name__ = <str subclass>`` is accepted by CPython -- ``type_set_name`` checks
    ``PyUnicode_Check``, which admits subclasses, and stores the object handed to it. So reading the
    name through the base slot moves the question from the metaclass to the name object, and only
    ``exact_text`` on the result closes it."""


RenamedByAHostileString.__name__ = _ExplodingNameText("RenamedByAHostileString")


HOSTILE_NAMED_TYPES = (
    HiddenName,
    ExplodingName,
    ImpersonatingName,
    RenamedByAHostileString,
)


def hugely_named_object(characters: int = 10_000) -> Any:
    """An instance of a class whose name is legal, ordinary to construct, and enormous."""

    return type("z" * characters, (), {})()


class HostileNamedList(list, metaclass=RaisingNameAccess):
    """A container that answers for its own type name. The depth cap and the cycle guard both
    publish `type(value).__name__` of a *container*, so the hostile shape there is not a scalar."""


class HostileNamedDict(dict, metaclass=RaisingNameAccess):
    """The dict half of the same pair."""


# ---------------------------------------------------------------------------
# Containers that answer for their own CONTENTS. The scalar generations above are defeated by
# reading the base value; a container is defeated the same way, but the argument for it is not the
# same. `json.dumps` spells an `int` subclass by its base value, so "what will a writer spell"
# settles the scalar case -- and it does NOT settle this one: measured, `json.dumps` of a `list`
# subclass reads the real storage while `json.dumps` of a `dict` subclass takes the overridden
# `items()`. The two halves answer opposite ways. What settles it instead is that the COPY is the
# record: the original never reaches a writer, and every later reader -- the checkpoint, the
# transcript, the preview, the operator's redact patterns -- sees what the walk copied.
# ---------------------------------------------------------------------------


class UnderstatedList(list):
    """Reports one element however many it holds.

    The ingress walk sizes the copy with `len` and then reads that many indices, so the copy is a
    prefix and the rest is dropped in silence -- while the preview slices the real storage and
    publishes all of it. The stored record and the published one disagree about what happened.
    """

    def __len__(self) -> int:
        return 1


class OverstatedList(list):
    """Reports more elements than it holds.

    `len` sizes the copy and `__getitem__` fills it, so the walk's two questions disagree and it
    raises `IndexError` -- unclassified, out of a boundary whose whole job is to classify.
    """

    def __len__(self) -> int:
        return 5


class SubstitutingList(list):
    """Answers every index and every slice with something other than what it stores."""

    def __getitem__(self, index: Any) -> Any:
        return "CLEAN"


class UniterableList(list):
    """Iterates as empty while holding its elements.

    `touches_redacted_path` walks a list with `__iter__` and the preview walks it with a slice, so
    this is the shape where the escape hatch and the publication disagree about whether the
    operator's pattern was touched at all -- the container twin of `EmptyClaimingPath`.
    """

    def __iter__(self) -> Any:
        return iter(())


class MisreportingItems(dict):
    """Answers `items()` with a mapping other than the one it stores."""

    def items(self) -> Any:  # type: ignore[override]
        return [("safe", 1)]


class UnderstatedDict(dict):
    """Answers `len()` with fewer keys than `items()` yields, so a walk deriving a truncation count
    from the difference reports a drop that never happened."""

    def __len__(self) -> int:
        return 1


HOSTILE_CONTAINERS = (
    UnderstatedList([1, 2, 3]),
    OverstatedList([1, 2]),
    SubstitutingList([1, 2]),
    UniterableList([1, 2]),
    MisreportingItems({"content": "SECRET", "safe": 1}),
    UnderstatedDict({"content": "SECRET", "safe": 1}),
)
