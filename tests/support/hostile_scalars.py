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
