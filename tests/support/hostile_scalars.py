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
