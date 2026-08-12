"""Shared coercion helpers for execution-boundary dataclasses.

Used by permission and runtime execution parsers to keep JSON array handling
consistent. Internal helper.

:func:`matches_the_kernel_defaults` lives here for a layering reason rather than a thematic
one: it is the shared body of every ``is_default`` gate the kernel has, and those gates sit on
dataclasses in two modules that cannot import each other (``core.spec`` imports
``permissions``). This module imports nothing of the package, so it is the one place both
sides can reach -- and one rule stated once is the whole point of the gate.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import fields
from typing import Any, Callable


def matches_the_kernel_defaults(config: Any, base: type) -> bool:
    """Whether every field ``base`` DECLARES is still at its default on ``config``.

    The shared body of the ``is_default`` gates, and the reason none of them is
    ``self == type(self)()``: a generated dataclass ``__eq__`` is class-exact, so a public
    extension subclass with every kernel field at its default compared unequal to the base
    default and was read as "the caller configured this". The kernel supports such subclasses
    deliberately -- every validator gates on ``isinstance`` and
    ``providers/base._copy_with_fields`` exists so normalization does not have to call an
    extension's narrower constructor -- so a gate that answers about the CLASS answers a
    question nobody asked.

    Fields are read off ``base`` rather than off ``type(config)``, and that asymmetry is the
    point rather than an accident: everything these gates guard reads the base's fields only.
    For the model configs that is the projection (``build_reasoning_payload``,
    ``build_generation_payload``, ``ReasoningConfig.to_json``), so an extension field is
    invisible to every wire, digest and echo downstream; for ``PermissionPolicy`` it is the
    enforcement (``check_paths`` and the redaction readers match ``deny_patterns`` /
    ``redact_patterns`` and nothing else). Something invisible to what the gate guards must be
    invisible to the gate too, or the two sides of one hop answer differently about a
    byte-identical request -- which is exactly what happened: the client computed "configured"
    from the subclass and demanded an echo, the server rebuilt a plain config from the wire,
    computed "default", sent none, and every turn was refused with no server-side fix available.
    """

    defaults = base()
    return all(
        getattr(config, declared.name) == getattr(defaults, declared.name)
        for declared in fields(base)
    )


def str_tuple(
    value: Any,
    *,
    type_error: str,
    empty_error: str | None = None,
    normalize: bool = False,
    error: Callable[[str], Exception] = ValueError,
) -> tuple[str, ...]:
    """Validate a JSON string array and return it as a tuple.

    A bare string (or any non-array) is rejected with ``type_error``. With
    ``normalize=True`` each item is stripped, lowercased, and empties are
    dropped (domain lists). Otherwise, when ``empty_error`` is given, an
    empty/whitespace item raises it. ``error`` selects the exception type.
    """
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise error(type_error)
    if not all(isinstance(item, str) for item in value):
        raise error(type_error)
    items = tuple(value)
    if normalize:
        return tuple(item.strip().lower() for item in items if item.strip())
    if empty_error is not None and any(not item.strip() for item in items):
        raise error(empty_error)
    return items


def dedupe(values: Iterable[str]) -> tuple[str, ...]:
    """Order-preserving de-duplication."""
    return tuple(dict.fromkeys(values))
