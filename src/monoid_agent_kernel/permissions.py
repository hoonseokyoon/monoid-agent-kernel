from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import pathspec

from monoid_agent_kernel._policy_util import dedupe, str_tuple
from monoid_agent_kernel.errors import PermissionDenied
from monoid_agent_kernel.workspace.paths import normalize_workspace_path

PermissionOperation = Literal["read", "write", "artifact", "run"]

# Gitignore wildcard matching, from `pathspec`. Chosen over the stdlib because the stdlib cannot
# express these semantics on this project's floor: `PurePath.full_match` and `glob.translate` are
# 3.13+ and `requires-python` is >=3.11, `fnmatch` lets `*` cross `/`, and `PurePath.match` is what
# was wrong. Supporting 3.13 natively and hand-rolling a backport would be two implementations of
# one security control -- the defect shape this codebase keeps paying for. One library, every
# version, and the pattern language is the one operators already know from `.gitignore`.
#
# `gitignore` and not `gitwildmatch`: the latter is deprecated in pathspec 1.x and the two disagree
# on `internal/*` against `internal/deep/a.txt` (deprecated: covered, current: not). The dependency
# is pinned to 1.x for the same reason -- see the note in `pyproject.toml`.
_PATTERN_STYLE = "gitignore"


def _literal_negation(pattern: str) -> str:
    """Keep a leading ``!`` literal instead of letting it mean "un-match".

    ``PurePath.match`` had no negation, so ``!secrets/x`` has always meant *a file named that*.
    ``gitwildmatch`` reads it as negation, and adopting that silently would have handed every
    ``deny_patterns`` list a way to punch holes in itself -- with the result depending on pattern
    *order*, which ``PermissionPolicy.merged`` does not preserve as meaningful: it concatenates two
    policies and de-duplicates, treating patterns as a set. A set of order-dependent rules is not a
    policy. So negation stays off, ``!`` keeps the meaning it already had, and
    ``PermissionPolicy.from_json`` rejects the pattern outright so an operator who wanted negation
    is told rather than left with a rule that quietly does nothing.
    """
    return "\\" + pattern if pattern.startswith("!") else pattern


@lru_cache(maxsize=512)
def _compiled(patterns: tuple[str, ...]) -> pathspec.PathSpec:
    return pathspec.PathSpec.from_lines(_PATTERN_STYLE, [_literal_negation(p) for p in patterns])


def matches_path_patterns(rel: str, patterns: tuple[str, ...]) -> bool:
    """True if a workspace-relative path matches any pattern, with ``.gitignore`` semantics.

    Four call sites share this function and they do **not** agree on which direction is safe:
    ``deny_patterns`` and a capability lease's ``denied_paths`` fail closed by matching, while a
    lease's ``allowed_paths`` fails closed by *not* matching. So this function decides nothing on
    their behalf, and in particular it does not absorb errors — ``normalize_workspace_path`` still
    raises ``WorkspaceError`` on an absolute path or a ``..`` traversal, because every caller
    already handles that and one of them **depends** on it: ``public_view._is_path_redacted``
    catches it and returns *redacted*, which is how ``x/../secrets/creds.txt`` stays out of the
    event stream. Swallowing it here would silently reopen that leak.

    Not ``Workspace.glob``'s twin, though both match a path against a glob. That one is ``fnmatch``
    over a discovery API the model calls to find files, where matching too much returns files it
    could already list. This one decides access, so the two want opposite defaults.
    """
    if not patterns:
        return False
    return _compiled(patterns).match_file(normalize_workspace_path(rel))


@dataclass(frozen=True)
class PermissionPolicy:
    """Workspace path patterns an operator declares for a run.

    Both lists are **`.gitignore` wildcard patterns**, matched against the workspace-relative path:

    - ``.env`` / ``*.key`` — no slash, so it matches that name at **any** depth (``a/b/.env`` too).
    - ``internal/**`` — everything under ``internal/``, at any depth. Anchored at the workspace
      root: it does not match ``vendor/internal/x``. Nor does it match ``internal`` itself; write
      ``internal`` for the directory and its contents.
    - ``**/id_rsa`` — that name anywhere, including at the root.

    Negation (a leading ``!``) is **not** supported and is rejected here. It would make the result
    depend on pattern order, and ``merged`` combines two policies by concatenating and
    de-duplicating — set semantics, under which an order-dependent rule has no defined meaning.

    Until v0.20 these were matched with ``PurePath.match``, where ``**`` behaved as a single ``*``:
    ``internal/**`` covered one level and missed ``internal/deep/x`` entirely, while also matching
    ``vendor/internal/x``. Both of the ``**`` examples above were already in the documentation and
    neither did what it says.
    """

    deny_patterns: tuple[str, ...] = ()
    redact_patterns: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, payload: dict[str, Any] | None) -> PermissionPolicy:
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("permission_policy must be an object")
        return cls(
            deny_patterns=cls._patterns(payload.get("deny_patterns") or ()),
            redact_patterns=cls._patterns(payload.get("redact_patterns") or ()),
        )

    @staticmethod
    def _patterns(raw: Any) -> tuple[str, ...]:
        patterns = str_tuple(
            raw,
            type_error="expected an array of path patterns",
            empty_error="empty path pattern is not allowed",
        )
        # Rejected at the config boundary rather than at match time. `matches_path_patterns` keeps
        # `!` literal -- the meaning it had before this used gitignore semantics -- so a pattern
        # arriving from somewhere other than a manifest is inert rather than dangerous. But inert is
        # silent, and an operator who wrote a negation meant something by it, so the place where
        # operator configuration enters says so out loud.
        negations = [pattern for pattern in patterns if pattern.startswith("!")]
        if negations:
            raise ValueError(
                "negated path patterns are not supported "
                f"(pattern order would decide the result): {negations[0]!r}"
            )
        return patterns

    def to_json(self) -> dict[str, list[str]]:
        return {
            "deny_patterns": list(self.deny_patterns),
            "redact_patterns": list(self.redact_patterns),
        }

    def merged(
        self,
        *,
        deny_patterns: tuple[str, ...] = (),
        redact_patterns: tuple[str, ...] = (),
    ) -> PermissionPolicy:
        return PermissionPolicy(
            deny_patterns=dedupe((*self.deny_patterns, *deny_patterns)),
            redact_patterns=dedupe((*self.redact_patterns, *redact_patterns)),
        )

    def check_paths(self, operation: PermissionOperation, paths: tuple[str, ...]) -> None:
        if operation in {"artifact", "run"}:
            return
        for raw in paths:
            rel = normalize_workspace_path(raw)
            if self.is_path_denied(rel):
                raise PermissionDenied(f"{operation} denied for path: {rel}")

    def is_path_denied(self, rel: str) -> bool:
        return matches_path_patterns(rel, self.deny_patterns)

    def is_path_redacted(self, rel: str) -> bool:
        return matches_path_patterns(rel, self.redact_patterns)
