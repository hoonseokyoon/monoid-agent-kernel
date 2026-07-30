from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import pathspec

from monoid_agent_kernel._policy_util import dedupe, str_tuple
from monoid_agent_kernel.errors import PermissionDenied
from monoid_agent_kernel.workspace.paths import normalize_workspace_path

PermissionOperation = Literal["read", "write", "artifact", "run"]

# Gitignore-style wildcard matching, from `pathspec`. Chosen over the stdlib because the stdlib cannot
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


def _canonical_pattern(pattern: str) -> str | None:
    """Canonicalize compatibility spellings before handing a line to pathspec."""
    # A double leading slash is an ambiguous/UNC-style anchor, not the documented single-root
    # spelling. Collapsing it to one slash can turn an inert allow-list typo (for example ``//**``)
    # into a match-all policy, so keep it a no-op at the low-level matcher and reject it at every
    # policy/config boundary through the ``None`` result.
    if pattern.startswith(("//", "\\\\")):
        return None
    anchored = pattern.startswith("/")
    body = pattern[1:] if anchored else pattern
    # Only this documented relative prefix is a compatibility spelling. Removing dot or empty
    # segments elsewhere can turn an inert allow pattern such as ``/./**`` into match-all.
    if not anchored and body.startswith("./"):
        body = body[2:]
    # Workspace input treats every backslash as a path separator. In a pathspec pattern it is an
    # escape, so accepting ``\secret`` would retarget an old inert allow rule to ``secret``. The
    # sole source-level exception is the documented unanchored ``\!`` wire spelling; the config
    # parser removes that escape before storing the internal pattern.
    wire_literal_negation = not anchored and body.startswith("\\!")
    remaining_body = body[2:] if wire_literal_negation else body
    if (body.startswith("\\") and not wire_literal_negation) or "\\" in remaining_body:
        return None
    # ``normalize_workspace_path`` rejects a literal drive prefix on every platform. Accepting one
    # here would leave a deny/redact rule permanently inert and conceal a likely absolute-path typo.
    if re.match(r"^[A-Za-z]:", body):
        return None
    segments = body.split("/")
    if segments and segments[-1] == "":
        segments.pop()
    if not segments or any(segment in {"", ".", ".."} for segment in segments):
        return None
    pattern = "/".join(segments)
    if anchored:
        pattern = "/" + pattern
    # pathspec 1.x's basic matcher compiles a root-only line as ``.`` (match everything), unlike
    # both PurePath and Git. Whitespace variants are included because pathspec strips the suffix.
    if pattern.startswith("/") and not pattern.strip("/ \t"):
        return None
    # PurePath treated ``secrets/`` as the directory node. Removing the one trailing directory
    # marker preserves the node and adds the documented subtree under basic gitignore matching.
    return pattern or None


def _first_pattern_segment_offset(pattern: str) -> int:
    for match in re.finditer(r"[^/]+", pattern):
        if match.group() != ".":
            return match.start()
    return len(pattern)


def _pathspec_pattern(pattern: str) -> str | None:
    """Keep leading gitignore control characters literal.

    ``PurePath.match`` had no negation, so ``!secrets/x`` has always meant *a file named that*.
    Gitignore matching reads it as negation, and adopting that silently would have handed every
    ``deny_patterns`` list a way to punch holes in itself -- with the result depending on pattern
    *order*, which ``PermissionPolicy.merged`` does not preserve as meaningful: it concatenates two
    policies and de-duplicates, treating patterns as a set. A set of order-dependent rules is not a
    policy. So negation stays off, ``!`` keeps the meaning it already had, and
    operator configuration rejects an unescaped pattern outright so an operator who wanted
    negation is told rather than left with a rule that quietly does nothing. ``\\!`` is the explicit
    configuration spelling for the literal form.

    A leading ``#`` was likewise a literal under ``PurePath.match`` but is a comment in gitignore
    syntax. Escape both controls before compilation so adopting the wildcard language does not make
    an existing deny or redact rule disappear.
    """
    pattern = _canonical_pattern(pattern)
    if pattern is None:
        return None

    trailing_whitespace = pattern[len(pattern.rstrip()) :]
    if any(char != " " for char in trailing_whitespace):
        # pathspec strips Unicode whitespace with ``rstrip()`` but only has an escape exception for
        # ASCII space. Accepting another trailing whitespace character would silently retarget an
        # allow rule (for example ``public\N{NO-BREAK SPACE}``) to ``public``.
        raise ValueError("non-ASCII trailing whitespace is not supported in path patterns")

    # Basic gitignore parsing strips unescaped trailing spaces. They were literal under
    # PurePath, so encode it as character classes instead of silently retargeting the rule.
    suffix_length = len(pattern) - len(pattern.rstrip(" "))
    if suffix_length:
        base = pattern[:-suffix_length]
        if not base.endswith("\\"):
            pattern = base + "\\ " * suffix_length

    return "\\" + pattern if pattern.startswith(("!", "#")) else pattern


@lru_cache(maxsize=512)
def _compiled(patterns: tuple[str, ...]) -> pathspec.PathSpec:
    compiled_patterns = [
        compiled
        for pattern in patterns
        if (compiled := _pathspec_pattern(pattern)) is not None
    ]
    # pathspec 1.1 emits Python's possible-set-operation FutureWarnings for valid glob classes such
    # as ``[[]`` (a literal ``[``) and ``[a&&b]``, even though the generated regex preserves today's
    # glob meaning. Keep warnings-as-errors applications usable while narrowing the suppression to
    # that warning family at the dependency's compile frame.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"Possible (?:nested set|set (?:difference|intersection|union|symmetric difference)) "
                r"at position \d+"
            ),
            category=FutureWarning,
            module=r"pathspec\.pattern",
        )
        return pathspec.PathSpec.from_lines(
            _PATTERN_STYLE,
            compiled_patterns,
        )


def validate_internal_path_patterns(raw: Any) -> tuple[str, ...]:
    """Validate already-internal patterns, retaining a literal leading ``!``."""
    patterns = str_tuple(
        raw,
        type_error="expected an array of path patterns",
        empty_error="empty path pattern is not allowed",
    )
    for source in patterns:
        if any(ord(char) < 32 or ord(char) == 127 for char in source):
            raise ValueError(f"control characters are not allowed in path patterns: {source!r}")
        canonical = _canonical_pattern(source)
        if canonical is None:
            raise ValueError(f"path pattern must name a workspace path: {source!r}")
        if canonical.startswith("\\!"):
            raise ValueError(
                "escaped leading ! is a configuration spelling; "
                f"direct patterns use a literal leading !: {source!r}"
            )
    try:
        _compiled(patterns)
    except (ValueError, re.error) as exc:
        raise ValueError(f"invalid path pattern: {exc}") from exc
    return patterns


def parse_path_patterns(raw: Any) -> tuple[str, ...]:
    """Parse and validate operator-supplied path-pattern configuration.

    A leading ``\\!`` is the wire spelling for a literal leading exclamation mark. The internal
    spelling remains ``!`` so direct Python construction and old matcher behavior round-trip.
    """
    patterns = str_tuple(
        () if raw is None else raw,
        type_error="expected an array of path patterns",
        empty_error="empty path pattern is not allowed",
    )
    parsed: list[str] = []
    for source in patterns:
        if any(ord(char) < 32 or ord(char) == 127 for char in source):
            raise ValueError(f"control characters are not allowed in path patterns: {source!r}")
        canonical = _canonical_pattern(source)
        if canonical is None:
            raise ValueError(f"path pattern must name a workspace path: {source!r}")
        if canonical.startswith("\\!"):
            marker = _first_pattern_segment_offset(source)
            source = source[:marker] + source[marker + 1 :]
            canonical = canonical[1:]
        elif canonical.startswith("!"):
            raise ValueError(
                "negated path patterns are not supported "
                f"(pattern order would decide the result): {source!r}"
            )
        parsed.append(source)

    return validate_internal_path_patterns(tuple(parsed))


def serialize_path_patterns(patterns: tuple[str, ...]) -> list[str]:
    """Encode internal literal-negation patterns for an unambiguous JSON round-trip."""
    serialized: list[str] = []
    for source in patterns:
        pattern = _canonical_pattern(source)
        if pattern is None:
            raise ValueError(f"path pattern must name a workspace path: {source!r}")
        if pattern.startswith("!"):
            marker = _first_pattern_segment_offset(source)
            source = source[:marker] + "\\" + source[marker:]
        serialized.append(source)
    return serialized


def matches_path_patterns(
    rel: str,
    patterns: tuple[str, ...],
) -> bool:
    """True if a workspace-relative path matches any independent gitignore-style pattern.

    Four call sites share this function. ``deny_patterns`` and a tool binding's ``denied_paths``
    fail closed by matching, while a binding's ``allowed_paths`` fails closed by *not* matching.
    This matcher therefore stays lexical; filesystem aliases require a workspace-root- and
    volume-aware backend decision. The function does not absorb errors —
    ``normalize_workspace_path`` still raises ``WorkspaceError`` on an absolute path or a ``..``
    traversal, because every caller already handles that and one of them **depends** on it:
    ``public_view._is_path_redacted``
    catches it and returns *redacted*, which is how ``x/../secrets/creds.txt`` stays out of the
    event stream. Swallowing it here would silently reopen that leak.

    Not ``Workspace.glob``'s twin, though both match a path against a glob. That one is ``fnmatch``
    over a discovery API the model calls to find files, where matching too much returns files it
    could already list. This one decides access, so the two want opposite defaults.
    """
    if not patterns:
        return False
    invalid_relative_root = next(
        (
            pattern
            for pattern in patterns
            if _canonical_pattern(pattern) is None and not pattern.startswith(("/", "\\\\"))
        ),
        None,
    )
    if invalid_relative_root is not None:
        raise ValueError(f"path pattern must name a workspace path: {invalid_relative_root!r}")
    if any(
        ord(char) < 32 or ord(char) == 127
        for pattern in patterns
        for char in pattern
    ):
        raise ValueError("control characters are not allowed in path patterns")
    normalized = normalize_workspace_path(rel)
    # ``.`` is the synthetic workspace root, not a workspace entry. PurePath matched none of the
    # old patterns against it; pathspec treats broad globs such as ``*`` and ``**`` as match-all.
    # Preserve the fail-closed allow-list behavior for root cwd/list operations.
    if normalized == ".":
        return False
    return _compiled(patterns).match_file(normalized)


@dataclass(frozen=True)
class PermissionPolicy:
    """Workspace path patterns an operator declares for a run.

    Both lists use **gitignore-style wildcard syntax**, matched independently against the
    workspace-relative path:

    - ``.env`` / ``*.key`` — no slash, so it matches that name at **any** depth (``a/b/.env`` too).
    - ``internal/**`` — everything under ``internal/``, at any depth. Anchored at the workspace
      root: it does not match ``vendor/internal/x``. Nor does it match ``internal`` itself; write
      ``internal`` for the directory and its contents.
    - ``**/id_rsa`` — that name anywhere, including at the root.

    Negation is **not** supported. Operator configuration rejects an unescaped leading ``!``
    because it would make the result depend on pattern order, while ``merged`` combines policies
    with set semantics. Write ``\\!`` for a literal leading exclamation mark. Direct Python tuples
    retain their historical literal ``!`` meaning and JSON serialization applies that escape.

    Until v0.20 these were matched with ``PurePath.match``, where ``**`` behaved as a single ``*``:
    ``internal/**`` covered one level and missed ``internal/deep/x`` entirely, while also matching
    ``vendor/internal/x``. Both of the ``**`` examples above were already in the documentation and
    neither did what it says.
    """

    deny_patterns: tuple[str, ...] = ()
    redact_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "deny_patterns", validate_internal_path_patterns(self.deny_patterns)
        )
        object.__setattr__(
            self, "redact_patterns", validate_internal_path_patterns(self.redact_patterns)
        )

    @classmethod
    def from_json(cls, payload: dict[str, Any] | None) -> PermissionPolicy:
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("permission_policy must be an object")
        return cls(
            deny_patterns=cls._patterns(payload.get("deny_patterns")),
            redact_patterns=cls._patterns(payload.get("redact_patterns")),
        )

    @staticmethod
    def _patterns(raw: Any) -> tuple[str, ...]:
        return parse_path_patterns(raw)

    def to_json(self) -> dict[str, list[str]]:
        return {
            "deny_patterns": serialize_path_patterns(self.deny_patterns),
            "redact_patterns": serialize_path_patterns(self.redact_patterns),
        }

    def merged(
        self,
        *,
        deny_patterns: tuple[str, ...] = (),
        redact_patterns: tuple[str, ...] = (),
    ) -> PermissionPolicy:
        # The incoming tuples are another operator-config boundary (the CLI flags use it). Validate
        # those without reinterpreting an already constructed policy: direct Python callers could
        # historically use a leading ``!`` as a literal filename, and the matcher preserves that
        # meaning deliberately.
        new_deny_patterns = self._patterns(deny_patterns)
        new_redact_patterns = self._patterns(redact_patterns)
        return PermissionPolicy(
            deny_patterns=dedupe((*self.deny_patterns, *new_deny_patterns)),
            redact_patterns=dedupe((*self.redact_patterns, *new_redact_patterns)),
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
