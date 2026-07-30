"""What a `PermissionPolicy` path pattern means, pinned case by case.

`matches_path_patterns` used `PurePath.match`, where `**` is a single `*` matched right-to-left. So
`internal/**` covered one level and missed `internal/deep/x`, while also matching
`vendor/internal/x`. Both of those are wrong in a way an operator cannot see from the outside, and
**both `**` patterns in the documentation were affected** -- `internal/**` in CONTRACTS/EMBEDDING
and `**/id_rsa` in the README example, the latter never matching a bare `id_rsa` at the root.

Nothing caught it because every fixture in the repo used one-level paths. Hence a table: the depths
are the point, not an afterthought, and the rows that describe *not* matching are as load-bearing
as the rows that match.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import pytest

from monoid_agent_kernel.core.tool_surface import ToolScope
from monoid_agent_kernel.errors import PermissionDenied, WorkspaceError
from monoid_agent_kernel.permissions import PermissionPolicy, _compiled, matches_path_patterns

# (pattern, path, matches) -- read as "an operator who writes <pattern> means <path> is covered".
CASES: tuple[tuple[str, str, bool], ...] = (
    # --- a trailing `**`: everything under a directory, at any depth, anchored at the root -------
    ("internal/**", "internal/a.txt", True),
    ("internal/**", "internal/deep/a.txt", True),  # the reported bug: was False
    ("internal/**", "internal/deep/deeper/a.txt", True),
    ("internal/**", "internal", False),  # the directory itself is not "under" it
    ("internal/**", "vendor/internal/a.txt", False),  # the other half: was True
    ("internal/**", "internals/a.txt", False),
    # --- a bare name: no slash, so any depth (what `.env` / `*.key` rely on) ---------------------
    (".env", ".env", True),
    (".env", "deep/.env", True),
    (".env", "a/b/c/.env", True),
    (".env", ".envx", False),
    ("*.key", "a.key", True),
    ("*.key", "deep/a.key", True),
    ("*.key", "key", False),
    # --- a bare directory: its node and subtree at any depth (broader than PurePath.match) --------
    ("internal", "internal", True),
    ("internal", "internal/a.txt", True),
    ("internal", "vendor/internal/a.txt", True),
    # --- a leading `**`: that name anywhere, root included ---------------------------------------
    ("**/id_rsa", "id_rsa", True),  # was False -- the README example missed the root
    ("**/id_rsa", ".ssh/id_rsa", True),
    ("**/id_rsa", "a/b/.ssh/id_rsa", True),
    ("**/id_rsa", "id_rsa_backup", False),
    # --- `**` on both ends: the escape hatch for "this directory wherever it appears" ------------
    ("**/secrets/**", "secrets/a", True),
    ("**/secrets/**", "vendor/secrets/a", True),
    ("**/secrets/**", "x/y/secrets/a/b", True),
    ("**/secrets", "secrets", True),
    ("**/secrets", "vendor/secrets", True),
    ("**/secrets", "x/y/secrets/a/b", True),
    # --- the reported case ------------------------------------------------------------------------
    ("secrets/**", "secrets/a", True),
    ("secrets/**", "secrets/a/b", True),  # was False
    ("secrets/**", "x/y/secrets/a", False),  # was True
    # --- compatibility spellings canonicalize before compilation --------------------------------
    ("./secrets/**", "secrets/deep/a", True),
    ("secrets/", "secrets", True),
    ("secrets/", "secrets/deep/a", True),
    # --- non-ASCII segments: the matcher is character-based, not byte-based ----------------------
    ("비밀/**", "비밀/문서.txt", True),
    ("비밀/**", "비밀/깊은/문서.txt", True),
    ("비밀/**", "공개/문서.txt", False),
    ("**/비밀문서.txt", "비밀문서.txt", True),
    ("**/비밀문서.txt", "a/b/비밀문서.txt", True),
    # --- a single `*` still stops at a separator --------------------------------------------------
    # This row is why the dependency is pinned to pathspec 1.x. Under 0.12.1 it is True (a matched
    # directory carries its subtree, which is what git itself does); under 1.x it is False (a
    # grandchild is not a direct child). On a deny list the difference is a hole that opens from an
    # upgrade alone, so the range admits only one answer. Write `internal/**` for the subtree.
    ("internal/*", "internal/a.txt", True),
    ("internal/*", "internal/deep/a.txt", False),
    # --- gitignore control characters keep their old literal meaning ------------------------------
    ("!odd", "!odd", True),
    ("!odd", "odd", False),
    ("#credentials", "#credentials", True),
    ("#credentials", "credentials", False),
)


@pytest.mark.parametrize(("pattern", "path", "expected"), CASES)
def test_pattern_matches(pattern: str, path: str, expected: bool) -> None:
    assert matches_path_patterns(path, (pattern,)) is expected


@pytest.mark.parametrize(("pattern", "path", "expected"), CASES)
def test_a_pattern_means_the_same_thing_on_every_caller(pattern: str, path: str, expected: bool) -> None:
    """One function backs four call sites and they do not agree on which direction is safe:
    `deny_patterns` and a binding's `denied_paths` fail closed by matching, while its
    `allowed_paths` fails closed by *not* matching. That is why semantics must not vary by caller."""
    assert PermissionPolicy(deny_patterns=(pattern,)).is_path_denied(path) is expected
    assert PermissionPolicy(redact_patterns=(pattern,)).is_path_redacted(path) is expected


def test_no_pattern_matches_nothing() -> None:
    assert matches_path_patterns("anything/at/all", ()) is False


@pytest.mark.parametrize("pattern", ["*", "*/", "**", "**/", "**/*", "/**"])
def test_the_synthetic_workspace_root_matches_no_pattern(pattern: str) -> None:
    assert matches_path_patterns(".", (pattern,)) is False
    assert matches_path_patterns("", (pattern,)) is False


def test_any_pattern_in_the_list_is_enough() -> None:
    patterns = ("internal/**", "*.key")
    assert matches_path_patterns("internal/deep/a.txt", patterns) is True
    assert matches_path_patterns("elsewhere/a.key", patterns) is True
    assert matches_path_patterns("elsewhere/a.txt", patterns) is False


def test_pattern_lists_are_compiled_independently() -> None:
    """The compiled spec is cached by pattern tuple. A cache keyed carelessly would let one
    policy's answer leak into another's, which on a security control is the worst kind of bug."""
    assert matches_path_patterns("secrets/a/b", ("secrets/**",)) is True
    assert matches_path_patterns("secrets/a/b", ("public/**",)) is False
    assert matches_path_patterns("secrets/a/b", ("secrets/**",)) is True
    assert matches_path_patterns("secrets/a/b", ("secrets/**", "public/**")) is True


def test_backslashes_normalize_before_matching() -> None:
    assert matches_path_patterns("internal\\deep\\a.txt", ("internal/**",)) is True
    assert matches_path_patterns("./internal/a.txt", ("internal/**",)) is True


@pytest.mark.parametrize(
    "pattern",
    [
        "/",
        "/ ",
        "/.",
        "/./",
        "//./",
        "//**",
        "//foo",
        "/./**",
        "/.//**",
        "/**//",
        "/**/./**",
        "/../**",
        "/**/../**",
        "/C:/**",
        r"\\server\share\**",
        r"/\\server\share\**",
        r"/\secret",
        r"/\!odd",
    ],
)
def test_inert_anchored_patterns_remain_no_op_at_the_low_level(pattern: str) -> None:
    """Canonicalization must not turn an inert anchor into a low-level match-all program."""
    assert matches_path_patterns("secrets/key.txt", (pattern,)) is False


@pytest.mark.parametrize(
    "pattern",
    [
        "",
        ".",
        "./",
        ".//./",
        r"./\\server\share\**",
        r"\secret",
        r"./\secret",
        r"foo/\secret",
        "secrets\\",
    ],
)
def test_invalid_relative_patterns_raise_at_the_low_level(pattern: str) -> None:
    with pytest.raises(ValueError, match="must name a workspace path"):
        matches_path_patterns("secrets/key.txt", (pattern,))


@pytest.mark.parametrize("pattern", ["safe ", "safe  ", "foo[ "])
def test_trailing_space_in_a_pattern_remains_literal(pattern: str) -> None:
    compiled = _compiled((pattern,))

    assert compiled.match_file(pattern) is True
    assert compiled.match_file(pattern.rstrip()) is False


# --- the raise is load-bearing ------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "C:/Windows/x", "x/../secrets/creds.txt", "../x", "safe\n", "safe\x1f"],
)
def test_an_unnormalizable_path_still_raises(path: str) -> None:
    """`public_view._is_path_redacted` catches `WorkspaceError` and returns *redacted*, which is the
    only reason `x/../secrets/creds.txt` stays out of the event stream. An earlier draft of this
    change absorbed the error and returned False, silently reopening that leak -- so the raise is
    pinned rather than left to whoever reads the code next.
    """
    with pytest.raises(WorkspaceError):
        matches_path_patterns(path, ("secrets/**",))


def test_the_redaction_path_still_fails_closed_on_a_traversal() -> None:
    from monoid_agent_kernel.public_view import REDACTED_PATH, public_path

    policy = PermissionPolicy(redact_patterns=("secrets/**",))
    assert public_path("x/../secrets/creds.txt", policy) == REDACTED_PATH
    assert public_path("/etc/passwd", policy) == REDACTED_PATH


def test_check_paths_denies_a_deep_path_under_a_denied_tree() -> None:
    """The whole point, at the enforcement boundary: `internal/**` denied one level before this."""
    policy = PermissionPolicy(deny_patterns=("internal/**",))

    policy.check_paths("read", ("public/a.txt",))
    with pytest.raises(PermissionDenied):
        policy.check_paths("read", ("internal/a.txt",))
    with pytest.raises(PermissionDenied):
        policy.check_paths("read", ("internal/deep/deeper/a.txt",))


def test_a_trailing_directory_pattern_denies_the_node_and_subtree() -> None:
    policy = PermissionPolicy.from_json({"deny_patterns": ["secrets/"]})

    with pytest.raises(PermissionDenied):
        policy.check_paths("write", ("secrets",))
    with pytest.raises(PermissionDenied):
        policy.check_paths("write", ("secrets/deep/a.txt",))


def test_production_directory_patterns_deny_nodes_and_subtrees() -> None:
    policy = PermissionPolicy.from_json({"deny_patterns": [".ssh", ".git"]})

    for path in (".ssh", ".ssh/keys/deep/id_ed25519", ".git", ".git/refs/heads/main"):
        with pytest.raises(PermissionDenied):
            policy.check_paths("write", (path,))


# --- negation ------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["deny_patterns", "redact_patterns"])
def test_from_json_rejects_a_negated_pattern(key: str) -> None:
    """Loud at the config boundary. Adopting gitignore negation silently would have handed every
    deny list a way to punch holes in itself, with the result depending on pattern *order* --
    and `merged` combines policies as a de-duplicated set, where order has no defined meaning."""
    with pytest.raises(ValueError, match="negated path patterns are not supported"):
        PermissionPolicy.from_json({key: ["secrets/**", "!secrets/README.md"]})


@pytest.mark.parametrize("key", ["allowed_paths", "denied_paths"])
def test_tool_scope_from_json_rejects_a_negated_pattern(key: str) -> None:
    with pytest.raises(ValueError, match="negated path patterns are not supported"):
        ToolScope.from_json({key: ["*", "!secrets/**"]})


@pytest.mark.parametrize(
    "pattern",
    [
        "/",
        "/ ",
        ".",
        "./",
        "/.",
        "/./",
        "//./",
        "//**",
        "//foo",
        "/./**",
        "/.//**",
        "/**//",
        "/**/./**",
        "**//**",
        "foo//bar",
        "foo/./bar",
        "../**",
        "a/../**",
        "a/../../**",
        "C:/**",
        "c:/secret/**",
        "./C:/**",
        "/C:/**",
        "C:foo",
        r"\\server\share\**",
        r"./\\server\share\**",
        r"/\\server\share\**",
        r"\secret",
        r"./\secret",
        r"/\secret",
        r"foo/\secret",
        r"/\!odd",
        "secrets\\",
    ],
)
@pytest.mark.parametrize(
    ("factory", "key"),
    [
        (PermissionPolicy.from_json, "deny_patterns"),
        (PermissionPolicy.from_json, "redact_patterns"),
        (ToolScope.from_json, "allowed_paths"),
        (ToolScope.from_json, "denied_paths"),
    ],
)
def test_config_rejects_inert_or_ambiguous_patterns(
    factory: Callable[[dict[str, list[str]]], object], key: str, pattern: str
) -> None:
    with pytest.raises(ValueError, match="must name a workspace path"):
        factory({key: [pattern]})


@pytest.mark.parametrize(
    "pattern",
    [
        "",
        ".",
        "./",
        "/",
        "/.",
        "/./",
        "//**",
        "//foo",
        "/./**",
        "/**//",
        "/**/./**",
        "**//**",
        "foo//bar",
        "../**",
        "a/../**",
        "C:/**",
        "./C:/**",
        "/C:/**",
        "C:foo",
        r"\\server\share\**",
        r"./\\server\share\**",
        r"/\\server\share\**",
        r"\secret",
        r"./\secret",
        r"/\secret",
        r"foo/\secret",
        r"/\!odd",
        "secrets\\",
    ],
)
@pytest.mark.parametrize(
    "factory",
    [
        lambda pattern: PermissionPolicy(deny_patterns=(pattern,)),
        lambda pattern: PermissionPolicy(redact_patterns=(pattern,)),
        lambda pattern: ToolScope(allowed_paths=(pattern,)),
        lambda pattern: ToolScope(denied_paths=(pattern,)),
    ],
)
def test_direct_objects_reject_inert_or_ambiguous_patterns(
    factory: Callable[[str], object], pattern: str
) -> None:
    with pytest.raises(ValueError, match="path pattern|empty path pattern"):
        factory(pattern)


@pytest.mark.parametrize("pattern", ["[z-a]", "[a-Z]"])
@pytest.mark.parametrize(
    ("factory", "key"),
    [
        (PermissionPolicy.from_json, "deny_patterns"),
        (PermissionPolicy.from_json, "redact_patterns"),
        (ToolScope.from_json, "allowed_paths"),
        (ToolScope.from_json, "denied_paths"),
    ],
)
def test_config_rejects_malformed_patterns(
    factory: Callable[[dict[str, list[str]]], object], key: str, pattern: str
) -> None:
    with pytest.raises(ValueError, match="invalid path pattern"):
        factory({key: [pattern]})


@pytest.mark.parametrize("pattern", ["safe\n", "safe\t", "safe\x7f"])
def test_config_rejects_control_characters(pattern: str) -> None:
    with pytest.raises(ValueError, match="control characters"):
        PermissionPolicy.from_json({"deny_patterns": [pattern]})


@pytest.mark.parametrize("pattern", ["public\u00a0", "public\u0085", "public\u2003", "public\u2028"])
@pytest.mark.parametrize(
    ("factory", "key"),
    [
        (PermissionPolicy.from_json, "deny_patterns"),
        (PermissionPolicy.from_json, "redact_patterns"),
        (ToolScope.from_json, "allowed_paths"),
        (ToolScope.from_json, "denied_paths"),
    ],
)
def test_config_rejects_unicode_trailing_whitespace(
    factory: Callable[[dict[str, list[str]]], object], key: str, pattern: str
) -> None:
    with pytest.raises(ValueError, match="trailing whitespace"):
        factory({key: [pattern]})


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PermissionPolicy(deny_patterns=("public\u00a0",)),
        lambda: PermissionPolicy(redact_patterns=("public\u00a0",)),
        lambda: ToolScope(allowed_paths=("public\u00a0",)),
        lambda: ToolScope(denied_paths=("public\u00a0",)),
    ],
)
def test_direct_objects_reject_unicode_trailing_whitespace(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(ValueError, match="trailing whitespace"):
        factory()


def test_literal_negation_has_an_unambiguous_policy_json_round_trip() -> None:
    policy = PermissionPolicy(
        deny_patterns=("!odd", "./secrets/**"), redact_patterns=("!private/", "private/")
    )

    payload = policy.to_json()
    assert payload == {
        "deny_patterns": [r"\!odd", "./secrets/**"],
        "redact_patterns": [r"\!private/", "private/"],
    }
    assert PermissionPolicy.from_json(payload) == policy


def test_literal_negation_has_an_unambiguous_tool_scope_json_round_trip() -> None:
    scope = ToolScope(
        allowed_paths=("!odd", "./public/**"), denied_paths=("!private/", "private/")
    )

    payload = scope.to_json()
    assert payload["allowed_paths"] == [r"\!odd", "./public/**"]
    assert payload["denied_paths"] == [r"\!private/", "private/"]
    assert ToolScope.from_json(payload) == scope


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PermissionPolicy(deny_patterns=(r"\!odd",)),
        lambda: PermissionPolicy(redact_patterns=(r"./\!odd",)),
        lambda: ToolScope(allowed_paths=(r"\!odd",)),
        lambda: ToolScope(denied_paths=(r"./\!odd",)),
    ],
)
def test_direct_objects_reject_the_wire_only_literal_negation_spelling(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(ValueError, match="configuration spelling"):
        factory()


def test_from_json_still_accepts_the_documented_patterns() -> None:
    policy = PermissionPolicy.from_json(
        {"deny_patterns": ["internal/**"], "redact_patterns": [".env", "*.key", "**/id_rsa"]}
    )

    assert policy.deny_patterns == ("internal/**",)
    assert policy.redact_patterns == (".env", "*.key", "**/id_rsa")
    assert policy.is_path_denied("internal/deep/a.txt") is True
    assert policy.is_path_redacted("id_rsa") is True


def test_merged_keeps_both_policies_patterns() -> None:
    merged = PermissionPolicy(deny_patterns=("internal/**",)).merged(deny_patterns=("secrets/**",))

    assert merged.is_path_denied("internal/deep/a.txt") is True
    assert merged.is_path_denied("secrets/deep/a.txt") is True


@pytest.mark.parametrize("key", ["deny_patterns", "redact_patterns"])
def test_merged_rejects_a_negated_pattern(key: str) -> None:
    """CLI path flags enter through ``merged`` rather than ``from_json``."""
    with pytest.raises(ValueError, match="negated path patterns are not supported"):
        PermissionPolicy().merged(**{key: ("!secrets/**",)})


def test_merged_does_not_reinterpret_an_existing_literal_negation() -> None:
    """Direct Python construction retains the literal meaning supported before v0.20."""
    merged = PermissionPolicy(deny_patterns=("!odd",)).merged(deny_patterns=("internal/**",))

    assert merged.deny_patterns == ("!odd", "internal/**")
    assert merged.is_path_denied("!odd") is True


def test_the_pattern_style_is_not_deprecated() -> None:
    """`gitwildmatch` is deprecated in pathspec 1.x and disagrees with `gitignore` on `dir/*`.

    Compiling a pattern must not emit a DeprecationWarning, or every run of an application that
    turns warnings into errors fails on a path check. Pinned as a test rather than trusted, because
    the style name is a string and a rename in the library would otherwise surface as noise in
    someone else's logs.
    """
    _compiled.cache_clear()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert matches_path_patterns("internal/deep/a.txt", ("internal/**",)) is True
    finally:
        _compiled.cache_clear()

    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


@pytest.mark.parametrize(
    ("pattern", "path"),
    [
        ("[[]", "["),
        ("foo[[]bar", "foo[bar"),
        ("[#--]", "+"),
        ("[a&&b]", "&"),
        ("[a||b]", "a"),
        ("[a~~b]", "~"),
    ],
)
def test_literal_open_bracket_patterns_work_with_warnings_as_errors(
    pattern: str, path: str
) -> None:
    _compiled.cache_clear()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert matches_path_patterns(path, (pattern,)) is True
    finally:
        _compiled.cache_clear()


def test_the_installed_pathspec_is_in_the_supported_range() -> None:
    """The version range in `pyproject.toml` exists because 0.12 and 1.x answer `internal/*`
    differently against a grandchild. If a resolver ever hands us a 0.x, the table above would
    silently start describing different semantics -- so say so here instead."""
    import pathspec

    major = int(pathspec.__version__.split(".")[0])
    assert major == 1, f"unsupported pathspec {pathspec.__version__}; see pyproject.toml"
