"""The public preview caps are byte caps, and they hold for every script.

These bounds were measured in bytes and applied in characters, which made them no bounds at all for
non-ASCII text: any string with at most ``PREVIEW_BYTE_BUDGET`` characters and more than
``PREVIEW_BYTE_THRESHOLD`` bytes cleared the threshold and then survived the slice intact, while the
payload reported ``truncated: True``. A Korean or emoji-bearing value was published in full through
a field whose whole purpose is to publish an excerpt.

So the assertions here are deliberately about *bytes on the wire*, not about "is it shorter than the
input". A character-count test passes against the broken code.

The same defect existed twice with different constants -- ``public_view.preview_value`` and
``shell.preview_command`` -- on paths that never meet, so both are exercised here from the same
table. That is the shape this release keeps finding: a rule proven on one site and never bound on
its twin.
"""

from __future__ import annotations

import pytest

from monoid_agent_kernel.core.tool_approval import redact_tool_arguments
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.public_view import (
    PREVIEW_BYTE_BUDGET,
    PREVIEW_BYTE_THRESHOLD,
    PREVIEW_MAX_DEPTH,
    PREVIEW_MAX_ITEMS,
    PREVIEW_MAX_KEYS,
    REDACTED_PATH,
    args_preview,
    preview_value,
    public_path,
    redacted_value,
    shell_args_preview,
    truncate_to_bytes,
    web_args_preview,
)
from monoid_agent_kernel.shell import COMMAND_PREVIEW_BYTE_BUDGET, preview_command

# One character, three scripts, three encoded widths: 1, 2, 3 and 4 bytes per character. The
# multi-byte rows are the ones that used to escape; ASCII is here to pin that nothing regressed for
# the case that always worked.
SCRIPTS = [
    pytest.param("x", 1, id="ascii-1b"),
    pytest.param("д", 2, id="cyrillic-2b"),
    pytest.param("가", 3, id="hangul-3b"),
    pytest.param("\U0001f600", 4, id="emoji-4b"),
]


@pytest.mark.parametrize(("char", "width"), SCRIPTS)
def test_preview_value_never_publishes_more_than_the_byte_budget(char: str, width: int) -> None:
    """The cap is a byte cap for every script, not just for the one-byte one."""
    value = char * 300
    assert len(value.encode()) > PREVIEW_BYTE_THRESHOLD

    result = preview_value("summary", value, PermissionPolicy())

    assert result["truncated"] is True
    assert result["bytes"] == 300 * width
    assert len(result["preview"].encode()) <= PREVIEW_BYTE_BUDGET


@pytest.mark.parametrize(("char", "width"), SCRIPTS)
def test_preview_command_never_publishes_more_than_its_byte_budget(char: str, width: int) -> None:
    """``preview_command`` is the twin: same defect, different constants, unrelated path."""
    command = char * 300

    result = preview_command(command)

    assert result.endswith("...")
    assert len(result.encode()) <= COMMAND_PREVIEW_BYTE_BUDGET + len(b"...")


# Only the multi-byte scripts can enter the window that leaked, and that is the whole point: an
# ASCII string short enough to survive a 160-*character* slice is at most 160 bytes, so it never
# cleared the 240-byte threshold in the first place. The bug was invisible in ASCII and total in
# Korean. Parametrizing ASCII here would produce a row that always skips -- coverage in the report
# and nothing behind it.
LEAK_WINDOW_SCRIPTS = [param for param in SCRIPTS if param.values[1] > 1]


@pytest.mark.parametrize(("char", "width"), LEAK_WINDOW_SCRIPTS)
def test_the_whole_value_is_not_published_inside_the_old_leak_window(char: str, width: int) -> None:
    """Long enough in bytes to be truncated, short enough in characters to have survived the slice.

    This is the exact window the old code published in full while reporting ``truncated: True``.
    """
    value = char * (PREVIEW_BYTE_BUDGET - 1)
    assert len(value) <= PREVIEW_BYTE_BUDGET, "not inside the window: too many characters"
    assert len(value.encode()) > PREVIEW_BYTE_THRESHOLD, "not inside the window: too few bytes"

    result = preview_value("summary", value, PermissionPolicy())

    assert result["preview"] != value, "the whole value was published under a 'truncated' flag"
    assert len(result["preview"].encode()) <= PREVIEW_BYTE_BUDGET


@pytest.mark.parametrize(("char", "width"), LEAK_WINDOW_SCRIPTS)
def test_the_whole_command_is_not_published_inside_the_old_leak_window(char: str, width: int) -> None:
    command = char * (COMMAND_PREVIEW_BYTE_BUDGET - 1)
    assert len(command.encode()) > PREVIEW_BYTE_THRESHOLD

    result = preview_command(command)

    assert result != command + "...", "the whole command was published with a false ellipsis"
    assert len(result.encode()) <= COMMAND_PREVIEW_BYTE_BUDGET + len(b"...")


def test_preview_command_is_byte_identical_to_the_previous_behaviour_for_ascii() -> None:
    """The fix must not move the ASCII case: that one was already correct."""
    assert preview_command("x" * 300) == "x" * COMMAND_PREVIEW_BYTE_BUDGET + "..."
    assert preview_command("echo hi") == "echo hi"


@pytest.mark.parametrize(("char", "width"), SCRIPTS)
def test_the_threshold_boundary_is_exact(char: str, width: int) -> None:
    """At most ``PREVIEW_BYTE_THRESHOLD`` bytes passes through untouched; one more truncates."""
    policy = PermissionPolicy()
    fits = char * (PREVIEW_BYTE_THRESHOLD // width)
    assert len(fits.encode()) <= PREVIEW_BYTE_THRESHOLD
    assert preview_value("summary", fits, policy) == fits

    over = fits + char
    assert len(over.encode()) > PREVIEW_BYTE_THRESHOLD
    assert preview_value("summary", over, policy)["truncated"] is True


def test_truncate_to_bytes_backs_off_to_a_codepoint_boundary() -> None:
    """A bare byte slice raises rather than returning a short string, so the backoff is load-bearing."""
    value = "가" * 300
    with pytest.raises(UnicodeDecodeError):
        value.encode()[:PREVIEW_BYTE_BUDGET].decode()

    result = truncate_to_bytes(value, PREVIEW_BYTE_BUDGET)

    assert len(result.encode()) <= PREVIEW_BYTE_BUDGET
    assert value.startswith(result), "truncation must return a prefix, not a repaired string"
    # 160 bytes holds 53 whole 3-byte characters with one byte to spare; the partial 54th is dropped
    # rather than mangled into a replacement character.
    assert result == "가" * 53
    assert "�" not in result


@pytest.mark.parametrize("budget", [0, -1])
def test_truncate_to_bytes_treats_a_non_positive_budget_as_empty(budget: int) -> None:
    """Guards the slice: ``encoded[:-1]`` would otherwise publish all but one byte."""
    assert truncate_to_bytes("가" * 300, budget) == ""


def test_a_short_value_is_returned_unchanged_rather_than_copied_through_the_encoder() -> None:
    assert truncate_to_bytes("가나다", 240) == "가나다"


def test_the_trace_and_approval_previews_disagree_on_secrets_on_purpose() -> None:
    """An asymmetry that looks like this release's defect shape and is not one.

    `redact_tool_arguments` masks secret-*named* keys; `args_preview` does not. Reading that as a
    twin-miss and "fixing" it reverses `0109e06`, which removed unconfigurable key-name guessing from
    `public_view` on purpose and made redaction beyond content fields the integrating backend's job
    via the `EventSink` seam (`examples/redacting_event_sink.py`). The approval record is a different
    artifact — a human acts on it — so it masks.

    Pinned from both sides so that neither half can drift without a deliberate decision, and so a
    future reader does not have to re-derive which behaviour is intended.
    """
    arguments = {"api_key": "sk-live-DEADBEEF"}

    assert args_preview(arguments, PermissionPolicy())["api_key"] == "sk-live-DEADBEEF"
    assert redact_tool_arguments(arguments)["api_key"] == "[redacted]"


def test_deep_nesting_is_capped_instead_of_raising() -> None:
    """``metadata`` and plan items are ``additionalProperties: True``, so depth is model-controlled.

    Before the cap this raised ``RecursionError`` inside tool dispatch -- a model could crash the
    writer with one argument. The read side already caught this; the write side did not.
    """
    deep: dict[str, object] = {"leaf": "x"}
    for _ in range(600):
        deep = {"n": deep}

    result = preview_value("metadata", deep, PermissionPolicy())

    # ``PREVIEW_MAX_DEPTH`` levels of nesting survive; the marker replaces the level below them.
    for _ in range(PREVIEW_MAX_DEPTH):
        assert isinstance(result, dict), "the cap fired earlier than the documented depth"
        result = result["n"]
    assert result == {"truncated": True, "type": "dict", "depth_exceeded": PREVIEW_MAX_DEPTH}


def test_a_wide_mapping_is_capped_and_says_how_much_it_dropped() -> None:
    wide = {f"k{index}": "v" for index in range(PREVIEW_MAX_KEYS + 5)}

    result = preview_value("metadata", wide, PermissionPolicy())

    assert len(result) == PREVIEW_MAX_KEYS + 1  # the kept keys plus the marker
    assert result["truncated_keys"] == 5


def test_the_web_and_shell_previews_do_not_bypass_the_cap() -> None:
    """Two of the four dispatch branches copied their descriptors straight through.

    `web_args_preview` exists to withhold the query and the URL, and `shell_args_preview` to withhold
    env *values* — so a `locale`, a `blocked_domains` entry or an env *key* carrying 20 KB made each
    branch a way to publish exactly what it was withholding, in the same event.
    """
    smuggled = "Z" * 5000

    web = web_args_preview(
        {"query": "hidden", "locale": smuggled, "blocked_domains": ["EXFIL-" + smuggled]},
        PermissionPolicy(),
    )
    shell = shell_args_preview({"command": "echo hi", "env": {"K_" + smuggled: "v"}}, PermissionPolicy())

    assert smuggled not in str(web)
    assert smuggled not in str(shell)
    assert web["query_preview"] and "hidden" not in str(web["query_preview"])


def test_an_unnormalizable_path_is_redacted_rather_than_raising() -> None:
    """`is_path_redacted` normalizes before matching and *raises* on an absolute or `..` path — both
    of which a model can put in a `path` argument. These builders sit on the emit path, so a raise
    here ends the run of any operator who merely configured `redact_patterns`. Fail closed."""
    policy = PermissionPolicy(redact_patterns=("secrets/**",))

    for path in ("/etc/passwd", "../../etc/passwd"):
        assert preview_value("path", path, policy) == redacted_value(path)
        # ...and `public_path`, which is the *other* builder on the same emit path. Guarding only
        # `preview_value` left this one raising, and it has twelve callers across `loop`,
        # `loop_phases`, `tasks`, `tool_services.shell` and `core.projections` — so the rule lives
        # in `public_path` itself rather than at any of them.
        assert public_path(path, policy) == REDACTED_PATH
    # A normal relative path outside the patterns is still published.
    assert preview_value("path", "notes/a.md", policy) == "notes/a.md"
    assert public_path("notes/a.md", policy) == "notes/a.md"


def test_the_width_cap_does_not_reach_top_level_argument_keys() -> None:
    """``narration`` and the studio activity feed read ``args_preview`` by key.

    Capping the top level would silently drop the key they render. ``args_preview`` builds the outer
    mapping itself, so the cap applies only from the first nested level down -- assert that, because
    nothing else would notice if the recursion started one level too early.
    """
    arguments = {f"arg{index}": "v" for index in range(PREVIEW_MAX_KEYS + 5)}

    result = args_preview(arguments, PermissionPolicy())

    assert len(result) == PREVIEW_MAX_KEYS + 5
    assert "truncated_keys" not in result


def test_depth_and_width_caps_leave_ordinary_payloads_untouched() -> None:
    """The caps are a ceiling on hostile input, not a reshaping of normal tool arguments."""
    arguments = {
        "path": "notes/a.md",
        "options": {"mode": "append", "retries": 3},
        "outputs": ["a.md", "b.md"],
    }

    assert args_preview(arguments, PermissionPolicy()) == arguments


def test_list_marker_applies_to_the_value_passed_in_and_not_to_nested_containers() -> None:
    """Pins the entry point the integration test cannot reach, because nothing calls it yet.

    `plan.updated` hands `preview_value` the typed array directly, so only the list branch's
    suppression is observable end to end. The *dict* branch has the same rule and no caller today —
    exactly the shape that gets "tidied up" into propagating again, silently suppressing markers on
    every list nested under a mapping the day someone does pass one.

    The rule is scope, not depth: `list_marker=False` describes the value handed in. A list one
    level down inside a mapping is a different value with a different consumer, and it keeps its
    marker.
    """
    policy = PermissionPolicy()
    long_list = [f"item-{index}" for index in range(PREVIEW_MAX_ITEMS + 7)]

    # Entry point 1: the value itself is the list — suppression applies.
    assert preview_value("items", long_list, policy, list_marker=False) == long_list[:PREVIEW_MAX_ITEMS]

    # Entry point 2: the value is a mapping — the nested list is not what the caller described.
    nested = preview_value("payload", {"refs": long_list}, policy, list_marker=False)
    assert nested["refs"][-1] == {"truncated_items": 7}

    # And the default still marks in both positions.
    assert preview_value("items", long_list, policy)[-1] == {"truncated_items": 7}
