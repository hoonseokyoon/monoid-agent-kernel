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

import ast
import contextlib
import json
import time
import tracemalloc
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from support.hostile_scalars import (
    HOSTILE_NAMED_TYPES,
    EmptyClaimingPath,
    HostileNamedDict,
    HostileNamedList,
    ExplodingComparisons,
    ExplodingText,
    ImpersonatingName,
    MisreportingKey,
    MisreportingText,
    ShoutingText,
    UnderstatedInteger,
    UnderstatedText,
    hugely_named_object,
)

import monoid_agent_kernel
from monoid_agent_kernel.core.json_ingress import (
    UnportableScalarError,
    normalize_json_ingress,
    portable_type_name,
)
from monoid_agent_kernel.core.tool_approval import _jsonish, redact_tool_arguments
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.public_view import (
    APPROVAL_BYTE_BUDGET,
    APPROVAL_BYTE_THRESHOLD,
    APPROVAL_PAYLOAD_BYTE_BUDGET,
    PREVIEW_BYTE_BUDGET,
    PREVIEW_BYTE_THRESHOLD,
    PREVIEW_MAX_DEPTH,
    PREVIEW_MAX_ITEMS,
    PREVIEW_MAX_KEYS,
    REDACTED_PATH,
    TRACE_PAYLOAD_BYTE_BUDGET,
    PayloadBudget,
    _budgeted_field,
    _fragment_cost,
    TRUNCATION_SUFFIX,
    args_preview,
    finish_args_preview,
    preview_value,
    public_event_payload,
    public_identifier,
    public_inline_path,
    public_proposal_payload,
    public_result_content,
    public_path,
    public_proposal_file,
    public_error_message,
    redacted_value,
    shell_args_preview,
    touches_redacted_path,
    truncate_inline_text,
    truncate_to_bytes,
    web_args_preview,
)
from monoid_agent_kernel.web import (
    public_query_preview,
    public_url_preview,
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
        # `preview_value` left this one raising, and it has fourteen call sites across `loop`,
        # `loop_phases`, `tasks`, `tool_services.shell` and `core.projections` — so the rule lives
        # in `public_path` itself rather than at any of them.
        assert public_path(path, policy) == REDACTED_PATH
        # ...and the third caller in this module, which after `public_path` was guarded was calling
        # both: `public_proposal_file` failed closed on its `path` and still raised on the identical
        # string one line earlier. Found by sweeping for the shape rather than by review.
        entry = public_proposal_file({"path": path, "snapshot_path": "snap/x"}, policy)
        assert entry["path"] == REDACTED_PATH
        assert entry["snapshot_path"] == REDACTED_PATH
    # A normal relative path outside the patterns is still published.
    assert preview_value("path", "notes/a.md", policy) == "notes/a.md"
    assert public_path("notes/a.md", policy) == "notes/a.md"
    assert public_proposal_file({"path": "notes/a.md", "snapshot_path": "snap/x"}, policy) == {
        "path": "notes/a.md",
        "kind": None,
        "size": None,
        "sha256": None,
        "base_sha256": None,
        "proposed_sha256": None,
        "snapshot_sha256": None,
        "change_kind": None,
        "snapshot_path": "snap/x",
    }


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


def test_a_body_smuggled_into_the_key_position_is_bounded_like_any_other_string() -> None:
    """The one string this traversal published at any length.

    `preview_value` capped every *value* and emitted `str(child_key)` untouched, so the identical
    payload came out `{"redacted": true}` in the value position and verbatim in the key position.
    `_is_content_field` cannot help: it reads the key to judge the value, so a body moved *into*
    the key has no name left to incriminate it. Measured in bytes and with multibyte text, because
    a character-slice cap is no cap at all for Hangul -- the defect this whole file exists for.
    """
    policy = PermissionPolicy()
    body = "비밀-" + "가" * 10_000

    as_value = preview_value("metadata", {"content": body}, policy)
    assert as_value["content"]["redacted"] is True

    published_key = next(iter(preview_value("metadata", {body: "x"}, policy)))
    assert published_key != body, "the key rode out verbatim"
    assert len(published_key.encode("utf-8")) <= PREVIEW_BYTE_BUDGET + len(TRUNCATION_SUFFIX.encode())
    assert published_key.endswith(TRUNCATION_SUFFIX), "a cut key must say it was cut"


def test_bounding_a_key_never_silently_merges_two_distinct_entries() -> None:
    """Truncation makes distinct keys collide, and a dict drops the loser without a word.

    That would trade one silent cap for another -- the failure this release exists to close -- so
    the collision is disambiguated instead. Both source entries have to survive with their own
    values, or the bound is buying egress at the price of correctness.
    """
    policy = PermissionPolicy()
    shared = "x" * (PREVIEW_BYTE_THRESHOLD + 50)

    preview = preview_value("metadata", {shared + "-alpha": 1, shared + "-beta": 2}, policy)

    assert len(preview) == 2, f"an entry was dropped: {preview}"
    assert sorted(preview.values()) == [1, 2]


def test_every_path_argument_the_registry_declares_is_redacted_not_just_the_three_hardcoded(
    tmp_path,
) -> None:
    """The redaction was keyed on `{path, root, cwd}`; the registry declares the real list.

    `fs.move`/`fs.copy` declare `("source_path", "destination_path")`, so one `fs.move` published
    `paths: ["[redacted-path]"]` and `args_preview.source_path: "secrets/creds.txt"` on the *same
    event*: the operator's redaction defeated by the field beside it. Driven off `builtin_tools`
    rather than a literal list so that declaring a new path argument cannot quietly reopen this.
    """
    from monoid_agent_kernel.tools.builtin import builtin_tools
    from monoid_agent_kernel.workspace.local import LocalWorkspaceBackend

    policy = PermissionPolicy(redact_patterns=("secrets/**", "secrets/*"))
    secret = "secrets/creds.txt"

    declared = {name for spec in builtin_tools(LocalWorkspaceBackend(tmp_path)) for name in spec.path_args}
    assert {"source_path", "destination_path"} <= declared, "registry no longer exercises this"

    for name in sorted(declared):
        assert preview_value(name, secret, policy) == {
            "redacted": True,
            "type": "str",
            "bytes": len(secret.encode()),
        }, f"{name} is a declared path argument and published verbatim"


def test_the_approval_card_never_shows_a_body_while_hiding_where_it_goes() -> None:
    """The decision surface was inverted for exactly one release candidate.

    `redact_tool_arguments` turns content redaction *off* so an approver can read what a write
    contains. Path redaction stayed on, so under `redact_patterns` the card rendered a private
    key's contents above `{"redacted": true}` where the destination should be -- it hid the field
    the decision turns on and showed the field that made hiding it pointless.

    Both fields now move together. The operator's explicit `redact_patterns` outranks the card's
    exemption (unlike the kernel's own `_is_content_field` default, which an approver is entitled
    to see past), so a redacted path withholds the whole call rather than half of it.
    """
    body = "-----BEGIN PRIVATE KEY-----"
    arguments = {"path": "secrets/deploy.key", "content": body}

    exposed = redact_tool_arguments(arguments, policy=PermissionPolicy())
    assert exposed == {"path": "secrets/deploy.key", "content": body}, "the approver must see both"

    withheld = redact_tool_arguments(arguments, policy=PermissionPolicy(redact_patterns=("secrets/*",)))
    assert withheld["path"]["redacted"] is True
    assert withheld["content"]["redacted"] is True, "a body shown beside a hidden path protects nothing"
    assert body not in str(withheld)


def test_the_same_path_is_bounded_identically_in_paths_and_in_the_result_beside_it() -> None:
    """One `workspace.file.changed` carried the same argument cut in one field and whole in another.

    `_public_paths_from_args` capped and marked `paths`; `public_result_content` diverted `path` to
    bare `public_path`, which redacts but never truncates. A cap the neighbouring field publishes
    around is not a cap -- the same way the `source_path` redaction was defeated.

    Not an end-to-end test because Windows cannot create the 300-byte path it would need; the
    coupling is the thing worth pinning, and it lives here.
    """
    policy = PermissionPolicy()
    long_path = "d" * 300 + "/note.md"

    on_the_call = public_inline_path(long_path, policy)
    in_the_result = public_result_content({"path": long_path}, policy)["path"]

    assert in_the_result == on_the_call
    assert on_the_call.endswith(TRUNCATION_SUFFIX)
    assert len(on_the_call.encode()) <= PREVIEW_BYTE_BUDGET + len(TRUNCATION_SUFFIX.encode())


def test_a_contract_path_is_never_truncated_even_though_the_log_one_is() -> None:
    """The other half, and the reason `public_inline_path` is a separate function.

    `proposal.json`'s `changed_paths` and `snapshot_path` are resolved back to real files by
    `core.proposal_file`, `core.packages` and `core.schemas`. Truncating them there would not
    publish a shorter path, it would break replay and packaging -- so "bind the rule at every
    caller of `public_path`" is the wrong generalisation here, and this pins that it stays wrong.
    """
    policy = PermissionPolicy()
    long_path = "d" * 300 + "/note.md"

    payload = public_proposal_payload(
        {"changed_paths": [long_path], "files": [{"path": long_path, "snapshot_path": long_path}]},
        policy,
    )

    assert payload["changed_paths"] == [long_path]
    assert payload["files"][0]["path"] == long_path
    assert payload["files"][0]["snapshot_path"] == long_path


def test_a_self_referencing_container_is_elided_instead_of_re_expanded() -> None:
    """The depth cap terminates the walk but does not bound its cost.

    A container reachable from itself is re-expanded once per edge per level, so a 21-object input
    with 20 self-referencing keys costs `20 ** PREVIEW_MAX_DEPTH` nodes. Measured before this guard:
    fanout 8 took 45 s and produced 1.1 GB, from an input that fits on one line. JSON cannot express
    sharing, so a tool *argument* cannot reach it -- but `public_result_content` previews a
    `ToolResult.content` built by a custom or MCP handler out of ordinary Python objects, which can.
    """
    policy = PermissionPolicy()
    cyclic: dict[str, Any] = {}
    for index in range(8):
        cyclic[f"k{index}"] = cyclic

    start = time.monotonic()
    published = json.dumps(preview_value("metadata", cyclic, policy))
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, f"the traversal is still re-expanding: {elapsed:.2f}s"
    assert len(published) < 10_000, f"output blew up to {len(published)} bytes"
    assert '"circular": true' in published


def test_a_value_shared_twice_without_a_cycle_still_renders_both_times() -> None:
    """The guard tracks ancestors on the current path, not everything seen.

    A global seen-set would be cheaper and wrong: the same small mapping appearing under two keys is
    ordinary, and eliding the second would report a cycle where there is none — a preview that lies
    about the shape of the data to save work it did not need to save.
    """
    shared = {"x": 1}

    assert preview_value("metadata", {"p": shared, "q": shared}, PermissionPolicy()) == {
        "p": {"x": 1},
        "q": {"x": 1},
    }


# Every builder that assembles an outer mapping from model-authored input. Parametrized rather than
# written out, because the previous fix bound keys inside `preview_value`'s traversal and left all
# five of these -- and depth 0 is the *only* depth a model names directly, since the outer mapping
# of a tool call is built here rather than by the traversal.
TOP_LEVEL_BUILDERS = [
    pytest.param(
        lambda args, policy: args_preview(args, policy),
        PREVIEW_BYTE_BUDGET, PREVIEW_BYTE_THRESHOLD, id="args_preview",
    ),
    pytest.param(
        lambda args, policy: finish_args_preview(args, policy),
        PREVIEW_BYTE_BUDGET, PREVIEW_BYTE_THRESHOLD, id="finish_args_preview",
    ),
    pytest.param(
        lambda args, policy: public_result_content(args, policy),
        PREVIEW_BYTE_BUDGET, PREVIEW_BYTE_THRESHOLD, id="public_result_content",
    ),
    pytest.param(
        lambda args, policy: public_event_payload(args, policy),
        PREVIEW_BYTE_BUDGET, PREVIEW_BYTE_THRESHOLD, id="public_event_payload",
    ),
    pytest.param(
        lambda args, policy: redact_tool_arguments(args, policy=policy),
        APPROVAL_BYTE_BUDGET, APPROVAL_BYTE_THRESHOLD, id="redact_tool_arguments",
    ),
]


@pytest.mark.parametrize(("build", "budget", "threshold"), TOP_LEVEL_BUILDERS)
def test_a_top_level_key_is_bounded_by_every_builder(build, budget: int, threshold: int) -> None:
    """A 90 KB body was `{"redacted": true}` as a value, 162 bytes as a nested key, and verbatim
    as a top-level key -- through all five of these, onto `events.jsonl`, `task.json` and
    `status.json`."""
    body = "비밀" * 15_000
    assert len(body.encode()) > budget

    published = next(iter(build({body: 1}, PermissionPolicy())))

    assert published != body, "the key rode out verbatim"
    assert len(published.encode()) <= budget + len(TRUNCATION_SUFFIX.encode())
    assert published.endswith(TRUNCATION_SUFFIX)


@pytest.mark.parametrize(("build", "budget", "threshold"), TOP_LEVEL_BUILDERS)
def test_no_builder_silently_merges_two_top_level_keys(build, budget: int, threshold: int) -> None:
    """The collision hazard follows the cap wherever it goes.

    Sized off the **threshold**, not the budget: at `budget + 50` four of these five builders got a
    216-byte key, under the 240-byte threshold, so nothing truncated, nothing collided, and the
    assertion passed without ever reaching the code it names. The nested-key twin in this same file
    already used the threshold — the two halves drifted apart.
    """
    shared = "x" * (max(threshold, budget) + 50)

    published = build({shared + "-alpha": 1, shared + "-beta": 2}, PermissionPolicy())

    assert len(published) == 2, f"an argument was dropped: {list(published)}"


def test_an_ordinary_argument_name_is_untouched_by_the_key_bound() -> None:
    """The cap is a ceiling on hostile input, not a reshaping of every tool call.

    Without this, "bound the keys" would satisfy every assertion above while renaming every
    argument an operator reads -- and `narration` and the Studio activity feed look arguments up by
    name, so a reshaped key blanks them.
    """
    arguments = {"path": "notes/a.md", "recursive": True, "max_bytes": 10}

    assert args_preview(arguments, PermissionPolicy()) == arguments


def test_a_model_chosen_tool_name_is_bounded() -> None:
    """`_aexecute_tool_call` handles a `call_name` the catalog cannot resolve and still emits it,
    so the name is model-authored text, not a kernel enum.

    Named for the function, not for the events: this exercises `public_identifier` directly and
    builds no event. It was called `..._on_the_events_that_carry_it` while asserting nothing about
    an event, and on one of those events the whole name was still going out through `error` at the
    time — a name promising end-to-end coverage over a unit test is how that survived.
    `tests/test_event_stream_bounds.py` owns the end-to-end half.
    """
    assert public_identifier("fs.write") == "fs.write"

    published = public_identifier("도구" * 6_000)

    assert len(published.encode()) <= PREVIEW_BYTE_BUDGET + len(TRUNCATION_SUFFIX.encode())
    assert published.endswith(TRUNCATION_SUFFIX)


@pytest.mark.parametrize(
    ("arguments", "label"),
    [
        ({"metadata": {"path": "secrets/id_rsa"}}, "one level down"),
        ({"metadata": {"deeper": {"source_path": "secrets/id_rsa"}}}, "two levels down"),
        ({"source_path": ["secrets/id_rsa"]}, "inside a list"),
    ],
)
def test_the_approval_escape_hatch_reaches_as_deep_as_the_traversal(arguments, label: str) -> None:
    """A regression this branch introduced and had to take back.

    `decision_surface` switched path redaction off for the approver, and `touches_redacted_path` --
    the escape hatch that switches it back on when the operator marked something secret -- looked
    only at `values.items()`. `preview_value` recurses, so a redacted path one level down got
    neither treatment and was published where the previous release rendered `{"redacted": true}`.
    Reachable from a builtin: `artifact.emit`'s `metadata` is `additionalProperties: true`, so it
    survives argument validation intact.

    The two halves of a pair have to walk the same shape or the shallower one is a hole.
    """
    policy = PermissionPolicy(redact_patterns=("secrets/**",))

    published = json.dumps(redact_tool_arguments(arguments, policy=policy))

    assert "secrets/id_rsa" not in published, f"published verbatim ({label}): {published}"


def test_a_path_naming_a_redacted_file_through_a_dotdot_is_still_redacted() -> None:
    """Fail closed for every path-naming field, and the reason the narrower rule was taken back.

    Scoping fail-closed to `path`/`root`/`cwd` — so a task result's absolute `report_path` would not
    be blanked by an unrelated pattern — re-opened the leak the widening had closed.
    `normalize_workspace_path` raises on any `..` component *before* resolving it, so
    `x/../secrets/creds.txt` raises while naming a file the operator's pattern matches, and was then
    published verbatim beside `paths: ["[redacted-path]"]` on the same event, with the approval
    card's content redaction going with it.

    Both failure modes are real. Only one is silent: an over-redacted field renders
    `{"redacted": true}` and an operator can see it and widen the glob, while an under-redacted one
    is indistinguishable from a field that was checked and cleared.
    """
    policy = PermissionPolicy(redact_patterns=("secrets/**",))

    for spelling in ("secrets/creds.txt", "x/../secrets/creds.txt", "./secrets/creds.txt"):
        assert args_preview({"source_path": spelling}, policy) == {
            "source_path": redacted_value(spelling)
        }, f"published verbatim: {spelling}"

    # And the approval card withholds the body alongside it, rather than showing one and not the other.
    card = redact_tool_arguments(
        {"source_path": "x/../secrets/creds.txt", "content": "-----BEGIN PRIVATE KEY-----"},
        policy=policy,
    )
    assert card["source_path"]["redacted"] is True
    assert card["content"]["redacted"] is True


def _shell_request(**overrides: Any) -> Any:
    from monoid_agent_kernel.shell import ShellApprovalRequest

    base = dict(
        run_id="run_1",
        tool_call_id="c1",
        command="echo hi",
        cwd=".",
        requested_timeout_s=30,
        effective_timeout_s=30,
        requested_max_output_bytes=1000,
        effective_max_output_bytes=1000,
        execution_workspace="workspace",
    )
    base.update(overrides)
    return ShellApprovalRequest(**base)


def test_the_shell_approval_payload_is_bounded_for_every_event_that_spreads_it() -> None:
    """The largest claim in the release notes, and until now the least tested.

    `ShellApprovalRequest.to_public_json` is `data=` for seven emit sites covering six event types —
    `tool.approval.requested` / `.approved` / `.denied` and `shell.exec.started` / `.finished` /
    `.failed`. The only tests in the area called `shell_args_preview` directly, which is the
    `tool.call.started` route: the half that already worked before this release. So the fix for the
    half that did *not* work was asserted by nothing.

    Testing the builder rather than the seven emit sites is deliberate — it is the single function
    they all go through, which is the whole reason the fix was made there.
    """
    policy = PermissionPolicy(redact_patterns=("secrets/**", "secrets"))
    big_key = "SECRET_" + "가" * 10_000

    published = _shell_request(cwd="secrets", env_keys=(big_key, "PATH")).to_public_json(policy)

    assert published["cwd"] == redacted_value("secrets"), "cwd bypassed redact_patterns"
    keys = published["env_keys"]
    assert big_key not in keys, "a 30 KB env key rode out verbatim"
    assert len(json.dumps(keys, ensure_ascii=False).encode()) < 1_000
    # A rejected call still spreads this payload, which is why the bound has to live here.
    assert "가" * 100 not in json.dumps(published, ensure_ascii=False)


def test_an_ordinary_shell_approval_payload_is_unchanged() -> None:
    """The bound is a ceiling on hostile input, not a reshaping of every approval card."""
    published = _shell_request(env_keys=("PATH", "HOME")).to_public_json(PermissionPolicy())

    assert published["cwd"] == "."
    assert published["env_keys"] == ["PATH", "HOME"]
    assert published["command_preview"] == "echo hi"
    assert published["effective_timeout_s"] == 30


def test_the_web_event_payload_is_bounded_on_the_events_either_side_of_the_call() -> None:
    """`WebService` builds its own `event_data` for `.started` / `.finished` / `.failed`.

    Same shape as the shell half: `web_args_preview` covers `tool.call.started`, and the three
    inline payloads either side of it were the ones publishing `locale` and the domain lists raw.
    """
    policy = PermissionPolicy()
    big_locale = "ko-" + "가" * 10_000

    published = public_event_payload(
        {"query_preview": {"redacted": True}, "locale": big_locale, "blocked_domains": ["evil.test"]},
        policy,
    )

    assert big_locale not in json.dumps(published, ensure_ascii=False)
    assert published["locale"]["truncated"] is True
    assert published["blocked_domains"] == ["evil.test"], "ordinary descriptors pass through"


# --- The payload budget: the per-value caps bound each piece, this bounds their sum ------------
#
# Depth, key, item and byte caps bound a preview's *shape*; none of them bounds the payload. Two
# measured consequences at the commit before this one: a mapping shared five ways across nine
# levels -- 46 objects, an input that fits on a line -- previewed to 25.78 MB in 1.02 s (sharing is
# not a cycle, so the ancestor guard never fires), and a payload chunked into cap-obeying pieces
# published all of them (20 items x 20 keys x 234 B ~= 95 KB per event, with nothing bounding the
# piece count). The budget is charged on what is *appended* -- keys, values and truncation markers
# alike -- so the serialized payload can never exceed it.


def _payload_bytes(published: Any) -> int:
    return len(json.dumps(published, ensure_ascii=False).encode("utf-8"))


def _widest_payload_bytes(published: Any) -> int:
    """The payload as the widest *stream* writer spells it: non-ASCII escaped, default separators.

    ``reference.studio``'s ``_sse_send`` serializes exactly this way and
    ``EventSubscriptionFrame.to_sse`` differs only by compact separators, so no writer that puts
    a payload on a stream or a log line can spell it larger. ``write_json_atomic``'s
    pretty-printed ``status.json`` and approval files are outside this measure by design — see
    ``_fragment_cost``, which states that reach.
    """
    return len(json.dumps(published).encode("utf-8"))


def test_a_value_shared_along_many_paths_costs_at_most_one_payload_budget() -> None:
    """Gap 7: re-expansion is legal per path, so only a total can bound it.

    The cycle guard tracks ancestors on the current path -- deliberately, because the same small
    mapping appearing twice is ordinary and both copies must render. That leaves fanout: nine
    levels shared five ways costs 5**8 depth markers at the floor. No per-value cap sees that
    coming, because every individual value here is tiny. Only reachable from a Python-object
    caller (a custom or MCP tool's ``ToolResult.content``); JSON cannot express sharing.
    """
    policy = PermissionPolicy()
    leaf: dict[str, Any] = {"deep": "x"}
    for _ in range(9):
        leaf = {f"k{index}": leaf for index in range(5)}

    start = time.monotonic()
    published = public_result_content({"payload": leaf}, policy)
    elapsed = time.monotonic() - start

    assert _payload_bytes(published) <= TRACE_PAYLOAD_BYTE_BUDGET, "the DAG re-expanded unbounded"
    assert elapsed < 1.0, f"the traversal is still walking the whole DAG: {elapsed:.2f}s"


def test_chunking_under_every_per_value_cap_cannot_exceed_the_payload_budget() -> None:
    """Gap 8: every cap is per-value or per-container, and the piece count was unbounded.

    Each piece here obeys every rule -- 234 bytes is under the threshold, so each value crosses
    whole -- and the top-level key count is deliberately uncapped (narration reads specific
    argument names). The budget is what makes "the stream is a bounded channel" true per payload
    rather than per piece; the cut announces itself through the key the width cap already uses.
    """
    policy = PermissionPolicy()
    word = "A" * 234
    arguments = {f"arg{index:04d}": word for index in range(2000)}  # ~= 490 KB of obedient pieces

    published = args_preview(arguments, policy)

    assert _payload_bytes(published) <= TRACE_PAYLOAD_BYTE_BUDGET
    assert published["truncated_keys"] > 0, "a cap that does not say it capped"


def test_the_budget_is_spent_per_payload_and_not_per_top_level_key() -> None:
    """The reverted ``PREVIEW_MAX_NODES`` was born once per top-level ``preview_value`` call, so
    400 keys each got a fresh allowance and the payload still reached 42 MB. One budget object per
    payload, threaded through the whole traversal, is the difference being pinned: this shape has
    every subtree comfortably inside a per-key allowance and the *sum* far outside one.
    """
    policy = PermissionPolicy()
    word = "B" * 234
    arguments = {
        f"arg{index:03d}": {f"k{n}": word for n in range(20)} for index in range(400)
    }  # each value ~= 5 KB; the sum ~= 2 MB

    published = args_preview(arguments, policy)

    assert _payload_bytes(published) <= TRACE_PAYLOAD_BYTE_BUDGET


def test_the_gap_8_exemplar_payload_is_published_byte_identical() -> None:
    """The budget sits far above anything the caps admit in ordinary shapes.

    This is the *maximum* cap-obeying flat payload from the gap-8 triage (20 items x 20 keys x
    234 B ~= 95 KB), and it crosses byte-identical: the budget exists for pathological nesting and
    sharing, not to reshape any payload the existing caps already describe. Green by design before
    and after the budget landed -- the guard that the constant stays above the ordinary ceiling.
    """
    policy = PermissionPolicy()
    word = "C" * 234
    items = [
        {"step": word, "status": "pending", **{f"extra{n}": word for n in range(18)}}
        for _ in range(20)
    ]

    published = preview_value("items", items, policy, list_marker=False)

    assert published == items


def test_the_fixed_key_builders_share_one_payload_budget_across_their_fields() -> None:
    """``shell_args_preview`` and ``web_args_preview`` skip ``public_mapping`` on purpose — their
    outer keys are kernel literals — which is exactly how they would skip the budget: each field
    previewing under its own fresh allowance is the reverted per-key accounting wearing builder
    clothes. The field *values* are model-controlled, so any of them can be a nested container,
    and a container's *structure* is what no per-value cap bounds — the depth and width caps
    admit ``5**6`` expansion paths from an input that fits on a line.
    """
    policy = PermissionPolicy()
    blob: dict[str, Any] = {"deep": "x"}
    for _ in range(6):
        blob = {f"k{index}": blob for index in range(5)}  # ~= 400 KB once re-expanded per path

    shell = shell_args_preview({"command": "echo", "timeout_s": blob, "env": {}}, policy)
    web = web_args_preview({"locale": blob, "max_results": blob}, policy)

    assert _payload_bytes(shell) <= TRACE_PAYLOAD_BYTE_BUDGET
    assert _payload_bytes(web) <= TRACE_PAYLOAD_BYTE_BUDGET


def test_an_integer_past_the_threshold_is_enveloped_like_a_string_of_that_size() -> None:
    """Gap 3's leak half: ``preview_value`` bounded ``str`` and returned every other scalar whole.

    A 4300-digit integer — model-authored content in base ten, and the largest the decoders admit —
    measured 4,300 bytes against a 240-byte threshold through ``args_preview``. The envelope is the
    string envelope's sibling, and its ``preview`` is spelled in hex because a decimal spelling is
    exactly what the interpreter's digit limit may refuse to build; small scalars keep their type,
    which is what the ``artifact.emitted.kind`` precedent demands of schema-typed neighbours.
    """
    policy = PermissionPolicy()
    big = int("9" * 4300)

    published = args_preview({"n": big, "small": 7}, policy)

    envelope = published["n"]
    assert envelope["truncated"] is True
    assert envelope["type"] == "int"
    assert envelope["preview"].startswith("0x")
    assert len(envelope["preview"].encode("utf-8")) <= PREVIEW_BYTE_BUDGET
    assert published["small"] == 7, "a small scalar must keep its type"
    assert "9" * 60 not in json.dumps(published), "the decimal value rode out whole"


@pytest.mark.parametrize(
    ("value", "enveloped"),
    [
        pytest.param(10**239, False, id="240-digit-positive-at-threshold"),
        pytest.param(10**240, True, id="241-digit-positive-over"),
        pytest.param(-(10**238), False, id="negative-240-bytes-at-threshold"),
        pytest.param(-(10**239), True, id="negative-241-bytes-over"),
    ],
)
def test_the_integer_threshold_boundary_is_exact_in_encoded_bytes(value: int, enveloped: bool) -> None:
    """Encoded bytes, sign included — mirroring the string threshold's exactness pin.

    The negative rows are the ones a digit-count rule gets wrong: ``-(10**239)`` has 240 digits
    and 241 encoded bytes, so it crosses the threshold its positive twin sits exactly on.
    """
    published = preview_value("n", value, PermissionPolicy())

    assert isinstance(published, dict) is enveloped


def test_a_scalar_outside_json_is_named_by_type_and_never_asked_to_speak() -> None:
    """Gap 3's crash half, as seen by the traversal: ``bytes`` used to pass through whole.

    The envelope names the type and nothing else — no ``repr``, no ``str``, no ``len`` — because
    the value is the one thing this branch must not consult: a hostile ``__repr__`` runs arbitrary
    code inside event construction, and a >4300-digit integer's decimal spelling is exactly the
    call that raises. Ingress refusal at the tool-result boundary is the primary defence; this is
    what any traversal-shaped route that skips it still gets.
    """
    policy = PermissionPolicy()

    class _AngryRepr:
        def __repr__(self) -> str:
            raise RuntimeError("repr must not be consulted")

    published = args_preview({"blob": b"\x00\x01", "angry": _AngryRepr()}, policy)

    assert published["blob"] == {"truncated": True, "type": "bytes"}
    assert published["angry"] == {"truncated": True, "type": "_AngryRepr"}


def test_the_trace_budget_does_not_leak_into_the_approval_surface() -> None:
    """A person authorizing a call reads the approval card; a log reader reads the trace.

    The two surfaces share one traversal, so a budget added there is inherited by both -- which is
    how the trace constant could silently become the approval card's ceiling. Same arguments, both
    surfaces: the trace cuts and says so; the approval card, whose own budget is far higher, shows
    every argument.
    """
    policy = PermissionPolicy()
    word = "D" * 234
    arguments = {f"arg{index:04d}": word for index in range(2000)}  # ~= 490 KB

    trace = args_preview(arguments, policy)
    approval = redact_tool_arguments(arguments, policy=policy)

    assert "truncated_keys" in trace
    assert "truncated_keys" not in approval, "the approval card lost arguments to the trace constant"
    assert len(approval) == 2000


def test_the_approval_surface_is_payload_bounded_too_just_far_higher() -> None:
    """Bounded, but readable -- the approval principle extends to the payload total.

    A pathological argument map should not put megabytes on ``task.started`` and ``task.json``
    just because the surface is a decision surface; it should stay readable up to a ceiling a
    human could conceivably scroll. Past that, the card cuts and says so, exactly like the trace.
    """
    policy = PermissionPolicy()
    word = "E" * 234
    arguments = {f"arg{index:04d}": word for index in range(6000)}  # ~= 1.5 MB

    approval = redact_tool_arguments(arguments, policy=policy)

    assert _payload_bytes(approval) <= APPROVAL_PAYLOAD_BYTE_BUDGET
    assert "truncated_keys" in approval


@pytest.mark.parametrize(
    ("label", "char", "utf8_width"),
    [
        ("hangul-3-byte", "가", 3),
        ("cyrillic-2-byte", "б", 2),
        ("emoji-surrogate-pair", "😀", 4),
    ],
)
def test_the_payload_budget_holds_in_the_widest_spelling_a_sink_uses(
    label: str, char: str, utf8_width: int
) -> None:
    """The same defect this file was written about, one level up: bytes measured in the wrong
    representation are not a bound.

    The per-value caps were once counted in characters and applied to UTF-8 bytes, which made
    them no cap at all for non-ASCII text. The payload budget arrived counting UTF-8 bytes while
    two live sinks -- ``EventSubscriptionFrame.to_sse`` and Studio's ``_sse_send`` -- serialize
    with ``ensure_ascii=True`` on purpose, because U+2028/U+2029/U+0085 survive an unescaped dump
    and split an SSE frame mid-string. Escaped, one BMP character costs six bytes however few it
    takes in UTF-8: 2x for Hangul, 2.87x for two-byte scripts, and the same for a non-BMP
    codepoint spelled as a surrogate pair. A payload charged just inside 256 KiB reached 503,579
    bytes out of the real frame writer.

    So the charge is the widest spelling any sink uses, and the arms here are the scripts whose
    ratios differ -- a Hangul-only pin would leave the worse two-byte case unmeasured, which is
    the "clean twin" shape this repository keeps re-earning.
    """
    policy = PermissionPolicy()
    # Each value sits at the per-value byte threshold, so nothing is cut per piece and the
    # payload total is the only thing that can bound this.
    value = char * (PREVIEW_BYTE_THRESHOLD // utf8_width)
    assert len(value.encode()) <= PREVIEW_BYTE_THRESHOLD
    arguments = {f"arg{index:05d}": value for index in range(4000)}

    published = args_preview(arguments, policy)

    assert _widest_payload_bytes(published) <= TRACE_PAYLOAD_BYTE_BUDGET, (
        f"{label}: the payload fits only when the escaping sinks are not counted"
    )
    assert "truncated_keys" in published, "the fixture must actually reach the budget"
    assert len(published) > 1, "the budget cut so hard the payload says nothing"


def test_an_argument_named_like_the_marker_is_renamed_not_erased() -> None:
    """The marker must not destroy the argument whose name it happens to share.

    A model names its own arguments, so one may be called ``truncated_keys``. The marker used to
    be written straight over it: the value vanished with no marker of its own, and the count
    under-reported by one, because the entry it replaced had already been counted as published.
    That is a cap that does not say what it capped, arriving through the very key meant to
    announce the cut -- and the top level is the worst place for it, because narration and the
    activity feed read these keys by name.

    The marker keeps the plain name and the argument takes the ``#N`` suffix, inverting
    ``_bounded_key``'s "first one wins" on purpose: the name is a contract a consumer looks up,
    and one that cannot find it reads a cut payload as a complete one.
    """
    policy = PermissionPolicy()
    # First, so it is published before the budget stops: a key the cut never reached is dropped
    # rather than overwritten, which is a different (and correctly reported) outcome.
    arguments = {"truncated_keys": "MODEL AUTHORED VALUE"}
    arguments.update({f"arg{index:05d}": "D" * 234 for index in range(2000)})

    published = args_preview(arguments, policy)

    assert published["truncated_keys"] == len(arguments) - (len(published) - 1)
    assert published["truncated_keys#2"] == "MODEL AUTHORED VALUE"
    assert _widest_payload_bytes(published) <= TRACE_PAYLOAD_BYTE_BUDGET


def test_a_nested_key_named_like_the_marker_is_renamed_too() -> None:
    """The same rule one level down, where the width cap has been overwriting since before this
    branch.

    The note this replaced called the loss acceptable because only nested dicts could cap and no
    consumer reads those by key. The payload budget made the first half false the moment the top
    level could cut, and a rule that holds at one depth and not its twin is this repository's
    house defect -- so the marker is disambiguated wherever it is written.
    """
    policy = PermissionPolicy()
    # First, for the same reason as above: past the width cap it would be dropped, not overwritten.
    inner: dict[str, Any] = {"truncated_keys": "NESTED MODEL VALUE"}
    inner.update({f"k{index}": index for index in range(PREVIEW_MAX_KEYS + 5)})

    published = preview_value("payload", {"inner": inner}, policy)["inner"]

    assert published["truncated_keys#2"] == "NESTED MODEL VALUE"
    assert published["truncated_keys"] == len(inner) - (len(published) - 1)


def test_a_mapping_that_was_not_cut_keeps_its_marker_named_key_untouched() -> None:
    """The guard: disambiguation happens only where a cut is actually announced.

    Reserving the name unconditionally would rename a key in a payload nothing was dropped from,
    which is the "ordinary payloads pass through unchanged" guarantee this module keeps elsewhere.
    """
    policy = PermissionPolicy()
    arguments = {"truncated_keys": "MODEL AUTHORED VALUE", "other": 1}

    assert args_preview(arguments, policy) == arguments

@contextlib.contextmanager
def _charged_bytes() -> Iterator[list[int]]:
    """Every byte charged, across every ``PayloadBudget`` built while this is open.

    Instrumenting the *ledger* rather than the payload is what turns "the payload is bounded" from
    a claim about the shapes someone thought to test into one about the accounting itself. A byte
    that lands uncharged is invisible to every ceiling assertion until some other shape widens the
    gap past the reserve -- which is exactly how three of them shipped: each container happens to
    over-charge two bytes (a key is charged its ``", "`` separator, and the first entry spells
    none), and that unstated cushion covered the holes at every size anyone had measured.
    """
    total = [0]
    real_charge = PayloadBudget.charge
    real_marker = PayloadBudget.charge_marker

    def charge(self: PayloadBudget, cost: int) -> bool:
        accepted = real_charge(self, cost)
        if accepted:
            total[0] += cost
        return accepted

    def charge_marker(self: PayloadBudget, cost: int) -> None:
        real_marker(self, cost)
        total[0] += cost

    PayloadBudget.charge = charge  # type: ignore[method-assign]
    PayloadBudget.charge_marker = charge_marker  # type: ignore[method-assign]
    try:
        yield total
    finally:
        PayloadBudget.charge = real_charge  # type: ignore[method-assign]
        PayloadBudget.charge_marker = real_marker  # type: ignore[method-assign]


def _marker_named_dict() -> dict[str, Any]:
    """A mapping the width cap must cut, whose first key is the marker's own name and whose ``#2``
    through ``#9`` are taken, so the rename needs a *two-digit* suffix.

    Three bytes against the two a container's first entry leaves spare. One such dict is covered
    by the cushion and says nothing; the shapes below carry six.
    """
    out: dict[str, Any] = {"truncated_keys": 0}
    out.update({f"truncated_keys#{suffix}": 0 for suffix in range(2, 10)})
    out.update({f"k{index}": index for index in range(PREVIEW_MAX_KEYS + 1)})
    return out


def _finish_prose_case_variants() -> dict[str, Any]:
    """Every case spelling of ``summary`` and ``notes``: 2**7 + 2**5 = 160 distinct keys, all of
    which ``finish_args_preview`` matches through ``key.lower()`` and redacts on the branch that
    skips the traversal."""
    variants: dict[str, Any] = {}
    for word in ("summary", "notes"):
        for mask in range(1 << len(word)):
            spelled = "".join(
                char.upper() if mask & (1 << index) else char for index, char in enumerate(word)
            )
            variants[spelled] = "model prose " * 8
    return variants


_UNCHARGED_BYTE_SHAPES = [
    pytest.param({"a": [_marker_named_dict() for _ in range(6)]}, id="marker-name-collisions"),
    pytest.param(_finish_prose_case_variants(), id="finish-prose-case-variants"),
    pytest.param(
        {"content": "x" * 4096, "path": "notes/a.md", "plain": 1}, id="callback-branches"
    ),
    pytest.param(
        {"argument": "y" * 4096, "count": 12, "nested": {"a": [1, 2, 3], "b": {"c": "d"}}},
        id="ordinary",
    ),
]

# Every builder that spends a ``PayloadBudget``, including the two that assemble their mapping by
# hand -- the hostile payload rides in one of their fields, since their outer keys are kernel
# literals. ``public_job_artifact`` is absent on purpose: it is documented as structurally bounded
# and takes no budget at all.
_BUDGETED_BUILDERS = [
    pytest.param(lambda payload, policy: args_preview(payload, policy), id="args_preview"),
    pytest.param(
        lambda payload, policy: finish_args_preview(payload, policy), id="finish_args_preview"
    ),
    pytest.param(
        lambda payload, policy: public_result_content(payload, policy), id="public_result_content"
    ),
    pytest.param(
        lambda payload, policy: public_event_payload(payload, policy), id="public_event_payload"
    ),
    pytest.param(
        lambda payload, policy: redact_tool_arguments(payload, policy=policy),
        id="redact_tool_arguments",
    ),
    pytest.param(
        lambda payload, policy: shell_args_preview({"cwd": payload}, policy), id="shell_args_preview"
    ),
    pytest.param(
        lambda payload, policy: web_args_preview({"max_results": payload}, policy),
        id="web_args_preview",
    ),
]


@pytest.mark.parametrize("build", _BUDGETED_BUILDERS)
@pytest.mark.parametrize("payload", _UNCHARGED_BYTE_SHAPES)
def test_no_builder_publishes_a_byte_it_did_not_charge(build, payload: dict[str, Any]) -> None:
    """The invariant the ceiling rests on, asserted directly instead of sampled through payloads.

    ``_preview_value``'s docstring calls charged-at-least-cost-per-appended-byte the reason the
    serialized payload is provably no larger than the budget. It was not: the truncation marker's
    collision rename widened a key already charged at its plain spelling, the two hand-assembled
    builders never charged their outer keys, and ``finish_args_preview``'s prose redaction skipped
    the traversal that would have charged it. Each is small per occurrence; each scales with the
    payload, and the first and third put a published payload past the ceiling.

    Both directions matter, so this asserts the ledger *and* the ceiling. A shape that never
    reaches the ceiling still reddens here the moment a byte lands unpaid for, which is the point
    -- the earlier pins all sat below the size where the cushion stops covering.
    """
    with _charged_bytes() as charged:
        published = build(payload, PermissionPolicy())

    assert _widest_payload_bytes(published) <= charged[0], (
        "the payload spells bytes the budget never charged"
    )
    assert _widest_payload_bytes(published) <= APPROVAL_PAYLOAD_BYTE_BUDGET


def _collision_tree(leaf_lists: int) -> list[Any]:
    """Nested lists of marker-named dicts: the cheapest way to buy many cut containers per byte,
    which is what turns a per-container byte into a payload-sized overshoot."""
    leaves = [[_marker_named_dict() for _ in range(20)] for _ in range(leaf_lists)]
    middle = [leaves[index : index + 20] for index in range(0, len(leaves), 20)]
    return [middle[index : index + 20] for index in range(0, len(middle), 20)]


def test_a_payload_of_marker_named_keys_stays_under_the_approval_ceiling() -> None:
    """The reachable breach, on the surface where it was widest.

    The approval card carries a 1 MiB ceiling against the same 1024-byte reserve, so it holds four
    times the payload and the same slack -- and an argument mapping full of keys named like the
    marker arrived 2,027 bytes past it. Nothing in the input is exotic: it round-trips through
    ``json.dumps``/``json.loads`` unchanged and nests four levels under a cap of eight.
    """
    published = redact_tool_arguments({"a": _collision_tree(800)}, policy=PermissionPolicy())

    assert _widest_payload_bytes(published) <= APPROVAL_PAYLOAD_BYTE_BUDGET
    assert "truncated_keys#10" in json.dumps(published), "the fixture never renamed anything"


def test_a_finish_call_of_case_variant_prose_keys_stays_under_the_trace_ceiling() -> None:
    """The largest of the three, and the one no cushion was ever going to cover.

    160 keys take the uncharged redaction branch in a single payload, and the call publishes
    ``tool.call.started`` *before* ``validate_args`` rejects it -- the same order
    ``shell_args_preview`` documents for ``timeout_s`` -- so ``additionalProperties: false`` on
    ``run.finish`` does not stop it being emitted.
    """
    arguments = _finish_prose_case_variants()
    arguments.update({f"arg{index:05d}": "D" * 234 for index in range(2000)})

    published = finish_args_preview(arguments, PermissionPolicy())

    assert _widest_payload_bytes(published) <= TRACE_PAYLOAD_BYTE_BUDGET
    assert published["truncated_keys"] > 0, "the fixture must actually reach the budget"

def _reference_hex_preview(value: int) -> str:
    """The spelling this envelope shipped with: materialize it all, then keep the prefix.

    Kept as a reference implementation rather than deleted, because the replacement is a
    derivation (bit-shift to the retained digits) and "it did not raise" is blind to a
    derivation that is merely *different*. Every assertion below compares against this.
    """
    return truncate_to_bytes(format(int.__index__(value), "#x"), PREVIEW_BYTE_BUDGET)


@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
@pytest.mark.parametrize(
    "bits",
    [1, 8, 600, 636, 640, 644, 1024, 65_536],
    ids=lambda bits: f"{bits}-bit",
)
def test_the_integer_preview_is_the_prefix_the_full_spelling_would_have_given(
    sign: int, bits: int
) -> None:
    """A grid across the retention boundary, both signs, against the old expression.

    ``PREVIEW_BYTE_BUDGET`` of 160 holds ``0x`` plus 158 hex digits, so 632 bits is where
    truncation starts and the sign steals one more digit — the rows straddle both. Sizes small
    enough that the reference implementation is cheap to run, which is the point: correctness is
    checked where materializing is free, and the allocation pin below covers where it is not.
    """
    value = sign * ((1 << (bits - 1)) | 0x9E3779B97F4A7C15)

    published = preview_value("n", value, PermissionPolicy())
    spelled = published["preview"] if isinstance(published, dict) else None

    if spelled is None:  # under the threshold the integer keeps its type, unchanged
        assert published == value
    else:
        assert spelled == _reference_hex_preview(value)


def test_the_integer_threshold_reads_the_value_not_the_object() -> None:
    """The twin of the ingress refusal's rule, on the side where a raise costs the whole run.

    ``_int_hex_preview`` takes ``int.__index__`` first and says why; the threshold decision two
    functions up did not, so ``value < 0`` and unary ``-`` were handed to a model-supplied
    object. This one is the worse half: it runs inside event construction, and it is reachable
    past the refusing boundaries, because ``update_plan`` normalizes with the default
    ``refuse_unportable_scalars=False``.

    Both directions, because a subclass can answer wrongly in two ways. Raising ends the run for
    a plain ``5``; understating itself publishes an integer no writer can spell -- and it slipped
    the budget too, since ``_fragment_cost`` reads ``json.dumps``'s refusal as "cannot price
    this" and lets the fragment through uncharged.
    """
    policy = PermissionPolicy()

    assert preview_value("n", ExplodingComparisons(5), policy) == 5

    published = preview_value("n", UnderstatedInteger(1 << 16_609), policy)

    assert published["truncated"] is True
    assert published["preview"].startswith("0x")
    assert json.dumps(published), "the payload still cannot be spelled by a writer"


def test_a_strings_own_encode_cannot_widen_the_preview_cap() -> None:
    """The cap asks the value how big it is, so the value must not be the one answering.

    Model text arrives at a tool boundary as a Python object, and a ``str`` subclass may override
    ``encode``. Reporting one byte, 5,000 characters cleared a 240-byte threshold and were
    published whole through the cap whose entire purpose is that they are not -- 31x the bound, on
    a run that completes normally and reports nothing unusual.
    """
    body = UnderstatedText("s" * 5_000)
    assert len(str.encode(body, "utf-8")) == 5_000, "the fixture must really be long"

    published = preview_value("note", body, PermissionPolicy())

    assert published["truncated"] is True
    assert published["bytes"] == 5_000, "the envelope repeated the value's own answer"
    # `str.encode`, not `.encode`: asking the published prefix how long it is asks the same liar,
    # and this assertion passed against the unfixed code for exactly that reason until a mutant
    # said so.
    assert len(str.encode(published["preview"], "utf-8")) <= PREVIEW_BYTE_BUDGET


def test_a_string_that_refuses_to_be_measured_does_not_end_the_run() -> None:
    """The other direction, on the side where it costs the most.

    This traversal runs *inside event construction*, so an ``encode`` that raises here is not a
    failed tool call -- it is the emit path, and the run dies as ``internal_error``. The same
    hazard ``_is_path_redacted`` fails closed against a few lines below.
    """
    published = preview_value("note", ExplodingText("s" * 5_000), PermissionPolicy())

    assert published["truncated"] is True
    assert published["bytes"] == 5_000


def test_the_redaction_marker_reports_the_bytes_it_actually_withheld() -> None:
    """The count is the only thing this marker still tells an operator.

    Taken from the value, it read ``"bytes": 1`` for a 5,000-character secret -- which reads as
    "something small was withheld" and is the opposite of what happened.
    """
    assert redacted_value(UnderstatedText("s" * 5_000)) == {
        "redacted": True,
        "type": "str",
        "bytes": 5_000,
    }


def test_a_key_that_misreports_its_own_name_is_still_judged_by_it() -> None:
    """``lowered`` decides whether a value is a file body, and the key answers ``lower()``.

    A key spelling ``content`` while reporting something else escaped the content redaction and
    was then published under its real name -- the rule and the reader disagreeing about the same
    string.
    """
    published = preview_value(MisreportingKey("content"), "SECRET BODY", PermissionPolicy())

    assert published == {"redacted": True, "type": "str", "bytes": len("SECRET BODY")}


def test_the_shell_and_web_previews_measure_the_base_string_too() -> None:
    """The same rule at the three measuring sites outside this module.

    ``preview_command`` splits *and* measures; the web descriptors publish a byte count and a
    digest as the whole of what they say about a value they withhold, so a lying ``encode`` makes
    them describe something other than what ran.

    The shell arm is driven by ``split`` rather than by ``encode``, and the difference is the
    point: ``" ".join(...)`` already returns an exact ``str``, so an understating ``encode`` never
    reaches that measurement and an arm built on it would be green for a reason that has nothing
    to do with the guard. ``split`` is the operator this site actually hands to the value.
    """
    previewed = preview_command(ExplodingText("echo " + "y" * 5_000))
    assert len(str.encode(previewed, "utf-8")) <= COMMAND_PREVIEW_BYTE_BUDGET + len("...")

    assert public_query_preview(UnderstatedText("q" * 5_000))["bytes"] == 5_000
    assert public_url_preview(UnderstatedText("https://e.example/" + "u" * 5_000))["bytes"] == 5_018

def test_the_shared_truncators_measure_the_base_string_when_called_directly() -> None:
    """The two truncators are called from outside this module, so they are pinned from outside it.

    ``shell.preview_command`` imports ``truncate_to_bytes``, and ``truncate_inline_text`` is what
    keeps a plan ``step`` a string for a renderer that prints it. The traversal now normalizes
    ahead of both, which would leave these two green for a reason that has nothing to do with
    their own code -- so they are measured where a caller actually reaches them.
    """
    body = UnderstatedText("s" * 5_000)

    # Measured through the base slot throughout. `truncate_to_bytes` returns its *input* when the
    # input fits, so a lying value comes back as itself and `result.encode()` asks the liar a
    # second time -- which is how the first draft of this pin passed against the unfixed code.
    kept = truncate_to_bytes(body, PREVIEW_BYTE_BUDGET)
    assert len(str.encode(kept, "utf-8")) <= PREVIEW_BYTE_BUDGET

    inline = truncate_inline_text(
        body, threshold=PREVIEW_BYTE_THRESHOLD, budget=PREVIEW_BYTE_BUDGET
    )
    assert str.endswith(inline, TRUNCATION_SUFFIX), "a cut value published without saying so"
    assert len(str.encode(inline, "utf-8")) <= PREVIEW_BYTE_BUDGET + len(
        TRUNCATION_SUFFIX.encode()
    )


_REDACTING_POLICY = PermissionPolicy(redact_patterns=("secrets/**",))


def test_a_container_key_is_taken_as_base_text_before_it_names_anything() -> None:
    """The published name and the rule that judges the value must be the same string.

    A key is model-authored text, and ``str(key)`` looks like it already normalizes -- it does,
    through ``type(key).__str__``, which a subclass overrides. Both depths, because the outer
    mapping is built by the builders and the inner one by the traversal, and this repository's
    house defect is a rule bound at one of a pair.
    """
    policy = PermissionPolicy()
    redacted = {"redacted": True, "type": "str", "bytes": len("SECRET BODY")}

    top = args_preview({MisreportingText("content"): "SECRET BODY"}, policy)
    assert top == {"content": redacted}, "the key was published under the name it claimed"

    nested = preview_value(
        "payload", {"outer": {MisreportingText("content"): "SECRET BODY"}}, policy
    )
    assert nested["outer"] == {"content": redacted}


def test_a_callback_that_reads_the_key_gets_the_base_text_too() -> None:
    """``public_mapping`` hands the key to a callback, and two of them judge it themselves.

    ``finish_args_preview`` matches ``key.lower()``, so a key spelling ``summary`` and answering
    something else published the model's final prose where the redaction marker belongs. The
    traversal normalizes its own key at entry, which is why this needs the *callback* arm to say
    anything at all.
    """
    withheld = {"summary": {"redacted": True, "type": "str", "bytes": len("FINAL ANSWER")}}

    # Both spellings of the lie, because the site read `str(key)` and the fix reads the base text:
    # a subclass overriding `__str__` is what defeated the code that shipped, and one overriding
    # `lower()` is what would defeat the bare `key` a later edit could leave here instead.
    for hostile in (MisreportingText("summary"), MisreportingKey("summary")):
        published = finish_args_preview({hostile: "FINAL ANSWER"}, PermissionPolicy())
        assert published == withheld, f"{type(hostile).__name__} was believed"


def test_a_path_that_claims_to_be_empty_is_still_matched_by_the_operators_pattern() -> None:
    """``normalize_workspace_path`` asks ``raw == ""`` before it resolves anything.

    So a value whose ``__eq__`` answers that question for itself walks past ``redact_patterns``.
    Both consequences are measured: the path publishes verbatim, and -- because the same predicate
    backs ``touches_redacted_path`` -- the approval card drops its content withholding along with
    it and prints the file body the operator asked to have hidden.
    """
    hostile = EmptyClaimingPath("secrets/creds.txt")

    assert preview_value("path", hostile, _REDACTING_POLICY) == {
        "redacted": True,
        "type": "str",
        "bytes": len("secrets/creds.txt"),
    }
    assert public_path(hostile, _REDACTING_POLICY) == REDACTED_PATH
    assert touches_redacted_path({"path": hostile}, _REDACTING_POLICY)


def test_a_key_that_hides_its_own_name_cannot_switch_the_approval_card_to_open() -> None:
    """``touches_redacted_path`` walks keys at every depth, and ``lowered`` decides.

    A key spelling ``path`` while hiding that name made the walk answer "this call touches nothing
    redacted", which turns ``decision_surface`` on -- publishing both the path and the file body it
    was hiding. Both depths, for the same reason as the pin above, and both ways of hiding.
    """
    for hostile in (MisreportingText("path"), MisreportingKey("path")):
        at_top = {hostile: "secrets/creds.txt"}
        nested = {"outer": {hostile: "secrets/creds.txt"}}
        named = type(hostile).__name__

        assert touches_redacted_path(at_top, _REDACTING_POLICY), named
        assert touches_redacted_path(nested, _REDACTING_POLICY), named

        card = redact_tool_arguments(dict(at_top, content="BODY"), policy=_REDACTING_POLICY)
        assert card["content"] == {"redacted": True, "type": "str", "bytes": len("BODY")}, named


def test_an_error_message_cannot_talk_its_way_past_the_key_redaction() -> None:
    """``public_error_message`` asks the value one question, and that question is the function."""
    shouted = ShoutingText("-----BEGIN RSA PRIVATE KEY----- MIIEpAIBAAKC")

    assert public_error_message(shouted) == "[redacted-sensitive-error]"


def test_an_approval_argument_key_is_masked_by_the_name_it_really_spells() -> None:
    """``_jsonish`` is what stores the approval request's ``arguments`` and feeds the
    ``approval_key`` preimage, and it converted keys with ``str(key)``. A key spelling ``api_key``
    and answering ``harmless`` was published unmasked, in the record a human reads to decide."""
    stored = _jsonish({MisreportingText("api_key"): "sk-live-XXXX"})

    assert list(stored) == ["api_key"]
    assert redact_tool_arguments(stored, policy=PermissionPolicy())["api_key"] != "sk-live-XXXX"

def test_the_hand_assembled_builders_read_the_argument_they_were_given() -> None:
    """The two builders that convert their arguments themselves, at the four model-facing fields.

    ``shell_args_preview`` and ``web_args_preview`` take their values through ``str(...)``, which
    is ``type(value).__str__`` -- so a command spelling ``rm -rf /`` and answering ``harmless``
    published the harmless one to the operator while the real one ran, and the web descriptors
    computed their digest over it. The env *key* is the same question one container down.
    """
    policy = PermissionPolicy()

    shell = shell_args_preview(
        {
            "command": MisreportingText("rm -rf / --no-preserve-root"),
            "env": {MisreportingText("AWS_SECRET_ACCESS_KEY"): "v"},
        },
        policy,
    )
    assert shell["command_preview"] == "rm -rf / --no-preserve-root"
    assert shell["env_keys"] == ["AWS_SECRET_ACCESS_KEY"]

    web = web_args_preview(
        {
            "query": MisreportingText("q" * 40),
            "url": MisreportingText("https://e.example/" + "u" * 40),
        },
        policy,
    )
    assert web["query_preview"]["bytes"] == 40
    assert web["url_preview"]["bytes"] == len("https://e.example/") + 40


def test_previewing_a_huge_integer_does_not_materialize_its_whole_spelling() -> None:
    """The envelope must cost the preview, not the number.

    ``format(v, "#x")`` is linear in the bit length, so a sparse big integer — cheap to build,
    and reachable through ``update_plan``, which normalizes without the refusing boundaries —
    allocated a full hexadecimal string inside event construction to keep 158 characters of it.
    Measured on a 20 Mbit value: 10.0 MB peak through this call, against 0.4 KB for the shifted
    derivation. Pinned with an order of magnitude of headroom rather than tightly, because this
    is a resource bound and the exact figure moves with allocator behaviour; what must not
    survive is *proportional to the input*.
    """
    huge = 1 << 20_000_000  # built outside the traced region: only the preview is measured
    items = [{"step": "compute", "status": "pending", "n": huge}]

    tracemalloc.start()
    try:
        published = preview_value("items", items, PermissionPolicy(), list_marker=False)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 1_000_000, f"the preview allocated {peak / 1e6:.1f} MB to publish 158 characters"
    assert published[0]["n"] == {
        "type": "int",
        "preview": _reference_hex_preview(huge),
        "truncated": True,
    }


def test_a_url_descriptor_is_bounded_on_started_exactly_as_on_the_events_beside_it() -> None:
    """The same descriptor, for the same call, must not differ by which surface prints it.

    ``public_url_preview`` copies ``scheme`` and ``domain`` out of the URL verbatim, and both are
    model-controlled strings of unbounded length: a hostname is valid at any length, and so is a
    scheme. ``web_args_preview`` assembled that fragment by hand and neither previewed nor charged
    it, so a 4 MB hostname published a 4,000,085-byte ``args_preview`` — 15.3x the ceiling, and
    growing linearly with the URL, so no ceiling at all.

    The web service's own ``.finished``/``.failed`` events carry the identical fragment through
    ``public_event_payload``, which does bound it. So the two surfaces of one call disagreed:
    549 bytes there, 4 MB here. Pinned as that agreement rather than as a byte count, because the
    disagreement is the defect — a rule proven on one of two parallel halves is this repository's
    house shape, and the halves here are one function apart.
    """
    policy = PermissionPolicy()
    url = "http://" + ("a" * 1_000_000) + "/p"

    started = web_args_preview({"url": url}, policy)
    beside = public_event_payload({"url_preview": public_url_preview(url)}, policy)

    assert started["url_preview"] == beside["url_preview"]
    assert _widest_payload_bytes(started) <= TRACE_PAYLOAD_BYTE_BUDGET
    assert started["url_preview"]["domain"]["truncated"] is True


def test_an_ordinary_url_descriptor_is_published_unchanged() -> None:
    """The guard on the pin above: bounding the descriptor must not reshape ordinary ones."""
    policy = PermissionPolicy()
    url = "https://example.com/docs?q=1"

    assert web_args_preview({"url": url}, policy)["url_preview"] == public_url_preview(url)
    assert web_args_preview({"query": "hello"}, policy)["query_preview"] == public_query_preview(
        "hello"
    )


# --------------------------------------------------------------------------------------
# The budget's roots, read off the source rather than remembered
# --------------------------------------------------------------------------------------

_KERNEL_PACKAGE = Path(monoid_agent_kernel.__file__).resolve().parent

_TRAVERSAL_ENTRIES = frozenset({"preview_value", "public_mapping"})
"""The two entry points into the bounded traversal. ``_preview_value`` is the traversal's own
recursion, reached only through them, so a call to either one is where a ``PayloadBudget`` is
born or must be threaded."""

_KNOWN_TRAVERSAL_ENTRY_OWNERS = frozenset(
    {
        ("core/tool_approval.py", "redact_tool_arguments"),
        ("loop.py", "AgentToolContext.emit_artifact"),
        ("loop.py", "AgentToolContext.update_plan"),
        ("public_view.py", "_budgeted_field"),
        ("public_view.py", "args_preview"),
        ("public_view.py", "finish_args_preview"),
        ("public_view.py", "public_event_payload"),
        ("public_view.py", "public_job_artifact"),
        ("public_view.py", "public_result_content"),
    }
)
"""Every function in the kernel that enters the traversal, by file and outermost function.

``shell_args_preview`` and ``web_args_preview`` are absent on purpose: they enter through
``_budgeted_field``, whose signature makes the budget an argument the caller must produce.
"""


def _traversal_entry_sites() -> list[tuple[str, str, ast.Call]]:
    """Every call to a traversal entry point in the kernel, with its owning function.

    Matches ``ast.Attribute`` as well as ``ast.Name`` so a refactor to module-qualified calls
    cannot leave this census silently green. Nested functions and lambdas attribute to their
    outermost enclosing function, because that is the frame that owns the payload: the
    ``public_result_content`` callback spending its builder's budget is one root, not two.
    """

    sites: list[tuple[str, str, ast.Call]] = []
    for path in sorted(_KERNEL_PACKAGE.rglob("*.py")):
        relative = path.relative_to(_KERNEL_PACKAGE).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                callee = func.id
            elif isinstance(func, ast.Attribute):
                callee = func.attr
            else:
                continue
            if callee not in _TRAVERSAL_ENTRIES:
                continue
            chain: list[ast.AST] = []
            cursor = parents.get(node)
            while cursor is not None:
                if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    chain.append(cursor)
                cursor = parents.get(cursor)
            chain.reverse()
            named: list[str] = []
            for scope in chain:
                named.append(scope.name)
                if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    break
            sites.append((relative, ".".join(named) or "<module>", node))
    return sites


def test_every_traversal_entry_in_the_kernel_is_a_known_root() -> None:
    """A new payload root must be classified by its author, not discovered by a review round.

    A hand-kept root list is an enumeration of what its author remembered, and every hand-kept
    census in this repository has fallen one generation behind the code. This one is read off the
    source: adding a call to ``preview_value`` or ``public_mapping`` anywhere in the kernel adds
    its owner here, and the ``==`` fails in both directions — a new root that nobody budgeted,
    and a table entry whose root no longer exists, so the pins above it test nothing.

    If this set grows: the new call site is a payload root or a builder's field. Decide which
    surface it publishes to (trace or approval), thread or self-own a ``PayloadBudget``
    accordingly, and only then extend the table.
    """

    found = {(relative, owner) for relative, owner, _call in _traversal_entry_sites()}

    assert found == _KNOWN_TRAVERSAL_ENTRY_OWNERS


def test_a_function_entering_the_traversal_twice_threads_one_budget() -> None:
    """The reverted bound's first failure shape, pinned as a relation rather than a list.

    ``PREVIEW_MAX_NODES`` was born per-top-level-key: each callback invocation started a fresh
    allowance, and 400 keys times a bounded value was 42 MB of bounded values. The budget is
    per payload precisely because it is created once and *threaded* — so a function that enters
    the traversal more than once is assembling one payload, and every one of its entries must
    name the budget it spends. A second entry without the keyword is a fresh wrapper-owned
    budget: per-key accounting, reborn.

    Single-entry functions may omit the keyword. There the wrapper self-owns a fresh budget,
    which for a single-root payload *is* per-payload accounting.
    """

    by_owner: dict[tuple[str, str], list[ast.Call]] = {}
    for relative, owner, call in _traversal_entry_sites():
        by_owner.setdefault((relative, owner), []).append(call)

    assert any(len(calls) > 1 for calls in by_owner.values()), (
        "census self-check: no multi-entry function found, so the relation below matched nothing"
    )

    offenders = {
        owner: [
            call.lineno
            for call in calls
            if not any(
                keyword.arg in ("_payload_budget", "payload_budget")
                for keyword in call.keywords
            )
        ]
        for owner, calls in by_owner.items()
        if len(calls) > 1
    }
    offenders = {owner: lines for owner, lines in offenders.items() if lines}

    assert not offenders, {
        "sites": offenders,
        "hint": "this function enters the traversal more than once but lets some entries "
        "self-own a budget; each unthreaded entry restarts the allowance, which is the "
        "per-key accounting the payload budget exists to end",
    }


def _hand_assembled_builder_fields() -> dict[str, list[tuple[int, bool]]]:
    """Every field a hand-assembling builder publishes, with whether it went through the budget.

    A published field is a value in a ``dict`` literal that is returned, or the right-hand side of
    an assignment into a subscript — the two ways these builders put a key on the wire. Known
    syntactic reach: a builder that assembled its mapping by ``update()`` or a comprehension would
    not be seen, which is a fact about today's code rather than a guarantee.
    """

    fields: dict[str, list[tuple[int, bool]]] = {}
    for path in sorted(_KERNEL_PACKAGE.rglob("*.py")):
        relative = path.relative_to(_KERNEL_PACKAGE).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for owner in ast.walk(tree):
            if not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            budgeted_calls = [
                node
                for node in ast.walk(owner)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_budgeted_field"
            ]
            if not budgeted_calls:
                continue
            published: list[ast.expr] = []
            for node in ast.walk(owner):
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
                    published.extend(value for value in node.value.values if value is not None)
                elif isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Subscript) for target in node.targets
                ):
                    published.append(node.value)
            fields[f"{relative}:{owner.name}"] = [
                (
                    value.lineno,
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "_budgeted_field",
                )
                for value in published
            ]
    return fields


def test_a_builder_that_budgets_one_field_budgets_all_of_them() -> None:
    """The half-threaded builder, stated as consistency so it needs no list of fields.

    ``web_args_preview`` created a budget, spent it on nine descriptors, and appended two more —
    ``query_preview`` and ``url_preview`` — straight from their helpers. The URL one copies a
    hostname and a scheme verbatim, so the ceiling this branch introduced was defeated by the very
    builder that declared it, and the builder's own docstring ("every descriptor below is
    previewed rather than copied") was false about the field most able to carry text.

    Written as "a function that budgets one field budgets all of them" rather than as a table of
    builders and their fields: a table is an enumeration of what its author remembered, and a
    field added next year is exactly what neither the author nor the table will remember. The
    bounded-by-construction fields go through the helper too — a bool costs five charged bytes and
    buys an invariant that needs no footnote, where leaving them out makes the ceiling "the budget
    plus whatever the unrouted fields happen to add", which is not a ceiling anyone can check.
    """

    inconsistent = {
        owner: {
            "budgeted": [line for line, ok in found if ok],
            "unbudgeted": [line for line, ok in found if not ok],
        }
        for owner, found in _hand_assembled_builder_fields().items()
        if len({ok for _line, ok in found}) > 1
    }

    assert _hand_assembled_builder_fields(), (
        "census self-check: no hand-assembling builder found, so this matched nothing"
    )
    assert not inconsistent, {
        "sites": inconsistent,
        "hint": "this builder charges some of its published fields against the payload budget "
        "and appends others straight from a helper; an uncharged field is outside the ceiling, "
        "and if its helper copies model-controlled text the ceiling is defeated entirely",
    }


def test_a_type_that_answers_for_its_own_name_cannot_escape_the_refusal() -> None:
    """The refusal message reads `type(value).__name__`, which dispatches to the *metaclass*.

    A metaclass that raises there replaces `UnportableScalarError` with an arbitrary exception, at
    the boundary whose entire job is to convert arbitrary exceptions into classified tool failures --
    the failure is inside the error path of the mechanism that exists to prevent unclassified
    failures. Four ways to answer, because the base slot alone closes only three: a class whose
    `__name__` is a `str` subclass moves the question from the metaclass onto the name object.
    """
    for hostile in HOSTILE_NAMED_TYPES:
        with pytest.raises(UnportableScalarError):
            normalize_json_ingress({"a": hostile()}, refuse_unportable_scalars=True)


def test_the_preview_names_a_type_without_letting_it_answer() -> None:
    """The same read, past the refusing boundaries and inside event construction.

    `update_plan` normalizes with the default `refuse_unportable_scalars=False`, so these envelopes
    are what a Python-object value meets with no boundary in front of it; a raise here ends the run.
    """
    policy = PermissionPolicy()

    for hostile in HOSTILE_NAMED_TYPES:
        published = preview_value("n", hostile(), policy)
        assert published["truncated"] is True
        assert type(published["type"]) is str

        marker = redacted_value(hostile())
        assert marker["redacted"] is True
        assert type(marker["type"]) is str

    # The lying arm publishes a name; the point is that it is not the name the value chose.
    assert preview_value("n", ImpersonatingName(), policy)["type"] == "ImpersonatingName"
    assert redacted_value(ImpersonatingName())["type"] == "ImpersonatingName"


def test_a_published_type_name_is_bounded_like_every_other_published_string() -> None:
    """A class name is legal at any length, and two of these sites publish it uncharged.

    Measured before this bound: a 1,000,000-character class name published a 1,000,038-byte payload
    against a 262,144-byte ceiling -- 3.8x, from a value the traversal had already refused, through
    the fallback that replaces it. The name is the only unbounded term at those sites.
    """
    policy = PermissionPolicy()

    # Large enough that the envelope carrying it does not fit the ceiling, which is what sends the
    # traversal down the fallback that publishes the name *uncharged* -- the site the measurement
    # above came from. `_charge_terminal_marker` deducts unconditionally, so nothing else stops it.
    huge = hugely_named_object(300_000)

    published = preview_value("n", huge, policy)
    assert len(published["type"]) <= 64, "the published name is not bounded"
    assert len(json.dumps({"n": published}).encode("utf-8")) <= TRACE_PAYLOAD_BYTE_BUDGET

    assert len(redacted_value(hugely_named_object(10_000))["type"]) <= 64

    with pytest.raises(UnportableScalarError) as refusal:
        normalize_json_ingress({"a": huge}, refuse_unportable_scalars=True)
    assert len(str(refusal.value)) <= 160, "the refusal message carries the name unbounded"


def test_every_ordinary_type_keeps_the_name_it_publishes_today() -> None:
    """The one regression this fix could cause: a different string on the wire.

    These names are published into events, so `portable_type_name` has to agree with
    `type(value).__name__` for every value that is not answering for itself. An equality oracle
    rather than "it did not raise" -- the same lesson a refactor on this branch already earned.
    """
    import collections
    import datetime
    import decimal
    import enum
    import fractions
    import re as _re
    import uuid
    from dataclasses import dataclass

    class Colour(enum.Enum):
        RED = 1

    class Level(enum.IntEnum):
        LOW = 1

    @dataclass
    class Boxed:
        value: int

    class Slotted:
        __slots__ = ("x",)

    Point = collections.namedtuple("Point", "x y")

    values = [
        None, True, 1, 1.5, "s", b"b", bytearray(b"b"), (), [], {}, set(), frozenset(),
        object(), type, int, Ellipsis, NotImplemented, range(3), slice(1), memoryview(b"x"),
        decimal.Decimal("1"), fractions.Fraction(1, 2), uuid.uuid4(), datetime.date(2020, 1, 1),
        datetime.datetime(2020, 1, 1), datetime.timedelta(1), datetime.timezone.utc,
        _re.compile("x"), _re.match("x", "x"), Colour.RED, Level.LOW, Boxed(1), Slotted(),
        Point(1, 2), collections.OrderedDict(), collections.deque(), collections.Counter(),
        ValueError("x"), KeyError("x"), Exception(), type("Dynamic", (), {})(),
        (lambda: None), iter([]), (i for i in []), PermissionPolicy(),
        MisreportingText("x"), UnderstatedInteger(1), EmptyClaimingPath("p"),
    ]

    divergent = [
        (type(value).__name__, portable_type_name(value))
        for value in values
        if portable_type_name(value) != type(value).__name__
    ]
    assert not divergent, divergent


def test_a_container_that_answers_for_its_own_type_name_is_capped_anyway() -> None:
    """The depth cap and the cycle guard name a *container*, so the hostile shape there is not a
    scalar. Both markers are built inside event construction, and both read the name off the class.
    """
    policy = PermissionPolicy()

    # Exactly `PREVIEW_MAX_DEPTH` wrappers, so the hostile container is the value the cap lands on.
    # One more and an ordinary dict is capped first, and the marker never reads the hostile name.
    deep: Any = HostileNamedDict({"leaf": 1})
    for _ in range(PREVIEW_MAX_DEPTH):
        deep = {"next": deep}
    published = preview_value("n", deep, policy)
    cursor = published
    while isinstance(cursor, dict) and "next" in cursor:
        cursor = cursor["next"]
    assert cursor["truncated"] is True
    assert cursor["depth_exceeded"] == PREVIEW_MAX_DEPTH
    assert type(cursor["type"]) is str

    circular = HostileNamedList([1])
    circular.append(circular)
    inner = preview_value("n", {"outer": circular}, policy)["outer"]
    assert any(
        isinstance(item, dict) and item.get("circular") is True and type(item["type"]) is str
        for item in inner
    ), inner


def test_a_type_name_costs_the_same_bound_in_every_script() -> None:
    """The cap was written in characters and the budget is charged in bytes.

    `_fragment_cost` measures the way the widest sink spells a fragment -- default separators,
    non-ASCII escaped -- so a 64-character Hangul name costs 415 bytes where the same length of
    ASCII costs 95. Capping by characters is not a bound on the surface that pays for it; this is
    the third time this repository has published a byte ceiling measured in the wrong unit.

    The cumulative arm matters because these fallback markers spend through `charge_marker`, which
    deducts unconditionally the way a marker must: whatever a fixed-field builder emits after
    exhaustion, one script must not buy more of it than another.
    """
    ceiling = _fragment_cost({"truncated": True, "type": "z" * 64})
    policy = PermissionPolicy()

    for script in ("\ud55c", "\u6f22", "\U0001f600"):  # Hangul, CJK, astral (a surrogate pair)
        hostile = type(script * 64, (), {})()
        published = preview_value("n", hostile, policy)
        assert _fragment_cost(published) <= ceiling, (script, _fragment_cost(published))
        assert _fragment_cost(redacted_value(hostile)) <= ceiling, script

    def terminal_spend(script: str) -> int:
        """What a builder's fixed fields spend once the regular budget is gone."""
        budget = PayloadBudget(TRACE_PAYLOAD_BYTE_BUDGET)
        budget.remaining = 0
        for index in range(13):
            _budgeted_field(f"f{index}", type(script * 64, (), {})(), policy, budget)
        return -budget.remaining

    assert terminal_spend("\ud55c") <= terminal_spend("z"), "a non-ASCII name buys more marker"
