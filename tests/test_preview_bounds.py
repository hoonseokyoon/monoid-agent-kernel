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

import json
import time
from typing import Any

import pytest

from monoid_agent_kernel.core.tool_approval import redact_tool_arguments
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
