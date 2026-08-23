from __future__ import annotations

import ast
import contextlib
import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from support.hostile_scalars import (
    HOSTILE_NAMED_TYPES,
    ExplodingComparisons,
    MisreportingItems,
    MisreportingKey,
    OverstatedList,
    SubstitutingList,
    UnderstatedDict,
    UnderstatedInteger,
    UnderstatedList,
    UnderstatedText,
    UniterableText,
    hugely_named_object,
)

import monoid_agent_kernel
from monoid_agent_kernel.core.json_ingress import (
    MAX_PORTABLE_CONTAINER_DEPTH,
    UnportableContainerError,
    UnportableScalarError,
    is_finite_json_number,
    is_portable_json_integer,
    loads_json_ingress,
    loads_model_envelope_json_ingress,
    loads_model_json_ingress,
    normalize_json_ingress,
    normalize_unicode_scalars,
    portable_class_name,
    portable_type_name,
)
from monoid_agent_kernel.permissions import _LegacyPathPattern
from monoid_agent_kernel.providers.base import (
    ModelAdapterError,
    ModelTurn,
    ToolCall,
    normalize_model_turn,
)
from monoid_agent_kernel.tools.base import (
    ToolResult,
    ToolSpec,
    normalize_tool_result,
    normalize_tool_spec,
)


def test_unicode_normalization_preserves_scalars_and_repairs_surrogates() -> None:
    ordinary = "한글 😀"

    assert normalize_unicode_scalars(ordinary) is ordinary
    assert normalize_unicode_scalars("\ud83d\ude00") == "😀"
    assert normalize_unicode_scalars("a\ud800b\udc00c") == "a�b�c"


def test_finite_json_number_check_is_total_for_arbitrarily_large_integers() -> None:
    class IntegerSubclass(int):
        pass

    assert is_finite_json_number(0)
    assert is_finite_json_number(1.5)
    assert not is_finite_json_number(10**400)
    assert not is_finite_json_number(float("inf"))
    assert not is_finite_json_number(True)
    assert not is_finite_json_number(IntegerSubclass(1))


def test_portable_json_integer_uses_the_writer_and_reader_digit_bound() -> None:
    class IntegerSubclass(int):
        pass

    assert is_portable_json_integer(0)
    assert is_portable_json_integer(-1)
    assert not is_portable_json_integer(10**5000)
    assert not is_portable_json_integer(True)
    assert not is_portable_json_integer(1.0)
    assert not is_portable_json_integer(IntegerSubclass(1))


def test_json_ingress_normalizes_nested_keys_values_and_nonfinite_numbers() -> None:
    source = {
        "\ud800-key": [float("nan"), float("inf"), -float("inf")],
        "nested": {"text": "\ud83d\ude00"},
    }

    normalized = normalize_json_ingress(source)

    assert normalized == {
        "�-key": [None, None, None],
        "nested": {"text": "😀"},
    }
    assert math.isnan(source["\ud800-key"][0])
    assert source["nested"]["text"] == "\ud83d\ude00"


def test_json_ingress_refuses_key_collisions_after_normalization() -> None:
    lone_surrogate = "\ud800"
    replacement_character = "�"
    with pytest.raises(ValueError, match="keys collide"):
        normalize_json_ingress({lone_surrogate: 1, replacement_character: 2})


def test_json_ingress_is_iterative_and_preserves_graph_topology() -> None:
    leaf: list[object] = []
    deep = leaf
    for _ in range(1_500):
        child: list[object] = []
        deep.append(child)
        deep = child

    shared = {"value": float("nan")}
    cycle: list[object] = []
    cycle.append(cycle)
    source = {"deep": leaf, "left": shared, "right": shared, "cycle": cycle}

    normalized = normalize_json_ingress(source)

    cursor = normalized["deep"]
    for _ in range(1_500):
        cursor = cursor[0]
    assert cursor == []
    assert normalized["left"] is normalized["right"]
    assert normalized["left"] == {"value": None}
    assert normalized["cycle"][0] is normalized["cycle"]


def test_a_container_does_not_decide_what_the_walk_copies() -> None:
    """The copy IS the record, so the container that produced it does not get to describe it.

    Every later reader sees what this walk copied and never the original -- the checkpoint's
    ``asdict``, the transcript, the preview, the operator's redact patterns -- so a ``dict`` or
    ``list`` subclass overriding ``items``, ``__len__`` or ``__getitem__`` was writing the record.
    The scalar generations' argument does not carry here: measured, ``json.dumps`` reads a ``list``
    subclass's real storage and takes a ``dict`` subclass's overridden ``items()``, so "what will a
    writer spell" answers opposite ways on the two halves and settles neither.
    """
    assert normalize_json_ingress(UnderstatedList([1, 2, 3])) == [1, 2, 3]
    assert normalize_json_ingress(SubstitutingList([1, 2])) == [1, 2]
    assert normalize_json_ingress(MisreportingItems({"content": "SECRET", "safe": 1})) == {
        "content": "SECRET",
        "safe": 1,
    }
    assert normalize_json_ingress(UnderstatedDict({"content": "SECRET", "safe": 1})) == {
        "content": "SECRET",
        "safe": 1,
    }
    # Nested, because a child is reached through a different arm of the walk than the root is.
    assert normalize_json_ingress({"outer": UnderstatedList([1, 2, 3])}) == {"outer": [1, 2, 3]}
    assert normalize_json_ingress([MisreportingItems({"a": 1, "b": 2})]) == [{"a": 1, "b": 2}]


def test_a_container_that_overstates_its_length_cannot_crash_the_boundary() -> None:
    """``len`` sizes the copy and ``__getitem__`` fills it -- two questions with no consistent answer.

    The raw ``IndexError`` came out of ``normalize_tool_result``, which converts
    ``UnportableScalarError`` and nothing else, so an unclassified exception escaped the boundary
    whose whole job is to classify. That is the container twin of the crash the scalar refusal
    closed, and it is reachable from any custom or MCP tool handler.
    """
    result = normalize_tool_result(ToolResult(ok=True, content={"items": OverstatedList([1, 2])}))

    assert result.content == {"items": [1, 2]}


def test_a_container_reachable_from_itself_is_refused_at_the_python_object_routes() -> None:
    """The walk's memo makes a cycle look finished, and the writers it hands the copy to disagree.

    ``normalize_json_ingress`` records a container *before* walking its children, so the second
    visit short-circuits and the walk ends -- for a self-referential list, at path depth 2 -- with
    a self-referential copy. ``json.dumps`` then raises ``ValueError: Circular reference
    detected`` and ``dataclasses.asdict`` raises ``RecursionError``, neither of them at the
    boundary that accepted it. So a cycle needs a refusal of its own: no depth bound will ever see
    one, because the memo stops the walk before any depth accumulates.
    """
    cyclic: dict = {}
    cyclic["self"] = cyclic

    with pytest.raises(UnportableContainerError, match="reachable from itself"):
        normalize_json_ingress(cyclic, refuse_unportable=True)

    nested: list = [1]
    nested.append(nested)
    with pytest.raises(UnportableContainerError, match="reachable from itself"):
        normalize_json_ingress({"outer": nested}, refuse_unportable=True)

    # Through a boundary, so the classified refusal an operator reads is pinned too.
    boundary_cycle: dict = {}
    boundary_cycle["self"] = boundary_cycle
    with pytest.raises(Exception) as refused:
        normalize_tool_result(ToolResult(ok=True, content=boundary_cycle))
    assert getattr(refused.value, "error_code", "") == "tool_result_unportable"


def test_a_value_shared_twice_is_not_a_cycle() -> None:
    """A memo hit is not the question; an *ancestor* hit is.

    The refusing boundaries carry shared graphs on purpose -- the preview renders a value shared
    twice twice, and the copy keeps the sharing -- so refusing on any second visit would convict
    every DAG that ever reaches a tool result.
    """
    shared = {"leaf": 1}
    normalized = normalize_json_ingress({"a": shared, "b": shared}, refuse_unportable=True)

    assert normalized["a"] is normalized["b"]
    assert normalized["a"] == {"leaf": 1}

    # The second reference reached *below* the first, which is where a walk that confused sharing
    # with recursion would answer differently depending on key order.
    deeper = normalize_json_ingress({"a": shared, "b": {"c": shared}}, refuse_unportable=True)
    assert deeper["a"] is deeper["b"]["c"]


def _nested_arguments(containers: int) -> dict:
    """A chain of exactly ``containers`` containers, the innermost holding a scalar."""

    inner: object = {"leaf": 1}
    for _ in range(containers - 1):
        inner = {"next": inner}
    assert isinstance(inner, dict)
    return inner


def _ask_path_refuses(arguments: dict) -> bool:
    from monoid_agent_kernel.core.tool_approval import build_tool_approval_task_request

    spec = ToolSpec(
        id="custom.deep",
        description="d",
        input_schema={"type": "object"},
        capability="",
        side_effect="read",
        handler=lambda _context, _arguments: ToolResult(ok=True),
    )
    try:
        build_tool_approval_task_request(
            spec=spec,
            binding_id="b",
            model_name="m",
            call_name="custom_deep",
            call_id="c1",
            arguments=arguments,
            reason="r",
            turn_id="t1",
            tool_event_id=None,
        )
    except ValueError:
        return True
    return False


def _allow_path_refuses(arguments: dict) -> bool:
    turn = ModelTurn(
        response_id="r1",
        tool_calls=(ToolCall(id="c1", name="custom_deep", arguments=arguments),),
    )
    try:
        normalize_model_turn(turn)
    except ModelAdapterError:
        return True
    return False


def test_the_two_approval_paths_answer_the_same_structure_the_same_way() -> None:
    """One structure, both gates, one verdict list -- asserted as a list so the shape is the claim.

    The bound has always existed on the ``ask`` path, through the approval-request builder, and
    never on ``allow``: those arguments went into the message history and out through
    ``RunCheckpoint.to_json``, whose ``dataclasses.asdict`` recurses in pure Python and dies at
    492 containers while the model-JSON decoder admits 512 -- so a depth in that window was
    accepted by every gate it met and killed the run at the checkpoint writer, with
    ``_CheckpointPersistError`` out of ``run_once`` and no classified record of why.

    Comparing the two *lists* rather than the two bounds is deliberate: an off-by-one on either
    side shows up as a differing element, and a rule bound to one of two parallel halves is
    exactly this repository's recurring defect. The sweep straddles the bound so neither list can
    be constant, and that is asserted too -- two lists agreeing on "never refuse" would pass.
    """
    depths = range(MAX_PORTABLE_CONTAINER_DEPTH - 2, MAX_PORTABLE_CONTAINER_DEPTH + 4)
    structures = {depth: _nested_arguments(depth) for depth in depths}

    ask = [_ask_path_refuses(structures[depth]) for depth in depths]
    allow = [_allow_path_refuses(structures[depth]) for depth in depths]

    assert ask == allow, f"ask={ask} allow={allow} over depths {list(depths)}"
    assert True in ask and False in ask, f"the sweep never crossed the bound: {ask}"


def test_a_cyclic_model_turn_is_a_classified_adapter_failure() -> None:
    """The fifth Python-object route: an adapter's own ``ToolCall.arguments``.

    That is the route to the checkpoint -- the arguments ride the assistant message into
    ``state.messages`` and out through ``RunCheckpoint.to_json`` -- and it was the one boundary of
    the five that did not refuse anything. This is the arm with no settled outcome, where the
    refusal escapes and is converted; the other two arms are the test below.
    """
    cyclic: dict = {}
    cyclic["self"] = cyclic
    turn = ModelTurn(
        response_id="r1",
        tool_calls=(ToolCall(id="c1", name="custom_tool", arguments=cyclic),),
    )

    with pytest.raises(ModelAdapterError, match="non-portable"):
        normalize_model_turn(turn)


@pytest.mark.parametrize(
    "settled",
    [
        pytest.param({"final_text": "the answer the provider was paid for"}, id="final_text"),
        pytest.param({"stop_reason": "length"}, id="stop_reason"),
    ],
)
def test_a_refused_call_beside_a_settled_answer_is_dropped_and_not_reported(
    settled: dict[str, str],
) -> None:
    """The other two arms of the same boundary, and they do NOT raise. Pinned, not assumed.

    ``_normalize_model_turn`` wraps each call's normalization in an ``except`` that re-raises only
    when the turn has no settled outcome. That predates this bound -- it exists so a legacy
    adapter's odd extra entry beside a paid answer does not fail the turn -- and it catches
    ``Exception``, so it catches this refusal too and the call is simply not appended.

    Left as it is, deliberately, and this test is the record of that decision. What the bound
    exists to prevent is unportable arguments reaching ``RunCheckpoint.to_json``; dropping the
    call prevents exactly that, and a settled answer wins in ``AgentLoop`` so the call was never
    going to execute. Raising instead would discard an answer the provider has already been paid
    for in order to stop something that is already stopped. What is genuinely lost is the RECORD
    that a call was refused, silently -- reporting it needs an observation channel this function
    does not have, and inventing one here would be a redesign of a pure normalizer.

    The asymmetry was noted in a docstring before it was pinned, which is why it survived a
    review: a sentence saying "this arm swallows it" reads as a description of the world, and
    only an assertion makes it a decision.
    """
    cyclic: dict = {}
    cyclic["self"] = cyclic
    turn = ModelTurn(
        response_id="r1",
        tool_calls=(ToolCall(id="c1", name="custom_tool", arguments=cyclic),),
        **settled,
    )

    normalized = normalize_model_turn(turn)

    assert normalized.tool_calls == ()
    # ...and the answer that was paid for survives, which is the whole reason for the leniency.
    # Absent fields normalize to ``None`` rather than ``""``, so the expectation reads them off
    # the same mapping that built the turn instead of spelling a default of its own.
    assert (normalized.final_text, normalized.stop_reason) == (
        settled.get("final_text"),
        settled.get("stop_reason"),
    )
    # A portable call on the same shape is kept, so this is a refusal being dropped and not the
    # settled arm discarding calls wholesale.
    kept = normalize_model_turn(
        ModelTurn(
            response_id="r1",
            tool_calls=(ToolCall(id="c1", name="custom_tool", arguments={"ok": 1}),),
            **settled,
        )
    )
    assert len(kept.tool_calls) == 1


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_external_json_rejects_nonfinite_constants(constant: str) -> None:
    with pytest.raises(ValueError, match="non-finite number"):
        loads_json_ingress('{"value": ' + constant + "}")


@pytest.mark.parametrize("number", ["1e9999", "-1e9999"])
def test_external_json_rejects_float_overflow(number: str) -> None:
    with pytest.raises(ValueError, match="non-finite number"):
        loads_json_ingress('{"max_duration_s": ' + number + "}")


@pytest.mark.parametrize("loader", [loads_json_ingress, loads_model_json_ingress])
@pytest.mark.parametrize(
    "document",
    [
        '{"approved":false,"approved":true}',
        '{"quota":{"max_calls":1,"max_calls":0}}',
        '{"\\ud800":1,"\\ufffd":2}',
    ],
)
def test_json_text_decoders_reject_duplicate_object_keys(loader, document: str) -> None:
    with pytest.raises(json.JSONDecodeError, match="duplicate JSON object key"):
        loader(document)


def test_external_json_reports_excessive_nesting_as_invalid_json() -> None:
    document = "[" * 2_000 + "0" + "]" * 2_000

    with pytest.raises(ValueError, match="JSON nesting exceeds the parser limit"):
        loads_json_ingress(document)


@pytest.mark.parametrize(
    "loader",
    [loads_json_ingress, loads_model_json_ingress, loads_model_envelope_json_ingress],
)
def test_json_nesting_limit_cannot_be_bypassed_by_a_string_subclass(loader) -> None:
    class EmptyIteratorString(str):
        def __iter__(self):
            return iter(())

    document = EmptyIteratorString("[" * 513 + "0" + "]" * 513)

    with pytest.raises(ValueError, match="nesting exceeds the parser limit"):
        loader(document)


@pytest.mark.parametrize("loader", [loads_json_ingress, loads_model_json_ingress])
def test_json_text_decoders_apply_a_portable_integer_digit_limit(loader) -> None:
    with pytest.raises(json.JSONDecodeError, match="integer decoder limit exceeded"):
        loader("1" * 5_000)


def test_json_nesting_limit_ignores_delimiters_inside_strings() -> None:
    payload = {"text": '\\"[{]}' * 1_000}

    assert loads_json_ingress(json.dumps(payload)) == payload
    assert loads_model_json_ingress(json.dumps(payload)) == payload


def test_model_json_substitutes_nonfinite_values_and_rejects_excessive_nesting() -> None:
    assert loads_model_json_ingress(
        '{"values": [NaN, Infinity, -Infinity, 1e9999], "text": "\\ud800"}'
    ) == {"values": [None, None, None, None], "text": "�"}

    document = "[" * 2_000 + "0" + "]" * 2_000
    with pytest.raises(ValueError, match="model JSON nesting exceeds the parser limit"):
        loads_model_json_ingress(document)


def test_model_envelope_parser_preserves_nonfinite_markers_for_control_validation() -> None:
    parsed = loads_model_envelope_json_ingress(
        '{"stop_reason": NaN, "arguments": {"score": 1e9999}}'
    )

    assert math.isnan(parsed["stop_reason"])
    assert math.isinf(parsed["arguments"]["score"])


@pytest.mark.parametrize("key", [None, 1, 1.5])
def test_json_ingress_rejects_non_string_object_keys(key: object) -> None:
    with pytest.raises(ValueError, match="JSON object keys must be strings"):
        normalize_json_ingress({key: "value"})


def test_external_json_repairs_surrogates_after_parse() -> None:
    assert loads_json_ingress('{"value": "\\ud800"}') == {"value": "�"}


def test_tool_normalizers_preserve_custom_init_extension_types() -> None:
    class CustomResult(ToolResult):
        def __init__(self) -> None:
            super().__init__(ok=True, content={"text": "\ud800", "number": float("nan")})

    class CustomSpec(ToolSpec):
        def __init__(self) -> None:
            super().__init__(
                id="custom.\ud800",
                description="description\ud800",
                input_schema={"example": "\ud800"},
                capability="custom.read",
                side_effect="read",
                handler=lambda _context, _arguments: ToolResult(ok=True),
            )

    result = normalize_tool_result(CustomResult())
    spec = normalize_tool_spec(CustomSpec())

    assert type(result) is CustomResult
    assert result.content == {"text": "�", "number": None}
    assert type(spec) is CustomSpec
    assert spec.id == "custom.�"
    assert spec.description == "description�"
    assert spec.input_schema == {"example": "�"}


@contextlib.contextmanager
def _interpreter_int_digit_limit(limit: int) -> Iterator[None]:
    """Run under a different ``sys.set_int_max_str_digits``, restoring it afterwards.

    The setting is process-global and this suite runs in one process, so the restore is not
    tidiness: leaking a lowered limit would redden unrelated tests in whatever order they run.
    """
    previous = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(limit)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(previous)


def test_the_integer_bound_narrows_to_what_this_interpreter_can_actually_spell() -> None:
    """A host may lower the digit limit; the refusal has to follow it down.

    ``PYTHONINTMAXSTRDIGITS`` and ``sys.set_int_max_str_digits`` are deployment hardening knobs,
    and below 4300 they make the *process* unable to spell integers this predicate was still
    admitting -- so a 1500-digit value passed the refusing boundary and then raised ``ValueError``
    inside ``json.dumps`` at the transcript write, which is the exact run-death the boundary was
    added to end. The bound is therefore the smaller of the portable ceiling and what this
    interpreter can spell, read at call time because the limit is settable at runtime.

    The acceptance side is pinned against the writer rather than against arithmetic: whatever
    ingress admits, ``json.dumps`` must be able to spell.
    """
    with _interpreter_int_digit_limit(1000):
        with pytest.raises(UnportableScalarError):
            normalize_json_ingress({"n": 10**1500}, refuse_unportable=True)

        # 1000 digits: the largest this process can spell, and it must survive ingress *and* a dump
        admitted = normalize_json_ingress({"n": 10**999}, refuse_unportable=True)
        assert json.dumps(admitted)

        # 1001 digits: one past what the process can spell
        with pytest.raises(UnportableScalarError):
            normalize_json_ingress({"n": 10**1000}, refuse_unportable=True)


def test_a_permissive_interpreter_does_not_widen_the_portable_bound() -> None:
    """The other direction: 0 disables the interpreter's limit, and 4300 still stands.

    Portability is a claim about every reader, not about this process. A host that turns its own
    limit off can spell a 4301-digit integer, and the bounded decoders on the other side of the
    wire still refuse to read one, so admitting it would move the failure to someone else.
    """
    with _interpreter_int_digit_limit(0):
        assert json.dumps(10**4300), "this process can spell it"
        with pytest.raises(UnportableScalarError):
            normalize_json_ingress({"n": 10**4300}, refuse_unportable=True)


def test_a_json_integer_past_the_interpreters_limit_is_a_decode_error() -> None:
    """The parallel half, pinned because it is the one that was already right.

    ``parse_bounded_json_int`` converts inside a ``try``, so a literal this process cannot spell
    comes back as a classified ``JSONDecodeError`` rather than a bare ``ValueError``. Pinned so a
    later tightening of the digit check cannot quietly move the decode route into the crash class
    the Python-object route just left.
    """
    with _interpreter_int_digit_limit(1000):
        with pytest.raises(json.JSONDecodeError):
            loads_json_ingress('{"n": ' + "9" * 1500 + "}")


def test_an_integer_subclass_is_judged_by_the_value_a_writer_would_spell() -> None:
    """The bound is a question about the writer, so it must be asked of the base value.

    A Python-object ingress can carry an ``int`` subclass with its own ordering, and ``<`` hands
    the question to the object. Both answers it can give are wrong here: raising turns the
    classified refusal this boundary promises into an unclassified exception -- on a value
    (``5``) every writer in this process spells without complaint -- and understating itself
    declares a past-the-bound integer portable, which puts the ``ValueError`` back at the writer
    the boundary exists to keep it away from. ``int.__index__`` asks the base slot instead, which
    is what ``json.dumps`` will spell.
    """
    spellable = ExplodingComparisons(5)
    assert json.dumps(spellable) == "5", "the fixture must be a value writers handle"

    assert normalize_json_ingress({"n": spellable}, refuse_unportable=True) == {"n": 5}

    with pytest.raises(UnportableScalarError):
        normalize_json_ingress(
            {"n": UnderstatedInteger(10**5000)}, refuse_unportable=True
        )


def test_the_ingress_scans_the_base_text_without_rewriting_the_callers_value() -> None:
    """Two halves of one contract, and the second is a guard against my own first attempt at it.

    The *scan* reads base text: this runs on the ingress path, where a hostile ``__iter__`` is an
    unclassified exception at exactly the boundary that exists to keep runs alive.

    What it must **not** do is hand back an exact ``str``. That was the tidier-looking rule -- one
    pass at the boundary instead of a rule each guard remembers -- and this kernel carries ``str``
    subclasses through here on purpose: ``permissions._LegacyPathPattern`` marks a retained
    pre-v0.20 pattern, and normalizing it away made a replayed pre-v0.20 tool scope fail
    validation as "escaped leading ! is a configuration spelling". The subclass rule belongs at
    the guards, where the question is actually asked.
    """
    unrepaired = UnderstatedText("abc")
    normalized = normalize_json_ingress({MisreportingKey("content"): unrepaired})
    key, value = next(iter(normalized.items()))

    assert value is unrepaired, "the normalizer rewrote a value it had nothing to repair"
    assert key == "content" and value == "abc", "normalizing changed the text"

    # A marker subclass survives the round trip it is carried through in production.
    marker = _LegacyPathPattern("internal/**")
    assert normalize_unicode_scalars(marker) is marker

    # Repaired text is rebuilt, so it comes back exact -- and the scan never asks the value.
    assert normalize_unicode_scalars(UniterableText("plain")) == "plain"
    repaired = normalize_json_ingress({"k": UniterableText("a\ud800b")})["k"]
    assert type(repaired) is str and repaired == "a\ufffdb"

# --------------------------------------------------------------------------------------
# The refusing boundaries, read off the source rather than remembered
# --------------------------------------------------------------------------------------

_KERNEL_PACKAGE = Path(monoid_agent_kernel.__file__).resolve().parent

_REFUSING_INGRESS_BOUNDARIES = frozenset(
    {
        ("loop.py", "AgentToolContext.emit_artifact"),
        ("providers/base.py", "_normalize_model_turn"),
        ("tasks.py", "TaskManager.start_task"),
        ("tasks.py", "TaskManager.report_result"),
        ("tools/base.py", "normalize_tool_result"),
    }
)
"""The five Python-object ingress boundaries: the places a value reaches the normalizer without
ever having crossed a JSON parse, so ``bytes``, past-the-bound integers and shapes no writer can
carry all arrive alive.

``update_plan`` (``loop.py``) is a sixth such route and is deliberately absent: a plan is not a
``RunCheckpoint`` field, ``status.json``'s copy is taken from the emitted event rather than from
``self.plan``, and the preview already publishes a ``circular`` marker for the shape. Nothing there
reaches a writer a cycle or a depth kills, so refusing would convict a value that costs nothing.
Recorded here because an ``==`` that does not explain its absences reads as an oversight."""

_REFUSAL_CONVERSIONS = {
    ("loop.py", "AgentToolContext.emit_artifact"): ("UnportableValueError",),
    ("tasks.py", "TaskManager.start_task"): ("UnportableValueError",),
    ("tasks.py", "TaskManager.report_result"): ("UnportableValueError",),
    ("tools/base.py", "normalize_tool_result"): ("UnportableValueError",),
    # The fifth converts one frame up, in ``normalize_model_turn``, which turns anything escaping
    # ``_normalize_model_turn`` into a classified ``ModelAdapterError`` and stamps the usage the
    # refused turn was already billed for. The handler seen *here* is the settled-outcome arm.
    ("providers/base.py", "_normalize_model_turn"): ("Exception",),
}
"""What each boundary is prepared to catch. The flag alone is only half a boundary: a refusal that
nothing converts is an unclassified exception with extra steps, which is the crash this whole
mechanism exists to replace."""


def _enclosing_handler_types(
    call: ast.Call, parents: dict[ast.AST, ast.AST]
) -> tuple[str, ...]:
    """The exception types the nearest enclosing ``try`` around ``call`` is prepared to catch.

    Only a ``try`` whose *body* contains the call counts: a call sitting inside an ``except`` arm
    is not protected by that arm. The search stops at the owning function, because a handler in a
    caller is a different function's promise and this census is about this one's.
    """

    child: ast.AST = call
    cursor = parents.get(call)
    while cursor is not None:
        if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ()
        if isinstance(cursor, ast.Try) and any(node is child for node in cursor.body):
            names: list[str] = []
            for handler in cursor.handlers:
                caught = handler.type
                if caught is None:
                    names.append("BareExcept")
                    continue
                targets = caught.elts if isinstance(caught, ast.Tuple) else [caught]
                for target in targets:
                    if isinstance(target, ast.Name):
                        names.append(target.id)
                    elif isinstance(target, ast.Attribute):
                        names.append(target.attr)
            return tuple(sorted(set(names)))
        child = cursor
        cursor = parents.get(cursor)
    return ()


def _normalize_json_ingress_call_sites() -> list[tuple[str, str, ast.Call, tuple[str, ...]]]:
    """Every call to ``normalize_json_ingress`` in the kernel, with its owning function.

    Matches ``ast.Attribute`` as well as ``ast.Name`` so a refactor to module-qualified calls
    cannot leave this census silently green. Nested functions and lambdas attribute to their
    outermost enclosing function. Each site also carries what its nearest enclosing ``try`` is
    prepared to catch, so the table below can fix the *conversion* and not only the flag.
    """

    sites: list[tuple[str, str, ast.Call, tuple[str, ...]]] = []
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
            if callee != "normalize_json_ingress":
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
            sites.append(
                (
                    relative,
                    ".".join(named) or "<module>",
                    node,
                    _enclosing_handler_types(node, parents),
                )
            )
    return sites


def _refusing_sites() -> list[tuple[str, str, tuple[str, ...]]]:
    return [
        (relative, owner, handlers)
        for relative, owner, call, handlers in _normalize_json_ingress_call_sites()
        if any(
            keyword.arg == "refuse_unportable"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
    ]


def test_a_hostile_type_name_cannot_speak_through_the_sites_that_publish_it() -> None:
    """The census says the reader is called; this says what calling it is worth.

    Each of these raises a classified error whose message names the type it refused. Reading that
    name the plain way hands the question to the METACLASS, and then three things can happen, all
    of them worse than the error being reported: the read raises and REPLACES the classified error
    with a ``RuntimeError`` from inside the raise statement (``HiddenName``, ``ExplodingName``);
    it answers a portable type's name and the message describes the wrong thing
    (``ImpersonatingName``); or the name object itself lies when the f-string formats it
    (``RenamedByAHostileString``). A fourth is legal and needs no hostility at all -- a class name
    can be any length, and these messages are published.

    Three sites, chosen because they are three different kinds of caller: a kernel entry point
    that validates what an integrator handed it, a provider-facing normalizer that refuses a
    fragment shape, and the typed-output coercion, which is the one site that reads TWO names --
    the value's and the model's.

    The fragment site is asserted on its ``__cause__`` rather than its type, and the difference
    is the point. ``normalize_model_stream_chunk`` catches ``Exception``, so the classified
    ``ModelAdapterError`` came out either way -- what changed is what it is chained to. Reading
    the name the plain way made the cause ``RuntimeError("hostile metaclass ...")``, so an
    operator debugging a refused fragment learned about the metaclass and never about the
    fragment. The refusal has to keep its own explanation.

    A note for whoever sees this fail: pytest's traceback formatter calls ``saferepr`` on every
    frame argument, and its fallback path is ``type(obj).__name__`` -- the exact read this test is
    about. A failure inside any call below therefore ends in ``INTERNALERROR`` from
    ``_pytest/_io/saferepr.py`` rather than an assertion diff. That is not a second bug in this
    test; it is the same one, one layer out.
    """
    from monoid_agent_kernel.core.agents import coerce_runtime_config_provider
    from monoid_agent_kernel.core.result import _coerce_output
    from monoid_agent_kernel.providers.base import normalize_model_stream_chunk

    for hostile in HOSTILE_NAMED_TYPES:
        instance = hostile()
        expected = portable_type_name(instance)

        with pytest.raises(TypeError) as config_error:
            coerce_runtime_config_provider(instance)
        assert expected in str(config_error.value)

        with pytest.raises(ModelAdapterError) as fragment_error:
            normalize_model_stream_chunk(instance)
        cause = fragment_error.value.__cause__
        assert isinstance(cause, ValueError), cause
        assert expected in str(cause), cause

        with pytest.raises(TypeError) as output_error:
            _coerce_output(instance, int)
        assert expected in str(output_error.value)

        # ...and the other half of that site: the MODEL is a class too, and the hostility is on
        # the type being coerced TO rather than on the value.
        with pytest.raises(TypeError) as model_error:
            _coerce_output(123, hostile)
        assert portable_class_name(hostile) in str(model_error.value)

    # The length arm, which needs no hostility: a legal 10,000-character name is bounded to the
    # 64 escaped bytes the accountant charges in, so the message cannot outgrow the surface that
    # pays for it.
    huge = hugely_named_object(10_000)
    with pytest.raises(TypeError) as huge_error:
        coerce_runtime_config_provider(huge)
    assert len(str(huge_error.value)) < 200, len(str(huge_error.value))


_NAME_READERS = ("exact_text", "portable_class_name", "portable_type_name")

# Every site in the kernel that reads a ``__name__`` without one of the readers above around it,
# as ``{(module, the expression it is asked OF, spelling): how many}``. A COUNT and not a set:
# two of these share all three keys, so a set would collapse them and a fourth unwrapped read in
# that module would leave the census green.
#
# Three sites, and each is here for a reason that does not generalise:
#
#   * ``json_ingress.py`` is the rule's own implementation -- the base getset slot every other
#     site reaches THROUGH. A reader cannot read itself.
#   * ``decorator.py`` asks ``fn``, the integrator's own function object, and both reads stay
#     inside the process that owns it: one names a ``TypeError`` about their own signature, the
#     other builds a pydantic model class name. Neither is published, and neither crosses a
#     boundary where an adversarial object could have supplied the callable.
#
# That module's THIRD read -- ``tool_id``, which IS published, as ``ToolSpec.id`` -- is not here,
# because it is wrapped and the walk below does not count a wrapped read. Unwrapping it puts it
# back in this table and reddens the assertion.
_TYPE_NAMES_ASKED_DIRECTLY = {
    ("core/json_ingress.py", "type", "dict-slot"): 1,
    ("tools/decorator.py", "fn", "attribute"): 2,
}


def _type_name_read_sites(tree: ast.AST) -> list[tuple[str, str]]:
    """``(owner expression, spelling)`` for every ``__name__`` read in one module.

    Three spellings, because a rule proven on one of them is a rule proven on one of three: the
    plain attribute, the reflective ``getattr`` that no attribute matcher sees, and the base slot
    itself. ``if __name__ == "__main__"`` is an ``ast.Name`` and not an ``ast.Attribute``, so it
    is excluded STRUCTURALLY rather than by a string filter that could also drop a real site.

    A read that is already the first argument of a sanctioned reader does not count -- wrapping
    is exactly what this census asks for. Counting them instead would force them into the
    allowlist, and an allowlist entry cannot tell a wrapped read from an unwrapped one.

    The climb to that argument passes through ``or`` and ``if/else`` and NOTHING ELSE, because
    those two hand back one of their operands unchanged: ``exact_text(id or fn.__name__)`` really
    does read whichever it picked. An f-string would not qualify even though it looks similar --
    ``f"{name}"`` calls the name object's own ``__format__`` before the reader ever sees it, so
    the lying value has already spoken. Widening this climb is how a census stops seeing.
    """

    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def wrapped(node: ast.AST) -> bool:
        cursor: ast.AST = node
        while True:
            parent = parents.get(cursor)
            if isinstance(parent, ast.BoolOp) and cursor in parent.values:
                cursor = parent
                continue
            if isinstance(parent, ast.IfExp) and cursor in (parent.body, parent.orelse):
                cursor = parent
                continue
            break
        parent = parents.get(cursor)
        if not isinstance(parent, ast.Call) or not parent.args or parent.args[0] is not cursor:
            return False
        func = parent.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        return name in _NAME_READERS

    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "__name__":
            if not wrapped(node):
                found.append((ast.unparse(node.value), "attribute"))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "__name__"
        ):
            found.append((ast.unparse(node.args[0]), "getattr"))
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "__dict__"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "__name__"
        ):
            found.append((ast.unparse(node.value.value), "dict-slot"))
    return found


def test_the_type_name_walk_sees_each_spelling_it_claims_to() -> None:
    """The matcher first, on a source holding all three -- and on the wrapper that hides one.

    A census whose matcher is broken finds nothing and passes. This one is checked against a
    module it does not read from disk, so it cannot be satisfied by the repository happening to
    be clean, and both arms of the wrapper climb are checked: a read reached through ``or``
    DISAPPEARS, and a read the reader only sees after an f-string already formatted it does not.
    """

    source = (
        "raw = type(value).__name__\n"
        'reflective = getattr(value, "__name__", "")\n'
        'slot = type.__dict__["__name__"]\n'
        "wrapped = portable_class_name(type(value).__name__)\n"
        "through_or = exact_text(given or fn.__name__)\n"
        'through_fstring = exact_text(f"{spoken.__name__}")\n'
        "if __name__ == '__main__':\n    pass\n"
    )

    found = _type_name_read_sites(ast.parse(source))

    assert sorted(found) == [
        ("spoken", "attribute"),
        ("type", "dict-slot"),
        ("type(value)", "attribute"),
        ("value", "getattr"),
    ], found


def test_a_type_name_is_asked_for_directly_only_where_it_may_be() -> None:
    """Every published type name goes through the reader, and the exceptions are named.

    ``cls.__name__`` is an attribute read on a *class*, so it dispatches to the METACLASS, and an
    in-process tool or task hands back objects whose type is whatever built them. The reader takes
    three answers away -- the metaclass, the name object itself, and the length -- and a site that
    skips it has none of them. These names reach observers, ``run.failed.error``, ``failure.json``,
    ``status.json``, the CLI's JSON output and an exit-code predicate, so "it is only an error
    message" is not a reason to leave one unread.

    The ``==`` fails in both directions and in a third: a new unwrapped read has to be justified
    here in the same edit that adds it, a site that disappears means a reader was removed and the
    table now describes code that is gone, and a SECOND read added beside an allowlisted one moves
    its count.
    """

    modules = sorted(_KERNEL_PACKAGE.rglob("*.py"))
    parsed = 0
    asked: dict[tuple[str, str, str], int] = {}
    for path in modules:
        relative = path.relative_to(_KERNEL_PACKAGE).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parsed += 1
        for expression, spelling in _type_name_read_sites(tree):
            key = (relative, expression, spelling)
            asked[key] = asked.get(key, 0) + 1

    # A walk that silently skipped modules would pass by reading less, so the arithmetic is
    # asserted rather than assumed -- the guard the older censuses in this file do not carry.
    assert parsed == len(modules) and parsed > 100, (parsed, len(modules))

    assert asked == _TYPE_NAMES_ASKED_DIRECTLY


def test_the_refusing_ingress_boundaries_are_exactly_the_python_object_routes() -> None:
    """``refuse_unportable=True`` appears at the five boundaries and nowhere else.

    Every other caller keeps the default. For scalars the reason is that they hand the normalizer
    values that came off a bounded JSON parse, so the decoders already refused what these
    boundaries refuse. The shape cap also remains an ingress contract after the v0.22 checkpoint
    writer became iterative: widening it would change what model/tool routes accept, and recursive
    JSON writers around those routes still need the margin. The route a model's own tool-call
    arguments take is therefore one of these boundaries.

    The ``==`` fails in both directions. A new ``True`` site means a new Python-object ingress was
    opened, and it belongs here only together with the conversion the others carry. A missing
    entry means a boundary stopped refusing, and the run-death this closed — ``json.dumps``
    raising at the transcript write, ``run.failed`` with no observation — comes back.

    Only a literal ``True`` counts: a site that grows a variable there drops out of this set and
    must be re-classified by whoever made the flag conditional.
    """

    sites = _normalize_json_ingress_call_sites()
    owners = {(relative, owner) for relative, owner, _call, _handlers in sites}

    assert ("core/json_ingress.py", "loads_json_ingress") in owners, (
        "census self-check: the normalizer's own module no longer shows its known caller, "
        "so this walk is not seeing what it claims to see"
    )

    refusing = {(relative, owner) for relative, owner, _handlers in _refusing_sites()}

    assert refusing == _REFUSING_INGRESS_BOUNDARIES


def test_every_refusing_boundary_converts_what_it_refuses() -> None:
    """The flag is half a boundary; the conversion is the other half.

    A refusal nothing catches is an unclassified exception with extra steps — the exact failure
    the scalar refusal was written to replace, re-earned one level up. So the handler types are
    read off the source too, and fixed by a table: a boundary that refuses without converting, or
    that narrows its ``except`` to one of the two ``UnportableValueError`` subclasses, fails here
    rather than in production.
    """

    conversions = {
        (relative, owner): handlers for relative, owner, handlers in _refusing_sites()
    }

    assert any("UnportableValueError" in handlers for handlers in conversions.values()), (
        "census self-check: no boundary shows the base exception in its handler, so the handler "
        "walk is not seeing what it claims to see"
    )
    assert conversions == _REFUSAL_CONVERSIONS
