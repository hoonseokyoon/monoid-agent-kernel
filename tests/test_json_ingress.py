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
)

import monoid_agent_kernel
from monoid_agent_kernel.core.json_ingress import (
    UnportableContainerError,
    UnportableScalarError,
    is_finite_json_number,
    loads_json_ingress,
    loads_model_envelope_json_ingress,
    loads_model_json_ingress,
    normalize_json_ingress,
    normalize_unicode_scalars,
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


def test_a_cyclic_model_turn_is_a_classified_adapter_failure() -> None:
    """The fifth Python-object route: an adapter's own ``ToolCall.arguments``.

    That is the route to the checkpoint -- the arguments ride the assistant message into
    ``state.messages`` and out through ``RunCheckpoint.to_json`` -- and it was the one boundary of
    the five that did not refuse anything. Driven with a tool-calls-only turn on purpose: a
    settled outcome beside the calls makes ``_normalize_model_turn`` swallow the failure, so a pin
    written on that shape would pass for the wrong reason.
    """
    cyclic: dict = {}
    cyclic["self"] = cyclic
    turn = ModelTurn(
        response_id="r1",
        tool_calls=(ToolCall(id="c1", name="custom_tool", arguments=cyclic),),
    )

    with pytest.raises(ModelAdapterError, match="non-portable"):
        normalize_model_turn(turn)


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


def test_the_refusing_ingress_boundaries_are_exactly_the_python_object_routes() -> None:
    """``refuse_unportable=True`` appears at the five boundaries and nowhere else.

    Every other caller keeps the default. For scalars the reason is that they hand the normalizer
    values that came off a bounded JSON parse, so the decoders already refused what these
    boundaries refuse. That reason does **not** extend to shape: the model-JSON decoders admit 512
    levels of nesting and ``dataclasses.asdict`` dies at 492, so the depth bound had to be carried
    by these boundaries rather than inherited from the parsers — which is why the route a model's
    own tool-call arguments take is now one of them.

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
