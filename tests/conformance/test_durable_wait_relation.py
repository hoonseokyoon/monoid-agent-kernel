"""A restart test must wait on a DURABLE fact before it reads one.

The flake this polices has one shape and it recurs because nothing checks for it. A backend test
waits for ``backend1._record(run_id)`` to reach some state -- an in-memory fact, set by the loop --
and then constructs a second backend over the same ``run_root`` and asserts on what that one reads
from disk. The status artifact is written after the in-memory transition, so the assertion races
the writer: it passes whenever the writer wins. When it loses, the failure lands in the assertion
about the SECOND backend, which points away from the wait that was missing.

Naming the offending tests in a list would police the nine that exist today and nothing written
tomorrow, so the rule is a RELATION over the source instead, and the violation set is asserted
empty. Three matchers make it checkable, and the file's own self-check proves each of them sees
what it claims to.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]

# A wait, by whatever name. ``wait_for_durable_status`` is a waiter AND is durable by
# construction; the other two are durable only when their predicate names something on disk.
WAITERS = frozenset({"eventually", "wait_until", "wait_for_durable_status"})
DURABLE_WAITERS = frozenset({"wait_for_durable_status"})
DURABLE_MARKERS = ("status.json", "failure.json", "exists", "checkpoint_store", "read_text", "is_file")
# The in-memory record, which is what the loop updates first and the disk learns about later.
IN_MEMORY_MARKERS = ("_record", "_records")


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _restart_shape(func: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[int, int, list[int]] | None:
    """``(in-memory wait, second construction, durable waits)`` for one test, or ``None``.

    A "second construction" is any call HANDED the run root -- by the parameter name, whatever
    expression the argument is, or positionally as ``run_root``. Deliberately not a list of
    builder names: that list is the thing this census exists to stop maintaining, and it is how
    the first draft of this walk missed ``_provider_backend`` and ``RunnerBackend`` both.
    """

    memory_wait: int | None = None
    durable_waits: list[int] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Call) or _call_name(node) not in WAITERS:
            continue
        rendered = ast.unparse(node)
        if _call_name(node) in DURABLE_WAITERS or any(m in rendered for m in DURABLE_MARKERS):
            durable_waits.append(node.lineno)
        elif any(m in rendered for m in IN_MEMORY_MARKERS):
            memory_wait = node.lineno if memory_wait is None else memory_wait
    if memory_wait is None:
        return None

    rebuilds = [
        node.lineno
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and _call_name(node) not in WAITERS
        and node.lineno > memory_wait
        and (
            any(keyword.arg == "run_root" for keyword in node.keywords)
            or any(isinstance(arg, ast.Name) and arg.id == "run_root" for arg in node.args)
        )
    ]
    if not rebuilds:
        return None
    return memory_wait, min(rebuilds), durable_waits


def _violations(tree: ast.AST) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not func.name.startswith("test_"):
            continue
        shape = _restart_shape(func)
        if shape is None:
            continue
        memory_wait, rebuild, durable_waits = shape
        # The durable wait has to fall BETWEEN. One that runs before the in-memory wait proves
        # nothing about the state the test went on to observe -- an earlier checkpoint of the
        # same run satisfies it, which is exactly what three of these tests were doing.
        if not any(memory_wait < line < rebuild for line in durable_waits):
            found.append((func.name, memory_wait))
    return found


def test_the_restart_relation_sees_both_arms_it_claims_to() -> None:
    """The matchers first, on sources this walk does not read from disk.

    A relation whose matchers are broken finds no violations and passes. Both arms are checked:
    the offending shape is FOUND, and the same shape with the wait added is not -- and so is the
    ordering, because a durable wait on the wrong side of the in-memory one is the bug three of
    these tests actually had.
    """

    offending = (
        "def test_x():\n"
        "    assert eventually(lambda: backend1._record(run_id).terminal)\n"
        "    backend2 = build(run_root, token)\n"
    )
    repaired = (
        "def test_x():\n"
        "    assert eventually(lambda: backend1._record(run_id).terminal)\n"
        "    wait_for_durable_status(run_root, run_id)\n"
        "    backend2 = build(run_root, token)\n"
    )
    out_of_order = (
        "def test_x():\n"
        "    wait_for_durable_status(run_root, run_id)\n"
        "    assert eventually(lambda: backend1._record(run_id).terminal)\n"
        "    backend2 = build(run_root, token)\n"
    )
    keyword_built = (
        "def test_x():\n"
        "    assert eventually(lambda: backend1._record(run_id).terminal)\n"
        '    backend2 = RunnerBackend(run_root=tmp_path / "runs")\n'
    )

    assert _violations(ast.parse(offending)) == [("test_x", 2)]
    assert _violations(ast.parse(repaired)) == []
    assert _violations(ast.parse(out_of_order)) == [("test_x", 3)]
    assert _violations(ast.parse(keyword_built)) == [("test_x", 2)]


def test_no_restart_test_reads_a_durable_fact_it_did_not_wait_for() -> None:
    """The relation over every test module, with the violation set asserted empty.

    Honest about what this buys: the wait does not make a slow writer fast. If the artifact is
    never written -- the swallowed ``OSError`` in the recorder's status projection is the known
    candidate -- these tests now fail deterministically at the wait, in the test that is missing
    the fact, instead of intermittently at an assertion three constructions later. Turning a flake
    into a legible failure is the whole claim; making the write reliable is a source-side change
    and is not in this commit.
    """

    modules = sorted(TESTS_ROOT.rglob("test_*.py"))
    parsed = 0
    offenders: list[tuple[str, str, int]] = []
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parsed += 1
        for name, lineno in _violations(tree):
            offenders.append((path.relative_to(TESTS_ROOT).as_posix(), name, lineno))

    # A walk that skipped modules would pass by reading less.
    assert parsed == len(modules) and parsed > 50, (parsed, len(modules))
    assert offenders == []
