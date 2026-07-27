"""The run-dir record that carries model-authored settled text.

``transcript.jsonl`` gains a ``settled_text`` variant keyed by content digest. This commit writes
it but nothing reads it — the settle events still publish ``final_text`` — so the assertions here
are about the record's shape, its selectivity, and the fact that the events are *unchanged*.

Selectivity is the part worth testing from both sides. "Model-authored text is recorded" and
"kernel-authored text is not" are separate claims, and an implementation that records everything
satisfies the first while quietly digesting the limit message an operator needs to read.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core.model_io import content_digest
from monoid_agent_kernel.core.schemas import validate_run_dir
from monoid_agent_kernel.core.spec import AgentRunSpec, RunLimits
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call


def _spec(tmp_path: Path, **limits: Any) -> AgentRunSpec:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        limits=RunLimits(**limits) if limits else RunLimits(),
    )


def _run(spec: AgentRunSpec, adapter: Any, *tool_ids: str) -> Path:
    loop = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config(*(tool_ids or ("run.finish",)))),
    )
    result = loop.run_once("go")
    return result.run_dir


def _records(run_dir: Path, kind: str) -> list[dict[str, Any]]:
    text = (run_dir / "transcript.jsonl").read_text(encoding="utf-8", errors="replace")
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            # A torn line is a legitimate transcript state (the repair confines a tear, it does
            # not recover the torn record), and the
            # reader under test skips them — so this helper must too, or it fails on the very
            # input the torn-tail case exists to exercise.
            continue
        if isinstance(record, dict) and record.get("kind") == kind:
            records.append(record)
    return records


def _events(run_dir: Path, event_type: str) -> list[dict[str, Any]]:
    text = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    return [
        event
        for event in (json.loads(line) for line in text.splitlines() if line.strip())
        if event.get("type") == event_type
    ]


def test_model_authored_settled_text_is_recorded(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="the model wrote this")])

    run_dir = _run(_spec(tmp_path), adapter)

    records = _records(run_dir, "settled_text")
    assert len(records) == 1
    assert records[0]["final_text"] == "the model wrote this"
    assert records[0]["final_text_len"] == len("the model wrote this")


def test_kernel_authored_settled_text_is_not_recorded(tmp_path: Path) -> None:
    """The other half of the rule: a limit message stays inline and gets no record.

    An implementation that records unconditionally passes the test above and fails only here.
    """
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id=f"r{index}",
                tool_calls=(fake_tool_call("fs_list", {"path": "."}, f"c{index}"),),
            )
            for index in range(4)
        ]
    )

    run_dir = _run(_spec(tmp_path, max_steps=1), adapter, "fs.list", "run.finish")

    assert _records(run_dir, "settled_text") == []
    settled = _events(run_dir, "run.finished")
    assert settled[-1]["data"]["final_text"] == "Stopped after reaching max steps."


def test_the_digest_is_content_digest_not_a_bare_sha256(tmp_path: Path) -> None:
    """Pins the one decision this record freezes.

    ``content_digest`` hashes canonical JSON under a shape key, so a text field cannot collide
    with a structured value's serialization. A reader that recomputes with ``sha256sum`` would
    miss every join, and once records carry a digest the choice cannot be revised.
    """
    text = "the model wrote this"
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text=text)])

    run_dir = _run(_spec(tmp_path), adapter)

    digest = _records(run_dir, "settled_text")[0]["final_text_digest"]
    assert digest == content_digest(text)
    assert digest != hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_identical_text_across_both_settle_events_yields_one_record(tmp_path: Path) -> None:
    # turn.settled and run.finished carry the same value, and the key is the content, so the
    # second write is redundant by construction rather than by a caller remembering.
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="one answer")])

    run_dir = _run(_spec(tmp_path), adapter)

    assert len(_events(run_dir, "turn.settled")) == 1
    assert len(_events(run_dir, "run.finished")) == 1
    assert len(_records(run_dir, "settled_text")) == 1


def test_distinct_text_across_turns_yields_a_record_each(tmp_path: Path) -> None:
    # The counterweight to de-duplication: distinct answers must not collapse into one record.
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", final_text="first answer"),
            ModelTurn(response_id="r2", final_text="second answer"),
        ]
    )
    loop = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
    )
    loop.open()
    loop.submit("first")
    loop.submit("second")
    result = loop.close()

    recorded = {record["final_text"] for record in _records(result.run_dir, "settled_text")}
    assert recorded == {"first answer", "second answer"}


def test_the_record_precedes_the_event_that_names_it(tmp_path: Path) -> None:
    """Ordering, not just presence.

    Writing after the emit would leave a window in which a committed event names text that is not
    on disk. Both files are append-only, so "the record was written first" is observable as the
    record existing by the time the event is committed — asserted here by reading the transcript
    through an event sink that fires during the emit itself.
    """
    seen: list[list[dict[str, Any]]] = []
    run_dir_holder: dict[str, Path] = {}

    class _Spy:
        def emit(self, event: Any) -> None:
            if event.type == "turn.settled" and "run_dir" in run_dir_holder:
                seen.append(_records(run_dir_holder["run_dir"], "settled_text"))

        def close(self) -> None:
            return None

    spec = _spec(tmp_path)
    run_dir_holder["run_dir"] = spec.run_root / spec.run_id
    loop = AgentLoop(
        spec=spec,
        model_adapter=FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="ordered")]),
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
        event_sinks=(_Spy(),),
    )
    loop.run_once("go")

    assert seen, "the turn.settled sink never fired"
    assert [record["final_text"] for record in seen[-1]] == ["ordered"]


def test_a_run_with_the_record_still_validates(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="valid run")])

    run_dir = _run(_spec(tmp_path), adapter)

    assert _records(run_dir, "settled_text")  # the record is actually present
    assert validate_run_dir(run_dir) == []


def test_a_run_without_the_record_still_validates(tmp_path: Path) -> None:
    # Older run dirs, and every kernel-settled run, have no settled_text record at all.
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id=f"r{index}",
                tool_calls=(fake_tool_call("fs_list", {"path": "."}, f"c{index}"),),
            )
            for index in range(4)
        ]
    )

    run_dir = _run(_spec(tmp_path, max_steps=1), adapter, "fs.list", "run.finish")

    assert _records(run_dir, "settled_text") == []
    assert validate_run_dir(run_dir) == []


def test_a_malformed_record_is_reported(tmp_path: Path) -> None:
    # The counterweight to "absence is fine": a present-but-wrong record must not pass silently.
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="valid run")])
    run_dir = _run(_spec(tmp_path), adapter)

    with (run_dir / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "settled_text", "final_text": "no digest"}) + "\n")

    # ``path`` carries the offending line ("transcript.jsonl:7"), not just the file name.
    issues = validate_run_dir(run_dir)
    assert any(issue.path.startswith("transcript.jsonl:") for issue in issues), issues


@pytest.mark.parametrize(
    "summary",
    [
        pytest.param("SENTINEL-a short final answer", id="short"),
        # Long values took a different route: the generic preview truncates to a 160-character
        # prefix rather than dropping the field, so it leaked a readable excerpt instead of the
        # whole answer. Both shapes have to be covered.
        pytest.param("SENTINEL-" + "x" * 400, id="long-enough-to-be-truncated"),
    ],
)
def test_the_run_finish_summary_never_reaches_the_tool_call_event(
    tmp_path: Path, summary: str
) -> None:
    """Settling through `run.finish` is the default flow, so its `summary` IS the final answer.

    It reached `events.jsonl` through `tool.call.started.data.args_preview` as well as the settle
    events — a second door that removing text from `turn.settled` alone would have left open.
    """
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call(
                        "run_finish", {"summary": summary, "outputs": ["notes.md"]}, "c1"
                    ),
                ),
            )
        ]
    )

    run_dir = _run(_spec(tmp_path), adapter)

    # ``tool`` carries the wire call name (``run_finish``), not the spec id (``run.finish``).
    started = [
        event for event in _events(run_dir, "tool.call.started") if event["data"]["tool"] == "run_finish"
    ]
    assert started, "the run.finish tool call never started"
    preview = started[-1]["data"]["args_preview"]
    assert preview["summary"] == {"redacted": True, "type": "str", "bytes": len(summary)}
    # Not the whole file: `outputs` is a path list, not prose, and must survive. A preview that
    # redacted everything would satisfy the assertion above and tell an operator nothing.
    assert preview["outputs"] == ["notes.md"]
    # And the sentinel is nowhere in the raw event line, not merely absent from the field checked.
    raw = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    started_lines = [line for line in raw.splitlines() if '"tool.call.started"' in line]
    assert started_lines and not any(summary[:30] in line for line in started_lines)


def _finish_spec() -> Any:
    """A spec carrying the real `run.finish` preview kind.

    Synthetic rather than the builtin, which needs a live workspace to construct. That the actual
    `run.finish` binding carries `preview_kind="finish"` is covered by
    `test_the_run_finish_summary_never_reaches_the_tool_call_event`, which drives a real run and
    only passes if the wiring is in place.
    """
    from monoid_agent_kernel.tools.base import ToolSpec

    return ToolSpec(
        id="run.finish",
        description="Finish the run.",
        input_schema={"type": "object"},
        capability="run.control",
        side_effect="run",
        handler=lambda *_args, **_kwargs: None,
        preview_kind="finish",
    )


def test_the_approval_request_redacts_the_summary_too() -> None:
    """The approval path is a second public route for the same argument.

    `build_tool_approval_task_request` stores `arguments_preview`, and the hosted task republishes
    it on `task.started`. Redacting only in `_tool_start_data` left this half unbound, so binding
    `run.finish` to `authorization="ask"` put the final answer straight back on `events.jsonl`.
    """
    from monoid_agent_kernel.core.tool_approval import build_tool_approval_task_request

    request = build_tool_approval_task_request(
        spec=_finish_spec(),
        binding_id="run.finish",
        model_name="run_finish",
        call_name="run_finish",
        call_id="c1",
        arguments={"summary": "SENTINEL-the-answer", "notes": "SENTINEL-notes", "outputs": ["n.md"]},
        reason="ask",
        turn_id="turn_0001",
        tool_event_id=None,
    )

    preview = request["arguments_preview"]
    assert "SENTINEL-the-answer" not in json.dumps(preview)
    assert "SENTINEL-notes" not in json.dumps(preview)
    # Counterweight: non-prose arguments must survive, or an approver sees nothing to judge.
    assert preview["outputs"] == ["n.md"]
    # And the private `arguments` still carry the real values — the approval decision is not
    # deprived of them; only the public preview is.
    assert request["arguments"]["summary"] == "SENTINEL-the-answer"


def test_a_tool_without_the_finish_preview_kind_is_unaffected() -> None:
    # The redaction is keyed on preview kind, not on argument names, so an ordinary tool whose
    # arguments happen to include `summary` is not silently gutted.
    from monoid_agent_kernel.core.tool_approval import redact_tool_arguments

    assert redact_tool_arguments({"summary": "plain"})["summary"] == "plain"


def test_finish_preview_leaves_an_absent_notes_alone() -> None:
    # `notes` is declared ["string", "null"]. A redaction marker on a value that was never there
    # tells an operator something was withheld when nothing was.
    from monoid_agent_kernel.permissions import PermissionPolicy
    from monoid_agent_kernel.public_view import finish_args_preview

    preview = finish_args_preview({"summary": "s", "notes": None}, PermissionPolicy())
    assert preview["notes"] is None


def test_a_refusal_writes_no_settled_text_record(tmp_path: Path) -> None:
    """Empty model-authored text is skipped.

    A digest of "" on the event, with no way to distinguish a lost record from a genuinely empty
    answer, is worse than leaving the field alone.
    """
    adapter = FakeModelAdapter(
        turns=[ModelTurn(response_id="r1", final_text=None, stop_reason="refusal")]
    )

    run_dir = _run(_spec(tmp_path), adapter)

    assert _records(run_dir, "settled_text") == []


def test_a_torn_transcript_tail_does_not_consume_the_next_record(tmp_path: Path) -> None:
    """A remnant without its trailing newline would glue the next record onto it, losing BOTH.

    On the recovery path the first write of the reopened recorder can be the settled-text record
    itself, which a committed `run.finished` then names — the exact "event names text that is not
    on disk" failure the write-before-emit ordering exists to prevent, with no crash in the
    recovered run.
    """
    from monoid_agent_kernel.recorder import AgentRecorder

    run_root = tmp_path / "runs"
    run_dir = run_root / "run-torn"
    run_dir.mkdir(parents=True)
    with (run_dir / "transcript.jsonl").open("wb") as handle:
        handle.write(b'{"kind":"model_turn","step":1,"final_text":"complete"}\n')
        handle.write(b'{"kind":"model_turn","step":2,"final_te')  # torn mid-write, no newline

    recorder = AgentRecorder(run_root=run_root, run_id="run-torn", reopen=True, status_file=False)
    try:
        digest = recorder.settled_text("survives the tear")
    finally:
        recorder.close()

    records = _records(run_dir, "settled_text")
    assert [record["final_text_digest"] for record in records] == [digest]
    # The torn remnant is still one broken line — confined to the record it tore, not spread.
    lines = (run_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    broken = [line for line in lines if line.strip() and not line.strip().endswith("}")]
    assert len(broken) == 1


def test_a_restored_run_records_its_settled_text_from_finalize(tmp_path: Path) -> None:
    """`finalize` is the SOLE writer on the restore path, and nothing else pinned it.

    Every other test reaches `finalize` only after `checkpoint_on_settle` already wrote the same
    content-keyed record, so the second call site is invisible — replacing its guard with
    `if False:` left the whole suite green. A restored run emits `run.finished` with no
    `turn.settled` at all, so deleting that call would publish a digest naming text that is not on
    disk for every resumed run.
    """
    from monoid_agent_kernel.core.checkpoint import RunCheckpoint

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")
    loop = AgentLoop(
        spec=spec,
        model_adapter=FakeModelAdapter(turns=[ModelTurn(response_id="r2", final_text="unused")]),
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
    )
    loop.restore(RunCheckpoint(run_id=spec.run_id, final_text="restored model answer"))
    result = loop.close()

    assert _events(result.run_dir, "turn.settled") == []  # the sole-writer precondition
    records = _records(result.run_dir, "settled_text")
    assert [record["final_text"] for record in records] == ["restored model answer"]


def test_a_failed_write_does_not_mark_the_digest_as_recorded(tmp_path: Path) -> None:
    """The de-dup set is marked only once the write succeeds.

    Marked first, a raising write (a full disk mid-flush) recorded the digest as present with
    nothing on disk, and a later call for the same text short-circuited and returned a digest that
    resolves to nothing — the failure mode the whole write-before-emit ordering exists to avoid.
    """
    from monoid_agent_kernel.recorder import AgentRecorder

    run_root = tmp_path / "runs"
    (run_root / "run-fail").mkdir(parents=True)
    recorder = AgentRecorder(run_root=run_root, run_id="run-fail", status_file=False)
    try:
        original_write = recorder._transcript_file.write

        def failing_write(_payload: str) -> int:
            raise OSError("no space left on device")

        recorder._transcript_file.write = failing_write  # type: ignore[method-assign]
        with pytest.raises(OSError):
            recorder.settled_text("never landed")
        recorder._transcript_file.write = original_write  # type: ignore[method-assign]

        # The retry must actually write, not short-circuit on a digest it never recorded.
        recorder.settled_text("never landed")
    finally:
        recorder.close()

    assert [record["final_text"] for record in _records(run_root / "run-fail", "settled_text")] == [
        "never landed"
    ]


def test_a_transcript_with_undecodable_bytes_is_reported(tmp_path: Path) -> None:
    """A validator that turns detected corruption into silence is worse than one that crashes.

    Strict whole-file decoding crashed `monoid validate` on a torn transcript, but repairing that
    with `errors="replace"` made a *complete* record holding an undecodable byte parse, validate,
    and the file report clean. The twin `_validate_event_file` detects and reports; so must this.
    """
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="valid run")])
    run_dir = _run(_spec(tmp_path), adapter)

    with (run_dir / "transcript.jsonl").open("ab") as handle:
        handle.write(
            b'{"kind":"model_turn","step":9,"response_id":"r","final_text":"ab\xffcd",'
            b'"tool_calls":[],"usage":{}}\n'
        )

    issues = validate_run_dir(run_dir)
    assert any(
        issue.path.startswith("transcript.jsonl:") and "UTF-8" in issue.message for issue in issues
    ), issues


@pytest.mark.parametrize("artifact", ["events.jsonl", "transcript.jsonl"])
def test_a_deeply_nested_line_is_reported_not_raised(tmp_path: Path, artifact: str) -> None:
    """Both validator halves, because hardening one of them is how this keeps going wrong.

    `RecursionError` is not a `ValueError`, so a deeply nested line escaped the catch entirely and
    crashed `monoid validate` on the corruption it exists to report. The transcript half was
    hardened first and the event half left — and since `events.jsonl` is validated *first*, that
    left the newly-hardened branch unreachable on a run dir corrupted in both.
    """
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="valid run")])
    run_dir = _run(_spec(tmp_path), adapter)

    depth = sys.getrecursionlimit() * 3
    nested = ("[" * depth) + ("]" * depth)
    with (run_dir / artifact).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(nested + "\n")

    issues = validate_run_dir(run_dir)  # must not raise
    assert any(issue.path.startswith(f"{artifact}:") for issue in issues), issues


@pytest.mark.parametrize(
    ("corruption", "expected"),
    [
        pytest.param(b'{"a": "b\xffc"}', "invalid UTF-8", id="undecodable-bytes"),
        pytest.param(None, "decoder limit exceeded", id="deeply-nested"),
        pytest.param(b"1" * 5000, "decoder limit exceeded", id="oversized-int"),
    ],
)
def test_a_corrupt_json_artifact_is_reported_not_raised(
    tmp_path: Path, corruption: bytes | None, expected: str
) -> None:
    """The THIRD validator sibling, which guards ten artifacts and runs before both JSONL halves.

    Hardening `_validate_jsonl_file` and `_validate_event_file` while leaving this one made both
    of those unreachable on a run dir whose corruption is in a JSON artifact — the same
    ordering argument that justified hardening the event half in the first place.
    """
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="valid run")])
    run_dir = _run(_spec(tmp_path), adapter)

    if corruption is None:
        depth = sys.getrecursionlimit() * 3
        corruption = (b"[" * depth) + (b"]" * depth)
    (run_dir / "metrics.json").write_bytes(corruption)

    issues = validate_run_dir(run_dir)  # must not raise
    assert any(
        issue.path == "metrics.json" and expected in issue.message for issue in issues
    ), issues


def test_the_approval_preview_leaves_an_absent_notes_alone() -> None:
    # The other half of the `notes: None` rule. Fixing it in `finish_args_preview` alone moved the
    # two halves in opposite directions inside one commit — the approval preview then badged an
    # absent value as withheld, which is exactly what the other half stopped doing.
    from monoid_agent_kernel.core.tool_approval import build_tool_approval_task_request

    request = build_tool_approval_task_request(
        spec=_finish_spec(),
        binding_id="run.finish",
        model_name="run_finish",
        call_name="run_finish",
        call_id="c1",
        arguments={"summary": "an answer", "notes": None},
        reason="ask",
        turn_id="turn_0001",
        tool_event_id=None,
    )

    assert request["arguments_preview"]["notes"] is None


def test_this_commit_does_not_change_the_settle_events(tmp_path: Path) -> None:
    """The no-op property. Hydration and the emit change land later; until then the events are
    byte-identical to what they were, so a Studio regression here cannot be blamed on the record.
    """
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="unchanged")])

    run_dir = _run(_spec(tmp_path), adapter)

    for event_type in ("turn.settled", "run.finished"):
        data = _events(run_dir, event_type)[-1]["data"]
        assert data["final_text"] == "unchanged", event_type
        assert "final_text_digest" not in data, event_type
        assert "final_text_len" not in data, event_type
