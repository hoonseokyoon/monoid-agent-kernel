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
    text = (run_dir / "transcript.jsonl").read_text(encoding="utf-8")
    return [
        record
        for record in (json.loads(line) for line in text.splitlines() if line.strip())
        if record.get("kind") == kind
    ]


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
