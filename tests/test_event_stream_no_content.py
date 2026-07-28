"""The fan-out event stream carries no model-authored settled text.

``docs/OBSERVABILITY.md`` documents ``events.jsonl`` as the public/redacted artifact, and
``EventBus.emit`` fans every event out to every registered sink with no level filtering — so what
lands in this file is what a redacting sink, an OTel exporter and ``monoid watch --json`` all see.

The pair of claims has to be tested from both sides. "Model text is gone" and "kernel text is still
there" are separate properties, and an implementation that digests everything satisfies the first
while destroying the one line an operator reads to find out why a run stopped. The kernel-inline
branch is the half nothing else covers: the run-limit path never produced a settle event in any
existing test, so it could have regressed silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core.model_io import content_digest
from monoid_agent_kernel.core.spec import AgentRunSpec, RunLimits
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call

SETTLE_TYPES = ("turn.settled", "run.finished")
MODEL_PROSE = "The answer is 42, and here is the reasoning behind it."


def _spec(tmp_path: Path, **limits: Any) -> AgentRunSpec:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        limits=RunLimits(**limits) if limits else RunLimits(),
    )


def _events(run_dir: Path, event_type: str) -> list[dict[str, Any]]:
    text = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [record for record in records if record["type"] == event_type]


def _run(spec: AgentRunSpec, adapter: Any, *tool_ids: str) -> Any:
    loop = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config(*(tool_ids or ("run.finish",)))),
    )
    return loop.run_once("go")


def test_model_authored_text_leaves_the_stream_as_a_digest(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text=MODEL_PROSE)])

    result = _run(_spec(tmp_path), adapter)

    for event_type in SETTLE_TYPES:
        data = _events(result.run_dir, event_type)[-1]["data"]
        assert "final_text" not in data, event_type
        assert data["final_text_digest"] == content_digest(MODEL_PROSE), event_type
        assert data["final_text_len"] == len(MODEL_PROSE), event_type


def test_the_prose_is_absent_from_the_whole_committed_file(tmp_path: Path) -> None:
    """Not just from the settle events — from every byte of ``events.jsonl``.

    Asserting per-event would miss a second route publishing the same string, which is exactly how
    this leak survived earlier audits: the settle events were checked and the delta channel, the
    approval preview and ``status.json`` were not.
    """
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text=MODEL_PROSE)])

    result = _run(_spec(tmp_path), adapter)

    assert MODEL_PROSE not in (result.run_dir / "events.jsonl").read_text(encoding="utf-8")
    # The private artifact still has it — the text moved, it was not destroyed.
    assert MODEL_PROSE in (result.run_dir / "transcript.jsonl").read_text(encoding="utf-8")


def test_the_caller_still_receives_the_text(tmp_path: Path) -> None:
    # The run result is the caller's answer, not a fan-out surface. A change that redacted it too
    # would pass every assertion above while breaking every embedder.
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text=MODEL_PROSE)])

    result = _run(_spec(tmp_path), adapter)

    assert result.final_text == MODEL_PROSE


def test_status_json_no_longer_carries_the_answer(tmp_path: Path) -> None:
    """``StatusJsonSink`` is a fan-out sink, so no hydration seam can reach it.

    Deleting the key is deliberate: keeping it would write ``""`` on every model-answered run with
    no schema failure, since ``STATUS_SCHEMA`` never declares it and allows additional properties.
    An empty string that used to be an answer is worse than an absent key.
    """
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text=MODEL_PROSE)])

    result = _run(_spec(tmp_path), adapter)

    status = json.loads((result.run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["terminal"] is True  # the run.finished branch really did fire
    assert "final_text" not in status
    assert MODEL_PROSE not in (result.run_dir / "status.json").read_text(encoding="utf-8")


def test_kernel_authored_limit_text_stays_inline(tmp_path: Path) -> None:
    """The selectivity half, on the path that produces it.

    "Stopped after reaching max steps." is not model output. Digesting it buys no privacy and costs
    an operator the one line explaining why the run stopped — and it would need a transcript join to
    read a string the kernel wrote. No other test drives a settle event out of the limit path, so
    without this the provenance flag could be stuck ``True`` and only a human would notice.
    """
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id=f"r{index}", tool_calls=[fake_tool_call("run.finish", {})])
            for index in range(4)
        ]
    )

    result = _run(_spec(tmp_path, max_steps=1), adapter)

    for event_type in SETTLE_TYPES:
        data = _events(result.run_dir, event_type)[-1]["data"]
        assert data["status"] == "limited", event_type
        assert data["final_text"] == "Stopped after reaching max steps.", event_type
        assert "final_text_digest" not in data, event_type
        assert "final_text_len" not in data, event_type
    # And nothing was written to the private channel for it — a kernel string is not model output,
    # so it has no business in the settled-text record either.
    transcript = (result.run_dir / "transcript.jsonl").read_text(encoding="utf-8")
    kinds = [json.loads(line)["kind"] for line in transcript.splitlines() if line.strip()]
    assert "settled_text" not in kinds
