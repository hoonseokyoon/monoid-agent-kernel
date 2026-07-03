from __future__ import annotations

import pytest

from monoid_agent_kernel.core.subagent_runtime import (
    SubagentRuntimeContext,
    is_descendant_run_id,
    root_run_id_from_descendant,
    subagent_diagnostics_from_events,
    validate_descendant_run_id,
)


def test_subagent_identity_and_payloads() -> None:
    ctx = SubagentRuntimeContext.create(
        parent_run_id="run_parent",
        task_id="task_1",
        definition_id="researcher",
        parent_depth=0,
        traceparent="00-" + "1" * 32 + "-" + "2" * 16 + "-01",
    )

    assert ctx.child_run_id == "run_parent.sub.task_1"
    assert ctx.depth == 1
    assert ctx.to_json()["subagent_type"] == "researcher"
    assert ctx.child_metadata()["parent_task_id"] == "task_1"
    assert ctx.started_event_data(background=False)["background"] is False

    terminal = ctx.terminal_event_data(
        status="completed",
        usage={"total_tokens": 10},
        error="",
        error_code="",
    )
    assert terminal["usage"]["total_tokens"] == 10

    result = ctx.result_payload(
        status="completed",
        final_text="done",
        error="",
        usage={"total_tokens": 10},
    )
    assert result["type"] == "subagent_result"
    assert result["message"] == "done"


def test_descendant_run_id_validation() -> None:
    validate_descendant_run_id("run_a", "run_a")
    validate_descendant_run_id("run_a", "run_a.sub.task_1")
    assert is_descendant_run_id("run_a", "run_a.sub.task_1.sub.task_2") is True

    with pytest.raises(ValueError, match="not a descendant"):
        validate_descendant_run_id("run_a", "run_b.sub.task_1")
    with pytest.raises(ValueError, match="invalid descendant"):
        validate_descendant_run_id("run_a", "run_a.sub.../escape")


def test_root_run_id_from_descendant() -> None:
    assert root_run_id_from_descendant("run_a.sub.task_1") == "run_a"
    assert root_run_id_from_descendant("run_a.sub.task_1.sub.task_2") == "run_a"
    assert root_run_id_from_descendant("run_a") is None
    assert root_run_id_from_descendant("../run_a.sub.task_1") is None
    assert root_run_id_from_descendant("run_a/sub/task_1") is None
    assert root_run_id_from_descendant("run_a\\sub\\task_1") is None


def test_subagent_diagnostics_summary() -> None:
    events = [
        {
            "seq": 3,
            "type": "subagent.started",
            "data": {
                "root_run_id": "run_a",
                "parent_run_id": "run_a",
                "child_run_id": "run_a.sub.task_1",
                "task_id": "task_1",
                "definition_id": "researcher",
                "depth": 1,
                "traceparent": "tp",
                "background": False,
            },
        },
        {
            "seq": 9,
            "type": "subagent.finished",
            "data": {
                "root_run_id": "run_a",
                "parent_run_id": "run_a",
                "child_run_id": "run_a.sub.task_1",
                "task_id": "task_1",
                "definition_id": "researcher",
                "depth": 1,
                "traceparent": "tp",
                "status": "completed",
                "usage": {"total_tokens": 10},
            },
        },
    ]

    summary = subagent_diagnostics_from_events(events)

    assert summary["count"] == 1
    item = summary["items"][0]
    assert item["child_run_id"] == "run_a.sub.task_1"
    assert item["started_seq"] == 3
    assert item["terminal_seq"] == 9
    assert item["status"] == "completed"
    assert item["usage"]["total_tokens"] == 10
