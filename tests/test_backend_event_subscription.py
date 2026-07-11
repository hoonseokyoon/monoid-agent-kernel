from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from support.runtime import runtime_config

from monoid_agent_kernel.errors import PermissionDenied
from monoid_agent_kernel.reference.backend.service import BackendRunRequest


def test_subscription_drains_recovered_run_and_resumes_without_replay(
    backend_factory: Any, tmp_path: Path
) -> None:
    workspace = backend_factory.workspace()
    run_root = tmp_path / "shared-runs"
    token_manager = backend_factory.token_manager()
    first = backend_factory.create(
        run_root=run_root,
        workspace=workspace,
        token_manager=token_manager,
    )
    submission = first.submit_run(
        BackendRunRequest(
            tenant_id="tenant",
            user_id="user",
            workspace_root=workspace,
            instruction="finish",
            runtime_config=runtime_config("run.finish"),
        )
    )
    assert first.wait_for_run(submission.run_id, timeout_s=10).value == "completed"
    first.shutdown(drain=True)

    recovered = backend_factory.create(
        run_root=run_root,
        workspace=workspace,
        token_manager=token_manager,
    )
    frames = list(recovered.subscribe_events(submission.run_id, submission.run_token).frames())
    event_frames = [frame for frame in frames if frame.kind == "event"]
    assert event_frames
    assert frames[-1].kind == "end"
    assert frames[-1].lifecycle is not None
    assert frames[-1].lifecycle["state"] == "completed"

    resumed = list(
        recovered.subscribe_events(
            submission.run_id,
            submission.run_token,
            last_event_id=event_frames[0].event_id,
        ).frames()
    )
    assert [frame.event_id for frame in resumed if frame.kind == "event"] == [
        frame.event_id for frame in event_frames[1:]
    ]


def test_descendant_subscription_preserves_ancestor_authorization(
    backend_factory: Any, tmp_path: Path
) -> None:
    workspace = backend_factory.workspace()
    backend = backend_factory.create(workspace=workspace)
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant",
            user_id="user",
            workspace_root=workspace,
            instruction="finish",
            runtime_config=runtime_config("run.finish"),
        )
    )
    assert backend.wait_for_run(submission.run_id, timeout_s=10).value == "completed"
    child_id = f"{submission.run_id}.sub.task_1"
    child_dir = backend.run_root / child_id
    child_dir.mkdir(parents=True)
    (child_dir / "events.jsonl").write_text(
        json.dumps({"seq": 1, "type": "child.started"})
        + "\n"
        + json.dumps({"seq": 2, "type": "child.finished"})
        + "\n",
        encoding="utf-8",
    )
    (child_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": child_id,
                "status": "completed",
                "last_event_seq": 2,
            }
        ),
        encoding="utf-8",
    )

    page = backend.subscribe_descendant_events(
        submission.run_id, submission.run_token, child_id
    ).poll()
    assert [event["seq"] for event in page["events"]] == [1, 2]
    frames = list(
        backend.subscribe_descendant_events(
            submission.run_id, submission.run_token, child_id
        ).frames()
    )
    assert [frame.kind for frame in frames] == ["event", "event", "end"]
    assert frames[-1].lifecycle is not None
    assert frames[-1].lifecycle["state"] == "completed"
    with pytest.raises(PermissionDenied):
        backend.subscribe_descendant_events(
            submission.run_id, submission.run_token, "unrelated.run"
        ).poll()
    with pytest.raises(Exception):
        backend.subscribe_descendant_events(submission.run_id, "bad-token", child_id).poll()
