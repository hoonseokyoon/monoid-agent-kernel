from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from support.runtime import runtime_config

from monoid_agent_kernel.errors import PermissionDenied
from monoid_agent_kernel.reference.backend.service import BackendRunRequest, RunnerBackend


def _record(seq: int) -> bytes:
    return (
        json.dumps(
            {"seq": seq, "type": "run.started", "data": {}},
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _completed_run(
    backend_factory: Any,
    *,
    event_index_max_sources: int = 128,
) -> tuple[Any, Any]:
    workspace = backend_factory.workspace()
    backend = backend_factory.create(
        workspace=workspace,
        event_index_max_sources=event_index_max_sources,
    )
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
    return backend, submission


def test_backend_event_pages_reach_bounded_warm_index_work(backend_factory: Any) -> None:
    backend, submission = _completed_run(backend_factory)
    events_path = submission.run_dir / "events.jsonl"
    count = 100_000
    events_path.write_bytes(b"".join(_record(seq) for seq in range(1, count + 1)))

    first = backend.events(
        submission.run_id,
        submission.run_token,
        from_seq=count - 4,
        limit=2,
    )
    second = backend.events(
        submission.run_id,
        submission.run_token,
        from_seq=first["next_seq"],
        limit=2,
    )
    source = backend._event_index.stats(events_path)
    cache = backend._event_index.cache_stats()

    assert [event["seq"] for event in first["events"]] == [count - 4, count - 3]
    assert [event["seq"] for event in second["events"]] == [count - 2, count - 1]
    assert source is not None
    assert source.from_zero_reads == 1
    assert source.pages == 2
    assert source.last_records_examined == 3
    assert source.last_source_bytes_read <= 4 * 64 * 1024
    assert cache.sources == 1
    assert cache.misses == 1
    assert cache.hits == 1


def test_backend_events_and_diagnostics_share_one_source_slot(backend_factory: Any) -> None:
    backend, submission = _completed_run(backend_factory)
    events_path = submission.run_dir / "events.jsonl"

    page = backend.events(
        submission.run_id,
        submission.run_token,
        from_seq=0,
        limit=1,
    )
    diagnostics = backend.diagnostics(
        submission.run_id,
        submission.run_token,
        event_limit=2,
    )
    source = backend._event_index.stats(events_path)
    cache = backend._event_index.cache_stats()

    assert page["events"]
    assert diagnostics["events"]["items"]
    assert source is not None
    assert source.pages == 2
    assert cache.sources == 1
    assert cache.misses == 1
    assert cache.hits == 1


def test_backend_descendant_pages_use_index_after_lineage_authorization(
    backend_factory: Any,
) -> None:
    backend, submission = _completed_run(backend_factory)
    child_id = f"{submission.run_id}.sub.task_1"
    child_path = backend.run_root / child_id / "events.jsonl"
    child_path.parent.mkdir(parents=True)
    child_path.write_bytes(_record(1) + _record(2))

    first = backend.descendant_events(
        submission.run_id,
        submission.run_token,
        child_id,
        from_seq=0,
        limit=1,
    )
    second = backend.descendant_events(
        submission.run_id,
        submission.run_token,
        child_id,
        from_seq=first["next_seq"],
        limit=1,
    )
    source = backend._event_index.stats(child_path)

    root = backend.events(
        submission.run_id,
        submission.run_token,
        from_seq=0,
        limit=1,
    )
    root_source = backend._event_index.stats(submission.run_dir / "events.jsonl")
    cache = backend._event_index.cache_stats()

    assert [event["seq"] for event in first["events"]] == [1]
    assert [event["seq"] for event in second["events"]] == [2]
    assert root["events"]
    assert source is not None
    assert source.from_zero_reads == 1
    assert source.pages == 2
    assert root_source is not None
    assert root_source.from_zero_reads == 1
    assert root_source.pages == 1
    assert cache.sources == 2
    assert cache.misses == 2
    assert cache.hits == 1


def test_failed_authorization_creates_no_event_index_slots(backend_factory: Any) -> None:
    backend, submission = _completed_run(backend_factory)
    root_path = submission.run_dir / "events.jsonl"
    child_id = f"{submission.run_id}.sub.task_1"
    child_path = backend.run_root / child_id / "events.jsonl"
    child_path.parent.mkdir(parents=True)
    child_path.write_bytes(_record(1))

    with pytest.raises(PermissionDenied):
        backend.events(submission.run_id, "bad-token", from_seq=0, limit=1)
    with pytest.raises(PermissionDenied):
        backend.diagnostics(submission.run_id, "bad-token", event_limit=1)
    with pytest.raises(PermissionDenied):
        backend.descendant_events(
            submission.run_id,
            submission.run_token,
            "unrelated.run",
            from_seq=0,
            limit=1,
        )

    assert backend._event_index.stats(root_path) is None
    assert backend._event_index.stats(child_path) is None
    cache = backend._event_index.cache_stats()
    assert cache.sources == 0
    assert cache.hits == 0
    assert cache.misses == 0
    assert cache.evictions == 0
    assert cache.bypasses == 0


def test_backend_event_index_capacity_zero_uses_uncached_pages(backend_factory: Any) -> None:
    backend, submission = _completed_run(
        backend_factory,
        event_index_max_sources=0,
    )

    first = backend.events(submission.run_id, submission.run_token, from_seq=0, limit=1)
    second = backend.events(submission.run_id, submission.run_token, from_seq=0, limit=1)
    cache = backend._event_index.cache_stats()

    assert first == second
    assert cache.max_sources == 0
    assert cache.sources == 0
    assert cache.misses == 2
    assert cache.bypasses == 2


def test_backend_instances_own_independent_indexes_for_shared_run_artifacts(
    backend_factory: Any,
    tmp_path: Path,
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
    events_path = submission.run_dir / "events.jsonl"

    first.events(submission.run_id, submission.run_token, from_seq=0, limit=1)
    second = backend_factory.create(
        run_root=run_root,
        workspace=workspace,
        token_manager=token_manager,
    )

    assert first._event_index is not second._event_index
    assert second._event_index.stats(events_path) is None

    second.events(submission.run_id, submission.run_token, from_seq=0, limit=1)
    first_stats = first._event_index.stats(events_path)
    second_stats = second._event_index.stats(events_path)

    assert first_stats is not None
    assert first_stats.from_zero_reads == 1
    assert first_stats.pages == 1
    assert second_stats is not None
    assert second_stats.from_zero_reads == 1
    assert second_stats.pages == 1
    assert first._event_index.cache_stats().misses == 1
    assert second._event_index.cache_stats().misses == 1


def test_backend_validates_event_index_capacity_before_creating_run_root(
    backend_factory: Any,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "invalid-runs"

    with pytest.raises(ValueError, match="max_sources"):
        backend_factory.create(
            run_root=run_root,
            workspace=backend_factory.workspace(),
            event_index_max_sources=-1,
        )

    assert not run_root.exists()


def test_backend_event_index_capacity_is_additive_keyword_only_config() -> None:
    parameter = inspect.signature(RunnerBackend).parameters["event_index_max_sources"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == 128
