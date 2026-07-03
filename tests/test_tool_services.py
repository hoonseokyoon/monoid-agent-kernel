from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from monoid_agent_kernel.core.tool_surface import ToolScope
from monoid_agent_kernel.tool_services import CallContext, JobsService, ShellService, WebService


@dataclass
class _RecordedEvent:
    event_id: str


@dataclass
class _RecordingRecorder:
    events: list[dict[str, Any]]

    def emit(
        self,
        event_type: str,
        *,
        turn_id: str | None = None,
        parent_id: str | None = None,
        data: dict[str, Any] | None = None,
        level: str = "info",
    ) -> _RecordedEvent:
        event_id = f"event_{len(self.events) + 1}"
        self.events.append(
            {
                "event_id": event_id,
                "type": event_type,
                "turn_id": turn_id,
                "parent_id": parent_id,
                "data": dict(data or {}),
                "level": level,
            }
        )
        return _RecordedEvent(event_id)


@dataclass
class _CapturingWebGateway:
    payloads: list[dict[str, Any]]

    def fetch(self, payload: dict[str, Any], *, token: str | None = None) -> dict[str, Any]:
        del token
        self.payloads.append(dict(payload))
        return {
            "final_url": payload["url"],
            "content": "ok",
            "content_bytes": 2,
            "truncated": False,
        }


def test_call_context_holds_in_flight_ids() -> None:
    call = CallContext(tool_call_id="c1", turn_id="turn_0001", tool_event_id="e1")
    assert (call.tool_call_id, call.turn_id, call.tool_event_id) == ("c1", "turn_0001", "e1")


def test_shell_service_metrics_start_at_zero() -> None:
    service = ShellService(
        run_id="r",
        workspace=None,  # type: ignore[arg-type]
        recorder=None,  # type: ignore[arg-type]
        job_manager=None,  # type: ignore[arg-type]
        permission_policy=None,  # type: ignore[arg-type]
    )
    assert service.metrics() == {
        "shell_calls": 0,
        "failed_shell_calls": 0,
        "total_shell_duration_s": 0.0,
    }


def test_web_service_metrics_keys() -> None:
    service = WebService(recorder=None)  # type: ignore[arg-type]
    assert set(service.metrics()) == {
        "web_search_calls",
        "web_fetch_calls",
        "web_context_calls",
        "web_failed_calls",
        "web_result_count",
        "web_bytes_returned",
        "web_context_source_count",
        "web_context_bytes_returned",
    }
    assert all(value == 0 for value in service.metrics().values())


def test_web_fetch_payload_carries_binding_domain_constraints() -> None:
    recorder = _RecordingRecorder(events=[])
    gateway = _CapturingWebGateway(payloads=[])
    service = WebService(recorder=recorder, web_gateway_client=gateway)  # type: ignore[arg-type]
    call = CallContext(
        tool_call_id="call_1",
        turn_id="turn_1",
        tool_event_id="tool_event_1",
        binding_id="fetch_docs",
        scope=ToolScope(
            allowed_domains=("docs.example.test",),
            blocked_domains=("blog.example.test",),
        ),
    )

    service.fetch({"url": "https://docs.example.test/page"}, call)

    assert gateway.payloads[0]["allowed_domains"] == ["docs.example.test"]
    assert gateway.payloads[0]["blocked_domains"] == ["blog.example.test"]
    assert recorder.events[0]["data"]["allowed_domains"] == ["docs.example.test"]
    assert recorder.events[0]["data"]["blocked_domains"] == ["blog.example.test"]


def test_web_fetch_domain_filter_uses_scope_relation_for_wildcards() -> None:
    recorder = _RecordingRecorder(events=[])
    gateway = _CapturingWebGateway(payloads=[])
    service = WebService(recorder=recorder, web_gateway_client=gateway)  # type: ignore[arg-type]
    call = CallContext(
        tool_call_id="call_1",
        turn_id="turn_1",
        tool_event_id="tool_event_1",
        binding_id="fetch_docs",
        scope=ToolScope(
            allowed_domains=("*.example.test",),
            blocked_domains=("blog.example.test",),
        ),
    )

    service.fetch(
        {
            "url": "https://docs.example.test/page",
            "allowed_domains": ["docs.example.test"],
            "blocked_domains": ["private.example.test"],
        },
        call,
    )

    assert gateway.payloads[0]["allowed_domains"] == ["docs.example.test"]
    assert gateway.payloads[0]["blocked_domains"] == ["blog.example.test", "private.example.test"]
    assert recorder.events[0]["data"]["allowed_domains"] == ["docs.example.test"]


def test_web_fetch_domain_filter_accepts_nested_wildcard_narrowing() -> None:
    recorder = _RecordingRecorder(events=[])
    gateway = _CapturingWebGateway(payloads=[])
    service = WebService(recorder=recorder, web_gateway_client=gateway)  # type: ignore[arg-type]
    call = CallContext(
        tool_call_id="call_1",
        turn_id="turn_1",
        tool_event_id="tool_event_1",
        binding_id="fetch_docs",
        scope=ToolScope(allowed_domains=("*.example.test",)),
    )

    service.fetch(
        {
            "url": "https://docs.example.test/page",
            "allowed_domains": ["*.docs.example.test"],
        },
        call,
    )

    assert gateway.payloads[0]["allowed_domains"] == ["*.docs.example.test"]


def test_web_fetch_domain_filter_rejects_requested_widening_before_gateway_call() -> None:
    recorder = _RecordingRecorder(events=[])
    gateway = _CapturingWebGateway(payloads=[])
    service = WebService(recorder=recorder, web_gateway_client=gateway)  # type: ignore[arg-type]
    call = CallContext(
        tool_call_id="call_1",
        turn_id="turn_1",
        tool_event_id="tool_event_1",
        binding_id="fetch_docs",
        scope=ToolScope(allowed_domains=("*.docs.example.test",)),
    )

    with pytest.raises(ValueError, match="allowed_domains exceeds signed scope"):
        service.fetch(
            {
                "url": "https://docs.example.test/page",
                "allowed_domains": ["*.example.test"],
            },
            call,
        )

    assert gateway.payloads == []


@dataclass
class _StubJobManager:
    jobs: list[dict[str, Any]]

    def list_jobs(self) -> list[dict[str, Any]]:
        return self.jobs


def test_jobs_service_background_metrics_aggregates_by_status() -> None:
    manager = _StubJobManager(
        jobs=[
            {"status": "exited", "duration_s": 1.5, "stdout_bytes": 10, "stderr_bytes": 2},
            {"status": "failed", "duration_s": 0.5, "stdout_bytes": 4, "stderr_bytes": 1},
            {"status": "cancelled", "duration_s": 0.0, "stdout_bytes": 0, "stderr_bytes": 0},
            {"status": "running", "stdout_bytes": 7, "stderr_bytes": 3},
        ]
    )
    service = JobsService(job_manager=manager)  # type: ignore[arg-type]
    metrics = service.background_metrics()
    assert metrics["background_jobs_started"] == 4
    assert metrics["background_jobs_finished"] == 1
    assert metrics["background_jobs_failed"] == 1
    assert metrics["background_jobs_cancelled"] == 1
    assert metrics["background_job_duration_s_total"] == 2.0
    assert metrics["background_job_bytes_stdout"] == 21
    assert metrics["background_job_bytes_stderr"] == 6
