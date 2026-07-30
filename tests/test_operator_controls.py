from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.model_call import ModelCallRunner
from monoid_agent_kernel.providers.fake import FakeModelAdapter
from monoid_agent_kernel.tools.base import ToolResult


def _loop(tmp_path: Path, **controls: Any) -> AgentLoop:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace,
            run_root=tmp_path / "runs",
        ),
        model_adapter=FakeModelAdapter(),
        runtime_config_provider=runtime_provider(runtime_config("fs.write")),
        **controls,
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "status_file",
        "emit_output_deltas",
        "inject_workspace_index",
        "capability_auto_redispatch",
    ],
)
def test_agent_loop_rejects_truthy_non_boolean_operator_switches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
) -> None:
    monkeypatch.setenv("MONOID_OUTPUT_DELTAS", "1")

    with pytest.raises(ValueError, match=field_name):
        _loop(tmp_path, **{field_name: 1})


@pytest.mark.parametrize(
    "field_name",
    [
        "async_tool_cancel_grace_s",
        "async_model_cancel_grace_s",
        "capability_rotate_skew_seconds",
    ],
)
@pytest.mark.parametrize("invalid", [True, "1", -0.01, float("nan"), float("inf"), 10**400])
def test_agent_loop_rejects_invalid_operator_durations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    invalid: object,
) -> None:
    monkeypatch.setenv("MONOID_OUTPUT_DELTAS", "1")

    with pytest.raises(ValueError, match=field_name):
        _loop(tmp_path, **{field_name: invalid})


def test_agent_loop_canonicalizes_valid_operator_durations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MONOID_OUTPUT_DELTAS", "1")

    loop = _loop(
        tmp_path,
        async_tool_cancel_grace_s=0,
        async_model_cancel_grace_s=2,
        capability_rotate_skew_seconds=3,
    )

    assert loop.async_tool_cancel_grace_s == 0.0
    assert loop.async_model_cancel_grace_s == 2.0
    assert loop.capability_rotate_skew_seconds == 3.0


def test_agent_loop_revalidates_a_live_output_delta_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MONOID_OUTPUT_DELTAS", "1")
    loop = _loop(tmp_path, emit_output_deltas=True)
    loop.emit_output_deltas = "false"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="emit_output_deltas"):
        loop._output_deltas_enabled()


def test_agent_loop_revalidates_the_live_tool_cancel_grace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MONOID_OUTPUT_DELTAS", "1")
    loop = _loop(tmp_path)
    loop.async_tool_cancel_grace_s = float("nan")

    async def pending() -> ToolResult:
        return ToolResult(ok=True)

    async def run() -> None:
        call = pending()
        try:
            with pytest.raises(ValueError, match="async_tool_cancel_grace_s"):
                await loop._await_native_tool_handler(call, None)
        finally:
            call.close()

    asyncio.run(run())


@pytest.mark.parametrize("invalid", [True, "1", -0.01, float("nan"), float("inf"), 10**400])
def test_model_call_runner_rejects_invalid_constructed_cancel_grace(invalid: object) -> None:
    with pytest.raises(ValueError, match="cancel_grace_s"):
        ModelCallRunner(adapter=FakeModelAdapter(), cancel_grace_s=invalid)


@pytest.mark.parametrize("invalid", [True, "1", -0.01, float("nan"), float("inf"), 10**400])
def test_model_call_runner_revalidates_the_live_cancel_grace(invalid: object) -> None:
    runner = ModelCallRunner(
        adapter=FakeModelAdapter(),
        current_cancel_grace_s=lambda: invalid,  # type: ignore[return-value]
    )

    with pytest.raises(ValueError, match="cancel_grace_s"):
        runner._grace_s()


def test_model_call_runner_revalidates_a_mutated_fallback_cancel_grace() -> None:
    runner = ModelCallRunner(adapter=FakeModelAdapter(), cancel_grace_s=0.5)
    runner.cancel_grace_s = float("nan")

    with pytest.raises(ValueError, match="cancel_grace_s"):
        runner._grace_s()
