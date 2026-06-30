from __future__ import annotations

import json
from pathlib import Path

from support.runtime import runtime_config, runtime_provider

from native_agent_runner.core.spec import AgentRunSpec, RunLimits
from native_agent_runner.loop import AgentLoop, _accumulate_usage
from native_agent_runner.providers._common import normalize_usage
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call


# ---- A: usage detail preservation -------------------------------------------------

def test_normalize_usage_text_only_stays_three_keys() -> None:
    # Backward compatible: no detail fields means the legacy three-key shape, unchanged.
    out = normalize_usage({"input_tokens": 1, "output_tokens": 2, "total_tokens": 3})
    assert out == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}


def test_normalize_usage_preserves_anthropic_cache() -> None:
    out = normalize_usage(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "cache_read_input_tokens": 80,
            "cache_creation_input_tokens": 10,
        }
    )
    assert out["cache_read_tokens"] == 80
    assert out["cache_creation_tokens"] == 10


def test_normalize_usage_preserves_openai_nested_details() -> None:
    out = normalize_usage(
        {
            "input_tokens": 50,
            "output_tokens": 40,
            "total_tokens": 90,
            "input_tokens_details": {"cached_tokens": 30, "audio_tokens": 4},
            "output_tokens_details": {"reasoning_tokens": 15},
        },
        legacy_aliases=True,
    )
    assert out["cache_read_tokens"] == 30
    assert out["reasoning_tokens"] == 15
    assert out["audio_tokens"] == 4


def test_accumulate_usage_sums_detail_keys() -> None:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    _accumulate_usage(total, ModelTurn(usage={"input_tokens": 10, "total_tokens": 15, "cache_read_tokens": 8}))
    _accumulate_usage(total, ModelTurn(usage={"input_tokens": 5, "total_tokens": 7, "cache_read_tokens": 2}))
    assert total["input_tokens"] == 15
    assert total["total_tokens"] == 22
    assert total["cache_read_tokens"] == 10


# ---- B: token budget enforcement --------------------------------------------------

def test_run_limits_token_budget_round_trip() -> None:
    limits = RunLimits(max_input_tokens=1000, max_total_tokens=2500)
    restored = RunLimits.from_json(json.loads(json.dumps(limits.to_json())))
    assert restored.max_input_tokens == 1000
    assert restored.max_total_tokens == 2500
    assert restored.max_output_tokens is None
    # Default is unbounded.
    assert RunLimits().max_total_tokens is None


def test_token_budget_settles_limited(tmp_path: Path) -> None:
    """Once accumulated usage crosses the cap, the next turn settles ``limited`` instead of
    starting (the run actuals, not an estimate, drive the check)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("hello\n", encoding="utf-8")
    # Turn 1 calls a tool (so the loop continues) and reports 10 total tokens; turn 2's
    # pre-call check sees 10 > 5 and stops before paying for it.
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_read", {"path": "notes.md"}, "c1"),),
                usage={"input_tokens": 6, "output_tokens": 4, "total_tokens": 10},
            ),
            ModelTurn(response_id="r2", final_text="done"),
        ]
    )
    spec = AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        limits=RunLimits(max_total_tokens=5),
    )

    result = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("fs.read", "run.finish")),
    ).run_once("read the notes")

    assert result.status == "limited"
    assert result.error_code == "total_tokens_exceeded"
    # The second turn was never sent.
    assert len(adapter.requests) == 1


def test_token_budget_off_by_default_completes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("run_finish", {"summary": "ok"}, "c"),),
                usage={"input_tokens": 999, "output_tokens": 999, "total_tokens": 1998},
            )
        ]
    )
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
    ).run_once("go")

    assert result.status == "completed"
