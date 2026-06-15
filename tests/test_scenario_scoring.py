from __future__ import annotations

import importlib.util
from pathlib import Path


def test_messy_workspace_fake_scenario_scores_pass(tmp_path: Path) -> None:
    module = _load_messy_scenario_module()

    result = module.run_scenario(
        mode="fake",
        model="gpt-5.5",
        reasoning_effort="low",
        root=tmp_path,
    )

    score = result["score"]
    assert score["ok"] is True, score
    assert score["score"] == score["total"]
    assert result["base_workspace_unchanged"]["summary_exists"] is False
    assert result["package_verify_ok"] is True
    assert result["secret_leak_detected"] is False


def test_messy_workspace_scorer_reports_failures() -> None:
    module = _load_messy_scenario_module()
    score = module.score_messy_workspace_result(
        {
            "status": "failed",
            "result_status": "failed",
            "changed_paths": [],
            "base_workspace_unchanged": {},
            "runner_usage": {"total_tokens": 0},
            "llm_gateway_usage": {"total_tokens": 0},
            "secret_leak_detected": True,
            "sensitive_event_mentions": [{"seq": 1}],
        }
    )

    assert score["ok"] is False
    assert "run_completed" in score["failed"]
    assert "no_secret_leak" in score["failed"]


def _load_messy_scenario_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "messy_workspace_cleanup.py"
    spec = importlib.util.spec_from_file_location("messy_workspace_cleanup", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scenario module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
