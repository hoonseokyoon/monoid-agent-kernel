from __future__ import annotations

from typing import Any


def score_messy_workspace_result(result: dict[str, Any]) -> dict[str, Any]:
    changed_paths = set(str(path) for path in result.get("changed_paths") or ())
    base = result.get("base_workspace_unchanged") if isinstance(result.get("base_workspace_unchanged"), dict) else {}
    runner_usage = result.get("runner_usage") if isinstance(result.get("runner_usage"), dict) else {}
    gateway_usage = result.get("llm_gateway_usage") if isinstance(result.get("llm_gateway_usage"), dict) else {}
    conflicts = result.get("conflicts") if isinstance(result.get("conflicts"), list) else []
    checks = {
        "run_completed": result.get("status") == "completed" and result.get("result_status") == "completed",
        "no_error": not result.get("error") and not result.get("error_code"),
        "expected_changed_paths": {"README.md", "SUMMARY.md", "TODO.md"}.issubset(changed_paths),
        "base_workspace_unchanged": base.get("summary_exists") is False
        and base.get("todo_exists") is False
        and base.get("readme_same") is True,
        "proposal_integrity": bool(result.get("proposal_hash")) and bool(result.get("diff_sha256")),
        "package_verified": result.get("package_verify_ok") is True and bool(result.get("package_hash")),
        "dry_run_ok": result.get("dry_run_status") == "dry_run",
        "full_apply_ok": result.get("full_apply_status") == "applied",
        "partial_apply_ok": result.get("partial_apply_status") == "applied"
        and result.get("partial_readme_unchanged") is True,
        "conflict_detected": result.get("conflict_status") == "conflict" and bool(conflicts),
        "usage_counted": int(runner_usage.get("total_tokens") or 0) > 0
        and int(runner_usage.get("total_tokens") or 0) == int(gateway_usage.get("total_tokens") or 0),
        "no_secret_leak": result.get("secret_leak_detected") is False
        and not result.get("sensitive_event_mentions"),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "ok": not failed,
        "score": len(checks) - len(failed),
        "total": len(checks),
        "checks": checks,
        "failed": failed,
    }
