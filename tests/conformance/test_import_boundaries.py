from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFORMANCE_SRC = ROOT / "src" / "monoid_agent_kernel" / "conformance"
CONFORMANCE_TESTS = ROOT / "tests" / "conformance"

REFERENCE_SCENARIO_STRINGS = {
    "completed",
    "multi-turn",
    "parked-hitl",
    "recoverable-multi-turn",
    "subagent-capability-revoked",
    "subagent-foreground",
    "tool-ask-approved",
    "tool-ask-denied",
    "tool-ask-stale-denied",
    "tool-quota-denied",
}

REFERENCE_FIXTURE_STRINGS = {
    "demo_approval",
    "mcp.demo.gated",
}


def test_conformance_source_does_not_import_reference_modules() -> None:
    offenders: list[str] = []
    for path in _python_files(CONFORMANCE_SRC):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported = _imported_module(node)
            if imported and imported.startswith("monoid_agent_kernel.reference"):
                offenders.append(_location(path, node))

    assert offenders == []


def test_conformance_imports_do_not_load_reference_modules() -> None:
    code = """
import importlib
import json
import pkgutil
import sys

import monoid_agent_kernel.conformance as conformance

for module in pkgutil.walk_packages(conformance.__path__, conformance.__name__ + "."):
    importlib.import_module(module.name)

loaded = sorted(name for name in sys.modules if name.startswith("monoid_agent_kernel.reference"))
print(json.dumps(loaded))
raise SystemExit(1 if loaded else 0)
"""
    env = dict(os.environ)
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_generic_profiles_do_not_call_reference_scenarios() -> None:
    offenders: list[str] = []
    forbidden_strings = REFERENCE_SCENARIO_STRINGS | REFERENCE_FIXTURE_STRINGS
    for path in _python_files(CONFORMANCE_SRC / "profiles"):
        if path.name in {"side_effect_tool_agent.py", "message_fabric.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node.func) == "submit_run":
                offenders.append(_location(path, node))
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in forbidden_strings:
                offenders.append(f"{_location(path, node)} uses {node.value!r}")

    assert offenders == []


def test_conformance_tests_import_reference_harness_only() -> None:
    offenders: list[str] = []
    for path in _python_files(CONFORMANCE_TESTS):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported = _imported_module(node)
            if (
                imported
                and imported.startswith("monoid_agent_kernel.reference")
                and imported != "monoid_agent_kernel.reference.conformance"
            ):
                offenders.append(_location(path, node))

    assert offenders == []


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _imported_module(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
        return next((name for name in names if name.startswith("monoid_agent_kernel.")), names[0])
    if isinstance(node, ast.ImportFrom):
        return node.module
    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _location(path: Path, node: ast.AST) -> str:
    return f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', 0)}"
