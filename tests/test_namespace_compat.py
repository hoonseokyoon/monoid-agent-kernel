import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_native_agent_runner_namespace_aliases_new_package() -> None:
    for module_name in list(sys.modules):
        if module_name == "native_agent_runner" or module_name.startswith("native_agent_runner."):
            sys.modules.pop(module_name)

    current = importlib.import_module("monoid_agent_kernel")
    current_spec = importlib.import_module("monoid_agent_kernel.core.spec")

    with pytest.warns(DeprecationWarning, match="monoid_agent_kernel"):
        legacy = importlib.import_module("native_agent_runner")

    legacy_spec = importlib.import_module("native_agent_runner.core.spec")

    assert legacy.AgentLoop is current.AgentLoop
    assert legacy_spec is current_spec
    assert legacy_spec.AgentRunSpec is current_spec.AgentRunSpec


def test_native_agent_runner_cli_module_execution_stays_compatible() -> None:
    root = Path(__file__).resolve().parents[1]
    src = str(root / "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    result = subprocess.run(
        [sys.executable, "-m", "native_agent_runner.cli", "--help"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Run Monoid Agent Kernel." in result.stdout
