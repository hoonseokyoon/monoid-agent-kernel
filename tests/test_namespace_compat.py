import importlib
import sys

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
