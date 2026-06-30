"""The tool_ids constants must stay in lockstep with the actual builtin tool surface, and be
reachable from the package root (the discovery surface that replaces bare-string ids)."""

from __future__ import annotations

import monoid_agent_kernel.tools.tool_ids as tool_ids_mod


def _declared_constants() -> dict[str, str]:
    return {
        name: value
        for name, value in vars(tool_ids_mod).items()
        if name.isupper() and isinstance(value, str)
    }


def test_constants_match_builtin_tool_ids() -> None:
    from monoid_agent_kernel import list_builtin_tools, tool_ids

    builtin_ids = {spec.id for spec in list_builtin_tools()}
    constant_values = set(_declared_constants().values())

    # Every builtin id has a matching constant...
    missing = builtin_ids - constant_values
    assert not missing, f"builtin ids without a tool_ids constant: {sorted(missing)}"

    # ...and the constants are exactly the builtin ids plus agent.spawn (registered only when
    # subagents are loaded, so it isn't in builtin_tools()).
    assert constant_values == builtin_ids | {tool_ids.AGENT_SPAWN}


def test_constants_are_reachable_from_package_root() -> None:
    from monoid_agent_kernel import list_builtin_tools, tool_ids

    assert tool_ids.FS_READ == "fs.read"
    assert tool_ids.RUN_FINISH == "run.finish"
    # The discovery helper works without a workspace (id/description only).
    specs = list_builtin_tools()
    assert any(spec.id == "fs.read" for spec in specs)
