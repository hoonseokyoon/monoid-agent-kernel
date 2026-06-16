from __future__ import annotations

import pytest

from native_agent_runner.core.profiles import PROFILE_NAMES, resolve_profile
from native_agent_runner.core.spec import AgentRunSpec


def _spec(payload_extra: dict) -> AgentRunSpec:
    return AgentRunSpec.from_json(
        {"instruction": "do it", "workspace_root": "ws", **payload_extra}
    )


def test_profile_names() -> None:
    assert PROFILE_NAMES == ("lightweight", "standard", "heavyweight")


def test_lightweight_is_read_only_with_no_write_shell_web() -> None:
    spec = _spec({"profile": "lightweight"})
    assert spec.mode == "read-only"
    assert spec.shell_policy.enabled is False
    assert spec.web_policy.enabled is False
    assert spec.limits.max_steps == 15
    assert spec.effective_capabilities() == frozenset(
        {"fs.read", "text.search", "artifact.control", "run.control"}
    )


def test_standard_matches_no_profile_defaults() -> None:
    with_profile = _spec({"profile": "standard"})
    without_profile = _spec({})
    assert with_profile.mode == without_profile.mode == "propose"
    assert with_profile.limits == without_profile.limits
    assert with_profile.shell_policy == without_profile.shell_policy
    assert with_profile.web_policy == without_profile.web_policy
    assert with_profile.effective_capabilities() == without_profile.effective_capabilities()
    assert "fs.write" in with_profile.effective_capabilities()
    assert "shell.exec" not in with_profile.effective_capabilities()


def test_heavyweight_enables_shell_and_web() -> None:
    spec = _spec({"profile": "heavyweight"})
    assert spec.mode == "propose"
    assert spec.shell_policy.enabled is True
    assert spec.web_policy.enabled is True
    assert spec.limits.max_steps == 60
    caps = spec.effective_capabilities()
    assert {"shell.exec", "job.control", "web.search", "web.fetch"}.issubset(caps)
    assert "web.context" not in caps  # context stays opt-in


def test_explicit_field_overrides_profile() -> None:
    spec = _spec({"profile": "heavyweight", "shell_policy": {"enabled": False}})
    assert spec.shell_policy.enabled is False
    assert "shell.exec" not in spec.effective_capabilities()
    # other heavyweight knobs still apply
    assert spec.web_policy.enabled is True
    assert spec.limits.max_steps == 60


def test_explicit_mode_and_limits_override_profile() -> None:
    spec = _spec({"profile": "lightweight", "mode": "propose", "limits": {"max_steps": 7}})
    assert spec.mode == "propose"
    assert spec.limits.max_steps == 7


def test_profile_recorded_in_metadata() -> None:
    spec = _spec({"profile": "heavyweight"})
    assert spec.metadata["profile"] == "heavyweight"


def test_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        _spec({"profile": "ultra"})


def test_resolve_profile_direct() -> None:
    assert resolve_profile("standard").mode == "propose"
    with pytest.raises(ValueError):
        resolve_profile("nope")
