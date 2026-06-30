from __future__ import annotations

from monoid_agent_kernel.core.prompt import BASE_SYSTEM_PROMPT, compose_system_prompt


def test_no_segments_returns_base() -> None:
    assert compose_system_prompt() == BASE_SYSTEM_PROMPT.strip() + "\n"


def test_segments_are_appended_in_order() -> None:
    out = compose_system_prompt("BASE", ("first", "second"))
    assert out == "BASE\n\nfirst\n\nsecond\n"


def test_blank_segments_are_dropped() -> None:
    assert compose_system_prompt("BASE", ("", "   ", "real")) == "BASE\n\nreal\n"


def test_custom_base_overrides_default() -> None:
    assert compose_system_prompt("You are a coding agent.", ()) == "You are a coding agent.\n"
