"""CLI-level tests for `monoid studio doctor` — the preflight that turns late, cryptic
setup failures (busy port, unwritable dir, missing key, no browser) into an upfront checklist.
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from monoid_agent_kernel.env import _FALSE_VALUES, _TRUE_VALUES
from monoid_agent_kernel.reference.studio.cli import studio


def _invoke(tmp_path: Path, *extra: str):
    args = [
        "doctor",
        "--workspace",
        str(tmp_path / "ws"),
        "--run-root",
        str(tmp_path / "runs"),
        "--port",
        "0",  # ephemeral → always "free", no busy-port flake
        *extra,
    ]
    return CliRunner().invoke(studio, args)


def test_doctor_offline_all_good(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "monoid_agent_kernel.reference.studio.window.find_chromium", lambda: "/usr/bin/chromium"
    )
    result = _invoke(tmp_path)
    assert result.exit_code == 0, result.output
    assert "[PASS]" in result.output
    assert "All hard checks passed" in result.output


def test_doctor_fails_on_a_typo_in_the_delta_kill_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed switch value is a startup error by design, so preflight has to catch it.

    Without this the doctor reported every hard check passing and `serve` then died in
    `AgentLoop.__post_init__` on the next command — the exact class of late, cryptic failure this
    command exists to convert into an upfront checklist. The doctor loads the same `.env` the loop
    reads, which is where the typo lives.

    The remedy line matters as much as the failure: `of` is a plausible spelling of `off`, so the
    output has to say what *is* accepted rather than only that this is not.
    """
    monkeypatch.setenv("MONOID_OUTPUT_DELTAS", "of")
    result = _invoke(tmp_path, "--no-env-file")
    assert result.exit_code == 1, result.output
    assert "MONOID_OUTPUT_DELTAS" in result.output
    assert "[FAIL]" in result.output
    # Driven off the constants, not a substring: `"off" in output` was satisfied by the doctor's
    # own earlier `[PASS] provider 'offline'` line, so this assertion held even when the remedy
    # named nothing at all.
    for accepted in sorted(_TRUE_VALUES | _FALSE_VALUES):
        assert repr(accepted) in result.output, f"the remedy never names {accepted!r}"


@pytest.mark.parametrize(
    ("env_value", "extra", "transport", "expected"),
    [
        (None, (), True, "live model content and private model-content.jsonl are enabled"),
        ("0", (), True, "disabled by MONOID_OUTPUT_DELTAS"),
        (None, ("--no-output-deltas",), True, "disabled by --no-output-deltas"),
        (None, (), False, "needs the [http-async] extra"),
    ],
)
def test_doctor_reports_the_effective_delta_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
    extra: tuple[str, ...],
    transport: bool,
    expected: str,
) -> None:
    """Reporting the resolved state, not just the absence of an error.

    This switch fails silently in the direction that matters: an operator who believes they turned
    off a channel publishing raw model text, and did not, is told nothing by a bare `[PASS]`.

    Every input `StudioServer.start` ANDs together gets a case, and each asserts on the *named
    cause* rather than on "off", because naming the wrong one sends someone to edit the wrong
    place. The transport case is the one a first version missed: without `httpx` — the base
    package's own default — the server uses one-shot turns and publishes no deltas at all, while
    the doctor announced raw model text was about to be published.

    `transport` is patched rather than inferred so the expected line does not depend on whether the
    machine running the suite happens to have the optional extra installed.
    """
    monkeypatch.setattr(
        "monoid_agent_kernel.reference.studio.cli._gateway_streaming_available", lambda: transport
    )
    if env_value is None:
        monkeypatch.delenv("MONOID_OUTPUT_DELTAS", raising=False)
    else:
        monkeypatch.setenv("MONOID_OUTPUT_DELTAS", env_value)

    result = _invoke(tmp_path, "--no-env-file", *extra)

    assert result.exit_code == 0, result.output
    assert expected in result.output


def test_doctor_openai_without_key_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = _invoke(tmp_path, "--provider", "openai", "--no-env-file")
    assert result.exit_code == 1
    assert "[FAIL]" in result.output
    assert "OPENAI_API_KEY" in result.output


def test_doctor_openai_without_sdk_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Key present but the optional [openai] extra not installed → the first turn would fail, so
    # doctor must report it instead of passing.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        "monoid_agent_kernel.reference.studio.cli._openai_sdk_importable", lambda: False
    )
    result = _invoke(tmp_path, "--provider", "openai")
    assert result.exit_code == 1
    assert "[FAIL]" in result.output
    assert "openai SDK" in result.output


def test_doctor_openai_reads_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "monoid_agent_kernel.reference.studio.cli._openai_sdk_importable", lambda: True
    )
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-from-env-file\n", encoding="utf-8")

    result = _invoke(tmp_path, "--provider", "openai", "--env-file", str(env_file))

    assert result.exit_code == 0, result.output
    assert f"OPENAI_API_KEY is set from {env_file}" in result.output


def test_doctor_env_file_does_not_override_existing_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-process")
    monkeypatch.setattr(
        "monoid_agent_kernel.reference.studio.cli._openai_sdk_importable", lambda: True
    )
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-from-file\n", encoding="utf-8")

    result = _invoke(tmp_path, "--provider", "openai", "--env-file", str(env_file))

    assert result.exit_code == 0, result.output
    assert os.environ["OPENAI_API_KEY"] == "sk-from-process"
    assert f"from {env_file}" not in result.output


def test_doctor_no_env_file_skips_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "monoid_agent_kernel.reference.studio.cli._openai_sdk_importable", lambda: True
    )
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-from-file\n", encoding="utf-8")

    result = _invoke(tmp_path, "--provider", "openai", "--env-file", str(env_file), "--no-env-file")

    assert result.exit_code == 1
    assert "OPENAI_API_KEY is not set" in result.output
    assert os.environ.get("OPENAI_API_KEY") is None


def test_dir_writable_does_not_clobber_existing_files(tmp_path: Path) -> None:
    from monoid_agent_kernel.reference.studio.cli import _dir_writable

    d = tmp_path / "ws"
    d.mkdir()
    sentinel = d / ".nar-doctor-probe"  # a file matching the old fixed probe name
    sentinel.write_text("user data", encoding="utf-8")

    assert _dir_writable(d) is True
    # the diagnostic neither overwrote nor deleted the user's file, and left no probe behind.
    assert sentinel.read_text(encoding="utf-8") == "user data"
    assert [p.name for p in d.iterdir()] == [".nar-doctor-probe"]


def test_openai_sdk_probe_rejects_legacy_sdk_without_responses_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A bare `import openai` succeeds on versions that predate the Responses API the adapter
    # calls (client.responses.create). The probe must reject those, not report a false pass.
    import sys
    import types

    from monoid_agent_kernel.reference.studio.cli import _openai_sdk_importable

    legacy = types.ModuleType("openai")
    legacy.OpenAI = type("OpenAI", (), {})  # no `responses` attribute → too old
    legacy.AsyncOpenAI = type("AsyncOpenAI", (), {})
    monkeypatch.setitem(sys.modules, "openai", legacy)
    assert _openai_sdk_importable() is False

    modern = types.ModuleType("openai")
    modern.OpenAI = type("OpenAI", (), {"responses": property(lambda self: None)})
    modern.AsyncOpenAI = type("AsyncOpenAI", (), {"responses": property(lambda self: None)})
    monkeypatch.setitem(sys.modules, "openai", modern)
    assert _openai_sdk_importable() is True


def test_doctor_missing_chromium_is_warning_not_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("monoid_agent_kernel.reference.studio.window.find_chromium", lambda: None)
    result = _invoke(tmp_path)
    # No browser is a WARN, not a hard failure — serve still works headless.
    assert result.exit_code == 0, result.output
    assert "[WARN]" in result.output
    assert "browser" in result.output.lower()


def test_accept_runs_offline_deterministic_checks(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        studio,
        [
            "accept",
            "--workspace",
            str(tmp_path / "ws"),
            "--run-root",
            str(tmp_path / "runs"),
            "--timeout",
            "10",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["chat"]["run_id"]
    assert payload["chat"]["transcript_messages"] >= 2
    assert any(check["name"] == "deterministic-chat" and check["ok"] for check in payload["checks"])
    assert any(check["name"] == "chat-transcript" and check["ok"] for check in payload["checks"])


def test_accept_falls_back_without_http_async_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "monoid_agent_kernel.reference.studio.server._gateway_streaming_available",
        lambda: False,
    )
    result = CliRunner().invoke(
        studio,
        [
            "accept",
            "--workspace",
            str(tmp_path / "ws"),
            "--run-root",
            str(tmp_path / "runs"),
            "--timeout",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["chat"]["state"] == "awaiting_input"
    assert payload["chat"]["final_text"]
    assert payload["chat"]["transcript_messages"] >= 2

    run_dir = tmp_path / "runs" / payload["chat"]["run_id"]
    events = [
        json.loads(line)
        for line in run_dir.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    event_types = {event["type"] for event in events}
    assert "model.output.delta" not in event_types
    assert {"model.turn.finished", "turn.settled"} <= event_types
    transcript = [
        json.loads(line)
        for line in run_dir.joinpath("studio.chat.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [message["role"] for message in transcript] == ["user", "assistant"]
