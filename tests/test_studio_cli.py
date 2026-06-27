"""CLI-level tests for `native-agent studio doctor` — the preflight that turns late, cryptic
setup failures (busy port, unwritable dir, missing key, no browser) into an upfront checklist.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from native_agent_runner.reference.studio.cli import studio


def _invoke(tmp_path: Path, *extra: str):
    args = [
        "doctor",
        "--workspace", str(tmp_path / "ws"),
        "--run-root", str(tmp_path / "runs"),
        "--port", "0",  # ephemeral → always "free", no busy-port flake
        *extra,
    ]
    return CliRunner().invoke(studio, args)


def test_doctor_offline_all_good(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "native_agent_runner.reference.studio.window.find_chromium", lambda: "/usr/bin/chromium"
    )
    result = _invoke(tmp_path)
    assert result.exit_code == 0, result.output
    assert "[PASS]" in result.output
    assert "All hard checks passed" in result.output


def test_doctor_openai_without_key_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = _invoke(tmp_path, "--provider", "openai")
    assert result.exit_code == 1
    assert "[FAIL]" in result.output
    assert "OPENAI_API_KEY" in result.output


def test_doctor_openai_without_sdk_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Key present but the optional [openai] extra not installed → the first turn would fail, so
    # doctor must report it instead of passing.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        "native_agent_runner.reference.studio.cli._openai_sdk_importable", lambda: False
    )
    result = _invoke(tmp_path, "--provider", "openai")
    assert result.exit_code == 1
    assert "[FAIL]" in result.output
    assert "openai SDK" in result.output


def test_dir_writable_does_not_clobber_existing_files(tmp_path: Path) -> None:
    from native_agent_runner.reference.studio.cli import _dir_writable

    d = tmp_path / "ws"
    d.mkdir()
    sentinel = d / ".nar-doctor-probe"  # a file matching the old fixed probe name
    sentinel.write_text("user data", encoding="utf-8")

    assert _dir_writable(d) is True
    # the diagnostic neither overwrote nor deleted the user's file, and left no probe behind.
    assert sentinel.read_text(encoding="utf-8") == "user data"
    assert [p.name for p in d.iterdir()] == [".nar-doctor-probe"]


def test_doctor_missing_chromium_is_warning_not_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "native_agent_runner.reference.studio.window.find_chromium", lambda: None
    )
    result = _invoke(tmp_path)
    # No browser is a WARN, not a hard failure — serve still works headless.
    assert result.exit_code == 0, result.output
    assert "[WARN]" in result.output
    assert "browser" in result.output.lower()
