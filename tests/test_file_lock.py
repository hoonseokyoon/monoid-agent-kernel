from __future__ import annotations

import pytest

from monoid_agent_kernel.core import _util


def test_file_lock_treats_permission_error_as_lock_contention(tmp_path, monkeypatch) -> None:
    lock_path = tmp_path / ".put.lock"
    lock_path.write_text("held", encoding="utf-8")
    real_open = _util.os.open
    calls = 0

    def flaky_open(path, flags, mode=0o777):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("sharing violation")
        return real_open(path, flags, mode)

    monkeypatch.setattr(_util.os, "open", flaky_open)

    with _util.file_lock(lock_path, timeout_s=1.0, stale_s=0.0):
        assert calls >= 2

    assert not lock_path.exists()


def test_file_lock_raises_permission_error_when_lock_cannot_be_created(tmp_path, monkeypatch) -> None:
    lock_path = tmp_path / ".put.lock"

    def denied_open(path, flags, mode=0o777):
        raise PermissionError("create denied")

    monkeypatch.setattr(_util.os, "open", denied_open)

    with pytest.raises(PermissionError):
        with _util.file_lock(lock_path, timeout_s=0.1, stale_s=0.0):
            raise AssertionError("unreachable")


def test_file_lock_retries_transient_permission_error_without_visible_lock(tmp_path, monkeypatch) -> None:
    lock_path = tmp_path / ".put.lock"
    real_open = _util.os.open
    calls = 0

    def flaky_open(path, flags, mode=0o777):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("transient sharing violation")
        return real_open(path, flags, mode)

    monkeypatch.setattr(_util.os, "open", flaky_open)

    with _util.file_lock(lock_path, timeout_s=1.0, stale_s=0.0):
        assert calls == 2
