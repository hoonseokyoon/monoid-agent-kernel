"""Low-level process and filesystem primitives shared by shell and jobs.

Domain-neutral helpers that both the foreground shell runner (``shell.py``) and
the background job manager (``jobs.py``) depend on. Internal only; not part of
the supported public surface.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import IO


def file_size(path: Path) -> int:
    """Size of ``path`` in bytes, or ``0`` if it does not exist."""
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def proc_group_kwargs() -> dict[str, object]:
    """Platform kwargs that put a child in its own process group so the whole tree can
    be terminated together by :func:`terminate_process`. Shared by the sync ``Popen`` and
    the asyncio (``create_subprocess_exec``) spawn paths so they behave identically."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"preexec_fn": os.setsid}


def spawn_process(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout: IO[bytes],
    stderr: IO[bytes],
) -> subprocess.Popen[bytes]:
    """Start ``argv`` in its own process group with stdin disabled.

    A new process group (Windows ``CREATE_NEW_PROCESS_GROUP`` / POSIX
    ``setsid``) lets the whole child tree be terminated together by
    :func:`terminate_process`.
    """
    return subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        **proc_group_kwargs(),  # type: ignore[arg-type]
    )


# The external group-killer (``taskkill``) is itself a spawned process; under load/AV its
# spawn can stall. Bound it so a stalled kill can never block the caller — the foreground
# shell loop or the background job monitor thread — indefinitely.
_TERMINATE_TIMEOUT_S = 10.0


def terminate_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate ``process`` and its group, falling back to a direct ``kill`` on error or if
    the group kill does not return promptly. The group kill (Windows ``taskkill``) is bounded
    by a timeout so a stalled killer escalates to ``process.kill()`` instead of hanging."""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=_TERMINATE_TIMEOUT_S,
            )
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        # taskkill timed out / failed, or killpg failed: terminate the child handle directly.
        try:
            process.kill()
        except Exception:
            pass
