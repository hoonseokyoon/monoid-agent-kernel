from __future__ import annotations

# ruff: noqa: E402

import contextlib
import faulthandler
import json
import os
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --- Hang guard -------------------------------------------------------------------------
# The threaded-HTTP / shared-asyncio-loop / subprocess tests have, very rarely, left a test
# blocked on a timing race. For a backgrounded, output-buffered run that looks like the whole
# suite silently stalled forever (no output, process alive but idle) — which is exactly what bit
# us. Arm a process-wide watchdog: if the run wedges past this deadline, dump every thread's
# stack (pinpointing the deadlock) and abort, instead of hanging indefinitely. The suite
# finishes well under this; override with NAR_TEST_HANG_TIMEOUT_S=0 to disable, or a custom value.
faulthandler.enable()
_HANG_TIMEOUT_S = float(os.environ.get("NAR_TEST_HANG_TIMEOUT_S", "240"))
if _HANG_TIMEOUT_S > 0:
    faulthandler.dump_traceback_later(_HANG_TIMEOUT_S, exit=True)


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    del session, exitstatus
    # The run completed within the deadline — disarm so the watchdog can't fire during a
    # (bounded) interpreter shutdown and turn a green run red.
    faulthandler.cancel_dump_traceback_later()

from native_agent_runner.core.agents import (
    AgentRuntimeConfig,
    RegistryToolRef,
    StaticRuntimeConfigProvider,
    ToolBinding,
)
from native_agent_runner.core.spec import ModelConfig
from native_agent_runner.core.tool_surface import ToolAuthorizationDecision, ToolExposure, ToolGuidance, ToolQuota, ToolScope

__all__ = ["StaticRuntimeConfigProvider"]


def tool_binding(
    tool_id: str,
    *,
    binding_id: str | None = None,
    model_name: str | None = None,
    exposure: ToolExposure = "immediate",
    authorization: ToolAuthorizationDecision = "allow",
    guidance: str = "",
    scope: ToolScope | None = None,
    quota: ToolQuota | None = None,
    runtime: dict | None = None,
) -> ToolBinding:
    resolved_binding_id = binding_id or tool_id
    return ToolBinding(
        binding_id=resolved_binding_id,
        model_name=model_name or resolved_binding_id.replace(".", "_"),
        ref=RegistryToolRef(tool_id),
        exposure=exposure,
        authorization=authorization,
        guidance=ToolGuidance(summary=guidance),
        scope=scope or ToolScope(),
        quota=quota or ToolQuota(),
        runtime=runtime or {},
    )


def runtime_config(
    *tool_ids: str,
    definition_id: str = "test-agent",
    version: int = 1,
    model: ModelConfig | None = None,
    bindings: tuple[ToolBinding, ...] | None = None,
) -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        definition_id=definition_id,
        config_version=version,
        model=model,
        tools=bindings if bindings is not None else tuple(tool_binding(tool_id) for tool_id in tool_ids),
    )


def runtime_provider(config: AgentRuntimeConfig) -> StaticRuntimeConfigProvider:
    return StaticRuntimeConfigProvider(config)


# --- Shared HTTP test harness (graceful server lifecycle + load-resilient client) --------


def wait_http_ready(base_url: str, *, timeout_s: float = 15.0) -> None:
    """Poll /healthz until the server answers. The generous timeout tolerates a server
    thread that starts slowly when the full suite's background threads contend for CPU."""
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(Request(f"{base_url}/healthz"), timeout=2) as response:
                response.read()
            return
        except Exception as exc:  # noqa: BLE001 - any failure means not-yet-ready
            last_error = exc
            time.sleep(0.02)
    raise TimeoutError(f"server did not become ready: {last_error}")


@contextlib.contextmanager
def serving(server: Any) -> Iterator[str]:
    """Run an HTTP server on a thread and shut it down gracefully on exit. Yields the base
    URL once the server is ready. Centralizes the start/ready/shutdown/close/join dance so
    every HTTP test tears down cleanly (no abandoned handler threads racing a closing
    socket — the source of the suite's intermittent connection-abort failures)."""
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        wait_http_ready(base_url)
        yield base_url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def http_json(
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    token: str | None = None,
    method: str | None = None,
    retries: int = 5,
) -> dict[str, Any]:
    """JSON request helper that is resilient to transient connection-level errors under
    load (reset/refused/disconnect), retrying with a short backoff. It NEVER retries an
    ``HTTPError`` — a 4xx/5xx response is a real result and propagates immediately so tests
    still assert real server behavior."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers: dict[str, str] = {}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    resolved_method = method or ("POST" if data is not None else "GET")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, data=data, headers=headers, method=resolved_method)
            with urlopen(request, timeout=5) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body else {}
        except HTTPError:
            raise  # a real HTTP response — never retry
        except (URLError, ConnectionError, OSError) as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    raise last_error if last_error is not None else RuntimeError("http_json failed without an error")
