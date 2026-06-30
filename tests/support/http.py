from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Iterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def wait_http_ready(base_url: str, *, timeout_s: float = 15.0) -> None:
    """Poll /healthz until the server answers."""
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
    """Run an HTTP server on a thread and shut it down gracefully."""
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
    """JSON request helper with transient connection retries."""
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
            raise
        except (URLError, ConnectionError, OSError) as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    raise last_error if last_error is not None else RuntimeError("http_json failed without an error")


def http_get_json(url: str, *, token: str | None = None, retries: int = 5) -> dict[str, Any]:
    return http_json(url, token=token, method="GET", retries=retries)


def http_post_json(
    url: str,
    payload: dict[str, Any],
    *,
    token: str | None = None,
    retries: int = 5,
) -> dict[str, Any]:
    return http_json(url, payload, token=token, method="POST", retries=retries)

