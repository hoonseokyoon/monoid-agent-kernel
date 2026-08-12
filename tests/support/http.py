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
        if thread.is_alive():
            raise AssertionError("HTTP server thread did not stop")


# --- the reader that can see a split frame ------------------------------------------------
#
# The characters an SSE writer must not put on the wire raw. JSON escapes every other member of
# ``str.splitlines``' set (the C0 controls) whatever ``ensure_ascii`` says; these three survive an
# ``ensure_ascii=False`` dump as themselves, so they are the whole of what a frame writer has to
# escape for itself.
LINE_SEPARATORS = "\u2028\u2029\u0085"


def sse_data_frames_by_line(
    url: str,
    *,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float = 15.0,
    stop_after: int | None = None,
) -> list[dict[str, Any]]:
    """Read an SSE route the way a third-party client does: split into LINES, parse ``data:`` ones.

    The suite's other SSE readers split the body on ``\\n\\n`` or read it whole, and neither can
    fail the way this one can -- which is exactly why a writer that splits its own frames passed
    them. ``httpx``'s line splitter is ``str.splitlines``, which breaks on U+2028, U+2029 and
    U+0085 as well as CR/LF; a browser's ``EventSource`` breaks on CR/LF only. So a frame carrying
    one of :data:`LINE_SEPARATORS` arrives whole in a browser and truncated mid-JSON here, and only
    a line reader states the difference.

    ``iter_lines`` is the sync twin of the ``aiter_lines`` an async consumer reads with and shares
    its decoder. Callers must ``pytest.importorskip("httpx")`` first. ``stop_after`` bounds a route
    that stays open rather than ending its stream.
    """

    import httpx

    request_headers = dict(headers or {})
    if token is not None:
        request_headers["Authorization"] = f"Bearer {token}"
    method = "POST" if payload is not None else "GET"
    frames: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout_s) as client:
        with client.stream(
            method, url, json=payload, headers=request_headers
        ) as response:
            assert response.headers.get("content-type", "").startswith("text/event-stream")
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line.removeprefix("data: ")
                try:
                    frames.append(json.loads(raw))
                except json.JSONDecodeError as exc:
                    raise AssertionError(
                        "an SSE frame did not survive a line-splitting reader -- the writer put a "
                        f"separator on the wire raw and the frame stops mid-JSON: {raw!r}"
                    ) from exc
                if stop_after is not None and len(frames) >= stop_after:
                    break
    return frames


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

