"""An early rejection must reach the client, not be lost to a TCP reset.

A reference server that answers a POST *before* consuming the request body leaves the client's
already-sent bytes unread in the kernel receive buffer. Closing that socket sends an RST on
Windows, and the RST discards whatever the client has not yet pulled out of its own buffer — so
the response the server successfully wrote and flushed never reaches the reader. What arrives
instead is ``ConnectionAbortedError (WinError 10053)``.

The cost is not a lost error message. It is a *reclassified* one: a client that would have read
``error_code: gateway_auth_error`` with ``retryable: false`` sees a transport-level abort, which
every retry policy in this repo treats as a transient network failure. An auth rejection then gets
retried as though the credential might work next time, on every transport that has a retry loop.

The race is one-sided in the tests below and only ~5-15% in the wild for one reason: the server's
header read is *buffered* (``rfile``, 8 KiB), so when the body shares the TCP segment its headers
arrive in, that read swallows the body incidentally and the socket is left clean. ``http.client``
sends headers and body as separate writes, so the two often land in separate segments — and a
single ``sendall`` from a hand-written probe usually does not, which is exactly why a naive
reproduction of this bug reports "cannot reproduce". These probes split the send deliberately.
"""

from __future__ import annotations

import ast
import socket
import time
from pathlib import Path
from typing import Any

import pytest

from support.http import serving

from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.llm_gateway.http import create_llm_gateway_server
from monoid_agent_kernel.reference.llm_gateway.service import LlmGatewayBackend
from monoid_agent_kernel.reference.mcp_gateway import FakeMcpServer, create_mcp_server

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "monoid_agent_kernel"
# Every reference server that speaks HTTP. The rule under test belongs to all of them, so the
# census below is driven by this list rather than by the two servers the probes can cheaply build.
REFERENCE_HTTP_MODULES = (
    "reference/mcp_gateway/http.py",
    "reference/llm_gateway/http.py",
    "reference/backend/http.py",
    "reference/web_gateway/http.py",
)


def _reject_probe(base_url: str, path: str, *, extra_headers: str = "") -> str:
    """POST with a body the server will reject, sending headers and body as separate writes.

    Returns the response's status line, or the name of the exception the read raised. The
    ``time.sleep`` between the two writes is what makes the failure deterministic: it guarantees
    the server's buffered header read completes before the body arrives, so the body is still
    unread when the handler answers.
    """

    host, port = base_url.removeprefix("http://").split(":")
    body = b'{"hello": "world"}'
    head = (
        f"POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\n"
        f"{extra_headers}Content-Length: {len(body)}\r\n\r\n"
    ).encode()

    sock = socket.create_connection((host, int(port)), timeout=10)
    try:
        sock.sendall(head)
        time.sleep(0.05)
        sock.sendall(body)
        chunks: list[bytes] = []
        try:
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                chunks.append(data)
        except OSError as exc:
            return f"{type(exc).__name__}({getattr(exc, 'winerror', exc.errno)})"
        received = b"".join(chunks)
        if not received:
            return "empty (the response was written, then lost with the connection)"
        return received.split(b"\r\n", 1)[0].decode("latin-1")
    finally:
        sock.close()


def _mcp_server() -> Any:
    return create_mcp_server(FakeMcpServer(), host="127.0.0.1", port=0, admin_token="secret")


def _llm_server() -> Any:
    return create_llm_gateway_server(
        LlmGatewayBackend(token_manager=TokenManager.from_secret("z" * 32)),
        host="127.0.0.1",
        port=0,
        admin_token="admin",
    )


# Each case is a reject path that answers before anything has read the body. The two *kinds*
# matter and are both represented: an auth rejection (the handler authorizes first by design) and
# an unknown path (no handler has run at all). A fix that only drains on the auth path leaves the
# second half broken, which is this repo's recurring shape.
REJECT_CASES = (
    pytest.param(_mcp_server, "/mcp", "Authorization: Bearer wrong\r\n", "401", id="mcp-401-auth"),
    pytest.param(_mcp_server, "/nope", "", "404", id="mcp-404-unknown-path"),
    pytest.param(_llm_server, "/internal/llm/turns", "", "401", id="llm-401-no-token"),
    pytest.param(_llm_server, "/nope", "", "404", id="llm-404-unknown-path"),
)


@pytest.mark.parametrize("build_server, path, headers, expected_status", REJECT_CASES)
def test_an_early_rejection_reaches_the_client(
    build_server: Any, path: str, headers: str, expected_status: str
) -> None:
    with serving(build_server()) as base_url:
        status_line = _reject_probe(base_url, path, extra_headers=headers)

    assert expected_status in status_line, {
        "got": status_line,
        "hint": "the server answered before consuming the request body, so closing the socket "
        "reset the connection and discarded the response the client had not yet read; a "
        "transport abort is retryable to every caller, while the status it replaced is not",
    }


def test_a_rejected_body_is_drained_before_the_response_not_after() -> None:
    """The ordering, stated separately from the outcome above.

    Draining *after* writing the response would still lose the race: the bytes have to leave the
    receive buffer before the socket closes, and the close follows the write immediately. This
    pins the observable consequence — a second request on a fresh connection is answered normally,
    which it cannot be if the first one left the server mid-body.
    """

    with serving(_mcp_server()) as base_url:
        first = _reject_probe(base_url, "/mcp", extra_headers="Authorization: Bearer wrong\r\n")
        second = _reject_probe(base_url, "/mcp", extra_headers="Authorization: Bearer wrong\r\n")

    assert "401" in first, {"got": first}
    assert "401" in second, {"got": second, "hint": "the server did not recover for the next call"}


def _write_error_functions() -> dict[str, ast.FunctionDef]:
    """Every ``_write_error`` defined by a reference HTTP module, by module path."""

    found: dict[str, ast.FunctionDef] = {}
    for relative in REFERENCE_HTTP_MODULES:
        tree = ast.parse((PACKAGE / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_write_error":
                assert relative not in found, {"two _write_error definitions in": relative}
                found[relative] = node
    return found


def test_every_reference_server_drains_before_it_writes_an_error() -> None:
    """The rule, bound on all four handlers rather than on the one whose test caught it.

    ``_write_exception`` funnels into ``_write_error`` on every server, so this is the single
    chokepoint an early rejection passes through — and a rule proved on one of four parallel
    handlers and left unbound on the other three is the defect shape this repo keeps producing.
    Derived from the modules, so a fifth reference server has to answer here too.
    """

    functions = _write_error_functions()
    assert set(functions) == set(REFERENCE_HTTP_MODULES), {
        "no _write_error found in": sorted(set(REFERENCE_HTTP_MODULES) - set(functions)),
        "hint": "a reference server that writes errors some other way needs its own drain",
    }
    undrained = sorted(
        relative
        for relative, node in functions.items()
        if not any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "drain_request_body"
            for call in ast.walk(node)
        )
    )
    assert undrained == [], {
        "writes_an_error_without_draining_the_request_body": undrained,
        "hint": "call drain_request_body(self) before writing the status, or this server's "
        "rejections are lost to a TCP reset whenever the body arrives after the headers",
    }
