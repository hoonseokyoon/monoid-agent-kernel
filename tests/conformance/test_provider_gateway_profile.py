from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler
from typing import Any

import pytest

from monoid_agent_kernel.conformance.profiles.provider_gateway import assert_provider_gateway_profile
from monoid_agent_kernel.reference._shared.http_util import HardenedThreadingHTTPServer
from monoid_agent_kernel.reference.conformance import ReferenceGatewayHarness
from monoid_agent_kernel.reference.web_gateway.providers import HttpFetchProvider
from monoid_agent_kernel.reference.web_gateway.service import FakeWebProvider
from monoid_agent_kernel.web import WebGatewayError


def test_reference_web_gateway_satisfies_provider_gateway_profile() -> None:
    assert_provider_gateway_profile(ReferenceGatewayHarness())


def test_reference_web_gateway_rejects_redirect_final_domain() -> None:
    class RedirectingProvider(FakeWebProvider):
        def fetch(
            self,
            url: str,
            *,
            format: str,
            allowed_domains: tuple[str, ...] = (),
            blocked_domains: tuple[str, ...] = (),
            timeout_s: int | None = None,
            max_bytes: int | None = None,
        ) -> dict[str, Any]:
            del url, format, allowed_domains, blocked_domains, timeout_s, max_bytes
            return {
                "title": "Redirected",
                "final_url": "https://blog.example.test/redirected",
                "content": "redirected content",
                "source": "test",
            }

    harness = ReferenceGatewayHarness(provider=RedirectingProvider())

    with pytest.raises(WebGatewayError, match="final domain is not allowed"):
        harness.call_gateway(
            "web.fetch",
            {"url": "https://docs.example.test/open-redirect"},
            signed_scope={"binding_id": "fetch_docs", "allowed_domains": ["docs.example.test"]},
        )


def test_http_fetch_provider_trims_to_requested_max_bytes() -> None:
    upstream = _LargeTextServer()
    upstream.start()
    try:
        fetch_provider = HttpFetchProvider(timeout_s=5, max_raw_bytes=20_000)
        fetched = fetch_provider.fetch(
            f"{upstream.base_url}/large-text",
            format="text",
            allowed_domains=("127.0.0.1",),
            max_bytes=5,
        )

        assert fetched["content"] == "abcde"
        assert len(fetched["content"].encode("utf-8")) == 5
    finally:
        upstream.stop()


class _LargeTextServer:
    def start(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return

            def do_GET(self) -> None:
                body = b"abcdef"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = HardenedThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.port = int(self.server.server_address[1])
        self.base_url = f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
