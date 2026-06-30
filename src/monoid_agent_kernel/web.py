from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from monoid_agent_kernel.errors import NativeAgentError

DEFAULT_WEB_GATEWAY_TOKEN_ENV = "NAR_WEB_GATEWAY_TOKEN"


class WebGatewayError(NativeAgentError):
    error_code = "web_gateway_error"

    def __init__(self, message: str, *, error_code: str | None = None, http_status: int | None = None) -> None:
        super().__init__(message, error_code=error_code)
        self.http_status = http_status


@dataclass
class WebGatewayClient:
    gateway_url: str
    token: str | None = None
    token_env: str = DEFAULT_WEB_GATEWAY_TOKEN_ENV
    token_file: Path | None = None

    def search(self, payload: dict[str, Any], *, token: str | None = None) -> dict[str, Any]:
        return self._post("/internal/web/search", payload, token=token)

    def fetch(self, payload: dict[str, Any], *, token: str | None = None) -> dict[str, Any]:
        return self._post("/internal/web/fetch", payload, token=token)

    def context(self, payload: dict[str, Any], *, token: str | None = None) -> dict[str, Any]:
        return self._post("/internal/web/context", payload, token=token)

    def _post(self, path: str, payload: dict[str, Any], *, token: str | None = None) -> dict[str, Any]:
        url = f"{self.gateway_url.rstrip('/')}{path}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        timeout_s = payload.get("timeout_s") or 30
        attempts = 3
        last_error: WebGatewayError | None = None
        for attempt in range(attempts):
            request = Request(url, data=body, headers=self._headers(token), method="POST")
            try:
                with urlopen(request, timeout=timeout_s) as response:
                    return _decode_json(response.read())
            except HTTPError as exc:
                raise _error_from_http(exc) from exc  # a real HTTP response — never retry
            except OSError as exc:
                # Transient connection-level failure (reset / aborted / refused / timeout;
                # URLError and TimeoutError are OSError subclasses) — retry with a short
                # backoff before surfacing. The final error preserves the original code/message.
                if isinstance(exc, TimeoutError):
                    last_error = WebGatewayError(
                        "web gateway request timed out", error_code="web_gateway_timeout"
                    )
                else:
                    reason = getattr(exc, "reason", exc)
                    last_error = WebGatewayError(f"web gateway request failed: {reason}")
                if attempt < attempts - 1:
                    time.sleep(0.05 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def _headers(self, token: str | None = None) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "monoid-agent-kernel/0.13",
        }
        # A per-call capability lease handle (when web tools are routed through the capability gate)
        # overrides the static run-start credential; otherwise fall back to the configured token.
        resolved = token if token is not None else self._resolve_token()
        if resolved:
            headers["Authorization"] = f"Bearer {resolved}"
        return headers

    def _resolve_token(self) -> str | None:
        if self.token is not None:
            return self.token
        if self.token_file is not None:
            return self.token_file.read_text(encoding="utf-8").strip()
        return os.environ.get(self.token_env)


def public_query_preview(query: str) -> dict[str, Any]:
    return {
        "redacted": True,
        "type": "str",
        "bytes": len(query.encode("utf-8")),
        "sha256_prefix": hashlib.sha256(query.encode("utf-8")).hexdigest()[:12],
    }


def public_url_preview(url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    return {
        "redacted": True,
        "scheme": parsed.scheme,
        "domain": domain_from_url(url),
        "bytes": len(url.encode("utf-8")),
    }


def domain_from_url(url: str) -> str:
    parsed = urlparse(url)
    return (parsed.hostname or "").lower()


def domain_allowed(
    domain: str,
    *,
    allowed_domains: tuple[str, ...],
    blocked_domains: tuple[str, ...],
) -> bool:
    normalized = domain.lower().strip(".")
    if not normalized:
        return False
    if any(domain_matches(normalized, pattern) for pattern in blocked_domains):
        return False
    if allowed_domains and not any(domain_matches(normalized, pattern) for pattern in allowed_domains):
        return False
    return True


def domain_matches(domain: str, pattern: str) -> bool:
    normalized_pattern = pattern.lower().strip()
    if normalized_pattern == "*":
        return True
    if normalized_pattern.startswith("*."):
        suffix = normalized_pattern[2:]
        return domain == suffix or domain.endswith(f".{suffix}")
    return domain == normalized_pattern


def _decode_json(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise WebGatewayError("web gateway returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise WebGatewayError("web gateway response must be an object")
    if "error" in payload:
        raise WebGatewayError(
            str(payload["error"]),
            error_code=str(payload.get("error_code") or "web_gateway_error"),
            http_status=int(payload["http_status"]) if payload.get("http_status") is not None else None,
        )
    return payload


def _error_from_http(exc: HTTPError) -> WebGatewayError:
    detail = exc.read().decode("utf-8", errors="replace")
    try:
        payload = json.loads(detail)
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return WebGatewayError(
        str(payload.get("error") or detail or f"HTTP {exc.code}"),
        error_code=str(payload.get("error_code") or "web_gateway_error"),
        http_status=int(exc.code),
    )
