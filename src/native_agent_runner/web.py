from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from native_agent_runner._policy_util import dedupe, str_tuple
from native_agent_runner.errors import NativeAgentError

DEFAULT_WEB_GATEWAY_TOKEN_ENV = "NAR_WEB_GATEWAY_TOKEN"


class WebGatewayError(NativeAgentError):
    error_code = "web_gateway_error"

    def __init__(self, message: str, *, error_code: str | None = None, http_status: int | None = None) -> None:
        super().__init__(message, error_code=error_code)
        self.http_status = http_status


@dataclass(frozen=True)
class WebPolicy:
    enabled: bool = False
    search_enabled: bool = True
    fetch_enabled: bool = True
    context_enabled: bool = False
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    max_search_calls: int = 20
    max_fetch_calls: int = 50
    max_context_calls: int = 10
    default_max_results: int = 5
    max_results: int = 10
    default_max_context_tokens: int = 8_192
    max_context_tokens: int = 32_768
    default_max_context_urls: int = 8
    max_context_urls: int = 20
    default_max_context_snippets: int = 50
    max_context_snippets: int = 256
    default_timeout_s: int = 30
    max_timeout_s: int = 60
    default_max_response_bytes: int = 100_000
    max_response_bytes: int = 1_000_000

    @classmethod
    def from_json(cls, payload: dict[str, Any] | None) -> WebPolicy:
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("web_policy must be an object")
        max_results = int(payload.get("max_results", 10))
        max_context_tokens = int(payload.get("max_context_tokens", 32_768))
        max_context_urls = int(payload.get("max_context_urls", 20))
        max_context_snippets = int(payload.get("max_context_snippets", 256))
        max_timeout_s = int(payload.get("max_timeout_s", 60))
        max_response_bytes = int(payload.get("max_response_bytes", 1_000_000))
        policy = cls(
            enabled=bool(payload.get("enabled", False)),
            search_enabled=bool(payload.get("search_enabled", True)),
            fetch_enabled=bool(payload.get("fetch_enabled", True)),
            context_enabled=bool(payload.get("context_enabled", False)),
            allowed_domains=str_tuple(
                payload.get("allowed_domains") or (),
                type_error="web policy domain lists must be arrays",
                normalize=True,
            ),
            blocked_domains=str_tuple(
                payload.get("blocked_domains") or (),
                type_error="web policy domain lists must be arrays",
                normalize=True,
            ),
            max_search_calls=int(payload.get("max_search_calls", 20)),
            max_fetch_calls=int(payload.get("max_fetch_calls", 50)),
            max_context_calls=int(payload.get("max_context_calls", 10)),
            default_max_results=min(int(payload.get("default_max_results", 5)), max_results),
            max_results=max_results,
            default_max_context_tokens=min(
                int(payload.get("default_max_context_tokens", 8_192)),
                max_context_tokens,
            ),
            max_context_tokens=max_context_tokens,
            default_max_context_urls=min(
                int(payload.get("default_max_context_urls", 8)),
                max_context_urls,
            ),
            max_context_urls=max_context_urls,
            default_max_context_snippets=min(
                int(payload.get("default_max_context_snippets", 50)),
                max_context_snippets,
            ),
            max_context_snippets=max_context_snippets,
            default_timeout_s=min(int(payload.get("default_timeout_s", 30)), max_timeout_s),
            max_timeout_s=max_timeout_s,
            default_max_response_bytes=min(
                int(payload.get("default_max_response_bytes", 100_000)),
                max_response_bytes,
            ),
            max_response_bytes=max_response_bytes,
        )
        return policy.validated()

    def merged(
        self,
        *,
        enabled: bool | None = None,
        allowed_domains: tuple[str, ...] = (),
        blocked_domains: tuple[str, ...] = (),
        max_search_calls: int | None = None,
        max_fetch_calls: int | None = None,
        max_context_calls: int | None = None,
        max_results: int | None = None,
        max_context_tokens: int | None = None,
        max_context_urls: int | None = None,
        max_context_snippets: int | None = None,
        timeout_s: int | None = None,
        max_response_bytes: int | None = None,
        context_enabled: bool | None = None,
    ) -> WebPolicy:
        new_max_results = self.max_results if max_results is None else max_results
        new_max_context_tokens = self.max_context_tokens if max_context_tokens is None else max_context_tokens
        new_max_context_urls = self.max_context_urls if max_context_urls is None else max_context_urls
        new_max_context_snippets = self.max_context_snippets if max_context_snippets is None else max_context_snippets
        return WebPolicy(
            enabled=self.enabled if enabled is None else enabled,
            search_enabled=self.search_enabled,
            fetch_enabled=self.fetch_enabled,
            context_enabled=self.context_enabled if context_enabled is None else context_enabled,
            allowed_domains=dedupe((*self.allowed_domains, *allowed_domains)),
            blocked_domains=dedupe((*self.blocked_domains, *blocked_domains)),
            max_search_calls=self.max_search_calls if max_search_calls is None else max_search_calls,
            max_fetch_calls=self.max_fetch_calls if max_fetch_calls is None else max_fetch_calls,
            max_context_calls=self.max_context_calls if max_context_calls is None else max_context_calls,
            default_max_results=min(self.default_max_results, new_max_results),
            max_results=new_max_results,
            default_max_context_tokens=min(self.default_max_context_tokens, new_max_context_tokens),
            max_context_tokens=new_max_context_tokens,
            default_max_context_urls=min(self.default_max_context_urls, new_max_context_urls),
            max_context_urls=new_max_context_urls,
            default_max_context_snippets=min(self.default_max_context_snippets, new_max_context_snippets),
            max_context_snippets=new_max_context_snippets,
            default_timeout_s=self.default_timeout_s if timeout_s is None else timeout_s,
            max_timeout_s=self.max_timeout_s,
            default_max_response_bytes=(
                self.default_max_response_bytes if max_response_bytes is None else max_response_bytes
            ),
            max_response_bytes=self.max_response_bytes,
        ).validated()

    def validated(self) -> WebPolicy:
        if self.max_search_calls < 0 or self.max_fetch_calls < 0 or self.max_context_calls < 0:
            raise ValueError("web call limits must be non-negative")
        if self.default_max_results < 1 or self.max_results < 1:
            raise ValueError("web result limits must be positive")
        if self.default_max_context_tokens < 1 or self.max_context_tokens < 1:
            raise ValueError("web context token limits must be positive")
        if self.default_max_context_urls < 1 or self.max_context_urls < 1:
            raise ValueError("web context URL limits must be positive")
        if self.default_max_context_snippets < 1 or self.max_context_snippets < 1:
            raise ValueError("web context snippet limits must be positive")
        if self.default_timeout_s < 1 or self.max_timeout_s < 1:
            raise ValueError("web timeouts must be positive")
        if self.default_max_response_bytes < 1 or self.max_response_bytes < 1:
            raise ValueError("web response byte limits must be positive")
        if self.default_max_results > self.max_results:
            raise ValueError("default_max_results cannot exceed max_results")
        if self.default_max_context_tokens > self.max_context_tokens:
            raise ValueError("default_max_context_tokens cannot exceed max_context_tokens")
        if self.default_max_context_urls > self.max_context_urls:
            raise ValueError("default_max_context_urls cannot exceed max_context_urls")
        if self.default_max_context_snippets > self.max_context_snippets:
            raise ValueError("default_max_context_snippets cannot exceed max_context_snippets")
        if self.default_timeout_s > self.max_timeout_s:
            raise ValueError("default_timeout_s cannot exceed max_timeout_s")
        if self.default_max_response_bytes > self.max_response_bytes:
            raise ValueError("default_max_response_bytes cannot exceed max_response_bytes")
        return self

    def to_json(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "search_enabled": self.search_enabled,
            "fetch_enabled": self.fetch_enabled,
            "context_enabled": self.context_enabled,
            "allowed_domains": list(self.allowed_domains),
            "blocked_domains": list(self.blocked_domains),
            "max_search_calls": self.max_search_calls,
            "max_fetch_calls": self.max_fetch_calls,
            "max_context_calls": self.max_context_calls,
            "default_max_results": self.default_max_results,
            "max_results": self.max_results,
            "default_max_context_tokens": self.default_max_context_tokens,
            "max_context_tokens": self.max_context_tokens,
            "default_max_context_urls": self.default_max_context_urls,
            "max_context_urls": self.max_context_urls,
            "default_max_context_snippets": self.default_max_context_snippets,
            "max_context_snippets": self.max_context_snippets,
            "default_timeout_s": self.default_timeout_s,
            "max_timeout_s": self.max_timeout_s,
            "default_max_response_bytes": self.default_max_response_bytes,
            "max_response_bytes": self.max_response_bytes,
        }

    def to_manifest(self) -> dict[str, Any]:
        return self.to_json()

    def effective_max_results(self, requested: Any) -> int:
        return min(_optional_positive_int(requested, self.default_max_results), self.max_results)

    def effective_timeout_s(self, requested: Any) -> int:
        return min(_optional_positive_int(requested, self.default_timeout_s), self.max_timeout_s)

    def effective_max_response_bytes(self, requested: Any) -> int:
        return min(_optional_positive_int(requested, self.default_max_response_bytes), self.max_response_bytes)

    def effective_max_context_tokens(self, requested: Any) -> int:
        return min(_optional_positive_int(requested, self.default_max_context_tokens), self.max_context_tokens)

    def effective_max_context_urls(self, requested: Any) -> int:
        return min(_optional_positive_int(requested, self.default_max_context_urls), self.max_context_urls)

    def effective_max_context_snippets(self, requested: Any) -> int:
        return min(_optional_positive_int(requested, self.default_max_context_snippets), self.max_context_snippets)


@dataclass
class WebGatewayClient:
    gateway_url: str
    token: str | None = None
    token_env: str = DEFAULT_WEB_GATEWAY_TOKEN_ENV
    token_file: Path | None = None

    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/internal/web/search", payload)

    def fetch(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/internal/web/fetch", payload)

    def context(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/internal/web/context", payload)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.gateway_url.rstrip('/')}{path}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=payload.get("timeout_s") or 30) as response:
                return _decode_json(response.read())
        except HTTPError as exc:
            raise _error_from_http(exc) from exc
        except URLError as exc:
            raise WebGatewayError(f"web gateway request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise WebGatewayError("web gateway request timed out", error_code="web_gateway_timeout") from exc

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "native-agent-runner/0.11",
        }
        token = self._resolve_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
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


def _optional_positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    result = int(value)
    if result < 1:
        raise ValueError("web limits must be positive")
    return result


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
