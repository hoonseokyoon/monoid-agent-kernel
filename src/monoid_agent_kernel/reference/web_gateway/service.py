from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from monoid_agent_kernel.reference._shared.tokens import TokenClaims, TokenError, TokenManager
from monoid_agent_kernel.errors import PermissionDenied
from monoid_agent_kernel.web import (
    WebGatewayError,
    domain_allowed,
    domain_from_url,
)


class WebProvider(Protocol):
    def search(self, query: str, *, max_results: int) -> list[dict[str, Any]]:
        ...

    def fetch(self, url: str, *, format: str) -> dict[str, Any]:
        ...

    def context(
        self,
        query: str,
        *,
        max_tokens: int,
        max_urls: int,
        max_snippets: int,
        locale: str | None,
        freshness: str | None,
        allowed_domains: tuple[str, ...],
        blocked_domains: tuple[str, ...],
    ) -> dict[str, Any]:
        ...


DEFAULT_FAKE_CORPUS: tuple[dict[str, str], ...] = (
    {
        "url": "https://docs.example.test/native-agent-runner/web",
        "title": "Monoid Agent Kernel Web Tools",
        "content": (
            "Monoid Agent Kernel exposes web.search and web.fetch through a WebGateway. "
            "The runner never receives provider API keys."
        ),
    },
    {
        "url": "https://docs.example.test/native-agent-runner/policy",
        "title": "Web Policy and Tenant Usage",
        "content": (
            "Tool bindings carry allowed domains, blocked domains, call limits, result limits, "
            "timeouts, and response byte caps. Tenant usage is counted by the gateway."
        ),
    },
    {
        "url": "https://blog.example.test/agent-observability",
        "title": "Agent Observability",
        "content": "Typed events expose tool calls, web search, web fetch, metrics, and proposal updates.",
    },
)


@dataclass
class FakeWebProvider:
    corpus: tuple[dict[str, str], ...] = DEFAULT_FAKE_CORPUS

    def search(self, query: str, *, max_results: int) -> list[dict[str, Any]]:
        terms = [term for term in query.lower().split() if term]
        scored: list[tuple[int, dict[str, str]]] = []
        for item in self.corpus:
            haystack = f"{item['title']} {item['content']} {item['url']}".lower()
            score = sum(haystack.count(term) for term in terms) if terms else 1
            if score:
                scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["url"]))
        return [
            {
                "title": item["title"],
                "url": item["url"],
                "domain": domain_from_url(item["url"]),
                "snippet": item["content"][:180],
                "source": "fake",
            }
            for _score, item in scored[:max_results]
        ]

    def fetch(self, url: str, *, format: str) -> dict[str, Any]:
        del format
        for item in self.corpus:
            if item["url"] == url:
                return {
                    "title": item["title"],
                    "url": item["url"],
                    "final_url": item["url"],
                    "domain": domain_from_url(item["url"]),
                    "content": item["content"],
                    "source": "fake",
                }
        raise WebGatewayError(f"web document not found: {url}", error_code="web_not_found")

    def context(
        self,
        query: str,
        *,
        max_tokens: int,
        max_urls: int,
        max_snippets: int,
        locale: str | None,
        freshness: str | None,
        allowed_domains: tuple[str, ...],
        blocked_domains: tuple[str, ...],
    ) -> dict[str, Any]:
        del locale, freshness
        results = [
            result
            for result in self.search(query, max_results=max_urls)
            if domain_allowed(
                str(result.get("domain") or domain_from_url(str(result.get("url") or ""))),
                allowed_domains=allowed_domains,
                blocked_domains=blocked_domains,
            )
        ][:max_urls]
        chunks: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        budget_chars = max_tokens * 4
        used_chars = 0
        for result in results:
            url = str(result.get("url") or "")
            page = self.fetch(url, format="markdown")
            content = str(page.get("content") or "")
            if not content:
                continue
            sources.append(
                {
                    "url": url,
                    "title": str(page.get("title") or result.get("title") or ""),
                    "domain": domain_from_url(url),
                    "source": "fake",
                }
            )
            text = content[: max(0, budget_chars - used_chars)]
            if not text:
                break
            chunks.append(
                {
                    "url": url,
                    "title": str(page.get("title") or result.get("title") or ""),
                    "domain": domain_from_url(url),
                    "text": text,
                }
            )
            used_chars += len(text)
            if len(chunks) >= max_snippets or used_chars >= budget_chars:
                break
        context = "\n\n".join(
            f"[{index}] {chunk.get('title')}\n{chunk.get('url')}\n{chunk.get('text')}"
            for index, chunk in enumerate(chunks, start=1)
        )
        return {
            "query": query,
            "context": context,
            "sources": sources,
            "chunks": chunks,
            "source": "fake",
        }


@dataclass
class WebGatewayUsage:
    tenant_id: str
    search_calls: int = 0
    fetch_calls: int = 0
    context_calls: int = 0
    failed_calls: int = 0
    result_count: int = 0
    context_source_count: int = 0
    bytes_returned: int = 0
    context_bytes_returned: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "search_calls": self.search_calls,
            "fetch_calls": self.fetch_calls,
            "context_calls": self.context_calls,
            "failed_calls": self.failed_calls,
            "result_count": self.result_count,
            "context_source_count": self.context_source_count,
            "bytes_returned": self.bytes_returned,
            "context_bytes_returned": self.context_bytes_returned,
        }


@dataclass
class _RunWebCounts:
    search_calls: int = 0
    fetch_calls: int = 0
    context_calls: int = 0
    binding_calls: dict[str, int] = field(default_factory=dict)


@dataclass
class WebGatewayBackend:
    token_manager: TokenManager
    provider: WebProvider = field(default_factory=FakeWebProvider)
    _usage: dict[str, WebGatewayUsage] = field(default_factory=dict, init=False, repr=False)
    _run_counts: dict[str, _RunWebCounts] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def handle_search(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        claims = self._authorize(token)
        self._check_binding_limit(claims, payload, error_code="web_search_limit_exceeded")
        query = str(payload.get("query") or "")
        if not query.strip():
            raise ValueError("query is required")
        effective_max_results = max(1, int(payload.get("max_results") or 5))
        request_allowed = _domain_tuple(payload.get("allowed_domains") or ())
        request_blocked = _domain_tuple(payload.get("blocked_domains") or ())
        raw_results = self.provider.search(query, max_results=effective_max_results)
        results = [
            result
            for result in raw_results
            if _result_allowed(result, request_allowed=request_allowed, request_blocked=request_blocked)
        ][:effective_max_results]
        with self._lock:
            self._counts(claims.run_id).search_calls += 1
            self._increment_binding_count(claims, payload)
            usage = self._tenant_usage(claims.tenant_id)
            usage.search_calls += 1
            usage.result_count += len(results)
        return {
            "protocol": "native-agent-runner.web-search-result.v1",
            "query": query,
            "results": results,
            "result_count": len(results),
            "requested_max_results": payload.get("max_results"),
            "effective_max_results": effective_max_results,
        }

    def handle_fetch(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        claims = self._authorize(token)
        self._check_binding_limit(claims, payload, error_code="web_fetch_limit_exceeded")
        url = str(payload.get("url") or "")
        if not url.strip():
            raise ValueError("url is required")
        domain = domain_from_url(url)
        request_allowed = _domain_tuple(payload.get("allowed_domains") or ())
        request_blocked = _domain_tuple(payload.get("blocked_domains") or ())
        if not domain_allowed(domain, allowed_domains=request_allowed, blocked_domains=request_blocked):
            raise WebGatewayError(
                f"domain is not allowed by binding constraints: {domain}",
                error_code="web_binding_denied",
            )
        output_format = str(payload.get("format") or "text")
        effective_timeout_s = max(1, int(payload.get("timeout_s") or 30))
        effective_max_bytes = max(1, int(payload.get("max_bytes") or 100_000))
        fetched = self.provider.fetch(url, format=output_format)
        content = str(fetched.get("content") or "")
        encoded = content.encode("utf-8")
        truncated = len(encoded) > effective_max_bytes
        if truncated:
            encoded = encoded[:effective_max_bytes]
            content = encoded.decode("utf-8", errors="ignore")
        content_bytes = len(encoded)
        response = {
            "protocol": "native-agent-runner.web-fetch-result.v1",
            "url": url,
            "final_url": str(fetched.get("final_url") or url),
            "domain": domain_from_url(str(fetched.get("final_url") or url)),
            "title": str(fetched.get("title") or ""),
            "format": output_format,
            "content": content,
            "content_bytes": content_bytes,
            "original_bytes": len(str(fetched.get("content") or "").encode("utf-8")),
            "truncated": truncated,
            "requested_timeout_s": payload.get("timeout_s"),
            "effective_timeout_s": effective_timeout_s,
            "requested_max_bytes": payload.get("max_bytes"),
            "effective_max_bytes": effective_max_bytes,
            "source": str(fetched.get("source") or ""),
        }
        with self._lock:
            self._counts(claims.run_id).fetch_calls += 1
            self._increment_binding_count(claims, payload)
            usage = self._tenant_usage(claims.tenant_id)
            usage.fetch_calls += 1
            usage.bytes_returned += content_bytes
        return response

    def handle_context(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        claims = self._authorize(token)
        self._check_binding_limit(claims, payload, error_code="web_context_limit_exceeded")
        query = str(payload.get("query") or "")
        if not query.strip():
            raise ValueError("query is required")
        request_allowed = _domain_tuple(payload.get("allowed_domains") or ())
        request_blocked = _domain_tuple(payload.get("blocked_domains") or ())
        effective_max_tokens = max(1, int(payload.get("max_tokens") or 8_192))
        effective_max_urls = max(1, int(payload.get("max_urls") or 8))
        effective_max_snippets = max(1, int(payload.get("max_snippets") or 50))
        provider_result = self.provider.context(
            query,
            max_tokens=effective_max_tokens,
            max_urls=effective_max_urls,
            max_snippets=effective_max_snippets,
            locale=_optional_string(payload.get("locale")),
            freshness=_freshness_from_payload(payload),
            allowed_domains=request_allowed,
            blocked_domains=request_blocked,
        )
        filtered = _filter_context_result(provider_result, request_allowed, request_blocked)
        context = str(filtered.get("context") or "")
        encoded = context.encode("utf-8")
        context_bytes = len(encoded)
        sources = filtered.get("sources") if isinstance(filtered.get("sources"), list) else []
        chunks = filtered.get("chunks") if isinstance(filtered.get("chunks"), list) else []
        response = {
            "protocol": "native-agent-runner.web-context-result.v1",
            "query": query,
            "context": context,
            "sources": sources,
            "chunks": chunks,
            "source_count": len(sources),
            "context_bytes": context_bytes,
            "estimated_tokens": max(1, context_bytes // 4) if context_bytes else 0,
            "requested_max_tokens": payload.get("max_tokens"),
            "effective_max_tokens": effective_max_tokens,
            "requested_max_urls": payload.get("max_urls"),
            "effective_max_urls": effective_max_urls,
            "requested_max_snippets": payload.get("max_snippets"),
            "effective_max_snippets": effective_max_snippets,
            "source": str(filtered.get("source") or ""),
        }
        with self._lock:
            self._counts(claims.run_id).context_calls += 1
            self._increment_binding_count(claims, payload)
            usage = self._tenant_usage(claims.tenant_id)
            usage.context_calls += 1
            usage.context_source_count += len(sources)
            usage.context_bytes_returned += context_bytes
        return response

    def tenant_usage(self, tenant_id: str) -> dict[str, Any]:
        with self._lock:
            return self._tenant_usage(tenant_id).to_json()

    def _authorize(self, token: str) -> TokenClaims:
        try:
            return self.token_manager.verify(token, kind="web_gateway", audience="csp.web-gateway")
        except TokenError as exc:
            raise PermissionDenied(str(exc)) from exc

    def _check_binding_limit(self, claims: TokenClaims, payload: dict[str, Any], *, error_code: str) -> None:
        binding_id = str(payload.get("binding_id") or "").strip()
        max_calls = int(payload.get("max_calls") or 0)
        if not binding_id or max_calls <= 0:
            return
        with self._lock:
            if self._counts(claims.run_id).binding_calls.get(binding_id, 0) >= max_calls:
                self._tenant_usage(claims.tenant_id).failed_calls += 1
                raise WebGatewayError("web binding call limit exceeded", error_code=error_code)

    def _increment_binding_count(self, claims: TokenClaims, payload: dict[str, Any]) -> None:
        binding_id = str(payload.get("binding_id") or "").strip()
        if not binding_id:
            return
        counts = self._counts(claims.run_id)
        counts.binding_calls[binding_id] = counts.binding_calls.get(binding_id, 0) + 1

    def _counts(self, run_id: str) -> _RunWebCounts:
        return self._run_counts.setdefault(run_id, _RunWebCounts())

    def _tenant_usage(self, tenant_id: str) -> WebGatewayUsage:
        return self._usage.setdefault(tenant_id, WebGatewayUsage(tenant_id))


def _result_allowed(
    result: dict[str, Any],
    *,
    request_allowed: tuple[str, ...],
    request_blocked: tuple[str, ...],
) -> bool:
    return domain_allowed(
        str(result.get("domain") or domain_from_url(str(result.get("url") or ""))),
        allowed_domains=request_allowed,
        blocked_domains=request_blocked,
    )


def _domain_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("domain filters must be arrays")
    return tuple(str(item).strip().lower() for item in value if str(item).strip())


def _filter_context_result(
    result: dict[str, Any],
    request_allowed: tuple[str, ...],
    request_blocked: tuple[str, ...],
) -> dict[str, Any]:
    sources = result.get("sources") if isinstance(result.get("sources"), list) else []
    chunks = result.get("chunks") if isinstance(result.get("chunks"), list) else []
    allowed_sources = [
        source
        for source in sources
        if isinstance(source, dict)
        and domain_allowed(
            str(source.get("domain") or domain_from_url(str(source.get("url") or ""))),
            allowed_domains=request_allowed,
            blocked_domains=request_blocked,
        )
    ]
    allowed_domains = {
        str(source.get("domain") or domain_from_url(str(source.get("url") or "")))
        for source in allowed_sources
        if isinstance(source, dict)
    }
    allowed_chunks = [
        chunk
        for chunk in chunks
        if isinstance(chunk, dict)
        and (
            not str(chunk.get("domain") or domain_from_url(str(chunk.get("url") or "")))
            or str(chunk.get("domain") or domain_from_url(str(chunk.get("url") or ""))) in allowed_domains
            or domain_allowed(
                str(chunk.get("domain") or domain_from_url(str(chunk.get("url") or ""))),
                allowed_domains=request_allowed,
                blocked_domains=request_blocked,
            )
        )
    ]
    context = str(result.get("context") or "")
    filters_active = bool(request_allowed or request_blocked)
    if filters_active and (len(allowed_sources) != len(sources) or len(allowed_chunks) != len(chunks)):
        context = "\n\n".join(str(chunk.get("text") or "") for chunk in allowed_chunks if isinstance(chunk, dict))
    if filters_active:
        sources = allowed_sources
        chunks = allowed_chunks
    return {**result, "context": context, "sources": sources, "chunks": chunks}


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _freshness_from_payload(payload: dict[str, Any]) -> str | None:
    freshness = _optional_string(payload.get("freshness"))
    if freshness:
        return freshness
    recency_days = payload.get("recency_days")
    if recency_days is None:
        return None
    try:
        days = int(recency_days)
    except (TypeError, ValueError):
        return None
    if days <= 1:
        return "pd"
    if days <= 7:
        return "pw"
    if days <= 31:
        return "pm"
    if days <= 365:
        return "py"
    return None
