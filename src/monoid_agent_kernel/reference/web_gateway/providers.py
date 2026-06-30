from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from monoid_agent_kernel.web import WebGatewayError, domain_from_url

DEFAULT_BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_BRAVE_LLM_CONTEXT_ENDPOINT = "https://api.search.brave.com/res/v1/llm/context"
DEFAULT_HTTP_USER_AGENT = "monoid-agent-kernel-webgateway/0.13"


class SearchProvider(Protocol):
    provider_name: str

    def search(self, query: str, *, max_results: int) -> list[dict[str, Any]]:
        ...


class FetchProvider(Protocol):
    provider_name: str

    def fetch(self, url: str, *, format: str) -> dict[str, Any]:
        ...


class ContextProvider(Protocol):
    provider_name: str

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


@dataclass(frozen=True)
class CompositeWebProvider:
    search_provider: SearchProvider
    fetch_provider: FetchProvider
    context_provider: ContextProvider | None = None

    def search(self, query: str, *, max_results: int) -> list[dict[str, Any]]:
        return self.search_provider.search(query, max_results=max_results)

    def fetch(self, url: str, *, format: str) -> dict[str, Any]:
        return self.fetch_provider.fetch(url, format=format)

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
        if self.context_provider is None:
            raise WebGatewayError("web context provider is not configured", error_code="web_disabled")
        provider = self.context_provider
        return provider.context(
            query,
            max_tokens=max_tokens,
            max_urls=max_urls,
            max_snippets=max_snippets,
            locale=locale,
            freshness=freshness,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )


@dataclass(frozen=True)
class BraveSearchProvider:
    api_key: str
    endpoint: str = DEFAULT_BRAVE_SEARCH_ENDPOINT
    country: str = "US"
    search_lang: str = "en"
    timeout_s: int = 10
    provider_name: str = "brave"

    @classmethod
    def from_env(
        cls,
        *,
        api_key_env: str = "BRAVE_SEARCH_API_KEY",
        endpoint: str = DEFAULT_BRAVE_SEARCH_ENDPOINT,
        country: str = "US",
        search_lang: str = "en",
        timeout_s: int = 10,
    ) -> BraveSearchProvider:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} is required for Brave web search")
        return cls(
            api_key=api_key,
            endpoint=endpoint,
            country=country,
            search_lang=search_lang,
            timeout_s=timeout_s,
        )

    def search(self, query: str, *, max_results: int) -> list[dict[str, Any]]:
        params = {
            "q": query,
            "count": max(1, min(int(max_results), 20)),
            "country": self.country,
            "search_lang": self.search_lang,
        }
        url = f"{self.endpoint}?{urlencode(params)}"
        payload = _request_json(
            url,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            },
            timeout_s=self.timeout_s,
        )
        return _normalize_brave_results(payload, max_results=max_results, source=self.provider_name)


@dataclass(frozen=True)
class BraveLlmContextProvider:
    api_key: str
    endpoint: str = DEFAULT_BRAVE_LLM_CONTEXT_ENDPOINT
    country: str = "US"
    search_lang: str = "en"
    timeout_s: int = 20
    provider_name: str = "brave-llm-context"

    @classmethod
    def from_env(
        cls,
        *,
        api_key_env: str = "BRAVE_SEARCH_API_KEY",
        endpoint: str = DEFAULT_BRAVE_LLM_CONTEXT_ENDPOINT,
        country: str = "US",
        search_lang: str = "en",
        timeout_s: int = 20,
    ) -> BraveLlmContextProvider:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} is required for Brave LLM context")
        return cls(
            api_key=api_key,
            endpoint=endpoint,
            country=country,
            search_lang=search_lang,
            timeout_s=timeout_s,
        )

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
        del allowed_domains, blocked_domains
        payload = {
            "q": query,
            "country": self.country,
            "search_lang": locale or self.search_lang,
            "count": max(1, min(int(max_urls), 50)),
            "maximum_number_of_urls": max(1, min(int(max_urls), 50)),
            "maximum_number_of_tokens": max(1024, min(int(max_tokens), 32_768)),
            "maximum_number_of_snippets": max(1, min(int(max_snippets), 256)),
        }
        if freshness:
            payload["freshness"] = freshness
        response = _request_json(
            self.endpoint,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-Subscription-Token": self.api_key,
            },
            timeout_s=self.timeout_s,
            method="POST",
            body=payload,
            error_prefix="context provider",
            error_code_prefix="web_context",
        )
        return _normalize_context_payload(response, query=query, source=self.provider_name)


@dataclass(frozen=True)
class HttpFetchProvider:
    timeout_s: int = 20
    max_raw_bytes: int = 2_000_000
    user_agent: str = DEFAULT_HTTP_USER_AGENT
    provider_name: str = "http"

    def fetch(self, url: str, *, format: str) -> dict[str, Any]:
        if domain_from_url(url) == "":
            raise WebGatewayError(f"invalid URL for fetch: {url}", error_code="web_bad_request")
        request = Request(
            url,
            headers={
                "Accept": "text/html, text/plain, application/xhtml+xml, */*;q=0.8",
                "User-Agent": self.user_agent,
            },
            method="GET",
        )
        # Retry transient connection-level failures (an HTTPError is a real response and
        # propagates immediately); the final error keeps the original message/code.
        attempts = 3
        final_url = url
        content_type = ""
        raw = b""
        for attempt in range(attempts):
            try:
                with urlopen(request, timeout=self.timeout_s) as response:
                    final_url = response.geturl()
                    content_type = response.headers.get("Content-Type", "")
                    raw = response.read(self.max_raw_bytes + 1)
                break
            except HTTPError as exc:
                raise WebGatewayError(
                    f"fetch failed with HTTP {exc.code}: {url}",
                    error_code="web_fetch_http_error",
                    http_status=int(exc.code),
                ) from exc
            except OSError as exc:  # URLError/TimeoutError + raw connection resets
                if attempt < attempts - 1:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                if isinstance(exc, TimeoutError):
                    raise WebGatewayError("fetch timed out", error_code="web_fetch_timeout") from exc
                reason = getattr(exc, "reason", exc)
                raise WebGatewayError(
                    f"fetch failed: {reason}", error_code="web_fetch_network_error"
                ) from exc

        if len(raw) > self.max_raw_bytes:
            raw = raw[: self.max_raw_bytes]
        text = _decode_body(raw, content_type)
        output_format = format if format in {"text", "markdown"} else "text"
        content = _html_to_text(text) if _looks_like_html(content_type, text) else text
        return {
            "title": _extract_title(text) if _looks_like_html(content_type, text) else "",
            "url": url,
            "final_url": final_url,
            "domain": domain_from_url(final_url),
            "content": content,
            "format": output_format,
            "content_type": content_type,
            "source": self.provider_name,
        }


@dataclass(frozen=True)
class ContextBuilder:
    max_chars_per_chunk: int = 2_000

    def build(
        self,
        query: str,
        fetched_pages: list[dict[str, Any]],
        *,
        max_tokens: int,
        max_snippets: int,
        source: str,
    ) -> dict[str, Any]:
        budget_chars = max(1, int(max_tokens)) * 4
        chunks: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        used_chars = 0
        for page in fetched_pages:
            url = str(page.get("final_url") or page.get("url") or "")
            title = str(page.get("title") or "")
            content = str(page.get("content") or "")
            if not url or not content:
                continue
            source_index = len(sources)
            sources.append(
                {
                    "url": url,
                    "title": title,
                    "domain": domain_from_url(url),
                    "source": str(page.get("source") or source),
                }
            )
            for chunk_text in _split_context_chunks(content, self.max_chars_per_chunk):
                if len(chunks) >= max_snippets or used_chars >= budget_chars:
                    break
                remaining = budget_chars - used_chars
                text = chunk_text[:remaining]
                if not text:
                    break
                chunks.append(
                    {
                        "url": url,
                        "title": title,
                        "domain": domain_from_url(url),
                        "text": text,
                        "source_index": source_index,
                    }
                )
                used_chars += len(text)
            if len(chunks) >= max_snippets or used_chars >= budget_chars:
                break
        context = _format_context(query, chunks)
        return {
            "query": query,
            "context": context,
            "sources": sources,
            "chunks": chunks,
            "source": source,
        }


@dataclass(frozen=True)
class SearchFetchContextProvider:
    search_provider: SearchProvider
    fetch_provider: FetchProvider
    builder: ContextBuilder = field(default_factory=ContextBuilder)
    provider_name: str = "search-fetch-context"

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
        results = self.search_provider.search(query, max_results=max_urls)
        filtered_results = [
            result
            for result in results
            if _domain_allowed_by_filters(str(result.get("domain") or domain_from_url(str(result.get("url") or ""))), allowed_domains, blocked_domains)
        ][:max_urls]
        pages: list[dict[str, Any]] = []
        for result in filtered_results:
            url = str(result.get("url") or "")
            if not url:
                continue
            try:
                pages.append(self.fetch_provider.fetch(url, format="markdown"))
            except WebGatewayError:
                continue
        return self.builder.build(
            query,
            pages,
            max_tokens=max_tokens,
            max_snippets=max_snippets,
            source=self.provider_name,
        )


def _urlopen_read_with_retry(
    request: Request,
    *,
    timeout_s: int,
    error_prefix: str,
    error_code_prefix: str,
) -> bytes:
    """Open ``request`` and read the body, retrying transient connection-level failures
    (reset / aborted / refused / timeout) with a short backoff. An ``HTTPError`` is a real
    response and propagates immediately; the final connection error keeps the original
    ``*_network_error`` / ``*_timeout`` code so callers and tests see unchanged behavior."""
    attempts = 3
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=timeout_s) as response:
                return response.read()
        except HTTPError as exc:
            raise WebGatewayError(
                f"{error_prefix} failed with HTTP {exc.code}",
                error_code=f"{error_code_prefix}_http_error",
                http_status=int(exc.code),
            ) from exc
        except OSError as exc:  # URLError/TimeoutError (OSError subclasses) + raw resets
            if attempt < attempts - 1:
                time.sleep(0.05 * (attempt + 1))
                continue
            if isinstance(exc, TimeoutError):
                raise WebGatewayError(
                    f"{error_prefix} timed out", error_code=f"{error_code_prefix}_timeout"
                ) from exc
            reason = getattr(exc, "reason", exc)
            raise WebGatewayError(
                f"{error_prefix} failed: {reason}", error_code=f"{error_code_prefix}_network_error"
            ) from exc
    raise AssertionError("unreachable: retry loop exited without returning or raising")


def _request_json(
    url: str,
    *,
    headers: dict[str, str],
    timeout_s: int,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    error_prefix: str = "search provider",
    error_code_prefix: str = "web_search",
) -> dict[str, Any]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    body = _urlopen_read_with_retry(
        Request(url, data=data, headers=headers, method=method),
        timeout_s=timeout_s,
        error_prefix=error_prefix,
        error_code_prefix=error_code_prefix,
    )
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise WebGatewayError(f"{error_prefix} returned invalid JSON", error_code=f"{error_code_prefix}_bad_response") from exc
    if not isinstance(payload, dict):
        raise WebGatewayError(f"{error_prefix} response must be an object", error_code=f"{error_code_prefix}_bad_response")
    return payload


def _normalize_brave_results(payload: dict[str, Any], *, max_results: int, source: str) -> list[dict[str, Any]]:
    raw_results = []
    web = payload.get("web")
    if isinstance(web, dict) and isinstance(web.get("results"), list):
        raw_results.extend(item for item in web["results"] if isinstance(item, dict))
    results: list[dict[str, Any]] = []
    for item in raw_results[:max_results]:
        url = str(item.get("url") or "")
        if not url:
            continue
        results.append(
            {
                "title": str(item.get("title") or ""),
                "url": url,
                "domain": domain_from_url(url),
                "snippet": str(item.get("description") or item.get("snippet") or ""),
                "source": source,
            }
        )
    return results


def _normalize_context_payload(payload: dict[str, Any], *, query: str, source: str) -> dict[str, Any]:
    sources = _normalize_sources(payload.get("sources") or payload.get("results") or payload.get("web", {}).get("results"))
    chunks = _normalize_chunks(payload.get("chunks") or payload.get("snippets") or payload.get("context_chunks"))
    if not chunks:
        chunks = _chunks_from_grounding(payload.get("grounding"))
    context = str(
        payload.get("context")
        or payload.get("llm_context")
        or payload.get("content")
        or payload.get("text")
        or ""
    )
    if not context and chunks:
        context = _format_context(query, chunks)
    return {
        "query": str(payload.get("query") or payload.get("q") or query),
        "context": context,
        "sources": sources,
        "chunks": chunks,
        "source": source,
        "raw_keys": sorted(str(key) for key in payload.keys()),
    }


def _normalize_sources(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        mapped_sources: list[dict[str, Any]] = []
        for url, item in value.items():
            if not isinstance(item, dict):
                continue
            url_text = str(item.get("url") or url or "")
            if not url_text:
                continue
            mapped_sources.append(
                {
                    "url": url_text,
                    "title": str(item.get("title") or item.get("name") or ""),
                    "domain": str(item.get("domain") or item.get("hostname") or domain_from_url(url_text)),
                    "snippet": str(item.get("snippet") or item.get("description") or "")[:500],
                    "source": str(item.get("source") or ""),
                }
            )
        return mapped_sources
    if not isinstance(value, list):
        return []
    sources: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("link") or item.get("href") or "")
        if not url:
            continue
        sources.append(
            {
                "url": url,
                "title": str(item.get("title") or item.get("name") or ""),
                "domain": str(item.get("domain") or domain_from_url(url)),
                "snippet": str(item.get("snippet") or item.get("description") or "")[:500],
                "source": str(item.get("source") or ""),
            }
        )
    return sources


def _normalize_chunks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    chunks: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            chunks.append({"text": item})
            continue
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("content") or item.get("snippet") or "")
        if not text:
            continue
        url = str(item.get("url") or item.get("source_url") or "")
        chunks.append(
            {
                "url": url,
                "title": str(item.get("title") or ""),
                "domain": str(item.get("domain") or domain_from_url(url)),
                "text": text,
            }
        )
    return chunks


def _chunks_from_grounding(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    chunks: list[dict[str, Any]] = []
    for section in value.values():
        if not isinstance(section, list):
            continue
        for item in section:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "")
            title = str(item.get("title") or "")
            snippets = item.get("snippets")
            if isinstance(snippets, str):
                snippets = [snippets]
            if not isinstance(snippets, list):
                continue
            for snippet in snippets:
                text = str(snippet or "")
                if not text:
                    continue
                chunks.append(
                    {
                        "url": url,
                        "title": title,
                        "domain": domain_from_url(url),
                        "text": text,
                    }
                )
    return chunks


def _split_context_chunks(content: str, max_chars: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in content.splitlines() if paragraph.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = paragraph[:max_chars]
    if current:
        chunks.append(current)
    return chunks


def _format_context(query: str, chunks: list[dict[str, Any]]) -> str:
    lines = [f"Query: {query}", ""]
    for index, chunk in enumerate(chunks, start=1):
        title = str(chunk.get("title") or "Untitled")
        url = str(chunk.get("url") or "")
        text = str(chunk.get("text") or "")
        lines.append(f"[{index}] {title}")
        if url:
            lines.append(url)
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip()


def _domain_allowed_by_filters(domain: str, allowed_domains: tuple[str, ...], blocked_domains: tuple[str, ...]) -> bool:
    normalized = domain.lower().strip(".")
    if not normalized:
        return False
    if any(_domain_matches(normalized, pattern) for pattern in blocked_domains):
        return False
    if allowed_domains and not any(_domain_matches(normalized, pattern) for pattern in allowed_domains):
        return False
    return True


def _domain_matches(domain: str, pattern: str) -> bool:
    normalized_pattern = pattern.lower().strip()
    if normalized_pattern == "*":
        return True
    if normalized_pattern.startswith("*."):
        suffix = normalized_pattern[2:]
        return domain == suffix or domain.endswith(f".{suffix}")
    return domain == normalized_pattern


def _decode_body(raw: bytes, content_type: str) -> str:
    charset = "utf-8"
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("charset="):
            charset = part.split("=", 1)[1].strip()
            break
    return raw.decode(charset, errors="replace")


def _looks_like_html(content_type: str, text: str) -> bool:
    lowered = content_type.lower()
    return "html" in lowered or "<html" in text[:500].lower() or "<body" in text[:500].lower()


def _extract_title(html: str) -> str:
    parser = _TextHtmlParser()
    parser.feed(html)
    return " ".join(parser.title.split())


def _html_to_text(html: str) -> str:
    parser = _TextHtmlParser()
    parser.feed(html)
    return "\n".join(line for line in (" ".join(chunk.split()) for chunk in parser.text_chunks) if line)


class _TextHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text_chunks: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in {"p", "br", "div", "section", "article", "li", "h1", "h2", "h3"}:
            self.text_chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        text = data.strip()
        if text:
            self.text_chunks.append(text)
