from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from native_agent_runner.errors import ToolExecutionError, error_code_for_exception
from native_agent_runner.public_view import public_error_message
from native_agent_runner.recorder import AgentRecorder
from native_agent_runner.tool_services.base import CallContext
from native_agent_runner.web import (
    WebGatewayClient,
    domain_from_url,
    public_query_preview,
    public_url_preview,
)


@dataclass
class WebService:
    """Orchestrates web search/fetch/context calls: gating, events, counters."""

    recorder: AgentRecorder
    web_gateway_client: WebGatewayClient | None = None
    web_search_calls: int = 0
    web_fetch_calls: int = 0
    web_context_calls: int = 0
    web_failed_calls: int = 0
    web_result_count: int = 0
    web_bytes_returned: int = 0
    web_context_source_count: int = 0
    web_context_bytes_returned: int = 0
    binding_call_counts: dict[str, int] = field(default_factory=dict)

    def metrics(self) -> dict[str, Any]:
        return {
            "web_search_calls": self.web_search_calls,
            "web_fetch_calls": self.web_fetch_calls,
            "web_context_calls": self.web_context_calls,
            "web_failed_calls": self.web_failed_calls,
            "web_result_count": self.web_result_count,
            "web_bytes_returned": self.web_bytes_returned,
            "web_context_source_count": self.web_context_source_count,
            "web_context_bytes_returned": self.web_context_bytes_returned,
        }

    def _check_enabled(
        self,
        *,
        call: CallContext,
        max_calls: int,
        limit_message: str,
        limit_code: str,
    ) -> None:
        if self.web_gateway_client is None:
            raise ToolExecutionError("web gateway is not configured", error_code="web_disabled")
        if self.binding_call_counts.get(call.binding_id, 0) >= max_calls:
            raise ToolExecutionError(limit_message, error_code=limit_code)

    def _runtime(self, call: CallContext) -> dict[str, Any]:
        runtime = call.runtime or {}
        if not isinstance(runtime, dict):
            return {}
        web_runtime = runtime.get("web", runtime)
        return dict(web_runtime) if isinstance(web_runtime, dict) else {}

    def _max_calls(self, call: CallContext, feature: str, default: int) -> int:
        runtime = self._runtime(call)
        return max(0, int(runtime.get("max_calls", runtime.get(f"max_{feature}_calls", default))))

    def _bounded_int(
        self,
        call: CallContext,
        requested: Any,
        *,
        default_key: str,
        max_key: str,
        default_value: int,
        max_value: int,
    ) -> int:
        runtime = self._runtime(call)
        effective_default = int(runtime.get(default_key, default_value))
        effective_max = int(runtime.get(max_key, max_value))
        value = effective_default if requested is None else int(requested)
        return max(1, min(value, effective_max))

    def _domain_filters(self, args: dict[str, Any], call: CallContext) -> tuple[list[str], list[str]]:
        requested_allowed = [str(item).strip().lower() for item in args.get("allowed_domains") or () if str(item).strip()]
        requested_blocked = [str(item).strip().lower() for item in args.get("blocked_domains") or () if str(item).strip()]
        scope_allowed = list(call.scope.allowed_domains)
        scope_blocked = list(call.scope.blocked_domains)
        if scope_allowed and requested_allowed:
            allowed = [domain for domain in requested_allowed if domain in scope_allowed]
        else:
            allowed = scope_allowed or requested_allowed
        return allowed, [*scope_blocked, *requested_blocked]

    def _run_call(
        self,
        prefix: str,
        call: CallContext,
        *,
        event_data: dict[str, Any],
        invoke: Callable[[], dict[str, Any]],
        on_success: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        started = self.recorder.emit(
            f"{prefix}.started",
            turn_id=call.turn_id,
            parent_id=call.tool_event_id,
            data=event_data,
        )
        try:
            result = invoke()
        except Exception as exc:
            self.web_failed_calls += 1
            self.recorder.emit(
                f"{prefix}.failed",
                turn_id=call.turn_id,
                parent_id=started.event_id,
                data={**event_data, "error": public_error_message(str(exc)), "error_code": error_code_for_exception(exc)},
                level="warning",
            )
            raise
        finished_extra = on_success(result)
        self.recorder.emit(
            f"{prefix}.finished",
            turn_id=call.turn_id,
            parent_id=started.event_id,
            data={**event_data, **finished_extra},
        )
        return result

    def search(self, args: dict[str, Any], call: CallContext) -> dict[str, Any]:
        self._check_enabled(
            call=call,
            max_calls=self._max_calls(call, "search", 20),
            limit_message="web search call limit exceeded",
            limit_code="web_search_limit_exceeded",
        )
        max_calls = self._max_calls(call, "search", 20)
        query = str(args["query"])
        requested_max_results = args.get("max_results")
        effective_max_results = self._bounded_int(
            call,
            requested_max_results,
            default_key="default_max_results",
            max_key="max_results",
            default_value=5,
            max_value=10,
        )
        allowed_domains, blocked_domains = self._domain_filters(args, call)
        event_data = {
            "query_preview": public_query_preview(query),
            "requested_max_results": requested_max_results,
            "effective_max_results": effective_max_results,
            "allowed_domains": allowed_domains,
            "blocked_domains": blocked_domains,
            "recency_days": args.get("recency_days"),
            "locale": args.get("locale"),
            "binding_id": call.binding_id,
        }
        payload = {
            "protocol": "native-agent-runner.web-search.v1",
            "binding_id": call.binding_id,
            "max_calls": max_calls,
            "query": query,
            "max_results": effective_max_results,
            "allowed_domains": allowed_domains,
            "blocked_domains": blocked_domains,
            "recency_days": args.get("recency_days"),
            "locale": args.get("locale"),
        }

        def on_success(result: dict[str, Any]) -> dict[str, Any]:
            result_count = int(result.get("result_count") or len(result.get("results") or ()))
            self.web_search_calls += 1
            self.binding_call_counts[call.binding_id] = self.binding_call_counts.get(call.binding_id, 0) + 1
            self.web_result_count += result_count
            return {"result_count": result_count}

        return self._run_call(
            "web.search",
            call,
            event_data=event_data,
            invoke=lambda: self.web_gateway_client.search(payload),
            on_success=on_success,
        )

    def fetch(self, args: dict[str, Any], call: CallContext) -> dict[str, Any]:
        self._check_enabled(
            call=call,
            max_calls=self._max_calls(call, "fetch", 50),
            limit_message="web fetch call limit exceeded",
            limit_code="web_fetch_limit_exceeded",
        )
        max_calls = self._max_calls(call, "fetch", 50)
        url = str(args["url"])
        requested_timeout_s = args.get("timeout_s")
        requested_max_bytes = args.get("max_bytes")
        effective_timeout_s = self._bounded_int(
            call,
            requested_timeout_s,
            default_key="default_timeout_s",
            max_key="max_timeout_s",
            default_value=30,
            max_value=60,
        )
        effective_max_bytes = self._bounded_int(
            call,
            requested_max_bytes,
            default_key="default_max_response_bytes",
            max_key="max_response_bytes",
            default_value=100_000,
            max_value=1_000_000,
        )
        event_data = {
            "url_preview": public_url_preview(url),
            "domain": domain_from_url(url),
            "format": args.get("format") or "text",
            "requested_timeout_s": requested_timeout_s,
            "effective_timeout_s": effective_timeout_s,
            "requested_max_bytes": requested_max_bytes,
            "effective_max_bytes": effective_max_bytes,
            "binding_id": call.binding_id,
        }
        payload = {
            "protocol": "native-agent-runner.web-fetch.v1",
            "binding_id": call.binding_id,
            "max_calls": max_calls,
            "url": url,
            "format": args.get("format") or "text",
            "timeout_s": effective_timeout_s,
            "max_bytes": effective_max_bytes,
        }

        def on_success(result: dict[str, Any]) -> dict[str, Any]:
            content_bytes = int(result.get("content_bytes") or len(str(result.get("content") or "").encode("utf-8")))
            self.web_fetch_calls += 1
            self.binding_call_counts[call.binding_id] = self.binding_call_counts.get(call.binding_id, 0) + 1
            self.web_bytes_returned += content_bytes
            return {
                "final_domain": domain_from_url(str(result.get("final_url") or url)),
                "content_bytes": content_bytes,
                "truncated": bool(result.get("truncated", False)),
            }

        return self._run_call(
            "web.fetch",
            call,
            event_data=event_data,
            invoke=lambda: self.web_gateway_client.fetch(payload),
            on_success=on_success,
        )

    def context(self, args: dict[str, Any], call: CallContext) -> dict[str, Any]:
        self._check_enabled(
            call=call,
            max_calls=self._max_calls(call, "context", 10),
            limit_message="web context call limit exceeded",
            limit_code="web_context_limit_exceeded",
        )
        max_calls = self._max_calls(call, "context", 10)
        query = str(args["query"])
        requested_max_tokens = args.get("max_tokens")
        requested_max_urls = args.get("max_urls")
        requested_max_snippets = args.get("max_snippets")
        effective_max_tokens = self._bounded_int(
            call,
            requested_max_tokens,
            default_key="default_max_context_tokens",
            max_key="max_context_tokens",
            default_value=8_192,
            max_value=32_768,
        )
        effective_max_urls = self._bounded_int(
            call,
            requested_max_urls,
            default_key="default_max_context_urls",
            max_key="max_context_urls",
            default_value=8,
            max_value=20,
        )
        effective_max_snippets = self._bounded_int(
            call,
            requested_max_snippets,
            default_key="default_max_context_snippets",
            max_key="max_context_snippets",
            default_value=50,
            max_value=256,
        )
        allowed_domains, blocked_domains = self._domain_filters(args, call)
        event_data = {
            "query_preview": public_query_preview(query),
            "requested_max_tokens": requested_max_tokens,
            "effective_max_tokens": effective_max_tokens,
            "requested_max_urls": requested_max_urls,
            "effective_max_urls": effective_max_urls,
            "requested_max_snippets": requested_max_snippets,
            "effective_max_snippets": effective_max_snippets,
            "allowed_domains": allowed_domains,
            "blocked_domains": blocked_domains,
            "recency_days": args.get("recency_days"),
            "locale": args.get("locale"),
            "binding_id": call.binding_id,
        }
        payload = {
            "protocol": "native-agent-runner.web-context.v1",
            "binding_id": call.binding_id,
            "max_calls": max_calls,
            "query": query,
            "max_tokens": effective_max_tokens,
            "max_urls": effective_max_urls,
            "max_snippets": effective_max_snippets,
            "allowed_domains": allowed_domains,
            "blocked_domains": blocked_domains,
            "recency_days": args.get("recency_days"),
            "locale": args.get("locale"),
        }

        def on_success(result: dict[str, Any]) -> dict[str, Any]:
            source_count = int(result.get("source_count") or len(result.get("sources") or ()))
            context_bytes = int(result.get("context_bytes") or len(str(result.get("context") or "").encode("utf-8")))
            self.web_context_calls += 1
            self.binding_call_counts[call.binding_id] = self.binding_call_counts.get(call.binding_id, 0) + 1
            self.web_context_source_count += source_count
            self.web_context_bytes_returned += context_bytes
            return {
                "source_count": source_count,
                "context_bytes": context_bytes,
                "estimated_tokens": result.get("estimated_tokens"),
            }

        return self._run_call(
            "web.context",
            call,
            event_data=event_data,
            invoke=lambda: self.web_gateway_client.context(payload),
            on_success=on_success,
        )
