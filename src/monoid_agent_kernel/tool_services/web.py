from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from monoid_agent_kernel.core.json_ingress import (
    normalize_json_ingress,
    normalize_unicode_scalars,
)
from monoid_agent_kernel.core.runtime_controls import (
    exact_runtime_integer,
    validate_web_runtime,
)
from monoid_agent_kernel.core.scope import effective_signed_scope
from monoid_agent_kernel.errors import ToolExecutionError, error_code_for_exception
from monoid_agent_kernel.identifiers import namespaced_id
from monoid_agent_kernel.public_view import public_error_message
from monoid_agent_kernel.recorder import AgentRecorder
from monoid_agent_kernel.tool_services.base import CallContext
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.public_view import public_event_payload
from monoid_agent_kernel.web import (
    WebGatewayClient,
    domain_from_url,
    public_query_preview,
    public_url_preview,
)


def _domain_array(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or not all(type(item) is str for item in value):
        raise ValueError(f"web tool {field_name} must be an array of strings")
    return [
        normalized.strip().lower()
        for item in value
        if (normalized := normalize_unicode_scalars(item)).strip()
    ]


def _web_text(value: Any, field_name: str, *, non_empty: bool = False) -> str:
    if type(value) is not str:
        raise ValueError(f"web tool {field_name} must be a string")
    normalized = normalize_unicode_scalars(value)
    if non_empty and not normalized.strip():
        raise ValueError(f"web tool {field_name} must be a non-empty string")
    return normalized


def _optional_web_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _web_text(value, field_name)


@dataclass
class WebService:
    """Orchestrates web search/fetch/context calls: gating, events, counters."""

    recorder: AgentRecorder
    web_gateway_client: WebGatewayClient | None = None
    # Supplied so the hand-built event payloads bound through the same policy the rest of the
    # run publishes under. No web descriptor is a path today, so this changes nothing yet --
    # it is here so that the day one is, this surface is not the one that missed it.
    permission_policy: PermissionPolicy = field(default_factory=PermissionPolicy)
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
        runtime = {} if call.runtime is None else call.runtime
        return validate_web_runtime(runtime)

    def _max_calls(self, call: CallContext, feature: str, default: int) -> int:
        runtime = self._runtime(call)
        for key in ("max_calls", f"max_{feature}_calls"):
            if key in runtime:
                return exact_runtime_integer(
                    runtime[key],
                    field_name=f"web binding runtime {key}",
                    minimum=0,
                )
        return default

    def _bounded_int(
        self,
        call: CallContext,
        requested: Any,
        *,
        default_key: str,
        max_keys: tuple[str, ...],
        default_value: int,
        max_value: int,
    ) -> int:
        runtime = self._runtime(call)
        effective_default = exact_runtime_integer(
            runtime.get(default_key, default_value),
            field_name=f"web binding runtime {default_key}",
            minimum=1,
        )
        selected_max_key = next((key for key in max_keys if key in runtime), max_keys[0])
        effective_max = exact_runtime_integer(
            runtime.get(selected_max_key, max_value),
            field_name=f"web binding runtime {selected_max_key}",
            minimum=1,
        )
        value = (
            effective_default
            if requested is None
            else exact_runtime_integer(
                requested,
                field_name=f"web tool argument for {selected_max_key}",
                minimum=1,
            )
        )
        return max(1, min(value, effective_max))

    def _domain_filters(
        self, args: dict[str, Any], call: CallContext
    ) -> tuple[list[str], list[str]]:
        requested: dict[str, Any] = {}
        requested_allowed = _domain_array(args.get("allowed_domains"), "allowed_domains")
        requested_blocked = _domain_array(args.get("blocked_domains"), "blocked_domains")
        if requested_allowed:
            requested["allowed_domains"] = requested_allowed
        if requested_blocked:
            requested["blocked_domains"] = requested_blocked
        scope: dict[str, Any] = {}
        if call.scope.allowed_domains:
            scope["allowed_domains"] = list(call.scope.allowed_domains)
        if call.scope.blocked_domains:
            scope["blocked_domains"] = list(call.scope.blocked_domains)
        effective = effective_signed_scope(scope, requested, numeric_keys=())
        return list(effective.get("allowed_domains") or ()), list(
            effective.get("blocked_domains") or ()
        )

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
            result = normalize_json_ingress(invoke())
            if not isinstance(result, dict):
                raise ValueError("web gateway result must be an object")
        except Exception as exc:
            self.web_failed_calls += 1
            self.recorder.emit(
                f"{prefix}.failed",
                turn_id=call.turn_id,
                parent_id=started.event_id,
                data={
                    **event_data,
                    "error": public_error_message(str(exc)),
                    "error_code": error_code_for_exception(exc),
                },
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

    def search(
        self, args: dict[str, Any], call: CallContext, *, capability_token: str | None = None
    ) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValueError("web search arguments must be an object")
        if capability_token is not None and type(capability_token) is not str:
            raise ValueError("web capability token must be a string or null")
        self._check_enabled(
            call=call,
            max_calls=self._max_calls(call, "search", 20),
            limit_message="web search call limit exceeded",
            limit_code="web_search_limit_exceeded",
        )
        max_calls = self._max_calls(call, "search", 20)
        query = _web_text(args.get("query"), "query", non_empty=True)
        recency_days = args.get("recency_days")
        if recency_days is not None:
            recency_days = exact_runtime_integer(
                recency_days,
                field_name="web tool recency_days",
                minimum=1,
            )
        locale = _optional_web_text(args.get("locale"), "locale")
        requested_max_results = args.get("max_results")
        effective_max_results = self._bounded_int(
            call,
            requested_max_results,
            default_key="default_max_results",
            max_keys=("max_results",),
            default_value=5,
            max_value=10,
        )
        allowed_domains, blocked_domains = self._domain_filters(args, call)
        # Bounded like the `tool.call.started` preview of the same call. `locale` and the
        # domain lists are model-authored and unconstrained by their schemas, so leaving them
        # raw here published on `.started`/`.finished`/`.failed` exactly what `web_args_preview`
        # was changed to withhold -- in the same event whose `query_preview` is a digest.
        event_data = public_event_payload(
            {
                "query_preview": public_query_preview(query),
                "requested_max_results": requested_max_results,
                "effective_max_results": effective_max_results,
                "allowed_domains": allowed_domains,
                "blocked_domains": blocked_domains,
                "recency_days": recency_days,
                "locale": locale,
                "binding_id": call.binding_id,
            },
            self.permission_policy,
        )
        payload = {
            "protocol": namespaced_id("web-search.v1"),
            "binding_id": call.binding_id,
            "max_calls": max_calls,
            "query": query,
            "max_results": effective_max_results,
            "allowed_domains": allowed_domains,
            "blocked_domains": blocked_domains,
            "recency_days": recency_days,
            "locale": locale,
        }

        def on_success(result: dict[str, Any]) -> dict[str, Any]:
            result_count = int(result.get("result_count") or len(result.get("results") or ()))
            self.web_search_calls += 1
            self.binding_call_counts[call.binding_id] = (
                self.binding_call_counts.get(call.binding_id, 0) + 1
            )
            self.web_result_count += result_count
            return {"result_count": result_count}

        return self._run_call(
            "web.search",
            call,
            event_data=event_data,
            invoke=lambda: self.web_gateway_client.search(payload, token=capability_token),
            on_success=on_success,
        )

    def fetch(
        self, args: dict[str, Any], call: CallContext, *, capability_token: str | None = None
    ) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValueError("web fetch arguments must be an object")
        if capability_token is not None and type(capability_token) is not str:
            raise ValueError("web capability token must be a string or null")
        self._check_enabled(
            call=call,
            max_calls=self._max_calls(call, "fetch", 50),
            limit_message="web fetch call limit exceeded",
            limit_code="web_fetch_limit_exceeded",
        )
        max_calls = self._max_calls(call, "fetch", 50)
        url = _web_text(args.get("url"), "url", non_empty=True)
        response_format = _web_text(args.get("format", "text"), "format")
        if response_format not in {"text", "markdown"}:
            raise ValueError("web tool format must be text or markdown")
        requested_timeout_s = args.get("timeout_s")
        requested_max_bytes = args.get("max_bytes")
        effective_timeout_s = self._bounded_int(
            call,
            requested_timeout_s,
            default_key="default_timeout_s",
            max_keys=("max_timeout_s", "timeout_s"),
            default_value=30,
            max_value=60,
        )
        effective_max_bytes = self._bounded_int(
            call,
            requested_max_bytes,
            default_key="default_max_response_bytes",
            max_keys=("max_response_bytes", "max_bytes"),
            default_value=100_000,
            max_value=1_000_000,
        )
        allowed_domains, blocked_domains = self._domain_filters(args, call)
        # Bounded like the `tool.call.started` preview of the same call -- defensively here, not
        # because a leak was measured on this one. Unlike `web.search`, none of `web.fetch`'s
        # descriptors is an unbounded model-authored string: `format` is an enum the dispatch-path
        # validator enforces, and the domain lists are not in its schema at all
        # (`additionalProperties: false`), so they can only arrive from the operator-signed
        # `call.scope`. Previewing them costs nothing and keeps the three web builders identical,
        # which is the property worth having. The first version of this comment was copy-pasted from
        # `web.search` and named `locale`/`query_preview`; its replacement named `format` and the
        # domain lists as model-authored. Both described a different tool than the one they sat on.
        event_data = public_event_payload(
            {
                "url_preview": public_url_preview(url),
                "domain": domain_from_url(url),
                "format": response_format,
                "requested_timeout_s": requested_timeout_s,
                "effective_timeout_s": effective_timeout_s,
                "requested_max_bytes": requested_max_bytes,
                "effective_max_bytes": effective_max_bytes,
                "allowed_domains": allowed_domains,
                "blocked_domains": blocked_domains,
                "binding_id": call.binding_id,
            },
            self.permission_policy,
        )
        payload = {
            "protocol": namespaced_id("web-fetch.v1"),
            "binding_id": call.binding_id,
            "max_calls": max_calls,
            "url": url,
            "format": response_format,
            "timeout_s": effective_timeout_s,
            "max_bytes": effective_max_bytes,
            "allowed_domains": allowed_domains,
            "blocked_domains": blocked_domains,
        }

        def on_success(result: dict[str, Any]) -> dict[str, Any]:
            content_bytes = int(
                result.get("content_bytes") or len(str(result.get("content") or "").encode("utf-8"))
            )
            self.web_fetch_calls += 1
            self.binding_call_counts[call.binding_id] = (
                self.binding_call_counts.get(call.binding_id, 0) + 1
            )
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
            invoke=lambda: self.web_gateway_client.fetch(payload, token=capability_token),
            on_success=on_success,
        )

    def context(
        self, args: dict[str, Any], call: CallContext, *, capability_token: str | None = None
    ) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValueError("web context arguments must be an object")
        if capability_token is not None and type(capability_token) is not str:
            raise ValueError("web capability token must be a string or null")
        self._check_enabled(
            call=call,
            max_calls=self._max_calls(call, "context", 10),
            limit_message="web context call limit exceeded",
            limit_code="web_context_limit_exceeded",
        )
        max_calls = self._max_calls(call, "context", 10)
        query = _web_text(args.get("query"), "query", non_empty=True)
        recency_days = args.get("recency_days")
        if recency_days is not None:
            recency_days = exact_runtime_integer(
                recency_days,
                field_name="web tool recency_days",
                minimum=1,
            )
        locale = _optional_web_text(args.get("locale"), "locale")
        requested_max_tokens = args.get("max_tokens")
        requested_max_urls = args.get("max_urls")
        requested_max_snippets = args.get("max_snippets")
        effective_max_tokens = self._bounded_int(
            call,
            requested_max_tokens,
            default_key="default_max_context_tokens",
            max_keys=("max_context_tokens", "max_tokens"),
            default_value=8_192,
            max_value=32_768,
        )
        effective_max_urls = self._bounded_int(
            call,
            requested_max_urls,
            default_key="default_max_context_urls",
            max_keys=("max_context_urls", "max_urls"),
            default_value=8,
            max_value=20,
        )
        effective_max_snippets = self._bounded_int(
            call,
            requested_max_snippets,
            default_key="default_max_context_snippets",
            max_keys=("max_context_snippets", "max_snippets"),
            default_value=50,
            max_value=256,
        )
        allowed_domains, blocked_domains = self._domain_filters(args, call)
        # Bounded like the `tool.call.started` preview of the same call. `locale` and the
        # domain lists are model-authored and unconstrained by their schemas, so leaving them
        # raw here published on `.started`/`.finished`/`.failed` exactly what `web_args_preview`
        # was changed to withhold -- in the same event whose `query_preview` is a digest.
        event_data = public_event_payload(
            {
                "query_preview": public_query_preview(query),
                "requested_max_tokens": requested_max_tokens,
                "effective_max_tokens": effective_max_tokens,
                "requested_max_urls": requested_max_urls,
                "effective_max_urls": effective_max_urls,
                "requested_max_snippets": requested_max_snippets,
                "effective_max_snippets": effective_max_snippets,
                "allowed_domains": allowed_domains,
                "blocked_domains": blocked_domains,
                "recency_days": recency_days,
                "locale": locale,
                "binding_id": call.binding_id,
            },
            self.permission_policy,
        )
        payload = {
            "protocol": namespaced_id("web-context.v1"),
            "binding_id": call.binding_id,
            "max_calls": max_calls,
            "query": query,
            "max_tokens": effective_max_tokens,
            "max_urls": effective_max_urls,
            "max_snippets": effective_max_snippets,
            "allowed_domains": allowed_domains,
            "blocked_domains": blocked_domains,
            "recency_days": recency_days,
            "locale": locale,
        }

        def on_success(result: dict[str, Any]) -> dict[str, Any]:
            source_count = int(result.get("source_count") or len(result.get("sources") or ()))
            context_bytes = int(
                result.get("context_bytes") or len(str(result.get("context") or "").encode("utf-8"))
            )
            self.web_context_calls += 1
            self.binding_call_counts[call.binding_id] = (
                self.binding_call_counts.get(call.binding_id, 0) + 1
            )
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
            invoke=lambda: self.web_gateway_client.context(payload, token=capability_token),
            on_success=on_success,
        )
