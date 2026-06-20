"""An MCP server exposed as a ``ToolProvider`` — lists its tools and proxies calls.

The core engine is untouched: MCP tools are ordinary ``ToolSpec``s whose handler proxies
``tools/call`` over a synchronous ``McpHttpClient``. Because the core runs tool handlers via
``asyncio.to_thread``, a blocking httpx call in the handler is the natural fit — no async
session/loop bridge is needed (unlike frameworks built on the async-only ``mcp`` SDK).

Provider tools are NOT auto-bound: a run only sees tools that have a matching ``ToolBinding``
in its runtime config. Use :meth:`McpToolProvider.tool_bindings` to generate those bindings
from the discovered tools and merge them into the runtime config.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from native_agent_runner.core.agents import RegistryToolRef, ToolBinding
from native_agent_runner.mcp.client import McpError, McpHttpClient
from native_agent_runner.tools.base import ToolResult, ToolSideEffect, ToolSpec


class McpToolProvider:
    """A ``ToolProvider`` backed by a remote MCP server (Streamable HTTP)."""

    def __init__(
        self,
        url: str,
        *,
        server: str,
        token: str | None = None,
        allowed_tools: Iterable[str] | None = None,
        blocked_tools: Iterable[str] | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._server = server
        self._client = McpHttpClient(url, token, timeout_s=timeout_s)
        self._allowed = set(allowed_tools) if allowed_tools is not None else None
        self._blocked = set(blocked_tools or ())
        self._tools: list[dict[str, Any]] | None = None  # discovered MCP tool descriptors (cached)

    def __enter__(self) -> McpToolProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # -- discovery ---------------------------------------------------------------------

    def _discover(self) -> list[dict[str, Any]]:
        if self._tools is None:
            self._client.initialize()
            self._tools = [t for t in self._client.list_tools() if self._selected(str(t.get("name") or ""))]
        return self._tools

    def _selected(self, name: str) -> bool:
        if name in self._blocked:
            return False
        return self._allowed is None or name in self._allowed

    # -- ToolProvider ------------------------------------------------------------------

    def get_tools(self, context: Any = None) -> Iterable[ToolSpec]:  # noqa: ARG002 - context unused
        for tool in self._discover():
            name = str(tool.get("name") or "")
            yield ToolSpec(
                id=f"mcp.{self._server}.{name}",
                description=str(tool.get("description") or ""),
                input_schema=dict(tool.get("inputSchema") or {"type": "object"}),
                capability=f"mcp.{self._server}",
                side_effect=_side_effect(tool.get("annotations")),
                handler=self._make_handler(name),
                provider_name=f"mcp_{self._server}_{name}",  # exported_name; avoids registry collisions
                annotations=dict(tool.get("annotations") or {}),
            )

    def tool_bindings(self) -> tuple[ToolBinding, ...]:
        """Bindings for every discovered tool, to merge into the runtime config so the run can
        see them (provider tools are not auto-bound)."""
        return tuple(
            ToolBinding(binding_id=spec.id, ref=RegistryToolRef(tool_id=spec.id), authorization="allow")
            for spec in self.get_tools()
        )

    def _make_handler(self, mcp_name: str):
        def handler(_context: Any, arguments: dict[str, Any]) -> ToolResult:
            try:
                result = self._client.call_tool(mcp_name, arguments)
            except McpError as exc:
                return ToolResult(ok=False, error=str(exc), error_code="mcp_call_failed", retryable=True)
            return _to_tool_result(result)

        return handler


def _side_effect(annotations: dict[str, Any] | None) -> ToolSideEffect:
    """Map MCP tool annotations to a side-effect tag. A read-only tool has no effect; anything
    else is treated as a general side-effecting action so it routes through the existing
    permission/approval path. MCP tools never touch the workspace, so ``run`` (no path_args)
    emits no workspace events."""
    if annotations and annotations.get("readOnlyHint") is True:
        return "read"
    return "run"


def _to_tool_result(call_result: dict[str, Any]) -> ToolResult:
    """Map an MCP ``CallToolResult`` to a ``ToolResult``. ``isError`` is a real tool failure
    (so the model can self-correct); content blocks are iterated explicitly rather than
    stringifying the whole object."""
    text = _join_text(call_result.get("content"))
    if call_result.get("isError"):
        return ToolResult(ok=False, error=text or "MCP tool error", error_code="mcp_tool_error", category="tool")
    content: dict[str, Any] = {}
    structured = call_result.get("structuredContent")
    if structured is not None:
        content["structured"] = structured
    if text:
        content["text"] = text
    media = _collect_media(call_result.get("content"))
    if media:
        content["media"] = media
    return ToolResult(ok=True, content=content)


def _join_text(blocks: Any) -> str:
    if not isinstance(blocks, list):
        return ""
    parts = [str(b.get("text") or "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(p for p in parts if p)


def _collect_media(blocks: Any) -> list[dict[str, Any]]:
    """Non-text content blocks as structured descriptors (base64 + mime), until real multimodal
    result forwarding lands in the core."""
    if not isinstance(blocks, list):
        return []
    media: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in ("image", "audio"):
            media.append({"type": block_type, "mime_type": block.get("mimeType"), "data": block.get("data")})
        elif block_type == "resource":
            resource = block.get("resource") or {}
            if isinstance(resource, dict):
                media.append({"type": "resource", "uri": resource.get("uri"), "mime_type": resource.get("mimeType")})
        elif block_type == "resource_link":
            media.append({"type": "resource_link", "uri": block.get("uri"), "name": block.get("name")})
    return media
