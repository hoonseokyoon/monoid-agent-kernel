"""An MCP server exposed as provider-level tools and optional context.

The core engine is untouched: MCP tools are ordinary ``ToolSpec``s whose handler proxies
``tools/call`` over a synchronous ``McpHttpClient``. Because the core runs tool handlers via
``asyncio.to_thread``, a blocking httpx call in the handler is the natural fit — no async
session/loop bridge is needed (unlike frameworks built on the async-only ``mcp`` SDK).

Resources and prompts remain provider-level too: when the MCP server advertises catalogs,
read-only helper tools expose ``resources/read`` and ``prompts/get`` through the same
``ToolProvider`` seam, and a ``ContextProvider`` catalog appears only while those helper tools
are bound.

Provider tools are NOT auto-bound: a run only sees tools that have a matching ``ToolBinding``
in its runtime config. Use :meth:`McpToolProvider.tool_bindings` to generate those bindings
from the discovered tools and merge them into the runtime config.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from monoid_agent_kernel.core.agents import RegistryToolRef, ToolBinding
from monoid_agent_kernel.mcp.client import McpError, McpHttpClient
from monoid_agent_kernel.tools.base import ToolResult, ToolSideEffect, ToolSpec


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
        self._resources: list[dict[str, Any]] | None = None  # discovered MCP resource descriptors
        self._prompts: list[dict[str, Any]] | None = None  # discovered MCP prompt descriptors

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

    def _discover_resources(self) -> list[dict[str, Any]]:
        if self._resources is None:
            self._client.initialize()
            self._resources = self._optional_catalog(self._client.list_resources)
        return self._resources

    def _discover_prompts(self) -> list[dict[str, Any]]:
        if self._prompts is None:
            self._client.initialize()
            self._prompts = self._optional_catalog(self._client.list_prompts)
        return self._prompts

    def _optional_catalog(self, load: Any) -> list[dict[str, Any]]:
        try:
            return [item for item in load() if isinstance(item, dict)]
        except McpError as exc:
            if exc.code == -32601:  # server does not implement this optional MCP surface
                return []
            raise

    def _selected(self, name: str) -> bool:
        if name in self._blocked:
            return False
        return self._allowed is None or name in self._allowed

    def invalidate_tools(self) -> None:
        """Drop the cached ``tools/list`` result after an observed tools/list_changed signal."""
        self._tools = None

    def invalidate_resources(self) -> None:
        """Drop the cached ``resources/list`` result after an observed resources/list_changed signal."""
        self._resources = None

    def invalidate_prompts(self) -> None:
        """Drop the cached ``prompts/list`` result after an observed prompts/list_changed signal."""
        self._prompts = None

    def handle_list_changed(self, method: str) -> bool:
        """Invalidate the matching catalog for MCP ``notifications/*/list_changed`` messages.

        This is a manual hook for embedders that already receive notifications. The provider does
        not start an SSE listener or subscription loop.
        """
        if method.endswith("tools/list_changed"):
            self.invalidate_tools()
            return True
        if method.endswith("resources/list_changed"):
            self.invalidate_resources()
            return True
        if method.endswith("prompts/list_changed"):
            self.invalidate_prompts()
            return True
        return False

    # -- ContextProvider ----------------------------------------------------------------

    def static_segment(self) -> str | None:
        return None

    def dynamic_segment(self, turn: Any) -> str | None:
        bound = getattr(turn, "bound_tools", frozenset())
        sections: list[str] = []
        if self._resource_read_tool_id() in bound:
            resources = self._discover_resources()
            if resources:
                sections.extend(
                    [
                        f"# MCP Resources ({self._server})",
                        "",
                        f"Read a resource with `{self._resource_read_tool_name()}` using its URI.",
                        "",
                        *_catalog_lines(resources, key="uri"),
                    ]
                )
        if self._prompt_get_tool_id() in bound:
            prompts = self._discover_prompts()
            if prompts:
                if sections:
                    sections.append("")
                sections.extend(
                    [
                        f"# MCP Prompts ({self._server})",
                        "",
                        f"Fetch a prompt template with `{self._prompt_get_tool_name()}` using its name.",
                        "",
                        *_catalog_lines(prompts, key="name"),
                    ]
                )
        return "\n".join(sections) if sections else None

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
        resources = self._discover_resources()
        if resources:
            yield ToolSpec(
                id=self._resource_read_tool_id(),
                description=(
                    f"Read a resource exposed by the MCP server '{self._server}'. Choose 'uri' "
                    "from the advertised MCP resource catalog."
                ),
                input_schema=_object_schema(
                    {
                        "uri": {
                            "type": "string",
                            "enum": [str(r.get("uri")) for r in resources if r.get("uri")],
                        },
                    },
                    required=["uri"],
                ),
                capability=f"mcp.{self._server}",
                side_effect="read",
                handler=self._resource_read_handler(),
                provider_name=self._resource_read_tool_name(),
            )
        prompts = self._discover_prompts()
        if prompts:
            yield ToolSpec(
                id=self._prompt_get_tool_id(),
                description=(
                    f"Get a prompt template exposed by the MCP server '{self._server}'. Choose "
                    "'name' from the advertised MCP prompt catalog and pass optional arguments."
                ),
                input_schema=_object_schema(
                    {
                        "name": {
                            "type": "string",
                            "enum": [str(p.get("name")) for p in prompts if p.get("name")],
                        },
                        "arguments": {"type": "object", "additionalProperties": True, "default": {}},
                    },
                    required=["name"],
                ),
                capability=f"mcp.{self._server}",
                side_effect="read",
                handler=self._prompt_get_handler(),
                provider_name=self._prompt_get_tool_name(),
            )

    def tool_bindings(self) -> tuple[ToolBinding, ...]:
        """Bindings for every discovered tool, to merge into the runtime config so the run can
        see them (provider tools are not auto-bound)."""
        return tuple(
            ToolBinding(binding_id=spec.id, ref=RegistryToolRef(tool_id=spec.id), authorization="allow")
            for spec in self.get_tools()
        )

    def catalog(self) -> dict[str, list[dict[str, Any]]]:
        """Plain catalog data for UIs. Tool descriptors keep id/description; resources/prompts
        expose only server-provided descriptors."""
        return {
            "tools": [
                {"id": f"mcp.{self._server}.{str(tool.get('name') or '')}", "description": str(tool.get("description") or "")}
                for tool in self._discover()
            ],
            "resources": list(self._discover_resources()),
            "prompts": list(self._discover_prompts()),
        }

    def _make_handler(self, mcp_name: str):
        def handler(_context: Any, arguments: dict[str, Any]) -> ToolResult:
            try:
                result = self._client.call_tool(mcp_name, arguments)
            except McpError as exc:
                return ToolResult(ok=False, error=str(exc), error_code="mcp_call_failed", retryable=True)
            return _to_tool_result(result)

        return handler

    def _resource_read_handler(self):
        def handler(_context: Any, arguments: dict[str, Any]) -> ToolResult:
            uri = str(arguments.get("uri") or "")
            try:
                result = self._client.read_resource(uri)
            except McpError as exc:
                return ToolResult(ok=False, error=str(exc), error_code="mcp_resource_read_failed", retryable=True)
            return ToolResult(ok=True, content=result)

        return handler

    def _prompt_get_handler(self):
        def handler(_context: Any, arguments: dict[str, Any]) -> ToolResult:
            name = str(arguments.get("name") or "")
            prompt_args = arguments.get("arguments") or {}
            try:
                result = self._client.get_prompt(name, prompt_args if isinstance(prompt_args, dict) else {})
            except McpError as exc:
                return ToolResult(ok=False, error=str(exc), error_code="mcp_prompt_get_failed", retryable=True)
            return ToolResult(ok=True, content=result)

        return handler

    def _resource_read_tool_id(self) -> str:
        return f"mcp.{self._server}.resource.read"

    def _prompt_get_tool_id(self) -> str:
        return f"mcp.{self._server}.prompt.get"

    def _resource_read_tool_name(self) -> str:
        return f"mcp_{self._server}_resource_read"

    def _prompt_get_tool_name(self) -> str:
        return f"mcp_{self._server}_prompt_get"


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


def _object_schema(properties: dict[str, Any], *, required: list[str] | None = None) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _catalog_lines(items: list[dict[str, Any]], *, key: str) -> list[str]:
    lines: list[str] = []
    for item in items:
        value = str(item.get(key) or "")
        if not value:
            continue
        name = str(item.get("name") or "")
        description = str(item.get("description") or "")
        mime = str(item.get("mimeType") or item.get("mime_type") or "")
        label = value if not name or name == value else f"{value} ({name})"
        suffix_parts = [part for part in (description, mime) if part]
        suffix = f": {'; '.join(suffix_parts)}" if suffix_parts else ""
        lines.append(f"- {label}{suffix}")
    return lines
