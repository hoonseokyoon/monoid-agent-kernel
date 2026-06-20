"""MCP (Model Context Protocol) client — connect to MCP servers and use their tools.

Opt-in: requires the ``[mcp]`` extra (httpx), imported lazily so the package imports without
it. The core engine is untouched — an MCP server is surfaced as an ordinary ``ToolProvider``.
"""

from native_agent_runner.mcp.client import McpError
from native_agent_runner.mcp.provider import McpToolProvider

__all__ = ["McpToolProvider", "McpError"]
