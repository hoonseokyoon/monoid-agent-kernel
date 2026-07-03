"""Offline reference MCP server (fake), so MCP can be demonstrated with no key/egress.

The counterpart of the echo LLM gateway and ``FakeWebProvider``, for MCP. Studio boots this
on a loopback port and points a core ``McpToolProvider`` at it.
"""

from monoid_agent_kernel.reference.mcp_gateway.http import create_mcp_server, make_mcp_handler
from monoid_agent_kernel.reference.mcp_gateway.service import (
    DEFAULT_FAKE_MCP_PROMPTS,
    DEFAULT_FAKE_MCP_RESOURCES,
    DEFAULT_FAKE_MCP_TOOLS,
    PROTOCOL_VERSION,
    FakeMcpError,
    FakeMcpServer,
)

__all__ = [
    "DEFAULT_FAKE_MCP_PROMPTS",
    "DEFAULT_FAKE_MCP_RESOURCES",
    "DEFAULT_FAKE_MCP_TOOLS",
    "PROTOCOL_VERSION",
    "FakeMcpError",
    "FakeMcpServer",
    "create_mcp_server",
    "make_mcp_handler",
]
