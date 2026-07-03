"""A fake, offline MCP server — the LLM/web-gateway "fake provider" ethos applied to MCP.

The ``McpToolProvider`` (core ``mcp/``) talks JSON-RPC over HTTP to a real MCP server. To
demonstrate MCP with **no key, no egress, and no external process** — the same promise the
echo LLM gateway and ``FakeWebProvider`` keep — studio boots this in-process on a loopback
port. The wire protocol mirrors the MCP Streamable-HTTP transport closely enough for the
production client: ``initialize`` / ``tools/*`` / ``resources/*`` / ``prompts/*`` /
``notifications/initialized``.

This module is pure logic (catalog + dispatch); the HTTP shell lives in ``http.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

PROTOCOL_VERSION = "2025-06-18"


class FakeMcpError(Exception):
    """A JSON-RPC error to return to the client (carries the JSON-RPC error ``code``)."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _impl_echo(args: Mapping[str, Any]) -> dict[str, Any]:
    text = str(args.get("text", ""))
    return {"content": [{"type": "text", "text": f"echo: {text}"}], "structuredContent": {"echoed": text}}


def _impl_uppercase(args: Mapping[str, Any]) -> dict[str, Any]:
    text = str(args.get("text", ""))
    upper = text.upper()
    return {"content": [{"type": "text", "text": upper}], "structuredContent": {"upper": upper}}


# Tool catalog + implementations together so they can't drift. ``echo`` mutates nothing but is
# left as a default "run" tool (no readOnlyHint); ``uppercase`` is annotated read-only so the
# provider maps it to side_effect="read" (the readOnlyHint path).
@dataclass(frozen=True)
class _FakeTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    impl: Callable[[Mapping[str, Any]], dict[str, Any]]
    read_only: bool = False

    def descriptor(self) -> dict[str, Any]:
        descriptor: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
        if self.read_only:
            descriptor["annotations"] = {"readOnlyHint": True}
        return descriptor


_TEXT_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

DEFAULT_FAKE_MCP_TOOLS: tuple[_FakeTool, ...] = (
    _FakeTool("echo", "Echo the input text back.", _TEXT_SCHEMA, _impl_echo),
    _FakeTool("uppercase", "Upper-case the input text.", _TEXT_SCHEMA, _impl_uppercase, read_only=True),
)


@dataclass(frozen=True)
class _FakeResource:
    uri: str
    name: str
    description: str
    mime_type: str
    text: str

    def descriptor(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }

    def content(self) -> dict[str, Any]:
        return {"uri": self.uri, "mimeType": self.mime_type, "text": self.text}


@dataclass(frozen=True)
class _FakePrompt:
    name: str
    description: str
    template: str
    arguments: tuple[dict[str, Any], ...] = ()

    def descriptor(self) -> dict[str, Any]:
        descriptor: dict[str, Any] = {"name": self.name, "description": self.description}
        if self.arguments:
            descriptor["arguments"] = list(self.arguments)
        return descriptor

    def render(self, args: Mapping[str, Any] | None) -> dict[str, Any]:
        values = {str(k): str(v) for k, v in (args or {}).items()}
        try:
            text = self.template.format(**values)
        except KeyError:
            text = self.template
        return {
            "description": self.description,
            "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
        }


DEFAULT_FAKE_MCP_RESOURCES: tuple[_FakeResource, ...] = (
    _FakeResource(
        "fake://studio/guide",
        "studio-guide",
        "Short guide for the fake Studio MCP server.",
        "text/plain",
        "Use echo for round trips and uppercase for read-only text transforms.",
    ),
)

DEFAULT_FAKE_MCP_PROMPTS: tuple[_FakePrompt, ...] = (
    _FakePrompt(
        "summarize",
        "Ask for a concise summary of a topic.",
        "Summarize {topic} in two sentences.",
        arguments=(
            {
                "name": "topic",
                "description": "Topic to summarize.",
                "required": True,
            },
        ),
    ),
)


@dataclass
class FakeMcpServer:
    """Offline MCP server logic. ``handle_*`` methods return JSON-RPC ``result`` payloads (the
    HTTP shell wraps them in the JSON-RPC envelope and manages the session id)."""

    tools: tuple[_FakeTool, ...] = DEFAULT_FAKE_MCP_TOOLS
    resources: tuple[_FakeResource, ...] = DEFAULT_FAKE_MCP_RESOURCES
    prompts: tuple[_FakePrompt, ...] = DEFAULT_FAKE_MCP_PROMPTS
    server_name: str = "studio-fake-mcp"
    session_id: str = "studio-mcp-session"
    protocol_version: str = PROTOCOL_VERSION
    _by_name: dict[str, _FakeTool] = field(init=False)
    _resources_by_uri: dict[str, _FakeResource] = field(init=False)
    _prompts_by_name: dict[str, _FakePrompt] = field(init=False)

    def __post_init__(self) -> None:
        self._by_name = {tool.name: tool for tool in self.tools}
        self._resources_by_uri = {resource.uri: resource for resource in self.resources}
        self._prompts_by_name = {prompt.name: prompt for prompt in self.prompts}

    def initialize(self) -> dict[str, Any]:
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": self.server_name, "version": "1"},
        }

    def list_tools(self) -> dict[str, Any]:
        return {"tools": [tool.descriptor() for tool in self.tools]}

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None) -> dict[str, Any]:
        tool = self._by_name.get(name)
        if tool is None:
            raise FakeMcpError(-32602, f"Unknown tool: {name}")
        return tool.impl(arguments or {})

    def list_resources(self) -> dict[str, Any]:
        return {"resources": [resource.descriptor() for resource in self.resources]}

    def read_resource(self, uri: str) -> dict[str, Any]:
        resource = self._resources_by_uri.get(uri)
        if resource is None:
            raise FakeMcpError(-32602, f"Unknown resource: {uri}")
        return {"contents": [resource.content()]}

    def list_prompts(self) -> dict[str, Any]:
        return {"prompts": [prompt.descriptor() for prompt in self.prompts]}

    def get_prompt(self, name: str, arguments: Mapping[str, Any] | None) -> dict[str, Any]:
        prompt = self._prompts_by_name.get(name)
        if prompt is None:
            raise FakeMcpError(-32602, f"Unknown prompt: {name}")
        return prompt.render(arguments)

    def catalog(self) -> list[dict[str, str]]:
        """Plain name+description list for a UI catalog (no schema/impl)."""
        return [{"name": tool.name, "description": tool.description} for tool in self.tools]
