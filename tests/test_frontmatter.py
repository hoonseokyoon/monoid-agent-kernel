from __future__ import annotations

from monoid_agent_kernel.core.frontmatter import parse_frontmatter, parse_scalar


def test_no_frontmatter_returns_body_unchanged() -> None:
    meta, body = parse_frontmatter("just a body\nno fences")
    assert meta == {}
    assert body == "just a body\nno fences"


def test_scalars_and_types() -> None:
    text = """---
name: code-reviewer
description: "Reviews code, carefully"
maxTurns: 5
background: true
model: ~
---
body here
"""
    meta, body = parse_frontmatter(text)
    assert meta["name"] == "code-reviewer"
    assert meta["description"] == "Reviews code, carefully"
    assert meta["maxTurns"] == 5
    assert meta["background"] is True
    assert meta["model"] is None
    assert body == "body here\n"


def test_inline_and_block_lists() -> None:
    text = """---
tools: [fs.read, fs.write]
disallowed:
  - shell.exec
  - mcp.*
---
"""
    meta, _ = parse_frontmatter(text)
    assert meta["tools"] == ["fs.read", "fs.write"]
    assert meta["disallowed"] == ["shell.exec", "mcp.*"]


def test_crlf_and_comments() -> None:
    text = "---\r\n# a comment\r\nname: x\r\ntools: [a]\r\n---\r\nbody\r\n"
    meta, body = parse_frontmatter(text)
    assert meta["name"] == "x"
    assert meta["tools"] == ["a"]
    assert body == "body\n"


def test_parse_scalar_quoted_list_with_comma() -> None:
    assert parse_scalar('["a, b", c]') == ["a, b", "c"]
    assert parse_scalar("-12") == -12
    assert parse_scalar("'quoted'") == "quoted"
    assert parse_scalar("false") is False
