"""Every ``monoid ...`` invocation in the living docs must resolve against the real Click tree.

Added after a review round spent on a documented command that cannot run: the release notes told
operators to disable the durable model-text delta channel with ``monoid studio --no-output-deltas``,
but the flag is registered by ``_common_server_options`` on the ``serve`` / ``app`` / ``doctor``
subcommands, not on the ``studio`` group — so following the instruction produces
``Error: No such option: --no-output-deltas`` and leaves the channel on. The same wrong form had been
copied into four places, which is the point: proof-reading catches the site a reviewer happens to
read, and this catches all of them.

Deliberately **not** applied to ``CHANGELOG.md``. A changelog is a historical record: an entry for
v0.9 describing a command that has since been renamed is correct as written, and failing on it would
push the next author to either edit history or delete the test. README and ``docs/`` are living
documentation — an invocation there is a present-tense instruction, so it has to work today.

Validation is by introspection, never execution: these commands start servers and touch workspaces.
Only the command *path* and *option names* are checked, since option values and placeholders
(``<run-id>``, ``$WORKSPACE``) are illustrative by nature.
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import pytest

from monoid_agent_kernel.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]

# Inline code spans and fenced-block lines both matter: the defect that prompted this was in an
# inline span, and `docs/` teaches through fenced blocks.
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_CONTINUATION = re.compile(r"\\\s*$")


def _living_docs() -> list[Path]:
    return [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").glob("*.md"))]


def _candidate_commands(text: str) -> list[str]:
    """Pull `monoid ...` invocations out of inline spans and fenced bash blocks."""
    found: list[str] = [span.strip() for span in _INLINE_CODE.findall(text) if span.strip().startswith("monoid ")]

    in_fence = False
    buffer = ""
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            buffer = ""
            continue
        if not in_fence:
            continue
        # Fenced examples wrap with a trailing backslash; join before parsing.
        if buffer or line.startswith("monoid "):
            buffer = f"{buffer} {_CONTINUATION.sub('', line).strip()}".strip()
            if not _CONTINUATION.search(raw):
                found.append(buffer)
                buffer = ""
    return found


def _tokens(command: str) -> list[str]:
    # `shlex` would choke on the placeholder syntax docs legitimately use; a whitespace split is
    # enough, because only the leading path and the `--flags` are inspected.
    return command.replace("\n", " ").split()


def _resolve(tokens: list[str]) -> tuple[click.Command, list[str]]:
    """Walk the group tree as far as the tokens lead, returning the command and its flags."""
    node: click.Command = main
    index = 1  # skip "monoid"
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("-"):
            break
        if not isinstance(node, click.Group):
            break
        child = node.get_command(click.Context(node), token)
        if child is None:
            raise AssertionError(f"no such command: {' '.join(tokens[: index + 1])}")
        node = child
        index += 1
    return node, [token for token in tokens[index:] if token.startswith("--")]


def _documented_invocations() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for path in _living_docs():
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for command in _candidate_commands(text):
            pairs.append((str(path.relative_to(REPO_ROOT)).replace("\\", "/"), command))
    return pairs


INVOCATIONS = _documented_invocations()


def test_the_docs_actually_contain_invocations_to_check() -> None:
    """Guards the guard. A regex that silently matches nothing turns every assertion below into a
    vacuous pass, which is worse than no test — it reads as coverage."""
    assert len(INVOCATIONS) >= 10, INVOCATIONS
    assert any("studio" in command for _, command in INVOCATIONS)


@pytest.mark.parametrize(
    ("source", "command"),
    INVOCATIONS,
    ids=[f"{source}::{command[:60]}" for source, command in INVOCATIONS],
)
def test_a_documented_invocation_resolves_to_a_real_command_and_real_options(
    source: str, command: str
) -> None:
    tokens = _tokens(command)
    node, flags = _resolve(tokens)

    known = {
        opt
        for param in node.get_params(click.Context(node))
        for opt in getattr(param, "opts", ()) + getattr(param, "secondary_opts", ())
    }
    for flag in flags:
        name = flag.split("=", 1)[0]
        assert name in known, (
            f"{source} documents `{command}`, but `{name}` is not an option of "
            f"`{node.name}` — an operator following this gets a Click usage error"
        )
