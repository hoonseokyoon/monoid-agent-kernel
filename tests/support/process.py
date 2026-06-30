from __future__ import annotations


def python_command(code: str) -> str:
    escaped = code.replace('"', '\\"')
    return f'python -c "{escaped}"'

