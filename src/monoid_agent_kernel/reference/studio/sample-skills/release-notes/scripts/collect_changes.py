from __future__ import annotations

import sys
from pathlib import Path


HEADINGS = {
    "add": "Added",
    "new": "Added",
    "change": "Changed",
    "update": "Changed",
    "fix": "Fixed",
    "bug": "Fixed",
    "remove": "Removed",
    "delete": "Removed",
    "security": "Security",
}


def bucket(line: str) -> str:
    lowered = line.lower()
    for marker, heading in HEADINGS.items():
        if marker in lowered:
            return heading
    return "Changed"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: collect_changes.py <workspace-text-file>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    lines = [line.strip("- ").strip() for line in path.read_text(encoding="utf-8").splitlines()]
    grouped: dict[str, list[str]] = {}
    for line in lines:
        if not line:
            continue
        grouped.setdefault(bucket(line), []).append(line)
    for heading in ("Added", "Changed", "Fixed", "Removed", "Security"):
        items = grouped.get(heading)
        if not items:
            continue
        print(f"## {heading}")
        for item in items:
            print(f"- {item}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
