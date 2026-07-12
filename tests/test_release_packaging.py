from __future__ import annotations

import tomllib
from pathlib import Path


def test_sdist_excludes_workspace_local_release_data() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads(project_root.joinpath("pyproject.toml").read_text(encoding="utf-8"))
    exclude = set(pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"])

    assert "/.tmp" in exclude
    assert "**/DX_NOTES.md" in exclude
