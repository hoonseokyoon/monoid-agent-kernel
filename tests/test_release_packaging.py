from __future__ import annotations

import tomllib
from pathlib import Path

from monoid_agent_kernel._version import FALLBACK_VERSION


def test_release_version_metadata_is_consistent() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads(project_root.joinpath("pyproject.toml").read_text(encoding="utf-8"))
    project_version = pyproject["project"]["version"]

    assert FALLBACK_VERSION == project_version
    assert f"## [{project_version}]" in project_root.joinpath("CHANGELOG.md").read_text(
        encoding="utf-8"
    )


def test_sdist_excludes_workspace_local_release_data() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads(project_root.joinpath("pyproject.toml").read_text(encoding="utf-8"))
    exclude = set(pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"])

    assert "/.tmp" in exclude
    assert "/studio-ui" in exclude
    assert "**/DX_NOTES.md" in exclude


def test_compiled_studio_assets_are_present_in_python_package() -> None:
    project_root = Path(__file__).resolve().parents[1]
    app_dir = project_root / "src/monoid_agent_kernel/reference/studio/web/dist"

    index = app_dir.joinpath("index.html").read_text(encoding="utf-8")
    assert '<div id="app"></div>' in index
    assert list(app_dir.glob("assets/*.js"))
    assert list(app_dir.glob("assets/*.css"))
