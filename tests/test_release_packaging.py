from __future__ import annotations

import tomllib
from pathlib import Path

from monoid_agent_kernel._version import FALLBACK_VERSION


EXPECTED_BASE_DEPENDENCIES = (
    "click>=8.1",
    "jsonschema>=4.21",
    "pathspec>=1.1,<2",
    "pydantic>=2.6",
)

EXPECTED_DURABLE_EXTRAS = {
    "postgres": (
        "psycopg[binary]>=3.2,<4",
        "psycopg-pool>=3.2,<4",
    ),
    "object-store-s3": ("boto3>=1.37.32,<2",),
    "temporal": ("temporalio>=1.17,<2",),
    "durable-host": (
        "psycopg[binary]>=3.2,<4",
        "psycopg-pool>=3.2,<4",
        "boto3>=1.37.32,<2",
        "temporalio>=1.17,<2",
    ),
}


def test_release_version_metadata_is_consistent() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads(project_root.joinpath("pyproject.toml").read_text(encoding="utf-8"))
    project_version = pyproject["project"]["version"]

    assert FALLBACK_VERSION == project_version
    assert f"## [{project_version}]" in project_root.joinpath("CHANGELOG.md").read_text(
        encoding="utf-8"
    )
    assert f'EXPECTED_VERSION = "{project_version}"' in project_root.joinpath(
        "tools", "release_wheel_audit.py"
    ).read_text(encoding="utf-8")


def test_v022_base_dependency_set_is_frozen_and_platform_neutral() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads(project_root.joinpath("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = tuple(pyproject["project"]["dependencies"])

    assert dependencies == EXPECTED_BASE_DEPENDENCIES
    lowered = "\n".join(dependencies).lower()
    assert all(
        platform_dependency not in lowered
        for platform_dependency in ("dbos", "psycopg", "redis", "temporal")
    )


def test_v023_durable_adapters_are_explicit_optional_extras() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads(project_root.joinpath("pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    for name, expected in EXPECTED_DURABLE_EXTRAS.items():
        assert tuple(extras[name]) == expected


def test_postgres_migration_resources_have_platform_stable_bytes() -> None:
    project_root = Path(__file__).resolve().parents[1]
    attributes = project_root.joinpath(".gitattributes").read_text(encoding="utf-8")

    assert "src/monoid_agent_kernel/adapters/postgres/sql/*.sql text eol=lf" in attributes


def test_publish_workflow_audits_the_wheel_that_it_uploads() -> None:
    project_root = Path(__file__).resolve().parents[1]
    workflow = project_root.joinpath(".github/workflows/publish.yml").read_text(encoding="utf-8")

    build_position = workflow.index("- name: Build distributions")
    audit_position = workflow.index("- name: Audit release wheel")
    upload_position = workflow.index("- uses: actions/upload-artifact@v7")

    assert build_position < audit_position < upload_position
    assert "run: python tools/release_wheel_audit.py dist" in workflow


def test_ci_install_smoke_installs_the_audited_exact_wheel() -> None:
    project_root = Path(__file__).resolve().parents[1]
    workflow = project_root.joinpath(".github/workflows/ci.yml").read_text(encoding="utf-8")

    build_position = workflow.index("- name: Build and audit exact release wheel")
    install_position = workflow.index("- name: Install exact release wheel")
    import_position = workflow.index("- name: Import public and optional surfaces")

    assert build_position < install_position < import_position
    assert "python tools/release_wheel_audit.py dist" in workflow
    assert 'python -m pip install "${wheel_path}${{ matrix.extras }}"' in workflow
    assert 'requirement: "."' not in workflow
    assert 'requirement: ".[openai,' not in workflow


def test_sdist_excludes_workspace_local_release_data() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads(project_root.joinpath("pyproject.toml").read_text(encoding="utf-8"))
    exclude = set(pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"])

    assert "/.tmp" in exclude
    assert "/.tmp/**" in exclude
    assert "/studio-ui" in exclude
    assert "**/DX_NOTES.md" in exclude


def test_compiled_studio_assets_are_present_in_python_package() -> None:
    project_root = Path(__file__).resolve().parents[1]
    app_dir = project_root / "src/monoid_agent_kernel/reference/studio/web/dist"

    index = app_dir.joinpath("index.html").read_text(encoding="utf-8")
    assert '<div id="app"></div>' in index
    assert list(app_dir.glob("assets/*.js"))
    assert list(app_dir.glob("assets/*.css"))
