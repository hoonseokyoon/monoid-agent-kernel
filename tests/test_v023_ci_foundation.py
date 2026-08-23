from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_ci_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("v023_ci", ROOT / "tools/v023_ci.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_campaign_lock_and_tracked_service_artifacts_are_consistent() -> None:
    module = _load_ci_module()
    lock = module.validate_lock()

    assert lock["services"]["postgres16"]["major"] == 16
    assert lock["services"]["postgres18"]["major"] == 18
    assert lock["services"]["temporal_cli"]["version"] == "v1.8.2"
    assert lock["services"]["temporal_cli"]["embedded_server"] == "1.31.2"
    assert lock["python_dependencies"]["botocore"]["exact"] == "1.43.78"
    temporal_archive = module.temporal_archive_spec(lock)
    assert temporal_archive["version"] == "v1.8.2"
    assert len(temporal_archive["sha256"]) == 64
    assert temporal_archive["filename"].endswith(".tar.gz")


def test_temporal_cli_preparation_rejects_a_corrupt_cached_archive(tmp_path: Path) -> None:
    module = _load_ci_module()
    lock = module.validate_lock()
    archive = module.temporal_archive_spec(lock)
    tmp_path.joinpath(archive["filename"]).write_bytes(b"not-the-locked-archive")

    with pytest.raises(ValueError, match="cached Temporal CLI archive checksum mismatch"):
        module.prepare_temporal_cli(lock, tmp_path)


def test_exact_sdk_verification_fails_when_a_locked_distribution_cannot_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_ci_module()
    lock = module.validate_lock()
    monkeypatch.setattr(
        module.importlib.metadata,
        "version",
        lambda name: lock["python_dependencies"][name]["exact"],
    )
    imported: list[str] = []

    def import_module(name: str) -> object:
        imported.append(name)
        if name == "temporalio":
            raise ImportError("simulated broken native dependency")
        return object()

    monkeypatch.setattr(module.importlib, "import_module", import_module)
    with pytest.raises(ValueError, match="cannot be imported: temporalio"):
        module.verify_installed(lock)
    assert imported == ["psycopg", "psycopg_pool", "boto3", "botocore", "temporalio"]


@pytest.mark.parametrize(
    ("head_ref", "label", "expected"),
    [
        ("codex/v0.23-pr01-ci-adapter-foundation", "ci:combined", "combined"),
        ("codex/v0.23-pr02-postgres-authority", "ci:postgres", "postgres"),
        ("codex/v0.23-pr05-object-store", "ci:objectstore", "objectstore"),
        ("codex/v0.23-pr09-temporal", "ci:temporal", "temporal"),
    ],
)
def test_service_profile_is_derived_from_the_campaign_plan(
    head_ref: str,
    label: str,
    expected: str,
) -> None:
    module = _load_ci_module()
    lock = module.validate_lock()
    assert module.resolve_profile(lock, head_ref=head_ref, labels=[label]) == expected


def test_service_profile_rejects_missing_multiple_and_wrong_labels() -> None:
    module = _load_ci_module()
    lock = module.validate_lock()

    with pytest.raises(ValueError, match="exactly one"):
        module.resolve_profile(lock, head_ref="codex/v0.23-pr01-ci", labels=[])
    with pytest.raises(ValueError, match="exactly one"):
        module.resolve_profile(
            lock,
            head_ref="codex/v0.23-pr01-ci",
            labels=["ci:combined", "ci:postgres"],
        )
    with pytest.raises(ValueError, match="requires ci:combined"):
        module.resolve_profile(
            lock,
            head_ref="codex/v0.23-pr01-ci",
            labels=["ci:postgres"],
        )


@pytest.mark.parametrize(
    ("head_ref", "labels", "expected"),
    [
        ("codex/v0.23-production-adapters", [], "combined"),
        ("codex/v0.23-production-adapters", ["ci:combined"], "combined"),
        ("develop", [], "combined"),
        ("feature/unrelated", [], "core"),
        ("feature/service-change", ["ci:postgres"], "postgres"),
    ],
)
def test_terminal_and_non_campaign_branches_have_total_profile_resolution(
    head_ref: str,
    labels: list[str],
    expected: str,
) -> None:
    module = _load_ci_module()
    lock = module.validate_lock()
    assert module.resolve_profile(lock, head_ref=head_ref, labels=labels) == expected


def test_terminal_branch_rejects_a_profile_weaker_than_combined() -> None:
    module = _load_ci_module()
    lock = module.validate_lock()
    with pytest.raises(ValueError, match="requires ci:combined"):
        module.resolve_profile(
            lock,
            head_ref="codex/v0.23-production-adapters",
            labels=["ci:core"],
        )


def test_service_modules_do_not_skip_selected_sdk_import_failures() -> None:
    for path in (
        ROOT / "tests/service/test_postgres_service.py",
        ROOT / "tests/service/test_object_store_service.py",
        ROOT / "tests/service/test_temporal_service.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "pytest.importorskip" not in source
        assert "allow_module_level=True" in source


def test_ci_workflows_encode_the_fast_full_and_cancel_boundaries() -> None:
    fast = (ROOT / ".github/workflows/pr-fast.yml").read_text(encoding="utf-8")
    full = (ROOT / ".github/workflows/pr-full.yml").read_text(encoding="utf-8")
    cancel = (ROOT / ".github/workflows/pr-full-cancel.yml").read_text(encoding="utf-8")
    integration = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "synchronize" in fast
    assert "fast-pr-${{ github.event.pull_request.number || github.ref }}" in fast
    assert "types: [opened, reopened, ready_for_review, labeled, unlabeled]" in full
    assert "ready_for_review" in full
    assert "workflow_dispatch" in full
    assert "name: Classify L2 trigger" in full
    assert "mode=dormant" in full
    assert "mode=diagnostic" in full
    assert "mode=required" in full
    assert full.count("contains(fromJSON(") == 2
    assert full.count('"ci:core","ci:postgres","ci:objectstore","ci:temporal","ci:combined"') == 2
    assert "name: L2 ${{ needs.trigger.outputs.mode || 'dormant' }} gate" in full
    assert "Record dormant trigger without qualification authority" in full
    assert "needs: [trigger, profile, core, service]" in full
    assert "full-${{" in full and "'dormant-'" in full
    assert ".[dev,openai,reference-dbos]" in full
    assert '(unit or contract) and serial and not service' in full
    assert "pr-${{ github.event.pull_request.number || inputs.pr_number || github.ref }}" in full
    assert "      - name: Record qualified merge candidate\n        if: always()" not in full
    assert (
        "      - uses: actions/upload-artifact@v4\n"
        "        if: always()\n"
        "        with:\n"
        "          name: v023-${{ needs.profile.outputs.profile }}-evidence"
        not in full
    )
    assert "          name: v023-${{ needs.profile.outputs.profile }}-evidence\n          path:" in full
    assert "converted_to_draft" in cancel
    assert "full-pr-${{ github.event.pull_request.number || github.ref }}" in cancel
    assert "pull_request:" not in integration
    assert "codex/v0.23-production-adapters" in integration
    assert "      - name: Record integration evidence\n        if: always()" not in integration
    assert (
        "      - uses: actions/upload-artifact@v4\n"
        "        if: always()\n"
        "        with:\n"
        "          name: v023-combined-evidence"
        not in integration
    )
    assert "          name: v023-combined-evidence\n          path:" in integration


def test_ci_helper_writes_public_safe_evidence(tmp_path: Path) -> None:
    output = tmp_path / "evidence.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/v023_ci.py"),
            "write-evidence",
            "--head-sha",
            "a" * 40,
            "--merge-sha",
            "b" * 40,
            "--profile",
            "combined",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["head_sha"] == "a" * 40
    assert evidence["merge_sha"] == "b" * 40
    serialized = output.read_text(encoding="utf-8").lower()
    assert "password" not in serialized
    assert "secret" not in serialized
    assert "access_key" not in serialized
