from __future__ import annotations

import json
import tomllib
from pathlib import Path

from monoid_agent_kernel import contracts
from monoid_agent_kernel.conformance.fixtures import load_compatibility_fixtures
from monoid_agent_kernel.conformance.provenance import CONFORMANCE_EVIDENCE_VERSION
from monoid_agent_kernel.conformance.report import (
    CONFORMANCE_REPORT_V1,
    CONFORMANCE_REPORT_V2,
    CONFORMANCE_REPORT_VERSION,
)
from monoid_agent_kernel.core.capability import (
    CAPABILITY_LEASE_VERSION,
    CAPABILITY_REQUEST_VERSION,
)
from monoid_agent_kernel.core.checkpoint import SCHEMA_VERSION as CHECKPOINT_SCHEMA_VERSION
from monoid_agent_kernel.core.compatibility import (
    PUBLIC_ARTIFACT_COMPATIBILITY,
    PUBLIC_COMPATIBILITY_ALIASES,
    compatibility_artifact,
    compatibility_registry,
)
from monoid_agent_kernel.core.control import CONTROL_PROTOCOL_VERSION
from monoid_agent_kernel.core.durable_metadata import RUN_METADATA_SCHEMA_VERSION
from monoid_agent_kernel.core.events import EVENT_SCHEMA_VERSION
from monoid_agent_kernel.core.external_agent_envelope import EXTERNAL_AGENT_ENVELOPE_VERSION
from monoid_agent_kernel.core.inbox import INBOX_PROTOCOL_VERSION
from monoid_agent_kernel.core.manifest import MANIFEST_SCHEMA_VERSION
from monoid_agent_kernel.core.outbox import OUTBOX_REQUEST_VERSION
from monoid_agent_kernel.core.packages import (
    APPLY_RESULT_SCHEMA_VERSION,
    APPROVAL_SCHEMA_VERSION,
    PACKAGE_SCHEMA_VERSION,
)
from monoid_agent_kernel.core.schemas import (
    APPLY_RESULT_SCHEMA,
    APPROVAL_SCHEMA,
    EVENT_SCHEMA,
    JOB_SCHEMA,
    MANIFEST_SCHEMA,
    PACKAGE_SCHEMA,
    PROPOSAL_SCHEMA,
    WORKSPACE_BASE_SCHEMA,
    WORKSPACE_INDEX_SCHEMA,
)
from monoid_agent_kernel.core.workspace_index import WORKSPACE_INDEX_SCHEMA_VERSION
from monoid_agent_kernel.reference.llm_gateway.service import LLM_TURN_PROTOCOL_VERSION
from monoid_agent_kernel.reference.command_inbox import (
    COMMAND_ENVELOPE_VERSION,
    COMMAND_RECEIPT_VERSION,
)
from monoid_agent_kernel.reference._shared.tokens import (
    LEGACY_TOKEN_HEADER_TYPE,
    TOKEN_HEADER_TYPE,
)
from monoid_agent_kernel.reference.studio.chat_projection import (
    CHAT_MESSAGE_SCHEMA_VERSION,
    CHAT_SCHEMA_VERSION,
)
from monoid_agent_kernel.identifiers import (
    BACKEND_AUDIENCE,
    LEGACY_BACKEND_AUDIENCE,
    LEGACY_NAMESPACE,
    LEGACY_TASK_CALLBACK_AUDIENCE,
    LEGACY_TOKEN_ISSUER,
    TASK_CALLBACK_AUDIENCE,
    TOKEN_ISSUER,
)
from monoid_agent_kernel.workspace.local import WORKSPACE_BASE_SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "COMPATIBILITY.md"


def test_registry_is_unique_serializable_and_canonically_namespaced() -> None:
    artifacts = PUBLIC_ARTIFACT_COMPATIBILITY

    assert len(artifacts) == 34
    assert len({artifact.key for artifact in artifacts}) == len(artifacts)
    assert len({artifact.current_writer for artifact in artifacts}) == len(artifacts)
    json.dumps(compatibility_registry(), sort_keys=True)

    for artifact in artifacts:
        assert artifact.current_writer in artifact.active_writers
        assert len(set(artifact.active_writers)) == len(artifact.active_writers)
        if artifact.current_writer.startswith("monoid."):
            assert artifact.current_writer in artifact.supported_readers or (
                artifact.reader_policy == "writer-only" and not artifact.supported_readers
            )
            for alias in artifact.namespace_aliases:
                assert alias == artifact.current_writer.replace(
                    "monoid.", "native-agent-runner.", 1
                )
        assert all(".v" in version for version in artifact.supported_readers)
        assert all(".v" in version for version in artifact.active_writers)

    assert len(PUBLIC_COMPATIBILITY_ALIASES) == 8
    assert len({alias.key for alias in PUBLIC_COMPATIBILITY_ALIASES}) == 8


def test_registry_matches_source_owned_version_constants() -> None:
    expected = {
        "capability-request": CAPABILITY_REQUEST_VERSION,
        "capability-lease": CAPABILITY_LEASE_VERSION,
        "control-command": CONTROL_PROTOCOL_VERSION,
        "inbox-message": INBOX_PROTOCOL_VERSION,
        "outbox-request": OUTBOX_REQUEST_VERSION,
        "external-agent-envelope": EXTERNAL_AGENT_ENVELOPE_VERSION,
        "llm-turn": LLM_TURN_PROTOCOL_VERSION,
        "checkpoint": CHECKPOINT_SCHEMA_VERSION,
        "backend-run": RUN_METADATA_SCHEMA_VERSION,
        "event": EVENT_SCHEMA_VERSION,
        "manifest": MANIFEST_SCHEMA_VERSION,
        "workspace-base": WORKSPACE_BASE_SCHEMA_VERSION,
        "workspace-index": WORKSPACE_INDEX_SCHEMA_VERSION,
        "proposal-package": PACKAGE_SCHEMA_VERSION,
        "approval": APPROVAL_SCHEMA_VERSION,
        "apply-result": APPLY_RESULT_SCHEMA_VERSION,
        "conformance-report": CONFORMANCE_REPORT_VERSION,
        "conformance-evidence": CONFORMANCE_EVIDENCE_VERSION,
        "conformance-fixtures": "monoid.conformance-fixtures.v1",
        "command-inbox": COMMAND_ENVELOPE_VERSION,
        "command-receipt": COMMAND_RECEIPT_VERSION,
        "studio-chat": CHAT_SCHEMA_VERSION,
        "studio-chat-message": CHAT_MESSAGE_SCHEMA_VERSION,
    }

    assert {key: compatibility_artifact(key).current_writer for key in expected} == expected


def test_packaged_compatibility_fixture_schema_matches_registry() -> None:
    assert load_compatibility_fixtures()
    assert compatibility_artifact("conformance-fixtures").current_writer == (
        "monoid.conformance-fixtures.v1"
    )


def test_conformance_report_default_writer_retains_the_v1_migration_path() -> None:
    report = compatibility_artifact("conformance-report")

    assert report.current_writer == CONFORMANCE_REPORT_VERSION
    assert report.reader_policy == "checked"
    assert report.supported_readers == (
        CONFORMANCE_REPORT_V1,
        CONFORMANCE_REPORT_V2,
    )
    assert report.active_writers == (
        CONFORMANCE_REPORT_V1,
        CONFORMANCE_REPORT_V2,
    )


def test_json_schema_reader_versions_match_registry() -> None:
    schemas = {
        "event": EVENT_SCHEMA,
        "manifest": MANIFEST_SCHEMA,
        "workspace-base": WORKSPACE_BASE_SCHEMA,
        "workspace-index": WORKSPACE_INDEX_SCHEMA,
        "proposal": PROPOSAL_SCHEMA,
        "background-job": JOB_SCHEMA,
        "proposal-package": PACKAGE_SCHEMA,
        "approval": APPROVAL_SCHEMA,
        "apply-result": APPLY_RESULT_SCHEMA,
    }

    for key, schema in schemas.items():
        registered = compatibility_artifact(key)
        assert registered.reader_policy == "json-schema"
        assert tuple(schema["properties"]["schema_version"]["enum"]) == registered.supported_readers


def test_registry_source_locations_exist() -> None:
    for artifact in PUBLIC_ARTIFACT_COMPATIBILITY:
        assert artifact.source
        for location in artifact.source:
            relative_path = location.split(":", 1)[0]
            assert (ROOT / "src" / "monoid_agent_kernel" / relative_path).is_file(), location


def test_alias_registry_matches_package_and_identifier_configuration() -> None:
    aliases = {alias.key: (alias.current, alias.alias) for alias in PUBLIC_COMPATIBILITY_ALIASES}
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["scripts"][aliases["cli-entry-point"][0]] == "monoid_agent_kernel.cli:main"
    assert project["scripts"][aliases["cli-entry-point"][1]] == "monoid_agent_kernel.cli:main"
    assert aliases["identifier-namespace"] == ("monoid.*", f"{LEGACY_NAMESPACE}.*")
    assert aliases["token-issuer"] == (TOKEN_ISSUER, LEGACY_TOKEN_ISSUER)
    assert aliases["token-header-type"] == (TOKEN_HEADER_TYPE, LEGACY_TOKEN_HEADER_TYPE)
    assert aliases["backend-audience"] == (BACKEND_AUDIENCE, LEGACY_BACKEND_AUDIENCE)
    assert aliases["task-callback-audience"] == (
        TASK_CALLBACK_AUDIENCE,
        LEGACY_TASK_CALLBACK_AUDIENCE,
    )


def test_documented_ledger_rows_match_registry_in_order() -> None:
    text = LEDGER.read_text(encoding="utf-8")
    table = text.split("<!-- compatibility-registry:start -->", 1)[1].split(
        "<!-- compatibility-registry:end -->", 1
    )[0]
    documented: list[tuple[str, str]] = []
    for line in table.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        documented.append((cells[0].strip("`"), cells[2].strip("`")))

    assert documented == [
        (artifact.key, artifact.current_writer) for artifact in PUBLIC_ARTIFACT_COMPATIBILITY
    ]
    alias_table = text.split("<!-- compatibility-aliases:start -->", 1)[1].split(
        "<!-- compatibility-aliases:end -->", 1
    )[0]
    documented_aliases: list[tuple[str, str, str]] = []
    for line in alias_table.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        documented_aliases.append((cells[0].strip("`"), cells[1].strip("`"), cells[2].strip("`")))
    assert documented_aliases == [
        (alias.key, alias.current, alias.alias) for alias in PUBLIC_COMPATIBILITY_ALIASES
    ]
    assert "`native_agent_runner`" in text
    assert "`native-agent`" in text
    assert "## Upgrade playbook" in text
    assert "## Rollback playbook" in text
    assert "## Schema changes and existing runs" in text


def test_registry_is_exported_from_stable_contract_surface() -> None:
    assert contracts.PUBLIC_ARTIFACT_COMPATIBILITY is PUBLIC_ARTIFACT_COMPATIBILITY
    assert (
        contracts.compatibility_artifact("checkpoint").current_writer == CHECKPOINT_SCHEMA_VERSION
    )
    assert contracts.compatibility_registry() == compatibility_registry()
    assert contracts.PUBLIC_COMPATIBILITY_ALIASES is PUBLIC_COMPATIBILITY_ALIASES
