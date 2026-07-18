"""Machine-readable compatibility inventory for public versioned artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from monoid_agent_kernel.identifiers import legacy_namespaced_id, namespaced_id

ArtifactKind = Literal["wire", "durable", "reference"]
ReaderPolicy = Literal["checked", "strict", "json-schema", "permissive", "writer-only"]
AliasKind = Literal["python", "cli", "identifier", "environment", "token"]


@dataclass(frozen=True)
class CompatibilityArtifact:
    """One public versioned surface and the versions its current reader supports."""

    key: str
    kind: ArtifactKind
    current_writer: str
    supported_readers: tuple[str, ...]
    namespace_aliases: tuple[str, ...]
    reader_policy: ReaderPolicy
    source: tuple[str, ...]
    accepts_missing_version: bool = False
    notes: str = ""

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CompatibilityAlias:
    """One supported source, command, identifier, environment, or token alias."""

    key: str
    kind: AliasKind
    current: str
    alias: str
    status: Literal["compatibility", "deprecated"]
    removal_policy: str

    def to_json(self) -> dict[str, str]:
        return asdict(self)


def _monoid_artifact(
    name: str,
    *,
    kind: ArtifactKind,
    reader_policy: ReaderPolicy,
    source: tuple[str, ...],
    legacy_reader: bool,
    accepts_missing_version: bool = False,
    notes: str = "",
) -> CompatibilityArtifact:
    current = namespaced_id(name)
    legacy = legacy_namespaced_id(name)
    readers = () if reader_policy == "writer-only" else (current,)
    aliases = (legacy,) if legacy_reader else ()
    if legacy_reader and reader_policy != "writer-only":
        readers += aliases
    return CompatibilityArtifact(
        key=name.rsplit(".v", 1)[0],
        kind=kind,
        current_writer=current,
        supported_readers=readers,
        namespace_aliases=aliases,
        reader_policy=reader_policy,
        source=source,
        accepts_missing_version=accepts_missing_version,
        notes=notes,
    )


PUBLIC_ARTIFACT_COMPATIBILITY: tuple[CompatibilityArtifact, ...] = (
    # Core wire contracts.
    _monoid_artifact(
        "capability-request.v1",
        kind="wire",
        reader_policy="writer-only",
        source=("core/capability.py:CapabilityRequest.to_json",),
        legacy_reader=False,
        notes="Integrators consume the typed request; no public JSON parser is promised.",
    ),
    _monoid_artifact(
        "capability-lease.v1",
        kind="wire",
        reader_policy="strict",
        source=("core/capability.py:CapabilityLease.from_json",),
        legacy_reader=True,
        accepts_missing_version=True,
    ),
    _monoid_artifact(
        "control-command.v1",
        kind="wire",
        reader_policy="strict",
        source=("core/control.py:ControlCommand.from_json",),
        legacy_reader=True,
        accepts_missing_version=True,
        notes="Command and result envelopes share this protocol id.",
    ),
    _monoid_artifact(
        "inbox-message.v1",
        kind="wire",
        reader_policy="strict",
        source=("core/inbox.py:InboxMessage.from_json",),
        legacy_reader=True,
        accepts_missing_version=True,
    ),
    _monoid_artifact(
        "outbox-request.v1",
        kind="wire",
        reader_policy="strict",
        source=("core/outbox.py:OutboxRequest.from_json",),
        legacy_reader=True,
        accepts_missing_version=True,
    ),
    _monoid_artifact(
        "external-agent-envelope.v1",
        kind="wire",
        reader_policy="strict",
        source=("core/external_agent_envelope.py:ExternalAgentEnvelope.from_json",),
        legacy_reader=True,
    ),
    _monoid_artifact(
        "llm-turn.v1",
        kind="wire",
        reader_policy="strict",
        source=("reference/llm_gateway/service.py:_parse_turn_request",),
        legacy_reader=True,
    ),
    _monoid_artifact(
        "llm-turn-result.v1",
        kind="wire",
        reader_policy="permissive",
        source=("providers/gateway.py:_parse_gateway_response",),
        legacy_reader=True,
        accepts_missing_version=True,
        notes="The current client parses the response shape without enforcing protocol.",
    ),
    _monoid_artifact(
        "web-search.v1",
        kind="wire",
        reader_policy="permissive",
        source=("reference/web_gateway/service.py:WebGatewayBackend.handle_search",),
        legacy_reader=True,
        accepts_missing_version=True,
    ),
    _monoid_artifact(
        "web-search-result.v1",
        kind="wire",
        reader_policy="permissive",
        source=("web.py:WebGatewayClient.search",),
        legacy_reader=True,
        accepts_missing_version=True,
    ),
    _monoid_artifact(
        "web-fetch.v1",
        kind="wire",
        reader_policy="permissive",
        source=("reference/web_gateway/service.py:WebGatewayBackend.handle_fetch",),
        legacy_reader=True,
        accepts_missing_version=True,
    ),
    _monoid_artifact(
        "web-fetch-result.v1",
        kind="wire",
        reader_policy="permissive",
        source=("web.py:WebGatewayClient.fetch",),
        legacy_reader=True,
        accepts_missing_version=True,
    ),
    _monoid_artifact(
        "web-context.v1",
        kind="wire",
        reader_policy="permissive",
        source=("reference/web_gateway/service.py:WebGatewayBackend.handle_context",),
        legacy_reader=True,
        accepts_missing_version=True,
    ),
    _monoid_artifact(
        "web-context-result.v1",
        kind="wire",
        reader_policy="permissive",
        source=("web.py:WebGatewayClient.context",),
        legacy_reader=True,
        accepts_missing_version=True,
    ),
    # Durable Core and Reference-backend artifacts.
    _monoid_artifact(
        "checkpoint.v1",
        kind="durable",
        reader_policy="checked",
        source=("core/checkpoint.py:CHECKPOINT_CODEC",),
        legacy_reader=True,
    ),
    _monoid_artifact(
        "backend-run.v1",
        kind="durable",
        reader_policy="checked",
        source=("core/durable_metadata.py:RUN_METADATA_CODEC",),
        legacy_reader=True,
    ),
    _monoid_artifact(
        "event.v1",
        kind="durable",
        reader_policy="json-schema",
        source=("core/schemas.py:EVENT_SCHEMA",),
        legacy_reader=True,
    ),
    _monoid_artifact(
        "manifest.v1",
        kind="durable",
        reader_policy="json-schema",
        source=("core/schemas.py:MANIFEST_SCHEMA",),
        legacy_reader=True,
    ),
    _monoid_artifact(
        "workspace-base.v1",
        kind="durable",
        reader_policy="json-schema",
        source=("core/schemas.py:WORKSPACE_BASE_SCHEMA",),
        legacy_reader=True,
    ),
    _monoid_artifact(
        "workspace-index.v1",
        kind="durable",
        reader_policy="json-schema",
        source=("core/schemas.py:WORKSPACE_INDEX_SCHEMA",),
        legacy_reader=True,
    ),
    _monoid_artifact(
        "proposal.v2",
        kind="durable",
        reader_policy="json-schema",
        source=("core/schemas.py:PROPOSAL_SCHEMA",),
        legacy_reader=True,
    ),
    _monoid_artifact(
        "background-job.v1",
        kind="durable",
        reader_policy="json-schema",
        source=("core/schemas.py:JOB_SCHEMA",),
        legacy_reader=True,
    ),
    _monoid_artifact(
        "task.v1",
        kind="durable",
        reader_policy="writer-only",
        source=("tasks.py:HostedTask.to_json",),
        legacy_reader=False,
        notes="Public task.json projection; recovery uses the checkpoint task shape.",
    ),
    _monoid_artifact(
        "proposal-package.v1",
        kind="durable",
        reader_policy="json-schema",
        source=("core/schemas.py:PACKAGE_SCHEMA", "core/packages.py:verify_package"),
        legacy_reader=True,
    ),
    _monoid_artifact(
        "approval.v1",
        kind="durable",
        reader_policy="json-schema",
        source=("core/schemas.py:APPROVAL_SCHEMA", "core/packages.py:apply_package"),
        legacy_reader=True,
    ),
    _monoid_artifact(
        "apply-result.v1",
        kind="durable",
        reader_policy="json-schema",
        source=("core/schemas.py:APPLY_RESULT_SCHEMA",),
        legacy_reader=True,
    ),
    _monoid_artifact(
        "failure.v1",
        kind="durable",
        reader_policy="permissive",
        source=("reference/backend/recovery.py:RecoveryService.write_failure_bundle",),
        legacy_reader=True,
        accepts_missing_version=True,
        notes="Diagnostics consume fields without enforcing schema_version.",
    ),
    _monoid_artifact(
        "command-inbox.v1",
        kind="durable",
        reader_policy="strict",
        source=("reference/command_inbox.py:COMMAND_ENVELOPE_VERSION",),
        legacy_reader=False,
        notes="Reference command stores persist sanitized envelopes without bearer credentials.",
    ),
    _monoid_artifact(
        "command-receipt.v1",
        kind="wire",
        reader_policy="writer-only",
        source=("reference/command_inbox.py:COMMAND_RECEIPT_VERSION",),
        legacy_reader=False,
        notes="HTTP and Python receipt projection for queued command state and results.",
    ),
    CompatibilityArtifact(
        key="conformance-report",
        kind="reference",
        current_writer=namespaced_id("conformance-report.v1"),
        supported_readers=(
            namespaced_id("conformance-report.v1"),
            namespaced_id("conformance-report.v2"),
        ),
        namespace_aliases=(),
        reader_policy="checked",
        source=("conformance/report.py:decode_conformance_report",),
        notes=(
            "Reader-first rollout: the checked reader migrates v1 reports to the v2 typed model "
            "before the external runner begins writing v2."
        ),
    ),
    CompatibilityArtifact(
        key="conformance-evidence",
        kind="reference",
        current_writer=namespaced_id("conformance-evidence.v1"),
        supported_readers=(namespaced_id("conformance-evidence.v1"),),
        namespace_aliases=(),
        reader_policy="strict",
        source=("conformance/provenance.py:verify_conformance_evidence",),
        notes="Exact-byte normalized evidence with size and SHA-256 verification.",
    ),
    _monoid_artifact(
        "conformance-fixtures.v1",
        kind="reference",
        reader_policy="strict",
        source=("conformance/fixtures/compatibility-v1.json",),
        legacy_reader=False,
        notes="Packaged historical fixtures consumed by load_compatibility_fixtures().",
    ),
    # Reference Studio export/projection formats use their own namespace.
    CompatibilityArtifact(
        key="studio-chat",
        kind="reference",
        current_writer="studio.chat.v1",
        supported_readers=("studio.chat.v1",),
        namespace_aliases=(),
        reader_policy="strict",
        source=("reference/studio/chat_projection.py:ChatProjection.response",),
    ),
    CompatibilityArtifact(
        key="studio-chat-message",
        kind="reference",
        current_writer="studio.chat.message.v1",
        supported_readers=("studio.chat.message.v1",),
        namespace_aliases=(),
        reader_policy="permissive",
        source=("reference/studio/chat_projection.py:ChatProjection.read",),
        accepts_missing_version=True,
        notes="JSONL projection reader skips malformed records and does not gate by version.",
    ),
)


_MAJOR_RELEASE_NOTICE = "major release after notice in at least two preceding minor releases"

PUBLIC_COMPATIBILITY_ALIASES: tuple[CompatibilityAlias, ...] = (
    CompatibilityAlias(
        key="python-package",
        kind="python",
        current="monoid_agent_kernel",
        alias="native_agent_runner",
        status="deprecated",
        removal_policy=_MAJOR_RELEASE_NOTICE,
    ),
    CompatibilityAlias(
        key="cli-entry-point",
        kind="cli",
        current="monoid",
        alias="native-agent",
        status="deprecated",
        removal_policy=_MAJOR_RELEASE_NOTICE,
    ),
    CompatibilityAlias(
        key="identifier-namespace",
        kind="identifier",
        current="monoid.*",
        alias="native-agent-runner.*",
        status="compatibility",
        removal_policy="major release plus a migration path for every retained durable artifact",
    ),
    CompatibilityAlias(
        key="environment-prefix",
        kind="environment",
        current="MONOID_*",
        alias="NAR_*",
        status="deprecated",
        removal_policy=_MAJOR_RELEASE_NOTICE,
    ),
    CompatibilityAlias(
        key="token-issuer",
        kind="token",
        current="monoid",
        alias="native-agent-runner",
        status="compatibility",
        removal_policy=_MAJOR_RELEASE_NOTICE,
    ),
    CompatibilityAlias(
        key="token-header-type",
        kind="token",
        current="MAK",
        alias="NAR",
        status="compatibility",
        removal_policy=_MAJOR_RELEASE_NOTICE,
    ),
    CompatibilityAlias(
        key="backend-audience",
        kind="token",
        current="monoid.backend",
        alias="native-agent-runner.backend",
        status="compatibility",
        removal_policy=_MAJOR_RELEASE_NOTICE,
    ),
    CompatibilityAlias(
        key="task-callback-audience",
        kind="token",
        current="monoid.task-callback",
        alias="native-agent-runner.task-callback",
        status="compatibility",
        removal_policy=_MAJOR_RELEASE_NOTICE,
    ),
)


def compatibility_registry() -> dict[str, object]:
    """Return a serialization-safe snapshot for tooling and release checks."""

    return {
        "artifacts": tuple(artifact.to_json() for artifact in PUBLIC_ARTIFACT_COMPATIBILITY),
        "aliases": tuple(alias.to_json() for alias in PUBLIC_COMPATIBILITY_ALIASES),
    }


def compatibility_artifact(key: str) -> CompatibilityArtifact:
    """Look up one registered artifact by stable ledger key."""

    for artifact in PUBLIC_ARTIFACT_COMPATIBILITY:
        if artifact.key == key:
            return artifact
    raise KeyError(key)
