from __future__ import annotations

import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest


EXPECTED_CONTRACTS_ALL = [
    "AgentLoop",
    "AgentRunSpec",
    "ModelConfig",
    "ModelRetryConfig",
    "ReasoningConfig",
    "GenerationConfig",
    "RunLimits",
    "AgentArtifact",
    "AgentRunResult",
    "AgentTurnResult",
    "Suspension",
    "TerminalOutcome",
    "InterruptionCause",
    "RetryEligibility",
    "DurableModelInvocation",
    "ModelEvidencePolicy",
    "RunCheckpoint",
    "CheckpointStore",
    "CheckedCheckpointStore",
    "CheckpointRecord",
    "ArtifactVersion",
    "DurableLoadResult",
    "DurableLoadStatus",
    "CompatibilityAlias",
    "CompatibilityArtifact",
    "PUBLIC_ARTIFACT_COMPATIBILITY",
    "PUBLIC_COMPATIBILITY_ALIASES",
    "compatibility_artifact",
    "compatibility_registry",
    "Workspace",
    "WorkspaceFactory",
    "FileEntry",
    "ChangedEntry",
    "SessionState",
    "AgentSession",
    "LoopSession",
    "SessionInspection",
    "SessionHealth",
    "CONTROL_PROTOCOL_VERSION",
    "ControlCommand",
    "ControlCommandType",
    "INBOX_PROTOCOL_VERSION",
    "InboxMessage",
    "OUTBOX_REQUEST_VERSION",
    "OutboxRequest",
    "OutboxReceipt",
    "OutboxSender",
    "ControlResult",
    "ControlDispatcher",
    "CAPABILITY_REQUEST_VERSION",
    "CAPABILITY_LEASE_VERSION",
    "CapabilityRequest",
    "CapabilityLease",
    "CapabilityDenial",
    "CapabilityPending",
    "CapabilityBroker",
    "AgentDefinition",
    "AgentRuntimeConfig",
    "BoundTool",
    "BoundToolCatalog",
    "PromptSpec",
    "RegistryToolRef",
    "RuntimeConfigProvider",
    "RuntimeConfigSource",
    "ToolBinding",
    "ToolSearchConfig",
    "ContextProvider",
    "TurnContext",
    # Model call kernel
    "InvocationContext",
    "CapturePolicy",
    "RedactionPolicy",
    "Redactor",
    "DefaultRedactor",
    "DEFAULT_SECRET_KEY_PARTS",
    "ModelCallReceipt",
    "ModelCallCapture",
    "ModelIOObserver",
    "ClosableModelIOObserver",
    "ModelIOSubscription",
    "ModelStreamChannel",
    "ModelStreamStatus",
    "ModelStreamContext",
    "ModelStreamDelta",
    "ModelStreamOutcome",
    "ModelStreamWriter",
    "ModelStreamObserver",
    "ModelStreamObserverFactory",
    "safe_open_model_stream",
    "ModelCallRunner",
    # Writing a typed ``settled_sink`` needs the argument type, and every other type
    # ``ModelCallRunner``'s public fields name is exported beside it.
    "SettledModelCall",
    "ValidatedCallRunner",
    "ValidatedCallResult",
    "AttemptDeltaConsumer",
    "AttemptStarted",
    "structured_output_support",
    "generation_support",
    "reasoning_support",
    "OutputValidator",
    "OutputValidatorBinding",
    "ValidationOutcome",
    "FinalOutputView",
    "OutputRetry",
    "OutputValidatorError",
    "ToolAuthorization",
    "ToolGuidance",
    "ToolQuota",
    "ToolScope",
    "ToolSearchEntry",
    "ToolSurfaceResolver",
    "ToolSurfaceSnapshot",
    "ContentPart",
    "TextPart",
    "ImagePart",
    "DocumentPart",
    "AudioPart",
    "VideoPart",
    "ModelAdapter",
    "AsyncModelAdapter",
    "StreamingModelAdapter",
    "MultimodalModelAdapter",
    "ProviderNamedModelAdapter",
    "ConfiguredModelAdapter",
    "AddressedModelAdapter",
    "report_provider_retried",
    "mark_provider_retried",
    "ModelRequest",
    "ModelTurn",
    "ToolCall",
    "ToolObservation",
    "RunStream",
    "ModelStreamChunk",
    "TextDelta",
    "ToolCallDelta",
    "TurnComplete",
    "SyncToolHandler",
    "AsyncToolHandler",
    "DynamicToolProvider",
    "ToolContext",
    "ToolHandler",
    "ToolProvider",
    "ToolRegistry",
    "ToolResult",
    "ToolSideEffect",
    "ToolSpec",
    "tool",
    "EVENT_SCHEMA_VERSION",
    "AgentEvent",
    "AgentEventLevel",
    "AgentEventType",
    "EventSink",
    "EventSequenceGap",
    "EventSubscription",
    "EventSubscriptionFrame",
    "SequenceCursor",
    "PermissionPolicy",
    "WebGatewayClient",
    "TaskExecutor",
    "ResultInjector",
    "TaskReporter",
    "SubagentDefinition",
    "SubagentTaskExecutor",
    "SkillDefinition",
    "SkillProvider",
]

REMOVED_PUBLIC_SURFACE_NAMES = [
    "WriterToken",
    "CommitResult",
    "ModelInvocationRecord",
    "StorageCapabilities",
    "FencedCheckpointStore",
    "FencedRunSink",
    "LocalFsCheckpointStore",
    "read_checkpoint",
    "write_checkpoint",
    "default_local_workspace_factory",
    "LEGAL_TRANSITIONS",
    "can_transition",
    "assert_transition",
    "state_from_suspension",
    "to_session_state",
    "parse_traceparent",
    "new_traceparent",
    "child_traceparent",
    "trace_id_of",
    "CapabilityVault",
    "AutoGrantBroker",
    "StaticRuntimeConfigProvider",
    "static_runtime_config",
    "coerce_runtime_config_provider",
    "compile_bound_tool_catalog",
    "generated_tool_bindings",
    "validate_runtime_config",
    "collect_runtime_config_issues",
    "DefaultToolSurfaceResolver",
    "is_inbox_envelope",
    "assemble_streamed_turn",
    "tool_ids",
    "list_builtin_tools",
    "agent_spawn_tool",
    "load_subagent_definitions",
    "parse_frontmatter",
    "load_skill_definitions",
    "FakeModelAdapter",
    "FakeStreamingModelAdapter",
    "GatewayModelAdapter",
    "OpenAIModelAdapter",
    "apply_package",
    "create_approval",
    "export_package",
    "import_package",
    "verify_package",
    "project_run_status",
    "narrate_event",
    "EventNarration",
    "OtelEventSink",
    "McpToolProvider",
    "McpError",
    "TaskManager",
    "public_job_artifact_for",
    "public_job_artifacts",
    "read_job_log_text",
    "request_job_cancel",
    "JsonlEventSink",
    "MemoryEventSink",
    "StatusJsonSink",
    "StdoutJsonlSink",
]


def test_contracts_public_surface_is_intentional() -> None:
    import monoid_agent_kernel.contracts as contracts

    assert contracts.__all__ == EXPECTED_CONTRACTS_ALL


def test_providers_package_exports_are_intentional() -> None:
    """Concrete adapters live in ``monoid_agent_kernel.providers``, never in ``contracts``
    (the removed-names census below bans them there). This is that surface's own census:
    the shipped adapters, plus the replay adapter with its typed miss beside it -- an
    error a caller must catch belongs where the class that raises it is found."""

    import monoid_agent_kernel.providers as providers

    assert providers.__all__ == [
        "FakeModelAdapter",
        "GatewayModelAdapter",
        "OpenAIModelAdapter",
        "ReplayMiss",
        "ReplayModelAdapter",
    ]


_OPTIONAL_ADAPTER_CAPABILITIES = ("supports_multimodal", "wire_image_encoding", "provider_name")


def test_base_adapter_protocols_do_not_require_optional_capabilities() -> None:
    """The base adapter protocols must declare only their required call method.

    A protocol member is required for structural typing even when the protocol body assigns
    it a default -- the default only reaches classes that explicitly inherit the protocol.
    Declaring an optional capability here therefore rejects a third-party adapter that
    implements ``next_turn`` and nothing else, which the engine accepts at runtime because it
    probes every capability with ``getattr`` and a default.
    """
    from monoid_agent_kernel.providers.base import AsyncModelAdapter, ModelAdapter

    for protocol in (ModelAdapter, AsyncModelAdapter):
        for name in _OPTIONAL_ADAPTER_CAPABILITIES:
            assert not hasattr(protocol, name), f"{protocol.__name__}.{name} is a protocol member"
            assert name not in getattr(protocol, "__annotations__", {}), (
                f"{protocol.__name__}.{name} is an annotated protocol member"
            )


def test_optional_capability_protocols_accept_classvar_implementations() -> None:
    """Each opt-in capability member stays a read-only property.

    Every shipped adapter declares these as ``ClassVar``. A protocol member annotated
    ``name: str`` demands an instance variable and rejects a ``ClassVar``; a read-only
    property is satisfied by a ``ClassVar``, an instance attribute, and a property alike.
    """
    from monoid_agent_kernel.providers.base import (
        ConfiguredModelAdapter,
        MultimodalModelAdapter,
        ProviderNamedModelAdapter,
    )

    declared = {
        MultimodalModelAdapter: ("supports_multimodal",),
        ProviderNamedModelAdapter: ("provider_name",),
        ConfiguredModelAdapter: ("config",),
    }
    for protocol, names in declared.items():
        for name in names:
            member = protocol.__dict__.get(name)
            assert isinstance(member, property), f"{protocol.__name__}.{name} is not a property"
            assert member.fset is None, f"{protocol.__name__}.{name} must be read-only"


def test_the_provider_name_protocol_admits_the_none_its_shipped_adapter_declares() -> None:
    """The declared type must accept the value the shipped adapter is *documented* to hold.

    ``ProviderNamedModelAdapter.provider_name`` declared ``str`` while its own docstring said
    omitting it means "do not tag" and ``GatewayModelAdapter.provider_name`` is ``str | None`` --
    the value a deployment sets when its gateway fronts an upstream with no reasoning artifacts.
    A protocol that rejects the adapter it was written for checks nothing.
    """
    import typing

    from monoid_agent_kernel.core.spec import ModelConfig
    from monoid_agent_kernel.providers.base import ProviderNamedModelAdapter
    from monoid_agent_kernel.providers.gateway import GatewayModelAdapter

    member = ProviderNamedModelAdapter.__dict__["provider_name"]
    declared = typing.get_type_hints(member.fget)["return"]
    assert type(None) in typing.get_args(declared), (
        f"provider_name is declared {declared!r}, which cannot hold the documented "
        '"do not tag" value'
    )
    assert GatewayModelAdapter(config=ModelConfig(), provider_name=None).provider_name is None


def test_a_capability_that_takes_an_argument_is_declared_as_a_method() -> None:
    """``AddressedModelAdapter`` is the one member of the family that is not a property.

    ``resolve_destination`` answers *for a given config*, so it takes an argument and a property
    cannot express it. Pinned rather than left implicit because the family's rule is the opposite
    one, and a member silently turned into a property would drop the parameter that makes it useful.
    """
    from monoid_agent_kernel.providers.base import AddressedModelAdapter

    member = AddressedModelAdapter.__dict__.get("resolve_destination")
    assert callable(member) and not isinstance(member, property)
    assert "config" in inspect.signature(member).parameters


def test_optional_capability_protocols_are_satisfied_by_shipped_adapters() -> None:
    """An opt-in protocol must stay satisfiable by the adapters this package ships.

    A capability protocol is only useful if it accepts the adapters that actually have the
    capability. Adding a member the shipped adapters leave to its default -- as
    ``wire_image_encoding`` is -- would reject every one of them.
    """
    from monoid_agent_kernel.core.spec import ModelConfig
    from monoid_agent_kernel.providers.base import (
        AddressedModelAdapter,
        ConfiguredModelAdapter,
        MultimodalModelAdapter,
        ProviderNamedModelAdapter,
    )
    from monoid_agent_kernel.providers.gateway import GatewayModelAdapter
    from monoid_agent_kernel.providers.openai import OpenAIModelAdapter

    expected = {
        MultimodalModelAdapter: (OpenAIModelAdapter, GatewayModelAdapter),
        # The gateway joined in X-3: it declares the UPSTREAM provider whose opaque reasoning
        # artifacts it relays, so the loop can tag them and replay them back through the hop.
        ProviderNamedModelAdapter: (OpenAIModelAdapter, GatewayModelAdapter),
        ConfiguredModelAdapter: (OpenAIModelAdapter, GatewayModelAdapter),
        AddressedModelAdapter: (GatewayModelAdapter,),
    }
    for protocol, adapters in expected.items():
        members = tuple(name for name in protocol.__dict__ if not name.startswith("_"))
        assert members, protocol.__name__
        for adapter in adapters:
            # Checked on an *instance*, not the class. ``config`` is a dataclass field, so it does
            # not exist on the class at all -- a class-level check would report every adapter as
            # failing to carry the config every one of them actually has. Methods and ``ClassVar``
            # capabilities answer the same either way, so one instance check covers the family.
            instance = adapter(config=ModelConfig())
            for name in members:
                assert hasattr(instance, name), (
                    f"{adapter.__name__} lacks {protocol.__name__}.{name}"
                )


def test_package_root_mirrors_contracts_surface() -> None:
    import monoid_agent_kernel as root
    import monoid_agent_kernel.contracts as contracts

    assert root.__all__ == contracts.__all__


def test_hosting_surface_is_narrow_and_explicit() -> None:
    import monoid_agent_kernel.hosting as hosting

    assert hosting.__all__ == [
        "WriterToken",
        "CommitResult",
        "ModelInvocationRecord",
        "StorageCapabilities",
        "FencedCheckpointStore",
        "FencedRunSink",
    ]


def test_public_conformance_surface_exposes_fenced_sink_contract() -> None:
    import monoid_agent_kernel.conformance as conformance

    expected = {
        "FencedRunSinkHarness",
        "FencedRunSinkHarnessFactory",
        "run_fenced_run_sink_contract",
    }

    assert expected <= set(conformance.__all__)
    assert all(hasattr(conformance, name) for name in expected)


def test_helpers_and_conveniences_are_not_root_or_contract_exports() -> None:
    import monoid_agent_kernel as root
    import monoid_agent_kernel.contracts as contracts

    for name in REMOVED_PUBLIC_SURFACE_NAMES:
        assert not hasattr(contracts, name), name
        assert not hasattr(root, name), name


def test_memory_surface_is_explicit_module_only() -> None:
    import monoid_agent_kernel as root
    import monoid_agent_kernel.contracts as contracts
    import monoid_agent_kernel.memory as memory

    assert memory.__all__ == [
        "MEMORY_ROOT",
        "MEMORY_TOOL_IDS",
        "MEMORY_SEARCH_TOOL_ID",
        "MEMORY_VIEW_TOOL_ID",
        "MEMORY_CREATE_TOOL_ID",
        "MEMORY_STR_REPLACE_TOOL_ID",
        "MEMORY_INSERT_TOOL_ID",
        "MEMORY_DELETE_TOOL_ID",
        "MEMORY_RENAME_TOOL_ID",
        "MemoryToolError",
        "MemoryStore",
        "MemoryProvider",
        "LocalFilesystemMemoryStore",
        "LocalFilesystemMemoryProvider",
    ]
    assert memory.MEMORY_TOOL_IDS == (
        "memory.search",
        "memory.view",
        "memory.create",
        "memory.str_replace",
        "memory.insert",
        "memory.delete",
        "memory.rename",
    )

    for name in memory.__all__:
        assert not hasattr(contracts, name), name
        assert not hasattr(root, name), name


def test_removed_names_remain_available_from_explicit_modules() -> None:
    from monoid_agent_kernel.core.agents import (
        coerce_runtime_config_provider,
        collect_runtime_config_issues,
        compile_bound_tool_catalog,
        generated_tool_bindings,
        static_runtime_config,
        validate_runtime_config,
    )
    from monoid_agent_kernel.core.capability import AutoGrantBroker, CapabilityVault
    from monoid_agent_kernel.core.checkpoint import (
        LocalFsCheckpointStore,
        read_checkpoint,
        write_checkpoint,
    )
    from monoid_agent_kernel.core.frontmatter import parse_frontmatter
    from monoid_agent_kernel.core.inbox import is_inbox_envelope
    from monoid_agent_kernel.core.lifecycle import (
        LEGAL_TRANSITIONS,
        assert_transition,
        can_transition,
        state_from_suspension,
        to_session_state,
    )
    from monoid_agent_kernel.core.packages import (
        apply_package,
        create_approval,
        export_package,
        import_package,
        verify_package,
    )
    from monoid_agent_kernel.core.projections import project_run_status
    from monoid_agent_kernel.core.trace_context import (
        child_traceparent,
        new_traceparent,
        parse_traceparent,
        trace_id_of,
    )
    from monoid_agent_kernel.core.tool_surface import DefaultToolSurfaceResolver
    from monoid_agent_kernel.mcp import McpError, McpToolProvider
    from monoid_agent_kernel.narration import EventNarration, narrate_event
    from monoid_agent_kernel.observability.otel import OtelEventSink
    from monoid_agent_kernel.providers.base import assemble_streamed_turn
    from monoid_agent_kernel.providers.fake import FakeModelAdapter
    from monoid_agent_kernel.providers.gateway import GatewayModelAdapter
    from monoid_agent_kernel.providers.openai import OpenAIModelAdapter
    from monoid_agent_kernel.recorder import (
        JsonlEventSink,
        MemoryEventSink,
        StatusJsonSink,
        StdoutJsonlSink,
    )
    from monoid_agent_kernel.skills import load_skill_definitions
    from monoid_agent_kernel.subagent_loader import load_subagent_definitions
    from monoid_agent_kernel.tasks import (
        TaskManager,
        public_job_artifact_for,
        public_job_artifacts,
        read_job_log_text,
        request_job_cancel,
    )
    from monoid_agent_kernel.tools import tool_ids
    from monoid_agent_kernel.tools.builtin import agent_spawn_tool
    from monoid_agent_kernel.tools.tool_ids import list_builtin_tools
    from monoid_agent_kernel.workspace.local import default_local_workspace_factory

    explicit_module_names = (
        LocalFsCheckpointStore,
        read_checkpoint,
        write_checkpoint,
        default_local_workspace_factory,
        LEGAL_TRANSITIONS,
        can_transition,
        assert_transition,
        state_from_suspension,
        to_session_state,
        parse_traceparent,
        new_traceparent,
        child_traceparent,
        trace_id_of,
        CapabilityVault,
        AutoGrantBroker,
        static_runtime_config,
        coerce_runtime_config_provider,
        compile_bound_tool_catalog,
        generated_tool_bindings,
        validate_runtime_config,
        collect_runtime_config_issues,
        DefaultToolSurfaceResolver,
        is_inbox_envelope,
        assemble_streamed_turn,
        tool_ids,
        list_builtin_tools,
        agent_spawn_tool,
        load_subagent_definitions,
        parse_frontmatter,
        load_skill_definitions,
        FakeModelAdapter,
        GatewayModelAdapter,
        OpenAIModelAdapter,
        apply_package,
        create_approval,
        export_package,
        import_package,
        verify_package,
        project_run_status,
        narrate_event,
        EventNarration,
        OtelEventSink,
        McpToolProvider,
        McpError,
        TaskManager,
        public_job_artifact_for,
        public_job_artifacts,
        read_job_log_text,
        request_job_cancel,
        JsonlEventSink,
        MemoryEventSink,
        StatusJsonSink,
        StdoutJsonlSink,
    )
    assert all(name is not None for name in explicit_module_names)


def test_audio_and_video_parts_are_public_contracts() -> None:
    import monoid_agent_kernel as root
    import monoid_agent_kernel.contracts as contracts
    from monoid_agent_kernel.core.content import AudioPart, VideoPart

    assert contracts.AudioPart is AudioPart
    assert contracts.VideoPart is VideoPart
    assert root.AudioPart is AudioPart
    assert root.VideoPart is VideoPart


def test_legacy_namespace_mirrors_narrow_contract_surface() -> None:
    for module_name in list(sys.modules):
        if module_name == "native_agent_runner" or module_name.startswith("native_agent_runner."):
            sys.modules.pop(module_name)

    import monoid_agent_kernel as root

    with pytest.warns(DeprecationWarning, match="monoid_agent_kernel"):
        legacy = importlib.import_module("native_agent_runner")

    assert legacy.__all__ == root.__all__
    assert legacy.AudioPart is root.AudioPart
    assert legacy.VideoPart is root.VideoPart
    assert not hasattr(legacy, "FakeModelAdapter")


def test_root_import_keeps_reference_and_optional_providers_lazy() -> None:
    root = Path(__file__).resolve().parents[1]
    src = str(root / "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    code = """
import sys
import typing
import monoid_agent_kernel
from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore

typing.get_type_hints(LocalFsCheckpointStore.capabilities.fget)
blocked = [
    name for name in sys.modules
    if name.startswith('monoid_agent_kernel.reference.')
    or name.startswith('monoid_agent_kernel.hosting')
    or name.startswith('monoid_agent_kernel.adapters')
    or name in {'openai', 'httpx', 'opentelemetry', 'dbos', 'psycopg', 'boto3', 'botocore', 'temporalio'}
    or name.startswith('openai.')
    or name.startswith('httpx.')
    or name.startswith('opentelemetry.')
    or name.startswith('dbos.')
    or name.startswith('psycopg.')
    or name.startswith('boto3.')
    or name.startswith('botocore.')
    or name.startswith('temporalio.')
]
if blocked:
    raise SystemExit('unexpected imports: ' + ', '.join(sorted(blocked)))
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_hosting_import_keeps_platform_implementations_out() -> None:
    root = Path(__file__).resolve().parents[1]
    src = str(root / "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    code = """
import sys
import monoid_agent_kernel.hosting
blocked = [
    name for name in sys.modules
    if name.startswith('monoid_agent_kernel.reference')
    or name.startswith('monoid_agent_kernel.adapters')
    or name in {'dbos', 'psycopg', 'psycopg2', 'redis', 'boto3', 'botocore', 'temporalio'}
    or name.startswith('dbos.')
    or name.startswith('psycopg.')
    or name.startswith('psycopg2.')
    or name.startswith('redis.')
    or name.startswith('boto3.')
    or name.startswith('botocore.')
    or name.startswith('temporalio.')
]
if blocked:
    raise SystemExit('unexpected imports: ' + ', '.join(sorted(blocked)))
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_adapter_namespace_imports_are_explicit_and_dependency_lazy() -> None:
    root = Path(__file__).resolve().parents[1]
    src = str(root / "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    code = """
import sys
import monoid_agent_kernel.adapters as adapters
import monoid_agent_kernel.adapters.postgres as postgres
import monoid_agent_kernel.adapters.object_store as object_store
import monoid_agent_kernel.adapters.temporal as temporal

assert adapters.__all__ == []
assert postgres.__all__ == []
assert object_store.__all__ == []
assert temporal.__all__ == []
blocked = [
    name for name in sys.modules
    if name in {'psycopg', 'psycopg2', 'boto3', 'botocore', 'temporalio'}
    or name.startswith('psycopg.')
    or name.startswith('psycopg2.')
    or name.startswith('boto3.')
    or name.startswith('botocore.')
    or name.startswith('temporalio.')
]
if blocked:
    raise SystemExit('unexpected imports: ' + ', '.join(sorted(blocked)))
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_the_relayed_provider_fields_are_keyword_only() -> None:
    """A field added to a shipped constructor must not rebind its positional arguments.

    ``llm_gateway_provider`` (and the loop-factory context's provider accessor) landed
    mid-dataclass, before fields embedders already pass positionally -- so an existing fifth
    positional ``model_adapter_factory`` would be stored as the relayed-provider string and the
    factory left unset, discovered only when ``resolve_relayed_provider`` calls ``.strip()`` on a
    callable. Keyword-only keeps the new knob beside the URL it describes without moving any
    argument that predates it.
    """
    import dataclasses

    from monoid_agent_kernel.reference.backend.loop_factory import BackendLoopFactoryContext
    from monoid_agent_kernel.reference.backend.service import RunnerBackend
    from monoid_agent_kernel.reference.studio.server import StudioConfig

    for cls, name in (
        (RunnerBackend, "llm_gateway_provider"),
        (BackendLoopFactoryContext, "llm_gateway_provider_provider"),
        (StudioConfig, "llm_gateway_provider"),
        # The recording switches followed the same insertion, for the same reason: each sits
        # beside the sidecar sibling whose semantics it shares rather than at the tail a
        # positional append would force. Listed by name because the tuple pins below catch a
        # dropped ``kw_only`` only as a confusing mid-tuple diff, and this says which rule broke.
        (RunnerBackend, "model_calls_file"),
        (RunnerBackend, "model_payload_file"),
        (BackendLoopFactoryContext, "model_calls_file_provider"),
        (BackendLoopFactoryContext, "model_payload_file_provider"),
    ):
        (fld,) = [f for f in dataclasses.fields(cls) if f.name == name]
        assert fld.kw_only, f"{cls.__name__}.{name} must be keyword-only to preserve positional order"


def test_v022_injected_constructor_dependencies_are_keyword_only() -> None:
    """New authority and lifecycle seams preserve every pre-v0.22 positional binding."""
    import dataclasses

    from monoid_agent_kernel.model_call import ModelCallRunner
    from monoid_agent_kernel.tasks import TaskManager

    for cls, name in (
        (ModelCallRunner, "current_write_authority"),
        (ModelCallRunner, "lifecycle_hook"),
        (TaskManager, "write_authority"),
    ):
        (fld,) = [item for item in dataclasses.fields(cls) if item.name == name]
        assert fld.kw_only, f"{cls.__name__}.{name} must preserve the positional ABI"


def test_positional_construction_keeps_its_pre_v021_meaning() -> None:
    """The behavioral half: the old positional shapes still mean what they meant.

    ``RunnerBackend``'s fifth positional was ``model_adapter_factory`` and ``StudioConfig``'s
    eleventh was ``stream_output_deltas``; both must still be.
    """
    from pathlib import Path as _Path

    from monoid_agent_kernel.reference.backend.service import RunnerBackend
    from monoid_agent_kernel.reference.studio.server import StudioConfig
    from monoid_agent_kernel.reference._shared.tokens import TokenManager

    def factory(*args: object, **kwargs: object) -> object:  # a stand-in adapter factory
        raise AssertionError("never called")

    backend = RunnerBackend(
        _Path("runs"), TokenManager(secret=b"s" * 32), (_Path("."),), "http://gateway", factory
    )
    assert backend.model_adapter_factory is factory
    assert backend.llm_gateway_provider == "openai"

    studio = StudioConfig(
        _Path("."), "127.0.0.1", 8799, "offline", _Path("runs"), None, False, True, None, None, False
    )
    assert studio.stream_output_deltas is False
    assert studio.llm_gateway_provider is None


def test_v022_positional_construction_keeps_its_pre_v022_meaning() -> None:
    from monoid_agent_kernel.model_call import ModelCallRunner
    from monoid_agent_kernel.tasks import TaskManager

    adapter = object()
    runner = ModelCallRunner(adapter, None, None, 0.05)

    assert runner.adapter is adapter
    assert runner.cancel_grace_s == 0.05
    assert runner.current_write_authority is None
    assert runner.lifecycle_hook is None

    restored_jobs = {}
    manager = TaskManager("run", object(), object(), object(), restored_jobs)

    assert manager.jobs is restored_jobs
    assert manager.write_authority.revoked is False


def test_stable_constructor_positional_order_is_append_only() -> None:
    """The positional signature of the shipped constructors is a compatibility surface.

    A field inserted mid-dataclass silently rebinds every positional argument after it -- no
    error, wrong object -- so growth must be appended or keyword-only. These literal pins make a
    mid-insert fail here; a legitimate append changes only the tail, which is a conscious,
    reviewable pin move. ``GatewayModelAdapter.provider_name`` is the one documented append.
    """
    import dataclasses

    from monoid_agent_kernel.providers.gateway import GatewayModelAdapter
    from monoid_agent_kernel.model_call import ModelCallRunner
    from monoid_agent_kernel.reference.backend.loop_factory import BackendLoopFactoryContext
    from monoid_agent_kernel.reference.backend.service import RunnerBackend
    from monoid_agent_kernel.reference.studio.server import StudioConfig
    from monoid_agent_kernel.tasks import TaskManager

    def positional(cls: type) -> tuple[str, ...]:
        return tuple(f.name for f in dataclasses.fields(cls) if f.init and not f.kw_only)

    assert positional(GatewayModelAdapter) == (
        "config", "gateway_url", "token", "token_env", "token_file", "token_provider",
        "provider_name",
    )
    assert positional(ModelCallRunner) == (
        "adapter", "current_adapter", "current_cancellation_token", "cancel_grace_s",
        "current_cancel_grace_s", "thread_name", "subscriptions", "settled_sink",
        "capture_request_preimage",
    )
    assert positional(TaskManager) == (
        "run_id", "workspace", "recorder", "permission_policy", "jobs",
    )
    assert positional(StudioConfig) == (
        "workspace", "host", "port", "provider", "run_root", "skills_directory", "mcp",
        "memory", "memory_directory", "env_file", "stream_output_deltas",
    )
    assert positional(BackendLoopFactoryContext) == (
        "run_root_provider", "llm_gateway_url_provider", "web_gateway_url_provider",
        "model_adapter_factory_provider", "token_manager_provider",
        "llm_gateway_token_ttl_s_provider", "checkpoint_store_provider",
        "emit_output_deltas_provider", "stream_model_calls_provider",
        "model_content_file_provider", "model_stream_observer_factories_provider",
        "extra_event_sink_factories_provider", "model_io_subscription_factories_provider",
        "subagent_definitions_provider", "tool_providers_provider",
        "context_providers_provider", "output_validators_provider",
        "capability_broker_factory_provider", "outbox_sender_factory_provider",
        "current_runtime_config", "record", "record_event", "persist_checkpoint_payload",
    )
    assert positional(RunnerBackend) == (
        "run_root", "token_manager", "allowed_workspace_roots", "llm_gateway_url",
        "model_adapter_factory", "web_gateway_url", "allowed_apply_roots", "run_token_ttl_s",
        "llm_gateway_token_ttl_s", "web_gateway_token_ttl_s", "task_callback_token_ttl_s",
        "idle_timeout_s", "max_session_lifetime_s", "max_turns", "task_wait_poll_s",
        "max_consecutive_turn_failures", "turn_retry", "emit_output_deltas",
        "stream_model_calls", "model_content_file", "model_stream_broker",
        "subagent_definitions", "extra_event_sink_factories",
        "model_io_subscription_factories", "tool_providers", "context_providers",
        "output_validators", "capability_broker_factory", "outbox_sender_factory",
        "outbox_max_attempts", "outbox_retry_base_s", "outbox_retry_factor",
        "outbox_retry_cap_s", "max_recover_attempts", "lease_ttl_s", "watchdog_interval_s",
        "max_message_bytes", "max_message_queue_depth", "max_concurrent_runs",
        "checkpoint_store", "lease_store", "command_store", "command_queue_limit",
        "command_claim_ttl_s",
    )
