"""Conformance profile metadata and harness protocols."""

from .harness import (
    BackendHarness,
    CapabilityHarness,
    ConformanceHarness,
    ControlPlaneHarness,
    DurableRunnerHarness,
    GatewayHarness,
    MessageFabricHarness,
    MinimalAgentHarness,
    MultiAgentBackendHarness,
    SideEffectHarness,
    ToolAgentHarness,
)
from .contracts import (
    CapabilityBrokerFactory,
    CheckpointStoreFactory,
    run_capability_broker_contract,
    run_checkpoint_store_contract,
)
from .fixtures import CompatibilityFixture, load_compatibility_fixtures
from .report import (
    CONFORMANCE_REPORT_VERSION,
    ConformanceObservation,
    ConformanceReport,
    ConformanceRuleOutcome,
)
from .profiles import PROFILES, PROFILE_BY_ID, ProfileMetadata, get_profile

__all__ = [
    "BackendHarness",
    "CapabilityHarness",
    "ConformanceHarness",
    "ControlPlaneHarness",
    "DurableRunnerHarness",
    "GatewayHarness",
    "MessageFabricHarness",
    "MinimalAgentHarness",
    "MultiAgentBackendHarness",
    "SideEffectHarness",
    "ToolAgentHarness",
    "PROFILES",
    "PROFILE_BY_ID",
    "ProfileMetadata",
    "get_profile",
    "CONFORMANCE_REPORT_VERSION",
    "ConformanceObservation",
    "ConformanceReport",
    "ConformanceRuleOutcome",
    "CapabilityBrokerFactory",
    "CheckpointStoreFactory",
    "run_capability_broker_contract",
    "run_checkpoint_store_contract",
    "CompatibilityFixture",
    "load_compatibility_fixtures",
]
