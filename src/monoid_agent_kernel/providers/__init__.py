"""Model adapters, and the typed miss the replay adapter raises.

Concrete adapters live here on purpose -- ``contracts`` exports protocols and the removed-names
census there keeps implementations off that surface. ``ReplayMiss`` sits beside the adapter that
raises it because a caller catches them together.
"""

from monoid_agent_kernel.providers.fake import FakeModelAdapter
from monoid_agent_kernel.providers.gateway import GatewayModelAdapter
from monoid_agent_kernel.providers.openai import OpenAIModelAdapter
from monoid_agent_kernel.providers.replay import ReplayMiss, ReplayModelAdapter

__all__ = [
    "FakeModelAdapter",
    "GatewayModelAdapter",
    "OpenAIModelAdapter",
    "ReplayMiss",
    "ReplayModelAdapter",
]
