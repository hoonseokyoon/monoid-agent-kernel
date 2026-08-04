"""Every runtime config this repo SHIPS must be runnable against the upstreams it points at.

A shipped config is an instruction: ``monoid builder init`` writes one for a new adopter, and
``examples/runtime-config.json`` is the file README, ``docs/CLI.md``, the first-skill tutorial
and ``docs/OBSERVABILITY.md`` all tell a reader to pass. So a policy field left at its
fail-closed default in one of them is not a default -- it is a shipped refusal.

The B1 ``reasoning_applied`` echo is the field that made this concrete. A config that sets
*any* reasoning key is no longer default (``ReasoningConfig.is_default`` is dataclass
equality), so the client demands proof from the hop; ``on_unsupported`` decides what happens
when no proof arrives. Under the inherited ``"fail"`` every turn against an upstream that
declares no ``reasoning_support`` -- the shipped ``monoid llm-gateway serve --provider fake``
echo adapter, any pre-B1 gateway, most third-party factories -- is refused
``gateway_reasoning_not_applied`` before the first tool call.

The sweep ENUMERATES rather than naming files: the same defect was fixed three times in
Python callers before anyone read the two JSON producers, and a config added to ``examples/``
tomorrow inherits this pin for free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from monoid_agent_kernel.builder import _default_runtime_config
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.providers._common import build_reasoning_payload

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _shipped_runtime_configs() -> dict[str, dict[str, Any]]:
    """Every runtime config the package or the repo hands an adopter, by source name."""

    configs: dict[str, dict[str, Any]] = {"builder init scaffold": _default_runtime_config()}
    for path in sorted(EXAMPLES_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:  # pragma: no cover - a non-JSON .json file is its own bug
            continue
        # "Parses as a runtime config" is the membership test, so ``run-spec.json`` and any
        # future fixture beside it stay out without being named here.
        if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
            configs[f"examples/{path.name}"] = payload
    return configs


def test_the_sweep_sees_both_shipped_config_producers() -> None:
    """Guard the enumeration itself: a glob that quietly matches nothing proves nothing."""

    found = _shipped_runtime_configs()
    assert "builder init scaffold" in found
    assert [name for name in found if name.startswith("examples/")], sorted(found)


@pytest.mark.parametrize("source", sorted(_shipped_runtime_configs()))
def test_a_shipped_runtime_config_can_still_answer_off_a_silent_upstream(source: str) -> None:
    """Reconstructed through the real codec, then asked the question the client asks."""

    from monoid_agent_kernel.providers.gateway import _check_reasoning_applied

    model = ModelConfig.from_json(_shipped_runtime_configs()[source]["model"])
    # Exactly the two lines the gateway client runs at each of its three enforcement sites.
    requested = None if model.reasoning.is_default else build_reasoning_payload(model.reasoning)
    try:
        _check_reasoning_applied(requested, model.reasoning.on_unsupported, None)
    except ModelAdapterError as refused:
        pytest.fail(
            f"{source} refuses a non-declaring upstream: {refused}\n"
            f"reasoning={model.reasoning}\n"
            "hint: a shipped config that configures reasoning must state "
            'on_unsupported="omit" unless it ships a proving upstream too'
        )
