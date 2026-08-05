"""Every runtime config this repo SHIPS must be runnable against the upstreams it points at.

A shipped config is an instruction: ``monoid builder init`` writes one for a new adopter, and
``examples/runtime-config.json`` is the file README, ``docs/CLI.md``, the first-skill tutorial
and ``docs/OBSERVABILITY.md`` all tell a reader to pass. So a policy field left at its
fail-closed default in one of them is not a default -- it is a shipped refusal.

The B1 ``reasoning_applied`` echo is the field that made this concrete. A config that sets
*any* reasoning key is no longer default (``ReasoningConfig.is_default`` compares the fields
the base class declares), so the client demands proof from the hop; ``on_unsupported`` decides
what happens when no proof arrives. Under the inherited ``"fail"`` every turn against an
upstream that declares no ``reasoning_support`` -- the shipped ``monoid llm-gateway serve
--provider fake`` echo adapter, any pre-B1 gateway, most third-party factories -- is refused
``gateway_reasoning_not_applied`` before the first tool call.

The sweep ENUMERATES rather than naming files, on BOTH axes. Across sources: the same defect
was fixed three times in Python callers before anyone read the two JSON producers, and a
config added to ``examples/`` tomorrow inherits this pin for free. Across proofs: the
generation knob governs two more checkers with the identical shipped-refusal shape, so all
three are replayed rather than the one that happened to bite first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from monoid_agent_kernel.builder import _default_runtime_config
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.providers._common import (
    build_generation_payload,
    build_reasoning_payload,
)

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


def _proofs_a_shipped_config_would_demand(model: ModelConfig) -> list[tuple[str, Any, Any, str]]:
    """Every applied-parameters proof the gateway client would demand for ``model``.

    One entry per checker, each derived EXACTLY as the real enforcement site derives it -- see
    ``providers/gateway.py`` around the ``_check_*_applied`` block that runs after a turn comes
    back. Enumerated rather than sampled: the reasoning proof is the one that made this sweep
    necessary, but all three share the shipped-refusal shape (a policy field left at its
    fail-closed default in a file an adopter is told to pass), and a sweep that watched only
    the one already known to bite would have to be rediscovered for the next one.
    """

    from monoid_agent_kernel.providers.gateway import (
        _check_generation_applied,
        _check_reasoning_applied,
        _check_schema_applied,
    )

    return [
        # The real site passes the projection itself and lets the checker's ``if not requested``
        # early return be the gate -- there is no ``is_default`` test on this one.
        (
            "generation",
            _check_generation_applied,
            build_generation_payload(model.generation),
            model.generation.on_unsupported,
        ),
        # Request-shaped, not config-shaped: the real site asks ``request.output_schema is not
        # None``, and a runtime config carries no request. A shipped config can therefore never
        # send a schema, so this arm is False here by construction rather than by policy --
        # spelled out so a future reader does not read the pass as proof of a policy.
        ("schema", _check_schema_applied, False, model.generation.on_unsupported),
        # The one gated on ``is_default``, because the DEFAULT reasoning config still projects
        # a non-empty provider block -- payload truthiness would claim every call configured it.
        (
            "reasoning",
            _check_reasoning_applied,
            None if model.reasoning.is_default else build_reasoning_payload(model.reasoning),
            model.reasoning.on_unsupported,
        ),
    ]


@pytest.mark.parametrize("source", sorted(_shipped_runtime_configs()))
def test_a_shipped_runtime_config_can_still_answer_off_a_silent_upstream(source: str) -> None:
    """Reconstructed through the real codec, then asked every question the client asks.

    ``applied=None`` is the silent upstream: the shipped ``--provider fake`` echo adapter, any
    pre-B1 gateway, most third-party factories. Two of the three arms are born green today --
    neither shipped config sets a sampling control, and neither can carry an output schema --
    and that is the point. This is an ENUMERATION-BREADTH pin, not a bug reproduction: its job
    is that a knob added to a shipped config tomorrow is checked against the whole proof family
    rather than against the one member that happened to bite first.
    """

    model = ModelConfig.from_json(_shipped_runtime_configs()[source]["model"])
    for family, checker, requested, on_unsupported in _proofs_a_shipped_config_would_demand(model):
        try:
            checker(requested, on_unsupported, None)
        except ModelAdapterError as refused:
            pytest.fail(
                f"{source} refuses a non-declaring upstream on the {family} proof: {refused}\n"
                f"requested={requested!r} on_unsupported={on_unsupported!r}\n"
                f"reasoning={model.reasoning}\ngeneration={model.generation}\n"
                f"hint: a shipped config that configures {family} must state "
                'on_unsupported="omit" unless it ships a proving upstream too'
            )
