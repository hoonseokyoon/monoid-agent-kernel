"""What identifies a model call: the replay key's field list, its generation, and its domain.

W6-0 (dx-note 2026-08-02-v0.21-contract-replay-scope.md §Track B, §5 decision 3). The replay
key must be reproducible from what a record can hold, so the deployment a call was routed
through leaves the key and becomes recorded metadata, and the key's field set becomes a
declared list rather than a serialized internal object.

**Mutation gate.** `_model_identity` is the single projection every consumer surface below
reads through. Mutating it must turn all four red -- if one survives, the binding is broken:

  1. the generation-1 literal (`tests/test_generation_config.py`),
  2. the omit-when-absent pins (`generation` here, `output_schema` in
     `tests/test_output_schema_delivery.py`),
  3. the transport-policy exclusion matrix in this file,
  4. the identifying-field inclusion matrix in this file.

A literal alone cannot see *conditional* inclusion -- `if model.timeout_s != 600: ...` keeps
the default-config key stable and changes every other one -- which is why 3 and 4 are
parameterized matrices rather than a single golden value.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from monoid_agent_kernel.core.spec import (
    GenerationConfig,
    ModelConfig,
    ModelRetryConfig,
    ReasoningConfig,
)
from monoid_agent_kernel.model_call import (
    _PROMPT_DIGEST_GENERATION,
    _REQUEST_DIGEST_GENERATION,
    ModelCallRunner,
    _digest,
    _model_identity,
    _prompt_payload,
    _request_payload,
)
from monoid_agent_kernel.providers.base import ModelRequest

_REQUEST = ModelRequest(instruction="hi", system_prompt="sys", tools=())


def _key(model: ModelConfig, *, provider: str = "fake") -> str:
    return _digest(_request_payload(_REQUEST, model, provider=provider))


def _terms(model: ModelConfig) -> dict[str, object]:
    payload = _request_payload(_REQUEST, model, provider="fake")
    return dict(payload[_REQUEST_DIGEST_GENERATION])


# --- the two digests name their own domains -------------------------------------------


def test_the_two_digests_cannot_share_a_key_space() -> None:
    """Each payload is one wrapper key, and the two wrappers differ.

    Before W6-0 the separation was incidental: `_request_payload` starts from the prompt terms
    and adds always-present keys, so a request payload could never *happen* to equal a prompt
    payload. That is a property of today's field lists rather than a rule, and it would have
    ended the first time a key the request payload adds became conditional. Domain separation
    on the whole preimage makes it a rule.
    """

    prompt = _prompt_payload(_REQUEST)
    whole = _request_payload(_REQUEST, ModelConfig(), provider="fake")

    assert set(prompt) == {_PROMPT_DIGEST_GENERATION}
    assert set(whole) == {_REQUEST_DIGEST_GENERATION}
    assert _PROMPT_DIGEST_GENERATION != _REQUEST_DIGEST_GENERATION


def test_the_generations_are_namespaced_ids() -> None:
    """The tag is spelled the way every other version tag in this repo is spelled.

    `namespaced_id` is what makes a bump legible: `.v1` -> `.v2` is simultaneously the
    canonicalization-change rule of `docs/CONTRACTS.md` and the deliberate disowning of a
    corpus recorded under the old rules.
    """

    assert _PROMPT_DIGEST_GENERATION == "monoid.model-prompt-digest.v1"
    assert _REQUEST_DIGEST_GENERATION == "monoid.model-request-digest.v1"


# --- surface 3: the transport-policy exclusion matrix ----------------------------------
#
# What a literal cannot see. A full revert to `model.to_json()` moves the generation-1 literal
# instantly, but `if model.timeout_s != 600: terms["timeout_s"] = ...` leaves the default-config
# key untouched and changes every other one. Only a matrix over non-default values catches that.


@pytest.mark.parametrize(
    "model",
    (
        pytest.param(ModelConfig(timeout_s=30), id="timeout_s"),
        pytest.param(ModelConfig(gateway_url="http://elsewhere.invalid/x"), id="gateway_url"),
        pytest.param(ModelConfig(retry=ModelRetryConfig(max_attempts=5)), id="retry.max_attempts"),
        pytest.param(ModelConfig(retry=ModelRetryConfig(jitter_s=0.9)), id="retry.jitter_s"),
        pytest.param(
            ModelConfig(retry=ModelRetryConfig(retry_on=("gateway_timeout",))), id="retry.retry_on"
        ),
    ),
)
def test_transport_policy_does_not_move_the_replay_key(model: ModelConfig) -> None:
    """How a call is carried is not what it asks for.

    These knobs never reach the provider: the gateway wire emits only model/reasoning/generation
    and each hop owns its own transport policy. Leaving them in the key meant an ops change --
    raising a timeout, widening a retry set -- silently invalidated every recorded key on the
    fleet, which is the structure the W7 track would have walked straight into.
    """

    assert _key(model) == _key(ModelConfig())


def test_the_projection_carries_no_transport_terms() -> None:
    assert set(_terms(ModelConfig())["model"]) == {"model", "reasoning"}


# --- surface 4: the identifying-field inclusion matrix ---------------------------------
#
# The counterweight. Guard, not red-first evidence: green before and after, because it states
# what must NOT have been dropped while the exclusions above were being made.


@pytest.mark.parametrize(
    "model",
    (
        pytest.param(ModelConfig(model="gpt-5.5-mini"), id="model"),
        pytest.param(ModelConfig(reasoning=ReasoningConfig(effort="high")), id="reasoning.effort"),
        pytest.param(ModelConfig(reasoning=ReasoningConfig(summary="auto")), id="reasoning.summary"),
        pytest.param(
            ModelConfig(reasoning=ReasoningConfig(on_unsupported="omit")),
            id="reasoning.on_unsupported",
        ),
        pytest.param(
            ModelConfig(generation=GenerationConfig(temperature=0.1)), id="generation.temperature"
        ),
        pytest.param(ModelConfig(generation=GenerationConfig(top_p=0.5)), id="generation.top_p"),
        pytest.param(
            ModelConfig(generation=GenerationConfig(max_output_tokens=16)),
            id="generation.max_output_tokens",
        ),
        pytest.param(
            ModelConfig(generation=GenerationConfig(on_unsupported="omit")),
            id="generation.on_unsupported",
        ),
    ),
)
def test_every_identifying_model_field_moves_the_replay_key(model: ModelConfig) -> None:
    assert _key(model) != _key(ModelConfig())


def test_a_generation_free_config_omits_the_block_entirely() -> None:
    """Omit-when-absent, one level down. See `_GENERATION_1_REQUEST_DIGEST` for the literal."""

    assert "generation" not in _terms(ModelConfig())["model"]
    assert "generation" in _terms(ModelConfig(generation=GenerationConfig(top_p=0.5)))["model"]


# --- the provider slot ------------------------------------------------------------------


class _Declaring:
    provider_name = "openai"


class _Silent:
    pass


def test_the_endpoint_does_not_move_the_replay_key() -> None:
    """Where a call was sent is not what it asked for.

    The signature carries no `destination` at all now, so this states the property structurally:
    the payload has no term a destination could occupy, and the terms it does have are the ones a
    record can hold. The endpoint's own fact lives on the receipt, beside the key rather than
    inside it.
    """

    assert "destination" not in _terms(ModelConfig())


def test_a_gateway_relayed_openai_call_and_a_direct_openai_call_share_a_key() -> None:
    """The transport a call arrived over is not what identifies it.

    A gateway relaying OpenAI and a direct OpenAI adapter answer the same request the same way,
    so a corpus recorded against one is replayable against the other. Both declare `openai`; the
    `ModelConfig.provider` that used to ride the key separately said `gateway` for one of them.
    """

    relayed = ModelConfig(provider="gateway")
    direct = ModelConfig(provider="openai")

    assert _key(relayed, provider="openai") == _key(direct, provider="openai")


def test_two_adapters_that_declare_nothing_are_still_told_apart_by_their_configured_provider() -> (
    None
):
    """Guard. The provider slot is `resolved_provider_name`, not the raw declaration.

    Dropping `ModelConfig.provider` and keeping only what an adapter declares would have collided
    a fake adapter with a gateway built `provider_name=None` -- both declare nothing, and the
    config was the only thing left telling them apart. `resolved_provider_name` is the repo's one
    expression for "who actually served this", declaration else config, and it is what the slot
    reads.
    """

    from monoid_agent_kernel.providers.base import resolved_provider_name

    fake = ModelConfig(provider="fake")
    gateway = ModelConfig(provider="gateway")

    assert resolved_provider_name(_Silent(), fake) == "fake"
    assert resolved_provider_name(_Silent(), gateway) == "gateway"
    assert resolved_provider_name(_Declaring(), gateway) == "openai"
    assert _key(fake, provider="fake") != _key(gateway, provider="gateway")


# --- the projection reflects over nothing ----------------------------------------------
#
# The one claim no behavioural test can express. A matrix says which fields move the key; it
# cannot say *how* the projection decided, and a reflective implementation would satisfy every
# matrix above while re-opening the hazard the hand-listing closed. Split over source text --
# rather than reading the repo directly -- so the mutation test below can show the pin moves.


_PROJECTION_ALLOWLIST = {
    "model",
    "reasoning",
    "generation",
    "effort",
    "summary",
    "on_unsupported",
    "temperature",
    "top_p",
    "max_output_tokens",
    "is_default",
}


def _assert_the_projection_reflects_over_nothing(source: str) -> None:
    tree = ast.parse(textwrap.dedent(source))

    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    } | {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    reflective = called & {"to_json", "asdict", "vars", "fields", "getattr", "dir"}
    assert reflective == set(), {
        "reflective_calls": sorted(reflective),
        "hint": "a serialized or enumerated config makes every field a co-author of the key",
    }

    read = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
    }
    assert read <= _PROJECTION_ALLOWLIST, {
        "outside_the_allowlist": sorted(read - _PROJECTION_ALLOWLIST),
        "hint": "a term the key covers must be listed here, deliberately",
    }


def test_the_request_identity_projection_is_hand_listed() -> None:
    _assert_the_projection_reflects_over_nothing(inspect.getsource(_model_identity))


def test_the_projection_pin_moves_on_the_edits_it_claims_to_catch() -> None:
    """A structural pin is a claim about which edits turn it red, and that claim is testable."""

    source = inspect.getsource(_model_identity)
    _assert_the_projection_reflects_over_nothing(source)

    def _mutant(old: str, new: str) -> str:
        mutated = source.replace(old, new)
        assert mutated != source, {"hint": "the mutation did not apply; the anchor moved"}
        return mutated

    for old, new in (
        ('"model": model.model,', '"model": model.gateway_url,'),
        ('"model": model.model,', '"model": model.to_json(),'),
        ('"effort": model.reasoning.effort,', '"timeout_s": model.timeout_s,'),
    ):
        with pytest.raises(AssertionError):
            _assert_the_projection_reflects_over_nothing(_mutant(old, new))


def test_the_replay_key_is_taken_after_normalization() -> None:
    """Order, not content: the key covers the normalized request, so normalization is key material.

    Nothing else in this file would notice if that order flipped -- every test here builds its own
    payload from an already-normalized request. A change to `normalize_model_request` silently
    rekeys every recorded call, and this is the only place that says so.
    """

    source = inspect.getsource(ModelCallRunner.acall)
    normalized_at = source.index("request = normalize_model_request(request)")
    keyed_at = source.index("_request_payload(")

    assert normalized_at < keyed_at, "the key must cover the request the adapter is handed"
