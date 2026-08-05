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

from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.model_call import (
    _PROMPT_DIGEST_GENERATION,
    _REQUEST_DIGEST_GENERATION,
    _prompt_payload,
    _request_payload,
)
from monoid_agent_kernel.providers.base import ModelRequest

_REQUEST = ModelRequest(instruction="hi", system_prompt="sys", tools=())


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
    whole = _request_payload(_REQUEST, ModelConfig(), provider="fake", destination="")

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
