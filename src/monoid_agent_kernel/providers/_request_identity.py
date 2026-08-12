"""What identifies one model call: the replay key's terms, their digests, and their generations.

Moved DOWN from ``model_call`` (W6-4b) so ``providers/replay.py`` can share the exact functions
the runner stamps receipts with. The direction is forced by the layering ``model_call``'s own
docstring documents: that module sits *above* ``providers`` and imports ``providers.base``, so a
replay adapter could not import the derivation from there without a cycle. ``model_call``
re-imports every name below, which keeps its callers and their tests untouched; the pins that
bind the projection (``tests/test_request_identity.py``) moved WITH the functions -- a move, not
a loosening -- and the re-export identity test there is what rules out a quiet mirror growing in
either home.

Private module by the ``_common.py`` precedent: the *contract* surface for replay is the adapter
and its corpus, not the key arithmetic. Everything here is pure over its arguments except the two
probe-reads ``effective_model_for`` and ``replay_lookup`` perform against an adapter, which are
the runner's own probes, inherited verbatim.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import Any

from monoid_agent_kernel.core._util import CANONICAL_JSON_ENCODER
from monoid_agent_kernel.core.json_ingress import normalize_unicode_scalars
from monoid_agent_kernel.core.model_io import MAX_MODEL_PAYLOAD_BYTES
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.identifiers import namespaced_id
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    normalize_model_config,
    resolved_provider_name,
)
from monoid_agent_kernel.tools.base import ToolSpec

_PROMPT_DIGEST_GENERATION = namespaced_id("model-prompt-digest.v1")
_REQUEST_DIGEST_GENERATION = namespaced_id("model-request-digest.v1")
"""The domain each digest is taken in, and the generation of the rules that produced it.

Domain separation on the *whole* preimage, applied in the payload builders rather than in
:func:`_digest` -- the same place :func:`core.model_io.content_digest` applies its shape key, and
for the same reason. Two jobs, one tag:

* **The two digests stop sharing a key space.** Their separation used to be incidental:
  :func:`_request_payload` starts from the prompt terms and adds keys that are always present, so a
  request payload could not *happen* to equal a prompt payload. That is a property of today's field
  lists, not a rule, and it would have ended the first time one of those added keys became
  conditional.
* **A rules change is announced instead of silent.** `docs/CONTRACTS.md` states it as the second
  stability rule: adding an omitted-when-unset field is not a generation change, but changing what
  the payload is made of is, and a generation change takes the domain with it. Bumping `.v1` to
  `.v2` disowns every key recorded under the old rules in one edit, which is the only honest way to
  retire a corpus that a change has invalidated.

Not inside :func:`_digest`: that function's contract is "the canonical-JSON digest of `payload`",
its tests are about the encoder, and a prefix fed to the hasher would bypass
`CANONICAL_JSON_ENCODER` and break the shared-instance invariant its comment in `core/_util.py`
guards.
"""


def _prompt_terms(request: ModelRequest) -> dict[str, Any]:
    """The assembled prompt, as the thing `prompt_digest` identifies.

    Returned unwrapped so :func:`_request_payload` can build on it without nesting a domain inside
    a domain; :func:`_prompt_payload` is the wrapped form that is actually hashed.

    Tool definitions and generation settings are deliberately absent: the question this digest
    answers is "did the model see the same conversation twice", which must stay true when a tool is
    added to the surface or the temperature changes around it.

    Everything that *constitutes* the conversation is present, including the by-reference shape.
    A request may carry its history as `messages`, or as a `previous_turn_handle` naming history the
    provider holds plus the `observations` produced since -- and in that second shape those two
    fields **are** the prompt. Hashing only `messages` made every by-reference continuation collide
    with every other, which is the ordinary case for a gateway client, not an edge one.

    `messages` keeps `None` apart from `()`, because the wire does. Both wire-writing shipped
    adapters select the request shape with `messages is not None` -- an empty tuple sends an empty conversation and
    drops the instruction, `None` sends the instruction or the handle. `or ()` read the field for
    emptiness when the field's own meaning is presence, so two requests the provider answers
    differently were handed one replay key.
    """

    return {
        "system_prompt": request.system_prompt,
        "instruction": request.instruction,
        "messages": None if request.messages is None else list(request.messages),
        "previous_turn_handle": request.previous_turn_handle,
        "observations": [observation.to_json() for observation in request.observations],
    }


def _prompt_payload(request: ModelRequest) -> dict[str, Any]:
    """:func:`_prompt_terms` in its own digest domain. See :data:`_PROMPT_DIGEST_GENERATION`."""

    return {_PROMPT_DIGEST_GENERATION: _prompt_terms(request)}


@dataclass(frozen=True)
class _DigestResult:
    """What the encoder said about one payload: a key, or a named refusal.

    ``status`` is the value ``ModelCallReceipt.digest_status`` records for the request key, minus
    the two members only other actors can produce (``withheld`` is a capture policy's, and
    ``not_reached`` is the boundary check's). The pair travels together so the two fields that
    describe one decision cannot be computed twice and disagree.
    """

    digest: str = ""
    status: str = "absent"
    preimage: bytes | None = None


def _encoded_digest(payload: dict[str, Any], *, want_preimage: bool = False) -> _DigestResult:
    """The canonical-JSON digest of `payload`, or a named refusal when no key can be issued.

    Streamed through the standard encoder rather than normalized first. Four rounds of review went
    into a hand-written normalizer that reshaped anything into something hashable -- stringified
    mapping keys, `<cycle:n>` markers, `repr` for values JSON had no form for -- and each fix
    revealed another way for two different requests to land on one digest: a `repr` shared by
    unrelated objects, a marker a caller could type as ordinary text, a lone surrogate that passed
    the type check and then failed at encode.

    The premise was wrong. A `ModelRequest` carries what will be sent to a provider over HTTP, so a
    payload canonical JSON cannot carry was never going to reach a model either. Reshaping it into
    something hashable invented an identity for a request that does not exist, and inventing
    identities is the one thing a replay key must not do.

    So: hash what encodes, and issue no key for what does not. Refusing is safe -- the call still
    happens, it simply is not replayable -- while a fabricated key returns the wrong call. An empty
    digest means *no key*; two unreplayable calls both carry `""` and are not thereby the same call.

    Every encoder failure means the same thing here -- no key -- so the clause catches `Exception`
    rather than a list of types. Naming them was itself a bug found four times over: circular
    references, unencodable primitives, unserializable objects, then a `dict` subclass whose
    `items()` raises. The question is never which exception the encoder chose, only whether it
    finished. `BaseException` is deliberately not caught: a cancellation or an interrupt is not a
    statement about the payload.

    Output is capped so a payload built from shared references cannot expand without bound; passing
    the cap also means no key, since a prefix would stand for the whole. The cap is
    :data:`~monoid_agent_kernel.core.model_io.MAX_MODEL_PAYLOAD_BYTES`, set to the same number as
    the default message-log bound so the band between a smaller digest cap and that bound -- which
    once shipped calls that transmitted successfully and silently had no replay key -- cannot
    reopen by drift. It is not the *wire's* bound: it covers the whole identity payload, so a
    request can clear every run limit and still exceed it. Exceeding it is a *named* refusal
    (``too_large``), distinct from ``absent``: one says the payload is too big to key, the other
    says it is malformed, and a status that conflated them said neither. A refusal reports the first reason the
    encoder hit; a payload both hostile and oversized is not diagnosed twice.
    """

    hasher = hashlib.sha256()
    encoded = 0
    parts: list[bytes] = []
    try:
        for chunk in CANONICAL_JSON_ENCODER.iterencode(payload):
            raw = chunk.encode("utf-8")
            encoded += len(raw)
            if encoded > MAX_MODEL_PAYLOAD_BYTES:
                return _DigestResult(status="too_large")
            hasher.update(raw)
            if want_preimage:
                parts.append(raw)
    except Exception:
        return _DigestResult(status="absent")
    return _DigestResult(
        digest=hasher.hexdigest(),
        status="ok",
        # The exact bytes the hasher consumed -- not a re-encoding of `payload`, which a caller
        # could have mutated by the time anyone looked. A refusal never carries bytes: a preimage
        # for a key that was not issued would be an identity for a call that does not exist.
        preimage=b"".join(parts) if want_preimage else None,
    )


def _digest(payload: dict[str, Any]) -> str:
    """The canonical-JSON digest of `payload`, or `""` when no key was issued.

    The refusal-collapsing view of :func:`_encoded_digest`, for the callers that only file or
    compare keys -- the prompt digest has no status field to feed, and the tests that pin encoder
    behavior pin it through here. The request-key path calls `_encoded_digest` directly, because
    it must record *why* a key is missing, not just that it is.
    """

    return _encoded_digest(payload).digest


def _tool_payload(spec: ToolSpec) -> dict[str, Any]:
    """A tool definition as the replay key sees it: every field except the ones JSON cannot carry.

    Read off the dataclass rather than listing fields by hand, so a field added to `ToolSpec` joins
    the digest automatically instead of quietly falling out of it. Reducing a tool to its `id` --
    which this did first -- made two requests offering the same id with different descriptions or
    input schemas produce the same replay key, though the provider was sent different tool
    definitions.

    Erring toward *more* than the wire carries is deliberate and asymmetric: an over-sensitive
    replay key costs a miss and a re-run, an under-sensitive one hands back the wrong call.
    """

    return {
        field_.name: getattr(spec, field_.name)
        for field_ in fields(spec)
        if not callable(getattr(spec, field_.name))
    }


def _model_identity(model: ModelConfig) -> dict[str, Any]:
    """The model config as the replay key sees it: a declared list, never a serialized object.

    `model.to_json()` used to go in whole, which made every consumer of that serializer a
    co-author of the replay key. A field added to `ModelConfig` for any reason rekeyed the entire
    corpus -- and `ModelRetryConfig` is scheduled to gain one, so this was not a hypothetical.
    The rule the list encodes:

        what the provider is asked for goes in the key; how the call is carried does not.

    So `timeout_s`, `retry` and `gateway_url` are absent. None of them reaches a provider: the
    gateway wire emits only model/reasoning/generation, and each hop owns its own transport
    policy. Their presence in the key was inherited from `to_json` emitting everything, not
    chosen, and it meant an ops change nobody thinks of as a contract change silently invalidated
    every recorded key on a fleet.

    Hand-listed all the way down, not just at the top: calling `reasoning.to_json()` here would
    move the same rekey hazard one level deeper, where the next added field would find it.

    `on_unsupported` is the one genuine close call, in both blocks. It is a client *acceptance*
    policy -- it cannot change what a provider returns, only whether the client keeps the answer --
    which argues transport. It goes in anyway because it rides the wire: the reference gateway
    rebuilds an upstream config from it, so a hop set to `omit` really can send a different
    upstream request than one set to `fail`. `_tool_payload` states the governing asymmetry: an
    over-sensitive replay key costs a miss and a re-run, an under-sensitive one hands back the
    wrong call.

    Known limit, inherited rather than introduced: for an adapter exposing no `config`,
    `effective_model_for` substitutes `ModelConfig()`, so this projects the default model name for
    a call that ran under something else. The receipt cannot say "unknown" and that is a limit of
    the field, not a distinction being drawn -- but this list now *promises* these fields identify
    the request, which the whole-object serialization never did explicitly.
    """

    identity: dict[str, Any] = {
        "model": model.model,
        "reasoning": {
            "effort": model.reasoning.effort,
            "summary": model.reasoning.summary,
            "on_unsupported": model.reasoning.on_unsupported,
        },
    }
    # Omit-when-absent, the same rule and the same reason as `output_schema` below: a config that
    # never set a sampling control keeps the key it had before the block existed.
    if not model.generation.is_default:
        identity["generation"] = {
            "temperature": model.generation.temperature,
            "top_p": model.generation.top_p,
            "max_output_tokens": model.generation.max_output_tokens,
            "on_unsupported": model.generation.on_unsupported,
        }
    return identity


def _request_payload(request: ModelRequest, model: ModelConfig, *, provider: str) -> dict[str, Any]:
    """The whole request, as the thing `request_digest` identifies -- the replay key.

    `model` is the *effective* config, resolved by the caller, not `request.model`. The request's is
    optional and the shipped adapters fall back to their own, so hashing `request.model or
    ModelConfig()` gave two calls on differently-configured adapters the same replay key. It enters
    through `_model_identity` rather than as `to_json()`; see there for what that list excludes.

    `provider` is the provider that ACTUALLY served the call -- `resolved_provider_name`, the
    adapter's declaration else the config's -- because that is what decides whether two identical
    requests get the same kind of answer. Reading only the declaration would collide a fake adapter
    with a gateway built without one; reading only the config would separate a direct call from a
    gateway relaying the same upstream, which is exactly the pair a corpus wants to share a key.

    **The destination is not here, and its absence is the point.** Where a call was sent used to be
    hashed in, on the reasoning that the same request answered by a different service is a different
    call -- which is true, and was the wrong place to say it. The value is deliberately never
    recorded, so no record could reconstruct the preimage: a key taken over it could not be
    recomputed, could not be verified, and a miss could not be told from a defect. It is now a fact
    beside the key (`ModelCallReceipt.destination_status` and `destination_digest`), where it can be
    compared without being disclosed. The distinction it used to draw was not lost; it moved from an
    opaque hash into something a consumer can actually read.
    """

    terms = _prompt_terms(request)
    terms["tools"] = [_tool_payload(spec) for spec in request.tools]
    terms["model"] = _model_identity(model)
    terms["provider"] = provider
    # Omit-when-absent (the W5 digest stability rule): a schema-free request keeps the
    # replay key it had before this field existed; setting a schema changes the key, which
    # is correct -- constrained and unconstrained calls are different requests.
    if request.output_schema is not None:
        terms["output_schema"] = request.output_schema
    return {_REQUEST_DIGEST_GENERATION: terms}


def effective_model_for(
    request: ModelRequest, adapter: Any
) -> tuple[ModelConfig, ModelConfig | None]:
    """The config recorded for a call on `adapter`, and an explicit dispatch override when one exists.

    The body ``ModelCallRunner._effective_model`` had; the runner now delegates here so the
    receipt's resolution and a replay lookup's are one implementation, not two that agree today.

    `ModelRequest.model` is optional and every shipped adapter falls back to its own
    `self.config`, so a receipt built from the request alone reports the *default* model no
    matter which one served the call -- a fabricated audit field, not merely a missing one.
    Probed via `getattr` and type-checked: see `ConfiguredModelAdapter`.
    """

    if request.model is not None:
        normalized = normalize_model_config(request.model) or ModelConfig()
        return normalized, normalized
    # Tolerant of a raising probe for the reason `_resolved_destination` gives: a replay key is
    # bookkeeping, and an adapter that cannot answer must not thereby lose its call. Plain
    # `getattr(..., None)` swallowed only `AttributeError`, so a `config` property that raised
    # anything else took the whole call down.
    #
    # A raising probe and an absent one both land on the default here, because the return type
    # admits nothing else -- the receipt cannot say "unknown". That is a known limit of this
    # field, not a distinction being drawn.
    try:
        configured = getattr(adapter, "config", None)
    except Exception:
        configured = None
    if isinstance(configured, ModelConfig):
        normalized = normalize_model_config(configured) or ModelConfig()
        return normalized, normalized
    return ModelConfig(), None


@dataclass(frozen=True)
class ReplayLookup:
    """One recomputation of a request's replay key, with the payload it was taken over.

    The pair travels together for the reason ``_DigestResult`` pairs its own fields: a miss
    diagnosis that rebuilt the payload separately could disagree with the key it is explaining --
    the double-read defect W6-0 paid for on ``provider_name``, one seam higher.
    """

    result: _DigestResult
    payload: dict[str, Any]


def replay_lookup(request: ModelRequest, adapter: Any) -> ReplayLookup:
    """The replay key for ``request`` as ``adapter`` would serve it, plus the payload it covers.

    The same composition ``ModelCallRunner.acall`` stamps into the receipt, applied to the request
    object the adapter was handed -- which *is* the runner's preimage object: the runner
    normalizes and swaps the effective config in before dispatch, and all four dispatch shapes
    forward that object uncopied. No re-normalization here, deliberately: recomputing over the
    handed object is what makes agreement with the receipt structural rather than an idempotence
    argument. The recompute-equals-stamp pin proves it end to end, per effective-config source.

    The declaration is read once and handed to ``resolved_provider_name`` -- the runner's own
    probe, tolerance included: a ``provider_name`` property that answers differently per read must
    not split this lookup from the receipt beside it.
    """

    model, _dispatch_override = effective_model_for(request, adapter)
    try:
        declared = normalize_unicode_scalars(str(getattr(adapter, "provider_name", "") or ""))
    except Exception:
        declared = ""
    payload = _request_payload(
        request,
        model,
        provider=resolved_provider_name(adapter, model, declared=declared) or "",
    )
    return ReplayLookup(result=_encoded_digest(payload), payload=payload)


def replay_lookup_digest(request: ModelRequest, adapter: Any) -> _DigestResult:
    """The key half of :func:`replay_lookup`, for callers with no diagnosis to feed."""

    return replay_lookup(request, adapter).result
