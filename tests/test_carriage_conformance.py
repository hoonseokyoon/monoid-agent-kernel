"""Machine-diff each semantic fact against every carrier obliged to repeat it.

The recurring defect in this repository is not a wrong value, it is a *missing* one: a fact
(``config_recoverable``, ``provider_retried``, a usage sub-count) rides N parallel carriers —
dataclass, checkpoint payload, event data, event schema, wire body, SSE frame, client reader,
transcript record — and a change binds N-1 of them.  Review finds those one at a time.  This
module finds them by construction: for each fact family it takes an AUTHORITY (the dataclass
fields, the ``__init__`` signature, the normalizer's emitted-key domain) and set-diffs it
against each carrier's key set, so a field added to the authority without a carrier fails here.

Three rules make that census trustworthy:

*Maximal builders, not synthesizers.*  Every family has one hand-written builder
(``_maximal_suspension``, ``_maximal_adapter_error``, ``_MAXIMAL_USAGE``) with every field set
to a distinguishable non-default value, plus a reflection guard that diffs the builder's
coverage against the authority.  A generic type-driven synthesizer would silently skip a new
field; the guard is what turns a new field into a failure.

*Behavioral where a dict diff is blind.*  A reader that ignores a wire key drops it without
leaving a trace in any key set, so the reader censuses feed a maximal wire body through the
real function and diff the *reconstructed attributes*.  That is the only way to see a
dropped read.

*Aliases are declared, not inferred.*  A fact is renamed at three hops (wire ``error_code``
carries ``provider_error_code``; ``RunCheckpoint`` spells ``http_status`` as
``provider_http_status``; ``awaiting_task_ids`` is ``task_ids`` on one event), so the alias
tables below are load-bearing: without them a rename reads as a missing field and the census
would have to be loosened until it proved nothing.

*And the census does not trust its own hand copies.*  Three of the mechanisms above were
defeated by drifts they were built to catch, all for one reason: an authority that was really a
transcription.  A key set copied off the normalizer missed an eighth emit branch; a probe that
re-implemented a server call missed a field threaded into the real one; a reader list written by
hand missed a fourth reader.  So each authority is now derived from the thing it describes —
the assignable key domain of the *live* ``normalize_usage``, the shipped ``_write_exception``
driven against a capturing host, the reader set discovered from the module.  Where a
hand-written EXPECTED remains it is pinned in full and diffed against a derived set, never
spot-checked; and the file-scan backstop reads code occurrences (AST), because substring
containment counted a comment as a carrier and so failed open.

*Then the derivations were attacked in turn, and they fell too.*  Every "read the live
callable" census read the wrong object, because ``inspect.getsource`` and ``inspect.signature``
both follow ``__wrapped__`` — so a ``functools.wraps`` wrapper adding an eighth usage key passed
untouched (``_live_callable`` now refuses a wrapper).  The assignable-domain reader claimed to
fail closed and looked at one of the four ways to write a key into a dict, so ``result.update``
walked past it.  Bucket *membership* was unpinned, so moving a field between two checkpoint
validation buckets changed the rule it is validated under while the union, the disjointness and
the count all stayed put — and a bucket nobody loops over validated nothing at all.  The wire
censuses pinned only the *maximal* request, so a writer that omitted a key whenever it held its
default value was indistinguishable from one that always writes it; each has a minimal-probe
twin now, and the difference between the two probes is the conditional half of the wire.  And
reader discovery was a predicate on where a function was *written* (module level) and how it
*spelled* its raise (a literal constructor), so a reader in a class body or one delegating to a
factory was not a reader.

*And each of those fixes was then attacked on its own twin, which is where they broke.*  A rule
proven on one of two parallel halves is this repository's dominant defect shape, and the census
was reproducing it five ways.  The write census that refuses what it cannot read was applied to
the *result* dict and to no other name, so a key written into the dict a copy loop reads
(``details["tool_tokens"] = ...``) reached the wire through a name nobody watched — and the
resolver twin beside it, the one that answers ``data=<name>`` at an emit site, had no refusal in
it at all.  Reader discovery still carried two spelled predicates: "reads the wire" meant
*calling a registered helper*, and "takes a mapping" meant the annotation said so, so an inline
``payload.get("error")`` and a ``payload: Any`` each walked past.  Family 7 had a minimal-input
twin for its two writers and family 2 had none, so an omit-when-falsy filter on either error
writer was invisible to every probe.  And two registered-gap pins censused one spelling of a read
(``getattr``, not ``exc.attr``) and one end of a two-ended path (the projection, not the route
that serves it), either of which would have stayed green through the very fix it exists to catch.

*And the repair that answers for every name still only watched what is written through it.*  A
dict leaves a name two other ways — handed to a call, and aliased — and neither was a
contribution the census refused, so ``_fold_web_counters(metrics_data, ctx)`` (the idiom this
package already uses in ``_accumulate_usage_mapping(state.total_usage, billed)``) and
``alias = result; alias["k"] = v`` each put a key on the wire that the census reported as
nonexistent.  An argument position is refused now unless the callee provably cannot reach a
mapping's mutators.  The same rule had a fourth resolution path that never got it: the extractors
that pull a *pinned* literal out of an argument dropped a ``**`` splat silently, so a merge into
the failure bundle or a transcript record widened the artifact while its equality pin stayed green.

*A registry entry with no assertion is prose.*  The registry's contract is "closing a gap breaks
this suite", which holds only for entries something actually asserts.  Five round-1 entries had
no pin at all — the drivers and consumers designed for a fact that ignore it, a closed schema,
an unconsumed event, a vocabulary collision — and the "Registered-gap pins" section below closes
that, one assertion per entry, each flipping the moment its gap is fixed.  The numeric claims in
:data:`FUTURE_FAMILIES` are pinned the same way, because a prediction carrying a stale count is
worse than no prediction.

**The green-with-registered-gaps contract.**  This suite is GREEN today, and today's reality
is not the ideal: many cells are unbound.  Every one of them is registered in
:data:`KNOWN_GAPS` with its carrier and disposition, and the assertions below encode reality
*exactly* — a pinned key set, an asserted loss.  So fixing a gap **breaks this suite**, and
that is the mechanism working: the fixer must update the EXPECTED constant and delete the
registry entry in the same change.  Do not loosen an assertion to accommodate a fix.

**And the families it does not cover are declared too.**  :data:`FUTURE_FAMILIES` names each
fact family deliberately left uncensused, with the authority a census would take and how many
hand-written carriers of it exist today.  Without that list the suite's silence is ambiguous —
a family with no failing cell reads exactly like a family nobody looked at — and given the
defect shape this repository keeps producing, an uncensused family with a carrier count is a
prediction rather than a footnote.

Seven families are censused: ``Suspension``, ``ModelAdapterError`` transport, usage counts, the
W5 applied-echo protocol, the tool catalog, the checkpoint validator's field coverage, and the
success envelope (the main wire).  :data:`KNOWN_GAPS` also carries cells that belong to no
census — the raw-vs-filtered error asymmetry, the run-status projections, the checkpoint's
un-carried run state — because a cell found by hand still has to be registered and pinned; those
pins live in the "Registered-gap pins" section rather than in a family.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import inspect
import json
import textwrap
from dataclasses import dataclass
from enum import IntEnum
from http import HTTPStatus
from pathlib import Path
from typing import Any, get_args, get_type_hints

import pytest

from monoid_agent_kernel.core.json_ingress import normalize_json_ingress
from monoid_agent_kernel.core.lifecycle import REASON_TO_STATE
from monoid_agent_kernel.core.manifest import _tool_spec_payload as _manifest_tool_spec_payload
from monoid_agent_kernel.core.result import (
    AgentTurnResult,
    Suspension,
    _SUSPENSION_REASONS,
    suspension_checkpoint_payload,
    suspension_from_checkpoint_payload,
)
from monoid_agent_kernel.core.schemas import EVENT_DATA_SCHEMAS, TRANSCRIPT_RECORD_SCHEMA
from monoid_agent_kernel.core.spec import GenerationConfig, ModelConfig, ReasoningConfig
from monoid_agent_kernel.core.tool_surface import (
    _tool_spec_payload as _transcript_tool_spec_payload,
)
from monoid_agent_kernel.errors import ModelAdapterError, TurnNotSettled
from monoid_agent_kernel.model_call import _recordable_usage
from monoid_agent_kernel.observability.otel import _chat_finish_attrs, _subagent_finish_attrs
from monoid_agent_kernel.providers import gateway as gateway_client
from monoid_agent_kernel.providers._common import normalize_usage
from monoid_agent_kernel.providers.base import ModelTurn, mark_provider_usage, provider_usage_of
from monoid_agent_kernel.providers.openai import _openai_tool_schema
from monoid_agent_kernel.reference.llm_gateway.http import (
    _error_body,
    _model_error_status,
    _stream_error_frame,
    make_llm_gateway_handler,
)
from monoid_agent_kernel.reference.llm_gateway.service import (
    LLM_TURN_PROTOCOL_VERSION,
    _TOOL_SCHEMA_KEYS,
    LlmGatewayBackend,
    LlmGatewayTurnRequest,
    LlmGatewayUsage,
    _applied_echoes,
    _parse_tool,
)
from monoid_agent_kernel.reference.studio.server import (
    _gateway_tool_schema as _studio_gateway_tool_schema,
)
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.tools.base import ToolSpec

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "monoid_agent_kernel"


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CarriageGap:
    """One fact/carrier cell the census found unbound, and what is meant to happen to it."""

    family: str
    field: str
    # "path:symbol", path relative to ``src/monoid_agent_kernel`` — checked to exist below, so a
    # renamed carrier rots this entry loudly instead of pointing at nothing.
    carrier: str
    gap: str
    disposition: str


DISPOSITIONS = frozenset({"burn-down", "v0.21-track:B1", "by-design"})

KNOWN_GAPS: tuple[CarriageGap, ...] = (
    CarriageGap(
        "suspension",
        "reason",
        "loop.py:_apump_turn",
        "turn.interrupted's data.reason is a *cause* vocabulary (\"user_stop\") while "
        "Suspension.reason is a *park* vocabulary (\"interrupted\"), one word for two "
        "vocabularies on one event; and the pause twin emits no turn.paused at all, only a "
        "session.state.changed, so the two sibling parks are not observable the same way",
        "burn-down",
    ),
    CarriageGap(
        "run-status projection",
        "limited",
        "core/lifecycle.py:state_from_suspension",
        "three mappers, three answers for one terminal budget-limited run: this one reports "
        "FAILED (a terminal park arrives as reason=\"terminal\" and only error_code=\"cancelled\" "
        "escapes that branch), session_state_from_run_status(\"limited\") reports LIMITED, and "
        "LoopSession.close reports COMPLETED because it tests only for \"failed\". The state an "
        "operator sees is decided by which surface they asked",
        "burn-down",
    ),
    CarriageGap(
        "suspension",
        "status",
        "reference/backend/recovery.py:run_recovered",
        "constructs Suspension(status=\"running\"), which is outside the vocabulary every "
        "durable reader accepts (core/result.py:suspension_from_checkpoint_payload admits only "
        "completed/failed/limited). Latent because this synthetic park is re-driven and never "
        "serialized — but nothing on the type says so, and the first code path that checkpoints "
        "it raises at the recovery boundary it was built to serve",
        "burn-down",
    ),
    CarriageGap(
        "suspension",
        "last_suspension",
        "core/checkpoint.py:_validate_checkpoint_payload",
        "the durable park payload is validated as \"an object or null\" and nothing more — it "
        "has no schema of its own, so every field the census pins on the writer/reader pair is "
        "unpinned on the durable artifact itself; this suite is currently its only twin",
        "burn-down",
    ),
    # --- the success wire ---------------------------------------------------------------
    CarriageGap(
        "success-envelope",
        "reasoning",
        "reference/llm_gateway/service.py:LlmGatewayBackend",
        "ModelTurn.reasoning reaches neither writer (the sync body nor the terminal frame) and "
        "neither client reader names the key, so the provider-native reasoning round-trip the "
        "adapters are built around (encrypted_content replay) is dead through the gateway. "
        "Symmetric across both transports, which is what makes it a missing feature rather than "
        "a twin that fell out of step: closing it means one wire key on two writers and two "
        "readers, in one change",
        "burn-down",
    ),
    # --- tool catalog ------------------------------------------------------------------
    CarriageGap(
        "tool-spec",
        "id",
        "providers/openai.py:_openai_tool_schema",
        "the Responses API keys a tool by name, so the kernel id has no slot and the mapping "
        "back is carried entirely by exported_name being stable: by design, and the reason a "
        "provider_name change is a wire-compatibility change",
        "by-design",
    ),
    # --- checkpoint carriage: what the snapshot leaves behind ---------------------------
    # Hand-found cells of the family FUTURE_FAMILIES declares as "run-state -> checkpoint
    # carriage": there is no census diffing ``snapshot()``'s written key set against the live
    # RunState + tool context, so these three were found one at a time and are pinned one at a
    # time.
    CarriageGap(
        "checkpoint",
        "output_failure_history",
        "loop.py:snapshot",
        "the sibling of a checkpointed field is not checkpointed: output_retries rides the "
        "snapshot and its history does not, so a run restored mid-repair renumbers its attempts "
        "from an empty history and loses failures_by_validator — the retry BUDGET survives and "
        "the evidence the budget was spent on does not",
        "burn-down",
    ),
    CarriageGap(
        "checkpoint",
        "subagent_count/subagent_usage/skill_activation_count",
        "loop.py:snapshot",
        "the context-owned counters have no checkpoint slot although their RunState twins "
        "(total_usage, total_tool_calls) do, and loop_phases.py:build_metrics writes all of "
        "them into one metrics.json — so a restored run reports pre-restart token totals beside "
        "post-restart subagent and skill counts, an artifact mixing two epochs with nothing "
        "saying which is which",
        "burn-down",
    ),
    CarriageGap(
        "checkpoint",
        "cancellation_requested",
        "loop.py:_rehydrate",
        "asymmetric write/read: snapshot() records the flag unconditionally, and the restore "
        "applies it only when a cancellation token is already installed on the loop. A recovery "
        "driver that rebuilds the loop without one silently un-cancels a run whose cancellation "
        "was durable",
        "burn-down",
    ),
    # --- checkpoint validation ---------------------------------------------------------
    CarriageGap(
        "checkpoint",
        "schema_version",
        "core/checkpoint.py:_validate_checkpoint_payload",
        "the one RunCheckpoint field no branch of the validator inspects; the codec owns the "
        "version envelope, so this is the documented exclusion rather than a hole -- pinned so "
        "a SECOND unvalidated field cannot join it quietly",
        "by-design",
    ),
    # --- by-design provenance splits ---------------------------------------------------
    CarriageGap(
        "applied-echo",
        "provider_retried",
        "providers/gateway.py:GatewayModelAdapter",
        "the frameless-stream check reads the client's own attempt>1 while the framed check "
        "reads chunk.provider_retried: by design, because a stream with no terminal frame "
        "carries no server-side retry evidence to read, and the client's own attempt count is "
        "the only true thing left to say",
        "by-design",
    ),
    # --- W5 applied-echo protocol ------------------------------------------------------
    CarriageGap(
        "applied-echo",
        "reasoning_applied",
        "reference/llm_gateway/service.py:_applied_echoes",
        "reasoning has the same fail/omit contract as generation (core/spec.py:ReasoningConfig "
        "on_unsupported) and travels the same wire, but there is no echo, no support probe and "
        "no client-side checker — a fail-closed reasoning request is accepted unproven",
        "v0.21-track:B1",
    ),
    # --- by-design: asserted and explained, never "fixed" ------------------------------
    CarriageGap(
        "suspension",
        "turn",
        "core/result.py:suspension_checkpoint_payload",
        "excluded by design: the AgentTurnResult is a projection artifact of local paths and "
        "metrics, and a recovery driver needs only the boundary facts to return the same park",
        "by-design",
    ),
    CarriageGap(
        "applied-echo",
        "generation",
        "core/spec.py:ModelConfig",
        "to_json omits a default generation block by design: this dict feeds the request digest "
        "and the runtime-config semantic hash, so a never-configured block must serialize "
        "byte-identically to a config predating the field",
        "by-design",
    ),
    CarriageGap(
        "transportable-error",
        "usage",
        "reference/llm_gateway/http.py:_error_body",
        "omitted-when-empty by design: an error raised before a provider was reached keeps its "
        "exact pre-usage wire shape",
        "by-design",
    ),
    CarriageGap(
        "applied-echo",
        "generation_applied/schema_applied",
        "reference/llm_gateway/http.py:_stream_error_frame",
        "error frames carry no applied-echoes by design: there is no turn to prove anything "
        "about, and a fail-closed client refuses on absence anyway",
        "by-design",
    ),
    CarriageGap(
        "usage",
        "provider_usage",
        "providers/gateway.py:_reported_error_usage",
        "lenient by design: arbitrary keys pass, because a malformed usage on an error path "
        "must not replace the failure being reported with a validation error",
        "by-design",
    ),
    CarriageGap(
        "usage",
        "cache_read_tokens/cache_creation_tokens/reasoning_tokens/audio_tokens",
        "observability/otel.py:_chat_finish_attrs",
        "only input/output token attributes are emitted (here and at its _subagent_finish_attrs "
        "/ _apply_capture twins), by design: GenAI semantic conventions define those two and a "
        "non-standard attribute is not portable across collectors",
        "by-design",
    ),
    CarriageGap(
        "transportable-error",
        "error_code",
        "reference/llm_gateway/http.py:_error_body",
        "compat-frozen alias: the wire key error_code carries provider_error_code, and the "
        "kernel-level error_code has no wire slot at all — readers reconstruct the class "
        "default. Renaming the key would break every deployed gateway client",
        "by-design",
    ),
    CarriageGap(
        "suspension",
        "http_status",
        "core/checkpoint.py:RunCheckpoint",
        "compat-frozen alias: the checkpoint spells this provider_http_status while the event, "
        "wire and Suspension all spell it http_status",
        "by-design",
    ),
    CarriageGap(
        "suspension",
        "awaiting_task_ids",
        "core/schemas.py:EVENT_DATA_SCHEMAS",
        "three spellings of one fact, alias-registered: Suspension.awaiting_task_ids, task_ids "
        "on run.awaiting_input, awaiting_task_ids again on the backend frame "
        "(reference/backend/run_execution.py)",
        "by-design",
    ),
)


@dataclass(frozen=True)
class FutureFamily:
    """A fact family this census does NOT cover yet, declared so the omission is a decision.

    ``KNOWN_GAPS`` says "this cell is unbound"; this says "this whole family is uncensused".
    Without it the suite's silence is ambiguous — a family with no failing cell reads exactly
    like a family nobody looked at — and the shape this repository keeps producing is a fact
    riding N hand-written carriers, so an uncensused family with a carrier count is a
    prediction, not a note.
    """

    family: str
    # "path:symbol" — the thing a census of this family would take as its authority.
    authority: str
    # How many independent hand-written carriers of that authority exist today.
    carrier_count: int
    risk: str
    disposition: str


FUTURE_FAMILIES: tuple[FutureFamily, ...] = (
    FutureFamily(
        "model-config wire block",
        "core/spec.py:ModelConfig",
        4,
        "the client writes the block (providers/gateway.py:_payload) and the server rebuilds it "
        "(reference/llm_gateway/service.py:_parse_turn_request), with two special carriages that "
        "exist because a missing key is the *default* on the rebuild, not 'unset': off-default "
        "on_unsupported, and effort='default' whose omission sentinel differs from the codec's "
        "reconstruction default. A third such field would be a silent policy override",
        "burn-down",
    ),
    FutureFamily(
        "model-config transport policy",
        "reference/llm_gateway/service.py:_upstream_model_config",
        1,
        "the rebuilt upstream config takes default timeout_s/retry rather than the caller's. "
        "By design: the wire block never carries them (providers/gateway.py:_payload emits only "
        "model/reasoning/generation), so each hop owns its own transport policy — the client's "
        "timeout bounds the call to the gateway, the server's bounds the call to the provider. "
        "The test that separates 'must ride' from 'per-hop' is whether a default silently "
        "*overrides a caller's stated intent*, which is exactly why on_unsupported was added",
        "by-design",
    ),
    FutureFamily(
        "run limits",
        "core/spec.py:RunLimits",
        1,
        "core/manifest.py:build_run_manifest carries 4 of 15 limits into the run manifest, so "
        "the durable record of what a run was allowed to do omits every token budget, every "
        "subagent bound and the message-log caps",
        "burn-down",
    ),
    FutureFamily(
        "stream frames",
        "providers/base.py:ModelStreamChunk",
        2,
        "reference/llm_gateway/service.py:_chunk_to_frame is a second hand-built frame writer "
        "beside the four to_json codecs, read back by providers/gateway.py:_chunk_from_event; "
        "a field added to a chunk type reaches neither unless both are edited",
        "burn-down",
    ),
    FutureFamily(
        "stream-outcome usage lane",
        "core/model_stream.py:ModelStreamOutcome",
        3,
        "outcome -> the stream_closed record (core/schemas.py, additionalProperties False) -> "
        "the live frame the studio reads; the record carries retryable but not "
        "config_recoverable, and a closed cap means a new fact is a schema change",
        "burn-down",
    ),
    FutureFamily(
        "outbox request codec",
        "core/outbox.py:OutboxRequest",
        3,
        "codec + event subsets + the backend's redrive read; a field added to the request that "
        "the event subset omits is invisible to an operator watching a stuck outbox",
        "burn-down",
    ),
    FutureFamily(
        "capability lease",
        "core/capability.py:CapabilityLease",
        3,
        "lease dataclass, checkpoint payload and the control envelope that grants it",
        "burn-down",
    ),
    FutureFamily(
        "hosted tasks",
        "tasks.py:HostedTask",
        3,
        "task record, checkpoint payload and the run.awaiting_input/backend frames that name "
        "its ids (already one registered alias: task_ids vs awaiting_task_ids)",
        "burn-down",
    ),
    FutureFamily(
        "control envelope",
        "core/control.py:ControlCommand",
        2,
        "the control-command.v1 wire shape and its dispatch table; an unknown command field is "
        "dropped silently at the boundary",
        "burn-down",
    ),
    FutureFamily(
        "content-part vocabulary",
        "core/content.py:ContentPart",
        3,
        "five part types spelled in the union, the to_json/from_json codec pair and the "
        "provider-side mappers; a sixth type reaching only the codec drops at the adapters",
        "burn-down",
    ),
    FutureFamily(
        "status.json projection",
        "recorder.py:StatusJsonSink",
        2,
        "the sink writes a metrics key STATUS_SCHEMA does not declare, so the operator-facing "
        "status file is already wider than its own schema",
        "burn-down",
    ),
    FutureFamily(
        "run-state -> checkpoint carriage",
        "loop.py:snapshot",
        2,
        "the authority is the key set snapshot() writes, and the carriers are the two live "
        "states it must cover: RunState and the AgentToolContext. Nothing diffs them, so the "
        "three registered cells (output_failure_history, the context-owned "
        "subagent/skill counters, and cancellation_requested's asymmetric restore) were each "
        "found by hand — which is the definition of an uncensused family, not of three "
        "unrelated bugs",
        "burn-down",
    ),
    FutureFamily(
        "settled-text digest",
        "loop_phases.py:_settled_text_fields",
        6,
        "one settle payload is EITHER final_text OR final_text_digest+final_text_len, and the "
        "choice is re-implemented at every hop: the writer here, two schema branches "
        "(core/schemas.py), the hydration reader "
        "(reference/backend/content_hydration.py:DIGEST_FIELD), a second hand reader in "
        "reference/studio/cli.py, a third path in reference/studio/chat_projection.py, and the "
        "sidecar join key in core/model_content.py. Renaming the fact, or adding a third "
        "encoding, has to land on all of them at once and nothing says so",
        "burn-down",
    ),
)


def test_registry_entries_are_well_formed() -> None:
    assert KNOWN_GAPS
    for gap in KNOWN_GAPS:
        assert gap.disposition in DISPOSITIONS, gap
        assert ":" in gap.carrier, gap
        assert gap.gap.strip(), gap
    assert FUTURE_FAMILIES
    for family in FUTURE_FAMILIES:
        assert family.disposition in DISPOSITIONS, family
        assert ":" in family.authority, family
        assert family.carrier_count >= 1, family
        assert family.risk.strip(), family
    # A declared future family must not silently overlap a family the census already covers.
    assert len({family.family for family in FUTURE_FAMILIES}) == len(FUTURE_FAMILIES)


def test_registry_carrier_locations_exist() -> None:
    """A renamed carrier must rot its registry entry rather than point at nothing."""

    missing: list[str] = []
    for location in [gap.carrier for gap in KNOWN_GAPS] + [
        family.authority for family in FUTURE_FAMILIES
    ]:
        relative_path, symbol = location.split(":", 1)
        path = PACKAGE / relative_path
        if not path.is_file():
            missing.append(f"{location} (no such file)")
            continue
        if symbol not in path.read_text(encoding="utf-8"):
            missing.append(f"{location} (symbol absent)")
    assert missing == [], {"stale_registry_entries": missing}


# --------------------------------------------------------------------------------------
# Live-callable guard — the census must read the object it just called
# --------------------------------------------------------------------------------------


def _unwrap_chain(function: Any) -> list[Any]:
    """``function`` followed by everything its ``__wrapped__`` chain points at."""

    chain = [function]
    seen = {id(function)}
    current = function
    while True:
        current = getattr(current, "__wrapped__", None)
        if current is None or id(current) in seen:
            return chain
        seen.add(id(current))
        chain.append(current)


def _live_callable(function: Any) -> Any:
    """``function``, refused if it is a ``functools.wraps`` wrapper over something else.

    ``inspect.getsource`` and ``inspect.signature`` both FOLLOW ``__wrapped__``. That defeated the
    "census the live callable, not the file" rule at the one place it was supposed to hold: a
    ``@functools.wraps`` wrapper around ``normalize_usage`` emitting an eighth key was invisible
    to every assertion here, because ``getsource`` handed back the *wrapped* function's body and
    the wrapper's own branch was never read. The signature censuses have the same hole.

    Refused rather than merged. A wrapper is usually not analyzable at all (``*args, **kwargs``
    and a call), so merging its key domain would mean teaching each census a second shape; and a
    wrapper installed over a censused carrier is a carriage change, which is exactly the kind of
    thing this suite exists to make someone declare.
    """

    chain = _unwrap_chain(function)
    assert len(chain) == 1, {
        "censused_callable": getattr(function, "__qualname__", repr(function)),
        "wraps": [getattr(item, "__qualname__", repr(item)) for item in chain[1:]],
        "hint": "inspect.getsource/signature follow __wrapped__, so this census would have read "
        "the wrapped function and missed every branch the wrapper adds",
    }
    return function


def _live_signature(function: Any) -> inspect.Signature:
    return inspect.signature(_live_callable(function))


def _live_source(function: Any) -> str:
    return inspect.getsource(_live_callable(function))


# --------------------------------------------------------------------------------------
# Maximal builders + their reflection guards
# --------------------------------------------------------------------------------------


def _maximal_turn() -> AgentTurnResult:
    return AgentTurnResult(
        status="failed",
        final_text="settled text",
        proposal_path=Path("proposal.json"),
        proposal_hash="deadbeef",
    )


def _maximal_suspension() -> Suspension:
    """Every Suspension field at a distinguishable non-default value.

    ``reason`` must be one of the declared Literal values; ``turn_failed`` is chosen because it
    is the reason that actually populates the classification trio the census cares about.
    """

    return Suspension(
        reason="turn_failed",
        status="failed",
        final_text="settled text",
        error="upstream refused",
        error_code="model_error",
        awaiting_task_ids=("task-1", "task-2"),
        has_external=True,
        turn=_maximal_turn(),
        retryable=True,
        http_status=429,
        config_recoverable=True,
        provider_error_code="insufficient_quota",
        provider_retried=True,
    )


def test_maximal_suspension_covers_every_authority_field() -> None:
    """The guard that turns a newly added Suspension field into a failure of this suite."""

    authority = {field.name for field in dataclasses.fields(Suspension)}
    built = _maximal_suspension()
    default_marker = Suspension(reason="settled", status="completed")
    undistinguished = {
        name
        for name in authority
        if name not in {"reason", "status"}
        and getattr(built, name) == getattr(default_marker, name)
    }
    assert undistinguished == set(), {
        "fields_left_at_their_default": sorted(undistinguished),
        "hint": "extend _maximal_suspension so every field is distinguishable from its default",
    }


# The transportable attributes of a ModelAdapterError: the keyword-only ``__init__`` params
# minus ``error_code`` handling quirks, plus the post-hoc ``provider_usage`` stamp
# (providers/base.py:mark_provider_usage), which is not a constructor argument.
_ADAPTER_ERROR_STAMPED_ATTRS = frozenset({"provider_usage"})


def _maximal_adapter_error() -> ModelAdapterError:
    """A ModelAdapterError with every transportable fact set to a distinguishable value.

    Traps encoded here: ``message`` is positional and sets no attribute (it is read back through
    ``str(exc)``); ``error_code`` lands in ``vars`` only because a non-None value is passed
    (otherwise the class attribute ``"model_error"`` shows through and ``vars`` disagrees with
    ``getattr``); ``provider_error_code=None`` would be coerced to ``""``; and ``provider_usage``
    has no constructor parameter at all.
    """

    exc = ModelAdapterError(
        "upstream refused the turn",
        error_code="custom_kernel_code",
        provider_error_code="insufficient_quota",
        retryable=True,
        http_status=429,
        provider_retried=True,
        config_recoverable=True,
    )
    mark_provider_usage(exc, {"input_tokens": 11, "output_tokens": 22, "total_tokens": 33})
    return exc


def test_maximal_adapter_error_covers_every_authority_attribute() -> None:
    """ModelAdapterError is not a dataclass: authority is the ctor signature plus the stamp."""

    signature = _live_signature(ModelAdapterError.__init__)
    keyword_only = {
        name
        for name, param in signature.parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    }
    authority = keyword_only | _ADAPTER_ERROR_STAMPED_ATTRS
    carried = set(vars(_maximal_adapter_error()))
    assert carried == authority, {
        "missing": sorted(authority - carried),
        "extra": sorted(carried - authority),
    }
    # ``message`` is positional and stores nothing; it is transportable only via ``str(exc)``.
    assert "message" not in carried
    assert str(_maximal_adapter_error()) == "upstream refused the turn"


# Every key ``normalize_usage`` can emit, fed by one input that reaches all seven branches.
_MAXIMAL_USAGE: dict[str, int] = {
    "input_tokens": 101,
    "output_tokens": 202,
    "total_tokens": 303,
    "cache_read_tokens": 404,
    "cache_creation_tokens": 505,
    "reasoning_tokens": 606,
    "audio_tokens": 707,
}

NORMALIZED_USAGE_KEYS = frozenset(_MAXIMAL_USAGE)


def test_maximal_usage_covers_the_normalizer_emitted_domain() -> None:
    """The usage authority is behavioral: what ``normalize_usage`` can put on the wire.

    Values must be positive ``int`` (the counters use ``type(v) is int``, which rejects ``bool``),
    or a sub-count would be silently dropped and read as "this key cannot be emitted".
    """

    emitted = set(normalize_usage(dict(_MAXIMAL_USAGE)))
    assert emitted == NORMALIZED_USAGE_KEYS, {
        "missing": sorted(NORMALIZED_USAGE_KEYS - emitted),
        "extra": sorted(emitted - NORMALIZED_USAGE_KEYS),
    }
    assert all(type(value) is int and value > 0 for value in _MAXIMAL_USAGE.values())


def test_the_usage_authority_is_the_normalizers_assignable_domain_not_one_probe() -> None:
    """The static half of the usage authority, and the one that catches a *new* emit branch.

    The probe above proves the seven keys are reachable; it cannot prove there is no eighth,
    because a branch keyed on an input the probe does not carry never runs. This reads the keys
    the function can assign into its result at all -- of the live callable, so a wrapper that
    adds a branch is censused too, not merely of the file as committed.
    """

    assignable = _emitted_result_keys(_live_function_body(normalize_usage))
    assert assignable == NORMALIZED_USAGE_KEYS, {
        "assignable_but_not_in_the_authority": sorted(assignable - NORMALIZED_USAGE_KEYS),
        "in_the_authority_but_never_assigned": sorted(NORMALIZED_USAGE_KEYS - assignable),
        "hint": "a new emit branch: extend _MAXIMAL_USAGE and every carrier census below",
    }


# --------------------------------------------------------------------------------------
# Alias tables
# --------------------------------------------------------------------------------------

# Wire key -> the ModelAdapterError fact it carries across the gateway hop.  Declared rather
# than inferred, because three of the seven are renames.
TRANSPORTABLE_ERROR_WIRE_ALIASES: dict[str, str] = {
    # ``message`` is positional-only-in-effect: read back through ``str(exc)``, never an attribute.
    "error": "message",
    # Compat-frozen rename. The wire's ``error_code`` is the PROVIDER code; the kernel-level
    # ``ModelAdapterError.error_code`` has no wire slot and resets to the class default on
    # reconstruction (registered by-design above).
    "error_code": "provider_error_code",
    "retryable": "retryable",
    # The classification whose only transport used to be the 4xx status ``_model_error_status``
    # picks; it has its own key now, written unconditionally like the two booleans beside it.
    "config_recoverable": "config_recoverable",
    # Server-derived: ``_model_error_status`` picks the status, it is not copied off the exception.
    "http_status": "http_status",
    "provider_retried": "provider_retried",
    # Omitted-when-empty, so any probe of this key must stamp a non-empty usage first.
    "usage": "provider_usage",
}

# Transportable facts with no wire key at all. Empty today -- ``config_recoverable`` was the last
# one and the burn-down gave it a key. Kept rather than deleted: the disjointness guards below are
# the mechanism that notices the *next* such fact, and they are vacuously true over an empty set
# instead of absent.
TRANSPORTABLE_ERROR_UNCARRIED: frozenset[str] = frozenset()

# Checkpoint spelling of the same fact (core/checkpoint.py:RunCheckpoint), registered by-design.
CHECKPOINT_FIELD_ALIASES: dict[str, str] = {"http_status": "provider_http_status"}

# Suspension.awaiting_task_ids has three spellings across its carriers.
AWAITING_TASK_IDS_SPELLINGS: dict[str, str] = {
    "core/result.py:Suspension": "awaiting_task_ids",
    "core/schemas.py:run.awaiting_input": "task_ids",
    "reference/backend/run_execution.py": "awaiting_task_ids",
}


def test_transportable_error_alias_table_is_total_over_the_authority() -> None:
    """Every transportable attribute is either aliased to a wire key or registered uncarried."""

    authority = set(vars(_maximal_adapter_error()))
    accounted = set(TRANSPORTABLE_ERROR_WIRE_ALIASES.values()) | TRANSPORTABLE_ERROR_UNCARRIED
    accounted.discard("message")  # not an attribute; asserted separately
    # ``error_code`` is in ``vars`` only because the maximal builder passes a non-None value.
    accounted.add("error_code")
    assert authority == accounted, {
        "unaccounted_attributes": sorted(authority - accounted),
        "aliases_without_an_attribute": sorted(accounted - authority),
    }


def test_awaiting_task_ids_spellings_are_each_present_at_their_site() -> None:
    assert AWAITING_TASK_IDS_SPELLINGS["core/result.py:Suspension"] in {
        field.name for field in dataclasses.fields(Suspension)
    }
    awaiting = EVENT_DATA_SCHEMAS["run.awaiting_input"]["properties"]
    assert AWAITING_TASK_IDS_SPELLINGS["core/schemas.py:run.awaiting_input"] in awaiting
    assert "awaiting_task_ids" not in awaiting
    backend = (PACKAGE / "reference" / "backend" / "run_execution.py").read_text(encoding="utf-8")
    assert '"awaiting_task_ids"' in backend


def test_checkpoint_alias_is_the_only_spelling_of_http_status_on_the_checkpoint() -> None:
    from monoid_agent_kernel.core.checkpoint import RunCheckpoint

    names = {field.name for field in dataclasses.fields(RunCheckpoint)}
    assert CHECKPOINT_FIELD_ALIASES["http_status"] in names
    assert "http_status" not in names


# --------------------------------------------------------------------------------------
# AST helpers — census the inline dict literals that no import can reach
# --------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _module_tree(relative_path: str) -> ast.Module:
    """Parsed once per module. ``loop.py`` alone is ~4k lines and a dozen censuses read it."""

    path = PACKAGE / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _dict_literal_keys(node: ast.Dict) -> tuple[frozenset[str], tuple[str, ...]]:
    """A dict literal's constant string keys, *and the parts of it that are not that*.

    ``_dict_keys`` returns only the first half. A ``**splat`` entry has a ``None`` key and a
    computed key is not a ``Constant``, and both simply vanish from the set it returns — so a
    literal that merges a mapping of unknown width was censused as the two keys spelled beside
    it, and every census that treats a literal as *the* key set inherited that silence. The
    second half is what a caller has to refuse on.
    """

    keys: set[str] = set()
    refusals: list[str] = []
    for key, value in zip(node.keys, node.values):
        if key is None:
            refusals.append(f"**{ast.unparse(value)}")
        elif isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.add(key.value)
        else:
            refusals.append(f"{ast.unparse(key)}: ...")
    return frozenset(keys), tuple(refusals)


def _pinned_dict_keys(node: ast.Dict, context: str) -> frozenset[str]:
    """A literal's keys, where the caller pins that key set as *the* thing the literal carries.

    The fourth resolution path onto the rule :func:`_dict_literal_keys` states, and the last one
    still reading the weak half of it: the two literal-argument extractors below (and the manifest
    census that spells one out inline) fed their matches through a helper that returned the
    constant string keys and dropped everything else. Every one of them hands its result to an
    equality pin, so adding ``**_error_fields(exc)`` to a pinned literal widened the artifact
    while the pin stayed green — the same fail-open :func:`_dict_writes` and
    :func:`_emit_data_keys` had already closed on the assignment side of the file.
    """

    keys, merged = _dict_literal_keys(node)
    assert merged == (), {
        "literal": context,
        "line": node.lineno,
        "merges_the_census_cannot_read": list(merged),
        "hint": "the pinned key set is no longer the whole literal — resolve the merge here "
        "rather than dropping it, or the pin passes while the artifact widens",
    }
    return keys


@dataclass(frozen=True)
class _DictWrites:
    """Everything written into one name-bound dict inside one scope.

    ``literals`` is what the name was *assigned*; ``added`` is every constant key written into it
    afterwards (a subscript write, an analyzable ``update({...})`` or ``|= {...}`` merge);
    ``dynamic`` is the subscript writes whose key is an expression, which only a caller that
    knows the surrounding loop can resolve; ``refusals`` is every contribution that cannot be
    read as a closed set of constant keys at all.

    So ``keys`` is a *lower bound* and is the answer only when nothing was refused — which is why
    every consumer below asserts ``unreadable`` empty rather than filtering it away.
    """

    name: str
    literals: tuple[frozenset[str], ...]
    added: frozenset[str]
    dynamic: tuple[ast.expr, ...]
    refusals: tuple[str, ...]

    @property
    def keys(self) -> frozenset[str]:
        return frozenset().union(self.added, *self.literals)

    @property
    def unreadable(self) -> tuple[str, ...]:
        return self.refusals + tuple(
            f"{self.name}[{ast.unparse(index)}] = ..." for index in self.dynamic
        )


# Dict methods that only read. Anything else called on a tracked name is a mutation this census
# cannot read, so it is refused by name rather than enumerated: the point is not to know every
# mutator, it is that an unknown one must never pass as "no keys were added".
_DICT_READ_METHODS = frozenset({"copy", "get", "items", "keys", "values"})

# The same rule on the other side of the call, which is where round 3's version of it broke.
# ``result.mutate()`` was refused and ``mutate(result)`` was not, so a tracked name *handed to* a
# function wrote keys through a callee this census cannot see into and passed as "no keys were
# added" — and that idiom is native to the censused code itself:
# ``_accumulate_usage_mapping(state.total_usage, billed)`` in ``loop.py:_apump_turn`` folds one
# mapping into another through exactly this shape. So a tracked name in an argument position is a
# refusal unless the callee is listed here.
#
# Every entry is a *builtin that cannot reach a mapping's mutators*: it reads keys and returns a
# new object. That keeps the list a proof rather than a per-callee audit — the alternative,
# "these particular helpers looked harmless when I read them", is the hand transcription this
# suite exists to stop trusting. Today's censused functions hand their dicts to exactly ONE call
# between them (the emit below); a future ``len(metrics_data)`` is meant to pass here, and a
# future ``_fold_web_counters(metrics_data, ctx)`` is meant to fail.
_DICT_READ_CALLS = frozenset(
    {
        "bool",  # emptiness test
        "dict",  # copies; writes into the copy are not writes into this name
        "frozenset",  # iterates keys into a new frozenset
        "len",  # reads the size
        "list",  # iterates keys into a new list
        "set",  # iterates keys into a new set
        "sorted",  # iterates keys into a new list
        "tuple",  # iterates keys into a new tuple
    }
)

# The one non-builtin call a censused function hands a tracked dict to today, and the reason it
# cannot simply be refused: it *is* the site :func:`_emit_data_keys` resolves the binding for
# (``recorder.emit("metrics.updated", turn_id=..., data=metrics_data)`` in ``_apump_turn``).
# Refusing it would refuse the census's own read. Narrow on purpose — the method name AND the
# keyword — so ``emit(data=x)`` stays readable while ``enrich(data=x)`` does not.
_CENSUSED_EMIT_ARGUMENT = ("emit", "data")


@functools.lru_cache(maxsize=None)
def _dict_writes(scope: ast.AST) -> dict[str, _DictWrites]:
    """Every dict-shaped name in ``scope``, with everything written into it.

    One walk, **every** name — not just the one name a caller happens to care about. That
    asymmetry is the round-3 defect this replaces: the write census was applied to the result
    dict and to nothing else, so a key written into the dict a copy loop *reads*
    (``details["tool_tokens"] = ...`` ahead of ``for key, value in details.items():
    result[key] = value``) reached the result through a name nobody was watching, and the census
    reported the loop as carrying the source literal's keys alone.

    A name is tracked if a dict literal is bound to it or a subscript is written into it. For a
    tracked name every later contribution is either folded in (constant keys) or refused; the
    one shape left to the caller is a computed subscript key, because only the caller knows
    whether the expression is a loop variable it can resolve.

    Round 3 read "contribution" as *what is written through this name here*, and both ways a dict
    leaves a name unwritten-through walked past it:

    * **handed to a call.** ``result.mutate()`` was refused and ``mutate(result)`` was not, so
      ``enrich(result)`` or ``_fold_sub_counts(metrics_data, ...)`` wrote keys through a callee
      this census cannot see into while the name reported a clean key set. An argument position is
      a refusal now unless the callee is a proven-read-only builtin (:data:`_DICT_READ_CALLS`) or
      the emit whose binding the census is there to read (:data:`_CENSUSED_EMIT_ARGUMENT`). A
      ``*name`` / ``**name`` splat is *not* refused: the callee receives a fresh tuple/mapping and
      cannot write back through it, which is the same reason ``.items()`` is a read.
    * **aliased.** ``alias = result`` gives the same object a second name, and ``alias["k"] = v``
      then makes ``alias`` tracked in its own right — with its own entry, which the consumer
      (asking about *one* name) never reads. Tracking the alias is therefore not enough without
      the flow analysis to merge the two entries, so the *source* refuses.
    """

    tracked: set[str] = set()
    for node in ast.walk(scope):
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and isinstance(value, ast.Dict):
            tracked.add(target.id)
        if isinstance(node, ast.Assign):
            for assigned in node.targets:
                if isinstance(assigned, ast.Subscript) and isinstance(assigned.value, ast.Name):
                    tracked.add(assigned.value.id)

    literals: dict[str, list[frozenset[str]]] = {}
    added: dict[str, set[str]] = {}
    dynamic: dict[str, list[ast.expr]] = {}
    refusals: dict[str, list[str]] = {}

    def _fold(name: str, node: ast.Dict, context: str) -> frozenset[str]:
        keys, merged = _dict_literal_keys(node)
        refusals.setdefault(name, []).extend(f"{context} merges {item}" for item in merged)
        return keys

    def _handed_to(argument: ast.Name, position: str, call: ast.Call) -> None:
        """A tracked name in an argument position of ``call`` — read, or written through?"""

        if argument.id not in tracked:
            return
        if isinstance(call.func, ast.Name) and call.func.id in _DICT_READ_CALLS:
            return
        refusals.setdefault(argument.id, []).append(
            f"{argument.id} handed to {ast.unparse(call.func)}(...) as {position}"
        )

    for node in ast.walk(scope):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            written = node.targets if isinstance(node, ast.Assign) else [node.target]
            # ``alias = tracked`` (or ``self.x = tracked``, or ``payload["k"] = tracked``): the
            # object escapes this name, and a write through the second handle is a write into
            # this dict that this name's own census never sees.
            if isinstance(node.value, ast.Name) and node.value.id in tracked:
                refusals.setdefault(node.value.id, []).extend(
                    f"{node.value.id} aliased to {ast.unparse(assigned)}" for assigned in written
                )
            for assigned in written:
                if isinstance(assigned, ast.Name) and assigned.id in tracked:
                    if isinstance(node.value, ast.Dict):
                        literals.setdefault(assigned.id, []).append(
                            _fold(assigned.id, node.value, f"{assigned.id} = {{...}}")
                        )
                    elif node.value is not None:
                        # A rebinding to something this census cannot read is a key set of
                        # unknown width standing where the literal's used to be.
                        refusals.setdefault(assigned.id, []).append(
                            f"{assigned.id} = {ast.unparse(node.value)}"
                        )
                elif (
                    isinstance(assigned, ast.Subscript)
                    and isinstance(assigned.value, ast.Name)
                    and assigned.value.id in tracked
                ):
                    index = assigned.slice
                    if isinstance(index, ast.Constant) and isinstance(index.value, str):
                        added.setdefault(assigned.value.id, set()).add(index.value)
                    else:
                        dynamic.setdefault(assigned.value.id, []).append(index)
            continue
        if isinstance(node, ast.AugAssign):
            mutated = node.target
            if isinstance(mutated, ast.Name):
                name = mutated.id
            elif isinstance(mutated, ast.Subscript) and isinstance(mutated.value, ast.Name):
                name = mutated.value.id
            else:
                continue
            if name not in tracked:
                continue
            if (
                isinstance(mutated, ast.Name)
                and isinstance(node.op, ast.BitOr)
                and isinstance(node.value, ast.Dict)
            ):
                added.setdefault(name, set()).update(
                    _fold(name, node.value, f"{name} |= {{...}}")
                )
            else:
                refusals.setdefault(name, []).append(
                    f"{ast.unparse(mutated)} {type(node.op).__name__}= ..."
                )
            continue
        if not isinstance(node, ast.Call):
            continue
        # (1) the name is what the call was HANDED.
        for position, argument in enumerate(node.args):
            if isinstance(argument, ast.Name):
                _handed_to(argument, f"positional {position}", node)
        for keyword in node.keywords:
            if keyword.arg is None or not isinstance(keyword.value, ast.Name):
                continue  # ``f(**name)``: the callee gets a fresh mapping, so it is a read
            if (
                isinstance(node.func, ast.Attribute)
                and (node.func.attr, keyword.arg) == _CENSUSED_EMIT_ARGUMENT
            ):
                continue  # the emit site whose binding _emit_data_keys resolves
            _handed_to(keyword.value, f"{keyword.arg}=", node)
        # (2) the name is what the call was made ON.
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            name, attribute = node.func.value.id, node.func.attr
            if name not in tracked:
                continue
            if (
                attribute == "update"
                and len(node.args) == 1
                and not node.keywords
                and isinstance(node.args[0], ast.Dict)
            ):
                added.setdefault(name, set()).update(
                    _fold(name, node.args[0], f"{name}.update({{...}})")
                )
            elif attribute not in _DICT_READ_METHODS:
                refusals.setdefault(name, []).append(f"{name}.{attribute}(...)")

    return {
        name: _DictWrites(
            name,
            tuple(literals.get(name, ())),
            frozenset(added.get(name, ())),
            tuple(dynamic.get(name, ())),
            tuple(refusals.get(name, ())),
        )
        for name in tracked
    }


@functools.lru_cache(maxsize=None)
def _emit_data_keys(relative_path: str, event_type: str) -> frozenset[str]:
    """Keys the module can put on ``recorder.emit("<event_type>", ..., data=...)``.

    The emit site is an inline literal inside a long pump method, so it cannot be imported and
    diffed — but it is exactly where a new key is added without a matching schema entry.

    Every call site for the event type is counted, whatever shape its ``data=`` argument has.
    Counting only the sites with a literal dict was a fail-open that the census's own comment
    described as a twin check: a second emit passing ``data=some_name`` (the shape
    ``metrics.updated`` already uses) left the count at one and the twin went uncensused. A site
    this function cannot resolve is now a failure naming the shape, not a site it skips.

    "Cannot resolve" then had to mean what it says. ``data=fn()`` refused, but the *resolver*
    behind ``data=<name>`` had no refusal in it at all: it read dict literals and constant-key
    subscript writes and dropped everything else, so ``data.update({...})``, ``data |= {...}``
    and a ``{**extra}`` splat inside the literal each put a key on the wire that this function
    reported as nonexistent — the same fail-open :func:`_emitted_result_keys` had closed on its
    own side of the file and never bound on its twin. :func:`_dict_writes` now folds in what it
    can read and refuses what it cannot, and an unreadable contribution fails the site.
    """

    tree = _module_tree(relative_path)
    bound = _dict_writes(tree)
    resolved: list[frozenset[str]] = []
    unresolved: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "emit" or not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and first.value == event_type):
            continue
        data = next((keyword.value for keyword in node.keywords if keyword.arg == "data"), None)
        if isinstance(data, ast.Dict):
            keys, merged = _dict_literal_keys(data)
            if merged:
                unresolved.append(f"line {node.lineno}: data={{...}} merges {list(merged)}")
            else:
                resolved.append(keys)
        elif isinstance(data, ast.Name):
            written = bound.get(data.id)
            if written is None or len(written.literals) != 1:
                bindings = 0 if written is None else len(written.literals)
                unresolved.append(
                    f"line {node.lineno}: data={data.id}, bound to {bindings} dict literals"
                )
            elif written.unreadable:
                unresolved.append(
                    f"line {node.lineno}: data={data.id}, written by "
                    f"{sorted(written.unreadable)}"
                )
            else:
                resolved.append(written.keys)
        else:
            unresolved.append(
                f"line {node.lineno}: data="
                + ("<absent>" if data is None else ast.unparse(data))
            )
    assert unresolved == [], {
        "event_type": event_type,
        "emit_sites_whose_data_the_census_cannot_read": unresolved,
        "hint": "resolve the shape here rather than skipping the site — a skipped emit is a wire "
        "key with no schema diff",
    }
    assert len(resolved) == 1, {
        "event_type": event_type,
        "emit_sites": len(resolved),
        "key_sets": [sorted(item) for item in resolved],
        "hint": "a second emit site is a twin that must be censused too",
    }
    return resolved[0]


def _literal_dict_keys_where(relative_path: str, key: str, value: str) -> list[frozenset[str]]:
    """Key sets of every dict literal in a module whose ``key`` is the constant ``value``.

    Resolved through :func:`_pinned_dict_keys`, so a ``**`` splat or a computed key in one of the
    matched literals is a failure here rather than a key set silently narrower than the record.
    """

    matches: list[frozenset[str]] = []
    for node in ast.walk(_module_tree(relative_path)):
        if not isinstance(node, ast.Dict):
            continue
        for dict_key, dict_value in zip(node.keys, node.values):
            if not (isinstance(dict_key, ast.Constant) and dict_key.value == key):
                continue
            if isinstance(dict_value, ast.Constant) and dict_value.value == value:
                matches.append(
                    _pinned_dict_keys(node, f"{relative_path}: {{{key!r}: {value!r}, ...}}")
                )
    return matches


def _call_dict_arg_keys(relative_path: str, method: str) -> list[frozenset[str]]:
    """Key sets of the single dict literal passed positionally to ``obj.<method>({...})``.

    Strict like :func:`_literal_dict_keys_where` above, and for the same reason: the callers pin
    the result as the whole artifact, so a merge into the literal must fail rather than vanish.
    """

    matches: list[frozenset[str]] = []
    for node in ast.walk(_module_tree(relative_path)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != method or not node.args:
            continue
        if isinstance(node.args[0], ast.Dict):
            matches.append(
                _pinned_dict_keys(node.args[0], f"{relative_path}: .{method}({{...}})")
            )
    return matches


_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _function_node(relative_path: str, name: str, *, within: str | None = None) -> _FunctionNode:
    """The one function/method named ``name`` (optionally inside class/function ``within``)."""

    root: ast.AST = _module_tree(relative_path)
    if within is not None:
        root = next(
            node
            for node in ast.walk(root)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == within
        )
    found = [
        node
        for node in ast.walk(root)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(found) == 1, {"function": name, "definitions_found": len(found)}
    return found[0]


def _live_function_body(function: Any) -> _FunctionNode:
    """Parse the source of the *live* callable, not of the file it was written in.

    ``_module_tree`` censuses a path; this censuses whatever object the name is bound to right
    now. That is the difference between "the repository declares N branches" and "the function
    this suite just called declares N branches", and only the second one survives a wrapper.
    """

    tree = ast.parse(textwrap.dedent(_live_source(function)))
    node = tree.body[0]
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)), type(node)
    return node


def _emitted_result_keys(function: _FunctionNode) -> frozenset[str]:
    """Every string key ``function`` can write into the dict it returns.

    The usage authority used to be a behavioral probe alone: feed one maximal input, collect the
    keys that came back. A branch keyed on an input the probe does not carry
    (``if usage.get("tool_tokens"): normalized["tool_tokens"] = ...``) is invisible to that, so
    the normalizer could grow an eighth emitted key and the census would stay green. This reads
    the *assignable domain* instead: the literal the result is built from, every constant-keyed
    subscript write, and the keys of any dict the function copies in wholesale through a
    ``for key, value in <dict>.items(): result[key] = value`` loop (which is how the priced
    sub-counts actually land). An emit shape it cannot analyze fails rather than passing empty.

    "Fails rather than passing empty" used to hold for exactly one shape — a computed subscript
    key — and the docstring claimed it in general. Three other ways to write a key into a dict
    were simply not looked at, so ``result.update({"tool_tokens": ...})``, ``result |= {...}``
    and any augmented assignment onto it were each an eighth emitted key this census reported as
    nonexistent.

    Then the write census itself turned out to be bound on *one name*. The result dict was
    watched; the dict a copy loop reads was not, so ``details["tool_tokens"] = ...`` written
    ahead of ``for key, value in details.items(): result[key] = value`` put an eighth key on the
    wire through a name nobody censused — and worse, a loop source with no dict literal bound to
    it resolved to the empty set instead of refusing. :func:`_dict_writes` now answers for every
    name, so a loop source resolves to *its* literal keys plus *its* constant writes, and a
    contribution to either dict that cannot be read as constant keys — a splat inside the
    literal, a computed key, a merge from a name, a rebinding, a mutator this census does not
    model — is refused. So is a ``return`` of anything but the counted name: ``return {**result,
    "tool_tokens": n}`` beside it is an emitted key no assignment ever spelled.
    """

    returned = [node.value for node in ast.walk(function) if isinstance(node, ast.Return)]
    named = [node for node in returned if isinstance(node, ast.Name)]
    assert len(named) == 1, {"returns_of_a_named_dict": len(named)}
    result_name = named[0].id

    unanalyzable: list[str] = [
        f"return {ast.unparse(node)}"
        for node in returned
        if node is not None and not isinstance(node, ast.Name)
    ]

    # ``for key, value in <name>.items(): result[key] = value`` -> key is bound to <name>'s keys.
    loop_sources: dict[str, str] = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Tuple):
            continue
        names = [element.id for element in node.target.elts if isinstance(element, ast.Name)]
        iterated = node.iter
        if (
            names
            and isinstance(iterated, ast.Call)
            and isinstance(iterated.func, ast.Attribute)
            and iterated.func.attr == "items"
            and isinstance(iterated.func.value, ast.Name)
        ):
            loop_sources[names[0]] = iterated.func.value.id

    writes = _dict_writes(function)
    nothing = _DictWrites("", (), frozenset(), (), ())

    def _source_keys(source: str) -> frozenset[str]:
        """The keys a ``<source>.items()`` copy loop can carry into the result."""

        written = writes.get(source, nothing)
        if not written.literals:
            unanalyzable.append(
                f"for key, value in {source}.items(): no dict literal is bound to {source}"
            )
            return frozenset()
        unanalyzable.extend(written.unreadable)
        return written.keys

    result = writes.get(result_name, nothing)
    unanalyzable.extend(result.refusals)
    keys = set(result.keys)
    for index in result.dynamic:
        if isinstance(index, ast.Name) and index.id in loop_sources:
            keys |= _source_keys(loop_sources[index.id])
        else:
            unanalyzable.append(f"{result_name}[{ast.unparse(index)}] = ...")
    assert unanalyzable == [], {
        "writes_the_census_cannot_read": sorted(set(unanalyzable)),
        "hint": "a new emit shape: teach _dict_writes about it rather than dropping it — "
        "an unread write is an emitted key the whole usage census below never hears about",
    }
    return frozenset(keys)


# The gateway module's wire-reading helpers. A function that both constructs a
# ``ModelAdapterError`` and reads the wire through one of these *is* an error reader.
#
# This is a HAND list, and every census below that resolves a key through it inherits its
# blindness: ``_literal_wire_keys`` picks a reader's keys out of the arguments it hands these
# helpers, so a ninth helper carrying a ninth key would make that key invisible and the pinned
# read-set would still match. ``test_2b_the_helper_list_is_every_wire_reading_helper_the_readers_use``
# below derives the same set from the module and diffs it against this one.
GATEWAY_WIRE_READ_HELPERS = frozenset(
    {
        "_gateway_string",
        "_gateway_fragment_string",
        "_exact_gateway_bool",
        "_exact_gateway_int",
        "_gateway_usage",
        "_gateway_http_status_hint",
        "_reported_error_usage",
        "_portable_gateway_payload",
    }
)

# The helpers a derived scan can find: they take the wire mapping itself and read a key off it
# (a literal one, or the ``key`` parameter their callers pass). Pinned in full, so a new
# ``_gateway_float(payload, key, *, http_status)`` carrying a new key fails here.
GATEWAY_MAPPING_READ_HELPERS = frozenset(
    {
        "_exact_gateway_bool",
        "_exact_gateway_int",
        "_gateway_fragment_string",
        "_gateway_http_status_hint",
        "_gateway_string",
        "_reported_error_usage",
    }
)
# The two registered helpers no mapping scan can reach: they validate an already-extracted
# ``value: Any``, so they carry no wire key of their own and the caller names the key.
GATEWAY_WIRE_VALUE_VALIDATORS = frozenset({"_gateway_usage", "_portable_gateway_payload"})


def _literal_wire_keys(function: ast.AST) -> frozenset[str]:
    """Wire keys named literally in ``function``: ``x.get("k")``, ``x["k"]``, ``"k" in x``, and
    the literal key arguments it hands to the module's wire-reading helpers."""

    keys: set[str] = set()
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            keys.add(node.args[0].value)
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                keys.add(node.slice.value)
        elif (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.In)
            and isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, str)
        ):
            keys.add(node.left.value)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in GATEWAY_WIRE_READ_HELPERS
        ):
            for argument in node.args[1:]:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    keys.add(argument.value)
    return frozenset(keys)


_HELPER_INTERNAL_READS: dict[str, frozenset[str]] = {}


def _wire_keys_read_in(function: _FunctionNode) -> frozenset[str]:
    """Every wire key ``function`` reads, including the ones a helper reads on its behalf.

    ``_reported_error_usage(payload)`` names no key at the call site -- the ``"usage"`` literal
    lives inside the helper -- so a purely local scan reported the caller as never reading a key
    it does read. One level of delegation is resolved, which is exactly as far as this module
    delegates.
    """

    keys = set(_literal_wire_keys(function))
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in GATEWAY_WIRE_READ_HELPERS
        ):
            helper = node.func.id
            if helper not in _HELPER_INTERNAL_READS:
                _HELPER_INTERNAL_READS[helper] = _literal_wire_keys(
                    _function_node("providers/gateway.py", helper)
                )
            keys |= _HELPER_INTERNAL_READS[helper]
    return frozenset(keys)


def _all_functions(relative_path: str) -> dict[str, list[_FunctionNode]]:
    """Every function in the module *including* methods, keyed by name.

    ``ast.walk`` rather than ``tree.body``: a scan of module-level definitions only is a
    predicate on where a function was written, not on what it does, and moving a reader into a
    class body is not a change to the wire it reads.
    """

    functions: dict[str, list[_FunctionNode]] = {}
    for node in ast.walk(_module_tree(relative_path)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.setdefault(node.name, []).append(node)
    return functions


def _called_local_names(function: ast.AST) -> frozenset[str]:
    return frozenset(
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )


def _constructs_directly(function: ast.AST, class_name: str) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == class_name
        for node in ast.walk(function)
    )


def _parameters(function: _FunctionNode) -> frozenset[str]:
    arguments = function.args
    return frozenset(
        argument.arg
        for argument in (
            list(arguments.posonlyargs) + list(arguments.args) + list(arguments.kwonlyargs)
        )
    )


def _mapping_parameters(function: _FunctionNode) -> frozenset[str]:
    """Parameters *annotated* as a mapping.

    Kept annotation-based on purpose, and used for one thing only: asserting that the two
    registered value validators still take an already-extracted value. What a function *does*
    is :func:`_reads_a_mapping_parameter`'s question, and it is a different one.
    """

    arguments = function.args
    return frozenset(
        argument.arg
        for argument in (
            list(arguments.posonlyargs) + list(arguments.args) + list(arguments.kwonlyargs)
        )
        if argument.annotation is not None
        and any(
            token in ast.unparse(argument.annotation)
            for token in ("Mapping", "dict", "Dict")
        )
    )


def _reads_a_mapping_parameter(function: _FunctionNode) -> bool:
    """``payload[k]`` / ``payload.get(k)`` / ``k in payload`` on *any* parameter.

    The key may be a literal or the ``key`` parameter the caller passes: both make the function
    the place a wire key is actually read, which is what the helper list is for.

    It used to require the parameter's *annotation* to name a mapping type, which made this a
    predicate on how a helper was spelled rather than on what it does: ``payload: Any`` — the
    annotation the two registered value validators already carry — read a wire key without ever
    being a wire reader. A parameter that is subscripted, ``.get``-ed or ``in``-tested is being
    used as a mapping whatever it was annotated as, and that is the whole test.
    """

    parameters = _parameters(function)
    if not parameters:
        return False
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in parameters
        ):
            return True
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in parameters
        ):
            return True
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.In)
            and isinstance(node.comparators[0], ast.Name)
            and node.comparators[0].id in parameters
        ):
            return True
    return False


# --------------------------------------------------------------------------------------
# The write census, censused
# --------------------------------------------------------------------------------------

# Synthetic on purpose, and the only tests in this file that are. Every mechanism above answers
# a question about code that does not exist yet -- the drift a future commit writes -- and the
# failure this suite keeps re-earning is a census that *answers* where it should refuse. Pinning
# the refusal on today's normalizer alone cannot show that: today's normalizer is readable, so
# the reader returns the same seven keys whether or not the refusals exist at all.


def _parsed_function(source: str) -> _FunctionNode:
    node = ast.parse(textwrap.dedent(source).strip()).body[0]
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)), type(node)
    return node


# Each of these was answered — silently, with a subset — before round 3.
_UNREADABLE_RESULT_SOURCES: dict[str, str] = {
    "a copy loop over a source with no dict literal bound to it": """
        def build(payload):
            result = {"input_tokens": 1}
            details = collect(payload)
            for key, value in details.items():
                result[key] = value
            return result
    """,
    "a computed subscript key written into the copy loop's source": """
        def build(name):
            result = {"input_tokens": 1}
            details = {"audio_tokens": 2}
            details[name] = 3
            for key, value in details.items():
                result[key] = value
            return result
    """,
    "a merge into the copy loop's source from a name": """
        def build(extra):
            result = {"input_tokens": 1}
            details = {"audio_tokens": 2}
            details.update(extra)
            for key, value in details.items():
                result[key] = value
            return result
    """,
    "a splat inside the result literal": """
        def build(extra):
            result = {"input_tokens": 1, **extra}
            return result
    """,
    "a second return that is not the counted name": """
        def build(flag, value):
            result = {"input_tokens": 1}
            if flag:
                return {**result, "tool_tokens": value}
            return result
    """,
    # Round 4. Every entry above is a *write through the name*; the two shapes below are the two
    # ways the dict leaves the name, and round 3 refused neither. The first is the realistic one —
    # the censused package folds one mapping into another exactly this way
    # (``_accumulate_usage_mapping(state.total_usage, billed)``), so the drift needs no new idiom.
    "the result handed to a call that can mutate it": """
        def build(payload):
            result = {"input_tokens": 1}
            enrich(result, payload)
            return result
    """,
    "the result aliased to a second name written through": """
        def build(value):
            result = {"input_tokens": 1}
            alias = result
            alias["tool_tokens"] = value
            return result
    """,
    # ...and the twin of the first, on the other name this census answers for: refusing the
    # argument position only where the *returned* dict is handed over would reproduce the
    # one-of-two-halves shape the round-3 repair was built to end.
    "the copy loop's source handed to a call": """
        def build(payload):
            result = {"input_tokens": 1}
            details = {"audio_tokens": 2}
            enrich(details, payload)
            for key, value in details.items():
                result[key] = value
            return result
    """,
}


@pytest.mark.parametrize("shape", sorted(_UNREADABLE_RESULT_SOURCES))
def test_the_assignable_domain_reader_refuses_the_writes_it_cannot_read(shape: str) -> None:
    with pytest.raises(AssertionError):
        _emitted_result_keys(_parsed_function(_UNREADABLE_RESULT_SOURCES[shape]))


# The other half, and the one that matters more: a shape it *can* read must be read in full,
# because a refusal and a subset are both failures of the same census. The first entry is the
# round-3 drift itself -- a key written into the dict the copy loop reads -- and it has to come
# back as an emitted key so the usage diff downstream fails on it, not as a refusal here.
_READABLE_RESULT_SOURCES: dict[str, tuple[str, frozenset[str]]] = {
    "a constant key written into the copy loop's source": (
        """
        def build():
            result = {"input_tokens": 1}
            details = {"audio_tokens": 2}
            details["tool_tokens"] = 3
            for key, value in details.items():
                result[key] = value
            return result
        """,
        frozenset({"input_tokens", "audio_tokens", "tool_tokens"}),
    ),
    "an analyzable update() merge into the result": (
        """
        def build(value):
            result = {"input_tokens": 1}
            result.update({"tool_tokens": value})
            return result
        """,
        frozenset({"input_tokens", "tool_tokens"}),
    ),
    "an analyzable |= merge into the result": (
        """
        def build(value):
            result = {"input_tokens": 1}
            result |= {"tool_tokens": value}
            return result
        """,
        frozenset({"input_tokens", "tool_tokens"}),
    ),
    # The read-only half of round 4's call-argument rule: a builtin that cannot reach a mapping's
    # mutators is a read, so refusing it would make the census answer "unreadable" about code that
    # is perfectly readable — a refusal and a subset are both failures of the same census.
    "the result handed to read-only builtins": (
        """
        def build():
            result = {"input_tokens": 1}
            if len(result) and sorted(result) and dict(result):
                result["tool_tokens"] = 2
            return result
        """,
        frozenset({"input_tokens", "tool_tokens"}),
    ),
}


def test_the_assignable_domain_reader_reads_the_writes_it_can() -> None:
    for shape, (source, expected) in sorted(_READABLE_RESULT_SOURCES.items()):
        assigned = _emitted_result_keys(_parsed_function(source))
        assert assigned == expected, {
            "shape": shape,
            "missing": sorted(expected - assigned),
            "extra": sorted(assigned - expected),
        }


# R2's twin: the ``data=<name>`` emit-site resolver reads bindings through the same mechanism,
# and used to have no refusal in it at all — it collected dict literals and constant-key
# subscript writes and dropped every other contribution, so a merge put a key on the wire that
# the schema diff below never saw.
_UNREADABLE_BINDING_SOURCES: dict[str, str] = {
    "a merge from a name": 'data = {"error": 1}\ndata.update(extra)\n',
    "an |= merge from a name": 'data = {"error": 1}\ndata |= extra\n',
    "a splat inside the literal": 'data = {"error": 1, **extra}\n',
    "a mutator this census does not model": 'data = {"error": 1}\ndata.setdefault("k", 2)\n',
    "a rebinding to something unreadable": 'data = {"error": 1}\ndata = enrich(data)\n',
    # Round 4, and the pair the round-3 rebinding entry above just misses: ``data = enrich(data)``
    # refused because the NAME was rebound, while the same call one character shorter left the
    # name bound to a dict this census had stopped being able to read.
    "a call that can mutate what it was handed": 'data = {"error": 1}\nenrich(data)\n',
    "an alias written through": 'data = {"error": 1}\nalias = data\nalias["attempt"] = 2\n',
}


def test_the_emit_binding_resolver_refuses_and_folds_like_the_result_census() -> None:
    for shape, source in sorted(_UNREADABLE_BINDING_SOURCES.items()):
        written = _dict_writes(ast.parse(source))["data"]
        assert written.unreadable, {
            "shape": shape,
            "resolved_to": sorted(written.keys),
            "hint": "an emit site resolving through this binding would report a subset of the "
            "keys it puts on the wire",
        }
    folded = _dict_writes(ast.parse('data = {"error": 1}\ndata.update({"attempt": 2})\n'))["data"]
    assert folded.unreadable == () and folded.keys == {"error", "attempt"}
    # The two argument positions the call rule must *not* refuse, or it refuses the census's own
    # read: the emit site whose binding ``_emit_data_keys`` resolves, and a read-only builtin.
    emitted = _dict_writes(
        ast.parse('data = {"error": 1}\nrecorder.emit("run.failed", data=data)\nlen(data)\n')
    )["data"]
    assert emitted.unreadable == () and emitted.keys == {"error"}, {
        "resolved_to": sorted(emitted.keys),
        "refused": sorted(emitted.unreadable),
        "hint": "refusing the emit binding would make every data=<name> site unresolvable",
    }


# The fourth resolution path onto the same rule, and the last one still reading the weak half of
# it: the extractors that pull a literal out of an *argument* (the transcript records matched by
# ``kind="model_turn"``, the ``write_failure`` bundle, the manifest's ``limits=``). Each hands its
# answer to an equality pin, so a splat added to one of those literals used to widen the artifact
# while the pin stayed green.
_UNPINNABLE_LITERALS: dict[str, str] = {
    "a splat merging a mapping of unknown width": '{"kind": "model_turn", **_error_fields(exc)}',
    "a computed key": '{"kind": "model_turn", name: value}',
}


def _parsed_literal(source: str) -> ast.Dict:
    expression = ast.parse(source).body[0]
    assert isinstance(expression, ast.Expr) and isinstance(expression.value, ast.Dict)
    return expression.value


def test_the_pinned_literal_extractor_refuses_what_it_cannot_read_in_full() -> None:
    for shape, source in sorted(_UNPINNABLE_LITERALS.items()):
        with pytest.raises(AssertionError):
            _pinned_dict_keys(_parsed_literal(source), shape)
    readable = _parsed_literal('{"kind": "model_turn", "step": step}')
    assert _pinned_dict_keys(readable, "readable") == {"kind", "step"}


# --------------------------------------------------------------------------------------
# Family 1 — Suspension
# --------------------------------------------------------------------------------------

# result.py:suspension_checkpoint_payload — ``turn`` is a projection artifact of local paths and
# metrics, so the durable observation deliberately omits it (registered by-design).
SUSPENSION_PAYLOAD_EXCLUSIONS = frozenset({"turn"})


def test_1a_checkpoint_payload_carries_every_suspension_field_but_the_exclusions() -> None:
    authority = {field.name for field in dataclasses.fields(Suspension)}
    expected = authority - SUSPENSION_PAYLOAD_EXCLUSIONS
    written = set(suspension_checkpoint_payload(_maximal_suspension()))
    assert written == expected, {
        "missing": sorted(expected - written),
        "extra": sorted(written - expected),
    }


def test_1b_every_written_key_is_read_back_non_default() -> None:
    """A parser that ignores a written key leaves no trace in any key set — only in the value."""

    suspension = _maximal_suspension()
    payload = suspension_checkpoint_payload(suspension)
    restored = suspension_from_checkpoint_payload(payload)
    assert restored == dataclasses.replace(suspension, turn=None)

    baseline = Suspension(reason="settled", status="completed")
    dropped = {
        key
        for key in payload
        if key not in {"reason", "status"} and getattr(restored, key) == getattr(baseline, key)
    }
    assert dropped == set(), {
        "written_but_read_back_as_default": sorted(dropped),
        "hint": "the writer emits this key and the parser ignores it",
    }


def test_1b_the_parser_is_lenient_about_absence_by_design() -> None:
    """A pre-v0.21 checkpoint carries neither ``config_recoverable`` nor its siblings.

    The reader defaults every absent key rather than refusing the payload (see the comment on
    ``config_recoverable`` in ``core/result.py:suspension_from_checkpoint_payload``): the
    default *is* what those runs meant, and a checkpoint that cannot be read is a run that
    cannot be recovered. Only ``reason``/``status`` are required, and their vocabularies are
    still validated. Pinned so tightening it reads as the compatibility break it would be.
    """

    minimal = suspension_from_checkpoint_payload({"reason": "settled", "status": "completed"})
    assert minimal == Suspension(reason="settled", status="completed")
    with pytest.raises(ValueError):
        suspension_from_checkpoint_payload({"reason": "invented", "status": "completed"})
    with pytest.raises(ValueError):
        suspension_from_checkpoint_payload({"reason": "settled", "status": "invented"})


def test_1b_one_reason_vocabulary_spelled_in_three_places() -> None:
    """The type, the decoder's guard and the FSM's edge table are three hand-kept copies.

    ``terminal`` is deliberately absent from the FSM map: it is the one reason whose target
    state depends on the suspension's error_code (cancelled vs failed), so
    ``lifecycle.state_from_suspension`` resolves it before consulting the table.
    """

    declared = set(get_args(get_type_hints(Suspension)["reason"]))
    assert declared == set(_SUSPENSION_REASONS), {
        "in_the_type_only": sorted(declared - _SUSPENSION_REASONS),
        "in_the_decoder_guard_only": sorted(_SUSPENSION_REASONS - declared),
    }
    assert set(REASON_TO_STATE) == declared - {"terminal"}, {
        "unmapped_reasons": sorted(declared - {"terminal"} - set(REASON_TO_STATE)),
        "mapped_but_not_a_reason": sorted(set(REASON_TO_STATE) - declared),
    }


# errors.py:TurnNotSettled copies the classification off the suspension it wraps, field by
# field, so a caller can branch without reaching through ``.suspension``. That copy is a hand
# list: a new Suspension classification field is not re-stamped unless someone remembers.
TURN_NOT_SETTLED_RESTAMPED = frozenset(
    {
        "reason",
        "retryable",
        "http_status",
        "config_recoverable",
        # The two the burn-down added to the park and to this copy in one change: the blocking
        # facade hands a driver an exception and nothing else, and ``insufficient_quota`` vs
        # ``rate_limit_exceeded`` is the same retryable/http_status pair with opposite answers.
        "provider_error_code",
        "provider_retried",
    }
)


def test_1b_the_not_settled_exception_restamps_exactly_the_registered_fields() -> None:
    suspension = _maximal_suspension()
    exc = TurnNotSettled(suspension)
    stamped = set(vars(exc)) - {"suspension"}
    assert stamped == TURN_NOT_SETTLED_RESTAMPED, {
        "newly_restamped": sorted(stamped - TURN_NOT_SETTLED_RESTAMPED),
        "no_longer_restamped": sorted(TURN_NOT_SETTLED_RESTAMPED - stamped),
    }
    # Every re-stamped name is a Suspension field carrying that suspension's value, so the copy
    # cannot drift into a rename or a stale value.
    fields = {field.name for field in dataclasses.fields(Suspension)}
    assert stamped <= fields
    for name in sorted(stamped):
        assert getattr(exc, name) == getattr(suspension, name), name
    assert exc.suspension is suspension


# core/schemas.py:EVENT_DATA_SCHEMAS["turn.failed"] — additionalProperties is False, so this is
# the total wire domain of the event, not a lower bound.
TURN_FAILED_EVENT_KEYS = frozenset(
    {
        "error",
        "error_code",
        "provider_error_code",
        "retryable",
        "http_status",
        "config_recoverable",
        # The burn-down's two: the attempt already made, and what the refused call already cost.
        # ``provider_usage`` is the KERNEL spelling — the gateway wire's compat-frozen alias for
        # the same fact is ``usage``, and this event names facts, not wire keys (see below).
        "provider_retried",
        "provider_usage",
    }
)


def test_1c_turn_failed_schema_and_emit_site_agree_on_one_key_set() -> None:
    schema = EVENT_DATA_SCHEMAS["turn.failed"]
    declared = frozenset(schema["properties"])
    assert schema["additionalProperties"] is False
    assert declared == TURN_FAILED_EVENT_KEYS, {
        "missing": sorted(TURN_FAILED_EVENT_KEYS - declared),
        "extra": sorted(declared - TURN_FAILED_EVENT_KEYS),
    }
    emitted = _emit_data_keys("loop.py", "turn.failed")
    assert emitted == declared, {
        "emitted_not_declared": sorted(emitted - declared),
        "declared_not_emitted": sorted(declared - emitted),
    }


def test_1c_turn_failed_now_names_every_fact_the_wire_carries() -> None:
    """Cross-check against family 2, so the event and the wire cannot drift independently.

    The event uses the KERNEL spelling of each fact where the wire uses its alias, so the
    comparison runs through :data:`TRANSPORTABLE_ERROR_WIRE_ALIASES` rather than over raw keys.
    """

    declared = frozenset(EVENT_DATA_SCHEMAS["turn.failed"]["properties"])
    wire_facts = {
        alias if alias != "message" else "error"
        for alias in TRANSPORTABLE_ERROR_WIRE_ALIASES.values()
    }

    # Both spellings of the provider code disagree with the wire's, on purpose: the event names
    # the fact (``provider_error_code``) where the compat-frozen wire calls it ``error_code``.
    assert "provider_error_code" in declared
    assert "provider_error_code" in wire_facts

    # The event named this from the start and the wire could not; the burn-down gave the wire a
    # key without touching the event, which is the whole point of comparing them here.
    assert "config_recoverable" in declared
    assert "config_recoverable" in wire_facts

    # And if a *next* fact joins the uncarried set, this event must still declare it.
    assert TRANSPORTABLE_ERROR_UNCARRIED <= declared

    # ...and the event now names every one of them. It used to drop the retry and the cost, and
    # the transcript record written on the very same failure kept both -- so the event was the
    # odd one out rather than the rule, which is what the burn-down closed.
    assert wire_facts - declared == set(), {
        "carried_by_the_wire_and_not_by_the_event": sorted(wire_facts - declared),
    }
    # The event spells each fact by its KERNEL name. ``usage`` is the wire's compat-frozen alias
    # for ``provider_usage``, exactly as ``error_code`` is the wire's alias for
    # ``provider_error_code``, and neither alias has a slot on this event.
    assert "usage" not in declared
    assert "provider_usage" in declared


# core/schemas.py:TRANSCRIPT_RECORD_SCHEMA, model_turn branch. ``config_recoverable`` was written
# by loop.py:_apump_turn and undeclared here, and the record stayed valid only because the branch
# sets additionalProperties=True -- so the pin below is now "declares exactly what is written",
# and a *next* undeclared key fails on it rather than being absorbed by the open cap.
TRANSCRIPT_MODEL_TURN_DECLARED = frozenset(
    {
        "kind",
        "step",
        "response_id",
        "final_text",
        "tool_calls",
        "usage",
        "error",
        "error_code",
        "provider_error_code",
        "retryable",
        "config_recoverable",
        "provider_retried",
        "http_status",
    }
)


def _transcript_model_turn_branch() -> dict[str, Any]:
    for variant in TRANSCRIPT_RECORD_SCHEMA["oneOf"]:
        if variant["properties"]["kind"]["const"] == "model_turn":
            return variant
    raise AssertionError("TRANSCRIPT_RECORD_SCHEMA has no model_turn branch")


def test_1d_the_transcript_schema_declares_every_key_the_writer_writes() -> None:
    """The open cap is a tolerance, not a declaration, so what it absorbs is pinned here."""

    branch = _transcript_model_turn_branch()
    declared = frozenset(branch["properties"])
    assert declared == TRANSCRIPT_MODEL_TURN_DECLARED, {
        "missing": sorted(TRANSCRIPT_MODEL_TURN_DECLARED - declared),
        "extra": sorted(declared - TRANSCRIPT_MODEL_TURN_DECLARED),
    }
    written = [
        keys
        for keys in _literal_dict_keys_where("loop.py", "kind", "model_turn")
        if "error" in keys
    ]
    assert len(written) == 1, {"model_turn_failure_records_found": len(written)}
    undeclared = written[0] - declared
    assert undeclared == set(), {
        "written_but_undeclared": sorted(undeclared),
        "hint": "additionalProperties=True keeps the record valid; declare the key anyway",
    }
    assert branch["additionalProperties"] is True


# The other ``kind="model_turn"`` record: the one written when the turn SUCCEEDS. It shares a
# schema branch with the failure record above, so it is the same declared vocabulary minus the
# error half -- ``provider_retried`` included, which is what the burn-down added: the record of a
# retried-then-successful call used to read as a clean single attempt.
TRANSCRIPT_MODEL_TURN_SUCCESS_KEYS = frozenset(
    {"kind", "step", "response_id", "final_text", "tool_calls", "usage", "provider_retried"}
)


def test_1d_the_success_transcript_record_states_the_retry_its_turn_reports() -> None:
    written = [
        keys
        for keys in _literal_dict_keys_where("loop.py", "kind", "model_turn")
        if "error" not in keys
    ]
    assert len(written) == 1, {"model_turn_success_records_found": len(written)}
    assert written[0] == TRANSCRIPT_MODEL_TURN_SUCCESS_KEYS, {
        "missing": sorted(TRANSCRIPT_MODEL_TURN_SUCCESS_KEYS - written[0]),
        "extra": sorted(written[0] - TRANSCRIPT_MODEL_TURN_SUCCESS_KEYS),
    }
    assert written[0] <= frozenset(_transcript_model_turn_branch()["properties"])
    # The fact is stated by the record and present on the object it was written from -- and its
    # failure twin states it too, so the two halves of one artifact cannot drift apart again.
    assert "provider_retried" in {field.name for field in dataclasses.fields(ModelTurn)}
    assert "provider_retried" in written[0]
    failure = [
        keys
        for keys in _literal_dict_keys_where("loop.py", "kind", "model_turn")
        if "error" in keys
    ]
    assert len(failure) == 1 and "provider_retried" in failure[0]


def test_1e_the_checkpoint_alias_is_copied_at_exactly_two_sites() -> None:
    """The rename lives in two hand-written copies: the snapshot and the restore.

    ``http_status`` is spelled ``provider_http_status`` on the checkpoint (registered by-design),
    and a rename is only safe while both ends of it are one edit apart. These are the two, named
    by the census so a third copy -- or a dropped one -- is a failure rather than a silence.
    """

    tree = _module_tree("loop.py")
    copies: dict[str, list[str]] = {"RunCheckpoint": [], "RunState": []}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in copies:
            continue
        for keyword in node.keywords:
            if keyword.arg != "provider_http_status":
                continue
            source = keyword.value
            assert isinstance(source, ast.Attribute), ast.dump(source)
            copies[node.func.id].append(f"{ast.unparse(source)}")
    assert copies == {
        # The snapshot reads the live run state; the restore reads the checkpoint back.
        "RunCheckpoint": ["state.provider_http_status"],
        "RunState": ["cp.provider_http_status"],
    }, {"copy_sites": copies, "hint": "a third copy of the alias, or a lost one"}


def test_1d_the_failure_bundle_and_the_event_beside_it_carry_the_same_status() -> None:
    """loop.py:_record_failure emits run.failed and writes failure.json from the same state.

    The bundle used to drop ``http_status`` while the event kept it, which is the same fact
    written twice from one place and bound once -- so the pin is now an intersection rather than
    an absence, and a key that reaches one of the two and not the other fails here.
    """

    event_keys = _emit_data_keys("loop.py", "run.failed")
    bundles = _call_dict_arg_keys("loop.py", "write_failure")
    assert len(bundles) == 1, {"write_failure_sites": len(bundles)}
    assert "http_status" in event_keys
    assert "http_status" in bundles[0], {
        "hint": "the operator's restore aid must keep the status the log kept",
    }
    shared = {
        "error",
        "error_code",
        "provider_error_code",
        "http_status",
        "retryable",
        "config_recoverable",
        "type",
    }
    assert shared <= (event_keys & bundles[0]), {
        "carried_by_only_one_of_the_two": sorted(shared - (event_keys & bundles[0])),
    }


# The terminal-failure trio, pinned in full rather than probed for one key. ``run.failed`` is
# ``turn.failed``'s terminal twin -- ``fail_recoverable`` promotes one into the other -- and the
# burn-down gave it the classification that promotion used to lose, on the event and on the
# bundle written from the same state in the same breath.
RUN_FAILED_EVENT_KEYS = frozenset(
    {
        "error",
        "error_code",
        "type",
        "provider_error_code",
        "http_status",
        "retryable",
        "config_recoverable",
    }
)
FAILURE_BUNDLE_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "error",
        "error_code",
        "provider_error_code",
        "http_status",
        "retryable",
        "config_recoverable",
        "type",
        "last_good_seq",
        "restore_hint",
    }
)

# The SECOND writer of ``failure.json``, and the reason it is pinned beside the first: the two
# are separate code writing one artifact under one schema id (``monoid.failure.v1``), and the
# reference backend's is the one a worker crash leaves behind -- the case where the bundle is the
# only record there is. It carries no ``provider_error_code`` (its callers hold an exception, not
# the loop's classified run state) and adds ``failed_at``; every other key must match, so a field
# added to one writer and not the other fails here rather than at an operator's console.
REFERENCE_FAILURE_BUNDLE_KEYS = (FAILURE_BUNDLE_KEYS - {"provider_error_code"}) | {"failed_at"}


def test_1d_run_failed_pins_its_schema_its_emit_site_and_its_bundle() -> None:
    schema = EVENT_DATA_SCHEMAS["run.failed"]
    declared = frozenset(schema["properties"])
    assert schema["additionalProperties"] is False
    assert declared == RUN_FAILED_EVENT_KEYS, {
        "missing": sorted(RUN_FAILED_EVENT_KEYS - declared),
        "extra": sorted(declared - RUN_FAILED_EVENT_KEYS),
    }
    emitted = _emit_data_keys("loop.py", "run.failed")
    assert emitted == declared, {
        "emitted_not_declared": sorted(emitted - declared),
        "declared_not_emitted": sorted(declared - emitted),
    }
    bundles = _call_dict_arg_keys("loop.py", "write_failure")
    assert bundles[0] == FAILURE_BUNDLE_KEYS, {
        "missing": sorted(FAILURE_BUNDLE_KEYS - bundles[0]),
        "extra": sorted(bundles[0] - FAILURE_BUNDLE_KEYS),
    }


def test_1d_the_reference_backend_writes_the_same_bundle_the_core_does() -> None:
    """The twin writer of one durable artifact, diffed against the core's key set.

    ``monoid.failure.v1`` has two independent writers and a permissive reader policy, so a field
    that reaches one of them is invisible to the other's consumers and to every schema check --
    which is exactly how the status this commit adds could have landed on one half. The
    difference between them is pinned as a set, not tolerated as "roughly the same".
    """

    writer = _function_node(
        "reference/backend/recovery.py", "write_failure_bundle", within="RecoveryService"
    )
    literals = [node for node in ast.walk(writer) if isinstance(node, ast.Dict)]
    assert len(literals) == 1, {"dict_literals_in_the_writer": len(literals)}
    written = _pinned_dict_keys(
        literals[0], "reference/backend/recovery.py:write_failure_bundle"
    )
    assert written == REFERENCE_FAILURE_BUNDLE_KEYS, {
        "missing": sorted(REFERENCE_FAILURE_BUNDLE_KEYS - written),
        "extra": sorted(written - REFERENCE_FAILURE_BUNDLE_KEYS),
        "hint": "a key reached one failure-bundle writer and not its twin",
    }
    assert "http_status" in written
    # Stated as a difference so a *new* divergence between the two writers is a failure, not a
    # silence: the reference writer holds an exception rather than the loop's classified state.
    core = _call_dict_arg_keys("loop.py", "write_failure")[0]
    assert core - written == {"provider_error_code"}
    assert written - core == {"failed_at"}


def test_1d_the_terminal_failure_record_keeps_the_classification_its_twin_carries() -> None:
    """``turn.failed`` -> ``run.failed`` is a real promotion path; what it loses is pinned.

    ``AgentLoop.fail_recoverable`` exists to turn an exhausted recoverable turn failure into the
    terminal record, and ``close()`` performs the same promotion for an unrecovered park -- so a
    config-recoverable terminal failure is an ordinary outcome, not a hypothetical. The
    classification now survives the promotion; the remainder is stated as a set rather than
    tolerated, so a *new* fact that reaches one event and not the other fails here.

    The two that stop at the turn event are per-CALL facts, not classifications of the run:
    ``provider_usage`` is what one refused call cost (the cumulative lane -- metrics.updated and
    metrics.json -- is where a run's cost is reported, and it already carries these tokens), and
    ``provider_retried`` describes attempts inside one adapter call (recorded per call on the
    transcript record and the receipt). Promoting either onto the terminal record would mean
    publishing one call's number as the run's.
    """

    turn_failed = frozenset(EVENT_DATA_SCHEMAS["turn.failed"]["properties"])
    lost = turn_failed - RUN_FAILED_EVENT_KEYS
    assert lost == {"provider_retried", "provider_usage"}, {
        "lost_on_promotion": sorted(lost),
        "hint": "a fact reached the turn event and not its terminal twin",
    }
    assert {"retryable", "config_recoverable"} <= RUN_FAILED_EVENT_KEYS
    # And the durable park payload keeps both, so the fact exists at the moment of promotion.
    payload = suspension_checkpoint_payload(_maximal_suspension())
    assert {"retryable", "config_recoverable"} <= set(payload)


# --------------------------------------------------------------------------------------
# Family 2 — ModelAdapterError transport
# --------------------------------------------------------------------------------------

SERVER_ERROR_BODY_KEYS = frozenset(TRANSPORTABLE_ERROR_WIRE_ALIASES)

# ``_error_body``'s own signature, pinned. Every parameter carries one wire key: the two
# positionals become ``http_status``/``error``, the keyword-only ones keep their names. So the
# parameter list and the wire key set determine each other, and a parameter added to the writer
# fails here even if no census probe happens to pass it (which is exactly how a half-threaded
# field stayed invisible: the probe below re-implements the call and simply never passed it).
SERVER_ERROR_BODY_POSITIONALS: tuple[str, ...] = ("status", "message")
SERVER_ERROR_BODY_POSITIONAL_WIRE_KEYS = frozenset({"http_status", "error"})


def test_2a_the_error_body_signature_and_the_wire_key_set_determine_each_other() -> None:
    signature = _live_signature(_error_body)
    positional = tuple(
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    keyword_only = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert positional == SERVER_ERROR_BODY_POSITIONALS
    accounted = keyword_only | SERVER_ERROR_BODY_POSITIONAL_WIRE_KEYS
    assert accounted == SERVER_ERROR_BODY_KEYS, {
        "parameters_without_a_wire_key": sorted(accounted - SERVER_ERROR_BODY_KEYS),
        "wire_keys_without_a_parameter": sorted(SERVER_ERROR_BODY_KEYS - accounted),
        "hint": "a new parameter must reach BOTH writers and the alias table, or fail here",
    }


def _maximal_error_body() -> dict[str, Any]:
    exc = _maximal_adapter_error()
    return _error_body(
        _model_error_status(exc),
        str(exc),
        error_code=exc.provider_error_code,
        retryable=exc.retryable,
        config_recoverable=exc.config_recoverable,
        provider_retried=exc.provider_retried,
        usage=provider_usage_of(exc),
    )


def _write_exception_body(exc: Exception) -> tuple[dict[str, Any], HTTPStatus]:
    """Drive the *real* ``_write_exception`` arm and capture what it would have sent.

    ``_maximal_error_body`` re-implements the argument list ``_write_exception`` passes, so a
    field threaded into the writer but not into that copy is invisible to every key diff -- the
    flagship gap could be half-fixed silently. This runs the shipped handler method instead, on
    an instance built without a socket, with only the final JSON write replaced.
    """

    handler_class = make_llm_gateway_handler(
        _NO_GATEWAY_NEEDED,  # type: ignore[arg-type]
        admin_token=None,
    )
    handler = handler_class.__new__(handler_class)
    captured: dict[str, Any] = {}

    def _capture(payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        captured["body"] = payload
        captured["status"] = status

    handler._write_json = _capture  # type: ignore[method-assign]
    handler._write_exception(exc)  # type: ignore[attr-defined]
    return captured["body"], captured["status"]


class _NoGatewayNeeded:
    """``_write_exception`` never touches the gateway the handler factory closes over."""


_NO_GATEWAY_NEEDED = _NoGatewayNeeded()


def test_2a_the_two_server_writers_answer_one_exception_identically() -> None:
    """The twin binding: separate code, one wire contract, so they are diffed against each other.

    ``_error_body`` is shared, but *what each writer passes it* is not -- and that argument list
    is where a field goes missing. Driving both from one exception makes a half-threaded field a
    difference in this dict rather than a silence.
    """

    exc = _maximal_adapter_error()
    written, status = _write_exception_body(exc)
    framed = _stream_error_frame(_StreamFrameHandler(), exc)

    assert framed["type"] == "error"
    assert written == {key: value for key, value in framed.items() if key != "type"}, {
        "non_streamed_body": written,
        "stream_error_frame": framed,
        "hint": "a field reached one server writer and not its twin",
    }
    assert frozenset(written) == SERVER_ERROR_BODY_KEYS, {
        "missing": sorted(SERVER_ERROR_BODY_KEYS - set(written)),
        "extra": sorted(set(written) - SERVER_ERROR_BODY_KEYS),
    }
    assert status == HTTPStatus.TOO_MANY_REQUESTS
    assert written["http_status"] == int(status)
    # The exception's own facts, not the writer's defaults.
    assert written["retryable"] is True
    assert written["config_recoverable"] is True
    assert written["provider_retried"] is True
    assert written["usage"] == provider_usage_of(exc)


def test_2a_a_config_recoverable_refusal_reaches_the_wire_as_a_422_and_as_a_flag() -> None:
    """Both halves of the same statement, pinned on the real writer.

    The status was the *only* thing a client could read this off, and it is a hint rather than a
    statement: a 4xx says the request was at fault, not that configuration fixes it, and a
    refusal raised with no HTTP status at all (an applied-parameters proof failure) had nothing
    to leave behind. Now the classification rides its own key and the status still agrees with
    it, so a change to either half is a failure here.
    """

    refused = ModelAdapterError(
        "the upstream refused the shape",
        provider_error_code="unsupported_request_shape",
        retryable=False,
        config_recoverable=True,
    )
    written, status = _write_exception_body(refused)
    assert status == HTTPStatus.UNPROCESSABLE_ENTITY
    assert written["http_status"] == 422
    assert written["config_recoverable"] is True
    assert TRANSPORTABLE_ERROR_UNCARRIED.isdisjoint(written)


def test_2a_server_error_body_writes_exactly_the_alias_table() -> None:
    body = _maximal_error_body()
    assert frozenset(body) == SERVER_ERROR_BODY_KEYS, {
        "missing": sorted(SERVER_ERROR_BODY_KEYS - set(body)),
        "extra": sorted(set(body) - SERVER_ERROR_BODY_KEYS),
    }
    # No transportable fact is left without a key; a new one must join the table, not this gap.
    assert TRANSPORTABLE_ERROR_UNCARRIED.isdisjoint(body)


class _StreamFrameHandler:
    """A ``_stream_error_frame`` handler is only touched on the non-ModelAdapterError path."""


def test_2a_stream_error_frame_is_the_body_plus_a_type_tag() -> None:
    """The twin writer: separate code, so a field added to one must be added to the other."""

    frame = _stream_error_frame(_StreamFrameHandler(), _maximal_adapter_error())
    expected = SERVER_ERROR_BODY_KEYS | {"type"}
    assert frozenset(frame) == expected, {
        "missing": sorted(expected - set(frame)),
        "extra": sorted(set(frame) - expected),
    }
    assert frame["type"] == "error"
    assert {key: value for key, value in frame.items() if key != "type"} == _maximal_error_body()


def test_2a_usage_is_omitted_when_the_failure_cost_nothing() -> None:
    """By-design omit-when-empty: an error raised before a provider keeps its old wire shape."""

    unstamped = ModelAdapterError("refused before dispatch", provider_error_code="gateway_error")
    body = _error_body(
        _model_error_status(unstamped), str(unstamped), usage=provider_usage_of(unstamped)
    )
    assert "usage" not in body
    assert frozenset(body) == SERVER_ERROR_BODY_KEYS - {"usage"}


# --- the minimal twin of the two maximal writer probes ---------------------------------

# What each shipped writer emits for a failure that carries nothing and cost nothing. The
# maximal probes above are the union of the wire; the difference between the two probes is the
# conditional half, which on this family is exactly ``usage`` (the registered omit-when-empty
# rule, and the only key ``_error_body`` makes conditional).
#
# Family 7 grew this twin in round 2 and family 2 did not, which left a hole with a name: every
# family-2 probe feeds a *maximal* exception, so every fact on it is non-default, so a writer
# that omitted a key whenever its value was falsy was indistinguishable from one that always
# writes it. ``_error_body`` is shared and could not drift that way -- but ``_write_exception``
# and ``_stream_error_frame`` each build their own argument list around it, and an
# omit-when-falsy filter added on *their* side of that call (dropping ``provider_retried=False``,
# say, which the client's default would silently absorb) is exactly the half-threaded-field shape
# this family exists to catch. Both writers are driven, because they are separate code.
GATEWAY_MINIMAL_ERROR_BODY_KEYS = SERVER_ERROR_BODY_KEYS - {"usage"}


def _minimal_adapter_error() -> ModelAdapterError:
    """The other end of ``_maximal_adapter_error``: every transportable fact at its default."""

    return ModelAdapterError("upstream refused before the call cost anything")


def test_2a_the_minimal_failure_pins_which_error_body_keys_are_conditional() -> None:
    """The non-streamed writer, driven through the shipped handler like its maximal twin."""

    exc = _minimal_adapter_error()
    # The probe is only minimal if the exception really is: every fact at its default.
    assert provider_usage_of(exc) == {}
    assert (
        exc.retryable,
        exc.config_recoverable,
        exc.provider_retried,
        exc.provider_error_code,
    ) == (False, False, False, "")

    written, status = _write_exception_body(exc)
    assert frozenset(written) == GATEWAY_MINIMAL_ERROR_BODY_KEYS, {
        "missing": sorted(GATEWAY_MINIMAL_ERROR_BODY_KEYS - set(written)),
        "extra": sorted(set(written) - GATEWAY_MINIMAL_ERROR_BODY_KEYS),
        "hint": "a key that vanished when its value was the default: that is an omit-when-empty "
        "contract, and ``usage`` is the only one this wire has declared",
    }
    assert SERVER_ERROR_BODY_KEYS - GATEWAY_MINIMAL_ERROR_BODY_KEYS == {"usage"}
    # Each surviving key carries the default itself, not an absence the client re-defaults.
    assert status == HTTPStatus.BAD_GATEWAY
    assert written["http_status"] == 502
    assert written["retryable"] is False
    assert written["config_recoverable"] is False
    assert written["provider_retried"] is False
    assert written["error_code"] == gateway_client.GATEWAY_BAD_RESPONSE
    assert written["error"] == str(exc)


def test_2a_the_minimal_failure_reaches_the_stream_writer_identically() -> None:
    """The streamed twin, and the two writers diffed against each other on the minimal input.

    ``test_2a_the_two_server_writers_answer_one_exception_identically`` does this for a maximal
    exception; a filter that only fires on a falsy value needs the minimal one to show.
    """

    exc = _minimal_adapter_error()
    framed = _stream_error_frame(_StreamFrameHandler(), exc)
    expected = GATEWAY_MINIMAL_ERROR_BODY_KEYS | {"type"}
    assert frozenset(framed) == expected, {
        "missing": sorted(expected - set(framed)),
        "extra": sorted(set(framed) - expected),
        "hint": "the frame writer and the body writer must be conditional about the same keys",
    }
    assert framed["type"] == "error"
    written, _ = _write_exception_body(exc)
    assert written == {key: value for key, value in framed.items() if key != "type"}, {
        "non_streamed_body": written,
        "stream_error_frame": framed,
        "hint": "a default reached one server writer and not its twin",
    }


def _maximal_wire_body() -> dict[str, Any]:
    """The server body, which is now the whole wire.

    This used to be ``_maximal_error_body()`` plus a hand-added ``config_recoverable`` — the one
    fact the writers could not emit, added here so a reader that *did* read it would show up in
    the behavioral census. The writers emit it now, so the two bodies are the same dict and the
    reader censuses below read exactly what the server writes.
    """

    return _maximal_error_body()


def _read_r1(body: dict[str, Any]) -> ModelAdapterError:
    with pytest.raises(ModelAdapterError) as caught:
        gateway_client._parse_gateway_response(dict(body))
    return caught.value


def _read_r2(body: dict[str, Any]) -> ModelAdapterError:
    with pytest.raises(ModelAdapterError) as caught:
        gateway_client._chunk_from_event({"type": "error", **body})
    return caught.value


def _read_r3(body: dict[str, Any]) -> ModelAdapterError:
    return gateway_client._error_from_status_body(int(body["http_status"]), json.dumps(body))


GATEWAY_ERROR_READERS = {
    "providers/gateway.py:_parse_gateway_response": _read_r1,
    "providers/gateway.py:_chunk_from_event": _read_r2,
    "providers/gateway.py:_error_from_status_body": _read_r3,
}


@pytest.mark.parametrize("reader", sorted(GATEWAY_ERROR_READERS))
def test_2b_every_reader_reconstructs_the_same_facts_and_loses_the_same_ones(reader: str) -> None:
    """Behavioral census: a dropped read is invisible in a key diff, visible in the attributes."""

    body = _maximal_wire_body()
    got = GATEWAY_ERROR_READERS[reader](body)

    carried = {
        "provider_error_code": got.provider_error_code,
        "retryable": got.retryable,
        "config_recoverable": got.config_recoverable,
        "http_status": got.http_status,
        "provider_retried": got.provider_retried,
        "provider_usage": provider_usage_of(got),
    }
    assert carried == {
        "provider_error_code": body["error_code"],
        "retryable": body["retryable"],
        "config_recoverable": body["config_recoverable"],
        "http_status": body["http_status"],
        "provider_retried": body["provider_retried"],
        "provider_usage": body["usage"],
    }, {"reader": reader, "reconstructed": carried}

    # The message survives, but R3 wraps it with the status line it was read from.
    assert body["error"] in str(got)

    # The registered loss. It flips to a failure the moment a reader starts binding it.
    assert got.error_code == ModelAdapterError.error_code, (
        "the kernel error_code has no wire slot; it must reconstruct to the class default"
    )
    # An instance attribute rather than a class default, so the read above is a real read.
    assert "config_recoverable" not in vars(type(got))
    assert "config_recoverable" in vars(got)


def test_2b_r3_takes_http_status_from_the_status_line_not_the_body() -> None:
    """The registered per-reader quirk: R3's status argument wins over the body's key."""

    body = _maximal_wire_body()
    got = gateway_client._error_from_status_body(500, json.dumps(body))
    assert body["http_status"] == 429
    assert got.http_status == 500


def test_2b_r3_defaults_retryable_from_the_status_when_the_body_is_silent() -> None:
    """R1/R2 default retryable to False; R3 derives it, so a silent 503 reads retryable."""

    silent = {key: value for key, value in _maximal_wire_body().items() if key != "retryable"}
    assert gateway_client._error_from_status_body(503, json.dumps(silent)).retryable is True
    assert gateway_client._error_from_status_body(400, json.dumps(silent)).retryable is False
    with pytest.raises(ModelAdapterError) as caught:
        gateway_client._parse_gateway_response(dict(silent))
    assert caught.value.retryable is False


# What each reader reconstructs from a body carrying nothing but the message. These are the
# values a *pre-field* gateway produces, so they are the compatibility contract of every added
# key -- and they are per reader, because R3 derives two of them from the status line it was
# read from while R1/R2 have no status to derive from.
#
# ``config_recoverable`` is in here for a different reason than it used to be: it was the fact no
# reader read, and its ``False`` was the constructor default. It is now READ, with an explicit
# ``default=False``, on all three readers -- so these entries are the compatibility contract of
# the added key (an older gateway omits it and every reader answers "not config-fixable"), and a
# reader that derived it from the status the way R3 derives ``retryable`` would fail here.
SILENT_BODY_DEFAULTS: dict[str, dict[str, Any]] = {
    "providers/gateway.py:_parse_gateway_response": {
        "provider_error_code": gateway_client.GATEWAY_BAD_RESPONSE,
        "retryable": False,
        "http_status": None,
        "provider_retried": False,
        "config_recoverable": False,
        "provider_usage": {},
    },
    "providers/gateway.py:_chunk_from_event": {
        "provider_error_code": gateway_client.GATEWAY_BAD_RESPONSE,
        "retryable": False,
        "http_status": None,
        "provider_retried": False,
        "config_recoverable": False,
        "provider_usage": {},
    },
    # Read from a 503: ``retryable`` and the provider code are derived from the status, and
    # ``http_status`` is the status line rather than an absent body key.
    "providers/gateway.py:_error_from_status_body": {
        "provider_error_code": gateway_client.GATEWAY_SERVER_ERROR,
        "retryable": True,
        "http_status": 503,
        "provider_retried": False,
        "config_recoverable": False,
        "provider_usage": {},
    },
}

_SILENT_BODY = {"error": "upstream refused with nothing else to say"}


def _read_silent_r1() -> ModelAdapterError:
    return _read_r1(dict(_SILENT_BODY))


def _read_silent_r2() -> ModelAdapterError:
    return _read_r2(dict(_SILENT_BODY))


def _read_silent_r3() -> ModelAdapterError:
    return gateway_client._error_from_status_body(503, json.dumps(_SILENT_BODY))


SILENT_BODY_READERS = {
    "providers/gateway.py:_parse_gateway_response": _read_silent_r1,
    "providers/gateway.py:_chunk_from_event": _read_silent_r2,
    "providers/gateway.py:_error_from_status_body": _read_silent_r3,
}


@pytest.mark.parametrize("reader", sorted(SILENT_BODY_READERS))
def test_2b_every_reader_defaults_an_absent_key_the_registered_way(reader: str) -> None:
    """The absent-key half of the reader census, over all three readers rather than two.

    A default is a contract: it is what every gateway that predates a field answers, so
    changing one silently reclassifies existing traffic (a ``retryable`` default flipped on the
    stream reader alone turns every terse stream error into a retry the other transports do not
    make). Pinning R1 and R3 while leaving R2 free is how that asymmetry stayed invisible.
    """

    got = SILENT_BODY_READERS[reader]()
    reconstructed = {
        "provider_error_code": got.provider_error_code,
        "retryable": got.retryable,
        "http_status": got.http_status,
        "provider_retried": got.provider_retried,
        "config_recoverable": got.config_recoverable,
        "provider_usage": provider_usage_of(got),
    }
    assert reconstructed == SILENT_BODY_DEFAULTS[reader], {
        "reader": reader,
        "reconstructed": reconstructed,
        "hint": "a default changed: that is a wire-compatibility change, not a refactor",
    }
    assert _SILENT_BODY["error"] in str(got)


# --- the reader set itself, and what each reader reads ---------------------------------

# Full read-key census per reader, pinned. A reader that grows a read of a key no writer
# emits, or stops reading one, fails here.
GATEWAY_READER_WIRE_KEYS: dict[str, frozenset[str]] = {
    "providers/gateway.py:_parse_gateway_response": frozenset(
        {
            # error half
            "error",
            "error_code",
            "retryable",
            "config_recoverable",
            "http_status",
            "provider_retried",
            "usage",
            # success half
            "tool_calls",
            "arguments",
            "id",
            "call_id",
            "name",
            "stop_reason",
            "response_id",
            "turn_handle",
            "final_text",
        }
    ),
    "providers/gateway.py:_chunk_from_event": frozenset(
        {
            "error",
            "error_code",
            "retryable",
            "config_recoverable",
            "http_status",
            "provider_retried",
            "usage",
            "type",
            "text",
            "index",
            "arguments_fragment",
            "id",
            "name",
            "stop_reason",
            "response_id",
            "turn_handle",
            "generation_applied",
            "schema_applied",
        }
    ),
    # Reads no ``http_status``: the status line it was handed wins (registered quirk above).
    # ``usage`` is read for it by ``_reported_error_usage``, not named at the call site.
    "providers/gateway.py:_error_from_status_body": frozenset(
        {"error", "error_code", "retryable", "config_recoverable", "provider_retried", "usage"}
    ),
}


def _discovered_gateway_error_readers() -> set[str]:
    """Every function in the module that turns the wire into a ``ModelAdapterError``.

    Two predicates, each widened after a drift walked straight past the old one:

    *Where it is written.* The scan read ``tree.body`` — module-level definitions only — so a
    reader born as a method inside a ``ClassDef`` was not a reader. ``_all_functions`` walks the
    whole tree.

    *How it raises.* "Constructs a ``ModelAdapterError``" meant a literal ``ModelAdapterError(...)``
    call in the function's own body, so a reader that hands the job to a one-line local factory
    evaded by moving the constructor one frame away. One level of delegation is resolved, which
    is as far as this module delegates anywhere else (see ``_wire_keys_read_in``).

    *How it reads.* "Reads the wire" meant *calling one of the registered helpers* — a spelled
    predicate, and the last one left in this discovery. A reader that does its own
    ``payload.get("error")`` and raises on it reads the wire by any definition and called no
    helper, so it was not a reader; it is now, because reading a key off a parameter used as a
    mapping is the same act whether a helper or the reader itself performs it. The registered
    helpers are excluded by name: every one of them reads a key and raises, and a helper is what
    the *other* census in this family is for
    (``test_2b_the_helper_list_is_every_wire_reading_helper_the_readers_use``), so counting them
    here would only move the hand list, not derive it.
    """

    functions = _all_functions("providers/gateway.py")
    constructing = {
        name
        for name, nodes in functions.items()
        if any(_constructs_directly(node, "ModelAdapterError") for node in nodes)
    }
    discovered: set[str] = set()
    for name, nodes in functions.items():
        if name in GATEWAY_WIRE_READ_HELPERS:
            continue
        for node in nodes:
            raises_a_model_error = _constructs_directly(node, "ModelAdapterError") or bool(
                (_called_local_names(node) & constructing) - {name}
            )
            reads_the_wire = bool(
                _called_local_names(node) & GATEWAY_WIRE_READ_HELPERS
            ) or _reads_a_mapping_parameter(node)
            if raises_a_model_error and reads_the_wire:
                discovered.add(f"providers/gateway.py:{name}")
    return discovered


def test_2b_the_registered_reader_list_is_every_reader_the_module_has() -> None:
    """The reader list was a hand-written dict of three, so a fourth sibling joined unseen."""

    discovered = _discovered_gateway_error_readers()
    assert discovered == set(GATEWAY_ERROR_READERS), {
        "unregistered_readers": sorted(discovered - set(GATEWAY_ERROR_READERS)),
        "registered_but_no_longer_a_reader": sorted(set(GATEWAY_ERROR_READERS) - discovered),
        "hint": "a fourth reader must join every reader census below, not just this one",
    }
    assert set(GATEWAY_ERROR_READERS) == set(SILENT_BODY_READERS) == set(GATEWAY_READER_WIRE_KEYS)


def test_2b_the_helper_list_is_every_wire_reading_helper_the_readers_use() -> None:
    """The second closed hand list in this family, derived rather than trusted.

    ``GATEWAY_WIRE_READ_HELPERS`` is what ``_literal_wire_keys`` resolves a reader's keys
    through, so a helper missing from it makes the keys it carries invisible to every read-key
    census — and the pinned sets keep matching, because the key was never counted on either
    side. Discovered here: a function that takes the wire mapping, reads a key off it, and is
    called by one of the registered readers.
    """

    functions = _all_functions("providers/gateway.py")
    called_by_readers: set[str] = set()
    for reader in GATEWAY_ERROR_READERS:
        for node in functions[reader.split(":", 1)[1]]:
            called_by_readers |= _called_local_names(node)
    discovered = {
        name
        for name in called_by_readers
        if name in functions
        and any(_reads_a_mapping_parameter(node) for node in functions[name])
    }
    assert discovered == GATEWAY_MAPPING_READ_HELPERS, {
        "newly_reading_the_wire_mapping": sorted(discovered - GATEWAY_MAPPING_READ_HELPERS),
        "no_longer_reading_it": sorted(GATEWAY_MAPPING_READ_HELPERS - discovered),
        "hint": "a new wire-reading helper: add it to GATEWAY_WIRE_READ_HELPERS or every read-key "
        "census below silently stops counting the keys it carries",
    }
    assert discovered <= GATEWAY_WIRE_READ_HELPERS
    # The hand list is exactly the discovered mapping readers plus the two value validators,
    # which take an already-extracted value and therefore name no key of their own.
    assert GATEWAY_MAPPING_READ_HELPERS | GATEWAY_WIRE_VALUE_VALIDATORS == (
        GATEWAY_WIRE_READ_HELPERS
    ), {
        "registered_but_neither_discovered_nor_declared_a_value_validator": sorted(
            GATEWAY_WIRE_READ_HELPERS - GATEWAY_MAPPING_READ_HELPERS - GATEWAY_WIRE_VALUE_VALIDATORS
        ),
    }
    for name in sorted(GATEWAY_WIRE_VALUE_VALIDATORS):
        # Both halves: the annotation still says "already-extracted value", and the body still
        # does not read a key off it. The second is the load-bearing one — ``value: Any`` is
        # exactly the annotation a helper that started subscripting its argument would keep.
        assert _mapping_parameters(functions[name][0]) == frozenset(), {
            "value_validator_now_takes_a_mapping": name,
            "hint": "it reads its own key now: move it to GATEWAY_MAPPING_READ_HELPERS",
        }
        assert not _reads_a_mapping_parameter(functions[name][0]), {
            "value_validator_now_reads_a_key_off_its_argument": name,
            "hint": "it carries a wire key of its own now: move it to "
            "GATEWAY_MAPPING_READ_HELPERS and account for the key it reads",
        }


@pytest.mark.parametrize("reader", sorted(GATEWAY_READER_WIRE_KEYS))
def test_2b_each_reader_reads_exactly_the_keys_the_census_accounts_for(reader: str) -> None:
    """Static twin of the behavioral census: a *read* that no writer answers is a dead read."""

    read = _wire_keys_read_in(_function_node("providers/gateway.py", reader.split(":", 1)[1]))
    expected = GATEWAY_READER_WIRE_KEYS[reader]
    assert read == expected, {
        "reader": reader,
        "newly_read": sorted(read - expected),
        "no_longer_read": sorted(expected - read),
    }
    # No reader may bind a fact the writers cannot emit, stated once per reader rather than
    # inferred. Vacuous while the uncarried set is empty, and the guard the next such fact meets.
    assert read.isdisjoint(TRANSPORTABLE_ERROR_UNCARRIED), {
        "hint": "a reader binds a fact with no wire key: give it one or register the gap",
    }
    # Every error-family key the server can write is read back here (R3's status quirk aside).
    unread = SERVER_ERROR_BODY_KEYS - read
    assert unread <= {"http_status"} and (
        not unread or reader.endswith("_error_from_status_body")
    ), {"reader": reader, "written_but_never_read": sorted(unread)}


def test_2c_round_trip_through_the_hop_loses_exactly_the_registered_facts() -> None:
    """Server writer -> wire -> R1, diffed against the exception the server started from."""

    origin = _maximal_adapter_error()
    restored = _read_r1(_maximal_wire_body())

    preserved = {
        "provider_error_code",
        "retryable",
        "config_recoverable",
        "http_status",
        "provider_retried",
    }
    for attribute in sorted(preserved):
        assert getattr(restored, attribute) == getattr(origin, attribute), attribute
    assert provider_usage_of(restored) == provider_usage_of(origin)

    lost = {
        attribute
        for attribute in vars(origin)
        if attribute not in preserved
        and attribute != "provider_usage"
        and getattr(restored, attribute, None) != getattr(origin, attribute)
    }
    assert lost == {"error_code"}, {
        "lost_across_the_hop": sorted(lost),
        "hint": "a shrinking loss set means a gap was fixed: update EXPECTED and the registry",
    }


GATEWAY_STATUS_FORWARDING_VALIDATORS = (
    "_exact_gateway_bool",
    "_gateway_string",
    "_exact_gateway_int",
    "_gateway_fragment_string",
    "_gateway_usage",
    "_portable_gateway_payload",
)


def test_2c_every_gateway_validator_forwards_the_status_it_knows() -> None:
    """Sibling census: the same defect shape as a missing wire key, one call frame in.

    Two of six carried the parameter, so the *same* malformed payload produced a classified
    failure naming HTTP 400 or one naming nothing at all, decided by which field of it happened
    to be malformed. Pinned as "all six", so a seventh validator that arrives without it fails
    rather than joining a majority.

    A signature pin is the right shape here and an incomplete proof on its own, because a
    parameter can be accepted and dropped:
    ``test_every_gateway_validator_puts_the_status_it_was_given_on_the_error_it_raises``
    (tests/test_gateway_provider.py) drives all six and reads the status back off the raise. Note
    what today's *callers* can supply: the status a reader knows is the error-envelope hint, and
    the four newly-parameterized validators run on the non-error branches, so they are threaded
    the same expression their siblings use and receive ``None`` from it. The asymmetry is what is
    fixed -- a validator that cannot name a status its caller holds -- not a live loss at these
    four call sites.
    """

    forwarding = {
        name
        for name in GATEWAY_STATUS_FORWARDING_VALIDATORS
        if "http_status" in _live_signature(getattr(gateway_client, name)).parameters
    }
    assert forwarding == set(GATEWAY_STATUS_FORWARDING_VALIDATORS), {
        "not_forwarding_http_status": sorted(
            set(GATEWAY_STATUS_FORWARDING_VALIDATORS) - forwarding
        ),
        "hint": "a validator that cannot name the status its caller already read",
    }


def test_2c_the_four_new_status_parameters_did_not_make_a_value_validator_read_a_key() -> None:
    """The helper-census corollary of the fix above, stated where the fix could break it.

    ``_gateway_usage`` and ``_portable_gateway_payload`` are registered as *value* validators:
    they take an already-extracted value, name no wire key of their own, and are excluded from
    the mapping-reader discovery on exactly that basis. Adding a scalar parameter must not change
    that -- and a scalar named ``http_status`` is one edit away from being read off the payload
    instead of passed in.
    """

    functions = _all_functions("providers/gateway.py")
    for name in sorted(GATEWAY_WIRE_VALUE_VALIDATORS):
        node = functions[name][0]
        assert "http_status" in {argument.arg for argument in node.args.kwonlyargs}
        assert _mapping_parameters(node) == frozenset(), {"now_takes_a_mapping": name}
        assert not _reads_a_mapping_parameter(node), {"now_reads_a_key": name}


def test_2c_the_openai_classifier_sets_both_flags_on_every_branch() -> None:
    """Every ModelAdapterError branch of the one adapter that can classify them.

    Pinned per branch, not as a union: a union says "some branch carries the flag", which is the
    half-bound shape this suite exists to catch. Both facts are properties of the call rather
    than of its classification -- whether the SDK already retried it, and whether the remedy is
    the caller's configuration -- so every branch must state both, including the tail that has
    neither a status nor a code to reason from.
    """

    tree = _module_tree("providers/openai.py")
    classifier = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_model_error_from_openai"
    )
    branches = [
        node
        for node in ast.walk(classifier)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelAdapterError"
    ]
    assert len(branches) == 4, {"classification_branches": len(branches)}
    flagged = [
        {
            keyword.arg
            for keyword in branch.keywords
            if keyword.arg in {"provider_retried", "config_recoverable"}
        }
        for branch in branches
    ]
    assert flagged == [{"provider_retried", "config_recoverable"}] * 4, {
        "flags_per_branch": [sorted(names) for names in flagged],
        "hint": "a branch stopped stating one of the two: bind it, do not narrow this pin",
    }
    # And the recoverability verdict comes from one predicate at all four sites rather than four
    # hand-written conditions, which is the only version of this that cannot fall out of step.
    recoverability = {
        ast.unparse(keyword.value).split("(")[0]
        for branch in branches
        for keyword in branch.keywords
        if keyword.arg == "config_recoverable"
    }
    assert recoverability == {"_config_shaped_refusal"}, {
        "config_recoverable_expressions": sorted(recoverability),
    }


# --------------------------------------------------------------------------------------
# Family 3 — usage
# --------------------------------------------------------------------------------------

# reference/llm_gateway/service.py:LlmGatewayUsage — the tenant meter's total emitted domain.
GATEWAY_METER_KEYS = frozenset({"tenant_id", "calls"}) | NORMALIZED_USAGE_KEYS


def test_3a_tenant_meter_sums_every_priced_sub_count() -> None:
    """Behavioral: ``add`` normalizes seven keys and must sum all seven.

    A key diff cannot show this on its own -- the meter used to normalize all seven and add
    three -- so the counts are read back by value against the maximal usage.
    """

    meter = LlmGatewayUsage(tenant_id="tenant-1")
    meter.add(dict(_MAXIMAL_USAGE))
    reported = meter.to_json()
    assert frozenset(reported) == GATEWAY_METER_KEYS, {
        "missing": sorted(GATEWAY_METER_KEYS - set(reported)),
        "extra": sorted(set(reported) - GATEWAY_METER_KEYS),
    }
    assert NORMALIZED_USAGE_KEYS - set(reported) == set()
    summed = {key: reported[key] for key in NORMALIZED_USAGE_KEYS}
    assert summed == _MAXIMAL_USAGE, {
        "metered": summed,
        "hint": "a sub-count is reported as a column and not actually summed into it",
    }


def test_3a_a_sub_count_only_billed_call_is_visible_to_the_meter() -> None:
    """Corollary A8, closed: the meter can now see a cost expressed only in sub-counts.

    ``total_tokens`` stays 0 -- that is what the provider reported, and this ledger does not
    invent a total the provider did not send. What changed is that the priced call is no longer
    invisible: its cost is reported in the columns it was actually expressed in.
    """

    meter = LlmGatewayUsage(tenant_id="tenant-1")
    meter.add({"cache_read_tokens": 5_000, "reasoning_tokens": 900})
    reported = meter.to_json()
    counted = (reported["input_tokens"], reported["output_tokens"], reported["total_tokens"])
    assert counted == (0, 0, 0)
    assert reported["cache_read_tokens"] == 5_000
    assert reported["reasoning_tokens"] == 900
    assert reported["calls"] == 1


# core/schemas.py:EVENT_DATA_SCHEMAS["metrics.updated"] — the whole normalized usage vocabulary
# now, plus the step/tool/web counters. It used to declare ``reasoning_tokens`` alone of the four
# priced sub-counts, so a cache-heavy run's priced detail never reached a live consumer.
METRICS_UPDATED_KEYS = (
    frozenset(
        {
            "step",
            "tool_calls",
            "web_search_calls",
            "web_fetch_calls",
            "web_context_calls",
            "web_failed_calls",
        }
    )
    | NORMALIZED_USAGE_KEYS
)


def test_3b_metrics_updated_declares_every_priced_sub_count() -> None:
    declared = frozenset(EVENT_DATA_SCHEMAS["metrics.updated"]["properties"])
    assert declared == METRICS_UPDATED_KEYS, {
        "missing": sorted(METRICS_UPDATED_KEYS - declared),
        "extra": sorted(declared - METRICS_UPDATED_KEYS),
    }
    assert NORMALIZED_USAGE_KEYS - declared == set(), {
        "emitted_by_the_normalizer_and_undeclared_here": sorted(
            NORMALIZED_USAGE_KEYS - declared
        ),
    }


# The one usage-bearing emit site the ``data={...}`` census cannot see: it passes a *name*
# (``data=metrics_data``), so the literal lives in an assignment several statements earlier and
# a key added there reaches the wire with no schema entry and no census.
#
# The four priced sub-counts are the conditional half: each is written only when the adapter
# reported one, so a run that used no cache does not publish a column of zeros. Everything else
# rides the literal unconditionally.
METRICS_UPDATED_CONDITIONAL_KEYS = frozenset(
    {"cache_read_tokens", "cache_creation_tokens", "reasoning_tokens", "audio_tokens"}
)
METRICS_UPDATED_UNCONDITIONAL_KEYS = METRICS_UPDATED_KEYS - METRICS_UPDATED_CONDITIONAL_KEYS


def test_3b_the_metrics_emit_site_and_its_schema_agree_key_for_key() -> None:
    """Read through :func:`_dict_writes`, so this census fails closed like the two beside it.

    It used to be bespoke: the ``AnnAssign`` literal plus constant-key subscript writes, and
    *nothing else looked at*. That is the same fail-open shape :func:`_emitted_result_keys` had
    already closed and :func:`_emit_data_keys` closed with it — a ``metrics_data.update({...})``
    or a ``{**extra}`` in the literal reached the wire with no schema entry and no failure here.
    """

    # The literal moved out of ``_apump_turn`` when the emission became one writer serving both
    # the settled and the billed-refused path; the census follows the binding to its function.
    pump = _function_node("loop.py", "_emit_metrics_updated")
    written = _dict_writes(pump).get("metrics_data")
    assert written is not None and len(written.literals) == 1, {
        "metrics_data_literals": 0 if written is None else len(written.literals),
    }
    assert written.unreadable == (), {
        "writes_the_census_cannot_read": sorted(written.unreadable),
        "hint": "a contribution to metrics_data this census cannot resolve is a wire key with "
        "no schema diff — teach _dict_writes about it rather than reporting the literal alone",
    }
    conditional = set(written.added)
    assert written.literals[0] == METRICS_UPDATED_UNCONDITIONAL_KEYS, {
        "emitted_always": sorted(written.literals[0]),
        "expected": sorted(METRICS_UPDATED_UNCONDITIONAL_KEYS),
    }
    # Each priced sub-count is added only when the adapter reported one, which is why none of
    # them can be censused off the literal alone.
    assert conditional == METRICS_UPDATED_CONDITIONAL_KEYS, {
        "emitted_conditionally": sorted(conditional),
        "expected": sorted(METRICS_UPDATED_CONDITIONAL_KEYS),
    }
    declared = frozenset(EVENT_DATA_SCHEMAS["metrics.updated"]["properties"])
    assert written.keys == declared, {
        "emitted_not_declared": sorted(written.keys - declared),
        "declared_not_emitted": sorted(declared - written.keys),
    }


def test_3e_the_otel_exporter_emits_the_two_standard_token_attributes_only() -> None:
    """The by-design pin (registered): GenAI semantic conventions define exactly these two.

    Pinned rather than asserted loosely, because the *reason* is portability across collectors:
    widening it is a deliberate decision about a shared convention, and it should have to be
    made here and not arrive as a side effect of adding a sub-count upstream.
    """

    data = {"usage": dict(_MAXIMAL_USAGE), "response_id": "resp_1", "has_final": True}
    chat = _chat_finish_attrs(data)
    subagent = _subagent_finish_attrs({"usage": dict(_MAXIMAL_USAGE), "status": "completed"})
    for attrs in (chat, subagent):
        usage_attrs = {key for key in attrs if key.startswith("gen_ai.usage.")}
        assert usage_attrs == {"gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"}, {
            "usage_attributes": sorted(usage_attrs),
            "hint": "widened? update EXPECTED and drop the by-design registry entry",
        }
    assert chat["gen_ai.usage.input_tokens"] == _MAXIMAL_USAGE["input_tokens"]
    assert chat["gen_ai.usage.output_tokens"] == _MAXIMAL_USAGE["output_tokens"]


def test_3e_the_lenient_error_usage_reader_passes_a_key_no_normalizer_emits() -> None:
    """The registered leniency, pinned as the behavior it is rather than left as prose.

    A malformed usage on an error path must not replace the failure being reported, so this
    reader validates values and not names -- an unknown counter rides through. Tightening it to
    the normalized domain is a real decision (it would silently drop a future sub-count a
    gateway already reports), so it fails here first.
    """

    passed = gateway_client._reported_error_usage({"usage": {"foo_tokens": 5, **_MAXIMAL_USAGE}})
    assert passed == {"foo_tokens": 5, **_MAXIMAL_USAGE}
    assert set(passed) - NORMALIZED_USAGE_KEYS == {"foo_tokens"}
    # Values are still judged: a bool is not a count, on this reader like on the other three.
    assert gateway_client._reported_error_usage({"usage": {"input_tokens": True}}) == {}


def test_3c_the_subagent_rollup_filters_through_the_usage_authority() -> None:
    """The parent budget reads the normalizer's whole domain, and reads it from one place.

    It used to read a literal three-key tuple, so a child's cache and reasoning tokens never
    reached the parent -- an undercount in exactly the aggregate a bound is checked against. The
    fix is a filter through the authority, not a wider hand copy, so this pin is two claims: the
    module constant IS the census's authority, and the roll-up spells no usage key of its own.
    """

    from monoid_agent_kernel.providers._common import (
        NORMALIZED_USAGE_KEYS as declared_authority,
    )

    assert declared_authority == NORMALIZED_USAGE_KEYS, {
        "in_the_module_only": sorted(declared_authority - NORMALIZED_USAGE_KEYS),
        "in_the_census_only": sorted(NORMALIZED_USAGE_KEYS - declared_authority),
        "hint": "the constant beside normalize_usage must be its emitted domain",
    }

    child = _function_node("loop.py", "_run_subagent_child")
    assert "NORMALIZED_USAGE_KEYS" in _names_in(child), {
        "hint": "the roll-up no longer names the authority it is meant to filter through",
    }
    # ...and no hand-written subset of it survives anywhere in the function: a tuple, list or set
    # display of usage keys is the shape this census caught the first time.
    hand_written: list[frozenset[str]] = []
    for node in ast.walk(child):
        if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            continue
        names = {
            element.value
            for element in node.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        }
        if names and names <= NORMALIZED_USAGE_KEYS:
            hand_written.append(frozenset(names))
    assert hand_written == [], {
        "hand_written_usage_key_sets": [sorted(item) for item in hand_written],
        "hint": "a second spelling of the usage domain beside the authority it filters through",
    }


class _SubclassedCount(IntEnum):
    """An ``int`` subclass a provider SDK can plausibly hand back as a token count."""

    SEVEN = 7


def test_3d_the_four_readers_of_one_stamp_agree_about_what_a_count_is() -> None:
    """Same stamp, four predicates, one verdict — pinned so a fifth spelling fails here.

    ``_recordable_usage`` used to answer ``isinstance(value, int)`` where its three siblings
    answer ``type(value) is int``, so an ``IntEnum`` a provider SDK hands back as a token count
    was a recordable usage on one path and no usage at all on the three that consume it — and
    the receipt this one feeds would then reject what it had just accepted.
    """

    from monoid_agent_kernel.core.model_io import ModelCallReceipt

    subclassed = {"input_tokens": _SubclassedCount.SEVEN}
    stamped = ModelAdapterError("billed then refused")
    mark_provider_usage(stamped, dict(subclassed))
    assert vars(stamped)["provider_usage"] == subclassed

    verdicts = {
        "providers/base.py:provider_usage_of": bool(provider_usage_of(stamped)),
        "providers/gateway.py:_reported_error_usage": bool(
            gateway_client._reported_error_usage({"usage": dict(subclassed)})
        ),
        "core/model_io.py:ModelCallReceipt.with_error": bool(
            dict(ModelCallReceipt().with_error(stamped).usage)
        ),
        "model_call.py:_recordable_usage": bool(_recordable_usage(dict(subclassed))),
    }
    assert verdicts == {
        "providers/base.py:provider_usage_of": False,
        "providers/gateway.py:_reported_error_usage": False,
        "core/model_io.py:ModelCallReceipt.with_error": False,
        "model_call.py:_recordable_usage": False,
    }, {
        "verdicts": verdicts,
        "hint": "one stamp, one verdict: a reader that disagrees reclassifies a billed call",
    }


def test_3d_a_bool_is_not_a_count_on_any_reader() -> None:
    """The one thing all four agree on, so the divergence above is about subclasses only."""

    from monoid_agent_kernel.core.model_io import ModelCallReceipt

    stamped = ModelAdapterError("boolean count")
    mark_provider_usage(stamped, {"input_tokens": True})
    assert provider_usage_of(stamped) == {}
    assert gateway_client._reported_error_usage({"usage": {"input_tokens": True}}) == {}
    assert dict(ModelCallReceipt().with_error(stamped).usage) == {}
    assert _recordable_usage({"input_tokens": True}) == {}


# --------------------------------------------------------------------------------------
# Family 4 — applied-echo
# --------------------------------------------------------------------------------------

# reference/llm_gateway/service.py:_applied_echoes — the full key domain it can emit.
APPLIED_ECHO_KEYS = frozenset({"generation_applied", "schema_applied"})


class _NativeEverything:
    """An adapter declaring native support for both echoed features, so both keys are emitted."""

    generation_support = "native"
    structured_output_support = "native"

    def next_turn(self, request: Any) -> Any:  # pragma: no cover - never called
        raise NotImplementedError


def _maximal_echo_request() -> LlmGatewayTurnRequest:
    return LlmGatewayTurnRequest(
        protocol=LLM_TURN_PROTOCOL_VERSION,
        model="gateway-model",
        system_prompt="",
        tools=(),
        reasoning=ReasoningConfig(effort="high", summary="auto", on_unsupported="fail"),
        generation=GenerationConfig(temperature=0.5),
        output_schema={"type": "object"},
    )


def test_4a_applied_echo_domain_is_generation_and_schema_only() -> None:
    request = _maximal_echo_request()
    config = ModelConfig(
        provider="openai",
        model=request.model,
        reasoning=request.reasoning,
        generation=request.generation,
    )
    echoes = _applied_echoes(request, _NativeEverything(), config)
    assert frozenset(echoes) == APPLIED_ECHO_KEYS, {
        "missing": sorted(APPLIED_ECHO_KEYS - set(echoes)),
        "extra": sorted(set(echoes) - APPLIED_ECHO_KEYS),
    }
    # Registered v0.21-track:B1 — reasoning has the same fail/omit contract and no echo.
    assert request.reasoning.on_unsupported == "fail"
    assert "reasoning_applied" not in echoes


def test_4a_a_default_generation_block_is_absent_from_the_config_wire_by_design() -> None:
    """The registered by-design omission, pinned in both directions.

    ``ModelConfig.to_json`` feeds the request digest and the runtime-config semantic hash, so a
    never-configured block must serialize byte-identically to a config predating the field. The
    moment anything is set, the block appears -- which is what makes the omission a sentinel
    rather than a loss.
    """

    default = ModelConfig().to_json()
    assert "generation" not in default, {
        "hint": "always emitted now? that changes every replay key: update the registry entry",
    }
    configured = ModelConfig(generation=GenerationConfig(temperature=0.5)).to_json()
    assert configured["generation"] == {
        "temperature": 0.5,
        "top_p": None,
        "max_output_tokens": None,
        "on_unsupported": "fail",
    }
    assert set(configured) - set(default) == {"generation"}


def test_4a_the_echo_domain_matches_the_writers_assignment_sites() -> None:
    """Static twin of the behavioral probe: a third key added to the function fails here too."""

    source = _live_source(_applied_echoes)
    assigned = {
        node.slice.value
        for node in ast.walk(ast.parse(inspect.cleandoc(source)))
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "echoes"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    assert assigned == set(APPLIED_ECHO_KEYS), {"assigned_in_source": sorted(assigned)}


def test_4b_the_client_terminal_frame_reads_exactly_what_the_server_can_emit() -> None:
    """Wire symmetry: the keys ``_applied_echoes`` writes are the keys the parser reads back."""

    chunk = gateway_client._chunk_from_event(
        {
            "type": "turn_complete",
            "turn_handle": "handle-1",
            "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            "stop_reason": "stop",
            "generation_applied": {"temperature": 0.5},
            "schema_applied": True,
        }
    )
    assert chunk.generation_applied == {"temperature": 0.5}
    assert chunk.schema_applied is True

    read = {
        node.args[0].value
        for node in ast.walk(
            next(
                item
                for item in ast.walk(_module_tree("providers/gateway.py"))
                if isinstance(item, ast.FunctionDef) and item.name == "_chunk_from_event"
            )
        )
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value in {"generation_applied", "schema_applied", "reasoning_applied"}
    }
    assert read == set(APPLIED_ECHO_KEYS), {
        "read_by_the_client": sorted(read),
        "written_by_the_server": sorted(APPLIED_ECHO_KEYS),
    }


# The client's three enforcement sites for the echo protocol. ``_chunk_from_event`` is only the
# *reader*; these are where an unproven turn is actually refused, and a checker wired into one
# of them is a policy that holds on one transport.
ECHO_ENFORCEMENT_SITES: dict[str, int] = {
    # (site -> how many times each checker is called there)
    "providers/gateway.py:GatewayModelAdapter.next_turn": 1,
    # Twice: once on the terminal frame, once on the frameless-stream fallback.
    "providers/gateway.py:GatewayModelAdapter.astream_turn": 2,
}
ECHO_CHECK_FUNCTIONS = frozenset({"_check_generation_applied", "_check_schema_applied"})


def _echo_check_calls(function: _FunctionNode) -> list[str]:
    return [
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("_check_")
        and node.func.id.endswith("_applied")
    ]


def test_4b_the_module_declares_exactly_the_checkers_the_census_knows() -> None:
    declared = {
        node.name
        for node in _module_tree("providers/gateway.py").body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("_check_")
        and node.name.endswith("_applied")
    }
    assert declared == ECHO_CHECK_FUNCTIONS, {
        "new_checker": sorted(declared - ECHO_CHECK_FUNCTIONS),
        "removed_checker": sorted(ECHO_CHECK_FUNCTIONS - declared),
        "hint": "a reasoning_applied checker must be wired at every site below",
    }


@pytest.mark.parametrize("site", sorted(ECHO_ENFORCEMENT_SITES))
def test_4b_every_enforcement_site_runs_every_checker(site: str) -> None:
    """The sync/stream/frameless triple, bound to one another rather than one at a time.

    ``test_4b`` above censuses the stream *reader*; a checker added there and not to the sync
    path (or to the frameless-stream fallback, which is the one an older gateway actually hits)
    is a fail-closed policy that only closes on one of three routes -- the exact asymmetry the
    frameless fallback was added to fix, one field later.
    """

    _, qualified = site.split(":", 1)
    class_name, method = qualified.split(".", 1)
    calls = _echo_check_calls(_function_node("providers/gateway.py", method, within=class_name))
    expected = ECHO_ENFORCEMENT_SITES[site]
    counts = {name: calls.count(name) for name in ECHO_CHECK_FUNCTIONS}
    assert counts == dict.fromkeys(ECHO_CHECK_FUNCTIONS, expected), {
        "site": site,
        "checker_calls": counts,
        "hint": "a checker is wired at this site a different number of times than its twin",
    }
    assert set(calls) == set(ECHO_CHECK_FUNCTIONS)


def test_4b_the_sync_reader_reads_the_same_echo_keys_the_stream_reader_does() -> None:
    """``next_turn`` reads the echo off a response body, ``_chunk_from_event`` off a frame."""

    sync_read = {
        key
        for key in _literal_wire_keys(
            _function_node("providers/gateway.py", "next_turn", within="GatewayModelAdapter")
        )
        if key.endswith("_applied")
    }
    assert sync_read == set(APPLIED_ECHO_KEYS), {
        "read_on_the_sync_transport": sorted(sync_read),
        "read_on_the_stream_transport": sorted(APPLIED_ECHO_KEYS),
    }


# --------------------------------------------------------------------------------------
# Family 5 — the tool catalog
# --------------------------------------------------------------------------------------
#
# The highest-carrier-count family in the repository and, until now, the least covered: one
# ``ToolSpec`` is projected by six independent hand-written dict builders (two request writers,
# one server reader, two record writers, one preview writer), each choosing its own subset. A
# field added to the dataclass reaches none of them, and nothing says so.


def _maximal_tool_spec() -> ToolSpec:
    """Every ToolSpec field at a distinguishable non-default value."""

    def _handler(_context: Any, _args: dict[str, Any]) -> Any:  # pragma: no cover - never run
        raise NotImplementedError

    return ToolSpec(
        id="fs.read",
        description="Read a file from the workspace.",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        capability="fs.read",
        side_effect="write",
        handler=_handler,
        provider_name="workspace_read",
        path_args=("path",),
        preview_kind="shell",
        emits_workspace_diff=True,
        changed_paths_source="result_content",
        result_payload_kind="shell_exec",
        skip_emit_if_background=True,
        guidance={"when": "reading a file"},
        examples=({"path": "notes.md"},),
        annotations={"readOnlyHint": True},
    )


def test_maximal_tool_spec_covers_every_authority_field() -> None:
    built = _maximal_tool_spec()
    default_marker = ToolSpec(
        id="x",
        description="x",
        input_schema={},
        capability="x",
        side_effect="read",
        handler=built.handler,
    )
    required = {"id", "description", "input_schema", "capability", "side_effect", "handler"}
    undistinguished = {
        field.name
        for field in dataclasses.fields(ToolSpec)
        if field.name not in required
        and getattr(built, field.name) == getattr(default_marker, field.name)
    }
    assert undistinguished == set(), {
        "fields_left_at_their_default": sorted(undistinguished),
        "hint": "extend _maximal_tool_spec so every field is distinguishable from its default",
    }
    # ``exported_name`` is a derived property, not a field: it is how ``provider_name`` travels.
    assert built.exported_name == "workspace_read"


TOOL_SPEC_AUTHORITY = frozenset(field.name for field in dataclasses.fields(ToolSpec))

# Shared justifications, so an omission is a stated reason rather than a shrug.
_NOT_SERIALIZABLE = "a Python callable; nothing to put on a wire or in a record"
_ENGINE_DISPATCH = (
    "engine-local dispatch hint: it describes how the kernel runs and previews the call, not "
    "what the model may send, so a provider/gateway request has no slot for it"
)
_MODEL_UNREADABLE = (
    "prompt-side enrichment the kernel renders into the tool surface itself; a provider tool "
    "definition has no field for it"
)


@dataclass(frozen=True)
class ToolSpecCarrier:
    """One projection of a ToolSpec, its wire keys, and what it does with each authority field."""

    carrier: str
    build: Any
    # wire key -> the authority field it carries ("" = a constant this projection adds).
    key_to_field: dict[str, str]
    # authority field -> why this projection omits it.
    omissions: dict[str, str]


TOOL_SPEC_CARRIERS: tuple[ToolSpecCarrier, ...] = (
    ToolSpecCarrier(
        "providers/gateway.py:_gateway_tool_schema",
        gateway_client._gateway_tool_schema,
        {
            "id": "id",
            "name": "provider_name",
            "description": "description",
            "input_schema": "input_schema",
            "capability": "capability",
            "side_effect": "side_effect",
        },
        {
            "handler": _NOT_SERIALIZABLE,
            "path_args": _ENGINE_DISPATCH,
            "preview_kind": _ENGINE_DISPATCH,
            "emits_workspace_diff": _ENGINE_DISPATCH,
            "changed_paths_source": _ENGINE_DISPATCH,
            "result_payload_kind": _ENGINE_DISPATCH,
            "skip_emit_if_background": _ENGINE_DISPATCH,
            "guidance": _MODEL_UNREADABLE,
            "examples": _MODEL_UNREADABLE,
            "annotations": _MODEL_UNREADABLE,
        },
    ),
    ToolSpecCarrier(
        "reference/studio/server.py:_gateway_tool_schema",
        _studio_gateway_tool_schema,
        {
            "id": "id",
            "name": "provider_name",
            "description": "description",
            "input_schema": "input_schema",
            "capability": "capability",
            "side_effect": "side_effect",
        },
        {
            "handler": _NOT_SERIALIZABLE,
            "path_args": _ENGINE_DISPATCH,
            "preview_kind": _ENGINE_DISPATCH,
            "emits_workspace_diff": _ENGINE_DISPATCH,
            "changed_paths_source": _ENGINE_DISPATCH,
            "result_payload_kind": _ENGINE_DISPATCH,
            "skip_emit_if_background": _ENGINE_DISPATCH,
            "guidance": _MODEL_UNREADABLE,
            "examples": _MODEL_UNREADABLE,
            "annotations": _MODEL_UNREADABLE,
        },
    ),
    ToolSpecCarrier(
        "providers/openai.py:_openai_tool_schema",
        _openai_tool_schema,
        {
            "type": "",  # the Responses API's own discriminator
            "name": "provider_name",
            "description": "description",
            "parameters": "input_schema",
        },
        {
            "handler": _NOT_SERIALIZABLE,
            # The provider addresses tools by name; the kernel id has no slot in this schema,
            # which is exactly why ``exported_name`` has to be stable.
            "id": "not carried: the Responses API keys a tool by name, and the kernel id is "
            "recovered from the exported name on the way back",
            "capability": "kernel authorization vocabulary; the provider neither reads nor "
            "enforces it",
            "side_effect": "kernel authorization vocabulary; the provider neither reads nor "
            "enforces it",
            "path_args": _ENGINE_DISPATCH,
            "preview_kind": _ENGINE_DISPATCH,
            "emits_workspace_diff": _ENGINE_DISPATCH,
            "changed_paths_source": _ENGINE_DISPATCH,
            "result_payload_kind": _ENGINE_DISPATCH,
            "skip_emit_if_background": _ENGINE_DISPATCH,
            "guidance": _MODEL_UNREADABLE,
            "examples": _MODEL_UNREADABLE,
            "annotations": _MODEL_UNREADABLE,
        },
    ),
    ToolSpecCarrier(
        "core/manifest.py:_tool_spec_payload",
        _manifest_tool_spec_payload,
        {
            "id": "id",
            "exported_name": "provider_name",
            "description": "description",
            "input_schema": "input_schema",
            "capability": "capability",
            "side_effect": "side_effect",
            "path_args": "path_args",
            "guidance": "guidance",
            "examples": "examples",
            "annotations": "annotations",
        },
        {
            "handler": _NOT_SERIALIZABLE,
            "preview_kind": _ENGINE_DISPATCH,
            "emits_workspace_diff": _ENGINE_DISPATCH,
            "changed_paths_source": _ENGINE_DISPATCH,
            "result_payload_kind": _ENGINE_DISPATCH,
            "skip_emit_if_background": _ENGINE_DISPATCH,
        },
    ),
    ToolSpecCarrier(
        "core/tool_surface.py:_tool_spec_payload",
        _transcript_tool_spec_payload,
        {
            "id": "id",
            "exported_name": "provider_name",
            "description": "description",
            "input_schema": "input_schema",
            "capability": "capability",
            "side_effect": "side_effect",
            "path_args": "path_args",
            "guidance": "guidance",
            "examples": "examples",
            "annotations": "annotations",
        },
        {
            "handler": _NOT_SERIALIZABLE,
            "preview_kind": _ENGINE_DISPATCH,
            "emits_workspace_diff": _ENGINE_DISPATCH,
            "changed_paths_source": _ENGINE_DISPATCH,
            "result_payload_kind": _ENGINE_DISPATCH,
            "skip_emit_if_background": _ENGINE_DISPATCH,
        },
    ),
)


@pytest.mark.parametrize("carrier", TOOL_SPEC_CARRIERS, ids=lambda item: item.carrier)
def test_5a_every_tool_spec_projection_accounts_for_every_authority_field(
    carrier: ToolSpecCarrier,
) -> None:
    """Each projection's key set is pinned, and each omitted field carries a stated reason.

    The point is not that the subsets are equal -- they should not be, a provider schema is not
    a durable record -- but that each difference is a decision someone wrote down. Six builders
    choosing silently is how a new declarative hint reaches the engine and nothing else.
    """

    produced = carrier.build(_maximal_tool_spec())
    assert frozenset(produced) == frozenset(carrier.key_to_field), {
        "carrier": carrier.carrier,
        "emitted_but_unregistered": sorted(set(produced) - set(carrier.key_to_field)),
        "registered_but_not_emitted": sorted(set(carrier.key_to_field) - set(produced)),
    }
    accounted = {field for field in carrier.key_to_field.values() if field} | set(
        carrier.omissions
    )
    assert accounted == TOOL_SPEC_AUTHORITY, {
        "carrier": carrier.carrier,
        "authority_fields_with_no_verdict": sorted(TOOL_SPEC_AUTHORITY - accounted),
        "verdicts_for_fields_that_do_not_exist": sorted(accounted - TOOL_SPEC_AUTHORITY),
        "hint": "a new ToolSpec field: carry it here or justify the omission",
    }
    for field_name, reason in carrier.omissions.items():
        assert reason.strip(), (carrier.carrier, field_name)
    # Every carried key really carries that field's value (a key set cannot show a mis-wiring).
    spec = _maximal_tool_spec()
    for key, field_name in carrier.key_to_field.items():
        if not field_name:
            continue
        expected = spec.exported_name if field_name == "provider_name" else getattr(
            spec, field_name
        )
        if isinstance(expected, tuple):
            expected = [dict(item) if isinstance(item, dict) else item for item in expected]
        assert produced[key] == expected, {"carrier": carrier.carrier, "key": key}


def test_5b_only_the_record_projections_substitute_a_non_finite_schema() -> None:
    """The request/record split, censused across all five projections at once.

    A request keeps the value so the provider boundary refuses the call as a classified,
    config-recoverable bad request; a record substitutes it because portable JSON cannot carry
    it and a run's durability must not depend on a tool author's schema. Which side each
    projection is on is a property of the projection, so it is pinned per projection.
    """

    spec = dataclasses.replace(
        _maximal_tool_spec(),
        input_schema={"type": "object", "enum": [float("nan")]},
    )
    substituting = set()
    for carrier in TOOL_SPEC_CARRIERS:
        produced = carrier.build(spec)
        schema_key = "parameters" if "parameters" in produced else "input_schema"
        if produced[schema_key]["enum"] == [None]:
            substituting.add(carrier.carrier)
        else:
            assert produced[schema_key]["enum"][0] != produced[schema_key]["enum"][0]  # NaN
    assert substituting == {
        "core/tool_surface.py:_tool_spec_payload",
        # An HTTP egress *describing* a request is a record of one, and this endpoint
        # serializes with allow_nan=False.
        "reference/studio/server.py:_gateway_tool_schema",
        # The third record projection. It used to be portable only because ``RunManifest.to_json``
        # normalizes the whole assembled manifest one frame up -- a property of its caller, not of
        # itself -- and the transcript twin beside it had already needed the substitution locally.
        "core/manifest.py:_tool_spec_payload",
    }, {
        "substituting_projections": sorted(substituting),
        "hint": "a projection changed sides: that is a request/record decision, not a refactor",
    }
    # And the manifest's own output is unchanged, because ``to_json`` normalizes an
    # already-normalized payload to itself: this closed a latent second-caller hazard without
    # rewriting a byte of any manifest a run has already written.
    from monoid_agent_kernel.core.manifest import RunManifest

    manifest_payload = _manifest_tool_spec_payload(spec)
    assert manifest_payload["input_schema"]["enum"] == [None]
    assert normalize_json_ingress(manifest_payload) == manifest_payload
    assert "normalize_json_ingress" in _live_source(RunManifest.to_json)


# reference/llm_gateway/service.py:_parse_tool — the server's reader. Its schema spellings come
# from the shared ``_TOOL_SCHEMA_KEYS`` tuple, so the reader and the server's ingress cannot
# disagree about which keys hold a schema (bc77022).
PARSE_TOOL_READ_KEYS = frozenset(
    {"id", "name", "description", "capability", "side_effect", "input_schema", "parameters"}
)


def test_5c_the_server_reader_reads_exactly_what_the_request_writers_write() -> None:
    parse_tool = _function_node("reference/llm_gateway/service.py", "_parse_tool")
    read = {
        node.args[1].value
        for node in ast.walk(parse_tool)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "parse_str"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    }
    # The two schema spellings arrive through the shared tuple, not as literals here.
    assert any(
        isinstance(node, ast.Name) and node.id == "_TOOL_SCHEMA_KEYS"
        for node in ast.walk(parse_tool)
    ), "the reader stopped using the shared schema-key tuple"
    read |= set(_TOOL_SCHEMA_KEYS)
    assert read == PARSE_TOOL_READ_KEYS, {
        "newly_read": sorted(read - PARSE_TOOL_READ_KEYS),
        "no_longer_read": sorted(PARSE_TOOL_READ_KEYS - read),
    }
    # Every key the gateway request writer emits is read back; the reverse holds through the
    # ``parameters`` alias, which exists for tool entries written by a non-kernel client.
    written = frozenset(gateway_client._gateway_tool_schema(_maximal_tool_spec()))
    assert written <= read
    assert read - written == {"parameters"}


def test_5c_the_server_reader_reconstructs_only_the_fields_the_wire_carries() -> None:
    """Behavioral twin: what a ToolSpec looks like after a round trip through the gateway."""

    spec = _maximal_tool_spec()
    restored = _parse_tool(gateway_client._gateway_tool_schema(spec))
    survived = {
        field.name
        for field in dataclasses.fields(ToolSpec)
        if field.name != "handler" and getattr(restored, field.name) == getattr(spec, field.name)
    }
    assert survived == {
        "id",
        "description",
        "input_schema",
        "capability",
        "side_effect",
        "provider_name",
    }, {
        "survived_the_hop": sorted(survived),
        "hint": "the request wire carries six fields; the rest are engine-local by design",
    }
    # The server's stand-in handler refuses execution rather than pretending to run the tool.
    assert restored.handler is not spec.handler


# --------------------------------------------------------------------------------------
# Family 6 — the checkpoint validator's field coverage
# --------------------------------------------------------------------------------------

# core/checkpoint.py:_validate_checkpoint_payload is driven by six hand-maintained field-name
# frozensets plus a handful of inline branches. It is the durable-recovery boundary and it fails
# OPEN: a field nobody listed is simply never type-checked, so a corrupt or hostile payload
# reaches ``RunCheckpoint(**payload)`` with an arbitrary Python value in it. The names below are
# the inline branches, listed here so the diff below is total.
CHECKPOINT_INLINE_VALIDATED = frozenset(
    {
        "run_id",
        "pending_user_input",
        "previous_runtime_config",
        "workspace_base",
        "last_suspension",
        "tool_call_counts",
        "total_usage",
        "revoked_before",
        "remaining_duration_s",
        "queued_messages",
        "active_input",
        "applied_input_receipts",
    }
)
# The documented exclusion: the codec owns the version envelope (core/durable.py), so the
# payload validator never sees a checkpoint whose schema_version it did not accept.
CHECKPOINT_UNVALIDATED = frozenset({"schema_version"})

# And each bucket's MEMBERSHIP, not only the union of them. Moving ``error_code`` from
# ``_CHECKPOINT_STRING_FIELDS`` to ``_CHECKPOINT_OPTIONAL_STRING_FIELDS`` — which changes it from
# "must be a string" to "may be null" at the recovery boundary — left the union, the disjointness
# and the count all unchanged and passed the whole suite. A validation bucket is a rule, and
# which rule a field is under is the fact.
CHECKPOINT_VALIDATION_BUCKETS: dict[str, frozenset[str]] = {
    "_CHECKPOINT_STRING_FIELDS": frozenset(
        {"status", "error", "error_code", "provider_error_code", "final_text"}
    ),
    "_CHECKPOINT_OPTIONAL_STRING_FIELDS": frozenset({"previous_turn_handle"}),
    "_CHECKPOINT_NONNEGATIVE_INT_FIELDS": frozenset(
        {
            "seq",
            "provider_http_status",
            "total_tool_calls",
            "output_retries",
            "session_step",
            "submit_local_step",
        }
    ),
    "_CHECKPOINT_BOOL_FIELDS": frozenset({"terminal", "revoked_all", "cancellation_requested"}),
    "_CHECKPOINT_LIST_OF_DICT_FIELDS": frozenset(
        {
            "pending_observations",
            "messages",
            "hosted_tasks",
            "workspace_delta",
            "capability_leases",
            "pending_capability_replays",
            "pending_tool_approval_replays",
            "outbox_requests",
        }
    ),
    "_CHECKPOINT_LIST_OF_STRING_FIELDS": frozenset(
        {
            "pending_binding_loads",
            "reentry_queue",
            "delivered_reentry_jobs",
            "revoked_lease_ids",
            "revoked_capabilities",
            "inbox_seen_ids",
            "applied_input_ids",
        }
    ),
}


def _live_checkpoint_buckets() -> dict[str, frozenset[str]]:
    from monoid_agent_kernel.core import checkpoint as checkpoint_module

    return {
        name: value
        for name, value in vars(checkpoint_module).items()
        if name.startswith("_CHECKPOINT_") and isinstance(value, frozenset)
    }


def test_6a_every_checkpoint_field_is_validated_by_exactly_one_mechanism() -> None:
    from monoid_agent_kernel.core.checkpoint import RunCheckpoint

    frozensets = _live_checkpoint_buckets()
    assert frozensets == CHECKPOINT_VALIDATION_BUCKETS, {
        "buckets_that_gained_a_field": {
            name: sorted(value - CHECKPOINT_VALIDATION_BUCKETS.get(name, frozenset()))
            for name, value in frozensets.items()
            if value - CHECKPOINT_VALIDATION_BUCKETS.get(name, frozenset())
        },
        "buckets_that_lost_a_field": {
            name: sorted(value - frozensets.get(name, frozenset()))
            for name, value in CHECKPOINT_VALIDATION_BUCKETS.items()
            if value - frozensets.get(name, frozenset())
        },
        "new_buckets": sorted(set(frozensets) - set(CHECKPOINT_VALIDATION_BUCKETS)),
        "removed_buckets": sorted(set(CHECKPOINT_VALIDATION_BUCKETS) - set(frozensets)),
        "hint": "a field that MOVED between buckets is a changed validation rule at the "
        "recovery boundary, and it leaves the union and the count untouched",
    }
    listed: set[str] = set()
    for name, value in frozensets.items():
        overlap = listed & set(value)
        assert overlap == set(), {"field_in_two_buckets": sorted(overlap), "bucket": name}
        listed |= set(value)

    authority = set(RunCheckpoint.__dataclass_fields__)
    covered = listed | CHECKPOINT_INLINE_VALIDATED
    assert covered - authority == set(), {
        "validated_but_not_a_field": sorted(covered - authority),
        "hint": "a bucket names a field the dataclass dropped: the check is dead",
    }
    assert authority - covered == CHECKPOINT_UNVALIDATED, {
        "unvalidated_fields": sorted(authority - covered),
        "hint": "a new RunCheckpoint field escapes validation entirely — this validator fails "
        "open, so an unlisted field is never type-checked at the recovery boundary",
    }


def test_6a_every_bucket_is_actually_consumed_by_the_validator() -> None:
    """A bucket nobody loops over is a validation rule that does not run.

    Membership and coverage are both computed from the bucket *contents*, so a new
    ``_CHECKPOINT_FLOAT_FIELDS`` naming a real field satisfies the census above and the field
    reads as validated — while ``_validate_checkpoint_payload`` never mentions it and the field
    reaches ``RunCheckpoint(**payload)`` unchecked. The validator is asked whether it reads each
    bucket by name.
    """

    validator = _function_node("core/checkpoint.py", "_validate_checkpoint_payload")
    consumed = {node.id for node in ast.walk(validator) if isinstance(node, ast.Name)}
    buckets = set(_live_checkpoint_buckets())
    dangling = buckets - consumed
    assert dangling == set(), {
        "buckets_the_validator_never_reads": sorted(dangling),
        "hint": "a declared bucket that no loop consumes validates nothing — this validator "
        "fails open, so its fields are silently unchecked",
    }
    # ...and the validator does not read a bucket that no longer exists.
    assert {name for name in consumed if name.startswith("_CHECKPOINT_")} == buckets


def test_6a_the_inline_branches_named_here_are_the_branches_that_exist() -> None:
    """The inline half is a hand list too, so it is diffed against the source it describes."""

    validator = _function_node("core/checkpoint.py", "_validate_checkpoint_payload")
    inline = {
        node.left.value
        for node in ast.walk(validator)
        if isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.In)
        and isinstance(node.left, ast.Constant)
        and isinstance(node.left.value, str)
    }
    inline |= {
        element.value
        for node in ast.walk(validator)
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple)
        for element in node.iter.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    }
    # ``run_id`` is read positionally rather than membership-tested, so it is added by hand.
    inline.add("run_id")
    assert inline == CHECKPOINT_INLINE_VALIDATED, {
        "newly_inline": sorted(inline - CHECKPOINT_INLINE_VALIDATED),
        "no_longer_inline": sorted(CHECKPOINT_INLINE_VALIDATED - inline),
    }


def test_6b_the_park_payload_is_validated_as_an_object_and_nothing_more() -> None:
    """The registered gap, pinned: ``last_suspension`` has no schema of its own.

    Every field family 1 pins on the writer/reader pair is unpinned on the durable artifact —
    the validator accepts any object at all, so a park payload with a string ``retryable``
    reaches the reader, which is where it becomes a ValueError instead of a clear rejection.
    """

    from monoid_agent_kernel.core.checkpoint import _validate_checkpoint_payload

    _validate_checkpoint_payload({"run_id": "run_1", "last_suspension": {"anything": [1, 2, 3]}})
    _validate_checkpoint_payload({"run_id": "run_1", "last_suspension": None})
    with pytest.raises(ValueError):
        _validate_checkpoint_payload({"run_id": "run_1", "last_suspension": "not an object"})


# --------------------------------------------------------------------------------------
# Family 7 — the success envelope (the main wire)
# --------------------------------------------------------------------------------------
#
# Families 2–4 census the failure wire and the proof wire. The wire that carries every ordinary
# turn had no census at all: two hand-built server writers (a JSON body and an SSE terminal
# frame) read back by two hand-written client parsers, with one fact renamed across the hop.

GATEWAY_SUCCESS_BODY_KEYS = frozenset(
    {
        "protocol",
        "turn_handle",
        "final_text",
        "tool_calls",
        "usage",
        "stop_reason",
        "provider_retried",
        "generation_applied",
        "schema_applied",
    }
)
GATEWAY_TERMINAL_FRAME_KEYS = frozenset(
    {
        "type",
        "turn_handle",
        "usage",
        "stop_reason",
        "provider_retried",
        "generation_applied",
        "schema_applied",
    }
)
# The frame is the body minus what the deltas already delivered, minus the protocol tag the
# stream states once in its content type. Declared, so a *new* omission is not read as one of
# these.
TERMINAL_FRAME_DELIBERATE_OMISSIONS = frozenset({"protocol", "final_text", "tool_calls"})


class _EverythingAdapter:
    """A stub upstream that answers one maximal turn and declares native support for both echoes."""

    generation_support = "native"
    structured_output_support = "native"

    def next_turn(self, request: Any) -> ModelTurn:
        del request
        return ModelTurn(
            response_id="provider_response_secret",
            final_text="answered",
            tool_calls=(),
            usage={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
            raw={"anything": True},
            reasoning=({"type": "reasoning", "id": "rs_1"},),
            stop_reason="stop",
            provider_retried=True,
        )


class _PlainAdapter:
    """The MINIMAL upstream: no retry to report, and no native-support declaration to echo.

    Its counterpart to ``_EverythingAdapter`` is the point. A census driven only by the maximal
    probe pins the *union* of the wire keys, so a writer that started omitting a key whenever it
    holds its default value — ``provider_retried`` when the call was not retried — kept passing:
    the maximal probe sets it True and the key stays. Required-vs-conditional is a property of
    the wire, and only a minimal request can state it.
    """

    def next_turn(self, request: Any) -> ModelTurn:
        del request
        return ModelTurn(
            response_id="provider_response_secret",
            final_text="answered",
            tool_calls=(),
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            stop_reason="stop",
            provider_retried=False,
        )


def _gateway_and_token(adapter: Any = None) -> tuple[LlmGatewayBackend, str]:
    manager = TokenManager.from_secret("c" * 32)
    upstream = _EverythingAdapter() if adapter is None else adapter
    gateway = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, _config: upstream,
    )
    token = manager.issue(
        kind="llm_gateway",
        audience="csp.llm-gateway",
        run_id="run_1",
        tenant_id="tenant_a",
        user_id="user_a",
        ttl_s=600,
        metadata={"agent_config_hash": "census"},
    )
    return gateway, token


def _maximal_turn_payload() -> dict[str, Any]:
    """A request that exercises every optional block, so every echo is emitted."""

    return {
        "protocol": LLM_TURN_PROTOCOL_VERSION,
        "model": "gateway-model",
        "system_prompt": "sys",
        "instruction": "do the thing",
        "generation": {"temperature": 0.5},
        "output_schema": {"type": "object"},
        "tools": [gateway_client._gateway_tool_schema(_maximal_tool_spec())],
    }


def _minimal_turn_payload() -> dict[str, Any]:
    """A request that uses no optional feature at all, so no echo can be emitted."""

    return {
        "protocol": LLM_TURN_PROTOCOL_VERSION,
        "model": "gateway-model",
        "system_prompt": "sys",
        "instruction": "do the thing",
        "tools": [],
    }


# What the two writers emit for a request that configured nothing, answered by an upstream with
# nothing to report. This is the REQUIRED half of each key set: the maximal sets above are the
# union, and the difference between the two is precisely the conditional keys.
GATEWAY_MINIMAL_BODY_KEYS = frozenset(
    {
        "protocol",
        "turn_handle",
        "final_text",
        "tool_calls",
        "usage",
        "stop_reason",
        "provider_retried",
    }
)
GATEWAY_MINIMAL_FRAME_KEYS = frozenset(
    {"type", "turn_handle", "usage", "stop_reason", "provider_retried"}
)
# The keys that appear only when the request asked for the feature (registered by-design on
# ``_applied_echoes``: traffic that configures neither keeps its exact pre-W5 wire shape).
GATEWAY_CONDITIONAL_WIRE_KEYS = frozenset(APPLIED_ECHO_KEYS)


def test_7a_the_minimal_request_pins_which_body_keys_are_conditional() -> None:
    """The minimal twin of the maximal body census — the half that sees an omission.

    ``provider_retried`` is the case this exists for: the maximal probe reports it True, so a
    writer that emitted it only when true was indistinguishable from one that always emits it,
    and the client's ``False`` default would have silently absorbed the change. Here the fact is
    False and the key must still be on the wire.
    """

    gateway, token = _gateway_and_token(_PlainAdapter())
    body = gateway.handle_turn(token, _minimal_turn_payload())
    assert frozenset(body) == GATEWAY_MINIMAL_BODY_KEYS, {
        "missing": sorted(GATEWAY_MINIMAL_BODY_KEYS - set(body)),
        "extra": sorted(set(body) - GATEWAY_MINIMAL_BODY_KEYS),
        "hint": "a key that vanished when its value was the default: that is an omit-when-empty "
        "contract, and it has to be declared like the two on _applied_echoes",
    }
    assert body["provider_retried"] is False
    assert GATEWAY_SUCCESS_BODY_KEYS - GATEWAY_MINIMAL_BODY_KEYS == GATEWAY_CONDITIONAL_WIRE_KEYS
    assert GATEWAY_MINIMAL_BODY_KEYS <= GATEWAY_SUCCESS_BODY_KEYS


def test_7a_the_minimal_request_pins_which_terminal_frame_keys_are_conditional() -> None:
    """The streamed twin of the probe above, because the two writers are separate code."""

    gateway, token = _gateway_and_token(_PlainAdapter())
    frames = list(gateway.handle_turn_stream(token, _minimal_turn_payload()))
    terminal = frames[-1]
    assert terminal["type"] == "turn_complete"
    assert frozenset(terminal) == GATEWAY_MINIMAL_FRAME_KEYS, {
        "missing": sorted(GATEWAY_MINIMAL_FRAME_KEYS - set(terminal)),
        "extra": sorted(set(terminal) - GATEWAY_MINIMAL_FRAME_KEYS),
        "hint": "the frame writer and the body writer must be conditional about the same keys",
    }
    assert terminal["provider_retried"] is False
    assert (
        GATEWAY_TERMINAL_FRAME_KEYS - GATEWAY_MINIMAL_FRAME_KEYS == GATEWAY_CONDITIONAL_WIRE_KEYS
    )
    # Both transports are conditional about exactly the same keys, which is the property that
    # makes the echo protocol a protocol rather than two behaviours.
    assert (GATEWAY_SUCCESS_BODY_KEYS - GATEWAY_MINIMAL_BODY_KEYS) == (
        GATEWAY_TERMINAL_FRAME_KEYS - GATEWAY_MINIMAL_FRAME_KEYS
    )


def test_7a_the_sync_success_body_writes_exactly_the_censused_key_set() -> None:
    gateway, token = _gateway_and_token()
    body = gateway.handle_turn(token, _maximal_turn_payload())
    assert frozenset(body) == GATEWAY_SUCCESS_BODY_KEYS, {
        "missing": sorted(GATEWAY_SUCCESS_BODY_KEYS - set(body)),
        "extra": sorted(set(body) - GATEWAY_SUCCESS_BODY_KEYS),
    }
    # The opaque handle is the gateway's own; the provider's response id never leaves.
    assert body["turn_handle"].startswith("turn_")
    assert "provider_response_secret" not in json.dumps(body)
    assert body["provider_retried"] is True


def test_7a_the_terminal_frame_is_the_body_minus_what_the_deltas_delivered() -> None:
    gateway, token = _gateway_and_token()
    frames = list(gateway.handle_turn_stream(token, _maximal_turn_payload()))
    terminal = frames[-1]
    assert terminal["type"] == "turn_complete"
    assert frozenset(terminal) == GATEWAY_TERMINAL_FRAME_KEYS, {
        "missing": sorted(GATEWAY_TERMINAL_FRAME_KEYS - set(terminal)),
        "extra": sorted(set(terminal) - GATEWAY_TERMINAL_FRAME_KEYS),
    }
    dropped = GATEWAY_SUCCESS_BODY_KEYS - GATEWAY_TERMINAL_FRAME_KEYS
    assert dropped == TERMINAL_FRAME_DELIBERATE_OMISSIONS, {
        "dropped_on_the_streamed_transport": sorted(dropped),
        "hint": "a NEW omission here is a fact one transport carries and the other does not",
    }
    # The dropped facts really were delivered as deltas, so the omission is a re-encoding.
    assert any(frame.get("type") == "text_delta" for frame in frames)
    # Both transports answer the same echoes, built by the same function.
    assert {key: terminal[key] for key in APPLIED_ECHO_KEYS} == {
        "generation_applied": {"temperature": 0.5},
        "schema_applied": True,
    }


def test_7b_reasoning_artifacts_do_not_cross_the_gateway_hop_on_either_transport() -> None:
    """Registered burn-down, and pinned as *symmetric* so it reads as a gap, not a drift.

    ``ModelTurn.reasoning`` is what the provider-native reasoning round-trip (DX-13a) replays,
    and the gateway's two writers have no slot for it — so a run routed through the gateway
    replays nothing, on both transports equally. Symmetry is the point: this is a feature the
    hop never carried, not a twin that fell out of step, and closing it means adding a wire key
    to both writers and both readers at once.
    """

    gateway, token = _gateway_and_token()
    upstream = _EverythingAdapter().next_turn(None)
    assert upstream.reasoning, "the stub must actually produce reasoning artifacts"

    body = gateway.handle_turn(token, _maximal_turn_payload())
    frames = list(gateway.handle_turn_stream(token, _maximal_turn_payload()))
    assert "reasoning" not in body
    assert all("reasoning" not in frame for frame in frames)
    # And the client cannot reconstruct one: neither reader names the key.
    for reader in GATEWAY_READER_WIRE_KEYS.values():
        assert "reasoning" not in reader


# What each client parser reads on the SUCCESS path, pinned separately from the error path so
# the two halves of one function cannot cover for each other.
SUCCESS_READS_R1 = frozenset(
    {
        "error",
        "provider_retried",
        "retryable",
        "usage",
        "tool_calls",
        "arguments",
        "id",
        "call_id",
        "name",
        "stop_reason",
        "response_id",
        "turn_handle",
        "final_text",
    }
)
TURN_COMPLETE_READS_R2 = frozenset(
    {
        "turn_handle",
        "response_id",
        "usage",
        "stop_reason",
        "generation_applied",
        "schema_applied",
    }
)
# Reads with no writer, each one accounted for.
SUCCESS_DANGLING_READS: dict[str, str] = {
    "error": "the envelope discriminator: one parser serves both shapes and tests this key "
    "before branching, so it is read on the success path and written only on the error one",
    "response_id": "alias: the gateway writes turn_handle, and the same parser also serves a "
    "non-gateway body that spells it response_id — declared in the reader's key order",
    "retryable": "defensive: the success path type-checks the key in case an error body arrives "
    "without its error field, so a malformed envelope is refused rather than half-read",
    "id": "alias inside a tool_call entry: the gateway writes call_id, the parser accepts either",
}


def _success_branch_reads() -> frozenset[str]:
    parser = _function_node("providers/gateway.py", "_parse_gateway_response")
    error_branch = next(
        node
        for node in parser.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Constant)
        and node.test.left.value == "error"
    )
    outside = ast.Module(
        body=[statement for statement in parser.body if statement is not error_branch],
        type_ignores=[],
    )
    return _literal_wire_keys(outside)


def _turn_complete_branch_reads() -> frozenset[str]:
    reader = _function_node("providers/gateway.py", "_chunk_from_event")
    branch = next(
        node
        for node in ast.walk(reader)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and node.test.comparators
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "turn_complete"
    )
    return _literal_wire_keys(branch)


def test_7c_the_client_reads_the_success_wire_the_server_writes() -> None:
    read = _success_branch_reads()
    assert read == SUCCESS_READS_R1, {
        "newly_read": sorted(read - SUCCESS_READS_R1),
        "no_longer_read": sorted(SUCCESS_READS_R1 - read),
    }
    # Nested tool-call keys are written per entry, not at the top level of the body.
    nested = {"arguments", "id", "call_id", "name"}
    top_level_read = read - nested
    unwritten = top_level_read - GATEWAY_SUCCESS_BODY_KEYS
    assert unwritten == set(SUCCESS_DANGLING_READS) - nested, {
        "read_but_never_written": sorted(unwritten),
        "accounted_for": sorted(SUCCESS_DANGLING_READS),
    }
    unread = GATEWAY_SUCCESS_BODY_KEYS - read - set(APPLIED_ECHO_KEYS)
    assert unread == {"protocol"}, {
        "written_but_never_read": sorted(unread),
        "hint": "the client ignores the protocol tag: version negotiation is the server's job",
    }
    for reason in SUCCESS_DANGLING_READS.values():
        assert reason.strip()


def test_7c_the_client_reads_the_terminal_frame_the_server_writes() -> None:
    read = _turn_complete_branch_reads()
    assert read == TURN_COMPLETE_READS_R2, {
        "newly_read": sorted(read - TURN_COMPLETE_READS_R2),
        "no_longer_read": sorted(TURN_COMPLETE_READS_R2 - read),
    }
    # ``type`` and ``provider_retried`` are read before the branch, for every frame type.
    pre_branch = _literal_wire_keys(_function_node("providers/gateway.py", "_chunk_from_event"))
    assert {"type", "provider_retried"} <= pre_branch
    unwritten = read - GATEWAY_TERMINAL_FRAME_KEYS
    assert unwritten == {"response_id"}, {
        "read_but_never_written": sorted(unwritten),
        "hint": "the same turn_handle/response_id alias the sync parser carries",
    }
    assert GATEWAY_TERMINAL_FRAME_KEYS - read - {"type", "provider_retried"} == set()


def test_7d_one_fact_two_spellings_across_the_success_hop() -> None:
    """The alias is declared here rather than inferred, like the three in the error family."""

    gateway, token = _gateway_and_token()
    body = gateway.handle_turn(token, _maximal_turn_payload())
    turn = gateway_client._parse_gateway_response(dict(body))
    # Written as ``turn_handle``, reconstructed as ``ModelTurn.response_id``.
    assert turn.response_id == body["turn_handle"]
    assert "response_id" not in body
    frames = list(gateway.handle_turn_stream(token, _maximal_turn_payload()))
    chunk = gateway_client._chunk_from_event(dict(frames[-1]))
    assert chunk.response_id == frames[-1]["turn_handle"]


# --------------------------------------------------------------------------------------
# Registered-gap pins — a registry entry with no assertion is prose
# --------------------------------------------------------------------------------------
#
# The registry's contract is "fixing a gap breaks this suite". That only holds for an entry with
# an assertion behind it, and five round-1 entries had none: the drivers and consumers that were
# *designed* for a fact and ignore it, the closed stream_closed schema, the unread ``turn.failed``
# and the ``turn.interrupted`` vocabulary collision were described and never pinned. Each one
# below flips to a failure the moment its gap is closed, which is the whole mechanism.


def _handled_event_types(function: _FunctionNode, subject: str) -> frozenset[str]:
    """Event types ``function`` dispatches on, read off ``<subject> == "x"`` / ``in {...}``."""

    handled: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if ast.unparse(node.left) != subject:
            continue
        comparator = node.comparators[0]
        if isinstance(node.ops[0], ast.Eq) and isinstance(comparator, ast.Constant):
            handled.add(comparator.value)
        elif isinstance(node.ops[0], ast.In) and isinstance(
            comparator, (ast.Set, ast.Tuple, ast.List)
        ):
            handled |= {
                element.value
                for element in comparator.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
    return frozenset(handled)


def _names_in(function: ast.AST) -> frozenset[str]:
    """Every identifier and string constant occurring in ``function``'s code."""

    constants, identifiers = _code_occurrences(function)
    return constants | identifiers


def _getattr_names(function: ast.AST) -> frozenset[str]:
    """The attribute names ``function`` reads through ``getattr(obj, "name", ...)``."""

    return frozenset(
        node.args[1].value
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    )


def _attribute_reads_on(function: ast.AST, name: str) -> frozenset[str]:
    """The attributes ``function`` reads straight off the local ``name`` — ``exc.retryable``.

    The twin of :func:`_getattr_names`, and the reason a pin needs both. A census that collects
    only the ``getattr(exc, "x", default)`` spelling pins the spelling and not the read: the
    natural way to close a "this consumer never names the fact" gap is a plain
    ``exc.config_recoverable``, which would leave a getattr-only pin green while the gap it
    registers no longer exists.
    """

    return frozenset(
        node.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == name
    )


def test_the_driver_records_config_recoverable_without_branching_on_it() -> None:
    """Was registered: the driver the classification was added for never named it.

    Closed by *recording* rather than by parking. The decision the driver makes is
    retry-vs-give-up, and a config-fixable failure is not fixable by the driver — it is fixable by
    whoever reads the surface the driver feeds. So the fact is promoted onto the record (which
    ``status()``/``result()`` serve) and control flow is unchanged, and both halves are pinned:
    the read exists, and ``retryable`` is still the only thing the retry branch tests.
    """

    driver = _function_node("reference/backend/session_drive.py", "drive_open_session")
    names = _names_in(driver)
    assert "retryable" in names, "the driver stopped branching on retryable — recheck this pin"
    assert "config_recoverable" in names, {
        "hint": "the driver stopped reading the classification it promotes onto the record",
    }
    # The read is off the suspension and the write is onto the record — not the reverse, and not
    # a second control-flow input smuggled in beside the first.
    assert "config_recoverable" in _attribute_reads_on(driver, "suspension")
    assert "config_recoverable" in _attribute_reads_on(driver, "record")
    branched_on = {
        ast.unparse(node.test)
        for node in ast.walk(driver)
        if isinstance(node, ast.If)
    }
    assert not [test for test in branched_on if "config_recoverable" in test], {
        "branches_testing_the_classification": sorted(
            test for test in branched_on if "config_recoverable" in test
        ),
        "hint": "D2 was 'expose it, do not park on it' — a branch here is a control-flow change",
    }
    assert "suspension.retryable" in branched_on


# core/model_io.py:ModelCallReceipt.with_error — every fact it lifts off the exception, by name.
RECEIPT_ERROR_FACTS = frozenset(
    {
        "error_code",
        "provider_error_code",
        "retryable",
        # The sixth: waiting-may-help and configuration-will-help are two facts, and the receipt
        # used to record only the first.
        "config_recoverable",
        "http_status",
        "provider_retried",
        # Read too, and for the same reason the wire carries it: a failure after a billed answer.
        "provider_usage",
    }
)


def test_the_call_receipt_reads_every_fact_the_exception_transports() -> None:
    """Was registered as "five facts and not the sixth"; the sixth is read now.

    Both spellings of a read are collected. ``with_error`` takes ``BaseException``, so it reaches
    every fact through ``getattr(exc, ..., default)`` — but a plain ``exc.config_recoverable`` on
    a narrowed branch is the other natural spelling, and a pin that censused the getattr form
    alone would have stayed green through the fix that closed the gap.
    """

    with_error = _function_node("core/model_io.py", "with_error", within="ModelCallReceipt")
    positional = [argument.arg for argument in with_error.args.args if argument.arg != "self"]
    assert positional == ["exc"], {"with_error_parameters": positional}
    read = _getattr_names(with_error) | _attribute_reads_on(with_error, "exc")
    assert read == RECEIPT_ERROR_FACTS, {
        "newly_read": sorted(read - RECEIPT_ERROR_FACTS),
        "no_longer_read": sorted(RECEIPT_ERROR_FACTS - read),
        "hint": "the receipt's read set is the record's whole vocabulary",
    }
    # ...and every fact it reads reaches the durable record, rather than being read and dropped.
    from monoid_agent_kernel.core.model_io import ModelCallReceipt

    written = set(ModelCallReceipt().to_json())
    assert read - written == {"provider_usage"}, {
        "read_but_never_written": sorted(read - written),
        "hint": "provider_usage is folded into ``usage``; anything else here is a dropped fact",
    }
    # Every fact it does read is a real attribute of the exception it reads them from.
    maximal = _maximal_adapter_error()
    assert read <= set(vars(maximal)) | {"provider_usage"}


# core/schemas.py, MODEL_CONTENT_RECORD_SCHEMA's stream_closed branch. additionalProperties is
# False, so this is the total domain of the record and adding a fact to it is a schema change.
STREAM_CLOSED_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "run_id",
        "stream_id",
        "status",
        "final_text",
        "usage",
        "error_code",
        "retryable",
        # Joined in the burn-down, schema and writer in one change because the cap below is
        # closed: the live lane now classifies with the same two words the park carries.
        "config_recoverable",
        "finished_at",
    }
)


def _stream_closed_branch() -> dict[str, Any]:
    from monoid_agent_kernel.core.schemas import MODEL_CONTENT_RECORD_SCHEMA

    for variant in MODEL_CONTENT_RECORD_SCHEMA["oneOf"]:
        if variant["properties"]["kind"].get("const") == "stream_closed":
            return variant
    raise AssertionError("MODEL_CONTENT_RECORD_SCHEMA has no stream_closed branch")


def test_the_stream_closed_record_classifies_with_the_whole_vocabulary() -> None:
    """Was registered: the live stream lane carried retryable and no config_recoverable.

    Pinned on all three carriers of one record, because ``additionalProperties`` is False: the
    outcome type, the writer's literal, and the schema branch have to move together or the record
    fails its own validation.
    """

    from monoid_agent_kernel.core.model_stream import ModelStreamOutcome

    branch = _stream_closed_branch()
    declared = frozenset(branch["properties"])
    assert declared == STREAM_CLOSED_RECORD_KEYS, {
        "missing": sorted(STREAM_CLOSED_RECORD_KEYS - declared),
        "extra": sorted(declared - STREAM_CLOSED_RECORD_KEYS),
    }
    assert branch["additionalProperties"] is False, {
        "hint": "the closed cap is why adding a fact here is a schema change, not a patch",
    }
    assert {"retryable", "config_recoverable"} <= declared
    # The writer's literal: what ``close()`` actually puts on the record.
    written = _literal_dict_keys_where("core/model_content.py", "kind", "stream_closed")
    assert len(written) == 1, {"stream_closed_writers": len(written)}
    assert written[0] == STREAM_CLOSED_RECORD_KEYS, {
        "written": sorted(written[0]),
        "hint": "the writer and the closed schema must name the same keys",
    }
    # ...and the outcome the writer reads them off carries both classifications.
    outcome_fields = {field.name for field in dataclasses.fields(ModelStreamOutcome)}
    assert {"retryable", "config_recoverable"} <= outcome_fields
    # Optional on the record, so a sidecar written before the key existed still validates.
    assert "config_recoverable" not in branch["required"]


# The three consumers of one event stream, and the event types each one dispatches on. Pinned
# per consumer, because the defect here was not a missing branch anywhere in particular — it was
# that no two of them handled the same set and nothing said so.
#
# Converged in the burn-down: the two parks (``run.waiting``, ``run.awaiting_input``) and the
# classification (``turn.failed``) are now handled by all three, so the sets differ only by the
# events a consumer's *surface* has no slot for. Those differences are named in
# ``EVENT_CONSUMER_SINK_ONLY`` / ``EVENT_CONSUMER_OFFLINE_ONLY`` below and asserted as the whole
# remaining diff, so a fourth divergence cannot appear quietly.
EVENT_CONSUMERS: dict[str, tuple[str, str | None, str]] = {
    # name -> (module, enclosing class, the expression it switches on)
    "core/projections.py:_apply_event_projection": (
        "core/projections.py",
        None,
        "event_type",
    ),
    "reference/backend/run_state.py:record_event": (
        "reference/backend/run_state.py",
        "RunStateMutationService",
        "event.type",
    ),
    "recorder.py:StatusJsonSink.emit": ("recorder.py", "StatusJsonSink", "event.type"),
}
# What every status consumer must handle: the lifecycle edges, both parks, and the recoverable
# turn failure whose classification is the only thing a parked run has to show.
EVENT_CONSUMER_CORE = frozenset(
    {
        "run.started",
        "run.finished",
        "run.failed",
        "run.waiting",
        "run.awaiting_input",
        "run.resumed",
        "turn.failed",
        "model.turn.started",
    }
)
# The progress detail two of the three project. The backend record has no slot for any of it —
# it carries lifecycle + classification and the run dir carries the rest — so this is a surface
# difference, not a hole.
EVENT_CONSUMER_PROGRESS = frozenset(
    {
        "agent.config.updated",
        "tool.call.started",
        "tool.call.finished",
        "tool.call.failed",
        "workspace.proposal.updated",
    }
)
EVENT_CONSUMER_HANDLED: dict[str, frozenset[str]] = {
    # The offline projection reads the *committed log*, so it can also fold the durable proposal
    # lifecycle that the two live sinks never see (they fire before the artifacts are written).
    "core/projections.py:_apply_event_projection": EVENT_CONSUMER_CORE
    | EVENT_CONSUMER_PROGRESS
    | frozenset(
        {
            "proposal.package.exported",
            "proposal.approved",
            "proposal.rejected",
            "proposal.applied",
            "proposal.conflict",
        }
    ),
    # The narrowest surface: an in-memory lifecycle record. No progress detail by design.
    "reference/backend/run_state.py:record_event": EVENT_CONSUMER_CORE,
    # The fan-out sink. Its two extras are the live-only projections nothing durable carries.
    # (``job.*`` is matched by prefix rather than equality, so it is outside this census by
    # construction.)
    "recorder.py:StatusJsonSink.emit": EVENT_CONSUMER_CORE
    | EVENT_CONSUMER_PROGRESS
    | frozenset({"plan.updated", "metrics.updated"}),
}


@pytest.mark.parametrize("consumer", sorted(EVENT_CONSUMERS))
def test_each_status_consumer_handles_the_whole_core_of_one_event_stream(
    consumer: str,
) -> None:
    """Was registered (three entries): two projections, each blind to the park the other saw."""

    relative_path, class_name, subject = EVENT_CONSUMERS[consumer]
    name = consumer.split(":", 1)[1].split(".")[-1]
    handled = _handled_event_types(
        _function_node(relative_path, name, within=class_name), subject
    )
    expected = EVENT_CONSUMER_HANDLED[consumer]
    assert handled == expected, {
        "consumer": consumer,
        "newly_handled": sorted(handled - expected),
        "no_longer_handled": sorted(expected - handled),
        "hint": "a consumer that gained or lost a branch changes what an operator can read",
    }
    # The convergence itself, restated per consumer so no single one can drift back out of it.
    assert EVENT_CONSUMER_CORE <= handled, {
        "consumer": consumer,
        "core_events_not_handled": sorted(EVENT_CONSUMER_CORE - handled),
        "hint": "the three readers of one stream agreed on this core; this one no longer does",
    }


def test_the_three_run_status_projections_differ_only_by_what_their_surface_carries() -> None:
    """The diff itself, so the convergence is the assertion rather than an inference."""

    offline = EVENT_CONSUMER_HANDLED["core/projections.py:_apply_event_projection"]
    backend = EVENT_CONSUMER_HANDLED["reference/backend/run_state.py:record_event"]
    control = EVENT_CONSUMER_HANDLED["recorder.py:StatusJsonSink.emit"]
    # Both parks and the classification, on all three.
    for handled in (offline, backend, control):
        assert {"run.waiting", "run.awaiting_input", "turn.failed"} <= handled
    # The whole remaining difference, named. Nothing is handled by one reader alone except the
    # durable proposal lifecycle (only the log reader sees it) and the two live-only projections
    # (only the fan-out sink can build them).
    assert offline - control == {
        "proposal.package.exported",
        "proposal.approved",
        "proposal.rejected",
        "proposal.applied",
        "proposal.conflict",
    }
    assert control - offline == {"plan.updated", "metrics.updated"}
    # The backend record is a strict subset of both: it carries lifecycle + classification only.
    assert backend < offline and backend < control
    assert backend == EVENT_CONSUMER_CORE


def test_gap_turn_interrupted_speaks_a_cause_vocabulary_the_park_type_cannot() -> None:
    """Registered: one word, two vocabularies, on one event.

    ``data.reason`` on ``turn.interrupted`` names the CAUSE (``user_stop``); ``Suspension.reason``
    names the PARK (``interrupted``). Pinned as the collision it is: the emit's literal and the
    proof that its value is not a member of the park vocabulary it shares a field name with.
    """

    emitted = _emit_data_keys("loop.py", "turn.interrupted")
    assert emitted == {"reason"}, {"emitted": sorted(emitted)}
    values = _literal_dict_keys_where("loop.py", "reason", "user_stop")
    assert len(values) == 1, {"turn_interrupted_reason_literals": len(values)}
    assert values[0] == {"reason"}
    assert "user_stop" not in _SUSPENSION_REASONS, {
        "hint": "the two vocabularies merged: drop the registry entry",
    }
    assert "interrupted" in _SUSPENSION_REASONS
    # The pause twin emits no event of its own, which is the other half of the entry.
    assert not [
        node
        for node in ast.walk(_module_tree("loop.py"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "emit"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "turn.paused"
    ], {"hint": "turn.paused exists now: the two parks are symmetric — drop the entry"}


def test_every_error_expression_the_result_payload_serves_is_filtered() -> None:
    """Was registered (round 2, leak-adjacent): three of four paths out of one payload filtered.

    ``record.error`` and ``metrics["error"]`` come pre-filtered from their writers
    (``public_view.py:public_error_message`` in ``run_state.py``/``loop_phases.py``); the ready
    branch of ``result()`` served ``result.error`` — the deliberately raw ``AgentRunResult.error``
    — straight to an HTTP response. The filter landed in the projection, so the route stays a
    dumb serializer.

    Pinned at *both* ends, and that is still the point: the route hands what ``result()`` returns
    to ``_write_json`` unchanged, so the wire's contents are decided by these two functions
    together. A filter that moved to the route, or a route that stopped writing the dict whole,
    is a change to where the boundary is and has to be re-argued.
    """

    projection = _function_node("reference/backend/projection.py", "result")
    served: list[str] = []
    for node in ast.walk(projection):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "error":
                served.append(ast.unparse(value))
    assert sorted(served) == [
        "public_error_message(result.error)",
        "record.error",
    ], {
        "error_expressions_served": sorted(served),
        "hint": "``record.error`` arrives pre-filtered from record_run_result; the raw "
        "AgentRunResult.error is filtered here. A third expression is an unproven one",
    }
    # The route is the dumb serializer this arrangement depends on: it writes the projection's
    # dict wholesale and applies no filter of its own.
    route = _function_node("reference/backend/http.py", "do_GET", within="make_backend_handler")
    served_wholesale = [
        ast.unparse(node.args[0])
        for node in ast.walk(route)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_write_json"
        and node.args
        and "backend.result(" in ast.unparse(node.args[0])
    ]
    assert served_wholesale == ["backend.result(run_id, self._bearer_token())"], {
        "how_the_route_writes_the_projection": served_wholesale,
        "hint": "the route stopped writing result() unchanged — the filter's placement assumed "
        "it did",
    }
    assert "public_error_message" not in _names_in(route), {
        "hint": "two filters on one value: decide which one owns it",
    }
    # The two sibling writers this branch now agrees with, asserted and not assumed.
    assert "public_error_message" in _names_in(
        _function_node("reference/backend/run_state.py", "record_run_result")
    )
    assert "public_error_message" in _names_in(
        _function_node("loop_phases.py", "build_metrics", within="LoopFinalizer")
    )


def _snapshot_written_keys() -> frozenset[str]:
    """The RunCheckpoint field names ``loop.py:snapshot`` actually writes."""

    snapshot = _function_node("loop.py", "snapshot")
    calls = [
        node
        for node in ast.walk(snapshot)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RunCheckpoint"
    ]
    assert len(calls) == 1, {"run_checkpoint_construction_sites": len(calls)}
    return frozenset(keyword.arg for keyword in calls[0].keywords if keyword.arg)


def test_gap_the_snapshot_omits_the_siblings_of_the_fields_it_writes() -> None:
    """Registered (round 2, three cells of the uncensused run-state carriage family)."""

    from monoid_agent_kernel.core.checkpoint import RunCheckpoint
    from monoid_agent_kernel.loop import AgentToolContext, RunState

    written = _snapshot_written_keys()
    state_fields = {field.name for field in dataclasses.fields(RunState)}
    context_fields = {field.name for field in dataclasses.fields(AgentToolContext)}

    # (a) output_retries rides; its history does not.
    assert {"output_retries", "output_failure_history"} <= state_fields
    assert "output_retries" in written
    assert "output_failure_history" not in written, {
        "hint": "checkpointed now? add it to RunCheckpoint's census and drop the registry entry",
    }
    assert "output_failure_history" not in RunCheckpoint.__dataclass_fields__

    # (b) the RunState counters ride; the context-owned ones do not, and metrics.json mixes them.
    context_counters = {"subagent_count", "subagent_usage", "skill_activation_count"}
    assert context_counters <= context_fields
    assert {"total_usage", "total_tool_calls"} <= written
    assert context_counters.isdisjoint(written), {
        "carried_now": sorted(context_counters & written),
        "hint": "one epoch at last: drop the registry entry",
    }
    assert context_counters.isdisjoint(RunCheckpoint.__dataclass_fields__)
    metrics_names = _names_in(
        _function_node("loop_phases.py", "build_metrics", within="LoopFinalizer")
    )
    assert context_counters <= metrics_names, {
        "hint": "these are the counters that reach metrics.json beside the restored totals",
    }


def test_gap_the_cancellation_flag_is_written_always_and_applied_conditionally() -> None:
    """Registered (round 2): a restore without a token un-cancels a durably-cancelled run."""

    assert "cancellation_requested" in _snapshot_written_keys()
    restore = _function_node("loop.py", "_rehydrate")
    guards = [
        ast.unparse(node.test)
        for node in ast.walk(restore)
        if isinstance(node, ast.If) and "cancellation_requested" in ast.unparse(node.test)
    ]
    assert guards == ["cp.cancellation_requested and self.cancellation_token is not None"], {
        "restore_guards": guards,
        "hint": "the read is unconditional now (or the guard changed): update EXPECTED and drop "
        "the registry entry",
    }


def test_the_two_tenant_meters_sum_the_same_seven_counts() -> None:
    """The gateway meter's twin, pinned behaviorally and cross-asserted against it.

    Two meters, one omission: both summed input/output/total and dropped the four priced
    sub-counts, and fixing one of them would have left the other. The cross-assert below is what
    makes them one rule rather than two that happen to agree today.
    """

    from monoid_agent_kernel.reference.backend.run_state import TenantUsage

    meter = TenantUsage("tenant-1")
    meter.add_metrics({**_MAXIMAL_USAGE, "web_search_calls": 3})
    reported = meter.to_json()
    dropped = NORMALIZED_USAGE_KEYS - set(reported)
    assert dropped == set(), {"dropped_sub_counts": sorted(dropped)}
    # ...and the gateway's ledger drops exactly the same nothing, so the two agree by rule.
    assert dropped == NORMALIZED_USAGE_KEYS - GATEWAY_METER_KEYS
    summed = {key: reported[key] for key in NORMALIZED_USAGE_KEYS}
    assert summed == _MAXIMAL_USAGE, {
        "metered": summed,
        "hint": "a sub-count is reported as a column and not actually summed into it",
    }
    assert reported["web_search_calls"] == 3


def test_both_terminal_paths_meter_the_run_through_one_seam() -> None:
    """``record_run_failure`` metered nothing at all while its sibling fed the ledger.

    Pinned as "one seam", not as "both call ``add_metrics``": the two paths report CUMULATIVE
    totals from different sources (the last checkpoint, and ``AgentRunResult.metrics``), so the
    guard against billing a metered-then-recovered-then-completed run twice is that a single
    function owns the per-run high-water mark. A second caller of ``add_metrics`` would be a
    second place that has to remember the mark.
    """

    failure = _function_node(
        "reference/backend/run_state.py", "record_run_failure", within="RunStateMutationService"
    )
    success = _function_node(
        "reference/backend/run_state.py", "record_run_result", within="RunStateMutationService"
    )
    seam = _function_node(
        "reference/backend/run_state.py", "_meter_run", within="RunStateMutationService"
    )
    assert "_meter_run" in _names_in(success)
    assert "_meter_run" in _names_in(failure), {
        "hint": "the failure path meters nothing again",
    }
    assert "add_metrics" in _names_in(seam)
    for path in (failure, success):
        assert "add_metrics" not in _names_in(path), {
            "hint": "a second ledger writer beside the seam that owns the high-water mark",
        }


def test_gap_three_mappers_answer_a_terminal_limited_run_three_ways() -> None:
    """Registered (round 2): one run, three surfaces, three states.

    Pinned value-level on all three, so a fix to any one of them trips this census and forces
    the semantic decision the divergence has been deferring.
    """

    from monoid_agent_kernel.core.lifecycle import (
        LoopSession,
        SessionState,
        session_state_from_run_status,
        state_from_suspension,
    )

    parked = Suspension(
        reason="terminal", status="limited", error_code="output_validator_unsatisfied"
    )

    class _ClosingLoop:
        def close(self) -> Any:
            return dataclasses.replace(_maximal_turn(), status="limited")

    session = LoopSession(loop=_ClosingLoop())  # type: ignore[arg-type]
    session.close()

    verdicts = {
        "core/lifecycle.py:state_from_suspension": state_from_suspension(parked),
        "core/lifecycle.py:session_state_from_run_status": session_state_from_run_status(
            "limited", terminal=True
        ),
        "core/lifecycle.py:LoopSession.close": session.state,
    }
    assert verdicts == {
        "core/lifecycle.py:state_from_suspension": SessionState.FAILED,
        "core/lifecycle.py:session_state_from_run_status": SessionState.LIMITED,
        "core/lifecycle.py:LoopSession.close": SessionState.COMPLETED,
    }, {
        "verdicts": {name: value.value for name, value in verdicts.items()},
        "hint": "harmonized? make them agree, update EXPECTED and drop the registry entry",
    }
    # And LIMITED is a real state all three could have returned.
    assert REASON_TO_STATE["limited"] is SessionState.LIMITED


def test_gap_the_recovery_park_is_built_outside_the_durable_status_vocabulary() -> None:
    """Registered (round 2): reference/backend/recovery.py mints Suspension(status="running").

    Latent, and pinned as latent: the synthetic park is re-driven rather than serialized, so the
    landmine is documented by showing what happens the first time something checkpoints it.
    """

    recovered = _function_node("reference/backend/recovery.py", "run_recovered")
    minted = [
        {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in node.keywords
            if keyword.arg
        }
        for node in ast.walk(recovered)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Suspension"
    ]
    assert minted == [
        {"reason": "'awaiting_tasks'", "status": "'running'", "has_external": "True"}
    ], {"suspensions_minted_during_recovery": minted}

    synthetic = Suspension(reason="awaiting_tasks", status="running", has_external=True)  # type: ignore[arg-type]
    payload = suspension_checkpoint_payload(synthetic)
    assert payload["status"] == "running"
    with pytest.raises(ValueError):
        suspension_from_checkpoint_payload(payload)


# --------------------------------------------------------------------------------------
# FUTURE_FAMILIES — the numeric claims, pinned so they rot loudly
# --------------------------------------------------------------------------------------
#
# A declared-but-uncensused family is a prediction, and a prediction with a stale number is
# worse than none. The counts in the risk notes are asserted here against the code they
# describe.


def _future_family(name: str) -> FutureFamily:
    return next(family for family in FUTURE_FAMILIES if family.family == name)


# core/manifest.py:build_run_manifest — the four limits the durable record of a run carries.
MANIFEST_LIMIT_KEYS = frozenset(
    {"max_steps", "max_tool_calls", "max_bytes_read", "max_duration_s"}
)
RUN_LIMITS_FIELD_COUNT = 15


def test_future_family_run_limits_carries_four_of_fifteen() -> None:
    from monoid_agent_kernel.core.spec import RunLimits

    fields = {field.name for field in dataclasses.fields(RunLimits)}
    assert len(fields) == RUN_LIMITS_FIELD_COUNT, {
        "run_limits_fields": sorted(fields),
        "hint": "the '4 of 15' claim in FUTURE_FAMILIES is now wrong: update both",
    }
    builder = _function_node("core/manifest.py", "build_run_manifest")
    carried = [
        # Strict, like the two literal-argument extractors above: this list is diffed for
        # equality against MANIFEST_LIMIT_KEYS, so a ``**`` merge into the literal must fail
        # rather than leave the pin green while the manifest carries more than four limits.
        _pinned_dict_keys(keyword.value, "core/manifest.py: limits={...}")
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "limits" and isinstance(keyword.value, ast.Dict)
    ]
    assert len(carried) == 1, {"manifest_limits_literals": len(carried)}
    assert carried[0] == MANIFEST_LIMIT_KEYS, {
        "carried_into_the_manifest": sorted(carried[0]),
        "hint": "widened? update EXPECTED and the FUTURE_FAMILIES risk note together",
    }
    assert carried[0] <= fields
    assert _future_family("run limits").carrier_count == 1


def test_future_family_stream_frames_has_two_hand_built_carriers() -> None:
    """The frame writer and the frame reader, and the chunk union they both re-implement."""

    from monoid_agent_kernel.providers.base import ModelStreamChunk

    members = {member.__name__ for member in get_args(ModelStreamChunk)}
    assert members == {"TextDelta", "ReasoningDelta", "ToolCallDelta", "TurnComplete"}, {
        "chunk_union": sorted(members),
        "hint": "a fifth chunk type must reach both hand-built carriers",
    }
    writer = _function_node("reference/llm_gateway/service.py", "_chunk_to_frame")
    branches = {
        ast.unparse(node.args[1])
        for node in ast.walk(writer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "isinstance"
        and len(node.args) == 2
    }
    # ``TurnComplete`` is dropped by design: the gateway mints its own terminal frame.
    assert branches == members - {"TurnComplete"}, {
        "frame_writer_branches": sorted(branches),
        "hint": "a chunk type the frame writer cannot translate is dropped silently",
    }
    # The two carriers bound to each other: every frame type the writer can put on the wire is a
    # frame type the reader dispatches on, and the reader's extra two are the gateway's own
    # terminal/error frames (which no ``ModelStreamChunk`` produces).
    written = {
        value.value
        for node in ast.walk(writer)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant)
        and key.value == "type"
        and isinstance(value, ast.Constant)
    }
    read = _handled_event_types(
        _function_node("providers/gateway.py", "_chunk_from_event"), "event_type"
    )
    assert written == {"text_delta", "reasoning_delta", "tool_call_delta"}, sorted(written)
    assert written <= read, {
        "written_but_not_dispatched_on": sorted(written - read),
        "hint": "a frame the writer emits and the reader drops",
    }
    assert read - written == {"turn_complete", "error"}
    assert _future_family("stream frames").carrier_count == 2


# core/schemas.py:STATUS_SCHEMA — what the operator-facing status file declares about itself.
STATUS_SCHEMA_KEYS = frozenset(
    {"run_id", "state", "terminal", "last_event_seq", "last_event_type", "updated_at"}
)


def test_future_family_status_json_is_already_wider_than_its_own_schema() -> None:
    from monoid_agent_kernel.core.schemas import STATUS_SCHEMA

    declared = frozenset(STATUS_SCHEMA["properties"])
    assert declared == STATUS_SCHEMA_KEYS, {
        "missing": sorted(STATUS_SCHEMA_KEYS - declared),
        "extra": sorted(declared - STATUS_SCHEMA_KEYS),
        "hint": "the schema moved: recheck the FUTURE_FAMILIES claim it backs",
    }
    assert STATUS_SCHEMA["additionalProperties"] is True, {
        "hint": "the cap closed: every undeclared key the sink writes is now a failure",
    }
    # The claim itself: the sink writes a ``metrics`` block the schema never declares.
    sink = _function_node("recorder.py", "emit", within="StatusJsonSink")
    written = {
        node.slice.value
        for node in ast.walk(sink)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "state"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    assert "metrics" in written and "metrics" not in declared, {
        "written_but_undeclared": sorted(written - declared),
        "hint": "declared now? update EXPECTED and the FUTURE_FAMILIES risk note",
    }
    assert _future_family("status.json projection").carrier_count == 2


# --------------------------------------------------------------------------------------
# New-carrier backstop
# --------------------------------------------------------------------------------------

# Every module that *carries* each headline field today, by AST occurrence rather than by
# substring. Substring containment fails open in the direction that matters: a file whose only
# mention is a comment or a docstring counted as a carrier, so the pinned set said the census
# had accounted for a file that has no code for the field at all (``validated_call.py`` for
# ``provider_retried`` was exactly that, and it makes the same mistake in reverse invisible --
# deleting the only real use while a prose mention keeps the entry green). A carrier is a file
# where the name occurs in *code*: an exact string constant, or an identifier that contains it
# (``mark_provider_usage``, ``provider_usage_of`` -- the helpers through which most files carry
# the fact). Paths are relative to ``src/monoid_agent_kernel``.
CARRIER_FILES: dict[str, frozenset[str]] = {
    "config_recoverable": frozenset(
        {
            # Joined in the burn-down: the immutable record of one call, the live stream lane
            # (outcome type, sidecar writer/reader) that used to classify with half the
            # vocabulary.
            "core/model_content.py",
            "core/model_io.py",
            "core/model_stream.py",
            "core/result.py",
            "core/schemas.py",
            "errors.py",
            "loop.py",
            "providers/gateway.py",
            "providers/openai.py",
            # Joined in the burn-down: the backend record that carries the classification to the
            # surfaces an operator reads (type, protocol, the driver that promotes it, and the
            # projection that serves it).
            "reference/backend/ports.py",
            "reference/backend/projection.py",
            "reference/backend/recovery.py",
            "reference/backend/run_state.py",
            "reference/backend/run_types.py",
            "reference/backend/service.py",
            "reference/backend/session_drive.py",
            "reference/llm_gateway/http.py",
        }
    ),
    "provider_retried": frozenset(
        {
            "contracts.py",
            "core/model_io.py",
            # Joined in the burn-down: the park type that records the failed turn, its event
            # schema, and the loop that writes both plus the two transcript records.
            "core/result.py",
            "core/schemas.py",
            "errors.py",
            "loop.py",
            "model_call.py",
            "observability/otel.py",
            "providers/base.py",
            "providers/gateway.py",
            # Joined in the burn-down: the classifier reports the retries its own SDK made.
            "providers/openai.py",
            "reference/llm_gateway/http.py",
            "reference/llm_gateway/service.py",
        }
    ),
    "provider_usage": frozenset(
        {
            "core/model_io.py",
            # Joined in the burn-down: turn.failed declares what the refused call cost.
            "core/schemas.py",
            "loop.py",
            "providers/base.py",
            "providers/gateway.py",
            "reference/llm_gateway/http.py",
            "reference/llm_gateway/service.py",
        }
    ),
    "reasoning_tokens": frozenset(
        {
            "core/schemas.py",
            "loop.py",
            "providers/_common.py",
            # Joined in the burn-down: the two tenant ledgers now sum the priced sub-counts
            # instead of folding them away.
            "reference/backend/run_state.py",
            "reference/llm_gateway/service.py",
        }
    ),
    # The W5 echo pair. ``reasoning_applied`` is the registered v0.21-track:B1 gap and has no
    # carrier at all yet — pinned empty, so the first file to carry it fails here and has to
    # join the echo censuses above rather than arriving on one transport quietly.
    "generation_applied": frozenset(
        {
            "providers/base.py",
            "providers/gateway.py",
            "reference/llm_gateway/service.py",
        }
    ),
    "schema_applied": frozenset(
        {
            "providers/base.py",
            "providers/gateway.py",
            "reference/llm_gateway/service.py",
        }
    ),
    "reasoning_applied": frozenset(),
}


def _python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _code_occurrences(tree: ast.AST) -> tuple[frozenset[str], frozenset[str]]:
    """(exact string constants, identifiers) named in code — never in comments or docstrings."""

    constants: set[str] = set()
    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            constants.add(node.value)
        elif isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            identifiers.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifiers.add(node.name)
        elif isinstance(node, ast.alias):
            identifiers.add(node.name)
            if node.asname:
                identifiers.add(node.asname)
    return frozenset(constants), frozenset(identifiers)


# Parsing the package once per headline field costs seconds; the census is meant to be cheap
# enough that nobody is tempted to skip it.
_CODE_OCCURRENCE_CACHE: dict[Path, tuple[frozenset[str], frozenset[str]]] = {}


_SOURCE_CACHE: dict[Path, str] = {}


def _carries_in_code(path: Path, field_name: str) -> bool:
    if path not in _SOURCE_CACHE:
        _SOURCE_CACHE[path] = path.read_text(encoding="utf-8")
    source = _SOURCE_CACHE[path]
    if field_name not in source:
        # Not named anywhere, comments included, so it cannot be named in code either.
        return False
    if path not in _CODE_OCCURRENCE_CACHE:
        _CODE_OCCURRENCE_CACHE[path] = _code_occurrences(ast.parse(source, filename=str(path)))
    constants, identifiers = _CODE_OCCURRENCE_CACHE[path]
    return field_name in constants or any(field_name in name for name in identifiers)


@pytest.mark.parametrize("field_name", sorted(CARRIER_FILES))
def test_no_unregistered_carrier_file_appears_for_a_headline_field(field_name: str) -> None:
    found = {
        path.relative_to(PACKAGE).as_posix()
        for path in _python_files(PACKAGE)
        if _carries_in_code(path, field_name)
    }
    expected = CARRIER_FILES[field_name]
    assert found == expected, {
        "new carrier file for": field_name,
        "detail": "new carrier file for "
        f"{field_name} — extend the census or register the site",
        "added": sorted(found - expected),
        "removed": sorted(expected - found),
    }


# The carriers no Python scan reaches. A wire key is a contract with things outside this
# package too: the shipped Studio bundle reads the event stream, and the two documents below
# are what a third-party client implements against. A rename that breaks them is a rename that
# breaks a deployment, so it fails a test that names the file.
EXTRA_CARRIERS: dict[str, tuple[str, ...]] = {
    # Glob, because the bundle's filename carries a content hash.
    "src/monoid_agent_kernel/reference/studio/web/dist/assets/index-*.js": (
        "turn.failed",
        "retryable",
        "metrics.updated",
        "input_tokens",
        "total_tokens",
    ),
    "docs/CONTRACTS.md": (
        "config_recoverable",
        "provider_retried",
        "retryable",
        "turn.failed",
        "http_status",
    ),
    "docs/OBSERVABILITY.md": (
        "metrics.updated",
        "input_tokens",
        "total_tokens",
        "retryable",
    ),
}


@pytest.mark.parametrize("pattern", sorted(EXTRA_CARRIERS))
def test_the_non_python_carriers_still_name_the_fields_they_read(pattern: str) -> None:
    """Substring here on purpose: minified JavaScript and prose have no AST this suite parses.

    The claim is only "this file still mentions this wire key", which is exactly the claim that
    catches a rename: the shipped UI reads `turn.failed`/`retryable`/`metrics.updated` and its
    token counters by name out of the event stream, and it is built from source this suite
    cannot see.
    """

    matches = sorted(ROOT.glob(pattern))
    assert matches, {"no_file_matches": pattern, "hint": "a carrier moved or was deleted"}
    for path in matches:
        text = path.read_text(encoding="utf-8", errors="replace")
        missing = [field_name for field_name in EXTRA_CARRIERS[pattern] if field_name not in text]
        assert missing == [], {
            "carrier": path.relative_to(ROOT).as_posix(),
            "fields_no_longer_named": missing,
            "hint": "a wire-key rename that this carrier was not part of",
        }


def test_every_registered_carrier_file_is_a_known_carrier_of_its_field() -> None:
    """The backstop's pinned sets and the gap registry must not drift apart."""

    registered_paths = {gap.carrier.split(":", 1)[0] for gap in KNOWN_GAPS}
    all_carriers = frozenset().union(*CARRIER_FILES.values())
    # ``core/checkpoint.py`` and ``core/spec.py`` carry aliased facts under other names
    # (``provider_http_status``, an omitted ``generation`` block), so no headline-name scan
    # reaches them and they are registered without appearing in one.
    alias_only = {
        "core/checkpoint.py",
        "core/spec.py",
        # Registrations whose fact is not a headline wire field: the three disagreeing
        # terminal-state mappers, a Suspension built outside the durable status vocabulary, and
        # the cause-vs-park vocabulary collision on one event name. Each is pinned by its own
        # assertion below; no name scan can reach them.
        "core/lifecycle.py",
        "reference/backend/recovery.py",
        "loop.py",
    }
    unaccounted = registered_paths - all_carriers - alias_only
    assert unaccounted == set(), {"registered_but_not_a_scanned_carrier": sorted(unaccounted)}
    # Every family the census covers must not also be declared as one it does not.
    covered_families = {gap.family for gap in KNOWN_GAPS}
    declared_future = {family.family for family in FUTURE_FAMILIES}
    assert covered_families.isdisjoint(declared_future), {
        "declared_uncensused_but_registered_as_covered": sorted(
            covered_families & declared_future
        ),
    }
