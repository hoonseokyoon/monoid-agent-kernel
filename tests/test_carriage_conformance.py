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
the assignable key domain of the *live* ``normalize_usage`` (a wrapper is censused too), the
shipped ``_write_exception`` driven against a capturing host, the reader set discovered from the
module.  Where a hand-written EXPECTED remains it is pinned in full and diffed against a derived
set, never spot-checked; and the file-scan backstop reads code occurrences (AST), because
substring containment counted a comment as a carrier and so failed open.

**The green-with-registered-gaps contract.**  This suite is GREEN today, and today's reality
is not the ideal: many cells are unbound.  Every one of them is registered in
:data:`KNOWN_GAPS` with its carrier and disposition, and the assertions below encode reality
*exactly* — a pinned key set, an asserted loss.  So fixing a gap **breaks this suite**, and
that is the mechanism working: the fixer must update the EXPECTED constant and delete the
registry entry in the same change.  Do not loosen an assertion to accommodate a fix.
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

from monoid_agent_kernel.core.lifecycle import REASON_TO_STATE
from monoid_agent_kernel.core.result import (
    AgentTurnResult,
    Suspension,
    _SUSPENSION_REASONS,
    suspension_checkpoint_payload,
    suspension_from_checkpoint_payload,
)
from monoid_agent_kernel.core.schemas import EVENT_DATA_SCHEMAS, TRANSCRIPT_RECORD_SCHEMA
from monoid_agent_kernel.core.spec import GenerationConfig, ModelConfig, ReasoningConfig
from monoid_agent_kernel.errors import ModelAdapterError, TurnNotSettled
from monoid_agent_kernel.model_call import _recordable_usage
from monoid_agent_kernel.observability.otel import _chat_finish_attrs, _subagent_finish_attrs
from monoid_agent_kernel.providers import gateway as gateway_client
from monoid_agent_kernel.providers._common import normalize_usage
from monoid_agent_kernel.providers.base import ModelTurn, mark_provider_usage, provider_usage_of
from monoid_agent_kernel.reference.llm_gateway.http import (
    _error_body,
    _model_error_status,
    _stream_error_frame,
    make_llm_gateway_handler,
)
from monoid_agent_kernel.reference.llm_gateway.service import (
    LLM_TURN_PROTOCOL_VERSION,
    LlmGatewayTurnRequest,
    LlmGatewayUsage,
    _applied_echoes,
)

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
    # --- config_recoverable: born on the client, dies at the hop -----------------------
    CarriageGap(
        "transportable-error",
        "config_recoverable",
        "reference/llm_gateway/http.py:_error_body",
        "no wire key: the one error field with no transport, so a config-fixable refusal "
        "arrives one hop out as an ordinary failure (only the 4xx status hints at it)",
        "burn-down",
    ),
    CarriageGap(
        "transportable-error",
        "config_recoverable",
        "providers/gateway.py:_parse_gateway_response",
        "reconstructs config_recoverable=False even when the body carries the key",
        "burn-down",
    ),
    CarriageGap(
        "transportable-error",
        "config_recoverable",
        "providers/gateway.py:_chunk_from_event",
        "stream-error twin of _parse_gateway_response: same dropped read",
        "burn-down",
    ),
    CarriageGap(
        "transportable-error",
        "config_recoverable",
        "providers/gateway.py:_error_from_status_body",
        "non-200 twin of _parse_gateway_response: same dropped read",
        "burn-down",
    ),
    CarriageGap(
        "suspension",
        "config_recoverable",
        "core/schemas.py:TRANSCRIPT_RECORD_SCHEMA",
        "the model_turn branch does not declare config_recoverable although loop.py:_apump_turn "
        "writes it; additionalProperties=True is what keeps the record valid, not the schema",
        "burn-down",
    ),
    CarriageGap(
        "transportable-error",
        "config_recoverable",
        "providers/openai.py:_model_error_from_openai",
        "no branch sets config_recoverable, so a provider-side config refusal is never flagged "
        "at the one adapter that can classify it",
        "burn-down",
    ),
    # --- http_status: carried by some siblings, dropped by the others ------------------
    CarriageGap(
        "transportable-error",
        "http_status",
        "loop.py:_record_failure",
        "the failure bundle (write_failure) omits http_status while the run.failed event beside "
        "it carries it, so the operator's restore aid loses the status the log kept",
        "burn-down",
    ),
    CarriageGap(
        "transportable-error",
        "http_status",
        "providers/gateway.py:_exact_gateway_int",
        "validator has no http_status parameter, so a malformed-payload error it raises loses "
        "the status its _exact_gateway_bool/_gateway_string siblings forward",
        "burn-down",
    ),
    CarriageGap(
        "transportable-error",
        "http_status",
        "providers/gateway.py:_gateway_fragment_string",
        "same missing parameter as _exact_gateway_int",
        "burn-down",
    ),
    CarriageGap(
        "transportable-error",
        "http_status",
        "providers/gateway.py:_gateway_usage",
        "same missing parameter as _exact_gateway_int",
        "burn-down",
    ),
    CarriageGap(
        "transportable-error",
        "http_status",
        "providers/gateway.py:_portable_gateway_payload",
        "fourth sibling with the same missing parameter",
        "burn-down",
    ),
    CarriageGap(
        "suspension",
        "retryable/config_recoverable",
        "core/schemas.py:EVENT_DATA_SCHEMAS",
        "run.failed (the terminal twin of turn.failed) declares neither, although "
        "loop.py:fail_recoverable promotes a turn.failed carrying both into exactly this "
        "record — so the terminal log of a config-fixable failure cannot say it was one",
        "burn-down",
    ),
    CarriageGap(
        "suspension",
        "retryable/config_recoverable",
        "loop.py:_record_failure",
        "the run.failed emit site and the write_failure bundle beside it are built from the "
        "same state and neither writes the classification the promoted turn.failed carried",
        "burn-down",
    ),
    # --- provider_retried / provider_usage: stamped, then unrecorded -------------------
    CarriageGap(
        "transportable-error",
        "provider_retried",
        "providers/openai.py:_model_error_from_openai",
        "no branch sets provider_retried, so the adapter that owns a retry loop never reports "
        "having run it",
        "burn-down",
    ),
    CarriageGap(
        "suspension",
        "provider_retried",
        "core/result.py:Suspension",
        "absent from every kernel record of a failed turn (Suspension, its checkpoint payload, "
        "and the turn.failed event), so it survives only inside the live exception",
        "burn-down",
    ),
    CarriageGap(
        "suspension",
        "provider_usage",
        "core/schemas.py:EVENT_DATA_SCHEMAS",
        "turn.failed declares no usage (nor provider_retried) although the transcript twin "
        "written on the same failure records both cost and classification",
        "burn-down",
    ),
    CarriageGap(
        "suspension",
        "provider_retried",
        "loop.py:_apump_turn",
        "the SUCCESS model_turn transcript record omits it although ModelTurn carries it and "
        "the receipt records it, so the private replay artifact of a retried-then-successful "
        "call reads as a clean single attempt (its failure twin omits it too)",
        "burn-down",
    ),
    CarriageGap(
        "suspension",
        "provider_error_code",
        "core/result.py:Suspension",
        "not a Suspension field, so the provider code a driver needs to decide re-attempt vs "
        "config-fix is lost through checkpoint recovery (it lives only on RunState)",
        "burn-down",
    ),
    # --- usage sub-counts: normalized, then flattened ----------------------------------
    CarriageGap(
        "usage",
        "cache_read_tokens/cache_creation_tokens/reasoning_tokens/audio_tokens",
        "reference/llm_gateway/service.py:LlmGatewayUsage",
        "the tenant meter sums only input/output/total, dropping the four priced sub-counts "
        "normalize_usage emits; corollary A8 — a billed failure that reports ONLY sub-counts "
        "meters as total=0, so the priced call is invisible to the meter entirely",
        "burn-down",
    ),
    CarriageGap(
        "usage",
        "cache_read_tokens/cache_creation_tokens/audio_tokens",
        "core/schemas.py:EVENT_DATA_SCHEMAS",
        "metrics.updated declares reasoning_tokens but not its three sibling sub-counts, so a "
        "cache-heavy run's priced detail never reaches a live consumer",
        "burn-down",
    ),
    CarriageGap(
        "usage",
        "cache_read_tokens/cache_creation_tokens/reasoning_tokens/audio_tokens",
        "loop.py:_run_subagent_child",
        "the parent roll-up hard-codes a 3-key tuple, so a child's sub-counts never reach the "
        "parent's budget — an undercount in exactly the aggregate a bound is checked against",
        "burn-down",
    ),
    CarriageGap(
        "usage",
        "provider_usage",
        "model_call.py:_recordable_usage",
        "accepts int subclasses (IntEnum) that providers/base.py:provider_usage_of, "
        "providers/gateway.py:_reported_error_usage and core/model_io.py:ModelCallReceipt "
        "all reject, so one stamp reads as three different usages depending on the consumer",
        "burn-down",
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


def test_registry_entries_are_well_formed() -> None:
    assert KNOWN_GAPS
    for gap in KNOWN_GAPS:
        assert gap.disposition in DISPOSITIONS, gap
        assert ":" in gap.carrier, gap
        assert gap.gap.strip(), gap


def test_registry_carrier_locations_exist() -> None:
    """A renamed carrier must rot its registry entry rather than point at nothing."""

    missing: list[str] = []
    for gap in KNOWN_GAPS:
        relative_path, symbol = gap.carrier.split(":", 1)
        path = PACKAGE / relative_path
        if not path.is_file():
            missing.append(f"{gap.carrier} (no such file)")
            continue
        if symbol not in path.read_text(encoding="utf-8"):
            missing.append(f"{gap.carrier} (symbol absent)")
    assert missing == [], {"stale_registry_entries": missing}


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

    signature = inspect.signature(ModelAdapterError.__init__)
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
# than inferred, because three of the six are renames and one has no key at all.
TRANSPORTABLE_ERROR_WIRE_ALIASES: dict[str, str] = {
    # ``message`` is positional-only-in-effect: read back through ``str(exc)``, never an attribute.
    "error": "message",
    # Compat-frozen rename. The wire's ``error_code`` is the PROVIDER code; the kernel-level
    # ``ModelAdapterError.error_code`` has no wire slot and resets to the class default on
    # reconstruction (registered by-design above).
    "error_code": "provider_error_code",
    "retryable": "retryable",
    # Server-derived: ``_model_error_status`` picks the status, it is not copied off the exception.
    "http_status": "http_status",
    "provider_retried": "provider_retried",
    # Omitted-when-empty, so any probe of this key must stamp a non-empty usage first.
    "usage": "provider_usage",
}

# The one transportable fact with no wire key at all (registered burn-down above).
TRANSPORTABLE_ERROR_UNCARRIED = frozenset({"config_recoverable"})

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


def _dict_keys(node: ast.Dict) -> frozenset[str]:
    return frozenset(
        key.value
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    )


def _emit_data_keys(relative_path: str, event_type: str) -> frozenset[str]:
    """Keys of the ``data=`` dict literal at ``recorder.emit("<event_type>", ..., data={...})``.

    The emit site is an inline literal inside a long pump method, so it cannot be imported and
    diffed — but it is exactly where a new key is added without a matching schema entry.
    """

    found: list[frozenset[str]] = []
    for node in ast.walk(_module_tree(relative_path)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "emit" or not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and first.value == event_type):
            continue
        for keyword in node.keywords:
            if keyword.arg == "data" and isinstance(keyword.value, ast.Dict):
                found.append(_dict_keys(keyword.value))
    assert len(found) == 1, {
        "event_type": event_type,
        "emit_sites_with_a_literal_data_dict": len(found),
        "hint": "a second emit site is a twin that must be censused too",
    }
    return found[0]


def _literal_dict_keys_where(relative_path: str, key: str, value: str) -> list[frozenset[str]]:
    """Key sets of every dict literal in a module whose ``key`` is the constant ``value``."""

    matches: list[frozenset[str]] = []
    for node in ast.walk(_module_tree(relative_path)):
        if not isinstance(node, ast.Dict):
            continue
        for dict_key, dict_value in zip(node.keys, node.values):
            if not (isinstance(dict_key, ast.Constant) and dict_key.value == key):
                continue
            if isinstance(dict_value, ast.Constant) and dict_value.value == value:
                matches.append(_dict_keys(node))
    return matches


def _call_dict_arg_keys(relative_path: str, method: str) -> list[frozenset[str]]:
    """Key sets of the single dict literal passed positionally to ``obj.<method>({...})``."""

    matches: list[frozenset[str]] = []
    for node in ast.walk(_module_tree(relative_path)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != method or not node.args:
            continue
        if isinstance(node.args[0], ast.Dict):
            matches.append(_dict_keys(node.args[0]))
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

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
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
    """

    returns = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Name)
    ]
    assert len(returns) == 1, {"returns_of_a_named_dict": len(returns)}
    result_name = returns[0].id  # type: ignore[union-attr]

    literals: dict[str, set[str]] = {}
    for node in ast.walk(function):
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and isinstance(value, ast.Dict):
            literals.setdefault(target.id, set()).update(_dict_keys(value))

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

    keys = set(literals.get(result_name, set()))
    unanalyzable: list[str] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
                continue
            if target.value.id != result_name:
                continue
            index = target.slice
            if isinstance(index, ast.Constant) and isinstance(index.value, str):
                keys.add(index.value)
            elif isinstance(index, ast.Name) and index.id in loop_sources:
                keys |= literals.get(loop_sources[index.id], set())
            else:
                unanalyzable.append(ast.dump(index))
    assert unanalyzable == [], {
        "writes_the_census_cannot_read": unanalyzable,
        "hint": "a new emit shape: teach _emitted_result_keys about it rather than dropping it",
    }
    return frozenset(keys)


# The gateway module's wire-reading helpers. A function that both constructs a
# ``ModelAdapterError`` and reads the wire through one of these *is* an error reader.
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


def _literal_wire_keys(function: _FunctionNode) -> frozenset[str]:
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
    {"reason", "retryable", "http_status", "config_recoverable"}
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
    {"error", "error_code", "provider_error_code", "retryable", "http_status", "config_recoverable"}
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


def test_1c_turn_failed_carries_the_wire_classification_but_not_the_cost() -> None:
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

    # The event carries the fact the wire cannot (family 2's headline burn-down), so a fix that
    # adds the wire key must not need to touch this event.
    assert TRANSPORTABLE_ERROR_UNCARRIED <= declared

    # ...and drops the two the wire carries. Registered burn-down: the transcript record written
    # on the very same failure keeps the cost, so this event is the odd one out, not the rule.
    assert wire_facts - declared == {"provider_retried", "provider_usage"}
    assert "usage" not in declared


# core/schemas.py:TRANSCRIPT_RECORD_SCHEMA, model_turn branch. ``config_recoverable`` is written
# by loop.py:_apump_turn but never declared here (registered burn-down); the record stays valid
# only because the branch sets additionalProperties=True.
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
        "http_status",
    }
)


def _transcript_model_turn_branch() -> dict[str, Any]:
    for variant in TRANSCRIPT_RECORD_SCHEMA["oneOf"]:
        if variant["properties"]["kind"]["const"] == "model_turn":
            return variant
    raise AssertionError("TRANSCRIPT_RECORD_SCHEMA has no model_turn branch")


def test_1d_transcript_schema_declares_less_than_the_writer_writes() -> None:
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
    assert undeclared == {"config_recoverable"}, {
        "written_but_undeclared": sorted(undeclared),
        "hint": "declare it in TRANSCRIPT_RECORD_SCHEMA and drop the registry entry",
    }
    assert branch["additionalProperties"] is True


# The other ``kind="model_turn"`` record: the one written when the turn SUCCEEDS. It shares a
# schema branch with the failure record above, so it is the same declared vocabulary minus the
# error half -- and it drops ``provider_retried``, which ``ModelTurn`` carries and the receipt
# records (registered burn-down).
TRANSCRIPT_MODEL_TURN_SUCCESS_KEYS = frozenset(
    {"kind", "step", "response_id", "final_text", "tool_calls", "usage"}
)


def test_1d_the_success_transcript_record_omits_the_retry_its_turn_reports() -> None:
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
    # The fact the record cannot state, present on the object it was written from.
    assert "provider_retried" in {field.name for field in dataclasses.fields(ModelTurn)}
    assert "provider_retried" not in written[0], {
        "hint": "carried now? update EXPECTED and drop the registry entry",
    }


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


def test_1d_failure_bundle_drops_the_status_the_event_beside_it_keeps() -> None:
    """loop.py:_record_failure emits run.failed and writes failure.json from the same state."""

    event_keys = _emit_data_keys("loop.py", "run.failed")
    bundles = _call_dict_arg_keys("loop.py", "write_failure")
    assert len(bundles) == 1, {"write_failure_sites": len(bundles)}
    assert "http_status" in event_keys
    assert "http_status" not in bundles[0], {
        "hint": "carried now? update EXPECTED and drop the registry entry",
    }
    assert {"error", "error_code", "provider_error_code"} <= (event_keys & bundles[0])


# The terminal-failure trio, pinned in full rather than probed for one key. ``run.failed`` is
# ``turn.failed``'s terminal twin -- ``fail_recoverable`` promotes one into the other -- and it
# declares neither ``retryable`` nor ``config_recoverable``, so the classification a driver used
# to decide "park for a config fix" is gone from the record of having given up (burn-down).
RUN_FAILED_EVENT_KEYS = frozenset(
    {"error", "error_code", "type", "provider_error_code", "http_status"}
)
FAILURE_BUNDLE_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "error",
        "error_code",
        "provider_error_code",
        "type",
        "last_good_seq",
        "restore_hint",
    }
)


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


def test_1d_the_terminal_failure_record_drops_the_classification_its_twin_carries() -> None:
    """``turn.failed`` -> ``run.failed`` is a real promotion path, and it loses two facts.

    ``AgentLoop.fail_recoverable`` exists to turn an exhausted recoverable turn failure into the
    terminal record, and ``close()`` performs the same promotion for an unrecovered park -- so a
    config-recoverable terminal failure is an ordinary outcome, not a hypothetical. The terminal
    record cannot say so: ``retryable`` and ``config_recoverable`` ride the turn event and stop
    there (registered burn-down).
    """

    turn_failed = frozenset(EVENT_DATA_SCHEMAS["turn.failed"]["properties"])
    lost = turn_failed - RUN_FAILED_EVENT_KEYS
    assert lost == {"retryable", "config_recoverable"}, {
        "lost_on_promotion": sorted(lost),
        "hint": "carried now? update EXPECTED and drop the registry entry",
    }
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
    signature = inspect.signature(_error_body)
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
    assert written["provider_retried"] is True
    assert written["usage"] == provider_usage_of(exc)


def test_2a_a_config_recoverable_refusal_reaches_the_wire_as_a_422_and_nothing_else() -> None:
    """The one status the wire *does* carry the uncarried fact through, end to end.

    ``config_recoverable`` has no wire key (registered burn-down), and the only thing a client
    can read it off is the 4xx ``_model_error_status`` picks. Pinned on the real writer so a
    change to either half is a failure here.
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
    assert TRANSPORTABLE_ERROR_UNCARRIED.isdisjoint(written)


def test_2a_server_error_body_writes_exactly_the_alias_table() -> None:
    body = _maximal_error_body()
    assert frozenset(body) == SERVER_ERROR_BODY_KEYS, {
        "missing": sorted(SERVER_ERROR_BODY_KEYS - set(body)),
        "extra": sorted(set(body) - SERVER_ERROR_BODY_KEYS),
    }
    # The one transportable fact with no key (registered burn-down).
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


def _maximal_wire_body() -> dict[str, Any]:
    """The server body plus ``config_recoverable`` — present so a reader that reads it would."""

    body = _maximal_error_body()
    body["config_recoverable"] = True
    return body


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
        "http_status": got.http_status,
        "provider_retried": got.provider_retried,
        "provider_usage": provider_usage_of(got),
    }
    assert carried == {
        "provider_error_code": body["error_code"],
        "retryable": body["retryable"],
        "http_status": body["http_status"],
        "provider_retried": body["provider_retried"],
        "provider_usage": body["usage"],
    }, {"reader": reader, "reconstructed": carried}

    # The message survives, but R3 wraps it with the status line it was read from.
    assert body["error"] in str(got)

    # Registered losses. Both flip to a failure the moment a reader starts binding them.
    assert got.config_recoverable is False, "config_recoverable is now read — update the census"
    assert got.error_code == ModelAdapterError.error_code, (
        "the kernel error_code has no wire slot; it must reconstruct to the class default"
    )
    assert "config_recoverable" not in vars(type(got))


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
# emits, or stops reading one, fails here -- including the moment ``config_recoverable``
# finally becomes a read (its registered burn-down), which must update these sets.
GATEWAY_READER_WIRE_KEYS: dict[str, frozenset[str]] = {
    "providers/gateway.py:_parse_gateway_response": frozenset(
        {
            # error half
            "error",
            "error_code",
            "retryable",
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
        {"error", "error_code", "retryable", "provider_retried", "usage"}
    ),
}


def test_2b_the_registered_reader_list_is_every_reader_the_module_has() -> None:
    """The reader list was a hand-written dict of three, so a fourth sibling joined unseen.

    A module-level function that both constructs a ``ModelAdapterError`` and reads the wire
    through this module's own reading helpers *is* an error reader, whatever it is called.
    """

    tree = _module_tree("providers/gateway.py")
    discovered = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        constructs_error = any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == "ModelAdapterError"
            for inner in ast.walk(node)
        )
        reads_the_wire = any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id in GATEWAY_WIRE_READ_HELPERS
            for inner in ast.walk(node)
        )
        if constructs_error and reads_the_wire:
            discovered.add(f"providers/gateway.py:{node.name}")

    assert discovered == set(GATEWAY_ERROR_READERS), {
        "unregistered_readers": sorted(discovered - set(GATEWAY_ERROR_READERS)),
        "registered_but_no_longer_a_reader": sorted(set(GATEWAY_ERROR_READERS) - discovered),
        "hint": "a fourth reader must join every reader census below, not just this one",
    }
    assert set(GATEWAY_ERROR_READERS) == set(SILENT_BODY_READERS) == set(GATEWAY_READER_WIRE_KEYS)


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
    # The registered dropped read, stated once per reader rather than inferred.
    assert read.isdisjoint(TRANSPORTABLE_ERROR_UNCARRIED), {
        "hint": "a reader started binding config_recoverable: update EXPECTED and the registry",
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

    preserved = {"provider_error_code", "retryable", "http_status", "provider_retried"}
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
    assert lost == {"config_recoverable", "error_code"}, {
        "lost_across_the_hop": sorted(lost),
        "hint": "a shrinking loss set means a gap was fixed: update EXPECTED and the registry",
    }


def test_2c_only_two_of_five_gateway_validators_forward_the_status_they_know() -> None:
    """Sibling census: the same defect shape as a missing wire key, one call frame in."""

    forwarding = set()
    for name in (
        "_exact_gateway_bool",
        "_gateway_string",
        "_exact_gateway_int",
        "_gateway_fragment_string",
        "_gateway_usage",
        "_portable_gateway_payload",
    ):
        parameters = inspect.signature(getattr(gateway_client, name)).parameters
        if "http_status" in parameters:
            forwarding.add(name)
    assert forwarding == {"_exact_gateway_bool", "_gateway_string"}, {
        "forwarding_http_status": sorted(forwarding),
        "hint": "a validator gained the parameter: update EXPECTED and drop its registry entry",
    }


def test_2c_the_openai_classifier_sets_neither_retry_nor_recoverability_flag() -> None:
    """Every ModelAdapterError branch of the one adapter that could classify them."""

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
    flagged = {
        keyword.arg
        for branch in branches
        for keyword in branch.keywords
        if keyword.arg in {"provider_retried", "config_recoverable"}
    }
    assert flagged == set(), {
        "flags_now_set": sorted(flagged),
        "hint": "the classifier started carrying a flag: drop its registry entry",
    }


# --------------------------------------------------------------------------------------
# Family 3 — usage
# --------------------------------------------------------------------------------------

# reference/llm_gateway/service.py:LlmGatewayUsage — the tenant meter's total emitted domain.
GATEWAY_METER_KEYS = frozenset(
    {"tenant_id", "calls", "input_tokens", "output_tokens", "total_tokens"}
)


def test_3a_tenant_meter_drops_every_priced_sub_count() -> None:
    """Behavioral: ``add`` normalizes seven keys and sums three, which no key diff can show."""

    meter = LlmGatewayUsage(tenant_id="tenant-1")
    meter.add(dict(_MAXIMAL_USAGE))
    reported = meter.to_json()
    assert frozenset(reported) == GATEWAY_METER_KEYS, {
        "missing": sorted(GATEWAY_METER_KEYS - set(reported)),
        "extra": sorted(set(reported) - GATEWAY_METER_KEYS),
    }
    dropped = NORMALIZED_USAGE_KEYS - set(reported)
    assert dropped == {
        "cache_read_tokens",
        "cache_creation_tokens",
        "reasoning_tokens",
        "audio_tokens",
    }, {"dropped_sub_counts": sorted(dropped)}


def test_3a_a_sub_count_only_billed_call_meters_as_zero() -> None:
    """Corollary A8: the meter cannot see a cost expressed only in sub-counts."""

    meter = LlmGatewayUsage(tenant_id="tenant-1")
    meter.add({"cache_read_tokens": 5_000, "reasoning_tokens": 900})
    reported = meter.to_json()
    counted = (reported["input_tokens"], reported["output_tokens"], reported["total_tokens"])
    assert counted == (0, 0, 0)
    assert reported["calls"] == 1


# core/schemas.py:EVENT_DATA_SCHEMAS["metrics.updated"] — reasoning_tokens is the only sub-count.
METRICS_UPDATED_KEYS = frozenset(
    {
        "step",
        "tool_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "reasoning_tokens",
        "web_search_calls",
        "web_fetch_calls",
        "web_context_calls",
        "web_failed_calls",
    }
)


def test_3b_metrics_updated_declares_one_of_the_four_sub_counts() -> None:
    declared = frozenset(EVENT_DATA_SCHEMAS["metrics.updated"]["properties"])
    assert declared == METRICS_UPDATED_KEYS, {
        "missing": sorted(METRICS_UPDATED_KEYS - declared),
        "extra": sorted(declared - METRICS_UPDATED_KEYS),
    }
    assert NORMALIZED_USAGE_KEYS - declared == {
        "cache_read_tokens",
        "cache_creation_tokens",
        "audio_tokens",
    }


# The one usage-bearing emit site the ``data={...}`` census cannot see: it passes a *name*
# (``data=metrics_data``), so the literal lives in an assignment several statements earlier and
# a key added there reaches the wire with no schema entry and no census.
METRICS_UPDATED_UNCONDITIONAL_KEYS = METRICS_UPDATED_KEYS - {"reasoning_tokens"}


def test_3b_the_metrics_emit_site_and_its_schema_agree_key_for_key() -> None:
    pump = _function_node("loop.py", "_apump_turn")
    literals = [
        _dict_keys(node.value)
        for node in ast.walk(pump)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "metrics_data"
        and isinstance(node.value, ast.Dict)
    ]
    assert len(literals) == 1, {"metrics_data_literals": len(literals)}
    conditional = {
        node.targets[0].slice.value
        for node in ast.walk(pump)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Subscript)
        and isinstance(node.targets[0].value, ast.Name)
        and node.targets[0].value.id == "metrics_data"
        and isinstance(node.targets[0].slice, ast.Constant)
    }
    assert literals[0] == METRICS_UPDATED_UNCONDITIONAL_KEYS, {
        "emitted_always": sorted(literals[0]),
        "expected": sorted(METRICS_UPDATED_UNCONDITIONAL_KEYS),
    }
    # ``reasoning_tokens`` is added only when the adapter reported one, which is why it is the
    # single sub-count on this event and why it cannot be censused off the literal alone.
    assert conditional == {"reasoning_tokens"}, {"emitted_conditionally": sorted(conditional)}
    declared = frozenset(EVENT_DATA_SCHEMAS["metrics.updated"]["properties"])
    assert (literals[0] | conditional) == declared, {
        "emitted_not_declared": sorted((literals[0] | conditional) - declared),
        "declared_not_emitted": sorted(declared - (literals[0] | conditional)),
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


def test_3c_subagent_rollup_hard_codes_three_of_the_seven_usage_keys() -> None:
    """The parent budget reads a literal tuple, so a sub-count can never reach it."""

    tree = _module_tree("loop.py")
    child = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_run_subagent_child"
    )
    rolled: list[frozenset[str]] = []
    for node in ast.walk(child):
        if not isinstance(node, ast.Tuple):
            continue
        names = {
            element.value
            for element in node.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        }
        if names and names <= NORMALIZED_USAGE_KEYS:
            rolled.append(frozenset(names))
    assert len(rolled) == 1, {"usage_key_tuples_in_the_rollup": [sorted(item) for item in rolled]}
    assert rolled[0] == {"input_tokens", "output_tokens", "total_tokens"}, {
        "rolled_up": sorted(rolled[0]),
        "hint": "widened? update EXPECTED and drop the registry entry",
    }


class _SubclassedCount(IntEnum):
    """An ``int`` subclass a provider SDK can plausibly hand back as a token count."""

    SEVEN = 7


def test_3d_the_four_readers_of_one_stamp_disagree_about_what_a_count_is() -> None:
    """Same stamp, four predicates, two verdicts — pinned so harmonizing them fails here."""

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
        "model_call.py:_recordable_usage": True,
    }, {
        "verdicts": verdicts,
        "hint": "harmonized? make them agree, update EXPECTED and drop the registry entry",
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

    source = inspect.getsource(_applied_echoes)
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
            "core/result.py",
            "core/schemas.py",
            "errors.py",
            "loop.py",
            "providers/gateway.py",
            "providers/openai.py",
            "reference/llm_gateway/http.py",
        }
    ),
    "provider_retried": frozenset(
        {
            "contracts.py",
            "core/model_io.py",
            "errors.py",
            "model_call.py",
            "observability/otel.py",
            "providers/base.py",
            "providers/gateway.py",
            "reference/llm_gateway/http.py",
            "reference/llm_gateway/service.py",
        }
    ),
    "provider_usage": frozenset(
        {
            "core/model_io.py",
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
    alias_only = {"core/checkpoint.py", "core/spec.py"}
    unaccounted = registered_paths - all_carriers - alias_only
    assert unaccounted == set(), {"registered_but_not_a_scanned_carrier": sorted(unaccounted)}
