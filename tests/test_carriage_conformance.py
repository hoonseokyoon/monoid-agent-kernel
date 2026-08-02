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
import inspect
import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import pytest

from monoid_agent_kernel.core.result import (
    AgentTurnResult,
    Suspension,
    suspension_checkpoint_payload,
    suspension_from_checkpoint_payload,
)
from monoid_agent_kernel.core.schemas import EVENT_DATA_SCHEMAS, TRANSCRIPT_RECORD_SCHEMA
from monoid_agent_kernel.core.spec import GenerationConfig, ModelConfig, ReasoningConfig
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.model_call import _recordable_usage
from monoid_agent_kernel.providers import gateway as gateway_client
from monoid_agent_kernel.providers._common import normalize_usage
from monoid_agent_kernel.providers.base import mark_provider_usage, provider_usage_of
from monoid_agent_kernel.reference.llm_gateway.http import (
    _error_body,
    _model_error_status,
    _stream_error_frame,
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


def _module_tree(relative_path: str) -> ast.Module:
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


# --------------------------------------------------------------------------------------
# Family 2 — ModelAdapterError transport
# --------------------------------------------------------------------------------------

SERVER_ERROR_BODY_KEYS = frozenset(TRANSPORTABLE_ERROR_WIRE_ALIASES)


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


def test_2a_server_error_body_writes_exactly_the_alias_table() -> None:
    body = _maximal_error_body()
    assert frozenset(body) == SERVER_ERROR_BODY_KEYS, {
        "missing": sorted(SERVER_ERROR_BODY_KEYS - set(body)),
        "extra": sorted(set(body) - SERVER_ERROR_BODY_KEYS),
    }
    # The one transportable fact with no key (registered burn-down).
    assert TRANSPORTABLE_ERROR_UNCARRIED.isdisjoint(body)


def test_2a_stream_error_frame_is_the_body_plus_a_type_tag() -> None:
    """The twin writer: separate code, so a field added to one must be added to the other."""

    class _Handler:
        """A ``_stream_error_frame`` handler is only touched on the non-ModelAdapterError path."""

    frame = _stream_error_frame(_Handler(), _maximal_adapter_error())
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


# --------------------------------------------------------------------------------------
# New-carrier backstop
# --------------------------------------------------------------------------------------

# Every module that mentions each headline field today.  Containment, not AST-exactness: the
# point is to notice a NEW carrier file, and a file that merely names the field is a file the
# census has to account for.  Paths are relative to ``src/monoid_agent_kernel``.
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
            "validated_call.py",
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
}


def _python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


@pytest.mark.parametrize("field_name", sorted(CARRIER_FILES))
def test_no_unregistered_carrier_file_appears_for_a_headline_field(field_name: str) -> None:
    found = {
        path.relative_to(PACKAGE).as_posix()
        for path in _python_files(PACKAGE)
        if field_name in path.read_text(encoding="utf-8")
    }
    expected = CARRIER_FILES[field_name]
    assert found == expected, {
        "new carrier file for": field_name,
        "detail": "new carrier file for "
        f"{field_name} — extend the census or register the site",
        "added": sorted(found - expected),
        "removed": sorted(expected - found),
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
