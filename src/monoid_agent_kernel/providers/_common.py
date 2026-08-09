"""Shared helpers for model adapters.

The OpenAI and gateway adapters build the same reasoning block and normalize
usage the same way; only their tool-schema shape genuinely differs. Keep the
common pieces here so the two adapters cannot drift.
"""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from monoid_agent_kernel.core.spec import GenerationConfig, ReasoningConfig

# Half of ``log(float_max)``: the per-chunk log budget :func:`capped_backoff` sizes its power
# by. Half, not all of it, so the chunk stays safe through the rounding of the division that
# sizes it -- the power lands at or under ``sqrt(float_max)`` -- at the cost of at most doubling
# a loop that runs four times in the worst case anyone can configure.
_CHUNK_LOG_BUDGET = math.log(sys.float_info.max) / 2.0


def retry_delay_s(
    attempt: int,
    initial_delay_s: float,
    max_delay_s: float,
    backoff_multiplier: float,
    jitter_s: float,
) -> float:
    """How long to wait after ``attempt`` failed, before the next one.

    The schedule itself, separated from waiting on it: the gateway's two loops and the
    kernel runner's loop all wait differently, and a backoff policy that differed between
    them would be a difference nobody chose. Moved here from ``gateway`` when the kernel
    layer became the third caller; ``gateway._retry_delay`` remains as an import alias
    because that module's tests pin and monkeypatch the schedule through the old name.

    Every input is validated finite and every answer is too. Jitter rides ON TOP of the cap
    -- deliberately, since the moment a herd most needs smearing is the moment every member
    is sitting at ``max_delay_s`` -- but ``max_delay_s`` and ``jitter_s`` are each bounded
    below and not above, and float addition returns ``inf`` rather than raising when their
    sum leaves the range. A non-finite wait is not a long wait: ``asyncio.sleep(inf)`` is a
    timer that never fires, so the loop stops being a loop. Hence the saturating add.
    """

    delay = capped_backoff(attempt, initial_delay_s, max_delay_s, backoff_multiplier)
    if jitter_s > 0:
        delay = min(delay + random.uniform(0, jitter_s), sys.float_info.max)
    return delay


def capped_backoff(
    attempt: int,
    initial_delay_s: float,
    max_delay_s: float,
    backoff_multiplier: float,
) -> float:
    """``initial_delay_s * backoff_multiplier ** (attempt - 1)``, never above ``max_delay_s``.

    ``max_delay_s`` caps the arithmetic, not only its result. Capping only the result --
    ``min(max_delay_s, initial * multiplier ** (attempt - 1))`` -- lets the power leave the float
    range before the cap is ever consulted, and ``float ** int`` raises ``OverflowError`` there
    rather than saturating at infinity. A policy the spec ACCEPTS reaches that: ``max_attempts``
    is validated as an integer above zero and ``backoff_multiplier`` as any positive finite
    number, neither with an upper bound, so ``1e308`` overflows on the third attempt and the
    default ``2.0`` on the 1025th -- while the configured cap still says four seconds. The
    reference backend's outbox knobs (``outbox_max_attempts``, ``outbox_retry_factor``) are
    plain operator-settable fields with no validation at all, so they reach it more cheaply.

    Every caller evaluates the schedule inside a failure handler -- the three model-retry loops
    around a retryable ``ModelAdapterError``, the reference backend's turn-retry handler, and its
    outbox dispatcher scheduling around a failed send. An arithmetic error raised there does not
    merely add noise: it REPLACES the failure being reported, so the layer above gets an
    unclassified ``OverflowError`` instead of the taxonomy (``retryable``, ``code``,
    ``http_status``) it retries, reports and stamps receipts on. So the exponent is bounded
    before the power -- which is also why this is one function and not a shape each loop
    re-derives. It is public for exactly that reason: the outbox applies FULL jitter
    (``uniform(0, ceiling)``) rather than jitter on top, so it needs the bounded ceiling itself
    rather than :func:`retry_delay_s`.

    One bound decides, and it is the exact one: at ``saturating`` the product has already reached
    the cap, so the answer IS the cap and no larger exponent can change it. Below that point the
    product is representable even where the power on its own is not -- a subnormal
    ``initial_delay_s`` under a large multiplier -- so the power is taken in chunks small enough
    not to overflow and folded into the running delay as it goes. A ceiling on the power alone
    would answer ``max_delay_s`` for those, turning a 1.8-second wait into a 1.8e308-second one.
    """

    exponent = max(0, attempt - 1)
    # Nothing to grow (the first attempt), or nothing a growth could change: a zero initial
    # delay keeps the product at zero and a zero cap keeps the answer there. Settled here so
    # the logarithms below only ever see a positive finite input.
    if exponent == 0 or initial_delay_s <= 0.0 or max_delay_s <= 0.0:
        return min(max_delay_s, initial_delay_s)
    # A multiplier that does not grow drives the power toward zero, never out of range.
    if backoff_multiplier <= 1.0:
        return min(max_delay_s, initial_delay_s * backoff_multiplier**exponent)
    growth_per_step = math.log(backoff_multiplier)
    # ``int >= float`` compares exactly in Python, so ``attempt`` is never itself converted --
    # the guard against an oversized exponent cannot be defeated by the exponent's own size.
    if exponent >= (math.log(max_delay_s) - math.log(initial_delay_s)) / growth_per_step:
        return max_delay_s
    # One chunk covers every schedule a real policy writes, and then this is the arithmetic it
    # always was. Growth is monotone, so an intermediate that reaches the cap settles the answer
    # for every step still owed -- which is also what keeps a chunk product that saturates to
    # infinity (multiplication, so no ``OverflowError``) from escaping as one.
    chunk = max(1, int(_CHUNK_LOG_BUDGET / growth_per_step))
    delay = initial_delay_s
    remaining = exponent
    while remaining > 0:
        taken = min(remaining, chunk)
        delay *= backoff_multiplier**taken
        if delay >= max_delay_s:
            return max_delay_s
        remaining -= taken
    return min(max_delay_s, delay)


def build_reasoning_payload(reasoning: ReasoningConfig) -> dict[str, Any]:
    """Reasoning block for a model request: ``{}`` when default/off, else effort/summary.

    Like its generation sibling below, this projection is also the gateway server's
    ``reasoning_applied`` echo and the client checker's expected value, so the two sides of
    that wire agree on the *shape* of "applied" by construction. Note the default config is
    NOT the empty block here (``ReasoningConfig()`` projects ``{"effort": "medium"}``), which
    is why "did the request use the feature" is answered by ``ReasoningConfig.is_default``
    rather than by this payload's truthiness.
    """
    payload: dict[str, Any] = {}
    if reasoning.effort != "default":
        payload["effort"] = reasoning.effort
    if reasoning.summary != "off":
        payload["summary"] = reasoning.summary
    return payload


def build_generation_payload(generation: GenerationConfig) -> dict[str, Any]:
    """Sampling block for a model request: ``{}`` when nothing is set, else only the set keys.

    ``on_unsupported`` never rides here -- it is the caller's policy, not a provider knob. The
    gateway server's ``generation_applied`` echo is this same block, so the two sides of that
    wire agree on the *shape* of "applied" by construction. Whether to emit it at all is a
    separate question the server answers from its upstream adapter's ``generation_support``
    declaration -- this builder cannot know what an adapter does with the config it is handed.
    """
    payload: dict[str, Any] = {}
    if generation.temperature is not None:
        payload["temperature"] = generation.temperature
    if generation.top_p is not None:
        payload["top_p"] = generation.top_p
    if generation.max_output_tokens is not None:
        payload["max_output_tokens"] = generation.max_output_tokens
    return payload


def reasoning_replay_window_start(messages: Sequence[Mapping[str, Any]]) -> int:
    """Index of the first message in the ACTIVE REPLAY WINDOW: one past the last ``user`` entry.

    Captured provider reasoning can only be replayed while it sits inside this window — the
    in-flight tool loop. Once a new user message lands, every earlier block is outside the
    window, and it stays outside forever because the window only ever moves forward.

    Two halves depend on that one fact and they must not drift: the OpenAI adapter decides what
    to REPLAY out of the window (``_reasoning_replay_flags``), and the kernel decides what is
    still worth SENDING into it (:func:`prune_dead_reasoning`). A log with no user message at
    all is entirely window (start ``0``), which is what the replay rule always did.
    """
    start = 0
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            start = index + 1
    return start


def prune_dead_reasoning(messages: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Drop the ``reasoning`` key from every message BEFORE the active replay window.

    Outside the window the block is unreachable: the adapter reconstructs those turns from
    ``content``/``tool_calls`` and never reads it. Sending it anyway is pure cost, and cost that
    grows with the conversation — one dead block per user turn, re-sent on every later request,
    counted against the wire-byte cap and the receiving server's body limit.

    This builds the ephemeral wire copy; the caller's messages are never mutated and the durable
    log keeps every block verbatim (see ``docs/CONTRACTS.md``). Messages that keep their block
    are passed through by identity, so the copy is cheap on the common short conversation.
    """
    start = reasoning_replay_window_start(messages)
    pruned: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if index < start and "reasoning" in message:
            pruned.append({key: value for key, value in message.items() if key != "reasoning"})
        else:
            pruned.append(message)
    return tuple(pruned)


def text_from_message_content(content: Any) -> str:
    """Project a by-value message ``content`` down to plain text for text-only adapters.

    ``content`` is either a ``str`` (already text) or a list of part-dicts (the multimodal
    by-reference shape produced by ``content_part_to_json``). Non-text parts (image,
    document) are dropped here — a text-only wire keeps working even once the durable log
    carries multimodal parts. Multimodal adapters bypass this and map the parts instead.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        segments = [
            str(part.get("text", "")).strip()
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "text"
            and str(part.get("text", "")).strip()
        ]
        return "\n\n".join(segments)
    return ""


def project_message_to_text(message: dict[str, Any]) -> dict[str, Any]:
    """Return ``message`` with list ``content`` collapsed to text; pass ``str`` through.

    Used by text-only adapter send paths so a durable multimodal message (list content)
    never reaches a provider that cannot read it.
    """
    content = message.get("content")
    if isinstance(content, list):
        return {**message, "content": text_from_message_content(content)}
    return message


# Every key :func:`normalize_usage` below can emit, and therefore the whole vocabulary of a
# normalized usage mapping. Declared here, beside the function that is its authority, so a
# consumer that must FILTER a wider mapping down to usage (the subagent roll-up folds a child's
# whole metrics dict into the parent's totals, where a stray ``duration_s`` would corrupt them)
# names the domain rather than hand-copying a subset of it. Kept in step with the function by
# ``tests/test_carriage_conformance.py``, which reads the keys the live callable can assign.
NORMALIZED_USAGE_KEYS: frozenset[str] = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
        "reasoning_tokens",
        "audio_tokens",
    }
)


def usage_reported_by(payload: Any) -> dict[str, int]:
    """Tokens a call that ends in a REFUSAL reported spending, read leniently.

    The stamp's source on every refusal path an adapter has. A payload the reader rejects for a
    malformed key was generated and BILLED before the reader ever looked at it, so the refusal is
    the only carrier left for its cost -- and without this the receipt, the run's token budget
    and (across a hop) the tenant ledger all record zero for a turn the provider charged for.

    Lenient on purpose, and that is the whole reason it is not :func:`normalize_usage`. This runs
    on a failure path, so a second malformation must not replace the failure being reported with
    a different one: anything unreadable simply reads as "not reported", including a ``usage``
    that is itself the malformed key. Values are judged, names are not -- an unknown counter a
    newer gateway reports rides through rather than being silently dropped.

    One function for both adapters. It began as the gateway client's ``_reported_error_usage``
    and the OpenAI adapter had no equivalent at all, which is exactly how the *source* reader --
    the one that sees the provider's own billed body first -- came to refuse it for free. Two
    copies of "what counts as a reported cost" is two copies that can disagree, and the census
    holds this one to the same verdict as the four other readers of the same stamp.
    """

    # The outer guard is the one the shared version needed: the gateway client only ever handed
    # this a decoded JSON object, while the OpenAI adapter hands it whatever the SDK returned, and
    # an ``AttributeError`` raised *inside* an except-handler would replace the failure being
    # reported -- the exact thing this function exists to avoid. The ``usage`` test stays ``dict``,
    # byte-identical to the verdict the gateway wire has always given.
    if not isinstance(payload, Mapping):
        return {}
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {
        str(key): value
        for key, value in usage.items()
        if type(value) is int and value >= 0
    }


def _usage_object(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"model usage {field_name} must be an object")
    return value


def _usage_count(payload: dict[str, Any], key: str) -> int | None:
    if key not in payload:
        return None
    value = payload[key]
    if type(value) is not int or value < 0:
        raise ValueError(f"model usage {key} must be a non-negative integer")
    return value


def _first_count(*candidates: tuple[dict[str, Any], str]) -> int | None:
    for payload, key in candidates:
        value = _usage_count(payload, key)
        if value is not None:
            return value
    return None


def normalize_usage(
    usage: dict[str, Any] | None, *, legacy_aliases: bool = False
) -> dict[str, int]:
    """Validate provider usage as ``{input_tokens, output_tokens, total_tokens}`` plus
    optional priced sub-counts (``cache_read_tokens``, ``cache_creation_tokens``,
    ``reasoning_tokens``, ``audio_tokens``) included **only when present**.

    The sub-counts are folded from the various provider shapes — Anthropic's flat
    ``cache_*_input_tokens``, OpenAI's nested ``*_tokens_details``, Gemini's
    ``cachedContentTokenCount`` / ``thoughtsTokenCount``, and an already-normalized
    passthrough — so cache and reasoning tokens (priced differently) survive instead of
    being flattened away. ``legacy_aliases`` also accepts OpenAI's older ``prompt_tokens`` /
    ``completion_tokens`` names as fallbacks.
    """
    usage = _usage_object(usage, "payload")
    input_candidates = [(usage, "input_tokens")]
    output_candidates = [(usage, "output_tokens")]
    if legacy_aliases:
        input_candidates.append((usage, "prompt_tokens"))
        output_candidates.append((usage, "completion_tokens"))
    input_tokens = _first_count(*input_candidates)
    output_tokens = _first_count(*output_candidates)
    normalized_input = input_tokens if input_tokens is not None else 0
    normalized_output = output_tokens if output_tokens is not None else 0
    reported_total = _usage_count(usage, "total_tokens")
    normalized = {
        "input_tokens": normalized_input,
        "output_tokens": normalized_output,
        "total_tokens": reported_total if reported_total is not None else 0,
    }
    input_details = _usage_object(
        usage.get("input_tokens_details", usage.get("prompt_tokens_details")),
        "input_tokens_details",
    )
    output_details = _usage_object(
        usage.get("output_tokens_details", usage.get("completion_tokens_details")),
        "output_tokens_details",
    )
    input_audio = _usage_count(input_details, "audio_tokens") or 0
    output_audio = _usage_count(output_details, "audio_tokens") or 0
    audio = input_audio + output_audio
    details = {
        "cache_read_tokens": _first_count(
            (usage, "cache_read_tokens"),
            (usage, "cache_read_input_tokens"),
            (input_details, "cached_tokens"),
            (usage, "cached_tokens"),
            (usage, "cachedContentTokenCount"),
        ),
        "cache_creation_tokens": _first_count(
            (usage, "cache_creation_tokens"),
            (usage, "cache_creation_input_tokens"),
        ),
        "reasoning_tokens": _first_count(
            (usage, "reasoning_tokens"),
            (output_details, "reasoning_tokens"),
            (output_details, "thinking_tokens"),
            (usage, "thoughtsTokenCount"),
        ),
        "audio_tokens": _first_count((usage, "audio_tokens")),
    }
    if details["audio_tokens"] is None and audio:
        details["audio_tokens"] = audio
    for key, value in details.items():
        if value:
            normalized[key] = value
    return normalized
