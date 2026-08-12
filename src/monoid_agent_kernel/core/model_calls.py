"""What one model call may say about itself once it is written down.

``model_calls.jsonl`` is the private run-directory ledger of settled model calls: one line per
call, successful or not, carrying metadata and the replay key and **no content**. W6-1
(dx-note ``2026-08-02-v0.21-contract-replay-scope.md`` §Track B, decision 1).

This module is pure. It builds a record and nothing else -- no file, no handle, no policy. The
recorder owns the writing, the same way it owns ``model-content.jsonl``, so that a run whose disk
is full loses a line rather than an answer.

**The record is a projection, not a serialized receipt.** ``ModelCallReceipt.to_json()`` exists
and is not used here, because it emits ``model.to_json()`` and ``ModelConfig.to_json()`` emits
``gateway_url``. The endpoint is the one thing the receipt is careful never to state in the
clear: :func:`~monoid_agent_kernel.core.model_io.destination_digest` keys it precisely so that a
record can compare two destinations without becoming a guessing oracle for an internal hostname.
Serializing the receipt would put the preimage of that digest in the adjacent field of the same
line -- and, because the key is per-process, one such line makes every other digest in the file
confirmable. The shipped scaffold configures a ``gateway_url`` (``builder.py``), so this is the
first line of a new agent's first run, not a hypothetical.

Which is why the field list is declared here, once, and why ``destination_digest`` is absent from
it entirely: a per-process HMAC written to a file that survives a restart splits within itself,
naming one destination with two digests and reading as a deployment change that never happened.
``destination_status`` carries the diagnosis, which is the part an auditor actually asks for.

Deliberately **not** shared with ``providers._request_identity._model_identity``. Both are hand-listed projections
of ``ModelConfig`` and they answer different questions -- that one says what the provider was
asked for, this one says what the call was. Sharing the function would mean that adding a field
for the ledger's sake rekeys every recorded replay key, which is the exact defect W6-0 closed.
"""

from __future__ import annotations

from typing import Any

from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.model_io import (
    ModelCallAttempt,
    ModelCallReceipt,
    is_absent_or_valid,
    is_recorded_digest,
    is_valid_idempotency_key,
)
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.identifiers import namespaced_id

MODEL_CALLS_SCHEMA_VERSION = namespaced_id("model-calls.v1")
MODEL_CALLS_FILENAME = "model_calls.jsonl"

# One record shape today, discriminated anyway: a discriminator retrofitted onto records that
# already exist cannot be applied to them (``MODEL_CONTENT_RECORD_SCHEMA`` avoids the same
# problem by construction). W6-2's payload records went to their own artifact rather than this
# file -- this ledger promises "no content" and is keyed as a sequence, while the corpus is
# content-classified and keyed largely as a set -- but the discriminator stays, because the next
# record shape that DOES belong in a metadata ledger will need it just as much.
MODEL_CALL_KIND = "model_call"


def _recorded_model(model: ModelConfig) -> dict[str, Any]:
    """The model config as the ledger records it: a declared list, never a serialized object.

    Excludes ``gateway_url`` (the module docstring's whole subject), and ``timeout_s`` / ``retry``
    because they are transport policy: they say how the call was carried, not what was asked for,
    and the gateway wire already omits them so each hop owns its own. Recording them would also
    put a second copy of an operational knob in an artifact an operator reads for identity.

    Includes ``provider``, which the replay key's twin projection deliberately leaves out -- the
    key carries a *resolved* provider as a sibling term, while a record wants the configured value
    too, because ``receipt.provider_name or receipt.model.provider`` is how every other reader of
    a receipt spells "who served this" (see ``observability/otel.py``).

    Hand-listed all the way down, not just at the top: calling ``reasoning.to_json()`` here would
    move the same hazard one level deeper, where the next added field would find it.

    Known limit, inherited rather than introduced: for an adapter exposing no ``config``,
    ``_effective_model`` substitutes ``ModelConfig()``, so a call that ran under something else is
    recorded under the default model name. The receipt cannot say "unknown". That was a live hint
    before and is a durable audit claim now, which is a reason to state it, not to fix it here.
    """

    recorded: dict[str, Any] = {
        "provider": model.provider,
        "model": model.model,
        "reasoning": {
            "effort": model.reasoning.effort,
            "summary": model.reasoning.summary,
            "on_unsupported": model.reasoning.on_unsupported,
        },
    }
    # Omit-when-absent, the same rule the replay key and the runtime-config hash both hold: a
    # config that never set a sampling control records what it recorded before the block existed.
    if not model.generation.is_default:
        recorded["generation"] = {
            "temperature": model.generation.temperature,
            "top_p": model.generation.top_p,
            "max_output_tokens": model.generation.max_output_tokens,
            "on_unsupported": model.generation.on_unsupported,
        }
    return recorded


def _recorded_context(context: InvocationContext) -> dict[str, Any]:
    """The caller's provenance, hand-listed for the reason the model block is.

    ``attributes`` is carried verbatim. It is an open caller-supplied map, but the kernel is one
    of those callers: ``AgentLoop`` writes ``root_run_id`` / ``parent_run_id`` / ``parent_task_id``
    / ``subagent_definition_id`` / ``subagent_depth`` there for every subagent call, so dropping it
    would lose subagent lineage from the artifact whose job is lineage. It is already constrained
    to ``str -> str`` at construction and re-normalized before a receipt carries it.

    Unbounded in size, and knowingly: a caller with a large map pays for it on every line. A cap
    invented here would silently truncate provenance, and the honest place for one is the ingress
    that accepts the map, not the record that reports it.
    """

    return {
        "run_id": context.run_id,
        "skill_id": context.skill_id,
        "skill_digest": context.skill_digest,
        "step_id": context.step_id,
        "attempt": context.attempt,
        "batch_id": context.batch_id,
        "item_id": context.item_id,
        "case_id": context.case_id,
        "traceparent": context.traceparent,
        "tracestate": context.tracestate,
        "attributes": dict(context.attributes),
    }


def _recorded_attempt(entry: ModelCallAttempt) -> dict[str, Any]:
    """One dispatch, hand-listed like every projection in this module.

    Not ``entry.to_json()``, although today the two agree key for key: a field added to
    ``ModelCallAttempt`` must be added HERE by name to reach the artifact, or every field of
    the source type becomes an author of the audit record -- the exact rule the reflection
    census on these projections enforces.

    ``backoff_ms`` is conditional, matching the entry's own wire rule: ``None`` means the entry
    was parsed from a line that predates the field, and absence is that fact's only honest
    spelling -- null is a value no writer ever wrote, and 0 is a measurement never taken.
    """

    recorded: dict[str, Any] = {
        "index": entry.index,
        "elapsed_ms": entry.elapsed_ms,
        "error_code": entry.error_code,
        "provider_error_code": entry.provider_error_code,
        "retryable": entry.retryable,
        "config_recoverable": entry.config_recoverable,
        "http_status": entry.http_status,
        "provider_retried": entry.provider_retried,
        "usage": dict(entry.usage),
        "stream_committed": entry.stream_committed,
    }
    if entry.backoff_ms is not None:
        recorded["backoff_ms"] = entry.backoff_ms
    return recorded


def model_call_record(
    receipt: ModelCallReceipt,
    *,
    run_id: str,
    root_run_id: str,
    call_index: int,
    recorded_at: str,
) -> dict[str, Any]:
    """One ledger line for one settled call.

    The envelope is the writer's, not the receipt's, because the receipt cannot prove any of it.
    ``run_id`` is the recorder's -- the directory the line is written into proves it, while
    ``context.run_id`` is a caller's claim and a caller may say anything. ``recorded_at`` is the
    wall clock the receipt does not carry (it has ``latency_ms`` and no instant). ``call_index``
    is the only way an append-only best-effort file can reveal its own dropped lines; it restarts
    at zero when a durable run reopens its directory, which is self-evident in a way a restarted
    *digest* would not have been.

    There is deliberately no ``turn_id`` or ``step``. ``AgentLoop`` puts its turn id in
    ``context.step_id``, but that is a convention of one caller: a standalone
    ``ValidatedCallRunner`` or a direct integrator means their own unit of work by that field, and
    promoting it to a named join key here would fabricate a relationship for them.

    Two receipt fields are also absent, for reasons that outlive this function:

    ``destination_digest`` -- keyed under a per-process secret. Live consumers can compare it;
    a file cannot, because the file outlives the process that keyed it. ``destination_status``
    says which of the four probe outcomes happened, which is the part that stays true.

    ``redaction_digest`` -- set only by the per-subscription narrowing, never on the receipt this
    seam is handed. Recording it would write "no redaction rules were applied" on every line,
    including the lines for calls where a redacted consumer applied rules. A constant that reads
    as a fact is worse than an absent key.

    The derived properties (``succeeded``, ``trace_id``, ``span_id``) are absent too: they are
    computed from ``error_code`` and ``traceparent``, both of which are here, and a recorded
    derivation is a value that can come to disagree with its source.

    ``idempotency_key`` IS here, and its meaning is fixed the same way ``redaction_digest``'s
    absence is: the recorded key says the call was *keyed* -- the runner issues one for every
    call that reaches the keying block -- not that any transport presented it. Only the gateway
    adapter puts it on the wire, so a key on a fake or replay call's line is a fact about
    issuance, never evidence a request was sent.
    """

    # The mint guard (W7-4): this is the single place a receipt becomes an artifact line, and
    # these three are the format-constrained fields ``from_json`` deliberately transports
    # without judging -- reader-lenient so a damaged receipt can be loaded and inspected,
    # schema-strict so a damaged LINE cannot be certified. The guard closes the route between
    # the two: a foreign receipt that parsed fine must not mint a line ``monoid validate``
    # then refuses. Empty stays admissible on all three -- a refused call was never keyed and
    # never digested, and a status field explains each -- so this cannot fire on a receipt
    # the runner built, which is valid by construction on every settle path. That empty arm
    # is asked through ``is_absent_or_valid`` rather than spelled here: ``value != ""`` is a
    # question the value answers, and a ``str`` subclass answering it falsely walked past
    # this guard with its underlying string intact. Request ingress had the same shape and
    # the same hole; one body serves both now. For the recorder, a raise here costs the one
    # line, not the run (its hostile-context containment); the offending value stays out of
    # the message, the same transport rule the key's logging sinks follow.
    for field_name, value, is_valid in (
        ("idempotency_key", receipt.idempotency_key, is_valid_idempotency_key),
        ("prompt_digest", receipt.prompt_digest, is_recorded_digest),
        ("request_digest", receipt.request_digest, is_recorded_digest),
    ):
        if not is_absent_or_valid(value, is_valid):
            raise ValueError(
                f"model call record {field_name} must be empty or the shape "
                "the ledger schema certifies"
            )

    record: dict[str, Any] = {
        "schema_version": MODEL_CALLS_SCHEMA_VERSION,
        "kind": MODEL_CALL_KIND,
        "run_id": run_id,
        "root_run_id": root_run_id,
        "call_index": call_index,
        "recorded_at": recorded_at,
        "context": _recorded_context(receipt.context),
        "model": _recorded_model(receipt.model),
        "provider_name": receipt.provider_name,
        "prompt_digest": receipt.prompt_digest,
        "request_digest": receipt.request_digest,
        "digest_generation": receipt.digest_generation,
        "digest_status": receipt.digest_status,
        "idempotency_key": receipt.idempotency_key,
        "destination_status": receipt.destination_status,
        "stop_reason": receipt.stop_reason,
        "usage": dict(receipt.usage),
        "latency_ms": receipt.latency_ms,
        "attempts": receipt.attempts,
        "provider_retried": receipt.provider_retried,
        "error_code": receipt.error_code,
        "provider_error_code": receipt.provider_error_code,
        "retryable": receipt.retryable,
        "config_recoverable": receipt.config_recoverable,
        "http_status": receipt.http_status,
        "capture_downgrades": receipt.capture_downgrades,
    }
    # One object per dispatch, hand-projected in ``_recorded_attempt`` -- and the key emitted
    # only when there is a dispatch to itemize, the receipt's own wire rule. Absence on a line
    # means nothing was itemized: a writer that predates the field, a refused call that never
    # dispatched, or a receipt built without a log at any count. The schema declares the key
    # and does not require it for those reasons, and the sweep relates only a NON-EMPTY log to
    # the line around it: a present ``[]`` is what every build before this one wrote for the
    # same value -- this projection emitted the key unconditionally -- so ``validate_run_dir``
    # keeps certifying the directories they filled.
    if receipt.attempt_log:
        record["attempt_log"] = [_recorded_attempt(entry) for entry in receipt.attempt_log]
    return record
