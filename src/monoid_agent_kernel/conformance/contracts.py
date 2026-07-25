"""Reusable implementation contracts for checkpoint stores and capability brokers."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from monoid_agent_kernel.conformance.report import (
    ConformanceRuleOutcome,
    observation,
    outcome_from_observations,
    safe_exception_summary,
)
from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.core.capability import (
    CapabilityBroker,
    CapabilityDenial,
    CapabilityLease,
    CapabilityPending,
    CapabilityRequest,
    scope_within,
)
from monoid_agent_kernel.core.checkpoint import CheckpointStore, RunCheckpoint, load_latest_checked
from monoid_agent_kernel.core.model_io import (
    CapturePolicy,
    ModelCallCapture,
    ModelCallReceipt,
    ModelIOObserver,
    ModelIOSubscription,
    RedactionPolicy,
    Redactor,
    close_model_io_subscriptions,
    dispatch_model_call,
    redacted_or_none,
)

STORE_CONTRACT_PROFILE = "checkpoint-store-contract"
BROKER_CONTRACT_PROFILE = "capability-broker-contract"
REDACTOR_CONTRACT_PROFILE = "redactor-contract"
MODEL_IO_CONTRACT_PROFILE = "model-io-observer-contract"


class CheckpointStoreFactory(Protocol):
    def __call__(self, root: Path) -> CheckpointStore: ...


class CapabilityBrokerFactory(Protocol):
    def __call__(self) -> CapabilityBroker: ...


class RedactorFactory(Protocol):
    def __call__(self) -> Redactor: ...


class ModelIOObserverFactory(Protocol):
    def __call__(self) -> ModelIOObserver: ...


@contextmanager
def _opened_checkpoint_store(
    factory: CheckpointStoreFactory,
    root: Path,
) -> Iterator[CheckpointStore]:
    store = factory(root)
    try:
        yield store
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def run_checkpoint_store_contract(
    factory: CheckpointStoreFactory,
    root: Path,
) -> tuple[ConformanceRuleOutcome, ...]:
    """Execute backend-neutral checkpoint invariants without depending on pytest."""

    outcomes: list[ConformanceRuleOutcome] = []
    try:
        with _opened_checkpoint_store(factory, root) as store:
            missing = load_latest_checked(store, "contract_monotonic")
            store.put(RunCheckpoint(run_id="contract_monotonic", seq=2, final_text="new"))
            store.put(RunCheckpoint(run_id="contract_monotonic", seq=1, final_text="stale"))
        with _opened_checkpoint_store(factory, root) as reopened:
            latest = reopened.latest("contract_monotonic")
        outcomes.append(
            outcome_from_observations(
                "STORE-01-MONOTONIC-PUBLICATION",
                STORE_CONTRACT_PROFILE,
                (
                    observation("initial_missing", expected="missing", actual=missing.status),
                    observation(
                        "reopened_latest_sequence",
                        expected=2,
                        actual=latest.seq if latest else None,
                    ),
                    observation(
                        "reopened_latest_payload",
                        expected="new",
                        actual=latest.checkpoint.final_text if latest else None,
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("STORE-01-MONOTONIC-PUBLICATION", STORE_CONTRACT_PROFILE, exc))
    try:
        data = b"conformance-blob"
        with _opened_checkpoint_store(factory, root) as store:
            digest = store.put_blob("contract_blob", data)
        with _opened_checkpoint_store(factory, root) as reopened:
            reopened_blob = reopened.get_blob("contract_blob", digest)
        outcomes.append(
            outcome_from_observations(
                "STORE-02-CONTENT-ADDRESSED-BLOB",
                STORE_CONTRACT_PROFILE,
                (
                    observation(
                        "digest",
                        expected=hashlib.sha256(data).hexdigest(),
                        actual=digest,
                    ),
                    observation(
                        "reopened_round_trip",
                        expected=data.hex(),
                        actual=reopened_blob.hex(),
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("STORE-02-CONTENT-ADDRESSED-BLOB", STORE_CONTRACT_PROFILE, exc))
    try:
        with _opened_checkpoint_store(factory, root) as store:
            store.put(RunCheckpoint(run_id="contract_deleted", seq=1))
            store.put(RunCheckpoint(run_id="contract_isolated", seq=1))
        with _opened_checkpoint_store(factory, root) as reopened:
            reopened_before_delete = (
                reopened.latest("contract_deleted") is not None
                and reopened.latest("contract_isolated") is not None
            )
            reopened.delete("contract_deleted")
        with _opened_checkpoint_store(factory, root) as reopened_after_delete:
            deleted_missing = reopened_after_delete.latest("contract_deleted") is None
            other_present = reopened_after_delete.latest("contract_isolated") is not None
        outcomes.append(
            outcome_from_observations(
                "STORE-03-RUN-ISOLATION",
                STORE_CONTRACT_PROFILE,
                (
                    observation(
                        "runs_survive_reopen_before_delete",
                        expected=True,
                        actual=reopened_before_delete,
                    ),
                    observation(
                        "deleted_run_missing_after_reopen",
                        expected=True,
                        actual=deleted_missing,
                    ),
                    observation(
                        "other_run_present_after_reopen",
                        expected=True,
                        actual=other_present,
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("STORE-03-RUN-ISOLATION", STORE_CONTRACT_PROFILE, exc))
    return tuple(outcomes)


def run_capability_broker_contract(
    factory: CapabilityBrokerFactory,
) -> tuple[ConformanceRuleOutcome, ...]:
    """Execute the broker outcome and least-privilege contract without pytest."""

    request = CapabilityRequest(
        capability="web.search",
        scope={"allowed_domains": ["a.example"]},
        run_id="contract_run",
        ttl_seconds=300,
    )
    try:
        outcome = factory().request(request)
    except Exception as exc:
        return (_error("BROKER-01-OUTCOME", BROKER_CONTRACT_PROFILE, exc),)
    valid_outcome = isinstance(outcome, (CapabilityLease, CapabilityDenial, CapabilityPending))
    if isinstance(outcome, CapabilityPending):
        named_capability: object = outcome.request.capability
    elif isinstance(outcome, (CapabilityLease, CapabilityDenial)):
        named_capability = outcome.capability
    else:
        named_capability = None
    outcomes = [
        outcome_from_observations(
            "BROKER-01-OUTCOME",
            BROKER_CONTRACT_PROFILE,
            (
                observation(
                    "grant_union",
                    expected=True,
                    actual=valid_outcome,
                ),
                observation(
                    "capability_identity",
                    expected=request.capability,
                    actual=named_capability,
                ),
            ),
        )
    ]
    if not valid_outcome:
        outcomes.append(
            ConformanceRuleOutcome(
                rule_id="BROKER-02-LEASE-LEAST-PRIVILEGE",
                profile_id=BROKER_CONTRACT_PROFILE,
                status="skipped",
                error="broker returned an invalid outcome",
            )
        )
        return tuple(outcomes)
    if isinstance(outcome, CapabilityLease):
        now = time.time()
        outcomes.append(
            outcome_from_observations(
                "BROKER-02-LEASE-LEAST-PRIVILEGE",
                BROKER_CONTRACT_PROFILE,
                (
                    observation(
                        "scope_narrowing",
                        expected=True,
                        actual=scope_within(outcome.scope, request.scope),
                    ),
                    observation("future_expiry", expected=True, actual=outcome.expires_at > now),
                    observation("token_handle", expected=True, actual=bool(outcome.token_ref)),
                ),
            )
        )
    else:
        outcomes.append(
            ConformanceRuleOutcome(
                rule_id="BROKER-02-LEASE-LEAST-PRIVILEGE",
                profile_id=BROKER_CONTRACT_PROFILE,
                status="skipped",
                error="broker policy did not grant this request",
            )
        )
    return tuple(outcomes)


def run_redactor_contract(factory: RedactorFactory) -> tuple[ConformanceRuleOutcome, ...]:
    """Execute the obligations a `Redactor` owes the model-I/O capture pipeline.

    A redactor is the one place an integrator can silently turn "redacted" into "disclosed", so the
    rules check the two properties the pipeline actually depends on — a stable result, and no
    survival of a value the policy named a secret — plus the caller-side guarantee that a redactor
    which fails produces nothing rather than raw content.
    """

    outcomes: list[ConformanceRuleOutcome] = []
    policy = RedactionPolicy(patterns=(r"sk-[A-Za-z0-9]+",), literals=("hunter2",))
    payload = {
        "api_key": "sk-live-must-not-survive",
        "prompt": "the key is sk-abc123 and the password is hunter2",
        "nested": {"Authorization": "Bearer must-not-survive", "count": 7},
        "items": ["sk-xyz789", 3, None],
    }

    try:
        # One instance, called twice -- not two instances called once. A ``CapturePolicy`` holds its
        # redactor for the life of the policy, so per-instance state is exactly the nondeterminism
        # production would hit, and constructing a second instance would hide it behind a fresh one.
        redactor = factory()
        first = redactor.redact(payload, policy=policy)
        second = redactor.redact(payload, policy=policy)
        outcomes.append(
            outcome_from_observations(
                "REDACTOR-01-DETERMINISTIC",
                REDACTOR_CONTRACT_PROFILE,
                (
                    # Canonical JSON, so key order cannot make two equal results look different.
                    observation(
                        "repeated_redaction_is_identical",
                        expected=canonical_sha256({"value": _jsonish(first)}),
                        actual=canonical_sha256({"value": _jsonish(second)}),
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("REDACTOR-01-DETERMINISTIC", REDACTOR_CONTRACT_PROFILE, exc))

    try:
        redacted = factory().redact(payload, policy=policy)
        rendered = json.dumps(_jsonish(redacted))
        outcomes.append(
            outcome_from_observations(
                "REDACTOR-02-NO-DEFAULT-SECRET-LEAK",
                REDACTOR_CONTRACT_PROFILE,
                (
                    # A secret-named key must not survive at any depth, in a mapping or inside a list.
                    observation("top_level_secret_key", expected=False, actual="sk-live-must-not-survive" in rendered),
                    observation("nested_secret_key", expected=False, actual="Bearer must-not-survive" in rendered),
                    # Non-secret data must survive, or "redact everything" would pass every rule.
                    observation("non_secret_value_survives", expected=True, actual="7" in rendered),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("REDACTOR-02-NO-DEFAULT-SECRET-LEAK", REDACTOR_CONTRACT_PROFILE, exc))

    try:
        raised = redacted_or_none(payload, policy=policy, redactor=_FailingRedactor())
        survived = redacted_or_none(payload, policy=policy, redactor=factory())
        outcomes.append(
            outcome_from_observations(
                "REDACTOR-03-FAILURE-IS-CONTAINED",
                REDACTOR_CONTRACT_PROFILE,
                (
                    # A raising redactor yields nothing. Falling back to the raw value would turn a
                    # redaction failure into a disclosure -- the opposite of what was asked for.
                    observation("failure_yields_nothing", expected=True, actual=raised is None),
                    observation("failure_does_not_propagate", expected=True, actual=True),
                    # ``None`` has to mean failure, not "redacted to empty", or the caller cannot
                    # tell a downgrade from empty content.
                    observation("success_is_distinguishable", expected=True, actual=survived is not None),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("REDACTOR-03-FAILURE-IS-CONTAINED", REDACTOR_CONTRACT_PROFILE, exc))

    try:
        mapping_result = factory().redact(payload, policy=policy)
        text_result = factory().redact("a sk-abc123 line", policy=policy)
        outcomes.append(
            outcome_from_observations(
                "REDACTOR-04-PRESERVES-THE-VALUE-SHAPE",
                REDACTOR_CONTRACT_PROFILE,
                (
                    # "Mask the whole payload" is a tempting one-liner that satisfies every leak rule
                    # and then hands the pipeline a scalar where it needs fields. The pipeline itself
                    # fails closed on this, but a redactor that trips it silently loses its consumer's
                    # content, so the contract names it rather than leaving it to be discovered.
                    observation("mapping_stays_a_mapping", expected=True, actual=isinstance(mapping_result, Mapping)),
                    observation("text_stays_text", expected=True, actual=isinstance(text_result, str)),
                    observation(
                        "mapping_keys_are_preserved",
                        expected=sorted(payload),
                        actual=sorted(mapping_result) if isinstance(mapping_result, Mapping) else None,
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("REDACTOR-04-PRESERVES-THE-VALUE-SHAPE", REDACTOR_CONTRACT_PROFILE, exc))

    return tuple(outcomes)


def run_model_io_observer_contract(
    factory: ModelIOObserverFactory,
) -> tuple[ConformanceRuleOutcome, ...]:
    """Execute the guarantees the capture pipeline gives a `ModelIOObserver`.

    The rules are pipeline guarantees rather than observer obligations, so each one drives the
    factory's observer through `dispatch_model_call` for tolerance and puts a recording observer of
    our own alongside it to witness what the capture actually held. An opaque implementation cannot
    report what it received, and asking it to would make the suite test the reporting rather than the
    contract.
    """

    outcomes: list[ConformanceRuleOutcome] = []
    content = {"final_text": "settled output", "api_key": "sk-must-not-survive"}
    receipt = ModelCallReceipt()

    try:
        witness = _RecordingObserver()
        subject = factory()
        returned = dispatch_model_call(
            receipt=receipt,
            content=content,
            subscriptions=(
                ModelIOSubscription(subject, CapturePolicy(mode="full")),
                ModelIOSubscription(witness, CapturePolicy(mode="full")),
            ),
        )
        close_model_io_subscriptions((ModelIOSubscription(subject, CapturePolicy()),))
        outcomes.append(
            outcome_from_observations(
                "MODELIO-01-PARTIAL-IMPLEMENTATION-LEGAL",
                MODEL_IO_CONTRACT_PROFILE,
                (
                    # An observer that declares only ``on_model_call`` is a complete implementation.
                    observation("declares_on_model_call", expected=True, actual=callable(getattr(subject, "on_model_call", None))),
                    observation("close_is_optional", expected=True, actual=True),
                    observation("delivery_reached_a_peer_observer", expected=1, actual=len(witness.captures)),
                    observation("receipt_returned", expected=True, actual=returned is not None),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("MODELIO-01-PARTIAL-IMPLEMENTATION-LEGAL", MODEL_IO_CONTRACT_PROFILE, exc))

    try:
        witness = _RecordingObserver()
        returned = dispatch_model_call(
            receipt=receipt,
            content=content,
            subscriptions=(
                ModelIOSubscription(_RaisingObserver(), CapturePolicy(mode="full")),
                ModelIOSubscription(factory(), CapturePolicy(mode="full")),
                ModelIOSubscription(witness, CapturePolicy(mode="full")),
            ),
        )
        outcomes.append(
            outcome_from_observations(
                "MODELIO-02-OBSERVER-FAILURE-CONTAINED",
                MODEL_IO_CONTRACT_PROFILE,
                (
                    # The call already happened and the provider has already been paid; a broken
                    # exporter does not get to undo that, nor to starve the observers behind it.
                    observation("dispatch_did_not_raise", expected=True, actual=True),
                    observation("later_observers_still_ran", expected=1, actual=len(witness.captures)),
                    observation("receipt_still_returned", expected=True, actual=returned is not None),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("MODELIO-02-OBSERVER-FAILURE-CONTAINED", MODEL_IO_CONTRACT_PROFILE, exc))

    try:
        witness = _RecordingObserver()
        dispatch_model_call(
            receipt=receipt,
            content=content,
            subscriptions=(
                ModelIOSubscription(factory(), CapturePolicy(mode="none")),
                ModelIOSubscription(witness, CapturePolicy(mode="none")),
            ),
        )
        captured = witness.captures[0]
        rendered = json.dumps(_jsonish({"content": captured.content, "digests": dict(captured.digests)}))
        outcomes.append(
            outcome_from_observations(
                "MODELIO-03-NONE-POLICY-RECEIVES-NO-CONTENT",
                MODEL_IO_CONTRACT_PROFILE,
                (
                    observation("mode", expected="none", actual=captured.mode),
                    observation("content_absent", expected=True, actual=captured.content is None),
                    # Not even a digest: ``none`` means the consumer learns nothing about the content,
                    # and a digest of a short prompt is a guessable one.
                    observation("digests_absent", expected=0, actual=len(captured.digests)),
                    observation("lengths_absent", expected=0, actual=len(captured.lengths)),
                    observation("nothing_leaked", expected=False, actual="sk-must-not-survive" in rendered),
                    # The receipt still arrives: it is metadata only, so it is safe at every mode.
                    observation("receipt_still_delivered", expected=True, actual=captured.receipt is not None),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("MODELIO-03-NONE-POLICY-RECEIVES-NO-CONTENT", MODEL_IO_CONTRACT_PROFILE, exc))

    return tuple(outcomes)


class _RecordingObserver:
    """Witnesses what a capture held, for rules an opaque implementation cannot report on."""

    def __init__(self) -> None:
        self.captures: list[ModelCallCapture] = []

    def on_model_call(self, capture: ModelCallCapture) -> None:
        self.captures.append(capture)


class _RaisingObserver:
    """An observer that always fails, for the containment rule."""

    def on_model_call(self, capture: ModelCallCapture) -> None:
        del capture
        raise RuntimeError("exporter unavailable")


class _FailingRedactor:
    """A redactor that always fails, for the fail-closed rule."""

    def redact(self, value: Any, *, policy: RedactionPolicy) -> Any:
        raise RuntimeError("redactor unavailable")


def _jsonish(value: Any) -> Any:
    """Coerce a redacted payload to JSON-safe types so it can be digested and searched."""
    if isinstance(value, Mapping):
        return {str(key): _jsonish(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonish(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _error(rule_id: str, profile_id: str, exc: Exception) -> ConformanceRuleOutcome:
    return ConformanceRuleOutcome(
        rule_id=rule_id,
        profile_id=profile_id,
        status="error",
        error=safe_exception_summary(exc),
    )
