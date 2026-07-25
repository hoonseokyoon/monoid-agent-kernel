"""Who is asking for this model call, and on whose behalf.

A model call needs an identity that is *not* a run. `AgentRunSpec` answers "what agent run is this",
but the kernel's model-call surface is reachable without a run at all — an LLM-only Skill has a
`skill_id`, maybe a batch item, and nothing else. `InvocationContext` is that identity: the caller
supplies it once, and every receipt, observation, and span the call produces carries it.

Everything is optional. A bare `InvocationContext()` is legal and means "an anonymous call", because
refusing to run without a `run_id` would put the kernel back where it started. The fields exist so
that a caller who *does* know its provenance can hand it over in one value rather than threading
eight keyword arguments through the stack.

Trace fields are carried, not enforced: `traceparent` and `tracestate` follow
`core.trace_context`'s contract, where a malformed header is ignored rather than fatal. That is
deliberate. Observability metadata must never be able to fail a call, so the constructor stores what
it is given verbatim and the derived accessors report `""` for anything unparseable. Use
`child_span()` to descend — it mints a fresh root when there is nothing usable to descend from, so
the result is always a valid `traceparent`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from monoid_agent_kernel.core.trace_context import (
    child_traceparent,
    parse_traceparent,
)
from monoid_agent_kernel.core.wire_validation import (
    WireValidationError,
    parse_int,
    parse_str,
    require_object,
)


@dataclass(frozen=True)
class InvocationContext:
    """The provenance of one model call.

    ``run_id`` is present when an agent run is driving the call and empty when it is not, which is
    the whole point of the type: the same runner serves both. ``step_id`` names the caller's own unit
    of work (a pipeline step, a Skill phase) and ``attempt`` counts the caller's retries of that unit
    — not the provider-level retries, which the adapter owns and the receipt reports separately.

    ``batch_id`` / ``item_id`` / ``case_id`` carry fan-out identity for callers that process many
    inputs through one Skill, so a receipt can be attributed to the input that produced it.
    ``attributes`` is an open string-to-string map for whatever the caller needs that this type does
    not name; it is copied on construction so a later mutation of the caller's dict cannot alter a
    context already handed to the kernel.
    """

    run_id: str = ""
    skill_id: str = ""
    skill_digest: str = ""
    step_id: str = ""
    attempt: int = 1
    batch_id: str = ""
    item_id: str = ""
    case_id: str = ""
    traceparent: str = ""
    tracestate: str = ""
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.attempt < 1:
            raise ValueError("invocation attempt must be 1 or greater")
        for key, value in self.attributes.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise WireValidationError("invocation attributes must be a mapping of str to str")
        # Frozen, so copy through ``object.__setattr__``. Without this the caller keeps a live handle
        # on the mapping inside a value the kernel is about to treat as immutable.
        object.__setattr__(self, "attributes", dict(self.attributes))

    @property
    def trace_id(self) -> str:
        """The stable end-to-end trace id, or ``""`` when there is no usable ``traceparent``."""
        parsed = parse_traceparent(self.traceparent)
        return parsed["trace_id"] if parsed is not None else ""

    @property
    def span_id(self) -> str:
        """The id of the span this call is a child of, or ``""`` when unparseable."""
        parsed = parse_traceparent(self.traceparent)
        return parsed["span_id"] if parsed is not None else ""

    def child_span(self) -> InvocationContext:
        """This context with ``traceparent`` advanced to a fresh child span.

        Doubles as "give me a trace to hang spans off": `child_traceparent` mints a new root when the
        parent is absent or malformed, so the result always parses. Everything else is preserved,
        `tracestate` included — it is an opaque vendor list that propagates verbatim.
        """
        return replace(self, traceparent=child_traceparent(self.traceparent))

    def with_attributes(self, **attributes: str) -> InvocationContext:
        """This context with ``attributes`` merged over the existing ones."""
        return replace(self, attributes={**self.attributes, **attributes})

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "skill_id": self.skill_id,
            "skill_digest": self.skill_digest,
            "step_id": self.step_id,
            "attempt": self.attempt,
            "batch_id": self.batch_id,
            "item_id": self.item_id,
            "case_id": self.case_id,
            "traceparent": self.traceparent,
            "tracestate": self.tracestate,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_json(cls, payload: Any) -> InvocationContext:
        payload = require_object(payload, "invocation context")
        # Absent and explicit ``null`` both mean "no attributes"; anything else present is checked, so
        # a falsy wrong type cannot pass for an empty map. Same rule as ``core.model_io`` uses, so the
        # two halves of one receipt payload do not disagree about what a malformed field means.
        raw_attributes = payload.get("attributes")
        attributes_payload = (
            {} if raw_attributes is None else require_object(raw_attributes, "attributes")
        )
        return cls(
            run_id=parse_str(payload, "run_id"),
            skill_id=parse_str(payload, "skill_id"),
            skill_digest=parse_str(payload, "skill_digest"),
            step_id=parse_str(payload, "step_id"),
            attempt=parse_int(payload, "attempt", default=1),
            batch_id=parse_str(payload, "batch_id"),
            item_id=parse_str(payload, "item_id"),
            case_id=parse_str(payload, "case_id"),
            traceparent=parse_str(payload, "traceparent"),
            tracestate=parse_str(payload, "tracestate"),
            attributes=attributes_payload,
        )
