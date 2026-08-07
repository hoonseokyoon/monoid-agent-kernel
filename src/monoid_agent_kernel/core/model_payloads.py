"""What one model call's content may say about itself once it is written down.

``model_payloads.jsonl`` (plus a ``model_payloads/`` chunk directory, created only once some
value is too large to sit on a JSONL line) is the private
run-directory replay corpus: the request preimage each ``request_digest`` was taken over, and the
settled turn that answered it. W6-2 (dx-note ``2026-08-02-v0.21-contract-replay-scope.md``
§Track B, decisions 4/5/6/8). It is the content-bearing sibling of ``model_calls.jsonl`` --
deliberately a **separate artifact**, because the ledger promises "metadata and the replay key
and no content" and the two files have different keys: the ledger is a sequence (one line per
call, ``call_index``), this corpus is mostly a set (one request record per digest, however many
calls shared it).

This module is pure, with one deliberate exception. It splits, reassembles, and shapes records;
the recorder owns every byte that reaches disk, so a run whose disk is full loses a record
rather than an answer. The exception is :func:`read_corpus_records` -- the lenient,
verified-descriptor line reader every corpus consumer whose conclusions carry authority shares
(the collector deletes on its say-so, the replay reader substitutes it for paid calls). It lives
here because a reader per consumer is the twin-drift shape this repo keeps paying for; the
validator's loop stays its own on purpose, interleaved as it is with per-line schema reporting
and pinned by the collector's spy test.

**A request record is a recipe, not a copy, and the recipe is verified.** The bytes that matter
are the exact bytes the replay key was hashed over. A record stores them as a payload tree whose
liftable values -- every value at least as large as the reference that would replace it, per tool
definition, per message and per observation -- are replaced by content-addressed chunk references,
so the ~97% of a first-turn preimage that repeats across calls (tool definitions: 17,210 of 17,782
bytes on the shipped default surface, and any real conversation only lowers the share) is stored
once per run, and so is every message a growing conversation resends.
Reassembly replaces each reference with its chunk's decoded value, re-encodes the whole through
``CANONICAL_JSON_ENCODER`` -- the same instance the digest hashed through, shared by identity,
never a settings twin -- and must reproduce the preimage byte for byte. The writer performs that
reassembly *before* writing and falls back to a verbatim payload (``refs=False``) when it fails,
so a broken decode/encode round-trip and any splitting defect are absorbed structurally rather
than defended by argument. A ``refs=False`` payload is never walked at all. Caller data shaped
like a chunk reference does not need that arm: a reference is a fixed size, so
:data:`MARKER_ENCODED_BYTES` makes every lookalike large enough to be lifted into a chunk, and a
resolved value is never re-walked.

**The response side records the turn's declared fields and nothing else.** ``raw`` is absent by
decision: it has no consumers outside the provider layer, no shape contract, and duplicates the
parsed fields -- a replayed turn answers with ``raw={}``, which is an honest statement ("this is
a replay") rather than a gap. ``reasoning`` is present *because* replay needs it: the loop
re-injects it into the next by-value turn, so a corpus without it derails one turn after every
replayed answer. Reasoning entries and the message text around them are model content; this whole
artifact is content-classified, unlike the ledger beside it.

Nothing here is truncated, ever. A response that cannot be canonically encoded, one that exceeds
:data:`~monoid_agent_kernel.core.model_io.MAX_MODEL_PAYLOAD_BYTES`, or one whose assembled record
line the corpus reader could not parse back (the recorder's refusal, not this module's), costs its
own record a typed ``unrecorded_reason`` -- the same doctrine as the replay key itself: refuse
whole, never invent a partial identity.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from monoid_agent_kernel.core._util import CANONICAL_JSON_ENCODER, sha256_bytes
from monoid_agent_kernel.core._verified_file import open_verified_regular_fd
from monoid_agent_kernel.core.json_ingress import loads_json_ingress
from monoid_agent_kernel.core.model_io import MAX_MODEL_PAYLOAD_BYTES
from monoid_agent_kernel.identifiers import namespaced_id

MODEL_PAYLOADS_SCHEMA_VERSION = namespaced_id("model-payloads.v1")
MODEL_PAYLOADS_FILENAME = "model_payloads.jsonl"
MODEL_PAYLOADS_DIRNAME = "model_payloads"

# A chunk reference, as it appears inside a request payload or a response field: an object whose
# single key is this namespaced id. Namespaced so a collision with real data is vanishingly rare
# -- and *still* not trusted: the writer verifies reassembly before writing and falls back to a
# verbatim payload, so a collision costs deduplication, never correctness.
PAYLOAD_CHUNK_REF_KEY = namespaced_id("payload-chunk.v1")

# Above this encoded size a chunk leaves the JSONL line for a file of its own in the chunk
# directory. Purely the recorder's storage decision -- *what* becomes a chunk is
# ``MARKER_ENCODED_BYTES``'s question, and this one only asks where the chunk lives, so a
# multimodal turn cannot put megabytes on one line. 256 KiB, Temporal's offload default for the
# same job.
PAYLOAD_OFFLOAD_THRESHOLD_BYTES = 262_144

PAYLOAD_CHUNK_KIND = "chunk"
MODEL_REQUEST_KIND = "model_request"
MODEL_RESPONSE_KIND = "model_response"

# Why a response record carries no response. Deliberately only the reasons a writer can actually
# produce: an early draft had a ``not_captured`` member for the request-side wiring gap (corpus
# on, preimage capture off), but that gap belongs to the *request* record -- whose absence beside
# a ledger line saying ``digest_status="ok"`` is the diagnosis -- and a response is built from
# the turn, which needs no preimage. A vocabulary member no writer emits is a fail-open pin.
UNRECORDED_REASONS = ("", "too_large", "unencodable")


@dataclass(frozen=True)
class SplitRequestPayload:
    """A verified recipe for one request preimage.

    ``refs=True`` means ``payload`` contains chunk references and ``chunks`` holds their bytes;
    ``refs=False`` means ``payload`` is the decoded preimage verbatim and reassembly must not
    walk it. Either way the constructor of this value has already proven that
    :func:`reassemble_request_preimage` reproduces the original bytes.
    """

    payload: Any
    chunks: dict[str, bytes] = field(default_factory=dict)
    refs: bool = True


@dataclass(frozen=True)
class RecordedResponse:
    """One settled turn as the corpus may record it, or the typed reason it may not.

    ``encoded`` is the canonical encoding of ``value`` -- the bytes an offloaded response file
    stores and hashes, kept beside the value so the recorder's size decision and the stored bytes
    cannot diverge.
    """

    value: dict[str, Any] | None = None
    encoded: bytes | None = None
    unrecorded_reason: str = ""


def chunk_marker(sha256: str) -> dict[str, str]:
    """The reference object that stands where an extracted value stood."""

    return {PAYLOAD_CHUNK_REF_KEY: sha256}


_HEX_DIGITS = frozenset("0123456789abcdef")


def is_chunk_sha256(value: Any) -> bool:
    """Whether ``value`` is the only thing a chunk reference is ever allowed to be.

    Every reference this package writes is ``sha256_bytes`` output, and the schema pins that shape
    on the ``chunk`` record's ``sha256`` field -- but a reference *inside* a payload is arbitrary
    caller-adjacent JSON, and a corpus arrives from wherever run directories arrive from. The
    constraint the writer holds by construction has to be re-established by every reader, because
    the readers turn this string into a filename: joined onto a directory an absolute or
    ``..``-relative string discards the base entirely. One predicate, both sides.
    """

    return isinstance(value, str) and len(value) == 64 and _HEX_DIGITS.issuperset(value)


def _is_marker(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and len(value) == 1
        and PAYLOAD_CHUNK_REF_KEY in value
        and is_chunk_sha256(value[PAYLOAD_CHUNK_REF_KEY])
    )


RESPONSE_INLINE = "inline"
RESPONSE_REFERENCE = "reference"
RESPONSE_MALFORMED = "malformed"


def response_reference(value: Any) -> tuple[str, str | None]:
    """What a ``model_response``'s ``response`` field is: inline data, a reference, or a lie.

    The trichotomy exists because its two consumers used to disagree at the third arm. A
    single-key ``{PAYLOAD_CHUNK_REF_KEY: ...}`` object is unmistakably writer-shaped -- the
    writer produces it in exactly one place and only with a ``sha256_bytes`` name -- so one
    carrying anything else is corruption, not data. The validator's inline check called it a
    reference whenever the value was a string (so ``"../../etc"`` was reported but ``123`` was
    silently data), while :func:`_is_marker` called it data whenever the sha was malformed (so
    reassembly would have embedded it verbatim). One function, one answer, both consumers: the
    replay reader refuses to resolve a ``malformed`` reference (it never becomes a filename),
    and the validator reports it as an integrity issue.

    ``inline`` covers everything else, including ``None`` (an unrecorded body) and dicts that
    merely contain the key among siblings -- those are ordinary payload data by the same
    single-key rule :func:`_is_marker` applies inside request recipes.
    """

    if isinstance(value, dict) and len(value) == 1 and PAYLOAD_CHUNK_REF_KEY in value:
        sha = value[PAYLOAD_CHUNK_REF_KEY]
        if is_chunk_sha256(sha):
            return RESPONSE_REFERENCE, sha
        return RESPONSE_MALFORMED, None
    return RESPONSE_INLINE, None


def iter_chunk_references(value: Any) -> Iterator[str]:
    """Every chunk sha ``value`` could let a reader resolve, by the writer's own predicate.

    The walk is uniform -- request recipes, response bodies, ``refs=False`` verbatim payloads,
    whole records -- because its consumer decides which directory files are garbage, and that
    decision's two mistakes are not symmetric: naming a sha no reader resolves keeps a file
    (bounded waste), while missing one a reader resolves deletes a referenced chunk, the one
    corruption ``validate_run_dir`` refuses. So this deliberately names more than any resolver
    walks -- a marker inside a verbatim payload is data to every reader and is still counted.
    Markers are recognized by :func:`_is_marker`, the predicate reassembly resolves through, never
    re-derived -- so a marker-shaped object carrying a malformed sha yields nothing, and it also
    names nothing a content-addressed directory could hold. Strings are never parsed. Depth needs no budget of
    its own: a record deeper than the ingress limit was refused by the writer and is unparseable
    to every reader that could hand a record to this walker.
    """

    if _is_marker(value):
        yield value[PAYLOAD_CHUNK_REF_KEY]
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from iter_chunk_references(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_chunk_references(item)


def corpus_keep_set(records: Iterable[dict[str, Any]]) -> set[str]:
    """The directory filenames ``records`` forbid a collector to remove.

    The union of every walked reference, plus each ``chunk`` record's own ``sha256``. The latter
    names an inline body rather than a file, and the writer cannot produce both under one name --
    the same bytes land on one side of the offload threshold deterministically -- so counting it
    can only over-keep: a same-named file would be an unreachable shadow (inline resolution
    wins), and keeping a shadow is the cheap side of the asymmetry
    :func:`iter_chunk_references` explains.
    """

    keep: set[str] = set()
    for record in records:
        keep.update(iter_chunk_references(record))
        if record.get("kind") == PAYLOAD_CHUNK_KIND:
            sha = record.get("sha256")
            if is_chunk_sha256(sha):
                keep.add(sha)
    return keep


def read_corpus_records(path: Path) -> tuple[str, list[dict[str, Any]], list[int]]:
    """(state, parseable records, damaged line numbers) -- THE lenient corpus line reader.

    Shared by every consumer whose conclusions carry authority: the collector deletes on what
    this returns (it was ``payload_gc._corpus_records``, moved here whole), and the replay
    reader substitutes recorded answers for paid provider calls on it. One function because a
    reader per consumer is a twin that drifts; the validator's loop is the deliberate
    exception (see the module docstring).

    The read goes through the verified opener because of that authority: a corpus reached
    through a planted link is not this run's corpus, and judging from it would turn the swap
    into a purge -- or into somebody else's answers replayed as this run's. A hard link is
    accepted (``require_single_link=False``) for the reason the chunk reader accepts one -- a
    hardlink-deduplicated archive is still these bytes. The line loop mirrors
    ``_validate_model_payload_digests`` exactly: blank lines skip silently, a line that fails
    ingress parsing or is not an object is damaged, the rest count.
    """

    try:
        path.lstat()
    except FileNotFoundError:
        return "absent", [], []
    except OSError:
        return "unreadable", [], []
    descriptor = open_verified_regular_fd(path, os.O_RDONLY, require_single_link=False)
    if descriptor is None:
        return "unreadable", [], []
    handle = None
    try:
        handle = os.fdopen(descriptor, "rb")
        descriptor = None  # owned by ``handle`` from here
        data = handle.read()
    except (OSError, ValueError):
        return "unreadable", [], []
    finally:
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        elif descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    records: list[dict[str, Any]] = []
    damaged: list[int] = []
    for index, raw_line in enumerate(data.split(b"\n"), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = loads_json_ingress(raw_line.decode("utf-8"))
        except Exception:  # noqa: BLE001 - unparseable is a classification here, not a failure
            damaged.append(index)
            continue
        if not isinstance(payload, dict):
            damaged.append(index)
            continue
        records.append(payload)
    return "ok", records, damaged


def _encoded(value: Any) -> bytes:
    return CANONICAL_JSON_ENCODER.encode(value).encode("utf-8")


class _ResolutionBudget:
    """How many resolved bytes one reassembly may still spend.

    Per *chunk* is not a bound on a reassembly: references are cheap and repeatable, so a payload
    holding one reference three thousand times expands a half-megabyte file into gigabytes. A
    faithful record reassembles to its preimage, which is at most
    :data:`~monoid_agent_kernel.core.model_io.MAX_MODEL_PAYLOAD_BYTES`, so this ceiling can only
    ever stop a record that was never going to verify -- every lifted value's encoding is a
    disjoint substring of the preimage, and a value referenced twice occupies its bytes twice
    there too.

    It counts *encoded* bytes, which is a proxy for the resident cost and not a measure of it:
    a pathological structure (deeply empty containers) decodes to roughly twenty times its
    serialized size, so the real peak at the ceiling is a couple of hundred megabytes rather than
    eight. Bounded, per record, and freed between records -- which is the property that matters
    here -- but the constant is not the number of bytes this will hold.
    """

    __slots__ = ("remaining",)

    def __init__(self, total: int) -> None:
        self.remaining = total

    def spend(self, amount: int) -> None:
        self.remaining -= amount
        if self.remaining < 0:
            raise ValueError("reassembly exceeds the payload ceiling")


def _filled(value: Any, resolve_chunk: Callable[[str], bytes], budget: _ResolutionBudget) -> Any:
    """``value`` with every chunk reference replaced by its chunk's decoded value.

    Recursive, and deliberately allowed to raise: a missing chunk, undecodable chunk bytes, a
    structure too deep to walk, and an expansion past the budget are all "this recipe cannot be
    reassembled", which every caller treats as refusal (the writer falls back, the validator
    reports an issue).
    """

    if _is_marker(value):
        chunk = resolve_chunk(value[PAYLOAD_CHUNK_REF_KEY])
        budget.spend(len(chunk))
        return json.loads(chunk.decode("utf-8"))
    if isinstance(value, dict):
        return {key: _filled(item, resolve_chunk, budget) for key, item in value.items()}
    if isinstance(value, list):
        return [_filled(item, resolve_chunk, budget) for item in value]
    return value


def reassemble_request_preimage(
    payload_value: Any,
    resolve_chunk: Callable[[str], bytes],
    *,
    refs: bool,
    max_bytes: int = MAX_MODEL_PAYLOAD_BYTES,
) -> bytes:
    """The exact preimage bytes a request record stands for.

    Value substitution followed by one canonical re-encoding -- never textual splicing of chunk
    bytes into a template, which would make the result depend on how the recipe happened to be
    stored rather than on what it means. Byte-identity with the original preimage holds because
    the canonical encoder is deterministic and decode∘encode is the identity on its own output;
    that is a *claim*, which is why the writer verifies it per record before writing and
    ``validate_run_dir`` re-verifies it per record after.

    ``refs=False`` skips the walk entirely: a verbatim payload is encoded as it stands, so data
    shaped like a chunk reference is never resolved. May raise; see :func:`_filled`.

    ``max_bytes`` bounds the *total* expansion, not one chunk: the per-read ceiling belongs to the
    file primitive, and a corpus that arrives from elsewhere can reference one modest chunk
    thousands of times. A faithful record cannot exceed it, because it reassembles to a preimage
    that was itself bounded by the same constant.
    """

    if not refs:
        return _encoded(payload_value)
    return _encoded(_filled(payload_value, resolve_chunk, _ResolutionBudget(max_bytes)))


# What one indirection costs: a chunk reference encodes to exactly this many bytes whatever it
# points at, because its key is fixed and its value is a fixed-width sha. It is the extraction
# threshold, and it earns that job twice. Below it a chunk cannot save anything. At or above it,
# every value that could *be* a reference is lifted -- a reference is exactly this long -- and
# reassembly never walks a value it resolved, so caller data shaped like a reference ends up inside
# a chunk, inert, instead of forcing the whole payload onto the verbatim arm.
MARKER_ENCODED_BYTES = len(
    CANONICAL_JSON_ENCODER.encode({PAYLOAD_CHUNK_REF_KEY: "0" * 64}).encode("utf-8")
)

# The terms whose *elements* are the dedup unit rather than the term itself. ``tools`` and
# ``messages`` are the load-bearing pair: both are resent whole every turn, so keying the block
# would key it by its own length -- the sha would change every turn, nothing would ever be reused,
# and a hundred-turn run would store the history a hundred times. Per element, turn N+1
# re-references turn N's chunks and adds one. ``observations`` is a per-turn delta rather than a
# growing log, so it deduplicates far less; it is elementwise for the within-turn half of the same
# argument -- one oversized tool result should not drag its siblings onto the JSONL line with it.
_ELEMENTWISE_TERMS = ("tools", "messages", "observations")


def _extracted(terms: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bytes]]:
    """The terms with their liftable values replaced by chunk references, and those chunks.

    One rule, size-scoped rather than field-scoped: lift every value whose canonical encoding is at
    least :data:`MARKER_ENCODED_BYTES`. The field-scoped predecessor named ``tools``,
    ``system_prompt``, ``messages`` and ``observations``, which left every term it did not name --
    ``instruction`` is caller-pasted text -- able to put megabytes on one JSONL line on the happy
    path, and left small values inline where a marker could hide.

    :data:`_ELEMENTWISE_TERMS` decides *what* a value is, not whether it is lifted. Per tool, not
    per tools-block, because surfaces change mid-run (hot-swap, skill binding, quota) and a
    block-level chunk would re-record all twenty-eight definitions because one left; per message
    for the reason in that constant's comment.

    A literal ``null`` system prompt falls out of the size rule rather than needing an exception:
    four bytes of ``null`` is indirection with nothing deduped.
    """

    chunks: dict[str, bytes] = {}

    def lifted(value: Any) -> Any:
        encoded = _encoded(value)
        if len(encoded) < MARKER_ENCODED_BYTES:
            return value
        sha = sha256_bytes(encoded)
        chunks[sha] = encoded
        return chunk_marker(sha)

    recipe: dict[str, Any] = {}
    for name, value in terms.items():
        if name in _ELEMENTWISE_TERMS and isinstance(value, list):
            recipe[name] = [lifted(item) for item in value]
        else:
            recipe[name] = lifted(value)
    return recipe, chunks


def split_request_payload(preimage: bytes, request_digest: str) -> SplitRequestPayload | None:
    """A verified recipe for ``preimage``, or ``None`` when no verifiable record can exist.

    Refuses -- rather than records -- twice. If ``request_digest`` is not the digest of
    ``preimage``, the caller is asking for a record whose key contradicts its own bytes, and a
    corpus entry that fails its own join is worse than an absent one. And if neither the
    chunked recipe nor the verbatim fallback reassembles to ``preimage`` (the decode/encode
    identity itself broken), nothing this function could write would survive
    ``validate_run_dir``, so nothing is written. Both refusals are the `_digest` doctrine:
    no fabricated identities.
    """

    if sha256_bytes(preimage) != request_digest:
        return None
    try:
        value = json.loads(preimage.decode("utf-8"))
    except Exception:
        return None

    tag = None
    if isinstance(value, dict) and len(value) == 1:
        tag = next(iter(value))
    if tag is not None and isinstance(value[tag], dict):
        try:
            recipe_terms, chunks = _extracted(value[tag])
            recipe = {tag: recipe_terms}
            if (
                chunks
                and reassemble_request_preimage(recipe, chunks.__getitem__, refs=True) == preimage
            ):
                return SplitRequestPayload(payload=recipe, chunks=chunks, refs=True)
        except Exception:
            pass  # fall through to the verbatim shape; the reason does not change the answer

    try:
        if reassemble_request_preimage(value, _no_resolution, refs=False) == preimage:
            return SplitRequestPayload(payload=value, chunks={}, refs=False)
    except Exception:
        pass
    return None


def _no_resolution(sha256: str) -> bytes:
    raise LookupError(f"a refs=False payload resolves no chunks (asked for {sha256})")


RECORDED_TURN_FIELDS = (
    "response_id",
    "final_text",
    "tool_calls",
    "reasoning",
    "usage",
    "stop_reason",
    "provider_retried",
)
"""Every field a recorded answer carries, declared once beside the writer that emits them.

The reader needs the same list to tell a recorded turn from any other object that happens to
be a JSON dict: without it, a corrupt or foreign body reconstructs into an *empty* turn, which
the loop then rejects as a model error and blames on a model it never called.
``response_record_body`` asserts it builds exactly these, so the two stay one list.
"""


def response_record_body(turn: Any) -> RecordedResponse:
    """One settled turn as a record body, or the typed reason there is none.

    The field list is declared here, once, and ``raw`` is not on it (module docstring). A tool
    call that does not carry the ``id``/``name``/``arguments`` triple -- the legacy
    preserved-beside-a-settled-answer shape -- makes the whole response ``unencodable`` rather
    than a bounded repr: the capture surface may describe such an entry, but a replay corpus
    that recorded a description would replay a fabricated call.
    """

    try:
        calls = []
        for call in getattr(turn, "tool_calls", ()) or ():
            call_id = getattr(call, "id", None)
            name = getattr(call, "name", None)
            arguments = getattr(call, "arguments", None)
            if (
                not isinstance(call_id, str)
                or not isinstance(name, str)
                or not isinstance(arguments, dict)
            ):
                return RecordedResponse(unrecorded_reason="unencodable")
            calls.append({"id": call_id, "name": name, "arguments": arguments})
        value: dict[str, Any] = {
            "response_id": getattr(turn, "response_id", None),
            "final_text": getattr(turn, "final_text", None),
            "tool_calls": calls,
            "reasoning": list(getattr(turn, "reasoning", ()) or ()),
            "usage": dict(getattr(turn, "usage", {}) or {}),
            "stop_reason": getattr(turn, "stop_reason", None),
            "provider_retried": bool(getattr(turn, "provider_retried", False)),
        }
        encoded = _encoded(value)
    except Exception:
        return RecordedResponse(unrecorded_reason="unencodable")
    if set(value) != set(RECORDED_TURN_FIELDS):
        # Outside the try, and a raise rather than an assert. Inside, an AssertionError is an
        # Exception like any other and drift would silently reclassify every recorded answer
        # as ``unencodable`` -- a corpus that records nothing, surfacing much later as a replay
        # miss blaming the corpus. And `python -O` erases an assert, so the rule the reader
        # depends on would simply not exist in an optimized deployment.
        raise RuntimeError(
            "the recorded-turn field list and the body this function builds have drifted: "
            f"{sorted(set(value) ^ set(RECORDED_TURN_FIELDS))}"
        )
    if len(encoded) > MAX_MODEL_PAYLOAD_BYTES:
        return RecordedResponse(unrecorded_reason="too_large")
    return RecordedResponse(value=value, encoded=encoded)


def _envelope(kind: str, *, run_id: str, root_run_id: str, recorded_at: str) -> dict[str, Any]:
    return {
        "schema_version": MODEL_PAYLOADS_SCHEMA_VERSION,
        "kind": kind,
        "run_id": run_id,
        "root_run_id": root_run_id,
        "recorded_at": recorded_at,
    }


def chunk_record(
    chunk: bytes, *, run_id: str, root_run_id: str, recorded_at: str
) -> dict[str, Any]:
    """One inline chunk. ``text`` is the canonical JSON fragment verbatim -- chunk bytes are
    always encoder output, so they are UTF-8 text and need no base64 detour; the sha is computed
    here from the same bytes, so a record cannot be built already lying about its content."""

    return {
        **_envelope(
            PAYLOAD_CHUNK_KIND, run_id=run_id, root_run_id=root_run_id, recorded_at=recorded_at
        ),
        "sha256": sha256_bytes(chunk),
        "text": chunk.decode("utf-8"),
    }


def model_request_record(
    payload_value: Any,
    *,
    refs: bool,
    request_digest: str,
    digest_generation: str,
    run_id: str,
    root_run_id: str,
    recorded_at: str,
) -> dict[str, Any]:
    """One request preimage, keyed by its digest. Set semantics: a run that issues the same
    request twice writes this once, and the ledger's two lines both join to it. There is
    deliberately no ``call_index`` here -- naming the call that happened to write first would
    turn a set member into a sequence entry."""

    return {
        **_envelope(
            MODEL_REQUEST_KIND, run_id=run_id, root_run_id=root_run_id, recorded_at=recorded_at
        ),
        "request_digest": request_digest,
        "digest_generation": digest_generation,
        "refs": refs,
        "payload": payload_value,
    }


def model_response_record(
    response: dict[str, Any] | None,
    *,
    call_index: int,
    request_digest: str,
    unrecorded_reason: str,
    run_id: str,
    root_run_id: str,
    recorded_at: str,
) -> dict[str, Any]:
    """One settled turn, keyed by the call that produced it. Sequence semantics: the same digest
    answered twice is two of these, because models are not functions and the corpus records what
    happened rather than deciding which answer is canonical (that policy is the replay adapter's,
    W6-4). ``response`` is the inline body, a chunk reference to an offloaded one, or ``null``
    with ``unrecorded_reason`` saying why. An empty ``request_digest`` is legal and joins the
    ledger line whose ``digest_status`` names the reason."""

    return {
        **_envelope(
            MODEL_RESPONSE_KIND, run_id=run_id, root_run_id=root_run_id, recorded_at=recorded_at
        ),
        "call_index": call_index,
        "request_digest": request_digest,
        "unrecorded_reason": unrecorded_reason,
        "response": response,
    }
