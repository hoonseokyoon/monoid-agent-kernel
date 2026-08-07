"""Reading a replay corpus back: file-order consumption, typed misses, verified resolution.

W6-4b B2, the consumer ``model_payloads.py`` writes for. The reader's conclusions substitute
recorded answers for paid provider calls, so it re-establishes on arrival every rule the writer
holds by construction: lines through :func:`~.model_payloads.read_corpus_records` (the same
verified-descriptor reader the collector deletes on), references through
:func:`~.model_payloads.response_reference` (the same trichotomy the validator reports through),
chunk bytes re-hashed before they are believed, and nothing caller-shaped ever joined onto a
directory.

**Selection policy is file order, each answer once** (D-c). The corpus records what happened --
``model_response`` is sequence-shaped because models are not functions -- and the reader's only
honest option is to hand those answers back in the order they were produced. That rule is also
what makes a durable-resumed run's corpus readable: ``call_index`` restarts per activation and
the request seen-set is activation-local, so duplicates and second zeros are the *ordinary*
shape, resolved by file position rather than by any per-record field. A record whose body cannot
be given back (``unrecorded_reason``, an unresolvable reference, a body that is not a recorded
turn) spends its slot **only when the caller moves the conversation past it** by serving that
call live: then the original's next call and this run's next call are at the same position
again. Parking on the refusal does not spend it, because the loop's contract for a
``config_recoverable`` failure is an idempotent re-attempt of the *same* call, and a refusal
that advanced would answer that re-attempt with the next call's recording.

**The miss vocabulary is exactly six** (D-i): ``no_key`` (the adapter's own refusal to key a
request -- named here so the vocabulary has one home, produced by no method of this class),
``absent``, ``not_recorded``, ``identity_mismatch``, ``exhausted``, ``generation_mismatch``.

**Diagnosis speaks config and structure, never content.** Identity terms (the model projection
and the provider) are compared by value and named with expected/actual -- the ledger beside this
corpus already records both in plaintext, so naming them discloses nothing new. Prompt terms are
named by term name and digest prefix only: a diverging ``observations`` is the nondeterministic-
tool shape and the operator needs its *name*, not its bytes, which land on public event
surfaces downstream.

**Concurrent callers get each-once, not an order.** ``consume`` holds a lock across the whole
take, so two callers can never be handed one slot and none is lost. Which of them gets which
recording is the scheduler's answer, not this class's -- and the kernel does drive calls
concurrently: a child loop is constructed with the parent's adapter instance, and background
subagents drain the reentry queue together. Two identical background children therefore issue
the *same* key (nothing run-scoped is in it) and divide that key's recordings between them in
whatever order they arrive. Replay is deterministic for a run whose calls are ordered; a run
whose concurrency the recording did not fix is replayed as faithfully as it was run, which is
to say not deterministically. Same family as the spawn-observation limit in ``docs/CLI.md``.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from monoid_agent_kernel.core._util import CANONICAL_JSON_ENCODER, sha256_bytes
from monoid_agent_kernel.core._verified_file import (
    VerifiedFileIdentity,
    file_identity,
    read_verified_bytes,
)
from monoid_agent_kernel.core.model_io import MAX_MODEL_PAYLOAD_BYTES
from monoid_agent_kernel.core.model_payloads import (
    MODEL_PAYLOADS_DIRNAME,
    MODEL_PAYLOADS_FILENAME,
    MODEL_PAYLOADS_SCHEMA_VERSION,
    MODEL_REQUEST_KIND,
    MODEL_RESPONSE_KIND,
    PAYLOAD_CHUNK_KIND,
    RESPONSE_MALFORMED,
    RESPONSE_REFERENCE,
    is_chunk_sha256,
    read_corpus_records,
    reassemble_request_preimage,
    response_reference,
)

MISS_NO_KEY = "no_key"
MISS_ABSENT = "absent"
MISS_NOT_RECORDED = "not_recorded"
MISS_IDENTITY_MISMATCH = "identity_mismatch"
MISS_EXHAUSTED = "exhausted"
MISS_GENERATION_MISMATCH = "generation_mismatch"

REPLAY_MISS_REASONS = (
    MISS_NO_KEY,
    MISS_ABSENT,
    MISS_NOT_RECORDED,
    MISS_IDENTITY_MISMATCH,
    MISS_EXHAUSTED,
    MISS_GENERATION_MISMATCH,
)
"""The approved vocabulary (D-i), in one place. ``no_key`` belongs to the adapter's own
encoder refusal and ``identity_mismatch`` to :meth:`ReplayCorpus.diagnose`; ``consume``
produces the other four. A reason outside this tuple is a contract change, not a detail."""

# How many diverging prompt terms one diagnosis will name, and how much of each digest.
# Bounds the message, not the comparison -- and a digest prefix is still content-free.
_DIAGNOSED_TERMS = 4
_DIGEST_PREFIX = 12


@dataclass(frozen=True)
class ReplayedResponse:
    """A hit: one recorded answer, verbatim, with the coordinates that let it join the ledger."""

    body: dict[str, Any]
    call_index: int
    recorded_at: str
    run_id: str
    slot: int
    """Position in this key's recorded sequence -- the coordinate :meth:`ReplayCorpus.release`
    needs to give the answer back if the caller turns out not to be able to use it."""


@dataclass(frozen=True)
class ReplayMissReason:
    """A typed refusal: which of the six reasons, and a content-free sentence of why."""

    reason: str
    detail: str


@dataclass(frozen=True)
class _ResponseEntry:
    response: Any
    unrecorded_reason: str
    call_index: int
    recorded_at: str
    run_id: str


@dataclass(frozen=True)
class _RequestEntry:
    generation: str
    refs: bool
    payload: Any
    run_id: str


def _term_digest(value: Any) -> str:
    try:
        return sha256_bytes(CANONICAL_JSON_ENCODER.encode(value).encode("utf-8"))
    except Exception:  # noqa: BLE001 - an unencodable term still deserves a stable name
        return "unencodable"


class ReplayCorpus:
    """Every record the named run directories hold, indexed for consumption and diagnosis."""

    # The shared line reader, held as an attribute so the share is pinnable by identity --
    # a re-definition here would be the mirror the move to model_payloads exists to prevent.
    _read = staticmethod(read_corpus_records)

    def __init__(self) -> None:
        self._requests: dict[str, _RequestEntry] = {}
        self._responses: dict[str, list[_ResponseEntry]] = {}
        self._cursors: dict[str, int] = {}
        self._chunks: dict[str, bytes] = {}
        self._chunk_dirs: list[Path] = []
        self._generations: list[str] = []
        self._run_ids: list[str] = []
        self._damaged = 0
        self._rejected = 0
        self._repeated_sources = 0
        self._terms_cache: dict[str, Mapping[str, Any] | None] = {}
        self._profiles: tuple[dict[str, Any], ...] | None = None
        self._lock = threading.Lock()

    # --- loading ---------------------------------------------------------------------------

    @classmethod
    def load(cls, run_dirs: Sequence[Path]) -> "ReplayCorpus":
        """Index ``run_dirs`` in argument order; file order within each is preserved.

        Refuses at construction -- not on the tenth turn -- when a named directory has no
        corpus *file* to read: a replay source that never recorded is an operator mistake, and
        every later miss it would cause misdirects toward the request rather than the source.

        A file that exists and holds nothing usable is not refused here, because "empty" and
        "damaged" are answers about content rather than about the source, and the reader
        reports them (``damaged_lines``, ``rejected_records``) for the preflight to act on. A
        library embedder that skips the preflight gets the misses, not the construction error.
        """

        corpus = cls()
        indexed: set[VerifiedFileIdentity] = set()
        for run_dir in run_dirs:
            run_dir = Path(run_dir)
            path = run_dir / MODEL_PAYLOADS_FILENAME
            state, records, damaged = cls._read(path)
            if state != "ok":
                raise ValueError(f"replay source has no readable corpus ({state}): {run_dir}")
            # The chunk directory joins the union even for a repeat: a corpus reached twice
            # is the same references, and a hardlinked copy in another directory keeps its
            # own offloaded bytes. Only the *records* are at risk of being counted twice.
            corpus._chunk_dirs.append(run_dir / MODEL_PAYLOADS_DIRNAME)
            identity = corpus._corpus_identity(path)
            if identity is not None and identity in indexed:
                # One directory named twice -- as an id and as a path, through a link, or
                # simply repeated -- is one source. Indexing it again would append every
                # answer to its queue a second time, and "each answer once" would quietly
                # become "each answer once per spelling": the call that should have earned
                # ``exhausted`` gets a stale recorded body instead.
                corpus._repeated_sources += 1
                continue
            if identity is not None:
                indexed.add(identity)
            corpus._damaged += len(damaged)
            for record in records:
                corpus._index(record)
        return corpus

    @staticmethod
    def _corpus_identity(path: Path) -> VerifiedFileIdentity | None:
        """The device/inode pair naming this corpus file, or ``None`` when it cannot be named.

        Unnameable means "index it": the reader already accepted these bytes, and refusing to
        index a readable source because its metadata is unavailable would lose answers rather
        than duplicate them.
        """

        try:
            return file_identity(path.lstat())
        except OSError:
            return None

    def _index(self, record: Mapping[str, Any]) -> None:
        if record.get("schema_version") != MODEL_PAYLOADS_SCHEMA_VERSION:
            # The validator enforces this on the same bytes, and a reader that skipped it
            # would serve a corpus the kernel's own `monoid validate` calls corrupt -- or,
            # after a version bump, serve v2 answers as though the v1 field semantics still
            # held. A version is a promise about what the other fields mean.
            self._rejected += 1
            return
        kind = record.get("kind")
        run_id = record.get("run_id")
        if isinstance(run_id, str) and run_id and run_id not in self._run_ids:
            self._run_ids.append(run_id)
        if kind == PAYLOAD_CHUNK_KIND:
            text = record.get("text")
            sha = record.get("sha256")
            if not isinstance(text, str) or not is_chunk_sha256(sha):
                self._rejected += 1
                return
            data = text.encode("utf-8")
            # Re-hashed before it is believed, exactly like the validator: an inline chunk
            # that lies about its name must not become resolvable under that name.
            if sha256_bytes(data) != sha:
                self._rejected += 1
                return
            self._chunks.setdefault(sha, data)
        elif kind == MODEL_REQUEST_KIND:
            digest = record.get("request_digest")
            refs = record.get("refs")
            generation = record.get("digest_generation")
            if (
                not is_chunk_sha256(digest)
                or not isinstance(refs, bool)
                or not isinstance(generation, str)
                or not generation
            ):
                self._rejected += 1
                return
            if generation not in self._generations:
                self._generations.append(generation)
            # First-wins across duplicates: a durable resume re-records the same digest with
            # an empty seen-set, and the earliest record is the one whose activation the
            # file-order answers below it belong to.
            self._requests.setdefault(
                digest,
                _RequestEntry(
                    generation=generation,
                    refs=refs,
                    payload=record.get("payload"),
                    run_id=str(record.get("run_id") or ""),
                ),
            )
        elif kind == MODEL_RESPONSE_KIND:
            digest = record.get("request_digest")
            if not is_chunk_sha256(digest):
                # An empty digest is legal in the file (it joins the ledger's refusal line)
                # but it can never be *asked for* by digest, so it has no queue to join.
                self._rejected += 1
                return
            call_index = record.get("call_index")
            if isinstance(call_index, bool) or not isinstance(call_index, int):
                self._rejected += 1
                return
            reason = record.get("unrecorded_reason")
            self._responses.setdefault(digest, []).append(
                _ResponseEntry(
                    response=record.get("response"),
                    unrecorded_reason=reason if isinstance(reason, str) else "",
                    call_index=call_index,
                    recorded_at=str(record.get("recorded_at") or ""),
                    run_id=str(record.get("run_id") or ""),
                )
            )

    # --- consumption -----------------------------------------------------------------------

    def consume(self, digest: str, *, generation: str) -> ReplayedResponse | ReplayMissReason:
        """The next unconsumed answer recorded under ``digest``, or the typed reason there is
        none. **Advances the cursor only when it hands an answer back.**

        Refusing and consuming are separate acts (D-c, refined in round-1 review). A refused
        body leaves the cursor where it was, because a refusal parks the turn and the loop's
        contract for a ``config_recoverable`` failure is an *idempotent re-attempt of the same
        call*: an advancing refusal would serve that re-attempt the next call's recorded
        answer, silently, as a valid turn. A caller that instead moves the conversation past
        the refused call -- by serving it live -- says so with :meth:`spend_refused`, and the
        sequence realigns.

        Materialization happens under the lock so that two concurrent callers cannot both be
        handed the slot one of them is about to take; the cost is that a chunk read serializes
        them, which is the right trade for a rule about not answering twice.
        """

        with self._lock:
            queue = self._responses.get(digest)
            if not queue:
                return self._absent_locked(digest, generation)
            cursor = self._cursors.get(digest, 0)
            if cursor >= len(queue):
                return ReplayMissReason(
                    MISS_EXHAUSTED,
                    f"all {len(queue)} recorded answer(s) under this key were already consumed",
                )
            entry = queue[cursor]
            body = self._entry_body(entry)
            if isinstance(body, ReplayMissReason):
                return body
            self._cursors[digest] = cursor + 1
        return ReplayedResponse(
            body=body,
            call_index=entry.call_index,
            recorded_at=entry.recorded_at,
            run_id=entry.run_id,
            slot=cursor,
        )

    def spend_refused(self, digest: str) -> None:
        """Advance past a slot this corpus could not give back, once the caller has served
        that call another way.

        The only honest reason to spend a refused slot: the conversation really has moved past
        the call it belongs to, so the original run's next call and this run's next call are
        at the same position again. Serving the miss live (``--replay-fallthrough``) is that
        reason; parking on it is not.
        """

        with self._lock:
            queue = self._responses.get(digest)
            if not queue:
                return
            cursor = self._cursors.get(digest, 0)
            if cursor < len(queue):
                self._cursors[digest] = cursor + 1

    def release(self, digest: str, slot: int) -> None:
        """Give back an answer :meth:`consume` handed out that the caller could not use.

        Only that exact slot, and only while nothing else has moved: under concurrent callers
        another turn may already hold the next one, and rewinding then would hand it out twice.
        """

        with self._lock:
            if self._cursors.get(digest, 0) == slot + 1:
                self._cursors[digest] = slot

    def _entry_body(self, entry: _ResponseEntry) -> dict[str, Any] | ReplayMissReason:
        """One entry's answer body, materialized and verified -- or the refusal it earns.

        Shared by :meth:`consume` and the cursor-free evidence view, so "what does this
        record actually say" has one answer however it is asked.
        """

        if entry.unrecorded_reason:
            return ReplayMissReason(
                MISS_NOT_RECORDED,
                f"the answer was not recorded ({entry.unrecorded_reason}); "
                f"run {entry.run_id} call_index {entry.call_index}",
            )
        shape, sha = response_reference(entry.response)
        if shape == RESPONSE_MALFORMED:
            return ReplayMissReason(
                MISS_NOT_RECORDED,
                "a response reference is not a content-addressed name "
                f"(run {entry.run_id} call_index {entry.call_index})",
            )
        if shape == RESPONSE_REFERENCE:
            assert sha is not None
            try:
                body_bytes = self._resolve_chunk(sha)
                body = json.loads(body_bytes.decode("utf-8"))
            except Exception as error:  # noqa: BLE001 - every failure is one refusal
                return ReplayMissReason(
                    MISS_NOT_RECORDED,
                    f"the recorded answer could not be resolved ({error}); "
                    f"run {entry.run_id} call_index {entry.call_index}",
                )
        else:
            body = entry.response
        if not isinstance(body, dict):
            return ReplayMissReason(
                MISS_NOT_RECORDED,
                f"the recorded answer is not an object; run {entry.run_id} "
                f"call_index {entry.call_index}",
            )
        return body

    # --- cursor-free evidence views ----------------------------------------------------------

    def request_terms_view(self) -> tuple[Mapping[str, Any], ...]:
        """The reassembled terms of every readable request record, in index order.

        For derivation-time evidence scans (the replay adapter's impersonation rules read
        recorded messages, not record lines -- any term at marker size is a chunk, P4).
        Reads nothing consumable: cursors are untouched.
        """

        return tuple(
            terms for digest in self._requests if (terms := self._request_terms(digest)) is not None
        )

    def response_bodies_view(self) -> tuple[Mapping[str, Any], ...]:
        """Every materializable answer body, grouped by key then in file order, cursors
        untouched.

        Not file order across the whole corpus: answers are indexed per key, so two keys whose
        records interleave in the file come back one key at a time. The only caller asks
        whether *any* body carries reasoning, which no ordering changes.

        Unrecorded and unresolvable entries are simply absent from the view: an evidence
        scan asks what the corpus can say, and a record that cannot testify says nothing.
        """

        bodies: list[Mapping[str, Any]] = []
        for queue in self._responses.values():
            for entry in queue:
                body = self._entry_body(entry)
                if not isinstance(body, ReplayMissReason):
                    bodies.append(body)
        return tuple(bodies)

    def _absent_locked(self, digest: str, generation: str) -> ReplayMissReason:
        if digest in self._requests:
            return ReplayMissReason(
                MISS_ABSENT,
                "a request record exists under this key but no answer was recorded "
                "(the original call failed, or its activation ended before answering)",
            )
        mismatch = self._generation_mismatch(generation)
        if mismatch is not None:
            return mismatch
        return ReplayMissReason(MISS_ABSENT, "no record carries this key")

    def generation_divergence(self, generation: str) -> str | None:
        """The sentence naming a wholesale generation retirement, or ``None``.

        One function for the preflight and for the miss diagnosis, the same way
        :meth:`identity_divergence` serves both -- a corpus retired by a generation bump can
        match nothing, and "before the run starts" is where the CHANGELOG promises to say so.
        """

        mismatch = self._generation_mismatch(generation)
        return None if mismatch is None else mismatch.detail

    def _generation_mismatch(self, generation: str) -> ReplayMissReason | None:
        if self._generations and generation not in self._generations:
            recorded = ", ".join(self._generations)
            return ReplayMissReason(
                MISS_GENERATION_MISMATCH,
                f"the corpus was recorded under {recorded}; this run computes keys "
                f"under {generation} -- every lookup will miss until one of them changes",
            )
        return None

    def _resolve_chunk(self, sha: str) -> bytes:
        """Bytes for ``sha``, from the inline map else the chunk directories, re-verified.

        Union-global on purpose: chunks are content-addressed and re-hashed here, so a chunk
        recorded by one family member safely serves a reference written by another -- the
        same bytes carry the same name wherever they landed.
        """

        if sha in self._chunks:
            return self._chunks[sha]
        if not is_chunk_sha256(sha):  # defense in depth; callers came through the trichotomy
            raise ValueError("chunk reference is not a content-addressed name")
        for chunk_dir in self._chunk_dirs:
            data = read_verified_bytes(chunk_dir / sha, max_bytes=MAX_MODEL_PAYLOAD_BYTES)
            if data is None:
                continue
            if sha256_bytes(data) != sha:
                raise ValueError(f"offloaded chunk {sha} does not match its name")
            return data
        raise LookupError(f"offloaded chunk {sha} is not present in any replay source")

    # --- diagnosis and preflight -----------------------------------------------------------

    def identity_profiles(self) -> tuple[dict[str, Any], ...]:
        """The distinct (provider, model) identities this corpus recorded requests under.

        Reassembled, never read off the record line: the size-scoped splitter lifts any term
        at or past marker size into a chunk, so the ``model`` block of a configured run is
        routinely a reference -- a reader that inspected raw payloads would go blind exactly
        on the corpora whose identity is worth checking (P4).
        """

        if self._profiles is not None:
            return self._profiles
        profiles: list[dict[str, Any]] = []
        seen: set[str] = set()
        for digest, entry in self._requests.items():
            terms = self._request_terms(digest)
            if terms is None:
                continue
            profile = {
                "provider": terms.get("provider"),
                "model": terms.get("model"),
                "run_id": entry.run_id,
            }
            key = _term_digest({"provider": profile["provider"], "model": profile["model"]})
            if key in seen:
                continue
            seen.add(key)
            profiles.append(profile)
        self._profiles = tuple(profiles)
        return self._profiles

    def identity_divergence(self, *, model: Any, provider: Any) -> str | None:
        """``None`` when some recorded identity matches; else expected/actual, named.

        One function for the preflight and for :meth:`diagnose`'s identity branch -- the two
        callers exist to give the same answer at different times, so they must not be two
        implementations. Values are config vocabulary the ledger already records in
        plaintext; nothing here is conversation content.
        """

        profiles = self.identity_profiles()
        if not profiles:
            return "the corpus holds no readable request identities to compare against"
        for profile in profiles:
            if profile["model"] == model and profile["provider"] == provider:
                return None
        expected = profiles[0]
        clauses: list[str] = []
        if expected["provider"] != provider:
            clauses.append(f"provider recorded {expected['provider']!r}, computing {provider!r}")
        recorded_model = expected["model"] if isinstance(expected["model"], dict) else {}
        live_model = model if isinstance(model, dict) else {}
        for name in sorted(set(recorded_model) | set(live_model)):
            if recorded_model.get(name) != live_model.get(name):
                clauses.append(
                    f"model.{name} recorded {recorded_model.get(name)!r}, "
                    f"computing {live_model.get(name)!r}"
                )
        if not clauses:
            clauses.append("the identity block differs in shape")
        suffix = f" ({len(profiles)} recorded identities)" if len(profiles) > 1 else ""
        return "; ".join(clauses) + suffix

    def diagnose(
        self, payload: Mapping[str, Any], *, generation: str, digest: str | None = None
    ) -> ReplayMissReason:
        """Refine an ``absent`` miss: the failed-call shape first, then generation, identity,
        and term-by-term.

        ``payload`` is the live, generation-wrapped identity payload the lookup hashed --
        handed in rather than rebuilt so the diagnosis and the key derive from one
        composition. ``digest`` (when the caller has one) lets the request-known-answer-absent
        shape (P6) keep its own sentence: comparing that request against itself would
        otherwise "diagnose" zero diverging terms and say nothing useful. The rest is the
        frequency order the design predicts: a wholesale generation change misses everything,
        a config divergence (the runtime config authors the key's model identity) misses
        everything under one name, and only then is it worth naming individual prompt terms.
        """

        if digest is not None and digest in self._requests and not self._responses.get(digest):
            return ReplayMissReason(
                MISS_ABSENT,
                "a request record exists under this key but no answer was recorded "
                "(the original call failed, or its activation ended before answering)",
            )
        mismatch = self._generation_mismatch(generation)
        if mismatch is not None:
            return mismatch
        live_terms = payload.get(generation)
        if not isinstance(live_terms, Mapping):
            return ReplayMissReason(
                MISS_ABSENT, "the live payload carries no terms under this generation"
            )
        divergence = self.identity_divergence(
            model=live_terms.get("model"), provider=live_terms.get("provider")
        )
        if divergence is not None:
            return ReplayMissReason(MISS_IDENTITY_MISMATCH, divergence)
        return ReplayMissReason(MISS_ABSENT, self._closest_divergence(live_terms))

    def _closest_divergence(self, live_terms: Mapping[str, Any]) -> str:
        live_digests = {name: _term_digest(value) for name, value in live_terms.items()}
        best: tuple[int, str, dict[str, str]] | None = None
        for digest in self._requests:
            terms = self._request_terms(digest)
            if terms is None:
                continue
            recorded = {name: _term_digest(value) for name, value in terms.items()}
            matches = sum(1 for name in live_digests if recorded.get(name) == live_digests[name])
            if best is None or matches > best[0]:
                best = (matches, self._requests[digest].run_id, recorded)
        if best is None:
            return "identity matches but the corpus holds no reassemblable request to compare"
        _matches, run_id, recorded = best
        diverging = sorted(
            name
            for name in set(live_digests) | set(recorded)
            if live_digests.get(name) != recorded.get(name)
        )
        named = diverging[:_DIAGNOSED_TERMS]
        clauses = [
            f"{name} live={live_digests.get(name, 'missing')[:_DIGEST_PREFIX]} "
            f"recorded={recorded.get(name, 'missing')[:_DIGEST_PREFIX]}"
            for name in named
        ]
        more = f" and {len(diverging) - len(named)} more" if len(diverging) > len(named) else ""
        return (
            "identity matches; diverging terms vs the closest recorded request "
            f"(run {run_id}): " + "; ".join(clauses) + more
        )

    def _request_terms(self, digest: str) -> Mapping[str, Any] | None:
        """The reassembled terms of one request record, or ``None`` when it cannot honor
        its own digest -- a corpus lying about a key must not testify about it either."""

        if digest in self._terms_cache:
            return self._terms_cache[digest]
        entry = self._requests[digest]
        terms: Mapping[str, Any] | None = None
        try:
            preimage = reassemble_request_preimage(
                entry.payload, self._resolve_chunk, refs=entry.refs
            )
            if sha256_bytes(preimage) == digest:
                value = json.loads(preimage.decode("utf-8"))
                inner = value.get(entry.generation) if isinstance(value, dict) else None
                if isinstance(inner, Mapping):
                    terms = inner
        except Exception:  # noqa: BLE001 - an unreassemblable candidate is no candidate
            terms = None
        self._terms_cache[digest] = terms
        return terms

    # --- reporting -------------------------------------------------------------------------

    def run_ids(self) -> tuple[str, ...]:
        return tuple(self._run_ids)

    @property
    def damaged_lines(self) -> int:
        return self._damaged

    @property
    def rejected_records(self) -> int:
        return self._rejected

    @property
    def repeated_sources(self) -> int:
        """How many named directories resolved to a corpus this union had already indexed."""

        return self._repeated_sources

    def request_count(self) -> int:
        return len(self._requests)

    def response_count(self) -> int:
        return sum(len(queue) for queue in self._responses.values())
