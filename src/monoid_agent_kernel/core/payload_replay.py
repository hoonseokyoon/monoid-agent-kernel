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
children publish into a reentry queue the parent drains. Two identical background children
therefore issue the *same* key (nothing run-scoped is in it) and divide that key's recordings
between them in whatever order they arrive. Replay is deterministic for a run whose calls are
ordered; a run whose concurrency the recording did not fix is replayed as faithfully as it was
run, which is to say not deterministically. Same family as the spawn-observation limit in
``docs/CLI.md``.

**A union is ordered, and that is not visible from the command line.** "File order" spans the
named sources in the order they were given, and it is decisive wherever two sources can answer
one key -- two recordings of one conversation, and equally a fan-out of two children with the
same definition and the same prompt, which record one key in two run directories for the same
reason the concurrency paragraph above gives: nothing run-scoped is in the key. Disjointness is
a property of the prompts, not of the family shape. This reader is the only place that can see
a key drawing answers from more than one source, so it counts them (``crossed_keys``) for the
preflight to say out loud.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from monoid_agent_kernel.core._util import CANONICAL_JSON_ENCODER, sha256_bytes
from monoid_agent_kernel.core._verified_file import (
    VerifiedFileIdentity,
    file_identity,
    read_verified_bytes,
)
from monoid_agent_kernel.core.json_ingress import loads_json_ingress
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
"""The approved vocabulary (D-i), in one place. Not a partition by producer: ``no_key`` is the
adapter's alone, ``consume`` produces ``absent``, ``not_recorded``, ``exhausted`` and
``generation_mismatch``, and :meth:`ReplayCorpus.diagnose` refines an ``absent`` into any of
``absent``, ``identity_mismatch`` or ``generation_mismatch``. A reason outside this tuple is a
contract change, not a detail."""

# How many diverging prompt terms one diagnosis will name, and how much of each digest.
# Bounds the message, not the comparison -- and a digest prefix is still content-free.
_DIAGNOSED_TERMS = 4
_DIGEST_PREFIX = 12
_IDENTITY_TERMS = ("provider", "model")
"""The terms a *config* change moves, named once so the two diagnoses agree about which they
are: `identity_divergence` compares them by value across the whole corpus, and the term-by-term
branch has to recognise them to say `identity_mismatch` rather than blame the conversation."""


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
    slot: int | None = None
    """The position this refusal is standing on, when a record earned it -- the coordinate
    :meth:`ReplayCorpus.spend_refused` needs to move past exactly that one and no other. An
    integer position discloses nothing; it is the same kind of fact as ``call_index``."""


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


_NAMED_VALUE_CHARS = 120
"""How much of one identity value a diagnosis will quote.

Identity terms are config vocabulary the ledger beside the corpus already records in plaintext,
so naming them discloses nothing new -- but the corpus is untrusted by this module's own threat
model, and ``repr`` of a recorded value has neither a length bound nor a key-count bound. A
recorded ``model.model`` of 200,000 characters produced a 200,043-character miss message, and
that message lands on ``turn.failed``, in ``failure.json``, in ``status.json`` and on stderr,
where nothing downstream truncates. The digest branch twelve lines below has been bounded since
it was written; this is its twin, and it was the unbounded one."""


def _short(text: str) -> str:
    """One corpus-supplied string, bounded. Every string in a message comes from the corpus.

    The bound below covered a *value* and left three other channels in the same sentences
    uncovered: the number of clauses, the term *names* interpolated into them, and the
    identifiers (``run_id``, ``unrecorded_reason``, the generation tags) that carry no value at
    all. Measured: 100,000 model keys produced a 4,477,883-character miss detail, and four long
    term names produced an 800 KB ``status.json`` beside an 800 KB ``failure.json`` and 2.4 MB
    of events. Bounding one of four channels is not bounding the message.
    """

    if len(text) <= _NAMED_VALUE_CHARS:
        return text
    return f"{text[:_NAMED_VALUE_CHARS]}... ({len(text)} chars)"


def _named(value: Any) -> str:
    """One identity value, in plaintext, bounded. Keep the vocabulary, bound the size."""

    return _short(repr(value))


def _where(entry: Any) -> str:
    """The coordinates a miss names. One function, so the four sites cannot drift apart."""

    return f"run {_short(str(entry.run_id))} call_index {entry.call_index}"


def _term_digest(value: Any) -> str:
    try:
        return sha256_bytes(CANONICAL_JSON_ENCODER.encode(value).encode("utf-8"))
    except Exception:  # noqa: BLE001 - an unencodable term still deserves a stable name
        return "unencodable"


class ReplayTake:
    """One take on one key: what the corpus offered, and the settlement it is owed.

    The cursor is the only thing linking a call to its answer, and the two ways a take can go
    unusable settle in **opposite directions**: a refusal standing on a record is spent
    *forward* once the caller served that call another way, while a record handed over and then
    rejected is given *back*. Which of the two you are holding is not visible from the call
    site -- and every one of the routes into the substitution failure was a wrong answer to
    that question, not to "did you remember to settle at all".

    So the choice lives here, in the object that owns the cursor, and the caller declares only
    the fact it actually knows: did the call happen?

        with corpus.take(digest, generation=...) as take:
            ...
            take.served()

    Leaving the block by any other route -- a raise, a return, a rejection -- settles unserved.
    Leaving it having declared *nothing* is a programming error and raises, deliberately: a
    silent default would convert "forgot to declare" into "the same answer served twice",
    which is another silent substitution, and a loud crash in development beats a wrong answer
    at exit 0.
    """

    __slots__ = ("_corpus", "_digest", "hit", "miss", "_declared")

    def __init__(
        self,
        corpus: ReplayCorpus,
        digest: str,
        outcome: ReplayedResponse | ReplayMissReason,
    ) -> None:
        self._corpus = corpus
        self._digest = digest
        self.hit = outcome if isinstance(outcome, ReplayedResponse) else None
        self.miss = outcome if isinstance(outcome, ReplayMissReason) else None
        self._declared: bool | None = None

    def served(self) -> None:
        """The call happened -- by this answer, or live. The conversation moved past this slot."""

        self._declare(True)

    def unserved(self) -> None:
        """The call did not happen, so the next attempt must meet the same position again."""

        self._declare(False)

    def _declare(self, served: bool) -> None:
        if self._declared is not None:
            raise RuntimeError("this take has already been settled")
        self._declared = served
        self._settle(served)

    def _settle(self, served: bool) -> None:
        if served:
            # A hit already advanced the cursor when it was handed over; only a refusal that
            # left the cursor standing has to be stepped past now.
            if self.hit is None and self.miss is not None and self.miss.slot is not None:
                self._corpus.spend_refused(self._digest, self.miss.slot)
        elif self.hit is not None:
            self._corpus.release(self._digest, self.hit.slot)

    def __enter__(self) -> ReplayTake:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._declared is not None:
            return False
        # Marked before settling, the way ``_declare`` does it. Settling first left ``_declared``
        # None if the settle raised, so the take could be settled a second time and in the
        # opposite direction -- a released slot then re-served. The shipped corpus cannot raise
        # here, but this class takes its corpus from a public constructor and its whole premise
        # is that a take is declared exactly once.
        self._declared = False
        self._settle(False)
        if exc_type is None:
            raise RuntimeError(
                "a replay take left its block without saying whether the call happened; "
                "declare served() or unserved() -- the slot has been given back"
            )
        return False


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
        self._unjoinable = 0
        self._repeated_sources = 0
        self._response_source: dict[str, int] = {}
        self._response_root: dict[str, str] = {}
        self._response_run: dict[str, str] = {}
        self._crossed: set[str] = set()
        self._crossed_within_one_run: set[str] = set()
        self._terms_cache: dict[str, Mapping[str, Any] | None] = {}
        self._profiles: tuple[dict[str, Any], ...] | None = None
        self._lock = threading.Lock()

    # --- loading ---------------------------------------------------------------------------

    @classmethod
    def load(cls, run_dirs: Sequence[Path]) -> "ReplayCorpus":
        """Index ``run_dirs`` in argument order; file order within each is preserved.

        Refuses at construction -- not on the tenth turn -- when a named directory yields no
        readable corpus at all: not only a missing file, but any state the reader cannot open
        or verify. A replay source that never recorded is an operator mistake, and every later
        miss it would cause misdirects toward the request rather than the source.

        A file that opens and holds nothing usable is not refused here, because "empty" and
        "damaged" are answers about content rather than about the source, and the reader
        reports them (``damaged_lines``, ``rejected_records``, ``crossed_keys``). The preflight
        warns on those; it does not refuse. A library embedder that skips the preflight gets
        the misses rather than the construction error, and hears nothing at all.
        """

        corpus = cls()
        indexed: set[VerifiedFileIdentity | str] = set()
        for run_dir in run_dirs:
            run_dir = Path(run_dir)
            path = run_dir / MODEL_PAYLOADS_FILENAME
            state, records, damaged = cls._read(path)
            if state != "ok":
                hint = ""
                if Path(run_dir).name == MODEL_PAYLOADS_FILENAME:
                    # The one wrong path an operator reaches by knowing more, not less: they
                    # found the corpus file and named it. Saying `absent` about the file they
                    # are looking at reads as a claim about its contents.
                    hint = f"; name the run directory instead, e.g. {Path(run_dir).parent}"
                raise ValueError(f"replay source has no readable corpus ({state}): {run_dir}{hint}")
            # The chunk directory joins the union even for a repeat: a corpus reached twice
            # is the same references, and a hardlinked copy in another directory keeps its
            # own offloaded bytes. Only the *records* are at risk of being counted twice.
            corpus._chunk_dirs.append(run_dir / MODEL_PAYLOADS_DIRNAME)
            identity = corpus._corpus_identity(path)
            if identity in indexed:
                # One directory named twice -- as an id and as a path, through a link, or
                # simply repeated -- is one source. Indexing it again would append every
                # answer to its queue a second time, and "each answer once" would quietly
                # become "each answer once per spelling": the call that should have earned
                # ``exhausted`` gets a stale recorded body instead.
                corpus._repeated_sources += 1
                continue
            indexed.add(identity)
            corpus._damaged += len(damaged)
            for record in records:
                corpus._index(record, source=len(indexed))
        return corpus

    @staticmethod
    def _corpus_identity(path: Path) -> VerifiedFileIdentity | str:
        """A hashable name for this corpus file: its device/inode pair where the platform
        proves one, its resolved path where the platform does not.

        A zero inode is not evidence. ``payload_gc`` states the rule for this repo and enforces
        it (``provable = bool(approved_directory.inode)``): an inode number is evidence only
        where the platform supplies one, and SMB shares, FAT/exFAT volumes and CPython's
        Windows directory-attribute fallback all report ``0``. Two *distinct* corpora that both
        report ``(0, 0)`` compare equal, so without a gate every source after the first is
        discarded as a repeat -- and on the family union, which ``docs/CLI.md`` documents as
        the required shape for a spawning run, that silently drops the child's answers and
        tells the operator they made a spelling mistake they did not make.

        But falling back to *no* identity only closes that half. The other half is one
        directory named twice on such a volume, which ``load`` calls routine, and there an
        absent identity re-opens the duplicate-indexing defect: every answer joins its queue
        once per spelling, ``repeated_sources`` reads zero so the preflight stays quiet, and
        the call that should have earned ``exhausted`` is handed a stale recording as a real
        turn. A loud loss is a bad trade for a silent wrong answer.

        The resolved, case-folded path is the identity that answers both: it distinguishes two
        corpora and it collapses the spellings ``load`` enumerates -- an id, a path, a symlink,
        a repeat. It does not collapse two hardlinks to one file, which the inode pair does;
        that is the residue, and it costs an over-indexed source on exactly the volumes where
        the loud failure is the one we are trading away from.
        """

        try:
            identity = file_identity(path.lstat())
        except OSError:
            identity = None
        if identity is not None and identity.inode:
            return identity
        return os.path.normcase(os.path.realpath(path))

    def _index(self, record: Mapping[str, Any], *, source: int = 0) -> None:
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
                # The empty digest, and only it. It is legal and deliberate -- a keyless call
                # still records its answer -- but it can never be *asked for* by digest, so it
                # has no queue to join. That is the reader declining to index a healthy record,
                # not damage: counting it as damage made every corpus holding one `too_large`
                # call (an operational condition, not a defect) announce itself as broken to an
                # operator whose `monoid validate` says it is clean.
                #
                # Anything else here is damage by construction. `schemas.py` allows exactly
                # `^(|[0-9a-f]{64})$` and the writer emits only those two shapes, so a
                # non-empty value that is not a name is a record `monoid validate` calls
                # corrupt -- and calling it healthy is the same silence, mirrored: the
                # preflight says nothing, and the miss it causes is diagnosed `absent`, which
                # blames the original call for a failure that is the corpus's.
                if digest == "":
                    self._unjoinable += 1
                else:
                    self._rejected += 1
                return
            call_index = record.get("call_index")
            if isinstance(call_index, bool) or not isinstance(call_index, int):
                self._rejected += 1
                return
            reason = record.get("unrecorded_reason")
            # "File order, each answer once" is a rule about one corpus. Across a union it
            # quietly becomes "the order the sources were named in", and the operator is told
            # nothing: two recordings of the same conversation -- the same prompt run twice, or
            # the crash-and-rerun union `docs/CONTRACTS.md` calls the ordinary durable-resume
            # shape -- interleave their answers by flag order. Reversing two flags then replays
            # a different conversation, or, where one source recorded a refusal at that
            # position and the other the answer, turns a union that demonstrably holds the
            # answer into a miss. This is the only place that can see it happen.
            # Type-checked like every other indexed field. Stringified, a planted 123 or
            # True or ["P"] compares equal across two unrelated sources and fires the
            # family warning, telling the operator to name children "in spawn order" for a
            # parent that never spawned anything -- the misdirection this warning was
            # written to remove.
            raw_root = record.get("root_run_id")
            root = raw_root if isinstance(raw_root, str) else ""
            raw_run = record.get("run_id")
            run = raw_run if isinstance(raw_run, str) else ""
            if self._response_source.setdefault(digest, source) != source:
                self._crossed.add(digest)
                # Which remedy is actionable depends on what crossed. Two recordings of one
                # conversation are two runs, and the operator reorders the flags. Two children
                # of ONE run are a fan-out -- nothing run-scoped is in the key, so identical
                # children issue the same key -- and "reverse the flags" tells them nothing:
                # they have to name the children in the order the parent spawned them.
                # Distinct run ids are part of the test, not a detail: a shared root_run_id is
                # also what a run and an archived COPY of itself have, and `--replay-from` takes
                # a directory or an id, so naming both is an ordinary slip. Two real children
                # always differ in run_id, so nothing true is lost -- but without this a two-turn
                # run with no subagent anywhere in it was told its keys "were recorded by
                # children of one run" and handed a spawn order that does not exist.
                same_run = self._response_run.get(digest) == run
                if root and self._response_root.get(digest) == root and not same_run:
                    self._crossed_within_one_run.add(digest)
            self._response_root.setdefault(digest, root)
            self._response_run.setdefault(digest, run)
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
                # Name the position the refusal stands on, so a caller that later moves past
                # it moves past *this* one -- see spend_refused.
                return replace(body, slot=cursor)
            self._cursors[digest] = cursor + 1
        return ReplayedResponse(
            body=body,
            call_index=entry.call_index,
            recorded_at=entry.recorded_at,
            run_id=entry.run_id,
            slot=cursor,
        )

    def spend_refused(self, digest: str, slot: int) -> None:
        """Advance past the slot ``slot``, once the caller has served that call another way.

        The only honest reason to spend a refused slot: the conversation really has moved past
        the call it belongs to, so the original run's next call and this run's next call are
        at the same position again. Serving the miss live (``--replay-fallthrough``) is that
        reason; parking on it is not.

        It names the slot for the reason :meth:`release` does. ``consume`` deliberately does
        not advance on a refusal, so every concurrent caller meets the *same* refused entry --
        and a blind increment would then spend one slot per caller, skipping recorded answers
        no caller ever sees. Naming it makes duplicate refusals idempotent, which is what they
        are: one slot, refused once, however many callers heard about it.
        """

        with self._lock:
            queue = self._responses.get(digest)
            if not queue:
                return
            if self._cursors.get(digest, 0) == slot < len(queue):
                self._cursors[digest] = slot + 1

    def release(self, digest: str, slot: int) -> None:
        """Give back an answer :meth:`consume` handed out that the caller could not use.

        Only that exact slot, and only while nothing else has moved: under concurrent callers
        another turn may already hold the next one, and rewinding then would hand it out twice.

        Bounded like :meth:`spend_refused`, because `cursor == slot + 1` alone is satisfied by
        ``slot = -1`` against a fresh cursor -- and a negative cursor makes ``consume`` read
        ``queue[-1]``, handing the last recording back as the first call's answer and again at
        the end. Nothing shipped can ask for a negative slot; the public constructor takes a
        corpus by value, so an embedder can.
        """

        with self._lock:
            queue = self._responses.get(digest)
            if not queue:
                return
            if 0 <= slot < len(queue) and self._cursors.get(digest, 0) == slot + 1:
                self._cursors[digest] = slot

    def take(self, digest: str, *, generation: str) -> ReplayTake:
        """A take on the next unconsumed answer, settled by the block that receives it.

        The safe way to reach :meth:`consume`: the settlement becomes a property of leaving the
        block rather than a rule a new call path has to re-attach by hand. Every route into the
        substitution failure so far has been a missed or mistimed settle at a call site, so the
        rule moves to where the cursor lives.

        Holds **no lock across the block**. All the locked work -- including the chunk file I/O
        in :meth:`_entry_body` -- happens inside ``consume`` before this returns, exactly as it
        does for a bare ``consume`` caller, so the block is free to make a provider call.
        """

        return ReplayTake(self, digest, self.consume(digest, generation=generation))

    def _entry_body(self, entry: _ResponseEntry) -> dict[str, Any] | ReplayMissReason:
        """One entry's answer body, materialized and verified -- or the refusal it earns.

        Shared by :meth:`consume` and the cursor-free evidence view, so "what does this
        record actually say" has one answer however it is asked.
        """

        if entry.unrecorded_reason:
            return ReplayMissReason(
                MISS_NOT_RECORDED,
                f"the answer was not recorded ({_short(str(entry.unrecorded_reason))}); "
                f"{_where(entry)}",
            )
        shape, sha = response_reference(entry.response)
        if shape == RESPONSE_MALFORMED:
            return ReplayMissReason(
                MISS_NOT_RECORDED,
                f"a response reference is not a content-addressed name ({_where(entry)})",
            )
        if shape == RESPONSE_REFERENCE:
            assert sha is not None
            try:
                body_bytes = self._resolve_chunk(sha)
                # The same parser the inline half gets through ``read_corpus_records``. One
                # body, two parsers was the gap: an offloaded answer bypassed bounded nesting,
                # unique-keys-after-normalization, bounded ints and the non-finite refusal, and
                # `_reconstruct` checks that ``arguments`` is a dict without walking its
                # values -- so a NaN in recorded tool arguments became a live tool invocation
                # carrying a value that exists in no recording.
                body = loads_json_ingress(body_bytes.decode("utf-8"))
            except Exception as error:  # noqa: BLE001 - every failure is one refusal
                return ReplayMissReason(
                    MISS_NOT_RECORDED,
                    f"the recorded answer could not be resolved ({_short(str(error))}); "
                    f"{_where(entry)}",
                )
        else:
            body = entry.response
        if not isinstance(body, dict):
            return ReplayMissReason(
                MISS_NOT_RECORDED,
                f"the recorded answer is not an object; {_where(entry)}",
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

    def _no_answer_reason(self) -> ReplayMissReason:
        """Why a key has a request record and no answer -- one sentence, both askers.

        It was hand-written twice, and only the copy in ``_absent_locked`` was pinned while the
        copy callers actually see is ``diagnose``'s: the adapter always refines a ``MISS_ABSENT``
        through ``diagnose``, so that is the string reaching ``turn.failed``, ``failure.json``
        and stderr. Softening one and leaving the other is how a repair buys half of what it
        promises.

        And the parenthetical widens when it must. Blaming the original call is right only when
        the answer was never written; where the reader *rejected* records or the file has
        damaged lines, the answer may well have been recorded and this corpus simply cannot
        present it -- a distinction the operator acts on differently, and which otherwise
        appears nowhere but a stderr warning that no run directory retains.
        """

        detail = "a request record exists under this key but no answer was recorded "
        if self._rejected or self._damaged:
            return ReplayMissReason(
                MISS_ABSENT,
                detail + "and this corpus is damaged "
                f"({self._damaged} unparseable line(s), {self._rejected} rejected record(s)), "
                "so an answer that was recorded may be among what could not be read",
            )
        return ReplayMissReason(
            MISS_ABSENT,
            detail + "(the original call failed, or its activation ended before answering)",
        )

    def _absent_locked(self, digest: str, generation: str) -> ReplayMissReason:
        if digest in self._requests:
            return self._no_answer_reason()
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
            recorded = _short(", ".join(sorted(self._generations)))
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
            clauses.append(
                f"provider recorded {_named(expected['provider'])}, computing {_named(provider)}"
            )
        recorded_model = expected["model"] if isinstance(expected["model"], dict) else {}
        live_model = model if isinstance(model, dict) else {}
        differing = [
            name
            for name in sorted(set(recorded_model) | set(live_model))
            if recorded_model.get(name) != live_model.get(name)
        ]
        # Capped like the term-by-term branch below, which has been capped since it was written.
        # This one iterated the union of two corpus-supplied key sets, so the clause COUNT was
        # the unbounded channel even once each value was bounded -- and the key names went in raw.
        for name in differing[:_DIAGNOSED_TERMS]:
            clauses.append(
                f"model.{_short(str(name))} recorded {_named(recorded_model.get(name))}, "
                f"computing {_named(live_model.get(name))}"
            )
        if len(differing) > _DIAGNOSED_TERMS:
            clauses.append(f"and {len(differing) - _DIAGNOSED_TERMS} more model term(s)")
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
            return self._no_answer_reason()
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
        return self._closest_divergence(live_terms)

    def _closest_divergence(self, live_terms: Mapping[str, Any]) -> ReplayMissReason:
        live_digests = {name: _term_digest(value) for name, value in live_terms.items()}
        best: tuple[int, str, dict[str, str], Mapping[str, Any]] | None = None
        for digest in self._requests:
            terms = self._request_terms(digest)
            if terms is None:
                continue
            recorded = {name: _term_digest(value) for name, value in terms.items()}
            # Closeness is scored over the CONVERSATION only. Scoring identity terms too let a
            # same-identity record tie with the identity-diverging one the call would actually
            # have used, and a strict `>` then hands the tie to whichever came first in file
            # order -- so the diagnosis reported `absent` with "identity matches" about a call
            # recorded under a different model, which is verbatim the failure the identity
            # branch below exists to remove. A file position is not a semantic tie-break.
            matches = sum(
                1
                for name in live_digests
                if name not in _IDENTITY_TERMS and recorded.get(name) == live_digests[name]
            )
            if best is None or matches > best[0]:
                best = (matches, self._requests[digest].run_id, recorded, terms)
        if best is None:
            return ReplayMissReason(
                MISS_ABSENT,
                "identity matches but the corpus holds no reassemblable request to compare",
            )
        _matches, run_id, recorded, recorded_terms = best
        diverging = sorted(
            name
            for name in set(live_digests) | set(recorded)
            if live_digests.get(name) != recorded.get(name)
        )
        identity_terms = [name for name in _IDENTITY_TERMS if name in diverging]
        # The conversation half is built once and used by both exits: an identity divergence
        # and a diverged conversation are independent facts about one comparison, and a call
        # can have both.
        conversation = [name for name in diverging if name not in _IDENTITY_TERMS]
        named = conversation[:_DIAGNOSED_TERMS]
        # The term NAME is corpus-supplied and needs the same bound the identity half below
        # already applies. It was left raw when that half was fixed -- one of two sibling
        # clause builders in one function -- and a hostile corpus reached 800,239 characters
        # through it, into failure.json, status.json and stderr. The digests either side are
        # bounded by their slices.
        clauses = [
            f"{_short(str(name))} live={live_digests.get(name, 'missing')[:_DIGEST_PREFIX]} "
            f"recorded={recorded.get(name, 'missing')[:_DIGEST_PREFIX]}"
            for name in named
        ]
        more = (
            f" and {len(conversation) - len(named)} more" if len(conversation) > len(named) else ""
        )
        if identity_terms:
            # `identity_divergence` answered "some recorded identity matches this run", which
            # is a different question from "the record this call would have used was recorded
            # under it". In a union of identities the first is true and the second false, and
            # the answer is a config change, not a diverged conversation -- so it earns the
            # reason that says so, and names the identity in the plaintext the preflight uses.
            # (Identity terms are config vocabulary the ledger already records in the clear;
            # everything below this branch stays digests.)
            identity_clauses = [
                f"{_short(str(name))} recorded {_named(recorded_terms.get(name))}, "
                f"computing {_named(live_terms.get(name))}"
                for name in identity_terms
            ]
            detail = (
                "this run's config reaches an identity the corpus recorded, but not the one "
                f"the closest recorded request (run {_short(str(run_id))}) was recorded under: "
                + "; ".join(identity_clauses)
            )
            if clauses:
                # Appended rather than swallowed by an early return. Reporting only the
                # identity sends the operator to fix the model, re-run, and earn `absent` for
                # a conversation term they were never told about -- two round trips for one
                # diagnosis the corpus could give at once.
                detail += "; the conversation diverges too: " + "; ".join(clauses) + more
            return ReplayMissReason(MISS_IDENTITY_MISMATCH, detail)
        return ReplayMissReason(
            MISS_ABSENT,
            "identity matches; diverging terms vs the closest recorded request "
            f"(run {_short(str(run_id))}): " + "; ".join(clauses) + more,
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
                # The reassembled preimage is corpus bytes too, and this projection feeds the
                # impersonation derivation and the CLI preflight -- the twin of the body site
                # above, and the one a census of "where does a corpus become objects" finds
                # while reading the response path alone does not.
                value = loads_json_ingress(preimage.decode("utf-8"))
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
    def unjoinable_records(self) -> int:
        """Healthy records the reader cannot index by key -- today, the deliberate answer of a
        keyless call. Separate from :attr:`rejected_records` because one is the corpus being
        well-formed about a call that had no key, and the other is the corpus being wrong."""

        return self._unjoinable

    @property
    def repeated_sources(self) -> int:
        """How many named directories resolved to a corpus this union had already indexed."""

        return self._repeated_sources

    @property
    def crossed_keys(self) -> int:
        """How many keys more than one source recorded an answer for.

        Non-zero means the answer a call gets depends on the order the sources were named in,
        which no rule this reader states makes visible.

        A family union is **not** exempt, though it is the documented multi-source shape.
        Nothing run-scoped is in the key -- the same fact the concurrency paragraph above rests
        on -- so two children with the same definition and the same prompt record one key in
        two run directories, and the order the flags were given decides which child gets which
        answer. A family whose children are distinguishable by prompt reads zero here; an
        ordinary fan-out of identical children does not.
        """

        return len(self._crossed)

    @property
    def crossed_within_one_run(self) -> int:
        """How many of those crossed keys were recorded by children of the *same* run.

        The distinction is about which remedy is actionable, not about severity. Two recordings
        of one conversation are two runs and the operator reorders the flags; two children of
        one run are a fan-out, and the flags have to be given in the order the parent spawned
        them -- an order that is not recoverable from the run ids, which are minted hex.
        """

        return len(self._crossed_within_one_run)

    def request_count(self) -> int:
        return len(self._requests)

    def response_count(self) -> int:
        """Joinable answer *records*, refused ones included.

        Deliberately not the same number as ``len(response_bodies_view())``, which resolves
        bodies and therefore skips every record whose body cannot be given back. A corpus with
        one refused record reads 2 here and 1 there, and an embedder comparing them without
        this sentence concludes the corpus lost records. This one counts supply -- what a cursor
        can be asked for; that one counts evidence -- what can be read.
        """

        return sum(len(queue) for queue in self._responses.values())
