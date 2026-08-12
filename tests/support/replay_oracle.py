"""Is corpus B a faithful replay of corpus A?

The replay failure class is one shape: an answer belonging to a *different call* is served, and
no surface reports it -- exit 0, ``completed``, ledger success, ``monoid validate`` clean, no
``failure.json``. A recorded answer is a structurally valid model answer, so nothing at
consumption time links a call to its answer except a per-key cursor, and an off-by-one is
undetectable downstream. Four review rounds found six independent routes into it, three of them
introduced by the fix for the route before.

So the check here is deliberately agnostic to *why*: record live into A, replay A while
recording into B, and ask whether B is the same corpus. A cursor that slips changes which
answer a call receives, which changes the next request, which changes B. That covers routes not
yet imagined, which is the only property that has held up over four rounds of naming them one
at a time.

Two things it does **not** cover, stated here because a silently narrow oracle is worse than
none:

* **Route 4 (a source indexed twice) is invisible to it.** Duplicate answers pile up *behind*
  the cursor -- the replay asks each key exactly as often as the recording answered it -- so B
  comes out byte-identical either way. That is a supply-multiplicity defect, not a
  cursor-position one; :func:`assert_supply_conserved` is its separate, cheap oracle.
* **Route 5 (union argument order)** has no derivable right answer, so it is a visibility
  question (``crossed_keys``), not an equivalence one.

Scope is pure replay. Under ``--replay-fallthrough`` a call the inner served is still stamped
with the wrapper's declaration, so a fallthrough corpus is interchangeable with a live recording
only on the branch where the derivation declares. :func:`assert_no_substitution` is the
fallthrough-tolerant half and takes live answers as expected; :func:`assert_pure_replay_equivalent`
is the strict half and must not be pointed at a fallthrough pair.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from monoid_agent_kernel.core._util import sha256_bytes
from monoid_agent_kernel.core.model_payloads import (
    MODEL_PAYLOADS_DIRNAME,
    MODEL_PAYLOADS_FILENAME,
    MODEL_RESPONSE_KIND,
    PAYLOAD_CHUNK_KIND,
    RESPONSE_REFERENCE,
    is_chunk_sha256,
    read_corpus_records,
    response_reference,
)

MASKED = ("run_id", "root_run_id", "recorded_at")
"""The only fields a faithful replay is allowed to differ in.

Everything else is asserted by equality rather than by an allowlist, deliberately: an allowlist
fails open the day a field is added to a record and nobody updates this module, which is the
exact shape ("a rule bound on one of two parallel halves") that produced three of the six
routes. ``call_index`` is **not** masked -- it is activation-local and restarts at 0, so it
differs only across a durable resume, and a fixture that resumes says so by masking it itself.
"""


@dataclass(frozen=True)
class Answer:
    """One recorded answer under one key, at its position in that key's file-order queue."""

    slot: int
    call_index: int
    unrecorded_reason: str
    placement: str
    body: Any


@dataclass(frozen=True)
class Slip:
    """A served answer that did not come from the source slot standing at its position.

    ``source_slot != served_position`` is the failure this module exists to name.

    ``source_slot is None`` is ambiguous on its own, and reading it as "live" was a hole: it
    covers both an answer in no source slot at all (a fallthrough served it live, which is not
    a substitution) *and* a recorded answer served more often than it was recorded, whose slots
    are simply used up. The second is a substitution -- one recorded answer handed to two
    different calls is exactly "an answer belonging to a different call is served" -- so
    ``oversupplied`` separates them rather than leaving the two under one ``None``.
    """

    digest: str
    served_position: int
    source_slot: int | None
    oversupplied: bool = False


@dataclass(frozen=True)
class CorpusView:
    run_dir: Path
    records: tuple[dict[str, Any], ...]
    damaged: tuple[int, ...]
    chunks: Mapping[str, bytes]
    answers: Mapping[str, tuple[Answer, ...]]


def _chunk_map(run_dir: Path, records: Iterable[dict[str, Any]]) -> dict[str, bytes]:
    """Every chunk this corpus can resolve, inline records first, then directory files."""

    chunks: dict[str, bytes] = {}
    # Inline first and first-wins, because that is what ``ReplayCorpus._chunks.setdefault`` does.
    # The oracle read last-wins, so a corpus with two records under one name resolved to a
    # different body here than the code under test resolves -- an oracle disagreeing with the
    # reader about what a chunk IS cannot testify about what the reader served.
    for record in records:
        if record.get("kind") != PAYLOAD_CHUNK_KIND:
            continue
        sha, text = record.get("sha256"), record.get("text")
        if not (isinstance(sha, str) and isinstance(text, str) and is_chunk_sha256(sha)):
            continue
        data = text.encode("utf-8")
        if sha256_bytes(data) != sha:
            continue  # the reader re-hashes before believing a name; so does this
        chunks.setdefault(sha, data)
    directory = run_dir / MODEL_PAYLOADS_DIRNAME
    if directory.is_dir():
        for entry in sorted(directory.iterdir()):
            if entry.is_file():
                chunks.setdefault(entry.name, entry.read_bytes())
    return chunks


def read_corpus(run_dir: Path) -> CorpusView:
    """A corpus as the oracle needs it: verbatim records, plus answers grouped by key.

    Reads through :func:`read_corpus_records` -- the sanctioned shared line reader, the one the
    collector deletes on and the replay reader answers from -- so the oracle cannot disagree
    with the shipped readers about what a line *is*. Placement comes from
    :func:`response_reference`, the same inline/reference/malformed trichotomy the validator
    reports through.
    """

    state, records, damaged = read_corpus_records(run_dir / MODEL_PAYLOADS_FILENAME)
    assert state == "ok", f"{run_dir} corpus is {state}, not readable"
    chunks = _chunk_map(run_dir, records)

    answers: dict[str, list[Answer]] = {}
    for record in records:
        if record.get("kind") != MODEL_RESPONSE_KIND:
            continue
        digest = record.get("request_digest")
        if not isinstance(digest, str) or not is_chunk_sha256(digest):
            continue  # unkeyable: the reader refuses it as a rejected record, not as an answer
        placement, sha = response_reference(record.get("response"))
        body: Any = record.get("response")
        if placement == RESPONSE_REFERENCE and sha is not None:
            raw = chunks.get(sha)
            body = json.loads(raw.decode("utf-8")) if raw is not None else None
        queue = answers.setdefault(digest, [])
        queue.append(
            Answer(
                slot=len(queue),
                call_index=record.get("call_index", -1),
                unrecorded_reason=record.get("unrecorded_reason", ""),
                placement=placement,
                body=body,
            )
        )
    return CorpusView(
        run_dir=run_dir,
        records=tuple(records),
        damaged=tuple(damaged),
        chunks=chunks,
        answers={digest: tuple(queue) for digest, queue in answers.items()},
    )


def _mask(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in MASKED}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=repr)


def structural_diff(a: CorpusView, b: CorpusView) -> list[str]:
    """Where B stops being the same corpus as A, masking only :data:`MASKED`.

    Equality after masking rather than a field-by-field comparison, so it subsumes the whole
    "must be identical" list in one statement -- schema version, kind, digest generation, refs,
    recipe shape, inline chunk sha and text, **inline-vs-offloaded placement** (both predicates
    are pure functions of canonical encoded length, so the same logical body cannot land inline
    in one run and offloaded in another), every recorded turn field, ``unrecorded_reason``, and
    record order -- and stays true when a record grows a field.

    Record equality is not enough on its own, and claiming "chunk sha and text" for the
    offloaded case was wrong: an offloaded body is not in any record, only its reference is.
    Comparing records alone, a replay could write a *completely different answer* under the same
    chunk name, or write no chunk file at all, and still compare equal -- which left the strict
    half weaker than the tolerant one, since ``alignment_report`` resolves through ``chunks``
    and would have caught it. So the bytes are compared too.
    """

    problems: list[str] = []
    if len(a.records) != len(b.records):
        problems.append(f"record count: source {len(a.records)}, replay {len(b.records)}")
    missing = sorted(set(a.chunks) - set(b.chunks))
    extra = sorted(set(b.chunks) - set(a.chunks))
    altered = sorted(sha for sha in set(a.chunks) & set(b.chunks) if a.chunks[sha] != b.chunks[sha])
    if missing or extra or altered:
        problems.append(
            "chunk bytes differ: "
            f"missing from replay {[sha[:12] for sha in missing]}, "
            f"only in replay {[sha[:12] for sha in extra]}, "
            f"contents differ {[sha[:12] for sha in altered]}"
        )
    for index, (left, right) in enumerate(zip(a.records, b.records)):
        masked_left, masked_right = _mask(left), _mask(right)
        if masked_left != masked_right:
            problems.append(
                f"record {index} differs\n  source: {_canonical(masked_left)}"
                f"\n  replay: {_canonical(masked_right)}"
            )
    if b.damaged:
        problems.append(f"replay corpus has damaged lines: {list(b.damaged)}")
    return problems


def masked_field_relations(a: CorpusView, b: CorpusView) -> list[str]:
    """The masked fields still have to hold a relation; masking is not licence.

    A replay run is a *different* run, so both its ids must differ from the source's rather than
    merely being ignored, and its clock must be non-decreasing in file order.

    What this deliberately does **not** claim: the recorder's one-clock-read-per-call rule. That
    rule is that ``model_calls.jsonl`` and ``model_payloads.jsonl`` share a single reading for
    one call, and this module never opens the ledger, so it cannot see the pair. An earlier
    docstring cited it anyway. Non-decreasing stamps are also satisfied by a recorder that read
    the clock once for the whole run, so the check is weaker than the sentence was.
    """

    problems: list[str] = []
    for field in ("run_id", "root_run_id"):
        source_ids = {r.get(field) for r in a.records if field in r}
        replay_ids = {r.get(field) for r in b.records if field in r}
        if len(replay_ids) > 1:
            problems.append(f"replay corpus carries more than one {field}: {sorted(replay_ids)}")
        shared = source_ids & replay_ids
        if shared:
            problems.append(f"replay reused the source's {field}: {sorted(shared)}")
    for label, view in (("source", a), ("replay", b)):
        stamps = [r["recorded_at"] for r in view.records if isinstance(r.get("recorded_at"), str)]
        if stamps != sorted(stamps):
            problems.append(f"{label} recorded_at is not non-decreasing in file order")
    return problems


def alignment_report(a: CorpusView, b: CorpusView) -> list[Slip]:
    """Which of B's answers came from a source slot other than the one at their position.

    Matching is by canonical body encoding, greedy in file order, each source answer accounting
    for at most one served answer. Fixtures must therefore use *distinguishable* answers; an
    ambiguous corpus is rejected rather than silently guessed at.
    """

    slips: list[Slip] = []
    for digest, served in b.answers.items():
        remaining: dict[str, list[int]] = {}
        for answer in a.answers.get(digest, ()):
            remaining.setdefault(_canonical(answer.body), []).append(answer.slot)
        for key, slots in remaining.items():
            if len(slots) > 1:
                raise AssertionError(
                    f"key {digest[:12]} records the same answer in slots {slots}, so no "
                    "alignment can be derived: build fixtures with distinguishable answers. "
                    f"({key[:80]})"
                )
        supplied = set(remaining)
        for position, answer in enumerate(served):
            key = _canonical(answer.body)
            slots = remaining.get(key)
            slot = slots.pop(0) if slots else None
            # A body that matches a source body whose slots are exhausted was served more often
            # than it was recorded. That is oversupply, not a live answer.
            oversupplied = slot is None and key in supplied
            if slot != position or oversupplied:
                slips.append(
                    Slip(
                        digest=digest,
                        served_position=position,
                        source_slot=slot,
                        oversupplied=oversupplied,
                    )
                )
    return slips


def assert_pure_replay_equivalent(source: Path, replay: Path) -> None:
    """B is the same corpus as A. The strict half -- pure replay only, never fallthrough."""

    a, b = read_corpus(source), read_corpus(replay)
    problems = structural_diff(a, b) + masked_field_relations(a, b)
    assert not problems, "the replay did not reproduce its source:\n" + "\n".join(problems)


def assert_no_substitution(source: Path, replay: Path) -> None:
    """No answer of A's was served for a call it does not belong to.

    Tolerates a live answer (a fallthrough served the call) and tolerates B holding fewer
    answers than A (a parked call was never answered). What it refuses is the one shape: a
    recorded answer served at a position that is not its slot.
    """

    a, b = read_corpus(source), read_corpus(replay)
    substitutions = [
        slip for slip in alignment_report(a, b) if slip.source_slot is not None or slip.oversupplied
    ]
    assert not substitutions, (
        "a recorded answer was served for a call it does not belong to: "
        + ", ".join(
            f"key {slip.digest[:12]} position {slip.served_position} was served "
            + (
                "an answer already used up -- one recording served twice"
                if slip.oversupplied
                else f"slot {slip.source_slot}"
            )
            for slip in substitutions
        )
    )


def _distinct_sources(sources: Sequence[Path]) -> dict[str, Path]:
    """Source identity computed *independently of the code under test*.

    ``realpath`` plus the sha256 of the corpus bytes: two names for one directory collapse, and
    two genuinely different directories that happen to hold identical bytes are still counted
    once -- which is the conservative direction, since a corpus is what supplies answers.
    """

    distinct: dict[str, Path] = {}
    for source in sources:
        path = Path(source)
        corpus = path if path.name == MODEL_PAYLOADS_FILENAME else path / MODEL_PAYLOADS_FILENAME
        try:
            data = corpus.read_bytes()
        except OSError as error:
            # Never quieter than the code under test: ``ReplayCorpus.load`` refuses an unreadable
            # source at construction, so an oracle that skipped one would compute ``expected``
            # from fewer sources than were named and pass by counting less. That is how a
            # fixture naming a path that exists only on the author's filesystem stayed green.
            raise AssertionError(
                f"the oracle cannot read a named source, so it cannot count its supply: "
                f"{corpus} ({error})"
            ) from error
        key = f"{os.path.realpath(corpus).casefold()}|{hashlib.sha256(data).hexdigest()}"
        distinct.setdefault(key, path)
    return distinct


def assert_supply_conserved(sources: Sequence[Path], corpus: Any) -> None:
    """Each distinct source contributed its answers exactly once -- route 4's oracle.

    The equivalence oracle cannot see a source indexed twice, because the duplicates sit behind
    the cursor and never get asked for. This counts supply instead of position, and it counts it
    from the files rather than from the reader, so it is agnostic to *why* dedupe failed.
    """

    expected = 0
    for path in _distinct_sources(sources).values():
        view = read_corpus(path)
        expected += sum(len(queue) for queue in view.answers.values())
    assert expected, (
        "the named sources supply no answers at all, so conservation is vacuously true; "
        "the fixture is not testing what it claims to test"
    )
    actual = corpus.response_count()
    diagnosis = (
        "a source was indexed more than once" if actual > expected else "a source is missing"
    )
    assert actual == expected, (
        f"the corpus holds {actual} joinable answer(s) but its "
        f"{len(_distinct_sources(sources))} distinct source(s) supply {expected}: {diagnosis}"
    )
