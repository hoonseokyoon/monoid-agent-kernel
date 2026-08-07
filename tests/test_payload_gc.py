"""Contract tests for the chunk-directory collector (W6-3).

The collector's one fatal mistake is deleting a chunk some record still lets a reader resolve, so
these tests bind its ingredients to the artifacts they must agree with: the reference walker to
the writer that produces references and to the validator that resolves them, the temp-name
predicate to the writer that authors temporary names, and (from the collector on) every deletion
to ``validate_run_dir`` reporting exactly what it reported before.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from monoid_agent_kernel.core import schemas
from monoid_agent_kernel.core._util import CANONICAL_JSON_ENCODER
from monoid_agent_kernel.core._verified_file import write_once_temp_stem
from monoid_agent_kernel.core.model_io import ModelCallReceipt
from monoid_agent_kernel.core.model_payloads import (
    MODEL_PAYLOADS_DIRNAME,
    MODEL_PAYLOADS_FILENAME,
    PAYLOAD_CHUNK_REF_KEY,
    PAYLOAD_OFFLOAD_THRESHOLD_BYTES,
    chunk_marker,
    chunk_record,
    corpus_keep_set,
    iter_chunk_references,
    model_request_record,
    model_response_record,
    split_request_payload,
)
from monoid_agent_kernel.core.payload_gc import PayloadGcEntry, PayloadGcReport, collect_payload_garbage
from monoid_agent_kernel.core.schemas import validate_run_dir
from monoid_agent_kernel.model_call import SettledModelCall
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.recorder import AgentRecorder

_GENERATION = "monoid.model-request-digest.v1"
_ENVELOPE = {
    "run_id": "run-1",
    "root_run_id": "run-1",
    "recorded_at": "2026-08-07T00:00:00Z",
}


def _preimage(payload: dict) -> tuple[bytes, str]:
    encoded = CANONICAL_JSON_ENCODER.encode(payload).encode("utf-8")
    return encoded, hashlib.sha256(encoded).hexdigest()


def _write_corpus(base: Path, records: list[dict]) -> Path:
    path = base / MODEL_PAYLOADS_FILENAME
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


# --- The reference walker -------------------------------------------------------------------------


def test_the_walker_names_every_reference_a_real_writer_produces() -> None:
    """The keep-set's floor: whatever the splitter lifted and whatever a response record offloaded
    must come back out of the walker, or the collector deletes a chunk the corpus still resolves."""
    payload = {
        _GENERATION: {
            "system_prompt": "s" * 4096,
            "tools": [{"name": "big", "description": "d" * 512}],
            "messages": [{"role": "user", "content": "c" * 300}],
        }
    }
    preimage, digest = _preimage(payload)
    split = split_request_payload(preimage, digest)
    assert split is not None and split.refs and split.chunks
    request = model_request_record(
        split.payload,
        refs=True,
        request_digest=digest,
        digest_generation=_GENERATION,
        **_ENVELOPE,
    )
    response_sha = "ab" * 32
    response = model_response_record(
        chunk_marker(response_sha),
        call_index=0,
        request_digest=digest,
        unrecorded_reason="",
        **_ENVELOPE,
    )

    walked = set(iter_chunk_references(request)) | set(iter_chunk_references(response))

    assert walked == set(split.chunks) | {response_sha}


def test_the_walker_over_keeps_refs_false_lookalikes_and_ignores_strings() -> None:
    """A verbatim payload's marker lookalike is data to every reader, and the walker still counts
    it: naming a sha no reader resolves keeps a file (bounded waste), missing one a reader
    resolves deletes a referenced chunk. Bare sha-shaped strings are never references -- the
    ledger's ``request_digest`` sits one field over from real markers and must not pin files --
    and a marker whose sha is malformed names nothing a content-addressed directory could hold."""
    sha = "cd" * 32
    verbatim = model_request_record(
        {
            "note": chunk_marker(sha),
            "digest_lookalike": "ef" * 32,
            "not_a_marker": {PAYLOAD_CHUNK_REF_KEY: "not-a-sha"},
            "wider_than_a_marker": {PAYLOAD_CHUNK_REF_KEY: "ab" * 32, "extra": 1},
        },
        refs=False,
        request_digest="12" * 32,
        digest_generation=_GENERATION,
        **_ENVELOPE,
    )

    assert set(iter_chunk_references(verbatim)) == {sha}


def test_the_keep_set_adds_inline_chunk_names_the_walker_cannot_see() -> None:
    """A ``chunk`` record's ``sha256`` names an inline body, not a file -- but a collector keying
    only on markers would treat a same-named file as garbage, and keeping an unreachable shadow is
    the cheap side of that asymmetry."""
    chunk = CANONICAL_JSON_ENCODER.encode({"k": "v"}).encode("utf-8")
    record = chunk_record(chunk, **_ENVELOPE)

    assert set(iter_chunk_references(record)) == set()
    assert corpus_keep_set([record]) == {record["sha256"]}


# --- The temp-name predicate ----------------------------------------------------------------------


def test_the_temp_name_predicate_matches_exactly_what_the_writer_creates() -> None:
    """The collector may only sweep litter the write-once writer demonstrably minted, so the
    predicate is authored beside the f-string that mints the shape and rejects every near miss --
    anything else in the directory is foreign and stays."""
    sha = "9" * 64
    real = f"{sha}.{os.getpid()}.{os.urandom(6).hex()}.tmp"
    assert write_once_temp_stem(real) == sha
    assert write_once_temp_stem(f"a.b.{os.getpid()}.{'0' * 12}.tmp") == "a.b"

    for near_miss in (
        f"{sha}.{os.getpid()}.{'a' * 12}.tmp2",
        f"{sha}.{os.getpid()}.{'a' * 11}.tmp",
        f"{sha}.{os.getpid()}.{'A' * 12}.tmp",
        f"{sha}..{'a' * 12}.tmp",
        f"{sha}.pid.{'a' * 12}.tmp",
        f"{'a' * 12}.tmp",
        sha,
        "",
    ):
        assert write_once_temp_stem(near_miss) is None, near_miss


# --- Binding the keep-set to the validator's resolution -------------------------------------------


def test_every_sha_the_validator_resolves_from_the_directory_is_in_the_keep_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The validator and the keep-set are two readers of one corpus, and this is the direction
    that must hold between them: every directory file the validator's ``resolve`` reads -- request
    reassembly and response references both funnel through it -- is a file the keep-set retains.
    Bound by observation rather than by sharing code, because the validator's response arm is
    deliberately laxer (a malformed sha must reach ``resolve`` and fail loudly), a laxness a
    keep-set must not inherit."""
    big = "x" * (PAYLOAD_OFFLOAD_THRESHOLD_BYTES + 1024)
    payload = {
        _GENERATION: {
            "system_prompt": "s",
            "tools": [],
            "messages": [{"role": "user", "content": big}],
        }
    }
    preimage, digest = _preimage(payload)
    split = split_request_payload(preimage, digest)
    assert split is not None and split.refs
    chunk_dir = tmp_path / MODEL_PAYLOADS_DIRNAME
    chunk_dir.mkdir()
    inline_records: list[dict] = []
    for sha, chunk in split.chunks.items():
        if len(chunk) > PAYLOAD_OFFLOAD_THRESHOLD_BYTES:
            (chunk_dir / sha).write_bytes(chunk)
        else:
            inline_records.append(chunk_record(chunk, **_ENVELOPE))
    response_bytes = CANONICAL_JSON_ENCODER.encode(
        {"final_text": "y" * (PAYLOAD_OFFLOAD_THRESHOLD_BYTES + 1024)}
    ).encode("utf-8")
    response_sha = hashlib.sha256(response_bytes).hexdigest()
    (chunk_dir / response_sha).write_bytes(response_bytes)
    records = [
        *inline_records,
        model_request_record(
            split.payload,
            refs=True,
            request_digest=digest,
            digest_generation=_GENERATION,
            **_ENVELOPE,
        ),
        model_response_record(
            chunk_marker(response_sha),
            call_index=0,
            request_digest=digest,
            unrecorded_reason="",
            **_ENVELOPE,
        ),
    ]
    _write_corpus(tmp_path, records)

    asked: list[str] = []
    real_reader = schemas.read_verified_bytes

    def spy(path: Path, *, max_bytes: int) -> bytes | None:
        asked.append(path.name)
        return real_reader(path, max_bytes=max_bytes)

    monkeypatch.setattr(schemas, "read_verified_bytes", spy)
    issues = validate_run_dir(tmp_path)

    assert asked, "the validator resolved nothing from the directory; this bind would be vacuous"
    assert set(asked) <= corpus_keep_set(records)
    assert not any(issue.path.startswith(MODEL_PAYLOADS_FILENAME) for issue in issues)


# --- The collector --------------------------------------------------------------------------------

_OLD_NS = 1_000_000_000_000_000_000  # 2001-09-09; far past any min-age this file uses
_DAY_S = 86_400.0


def _recorder(base: Path, *, reopen: bool = False) -> AgentRecorder:
    return AgentRecorder(
        base / "runs",
        "run-1",
        status_file=False,
        model_calls_file=True,
        model_payload_file=True,
        reopen=reopen,
    )


def _offloadable() -> tuple[bytes, str, str]:
    payload = {
        _GENERATION: {
            "system_prompt": "s" * (PAYLOAD_OFFLOAD_THRESHOLD_BYTES + 4096),
            "tools": [],
            "messages": [],
        }
    }
    preimage, digest = _preimage(payload)
    split = split_request_payload(preimage, digest)
    assert split is not None
    sha = next(s for s, c in split.chunks.items() if len(c) > PAYLOAD_OFFLOAD_THRESHOLD_BYTES)
    return preimage, digest, sha


def _recorded_run(base: Path) -> tuple[Path, str]:
    """One real run whose corpus references exactly one directory chunk; (run_dir, its sha)."""
    preimage, digest, sha = _offloadable()
    recorder = _recorder(base)
    recorder.record_settled_call(
        SettledModelCall(
            receipt=ModelCallReceipt(request_digest=digest, digest_generation=_GENERATION),
            request_preimage=preimage,
            turn=ModelTurn(response_id="r", final_text="answer"),
        )
    )
    recorder.close()
    return recorder.run_dir, sha


def _backdate(path: Path) -> None:
    os.utime(path, ns=(_OLD_NS, _OLD_NS))


def _issues(run_dir: Path) -> list[tuple[str, str]]:
    return sorted((issue.path, issue.message) for issue in validate_run_dir(run_dir))


def _entry(report: PayloadGcReport, name: str) -> PayloadGcEntry:
    matches = [entry for entry in report.entries if entry.name == name]
    assert len(matches) == 1, (name, report.entries)
    return matches[0]


def test_a_real_offloaded_run_has_zero_orphans_and_gc_touches_nothing(tmp_path: Path) -> None:
    """The baseline the whole verb rests on: a healthy corpus yields no candidates, and even
    ``--apply`` with no age gate at all deletes nothing, because protection of a referenced chunk
    is membership in the keep-set, never its age."""
    run_dir, sha = _recorded_run(tmp_path)
    before = _issues(run_dir)

    report = collect_payload_garbage(run_dir, min_age_s=0.0, apply=True)

    kept = _entry(report, sha)
    assert kept.classification == "kept" and kept.deleted is False
    assert report.chunk_dir_state == "ok" and report.corpus_state == "ok"
    assert report.candidate_bytes == 0 and report.reclaimed_bytes == 0
    assert (run_dir / MODEL_PAYLOADS_DIRNAME / sha).exists()
    assert _issues(run_dir) == before


def test_an_unreferenced_chunk_file_is_collected_and_validate_cannot_tell(tmp_path: Path) -> None:
    """The headline behavior, with its oracle: an unreferenced file is exactly what the validator
    tolerates by contract, so removing it must leave ``validate_run_dir`` reporting precisely
    what it reported before."""
    run_dir, sha = _recorded_run(tmp_path)
    orphan = run_dir / MODEL_PAYLOADS_DIRNAME / ("f" * 64)
    orphan.write_bytes(b"j" * 4096)
    _backdate(orphan)
    before = _issues(run_dir)

    report = collect_payload_garbage(run_dir, min_age_s=_DAY_S, apply=True)

    entry = _entry(report, "f" * 64)
    assert entry.classification == "orphan" and entry.deleted is True and entry.error == ""
    assert report.reclaimed_bytes == 4096
    assert not orphan.exists()
    assert (run_dir / MODEL_PAYLOADS_DIRNAME / sha).exists()
    assert _issues(run_dir) == before


def test_a_referenced_chunk_survives_whatever_its_age(tmp_path: Path) -> None:
    """Age gates candidates; it never nominates them. A referenced chunk backdated to 2001 with a
    zero min-age is still not garbage."""
    run_dir, sha = _recorded_run(tmp_path)
    stored = run_dir / MODEL_PAYLOADS_DIRNAME / sha
    _backdate(stored)
    before = _issues(run_dir)

    report = collect_payload_garbage(run_dir, min_age_s=0.0, apply=True)

    assert _entry(report, sha).classification == "kept"
    assert stored.exists()
    assert _issues(run_dir) == before


def test_a_young_orphan_is_spared_by_min_age(tmp_path: Path) -> None:
    """The adoption shield: an orphan younger than the gate may be a chunk a writer just stored
    or just re-derived (adoption refreshes times), so youth is protection -- reported, counted as
    nothing, left in place."""
    run_dir, _sha = _recorded_run(tmp_path)
    orphan = run_dir / MODEL_PAYLOADS_DIRNAME / ("e" * 64)
    orphan.write_bytes(b"j" * 128)

    report = collect_payload_garbage(run_dir, min_age_s=_DAY_S, apply=True)

    entry = _entry(report, "e" * 64)
    assert entry.classification == "orphan" and entry.deleted is False and entry.error == ""
    assert report.candidate_bytes == 0
    assert orphan.exists()


def test_a_dead_writers_old_temp_goes_and_a_young_one_stays(tmp_path: Path) -> None:
    """The cross-pid litter the recorder's own sweep deliberately leaves (its temp names carry a
    pid, and a pid cannot prove death): an old temporary is collected here, a young one is left
    for the writer that may be mid-``os.replace`` with it."""
    run_dir, _sha = _recorded_run(tmp_path)
    chunk_dir = run_dir / MODEL_PAYLOADS_DIRNAME
    dead = chunk_dir / f"{'a' * 64}.{os.getpid() + 1}.{'0' * 12}.tmp"
    dead.write_bytes(b"partial")
    _backdate(dead)
    fresh = chunk_dir / f"{'b' * 64}.{os.getpid() + 1}.{'1' * 12}.tmp"
    fresh.write_bytes(b"partial")

    report = collect_payload_garbage(run_dir, min_age_s=_DAY_S, apply=True)

    gone = _entry(report, dead.name)
    assert gone.classification == "temp" and gone.deleted is True
    assert not dead.exists()
    young = _entry(report, fresh.name)
    assert young.classification == "temp" and young.deleted is False
    assert fresh.exists()


def test_foreign_entries_are_never_touched(tmp_path: Path) -> None:
    """The collector deletes only what the corpus writer demonstrably mints: 64-lowercase-hex
    regular files and its temporary shape over a sha stem. Everything else in the directory is
    somebody else's, however old, and no age or flag changes that."""
    run_dir, _sha = _recorded_run(tmp_path)
    chunk_dir = run_dir / MODEL_PAYLOADS_DIRNAME
    foreign_names = []
    (chunk_dir / "subdir").mkdir()
    foreign_names.append("subdir")
    stray = chunk_dir / "README.txt"
    stray.write_bytes(b"note")
    _backdate(stray)
    foreign_names.append("README.txt")
    upper = chunk_dir / ("F" * 64 + "x")
    upper.write_bytes(b"x")
    _backdate(upper)
    foreign_names.append(upper.name)
    odd_temp = chunk_dir / f"not-a-sha.{os.getpid()}.{'2' * 12}.tmp"
    odd_temp.write_bytes(b"x")
    _backdate(odd_temp)
    foreign_names.append(odd_temp.name)
    try:
        os.symlink(stray, chunk_dir / ("d" * 64))
        foreign_names.append("d" * 64)
    except OSError:
        pass  # symlink privilege is optional; the other four shapes still cover the rule

    report = collect_payload_garbage(run_dir, min_age_s=0.0, apply=True)

    for name in foreign_names:
        entry = _entry(report, name)
        assert entry.classification == "foreign" and entry.deleted is False, name
        (chunk_dir / name).lstat()  # still present, whatever it is


def test_chunks_referenced_only_by_damaged_lines_are_collectable_and_the_lines_are_named(
    tmp_path: Path,
) -> None:
    """D-d: a line no reader parses references nothing any reader can reach -- the validator
    skips it, the replay reader will skip it -- so the files only it referenced are garbage. The
    damage itself is not silent: the report names the line numbers."""
    big = "x" * (PAYLOAD_OFFLOAD_THRESHOLD_BYTES + 1024)
    kept_payload = {
        _GENERATION: {
            "system_prompt": "s",
            "tools": [],
            "messages": [{"role": "user", "content": big}],
        }
    }
    kept_pre, kept_digest = _preimage(kept_payload)
    kept_split = split_request_payload(kept_pre, kept_digest)
    assert kept_split is not None and kept_split.refs
    lost_payload = {
        _GENERATION: {
            "system_prompt": "s2",
            "tools": [],
            "messages": [{"role": "user", "content": "z" + big}],
        }
    }
    lost_pre, lost_digest = _preimage(lost_payload)
    lost_split = split_request_payload(lost_pre, lost_digest)
    assert lost_split is not None and lost_split.refs

    chunk_dir = tmp_path / MODEL_PAYLOADS_DIRNAME
    chunk_dir.mkdir()
    for split in (kept_split, lost_split):
        for sha, chunk in split.chunks.items():
            if len(chunk) > PAYLOAD_OFFLOAD_THRESHOLD_BYTES:
                (chunk_dir / sha).write_bytes(chunk)
                _backdate(chunk_dir / sha)
    kept_record = model_request_record(
        kept_split.payload,
        refs=True,
        request_digest=kept_digest,
        digest_generation=_GENERATION,
        **_ENVELOPE,
    )
    lost_record = model_request_record(
        lost_split.payload,
        refs=True,
        request_digest=lost_digest,
        digest_generation=_GENERATION,
        **_ENVELOPE,
    )
    lost_line = json.dumps(lost_record, ensure_ascii=False)
    (tmp_path / MODEL_PAYLOADS_FILENAME).write_text(
        json.dumps(kept_record, ensure_ascii=False) + "\n"
        + "\n"
        + lost_line[: len(lost_line) // 2] + "\n",
        encoding="utf-8",
    )
    before = _issues(tmp_path)

    report = collect_payload_garbage(tmp_path, min_age_s=_DAY_S, apply=True)

    assert report.damaged_lines == (3,)
    kept_shas = {
        s for s, c in kept_split.chunks.items() if len(c) > PAYLOAD_OFFLOAD_THRESHOLD_BYTES
    }
    lost_shas = {
        s for s, c in lost_split.chunks.items() if len(c) > PAYLOAD_OFFLOAD_THRESHOLD_BYTES
    }
    for sha in kept_shas:
        assert (chunk_dir / sha).exists() and _entry(report, sha).classification == "kept"
    for sha in lost_shas:
        assert not (chunk_dir / sha).exists() and _entry(report, sha).deleted is True
    assert _issues(tmp_path) == before


def test_gc_and_the_validator_agree_which_lines_are_damaged(tmp_path: Path) -> None:
    """Two lenient readers of one file must skip the same lines for the same reasons: blank lines
    silently, everything unparseable-or-not-an-object loudly. Every line the collector calls
    damaged is a line the validator flags."""
    record = chunk_record(CANONICAL_JSON_ENCODER.encode({"k": "v"}).encode("utf-8"), **_ENVELOPE)
    (tmp_path / MODEL_PAYLOADS_FILENAME).write_text(
        json.dumps(record, ensure_ascii=False) + "\n"
        + "not json at all\n"
        + "\n"
        + "[1, 2, 3]\n"
        + '{"kind": "chunk", torn',
        encoding="utf-8",
    )

    report = collect_payload_garbage(tmp_path, min_age_s=_DAY_S, apply=False)
    issues = validate_run_dir(tmp_path)

    assert report.damaged_lines == (2, 4, 5)
    flagged = {issue.path for issue in issues}
    assert {
        f"{MODEL_PAYLOADS_FILENAME}:{index}" for index in report.damaged_lines
    } <= flagged


def test_a_symlinked_corpus_refuses_content_collection_but_still_sweeps_temps(
    tmp_path: Path,
) -> None:
    """The collector deletes on the strength of what it read, so it re-establishes the writer's
    open discipline: a corpus reached through a planted link is not this run's corpus, and no
    chunk-shaped file gets judged from it. Temporaries need no corpus -- no record ever
    references one -- so the litter half still runs."""
    real = tmp_path / "elsewhere.jsonl"
    real.write_text("", encoding="utf-8")
    try:
        os.symlink(real, tmp_path / MODEL_PAYLOADS_FILENAME)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")
    chunk_dir = tmp_path / MODEL_PAYLOADS_DIRNAME
    chunk_dir.mkdir()
    chunk = chunk_dir / ("c" * 64)
    chunk.write_bytes(b"x" * 64)
    _backdate(chunk)
    temp = chunk_dir / f"{'a' * 64}.{os.getpid() + 1}.{'0' * 12}.tmp"
    temp.write_bytes(b"partial")
    _backdate(temp)

    report = collect_payload_garbage(tmp_path, min_age_s=_DAY_S, apply=True)

    assert report.corpus_state == "unreadable"
    assert _entry(report, "c" * 64).classification == "unjudged"
    assert chunk.exists()
    assert _entry(report, temp.name).deleted is True and not temp.exists()


def test_a_corpus_that_is_not_a_regular_file_refuses_content_collection(tmp_path: Path) -> None:
    """The privilege-free twin of the symlinked-corpus case (a directory needs no symlink
    right to plant): anything at the corpus name that is not this run's own regular file means
    no chunk-shaped file gets judged, while the corpus-free temp half still runs."""
    (tmp_path / MODEL_PAYLOADS_FILENAME).mkdir()
    chunk_dir = tmp_path / MODEL_PAYLOADS_DIRNAME
    chunk_dir.mkdir()
    chunk = chunk_dir / ("b" * 64)
    chunk.write_bytes(b"x" * 64)
    _backdate(chunk)
    temp = chunk_dir / f"{'a' * 64}.{os.getpid() + 1}.{'0' * 12}.tmp"
    temp.write_bytes(b"partial")
    _backdate(temp)

    report = collect_payload_garbage(tmp_path, min_age_s=_DAY_S, apply=True)

    assert report.corpus_state == "unreadable"
    assert _entry(report, "b" * 64).classification == "unjudged"
    assert chunk.exists()
    assert _entry(report, temp.name).deleted is True and not temp.exists()


def test_an_absent_corpus_beside_stored_chunks_is_a_refusal_not_a_purge(tmp_path: Path) -> None:
    """No corpus, chunk-shaped files present: mutilation and a first-call crash whose very first
    chunk was directory-sized leave this same state (the chunk file lands before the corpus
    file's lazy create), so the collector refuses the judgment and names the state rather than
    inventing an empty keep-set and purging."""
    chunk_dir = tmp_path / MODEL_PAYLOADS_DIRNAME
    chunk_dir.mkdir()
    chunk = chunk_dir / ("9" * 64)
    chunk.write_bytes(b"x" * 256)
    _backdate(chunk)
    temp = chunk_dir / f"{'8' * 64}.{os.getpid() + 1}.{'0' * 12}.tmp"
    temp.write_bytes(b"partial")
    _backdate(temp)

    report = collect_payload_garbage(tmp_path, min_age_s=_DAY_S, apply=True)

    assert report.corpus_state == "absent"
    assert _entry(report, "9" * 64).classification == "unjudged" and chunk.exists()
    assert _entry(report, temp.name).deleted is True and not temp.exists()


def test_a_hardlinked_orphan_is_deleted_without_touching_the_other_names_bytes(
    tmp_path: Path,
) -> None:
    """Deleting one name of a multiply-linked inode removes the name, never the bytes behind the
    archive's other name -- the same doctrine that lets the reader accept hardlinked chunks."""
    run_dir, _sha = _recorded_run(tmp_path)
    orphan = run_dir / MODEL_PAYLOADS_DIRNAME / ("7" * 64)
    orphan.write_bytes(b"archived" * 64)
    archive = tmp_path / "archive.bin"
    os.link(orphan, archive)
    _backdate(orphan)

    report = collect_payload_garbage(run_dir, min_age_s=_DAY_S, apply=True)

    assert _entry(report, "7" * 64).deleted is True
    assert not orphan.exists()
    assert archive.read_bytes() == b"archived" * 64


def test_one_undeletable_entry_costs_itself_not_the_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows keeps a file another process holds open; the answer is a per-entry error and a
    finished sweep, never an abort that leaves the rest of the litter uncollected."""
    run_dir, _sha = _recorded_run(tmp_path)
    chunk_dir = run_dir / MODEL_PAYLOADS_DIRNAME
    stuck = chunk_dir / ("5" * 64)
    stuck.write_bytes(b"x" * 32)
    _backdate(stuck)
    loose = chunk_dir / ("6" * 64)
    loose.write_bytes(b"y" * 32)
    _backdate(loose)
    real_unlink = os.unlink

    def selective(path, *args, **kwargs):
        if os.path.basename(os.fspath(path)) == "5" * 64:
            raise PermissionError(13, "held open elsewhere")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", selective)

    report = collect_payload_garbage(run_dir, min_age_s=_DAY_S, apply=True)

    stuck_entry = _entry(report, "5" * 64)
    assert stuck_entry.deleted is False and stuck_entry.error != ""
    assert _entry(report, "6" * 64).deleted is True
    assert stuck.exists() and not loose.exists()
    assert report.reclaimed_bytes == 32


def test_report_mode_leaves_the_directory_byte_identical(tmp_path: Path) -> None:
    """The default mode judges and counts -- candidate_bytes says what --apply would reclaim --
    and provably changes nothing, sizes and times included."""
    run_dir, _sha = _recorded_run(tmp_path)
    chunk_dir = run_dir / MODEL_PAYLOADS_DIRNAME
    orphan = chunk_dir / ("4" * 64)
    orphan.write_bytes(b"x" * 100)
    _backdate(orphan)
    temp = chunk_dir / f"{'3' * 64}.{os.getpid() + 1}.{'0' * 12}.tmp"
    temp.write_bytes(b"y" * 50)
    _backdate(temp)
    snapshot = {
        path.name: (path.lstat().st_size, path.lstat().st_mtime_ns)
        for path in chunk_dir.iterdir()
    }

    report = collect_payload_garbage(run_dir, min_age_s=_DAY_S, apply=False)

    assert report.applied is False
    assert _entry(report, "4" * 64).deleted is False
    assert _entry(report, temp.name).deleted is False
    assert report.candidate_bytes == 150 and report.reclaimed_bytes == 0
    assert {
        path.name: (path.lstat().st_size, path.lstat().st_mtime_ns)
        for path in chunk_dir.iterdir()
    } == snapshot


def test_an_absent_chunk_dir_reports_empty_and_clean(tmp_path: Path) -> None:
    """A run that never offloaded has no directory, and that is a zero report, not an anomaly."""
    (tmp_path / MODEL_PAYLOADS_FILENAME).write_text("", encoding="utf-8")

    report = collect_payload_garbage(tmp_path, min_age_s=_DAY_S, apply=True)

    assert report.chunk_dir_state == "absent"
    assert report.corpus_state == "ok"
    assert report.entries == ()
    assert report.candidate_bytes == 0 and report.reclaimed_bytes == 0


def test_a_redirected_chunk_directory_is_refused_untouched(tmp_path: Path) -> None:
    """The sweep's twin of the writer's gate: a redirection wearing the chunk directory's name
    would turn enumeration and deletion into operations in a directory of somebody else's
    choosing, so nothing is enumerated at all."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / ("2" * 64)).write_bytes(b"theirs")
    try:
        os.symlink(outside, tmp_path / MODEL_PAYLOADS_DIRNAME, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    (tmp_path / MODEL_PAYLOADS_FILENAME).write_text("", encoding="utf-8")

    report = collect_payload_garbage(tmp_path, min_age_s=0.0, apply=True)

    assert report.chunk_dir_state == "unsafe"
    assert report.entries == ()
    assert (outside / ("2" * 64)).read_bytes() == b"theirs"
