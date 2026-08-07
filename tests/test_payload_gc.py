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
from monoid_agent_kernel.core.schemas import validate_run_dir

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
