"""The recorded payload: split, byte-exact reassembly, and the shape the schema pins.

W6-2 (dx-note ``2026-08-02-v0.21-contract-replay-scope.md`` §Track B, decisions 4/5/6/8). The
corpus's headline property is that a recorded request is not a copy of the preimage but a recipe
that **reassembles to the exact bytes the replay key was taken over** -- re-encode, re-hash,
compare. Everything else here defends that property:

  1. the round-trip tests (unicode, floats, nesting, the by-reference request shape),
  2. the size rule that keeps marker-shaped caller data out of the recipe entirely (a reference is
     a fixed size, so a lookalike is always lifted into a chunk and never re-walked), with the
     verified verbatim fallback behind it for a preimage the recipe shape does not fit,
  3. the encoder-identity pin (the digest and the chunks are encoded by the same object, so the
     twin cannot drift),
  4. the writer/schema key agreement, in both directions, per record kind.

The response side has no digest to verify against; its properties are the declared field list
(``raw`` is deliberately absent) and the refuse-whole rule -- an oversized or unreconstructable
turn costs the response record, never a truncation and never a fabrication.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from monoid_agent_kernel.core._util import CANONICAL_JSON_ENCODER
from monoid_agent_kernel.core.json_ingress import json_nesting_within_limit
from monoid_agent_kernel.core.model_payloads import (
    MODEL_PAYLOADS_DIRNAME,
    MODEL_PAYLOADS_FILENAME,
    MODEL_PAYLOADS_SCHEMA_VERSION,
    MODEL_REQUEST_KIND,
    MODEL_RESPONSE_KIND,
    PAYLOAD_CHUNK_KIND,
    MARKER_ENCODED_BYTES,
    PAYLOAD_CHUNK_REF_KEY,
    PAYLOAD_OFFLOAD_THRESHOLD_BYTES,
    chunk_marker,
    chunk_record,
    model_request_record,
    model_response_record,
    reassemble_request_preimage,
    response_record_body,
    split_request_payload,
)
from monoid_agent_kernel.core import schemas
from monoid_agent_kernel.core.schemas import MODEL_PAYLOADS_RECORD_SCHEMA, validate_run_dir
from monoid_agent_kernel.core.spec import GenerationConfig, ModelConfig
from monoid_agent_kernel.model_call import _request_payload, _tool_payload
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn, ToolCall
from monoid_agent_kernel.tools.base import ToolSpec

_ENDPOINT = "https://gateway.internal.example/tenant-a/llm/turns"

_ENVELOPE = {
    "run_id": "run-1",
    "root_run_id": "run-1",
    "recorded_at": "2026-08-07T00:00:00Z",
}


def _tool(tool_id: str, description: str = "reads a file") -> ToolSpec:
    return ToolSpec(
        id=tool_id,
        description=description,
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        capability="fs.read",
        side_effect="read",
        handler=lambda **kwargs: None,
    )


def _preimage_of(request: ModelRequest, model: ModelConfig) -> tuple[bytes, str]:
    payload = _request_payload(request, model, provider="gateway")
    preimage = CANONICAL_JSON_ENCODER.encode(payload).encode("utf-8")
    return preimage, hashlib.sha256(preimage).hexdigest()


def _request(**changes: object) -> ModelRequest:
    base: dict[str, object] = {
        "instruction": "안녕 — do the thing with π ≈ 3.14159 and ☃",
        "system_prompt": "You are a careful assistant.\nLine two.",
        "tools": (_tool("fs.read"), _tool("fs.write", "writes a file")),
        "messages": ({"role": "user", "content": "hello"},),
    }
    base.update(changes)
    return ModelRequest(**base)  # type: ignore[arg-type]


_MODEL = ModelConfig(
    provider="gateway",
    model="gpt-5.5",
    gateway_url=_ENDPOINT,
    generation=GenerationConfig(temperature=0.25),
)


def _schema_errors(record: dict[str, object]) -> list[str]:
    return [
        error.message
        for error in Draft202012Validator(MODEL_PAYLOADS_RECORD_SCHEMA).iter_errors(record)
    ]


# --- Split and reassembly: the headline property --------------------------------------------------


def test_split_and_reassembly_return_the_exact_bytes_the_key_was_taken_over() -> None:
    preimage, digest = _preimage_of(_request(), _MODEL)

    split = split_request_payload(preimage, digest)

    assert split is not None
    assert split.refs is True
    # Witness before absence-style claims elsewhere: the split genuinely extracted something, and
    # a count alone would not say what. Each tool definition is its own chunk, by value.
    terms = split.payload["monoid.model-request-digest.v1"]
    assert all(set(entry) == {PAYLOAD_CHUNK_REF_KEY} for entry in terms["tools"])
    assert {
        CANONICAL_JSON_ENCODER.encode(_tool_payload(tool)).encode("utf-8")
        for tool in _request().tools
    } <= set(split.chunks.values())
    rebuilt = reassemble_request_preimage(split.payload, split.chunks.__getitem__, refs=True)
    assert rebuilt == preimage
    assert hashlib.sha256(rebuilt).hexdigest() == digest


def test_every_tool_is_its_own_chunk_so_one_surface_change_costs_one_chunk() -> None:
    """Per-tool granularity: a surface losing one tool must reuse every other tool's chunk."""
    preimage_a, digest_a = _preimage_of(_request(), _MODEL)
    preimage_b, digest_b = _preimage_of(
        _request(tools=(_tool("fs.read"),)), _MODEL
    )

    split_a = split_request_payload(preimage_a, digest_a)
    split_b = split_request_payload(preimage_b, digest_b)

    assert split_a is not None and split_b is not None
    shared = set(split_a.chunks) & set(split_b.chunks)
    # fs.read's definition survives the surface change; fs.write's does not.
    read_chunk = CANONICAL_JSON_ENCODER.encode(_tool_payload(_tool("fs.read"))).encode("utf-8")
    assert read_chunk in {split_a.chunks[sha] for sha in shared}
    assert len(set(split_a.chunks) - set(split_b.chunks)) == 1


def test_a_null_system_prompt_stays_inline_rather_than_becoming_a_chunk_of_null() -> None:
    preimage, digest = _preimage_of(_request(system_prompt=None), _MODEL)

    split = split_request_payload(preimage, digest)

    assert split is not None
    # No chunk holding the four bytes of ``null``: the size rule declines it without an exception.
    assert split.payload["monoid.model-request-digest.v1"]["system_prompt"] is None
    assert b"null" not in set(split.chunks.values())
    rebuilt = reassemble_request_preimage(split.payload, split.chunks.__getitem__, refs=True)
    assert rebuilt == preimage


def test_the_by_reference_request_shape_round_trips() -> None:
    """`messages=None` vs `()` is the wire distinction the capture surface deliberately flattens
    and a replay preimage must not."""
    by_ref = _request(
        instruction=None,
        messages=None,
        previous_turn_handle="resp_abc",
    )
    preimage, digest = _preimage_of(by_ref, _MODEL)

    split = split_request_payload(preimage, digest)

    assert split is not None
    rebuilt = reassemble_request_preimage(split.payload, split.chunks.__getitem__, refs=split.refs)
    assert rebuilt == preimage


def test_a_message_smaller_than_a_reference_stays_inline_and_a_bigger_one_is_lifted() -> None:
    """The threshold is what an indirection costs, so it is the point below which one cannot pay
    for itself. (Whether a lifted chunk then stays in the JSONL line or moves to the directory is
    the recorder's separate `PAYLOAD_OFFLOAD_THRESHOLD_BYTES` decision.)"""
    # Straddle the threshold itself rather than clearing it by three orders of magnitude: a
    # message one byte under must stay inline and one byte over must lift, or a regression that
    # moves MARKER_ENCODED_BYTES anywhere at all goes unnoticed.
    overhead = len(CANONICAL_JSON_ENCODER.encode({"role": "user", "content": ""}).encode("utf-8"))
    under = {"role": "user", "content": "u" * (MARKER_ENCODED_BYTES - overhead - 1)}
    over = {"role": "user", "content": "o" * (MARKER_ENCODED_BYTES - overhead)}
    assert len(CANONICAL_JSON_ENCODER.encode(under).encode("utf-8")) == MARKER_ENCODED_BYTES - 1
    assert len(CANONICAL_JSON_ENCODER.encode(over).encode("utf-8")) == MARKER_ENCODED_BYTES

    small_preimage, small_digest = _preimage_of(_request(messages=(under,)), _MODEL)
    big_preimage, big_digest = _preimage_of(_request(messages=(over,)), _MODEL)

    small = split_request_payload(small_preimage, small_digest)
    big = split_request_payload(big_preimage, big_digest)

    assert small is not None and big is not None
    assert small.payload["monoid.model-request-digest.v1"]["messages"] == [under]
    assert big.payload["monoid.model-request-digest.v1"]["messages"] == [
        chunk_marker(hashlib.sha256(CANONICAL_JSON_ENCODER.encode(over).encode("utf-8")).hexdigest())
    ]
    assert (
        reassemble_request_preimage(big.payload, big.chunks.__getitem__, refs=True) == big_preimage
    )


def _no_chunks(sha: str) -> bytes:
    raise AssertionError(f"a refs=False reassembly must not resolve chunks, asked for {sha}")


def test_a_preimage_whose_digest_disagrees_is_refused_rather_than_recorded() -> None:
    """A record whose key contradicts its own bytes would poison every consumer downstream of the
    join; refusing to produce one is the `_digest` doctrine applied to the writer."""
    preimage, _digest_value = _preimage_of(_request(), _MODEL)

    assert split_request_payload(preimage, "f" * 64) is None


def test_a_recorded_request_never_carries_the_endpoint_it_is_keyed_beside() -> None:
    """Witness first: the projection saw a gateway-shaped call (provider present), and then the
    endpoint is absent -- the same two-step W6-1's ledger test uses, so a projection handed
    nothing cannot pass vacuously. Structural since W6-0 (`_model_identity` excludes it); this
    pins the property at the artifact boundary where it is now durable."""
    preimage, digest = _preimage_of(_request(), _MODEL)
    split = split_request_payload(preimage, digest)
    assert split is not None

    record = model_request_record(
        split.payload,
        refs=split.refs,
        request_digest=digest,
        digest_generation="monoid.model-request-digest.v1",
        **_ENVELOPE,
    )
    everything = json.dumps(record, ensure_ascii=False) + "".join(
        chunk.decode("utf-8") for chunk in split.chunks.values()
    )

    # The witness has to be a value only the model config could have supplied. `provider` cannot
    # serve: `_preimage_of` passes it in literally, so the assertion would hold with the whole
    # model block dropped from the projection -- which is exactly the regression it is here to
    # catch. Its runtime sibling in `test_model_payloads.py` names the same trap.
    assert '"model":"gpt-5.5"' in everything
    assert _ENDPOINT not in everything


def test_the_digest_encoder_and_the_chunk_encoder_are_the_same_object() -> None:
    """Two instances with equal settings are a resemblance, not a binding; the corpus's
    self-verification rests on the chunk bytes being what the digest hashed.

    The digest side is ``providers/_request_identity`` since W6-4b moved the key derivation
    there (the pin moved with it -- same claim, digest's current home)."""
    from monoid_agent_kernel.core import _util, model_payloads
    from monoid_agent_kernel.providers import _request_identity

    assert _request_identity.CANONICAL_JSON_ENCODER is _util.CANONICAL_JSON_ENCODER
    assert model_payloads.CANONICAL_JSON_ENCODER is _util.CANONICAL_JSON_ENCODER


# --- The response body ----------------------------------------------------------------------------


def _turn(**changes: object) -> ModelTurn:
    base: dict[str, object] = {
        "response_id": "resp_1",
        "final_text": "the answer",
        "tool_calls": (ToolCall(id="c1", name="fs.read", arguments={"path": "a.txt"}),),
        "usage": {"input_tokens": 10, "output_tokens": 2},
        "raw": {"provider_blob": "never recorded"},
        "reasoning": ({"type": "reasoning", "encrypted_content": "OPAQUE"},),
        "stop_reason": "stop",
        "provider_retried": True,
    }
    base.update(changes)
    return ModelTurn(**base)  # type: ignore[arg-type]


def test_a_response_body_names_the_turn_fields_and_omits_raw() -> None:
    recorded = response_record_body(_turn())

    assert recorded.unrecorded_reason == ""
    assert recorded.value is not None
    assert set(recorded.value) == {
        "response_id",
        "final_text",
        "tool_calls",
        "reasoning",
        "usage",
        "stop_reason",
        "provider_retried",
    }
    assert recorded.value["tool_calls"] == [
        {"id": "c1", "name": "fs.read", "arguments": {"path": "a.txt"}}
    ]
    assert recorded.value["reasoning"] == [{"type": "reasoning", "encrypted_content": "OPAQUE"}]
    assert recorded.value["provider_retried"] is True
    assert "provider_blob" not in json.dumps(recorded.value)
    assert recorded.encoded is not None
    assert json.loads(recorded.encoded.decode("utf-8")) == recorded.value


def test_an_unreconstructable_tool_call_costs_the_response_not_a_fabrication() -> None:
    """The capture surface records a bounded repr for these; a replay corpus must not, because a
    replayed turn built from a repr is a fabricated call."""

    class Odd:  # no id/name/arguments
        pass

    recorded = response_record_body(_turn(final_text="settled", tool_calls=(Odd(),)))

    assert recorded.value is None
    assert recorded.encoded is None
    assert recorded.unrecorded_reason == "unencodable"


def test_an_oversized_response_is_refused_whole_never_truncated() -> None:
    recorded = response_record_body(_turn(final_text="x" * 8_000_001))

    assert recorded.value is None
    assert recorded.unrecorded_reason == "too_large"


# --- Writer/schema agreement ----------------------------------------------------------------------


def _request_record() -> dict[str, object]:
    preimage, digest = _preimage_of(_request(), _MODEL)
    split = split_request_payload(preimage, digest)
    assert split is not None
    return model_request_record(
        split.payload,
        refs=split.refs,
        request_digest=digest,
        digest_generation="monoid.model-request-digest.v1",
        **_ENVELOPE,
    )


def _response_record() -> dict[str, object]:
    recorded = response_record_body(_turn())
    assert recorded.value is not None
    return model_response_record(
        recorded.value,
        call_index=0,
        request_digest="b" * 64,
        unrecorded_reason="",
        **_ENVELOPE,
    )


def _chunk_record() -> dict[str, object]:
    return chunk_record(CANONICAL_JSON_ENCODER.encode("prompt").encode("utf-8"), **_ENVELOPE)


def test_every_record_kind_validates_against_its_schema_branch() -> None:
    for record in (_request_record(), _response_record(), _chunk_record()):
        assert _schema_errors(record) == []


def test_a_null_and_a_marker_response_both_validate() -> None:
    null_record = model_response_record(
        None, call_index=1, request_digest="", unrecorded_reason="too_large", **_ENVELOPE
    )
    marker_record = model_response_record(
        chunk_marker("a" * 64), call_index=2, request_digest="b" * 64, unrecorded_reason="", **_ENVELOPE
    )

    assert _schema_errors(null_record) == []
    assert _schema_errors(marker_record) == []


def test_the_writer_and_the_schema_declare_the_same_keys() -> None:
    """Both directions, per kind: a writer key the schema does not admit fails on
    additionalProperties, and a schema requirement the writer omits fails on required -- so the
    two lists cannot drift apart silently."""
    by_kind = {
        record["kind"]: record
        for record in (_request_record(), _response_record(), _chunk_record())
    }
    branches = {
        branch["properties"]["kind"]["const"]: branch
        for branch in MODEL_PAYLOADS_RECORD_SCHEMA["oneOf"]
    }

    assert set(by_kind) == set(branches) == {
        PAYLOAD_CHUNK_KIND,
        MODEL_REQUEST_KIND,
        MODEL_RESPONSE_KIND,
    }
    for kind, record in by_kind.items():
        branch = branches[kind]
        assert set(record) == set(branch["required"]) == set(branch["properties"]), kind
        assert branch["additionalProperties"] is False, kind


def test_the_kinds_are_mutually_exclusive_under_the_schema() -> None:
    """oneOf, not anyOf: a record must match exactly one branch, so a line that somehow carried
    two kinds' fields is malformed rather than ambiguous."""
    confused = _chunk_record() | {"kind": MODEL_REQUEST_KIND}

    assert _schema_errors(confused) != []


def test_the_schema_version_is_the_single_payloads_namespace() -> None:
    for branch in MODEL_PAYLOADS_RECORD_SCHEMA["oneOf"]:
        assert branch["properties"]["schema_version"] == {
            "enum": [MODEL_PAYLOADS_SCHEMA_VERSION]
        }


# --- validate_run_dir -----------------------------------------------------------------------------


def test_validate_run_dir_treats_model_payloads_as_optional(tmp_path: Path) -> None:
    issues = validate_run_dir(tmp_path)

    assert not any(issue.path.startswith(MODEL_PAYLOADS_FILENAME) for issue in issues)


def _write_run_payloads(tmp_path: Path, *records: dict[str, object]) -> Path:
    path = tmp_path / MODEL_PAYLOADS_FILENAME
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_validate_run_dir_reassembles_every_request_record(tmp_path: Path) -> None:
    preimage, digest = _preimage_of(_request(), _MODEL)
    split = split_request_payload(preimage, digest)
    assert split is not None
    records = [chunk_record(chunk, **_ENVELOPE) for chunk in split.chunks.values()]
    records.append(
        model_request_record(
            split.payload,
            refs=split.refs,
            request_digest=digest,
            digest_generation="monoid.model-request-digest.v1",
            **_ENVELOPE,
        )
    )
    _write_run_payloads(tmp_path, *records)

    assert not any(
        issue.path.startswith(MODEL_PAYLOADS_FILENAME) for issue in validate_run_dir(tmp_path)
    )


def test_validate_run_dir_catches_a_corrupted_chunk(tmp_path: Path) -> None:
    """One flipped byte in one chunk must fail the reassembly of every record that references it;
    the corpus is self-verifying or it is nothing."""
    preimage, digest = _preimage_of(_request(), _MODEL)
    split = split_request_payload(preimage, digest)
    assert split is not None
    sha, chunk = next(iter(split.chunks.items()))
    tampered = chunk[:-2] + (b"X" if chunk[-2:-1] != b"X" else b"Y") + chunk[-1:]
    records = [
        # The tampered chunk keeps its RECORDED sha, so the chunk's own integrity check fires.
        {**chunk_record(chunk, **_ENVELOPE), "text": tampered.decode("utf-8")},
        *(
            chunk_record(other, **_ENVELOPE)
            for other_sha, other in split.chunks.items()
            if other_sha != sha
        ),
        model_request_record(
            split.payload,
            refs=split.refs,
            request_digest=digest,
            digest_generation="monoid.model-request-digest.v1",
            **_ENVELOPE,
        ),
    ]
    _write_run_payloads(tmp_path, *records)

    assert any(issue.path.startswith(MODEL_PAYLOADS_FILENAME) for issue in validate_run_dir(tmp_path))


def test_validate_run_dir_resolves_offloaded_chunks_from_the_directory(tmp_path: Path) -> None:
    big_text = "x" * (PAYLOAD_OFFLOAD_THRESHOLD_BYTES + 1024)
    preimage, digest = _preimage_of(
        _request(messages=({"role": "user", "content": big_text},)), _MODEL
    )
    split = split_request_payload(preimage, digest)
    assert split is not None
    inline: list[dict[str, object]] = []
    payload_dir = tmp_path / MODEL_PAYLOADS_DIRNAME
    payload_dir.mkdir()
    for sha, chunk in split.chunks.items():
        if len(chunk) > PAYLOAD_OFFLOAD_THRESHOLD_BYTES:
            (payload_dir / sha).write_bytes(chunk)
        else:
            inline.append(chunk_record(chunk, **_ENVELOPE))
    inline.append(
        model_request_record(
            split.payload,
            refs=split.refs,
            request_digest=digest,
            digest_generation="monoid.model-request-digest.v1",
            **_ENVELOPE,
        )
    )
    _write_run_payloads(tmp_path, *inline)

    assert not any(
        issue.path.startswith(MODEL_PAYLOADS_FILENAME) for issue in validate_run_dir(tmp_path)
    )


def test_validate_run_dir_reports_a_dangling_response_marker(tmp_path: Path) -> None:
    _write_run_payloads(
        tmp_path,
        model_response_record(
            chunk_marker("c" * 64),
            call_index=0,
            request_digest="b" * 64,
            unrecorded_reason="",
            **_ENVELOPE,
        ),
    )

    assert any(issue.path.startswith(MODEL_PAYLOADS_FILENAME) for issue in validate_run_dir(tmp_path))


def test_a_chunk_reference_naming_a_path_is_refused_before_anything_is_opened(
    tmp_path: Path, monkeypatch
) -> None:
    """A chunk reference is a content-addressed *name*, and the reader has to re-establish that --
    the writer's is always 64 hex, but a corpus arrives from wherever run directories arrive from.
    Joined onto the chunk directory an absolute or `..`-relative string discards the base entirely,
    so `monoid validate` would read a file the record chose: outside the run directory, unbounded,
    or a FIFO that never returns. The hash check cannot defend it, because it happens after the
    read."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim.txt").write_text("not the agent's to read", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    escape = str((outside / "victim.txt").resolve())
    _write_run_payloads(
        run_dir,
        model_request_record(
            {"monoid.model-request-digest.v1": {"system_prompt": chunk_marker(escape)}},
            refs=True,
            request_digest="a" * 64,
            digest_generation="monoid.model-request-digest.v1",
            **_ENVELOPE,
        ),
        model_response_record(
            chunk_marker(escape),
            call_index=0,
            request_digest="a" * 64,
            unrecorded_reason="",
            **_ENVELOPE,
        ),
    )
    # Spy the reader the validator actually calls. `Path.read_bytes` was the pre-fix route and is
    # no longer on it, so watching that would have watched nothing -- the escape would sail past a
    # green test.
    asked: list[str] = []
    real_reader = schemas.read_verified_bytes

    def spy(path: Path, *, max_bytes: int) -> bytes | None:
        asked.append(str(path))
        return real_reader(path, max_bytes=max_bytes)

    monkeypatch.setattr(schemas, "read_verified_bytes", spy)

    issues = validate_run_dir(run_dir)

    assert [path for path in asked if str(run_dir) not in path] == []
    # Both branches still report the record; refusing the name must not cost the diagnosis.
    assert len([issue for issue in issues if issue.path.startswith(MODEL_PAYLOADS_FILENAME)]) == 2


def test_validate_run_dir_reports_a_request_record_that_no_longer_reassembles(
    tmp_path: Path,
) -> None:
    """The artifact's headline promise, pinned where it is made. Every chunk here hashes correctly,
    so the chunk-integrity check stays silent and only the reassembly comparison can speak."""
    preimage, digest = _preimage_of(_request(), _MODEL)
    split = split_request_payload(preimage, digest)
    assert split is not None
    assert split.refs is True
    tampered = json.loads(json.dumps(split.payload))
    terms = tampered["monoid.model-request-digest.v1"]
    terms["tools"] = list(terms["tools"])[:-1]  # a definition the preimage had, dropped
    records = [chunk_record(chunk, **_ENVELOPE) for chunk in split.chunks.values()]
    records.append(
        model_request_record(
            tampered,
            refs=True,
            request_digest=digest,
            digest_generation="monoid.model-request-digest.v1",
            **_ENVELOPE,
        )
    )
    _write_run_payloads(tmp_path, *records)

    assert [
        issue.message
        for issue in validate_run_dir(tmp_path)
        if issue.path.startswith(MODEL_PAYLOADS_FILENAME)
    ] == ["request payload does not reassemble to its request_digest"]


def test_validate_run_dir_catches_an_offloaded_chunk_that_does_not_match_its_name(
    tmp_path: Path,
) -> None:
    """The directory twin of the inline chunk check. A content-addressed file is trusted by name by
    every consumer, so the one reader that can disprove the name has to do it."""
    big_text = "x" * (PAYLOAD_OFFLOAD_THRESHOLD_BYTES + 1024)
    preimage, digest = _preimage_of(
        _request(messages=({"role": "user", "content": big_text},)), _MODEL
    )
    split = split_request_payload(preimage, digest)
    assert split is not None
    inline: list[dict[str, object]] = []
    payload_dir = tmp_path / MODEL_PAYLOADS_DIRNAME
    payload_dir.mkdir()
    for sha, chunk in split.chunks.items():
        if len(chunk) > PAYLOAD_OFFLOAD_THRESHOLD_BYTES:
            (payload_dir / sha).write_bytes(chunk[:-1] + b"Z")  # same name, other bytes
        else:
            inline.append(chunk_record(chunk, **_ENVELOPE))
    inline.append(
        model_request_record(
            split.payload,
            refs=True,
            request_digest=digest,
            digest_generation="monoid.model-request-digest.v1",
            **_ENVELOPE,
        )
    )
    _write_run_payloads(tmp_path, *inline)

    assert any(
        issue.path.startswith(MODEL_PAYLOADS_FILENAME) for issue in validate_run_dir(tmp_path)
    )


def test_validate_run_dir_never_walks_a_verbatim_payload(tmp_path: Path) -> None:
    """`refs=False` is only an answer if the reader honours it: walking a verbatim payload would
    resolve the lookalike inside it and report a perfectly faithful corpus as broken. Built by
    hand, because the writer can no longer be induced to emit one from caller data -- the file
    format still permits it, and a reader answers to the file."""
    verbatim = {"anything": [{PAYLOAD_CHUNK_REF_KEY: "0" * 64}]}
    encoded = CANONICAL_JSON_ENCODER.encode(verbatim).encode("utf-8")
    _write_run_payloads(
        tmp_path,
        model_request_record(
            verbatim,
            refs=False,
            request_digest=hashlib.sha256(encoded).hexdigest(),
            digest_generation="monoid.model-request-digest.v1",
            **_ENVELOPE,
        ),
    )

    assert not any(
        issue.path.startswith(MODEL_PAYLOADS_FILENAME) for issue in validate_run_dir(tmp_path)
    )


def _message(index: int) -> dict[str, object]:
    return {"role": "user", "content": f"turn {index}: " + "detail " * 20}


def test_a_growing_conversation_re_references_the_messages_it_already_recorded() -> None:
    """Dedup is per *element* for the history exactly as it is for tools, and for the same reason:
    a by-value conversation resends every earlier message every turn. Chunking the block as one
    value keys it by the whole history, so the sha changes every turn and nothing is ever reused --
    a hundred-turn run would write the history a hundred times, growing quadratically in the one
    dimension a long run grows in."""
    early, early_digest = _preimage_of(_request(messages=tuple(_message(i) for i in range(3))), _MODEL)
    later, later_digest = _preimage_of(_request(messages=tuple(_message(i) for i in range(4))), _MODEL)
    first = split_request_payload(early, early_digest)
    second = split_request_payload(later, later_digest)
    assert first is not None and second is not None

    assert set(first.chunks) < set(second.chunks)
    assert len(set(second.chunks) - set(first.chunks)) == 1  # one new message, one new chunk


def test_a_value_big_enough_to_be_a_chunk_reference_is_never_left_inline() -> None:
    """The verbatim fallback exists for marker-shaped caller data, and this is what makes it
    unreachable *from* caller data: a reference encodes to a fixed size, so lifting every value at
    least that large means a lookalike is always inside a chunk -- and reassembly never walks a
    value it resolved. The collision costs one indirection that was already worth making."""
    lookalike = {PAYLOAD_CHUNK_REF_KEY: "0" * 64}
    preimage, digest = _preimage_of(
        _request(messages=({"role": "user", "content": [lookalike]},)), _MODEL
    )

    split = split_request_payload(preimage, digest)

    assert split is not None
    assert split.refs is True
    assert reassemble_request_preimage(split.payload, split.chunks.__getitem__, refs=True) == preimage


def test_a_preimage_the_recipe_shape_does_not_fit_still_takes_the_verbatim_arm() -> None:
    """The fallback is not dead code. It no longer triggers on marker-shaped caller data -- that is
    the point of the size rule -- but a preimage that is not a single-key wrapper around a term
    object extracts nothing, and a recipe with no references is a verbatim payload."""
    preimage = CANONICAL_JSON_ENCODER.encode([1, 2, 3]).encode("utf-8")
    digest = hashlib.sha256(preimage).hexdigest()

    split = split_request_payload(preimage, digest)

    assert split is not None
    assert split.refs is False
    assert split.chunks == {}
    assert reassemble_request_preimage(split.payload, _no_chunks, refs=False) == preimage


def test_a_pasted_instruction_does_not_land_whole_on_one_jsonl_line() -> None:
    """Extraction is size-scoped, not field-scoped. The old rule named two fields, so a term it did
    not name -- `instruction`, which is caller text -- put megabytes on one line on the happy
    path."""
    preimage, digest = _preimage_of(_request(instruction="p" * 3_000_000), _MODEL)
    split = split_request_payload(preimage, digest)
    assert split is not None

    record = model_request_record(
        split.payload,
        refs=split.refs,
        request_digest=digest,
        digest_generation="monoid.model-request-digest.v1",
        **_ENVELOPE,
    )

    assert len(json.dumps(record, ensure_ascii=False)) < 4096


def test_a_deep_tool_result_still_gets_its_request_record() -> None:
    """The corpus does not contain what its own validator cannot read -- but what the validator
    reads is the recorded *line*, not the preimage, and the size rule lifts a deep value into a
    chunk, where its brackets sit inside a JSON string and count for nothing. Bounding the preimage
    instead refused a record that reassembles byte-for-byte and that `validate_run_dir` accepts
    with no issues, and it refused it for every later call in the run, because a tool result stays
    in the by-value message log."""
    deep: object = "leaf"
    for _ in range(600):
        deep = [deep]
    preimage, digest = _preimage_of(
        _request(messages=({"role": "user", "content": deep},)), _MODEL
    )

    split = split_request_payload(preimage, digest)

    assert split is not None
    assert split.refs is True
    assert reassemble_request_preimage(split.payload, split.chunks.__getitem__, refs=True) == preimage
    # ... and the line the reader would parse is shallow, which is why the refusal was false.
    record = model_request_record(
        split.payload,
        refs=True,
        request_digest=digest,
        digest_generation="monoid.model-request-digest.v1",
        **_ENVELOPE,
    )
    assert json_nesting_within_limit(json.dumps(record, ensure_ascii=False))


def test_a_record_that_matches_no_branch_is_reported_without_reprinting_itself(
    tmp_path: Path,
) -> None:
    """`monoid validate` writes its issues to a terminal and into `--json` output, and jsonschema
    builds its message out of the *instance* -- so an unmatched line would republish a whole
    conversation, or a whole system prompt for a chunk record, out of the private run directory.
    The trigger is the ordinary one, not an attack: this artifact pins a literal single-element
    `schema_version` enum, so the first version bump makes every line of every retained run
    directory match no branch at once."""
    secret = "the tenant's confidential prompt text"
    intact = chunk_record(secret.encode("utf-8"), **_ENVELOPE)
    cases = {
        "version bump": {**intact, "schema_version": "monoid.model-payloads.v2"},
        "extra property": {**intact, "surprise": secret},
        "wrong type": {**intact, "text": 5},
    }
    reported = {}
    for name, record in cases.items():
        run_dir = tmp_path / name.replace(" ", "-")
        run_dir.mkdir()
        _write_run_payloads(run_dir, record)
        reported[name] = [
            issue.message
            for issue in validate_run_dir(run_dir)
            if issue.path.startswith(MODEL_PAYLOADS_FILENAME)
        ]

    assert all(messages for messages in reported.values()), reported
    assert all(secret not in message for messages in reported.values() for message in messages)
    # Redacting must not flatten the diagnosis. Both schemas are a bare top-level `oneOf`, so
    # reporting the outer keyword alone gives one identical line for every failure mode -- and a
    # version bump makes every line of a retained run directory fail at once.
    assert len({tuple(messages) for messages in reported.values()}) == len(cases), reported
    assert reported["version bump"] == ["does not satisfy the enum constraint at $.schema_version"]


def test_the_other_content_artifact_is_redacted_too(tmp_path: Path) -> None:
    """`model-content.jsonl` carries the same class of value behind a literal v1-only version enum, and
    the code binds both. Binding the test to one of the two is the twin miss the comment beside the
    code warns about."""
    from monoid_agent_kernel.core.model_content import MODEL_CONTENT_FILENAME

    secret = "the tenant's confidential message text"
    (tmp_path / MODEL_CONTENT_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": "monoid.model-content.v2",
                "kind": "settled_text",
                "text": secret,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    issues = [
        issue
        for issue in validate_run_dir(tmp_path)
        if issue.path.startswith(MODEL_CONTENT_FILENAME)
    ]

    assert issues
    assert all(secret not in issue.message for issue in issues)


def test_reassembly_stops_expanding_at_the_payload_ceiling() -> None:
    """A per-read ceiling is not a bound on a reassembly: references are cheap and repeatable, so a
    payload naming one modest chunk thousands of times expands a half-megabyte corpus into
    gigabytes inside `monoid validate`. A faithful record cannot hit the ceiling, because it
    reassembles to a preimage the same constant already bounded."""
    chunk = CANONICAL_JSON_ENCODER.encode("x" * 10_000).encode("utf-8")
    sha = hashlib.sha256(chunk).hexdigest()
    payload = {"messages": [chunk_marker(sha)] * 50}

    assert reassemble_request_preimage(payload, lambda _sha: chunk, refs=True)

    with pytest.raises(ValueError):
        reassemble_request_preimage(
            payload, lambda _sha: chunk, refs=True, max_bytes=100_000
        )
