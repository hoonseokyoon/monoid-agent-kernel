"""The replay corpus as a run actually writes it.

W6-2. ``tests/test_model_payloads_schema.py`` pins what a record may say and that reassembly is
byte-exact; this pins that a run produces those records -- deduplicated, offloaded, index-aligned
with the ledger -- without ever being able to fail the run, and that the two sidecar arms fail
independently (one file's disk problem must not silence the other).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core._util import CANONICAL_JSON_ENCODER
from monoid_agent_kernel.core.model_calls import MODEL_CALLS_FILENAME
from monoid_agent_kernel.core.model_io import ModelCallReceipt
from monoid_agent_kernel.core.model_payloads import (
    MODEL_PAYLOADS_DIRNAME,
    MODEL_PAYLOADS_FILENAME,
    MODEL_REQUEST_KIND,
    MODEL_RESPONSE_KIND,
    PAYLOAD_CHUNK_KIND,
    PAYLOAD_OFFLOAD_THRESHOLD_BYTES,
    reassemble_request_preimage,
    split_request_payload,
)
from monoid_agent_kernel.core.schemas import validate_run_dir
from monoid_agent_kernel.core.spec import AgentRunSpec, RunLimits
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.model_call import SettledModelCall
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn
from monoid_agent_kernel.recorder import AgentRecorder


class _Adapter:
    supports_multimodal = False
    provider_name = "test-provider"

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        del request
        return ModelTurn(
            response_id="r1",
            final_text="answer",
            reasoning=({"type": "reasoning", "encrypted_content": "OPAQUE"},),
            raw={"provider_blob": "never recorded"},
        )


class _FailingAdapter:
    supports_multimodal = False
    provider_name = "test-provider"

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        del request
        raise ModelAdapterError("upstream refused", retryable=True, http_status=429)


def _loop(
    tmp_path: Path,
    adapter: object,
    *,
    model_payload_file: bool = True,
    model_calls_file: bool = True,
) -> AgentLoop:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,  # type: ignore[arg-type]
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
        model_calls_file=model_calls_file,
        model_payload_file=model_payload_file,
    )


def _records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / MODEL_PAYLOADS_FILENAME
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _by_kind(records: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [record for record in records if record["kind"] == kind]


def test_a_run_writes_no_payload_corpus_unless_it_is_asked_to(tmp_path: Path) -> None:
    """Opt-in like its two sidecar siblings, and independent of the ledger switch: a ledger-only
    run keeps exactly the W6-1 run-directory shape."""
    result = _loop(tmp_path, _Adapter(), model_payload_file=False).run_once("hi")

    assert not (result.run_dir / MODEL_PAYLOADS_FILENAME).exists()
    assert not (result.run_dir / MODEL_PAYLOADS_DIRNAME).exists()
    assert (result.run_dir / MODEL_CALLS_FILENAME).exists()


def test_a_run_records_its_request_and_response_and_they_reassemble(tmp_path: Path) -> None:
    """The headline property on real bytes: what the run wrote resolves back to the exact
    preimage of the key its own ledger recorded."""
    result = _loop(tmp_path, _Adapter()).run_once("hi")
    records = _records(result.run_dir)
    requests = _by_kind(records, MODEL_REQUEST_KIND)
    responses = _by_kind(records, MODEL_RESPONSE_KIND)
    chunks = {record["sha256"]: record["text"].encode("utf-8") for record in _by_kind(records, PAYLOAD_CHUNK_KIND)}

    assert len(requests) == 1
    assert len(responses) == 1
    rebuilt = reassemble_request_preimage(
        requests[0]["payload"], chunks.__getitem__, refs=requests[0]["refs"]
    )
    assert hashlib.sha256(rebuilt).hexdigest() == requests[0]["request_digest"]

    response = responses[0]["response"]
    assert response is not None
    assert response["final_text"] == "answer"
    assert response["response_id"] == "r1"
    assert response["reasoning"] == [{"type": "reasoning", "encrypted_content": "OPAQUE"}]
    assert "raw" not in response
    assert "provider_blob" not in json.dumps(records)


def test_the_corpus_and_the_ledger_agree_on_which_call_is_which(tmp_path: Path) -> None:
    """The join contract: response record N describes ledger line N."""
    result = _loop(tmp_path, _Adapter()).run_once("hi")
    ledger = [
        json.loads(line)
        for line in (result.run_dir / MODEL_CALLS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    responses = _by_kind(_records(result.run_dir), MODEL_RESPONSE_KIND)

    assert len(ledger) == len(responses) == 1
    assert responses[0]["call_index"] == ledger[0]["call_index"]
    assert responses[0]["request_digest"] == ledger[0]["request_digest"]


def test_a_failed_call_records_its_request_and_no_response(tmp_path: Path) -> None:
    """What was asked is exactly as recordable for a failure; what was answered does not exist.
    The failure's classification lives in the ledger line the request joins through its digest."""
    result = _loop(tmp_path, _FailingAdapter()).run_once("hi")
    records = _records(result.run_dir)

    assert len(_by_kind(records, MODEL_REQUEST_KIND)) == 1
    assert _by_kind(records, MODEL_RESPONSE_KIND) == []
    assert validate_run_dir(result.run_dir) == []


def test_two_calls_sharing_a_surface_share_their_chunks(tmp_path: Path) -> None:
    """Dedup on real runs: the second call's tool definitions and system prompt are already in
    the file, so it adds no chunk records -- and the two calls' digests differ (the conversation
    grew), so both request records exist."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    loop = AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace,
            run_root=tmp_path / "runs",
            limits=RunLimits(max_steps=3),
        ),
        model_adapter=_ToolThenAnswerAdapter(),  # type: ignore[arg-type]
        runtime_config_provider=runtime_provider(runtime_config("fs.read", "run.finish")),
        model_calls_file=True,
        model_payload_file=True,
    )
    (workspace / "a.txt").write_text("content", encoding="utf-8")
    result = loop.run_once("hi")
    records = _records(result.run_dir)
    requests = _by_kind(records, MODEL_REQUEST_KIND)
    chunk_records = _by_kind(records, PAYLOAD_CHUNK_KIND)

    assert len(requests) == 2
    # Every chunk sha appears exactly once even though both requests reference the shared ones.
    shas = [record["sha256"] for record in chunk_records]
    assert len(shas) == len(set(shas))
    assert validate_run_dir(result.run_dir) == []


class _ToolThenAnswerAdapter:
    supports_multimodal = False
    provider_name = "test-provider"

    def __init__(self) -> None:
        self._calls = 0

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        del request
        self._calls += 1
        if self._calls == 1:
            from monoid_agent_kernel.providers.fake import fake_tool_call

            return ModelTurn(tool_calls=(fake_tool_call("fs.read", {"path": "a.txt"}),))
        return ModelTurn(response_id="r2", final_text="done")


def test_an_oversized_message_block_is_offloaded_and_still_reassembles(tmp_path: Path) -> None:
    big = "x" * (PAYLOAD_OFFLOAD_THRESHOLD_BYTES + 4096)
    result = _loop(tmp_path, _Adapter()).run_once(big)
    records = _records(result.run_dir)
    requests = _by_kind(records, MODEL_REQUEST_KIND)
    chunk_dir = result.run_dir / MODEL_PAYLOADS_DIRNAME

    assert len(requests) == 1
    offloaded = list(chunk_dir.iterdir()) if chunk_dir.exists() else []
    assert offloaded, "an oversized value must leave the JSONL line"
    for stored in offloaded:
        assert hashlib.sha256(stored.read_bytes()).hexdigest() == stored.name
    assert validate_run_dir(result.run_dir) == []


def test_the_corpus_arm_failing_does_not_silence_the_ledger_arm(tmp_path: Path) -> None:
    """Twin independence, driven at the recorder directly (the runner's containment must not be
    what this rests on -- W6-1 lesson). The corpus handle is poisoned; the ledger keeps going,
    and the reverse direction holds too."""
    recorder = _standalone_recorder(tmp_path / "one")
    recorder._model_payloads_failed = True
    recorder.record_settled_call(
        SettledModelCall(receipt=ModelCallReceipt(), request_preimage=None, turn=None)
    )
    assert (recorder.run_dir / MODEL_CALLS_FILENAME).exists()
    assert not (recorder.run_dir / MODEL_PAYLOADS_FILENAME).exists()
    recorder.close()

    twin = _standalone_recorder(tmp_path / "two")
    twin._model_calls_failed = True
    twin.record_settled_call(
        SettledModelCall(
            receipt=ModelCallReceipt(),
            request_preimage=None,
            turn=ModelTurn(response_id="r", final_text="answer"),
        )
    )
    assert not (twin.run_dir / MODEL_CALLS_FILENAME).exists()
    assert (twin.run_dir / MODEL_PAYLOADS_FILENAME).exists()
    twin.close()


def _standalone_recorder(base: Path) -> AgentRecorder:
    return AgentRecorder(
        base / "runs",
        "run-1",
        status_file=False,
        model_calls_file=True,
        model_payload_file=True,
    )


def test_a_record_arriving_after_close_is_ignored_not_reopened(tmp_path: Path) -> None:
    recorder = _standalone_recorder(tmp_path)
    recorder.record_settled_call(
        SettledModelCall(
            receipt=ModelCallReceipt(),
            turn=ModelTurn(response_id="r", final_text="answer"),
        )
    )
    recorder.close()
    before = (recorder.run_dir / MODEL_PAYLOADS_FILENAME).read_bytes()

    recorder.record_settled_call(
        SettledModelCall(
            receipt=ModelCallReceipt(),
            turn=ModelTurn(response_id="r2", final_text="late"),
        )
    )

    assert (recorder.run_dir / MODEL_PAYLOADS_FILENAME).read_bytes() == before


@pytest.mark.parametrize("link_kind", ["hardlink", "symlink"])
def test_a_planted_link_stops_the_corpus_not_the_run(tmp_path: Path, link_kind: str) -> None:
    """The corpus twin of the ledger's planted-link test: the verified open refuses, the arm goes
    terminal, nothing is appended through somebody else's name -- and the ledger still writes."""
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "someone-elses.jsonl"
    target.write_text("theirs\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    planted = run_dir / MODEL_PAYLOADS_FILENAME
    if link_kind == "hardlink":
        os.link(target, planted)
    else:
        try:
            os.symlink(target, planted)
        except OSError as error:
            pytest.skip(f"file symlinks are unavailable: {error}")
    recorder = AgentRecorder(
        tmp_path / "runs",
        "run-1",
        status_file=False,
        model_calls_file=True,
        model_payload_file=True,
    )

    recorder.record_settled_call(
        SettledModelCall(
            receipt=ModelCallReceipt(),
            turn=ModelTurn(response_id="r", final_text="answer"),
        )
    )
    recorder.close()

    assert target.read_text(encoding="utf-8") == "theirs\n"
    assert (run_dir / MODEL_CALLS_FILENAME).exists()


def test_a_child_loop_records_into_its_own_run_directory(tmp_path: Path) -> None:
    """The switch inherits; each recorder owns one directory; ``root_run_id`` is the join.

    Same routing shape as the ledger's subagent test: the adapter answers as the child when it
    sees the child's persona marker, so parent and child each settle at least one call.
    """
    from monoid_agent_kernel.core.agents import AgentRuntimeConfig, PromptSpec, SubagentDefinition
    from support.runtime import tool_binding

    child_marker = "CHILD-PERSONA-MARKER"

    class RoutingAdapter:
        supports_multimodal = False
        provider_name = "test-provider"
        parent_calls = 0

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            if child_marker in request.system_prompt:
                return ModelTurn(response_id="child-1", final_text="child done")
            self.parent_calls += 1
            if self.parent_calls == 1:
                from monoid_agent_kernel.providers.base import ToolCall

                return ModelTurn(
                    response_id="parent-1",
                    tool_calls=(
                        ToolCall(
                            id="spawn-1",
                            name="agent_spawn",
                            arguments={"subagent_type": "child", "prompt": "work"},
                        ),
                    ),
                )
            return ModelTurn(response_id="parent-2", final_text="parent done")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=RoutingAdapter(),  # type: ignore[arg-type]
        runtime_config_provider=runtime_provider(
            AgentRuntimeConfig(
                definition_id="parent",
                prompt=PromptSpec(persona_segments=("PARENT",)),
                tools=(tool_binding("agent.spawn"),),
            )
        ),
        subagent_definitions={
            "child": SubagentDefinition(prompt=PromptSpec(persona_segments=(child_marker,)))
        },
        model_calls_file=True,
        model_payload_file=True,
    )
    result = loop.run_once("delegate")

    corpora = sorted((tmp_path / "runs").glob(f"*/{MODEL_PAYLOADS_FILENAME}"))
    assert len(corpora) == 2  # the parent's and the child's, each in its own directory
    child_paths = [path for path in corpora if path.parent != result.run_dir]
    assert len(child_paths) == 1
    child_records = [
        json.loads(line)
        for line in child_paths[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert child_records, "the child inherited the switch"
    assert all(record["root_run_id"] == result.run_id for record in child_records)
    assert all(record["run_id"] != result.run_id for record in child_records)


def test_the_corpus_never_carries_the_configured_endpoint(tmp_path: Path) -> None:
    """The artifact-boundary twin of the ledger's endpoint test, with the same witness-first
    shape: the run really was gateway-configured (the ledger's model block proves it), and then
    the corpus bytes carry neither the URL nor its host."""
    from monoid_agent_kernel.core.spec import ModelConfig

    endpoint = "https://gateway.internal.example/tenant-a/llm/turns"
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=_Adapter(),  # type: ignore[arg-type]
        runtime_config_provider=runtime_provider(
            runtime_config(
                "run.finish",
                model=ModelConfig(
                    provider="gateway", model="gpt-witness", gateway_url=endpoint
                ),
            )
        ),
        model_calls_file=True,
        model_payload_file=True,
    )
    result = loop.run_once("hi")

    ledger_raw = (result.run_dir / MODEL_CALLS_FILENAME).read_text(encoding="utf-8")
    assert '"provider":"gateway"' in ledger_raw  # witness: the config reached the receipt

    corpus = (result.run_dir / MODEL_PAYLOADS_FILENAME).read_text(encoding="utf-8")
    chunk_dir = result.run_dir / MODEL_PAYLOADS_DIRNAME
    everything = corpus + "".join(
        stored.read_text(encoding="utf-8") for stored in chunk_dir.iterdir()
    ) if chunk_dir.exists() else corpus

    # Witness: the preimage's model block really came from the config block that carries the
    # endpoint -- the resolved provider term cannot serve here, because the adapter's declared
    # provider_name outranks the config's.
    assert '"model":"gpt-witness"' in everything
    assert endpoint not in everything
    assert "gateway.internal.example" not in everything


def test_a_run_with_only_the_corpus_still_indexes_its_responses(tmp_path: Path) -> None:
    """The ledger switch off, the corpus on: responses still carry call indices (the counter is
    the recorder's, not the ledger file's) and the run directory validates."""
    result = _loop(tmp_path, _Adapter(), model_calls_file=False).run_once("hi")
    records = _records(result.run_dir)
    responses = _by_kind(records, MODEL_RESPONSE_KIND)

    assert not (result.run_dir / MODEL_CALLS_FILENAME).exists()
    assert len(responses) == 1
    assert responses[0]["call_index"] == 0
    assert validate_run_dir(result.run_dir) == []


def test_the_whole_run_directory_validates_with_both_sidecars_on(tmp_path: Path) -> None:
    result = _loop(tmp_path, _Adapter()).run_once("hi")

    assert validate_run_dir(result.run_dir) == []


def _offloadable_call() -> tuple[bytes, str, str]:
    """A settled call whose request extracts exactly one chunk, and that chunk is directory-sized."""
    payload = {
        "monoid.model-request-digest.v1": {
            "system_prompt": "s" * (PAYLOAD_OFFLOAD_THRESHOLD_BYTES + 4096),
            "tools": [],
            "messages": [],
        }
    }
    preimage = CANONICAL_JSON_ENCODER.encode(payload).encode("utf-8")
    digest = hashlib.sha256(preimage).hexdigest()
    split = split_request_payload(preimage, digest)
    assert split is not None
    sha = next(
        one for one, chunk in split.chunks.items() if len(chunk) > PAYLOAD_OFFLOAD_THRESHOLD_BYTES
    )
    return preimage, digest, sha


@pytest.mark.parametrize("link_kind", ["hardlink", "symlink"])
def test_a_link_planted_at_a_chunk_name_is_refused_not_reported_as_stored(
    tmp_path: Path, link_kind: str
) -> None:
    """Content-addressed names are the *most* predictable names in the run directory -- the same
    surface yields the same sha on every run -- so the write-once short-circuit is exactly where an
    indirection gets planted. Answering "already stored" there means the chunk is never written,
    never retried (write-once), and every reader resolves whoever's bytes the link names, while the
    arm believes it succeeded. The JSONL twin has refused this since W6-1; this is the file writer.
    """
    preimage, digest, sha = _offloadable_call()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "someone-elses.bin"
    target.write_bytes(b"theirs")
    run_dir = tmp_path / "runs" / "run-1"
    (run_dir / MODEL_PAYLOADS_DIRNAME).mkdir(parents=True)
    planted = run_dir / MODEL_PAYLOADS_DIRNAME / sha
    if link_kind == "hardlink":
        os.link(target, planted)
    else:
        try:
            os.symlink(target, planted)
        except OSError as error:
            pytest.skip(f"file symlinks are unavailable: {error}")
    recorder = _standalone_recorder(tmp_path)

    recorder.record_settled_call(
        SettledModelCall(
            receipt=ModelCallReceipt(
                request_digest=digest,
                digest_generation="monoid.model-request-digest.v1",
            ),
            request_preimage=preimage,
            turn=ModelTurn(response_id="r", final_text="answer"),
        )
    )
    recorder.close()

    assert target.read_bytes() == b"theirs"
    assert recorder._model_payloads_failed is True
    corpus = run_dir / MODEL_PAYLOADS_FILENAME
    records = _records(run_dir) if corpus.exists() else []
    # No record may reference a chunk that was never stored.
    assert _by_kind(records, MODEL_REQUEST_KIND) == []
    assert (run_dir / MODEL_CALLS_FILENAME).exists()
