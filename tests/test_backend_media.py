from __future__ import annotations

from support.backend_harness import (
    BackendRunRequest,
    FakeMultimodalModelAdapter,
    ImagePart,
    ModelTurn,
    Path,
    RunnerBackend,
    TextPart,
    _PNG_1x1,
    _default_config,
    _token_manager,
    _workspace,
    eventually,
    pytest,
)

pytestmark = pytest.mark.integration


def test_send_message_forwards_multimodal_image_by_reference(tmp_path: Path) -> None:
    # The string-only blocker fix: send_message accepts content parts; the loop resolves the
    # workspace image reference to a base64 wire block and forwards it to a multimodal adapter.
    workspace = _workspace(tmp_path)
    workspace.joinpath("shot.png").write_bytes(_PNG_1x1)
    run_root = tmp_path / "runs"

    adapters: list = []

    def factory(spec, llm_gateway_token):
        del spec, llm_gateway_token
        adapter = FakeMultimodalModelAdapter(
            turns=[ModelTurn(response_id="r1", final_text="first"), ModelTurn(response_id="r2", final_text="an image")]
        )
        adapters.append(adapter)
        return adapter

    backend = RunnerBackend(
        run_root=run_root,
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
    )
    backend.idle_timeout_s = 30.0
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hi",
            runtime_config=_default_config(),
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token

    assert eventually(lambda: backend._record(run_id).status == "awaiting_input", timeout_s=20)

    # Deliver a multimodal follow-up: a workspace image reference + a text part.
    backend.send_message(
        run_id,
        token,
        [TextPart("describe this"), ImagePart(source_ref="shot.png", mime_type="image/png")],
    )

    # The resolved request carries a base64 image block (resolution happened at wire-build time).
    def _image_block_forwarded() -> bool:
        for adapter in adapters:
            for req in adapter.requests:
                for msg in req.messages:
                    content = msg.get("content")
                    if isinstance(content, list) and any(
                        isinstance(b, dict) and b.get("type") == "image" for b in content
                    ):
                        return True
        return False

    assert eventually(_image_block_forwarded, timeout_s=20)

    backend.cancel_run(run_id, token)
    assert backend.wait_for_run(run_id, timeout_s=20) in {"completed", "limited", "failed"}


def test_send_message_inline_media_is_blobified_before_queue(tmp_path: Path) -> None:
    # R13b edge fix: an inline (data:) follow-up is normalized to a durable blob at send_message —
    # BEFORE it enters the queue — so the queue (and any checkpoint of it) never carries the bytes
    # inline. The loop then resolves the blob: ref via the checkpoint-store blob_reader fallback.
    import base64
    import hashlib

    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    adapters: list = []

    def factory(spec, llm_gateway_token):
        del spec, llm_gateway_token
        adapter = FakeMultimodalModelAdapter(
            turns=[ModelTurn(response_id="r1", final_text="first"), ModelTurn(response_id="r2", final_text="img")]
        )
        adapters.append(adapter)
        return adapter

    backend = RunnerBackend(
        run_root=run_root,
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
    )
    backend.idle_timeout_s = 30.0
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a", user_id="user_a", workspace_root=workspace,
            instruction="hi", runtime_config=_default_config(), multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token

    assert eventually(lambda: backend._record(run_id).status == "awaiting_input", timeout_s=20)

    data_uri = "data:image/png;base64," + base64.b64encode(_PNG_1x1).decode()
    backend.send_message(run_id, token, [TextPart("look"), ImagePart(source_ref=data_uri, mime_type="image/png")])

    # The bytes were persisted to the blob store at ingress (so the queue held only a blob: ref).
    sha = hashlib.sha256(_PNG_1x1).hexdigest()
    assert backend.checkpoint_store.get_blob(run_id, sha) == _PNG_1x1

    # The loop resolves that blob (via the store fallback) and forwards the image.
    def _image_forwarded() -> bool:
        return any(
            isinstance(m.get("content"), list)
            and any(isinstance(b, dict) and b.get("type") == "image" for b in m["content"])
            for a in adapters for r in a.requests for m in r.messages
        )

    assert eventually(_image_forwarded, timeout_s=20)
    # The durable log carries a blob: ref, never the data: bytes.
    record = backend._record(run_id)
    assert any(
        isinstance(m.get("content"), list)
        and any(isinstance(b, dict) and str(b.get("source_ref", "")).startswith("blob:") for b in m["content"])
        for m in record.loop._session.state.messages  # type: ignore[union-attr]
    )

    backend.cancel_run(run_id, token)
    assert backend.wait_for_run(run_id, timeout_s=20) in {"completed", "limited", "failed"}


def test_queued_multimodal_message_round_trips_through_checkpoint() -> None:
    # The queue + checkpoint stay JSON-native, so a parked multimodal message survives a restart.
    from native_agent_runner.core.checkpoint import RunCheckpoint
    from native_agent_runner.reference.backend.service import (
        _normalize_inbound_message,
        _queued_message_to_loop_input,
    )

    wire = _normalize_inbound_message(
        [TextPart("describe"), ImagePart(source_ref="shot.png", mime_type="image/png")]
    )
    assert isinstance(wire, list)  # JSON-native: a list of content-part dicts

    # Survives the durable checkpoint round-trip with no dataclass (de)serialization.
    ckpt = RunCheckpoint(run_id="r", queued_messages=["plain text", wire])
    restored = RunCheckpoint.from_json(ckpt.to_json())
    assert restored is not None
    assert restored.queued_messages[0] == "plain text"

    # On dequeue it rebuilds the typed parts for the loop.
    loop_input = _queued_message_to_loop_input(restored.queued_messages[1])
    assert isinstance(loop_input, tuple)
    assert isinstance(loop_input[0], TextPart) and isinstance(loop_input[1], ImagePart)
    assert loop_input[1].source_ref == "shot.png"
