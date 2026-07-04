from __future__ import annotations

from support.studio_harness import (
    FakeModelAdapter,
    ModelTurn,
    NativeAgentError,
    Path,
    StudioConfig,
    StudioServer,
    _wait_proposal,
    _wait_settled,
    fake_tool_call,
    pytest,
    time,
)

pytestmark = pytest.mark.integration


def test_agent_write_is_staged_then_applied(tmp_path: Path) -> None:
    # The propose->apply loop: the agent writes a file (staged in the overlay, not on disk),
    # Studio surfaces it as a diff, and apply materializes it into the workspace.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    target = workspace / "OUT.md"

    fake = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("fs_write", {"path": "OUT.md", "content": "hello\n"}, "c1"),)),
            ModelTurn(final_text="Wrote OUT.md."),
        ]
    )
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    server.start()
    try:
        run_id = server.start_chat("create OUT.md")["run_id"]
        _wait_settled(server, run_id, 1)

        # Staged, not yet on disk.
        assert not target.exists()
        proposal = _wait_proposal(server, run_id)
        assert proposal["ready"]
        assert "OUT.md" in proposal["diff"]
        assert "hello" in proposal["diff"]

        # Approve & apply -> the file lands in the workspace.
        result = server.apply(run_id)
        assert result["status"] != "conflict"
        assert "OUT.md" in str(result.get("applied_paths"))
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "hello\n"
    finally:
        server.shutdown()


def test_partial_approval_applies_only_selected_paths(tmp_path: Path) -> None:
    # R9: the per-file approval gate. The agent stages two files; Studio approves only one, so
    # apply writes that file and reports the other as skipped (never touching disk for it).
    workspace = tmp_path / "ws"
    workspace.mkdir()

    fake = FakeModelAdapter(
        turns=[
            ModelTurn(
                tool_calls=(
                    fake_tool_call("fs_write", {"path": "KEEP.md", "content": "keep\n"}, "c1"),
                    fake_tool_call("fs_write", {"path": "DROP.md", "content": "drop\n"}, "c2"),
                )
            ),
            ModelTurn(final_text="Wrote two files."),
        ]
    )
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    server.start()
    try:
        run_id = server.start_chat("stage two files")["run_id"]
        _wait_settled(server, run_id, 1)
        _wait_proposal(server, run_id)

        result = server.apply(run_id, approved_paths=("KEEP.md",))
        assert result["status"] != "conflict"
        assert "KEEP.md" in str(result.get("applied_paths"))
        assert "DROP.md" in str(result.get("skipped_paths"))
        assert (workspace / "KEEP.md").exists()
        assert not (workspace / "DROP.md").exists()
    finally:
        server.shutdown()


def test_export_package_returns_digest_receipt_fetched_as_bytes(tmp_path: Path) -> None:
    # R9 (fundamental): export returns a RECEIPT (digest) — never a server path — and the bytes are
    # fetched back by digest through the data-returning seam (works co-located or remote).
    import io
    import tarfile

    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("fs_write", {"path": "OUT.md", "content": "hi\n"}, "c1"),)),
            ModelTurn(final_text="done"),
        ]
    )
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    server.start()
    try:
        run_id = server.start_chat("write OUT.md")["run_id"]
        _wait_settled(server, run_id, 1)
        _wait_proposal(server, run_id)

        receipt = server.export_package(run_id)
        # The receipt is a handle, not a path — no run_dir path leaks across the boundary.
        assert "package_path" not in receipt
        assert len(receipt["digest"]) == 64
        assert receipt["size_bytes"] > 0

        data, name = server.read_artifact(run_id, receipt["digest"])
        # The fetched bytes match the digest (content-addressed self-verification).
        import hashlib

        assert hashlib.sha256(data).hexdigest() == receipt["digest"]
        assert name.endswith(".tar")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r") as archive:
            names = archive.getnames()
        assert "proposal.package.json" in names
        assert any(n == "proposal.json" for n in names)

        # A malformed digest is rejected (ValueError → 400); an unknown well-formed one is
        # not-found (KeyError → 404).
        with pytest.raises(ValueError):
            server.read_artifact(run_id, "not-a-digest")
        with pytest.raises(KeyError):
            server.read_artifact(run_id, "f" * 64)
    finally:
        server.shutdown()


def test_continue_chat_resumes_a_parked_session_after_restart(tmp_path: Path) -> None:
    # The studio "continue an old chat" path: a multi-turn session parked awaiting input is
    # evicted from memory (simulating a process restart), then continue_chat transparently
    # resumes it from the checkpoint and delivers the follow-up.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", final_text="first"),
            ModelTurn(response_id="r2", final_text="second"),
        ]
    )
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    server.start()
    try:
        run_id = server.start_chat("hello")["run_id"]
        _wait_settled(server, run_id, 1)

        def _await_state(target: str, timeout: float = 10.0) -> bool:
            deadline = time.time() + timeout
            while time.time() < deadline:
                if server.run_status(run_id).get("state") == target:
                    return True
                time.sleep(0.05)
            return False

        assert _await_state("awaiting_input")

        # Simulate a restart: drop the in-memory record. A bare send_message would now KeyError;
        # continue_chat must resume from the durable checkpoint first.
        backend = server._backend
        assert backend.checkpoint_store.latest(run_id) is not None
        with backend._lock:
            backend._records.pop(run_id)

        result = server.continue_chat(run_id, "again")
        assert result["status"] == "queued"

        # The resumed session threads the follow-up as a real second model turn, with the
        # conversation rebuilt from the checkpoint (user "hello" → assistant "first" → user "again").
        def _again_threaded() -> bool:
            return any(
                msg.get("role") == "user" and msg.get("content") == "again"
                for req in fake.requests
                for msg in (req.messages or [])
            )

        deadline = time.time() + 10.0
        while time.time() < deadline and not _again_threaded():
            time.sleep(0.05)
        assert _again_threaded()
        # And the prior turn's assistant reply survived the restart (proves checkpoint restore, not
        # a fresh conversation).
        assert any(
            msg.get("role") == "assistant" and msg.get("content") == "first"
            for req in fake.requests
            for msg in (req.messages or [])
        )
    finally:
        server.shutdown()


def test_start_chat_attaches_image_and_forwards_resolved_block(tmp_path: Path) -> None:
    # R13: an attached image is persisted under the workspace and forwarded to a multimodal
    # adapter as a resolved base64 block (the loop resolves the by-reference source_ref).
    import base64

    from monoid_agent_kernel.providers.fake import FakeMultimodalModelAdapter

    png_1x1 = base64.b64encode(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk"
            "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
    ).decode("ascii")

    workspace = tmp_path / "ws"
    workspace.mkdir()
    adapter = FakeMultimodalModelAdapter(turns=[ModelTurn(final_text="I see a 1x1 image.")])
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: adapter,
    )
    server.start()
    try:
        result = server.start_chat(
            "what is this?",
            [{"name": "pic.png", "mime": "image/png", "data_b64": png_1x1}],
        )
        run_id = result["run_id"]
        _wait_settled(server, run_id, 1)

        # Inline ingress: the studio writes NO attachment file into the workspace — the bytes ride
        # a data: URI and the core normalizes them to a content-addressed blob.
        assert not (workspace / ".studio-attachments").exists()

        # The adapter received a resolved base64 image block on the user turn.
        def _image_forwarded() -> bool:
            return any(
                isinstance(msg.get("content"), list)
                and any(isinstance(b, dict) and b.get("type") == "image" for b in msg["content"])
                for req in adapter.requests
                for msg in req.messages
            )

        assert _image_forwarded()
    finally:
        server.shutdown()


def test_file_viewer_previews_images(studio: StudioServer) -> None:
    """The file viewer flags an image for inline <img> preview and serves its raw bytes via the
    read_image (/api/file-raw) seam, while text files keep rendering inline and traversal / non-image
    reads are refused."""
    import base64

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    ws = studio.workspace
    (ws / "pic.png").write_bytes(png)
    (ws / "code.py").write_text("print('hi')\n", encoding="utf-8")

    meta = studio.read_file("pic.png")  # image: flagged for <img>, no inline content
    assert meta["image"] is True and meta["mime"] == "image/png" and meta["content"] == ""

    code = studio.read_file("code.py")  # text: still rendered inline, not an image
    assert code["image"] is False and "print" in code["content"]

    data, mime = studio.read_image("pic.png")  # raw bytes for /api/file-raw
    assert data == png and mime == "image/png"

    with pytest.raises(NativeAgentError):
        studio.read_image("code.py")  # non-image refused by the raw endpoint
    with pytest.raises(NativeAgentError):
        studio.read_image("../escape.png")  # traversal rejected


def test_proposal_panel_previews_proposed_image(tmp_path: Path) -> None:
    """A generated image staged in the proposal (not yet on disk) is previewable: proposal_image
    returns its bytes + content-type from the token-scoped proposal snapshot. Uses an SVG (a text
    image) so the path is exercised end-to-end without a real binary plot."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/></svg>'
    fake = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("fs_write", {"path": "chart.svg", "content": svg}, "c1"),)),
            ModelTurn(final_text="Wrote chart.svg."),
        ]
    )
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    server.start()
    try:
        run_id = server.start_chat("draw a chart")["run_id"]
        _wait_settled(server, run_id, 1)
        proposal = _wait_proposal(server, run_id)
        assert "chart.svg" in (proposal.get("changed_paths") or [])

        # Not on disk yet (propose mode) — but previewable straight from the proposal snapshot.
        assert not (workspace / "chart.svg").exists()
        data, mime = server.proposal_image(run_id, "chart.svg")
        assert mime == "image/svg+xml" and data.decode("utf-8") == svg

        with pytest.raises(NativeAgentError):
            server.proposal_image(run_id, "notes.txt")  # non-image refused
    finally:
        server.shutdown()
