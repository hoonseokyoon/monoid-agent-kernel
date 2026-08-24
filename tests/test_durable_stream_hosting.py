from __future__ import annotations

import contextvars
import hashlib
import threading
from dataclasses import replace

import pytest

from monoid_agent_kernel.conformance import run_durable_stream_store_contract
from monoid_agent_kernel.core.authority import ActivationWriteAuthority
from monoid_agent_kernel.core.model_invocation import logical_model_call_id
from monoid_agent_kernel.core.model_stream import (
    ModelStreamContext,
    ModelStreamDelta,
    ModelStreamDispatchAwareWriter,
    ModelStreamOutcome,
    ModelStreamSettlementAwareWriter,
    ModelStreamStatus,
)
from monoid_agent_kernel.hosting import (
    DurableModelStreamObserver,
    DurableStreamAppendResult,
    DurableStreamChunk,
    DurableStreamHead,
    DurableStreamIdentity,
    DurableStreamOpenResult,
    DurableStreamReadChunk,
    DurableStreamReadResult,
    DurableStreamResetResult,
    DurableStreamSealResult,
    DurableStreamWriteError,
    WriterToken,
    durable_model_stream_id,
)


_MARKER = contextvars.ContextVar("durable_stream_marker", default="missing")


class _MemoryStreamStore:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.heads: dict[DurableStreamIdentity, DurableStreamHead] = {}
        self.chunks: dict[tuple[DurableStreamIdentity, int], list[DurableStreamReadChunk]] = {}
        self.reset_receipts: dict[tuple[DurableStreamIdentity, str], int] = {}
        self.appended = threading.Event()
        self.append_contexts: list[str] = []
        self.reject_append_status: str | None = None
        self.current_tokens: dict[str, WriterToken] = {}

    def _token_is_current(self, writer_token: WriterToken) -> bool:
        current = self.current_tokens.setdefault(writer_token.run_id, writer_token)
        return current == writer_token

    def replace_writer(self, run_id: str) -> WriterToken:
        with self.lock:
            previous = self.current_tokens[run_id]
            current = WriterToken(
                run_id=run_id,
                owner_id="replacement-worker",
                generation=previous.generation + 1,
            )
            self.current_tokens[run_id] = current
            return current

    def open(
        self,
        identity: DurableStreamIdentity,
        *,
        writer_token: WriterToken,
    ) -> DurableStreamOpenResult:
        if identity.run_id != writer_token.run_id:
            return DurableStreamOpenResult(status="fenced")
        with self.lock:
            if not self._token_is_current(writer_token):
                return DurableStreamOpenResult(status="fenced")
            head = self.heads.get(identity)
            if head is not None:
                return DurableStreamOpenResult(
                    status="already_open" if head.state == "open" else "sealed",
                    head=head,
                )
            head = DurableStreamHead(
                identity=identity,
                generation=1,
                cursor_bytes=0,
                next_chunk_sequence=1,
                state="open",
            )
            self.heads[identity] = head
            return DurableStreamOpenResult(status="opened", head=head)

    def reset(
        self,
        identity: DurableStreamIdentity,
        *,
        expected_generation: int,
        reset_id: str,
        writer_token: WriterToken,
    ) -> DurableStreamResetResult:
        if identity.run_id != writer_token.run_id:
            return DurableStreamResetResult(status="fenced")
        with self.lock:
            if not self._token_is_current(writer_token):
                return DurableStreamResetResult(status="fenced")
            head = self.heads.get(identity)
            if head is None:
                return DurableStreamResetResult(status="conflict")
            receipt = self.reset_receipts.get((identity, reset_id))
            if receipt is not None:
                return DurableStreamResetResult(
                    status="already_reset",
                    head=head,
                    applied_generation=receipt,
                )
            if head.generation != expected_generation:
                return DurableStreamResetResult(status="old_generation", head=head)
            generation = head.generation + 1
            head = DurableStreamHead(
                identity=identity,
                generation=generation,
                cursor_bytes=0,
                next_chunk_sequence=1,
                state="open",
            )
            self.heads[identity] = head
            self.reset_receipts[(identity, reset_id)] = generation
            return DurableStreamResetResult(
                status="reset",
                head=head,
                applied_generation=generation,
            )

    def append(
        self,
        identity: DurableStreamIdentity,
        *,
        generation: int,
        start_offset: int,
        data: bytes,
        writer_token: WriterToken,
    ) -> DurableStreamAppendResult:
        if identity.run_id != writer_token.run_id:
            return DurableStreamAppendResult(status="fenced")
        if self.reject_append_status is not None:
            return DurableStreamAppendResult(status=self.reject_append_status)  # type: ignore[arg-type]
        with self.lock:
            if not self._token_is_current(writer_token):
                return DurableStreamAppendResult(status="fenced")
            head = self.heads[identity]
            if generation != head.generation:
                return DurableStreamAppendResult(status="old_generation", head=head)
            values = self.chunks.setdefault((identity, generation), [])
            existing = next(
                (value for value in values if value.chunk.start_offset == start_offset),
                None,
            )
            sha256 = hashlib.sha256(data).hexdigest()
            if existing is not None:
                if existing.chunk.sha256 != sha256 or existing.data != data:
                    return DurableStreamAppendResult(status="conflict", head=head)
                return DurableStreamAppendResult(
                    status="already_committed",
                    head=head,
                    chunk=existing.chunk,
                )
            if head.state == "sealed":
                return DurableStreamAppendResult(status="sealed", head=head)
            if start_offset != head.cursor_bytes:
                return DurableStreamAppendResult(status="gap", head=head)
            chunk = DurableStreamChunk(
                identity=identity,
                generation=generation,
                sequence=head.next_chunk_sequence,
                start_offset=start_offset,
                end_offset=start_offset + len(data),
                sha256=sha256,
                locator=f"memory:{sha256}",
            )
            head = replace(
                head,
                cursor_bytes=chunk.end_offset,
                next_chunk_sequence=head.next_chunk_sequence + 1,
            )
            self.heads[identity] = head
            values.append(DurableStreamReadChunk(chunk=chunk, data=data))
            self.append_contexts.append(_MARKER.get())
            self.appended.set()
            return DurableStreamAppendResult(status="committed", head=head, chunk=chunk)

    def seal(
        self,
        identity: DurableStreamIdentity,
        *,
        generation: int,
        final_size_bytes: int,
        final_sha256: str,
        writer_token: WriterToken,
    ) -> DurableStreamSealResult:
        if identity.run_id != writer_token.run_id:
            return DurableStreamSealResult(status="fenced")
        with self.lock:
            if not self._token_is_current(writer_token):
                return DurableStreamSealResult(status="fenced")
            head = self.heads[identity]
            if generation != head.generation:
                return DurableStreamSealResult(status="old_generation", head=head)
            if head.state == "sealed":
                return DurableStreamSealResult(
                    status=(
                        "already_sealed"
                        if head.cursor_bytes == final_size_bytes
                        and head.final_sha256 == final_sha256
                        else "conflict"
                    ),
                    head=head,
                )
            data = b"".join(
                value.data for value in self.chunks.get((identity, generation), [])
            )
            if len(data) != final_size_bytes or hashlib.sha256(data).hexdigest() != final_sha256:
                return DurableStreamSealResult(status="conflict", head=head)
            head = replace(head, state="sealed", final_sha256=final_sha256)
            self.heads[identity] = head
            return DurableStreamSealResult(status="sealed", head=head)

    def read_after(
        self,
        identity: DurableStreamIdentity,
        *,
        generation: int,
        cursor: int,
        limit: int = 100,
    ) -> DurableStreamReadResult:
        with self.lock:
            head = self.heads.get(identity)
            if head is None:
                return DurableStreamReadResult(
                    status="not_found",
                    requested_generation=generation,
                    requested_cursor=cursor,
                )
            if generation < head.generation:
                return DurableStreamReadResult(
                    status="reset",
                    requested_generation=generation,
                    requested_cursor=cursor,
                    head=head,
                )
            if generation > head.generation or cursor > head.cursor_bytes:
                return DurableStreamReadResult(
                    status="gap",
                    requested_generation=generation,
                    requested_cursor=cursor,
                    head=head,
                )
            values = self.chunks.get((identity, generation), [])
            boundaries = {0, *(value.chunk.end_offset for value in values)}
            if cursor not in boundaries:
                return DurableStreamReadResult(
                    status="gap",
                    requested_generation=generation,
                    requested_cursor=cursor,
                    head=head,
                )
            selected = tuple(
                value for value in values if value.chunk.start_offset >= cursor
            )[:limit]
            return DurableStreamReadResult(
                status="ok",
                requested_generation=generation,
                requested_cursor=cursor,
                head=head,
                chunks=selected,
            )


def _context() -> ModelStreamContext:
    return ModelStreamContext(
        run_id="run-stream-hosting",
        root_run_id="run-stream-hosting",
        turn_id="turn-1",
        stream_id="stream-1",
        step=1,
        provider="test",
        model="test-model",
        started_at="2026-08-25T00:00:00Z",
    )


def _observer(
    store: _MemoryStreamStore,
    authority: ActivationWriteAuthority,
    **kwargs: object,
) -> DurableModelStreamObserver:
    return DurableModelStreamObserver(
        store,
        writer_token=WriterToken(
            run_id="run-stream-hosting",
            owner_id="worker-1",
            generation=1,
        ),
        write_authority=authority,
        **kwargs,
    )


class _MemoryConformanceHarness:
    def __init__(self, run_id: str) -> None:
        self.store = _MemoryStreamStore()
        self.writer_token = WriterToken(
            run_id=run_id,
            owner_id="contract-worker",
            generation=1,
        )

    def replace_writer(self) -> WriterToken:
        self.writer_token = self.store.replace_writer(self.writer_token.run_id)
        return self.writer_token


def test_stream_values_reject_non_utf8_and_inconsistent_seal() -> None:
    identity = DurableStreamIdentity(
        run_id="run-1",
        stream_id="stream-1",
        logical_call_id="call-1",
        channel="host_private",
    )
    with pytest.raises(ValueError, match="channel"):
        replace(identity, channel="private content")
    with pytest.raises(ValueError, match="complete UTF-8"):
        DurableStreamReadChunk(
            chunk=DurableStreamChunk(
                identity=identity,
                generation=1,
                sequence=1,
                start_offset=0,
                end_offset=1,
                sha256="0" * 64,
                locator="memory:invalid",
            ),
            data=b"\xff",
        )
    with pytest.raises(ValueError, match="chunk SHA-256"):
        DurableStreamReadChunk(
            chunk=DurableStreamChunk(
                identity=identity,
                generation=1,
                sequence=1,
                start_offset=0,
                end_offset=1,
                sha256="0" * 64,
                locator="memory:digest-mismatch",
            ),
            data=b"x",
        )
    with pytest.raises(ValueError, match="sealed"):
        DurableStreamHead(
            identity=identity,
            generation=1,
            cursor_bytes=0,
            next_chunk_sequence=1,
            state="open",
            final_sha256=hashlib.sha256(b"").hexdigest(),
        )


def test_durable_stream_store_conformance_is_reusable() -> None:
    outcomes = run_durable_stream_store_contract(_MemoryConformanceHarness)

    assert len(outcomes) == 3
    assert all(outcome.passed for outcome in outcomes), [outcome.to_json() for outcome in outcomes]


def test_model_stream_observer_coalesces_utf8_batches_and_seals_each_lane() -> None:
    store = _MemoryStreamStore()
    authority = ActivationWriteAuthority()
    observer = _observer(
        store,
        authority,
        chunk_bytes=8,
        flush_interval_s=10,
        max_buffer_bytes=64,
    )
    writer = observer.open(_context())

    writer.push(ModelStreamDelta(channel="output", text="hello "))
    writer.push(ModelStreamDelta(channel="output", text="세계"))
    writer.push(ModelStreamDelta(channel="reasoning", text="why"))
    writer.close(ModelStreamOutcome(status="completed", final_text="hello 세계"))

    output = next(identity for identity in store.heads if identity.channel == "output")
    reasoning = next(identity for identity in store.heads if identity.channel == "reasoning")
    assert output.stream_id == durable_model_stream_id(_context().run_id, _context().turn_id)
    assert output.stream_id != _context().stream_id
    output_bytes = b"".join(
        value.data for value in store.chunks[(output, store.heads[output].generation)]
    )
    reasoning_bytes = b"".join(
        value.data for value in store.chunks[(reasoning, store.heads[reasoning].generation)]
    )
    assert output_bytes.decode() == "hello 세계"
    assert reasoning_bytes == b"why"
    assert all(value.chunk.size_bytes <= 8 for values in store.chunks.values() for value in values)
    assert store.heads[output].final_sha256 == hashlib.sha256(output_bytes).hexdigest()
    assert store.heads[reasoning].final_sha256 == hashlib.sha256(reasoning_bytes).hexdigest()


def test_model_stream_observer_flushes_on_time_and_copies_context() -> None:
    store = _MemoryStreamStore()
    authority = ActivationWriteAuthority()
    marker = _MARKER.set("tenant-a")
    try:
        writer = _observer(
            store,
            authority,
            chunk_bytes=1024,
            flush_interval_s=0.02,
            max_buffer_bytes=2048,
        ).open(_context())
    finally:
        _MARKER.reset(marker)
    writer.push(ModelStreamDelta(channel="output", text="timed"))

    assert store.appended.wait(2)
    writer.close(ModelStreamOutcome(status="completed", final_text="timed"))
    assert store.append_contexts == ["tenant-a"]


def test_settlement_preparation_flushes_both_lanes_before_recovery_seals_them() -> None:
    store = _MemoryStreamStore()
    context = _context()
    writer = _observer(
        store,
        ActivationWriteAuthority(),
        chunk_bytes=1024,
        flush_interval_s=10,
        max_buffer_bytes=4096,
    ).open(context)
    assert isinstance(writer, ModelStreamSettlementAwareWriter)
    writer.push(ModelStreamDelta(channel="reasoning", text="complete reasoning"))
    writer.push(ModelStreamDelta(channel="output", text="complete answer"))

    writer.prepare_settlement()

    identities = {identity.channel: identity for identity in store.heads}
    expected = {"output": b"complete answer", "reasoning": b"complete reasoning"}
    for channel, data in expected.items():
        identity = identities[channel]
        head = store.heads[identity]
        assert head.generation == 1
        assert head.state == "open"
        assert b"".join(chunk.data for chunk in store.chunks[(identity, 1)]) == data

    # A process can disappear after invocation settlement and before observer close. Recovery
    # must preserve and seal the fully prepared generation without reconstructing reasoning.
    writer.abort()  # type: ignore[attr-defined]
    current = store.replace_writer(context.run_id)
    recovered = DurableModelStreamObserver(
        store,
        writer_token=current,
        write_authority=ActivationWriteAuthority(),
    ).open(context)
    recovered.close(ModelStreamOutcome(status="completed", final_text="complete answer"))

    for channel, data in expected.items():
        identity = identities[channel]
        assert store.heads[identity].generation == 1
        assert store.heads[identity].state == "sealed"
        assert b"".join(chunk.data for chunk in store.chunks[(identity, 1)]) == data


def test_failed_settlement_preparation_aborts_without_sealing_partial_content() -> None:
    store = _MemoryStreamStore()
    writer = _observer(
        store,
        ActivationWriteAuthority(),
        chunk_bytes=1024,
        flush_interval_s=10,
        max_buffer_bytes=4096,
    ).open(_context())
    writer.push(ModelStreamDelta(channel="reasoning", text="uncommitted reasoning"))
    store.reject_append_status = "conflict"

    with pytest.raises(DurableStreamWriteError, match="flush worker failed"):
        writer.prepare_settlement()  # type: ignore[attr-defined]
    writer.abort()  # type: ignore[attr-defined]

    for head in store.heads.values():
        assert head.state == "open"
        assert head.cursor_bytes == 0


def test_model_stream_observer_fencing_revokes_activation_authority() -> None:
    store = _MemoryStreamStore()
    store.reject_append_status = "fenced"
    authority = ActivationWriteAuthority()
    writer = _observer(
        store,
        authority,
        chunk_bytes=4,
        flush_interval_s=0.01,
        max_buffer_bytes=16,
    ).open(_context())
    writer.push(ModelStreamDelta(channel="output", text="stop"))

    with pytest.raises(DurableStreamWriteError, match="flush worker failed"):
        writer.close(ModelStreamOutcome(status="failed"))
    assert authority.revoked is True


def test_model_stream_observer_local_revocation_fences_queued_flush() -> None:
    store = _MemoryStreamStore()
    authority = ActivationWriteAuthority()
    writer = _observer(
        store,
        authority,
        chunk_bytes=1024,
        flush_interval_s=10,
        max_buffer_bytes=2048,
    ).open(_context())
    writer.push(ModelStreamDelta(channel="output", text="queued"))

    authority.revoke()

    with pytest.raises(DurableStreamWriteError, match="flush worker failed"):
        writer.close(ModelStreamOutcome(status="failed"))
    identity = next(iter(store.heads))
    assert store.heads[identity].cursor_bytes == 0
    assert store.appended.is_set() is False


def test_model_stream_observer_late_channel_cannot_open_after_close() -> None:
    store = _MemoryStreamStore()
    authority = ActivationWriteAuthority()
    writer = _observer(store, authority).open(_context())
    writer.close(ModelStreamOutcome(status="completed", final_text=""))

    with pytest.raises(DurableStreamWriteError, match="closed"):
        writer.push(ModelStreamDelta(channel="reasoning", text="late"))
    assert {identity.channel for identity in store.heads} == {"output"}


def test_model_stream_observer_resets_abandoned_open_lanes_before_replacement() -> None:
    store = _MemoryStreamStore()
    authority = ActivationWriteAuthority()
    context = _context()
    token = WriterToken(
        run_id=context.run_id,
        owner_id="worker-1",
        generation=1,
    )
    stream_id = durable_model_stream_id(context.run_id, context.turn_id)
    logical_call_id = logical_model_call_id(context.run_id, context.turn_id)
    output = DurableStreamIdentity(
        run_id=context.run_id,
        stream_id=stream_id,
        logical_call_id=logical_call_id,
        channel="output",
    )
    reasoning = replace(output, channel="reasoning")
    for identity, data in ((output, b"stale output"), (reasoning, b"stale reasoning")):
        assert store.open(identity, writer_token=token).status == "opened"
        assert store.append(
            identity,
            generation=1,
            start_offset=0,
            data=data,
            writer_token=token,
        ).status == "committed"

    replacement = _observer(
        store,
        authority,
        chunk_bytes=16,
        flush_interval_s=10,
    ).open(context)
    replacement.push(ModelStreamDelta(channel="output", text="fresh"))
    replacement.close(ModelStreamOutcome(status="completed", final_text="fresh"))

    assert store.heads[output].generation == 2
    assert store.heads[reasoning].generation == 2
    assert b"".join(chunk.data for chunk in store.chunks[(output, 2)]) == b"fresh"
    assert store.chunks.get((reasoning, 2), []) == []
    assert store.heads[reasoning].state == "sealed"


def test_model_stream_observer_preserves_open_lanes_when_recovery_emits_no_delta() -> None:
    store = _MemoryStreamStore()
    authority = ActivationWriteAuthority()
    context = _context()
    token = WriterToken(run_id=context.run_id, owner_id="worker-1", generation=1)
    output = DurableStreamIdentity(
        run_id=context.run_id,
        stream_id=durable_model_stream_id(context.run_id, context.turn_id),
        logical_call_id=logical_model_call_id(context.run_id, context.turn_id),
        channel="output",
    )
    assert store.open(output, writer_token=token).status == "opened"
    assert store.append(
        output,
        generation=1,
        start_offset=0,
        data=b"recovered",
        writer_token=token,
    ).status == "committed"

    recovered = _observer(store, authority).open(context)
    recovered.close(ModelStreamOutcome(status="completed", final_text="recovered"))

    assert store.heads[output].generation == 1
    assert store.heads[output].state == "sealed"
    assert b"".join(chunk.data for chunk in store.chunks[(output, 1)]) == b"recovered"


@pytest.mark.parametrize("status", ("failed", "interrupted", "cancelled", "timed_out"))
def test_model_stream_observer_resets_stale_lanes_when_replacement_ends_without_delta(
    status: ModelStreamStatus,
) -> None:
    store = _MemoryStreamStore()
    authority = ActivationWriteAuthority()
    context = _context()
    token = WriterToken(run_id=context.run_id, owner_id="worker-1", generation=1)
    output = DurableStreamIdentity(
        run_id=context.run_id,
        stream_id=durable_model_stream_id(context.run_id, context.turn_id),
        logical_call_id=logical_model_call_id(context.run_id, context.turn_id),
        channel="output",
    )
    reasoning = replace(output, channel="reasoning")
    for identity, data in ((output, b"stale output"), (reasoning, b"stale reasoning")):
        assert store.open(identity, writer_token=token).status == "opened"
        assert store.append(
            identity,
            generation=1,
            start_offset=0,
            data=data,
            writer_token=token,
        ).status == "committed"

    replacement = _observer(store, authority).open(context)
    assert isinstance(replacement, ModelStreamDispatchAwareWriter)
    replacement.begin_dispatch()
    replacement.close(ModelStreamOutcome(status=status))

    for identity in (output, reasoning):
        assert store.heads[identity].generation == 2
        assert store.heads[identity].state == "sealed"
        assert store.heads[identity].cursor_bytes == 0
        assert store.chunks.get((identity, 2), []) == []


@pytest.mark.parametrize("status", ("failed", "interrupted", "cancelled", "timed_out"))
def test_model_stream_observer_preserves_partial_failure_recovery_without_new_dispatch(
    status: ModelStreamStatus,
) -> None:
    store = _MemoryStreamStore()
    authority = ActivationWriteAuthority()
    context = _context()
    token = WriterToken(run_id=context.run_id, owner_id="worker-1", generation=1)
    output = DurableStreamIdentity(
        run_id=context.run_id,
        stream_id=durable_model_stream_id(context.run_id, context.turn_id),
        logical_call_id=logical_model_call_id(context.run_id, context.turn_id),
        channel="output",
    )
    reasoning = replace(output, channel="reasoning")
    prior = ((output, b"partial output"), (reasoning, b"partial reasoning"))
    for identity, data in prior:
        assert store.open(identity, writer_token=token).status == "opened"
        assert store.append(
            identity,
            generation=1,
            start_offset=0,
            data=data,
            writer_token=token,
        ).status == "committed"

    recovered = _observer(store, authority).open(context)
    recovered.close(ModelStreamOutcome(status=status))

    for identity, data in prior:
        assert store.heads[identity].generation == 1
        assert store.heads[identity].state == "sealed"
        assert b"".join(chunk.data for chunk in store.chunks[(identity, 1)]) == data


@pytest.mark.parametrize("final_text", (None, ""))
def test_model_stream_observer_treats_completed_no_text_as_authoritative_empty(
    final_text: str | None,
) -> None:
    store = _MemoryStreamStore()
    authority = ActivationWriteAuthority()
    context = _context()
    token = WriterToken(run_id=context.run_id, owner_id="worker-1", generation=1)
    output = DurableStreamIdentity(
        run_id=context.run_id,
        stream_id=durable_model_stream_id(context.run_id, context.turn_id),
        logical_call_id=logical_model_call_id(context.run_id, context.turn_id),
        channel="output",
    )
    reasoning = replace(output, channel="reasoning")
    for identity, data in ((output, b"stale output"), (reasoning, b"stale reasoning")):
        assert store.open(identity, writer_token=token).status == "opened"
        assert store.append(
            identity,
            generation=1,
            start_offset=0,
            data=data,
            writer_token=token,
        ).status == "committed"

    replacement = _observer(store, authority).open(context)
    assert isinstance(replacement, ModelStreamDispatchAwareWriter)
    replacement.begin_dispatch()
    replacement.close(ModelStreamOutcome(status="completed", final_text=final_text))

    for identity in (output, reasoning):
        assert store.heads[identity].generation == 2
        assert store.heads[identity].state == "sealed"
        assert store.heads[identity].cursor_bytes == 0
        assert store.chunks.get((identity, 2), []) == []

    recovered = _observer(store, authority).open(context)
    recovered.close(ModelStreamOutcome(status="completed", final_text=final_text))
    assert store.heads[output].generation == 2
    assert store.heads[reasoning].generation == 2


def test_model_stream_observer_preserves_tool_call_recovery_without_new_dispatch() -> None:
    store = _MemoryStreamStore()
    authority = ActivationWriteAuthority()
    context = _context()
    token = WriterToken(run_id=context.run_id, owner_id="worker-1", generation=1)
    output = DurableStreamIdentity(
        run_id=context.run_id,
        stream_id=durable_model_stream_id(context.run_id, context.turn_id),
        logical_call_id=logical_model_call_id(context.run_id, context.turn_id),
        channel="output",
    )
    reasoning = replace(output, channel="reasoning")
    assert store.open(output, writer_token=token).status == "opened"
    assert store.open(reasoning, writer_token=token).status == "opened"
    assert store.append(
        reasoning,
        generation=1,
        start_offset=0,
        data=b"recovered reasoning",
        writer_token=token,
    ).status == "committed"

    recovered = _observer(store, authority).open(context)
    recovered.close(ModelStreamOutcome(status="completed", final_text=None))

    assert store.heads[output].generation == 1
    assert store.heads[reasoning].generation == 1
    assert store.heads[output].state == "sealed"
    assert store.heads[reasoning].state == "sealed"
    assert b"".join(chunk.data for chunk in store.chunks[(reasoning, 1)]) == b"recovered reasoning"


def test_model_stream_observer_rebuilds_truncated_recovered_output_from_final_text() -> None:
    store = _MemoryStreamStore()
    authority = ActivationWriteAuthority()
    context = _context()
    token = WriterToken(run_id=context.run_id, owner_id="worker-1", generation=1)
    output = DurableStreamIdentity(
        run_id=context.run_id,
        stream_id=durable_model_stream_id(context.run_id, context.turn_id),
        logical_call_id=logical_model_call_id(context.run_id, context.turn_id),
        channel="output",
    )
    assert store.open(output, writer_token=token).status == "opened"
    assert store.append(
        output,
        generation=1,
        start_offset=0,
        data=b"truncated ",
        writer_token=token,
    ).status == "committed"

    recovered = _observer(store, authority, chunk_bytes=8).open(context)
    recovered.close(
        ModelStreamOutcome(
            status="completed",
            final_text="authoritative 세계 output",
        )
    )

    final_bytes = b"".join(chunk.data for chunk in store.chunks[(output, 2)])
    assert final_bytes.decode() == "authoritative 세계 output"
    assert all(chunk.chunk.size_bytes <= 8 for chunk in store.chunks[(output, 2)])
    assert store.heads[output].generation == 2
    assert store.heads[output].state == "sealed"
    assert store.heads[output].final_sha256 == hashlib.sha256(final_bytes).hexdigest()


def test_model_stream_observer_rehydrates_and_idempotently_recloses_a_sealed_lane() -> None:
    store = _MemoryStreamStore()
    authority = ActivationWriteAuthority()
    observer = _observer(
        store,
        authority,
        chunk_bytes=16,
        flush_interval_s=10,
        max_buffer_bytes=64,
    )
    first = observer.open(_context())
    first.push(ModelStreamDelta(channel="output", text="persisted"))
    first.close(ModelStreamOutcome(status="completed", final_text="persisted"))

    second = observer.open(_context())
    second.close(ModelStreamOutcome(status="completed", final_text="persisted"))

    identity = next(iter(store.heads))
    assert store.heads[identity].state == "sealed"
    assert store.heads[identity].final_sha256 == hashlib.sha256(b"persisted").hexdigest()


def test_model_stream_observer_lazily_resets_sealed_generation_on_new_delta() -> None:
    store = _MemoryStreamStore()
    authority = ActivationWriteAuthority()
    observer = _observer(store, authority, chunk_bytes=16, flush_interval_s=10)
    first = observer.open(_context())
    first.close(ModelStreamOutcome(status="failed"))

    replacement = observer.open(_context())
    replacement.push(ModelStreamDelta(channel="output", text="retried"))
    replacement.close(ModelStreamOutcome(status="completed", final_text="retried"))

    identity = next(iter(store.heads))
    head = store.heads[identity]
    assert head.generation == 2
    assert head.state == "sealed"
    assert b"".join(chunk.data for chunk in store.chunks[(identity, 2)]) == b"retried"
