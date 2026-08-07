"""The private model-call ledger as a run actually writes it.

W6-1. ``tests/test_model_calls_schema.py`` pins what a record may say; this pins that the run
writes one per settled call, into the right directory, without ever being able to fail the run.

The property worth stating up front, because it is the reason the sink exists at all: a failed
model call publishes its receipt and re-raises without stamping it on the exception, so the loop's
own ``turn, _receipt = await runner.acall(...)`` never sees it. A ledger wired to that return
value would record only the calls that succeeded.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from support.runtime import runtime_config, runtime_provider, tool_binding

from monoid_agent_kernel.core.agents import AgentRuntimeConfig, PromptSpec, SubagentDefinition
from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.model_calls import MODEL_CALLS_FILENAME
from monoid_agent_kernel.core.schemas import validate_run_dir
from monoid_agent_kernel.core.spec import AgentRunSpec, ModelConfig
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.model_call import SettledModelCall
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn


class _Adapter:
    supports_multimodal = False
    provider_name = "test-provider"

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        del request
        return ModelTurn(response_id="r1", final_text="answer")


class _FailingAdapter:
    supports_multimodal = False
    provider_name = "test-provider"

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        del request
        raise ModelAdapterError(
            "upstream refused",
            provider_error_code="rate_limit",
            retryable=True,
            http_status=429,
        )


def _loop(
    tmp_path: Path,
    adapter: object,
    *,
    model_calls_file: bool = True,
    invocation_context: InvocationContext | None = None,
    config: AgentRuntimeConfig | None = None,
) -> AgentLoop:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,  # type: ignore[arg-type]
        runtime_config_provider=runtime_provider(
            config if config is not None else runtime_config("run.finish")
        ),
        model_calls_file=model_calls_file,
        invocation_context=invocation_context,
    )


def _records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / MODEL_CALLS_FILENAME
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_a_run_writes_no_model_call_sidecar_unless_it_is_asked_to(tmp_path: Path) -> None:
    """Opt-in, like the content sidecar beside it.

    The run-directory shape is something embedders build retention and shipping rules around, so
    a new artifact appearing in every run because a kernel version changed is a change they did
    not make. A run that wants the ledger asks for it.
    """
    result = _loop(tmp_path, _Adapter(), model_calls_file=False).run_once("hi")

    assert not (result.run_dir / MODEL_CALLS_FILENAME).exists()


def test_every_settled_call_including_a_failed_one_gets_exactly_one_line(tmp_path: Path) -> None:
    """The seam's whole reason: the failure arm is the one a loop-side recorder cannot reach."""
    ok = _loop(tmp_path / "ok", _Adapter()).run_once("hi")
    ok_records = _records(ok.run_dir)

    assert len(ok_records) == 1
    assert ok_records[0]["error_code"] == ""
    assert ok_records[0]["request_digest"]
    assert ok_records[0]["digest_status"] == "ok"
    assert ok_records[0]["provider_name"] == "test-provider"

    failed = _loop(tmp_path / "bad", _FailingAdapter()).run_once("hi")
    failed_records = _records(failed.run_dir)

    assert len(failed_records) == 1
    record = failed_records[0]
    assert record["error_code"] != ""
    assert record["provider_error_code"] == "rate_limit"
    assert record["http_status"] == 429
    assert record["retryable"] is True
    # Identifiable despite failing: the key is taken before dispatch, which is what lets a failed
    # call sit in the same ledger as the successful ones rather than in a lane of its own.
    assert record["request_digest"]


def test_the_ledger_a_run_writes_validates_against_its_own_schema(tmp_path: Path) -> None:
    """``monoid validate`` is the ledger's only reader in this release, so it has to agree."""
    result = _loop(tmp_path, _Adapter()).run_once("hi")

    assert (result.run_dir / MODEL_CALLS_FILENAME).exists()
    assert [issue for issue in validate_run_dir(result.run_dir) if "model_calls" in issue.path] == []


def test_a_written_line_never_carries_the_configured_endpoint(tmp_path: Path) -> None:
    """The schema-level exclusion, asserted again against bytes a run actually produced.

    The projection test proves the record omits it; this proves nothing downstream puts it back --
    the writer normalizes and serializes, and either step could have re-serialized the receipt.
    """
    endpoint = "https://gateway.internal.example/tenant-a/llm/turns"
    result = _loop(
        tmp_path,
        _Adapter(),
        config=runtime_config(
            "run.finish",
            model=ModelConfig(provider="gateway", gateway_url=endpoint),
        ),
    ).run_once("hi")

    raw = (result.run_dir / MODEL_CALLS_FILENAME).read_text(encoding="utf-8")

    assert raw.strip()
    # Non-vacuity first. If the configured model never reached the receipt, every assertion below
    # would pass for the wrong reason -- an absence proves nothing about a projection that was
    # handed nothing. `provider` is the witness: it comes from the same config block the endpoint
    # does, and the ledger records it.
    assert _records(result.run_dir)[0]["model"]["provider"] == "gateway"

    assert endpoint not in raw
    assert "gateway.internal.example" not in raw
    assert "gateway_url" not in raw
    assert "destination_digest" not in raw


def test_the_recorded_run_id_is_the_recorder_s_and_not_a_caller_s_claim(tmp_path: Path) -> None:
    """The envelope's ``run_id`` is proven by the directory the line is written into.

    ``InvocationContext.run_id`` is caller-supplied and the loop overwrites it with its own before
    the call -- but the record must not depend on that, because the receipt's context is a claim
    and the recorder's identity is a fact. Both are kept: the claim stays inside ``context``.
    """
    result = _loop(
        tmp_path,
        _Adapter(),
        invocation_context=InvocationContext(run_id="not-this-run", skill_id="sk"),
    ).run_once("hi")

    record = _records(result.run_dir)[0]

    assert record["run_id"] == result.run_id
    assert record["root_run_id"] == result.run_id
    assert record["context"]["run_id"] == result.run_id
    assert record["context"]["skill_id"] == "sk"


def test_a_child_loop_records_into_its_own_run_directory(tmp_path: Path) -> None:
    """A subagent inherits the switch, and each recorder owns one directory.

    So the coverage is free but the *reading* is not: a child's calls are in the child's ledger,
    which is why every line carries ``root_run_id``. That field is the only thing that makes the
    tree joinable without a reader walking the parent's events first.
    """
    parent_marker = "[[parent-ledger-test]]"
    child_marker = "[[child-ledger-test]]"

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
                prompt=PromptSpec(persona_segments=(parent_marker,)),
                tools=(tool_binding("agent.spawn"),),
            )
        ),
        subagent_definitions={
            "child": SubagentDefinition(prompt=PromptSpec(persona_segments=(child_marker,)))
        },
        model_calls_file=True,
    )

    result = loop.run_once("delegate")

    parent_records = _records(result.run_dir)
    assert len(parent_records) == 2
    assert {record["run_id"] for record in parent_records} == {result.run_id}

    child_dirs = [
        path
        for path in (tmp_path / "runs").iterdir()
        if path.is_dir() and path.name != result.run_id
    ]
    assert len(child_dirs) == 1
    child_records = _records(child_dirs[0])
    assert len(child_records) == 1
    assert child_records[0]["run_id"] != result.run_id
    assert child_records[0]["root_run_id"] == result.run_id
    # Every line in the tree agrees on the root, which is what makes the tree joinable at all.
    assert {record["root_run_id"] for record in parent_records + child_records} == {result.run_id}


def test_call_index_counts_within_one_activation(tmp_path: Path) -> None:
    """The only way an append-only best-effort file can reveal its own dropped lines."""

    class TwoTurnAdapter:
        supports_multimodal = False
        provider_name = "test-provider"
        calls = 0

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            self.calls += 1
            if self.calls == 1:
                from monoid_agent_kernel.providers.base import ToolCall

                return ModelTurn(
                    response_id="t1",
                    tool_calls=(ToolCall(id="c1", name="fs_list", arguments={"path": "."}),),
                )
            return ModelTurn(response_id="t2", final_text="done")

    result = _loop(
        tmp_path,
        TwoTurnAdapter(),
        config=runtime_config("run.finish", "fs.list"),
    ).run_once("hi")

    assert [record["call_index"] for record in _records(result.run_dir)] == [0, 1]


def test_a_sidecar_write_failure_disables_the_handle_without_failing_the_run(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A recorder that cannot write is not a reason to discard an answer already paid for.

    The handle is disabled rather than retried, for the reason the content sidecar gives: a
    partial write may have torn the current line, and appending after it would glue the next
    record onto the remnant and lose both.
    """
    from monoid_agent_kernel import recorder as recorder_module

    def explode(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise OSError("disk is full")

    monkeypatch.setattr(recorder_module.AgentRecorder, "_append_model_call", explode)

    result = _loop(tmp_path, _Adapter()).run_once("hi")

    assert result.final_text == "answer"
    assert result.status == "completed"


def test_a_hostile_context_costs_its_own_line_and_not_the_run(tmp_path: Path) -> None:
    """Encoding failure is contained to the record, the rule the content sidecar's append holds."""
    result = _loop(
        tmp_path,
        _Adapter(),
        invocation_context=InvocationContext(attributes={"lone-surrogate": "\ud800"}),
    ).run_once("hi")

    assert result.final_text == "answer"
    # Written or skipped, the file must never be left holding something its own schema refuses.
    assert [issue for issue in validate_run_dir(result.run_dir) if "model_calls" in issue.path] == []


def test_a_reopened_run_appends_to_the_ledger_it_already_has(tmp_path: Path) -> None:
    """One file spans activations, so `call_index` restarts inside it.

    That is deliberate and is why the field is a gap detector rather than a join key: a restart is
    self-evident (the index drops while ``recorded_at`` advances), unlike a per-process digest,
    which would name one destination two ways and read as a change that never happened.
    """
    first = _loop(tmp_path, _Adapter())
    first_result = first.run_once("hi")
    run_dir = first_result.run_dir
    assert len(_records(run_dir)) == 1

    reopened = AgentLoop(
        spec=AgentRunSpec(
            workspace_root=tmp_path / "workspace",
            run_root=tmp_path / "runs",
            run_id=first_result.run_id,
        ),
        model_adapter=_Adapter(),  # type: ignore[arg-type]
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
        model_calls_file=True,
    )
    reopened._restoring = True
    second_result = reopened.run_once("again")

    assert second_result.run_id == first_result.run_id
    records = _records(run_dir)
    assert len(records) == 2
    assert [record["call_index"] for record in records] == [0, 0]
    assert records[0]["recorded_at"] <= records[1]["recorded_at"]


def _standalone_recorder(tmp_path: Path) -> Any:
    from monoid_agent_kernel.recorder import AgentRecorder

    return AgentRecorder(tmp_path / "runs", "run-1", model_calls_file=True, status_file=False)


def test_concurrent_records_never_share_a_call_index(tmp_path: Path) -> None:
    """Reserving the index and writing the line are one operation, not two.

    `call_index` has exactly one job -- letting a reader notice that a best-effort append-only file
    dropped something -- and two records sharing an index defeats it silently, in the direction
    that reads as "nothing was lost". The recorder takes a lock precisely because it is shared with
    tool and job threads, so "the loop calls this sequentially" is not a guarantee this method may
    rely on: it is a property of one caller.
    """
    import threading

    from monoid_agent_kernel.core.model_io import ModelCallReceipt

    recorder = _standalone_recorder(tmp_path)
    start = threading.Barrier(8)

    def write() -> None:
        start.wait()
        for _ in range(20):
            recorder.record_settled_call(SettledModelCall(receipt=ModelCallReceipt()))

    threads = [threading.Thread(target=write) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    recorder.close()

    indices = [record["call_index"] for record in _records(recorder.run_dir)]

    assert len(indices) == 160
    assert sorted(indices) == list(range(160))


def test_the_recorder_contains_its_own_write_failure(tmp_path: Path, monkeypatch: Any) -> None:
    """The promise is the recorder's, not the runner's.

    ``record_settled_call`` is public and its docstring says nothing here raises. Proving that only
    through a run leans on ``ModelCallRunner._record``'s own guard -- so the promise would survive
    a refactor that removed it from the recorder, and only fail for whoever calls the recorder
    directly.
    """
    from monoid_agent_kernel.core.model_io import ModelCallReceipt
    from monoid_agent_kernel import recorder as recorder_module

    def explode(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise OSError("disk is full")

    monkeypatch.setattr(recorder_module.AgentRecorder, "_append_model_call", explode)
    recorder = _standalone_recorder(tmp_path)

    recorder.record_settled_call(SettledModelCall(receipt=ModelCallReceipt()))  # must not raise

    recorder.close()


def test_a_record_arriving_after_close_does_not_reopen_the_ledger(tmp_path: Path) -> None:
    """Closing the recorder ends the ledger, rather than leaving a handle to be re-acquired.

    The handle is opened lazily, so without an explicit closed state a late record silently
    reopens the file the recorder just released -- leaking the descriptor past the lifetime that
    owns it, and on Windows holding a lock on a run directory a caller is entitled to move or
    delete. `ModelContentStore` refuses post-close work for the same reason; this is that rule for
    the sibling artifact. Refused rather than failed: nothing here raises either.
    """
    from monoid_agent_kernel.core.model_io import ModelCallReceipt

    recorder = _standalone_recorder(tmp_path)
    recorder.record_settled_call(SettledModelCall(receipt=ModelCallReceipt()))
    recorder.close()

    recorder.record_settled_call(SettledModelCall(receipt=ModelCallReceipt()))  # must not raise, and must not write

    assert len(_records(recorder.run_dir)) == 1
    assert recorder._model_calls_handle is None


def _plant_hardlink(link: Path, target: Path) -> None:
    import os

    try:
        os.link(target, link)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"hard links are unavailable: {exc}")


def _plant_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")


@pytest.mark.parametrize("plant", [_plant_hardlink, _plant_symlink], ids=["hardlink", "symlink"])
def test_a_planted_link_stops_the_ledger_instead_of_being_written_through(
    tmp_path: Path,
    plant: Any,
) -> None:
    """The ledger writes to a file this process made, or it writes nothing at all.

    The handle is opened lazily, so between the run directory existing and the first receipt
    arriving there is a window in which the ledger's pathname can be replaced -- and a reopened
    durable run widens that window to "any time since the last activation". A plain ``open(path,
    "a")`` follows a symlink, and a hard link is a second name for an inode anywhere on the volume,
    so either one makes the agent append its own JSONL, with its own credentials, to a file
    somebody else chose. The sibling ``model-content.jsonl`` verified this before appending; this
    is that same rule, from the same function, for the second sidecar in the same directory.

    Fail **closed**, not "skip this line": the reason a verified open refused is a property of the
    path, so a retry on the next receipt only re-runs the same refusal. Setting the disable flag is
    what makes the refusal terminal, exactly as a torn write does.

    The witness at the end is load-bearing. "The outside file did not change" is a claim that passes
    for free if the recorder was handed a receipt it could not have written anyway, so a second
    recorder with an untouched directory records the *same* receipt and is asserted to produce a
    line.
    """
    from monoid_agent_kernel.core.model_io import ModelCallReceipt
    from monoid_agent_kernel.recorder import AgentRecorder

    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "someone-elses.jsonl"
    target.write_bytes(b'{"kind":"not the agent\'s"}\n')
    original = target.read_bytes()

    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    plant(run_dir / MODEL_CALLS_FILENAME, target)

    receipt = ModelCallReceipt()
    recorder = AgentRecorder(tmp_path / "runs", "run-1", model_calls_file=True, status_file=False)
    recorder.record_settled_call(SettledModelCall(receipt=receipt))

    assert target.read_bytes() == original
    assert recorder._model_calls_failed is True
    assert recorder._model_calls_handle is None

    # Terminal, not per-line: a second receipt must not re-attempt the same refused open.
    recorder.record_settled_call(SettledModelCall(receipt=receipt))
    assert target.read_bytes() == original
    assert recorder._model_calls_index == 0
    recorder.close()

    witness = AgentRecorder(tmp_path / "clean", "run-1", model_calls_file=True, status_file=False)
    witness.record_settled_call(SettledModelCall(receipt=receipt))
    witness.close()
    assert len(_records(witness.run_dir)) == 1


@pytest.mark.parametrize(
    ("filename", "switch"),
    [
        (MODEL_CALLS_FILENAME, "model_calls_file"),
        ("model_payloads.jsonl", "model_payload_file"),
        ("model-content.jsonl", "model_content_file"),
    ],
    ids=["ledger", "corpus", "content"],
)
def test_a_refused_sidecar_says_so_at_warning_level(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, filename: str, switch: str
) -> None:
    """A sidecar nobody asked for is absent; a sidecar somebody asked for and did not get is a
    fault, and the operator who asked has to be able to learn it.

    Every one of these three writers fails closed on a refused verified open, which is right, and
    then said so only at ``debug`` -- so the shape the refusal exists to catch (a link planted where
    a reopened run expects its artifact, which is also what a hardlink-deduplicating backup leaves
    behind) produced a run that exits zero, reports ``completed``, and writes nothing where the
    operator asked for a record. `monoid validate` then reports the directory clean, because each
    artifact is optional. Nothing anywhere said no.

    Parametrized across all three because they are three copies of one rule, and this repository's
    recurring defect is a rule bound on one of parallel halves. ``WARNING`` specifically: below it,
    Python's last-resort handler drops the message, so a CLI operator who configured no logging --
    the shape that runs `monoid run` -- would still see nothing.
    """
    from monoid_agent_kernel.core.model_io import ModelCallReceipt
    from monoid_agent_kernel.core.model_stream import ModelStreamContext, ModelStreamDelta
    from monoid_agent_kernel.recorder import AgentRecorder

    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "someone-elses.jsonl"
    target.write_bytes(b"{}\n")

    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    _plant_hardlink(run_dir / filename, target)

    recorder = AgentRecorder(tmp_path / "runs", "run-1", status_file=False, **{switch: True})
    with caplog.at_level(logging.WARNING):
        recorder.record_settled_call(
            # A turn, so the corpus arm has an answer to record and actually reaches for its
            # handle: an empty receipt gives the corpus nothing to write, and a refusal nobody
            # reached is not the refusal under test.
            SettledModelCall(receipt=ModelCallReceipt(), turn=ModelTurn(final_text="answer"))
        )
        writer = recorder.open_model_stream(
            ModelStreamContext(
                run_id="run-1",
                root_run_id="run-1",
                turn_id="t1",
                stream_id="s1",
                step=0,
                provider="fake",
                model="m",
                started_at="2026-01-01T00:00:00Z",
            )
        )
        writer.push(ModelStreamDelta("output", "answer"))
    recorder.close()

    warnings = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert warnings, f"the refused {filename} never reached warning level"
    assert any(filename in record.getMessage() for record in warnings), (
        f"the warning must name the artifact that was refused: "
        f"{[record.getMessage() for record in warnings]}"
    )
    # The witness that this is a refusal and not merely a quiet run: nothing was written.
    assert target.read_bytes() == b"{}\n"


def test_a_reopened_ledger_isolates_the_tail_the_crashed_activation_tore(tmp_path: Path) -> None:
    """A record torn by a crash costs its own line, never the next activation's first one.

    Appending after a line with no trailing newline glues the remnant and the new record into one
    unparseable line and loses **both**. The check now runs on the descriptor the verified open just
    validated rather than by reopening the pathname, because a second ``open`` of the same name is a
    second chance to be handed a different file; this pins that the property survived that move.
    """
    from monoid_agent_kernel.core.model_io import ModelCallReceipt
    from monoid_agent_kernel.recorder import AgentRecorder

    recorder = _standalone_recorder(tmp_path)
    recorder.record_settled_call(SettledModelCall(receipt=ModelCallReceipt()))
    recorder.close()
    ledger = recorder.run_dir / MODEL_CALLS_FILENAME
    with ledger.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write('{"torn":')

    reopened = AgentRecorder(
        tmp_path / "runs",
        "run-1",
        model_calls_file=True,
        status_file=False,
        reopen=True,
    )
    reopened.record_settled_call(SettledModelCall(receipt=ModelCallReceipt()))
    reopened.close()

    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert lines[1] == '{"torn":'
    assert json.loads(lines[2])["call_index"] == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_the_ledger_switch_does_not_select_provider_streaming(
    tmp_path: Path,
    enabled: bool,
) -> None:
    """`model_content_file` drives `astream_turn` selection; this one must not join it.

    A receipt ledger needs no provider streaming, and coupling them would silently change
    cancellation and interrupt granularity for every run that wanted only receipts.
    """

    class StreamProbe:
        supports_multimodal = False
        provider_name = "test-provider"
        streamed = False

        async def astream_turn(self, request: ModelRequest):  # noqa: ANN202
            del request
            self.streamed = True
            from monoid_agent_kernel.providers.base import TextDelta, TurnComplete

            yield TextDelta("answer")
            yield TurnComplete(response_id="r1")

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(response_id="r1", final_text="answer")

    adapter = StreamProbe()
    _loop(tmp_path, adapter, model_calls_file=enabled).run_once("hi")

    assert adapter.streamed is False
