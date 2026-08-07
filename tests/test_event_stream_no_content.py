"""The fan-out event stream carries no model-authored settled text.

``docs/OBSERVABILITY.md`` documents ``events.jsonl`` as the public/redacted artifact, and
``EventBus.emit`` fans every event out to every registered sink with no level filtering — so what
lands in this file is what a redacting sink, an OTel exporter and ``monoid watch --json`` all see.

The pair of claims has to be tested from both sides. "Model text is gone" and "kernel text is still
there" are separate properties, and an implementation that digests everything satisfies the first
while destroying the one line an operator reads to find out why a run stopped. The kernel-inline
branch is the half nothing else covers: the run-limit path never produced a settle event in any
existing test, so it could have regressed silently.
"""

from __future__ import annotations

import itertools

from monoid_agent_kernel.reference.studio import server as server_module

import json
import os
from pathlib import Path
from typing import Any

import pytest
from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.env import OUTPUT_DELTAS_ENV, getenv_bool

from monoid_agent_kernel.core.model_io import content_digest
from monoid_agent_kernel.core.spec import AgentRunSpec, RunLimits
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn, TextDelta, TurnComplete
from monoid_agent_kernel.providers.fake import (
    FakeModelAdapter,
    FakeStreamingModelAdapter,
    fake_tool_call,
)

SETTLE_TYPES = ("turn.settled", "run.finished")
MODEL_PROSE = "The answer is 42, and here is the reasoning behind it."


def _spec(tmp_path: Path, **limits: Any) -> AgentRunSpec:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        limits=RunLimits(**limits) if limits else RunLimits(),
    )


def _events(run_dir: Path, event_type: str) -> list[dict[str, Any]]:
    text = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [record for record in records if record["type"] == event_type]


def _run(spec: AgentRunSpec, adapter: Any, *tool_ids: str) -> Any:
    loop = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config(*(tool_ids or ("run.finish",)))),
    )
    return loop.run_once("go")


def test_model_authored_text_leaves_the_stream_as_a_digest(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text=MODEL_PROSE)])

    result = _run(_spec(tmp_path), adapter)

    for event_type in SETTLE_TYPES:
        data = _events(result.run_dir, event_type)[-1]["data"]
        assert "final_text" not in data, event_type
        assert data["final_text_digest"] == content_digest(MODEL_PROSE), event_type
        assert data["final_text_len"] == len(MODEL_PROSE), event_type


def test_the_prose_is_absent_from_the_whole_committed_file(tmp_path: Path) -> None:
    """Not just from the settle events — from every byte of ``events.jsonl``.

    Asserting per-event would miss a second route publishing the same string, which is exactly how
    this leak survived earlier audits: the settle events were checked and the delta channel, the
    approval preview and ``status.json`` were not.
    """
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text=MODEL_PROSE)])

    result = _run(_spec(tmp_path), adapter)

    assert MODEL_PROSE not in (result.run_dir / "events.jsonl").read_text(encoding="utf-8")
    # The private artifact still has it — the text moved, it was not destroyed.
    assert MODEL_PROSE in (result.run_dir / "transcript.jsonl").read_text(encoding="utf-8")


def test_the_caller_still_receives_the_text(tmp_path: Path) -> None:
    # The run result is the caller's answer, not a fan-out surface. A change that redacted it too
    # would pass every assertion above while breaking every embedder.
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text=MODEL_PROSE)])

    result = _run(_spec(tmp_path), adapter)

    assert result.final_text == MODEL_PROSE


def test_status_json_no_longer_carries_the_answer(tmp_path: Path) -> None:
    """``StatusJsonSink`` is a fan-out sink, so no hydration seam can reach it.

    Deleting the key is deliberate: keeping it would write ``""`` on every model-answered run with
    no schema failure, since ``STATUS_SCHEMA`` never declares it and allows additional properties.
    An empty string that used to be an answer is worse than an absent key.
    """
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text=MODEL_PROSE)])

    result = _run(_spec(tmp_path), adapter)

    status = json.loads((result.run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["terminal"] is True  # the run.finished branch really did fire
    assert "final_text" not in status
    assert MODEL_PROSE not in (result.run_dir / "status.json").read_text(encoding="utf-8")


def test_the_delta_channel_republishes_the_answer_unless_it_is_switched_off(
    tmp_path: Path, monkeypatch
) -> None:
    """The configuration every other test in this file avoids.

    Everything above runs with `emit_output_deltas` off, which is this dataclass's default and *not*
    the shipped product: Studio sets it from `find_spec("httpx") is not None`. So the file whose
    docstring says the leak survived because the delta channel went unchecked was itself asserting
    only in the one configuration where the channel does not exist. With deltas on, the answer
    reassembles byte-exactly out of `events.jsonl` even though it never appears verbatim in any
    single record — which is also why a grep-based audit finds nothing.

    `MONOID_OUTPUT_DELTAS=0` is the supported way to close it.
    """
    # Split across fragments none of which contains the whole sentence -- that is why grepping the
    # event log for the answer finds nothing while the answer is nonetheless present.
    fragments = ["The answer is 42, ", "and here is ", "the reasoning behind it."]
    assert "".join(fragments) == MODEL_PROSE

    def _streaming() -> FakeStreamingModelAdapter:
        return FakeStreamingModelAdapter(
            chunk_turns=[[*(TextDelta(part) for part in fragments), TurnComplete(response_id="r1")]]
        )

    def _loop(root: Path) -> AgentLoop:
        return AgentLoop(
            spec=_spec(root),
            model_adapter=_streaming(),
            runtime_config_provider=runtime_provider(runtime_config("run.finish")),
            emit_output_deltas=True,
        )

    on = _loop(tmp_path)
    assert on.emit_output_deltas is True
    on_result = on.run_once("go")

    published = (on_result.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert MODEL_PROSE not in published, "no single record holds it; that is the point"
    deltas = _events(on_result.run_dir, "model.output.delta")
    assert "".join(record["data"]["text"] for record in deltas) == MODEL_PROSE

    # Now the switch, same configuration otherwise.
    monkeypatch.setenv("MONOID_OUTPUT_DELTAS", "0")
    (tmp_path / "off").mkdir()
    off = _loop(tmp_path / "off")
    # Resolved into the field, not at the emit site: a loop reporting `True` while streaming nothing
    # is the same half-bound shape the switch exists to remove.
    assert off.emit_output_deltas is False
    off_result = off.run_once("go")

    assert _events(off_result.run_dir, "model.output.delta") == []
    assert off_result.final_text == MODEL_PROSE  # the caller's answer is untouched


def test_an_unparseable_switch_value_is_an_error_rather_than_a_silent_default() -> None:
    """A kill switch that reads a typo as "leave it on" is worse than no switch.

    The one existing env-boolean precedent is `getenv(...) != "1"`, which is fail-closed — correct
    for a permission, wrong here, because every value except exactly `"1"` means off. An operator
    who set the variable at all meant to change something.
    """
    previous = os.environ.get(OUTPUT_DELTAS_ENV)
    os.environ[OUTPUT_DELTAS_ENV] = "of"  # a plausible typo for "off"
    try:
        with pytest.raises(ValueError, match="not a boolean"):
            getenv_bool(OUTPUT_DELTAS_ENV, default=True)
        for value, expected in (
            ("0", False),
            ("off", False),
            ("no", False),
            ("1", True),
            ("on", True),
        ):
            os.environ[OUTPUT_DELTAS_ENV] = value
            assert getenv_bool(OUTPUT_DELTAS_ENV, default=True) is expected
        # An *empty* value is the one string that must not raise. `MONOID_FOO=` is how a dotenv file
        # blanks a key, and `load_env_file` copies that empty value into `os.environ`, so treating it
        # as unparseable would fail every run of anyone who commented out a line. Asserted because
        # the rule lives in one `if not normalized` that reads like a redundant guard next to the
        # `raw is None` check above it — the shape a later simplification deletes.
        for blank in ("", "   "):
            os.environ[OUTPUT_DELTAS_ENV] = blank
            assert getenv_bool(OUTPUT_DELTAS_ENV, default=True) is True
            assert getenv_bool(OUTPUT_DELTAS_ENV, default=False) is False
    finally:
        if previous is None:
            os.environ.pop(OUTPUT_DELTAS_ENV, None)
        else:
            os.environ[OUTPUT_DELTAS_ENV] = previous


def test_studio_wires_content_permission_without_disabling_provider_streaming(
    tmp_path: Path, monkeypatch
) -> None:
    """Studio keeps streaming/Stop responsive while gating both content egress surfaces."""
    from monoid_agent_kernel.reference.studio.cli import _studio_config
    from monoid_agent_kernel.reference.studio.server import StudioConfig, StudioServer

    # 1. The CLI flag reaches the config.
    common = {
        "workspace": tmp_path,
        "host": "127.0.0.1",
        "port": 0,
        "provider": "offline",
        "run_root": tmp_path / "runs",
        "skills_directory": tmp_path,
        "no_skills": True,
        "mcp": False,
        "env_file": tmp_path / ".env",
        "no_env_file": True,
    }
    assert _studio_config(**common, no_output_deltas=True).stream_output_deltas is False
    assert _studio_config(**common).stream_output_deltas is True

    # 2. The config reaches the backend the runs are built from.
    # Distinct dirs per *call*, not per value: this helper is now invoked more than once with the
    # same `stream`, and keying the workspace on the value alone collided on the second.
    calls = itertools.count()

    # The tuple carries EVERY backend switch that can put model content on disk or on a wire, not
    # the ones that existed when it was written: `model_payload_file` is the second
    # content-classified artifact on this dataclass, and a census that enumerates one of two is
    # how a rule stops reaching the half nobody added. Studio sets neither of the two recording
    # switches, so both tail entries are False in every row below -- which is the claim: turning
    # Studio's egress toggle on grants live delivery and the content sidecar, never the corpus.
    def backend_stream_state(*, egress: bool) -> tuple[bool, bool, bool, bool, bool, bool]:
        workspace = tmp_path / f"ws-{egress}-{next(calls)}"
        workspace.mkdir()
        server = StudioServer(
            StudioConfig(
                workspace=workspace,
                port=0,
                run_root=tmp_path / f"runs-{workspace.name}",
                stream_output_deltas=egress,
            )
        )
        try:
            server.start()
            assert server._backend is not None
            return (
                server._backend.emit_output_deltas,
                server._backend.stream_model_calls,
                server._backend.model_content_file,
                server._backend.model_stream_broker is not None,
                server._backend.model_calls_file,
                server._backend.model_payload_file,
            )
        finally:
            server.shutdown()

    monkeypatch.delenv("MONOID_OUTPUT_DELTAS", raising=False)
    monkeypatch.setattr(server_module, "_gateway_streaming_available", lambda: True)
    assert backend_stream_state(egress=False) == (False, True, False, False, False, False)
    assert backend_stream_state(egress=True) == (False, True, True, True, False, False)

    monkeypatch.setenv("MONOID_OUTPUT_DELTAS", "0")
    assert backend_stream_state(egress=True) == (False, True, False, False, False, False)

    monkeypatch.delenv("MONOID_OUTPUT_DELTAS", raising=False)
    monkeypatch.setattr(server_module, "_gateway_streaming_available", lambda: False)
    assert backend_stream_state(egress=True) == (False, False, False, False, False, False)


def test_kernel_authored_limit_text_stays_inline(tmp_path: Path) -> None:
    """The selectivity half, on the path that produces it.

    "Stopped after reaching max steps." is not model output. Digesting it buys no privacy and costs
    an operator the one line explaining why the run stopped — and it would need a transcript join to
    read a string the kernel wrote. No other test drives a settle event out of the limit path, so
    without this the provenance flag could be stuck ``True`` and only a human would notice.
    """
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id=f"r{index}", tool_calls=[fake_tool_call("run.finish", {})])
            for index in range(4)
        ]
    )

    result = _run(_spec(tmp_path, max_steps=1), adapter)

    for event_type in SETTLE_TYPES:
        data = _events(result.run_dir, event_type)[-1]["data"]
        assert data["status"] == "limited", event_type
        assert data["final_text"] == "Stopped after reaching max steps.", event_type
        assert "final_text_digest" not in data, event_type
        assert "final_text_len" not in data, event_type
    # And nothing was written to the private channel for it — a kernel string is not model output,
    # so it has no business in the settled-text record either.
    transcript = (result.run_dir / "transcript.jsonl").read_text(encoding="utf-8")
    kinds = [json.loads(line)["kind"] for line in transcript.splitlines() if line.strip()]
    assert "settled_text" not in kinds
