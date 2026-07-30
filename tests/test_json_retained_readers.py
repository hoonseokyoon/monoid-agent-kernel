from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from monoid_agent_kernel.core._event_log import iter_committed_event_records
from monoid_agent_kernel.core.packages import _read_json
from monoid_agent_kernel.reference.backend.proposal_reader import read_proposal_snapshot
from monoid_agent_kernel.reference.studio import cli as studio_cli
from monoid_agent_kernel.reference.studio.chat_projection import ChatProjection
from monoid_agent_kernel.reference.studio.server import StudioConfig, StudioServer


@dataclass
class _ProposalRecord:
    run_dir: Path


def test_committed_event_reader_repairs_escaped_surrogates(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"seq":1,"data":{"text":"bad\\ud800text"}}\n')

    records = list(iter_committed_event_records(path))

    assert records[0].payload["data"]["text"] == "bad\ufffdtext"


def test_proposal_and_package_readers_repair_surrogates_and_reject_nonfinite(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    proposal_path = run_dir / "proposal.json"
    proposal_path.write_text('{"title":"bad\\ud800title"}', encoding="utf-8")

    assert read_proposal_snapshot(_ProposalRecord(run_dir)) == {
        "title": "bad\ufffdtitle"
    }
    assert _read_json(proposal_path) == {"title": "bad\ufffdtitle"}
    assert _read_json({"score": float("nan"), "title": "bad\ud800title"}) == {
        "score": None,
        "title": "bad\ufffdtitle",
    }

    proposal_path.write_text('{"score":NaN}', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError, match="non-finite number"):
        read_proposal_snapshot(_ProposalRecord(run_dir))
    with pytest.raises(json.JSONDecodeError, match="non-finite number"):
        _read_json(proposal_path)


def test_studio_chat_and_run_sidecar_readers_keep_their_tolerant_contract(
    tmp_path: Path,
) -> None:
    chat = ChatProjection(tmp_path / "chat-run")
    chat.run_dir.mkdir()
    chat.path.write_text(
        '{"content":"bad\\ud800chat"}\n'
        '{"content":"ignored","score":Infinity}\n',
        encoding="utf-8",
    )
    assert chat.read() == [{"content": "bad\ufffdchat"}]

    legacy = ChatProjection(tmp_path / "legacy-run")
    legacy.run_dir.mkdir()
    (legacy.run_dir / "run.json").write_text(
        '{"title":"bad\\ud800title","created_at":1.0}',
        encoding="utf-8",
    )
    legacy.ensure_legacy_user_from_run_meta()
    assert legacy.read()[0]["content"] == "bad\ufffdtitle"

    corrupt = ChatProjection(tmp_path / "corrupt-run")
    corrupt.run_dir.mkdir()
    (corrupt.run_dir / "run.json").write_text(
        '{"title":"ignored","created_at":NaN}',
        encoding="utf-8",
    )
    corrupt.ensure_legacy_user_from_run_meta()
    assert corrupt.read() == []


def test_studio_profile_sidecar_repairs_surrogates_and_falls_back_on_nonfinite(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs"
    run_root.mkdir()
    studio = StudioServer(
        StudioConfig(
            workspace=tmp_path / "workspace",
            run_root=run_root,
            skills_directory=None,
            memory=False,
        )
    )
    profile_path = studio._profile_store_path()
    profile_path.write_text(
        '{"profiles":{"agent":{"name":"bad\\ud800name"}},"runs":{"run_1":"agent"}}',
        encoding="utf-8",
    )
    assert studio._load_profile_store() == {
        "profiles": {"agent": {"name": "bad\ufffdname"}},
        "runs": {"run_1": "agent"},
    }

    profile_path.write_text(
        '{"profiles":{"agent":{"score":NaN}},"runs":{}}',
        encoding="utf-8",
    )
    assert studio._load_profile_store() == {"profiles": {}, "runs": {}}


class _HttpResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> _HttpResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_studio_cli_response_reader_is_strict_and_repairs_surrogates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _HttpResponse(b'{"title":"bad\\ud800title"}')
    monkeypatch.setattr(studio_cli.urlrequest, "urlopen", lambda *args, **kwargs: response)
    assert studio_cli._http_json("https://studio.example.test") == {
        "title": "bad\ufffdtitle"
    }

    response.body = b'{"score":Infinity}'
    with pytest.raises(json.JSONDecodeError, match="non-finite number"):
        studio_cli._http_json("https://studio.example.test")

    response.body = b"[]"
    with pytest.raises(ValueError, match="must be a JSON object"):
        studio_cli._http_json("https://studio.example.test")
