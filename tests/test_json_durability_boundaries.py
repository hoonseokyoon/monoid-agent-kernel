from __future__ import annotations

import io
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from monoid_agent_kernel.core._event_log import EventLogCorruption
from monoid_agent_kernel.core._util import write_json_atomic
from monoid_agent_kernel.core.checkpoint import (
    CHECKPOINT_FILENAME,
    LocalFsCheckpointStore,
    RunCheckpoint,
    read_checkpoint_checked,
)
from monoid_agent_kernel.core.control import ControlResult
from monoid_agent_kernel.core.durable_metadata import (
    RUN_METADATA_FILENAME,
    RUN_METADATA_SCHEMA_VERSION,
    read_run_metadata_checked,
)
from monoid_agent_kernel.core.projections import _read_json_if_exists
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.providers.base import ModelRequest
from monoid_agent_kernel.providers.gateway import GatewayModelAdapter
from monoid_agent_kernel.recorder import _write_jsonl, append_event_to_run
from monoid_agent_kernel.reference._shared.http_util import read_json_limited
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.backend.projection import _json_safe, _read_optional_json
from monoid_agent_kernel.reference.backend.recovery import RecoveryService
from monoid_agent_kernel.reference.command_inbox import (
    CommandPrincipal,
    SqliteCommandStore,
    StoredCommand,
)
from monoid_agent_kernel.reference.stores.lease import LocalFsLeaseStore
from monoid_agent_kernel.reference.stores.sqlite import SqliteCheckpointStore
from monoid_agent_kernel.reference.web_gateway import providers as web_providers
from monoid_agent_kernel.reference.web_gateway.service import WebGatewayBackend
from monoid_agent_kernel.tasks import _read_job_artifact, run_permission_policy
from monoid_agent_kernel.web import WebGatewayError


def test_http_json_reader_rejects_duplicate_control_keys() -> None:
    body = b'{"approved":false,"approved":true}'
    handler = type(
        "_Handler",
        (),
        {
            "headers": {"Content-Length": str(len(body))},
            "rfile": io.BytesIO(body),
            "close_connection": False,
        },
    )()

    with pytest.raises(ValueError, match="invalid JSON request body"):
        read_json_limited(handler)


def _nonfinite_checkpoint_text() -> str:
    checkpoint = RunCheckpoint(
        run_id="run_1",
        seq=1,
        messages=[{"role": "assistant", "score": float("nan")}],
    )
    return json.dumps(checkpoint.to_json())


def _nonfinite_metadata_text() -> str:
    return json.dumps(
        {
            "schema_version": RUN_METADATA_SCHEMA_VERSION,
            "run_id": "run_1",
            "extension": float("inf"),
        }
    )


def _legacy_surrogate_checkpoint_text() -> str:
    checkpoint = RunCheckpoint(
        run_id="run_1",
        seq=1,
        messages=[{"role": "assistant", "text": "bad\ud800text"}],
    )
    return json.dumps(checkpoint.to_json())


def _legacy_surrogate_metadata_text() -> str:
    return json.dumps(
        {
            "schema_version": RUN_METADATA_SCHEMA_VERSION,
            "run_id": "run_1",
            "extension": "bad\ud800text",
        }
    )


def test_single_file_durable_readers_reject_nonfinite_constants(tmp_path: Path) -> None:
    (tmp_path / CHECKPOINT_FILENAME).write_text(_nonfinite_checkpoint_text(), encoding="utf-8")
    assert read_checkpoint_checked(tmp_path).status == "corrupt"

    (tmp_path / RUN_METADATA_FILENAME).write_text(_nonfinite_metadata_text(), encoding="utf-8")
    assert read_run_metadata_checked(tmp_path).status == "corrupt"

    (tmp_path / CHECKPOINT_FILENAME).write_text(
        _legacy_surrogate_checkpoint_text(), encoding="utf-8"
    )
    checkpoint = read_checkpoint_checked(tmp_path)
    assert checkpoint.status == "loaded"
    assert checkpoint.value is not None
    assert checkpoint.value.messages[0]["text"] == "bad\ufffdtext"

    (tmp_path / RUN_METADATA_FILENAME).write_text(
        _legacy_surrogate_metadata_text(), encoding="utf-8"
    )
    metadata = read_run_metadata_checked(tmp_path)
    assert metadata.status == "loaded"
    assert metadata.value is not None
    assert metadata.value["extension"] == "bad\ufffdtext"


@pytest.mark.parametrize("backend", ["local_fs", "sqlite"])
def test_checkpoint_stores_reject_forged_nonfinite_json(tmp_path: Path, backend: str) -> None:
    if backend == "local_fs":
        store: Any = LocalFsCheckpointStore(tmp_path / "local")
    else:
        store = SqliteCheckpointStore(tmp_path / "checkpoints.db")
    store.put(RunCheckpoint(run_id="run_1", seq=1))
    store.put_run_metadata(
        "run_1",
        {"schema_version": RUN_METADATA_SCHEMA_VERSION, "run_id": "run_1"},
    )

    if backend == "local_fs":
        durable_dir = tmp_path / "local" / "run_1" / "checkpoints"
        (durable_dir / "1" / "manifest.json").write_text(
            _nonfinite_checkpoint_text(), encoding="utf-8"
        )
        (durable_dir / "run_meta.json").write_text(_nonfinite_metadata_text(), encoding="utf-8")
    else:
        with sqlite3.connect(tmp_path / "checkpoints.db") as connection:
            connection.execute(
                "UPDATE checkpoints SET manifest=? WHERE run_id=? AND seq=?",
                (_nonfinite_checkpoint_text(), "run_1", 1),
            )
            connection.execute(
                "UPDATE run_metadata SET metadata=? WHERE run_id=?",
                (_nonfinite_metadata_text(), "run_1"),
            )

    assert store.latest_checked("run_1").status == "corrupt"
    assert store.latest("run_1") is None
    assert store.run_metadata_checked("run_1").status == "corrupt"
    assert store.run_metadata("run_1") is None

    if backend == "local_fs":
        (durable_dir / "1" / "manifest.json").write_text(
            _legacy_surrogate_checkpoint_text(), encoding="utf-8"
        )
        (durable_dir / "run_meta.json").write_text(
            _legacy_surrogate_metadata_text(), encoding="utf-8"
        )
    else:
        with sqlite3.connect(tmp_path / "checkpoints.db") as connection:
            connection.execute(
                "UPDATE checkpoints SET manifest=? WHERE run_id=? AND seq=?",
                (_legacy_surrogate_checkpoint_text(), "run_1", 1),
            )
            connection.execute(
                "UPDATE run_metadata SET metadata=? WHERE run_id=?",
                (_legacy_surrogate_metadata_text(), "run_1"),
            )

    checkpoint = store.latest_checked("run_1")
    assert checkpoint.status == "loaded"
    assert checkpoint.value.checkpoint.messages[0]["text"] == "bad\ufffdtext"
    metadata = store.run_metadata_checked("run_1")
    assert metadata.status == "loaded"
    assert metadata.value["extension"] == "bad\ufffdtext"


def test_json_writer_fallbacks_replace_lone_surrogates(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    write_json_atomic(path, {"text": "left\ud800right"})

    raw = path.read_text(encoding="utf-8")
    assert json.loads(raw) == {"text": "left\ufffdright"}
    assert r"\ud800" not in raw.lower()

    handle = io.StringIO()
    _write_jsonl(handle, {"data": {"text": "left\ud800right"}})
    line = handle.getvalue()
    assert json.loads(line) == {"data": {"text": "left\ufffdright"}}
    assert r"\ud800" not in line.lower()


def test_event_append_rejects_nonfinite_status_before_writing_event(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    status_path = run_dir / "status.json"
    status_path.write_text('{"run_id":"run_1","extension":NaN}', encoding="utf-8")

    with pytest.raises(EventLogCorruption, match="watermark cannot be verified"):
        append_event_to_run(run_dir, "run.started")

    assert not (run_dir / "events.jsonl").exists()
    assert status_path.read_text(encoding="utf-8") == '{"run_id":"run_1","extension":NaN}'


def test_operational_artifact_readers_reject_nonfinite_constants(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"permission_policy":{"redact_patterns":[]},"extension":NaN}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest.json"):
        run_permission_policy(tmp_path)

    job_path = tmp_path / "job.json"
    job_path.write_text('{"job_id":"job_1","score":Infinity}', encoding="utf-8")
    with pytest.raises(ValueError, match="job artifact"):
        _read_job_artifact(job_path)

    assert _read_json_if_exists(job_path) == {}
    assert _read_optional_json(job_path) is None

    job_path.write_text('{"job_id":"job_1","detail":"\\ud800"}', encoding="utf-8")
    assert _read_job_artifact(job_path)["detail"] == "\ufffd"
    assert _read_json_if_exists(job_path)["detail"] == "\ufffd"
    assert _read_optional_json(job_path)["detail"] == "\ufffd"


def test_corrupt_recovery_counter_and_lease_fail_open_for_recovery(tmp_path: Path) -> None:
    recovery = object.__new__(RecoveryService)
    (tmp_path / "recover_attempts.json").write_text('{"count":Infinity}', encoding="utf-8")
    assert recovery.read_recover_attempts(tmp_path) == 0

    run_dir = tmp_path / "run_lease"
    run_dir.mkdir()
    (run_dir / "lease.json").write_text(
        '{"run_id":"run_lease","worker_id":"dead","heartbeat_at":NaN,"lease_ttl_s":30}',
        encoding="utf-8",
    )
    lease = LocalFsLeaseStore(tmp_path)

    assert lease.owner("run_lease") is None
    assert lease.is_stale("run_lease") is True
    assert lease.try_claim("run_lease", "worker_new", 30) is True
    assert lease.owner("run_lease") == "worker_new"


def test_backend_json_safe_projection_normalizes_non_json_numbers_and_unicode() -> None:
    projected = _json_safe({"score": float("nan"), "text": "bad\ud800text"})

    assert projected == {"score": None, "text": "bad\ufffdtext"}
    json.dumps(projected, ensure_ascii=False, allow_nan=False).encode("utf-8")


def test_backend_json_safe_projection_bounds_excessive_nesting() -> None:
    value: Any = "leaf"
    for _ in range(sys.getrecursionlimit() + 100):
        value = [value]

    assert _json_safe(value) == "<value exceeds JSON nesting limit>"


class _NonJsonWebProvider:
    def search(self, query: str, *, max_results: int) -> list[dict[str, Any]]:
        del query, max_results
        return [
            {
                "url": "https://docs.example.test/result",
                "title": "bad\ud800title",
                "score": float("nan"),
            }
        ]

    def fetch(
        self,
        url: str,
        *,
        format: str,
        allowed_domains: tuple[str, ...] = (),
        blocked_domains: tuple[str, ...] = (),
        timeout_s: int | None = None,
        max_bytes: int | None = None,
    ) -> dict[str, Any]:
        del format, allowed_domains, blocked_domains, timeout_s, max_bytes
        return {
            "final_url": url,
            "title": "title",
            "content": "bad\ud800content",
            "source": "test",
        }

    def context(
        self,
        query: str,
        *,
        max_tokens: int,
        max_urls: int,
        max_snippets: int,
        locale: str | None,
        freshness: str | None,
        allowed_domains: tuple[str, ...],
        blocked_domains: tuple[str, ...],
    ) -> dict[str, Any]:
        del (
            query,
            max_tokens,
            max_urls,
            max_snippets,
            locale,
            freshness,
            allowed_domains,
            blocked_domains,
        )
        return {
            "context": "bad\ud800context",
            "sources": [{"url": "https://docs.example.test/result", "score": float("-inf")}],
            "chunks": [],
            "source": "test",
        }


def _web_token(manager: TokenManager) -> str:
    return manager.issue(
        kind="web_gateway",
        audience="csp.web-gateway",
        run_id="run_1",
        tenant_id="tenant_a",
        user_id="user_a",
        ttl_s=600,
        metadata={"agent_config_hash": "test"},
    )


def test_web_gateway_normalizes_provider_results_before_projection() -> None:
    manager = TokenManager.from_secret("w" * 32)
    gateway = WebGatewayBackend(token_manager=manager, provider=_NonJsonWebProvider())
    token = _web_token(manager)

    search = gateway.handle_search(token, {"query": "test"})
    fetched = gateway.handle_fetch(token, {"url": "https://docs.example.test/result"})
    context = gateway.handle_context(token, {"query": "test"})

    assert search["results"][0]["score"] is None
    assert search["results"][0]["title"] == "bad\ufffdtitle"
    assert fetched["content"] == "bad\ufffdcontent"
    assert context["context"] == "bad\ufffdcontext"
    assert context["sources"][0]["score"] is None
    json.dumps(
        {"search": search, "fetch": fetched, "context": context},
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def test_provider_http_json_boundary_normalizes_writes_and_parses_strictly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, bytes | None] = {}

    def successful_request(request: Any, **kwargs: Any) -> bytes:
        del kwargs
        captured["data"] = request.data
        return b'{"text":"\\ud800"}'

    monkeypatch.setattr(web_providers, "_urlopen_read_with_retry", successful_request)
    response = web_providers._request_json(
        "https://provider.example.test",
        headers={},
        timeout_s=1,
        method="POST",
        body={"score": float("nan"), "text": "bad\ud800text"},
    )

    assert response == {"text": "\ufffd"}
    assert captured["data"] is not None
    assert json.loads(captured["data"].decode("utf-8")) == {
        "score": None,
        "text": "bad\ufffdtext",
    }

    monkeypatch.setattr(
        web_providers,
        "_urlopen_read_with_retry",
        lambda request, **kwargs: b'{"score":NaN}',
    )
    with pytest.raises(WebGatewayError, match="returned invalid JSON"):
        web_providers._request_json("https://provider.example.test", headers={}, timeout_s=1)


def test_sqlite_command_inbox_normalizes_content_before_identity_and_writes(
    tmp_path: Path,
) -> None:
    store = SqliteCommandStore(tmp_path / "commands.db")
    principal = CommandPrincipal("tenant", "user", "operator")
    lone_surrogate = chr(0xD800)
    submitted = StoredCommand(
        run_id="run_1",
        command_id=f"cmd_{lone_surrogate}",
        type="status",
        args={"score": float("nan"), "text": f"bad{lone_surrogate}text"},
        principal=principal,
    )
    first = store.append(submitted, max_pending=10)
    duplicate = store.append(submitted, max_pending=10)

    assert duplicate == first
    assert first.command_id == "cmd_\ufffd"
    persisted = store.read_command("run_1", f"cmd_{lone_surrogate}")
    assert persisted is not None
    assert persisted.args == {"score": None, "text": "bad\ufffdtext"}

    invalid_control = StoredCommand(
        run_id="run_1",
        command_id="cmd_invalid_control",
        type="status",
        args={},
        principal=principal,
        created_at=float("nan"),
    )
    with pytest.raises(ValueError, match="created_at must be a finite number"):
        store.append(invalid_control, max_pending=10)
    assert store.read_command("run_1", "cmd_invalid_control") is None

    valid = StoredCommand(
        run_id="run_2",
        command_id="cmd_valid",
        type="status",
        args={},
        principal=principal,
    )
    store.append(valid, max_pending=10)
    assert store.claim("run_2", "worker_1", claim_ttl_s=60) is not None
    receipt = store.acknowledge(
        "run_2",
        "cmd_valid",
        "worker_1",
        ControlResult(
            run_id="run_1",
            type="status",
            status="ok",
            data={"score": float("inf")},
        ),
    )
    assert receipt.result is not None
    assert receipt.result["data"]["score"] is None


def test_sqlite_command_inbox_readers_reject_legacy_nonfinite_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "commands.db"
    store = SqliteCommandStore(db_path)
    command = StoredCommand(
        run_id="run_1",
        command_id="cmd_legacy",
        type="status",
        args={},
        principal=CommandPrincipal("tenant", "user", "operator"),
    )
    store.append(command, max_pending=10)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE command_inbox SET args=? WHERE run_id=? AND command_id=?",
            ('{"text":"\\ud800"}', "run_1", "cmd_legacy"),
        )
    loaded = store.read_command("run_1", "cmd_legacy")
    assert loaded is not None
    assert loaded.args == {"text": "\ufffd"}

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE command_inbox SET args=? WHERE run_id=? AND command_id=?",
            ('{"score":NaN}', "run_1", "cmd_legacy"),
        )
    with pytest.raises(json.JSONDecodeError, match="non-finite number"):
        store.read_command("run_1", "cmd_legacy")

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE command_inbox SET args=?, result=? WHERE run_id=? AND command_id=?",
            ("{}", '{"score":Infinity}', "run_1", "cmd_legacy"),
        )
    with pytest.raises(json.JSONDecodeError, match="non-finite number"):
        store.receipt("run_1", "cmd_legacy")


def test_sqlite_command_inbox_readers_reject_array_of_pairs_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "commands.db"
    store = SqliteCommandStore(db_path)
    command = StoredCommand(
        run_id="run_1",
        command_id="cmd_shape",
        type="status",
        args={},
        principal=CommandPrincipal("tenant", "user", "operator"),
    )
    store.append(command, max_pending=10)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE command_inbox SET args=? WHERE run_id=? AND command_id=?",
            ('[["message","coerced"]]', "run_1", "cmd_shape"),
        )
    with pytest.raises(ValueError, match="command args must be an object"):
        store.read_command("run_1", "cmd_shape")

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE command_inbox SET args=?, principal=? " "WHERE run_id=? AND command_id=?",
            (
                "{}",
                '[["tenant_id","tenant"],["user_id","user"]]',
                "run_1",
                "cmd_shape",
            ),
        )
    with pytest.raises(ValueError, match="command principal must be an object"):
        store.read_command("run_1", "cmd_shape")

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE command_inbox SET principal=?, result=? " "WHERE run_id=? AND command_id=?",
            (
                '{"tenant_id":"tenant","user_id":"user"}',
                '[["status","ok"]]',
                "run_1",
                "cmd_shape",
            ),
        )
    retained = store.read_command("run_1", "cmd_shape")
    assert retained is not None
    assert retained.principal == CommandPrincipal("tenant", "user", "")
    with pytest.raises(ValueError, match="command result must be an object"):
        store.receipt("run_1", "cmd_shape")


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (("tenant_id", True), ("user_id", 7), ("issuer", None)),
)
def test_sqlite_command_inbox_reader_rejects_non_string_principal_fields(
    tmp_path: Path,
    field_name: str,
    invalid_value: Any,
) -> None:
    db_path = tmp_path / "commands.db"
    store = SqliteCommandStore(db_path)
    command = StoredCommand(
        run_id="run_1",
        command_id="cmd_principal",
        type="status",
        args={},
        principal=CommandPrincipal("tenant", "user", "operator"),
    )
    store.append(command, max_pending=10)
    principal: dict[str, Any] = {
        "tenant_id": "tenant",
        "user_id": "user",
        "issuer": "operator",
    }
    principal[field_name] = invalid_value

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE command_inbox SET principal=? WHERE run_id=? AND command_id=?",
            (json.dumps(principal), "run_1", "cmd_principal"),
        )

    with pytest.raises(ValueError, match=f"command principal {field_name} must be a string"):
        store.read_command("run_1", "cmd_principal")


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    (
        ("run_id", True, "command result run_id must be a string"),
        ("type", 7, "command result type must be a string"),
        ("status", None, "command result status must be a string"),
        ("data", [["message", "coerced"]], "command result data must be an object"),
    ),
)
def test_sqlite_command_inbox_reader_rejects_invalid_result_fields(
    tmp_path: Path,
    field_name: str,
    invalid_value: Any,
    message: str,
) -> None:
    db_path = tmp_path / "commands.db"
    store = SqliteCommandStore(db_path)
    command = StoredCommand(
        run_id="run_1",
        command_id="cmd_result",
        type="status",
        args={},
        principal=CommandPrincipal("tenant", "user", "operator"),
    )
    store.append(command, max_pending=10)
    result = ControlResult(run_id="run_1", type="status", status="ok").to_json()
    result[field_name] = invalid_value

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE command_inbox SET result=? WHERE run_id=? AND command_id=?",
            (json.dumps(result), "run_1", "cmd_result"),
        )

    with pytest.raises(ValueError, match=message):
        store.receipt("run_1", "cmd_result")


def test_gateway_http_writer_rejects_nonfinite_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = GatewayModelAdapter(
        ModelConfig(gateway_url="https://llm-gateway.example.test/v1/turns"), token="token"
    )
    monkeypatch.setattr(adapter, "_payload", lambda request: {"score": float("nan")})

    with pytest.raises(ValueError, match="Out of range float values"):
        adapter.next_turn(ModelRequest(instruction="test", system_prompt="sys", tools=()))


@pytest.mark.asyncio
async def test_gateway_stream_http_writer_rejects_nonfinite_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = GatewayModelAdapter(
        ModelConfig(gateway_url="https://llm-gateway.example.test/v1/turns"), token="token"
    )
    monkeypatch.setattr(adapter, "_payload", lambda request: {"score": float("nan")})

    stream = adapter.astream_turn(ModelRequest(instruction="test", system_prompt="sys", tools=()))
    with pytest.raises(ValueError, match="Out of range float values"):
        await anext(stream)
