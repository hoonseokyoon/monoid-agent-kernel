from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

from support.http import http_get_json as _json_get
from support.http import serving

from monoid_agent_kernel.core.agents import AgentRuntimeConfig, RegistryToolRef, ToolBinding
from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore
from monoid_agent_kernel.core.tool_surface import ToolGuidance
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn
from monoid_agent_kernel.providers.fake import fake_tool_call
from monoid_agent_kernel.reference.backend.http import create_backend_server
from monoid_agent_kernel.reference.backend.service import BackendRunRequest


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("hello\n", encoding="utf-8")
    return workspace


def _binding(tool_id: str, *, guidance: str = "") -> ToolBinding:
    return ToolBinding(
        binding_id=tool_id,
        model_name=tool_id.replace(".", "_"),
        ref=RegistryToolRef(tool_id),
        guidance=ToolGuidance(summary=guidance),
        title=tool_id,
    )


def _config(version: int, *bindings: ToolBinding) -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        definition_id="backend-agent",
        config_version=version,
        tools=bindings,
    )


class _BlockingAdapter:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.first_call_started = threading.Event()
        self.allow_first_return = threading.Event()

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        if len(self.requests) == 1:
            self.first_call_started.set()
            assert self.allow_first_return.wait(timeout=5)
            return ModelTurn(
                response_id="turn_1",
                tool_calls=(fake_tool_call("fs_read", {"path": "notes.md"}, "read1"),),
            )
        return ModelTurn(final_text="done")


class _FailingRunMetadataStore(LocalFsCheckpointStore):
    fail_run_metadata = False

    def put_run_metadata(self, run_id: str, metadata: Mapping[str, Any]) -> None:
        if self.fail_run_metadata:
            raise OSError("shared metadata unavailable")
        super().put_run_metadata(run_id, metadata)


def test_backend_runtime_config_endpoint_updates_next_turn(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    adapter = _BlockingAdapter()
    backend = backend_factory.create(
        workspace=workspace, model_adapter_factory=lambda _spec, _token: adapter
    )
    initial = _config(1, _binding("fs.read", guidance="initial read"), _binding("run.finish"))
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Read and finish.",
            runtime_config=initial,
        )
    )
    assert adapter.first_call_started.wait(timeout=5)

    current = backend.runtime_config(submission.run_id, submission.run_token)
    assert current["config_version"] == 1
    replacement = _config(2, _binding("fs.read", guidance="replacement read"), _binding("run.finish"))
    updated = backend.replace_runtime_config(
        submission.run_id,
        submission.run_token,
        expected_version=1,
        issuer="test",
        reason="replace guidance",
        config=replacement,
    )
    assert updated["config_version"] == 2
    assert updated["config_hash"] == replacement.config_hash
    assert updated["committed_at"] >= current["committed_at"]
    run_meta = json.loads((submission.run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_meta["runtime_config"]["config_version"] == 2
    assert run_meta["runtime_config_hash"] == replacement.config_hash
    assert run_meta["runtime_config_issuer"] == "test"
    assert run_meta["runtime_config_reason"] == "replace guidance"
    assert run_meta["runtime_config_committed_at"] == updated["committed_at"]
    adapter.allow_first_return.set()

    assert backend.wait_for_run(submission.run_id, timeout_s=5) == "completed"
    second_read = next(tool for tool in adapter.requests[1].tools if tool.id == "fs.read")
    assert "replacement read" in second_read.description


def test_runtime_config_metadata_store_failure_keeps_local_descriptor_unchanged(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    adapter = _BlockingAdapter()
    run_root = tmp_path / "runs"
    store = _FailingRunMetadataStore(run_root)
    backend = backend_factory.create(
        run_root=run_root,
        workspace=workspace,
        model_adapter_factory=lambda _spec, _token: adapter,
        checkpoint_store=store,
    )
    initial = _config(1, _binding("fs.read", guidance="initial read"), _binding("run.finish"))
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Read and finish.",
            runtime_config=initial,
        )
    )
    assert adapter.first_call_started.wait(timeout=5)

    run_meta_before = json.loads((submission.run_dir / "run.json").read_text(encoding="utf-8"))
    stored_meta_before = store.run_metadata(submission.run_id)
    store.fail_run_metadata = True
    replacement = _config(2, _binding("fs.read", guidance="replacement read"), _binding("run.finish"))

    with pytest.raises(OSError, match="shared metadata unavailable"):
        backend.replace_runtime_config(
            submission.run_id,
            submission.run_token,
            expected_version=1,
            issuer="test",
            reason="replace guidance",
            config=replacement,
        )

    assert backend.runtime_config(submission.run_id, submission.run_token)["config_version"] == 1
    run_meta_after = json.loads((submission.run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_meta_after == run_meta_before
    assert store.run_metadata(submission.run_id) == stored_meta_before
    adapter.allow_first_return.set()
    assert backend.wait_for_run(submission.run_id, timeout_s=5) == "completed"


def test_backend_http_runtime_config_get_post_and_version_mismatch(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    adapter = _BlockingAdapter()
    backend = backend_factory.create(
        workspace=workspace, model_adapter_factory=lambda _spec, _token: adapter
    )
    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    with serving(server) as base_url:
        created = _json_post(
            f"{base_url}/v1/runs",
            {
                "tenant_id": "tenant_a",
                "user_id": "user_a",
                "workspace_root": str(workspace),
                "instruction": "Read and finish.",
                "runtime_config": _config(
                    1,
                    _binding("fs.read", guidance="http initial"),
                    _binding("run.finish"),
                ).to_json(),
            },
            token="admin",
        )
        run_id = created["run_id"]
        run_token = created["run_token"]
        assert adapter.first_call_started.wait(timeout=5)

        current = _json_get(f"{base_url}/v1/runs/{run_id}/runtime-config", token=run_token)
        assert current["ready"] is True
        assert current["config_version"] == 1

        mismatch = _json_post_raw(
            f"{base_url}/v1/runs/{run_id}/runtime-config",
            {
                "expected_version": 0,
                "issuer": "test",
                "reason": "bad version",
                "config": _config(2, _binding("run.finish")).to_json(),
            },
            token=run_token,
        )
        assert mismatch["status"] == 400

        invalid_tool = _json_post_raw(
            f"{base_url}/v1/runs/{run_id}/runtime-config",
            {
                "expected_version": 1,
                "issuer": "test",
                "reason": "bad tool",
                "config": _config(2, _binding("missing.tool")).to_json(),
            },
            token=run_token,
        )
        assert invalid_tool["status"] == 400
        assert "unknown registry tool" in invalid_tool["body"]["error"]

        updated = _json_post(
            f"{base_url}/v1/runs/{run_id}/runtime-config",
            {
                "expected_version": 1,
                "issuer": "test",
                "reason": "replace",
                "config": _config(
                    2,
                    _binding("fs.read", guidance="http replacement"),
                    _binding("run.finish"),
                ).to_json(),
            },
            token=run_token,
        )
        assert updated["config_version"] == 2
        adapter.allow_first_return.set()
        assert backend.wait_for_run(run_id, timeout_s=5) == "completed"
        second_read = next(tool for tool in adapter.requests[1].tools if tool.id == "fs.read")
        assert "http replacement" in second_read.description
    adapter.allow_first_return.set()


def _json_post(url: str, payload: dict, *, token: str) -> dict:
    result = _json_post_raw(url, payload, token=token)
    assert result["status"] < 400, result
    return result["body"]


def _json_post_raw(url: str, payload: dict, *, token: str) -> dict:
    # Captures the HTTP response (including 4xx error bodies) rather than raising, while
    # retrying transient connection-level errors under load (never an HTTPError, which is a
    # real response).
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urlopen(Request(url, data=body, headers=headers, method="POST"), timeout=5) as response:
                return {"status": response.status, "body": json.loads(response.read().decode("utf-8"))}
        except HTTPError as exc:
            return {"status": exc.code, "body": json.loads(exc.read().decode("utf-8"))}
        except (URLError, ConnectionError, OSError) as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    raise last_error if last_error is not None else RuntimeError("request failed without an error")
