from __future__ import annotations

import base64
import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

from support.http import (
    http_get_json as _json_get,
    http_post_json as _json_request,
    wait_http_ready as _wait_http_ready,
)
from support.backend_factory import current_backend_factory
from support.process import python_command as _python_command
from support.runtime import runtime_config, tool_binding
from support.waiting import eventually

from monoid_agent_kernel.core._util import write_json_atomic
from monoid_agent_kernel.core.agents import AgentRuntimeConfig, OutputValidatorBinding
from monoid_agent_kernel.core.checkpoint import RunCheckpoint
from monoid_agent_kernel.core.tool_surface import ToolScope
from monoid_agent_kernel.core.spec import ModelRetryConfig
from monoid_agent_kernel.errors import AgentConfigError, ModelAdapterError, PermissionDenied
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.core.content import ImagePart, TextPart
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import (
    FakeModelAdapter,
    FakeMultimodalModelAdapter,
    fake_tool_call,
)
from monoid_agent_kernel.reference._shared.tokens import TokenError, TokenManager
from monoid_agent_kernel.skills import SkillDefinition, SkillProvider
from monoid_agent_kernel.reference.backend.http import create_backend_server
from monoid_agent_kernel.reference.backend.service import (
    _RUN_META_SCHEMA_VERSION,
    BackendRunRequest,
    RunnerBackend as _RunnerBackend,
)
from monoid_agent_kernel.reference.stores.sqlite import SqliteCheckpointStore, SqliteLeaseStore

__all__ = [
    'AgentConfigError',
    'AgentRuntimeConfig',
    'BackendRunRequest',
    'FakeModelAdapter',
    'FakeMultimodalModelAdapter',
    'HTTPError',
    'ImagePart',
    'ModelAdapterError',
    'ModelRetryConfig',
    'ModelTurn',
    'OutputValidatorBinding',
    'Path',
    'PermissionDenied',
    'PermissionPolicy',
    'Request',
    'RunCheckpoint',
    'RunnerBackend',
    'SkillDefinition',
    'SkillProvider',
    'SqliteCheckpointStore',
    'SqliteLeaseStore',
    'TextPart',
    'TokenError',
    'TokenManager',
    'ToolScope',
    'URLError',
    '_BlockingAdapter',
    '_InterruptingTurnAdapter',
    '_PNG_1x1',
    '_RUN_META_SCHEMA_VERSION',
    '_ScriptedTurnAdapter',
    '_backend',
    '_calls',
    '_default_config',
    '_hitl_backend',
    '_json_get',
    '_json_request',
    '_provider_backend',
    '_python_command',
    '_recoverable_backend',
    '_running_hitl_tasks',
    '_scripted_backend',
    '_skill_provider',
    '_stale_lease_payload',
    '_start_server',
    '_submit_multi_turn',
    '_token_manager',
    '_wait_http_ready',
    '_workspace',
    'annotations',
    'base64',
    'create_backend_server',
    'eventually',
    'fake_tool_call',
    'json',
    'pytest',
    'runtime_config',
    'threading',
    'time',
    'tool_binding',
    'urlopen',
    'write_json_atomic',
]


class RunnerBackend(_RunnerBackend):
    """Test-support alias that registers direct constructions with the active factory."""

    def __post_init__(self) -> None:
        super().__post_init__()
        managed_factory = current_backend_factory()
        if managed_factory is not None:
            managed_factory.track(self)


def _token_manager() -> TokenManager:
    return TokenManager.from_secret("x" * 32)


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("notes\n", encoding="utf-8")
    return workspace


def _default_config():
    return runtime_config("fs.read", "fs.write", "run.finish")


def _backend(tmp_path: Path, workspace: Path, captured_gateway_tokens: list[str]) -> RunnerBackend:
    token_manager = _token_manager()

    def factory(spec, llm_gateway_token):
        captured_gateway_tokens.append(llm_gateway_token)
        claims = token_manager.verify(
            llm_gateway_token,
            kind="llm_gateway",
            audience="csp.llm-gateway",
            run_id=spec.run_id,
        )
        assert claims.tenant_id == "tenant_a"
        return FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="turn_1",
                    final_text="done",
                    usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                )
            ]
        )

    managed_factory = current_backend_factory()
    if managed_factory is not None:
        return managed_factory.create(
            run_root=tmp_path / "runs",
            workspace=workspace,
            token_manager=token_manager,
            model_adapter_factory=factory,
        )
    return RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
    )


def _hitl_backend(tmp_path: Path, workspace: Path, adapters: list, turns: list) -> RunnerBackend:
    token_manager = _token_manager()

    def factory(spec, llm_gateway_token):
        del spec, llm_gateway_token
        adapter = FakeModelAdapter(turns=list(turns))
        adapters.append(adapter)
        return adapter

    managed_factory = current_backend_factory()
    if managed_factory is not None:
        return managed_factory.create(
            run_root=tmp_path / "runs",
            workspace=workspace,
            token_manager=token_manager,
            model_adapter_factory=factory,
        )
    return RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
    )


def _running_hitl_tasks(backend: RunnerBackend, run_id: str) -> list:
    record = backend._record(run_id)
    loop = record.loop
    if loop is None or loop._session is None:
        return []
    manager = loop._session.res.context.job_manager
    return [task for task in list(manager.jobs.values()) if task.kind == "hitl" and task.status == "running"]


def _recoverable_backend(run_root: Path, token_manager: TokenManager, workspace: Path, adapters: list, turns: list) -> RunnerBackend:
    def factory(spec, llm_gateway_token):
        del spec, llm_gateway_token
        adapter = FakeModelAdapter(turns=list(turns))
        adapters.append(adapter)
        return adapter

    managed_factory = current_backend_factory()
    if managed_factory is not None:
        return managed_factory.create(
            run_root=run_root,
            workspace=workspace,
            token_manager=token_manager,
            model_adapter_factory=factory,
        )
    return RunnerBackend(
        run_root=run_root,
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
    )


def _skill_provider() -> SkillProvider:
    return SkillProvider(
        {
            "greeter": SkillDefinition(
                name="greeter",
                description="Greet the user warmly.",
                instructions="Say a warm hello to the user.",
            )
        }
    )


def _provider_backend(
    run_root: Path,
    token_manager: TokenManager,
    workspace: Path,
    *,
    turns: list,
    adapters: list | None = None,
    provider: SkillProvider | None = None,
) -> RunnerBackend:
    sink = adapters if adapters is not None else []

    def factory(spec, llm_gateway_token):
        del spec, llm_gateway_token
        adapter = FakeModelAdapter(turns=list(turns))
        sink.append(adapter)
        return adapter

    skill = provider if provider is not None else _skill_provider()
    managed_factory = current_backend_factory()
    if managed_factory is not None:
        return managed_factory.create(
            run_root=run_root,
            workspace=workspace,
            token_manager=token_manager,
            model_adapter_factory=factory,
            tool_providers=(skill,),
            context_providers=(skill,),
        )
    return RunnerBackend(
        run_root=run_root,
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
        tool_providers=(skill,),
        context_providers=(skill,),
    )


_PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _stale_lease_payload(run_id: str) -> dict:
    # A lease whose heartbeat is far in the past -> the owning worker is presumed crashed.
    return {"run_id": run_id, "worker_id": "dead", "pid": 1, "heartbeat_at": time.time() - 1000.0, "lease_ttl_s": 30.0}


class _BlockingAdapter:
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self._started = started
        self._release = release

    def next_turn(self, _request):
        self._started.set()
        self._release.wait(timeout=10)
        return ModelTurn(response_id="r1", final_text="done", usage={"total_tokens": 1})


def _start_server(backend: RunnerBackend):
    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    _wait_http_ready(base_url)
    return server, thread, base_url


class _ScriptedTurnAdapter:
    """Drives a script of turns/exceptions: a ModelTurn is returned, a BaseException raised."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.requests: list = []

    def next_turn(self, request):  # noqa: ANN001
        self.requests.append(request)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _scripted_backend(tmp_path: Path, workspace: Path, adapters: list, script: list) -> RunnerBackend:
    token_manager = _token_manager()

    def factory(spec, llm_gateway_token):  # noqa: ANN001
        del spec, llm_gateway_token
        adapter = _ScriptedTurnAdapter(script)
        adapters.append(adapter)
        return adapter

    managed_factory = current_backend_factory()
    if managed_factory is not None:
        backend = managed_factory.create(
            run_root=tmp_path / "runs",
            workspace=workspace,
            token_manager=token_manager,
            model_adapter_factory=factory,
        )
    else:
        backend = RunnerBackend(
            run_root=tmp_path / "runs",
            token_manager=token_manager,
            allowed_workspace_roots=(workspace,),
            llm_gateway_url="http://llm-gateway.internal/v1/turns",
            model_adapter_factory=factory,
        )
    backend.turn_retry = ModelRetryConfig(initial_delay_s=0.0, jitter_s=0.0, max_delay_s=0.0)
    return backend


def _calls(adapters: list) -> int:
    return sum(len(a.requests) for a in adapters)


def _submit_multi_turn(backend: RunnerBackend, workspace: Path):
    return backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hi",
            runtime_config=_default_config(),
            multi_turn=True,
        )
    )


class _InterruptingTurnAdapter:
    """First model turn calls a tool; the second grabs the loop (handed in via ``loop_box``)
    and interrupts the turn — simulating a user "stop" mid-turn — then yields another tool
    call so a step boundary trips. A third call (after the user resumes) settles."""

    def __init__(self) -> None:
        self.requests: list = []
        self.loop_box: list = []
        self.calls = 0

    def next_turn(self, request):  # noqa: ANN001
        self.requests.append(request)
        self.calls += 1
        if self.calls == 1:
            return ModelTurn(response_id="r1", tool_calls=(fake_tool_call("fs_read", {"path": "x.md"}, "c1"),))
        if self.calls == 2:
            deadline = time.time() + 5.0
            while not self.loop_box and time.time() < deadline:
                time.sleep(0.01)
            self.loop_box[0].interrupt_turn()
            return ModelTurn(response_id="r2", tool_calls=(fake_tool_call("fs_read", {"path": "x.md"}, "c2"),))
        return ModelTurn(response_id="r3", final_text="resumed ok")
