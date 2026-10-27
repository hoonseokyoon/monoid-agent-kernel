from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

from support.runtime import runtime_config, tool_binding
from monoid_agent_kernel.core.authority import ActivationWriteAuthority, WriteAuthorityRevoked
from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore
from monoid_agent_kernel.core.interruption import InterruptionCause
from monoid_agent_kernel.core.spec import AgentRunSpec, RunLimits
from monoid_agent_kernel.core.tool_surface import DefaultToolSurfaceResolver, ToolScope
from monoid_agent_kernel.errors import (
    AgentConfigError,
    NativeAgentError,
    RunCancelled,
    RunTimeout,
    ToolExecutionError,
    TurnNotSettled,
)
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.providers.fake import FakeModelAdapter, ModelTurn, fake_tool_call
from monoid_agent_kernel.reference.host_tool_report import run_report
from monoid_agent_kernel.tool_loader import FunctionToolProvider
from monoid_agent_kernel.tools.base import ToolResult, ToolSpec
from monoid_agent_kernel.tools.host_bindings import (
    HostToolBinding,
    HostToolKey,
    ToolWorkspace,
    require_host_tool_binding,
)
from monoid_agent_kernel.workspace.local import LocalWorkspaceBackend

pytestmark = pytest.mark.serial
WS = HostToolKey[ToolWorkspace](
    "test.workspace", accepts=lambda view: isinstance(view, ToolWorkspace)
)


class Resource:
    def __init__(self, bind: Any = None, close: Any = None) -> None:
        self.bind = bind or (lambda lease: lease)
        self.on_close = close or (lambda: None)
        self.bound = []
        self.closed = 0

    def bind_call(self, lease: Any) -> Any:
        self.bound.append(lease)
        return self.bind(lease)

    def close(self) -> None:
        self.closed += 1
        return self.on_close()


def _spec(root: Path, **kwargs: Any) -> AgentRunSpec:
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return AgentRunSpec(workspace_root=workspace, run_root=root / "runs", **kwargs)


def _tool(handler: Any, side_effect: str = "write") -> ToolSpec:
    return ToolSpec("test.action", "test action", {"type": "object"}, "", side_effect, handler)


def _loop(
    root: Path,
    handler: Any = None,
    *,
    bindings: Any = None,
    side_effect: str = "write",
    calls: int = 1,
    scope: ToolScope | None = None,
    spec: AgentRunSpec | None = None,
    **kwargs: Any,
) -> AgentLoop:
    action = _tool(
        handler or (lambda _ctx, _args: ToolResult(ok=True, content={"ok": True})), side_effect
    )
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                tool_calls=tuple(fake_tool_call("test_action", {}, f"c{i}") for i in range(calls))
            ),
            ModelTurn(final_text="done"),
        ]
    )
    injections = (
        tuple(bindings)
        if bindings is not None
        else (HostToolBinding(WS, open=lambda activation: activation.workspace_resource()),)
    )
    if scope is None:
        return AgentLoop.from_tools(
            spec or _spec(root), adapter, [action], host_tool_bindings=injections, **kwargs
        )
    return AgentLoop.from_config(
        spec or _spec(root),
        adapter,
        runtime_config(bindings=(tool_binding("test.action", scope=scope),)),
        tool_providers=(FunctionToolProvider(lambda _ctx: (action,)),),
        host_tool_bindings=injections,
        **kwargs,
    )


def _outcomes(loop: AgentLoop) -> list[dict[str, Any]]:
    return [
        obs.output
        for request in loop.model_adapter.requests
        for obs in request.observations
        if obs.tool_name == "test_action"
    ]


def test_report_proposal_approval_release_and_fresh_restore(tmp_path: Path) -> None:
    assert run_report(tmp_path) == {
        "status": "completed",
        "activations": 2,
        "approved_read": True,
        "proposal_only": True,
    }


@pytest.mark.parametrize(
    "invalid", ["duplicate_key", "duplicate_name", "invalid_entry", "async_open", "async_accepts"]
)
def test_registration_validation_precedes_allocation(tmp_path: Path, invalid: str) -> None:
    async def async_value(*args: Any) -> Any:
        return None

    factory_calls = []
    key = HostToolKey("key", accepts=lambda _: True)
    first = HostToolBinding(key, open=lambda _: factory_calls.append(True))
    cases = {
        "duplicate_key": (first, first),
        "duplicate_name": (
            first,
            HostToolBinding(HostToolKey("key", accepts=lambda _: True), open=lambda _: None),
        ),
        "invalid_entry": (object(),),
        "async_open": (HostToolBinding(key, open=async_value),),
        "async_accepts": (
            HostToolBinding(HostToolKey("async", accepts=async_value), open=lambda _: None),
        ),
    }
    spec = _spec(tmp_path)
    authority = ActivationWriteAuthority()
    with pytest.raises(AgentConfigError):
        _loop(tmp_path, spec=spec, bindings=cases[invalid], write_authority=authority)
    assert not factory_calls and not spec.run_root.exists()
    assert not authority._callbacks


def test_tuple_snapshot_identity_lookup_and_provider_has_no_call(tmp_path: Path) -> None:
    resource = Resource()
    key = HostToolKey("key", accepts=lambda _: True)
    key_twin = HostToolKey("key", accepts=lambda _: True)
    seen = []

    def handler(ctx: Any, _args: Any) -> ToolResult:
        lease = require_host_tool_binding(ctx, key)
        assert require_host_tool_binding(ctx, key) is lease
        with pytest.raises(ToolExecutionError) as missing:
            require_host_tool_binding(ctx, key_twin)
        assert missing.value.error_code == "host_tool_binding_missing"
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            lease.binding_id = "replacement"
        seen.append((lease.run_id, lease.binding_id, lease.tool_call_id, ctx, lease))
        return ToolResult(ok=True, content={"ok": True})

    action = _tool(handler)

    def provide(ctx: Any) -> tuple[ToolSpec, ...]:
        with pytest.raises(ToolExecutionError) as outside:
            require_host_tool_binding(ctx, key)
        assert outside.value.error_code == "tool_call_not_in_flight"
        return (action,)

    registrations = [HostToolBinding(key, open=lambda _: resource)]
    spec = _spec(tmp_path)
    loop = AgentLoop.from_config(
        spec,
        FakeModelAdapter(turns=[ModelTurn(tool_calls=(fake_tool_call("test_action", {}),))]),
        runtime_config("test.action"),
        tool_providers=(FunctionToolProvider(provide),),
        host_tool_bindings=registrations,
    )
    registrations.clear()
    assert loop.run_once("go").status == "completed"
    assert len(resource.bound) == 1 and resource.closed == 1
    assert seen[0][:3] == (spec.run_id, "test.action", "call_fake")
    for operation in (lambda: require_host_tool_binding(seen[0][3], key), seen[0][4].assert_active):
        with pytest.raises(ToolExecutionError, match="not in flight"):
            operation()
    with pytest.raises(ToolExecutionError) as unsupported:
        require_host_tool_binding(object(), key)
    assert unsupported.value.error_code == "host_tool_context_unavailable"


@pytest.mark.parametrize("failure", ["open", "revoke", "duplicate", "invalid_bind", "awaitable"])
def test_partial_factory_cleanup_is_reverse_and_once(tmp_path: Path, failure: str) -> None:
    order = []
    authority = ActivationWriteAuthority()
    first = Resource(close=lambda: order.append("first"))
    second = Resource(close=lambda: order.append("second"))

    def open_second(_activation: Any) -> Any:
        if failure == "open":
            raise ValueError("SECRET_FACTORY")
        if failure == "revoke":
            authority.revoke()
        if failure == "duplicate":
            return first
        if failure == "invalid_bind":
            second.bind_call = None
        if failure == "awaitable":

            async def pending() -> Resource:
                return second

            return pending()
        return second

    bindings = (
        HostToolBinding(HostToolKey("first", accepts=lambda _: True), open=lambda _: first),
        HostToolBinding(HostToolKey("second", accepts=lambda _: True), open=open_second),
    )
    loop = _loop(tmp_path, bindings=bindings, write_authority=authority)
    with pytest.raises(
        WriteAuthorityRevoked if failure == "revoke" else AgentConfigError
    ) as caught:
        loop.open()
    assert "SECRET_FACTORY" not in str(caught.value)
    assert order == (["second", "first"] if failure in {"revoke", "invalid_bind"} else ["first"])
    assert loop._session is None and loop._bootstrap_resources is None
    loop.discard_uncommitted()
    assert first.closed == 1 and second.closed == (failure in {"revoke", "invalid_bind"})


@pytest.mark.parametrize("duplicate", [True, False])
def test_factory_return_after_public_discard_disposes_each_adapter_once(
    tmp_path: Path, duplicate: bool
) -> None:
    first, second = Resource(), Resource()

    def open_second(_activation: Any) -> Resource:
        loop.discard_uncommitted()
        return first if duplicate else second

    loop = _loop(
        tmp_path,
        bindings=(
            HostToolBinding(HostToolKey("first", accepts=lambda _: True), open=lambda _: first),
            HostToolBinding(HostToolKey("second", accepts=lambda _: True), open=open_second),
        ),
    )
    with pytest.raises(ToolExecutionError) as caught:
        loop.open()
    assert caught.value.error_code == "host_tool_resource_closed"
    assert first.closed == 1 and second.closed == (not duplicate)
    assert loop._session is None and loop._host_tool_registry is None
    loop.discard_uncommitted()
    assert first.closed == 1 and second.closed == (not duplicate)


@pytest.mark.parametrize("stage", ["bind", "accepts", "type", "async_bind", "async_accepts"])
def test_lookup_refusal_has_safe_diagnostics(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, stage: str
) -> None:
    def fail(*args: Any) -> Any:
        raise ValueError("SECRET_CALLBACK")

    def awaitable(*args: Any) -> Any:
        async def pending() -> Any:
            return "value"

        return pending()

    resource = Resource(
        bind=fail if stage == "bind" else awaitable if stage == "async_bind" else None
    )
    accepts = (
        fail
        if stage == "accepts"
        else awaitable
        if stage == "async_accepts"
        else (lambda _: stage != "type")
    )
    key = HostToolKey("safe", accepts=accepts)

    def handler(ctx: Any, _args: Any) -> ToolResult:
        return ToolResult(ok=True, content={"value": require_host_tool_binding(ctx, key)})

    loop = _loop(tmp_path, handler, bindings=(HostToolBinding(key, open=lambda _: resource),))
    loop.run_once("go")
    outcome = _outcomes(loop)[0]
    assert not outcome["ok"] and outcome["error"]["code"].startswith("host_tool_binding_")
    disk = "".join(p.read_text(encoding="utf-8") for p in loop.spec.run_root.rglob("*.json*"))
    assert "SECRET_CALLBACK" not in disk + caplog.text + json.dumps(outcome)
    assert resource.closed == 1


@pytest.mark.parametrize(
    "operation,path,side_effect,expected",
    [
        ("read", "private.txt", "write", "tool_scope_denied"),
        ("exists", "private.txt", "write", "tool_scope_denied"),
        ("write", "private.txt", "write", "tool_scope_denied"),
        ("read", "reports/secret.txt", "write", "permission_denied"),
        ("write", "reports/new.txt", "read", "permission_denied"),
        ("write", "reports/new.txt", "artifact", "permission_denied"),
        ("write", "reports/new.txt", "run", "permission_denied"),
        ("write", "reports/new.txt", "shell", "permission_denied"),
    ],
)
def test_computed_paths_check_scope_policy_and_admitted_operation(
    tmp_path: Path, operation: str, path: str, side_effect: str, expected: str
) -> None:
    def handler(ctx: Any, _args: Any) -> ToolResult:
        view = require_host_tool_binding(ctx, WS)
        if operation == "write":
            view.write_bytes(path, b"data", create_dirs=True)
        elif operation == "read":
            view.read_bytes(path)
        else:
            view.exists(path)
        return ToolResult(ok=True, content={})

    loop = _loop(
        tmp_path,
        handler,
        side_effect=side_effect,
        scope=ToolScope(allowed_paths=("reports/**",)),
        permission_policy=PermissionPolicy(deny_patterns=("reports/secret.txt",)),
    )
    loop.run_once("go")
    assert _outcomes(loop)[0]["error"]["code"] == expected
    assert not (loop.spec.workspace_root / path).exists()


@pytest.mark.parametrize("backend", ["overlay", "staging"])
def test_workspace_modes_hash_and_normalized_surface_scope(tmp_path: Path, backend: str) -> None:
    class Resolver(DefaultToolSurfaceResolver):
        def resolve(self, **kwargs: Any) -> Any:
            snapshot = super().resolve(**kwargs)
            return replace(
                snapshot,
                authorizations={
                    key: replace(
                        value, surface_scope=ToolScope(denied_paths=("reports/blocked.txt",))
                    )
                    for key, value in snapshot.authorizations.items()
                },
            )

    def handler(ctx: Any, _args: Any) -> ToolResult:
        view = require_host_tool_binding(ctx, WS)
        digest = view.write_bytes("reports//draft.txt", b"first", create_dirs=True)
        assert view.read_bytes("reports/draft.txt") == (b"first", digest)
        assert view.exists("reports/draft.txt")
        for path, options in [
            ("reports/draft.txt", {"overwrite": False}),
            ("reports/draft.txt", {"expected_sha256": "0" * 64}),
            ("../escape.txt", {}),
            ("reports/blocked.txt", {}),
        ]:
            with pytest.raises(NativeAgentError):
                view.write_bytes(path, b"invalid", **options)
        with pytest.raises(NativeAgentError):
            view.read_bytes("reports/draft.txt", max_bytes=1)
        digest = view.write_bytes("reports/draft.txt", b"second", expected_sha256=digest)
        assert view.read_bytes("reports/draft.txt") == (b"second", digest)
        return ToolResult(ok=True, content={"digest": digest})

    spec = _spec(tmp_path, mode="propose", workspace_backend=backend)
    loop = _loop(
        tmp_path,
        handler,
        spec=spec,
        scope=ToolScope(allowed_paths=("reports/**",)),
        tool_surface_resolver=Resolver(),
    )
    assert loop.run_once("go").status == "completed"
    assert _outcomes(loop)[0]["ok"]
    if backend == "overlay":
        assert not (spec.workspace_root / "reports/draft.txt").exists()
    else:
        assert (spec.workspace_root / "reports/draft.txt").read_bytes() == b"second"


@pytest.mark.parametrize(
    "operation",
    ["close", "discard_uncommitted", "release_parked", "revoked_close", "revoked_release"],
)
def test_lifecycle_disposes_once_without_revoking_shared_writer(
    tmp_path: Path, operation: str
) -> None:
    resource = Resource()
    key = HostToolKey("resource", accepts=lambda _: True)
    authority = ActivationWriteAuthority()
    loop = _loop(
        tmp_path,
        bindings=(HostToolBinding(key, open=lambda _: resource),),
        write_authority=authority,
    )
    loop.open()
    loop.run_until_suspended("go")
    record = LocalFsCheckpointStore(loop.spec.run_root).latest(loop.spec.run_id)
    assert record is not None
    if operation.startswith("revoked_"):
        authority.revoke()
        with pytest.raises(WriteAuthorityRevoked):
            getattr(loop, "close" if operation == "revoked_close" else "release_parked")()
    else:
        getattr(loop, operation)()
        assert not authority.revoked
    assert resource.closed == 1 and loop._session is None and loop._owned_loop is None
    loop.discard_uncommitted()
    assert resource.closed == 1
    if operation != "close":
        assert LocalFsCheckpointStore(loop.spec.run_root).latest(loop.spec.run_id) is not None


def test_uncommitted_release_keeps_resource_and_recordable_failure_closes_it(
    tmp_path: Path,
) -> None:
    resource = Resource()
    key = HostToolKey("resource", accepts=lambda _: True)
    loop = _loop(tmp_path, bindings=(HostToolBinding(key, open=lambda _: resource),))
    loop.open()
    try:
        with pytest.raises(NativeAgentError) as refused:
            loop.release_parked()
        assert refused.value.error_code == "run_not_durably_parked" and resource.closed == 0
    finally:
        loop.discard_uncommitted()
    other = Resource()
    failed = _loop(tmp_path / "failed", bindings=(HostToolBinding(key, open=lambda _: other),))

    class BadConfig:
        def current_config(self, _run_id: str) -> Any:
            raise ValueError("bad config")

    failed.runtime_config_provider = BadConfig()
    failed.open()
    assert other.closed == 1 and failed._session is not None
    assert not failed._session.res.recorder._transcript_file.closed
    assert failed.close().status == "failed" and other.closed == 1


@pytest.mark.parametrize("operation", ["close", "discard_uncommitted", "release_parked"])
def test_dispose_failures_attempt_rest_and_finish_teardown(
    tmp_path: Path, operation: str, caplog: pytest.LogCaptureFixture
) -> None:
    order = []

    def fail(number: int) -> None:
        order.append(number)
        raise ValueError("SECRET_CLOSE")

    resources = [Resource(close=lambda i=i: fail(i)) for i in range(3)]
    bindings = tuple(
        HostToolBinding(HostToolKey(str(i), accepts=lambda _: True), open=lambda _, r=r: r)
        for i, r in enumerate(resources)
    )
    loop = _loop(tmp_path, bindings=bindings)
    loop.open()
    loop.run_until_suspended("go")
    recorder = loop._session.res.recorder
    with pytest.raises(ToolExecutionError) as error:
        getattr(loop, operation)()
    assert error.value.error_code == "host_tool_binding_close"
    assert error.value.__context__ is None
    assert "SECRET_CLOSE" not in caplog.text + str(error.value)
    assert order == [2, 1, 0] and recorder._transcript_file.closed
    assert loop._session is None and loop._owned_loop is None
    loop.discard_uncommitted()
    assert all(r.closed == 1 for r in resources)


def test_retained_call_a_cannot_borrow_call_b_on_an_explicit_thread(tmp_path: Path) -> None:
    saved = []
    errors = []

    def handler(ctx: Any, _args: Any) -> ToolResult:
        view = require_host_tool_binding(ctx, WS)
        if not saved:
            saved.append((ctx, view))
            allowed = []
            thread = threading.Thread(
                target=lambda: allowed.append(require_host_tool_binding(ctx, WS).exists("ok.txt"))
            )
            thread.start()
            thread.join(timeout=3)
            assert not thread.is_alive() and allowed == [False]
        else:

            def late() -> None:
                for operation in (
                    lambda: require_host_tool_binding(saved[0][0], WS),
                    lambda: saved[0][1].write_bytes("late.txt", b"late"),
                ):
                    try:
                        operation()
                    except ToolExecutionError as exc:
                        errors.append(exc.error_code)

            thread = threading.Thread(target=late)
            thread.start()
            thread.join(timeout=3)
            assert not thread.is_alive()
            assert not view.exists("late.txt")
        return ToolResult(ok=True, content={})

    loop = _loop(tmp_path, handler, calls=2)
    loop.run_once("go")
    assert errors == ["tool_call_not_in_flight"] * 2
    assert all(result["ok"] for result in _outcomes(loop))


def test_concurrent_lookup_binds_once_without_holding_authority_locks(tmp_path: Path) -> None:
    key = HostToolKey("resource", accepts=lambda _: True)

    def bind(lease: Any) -> Any:
        checked = threading.Event()
        checker = threading.Thread(target=lambda: (lease.assert_active(), checked.set()))
        checker.start()
        checker.join(timeout=3)
        assert checked.is_set()
        return lease

    resource = Resource(bind=bind)

    def handler(ctx: Any, _args: Any) -> ToolResult:
        barrier = threading.Barrier(3)
        values = []

        def lookup() -> None:
            barrier.wait(timeout=3)
            values.append(require_host_tool_binding(ctx, key))

        threads = [threading.Thread(target=lookup) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=3)
        for thread in threads:
            thread.join(timeout=3)
        assert len(values) == 2 and values[0] is values[1]
        return ToolResult(ok=True, content={})

    loop = _loop(tmp_path, handler, bindings=(HostToolBinding(key, open=lambda _: resource),))
    loop.run_once("go")
    assert _outcomes(loop)[0]["ok"] and len(resource.bound) == 1


@pytest.mark.parametrize("shape", ["async", "sync_awaitable"])
def test_call_lives_through_await_and_ends_on_handler_cancellation(
    tmp_path: Path, shape: str
) -> None:
    retained = []

    async def handler(ctx: Any, _args: Any) -> ToolResult:
        view = require_host_tool_binding(ctx, WS)
        retained.append(view)
        await asyncio.sleep(0)
        assert require_host_tool_binding(ctx, WS) is view
        view.write_bytes("before.txt", b"before")
        raise asyncio.CancelledError()

    callback = handler if shape == "async" else lambda ctx, args: handler(ctx, args)
    loop = _loop(tmp_path, callback)
    loop.run_once("go")
    assert _outcomes(loop)[0]["error"]["code"] == "tool_handler_cancelled"
    with pytest.raises(ToolExecutionError, match="not in flight"):
        retained[0].exists("before.txt")


@pytest.mark.parametrize("boundary", ["cancel", "timeout"])
def test_abandoned_worker_cannot_write_after_call_end(tmp_path: Path, boundary: str) -> None:
    entered, proceed, finished = threading.Event(), threading.Event(), threading.Event()
    errors = []
    token = CancellationToken()

    def handler(ctx: Any, _args: Any) -> ToolResult:
        view = require_host_tool_binding(ctx, WS)
        entered.set()
        assert proceed.wait(5)
        try:
            view.write_bytes("late.txt", b"late")
        except ToolExecutionError as exc:
            errors.append(exc.error_code)
        finally:
            finished.set()
        return ToolResult(ok=True)

    loop = _loop(
        tmp_path,
        handler,
        spec=_spec(
            tmp_path,
            mode="apply",
            limits=RunLimits(max_duration_s=1 if boundary == "timeout" else None),
        ),
        cancellation_token=token,
        async_tool_cancel_grace_s=0,
    )
    results = []
    runner = threading.Thread(target=lambda: results.append(loop.run_once("go")))
    runner.start()
    try:
        assert entered.wait(3)
        if boundary == "cancel":
            token.cancel()
        runner.join(timeout=5)
        assert not runner.is_alive() and results
        assert results[0].error_code == ("cancelled" if boundary == "cancel" else "run_timeout")
    finally:
        proceed.set()
        runner.join(timeout=5)
    assert finished.wait(3) and errors == ["tool_call_not_in_flight"]
    assert not (loop.spec.workspace_root / "late.txt").exists()


@pytest.mark.parametrize("boundary", ["close", "revoke"])
def test_entered_local_write_finishes_before_lifetime_boundary_returns(
    tmp_path: Path, boundary: str
) -> None:
    entered, proceed, ended = threading.Event(), threading.Event(), threading.Event()
    authority = ActivationWriteAuthority()
    write_results, errors, boundaries = [], [], []

    class Workspace(LocalWorkspaceBackend):
        def write_bytes(self, path: str, data: bytes, **kwargs: Any) -> str:
            if path == "entered.txt":
                entered.set()
                assert proceed.wait(5)
            return super().write_bytes(path, data, **kwargs)

    threads = []

    def handler(ctx: Any, _args: Any) -> ToolResult:
        view = require_host_tool_binding(ctx, WS)

        def mutate() -> None:
            try:
                write_results.append(view.write_bytes("entered.txt", b"entered"))
            except NativeAgentError as exc:
                errors.append(exc.error_code)

        worker = threading.Thread(target=mutate)
        worker.start()
        threads.append(worker)
        assert entered.wait(3)

        def stop() -> None:
            if boundary == "revoke":
                authority.revoke()
            else:
                loop._end_host_tool_resources()
            boundaries.append("returned")
            ended.set()

        stopper = threading.Thread(target=stop)
        stopper.start()
        threads.append(stopper)
        # The production boundary publishes its Event before joining this entered mutation.
        event = (
            authority._revoke_requested
            if boundary == "revoke"
            else loop._host_tool_registry.lifetime.closed
        )
        assert event.wait(3) and not ended.is_set()
        proceed.set()
        worker.join(timeout=3)
        stopper.join(timeout=3)
        assert ended.is_set()
        with pytest.raises(NativeAgentError):
            view.write_bytes("after.txt", b"after")
        return ToolResult(ok=True)

    loop = _loop(
        tmp_path,
        handler,
        spec=_spec(tmp_path, mode="apply"),
        write_authority=authority,
        workspace_factory=lambda spec: Workspace(spec.workspace_root, mode=spec.mode),
    )
    try:
        if boundary == "revoke":
            with pytest.raises(WriteAuthorityRevoked):
                loop.run_once("go")
        else:
            loop.run_once("go")
    finally:
        proceed.set()
        for thread in threads:
            thread.join(timeout=3)
        loop.discard_uncommitted()
    assert boundaries == ["returned"] and not write_results and errors
    assert (loop.spec.workspace_root / "entered.txt").read_bytes() == b"entered"
    assert not (loop.spec.workspace_root / "after.txt").exists()


def test_external_guard_postcheck_and_original_control_survive_cleanup(tmp_path: Path) -> None:
    authority = ActivationWriteAuthority()
    primary = RunCancelled("control")

    def cleanup() -> None:
        raise ValueError("SECRET_CLOSE")

    resource = Resource(close=cleanup)
    key = HostToolKey("client", accepts=lambda _: True)

    def handler(ctx: Any, _args: Any) -> ToolResult:
        lease = require_host_tool_binding(ctx, key)
        lease.guard_external_call(authority.revoke)
        pytest.fail("revoked external result was published")

    loop = _loop(
        tmp_path,
        handler,
        bindings=(HostToolBinding(key, open=lambda _: resource),),
        write_authority=authority,
    )
    with pytest.raises(WriteAuthorityRevoked):
        loop.run_once("go")
    assert resource.closed == 1
    other = Resource(close=cleanup)

    def fail(_activation: Any) -> Any:
        raise primary

    failed = _loop(
        tmp_path / "control",
        bindings=(
            HostToolBinding(key, open=lambda _: other),
            HostToolBinding(HostToolKey("control", accepts=lambda _: True), open=fail),
        ),
    )
    with pytest.raises(RunCancelled) as caught:
        failed.open()
    assert caught.value is primary and other.closed == 1


@pytest.mark.parametrize("boundary", ["cancel", "timeout", "model_failure", "host_shutdown"])
@pytest.mark.parametrize("disposal", ["ok", "error", "revoke"])
def test_disposer_preserves_recorded_failure_and_existing_unwind_priority(
    tmp_path: Path, boundary: str, disposal: str
) -> None:
    authority = ActivationWriteAuthority()
    token = CancellationToken()

    def cleanup() -> None:
        if disposal == "revoke":
            authority.revoke()
        elif disposal == "error":
            raise ValueError("SECRET_CLOSE")

    def handler(_ctx: Any, _args: Any) -> ToolResult:
        if boundary == "timeout":
            raise RunTimeout("original timeout")
        if boundary == "host_shutdown":
            raise RunCancelled("original shutdown", interruption_cause=InterruptionCause.HOST_SHUTDOWN)
        return ToolResult(ok=True, content={"ok": True})

    class Model(FakeModelAdapter):
        def next_turn(self, request: Any) -> ModelTurn:
            if boundary == "model_failure":
                raise ValueError("original model failure")
            return super().next_turn(request)

    resource = Resource(close=cleanup)
    loop = AgentLoop.from_tools(
        _spec(tmp_path),
        Model(turns=[ModelTurn(tool_calls=(fake_tool_call("test_action", {}),))]),
        [_tool(handler)],
        cancellation_token=token,
        write_authority=authority,
        host_tool_bindings=(
            HostToolBinding(HostToolKey("client", accepts=lambda _: True), open=lambda _: resource),
        ),
    )
    if boundary == "cancel":
        token.cancel()
    if boundary == "host_shutdown":
        with pytest.raises(TurnNotSettled) as caught:
            loop.run_once("go")
        assert caught.value.suspension.interruption_cause is InterruptionCause.HOST_SHUTDOWN
    elif disposal == "revoke":
        with pytest.raises(WriteAuthorityRevoked):
            loop.run_once("go")
    else:
        result = loop.run_once("go")
        expected = {
            "cancel": ("limited", "cancelled"),
            "timeout": ("limited", "run_timeout"),
            "model_failure": ("failed", "model_error"),
        }[boundary]
        assert (result.status, result.error_code) == expected
    assert resource.closed == 1 and loop._session is None and loop._host_tool_registry is None
    loop.discard_uncommitted()
    assert resource.closed == 1


@pytest.mark.parametrize("stage", ["open", "close"])
def test_async_lifecycle_cancel_waits_then_disposes_once(tmp_path: Path, stage: str) -> None:
    entered, proceed = threading.Event(), threading.Event()

    def wait() -> None:
        entered.set()
        assert proceed.wait(5)

    resource = Resource(close=wait if stage == "close" else None)
    key = HostToolKey("client", accepts=lambda _: True)

    def factory(_activation: Any) -> Resource:
        if stage == "open":
            wait()
        return resource

    loop = _loop(tmp_path, bindings=(HostToolBinding(key, open=factory),))

    async def run() -> None:
        if stage == "close":
            await loop.aopen()
        task = asyncio.create_task(loop.aopen() if stage == "open" else loop.aclose())
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert resource.closed == 1 and loop._session is None and loop._host_tool_registry is None


def test_readonly_workspace_refuses_write_and_factory_descriptor_is_deferred(
    tmp_path: Path,
) -> None:
    descriptors = []

    def factory(activation: Any) -> Any:
        descriptors.append(activation)
        assert not hasattr(activation, "workspace") and not hasattr(activation, "root")
        resource = activation.workspace_resource()
        assert not hasattr(resource, "read_bytes")
        return resource

    def handler(ctx: Any, _args: Any) -> ToolResult:
        require_host_tool_binding(ctx, WS).write_bytes("refused.txt", b"no")
        return ToolResult(ok=True)

    loop = _loop(
        tmp_path,
        handler,
        spec=_spec(tmp_path, mode="read-only"),
        bindings=(HostToolBinding(WS, open=factory),),
    )
    loop.run_once("go")
    assert not _outcomes(loop)[0]["ok"]
    assert len(descriptors) == 1
    with pytest.raises(ToolExecutionError, match="closed"):
        descriptors[0].workspace_resource()


@pytest.mark.parametrize("operation", ["close", "release_parked"])
def test_disposer_revoke_preserves_lease_loss_after_all_local_disposals(
    tmp_path: Path, operation: str
) -> None:
    authority = ActivationWriteAuthority()
    first, second = Resource(), Resource(close=authority.revoke)
    bindings = tuple(
        HostToolBinding(HostToolKey(str(i), accepts=lambda _: True), open=lambda _, r=r: r)
        for i, r in enumerate((first, second))
    )
    loop = _loop(tmp_path, bindings=bindings, write_authority=authority)
    loop.open()
    loop.run_until_suspended("go")
    with pytest.raises(WriteAuthorityRevoked):
        getattr(loop, operation)()
    assert first.closed == second.closed == 1 and loop._session is None


def test_host_values_stay_out_of_checkpoint_schema_and_observations(tmp_path: Path) -> None:
    class Secret:
        def __repr__(self) -> str:
            pytest.fail("host resource was rendered")

    secret = Secret()
    resource = Resource(bind=lambda _: secret)
    key = HostToolKey("SECRET_KEY_DO_NOT_SERIALIZE", accepts=lambda value: value is secret)

    def handler(ctx: Any, _args: Any) -> ToolResult:
        assert require_host_tool_binding(ctx, key) is secret
        return ToolResult(ok=True)

    loop = _loop(tmp_path, handler, bindings=(HostToolBinding(key, open=lambda _: resource),))
    loop.open()
    try:
        loop.run_until_suspended("go")
        record = LocalFsCheckpointStore(loop.spec.run_root).latest(loop.spec.run_id)
        assert record is not None
        disk = "".join(p.read_text(encoding="utf-8") for p in loop.spec.run_root.rglob("*.json*"))
        assert key.name not in disk and "host_tool_bindings" not in disk
        assert all(result["ok"] for result in _outcomes(loop))
    finally:
        loop.discard_uncommitted()


def test_sync_closer_returning_awaitable_is_refused_once(tmp_path: Path) -> None:
    async def pending() -> None:
        pass

    resource = Resource(close=pending)
    key = HostToolKey("resource", accepts=lambda _: True)
    loop = _loop(tmp_path, bindings=(HostToolBinding(key, open=lambda _: resource),))
    loop.open()
    with pytest.raises(ToolExecutionError, match="synchronous"):
        loop.discard_uncommitted()
    loop.discard_uncommitted()
    assert resource.closed == 1 and loop._session is None
