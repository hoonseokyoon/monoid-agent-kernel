from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from support.runtime import runtime_config, tool_binding
from monoid_agent_kernel.core.agents import SubagentDefinition
from monoid_agent_kernel.core.authority import WriteAuthorityRevoked
from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore
from monoid_agent_kernel.core.spec import AgentRunSpec, RunLimits
from monoid_agent_kernel.core.tool_surface import ToolScope
from monoid_agent_kernel.errors import AgentConfigError, NativeAgentError, TurnNotSettled
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.providers.fake import FakeModelAdapter, ModelTurn, fake_tool_call
from monoid_agent_kernel.reference.host_tool_report import run_child_reports
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
FILES = HostToolKey[ToolWorkspace]("child.files", accepts=lambda v: isinstance(v, ToolWorkspace))


class Resource:
    def __init__(self, activation: Any) -> None:
        self.activation = activation
        self.inner = activation.workspace_resource()
        self.leases: list[Any] = []
        self.closed = 0
        self.disposed = threading.Event()

    def bind_call(self, lease: Any) -> ToolWorkspace:
        self.leases.append(lease)
        return self.inner.bind_call(lease)

    def close(self) -> None:
        self.closed += 1
        self.inner.close()
        self.disposed.set()


def _call(name: str = "test_action", call_id: str = "action", **args: Any) -> ModelTurn:
    return ModelTurn(tool_calls=(fake_tool_call(name, args, call_id),))


def _spawn(call_id: str = "spawn", **args: Any) -> ModelTurn:
    return _call("agent_spawn", call_id, subagent_type="worker", prompt=call_id, **args)


def _loop(
    root: Path,
    turns: list[ModelTurn],
    handler: Any,
    *,
    bindings: tuple = (),
    context: str = "fresh",
    spec: AgentRunSpec | None = None,
    scope: ToolScope | None = None,
    **kwargs: Any,
) -> AgentLoop:
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    action = ToolSpec("test.action", "action", {"type": "object"}, "", "write", handler)
    return AgentLoop.from_config(
        spec
        or AgentRunSpec(
            workspace_root=workspace,
            run_root=root / "runs",
            limits=RunLimits(max_subagent_depth=3),
        ),
        FakeModelAdapter(turns=list(turns)),
        runtime_config(
            bindings=(
                tool_binding("agent.spawn"),
                tool_binding("test.action", scope=scope),
                tool_binding("fs.write"),
            )
        ),
        subagent_definitions={"worker": SubagentDefinition(context=context)},
        tool_providers=(FunctionToolProvider(lambda _ctx: (action,)),),
        host_tool_bindings=bindings,
        **kwargs,
    )


def _binding(resources: list[Resource], *, inherit: bool = True) -> HostToolBinding:
    def factory(activation: Any) -> Resource:
        resource = Resource(activation)
        resources.append(resource)
        return resource

    if inherit:
        return HostToolBinding(FILES, open=factory, inherit_to_children=True)
    return HostToolBinding(FILES, open=factory)


@pytest.mark.parametrize("context", ["fresh", "fork"])
@pytest.mark.parametrize("inherit", [False, True])
def test_child_bindings_default_off_and_explicit_opt_in(
    tmp_path: Path, context: str, inherit: bool
) -> None:
    resources, outcomes = [], []

    def handler(ctx: Any, _args: Any) -> ToolResult:
        try:
            view = require_host_tool_binding(ctx, FILES)
            outcomes.append(view.exists("report.txt"))
        except NativeAgentError as exc:
            outcomes.append(exc.error_code)
        return ToolResult(ok=True)

    loop = _loop(
        tmp_path,
        [_spawn(), _call(), ModelTurn(final_text="child"), ModelTurn(final_text="parent")],
        handler,
        bindings=(_binding(resources, inherit=inherit),),
        context=context,
    )
    assert loop.run_once("go").status == "completed"
    assert outcomes == ([False] if inherit else ["host_tool_binding_missing"])
    assert len(resources) == (2 if inherit else 1)
    assert all(resource.closed == 1 for resource in resources)
    if inherit:
        parent, child = (r.activation for r in resources)
        assert child.parent_run_id == parent.run_id == child.root_run_id
        assert child.run_id != parent.run_id
        assert resources[0] is not resources[1]


@pytest.mark.parametrize("context", ["fresh", "fork"])
@pytest.mark.parametrize("policy_source", ["spec", "loop", "override"])
def test_child_inherits_effective_operator_policy(
    tmp_path: Path, context: str, policy_source: str
) -> None:
    denied = PermissionPolicy(deny_patterns=("reports/secret.txt",))
    spec_policy = denied if policy_source == "spec" else PermissionPolicy()
    if policy_source == "override":
        spec_policy = PermissionPolicy(deny_patterns=("reports/spec-only.txt",))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "reports").mkdir()
    spec = AgentRunSpec(
        workspace_root=workspace, run_root=tmp_path / "runs", permission_policy=spec_policy
    )
    loop = _loop(
        tmp_path,
        [
            _call("fs_write", "parent-write", path="reports/secret.txt", content="parent"),
            _spawn(),
            _call("fs_write", "child-write", path="reports/secret.txt", content="child"),
            ModelTurn(final_text="child"),
            ModelTurn(final_text="parent"),
        ],
        lambda _ctx, _args: ToolResult(ok=True),
        context=context,
        spec=spec,
        **({"permission_policy": denied} if policy_source != "spec" else {}),
    )
    assert loop.run_once("go").status == "completed"
    observations = {
        obs.call_id: obs.output
        for request in loop.model_adapter.requests
        for obs in request.observations
    }
    for call_id in ("parent-write", "child-write"):
        assert observations[call_id].get("error", {}).get("code") == "permission_denied"


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_inheritance_flag_validation_precedes_factory(tmp_path: Path, value: Any) -> None:
    opened = []
    with pytest.raises(AgentConfigError, match="inherit_to_children must be a boolean"):
        _loop(
            tmp_path,
            [],
            None,
            bindings=(
                HostToolBinding(FILES, open=lambda a: opened.append(a), inherit_to_children=value),
            ),
        )
    assert opened == [] and not (tmp_path / "runs").exists()


@pytest.mark.parametrize("context", ["fresh", "fork"])
def test_inherited_workspace_enforces_computed_scope_and_operator_policy(
    tmp_path: Path, context: str
) -> None:
    resources, checked = [], []

    def handler(ctx: Any, _args: Any) -> ToolResult:
        view = require_host_tool_binding(ctx, FILES)
        for path, code in (
            ("outside.txt", "tool_scope_denied"),
            ("reports/secret.txt", "permission_denied"),
        ):
            with pytest.raises(NativeAgentError) as caught:
                view.write_bytes(path, b"refused", create_dirs=True)
            checked.append(caught.value.error_code)
            assert caught.value.error_code == code
        digest = view.write_bytes("reports/allowed.txt", b"child", create_dirs=True)
        assert view.read_bytes("reports/allowed.txt") == (b"child", digest)
        return ToolResult(ok=True)

    loop = _loop(
        tmp_path,
        [_spawn(), _call(), ModelTurn(final_text="child"), ModelTurn(final_text="parent")],
        handler,
        bindings=(_binding(resources),),
        context=context,
        scope=ToolScope(allowed_paths=("reports/**",)),
        permission_policy=PermissionPolicy(deny_patterns=("reports/secret.txt",)),
    )
    assert loop.run_once("go").status == "completed"
    assert checked == ["tool_scope_denied", "permission_denied"]
    assert not (loop.spec.workspace_root / "reports/allowed.txt").exists()
    assert len(resources) == 2 and all(r.closed == 1 for r in resources)


@pytest.mark.parametrize("context", ["fresh", "fork"])
def test_child_cleanup_preserves_parent_and_sibling_resources(tmp_path: Path, context: str) -> None:
    resources, seen, retained = [], [], []

    def handler(ctx: Any, args: Any) -> ToolResult:
        role = args["role"]
        view = require_host_tool_binding(ctx, FILES)
        seen.append(role)
        assert resources[0].closed == 0
        if role == "root-before":
            view.write_bytes("shared.txt", b"parent")
            retained.append(view)
        elif role.startswith("child"):
            assert not view.exists("shared.txt")
            view.write_bytes("shared.txt", role.encode())
            if role == "child-two":
                assert resources[1].closed == 1 and resources[2].closed == 0
        else:
            assert view.read_bytes("shared.txt")[0] == b"parent"
            assert all(r.closed == 1 for r in resources[1:])
            with pytest.raises(NativeAgentError) as caught:
                retained[0].exists("shared.txt")
            assert caught.value.error_code == "tool_call_not_in_flight"
        assert len(loop.write_authority._callbacks) == 1
        return ToolResult(ok=True)

    loop = _loop(
        tmp_path,
        [
            _call(role="root-before"),
            _spawn("first"),
            _call(role="child-one"),
            ModelTurn(final_text="first child"),
            _call(role="root-between"),
            _spawn("second"),
            _call(role="child-two"),
            ModelTurn(final_text="second child"),
            _call(role="root-after"),
            ModelTurn(final_text="parent"),
        ],
        handler,
        bindings=(_binding(resources),),
        context=context,
    )
    assert loop.run_once("go").status == "completed"
    assert seen == ["root-before", "child-one", "root-between", "child-two", "root-after"]
    assert len({r.activation.run_id for r in resources}) == 3
    assert all(r.closed == 1 for r in resources)
    for resource in resources:
        writer = resource.leases[0]._call.registry.lifetime.writer
        assert not writer.revoked and not writer._callbacks


@pytest.mark.parametrize("context", ["fresh", "fork"])
@pytest.mark.parametrize("ancestor", [0, 1])
@pytest.mark.parametrize("boundary", ["end", "revoke"])
def test_nested_active_call_checks_every_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, context: str, ancestor: int, boundary: str
) -> None:
    loops, resources, denied, plain, external = [], [], [], [], []
    original = AgentLoop.__post_init__

    def capture(loop: AgentLoop) -> None:
        original(loop)
        loops.append(loop)

    monkeypatch.setattr(AgentLoop, "__post_init__", capture)

    async def handler(ctx: Any, args: Any) -> ToolResult:
        if args.get("plain"):
            plain.append("ran")
            return ToolResult(ok=True)
        view = require_host_tool_binding(ctx, FILES)
        lease = resources[-1].leases[-1]
        await asyncio.sleep(0)
        view.write_bytes("before.txt", b"before")
        parent = loops[ancestor]
        if boundary == "revoke":
            parent.lose_writer_authority()
        else:
            parent._end_host_tool_resources()
        # No await after retirement: the same native handler is still admitted for every check.
        assert not lease._call.closed.is_set()
        for operation in (
            lambda: view.exists("before.txt"),
            lambda: view.read_bytes("before.txt"),
            lambda: view.write_bytes("after.txt", b"after"),
            lambda: require_host_tool_binding(ctx, FILES),
            lambda: lease.guard_external_call(lambda: external.append("called")),
        ):
            with pytest.raises(NativeAgentError) as caught:
                operation()
            denied.append(caught.value.error_code)
        assert not lease._call.closed.is_set()
        return ToolResult(ok=True)

    loop = _loop(
        tmp_path,
        [
            _spawn("child"),
            _spawn("grandchild"),
            _call(),
            _call(plain=True),
            ModelTurn(final_text="grandchild"),
            ModelTurn(final_text="child"),
            ModelTurn(final_text="parent"),
        ],
        handler,
        bindings=(_binding(resources),),
        context=context,
    )
    try:
        loop.run_once("go")
    except TurnNotSettled as exc:
        assert boundary == "revoke" and ancestor == 1
        assert exc.interruption_cause == "lease_lost"
    except WriteAuthorityRevoked:
        assert boundary == "revoke"
    finally:
        loop.discard_uncommitted()
    assert len(loops) == len(resources) == 3
    assert denied == ["lease_lost" if boundary == "revoke" else "host_tool_resource_closed"] * 5
    assert external == []
    if boundary == "end":
        assert plain == ["ran"]  # Parent resource lifetime does not change general tool dispatch.
    assert all(r.disposed.wait(3) and r.closed == 1 for r in resources)
    assert resources[2].activation.parent_run_id == resources[1].activation.run_id
    assert resources[2].activation.root_run_id == resources[0].activation.run_id


@pytest.mark.parametrize("boundary", ["end", "revoke"])
def test_parent_retirement_during_child_factory_disposes_returned_adapter(
    tmp_path: Path, boundary: str
) -> None:
    resources, calls = [], []

    def factory(activation: Any) -> Resource:
        resource = Resource(activation)
        resources.append(resource)
        if activation.parent_run_id:
            if boundary == "revoke":
                loop.lose_writer_authority()
            else:
                loop._end_host_tool_resources()
        return resource

    loop = _loop(
        tmp_path,
        [_spawn(), ModelTurn(final_text="parent")],
        lambda _ctx, _args: calls.append("handler"),
        bindings=(HostToolBinding(FILES, open=factory, inherit_to_children=True),),
    )
    try:
        loop.run_once("go")
    except WriteAuthorityRevoked:
        pass
    finally:
        loop.discard_uncommitted()
    assert calls == [] and len(resources) == 2
    assert all(r.closed == 1 for r in resources)


@pytest.mark.parametrize("boundary", ["end", "revoke"])
def test_parent_boundary_joins_entered_child_write_without_publishing_success(
    tmp_path: Path, boundary: str
) -> None:
    entered, proceed, ended = threading.Event(), threading.Event(), threading.Event()
    resources, errors, successes, failures = [], [], [], []

    class Workspace(LocalWorkspaceBackend):
        def write_bytes(self, path: str, data: bytes, **kwargs: Any) -> str:
            if path == "entered.txt":
                entered.set()
                assert proceed.wait(5)
            return super().write_bytes(path, data, **kwargs)

    def handler(ctx: Any, _args: Any) -> ToolResult:
        view = require_host_tool_binding(ctx, FILES)
        for path in ("entered.txt", "after.txt"):
            try:
                successes.append(view.write_bytes(path, b"child"))
            except NativeAgentError as exc:
                errors.append(exc.error_code)
        return ToolResult(ok=True)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    loop = _loop(
        tmp_path,
        [_spawn(), _call(), ModelTurn(final_text="child"), ModelTurn(final_text="parent")],
        handler,
        bindings=(_binding(resources),),
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs", mode="apply"),
        workspace_factory=lambda spec: Workspace(spec.workspace_root, mode=spec.mode),
    )

    def run() -> None:
        try:
            loop.run_once("go")
        except WriteAuthorityRevoked:
            pass
        except BaseException as exc:
            failures.append(exc)

    def stop() -> None:
        if boundary == "revoke":
            loop.lose_writer_authority()
        else:
            loop._end_host_tool_resources()
        ended.set()

    runner, stopper = threading.Thread(target=run), threading.Thread(target=stop)
    runner.start()
    try:
        assert entered.wait(4)
        event = (
            loop.write_authority._revoke_requested
            if boundary == "revoke"
            else loop._host_tool_registry.lifetime.closed
        )
        stopper.start()
        assert event.wait(3) and not ended.is_set()
    finally:
        proceed.set()
        runner.join(timeout=5)
        if stopper.ident is not None:
            stopper.join(timeout=5)
        loop.discard_uncommitted()
    assert not runner.is_alive() and not stopper.is_alive() and ended.is_set()
    assert not successes and not failures and len(errors) == 2
    assert (workspace / "entered.txt").read_bytes() == b"child"
    assert not (workspace / "after.txt").exists()
    assert all(r.closed == 1 for r in resources)


@pytest.mark.parametrize("context", ["fresh", "fork"])
def test_restored_parent_opens_fresh_child_lineage(tmp_path: Path, context: str) -> None:
    resources, views = [], []

    def handler(ctx: Any, _args: Any) -> ToolResult:
        view = require_host_tool_binding(ctx, FILES)
        views.append(view)
        view.write_bytes("child.txt", b"fresh")
        return ToolResult(ok=True)

    turns = [_spawn(), _call(), ModelTurn(final_text="child"), ModelTurn(final_text="parent")]
    bindings = (_binding(resources),)
    source = _loop(tmp_path, turns, handler, bindings=bindings, context=context)
    source.open()
    assert source.run_until_suspended("first").reason == "settled"
    old_lifetime = source._host_tool_registry.lifetime
    record = LocalFsCheckpointStore(source.spec.run_root).latest(source.spec.run_id)
    assert record is not None
    source.release_parked()
    assert all(r.closed == 1 for r in resources)
    restored = _loop(tmp_path, turns, handler, bindings=bindings, context=context, spec=source.spec)
    restored.restore(record.checkpoint, blobs=record.blob)
    try:
        new_lifetime = restored._host_tool_registry.lifetime
        assert new_lifetime is not old_lifetime
        old_lifetime.end()
        assert restored.run_until_suspended("again").reason == "settled"
        assert resources[-1].leases[-1]._call.registry.lifetime.parent is new_lifetime
        restored.close()
    finally:
        restored.discard_uncommitted()
        source.discard_uncommitted()
    assert len(resources) == 4 and all(r.closed == 1 for r in resources)
    assert len({id(r) for r in resources}) == 4 and len(views) == 2
    for view in views:
        with pytest.raises(NativeAgentError):
            view.write_bytes("late.txt", b"late")


def test_committed_parent_release_blocks_a_still_active_background_child(tmp_path: Path) -> None:
    entered, proceed, checked = threading.Event(), threading.Event(), threading.Event()
    resources, denied = [], []

    async def handler(ctx: Any, _args: Any) -> ToolResult:
        view = require_host_tool_binding(ctx, FILES)
        lease = resources[-1].leases[-1]
        entered.set()
        assert await asyncio.to_thread(proceed.wait, 5)
        assert not lease._call.closed.is_set()
        for operation in (
            lambda: view.exists("report.txt"),
            lambda: require_host_tool_binding(ctx, FILES),
        ):
            with pytest.raises(NativeAgentError) as caught:
                operation()
            denied.append(caught.value.error_code)
        checked.set()
        return ToolResult(ok=True)

    class Adapter:
        def next_turn(self, request: Any) -> ModelTurn:
            if request.instruction == "child":
                return _call()
            return ModelTurn(final_text="ready")

    loop = _loop(tmp_path, [], handler, bindings=(_binding(resources),))
    loop.model_adapter = Adapter()
    loop.open()
    manager = loop._session.res.context._job_manager
    try:
        # A host-driven child with no automatic reentry permits a durable parent park.
        # Model-spawned background resume jobs keep snapshot/release unavailable while running.
        loop.create_task(
            "subagent",
            {
                "definition_id": "worker",
                "prompt": "child",
                "depth": 0,
                "background": True,
                "resume_on_exit": False,
            },
        )
        assert entered.wait(3)
        assert loop.run_until_suspended("go").reason == "settled"
        future = next(job.runtime_future for job in manager.jobs.values() if job.kind == "subagent")
        loop.release_parked()
        assert not loop.write_authority.revoked
        assert resources[0].closed == 1 and resources[1].closed == 0
        proceed.set()
        assert checked.wait(3)
        future.result(timeout=5)
    finally:
        proceed.set()
        loop.discard_uncommitted()
        manager._shutdown_task_loop()
    assert denied == ["host_tool_resource_closed"] * 2
    assert all(r.disposed.wait(3) and r.closed == 1 for r in resources)
    assert LocalFsCheckpointStore(loop.spec.run_root).latest(loop.spec.run_id) is not None


def test_only_opted_in_keys_reach_each_hop(tmp_path: Path) -> None:
    resources, observed = [], []
    private = HostToolKey("parent.private", accepts=lambda view: isinstance(view, ToolWorkspace))

    def parent_only(activation: Any) -> Resource:
        resource = Resource(activation)
        resources.append(resource)
        return resource

    def handler(ctx: Any, _args: Any) -> ToolResult:
        assert not require_host_tool_binding(ctx, FILES).exists("report.txt")
        with pytest.raises(NativeAgentError) as caught:
            require_host_tool_binding(ctx, private)
        observed.append(caught.value.error_code)
        return ToolResult(ok=True)

    loop = _loop(
        tmp_path,
        [
            _spawn("child"),
            _spawn("grandchild"),
            _call(),
            ModelTurn(final_text="grandchild"),
            _call(),
            ModelTurn(final_text="child"),
            ModelTurn(final_text="parent"),
        ],
        handler,
        bindings=(_binding(resources), HostToolBinding(private, open=parent_only)),
    )
    assert loop.run_once("go").status == "completed"
    assert observed == ["host_tool_binding_missing"] * 2
    assert len(resources) == 4 and all(r.closed == 1 for r in resources)


@pytest.mark.parametrize("context", ["fresh", "fork"])
def test_packaged_child_report_consumer(tmp_path: Path, context: str) -> None:
    assert run_child_reports(tmp_path, context=context) == {
        "status": "completed",
        "context": context,
        "activations": 2,
        "proposal_only": True,
    }
