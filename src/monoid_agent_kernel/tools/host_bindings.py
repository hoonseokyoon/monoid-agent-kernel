"""Typed host resources, owned by an activation and accessed through one admitted call.

Factories, binders and disposers are synchronous. Disposers release local resources;
external effects need host fencing. General client adapters must check their supplied lease
at every operation. ``ToolWorkspace`` performs these checks inside the kernel.
"""

from __future__ import annotations

import inspect
import logging
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import Future, TimeoutError as FutureTimeout
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, cast, runtime_checkable

from monoid_agent_kernel.core._sync_bridge import dispose_unawaited, is_async_callable
from monoid_agent_kernel.core.authority import ActivationWriteAuthority
from monoid_agent_kernel.core.spec import RunMode
from monoid_agent_kernel.core.tool_surface import ToolAuthorization
from monoid_agent_kernel.core.workspace import Workspace
from monoid_agent_kernel.errors import (
    AgentConfigError,
    PermissionDenied,
    RunCancelled,
    RunTimeout,
    ToolExecutionError,
    TurnInterrupted,
    TurnPaused,
)
from monoid_agent_kernel.permissions import PermissionPolicy, matches_path_patterns

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_LOGGER = logging.getLogger(__name__)
_CONTROLS = (RunCancelled, RunTimeout, TurnInterrupted, TurnPaused)


@dataclass(frozen=True, eq=False)
class HostToolKey(Generic[_T]):
    """Process-local identity key; share the same instance between host and handlers."""

    name: str
    accepts: Callable[[object], bool] = field(repr=False, kw_only=True)


class HostToolResource(Protocol[_T_co]):
    """Activation-owned adapter. ``close`` performs local disposal, at most once."""

    def bind_call(self, lease: HostToolCallLease) -> _T_co: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class HostToolBinding(Generic[_T]):
    key: HostToolKey[_T]
    open: Callable[[HostToolActivation], HostToolResource[_T]] = field(repr=False, kw_only=True)
    inherit_to_children: bool = field(default=False, kw_only=True)


@runtime_checkable
class ToolWorkspace(Protocol):
    """Call-scoped workspace operations using the existing backend's modes and hashes."""

    def read_bytes(self, path: str, *, max_bytes: int | None = None) -> tuple[bytes, str]: ...

    def exists(self, path: str) -> bool: ...

    def write_bytes(
        self,
        path: str,
        data: bytes,
        *,
        create_dirs: bool = False,
        expected_sha256: str | None = None,
        overwrite: bool = True,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class HostToolActivation:
    """Factory metadata and a deferred workspace adapter, available before rehydration."""

    run_id: str
    root_run_id: str
    parent_run_id: str | None
    mode: RunMode
    _workspace_resource: Callable[[], HostToolResource[ToolWorkspace]] = field(repr=False)

    def workspace_resource(self) -> HostToolResource[ToolWorkspace]:
        return self._workspace_resource()


@dataclass(frozen=True, slots=True)
class HostToolCallLease:
    """Read-only identity and authority for the original admitted invocation."""

    _call: _HostToolCall = field(repr=False)

    @property
    def run_id(self) -> str:
        return self._call.registry.activation.run_id

    @property
    def tool_call_id(self) -> str:
        return self._call.call_id

    @property
    def binding_id(self) -> str:
        return self._call.authorization.binding_id

    @property
    def tool_id(self) -> str:
        return self._call.authorization.tool_id

    def assert_active(self) -> None:
        self._call.assert_active()

    def guard_external_call(self, operation: Callable[[], _T]) -> _T:
        """Check before and after a synchronous callback without holding authority locks."""
        self.assert_active()
        try:
            return operation()
        finally:
            self.assert_active()


@runtime_checkable
class _BindingContext(Protocol):
    def _require_host_tool_binding(self, key: HostToolKey[_T]) -> _T: ...


def require_host_tool_binding(ctx: object, key: HostToolKey[_T]) -> _T:
    """Resolve a typed view from the supplied call facade, including on explicit threads."""
    if not isinstance(ctx, _BindingContext):
        raise ToolExecutionError(
            "context does not support host tool bindings",
            error_code="host_tool_context_unavailable",
        )
    return ctx._require_host_tool_binding(key)


def _safe_callback(operation: Callable[[], _T], stage: str, *, config: bool = False) -> _T:
    # Raise outside the except suite: even the safe exception's implicit context must not retain
    # the host's original exception or secret-bearing traceback in a cached binding failure.
    failed = False
    try:
        value = operation()
    except _CONTROLS:
        raise
    except Exception:
        failed = True
    else:
        if not inspect.isawaitable(value):
            return value
        try:
            dispose_unawaited(value)
        except _CONTROLS:
            raise
        except Exception:
            pass  # A broken awaitable disposer still receives the same safe refusal below.
    error_type = AgentConfigError if config else ToolExecutionError
    detail = "failed" if failed else "must return a synchronous value"
    raise error_type(
        f"host tool resource {stage} {detail}", error_code=f"host_tool_binding_{stage}"
    )


def _validate_host_tool_bindings(
    bindings: Iterable[HostToolBinding[Any]],
) -> tuple[HostToolBinding[Any], ...]:
    failed = False
    try:
        snapshot = tuple(bindings)
    except Exception:
        failed = True
    if failed:
        raise AgentConfigError("host_tool_bindings must be an iterable of HostToolBinding values")
    keys: set[HostToolKey[Any]] = set()
    names: set[str] = set()
    for binding in snapshot:
        if not isinstance(binding, HostToolBinding) or not isinstance(binding.key, HostToolKey):
            raise AgentConfigError("host_tool_bindings must contain HostToolBinding values")
        if type(binding.inherit_to_children) is not bool:
            raise AgentConfigError("inherit_to_children must be a boolean")
        key = binding.key
        if type(key.name) is not str or not key.name.strip():
            raise AgentConfigError("host tool key name must be a non-empty string")
        if key in keys or key.name in names:
            raise AgentConfigError("host tool binding keys and names must be unique")
        if any(not callable(fn) or is_async_callable(fn) for fn in (key.accepts, binding.open)):
            raise AgentConfigError("host tool open and accepts must be synchronous callables")
        keys.add(key)
        names.add(key.name)
    return snapshot


class _ResourceLifetime:
    def __init__(
        self, writer: ActivationWriteAuthority, parent: _ResourceLifetime | None = None
    ) -> None:
        self.writer = writer
        self.parent = parent
        self.lock = threading.RLock()
        self.closed = threading.Event()

    def assert_active(self) -> None:
        if self.parent is not None:
            self.parent.assert_active()
        self.assert_local_active()

    def assert_local_active(self) -> None:
        if self.closed.is_set():
            raise ToolExecutionError(
                "host tool resources are closed", error_code="host_tool_resource_closed"
            )
        self.writer.assert_active()

    def end(self) -> None:
        self.closed.set()
        with self.lock:
            pass  # Join only an entered local workspace operation; arbitrary callbacks hold no lock.

    def guard_local(self, operation: Callable[[], _T]) -> _T:
        # Every descendant enters ancestor activation/writer guards before its own guards.
        # An ancestor's sticky end/revoke request suppresses the result of an entered write.
        if self.parent is not None:
            return self.parent.guard_local(lambda: self._guard_local(operation))
        return self._guard_local(operation)

    def _guard_local(self, operation: Callable[[], _T]) -> _T:
        with self.lock:
            self.assert_local_active()
            try:
                return self.writer.guard_local_mutation(operation)
            finally:
                self.assert_local_active()

    def guard_external(self, operation: Callable[[], _T]) -> _T:
        self.assert_active()
        try:
            return operation()
        finally:
            self.assert_active()


class _HostToolCall:
    def __init__(
        self,
        registry: _HostToolRegistry,
        call_id: str,
        authorization: ToolAuthorization,
        side_effect: str,
    ) -> None:
        self.registry = registry
        self.call_id = call_id
        self.authorization = authorization
        self.side_effect = side_effect
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.views: dict[HostToolKey[Any], tuple[int, Future[Any]]] = {}
        self.lease = HostToolCallLease(self)

    def assert_active(self) -> None:
        if self.closed.is_set():
            raise ToolExecutionError(
                "tool call is not in flight", error_code="tool_call_not_in_flight"
            )
        self.registry.lifetime.assert_active()

    def end(self) -> None:
        self.closed.set()
        with self.lock:
            self.views.clear()

    def guard_local(self, operation: Callable[[], _T]) -> _T:
        with self.lock:
            self.assert_active()
            try:
                return self.registry.lifetime.guard_local(operation)
            finally:
                self.assert_active()

    def require(self, key: HostToolKey[_T]) -> _T:
        with self.lock:
            self.assert_active()
            existing = self.views.get(key)
            if existing is None:
                future: Future[Any] = Future()
                self.views[key] = (threading.get_ident(), future)
            else:
                owner, future = existing
                if owner == threading.get_ident() and not future.done():
                    raise ToolExecutionError(
                        "recursive host tool binding", error_code="host_tool_binding_bind"
                    )
        if existing is None:
            try:
                value = self.registry.bind(key, self.lease)
                self.assert_active()
                future.set_result(value)
            except BaseException as exc:
                future.set_exception(exc)
        while True:
            self.assert_active()
            try:
                value = future.result(timeout=0.05)
            except FutureTimeout:
                if future.done():
                    raise  # A callback-raised TimeoutError is a result, not a wait timeout.
            else:
                self.assert_active()
                return cast(_T, value)


class _HostToolRegistry:
    def __init__(
        self,
        *,
        run_id: str,
        root_run_id: str,
        parent_run_id: str | None,
        mode: RunMode,
        workspace: Workspace,
        policy: PermissionPolicy,
        writer: ActivationWriteAuthority,
        parent: _ResourceLifetime | None = None,
    ) -> None:
        self.lifetime = _ResourceLifetime(writer, parent)
        self.policy = policy
        self.workspace: Workspace | None = workspace
        self.resources: dict[HostToolKey[Any], HostToolResource[Any]] = {}
        self.stack = ExitStack()
        self.close_lock = threading.Lock()
        self.disposed = False
        self.close_errors: list[BaseException] = []
        self.activation = HostToolActivation(
            run_id, root_run_id, parent_run_id, mode, self.workspace_resource
        )

    def workspace_resource(self) -> HostToolResource[ToolWorkspace]:
        self.lifetime.assert_active()
        return _WorkspaceResource(self)

    def open(self, bindings: Iterable[HostToolBinding[Any]]) -> None:
        # Keep identities through a factory that returns after concurrent/reentrant close.
        # References belong to this open attempt and are released when it unwinds.
        acquired: dict[int, HostToolResource[Any]] = {}
        try:
            for binding in bindings:

                def acquire(binding: HostToolBinding[Any] = binding) -> HostToolResource[Any]:
                    resource = _safe_callback(
                        lambda: binding.open(self.activation), "open", config=True
                    )
                    closer = getattr(resource, "close", None)
                    if not callable(closer):
                        raise AgentConfigError(
                            "host tool resource must provide a local close callable"
                        )
                    with self.close_lock:
                        if id(resource) in acquired:
                            raise AgentConfigError(
                                "host tool factories must return distinct resource adapters"
                            )
                        acquired[id(resource)] = resource
                        late = self.disposed
                        if not late:
                            self.stack.callback(self._dispose, closer)
                    if late:
                        _safe_callback(closer, "close")
                        self.lifetime.assert_active()
                    binder = getattr(resource, "bind_call", None)
                    if not callable(binder) or is_async_callable(binder) or is_async_callable(closer):
                        raise AgentConfigError(
                            "host tool resource bind_call and close must be synchronous"
                        )
                    return resource

                self.lifetime.assert_active()
                resource = self.lifetime.guard_external(
                    lambda: _safe_callback(acquire, "open", config=True)
                )
                with self.close_lock:
                    self.lifetime.assert_active()
                    self.resources[binding.key] = resource
        finally:
            acquired.clear()

    def bind(self, key: HostToolKey[_T], lease: HostToolCallLease) -> _T:
        lease.assert_active()
        resource = self.resources.get(key)
        if resource is None:
            raise ToolExecutionError(
                "host tool binding is missing", error_code="host_tool_binding_missing"
            )
        view = lease.guard_external_call(
            lambda: _safe_callback(lambda: resource.bind_call(lease), "bind")
        )
        accepted = lease.guard_external_call(
            lambda: _safe_callback(lambda: key.accepts(view), "accepts")
        )
        if accepted is not True:
            raise ToolExecutionError(
                "host tool binding view has the wrong type", error_code="host_tool_binding_type"
            )
        return cast(_T, view)

    def close(self) -> None:
        self.lifetime.end()
        with self.close_lock:
            if self.disposed:
                return
            self.disposed = True
            # Pop ownership before callbacks: reentrant or repeated cleanup cannot call one twice.
            pending = self.stack.pop_all()
            self.resources.clear()
            self.workspace = None
        pending.close()
        errors, self.close_errors = self.close_errors, []
        if errors:
            raise errors[0]

    def _dispose(self, closer: Callable[[], Any]) -> None:
        # ExitStack pops before invoking. Trap each error so later cleanup cannot replace the first.
        try:
            _safe_callback(closer, "close")
        except BaseException as exc:
            if self.close_errors:
                _LOGGER.warning("additional host tool resource close failed")
            self.close_errors.append(exc)


class _WorkspaceResource:
    def __init__(self, registry: _HostToolRegistry) -> None:
        self.registry = registry

    def bind_call(self, lease: HostToolCallLease) -> ToolWorkspace:
        if lease._call.registry is not self.registry:
            raise AgentConfigError("workspace resource belongs to another activation")
        return _ToolWorkspaceView(self.registry, lease)

    def close(self) -> None:
        pass  # The registry drops the borrowed backend; it has no external disposal ownership.


@dataclass(frozen=True, slots=True)
class _ToolWorkspaceView:
    _registry: _HostToolRegistry
    _lease: HostToolCallLease

    def _operate(self, path: str, operation: str, callback: Callable[[Workspace, str], _T]) -> _T:
        def apply() -> _T:
            call = self._lease._call
            workspace = self._registry.workspace
            assert workspace is not None  # The activation guard precedes every operation.
            normalized = workspace.normalize(path)
            if operation == "write" and call.side_effect != "write":
                raise PermissionDenied("tool was not admitted for workspace writes")
            for scope in (call.authorization.scope, call.authorization.surface_scope):
                if (
                    scope.allowed_paths
                    and not matches_path_patterns(normalized, scope.allowed_paths)
                ) or matches_path_patterns(normalized, scope.denied_paths):
                    raise PermissionDenied(
                        "workspace path is outside tool scope", error_code="tool_scope_denied"
                    )
            self._registry.policy.check_paths(operation, (normalized,))  # type: ignore[arg-type]
            return callback(workspace, normalized)

        return self._lease._call.guard_local(apply)

    def read_bytes(self, path: str, *, max_bytes: int | None = None) -> tuple[bytes, str]:
        return self._operate(path, "read", lambda ws, rel: ws.read_bytes(rel, max_bytes=max_bytes))

    def exists(self, path: str) -> bool:
        return self._operate(path, "read", lambda ws, rel: ws.exists(rel))

    def write_bytes(
        self,
        path: str,
        data: bytes,
        *,
        create_dirs: bool = False,
        expected_sha256: str | None = None,
        overwrite: bool = True,
    ) -> str:
        return self._operate(
            path,
            "write",
            lambda ws, rel: ws.write_bytes(
                rel,
                data,
                create_dirs=create_dirs,
                expected_sha256=expected_sha256,
                overwrite=overwrite,
            ),
        )


__all__ = [
    "HostToolActivation",
    "HostToolBinding",
    "HostToolCallLease",
    "HostToolKey",
    "HostToolResource",
    "ToolWorkspace",
    "require_host_tool_binding",
]
