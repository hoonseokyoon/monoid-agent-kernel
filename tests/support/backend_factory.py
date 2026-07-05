from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.backend.service import RunnerBackend


_CURRENT_FACTORY: ManagedBackendFactory | None = None


def set_current_backend_factory(factory: ManagedBackendFactory | None) -> None:
    global _CURRENT_FACTORY
    _CURRENT_FACTORY = factory


def current_backend_factory() -> ManagedBackendFactory | None:
    return _CURRENT_FACTORY


class ManagedBackendFactory:
    """Create RunnerBackend instances with per-instance future tracking.

    Tests use this instead of a process-wide RunnerBackend monkeypatch. The factory owns
    every backend it creates and enforces a clean drain at teardown.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self._backends: list[RunnerBackend] = []
        self._futures: dict[int, list[Any]] = {}
        self._counter = 0

    def workspace(self, name: str = "workspace") -> Path:
        workspace = self.tmp_path / name
        workspace.mkdir(parents=True, exist_ok=True)
        notes = workspace / "notes.md"
        if not notes.exists():
            notes.write_text("notes\n", encoding="utf-8")
        return workspace

    def token_manager(self) -> TokenManager:
        return TokenManager.from_secret("x" * 32)

    def create(
        self,
        *,
        run_root: Path | None = None,
        workspace: Path | None = None,
        token_manager: TokenManager | None = None,
        turns: Iterable[ModelTurn] | None = None,
        adapters: list[Any] | None = None,
        model_adapter_factory: Callable[..., Any] | None = None,
        llm_gateway_url: str = "http://llm-gateway.internal/v1/turns",
        **kwargs: Any,
    ) -> RunnerBackend:
        if run_root is None:
            self._counter += 1
            run_root = self.tmp_path / f"runs-{self._counter}"
        if workspace is None:
            workspace = self.workspace()
        if token_manager is None:
            token_manager = self.token_manager()

        if model_adapter_factory is None:
            scripted_turns = list(turns or [ModelTurn(response_id="turn_1", final_text="done")])

            def model_adapter_factory(spec: Any, llm_gateway_token: str) -> FakeModelAdapter:
                del spec, llm_gateway_token
                adapter = FakeModelAdapter(turns=list(scripted_turns))
                if adapters is not None:
                    adapters.append(adapter)
                return adapter

        backend = RunnerBackend(
            run_root=run_root,
            token_manager=token_manager,
            allowed_workspace_roots=(workspace,),
            llm_gateway_url=llm_gateway_url,
            model_adapter_factory=model_adapter_factory,
            **kwargs,
        )
        self._track_backend(backend)
        return backend

    def track(self, backend: RunnerBackend) -> RunnerBackend:
        self._track_backend(backend)
        return backend

    def close(self) -> None:
        cleanup_errors: list[str] = []
        for backend in list(self._backends):
            try:
                pending = backend.drain(timeout_s=5.0)
                if pending:
                    cleanup_errors.append(f"backend {id(backend)} left pending runs: {pending}")
                backend.shutdown(drain=False)
            except Exception as exc:  # noqa: BLE001 - surface all teardown failures together
                cleanup_errors.append(f"backend {id(backend)} shutdown failed: {exc!r}")
        for backend in list(self._backends):
            for future in self._futures.get(id(backend), []):
                try:
                    future.result(timeout=0.5)
                except FutureTimeoutError:
                    future.cancel()
                    time.sleep(0.05)
                    try:
                        future.result(timeout=1.0)
                    except FutureTimeoutError:
                        cleanup_errors.append(
                            f"backend {id(backend)} future stayed pending after cancel"
                        )
                    except Exception:
                        pass
                except Exception:
                    pass
        self._futures.clear()
        self._backends.clear()
        if cleanup_errors:
            raise AssertionError("\n".join(cleanup_errors))

    def _track_backend(self, backend: RunnerBackend) -> None:
        if backend in self._backends:
            return
        self._backends.append(backend)
        self._futures.setdefault(id(backend), [])
        original_spawn = backend._spawn

        def tracked_spawn(coro: Any) -> Any:
            future = original_spawn(coro)
            self._futures.setdefault(id(backend), []).append(future)
            return future

        backend._spawn = tracked_spawn  # type: ignore[method-assign]
