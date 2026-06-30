"""Compatibility namespace for the renamed ``monoid_agent_kernel`` package.

Use ``monoid_agent_kernel`` for new code.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import warnings
from types import ModuleType

_ALIAS_PACKAGE = __name__
_TARGET_PACKAGE = "monoid_agent_kernel"


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, alias_name: str, target_name: str) -> None:
        self._alias_name = alias_name
        self._target_name = target_name

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        module = importlib.import_module(self._target_name)
        sys.modules[self._alias_name] = module
        return module

    def exec_module(self, module: ModuleType) -> None:
        sys.modules[self._alias_name] = module


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: object | None = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname == _ALIAS_PACKAGE or not fullname.startswith(f"{_ALIAS_PACKAGE}."):
            return None

        target_name = _TARGET_PACKAGE + fullname[len(_ALIAS_PACKAGE) :]
        target_spec = importlib.util.find_spec(target_name)
        if target_spec is None:
            return None

        is_package = target_spec.submodule_search_locations is not None
        spec = importlib.machinery.ModuleSpec(
            fullname,
            _AliasLoader(fullname, target_name),
            is_package=is_package,
        )
        spec.origin = target_spec.origin
        spec.has_location = target_spec.has_location
        if is_package:
            spec.submodule_search_locations = list(target_spec.submodule_search_locations or [])
        return spec


def _install_alias_finder() -> None:
    if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())


_install_alias_finder()

warnings.warn(
    "native_agent_runner is deprecated; use monoid_agent_kernel instead.",
    DeprecationWarning,
    stacklevel=2,
)

_target_module = importlib.import_module(_TARGET_PACKAGE)

__path__ = _target_module.__path__
__file__ = _target_module.__file__
__all__ = getattr(_target_module, "__all__", [])

for _name in __all__:
    globals()[_name] = getattr(_target_module, _name)


def __getattr__(name: str) -> object:
    return getattr(_target_module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_target_module)))
