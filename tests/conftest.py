from __future__ import annotations

# ruff: noqa: E402

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from support.backend_factory import ManagedBackendFactory, set_current_backend_factory
from support.studio_harness import studio as studio
from support.test_tiers import classify_items


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    del config
    errors = classify_items(items)
    if errors:
        raise pytest.UsageError("invalid test tier classification:\n" + "\n".join(errors))


@pytest.fixture(autouse=True)
def backend_factory(tmp_path: Path) -> Any:
    factory = ManagedBackendFactory(tmp_path)
    set_current_backend_factory(factory)
    try:
        yield factory
    finally:
        set_current_backend_factory(None)
        factory.close()
