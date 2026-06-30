from __future__ import annotations

from pathlib import Path

from monoid_agent_kernel.reference._shared.tokens import TokenManager


def token_manager() -> TokenManager:
    return TokenManager.ephemeral()


def workspace_root(tmp_path: Path, name: str = "workspace") -> Path:
    workspace = tmp_path / name
    workspace.mkdir()
    return workspace

