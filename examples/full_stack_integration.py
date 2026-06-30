from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from monoid_agent_kernel.reference.backend.service import BackendRunRequest, RunnerBackend
from monoid_agent_kernel.reference._shared.tokens import TokenClaims, TokenManager
from monoid_agent_kernel.core.agents import AgentRuntimeConfig, RegistryToolRef, ToolBinding
from monoid_agent_kernel.core.spec import ModelConfig, ReasoningConfig
from monoid_agent_kernel.reference.llm_gateway.http import create_llm_gateway_server
from monoid_agent_kernel.reference.llm_gateway.service import LlmGatewayBackend
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn, ToolCall


DEFAULT_INSTRUCTION = """Read notes.md and create SUMMARY.md.
The summary must include:
- three concise bullets
- a Security boundaries section
- a Follow-up tasks checklist
Finish by calling run.finish with SUMMARY.md as an output.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the native agent full-stack integration scenario.")
    parser.add_argument("--mode", choices=["fake", "real"], required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--keep-run", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix=f"nar-{args.mode}-") as tmp:
        tmp_path = Path(tmp)
        result = run_scenario(
            mode=args.mode,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            root=tmp_path,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        if args.keep_run:
            kept_root = PACKAGE_ROOT / "runs" / f"integration-{args.mode}-{int(time.time())}"
            _copy_tree(result["_run_dir"], kept_root)
            print(json.dumps({"kept_run_dir": str(kept_root)}, ensure_ascii=False))


def run_scenario(*, mode: str, model: str, reasoning_effort: str, root: Path) -> dict[str, Any]:
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    workspace.joinpath("notes.md").write_text(_notes_fixture(), encoding="utf-8")
    run_root = root / "runs"
    token_secret = secrets.token_urlsafe(32)
    token_manager = TokenManager.from_secret(token_secret)
    admin_token = "admin-integration"

    if mode == "fake":
        gateway_url, stop_gateway, gateway_usage = _start_fake_gateway(token_manager)
        api_key_for_scan = None
    else:
        env_values = _load_env_file(PACKAGE_ROOT / ".env")
        api_key = env_values.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(f"OPENAI_API_KEY is missing from {PACKAGE_ROOT / '.env'}")
        os.environ.pop("OPENAI_API_KEY", None)
        gateway_url, stop_gateway, gateway_usage = _start_real_gateway_subprocess(
            token_secret=token_secret,
            admin_token=admin_token,
            openai_api_key=api_key,
        )
        api_key_for_scan = api_key

    try:
        runner_backend = RunnerBackend(
            run_root=run_root,
            token_manager=token_manager,
            allowed_workspace_roots=(workspace,),
            llm_gateway_url=gateway_url,
        )
        submission = runner_backend.submit_run(
            BackendRunRequest(
                tenant_id="tenant_integration",
                user_id="user_integration",
                workspace_root=workspace,
                instruction=DEFAULT_INSTRUCTION,
                mode="propose",
                runtime_config=_runtime_config(model=model, reasoning_effort=reasoning_effort),
                max_steps=8,
                max_tool_calls=20,
            )
        )
        status = runner_backend.wait_for_run(submission.run_id, timeout_s=180)
        result = runner_backend.result(submission.run_id, submission.run_token)
        proposal = runner_backend.proposal(submission.run_id, submission.run_token)
        proposed_file = _try_get_proposed_file(runner_backend, submission.run_id, submission.run_token, "SUMMARY.md")
        events = runner_backend.events(submission.run_id, submission.run_token)["events"]
        status_payload = runner_backend.status(submission.run_id, submission.run_token)
        runner_usage = runner_backend.tenant_usage("tenant_integration")
        llm_usage = gateway_usage("tenant_integration")
        run_dir = Path(str(result.get("run_dir") or submission.run_dir))
        leak_scan = _scan_run_for_secret(run_dir, api_key_for_scan)

        return {
            "mode": mode,
            "status": status,
            "run_id": submission.run_id,
            "workspace_summary_exists": workspace.joinpath("SUMMARY.md").exists(),
            "result_ready": result.get("ready"),
            "final_text": result.get("final_text"),
            "error": result.get("error", ""),
            "changed_paths": proposal.get("changed_paths", []),
            "proposal_ready": proposal.get("ready"),
            "proposal_files": [
                {
                    "path": item.get("path"),
                    "kind": item.get("kind"),
                    "size": item.get("size"),
                }
                for item in proposal.get("files", [])
                if isinstance(item, dict)
            ],
            "proposed_summary_preview": _preview(proposed_file.get("content", "")) if proposed_file else "",
            "event_types": [event.get("type") for event in events],
            "last_event_type": status_payload.get("last_event_type"),
            "status_error": status_payload.get("error", ""),
            "runner_usage": runner_usage,
            "llm_gateway_usage": llm_usage,
            "secret_leak_detected": leak_scan["found"],
            "secret_scan_files": leak_scan["files"],
            "diff_contains_summary": "SUMMARY.md" in str(result.get("diff", "")),
            "_run_dir": str(run_dir),
        }
    finally:
        stop_gateway()


def _runtime_config(*, model: str, reasoning_effort: str) -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        definition_id="full-stack-integration",
        model=ModelConfig(
            model=model,
            reasoning=ReasoningConfig(effort=reasoning_effort),
        ),
        tools=(
            ToolBinding(
                binding_id="fs_read",
                model_name="fs_read",
                ref=RegistryToolRef("fs.read"),
            ),
            ToolBinding(
                binding_id="fs_write",
                model_name="fs_write",
                ref=RegistryToolRef("fs.write"),
            ),
            ToolBinding(
                binding_id="run_finish",
                model_name="run_finish",
                ref=RegistryToolRef("run.finish"),
            ),
        ),
    )


def _start_fake_gateway(token_manager: TokenManager) -> tuple[str, Callable[[], None], Callable[[str], dict[str, Any]]]:
    turn_counts: dict[str, int] = {}

    def factory(claims: TokenClaims, _config: ModelConfig):
        class ScriptedAdapter:
            def next_turn(self, request: ModelRequest) -> ModelTurn:
                count = turn_counts.get(claims.run_id, 0)
                turn_counts[claims.run_id] = count + 1
                if count == 0:
                    return ModelTurn(
                        response_id=f"fake-provider-{claims.run_id}-1",
                        tool_calls=(ToolCall("call_read", "fs_read", {"path": "notes.md"}),),
                        usage={"input_tokens": 120, "output_tokens": 20, "total_tokens": 140},
                    )
                if count == 1:
                    assert request.observations and request.observations[0].output.get("ok") is True
                    return ModelTurn(
                        response_id=f"fake-provider-{claims.run_id}-2",
                        tool_calls=(
                            ToolCall(
                                "call_write",
                                "fs_write",
                                {
                                    "path": "SUMMARY.md",
                                    "content": _summary_fixture(),
                                    "create_dirs": False,
                                },
                            ),
                            ToolCall(
                                "call_finish",
                                "run_finish",
                                {
                                    "summary": "Created SUMMARY.md from notes.md",
                                    "outputs": ["SUMMARY.md"],
                                },
                            ),
                        ),
                        usage={"input_tokens": 220, "output_tokens": 90, "total_tokens": 310},
                    )
                return ModelTurn(
                    response_id=f"fake-provider-{claims.run_id}-done",
                    final_text="done",
                    usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

        return ScriptedAdapter()

    gateway = LlmGatewayBackend(token_manager=token_manager, provider_adapter_factory=factory)
    server = create_llm_gateway_server(gateway, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/internal/llm/turns"

    def stop() -> None:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    return url, stop, gateway.tenant_usage


def _start_real_gateway_subprocess(
    *,
    token_secret: str,
    admin_token: str,
    openai_api_key: str,
) -> tuple[str, Callable[[], None], Callable[[str], dict[str, Any]]]:
    port = _free_port()
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["OPENAI_API_KEY"] = openai_api_key
    env["MONOID_BACKEND_TOKEN_SECRET"] = token_secret
    env["MONOID_LLM_GATEWAY_ADMIN_TOKEN"] = admin_token
    env["PYTHONPATH"] = str(PACKAGE_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "monoid_agent_kernel.cli",
        "llm-gateway",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    process = subprocess.Popen(
        command,
        cwd=PACKAGE_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    _wait_for_health(base_url, process)

    def stop() -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def usage(tenant_id: str) -> dict[str, Any]:
        request = Request(
            f"{base_url}/internal/llm/tenants/{tenant_id}/usage",
            headers={"Authorization": f"Bearer {admin_token}"},
            method="GET",
        )
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    return f"{base_url}/internal/llm/turns", stop, usage


def _try_get_proposed_file(
    backend: RunnerBackend,
    run_id: str,
    token: str,
    path: str,
) -> dict[str, Any] | None:
    try:
        return backend.proposal_file(run_id, token, path)
    except Exception:
        return None


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        value = value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


def _wait_for_health(base_url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.time() + 15
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("LLM gateway process exited before becoming healthy")
        try:
            with urlopen(f"{base_url}/healthz", timeout=1) as response:
                if response.status == 200:
                    return
        except URLError:
            time.sleep(0.2)
    raise TimeoutError("LLM gateway did not become healthy")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _scan_run_for_secret(run_dir: Path, secret: str | None) -> dict[str, Any]:
    if not secret:
        return {"found": False, "files": []}
    found: list[str] = []
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            data = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if secret in data:
            found.append(str(path.relative_to(run_dir).as_posix()))
    return {"found": bool(found), "files": found}


def _copy_tree(source: str, target: Path) -> None:
    import shutil

    shutil.copytree(source, target)


def _preview(text: str, limit: int = 500) -> str:
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _notes_fixture() -> str:
    return """# Monoid Integration Notes

Monoid must keep provider credentials outside the agent container.
LLM calls should go through a gateway that validates short-lived run-scoped tokens.
Every run must be attributable to a tenant for usage tracking.
In propose mode, the agent can stage file changes without mutating the base workspace.
Operators need events, status, diff, and proposal snapshots to inspect progress.

TODO:
- Add an approval path that applies a reviewed proposal.
- Add tenant budget checks to the gateway.
- Add a compact UI view for proposal files.
"""


def _summary_fixture() -> str:
    return """# Summary

- Provider credentials stay inside the LLM gateway, not the agent kernel.
- Run-scoped gateway tokens let the backend bind LLM calls to tenant and user identity.
- Propose mode stages changes as observable snapshots without mutating the workspace.

## Security boundaries

- Monoid receives only a short-lived `llm_gateway` token.
- The provider API key remains in the gateway process.
- Public events avoid file contents and credential-looking payloads.

## Follow-up tasks

- [ ] Add approval-to-apply workflow.
- [ ] Add tenant budget enforcement.
- [ ] Build a compact proposal review UI.
"""


if __name__ == "__main__":
    main()
