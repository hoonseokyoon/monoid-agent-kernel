from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from support.http import (
    http_get_json as _json_get,
    http_json,
    serving,
    wait_http_ready as _wait_http_ready,
)
from support.runtime import runtime_config

from monoid_agent_kernel.reference.backend.http import create_backend_server
from monoid_agent_kernel.reference.backend.service import BackendRunRequest, RunnerBackend
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.errors import ModelAdapterError, PermissionDenied
from monoid_agent_kernel.reference.llm_gateway.http import create_llm_gateway_server
from monoid_agent_kernel.reference.llm_gateway.providers import offline_provider_factory
from monoid_agent_kernel.reference.llm_gateway.service import (
    LlmGatewayBackend,
    LlmGatewayTurnRecord,
)
from monoid_agent_kernel.providers.openai import OpenAIModelAdapter
from monoid_agent_kernel.providers.base import (
    ModelTurn,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    TurnComplete,
)
from monoid_agent_kernel.providers.gateway import _chunk_from_event, _parse_gateway_response
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call


def _token_manager() -> TokenManager:
    return TokenManager.from_secret("y" * 32)


def _llm_token(manager: TokenManager, *, run_id: str = "run_1", tenant_id: str = "tenant_a") -> str:
    return manager.issue(
        kind="llm_gateway",
        audience="csp.llm-gateway",
        run_id=run_id,
        tenant_id=tenant_id,
        user_id="user_a",
        ttl_s=600,
        metadata={"agent_config_hash": "test"},
    )


def _payload(*, previous_turn_handle: str | None = None) -> dict:
    payload = {
        "protocol": "monoid.llm-turn.v1",
        "model": "gpt-5.5",
        "system_prompt": "sys",
        "reasoning": {"effort": "low"},
        "tools": [
            {
                "id": "fs.read",
                "name": "fs_read",
                "description": "Read file.",
                "input_schema": {"type": "object"},
                "capability": "fs.read",
                "side_effect": "read",
            }
        ],
    }
    if previous_turn_handle:
        payload["previous_turn_handle"] = previous_turn_handle
        payload["observations"] = [
            {"call_id": "call_1", "tool_name": "fs_read", "output": {"ok": True}}
        ]
    else:
        payload["instruction"] = "Read notes."
    return payload


def test_llm_gateway_validates_token_and_returns_opaque_turn_handle() -> None:
    manager = _token_manager()
    seen_previous_ids: list[str | None] = []

    def factory(_claims, _config):
        index = len(seen_previous_ids)

        class Adapter:
            def next_turn(self, request):
                seen_previous_ids.append(request.previous_turn_handle)
                if index == 0:
                    return ModelTurn(
                        response_id="provider_response_secret_1",
                        tool_calls=(ToolCall("call_1", "fs_read", {"path": "notes.md"}),),
                        usage={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
                    )
                return ModelTurn(
                    response_id="provider_response_secret_2",
                    final_text="done",
                    usage={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
                )

        return Adapter()

    gateway = LlmGatewayBackend(token_manager=manager, provider_adapter_factory=factory)
    token = _llm_token(manager)

    first = gateway.handle_turn(token, _payload())
    assert first["turn_handle"].startswith("turn_")
    assert "provider_response_secret_1" not in json.dumps(first)
    assert first["tool_calls"][0]["name"] == "fs_read"

    second = gateway.handle_turn(token, _payload(previous_turn_handle=first["turn_handle"]))
    assert second["final_text"] == "done"
    assert seen_previous_ids == [None, "provider_response_secret_1"]
    assert gateway.tenant_usage("tenant_a")["total_tokens"] == 14

    other_model = _payload()
    other_model["model"] = "other-model"
    assert gateway.handle_turn(token, other_model)["turn_handle"].startswith("turn_")


def test_llm_gateway_python_boundary_normalizes_request_and_response_values() -> None:
    manager = _token_manager()
    seen_requests = []

    class Adapter:
        def next_turn(self, request):
            seen_requests.append(request)
            return ModelTurn(
                final_text="done\ud800",
                tool_calls=(
                    ToolCall(
                        "call\ud800",
                        "tool\udc00",
                        {"text": "\ud800", "number": float("nan")},
                    ),
                ),
                raw={"number": float("inf")},
            )

    gateway = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, _config: Adapter(),
    )
    payload = _payload()
    payload["instruction"] = "prompt\ud800"
    payload["tools"][0]["input_schema"]["example"] = {
        "text": "\ud800",
        "number": -float("inf"),
    }

    result = gateway.handle_turn(_llm_token(manager), payload)

    assert seen_requests[0].instruction == "prompt�"
    # The schema's *strings* are repaired like any other; its non-finite number is not
    # substituted. A tool schema is a control document the request promises to forward
    # verbatim -- rewriting `-inf` to `null` would hand the upstream provider a different
    # constraint than the caller wrote -- so the value survives to the strict serializer,
    # which refuses the request there (see tests/test_tool_schema_delivery.py).
    forwarded_schema = seen_requests[0].tools[0].input_schema["example"]
    assert forwarded_schema["text"] == "�"
    assert math.isinf(forwarded_schema["number"]) and forwarded_schema["number"] < 0
    assert result["final_text"] == "done�"
    assert result["tool_calls"] == [
        {"call_id": "call�", "name": "tool�", "arguments": {"text": "�", "number": None}}
    ]


def test_llm_gateway_stream_preserves_split_surrogate_pairs_per_channel() -> None:
    manager = _token_manager()

    class Adapter:
        async def astream_turn(self, request):
            del request
            yield TextDelta("\ud83d")
            yield ReasoningDelta("\ud83d")
            yield ToolCallDelta(
                index=0,
                arguments_fragment='{"emoji":"\ud83d',
                id="call\ud800",
                name="tool\udc00",
            )
            yield ReasoningDelta("\ude00")
            yield TextDelta("\ude00")
            yield ToolCallDelta(index=0, arguments_fragment='\ude00"}')
            yield TurnComplete(response_id="response\ud800", usage={"total_tokens": 1})

    gateway = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, _config: Adapter(),
    )

    frames = list(gateway.handle_turn_stream(_llm_token(manager), _payload()))

    assert "".join(frame["text"] for frame in frames if frame["type"] == "text_delta") == "😀"
    assert "".join(frame["text"] for frame in frames if frame["type"] == "reasoning_delta") == "😀"
    tool_frames = [frame for frame in frames if frame["type"] == "tool_call_delta"]
    assert "".join(frame["arguments_fragment"] for frame in tool_frames) == '{"emoji":"😀"}'
    assert tool_frames[0]["id"] == "call�"
    assert tool_frames[0]["name"] == "tool�"
    assert frames[-1]["type"] == "turn_complete"


def test_llm_gateway_flushes_pending_surrogate_before_provider_error() -> None:
    manager = _token_manager()

    class Adapter:
        async def astream_turn(self, request):
            del request
            yield TextDelta("\ud800")
            raise RuntimeError("provider stream failed")

    gateway = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, _config: Adapter(),
    )
    frames = gateway.handle_turn_stream(_llm_token(manager), _payload())

    assert next(frames) == {"type": "text_delta", "text": ""}
    assert next(frames) == {"type": "text_delta", "text": "�"}
    with pytest.raises(RuntimeError, match="provider stream failed"):
        next(frames)


def test_llm_gateway_classifies_nonportable_provider_output_as_bad_gateway() -> None:
    manager = _token_manager()

    class Adapter:
        def next_turn(self, request):
            del request
            arguments = {chr(0xD800): 1}
            arguments["�"] = 2
            return ModelTurn(
                tool_calls=(
                    ToolCall(
                        "call_1",
                        "tool_1",
                        arguments,
                    ),
                )
            )

    gateway = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, _config: Adapter(),
    )

    with pytest.raises(ModelAdapterError, match="non-portable response"):
        gateway.handle_turn(_llm_token(manager), _payload())


def test_llm_gateway_accepts_legacy_turn_protocol_during_migration() -> None:
    manager = _token_manager()
    gateway = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, _config: FakeModelAdapter(
            turns=[ModelTurn(response_id="provider_1", final_text="done")]
        ),
    )
    payload = _payload()
    payload["protocol"] = "native-agent-runner.llm-turn.v1"

    result = gateway.handle_turn(_llm_token(manager), payload)

    assert result["protocol"] == "monoid.llm-turn-result.v1"
    assert result["final_text"] == "done"


def test_llm_gateway_rejects_cross_run_turn_handle() -> None:
    manager = _token_manager()
    gateway = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, _config: FakeModelAdapter(
            turns=[ModelTurn(response_id="provider_1", final_text="done")]
        ),
    )
    first = gateway.handle_turn(_llm_token(manager, run_id="run_a"), _payload())

    with pytest.raises(PermissionDenied):
        gateway.handle_turn(
            _llm_token(manager, run_id="run_b"),
            _payload(previous_turn_handle=first["turn_handle"]),
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.update({"reasoning": []}),
        lambda payload: payload.update({"messages": {}}),
        lambda payload: payload.update({"tools": {}}),
        lambda payload: payload.update({"observations": {}}),
        lambda payload: payload["tools"][0].update({"input_schema": []}),
        lambda payload: payload.update(
            {
                "previous_turn_handle": "turn_1",
                "observations": [{"call_id": "c1", "is_background": "false"}],
            }
        ),
        lambda payload: payload.update(
            {"previous_turn_handle": "turn_1", "observations": [{"call_id": "c1", "output": []}]}
        ),
    ],
)
def test_llm_gateway_rejects_present_wrong_type_payload_fields(mutator) -> None:
    manager = _token_manager()
    gateway = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, _config: FakeModelAdapter(
            turns=[ModelTurn(response_id="provider_1", final_text="done")]
        ),
    )
    payload = _payload()
    mutator(payload)

    with pytest.raises(ValueError):
        gateway.handle_turn(_llm_token(manager), payload)


def test_llm_gateway_http_endpoint_and_usage(tmp_path: Path) -> None:
    manager = _token_manager()
    gateway = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, _config: FakeModelAdapter(
            turns=[
                ModelTurn(response_id="provider_1", final_text="done", usage={"total_tokens": 9})
            ]
        ),
    )
    server = create_llm_gateway_server(gateway, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        _wait_http_ready(base_url)
        with pytest.raises(HTTPError) as exc_info:
            _json_post(f"{base_url}/internal/llm/turns", _payload())
        assert exc_info.value.code == 401

        result = _json_post(
            f"{base_url}/internal/llm/turns",
            _payload(),
            token=_llm_token(manager),
        )
        assert result["final_text"] == "done"
        usage = _json_get(f"{base_url}/internal/llm/tenants/tenant_a/usage", token="admin")
        assert usage["calls"] == 1
        assert usage["total_tokens"] == 9
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_llm_gateway_http_rejects_present_wrong_type_payload_field() -> None:
    manager = _token_manager()
    gateway = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, _config: FakeModelAdapter(
            turns=[ModelTurn(response_id="provider_1", final_text="done")]
        ),
    )
    server = create_llm_gateway_server(gateway, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    payload = _payload()
    payload["reasoning"] = []
    try:
        _wait_http_ready(base_url)
        with pytest.raises(HTTPError) as exc_info:
            _json_post(f"{base_url}/internal/llm/turns", payload, token=_llm_token(manager))
        assert exc_info.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_llm_gateway_http_normalizes_model_adapter_error() -> None:
    manager = _token_manager()

    class FailingAdapter:
        def next_turn(self, _request):
            raise ModelAdapterError(
                "provider overloaded",
                provider_error_code="gateway_server_error",
                retryable=True,
            )

    gateway = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, _config: FailingAdapter(),
    )
    server = create_llm_gateway_server(gateway, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        _wait_http_ready(base_url)
        with pytest.raises(HTTPError) as exc_info:
            _json_post(
                f"{base_url}/internal/llm/turns",
                _payload(),
                token=_llm_token(manager),
            )
        body = json.loads(exc_info.value.read().decode("utf-8"))
        assert exc_info.value.code == 503
        assert body["error_code"] == "gateway_server_error"
        assert body["retryable"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_runner_backend_can_use_http_llm_gateway_end_to_end(tmp_path: Path) -> None:
    manager = _token_manager()
    gateway = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, _config: FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="provider_1", final_text="gateway done", usage={"total_tokens": 11}
                )
            ]
        ),
    )
    server = create_llm_gateway_server(gateway, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    gateway_url = f"http://127.0.0.1:{server.server_address[1]}"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("notes\n", encoding="utf-8")
    try:
        _wait_http_ready(gateway_url)
        runner_backend = RunnerBackend(
            run_root=tmp_path / "runs",
            token_manager=manager,
            allowed_workspace_roots=(workspace,),
            llm_gateway_url=f"{gateway_url}/internal/llm/turns",
        )
        submission = runner_backend.submit_run(
            BackendRunRequest(
                tenant_id="tenant_a",
                user_id="user_a",
                workspace_root=workspace,
                instruction="Finish through gateway.",
                runtime_config=runtime_config("run.finish"),
            )
        )
        assert runner_backend.wait_for_run(submission.run_id, timeout_s=20) == "completed"
        result = runner_backend.result(submission.run_id, submission.run_token)
        assert result["final_text"] == "gateway done"
        assert runner_backend.tenant_usage("tenant_a")["total_tokens"] == 11
        assert gateway.tenant_usage("tenant_a")["total_tokens"] == 11
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_fake_full_stack_contract_propose_proposal_usage_and_auth(tmp_path: Path) -> None:
    manager = _token_manager()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("alpha notes\n", encoding="utf-8")
    sentinel_key = "sk-test-hidden-provider-key"
    adapters: dict[str, FakeModelAdapter] = {}

    def factory(claims, _config):
        if claims.run_id not in adapters:
            adapters[claims.run_id] = FakeModelAdapter(
                turns=[
                    ModelTurn(
                        response_id="provider_1",
                        tool_calls=(
                            fake_tool_call("fs_read", {"path": "notes.md"}, "call_read"),
                            fake_tool_call(
                                "fs_write",
                                {
                                    "path": "SUMMARY.md",
                                    "content": "Summary from fake gateway\n",
                                    "create_dirs": False,
                                },
                                "call_write",
                            ),
                        ),
                        usage={"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
                    ),
                    ModelTurn(
                        response_id="provider_2",
                        tool_calls=(
                            fake_tool_call(
                                "run_finish",
                                {"summary": "Created SUMMARY.md", "outputs": ["SUMMARY.md"]},
                                "call_finish",
                            ),
                        ),
                        usage={"input_tokens": 2, "output_tokens": 5, "total_tokens": 7},
                    ),
                ]
            )
        return adapters[claims.run_id]

    gateway = LlmGatewayBackend(token_manager=manager, provider_adapter_factory=factory)
    gateway_server = create_llm_gateway_server(
        gateway, host="127.0.0.1", port=0, admin_token="gateway-admin"
    )
    gateway_thread = threading.Thread(target=gateway_server.serve_forever, daemon=True)
    gateway_thread.start()
    gateway_url = f"http://127.0.0.1:{gateway_server.server_address[1]}"
    runner_backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url=f"{gateway_url}/internal/llm/turns",
    )
    runner_server = create_backend_server(
        runner_backend, host="127.0.0.1", port=0, admin_token="runner-admin"
    )
    runner_thread = threading.Thread(target=runner_server.serve_forever, daemon=True)
    runner_thread.start()
    runner_url = f"http://127.0.0.1:{runner_server.server_address[1]}"
    try:
        _wait_http_ready(gateway_url)
        _wait_http_ready(runner_url)
        created = _json_post(
            f"{runner_url}/v1/runs",
            {
                "tenant_id": "tenant_a",
                "user_id": "user_a",
                "workspace_root": str(workspace),
                "instruction": "Read notes.md and propose SUMMARY.md.",
                "mode": "propose",
                "runtime_config": runtime_config("fs.read", "fs.write", "run.finish").to_json(),
            },
            token="runner-admin",
        )
        run_id = created["run_id"]
        run_token = created["run_token"]
        assert runner_backend.wait_for_run(run_id, timeout_s=5) == "completed"

        with pytest.raises(HTTPError) as exc_info:
            urlopen(Request(f"{runner_url}/v1/runs/{run_id}/result", method="GET"), timeout=5)
        assert exc_info.value.code == 401

        assert not workspace.joinpath("SUMMARY.md").exists()
        proposal = _json_get(f"{runner_url}/v1/runs/{run_id}/proposal", token=run_token)
        assert proposal["ready"] is True
        assert proposal["proposal_hash"]
        assert proposal["diff_sha256"]
        assert proposal["files"][0]["path"] == "SUMMARY.md"
        proposed_file = _json_get(
            f"{runner_url}/v1/runs/{run_id}/proposal/files/SUMMARY.md", token=run_token
        )
        assert proposed_file["content"] == "Summary from fake gateway\n"
        result = _json_get(f"{runner_url}/v1/runs/{run_id}/result", token=run_token)
        assert result["ready"] is True
        assert result["metrics"]["total_tokens"] == 14
        assert result["proposal"]["proposal_hash"] == proposal["proposal_hash"]
        events = _json_get(f"{runner_url}/v1/runs/{run_id}/events", token=run_token)["events"]
        assert events[0]["type"] == "run.started"
        assert events[-1]["type"] == "run.finished"
        assert any(event["type"] == "workspace.proposal.updated" for event in events)
        assert runner_backend.tenant_usage("tenant_a")["total_tokens"] == 14
        assert gateway.tenant_usage("tenant_a")["total_tokens"] == 14
        run_text = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in Path(result["run_dir"]).rglob("*")
            if path.is_file()
        )
        assert sentinel_key not in run_text
        assert "OPENAI_API_KEY" not in run_text
    finally:
        runner_server.shutdown()
        runner_server.server_close()
        runner_thread.join(timeout=5)
        gateway_server.shutdown()
        gateway_server.server_close()
        gateway_thread.join(timeout=5)


def _json_post(url: str, payload: dict, *, token: str | None = None) -> dict:
    return http_json(url, payload, token=token)


def test_llm_gateway_offline_provider_answers_without_a_key() -> None:
    # DX-1: the gateway can serve turns with zero credentials via the offline provider,
    # the LLM-side counterpart of the WebGateway's fake provider.
    manager = _token_manager()
    gateway = LlmGatewayBackend(
        token_manager=manager, provider_adapter_factory=offline_provider_factory
    )
    result = gateway.handle_turn(_llm_token(manager), _payload())
    assert "echo model" in result["final_text"].lower()
    assert result["tool_calls"] == []
    assert result["usage"]["total_tokens"] > 0


class _OneShotRetriedBackend:
    """A backend adapter whose own retry loop ran, and that cannot stream.

    ``shape`` picks what the turn contains, because that decides which synthesized chunk is the
    *only* carrier of the retry: a turn with text has a `TextDelta` to hold it, one with neither text
    nor tool calls has nothing but the terminal chunk.
    """

    def __init__(self, *, retried: bool, shape: str = "text") -> None:
        self._retried = retried
        self._shape = shape

    def next_turn(self, request):
        del request
        return ModelTurn(
            response_id="r1",
            final_text="done" if self._shape == "text" else None,
            tool_calls=(
                (ToolCall(id="c1", name="fs_read", arguments={"path": "A.md"}),)
                if self._shape == "tools"
                else ()
            ),
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            provider_retried=self._retried,
        )


class _RetriedBackend(_OneShotRetriedBackend):
    """The same, able to stream. Separate class rather than a flag: the gateway selects the path
    with ``getattr(adapter, "astream_turn", None)``, and an instance cannot hide a class attribute.
    """

    async def astream_turn(self, request):
        del request
        yield TextDelta(text="do", provider_retried=self._retried)
        yield TextDelta(text="ne", provider_retried=self._retried)
        yield TurnComplete(
            response_id="r1", usage={}, stop_reason="stop", provider_retried=self._retried
        )


def _retried_gateway(retried: bool, *, streams: bool = True, shape: str = "text"):
    manager = _token_manager()
    backend = _RetriedBackend if streams else _OneShotRetriedBackend
    gateway = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda claims, config: backend(retried=retried, shape=shape),
    )
    return gateway, _llm_token(manager)


@pytest.mark.parametrize("retried", [True, False])
def test_a_backend_retry_survives_the_one_shot_gateway_round_trip(retried: bool) -> None:
    """Two retry loops sit on this path and the client can only see its own.

    A gateway whose backend retried, answering a request this client got right the first time,
    recorded a clean single attempt -- the exact failure ``provider_retried`` exists to prevent.
    The parametrization is the counterweight: a wire that always said "retried" would pass the
    True case alone.
    """
    gateway, token = _retried_gateway(retried)
    response = gateway.handle_turn(token, _payload())
    assert response["provider_retried"] is retried
    assert _parse_gateway_response(response).provider_retried is retried


@pytest.mark.parametrize("retried", [True, False])
def test_a_backend_retry_survives_the_streamed_gateway_round_trip(retried: bool) -> None:
    """Carried by the delta frames, not only by ``turn_complete``.

    A stream cancelled mid-flight never delivers the terminal frame, so evidence that rides only
    there is lost exactly when it matters. Asserted on the frames *before* the terminal one.
    """
    gateway, token = _retried_gateway(retried)
    frames = list(gateway.handle_turn_stream(token, _payload()))
    deltas = [frame for frame in frames if frame["type"] != "turn_complete"]
    assert deltas, "the fixture must produce frames before the terminal one"
    assert all(frame.get("provider_retried", False) is retried for frame in deltas)
    assert frames[-1]["provider_retried"] is retried

    chunks = [_chunk_from_event(frame) for frame in frames]
    assert all(chunk.provider_retried is retried for chunk in chunks if chunk is not None)


@pytest.mark.parametrize("retried", [True, False])
@pytest.mark.parametrize(
    ("shape", "expected_types"),
    [
        ("text", ["text_delta", "turn_complete"]),
        ("tools", ["tool_call_delta", "turn_complete"]),
        ("silent", ["turn_complete"]),
    ],
)
def test_a_backend_that_cannot_stream_still_reports_its_retry(
    retried: bool, shape: str, expected_types: list[str]
) -> None:
    """The synthesized deltas stand in for a stream, so they carry what the turn reported.

    Every shape, because each one moves where the evidence has to survive. Only the text case was
    exercised, and there a `TextDelta` carries the fact into the assembled turn, which is what the
    terminal frame is built from -- so the flag on the synthesized `ToolCallDelta` and on the
    synthesized `TurnComplete` was redundant and free to be dropped. A turn with neither text nor
    tool calls has *nothing* but the terminal chunk, and dropping it there loses the retry outright.
    """
    gateway, token = _retried_gateway(retried, streams=False, shape=shape)
    frames = list(gateway.handle_turn_stream(token, _payload()))
    assert [frame["type"] for frame in frames] == expected_types
    assert all(frame.get("provider_retried", False) is retried for frame in frames)


@pytest.mark.parametrize("retried", [True, False])
def test_a_failing_backends_retry_reaches_the_wire_too(retried: bool) -> None:
    """A failure is where the retry matters most, and it travels by a different route.

    On the success side the turn carries the fact. A provider that *fails* produces no turn, so the
    error body is the only carrier -- and it is the case where the client's own loop contributes
    nothing to mask a loss, since a 400 is not retryable. `_write_exception` reads the fact off the
    exception for exactly this; nothing observed that it did.
    """

    class _FailingBackend:
        def next_turn(self, request):
            del request
            raise ModelAdapterError(
                "upstream said no",
                provider_error_code="gateway_bad_request",
                retryable=False,
                http_status=400,
                provider_retried=retried,
            )

    manager = _token_manager()
    gateway = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, _config: _FailingBackend(),
    )
    server = create_llm_gateway_server(gateway, host="127.0.0.1", port=0, admin_token="admin")
    with serving(server) as base_url:
        with pytest.raises(HTTPError) as caught:
            _json_post(f"{base_url}/internal/llm/turns", _payload(), token=_llm_token(manager))
        body = json.loads(caught.value.read().decode("utf-8"))

    assert caught.value.code == 400
    assert body["error_code"] == "gateway_bad_request"
    assert body["provider_retried"] is retried
    # And the client reads back what the wire said, which is the half a receipt is built from.
    error = pytest.raises(ModelAdapterError, _parse_gateway_response, body).value
    assert error.provider_retried is retried


def test_the_clients_own_retry_is_combined_with_the_gateways_not_written_over_it() -> None:
    """``attempt > 1`` is one of two independent facts, and either one is worth recording."""
    gateway, token = _retried_gateway(True)
    response = gateway.handle_turn(token, _payload())
    # The client succeeded on its first HTTP attempt, so its own loop contributes nothing.
    assert _parse_gateway_response(response).provider_retried is True


def test_the_by_reference_refusal_reaches_the_wire_as_a_classified_422() -> None:
    """Blast radius of the OpenAI adapter's by-reference refusal, end to end over the hop.

    The refusal itself is unit-tested against ``_payload``, and ``_model_error_status`` is
    tested against the exception -- but those are two halves that only *compose* into the 422
    a client sees. This drives the whole chain the deployment actually runs: a by-reference
    continuation request → ``handle_turn`` → the real ``OpenAIModelAdapter`` upstream (the
    gateway's default) → its boundary refusal → ``handle_turn``'s ``except ModelAdapterError``
    arm → ``_write_exception`` → the non-200 body. Break any link and this fails.

    The handle→provider-response mapping is seeded directly: recording it the normal way needs
    a live upstream turn, and the shape under test is selected by the *lookup*, not by how the
    record got there. No API key is used -- ``_payload`` refuses before any client is built.
    """

    pytest.importorskip("openai")
    manager = _token_manager()
    gateway = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, config: OpenAIModelAdapter(
            config, api_key="test-key-never-used", allow_direct_provider_api=True
        ),
    )
    gateway._turns["turn_seeded"] = LlmGatewayTurnRecord(
        turn_handle="turn_seeded",
        provider_response_id="provider_response_1",
        run_id="run_1",
        tenant_id="tenant_a",
        user_id="user_a",
        model="gpt-5.5",
        created_at=time.time(),
    )
    server = create_llm_gateway_server(gateway, host="127.0.0.1", port=0, admin_token="admin")
    with serving(server) as base_url:
        with pytest.raises(HTTPError) as caught:
            _json_post(
                f"{base_url}/internal/llm/turns",
                _payload(previous_turn_handle="turn_seeded"),
                token=_llm_token(manager),
            )
        body = json.loads(caught.value.read().decode("utf-8"))

    assert caught.value.code == 422
    assert body["error_code"] == "unsupported_request_shape"
    assert body["retryable"] is False
    assert body["http_status"] == 422
    assert "messages" in body["error"]
    # The classification survives the hop for the client that has to act on it.
    reconstructed = pytest.raises(ModelAdapterError, _parse_gateway_response, body).value
    assert reconstructed.provider_error_code == "unsupported_request_shape"
    assert reconstructed.http_status == 422
