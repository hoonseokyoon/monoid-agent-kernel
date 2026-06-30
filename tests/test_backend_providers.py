from __future__ import annotations

from support.backend_harness import (
    AgentConfigError,
    AgentRuntimeConfig,
    BackendRunRequest,
    FakeModelAdapter,
    ModelTurn,
    OutputValidatorBinding,
    Path,
    RunnerBackend,
    _default_config,
    _provider_backend,
    _recoverable_backend,
    _token_manager,
    _workspace,
    eventually,
    fake_tool_call,
    json,
    pytest,
    runtime_config,
    tool_binding,
)

pytestmark = pytest.mark.integration


def test_backend_validates_provider_tool_only_when_attached(tmp_path: Path) -> None:
    # A binding to a provider tool (`skill`) is "unknown" to validation unless the provider is
    # attached — and accepted once it is. This is the DX-15 / agent_spawn precedent for providers.
    workspace = _workspace(tmp_path)
    token_manager = _token_manager()
    request = BackendRunRequest(
        tenant_id="tenant_a",
        user_id="user_a",
        workspace_root=workspace,
        instruction="hi",
        runtime_config=runtime_config("skill", "run.finish"),
    )

    bare = RunnerBackend(
        run_root=tmp_path / "runs-bare",
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=lambda _s, _t: FakeModelAdapter(turns=[ModelTurn(final_text="x")]),
    )
    with pytest.raises(AgentConfigError):
        bare.submit_run(request)

    attached = _provider_backend(
        tmp_path / "runs", token_manager, workspace, turns=[ModelTurn(final_text="x")]
    )
    submission = attached.submit_run(request)
    assert attached.wait_for_run(submission.run_id, timeout_s=20) == "completed"


def test_backend_replace_runtime_config_accepts_provider_tool(tmp_path: Path) -> None:
    # The hot-swap path (settings change) re-validates the config; a provider tool added on the
    # swap must remain valid (covers the replace_runtime_config validation call site).
    workspace = _workspace(tmp_path)
    token_manager = _token_manager()
    backend = _provider_backend(
        tmp_path / "runs", token_manager, workspace, turns=[ModelTurn(response_id="r1", final_text="first")]
    )
    backend.idle_timeout_s = 30.0
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hello",
            runtime_config=_default_config(),  # no skill binding yet
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert eventually(lambda: backend._record(run_id).status == "awaiting_input")

    current = backend.current_runtime_config(run_id)
    result = backend.replace_runtime_config(
        run_id,
        token,
        expected_version=current.config_version,
        issuer="test",
        reason="enable skills mid-run",
        config=runtime_config("fs.read", "fs.write", "run.finish", "skill"),
    )
    assert result["config_version"] > current.config_version
    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_replace_runtime_config_preserves_output_validators(tmp_path: Path) -> None:
    # A hot-swap whose version isn't greater triggers an auto-bump rebuild; that rebuild must NOT
    # drop the output_validators opt-out (else a disabled validator keeps running).
    workspace = _workspace(tmp_path)
    token_manager = _token_manager()
    backend = _provider_backend(
        tmp_path / "runs", token_manager, workspace, turns=[ModelTurn(response_id="r1", final_text="first")]
    )
    backend.idle_timeout_s = 30.0
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hi",
            runtime_config=_default_config(),
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert eventually(lambda: backend._record(run_id).status == "awaiting_input")

    current = backend.current_runtime_config(run_id)
    binding = OutputValidatorBinding(validator_id="x", enabled=False)
    new_cfg = AgentRuntimeConfig(
        definition_id="t",
        config_version=current.config_version,  # not greater → auto-bump rebuild path
        tools=(tool_binding("fs.read"), tool_binding("fs.write"), tool_binding("run.finish")),
        output_validators=(binding,),
    )
    backend.replace_runtime_config(
        run_id, token, expected_version=current.config_version, issuer="test", reason="opt out", config=new_cfg
    )
    assert backend._record(run_id).runtime_config.output_validators == (binding,)
    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_backend_executes_provider_tool_through_loop(tmp_path: Path) -> None:
    # End-to-end: a `skill` tool call issued by the model is registered from the backend's
    # provider, executed, and its result reaches the model — proving the provider is actually
    # wired into the run's tool registry, not just past validation.
    workspace = _workspace(tmp_path)
    token_manager = _token_manager()
    adapters: list = []
    backend = _provider_backend(
        tmp_path / "runs",
        token_manager,
        workspace,
        adapters=adapters,
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("skill", {"name": "greeter"}, "c1"),)),
            ModelTurn(response_id="r2", final_text="greeted"),
        ],
    )
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="greet the user",
            runtime_config=runtime_config("skill", "run.finish"),
        )
    )
    assert backend.wait_for_run(submission.run_id, timeout_s=20) == "completed"
    observations = [obs for a in adapters for r in a.requests for obs in r.observations]
    skill_obs = [obs for obs in observations if obs.tool_name == "skill"]
    assert skill_obs, "the skill tool result never reached the model"
    assert "warm hello" in json.dumps(skill_obs[0].output, default=str)


def test_backend_runs_output_validator_with_retry(tmp_path: Path) -> None:
    # End-to-end (Q3): a validator attached to the backend (the output_validators seam) runs inside
    # a backend-driven loop — default-on, no config binding needed. It rejects the first (bad) final
    # response, the loop re-prompts, and the run settles ``completed`` on the corrected answer.
    from monoid_agent_kernel.core.output_validator import ValidationOutcome

    class ContainsOkValidator:
        id = "contains.ok"
        schema = None

        def validate(self, view):
            if "ok" in view.final_text:
                return ValidationOutcome(ok=True, value=view.final_text)
            return ValidationOutcome(ok=False, feedback="answer must contain 'ok'")

    workspace = _workspace(tmp_path)
    token_manager = _token_manager()
    adapters: list = []

    def factory(spec, llm_gateway_token):
        del spec, llm_gateway_token
        adapter = FakeModelAdapter(
            turns=[
                ModelTurn(response_id="r1", final_text="nope", stop_reason="stop"),
                ModelTurn(response_id="r2", final_text="ok done", stop_reason="stop"),
            ]
        )
        adapters.append(adapter)
        return adapter

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
        output_validators=(ContainsOkValidator(),),
    )
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="go",
            runtime_config=runtime_config("run.finish"),
        )
    )

    assert backend.wait_for_run(submission.run_id, timeout_s=20) == "completed"
    # Proof the validator ran through the backend loop: the bad answer was rejected and the model
    # was re-prompted — two model requests for one user turn.
    assert sum(len(a.requests) for a in adapters) == 2
    # The validated value is surfaced through the backend result projection.
    assert backend.result(submission.run_id, submission.run_token)["final_output"] == "ok done"
    # ...and through the status projection.
    assert backend.status(submission.run_id, submission.run_token)["final_output"] == "ok done"

    # Stream-driven runs (astream_run) record an AgentRunResult without ever passing through
    # _drive_open_session, so last_final_output stays None. status() must then fall back to the
    # terminal result's final_output rather than reporting null.
    record = backend._records[submission.run_id]
    record.last_final_output = None
    assert backend.status(submission.run_id, submission.run_token)["final_output"] == "ok done"


def test_json_safe_sanitizes_nested_non_json() -> None:
    # A validator value nested inside a dict/list that isn't JSON-serializable must not 500 the
    # status/result projection (review fix ③: _json_safe is deep, not shallow).
    from monoid_agent_kernel.reference.backend.service import _json_safe

    class _Model:
        def __repr__(self) -> str:
            return "Model()"

    out = _json_safe({"a": [_Model()], "b": {"c": _Model()}})
    json.dumps(out)  # must not raise — proves the whole structure is JSON-safe
    assert out["a"][0] == "Model()"
    assert out["b"]["c"] == "Model()"


def test_backend_resume_carries_providers(tmp_path: Path) -> None:
    # A parked run resumed by a fresh backend (simulated restart) must re-attach providers —
    # they are backend-instance fields read at every loop build, including the resume site.
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    token_manager = _token_manager()

    backend1 = _provider_backend(
        run_root, token_manager, workspace, turns=[ModelTurn(response_id="r1", final_text="first")]
    )
    backend1.idle_timeout_s = 30.0
    submission = backend1.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hello",
            runtime_config=runtime_config("skill", "run.finish"),
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert eventually(lambda: backend1.checkpoint_store.latest(run_id) is not None)
    assert eventually(lambda: backend1._record(run_id).status == "awaiting_input")

    # Fresh backend (empty _records) with its own provider instance resumes the run, then a
    # follow-up triggers a skill call that must resolve against the re-attached provider.
    resumed: list = []
    backend2 = _provider_backend(
        run_root,
        token_manager,
        workspace,
        adapters=resumed,
        turns=[
            ModelTurn(response_id="r2", tool_calls=(fake_tool_call("skill", {"name": "greeter"}, "c2"),)),
            ModelTurn(response_id="r3", final_text="greeted"),
        ],
    )
    backend2.idle_timeout_s = 30.0
    backend2.max_recover_attempts = 10_000
    assert backend2.resume_run(run_id, token)["resumed"] is True
    assert backend2.send_message(run_id, token, "use the greeter skill")["status"] == "queued"
    assert eventually(
        lambda: any(obs.tool_name == "skill" for a in resumed for r in a.requests for obs in r.observations)
    ), "the skill tool did not resolve after resume — providers were not re-attached"

    backend2.cancel_run(run_id, token)
    backend2.wait_for_run(run_id, timeout_s=20)
    backend1.cancel_run(run_id, token)


def test_read_run_artifact_by_digest_with_slicing(tmp_path: Path) -> None:
    # The R9 data-returning seam: a blob put under a run is fetched back by its sha256 digest,
    # token-scoped, with offset/limit slicing — and malformed/unknown digests are rejected.
    import hashlib

    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    token_manager = _token_manager()
    backend = _recoverable_backend(run_root, token_manager, workspace, [], turns=[ModelTurn(final_text="x")])
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hi",
            runtime_config=_default_config(),
        )
    )
    run_id, token = submission.run_id, submission.run_token
    backend.wait_for_run(run_id, timeout_s=20)

    data = b"PK\x03\x04artifact-bytes-0123456789"
    digest = backend.checkpoint_store.put_blob(run_id, data)
    assert digest == hashlib.sha256(data).hexdigest()

    assert backend.read_run_artifact(run_id, token, digest) == data
    assert backend.read_run_artifact(run_id, token, digest, offset=2) == data[2:]
    assert backend.read_run_artifact(run_id, token, digest, offset=2, limit=4) == data[2:6]

    with pytest.raises(ValueError):
        backend.read_run_artifact(run_id, token, "ZZZ")  # malformed digest → 400
    with pytest.raises(KeyError):
        backend.read_run_artifact(run_id, token, "a" * 64)  # unknown digest → 404


def test_resume_run_rejects_terminal_and_unknown(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    token_manager = _token_manager()
    backend = _recoverable_backend(run_root, token_manager, workspace, [], turns=[ModelTurn(final_text="x")])

    # An unknown run id (no run.json) is rejected even with a syntactically valid token.
    bogus = token_manager.issue(
        kind="run_access", audience="monoid.backend",
        run_id="run_missing", tenant_id="tenant_a", user_id="user_a", ttl_s=300,
    )
    with pytest.raises(KeyError):
        backend.resume_run("run_missing", bogus)
