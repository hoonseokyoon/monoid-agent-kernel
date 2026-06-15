"""Reference implementations of the native-agent-runner contracts.

These packages are EXAMPLES, not part of the supported public surface:

- ``native_agent_runner.reference.backend`` — a run-orchestration backend with an HTTP API.
- ``native_agent_runner.reference.llm_gateway`` — a credential-boundary LLM gateway.
- ``native_agent_runner.reference.web_gateway`` — a web search/fetch/context gateway.

Real integrators are expected to build their own services against
``native_agent_runner.contracts`` and the wire contracts documented in ``docs/CONTRACTS.md``.

Subpackages are imported explicitly (e.g. ``from native_agent_runner.reference.backend import
RunnerBackend``); this package intentionally performs no eager imports so that importing it
does not pull in the reference HTTP servers.
"""
