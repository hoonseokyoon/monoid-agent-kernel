"""Reference implementations of the Monoid Agent Kernel contracts.

These packages are reference examples:

- ``monoid_agent_kernel.reference.backend`` — a run-orchestration backend with an HTTP API.
- ``monoid_agent_kernel.reference.llm_gateway`` — a credential-boundary LLM gateway.
- ``monoid_agent_kernel.reference.web_gateway`` — a web search/fetch/context gateway.

Real integrators build production services against ``monoid_agent_kernel.contracts``
and the wire contracts documented in ``docs/CONTRACTS.md``.

Subpackages are imported explicitly (e.g. ``from monoid_agent_kernel.reference.backend import
RunnerBackend``); this package intentionally performs no eager imports so that importing it
does not pull in the reference HTTP servers.
"""
