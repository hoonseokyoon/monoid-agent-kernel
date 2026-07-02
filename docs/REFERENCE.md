# Reference Services

The Reference services are runnable examples assembled from the Monoid Agent Kernel contract and
Core Helper Kit. They show how backend, LLM gateway, web gateway, MCP gateway, stores, Studio, and
conformance harnesses can be wired together for local smoke testing and documentation.

## Role

- Demonstrate contract wiring with real processes and local storage.
- Dogfood Core Helper Kit modules as they are introduced.
- Provide the baseline implementation for conformance profiles.
- Support Studio and gateway smoke paths for release confidence.

## Reference Packages

- `monoid_agent_kernel.reference.backend`: durable backend and control-plane example.
- `monoid_agent_kernel.reference.llm_gateway`: LLM gateway example and offline provider path.
- `monoid_agent_kernel.reference.web_gateway`: scoped web/search/fetch/context gateway example.
- `monoid_agent_kernel.reference.mcp_gateway`: MCP gateway example for brokered tool integration.
- `monoid_agent_kernel.reference.outbox`: durable senders, including the Reference message-fabric
  adapter that routes external-agent envelopes into peer inboxes.
- `monoid_agent_kernel.reference.stores`: local durable store examples.
- `monoid_agent_kernel.reference.studio`: browser UI and smoke surface.
- `monoid_agent_kernel.reference.conformance`: public Reference harnesses for conformance profiles.

## Conformance Role

The Reference implementation is the first target for each conformance profile. When an operational
rule is added, Reference gets an adapter or smoke path that proves the rule against the bundled example.
External backends can use that adapter shape as a starting point for their own profile harnesses.

`monoid_agent_kernel.reference.conformance` provides that adapter shape as a public Reference
example. `ReferenceConformanceFactory` creates fresh backend, capability, gateway, side-effect, and
message-fabric harnesses for profile assertions, then runs an offline Studio smoke path for
`reference-full`.

The harness keeps profile tests focused on observable behavior. The Reference services keep the
actual backend, gateway, and Studio wiring visible as a runnable example.

## Review Focus

Reference review should identify whether the example assembles the contract and helper kit clearly.
Repeated runtime invariants move into Contract rules, Core Helper Kit modules, and conformance
profiles. `docs/OPERATIONAL_RULE_COVERAGE.md` shows the current mapping from rule ids to Reference
harness cases and tests.
