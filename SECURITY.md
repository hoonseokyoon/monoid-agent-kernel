# Security Policy

## Reporting a vulnerability

Please report security issues **privately**. Do not open a public issue for a
vulnerability.

- Use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  (the **Security → Report a vulnerability** tab), or
- email **hoonseok.ai@gmail.com** with details and, if possible, a reproduction.

We aim to acknowledge reports within a few days. Please give us a reasonable
window to address the issue before public disclosure.

## Scope and design notes

This project is a **pre-1.0 research package** (`0.x`). Treat it as a building
block, not a hardened product, and review the security model before deploying it.

Key boundaries the design relies on (see `README.md` and `docs/CONTRACTS.md`):

- **Provider credentials stay outside the runner.** The default `GatewayModelAdapter`
  talks to a gateway you operate; the runner does not receive OpenAI/Anthropic/Brave keys.
- **Secrets never enter the core.** Capability leases carry handles (`token_ref`),
  never raw secrets; the edge resolves them.
- **Public event streams are not heuristically scrubbed.** The core keeps file
  contents out of the public stream and honors `redact_patterns`, but redacting
  secret-bearing tool arguments or shell commands is the integrator's responsibility
  (see the Event Sinks section of the README).
- **`reference/*` is example code, not a supported/hardened surface.** Build your own
  services against the contracts for production use.

If you are evaluating this for a sensitive deployment and have questions about the
trust boundaries, reach out via the contact above.
