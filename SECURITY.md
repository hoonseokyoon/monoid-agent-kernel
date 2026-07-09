# Security Policy

## Reporting a vulnerability

Please report security issues **privately**. Do not open a public issue for a
vulnerability.

- Use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  (the **Security → Report a vulnerability** tab), or
- email **yoonhs9dev@gmail.com** with details and, if possible, a reproduction.

We aim to acknowledge reports within a few days. Please give us a reasonable
window to address the issue before public disclosure.

## Scope and design notes

This project is a **pre-1.0 agent kernel** (`0.x`). Treat it as a building
block and review the security model before deploying it.

The `docs/security/` cluster covers this in depth:
[SECURITY_MODEL.md](docs/security/SECURITY_MODEL.md) (intended boundaries and
verified invariants), [THREAT_MODEL.md](docs/security/THREAT_MODEL.md)
(threat-by-threat: what the kernel defends vs. the integrator's responsibility),
and [PRODUCTION_CHECKLIST.md](docs/security/PRODUCTION_CHECKLIST.md) (pre-deploy
steps).

Key boundaries the design relies on (see `README.md` and `docs/CONTRACTS.md`):

- **The default is permissive by design.** Path permission defaults treat every
  root-contained file — including dotfiles, `.env`, and keys — as a normal
  workspace file. There is no default deny/redact policy. If a run's workspace
  holds secrets, pass a deny/redact policy or the model can read them. See
  [Threat Model → permissive by default](docs/security/THREAT_MODEL.md#permissive-by-default).
- **Provider credentials stay outside the kernel.** The default `GatewayModelAdapter`
  talks to a gateway you operate; OpenAI/Anthropic/Brave keys stay in that gateway.
- **Secrets never enter the core.** Capability leases carry handles (`token_ref`),
  never raw secrets; the edge resolves them.
- **Public event streams keep file contents out by design.** The core honors
  `redact_patterns`; integrators own extra redaction for secret-bearing tool arguments
  or shell commands (see [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md#event-sinks)).
- **`reference/*` is example code.** Build production services against the contracts.

If you are evaluating this for a sensitive deployment and have questions about the
trust boundaries, reach out via the contact above.
