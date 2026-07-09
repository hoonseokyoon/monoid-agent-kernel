# Threat Model

Monoid is a **pre-1.0 building block**, not a turnkey secure deployment. It is a
kernel you embed inside a product you operate. Whether a deployment is safe
depends as much on how you wire the gateways, permission policy, and event sinks
as on the kernel itself.

This document states, for the surfaces the kernel exposes, **what the kernel
defends by design** and **what is the integrator's responsibility**. If a row
below is an integrator responsibility, the kernel does not protect you from it
out of the box.

For how to report a vulnerability, see [SECURITY.md](../SECURITY.md).

## Trust boundaries at a glance

```
 model (LLM)  ──►  LLM gateway  ──►  kernel  ──►  tools / shell / workspace
                    (your creds)      (no creds)    (your data)
                                        │
 web providers ──► Web gateway ────────┘
                    (your creds)
```

- The **kernel process** never holds provider API keys. It calls your LLM and
  Web gateways with short-lived, scoped tokens.
- Your **gateway processes** are the credential boundary. Keys (OpenAI,
  Anthropic, Brave, …) live there and nowhere else.
- The **workspace** is tenant data the model can read and mutate through tools.
  It is inside the trust boundary of the run, not outside it.

## Permissive by default

> ⚠️ **The kernel does not protect workspace secrets unless you tell it to.**

Path permission defaults are **permissive**: the kernel treats every
root-contained file as a normal workspace file — **including dotfiles, `.env`,
and private keys**. There is **no** default deny or redact policy.

This is intentional (the kernel does not guess at your secret layout), but it
means:

> If a run's workspace contains secrets, and you do not pass a deny/redact
> policy, the model can read those secrets and they can flow into tool arguments,
> shell commands, and the model provider.

Mitigate per run with `--deny-path` / `--redact-path`, or a
`--permission-policy-file`:

```json
{
  "deny_patterns": [".env", "*.key", "**/id_rsa"],
  "redact_patterns": ["internal/**"]
}
```

`deny_patterns` blocks tool and shell access; `redact_patterns` only masks paths
in the public event/status stream (private run artifacts keep real paths and
contents). See [CLI.md](CLI.md#path-permissions).

## Threat-by-threat

| Threat | Kernel defense | Integrator responsibility |
|--------|----------------|---------------------------|
| **Provider key theft** | Kernel never receives provider keys; they stay in your gateway. | Secure the gateway process, its env, and its logs. |
| **Gateway / run token theft** | Tokens are short-lived and scoped; reference tokens carry `kid` and support rotation + revocation. | Set a strong `MONOID_BACKEND_TOKEN_SECRET`; rotate keys; keep token TTLs short. |
| **Token scope widening** | Payload-level domain/binding/call-limit values can only *narrow* a signed scope, never widen it. | Issue minimally-scoped tokens per run. |
| **Capability lease replay / secret leak** | Leases carry handles (`token_ref`), never raw secrets; the edge resolves them; leases are scoped and short-lived. | Implement a `CapabilityBroker` that fails closed; do not auto-grant in production (`--auto-grant-capabilities` is dev-only). |
| **Workspace secret leakage** | `redact_patterns` masks paths in the public stream; file contents are kept out of `events.jsonl`. | Provide deny/redact policy; add a redacting event sink for secret-bearing tool args / shell commands. |
| **Path traversal / escape** | Workspace access is mediated through the `Workspace` contract, rooted at the run workspace. | Use a workspace backend whose contract suite passes; do not mount host-sensitive roots as the workspace. |
| **Prompt injection from files / web** | Kernel isolates runs and keeps provider egress behind the gateway. | Treat model output as untrusted; gate side-effecting tools behind `ask`/approval and capability leases. |
| **Shell command injection / abuse** | `shell.exec` is only available when explicitly bound; background jobs are run-scoped. | Bind shell only when needed; sandbox the run's execution environment; apply deny policy. |
| **Malicious tool / skill / MCP module** | Tools, skills, and MCP servers are opt-in via explicit flags/config, off by default. | Only load modules you trust; review `SKILL.md` bundles and MCP endpoints before enabling. |
| **Subagent fan-out abuse** | `agent.spawn` is off unless `--agents-directory` is set; depth/fan-out caps apply. | Keep child tool surfaces minimal; set limits appropriate to your budget. |
| **Event / transcript retention** | `events.jsonl` is public/redacted; `transcript.jsonl` is private with full payloads. | Protect the run directory; the private transcript is not for public exposure. |
| **Public vs private artifacts** | Proposed file contents are exposed only via the run directory or run-token-protected proposal APIs. | Do not serve the run directory publicly; gate proposal APIs behind run tokens. |
| **Tenant isolation** | Runs are per-run isolated; usage endpoints are admin-scoped. | Enforce tenant separation at the backend and storage layer; the reference backend is an example, not a hardened multi-tenant service. |

## What is explicitly out of scope

- **`reference/*` is example code.** The reference backend, gateways, Studio, and
  stores demonstrate wiring; they are not hardened production services. Build
  production services against the contracts in [CONTRACTS.md](CONTRACTS.md).
- **Sandboxing the shell / execution host.** The kernel does not containerize or
  jail shell execution. Run it in an environment you have already sandboxed.
- **Network egress control beyond the gateway seam.** The kernel does not perform
  direct web egress, but it cannot stop a misconfigured gateway from doing so.

If you are evaluating Monoid for a sensitive deployment and have questions about
these boundaries, reach out via the contact in [SECURITY.md](../SECURITY.md).
