# Production Security Checklist

Use the deployment and ownership paths in [the production embedding handbook](../EMBEDDING.md)
alongside this checklist.

Work through this before running Monoid outside local development. It is the
actionable form of the integrator responsibilities in
[SECURITY_MODEL.md](SECURITY_MODEL.md); the rationale for each item is in
[THREAT_MODEL.md](THREAT_MODEL.md).

Items marked **(default is unsafe)** flag places where the local-development
default must be changed for production.

## Gateway and credentials

- [ ] Provider keys (OpenAI/Anthropic/Brave/…) live only in the gateway/edge
      secret manager — never in kernel env or run config.
- [ ] The kernel process has no provider API keys; the direct provider adapter
      (`--allow-direct-provider-api`) is **not** enabled in production.
- [ ] `MONOID_BACKEND_TOKEN_SECRET` is 32+ random bytes and not shared with
      non-gateway services.
- [ ] Gateway tokens are short-lived and scoped; signing keys support rotation
      (`kid`) and revocation (token-id + issued-before).

## Workspace and files — **(default is unsafe)**

- [ ] Hosted runs use `mode="propose"` with an isolated `overlay` or `staging` workspace backend;
      `mode="apply"` is reserved for explicitly privileged paths.
- [ ] A deny/redact policy is set per run — there is no secure default. At
      minimum deny `.env`, `*.key`, `*.pem`, `**/id_rsa`, `.ssh/**`, `.git/**`.
- [ ] The workspace root is per-tenant isolated; no host-sensitive directory is
      mounted as a workspace.
- [ ] The `Workspace` backend passes `tests/test_workspace_contract.py`, and its
      symlink behavior is documented/tested.

## Tool surface

- [ ] Tool bindings are explicit; no unintended registry tool is exposed.
- [ ] Dangerous tools use `authorization="ask"` or `"deny"`.
- [ ] `shell.exec` is bound only where required, with `command_allow_prefixes`,
      an `env_allowlist`, and `max output bytes` + timeout set.
- [ ] Web tools have domain allowlists and byte/time caps at the Web gateway.
- [ ] Custom side-effect tools declare their delivery semantics (durable outbox
      or explicit idempotency).
- [ ] Capability-gated tools run behind a `CapabilityBroker` that fails closed;
      `--auto-grant-capabilities` is **not** used in production.
- [ ] Each run selects exactly one activation authority and proves fencing, per-run input ordering,
      idempotent receipts, admission limits, and credential-free durable records.
- [ ] Deployments derived from the Reference inbox assembly share durable checkpoint and lease
      stores plus one transactional command store across instances, enable owner watchdogs, set
      queue limits, isolate run roots and database access by tenant, and monitor persisted command
      rows for credential leakage. The bundled SQLite stores are a single-host Reference fixture.
- [ ] Experimental Reference profiles stay within their documented scope. The Reference inbox
      assembly and optional DBOS activation-recovery profile run as mutually exclusive
      activation-authority compositions for a run. Inside the DBOS profile, one private runtime
      host owns hosted control and run lifecycle together. The profile excludes `LeaseStore`,
      `CommandStore`, `RecoveryService`, and watchdog lifecycle ownership. Production qualification
      remains a future milestone after v0.19.2.

## Events, artifacts, and logs

- [ ] The public event stream is redacted; a redacting event sink masks
      secret-bearing tool args / shell commands.
- [ ] Run directories, `transcript.jsonl`, and checkpoints are access-controlled
      and not served publicly.
- [ ] Only runtime event and metadata owners can write `run_root`. Tool workspaces,
      MCP servers, and untrusted processes cannot modify committed `events.jsonl`
      prefixes; the Reference warm offset index relies on this append-only boundary.
- [ ] Retention policy for private artifacts is defined.
- [ ] Application logs and OTel exporters do not carry bearer tokens or lease
      material.

## Skills, MCP, and memory

- [ ] Skill bundles are treated as code — reviewed or signed before loading.
- [ ] MCP servers are pinned and allowlisted; untrusted servers are not enabled.
- [ ] Memory retention/deletion policy is defined per tenant.

## Subagents and side effects

- [ ] Child tool surfaces are minimal; depth/fan-out limits match your budget.
- [ ] Outbox senders use idempotency keys for non-idempotent targets.
- [ ] Retry/backoff and dead-letter behavior is monitored; recovery paths are
      tested.

## Conformance

- [ ] The conformance profiles relevant to your runtime pass (see
      [CONFORMANCE.md](../CONFORMANCE.md)).
- [ ] `provider-gateway` passes for your gateway; `capability-security` passes for
      your broker; the `Workspace` contract suite passes for your backend.
