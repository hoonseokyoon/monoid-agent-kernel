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
      minimum deny `.env`, `*.key`, `*.pem`, `**/id_rsa`, `.ssh`, `.git`.
      The bare directory names cover each node and its full subtree, including nested copies.
      Patterns use gitignore-style wildcard syntax — see [CLI.md](../CLI.md#path-permissions).
      **On < v0.20 the previous `.ssh/**` / `.git/**` list left holes**: `**` behaved as a
      single `*`, missing deep entries, and those spellings did not protect the directory node from
      recursive move/delete. `**/id_rsa` also missed a bare `id_rsa` at the workspace root.
      Re-check any policy written against an earlier version.
- [ ] The workspace root is per-tenant isolated; no host-sensitive directory is
      mounted as a workspace.
- [ ] The `Workspace` backend passes `tests/test_workspace_contract.py`. Its case and Unicode
      normalization aliases, symlink behavior, and hardlink behavior are documented and tested
      against every supported volume, including case-sensitive directories and custom/network
      filesystems. The path-pattern matcher is lexical; the backend, including the built-in local
      backend on case- or normalization-insensitive volumes, must canonicalize or reject any other
      spelling that resolves to the same filesystem object before policy evaluation.
- [ ] Recursive copy, move, delete, archive, and similar tree operations preflight every descendant
      against deny rules, or are disabled for trees that can mix allowed and denied paths. The
      built-in recursive file tools currently present only the operation root to the policy, so an
      allowed ancestor does not protect a denied descendant by itself.

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
      and not served publicly. The same applies to the opt-in private sidecars when
      enabled — `model-content.jsonl`, `model_calls.jsonl`, and `model_payloads.jsonl`
      with its `model_payloads/` directory. Two of those three carry content
      (`model_calls.jsonl` carries metadata only), and no projection, hydration
      path or HTTP route serves the replay corpus, so filesystem access is the
      only control over it (`monoid validate` and `monoid gc` read it in place).
- [ ] Only runtime event and metadata owners can write `run_root`. Tool workspaces,
      MCP servers, and untrusted processes cannot modify committed `events.jsonl`
      prefixes; the Reference warm offset index relies on this append-only boundary.
- [ ] Retention policy for private artifacts is defined. Nothing in the kernel
      deletes them: `monoid gc` collects only chunks no record resolves, so a healthy
      replay corpus grows with the conversation until you remove it.
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
