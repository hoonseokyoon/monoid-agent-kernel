# Backend and gateway walkthrough (reference)

> Reference example (`monoid_agent_kernel.reference.backend`). Build production
> backends against the contracts in [CONTRACTS.md](CONTRACTS.md). See
> [REFERENCE.md](REFERENCE.md) for the reference role and smoke targets.

This walkthrough wires the reference backend, LLM gateway, and Web gateway
together so you can create a run over HTTP with a real token boundary. Provider
credentials live only in the gateway processes; the kernel never receives them.
For the security rationale, see [THREAT_MODEL.md](THREAT_MODEL.md).

The reference backend issues run tokens, starts kernel runs, and exposes lifecycle,
result, event, and tenant usage APIs. Lifecycle payloads use `state` plus `terminal`;
ready result payloads keep `status` for the terminal `AgentRunResult.status`.
Provider API keys stay outside the Monoid backend.

## Start the LLM gateway (credential boundary)

This process is the provider-credential boundary:

```bash
export MONOID_BACKEND_ADMIN_TOKEN="admin-dev-token"
export MONOID_LLM_GATEWAY_ADMIN_TOKEN="llm-admin-dev-token"
export MONOID_BACKEND_TOKEN_SECRET="replace-with-32-plus-random-bytes"

monoid llm-gateway serve \
  --host 127.0.0.1 \
  --port 8080
```

## Start the backend

Start the Monoid backend in another process. It shares the token signing secret
with the LLM and Web gateways so it can issue scoped gateway tokens.

Reference gateway tokens include a `kid` header. The shared `TokenManager` supports keyring-based
rotation with a grace window plus token-id and issued-before revocation checks.

```bash
monoid backend serve \
  --workspace-root /workspaces \
  --run-root ./runs \
  --llm-gateway-url http://127.0.0.1:8080/internal/llm/turns \
  --web-gateway-url http://127.0.0.1:8090
```

## Start the Web gateway

For local contract testing, start the reference fake WebGateway:

```bash
export MONOID_WEB_GATEWAY_ADMIN_TOKEN="web-admin-dev-token"

monoid web-gateway serve \
  --host 127.0.0.1 \
  --port 8090 \
  --provider fake
```

For a real search smoke, use Brave Search for `web.search` and the gateway's
direct HTTP fetcher for `web.fetch`. Add `--context-provider brave-llm` to use
Brave's LLM Context endpoint for `web.context`, or `--context-provider
search-fetch` to build context from the configured search/fetch providers.
Provider credentials stay in the WebGateway process and are never passed to Monoid:

```bash
export BRAVE_SEARCH_API_KEY="..."

monoid web-gateway serve \
  --host 127.0.0.1 \
  --port 8090 \
  --provider brave-http \
  --context-provider brave-llm \
  --brave-api-key-env BRAVE_SEARCH_API_KEY
```

## Create a run

```bash
curl -sS -X POST http://127.0.0.1:8765/v1/runs \
  -H "Authorization: Bearer $MONOID_BACKEND_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "tenant_a",
    "user_id": "user_a",
    "workspace_root": "/workspaces/demo",
    "instruction": "Read notes.md and create SUMMARY.md.",
    "mode": "propose",
    "runtime_config": {
      "definition_id": "markdown-editor",
      "config_version": 1,
      "model": {"provider": "gateway", "model": "gpt-5.5"},
      "tools": [
        {"binding_id": "read_file", "ref": {"kind": "registry", "tool_id": "fs.read"}},
        {"binding_id": "write_file", "ref": {"kind": "registry", "tool_id": "fs.write"}},
        {"binding_id": "finish", "ref": {"kind": "registry", "tool_id": "run.finish"}}
      ],
      "tool_search": {"enabled": true, "top_k": 5}
    }
  }'
```

The response includes a `run_token`. Use that token for:

```bash
curl -H "Authorization: Bearer $RUN_TOKEN" \
  http://127.0.0.1:8765/v1/runs/$RUN_ID/status

curl -H "Authorization: Bearer $RUN_TOKEN" \
  http://127.0.0.1:8765/v1/runs/$RUN_ID/result

curl -H "Authorization: Bearer $RUN_TOKEN" \
  http://127.0.0.1:8765/v1/runs/$RUN_ID/events

curl -H "Authorization: Bearer $RUN_TOKEN" \
  "http://127.0.0.1:8765/v1/runs/$RUN_ID/events?from_seq=1&limit=100"

curl -H "Authorization: Bearer $RUN_TOKEN" \
  "http://127.0.0.1:8765/v1/runs/$RUN_ID/diagnostics?event_limit=50"

curl -H "Authorization: Bearer $RUN_TOKEN" \
  http://127.0.0.1:8765/v1/runs/$RUN_ID/proposal

curl -H "Authorization: Bearer $RUN_TOKEN" \
  http://127.0.0.1:8765/v1/runs/$RUN_ID/proposal/files/SUMMARY.md

curl -H "Authorization: Bearer $RUN_TOKEN" \
  http://127.0.0.1:8765/v1/runs/$RUN_ID/runtime-config

# POST replaces the run's config (optimistic concurrency via expected_version); the kernel
# applies it at the next turn boundary. See CONTRACTS.md for the request schema.
curl -sS -X POST http://127.0.0.1:8765/v1/runs/$RUN_ID/runtime-config \
  -H "Authorization: Bearer $RUN_TOKEN" \
  -H "Content-Type: application/json" \
  -d @new-runtime-config.json

curl -H "Authorization: Bearer $RUN_TOKEN" \
  http://127.0.0.1:8765/v1/runs/$RUN_ID/jobs

curl -H "Authorization: Bearer $RUN_TOKEN" \
  http://127.0.0.1:8765/v1/runs/$RUN_ID/jobs/$JOB_ID/logs?stream=stdout
```

`/status` returns lifecycle state, for example `{"state":"running","terminal":false}`.
`/result` returns `ready=false` with lifecycle state while a run is open; when `ready=true`,
its `status` field is the terminal result status (`completed`, `failed`, or `limited`).

## Usage endpoints

Tenant usage is admin-scoped:

```bash
curl -H "Authorization: Bearer $MONOID_BACKEND_ADMIN_TOKEN" \
  http://127.0.0.1:8765/v1/tenants/tenant_a/usage
```

The backend generates a separate `llm_gateway` token for the kernel-to-gateway
call. That token is passed only to `GatewayModelAdapter` and is not returned from
the run APIs. For web-enabled runs, it also generates a separate `web_gateway`
token for `WebGatewayClient`.

The LLM gateway validates `llm_gateway` tokens, calls the provider adapter, and returns only
opaque `turn_handle` values to the kernel. The default by-value `messages` request is
forwarded statelessly; for handle-based continuation it stores provider continuation ids
server-side. The turn request carries the effective model from runtime config. Its usage endpoint is
admin-scoped:

```bash
curl -H "Authorization: Bearer $MONOID_LLM_GATEWAY_ADMIN_TOKEN" \
  http://127.0.0.1:8080/internal/llm/tenants/tenant_a/usage
```

The WebGateway validates `web_gateway` tokens, enforces signed token scope for brokered web
capabilities before calling a provider, and reports tenant usage. Payload-level domain, binding,
and call-limit values can narrow the signed scope; they cannot widen it. The reference ships a
deterministic fake provider plus Brave-backed search/fetch/context providers behind the
provider-neutral `ContextProvider` seam, so the search backend can be swapped without
changing kernel tools.

```bash
curl -H "Authorization: Bearer $MONOID_WEB_GATEWAY_ADMIN_TOKEN" \
  http://127.0.0.1:8090/internal/web/tenants/tenant_a/usage
```
