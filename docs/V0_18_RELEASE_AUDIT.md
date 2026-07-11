# v0.18.0 Release Audit

Status: completed for the release-hardening pull request.

The audit froze this range before fixes:

- release base: `4cbec7508de4301a64ee66e79e584b867335cd03` (`main`, v0.17.1);
- initial candidate: `95f1a0a532724219fe30dd711c148828c04fa300`;
- implementation base: `db8361cfce96d4d8b9bc625418266b2c3745fb6c` (`develop` after PR 32).

## Coverage

| Perspectives | Audit owner |
|---|---|
| Architecture/boundaries; code quality/maintainability | Red-team DBOS-direction agent |
| Correctness/state machines; durability/migration | v0.18 goal-recheck agent |
| Security/isolation; tests/conformance | DBOS-boundary repository-delta agent |
| Developer experience/docs; release readiness | Root release auditor |

The agents inspected the frozen range independently. Each finding included severity, concrete code
or runtime evidence, impact, and a recommended action. A second agent challenged every initial P1
finding before implementation.

## Findings and dispositions

| ID | Severity | Finding | Disposition |
|---|---:|---|---|
| `COR-01` | P1 | Consuming an approval-replay batch before its handlers discarded every unstarted tail item after a process loss. | Addressed: the loop consumes and checkpoints one durable head at a time and carries prior observations into the next barrier. |
| `DUR-01` | P1 | Checked LocalFS/SQLite reads accepted checkpoint payloads whose embedded run ID or sequence disagreed with the lookup key and committed pointer. | Addressed: store readers, the generic checked adapter, and recovery bind both identities and classify mismatch as `corrupt`. |
| `DUR-02` | P1 | Current-version checkpoint and run-metadata readers accepted invalid field types and recovery invariants as `loaded`. | Addressed: structural validators cover recovery-critical scalars, containers, identities, active input, receipts, metadata config, and hashes. |
| `COR-02` | P2 | Native async model calls and streams could delay cancellation and the run deadline indefinitely. | Addressed: native model tasks race cancellation/deadline and receive bounded cleanup; the sync-thread boundary is documented. |
| `DUR-03` | P2 | A failed local metadata update could leave shared and local authorities at different versions. | Addressed: additive `metadata_generation` selects and repairs the newer copy; equal-generation divergence is `corrupt`; legacy copies retain local authority. |
| `SEC-01` | P2 | JSON coercion of bytes or custom objects could reintroduce a bearer after the legacy inbox's first redaction pass. | Addressed: durable arguments use redact → sanitize → redact while owner-local execution keeps the transient payload. |
| `SEC-02` | P2 | External conformance reports and stderr copied raw harness exception bodies into retained artifacts and CI logs. | Addressed: JSON, JUnit, stdout, and stderr retain a safe built-in exception category with the body redacted. |
| `TST-01` | P2 | The reusable checkpoint-store contract never reopened its factory, so a process-local store could pass the durability rules. | Addressed: checkpoint, blob, deletion, and isolation assertions now cross fresh store instances. |
| `TST-02` | P2 | Release guidance implied that the v0.18 external CLI could run store, broker, and every profile contract. | Addressed: docs scope the CLI to `minimal-agent` and name the direct store/broker contract functions. |
| `OPS-01` | P2 | A timed-out watchdog stop cleared its live thread handle and allowed a second operational thread to start. | Addressed: start/stop ownership is locked, a live timed-out handle remains registered, and stop returns its bounded result. |
| `COMP-01` | P2 | Experimental DBOS record constants appeared to fall under the stable Core compatibility inventory. | Addressed: compatibility docs classify them as optional Reference operational records with no v0.18 rolling-reader guarantee. |
| `DX-01` | P2 | README and responsibility-map text still described xdist and coverage as advisory. | Addressed: commands and prose match the required v0.18 CI shards and coverage floor. |
| `DX-02` | P2 | CI named the optional DBOS proof as a full Reference lifecycle. | Addressed: job and install-step names use experimental activation-recovery terminology. |
| `ARC-01` | P2 | The control and run DBOS proofs each own an isolated runtime; broader adoption could create parallel lifecycle authorities. | Deferred to [issue 39](https://github.com/hoonseokyoon/monoid-agent-kernel/issues/39). v0.18 keeps both proofs optional and excludes Reference/Studio migration. |
| `PERF-01` | P2 | Reference JSONL subscription polling rescans the event file from the beginning for every page. | Deferred to [issue 37](https://github.com/hoonseokyoon/monoid-agent-kernel/issues/37). v0.18 preserves correct cursor semantics; indexed tail I/O needs a separate storage design. |
| `TST-03` | P3 | External reports lack target provenance and source-evidence digests, so they remain trusted-harness attestations. | Deferred to [issue 38](https://github.com/hoonseokyoon/monoid-agent-kernel/issues/38) with report-version and secret-safety acceptance criteria. |
| `TST-04` | P3 | The threaded SQLite command-store race ran in the parallel contract shard. | Addressed: the module is now an enforced serial contract. |

No P0 finding was reported.

## P1 challenge results

| Finding | Second-pass result |
|---|---|
| `COR-01` | Confirmed P1 with a two-replay crash reproduction; the second approved effect disappeared from the committed checkpoint. |
| `DUR-01` | Confirmed P1 independently on LocalFS and SQLite with embedded `run_id` and `seq` tampering. |
| `DUR-02` | Confirmed P1: malformed `terminal`, `seq`, run identity, applied-input ledger, and runtime config reached recovery as `loaded`. |
| `SEC-01` | Technical defect confirmed; exploitability review lowered it to P2 because the reproducer uses the in-process Python surface. The explicit credential-at-rest invariant required the fix. |
| `COR-02` | Behavior confirmed and severity lowered to P2. The accepted v0.18 async workstream requires consistent cancellation/deadline handling, so the fix remains in scope. |

## Hardening validation

- Agent durability/recovery suite: `153 passed`.
- Agent Reference/security/conformance suite: `37 passed`.
- Agent replay/async suite: `9 passed`.
- Independent integrated focused suite: `283 passed in 61.37s`.
- Full repository suite: `1279 passed, 6 skipped in 404.90s`.
- `python -m ruff check .`: passed.
- `python -m compileall -q src tests examples`: passed.
- `git diff --check`: passed.

The frozen candidate also produced a `0.18.0` wheel and source distribution, passed minimal-wheel
imports and CLI smoke, and passed all four installed-wheel external conformance rules. Final clean
minimal/all-extras artifact validation runs from merged `develop` and belongs in the release pull
request evidence.

Final clean validation found workspace-local `.tmp/` scratch files in the source distribution.
The release follow-up explicitly excludes that root and adds a packaging-configuration regression
test; the rebuilt source distribution receives the same member and metadata checks.

Installed-wheel validation also found that minimal Studio selected its optional async gateway
transport unconditionally. The release follow-up keeps live token deltas behind `[http-async]`,
falls back to complete one-shot gateway turns in the minimal install, and runs Studio acceptance
from the installed minimal package in CI.
