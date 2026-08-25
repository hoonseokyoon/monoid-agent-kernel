# Production adapter operations

This runbook covers the v0.23 PostgreSQL, S3-compatible ObjectStore, and Temporal hosting path.
The host owns process lifecycle, schedules, credentials, retention policy, alert thresholds, and
operator authorization. The kernel supplies explicit readiness, migration, aggregate telemetry,
fencing, checked-read, inventory, and garbage-collection primitives.

## Ownership and roles

Use separate deployment identities for these responsibilities:

| identity | required responsibility |
|---|---|
| migration | PostgreSQL schema creation, migration DDL, migration metadata writes, and the migration advisory lock |
| runtime | PostgreSQL schema usage plus the table DML required by the configured stores |
| ObjectStore runtime | conditional create, checked head/get, and multipart upload for the configured content prefix |
| ObjectStore admin | bucket head/versioning read, version inventory/delete, multipart inventory/abort, and retention operations |
| Temporal worker | poll the selected Workflow and Activity task queues and operate the per-run Workflow |

Keep the ObjectStore admin identity out of ordinary workers. Keep PostgreSQL migration authority out
of the runtime identity. Supply DSNs, endpoints, bucket names, expected owners, TLS roots, and KMS
identifiers through the host's secret/configuration system.

## Startup and readiness

Run startup in this order:

1. Construct and open `PostgresDatabase`.
2. Run `PostgresMigrations.status()` and `plan()` under the migration identity.
3. Review the exact ordered plan and call `apply()` as an explicit deployment step.
4. Run `PostgresMigrations.doctor()` and require `ok=True`.
5. Call `check_ready()` on every configured PostgreSQL store and `PostgresOperations` before serving
   traffic.
6. Run `S3ObjectStoreAdmin.doctor()` and require bucket reachability plus enabled versioning.
7. Start Temporal Workflow workers, then Activity workers, then admission dispatchers.

Adapter construction never applies migrations. A readiness failure blocks traffic. Doctor errors
carry portable types and exclude DSNs, endpoints, bucket names, expected-owner values, credentials,
and exception prose. PostgreSQL migration status names the configured schema, so keep the complete
PostgreSQL doctor report on a private operator route. The S3 doctor report omits its bucket and
endpoint.

`S3ObjectStoreAdmin.doctor()` performs read-only bucket head and versioning calls. Its
`*_supported` fields describe adapter capabilities. Actual conditional-create, checked-read,
inventory, and version-delete semantics belong in pre-production service qualification.

## Operational snapshots and metrics

`PostgresOperations.snapshot()` opens a `REPEATABLE READ, READ ONLY` transaction and uses the
database clock for one aggregate snapshot. It emits fixed metric names and low-cardinality
`queue`/`state` attributes. It never emits a run ID, owner ID, schema, locator, digest, payload,
checkpoint, prompt, output, reasoning text, or credential.

```python
from monoid_agent_kernel.adapters.postgres import PostgresOperations
from monoid_agent_kernel.hosting import record_operational_snapshot
from monoid_agent_kernel.observability import OtelOperationalMetricSink

operations = PostgresOperations(database)
operations.check_ready()

snapshot = operations.snapshot()
record_operational_snapshot(
    snapshot,
    OtelOperationalMetricSink(meter_provider=host_meter_provider),
)
```

The OpenTelemetry sink retains the latest value for each metric identity behind observable gauges.
The application owns the meter provider, reader, exporter, resource attributes, collection cadence,
and exporter failure policy.

| metric | unit | dimensions / meaning |
|---|---:|---|
| `monoid.postgres.schema.version` | `1` | installed migration ordinal |
| `monoid.postgres.authority.count` | `1` | `state=total|active|expired|revoked` |
| `monoid.postgres.authority.seconds_to_next_expiry` | `s` | next active expiry; combine with active count |
| `monoid.postgres.outbox.count` | `1` | `queue=activation|model_evidence`, bounded delivery state |
| `monoid.postgres.outbox.max_attempts` | `1` | maximum actionable-row attempt count by queue |
| `monoid.postgres.outbox.oldest_age` | `s` | oldest actionable row age by queue |
| `monoid.postgres.invocation.count` | `1` | current head by `reserved|dispatch_started|settled|unknown` |
| `monoid.postgres.stream.head.count` | `1` | current head by `open|sealed` |
| `monoid.postgres.stream.current_bytes` | `By` | current head bytes for `all|open` |
| `monoid.postgres.stream.oldest_open_age` | `s` | oldest current open generation age; reset receipt anchors generation start |
| `monoid.postgres.stream.chunk.count` | `1` | retained chunk rows across generations |
| `monoid.postgres.stream.chunk.bytes` | `By` | retained chunk bytes across generations |
| `monoid.postgres.object.count` | `1` | metadata by `available|deleted` |
| `monoid.postgres.object.bytes` | `By` | available object metadata bytes |
| `monoid.postgres.object.association.count` | `1` | run/object associations |
| `monoid.postgres.object.orphan_metadata.count` | `1` | available metadata with no run association |
| `monoid.postgres.object_gc.receipt.count` | `1` | cumulative receipt class `deleted|skipped|precondition_failed` |

Each collection performs exact aggregate queries. PostgreSQL statement timeout bounds every query.
Schema compatibility, schema version, and all aggregates come from one `REPEATABLE READ, READ ONLY`
transaction. `PostgresDatabase.read_snapshot()` captures the first setup statement's database time
and returns it with the connection. Operations uses that boundary as `collected_at` and anchors all
time predicates to it. Outbox lag and maximum-attempt signals include pending rows available at
collection time plus rows whose lease has expired; delayed rows and active leases do not contribute.
Choose a cadence that matches table volume, retain collection latency in host telemetry, and move
high-frequency product dashboards to host-owned projections.

Alert on sustained signals and service objectives. Useful signals include pending age growth,
repeated outbox attempts, any unknown invocation, imminent lease expiry with active work, long-lived
open streams, orphan metadata growth, and GC precondition failures. A zero next-expiry value means
there is no future active expiry or the next expiry is already due; interpret it with authority
counts.

## Migration, rolling deploy, and rollback

Every installed migration records `reader_floor` and `writer_floor`. A binary calls
`require_reader_compatible()` or `require_writer_compatible()` through its store readiness check.
Unknown forward migrations remain usable only while their declared floors stay at or below the
binary's `SCHEMA_VERSION`.

Use this rolling sequence:

1. Back up PostgreSQL and record the ObjectStore versioning state.
2. Run `status()` and archive the installed migration IDs, checksums, reader floor, and writer floor.
3. Deploy code that can read the current schema while old writers remain active.
4. Apply reviewed additive migrations with one migration job. The advisory lock serializes
   concurrent migration attempts.
5. Run doctor and readiness from both the old and new binary sets.
6. Start new workers and dispatchers. Drain old Activity workers within their configured graceful
   shutdown bound.
7. Confirm outbox age, unknown invocations, active leases, stream heads, and error rates.
8. Remove old binaries after their compatible-reader window closes.

A migration that raises the writer floor requires old-writer drain before apply. A migration that
raises the reader floor requires complete old-binary removal before apply. Rollback selects a binary
whose reader and writer floors accept the installed schema. The runtime leaves migration rows and
schema objects intact; restoration uses the reviewed database backup when the target binary cannot
accept the forward schema.

## Drain, lease expiry, and takeover

Stop admission dispatch before a planned global drain. Ask `TemporalWorkerGroup` to close with its
bounded graceful timeout. The Activity requests `graceful_drain`, settles a canonical receipt where
possible, and releases or revokes its writer lease.

Inspect one known run through `PostgresWriterAuthorityStore.read(run_id)`. The returned authority
contains current generation, owner, expiry, revocation, and active status. Keep this run-scoped
inspection on an authorized private operator route.

A crashed worker retains authority until release, revocation, or database-clock expiry. A
replacement claims a higher generation after that boundary. Stale generations fail every durable
mutation. Preserve the same invocation idempotency key during recovery. A `dispatch_started` or
`unknown` invocation blocks automatic paid-provider retry until reconciliation establishes a safe
result.

A retryable Temporal Activity failure receives the configured attempts within one retry batch. An
exhausted batch remains in the same per-run Workflow, exposes
`temporal_activity_retry_exhausted` through the public-safe status, and redrives the same admitted
command after five seconds. The Workflow uses a distinct Activity ID for each redrive and rolls over
with the command restored to the pending set when Temporal suggests Continue-As-New or 100 batches
have exhausted. Investigate repeated exhaustion as an infrastructure incident; keep the Workflow
and PostgreSQL admission records intact while restoring the Activity worker or its dependencies.

## Object inventory, garbage collection, and multipart cleanup

Keep bucket versioning enabled. `S3ObjectStoreAdmin.inventory_page()` returns bounded content-address
entries with exact version-aware delete tokens. `PostgresObjectGarbageCollector.plan()` is the
dry-run boundary. Archive and review its plan before `apply()`.

`apply()` rechecks the object generation and run association under the digest lock, performs the
version-aware delete, and records a receipt. Concurrent association or recreation converges to a
skip or precondition-failed receipt. A GC scheduler and retention policy belong to the host.

List incomplete multipart uploads with `incomplete_multipart_page()`. Abort only uploads older than
the host's reviewed retention boundary. Record the upload identity and result in the private
operator audit log. Keep multipart cleanup independent from object GC.

## Backup and restore

PostgreSQL is canonical for authority, checkpoints, invocation state, admission, receipts, terminal
state, object associations, and stream metadata. ObjectStore holds immutable bytes referenced by
PostgreSQL. Temporal holds orchestration history and content-free references.

Create a consistent recovery point as follows:

1. Stop admission and drain writers, or establish a host-owned write barrier.
2. Capture the PostgreSQL backup and its database timestamp.
3. Preserve every ObjectStore version reachable at that PostgreSQL cut. Keeping additional immutable
   objects is safe; GC can classify them after restore.
4. Back up Temporal namespace state according to the selected Temporal deployment policy.
5. Record image/SDK versions, schema version, bucket versioning status, and backup identifiers in the
   private operator ledger.

Restore ObjectStore versions before PostgreSQL. Start with admission and workers disabled. Run
PostgreSQL doctor, every store readiness check, S3 doctor, sampled checked reads, stream digest
verification, and a GC dry run. Start Temporal workers before dispatchers. Reconcile pending and
unknown invocations before enabling paid calls.

## Corruption response

Stop new work for the affected run and preserve the PostgreSQL rows, ObjectStore versions, Temporal
history, and relevant public telemetry. Reproduce the failure with the checked reader that detected
it. Missing, size-mismatched, digest-mismatched, future-version, and malformed records remain typed
failures.

Keep automatic repair disabled. Restore a known version or rebuild metadata only through an
authorized, reviewed incident procedure with immutable before/after evidence. Re-run checked reads
and the affected conformance profile before releasing the run. Treat unknown paid-call state as a
reconciliation case and keep automatic provider retry at zero.

## Security and privacy checks

- Restrict runtime and admin PostgreSQL roles to the selected schema.
- Require TLS and certificate validation for remote PostgreSQL, ObjectStore, and Temporal endpoints.
- Use bucket policy and IAM to enforce the runtime/admin split and the configured prefix.
- Configure server-side encryption and key policy through the host. Doctor reports only whether
  encryption was configured on the adapter.
- Export only `OperationalSnapshot`, doctor reports, and reviewed public events to routine telemetry.
- Keep run-scoped authority inspection, ObjectStore inventory tokens, GC plans, and backup IDs on
  private operator routes.
- Scan Temporal history, receipts, public events/logs/traces, operational snapshots, and doctor
  reports for seeded private markers and credentials during qualification.

The actual-service campaign runs PostgreSQL 16 and 18, versioned MinIO, and a pinned Temporal local
server together. It covers response loss, duplicate delivery, process kill/takeover, paid-call
ambiguity, durable stream reconnect, terminal races, migration floors, checked corruption, and
public-surface privacy. See [the v0.23 release audit](V0_23_RELEASE_AUDIT.md) for the exact release
commit and CI evidence.
