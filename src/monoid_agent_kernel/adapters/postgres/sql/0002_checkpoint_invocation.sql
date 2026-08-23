CREATE TABLE __MONOID_SCHEMA__.bytea_blob (
    sha256 character(64) PRIMARY KEY CHECK (
        sha256 ~ '^[0-9a-f]{64}$'
    ),
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    content bytea NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    CHECK (size_bytes = pg_catalog.octet_length(content))
);

CREATE TABLE __MONOID_SCHEMA__.run_blob (
    run_id text NOT NULL REFERENCES __MONOID_SCHEMA__.run_authority (run_id)
        ON DELETE CASCADE,
    sha256 character(64) NOT NULL REFERENCES __MONOID_SCHEMA__.bytea_blob (sha256),
    associated_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    PRIMARY KEY (run_id, sha256)
);

CREATE TABLE __MONOID_SCHEMA__.checkpoint_record (
    run_id text NOT NULL REFERENCES __MONOID_SCHEMA__.run_authority (run_id)
        ON DELETE CASCADE,
    sequence bigint NOT NULL CHECK (sequence >= 0),
    schema_version text NOT NULL,
    content_digest character(64) NOT NULL CHECK (
        content_digest ~ '^[0-9a-f]{64}$'
    ),
    -- json preserves canonical numeric identity and escaped NUL text across a round trip.
    -- The typed columns remain authoritative; checked readers bind them to this payload.
    payload json NOT NULL CHECK (pg_catalog.json_typeof(payload) = 'object'),
    submitted_blobs jsonb NOT NULL CHECK (
        pg_catalog.jsonb_typeof(submitted_blobs) = 'object'
    ),
    committed_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    PRIMARY KEY (run_id, sequence)
);

CREATE TABLE __MONOID_SCHEMA__.checkpoint_head (
    run_id text PRIMARY KEY,
    sequence bigint NOT NULL,
    updated_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    FOREIGN KEY (run_id, sequence)
        REFERENCES __MONOID_SCHEMA__.checkpoint_record (run_id, sequence)
        ON DELETE CASCADE
);

CREATE TABLE __MONOID_SCHEMA__.invocation_record (
    run_id text NOT NULL REFERENCES __MONOID_SCHEMA__.run_authority (run_id)
        ON DELETE CASCADE,
    logical_call_id text NOT NULL,
    revision bigint NOT NULL CHECK (revision > 0),
    schema_version text NOT NULL,
    dispatch_id text NOT NULL,
    dispatch_attempt bigint NOT NULL CHECK (dispatch_attempt > 0),
    dispatch_state text NOT NULL CHECK (
        dispatch_state IN ('reserved', 'dispatch_started', 'settled', 'unknown')
    ),
    idempotency_key text NOT NULL,
    request_digest character(64) NOT NULL CHECK (
        request_digest ~ '^[0-9a-f]{64}$'
    ),
    digest_generation text NOT NULL,
    evidence_policy text NOT NULL CHECK (evidence_policy IN ('passive', 'required')),
    result_ref text NOT NULL,
    failure_code text NOT NULL,
    content_digest character(64) NOT NULL CHECK (
        content_digest ~ '^[0-9a-f]{64}$'
    ),
    -- Keep the full canonical value representation; jsonb normalizes negative zero.
    payload json NOT NULL CHECK (pg_catalog.json_typeof(payload) = 'object'),
    submitted_blobs jsonb NOT NULL CHECK (
        pg_catalog.jsonb_typeof(submitted_blobs) = 'object'
    ),
    committed_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    PRIMARY KEY (run_id, logical_call_id, revision)
);

CREATE TABLE __MONOID_SCHEMA__.invocation_head (
    run_id text NOT NULL,
    logical_call_id text NOT NULL,
    revision bigint NOT NULL,
    updated_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    PRIMARY KEY (run_id, logical_call_id),
    FOREIGN KEY (run_id, logical_call_id, revision)
        REFERENCES __MONOID_SCHEMA__.invocation_record (run_id, logical_call_id, revision)
        ON DELETE CASCADE
);

CREATE INDEX invocation_record_dispatch_idx
    ON __MONOID_SCHEMA__.invocation_record (run_id, logical_call_id, dispatch_id);
