CREATE TABLE __MONOID_SCHEMA__.object_blob (
    sha256 character(64) PRIMARY KEY CHECK (
        sha256 ~ '^[0-9a-f]{64}$'
    ),
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    locator text NOT NULL CHECK (
        pg_catalog.octet_length(locator) BETWEEN 1 AND 2048
        AND locator ~ '^[ -~]+$'
    ),
    generation bigint NOT NULL CHECK (generation > 0),
    state text NOT NULL CHECK (state IN ('available', 'deleted')),
    first_seen_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    verified_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    deleted_at timestamp with time zone,
    CHECK (
        (state = 'available' AND deleted_at IS NULL)
        OR (state = 'deleted' AND deleted_at IS NOT NULL)
    )
);

CREATE TABLE __MONOID_SCHEMA__.run_object_blob (
    run_id text NOT NULL REFERENCES __MONOID_SCHEMA__.run_authority (run_id)
        ON DELETE CASCADE,
    sha256 character(64) NOT NULL REFERENCES __MONOID_SCHEMA__.object_blob (sha256),
    associated_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    PRIMARY KEY (run_id, sha256)
);

CREATE INDEX run_object_blob_digest_idx
    ON __MONOID_SCHEMA__.run_object_blob (sha256, run_id);

CREATE TABLE __MONOID_SCHEMA__.object_gc_receipt (
    plan_id character(64) NOT NULL CHECK (
        plan_id ~ '^[0-9a-f]{64}$'
    ),
    -- Generation-0 raw orphans have no object_blob row before a successful delete.
    sha256 character(64) NOT NULL CHECK (
        sha256 ~ '^[0-9a-f]{64}$'
    ),
    candidate_generation bigint NOT NULL CHECK (candidate_generation >= 0),
    observed_generation bigint NOT NULL CHECK (observed_generation >= 0),
    status text NOT NULL CHECK (
        status IN (
            'deleted',
            'already_missing',
            'skipped_associated',
            'skipped_generation',
            'precondition_failed'
        )
    ),
    delete_token_sha256 character(64) NOT NULL CHECK (
        delete_token_sha256 ~ '^[0-9a-f]{64}$'
    ),
    recorded_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    PRIMARY KEY (plan_id, sha256)
);
