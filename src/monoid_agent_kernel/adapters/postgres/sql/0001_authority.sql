CREATE TABLE monoid_schema_migrations (
    migration_id text PRIMARY KEY CHECK (
        migration_id ~ '^[0-9]{4}_[A-Za-z0-9_]{1,120}$'
    ),
    ordinal smallint NOT NULL UNIQUE CHECK (ordinal > 0),
    checksum_sha256 character(64) NOT NULL CHECK (
        checksum_sha256 ~ '^[0-9a-f]{64}$'
    ),
    reader_floor smallint NOT NULL CHECK (reader_floor > 0),
    writer_floor smallint NOT NULL CHECK (writer_floor > 0),
    applied_at timestamp with time zone NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE run_authority (
    run_id text PRIMARY KEY CHECK (
        octet_length(run_id) BETWEEN 1 AND 256
        AND run_id ~ '^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,255}$'
    ),
    owner_id text NOT NULL CHECK (
        octet_length(owner_id) BETWEEN 1 AND 256
        AND owner_id ~ '^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,255}$'
    ),
    generation bigint NOT NULL CHECK (generation > 0),
    leased_until timestamp with time zone NOT NULL,
    revoked boolean NOT NULL DEFAULT false,
    updated_at timestamp with time zone NOT NULL,
    CHECK (NOT revoked OR leased_until <= updated_at)
);

CREATE INDEX run_authority_expiry_idx
    ON run_authority (leased_until)
    WHERE NOT revoked;
