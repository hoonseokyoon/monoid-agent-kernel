CREATE TABLE __MONOID_SCHEMA__.durable_stream_head (
    run_id text NOT NULL REFERENCES __MONOID_SCHEMA__.run_authority (run_id)
        ON DELETE CASCADE,
    stream_id text NOT NULL CHECK (
        pg_catalog.octet_length(stream_id) BETWEEN 1 AND 256
        AND stream_id ~ '^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,255}$'
    ),
    channel text NOT NULL CHECK (
        pg_catalog.octet_length(channel) BETWEEN 1 AND 128
        AND channel ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$'
    ),
    logical_call_id text NOT NULL CHECK (
        pg_catalog.octet_length(logical_call_id) BETWEEN 1 AND 256
        AND logical_call_id ~ '^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,255}$'
    ),
    generation bigint NOT NULL CHECK (
        generation BETWEEN 1 AND 9007199254740991
    ),
    cursor_bytes bigint NOT NULL CHECK (
        cursor_bytes BETWEEN 0 AND 9007199254740991
    ),
    next_chunk_sequence bigint NOT NULL CHECK (
        next_chunk_sequence BETWEEN 1 AND 9007199254740991
    ),
    state text NOT NULL CHECK (state IN ('open', 'sealed')),
    final_sha256 text NOT NULL DEFAULT '' CHECK (
        final_sha256 = '' OR final_sha256 ~ '^[0-9a-f]{64}$'
    ),
    opened_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    updated_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    sealed_at timestamp with time zone,
    PRIMARY KEY (run_id, stream_id, channel),
    CHECK (
        (state = 'open' AND final_sha256 = '' AND sealed_at IS NULL)
        OR (state = 'sealed' AND final_sha256 <> '' AND sealed_at IS NOT NULL)
    )
);

CREATE TABLE __MONOID_SCHEMA__.durable_stream_reset_receipt (
    run_id text NOT NULL,
    stream_id text NOT NULL,
    channel text NOT NULL,
    reset_id text NOT NULL CHECK (
        pg_catalog.octet_length(reset_id) BETWEEN 1 AND 256
        AND reset_id ~ '^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,255}$'
    ),
    generation bigint NOT NULL CHECK (
        generation BETWEEN 2 AND 9007199254740991
    ),
    recorded_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    PRIMARY KEY (run_id, stream_id, channel, reset_id),
    UNIQUE (run_id, stream_id, channel, generation),
    FOREIGN KEY (run_id, stream_id, channel)
        REFERENCES __MONOID_SCHEMA__.durable_stream_head (run_id, stream_id, channel)
        ON DELETE CASCADE
);

CREATE TABLE __MONOID_SCHEMA__.durable_stream_chunk (
    run_id text NOT NULL,
    stream_id text NOT NULL,
    channel text NOT NULL,
    generation bigint NOT NULL CHECK (
        generation BETWEEN 1 AND 9007199254740991
    ),
    chunk_sequence bigint NOT NULL CHECK (
        chunk_sequence BETWEEN 1 AND 9007199254740991
    ),
    start_offset bigint NOT NULL CHECK (
        start_offset BETWEEN 0 AND 9007199254740991
    ),
    end_offset bigint NOT NULL CHECK (
        end_offset BETWEEN 1 AND 9007199254740991
        AND end_offset > start_offset
        AND end_offset - start_offset <= 4194304
    ),
    sha256 character(64) NOT NULL REFERENCES __MONOID_SCHEMA__.object_blob (sha256),
    committed_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    PRIMARY KEY (run_id, stream_id, channel, generation, chunk_sequence),
    UNIQUE (run_id, stream_id, channel, generation, start_offset),
    FOREIGN KEY (run_id, stream_id, channel)
        REFERENCES __MONOID_SCHEMA__.durable_stream_head (run_id, stream_id, channel)
        ON DELETE CASCADE
);

CREATE INDEX durable_stream_chunk_replay_idx
    ON __MONOID_SCHEMA__.durable_stream_chunk
    (run_id, stream_id, channel, generation, start_offset);

CREATE INDEX durable_stream_chunk_digest_idx
    ON __MONOID_SCHEMA__.durable_stream_chunk (sha256, run_id);
