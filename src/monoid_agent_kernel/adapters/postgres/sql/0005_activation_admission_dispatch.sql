CREATE TABLE __MONOID_SCHEMA__.activation_admission_head (
    run_id text PRIMARY KEY REFERENCES __MONOID_SCHEMA__.run_authority (run_id)
        ON DELETE CASCADE,
    sequence bigint NOT NULL CHECK (sequence >= 0),
    updated_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp()
);

CREATE TABLE __MONOID_SCHEMA__.activation_admission_record (
    run_id text NOT NULL REFERENCES __MONOID_SCHEMA__.run_authority (run_id)
        ON DELETE CASCADE,
    command_id text NOT NULL CHECK (
        pg_catalog.octet_length(command_id) BETWEEN 1 AND 256
        AND command_id ~ '^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,255}$'
    ),
    command_sequence bigint NOT NULL CHECK (command_sequence > 0),
    command_kind text NOT NULL CHECK (command_kind IN ('input', 'control')),
    request_digest character(64) NOT NULL CHECK (
        request_digest ~ '^[0-9a-f]{64}$'
    ),
    payload_ref text NOT NULL CHECK (
        pg_catalog.octet_length(payload_ref) BETWEEN 3 AND 256
        AND payload_ref ~ '^[a-z][a-z0-9._-]{0,31}:[A-Za-z0-9][A-Za-z0-9._:+/-]*$'
    ),
    request_identity_sha256 character(64) NOT NULL CHECK (
        request_identity_sha256 ~ '^[0-9a-f]{64}$'
    ),
    admitted_identity_sha256 character(64) NOT NULL UNIQUE CHECK (
        admitted_identity_sha256 ~ '^[0-9a-f]{64}$'
    ),
    admitted_content_digest character(64) NOT NULL CHECK (
        admitted_content_digest ~ '^[0-9a-f]{64}$'
    ),
    admitted_payload json NOT NULL CHECK (
        pg_catalog.json_typeof(admitted_payload) = 'object'
    ),
    activation_identity_sha256 character(64) UNIQUE CHECK (
        activation_identity_sha256 IS NULL
        OR activation_identity_sha256 ~ '^[0-9a-f]{64}$'
    ),
    activation_content_digest character(64) CHECK (
        activation_content_digest IS NULL
        OR activation_content_digest ~ '^[0-9a-f]{64}$'
    ),
    activation_payload json CHECK (
        activation_payload IS NULL
        OR pg_catalog.json_typeof(activation_payload) = 'object'
    ),
    created_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    updated_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    PRIMARY KEY (run_id, command_id),
    UNIQUE (run_id, command_sequence),
    CHECK (
        (activation_identity_sha256 IS NULL)
        = (activation_content_digest IS NULL)
        AND (activation_identity_sha256 IS NULL) = (activation_payload IS NULL)
    )
);

CREATE TABLE __MONOID_SCHEMA__.activation_dispatch_outbox (
    run_id text NOT NULL,
    command_id text NOT NULL,
    delivery_state text NOT NULL DEFAULT 'pending' CHECK (
        delivery_state IN ('pending', 'leased', 'delivered', 'dead_letter')
    ),
    attempt_count integer NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    available_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    claim_owner text CHECK (
        claim_owner IS NULL
        OR (
            pg_catalog.octet_length(claim_owner) BETWEEN 1 AND 256
            AND claim_owner ~ '^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,255}$'
        )
    ),
    claim_id text UNIQUE CHECK (
        claim_id IS NULL
        OR (
            pg_catalog.octet_length(claim_id) BETWEEN 1 AND 256
            AND claim_id ~ '^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,255}$'
        )
    ),
    claim_generation bigint NOT NULL DEFAULT 0 CHECK (claim_generation >= 0),
    leased_until timestamp with time zone,
    dispatch_ref text NOT NULL DEFAULT '' CHECK (
        dispatch_ref = ''
        OR (
            pg_catalog.octet_length(dispatch_ref) BETWEEN 3 AND 256
            AND dispatch_ref ~ '^[a-z][a-z0-9._-]{0,31}:[A-Za-z0-9][A-Za-z0-9._:+/-]*$'
        )
    ),
    delivered_at timestamp with time zone,
    last_error_code text NOT NULL DEFAULT '' CHECK (
        last_error_code = ''
        OR (
            pg_catalog.octet_length(last_error_code) BETWEEN 1 AND 128
            AND last_error_code ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$'
        )
    ),
    created_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    updated_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    PRIMARY KEY (run_id, command_id),
    FOREIGN KEY (run_id, command_id)
        REFERENCES __MONOID_SCHEMA__.activation_admission_record (run_id, command_id)
        ON DELETE CASCADE,
    CHECK (
        (delivery_state = 'leased' AND claim_owner IS NOT NULL
            AND claim_id IS NOT NULL AND leased_until IS NOT NULL)
        OR (delivery_state <> 'leased' AND leased_until IS NULL)
    ),
    CHECK (
        (delivery_state = 'delivered' AND delivered_at IS NOT NULL
            AND dispatch_ref <> '' AND last_error_code = '')
        OR (delivery_state <> 'delivered' AND delivered_at IS NULL AND dispatch_ref = '')
    ),
    CHECK (
        delivery_state <> 'dead_letter' OR last_error_code <> ''
    )
);

CREATE INDEX activation_dispatch_outbox_pending_idx
    ON __MONOID_SCHEMA__.activation_dispatch_outbox
    (available_at, created_at, run_id, command_id)
    WHERE delivery_state IN ('pending', 'leased');
