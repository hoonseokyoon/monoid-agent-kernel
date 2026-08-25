ALTER TABLE __MONOID_SCHEMA__.invocation_record
    DROP CONSTRAINT invocation_record_evidence_policy_check;

ALTER TABLE __MONOID_SCHEMA__.invocation_record
    ADD CONSTRAINT invocation_record_evidence_policy_check CHECK (
        evidence_policy IN ('passive', 'required', 'outbox')
    );

CREATE TABLE __MONOID_SCHEMA__.event_record (
    run_id text NOT NULL REFERENCES __MONOID_SCHEMA__.run_authority (run_id)
        ON DELETE CASCADE,
    sequence bigint NOT NULL CHECK (sequence > 0),
    event_id text NOT NULL,
    schema_version text NOT NULL,
    event_timestamp timestamp with time zone NOT NULL,
    event_type text NOT NULL,
    event_level text NOT NULL,
    content_digest character(64) NOT NULL CHECK (
        content_digest ~ '^[0-9a-f]{64}$'
    ),
    -- json preserves canonical numeric identity and escaped NUL text.
    payload json NOT NULL CHECK (pg_catalog.json_typeof(payload) = 'object'),
    committed_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    PRIMARY KEY (run_id, sequence),
    UNIQUE (run_id, event_id)
);

CREATE TABLE __MONOID_SCHEMA__.event_head (
    run_id text PRIMARY KEY,
    sequence bigint NOT NULL,
    updated_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    FOREIGN KEY (run_id, sequence)
        REFERENCES __MONOID_SCHEMA__.event_record (run_id, sequence)
        ON DELETE CASCADE
);

CREATE TABLE __MONOID_SCHEMA__.terminal_record (
    run_id text PRIMARY KEY REFERENCES __MONOID_SCHEMA__.run_authority (run_id)
        ON DELETE CASCADE,
    schema_version text NOT NULL,
    outcome_kind text NOT NULL,
    retry_eligibility text NOT NULL,
    checkpoint_sequence bigint CHECK (checkpoint_sequence >= 0),
    error_code text NOT NULL,
    content_digest character(64) NOT NULL CHECK (
        content_digest ~ '^[0-9a-f]{64}$'
    ),
    payload json NOT NULL CHECK (pg_catalog.json_typeof(payload) = 'object'),
    committed_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp()
);

CREATE TABLE __MONOID_SCHEMA__.model_evidence_record (
    run_id text NOT NULL,
    logical_call_id text NOT NULL,
    revision bigint NOT NULL CHECK (revision > 0),
    schema_version text NOT NULL,
    evidence_policy text NOT NULL CHECK (
        evidence_policy IN ('passive', 'required', 'outbox')
    ),
    content_digest character(64) NOT NULL CHECK (
        content_digest ~ '^[0-9a-f]{64}$'
    ),
    payload json NOT NULL CHECK (pg_catalog.json_typeof(payload) = 'object'),
    committed_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    PRIMARY KEY (run_id, logical_call_id, revision),
    FOREIGN KEY (run_id, logical_call_id, revision)
        REFERENCES __MONOID_SCHEMA__.invocation_record (run_id, logical_call_id, revision)
        ON DELETE CASCADE
);

CREATE TABLE __MONOID_SCHEMA__.model_evidence_outbox (
    run_id text NOT NULL,
    logical_call_id text NOT NULL,
    revision bigint NOT NULL CHECK (revision > 0),
    schema_version text NOT NULL,
    evidence_policy text NOT NULL CHECK (evidence_policy = 'outbox'),
    content_digest character(64) NOT NULL CHECK (
        content_digest ~ '^[0-9a-f]{64}$'
    ),
    payload json NOT NULL CHECK (pg_catalog.json_typeof(payload) = 'object'),
    delivery_state text NOT NULL DEFAULT 'pending' CHECK (
        delivery_state IN ('pending', 'delivered', 'dead_letter')
    ),
    attempt_count integer NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    available_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    lease_owner text,
    leased_until timestamp with time zone,
    delivered_at timestamp with time zone,
    last_error_code text NOT NULL DEFAULT '',
    created_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    updated_at timestamp with time zone NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    PRIMARY KEY (run_id, logical_call_id, revision),
    FOREIGN KEY (run_id, logical_call_id, revision)
        REFERENCES __MONOID_SCHEMA__.invocation_record (run_id, logical_call_id, revision)
        ON DELETE CASCADE,
    CHECK ((lease_owner IS NULL) = (leased_until IS NULL)),
    CHECK (
        (delivery_state = 'delivered' AND delivered_at IS NOT NULL)
        OR (delivery_state <> 'delivered' AND delivered_at IS NULL)
    )
);

CREATE INDEX model_evidence_outbox_pending_idx
    ON __MONOID_SCHEMA__.model_evidence_outbox
    (delivery_state, available_at, created_at)
    WHERE delivery_state = 'pending';
