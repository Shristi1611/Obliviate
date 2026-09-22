-- Obliviate: SISA Unlearning Database Schema

CREATE TABLE IF NOT EXISTS shards (
    shard_id        INTEGER PRIMARY KEY,
    record_count    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS slices (
    slice_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    shard_id        INTEGER NOT NULL,
    slice_order     INTEGER NOT NULL,
    record_count    INTEGER DEFAULT 0,
    UNIQUE(shard_id, slice_order),
    FOREIGN KEY (shard_id) REFERENCES shards(shard_id)
);

CREATE TABLE IF NOT EXISTS records (
    record_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    shard_id        INTEGER NOT NULL,
    slice_id        INTEGER NOT NULL,
    features        TEXT NOT NULL,
    label           INTEGER NOT NULL,
    is_deleted      BOOLEAN DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (shard_id) REFERENCES shards(shard_id),
    FOREIGN KEY (slice_id) REFERENCES slices(slice_id)
);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    shard_id        INTEGER NOT NULL,
    slice_order     INTEGER NOT NULL,
    model_path      TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (shard_id) REFERENCES shards(shard_id)
);

CREATE TABLE IF NOT EXISTS audit_log (
    log_id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    operation               TEXT NOT NULL,      -- 'insert' | 'update' | 'delete'
    record_id               INTEGER,
    shard_id                INTEGER,
    slice_id                INTEGER,
    rolled_back_to_slice    INTEGER,
    retrained_slices        INTEGER,
    accuracy_before          REAL,
    accuracy_after           REAL,
    duration_seconds        REAL,
    timestamp               TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_versions (
    version_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    description     TEXT,
    overall_accuracy REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);