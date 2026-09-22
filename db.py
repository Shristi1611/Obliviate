"""
db.py — Database connectivity and CRUD operations for Obliviate.
"""
import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "obliviate.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(reset=False):
    if reset and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = get_connection()
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()


def create_shards_and_slices(num_shards, num_slices_per_shard):
    conn = get_connection()
    cur = conn.cursor()
    for shard_id in range(num_shards):
        cur.execute("INSERT OR IGNORE INTO shards (shard_id, record_count) VALUES (?, 0)", (shard_id,))
        for slice_order in range(num_slices_per_shard):
            cur.execute(
                "INSERT OR IGNORE INTO slices (shard_id, slice_order, record_count) VALUES (?, ?, 0)",
                (shard_id, slice_order),
            )
    conn.commit()
    conn.close()


def get_slice_id(shard_id, slice_order):
    conn = get_connection()
    row = conn.execute(
        "SELECT slice_id FROM slices WHERE shard_id=? AND slice_order=?", (shard_id, slice_order)
    ).fetchone()
    conn.close()
    return row["slice_id"] if row else None


def insert_record(shard_id, slice_order, features, label):
    slice_id = get_slice_id(shard_id, slice_order)
    if slice_id is None:
        raise ValueError(f"No such slice: shard {shard_id}, slice_order {slice_order}")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO records (shard_id, slice_id, features, label) VALUES (?, ?, ?, ?)",
        (shard_id, slice_id, json.dumps(features), int(label)),
    )
    record_id = cur.lastrowid
    cur.execute("UPDATE shards SET record_count = record_count + 1 WHERE shard_id=?", (shard_id,))
    cur.execute("UPDATE slices SET record_count = record_count + 1 WHERE slice_id=?", (slice_id,))
    conn.commit()
    conn.close()
    return record_id


def bulk_insert_records(records):
    ids = []
    for r in records:
        ids.append(insert_record(r["shard_id"], r["slice_order"], r["features"], r["label"]))
    return ids


def get_record(record_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM records WHERE record_id=?", (record_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_records_by_shard(shard_id, include_deleted=False):
    conn = get_connection()
    q = "SELECT * FROM records WHERE shard_id=?"
    if not include_deleted:
        q += " AND is_deleted=0"
    rows = conn.execute(q, (shard_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_records_by_slice(slice_id, include_deleted=False):
    conn = get_connection()
    q = "SELECT * FROM records WHERE slice_id=?"
    if not include_deleted:
        q += " AND is_deleted=0"
    rows = conn.execute(q, (slice_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_records(include_deleted=False):
    conn = get_connection()
    q = "SELECT * FROM records"
    if not include_deleted:
        q += " WHERE is_deleted=0"
    rows = conn.execute(q).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_slices_for_shard(shard_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM slices WHERE shard_id=? ORDER BY slice_order", (shard_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_shards():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM shards ORDER BY shard_id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_checkpoints_for_shard(shard_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM checkpoints WHERE shard_id=? ORDER BY slice_order", (shard_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_record_label(record_id, new_label):
    conn = get_connection()
    conn.execute("UPDATE records SET label=? WHERE record_id=?", (int(new_label), record_id))
    conn.commit()
    conn.close()


def soft_delete_record(record_id):
    conn = get_connection()
    row = conn.execute("SELECT shard_id, slice_id FROM records WHERE record_id=?", (record_id,)).fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"No such record: {record_id}")
    conn.execute("UPDATE records SET is_deleted=1 WHERE record_id=?", (record_id,))
    conn.execute("UPDATE shards SET record_count = record_count - 1 WHERE shard_id=?", (row["shard_id"],))
    conn.execute("UPDATE slices SET record_count = record_count - 1 WHERE slice_id=?", (row["slice_id"],))
    conn.commit()
    conn.close()
    return dict(row)


def save_checkpoint_record(shard_id, slice_order, model_path):
    conn = get_connection()
    conn.execute(
        "INSERT INTO checkpoints (shard_id, slice_order, model_path) VALUES (?, ?, ?)",
        (shard_id, slice_order, model_path),
    )
    conn.commit()
    conn.close()


def log_audit_event(operation, record_id, shard_id, slice_id, rolled_back_to_slice,
                     retrained_slices, accuracy_before, accuracy_after, duration_seconds):
    conn = get_connection()
    conn.execute(
        """INSERT INTO audit_log
           (operation, record_id, shard_id, slice_id, rolled_back_to_slice, retrained_slices,
            accuracy_before, accuracy_after, duration_seconds)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (operation, record_id, shard_id, slice_id, rolled_back_to_slice, retrained_slices,
         accuracy_before, accuracy_after, duration_seconds),
    )
    conn.commit()
    conn.close()


def get_audit_log():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM audit_log ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_model_version(description, overall_accuracy):
    conn = get_connection()
    conn.execute(
        "INSERT INTO model_versions (description, overall_accuracy) VALUES (?, ?)",
        (description, overall_accuracy),
    )
    conn.commit()
    conn.close()