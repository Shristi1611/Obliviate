"""
sisa.py — SISA (Sharded, Isolated, Sliced, Aggregated) training and unlearning
engine, using TabNet (Google Research's attentive tabular deep learning
architecture) as the per-shard model.

Core idea (unchanged from the Logistic Regression version):
  - Dataset is split into N shards (disjoint partitions).
  - Each shard is further split into M slices (ordered sub-partitions).
  - Each shard's model is trained INCREMENTALLY, slice by slice, with a
    checkpoint saved after each slice is folded in.
  - To "unlearn" a record: find its shard + slice, roll back to the checkpoint
    saved just BEFORE that slice was added, then retrain forward through the
    remaining slices (with the deleted record excluded).
  - Prediction time: aggregate all shard models' predictions via majority vote.

Why TabNet instead of a plain classifier: TabNet is a real, production-used
architecture (it powers Google Cloud's AutoML Tables) built specifically for
tabular data, and because it trains via epochs like any neural network,
"checkpoint after each slice" is a natural fit rather than a forced one --
this is architecturally the same situation the original SISA paper was
designed around (deep learning models), just applied to tabular clinical
data instead of images.
"""
import os
import json
import time
import shutil
import tempfile
import warnings
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score

import db

warnings.filterwarnings("ignore")  # TabNet is chatty about minor CPU/GPU notices

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Kept deliberately small: this is a mini-project, not a production model.
# Small epoch count per slice keeps unlearning fast, which is the entire
# point of the system -- a slow-to-retrain model would defeat the purpose.
MAX_EPOCHS_PER_SLICE = 20


def _checkpoint_path(shard_id, slice_order, checkpoint_dir=None):
    # TabNet's own save_model() appends ".zip" to whatever path we give it,
    # so we store the path WITHOUT the extension and add it back on load.
    base = checkpoint_dir if checkpoint_dir is not None else CHECKPOINT_DIR
    return os.path.join(base, f"shard{shard_id}_slice{slice_order}")


def _new_model():
    # n_d/n_a/n_steps kept small -- this is a small tabular dataset, not
    # ImageNet-scale data, so a smaller network trains faster and avoids
    # overfitting on a few hundred rows per shard.
    return TabNetClassifier(
        n_d=8, n_a=8, n_steps=3, gamma=1.3,
        seed=42, verbose=0,
        device_name="cpu",
    )


def _is_trained(model):
    # TabNetClassifier only has a real `.network` after its first successful fit.
    return getattr(model, "network", None) is not None


def _safe_batch_sizes(n_samples):
    # TabNet requires batch_size >= 2 (BatchNorm needs more than one sample
    # per channel). TabNet also internally re-splits each batch into
    # "virtual" mini-batches for Ghost Batch Normalization -- if that split
    # doesn't divide evenly, the SAME size-1 remainder problem recurs one
    # level deeper, independent of the DataLoader's own batching. At this
    # project's scale, the simplest robust fix is to make the virtual batch
    # size equal to the real batch size, which disables that internal
    # re-splitting entirely (equivalent to standard BatchNorm, not Ghost
    # BatchNorm) -- a reasonable, explainable simplification for a dataset
    # this small, where GBN's regularization benefit is negligible anyway.
    batch_size = max(2, min(256, n_samples))
    virtual_batch_size = batch_size
    return batch_size, virtual_batch_size


def _fit(model, X, y):
    """
    Fits (or continues fitting, if the model was loaded from a checkpoint)
    on the given accumulated data. Returns True if a real fit happened,
    False if there wasn't enough data/class diversity yet to fit meaningfully.
    """
    y = np.asarray(y)
    if len(X) < 2 or len(set(y.tolist())) < 2:
        return False

    batch_size, virtual_batch_size = _safe_batch_sizes(len(X))
    # If the dataset doesn't divide evenly into batch_size, the trailing
    # mini-batch can end up with exactly 1 sample -- PyTorch's BatchNorm
    # can't compute statistics on a single sample and raises an error.
    # Dropping that incomplete final batch is safe as long as there's at
    # least one full batch remaining (i.e. more samples than batch_size);
    # otherwise the whole dataset IS the one batch, so there's no remainder.
    drop_last = len(X) > batch_size
    model.fit(
        np.asarray(X, dtype=np.float32), y,
        max_epochs=MAX_EPOCHS_PER_SLICE,
        patience=0,          # no eval set at this scale, so no early stopping
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
        num_workers=0,
        drop_last=drop_last,
        warm_start=True,      # continue training the existing network, don't reinit
    )
    return True


def _save_model(model, path):
    model.save_model(path)  # writes path + ".zip"


def _load_model(path):
    model = _new_model()
    model.load_model(path + ".zip")
    return model


# ---------------------------------------------------------------------------
# TRAINING (initial, incremental slice-by-slice per shard)
# ---------------------------------------------------------------------------

def train_shard_from_scratch(shard_id, num_slices, checkpoint_dir=None, persist_to_db=True):
    model = _new_model()
    X_accum, y_accum = [], []

    for slice_order in range(num_slices):
        records = db.get_records_by_slice(
            db.get_slice_id(shard_id, slice_order), include_deleted=False
        )
        if records:
            X_accum.extend(json.loads(r["features"]) for r in records)
            y_accum.extend(r["label"] for r in records)
            print(f"  [shard {shard_id}] fitting slice {slice_order}/{num_slices - 1} "
                  f"({len(X_accum)} accumulated records)...", flush=True)
            _fit(model, X_accum, y_accum)

        if _is_trained(model):
            path = _checkpoint_path(shard_id, slice_order, checkpoint_dir)
            _save_model(model, path)
            if persist_to_db:
                db.save_checkpoint_record(shard_id, slice_order, path)

    return model


def train_all_shards(num_shards, num_slices, checkpoint_dir=None, persist_to_db=True):
    models = {}
    for shard_id in range(num_shards):
        print(f"Training shard {shard_id}/{num_shards - 1}...", flush=True)
        models[shard_id] = train_shard_from_scratch(
            shard_id, num_slices, checkpoint_dir=checkpoint_dir, persist_to_db=persist_to_db
        )
    return models


# ---------------------------------------------------------------------------
# AGGREGATION (majority vote across shard models)
# ---------------------------------------------------------------------------

def load_latest_shard_model(shard_id, num_slices, checkpoint_dir=None):
    for slice_order in range(num_slices - 1, -1, -1):
        path = _checkpoint_path(shard_id, slice_order, checkpoint_dir)
        if os.path.exists(path + ".zip"):
            return _load_model(path)
    return None


def aggregate_predict(X, num_shards, num_slices, checkpoint_dir=None):
    X = np.asarray(X, dtype=np.float32)
    all_preds = []
    for shard_id in range(num_shards):
        model = load_latest_shard_model(shard_id, num_slices, checkpoint_dir)
        if model is None or not _is_trained(model):
            continue
        all_preds.append(model.predict(X))
    if not all_preds:
        raise RuntimeError("No trained shard models available for aggregation.")
    all_preds = np.array(all_preds)
    final = []
    for col in range(all_preds.shape[1]):
        votes = all_preds[:, col]
        values, counts = np.unique(votes, return_counts=True)
        final.append(values[np.argmax(counts)])
    return np.array(final)


def evaluate_aggregate_accuracy(X_test, y_test, num_shards, num_slices, checkpoint_dir=None):
    preds = aggregate_predict(X_test, num_shards, num_slices, checkpoint_dir)
    return accuracy_score(y_test, preds)


# ---------------------------------------------------------------------------
# UNLEARNING (the core feature)
# ---------------------------------------------------------------------------

def unlearn_record(record_id, num_slices, X_test=None, y_test=None, num_shards=None):
    start = time.time()

    record = db.get_record(record_id)
    if record is None:
        raise ValueError(f"No such record: {record_id}")

    accuracy_before = None
    if X_test is not None and y_test is not None and num_shards is not None:
        accuracy_before = evaluate_aggregate_accuracy(X_test, y_test, num_shards, num_slices)

    shard_id = record["shard_id"]
    slice_id = record["slice_id"]

    conn = db.get_connection()
    slice_row = conn.execute("SELECT slice_order FROM slices WHERE slice_id=?", (slice_id,)).fetchone()
    conn.close()
    target_slice_order = slice_row["slice_order"]

    db.soft_delete_record(record_id)

    # Roll back: load the checkpoint from the slice BEFORE the target slice
    # (or start a fresh, untrained network if the record was in slice 0).
    if target_slice_order == 0:
        model = _new_model()
    else:
        prev_path = _checkpoint_path(shard_id, target_slice_order - 1)
        model = _load_model(prev_path) if os.path.exists(prev_path + ".zip") else _new_model()

    # Rebuild the accumulated training set up through (not including) the
    # target slice -- this reflects exactly what the loaded checkpoint
    # already "knows".
    X_accum, y_accum = [], []
    for so in range(target_slice_order):
        records = db.get_records_by_slice(db.get_slice_id(shard_id, so), include_deleted=False)
        X_accum.extend(json.loads(r["features"]) for r in records)
        y_accum.extend(r["label"] for r in records)

    retrained_count = 0
    for so in range(target_slice_order, num_slices):
        records = db.get_records_by_slice(db.get_slice_id(shard_id, so), include_deleted=False)
        X_accum.extend(json.loads(r["features"]) for r in records)
        y_accum.extend(r["label"] for r in records)

        _fit(model, X_accum, y_accum)

        if _is_trained(model):
            _save_model(model, _checkpoint_path(shard_id, so))
            db.save_checkpoint_record(shard_id, so, _checkpoint_path(shard_id, so))
        retrained_count += 1

    duration = time.time() - start

    accuracy_after = None
    if X_test is not None and y_test is not None and num_shards is not None:
        accuracy_after = evaluate_aggregate_accuracy(X_test, y_test, num_shards, num_slices)

    db.log_audit_event(
        operation="delete",
        record_id=record_id,
        shard_id=shard_id,
        slice_id=slice_id,
        rolled_back_to_slice=target_slice_order - 1,
        retrained_slices=retrained_count,
        accuracy_before=accuracy_before,
        accuracy_after=accuracy_after,
        duration_seconds=duration,
    )

    return {
        "record_id": record_id,
        "shard_id": shard_id,
        "rolled_back_to_slice": target_slice_order - 1,
        "retrained_slices": retrained_count,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "duration_seconds": duration,
    }


def update_record_and_retrain(record_id, new_label, num_slices, X_test=None, y_test=None, num_shards=None):
    """
    Updates a record's label AND retrains only the affected slice forward --
    the same chunked-retraining trick used by unlearn_record(), applied to
    a label change instead of a deletion. A changed label only needs to be
    reflected from its own slice onward; slices before it never saw this
    record with its old label being "wrong" in a way that needs undoing,
    so there's nothing to roll back further than that.
    """
    start = time.time()

    record = db.get_record(record_id)
    if record is None:
        raise ValueError(f"No such record: {record_id}")

    accuracy_before = None
    if X_test is not None and y_test is not None and num_shards is not None:
        accuracy_before = evaluate_aggregate_accuracy(X_test, y_test, num_shards, num_slices)

    shard_id = record["shard_id"]
    slice_id = record["slice_id"]

    conn = db.get_connection()
    slice_row = conn.execute("SELECT slice_order FROM slices WHERE slice_id=?", (slice_id,)).fetchone()
    conn.close()
    target_slice_order = slice_row["slice_order"]

    # Apply the actual label change first, so the retrain below picks it up.
    db.update_record_label(record_id, new_label)

    result = _rollback_and_retrain_shard(shard_id, target_slice_order, num_slices)

    duration = time.time() - start

    accuracy_after = None
    if X_test is not None and y_test is not None and num_shards is not None:
        accuracy_after = evaluate_aggregate_accuracy(X_test, y_test, num_shards, num_slices)

    db.log_audit_event(
        operation="update",
        record_id=record_id,
        shard_id=shard_id,
        slice_id=slice_id,
        rolled_back_to_slice=target_slice_order - 1,
        retrained_slices=result["retrained_slices"],
        accuracy_before=accuracy_before,
        accuracy_after=accuracy_after,
        duration_seconds=duration,
    )
    db.log_model_version(
        description=f"Update record #{record_id} (shard {shard_id}, retrained from slice {target_slice_order})",
        overall_accuracy=accuracy_after if accuracy_after is not None else accuracy_before,
    )

    return {
        "record_id": record_id,
        "shard_id": shard_id,
        "rolled_back_to_slice": target_slice_order - 1,
        "retrained_slices": result["retrained_slices"],
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "duration_seconds": duration,
    }


def insert_record_and_retrain(shard_id, slice_order, features, label, num_slices,
                               X_test=None, y_test=None, num_shards=None):
    """
    Inserts a new record into the given shard+slice AND retrains only that
    slice forward -- the same chunked-retraining trick used by unlearn_record()
    and update_record_and_retrain(), applied to a brand-new record instead of
    a changed or removed one. Checkpoints before the target slice never saw
    this record and don't need to; from the target slice onward, the model
    needs to be retrained to actually reflect the new data.
    """
    start = time.time()

    accuracy_before = None
    if X_test is not None and y_test is not None and num_shards is not None:
        accuracy_before = evaluate_aggregate_accuracy(X_test, y_test, num_shards, num_slices)

    # Insert the record first, so the retrain below picks it up.
    record_id = db.insert_record(shard_id, slice_order, features, label)

    result = _rollback_and_retrain_shard(shard_id, slice_order, num_slices)

    duration = time.time() - start

    accuracy_after = None
    if X_test is not None and y_test is not None and num_shards is not None:
        accuracy_after = evaluate_aggregate_accuracy(X_test, y_test, num_shards, num_slices)

    db.log_audit_event(
        operation="insert",
        record_id=record_id,
        shard_id=shard_id,
        slice_id=db.get_slice_id(shard_id, slice_order),
        rolled_back_to_slice=slice_order - 1,
        retrained_slices=result["retrained_slices"],
        accuracy_before=accuracy_before,
        accuracy_after=accuracy_after,
        duration_seconds=duration,
    )
    db.log_model_version(
        description=f"Insert record #{record_id} (shard {shard_id}, retrained from slice {slice_order})",
        overall_accuracy=accuracy_after if accuracy_after is not None else accuracy_before,
    )

    return {
        "record_id": record_id,
        "shard_id": shard_id,
        "rolled_back_to_slice": slice_order - 1,
        "retrained_slices": result["retrained_slices"],
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "duration_seconds": duration,
    }


def _rollback_and_retrain_shard(shard_id, target_slice_order, num_slices):
    """
    Shared core of the "roll back to checkpoint before target_slice_order,
    then retrain forward through num_slices" operation used by insert,
    update, and unlearn -- they all boil down to this same mechanic, just
    triggered by a different kind of data change.
    """
    if target_slice_order == 0:
        model = _new_model()
    else:
        prev_path = _checkpoint_path(shard_id, target_slice_order - 1)
        model = _load_model(prev_path) if os.path.exists(prev_path + ".zip") else _new_model()

    X_accum, y_accum = [], []
    for so in range(target_slice_order):
        records = db.get_records_by_slice(db.get_slice_id(shard_id, so), include_deleted=False)
        X_accum.extend(json.loads(r["features"]) for r in records)
        y_accum.extend(r["label"] for r in records)

    retrained_count = 0
    for so in range(target_slice_order, num_slices):
        records = db.get_records_by_slice(db.get_slice_id(shard_id, so), include_deleted=False)
        X_accum.extend(json.loads(r["features"]) for r in records)
        y_accum.extend(r["label"] for r in records)

        print(f"  [shard {shard_id}] retraining slice {so}/{num_slices - 1} "
              f"({len(X_accum)} accumulated records)...", flush=True)
        _fit(model, X_accum, y_accum)

        if _is_trained(model):
            path = _checkpoint_path(shard_id, so)
            _save_model(model, path)
            db.save_checkpoint_record(shard_id, so, path)
        retrained_count += 1

    return {"model": model, "retrained_slices": retrained_count}


def full_retrain_baseline_accuracy(num_shards, num_slices, X_test, y_test):
    """
    Trains a completely fresh set of shard models from scratch, purely to
    compare their accuracy against the fast-unlearned models. This trains
    into an isolated temp directory and never writes to db's checkpoints
    table, so it CANNOT overwrite the real checkpoints that fast unlearning
    relies on -- running this baseline is now non-destructive.
    """
    tmp_dir = tempfile.mkdtemp(prefix="obliviate_baseline_")
    try:
        train_all_shards(num_shards, num_slices, checkpoint_dir=tmp_dir, persist_to_db=False)
        return evaluate_aggregate_accuracy(X_test, y_test, num_shards, num_slices, checkpoint_dir=tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def all_shards_trained(num_shards, num_slices):
    """
    True if every shard already has a checkpoint for its final slice --
    i.e. training previously completed and doesn't need to be redone.
    """
    for shard_id in range(num_shards):
        if not os.path.exists(_checkpoint_path(shard_id, num_slices - 1) + ".zip"):
            return False
    return True