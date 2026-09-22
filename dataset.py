"""
dataset.py — Loads the PhysioNet 2019 Sepsis Challenge dataset, aggregates
each patient's many hourly rows into a single summary row, and seeds the
Obliviate database with the aggregated records.

To get the raw data: run download_sepsis_data.py once (see that file).
"""
import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import db

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "sepsis_raw")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.npz")

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
CLINICAL_COLUMNS = VITALS + LABS

STATIC_COLUMNS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]
LOS_COLUMN = "ICULOS"
LABEL_COLUMN = "SepsisLabel"


def aggregate_patient_file(filepath):
    df = pd.read_csv(filepath, sep="|")

    features = {}
    for col in CLINICAL_COLUMNS:
        if col in df.columns:
            features[f"{col}_mean"] = df[col].mean(skipna=True)
            features[f"{col}_max"] = df[col].max(skipna=True)
        else:
            features[f"{col}_mean"] = np.nan
            features[f"{col}_max"] = np.nan

    for col in STATIC_COLUMNS:
        features[col] = df[col].iloc[0] if col in df.columns else np.nan

    features["length_of_stay_hours"] = (
        df[LOS_COLUMN].max() if LOS_COLUMN in df.columns else len(df)
    )

    label = int(df[LABEL_COLUMN].max()) if LABEL_COLUMN in df.columns else 0

    return features, label


def build_aggregated_dataset(raw_dir=RAW_DATA_DIR, max_patients=None):
    filepaths = sorted(glob.glob(os.path.join(raw_dir, "**", "*.psv"), recursive=True))
    if max_patients:
        filepaths = filepaths[:max_patients]

    if not filepaths:
        raise FileNotFoundError(
            f"No .psv files found under {raw_dir}. Run download_sepsis_data.py first."
        )

    total = len(filepaths)
    print(f"Aggregating {total} patient files...", flush=True)
    rows, labels = [], []
    for i, fp in enumerate(filepaths):
        feats, label = aggregate_patient_file(fp)
        rows.append(feats)
        labels.append(label)
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  ...{i + 1}/{total} patient files aggregated", flush=True)

    agg_df = pd.DataFrame(rows)
    feature_names = list(agg_df.columns)

    agg_df = agg_df.fillna(agg_df.mean(numeric_only=True))
    agg_df = agg_df.fillna(0)

    X = agg_df.values
    y = np.array(labels)
    return X, y, feature_names


def load_and_seed(num_shards=4, num_slices=4, test_size=0.2, random_state=42,
                   raw_dir=RAW_DATA_DIR, max_patients=None):
    X, y, feature_names = build_aggregated_dataset(raw_dir, max_patients)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    db.init_db(reset=True)
    db.create_shards_and_slices(num_shards, num_slices)

    n = len(X_train)
    print(f"Seeding {n} training records into {num_shards} shards / {num_slices} slices...", flush=True)
    for i in range(n):
        shard_id = i % num_shards
        slice_order = (i // num_shards) % num_slices
        db.insert_record(
            shard_id=shard_id,
            slice_order=slice_order,
            features=X_train[i].tolist(),
            label=int(y_train[i]),
        )
        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(f"  ...{i + 1}/{n} records seeded", flush=True)
    print("Seeding complete. Starting SISA training next.", flush=True)

    # Persist the held-out test set to disk. Without this, every server
    # restart would lose the ability to evaluate accuracy against a
    # consistent test set, forcing a full re-seed (and re-split, which
    # could even land on DIFFERENT test rows) just to resume normal use.
    np.savez(TEST_SET_PATH, X_test=X_test, y_test=y_test,
              feature_names=np.array(feature_names, dtype=object))

    return X_test, y_test, feature_names


def load_existing_test_set():
    """Loads a previously-saved test set from disk, or returns None if none exists."""
    if not os.path.exists(TEST_SET_PATH):
        return None
    data = np.load(TEST_SET_PATH, allow_pickle=True)
    return data["X_test"], data["y_test"], list(data["feature_names"])


def already_seeded():
    """True if a previous run already seeded the DB and saved a test set."""
    return os.path.exists(db.DB_PATH) and os.path.exists(TEST_SET_PATH)