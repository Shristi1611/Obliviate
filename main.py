"""
main.py — FastAPI backend for Obliviate, serving the hero page, the
dashboard page, and the JSON API that wraps db.py / sisa.py.

Run with: uvicorn main:app --reload
Then open http://127.0.0.1:8000 in your browser.
"""
import os
import time
import traceback
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

import db
import sisa
import dataset
from dataset import load_and_seed

NUM_SHARDS = 4
NUM_SLICES = 4
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="Obliviate API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state held across requests within this process, so /unlearn
# and /overview can evaluate accuracy against the same held-out test set
# that /init created.
STATE = {"X_test": None, "y_test": None, "feature_names": None, "ready": False}


# ---------------------------------------------------------------------------
# Request/response models (FastAPI validates these automatically)
# ---------------------------------------------------------------------------

class InitRequest(BaseModel):
    max_patients: Optional[int] = None
    force: Optional[bool] = False


def _try_resume():
    """
    If a previous run already seeded the DB and fully trained every shard,
    load the saved test set into STATE and return True -- skipping the
    (slow) seed + train step entirely. Returns False if there's nothing
    to resume from, meaning a full init is genuinely needed.
    """
    if not dataset.already_seeded():
        return False
    if not sisa.all_shards_trained(NUM_SHARDS, NUM_SLICES):
        return False

    existing = dataset.load_existing_test_set()
    if existing is None:
        return False

    X_test, y_test, feature_names = existing
    STATE["X_test"] = X_test
    STATE["y_test"] = y_test
    STATE["feature_names"] = feature_names
    STATE["ready"] = True
    return True


@app.on_event("startup")
def resume_on_startup():
    # Silently try to pick up where a previous run left off, so restarting
    # the server (or --reload triggering a restart) doesn't force a full
    # re-seed and re-train of already-trained models.
    _try_resume()


class InsertRequest(BaseModel):
    shard_id: int
    slice_order: int
    label: int


class UpdateRequest(BaseModel):
    record_id: int
    new_label: int


class UnlearnRequest(BaseModel):
    record_id: int


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.get("/")
def serve_hero():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/dashboard")
def serve_dashboard():
    return FileResponse(os.path.join(STATIC_DIR, "dashboard.html"))


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.post("/api/init")
def api_init(body: InitRequest):
    try:
        if not body.force and _try_resume():
            return {
                "status": "ok",
                "resumed": True,
                "accuracy": sisa.evaluate_aggregate_accuracy(
                    STATE["X_test"], STATE["y_test"], NUM_SHARDS, NUM_SLICES
                ),
                "num_features": len(STATE["feature_names"]),
            }

        X_test, y_test, feature_names = load_and_seed(
            num_shards=NUM_SHARDS, num_slices=NUM_SLICES, max_patients=body.max_patients
        )
        sisa.train_all_shards(NUM_SHARDS, NUM_SLICES)
        acc = sisa.evaluate_aggregate_accuracy(X_test, y_test, NUM_SHARDS, NUM_SLICES)
        db.log_model_version("Initial full training", acc)

        STATE["X_test"] = X_test
        STATE["y_test"] = y_test
        STATE["feature_names"] = feature_names
        STATE["ready"] = True

        return {"status": "ok", "resumed": False, "accuracy": acc, "num_features": len(feature_names)}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/api/status")
def api_status():
    return {"ready": STATE["ready"]}


@app.get("/api/overview")
def api_overview():
    if not STATE["ready"]:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Not initialized"})
    shards = db.get_all_shards()
    for s in shards:
        s["slices"] = db.get_slices_for_shard(s["shard_id"])
    acc = sisa.evaluate_aggregate_accuracy(STATE["X_test"], STATE["y_test"], NUM_SHARDS, NUM_SLICES)
    return {"shards": shards, "accuracy": acc, "num_shards": NUM_SHARDS, "num_slices": NUM_SLICES}


@app.get("/api/records")
def api_records(shard_id: Optional[int] = None):
    if shard_id is not None:
        records = db.get_records_by_shard(shard_id)
    else:
        records = db.get_all_records()
    return {"records": records}


@app.get("/api/checkpoints")
def api_checkpoints(shard_id: int = 0):
    return {"checkpoints": db.get_checkpoints_for_shard(shard_id)}


@app.post("/api/insert")
def api_insert(body: InsertRequest):
    try:
        num_features = len(STATE["feature_names"]) if STATE["feature_names"] else 74
        synthetic_features = np.random.normal(0, 1, size=num_features).tolist()
        result = sisa.insert_record_and_retrain(
            shard_id=body.shard_id,
            slice_order=body.slice_order,
            features=synthetic_features,
            label=body.label,
            num_slices=NUM_SLICES,
            X_test=STATE["X_test"],
            y_test=STATE["y_test"],
            num_shards=NUM_SHARDS,
        )
        return {"status": "ok", "result": result}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/api/update")
def api_update(body: UpdateRequest):
    try:
        result = sisa.update_record_and_retrain(
            record_id=body.record_id,
            new_label=body.new_label,
            num_slices=NUM_SLICES,
            X_test=STATE["X_test"],
            y_test=STATE["y_test"],
            num_shards=NUM_SHARDS,
        )
        return {"status": "ok", "result": result}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/api/unlearn")
def api_unlearn(body: UnlearnRequest):
    try:
        result = sisa.unlearn_record(
            record_id=body.record_id,
            num_slices=NUM_SLICES,
            X_test=STATE["X_test"],
            y_test=STATE["y_test"],
            num_shards=NUM_SHARDS,
        )
        return {"status": "ok", "result": result}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/api/full-retrain-baseline")
def api_full_retrain():
    try:
        start = time.time()
        acc = sisa.full_retrain_baseline_accuracy(NUM_SHARDS, NUM_SLICES, STATE["X_test"], STATE["y_test"])
        duration = time.time() - start
        return {"status": "ok", "accuracy": acc, "duration_seconds": duration}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/api/audit-log")
def api_audit_log():
    return {"log": db.get_audit_log()}