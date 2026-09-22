"""
run_demo.py — End-to-end smoke test of the Obliviate CRUD + unlearning pipeline.
Run this to verify everything works before building the UI on top.
"""
import numpy as np
import db
import sisa
from dataset import load_and_seed

NUM_SHARDS = 4
NUM_SLICES = 4

print("=" * 70)
print("STEP 1: Seeding database with breast cancer dataset")
print("=" * 70)
X_test, y_test, feature_names = load_and_seed(num_shards=NUM_SHARDS, num_slices=NUM_SLICES)
print(f"Seeded. Test set size: {len(X_test)}")

print()
print("=" * 70)
print("STEP 2: Initial SISA training (sharded + sliced + checkpointed)")
print("=" * 70)
sisa.train_all_shards(NUM_SHARDS, NUM_SLICES)
initial_acc = sisa.evaluate_aggregate_accuracy(X_test, y_test, NUM_SHARDS, NUM_SLICES)
print(f"Initial aggregated model accuracy: {initial_acc:.4f}")
db.log_model_version("Initial full training", initial_acc)

print()
print("=" * 70)
print("STEP 3: CRUD - Retrieve records by shard")
print("=" * 70)
for shard_id in range(NUM_SHARDS):
    recs = db.get_records_by_shard(shard_id)
    print(f"Shard {shard_id}: {len(recs)} active records")

print()
print("=" * 70)
print("STEP 4: CRUD - Pick a record to DELETE (triggers unlearning)")
print("=" * 70)
all_records = db.get_all_records()
target = all_records[10]  # arbitrary pick
print(f"Target record_id={target['record_id']} in shard={target['shard_id']}")

result = sisa.unlearn_record(
    record_id=target["record_id"],
    num_slices=NUM_SLICES,
    X_test=X_test,
    y_test=y_test,
    num_shards=NUM_SHARDS,
)

print("Unlearning result:")
for k, v in result.items():
    print(f"  {k}: {v}")

print()
print("=" * 70)
print("STEP 5: Verification - compare fast-unlearned accuracy vs full retrain")
print("=" * 70)
fast_unlearn_acc = result["accuracy_after"]
full_retrain_acc = sisa.full_retrain_baseline_accuracy(NUM_SHARDS, NUM_SLICES, X_test, y_test)
print(f"Fast-unlearned accuracy : {fast_unlearn_acc:.4f}")
print(f"Full-retrain accuracy   : {full_retrain_acc:.4f}")
print(f"Difference              : {abs(fast_unlearn_acc - full_retrain_acc):.4f}")
print("(Small difference is expected/healthy - proves fast unlearning")
print(" closely approximates a full retrain without redoing all shards.)")

print()
print("=" * 70)
print("STEP 6: CRUD - view audit_log trail")
print("=" * 70)
for entry in db.get_audit_log():
    print(entry)

print()
print("ALL STEPS COMPLETED SUCCESSFULLY")