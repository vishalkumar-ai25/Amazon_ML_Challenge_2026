#!/usr/bin/env python3
"""
High-Capacity 500,000-Pair SOTA LightGBM Trainer v2
Amazon ML Challenge 2026

Ultra-optimized vectorized pairing and 48-core parallel feature extraction.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import gc
import json
import multiprocessing as mp
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_sota import (
    FEATURE_NAMES, compute_record_meta, extract_features_single
)

DATA_DIR = "dataset/student_resource/dataset/train"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_OUT = os.path.join(MODEL_DIR, "lgbm_sota_500k.txt")
META_OUT = os.path.join(MODEL_DIR, "lgbm_sota_500k_meta.json")

SEED = 42

def compute_macro_f05(true_dict, pred_dict, all_ids):
    """Computes exact competition Macro F0.5 metric."""
    scores = []
    for sid in all_ids:
        t = true_dict.get(sid, set())
        p = pred_dict.get(sid, set())
        if len(t) == 0 and len(p) == 0:
            scores.append(1.0)
            continue
        if len(t) == 0 or len(p) == 0:
            scores.append(0.0)
            continue
        tp = len(t & p)
        if tp == 0:
            scores.append(0.0)
            continue
        prec = tp / len(p)
        rec = tp / len(t)
        scores.append(1.25 * prec * rec / (0.25 * prec + rec))
    return float(np.mean(scores))

_G_META = None

def _extract_chunk_worker(pairs_chunk):
    s1_dict, s23_dict = _G_META
    X_rows = []
    y_vals = []
    for sid, cid, rank, is_s3, label in pairs_chunk:
        if sid in s1_dict and cid in s23_dict:
            feats = extract_features_single(s1_dict[sid], s23_dict[cid], int(rank), bool(is_s3))
            X_rows.append(feats)
            y_vals.append(int(label))
    return X_rows, y_vals

def main():
    print("=" * 80)
    print("  AMAZON ML CHALLENGE 2026: 500,000-PAIR SOTA RERANKER TRAINING")
    print("=" * 80, flush=True)
    t_start = time.time()

    # 1. Load Ground Truth and True Matches
    print("\n[1/5] Loading 100,000 training queries from train_ground_truth.tsv...", flush=True)
    t0 = time.time()
    gt_df = pd.read_csv(f"{DATA_DIR}/train_ground_truth.tsv", sep="\t", dtype=str, nrows=100000).fillna("")
    train_s1_ids = set(gt_df["source1_entity_id"].values)
    gt_map = dict(zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]))

    pos_pairs = []
    needed_s23_ids = set()

    for sid, m_str in gt_map.items():
        if m_str:
            for rank, mid in enumerate(m_str.split(","), 1):
                if mid:
                    pos_pairs.append((sid, mid, rank, mid.startswith("S3-"), 1))
                    needed_s23_ids.add(mid)

    print(f"Collected {len(pos_pairs):,} positive matches across {len(train_s1_ids):,} S1 entities in {time.time()-t0:.1f}s")
    print(f"Unique candidate entities needed from S2/S3: {len(needed_s23_ids):,}")

    # 2. Load S1 metadata for these queries
    print("\n[2/5] Loading S1 metadata for the 100k queries...", flush=True)
    t0 = time.time()
    s1_raw = {}
    for chunk in pd.read_csv(f"{DATA_DIR}/train_source1.tsv", sep="\t", dtype=str, chunksize=500000):
        sub = chunk[chunk["entity_id"].isin(train_s1_ids)]
        for _, r in sub.iterrows():
            s1_raw[r["entity_id"]] = (r["business_name"], r["business_address"], r["country"])
        train_s1_ids -= set(sub["entity_id"])
        if not train_s1_ids:
            break
    print(f"Loaded {len(s1_raw):,} S1 records in {time.time()-t0:.1f}s")

    # 3. Stream S2 & S3 to load candidate records
    print("\n[3/5] Streaming train_source2.tsv & train_source3.tsv...", flush=True)
    t0 = time.time()
    s23_raw = {}
    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        path = f"{DATA_DIR}/{s_file}"
        for chunk in pd.read_csv(path, sep="\t", dtype=str, usecols=["entity_id", "business_name", "business_address", "country"], chunksize=1000000):
            sub = chunk[chunk["entity_id"].isin(needed_s23_ids)]
            for _, r in sub.iterrows():
                s23_raw[r["entity_id"]] = (r["business_name"], r["business_address"], r["country"])
            needed_s23_ids -= set(sub["entity_id"])
            if not needed_s23_ids:
                break
    print(f"Loaded {len(s23_raw):,} candidate records in {time.time()-t0:.1f}s")

    # 4. Mine Hard Negatives using Vectorized Indexing
    print("\n[4/5] Vectorized mining of hard negative pairs...", flush=True)
    t0 = time.time()
    cand_keys = np.array(list(s23_raw.keys()))
    s1_keys = np.array(list(s1_raw.keys()))
    n_pos = len(pos_pairs)
    target_neg = max(150000, 500000 - n_pos)

    # Fast vectorized sampling
    np.random.seed(SEED)
    rand_s1_indices = np.random.randint(0, len(s1_keys), size=target_neg * 2)
    rand_cand_indices = np.random.randint(0, len(cand_keys), size=target_neg * 2)

    neg_pairs = []
    for s_idx, c_idx in zip(rand_s1_indices, rand_cand_indices):
        sid = s1_keys[s_idx]
        cid = cand_keys[c_idx]
        true_mids = gt_map.get(sid, "")
        if cid not in true_mids:
            neg_pairs.append((sid, cid, 5, cid.startswith("S3-"), 0))
            if len(neg_pairs) >= target_neg:
                break

    all_pairs = pos_pairs + neg_pairs
    print(f"Total Dataset: {len(all_pairs):,} pairs ({len(pos_pairs):,} positive, {len(neg_pairs):,} negative) mined in {time.time()-t0:.2f}s!")

    # 5. Precompute metadata and extract features across 48 cores
    print("\n[5/5] Precomputing metadata and extracting 35 features across 48 cores...", flush=True)
    t0 = time.time()
    s1_meta = {sid: compute_record_meta(n, a, c) for sid, (n, a, c) in s1_raw.items()}
    s23_meta = {cid: compute_record_meta(n, a, c) for cid, (n, a, c) in s23_raw.items()}

    global _G_META
    _G_META = (s1_meta, s23_meta)

    # Slicing without np.array string conversion
    n_workers = min(48, os.cpu_count() or 4)
    chunks = [all_pairs[i::n_workers] for i in range(n_workers)]

    with mp.Pool(n_workers) as pool:
        results = pool.map(_extract_chunk_worker, chunks)

    X_all = []
    y_all = []
    for x_c, y_c in results:
        X_all.extend(x_c)
        y_all.extend(y_c)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)
    print(f"Extracted {len(X):,} feature rows ({X.shape[1]} features) in {time.time()-t0:.1f}s ({len(X)/max(1, time.time()-t0):,.0f} pairs/sec)!")

    # 6. Train LightGBM Booster
    print("\nTraining High-Capacity LightGBM Booster across 48 cores...", flush=True)
    t0 = time.time()
    split = int(0.80 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    val_pairs = all_pairs[split:]
    val_s1_ids = list({p[0] for p in val_pairs})
    val_gt = {sid: set(gt_map.get(sid, "").split(",")) - {""} for sid in val_s1_ids}

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_weight = float(n_neg / max(1, n_pos))

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=FEATURE_NAMES)

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "learning_rate": 0.06,
        "num_leaves": 127,
        "max_depth": 10,
        "scale_pos_weight": scale_weight,
        "n_jobs": n_workers,
        "verbose": -1,
        "random_state": SEED
    }

    evals_result = {}
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, verbose=False),
            lgb.record_evaluation(evals_result)
        ]
    )

    val_probs = booster.predict(X_val)
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"Booster trained in {time.time()-t0:.1f}s! Best iteration: {booster.best_iteration} | Validation AUC: {val_auc:.5f}")

    # Optimize threshold for Macro F0.5
    print("\nOptimizing Decision Threshold for Competition Macro F0.5...", flush=True)
    val_preds_by_s1 = {}
    for (sid, cid, _, _, _), prob in zip(val_pairs, val_probs):
        if sid not in val_preds_by_s1:
            val_preds_by_s1[sid] = []
        val_preds_by_s1[sid].append((cid, prob))

    best_tau = 0.50
    best_f05 = 0.0

    for tau_test in np.arange(0.50, 0.98, 0.03):
        pred_dict = {}
        for sid in val_s1_ids:
            matches = [cid for cid, p in val_preds_by_s1.get(sid, []) if p >= tau_test]
            pred_dict[sid] = set(matches)
        f05 = compute_macro_f05(val_gt, pred_dict, val_s1_ids)
        if f05 > best_f05:
            best_f05 = f05
            best_tau = float(tau_test)
        print(f"  tau = {tau_test:.2f} -> Holdout Macro F0.5 = {f05:.4f}")

    print(f"\n★ BEST OPTIMAL THRESHOLD: tau = {best_tau:.2f} (Macro F0.5 = {best_f05:.4f})")

    booster.save_model(MODEL_OUT)
    meta_info = {
        "feature_names": FEATURE_NAMES,
        "best_threshold": best_tau,
        "best_f05": best_f05,
        "val_auc": val_auc,
        "sample_size": len(all_pairs),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(META_OUT, "w") as fm:
        json.dump(meta_info, fm, indent=2)

    print(f"Saved model to {MODEL_OUT} ({os.path.getsize(MODEL_OUT)/(1024*1024):.1f} MB)")
    print(f"Saved metadata to {META_OUT}")
    print("\n" + "=" * 80)
    print(f"  TRAINING COMPLETE IN {time.time()-t_start:.1f}s ({((time.time()-t_start)/60):.1f} min)")
    print("=" * 80)

if __name__ == "__main__":
    main()
