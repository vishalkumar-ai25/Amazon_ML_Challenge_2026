#!/usr/bin/env python3
"""
SOTA LightGBM Training from Realistic Candidate Distribution (30k Queries)
Amazon ML Challenge 2026

Uses output/validation_candidates_30k_top100.pkl to train LightGBM Booster on:
- All true positive matches (y=1)
- Realistic hard negatives sampled across all ranks (1 to 35) (y=0)

Evaluates on a strict holdout split (5,999 S1 queries) using the official
Entity-Level Macro F0.5 metric with Global M2O, optimizing the threshold tau*.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import gc
import json
import pickle
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
PKL_PATH = "output/validation_candidates_30k_top100.pkl"
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
    print("  AMAZON ML CHALLENGE 2026: SOTA RERANKER TRAINING (30K CANDIDATES)")
    print("=" * 80, flush=True)
    t_start = time.time()

    # 1. Load candidate pickle
    print(f"\n[1/6] Loading candidate pool from {PKL_PATH}...", flush=True)
    t0 = time.time()
    with open(PKL_PATH, "rb") as f:
        cand_data = pickle.load(f)
    cands_chA = cand_data["cand_chA_100"]
    cands_chB = cand_data["cand_chB_100"]
    all_s1_keys = list(cands_chA.keys())
    print(f"Loaded {len(all_s1_keys):,} S1 entities from pickle in {time.time()-t0:.2f}s")

    # 2. Load Ground Truth for these 30k entities
    print(f"\n[2/6] Loading Ground Truth for these {len(all_s1_keys):,} entities...", flush=True)
    t0 = time.time()
    s1_set = set(all_s1_keys)
    gt_map = {}
    for chunk in pd.read_csv(f"{DATA_DIR}/train_ground_truth.tsv", sep="\t", dtype=str, usecols=["source1_entity_id", "matched_entity_ids"], chunksize=500000):
        sub = chunk[chunk["source1_entity_id"].isin(s1_set)]
        for _, r in sub.iterrows():
            gt_map[r["source1_entity_id"]] = r["matched_entity_ids"] if pd.notna(r["matched_entity_ids"]) else ""
        s1_set -= set(sub["source1_entity_id"])
        if not s1_set:
            break
    print(f"Loaded {len(gt_map):,} GT mappings in {time.time()-t0:.2f}s")

    # 3. Stratified Group Split: 24,000 Train S1 queries, 5,999 Validation S1 queries
    np.random.seed(SEED)
    shuffled_s1 = np.random.permutation(all_s1_keys)
    n_train = 24000
    train_s1_ids = list(shuffled_s1[:n_train])
    val_s1_ids = list(shuffled_s1[n_train:])
    print(f"Split: {len(train_s1_ids):,} Train queries | {len(val_s1_ids):,} Validation queries")

    # Build Training and Validation candidate pairs
    print(f"\n[3/6] Assembling Realistic Train & Validation Pairs...", flush=True)
    t0 = time.time()
    needed_s23_ids = set()
    train_pairs = []

    # Assemble Train pairs (all true matches + hard negatives across all ranks 1..35)
    for sid in train_s1_ids:
        raw_gt = gt_map.get(sid, "")
        true_mids = set(raw_gt.split(",")) - {""}

        # Union and score candidates from channel A and B
        dict_a = cands_chA.get(sid, {})
        dict_b = cands_chB.get(sid, {})
        all_cids = set(dict_a.keys()) | set(dict_b.keys())
        scored = [(cid, max(dict_a.get(cid, 0.0), dict_b.get(cid, 0.0))) for cid in all_cids]
        scored.sort(key=lambda x: x[1], reverse=True)

        found_in_cands = set()
        cand_negs = []
        for rank, (cid, sc) in enumerate(scored[:35], 1):
            if cid in true_mids:
                train_pairs.append((sid, cid, rank, cid.startswith("S3-"), 1))
                needed_s23_ids.add(cid)
                found_in_cands.add(cid)
            else:
                cand_negs.append((sid, cid, rank, cid.startswith("S3-"), 0))
                needed_s23_ids.add(cid)

        # Include any true match missed by top 100 blocking
        for mid in (true_mids - found_in_cands):
            train_pairs.append((sid, mid, 1, mid.startswith("S3-"), 1))
            needed_s23_ids.add(mid)

        train_pairs.extend(cand_negs)

    # Assemble Validation pairs (top 35 candidates per entity)
    val_pairs = []
    val_gt = {}
    for sid in val_s1_ids:
        raw_gt = gt_map.get(sid, "")
        true_mids = set(raw_gt.split(",")) - {""}
        val_gt[sid] = true_mids

        dict_a = cands_chA.get(sid, {})
        dict_b = cands_chB.get(sid, {})
        all_cids = set(dict_a.keys()) | set(dict_b.keys())
        scored = [(cid, max(dict_a.get(cid, 0.0), dict_b.get(cid, 0.0))) for cid in all_cids]
        scored.sort(key=lambda x: x[1], reverse=True)

        for rank, (cid, sc) in enumerate(scored[:35], 1):
            label = 1 if cid in true_mids else 0
            val_pairs.append((sid, cid, rank, cid.startswith("S3-"), label))
            needed_s23_ids.add(cid)

    n_pos_train = sum(1 for p in train_pairs if p[4] == 1)
    n_neg_train = len(train_pairs) - n_pos_train
    print(f"Train Pairs: {len(train_pairs):,} ({n_pos_train:,} pos, {n_neg_train:,} hard neg) [Ratio 1:{n_neg_train/max(1,n_pos_train):.1f}]")
    print(f"Validation Pairs: {len(val_pairs):,} across {len(val_s1_ids):,} entities")
    print(f"Unique candidate entities needed from S2/S3: {len(needed_s23_ids):,} in {time.time()-t0:.2f}s")

    # 4. Stream Source1, Source2, Source3 records
    print(f"\n[4/6] Streaming Source1, Source2, Source3 records...", flush=True)
    t0 = time.time()
    all_needed_s1 = set(all_s1_keys)
    s1_raw = {}
    for chunk in pd.read_csv(f"{DATA_DIR}/train_source1.tsv", sep="\t", dtype=str, chunksize=500000):
        sub = chunk[chunk["entity_id"].isin(all_needed_s1)]
        for _, r in sub.iterrows():
            s1_raw[r["entity_id"]] = (r["business_name"], r["business_address"], r["country"])
        all_needed_s1 -= set(sub["entity_id"])
        if not all_needed_s1:
            break
    print(f"  Loaded {len(s1_raw):,} S1 records in {time.time()-t0:.2f}s")

    t1 = time.time()
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
    print(f"  Loaded {len(s23_raw):,} S2/S3 candidate records in {time.time()-t1:.2f}s")

    # 5. Precompute metadata and extract features across 48 cores (contiguous chunks)
    print(f"\n[5/6] Precomputing metadata and extracting features across 48 cores...", flush=True)
    t0 = time.time()
    s1_meta = {sid: compute_record_meta(n, a, c) for sid, (n, a, c) in s1_raw.items()}
    s23_meta = {cid: compute_record_meta(n, a, c) for cid, (n, a, c) in s23_raw.items()}

    global _G_META
    _G_META = (s1_meta, s23_meta)

    n_workers = min(48, os.cpu_count() or 4)

    # Feature extraction on Train (contiguous chunks for exact ordering)
    t_csize = (len(train_pairs) + n_workers - 1) // n_workers
    train_chunks = [train_pairs[i * t_csize : (i + 1) * t_csize] for i in range(n_workers)]
    with mp.Pool(n_workers) as pool:
        train_results = pool.map(_extract_chunk_worker, train_chunks)

    X_train_list, y_train_list = [], []
    for x_c, y_c in train_results:
        X_train_list.extend(x_c)
        y_train_list.extend(y_c)
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)

    # Feature extraction on Val (contiguous chunks for exact ordering)
    v_csize = (len(val_pairs) + n_workers - 1) // n_workers
    val_chunks = [val_pairs[i * v_csize : (i + 1) * v_csize] for i in range(n_workers)]
    with mp.Pool(n_workers) as pool:
        val_results = pool.map(_extract_chunk_worker, val_chunks)

    X_val_list, y_val_list = [], []
    for x_c, y_c in val_results:
        X_val_list.extend(x_c)
        y_val_list.extend(y_c)
    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.int32)

    print(f"Extracted {len(X_train):,} Train rows and {len(X_val):,} Val rows in {time.time()-t0:.2f}s!")

    # 6. Train High-Capacity LightGBM Booster
    print(f"\n[6/6] Training High-Capacity LightGBM Booster across {n_workers} cores...", flush=True)
    t0 = time.time()

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=FEATURE_NAMES)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": 10,
        "min_child_samples": 30,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": n_workers,
        "verbose": -1,
        "random_state": SEED
    }

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=150,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[lgb.log_evaluation(period=50)]
    )

    val_probs = booster.predict(X_val)
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"Booster trained in {time.time()-t0:.1f}s! Val AUC: {val_auc:.5f}")

    # Feature Importances
    importances = booster.feature_importance(importance_type="gain")
    top_feats = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 Features by Gain:")
    for fn, g in top_feats:
        print(f"  {fn:25s}: {g:12.1f}")

    # Holdout Threshold Optimization on 5,999 Queries with Global M2O
    print(f"\nEvaluating Holdout Macro F0.5 on {len(val_s1_ids):,} Validation Entities with Global M2O...", flush=True)
    claims = []
    for (sid, cid, rank, is_s3, label), p in zip(val_pairs, val_probs):
        claims.append((sid, cid, float(p)))
    claims.sort(key=lambda x: x[2], reverse=True)

    best_tau = 0.50
    best_f05 = 0.0

    print(f"{'Threshold (tau)':>18} | {'Macro F0.5':>12} | {'Avg Links/Entity':>18} | {'Singletons %':>14}")
    print("-" * 70)

    for tau_test in np.arange(0.60, 0.99, 0.03):
        assigned = set()
        s1_matches = {sid: [] for sid in val_s1_ids}
        for sid, cid, p in claims:
            if p < tau_test: break
            if cid not in assigned:
                assigned.add(cid)
                s1_matches[sid].append(cid)

        pred_dict = {}
        total_links = 0
        singletons = 0
        for sid in val_s1_ids:
            matches = s1_matches.get(sid, [])
            s2_m = [m for m in matches if m.startswith("S2-")][:5]
            s3_m = [m for m in matches if m.startswith("S3-")][:6]
            final_m = s2_m + s3_m
            pred_dict[sid] = set(final_m)
            total_links += len(final_m)
            if not final_m:
                singletons += 1

        f05 = compute_macro_f05(val_gt, pred_dict, val_s1_ids)
        avg_links = total_links / len(val_s1_ids)
        sing_pct = (singletons / len(val_s1_ids)) * 100.0

        if f05 > best_f05:
            best_f05 = f05
            best_tau = float(tau_test)

        print(f"{tau_test:18.2f} | {f05:12.4f} | {avg_links:18.2f} | {sing_pct:13.2f}%")

    # Fine-grained threshold sweep around best
    print("\nFine-grained sweep around best threshold:")
    for tau_fine in np.arange(max(0.40, best_tau - 0.04), min(0.995, best_tau + 0.04), 0.01):
        assigned = set()
        s1_matches = {sid: [] for sid in val_s1_ids}
        for sid, cid, p in claims:
            if p < tau_fine: break
            if cid not in assigned:
                assigned.add(cid)
                s1_matches[sid].append(cid)

        pred_dict = {}
        for sid in val_s1_ids:
            matches = s1_matches.get(sid, [])
            s2_m = [m for m in matches if m.startswith("S2-")][:5]
            s3_m = [m for m in matches if m.startswith("S3-")][:6]
            pred_dict[sid] = set(s2_m + s3_m)
        f05 = compute_macro_f05(val_gt, pred_dict, val_s1_ids)
        if f05 > best_f05:
            best_f05 = f05
            best_tau = float(tau_fine)
        print(f"  tau = {tau_fine:.2f} -> Macro F0.5 = {f05:.4f}")

    print(f"\n{'*'*80}")
    print(f"  ★ BEST OPTIMAL THRESHOLD: tau = {best_tau:.2f} (Macro F0.5 = {best_f05:.4f})")
    print(f"{'*'*80}")

    booster.save_model(MODEL_OUT)
    meta_info = {
        "feature_names": FEATURE_NAMES,
        "best_threshold": best_tau,
        "best_f05": best_f05,
        "val_auc": val_auc,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(META_OUT, "w") as fm:
        json.dump(meta_info, fm, indent=2)

    print(f"\nSaved booster to {MODEL_OUT} ({os.path.getsize(MODEL_OUT)/(1024*1024):.1f} MB)")
    print(f"Saved metadata to {META_OUT}")
    print("\n" + "=" * 80)
    print(f"  TRAINING PIPELINE COMPLETE IN {time.time()-t_start:.1f}s ({((time.time()-t_start)/60):.1f} min)")
    print("=" * 80)

if __name__ == "__main__":
    main()
