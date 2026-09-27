#!/usr/bin/env python3
"""
High-Speed 64-Thread LightGBM Reranker Trainer v2
Amazon ML Challenge 2026

Trains 35-feature LightGBM reranker with 5-fold cross-validation and exact Macro F0.5 optimization.
Utilizes 64 parallel CPU threads for sub-3-minute feature extraction.
"""

import os
import sys
import time
import json
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalization_v2 import parallel_process_corpus_v2
from blocking_v2 import MultiChannelBlockingEngineV2
from features_v2 import FEATURE_NAMES, extract_pairwise_features_v2

DATA_DIR = "dataset/student_resource/dataset/train"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42

def compute_f05_macro(true_dict, pred_dict, all_ids):
    """Computes exact competition Macro F0.5 metric across all S1 entities."""
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train LightGBM Reranker v2 with 64 Threads")
    parser.add_argument("--sample-size", type=int, default=30000, help="Number of S1 queries to sample")
    parser.add_argument("--n-workers", type=int, default=64, help="Number of parallel workers")
    args = parser.parse_args()

    sample_size = args.sample_size
    n_workers = args.n_workers

    print("=" * 80)
    print("  AMAZON ML CHALLENGE 2026: SOTA LIGHTGBM RERANKER v2 TRAINING")
    print(f"  Sample: {sample_size:,} queries | 64 Parallel Workers | 35 Features")
    print("=" * 80, flush=True)
    t_start = time.time()

    # 1. Load S1 data and Ground Truth
    print("[1/5] Loading training data...", flush=True)
    s1 = pd.read_csv(f"{DATA_DIR}/train_source1.tsv", sep="\t", dtype=str).fillna("")
    gt = pd.read_csv(f"{DATA_DIR}/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

    gt_map = dict(zip(gt['source1_entity_id'], gt['matched_entity_ids']))
    del gt
    gc.collect()

    np.random.seed(SEED)
    sample_indices = []
    for country in ['US', 'India']:
        c_idx = s1[s1['country'] == country].index.values
        n_c = int(sample_size * len(c_idx) / len(s1))
        sample_indices.extend(np.random.choice(c_idx, size=min(n_c, len(c_idx)), replace=False))

    s1_samp = s1.loc[sample_indices].reset_index(drop=True)
    sample_ids = s1_samp['entity_id'].values

    gt_dict = {
        sid: set(str(gt_map.get(sid, "")).split(",")) if gt_map.get(sid) else set()
        for sid in sample_ids
    }
    total_true_matches = sum(len(v) for v in gt_dict.values())
    n_singletons = sum(1 for v in gt_dict.values() if len(v) == 0)
    print(f"  Sampled {len(s1_samp):,} S1 entities ({len(s1_samp) - n_singletons:,} matched, {n_singletons:,} singletons, {total_true_matches:,} true links).", flush=True)

    # 2. Load S2 and S3 corpora
    print("[2/5] Loading S2 and S3 pools...", flush=True)
    s2 = pd.read_csv(f"{DATA_DIR}/train_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(f"{DATA_DIR}/train_source3.tsv", sep="\t", dtype=str).fillna("")
    s2['is_s3'] = 0.0
    s3['is_s3'] = 1.0
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()
    print(f"  S2+S3 Corpus: {len(s2s3):,} records.", flush=True)

    all_features = []
    all_labels = []
    all_triples = []
    captured_true_total = 0
    total_candidates_generated = 0

    for country in ['US', 'India']:
        print("\n" + "-" * 75)
        print(f"  PROCESSING COUNTRY: {country}")
        print("-" * 75, flush=True)

        s1_c = s1_samp[s1_samp['country'] == country].reset_index(drop=True)
        s2s3_c = s2s3[s2s3['country'] == country].reset_index(drop=True)
        n_s1_c = len(s1_c)
        n_c = len(s2s3_c)
        print(f"  S1 Sample: {n_s1_c:,} | S2+S3 Pool: {n_c:,}", flush=True)

        # 64-Worker Parallel Preprocessing
        print(f"  Parallel preprocessing S1 records ({n_workers} workers)...", flush=True)
        t0 = time.time()
        (s1_names_norm, s1_names_clean, s1_names_sp, s1_name_words, s1_first_words,
         s1_addrs_norm, s1_hns, s1_pcs, s1_addr_tokens, s1_nums) = parallel_process_corpus_v2(
            s1_c['business_name'].values, s1_c['business_address'].values, n_workers=n_workers
        )
        s1_ids = s1_c['entity_id'].values
        print(f"  S1 preprocessed in {time.time()-t0:.1f}s", flush=True)

        print(f"  Parallel preprocessing S2+S3 records ({n_workers} workers)...", flush=True)
        t0 = time.time()
        (c_names_norm, c_names_clean, c_names_sp, c_name_words, c_first_words,
         c_addrs_norm, c_hns, c_pcs, c_addr_tokens, c_nums) = parallel_process_corpus_v2(
            s2s3_c['business_name'].values, s2s3_c['business_address'].values, n_workers=n_workers
        )
        c_ids = s2s3_c['entity_id'].values
        c_is_s3 = s2s3_c['is_s3'].values
        print(f"  S2+S3 preprocessed in {time.time()-t0:.1f}s", flush=True)

        # 5-Channel Blocking v2
        engine = MultiChannelBlockingEngineV2(top_k_per_channel=40, max_union=80)
        engine.fit_and_transform_corpus(
            s1_names_norm, s1_addrs_norm, s1_names_sp,
            c_names_norm, c_addrs_norm, c_names_sp, c_pcs, c_hns
        )

        # Batch Candidate Extraction
        BATCH_SIZE = 1000
        n_batches = int(np.ceil(n_s1_c / BATCH_SIZE))
        country_true = sum(len(gt_dict[sid]) for sid in s1_ids)
        country_captured = 0

        print(f"  Extracting features from blocking candidates across {n_batches} batches...", flush=True)
        t_batch_start = time.time()

        for b in range(n_batches):
            b_start = b * BATCH_SIZE
            b_end = min(b_start + BATCH_SIZE, n_s1_c)

            candidates_batch = engine.retrieve_candidates_for_batch(
                b_start, b_end, s1_pcs, s1_names_sp, s1_name_words, c_name_words
            )

            for i, cands in enumerate(candidates_batch):
                g_i = b_start + i
                sid = s1_ids[g_i]
                true_set = gt_dict[sid]

                s1_pack = (
                    s1_names_norm[g_i], s1_names_clean[g_i], s1_names_sp[g_i], s1_name_words[g_i], s1_first_words[g_i],
                    s1_addrs_norm[g_i], s1_addr_tokens[g_i], s1_hns[g_i], s1_pcs[g_i], s1_nums[g_i]
                )

                total_candidates_generated += len(cands)

                for rank, (cidx, scores) in enumerate(cands, 1):
                    cid = c_ids[cidx]
                    is_match = 1 if cid in true_set else 0
                    if is_match:
                        country_captured += 1
                        captured_true_total += 1

                    c_pack = (
                        c_names_norm[cidx], c_names_clean[cidx], c_names_sp[cidx], c_name_words[cidx], c_first_words[cidx],
                        c_addrs_norm[cidx], c_addr_tokens[cidx], c_hns[cidx], c_pcs[cidx], c_nums[cidx]
                    )

                    feats = extract_pairwise_features_v2(
                        s1_pack, c_pack, scores, rank, c_is_s3[cidx]
                    )

                    all_features.append(feats)
                    all_labels.append(is_match)
                    all_triples.append((sid, cid))

            if (b + 1) % 2 == 0 or b == n_batches - 1:
                elapsed = time.time() - t_batch_start
                pct = 100.0 * (b + 1) / n_batches
                print(f"    Batch {b+1}/{n_batches} ({pct:.0f}%) | Captured true: {country_captured:,}/{country_true:,} ({100*country_captured/max(1, country_true):.2f}%) | {elapsed:.0f}s", flush=True)

        recall_pct = 100.0 * country_captured / max(1, country_true)
        print(f"  [{country}] Blocking Pair Completeness: {recall_pct:.2f}% ({country_captured:,}/{country_true:,})", flush=True)
        del engine, s1_c, s2s3_c
        gc.collect()

    del s2s3
    gc.collect()

    overall_recall = 100.0 * captured_true_total / max(1, total_true_matches)
    print("\n" + "=" * 80)
    print(f"  TOTAL BLOCKING SUMMARY:")
    print(f"  Generated {total_candidates_generated:,} candidate pairs ({total_candidates_generated/len(sample_ids):.1f} per query)")
    print(f"  Blocking Pair Completeness (Recall Ceiling): {overall_recall:.2f}% ({captured_true_total:,}/{total_true_matches:,})")
    print("=" * 80, flush=True)

    # 3. Train LightGBM with 5-Fold Stratified CV
    print("\n[3/5] Converting feature matrices...", flush=True)
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    del all_features, all_labels
    gc.collect()

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    pos_weight = float(n_neg) / float(max(1, n_pos))
    print(f"  Dataset: {len(X):,} pairs | Positives: {n_pos:,} ({100*n_pos/len(X):.2f}%) | Negatives: {n_neg:,} | PosWeight: {pos_weight:.2f}", flush=True)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'max_depth': 9,
        'min_child_samples': 40,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': pos_weight * 0.7,
        'num_iterations': 600,
        'verbose': -1,
        'random_state': SEED,
        'num_threads': n_workers
    }

    print("\n[4/5] Running 5-Fold Stratified Cross-Validation...", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X), dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        t_fold = time.time()
        trn_data = lgb.Dataset(X[trn_idx], label=y[trn_idx], feature_name=FEATURE_NAMES)
        val_data = lgb.Dataset(X[val_idx], label=y[val_idx], feature_name=FEATURE_NAMES, reference=trn_data)

        clf = lgb.train(
            params, trn_data, valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        oof_preds[val_idx] = clf.predict(X[val_idx])
        print(f"  Fold {fold+1}/5 trained in {time.time()-t_fold:.1f}s | Best iteration: {clf.best_iteration}", flush=True)

    # Feature importances
    print("\nTop 15 Feature Importances (gain):")
    imp = clf.feature_importance(importance_type='gain')
    for name, gain in sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])[:15]:
        print(f"  {name:25s}: {gain:,.1f}")

    # Threshold sweep directly on Entity-Level Macro F0.5 with M2O
    print("\n[5/5] Optimizing Threshold for Macro F0.5 with Global M2O...", flush=True)
    best_tau = 0.50
    best_f05 = 0.0
    best_avg_links = 0.0

    for tau in np.arange(0.50, 0.96, 0.03):
        mask = oof_preds >= tau
        cand_indices = np.where(mask)[0]
        sorted_cand_idx = cand_indices[np.argsort(-oof_preds[cand_indices])]

        assigned_cids = set()
        pred_dict = {sid: set() for sid in sample_ids}
        total_p = 0

        for idx in sorted_cand_idx:
            sid, cid = all_triples[idx]
            if cid not in assigned_cids:
                assigned_cids.add(cid)
                pred_dict[sid].add(cid)
                total_p += 1

        f05 = compute_f05_macro(gt_dict, pred_dict, sample_ids)
        avg_preds = total_p / len(sample_ids)
        marker = " ★ BEST" if f05 > best_f05 else ""
        if f05 > best_f05:
            best_f05 = f05
            best_tau = tau
            best_avg_links = avg_preds
        print(f"  tau = {tau:.2f} | Macro F0.5 = {f05:.4f} | Avg links/entity = {avg_preds:.2f}{marker}", flush=True)

    print("\n" + "=" * 80)
    print(f"  CV BENCHMARK RESULT: Macro F0.5 = {best_f05:.4f} at tau = {best_tau:.2f}")
    print(f"  Target links/entity: {best_avg_links:.2f} (Ground truth: 3.46)")
    print("=" * 80, flush=True)

    # Train final production model on full data
    print("\nTraining final production model on full sample...", flush=True)
    full_train = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)
    final_model = lgb.train(params, full_train)

    model_path = os.path.join(MODEL_DIR, "lgbm_reranker_v2.txt")
    final_model.save_model(model_path)
    print(f"Saved model to {model_path}", flush=True)

    meta_path = os.path.join(MODEL_DIR, "lgbm_reranker_v2_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "feature_names": FEATURE_NAMES,
            "best_threshold": float(best_tau),
            "best_f05": float(best_f05),
            "avg_links": float(best_avg_links),
            "sample_size": sample_size,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"Saved metadata to {meta_path}", flush=True)
    print(f"Total time elapsed: {time.time()-t_start:.1f}s ({time.time()-t_start/60:.1f} min)")

if __name__ == "__main__":
    main()
