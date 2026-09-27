#!/usr/bin/env python3
"""
High-Speed LightGBM Reranker Trainer v3
Amazon ML Challenge 2026

- 200,000 S1 entities with Country+MatchCount stratification
- Hard negative mining (2-pass training)
- 3-fold CV with country-specific threshold optimization
- Sequential country processing to minimize RAM usage
"""

import os
import sys
import time
import json
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalization_v3 import parallel_process_corpus_v3
from blocking_v3 import MultiChannelBlockingEngineV3
from features_v3 import FEATURE_NAMES, extract_pairwise_features_v3

DATA_DIR = "dataset/student_resource/dataset/train"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42

def compute_f05_macro(true_dict, pred_dict, eval_ids):
    """Computes exact competition Macro F0.5 metric across given S1 entities."""
    scores = []
    for sid in eval_ids:
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
    parser = argparse.ArgumentParser(description="Train LightGBM Reranker v3")
    parser.add_argument("--sample-size", type=int, default=200000, help="Number of S1 queries to sample")
    parser.add_argument("--n-workers", type=int, default=48, help="Number of parallel workers")
    args = parser.parse_args()

    sample_size = args.sample_size
    n_workers = args.n_workers

    print("=" * 80)
    print("  AMAZON ML CHALLENGE 2026: SOTA LIGHTGBM RERANKER v3 TRAINING")
    print(f"  Sample: {sample_size:,} queries | {n_workers} Parallel Workers")
    print("=" * 80, flush=True)
    t_start = time.time()

    print("[1/5] Loading training data...", flush=True)
    s1 = pd.read_csv(f"{DATA_DIR}/train_source1.tsv", sep="\t", dtype=str).fillna("")
    gt = pd.read_csv(f"{DATA_DIR}/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

    gt_map = dict(zip(gt['source1_entity_id'], gt['matched_entity_ids']))
    del gt
    gc.collect()

    s1['match_count'] = s1['entity_id'].map(lambda x: len(str(gt_map.get(x, "")).split(",")) if str(gt_map.get(x, "")) != "" else 0)
    def get_bin(c):
        if c == 0: return '0'
        elif c <= 2: return '1-2'
        elif c <= 5: return '3-5'
        else: return '6+'
    s1['match_bin'] = s1['match_count'].map(get_bin)
    s1['stratify_col'] = s1['country'] + "_" + s1['match_bin']

    if sample_size >= len(s1):
        s1_samp = s1.copy()
    else:
        try:
            _, s1_samp = train_test_split(s1, test_size=sample_size, stratify=s1['stratify_col'], random_state=SEED)
        except ValueError:
            print("Stratification failed, falling back to random sampling.")
            _, s1_samp = train_test_split(s1, test_size=sample_size, random_state=SEED)
            
    s1_samp = s1_samp.reset_index(drop=True)
    sample_ids = s1_samp['entity_id'].values

    gt_dict = {
        sid: set(str(gt_map.get(sid, "")).split(",")) if gt_map.get(sid) else set()
        for sid in sample_ids
    }
    
    total_true_matches = sum(len(v) for v in gt_dict.values())
    n_singletons = sum(1 for v in gt_dict.values() if len(v) == 0)
    print(f"  Sampled {len(s1_samp):,} S1 entities ({len(s1_samp) - n_singletons:,} matched, {n_singletons:,} singletons, {total_true_matches:,} true links).", flush=True)

    del s1
    gc.collect()

    print("[2/5] Loading S2 and S3 pools...", flush=True)
    s2 = pd.read_csv(f"{DATA_DIR}/train_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(f"{DATA_DIR}/train_source3.tsv", sep="\t", dtype=str).fillna("")
    s2['is_s3'] = 0.0
    s3['is_s3'] = 1.0
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    all_features = []
    all_labels = []
    all_triples = []
    all_countries = []
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

        print(f"  Parallel preprocessing S1 records ({n_workers} workers)...", flush=True)
        t0 = time.time()
        (s1_names_norm, s1_names_clean, s1_names_sp, s1_name_words, s1_first_words,
         s1_addrs_norm, s1_hns, s1_pcs, s1_addr_tokens, s1_nums,
         s1_pks, s1_cities, s1_states, s1_lss, s1_cts) = parallel_process_corpus_v3(
            s1_c['business_name'].values, s1_c['business_address'].values, n_workers=n_workers
        )
        s1_ids = s1_c['entity_id'].values
        s1_first3 = [n[:3] for n in s1_names_clean]
        print(f"  S1 preprocessed in {time.time()-t0:.1f}s", flush=True)

        print(f"  Parallel preprocessing S2+S3 records ({n_workers} workers)...", flush=True)
        t0 = time.time()
        (c_names_norm, c_names_clean, c_names_sp, c_name_words, c_first_words,
         c_addrs_norm, c_hns, c_pcs, c_addr_tokens, c_nums,
         c_pks, c_cities, c_states, c_lss, c_cts) = parallel_process_corpus_v3(
            s2s3_c['business_name'].values, s2s3_c['business_address'].values, n_workers=n_workers
        )
        c_ids = s2s3_c['entity_id'].values
        c_is_s3 = s2s3_c['is_s3'].values
        c_first3 = [n[:3] for n in c_names_clean]
        print(f"  S2+S3 preprocessed in {time.time()-t0:.1f}s", flush=True)

        engine = MultiChannelBlockingEngineV3(top_k_per_channel=50, max_union=100)
        engine.fit_and_transform_corpus(
            s1_names_norm, s1_addrs_norm, s1_names_sp,
            c_names_norm, c_addrs_norm, c_names_sp, c_pcs, c_hns,
            c_phonetics=c_pks, c_first3=c_first3
        )

        BATCH_SIZE = 2000
        n_batches = int(np.ceil(n_s1_c / BATCH_SIZE))
        country_true = sum(len(gt_dict[sid]) for sid in s1_ids)
        country_captured = 0

        print(f"  Extracting features from blocking candidates across {n_batches} batches...", flush=True)
        t_batch_start = time.time()

        for b in range(n_batches):
            b_start = b * BATCH_SIZE
            b_end = min(b_start + BATCH_SIZE, n_s1_c)

            candidates_batch = engine.retrieve_candidates_for_batch(
                b_start, b_end, s1_pcs, s1_names_sp, s1_name_words, c_name_words,
                s1_phonetics=s1_pks, s1_first3=s1_first3
            )

            for i, cands in enumerate(candidates_batch):
                g_i = b_start + i
                sid = s1_ids[g_i]
                true_set = gt_dict[sid]

                s1_pack = (
                    s1_names_norm[g_i], s1_names_clean[g_i], s1_names_sp[g_i], s1_name_words[g_i], s1_first_words[g_i],
                    s1_addrs_norm[g_i], s1_addr_tokens[g_i], s1_hns[g_i], s1_pcs[g_i], s1_nums[g_i],
                    s1_pks[g_i], s1_cities[g_i], s1_states[g_i], s1_lss[g_i], s1_cts[g_i]
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
                        c_addrs_norm[cidx], c_addr_tokens[cidx], c_hns[cidx], c_pcs[cidx], c_nums[cidx],
                        c_pks[cidx], c_cities[cidx], c_states[cidx], c_lss[cidx], c_cts[cidx]
                    )

                    feats = extract_pairwise_features_v3(
                        s1_pack, c_pack, scores, rank, c_is_s3[cidx]
                    )

                    all_features.append(feats)
                    all_labels.append(is_match)
                    all_triples.append((sid, cid))
                    all_countries.append(country)

            if (b + 1) % 5 == 0 or b == n_batches - 1:
                elapsed = time.time() - t_batch_start
                pct = 100.0 * (b + 1) / n_batches
                print(f"    Batch {b+1}/{n_batches} ({pct:.0f}%) | Captured true: {country_captured:,}/{country_true:,} ({100*country_captured/max(1, country_true):.2f}%) | {elapsed:.0f}s", flush=True)

        recall_pct = 100.0 * country_captured / max(1, country_true)
        print(f"  [{country}] Blocking Pair Completeness: {recall_pct:.2f}% ({country_captured:,}/{country_true:,})", flush=True)
        
        # Aggressive memory cleanup
        del engine, s1_c, s2s3_c
        del s1_names_norm, s1_names_clean, s1_names_sp, s1_name_words, s1_first_words
        del s1_addrs_norm, s1_hns, s1_pcs, s1_addr_tokens, s1_nums
        del s1_pks, s1_cities, s1_states, s1_lss, s1_cts, s1_first3
        del c_names_norm, c_names_clean, c_names_sp, c_name_words, c_first_words
        del c_addrs_norm, c_hns, c_pcs, c_addr_tokens, c_nums
        del c_pks, c_cities, c_states, c_lss, c_cts, c_first3
        gc.collect()

    del s2s3
    gc.collect()

    overall_recall = 100.0 * captured_true_total / max(1, total_true_matches)
    print("\n" + "=" * 80)
    print(f"  TOTAL BLOCKING SUMMARY:")
    print(f"  Generated {total_candidates_generated:,} candidate pairs ({total_candidates_generated/len(sample_ids):.1f} per query)")
    print(f"  Blocking Pair Completeness (Recall Ceiling): {overall_recall:.2f}% ({captured_true_total:,}/{total_true_matches:,})")
    print("=" * 80, flush=True)

    print("\n[3/5] Converting feature matrices...", flush=True)
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    c_arr = np.array(all_countries)
    del all_features, all_labels, all_countries
    gc.collect()

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    pos_weight = float(n_neg) / float(max(1, n_pos))
    print(f"  Dataset: {len(X):,} pairs | Positives: {n_pos:,} ({100*n_pos/len(X):.2f}%) | Negatives: {n_neg:,} | PosWeight: {pos_weight:.2f}", flush=True)

    params = {
        'objective': 'binary',
        'metric': ['auc', 'binary_logloss'],
        'learning_rate': 0.03,
        'num_leaves': 255,
        'max_depth': 12,
        'min_child_samples': 30,
        'subsample': 0.75,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.05,
        'reg_lambda': 0.5,
        'scale_pos_weight': pos_weight * 0.7,
        'num_iterations': 1500,
        'min_gain_to_split': 0.01,
        'path_smooth': 0.1,
        'verbose': -1,
        'num_threads': 48,
    }

    print("\n[4/5] Hard Negative Mining (2-pass training)...", flush=True)
    print("  Pass 1: Training initial model on all data...")
    p1_params = params.copy()
    p1_params['num_iterations'] = 300
    p1_train = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)
    p1_model = lgb.train(p1_params, p1_train)
    
    print("  Pass 1: Predicting to find hard negatives...")
    p1_preds = p1_model.predict(X)
    
    weights = np.ones(len(y), dtype=np.float32)
    hard_neg_mask = (y == 0) & (p1_preds > 0.1)
    weights[hard_neg_mask] = 3.0
    n_hard_negs = hard_neg_mask.sum()
    print(f"  Found {n_hard_negs:,} hard negatives. Assigned 3x sample_weight.", flush=True)
    del p1_model, p1_preds, p1_train
    gc.collect()

    print("\n[5/5] Running 3-Fold Stratified Cross-Validation with Hard Negatives...", flush=True)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X), dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        t_fold = time.time()
        trn_data = lgb.Dataset(X[trn_idx], label=y[trn_idx], weight=weights[trn_idx], feature_name=FEATURE_NAMES)
        val_data = lgb.Dataset(X[val_idx], label=y[val_idx], weight=weights[val_idx], feature_name=FEATURE_NAMES, reference=trn_data)

        clf = lgb.train(
            params, trn_data, valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        oof_preds[val_idx] = clf.predict(X[val_idx])
        print(f"  Fold {fold+1}/3 trained in {time.time()-t_fold:.1f}s | Best iteration: {clf.best_iteration}", flush=True)

    print("\nTop 20 Feature Importances (gain):")
    imp = clf.feature_importance(importance_type='gain')
    for name, gain in sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])[:20]:
        print(f"  {name:25s}: {gain:,.1f}")

    print("\n[6/5] Optimizing Thresholds for Macro F0.5 with Global M2O...", flush=True)
    
    def evaluate_threshold(tau, eval_indices, s_ids):
        mask = oof_preds[eval_indices] >= tau
        cand_indices = eval_indices[mask]
        sorted_cand_idx = cand_indices[np.argsort(-oof_preds[cand_indices])]

        assigned_cids = set()
        pred_dict = {sid: set() for sid in s_ids}
        total_p = 0

        for idx in sorted_cand_idx:
            sid, cid = all_triples[idx]
            if cid not in assigned_cids:
                assigned_cids.add(cid)
                pred_dict[sid].add(cid)
                total_p += 1

        f05 = compute_f05_macro(gt_dict, pred_dict, s_ids)
        avg_preds = total_p / len(s_ids) if len(s_ids) > 0 else 0
        return f05, avg_preds

    best_tau_global = 0.50
    best_f05_global = 0.0
    best_avg_links_global = 0.0

    tau_range = np.arange(0.40, 0.99, 0.01)
    
    print("\n  Sweeping Global Threshold...")
    all_indices = np.arange(len(X))
    for tau in tau_range:
        f05, avg_preds = evaluate_threshold(tau, all_indices, sample_ids)
        marker = " ★ BEST" if f05 > best_f05_global else ""
        if f05 > best_f05_global:
            best_f05_global = f05
            best_tau_global = tau
            best_avg_links_global = avg_preds
        if marker:
            print(f"  tau = {tau:.2f} | Macro F0.5 = {f05:.4f} | Avg links/entity = {avg_preds:.2f}{marker}", flush=True)

    print(f"\n  Global Best: Macro F0.5 = {best_f05_global:.4f} at tau = {best_tau_global:.2f}")

    country_thresholds = {}
    country_f05s = {}
    
    for country in ['US', 'India']:
        print(f"\n  Sweeping Threshold for {country}...")
        c_mask = c_arr == country
        c_indices = np.where(c_mask)[0]
        c_s_ids = np.unique([all_triples[i][0] for i in c_indices])
        
        if len(c_indices) == 0:
            continue
            
        best_tau_c = 0.50
        best_f05_c = 0.0
        
        for tau in tau_range:
            f05, avg_preds = evaluate_threshold(tau, c_indices, c_s_ids)
            marker = " ★ BEST" if f05 > best_f05_c else ""
            if f05 > best_f05_c:
                best_f05_c = f05
                best_tau_c = tau
            if marker:
                print(f"    tau = {tau:.2f} | {country} Macro F0.5 = {f05:.4f}{marker}", flush=True)
                
        country_thresholds[country] = float(best_tau_c)
        country_f05s[country] = float(best_f05_c)
        print(f"  {country} Best: Macro F0.5 = {best_f05_c:.4f} at tau = {best_tau_c:.2f}")

    print("\n" + "=" * 80)
    print(f"  CV BENCHMARK RESULT: Global Macro F0.5 = {best_f05_global:.4f} at tau = {best_tau_global:.2f}")
    print(f"  Country Thresholds: US = {country_thresholds.get('US', 0.5):.2f}, India = {country_thresholds.get('India', 0.5):.2f}")
    print("=" * 80, flush=True)

    print("\nTraining final production model on full sample with hard negative weights...", flush=True)
    full_train = lgb.Dataset(X, label=y, weight=weights, feature_name=FEATURE_NAMES)
    final_model = lgb.train(params, full_train)

    model_path = os.path.join(MODEL_DIR, "lgbm_reranker_v3.txt")
    final_model.save_model(model_path)
    print(f"Saved model to {model_path}", flush=True)

    meta_path = os.path.join(MODEL_DIR, "lgbm_reranker_v3_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "feature_names": FEATURE_NAMES,
            "best_threshold_global": float(best_tau_global),
            "country_thresholds": country_thresholds,
            "best_f05_global": float(best_f05_global),
            "country_f05s": country_f05s,
            "avg_links_global": float(best_avg_links_global),
            "sample_size": sample_size,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"Saved metadata to {meta_path}", flush=True)
    print(f"Total time elapsed: {time.time()-t_start:.1f}s ({time.time()-t_start/60:.1f} min)")

if __name__ == "__main__":
    main()
