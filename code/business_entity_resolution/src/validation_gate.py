#!/usr/bin/env python3
"""
VALIDATION GATE: Two-Channel Blocking + Global One-to-One Assignment
Evaluated on the exact 30k stratified training sample (same seed=42 as sprint1_eval.py).

Channel A: TF-IDF on 2*name + address (top-30)
Channel B: Address-only EXACT word-Jaccard via sparse binary matmul (top-30)
Candidate Union: Union of Channel A & Channel B (dedup, cap at 50)
Scoring & M2O: Post-threshold global greedy one-to-one assignment on S2/S3 IDs.
"""

import os
import sys
import time
import gc
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse

DATA_ROOT = "dataset/student_resource/dataset"

def normalize_text(text):
    if not text or pd.isna(text) or text == "nan":
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'^(--|<<|>>|\.\.)\s*', '', text)
    text = re.sub(r'\.(com|net|org|co|io)\b', '', text)
    text = re.sub(r"[^\w\s]", ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).strip()

def compute_f05_macro(true_dict, pred_dict, all_ids):
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

def run_validation_gate(sample_size=30000):
    print("=" * 80)
    print("  AMAZON ML CHALLENGE 2026 — VALIDATION GATE")
    print("  Testing Two-Channel Blocking + Global One-to-One Assignment")
    print("=" * 80)
    t_gate_start = time.time()
    
    # 1. Load train S1 and sample identically to sprint1_eval.py
    print("\n[1/5] Loading 30k stratified training sample (seed=42)...", end=" ", flush=True)
    t0 = time.time()
    s1 = pd.read_csv(f"{DATA_ROOT}/train/train_source1.tsv", sep="\t", dtype=str).fillna("")
    np.random.seed(42)
    sample_indices = []
    for country in s1['country'].unique():
        cidx = s1[s1['country'] == country].index.values
        n = max(1, int(sample_size * len(cidx) / len(s1)))
        sample_indices.extend(np.random.choice(cidx, size=min(n, len(cidx)), replace=False))
        
    sample_s1 = s1.loc[sample_indices].reset_index(drop=True)
    sample_ids = sample_s1['entity_id'].values
    print(f"Done in {time.time()-t0:.1f}s ({len(sample_s1):,} S1 entities)")
    for c in sorted(sample_s1['country'].unique()):
        print(f"       Country {c}: {int((sample_s1['country']==c).sum()):,}")
        
    # 2. Load ground truth for sample
    print("\n[2/5] Loading ground truth...", end=" ", flush=True)
    t0 = time.time()
    gt = pd.read_csv(f"{DATA_ROOT}/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    gt_map = dict(zip(gt['source1_entity_id'], gt['matched_entity_ids']))
    gt_dict = {}
    for sid in sample_ids:
        mids = str(gt_map.get(sid, "")).strip()
        gt_dict[sid] = set(mids.split(",")) if mids else set()
    total_true = sum(len(v) for v in gt_dict.values())
    print(f"Done in {time.time()-t0:.1f}s (Total true match links: {total_true:,})")
    del s1, gt, gt_map
    gc.collect()
    
    # 3. Load S2 and S3 pools
    print("\n[3/5] Loading S2 and S3 pools...", end=" ", flush=True)
    t0 = time.time()
    s2 = pd.read_csv(f"{DATA_ROOT}/train/train_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(f"{DATA_ROOT}/train/train_source3.tsv", sep="\t", dtype=str).fillna("")
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()
    print(f"Done in {time.time()-t0:.1f}s ({len(s2s3):,} S2/S3 records)")
    
    # Dictionaries to accumulate candidates and scores across countries
    cand_chA = {sid: {} for sid in sample_ids}       # sid -> {cid: score_A}
    cand_chB = {sid: {} for sid in sample_ids}       # sid -> {cid: score_B}
    cand_union = {sid: {} for sid in sample_ids}     # sid -> {cid: combined_score}
    
    # 4. Two-Channel Blocking per Country
    print("\n[4/5] Executing Two-Channel Blocking...")
    for country in sorted(sample_s1['country'].unique()):
        print(f"\n--- Country: {country} ---")
        t_country = time.time()
        s1_c = sample_s1[sample_s1['country'] == country].reset_index(drop=True)
        s2s3_c = s2s3[s2s3['country'] == country].reset_index(drop=True)
        n_s1 = len(s1_c)
        n_s2s3 = len(s2s3_c)
        print(f"  S1 queries: {n_s1:,} | S2/S3 corpus: {n_s2s3:,}")
        
        s1_ids_c = s1_c['entity_id'].values
        s2s3_ids_c = s2s3_c['entity_id'].values
        
        # 4a. Channel A: TF-IDF on 2*name + address
        print("  [Channel A] Building composite full-text TF-IDF...", end=" ", flush=True)
        t_ca = time.time()
        s1_full = (s1_c['business_name'] + " " + s1_c['business_name'] + " " + s1_c['business_address']).apply(normalize_text).values
        s2s3_full = (s2s3_c['business_name'] + " " + s2s3_c['business_name'] + " " + s2s3_c['business_address']).apply(normalize_text).values
        
        vec_a = TfidfVectorizer(
            analyzer='word',
            max_features=150000,
            min_df=2,
            max_df=0.01,
            sublinear_tf=True,
            norm='l2',
            dtype=np.float32,
            token_pattern=r'(?u)\b\w+\b'
        )
        sample_fit_a = np.concatenate([s1_full[:100000], s2s3_full[:500000]])
        vec_a.fit(sample_fit_a)
        del sample_fit_a
        gc.collect()
        
        s1_mat_a = vec_a.transform(s1_full)
        s2s3_mat_a = vec_a.transform(s2s3_full)
        s2s3_mat_a_T = s2s3_mat_a.T.tocsc()
        del vec_a, s1_full, s2s3_full, s2s3_mat_a
        gc.collect()
        print(f"Done in {time.time()-t_ca:.1f}s")
        
        # 4b. Channel B: Address-only EXACT word-Jaccard via sparse binary vectors
        print("  [Channel B] Building address-only binary presence matrices...", end=" ", flush=True)
        t_cb = time.time()
        s1_addr = s1_c['business_address'].apply(normalize_text).values
        s2s3_addr = s2s3_c['business_address'].apply(normalize_text).values
        
        vec_b = CountVectorizer(
            binary=True,
            analyzer='word',
            token_pattern=r'(?u)\b\w+\b',
            min_df=2,
            max_df=0.02,
            max_features=100000
        )
        sample_fit_b = np.concatenate([s1_addr[:100000], s2s3_addr[:500000]])
        vec_b.fit(sample_fit_b)
        del sample_fit_b
        gc.collect()
        
        A = vec_b.transform(s1_addr)       # (n_s1, V) binary CSR
        B = vec_b.transform(s2s3_addr)     # (n_s2s3, V) binary CSR
        B_T = B.T.tocsc()                  # (V, n_s2s3) binary CSC
        
        # Row token totals
        len_A = np.diff(A.indptr)          # (n_s1,) token counts
        len_B = np.diff(B.indptr)          # (n_s2s3,) token counts
        del vec_b, s1_addr, s2s3_addr, B
        gc.collect()
        print(f"Done in {time.time()-t_cb:.1f}s (A: {A.shape}, B: ({n_s2s3}, {B_T.shape[0]}))")
        
        # 4c. Batch inference for Country
        BATCH_SIZE = 10000
        n_batches = (n_s1 + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Computing similarities across {n_batches} batches...")
        t_sim = time.time()
        
        for b in range(n_batches):
            b_start = b * BATCH_SIZE
            b_end = min(b_start + BATCH_SIZE, n_s1)
            
            # --- Channel A Sparse Matmul ---
            b_sims_a = s1_mat_a[b_start:b_end] @ s2s3_mat_a_T
            if not isinstance(b_sims_a, sparse.csr_matrix):
                b_sims_a = b_sims_a.tocsr()
                
            # --- Channel B Sparse Matmul (Intersection Counts) ---
            b_inter_b = A[b_start:b_end] @ B_T
            if not isinstance(b_inter_b, sparse.csr_matrix):
                b_inter_b = b_inter_b.tocsr()
                
            for i in range(b_end - b_start):
                global_i = b_start + i
                sid = s1_ids_c[global_i]
                
                # --- Channel A top-30 ---
                p0_a, p1_a = b_sims_a.indptr[i], b_sims_a.indptr[i+1]
                top_a = {}
                if p0_a < p1_a:
                    data_a = b_sims_a.data[p0_a:p1_a]
                    indices_a = b_sims_a.indices[p0_a:p1_a]
                    k_a = min(30, len(data_a))
                    top_idx_a = np.argpartition(data_a, -k_a)[-k_a:] if len(data_a) > 30 else np.arange(len(data_a))
                    top_idx_a = top_idx_a[np.argsort(-data_a[top_idx_a])]
                    for idx in top_idx_a:
                        cid = s2s3_ids_c[indices_a[idx]]
                        score_val = float(data_a[idx])
                        top_a[cid] = score_val
                        cand_chA[sid][cid] = score_val
                        
                # --- Channel B top-30 by exact Jaccard ---
                p0_b, p1_b = b_inter_b.indptr[i], b_inter_b.indptr[i+1]
                top_b = {}
                if p0_b < p1_b and len_A[global_i] > 0:
                    inter_data = b_inter_b.data[p0_b:p1_b]
                    inter_indices = b_inter_b.indices[p0_b:p1_b]
                    
                    # Vectorized Jaccard: inter / (len_A[i] + len_B[j] - inter)
                    denoms = len_A[global_i] + len_B[inter_indices] - inter_data
                    jaccard_scores = inter_data / denoms
                    
                    k_b = min(30, len(jaccard_scores))
                    top_idx_b = np.argpartition(jaccard_scores, -k_b)[-k_b:] if len(jaccard_scores) > 30 else np.arange(len(jaccard_scores))
                    top_idx_b = top_idx_b[np.argsort(-jaccard_scores[top_idx_b])]
                    for idx in top_idx_b:
                        cid = s2s3_ids_c[inter_indices[idx]]
                        score_val = float(jaccard_scores[idx])
                        top_b[cid] = score_val
                        cand_chB[sid][cid] = score_val
                        
                # --- Union of Channel A & Channel B (dedup; max score) ---
                union_cands = {}
                for cid, s in top_a.items():
                    union_cands[cid] = s
                for cid, s in top_b.items():
                    if cid in union_cands:
                        union_cands[cid] = max(union_cands[cid], s)
                    else:
                        union_cands[cid] = s
                cand_union[sid] = union_cands
                
            print(f"    Batch {b+1}/{n_batches} processed ({time.time()-t_sim:.1f}s elapsed)")
            
        del s1_mat_a, s2s3_mat_a_T, A, B_T, len_A, len_B, s1_ids_c, s2s3_ids_c
        gc.collect()
        print(f"  Finished {country} in {time.time()-t_country:.1f}s")
        
    del s2s3, sample_s1
    gc.collect()
    
    # 5. Measure Blocking Pair Completeness (Recall)
    print("\n[5/5] Evaluating Blocking Pair Completeness...")
    found_A = sum(len(gt_dict[sid] & set(cand_chA[sid].keys())) for sid in sample_ids)
    found_B = sum(len(gt_dict[sid] & set(cand_chB[sid].keys())) for sid in sample_ids)
    found_union = sum(len(gt_dict[sid] & set(cand_union[sid].keys())) for sid in sample_ids)
    
    pc_A = found_A / total_true
    pc_B = found_B / total_true
    pc_union = found_union / total_true
    
    print("\n" + "=" * 80)
    print(f"  BLOCKING PAIR COMPLETENESS (PC) COMPARISON (on {total_true:,} true links):")
    print(f"    Channel A only (Baseline TF-IDF top-30):    {pc_A:.4f} ({found_A:,} / {total_true:,})")
    print(f"    Channel B only (Address Jaccard top-30):    {pc_B:.4f} ({found_B:,} / {total_true:,})")
    print(f"    Two-Channel Union (Channel A ∪ Channel B): {pc_union:.4f} ({found_union:,} / {total_true:,})")
    print(f"    Absolute Recall Gain:                     {+(pc_union - pc_A)*100:+.2f}%")
    print("=" * 80)
    
    # 6. Evaluate F_0.5 at tau=0.780 and fresh tau re-sweep (0.20 to 0.92)
    print("\n--- Sweeping Thresholds with Global One-to-One Assignment ---")
    
    # Helper to apply M2O given a threshold
    def evaluate_with_m2o(tau):
        # Collect all post-threshold triples
        s1_list = []
        cid_list = []
        score_list = []
        for sid in sample_ids:
            for cid, s in cand_union[sid].items():
                if s >= tau:
                    s1_list.append(sid)
                    cid_list.append(cid)
                    score_list.append(s)
                    
        if not score_list:
            pred_dict = {sid: set() for sid in sample_ids}
            return compute_f05_macro(gt_dict, pred_dict, sample_ids)
            
        scores_arr = np.array(score_list, dtype=np.float32)
        sort_order = np.argsort(-scores_arr)
        
        assigned_cands = set()
        pred_dict = {sid: set() for sid in sample_ids}
        
        for idx in sort_order:
            cid = cid_list[idx]
            if cid not in assigned_cands:
                assigned_cands.add(cid)
                pred_dict[s1_list[idx]].add(cid)
                
        return compute_f05_macro(gt_dict, pred_dict, sample_ids)
        
    def evaluate_without_m2o(tau):
        pred_dict = {}
        for sid in sample_ids:
            pred_dict[sid] = {cid for cid, s in cand_union[sid].items() if s >= tau}
        return compute_f05_macro(gt_dict, pred_dict, sample_ids)

    # First evaluate at tau = 0.780
    f05_old_tau_nom2o = evaluate_without_m2o(0.780)
    f05_old_tau_m2o = evaluate_with_m2o(0.780)
    print(f"  τ = 0.780 (without M2O): F₀.₅ = {f05_old_tau_nom2o:.4f}")
    print(f"  τ = 0.780 (with M2O):    F₀.₅ = {f05_old_tau_m2o:.4f}")
    
    # Fresh tau sweep from 0.20 to 0.92
    best_tau = 0.780
    best_f05 = f05_old_tau_m2o
    
    print("\n  Full Threshold Re-Sweep (0.20 to 0.92 in steps of 0.04):")
    sweep_taus = np.arange(0.20, 0.94, 0.04)
    for tau in sweep_taus:
        f05_val = evaluate_with_m2o(tau)
        marker = " ★ BEST" if f05_val > best_f05 else ""
        if f05_val > best_f05:
            best_f05 = f05_val
            best_tau = tau
        print(f"    τ = {tau:.3f} | Macro F₀.₅ = {f05_val:.4f}{marker}")
        
    # Also fine sweep around best_tau
    fine_taus = np.arange(max(0.20, best_tau - 0.04), min(0.94, best_tau + 0.05), 0.01)
    for tau in fine_taus:
        f05_val = evaluate_with_m2o(tau)
        marker = " ★ BEST" if f05_val > best_f05 else ""
        if f05_val > best_f05:
            best_f05 = f05_val
            best_tau = tau
        print(f"    [fine] τ = {tau:.3f} | Macro F₀.₅ = {f05_val:.4f}{marker}")
        
    # Summary Before / After Table
    baseline_pc = 0.8997
    baseline_f05 = 0.6636
    
    print("\n" + "=" * 80)
    print("  VALIDATION GATE: BEFORE / AFTER COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Pipeline':<45} | {'Pair Completeness':<18} | {'Macro F_0.5':<12}")
    print("-" * 80)
    print(f"{'Old Baseline (Single-Channel, No M2O, τ=0.750)':<45} | {baseline_pc*100:>17.2f}% | {baseline_f05:>12.4f}")
    print(f"{'Old Pipeline at τ=0.780 (Single-Channel, No M2O)':<45} | {baseline_pc*100:>17.2f}% | {0.6648:>12.4f}")
    print(f"{'Patched Pipeline at τ=0.780 (Two-Channel + M2O)':<45} | {pc_union*100:>17.2f}% | {f05_old_tau_m2o:>12.4f}")
    print(f"{f'Patched Pipeline at Best τ={best_tau:.3f} (Two-Channel + M2O)':<45} | {pc_union*100:>17.2f}% | {best_f05:>12.4f}")
    print("=" * 80)
    
    passed = (pc_union > baseline_pc) and (best_f05 > baseline_f05)
    print(f"\nVALIDATION GATE STATUS: {'PASSED [OK TO SCALE]' if passed else 'FAILED'}")
    print(f"Total validation elapsed: {time.time()-t_gate_start:.1f}s")
    return passed, best_tau, best_f05, pc_union

if __name__ == "__main__":
    run_validation_gate(sample_size=30000)
