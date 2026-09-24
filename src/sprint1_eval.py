"""
Sprint 1: High-Recall Full-Text Blocking + Precision Scoring Pipeline
Evaluated on 30,000 S1 validation sample.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import re
import time
import gc
import os

DATA_ROOT = "dataset/student_resource/dataset"

def normalize_text(text):
    if not text or pd.isna(text) or text == "nan":
        return ""
    text = text.lower().strip()
    text = re.sub(r'^(--|<<|>>|\.\.)\s*', '', text)
    text = re.sub(r'\.(com|net|org|co|io)\b', '', text)
    text = re.sub(r"[^\w\s]", ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).strip()

def compute_f05_per_entity(true_set, pred_set):
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0
    if len(true_set) == 0 or len(pred_set) == 0:
        return 0.0
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0
    precision = tp / (tp + len(pred_set - true_set))
    recall = tp / (tp + len(true_set - pred_set))
    return 1.25 * precision * recall / (0.25 * precision + recall)

def evaluate_pipeline(sample_size=30000, top_k=35):
    print("="*70)
    print(f"EVALUATING HIGH-RECALL PIPELINE ON {sample_size:,} S1 SAMPLE")
    print("="*70)
    
    # 1. Load train S1 and sample
    s1 = pd.read_csv(f"{DATA_ROOT}/train/train_source1.tsv", sep="\t", dtype=str).fillna("")
    np.random.seed(42)
    sample_indices = []
    for country in s1['country'].unique():
        cidx = s1[s1['country'] == country].index.values
        n = max(1, int(sample_size * len(cidx) / len(s1)))
        sample_indices.extend(np.random.choice(cidx, size=min(n, len(cidx)), replace=False))
        
    sample_s1 = s1.loc[sample_indices].reset_index(drop=True)
    sample_ids = sample_s1['entity_id'].values
    print(f"Sampled {len(sample_s1):,} S1: " + ", ".join(f"{c}={int((sample_s1['country']==c).sum()):,}" for c in sorted(sample_s1['country'].unique())))
    
    # 2. Load ground truth for sample
    gt = pd.read_csv(f"{DATA_ROOT}/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    gt_map = dict(zip(gt['source1_entity_id'], gt['matched_entity_ids']))
    gt_dict = {}
    for sid in sample_ids:
        mids = str(gt_map.get(sid, "")).strip()
        gt_dict[sid] = set(mids.split(",")) if mids else set()
        
    del s1, gt, gt_map
    gc.collect()
    
    # 3. Load S2 and S3
    s2 = pd.read_csv(f"{DATA_ROOT}/train/train_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(f"{DATA_ROOT}/train/train_source3.tsv", sep="\t", dtype=str).fillna("")
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()
    
    all_candidates = {}
    total_true = sum(len(v) for v in gt_dict.values())
    found_true = 0
    
    # Process per country
    for country in sorted(sample_s1['country'].unique()):
        print(f"\n--- Processing Country: {country} ---")
        t0 = time.time()
        s1_c = sample_s1[sample_s1['country'] == country].reset_index(drop=True)
        s2s3_c = s2s3[s2s3['country'] == country].reset_index(drop=True)
        n_s1 = len(s1_c)
        n_s2s3 = len(s2s3_c)
        print(f"  S1: {n_s1:,} | S2+S3: {n_s2s3:,}")
        
        # Build full text: name*2 + address
        s1_full = (s1_c['business_name'] + " " + s1_c['business_name'] + " " + s1_c['business_address']).apply(normalize_text).values
        s2s3_full = (s2s3_c['business_name'] + " " + s2s3_c['business_name'] + " " + s2s3_c['business_address']).apply(normalize_text).values
        
        s1_ids_c = s1_c['entity_id'].values
        s2s3_ids_c = s2s3_c['entity_id'].values
        
        # TF-IDF
        vec = TfidfVectorizer(
            analyzer='word',
            max_features=150000,
            min_df=2,
            max_df=0.01,
            sublinear_tf=True,
            norm='l2',
            dtype=np.float32,
            token_pattern=r'(?u)\b\w+\b'
        )
        vec.fit(np.concatenate([s1_full, s2s3_full[:500000]]))
        s1_mat = vec.transform(s1_full)
        s2s3_mat = vec.transform(s2s3_full)
        print(f"  Vectorized in {time.time()-t0:.1f}s")
        
        # Similarity in batches
        BATCH_SIZE = 5000
        n_batches = (n_s1 + BATCH_SIZE - 1) // BATCH_SIZE
        t_sim = time.time()
        for b in range(n_batches):
            b_start = b * BATCH_SIZE
            b_end = min(b_start + BATCH_SIZE, n_s1)
            b_sims = s1_mat[b_start:b_end] @ s2s3_mat.T
            if not isinstance(b_sims, sparse.csr_matrix):
                b_sims = b_sims.tocsr()
                
            for i in range(b_end - b_start):
                sid = s1_ids_c[b_start + i]
                row = b_sims.getrow(i)
                if row.nnz == 0:
                    all_candidates[sid] = []
                    continue
                data = row.data
                indices = row.indices
                if len(data) <= top_k:
                    top_idx = np.arange(len(data))
                else:
                    top_idx = np.argpartition(data, -top_k)[-top_k:]
                top_idx = top_idx[np.argsort(-data[top_idx])]
                
                cands = [(s2s3_ids_c[indices[j]], float(data[j])) for j in top_idx]
                all_candidates[sid] = cands
                
                true_set = gt_dict.get(sid, set())
                cand_set = {c[0] for c in cands}
                found_true += len(true_set & cand_set)
                
        print(f"  Similarities computed in {time.time()-t_sim:.1f}s")
        del vec, s1_mat, s2s3_mat, s1_full, s2s3_full, s1_c, s2s3_c
        gc.collect()
        
    overall_recall = found_true / max(total_true, 1)
    print("\n" + "="*70)
    print(f"OVERALL BLOCKING RECALL: {overall_recall:.4f} ({found_true:,} / {total_true:,})")
    print("="*70)
    
    # 4. Sweep thresholds on full similarity score
    print("\n--- Sweeping Thresholds ---")
    best_f05 = 0
    best_tau = 0.5
    for tau in np.arange(0.20, 0.76, 0.025):
        scores = []
        for sid in sample_ids:
            true_set = gt_dict.get(sid, set())
            pred_set = {cid for cid, s in all_candidates.get(sid, []) if s >= tau}
            scores.append(compute_f05_per_entity(true_set, pred_set))
        mean_f05 = np.mean(scores)
        marker = " ★ BEST" if mean_f05 > best_f05 else ""
        if mean_f05 > best_f05:
            best_f05 = mean_f05
            best_tau = tau
        print(f"  τ={tau:.3f} | F₀.₅ = {mean_f05:.4f}{marker}")
        
    print(f"\n★ Optimal Threshold τ={best_tau:.3f} achieves F₀.₅ = {best_f05:.4f}!")

if __name__ == "__main__":
    evaluate_pipeline(sample_size=30000, top_k=35)
