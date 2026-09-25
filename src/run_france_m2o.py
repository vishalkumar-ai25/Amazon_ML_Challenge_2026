#!/usr/bin/env python3
"""
Targeted France Pipeline: Two-Channel Blocking + Strict Global One-to-One M2O Assignment
Amazon ML Challenge 2026
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

DATA_DIR = "dataset/student_resource/dataset/test"
OUTPUT_DIR = "output"
PARTS_DIR = "output/parts"
os.makedirs(PARTS_DIR, exist_ok=True)

TOP_K_CHANNEL_A = 30
TOP_K_CHANNEL_B = 30
MAX_UNION_CANDIDATES = 50
DECISION_THRESHOLD = 0.800
BATCH_SIZE = 25000

def normalize_text(text):
    if not text or pd.isna(text) or text == "nan":
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'^(--|<<|>>|\.\.)\s*', '', text)
    text = re.sub(r'\.(com|net|org|co|io)\b', '', text)
    text = re.sub(r"[^\w\s]", ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).strip()

def process_france():
    country = "France"
    print("=" * 80)
    print("  AMAZON ML CHALLENGE 2026 — PROCESSING FRANCE (Two-Channel + Global M2O)")
    print("=" * 80)
    t_start = time.time()
    
    part_match = os.path.join(PARTS_DIR, f"match_{country}.tsv")
    part_cand = os.path.join(PARTS_DIR, f"cand_{country}.tsv")
    
    # 1. Load S1 test records
    print(f"[{country}] Loading S1 test records...", end=" ", flush=True)
    t0 = time.time()
    s1 = pd.read_csv(os.path.join(DATA_DIR, "test_source1.tsv"), sep="\t", dtype=str).fillna("")
    s1 = s1[s1['country'] == country].reset_index(drop=True)
    n_s1 = len(s1)
    print(f"{n_s1:,} records ({time.time()-t0:.1f}s)")
    
    # 2. Load S2 and S3 test corpus
    print(f"[{country}] Loading S2+S3 test corpus...", end=" ", flush=True)
    t0 = time.time()
    s2 = pd.read_csv(os.path.join(DATA_DIR, "test_source2.tsv"), sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(os.path.join(DATA_DIR, "test_source3.tsv"), sep="\t", dtype=str).fillna("")
    
    s2_c = s2[s2['country'] == country]
    s3_c = s3[s3['country'] == country]
    if len(s2_c) == 0 and len(s3_c) == 0:
        s2s3 = pd.concat([s2, s3], ignore_index=True)
    else:
        s2s3 = pd.concat([s2_c, s3_c], ignore_index=True)
        
    del s2, s3, s2_c, s3_c
    gc.collect()
    n_s2s3 = len(s2s3)
    print(f"{n_s2s3:,} records ({time.time()-t0:.1f}s)")
    
    # 3. Text Preparation
    print(f"[{country}] Preparing normalized full-text and address strings...", end=" ", flush=True)
    t0 = time.time()
    s1_full = (s1['business_name'] + " " + s1['business_name'] + " " + s1['business_address']).apply(normalize_text).values
    s2s3_full = (s2s3['business_name'] + " " + s2s3['business_name'] + " " + s2s3['business_address']).apply(normalize_text).values
    
    s1_addr = s1['business_address'].apply(normalize_text).values
    s2s3_addr = s2s3['business_address'].apply(normalize_text).values
    
    s1_ids = s1['entity_id'].values
    s2s3_ids = s2s3['entity_id'].values
    del s1, s2s3
    gc.collect()
    print(f"Done ({time.time()-t0:.1f}s)")
    
    # 4a. Channel A: Full-Text TF-IDF
    print(f"[{country}] [Channel A] Building composite full-text TF-IDF...", end=" ", flush=True)
    t0 = time.time()
    vec_a = TfidfVectorizer(
        analyzer='word',
        max_features=80000,
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
    print(f"Done ({time.time()-t0:.1f}s)")
    
    # 4b. Channel B: Address-only Binary Jaccard
    print(f"[{country}] [Channel B] Building address-only binary matrices...", end=" ", flush=True)
    t0 = time.time()
    vec_b = CountVectorizer(
        binary=True,
        analyzer='word',
        token_pattern=r'(?u)\b\w+\b',
        min_df=2,
        max_df=0.02,
        max_features=60000
    )
    sample_fit_b = np.concatenate([s1_addr[:100000], s2s3_addr[:500000]])
    vec_b.fit(sample_fit_b)
    del sample_fit_b
    gc.collect()
    
    A = vec_b.transform(s1_addr)
    B = vec_b.transform(s2s3_addr)
    B_T = B.T.tocsc()
    len_A = np.diff(A.indptr)
    len_B = np.diff(B.indptr)
    del vec_b, s1_addr, s2s3_addr, B
    gc.collect()
    print(f"Done ({time.time()-t0:.1f}s)")
    
    # 5. Batched Inference & Candidate Generation
    n_batches = int(np.ceil(n_s1 / BATCH_SIZE))
    print(f"[{country}] Running Two-Channel inference across {n_batches} batches (batch_size={BATCH_SIZE})...")
    t_inf = time.time()
    
    s1_triple_indices = []
    cid_triple_values = []
    score_triple_values = []
    
    with open(part_cand, "w", encoding="utf-8") as fc:
        for b in range(n_batches):
            b_start = b * BATCH_SIZE
            b_end = min(b_start + BATCH_SIZE, n_s1)
            
            # Channel A Matmul
            b_sims_a = s1_mat_a[b_start:b_end] @ s2s3_mat_a_T
            if not isinstance(b_sims_a, sparse.csr_matrix):
                b_sims_a = b_sims_a.tocsr()
                
            # Channel B Matmul
            b_inter_b = A[b_start:b_end] @ B_T
            if not isinstance(b_inter_b, sparse.csr_matrix):
                b_inter_b = b_inter_b.tocsr()
                
            for i in range(b_end - b_start):
                global_i = b_start + i
                sid = s1_ids[global_i]
                
                # Channel A top-30
                p0_a, p1_a = b_sims_a.indptr[i], b_sims_a.indptr[i+1]
                top_a = {}
                if p0_a < p1_a:
                    data_a = b_sims_a.data[p0_a:p1_a]
                    indices_a = b_sims_a.indices[p0_a:p1_a]
                    k_a = min(TOP_K_CHANNEL_A, len(data_a))
                    top_idx_a = np.argpartition(data_a, -k_a)[-k_a:] if len(data_a) > TOP_K_CHANNEL_A else np.arange(len(data_a))
                    top_idx_a = top_idx_a[np.argsort(-data_a[top_idx_a])]
                    for idx in top_idx_a:
                        cid = s2s3_ids[indices_a[idx]]
                        top_a[cid] = float(data_a[idx])
                        
                # Channel B top-30
                p0_b, p1_b = b_inter_b.indptr[i], b_inter_b.indptr[i+1]
                top_b = {}
                if p0_b < p1_b and len_A[global_i] > 0:
                    inter_data = b_inter_b.data[p0_b:p1_b]
                    inter_indices = b_inter_b.indices[p0_b:p1_b]
                    denoms = len_A[global_i] + len_B[inter_indices] - inter_data
                    jaccard_scores = inter_data / denoms
                    k_b = min(TOP_K_CHANNEL_B, len(jaccard_scores))
                    top_idx_b = np.argpartition(jaccard_scores, -k_b)[-k_b:] if len(jaccard_scores) > TOP_K_CHANNEL_B else np.arange(len(jaccard_scores))
                    top_idx_b = top_idx_b[np.argsort(-jaccard_scores[top_idx_b])]
                    for idx in top_idx_b:
                        cid = s2s3_ids[inter_indices[idx]]
                        top_b[cid] = float(jaccard_scores[idx])
                        
                # Union of Channel A & Channel B (dedup, max score)
                union_cands = {}
                for cid, s in top_a.items():
                    union_cands[cid] = s
                for cid, s in top_b.items():
                    if cid in union_cands:
                        union_cands[cid] = max(union_cands[cid], s)
                    else:
                        union_cands[cid] = s
                        
                sorted_union = sorted(union_cands.items(), key=lambda x: -x[1])[:MAX_UNION_CANDIDATES]
                cand_id_list = [c[0] for c in sorted_union]
                fc.write(f"{sid}\t{','.join(cand_id_list)}\n")
                
                # Collect post-threshold candidates for Global M2O
                for cid, s in sorted_union:
                    if s >= DECISION_THRESHOLD:
                        s1_triple_indices.append(global_i)
                        cid_triple_values.append(cid)
                        score_triple_values.append(s)
                        
            elapsed = time.time() - t_inf
            pct = 100.0 * (b + 1) / n_batches
            eta = (elapsed / (b + 1)) * (n_batches - b - 1)
            print(f"  [{country}] Batch {b+1}/{n_batches} ({pct:.0f}%) | Processed: {b_end:,} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s", flush=True)
            
    del s1_mat_a, s2s3_mat_a_T, A, B_T, len_A, len_B
    gc.collect()
    
    # 6. Global One-to-One Conflict Resolution
    print(f"[{country}] Resolving Global One-to-One assignments on {len(score_triple_values):,} candidate claims...", end=" ", flush=True)
    t_m2o = time.time()
    
    assigned_matches_per_s1 = {i: [] for i in range(n_s1)}
    if len(score_triple_values) > 0:
        scores_arr = np.array(score_triple_values, dtype=np.float32)
        sort_order = np.argsort(-scores_arr)
        del scores_arr
        
        assigned_cids = set()
        for idx in sort_order:
            cid = cid_triple_values[idx]
            if cid not in assigned_cids:
                assigned_cids.add(cid)
                s1_idx = s1_triple_indices[idx]
                assigned_matches_per_s1[s1_idx].append(cid)
                
        del sort_order, assigned_cids
    del s1_triple_indices, cid_triple_values, score_triple_values
    gc.collect()
    print(f"Done in {time.time()-t_m2o:.1f}s")
    
    # Write resolved matches to part_match
    print(f"[{country}] Writing resolved matches to {part_match}...", end=" ", flush=True)
    country_matched = 0
    country_preds = 0
    with open(part_match, "w", encoding="utf-8") as fm:
        for i in range(n_s1):
            sid = s1_ids[i]
            matched_list = assigned_matches_per_s1[i]
            if matched_list:
                country_matched += 1
                country_preds += len(matched_list)
                fm.write(f"{sid}\t{','.join(matched_list)}\n")
            else:
                fm.write(f"{sid}\t\n")
    print("Done")
    
    del assigned_matches_per_s1, s1_ids, s2s3_ids
    gc.collect()
    
    total_time = time.time() - t_start
    print(f"[{country}] Finished in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"[{country}] S1 Entities: {n_s1:,} | Matched: {country_matched:,} ({100.0*country_matched/n_s1:.1f}%) | Preds: {country_preds:,}")

if __name__ == "__main__":
    process_france()
