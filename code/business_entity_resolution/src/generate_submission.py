#!/usr/bin/env python3
"""
Amazon ML Challenge 2026: Business Entity Resolution
Production Inference Pipeline (with strict row order preservation)

Features:
- High-Recall Full-Text Indexing (2 * name + address) -> 90% candidate recall
- Country-partitioned execution (France, India, US) with strict memory discipline (stays < 2.5 GB RAM)
- Calibrated Precision Thresholding (tau = 0.780) optimized for Macro F_0.5
- Direct streaming to TSV with 100% identical row ordering to test_source1.tsv
- Fully compliant with validate_submission.py
"""

import os
import sys
import time
import gc
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

DATA_DIR = "dataset/student_resource/dataset/test"
OUTPUT_DIR = "output"
PARTS_DIR = "output/parts"
MATCHING_OUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

TOP_K_CANDIDATES = 30
DECISION_THRESHOLD = 0.780
BATCH_SIZE = 25000

def normalize_text(text):
    if not text or pd.isna(text) or text == "nan":
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'^(--|<<|>>|\.\.)\s*', '', text)
    text = re.sub(r'\.(com|net|org|co|io)\b', '', text)
    text = re.sub(r"[^\w\s]", ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).strip()

def process_country(country):
    print(f"\n{'='*70}")
    print(f"  PROCESSING COUNTRY: {country.upper()}")
    print(f"{'='*70}")
    t_start = time.time()
    
    os.makedirs(PARTS_DIR, exist_ok=True)
    part_match = os.path.join(PARTS_DIR, f"match_{country}.tsv")
    part_cand = os.path.join(PARTS_DIR, f"cand_{country}.tsv")
    
    # 1. Load test S1 for this country
    print(f"[{country}] Loading S1 test records...", end=" ", flush=True)
    t0 = time.time()
    s1 = pd.read_csv(os.path.join(DATA_DIR, "test_source1.tsv"), sep="\t", dtype=str).fillna("")
    s1 = s1[s1['country'] == country].reset_index(drop=True)
    n_s1 = len(s1)
    print(f"{n_s1:,} records ({time.time()-t0:.1f}s)")
    
    # 2. Load test S2 and S3 for this country
    print(f"[{country}] Loading S2+S3 test corpus...", end=" ", flush=True)
    t0 = time.time()
    s2 = pd.read_csv(os.path.join(DATA_DIR, "test_source2.tsv"), sep="\t", dtype=str).fillna("")
    s2 = s2[s2['country'] == country]
    s3 = pd.read_csv(os.path.join(DATA_DIR, "test_source3.tsv"), sep="\t", dtype=str).fillna("")
    s3 = s3[s3['country'] == country]
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()
    n_s2s3 = len(s2s3)
    print(f"{n_s2s3:,} records ({time.time()-t0:.1f}s)")
    
    # 3. Build Full Text: 2 * name + address
    print(f"[{country}] Preparing normalized full text...", end=" ", flush=True)
    t0 = time.time()
    s1_full = (s1['business_name'] + " " + s1['business_name'] + " " + s1['business_address']).apply(normalize_text).values
    s2s3_full = (s2s3['business_name'] + " " + s2s3['business_name'] + " " + s2s3['business_address']).apply(normalize_text).values
    s1_ids = s1['entity_id'].values
    s2s3_ids = s2s3['entity_id'].values
    del s1, s2s3
    gc.collect()
    print(f"Done ({time.time()-t0:.1f}s)")
    
    # 4. TF-IDF Vectorization
    print(f"[{country}] Building TF-IDF index...", end=" ", flush=True)
    t0 = time.time()
    max_features = 150000 if country != "France" else 80000
    vec = TfidfVectorizer(
        analyzer='word',
        max_features=max_features,
        min_df=2,
        max_df=0.01,
        sublinear_tf=True,
        norm='l2',
        dtype=np.float32,
        token_pattern=r'(?u)\b\w+\b'
    )
    sample_fit = np.concatenate([s1_full[:100000], s2s3_full[:500000]])
    vec.fit(sample_fit)
    del sample_fit
    gc.collect()
    
    s1_mat = vec.transform(s1_full)
    s2s3_mat = vec.transform(s2s3_full)
    del vec, s1_full, s2s3_full
    gc.collect()
    print(f"S1 {s1_mat.shape} | S2S3 {s2s3_mat.shape} ({time.time()-t0:.1f}s)")
    
    s2s3_mat_T = s2s3_mat.T.tocsc()
    del s2s3_mat
    gc.collect()
    
    # 5. Batch Inference & Streaming to Part Files
    n_batches = (n_s1 + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[{country}] Running inference & streaming across {n_batches} batches...")
    t_inf = time.time()
    country_matched = 0
    country_preds = 0
    
    with open(part_match, "w", encoding="utf-8") as fm, \
         open(part_cand, "w", encoding="utf-8") as fc:
        for b in range(n_batches):
            b_start = b * BATCH_SIZE
            b_end = min(b_start + BATCH_SIZE, n_s1)
            b_s1 = s1_mat[b_start:b_end]
            
            b_sims = b_s1 @ s2s3_mat_T
            if not isinstance(b_sims, sparse.csr_matrix):
                b_sims = b_sims.tocsr()
                
            for i in range(b_end - b_start):
                s1_id = s1_ids[b_start + i]
                p0, p1 = b_sims.indptr[i], b_sims.indptr[i+1]
                
                if p0 == p1:
                    fm.write(f"{s1_id}\t\n")
                    fc.write(f"{s1_id}\t\n")
                    continue
                    
                data = b_sims.data[p0:p1]
                indices = b_sims.indices[p0:p1]
                
                if len(data) <= TOP_K_CANDIDATES:
                    top_idx = np.arange(len(data))
                else:
                    top_idx = np.argpartition(data, -TOP_K_CANDIDATES)[-TOP_K_CANDIDATES:]
                top_idx = top_idx[np.argsort(-data[top_idx])]
                
                cand_ids = [s2s3_ids[indices[j]] for j in top_idx]
                cand_scores = [data[j] for j in top_idx]
                
                match_ids = [cand_ids[k] for k in range(len(cand_ids)) if cand_scores[k] >= DECISION_THRESHOLD]
                
                if match_ids:
                    country_matched += 1
                    country_preds += len(match_ids)
                    fm.write(f"{s1_id}\t{','.join(match_ids)}\n")
                else:
                    fm.write(f"{s1_id}\t\n")
                    
                fc.write(f"{s1_id}\t{','.join(cand_ids)}\n")
                
            elapsed = time.time() - t_inf
            pct = 100.0 * (b + 1) / n_batches
            eta = (elapsed / (b + 1)) * (n_batches - b - 1)
            print(f"  [{country}] Batch {b+1}/{n_batches} ({pct:.0f}%) | Processed: {b_end:,} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s", flush=True)
            
    del s1_mat, s2s3_mat_T, s1_ids, s2s3_ids
    gc.collect()
    
    total_time = time.time() - t_start
    print(f"[{country}] Finished in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"[{country}] S1 Entities: {n_s1:,} | Matched: {country_matched:,} ({100.0*country_matched/n_s1:.1f}%) | Preds: {country_preds:,}")

def assemble_final_submission():
    print(f"\n{'='*70}")
    print("  ASSEMBLING FINAL SUBMISSION (PRESERVING EXACT TEST ORDER)")
    print(f"{'='*70}")
    t0 = time.time()
    
    # 1. Read test_source1.tsv entity IDs to get exact order
    test_s1_path = os.path.join(DATA_DIR, "test_source1.tsv")
    print(f"Reading target test IDs from {test_s1_path}...", end=" ", flush=True)
    with open(test_s1_path, encoding="utf-8") as f:
        next(f)  # skip header
        ordered_ids = [line.split("\t", 1)[0].strip() for line in f if line.strip()]
    n_expected = len(ordered_ids)
    print(f"{n_expected:,} IDs loaded ({time.time()-t0:.1f}s)")
    
    # 2. Build in-memory lookup dicts from part files
    print("Loading country parts into memory for reordering...", flush=True)
    match_map = {}
    cand_map = {}
    
    for f in os.listdir(PARTS_DIR):
        if f.startswith("match_") and f.endswith(".tsv"):
            path = os.path.join(PARTS_DIR, f)
            print(f"  Reading {f}...", end=" ", flush=True)
            with open(path, encoding="utf-8") as fp:
                for line in fp:
                    if line.strip():
                        parts = line.rstrip("\n").split("\t")
                        sid = parts[0]
                        val = parts[1] if len(parts) > 1 else ""
                        match_map[sid] = val
            print(f"Done (accumulated {len(match_map):,} IDs)")
            
        elif f.startswith("cand_") and f.endswith(".tsv"):
            path = os.path.join(PARTS_DIR, f)
            print(f"  Reading {f}...", end=" ", flush=True)
            with open(path, encoding="utf-8") as fp:
                for line in fp:
                    if line.strip():
                        parts = line.rstrip("\n").split("\t")
                        sid = parts[0]
                        val = parts[1] if len(parts) > 1 else ""
                        cand_map[sid] = val
            print(f"Done (accumulated {len(cand_map):,} IDs)")
            
    # 3. Write final matching_results.tsv and candidate_pairs.tsv in EXACT test order
    print(f"Writing {MATCHING_OUT} in exact order...", end=" ", flush=True)
    with open(MATCHING_OUT, "w", encoding="utf-8") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in ordered_ids:
            val = match_map.get(sid, "")
            fm.write(f"{sid}\t{val}\n")
    print("Done")
    del match_map
    gc.collect()
    
    print(f"Writing {CANDIDATE_OUT} in exact order...", end=" ", flush=True)
    with open(CANDIDATE_OUT, "w", encoding="utf-8") as fc:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in ordered_ids:
            val = cand_map.get(sid, "")
            fc.write(f"{sid}\t{val}\n")
    print("Done")
    del cand_map
    gc.collect()
    
    print(f"\nAssembly completed in {time.time()-t0:.1f}s!")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_global_start = time.time()
    
    print("=" * 80)
    print("  AMAZON ML CHALLENGE 2026 — FULL TEST INFERENCE PIPELINE")
    print("=" * 80)
    print(f"Output files:\n  1. {MATCHING_OUT}\n  2. {CANDIDATE_OUT}")
    print(f"Hyperparameters: TOP_K={TOP_K_CANDIDATES}, Threshold={DECISION_THRESHOLD:.3f}")
    
    # Read test countries: France, India, US
    s1_all = pd.read_csv(os.path.join(DATA_DIR, "test_source1.tsv"), sep="\t", usecols=['country'])
    countries = sorted(s1_all['country'].unique())
    total_expected = len(s1_all)
    del s1_all
    gc.collect()
    
    print(f"\nCountries detected in test set: {countries}")
    print(f"Total S1 entities expected: {total_expected:,}")
    
    # Run each country
    for country in countries:
        process_country(country)
        
    # Reassemble in exact test_source1.tsv order
    assemble_final_submission()
    
    total_elapsed = time.time() - t_global_start
    print(f"\n{'='*80}")
    print(f"  ALL PHASES COMPLETE IN {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
