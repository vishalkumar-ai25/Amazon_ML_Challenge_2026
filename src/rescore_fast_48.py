#!/usr/bin/env python3
"""
High-Speed 48-Core Vectorized SOTA Re-Scorer & Submission Engine
Amazon ML Challenge 2026

Fully parallelized across 48 cores:
1. 48-core parallel S1 metadata parsing (3s)
2. 48-core parallel candidate metadata parsing (20s)
3. 48-core chunk-level vectorized LightGBM scoring (90s)
4. Global Greedy M2O + Precision Hardening Guards (30s)
5. TSV export, validator check, and ZIP packaging (30s)

Total runtime: ~3 to 4 minutes!
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import sys
import time
import gc
import json
import zipfile
import multiprocessing as mp
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_sota import (
    FEATURE_NAMES, compute_record_meta, extract_features_single
)

DATA_DIR = "dataset/student_resource/dataset/test"
OUTPUT_DIR = "output"
PARTS_V2_DIR = os.path.join(OUTPUT_DIR, "parts_v2")
PARTS_SOTA_DIR = os.path.join(OUTPUT_DIR, "parts_sota_098")
MODEL_PATH = "models/lgbm_sota_500k.txt"
META_PATH = "models/lgbm_sota_500k_meta.json"

MATCHING_OUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
ZIP_OUT = os.path.join(OUTPUT_DIR, "matching_results.zip")

def _meta_worker(records_chunk):
    res = {}
    for cid, n, a, c in records_chunk:
        res[cid] = compute_record_meta(n, a, c)
    return res

_G_RESCORER = None
_MODEL = None

def _rescore_chunk_worker(chunk_items):
    global _MODEL
    s1_meta, s23_meta, model_path = _G_RESCORER
    if _MODEL is None:
        _MODEL = lgb.Booster(model_file=model_path)
    
    pair_feats = []
    pair_sids = []
    pair_cids = []

    for sid, cand_list in chunk_items:
        s1_m = s1_meta.get(sid)
        if not s1_m: continue
        
        for rank, cid in enumerate(cand_list, 1):
            c_m = s23_meta.get(cid)
            if not c_m: continue
            is_s3 = cid.startswith("S3-")
            feats = extract_features_single(s1_m, c_m, rank, is_s3)
            pair_feats.append(feats)
            pair_sids.append(sid)
            pair_cids.append(cid)

    claims = []
    if pair_feats:
        X_chunk = np.array(pair_feats, dtype=np.float32)
        # Vectorized chunk prediction on 1 thread: ~0.05s
        probs = _MODEL.predict(X_chunk, num_threads=1)
        mask = probs >= 0.40
        matched_indices = np.where(mask)[0]
        for idx in matched_indices:
            claims.append((pair_sids[idx], pair_cids[idx], float(probs[idx])))
            
    return claims

def main():
    print("=" * 80)
    print("  AMAZON ML CHALLENGE 2026: SOTA 48-CORE CANDIDATE RE-SCORING")
    print("=" * 80, flush=True)
    t_start = time.time()
    n_workers = min(48, os.cpu_count() or 4)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at {MODEL_PATH}!")
        sys.exit(1)

    tau = 0.67
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            meta = json.load(f)
            tau = meta.get("best_threshold", 0.67)
    print(f"Loaded optimal threshold tau = {tau:.2f} from {META_PATH}")

    # 1. Load test S1 with 48-core parallel parsing
    print("\n[1/5] Loading test_source1.tsv and parsing across 48 cores...", flush=True)
    t0 = time.time()
    s1_df = pd.read_csv(f"{DATA_DIR}/test_source1.tsv", sep="\t", dtype=str).fillna("")
    ordered_s1_ids = s1_df["entity_id"].tolist()
    total_s1 = len(ordered_s1_ids)

    s1_records = list(zip(s1_df["entity_id"], s1_df["business_name"], s1_df["business_address"], s1_df["country"]))
    del s1_df
    gc.collect()

    # Parallel parse
    csize = (len(s1_records) + n_workers - 1) // n_workers
    chunks = [s1_records[i*csize:(i+1)*csize] for i in range(n_workers) if i*csize < len(s1_records)]
    with mp.Pool(n_workers) as pool:
        meta_dicts = pool.map(_meta_worker, chunks)

    s1_meta = {}
    name_freq = {}
    for d in meta_dicts:
        s1_meta.update(d)
        for sid, meta in d.items():
            nm = meta[2]
            if nm: name_freq[nm] = name_freq.get(nm, 0) + 1
    del meta_dicts, s1_records
    gc.collect()
    print(f"Parsed {total_s1:,} S1 queries in {time.time()-t0:.1f}s")

    # 2. Collect candidate IDs for India and US
    print("\n[2/5] Reading candidate pools for India & US (top 35 per query)...", flush=True)
    t0 = time.time()
    country_queries = {"India": [], "US": []}
    needed_cids = set()

    for ctry, fname in [("India", "cand_India.tsv"), ("US", "cand_US.tsv")]:
        cp = os.path.join(PARTS_SOTA_DIR, fname)
        print(f"  Reading {fname}...", end=" ", flush=True)
        with open(cp, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    sid = parts[0]
                    cands = [c for c in parts[1].split(",") if c][:35]
                    country_queries[ctry].append((sid, cands))
                    for c in cands:
                        needed_cids.add(c)
        print(f"Done ({len(country_queries[ctry]):,} queries)")

    print(f"Loaded {len(country_queries['India']) + len(country_queries['US']):,} queries across {len(needed_cids):,} unique candidates in {time.time()-t0:.1f}s")

    # 3. Stream S2 and S3 raw records and parse across 48 cores
    print("\n[3/5] Streaming candidate metadata and parsing across 48 cores...", flush=True)
    t0 = time.time()
    cand_records = []
    for s_file in ["test_source2.tsv", "test_source3.tsv"]:
        path = f"{DATA_DIR}/{s_file}"
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 4 and p[0] in needed_cids:
                    cand_records.append((p[0], p[1], p[2], p[3]))
                    needed_cids.remove(p[0])
                    if not needed_cids:
                        break
    print(f"  Collected {len(cand_records):,} raw candidate records in {time.time()-t0:.1f}s")

    t1 = time.time()
    csize = (len(cand_records) + n_workers - 1) // n_workers
    chunks = [cand_records[i*csize:(i+1)*csize] for i in range(n_workers) if i*csize < len(cand_records)]
    del cand_records
    gc.collect()

    with mp.Pool(n_workers) as pool:
        cand_meta_dicts = pool.map(_meta_worker, chunks)

    s23_meta = {}
    for d in cand_meta_dicts:
        s23_meta.update(d)
    del cand_meta_dicts, chunks
    gc.collect()
    print(f"  Parsed {len(s23_meta):,} candidate records across 48 cores in {time.time()-t1:.1f}s")

    # 4. Multi-Core Scoring with Vectorized Chunk Predictions
    print(f"\n[4/5] Multi-core Candidate Re-scoring across {n_workers} cores...", flush=True)
    t0 = time.time()
    global _G_RESCORER
    _G_RESCORER = (s1_meta, s23_meta, MODEL_PATH)

    all_claims = []

    for ctry in ["India", "US"]:
        print(f"  Re-scoring {ctry} ({len(country_queries[ctry]):,} queries)...", end=" ", flush=True)
        t_c = time.time()
        c_items = country_queries[ctry]
        n_chunks = n_workers * 4
        csize = (len(c_items) + n_chunks - 1) // n_chunks
        chunks = [c_items[i*csize:(i+1)*csize] for i in range(n_chunks) if i*csize < len(c_items)]

        with mp.Pool(n_workers) as pool:
            results = pool.map(_rescore_chunk_worker, chunks)

        country_claims = []
        for r_chunk in results:
            country_claims.extend(r_chunk)

        print(f"Done in {time.time()-t_c:.1f}s ({len(country_claims):,} claims generated)")
        all_claims.extend(country_claims)

    print(f"Total candidate claims scored: {len(all_claims):,} in {time.time()-t0:.1f}s")

    # 5. Greedy Global M2O Bipartite Resolution
    print("\n[5/5] Global Greedy M2O & Precision Hardening...", flush=True)
    t0 = time.time()

    all_claims.sort(key=lambda x: x[2], reverse=True)

    assigned_cids = set()
    s1_rescore_matches = {}

    for sid, cid, prob in all_claims:
        if prob >= tau:
            if cid not in assigned_cids:
                assigned_cids.add(cid)
                if sid not in s1_rescore_matches:
                    s1_rescore_matches[sid] = []
                s1_rescore_matches[sid].append(cid)

    del all_claims, assigned_cids
    gc.collect()

    # Load France matches from parts_v2
    france_matches = {}
    france_file = os.path.join(PARTS_V2_DIR, "match_France.tsv")
    print(f"  Loading France v2 calibrated matches ({france_file})...", end=" ", flush=True)
    with open(france_file, "r", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1]:
                france_matches[p[0]] = p[1].split(",")
            elif len(p) >= 1:
                france_matches[p[0]] = []
    print(f"Done ({len(france_matches):,} queries)")

    # Combine and Apply Precision Guards
    final_matches = {}
    total_kept_pairs = 0
    final_singletons = 0

    for sid in ordered_s1_ids:
        s1_info = s1_meta.get(sid)
        if not s1_info:
            final_matches[sid] = []
            final_singletons += 1
            continue

        s1_name, s1_addr, s1_cn, s1_sp, s1_hn, s1_pc, s1_wn, s1_wa, s1_na, s1_bi, s1_tri, s1_atri, s1_p3, s1_p5, s1_c = s1_info

        if s1_c == "France":
            mids = france_matches.get(sid, [])
            final_matches[sid] = mids
            total_kept_pairs += len(mids)
            if not mids:
                final_singletons += 1
            continue

        mids = s1_rescore_matches.get(sid, [])
        if not mids:
            final_matches[sid] = []
            final_singletons += 1
            continue

        # Precision Guards
        kept = []
        for mid in mids:
            c_info = s23_meta.get(mid)
            if not c_info:
                kept.append(mid)
                continue

            c_name, c_addr, c_cn, c_sp, c_hn, c_pc, c_wn, c_wa, c_na, c_bi, c_tri, c_atri, c_p3, c_p5, c_c = c_info

            # HN guard: if both have explicit house numbers and they differ, prune!
            if s1_hn and c_hn and s1_hn != c_hn:
                if not (s1_hn in c_addr or c_hn in s1_addr):
                    continue

            # PC guard: if both have explicit postal codes and they differ, prune!
            if s1_pc and c_pc and s1_pc != c_pc:
                continue

            # Chain guard
            if name_freq.get(s1_cn, 0) >= 20:
                if s1_pc and c_pc and s1_pc != c_pc:
                    continue

            kept.append(mid)

        # Degree cap: max 5 S2, max 6 S3
        s2_m = [m for m in kept if m.startswith("S2-")][:5]
        s3_m = [m for m in kept if m.startswith("S3-")][:6]
        final_list = s2_m + s3_m

        final_matches[sid] = final_list
        total_kept_pairs += len(final_list)
        if not final_list:
            final_singletons += 1

    print(f"Assembly & Guards complete in {time.time()-t0:.1f}s")
    print(f"Total Matches: {total_kept_pairs:,} across {total_s1:,} queries")
    print(f"Singletons: {final_singletons:,} ({final_singletons/total_s1*100:.2f}%) [Target: ~5.58%]")
    print(f"Average Links / Entity: {total_kept_pairs/total_s1:.2f} [Target: ~3.3 - 3.4]")

    # Write matching_results.tsv
    print(f"\nWriting {MATCHING_OUT} in exact test order...", end=" ", flush=True)
    t0 = time.time()
    with open(MATCHING_OUT, "w", encoding="utf-8") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in ordered_s1_ids:
            fm.write(f"{sid}\t{','.join(final_matches.get(sid, []))}\n")
    print(f"Done ({os.path.getsize(MATCHING_OUT)/(1024*1024):.1f} MB in {time.time()-t0:.1f}s)")

    # Package ZIP
    t0 = time.time()
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(MATCHING_OUT, arcname="matching_results.tsv")
    print(f"Packaged {ZIP_OUT} ({os.path.getsize(ZIP_OUT)/(1024*1024):.1f} MB in {time.time()-t0:.1f}s)")

    # Validate
    print("\nRunning official submission validator...", flush=True)
    val_cmd = f"{sys.executable} src/validate_submission.py --matching {MATCHING_OUT} --candidate {CANDIDATE_OUT} --test-dir {DATA_DIR} --check-ids"
    os.system(val_cmd)

    print("\n" + "=" * 80)
    print(f"  ALL SOTA RE-SCORING & SUBMISSION READY IN {time.time()-t_start:.1f}s ({((time.time()-t_start)/60):.1f} min)")
    print("=" * 80)

if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    main()
