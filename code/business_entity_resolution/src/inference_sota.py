#!/usr/bin/env python3
"""
Production SOTA Inference Engine
Amazon ML Challenge 2026

Runs full-scale 5-channel blocking + 32-feature LightGBM reranking + Global M2O
across all 1,732,544 test entities.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import time
import gc
import json
import zipfile
import shutil
import multiprocessing as mp
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalization import (
    universal_normalize, normalize_name_clean,
    extract_house_number, extract_postal_code, extract_address_tokens,
    parallel_process_corpus
)
from blocking import MultiChannelBlockingEngine
from features import extract_pairwise_features

DATA_DIR = "dataset/student_resource/dataset/test"
OUTPUT_DIR = "output"
PARTS_DIR = os.path.join(OUTPUT_DIR, "parts_sota_098")
MODEL_PATH = "models/lgbm_reranker_098.txt"
META_PATH = "models/lgbm_reranker_098_meta.json"

MATCHING_OUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
ZIP_OUT = os.path.join(OUTPUT_DIR, "matching_results.zip")

os.makedirs(PARTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Shard directories
SHARDS_DIR = os.path.join(PARTS_DIR, "shards")
os.makedirs(SHARDS_DIR, exist_ok=True)

_G_DATA = None

def _score_worker_chunk(chunk_args):
    chunk_id, b_start, b_end, threshold, model_path, cand_shard_path, claims_shard_path = chunk_args
    eng = _G_DATA['engine']
    s1_ids = _G_DATA['s1_ids']
    c_ids = _G_DATA['c_ids']
    c_is_s3 = _G_DATA['c_is_s3']
    s1_p = _G_DATA['s1_pack']
    c_p = _G_DATA['c_pack']
    
    model = lgb.Booster(model_file=model_path)
    cands_batch = eng.retrieve_candidates_for_batch(b_start, b_end, s1_p[7], s1_p[2], c_p[2])
    
    batch_feature_rows = []
    batch_meta = []
    n_cands_total = 0
    
    with open(cand_shard_path, "w", encoding="utf-8") as f_cand:
        for i, cands in enumerate(cands_batch):
            g_i = b_start + i
            sid = s1_ids[g_i]
            n_cands_total += len(cands)
            cand_id_list = [c_ids[cidx] for cidx, _ in cands]
            f_cand.write(f"{sid}\t{','.join(cand_id_list)}\n")
            
            s1_item = (s1_p[0][g_i], s1_p[1][g_i], s1_p[2][g_i], s1_p[3][g_i], s1_p[4][g_i], s1_p[5][g_i], s1_p[6][g_i], s1_p[7][g_i], s1_p[8][g_i])
            for rank, (cidx, scores) in enumerate(cands, 1):
                c_item = (c_p[0][cidx], c_p[1][cidx], c_p[2][cidx], c_p[3][cidx], c_p[4][cidx], c_p[5][cidx], c_p[6][cidx], c_p[7][cidx], c_p[8][cidx])
                feats = extract_pairwise_features(s1_item, c_item, scores, rank, c_is_s3[cidx])
                batch_feature_rows.append(feats)
                batch_meta.append((g_i, c_ids[cidx]))
                
    n_claims = 0
    if batch_feature_rows:
        X_batch = np.array(batch_feature_rows, dtype=np.float32)
        probs = model.predict(X_batch, num_threads=1)
        pass_idx = np.where(probs >= threshold)[0]
        n_claims = len(pass_idx)
        with open(claims_shard_path, "w", encoding="utf-8") as f_claims:
            for p_i in pass_idx:
                g_i, cid = batch_meta[p_i]
                f_claims.write(f"{g_i},{cid},{probs[p_i]:.4f}\n")
                
    return chunk_id, b_end - b_start, len(batch_feature_rows), n_claims

def process_country(country, threshold):
    print("\n" + "=" * 80)
    print(f"  PROCESSING COUNTRY: {country.upper()} (5-Channel + 32-Feat LightGBM + M2O)")
    print("=" * 80, flush=True)
    t_start = time.time()
    
    part_match = os.path.join(PARTS_DIR, f"match_{country}.tsv")
    part_cand = os.path.join(PARTS_DIR, f"cand_{country}.tsv")
    
    if os.path.exists(part_match) and os.path.exists(part_cand):
        sz_m = os.path.getsize(part_match)
        sz_c = os.path.getsize(part_cand)
        if sz_m > 1000 and sz_c > 1000:
            print(f"[{country}] Already completed (match: {sz_m:,} bytes, cand: {sz_c:,} bytes). SKIPPING!", flush=True)
            return
            
    # 1. Load test data
    print(f"[{country}] Loading S1 test records...", end=" ", flush=True)
    s1 = pd.read_csv(f"{DATA_DIR}/test_source1.tsv", sep="\t", dtype=str).fillna("")
    s1_c = s1[s1['country'] == country].reset_index(drop=True)
    n_s1 = len(s1_c)
    del s1
    gc.collect()
    print(f"{n_s1:,} queries.", flush=True)
    
    print(f"[{country}] Loading S2 and S3 candidate corpus...", end=" ", flush=True)
    s2 = pd.read_csv(f"{DATA_DIR}/test_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(f"{DATA_DIR}/test_source3.tsv", sep="\t", dtype=str).fillna("")
    s2_c = s2[s2['country'] == country].copy()
    s3_c = s3[s3['country'] == country].copy()
    s2_c['is_s3'] = 0.0
    s3_c['is_s3'] = 1.0
    s2s3 = pd.concat([s2_c, s3_c], ignore_index=True)
    n_c = len(s2s3)
    del s2, s3, s2_c, s3_c
    gc.collect()
    print(f"{n_c:,} candidates.", flush=True)
    
    # 2. Universal Preprocessing
    n_cpu = min(32, os.cpu_count() or 4)
    print(f"[{country}] Parallel preprocessing S1 records ({n_cpu} workers)...", flush=True)
    t0 = time.time()
    s1_names_raw = s1_c['business_name'].values
    s1_addrs_raw = s1_c['business_address'].values
    s1_ids = s1_c['entity_id'].values
    
    (s1_names_norm, s1_names_clean, s1_name_words, s1_first_words,
     s1_addrs_norm, s1_hns, s1_pcs, s1_addr_tokens, s1_nums) = parallel_process_corpus(
        s1_names_raw, s1_addrs_raw, n_workers=n_cpu
    )
    print(f"[{country}] S1 preprocessed in {time.time()-t0:.1f}s", flush=True)
    
    print(f"[{country}] Parallel preprocessing S2+S3 records ({n_cpu} workers)...", flush=True)
    t0 = time.time()
    c_names_raw = s2s3['business_name'].values
    c_addrs_raw = s2s3['business_address'].values
    c_ids = s2s3['entity_id'].values
    c_is_s3 = s2s3['is_s3'].values
    
    (c_names_norm, c_names_clean, c_name_words, c_first_words,
     c_addrs_norm, c_hns, c_pcs, c_addr_tokens, c_nums) = parallel_process_corpus(
        c_names_raw, c_addrs_raw, n_workers=n_cpu
    )
    print(f"[{country}] S2+S3 preprocessed in {time.time()-t0:.1f}s", flush=True)
    
    # 3. 5-Channel Blocking
    engine = MultiChannelBlockingEngine(top_k_per_channel=40, max_union=80)
    engine.fit_and_transform_corpus(
        s1_names_norm, s1_addrs_norm,
        c_names_norm, c_addrs_norm, c_pcs, c_hns
    )
    
    # 4. Multi-Worker Batched Scoring
    CHUNK_SIZE = 1000
    n_chunks = int(np.ceil(n_s1 / CHUNK_SIZE))
    n_workers = min(12, os.cpu_count() or 4)
    print(f"[{country}] Parallel scoring {n_s1:,} queries across {n_chunks} chunks using {n_workers} workers...", flush=True)
    
    global _G_DATA
    _G_DATA = {
        'engine': engine,
        's1_ids': s1_ids,
        'c_ids': c_ids,
        'c_is_s3': c_is_s3,
        's1_pack': (s1_names_raw, s1_names_clean, s1_name_words, s1_first_words, s1_addrs_raw, s1_addr_tokens, s1_hns, s1_pcs, s1_nums),
        'c_pack': (c_names_raw, c_names_clean, c_name_words, c_first_words, c_addrs_raw, c_addr_tokens, c_hns, c_pcs, c_nums)
    }
    
    chunks = []
    shard_cands = []
    shard_claims = []
    already_done = 0
    for c_idx in range(n_chunks):
        b_start = c_idx * CHUNK_SIZE
        b_end = min(b_start + CHUNK_SIZE, n_s1)
        cand_sp = os.path.join(SHARDS_DIR, f"cand_{country}_{c_idx:04d}.tsv")
        claim_sp = os.path.join(SHARDS_DIR, f"claim_{country}_{c_idx:04d}.csv")
        shard_cands.append(cand_sp)
        shard_claims.append(claim_sp)
        if os.path.exists(cand_sp) and os.path.exists(claim_sp) and os.path.getsize(cand_sp) > 0 and os.path.getsize(claim_sp) > 0:
            already_done += 1
            continue
        chunks.append((c_idx, b_start, b_end, threshold, MODEL_PATH, cand_sp, claim_sp))
        
    n_todo = len(chunks)
    print(f"[{country}] Shard status: {already_done}/{n_chunks} chunks already done on disk. To score: {n_todo} chunks.", flush=True)
    
    t_inf_start = time.time()
    if n_todo > 0:
        with mp.get_context('fork').Pool(n_workers) as pool:
            completed = 0
            total_pairs_scored = 0
            total_claims_found = 0
            for chunk_id, q_cnt, p_cnt, c_cnt in pool.imap_unordered(_score_worker_chunk, chunks):
                completed += 1
                total_pairs_scored += p_cnt
                total_claims_found += c_cnt
                if completed % 10 == 0 or completed == n_todo:
                    pct = 100.0 * completed / n_todo
                    elapsed = time.time() - t_inf_start
                    eta = (elapsed / completed) * (n_todo - completed) if completed < n_todo else 0
                    print(f"  [{country}] Chunk {completed}/{n_todo} ({pct:.0f}%) | Claims >= {threshold:.2f}: {total_claims_found:,} | Pairs: {total_pairs_scored:,} | {elapsed:.0f}s | ETA: {eta:.0f}s", flush=True)
                    
    # Concatenate candidate shards in exact chunk order
    import shutil
    print(f"[{country}] Concatenating {n_chunks} candidate shards to {part_cand}...", flush=True)
    with open(part_cand, "wb") as f_out:
        for sp in shard_cands:
            if os.path.exists(sp):
                with open(sp, "rb") as f_in:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(sp)
                
    # Read claims from shards
    claims_s1_idx = []
    claims_cid = []
    claims_prob = []
    for sp in shard_claims:
        if os.path.exists(sp):
            with open(sp, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    parts = line.strip().split(",")
                    if len(parts) == 3:
                        claims_s1_idx.append(int(parts[0]))
                        claims_cid.append(parts[1])
                        claims_prob.append(float(parts[2]))
            os.remove(sp)
            
    _G_DATA = None
    del engine, s1_c, s2s3
    gc.collect()
    
    # 5. Greedy Global Many-to-One (M2O) Bipartite Resolution
    print(f"[{country}] Running Global M2O Bipartite Resolution on {len(claims_prob):,} claims...", flush=True)
    t_m2o = time.time()
    
    sort_order = np.argsort(-np.array(claims_prob, dtype=np.float32))
    assigned_cids = set()
    s1_matches = {i: [] for i in range(n_s1)}
    
    for idx in sort_order:
        cid = claims_cid[idx]
        if cid not in assigned_cids:
            assigned_cids.add(cid)
            g_i = claims_s1_idx[idx]
            s1_matches[g_i].append(cid)
            
    print(f"  M2O complete in {time.time()-t_m2o:.1f}s", flush=True)
    
    # 6. Write match partition
    n_matched = 0
    total_preds = 0
    with open(part_match, "w", encoding="utf-8") as f_match:
        for i in range(n_s1):
            sid = s1_ids[i]
            mids = s1_matches[i]
            if mids:
                n_matched += 1
                total_preds += len(mids)
                f_match.write(f"{sid}\t{','.join(mids)}\n")
            else:
                f_match.write(f"{sid}\t\n")
                
    elapsed_total = time.time() - t_start
    print(f"[{country}] COMPLETE in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"[{country}] Matched: {n_matched:,}/{n_s1:,} ({100*n_matched/n_s1:.1f}%) | Singletons: {n_s1 - n_matched:,} ({100*(n_s1-n_matched)/n_s1:.1f}%) | Preds: {total_preds:,} ({total_preds/n_s1:.2f} per entity)")
    return n_s1, n_matched, total_preds

def assemble_and_validate():
    print("\n" + "=" * 80)
    print("  ASSEMBLING FINAL SUBMISSION IN EXACT TEST_SOURCE1 ORDER")
    print("=" * 80, flush=True)
    
    s1_all = pd.read_csv(f"{DATA_DIR}/test_source1.tsv", sep="\t", usecols=['entity_id'])
    ordered_ids = s1_all['entity_id'].tolist()
    total_expected = len(ordered_ids)
    del s1_all
    
    match_map = {}
    cand_map = {}
    for country in ["France", "India", "US"]:
        mp = os.path.join(PARTS_DIR, f"match_{country}.tsv")
        cp = os.path.join(PARTS_DIR, f"cand_{country}.tsv")
        with open(mp, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                match_map[parts[0]] = parts[1] if len(parts) >= 2 else ""
        with open(cp, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                cand_map[parts[0]] = parts[1] if len(parts) >= 2 else ""
                
    # Write matching_results.tsv
    with open(MATCHING_OUT, "w", encoding="utf-8") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in ordered_ids:
            fm.write(f"{sid}\t{match_map.get(sid, '')}\n")
            
    # Write candidate_pairs.tsv
    with open(CANDIDATE_OUT, "w", encoding="utf-8") as fc:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in ordered_ids:
            fc.write(f"{sid}\t{cand_map.get(sid, '')}\n")
            
    # Copy matching_results.tsv to root
    import shutil
    shutil.copyfile(MATCHING_OUT, "matching_results.tsv")
    
    # Create zip
    with zipfile.ZipFile(ZIP_OUT, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(MATCHING_OUT, arcname="matching_results.tsv")
        
    print(f"Assembly complete. Output created at {MATCHING_OUT} and {ZIP_OUT} ({os.path.getsize(ZIP_OUT)/(1024*1024):.1f} MB)", flush=True)

def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}! Run train_reranker.py first.")
        sys.exit(1)
        
    print(f"Verified trained LightGBM model exists at {MODEL_PATH}")
    
    tau = 0.55
    if os.path.isfile(META_PATH):
        with open(META_PATH, "r") as f:
            meta = json.load(f)
            tau = meta.get("best_threshold", 0.55)
            print(f"Loaded optimal threshold tau = {tau:.2f} from {META_PATH}")
            
    # Calibrated thresholds per country
    thresholds = {
        'US': tau,
        'India': tau,
        'France': tau + 0.05  # Extra conservative for unseen French country
    }
    
    for country in ["France", "India", "US"]:
        process_country(country, thresholds[country])
        
    assemble_and_validate()
    print("\nALL TASKS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
