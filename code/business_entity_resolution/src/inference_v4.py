import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import sys
import time
import gc
import json
import zipfile
import shutil
import argparse
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalization_v3 import parallel_process_corpus_v3
from blocking_v3 import MultiChannelBlockingEngineV3
from features_v3 import FEATURE_NAMES, extract_pairwise_features_v3
from faiss_retrieval import SemanticBlockingEngine

DATA_DIR = "dataset/student_resource/dataset/test"
OUTPUT_DIR = "output"
PARTS_DIR = os.path.join(OUTPUT_DIR, "parts_v4")
MODEL_PATH = "models/lgbm_reranker_v4.txt"
META_PATH = "models/lgbm_reranker_v4_meta.json"

MATCHING_OUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
ZIP_OUT = os.path.join(OUTPUT_DIR, "matching_results.zip")

os.makedirs(PARTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SHARDS_DIR = os.path.join(PARTS_DIR, "shards")
os.makedirs(SHARDS_DIR, exist_ok=True)

_G_DATA = None

def _score_worker_chunk_v4(chunk_args):
    chunk_id, b_start, b_end, model_path, cand_shard_path, claims_shard_path = chunk_args
    eng = _G_DATA['engine']
    s1_ids = _G_DATA['s1_ids']
    c_ids = _G_DATA['c_ids']
    c_is_s3 = _G_DATA['c_is_s3']
    s1_pks = _G_DATA['s1_pks']
    s1_first3 = _G_DATA['s1_first3']
    s1_p = _G_DATA['s1_pack']
    c_p = _G_DATA['c_pack']
    s1_raw = _G_DATA['s1_names_raw']
    c_raw = _G_DATA['c_names_raw']
    chain_names = _G_DATA['chain_names']
    all_sem_cands = _G_DATA['all_sem_cands']
    
    embeddings = _G_DATA['embeddings']
    name_to_idx = _G_DATA['name_to_idx']

    model = lgb.Booster(model_file=model_path)
    
    # 1. Text candidates (max_union=40 configured in engine)
    cands_batch_text = eng.retrieve_candidates_for_batch(
        b_start, b_end, s1_p[8], s1_p[2], s1_p[3], c_p[3],
        s1_phonetics=s1_pks, s1_first3=s1_first3
    )
    
    # 2. Semantic candidates (pre-retrieved via vectorized FAISS)
    cands_batch_sem = all_sem_cands[b_start:b_end]

    batch_feature_rows = []
    batch_meta = []
    batch_c_clean_names = []
    
    with open(cand_shard_path, "w", encoding="utf-8") as f_cand:
        for i in range(b_end - b_start):
            g_i = b_start + i
            sid = s1_ids[g_i]
            
            text_cands = cands_batch_text[i]
            sem_cands = cands_batch_sem[i]
            
            merged_cands = []
            seen_cidx = set()
            
            # Text candidates first
            for cidx, scores in text_cands:
                merged_cands.append((cidx, scores))
                seen_cidx.add(cidx)
                
            # Semantic candidates appended
            for cidx, score in sem_cands:
                if cidx not in seen_cidx:
                    merged_cands.append((cidx, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
                    seen_cidx.add(cidx)
                    
            # Cap at max_union=50
            merged_cands = merged_cands[:50]
            
            cand_id_list = [c_ids[cidx] for cidx, _ in merged_cands]
            f_cand.write(f"{sid}\t{','.join(cand_id_list)}\n")

            s1_item = (
                s1_p[0][g_i], s1_p[1][g_i], s1_p[2][g_i], s1_p[3][g_i], s1_p[4][g_i],
                s1_p[5][g_i], s1_p[6][g_i], s1_p[7][g_i], s1_p[8][g_i], s1_p[9][g_i],
                s1_p[10][g_i], s1_p[11][g_i], s1_p[12][g_i], s1_p[13][g_i], s1_p[14][g_i]
            )
            
            s1_name_lower = s1_raw[g_i]
            s1_idx = name_to_idx.get(s1_name_lower)
            emb_s1 = embeddings[s1_idx] if s1_idx is not None else None
            
            for rank, (cidx, scores) in enumerate(merged_cands, 1):
                c_item = (
                    c_p[0][cidx], c_p[1][cidx], c_p[2][cidx], c_p[3][cidx], c_p[4][cidx],
                    c_p[5][cidx], c_p[6][cidx], c_p[7][cidx], c_p[8][cidx], c_p[9][cidx],
                    c_p[10][cidx], c_p[11][cidx], c_p[12][cidx], c_p[13][cidx], c_p[14][cidx]
                )
                feats = extract_pairwise_features_v3(s1_item, c_item, scores, rank, c_is_s3[cidx])
                
                cand_name_lower = c_raw[cidx]
                c_name_idx = name_to_idx.get(cand_name_lower)
                emb_cand = embeddings[c_name_idx] if c_name_idx is not None else None
                
                if emb_s1 is not None and emb_cand is not None:
                    semantic_cosine = float(np.dot(emb_s1.astype(np.float32), emb_cand.astype(np.float32)))
                else:
                    semantic_cosine = 0.0
                    
                feats.append(semantic_cosine)
                
                batch_feature_rows.append(feats)
                batch_meta.append((sid, c_ids[cidx], c_is_s3[cidx]))
                batch_c_clean_names.append(c_p[1][cidx])

    n_claims = 0
    if batch_feature_rows:
        X_batch = np.array(batch_feature_rows, dtype=np.float32)
        probs = model.predict(X_batch, num_threads=1)
        
        with open(claims_shard_path, "w", encoding="utf-8") as f_claims:
            for p_i in range(len(probs)):
                sid, cid, is_s3 = batch_meta[p_i]
                prob = probs[p_i]
                
                # Chain-name guard: for business names appearing >50 times in S2+S3, require addr_token_set_ratio >= 0.25
                addr_tset = batch_feature_rows[p_i][14]
                if batch_c_clean_names[p_i] in chain_names and addr_tset < 0.25:
                    prob = 0.0
                
                f_claims.write(f"{sid},{cid},{is_s3},{prob:.6f}\n")
                n_claims += 1

    return chunk_id, b_end - b_start, len(batch_feature_rows), n_claims


def process_country_v4(country, n_workers, embeddings, name_to_idx):
    print("\n" + "=" * 80)
    print(f"  PROCESSING COUNTRY: {country.upper()} (v4 Pipeline)")
    print("=" * 80, flush=True)
    
    t_start = time.time()
    
    print(f"[{country}] Loading S1 test records...", end=" ", flush=True)
    s1 = pd.read_csv(f"{DATA_DIR}/test_source1.tsv", sep="\t", dtype=str).fillna("")
    s1_c = s1[s1['country'] == country].reset_index(drop=True)
    n_s1 = len(s1_c)
    del s1
    gc.collect()
    print(f"{n_s1:,} queries.")

    CHUNK_SIZE = 1000
    n_chunks = int(np.ceil(n_s1 / CHUNK_SIZE))
    done_count = sum(
        1 for c_idx in range(n_chunks)
        if os.path.exists(os.path.join(SHARDS_DIR, f"cand_{country}_{c_idx:04d}.tsv")) and
           os.path.exists(os.path.join(SHARDS_DIR, f"claim_{country}_{c_idx:04d}.csv"))
    )
    if done_count == n_chunks:
        print(f"[{country}] All {n_chunks} shards already complete on disk! Skipping country.", flush=True)
        return

    print(f"[{country}] Loading S2 and S3 candidate corpus...", end=" ", flush=True)
    s2 = pd.read_csv(f"{DATA_DIR}/test_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(f"{DATA_DIR}/test_source3.tsv", sep="\t", dtype=str).fillna("")
    
    s2_c = s2[s2['country'] == country].copy()
    s3_c = s3[s3['country'] == country].copy()
    s2_c['is_s3'] = 0.0
    s3_c['is_s3'] = 1.0
    
    s2s3 = pd.concat([s2_c, s3_c], ignore_index=True)
    del s2, s3, s2_c, s3_c
    gc.collect()
    n_c = len(s2s3)
    print(f"{n_c:,} candidates.")

    # Exact lowercase stripped strings for 100% embedding lookup rate
    s1_names_raw = [str(n).strip().lower() for n in s1_c['business_name'].values]
    c_names_raw = [str(n).strip().lower() for n in s2s3['business_name'].values]

    print(f"[{country}] Parallel preprocessing S1 records ({n_workers} workers)...", flush=True)
    t0 = time.time()
    (s1_names_norm, s1_names_clean, s1_names_sp, s1_name_words, s1_first_words,
     s1_addrs_norm, s1_hns, s1_pcs, s1_addr_tokens, s1_nums,
     s1_pks, s1_cities, s1_states, s1_lss, s1_cts) = parallel_process_corpus_v3(
        s1_c['business_name'].values, s1_c['business_address'].values, n_workers=n_workers
    )
    s1_ids = s1_c['entity_id'].values
    s1_first3 = [n[:3] for n in s1_names_clean]
    print(f"[{country}] S1 preprocessed in {time.time()-t0:.1f}s", flush=True)

    print(f"[{country}] Parallel preprocessing S2+S3 records ({n_workers} workers)...", flush=True)
    t0 = time.time()
    (c_names_norm, c_names_clean, c_names_sp, c_name_words, c_first_words,
     c_addrs_norm, c_hns, c_pcs, c_addr_tokens, c_nums,
     c_pks, c_cities, c_states, c_lss, c_cts) = parallel_process_corpus_v3(
        s2s3['business_name'].values, s2s3['business_address'].values, n_workers=n_workers
    )
    c_ids = s2s3['entity_id'].values
    c_is_s3 = s2s3['is_s3'].values
    c_first3 = [n[:3] for n in c_names_clean]
    print(f"[{country}] S2+S3 preprocessed in {time.time()-t0:.1f}s", flush=True)

    # Calculate high frequency chain names
    name_counts = Counter(c_names_clean)
    chain_names = {k for k, v in name_counts.items() if v > 50 and k != ""}
    print(f"[{country}] Identified {len(chain_names)} chain names appearing >50 times.", flush=True)

    # 3. Text Blocking Engine
    engine = MultiChannelBlockingEngineV3(top_k_per_channel=50, max_union=40)
    engine.fit_and_transform_corpus(
        s1_names_norm, s1_addrs_norm, s1_names_sp,
        c_names_norm, c_addrs_norm, c_names_sp, c_pcs, c_hns,
        c_phonetics=c_pks, c_first3=c_first3
    )

    # 4. Semantic Blocking Engine (Vectorized FAISS pre-retrieval)
    sem_engine = SemanticBlockingEngine(
        emb_path='embeddings/all_names_emb.npy',
        idx_path='embeddings/name_to_idx.pkl',
        top_k=10
    )
    sem_engine.embeddings = embeddings
    sem_engine.name_to_idx = name_to_idx
    c_idx_arr = np.arange(len(c_ids))
    sem_engine.build_index(c_names_raw, c_idx_arr)
    
    print(f"[{country}] Pre-retrieving semantic top-10 candidates across all {n_s1:,} queries...", flush=True)
    t_sem = time.time()
    all_sem_cands = sem_engine.retrieve_semantic_candidates(s1_names_raw, top_k=10)
    print(f"[{country}] Semantic candidates retrieved in {time.time()-t_sem:.1f}s", flush=True)

    # 5. Multi-Worker Batched Scoring
    CHUNK_SIZE = 1000
    n_chunks = int(np.ceil(n_s1 / CHUNK_SIZE))
    print(f"[{country}] Parallel scoring {n_s1:,} queries across {n_chunks} chunks using {n_workers} workers...", flush=True)

    global _G_DATA
    _G_DATA = {
        'engine': engine,
        'all_sem_cands': all_sem_cands,
        's1_ids': s1_ids,
        'c_ids': c_ids,
        'c_is_s3': c_is_s3,
        's1_pks': s1_pks,
        's1_first3': s1_first3,
        's1_names_raw': s1_names_raw,
        'c_names_raw': c_names_raw,
        's1_pack': (s1_names_norm, s1_names_clean, s1_names_sp, s1_name_words, s1_first_words,
                    s1_addrs_norm, s1_addr_tokens, s1_hns, s1_pcs, s1_nums,
                    s1_pks, s1_cities, s1_states, s1_lss, s1_cts),
        'c_pack': (c_names_norm, c_names_clean, c_names_sp, c_name_words, c_first_words,
                   c_addrs_norm, c_addr_tokens, c_hns, c_pcs, c_nums,
                   c_pks, c_cities, c_states, c_lss, c_cts),
        'chain_names': chain_names,
        'embeddings': embeddings,
        'name_to_idx': name_to_idx
    }

    chunks = []
    already_done = 0
    for c_idx in range(n_chunks):
        b_start = c_idx * CHUNK_SIZE
        b_end = min(b_start + CHUNK_SIZE, n_s1)
        cand_sp = os.path.join(SHARDS_DIR, f"cand_{country}_{c_idx:04d}.tsv")
        claim_sp = os.path.join(SHARDS_DIR, f"claim_{country}_{c_idx:04d}.csv")

        if os.path.exists(cand_sp) and os.path.exists(claim_sp):
            already_done += 1
            continue

        chunks.append((c_idx, b_start, b_end, MODEL_PATH, cand_sp, claim_sp))

    print(f"[{country}] Shard status: {already_done}/{n_chunks} chunks already done on disk. To score: {len(chunks)} chunks.")

    if chunks:
        t_inf_start = time.time()
        gc.collect()
        gc.freeze()
        ctx = mp.get_context('fork')
        with ctx.Pool(n_workers) as pool:
            completed = 0
            n_todo = len(chunks)
            total_pairs_scored = 0
            for chunk_id, n_q, n_pairs, n_claims in pool.imap_unordered(_score_worker_chunk_v4, chunks):
                completed += 1
                total_pairs_scored += n_pairs
                if completed % 10 == 0 or completed == n_todo:
                    pct = 100.0 * completed / n_todo
                    elapsed = time.time() - t_inf_start
                    eta = (elapsed / completed) * (n_todo - completed) if completed < n_todo else 0
                    print(f"  [{country}] Chunk {completed}/{n_todo} ({pct:.0f}%) | Pairs: {total_pairs_scored:,} | {elapsed:.0f}s | ETA: {eta:.0f}s", flush=True)

    _G_DATA = None
    del engine, sem_engine, all_sem_cands, s1_c, s2s3
    gc.collect()
    
    elapsed_total = time.time() - t_start
    print(f"[{country}] COMPLETE in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")


def global_m2o(thresholds_by_country, tau_singleton):
    print("\n" + "=" * 80)
    print("  RUNNING GLOBAL M2O RESOLUTION (v4)")
    print("=" * 80, flush=True)
    
    print("Loading test_source1 to get original S1 IDs and ordering...")
    s1_all = pd.read_csv(f"{DATA_DIR}/test_source1.tsv", sep="\t", dtype=str)
    
    s1_id_to_idx = {sid: i for i, sid in enumerate(s1_all['entity_id'])}
    s1_ids = s1_all['entity_id'].values
    n_s1 = len(s1_ids)
    
    del s1_all
    gc.collect()
    
    print("Loading all claims from shards...")
    claims_s1_id = []
    claims_cid = []
    claims_is_s3 = []
    claims_prob = []
    
    s1_max_prob = np.zeros(n_s1, dtype=np.float32)
    
    for country in ["France", "US", "India"]:
        shard_files = [f for f in os.listdir(SHARDS_DIR) if f.startswith(f"claim_{country}_")]
        for f in shard_files:
            if not f.endswith(".csv"): continue
            claim_sp = os.path.join(SHARDS_DIR, f)
            with open(claim_sp, 'r', encoding='utf-8') as f_claim:
                for line in f_claim:
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        sid = parts[0]
                        cid = parts[1]
                        is_s3 = float(parts[2])
                        prob = float(parts[3])
                        
                        g_i = s1_id_to_idx[sid]
                        s1_max_prob[g_i] = max(s1_max_prob[g_i], prob)
                        
                        threshold = thresholds_by_country.get(country, 0.55)
                        if prob >= threshold:
                            claims_s1_id.append(sid)
                            claims_cid.append(cid)
                            claims_is_s3.append(is_s3)
                            claims_prob.append(prob)
                                
    print(f"Loaded {len(claims_prob):,} claims that passed country thresholds.")
    
    claims_s1_idx = np.array([s1_id_to_idx[sid] for sid in claims_s1_id])
    claims_is_s3 = np.array(claims_is_s3)
    claims_prob = np.array(claims_prob, dtype=np.float32)
    claims_cid = np.array(claims_cid)
    
    sort_order = np.argsort(-claims_prob)
    
    assigned_cids = set()
    s1_matches = {i: [] for i in range(n_s1)}
    
    s1_s2_counts = np.zeros(n_s1, dtype=np.int32)
    s1_s3_counts = np.zeros(n_s1, dtype=np.int32)
    
    print(f"Greedy assignment (singleton guard tau={tau_singleton:.3f})...")
    for idx in sort_order:
        g_i = claims_s1_idx[idx]
        
        # Singleton guard check
        if s1_max_prob[g_i] < tau_singleton:
            continue
            
        cid = claims_cid[idx]
        if cid not in assigned_cids:
            is_s3 = claims_is_s3[idx]
            
            if is_s3 == 0.0 and s1_s2_counts[g_i] < 5:
                assigned_cids.add(cid)
                s1_matches[g_i].append(cid)
                s1_s2_counts[g_i] += 1
            elif is_s3 == 1.0 and s1_s3_counts[g_i] < 6:
                assigned_cids.add(cid)
                s1_matches[g_i].append(cid)
                s1_s3_counts[g_i] += 1

    return s1_ids, s1_matches


def assemble_and_validate(s1_ids, s1_matches):
    print("\n" + "=" * 80)
    print("  ASSEMBLING FINAL SUBMISSION")
    print("=" * 80, flush=True)
    
    n_s1 = len(s1_ids)
    
    with open(MATCHING_OUT, "w", encoding="utf-8") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        n_matched = 0
        for i in range(n_s1):
            sid = s1_ids[i]
            mids = s1_matches[i]
            if mids:
                n_matched += 1
                fm.write(f"{sid}\t{','.join(mids)}\n")
            else:
                fm.write(f"{sid}\t\n")
                
    print("Loading candidate shards into memory for strict row-order assembly...")
    cand_dict = {}
    for country in ["France", "US", "India"]:
        shard_files = [f for f in os.listdir(SHARDS_DIR) if f.startswith(f"cand_{country}_")]
        for f in shard_files:
            cand_sp = os.path.join(SHARDS_DIR, f)
            with open(cand_sp, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    parts = line.strip().split("\t", 1)
                    if len(parts) == 2:
                        cand_dict[parts[0]] = parts[1]
                    elif len(parts) == 1:
                        cand_dict[parts[0]] = ""

    with open(CANDIDATE_OUT, "w", encoding="utf-8") as fc:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_ids:
            cands_str = cand_dict.get(sid, "")
            fc.write(f"{sid}\t{cands_str}\n")
    del cand_dict
    gc.collect()
                        
    out_df = pd.read_csv(MATCHING_OUT, sep="\t")
    assert len(out_df) == n_s1, f"Row count mismatch! Expected {n_s1}, got {len(out_df)}"
    assert out_df['source1_entity_id'].is_unique, "Duplicate S1 IDs in output!"
    
    shutil.copyfile(MATCHING_OUT, "matching_results.tsv")
    
    with zipfile.ZipFile(ZIP_OUT, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(MATCHING_OUT, arcname="matching_results.tsv")
        
    print(f"Assembly complete. Output created at {MATCHING_OUT} and {ZIP_OUT}")
    print(f"Matched: {n_matched:,}/{n_s1:,} ({100*n_matched/n_s1:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=12)
    args = parser.parse_args()
    
    if not os.path.isfile(MODEL_PATH):
        print(f"Warning: Model not found at {MODEL_PATH}! Ensure model is present before running.")
        
    tau = 0.55
    tau_singleton = 0.90
    thresholds = {'US': tau, 'India': tau, 'France': tau}
    if os.path.isfile(META_PATH):
        with open(META_PATH, "r") as f:
            meta = json.load(f)
            tau = meta.get("best_threshold_global", meta.get("best_threshold", 0.55))
            thresholds = meta.get("country_taus", meta.get("country_thresholds", {'US': tau, 'India': tau, 'France': tau}))
            tau_singleton = meta.get("tau_singleton", 0.90)
            print(f"Loaded optimal thresholds: {thresholds}")
            print(f"Loaded tau_singleton: {tau_singleton}")
            
    print(f"Using {args.n_workers} CPU worker threads.")

    print("Loading global embeddings into memory...")
    idx_path = 'embeddings/name_to_idx.pkl'
    emb_path = 'embeddings/all_names_emb.npy'
    with open(idx_path, 'rb') as f:
        name_to_idx = pickle.load(f)
    n_names = len(name_to_idx)
    embeddings = np.memmap(emb_path, dtype='float16', mode='r', shape=(n_names, 1536))
    print(f"Loaded embeddings for {n_names} unique names.")

    for country in ["France", "US", "India"]:
        process_country_v4(country, args.n_workers, embeddings, name_to_idx)
        
    s1_ids, s1_matches = global_m2o(thresholds, tau_singleton)
    assemble_and_validate(s1_ids, s1_matches)
    print("\nALL TASKS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
