#!/usr/bin/env python3
"""
Quick targeted worker to complete only the 8 missing France shards.
Uses brute-force FAISS IndexFlatIP (exact, 0-training) across 32 threads for only 8,000 queries.
Completes in ~1 minute.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys, time, gc, pickle
from collections import Counter
import numpy as np
import pandas as pd
import lightgbm as lgb
import faiss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalization_v3 import parallel_process_corpus_v3
from blocking_v3 import MultiChannelBlockingEngineV3
from features_v3 import extract_pairwise_features_v3

DATA_DIR = "dataset/student_resource/dataset/test"
SHARDS_DIR = "output/parts_v4/shards"
MODEL_PATH = "models/lgbm_reranker_v4.txt"
CHUNK_SIZE = 1000

# Identify missing France chunks
existing = set()
for f in os.listdir(SHARDS_DIR):
    if f.startswith('cand_France_'):
        existing.add(int(f.split('_')[2].split('.')[0]))
missing_chunks = [i for i in range(260) if i not in existing]
print(f"France missing chunks ({len(missing_chunks)}): {missing_chunks}", flush=True)

if not missing_chunks:
    print("All France chunks already exist! Nothing to do.", flush=True)
    sys.exit(0)

# Load data
print("Loading France S1, S2, S3...", flush=True)
s1 = pd.read_csv(f"{DATA_DIR}/test_source1.tsv", sep="\t", dtype=str).fillna("")
s1_c = s1[s1['country'] == 'France'].reset_index(drop=True)
n_s1 = len(s1_c)
del s1
gc.collect()

s2 = pd.read_csv(f"{DATA_DIR}/test_source2.tsv", sep="\t", dtype=str).fillna("")
s3 = pd.read_csv(f"{DATA_DIR}/test_source3.tsv", sep="\t", dtype=str).fillna("")
s2_c = s2[s2['country'] == 'France'].copy()
s3_c = s3[s3['country'] == 'France'].copy()
s2_c['is_s3'] = 0.0
s3_c['is_s3'] = 1.0
s2s3 = pd.concat([s2_c, s3_c], ignore_index=True)
del s2, s3, s2_c, s3_c
gc.collect()
n_c = len(s2s3)

s1_names_raw = [str(n).strip().lower() for n in s1_c['business_name'].values]
c_names_raw = [str(n).strip().lower() for n in s2s3['business_name'].values]

print("Preprocessing with 16 workers...", flush=True)
(s1_names_norm, s1_names_clean, s1_names_sp, s1_name_words, s1_first_words,
 s1_addrs_norm, s1_hns, s1_pcs, s1_addr_tokens, s1_nums,
 s1_pks, s1_cities, s1_states, s1_lss, s1_cts) = parallel_process_corpus_v3(
    s1_c['business_name'].values, s1_c['business_address'].values, n_workers=16
)
s1_ids = s1_c['entity_id'].values
s1_first3 = [n[:3] for n in s1_names_clean]

(c_names_norm, c_names_clean, c_names_sp, c_name_words, c_first_words,
 c_addrs_norm, c_hns, c_pcs, c_addr_tokens, c_nums,
 c_pks, c_cities, c_states, c_lss, c_cts) = parallel_process_corpus_v3(
    s2s3['business_name'].values, s2s3['business_address'].values, n_workers=16
)
c_ids = s2s3['entity_id'].values
c_is_s3 = s2s3['is_s3'].values
c_first3 = [n[:3] for n in c_names_clean]

name_counts = Counter(c_names_clean)
chain_names = {k for k, v in name_counts.items() if v > 50 and k != ""}

print("Fitting text blocking engine...", flush=True)
engine = MultiChannelBlockingEngineV3(top_k_per_channel=50, max_union=40)
engine.fit_and_transform_corpus(
    s1_names_norm, s1_addrs_norm, s1_names_sp,
    c_names_norm, c_addrs_norm, c_names_sp, c_pcs, c_hns,
    c_phonetics=c_pks, c_first3=c_first3
)

print("Loading embeddings...", flush=True)
with open('embeddings/name_to_idx.pkl', 'rb') as f:
    name_to_idx = pickle.load(f)
embeddings = np.memmap('embeddings/all_names_emb.npy', dtype='float16', mode='r', shape=(len(name_to_idx), 1536))

# Extract candidate vectors into float32 array
print("Building exact FAISS IndexFlatIP (32 threads, 0-training)...", flush=True)
c_idx_valid = []
c_map = []
for i, n in enumerate(c_names_raw):
    idx = name_to_idx.get(n)
    if idx is not None:
        c_idx_valid.append(idx)
        c_map.append(i)

cand_embs = np.empty((len(c_idx_valid), 1536), dtype=np.float32)
for s in range(0, len(c_idx_valid), 250000):
    e = min(s + 250000, len(c_idx_valid))
    cand_embs[s:e] = embeddings[c_idx_valid[s:e]].astype(np.float32)

faiss.omp_set_num_threads(32)
flat_index = faiss.IndexFlatIP(1536)
flat_index.add(cand_embs)
del cand_embs
faiss.omp_set_num_threads(1)
print(f"Exact FAISS index ready with {flat_index.ntotal} vectors", flush=True)

# Load model
model = lgb.Booster(model_file=MODEL_PATH)

# Score each missing chunk
s1_p = (s1_names_norm, s1_names_clean, s1_names_sp, s1_name_words, s1_first_words,
        s1_addrs_norm, s1_hns, s1_pcs, s1_addr_tokens, s1_nums,
        s1_pks, s1_cities, s1_states, s1_lss, s1_cts)
c_p = (c_names_norm, c_names_clean, c_names_sp, c_name_words, c_first_words,
       c_addrs_norm, c_hns, c_pcs, c_addr_tokens, c_nums,
       c_pks, c_cities, c_states, c_lss, c_cts)

for c_idx in missing_chunks:
    b_start = c_idx * CHUNK_SIZE
    b_end = min(b_start + CHUNK_SIZE, n_s1)
    cand_sp = os.path.join(SHARDS_DIR, f"cand_France_{c_idx:04d}.tsv")
    claim_sp = os.path.join(SHARDS_DIR, f"claim_France_{c_idx:04d}.csv")
    
    print(f"Scoring chunk {c_idx} (queries {b_start}:{b_end})...", flush=True)
    
    # Text candidates
    cands_text = engine.retrieve_candidates_for_batch(
        b_start, b_end, s1_p[8], s1_p[2], s1_p[3], c_p[3],
        s1_phonetics=s1_pks, s1_first3=s1_first3
    )
    
    # Semantic candidates via exact FAISS
    q_names = s1_names_raw[b_start:b_end]
    q_embs = np.zeros((len(q_names), 1536), dtype=np.float32)
    for qi, qn in enumerate(q_names):
        qidx = name_to_idx.get(qn)
        if qidx is not None:
            q_embs[qi] = embeddings[qidx].astype(np.float32)
            
    faiss.omp_set_num_threads(32)
    dists, indices = flat_index.search(q_embs, 10)
    faiss.omp_set_num_threads(1)
    
    batch_feature_rows = []
    batch_meta = []
    batch_c_clean_names = []
    
    with open(cand_sp, "w", encoding="utf-8") as f_cand:
        for i in range(b_end - b_start):
            g_i = b_start + i
            sid = s1_ids[g_i]
            
            merged = []
            seen = set()
            for cidx, scs in cands_text[i]:
                merged.append((cidx, scs))
                seen.add(cidx)
                
            for j in range(10):
                fidx = indices[i, j]
                if fidx != -1:
                    orig_cidx = c_map[fidx]
                    if orig_cidx not in seen:
                        merged.append((orig_cidx, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
                        seen.add(orig_cidx)
                        
            merged = merged[:50]
            cand_id_list = [c_ids[cidx] for cidx, _ in merged]
            f_cand.write(f"{sid}\t{','.join(cand_id_list)}\n")
            
            s1_item = tuple(s1_p[k][g_i] for k in range(15))
            s1_name_lower = s1_names_raw[g_i]
            s1_idx = name_to_idx.get(s1_name_lower)
            emb_s1 = embeddings[s1_idx] if s1_idx is not None else None
            
            for rank, (cidx, scores) in enumerate(merged, 1):
                c_item = tuple(c_p[k][cidx] for k in range(15))
                feats = extract_pairwise_features_v3(s1_item, c_item, scores, rank, c_is_s3[cidx])
                
                cand_name_lower = c_names_raw[cidx]
                c_name_idx = name_to_idx.get(cand_name_lower)
                emb_cand = embeddings[c_name_idx] if c_name_idx is not None else None
                
                if emb_s1 is not None and emb_cand is not None:
                    cosine = float(np.dot(emb_s1.astype(np.float32), emb_cand.astype(np.float32)))
                else:
                    cosine = 0.0
                feats.append(cosine)
                
                batch_feature_rows.append(feats)
                batch_meta.append((sid, c_ids[cidx], c_is_s3[cidx]))
                batch_c_clean_names.append(c_p[1][cidx])
                
    if batch_feature_rows:
        X_batch = np.array(batch_feature_rows, dtype=np.float32)
        probs = model.predict(X_batch, num_threads=16)
        with open(claim_sp, "w", encoding="utf-8") as f_claims:
            for p_i in range(len(probs)):
                sid, cid, is_s3 = batch_meta[p_i]
                prob = probs[p_i]
                addr_tset = batch_feature_rows[p_i][14]
                if batch_c_clean_names[p_i] in chain_names and addr_tset < 0.25:
                    prob = 0.0
                f_claims.write(f"{sid},{cid},{is_s3},{prob:.6f}\n")
    print(f"  Chunk {c_idx} written ({len(batch_feature_rows):,} pairs scored)", flush=True)

print("=" * 80)
print("ALL MISSING FRANCE CHUNKS COMPLETED SUCCESSFULLY!")
print("=" * 80, flush=True)
