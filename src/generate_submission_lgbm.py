#!/usr/bin/env python3
"""
Amazon ML Challenge 2026: Production Inference Pipeline with LightGBM Reranker
=============================================================================
Replaces heuristic threshold with trained LightGBM classifier.

Flow:
  1. Load LightGBM model
  2. For each country: blocking → pairwise features → LightGBM predict → M2O
  3. Assemble final submission in exact test_source1.tsv row order

Usage:
  python src/generate_submission_lgbm.py
"""

import os
import sys
import time
import gc
import re
import json
import unicodedata
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse
import rapidfuzz

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("ERROR: lightgbm not installed!")
    sys.exit(1)

# ── Configuration ───────────────────────────────────────────────
DATA_DIR = "dataset/student_resource/dataset/test"
OUTPUT_DIR = "output"
PARTS_DIR = "output/parts_lgbm"
MODEL_PATH = "models/lgbm_reranker_final.txt"
META_PATH = "models/lgbm_reranker_meta.json"
MATCHING_OUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

os.makedirs(PARTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_K_CHANNEL_A = 30
TOP_K_CHANNEL_B = 30
MAX_UNION_CANDIDATES = 50
BATCH_SIZE = 15000

# ── Import feature functions from reranker module ──────────────
# Copy the same functions to ensure consistency
LEGAL_SUFFIXES = re.compile(
    r'\b(inc|incorporated|corp|corporation|co|company|ltd|limited|llc|llp|'
    r'pvt|private|plc|sa|sas|sarl|srl|gmbh|ag|pty|nv|bv|oy|ab|as|aps|'
    r'holding|holdings|group|enterprises?|associates?|services?|solutions?|'
    r'technologies|tech|systems?|international|intl|global|worldwide|'
    r'industries|industrial|manufacturing|mfg|consulting|consultants?|'
    r'management|mgmt|trading|traders?|marketing|properties|realty|'
    r'investments?|financial|finance|capital|ventures?|partners?|'
    r'partnership|foundation|trust|society|association|assoc|'
    r'limited|limitee|limitada)\b\.?',
    re.IGNORECASE | re.UNICODE
)

DBA_PREFIX = re.compile(
    r'^(dba|d/b/a|t/a|trading\s+as|doing\s+business\s+as|formerly|fka|aka)\s*[:\-]?\s*',
    re.IGNORECASE
)


def normalize_text(text):
    if not text or pd.isna(text) or text == "nan":
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'^(--|\<\<|\>\>|\.\.)[\s]*', '', text)
    text = re.sub(r'\.(com|net|org|co|io)\b', '', text)
    text = re.sub(r"[^\w\s]", ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_name_deep(text):
    if not text or pd.isna(text) or text == "nan":
        return ""
    text = str(text).lower().strip()
    text = DBA_PREFIX.sub('', text)
    text = re.sub(r'^(--|\<\<|\>\>|\.\.)[\s]*', '', text)
    text = re.sub(r'\.(com|net|org|co|io)\b', '', text)
    text = LEGAL_SUFFIXES.sub('', text)
    text = re.sub(r"[^\w\s]", ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).strip()


def extract_postal_code(address):
    if not address or pd.isna(address) or address == "nan":
        return ""
    address = str(address)
    m = re.search(r'\b(\d{5})(?:-\d{4})?\b', address)
    if m: return m.group(1)
    m = re.search(r'\b(\d{6})\b', address)
    if m: return m.group(1)
    return ""


def extract_house_number(address):
    if not address or pd.isna(address) or address == "nan":
        return ""
    address = str(address).strip()
    m = re.match(r'^(\d+[\-/]?\d*)\s', address)
    return m.group(1) if m else ""


def get_word_set(text):
    if not text: return set()
    return {w for w in text.split() if len(w) > 1}


def unicode_script(char):
    try:
        name = unicodedata.name(char, '')
        if 'DEVANAGARI' in name: return 'devanagari'
        if 'TELUGU' in name: return 'telugu'
        if 'KANNADA' in name: return 'kannada'
        if 'TAMIL' in name: return 'tamil'
        if 'ARABIC' in name: return 'arabic'
        return 'latin'
    except:
        return 'unknown'


def detect_script(text):
    if not text: return 'empty'
    scripts = Counter()
    for c in text:
        if c.isalpha(): scripts[unicode_script(c)] += 1
    return scripts.most_common(1)[0][0] if scripts else 'empty'


FEATURE_NAMES = [
    'tfidf_composite', 'name_exact', 'name_seqmatch', 'name_jaccard',
    'name_containment', 'name_len_ratio', 'name_word_diff',
    'name_first_word_match', 'script_mismatch',
    'addr_exact', 'addr_jaccard', 'addr_seqmatch',
    'postal_match', 'house_num_match', 'both_addr_missing',
    'one_addr_missing', 'name_in_addr_overlap', 'addr_containment'
]


def compute_pairwise_features(s1_name, s1_addr, s2_name, s2_addr, tfidf_score):
    features = {}
    features['tfidf_composite'] = tfidf_score
    
    name1 = normalize_name_deep(s1_name)
    name2 = normalize_name_deep(s2_name)
    
    features['name_exact'] = 1.0 if name1 == name2 and name1 != '' else 0.0
    features['name_seqmatch'] = rapidfuzz.fuzz.ratio(name1, name2) / 100.0 if name1 and name2 else 0.0
    
    w1 = get_word_set(name1)
    w2 = get_word_set(name2)
    features['name_jaccard'] = len(w1 & w2) / len(w1 | w2) if w1 and w2 else 0.0
    features['name_containment'] = len(w1 & w2) / min(len(w1), len(w2)) if w1 and w2 else 0.0
    features['name_len_ratio'] = min(len(name1), len(name2)) / max(len(name1), len(name2)) if name1 and name2 else 0.0
    features['name_word_diff'] = abs(len(w1) - len(w2))
    
    words1 = name1.split() if name1 else []
    words2 = name2.split() if name2 else []
    features['name_first_word_match'] = 1.0 if words1 and words2 and words1[0] == words2[0] and len(words1[0]) > 1 else 0.0
    
    features['script_mismatch'] = 0.0 if detect_script(s1_name) == detect_script(s2_name) else 1.0
    
    addr1 = normalize_text(s1_addr)
    addr2 = normalize_text(s2_addr)
    features['addr_exact'] = 1.0 if addr1 == addr2 and addr1 != '' else 0.0
    
    aw1 = get_word_set(addr1)
    aw2 = get_word_set(addr2)
    features['addr_jaccard'] = len(aw1 & aw2) / len(aw1 | aw2) if aw1 and aw2 else 0.0
    features['addr_seqmatch'] = rapidfuzz.fuzz.ratio(addr1, addr2) / 100.0 if addr1 and addr2 else 0.0
    
    pc1 = extract_postal_code(s1_addr)
    pc2 = extract_postal_code(s2_addr)
    features['postal_match'] = (1.0 if pc1 == pc2 else -1.0) if pc1 and pc2 else 0.0
    
    hn1 = extract_house_number(s1_addr)
    hn2 = extract_house_number(s2_addr)
    features['house_num_match'] = (1.0 if hn1 == hn2 else -1.0) if hn1 and hn2 else 0.0
    
    features['both_addr_missing'] = 1.0 if not addr1 and not addr2 else 0.0
    features['one_addr_missing'] = 1.0 if bool(addr1) != bool(addr2) else 0.0
    features['name_in_addr_overlap'] = len(w1 & aw2) / len(w1) if w1 and aw2 else 0.0
    features['addr_containment'] = len(aw1 & aw2) / min(len(aw1), len(aw2)) if aw1 and aw2 else 0.0
    
    return features


def process_country_lgbm(country, model, threshold):
    """Process one country using LightGBM reranker instead of threshold."""
    print(f"\n{'='*70}")
    print(f"  PROCESSING: {country.upper()} (LightGBM Reranker + Global M2O)")
    print(f"{'='*70}")
    t_start = time.time()
    
    part_match = os.path.join(PARTS_DIR, f"match_{country}.tsv")
    part_cand = os.path.join(PARTS_DIR, f"cand_{country}.tsv")
    
    if os.path.isfile(part_match) and os.path.isfile(part_cand):
        s1_count_df = pd.read_csv(os.path.join(DATA_DIR, "test_source1.tsv"), sep="\t", usecols=['country'])
        expected_n = int((s1_count_df['country'] == country).sum())
        del s1_count_df
        
        country_matched = 0
        country_preds = 0
        n_lines = 0
        with open(part_match, "r", encoding="utf-8") as fm:
            for line in fm:
                n_lines += 1
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[1].strip():
                    country_matched += 1
                    country_preds += len(parts[1].split(","))
        
        if n_lines == expected_n:
            avg_preds = country_preds / n_lines if n_lines > 0 else 0
            avg_per_matched = country_preds / country_matched if country_matched > 0 else 0
            print(f"[{country}] Checkpoint found! Skipping recomputation ({n_lines:,} entities).")
            print(f"[{country}] Matched: {country_matched:,}/{n_lines:,} ({100*country_matched/n_lines:.1f}%) | Preds: {country_preds:,} (avg {avg_preds:.2f}/entity, {avg_per_matched:.2f}/non-empty entity)")
            return n_lines, country_matched, country_preds
    
    # 1. Load data
    print(f"[{country}] Loading test data...", end=" ", flush=True)
    s1 = pd.read_csv(os.path.join(DATA_DIR, "test_source1.tsv"), sep="\t", dtype=str).fillna("")
    s1 = s1[s1['country'] == country].reset_index(drop=True)
    n_s1 = len(s1)
    
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
    print(f"S1={n_s1:,} S2S3={n_s2s3:,}")
    
    # 2. Prepare text
    print(f"[{country}] Preparing text...", end=" ", flush=True)
    s1_full = (s1['business_name'] + " " + s1['business_name'] + " " + s1['business_address']).apply(normalize_text).values
    s2s3_full = (s2s3['business_name'] + " " + s2s3['business_name'] + " " + s2s3['business_address']).apply(normalize_text).values
    s1_addr_text = s1['business_address'].apply(normalize_text).values
    s2s3_addr_text = s2s3['business_address'].apply(normalize_text).values
    
    s1_ids = s1['entity_id'].values
    s2s3_ids = s2s3['entity_id'].values
    s2s3_id_to_idx = {eid: idx for idx, eid in enumerate(s2s3_ids)}
    s1_names = s1['business_name'].values
    s1_addrs_raw = s1['business_address'].values
    s2s3_names = s2s3['business_name'].values
    s2s3_addrs_raw = s2s3['business_address'].values
    del s1, s2s3
    gc.collect()
    print("Done")
    
    # 3. Channel A: TF-IDF
    print(f"[{country}] Building TF-IDF...", end=" ", flush=True)
    t0 = time.time()
    vec_a = TfidfVectorizer(
        analyzer='word', max_features=80000, min_df=2, max_df=0.01,
        sublinear_tf=True, norm='l2', dtype=np.float32,
        token_pattern=r'(?u)\b\w+\b'
    )
    fit_sample = np.concatenate([s1_full[:min(100000, n_s1)], s2s3_full[:min(500000, n_s2s3)]])
    vec_a.fit(fit_sample)
    del fit_sample
    s1_mat_a = vec_a.transform(s1_full)
    s2s3_mat_a = vec_a.transform(s2s3_full)
    s2s3_mat_a_T = s2s3_mat_a.T.tocsc()
    del vec_a, s2s3_mat_a
    gc.collect()
    print(f"Done ({time.time()-t0:.1f}s)")
    
    # 4. Channel B: Address Jaccard
    print(f"[{country}] Building Address Jaccard...", end=" ", flush=True)
    t0 = time.time()
    vec_b = CountVectorizer(
        binary=True, analyzer='word', token_pattern=r'(?u)\b\w+\b',
        min_df=2, max_df=0.02, max_features=60000
    )
    fit_sample = np.concatenate([s1_addr_text[:min(100000, n_s1)], s2s3_addr_text[:min(500000, n_s2s3)]])
    vec_b.fit(fit_sample)
    del fit_sample
    A = vec_b.transform(s1_addr_text)
    B = vec_b.transform(s2s3_addr_text)
    B_T = B.T.tocsc()
    len_A = np.diff(A.indptr)
    len_B = np.diff(B.indptr)
    del vec_b, B, s1_full, s2s3_full, s1_addr_text, s2s3_addr_text
    gc.collect()
    print(f"Done ({time.time()-t0:.1f}s)")
    
    # 5. Batched inference with LightGBM
    n_batches = int(np.ceil(n_s1 / BATCH_SIZE))
    print(f"[{country}] Running LightGBM inference ({n_batches} batches)...")
    t_inf = time.time()
    
    # Collect all predictions for M2O
    s1_triple_indices = []
    cid_triple_values = []
    score_triple_values = []  # LightGBM probability
    
    with open(part_cand, "w", encoding="utf-8") as fc:
        for b in range(n_batches):
            b_start = b * BATCH_SIZE
            b_end = min(b_start + BATCH_SIZE, n_s1)
            
            # Channel A
            b_sims_a = s1_mat_a[b_start:b_end] @ s2s3_mat_a_T
            if not isinstance(b_sims_a, sparse.csr_matrix):
                b_sims_a = b_sims_a.tocsr()
            
            # Channel B
            b_inter_b = A[b_start:b_end] @ B_T
            if not isinstance(b_inter_b, sparse.csr_matrix):
                b_inter_b = b_inter_b.tocsr()
            
            batch_feat_rows = []
            batch_meta = []  # (local_i_within_batch, cid)
            
            for i in range(b_end - b_start):
                global_i = b_start + i
                sid = s1_ids[global_i]
                
                # Channel A top-K
                p0_a, p1_a = b_sims_a.indptr[i], b_sims_a.indptr[i+1]
                top_a = {}
                if p0_a < p1_a:
                    data_a = b_sims_a.data[p0_a:p1_a]
                    indices_a = b_sims_a.indices[p0_a:p1_a]
                    k_a = min(TOP_K_CHANNEL_A, len(data_a))
                    if len(data_a) > TOP_K_CHANNEL_A:
                        top_idx_a = np.argpartition(data_a, -k_a)[-k_a:]
                    else:
                        top_idx_a = np.arange(len(data_a))
                    for idx in top_idx_a:
                        cid = s2s3_ids[indices_a[idx]]
                        top_a[cid] = float(data_a[idx])
                
                # Channel B top-K
                p0_b, p1_b = b_inter_b.indptr[i], b_inter_b.indptr[i+1]
                top_b = {}
                if p0_b < p1_b and len_A[global_i] > 0:
                    inter_data = b_inter_b.data[p0_b:p1_b]
                    inter_indices = b_inter_b.indices[p0_b:p1_b]
                    denoms = len_A[global_i] + len_B[inter_indices] - inter_data
                    jaccard_scores = inter_data / denoms
                    k_b = min(TOP_K_CHANNEL_B, len(jaccard_scores))
                    if len(jaccard_scores) > TOP_K_CHANNEL_B:
                        top_idx_b = np.argpartition(jaccard_scores, -k_b)[-k_b:]
                    else:
                        top_idx_b = np.arange(len(jaccard_scores))
                    for idx in top_idx_b:
                        cid = s2s3_ids[inter_indices[idx]]
                        top_b[cid] = float(jaccard_scores[idx])
                
                # Union
                union_cands = {}
                for cid, s in top_a.items(): union_cands[cid] = s
                for cid, s in top_b.items():
                    union_cands[cid] = max(union_cands.get(cid, 0), s)
                
                sorted_union = sorted(union_cands.items(), key=lambda x: -x[1])[:MAX_UNION_CANDIDATES]
                cand_id_list = [c[0] for c in sorted_union]
                fc.write(f"{sid}\t{','.join(cand_id_list)}\n")
                
                # Extract features for LightGBM
                for cid, tfidf_score in sorted_union:
                    cid_idx = s2s3_id_to_idx.get(cid)
                    if cid_idx is None: continue
                    
                    feats = compute_pairwise_features(
                        s1_names[global_i], s1_addrs_raw[global_i],
                        s2s3_names[cid_idx], s2s3_addrs_raw[cid_idx],
                        tfidf_score
                    )
                    batch_feat_rows.append([feats[fn] for fn in FEATURE_NAMES])
                    batch_meta.append((global_i, cid))
            
            # Batch LightGBM prediction
            if batch_feat_rows:
                X_batch = np.array(batch_feat_rows, dtype=np.float32)
                probs = model.predict(X_batch)
                
                for j, (global_i, cid) in enumerate(batch_meta):
                    if probs[j] >= threshold:
                        s1_triple_indices.append(global_i)
                        cid_triple_values.append(cid)
                        score_triple_values.append(float(probs[j]))
            
            batch_feat_rows.clear()
            batch_meta.clear()
            
            elapsed = time.time() - t_inf
            pct = 100.0 * (b + 1) / n_batches
            eta = (elapsed / (b + 1)) * (n_batches - b - 1)
            if (b + 1) % 3 == 0 or b == n_batches - 1:
                print(f"  [{country}] Batch {b+1}/{n_batches} ({pct:.0f}%) | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")
    
    del s1_mat_a, s2s3_mat_a_T, A, B_T, len_A, len_B
    gc.collect()
    
    # 6. Global M2O
    print(f"[{country}] Global M2O on {len(score_triple_values):,} claims...", end=" ", flush=True)
    t_m2o = time.time()
    
    assigned_matches_per_s1 = {i: [] for i in range(n_s1)}
    if len(score_triple_values) > 0:
        scores_arr = np.array(score_triple_values, dtype=np.float32)
        sort_order = np.argsort(-scores_arr)
        assigned_cids = set()
        for idx in sort_order:
            cid = cid_triple_values[idx]
            if cid not in assigned_cids:
                assigned_cids.add(cid)
                s1_idx = s1_triple_indices[idx]
                assigned_matches_per_s1[s1_idx].append(cid)
    
    del s1_triple_indices, cid_triple_values, score_triple_values
    gc.collect()
    print(f"Done ({time.time()-t_m2o:.1f}s)")
    
    # Write matches
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
    
    total_time = time.time() - t_start
    avg_preds = country_preds / n_s1 if n_s1 > 0 else 0
    avg_per_matched = country_preds / country_matched if country_matched > 0 else 0
    print(f"[{country}] Done in {total_time:.1f}s | Matched: {country_matched:,}/{n_s1:,} ({100*country_matched/n_s1:.1f}%) | Preds: {country_preds:,} (avg {avg_preds:.2f}/entity, {avg_per_matched:.2f}/non-empty entity)")
    
    del assigned_matches_per_s1, s1_ids, s2s3_ids
    gc.collect()
    
    return n_s1, country_matched, country_preds


def assemble_submission():
    """Assemble all country parts into final submission files in test_source1.tsv order."""
    print(f"\n{'='*70}")
    print("  ASSEMBLING FINAL SUBMISSION (LightGBM)")
    print(f"{'='*70}")
    
    s1_all = pd.read_csv(os.path.join(DATA_DIR, "test_source1.tsv"), sep="\t", usecols=['entity_id'])
    ordered_ids = s1_all['entity_id'].tolist()
    total = len(ordered_ids)
    
    # Load all match parts
    match_map = {}
    cand_map = {}
    for country in ["France", "India", "US"]:
        mp = os.path.join(PARTS_DIR, f"match_{country}.tsv")
        cp = os.path.join(PARTS_DIR, f"cand_{country}.tsv")
        if os.path.isfile(mp):
            with open(mp, "r") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    match_map[parts[0]] = parts[1] if len(parts) >= 2 else ""
        if os.path.isfile(cp):
            with open(cp, "r") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    cand_map[parts[0]] = parts[1] if len(parts) >= 2 else ""
    
    # Write
    with open(MATCHING_OUT, "w") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in ordered_ids:
            fm.write(f"{sid}\t{match_map.get(sid, '')}\n")
    
    with open(CANDIDATE_OUT, "w") as fc:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in ordered_ids:
            fc.write(f"{sid}\t{cand_map.get(sid, '')}\n")
    
    print(f"  Written {total:,} rows to {MATCHING_OUT} and {CANDIDATE_OUT}")
    
    # Create zip
    import shutil, zipfile
    zip_path = os.path.join(OUTPUT_DIR, "matching_results.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(MATCHING_OUT, "matching_results.tsv")
    print(f"  Created {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.1f} MB)")


def main():
    print("\n╔" + "═" * 78 + "╗")
    print("║" + " SUBMISSION WITH LIGHTGBM RERANKER ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run src/lgbm_reranker.py first to train the model.")
        sys.exit(1)
    
    # Load model & metadata
    model = lgb.Booster(model_file=MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    threshold = meta['best_threshold']
    print(f"Loaded model from {MODEL_PATH}")
    print(f"Using threshold: {threshold:.3f} (CV F₀.₅: {meta['avg_f05']:.4f})")
    
    t_total = time.time()
    stats = []
    
    for country in ["France", "India", "US"]:
        n, matched, preds = process_country_lgbm(country, model, threshold)
        stats.append((country, n, matched, preds))
    
    # Assemble
    assemble_submission()
    
    # Summary
    total_time = time.time() - t_total
    print(f"\n{'='*80}")
    print(f"  SUBMISSION COMPLETE ({total_time:.1f}s = {total_time/60:.1f} min)")
    print(f"{'='*80}")
    total_preds = 0
    total_entities = 0
    total_matched = 0
    for c, n, m, p in stats:
        avg_m = p / m if m > 0 else 0
        print(f"  {c:8s}: {m:,}/{n:,} matched ({100*m/n:.1f}%) | {p:,} preds ({p/n:.2f}/entity, {avg_m:.2f}/non-empty)")
        total_preds += p
        total_entities += n
        total_matched += m
    total_avg_m = total_preds / total_matched if total_matched > 0 else 0
    print(f"  {'TOTAL':8s}: {total_entities:,} entities | {total_preds:,} predictions ({total_preds/total_entities:.2f}/entity, {total_avg_m:.2f}/non-empty)")


if __name__ == "__main__":
    main()
