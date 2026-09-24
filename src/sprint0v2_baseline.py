#!/usr/bin/env python3
"""
Sprint 0 v2: Improved Entity Resolution Pipeline
Amazon ML Challenge 2026

Key improvements over v1:
- Word-level TF-IDF on COMBINED name+address (not name-only)
- Much faster: word-level is 5-10x faster than char n-grams
- Better name normalization (remove common business suffixes)
- Address-aware scoring eliminates location-mismatched false positives
- Blocking recall analysis
- Higher threshold sweep range

Usage:
    python src/sprint0v2_baseline.py [--eval-only] [--sample-size 50000]
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import re
import os
import time
import gc
import argparse
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
DATA_ROOT = "dataset/student_resource/dataset"
OUTPUT_DIR = "output"

# Word-level TF-IDF — much faster than char n-grams
TFIDF_NAME_PARAMS = dict(
    analyzer='word',
    max_features=200000,
    min_df=2,
    max_df=0.005,       # Aggressive: remove tokens in >0.5% of docs
    sublinear_tf=True,
    norm='l2',
    dtype=np.float32,
    token_pattern=r'(?u)\b\w[\w\'-]*\b',  # handles unicode, apostrophes, hyphens
)

TFIDF_ADDR_PARAMS = dict(
    analyzer='word',
    max_features=100000,
    min_df=2,
    max_df=0.005,
    sublinear_tf=True,
    norm='l2',
    dtype=np.float32,
    token_pattern=r'(?u)\b\w[\w\'-]*\b',
)

BATCH_SIZE = 10000    # Larger batches OK with word-level (sparser)
TOP_K = 30            # Candidates per S1 entity
EVAL_SAMPLE = 50000

THRESHOLDS_TO_SWEEP = np.arange(0.10, 0.96, 0.025)

# Name weight vs address weight for combined scoring
NAME_WEIGHT = 0.55
ADDR_WEIGHT = 0.45

# ============================================================
# COMMON BUSINESS SUFFIXES TO REMOVE (improve name discrimination)
# ============================================================
BUSINESS_SUFFIXES_EN = [
    r'\b(llc|llp|inc|corp|corporation|co|company|ltd|limited|plc)\b',
    r'\b(pvt|private|public|services|solutions|enterprises|holdings)\b',
    r'\b(group|international|associates|consultants|partners|partnership)\b',
    r'\b(industries|technologies|tech|systems|global|ventures|capital)\b',
    r'\b(foundation|institute|academy|trading|traders|exports|imports)\b',
    r'\b(the|and|of|for|a|an)\b',
]

BUSINESS_SUFFIXES_HI = [
    r'प्राइवेट\s*लिमिटेड', r'प्रा\.?\s*लि\.?', r'लिमिटेड', r'प्राइवेट',
    r'एलएलपी', r'एलएलसी', r'कंपनी', r'ग्रुप',
]

BUSINESS_SUFFIXES_FR = [
    r'\b(sarl|sas|sa|sci|eurl|sasu|ste|societe|société|cie|et)\b',
]

SUFFIX_PATTERN = re.compile(
    '|'.join(BUSINESS_SUFFIXES_EN + BUSINESS_SUFFIXES_HI + BUSINESS_SUFFIXES_FR),
    re.IGNORECASE | re.UNICODE
)

ADDRESS_ABBREVS = {
    r'\brd\b': 'road', r'\bst\b': 'street', r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard', r'\bdr\b': 'drive', r'\bln\b': 'lane',
    r'\bct\b': 'court', r'\bpl\b': 'place', r'\bhwy\b': 'highway',
    r'\bpkwy\b': 'parkway', r'\bcir\b': 'circle', r'\bsq\b': 'square',
    r'\br\.\b': 'rue', r'\brue\b': 'rue', r'\bbd\b': 'boulevard',
}

# ============================================================
# DATA LOADING
# ============================================================
def load_tsv(path):
    """Load a TSV file with proper type handling."""
    print(f"  Loading {os.path.basename(path)}...", end=" ", flush=True)
    t0 = time.time()
    df = pd.read_csv(path, sep='\t', dtype=str)
    for col in ['business_name', 'business_address', 'country']:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    print(f"({len(df):,} rows, {time.time()-t0:.1f}s)")
    return df


def load_ground_truth(path):
    """Load ground truth → dict: s1_id → set of matched ids."""
    print(f"  Loading ground truth...", end=" ", flush=True)
    t0 = time.time()
    df = pd.read_csv(path, sep='\t', dtype=str)
    df['matched_entity_ids'] = df['matched_entity_ids'].fillna("")
    gt = {}
    for _, row in df.iterrows():
        s1_id = row['source1_entity_id']
        mids = row['matched_entity_ids'].strip()
        gt[s1_id] = set(mids.split(",")) if mids else set()
    print(f"({len(gt):,} entities, {time.time()-t0:.1f}s)")
    return gt


# ============================================================
# TEXT NORMALIZATION
# ============================================================
def normalize_name(name):
    """Normalize business name: lowercase, remove suffixes, clean."""
    if not name:
        return ""
    name = name.lower().strip()
    # Remove noise prefixes
    name = re.sub(r'^(--|<<|>>|\.\.)\s*', '', name)
    # Remove .com/.net/.org suffixes (website names used as business names)
    name = re.sub(r'\.(com|net|org|co|io)\b', '', name)
    # Remove common business suffixes
    name = SUFFIX_PATTERN.sub(' ', name)
    # Remove punctuation except apostrophes and hyphens
    name = re.sub(r"[^\w\s\'-]", ' ', name, flags=re.UNICODE)
    # Normalize whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def normalize_address(addr):
    """Normalize address: lowercase, expand abbreviations, clean."""
    if not addr:
        return ""
    addr = addr.lower().strip()
    # Expand common abbreviations
    for pattern, replacement in ADDRESS_ABBREVS.items():
        addr = re.sub(pattern, replacement, addr, flags=re.IGNORECASE)
    # Remove punctuation
    addr = re.sub(r"[^\w\s]", ' ', addr, flags=re.UNICODE)
    # Remove common noise words
    addr = re.sub(r'\b(near|opp|opposite|behind|beside|next to|adjacent)\b', ' ', addr)
    # Normalize whitespace
    addr = re.sub(r'\s+', ' ', addr).strip()
    return addr


# ============================================================
# EVALUATION: F_0.5 SCORE
# ============================================================
def compute_f05_per_entity(true_set, pred_set):
    """Compute F_0.5 for a single entity."""
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


def compute_macro_f05(predictions, ground_truth, s1_ids):
    """Compute macro-averaged F_0.5."""
    scores = []
    for s1_id in s1_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_set = set(predictions.get(s1_id, []))
        scores.append(compute_f05_per_entity(true_set, pred_set))
    return np.mean(scores), {
        'n_entities': len(scores),
        'n_perfect': sum(1 for s in scores if s == 1.0),
        'n_zero': sum(1 for s in scores if s == 0.0),
        'mean_score': np.mean(scores),
    }


# ============================================================
# BLOCKING: DUAL TF-IDF (NAME + ADDRESS)
# ============================================================
def tfidf_dual_block_country(s1_df, s2s3_df, country, top_k=TOP_K):
    """
    Two-stage TF-IDF blocking per country:
    1. Name TF-IDF for primary candidate retrieval
    2. Address TF-IDF for re-ranking
    Returns: list of (s1_id, [(s2s3_id, name_score, addr_score), ...])
    """
    n_s1 = len(s1_df)
    n_s2s3 = len(s2s3_df)
    print(f"\n  [{country}] S1: {n_s1:,} | S2+S3: {n_s2s3:,}")
    
    if n_s1 == 0 or n_s2s3 == 0:
        return [(sid, []) for sid in s1_df['entity_id'].values]
    
    t0 = time.time()
    
    # Normalize
    print(f"    Normalizing text...", end=" ", flush=True)
    t1 = time.time()
    s1_names = s1_df['business_name'].apply(normalize_name).values
    s2s3_names = s2s3_df['business_name'].apply(normalize_name).values
    s1_addrs = s1_df['business_address'].apply(normalize_address).values
    s2s3_addrs = s2s3_df['business_address'].apply(normalize_address).values
    s1_ids = s1_df['entity_id'].values
    s2s3_ids = s2s3_df['entity_id'].values
    print(f"{time.time()-t1:.1f}s")
    
    # ---- NAME TF-IDF ----
    print(f"    Building NAME TF-IDF...", end=" ", flush=True)
    t1 = time.time()
    name_vec = TfidfVectorizer(**TFIDF_NAME_PARAMS)
    name_vec.fit(np.concatenate([s1_names, s2s3_names]))
    s1_name_tfidf = name_vec.transform(s1_names)
    s2s3_name_tfidf = name_vec.transform(s2s3_names)
    print(f"vocab={len(name_vec.vocabulary_):,}, {time.time()-t1:.1f}s")
    del name_vec
    gc.collect()
    
    # ---- ADDRESS TF-IDF ----
    print(f"    Building ADDR TF-IDF...", end=" ", flush=True)
    t1 = time.time()
    addr_vec = TfidfVectorizer(**TFIDF_ADDR_PARAMS)
    addr_vec.fit(np.concatenate([s1_addrs, s2s3_addrs]))
    s1_addr_tfidf = addr_vec.transform(s1_addrs)
    s2s3_addr_tfidf = addr_vec.transform(s2s3_addrs)
    print(f"vocab={len(addr_vec.vocabulary_):,}, {time.time()-t1:.1f}s")
    del addr_vec, s1_names, s2s3_names, s1_addrs, s2s3_addrs
    gc.collect()
    
    # ---- BATCH SIMILARITY ----
    results = []
    n_batches = (n_s1 + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"    Computing dual similarities ({n_batches} batches)...", flush=True)
    t2 = time.time()
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, n_s1)
        
        # Name similarities
        batch_name = s1_name_tfidf[batch_start:batch_end]
        name_sims = batch_name @ s2s3_name_tfidf.T
        if not isinstance(name_sims, sparse.csr_matrix):
            name_sims = name_sims.tocsr()
        
        # Address similarities
        batch_addr = s1_addr_tfidf[batch_start:batch_end]
        addr_sims = batch_addr @ s2s3_addr_tfidf.T
        if not isinstance(addr_sims, sparse.csr_matrix):
            addr_sims = addr_sims.tocsr()
        
        for i in range(batch_end - batch_start):
            s1_id = s1_ids[batch_start + i]
            
            name_row = name_sims.getrow(i)
            
            if name_row.nnz == 0:
                results.append((s1_id, []))
                continue
            
            # Get top-K by name similarity
            n_data = name_row.data
            n_indices = name_row.indices
            
            if len(n_data) <= top_k:
                top_idx = np.arange(len(n_data))
            else:
                top_idx = np.argpartition(n_data, -top_k)[-top_k:]
            
            top_idx = top_idx[np.argsort(-n_data[top_idx])]
            
            # Get address scores for top-K candidates
            addr_row = addr_sims.getrow(i)
            # Convert to dict for fast lookup
            addr_dict = {}
            if addr_row.nnz > 0:
                for j_idx, j_val in zip(addr_row.indices, addr_row.data):
                    addr_dict[j_idx] = j_val
            
            candidates = []
            for j in top_idx:
                col_idx = n_indices[j]
                name_score = float(n_data[j])
                addr_score = float(addr_dict.get(col_idx, 0.0))
                candidates.append((s2s3_ids[col_idx], name_score, addr_score))
            
            results.append((s1_id, candidates))
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - t2
            pct = 100.0 * (batch_idx + 1) / n_batches
            eta = elapsed / (batch_idx + 1) * (n_batches - batch_idx - 1)
            print(f"      Batch {batch_idx+1}/{n_batches} ({pct:.0f}%) "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)
    
    total_time = time.time() - t0
    total_candidates = sum(len(c) for _, c in results)
    print(f"    Done: {len(results):,} S1, {total_candidates:,} candidates, {total_time:.1f}s")
    
    del s1_name_tfidf, s2s3_name_tfidf, s1_addr_tfidf, s2s3_addr_tfidf
    gc.collect()
    
    return results


def run_pipeline(s1_df, s2_df, s3_df):
    """Run full blocking pipeline across all countries."""
    s2s3_df = pd.concat([s2_df, s3_df], ignore_index=True)
    countries = sorted(s1_df['country'].unique())
    print(f"\nCountries to process: {countries}")
    
    all_results = {}
    for country in countries:
        s1_c = s1_df[s1_df['country'] == country].copy()
        s2s3_c = s2s3_df[s2s3_df['country'] == country].copy()
        
        results = tfidf_dual_block_country(s1_c, s2s3_c, country)
        for s1_id, candidates in results:
            all_results[s1_id] = candidates
        
        del s1_c, s2s3_c, results
        gc.collect()
    
    # Ensure all S1 entities have an entry
    for s1_id in s1_df['entity_id'].values:
        if s1_id not in all_results:
            all_results[s1_id] = []
    
    print(f"\nPipeline complete: {len(all_results):,} S1 entities")
    return all_results


def compute_combined_score(name_score, addr_score):
    """Combine name and address scores."""
    return NAME_WEIGHT * name_score + ADDR_WEIGHT * addr_score


def apply_threshold(all_results, threshold):
    """Apply combined score threshold."""
    predictions = {}
    for s1_id, candidates in all_results.items():
        matched = [
            cid for cid, ns, ads in candidates
            if compute_combined_score(ns, ads) >= threshold
        ]
        predictions[s1_id] = matched
    return predictions


# ============================================================
# BLOCKING RECALL ANALYSIS
# ============================================================
def analyze_blocking_recall(all_results, ground_truth, s1_ids):
    """What fraction of true matches appear in our candidate set?"""
    print(f"\n--- Blocking Recall Analysis ---")
    total_true = 0
    total_found = 0
    total_missed = 0
    
    for s1_id in s1_ids:
        true_matches = ground_truth.get(s1_id, set())
        candidate_ids = set(cid for cid, _, _ in all_results.get(s1_id, []))
        
        found = len(true_matches & candidate_ids)
        missed = len(true_matches - candidate_ids)
        
        total_true += len(true_matches)
        total_found += found
        total_missed += missed
    
    recall = total_found / max(total_true, 1)
    print(f"  True matches in sample: {total_true:,}")
    print(f"  Found in candidates: {total_found:,} ({100*recall:.1f}%)")
    print(f"  Missed by blocking: {total_missed:,} ({100*total_missed/max(total_true,1):.1f}%)")
    print(f"  → Blocking recall = {recall:.4f} (upper bound for F₀.₅)")
    return recall


# ============================================================
# THRESHOLD TUNING
# ============================================================
def tune_threshold(all_results, ground_truth, s1_ids):
    """Sweep thresholds to find optimal F_0.5."""
    print(f"\n{'='*60}")
    print(f"  THRESHOLD TUNING (name_w={NAME_WEIGHT}, addr_w={ADDR_WEIGHT})")
    print(f"{'='*60}")
    
    best_f05 = -1
    best_threshold = 0.5
    
    for threshold in THRESHOLDS_TO_SWEEP:
        predictions = apply_threshold(all_results, threshold)
        f05, details = compute_macro_f05(predictions, ground_truth, s1_ids)
        
        n_matched = sum(1 for v in predictions.values() if len(v) > 0)
        n_total_preds = sum(len(v) for v in predictions.values())
        avg_preds = n_total_preds / max(len(s1_ids), 1)
        
        marker = " ★" if f05 > best_f05 else ""
        print(f"  τ={threshold:.3f} | F₀.₅={f05:.4f} | "
              f"avg_preds={avg_preds:.1f} | matched={n_matched:,} | "
              f"perfect={details['n_perfect']:,} | zero={details['n_zero']:,}{marker}")
        
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = threshold
    
    print(f"\n  ★ Best threshold: {best_threshold:.3f} → F₀.₅ = {best_f05:.4f}")
    return best_threshold, best_f05


# ============================================================
# OUTPUT GENERATION
# ============================================================
def generate_output(predictions, s1_ids, output_dir, all_results=None):
    """Generate matching_results.tsv and candidate_pairs.tsv."""
    os.makedirs(output_dir, exist_ok=True)
    
    # matching_results.tsv
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    with open(matching_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ids:
            matched = predictions.get(s1_id, [])
            # Deduplicate (defensive)
            seen = set()
            unique_matched = []
            for m in matched:
                if m not in seen:
                    seen.add(m)
                    unique_matched.append(m)
            f.write(f"{s1_id}\t{','.join(unique_matched)}\n")
    print(f"  Written: {matching_path}")
    
    # candidate_pairs.tsv
    if all_results is not None:
        candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
        with open(candidate_path, 'w', encoding='utf-8') as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in s1_ids:
                candidates = all_results.get(s1_id, [])
                cand_ids = list(dict.fromkeys(cid for cid, _, _ in candidates))  # dedupe, preserve order
                f.write(f"{s1_id}\t{','.join(cand_ids)}\n")
        print(f"  Written: {candidate_path}")
    
    n_matched = sum(1 for s1_id in s1_ids if len(predictions.get(s1_id, [])) > 0)
    n_total_preds = sum(len(predictions.get(s1_id, [])) for s1_id in s1_ids)
    print(f"\n  Total S1: {len(s1_ids):,} | Matched: {n_matched:,} "
          f"({100.0*n_matched/len(s1_ids):.1f}%) | Preds: {n_total_preds:,} "
          f"| Avg: {n_total_preds/len(s1_ids):.2f}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Sprint 0 v2: Dual TF-IDF Entity Resolution")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--sample-size", type=int, default=EVAL_SAMPLE)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    
    t_start = time.time()
    
    print("█" * 60)
    print("  SPRINT 0 v2: DUAL TF-IDF ENTITY RESOLUTION")
    print("█" * 60)
    
    # ---- Load Training Data ----
    print(f"\n--- Loading Training Data ---")
    train_s1 = load_tsv(f"{DATA_ROOT}/train/train_source1.tsv")
    train_s2 = load_tsv(f"{DATA_ROOT}/train/train_source2.tsv")
    train_s3 = load_tsv(f"{DATA_ROOT}/train/train_source3.tsv")
    gt = load_ground_truth(f"{DATA_ROOT}/train/train_ground_truth.tsv")
    
    # ---- Stratified Sample ----
    sample_size = min(args.sample_size, len(train_s1))
    print(f"\n--- Sampling {sample_size:,} S1 entities ---")
    np.random.seed(42)
    sample_indices = []
    for country in train_s1['country'].unique():
        cidx = train_s1[train_s1['country'] == country].index.values
        n = max(1, int(sample_size * len(cidx) / len(train_s1)))
        sample_indices.extend(np.random.choice(cidx, size=min(n, len(cidx)), replace=False))
    
    sample_s1 = train_s1.loc[sample_indices].reset_index(drop=True)
    sample_ids = sample_s1['entity_id'].values
    print(f"  Sampled {len(sample_s1):,}: " + 
          ", ".join(f"{c}={int((sample_s1['country']==c).sum()):,}" for c in sorted(sample_s1['country'].unique())))
    
    # ---- Run Pipeline ----
    print(f"\n{'='*60}")
    print(f"  PHASE 1: TRAINING VALIDATION")
    print(f"{'='*60}")
    
    sample_results = run_pipeline(sample_s1, train_s2, train_s3)
    
    # ---- Blocking Recall ----
    analyze_blocking_recall(sample_results, gt, sample_ids)
    
    # ---- Threshold Tuning ----
    if args.threshold is not None:
        best_threshold = args.threshold
        predictions = apply_threshold(sample_results, best_threshold)
        best_f05, _ = compute_macro_f05(predictions, gt, sample_ids)
        print(f"\n  Fixed threshold: {best_threshold:.3f} → F₀.₅ = {best_f05:.4f}")
    else:
        best_threshold, best_f05 = tune_threshold(sample_results, gt, sample_ids)
    
    if args.eval_only:
        print(f"\n  [eval-only] Total time: {time.time()-t_start:.0f}s")
        return
    
    # ---- Free memory ----
    del train_s1, train_s2, train_s3, gt, sample_s1, sample_results
    gc.collect()
    
    # ---- Test Prediction ----
    print(f"\n{'='*60}")
    print(f"  PHASE 2: TEST PREDICTION")
    print(f"{'='*60}")
    
    test_s1 = load_tsv(f"{DATA_ROOT}/test/test_source1.tsv")
    test_s2 = load_tsv(f"{DATA_ROOT}/test/test_source2.tsv")
    test_s3 = load_tsv(f"{DATA_ROOT}/test/test_source3.tsv")
    test_ids = test_s1['entity_id'].values
    
    test_results = run_pipeline(test_s1, test_s2, test_s3)
    test_predictions = apply_threshold(test_results, best_threshold)
    
    print(f"\n--- Generating Output ---")
    generate_output(test_predictions, test_ids, OUTPUT_DIR, all_results=test_results)
    
    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  SPRINT 0 v2 COMPLETE")
    print(f"{'='*60}")
    print(f"  Val F₀.₅: {best_f05:.4f} | Threshold: {best_threshold:.3f}")
    print(f"  Total: {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
