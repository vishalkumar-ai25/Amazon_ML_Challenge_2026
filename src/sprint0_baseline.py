#!/usr/bin/env python3
"""
Sprint 0: Baseline Entity Resolution Pipeline
Amazon ML Challenge 2026

Strategy:
- Blocking: TF-IDF character n-grams on business_name, per country
- Matching: Cosine similarity threshold (no ML model)
- Threshold tuning: Sweep on training sample to maximize F_0.5

Usage:
    python src/sprint0_baseline.py [--eval-only] [--sample-size 50000]
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import re
import os
import sys
import time
import gc
import argparse
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
DATA_ROOT = "dataset/student_resource/dataset"
OUTPUT_DIR = "output"

TFIDF_PARAMS = dict(
    analyzer='char_wb',
    ngram_range=(3, 5),
    max_features=50000,
    min_df=3,
    max_df=0.01,       # Remove n-grams appearing in >1% of docs (common patterns)
    sublinear_tf=True,
    norm='l2',
    dtype=np.float32,
)

BATCH_SIZE = 5000     # S1 entities per batch for similarity computation
TOP_K = 30            # Candidates per S1 entity from blocking
EVAL_SAMPLE = 50000   # Training S1 entities for validation

THRESHOLDS_TO_SWEEP = np.arange(0.25, 0.75, 0.025)

# ============================================================
# DATA LOADING
# ============================================================
def load_tsv(path):
    """Load a TSV file with proper type handling."""
    print(f"  Loading {os.path.basename(path)}...", end=" ", flush=True)
    t0 = time.time()
    df = pd.read_csv(path, sep='\t', dtype=str)
    # Defensive sanitization: fill NaN, ensure str
    for col in ['business_name', 'business_address', 'country']:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    print(f"({len(df):,} rows, {time.time()-t0:.1f}s)")
    return df


def load_ground_truth(path):
    """Load ground truth and parse into dict: s1_id → set of matched ids."""
    print(f"  Loading ground truth...", end=" ", flush=True)
    t0 = time.time()
    df = pd.read_csv(path, sep='\t', dtype=str)
    df['matched_entity_ids'] = df['matched_entity_ids'].fillna("")
    
    gt = {}
    for _, row in df.iterrows():
        s1_id = row['source1_entity_id']
        mids = row['matched_entity_ids'].strip()
        if mids:
            gt[s1_id] = set(mids.split(","))
        else:
            gt[s1_id] = set()
    print(f"({len(gt):,} entities, {time.time()-t0:.1f}s)")
    return gt


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
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    if tp == 0:
        return 0.0
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    beta_sq = 0.25  # beta=0.5, beta^2=0.25
    f05 = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    return f05


def compute_macro_f05(predictions, ground_truth, s1_ids):
    """Compute macro-averaged F_0.5 across all S1 entities."""
    scores = []
    for s1_id in s1_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_set = set(predictions.get(s1_id, []))
        scores.append(compute_f05_per_entity(true_set, pred_set))
    
    macro_f05 = np.mean(scores)
    
    # Also compute detailed breakdown
    n_perfect = sum(1 for s in scores if s == 1.0)
    n_zero = sum(1 for s in scores if s == 0.0)
    n_partial = len(scores) - n_perfect - n_zero
    
    return macro_f05, {
        'n_entities': len(scores),
        'n_perfect': n_perfect,
        'n_zero': n_zero,
        'n_partial': n_partial,
        'mean_score': macro_f05,
        'median_score': np.median(scores),
    }


# ============================================================
# TEXT NORMALIZATION
# ============================================================
def normalize_name(name):
    """Basic name normalization for TF-IDF features."""
    if not name:
        return ""
    name = name.lower().strip()
    # Remove common noise prefixes
    name = re.sub(r'^(--|<<|>>)\s*', '', name)
    # Normalize whitespace
    name = re.sub(r'\s+', ' ', name)
    return name.strip()


# ============================================================
# TF-IDF BLOCKING & MATCHING (PER COUNTRY)
# ============================================================
def tfidf_block_country(s1_df, s2s3_df, country, top_k=TOP_K):
    """
    For a single country:
    1. Build TF-IDF on business_name for S2+S3
    2. Query S1 against S2+S3
    3. Return (s1_id, s2s3_id, similarity) triples
    
    Returns: list of (s1_id, candidate_ids_with_scores) tuples
    """
    n_s1 = len(s1_df)
    n_s2s3 = len(s2s3_df)
    print(f"\n  [{country}] S1: {n_s1:,} | S2+S3: {n_s2s3:,}")
    
    if n_s1 == 0 or n_s2s3 == 0:
        return []
    
    t0 = time.time()
    
    # Normalize names
    s1_names = s1_df['business_name'].apply(normalize_name).values
    s2s3_names = s2s3_df['business_name'].apply(normalize_name).values
    s1_ids = s1_df['entity_id'].values
    s2s3_ids = s2s3_df['entity_id'].values
    
    # Build TF-IDF vectorizer
    print(f"    Building TF-IDF vectorizer...", end=" ", flush=True)
    t1 = time.time()
    
    # Fit on combined vocabulary for best coverage
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    all_names = np.concatenate([s1_names, s2s3_names])
    vectorizer.fit(all_names)
    del all_names
    gc.collect()
    
    # Transform separately
    s1_tfidf = vectorizer.transform(s1_names)
    s2s3_tfidf = vectorizer.transform(s2s3_names)
    
    vocab_size = len(vectorizer.vocabulary_)
    print(f"vocab={vocab_size:,}, {time.time()-t1:.1f}s")
    
    del vectorizer, s1_names, s2s3_names
    gc.collect()
    
    # Batch cosine similarity
    results = []
    n_batches = (n_s1 + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"    Computing similarities ({n_batches} batches)...", flush=True)
    t2 = time.time()
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, n_s1)
        
        # Compute cosine similarity (sparse × sparse.T)
        batch_tfidf = s1_tfidf[batch_start:batch_end]
        sims = batch_tfidf @ s2s3_tfidf.T  # shape: (batch_size, n_s2s3)
        
        # Convert to CSR for efficient row access
        if not isinstance(sims, sparse.csr_matrix):
            sims = sims.tocsr()
        
        # Extract top-K per S1 entity
        for i in range(batch_end - batch_start):
            s1_id = s1_ids[batch_start + i]
            row = sims.getrow(i)
            
            if row.nnz == 0:
                results.append((s1_id, []))
                continue
            
            data = row.data
            indices = row.indices
            
            # Get top-K by score
            if len(data) <= top_k:
                top_idx = np.arange(len(data))
            else:
                top_idx = np.argpartition(data, -top_k)[-top_k:]
            
            # Sort by score descending
            top_idx = top_idx[np.argsort(-data[top_idx])]
            
            candidates = [
                (s2s3_ids[indices[j]], float(data[j]))
                for j in top_idx
            ]
            results.append((s1_id, candidates))
        
        if (batch_idx + 1) % 20 == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - t2
            pct = 100.0 * (batch_idx + 1) / n_batches
            eta = elapsed / (batch_idx + 1) * (n_batches - batch_idx - 1)
            print(f"      Batch {batch_idx+1}/{n_batches} ({pct:.0f}%) "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)
    
    total_time = time.time() - t0
    total_candidates = sum(len(c) for _, c in results)
    print(f"    Done: {len(results):,} S1 entities, "
          f"{total_candidates:,} candidate pairs, {total_time:.1f}s")
    
    del s1_tfidf, s2s3_tfidf
    gc.collect()
    
    return results


def run_pipeline(s1_df, s2_df, s3_df):
    """
    Run the full blocking pipeline across all countries.
    
    Returns: dict of s1_id → [(s2s3_id, score), ...]
    """
    # Combine S2 and S3
    s2s3_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    countries = sorted(s1_df['country'].unique())
    print(f"\nCountries to process: {countries}")
    
    all_results = {}
    
    for country in countries:
        s1_country = s1_df[s1_df['country'] == country].copy()
        s2s3_country = s2s3_df[s2s3_df['country'] == country].copy()
        
        results = tfidf_block_country(s1_country, s2s3_country, country)
        
        for s1_id, candidates in results:
            all_results[s1_id] = candidates
        
        gc.collect()
    
    # Add missing S1 entities (should not happen, but defensive)
    for s1_id in s1_df['entity_id'].values:
        if s1_id not in all_results:
            all_results[s1_id] = []
    
    print(f"\nPipeline complete: {len(all_results):,} S1 entities processed")
    return all_results


def apply_threshold(all_results, threshold):
    """Apply similarity threshold to get final matches."""
    predictions = {}
    for s1_id, candidates in all_results.items():
        matched = [cid for cid, score in candidates if score >= threshold]
        predictions[s1_id] = matched
    return predictions


# ============================================================
# THRESHOLD TUNING
# ============================================================
def tune_threshold(all_results, ground_truth, s1_ids):
    """Sweep thresholds to find optimal F_0.5."""
    print(f"\n{'='*60}")
    print(f"  THRESHOLD TUNING")
    print(f"{'='*60}")
    
    best_f05 = -1
    best_threshold = 0.5
    results_table = []
    
    for threshold in THRESHOLDS_TO_SWEEP:
        predictions = apply_threshold(all_results, threshold)
        f05, details = compute_macro_f05(predictions, ground_truth, s1_ids)
        
        n_matched = sum(1 for v in predictions.values() if len(v) > 0)
        n_total_preds = sum(len(v) for v in predictions.values())
        
        results_table.append({
            'threshold': threshold,
            'f05': f05,
            'n_matched': n_matched,
            'n_total_preds': n_total_preds,
            'n_perfect': details['n_perfect'],
            'n_zero': details['n_zero'],
        })
        
        marker = " ← BEST" if f05 > best_f05 else ""
        print(f"  τ={threshold:.3f} | F₀.₅={f05:.4f} | "
              f"matched={n_matched:,} | preds={n_total_preds:,} | "
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
            matched_str = ",".join(matched) if matched else ""
            f.write(f"{s1_id}\t{matched_str}\n")
    
    print(f"  Written: {matching_path}")
    
    # candidate_pairs.tsv
    if all_results is not None:
        candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
        with open(candidate_path, 'w', encoding='utf-8') as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in s1_ids:
                candidates = all_results.get(s1_id, [])
                cand_ids = [cid for cid, _ in candidates]
                cand_str = ",".join(cand_ids) if cand_ids else ""
                f.write(f"{s1_id}\t{cand_str}\n")
        
        print(f"  Written: {candidate_path}")
    
    # Summary statistics
    n_matched = sum(1 for s1_id in s1_ids if len(predictions.get(s1_id, [])) > 0)
    n_singleton = len(s1_ids) - n_matched
    n_total_preds = sum(len(predictions.get(s1_id, [])) for s1_id in s1_ids)
    print(f"\n  Summary:")
    print(f"    Total S1 entities: {len(s1_ids):,}")
    print(f"    Matched entities: {n_matched:,} ({100.0*n_matched/len(s1_ids):.1f}%)")
    print(f"    Singletons: {n_singleton:,} ({100.0*n_singleton/len(s1_ids):.1f}%)")
    print(f"    Total predictions: {n_total_preds:,}")
    print(f"    Mean preds/entity: {n_total_preds/len(s1_ids):.2f}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Sprint 0: Baseline Entity Resolution")
    parser.add_argument("--eval-only", action="store_true", 
                        help="Only evaluate on training sample, skip test prediction")
    parser.add_argument("--sample-size", type=int, default=EVAL_SAMPLE,
                        help=f"Training sample size for validation (default: {EVAL_SAMPLE})")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed threshold (skip tuning)")
    args = parser.parse_args()
    
    t_start = time.time()
    
    print("█" * 60)
    print("  SPRINT 0: BASELINE ENTITY RESOLUTION PIPELINE")
    print("█" * 60)
    
    # ---- Load Training Data ----
    print(f"\n--- Loading Training Data ---")
    train_s1 = load_tsv(f"{DATA_ROOT}/train/train_source1.tsv")
    train_s2 = load_tsv(f"{DATA_ROOT}/train/train_source2.tsv")
    train_s3 = load_tsv(f"{DATA_ROOT}/train/train_source3.tsv")
    gt = load_ground_truth(f"{DATA_ROOT}/train/train_ground_truth.tsv")
    
    # ---- Sample for Validation ----
    sample_size = min(args.sample_size, len(train_s1))
    print(f"\n--- Sampling {sample_size:,} S1 entities for validation ---")
    
    np.random.seed(42)
    # Stratified sample by country
    sample_indices = []
    for country in train_s1['country'].unique():
        country_indices = train_s1[train_s1['country'] == country].index.values
        n_sample = int(sample_size * len(country_indices) / len(train_s1))
        n_sample = max(1, min(n_sample, len(country_indices)))
        chosen = np.random.choice(country_indices, size=n_sample, replace=False)
        sample_indices.extend(chosen)
    
    sample_s1 = train_s1.loc[sample_indices].reset_index(drop=True)
    sample_s1_ids = sample_s1['entity_id'].values
    print(f"  Sampled: {len(sample_s1):,} entities")
    for c in sample_s1['country'].unique():
        n = (sample_s1['country'] == c).sum()
        print(f"    {c}: {n:,}")
    
    # ---- Run Pipeline on Training Sample ----
    print(f"\n{'='*60}")
    print(f"  PHASE 1: TRAINING VALIDATION")
    print(f"{'='*60}")
    
    sample_results = run_pipeline(sample_s1, train_s2, train_s3)
    
    # ---- Threshold Tuning ----
    if args.threshold is not None:
        best_threshold = args.threshold
        predictions = apply_threshold(sample_results, best_threshold)
        f05, details = compute_macro_f05(predictions, gt, sample_s1_ids)
        print(f"\n  Fixed threshold: {best_threshold:.3f} → F₀.₅ = {f05:.4f}")
        best_f05 = f05
    else:
        best_threshold, best_f05 = tune_threshold(sample_results, gt, sample_s1_ids)
    
    if args.eval_only:
        print(f"\n  [eval-only mode] Skipping test prediction.")
        print(f"\n  Total time: {time.time()-t_start:.0f}s")
        return
    
    # ---- Free training data ----
    del train_s1, train_s2, train_s3, gt, sample_s1, sample_results
    gc.collect()
    
    # ---- Load Test Data ----
    print(f"\n{'='*60}")
    print(f"  PHASE 2: TEST PREDICTION")
    print(f"{'='*60}")
    
    print(f"\n--- Loading Test Data ---")
    test_s1 = load_tsv(f"{DATA_ROOT}/test/test_source1.tsv")
    test_s2 = load_tsv(f"{DATA_ROOT}/test/test_source2.tsv")
    test_s3 = load_tsv(f"{DATA_ROOT}/test/test_source3.tsv")
    
    test_s1_ids = test_s1['entity_id'].values
    
    # ---- Run Pipeline on Test Data ----
    test_results = run_pipeline(test_s1, test_s2, test_s3)
    
    # ---- Apply Threshold ----
    test_predictions = apply_threshold(test_results, best_threshold)
    
    # ---- Generate Output ----
    print(f"\n--- Generating Output ---")
    generate_output(test_predictions, test_s1_ids, OUTPUT_DIR, all_results=test_results)
    
    # ---- Summary ----
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  SPRINT 0 COMPLETE")
    print(f"{'='*60}")
    print(f"  Validation F₀.₅: {best_f05:.4f} (threshold={best_threshold:.3f})")
    print(f"  Output: {OUTPUT_DIR}/matching_results.tsv")
    print(f"  Output: {OUTPUT_DIR}/candidate_pairs.tsv")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
