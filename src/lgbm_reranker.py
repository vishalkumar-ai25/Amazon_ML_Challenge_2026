#!/usr/bin/env python3
"""
Amazon ML Challenge 2026: LightGBM Pairwise Reranker
=====================================================
Replaces heuristic threshold (τ=0.80) with a trained binary classifier.
This is THE single highest-impact improvement for entity resolution.

Flow:
  1. Sample training data (stratified by country)
  2. Generate blocking candidates using two-channel blocking (same as production)
  3. Label candidates: positive (in GT), negative (in candidates but not GT)
  4. Extract 15+ pairwise similarity features
  5. Train LightGBM binary classifier with 5-fold CV
  6. Report per-fold F_0.5 and feature importances
  7. Save model for production inference

Target: Replace τ=0.80 threshold → LightGBM probability + M2O
Expected impact: 0.65 → 0.85+ (conservative estimate)
"""

import os
import sys
import time
import gc
import re
import json
import pickle
import hashlib
import unicodedata
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from scipy import sparse
from difflib import SequenceMatcher

# ── Optional imports ────────────────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: lightgbm not installed. Install with: pip install lightgbm")

# ── Configuration ───────────────────────────────────────────────
DATA_DIR = "dataset/student_resource/dataset/train"
OUTPUT_DIR = "output"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_SIZE_PER_COUNTRY = 15000  # Quick validation; scale to 50K for production
TOP_K_CHANNEL_A = 30
TOP_K_CHANNEL_B = 30
MAX_UNION_CANDIDATES = 50
BATCH_SIZE = 10000
SEED = 42
N_FOLDS = 5

# ── Text Normalization ─────────────────────────────────────────
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
    """Basic normalization matching the production pipeline."""
    if not text or pd.isna(text) or text == "nan":
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'^(--|\<\<|\>\>|\.\.)[\s]*', '', text)
    text = re.sub(r'\.(com|net|org|co|io)\b', '', text)
    text = re.sub(r"[^\w\s]", ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_name_deep(text):
    """Aggressive name normalization: strip legal suffixes, DBA, etc."""
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
    """Extract postal/zip code from address string."""
    if not address or pd.isna(address) or address == "nan":
        return ""
    address = str(address)
    # US zip: 5 digits or 5+4
    m = re.search(r'\b(\d{5})(?:-\d{4})?\b', address)
    if m:
        return m.group(1)
    # India pin: 6 digits
    m = re.search(r'\b(\d{6})\b', address)
    if m:
        return m.group(1)
    # France: 5 digits (typically starts with specific dept codes)
    m = re.search(r'\b(\d{5})\b', address)
    if m:
        return m.group(1)
    return ""


def extract_house_number(address):
    """Extract leading house/building number from address."""
    if not address or pd.isna(address) or address == "nan":
        return ""
    address = str(address).strip()
    m = re.match(r'^(\d+[\-/]?\d*)\s', address)
    if m:
        return m.group(1)
    return ""


def get_word_set(text):
    """Get set of non-trivial words (len > 1)."""
    if not text:
        return set()
    return {w for w in text.split() if len(w) > 1}


def unicode_script(char):
    """Get Unicode script category of a character."""
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
    """Detect dominant script in text."""
    if not text:
        return 'empty'
    scripts = Counter()
    for c in text:
        if c.isalpha():
            scripts[unicode_script(c)] += 1
    if not scripts:
        return 'empty'
    return scripts.most_common(1)[0][0]


# ── Pairwise Feature Extraction ────────────────────────────────
def compute_pairwise_features(s1_name, s1_addr, s2_name, s2_addr, tfidf_score):
    """
    Compute 15+ pairwise similarity features between an S1 entity and a candidate.
    These features capture different aspects of name/address similarity.
    """
    features = {}
    
    # 0. TF-IDF composite score (from blocking)
    features['tfidf_composite'] = tfidf_score
    
    # ── Name Features ──────────────────────────────────────────
    name1 = normalize_name_deep(s1_name)
    name2 = normalize_name_deep(s2_name)
    name1_basic = normalize_text(s1_name)
    name2_basic = normalize_text(s2_name)
    
    # 1. Name exact match (after normalization)
    features['name_exact'] = 1.0 if name1 == name2 and name1 != '' else 0.0
    
    # 2. Name SequenceMatcher ratio (character-level edit distance)
    features['name_seqmatch'] = SequenceMatcher(None, name1, name2).ratio() if name1 and name2 else 0.0
    
    # 3. Name Jaccard (word-level)
    w1 = get_word_set(name1)
    w2 = get_word_set(name2)
    if w1 and w2:
        features['name_jaccard'] = len(w1 & w2) / len(w1 | w2)
    else:
        features['name_jaccard'] = 0.0
    
    # 4. Name containment (is one name a subset of the other?)
    if w1 and w2:
        features['name_containment'] = len(w1 & w2) / min(len(w1), len(w2))
    else:
        features['name_containment'] = 0.0
    
    # 5. Name length ratio
    if name1 and name2:
        features['name_len_ratio'] = min(len(name1), len(name2)) / max(len(name1), len(name2))
    else:
        features['name_len_ratio'] = 0.0
    
    # 6. Name word count difference
    features['name_word_diff'] = abs(len(w1) - len(w2))
    
    # 7. Name first-word match (business names often share first meaningful word)
    words1 = name1.split() if name1 else []
    words2 = name2.split() if name2 else []
    features['name_first_word_match'] = 1.0 if words1 and words2 and words1[0] == words2[0] and len(words1[0]) > 1 else 0.0
    
    # 8. Script mismatch (cross-script matching is hard)
    script1 = detect_script(s1_name) if s1_name else 'empty'
    script2 = detect_script(s2_name) if s2_name else 'empty'
    features['script_mismatch'] = 0.0 if script1 == script2 else 1.0
    
    # ── Address Features ───────────────────────────────────────
    addr1 = normalize_text(s1_addr)
    addr2 = normalize_text(s2_addr)
    
    # 9. Address exact match
    features['addr_exact'] = 1.0 if addr1 == addr2 and addr1 != '' else 0.0
    
    # 10. Address Jaccard (word-level)
    aw1 = get_word_set(addr1)
    aw2 = get_word_set(addr2)
    if aw1 and aw2:
        features['addr_jaccard'] = len(aw1 & aw2) / len(aw1 | aw2)
    else:
        features['addr_jaccard'] = 0.0
    
    # 11. Address SequenceMatcher ratio
    features['addr_seqmatch'] = SequenceMatcher(None, addr1, addr2).ratio() if addr1 and addr2 else 0.0
    
    # 12. Postal code match / mismatch / missing
    pc1 = extract_postal_code(s1_addr)
    pc2 = extract_postal_code(s2_addr)
    if pc1 and pc2:
        features['postal_match'] = 1.0 if pc1 == pc2 else -1.0  # -1 = conflict
    else:
        features['postal_match'] = 0.0  # missing
    
    # 13. House number match
    hn1 = extract_house_number(s1_addr)
    hn2 = extract_house_number(s2_addr)
    if hn1 and hn2:
        features['house_num_match'] = 1.0 if hn1 == hn2 else -1.0
    else:
        features['house_num_match'] = 0.0
    
    # 14. Both addresses missing
    features['both_addr_missing'] = 1.0 if not addr1 and not addr2 else 0.0
    
    # 15. One address missing
    features['one_addr_missing'] = 1.0 if bool(addr1) != bool(addr2) else 0.0
    
    # 16. Name token overlap with address (name appears in address = possible FP)
    if w1 and aw2:
        features['name_in_addr_overlap'] = len(w1 & aw2) / len(w1)
    else:
        features['name_in_addr_overlap'] = 0.0
    
    # 17. Address containment
    if aw1 and aw2:
        features['addr_containment'] = len(aw1 & aw2) / min(len(aw1), len(aw2))
    else:
        features['addr_containment'] = 0.0
    
    return features


# ── Feature names (must match compute_pairwise_features output) ──
FEATURE_NAMES = [
    'tfidf_composite', 'name_exact', 'name_seqmatch', 'name_jaccard',
    'name_containment', 'name_len_ratio', 'name_word_diff', 
    'name_first_word_match', 'script_mismatch',
    'addr_exact', 'addr_jaccard', 'addr_seqmatch',
    'postal_match', 'house_num_match', 'both_addr_missing',
    'one_addr_missing', 'name_in_addr_overlap', 'addr_containment'
]


def build_training_pairs(sample_size=SAMPLE_SIZE_PER_COUNTRY):
    """
    Build labeled training pairs:
    1. Sample S1 entities (stratified by country)
    2. Run two-channel blocking to get candidates
    3. Label: positive if candidate is in GT, negative otherwise
    4. Extract pairwise features
    
    Returns: X (features), y (labels), metadata
    """
    print("=" * 80)
    print("  BUILDING TRAINING PAIRS FOR LIGHTGBM RERANKER")
    print("=" * 80)
    t0 = time.time()
    
    # 1. Load training data
    print("Loading training data...", end=" ", flush=True)
    s1 = pd.read_csv(os.path.join(DATA_DIR, "train_source1.tsv"), sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(os.path.join(DATA_DIR, "train_source2.tsv"), sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(os.path.join(DATA_DIR, "train_source3.tsv"), sep="\t", dtype=str).fillna("")
    gt = pd.read_csv(os.path.join(DATA_DIR, "train_ground_truth.tsv"), sep="\t", dtype=str).fillna("")
    print(f"Done. S1={len(s1):,} S2={len(s2):,} S3={len(s3):,} GT={len(gt):,}")
    
    # Build GT lookup: S1_id -> set of matched S2/S3 ids
    print("Building GT lookup...", end=" ", flush=True)
    gt_map = {}
    for _, row in gt.iterrows():
        sid = row['source1_entity_id']
        mids = row['matched_entity_ids']
        gt_map[sid] = set(mids.split(',')) if mids else set()
    print(f"Done ({len(gt_map):,} entries)")
    
    # 2. Sample S1 entities per country
    print(f"\nSampling {sample_size:,} S1 entities per country...")
    sampled_s1_list = []
    for country in ['US', 'India']:
        s1_c = s1[s1['country'] == country]
        n_sample = min(sample_size, len(s1_c))
        sampled = s1_c.sample(n=n_sample, random_state=SEED)
        sampled_s1_list.append(sampled)
        print(f"  {country}: {n_sample:,} sampled from {len(s1_c):,}")
    
    sampled_s1 = pd.concat(sampled_s1_list, ignore_index=True)
    sampled_countries = sampled_s1['country'].values
    print(f"  Total sampled: {len(sampled_s1):,}")
    
    # 3. Merge S2 + S3 as corpus
    s2s3 = pd.concat([s2, s3], ignore_index=True).reset_index(drop=True)
    s2s3_lookup = dict(zip(s2s3['entity_id'], range(len(s2s3))))
    del s2, s3
    gc.collect()
    
    # 4. Run two-channel blocking per country to get candidates
    all_features = []
    all_labels = []
    all_meta = []  # (s1_id, s2s3_id, country)
    
    for country in ['US', 'India']:
        print(f"\n--- Processing {country} ---")
        s1_c = sampled_s1[sampled_s1['country'] == country].reset_index(drop=True)
        
        # Filter S2+S3 to country
        s2s3_c = s2s3[s2s3['country'] == country].reset_index(drop=True)
        n_s1 = len(s1_c)
        n_s2s3 = len(s2s3_c)
        print(f"  S1: {n_s1:,} | S2+S3: {n_s2s3:,}")
        
        # Prepare text
        s1_full = (s1_c['business_name'] + " " + s1_c['business_name'] + " " + s1_c['business_address']).apply(normalize_text).values
        s2s3_full = (s2s3_c['business_name'] + " " + s2s3_c['business_name'] + " " + s2s3_c['business_address']).apply(normalize_text).values
        
        s1_addr = s1_c['business_address'].apply(normalize_text).values
        s2s3_addr = s2s3_c['business_address'].apply(normalize_text).values
        
        # Channel A: TF-IDF
        print(f"  Building TF-IDF (Channel A)...", end=" ", flush=True)
        t_ch = time.time()
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
        print(f"Done ({time.time()-t_ch:.1f}s)")
        
        # Channel B: Address Jaccard
        print(f"  Building Address Jaccard (Channel B)...", end=" ", flush=True)
        t_ch = time.time()
        vec_b = CountVectorizer(
            binary=True, analyzer='word', token_pattern=r'(?u)\b\w+\b',
            min_df=2, max_df=0.02, max_features=60000
        )
        fit_sample = np.concatenate([s1_addr[:min(100000, n_s1)], s2s3_addr[:min(500000, n_s2s3)]])
        vec_b.fit(fit_sample)
        del fit_sample
        
        A = vec_b.transform(s1_addr)
        B = vec_b.transform(s2s3_addr)
        B_T = B.T.tocsc()
        len_A = np.diff(A.indptr)
        len_B = np.diff(B.indptr)
        del vec_b, B
        gc.collect()
        print(f"Done ({time.time()-t_ch:.1f}s)")
        
        # Blocking + Feature Extraction
        s1_ids = s1_c['entity_id'].values
        s2s3_ids = s2s3_c['entity_id'].values
        s2s3_id_to_idx = {eid: idx for idx, eid in enumerate(s2s3_ids)}  # O(1) lookup
        s1_names = s1_c['business_name'].values
        s1_addrs_raw = s1_c['business_address'].values
        s2s3_names = s2s3_c['business_name'].values
        s2s3_addrs_raw = s2s3_c['business_address'].values
        
        n_batches = int(np.ceil(n_s1 / BATCH_SIZE))
        country_pos = 0
        country_neg = 0
        
        print(f"  Extracting features from blocking candidates ({n_batches} batches)...")
        t_feat = time.time()
        
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
            
            for i in range(b_end - b_start):
                global_i = b_start + i
                sid = s1_ids[global_i]
                gt_set = gt_map.get(sid, set())
                
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
                for cid, s in top_a.items():
                    union_cands[cid] = s
                for cid, s in top_b.items():
                    if cid in union_cands:
                        union_cands[cid] = max(union_cands[cid], s)
                    else:
                        union_cands[cid] = s
                
                sorted_union = sorted(union_cands.items(), key=lambda x: -x[1])[:MAX_UNION_CANDIDATES]
                
                # Extract features for each candidate
                for cid, tfidf_score in sorted_union:
                    # O(1) lookup for candidate index
                    cid_idx = s2s3_id_to_idx.get(cid)
                    if cid_idx is None:
                        continue
                    
                    feats = compute_pairwise_features(
                        s1_names[global_i], s1_addrs_raw[global_i],
                        s2s3_names[cid_idx], s2s3_addrs_raw[cid_idx],
                        tfidf_score
                    )
                    
                    label = 1 if cid in gt_set else 0
                    
                    all_features.append([feats[fn] for fn in FEATURE_NAMES])
                    all_labels.append(label)
                    all_meta.append((sid, cid, country))
                    
                    if label == 1:
                        country_pos += 1
                    else:
                        country_neg += 1
            
            if (b + 1) % 5 == 0 or b == n_batches - 1:
                elapsed = time.time() - t_feat
                pct = 100.0 * (b + 1) / n_batches
                print(f"    Batch {b+1}/{n_batches} ({pct:.0f}%) | Pos: {country_pos:,} | Neg: {country_neg:,} | {elapsed:.0f}s")
        
        del s1_mat_a, s2s3_mat_a_T, A, B_T, len_A, len_B
        del s1_full, s2s3_full, s1_addr, s2s3_addr
        del s1_names, s1_addrs_raw, s2s3_names, s2s3_addrs_raw
        gc.collect()
        
        print(f"  [{country}] Positive pairs: {country_pos:,} | Negative pairs: {country_neg:,}")
    
    # Convert to arrays
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    meta = all_meta
    
    elapsed = time.time() - t0
    print(f"\nTraining data built in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Total pairs: {len(X):,} | Positive: {y.sum():,} ({100*y.mean():.1f}%) | Negative: {(1-y).sum():,}")
    
    return X, y, meta


def train_lgbm(X, y, meta):
    """Train LightGBM binary classifier with 5-fold CV."""
    if not HAS_LGB:
        print("ERROR: lightgbm not installed!")
        return None
    
    print("\n" + "=" * 80)
    print("  TRAINING LIGHTGBM RERANKER (5-Fold CV)")
    print("=" * 80)
    
    # LightGBM parameters tuned for entity resolution
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 7,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': (1 - y.mean()) / y.mean(),  # Handle class imbalance
        'verbose': -1,
        'random_state': SEED,
        'n_jobs': -1,
    }
    
    # Extract country for stratification
    countries = np.array([m[2] for m in meta])
    strat_key = np.array([f"{l}_{c}" for l, c in zip(y, countries)])
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    fold_metrics = []
    fold_models = []
    all_preds = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, strat_key)):
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"  Train: {len(X_train):,} (pos={y_train.sum():,}) | Val: {len(X_val):,} (pos={y_val.sum():,})")
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_data)
        
        callbacks = [
            lgb.log_evaluation(100),
            lgb.early_stopping(50),
        ]
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=callbacks
        )
        
        # Predict
        preds = model.predict(X_val)
        all_preds[val_idx] = preds
        
        # Compute F_0.5 at various thresholds
        best_f05 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.95, 0.025):
            pred_labels = (preds >= thresh).astype(int)
            tp = ((pred_labels == 1) & (y_val == 1)).sum()
            fp = ((pred_labels == 1) & (y_val == 0)).sum()
            fn = ((pred_labels == 0) & (y_val == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            beta = 0.5
            if precision + recall > 0:
                f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
            else:
                f05 = 0
            
            if f05 > best_f05:
                best_f05 = f05
                best_thresh = thresh
        
        # Report metrics at best threshold
        pred_labels = (preds >= best_thresh).astype(int)
        tp = ((pred_labels == 1) & (y_val == 1)).sum()
        fp = ((pred_labels == 1) & (y_val == 0)).sum()
        fn = ((pred_labels == 0) & (y_val == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"  Best τ={best_thresh:.3f} → F₀.₅={best_f05:.4f} (P={precision:.4f}, R={recall:.4f})")
        print(f"  TP={tp:,} FP={fp:,} FN={fn:,}")
        
        fold_metrics.append({
            'fold': fold+1,
            'f05': best_f05,
            'precision': precision,
            'recall': recall,
            'best_threshold': best_thresh,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
            'n_trees': model.best_iteration if model.best_iteration else model.num_trees()
        })
        fold_models.append(model)
    
    # Summary
    print("\n" + "=" * 80)
    print("  CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    avg_f05 = np.mean([m['f05'] for m in fold_metrics])
    avg_precision = np.mean([m['precision'] for m in fold_metrics])
    avg_recall = np.mean([m['recall'] for m in fold_metrics])
    avg_thresh = np.mean([m['best_threshold'] for m in fold_metrics])
    
    for m in fold_metrics:
        print(f"  Fold {m['fold']}: F₀.₅={m['f05']:.4f} | P={m['precision']:.4f} | R={m['recall']:.4f} | τ={m['best_threshold']:.3f} | Trees={m['n_trees']}")
    print(f"\n  MEAN:  F₀.₅={avg_f05:.4f} | P={avg_precision:.4f} | R={avg_recall:.4f} | τ={avg_thresh:.3f}")
    
    # Feature importance
    print("\n  Feature Importances (gain):")
    imp = fold_models[0].feature_importance(importance_type='gain')
    sorted_idx = np.argsort(-imp)
    for idx in sorted_idx:
        print(f"    {FEATURE_NAMES[idx]:25s}: {imp[idx]:,.0f}")
    
    # Save best model (use full training for final model)
    best_fold_idx = np.argmax([m['f05'] for m in fold_metrics])
    best_model = fold_models[best_fold_idx]
    
    model_path = os.path.join(MODEL_DIR, "lgbm_reranker.txt")
    best_model.save_model(model_path)
    print(f"\n  Best model (fold {best_fold_idx+1}) saved to {model_path}")
    
    # Save metadata
    meta_path = os.path.join(MODEL_DIR, "lgbm_reranker_meta.json")
    with open(meta_path, 'w') as f:
        json.dump({
            'feature_names': FEATURE_NAMES,
            'fold_metrics': fold_metrics,
            'avg_f05': float(avg_f05),
            'avg_precision': float(avg_precision),
            'avg_recall': float(avg_recall),
            'best_threshold': float(avg_thresh),
            'params': params,
            'sample_size_per_country': SAMPLE_SIZE_PER_COUNTRY,
        }, f, indent=2, default=str)
    print(f"  Metadata saved to {meta_path}")
    
    # Train final model on ALL data (for production)
    print("\n  Training final model on ALL data...")
    final_train = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)
    final_model = lgb.train(
        {**params, 'verbose': -1},
        final_train,
        num_boost_round=int(np.mean([m['n_trees'] for m in fold_metrics])),
    )
    final_path = os.path.join(MODEL_DIR, "lgbm_reranker_final.txt")
    final_model.save_model(final_path)
    print(f"  Final model saved to {final_path}")
    
    return final_model, avg_thresh, fold_metrics


def main():
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " AMAZON ML CHALLENGE 2026: LIGHTGBM PAIRWISE RERANKER ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    t_total = time.time()
    
    # Step 1: Build training pairs
    X, y, meta = build_training_pairs()
    
    # Step 2: Train LightGBM
    model, threshold, metrics = train_lgbm(X, y, meta)
    
    elapsed = time.time() - t_total
    print(f"\n{'='*80}")
    print(f"  TOTAL TIME: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
