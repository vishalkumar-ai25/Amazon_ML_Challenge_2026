#!/usr/bin/env python3
"""
STEP 2 & STEP 3: Diagnostic Scoping & Semantic Embedding Validation
Evaluated on the exact 30k stratified sample (seed=42).

1. Reproduces two-channel blocking (TF-IDF + Address Jaccard) with top-100 retention.
2. Identifies target-set entities (the ~4.26% residual misses in top-30).
3. Diagnostic analysis & manual inspection of 30 misses.
4. Evaluates semantic embedding signal (paraphrase-multilingual-MiniLM-L12-v2 & multilingual-e5-small)
   on the target set.
5. Computes Pair Completeness and Macro F_0.5 before & after M2O assignment.
"""

import os
import sys
import time
import gc
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from feature_semantic_embedding import get_embedding_scores, SemanticEmbedder

DATA_ROOT = "dataset/student_resource/dataset"
CACHE_PATH = "output/validation_candidates_30k_top100.pkl"

def normalize_text(text):
    if not text or pd.isna(text) or text == "nan":
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'^(--|<<|>>|\.\.)\s*', '', text)
    text = re.sub(r'\.(com|net|org|co|io)\b', '', text)
    text = re.sub(r"[^\w\s]", ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).strip()

def compute_f05_macro(true_dict, pred_dict, all_ids):
    scores = []
    for sid in all_ids:
        t = true_dict.get(sid, set())
        p = pred_dict.get(sid, set())
        if len(t) == 0 and len(p) == 0:
            scores.append(1.0)
            continue
        if len(t) == 0 or len(p) == 0:
            scores.append(0.0)
            continue
        tp = len(t & p)
        if tp == 0:
            scores.append(0.0)
            continue
        prec = tp / len(p)
        rec = tp / len(t)
        scores.append(1.25 * prec * rec / (0.25 * prec + rec))
    return float(np.mean(scores))

def evaluate_with_m2o(sample_ids, cand_dict, gt_dict, tau):
    """
    Applies global greedy one-to-one assignment for S2/S3 IDs at threshold tau.
    """
    s1_list = []
    cid_list = []
    score_list = []
    for sid in sample_ids:
        for cid, s in cand_dict[sid].items():
            if s >= tau:
                s1_list.append(sid)
                cid_list.append(cid)
                score_list.append(s)
                
    if not score_list:
        pred_dict = {sid: set() for sid in sample_ids}
        return compute_f05_macro(gt_dict, pred_dict, sample_ids)
        
    scores_arr = np.array(score_list, dtype=np.float32)
    sort_order = np.argsort(-scores_arr)
    
    assigned_cands = set()
    pred_dict = {sid: set() for sid in sample_ids}
    
    for idx in sort_order:
        cid = cid_list[idx]
        if cid not in assigned_cands:
            assigned_cands.add(cid)
            pred_dict[s1_list[idx]].add(cid)
            
    return compute_f05_macro(gt_dict, pred_dict, sample_ids)

def main():
    print("=" * 80)
    print("  AMAZON ML CHALLENGE 2026 — STEP 2 & 3: SEMANTIC EMBEDDING MODULE")
    print("=" * 80)
    os.makedirs("output", exist_ok=True)
    t_start = time.time()
    
    # 1. Load train S1 and sample identically (seed=42)
    print("\n[1/6] Loading 30k stratified training sample (seed=42)...", end=" ", flush=True)
    s1 = pd.read_csv(f"{DATA_ROOT}/train/train_source1.tsv", sep="\t", dtype=str).fillna("")
    np.random.seed(42)
    sample_indices = []
    sample_size = 30000
    for country in s1['country'].unique():
        cidx = s1[s1['country'] == country].index.values
        n = max(1, int(sample_size * len(cidx) / len(s1)))
        sample_indices.extend(np.random.choice(cidx, size=min(n, len(cidx)), replace=False))
        
    sample_s1 = s1.loc[sample_indices].reset_index(drop=True)
    sample_ids = sample_s1['entity_id'].values
    sample_s1_map = {row['entity_id']: row for _, row in sample_s1.iterrows()}
    print(f"Done ({len(sample_s1):,} S1 entities)")
    for c in sorted(sample_s1['country'].unique()):
        print(f"       Country {c}: {int((sample_s1['country']==c).sum()):,}")
        
    # 2. Load ground truth
    print("\n[2/6] Loading ground truth...", end=" ", flush=True)
    gt = pd.read_csv(f"{DATA_ROOT}/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    gt_map = dict(zip(gt['source1_entity_id'], gt['matched_entity_ids']))
    gt_dict = {}
    for sid in sample_ids:
        mids = str(gt_map.get(sid, "")).strip()
        gt_dict[sid] = set(mids.split(",")) if mids else set()
    total_true = sum(len(v) for v in gt_dict.values())
    print(f"Done (Total true match links: {total_true:,})")
    
    # 3. Load S2 and S3 pools
    print("\n[3/6] Loading S2 and S3 pools...", end=" ", flush=True)
    s2 = pd.read_csv(f"{DATA_ROOT}/train/train_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(f"{DATA_ROOT}/train/train_source3.tsv", sep="\t", dtype=str).fillna("")
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    s2s3_map = dict(zip(s2s3['entity_id'], s2s3.to_dict('records')))
    print(f"Done ({len(s2s3):,} records)")
    del s1, s2, s3, gt, gt_map
    gc.collect()
    
    # 4. Two-Channel Blocking (with top-100 retention and caching)
    if os.path.exists(CACHE_PATH):
        print(f"\n[4/6] Loading cached candidate sets from {CACHE_PATH}...", end=" ", flush=True)
        with open(CACHE_PATH, "rb") as f:
            cache_data = pickle.load(f)
        cand_chA_100 = cache_data["cand_chA_100"]
        cand_chB_100 = cache_data["cand_chB_100"]
        print("Done!")
    else:
        print("\n[4/6] Executing Two-Channel Blocking (computing top-100 per channel)...")
        cand_chA_100 = {sid: {} for sid in sample_ids}
        cand_chB_100 = {sid: {} for sid in sample_ids}
        
        for country in sorted(sample_s1['country'].unique()):
            print(f"\n--- Country: {country} ---")
            t_country = time.time()
            s1_c = sample_s1[sample_s1['country'] == country].reset_index(drop=True)
            s2s3_c = s2s3[s2s3['country'] == country].reset_index(drop=True)
            n_s1 = len(s1_c)
            n_s2s3 = len(s2s3_c)
            print(f"  S1 queries: {n_s1:,} | S2/S3 corpus: {n_s2s3:,}")
            
            s1_ids_c = s1_c['entity_id'].values
            s2s3_ids_c = s2s3_c['entity_id'].values
            
            # Channel A: TF-IDF on 2*name + address
            print("  [Channel A] Building composite full-text TF-IDF...", end=" ", flush=True)
            t_ca = time.time()
            s1_full = (s1_c['business_name'] + " " + s1_c['business_name'] + " " + s1_c['business_address']).apply(normalize_text).values
            s2s3_full = (s2s3_c['business_name'] + " " + s2s3_c['business_name'] + " " + s2s3_c['business_address']).apply(normalize_text).values
            
            vec_a = TfidfVectorizer(
                analyzer='word',
                max_features=150000,
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
            print(f"Done in {time.time()-t_ca:.1f}s")
            
            # Channel B: Address Jaccard
            print("  [Channel B] Building address presence matrices...", end=" ", flush=True)
            t_cb = time.time()
            s1_addr = s1_c['business_address'].apply(normalize_text).values
            s2s3_addr = s2s3_c['business_address'].apply(normalize_text).values
            
            vec_b = CountVectorizer(
                binary=True,
                analyzer='word',
                token_pattern=r'(?u)\b\w+\b',
                min_df=2,
                max_df=0.02,
                max_features=100000
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
            print(f"Done in {time.time()-t_cb:.1f}s")
            
            # Similarities in batches
            BATCH_SIZE = 10000
            n_batches = (n_s1 + BATCH_SIZE - 1) // BATCH_SIZE
            for b in range(n_batches):
                b_start = b * BATCH_SIZE
                b_end = min(b_start + BATCH_SIZE, n_s1)
                
                b_sims_a = s1_mat_a[b_start:b_end] @ s2s3_mat_a_T
                if not isinstance(b_sims_a, sparse.csr_matrix):
                    b_sims_a = b_sims_a.tocsr()
                    
                b_inter_b = A[b_start:b_end] @ B_T
                if not isinstance(b_inter_b, sparse.csr_matrix):
                    b_inter_b = b_inter_b.tocsr()
                    
                for i in range(b_end - b_start):
                    global_i = b_start + i
                    sid = s1_ids_c[global_i]
                    
                    # Channel A top-100
                    p0_a, p1_a = b_sims_a.indptr[i], b_sims_a.indptr[i+1]
                    if p0_a < p1_a:
                        data_a = b_sims_a.data[p0_a:p1_a]
                        indices_a = b_sims_a.indices[p0_a:p1_a]
                        k_a = min(100, len(data_a))
                        top_idx_a = np.argpartition(data_a, -k_a)[-k_a:] if len(data_a) > 100 else np.arange(len(data_a))
                        top_idx_a = top_idx_a[np.argsort(-data_a[top_idx_a])]
                        for idx in top_idx_a:
                            cid = s2s3_ids_c[indices_a[idx]]
                            cand_chA_100[sid][cid] = float(data_a[idx])
                            
                    # Channel B top-100
                    p0_b, p1_b = b_inter_b.indptr[i], b_inter_b.indptr[i+1]
                    if p0_b < p1_b and len_A[global_i] > 0:
                        inter_data = b_inter_b.data[p0_b:p1_b]
                        inter_indices = b_inter_b.indices[p0_b:p1_b]
                        denoms = len_A[global_i] + len_B[inter_indices] - inter_data
                        jaccard_scores = inter_data / denoms
                        k_b = min(100, len(jaccard_scores))
                        top_idx_b = np.argpartition(jaccard_scores, -k_b)[-k_b:] if len(jaccard_scores) > 100 else np.arange(len(jaccard_scores))
                        top_idx_b = top_idx_b[np.argsort(-jaccard_scores[top_idx_b])]
                        for idx in top_idx_b:
                            cid = s2s3_ids_c[inter_indices[idx]]
                            cand_chB_100[sid][cid] = float(jaccard_scores[idx])
                            
            del s1_mat_a, s2s3_mat_a_T, A, B_T, len_A, len_B, s1_ids_c, s2s3_ids_c
            gc.collect()
            print(f"  Finished {country} in {time.time()-t_country:.1f}s")
            
        print(f"Caching candidates to {CACHE_PATH}...")
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"cand_chA_100": cand_chA_100, "cand_chB_100": cand_chB_100}, f)
            
    # Extract top-30 per channel and top-30 union
    cand_chA_30 = {}
    cand_chB_30 = {}
    cand_union_30 = {}
    cand_union_100 = {}
    
    for sid in sample_ids:
        # top 30 chA
        items_a = sorted(cand_chA_100[sid].items(), key=lambda x: -x[1])[:30]
        cand_chA_30[sid] = dict(items_a)
        
        # top 30 chB
        items_b = sorted(cand_chB_100[sid].items(), key=lambda x: -x[1])[:30]
        cand_chB_30[sid] = dict(items_b)
        
        # union top 30
        u30 = {}
        for cid, s in items_a:
            u30[cid] = s
        for cid, s in items_b:
            u30[cid] = max(u30.get(cid, 0.0), s)
        cand_union_30[sid] = u30
        
        # union top 100
        u100 = {}
        for cid, s in cand_chA_100[sid].items():
            u100[cid] = s
        for cid, s in cand_chB_100[sid].items():
            u100[cid] = max(u100.get(cid, 0.0), s)
        cand_union_100[sid] = u100
        
    # Baseline validation check
    found_A = sum(len(gt_dict[sid] & set(cand_chA_30[sid].keys())) for sid in sample_ids)
    found_B = sum(len(gt_dict[sid] & set(cand_chB_30[sid].keys())) for sid in sample_ids)
    found_union_30 = sum(len(gt_dict[sid] & set(cand_union_30[sid].keys())) for sid in sample_ids)
    found_union_100 = sum(len(gt_dict[sid] & set(cand_union_100[sid].keys())) for sid in sample_ids)
    
    pc_union_30 = found_union_30 / total_true
    pc_union_100 = found_union_100 / total_true
    
    print("\n" + "=" * 80)
    print("  BASELINE REPRODUCTION CHECK")
    print("=" * 80)
    print(f"  Channel A top-30 PC:       {found_A / total_true * 100:.2f}% ({found_A:,} / {total_true:,})")
    print(f"  Channel B top-30 PC:       {found_B / total_true * 100:.2f}% ({found_B:,} / {total_true:,})")
    print(f"  Two-Channel Union top-30:  {pc_union_30 * 100:.2f}% ({found_union_30:,} / {total_true:,})")
    print(f"  Two-Channel Union top-100: {pc_union_100 * 100:.2f}% ({found_union_100:,} / {total_true:,})")
    
    baseline_f05_080 = evaluate_with_m2o(sample_ids, cand_union_30, gt_dict, 0.800)
    print(f"  Baseline Macro F_0.5 at tau=0.800 (with M2O): {baseline_f05_080:.4f}")
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 2 — SCOPE THE HARD-CASE SUBSET
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  STEP 2: SCOPE THE HARD-CASE SUBSET")
    print("=" * 80)
    
    # Identify S1 entities with at least one true match missed in top-30 union
    target_s1_misses = {}  # sid -> list of missed true cids
    target_by_country = {"India": 0, "US": 0}
    missed_link_counts_by_country = {"India": 0, "US": 0}
    
    for sid in sample_ids:
        true_set = gt_dict[sid]
        found_in_30 = set(cand_union_30[sid].keys()) & true_set
        missed = true_set - found_in_30
        if missed:
            target_s1_misses[sid] = list(missed)
            c = sample_s1_map[sid]['country']
            target_by_country[c] += 1
            missed_link_counts_by_country[c] += len(missed)
            
    total_target_s1 = len(target_s1_misses)
    total_missed_links = sum(len(m) for m in target_s1_misses.values())
    
    print(f"Total S1 entities in 30k sample:           {len(sample_ids):,}")
    print(f"Target S1 entities with missed matches:     {total_target_s1:,} ({total_target_s1/len(sample_ids)*100:.2f}% of S1 entities)")
    print(f"Total missed true links (residual ~4.26%):  {total_missed_links:,} ({total_missed_links/total_true*100:.2f}% of true links)")
    print("\nCountry breakdown of Target S1 entities:")
    for c, cnt in target_by_country.items():
        print(f"  Country {c:<6}: {cnt:>5,} S1 entities ({cnt/total_target_s1*100:.1f}%), {missed_link_counts_by_country[c]:>5,} missed true links")
        
    # Check how many of these missed true links appear in the widened top-100 pool!
    recovered_in_100 = 0
    for sid, missed_cids in target_s1_misses.items():
        cands_100 = set(cand_union_100[sid].keys())
        for mcid in missed_cids:
            if mcid in cands_100:
                recovered_in_100 += 1
    print(f"\nOf {total_missed_links:,} missed links, present in top-100 candidate pool: {recovered_in_100:,} ({recovered_in_100/total_missed_links*100:.2f}%)")
    
    # 3. Manual Inspection of 30 Misses
    print("\n" + "=" * 80)
    print("  STEP 2.3: MANUAL FORENSIC INSPECTION OF 30 TARGET-SET MISSES")
    print("=" * 80)
    
    inspection_records = []
    # Pick a balanced selection of India and US misses
    india_miss_sids = [sid for sid in target_s1_misses if sample_s1_map[sid]['country'] == 'India']
    us_miss_sids = [sid for sid in target_s1_misses if sample_s1_map[sid]['country'] == 'US']
    
    inspect_sids = india_miss_sids[:20] + us_miss_sids[:10]
    
    category_counts = {
        "Script-Switch (Indic vs Latin)": 0,
        "Extreme Typo / Decoy / Name Disconnect": 0,
        "Truncated / Disjoint Address": 0,
        "Unfixable / Noise (No Overlap)": 0
    }
    
    for idx, sid in enumerate(inspect_sids):
        s1_row = sample_s1_map[sid]
        missed_cids = target_s1_misses[sid]
        cid = missed_cids[0]
        c_row = s2s3_map.get(cid, {'business_name': '[NOT_FOUND]', 'business_address': '[NOT_FOUND]'})
        
        s1_n = s1_row['business_name']
        s1_a = s1_row['business_address']
        c_n = c_row['business_name']
        c_a = c_row['business_address']
        country = s1_row['country']
        
        # Categorize
        has_non_ascii_s1 = any(ord(char) > 127 for char in s1_n)
        has_non_ascii_c = any(ord(char) > 127 for char in c_n)
        in_top_100 = cid in cand_union_100[sid]
        
        if has_non_ascii_s1 or has_non_ascii_c:
            cat = "Script-Switch (Indic vs Latin)"
        elif len(s1_a.strip()) == 0 or len(c_a.strip()) == 0 or len(set(normalize_text(s1_a).split()) & set(normalize_text(c_a).split())) == 0:
            cat = "Truncated / Disjoint Address"
        elif len(set(normalize_text(s1_n).split()) & set(normalize_text(c_n).split())) == 0:
            cat = "Extreme Typo / Decoy / Name Disconnect"
        else:
            cat = "Unfixable / Noise (No Overlap)"
            
        category_counts[cat] += 1
        
        print(f"\n[Miss #{idx+1:02d}] S1 ID: {sid} | True Target ID: {cid} | Country: {country} | In Top-100: {in_top_100}")
        print(f"  Category: {cat}")
        print(f"  S1  Name   : {s1_n}")
        print(f"  True Name  : {c_n}")
        print(f"  S1  Address: {s1_a}")
        print(f"  True Addr  : {c_a}")
        
    print("\n" + "-" * 80)
    print("  DIAGNOSTIC SUMMARY OF INSPECTED MISSES:")
    for cat, cnt in category_counts.items():
        print(f"    - {cat:<40}: {cnt:>2} / 30 ({cnt/30*100:.1f}%)")
    print("-" * 80)
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 3 — BUILD & EVALUATE SEMANTIC EMBEDDING SIGNAL
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  STEP 3: EVALUATING SEMANTIC EMBEDDING SIGNAL ON TARGET SET")
    print("=" * 80)
    
    target_s1_records = sample_s1[sample_s1['entity_id'].isin(target_s1_misses.keys())].reset_index(drop=True)
    target_s1_set = set(target_s1_records['entity_id'].values)
    
    # Collect candidate pairs to score with embedding model:
    # Union of top-100 candidates for all target-set entities
    pairs_to_score = []
    needed_cand_ids = set()
    for sid in target_s1_set:
        for cid in cand_union_100[sid].keys():
            pairs_to_score.append((sid, cid))
            needed_cand_ids.add(cid)
            
    print(f"Target set S1 queries: {len(target_s1_set):,}")
    print(f"Target candidate pool (top-100): {len(pairs_to_score):,} pairs across {len(needed_cand_ids):,} unique candidates")
    
    # Candidate records DataFrame
    cand_records_list = [s2s3_map[cid] for cid in needed_cand_ids if cid in s2s3_map]
    cand_records_df = pd.DataFrame(cand_records_list)
    
    # Test Model 1: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    m1_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"\n--- Testing Model 1: {m1_name} ---")
    t_m1 = time.time()
    m1_scores = get_embedding_scores(
        s1_records=target_s1_records,
        candidate_records=cand_records_df,
        candidate_pairs=pairs_to_score,
        model_name=m1_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=512,
        composite=False
    )
    t_m1_elapsed = time.time() - t_m1
    throughput_m1 = len(pairs_to_score) / max(t_m1_elapsed, 1e-6)
    print(f"Model 1 throughput: {throughput_m1:.1f} pairs/sec (Total wall-clock: {t_m1_elapsed:.1f}s)")
    
    # ═══════════════════════════════════════════════════════════════
    # VALIDATION GATE — COMBINING LEXICAL & SEMANTIC SIGNALS
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  VALIDATION GATE: EVALUATING COMBINED PIPELINE")
    print("=" * 80)
    
    # We evaluate different combination rules:
    # Rule 1: max(lexical_score, semantic_score) on top-30 candidates
    # Rule 2: max(lexical_score, semantic_score) on widened top-100 candidates for target entities
    # Rule 3: 0.5 * lexical + 0.5 * semantic
    
    def run_combination_experiment(rule_name, cand_scorer_fn):
        combined_cands = {}
        # Ensure outside entities are 100% untouched
        for sid in sample_ids:
            if sid not in target_s1_set:
                combined_cands[sid] = dict(cand_union_30[sid])
            else:
                combined_cands[sid] = cand_scorer_fn(sid)
                
        # Check pair completeness
        found = sum(len(gt_dict[sid] & set(combined_cands[sid].keys())) for sid in sample_ids)
        pc = found / total_true
        
        # Sweep tau with M2O
        best_tau = 0.800
        best_f05 = 0.0
        for tau in [0.760, 0.780, 0.800, 0.810, 0.820, 0.840]:
            f05 = evaluate_with_m2o(sample_ids, combined_cands, gt_dict, tau)
            if f05 > best_f05:
                best_f05 = f05
                best_tau = tau
                
        return pc, best_tau, best_f05
        
    # Combination A: max(lexical, semantic) on top-30 candidates
    def scorer_rule_a(sid):
        res = dict(cand_union_30[sid])
        sem = m1_scores.get(sid, {})
        for cid in res:
            if cid in sem:
                res[cid] = max(res[cid], sem[cid])
        return res
        
    # Combination B: max(lexical, semantic) on widened top-100 candidates
    def scorer_rule_b(sid):
        res = dict(cand_union_100[sid])
        sem = m1_scores.get(sid, {})
        for cid in res:
            if cid in sem:
                res[cid] = max(res[cid], sem[cid])
        return res
        
    # Combination C: blended 0.6 lexical + 0.4 semantic on top-30
    def scorer_rule_c(sid):
        res = {}
        lex = cand_union_30[sid]
        sem = m1_scores.get(sid, {})
        for cid, ls in lex.items():
            ss = sem.get(cid, 0.0)
            res[cid] = 0.6 * ls + 0.4 * ss
        return res
        
    pc_a, tau_a, f05_a = run_combination_experiment("Rule A: max(lexical, sem) on top-30", scorer_rule_a)
    pc_b, tau_b, f05_b = run_combination_experiment("Rule B: max(lexical, sem) on top-100", scorer_rule_b)
    pc_c, tau_c, f05_c = run_combination_experiment("Rule C: 0.6*lex + 0.4*sem on top-30", scorer_rule_c)
    
    # Check false positive and recovery statistics on target set
    # How many of the originally missed true matches got recovered?
    recovered_a = sum(len(gt_dict[sid] & set(scorer_rule_a(sid).keys()) - set(cand_union_30[sid].keys())) for sid in target_s1_set)
    recovered_b = sum(len(gt_dict[sid] & set(scorer_rule_b(sid).keys()) - set(cand_union_30[sid].keys())) for sid in target_s1_set)
    
    # Summary Table
    print("\n" + "=" * 90)
    print("  RAW VALIDATION GATE: BEFORE / AFTER COMPARISON TABLE")
    print("=" * 90)
    print(f"{'Pipeline':<55} | {'Pair Completeness':<18} | {'Macro F_0.5':<12}")
    print("-" * 90)
    print(f"{'Baseline (Two-channel + M2O, τ=0.800)':<55} | {pc_union_30*100:>17.2f}% | {baseline_f05_080:>12.4f}")
    print(f"{f'+ Embedding max(lex, sem) on top-30 (τ={tau_a:.3f})':<55} | {pc_a*100:>17.2f}% | {f05_a:>12.4f}")
    print(f"{f'+ Embedding max(lex, sem) on widened top-100 (τ={tau_b:.3f})':<55} | {pc_b*100:>17.2f}% | {f05_b:>12.4f}")
    print(f"{f'+ Embedding 0.6*lex + 0.4*sem on top-30 (τ={tau_c:.3f})':<55} | {pc_c*100:>17.2f}% | {f05_c:>12.4f}")
    print("=" * 90)
    
    print(f"\nOriginally missed true matches in top-30: {total_missed_links:,}")
    print(f"Newly recovered in candidate pool (Rule B): {recovered_b:,}")
    print(f"Macro F_0.5 delta with Rule B: {f05_b - baseline_f05_080:+.4f}")
    
    print(f"\nTotal elapsed time: {time.time()-t_start:.1f}s")

if __name__ == "__main__":
    main()
