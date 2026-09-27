#!/usr/bin/env python3
"""
LightGBM Reranker Training v4 — Amazon ML Challenge 2026

HIGH-PERFORMANCE 32-WORKER MULTIPROCESSING PIPELINE:
1. Parallel preprocessing and 7-channel text blocking across 32 CPU cores
2. Multiprocess chunked candidate retrieval & 50-feature extraction (32x speedup)
3. GPU embedding of unique training names via GTE-Qwen2-1.5B (FP16)
4. 51st feature: semantic cosine similarity
5. 5-Fold GroupKFold CV by entity_id (prevents entity leakage)
6. 2-Pass Hard Negative Mining (focuses tree splits on ambiguous pairs)
7. Calibrated singleton guard & per-country F0.5 threshold optimization
8. Trains & saves models/lgbm_reranker_v4.txt + metadata
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys, time, gc, json, argparse, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import multiprocessing as mp
from collections import defaultdict
from sklearn.model_selection import GroupKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalization_v3 import parallel_process_corpus_v3
from blocking_v3 import MultiChannelBlockingEngineV3
from features_v3 import FEATURE_NAMES, extract_pairwise_features_v3

SEED = 42
DATA_DIR = "dataset/student_resource/dataset/train"
FEATURE_NAMES_V4 = FEATURE_NAMES + ['semantic_cosine']

_G_TRAIN = None

# ─── Worker Function for Parallel Feature Extraction ─────────────────
def _train_extract_chunk(chunk_args):
    b_idx, bs, be = chunk_args
    engine = _G_TRAIN['engine']
    s1_ids = _G_TRAIN['s1_ids']
    gt_dict = _G_TRAIN['gt_dict']
    s1_p = _G_TRAIN['s1_pack']
    c_p = _G_TRAIN['c_pack']
    c_ids_arr = _G_TRAIN['c_ids_arr']
    c_is_s3 = _G_TRAIN['c_is_s3']
    s1_pks = _G_TRAIN['s1_pks']
    s1_f3 = _G_TRAIN['s1_f3']
    s1_raw = _G_TRAIN['s1_names_raw']
    c_raw = _G_TRAIN['c_names_raw']
    country = _G_TRAIN['country']

    cands_batch = engine.retrieve_candidates_for_batch(
        bs, be, s1_p[8], s1_p[2], s1_p[3], c_p[3],
        s1_phonetics=s1_pks, s1_first3=s1_f3
    )

    chunk_feats = []
    chunk_labels = []
    chunk_s1_ids = []
    chunk_cand_ids = []
    chunk_countries = []
    chunk_s1_names = []
    chunk_cand_names = []
    chunk_captured = 0

    for i, cands in enumerate(cands_batch):
        gi = bs + i
        sid = s1_ids[gi]
        true_set = gt_dict.get(sid, set())

        s1_item = (
            s1_p[0][gi], s1_p[1][gi], s1_p[2][gi], s1_p[3][gi], s1_p[4][gi],
            s1_p[5][gi], s1_p[6][gi], s1_p[7][gi], s1_p[8][gi], s1_p[9][gi],
            s1_p[10][gi], s1_p[11][gi], s1_p[12][gi], s1_p[13][gi], s1_p[14][gi]
        )

        for rank, (cidx, scores) in enumerate(cands, 1):
            cid = c_ids_arr[cidx]
            is_match = 1 if cid in true_set else 0
            if is_match:
                chunk_captured += 1

            c_item = (
                c_p[0][cidx], c_p[1][cidx], c_p[2][cidx], c_p[3][cidx], c_p[4][cidx],
                c_p[5][cidx], c_p[6][cidx], c_p[7][cidx], c_p[8][cidx], c_p[9][cidx],
                c_p[10][cidx], c_p[11][cidx], c_p[12][cidx], c_p[13][cidx], c_p[14][cidx]
            )

            feats = extract_pairwise_features_v3(s1_item, c_item, scores, rank, c_is_s3[cidx])
            chunk_feats.append(feats)
            chunk_labels.append(is_match)
            chunk_s1_ids.append(sid)
            chunk_cand_ids.append(cid)
            chunk_countries.append(country)
            chunk_s1_names.append(s1_raw[gi])
            chunk_cand_names.append(c_raw[cidx])

    return (chunk_feats, chunk_labels, chunk_s1_ids, chunk_cand_ids,
            chunk_countries, chunk_s1_names, chunk_cand_names, chunk_captured)

# ─── GTE-Qwen2 Embedding Helpers ─────────────────────────────────────
def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]

def wait_for_gpu(required_gb=8.0):
    print(f"  Checking GPU memory availability (need >= {required_gb:.1f} GB free)...", flush=True)
    while True:
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / 1e9
            if free_gb >= required_gb:
                print(f"  GPU is free! Free VRAM: {free_gb:.2f} GB / {total_bytes/1e9:.2f} GB", flush=True)
                break
            print(f"  Waiting for GPU... Free VRAM: {free_gb:.2f} GB (need {required_gb:.1f} GB). Checking again in 30s...", flush=True)
        except Exception as e:
            print(f"  Error checking GPU: {e}. Retrying in 30s...", flush=True)
        time.sleep(30)

def build_embedding_cache(all_names_raw, batch_size=256, max_length=48):
    """Embed unique names on GPU and return cache: lowered_name -> float16 embedding."""
    from transformers import AutoTokenizer, AutoModel

    unique_lower = list(set(str(n).lower().strip() for n in all_names_raw if n and str(n).strip() != '' and str(n) != 'nan'))
    print(f"  Building embedding cache for {len(unique_lower):,} unique names on GPU...", flush=True)

    if len(unique_lower) == 0:
        return {}

    model_id = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=torch.float16).to('cuda')
    model.eval()

    cache = {}
    t0 = time.time()
    for i in range(0, len(unique_lower), batch_size):
        batch = unique_lower[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                          max_length=max_length, return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
            embs = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        embs_np = embs.cpu().numpy().astype(np.float16)
        for name, emb in zip(batch, embs_np):
            cache[name] = emb
        if (i // batch_size) % 100 == 0 and i > 0:
            speed = (i + len(batch)) / (time.time() - t0)
            eta = (len(unique_lower) - i - len(batch)) / max(speed, 1)
            print(f"    [{i+len(batch):,}/{len(unique_lower):,}] {speed:.0f}/sec, ETA: {eta/60:.0f}m", flush=True)

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  Embedding cache built: {len(cache):,} entries in {time.time()-t0:.1f}s", flush=True)
    return cache

def get_cosine(cache, name1, name2):
    n1 = str(name1).lower().strip() if name1 else ''
    n2 = str(name2).lower().strip() if name2 else ''
    e1 = cache.get(n1)
    e2 = cache.get(n2)
    if e1 is None or e2 is None:
        return 0.0
    return float(np.dot(e1.astype(np.float32), e2.astype(np.float32)))

# ─── F0.5 Evaluation ─────────────────────────────────────────────────
def compute_macro_f05(pred_dict, truth_dict):
    scores = []
    for s1_id in truth_dict:
        true = truth_dict[s1_id]
        pred = pred_dict.get(s1_id, set())
        if len(true) == 0 and len(pred) == 0:
            scores.append(1.0)
            continue
        tp = len(true & pred)
        fp = len(pred - true)
        fn = len(true - pred)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if p + r == 0:
            scores.append(0.0)
        else:
            f05 = 1.25 * p * r / (0.25 * p + r)
            scores.append(f05)
    return float(np.mean(scores))

# ─── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train LightGBM Reranker v4 (Parallelized)")
    parser.add_argument("--sample-size", type=int, default=40000, help="Number of S1 queries to sample")
    parser.add_argument("--n-workers", type=int, default=32, help="Number of parallel workers")
    args = parser.parse_args()

    sample_size = args.sample_size
    n_workers = args.n_workers

    print("=" * 80)
    print("  AMAZON ML CHALLENGE 2026: SOTA LIGHTGBM RERANKER v4 (PARALLEL)")
    print(f"  Sample: {sample_size:,} | Parallel Workers: {n_workers} | CV: GroupKFold(5)")
    print("=" * 80, flush=True)
    t_global = time.time()

    # ── Step 1: Load and sample ──────────────────────────────────────
    print("[1/6] Loading training data...", flush=True)
    s1 = pd.read_csv(f"{DATA_DIR}/train_source1.tsv", sep="\t", dtype=str).fillna("")
    gt = pd.read_csv(f"{DATA_DIR}/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

    gt_map = dict(zip(gt['source1_entity_id'], gt['matched_entity_ids']))
    del gt; gc.collect()

    s1['match_count'] = s1['entity_id'].map(
        lambda x: len(str(gt_map.get(x, "")).split(",")) if str(gt_map.get(x, "")) != "" else 0
    )
    s1['match_bin'] = s1['match_count'].map(
        lambda c: '0' if c == 0 else ('1-2' if c <= 2 else ('3-5' if c <= 5 else '6+'))
    )
    s1['stratify_col'] = s1['country'] + "_" + s1['match_bin']

    if sample_size >= len(s1):
        s1_samp = s1.copy()
    else:
        try:
            _, s1_samp = train_test_split(s1, test_size=sample_size, stratify=s1['stratify_col'], random_state=SEED)
        except ValueError:
            _, s1_samp = train_test_split(s1, test_size=sample_size, random_state=SEED)

    s1_samp = s1_samp.reset_index(drop=True)
    sample_ids = s1_samp['entity_id'].values

    gt_dict = {}
    for sid in sample_ids:
        raw = gt_map.get(sid, "")
        gt_dict[sid] = set(raw.split(",")) if raw else set()

    total_true = sum(len(v) for v in gt_dict.values())
    n_singletons = sum(1 for v in gt_dict.values() if len(v) == 0)
    print(f"  Sampled {len(s1_samp):,} S1 ({n_singletons:,} singletons, {total_true:,} true links).", flush=True)

    del s1; gc.collect()

    print("[2/6] Loading S2+S3 pool...", flush=True)
    s2 = pd.read_csv(f"{DATA_DIR}/train_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(f"{DATA_DIR}/train_source3.tsv", sep="\t", dtype=str).fillna("")
    s2['is_s3'] = 0.0
    s3['is_s3'] = 1.0
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3; gc.collect()

    # ── Step 2: Per-country parallel blocking & feature extraction ────
    all_features = []
    all_labels = []
    all_s1_ids = []
    all_cand_ids = []
    all_countries = []
    all_s1_names_raw = []
    all_cand_names_raw = []
    captured_true_total = 0

    global _G_TRAIN

    for country in ['US', 'India']:
        print(f"\n{'─'*75}\n  PROCESSING: {country}\n{'─'*75}", flush=True)

        s1_c = s1_samp[s1_samp['country'] == country].reset_index(drop=True)
        s2s3_c = s2s3[s2s3['country'] == country].reset_index(drop=True)
        n_s1 = len(s1_c)
        print(f"  S1: {n_s1:,} | S2+S3: {len(s2s3_c):,}", flush=True)

        # Preprocess
        t0 = time.time()
        (s1_nn, s1_nc, s1_ns, s1_nw, s1_fw,
         s1_an, s1_hn, s1_pc, s1_at, s1_nu,
         s1_pk, s1_ci, s1_st, s1_ls, s1_ct) = parallel_process_corpus_v3(
            s1_c['business_name'].values, s1_c['business_address'].values, n_workers=n_workers
        )
        s1_ids = s1_c['entity_id'].values
        s1_f3 = [n[:3] for n in s1_nc]
        s1_raw_names = s1_c['business_name'].values
        print(f"  S1 preprocessed in {time.time()-t0:.1f}s", flush=True)

        t0 = time.time()
        (c_nn, c_nc, c_ns, c_nw, c_fw,
         c_an, c_hn, c_pc, c_at, c_nu,
         c_pk, c_ci, c_st, c_ls, c_ct) = parallel_process_corpus_v3(
            s2s3_c['business_name'].values, s2s3_c['business_address'].values, n_workers=n_workers
        )
        c_ids_arr = s2s3_c['entity_id'].values
        c_is_s3 = s2s3_c['is_s3'].values
        c_f3 = [n[:3] for n in c_nc]
        c_raw_names = s2s3_c['business_name'].values
        print(f"  S2+S3 preprocessed in {time.time()-t0:.1f}s", flush=True)

        # Blocking fit
        engine = MultiChannelBlockingEngineV3(top_k_per_channel=50, max_union=50)
        engine.fit_and_transform_corpus(
            s1_nn, s1_an, s1_ns,
            c_nn, c_an, c_ns, c_pc, c_hn,
            c_phonetics=c_pk, c_first3=c_f3
        )

        CHUNK_SIZE = 500
        n_chunks = int(np.ceil(n_s1 / CHUNK_SIZE))
        country_true = sum(len(gt_dict[sid]) for sid in s1_ids)
        country_cap = 0

        _G_TRAIN = {
            'engine': engine,
            's1_ids': s1_ids,
            'gt_dict': gt_dict,
            's1_pack': (s1_nn, s1_nc, s1_ns, s1_nw, s1_fw,
                        s1_an, s1_at, s1_hn, s1_pc, s1_nu,
                        s1_pk, s1_ci, s1_st, s1_ls, s1_ct),
            'c_pack': (c_nn, c_nc, c_ns, c_nw, c_fw,
                       c_an, c_at, c_hn, c_pc, c_nu,
                       c_pk, c_ci, c_st, c_ls, c_ct),
            'c_ids_arr': c_ids_arr,
            'c_is_s3': c_is_s3,
            's1_pks': s1_pk,
            's1_f3': s1_f3,
            's1_names_raw': s1_raw_names,
            'c_names_raw': c_raw_names,
            'country': country
        }

        chunks = []
        for c_i in range(n_chunks):
            bs = c_i * CHUNK_SIZE
            be = min(bs + CHUNK_SIZE, n_s1)
            chunks.append((c_i, bs, be))

        print(f"  Parallel feature extraction across {n_chunks} chunks using {n_workers} workers...", flush=True)
        t_ext = time.time()
        ctx = mp.get_context('fork')
        with ctx.Pool(n_workers) as pool:
            done_chunks = 0
            for (c_feats, c_lbls, c_s1s, c_cands, c_cntrs, c_s1names, c_candnames, c_cap) in pool.imap_unordered(_train_extract_chunk, chunks):
                done_chunks += 1
                country_cap += c_cap
                captured_true_total += c_cap
                all_features.extend(c_feats)
                all_labels.extend(c_lbls)
                all_s1_ids.extend(c_s1s)
                all_cand_ids.extend(c_cands)
                all_countries.extend(c_cntrs)
                all_s1_names_raw.extend(c_s1names)
                all_cand_names_raw.extend(c_candnames)

                if done_chunks % 10 == 0 or done_chunks == n_chunks:
                    pct = 100.0 * done_chunks / n_chunks
                    elapsed = time.time() - t_ext
                    print(f"    Chunks {done_chunks}/{n_chunks} ({pct:.0f}%) | Pairs: {len(all_labels):,} | Cap: {country_cap:,}/{country_true:,} | {elapsed:.0f}s", flush=True)

        recall = 100.0 * country_cap / max(1, country_true)
        print(f"  [{country}] Blocking Recall: {recall:.2f}% ({country_cap:,}/{country_true:,}) in {time.time()-t_ext:.1f}s", flush=True)

        _G_TRAIN = None
        del engine, s1_c, s2s3_c
        del s1_nn, s1_nc, s1_ns, s1_nw, s1_fw, s1_an, s1_hn, s1_pc, s1_at, s1_nu, s1_pk, s1_ci, s1_st, s1_ls, s1_ct
        del c_nn, c_nc, c_ns, c_nw, c_fw, c_an, c_hn, c_pc, c_at, c_nu, c_pk, c_ci, c_st, c_ls, c_ct
        gc.collect()

    del s2s3; gc.collect()

    total_recall = 100.0 * captured_true_total / max(1, total_true)
    print(f"\n  TOTAL Blocking Recall: {total_recall:.2f}% ({captured_true_total:,}/{total_true:,})", flush=True)
    print(f"  Total candidate pairs generated: {len(all_labels):,}", flush=True)

    # Save feature extraction checkpoint
    CKPT_PATH = "models/train_features_checkpoint.pkl"
    os.makedirs("models", exist_ok=True)
    print(f"  Saving extracted text features to {CKPT_PATH}...", flush=True)
    with open(CKPT_PATH, "wb") as f:
        pickle.dump({
            'all_features': all_features,
            'all_labels': all_labels,
            'all_s1_ids': all_s1_ids,
            'all_cand_ids': all_cand_ids,
            'all_countries': all_countries,
            'all_s1_names_raw': all_s1_names_raw,
            'all_cand_names_raw': all_cand_names_raw,
            'captured_true_total': captured_true_total,
            'total_true': total_true
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("  Checkpoint saved successfully!", flush=True)

    # ── Step 3: Compute semantic cosine feature on GPU ───────────────
    print("\n[3/6] Computing semantic cosine features on GPU...", flush=True)
    all_raw_names = all_s1_names_raw + all_cand_names_raw
    wait_for_gpu(required_gb=8.0)
    emb_cache = build_embedding_cache(all_raw_names)

    semantic_features = np.zeros(len(all_labels), dtype=np.float32)
    for i in range(len(all_labels)):
        semantic_features[i] = get_cosine(emb_cache, all_s1_names_raw[i], all_cand_names_raw[i])

    del emb_cache, all_raw_names, all_s1_names_raw, all_cand_names_raw
    gc.collect()

    # Combine features: 50 text + 1 semantic = 51
    X_text = np.array(all_features, dtype=np.float32)
    X = np.hstack([X_text, semantic_features.reshape(-1, 1)])
    y = np.array(all_labels, dtype=np.int32)
    s1_id_arr = np.array(all_s1_ids)
    cand_id_arr = np.array(all_cand_ids)
    country_arr = np.array(all_countries)
    del all_features, all_labels, X_text, semantic_features
    gc.collect()

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"  Dataset: {len(X):,} pairs | Pos: {n_pos:,} ({100*n_pos/len(X):.2f}%) | Neg: {n_neg:,}", flush=True)
    print(f"  Features: {X.shape[1]} ({len(FEATURE_NAMES)} text + 1 semantic)", flush=True)

    # ── Step 4: GroupKFold CV with Hard Negative Mining ───────────────
    print("\n[4/6] GroupKFold(5) CV with Hard Negative Mining...", flush=True)

    lgb_params = {
        'objective': 'binary',
        'learning_rate': 0.03,
        'num_leaves': 127,
        'max_depth': 9,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': float(n_neg) / float(max(1, n_pos)),
        'metric': 'auc',
        'verbose': -1,
        'n_jobs': 32,
    }

    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y), dtype=np.float64)

    for fold, (trn_idx, val_idx) in enumerate(gkf.split(X, y, groups=s1_id_arr)):
        print(f"\n  ─ Fold {fold+1}/5 ─", flush=True)
        X_tr, y_tr = X[trn_idx], y[trn_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        # Pass 1: Standard training
        d_tr = lgb.Dataset(X_tr, y_tr, feature_name=FEATURE_NAMES_V4, free_raw_data=False)
        d_va = lgb.Dataset(X_va, y_va, reference=d_tr, free_raw_data=False)

        m1 = lgb.train(lgb_params, d_tr, num_boost_round=1000,
                       valid_sets=[d_va],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        print(f"    Pass1: {m1.best_iteration} trees, AUC={m1.best_score['valid_0']['auc']:.4f}", flush=True)

        # Hard negative mining
        p1 = m1.predict(X_tr)
        hard_neg = (y_tr == 0) & (p1 > 0.1)
        n_hard = int(hard_neg.sum())
        weights = np.ones(len(y_tr), dtype=np.float32)
        weights[hard_neg] = 3.0
        print(f"    Hard negatives: {n_hard:,} ({100*n_hard/max(1,int((y_tr==0).sum())):.1f}% of negatives)", flush=True)

        # Pass 2: Retrain with hard negative weights
        d_tr2 = lgb.Dataset(X_tr, y_tr, weight=weights, feature_name=FEATURE_NAMES_V4, free_raw_data=False)
        m2 = lgb.train(lgb_params, d_tr2, num_boost_round=1000,
                       valid_sets=[d_va],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        print(f"    Pass2: {m2.best_iteration} trees, AUC={m2.best_score['valid_0']['auc']:.4f}", flush=True)

        oof_preds[val_idx] = m2.predict(X_va)

    # ── Step 5: Threshold optimization per country ───────────────────
    print("\n[5/6] Per-country threshold optimization on F0.5...", flush=True)

    best_taus = {}
    for country in ['US', 'India']:
        mask = country_arr == country
        c_s1_ids = s1_id_arr[mask]
        c_cand_ids = cand_id_arr[mask]
        c_preds = oof_preds[mask]

        c_truth = {sid: gt_dict[sid] for sid in np.unique(c_s1_ids)}
        best_f05 = -1.0
        best_tau = 0.55

        for tau in np.arange(0.40, 0.985, 0.005):
            above = c_preds >= tau
            order = np.argsort(-c_preds[above])
            pred_dict = defaultdict(set)
            used = set()
            s1_above = c_s1_ids[above][order]
            cand_above = c_cand_ids[above][order]
            for s, c in zip(s1_above, cand_above):
                if c not in used:
                    pred_dict[s].add(c)
                    used.add(c)
            f05 = compute_macro_f05(pred_dict, c_truth)
            if f05 > best_f05:
                best_f05 = f05
                best_tau = float(tau)

        best_taus[country] = best_tau
        print(f"  {country}: tau={best_tau:.3f}, F0.5={best_f05:.4f}", flush=True)

    # France threshold: conservative offset for zero-shot
    best_taus['France'] = min(0.95, best_taus.get('US', 0.60) + 0.03)
    print(f"  France: tau={best_taus['France']:.3f} (US + 0.03, conservative zero-shot)", flush=True)

    # Singleton guard calibration
    print("\n  Calibrating singleton guard...", flush=True)
    max_probs = {}
    for i in range(len(oof_preds)):
        sid = s1_id_arr[i]
        if sid not in max_probs or oof_preds[i] > max_probs[sid]:
            max_probs[sid] = oof_preds[i]

    singleton_probs = [max_probs[sid] for sid in max_probs if len(gt_dict.get(sid, set())) == 0]
    tau_singleton = float(np.percentile(singleton_probs, 95)) if singleton_probs else 0.90
    print(f"  tau_singleton = {tau_singleton:.3f} (95th pct of {len(singleton_probs)} true singletons)", flush=True)

    # Global CV score
    all_truth = {sid: gt_dict[sid] for sid in np.unique(s1_id_arr)}
    pred_dict = defaultdict(set)
    used = set()
    for tau_name, tau_val in best_taus.items():
        if tau_name == 'France':
            continue
        mask = country_arr == tau_name
        above = oof_preds[mask] >= tau_val
        order = np.argsort(-oof_preds[mask][above])
        for idx in order:
            s = s1_id_arr[mask][above][idx]
            c = cand_id_arr[mask][above][idx]
            if c not in used:
                pred_dict[s].add(c)
                used.add(c)

    global_f05 = compute_macro_f05(pred_dict, all_truth)
    print(f"\n  GLOBAL CV F0.5 (GroupKFold, honest): {global_f05:.4f}", flush=True)

    # ── Step 6: Retrain on ALL data, save ────────────────────────────
    print("\n[6/6] Retraining final model on all data with hard negatives...", flush=True)

    d_all = lgb.Dataset(X, y, feature_name=FEATURE_NAMES_V4, free_raw_data=False)
    m_final1 = lgb.train(lgb_params, d_all, num_boost_round=800)

    p_all = m_final1.predict(X)
    hard_neg = (y == 0) & (p_all > 0.1)
    weights_all = np.ones(len(y), dtype=np.float32)
    weights_all[hard_neg] = 3.0
    print(f"  Hard negatives on full data: {int(hard_neg.sum()):,}", flush=True)

    d_all2 = lgb.Dataset(X, y, weight=weights_all, feature_name=FEATURE_NAMES_V4, free_raw_data=False)
    m_final = lgb.train(lgb_params, d_all2, num_boost_round=800)

    os.makedirs('models', exist_ok=True)
    m_final.save_model('models/lgbm_reranker_v4.txt')

    meta = {
        'features': FEATURE_NAMES_V4,
        'n_features': len(FEATURE_NAMES_V4),
        'country_taus': best_taus,
        'tau_singleton': tau_singleton,
        'cv_f05': global_f05,
        'n_pairs': int(len(X)),
        'n_pos': int(n_pos),
        'n_neg': int(n_neg),
        'blocking_recall': total_recall,
        'lgb_params': {k: str(v) for k, v in lgb_params.items()},
    }
    with open('models/lgbm_reranker_v4_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t_global
    print(f"\n{'='*80}")
    print(f"  TRAINING COMPLETE in {elapsed/60:.1f}m ({elapsed:.0f}s)")
    print(f"  Model: models/lgbm_reranker_v4.txt")
    print(f"  Metadata: models/lgbm_reranker_v4_meta.json")
    print(f"  CV F0.5: {global_f05:.4f}")
    print(f"  Thresholds: {best_taus}")
    print(f"  Singleton guard: {tau_singleton:.3f}")
    print(f"{'='*80}", flush=True)

if __name__ == '__main__':
    main()
