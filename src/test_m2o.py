"""
Fast test: Evaluate higher thresholds (0.70 - 0.90) and Many-to-One post-processing constraint.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import re
import time
import gc

DATA_ROOT = "dataset/student_resource/dataset"

def normalize_text(text):
    if not text or pd.isna(text) or text == "nan":
        return ""
    text = text.lower().strip()
    text = re.sub(r'^(--|<<|>>|\.\.)\s*', '', text)
    text = re.sub(r'\.(com|net|org|co|io)\b', '', text)
    text = re.sub(r"[^\w\s]", ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).strip()

def compute_f05(true_dict, pred_dict, all_ids):
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
    return np.mean(scores)

def test_higher_thresholds():
    print("Testing higher thresholds and Many-to-One constraint on 10,000 India sample...")
    s1 = pd.read_csv(f"{DATA_ROOT}/train/train_source1.tsv", sep="\t", dtype=str).fillna("")
    s1_sample = s1[s1['country'] == 'India'].head(10000).reset_index(drop=True)
    sample_ids = s1_sample['entity_id'].values
    
    gt = pd.read_csv(f"{DATA_ROOT}/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    gt_sub = gt[gt['source1_entity_id'].isin(set(sample_ids))]
    gt_dict = {r['source1_entity_id']: set(r['matched_entity_ids'].split(",")) if r['matched_entity_ids'].strip() else set() for _, r in gt_sub.iterrows()}
    
    s2 = pd.read_csv(f"{DATA_ROOT}/train/train_source2.tsv", sep="\t", dtype=str).fillna("")
    s2_india = s2[s2['country'] == 'India']
    s3 = pd.read_csv(f"{DATA_ROOT}/train/train_source3.tsv", sep="\t", dtype=str).fillna("")
    s3_india = s3[s3['country'] == 'India']
    s2s3 = pd.concat([s2_india, s3_india], ignore_index=True)
    
    del s1, s2, s3, s2_india, s3_india, gt, gt_sub
    gc.collect()
    
    s1_full = (s1_sample['business_name'] + " " + s1_sample['business_name'] + " " + s1_sample['business_address']).apply(normalize_text).values
    s2s3_full = (s2s3['business_name'] + " " + s2s3['business_name'] + " " + s2s3['business_address']).apply(normalize_text).values
    s2s3_ids = s2s3['entity_id'].values
    
    vec = TfidfVectorizer(
        analyzer='word',
        max_features=150000,
        min_df=2,
        max_df=0.01,
        sublinear_tf=True,
        norm='l2',
        dtype=np.float32,
        token_pattern=r'(?u)\b\w+\b'
    )
    vec.fit(np.concatenate([s1_full, s2s3_full[:500000]]))
    s1_mat = vec.transform(s1_full)
    s2s3_mat = vec.transform(s2s3_full)
    
    print("Multiplication...")
    t0 = time.time()
    sims = s1_mat @ s2s3_mat.T
    if not isinstance(sims, sparse.csr_matrix):
        sims = sims.tocsr()
    print(f"Done in {time.time()-t0:.1f}s")
    
    # Store candidate pairs with scores
    candidates = {}
    top_k = 35
    for i in range(len(sample_ids)):
        sid = sample_ids[i]
        p0, p1 = sims.indptr[i], sims.indptr[i+1]
        if p0 == p1:
            candidates[sid] = []
            continue
        data = sims.data[p0:p1]
        indices = sims.indices[p0:p1]
        if len(data) <= top_k:
            top_idx = np.arange(len(data))
        else:
            top_idx = np.argpartition(data, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(-data[top_idx])]
        candidates[sid] = [(s2s3_ids[indices[j]], float(data[j])) for j in top_idx]
        
    print("\n--- Standard Threshold Sweep ---")
    best_f05 = 0
    best_tau = 0.75
    for tau in np.arange(0.70, 0.93, 0.02):
        pred_dict = {sid: {cid for cid, s in cands if s >= tau} for sid, cands in candidates.items()}
        score = compute_f05(gt_dict, pred_dict, sample_ids)
        marker = " ★ BEST" if score > best_f05 else ""
        if score > best_f05:
            best_f05 = score
            best_tau = tau
        print(f"  τ={tau:.3f} | F₀.₅ = {score:.4f}{marker}")
        
    print("\n--- Many-to-One Post-Processing (Greedy Best-Match) ---")
    # In many-to-one, each target ID is assigned only to the S1 entity that has highest similarity to it
    for tau in [best_tau, best_tau - 0.02, best_tau + 0.02]:
        all_triples = []
        for sid, cands in candidates.items():
            for cid, s in cands:
                if s >= tau:
                    all_triples.append((s, sid, cid))
        # Sort descending by similarity
        all_triples.sort(key=lambda x: -x[0])
        
        assigned_cids = set()
        m2o_preds = {sid: set() for sid in sample_ids}
        for s, sid, cid in all_triples:
            if cid not in assigned_cids:
                assigned_cids.add(cid)
                m2o_preds[sid].add(cid)
                
        score_m2o = compute_f05(gt_dict, m2o_preds, sample_ids)
        print(f"  Many-to-One at τ={tau:.3f} | F₀.₅ = {score_m2o:.4f}")

if __name__ == "__main__":
    test_higher_thresholds()
