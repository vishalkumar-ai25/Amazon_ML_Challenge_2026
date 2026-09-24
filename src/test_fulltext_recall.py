"""
Test blocking recall with full_text (name + address)
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
    text = re.sub(r'\.(com|net|org|co|io)\b', '', text)
    text = re.sub(r"[^\w\s]", ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).strip()

def test_full_text_recall():
    print("Testing full-text blocking recall on India...")
    s1 = pd.read_csv(f"{DATA_ROOT}/train/train_source1.tsv", sep="\t", dtype=str).fillna("")
    s1_india = s1[s1['country'] == 'India'].head(5000)
    s1_ids = set(s1_india['entity_id'])
    
    # Ground truth for these 5000 S1
    gt = pd.read_csv(f"{DATA_ROOT}/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    gt_sub = gt[gt['source1_entity_id'].isin(s1_ids)]
    gt_dict = {}
    for _, r in gt_sub.iterrows():
        mids = [m.strip() for m in r['matched_entity_ids'].split(",") if m.strip()]
        gt_dict[r['source1_entity_id']] = set(mids)
        
    total_true = sum(len(v) for v in gt_dict.values())
    print(f"5,000 India S1 entities with {total_true:,} true matches")
    
    # Load India S2+S3 (or sample / full)
    s2 = pd.read_csv(f"{DATA_ROOT}/train/train_source2.tsv", sep="\t", dtype=str).fillna("")
    s2_india = s2[s2['country'] == 'India']
    s3 = pd.read_csv(f"{DATA_ROOT}/train/train_source3.tsv", sep="\t", dtype=str).fillna("")
    s3_india = s3[s3['country'] == 'India']
    s2s3 = pd.concat([s2_india, s3_india], ignore_index=True)
    print(f"India S2+S3 corpus: {len(s2s3):,} records")
    
    del s1, s2, s3, s2_india, s3_india, gt, gt_sub
    gc.collect()
    
    # Create combined full text: name (repeated 2x to give weight) + address
    print("Preparing full text: 2*name + address...")
    s1_full = (s1_india['business_name'] + " " + s1_india['business_name'] + " " + s1_india['business_address']).apply(normalize_text).values
    s2s3_full = (s2s3['business_name'] + " " + s2s3['business_name'] + " " + s2s3['business_address']).apply(normalize_text).values
    
    s1_id_list = s1_india['entity_id'].values
    s2s3_id_list = s2s3['entity_id'].values
    
    print("Fitting TF-IDF on full text...")
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
    vec.fit(np.concatenate([s1_full, s2s3_full[:500000]]))  # Fit on sample for speed
    
    print("Transforming...")
    s1_mat = vec.transform(s1_full)
    s2s3_mat = vec.transform(s2s3_full)
    print(f"S1: {s1_mat.shape}, S2S3: {s2s3_mat.shape}")
    
    print("Querying top 30 candidates...")
    t0 = time.time()
    sims = s1_mat @ s2s3_mat.T
    if not isinstance(sims, sparse.csr_matrix):
        sims = sims.tocsr()
    print(f"Multiplication took {time.time()-t0:.1f}s")
    
    found_matches = 0
    top_k = 30
    for i in range(len(s1_id_list)):
        sid = s1_id_list[i]
        true_set = gt_dict.get(sid, set())
        if not true_set:
            continue
        row = sims.getrow(i)
        if row.nnz == 0:
            continue
        data = row.data
        indices = row.indices
        if len(data) <= top_k:
            cand_ids = {s2s3_id_list[idx] for idx in indices}
        else:
            top_idx = np.argpartition(data, -top_k)[-top_k:]
            cand_ids = {s2s3_id_list[indices[j]] for j in top_idx}
        found_matches += len(true_set & cand_ids)
        
    recall = found_matches / total_true
    print(f"\n==========================================")
    print(f"FULL-TEXT BLOCKING RECALL (top {top_k}): {recall:.4f} ({found_matches:,} / {total_true:,})")
    print(f"Previous name-only recall was: 0.5025")
    print(f"==========================================")

if __name__ == "__main__":
    test_full_text_recall()
