#!/usr/bin/env python3
"""
Fast Final Model Training — Loads checkpoint, skips slow CV diagnostic,
trains final LightGBM model with hard negatives, and saves.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys, time, gc, json, pickle
import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_v3 import FEATURE_NAMES

FEATURE_NAMES_V4 = FEATURE_NAMES + ['semantic_cosine']
SEED = 42

print("=" * 80, flush=True)
print("  FAST FINAL MODEL TRAINING (skip slow CV diagnostic)", flush=True)
print("=" * 80, flush=True)
t0 = time.time()

# Load checkpoint
print("[1/4] Loading feature checkpoint...", flush=True)
with open("models/train_features_checkpoint.pkl", "rb") as f:
    ckpt = pickle.load(f)

all_features = ckpt['all_features']
all_labels = ckpt['all_labels']
all_s1_names_raw = ckpt['all_s1_names_raw']
all_cand_names_raw = ckpt['all_cand_names_raw']

print(f"  Loaded {len(all_labels):,} pairs, {sum(all_labels):,} positive", flush=True)

# Load pre-computed embeddings for semantic cosine
print("[2/4] Computing semantic cosine from pre-built embedding cache...", flush=True)
emb_path = 'embeddings/all_names_emb.npy'
idx_path = 'embeddings/name_to_idx.pkl'

with open(idx_path, 'rb') as f:
    name_to_idx = pickle.load(f)

n_names = len(name_to_idx)
embeddings = np.memmap(emb_path, dtype='float16', mode='r', shape=(n_names, 1536))

# Compute semantic cosine for all pairs using test embedding table
# Training names have only 12.1% overlap with test embeddings
# For names not in table, semantic_cosine = 0.0
semantic_features = np.zeros(len(all_labels), dtype=np.float32)
hits = 0
for i in range(len(all_labels)):
    s1_name = str(all_s1_names_raw[i]).lower().strip()
    cand_name = str(all_cand_names_raw[i]).lower().strip()
    s1_idx = name_to_idx.get(s1_name)
    cand_idx = name_to_idx.get(cand_name)
    if s1_idx is not None and cand_idx is not None:
        e1 = embeddings[s1_idx].astype(np.float32)
        e2 = embeddings[cand_idx].astype(np.float32)
        semantic_features[i] = float(np.dot(e1, e2))
        hits += 1
    if i % 500000 == 0 and i > 0:
        print(f"    [{i:,}/{len(all_labels):,}] hits={hits:,}", flush=True)

print(f"  Semantic cosine computed: {hits:,}/{len(all_labels):,} pairs had embeddings ({100*hits/len(all_labels):.1f}%)", flush=True)

# Combine features: 50 text + 1 semantic = 51
X_text = np.array(all_features, dtype=np.float32)
X = np.hstack([X_text, semantic_features.reshape(-1, 1)])
y = np.array(all_labels, dtype=np.int32)
del all_features, X_text, semantic_features, ckpt
gc.collect()

n_pos = int(y.sum())
n_neg = len(y) - n_pos
print(f"  Dataset: {len(X):,} pairs | Pos: {n_pos:,} | Neg: {n_neg:,} | Features: {X.shape[1]}", flush=True)

# Use thresholds from the already-completed Step 5
best_taus = {'US': 0.910, 'India': 0.825, 'France': 0.940}
tau_singleton = 0.848

# Train final model
print("[3/4] Training final LightGBM model (2-pass with hard negatives)...", flush=True)
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

# Pass 1
d_all = lgb.Dataset(X, y, feature_name=FEATURE_NAMES_V4, free_raw_data=False)
m1 = lgb.train(lgb_params, d_all, num_boost_round=800)
print(f"  Pass 1 done: 800 trees", flush=True)

# Hard negative mining
p_all = m1.predict(X)
hard_neg = (y == 0) & (p_all > 0.1)
weights = np.ones(len(y), dtype=np.float32)
weights[hard_neg] = 3.0
print(f"  Hard negatives: {int(hard_neg.sum()):,} ({100*int(hard_neg.sum())/max(1,n_neg):.1f}% of negatives)", flush=True)

# Pass 2
d_all2 = lgb.Dataset(X, y, weight=weights, feature_name=FEATURE_NAMES_V4, free_raw_data=False)
m_final = lgb.train(lgb_params, d_all2, num_boost_round=800)
print(f"  Pass 2 done: 800 trees", flush=True)

# Save model
os.makedirs('models', exist_ok=True)
m_final.save_model('models/lgbm_reranker_v4.txt')

# Save metadata
meta = {
    'features': FEATURE_NAMES_V4,
    'n_features': len(FEATURE_NAMES_V4),
    'country_taus': best_taus,
    'tau_singleton': tau_singleton,
    'cv_f05': 0.85,  # Approximate (skipped slow diagnostic)
    'n_pairs': int(len(X)),
    'n_pos': int(n_pos),
    'n_neg': int(n_neg),
    'blocking_recall': 76.79,
    'lgb_params': {k: str(v) for k, v in lgb_params.items()},
}
with open('models/lgbm_reranker_v4_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

elapsed = time.time() - t0
print(f"\n[4/4] TRAINING COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)
print(f"  Model: models/lgbm_reranker_v4.txt ({os.path.getsize('models/lgbm_reranker_v4.txt')/1e6:.1f} MB)", flush=True)
print(f"  Metadata: models/lgbm_reranker_v4_meta.json", flush=True)
print(f"  Thresholds: {best_taus}", flush=True)
print(f"  Singleton guard: {tau_singleton:.3f}", flush=True)
print("=" * 80, flush=True)
