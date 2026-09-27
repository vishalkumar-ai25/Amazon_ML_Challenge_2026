#!/usr/bin/env python3
"""
GPU Embedding Pipeline for GTE-Qwen2-1.5B-instruct
Embeds all unique business names and saves as memory-mapped numpy array.

Amazon ML Challenge 2026 - Entity Resolution v4
"""

import os, sys, time, gc, pickle
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# ─── Config ───────────────────────────────────────────────────────────
MODEL_ID = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
BATCH_SIZE = 256
MAX_LENGTH = 48
EMB_DIM = 1536

DATA_DIR = "dataset/student_resource/dataset"
EMB_DIR = "embeddings"
os.makedirs(EMB_DIR, exist_ok=True)

# ─── Pooling (GTE-Qwen2 uses last-token pooling) ─────────────────────
def last_token_pool(last_hidden_states, attention_mask):
    """Pool the last non-padding token's hidden state (GTE standard)."""
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

# ─── Step 1: Collect all unique names ────────────────────────────────
def load_all_unique_names():
    """Load and deduplicate business names from test + train datasets."""
    print("=" * 70, flush=True)
    print("STEP 1: Loading all entity names...", flush=True)
    t0 = time.time()

    unique_names = set()
    file_list = [
        # Test data only (critical for inference)
        # Training names will be embedded on-the-fly during training for the 200K sample
        ("test/test_source1.tsv", "Test S1"),
        ("test/test_source2.tsv", "Test S2"),
        ("test/test_source3.tsv", "Test S3"),
    ]

    for fname, label in file_list:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"  SKIP {label}: {path} not found", flush=True)
            continue
        df = pd.read_csv(path, sep='\t', usecols=['business_name'])
        names = df['business_name'].fillna('').astype(str).str.strip()
        before = len(unique_names)
        for n in names:
            nl = n.lower()
            if nl and nl != 'nan':
                unique_names.add(nl)
        added = len(unique_names) - before
        print(f"  {label}: {len(names):>10,} rows  →  +{added:>9,} new unique  (total: {len(unique_names):,})", flush=True)
        del df, names
        gc.collect()

    # Sort for deterministic indexing
    name_list = sorted(unique_names)
    name_to_idx = {n: i for i, n in enumerate(name_list)}

    elapsed = time.time() - t0
    print(f"\nTotal unique names: {len(name_list):,}  (collected in {elapsed:.1f}s)", flush=True)
    print(f"Estimated embedding time: {len(name_list)/917/3600:.1f}h at 917 texts/sec", flush=True)
    return name_list, name_to_idx

# ─── Step 2: Embed all names ─────────────────────────────────────────
def embed_all_names(name_list):
    """Embed all names with GTE-Qwen2-1.5B, save as memory-mapped .npy."""
    n_total = len(name_list)

    # Create memory-mapped output
    emb_path = os.path.join(EMB_DIR, "all_names_emb.npy")
    meta_path = os.path.join(EMB_DIR, "embedding_meta.json")

    # Check for resume (if previous run was interrupted)
    start_batch = 0
    if os.path.exists(emb_path):
        existing = np.memmap(emb_path, dtype='float16', mode='r', shape=(n_total, EMB_DIM))
        # Find last non-zero row
        for i in range(n_total - 1, -1, -BATCH_SIZE):
            chunk_start = max(0, i - BATCH_SIZE + 1)
            chunk = np.array(existing[chunk_start:i+1])
            if np.any(chunk != 0):
                start_batch = (i // BATCH_SIZE) + 1
                break
        del existing
        if start_batch > 0:
            print(f"RESUMING from batch {start_batch} ({start_batch * BATCH_SIZE:,} names already embedded)", flush=True)

    embeddings = np.memmap(emb_path, dtype='float16', mode='r+' if start_batch > 0 else 'w+',
                           shape=(n_total, EMB_DIM))

    # Load model
    print("=" * 70, flush=True)
    print("STEP 2: Loading GTE-Qwen2-1.5B-instruct...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.float16
    ).to('cuda')
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s  (VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB)", flush=True)

    # Embed in batches
    print("=" * 70, flush=True)
    print(f"STEP 3: Embedding {n_total:,} names  (batch={BATCH_SIZE}, max_len={MAX_LENGTH})", flush=True)
    t_start = time.time()
    n_done = start_batch * BATCH_SIZE

    for batch_idx in range(start_batch, (n_total + BATCH_SIZE - 1) // BATCH_SIZE):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_total)
        batch = name_list[start:end]

        # Tokenize
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors='pt'
        ).to('cuda')

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
            embs = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])

        # L2 normalize on GPU (much faster than CPU)
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)

        # Store
        embeddings[start:end] = embs.cpu().numpy().astype(np.float16)
        n_done = end

        # Progress logging
        if batch_idx % 200 == 0 or end == n_total:
            elapsed = time.time() - t_start
            speed = (n_done - start_batch * BATCH_SIZE) / max(elapsed, 1)
            remaining = (n_total - n_done) / max(speed, 1)
            pct = n_done / n_total * 100
            print(
                f"  [{n_done:>10,}/{n_total:,}] {pct:5.1f}%  "
                f"| {speed:.0f} texts/sec  "
                f"| elapsed: {elapsed/60:.0f}m  "
                f"| ETA: {remaining/60:.0f}m ({remaining/3600:.1f}h)",
                flush=True
            )

        # Flush to disk every 2000 batches (~512K names)
        if batch_idx % 2000 == 0:
            embeddings.flush()

    embeddings.flush()
    total_time = time.time() - t_start
    file_size = os.path.getsize(emb_path) / 1e9

    print("=" * 70, flush=True)
    print(f"EMBEDDING COMPLETE!", flush=True)
    print(f"  Names embedded:  {n_total:,}", flush=True)
    print(f"  Total time:      {total_time/3600:.2f}h ({total_time:.0f}s)", flush=True)
    print(f"  Throughput:      {n_total/total_time:.0f} texts/sec", flush=True)
    print(f"  File size:       {file_size:.2f} GB", flush=True)
    print(f"  Shape:           ({n_total}, {EMB_DIM}) float16", flush=True)
    print(f"  Path:            {emb_path}", flush=True)

    return emb_path

# ─── Main ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70, flush=True)
    print("GTE-Qwen2-1.5B Embedding Pipeline v4", flush=True)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 70, flush=True)

    # Collect names
    name_list, name_to_idx = load_all_unique_names()

    # Save lookup structures
    idx_path = os.path.join(EMB_DIR, "name_to_idx.pkl")
    with open(idx_path, 'wb') as f:
        pickle.dump(name_to_idx, f, protocol=pickle.HIGHEST_PROTOCOL)

    list_path = os.path.join(EMB_DIR, "name_list.pkl")
    with open(list_path, 'wb') as f:
        pickle.dump(name_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved name_to_idx ({len(name_to_idx):,} entries) → {idx_path}", flush=True)
    print(f"Saved name_list → {list_path}", flush=True)

    # Embed
    emb_path = embed_all_names(name_list)

    # Quick sanity check
    print("\n=== SANITY CHECK ===", flush=True)
    embs = np.memmap(emb_path, dtype='float16', mode='r', shape=(len(name_list), EMB_DIM))

    test_pairs = [
        ("amazon development centre pvt ltd", "amazon dev center india"),
        ("mcdonald's", "mcdonald's fast food"),
        ("boulangerie artisanale", "artisan bakery"),
    ]
    for a, b in test_pairs:
        if a in name_to_idx and b in name_to_idx:
            ea = embs[name_to_idx[a]].astype(np.float32)
            eb = embs[name_to_idx[b]].astype(np.float32)
            cos = float(ea @ eb)
            print(f"  cos('{a}', '{b}') = {cos:.4f}", flush=True)
        else:
            missing = [x for x in [a, b] if x not in name_to_idx]
            print(f"  SKIP: {missing} not in vocabulary", flush=True)

    print(f"\nFinished at: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("ALL DONE! ✓", flush=True)

if __name__ == '__main__':
    main()
