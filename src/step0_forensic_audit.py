"""
STEP 0+1: FORENSIC DATA TOPOLOGY AUDIT
Amazon ML Challenge 2026 — Business Entity Resolution
"""
import pandas as pd
import numpy as np
import sys

DATA_ROOT = "dataset/student_resource/dataset"

def load_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str)

def audit_source(name, path):
    print(f"\n{'='*70}")
    print(f"  {name}: {path}")
    print(f"{'='*70}")
    df = load_tsv(path)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")
    
    # Missing values
    print(f"\n--- Missing Values ---")
    for c in df.columns:
        n_miss = df[c].isna().sum()
        pct = 100.0 * n_miss / len(df)
        print(f"  {c}: {n_miss:,} ({pct:.2f}%)")
    
    # Text length distributions
    for c in ['business_name', 'business_address']:
        if c in df.columns:
            lengths = df[c].fillna("").str.len()
            print(f"\n--- {c} length stats ---")
            print(f"  min={lengths.min()}, median={lengths.median():.0f}, "
                  f"mean={lengths.mean():.1f}, max={lengths.max()}, "
                  f"zeros={int((lengths==0).sum())}")
    
    # Country distribution
    if 'country' in df.columns:
        print(f"\n--- Country Distribution ---")
        vc = df['country'].fillna("__NULL__").value_counts()
        for k, v in vc.items():
            print(f"  {k}: {v:,} ({100.0*v/len(df):.1f}%)")
    
    # Entity ID prefix check
    if 'entity_id' in df.columns:
        prefixes = df['entity_id'].str[:3].value_counts()
        print(f"\n--- Entity ID Prefixes ---")
        for k, v in prefixes.items():
            print(f"  {k}: {v:,}")
        # Check uniqueness
        n_unique = df['entity_id'].nunique()
        print(f"  Unique IDs: {n_unique:,} (duplicates: {len(df) - n_unique:,})")
    
    print(f"\n--- Sample Rows ---")
    print(df.head(3).to_string(index=False))
    return df

def audit_ground_truth(path):
    print(f"\n{'='*70}")
    print(f"  GROUND TRUTH: {path}")
    print(f"{'='*70}")
    df = load_tsv(path)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"Columns: {list(df.columns)}")
    
    # Parse matched_entity_ids
    df['matched_entity_ids'] = df['matched_entity_ids'].fillna("")
    
    # Singletons vs matched
    is_singleton = df['matched_entity_ids'].str.strip() == ""
    n_singletons = is_singleton.sum()
    n_matched = (~is_singleton).sum()
    print(f"\nSingletons (no matches): {n_singletons:,} ({100.0*n_singletons/len(df):.1f}%)")
    print(f"Matched (≥1 match):      {n_matched:,} ({100.0*n_matched/len(df):.1f}%)")
    
    # Match count distribution
    match_counts = df['matched_entity_ids'].apply(
        lambda x: len([i for i in x.split(",") if i.strip()]) if x.strip() else 0
    )
    print(f"\n--- Match Count Distribution ---")
    vc = match_counts.value_counts().sort_index()
    for k, v in vc.items():
        print(f"  {k} matches: {v:,} ({100.0*v/len(df):.1f}%)")
    print(f"\n  Total matches: {match_counts.sum():,}")
    print(f"  Mean matches per entity: {match_counts.mean():.3f}")
    print(f"  Max matches for single entity: {match_counts.max()}")
    
    # S2 vs S3 breakdown in matches
    all_matches = df.loc[~is_singleton, 'matched_entity_ids'].str.cat(sep=",")
    all_ids = [x.strip() for x in all_matches.split(",") if x.strip()]
    s2_count = sum(1 for x in all_ids if x.startswith("S2-"))
    s3_count = sum(1 for x in all_ids if x.startswith("S3-"))
    print(f"\n--- Match Source Breakdown ---")
    print(f"  S2 matches: {s2_count:,}")
    print(f"  S3 matches: {s3_count:,}")
    print(f"  Total match references: {len(all_ids):,}")
    
    # Check for duplicate IDs across match lists
    from collections import Counter
    id_counter = Counter(all_ids)
    multi_matched = {k: v for k, v in id_counter.items() if v > 1}
    print(f"\n  S2/S3 IDs matched to multiple S1 entities: {len(multi_matched):,}")
    if multi_matched:
        top5 = sorted(multi_matched.items(), key=lambda x: -x[1])[:5]
        for k, v in top5:
            print(f"    {k}: appears in {v} S1 match lists")
    
    print(f"\n--- Sample Rows ---")
    print(df.head(5).to_string(index=False))
    return df

if __name__ == "__main__":
    print("\n" + "█"*70)
    print("  AMAZON ML CHALLENGE 2026 — FORENSIC DATA TOPOLOGY AUDIT")
    print("█"*70)
    
    # TRAINING SET
    tr_s1 = audit_source("TRAIN Source 1", f"{DATA_ROOT}/train/train_source1.tsv")
    tr_s2 = audit_source("TRAIN Source 2", f"{DATA_ROOT}/train/train_source2.tsv")
    tr_s3 = audit_source("TRAIN Source 3", f"{DATA_ROOT}/train/train_source3.tsv")
    gt = audit_ground_truth(f"{DATA_ROOT}/train/train_ground_truth.tsv")
    
    # TEST SET
    te_s1 = audit_source("TEST Source 1", f"{DATA_ROOT}/test/test_source1.tsv")
    te_s2 = audit_source("TEST Source 2", f"{DATA_ROOT}/test/test_source2.tsv")
    te_s3 = audit_source("TEST Source 3", f"{DATA_ROOT}/test/test_source3.tsv")
    
    # CROSS-SET ANALYSIS
    print(f"\n{'='*70}")
    print(f"  CROSS-SET ANALYSIS")
    print(f"{'='*70}")
    
    # Country overlap
    train_countries = set(pd.concat([tr_s1, tr_s2, tr_s3])['country'].fillna("").unique())
    test_countries = set(pd.concat([te_s1, te_s2, te_s3])['country'].fillna("").unique())
    print(f"\nTrain countries: {train_countries}")
    print(f"Test countries:  {test_countries}")
    print(f"New in test:     {test_countries - train_countries}")
    
    # ID overlap check (should be zero)
    train_ids = set(tr_s1['entity_id']) | set(tr_s2['entity_id']) | set(tr_s3['entity_id'])
    test_ids = set(te_s1['entity_id']) | set(te_s2['entity_id']) | set(te_s3['entity_id'])
    overlap = train_ids & test_ids
    print(f"\nTrain-Test ID overlap: {len(overlap)} (should be 0)")
    
    # S1 ground truth coverage
    gt_s1_ids = set(gt['source1_entity_id'])
    tr_s1_ids = set(tr_s1['entity_id'])
    print(f"\nGT S1 IDs: {len(gt_s1_ids):,} | Train S1 IDs: {len(tr_s1_ids):,}")
    print(f"GT covers all Train S1: {gt_s1_ids == tr_s1_ids}")
    print(f"Missing from GT: {len(tr_s1_ids - gt_s1_ids)}")
    print(f"Extra in GT: {len(gt_s1_ids - tr_s1_ids)}")
    
    # Match ID coverage  
    gt_match_ids = set()
    for _, row in gt.iterrows():
        mids = row['matched_entity_ids']
        if isinstance(mids, str) and mids.strip():
            for mid in mids.split(","):
                gt_match_ids.add(mid.strip())
    
    tr_s2_ids = set(tr_s2['entity_id'])
    tr_s3_ids = set(tr_s3['entity_id'])
    gt_s2 = {x for x in gt_match_ids if x.startswith("S2-")}
    gt_s3 = {x for x in gt_match_ids if x.startswith("S3-")}
    
    print(f"\nGT references {len(gt_s2):,} unique S2 IDs | Train S2 has {len(tr_s2_ids):,}")
    print(f"  S2 in GT but not in Train S2: {len(gt_s2 - tr_s2_ids)}")
    print(f"  S2 in Train S2 but not in GT: {len(tr_s2_ids - gt_s2):,} (unmatched S2 records)")
    
    print(f"\nGT references {len(gt_s3):,} unique S3 IDs | Train S3 has {len(tr_s3_ids):,}")
    print(f"  S3 in GT but not in Train S3: {len(gt_s3 - tr_s3_ids)}")
    print(f"  S3 in Train S3 but not in GT: {len(tr_s3_ids - gt_s3):,} (unmatched S3 records)")
    
    # Business name overlap analysis
    print(f"\n--- Business Name Overlap (exact, lowercased) ---")
    def get_names(df):
        return set(df['business_name'].fillna("").str.lower().str.strip())
    
    for label, s_df, ref_df, ref_label in [
        ("Train S1 vs S2", tr_s1, tr_s2, "S2"),
        ("Train S1 vs S3", tr_s1, tr_s3, "S3"),
        ("Test S1 vs S2", te_s1, te_s2, "S2"),
        ("Test S1 vs S3", te_s1, te_s3, "S3"),
    ]:
        n1 = get_names(s_df)
        n2 = get_names(ref_df)
        overlap = n1 & n2
        print(f"  {label}: {len(overlap):,} exact name matches ({100.0*len(overlap)/len(n1):.1f}% of S1)")
    
    print(f"\n{'='*70}")
    print(f"  AUDIT COMPLETE")
    print(f"{'='*70}")
