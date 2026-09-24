import pandas as pd

DATA_ROOT = "dataset/student_resource/dataset"

def inspect():
    # Read first 20 S1 entities
    s1 = pd.read_csv(f"{DATA_ROOT}/train/train_source1.tsv", sep="\t", nrows=20, dtype=str).fillna("")
    s1_ids = set(s1['entity_id'])
    
    # Read ground truth for these S1 IDs
    gt_matches = {}
    needed_s2 = set()
    needed_s3 = set()
    for chunk in pd.read_csv(f"{DATA_ROOT}/train/train_ground_truth.tsv", sep="\t", chunksize=100000, dtype=str):
        matched = chunk[chunk['source1_entity_id'].isin(s1_ids)]
        for _, r in matched.iterrows():
            mids = [m.strip() for m in str(r['matched_entity_ids']).split(",") if m.strip()]
            gt_matches[r['source1_entity_id']] = mids
            for m in mids:
                if m.startswith("S2-"):
                    needed_s2.add(m)
                elif m.startswith("S3-"):
                    needed_s3.add(m)
        if len(gt_matches) >= len(s1_ids):
            break
            
    # Find records in S2
    s2_dict = {}
    for chunk in pd.read_csv(f"{DATA_ROOT}/train/train_source2.tsv", sep="\t", chunksize=200000, dtype=str):
        matched = chunk[chunk['entity_id'].isin(needed_s2)]
        for _, r in matched.iterrows():
            s2_dict[r['entity_id']] = r.to_dict()
        if len(s2_dict) >= len(needed_s2):
            break
            
    # Find records in S3
    s3_dict = {}
    for chunk in pd.read_csv(f"{DATA_ROOT}/train/train_source3.tsv", sep="\t", chunksize=200000, dtype=str):
        matched = chunk[chunk['entity_id'].isin(needed_s3)]
        for _, r in matched.iterrows():
            s3_dict[r['entity_id']] = r.to_dict()
        if len(s3_dict) >= len(needed_s3):
            break
            
    print("\n" + "="*80)
    print("DETAILED COMPARISON OF GROUND TRUTH MATCHES:")
    print("="*80)
    for _, r1 in s1.iterrows():
        sid = r1['entity_id']
        matches = gt_matches.get(sid, [])
        print(f"\n[S1: {sid}] ({r1['country']})")
        print(f"  Name:    '{r1['business_name']}'")
        print(f"  Address: '{r1['business_address']}'")
        if not matches:
            print("  --> SINGLETON (no matches)")
            continue
        for mid in matches:
            rec = s2_dict.get(mid) or s3_dict.get(mid)
            if rec:
                print(f"  --> MATCH [{mid}]:")
                print(f"      Name:    '{rec['business_name']}'")
                print(f"      Address: '{rec['business_address']}'")
            else:
                print(f"  --> MATCH [{mid}]: (not found in early chunks)")

if __name__ == "__main__":
    inspect()
