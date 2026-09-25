#!/usr/bin/env python3
"""
Assemble Submission #2: France (Clean Two-Channel M2O) + India (Clean M2O) + US (Clean M2O)
Amazon ML Challenge 2026
"""

import os
import sys
import time
import gc
import pandas as pd

DATA_DIR = "dataset/student_resource/dataset/test"
OUTPUT_DIR = "output"
PARTS_DIR = "output/parts"
MATCHING_OUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

def assemble():
    print("=" * 80)
    print("  ASSEMBLING SUBMISSION #2 (FRANCE + INDIA + US ALL ZERO-DUPLICATE M2O)")
    print("=" * 80)
    t0 = time.time()
    
    # 1. Verify all 6 part files exist
    countries = ["France", "India", "US"]
    match_parts = {}
    cand_parts = {}
    for c in countries:
        mp = os.path.join(PARTS_DIR, f"match_{c}.tsv")
        cp = os.path.join(PARTS_DIR, f"cand_{c}.tsv")
        if not os.path.isfile(mp) or not os.path.isfile(cp):
            print(f"Error: Missing part files for {c} ({mp}, {cp})")
            sys.exit(1)
        match_parts[c] = mp
        cand_parts[c] = cp
        print(f"Verified {c}: {mp} ({os.path.getsize(mp)/(1024*1024):.1f} MB), {cp} ({os.path.getsize(cp)/(1024*1024):.1f} MB)")
        
    # 2. Check duplicate assignments in each country
    print("\n--- Auditing Duplicate Assignments per Country ---")
    for c in countries:
        df_m = pd.read_csv(match_parts[c], sep="\t", dtype=str, names=['sid', 'mids']).fillna("")
        all_mids = []
        for m in df_m['mids']:
            if m: all_mids.extend(m.split(','))
        dups = len(all_mids) - len(set(all_mids))
        print(f"  [{c}] Matches: {len(all_mids):,} | Unique: {len(set(all_mids)):,} | Duplicates: {dups:,}")
        del df_m, all_mids
        gc.collect()
        
    # 3. Load exact test order from test_source1.tsv
    print("\nLoading test_source1.tsv ordering...", end=" ", flush=True)
    s1_all = pd.read_csv(os.path.join(DATA_DIR, "test_source1.tsv"), sep="\t", usecols=['entity_id'])
    ordered_ids = s1_all['entity_id'].tolist()
    total_expected = len(ordered_ids)
    del s1_all
    gc.collect()
    print(f"Done ({total_expected:,} entities)")
    
    # 4. Load match map from all parts
    print("Loading match parts into memory...", end=" ", flush=True)
    match_map = {}
    for c in countries:
        with open(match_parts[c], "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    match_map[parts[0]] = parts[1]
                elif len(parts) == 1:
                    match_map[parts[0]] = ""
    print(f"Done ({len(match_map):,} keys)")
    
    # 5. Write matching_results.tsv
    print(f"Writing {MATCHING_OUT} in exact test order...", end=" ", flush=True)
    with open(MATCHING_OUT, "w", encoding="utf-8") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in ordered_ids:
            fm.write(f"{sid}\t{match_map.get(sid, '')}\n")
    print("Done")
    del match_map
    gc.collect()
    
    # 6. Load candidate map from all parts
    print("Loading candidate parts into memory...", end=" ", flush=True)
    cand_map = {}
    for c in countries:
        with open(cand_parts[c], "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    cand_map[parts[0]] = parts[1]
                elif len(parts) == 1:
                    cand_map[parts[0]] = ""
    print(f"Done ({len(cand_map):,} keys)")
    
    # 7. Write candidate_pairs.tsv
    print(f"Writing {CANDIDATE_OUT} in exact test order...", end=" ", flush=True)
    with open(CANDIDATE_OUT, "w", encoding="utf-8") as fc:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in ordered_ids:
            fc.write(f"{sid}\t{cand_map.get(sid, '')}\n")
    print("Done")
    del cand_map
    gc.collect()
    
    # Copy matching_results.tsv to root for easy UI upload
    root_m = "matching_results.tsv"
    import shutil
    shutil.copyfile(MATCHING_OUT, root_m)
    print(f"Copied {MATCHING_OUT} -> {root_m}")
    
    print(f"\n{'='*80}")
    print(f"  SUBMISSION #2 ASSEMBLED SUCCESSFULLY IN {time.time()-t0:.1f}s")
    print(f"{'='*80}")

if __name__ == "__main__":
    assemble()
