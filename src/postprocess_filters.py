import re
import pandas as pd
import numpy as np

def extract_house_number(address):
    if pd.isna(address): return ""
    match = re.search(r'^(\d+)', str(address).strip())
    return match.group(1) if match else ""

def extract_postal_code(address):
    if pd.isna(address): return ""
    match = re.findall(r'\b(\d{5,6})\b', str(address))
    return match[-1] if match else ""

def apply_filters(row, s23_dict, max_matches=8):
    """
    Apply post-processing rules to a single S1 row.
    """
    matches_str = row['matched_entity_ids']
    if not matches_str or pd.isna(matches_str):
        return ""
    
    matches = matches_str.split(',')
    
    # Rule 1: Excessive candidates
    if len(matches) > max_matches:
        return "" # If it matches too many, it's likely a generic generic, prune entirely or prune to max? Let's aggressively prune entirely to be safe and maximize precision, as precision is 4x weighted. Wait, no, maybe just return empty or slice. We'll return empty string for extreme over-predictions.
        
    s1_addr = row['business_address']
    s1_hn = extract_house_number(s1_addr)
    s1_pc = extract_postal_code(s1_addr)
    
    valid_matches = []
    for m in matches:
        if m not in s23_dict:
            valid_matches.append(m)
            continue
            
        s23_addr = s23_dict[m]
        s23_hn = extract_house_number(s23_addr)
        s23_pc = extract_postal_code(s23_addr)
        
        # Conflict check
        if s1_hn and s23_hn and s1_hn != s23_hn:
            continue
        if s1_pc and s23_pc and s1_pc != s23_pc:
            continue
            
        valid_matches.append(m)
        
    return ",".join(valid_matches)

def process_file(matching_path, s1_path, s2_path, s3_path, out_path):
    print("Loading datasets for filtering...")
    preds = pd.read_csv(matching_path, sep='\t', dtype=str).fillna('')
    s1 = pd.read_csv(s1_path, sep='\t', dtype=str, usecols=['entity_id', 'business_address']).set_index('entity_id')
    s2 = pd.read_csv(s2_path, sep='\t', dtype=str, usecols=['entity_id', 'business_address']).set_index('entity_id')
    s3 = pd.read_csv(s3_path, sep='\t', dtype=str, usecols=['entity_id', 'business_address']).set_index('entity_id')
    
    s23_dict = pd.concat([s2, s3])['business_address'].to_dict()
    
    preds = preds.join(s1, on='source1_entity_id')
    
    print("Applying filters...")
    preds['matched_entity_ids'] = preds.apply(lambda row: apply_filters(row, s23_dict), axis=1)
    
    preds[['source1_entity_id', 'matched_entity_ids']].to_csv(out_path, sep='\t', index=False)
    print(f"Saved filtered results to {out_path}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 6:
        process_file(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
