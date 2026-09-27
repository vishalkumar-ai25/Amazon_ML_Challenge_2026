#!/usr/bin/env python3
"""
High-Performance Universal Normalizer v2
Amazon ML Challenge 2026

Features:
1. AnyAscii Full Transliteration (Devanagari, Telugu, Tamil, Punjabi, Gujarati, Bengali, etc.)
2. Domain Extension & URL Stripping (.com, .in, .co.in, .org, .net, etc.)
3. Spaceless Canonical Form for Domain & Concatenated Name Matching
4. Numeric & Premise Cleaning (strips leading zeros: 00201 -> 201)
5. 64-Worker Multiprocessing Pool
"""

import re
import unicodedata
import anyascii
import pandas as pd
import multiprocessing as mp

STREET_EXPANSIONS = {
    r'\brd\b': 'road', r'\bdr\b': 'drive', r'\bst\b': 'street', r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard', r'\bpkwy\b': 'parkway', r'\bln\b': 'lane', r'\bct\b': 'court',
    r'\bhwy\b': 'highway', r'\br\.\b': 'rue', r'\brue\b': 'rue', r'\bbd\b': 'boulevard',
    r'\bav\b': 'avenue', r'\bimp\b': 'impasse', r'\bpl\b': 'place', r'\baly\b': 'alley',
    r'\bsq\b': 'square'
}

LEGAL_SUFFIX_MAP = {
    r'\bpvt\s+ltd\b': 'private limited', r'\bpvt\b': 'private', r'\bltd\b': 'limited',
    r'\binc\b': 'incorporated', r'\bcorp\b': 'corporation', r'\bco\b': 'company',
    r'\bllc\b': 'limited liability', r'\bllp\b': 'limited liability partnership',
    r'\bsarl\b': 'sarl', r'\bsas\b': 'sas', r'\bsasu\b': 'sasu', r'\beurl\b': 'eurl',
    r'\bsci\b': 'sci', r'\bgmbh\b': 'gmbh'
}

PREFIX_REGEX = re.compile(
    r'^[\.\-\<\>\#\@\*\_\~\s]+|^(dba|d/b/a|t/a|fka|aka|formerly|m/s|mr|dr|shri|smt)\s*[:\-]?\s*',
    re.IGNORECASE
)

DOMAIN_REGEX = re.compile(
    r'\.(com|in|org|net|co\.in|co|gov|edu|io|ai|biz|info|fr|online|store|tech)\b',
    re.IGNORECASE
)

def universal_normalize(text):
    """
    Universal normalizer:
    1. Transliterate all non-Latin Unicode scripts (Indic, etc.) to Latin via anyascii
    2. NFKD Unicode accent stripping (handles French diacritics: e, e, a, c)
    3. Strip noise prefixes and domain extensions
    4. Strip leading zeros on numeric tokens (00201 -> 201)
    5. Standardize legal and street terms
    """
    if not text or pd.isna(text) or str(text).lower() in ['nan', 'null', 'none']:
        return ""
    text = str(text)
    
    # 1. AnyAscii transliteration
    text = anyascii.anyascii(text)
    
    # 2. NFKD Unicode accent stripping
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower()
    
    # 3. Strip noise prefixes and domain extensions
    text = PREFIX_REGEX.sub('', text)
    text = DOMAIN_REGEX.sub('', text)
    
    # 4. Remove non-alphanumeric (keep spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 5. Strip leading zeros on numbers (00201 -> 201)
    text = re.sub(r'\b0+(\d+)\b', r'\1', text)
    
    # 6. Expand abbreviations
    for pat, rep in STREET_EXPANSIONS.items():
        text = re.sub(pat, rep, text)
    for pat, rep in LEGAL_SUFFIX_MAP.items():
        text = re.sub(pat, rep, text)
        
    return re.sub(r'\s+', ' ', text).strip()

def normalize_name_clean(text):
    """Deep name normalization: universal normalize + strip legal suffixes for pure name matching."""
    norm = universal_normalize(text)
    for term in ['private limited', 'limited liability partnership', 'limited liability',
                 'corporation', 'incorporated', 'private', 'limited', 'company',
                 'sarl', 'sas', 'sasu', 'eurl', 'sci']:
        norm = re.sub(r'\b' + term + r'\b', '', norm)
    return re.sub(r'\s+', ' ', norm).strip()

def extract_spaceless_name(text):
    """Extract spaceless alphanumeric string for website/domain/concatenated matching."""
    norm = universal_normalize(text)
    return re.sub(r'[^a-z0-9]', '', norm)

def extract_house_number(addr):
    """Extract premise / building number with leading zeros stripped."""
    if not addr or pd.isna(addr):
        return ""
    addr_str = str(addr).lower()
    m1 = re.search(r'\b(?:plot\s*no\.?|shop\s*no\.?|flat\s*no\.?|door\s*no\.?|d\.?no\.?|no\.?)\s*([a-z0-9\/\-]+)', addr_str)
    if m1:
        return m1.group(1).lstrip('0')
    m2 = re.search(r'\b(\d+[\w\/\-]*)\b', addr_str)
    if m2:
        return m2.group(1).lstrip('0')
    return ""

def extract_postal_code(addr):
    """Extract postal code: 5-digit US/FR zip or 6-digit Indian PIN."""
    if not addr or pd.isna(addr):
        return ""
    addr_str = str(addr)
    # 6-digit Indian PIN
    m = re.search(r'\b([1-9]\d{5})\b', addr_str)
    if m:
        return m.group(1)
    # 5-digit US/France zip
    m = re.search(r'\b(\d{5})\b', addr_str)
    if m:
        return m.group(1)
    return ""

def extract_address_tokens(addr):
    """Extract clean set of address tokens excluding generic stopwords."""
    norm = universal_normalize(addr)
    ignore = {
        'road', 'street', 'drive', 'avenue', 'lane', 'court', 'boulevard',
        'highway', 'parkway', 'unit', 'apt', 'apartment', 'suite', 'ste',
        'near', 'opp', 'opposite', 'behind', 'floor', 'block', 'sector',
        'plot', 'shop', 'flat', 'door', 'no', 'null', 'india', 'usa', 'france'
    }
    return {w for w in norm.split() if len(w) > 1 and w not in ignore}

def _batch_process_records_v2(chunk_names, chunk_addrs):
    names_norm, names_clean, names_spaceless, name_words, first_words = [], [], [], [], []
    addrs_norm, hns, pcs, addr_tokens, nums_list = [], [], [], [], []
    
    for n in chunk_names:
        norm = universal_normalize(n)
        clean = normalize_name_clean(n)
        sp = re.sub(r'[^a-z0-9]', '', norm)
        words = set(w for w in norm.split() if len(w) > 1)
        fw = norm.split()[0] if norm.split() else ""
        names_norm.append(norm)
        names_clean.append(clean)
        names_spaceless.append(sp)
        name_words.append(words)
        first_words.append(fw)
        
    for a in chunk_addrs:
        norm = universal_normalize(a)
        hn = extract_house_number(a)
        pc = extract_postal_code(a)
        tokens = extract_address_tokens(a)
        nums = {w.lstrip('0') for w in norm.split() if any(c.isdigit() for c in w) and w.lstrip('0')}
        addrs_norm.append(norm)
        hns.append(hn)
        pcs.append(pc)
        addr_tokens.append(tokens)
        nums_list.append(nums)
        
    return (names_norm, names_clean, names_spaceless, name_words, first_words,
            addrs_norm, hns, pcs, addr_tokens, nums_list)

def _worker_wrapper_v2(args):
    return _batch_process_records_v2(*args)

def parallel_process_corpus_v2(names, addrs, n_workers=64):
    """Parallelized corpus preprocessing scaling to 64 workers."""
    n = len(names)
    if n == 0:
        return ([], [], [], [], [], [], [], [], [], [])
    if n <= 5000 or n_workers <= 1:
        return _batch_process_records_v2(names, addrs)
        
    chunk_size = max(2000, n // (n_workers * 4) + 1)
    chunks = [
        (names[i:i+chunk_size], addrs[i:i+chunk_size])
        for i in range(0, n, chunk_size)
    ]
    with mp.Pool(n_workers) as pool:
        results = pool.map(_worker_wrapper_v2, chunks)
        
    names_norm = [x for r in results for x in r[0]]
    names_clean = [x for r in results for x in r[1]]
    names_spaceless = [x for r in results for x in r[2]]
    name_words = [x for r in results for x in r[3]]
    first_words = [x for r in results for x in r[4]]
    addrs_norm = [x for r in results for x in r[5]]
    hns = [x for r in results for x in r[6]]
    pcs = [x for r in results for x in r[7]]
    addr_tokens = [x for r in results for x in r[8]]
    nums_list = [x for r in results for x in r[9]]
    
    return (names_norm, names_clean, names_spaceless, name_words, first_words,
            addrs_norm, hns, pcs, addr_tokens, nums_list)
