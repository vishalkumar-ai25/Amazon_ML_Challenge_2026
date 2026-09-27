#!/usr/bin/env python3
"""
High-Performance Universal Normalizer v3
Amazon ML Challenge 2026

Features:
1. AnyAscii Full Transliteration (Devanagari, Telugu, Tamil, Punjabi, Gujarati, Bengali, etc.)
2. Domain Extension & URL Stripping (.com, .in, .co.in, .org, .net, etc.)
3. Spaceless Canonical Form for Domain & Concatenated Name Matching
4. Numeric & Premise Cleaning (strips leading zeros: 00201 -> 201)
5. Extended French & Legal Normalization
6. Phonetic Keys, Char N-grams, City/State Extraction
7. 64-Worker Multiprocessing Pool
"""

import re
import unicodedata
import anyascii
import pandas as pd
import multiprocessing as mp

STREET_EXPANSIONS = {
    r'\brd\b': 'road', r'\bdr\b': 'drive', r'\bst\b': 'street', r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard', r'\bpkwy\b': 'parkway', r'\bln\b': 'lane', r'\bct\b': 'court',
    r'\bhwy\b': 'highway', r'\br\b': 'rue', r'\brue\b': 'rue', r'\bbd\b': 'boulevard',
    r'\bav\b': 'avenue', r'\bimp\b': 'impasse', r'\bpl\b': 'place', r'\baly\b': 'alley',
    r'\bsq\b': 'square', r'\bch\b': 'chemin', r'\brte\b': 'route', r'\ball\b': 'allee',
    r'\bcrs\b': 'cours'
}

LEGAL_SUFFIX_MAP = {
    r'\bpvt\s+ltd\b': 'private limited', r'\bpvt\b': 'private', r'\bltd\b': 'limited',
    r'\binc\b': 'incorporated', r'\bcorp\b': 'corporation', r'\bco\b': 'company',
    r'\bllc\b': 'limited liability', r'\bllp\b': 'limited liability partnership',
    r'\bsarl\b': 'sarl', r'\bsas\b': 'sas', r'\bsasu\b': 'sasu', r'\beurl\b': 'eurl',
    r'\bsci\b': 'sci', r'\bsa\b': 'sa', r'\bsnc\b': 'snc', r'\bsca\b': 'sca',
    r'\bgie\b': 'gie', r'\bscop\b': 'scop', r'\bscea\b': 'scea', r'\bgaec\b': 'gaec',
    r'\bearl\b': 'earl', r'\bselarl\b': 'selarl',
    r'\bsociete\b': 'societe', r'\bste\b': 'ste',
    r'\betablissements\b': 'etablissements', r'\bets\b': 'ets',
    r'\bet\s+fils\b': 'et fils', r'\bet\s+freres\b': 'et freres',
    r'\bgmbh\b': 'gmbh'
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
    legal_terms = [
        'private limited', 'limited liability partnership', 'limited liability',
        'corporation', 'incorporated', 'private', 'limited', 'company',
        'sarl', 'sas', 'sasu', 'eurl', 'sci', 'sa', 'snc', 'sca', 'gie', 'scop',
        'scea', 'gaec', 'earl', 'selarl', 'societe', 'ste', 'etablissements',
        'ets', 'et fils', 'et freres', 'gmbh'
    ]
    for term in legal_terms:
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
        'plot', 'shop', 'flat', 'door', 'no', 'null', 'india', 'usa', 'france',
        'rue', 'impasse', 'place', 'alley', 'square', 'chemin', 'route', 'allee', 'cours'
    }
    return {w for w in norm.split() if len(w) > 1 and w not in ignore}

def extract_phonetic_keys(name):
    """Extract simple consonant-skeleton phonetic keys for the first 3 significant words."""
    norm = universal_normalize(name)
    words = [w for w in norm.split() if len(w) > 1][:3]
    keys = []
    for w in words:
        w = re.sub(r'[aeiouhw]', '', w)
        w = re.sub(r'(.)\1+', r'\1', w)
        if w:
            keys.append(w)
    return frozenset(keys)

def extract_char_ngrams(text, n=3):
    """Extract character n-grams from text."""
    text = str(text).replace(' ', '_').lower()
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text)-n+1)}

def extract_city(addr):
    """Extract city from address (heuristic: 2nd to last component)."""
    if not addr or pd.isna(addr):
        return ""
    parts = [p.strip() for p in str(addr).split(',')]
    if len(parts) >= 2:
        return parts[-2].lower()
    return ""

def extract_state(addr):
    """Extract state from address (heuristic: last component)."""
    if not addr or pd.isna(addr):
        return ""
    parts = [p.strip() for p in str(addr).split(',')]
    if len(parts) >= 1:
        return parts[-1].lower()
    return ""

def extract_legal_suffix(name):
    """Extract legal suffix from a raw name."""
    if not name or pd.isna(name):
        return ""
    name_str = str(name).lower()
    for pat, rep in LEGAL_SUFFIX_MAP.items():
        if re.search(pat, name_str):
            return rep
    return ""

def _batch_process_records_v3(chunk_names, chunk_addrs):
    names_norm, names_clean, names_spaceless, name_words, first_words = [], [], [], [], []
    addrs_norm, hns, pcs, addr_tokens, nums_list = [], [], [], [], []
    phonetic_keys, cities, states, legal_suffixes, char_trigrams_name = [], [], [], [], []
    
    for n in chunk_names:
        norm = universal_normalize(n)
        clean = normalize_name_clean(n)
        sp = extract_spaceless_name(n)
        words = set(w for w in norm.split() if len(w) > 1)
        fw = norm.split()[0] if norm.split() else ""
        
        names_norm.append(norm)
        names_clean.append(clean)
        names_spaceless.append(sp)
        name_words.append(words)
        first_words.append(fw)
        
        phonetic_keys.append(extract_phonetic_keys(norm))
        legal_suffixes.append(extract_legal_suffix(n))
        char_trigrams_name.append(extract_char_ngrams(norm, n=3))
        
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
        
        cities.append(extract_city(a))
        states.append(extract_state(a))
        
    return (names_norm, names_clean, names_spaceless, name_words, first_words,
            addrs_norm, hns, pcs, addr_tokens, nums_list,
            phonetic_keys, cities, states, legal_suffixes, char_trigrams_name)

def _worker_wrapper_v3(args):
    return _batch_process_records_v3(*args)

def parallel_process_corpus_v3(names, addrs, n_workers=64):
    """Parallelized corpus preprocessing scaling to 64 workers."""
    n = len(names)
    if n == 0:
        return tuple([] for _ in range(15))
    if n <= 5000 or n_workers <= 1:
        return _batch_process_records_v3(names, addrs)
        
    chunk_size = max(2000, n // (n_workers * 4) + 1)
    chunks = [
        (names[i:i+chunk_size], addrs[i:i+chunk_size])
        for i in range(0, n, chunk_size)
    ]
    with mp.Pool(n_workers) as pool:
        results = pool.map(_worker_wrapper_v3, chunks)
        
    output = []
    for i in range(15):
        output.append([x for r in results for x in r[i]])
        
    return tuple(output)
