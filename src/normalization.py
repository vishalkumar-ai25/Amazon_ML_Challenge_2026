#!/usr/bin/env python3
"""
Universal Multilingual Normalizer for Entity Resolution
Amazon ML Challenge 2026

Handles:
1. NFKD Unicode accent stripping (handles French diacritics: é, è, ê, à, ç, etc.)
2. Indic script phonetic transliteration (Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, etc.)
3. Legal entity suffix expansion and standardization
4. Street and address keyword expansions
5. Domain and noise symbol removal
6. House number / premise and postal code extraction
"""

import re
import unicodedata
import pandas as pd

# Indic Unicode blocks -> Latin phonetic transliteration
INDIC_TO_LATIN = {
    # Devanagari (Hindi, Marathi, Nepali)
    0x0905: 'a', 0x0906: 'aa', 0x0907: 'i', 0x0908: 'ee', 0x0909: 'u', 0x090A: 'oo',
    0x090F: 'e', 0x0910: 'ai', 0x0913: 'o', 0x0914: 'au',
    0x0915: 'k', 0x0916: 'kh', 0x0917: 'g', 0x0918: 'gh', 0x0919: 'ng',
    0x091A: 'ch', 0x091B: 'chh', 0x091C: 'j', 0x091D: 'jh', 0x091E: 'ny',
    0x091F: 't', 0x0920: 'th', 0x0921: 'd', 0x0922: 'dh', 0x0923: 'n',
    0x0924: 't', 0x0925: 'th', 0x0926: 'd', 0x0927: 'dh', 0x0928: 'n',
    0x092A: 'p', 0x092B: 'ph', 0x092C: 'b', 0x092D: 'bh', 0x092E: 'm',
    0x092F: 'y', 0x0930: 'r', 0x0932: 'l', 0x0935: 'v',
    0x0936: 'sh', 0x0937: 'sh', 0x0938: 's', 0x0939: 'h',
    0x093E: 'aa', 0x093F: 'i', 0x0940: 'ee', 0x0941: 'u', 0x0942: 'oo',
    0x0947: 'e', 0x0948: 'ai', 0x094B: 'o', 0x094C: 'au', 0x094D: '', 0x0902: 'n', 0x0901: 'n',
    # Telugu
    0x0C05: 'a', 0x0C06: 'aa', 0x0C07: 'i', 0x0C08: 'ee', 0x0C09: 'u', 0x0C0A: 'oo',
    0x0C0E: 'e', 0x0C0F: 'ee', 0x0C10: 'ai', 0x0C12: 'o', 0x0C13: 'oo', 0x0C14: 'au',
    0x0C15: 'k', 0x0C16: 'kh', 0x0C17: 'g', 0x0C18: 'gh', 0x0C19: 'ng',
    0x0C1A: 'ch', 0x0C1B: 'chh', 0x0C1C: 'j', 0x0C1D: 'jh', 0x0C1E: 'ny',
    0x0C1F: 't', 0x0C20: 'th', 0x0C21: 'd', 0x0C22: 'dh', 0x0C23: 'n',
    0x0C24: 't', 0x0C25: 'th', 0x0C26: 'd', 0x0C27: 'dh', 0x0C28: 'n',
    0x0C2A: 'p', 0x0C2B: 'ph', 0x0C2C: 'b', 0x0C2D: 'bh', 0x0C2E: 'm',
    0x0C2F: 'y', 0x0C30: 'r', 0x0C31: 'r', 0x0C32: 'l', 0x0C35: 'v',
    0x0C36: 'sh', 0x0C37: 'sh', 0x0C38: 's', 0x0C39: 'h',
    0x0C3E: 'aa', 0x0C3F: 'i', 0x0C40: 'ee', 0x0C41: 'u', 0x0C42: 'oo',
    0x0C46: 'e', 0x0C47: 'ee', 0x0C48: 'ai', 0x0C4A: 'o', 0x0C4B: 'oo', 0x0C4C: 'au',
    0x0C4D: '', 0x0C02: 'n', 0x0C01: 'n'
}

# Street expansions (US, India, France)
STREET_EXPANSIONS = {
    r'\brd\b': 'road', r'\bdr\b': 'drive', r'\bst\b': 'street', r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard', r'\bpkwy\b': 'parkway', r'\bln\b': 'lane', r'\bct\b': 'court',
    r'\bhwy\b': 'highway', r'\br\.\b': 'rue', r'\brue\b': 'rue', r'\bbd\b': 'boulevard',
    r'\bav\b': 'avenue', r'\bimp\b': 'impasse', r'\bpl\b': 'place', r'\baly\b': 'alley',
    r'\bsq\b': 'square'
}

# Legal suffixes across US, India, France
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

DOMAIN_REGEX = re.compile(r'\.(com|in|org|net|co|io|biz|info|fr|gov|edu)\b', re.IGNORECASE)

def universal_normalize(text):
    """
    Universal normalizer:
    1. Transliterate Indic Unicode to Latin
    2. NFKD Unicode accent stripping (handles French diacritics)
    3. Strip noise prefixes, symbols, web domains
    4. Numeric leading zero stripping (0017 -> 17)
    5. Street, legal abbreviation expansion
    """
    if not text or pd.isna(text) or str(text).lower() in ['nan', 'null', 'none']:
        return ""
    text = str(text)
    
    # Transliterate Indic characters
    if any(0x0900 <= ord(c) <= 0x0D7F for c in text):
        res = []
        for ch in text:
            cp = ord(ch)
            if cp in INDIC_TO_LATIN:
                res.append(INDIC_TO_LATIN[cp])
            elif 0x0900 <= cp <= 0x0D7F:
                dev_cp = 0x0900 + (cp % 0x80)
                res.append(INDIC_TO_LATIN.get(dev_cp, ''))
            elif cp < 128:
                res.append(ch)
            else:
                res.append(' ')
        text = ''.join(res)
        
    # NFKD Unicode accent stripping
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower()
    
    # Strip noise prefixes and domains
    text = PREFIX_REGEX.sub('', text)
    text = DOMAIN_REGEX.sub('', text)
    
    # Remove non-alphanumeric (keep whitespace)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Strip leading zeros on numeric tokens
    text = re.sub(r'\b0+(\d+)\b', r'\1', text)
    
    # Expand street and legal terms
    for pat, rep in STREET_EXPANSIONS.items():
        text = re.sub(pat, rep, text)
    for pat, rep in LEGAL_SUFFIX_MAP.items():
        text = re.sub(pat, rep, text)
        
    return re.sub(r'\s+', ' ', text).strip()

def normalize_name_clean(text):
    """Deep name normalization: universal normalize + strip legal suffixes for matching."""
    norm = universal_normalize(text)
    # Strip expanded legal terms for pure core name comparison
    for term in ['private limited', 'limited liability partnership', 'limited liability',
                 'corporation', 'incorporated', 'private', 'limited', 'company',
                 'sarl', 'sas', 'sasu', 'eurl', 'sci']:
        norm = re.sub(r'\b' + term + r'\b', '', norm)
    return re.sub(r'\s+', ' ', norm).strip()

def extract_house_number(addr):
    """Extract premise / building number."""
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

import multiprocessing as mp

def _batch_process_records(chunk_names, chunk_addrs):
    names_norm, names_clean, name_words, first_words = [], [], [], []
    addrs_norm, hns, pcs, addr_tokens, nums_list = [], [], [], [], []
    
    for n in chunk_names:
        norm = universal_normalize(n)
        clean = normalize_name_clean(n)
        words = set(w for w in norm.split() if len(w) > 1)
        fw = norm.split()[0] if norm.split() else ""
        names_norm.append(norm)
        names_clean.append(clean)
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
        
    return (names_norm, names_clean, name_words, first_words,
            addrs_norm, hns, pcs, addr_tokens, nums_list)

def _worker_wrapper(args):
    return _batch_process_records(*args)

def parallel_process_corpus(names, addrs, n_workers=32):
    n = len(names)
    if n == 0:
        return ([], [], [], [], [], [], [], [], [])
    if n <= 10000 or n_workers <= 1:
        return _batch_process_records(names, addrs)
        
    chunk_size = max(5000, n // (n_workers * 4) + 1)
    chunks = [
        (names[i:i+chunk_size], addrs[i:i+chunk_size])
        for i in range(0, n, chunk_size)
    ]
    with mp.Pool(n_workers) as pool:
        results = pool.map(_worker_wrapper, chunks)
        
    names_norm = [x for r in results for x in r[0]]
    names_clean = [x for r in results for x in r[1]]
    name_words = [x for r in results for x in r[2]]
    first_words = [x for r in results for x in r[3]]
    addrs_norm = [x for r in results for x in r[4]]
    hns = [x for r in results for x in r[5]]
    pcs = [x for r in results for x in r[6]]
    addr_tokens = [x for r in results for x in r[7]]
    nums_list = [x for r in results for x in r[8]]
    
    return (names_norm, names_clean, name_words, first_words,
            addrs_norm, hns, pcs, addr_tokens, nums_list)

