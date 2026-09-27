#!/usr/bin/env python3
"""
50-Feature Dense Vectorized Pairwise Extractor v3
Amazon ML Challenge 2026

Extracts 50 dense features for business entity resolution.
"""

import numpy as np
from rapidfuzz import fuzz, distance

FEATURE_NAMES = [
    # 1-12: Existing name features
    'name_ratio', 'name_token_sort_ratio', 'name_token_set_ratio', 'name_partial_ratio',
    'name_levenshtein_sim', 'name_jaro_winkler', 'name_word_jaccard', 'name_containment',
    'name_len_ratio', 'name_word_diff', 'name_first_word_match', 'name_clean_ratio',
    # 13-18: Existing address features
    'addr_ratio', 'addr_token_sort_ratio', 'addr_token_set_ratio', 'addr_levenshtein_sim',
    'addr_word_jaccard', 'addr_containment',
    # 19-24: Existing structural features
    'house_number_match', 'postal_code_match', 'numeric_overlap',
    'both_addr_missing', 'one_addr_missing', 'name_in_cand_addr',
    # 25-32: Existing channel/rank features  
    'tfidf_word_score', 'addr_jaccard_score', 'tfidf_char_score', 'tfidf_sorted_score',
    'max_channel_score', 'mean_channel_score', 'cand_rank_inv', 'is_source3',
    # 33-35: Existing spaceless/premise features
    'spaceless_exact_match', 'spaceless_containment', 'hn_exact_clean',
    # 36-50: NEW FEATURES
    'name_bigram_jaccard',       # char bigram Jaccard on name
    'name_trigram_jaccard',      # char trigram Jaccard on name  
    'addr_trigram_jaccard',      # char trigram Jaccard on address
    'city_match',                # extracted city exact match
    'state_match',               # extracted state/region match
    'name_prefix_match_3',       # first 3 chars of cleaned name match
    'name_prefix_match_5',       # first 5 chars of cleaned name match
    'addr_numeric_token_jaccard', # Jaccard on numeric-only address tokens
    'name_common_word_count',    # count of shared words
    'addr_common_word_count',    # count of shared address words
    'name_unique_word_ratio',    # ratio of non-shared to total words
    'combined_name_addr_score',  # 0.6*name_tsr + 0.4*addr_jaccard
    'legal_suffix_match',        # legal suffixes match (LLC<->LLC etc)
    'phonetic_key_overlap',      # fraction of matching phonetic keys
    'n_blocking_channels_hit',   # how many blocking channels found this pair
]

def _get_bigrams(s):
    if not s: return set()
    return set(s[i:i+2] for i in range(len(s)-1))

def _get_trigrams(s):
    if not s: return set()
    return set(s[i:i+3] for i in range(len(s)-2))

def extract_pairwise_features_v3(s1_data, cand_data, channel_scores, rank, is_s3):
    """
    Computes all 50 features between S1 and candidate record.
    s1_data: (name_raw, name_clean, name_spaceless, name_words, first_word,
              addr_raw, addr_tokens, hn, pc, nums,
              phonetic_keys, city, state, legal_suffix, char_trigrams)
    cand_data: (name_raw, name_clean, name_spaceless, name_words, first_word,
                addr_raw, addr_tokens, hn, pc, nums,
                phonetic_keys, city, state, legal_suffix, char_trigrams)
    channel_scores: (score_word, score_addr, score_char, score_sorted, score_phonetic, score_prefix, score_spaceless)
    """
    (s1_nr, s1_nc, s1_sp, s1_nw, s1_fw, s1_ar, s1_at, s1_hn, s1_pc, s1_nums, s1_pk, s1_city, s1_state, s1_ls, s1_ct) = s1_data
    (c_nr, c_nc, c_sp, c_nw, c_fw, c_ar, c_at, c_hn, c_pc, c_nums, c_pk, c_city, c_state, c_ls, c_ct) = cand_data
    (sc_w, sc_a, sc_c, sc_s, sc_ph, sc_pr, sc_sp) = channel_scores

    # 1-12: Name Features
    if s1_nr and c_nr:
        name_ratio = fuzz.ratio(s1_nr, c_nr) / 100.0
        name_tsr = fuzz.token_sort_ratio(s1_nr, c_nr) / 100.0
        name_tset = fuzz.token_set_ratio(s1_nr, c_nr) / 100.0
        name_pr = fuzz.partial_ratio(s1_nr, c_nr) / 100.0
        name_lev = 1.0 - distance.Levenshtein.normalized_distance(s1_nr, c_nr)
        name_jw = distance.JaroWinkler.similarity(s1_nr, c_nr)
    else:
        name_ratio = name_tsr = name_tset = name_pr = name_lev = name_jw = 0.0

    n_inter = len(s1_nw & c_nw)
    n_union = len(s1_nw | c_nw)
    name_jaccard = n_inter / n_union if n_union > 0 else 0.0
    min_name = min(len(s1_nw), len(c_nw))
    name_contain = n_inter / min_name if min_name > 0 else 0.0
    name_len_ratio = min(len(s1_nr), len(c_nr)) / max(1, max(len(s1_nr), len(c_nr))) if s1_nr or c_nr else 0.0
    name_word_diff = float(abs(len(s1_nw) - len(c_nw)))
    first_match = 1.0 if s1_fw and s1_fw == c_fw else 0.0

    if s1_nc and c_nc:
        name_clean_ratio = fuzz.token_sort_ratio(s1_nc, c_nc) / 100.0
    else:
        name_clean_ratio = name_tsr

    # 13-18: Address Features
    if s1_ar and c_ar:
        addr_ratio = fuzz.ratio(s1_ar, c_ar) / 100.0
        addr_tsr = fuzz.token_sort_ratio(s1_ar, c_ar) / 100.0
        addr_tset = fuzz.token_set_ratio(s1_ar, c_ar) / 100.0
        addr_lev = 1.0 - distance.Levenshtein.normalized_distance(s1_ar, c_ar)
    else:
        addr_ratio = addr_tsr = addr_tset = addr_lev = 0.0

    a_inter = len(s1_at & c_at)
    a_union = len(s1_at | c_at)
    addr_jaccard = a_inter / a_union if a_union > 0 else 0.0
    min_addr = min(len(s1_at), len(c_at))
    addr_contain = a_inter / min_addr if min_addr > 0 else 0.0

    # 19-21: House number, Postal code, Numerics
    if s1_hn and c_hn:
        house_match = 1.0 if (s1_hn == c_hn or s1_hn in c_hn or c_hn in s1_hn) else -1.0
    else:
        house_match = 0.0

    if s1_pc and c_pc:
        postal_match = 1.0 if s1_pc == c_pc else -1.0
    else:
        postal_match = 0.0

    num_inter = len(s1_nums & c_nums)
    num_union = len(s1_nums | c_nums)
    numeric_overlap = num_inter / num_union if num_union > 0 else 0.0

    # 22-24: Missing flags and cross-field overlap
    both_missing = 1.0 if (not s1_ar and not c_ar) else 0.0
    one_missing = 1.0 if (bool(s1_ar) != bool(c_ar)) else 0.0
    name_in_cand_addr = len(s1_nw & c_at) / max(1, len(s1_nw)) if s1_nw else 0.0

    # 25-32: Channel Scores and Structural (Wait, keeping exactly 35 existing features mapping)
    # The existing features uses sc_w, sc_a, sc_c, sc_s for tfidf... and max/mean
    scores_4 = [sc_w, sc_a, sc_c, sc_s]
    max_score = max(scores_4)
    non_zeros_4 = [s for s in scores_4 if s > 0.0]
    mean_score = sum(non_zeros_4) / len(non_zeros_4) if non_zeros_4 else 0.0
    rank_inv = 1.0 / rank if rank > 0 else 0.0
    is_s3_flag = float(is_s3)

    # 33-35: Spaceless & Premise Features
    sp_exact = 1.0 if (s1_sp and c_sp and s1_sp == c_sp) else 0.0
    sp_contain = 1.0 if (s1_sp and c_sp and len(s1_sp) >= 5 and len(c_sp) >= 5 and (s1_sp in c_sp or c_sp in s1_sp)) else 0.0
    hn_exact = 1.0 if (s1_hn and c_hn and s1_hn == c_hn) else 0.0

    # 36-50: NEW FEATURES
    # name_bigram_jaccard
    s1_nb = _get_bigrams(s1_nc)
    c_nb = _get_bigrams(c_nc)
    nb_union = len(s1_nb | c_nb)
    name_bigram_jaccard = len(s1_nb & c_nb) / nb_union if nb_union > 0 else 0.0

    # name_trigram_jaccard
    nt_union = len(s1_ct | c_ct)
    name_trigram_jaccard = len(s1_ct & c_ct) / nt_union if nt_union > 0 else 0.0

    # addr_trigram_jaccard
    s1_at_tri = _get_trigrams(s1_ar)
    c_at_tri = _get_trigrams(c_ar)
    at_union = len(s1_at_tri | c_at_tri)
    addr_trigram_jaccard = len(s1_at_tri & c_at_tri) / at_union if at_union > 0 else 0.0

    # city_match
    city_match = 1.0 if (s1_city and c_city and s1_city == c_city) else 0.0

    # state_match
    state_match = 1.0 if (s1_state and c_state and s1_state == c_state) else 0.0

    # name_prefix_match_3 & 5
    name_prefix_match_3 = 1.0 if (s1_nc and c_nc and len(s1_nc) >= 3 and len(c_nc) >= 3 and s1_nc[:3] == c_nc[:3]) else 0.0
    name_prefix_match_5 = 1.0 if (s1_nc and c_nc and len(s1_nc) >= 5 and len(c_nc) >= 5 and s1_nc[:5] == c_nc[:5]) else 0.0

    # addr_numeric_token_jaccard
    s1_at_num = {t for t in s1_at if t.isdigit()}
    c_at_num = {t for t in c_at if t.isdigit()}
    at_num_union = len(s1_at_num | c_at_num)
    addr_numeric_token_jaccard = len(s1_at_num & c_at_num) / at_num_union if at_num_union > 0 else 0.0

    # name_common_word_count
    name_common_word_count = float(n_inter)

    # addr_common_word_count
    addr_common_word_count = float(a_inter)

    # name_unique_word_ratio
    name_unique_word_ratio = (n_union - n_inter) / n_union if n_union > 0 else 0.0

    # combined_name_addr_score
    combined_name_addr_score = 0.6 * name_tsr + 0.4 * addr_jaccard

    # legal_suffix_match
    legal_suffix_match = 1.0 if (s1_ls and c_ls and s1_ls == c_ls) else 0.0

    # phonetic_key_overlap
    pk_union = len(s1_pk | c_pk)
    phonetic_key_overlap = len(s1_pk & c_pk) / pk_union if pk_union > 0 else 0.0

    # n_blocking_channels_hit
    all_scores = [sc_w, sc_a, sc_c, sc_s, sc_ph, sc_pr, sc_sp]
    n_blocking_channels_hit = float(sum(1 for s in all_scores if s > 0.0))

    return [
        # 1-12
        name_ratio, name_tsr, name_tset, name_pr,
        name_lev, name_jw, name_jaccard, name_contain,
        name_len_ratio, name_word_diff, first_match, name_clean_ratio,
        # 13-18
        addr_ratio, addr_tsr, addr_tset, addr_lev,
        addr_jaccard, addr_contain,
        # 19-24
        house_match, postal_match, numeric_overlap,
        both_missing, one_missing, name_in_cand_addr,
        # 25-32
        sc_w, sc_a, sc_c, sc_s,
        max_score, mean_score, rank_inv, is_s3_flag,
        # 33-35
        sp_exact, sp_contain, hn_exact,
        # 36-50
        name_bigram_jaccard, name_trigram_jaccard, addr_trigram_jaccard,
        city_match, state_match,
        name_prefix_match_3, name_prefix_match_5,
        addr_numeric_token_jaccard,
        name_common_word_count, addr_common_word_count,
        name_unique_word_ratio,
        combined_name_addr_score,
        legal_suffix_match,
        phonetic_key_overlap,
        n_blocking_channels_hit
    ]
