#!/usr/bin/env python3
"""
7-Channel Multi-Index Candidate Generation Engine v3
Amazon ML Challenge 2026

Channels:
1. Word Composite TF-IDF (2*Name + Address)
2. Address Binary Token Jaccard Matrix
3. Char 2-5 Subword N-Gram TF-IDF (Phonetic & Typos via AnyAscii)
4. Sorted-Token Word TF-IDF (Word Permutation Invariance)
5. Spaceless Domain / Website Exact Index + Postal Inverted Index
6. Phonetic Key Inverted Index
7. First-3-char Inverted Index
"""

import time
from collections import defaultdict
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class MultiChannelBlockingEngineV3:
    def __init__(self, top_k_per_channel=50, max_union=100):
        self.top_k = top_k_per_channel
        self.max_union = max_union
        self.postal_index = {}
        self.spaceless_index = defaultdict(list)
        self.phonetic_index = defaultdict(list)
        self.first3_index = defaultdict(list)

    def fit_and_transform_corpus(
        self,
        s1_names, s1_addrs, s1_spaceless,
        c_names, c_addrs, c_spaceless, c_pcs, c_hns,
        c_phonetics=None, c_first3=None
    ):
        t0 = time.time()
        n_s1 = len(s1_names)
        n_c = len(c_names)

        # Composite text: 2*name + address
        s1_comp = [(f"{n} {n} {a}").strip() for n, a in zip(s1_names, s1_addrs)]
        c_comp = [(f"{n} {n} {a}").strip() for n, a in zip(c_names, c_addrs)]

        # Sorted names
        s1_sort_names = [' '.join(sorted(n.split())) for n in s1_names]
        c_sort_names = [' '.join(sorted(n.split())) for n in c_names]

        # -------------------------------------------------------------
        # Channel 1: Word Composite TF-IDF
        # -------------------------------------------------------------
        print("  [Blocking v3] Channel 1: Fitting Word Composite TF-IDF...", flush=True)
        t_ch = time.time()
        sample_w = np.concatenate([s1_comp[:min(100000, n_s1)], c_comp[:min(400000, n_c)]])
        max_df_w = max(0.03, 10.0 / max(1, len(sample_w))) if len(sample_w) < 500 else 0.03
        self.vec_w = TfidfVectorizer(
            analyzer='word', max_features=150000, min_df=min(2, max(1, len(sample_w))), max_df=min(1.0, max_df_w),
            sublinear_tf=True, norm='l2', dtype=np.float32, token_pattern=r'(?u)\b\w+\b'
        )
        self.vec_w.fit(sample_w)
        del sample_w

        self.s1_mat_w = self.vec_w.transform(s1_comp)
        self.c_mat_w_T = self.vec_w.transform(c_comp).T.tocsc()
        print(f"    Done in {time.time()-t_ch:.1f}s")

        # -------------------------------------------------------------
        # Channel 2: Address Binary Jaccard
        # -------------------------------------------------------------
        print("  [Blocking v3] Channel 2: Fitting Address Binary Jaccard...", flush=True)
        t_ch = time.time()
        sample_a = np.concatenate([s1_addrs[:min(100000, n_s1)], c_addrs[:min(400000, n_c)]])
        max_df_a = max(0.04, 10.0 / max(1, len(sample_a))) if len(sample_a) < 500 else 0.04
        self.vec_a = CountVectorizer(
            binary=True, analyzer='word', token_pattern=r'(?u)\b\w+\b',
            min_df=min(2, max(1, len(sample_a))), max_df=min(1.0, max_df_a), max_features=80000
        )
        self.vec_a.fit(sample_a)
        del sample_a

        self.s1_mat_a = self.vec_a.transform(s1_addrs)
        self.c_mat_a_T = self.vec_a.transform(c_addrs).T.tocsc()
        self.len_s1_a = np.diff(self.s1_mat_a.indptr)
        self.len_c_a = np.diff(self.c_mat_a_T.indptr)
        print(f"    Done in {time.time()-t_ch:.1f}s")

        # -------------------------------------------------------------
        # Channel 3: Char 2-5 Subword N-Gram TF-IDF
        # -------------------------------------------------------------
        print("  [Blocking v3] Channel 3: Fitting Char 2-5 Subword N-Gram...", flush=True)
        t_ch = time.time()
        sample_c = np.concatenate([s1_names[:min(100000, n_s1)], c_names[:min(400000, n_c)]])
        max_df_c = max(0.01, 10.0 / max(1, len(sample_c))) if len(sample_c) < 1000 else 0.01
        self.vec_c = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 5), max_features=80000,
            min_df=min(2, max(1, len(sample_c))), max_df=min(1.0, max_df_c),
            sublinear_tf=True, norm='l2', dtype=np.float32
        )
        self.vec_c.fit(sample_c)
        del sample_c

        self.s1_mat_c = self.vec_c.transform(s1_names)
        self.c_mat_c_T = self.vec_c.transform(c_names).T.tocsc()
        print(f"    Done in {time.time()-t_ch:.1f}s")

        # -------------------------------------------------------------
        # Channel 4: Sorted-Token Word TF-IDF on Name
        # -------------------------------------------------------------
        print("  [Blocking v3] Channel 4: Fitting Sorted-Token Word TF-IDF...", flush=True)
        t_ch = time.time()
        sample_s = np.concatenate([s1_sort_names[:min(100000, n_s1)], c_sort_names[:min(400000, n_c)]])
        max_df_s = max(0.04, 10.0 / max(1, len(sample_s))) if len(sample_s) < 500 else 0.04
        self.vec_s = TfidfVectorizer(
            analyzer='word', max_features=60000,
            min_df=min(2, max(1, len(sample_s))), max_df=min(1.0, max_df_s),
            sublinear_tf=True, norm='l2', dtype=np.float32, token_pattern=r'(?u)\b\w+\b'
        )
        self.vec_s.fit(sample_s)
        del sample_s

        self.s1_mat_s = self.vec_s.transform(s1_sort_names)
        self.c_mat_s_T = self.vec_s.transform(c_sort_names).T.tocsc()
        print(f"    Done in {time.time()-t_ch:.1f}s")

        # -------------------------------------------------------------
        # Channel 5: Spaceless Domain Index & Postal Inverted Index
        # -------------------------------------------------------------
        print("  [Blocking v3] Channel 5: Building Domain & Postal Inverted Indexes...", flush=True)
        t_ch = time.time()
        
        # Build Spaceless Name Index (keys of length >= 4)
        self.spaceless_index = defaultdict(list)
        for j, sp in enumerate(c_spaceless):
            if sp and len(sp) >= 4:
                self.spaceless_index[sp].append(j)
        # Cap high-frequency collision buckets (> 200 records)
        self.spaceless_index = {k: v for k, v in self.spaceless_index.items() if len(v) <= 200}
        
        # Build Postal Index
        self.postal_index = defaultdict(list)
        for j, pc in enumerate(c_pcs):
            if pc:
                self.postal_index[pc].append(j)
        # Filter overly dense buckets (> 2000 records)
        self.postal_index = {k: v for k, v in self.postal_index.items() if len(v) <= 2000}
        
        print(f"    Indexed {len(self.spaceless_index):,} spaceless keys & {len(self.postal_index):,} postal clusters in {time.time()-t_ch:.1f}s")

        # -------------------------------------------------------------
        # Channel 6: Phonetic Key Inverted Index
        # -------------------------------------------------------------
        if c_phonetics is not None:
            print("  [Blocking v3] Channel 6: Building Phonetic Index...", flush=True)
            t_ch = time.time()
            self.phonetic_index = defaultdict(list)
            for j, pks in enumerate(c_phonetics):
                if not pks: continue
                if isinstance(pks, str):
                    pks = [pks]
                for pk in pks:
                    if pk:
                        self.phonetic_index[pk].append(j)
            self.phonetic_index = {k: v for k, v in self.phonetic_index.items() if len(v) <= 500}
            print(f"    Indexed {len(self.phonetic_index):,} phonetic keys in {time.time()-t_ch:.1f}s")

        # -------------------------------------------------------------
        # Channel 7: First-3-char Inverted Index
        # -------------------------------------------------------------
        if c_first3 is not None:
            print("  [Blocking v3] Channel 7: Building First-3 Index...", flush=True)
            t_ch = time.time()
            self.first3_index = defaultdict(list)
            for j, f3 in enumerate(c_first3):
                if f3:
                    self.first3_index[f3].append(j)
            self.first3_index = {k: v for k, v in self.first3_index.items() if len(v) <= 300}
            print(f"    Indexed {len(self.first3_index):,} first-3 keys in {time.time()-t_ch:.1f}s")
            
        print(f"  [Blocking v3] All channels fitted in {time.time()-t0:.1f}s")

    def retrieve_candidates_for_batch(
        self, b_start, b_end, s1_pcs, s1_spaceless, s1_name_words, c_name_words,
        s1_phonetics=None, s1_first3=None
    ):
        """
        Retrieves candidates for S1 queries in range [b_start, b_end).
        Returns list of candidate dicts: {cidx: [score_w, score_a, score_c, score_s, score_5, score_6, score_7]}
        """
        sims_w = (self.s1_mat_w[b_start:b_end] @ self.c_mat_w_T).tocsr()
        inter_a = (self.s1_mat_a[b_start:b_end] @ self.c_mat_a_T).tocsr()
        sims_c = (self.s1_mat_c[b_start:b_end] @ self.c_mat_c_T).tocsr()
        sims_s = (self.s1_mat_s[b_start:b_end] @ self.c_mat_s_T).tocsr()

        batch_candidates = []
        n_queries = b_end - b_start
        k = self.top_k

        for i in range(n_queries):
            g_i = b_start + i
            cands = {}

            # Channel 1: Word TF-IDF
            p0, p1 = sims_w.indptr[i], sims_w.indptr[i+1]
            if p0 < p1:
                data = sims_w.data[p0:p1]
                idx = sims_w.indices[p0:p1]
                top_k = np.argpartition(data, -min(k, len(data)))[-min(k, len(data)):]
                for t in top_k:
                    cidx = idx[t]
                    cands[cidx] = [float(data[t]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            # Channel 2: Address Jaccard
            p0, p1 = inter_a.indptr[i], inter_a.indptr[i+1]
            if p0 < p1 and self.len_s1_a[g_i] > 0:
                data = inter_a.data[p0:p1]
                idx = inter_a.indices[p0:p1]
                denoms = self.len_s1_a[g_i] + self.len_c_a[idx] - data
                jacc = data / denoms
                top_k = np.argpartition(jacc, -min(k, len(jacc)))[-min(k, len(jacc)):]
                for t in top_k:
                    cidx = idx[t]
                    if cidx not in cands:
                        cands[cidx] = [0.0, float(jacc[t]), 0.0, 0.0, 0.0, 0.0, 0.0]
                    else:
                        cands[cidx][1] = float(jacc[t])

            # Channel 3: Char Subword N-Grams
            p0, p1 = sims_c.indptr[i], sims_c.indptr[i+1]
            if p0 < p1:
                data = sims_c.data[p0:p1]
                idx = sims_c.indices[p0:p1]
                top_k = np.argpartition(data, -min(k, len(data)))[-min(k, len(data)):]
                for t in top_k:
                    cidx = idx[t]
                    if cidx not in cands:
                        cands[cidx] = [0.0, 0.0, float(data[t]), 0.0, 0.0, 0.0, 0.0]
                    else:
                        cands[cidx][2] = float(data[t])

            # Channel 4: Sorted-Token TF-IDF
            p0, p1 = sims_s.indptr[i], sims_s.indptr[i+1]
            if p0 < p1:
                data = sims_s.data[p0:p1]
                idx = sims_s.indices[p0:p1]
                top_k = np.argpartition(data, -min(k, len(data)))[-min(k, len(data)):]
                for t in top_k:
                    cidx = idx[t]
                    if cidx not in cands:
                        cands[cidx] = [0.0, 0.0, 0.0, float(data[t]), 0.0, 0.0, 0.0]
                    else:
                        cands[cidx][3] = float(data[t])

            # Pre-fetch S1 words for token overlap checks
            s1_words = s1_name_words[g_i] if g_i < len(s1_name_words) else set()

            # Channel 5A: Exact Spaceless / Domain Name Match
            s1_sp = s1_spaceless[g_i] if g_i < len(s1_spaceless) else ""
            if s1_sp and len(s1_sp) >= 4:
                if s1_sp in self.spaceless_index:
                    for cidx in self.spaceless_index[s1_sp]:
                        if cidx not in cands:
                            cands[cidx] = [0.85, 0.0, 0.85, 0.85, 0.85, 0.0, 0.0]
                        else:
                            cands[cidx][4] = max(cands[cidx][4], 0.85)

            # Channel 5B: Postal Code Match with Token Overlap
            pc = s1_pcs[g_i] if g_i < len(s1_pcs) else ""
            if pc and pc in self.postal_index:
                for cidx in self.postal_index[pc]:
                    c_words = c_name_words[cidx] if cidx < len(c_name_words) else set()
                    if s1_words and c_words and len(s1_words & c_words) > 0:
                        if cidx not in cands:
                            cands[cidx] = [0.40, 0.30, 0.40, 0.40, 0.40, 0.0, 0.0]
                        else:
                            cands[cidx][4] = max(cands[cidx][4], 0.40)

            # Channel 6: Phonetic Key Match with Token Overlap
            if s1_phonetics is not None and g_i < len(s1_phonetics):
                pks = s1_phonetics[g_i]
                if pks:
                    if isinstance(pks, str):
                        pks = [pks]
                    for pk in pks:
                        if pk in self.phonetic_index:
                            for cidx in self.phonetic_index[pk]:
                                c_words = c_name_words[cidx] if cidx < len(c_name_words) else set()
                                if s1_words and c_words and len(s1_words & c_words) > 0:
                                    if cidx not in cands:
                                        cands[cidx] = [0.40, 0.0, 0.40, 0.0, 0.0, 0.85, 0.0]
                                    else:
                                        cands[cidx][5] = 0.85

            # Channel 7: First-3-char Match (scored by name token Jaccard)
            if s1_first3 is not None and g_i < len(s1_first3):
                f3 = s1_first3[g_i]
                if f3 and f3 in self.first3_index:
                    for cidx in self.first3_index[f3]:
                        c_words = c_name_words[cidx] if cidx < len(c_name_words) else set()
                        overlap = len(s1_words & c_words)
                        union = len(s1_words | c_words)
                        jacc = overlap / union if union > 0 else 0.0
                        if jacc > 0:
                            if cidx not in cands:
                                cands[cidx] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, jacc]
                            else:
                                cands[cidx][6] = max(cands[cidx][6], jacc)

            # Sort by highest individual channel score and cap at max_union
            sorted_cands = sorted(
                cands.items(),
                key=lambda x: -max(x[1])
            )[:self.max_union]

            batch_candidates.append(sorted_cands)

        return batch_candidates
