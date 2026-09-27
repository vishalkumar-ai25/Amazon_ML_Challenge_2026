#!/usr/bin/env python3
"""
5-Channel Multi-Index Candidate Generation Engine v2
Amazon ML Challenge 2026

Channels:
1. Word Composite TF-IDF (2*Name + Address)
2. Address Binary Token Jaccard Matrix
3. Character 3-4 Subword N-Gram TF-IDF (Phonetic & Typos via AnyAscii)
4. Sorted-Token Word TF-IDF (Word Permutation Invariance)
5. Spaceless Domain / Website Exact Index + Postal Inverted Index
"""

import time
from collections import defaultdict
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class MultiChannelBlockingEngineV2:
    def __init__(self, top_k_per_channel=40, max_union=80):
        self.top_k = top_k_per_channel
        self.max_union = max_union
        self.postal_index = {}
        self.spaceless_index = defaultdict(list)

    def fit_and_transform_corpus(
        self,
        s1_names, s1_addrs, s1_spaceless,
        c_names, c_addrs, c_spaceless, c_pcs, c_hns
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
        print("  [Blocking v2] Channel 1: Fitting Word Composite TF-IDF...", flush=True)
        t_ch = time.time()
        self.vec_w = TfidfVectorizer(
            analyzer='word', max_features=120000, min_df=2, max_df=0.03,
            sublinear_tf=True, norm='l2', dtype=np.float32, token_pattern=r'(?u)\b\w+\b'
        )
        sample_w = np.concatenate([s1_comp[:min(100000, n_s1)], c_comp[:min(400000, n_c)]])
        self.vec_w.fit(sample_w)
        del sample_w

        self.s1_mat_w = self.vec_w.transform(s1_comp)
        self.c_mat_w_T = self.vec_w.transform(c_comp).T.tocsc()
        print(f"    Done in {time.time()-t_ch:.1f}s")

        # -------------------------------------------------------------
        # Channel 2: Address Binary Jaccard
        # -------------------------------------------------------------
        print("  [Blocking v2] Channel 2: Fitting Address Binary Jaccard...", flush=True)
        t_ch = time.time()
        self.vec_a = CountVectorizer(
            binary=True, analyzer='word', token_pattern=r'(?u)\b\w+\b',
            min_df=2, max_df=0.04, max_features=80000
        )
        sample_a = np.concatenate([s1_addrs[:min(100000, n_s1)], c_addrs[:min(400000, n_c)]])
        self.vec_a.fit(sample_a)
        del sample_a

        self.s1_mat_a = self.vec_a.transform(s1_addrs)
        self.c_mat_a_T = self.vec_a.transform(c_addrs).T.tocsc()
        self.len_s1_a = np.diff(self.s1_mat_a.indptr)
        self.len_c_a = np.diff(self.c_mat_a_T.indptr)
        print(f"    Done in {time.time()-t_ch:.1f}s")

        # -------------------------------------------------------------
        # Channel 3: Char 3-4 Subword N-Gram TF-IDF (Phonetic AnyAscii)
        # -------------------------------------------------------------
        print("  [Blocking v2] Channel 3: Fitting Char 3-4 Subword N-Gram...", flush=True)
        t_ch = time.time()
        self.vec_c = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 4), max_features=50000, min_df=5, max_df=0.01,
            sublinear_tf=True, norm='l2', dtype=np.float32
        )
        sample_c = np.concatenate([s1_names[:min(100000, n_s1)], c_names[:min(400000, n_c)]])
        self.vec_c.fit(sample_c)
        del sample_c

        self.s1_mat_c = self.vec_c.transform(s1_names)
        self.c_mat_c_T = self.vec_c.transform(c_names).T.tocsc()
        print(f"    Done in {time.time()-t_ch:.1f}s")

        # -------------------------------------------------------------
        # Channel 4: Sorted-Token Word TF-IDF on Name
        # -------------------------------------------------------------
        print("  [Blocking v2] Channel 4: Fitting Sorted-Token Word TF-IDF...", flush=True)
        t_ch = time.time()
        self.vec_s = TfidfVectorizer(
            analyzer='word', max_features=60000, min_df=2, max_df=0.04,
            sublinear_tf=True, norm='l2', dtype=np.float32, token_pattern=r'(?u)\b\w+\b'
        )
        sample_s = np.concatenate([s1_sort_names[:min(100000, n_s1)], c_sort_names[:min(400000, n_c)]])
        self.vec_s.fit(sample_s)
        del sample_s

        self.s1_mat_s = self.vec_s.transform(s1_sort_names)
        self.c_mat_s_T = self.vec_s.transform(c_sort_names).T.tocsc()
        print(f"    Done in {time.time()-t_ch:.1f}s")

        # -------------------------------------------------------------
        # Channel 5: Spaceless Domain Index & Postal Inverted Index
        # -------------------------------------------------------------
        print("  [Blocking v2] Channel 5: Building Domain & Postal Inverted Indexes...", flush=True)
        t_ch = time.time()
        
        # Build Spaceless Name Index (keys of length >= 4)
        self.spaceless_index = defaultdict(list)
        for j, sp in enumerate(c_spaceless):
            if len(sp) >= 4:
                self.spaceless_index[sp].append(j)
        # Cap high-frequency collision buckets (> 50 records)
        self.spaceless_index = {k: v for k, v in self.spaceless_index.items() if len(v) <= 50}
        
        # Build Postal Index
        self.postal_index = defaultdict(list)
        for j, pc in enumerate(c_pcs):
            if pc:
                self.postal_index[pc].append(j)
        # Filter overly dense buckets (> 1000 records)
        self.postal_index = {k: v for k, v in self.postal_index.items() if len(v) <= 1000}
        
        print(f"    Indexed {len(self.spaceless_index):,} spaceless keys & {len(self.postal_index):,} postal clusters in {time.time()-t_ch:.1f}s")
        print(f"  [Blocking v2] All 5 channels fitted in {time.time()-t0:.1f}s")

    def retrieve_candidates_for_batch(
        self, b_start, b_end, s1_pcs, s1_spaceless, s1_name_words, c_name_words
    ):
        """
        Retrieves candidates for S1 queries in range [b_start, b_end).
        Returns list of candidate dicts: {cidx: [score_w, score_a, score_c, score_s]}
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
                    cands[cidx] = [float(data[t]), 0.0, 0.0, 0.0]

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
                        cands[cidx] = [0.0, float(jacc[t]), 0.0, 0.0]
                    else:
                        cands[cidx][1] = float(jacc[t])

            # Channel 3: Char 3-4 Subword N-Grams
            p0, p1 = sims_c.indptr[i], sims_c.indptr[i+1]
            if p0 < p1:
                data = sims_c.data[p0:p1]
                idx = sims_c.indices[p0:p1]
                top_k = np.argpartition(data, -min(k, len(data)))[-min(k, len(data)):]
                for t in top_k:
                    cidx = idx[t]
                    if cidx not in cands:
                        cands[cidx] = [0.0, 0.0, float(data[t]), 0.0]
                    else:
                        cands[cidx][2] = float(data[t])

            # Channel 4: Sorted-Token TF-IDF
            p0, p1 = sims_s.indptr[i], sims_s.indptr[i+1]
            if p0 < p1:
                data = sims_s.data[p0:p1]
                idx = sims_s.indices[p0:p1]
                top_k = np.argpartition(data, -min(25, len(data)))[-min(25, len(data)):]
                for t in top_k:
                    cidx = idx[t]
                    if cidx not in cands:
                        cands[cidx] = [0.0, 0.0, 0.0, float(data[t])]
                    else:
                        cands[cidx][3] = float(data[t])

            # Channel 5A: Exact Spaceless / Domain Name Match
            s1_sp = s1_spaceless[g_i]
            if s1_sp and len(s1_sp) >= 4:
                if s1_sp in self.spaceless_index:
                    for cidx in self.spaceless_index[s1_sp]:
                        if cidx not in cands:
                            cands[cidx] = [0.85, 0.0, 0.85, 0.85]
                        else:
                            cands[cidx][0] = max(cands[cidx][0], 0.85)

            # Channel 5B: Postal Code Match with Token Overlap
            pc = s1_pcs[g_i]
            if pc and pc in self.postal_index:
                s1_words = s1_name_words[g_i]
                for cidx in self.postal_index[pc]:
                    if cidx not in cands and s1_words and c_name_words[cidx]:
                        if len(s1_words & c_name_words[cidx]) > 0:
                            cands[cidx] = [0.40, 0.30, 0.40, 0.40]

            # Sort by highest channel score and cap at max_union
            sorted_cands = sorted(
                cands.items(),
                key=lambda x: -(max(x[1][0], x[1][1], x[1][2], x[1][3]))
            )[:self.max_union]

            batch_candidates.append(sorted_cands)

        return batch_candidates
