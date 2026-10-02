# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** EnsembleGrandmaster  
**Team Members:** Vishal Kumar  
**Submission Date:** October 1, 2026

---

## 1. Executive Summary

In large-scale commercial e-commerce platforms, entity records arrive asynchronously across heterogeneous data sources with no shared primary keys, severe typographical divergence, multi-script variations, and incomplete address fragments. We formulated business entity resolution across ~28 million records as a multi-stage retrieval, supervised reranking, and precision-hardened bipartite resolution problem evaluated under macro-averaged $F_{0.5}$. Our architecture couples a high-recall multi-channel blocking engine (>96% pair completeness) with a 50+ feature LightGBM classifier and a specialized Precision Hardening Engine (address premise normalization, postal validation, and chain-store disambiguation). The pipeline achieves a verified leaderboard macro $F_{0.5}$ score of **0.9142**, running strictly within memory and operational bounds while generalizing zero-shot to the unseen France country partition.

---

## 2. Methodology

### 2.1 Problem Analysis

A comprehensive forensic audit of the dataset (~2.2M Source 1 reference entities, ~5.0M Source 2 records, ~5.3M Source 3 records) revealed key structural challenges that governed our design:

1. **Metric Asymmetry ($F_{0.5}$):** The competition metric is $F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$. Because precision is weighted $4\times$ heavier than recall in the denominator ($1/\beta^2 = 4$), a single false positive merge penalizes the score drastically more than a missed link.
2. **Cardinality & Singleton Penalty:** Approximately 5.6% of Source 1 entities in the ground truth are strict singletons (zero matching records). Singletons correctly identified as empty yield a perfect 1.0 macro score, whereas assigning even one spurious match collapses that entity's score to 0.0.
3. **Many-to-One Physical Constraint:** In the ground truth data, zero S2 or S3 entity IDs map to multiple distinct S1 entities. Each target record corresponds to at most one real-world enterprise. Unconstrained local thresholding leads to multi-matching collisions; global greedy bipartite resolution is strictly required.
4. **Covariate Shift (Unseen Country):** The training corpus only contains entities from the United States (60%) and India (40%). The test set introduces **France (15.0% of test records, 259,452 queries)** with zero training labels. French corporate legal entity designations (*SARL, SAS, SASU, EURL, SCI, GAEC, EARL*) cause false-merge clusters under raw string metrics if not normalized.
5. **Cross-Script Divergence:** Indian business records in S2/S3 frequently appear in regional Indic scripts (Devanagari, Telugu, Tamil, Bengali) while S1 contains English transliterations. However, numeric components of the address (house numbers, plot numbers, PIN codes) remain highly conserved across sources, acting as critical anchor signals.

### 2.2 Solution Strategy

**Approach Type:** Multi-Channel Blocking → Supervised LightGBM Reranker → Precision Hardening Engine → Global Greedy Many-to-One (M2O) Bipartite Assignment  
**Core Innovation:** Addressing cross-lingual divergence via full-text composite document indexing (`2 * name + address`) alongside universal AnyAscii transliteration; training a supervised LightGBM pairwise model with country-calibrated decision thresholds; and executing a multi-guard precision hardening layer that prunes over 81,000 false positive links (premise conflicts, cross-city postal conflicts, and chain-store collisions).

```
   Source 1 (Test: 1.73M)            Source 2 + 3 (Test: 9.97M)
             │                                    │
             ▼                                    ▼
      ┌──────────────────────────────────────────────────┐
      │   Universal Normalization Engine (v3)            │
      │   AnyAscii Transliteration + French Suffixes     │
      │   + Consonant Skeletons + Premise Extraction     │
      └────────────────────────┬─────────────────────────┘
                               ▼
      ┌──────────────────────────────────────────────────┐
      │        7-Channel Multi-Index Blocking            │
      │  Ch 1: Word TF-IDF Full-Text (2*Name + Addr)     │
      │  Ch 2: Address Binary Jaccard Sparse Dot Product │
      │  Ch 3: Subword Char 3-5 N-Gram TF-IDF            │
      │  Ch 4: Sorted-Token Permutation Invariance       │
      │  Ch 5: Postal / PIN Inverted Index Bucket        │
      │  Ch 6: Phonetic Consonant Skeleton Inverted Index│
      │  Ch 7: First-3 Prefix Inverted Index             │
      │  + Dense Vector Semantic Retrieval (FAISS)       │
      │         Top-50 Merged Union Candidates           │
      │         Pair Completeness (Recall): >96%         │
      └────────────────────────┬─────────────────────────┘
                               ▼
      ┌──────────────────────────────────────────────────┐
      │        Vectorized Pairwise Feature Space         │
      │  50 Dense Features: String similarities (Lev,    │
      │  Jaro-Winkler, Token Sort/Set), Premise Match,   │
      │  Postal Match, State Match, Channel Scores       │
      └────────────────────────┬─────────────────────────┘
                               ▼
      ┌──────────────────────────────────────────────────┐
      │     Supervised LightGBM Reranker (5-Fold CV)     │
      │  Learns Calibrated P(Match | x)                  │
      │  Country Thresholds: US=0.910, IN=0.825, FR=0.940│
      └────────────────────────┬─────────────────────────┘
                               ▼
      ┌──────────────────────────────────────────────────┐
      │        SOTA Precision Hardening Multi-Guard      │
      │  • Premise / House Number Conflict Guard         │
      │  • Postal / PIN Cross-City Conflict Guard        │
      │  • Multi-Location Chain Store Disambiguation     │
      │  • Marginal Singleton Protection Guard           │
      │  • Degree Bounds: Max 5 from S2, Max 6 from S3   │
      └────────────────────────┬─────────────────────────┘
                               ▼
      ┌──────────────────────────────────────────────────┐
      │    Global Greedy M2O Bipartite Resolution        │
      │  Priority queue by confidence descending         │
      │  Zero duplicate candidate allocations            │
      └────────────────────────┬─────────────────────────┘
                               ▼
              output/matching_results.tsv (0.9142 F₀.₅)
              output/candidate_pairs.tsv
```

---

## 3. Candidate Generation (Blocking)

To reduce the comparison space from $1.73 \times 10^6 \times 9.97 \times 10^6 \approx 1.7 \times 10^{13}$ pairwise comparisons down to a tractable candidate set, we employ a **7-channel multi-index union blocking** strategy strictly partitioned by country:

| Channel | Methodology & Representation | Primary Signal Captured |
|---|---|---|
| **Ch 1: Word Composite TF-IDF** | Sublinear term frequency on `2 * name + address`, $L_2$ normalized, $\text{max\_df} = 0.01$ | Simultaneous lexical name and address alignment |
| **Ch 2: Address Binary Jaccard** | Sparse boolean token matrix dot product via SciPy CSC | Location co-occurrence across script variations |
| **Ch 3: Subword Char N-Grams** | Character 3–5 subword TF-IDF on cleaned entity names | Typographical errors, minor prefix/suffix mutations |
| **Ch 4: Sorted-Token TF-IDF** | Alphabetically sorted word tokens on entity names | Word order permutation invariance |
| **Ch 5: Postal / PIN Inverted Index** | Postal code cluster lookup with address token overlap | Direct geographic neighborhood filtering |
| **Ch 6: Phonetic Consonant Skeleton** | Vowel/duplicate stripped consonant hash index | Phonetic identity across transliteration boundaries |
| **Ch 7: First-3 Prefix Index** | 3-character prefix hash scored by token Jaccard | Fast retrieval for short business names |

- **Candidate pairs generated:** Top-50 candidates per Source 1 entity (Total: ~86.6M candidate pairs).
- **Candidate generation metrics:**
  - **Blocking Recall (Pair Completeness):** **>96.0%** across validation benchmarks.
  - **Reduction Ratio:** **>99.9997%** reduction of the Cartesian comparison space.
- **Ensuring true matches were not lost:** Independent retrieval channels compensate for individual failure modes. When name tokens diverge due to regional scripts, Channels 1, 2, and 5 retrieve candidates via address and postal consistency; when address data is missing, Channels 3, 4, 6, and 7 recover matches via phonetic and subword name similarity.

---

## 4. Matching Model

### Features Used (50 Dense Dimensions)

1. **Name Lexical Metrics (1–12):** `fuzz.ratio`, `token_sort_ratio`, `token_set_ratio`, `partial_ratio`, normalized Levenshtein distance, Jaro-Winkler similarity, word-level Jaccard, token containment ratio, string length ratio, word count difference, first-word exact match indicator, clean name ratio.
2. **Address Metrics (13–18):** Address `fuzz.ratio`, `token_sort_ratio`, `token_set_ratio`, Levenshtein similarity, word-level Jaccard, address token containment.
3. **Structural & Geographic Signals (19–24):** House/premise number agreement (+1 match, −1 conflict, 0 missing), postal code exact match, numeric token overlap, both-address-missing flag, one-address-missing flag, reference name in candidate address overlap.
4. **Channel Retrieval Signals (25–35):** Per-channel retrieval scores (composite TF-IDF, address Jaccard, char n-gram, sorted token, phonetic, prefix, spaceless), max and mean channel score, candidate retrieval rank inverse ($1/\text{rank}$), Source 3 origin indicator, spaceless exact match, spaceless containment, cleaned house number match.
5. **Contextual & N-Gram Features (36–50):** Name character bigram Jaccard, name character trigram Jaccard, address character trigram Jaccard, extracted city exact match, extracted state/region match, 3-char and 5-char name prefix matches, numeric address token Jaccard, shared word count, shared address word count, unique word ratio, composite weighted score ($0.6 \times \text{name} + 0.4 \times \text{address}$), legal suffix match flag, phonetic key overlap ratio, and total number of blocking channels that retrieved the pair.

### Model Architecture & Training

- **Model Type:** LightGBM Gradient Boosted Decision Trees (GBDT) binary classifier.
- **Objective & Metrics:** `binary_logloss` with AUC monitoring, `num_leaves=63–127`, `max_depth=7–9`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`.
- **Cross-Validation Strategy:** 5-fold GroupKFold stratified by `entity_id` to strictly prevent candidate pair leakage belonging to the same reference business across train and validation splits.
- **Hard Negative Mining:** 2-pass training where candidate pairs with high model confidence ($P(\text{Match}) > 0.1$) on non-matching links are extracted as hard negatives and re-weighted (3.0×) in the subsequent training pass.

### Threshold Selection & Precision Hardening Engine

To maximize macro $F_{0.5}$, standard classification boundaries ($\tau = 0.50$) are suboptimal due to the severe penalty on false merges. We implemented country-calibrated thresholds combined with a specialized **Multi-Guard Precision Hardening Engine**:

1. **Country Threshold Calibration:**
   - **United States:** $\tau = 0.910$ (high lexical density, address-anchored).
   - **India:** $\tau = 0.825$ (calibrated for multi-script transliteration variance).
   - **France:** $\tau = 0.940$ (conservative zero-shot threshold preventing false merges on common corporate suffixes like *SARL*, *SAS*).
2. **Precision Hardening Multi-Guard Filters:**
   - **Premise / House Number Guard:** Drops links where extracted house numbers conflict (pruned 68,457 false positives).
   - **Postal / PIN Guard:** Prunes links with conflicting 5/6 digit postal codes (pruned 1,074 cross-city false positives).
   - **Multi-Location Chain Store Guard:** High-frequency brand names ($\ge 20$ occurrences across the corpus) require verified geographic anchoring (postal match, state match, or address token overlap) to eliminate cross-branch collisions (pruned 7,688 false links).
   - **Marginal Singleton Guard:** Protects true singletons by requiring higher confidence on single-candidate predictions (pruned 1,808 spurious single-match links).
   - **Degree Bounds:** Limits match cardinality to ground-truth distribution maximums (max 5 from S2, max 6 from S3).
   - **Total False Positives Filtered:** **81,181 spurious pairs eliminated**, adjusting the average links per entity to 3.20 (matching the ground truth distribution of 3.2–3.4).

### Global Many-to-One (M2O) Bipartite Resolution

All candidate claims passing calibrated thresholds are entered into a global priority queue sorted by confidence descending. A greedy assignment loop guarantees that every S2 or S3 entity ID is linked to at most one reference S1 entity, strictly enforcing the physical uniqueness invariant observed in ground truth.

---

## 5. Results & Error Analysis

- **Public Leaderboard Score (macro $F_{0.5}$):** **0.9142**
- **Internal 5-Fold Cross-Validation Metrics:**
  - Fold 1: $F_{0.5} = 0.8934$, Precision = 0.9148, Recall = 0.8172
  - Fold 2: $F_{0.5} = 0.8931$, Precision = 0.9243, Recall = 0.7870
  - Fold 3: $F_{0.5} = 0.8992$, Precision = 0.9288, Recall = 0.7973
  - Fold 4: $F_{0.5} = 0.8983$, Precision = 0.9293, Recall = 0.7928
  - Fold 5: $F_{0.5} = 0.8947$, Precision = 0.9321, Recall = 0.7710
  - **Mean Out-of-Fold Macro $F_{0.5}$:** **0.8957** (Precision: 0.9258, Recall: 0.7931)
  - Post-Hardening Validation $F_{0.5}$: **0.914+**
- **Blocking Recall:** **>96.0%**

### Common False Positives (Wrong Merges)
1. **Multi-location retail & food chains:** Independent branches of national franchises (e.g., "Subway", "Starbucks", regional healthcare networks) that share identical trade names across different streets or shopping centers. *Mitigated by the multi-location chain guard requiring premise number and street token agreement.*
2. **Co-located businesses in commercial malls:** Disjoint companies operating within the same high-density office complex sharing identical street addresses and postal codes. *Mitigated by strict legal entity suffix and name token containment checks.*

### Common False Negatives (Missed Matches)
1. **Double-omission records:** Candidate pairs where both S1 and target records feature blank/missing addresses (`NaN`) combined with severe phonetic name typos.
2. **Extreme cross-script phonetic divergence:** Complex colloquial enterprise titles transliterated into English with non-standard phonetic spellings where address tokens were completely omitted in secondary sources.

---

## 6. Conclusion

Solving entity resolution across 28 million noisy records under the precision-heavy macro $F_{0.5}$ metric requires optimizing the full system lifecycle: language-agnostic AnyAscii normalization for multi-script inputs, 7-channel multi-index blocking to ensure >96% candidate recall, supervised LightGBM reranking with GroupKFold cross-validation, precision-hardening filters eliminating premise and geographic conflicts, and global greedy bipartite resolution to enforce physical uniqueness. This unified architecture achieved a verified **0.9142** score on the leaderboard and generalized robustly to unseen country domains.

---

## Appendix

### A. Code Artefacts

All runnable code is organized under `code/business_entity_resolution/`:

| File Path | Description & Role in Pipeline |
|---|---|
| `src/normalization_v3.py` | Universal normalizer (AnyAscii transliteration, French suffixes, consonant skeletons) |
| `src/blocking_v3.py` | 7-channel multi-index blocking engine (TF-IDF, Jaccard, N-grams, Phonetic, Prefix) |
| `src/features_v3.py` | 50-dimensional pairwise feature extractor |
| `src/lgbm_reranker.py` | LightGBM pairwise reranker trainer with 5-fold cross-validation |
| `src/train_reranker_v3.py` | Supervised training harness with 2-pass hard negative mining |
| `src/run_hardening_now.py` | Production precision-hardening engine (HN, postal, chain, and singleton guards) |
| `src/precision_hardening_v3.py` | Core multi-guard precision filtering module |
| `src/inference_sota.py` | Sharded parallel candidate generation and scoring |
| `src/generate_submission.py` | Self-contained baseline end-to-end reproduction script |
| `src/validate.py` | Schema and format integrity assertion test suite |
| `requirements.txt` | Pinned dependency versions for execution environment |
| `README.md` | Complete reproduction and execution guide |

**Reproduction Entry Points:**
- **Full Production Pipeline:** `python3 src/run_hardening_now.py`
- **Baseline Pipeline:** `python3 src/generate_submission.py`
- **Output Validator:** `python3 dataset/student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/student_resource/dataset/test`

### B. Computational Infrastructure

- **Hardware Used:** Multi-core server (32–64 CPU cores, 128+ GB RAM) for distributed blocking and scoring; local validation on Apple Silicon.
- **Throughput:** ~150,000 entity queries per minute across 32 multiprocessing workers.
- **Peak RAM:** ~8 GB per country partition via streaming disk I/O and chunked matrix projections.

### C. Open-Source Licensing Compliance

- **LightGBM:** MIT License
- **RapidFuzz:** MIT License
- **AnyAscii:** MIT / ISC License
- **Scikit-Learn, SciPy, NumPy, Pandas:** BSD Permissive Licenses
- All models, libraries, and utilities strictly adhere to the competition's open-source and fair-play requirements.
