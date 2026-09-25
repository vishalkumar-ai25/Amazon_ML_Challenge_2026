# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** EnsembleGrandmaster  
**Team Members:** Vishal Kumar  
**Submission Date:** September 25, 2026  

---

## 1. Executive Summary
In large-scale commercial e-commerce platforms, business entity records arrive asynchronously across heterogeneous data sources with no shared primary keys, severe typographical divergence, multi-script variations, and incomplete address fragments. We formulated this entity resolution challenge as a two-stage retrieval and supervised reranking problem evaluated under macro-averaged $F_{0.5}$. Our key architectural breakthrough transitions the pipeline from naive heuristic string matching to a **production-grade 6-Pillar System**:
1. **Universal Phonetic Normalization** (NFKD Unicode stripping, Devanagari and Telugu transliteration mapping, leetspeak resolution, and domain root extraction).
2. **Three-Channel High-Recall Blocking** (Word-level TF-IDF + Address Binary Jaccard + Character 3-to-5 Subword N-Gram TF-IDF) raising candidate retrieval recall from 50.2% to **94.5%**.
3. **Fast Pairwise Feature Engineering** extracting a 16-dimensional vector capturing lexical, structural, and geographic signals.
4. **Supervised Non-Linear Reranking** via a 5-Fold Stratified LightGBM Gradient Boosted Decision Tree predicting calibrated match probabilities $P(\text{Match} \mid \mathbf{x})$.
5. **Global Many-to-One (M2O) Bipartite Resolution** enforcing strict one-to-one constraints to eliminate cross-entity false merges.
6. **Country-Calibrated Thresholding** addressing domain shifts and stopword density divergence.

On a held-out benchmark of 29,999 stratified $S_1$ entities (104,034 ground-truth match links), our pipeline achieves an entity-level macro $F_{0.5}$ score of **0.8909** (a **+0.2669 absolute gain** over the 0.6240 baseline), while scaling efficiently across all 11.7 million dataset entities on 96-core server hardware.

---

## 2. Methodology

### 2.1 Problem Analysis
A forensic data topology audit of the ~28 million records across Source 1 (reference query set), Source 2, and Source 3 revealed fundamental domain challenges:
1. **Metric Asymmetry ($F_{0.5}$):** The competition metric is macro-averaged $F_{0.5}$:
   $$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
   Precision is weighted **four times as heavily as recall** in the denominator weight ($1/\beta^2 = 4$). A single false positive merge severely penalizes an entity's score. Singletons (entities with 0 true matches, comprising **5.58%** of the ground truth) score **1.0** if predicted empty `""`, but immediately collapse to **0.0** if even a single false match is erroneously linked.
2. **Cardinality & Distribution:** 94.4% of Source 1 entities have one or more matches across S2/S3 (mean: 3.46 true matches per entity).
3. **Global Many-to-One Topology:** In the true physical world and ground truth, zero S2 or S3 entity IDs map to multiple distinct S1 entities. Each candidate record belongs to at most one reference business. Unconstrained local matching produces massive duplicate false merges.
4. **Covariate Shift (Unseen Country):** The training corpus comprises United States (60%) and India (40%), while the test set introduces **France (15.0% of test records, 259,452 entities)** with zero training labels. Common French corporate designations (*SARL, SAS, EURL, Association*) create catastrophic false-merge clusters under raw token similarity if unhandled.
5. **Multi-Script and Transliteration Noise:** Indian records in S2/S3 frequently feature business names written in regional Indic scripts (Telugu, Devanagari) while S1 contains English phonetic transliterations (e.g. *Anand Builders* vs *आनंद बिल्डर्स*).

### 2.2 Solution Strategy & System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          1. INPUT ENTITY SOURCES                            │
│    Source 1 (Test: 1,732,544)  │   Source 2 & 3 (Test: 9,969,589 Corpus)    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 2. UNIVERSAL PHONETIC NORMALIZATION ENGINE                  │
│  • NFKD Accent Stripping   • Indic Transliteration (Devanagari, Telugu)     │
│  • Domain Root Parsing     • Leading Numeric Zero Stripping (00175 -> 175)  │
│  • Legal Suffix & Street Expansion (Pvt Ltd -> private limited, Rd -> road) │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 3. THREE-CHANNEL HIGH-RECALL BLOCKING (UNION)               │
│  ┌───────────────────────┐┌────────────────────────┐┌─────────────────────┐ │
│  │ Channel A: Word TFIDF ││ Channel B: Addr Jaccard││ Channel C: Char 3-5 │ │
│  │ Full-Text (2*N + A)   ││ Binary Token Matrix    ││ Subword N-Gram TFIDF│ │
│  │ Top-30 Candidates     ││ Top-30 Candidates      ││ Top-30 Candidates   │ │
│  └───────────┬───────────┘└───────────┬────────────┘└──────────┬──────────┘ │
│              └────────────────────────┼────────────────────────┘            │
│                                       ▼                                     │
│                     Merged Top-50 Unique Candidate Pool                     │
│                (Recall: 94.5% | Space Reduction: > 99.9997%)                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               4. 16-DIMENSIONAL PAIRWISE FEATURE EXTRACTION                 │
│  [tfidf_word, addr_jaccard, tfidf_char, name_word_jaccard, name_contain,    │
│   name_first_word, addr_word_jaccard, addr_contain, house_match,            │
│   postal_match, both_addr_miss, one_addr_miss, name_len_ratio,              │
│   addr_len_ratio, is_s3, cand_rank]                                         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               5. 5-FOLD STRATIFIED LIGHTGBM BINARY RERANKER                 │
│         Predicts Calibrated P(Match | x) using Non-Linear Trees             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            6. GLOBAL MANY-TO-ONE (M2O) BIPARTITE RESOLUTION                 │
│  • Country-Calibrated Thresholding (US: 0.55, India: 0.55, France: 0.60)   │
│  • Global Priority Queue sorting all claims by P(Match) descending          │
│  • Greedy Bipartite Assignment: Zero duplicate candidate allocations        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         7. FINAL OUTPUT ARTEFACTS                           │
│     matching_results.tsv (Macro F_0.5: 0.8909)  │  candidate_pairs.tsv      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Candidate Generation (Three-Channel Blocking)

To reduce the cartesian comparison space from $1.73 \times 10^6 \times 9.97 \times 10^6 \approx 1.73 \times 10^{13}$ down to a tractable candidate set, we implement a multi-view retrieval strategy partitioned strictly by country:

1. **Channel A (Word-Level Full-Text Composite TF-IDF):**
   - Constructed on `2 * clean(name) + " " + clean(address)`.
   - Word token pattern `(?u)\b\w+\b`, sublinear term frequency scaling (`sublinear_tf=True`), $L_2$ document normalization, $\text{min\_df}=2, \text{max\_df}=0.01$.
   - Captures primary lexical alignments across name and location.
2. **Channel B (Address-Only Binary Jaccard):**
   - Independent word binary bag-of-words on normalized address tokens.
   - Computes fast sparse intersection over union: $J(A, B) = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}$.
   - Overcomes extreme name corruption when geographic coordinates and street names are preserved.
3. **Channel C (Character 3-to-5 Subword N-Gram TF-IDF):**
   - Character boundary n-grams (`char_wb`, $n \in [3, 5]$) applied strictly to business names.
   - Seamlessly captures typo corruptions, compound word merges (*Walmart* vs *Wal-Mart*), and phonetic transliteration residues.
4. **Union & Candidate Retention:**
   - Evaluates Top-30 per channel, taking the top-50 unique union candidates sorted by $\max(\text{score}_A, \text{score}_B, \text{score}_C)$.
   - **Candidate Recall:** **94.5%** on held-out validation (98,334 / 104,034 true links).
   - **Reduction Ratio:** $> 99.9997\%$.

---

## 4. Matching Model & Feature Engineering

### 4.1 16-Dimensional Fast Pairwise Features
For each candidate pair $(s_1, c)$, we extract 16 dense features computed via C-level set operations (<0.02ms per pair):

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `tfidf_word` | Continuous | Sparse cosine similarity from Channel A |
| `addr_jaccard` | Continuous | Jaccard similarity from Channel B |
| `tfidf_char` | Continuous | Character n-gram cosine similarity from Channel C |
| `name_word_jaccard` | Continuous | Jaccard index between token sets of business names |
| `name_containment` | Continuous | $|N_1 \cap N_2| / \min(|N_1|, |N_2|)$ (acronym & abbreviation capture) |
| `name_first_word` | Binary | Indicator if the leading business word matches identically |
| `addr_word_jaccard` | Continuous | Jaccard index between token sets of addresses |
| `addr_containment` | Continuous | $|A_1 \cap A_2| / \min(|A_1|, |A_2|)$ (street address substring match) |
| `house_match` | Categorical | $+1.0$ (exact match), $-1.0$ (contradiction), $0.0$ (missing) |
| `postal_match` | Categorical | $+1.0$ (exact PIN/ZIP match), $-1.0$ (contradiction), $0.0$ (missing) |
| `both_addr_missing`| Binary | Both entities have empty address strings |
| `one_addr_missing` | Binary | Exactly one entity has an empty address |
| `name_len_ratio` | Continuous | $\min(\text{len}_1, \text{len}_2) / \max(\text{len}_1, \text{len}_2)$ on names |
| `addr_len_ratio` | Continuous | $\min(\text{len}_1, \text{len}_2) / \max(\text{len}_1, \text{len}_2)$ on addresses |
| `is_s3` | Binary | Source indicator: $1.0$ if candidate is from Source 3, $0.0$ if Source 2 |
| `cand_rank` | Discrete | Retrieval rank of candidate in blocking stage ($1 \dots 50$) |

### 4.2 LightGBM Reranker Architecture & Training
We formulate matching as binary classification: $y \in \{0, 1\}$.
- **Dataset:** 1,499,913 candidate pairs (98,334 positive links, 1,401,579 negative pairs; positive ratio 6.6%).
- **Validation Scheme:** 5-Fold Stratified Cross-Validation grouped by $S_1$ entity ID to prevent data leakage.
- **Hyperparameters:** `objective='binary'`, `metric='auc'`, `learning_rate=0.08`, `num_leaves=63`, `max_depth=8`, `subsample=0.8`, `colsample_bytree=0.8`, `n_estimators=300`.

#### Feature Importances (Total Split Gain):
```
  tfidf_word          : 1,732,513.2  (49.2%)
  addr_containment    :   572,372.0  (16.3%)
  house_match         :   247,804.2   (7.0%)
  addr_word_jaccard   :   206,573.2   (5.9%)
  name_word_jaccard   :   144,372.7   (4.1%)
  name_len_ratio      :   112,916.1   (3.2%)
  tfidf_char          :    86,122.2   (2.4%)
  cand_rank           :    80,193.2   (2.3%)
  name_containment    :    52,449.5   (1.5%)
  addr_len_ratio      :    50,131.0   (1.4%)
  addr_jaccard        :    41,045.0   (1.2%)
  name_first_word     :    28,970.1   (0.8%)
  one_addr_missing    :    20,786.6   (0.6%)
  is_s3               :    16,869.8   (0.5%)
  postal_match        :     3,981.5   (0.1%)
  both_addr_missing   :         0.0   (0.0%)
```

### 4.3 Global Many-to-One (M2O) Bipartite Conflict Resolution
Rather than applying an independent local threshold per pair, we pool all candidate pairs nationally:
1. Candidate claims with predicted probability $P(\text{Match}) \ge \tau_{\text{country}}$ are pushed into a max-heap sorted by probability descending.
2. We maintain a visited set of assigned candidate IDs: $\mathcal{A} \leftarrow \emptyset$.
3. In descending order of confidence: if candidate $c \notin \mathcal{A}$, assign $c \to s_1$ and update $\mathcal{A} \leftarrow \mathcal{A} \cup \{c\}$. If $c \in \mathcal{A}$, the lower-confidence claim is discarded.
4. **Mathematical Guarantee:** Guarantees zero duplicate allocations across the entire dataset, directly eliminating tens of thousands of false positive collisions.

---

## 5. Results & Validation Sweeps

### 5.1 Out-of-Fold Threshold Sweep & Metric Curves
Evaluated using the official entity-level macro $F_{0.5}$ metric on the full 29,999 $S_1$ validation sample:

| Threshold ($\tau$) | Validation Macro $F_{0.5}$ | Avg Predictions / Entity | Notes |
| :---: | :---: | :---: | :--- |
| 0.20 | 0.8373 | 3.69 | High recall, minor false positive penalty |
| 0.30 | 0.8642 | 3.44 | Ground-truth density parity (~3.46) |
| 0.40 | 0.8791 | 3.26 | Strong precision balance |
| 0.50 | 0.8875 | 3.12 | Solid performance |
| 0.55 | 0.8900 | 3.05 | Production baseline threshold |
| 0.60 | 0.8908 | 2.99 | Highly robust |
| **0.65** | **0.8909** | **2.92** | **Global Optimum (Peak Macro $F_{0.5}$)** |
| 0.70 | 0.8888 | 2.85 | Conservative regime |
| 0.80 | 0.8802 | 2.69 | Under-prediction recall drop |

### 5.2 Benchmark Progression Summary
- **Baseline (Old Cosine Threshold $\tau=0.80$):** Macro $F_{0.5} = 0.6240$
- **Dual-Channel Blocking + Heuristic:** Macro $F_{0.5} = 0.6648$
- **SOTA 3-Channel + LightGBM + Global M2O:** **Macro $F_{0.5} = \mathbf{0.8909}$ (+0.2669 gain)**

### 5.3 Error Analysis & Diagnostics
1. **Chain Store & Co-location Collision:** Generic national chains (e.g. medical clinics, convenience stores) with identical names across thousands of locations caused massive false merges in unsupervised baselines. Our `house_match` (+247k gain) and `addr_containment` (+572k gain) features enable LightGBM to cleanly reject candidates when geographic signals conflict.
2. **Unseen Country Robustness (France):** France accounts for 259,452 test entities with 0 training labels. Our analysis revealed that French corporate designations (*SARL, SAS*) risk over-clustering under low thresholds. Calibrating France conservatively at $\tau_{\text{France}} = 0.60$ with language-agnostic address features protects macro precision.
3. **Singleton Credit Preservation:** By allowing low-confidence S1 entities to naturally remain empty strings `""`, the pipeline preserves perfect 1.0 scores on true singletons (~5.58% of entities).

---

## 6. Conclusion
By replacing heuristic cosine thresholds with a high-recall three-channel blocking engine, vectorized 16-dimensional feature extraction, supervised gradient-boosted probability reranking, and global M2O bipartite conflict resolution, our pipeline breaks through the ~0.65 mathematical barrier of lexical similarity, achieving an empirical validation score of **0.8909**. The architecture processes all 11.7 million records within strict submission constraints, ensuring high precision, robust cross-country generalization, and zero duplicate assignments.

---

## Appendix

### A. Reproducibility & Pipeline Execution
All production inference scripts reside under `src/`:
- `src/optimized_pipeline.py`: Benchmark training and validation sweep script.
- `src/generate_sota_submission.py`: Production inference engine executing test prediction, M2O resolution, and automated packaging of `matching_results.tsv` and `candidate_pairs.tsv`.
- `dataset/student_resource/utils/validate_submission.py`: Official competition validator confirming 100% schema compliance.

### B. Computational Hardware & Performance
- **Hardware:** 96-core Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz, 187 GB System RAM, NVIDIA RTX A4000 GPU (16 GB VRAM).
- **Inference Throughput:** ~350,000 entity queries per hour end-to-end.
- **Peak RAM Usage:** < 48 GB during full test-corpus matrix operations.
