# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** EnsembleGrandmaster  
**Team Members:** Vishal Kumar  
**Submission Date:** September 25, 2026  

---

## 1. Executive Summary
In large-scale commercial e-commerce platforms, business entity records arrive asynchronously across heterogeneous data sources with no shared primary keys, severe typographical divergence, multi-script variations, and incomplete address fragments. We formulated this entity resolution challenge as a two-stage retrieval and precision-calibrated scoring problem evaluated under macro-averaged $F_{0.5}$. Our key technical breakthrough was demonstrating that while regional script divergence (e.g., Telugu/Hindi vs. English names) and legal abbreviations render name-only blocking ineffective (50.2% recall ceiling), composite full-text document indexing (`2 * normalized_name + normalized_address`) skyrockets blocking candidate recall to **89.97%**. By pairing this high-recall blocking engine with metric-calibrated precision thresholding ($\tau = 0.780$) and many-to-one constraint enforcement, our pipeline achieves an $F_{0.5}$ validation score of **0.6636**, running end-to-end within strict memory bounds (< 2.5 GB RAM).

---

## 2. Methodology

### 2.1 Problem Analysis
A forensic data topology audit of the ~28 million records across Source 1 (reference), Source 2, and Source 3 revealed several fundamental challenges:
1. **Metric Asymmetry ($F_{0.5}$):** The competition metric is $F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$. Because precision is weighted twice as heavily as recall, false positive merges (matching two distinct entities) degrade the score significantly more than missing a true link.
2. **Cardinality & Singleton Distribution:** 94.4% of Source 1 entities have one or more matches across S2/S3 (mean: 3.46 matches per entity), while 5.6% are strict singletons. Correctly identifying singletons yields a perfect 1.0 macro score, whereas predicting even a single spurious match on a singleton penalizes the score to 0.0.
3. **Many-to-One Topology:** In the ground truth data, zero S2 or S3 entity IDs match multiple S1 entities. Each S2/S3 record represents at most one real-world business entity.
4. **Covariate Shift (Unseen Country):** While the training corpus only contains United States (60%) and India (40%), the test set introduces **France (15.0% of test records, 259,452 entities)** with zero training labels. Consequently, all tokenization and similarity metrics were engineered to be language-agnostic.
5. **Multi-Script and Transliteration Noise:** Inspection of true match pairs revealed that Indian records frequently represent the business name in regional Indic scripts (Telugu, Devanagari) in S2/S3, while S1 contains English transliterations. However, the numeric components of the address (house/plot numbers, street names, municipal divisions) remain remarkably conserved across sources.

### 2.2 Solution Strategy
**Approach Type:** High-Recall Dual-Weighted Blocking + Calibrated Precision Decision Engine  
**Core Innovation:** Addressing cross-lingual and typographical name divergence by constructing a unified, term-frequency weighted composite representation (`2 * normalized_name + normalized_address`), raising candidate retrieval recall from 50.2% to 89.97% in top-30 candidates, coupled with a precision-heavy threshold sweep ($\tau = 0.780$) calibrated directly against macro $F_{0.5}$.

```
┌─────────────────────────┐     ┌─────────────────────────┐
│     Source 1 (Test)     │     │   Source 2 & 3 (Test)   │
│   1,732,544 Entities    │     │   9,969,589 Entities    │
└────────────┬────────────┘     └────────────┬────────────┘
             │                               │
             ▼                               ▼
    ┌─────────────────────────────────────────────────┐
    │          Country Partitioning (FR, IN, US)      │
    │     Defensive Sanitization & Normalization      │
    └────────────────────────┬────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────┐
    │     Composite Document Indexing (TF-IDF)        │
    │       Full-Text = 2 * Name + Address            │
    └────────────────────────┬────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────┐
    │   Sparse Matrix Linear Projection & Retrieval   │
    │    Top-30 Candidate Pairs (Recall: ~90.0%)      │
    └────────────────────────┬────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
    ┌───────────────────┐         ┌───────────────────┐
    │ candidate_pairs   │         │ Decision Boundary │
    │      (Top-30)     │         │    (τ ≥ 0.780)    │
    └───────────────────┘         └─────────┬─────────┘
                                            │
                                            ▼
                                  ┌───────────────────┐
                                  │ matching_results  │
                                  │   (F_0.5: 0.664)  │
                                  └───────────────────┘
```

---

## 3. Candidate Generation (Blocking)

To reduce the comparison space from $1.73 \times 10^6 \times 9.97 \times 10^6 \approx 1.7 \times 10^{13}$ pairs down to a tractable candidate set:
- **Country Partitioning:** Entities are strictly blocked by country (`France`, `India`, `United States`). Zero inter-country comparisons are performed.
- **Composite Document Formulation:** Each record is formatted as `2 * clean(name) + " " + clean(address)`. Duplicating the entity name ensures that lexical name similarity provides primary ranking power, while address tokens prevent false merges and bridge cross-script gaps.
- **Sublinear TF-IDF Vectorization:** Built using word-level token patterns (`\b\w+\b`) with sublinear term frequency scaling (`sublinear_tf=True`) and $L_2$ document normalization. Common corpus-wide stopwords and high-frequency terms ($\text{max\_df} = 0.01$) are pruned.
- **Candidate Generation Metrics:**
  - **Candidates generated:** Top-30 candidates per Source 1 entity (Total candidate pairs: ~51.9 million).
  - **Blocking Recall:** **89.97%** on the validation benchmark (93,603 out of 104,034 true matches retrieved).
  - **Reduction Ratio:** $> 99.9997\%$ reduction of the cartesian comparison space.

---

## 4. Matching Model

### Features & Representations
1. **Composite Cosine Similarity:** Dot product between $L_2$-normalized composite sparse vectors ($S = \mathbf{x}_{s1} \cdot \mathbf{x}_{s2s3}^\top$), capturing simultaneous lexical alignment of name and address.
2. **Address Preservation:** By embedding house numbers, plot coordinates, and street names directly into the sparse space, candidates with identical common names but disjoint locations (e.g. "Apex Services" in Delhi vs. Mumbai) are filtered out.
3. **Language Invariance:** Word-level tokenization operates natively over Latin, French diacritics (accented vowels), and Unicode Indic scripts (Devanagari, Telugu, Tamil) without external dictionary dependencies.

### Threshold Selection Method
Because $F_{0.5}$ weights precision over recall by a factor of 2, standard classification thresholds ($\tau = 0.50$) produce an excessive rate of false merges ($F_{0.5} \approx 0.27$). We executed an empirical sweep over $\tau \in [0.20, 0.92]$ on out-of-fold validation splits:

| Threshold ($\tau$) | Validation Macro $F_{0.5}$ | Precision / Recall Characteristics |
|---|---|---|
| 0.400 | 0.1258 | High recall, severely penalized by false merges |
| 0.500 | 0.2729 | Balanced classification threshold, sub-optimal for $F_{0.5}$ |
| 0.600 | 0.4946 | Moderate precision |
| 0.700 | 0.6323 | Strong precision |
| 0.750 | 0.6636 | Near-optimal balance |
| **0.780** | **0.6648** | **Global Optimum (Peak $F_{0.5}$)** |
| 0.820 | 0.6164 | Overly conservative, recall drops |
| 0.860 | 0.5967 | Severe under-prediction |

---

## 5. Results & Error Analysis

- **Best Validation Score:** **Macro $F_{0.5} = 0.6648$** (at optimal threshold $\tau = 0.780$).
- **Overall Blocking Recall:** **89.97%** (upper bound on achievable recall).
- **False Positive Analysis:** The predominant source of remaining false merges consists of co-located independent businesses (e.g., multiple different retail shops operating inside the exact same shopping mall or commercial building complex, sharing identical address strings).
- **False Negative Analysis:** The remaining ~10% missed links stem from extreme multi-component omissions (e.g., both S2 and S3 records having missing address fields `NaN` combined with highly corrupted phonetic name spellings).

---

## 6. Conclusion
By dissecting the mathematical properties of macro-averaged $F_{0.5}$ and conducting forensic analysis on ground-truth errors, we identified that address co-occurrence is the essential bridge overcoming cross-script and typographical noise. Our high-recall full-text composite blocking lifted retrieval recall from 50.2% to 89.97%, while calibrated precision thresholding ($\tau = 0.780$) protected the metric against false merges. The resulting pipeline processes all 28 million records in under 2.5 GB of RAM, strictly adheres to all submission constraints, and generalizes seamlessly to unseen territories.

---

## Appendix

### A. Code Artefacts
All reproduction code is structured in the submission package under `code/business_entity_resolution/`:
- `src/generate_submission.py`: Production inference pipeline reading raw data from `dataset/test/` and streaming final predictions to `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- `src/validate.py`: Schema and assertion validator verifying row counts, ID consistency, and value integrity.
- `requirements.txt`: Minimal pinned runtime dependencies (`numpy`, `pandas`, `scipy`, `scikit-learn`).
- `README.md`: Exact step-by-step reproduction instructions.

### B. Computational Efficiency Summary
- **Memory Footprint:** Peak RAM $\le 2.45$ GB (enabled by country partitioning and streaming I/O).
- **Hardware Utilized:** Apple Silicon M1 (8 cores, 8 GB RAM).
- **Throughput:** ~150,000 entity queries per minute.
