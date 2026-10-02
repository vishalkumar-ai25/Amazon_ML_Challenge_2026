# Amazon ML Challenge 2026: Large-Scale Business Entity Resolution

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Metric](https://img.shields.io/badge/Leaderboard%20Macro%20F0.5-0.9142-FF9900?logo=amazon&logoColor=white)](https://unstop.com)
[![Model](https://img.shields.io/badge/Model-LightGBM%20GBDT-2E7D32?logo=tree&logoColor=white)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI Pipeline](https://github.com/vishalkumar-ai25/Amazon_ML_Challenge_2026/actions/workflows/ci.yml/badge.svg)](https://github.com/vishalkumar-ai25/Amazon_ML_Challenge_2026/actions/workflows/ci.yml)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A high-throughput, precision-calibrated entity resolution engine resolving ~28 million multi-source enterprise records under extreme metric asymmetry and zero-shot geographic domain shifts.**

</div>

---

## 📌 Executive Summary

Modern commercial e-commerce platforms ingest business identity records asynchronously across heterogeneous secondary sources (catalogs, registries, merchant portals) lacking universal primary keys. This repository implements an end-to-end entity resolution pipeline resolving **~28 million records** across three noisy data sources (`Source 1`, `Source 2`, `Source 3`) evaluated under **macro-averaged $F_{0.5}$**.

### Key Highlights
- **Top-Tier Performance:** Achieved a verified **0.9142 macro $F_{0.5}$** on the official competition leaderboard.
- **Extreme Precision Calibration:** Formulated to optimize the asymmetric $F_{0.5}$ objective function, where precision is weighted $4\times$ heavier than recall ($1/\beta^2 = 4$).
- **Zero-Shot Geographic Generalization:** Trained exclusively on US (60%) and Indian (40%) enterprise data, generalizing seamlessly to the unseen **France (259,452 entities, 15% of test queries)** partition via language-agnostic phonetic normalization and country-specific threshold calibration.
- **Scalable Distributed Throughput:** Scales linearly across multi-core systems (32–64 CPU cores), processing over **150,000 entity queries per minute** while maintaining strict memory bounds (< 8 GB RAM per partition) via streaming sparse projections.

---

## 🏛️ System Architecture

The pipeline operates as a synchronized 6-stage distributed workflow:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          1. INPUT ENTITY SOURCES                            │
│    Source 1 (Test: 1,732,544 Queries) │ Source 2 & 3 (Test: 9,969,589 Corpus│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              2. UNIVERSAL PHONETIC NORMALIZATION ENGINE (v3)                │
│  • NFKD Unicode Accent Stripping        • AnyAscii Script Transliteration   │
│  • French Legal Morphology Expansion    • Consonant Skeleton Phonetic Keys  │
│  • Spaceless Domain/Brand Normalization • Standardized Premise / HN Parsing │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 3. 7-CHANNEL HIGH-RECALL BLOCKING ENGINE (UNION)            │
│  ┌───────────────────────┐┌────────────────────────┐┌─────────────────────┐ │
│  │ Channel 1: Word TF-IDF││ Channel 2: Addr Jaccard││ Channel 3: Subword  │ │
│  │ Full-Text (2*N + A)   ││ Binary Token Matrix    ││ Char 3-5 N-Gram     │ │
│  └───────────┬───────────┘└───────────┬────────────┘└──────────┬──────────┘ │
│  ┌───────────┴───────────┐┌───────────┴────────────┐┌──────────┴──────────┐ │
│  │ Channel 4: Sorted Word││ Channel 5: Postal/PIN  ││ Ch 6-7: Phonetic &  │ │
│  │ Permutation Invariance││ Inverted Index Bucket  ││ Prefix Token Index  │ │
│  └───────────┬───────────┘└───────────┬────────────┘└──────────┬──────────┘ │
│              └────────────────────────┼────────────────────────┘            │
│                                       ▼                                     │
│                     Merged Unique Candidate Pool (Top-35)                   │
│               Pair Completeness (Recall): >96.0% | Reduction >99.9997%      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              4. VECTORIZED PAIRWISE FEATURE EXTRACTION (50D)                │
│  • C-Accelerated Levenshtein, Jaro-Winkler, Token Sort & Set Ratios        │
│  • Premise Agreement (+1 match, -1 conflict), Postal & State Indicators     │
│  • Channel Rank Inverses, Spaceless Matches, Numeric Token Overlaps         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            5. SUPERVISED LIGHTGBM RERANKER (2-Pass Hard Negative Mining)    │
│  • 5-Fold GroupKFold CV stratified by Entity ID (zero leakage)              │
│  • Pass 1: Learn decision surface | Pass 2: Upweight hard negatives (3x)   │
│  • Country Thresholds: US (τ=0.910), India (τ=0.825), France (τ=0.940)      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            6. SOTA PRECISION HARDENING & GLOBAL BIPARTITE RESOLUTION        │
│  • Premise / House Number Conflict Guard: -68,457 False Positives Pruned    │
│  • Cross-City Postal / PIN Conflict Guard: -1,074 False Positives Pruned    │
│  • Multi-Location Chain Disambiguation:   -7,688 False Positives Pruned     │
│  • Marginal Singleton Protection:         -1,808 False Positives Pruned     │
│  • Global Priority Queue Greedy Assignment: Zero Duplicate Record Merges    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FINAL SUBMISSION ARTIFACTS                          │
│  • output/matching_results.tsv   (1,732,544 rows | Macro F0.5 = 0.9142)     │
│  • output/candidate_pairs.tsv    (1,732,544 rows | 0 Subset Violations)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Core Engineering Innovations

### 1. Mathematical Objective & Singleton Protection
The competition evaluates submissions via the macro-averaged $F_{0.5}$ metric:
$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

In this formulation, **precision is weighted $4\times$ heavier than recall** in the denominator. Approximately **5.6% of Source 1 entities in ground truth are strict singletons** (zero matches). Correctly predicting an empty match set yields an entity score of $1.0$; assigning even a single spurious false positive collapses the entity score to $0.0$. We implemented a dedicated **Marginal Singleton Guard** that enforces higher confidence thresholds on candidate singletons, protecting this critical performance margin.

### 2. Multi-Channel High-Recall Blocking Engine
Standard name-only blocking hits an upper recall ceiling of **50.2%** due to regional Indic scripts (Devanagari, Telugu, Tamil, Bengali) appearing in secondary sources against English transliterations in Source 1. 

We formulated a 7-channel union blocking strategy that raises candidate retrieval recall to **>96.0%** while pruning **>99.9997%** of the $1.7 \times 10^{13}$ pairwise Cartesian space:
1. **Word Composite TF-IDF:** Sublinear TF on `2 * clean(name) + clean(address)`. Address co-occurrence acts as a bridge across transliterated name scripts.
2. **Address Binary Jaccard:** Sparse matrix dot products over binary address token vectors.
3. **Subword Character 3–5 N-Grams:** Captures typographical errors, compounding, and abbreviations.
4. **Sorted-Token Word TF-IDF:** Invariant to word order permutations (*"Apex Motors Pvt Ltd"* vs *"Motors Apex"*).
5. **Postal / PIN Code Inverted Index:** Partitions candidates into tight geographic neighborhoods.
6. **Consonant Skeleton Phonetic Index:** Strips vowels and duplicate consonants to align phonetic roots across languages.
7. **First-3 Prefix Inverted Index:** High-speed lookup for short corporate names.

### 3. Precision Hardening Multi-Guard Engine
Post-classification, our precision hardening layer applies physical and domain rules to eliminate false merges:
- **Premise / House Number Veto:** If both records contain extracted premise numbers that conflict (e.g., *Suite 104* vs *Suite 108* on the same commercial boulevard), the link is vetoed unless one is a documented unit sub-string.
- **Cross-City Postal Veto:** Prunes links between identical enterprise names operating in different postal jurisdictions.
- **Multi-Location Chain Disambiguation:** Enterprises with high corpus frequency ($\ge 20$, such as national banks or retail franchises) are required to exhibit strict geographic anchoring (matching state, postal code, or premise number).
- **Cardinality Bounds:** Enforces ground-truth empirical degree bounds (max 5 matches from S2, max 6 from S3; average 3.20 links/entity).

### 4. Global Many-to-One (M2O) Bipartite Resolution
In physical reality, each candidate record in Source 2 or Source 3 represents at most one real-world enterprise. Unconstrained local thresholding causes distinct Source 1 queries to compete for and duplicate identical secondary records. We route all candidate claims into a global priority queue sorted by confidence descending, greedily assigning each secondary ID to at most one reference entity.

---

## 📊 Empirical Progression & Ablation Study

| Iteration / Milestone | Methodology & Architecture | Blocking Recall | Validation $F_{0.5}$ | Public LB $F_{0.5}$ | Key Architectural Discovery |
|---|---|:---:|:---:|:---:|---|
| **Sprint 0 Baseline** | Exact string & token Jaccard matching | 50.2% | ~0.420 | — | Name-only blocking fails on regional scripts |
| **Sprint 1 Forensic** | Full-text composite TF-IDF (`2*name + addr`) | 89.97% | 0.6240 | — | Address co-occurrence bridges script mismatches |
| **Milestone 2** | Dual-channel blocking + Calibrated Threshold ($\tau=0.78$) + M2O | 95.74% | 0.6648 | 0.6636 | Enforcing physical uniqueness avoids collisions |
| **Milestone 3 (LGBM v1)** | 18 dense features + LightGBM GBDT (5-Fold CV) | 94.50% | 0.8957 | 0.8934 | Supervised reranking outperforms manual heuristics |
| **Milestone 4 (SOTA v2)** | 7-channel blocking + AnyAscii + 50 features + Country thresholds | >96.0% | 0.9021 | 0.9103 | Language-agnostic normalization stabilizes France |
| **Production Final** | **50D LGBM + Precision Hardening Multi-Guards + M2O** | **>96.0%** | **0.9142** | **0.9142** | **-81,181 false positives eliminated** |

---

## 📂 Repository Directory Map

```text
Amazon_ML_Challenge_2026/
├── LICENSE                                    # Open-source MIT License
├── pyproject.toml                             # Standard PEP 518/621 build & tool configs
├── Documentation_template.md                  # Comprehensive official methodology write-up
├── .github/
│   └── workflows/
│       └── ci.yml                             # GitHub Actions CI linting & test suite
│
├── code/
│   └── business_entity_resolution/           # Self-contained reproduction package
│       ├── README.md                          # Reproduction guide for competition auditors
│       ├── requirements.txt                   # Pinned runtime dependencies
│       ├── Documentation_template.md          # Completed methodology write-up
│       └── src/
│           ├── normalization_v3.py            # Universal AnyAscii + French legal normalizer
│           ├── blocking_v3.py                 # 7-channel multi-index blocking engine
│           ├── features_v3.py                 # 50-dimensional vectorized feature extractor
│           ├── run_hardening_now.py           # Production precision hardening & inference engine
│           ├── precision_hardening_v3.py      # Core multi-guard precision filter library
│           ├── lgbm_reranker.py               # LightGBM model trainer & 5-fold CV evaluation
│           ├── train_reranker_v3.py           # 2-pass hard negative mining training harness
│           ├── inference_sota.py              # High-throughput sharded inference engine
│           ├── generate_submission.py         # Standalone baseline reproduction script
│           ├── assemble_submission2.py        # Shard assembly & row alignment utility
│           └── validate.py                    # Schema and assertion test suite
│
├── dataset/student_resource/                  # Official challenge resources & validation tools
│   └── utils/
│       └── validate_submission.py             # Official submission formatting validator
│
└── models/
    ├── lgbm_reranker_final.txt                # Serialized production LightGBM Booster
    └── lgbm_reranker_meta.json                # Cross-validation metrics & threshold metadata
```

---

## ⚡ Quickstart & Reproduction Guide

### 1. Prerequisites & Environment Setup
- Python 3.8+ (Python 3.10 or 3.11 recommended)
- 16 GB+ RAM (32 GB+ recommended for parallel processing)

```bash
# Clone the repository
git clone https://github.com/vishalkumar-ai25/Amazon_ML_Challenge_2026.git
cd Amazon_ML_Challenge_2026

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. End-to-End Production Reproduction (Matches 0.9142 Leaderboard)
To run the full precision-hardened inference pipeline:

```bash
python3 code/business_entity_resolution/src/run_hardening_now.py
```
This produces:
- `output/matching_results.tsv` — Scored submission file identical to the public leaderboard.
- `output/candidate_pairs.tsv` — Blocking candidate set (audited for 0 subset violations).

### 3. Standalone Baseline Reproduction (CPU-Only)
To run a self-contained single-command pipeline from raw data without precomputed shards:

```bash
python3 code/business_entity_resolution/src/generate_submission.py
```

### 4. Verification with Official Evaluator
Validate formatting, row count, column headers, and subset integrity:

```bash
python3 dataset/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/student_resource/dataset/test
```
**Expected Output:**
```text
ML Challenge 2026 — submission validator
  test dir: dataset/student_resource/dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (109495 empty, 1623049 non-empty).
  candidate_pairs.tsv:  1732544 rows (12 empty, 1732532 non-empty).

PASS — no blocking issues found. Safe to submit.
```

---

## 🛡️ License & Third-Party Compliance

All software, models, and dependencies used in this pipeline strictly comply with open-source licensing and fair-play regulations:
- **LightGBM:** MIT License
- **RapidFuzz:** MIT License
- **AnyAscii:** MIT / ISC License
- **Scikit-Learn, SciPy, NumPy, Pandas:** BSD Permissive Licenses

The complete codebase in this repository is distributed under the [MIT License](LICENSE).

---

## 👥 Authors & Acknowledgments

- **Vishal Kumar** ([@vishalkumar-ai25](https://github.com/vishalkumar-ai25)) — Team Jugaad.ai / EnsembleGrandmaster
- Developed for the **Amazon ML Challenge 2026**.
