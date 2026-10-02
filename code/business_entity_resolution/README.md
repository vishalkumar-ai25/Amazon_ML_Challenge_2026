# Amazon ML Challenge 2026: Business Entity Resolution Pipeline

**Team:** Jugaad.ai  
**Best Leaderboard Score:** Macro $F_{0.5} = 0.9142$

## Overview

This repository contains the self-contained, end-to-end entity resolution pipeline for multi-source business record linkage across ~28 million records. The solution links entities across 3 heterogeneous, noisy data sources without primary keys, handling multilingual content (English, French, Hindi, Telugu, Tamil, Bengali), severe typographical noise, and zero-shot generalization to unseen countries (France).

## Pipeline Architecture

```
Input Data (S1: 1.73M queries, S2+S3: 9.97M candidates)
    │
    ├─ 1. Universal Phonetic Normalization (src/normalization_v3.py)
    │     AnyAscii transliteration, French corporate suffixes, consonant skeletons
    │
    ├─ 2. 7-Channel Multi-Index Blocking (src/blocking_v3.py)
    │     Composite TF-IDF, Address Jaccard, Char N-Grams, Sorted-Token TF-IDF,
    │     Postal Index, Phonetic Keys, Prefix Index
    │     → Top-50 candidates per query, >96% recall ceiling
    │
    ├─ 3. 50-Feature Pairwise Scoring (src/features_v3.py)
    │     String distances (RapidFuzz), Premise/HN Match, Postal Match, State Match,
    │     Channel scores, N-gram overlaps
    │
    ├─ 4. LightGBM Reranker (src/lgbm_reranker.py, src/train_reranker_v3.py)
    │     Country-calibrated decision thresholds:
    │     US: τ = 0.910  |  India: τ = 0.825  |  France: τ = 0.940
    │
    ├─ 5. Precision Hardening Multi-Guard (src/run_hardening_now.py)
    │     Premise conflict guard (-68k FPs), Postal conflict guard (-1k FPs),
    │     Chain-store disambiguation (-7.6k FPs), Singleton guard, Degree bounds
    │
    └─ 6. Global Greedy M2O Bipartite Resolution
          → output/matching_results.tsv (Score: 0.9142)
          → output/candidate_pairs.tsv
```

## Directory Structure

```
code/business_entity_resolution/
├── README.md                    # This reproduction guide
├── requirements.txt             # Pinned dependency versions
├── Documentation_template.md    # Methodology write-up
└── src/
    ├── normalization_v3.py      # Universal normalizer (AnyAscii + French suffixes)
    ├── blocking_v3.py           # 7-channel multi-index blocking engine
    ├── features_v3.py           # 50-feature pairwise feature extractor
    ├── lgbm_reranker.py         # LightGBM model trainer & CV evaluation
    ├── train_reranker_v3.py     # Scaled trainer with 2-pass hard negative mining
    ├── run_hardening_now.py     # Production precision hardening & output generation
    ├── precision_hardening_v3.py# Core precision hardening filter library
    ├── inference_sota.py        # Distributed blocking and scoring engine
    ├── generate_submission.py   # Standalone baseline reproduction script
    ├── validate.py              # Assertions and format checks
    ├── assemble_submission2.py  # Country shard merger
    ├── postprocess_filters.py   # Auxiliary post-processing filters
    └── ...                      # Full modular pipeline source files
```

## Environment Setup

Python 3.8+ is supported. Install dependencies:

```bash
pip install -r requirements.txt
```

## Reproducing Results

### 1. Generating Outputs via the Hardened Pipeline (Matches 0.914 Submission)

To reproduce the exact submission outputs with all precision-hardening guards applied:

```bash
python3 src/run_hardening_now.py
```

This generates:
- `output/matching_results.tsv` — Final matching results (identical to leaderboard submission)
- `output/candidate_pairs.tsv` — Candidate pairs from the blocking stage

### 2. Standalone Baseline Reproduction (CPU Only, No External Cache)

For a self-contained single-command run from raw data:

```bash
python3 src/generate_submission.py
```

## Validating Submission Output

Validate the generated outputs against all competition formatting and integrity constraints:

```bash
python3 dataset/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/student_resource/dataset/test
```

## Open-Source Licensing Compliance

All third-party libraries and models utilized (`LightGBM`, `RapidFuzz`, `AnyAscii`, `Scikit-Learn`, `SciPy`, `NumPy`, `Pandas`) are distributed under MIT and BSD permissive licenses, fully compliant with competition rules.
