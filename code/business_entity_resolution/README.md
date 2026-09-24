# Amazon ML Challenge 2026: Business Entity Resolution Pipeline

## Overview
This repository contains the complete, self-contained end-to-end entity resolution pipeline for the Amazon ML Challenge 2026. The solution determines which records across 3 independent data sources refer to the same real-world business entity using high-recall full-text blocking and metric-calibrated precision thresholding.

## Pipeline Architecture
1. **Defensive Preprocessing**: Unicode cleaning, noise token removal (`--`, `<<`), URL/web domain stripping, and lowercase normalization.
2. **High-Recall Full-Text Indexing**: Synthesizes a composite document `2 * normalized_name + normalized_address` per record. This overcomes severe regional script mismatches (e.g. Telugu/Hindi vs English names) and typographical alterations by leveraging stable address co-occurrences.
3. **Partitioned In-Memory Blocking**: Evaluates queries strictly within country partitions (France, India, US) with sparse matrix linear algebra, keeping RAM footprint under 2.5 GB.
4. **Candidate Generation**: Retrieves top-30 candidates per Source 1 entity for `candidate_pairs.tsv` (blocking recall ~90%).
5. **Calibrated Decision Thresholding**: Applies a precision-biased threshold ($\tau = 0.780$) directly optimized for the macro-averaged $F_{0.5}$ evaluation metric, eliminating false merges while capturing high-confidence matches.
6. **Strict Schema & Order Compliance**: Outputs are generated with 100% preservation of the original `test_source1.tsv` row ordering.

## Environment Setup
Python 3.8+ is supported. Install dependencies via pip:
```bash
pip install -r requirements.txt
```

## Reproducing Results

To generate both `matching_results.tsv` and `candidate_pairs.tsv` from the raw dataset:

```bash
# Run the end-to-end inference pipeline
python src/generate_submission.py
```

The output files will be written to:
- `output/matching_results.tsv` (Leaderboard submission file)
- `output/candidate_pairs.tsv` (Blocking candidate set)

## Validating Submission Output
Validate the generated output files against all formatting and integrity constraints:
```bash
python3 dataset/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/student_resource/dataset/test
```
