#!/bin/bash
set -e
cd ~/Amazon_ML_Challenge_2026

echo "=========================================="
echo "FAST INFERENCE V4 (32-Thread Accelerated)"
echo "Started at $(date)"
echo "=========================================="

# Run inference with 32 workers
echo "[1/2] Running inference_v4.py with 32 threads..."
venv/bin/python3 src/inference_v4.py --n-workers 32 > logs/inference_v4.log 2>&1
echo "[1/2] DONE at $(date)"

# Validate
echo "[2/2] Validating submission..."
venv/bin/python3 src/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/student_resource/dataset/test > logs/validate_v4.log 2>&1
cat logs/validate_v4.log

echo "=========================================="
echo "ALL DONE at $(date)!"
echo "Submission: output/matching_results.zip"
echo "=========================================="
