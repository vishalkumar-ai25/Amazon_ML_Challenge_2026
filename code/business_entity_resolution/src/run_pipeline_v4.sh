#!/bin/bash
set -e
cd ~/Amazon_ML_Challenge_2026

echo "=========================================================="
echo "AMAZON ML CHALLENGE 2026: SOTA V4 AUTOMATED PIPELINE"
echo "Started at $(date)"
echo "=========================================================="

mkdir -p logs models output

# 1. Run train_v4.py with 32 parallel workers
echo "[PIPELINE] Launching train_v4.py..."
venv/bin/python3 src/train_v4.py --sample-size 40000 --n-workers 32 > logs/train_v4.log 2>&1

echo "[PIPELINE] Training completed at $(date)!"

# 2. Wait for embed_all_names.py to be completely finished (if still running)
echo "[PIPELINE] Verifying test embeddings..."
while pgrep -f embed_all_names > /dev/null; do
    echo "Waiting for embed_all_names to complete..."
    sleep 30
done

# Verify embedding file exists
if [ ! -f "embeddings/all_names_emb.npy" ] || [ ! -f "embeddings/name_to_idx.pkl" ]; then
    echo "ERROR: Embeddings file not found!"
    exit 1
fi

echo "[PIPELINE] Embeddings verified! Launching inference_v4.py..."
venv/bin/python3 src/inference_v4.py --n-workers 32 > logs/inference_v4.log 2>&1

echo "[PIPELINE] Inference completed at $(date)!"

# 3. Validate submission
echo "[PIPELINE] Validating submission..."
venv/bin/python3 src/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/student_resource/dataset/test > logs/validate_v4.log 2>&1
cat logs/validate_v4.log

echo "=========================================================="
echo "PIPELINE V4 COMPLETE at $(date)!"
echo "Final submission archive: output/matching_results.zip"
echo "=========================================================="
