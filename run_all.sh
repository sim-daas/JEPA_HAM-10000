#!/bin/bash
set -e

echo "Starting HAM10000 Evaluation Pipeline..."

# 1. Full Fine-Tuning (Condition B)
echo "Running Full Fine-Tuning (takes ~4.5 hours)..."
python src/train_ijepa_full.py \
    --images_dir datasets/HAM10000_preprocessed \
    --metadata_csv datasets/HAM10000_preprocessed/metadata.csv \
    --ckpt_path models/IN1K-vit.h.14-300e.pth.tar

# 2. Param-Matched Naive Unfreeze (Condition D)
echo "Running Param-Matched Naive Unfreeze..."
python src/train_ijepa_naive.py \
    --images_dir datasets/HAM10000_preprocessed \
    --metadata_csv datasets/HAM10000_preprocessed/metadata.csv \
    --ckpt_path models/IN1K-vit.h.14-300e.pth.tar

# 3. Statistical Analysis
echo "Running Statistical Analysis (Bootstrap CI & Wilcoxon)..."
python src/stats_analysis.py > logs/stats_summary.txt

echo "Pipeline Complete! Results saved to logs/stats_summary.txt"
