#!/bin/bash
# run_repeated_kfold.sh
# Runs full 5-fold CV for LoRA and Naive Unfreeze on new seeds to tighten the Bootstrap CI.

set -e

IMAGES_DIR="datasets/HAM10000_preprocessed"
META_CSV="datasets/HAM10000_metadata.csv"
CKPT="models/IN1K-vit.h.14-300e.pth.tar"

for SEED in 42 100 2026; do
    echo "=========================================================="
    echo "Running LoRA - All Folds - Seed $SEED"
    echo "=========================================================="
    python src/train_ijepa_lora.py \
        --images_dir "$IMAGES_DIR" \
        --metadata_csv "$META_CSV" \
        --ckpt_path "$CKPT" \
        --seed "$SEED"

    echo "=========================================================="
    echo "Running Naive Unfreeze - All Folds - Seed $SEED"
    echo "=========================================================="
    python src/train_ijepa_naive.py \
        --images_dir "$IMAGES_DIR" \
        --metadata_csv "$META_CSV" \
        --ckpt_path "$CKPT" \
        --seed "$SEED"
done

echo "Done running repeated k-folds."
