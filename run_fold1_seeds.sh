#!/bin/bash
# run_fold1_seeds.sh
# Tests LoRA robustness on Fold 1 across different random initialization seeds.

set -e

IMAGES_DIR="datasets/HAM10000_preprocessed"
META_CSV="datasets/HAM10000_metadata.csv"
CKPT="models/IN1K-vit.h.14-300e.pth.tar"

for SEED in 42 100 2026; do
    echo "=========================================================="
    echo "Running LoRA on Fold 1 with Seed $SEED"
    echo "=========================================================="
    python src/train_ijepa_lora.py \
        --images_dir "$IMAGES_DIR" \
        --metadata_csv "$META_CSV" \
        --ckpt_path "$CKPT" \
        --only_fold 0 \
        --seed "$SEED"
done

echo "Done running LoRA Fold 1 seeds."
