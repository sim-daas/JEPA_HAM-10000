#!/bin/bash
# run_fold0_extra_seeds.sh
# Tests Fold 0 near-tie with 2 extra seeds (420, 2024)

set -e

IMAGES_DIR="datasets/HAM10000_preprocessed"
META_CSV="datasets/HAM10000_metadata.csv"
CKPT="models/IN1K-vit.h.14-300e.pth.tar"

for SEED in 420 2024; do
    echo "=========================================================="
    echo "Running LoRA on Fold 0 with Seed $SEED"
    echo "=========================================================="
    python src/train_ijepa_lora.py \
        --images_dir "$IMAGES_DIR" \
        --metadata_csv "$META_CSV" \
        --ckpt_path "$CKPT" \
        --only_fold 0 \
        --seed "$SEED"

    echo "=========================================================="
    echo "Running Naive on Fold 0 with Seed $SEED"
    echo "=========================================================="
    python src/train_ijepa_naive.py \
        --images_dir "$IMAGES_DIR" \
        --metadata_csv "$META_CSV" \
        --ckpt_path "$CKPT" \
        --only_fold 0 \
        --seed "$SEED"
done

echo "Done running extra seeds for Fold 0."
