# HAM10000 Research Progress & Evaluation Log

## 1. Frozen Probe Baselines (`F_frozen`)
**Goal**: Evaluate the linear separability of frozen representations from I-JEPA (ViT-H/14) and DINOv2 (ViT-B/14) using MLP probes.

- **I-JEPA (ViT-H/14) + 2-layer MLP**:
  - Macro-F1: 0.6203
- **I-JEPA (ViT-H/14) + 5-layer MLP**:
  - Macro-F1: 0.56 (Significantly dropped; confirms 5-layer MLP drastically overfits on ~8,000 training images per fold).
- **DINOv2 (ViT-B/14) + 2-layer MLP**:
  - Macro-F1: 0.5995

*Decision*: We must use the 2-layer MLP as the standard probe architecture going forward to prevent overfitting on this dataset.

## 2. Supervised Fine-Tuning Baselines
**Goal**: Fix the 13-point gap observed previously by strictly controlling the training regime (caching initialization, freezing BatchNorm2d, CosineAnnealingLR) and run an equivalent-capacity model.

- **ResNet-50 Supervised**:
  - (Previous broken baseline was poor; skipped in favor of ViT comparison).
- **ViT-Base (`vit_base_patch16_224`) Supervised**:
  - Macro-F1: 0.6371 (Successfully overtook the frozen I-JEPA baseline of 0.6203, establishing a proper supervised baseline).

## 3. Full Fine-Tuning (`F_full`) - *Pending Execution*
**Goal**: Fully fine-tune the I-JEPA ViT-H/14 architecture end-to-end to establish the upper bound for the LoRA gap-closure calculation.

**Status**: Script (`src/train_ijepa_full.py`) implemented, verified with mixed-precision (AMP) and gradient accumulation, and handles intra-fold early stopping.

### Execution Commands (Real Dataset)
To run the full fine-tuning on the actual HAM10000 dataset, execute the following commands on a machine with a 16GB GPU:

```bash
# 1. Ensure the environment is active
conda activate jepa

# 2. Run the training script with an effective batch size of 32 (micro_batch=4, accumulation=8)
python src/train_ijepa_full.py \
    --images_dir datasets/HAM10000_preprocessed \
    --metadata_csv datasets/HAM10000_metadata.csv \
    --ckpt_path models/IN1K-vit.h.14-300e.pth.tar \
    --epochs 10 \
    --micro_batch_size 4 \
    --accumulation_steps 8
```

*(If OOM occurs, reduce to `--micro_batch_size 2` and `--accumulation_steps 16`)*

**Results**: 
*(To be populated after execution)*
- Macro F1: ...
- Macro Precision: ...
- Macro Recall: ...
