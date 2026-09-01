# Experimental Architecture: Supporting the Claim
## "LoRA-adapted I-JEPA closes X% of the gap to full fine-tuning at <5% of the trainable parameters"

This document is scoped narrowly around **one claim**. Everything here exists to make that claim statistically sound and reviewer-proof — not to explore the space broadly. Anything from the earlier roadmap (external data, cross-dataset generalization) is now secondary/optional and should not distract from this core result.

---

## 1. Precisely Define the Claim (do this before writing a line of code)

The claim has three quantities. Define them exactly, in your paper's Methods section, before reporting anything:

```
F_frozen  = Macro-F1 of frozen I-JEPA + probe head        (lower anchor)
F_full    = Macro-F1 of FULLY fine-tuned I-JEPA + probe head   (upper anchor)
F_lora    = Macro-F1 of LoRA-adapted I-JEPA + probe head   (test condition)

Gap-closed (%) = (F_lora − F_frozen) / (F_full − F_frozen) × 100
```

**Critical requirement:** `F_full` must come from fine-tuning the **I-JEPA encoder itself**, all parameters unfrozen, with the *same* probe head architecture as the other two conditions. Your current ViT-B/16 supervised result is a *different pretrained backbone entirely* (supervised ImageNet vs. I-JEPA's self-supervised objective) — using it as the "full fine-tuning" anchor conflates the adaptation method with the pretraining paradigm. Keep ViT-B and DINO as **secondary reference rows** in your table (useful context), but they are not part of the gap-closing calculation.

**You are missing one required experiment: full fine-tuning of the I-JEPA encoder.** Run this before anything else in this document — everything downstream depends on it.

---

## 2. Required Experimental Conditions

All four conditions below must share: identical lesion-grouped folds, identical probe head architecture, identical preprocessing/augmentation pipeline. Only the backbone-adaptation strategy differs.

| Condition | Backbone params trainable | Probe head trainable | Role |
|---|---|---|---|
| **A. Frozen probe** | 0% | Yes | Lower anchor (`F_frozen`) |
| **B. Full fine-tune** | 100% | Yes | Upper anchor (`F_full`) — *must run this* |
| **C. LoRA-adapted** | LoRA adapters only (target <5%) | Yes | Test condition (`F_lora`) |
| **D. Param-matched naive unfreeze** | Last-N blocks unfrozen, matched to ~same param count as C | Yes | Control — see §4.2 |

Condition D is not optional if you want a strong paper: without it, a reviewer can reasonably ask "maybe *any* method that unfreezes ~5% of parameters would do just as well as LoRA — what's special about LoRA specifically?" Condition D answers that directly.

---

## 3. Parameter Budget & Accounting Rules

Be exact and report this table in the paper — reviewers will check the arithmetic.

```python
def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, 100 * trainable / total

# Report for EVERY condition:
# - trainable params (backbone only)
# - trainable params (backbone + head, since head is trainable in all conditions —
#   report both so the <5% claim is unambiguous about what it includes/excludes)
# - % of total backbone params
```

**Decide and state explicitly whether your "<5%" claim is:**
- (a) LoRA adapters as % of backbone params only, or
- (b) LoRA adapters + probe head as % of (backbone + head) params.

These give different numbers. (a) is the more standard PEFT-literature convention (the head is small and present in every condition, so it's usually excluded from the "adaptation efficiency" accounting) — use (a), but report both in an appendix table so nobody can accuse you of cherry-picking the denominator.

**Rank selection to hit the <5% target:** for a ViT-H/14 backbone (~632M params) with LoRA on Q/K/V/O projections + FFN across all layers, work backward from the 5% budget to pick the maximum rank that stays under it, then also report one rank below that as a robustness check. Don't just pick r=8 by convention — show the arithmetic.

---

## 4. Fair-Comparison Controls (this is where papers get rejected if skipped)

### 4.1 Per-condition hyperparameter tuning
Full fine-tuning, LoRA, and the frozen probe each need **their own tuned learning rate and schedule** — using one LR for all three is not fair to any of them and will bias your result in an unpredictable direction. Run a small grid (e.g., 3–4 LR values, cosine decay, fixed epoch budget) per condition on a validation fold, pick the best per condition, then report final numbers via the full 5-fold protocol at the selected hyperparameters. Document the grid and selection process in an appendix — this preempts "did you just undertune full fine-tuning to make LoRA look better?" (the exact failure mode you already hit once with the ResNet-50 baseline).

### 4.2 The param-matched naive-unfreeze control (Condition D)
Unfreeze the last N transformer blocks (count blocks until trainable-param % ≈ your LoRA %), keep everything else frozen, train with the same tuned-LR protocol as Condition C. This isolates **LoRA's structural benefit** from **"just having a similar parameter budget."** If LoRA beats this control at matched budget, that's a real, citable, structural finding — not just an efficiency footnote.

### 4.3 Guard against an inflated `F_full`
Full fine-tuning on a dataset this size (even with 5-fold CV) is prone to overfitting, which could artificially *lower* `F_full` and inflate your gap-closed percentage in a way that looks good but isn't trustworthy. Use early stopping on a held-out validation split within each training fold, and report the train/val loss curves in supplementary material so a reviewer can verify convergence wasn't cut short or run past overfitting.

### 4.4 Handle a negative or >100% result honestly
It's entirely possible LoRA-adapted I-JEPA **matches or beats** full fine-tuning (small, imbalanced datasets often favor lighter-touch adaptation, since full fine-tuning of a large ViT on ~8K training images per fold is itself prone to overfitting). If `F_lora ≥ F_full`, report gap-closed as "≥100%" and frame it as a *stronger* result ("LoRA not only closes the gap but exceeds full fine-tuning while using <5% of trainable parameters — consistent with full fine-tuning overfitting on this dataset size"), rather than treating it as a broken calculation.

---

## 5. Statistical Methodology

The gap-closed percentage is a **ratio of two differences**, each of which is itself estimated with fold-to-fold variance — don't report a single point estimate without an interval.

### 5.1 Bootstrap CI on the gap-closed statistic
```python
import numpy as np

def bootstrap_gap_closed(f_frozen_folds, f_full_folds, f_lora_folds, n_boot=10000, seed=0):
    """
    Each *_folds is a length-5 array of per-fold macro-F1 scores.
    Returns point estimate and 95% CI for gap-closed %.
    """
    rng = np.random.default_rng(seed)
    n = len(f_frozen_folds)
    boot_stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # resample folds with replacement
        ff = np.mean(np.array(f_frozen_folds)[idx])
        fu = np.mean(np.array(f_full_folds)[idx])
        fl = np.mean(np.array(f_lora_folds)[idx])
        denom = fu - ff
        if abs(denom) < 1e-6:
            continue  # skip degenerate resamples where anchors nearly coincide
        boot_stats.append(100 * (fl - ff) / denom)
    boot_stats = np.array(boot_stats)
    point = 100 * (np.mean(f_lora_folds) - np.mean(f_frozen_folds)) / (np.mean(f_full_folds) - np.mean(f_frozen_folds))
    ci_low, ci_high = np.percentile(boot_stats, [2.5, 97.5])
    return point, (ci_low, ci_high)
```
Report as: *"LoRA-adapted I-JEPA closes 71.3% (95% CI: 58.2–84.1%) of the gap to full fine-tuning."* This single sentence with a CI is far stronger than a bare percentage.

### 5.2 Paired significance tests
- `F_lora` vs. `F_frozen` (does adaptation help at all — should be significant and is the easy check)
- `F_lora` vs. Condition D (param-matched naive unfreeze) — this is your key structural claim, needs to be significant to say "LoRA specifically helps," not just "more trainable params help"
- `F_lora` vs. `F_full` (is the residual gap statistically distinguishable from zero, or has LoRA fully closed it)

Use paired Wilcoxon signed-rank test across the 5 fold-level scores for each comparison (paired because folds are matched across conditions).

---

## 6. LoRA Rank Sweep (secondary figure, strengthens the paper)

Run Condition C at `r ∈ {4, 8, 16, 32}`, plot trainable-param-% on the x-axis vs. macro-F1 (with fold error bars) on the y-axis, with horizontal reference lines for `F_frozen` and `F_full`. This is a clean, single-figure way to show: (a) where the <5% operating point sits on the curve, (b) that you didn't cherry-pick the rank that happened to look best, and (c) diminishing returns as rank increases toward full fine-tuning's parameter count.

---

## 7. Results Table Template

```
| Condition                    | Backbone train% | Total train%(+head) | Macro-F1 (mean±std) | Gap-closed % (95% CI) |
|-------------------------------|-----------------|----------------------|----------------------|------------------------|
| A. Frozen I-JEPA + probe      | 0.0%            | X%                   | F_frozen ± s         | —  (anchor)            |
| D. Naive unfreeze (matched)   | ~4.x%           | ~X%                  | F_D ± s              | —  (control)           |
| C. LoRA-adapted I-JEPA        | <5.0%           | X%                   | F_lora ± s           | Y% (CI_low–CI_high)    |
| B. Full fine-tune I-JEPA      | 100.0%          | X%                   | F_full ± s           | 100% (anchor)          |
| —— secondary reference rows (different backbones, not part of gap calc) ——
| ViT-B/16 supervised           | 100.0%          | —                    | 0.6371 ± 0.0267      | n/a                    |
| DINO (frozen probe)           | 0.0%            | —                    | 0.5947 ± 0.0251      | n/a                    |
```

Wall-clock training time and peak GPU memory per condition go in a companion efficiency table — this materially supports the "<5% params" framing with a concrete compute-cost story, which workshop reviewers like.

---

## 8. Priority Order

1. **Run Condition B (full I-JEPA fine-tuning) — you don't have this yet and the entire claim depends on it.**
2. Set up Condition A (you already have this — frozen probe, existing results) and confirm identical folds/head/preprocessing vs. B.
3. Run Condition C (LoRA) at your target rank, tuned LR — compute the point-estimate gap-closed %.
4. Run Condition D (param-matched naive unfreeze) — this is the control that makes "LoRA specifically" a defensible claim rather than "more params helps."
5. Bootstrap CI + paired significance tests (§5) — turns point estimates into a claim a reviewer can trust.
6. LoRA rank sweep (§6) — secondary figure, do this once the core 4-condition result is solid.
7. Efficiency table (wall-clock, peak memory) — cheap to collect alongside step 1–4, don't leave it for last.

Steps 1–5 are the paper's core result. Step 6–7 are strengthening figures, not prerequisites — if you're tight on time before a deadline, 1–5 alone supports the claim.
