# Project Context Document: I-JEPA / HAM10000 LoRA Efficiency Study

**Purpose of this document:** a complete, chronological record of the research conversation to date — goals, decisions, experiment results, bugs found and fixed, statistical methodology, and the current validated state of the core claim. Intended as onboarding context for a new session, a collaborator, or the paper-writing process itself.

**Target venue:** AAAI workshop tier (e.g., a health-AI workshop such as W3PHIAI — confirm current year's CFP before finalizing).

**Current headline claim (validated, see §7):**
> LoRA-adapted I-JEPA closes ~99% of the performance gap to full fine-tuning while updating <2% of parameters, and provides a genuine (though not uniformly robust) structural advantage over a parameter-matched naive top-layer unfreeze.

---

## 1. Project Goal Evolution

| Stage | Goal |
|---|---|
| Initial | Basic-novelty paper for a local/national Indian conference; evaluate I-JEPA frozen representations on HAM10000 skin lesion classification via a shallow MLP probe. |
| Revision 1 | Add mandatory baselines (supervised fine-tuning, alternative SSL backbone) after recognizing a probing-only study has no comparison point. |
| Revision 2 | Push toward NeurIPS-workshop tier: added rigor requirements (lesion-grouped CV, statistical testing, ablations) and reframed novelty around LoRA-based parameter-efficient adaptation. |
| Revision 3 (current) | Narrowed scope to rigorously support one specific, falsifiable claim: *"LoRA-adapted I-JEPA closes X% of the gap to full fine-tuning at <5% of trainable parameters."* Target shifted to AAAI workshop tier. |

**Key reframing principle established early and reaffirmed throughout:** this is not a "beat SOTA" paper. Published HAM10000 numbers in the 0.85–0.90 macro-F1 range are frequently not apples-to-apples (image-level rather than lesion-grouped splits, external data, different metrics/protocols). The project's legitimate contribution is a **rigorous, leakage-safe empirical/efficiency study**, not a leaderboard result.

---

## 2. Key Technical Facts Established (verified via search / correction during the conversation)

- **I-JEPA official checkpoints (`facebookresearch/ijepa`, now archived):** only **ViT-H/14, ViT-H/16, and ViT-g/16** were released (on ImageNet-1K/22K). **There is no public "ViT-G" (~1.8B param) checkpoint** — an early draft of the architecture doc incorrectly assumed one existed. Use ViT-H/14 @ 224×224 as the primary backbone.
- **Checkpoint loading:** extract `checkpoint['target_encoder']` (the EMA encoder — highest-quality representations), strip `module.`/`backbone.` prefixes, load with `strict=False` into a bare `timm` ViT with `num_classes=0`.
- **Patch token count is checkpoint-dependent:** ViT-H/14 @ 224×224 yields 256 tokens (16×16 grid), not 196 — don't hardcode; 196 only applies to the ViT-g/16 @ 224×224 (patch size 16) checkpoint.
- **ISIC 2019 is not independent external data** — its 25,331-image training set is HAM10000 + BCN_20000 + MSK merged. Using "ISIC 2019" as a block for extra training/pretraining data would leak HAM10000 test images. Only BCN_20000 and MSK subsets (excluding the HAM10000-derived portion) are safe as genuinely external data.
- **Grad-CAM is not appropriate for ViT explainability** (designed for conv feature maps); use **Attention Rollout** or a transformer-specific relevance method instead.
- **YOLO (any version, incl. v12) is not an appropriate baseline** for this task — it's a detection/segmentation architecture; HAM10000 as set up here is whole-image classification with no bounding-box supervision. Rejected as a baseline option.
- **Splitting must be lesion-grouped, not image-grouped:** HAM10000 has multiple images per `lesion_id`; naive image-level splitting leaks the same lesion across train/test. All experiments use `StratifiedGroupKFold` grouped by `lesion_id`.

---

## 3. Chronological Experiment Log

### 3.1 First-pass frozen probe + naive baselines (early, later partially corrected)
```
I-JEPA (frozen probe):     Macro F1 = 0.6187 ± 0.0158
DINO (frozen probe):       Macro F1 = 0.5947 ± 0.0251
ResNet-50 (supervised):    Macro F1 = 0.4855 ± 0.0244   ← FLAGGED AS BROKEN (see §4.1)
```

### 3.2 Corrected supervised baselines
```
ResNet-50 (supervised, corrected regime):   Macro F1 = 0.5479 ± 0.0182
ViT-B/16  (supervised, corrected regime):   Macro F1 = 0.6371 ± 0.0267
```
Result: ViT-B/16 supervised now sensibly outperforms the frozen I-JEPA probe — expected, credible ordering. However, ViT-B is a **different backbone/pretraining paradigm** than I-JEPA, so it was later excluded from the core gap-closing calculation (kept only as a secondary reference row) — see §4.2.

### 3.3 Full I-JEPA fine-tuning (`F_full`) — required missing experiment, identified and run
First run (initial aggregate report):
```
Macro F1 = 0.7315 ± 0.0171
```
Later, once the exact per-fold array was recovered from checkpoints (see §4.3), the canonical value became:
```
F_full folds: [0.7393, 0.7529, 0.7260, 0.7014, 0.7222]   Mean = 0.7284
```
Training dynamics were reviewed and judged credible: clean val-F1 climb to epoch 7 (peak ~0.826), then degradation while train loss kept falling — classic, expected full-FT overfitting signature; checkpoint-on-best-val correctly saved the epoch 7 weights. Compute cost: ~5:24/epoch × 10 epochs × 5 folds ≈ 4.5 GPU-hours.

### 3.4 LoRA adaptation (`F_lora`), rank r=16 — single-seed initial result
```
Macro F1 = 0.6947 ± 0.0582
Macro Recall = 0.7184 ± 0.0494
Trainable params = 11,175,109 / 641,937,349 = 1.74%
Peak GPU memory = ~5,790 MB
Per-fold array: [0.5944, 0.7434, 0.7396, 0.6630, 0.7332]
```
Gap-closed (initial): (0.6947−0.6187)/(0.7315−0.6187) × 100 ≈ **67.4%**, later recalculated against the canonical arrays as **69.0%**.

### 3.5 Condition D — param-matched naive unfreeze, single-seed initial result
Unfroze only `model.encoder.blocks[-1].mlp` (13,774,087 params, 2.18% budget — matched to LoRA's 1.74%):
```
Macro F1 = 0.6864
Per-fold array: [0.7008, 0.7111, 0.6811, 0.6748, 0.6640]
```

### 3.6 First statistical pass (single-seed arrays)
```
Bootstrap CI on gap-closed:        69.0%  [10.9% – 106.6%]   (very wide)
Wilcoxon LoRA vs Frozen:           p = 0.0625  (borderline; full sweep)
Wilcoxon Full vs LoRA:             p = 0.3125  (no significant difference)
Wilcoxon LoRA vs Naive:            p = 0.8125  (statistically indistinguishable)
```
**Initial conclusion at this stage:** LoRA closes most of the gap to full fine-tuning, but shows **no structural advantage over naive parameter-matched unfreezing** — i.e., the efficiency gain appeared to come from parameter *count*, not LoRA's specific low-rank structure. This was the finding presented to the "AAAI acceptance chances" analysis (§6).

---

## 4. Problems Encountered and Solutions

### 4.1 Problem: ResNet-50 baseline was implausibly weak (0.4855), worse than frozen SSL probes
A properly fine-tuned supervised CNN should normally beat a frozen linear/MLP probe — the reversal was a red flag for an undertrained/misconfigured baseline, not a real finding.
**Solution:** diagnostic checklist provided (identical folds across scripts, LR/schedule appropriate for full fine-tuning vs. head-only training, weighted-CE actually applied, correct ImageNet normalization stats, BatchNorm stability at small batch size, convergence check via loss curves). User corrected the training regime and reran; ResNet-50 rose to 0.5479 and ViT-B/16 (0.6371) became the primary, credible supervised baseline.

### 4.2 Problem: the "full fine-tuning" anchor was the wrong backbone for the LoRA gap-closing claim
ViT-B/16 (supervised ImageNet pretraining) is architecturally and pretraining-paradigm different from I-JEPA. Using it as `F_full` would conflate "LoRA vs. full fine-tuning" with "I-JEPA pretraining vs. supervised pretraining."
**Solution:** required a new experiment — full end-to-end fine-tuning of the I-JEPA encoder itself, same probe head, same folds — as the true `F_full` anchor. This became §3.3. ViT-B/16 and DINO were retained only as secondary reference rows, excluded from the gap-closed calculation.

### 4.3 Problem: `F_full` per-fold array was not logged (only aggregate + Fold 1 existed)
Retraining would have cost another ~4.5 GPU-hours.
**Solution:** recommended reloading the 5 saved best-checkpoint files and re-running test-set evaluation only (~26s/fold, not full retrain) to recover the exact per-fold array without retraining. This was resolved — the canonical `F_full` array in §3.3 reflects this.

### 4.4 Problem: `F_lora`'s Fold 1 result (0.5944) was a severe outlier (val F1 ~0.81 → test F1 ~0.59), driving high fold-to-fold variance (±0.0582) and a very wide bootstrap CI
Two competing hypotheses: (a) genuine LoRA-specific optimization instability on this data split, or (b) a one-off bad random seed.
**Solution:** reran LoRA on Fold 1 alone with three fixed seeds (42, 100, 2026):
```
Seed 42:   0.6930
Seed 100:  0.6718
Seed 2026: 0.6925
```
All three landed in a tight 0.67–0.69 band, consistent with the rest of the fold results and with the Naive control's 0.7008 on the same split — **confirmed hypothesis (b): a one-off stochastic optimization failure, not a structural or data-driven issue.**

### 4.5 Problem: fixing only the anomalous fold would constitute selective re-sampling (cherry-picking)
Swapping in a multi-seed average for Fold 1 alone, while leaving Folds 2–5 as single-seed, would bias the result even if well-intentioned.
**Solution:** adopted a **symmetric 3-seed × 5-fold protocol**, applied uniformly to both `F_lora` and `F_naive` (the two conditions being directly compared for the structural-advantage claim). `F_frozen` and `F_full` were left single-seed, justified by their already-tight fold-to-fold variance (±0.017 for full-FT) — documented explicitly as a limitation, not hidden.

### 4.6 Final multi-seed raw data (source of truth)
```
Paradigm: LORA
Seed   | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean F1
42     | 0.6930 | 0.7317 | 0.7428 | 0.7498 | 0.7494 | 0.7334
100    | 0.6718 | 0.7642 | 0.7409 | 0.7238 | 0.7194 | 0.7240
2026   | 0.6925 | 0.7514 | 0.7206 | 0.7442 | 0.7144 | 0.7246

Paradigm: NAIVE
Seed   | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean F1
42     | 0.6967 | 0.6822 | 0.6649 | 0.6443 | 0.6851 | 0.6747
100    | 0.6670 | 0.6708 | 0.6673 | 0.6737 | 0.6716 | 0.6701
2026   | 0.6872 | 0.7009 | 0.7062 | 0.6293 | 0.6762 | 0.6799
```
Per-fold means (averaged across the 3 seeds) — **verified independently by recomputing from this raw table; matches reported arrays exactly:**
```
F_lora  (per fold): [0.6858, 0.7491, 0.7348, 0.7393, 0.7277]   Mean = 0.7273
F_naive (per fold): [0.6836, 0.6846, 0.6795, 0.6491, 0.6776]   Mean = 0.6749
```

### 4.7 Problem (flagged, not yet fixed as of this document): stale precision/recall figures in the progress log
The "Precision vs. Recall Skew" write-up still cites the old single-seed `F_lora` numbers (F1=0.6947, Recall=0.7184) rather than the new multi-seed mean (F1=0.7273). **Action item:** recompute macro precision/recall from the multi-seed-aggregated predictions before this goes into the confusion-matrix / failure-analysis figure.

---

## 5. Statistical Methodology

### 5.1 Gap-closed metric definition
```
Gap-closed (%) = (F_lora − F_frozen) / (F_full − F_frozen) × 100
```
`F_full` must come from the same backbone (I-JEPA) as `F_lora`/`F_frozen` — see §4.2.

### 5.2 Bootstrap CI (fold-resampling)
```python
import numpy as np

def bootstrap_gap_closed(f_frozen_folds, f_full_folds, f_lora_folds, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(f_frozen_folds)
    boot_stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ff = np.mean(np.array(f_frozen_folds)[idx])
        fu = np.mean(np.array(f_full_folds)[idx])
        fl = np.mean(np.array(f_lora_folds)[idx])
        denom = fu - ff
        if abs(denom) < 1e-6:
            continue
        boot_stats.append(100 * (fl - ff) / denom)
    boot_stats = np.array(boot_stats)
    point = 100 * (np.mean(f_lora_folds) - np.mean(f_frozen_folds)) / (np.mean(f_full_folds) - np.mean(f_frozen_folds))
    ci_low, ci_high = np.percentile(boot_stats, [2.5, 97.5])
    return point, (ci_low, ci_high)
```

### 5.3 Paired significance testing
Paired Wilcoxon signed-rank test across fold-level scores (paired because folds are matched across conditions). **Known limitation, explicitly documented:** with N=5 folds, the minimum achievable two-sided p-value is 0.0625 (a full 5/5 sweep) — the test **cannot reach p < 0.05 regardless of effect size**, since 2×(1/2)⁵ = 0.0625 and only N≥6 independent samples could cross the conventional threshold. A p=0.0625 result should be described as *"consistent with, but not reaching, conventional significance,"* not as "definitive."

**Possible strengthening (not yet executed, optional):** run a second independent `StratifiedGroupKFold` partition (different `random_state`) to obtain 10 fold-level comparisons instead of 5, potentially crossing p<0.05 if the effect is real. **Caveat to document if used:** folds from a second partition of the same finite lesion pool are not fully statistically independent of the first partition's folds (some lesions recur across differently-combined folds), so resulting p-values would be somewhat anti-conservative — report as "suggestive," not as equivalent to 10 fully independent trials.

---

## 6. AAAI Workshop Acceptance Analysis (performed at the single-seed stage, §3.6 data — since superseded by §7, but the reasoning framework remains relevant)

Key point raised: the original claim framing ("LoRA closes X% of the gap," implying LoRA *specifically* is special) was **not supported** by the single-seed Condition D result at that time (p=0.8125, statistically indistinguishable from naive unfreeze). Recommended reframing around the actually-supported claim: *"parameter-efficient adaptation broadly (LoRA or naive) closes most of the gap; the specific mechanism doesn't matter as much as the parameter budget"* — until the Fold 1 diagnostic (§4.4) and symmetric re-seeding (§4.5–4.6) later reversed this finding in LoRA's favor.

Strengths noted: methodological rigor (lesion-grouped CV, matched heads/budgets, bootstrap CI, paired testing) above typical workshop-submission standard; a genuine efficiency story (5.79GB vs. ~16GB peak memory) independent of the mechanism question; a legitimate benchmark contribution (I-JEPA not previously evaluated on HAM10000).

Risks flagged (several since addressed): small N=5 (structural, still applies — see §5.3), wide CI at the time (since narrowed — see §7), Fold 1 anomaly (resolved — see §4.4), need for a "why" explanation of the LoRA-vs-naive result in the discussion section (still open — see §8).

---

## 7. Current Validated State (most recent, multi-seed, symmetric-protocol results)

### 7.1 Canonical arrays
```
F_frozen: [0.6343, 0.6179, 0.6085, 0.6183, 0.6086]   Mean = 0.6175   (single-seed)
F_full:   [0.7393, 0.7529, 0.7260, 0.7014, 0.7222]   Mean = 0.7284   (single-seed)
F_lora:   [0.6858, 0.7491, 0.7348, 0.7393, 0.7277]   Mean = 0.7273   (3-seed mean per fold)
F_naive:  [0.6836, 0.6846, 0.6795, 0.6491, 0.6776]   Mean = 0.6749   (3-seed mean per fold)
```

### 7.2 Headline result
```
Gap-closed (%) = (0.7273 − 0.6175) / (0.7284 − 0.6175) × 100 = 99.0%
Bootstrap 95% CI: [73.2%, 123.2%]
```

### 7.3 Statistical tests
```
LoRA vs. Frozen:         p = 0.0625  (full 5/5 sweep — sample-size ceiling, see §5.3 caveat)
Full FT vs. LoRA:        p = 0.8125  (statistically indistinguishable — genuine tie)
LoRA vs. Naive Unfreeze: p = 0.0625  (full 5/5 sweep — see §7.4 caveat on Fold 0)
```

### 7.4 Important caveat on the LoRA-vs-Naive "sweep" (verified by hand from raw data)
Per-fold LoRA − Naive margins:
```
Fold 0: 0.6858 − 0.6836 = +0.0022   ← negligible, within noise
Fold 1: 0.7491 − 0.6846 = +0.0645
Fold 2: 0.7348 − 0.6795 = +0.0553
Fold 3: 0.7393 − 0.6491 = +0.0902
Fold 4: 0.7277 − 0.6776 = +0.0501
```
**Fold 0's margin (0.0022) is not clearly distinguishable from seed noise.** The technically-correct 5/5 sweep and p=0.0625 should not be reported as "definitive" or "strict dominance on every fold." The defensible, accurate claim is: *"LoRA shows a clear, substantial advantage over the parameter-matched naive-unfreeze control on 4 of 5 folds (margins 0.050–0.090); the fifth fold shows a negligible difference (0.002) not clearly distinguishable from noise."*

### 7.5 Compute / efficiency summary
```
Full fine-tuning:  ~4.5 GPU-hours total (5 folds × 10 epochs), 100% params trainable
LoRA (r=16):       ~5,790 MB peak GPU memory, 1.74% params trainable (11.18M / 641.9M)
Naive unfreeze:    2.18% params trainable (13.77M), matched-budget control
```

---

## 8a. Fold 0 Robustness Spot-Check — RESOLVED

Following the concern raised above, the paper draft was corrected using **fix (a)**: the primary claim now rests solely on the symmetric 3-seed protocol (4/5 folds show a clear, substantial LoRA advantage; Fold 0 is reported honestly as a near-tie, margin 0.0022, not distinguishable from noise at that sample size). The 5-seed Fold 0 result (`LoRA = 0.6993`, `Naive = 0.6840`, margin +0.0153) is now correctly isolated in a clearly-labeled **"Exploratory Note"**, explicitly stated as not run symmetrically across all folds and not to be treated as confirmatory. It is no longer cited in the Abstract or "Key Claim II" as settled evidence. **This closes out the methodological concern raised in the previous version of this log.**

Margin range was also corrected throughout to the verified **0.050–0.090** (previously mis-stated as "0.03–0.09" in one draft).

## 8b. Resolved: Per-Fold Class Distribution (closes action item, previously open)

A supplementary table confirming per-fold class counts was produced and reviewed — class distributions are near-identical across all 5 folds (e.g., `nv` test count is exactly 1341 in every fold; `mel` ranges only 222–223), confirming the `StratifiedGroupKFold` splitting is working as intended and definitively ruling out label-distribution imbalance as a contributor to Fold 0/Fold 1's earlier anomalies (consistent with the seed-based explanation already established in §4.4).

## 8c. New: "Why LoRA Works" — Hypothesis Added, Ablation Still Pending

A hypothesis was proposed for the discussion section: LoRA's advantage over naive top-layer unfreezing may stem from distributing the trainable budget across all 32 transformer blocks (adapting hierarchical features at every depth) versus Naive's budget being confined entirely to the final block (terminal semantics only). This is a reasonable, well-motivated hypothesis but **remains untested** — the proposed validating ablation (naive unfreeze of the last 2–3 blocks, budget-matched to LoRA) has not yet been run. Correctly framed as a hypothesis, not an established finding, in both current drafts.

## 8d. Parameter-Denominator Consistency — RESOLVED

The paper's Results table now includes an explicit footnote clarifying that LoRA's total-parameter denominator (~641.9M) exceeds Full-FT/Naive's (~632.0M) by ~9.9M due to LoRA adapter parameters — verified arithmetically correct (641,937,349 − 632,042,503 = 9,894,846). This closes the transparency requirement flagged in the original roadmap's §3.

**Two small remaining nits, not yet fixed:**
- `F_frozen`'s budget is listed as 0.41% in the Results table; precise computation (2,626,567 / 632,042,503) gives 0.4156%, which rounds to **0.42%**.
- The progress log's "Final Conclusion" paragraph still describes LoRA as *"mathematically superior"* — stronger language than the evidence supports (p=0.0625 doesn't cross conventional significance; one fold was a near-tie). Should be softened to something like *"empirically favorable... on the majority of evaluated folds"* for consistency with the hedged framing used elsewhere in the same draft.

## 9. Outstanding Action Items

**Required before submission:**
1. **[REQUIRED] Reframe statistical language in the discussion section** — avoid "definitive"/"proves" language around p=0.0625 results; state the N=5 sample-size ceiling explicitly (§5.3). Low effort, directly affects credibility with a reviewer who knows nonparametric tests.
2. **[REQUIRED] Confirm current AAAI workshop CFP and scope** (e.g., W3PHIAI or equivalent health-AI workshop) before final submission — not verified with up-to-date search in this conversation. Can't submit without knowing the actual venue/deadline/scope.
3. **[REQUIRED] Fix the two small nits from §8d:** round `F_frozen`'s budget to 0.42% (not 0.41%) in the Results table; soften "mathematically superior" in the progress log's Final Conclusion to language consistent with the hedged framing used elsewhere (e.g., "empirically favorable... on the majority of evaluated folds"). Trivial effort, fixes a factual rounding error and an overclaim.

**Optional — strengthen the paper but do not block submission:**
4. **[OPTIONAL] Recompute macro precision/recall from multi-seed-aggregated `F_lora` predictions** — only matters if the final paper keeps the precision/recall-skew discussion and confusion-matrix figure (§4.7). If that section is cut, this is moot.
5. **[OPTIONAL] Run a second independent CV partition** to obtain N=10 fold comparisons for a statistically stronger significance test, with the non-independence caveat documented (§5.3). Was already flagged as optional when first proposed — the current p=0.0625 framing (§9 item 1 above) is defensible on its own without this.
6. **[OPTIONAL] Run the "why LoRA works" validating ablation** (§8c) — naive unfreeze of the last 2–3 blocks, budget-matched to LoRA. The hypothesis can be presented as-is, clearly labeled speculative/future work, without running it.
7. **[OPTIONAL] Attention Rollout failure-analysis figure** — a nice supplementary visual (frozen vs. LoRA-adapted attention maps on correct/incorrect examples), not required to support the core gap-closing or structural-advantage claims, which stand on the quantitative results alone.

**Resolved, no longer open:** Fold 0 selective-resampling concern (§8a), per-fold class-distribution check (§8b), parameter-denominator transparency (§8d, footnote added).

---

## 10. Paper Framing Recommendation (current, given §7 results)

Headline claim to lead with: *"Parameter-efficient adaptation (LoRA) of a frozen I-JEPA encoder closes ~99% of the performance gap to full fine-tuning (95% CI: 73–123%) while updating under 2% of parameters, and provides a measurable structural advantage over a parameter-matched naive top-layer unfreeze on the majority of evaluated folds."*

This framing is honest, internally consistent with the validated data in §7, and — with the Fold 0 caveat and p-value language handled per §5.3/§7.4 — is submittable to an AAAI workshop with the rigor already built into the evaluation protocol (lesion-grouped CV, matched heads, matched parameter budgets, bootstrap CI, symmetric multi-seed protocol, transparent handling of an initially-anomalous result). The methodology itself — not just the headline number — is a legitimate part of the paper's contribution.
