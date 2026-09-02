import os
import glob
import json
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def get_latest_metrics(log_dir_prefix):
    """Finds the most recent metrics.json for a given run paradigm."""
    dirs = glob.glob(f"logs/{log_dir_prefix}/*")
    if not dirs:
        raise ValueError(f"No logs found for {log_dir_prefix}")
    latest_dir = max(dirs, key=os.path.getmtime)
    metrics_path = os.path.join(latest_dir, "metrics.json")
    with open(metrics_path, "r") as f:
        return json.load(f)

def bootstrap_gap_closed(f_frozen, f_full, f_lora, num_samples=10000, seed=42):
    """
    Bootstrap the Gap Closed metric.
    Gap Closed = (F_lora - F_frozen) / (F_full - F_frozen)
    """
    np.random.seed(seed)
    n = len(f_frozen)
    bootstrapped_gaps = []
    
    for _ in range(num_samples):
        # Sample indices with replacement
        indices = np.random.choice(n, size=n, replace=True)
        
        # Calculate mean for this bootstrap sample
        mean_frozen = np.mean([f_frozen[i] for i in indices])
        mean_full = np.mean([f_full[i] for i in indices])
        mean_lora = np.mean([f_lora[i] for i in indices])
        
        # Calculate gap closed
        if mean_full - mean_frozen != 0:
            gap = (mean_lora - mean_frozen) / (mean_full - mean_frozen)
            bootstrapped_gaps.append(gap * 100.0) # percentage
            
    # Calculate 95% CI
    ci_lower = np.percentile(bootstrapped_gaps, 2.5)
    ci_upper = np.percentile(bootstrapped_gaps, 97.5)
    mean_gap = np.mean(bootstrapped_gaps)
    
    return mean_gap, ci_lower, ci_upper

def main():
    print("--- Statistical Analysis ---")
    
    # We use the hardcoded lora array provided initially:
    # f_lora = np.array([0.5944, 0.7434, 0.7396, 0.6630, 0.7332])
    # Let's load them dynamically to be rigorous.
    
    try:
        frozen_metrics = get_latest_metrics("frozen_probe_ijepa")
        full_metrics = get_latest_metrics("full_finetune")
        lora_metrics = get_latest_metrics("lora")
        naive_metrics = get_latest_metrics("naive")
        
        f_frozen = np.array([frozen_metrics["folds"][f"fold_{i}"] for i in range(5)])
        f_full = np.array([full_metrics["folds"][f"fold_{i}"] for i in range(5)])
        f_lora = np.array([lora_metrics["folds"][f"fold_{i}"] for i in range(5)])
        f_naive = np.array([naive_metrics["folds"][f"fold_{i}"] for i in range(5)])
        
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return
        
    print(f"F_frozen array: {f_frozen.round(4)}")
    print(f"F_full array:   {f_full.round(4)}")
    print(f"F_lora array:   {f_lora.round(4)}")
    print(f"F_naive array:  {f_naive.round(4)}")
    
    print("\n--- Wilcoxon Paired Tests ---")
    # F_lora vs F_frozen (Did LoRA improve over frozen?)
    stat, p_frozen = stats.wilcoxon(f_lora, f_frozen, alternative='greater')
    print(f"LoRA > Frozen: p = {p_frozen:.4f}")
    
    # F_lora vs F_full (Is LoRA statistically worse than Full?)
    stat, p_full = stats.wilcoxon(f_full, f_lora, alternative='greater')
    print(f"Full > LoRA:   p = {p_full:.4f}")
    
    # F_lora vs F_naive (Is LoRA structurally better than just unfreezing the same param count?)
    stat, p_naive = stats.wilcoxon(f_lora, f_naive, alternative='two-sided')
    print(f"LoRA vs Naive: p = {p_naive:.4f}")
    
    print("\n--- Bootstrap CI for Gap Closed ---")
    mean_gap, ci_lower, ci_upper = bootstrap_gap_closed(f_frozen, f_full, f_lora)
    print(f"Gap Closed: {mean_gap:.1f}% [95% CI: {ci_lower:.1f}% - {ci_upper:.1f}%]")
    

if __name__ == "__main__":
    main()
