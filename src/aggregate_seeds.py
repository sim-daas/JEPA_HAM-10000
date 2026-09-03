import os
import json
import glob
from collections import defaultdict

def aggregate_logs():
    results = defaultdict(lambda: defaultdict(dict))
    
    # metrics structure inside json is like:
    # "fold_0_f1": 0.594, etc.
    # We will also parse hparams.seed if present, defaulting to 42 for legacy logs
    
    log_files = glob.glob("logs/*/*/metrics.json")
    for f in log_files:
        paradigm = f.split(os.sep)[1]  # logs/lora/... -> lora
        if paradigm not in ["lora", "naive"]:
            continue
            
        with open(f, 'r') as file:
            try:
                data = json.load(file)
            except:
                continue
                
        hparams = data.get("hparams", {})
        seed = hparams.get("seed", 42)
        
        folds_data = data.get("folds", {})
        for fold in range(5):
            fold_key = f"fold_{fold}"
            if fold_key in folds_data:
                results[paradigm][seed][fold] = folds_data[fold_key]

    with open("logs/robustness_summary.txt", "w") as out:
        out.write("Robustness & Repeated K-Fold Summary\n")
        out.write("=====================================\n\n")
        
        for paradigm in ["lora", "naive"]:
            out.write(f"Paradigm: {paradigm.upper()}\n")
            out.write("-" * 30 + "\n")
            seeds = sorted(results[paradigm].keys())
            
            # Print table header
            out.write(f"{'Seed':<6}")
            for fold in range(5):
                out.write(f" | Fold {fold}")
            out.write(" | Mean F1\n")
            out.write("-" * 55 + "\n")
            
            for seed in seeds:
                row_str = f"{seed:<6}"
                fold_f1s = []
                for fold in range(5):
                    f1 = results[paradigm][seed].get(fold, None)
                    if f1 is not None:
                        row_str += f" | {f1:.4f}"
                        fold_f1s.append(f1)
                    else:
                        row_str += f" | {'N/A':<6}"
                
                mean_f1 = sum(fold_f1s) / len(fold_f1s) if fold_f1s else 0
                row_str += f" | {mean_f1:.4f}\n"
                out.write(row_str)
            out.write("\n")
            
if __name__ == "__main__":
    aggregate_logs()
    print("Aggregated results saved to logs/robustness_summary.txt")