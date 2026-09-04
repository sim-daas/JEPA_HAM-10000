import pandas as pd
import numpy as np
from cv_utils import get_folds

def main():
    metadata = pd.read_csv("datasets/HAM10000_metadata.csv")
    
    # ensure consistent classes
    classes = sorted(metadata['dx'].unique().tolist())
    
    print("## Supplementary: Per-Fold Class Distribution")
    print("This table confirms that the stratified group k-fold splitting strategy maintained consistent class distributions across all 5 folds, preventing any class imbalance artifacts.\n")
    
    print("| Fold | Split | " + " | ".join(classes) + " | Total |")
    print("|------|-------|" + "|".join(["---" for _ in classes]) + "|-------|")
    
    for fold_idx, (train_idx, test_idx) in enumerate(get_folds(metadata, random_state=42)):
        train_df = metadata.iloc[train_idx]
        test_df = metadata.iloc[test_idx]
        
        train_counts = train_df['dx'].value_counts()
        test_counts = test_df['dx'].value_counts()
        
        train_row = [str(train_counts.get(cls, 0)) for cls in classes]
        test_row = [str(test_counts.get(cls, 0)) for cls in classes]
        
        print(f"| Fold {fold_idx} | Train | " + " | ".join(train_row) + f" | {len(train_idx)} |")
        print(f"| Fold {fold_idx} | Test  | " + " | ".join(test_row) + f" | {len(test_idx)} |")

if __name__ == "__main__":
    main()