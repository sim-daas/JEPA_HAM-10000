import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import json
import glob
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import argparse
from pathlib import Path

from cv_utils import get_folds
from models import LoRAIJEPAModel

class HAM10000Dataset(Dataset):
    def __init__(self, metadata_df, images_dir, transform=None):
        self.metadata_df = metadata_df
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        self.classes = sorted(self.metadata_df['dx'].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.metadata_df)
        
    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        img_id = row['image_id']
        label = self.class_to_idx[row['dx']]
        
        img_path = self.images_dir / f"{img_id}.jpg"
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def find_target_runs():
    """Finds the most recent successful run directories for seeds 42, 100, 2026."""
    target_seeds = {42, 100, 2026}
    selected_runs = {}
    
    log_files = glob.glob("logs/lora/*/metrics.json")
    # Sort by modification time to get the latest runs
    log_files.sort(key=os.path.getmtime, reverse=True)
    
    for f in log_files:
        with open(f, 'r') as file:
            try:
                data = json.load(file)
            except:
                continue
                
        hparams = data.get("hparams", {})
        seed = hparams.get("seed", 42)
        
        # Only select full runs (epochs=10, only_fold=-1)
        if hparams.get("epochs") == 10 and hparams.get("only_fold", -1) == -1:
            if seed in target_seeds and seed not in selected_runs:
                selected_runs[seed] = os.path.dirname(f)
                
    return selected_runs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default="datasets/HAM10000_preprocessed")
    parser.add_argument("--metadata_csv", type=str, default="datasets/HAM10000_metadata.csv")
    parser.add_argument("--ckpt_path", type=str, default="models/IN1K-vit.h.14-300e.pth.tar")
    parser.add_argument("--dry_run", action="store_true", help="Process only 1 batch per fold for testing")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    df = pd.read_csv(args.metadata_csv)
    num_classes = len(df['dx'].unique())
    classes = sorted(df['dx'].unique())
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    num_folds = min(5, len(df['lesion_id'].unique()) // 3)
    if num_folds < 2: num_folds = 2

    # Verify runs
    target_runs = find_target_runs()
    print("Found runs:")
    for seed, path in target_runs.items():
        print(f"Seed {seed}: {path}")
        
    if len(target_runs) < 3:
        print(f"Error: Expected 3 runs, found {len(target_runs)}. Missing seeds: {set([42, 100, 2026]) - set(target_runs.keys())}")
        return

    all_targets = []
    all_preds = []

    # Get folds exactly as in training
    folds = list(get_folds(df, num_folds=num_folds, random_state=42))

    # Initialize model once
    model = LoRAIJEPAModel(ckpt_path=args.ckpt_path, rank=16, num_classes=num_classes).to(device)

    for seed in [42, 100, 2026]:
        run_dir = target_runs[seed]
        print(f"\nProcessing Seed {seed} from {run_dir}")
        
        for fold, (train_idx, test_idx) in enumerate(folds):
            print(f"  Fold {fold+1}/{num_folds}...")
            
            checkpoint_path = os.path.join(run_dir, "checkpoints", f"fold_{fold+1}_best.pth")
            if not os.path.exists(checkpoint_path):
                print(f"  Error: Missing checkpoint at {checkpoint_path}")
                continue
                
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()
            
            test_df = df.iloc[test_idx]
            test_dataset = HAM10000Dataset(test_df, args.images_dir, transform=transform_test)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    for i, (imgs, labels) in enumerate(tqdm(test_loader, desc=f"Seed {seed} Fold {fold+1}", leave=False)):
                        imgs = imgs.to(device)
                        outputs = model(imgs)
                        preds = outputs.argmax(dim=1).cpu().numpy()
                        
                        all_preds.extend(preds)
                        all_targets.extend(labels.numpy())
                        
                        if args.dry_run and i >= 1: # process 2 batches for dry run
                            break

    # Calculate global metrics
    print("\n" + "="*40)
    print("GLOBAL AGGREGATED METRICS (3 Seeds x 5 Folds)")
    print("="*40)
    
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=np.arange(num_classes))
    
    print(f"Total Predictions Pooled: {len(all_preds)}")
    print(f"Macro Precision: {prec:.4f}")
    print(f"Macro Recall:    {rec:.4f}")
    print(f"Macro F1:        {f1:.4f}")
    
    print("\nConfusion Matrix:")
    # Print nice CM
    header = f"{'True \ Pred':<15}" + "".join([f"{c[:4]:<6}" for c in classes])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = "".join([f"{val:<6}" for val in row])
        print(f"{classes[i]:<15}{row_str}")

if __name__ == "__main__":
    main()
