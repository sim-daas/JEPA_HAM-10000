import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import timm
import argparse
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from pathlib import Path
from cv_utils import make_class_weights, get_folds
from logger import RunLogger

class HAM10000Dataset(Dataset):
    def __init__(self, metadata_df, images_dir, transform=None):
        self.metadata_df = metadata_df
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        # Sort classes for consistent indexing
        self.classes = sorted(self.metadata_df['dx'].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.metadata_df)
        
    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        img_id = row['image_id']
        label = self.class_to_idx[row['dx']]
        lesion_id = row['lesion_id']
        
        img_path = self.images_dir / f"{img_id}.jpg"
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, lesion_id

def train_and_evaluate_fold(model, train_loader, test_loader, num_classes, train_labels, epochs=5, device="cuda", logger=None, fold=0):
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # Lower LR to prevent catastrophic forgetting
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Train
    for epoch in range(epochs):
        model.train()
        
        # Freeze BatchNorm to prevent instability from small batch size
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            try:
                out = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("\nWARNING: Out of Memory error during forward/backward pass.")
                    print("Try reducing the batch_size in train_supervised.py.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise e
                else:
                    raise e
                    
        scheduler.step()
        
    # Save fold checkpoint
    if logger:
        ckpt_dir = os.path.join(logger.log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"fold_{fold+1}_best.pth"))
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            preds = out.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            
    f1 = f1_score(all_targets, all_preds, average="macro")
    prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=np.arange(num_classes))
    
    return f1, prec, rec, cm

def main(images_dir: str, metadata_csv: str, model_name: str):
    logger = RunLogger(paradigm=f"supervised_{model_name}")
    logger.log_hparams({"model_name": model_name})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(metadata_csv)
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    num_classes = len(df['dx'].unique())
    
    # Initialize model once to avoid repeated HF hub hits
    print(f"Initializing {model_name}...")
    model_template = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    torch.save(model_template.state_dict(), "initial_weights.pth")
    
    num_folds = min(5, len(df['lesion_id'].unique()) // 3)
    if num_folds < 2:
        num_folds = 2
        
    f1s, precs, recs, cms = [], [], [], []
    
    for fold, (train_idx, test_idx) in enumerate(get_folds(df, num_folds=num_folds, random_state=42)):
        print(f"--- Fold {fold+1}/{num_folds} ---")
        
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        train_dataset = HAM10000Dataset(train_df, images_dir, transform=transform_train)
        test_dataset = HAM10000Dataset(test_df, images_dir, transform=transform_test)
        
        # Batch size of 8 should fit in 16GB VRAM for ResNet50. May need adjusting for larger ViTs.
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
        
        # Re-initialize model per fold from local cache
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes).to(device)
        # Handle weights_only explicitly to avoid warnings
        model.load_state_dict(torch.load("initial_weights.pth", weights_only=True))
        
        # map back to int for the function
        classes = sorted(df['dx'].unique())
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_train_int = [class_to_idx[l] for l in train_df['dx'].values]
        
        f1, prec, rec, cm = train_and_evaluate_fold(
            model, train_loader, test_loader, num_classes, y_train_int, epochs=2, device=device, logger=logger, fold=fold
        )
        
        f1s.append(f1)
        if logger: logger.log_fold_result(fold, f1)
        precs.append(prec)
        recs.append(rec)
        cms.append(cm)
    print(f"\n--- Evaluating Supervised {model_name} Baseline ---")
    print(f"Macro F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Macro Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"Macro Recall: {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    
    if logger:
        logger.finish({
            "macro_f1_mean": np.mean(f1s),
            "macro_f1_std": np.std(f1s)
        })
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing preprocessed images")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Path to metadata CSV")
    parser.add_argument("--model_name", type=str, default="resnet50", help="Timm model name for baseline")
    args = parser.parse_args()
    
    main(args.images_dir, args.metadata_csv, args.model_name)
