import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import argparse
from pathlib import Path
from cv_utils import make_class_weights, get_folds
from models import LoRAIJEPAModel, count_trainable_params
from logger import RunLogger
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

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

def train_and_evaluate_fold(model, train_df, test_df, images_dir, num_classes, 
                            transform_train, transform_test,
                            epochs=10, micro_batch_size=4, accumulation_steps=8, device="cuda",
                            logger=None, fold=0, lr=5e-4):
    
    # StratifiedShuffleSplit for intra-fold validation (90% train, 10% val)
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        sub_train_idx, val_idx = next(splitter.split(train_df, train_df['dx']))
    except ValueError:
        # Fallback to random split if stratification fails (e.g. on very small mock data)
        sub_train_idx, val_idx = train_test_split(np.arange(len(train_df)), test_size=0.1, random_state=42)
    
    sub_train_df = train_df.iloc[sub_train_idx].copy()
    val_df = train_df.iloc[val_idx].copy()
    
    sub_train_dataset = HAM10000Dataset(sub_train_df, images_dir, transform=transform_train)
    val_dataset = HAM10000Dataset(val_df, images_dir, transform=transform_test)
    test_dataset = HAM10000Dataset(test_df, images_dir, transform=transform_test)
    
    sub_train_loader = DataLoader(sub_train_dataset, batch_size=micro_batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=micro_batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=micro_batch_size, shuffle=False, num_workers=2)
    
    train_labels = sub_train_df['dx'].map(sub_train_dataset.class_to_idx).values
    weights = make_class_weights(train_labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_f1 = -1.0
    ckpt_dir = os.path.join(logger.log_dir, "checkpoints") if logger else "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_model_path = os.path.join(ckpt_dir, f"fold_{fold+1}_best.pth")
    
    global_step = 0
    for epoch in range(epochs):
        if logger: logger.epoch_start()
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for i, (imgs, labels) in enumerate(tqdm(sub_train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            imgs, labels = imgs.to(device), labels.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(sub_train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            running_loss += loss.item() * accumulation_steps
            global_step += 1
            if logger and global_step % accumulation_steps == 0:
                logger.log_step("train", {"loss": loss.item() * accumulation_steps, "lr": scheduler.get_last_lr()[0]}, global_step // accumulation_steps)
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                imgs = imgs.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.numpy())
                
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        epoch_loss = running_loss/len(sub_train_loader)
        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Val F1: {val_f1:.4f}")
        if logger:
            logger.log_epoch(fold, epoch, {"train_loss": epoch_loss, "val_f1": val_f1, "lr": scheduler.get_last_lr()[0]})
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print("  --> Saved new best model")
            
    # Load best model for testing
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(labels.numpy())
            
    f1 = f1_score(test_targets, test_preds, average="macro", zero_division=0)
    prec = precision_score(test_targets, test_preds, average="macro", zero_division=0)
    rec = recall_score(test_targets, test_preds, average="macro", zero_division=0)
    cm = confusion_matrix(test_targets, test_preds, labels=np.arange(num_classes))
    
    return f1, prec, rec, cm

def main(images_dir, metadata_csv, ckpt_path, epochs, micro_batch_size, accumulation_steps, rank, lr):
    logger = RunLogger(paradigm="lora")
    logger.log_hparams({"epochs": epochs, "micro_batch_size": micro_batch_size, "rank": rank, "lr": lr})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    df = pd.read_csv(metadata_csv)
    num_classes = len(df['dx'].unique())
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    num_folds = min(5, len(df['lesion_id'].unique()) // 3)
    if num_folds < 2: num_folds = 2
        
    f1s, precs, recs, cms = [], [], [], []
    
    for fold, (train_idx, test_idx) in enumerate(get_folds(df, num_folds=num_folds, random_state=42)):
        print(f"\n--- Fold {fold+1}/{num_folds} ---")
        
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        # Instantiate fresh model for each fold
        model = LoRAIJEPAModel(ckpt_path=ckpt_path, rank=rank, num_classes=num_classes).to(device)
        
        if fold == 0:
            trainable, total, pct = count_trainable_params(model)
            print(f"\n[BUDGET] Trainable Params: {trainable:,} / {total:,} ({pct:.2f}%)")
            if pct >= 5.0:
                print("[WARNING] Parameter budget exceeds 5%!")
        
        f1, prec, rec, cm = train_and_evaluate_fold(
            model, train_df, test_df, images_dir, num_classes,
            transform_train, transform_test,
            epochs=epochs, micro_batch_size=micro_batch_size, 
            accumulation_steps=accumulation_steps, device=device,
            logger=logger, fold=fold, lr=lr
        )
        
        f1s.append(f1)
        if logger: logger.log_fold_result(fold, f1)
        precs.append(prec)
        recs.append(rec)
        cms.append(cm)
        
    print("\n=== Final Results ===")
    print(f"Macro F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Macro Recall: {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    
    if logger:
        logger.finish({
            "macro_f1_mean": np.mean(f1s),
            "macro_f1_std": np.std(f1s)
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()
    
    try:
        main(args.images_dir, args.metadata_csv, args.ckpt_path, 
             args.epochs, args.micro_batch_size, args.accumulation_steps,
             args.rank, args.lr)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n[ERROR] CUDA Out of Memory!")
            print("Try running with smaller micro_batch_size (e.g. 2) and higher accumulation_steps (e.g. 16).")
        else:
            raise e