import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import argparse
from pathlib import Path
import pandas as pd
from cv_utils import make_class_weights, get_folds

from models import ProbeHead
def train_and_evaluate_fold(X_train, y_train, X_test, y_test, num_classes, epochs=20, batch_size=32, device="cuda", num_layers=2):
    model = ProbeHead(in_dim=X_train.shape[1], num_classes=num_classes, num_layers=num_layers).to(device)
    
    # Compute weights from training fold ONLY
    weights = make_class_weights(y_train, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test).float().to(device)
        preds = model(X_test_tensor).argmax(dim=1).cpu().numpy()
        
    f1 = f1_score(y_test, preds, average="macro")
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=np.arange(num_classes))
    
    return f1, prec, rec, cm

def evaluate_features(features_path, labels_path, metadata_csv, num_classes=7, num_folds=5, epochs=20, num_layers=2):
    X = np.load(features_path)
    y = np.load(labels_path)
    df = pd.read_csv(metadata_csv)
    
    f1s, precs, recs = [], [], []
    cms = []
    
    for fold, (train_idx, test_idx) in enumerate(get_folds(df, num_folds=num_folds, random_state=42)):
        f1, prec, rec, cm = train_and_evaluate_fold(
            X[train_idx], y[train_idx], 
            X[test_idx], y[test_idx], 
            num_classes, epochs=epochs, num_layers=num_layers
        )
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)
        cms.append(cm)
        
    return {
        "macro_f1": (np.mean(f1s), np.std(f1s)),
        "macro_prec": (np.mean(precs), np.std(precs)),
        "macro_rec": (np.mean(recs), np.std(recs)),
        "cm": np.sum(cms, axis=0)
    }

def main(features_dir: str, num_layers: int):
    feat_dir = Path(features_dir)
    labels_path = feat_dir / "labels.npy"
    groups_path = feat_dir / "lesion_ids.npy"
    metadata_csv = feat_dir.parent / "HAM10000_preprocessed" / "metadata.csv"
    # Since mock data might not have enough samples for 5 folds across all classes,
    # we dynamically determine fold count based on dataset size for robustness.
    # At minimum 2 folds for testing
    num_folds = min(5, len(set(np.load(groups_path))) // 3)
    if num_folds < 2: num_folds = 2
    
    print(f"\n--- Evaluating I-JEPA (Probe Layers: {num_layers}) ---")
    ijepa_results = evaluate_features(feat_dir / "ijepa_features.npy", labels_path, str(metadata_csv), num_folds=num_folds, num_layers=num_layers)
    print(f"Macro F1: {ijepa_results['macro_f1'][0]:.4f} ± {ijepa_results['macro_f1'][1]:.4f}")
    print(f"Macro Precision: {ijepa_results['macro_prec'][0]:.4f} ± {ijepa_results['macro_prec'][1]:.4f}")
    print(f"Macro Recall: {ijepa_results['macro_rec'][0]:.4f} ± {ijepa_results['macro_rec'][1]:.4f}")
    
    print(f"\n--- Evaluating DINO (Probe Layers: {num_layers}) ---")
    dino_results = evaluate_features(feat_dir / "dino_features.npy", labels_path, str(metadata_csv), num_folds=num_folds, num_layers=num_layers)
    print(f"Macro F1: {dino_results['macro_f1'][0]:.4f} ± {dino_results['macro_f1'][1]:.4f}")
    print(f"Macro Precision: {dino_results['macro_prec'][0]:.4f} ± {dino_results['macro_prec'][1]:.4f}")
    print(f"Macro Recall: {dino_results['macro_rec'][0]:.4f} ± {dino_results['macro_rec'][1]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, required=True, help="Directory containing cached features")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the MLP probe")
    args = parser.parse_args()
    
    main(args.features_dir, args.num_layers)
