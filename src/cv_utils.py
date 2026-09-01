import torch
from sklearn.model_selection import StratifiedGroupKFold

def make_class_weights(labels, num_classes=7):
    """
    Computes class weights inversely proportional to their frequencies.
    Handles missing classes by clamping the count to 1 to avoid division by zero.
    """
    labels_tensor = torch.tensor(labels)
    counts = torch.bincount(labels_tensor, minlength=num_classes).float()
    n_total = counts.sum()
    weights = n_total / (num_classes * counts.clamp(min=1))
    return weights

def get_folds(df, num_folds=5, random_state=42):
    """
    Returns an iterator of (train_idx, test_idx) using StratifiedGroupKFold.
    Ensures identical splits across all scripts.
    """
    sgkf = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    return sgkf.split(df, df['dx'], df['lesion_id'])
