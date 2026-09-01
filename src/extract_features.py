import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import timm
from timm.models.vision_transformer import VisionTransformer
from PIL import Image
from torchvision import transforms
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class HAM10000Dataset(Dataset):
    def __init__(self, metadata_df, images_dir, transform=None):
        self.metadata_df = metadata_df
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        # Mapping labels to integers
        self.classes = sorted(self.metadata_df['dx'].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.metadata_df)
        
    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        img_id = row['image_id']
        lesion_id = row['lesion_id']
        label = self.class_to_idx[row['dx']]
        
        img_path = self.images_dir / f"{img_id}.jpg"
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, lesion_id

from models import load_ijepa_target_encoder
def load_dino_model(device="cuda"):
    model = timm.create_model('vit_base_patch14_dinov2', pretrained=True, num_classes=0, dynamic_img_size=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval().to(device)
    return model

@torch.no_grad()
def extract_and_cache_features(model, dataloader, device="cuda"):
    all_feats, all_labels, all_ids = [], [], []
    for imgs, labels, lesion_ids in tqdm(dataloader, desc="Extracting"):
        imgs = imgs.to(device)
        
        # Forward pass depending on the model's output
        # For timm models with num_classes=0, they might output pooled features or tokens
        # timm VisionTransformer with global_pool="" outputs tokens (B, N, D)
        # timm dinov2 with num_classes=0 outputs pooled features (B, D) depending on global_pool
        # Let's check output shape dynamically
        
        tokens = model(imgs) 
        
        if tokens.dim() == 3:
            # (B, N, D) -> pool over patch tokens
            pooled = tokens.mean(dim=1)
        else:
            # (B, D)
            pooled = tokens
            
        all_feats.append(pooled.cpu())
        all_labels.append(labels)
        all_ids.extend(lesion_ids)
        
    return torch.cat(all_feats).numpy(), torch.cat(all_labels).numpy(), np.array(all_ids)

def main(images_dir: str, metadata_csv: str, output_dir: str, ckpt_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(metadata_csv)
    
    # Standard ImageNet normalization since I-JEPA uses it
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = HAM10000Dataset(df, images_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Extract I-JEPA features
    print("Loading I-JEPA model...")
    ijepa_model = load_ijepa_target_encoder(ckpt_path, device)
    print("Extracting I-JEPA features...")
    ijepa_feats, labels, lesion_ids = extract_and_cache_features(ijepa_model, dataloader, device)
    
    # Extract DINO features
    print("Loading DINO model...")
    dino_model = load_dino_model(device)
    print("Extracting DINO features...")
    dino_feats, _, _ = extract_and_cache_features(dino_model, dataloader, device)
    
    # Save features
    np.save(out_dir / "ijepa_features.npy", ijepa_feats)
    np.save(out_dir / "dino_features.npy", dino_feats)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "lesion_ids.npy", lesion_ids)
    
    print(f"Saved cached features to {output_dir}")
    print(f"I-JEPA shape: {ijepa_feats.shape}, DINO shape: {dino_feats.shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing preprocessed images")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Path to metadata CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save cached features")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to I-JEPA checkpoint")
    args = parser.parse_args()
    
    main(args.images_dir, args.metadata_csv, args.output_dir, args.ckpt_path)
