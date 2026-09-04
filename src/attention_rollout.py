import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

from src.models import load_ijepa_target_encoder, LoRAIJEPAModel
from src.cv_utils import get_folds

def get_attention_maps(model, img_tensor, is_lora=False):
    attn_weights = []
    
    def hook(module, input, output):
        # input to attn_drop is the attention matrix: [B, num_heads, num_queries, num_keys]
        attn_weights.append(input[0].detach().clone().cpu())
    
    # Disable fused attention to allow attn_drop hook to fire
    if is_lora:
        blk = model.encoder.base_model.model.blocks[-1].attn
    else:
        blk = model.blocks[-1].attn
    
    orig_fused = blk.fused_attn
    blk.fused_attn = False
    
    handle = blk.attn_drop.register_forward_hook(hook)
    
    with torch.no_grad():
        _ = model(img_tensor)
        
    handle.remove()
    blk.fused_attn = orig_fused
    
    # [1, 16, 256, 256]
    attn = attn_weights[0]
    
    # Average across heads: [1, 256, 256]
    attn = attn.mean(dim=1)
    
    # Average across queries (how much attention does each key/patch receive from all others)
    # [1, 256]
    attn = attn.mean(dim=1).squeeze(0)
    
    # Reshape to spatial dimensions (16x16)
    attn = attn.reshape(16, 16).numpy()
    
    # Normalize to 0-1 for visualization
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    
    return attn

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metadata = pd.read_csv("datasets/HAM10000_metadata.csv")
    classes = sorted(metadata['dx'].unique())
    class_mapping = {c: i for i, c in enumerate(classes)}
    
    folds = list(get_folds(metadata, random_state=42))
    # 1. Load Models
    frozen_encoder = load_ijepa_target_encoder("models/IN1K-vit.h.14-300e.pth.tar", device)
    lora_model = LoRAIJEPAModel("models/IN1K-vit.h.14-300e.pth.tar", rank=16).to(device)
    
    ckpt = torch.load("logs/lora/run_20260904_170306/checkpoints/fold_1_best.pth", map_location=device, weights_only=True)
    lora_model.load_state_dict(ckpt)
    lora_model.eval()
    
    # 2. Get some test images
    metadata = pd.read_csv("datasets/HAM10000_metadata.csv")
    folds = list(get_folds(metadata, random_state=42))
    _, test_idx = folds[0]
    test_df = metadata.iloc[test_idx].reset_index(drop=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    correct_idx = None
    incorrect_idx = None
    
    # Find one correct and one incorrect
    for idx in range(len(test_df)):
        img_id = test_df.loc[idx, 'image_id']
        label_str = test_df.loc[idx, 'dx']
        label_idx = class_mapping[label_str]
        
        img_path = f"datasets/HAM10000_preprocessed/{img_id}.jpg"
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = lora_model(tensor)
            pred = out.argmax(dim=1).item()
            
        if pred == label_idx and correct_idx is None:
            correct_idx = idx
        elif pred != label_idx and incorrect_idx is None:
            incorrect_idx = idx
            
        if correct_idx is not None and incorrect_idx is not None:
            break
            
    # 3. Process and Plot
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Attention Rollout: Frozen Backbone vs. LoRA Adaptation", fontsize=16)
    for row, target_idx in enumerate([correct_idx, incorrect_idx]):
        img_id = test_df.loc[target_idx, 'image_id']
        label_str = test_df.loc[target_idx, 'dx']
        img_path = f"datasets/HAM10000_preprocessed/{img_id}.jpg"
        
        orig_img = Image.open(img_path).convert('RGB')
        img_tensor = transform(orig_img).unsqueeze(0).to(device)
        
        # Get predictions
        with torch.no_grad():
            pred_idx = lora_model(img_tensor).argmax(dim=1).item()
            pred_str = list(class_mapping.keys())[list(class_mapping.values()).index(pred_idx)]
        
        # Get attention
        frozen_attn = get_attention_maps(frozen_encoder, img_tensor, is_lora=False)
        lora_attn = get_attention_maps(lora_model, img_tensor, is_lora=True)
        
        # Upsample attention
        frozen_attn = np.array(Image.fromarray(frozen_attn).resize((224, 224), resample=Image.BILINEAR))
        lora_attn = np.array(Image.fromarray(lora_attn).resize((224, 224), resample=Image.BILINEAR))
        
        # Plot Original
        axes[row, 0].imshow(orig_img.resize((224, 224)))
        axes[row, 0].set_title(f"True: {label_str} | Pred: {pred_str}\n({'Correct' if row == 0 else 'Incorrect'})")
        axes[row, 0].axis('off')
        
        # Plot Frozen
        axes[row, 1].imshow(orig_img.resize((224, 224)))
        axes[row, 1].imshow(frozen_attn, cmap='jet', alpha=0.5)
        axes[row, 1].set_title("Frozen I-JEPA Attention")
        axes[row, 1].axis('off')
        
        # Plot LoRA
        axes[row, 2].imshow(orig_img.resize((224, 224)))
        axes[row, 2].imshow(lora_attn, cmap='jet', alpha=0.5)
        axes[row, 2].set_title("LoRA-Adapted Attention")
        axes[row, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig("logs/attention_rollout.png", dpi=150, bbox_inches='tight')
    print("Saved figure to logs/attention_rollout.png")

if __name__ == "__main__":
    main()