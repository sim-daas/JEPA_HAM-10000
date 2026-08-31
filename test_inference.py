import sys
import torch

sys.path.append('test/ijepa')
from src.models.vision_transformer import vit_huge

def main():
    print("Initializing ViT-Huge (patch_size=14)...")
    model = vit_huge(patch_size=14)
    
    ckpt_path = 'models/IN1K-vit.h.14-300e.pth.tar'
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Extract encoder weights and remove 'module.' prefix
    state_dict = ckpt['target_encoder']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    
    print("Running forward pass with dummy tensor...")
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"Success! Output shape: {output.shape}")

if __name__ == '__main__':
    main()
